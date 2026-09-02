import { useState, useEffect, useRef, useCallback } from "react";
import AgoraRTC from "agora-rtc-sdk-ng";
import { AIDenoiserExtension } from "agora-extension-ai-denoiser";

// The denoiser's WASM is loaded from the CDN (kept out of the bundle). If the
// browser blocks it, we fall back to the raw mic — see makeMicTrack().
const DENOISER_ASSETS =
  "https://cdn.jsdelivr.net/npm/agora-extension-ai-denoiser@2.0.2/external";
const AVATAR_COLORS = ["#4f8cff", "#a855f7", "#ef4444", "#14b8a6", "#f59e0b", "#64748b"];

const initials = (name) => (name || "?").slice(0, 1).toUpperCase();

export default function App() {
  const [agents, setAgents] = useState([]); // [{id,name,title}]
  const [joined, setJoined] = useState(false);
  const [status, setStatus] = useState("Idle.");
  const [speaking, setSpeaking] = useState(null); // agent id
  const [thinking, setThinking] = useState(false);
  const [agentCap, setAgentCap] = useState(null); // {name,title,text}
  const [candCap, setCandCap] = useState(null); // text
  const [coverage, setCoverage] = useState([]); // [{key,name,value}]
  const [claims, setClaims] = useState([]); // [{text,competency,strength,status,turn,contradicts}]
  const [contradictions, setContradictions] = useState(0);
  const [showDebug, setShowDebug] = useState(false);
  const [logLines, setLogLines] = useState([]);
  const logRef = useRef(null);

  const client = useRef(null);
  const mic = useRef(null);
  const ws = useRef(null);
  const denoiser = useRef(null);

  const log = useCallback(
    (m) => setLogLines((l) => [...l.slice(-200), `[${new Date().toLocaleTimeString()}] ${m}`]),
    []
  );
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [logLines]);

  // Load the roster up front so tiles render before joining.
  useEffect(() => {
    fetch("/panel")
      .then((r) => r.json())
      .then((d) => setAgents(d.agents || []))
      .catch(() => {});
  }, []);

  function openEvents() {
    const url = location.origin.replace(/^http/, "ws") + "/session/events";
    const sock = new WebSocket(url);
    ws.current = sock;
    sock.onmessage = (e) => {
      const ev = JSON.parse(e.data);
      if (ev.type === "panel") setAgents(ev.agents);
      else if (ev.type === "thinking") {
        setThinking(true);
        setSpeaking(null);
      } else if (ev.type === "speaking") {
        setThinking(false);
        setSpeaking(ev.agent);
        setAgentCap({ name: ev.name, title: ev.title, text: ev.text });
        log(`🗣️ ${ev.name} (${ev.title})`);
      } else if (ev.type === "idle") {
        setThinking(false);
        setSpeaking(null);
      } else if (ev.type === "heard") {
        setCandCap(ev.text);
      } else if (ev.type === "ledger") {
        setCoverage(ev.coverage || []);
        setClaims(ev.claims || []);
        setContradictions(ev.contradictions || 0);
      }
    };
    sock.onclose = () => log("events socket closed");
    sock.onerror = () => log("events socket error");
  }

  async function makeMicTrack() {
    const track = await AgoraRTC.createMicrophoneAudioTrack({ AEC: true, ANS: true, AGC: true });
    try {
      if (!denoiser.current) {
        denoiser.current = new AIDenoiserExtension({ assetsPath: DENOISER_ASSETS });
        denoiser.current.onloaderror = (e) => log("denoiser load error: " + e);
        AgoraRTC.registerExtensions([denoiser.current]);
      }
      const proc = denoiser.current.createProcessor();
      await proc.enable();
      // NSNG = neural noise suppression. The ESM build doesn't export the mode
      // enum, but the mode is just the string "NSNG".
      try {
        proc.setMode("NSNG");
      } catch (_) {}
      track.pipe(proc).pipe(track.processorDestination);
      log("🧹 AI noise suppression (NSNG) enabled");
    } catch (e) {
      log("AI denoiser unavailable, using raw mic (" + (e.message || e) + ")");
    }
    return track;
  }

  async function join() {
    setStatus("Starting session…");
    try {
      const resp = await fetch("/session/start", { method: "POST" });
      if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
      const s = await resp.json();
      log(`channel=${s.channel} uid=${s.uid}`);
      openEvents();

      client.current = AgoraRTC.createClient({ mode: "rtc", codec: "vp8" });
      client.current.on("user-published", async (user, mt) => {
        await client.current.subscribe(user, mt);
        if (mt === "audio") user.audioTrack.play();
      });
      setStatus("Joining channel…");
      await client.current.join(s.app_id, s.channel, s.token, s.uid);
      mic.current = await makeMicTrack();
      await client.current.publish([mic.current]);
      setJoined(true);
      setStatus("Live — the panel will greet you shortly.");
      log("🎤 mic published");
    } catch (e) {
      log("ERROR: " + (e.message || e));
      setStatus("Failed — see log.");
    }
  }

  async function leave() {
    try {
      if (ws.current) {
        ws.current.close();
        ws.current = null;
      }
      if (mic.current) {
        mic.current.stop();
        mic.current.close();
        mic.current = null;
      }
      if (client.current) {
        await client.current.leave();
        client.current = null;
      }
      await fetch("/session/stop", { method: "POST" });
    } catch (e) {
      log("leave error: " + (e.message || e));
    }
    setJoined(false);
    setSpeaking(null);
    setThinking(false);
    setStatus("Idle.");
    log("left channel");
  }

  async function interrupt() {
    try {
      const r = await fetch("/session/interrupt", { method: "POST" });
      const d = await r.json();
      log(d.interrupted ? "✋ interrupted — go ahead and speak" : "✋ (nobody is speaking)");
    } catch (e) {
      log("interrupt error: " + (e.message || e));
    }
  }

  return (
    <div className="wrap">
      <h1>968ms — AI Interview Panel</h1>
      <p className="sub">
        Five AI interviewers. Whoever's speaking lights up. Answer with your mic; use Interrupt to
        cut in.
      </p>

      <div className="controls">
        <button className="join" onClick={join} disabled={joined}>
          Join Interview
        </button>
        <button className="interrupt" onClick={interrupt} disabled={!joined}>
          ✋ Interrupt
        </button>
        <button onClick={leave} disabled={!joined}>
          Leave
        </button>
        <span className="status">{status}</span>
      </div>

      <div className="howto">
        🎤 <b>Your mic</b> is for <b>answering</b> — speak, then pause. It won't cut the interviewer
        off. &nbsp;•&nbsp; ✋ <b>Interrupt</b> is the only way to cut in while someone's talking.
        &nbsp;•&nbsp; 🧹 AI noise suppression cleans your mic automatically.
      </div>

      <div className="tiles">
        {agents.map((a, i) => (
          <div
            key={a.id}
            className={"tile" + (speaking === a.id ? " speaking" : thinking ? " thinking" : "")}
          >
            <span className="badge">SPEAKING</span>
            <div className="avatar" style={{ background: AVATAR_COLORS[i % AVATAR_COLORS.length] }}>
              {initials(a.name)}
            </div>
            <div className="name">{a.name}</div>
            <div className="role">{a.title}</div>
          </div>
        ))}
      </div>
      <div className="thinkingbar">{thinking ? "the panel is deciding who asks next…" : ""}</div>

      <div className="captions">
        {agentCap && (
          <div className="cap agent">
            <div className="who">
              {agentCap.name} · {agentCap.title}
            </div>
            <div className="txt">{agentCap.text}</div>
          </div>
        )}
        {candCap && (
          <div className="cap cand">
            <div className="who">You said</div>
            <div className="txt">{candCap}</div>
          </div>
        )}
      </div>

      <div className="debugbar">
        <button className="ghost" onClick={() => setShowDebug((s) => !s)}>
          {showDebug ? "▾ Hide panel internals" : "▸ Panel internals (coverage & evidence)"}
        </button>
        {contradictions > 0 && (
          <span className="flag">⚠ {contradictions} contradiction{contradictions > 1 ? "s" : ""}</span>
        )}
      </div>

      {showDebug && (
        <div className="debug">
          <div className="cov">
            <div className="cov-title">Competency coverage</div>
            {coverage.length === 0 && <div className="empty">— no evidence yet —</div>}
            {coverage.map((c) => (
              <div className="cov-row" key={c.key}>
                <span className="cov-name">{c.name}</span>
                <span className="cov-bar">
                  <span className="cov-fill" style={{ width: `${Math.round(c.value * 100)}%` }} />
                </span>
                <span className="cov-val">{Math.round(c.value * 100)}%</span>
              </div>
            ))}
          </div>
          <div className="claimlist">
            <div className="cov-title">Evidence ledger ({claims.length})</div>
            {claims.length === 0 && <div className="empty">— no claims extracted yet —</div>}
            {claims
              .slice()
              .reverse()
              .map((cl, i) => (
                <div
                  className={"claim " + (cl.status === "VAGUE" ? "vague" : "solid") + (cl.contradicts ? " contra" : "")}
                  key={i}
                >
                  <span className="claim-status">{cl.status}</span>
                  <span className="claim-text">{cl.text}</span>
                  <span className="claim-meta">
                    {cl.competency} · {Math.round(cl.strength * 100)}% · turn {cl.turn}
                    {cl.contradicts ? " · ⚠ contradiction" : ""}
                  </span>
                </div>
              ))}
          </div>
        </div>
      )}

      <div className="log" ref={logRef}>
        {logLines.join("\n")}
      </div>
    </div>
  );
}
