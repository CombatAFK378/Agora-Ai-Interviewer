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
  const [talking, setTalking] = useState(false); // in voice with the panel (Phase 6)
  const [status, setStatus] = useState("Idle.");
  const [speaking, setSpeaking] = useState(null); // agent id
  const [thinking, setThinking] = useState(false);
  const [agentCap, setAgentCap] = useState(null); // {name,title,text}
  const [candCap, setCandCap] = useState(null); // text
  const [coverage, setCoverage] = useState([]); // [{key,name,value}]
  const [claims, setClaims] = useState([]); // [{text,competency,strength,status,turn,contradicts}]
  const [contradictions, setContradictions] = useState(0);
  const [showDebug, setShowDebug] = useState(false);
  const [report, setReport] = useState(null);
  const [scoring, setScoring] = useState(false);
  // Ask the Panel (Phase 6)
  const [askTarget, setAskTarget] = useState("");
  const [askQ, setAskQ] = useState("");
  const [asking, setAsking] = useState(false);
  const [qa, setQa] = useState([]); // [{q, by, a}]
  const [cfAgent, setCfAgent] = useState("");
  const [cfTurn, setCfTurn] = useState("");
  const [cfHypo, setCfHypo] = useState("");
  const [ovDecision, setOvDecision] = useState("");
  const [ovReason, setOvReason] = useState("");
  const [overrides, setOverrides] = useState([]);
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
      } else if (ev.type === "override" && ev.override) {
        setOverrides((o) => [...o, ev.override]); // override fired by voice
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

  async function connectAgora(s) {
    openEvents();
    client.current = AgoraRTC.createClient({ mode: "rtc", codec: "vp8" });
    client.current.on("user-published", async (user, mt) => {
      await client.current.subscribe(user, mt);
      if (mt === "audio") user.audioTrack.play();
    });
    await client.current.join(s.app_id, s.channel, s.token, s.uid);
    mic.current = await makeMicTrack();
    await client.current.publish([mic.current]);
  }

  async function disconnectAgora() {
    if (ws.current) { ws.current.close(); ws.current = null; }
    if (mic.current) { mic.current.stop(); mic.current.close(); mic.current = null; }
    if (client.current) { try { await client.current.leave(); } catch (_) {} client.current = null; }
  }

  async function join() {
    setStatus("Starting session…");
    try {
      const resp = await fetch("/session/start", { method: "POST" });
      if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
      const s = await resp.json();
      log(`channel=${s.channel} uid=${s.uid}`);
      setStatus("Joining channel…");
      await connectAgora(s);
      setJoined(true);
      setStatus("Live — the panel will greet you shortly.");
      log("🎤 mic published");
    } catch (e) {
      log("ERROR: " + (e.message || e));
      setStatus("Failed — see log.");
    }
  }

  async function panelJoin() {
    setStatus("Joining the panel…");
    try {
      await disconnectAgora();          // leave the interview channel first
      const resp = await fetch("/panel/join", { method: "POST" });
      if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
      const s = await resp.json();
      await connectAgora(s);
      setJoined(true);
      setTalking(true);
      setSpeaking(null);
      setStatus("In voice with the panel — just talk.");
      log("🎙️ joined the panel — ask by voice");
    } catch (e) {
      log("panel join error: " + (e.message || e));
      setStatus("Failed — see log.");
    }
  }

  async function leave() {
    try {
      await disconnectAgora();
      await fetch("/session/stop", { method: "POST" });
    } catch (e) {
      log("leave error: " + (e.message || e));
    }
    setJoined(false);
    setTalking(false);
    setSpeaking(null);
    setThinking(false);
    setStatus("Idle.");
    log("left channel");
  }

  const nameOf = (id) => agents.find((a) => a.id === id)?.name || id;
  const titleOf = (id) => agents.find((a) => a.id === id)?.title || id;

  async function finish() {
    setScoring(true);
    log("📋 scoring the interview — locking, debating, concluding…");
    try {
      const r = await fetch("/session/conclude", { method: "POST" });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      setReport(await r.json());
      log("📋 report ready");
    } catch (e) {
      log("scoring error: " + (e.message || e));
    }
    setScoring(false);
  }

  async function ask() {
    if (!askQ.trim()) return;
    setAsking(true);
    try {
      const r = await fetch("/panel/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: askQ, mode: askTarget ? "addressed" : "open", target: askTarget || null }),
      });
      const d = await r.json();
      setQa((q) => [...q, { q: askQ, by: nameOf(d.answered_by), a: d.answer }]);
      if (d.override) setOverrides((o) => [...o, d.override]); // override tool fired
      setAskQ("");
    } catch (e) { log("ask error: " + (e.message || e)); }
    setAsking(false);
  }

  async function askCounterfactual() {
    if (!cfAgent || !cfTurn || !cfHypo.trim()) return;
    setAsking(true);
    try {
      const r = await fetch("/panel/counterfactual", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ turn: Number(cfTurn), hypothetical: cfHypo, agent_id: cfAgent }),
      });
      const d = await r.json();
      setQa((q) => [...q, {
        q: `Counterfactual @turn ${cfTurn} for ${nameOf(cfAgent)}: “${cfHypo}”`,
        by: nameOf(cfAgent),
        a: `Would move ${Math.round(d.original_overall * 100)} → ${Math.round(d.new_overall * 100)}. ${d.changes} (locked score unchanged)`,
      }]);
      setCfHypo("");
    } catch (e) { log("counterfactual error: " + (e.message || e)); }
    setAsking(false);
  }

  async function submitOverride() {
    if (!ovDecision) return;
    try {
      const r = await fetch("/panel/override", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ decision: ovDecision, reason: ovReason }),
      });
      const d = await r.json();
      setOverrides((o) => [...o, d]);
      setOvReason("");
    } catch (e) { log("override error: " + (e.message || e)); }
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
        <button className="finish" onClick={finish} disabled={!joined || scoring}>
          {scoring ? "Scoring…" : "Finish & score"}
        </button>
        <button onClick={leave} disabled={!joined}>
          Leave
        </button>
        <span className="status">{status}</span>
      </div>

      {report && (
        <div className="report">
          <div className="rec-line">
            <span className={"rec rec-" + report.conclusion.recommendation}>
              {report.conclusion.recommendation.replace(/_/g, " ")}
            </span>
            {overrides.length > 0 && (
              <span className={"rec rec-" + overrides[overrides.length - 1].decision}>
                → overridden: {overrides[overrides.length - 1].decision.replace(/_/g, " ")}
              </span>
            )}
          </div>
          <div className="headline">{report.conclusion.headline}</div>

          <div className="score-grid">
            {report.scores.map((s) => (
              <div className="scorecard" key={s.agent_id}>
                <div className="sc-head">
                  <span className="sc-name">{nameOf(s.agent_id)}</span>
                  <span className={"chip " + (s.conviction === "STRONG" ? "strong" : "neutral")}>
                    {s.conviction}
                  </span>
                </div>
                <div className="sc-role">{titleOf(s.agent_id)}</div>
                <div className="sc-overall">
                  {Math.round(s.overall * 100)}<span>/100</span>
                </div>
                {Object.entries(s.competency_scores).map(([k, v]) => (
                  <div className="sc-comp" key={k}>
                    <span className="sc-comp-name">{k}</span>
                    <span className="sc-bar">
                      <span className="sc-fill" style={{ width: `${Math.round(v * 100)}%` }} />
                    </span>
                  </div>
                ))}
                {s.rationale && <div className="sc-rat">{s.rationale}</div>}
              </div>
            ))}
          </div>

          <div className="rep-section">
            <div className="rep-title">Debate</div>
            {report.debate.map((d, i) => (
              <div className="deb" key={i}>
                <span className={"deb-act " + (d.rejected ? "rej" : d.action.toLowerCase())}>
                  {d.rejected ? "MOVE→held" : d.action}
                </span>
                <b>{nameOf(d.agent_id)}</b>
                {d.action === "MOVE" && !d.rejected && (
                  <span className="deb-move">
                    {" "}{Math.round(d.score_before * 100)}→{Math.round(d.score_after * 100)}
                  </span>
                )}
                <span className="deb-text"> {d.statement}</span>
              </div>
            ))}
          </div>

          <div className="rep-section">
            <div className="rep-title">Conclusion</div>
            <p className="rep-reason">{report.conclusion.reasoning}</p>
            {report.conclusion.unresolved.length > 0 && (
              <div className="rep-title2">Unresolved — a human should verify</div>
            )}
            {report.conclusion.unresolved.map((u, i) => (
              <div className="unres" key={i}>
                <span className="unres-item">{u.item}</span>
                <span className="unres-ev">{u.evidence}</span>
              </div>
            ))}
          </div>

          <div className="rep-hash">🔒 locked record · SHA-256 {report.locked_hash.slice(0, 24)}…</div>

          <div className="rep-section askpanel">
            <div className="rep-title">Ask the panel</div>

            <div className="voice-join">
              {!talking ? (
                <button className="join" onClick={panelJoin}>🎙️ Join to talk to the panel (voice)</button>
              ) : (
                <span className="talking-note">
                  🎙️ In voice with the panel — just speak.
                  {speaking ? ` ${nameOf(speaking)} is answering…` : ""}
                </span>
              )}
              <span className="voice-or">— or type —</span>
            </div>

            {qa.map((x, i) => (
              <div className="qa" key={i}>
                <div className="qa-q">Q: {x.q}</div>
                <div className="qa-a"><b>{x.by}:</b> {x.a}</div>
              </div>
            ))}

            <div className="ask-row">
              <select value={askTarget} onChange={(e) => setAskTarget(e.target.value)}>
                <option value="">Open (host)</option>
                {report.scores.map((s) => (
                  <option key={s.agent_id} value={s.agent_id}>{nameOf(s.agent_id)}</option>
                ))}
              </select>
              <input
                value={askQ}
                placeholder="e.g. Why did you flag this candidate?"
                onChange={(e) => setAskQ(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && ask()}
              />
              <button onClick={ask} disabled={asking}>{asking ? "…" : "Ask"}</button>
            </div>

            <div className="ask-sub">Counterfactual — “what if they had said…”</div>
            <div className="ask-row">
              <select value={cfAgent} onChange={(e) => setCfAgent(e.target.value)}>
                <option value="">interviewer…</option>
                {report.scores.map((s) => (
                  <option key={s.agent_id} value={s.agent_id}>{nameOf(s.agent_id)}</option>
                ))}
              </select>
              <input className="cf-turn" type="number" value={cfTurn}
                placeholder="turn" onChange={(e) => setCfTurn(e.target.value)} />
              <input value={cfHypo} placeholder="hypothetical answer…"
                onChange={(e) => setCfHypo(e.target.value)} />
              <button onClick={askCounterfactual} disabled={asking}>Re-score</button>
            </div>

            <div className="ask-sub">Override the recommendation</div>
            <div className="ask-row">
              <select value={ovDecision} onChange={(e) => setOvDecision(e.target.value)}>
                <option value="">decision…</option>
                {["PROCEED", "PROCEED_FLAGGED", "INSUFFICIENT_SIGNAL", "DECLINE"].map((d) => (
                  <option key={d} value={d}>{d.replace(/_/g, " ")}</option>
                ))}
              </select>
              <input value={ovReason} placeholder="reason (logged)"
                onChange={(e) => setOvReason(e.target.value)} />
              <button onClick={submitOverride}>Log override</button>
            </div>
            {overrides.map((o, i) => (
              <div className="ovr" key={i}>
                override: <b>{o.original_recommendation.replace(/_/g, " ")}</b> →{" "}
                <b>{o.decision.replace(/_/g, " ")}</b> — {o.reason}
                <span className="ovr-note"> (original recommendation kept)</span>
              </div>
            ))}
          </div>
        </div>
      )}

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
