import { useState, useEffect, useRef, useCallback } from "react";
import AgoraRTC from "agora-rtc-sdk-ng";
import { AIDenoiserExtension } from "agora-extension-ai-denoiser";
import { startGeminiCoding } from "./geminiCoding";

// The denoiser's WASM is loaded from the CDN (kept out of the bundle). If the
// browser blocks it, we fall back to the raw mic — see makeMicTrack().
const DENOISER_ASSETS =
  "https://cdn.jsdelivr.net/npm/agora-extension-ai-denoiser@2.0.2/external";
const AVATAR_COLORS = ["#4f8cff", "#a855f7", "#ef4444", "#14b8a6", "#f59e0b", "#64748b"];

const initials = (name) => (name || "?").slice(0, 1).toUpperCase();

const prettyKey = (k) =>
  (k || "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());

// Confidence trajectory (§11): mean coverage over turns, bold, with faint
// per-competency lines. Pure inline SVG — no chart library.
function ConfidenceChart({ trajectory }) {
  if (!trajectory || trajectory.length < 2) return null;
  const W = 640, H = 200, pad = 28;
  const n = trajectory.length;
  const x = (i) => pad + (i * (W - 2 * pad)) / (n - 1);
  const y = (v) => H - pad - Math.max(0, Math.min(1, v)) * (H - 2 * pad);
  const keys = Object.keys(trajectory[trajectory.length - 1].coverage || {});
  const line = (getter) => trajectory.map((p, i) => `${x(i)},${y(getter(p))}`).join(" ");
  const meanPts = line((p) => p.mean);
  const areaPts = `${pad},${H - pad} ${meanPts} ${x(n - 1)},${H - pad}`;
  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" className="chart-svg">
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line x1={pad} y1={y(g)} x2={W - pad} y2={y(g)} className="grid" />
            <text x={4} y={y(g) + 3} className="axis">{Math.round(g * 100)}</text>
          </g>
        ))}
        {keys.map((k, ci) => (
          <polyline
            key={k}
            points={line((p) => (p.coverage || {})[k] ?? 0)}
            fill="none"
            stroke={AVATAR_COLORS[ci % AVATAR_COLORS.length]}
            strokeWidth="1"
            opacity="0.45"
          />
        ))}
        <polygon points={areaPts} className="chart-area" />
        <polyline points={meanPts} className="chart-mean" fill="none" />
        {trajectory.map((p, i) => (
          <circle key={i} cx={x(i)} cy={y(p.mean)} r="2.5" className="chart-dot" />
        ))}
      </svg>
      <div className="chart-legend">
        <span className="lg lg-mean">panel mean</span>
        {keys.map((k, ci) => (
          <span className="lg" key={k}>
            <i style={{ background: AVATAR_COLORS[ci % AVATAR_COLORS.length] }} />
            {prettyKey(k)}
          </span>
        ))}
      </div>
    </div>
  );
}

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
  // Phase 7 — dossier: JD + résumé grounding.
  const [jd, setJd] = useState("");
  const [resume, setResume] = useState("");
  const [jdMeta, setJdMeta] = useState(null); // {filename, pages}
  const [resumeMeta, setResumeMeta] = useState(null);
  const [uploading, setUploading] = useState(null); // "jd" | "resume" | null
  const [dossier, setDossier] = useState(null); // parsed preview
  const [parsing, setParsing] = useState(false);
  const [activePanel, setActivePanel] = useState([]); // dossier-selected interviewer ids
  // Phase 8 — recruiter dashboard.
  const [view, setView] = useState("room"); // "room" | "dashboard"
  const [interviews, setInterviews] = useState([]); // past interview summaries
  const [storedId, setStoredId] = useState(null); // opened stored interview id
  // Phase 8 — coding round (screen share + vision).
  const [codingTask, setCodingTask] = useState(null);       // snapshot-mode task text
  const [geminiTask, setGeminiTask] = useState(null);       // Gemini-mode task text
  const [geminiActive, setGeminiActive] = useState(false);  // Gemini Live session running
  const [sharing, setSharing] = useState(false);
  const [finished, setFinished] = useState([]); // interviewer ids done (e.g. coding)
  const screenRef = useRef(null); // MediaStream
  const frameTimer = useRef(null);
  const geminiRef = useRef(null); // Gemini Live controller
  const logRef = useRef(null);

  const client = useRef(null);
  const mic = useRef(null);
  const ws = useRef(null);
  const denoiser = useRef(null);

  const log = useCallback((m) => {
    const line = `[${new Date().toLocaleTimeString()}] ${m}`;
    setLogLines((l) => [...l.slice(-200), line]);
    // Mirror to the server (data/client.log) so it's inspectable outside the browser.
    try {
      fetch("/client-log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ line }),
        keepalive: true,
      });
    } catch { /* best-effort */ }
  }, []);
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
      } else if (ev.type === "coding_task") {
        setCodingTask(ev.text);
        log("🖥️ coding task set — share your screen so Liam can watch");
      } else if (ev.type === "coding_gemini") {
        setGeminiTask(ev.task);
        log("🖥️ coding task set — share your entire screen to code live with Liam");
      } else if (ev.type === "screen_read") {
        log("👁️ Liam sees: " + ev.text);
      } else if (ev.type === "coding_done") {
        setFinished((f) => (f.includes("coding") ? f : [...f, "coding"]));
        setCodingTask(null);
        setGeminiTask(null);
        setGeminiActive(false);
        stopSharing();
        stopGemini();
        if (mic.current) { try { mic.current.setMuted(false); } catch { /* noop */ } }
        log(
          ev.verdict === "cheating"
            ? "🚩 Liam flagged the coding round (outside help) — handing back to the panel"
            : "✅ coding round complete — Liam is done, panel continues"
        );
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

  async function uploadPdf(which, fileEl) {
    const f = fileEl.files && fileEl.files[0];
    if (!f) return;
    setUploading(which);
    setStatus(`Reading ${f.name}…`);
    try {
      const form = new FormData();
      form.append("file", f);
      const resp = await fetch("/parse-pdf", { method: "POST", body: form });
      if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
      const d = await resp.json();
      const meta = { filename: d.filename, pages: d.pages };
      if (which === "jd") {
        setJd(d.text);
        setJdMeta(meta);
      } else {
        setResume(d.text);
        setResumeMeta(meta);
      }
      setDossier(null); // stale once inputs change
      setStatus(`${d.filename} — ${d.pages} page${d.pages === 1 ? "" : "s"} read.`);
    } catch (e) {
      log("ERROR: " + (e.message || e));
      setStatus("PDF upload failed — see log.");
    } finally {
      setUploading(null);
      fileEl.value = ""; // allow re-selecting the same file
    }
  }

  async function previewDossier() {
    if (!jd.trim() && !resume.trim()) return;
    setParsing(true);
    setStatus("Reading JD + résumé…");
    try {
      const resp = await fetch("/session/dossier", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jd, resume }),
      });
      if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
      const d = await resp.json();
      setDossier(d);
      setStatus(`Dossier ready — ${d.seniority} ${d.role}, ${d.panel.length} on the panel.`);
    } catch (e) {
      log("ERROR: " + (e.message || e));
      setStatus("Couldn't parse — see log.");
    } finally {
      setParsing(false);
    }
  }

  async function join() {
    setStatus("Starting session…");
    try {
      const resp = await fetch("/session/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jd, resume }),
      });
      if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
      const s = await resp.json();
      setActivePanel(s.panel || []);
      setFinished([]);
      setCodingTask(null);
      setGeminiTask(null);
      setGeminiActive(false);
      log(`channel=${s.channel} uid=${s.uid} panel=${(s.panel || []).join(",")}`);
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
      const resp = await fetch("/panel/join", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ interview_id: storedId || null }),
      });
      if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
      const s = await resp.json();
      await connectAgora(s);
      setJoined(true);
      setTalking(true);
      setSpeaking(null);
      setFinished([]);    // the "coding done" greying is interview-only; Liam answers here
      setActivePanel([]); // Ask-the-Panel: every interviewer can speak, none greyed
      setStatus("In voice with the panel — just talk.");
      log("🎙️ joined the panel — ask by voice");
    } catch (e) {
      log("panel join error: " + (e.message || e));
      setStatus("Failed — see log.");
    }
  }

  async function shareScreen() {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: "monitor", frameRate: 1 }, // hint: whole screen
        audio: false,
      });
      // Enforce entire-screen only: the browser won't let us remove the window/tab
      // options, but we refuse anything that isn't the full monitor so a candidate
      // can't hide an AI tool in an unshared window (§8).
      const surface = stream.getVideoTracks()[0].getSettings().displaySurface;
      if (surface && surface !== "monitor") {
        stream.getTracks().forEach((t) => t.stop());
        setStatus("Please share your ENTIRE screen (not a window or tab), then try again.");
        log(`❌ shared a ${surface}, not the whole screen — rejected. Share entire screen.`);
        return;
      }
      screenRef.current = stream;
      setSharing(true);
      const video = document.createElement("video");
      video.srcObject = stream;
      video.muted = true;
      await video.play();
      const canvas = document.createElement("canvas");
      const capture = async () => {
        const w = video.videoWidth, h = video.videoHeight;
        if (!w || !h) return;
        const scale = Math.min(1, 1280 / w); // cap width so frames stay small
        canvas.width = Math.round(w * scale);
        canvas.height = Math.round(h * scale);
        canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
        const frame = canvas.toDataURL("image/jpeg", 0.6);
        try {
          await fetch("/coding/frame", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame }),
          });
        } catch { /* transient — next tick retries */ }
      };
      await capture();
      frameTimer.current = setInterval(capture, 5000); // a frame every 5s
      stream.getVideoTracks()[0].onended = stopSharing; // user hit browser "Stop sharing"
      log("🖥️ screen sharing started — Liam can see your editor");
    } catch (e) {
      log("screen share error: " + (e.message || e));
    }
  }

  function stopSharing() {
    if (frameTimer.current) {
      clearInterval(frameTimer.current);
      frameTimer.current = null;
    }
    if (screenRef.current) {
      screenRef.current.getTracks().forEach((t) => t.stop());
      screenRef.current = null;
    }
    setSharing(false);
  }

  function stopGemini() {
    if (geminiRef.current) {
      try { geminiRef.current.stop(); } catch { /* noop */ }
      geminiRef.current = null;
    }
  }

  // Gemini Live coding round (§8): share the ENTIRE screen, then talk + code live
  // with Liam (browser-side). The Agora panel is muted for the duration.
  async function startCodingGemini() {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: "monitor", frameRate: 2 },
        audio: false,
      });
      const surface = stream.getVideoTracks()[0].getSettings().displaySurface;
      if (surface && surface !== "monitor") {
        stream.getTracks().forEach((t) => t.stop());
        setStatus("Please share your ENTIRE screen (not a window or tab), then try again.");
        log(`❌ shared a ${surface}, not the whole screen — rejected.`);
        return;
      }
      screenRef.current = stream;
      stream.getVideoTracks()[0].onended = () => stopGemini();

      const r = await fetch("/coding/gemini-token", { method: "POST" });
      if (!r.ok) throw new Error(`token ${r.status} ${await r.text()}`);
      const { token, model } = await r.json();

      if (mic.current) { try { mic.current.setMuted(true); } catch { /* noop */ } } // silence Agora panel
      setGeminiActive(true);
      setStatus("Live coding with Liam — talk and code, he can see your screen.");
      log("🎙️ Gemini Live coding round started");

      geminiRef.current = await startGeminiCoding({
        token,
        model,
        task: geminiTask || "",
        screenStream: stream,
        log: (m) => log("gemini: " + m),
        onStatus: (s) => log("gemini: " + s),
        onSpeaking: (on) => setSpeaking(on ? "coding" : null),
        onFinish: async (verdict, summary) => {
          log(`gemini: round ended (${verdict})`);
          try {
            await fetch("/coding/result", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ verdict, summary }),
            });
          } catch (e) {
            log("coding result post failed: " + (e.message || e));
          }
          if (screenRef.current) {
            screenRef.current.getTracks().forEach((t) => t.stop());
            screenRef.current = null;
          }
          geminiRef.current = null;
          setGeminiActive(false);
          // coding_done event will clear the banner + unmute the panel
        },
      });
    } catch (e) {
      log("gemini coding error: " + (e.message || e));
      setStatus("Couldn't start live coding — see log.");
      setGeminiActive(false);
      if (mic.current) { try { mic.current.setMuted(false); } catch { /* noop */ } }
      // Tell the server so the round doesn't stall; it ends gracefully.
      try {
        await fetch("/coding/result", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ verdict: "error", summary: String(e.message || e) }),
        });
      } catch { /* noop */ }
    }
  }

  async function loadInterviews() {
    try {
      const r = await fetch("/interviews");
      if (!r.ok) throw new Error(`${r.status}`);
      setInterviews(await r.json());
    } catch (e) {
      log("dashboard load error: " + (e.message || e));
    }
  }

  async function openInterview(summary) {
    try {
      const r = await fetch(`/interviews/${summary.interview_id}/open`, { method: "POST" });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const rep = await r.json();
      setReport(rep);
      setStoredId(summary.interview_id);
      setQa([]);
      setOverrides(
        summary.override
          ? [{ original_recommendation: rep.conclusion.recommendation,
               decision: summary.override, reason: "(logged earlier)" }]
          : []
      );
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) {
      log("open interview error: " + (e.message || e));
    }
  }

  function backToList() {
    setStoredId(null);
    setReport(null);
    setOverrides([]);
    setQa([]);
    loadInterviews();
  }

  function switchView(v) {
    if (v === "room") {
      // Returning to the live room from the dashboard: drop any stored report we
      // were viewing (and leave a panel voice call if we joined one) so the room
      // resets to the fresh setup screen. A live interview can't reach here — the
      // dashboard tab is disabled while one is running.
      if (talking) leave();
      setStoredId(null);
      setReport(null);
      setOverrides([]);
      setQa([]);
      setView("room");
      if (!talking) setStatus("Idle.");
      return;
    }
    setView("dashboard");
    setStoredId(null);
    if (!joined) setReport(null);
    loadInterviews();
  }

  async function leave() {
    stopSharing();
    stopGemini();
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
    setCodingTask(null);
    setGeminiTask(null);
    setGeminiActive(false);
    setFinished([]);
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
      <div className="disclosure" role="note">
        <span className="dot" aria-hidden="true" />
        AI interview — every interviewer is AI. This session is recorded and transcribed.
      </div>
      <h1>968ms — AI Interview Panel</h1>
      <p className="sub">
        Five AI interviewers. Whoever's speaking lights up. Answer with your mic; use Interrupt to
        cut in.
      </p>

      <div className="nav">
        <button className={view === "room" ? "nav-on" : ""} onClick={() => switchView("room")}>
          🎥 Live interview
        </button>
        <button
          className={view === "dashboard" ? "nav-on" : ""}
          onClick={() => switchView("dashboard")}
          disabled={joined && !talking}
        >
          📋 Past interviews
        </button>
      </div>

      {view === "dashboard" && !report && (
        <div className="dash">
          <div className="rep-title">Past interviews ({interviews.length})</div>
          {interviews.length === 0 && (
            <div className="empty">— no interviews yet. Run one from the Live tab. —</div>
          )}
          {interviews.map((it) => (
            <button className="dash-row" key={it.interview_id} onClick={() => openInterview(it)}>
              <div className="dash-main">
                <span className="dash-name">{it.candidate_name || "Unnamed candidate"}</span>
                <span className="dash-role">{it.role || "—"}</span>
              </div>
              <div className="dash-meta">
                <span className={"rec rec-" + it.recommendation}>
                  {(it.recommendation || "").replace(/_/g, " ")}
                </span>
                {it.override && (
                  <span className={"rec rec-" + it.override}>
                    → {it.override.replace(/_/g, " ")}
                  </span>
                )}
                <span className="dash-date">
                  {it.created_at ? new Date(it.created_at * 1000).toLocaleString() : ""}
                </span>
              </div>
            </button>
          ))}
        </div>
      )}

      {view === "room" && !joined && !report && (
        <div className="setup">
          <div className="setup-head">
            <strong>Interview dossier</strong>
            <span className="setup-sub">
              Upload the job description and the candidate's résumé as PDFs (or paste the text) —
              the panel, competency weights, and questions adapt to the role. Optional; leave
              blank for a generic panel.
            </span>
          </div>
          <div className="setup-cols">
            <div className="setup-col">
              <label className="filebtn">
                <input
                  type="file"
                  accept="application/pdf,.pdf"
                  onChange={(e) => uploadPdf("jd", e.target)}
                  disabled={joined || uploading === "jd"}
                />
                {uploading === "jd"
                  ? "Reading…"
                  : jdMeta
                  ? `📄 ${jdMeta.filename} (${jdMeta.pages}p)`
                  : "📎 Upload JD PDF"}
              </label>
              <textarea
                className="setup-ta"
                placeholder="…or paste the job description here"
                value={jd}
                onChange={(e) => {
                  setJd(e.target.value);
                  setJdMeta(null);
                }}
                disabled={joined}
              />
            </div>
            <div className="setup-col">
              <label className="filebtn">
                <input
                  type="file"
                  accept="application/pdf,.pdf"
                  onChange={(e) => uploadPdf("resume", e.target)}
                  disabled={joined || uploading === "resume"}
                />
                {uploading === "resume"
                  ? "Reading…"
                  : resumeMeta
                  ? `📄 ${resumeMeta.filename} (${resumeMeta.pages}p)`
                  : "📎 Upload résumé PDF"}
              </label>
              <textarea
                className="setup-ta"
                placeholder="…or paste the candidate résumé here"
                value={resume}
                onChange={(e) => {
                  setResume(e.target.value);
                  setResumeMeta(null);
                }}
                disabled={joined}
              />
            </div>
          </div>
          <div className="setup-actions">
            <button
              onClick={previewDossier}
              disabled={parsing || (!jd.trim() && !resume.trim())}
            >
              {parsing ? "Reading…" : "Preview dossier"}
            </button>
          </div>
          {dossier && (
            <div className="dossier">
              <div className="dossier-role">
                {dossier.seniority} {dossier.role}
                {dossier.candidate_name && (
                  <span className="dossier-sum"> · candidate: {dossier.candidate_name}</span>
                )}
                {dossier.summary && <span className="dossier-sum"> — {dossier.summary}</span>}
              </div>
              {(dossier.focus || []).length > 0 && (
                <div className="dossier-focus">
                  <span className="dossier-label">Role focus:</span>{" "}
                  {dossier.focus.map((f, i) => (
                    <span className="pill" key={i}>
                      {f}
                    </span>
                  ))}
                </div>
              )}
              <div className="dossier-panel">
                <span className="dossier-label">Panel:</span>{" "}
                {dossier.panel.map((a) => (
                  <span className="pill" key={a}>
                    {(agents.find((g) => g.id === a) || {}).name || a}
                  </span>
                ))}
              </div>
              {Object.keys(dossier.competency_weights || {}).length > 0 && (
                <div className="dossier-weights">
                  <span className="dossier-label">Weights:</span>{" "}
                  {Object.entries(dossier.competency_weights).map(([k, v]) => (
                    <span className="pill pill-w" key={k}>
                      {k} {Math.round(v * 100)}%
                    </span>
                  ))}
                </div>
              )}
              {(dossier.resume_claims || []).length > 0 && (
                <div className="dossier-claims">
                  <span className="dossier-label">
                    Résumé claims to verify ({dossier.resume_claims.length}):
                  </span>
                  <ul>
                    {dossier.resume_claims.slice(0, 8).map((c, i) => (
                      <li key={i}>
                        {c.text} <em>({c.competency})</em>
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {view === "room" && (
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
      )}

      {view === "room" && joined && !talking && geminiTask && (
        <div className="coding">
          <div className="coding-head">
            <strong>🖥️ Live coding task (with Liam)</strong>
            {geminiActive ? (
              <button className="ghost" onClick={stopGemini}>End coding</button>
            ) : (
              <button className="join" onClick={startCodingGemini}>Share entire screen &amp; start</button>
            )}
          </div>
          <div className="coding-task">{geminiTask}</div>
          <div className="coding-note">
            {geminiActive
              ? "Live with Liam — he can see your screen and hear you. Talk through your code as you go."
              : "🎧 Use headphones (avoids echo). Share your ENTIRE screen to start — Liam watches you code and talks in real time. A window or tab share is rejected."}
          </div>
        </div>
      )}

      {view === "room" && joined && !talking && codingTask && (
        <div className="coding">
          <div className="coding-head">
            <strong>🖥️ Live coding task</strong>
            {sharing ? (
              <button className="ghost" onClick={stopSharing}>Stop sharing</button>
            ) : (
              <button className="join" onClick={shareScreen}>Share entire screen</button>
            )}
          </div>
          <div className="coding-task">{codingTask}</div>
          <div className="coding-note">
            {sharing
              ? "Sharing your whole screen — Liam can see your editor and will react to your code."
              : "You must share your ENTIRE screen (a window or tab will be rejected) so Liam can watch you code. Think out loud as you go."}
          </div>
        </div>
      )}

      {report && (
        <div className="report">
          {(storedId || talking) && (
            <div className="report-bar">
              {storedId && (
                <button className="ghost" onClick={backToList}>← Back to interviews</button>
              )}
              {talking && (
                <button onClick={leave}>Leave voice</button>
              )}
              <span className="status">{status}</span>
            </div>
          )}
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

          {report.trajectory && report.trajectory.length >= 2 && (
            <div className="rep-section">
              <div className="rep-title">Confidence trajectory (evidence coverage per turn)</div>
              <ConfidenceChart trajectory={report.trajectory} />
            </div>
          )}

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

      {view === "room" && (
        <div className="howto">
          🎤 <b>Your mic</b> is for <b>answering</b> — speak, then pause. It won't cut the interviewer
          off. &nbsp;•&nbsp; ✋ <b>Interrupt</b> is the only way to cut in while someone's talking.
          &nbsp;•&nbsp; 🧹 AI noise suppression cleans your mic automatically.
        </div>
      )}

      {(view === "room" || talking) && (
      <>
      <div className="tiles">
        {agents.map((a, i) => {
          const done = finished.includes(a.id);
          const offPanel = !done && activePanel.length > 0 && !activePanel.includes(a.id);
          const dim = offPanel || done;
          return (
            <div
              key={a.id}
              className={
                "tile" +
                (dim ? " off-panel" : "") +
                (!dim && speaking === a.id ? " speaking" : !dim && thinking ? " thinking" : "")
              }
            >
              <span className="badge">SPEAKING</span>
              {done && <span className="offbadge">✓ coding done</span>}
              {offPanel && <span className="offbadge">not on this panel</span>}
              <div className="avatar" style={{ background: AVATAR_COLORS[i % AVATAR_COLORS.length] }}>
                {initials(a.name)}
              </div>
              <div className="name">{a.name}</div>
              <div className="role">{a.title}</div>
            </div>
          );
        })}
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
      </>
      )}

      {view === "room" && (
        <div className="debugbar">
          <button className="ghost" onClick={() => setShowDebug((s) => !s)}>
            {showDebug ? "▾ Hide panel internals" : "▸ Panel internals (coverage & evidence)"}
          </button>
          {contradictions > 0 && (
            <span className="flag">⚠ {contradictions} contradiction{contradictions > 1 ? "s" : ""}</span>
          )}
        </div>
      )}

      {view === "room" && showDebug && (
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
