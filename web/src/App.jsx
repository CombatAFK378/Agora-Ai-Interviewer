import { useState, useEffect, useRef, useCallback } from "react";
import AgoraRTC from "agora-rtc-sdk-ng";
import { AIDenoiserExtension } from "agora-extension-ai-denoiser";
import { startGeminiCoding } from "./geminiCoding";

import PanelTiles from "./components/PanelTiles";
import Captions from "./components/Captions";
import LiveEvidence from "./components/LiveEvidence";
import SetupDossier from "./components/SetupDossier";
import Dashboard from "./components/Dashboard";
import Report from "./components/Report";
import AskPanel from "./components/AskPanel";
import CodingRound, { CodingVerdict } from "./components/CodingRound";
import Stage from "./components/Stage";
import Timeline from "./components/Timeline";
import { startLevelMeters, createPanner } from "./lib/audioLevel";

// The denoiser's WASM is loaded from the CDN (kept out of the bundle). If the
// browser blocks it, we fall back to the raw mic - see makeMicTrack().
const DENOISER_ASSETS =
  "https://cdn.jsdelivr.net/npm/agora-extension-ai-denoiser@2.0.2/external";

export default function App() {
  const [agents, setAgents] = useState([]); // [{id,name,title}]
  const [joined, setJoined] = useState(false);
  const [talking, setTalking] = useState(false); // in voice with the panel (Phase 6)
  const [status, setStatus] = useState("Idle.");
  const [busy, setBusy] = useState(false); // status spinner
  const [micMuted, setMicMuted] = useState(false); // local mic mute (candidate / recruiter)
  const [liveTranscript, setLiveTranscript] = useState(""); // streamed transcript preview
  const [speaking, setSpeaking] = useState(null); // agent id
  const [thinking, setThinking] = useState(false);
  const [agentCap, setAgentCap] = useState(null); // {name,title,text}
  const [candCap, setCandCap] = useState(null); // text
  const [coverage, setCoverage] = useState([]); // [{key,name,value}]
  const [claims, setClaims] = useState([]); // [{text,competency,strength,status,turn,contradicts}]
  const [contradictions, setContradictions] = useState(0);
  const [report, setReport] = useState(null);
  const [scoring, setScoring] = useState(false);
  // Ask the Panel (Phase 6)
  const [askTarget, setAskTarget] = useState("");
  const [askQ, setAskQ] = useState("");
  const [asking, setAsking] = useState(false);
  const [qa, setQa] = useState([]); // [{q, by, a, agentId, mode, ts}]
  const [cfAgent, setCfAgent] = useState("");
  const [cfTurn, setCfTurn] = useState("");
  const [cfHypo, setCfHypo] = useState("");
  const [ovDecision, setOvDecision] = useState("");
  const [ovReason, setOvReason] = useState("");
  const [overrides, setOverrides] = useState([]);
  const [logLines, setLogLines] = useState([]);
  const [showLog, setShowLog] = useState(false);
  // Presenter mode: scales the whole type and spacing scale for a projector.
  const [presenting, setPresenting] = useState(false);
  // True only for a report that just came from Finish and score, so the reveal
  // animation plays at the moment it means something and not every time a
  // stored interview is reopened.
  const [freshReport, setFreshReport] = useState(false);
  // Phase 7 - dossier: JD + resume grounding.
  const [jd, setJd] = useState("");
  const [resume, setResume] = useState("");
  const [jdMeta, setJdMeta] = useState(null); // {filename, pages}
  const [resumeMeta, setResumeMeta] = useState(null);
  const [uploading, setUploading] = useState(null); // "jd" | "resume" | null
  const [dossier, setDossier] = useState(null); // parsed preview
  const [parsing, setParsing] = useState(false);
  const [activePanel, setActivePanel] = useState([]); // dossier-selected interviewer ids
  // Phase 8 - recruiter dashboard.
  const [view, setView] = useState("room"); // "room" | "dashboard"
  const [interviews, setInterviews] = useState([]); // past interview summaries
  const [storedId, setStoredId] = useState(null); // opened stored interview id
  // Phase 8 - coding round (screen share + vision).
  const [codingTask, setCodingTask] = useState(null);       // snapshot-mode task text
  const [geminiTask, setGeminiTask] = useState(null);       // Gemini-mode task text
  const [geminiActive, setGeminiActive] = useState(false);  // Gemini Live session running
  const [sharing, setSharing] = useState(false);
  const [screenRead, setScreenRead] = useState(null);       // what the vision model sees
  const [codingResult, setCodingResult] = useState(null);   // {verdict, summary}
  const [finished, setFinished] = useState([]); // interviewer ids done (e.g. coding)
  // Live stage: elapsed clock and turn count, both derived from events we
  // already receive rather than from anything new on the wire.
  const [elapsed, setElapsed] = useState(0);
  const [turnCount, setTurnCount] = useState(0);
  // Bumped when the inbound track arrives. remoteTrack is a ref written inside
  // an async Agora callback, which triggers no render, so without this the
  // level meters could start against null and never retry.
  const [remoteReady, setRemoteReady] = useState(0);
  // The interview, kept rather than discarded. Every frame is assembled from
  // events already arriving; nothing extra crosses the wire.
  const [timeline, setTimeline] = useState([]);
  const [scrubIndex, setScrubIndex] = useState(null); // null = live
  // Event handlers are registered once, so they need refs to read current
  // values rather than the values captured when the socket was opened.
  const coverageRef = useRef([]);
  const claimsRef = useRef([]);
  const contraRef = useRef(0);
  const elapsedRef = useRef(0);
  const turnRef = useRef(0);
  const frameId = useRef(0);
  const stageRef = useRef(null);
  const pushFrame = useCallback((entry) => {
    setTimeline((tl) => [...tl, {
      id: ++frameId.current,
      t: elapsedRef.current,
      turn: turnRef.current,
      coverage: coverageRef.current,
      claims: claimsRef.current,
      claimCount: claimsRef.current.length,
      contradictions: contraRef.current,
      ...entry,
    }]);
  }, []);
  const remoteTrack = useRef(null);   // the panel's single inbound audio track
  const stopMeters = useRef(null);
  const panner = useRef(null);
  // Joining takes seconds (session start, channel join, getUserMedia, the
  // denoiser WASM). A second click in that window would build a second client
  // and a second audio graph, and everyone would be heard twice.
  const joining = useRef(false);

  const screenRef = useRef(null); // MediaStream
  const frameTimer = useRef(null);
  const geminiRef = useRef(null); // Gemini Live controller
  const codingSummary = useRef(null); // held so the verdict card can show it
  const logRef = useRef(null);

  const client = useRef(null);
  const mic = useRef(null);
  const cam = useRef(null);       // candidate camera track (mandatory in interview)
  const selfRef = useRef(null);   // self-view container
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

  useEffect(() => { coverageRef.current = coverage; }, [coverage]);
  useEffect(() => { claimsRef.current = claims; }, [claims]);
  useEffect(() => { contraRef.current = contradictions; }, [contradictions]);
  useEffect(() => { elapsedRef.current = elapsed; }, [elapsed]);

  // The live transcript preview is driven by the server's "partial" events (Sarvam
  // transcribes each phrase as the candidate pauses). Browser speech recognition
  // can't be used here — Agora holds the mic, so it would get no audio.

  // Presenter mode drives a class on <html> so the token overrides reach
  // everything, including elements outside the React root.
  useEffect(() => {
    document.documentElement.classList.toggle("presenting", presenting);
  }, [presenting]);

  // Shortcut for the projector, ignored while typing so it cannot fire from a
  // job description being pasted into a textarea.
  useEffect(() => {
    const onKey = (e) => {
      const t = e.target;
      const typing = t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" ||
                           t.tagName === "SELECT" || t.isContentEditable);
      if (typing || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === "p" || e.key === "P") setPresenting((v) => !v);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Phosphor comes from a CDN. A venue proxy or an ad blocker can drop it, and
  // an icon font that fails renders nothing at all, leaving the avatars and the
  // stage seats as empty circles. Detect it once and let the text stand in.
  useEffect(() => {
    let cancelled = false;
    const mark = () => {
      if (cancelled) return;
      const ok = document.fonts && document.fonts.check('1em "Phosphor"');
      if (!ok) document.documentElement.classList.add("no-icons");
    };
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(mark).catch(mark);
    } else {
      mark();
    }
    return () => { cancelled = true; };
  }, []);

  // Load the roster up front so tiles render before joining.
  useEffect(() => {
    fetch("/panel")
      .then((r) => r.json())
      .then((d) => setAgents(d.agents || []))
      .catch(() => {});
  }, []);

  // Interview clock. Only runs during a live interview, not in Ask the Panel.
  useEffect(() => {
    if (!joined || talking) return;
    const t = setInterval(() => setElapsed((s) => s + 1), 1000);
    return () => clearInterval(t);
  }, [joined, talking]);

  // Voice levels for the speaking ring and the candidate's own mic meter.
  // These write CSS custom properties at frame rate and never touch state.
  useEffect(() => {
    if (!joined || talking || report || !stageRef.current) return;
    stopMeters.current = startLevelMeters({
      el: stageRef.current,
      remoteTrack: remoteTrack.current,
      micTrack: (() => {
        try { return mic.current ? mic.current.getMediaStreamTrack() : null; }
        catch { return null; }
      })(),
    });
    return () => {
      if (stopMeters.current) { stopMeters.current(); stopMeters.current = null; }
    };
  }, [joined, talking, report, remoteReady]);

  // Move the voice to the speaking interviewer's seat.
  useEffect(() => {
    if (!panner.current) return;
    const seated = agents.filter(
      (a) => !finished.includes(a.id) &&
             (activePanel.length === 0 || activePanel.includes(a.id))
    );
    const i = seated.findIndex((a) => a.id === speaking);
    if (i < 0 || seated.length < 2) { panner.current.setSeat(0); return; }
    // spread the seats evenly across the stereo field, left to right
    panner.current.setSeat((i / (seated.length - 1)) * 2 - 1);
  }, [speaking, agents, activePanel, finished]);

  function openEvents() {
    const url = location.origin.replace(/^http/, "ws") + "/session/events";
    const sock = new WebSocket(url);
    ws.current = sock;
    sock.onmessage = (e) => {
      const ev = JSON.parse(e.data);
      if (ev.type === "panel") setAgents(ev.agents || []);
      else if (ev.type === "thinking") {
        setThinking(true);
        setSpeaking(null);
      } else if (ev.type === "speaking") {
        setThinking(false);
        setSpeaking(ev.agent);
        setAgentCap({ id: ev.agent, name: ev.name, title: ev.title, text: ev.text });
        turnRef.current += 1;
        setTurnCount(turnRef.current);
        pushFrame({ kind: "ask", agentId: ev.agent, name: ev.name, title: ev.title, text: ev.text });
        log(`${ev.name} (${ev.title}) is speaking`);
      } else if (ev.type === "idle") {
        setThinking(false);
        setSpeaking(null);
      } else if (ev.type === "partial") {
        setLiveTranscript(ev.text || "");   // streamed transcript preview
      } else if (ev.type === "heard") {
        setLiveTranscript("");              // final Sarvam transcript takes over
        setCandCap(ev.text);
        pushFrame({ kind: "answer", text: ev.text });
      } else if (ev.type === "redo") {
        setLiveTranscript("");
        setCandCap(null);
      } else if (ev.type === "ledger") {
        setCoverage(ev.coverage || []);
        setClaims(ev.claims || []);
        setContradictions(ev.contradictions || 0);
        // The ledger lands just after the turn that produced it, so fold the
        // new evidence back onto the frame it belongs to.
        setTimeline((tl) => {
          if (!tl.length) return tl;
          const head = tl[tl.length - 1];
          return [...tl.slice(0, -1), {
            ...head,
            coverage: ev.coverage || [],
            claims: ev.claims || [],
            claimCount: (ev.claims || []).length,
            contradictions: ev.contradictions || 0,
          }];
        });
      } else if (ev.type === "override" && ev.override) {
        setOverrides((o) => [...o, ev.override]); // override fired by voice
      } else if (ev.type === "coding_task") {
        setCodingTask(ev.text);
        setCodingResult(null);
        log("coding task set, share your screen so the coding interviewer can watch");
      } else if (ev.type === "coding_gemini") {
        setGeminiTask(ev.task);
        setCodingResult(null);
        log("coding task set, share your entire screen to code live");
      } else if (ev.type === "screen_read") {
        setScreenRead(ev.text);
        log("screen read: " + ev.text);
      } else if (ev.type === "coding_done") {
        setFinished((f) => (f.includes("coding") ? f : [...f, "coding"]));
        setCodingTask(null);
        setGeminiTask(null);
        setGeminiActive(false);
        setScreenRead(null);
        setCodingResult({ verdict: ev.verdict, summary: codingSummary.current });
        stopSharing();
        stopGemini();
        if (mic.current) { try { mic.current.setMuted(false); } catch { /* noop */ } }
        log(
          ev.verdict === "cheating"
            ? "coding round flagged for outside help, handing back to the panel"
            : "coding round complete, the panel continues"
        );
      }
    };
    sock.onclose = () => log("events socket closed");
    sock.onerror = () => log("events socket error");
  }

  async function makeMicTrack() {
    // Keep AEC (echo), ANS (light noise), AND AGC (auto-gain) — AGC normalizes the
    // level so the server's RMS gates work across mics/rooms; without it real speech
    // fell below the threshold and got dropped. The real STT-killer was the heavy AI
    // denoiser (it mangles phonetics), so that stays OFF. High-quality encoder keeps
    // the signal to the bot clean.
    const track = await AgoraRTC.createMicrophoneAudioTrack({
      AEC: true, ANS: true, AGC: true,
      encoderConfig: "high_quality",   // 48 kHz mono, higher bitrate
    });
    log("mic: high-quality profile, AEC + ANS + AGC on, AI-denoiser off");
    return track;
  }

  async function connectAgora(s, withCamera = false) {
    openEvents();
    client.current = AgoraRTC.createClient({ mode: "rtc", codec: "vp8" });
    client.current.on("user-published", async (user, mt) => {
      try {
        await client.current.subscribe(user, mt);
      } catch (e) {
        // Agora does not re-emit for a failed subscribe, so without this log an
        // audio failure is completely silent, in both senses.
        log("subscribe failed: " + (e.message || e));
        return;
      }
      if (mt === "audio") {
        // Agora re-emits user-published after a republish or a reconnect. Without
        // this teardown the old graph stays wired to its own destination and is
        // unreachable, so the panel plays twice and cannot be stopped.
        if (panner.current) { panner.current.stop(); panner.current = null; }

        let track = null;
        try { track = user.audioTrack.getMediaStreamTrack(); } catch { /* noop */ }
        remoteTrack.current = track;
        setRemoteReady((v) => v + 1);

        // Route through our own graph so each interviewer can be panned to
        // their seat. If that fails we hand playback straight back to Agora,
        // so the worst case is centred audio, never silence.
        const p = track ? createPanner(track) : null;
        if (p) {
          panner.current = p;
          // Construction succeeding does not mean audio is flowing: an
          // AudioContext built off the user gesture can stay suspended. Check
          // shortly after and fall back on our own, because nobody can recover
          // from silence manually mid-demo.
          setTimeout(() => {
            if (panner.current === p && !p.running()) {
              p.stop();
              panner.current = null;
              try { user.audioTrack.play(); } catch { /* noop */ }
              log("spatial audio did not start, fell back to plain playback");
            }
          }, 800);
          log("spatial audio on, each interviewer is panned to their seat");
        } else {
          user.audioTrack.play();
        }
      }
    });
    const dropRemote = () => {
      if (panner.current) { panner.current.stop(); panner.current = null; }
      remoteTrack.current = null;
    };
    client.current.on("user-unpublished", (_u, mt) => { if (mt === "audio") dropRemote(); });
    client.current.on("user-left", () => dropRemote());
    client.current.on("connection-state-change", (state) => {
      if (state === "DISCONNECTED" || state === "RECONNECTING") dropRemote();
    });
    await client.current.join(s.app_id, s.channel, s.token, s.uid);
    mic.current = await makeMicTrack();
    const tracks = [mic.current];
    if (withCamera) {
      // Camera is MANDATORY for the interview — if it can't be opened, fail the
      // join so the candidate must grant access.
      try {
        cam.current = await AgoraRTC.createCameraVideoTrack({ encoderConfig: "480p_1" });
      } catch (e) {
        throw new Error("CAMERA_REQUIRED: " + (e.message || e));
      }
      tracks.push(cam.current);
      if (selfRef.current) { try { cam.current.play(selfRef.current); } catch { /* noop */ } }
    }
    await client.current.publish(tracks);
  }

  async function disconnectAgora() {
    if (stopMeters.current) { stopMeters.current(); stopMeters.current = null; }
    if (panner.current) { panner.current.stop(); panner.current = null; }
    remoteTrack.current = null;
    if (ws.current) { ws.current.close(); ws.current = null; }
    if (mic.current) { mic.current.stop(); mic.current.close(); mic.current = null; }
    if (cam.current) { cam.current.stop(); cam.current.close(); cam.current = null; }
    if (client.current) { try { await client.current.leave(); } catch (_) {} client.current = null; }
  }

  async function uploadPdf(which, fileEl) {
    const f = fileEl.files && fileEl.files[0];
    if (!f) return;
    setUploading(which);
    setStatus(`Reading ${f.name}`);
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
      setStatus(`${d.filename}, ${d.pages} page${d.pages === 1 ? "" : "s"} read.`);
    } catch (e) {
      log("ERROR: " + (e.message || e));
      setStatus("PDF upload failed, see the log.");
    } finally {
      setUploading(null);
      fileEl.value = ""; // allow re-selecting the same file
    }
  }

  async function previewDossier() {
    if (!jd.trim() && !resume.trim()) return;
    setParsing(true);
    setBusy(true);
    setStatus("Reading the job description and resume");
    try {
      const resp = await fetch("/session/dossier", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ jd, resume }),
      });
      if (!resp.ok) throw new Error(`${resp.status} ${await resp.text()}`);
      const d = await resp.json();
      setDossier(d);
      setStatus(`Dossier ready. ${d.seniority} ${d.role}, ${d.panel.length} on the panel.`);
    } catch (e) {
      log("ERROR: " + (e.message || e));
      setStatus("Could not parse that, see the log.");
    } finally {
      setParsing(false);
      setBusy(false);
    }
  }

  async function join() {
    if (joining.current || joined) return; // re-entrancy guard, see joining ref
    joining.current = true;
    setBusy(true);
    setStatus("Starting session, building the panel from the dossier");
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
      setCodingResult(null);
      codingSummary.current = null;
      setElapsed(0);
      setTurnCount(0);
      turnRef.current = 0;
      setTimeline([]);
      setScrubIndex(null);
      setAgentCap(null);
      setCandCap(null);
      log(`channel=${s.channel} uid=${s.uid} panel=${(s.panel || []).join(",")}`);
      setStatus("Joining the channel");
      await connectAgora(s, true);   // camera is mandatory for the interview
      setJoined(true);
      setMicMuted(false);
      setStatus("Live. The panel will greet you shortly.");
      log("mic + camera published");
    } catch (e) {
      log("ERROR: " + (e.message || e));
      if (String(e.message || e).includes("CAMERA_REQUIRED")) {
        setStatus("Camera is required for the interview — allow camera access and Join again.");
      } else {
        setStatus("Failed to start, see the log.");
      }
      try { await disconnectAgora(); } catch { /* cleanup */ } // don't leave a half-joined session
      await fetch("/session/stop", { method: "POST" }).catch(() => {});
    } finally {
      joining.current = false;
      setBusy(false);
    }
  }

  async function panelJoin() {
    setBusy(true);
    setStatus("Joining the panel");
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
      setMicMuted(false);
      setFinished([]);    // the "coding done" greying is interview-only
      setActivePanel([]); // Ask-the-Panel: every interviewer can speak, none greyed
      setStatus("In voice with the panel, just talk.");
      log("joined the panel, ask by voice");
    } catch (e) {
      log("panel join error: " + (e.message || e));
      setStatus("Failed to join, see the log.");
    } finally {
      setBusy(false);
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
      // can't hide an AI tool in an unshared window.
      const surface = stream.getVideoTracks()[0].getSettings().displaySurface;
      if (surface && surface !== "monitor") {
        stream.getTracks().forEach((t) => t.stop());
        setStatus("Please share your ENTIRE screen, not a window or tab, then try again.");
        log(`shared a ${surface}, not the whole screen. Rejected.`);
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
        } catch { /* transient - next tick retries */ }
      };
      await capture();
      frameTimer.current = setInterval(capture, 5000); // a frame every 5s
      stream.getVideoTracks()[0].onended = stopSharing; // user hit browser "Stop sharing"
      log("screen sharing started");
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

  // Gemini Live coding round: share the ENTIRE screen, then talk + code live
  // (browser-side). The Agora panel is muted for the duration.
  async function startCodingGemini() {
    try {
      const stream = await navigator.mediaDevices.getDisplayMedia({
        video: { displaySurface: "monitor", frameRate: 2 },
        audio: false,
      });
      const surface = stream.getVideoTracks()[0].getSettings().displaySurface;
      if (surface && surface !== "monitor") {
        stream.getTracks().forEach((t) => t.stop());
        setStatus("Please share your ENTIRE screen, not a window or tab, then try again.");
        log(`shared a ${surface}, not the whole screen. Rejected.`);
        return;
      }
      screenRef.current = stream;
      stream.getVideoTracks()[0].onended = () => stopGemini();

      const r = await fetch("/coding/gemini-token", { method: "POST" });
      if (!r.ok) throw new Error(`token ${r.status} ${await r.text()}`);
      const { token, model } = await r.json();

      if (mic.current) { try { mic.current.setMuted(true); } catch { /* noop */ } } // silence Agora panel
      setGeminiActive(true);
      setStatus("Live coding. Talk and code, the interviewer can see your screen.");
      log("Gemini Live coding round started");

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
          codingSummary.current = summary; // held for the verdict card
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
      setStatus("Could not start live coding, see the log.");
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
      setFreshReport(false); // reopened, not just decided: no reveal
      setStoredId(summary.interview_id);
      setQa([]);
      setOverrides(
        summary.override
          ? [{ original_recommendation: (rep.conclusion || {}).recommendation,
               decision: summary.override, reason: "" }]
          : []
      );
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (e) {
      log("open interview error: " + (e.message || e));
    }
  }

  function backToList() {
    if (talking) leave(); // otherwise the mic stays hot with no visible control
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
      // resets to the fresh setup screen. A live interview can't reach here - the
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
    setMicMuted(false);
    clearLive();
    setCodingTask(null);
    setGeminiTask(null);
    setGeminiActive(false);
    setScreenRead(null);
    setCodingResult(null);
    codingSummary.current = null;
    turnRef.current = 0;
    setTimeline([]);
    setScrubIndex(null);
    setFinished([]);
    setStatus("Idle.");
    log("left channel");
  }

  const nameOf = (id) => agents.find((a) => a.id === id)?.name || id;
  const titleOf = (id) => agents.find((a) => a.id === id)?.title || id;

  async function finish() {
    setScoring(true);
    setBusy(true);
    setStatus("Locking the record, scoring, debating, concluding");
    log("scoring the interview");
    try {
      const r = await fetch("/session/conclude", { method: "POST" });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      setReport(await r.json());
      setFreshReport(true);
      setStatus("Report ready.");
      log("report ready");
    } catch (e) {
      log("scoring error: " + (e.message || e));
      setStatus("Scoring failed, see the log.");
    }
    setScoring(false);
    setBusy(false);
  }

  async function ask() {
    if (!askQ.trim()) return;
    setAsking(true);
    const asked = askQ;
    try {
      const r = await fetch("/panel/ask", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: askQ, mode: askTarget ? "addressed" : "open", target: askTarget || null }),
      });
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const d = await r.json();
      setQa((q) => [...q, {
        q: asked,
        by: nameOf(d.answered_by),
        a: d.answer,
        agentId: d.answered_by,
        mode: d.mode,
        ts: d.ts,
      }]);
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
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const d = await r.json();
      setQa((q) => [...q, {
        q: `If at turn ${cfTurn} they had said: "${cfHypo}"`,
        by: nameOf(cfAgent),
        a: `Would move ${Math.round(d.original_overall * 100)} to ${Math.round(d.new_overall * 100)}. ${d.changes} The locked score is unchanged.`,
        agentId: cfAgent,
        ts: d.ts,
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
      // Without this an error body was pushed into overrides and rendered as a
      // convincing "decision overridden" card for a write that never happened.
      if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
      const d = await r.json();
      setOverrides((o) => [...o, d]);
      setOvReason("");
    } catch (e) {
      log("override error: " + (e.message || e));
      setStatus("Override was not recorded, see the log.");
    }
  }

  async function interrupt() {
    try {
      const r = await fetch("/session/interrupt", { method: "POST" });
      const d = await r.json();
      // The response says whether anyone was actually cut off. That used to go
      // to the console log only, so the button gave no visible feedback.
      setStatus(d.interrupted ? "Interrupted. Go ahead and speak." : "Nobody is speaking right now.");
      log(d.interrupted ? "interrupted" : "interrupt: nobody was speaking");
    } catch (e) {
      log("interrupt error: " + (e.message || e));
    }
  }

  async function skipIntro() {
    try {
      await fetch("/session/interrupt", { method: "POST" }); // cut the host's disclosure
      setStatus("Skipped the intro.");
      log("skipped the intro");
    } catch (e) {
      log("skip intro error: " + (e.message || e));
    }
  }

  function toggleMic() {
    const next = !micMuted;
    try { mic.current && mic.current.setMuted(next); } catch { /* noop */ }
    setMicMuted(next);
    setStatus(next ? "Microphone muted." : "Microphone on.");
    log(next ? "🔇 mic muted" : "🎤 mic on");
  }

  function clearLive() {
    setLiveTranscript("");
  }

  async function finishTurn() {
    try {
      await fetch("/session/finish-turn", { method: "POST" });
      clearLive();
      setStatus("Got it — the panel is responding.");
      log("done — finalizing my turn");
    } catch (e) {
      log("finish-turn error: " + (e.message || e));
    }
  }

  async function talkAgain() {
    try {
      await fetch("/session/redo", { method: "POST" });
      clearLive();
      setCandCap(null);
      setStatus("Okay — go ahead and say that again.");
      log("talk again — discarded last answer, listening");
    } catch (e) {
      log("redo error: " + (e.message || e));
    }
  }

  // Rewind. Scrubbing is a pure view concern: it never touches the session,
  // the audio, or the controls, and live events keep appending underneath.
  const frame = scrubIndex != null ? timeline[scrubIndex] : null;
  let viewSpeaking = speaking, viewAgentCap = agentCap, viewCandCap = candCap;
  let viewCoverage = coverage, viewClaims = claims, viewContra = contradictions;
  let viewTurn = turnCount, viewElapsed = elapsed;
  if (frame) {
    viewCoverage = frame.coverage;
    viewClaims = frame.claims;
    viewContra = frame.contradictions;
    viewTurn = frame.turn;
    viewElapsed = frame.t;
    // Walk back for the question and the answer in force at that moment, so an
    // answer frame still shows the question it was answering.
    let ask = null, ans = null;
    for (let i = scrubIndex; i >= 0 && (!ask || !ans); i--) {
      const f = timeline[i];
      if (!ask && f.kind === "ask") ask = f;
      if (!ans && f.kind === "answer") ans = f;
    }
    viewAgentCap = ask
      ? { id: ask.agentId, name: ask.name, title: ask.title, text: ask.text }
      : null;
    viewCandCap = ans ? ans.text : null;
    // Only light the seat if the panel was actually mid-question here.
    viewSpeaking = frame.kind === "ask" ? frame.agentId : null;
  }

  const codingName = nameOf("coding");
  const showRoom = view === "room";
  // A live interview takes over the room: the setup chrome, the control strip,
  // the small tiles and the evidence cards are all replaced by the stage.
  const inInterview = showRoom && joined && !talking && !report;
  const showTiles = (showRoom || talking) && !inInterview;

  return (
    <div className="wrap">
      <header className="masthead">
        <div className="mast-id">
          <h1 className="mast-title"><b>968ms</b> AI Interview Panel</h1>
          <p className="mast-sub">
            A panel of AI interviewers with separate objectives. Whoever holds the floor
            lights up. Answer with your mic, and use Interrupt to cut in.
          </p>
        </div>
        <div className="mast-right">
          <div className="disclosure" role="note">
            <span className="dot" aria-hidden="true" />
            Every interviewer is AI. This session is recorded and transcribed.
          </div>
          <button
            className="ghost present-toggle"
            onClick={() => setPresenting((v) => !v)}
            aria-pressed={presenting}
            title="Scale the interface up for a projector (shortcut: P)"
          >
            <i className={"ph " + (presenting ? "ph-corners-in" : "ph-corners-out")}
               aria-hidden="true" />
            {presenting ? "Exit presenter mode" : "Presenter mode"}
          </button>
        </div>
      </header>

      <nav className="nav">
        <button className={showRoom ? "nav-on" : ""} onClick={() => switchView("room")}>
          <i className="ph-fill ph-broadcast" aria-hidden="true" />
          Live interview
        </button>
        <button
          className={view === "dashboard" ? "nav-on" : ""}
          onClick={() => switchView("dashboard")}
          disabled={joined && !talking}
        >
          <i className="ph ph-folders" aria-hidden="true" />
          Past interviews
        </button>
      </nav>

      {view === "dashboard" && !report && (
        <Dashboard interviews={interviews} onOpen={openInterview} />
      )}

      {showRoom && !joined && !report && (
        <SetupDossier
          jd={jd} setJd={setJd} resume={resume} setResume={setResume}
          jdMeta={jdMeta} setJdMeta={setJdMeta}
          resumeMeta={resumeMeta} setResumeMeta={setResumeMeta}
          uploading={uploading} uploadPdf={uploadPdf}
          parsing={parsing} previewDossier={previewDossier}
          dossier={dossier} joined={joined} nameOf={nameOf}
        />
      )}

      {showRoom && !inInterview && (
        <div className="controls">
          <button className="btn-primary" onClick={join} disabled={joined || busy}>
            <i className="ph-fill ph-play" aria-hidden="true" />
            Join interview
          </button>
          <button className="btn-flag" onClick={interrupt} disabled={!joined}>
            <i className="ph ph-hand-palm" aria-hidden="true" />
            Interrupt
          </button>
          <button className="btn-live" onClick={finish} disabled={!joined || scoring}>
            <i className="ph ph-flag-checkered" aria-hidden="true" />
            {scoring ? "Scoring" : "Finish and score"}
          </button>
          <button onClick={leave} disabled={!joined}>Leave</button>
          <span className="status">
            {busy && <i className="ph ph-circle-notch spin" aria-hidden="true" />}
            {status}
          </span>
        </div>
      )}

      {inInterview && (
        <Stage
          stageRef={stageRef}
          agents={agents}
          speaking={viewSpeaking}
          thinking={frame ? false : thinking}
          activePanel={activePanel}
          finished={finished}
          agentCap={viewAgentCap}
          candCap={viewCandCap}
          liveTranscript={frame ? "" : liveTranscript}
          elapsed={viewElapsed}
          turnCount={viewTurn}
          coverage={viewCoverage}
          claims={viewClaims}
          contradictions={viewContra}
          rewound={!!frame}
          timeline={
            <Timeline
              timeline={timeline}
              scrubIndex={scrubIndex}
              onScrub={setScrubIndex}
              onLive={() => setScrubIndex(null)}
            />
          }
          onInterrupt={interrupt}
          onFinish={finish}
          onLeave={leave}
          onDone={finishTurn}
          onTalkAgain={talkAgain}
          onToggleMic={toggleMic}
          onSkipIntro={skipIntro}
          micMuted={micMuted}
          scoring={scoring}
          status={status}
          busy={busy}
        />
      )}

      {showRoom && joined && !talking && (geminiTask || codingTask) && (
        <CodingRound
          mode={geminiTask ? "gemini" : "snapshot"}
          task={geminiTask || codingTask}
          active={geminiActive}
          sharing={sharing}
          screenRead={screenRead}
          codingName={codingName}
          onStartGemini={startCodingGemini}
          onStopGemini={stopGemini}
          onShareScreen={shareScreen}
          onStopSharing={stopSharing}
          status={status}
        />
      )}

      {showRoom && joined && codingResult && (
        <CodingVerdict
          verdict={codingResult.verdict}
          summary={codingResult.summary}
          codingName={codingName}
        />
      )}

      {report && (
        <Report
          report={report}
          overrides={overrides}
          nameOf={nameOf}
          titleOf={titleOf}
          fresh={freshReport}
          topBar={
            (storedId || talking) && (
              <div className="report-bar">
                {storedId && (
                  <button className="ghost" onClick={backToList}>
                    <i className="ph ph-arrow-left" aria-hidden="true" />
                    Back to interviews
                  </button>
                )}
                {talking && <button onClick={leave}>Leave voice</button>}
                <span className="status">{status}</span>
              </div>
            )
          }
        >
          <AskPanel
            report={report} nameOf={nameOf} talking={talking} speaking={speaking}
            onJoinVoice={panelJoin}
            onToggleMic={toggleMic} micMuted={micMuted}
            qa={qa}
            askTarget={askTarget} setAskTarget={setAskTarget}
            askQ={askQ} setAskQ={setAskQ} asking={asking} onAsk={ask}
            cfAgent={cfAgent} setCfAgent={setCfAgent}
            cfTurn={cfTurn} setCfTurn={setCfTurn}
            cfHypo={cfHypo} setCfHypo={setCfHypo}
            onCounterfactual={askCounterfactual}
            ovDecision={ovDecision} setOvDecision={setOvDecision}
            ovReason={ovReason} setOvReason={setOvReason}
            onOverride={submitOverride} overrides={overrides}
          />
        </Report>
      )}

      {showTiles && (
        <>
          <PanelTiles
            agents={agents}
            speaking={speaking}
            thinking={thinking}
            activePanel={activePanel}
            finished={finished}
          />
          <div className="thinkingbar">
            {thinking ? "the panel is deciding who asks next" : ""}
          </div>
          <Captions agentCap={agentCap} candCap={candCap} />
        </>
      )}

      {showRoom && !inInterview && (
        <LiveEvidence
          coverage={coverage}
          claims={claims}
          contradictions={contradictions}
        />
      )}

      {showRoom && !inInterview && (
        <div className="howto">
          <div className="howto-item">
            <b>Just speak — pauses are fine</b>
            The panel waits through your thinking pauses instead of cutting in.
            Press <b>Done</b> to send immediately if you want.
          </div>
          <div className="howto-item">
            <b>Talk again fixes a bad transcript</b>
            Misheard? Press Talk again to discard it and re-record.
          </div>
          <div className="howto-item">
            <b>Interrupt, and mic on/off</b>
            Talk over an interviewer to cut in (or use Interrupt); mute your mic
            anytime. Headphones avoid echo.
          </div>
        </div>
      )}

      <div className="logbar">
        <button className="ghost" onClick={() => setShowLog((s) => !s)}>
          <i className={"ph " + (showLog ? "ph-caret-down" : "ph-caret-right")} aria-hidden="true" />
          Session log
        </button>
      </div>
      {showLog && (
        <div className="log" ref={logRef}>
          {logLines.join("\n")}
        </div>
      )}

      {/* Mandatory candidate camera — self-view, always visible during the interview. */}
      <div className="selfview" hidden={!(joined && !talking)}>
        <div className="selfview-frame" ref={selfRef} />
        <span className="selfview-tag">
          <i className="ph-fill ph-video-camera" aria-hidden="true" /> You · camera on
        </span>
      </div>
    </div>
  );
}
