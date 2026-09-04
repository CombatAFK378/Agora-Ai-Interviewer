// Real audio amplitude, tapped without touching playback.
//
// Agora owns the playback path. We build a second MediaStream from the same
// track purely to analyse it, so nothing here can break what the candidate
// hears. Levels are written straight to a CSS custom property rather than to
// React state: this runs at animation frame rate and must never re-render.

function meterFor(track) {
  if (!track) return null;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    const ctx = new Ctx();
    const src = ctx.createMediaStreamSource(new MediaStream([track]));
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    analyser.smoothingTimeConstant = 0.65;
    src.connect(analyser);
    const data = new Uint8Array(analyser.fftSize);
    return {
      ctx,
      read() {
        analyser.getByteTimeDomainData(data);
        let peak = 0;
        for (let i = 0; i < data.length; i++) {
          const v = Math.abs(data[i] - 128) / 128;
          if (v > peak) peak = v;
        }
        return peak;
      },
      stop() {
        try { src.disconnect(); } catch { /* noop */ }
        try { ctx.close(); } catch { /* noop */ }
      },
    };
  } catch {
    return null; // no Web Audio, no meter. Everything else still works.
  }
}

/**
 * Drive `--level` (agent voice) and `--mic` (candidate voice) on `el`.
 * Returns a stop function. Safe to call with nulls.
 */
export function startLevelMeters({ el, remoteTrack, micTrack }) {
  const remote = meterFor(remoteTrack);
  const mic = meterFor(micTrack);
  if (!el || (!remote && !mic)) return () => {};

  let raf = 0;
  let lastR = 0;
  let lastM = 0;

  const tick = () => {
    // Ease downward so the ring falls off smoothly instead of flickering.
    const r = remote ? remote.read() : 0;
    const m = mic ? mic.read() : 0;
    lastR = r > lastR ? r : lastR * 0.86 + r * 0.14;
    lastM = m > lastM ? m : lastM * 0.86 + m * 0.14;
    el.style.setProperty("--level", lastR.toFixed(3));
    el.style.setProperty("--mic", lastM.toFixed(3));
    raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);

  return () => {
    cancelAnimationFrame(raf);
    if (remote) remote.stop();
    if (mic) mic.stop();
    el.style.removeProperty("--level");
    el.style.removeProperty("--mic");
  };
}

/**
 * Pan the panel's single audio track to match the speaker's seat.
 *
 * Agora sends one track for all five interviewers, so without this every voice
 * arrives from dead centre. Panning to the speaking seat is what makes five
 * voices read as five people around a table rather than one speaker changing
 * costume.
 *
 * IMPORTANT: this takes over playback. The caller must NOT also call
 * agoraTrack.play(), or the candidate hears everything twice. If this returns
 * null for any reason, the caller must fall back to Agora's own playback.
 */
export function createPanner(track) {
  if (!track) return null;
  try {
    const Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    const ctx = new Ctx();
    if (typeof ctx.createStereoPanner !== "function") { ctx.close(); return null; }

    const src = ctx.createMediaStreamSource(new MediaStream([track]));
    const panner = ctx.createStereoPanner();
    src.connect(panner).connect(ctx.destination);

    // A remote WebRTC track fed only into Web Audio can stop being pulled in
    // some browsers. A muted element sink keeps it flowing and costs nothing.
    let sink = null;
    try {
      sink = new Audio();
      sink.srcObject = new MediaStream([track]);
      sink.muted = true;
      sink.play().catch(() => {});
    } catch { /* noop */ }

    // The join click is our user gesture, but this graph is built several
    // awaits later inside an async callback, so the context can still come up
    // suspended. Resume is best effort; running() is how the caller finds out.
    if (ctx.state === "suspended") ctx.resume().catch(() => {});

    return {
      // Construction succeeding is NOT the same as audio flowing. The caller
      // must poll this and fall back to Agora playback if it stays false, or a
      // suspended context means a silent interview with no visible symptom.
      running: () => ctx.state === "running",
      // seat runs -1 (far left) to 1 (far right); kept well inside the extremes
      // so nobody ends up entirely in one ear.
      setSeat(seat) {
        const v = Math.max(-1, Math.min(1, seat)) * 0.6;
        try { panner.pan.setTargetAtTime(v, ctx.currentTime, 0.12); }
        catch { panner.pan.value = v; }
      },
      stop() {
        if (sink) {
          try { sink.pause(); sink.srcObject = null; } catch { /* noop */ }
          sink = null;
        }
        try { src.disconnect(); } catch { /* noop */ }
        try { ctx.close(); } catch { /* noop */ }
      },
    };
  } catch {
    return null;
  }
}
