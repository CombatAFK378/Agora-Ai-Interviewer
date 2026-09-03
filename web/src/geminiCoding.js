// Gemini Live coding round (§8): the browser streams the candidate's mic + entire
// screen to Gemini Live and plays Liam's voice back, all client-side. The server
// only mints the ephemeral token and receives the final verdict. Isolated from the
// Agora panel — this runs only during the locked coding phase.
import { GoogleGenAI, Modality, Type, ActivityHandling } from "@google/genai";

const FINISH_TOOL = {
  functionDeclarations: [
    {
      name: "finish_coding",
      description:
        "End the live coding round. Call this when the candidate has finished, " +
        "solved it, given up, or wants to move on (verdict 'done'), OR when you " +
        "actually SEE an AI assistant/chatbot open on their screen or a full " +
        "solution pasted in (verdict 'cheating'). Do not call it in the first " +
        "couple of minutes unless there is clear cheating.",
      parameters: {
        type: Type.OBJECT,
        properties: {
          verdict: {
            type: Type.STRING,
            enum: ["done", "cheating"],
            description: "'done' or 'cheating'",
          },
          summary: {
            type: Type.STRING,
            description:
              "2-3 sentences: what they built, correctness, approach, and code " +
              "quality — or, if cheating, what you saw.",
          },
        },
        required: ["verdict", "summary"],
      },
    },
  ],
};

function systemInstruction(task) {
  return (
    "You are Liam, a warm but sharp coding interviewer on an AI hiring panel. You " +
    "are running ONE live coding exercise and you can SEE the candidate's screen and " +
    "HEAR them speak.\n\n" +
    `THE TASK: ${task}\n\n` +
    "Behave like a real engineer looking over their shoulder. Greet them, restate " +
    "the task in one short line, then WATCH and coach: react to what they actually " +
    "type ('nice, you set up the hashmap'), give a small nudge if they're stuck, and " +
    "answer their questions. Keep every spoken reply to 1-2 short sentences.\n\n" +
    "POLICY: they may use Google or official docs for SYNTAX, but NOT AI assistants " +
    "(ChatGPT, Claude, Copilot, Gemini, Perplexity), and the core logic must be their " +
    "own. Their own code that imports or calls AI libraries (openai, langchain, etc.) " +
    "is NORMAL work — that is NOT cheating.\n\n" +
    "Give them a few minutes to actually work. When the round is genuinely over, or if " +
    "you clearly see an AI assistant open on their screen, FIRST say your closing line " +
    "OUT LOUD (e.g. 'I can see ChatGPT open there — that's not allowed for the logic, so " +
    "that's my read on this round; I'll hand back to the panel.'), and THEN call the " +
    "finish_coding function with the verdict and a short summary. Never end silently — " +
    "always speak before you call the function."
  );
}

function b64ToInt16(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Int16Array(bytes.buffer);
}

function floatTo16BitPCM(float32) {
  const out = new Int16Array(float32.length);
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]));
    out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return out;
}

function int16ToBase64(int16) {
  const bytes = new Uint8Array(int16.buffer);
  let bin = "";
  for (let i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
  return btoa(bin);
}

function downsample(input, inRate, outRate) {
  if (outRate >= inRate) return input;
  const ratio = inRate / outRate;
  const outLen = Math.floor(input.length / ratio);
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) out[i] = input[Math.floor(i * ratio)];
  return out;
}

// Start a Gemini Live coding session. Returns { stop() }.
export async function startGeminiCoding({
  token,
  model,
  task,
  screenStream,
  onStatus,
  onSpeaking,
  onFinish,
  log = () => {},
}) {
  const ai = new GoogleGenAI({ apiKey: token, httpOptions: { apiVersion: "v1alpha" } });
  let msgCount = 0, audioChunks = 0, micChunks = 0, frameCount = 0;

  // --- audio playback (Gemini streams 24kHz PCM) ---
  const outCtx = new (window.AudioContext || window.webkitAudioContext)();
  let nextStart = 0;
  const sources = new Set();
  function playPcm(b64) {
    if (ended) return;
    const pcm = b64ToInt16(b64);
    const f32 = new Float32Array(pcm.length);
    for (let i = 0; i < pcm.length; i++) f32[i] = pcm[i] / 32768;
    const buf = outCtx.createBuffer(1, f32.length, 24000); // context resamples
    buf.getChannelData(0).set(f32);
    const src = outCtx.createBufferSource();
    src.buffer = buf;
    src.connect(outCtx.destination);
    const start = Math.max(outCtx.currentTime, nextStart);
    src.start(start);
    nextStart = start + buf.duration;
    sources.add(src);
    src.onended = () => sources.delete(src);
  }
  function stopPlayback() {
    for (const s of sources) { try { s.stop(); } catch { /* already stopped */ } }
    sources.clear();
    nextStart = 0;
  }

  let finished = false;   // stop sending input
  let ended = false;      // fully torn down + reported
  let session = null;
  let micStream = null;
  let inCtx = null;
  let proc = null;
  let frameTimer = null;

  function stopInput() {
    if (frameTimer) clearInterval(frameTimer);
    try { proc && proc.disconnect(); } catch { /* noop */ }
    try { micStream && micStream.getTracks().forEach((t) => t.stop()); } catch { /* noop */ }
    try { inCtx && inCtx.close(); } catch { /* noop */ }
  }
  function teardown(verdict, summary) {
    if (ended) return;
    ended = true;
    stopPlayback();
    try { outCtx.close(); } catch { /* noop */ }
    try { session && session.close(); } catch { /* noop */ }
    onFinish && onFinish(verdict, summary);
  }
  // Stop capturing immediately, but let any final spoken line (the verdict) play
  // out before we tear down and report — so the flag isn't chopped off mid-word.
  function finish(verdict, summary) {
    if (finished) return;
    finished = true;
    stopInput();
    const t0 = Date.now();
    const drain = () => {
      const remaining = nextStart - outCtx.currentTime; // seconds of audio still queued
      if (remaining <= 0.15 || Date.now() - t0 > 9000) teardown(verdict, summary);
      else setTimeout(drain, 250);
    };
    setTimeout(drain, 400);
  }

  session = await ai.live.connect({
    model,
    config: {
      responseModalities: [Modality.AUDIO],
      systemInstruction: systemInstruction(task),
      tools: [FINISH_TOOL],
      // Liam's voice — a male prebuilt voice so he doesn't default to female.
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: "Charon" } },
      },
      // Don't let ambient sound (your voice, thinking out loud, echo) cut Liam off
      // mid-sentence. He still HEARS you — he just finishes his short remark before
      // responding, instead of flushing his audio on every noise. Default turn
      // detection otherwise, so he's still responsive when he's not speaking.
      realtimeInputConfig: {
        activityHandling: ActivityHandling.NO_INTERRUPTION,
      },
    },
    callbacks: {
      onopen: () => log("session open"),
      onmessage: (msg) => {
        try {
          msgCount += 1;
          // Play audio from EXACTLY ONE source. The SDK exposes the same PCM both
          // as msg.data and inside serverContent.modelTurn.parts[].inlineData —
          // playing both doubled the voice and made it lag. Prefer msg.data.
          if (msg.data) {
            audioChunks += 1;
            onSpeaking && onSpeaking(true);
            playPcm(msg.data);
          } else {
            const parts = msg.serverContent?.modelTurn?.parts || [];
            for (const p of parts) {
              if (p.inlineData?.data) {
                audioChunks += 1;
                onSpeaking && onSpeaking(true);
                playPcm(p.inlineData.data);
              }
              if (p.text) log("liam(text): " + p.text.slice(0, 80));
            }
          }
          if (msg.serverContent?.interrupted) { log("interrupted → flush queue"); stopPlayback(); }
          if (msg.serverContent?.turnComplete) {
            log(`turn complete (msgs=${msgCount}, audio=${audioChunks}, mic=${micChunks}, frames=${frameCount})`);
            onSpeaking && onSpeaking(false);
          }
        } catch (e) {
          log("onmessage error: " + (e?.message || e));
        }
        // tool call → end the round
        const calls = msg.toolCall?.functionCalls || [];
        for (const c of calls) {
          if (c.name === "finish_coding") {
            const verdict = c.args?.verdict || "done";
            const summary = c.args?.summary || "";
            log(`finish_coding(${verdict})`);
            try {
              session.sendToolResponse({
                functionResponses: [{ id: c.id, name: c.name, response: { ok: true } }],
              });
            } catch { /* noop */ }
            finish(verdict, summary);
          }
        }
      },
      onerror: (e) => { log("session error: " + (e?.message || e)); finish("error", String(e?.message || e)); },
      onclose: (e) => { log("session closed: " + (e?.reason || "")); if (!finished) finish("error", "connection closed"); },
    },
  });
  log("connected; sending greeting trigger");

  // Make Liam speak first (system instruction has the task).
  session.sendClientContent({
    turns: [
      {
        role: "user",
        parts: [
          {
            text:
              "Start the coding round now: greet me by voice, restate the task in one " +
              "sentence, and then watch my screen and coach me as I code.",
          },
        ],
      },
    ],
    turnComplete: true,
  });

  // --- mic capture → 16kHz PCM ---
  micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
  });
  inCtx = new (window.AudioContext || window.webkitAudioContext)();
  const srcNode = inCtx.createMediaStreamSource(micStream);
  proc = inCtx.createScriptProcessor(4096, 1, 1);
  srcNode.connect(proc);
  // Route through a MUTED gain node: the processor must be connected to keep
  // firing, but we must NOT play the mic back to the speakers (that was the echo).
  const sink = inCtx.createGain();
  sink.gain.value = 0;
  proc.connect(sink);
  sink.connect(inCtx.destination);
  log(`mic capturing at ${inCtx.sampleRate}Hz → 16kHz`);
  proc.onaudioprocess = (e) => {
    if (finished) return;
    const input = e.inputBuffer.getChannelData(0);
    const down = downsample(input, inCtx.sampleRate, 16000);
    const b64 = int16ToBase64(floatTo16BitPCM(down));
    try {
      session.sendRealtimeInput({ audio: { data: b64, mimeType: "audio/pcm;rate=16000" } });
      micChunks += 1;
      if (micChunks === 1) log("first mic chunk sent");
    } catch { /* session may be closing */ }
  };

  // --- screen frames → Gemini (every 2s) ---
  const video = document.createElement("video");
  video.srcObject = screenStream;
  video.muted = true;
  await video.play();
  const canvas = document.createElement("canvas");
  frameTimer = setInterval(() => {
    if (finished) return;
    const w = video.videoWidth, h = video.videoHeight;
    if (!w || !h) return;
    // Keep frames light so they don't congest the audio stream (which makes
    // replies feel laggy): smaller + more compressed, every 3s.
    const scale = Math.min(1, 1024 / w);
    canvas.width = Math.round(w * scale);
    canvas.height = Math.round(h * scale);
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    const b64 = canvas.toDataURL("image/jpeg", 0.4).split(",")[1];
    try {
      session.sendRealtimeInput({ video: { data: b64, mimeType: "image/jpeg" } });
      frameCount += 1;
      if (frameCount === 1) log(`first screen frame sent (${canvas.width}x${canvas.height})`);
    } catch { /* session may be closing */ }
  }, 3000);
  log("screen frames streaming every 3s");

  return { stop: () => finish("done", "(coding round ended)") };
}
