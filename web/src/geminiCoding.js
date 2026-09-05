// Gemini Live coding round (§8): the browser streams the candidate's mic + entire
// screen to Gemini Live and plays Liam's voice back, all client-side. The server
// only mints the ephemeral token and receives the final verdict. Isolated from the
// Agora panel — this runs only during the locked coding phase.
import {
  GoogleGenAI, Modality, Type, ActivityHandling, StartSensitivity, EndSensitivity,
} from "@google/genai";

const FINISH_TOOL = {
  functionDeclarations: [
    {
      name: "finish_coding",
      description:
        "End the live coding round. Call this when the candidate has finished, " +
        "solved it, given up, or wants to move on (verdict 'done'), OR when they are " +
        "UNMISTAKABLY USING an AI assistant for this task — a real chat conversation is " +
        "the focused window with the problem/solution in it, or they're copying a " +
        "generated solution (verdict 'cheating'). Background AI tabs, shortcut tiles, " +
        "or the browser's own 'Ask Gemini'/'AI Mode' buttons are NOT cheating.",
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
    "Behave like a real engineer looking over their shoulder. Greet them and restate " +
    "the task in one short line. Then WATCH and coach, but describe ONLY what is " +
    "GENUINELY visible on their screen right now — never invent or assume progress. If " +
    "there is no code editor open, or no code written yet, do NOT pretend they're " +
    "coding; instead tell them to open an editor and start (e.g. 'I don't see an editor " +
    "yet — open one and start writing the function'). Only when you actually see code " +
    "should you react to it or nudge them. Keep every spoken reply to 1-2 short " +
    "sentences.\n\n" +
    "POLICY: they may use Google or official docs for SYNTAX. The core logic must be " +
    "their own.\n\n" +
    "CHEATING has a HIGH bar — it means they are ACTIVELY USING an AI assistant to get " +
    "the answer: an AI chat (ChatGPT, Claude, Copilot, Perplexity, Gemini) is the FOCUSED " +
    "foreground window with THIS coding problem or its solution typed/visible in it, or " +
    "they are clearly copying a generated solution into their editor. Be strict — do NOT " +
    "flag any of these normal things:\n" +
    "- AI-tool TABS merely open in the browser tab strip, or shortcut tiles / bookmarks " +
    "for Claude, Perplexity, ChatGPT on a new-tab page.\n" +
    "- The browser's own built-in buttons like 'Ask Gemini', 'AI Mode', or a Copilot " +
    "icon — those are part of the browser, not something the candidate is using.\n" +
    "- Their own code importing or calling AI libraries (openai, langchain, etc.).\n" +
    "- Other apps or tabs in the background.\n" +
    "Only flag when an AI assistant is UNMISTAKABLY IN USE for this task (a real chat " +
    "conversation about the problem is open and focused). If you are not sure it's being " +
    "used, do NOT flag — assume innocent.\n\n" +
    "Give them a few minutes to actually work. Two ways the round ends:\n" +
    "- If you clearly see them USING an AI assistant (as defined above): say EXACTLY " +
    "this kind of line and nothing else — 'I can see [ChatGPT/Claude] open there — that's " +
    "not allowed, so that's my read on this round; I'll hand it back to the panel.' — " +
    "then IMMEDIATELY call finish_coding with verdict 'cheating'.\n" +
    "- If they finish, solve it, or give up: briefly say how it went, then call " +
    "finish_coding with verdict 'done'.\n" +
    "The function call is what ends the round.\n\n" +
    "CRITICAL: say the closing line ONCE and call finish_coding right away. Do NOT keep " +
    "repeating yourself, and do NOT drift into unrelated commentary — one observation, " +
    "then the function call."
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
  let watchTimer = null;

  function stopInput() {
    if (frameTimer) clearInterval(frameTimer);
    if (watchTimer) clearInterval(watchTimer);
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
      // Don't let ambient sound cut Liam off mid-sentence (NO_INTERRUPTION), AND make
      // his speech detection LOW-sensitivity with a longer silence window so quiet
      // background voices / room noise don't register as the candidate talking and
      // trigger a reply. He still responds to clear, direct speech.
      realtimeInputConfig: {
        activityHandling: ActivityHandling.NO_INTERRUPTION,
        automaticActivityDetection: {
          startOfSpeechSensitivity: StartSensitivity.START_SENSITIVITY_LOW,
          endOfSpeechSensitivity: EndSensitivity.END_SENSITIVITY_LOW,
          prefixPaddingMs: 300,
          silenceDurationMs: 1000,
        },
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
          // Do NOT flush on `interrupted`. With NO_INTERRUPTION Liam shouldn't be cut
          // off, but Gemini still emits this signal whenever it hears the candidate or
          // background noise — flushing here chopped his audio to "one word then stops."
          // We let his current line finish; he replies on his next turn.
          if (msg.serverContent?.interrupted) log("interrupted signal (ignored — letting Liam finish)");
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
  }, 1500);
  log("screen frames streaming every 1.5s");

  // Proactive "watch check": while the candidate codes silently there's no speech
  // to trigger a turn, so Gemini wouldn't look at the screen. Nudge it gently for a
  // progress comment. NOTE: cheating handling lives ONLY in the system instruction —
  // re-asking about it every tick made Liam narrate "I see Claude" on a loop instead
  // of calling finish_coding. So this nudge is purely about progress.
  watchTimer = setInterval(() => {
    if (finished) return;
    try {
      session.sendClientContent({
        turns: [{
          role: "user",
          parts: [{
            text:
              "[watch check — I may be coding silently] Glance at my screen. If I have " +
              "code visible, react to what's ACTUALLY there or nudge me if I'm stuck. If " +
              "there's no editor or code yet, remind me to open one and start. One short " +
              "sentence — describe only what you truly see, never invent progress, and " +
              "don't repeat something you've already said.",
          }],
        }],
        turnComplete: true,
      });
    } catch { /* session may be closing */ }
  }, 15000);
  log("watch-check nudges every 15s");

  // Hard backstop against any loop: if the round never ends on its own, close it out
  // after CODING_MAX_MS so Liam can't narrate forever.
  const CODING_MAX_MS = 4 * 60 * 1000;
  setTimeout(() => { if (!finished) { log("coding round hit time cap — ending"); finish("done", "(coding round timed out)"); } }, CODING_MAX_MS);

  return { stop: () => finish("done", "(coding round ended)") };
}
