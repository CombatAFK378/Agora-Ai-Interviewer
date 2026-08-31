"""Streaming voice-activity detection with Silero VAD.

Audio arrives from Agora in tiny chunks (~10 ms each). This class buffers them
into the 512-sample (32 ms) windows Silero expects, asks the model "is this
speech?" for each window, and runs a small state machine:

    silence -> (speech detected) -> collecting -> (long enough pause) -> DONE

When a pause longer than `stop_secs` follows speech, `process()` returns the full
PCM of everything the person just said (one utterance). Otherwise it returns None
and keeps listening.

We keep a short pre-roll so the first syllable isn't clipped.
"""
from collections import deque
from typing import Optional

import numpy as np
import torch
from silero_vad import load_silero_vad

# Silero at 16 kHz requires exactly 512-sample windows.
WINDOW_SAMPLES = 512
WINDOW_BYTES = WINDOW_SAMPLES * 2          # int16 => 2 bytes/sample
WINDOW_MS = WINDOW_SAMPLES / 16000 * 1000  # 32 ms


class SileroVAD:
    def __init__(
        self,
        sample_rate: int = 16000,
        stop_secs: float = 0.6,
        threshold: float = 0.5,
        preroll_ms: float = 200.0,
    ):
        if sample_rate != 16000:
            raise ValueError("This VAD wrapper assumes 16 kHz audio.")
        self.model = load_silero_vad(onnx=True)
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.stop_windows = max(1, int(stop_secs * 1000 / WINDOW_MS))
        self.preroll_windows = max(0, int(preroll_ms / WINDOW_MS))

        self._leftover = bytearray()          # bytes not yet forming a full window
        self._preroll = deque(maxlen=self.preroll_windows)
        self._speech = bytearray()            # accumulated current utterance
        self._triggered = False
        self._silence_run = 0                 # consecutive non-speech windows

    def process(self, pcm_bytes: bytes) -> Optional[bytes]:
        """Feed one chunk of PCM. Returns a completed utterance's PCM, or None."""
        self._leftover.extend(pcm_bytes)
        result: Optional[bytes] = None

        while len(self._leftover) >= WINDOW_BYTES:
            window = bytes(self._leftover[:WINDOW_BYTES])
            del self._leftover[:WINDOW_BYTES]
            finished = self._feed_window(window)
            if finished is not None:
                # If two utterances complete in one call (rare), the last wins;
                # in practice a chunk is ~10 ms so this holds one utterance.
                result = finished
        return result

    def _is_speech(self, window: bytes) -> bool:
        samples = np.frombuffer(window, dtype=np.int16).astype(np.float32) / 32768.0
        prob = self.model(torch.from_numpy(samples), self.sample_rate).item()
        return prob >= self.threshold

    def _feed_window(self, window: bytes) -> Optional[bytes]:
        speech = self._is_speech(window)

        if not self._triggered:
            self._preroll.append(window)
            if speech:
                # Start a new utterance, including the buffered pre-roll.
                self._triggered = True
                self._silence_run = 0
                self._speech = bytearray()
                for w in self._preroll:
                    self._speech.extend(w)
                self._preroll.clear()
            return None

        # Currently inside an utterance.
        self._speech.extend(window)
        if speech:
            self._silence_run = 0
        else:
            self._silence_run += 1
            if self._silence_run >= self.stop_windows:
                return self._finalize()
        return None

    def _finalize(self) -> bytes:
        utterance = bytes(self._speech)
        self._speech = bytearray()
        self._triggered = False
        self._silence_run = 0
        self.model.reset_states()  # clear LSTM state between utterances
        return utterance
