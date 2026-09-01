"""End-of-turn detection with Smart Turn v3.1 (ONNX).

Silero VAD tells us *when the candidate paused*. That is not the same as *the
candidate is done* — people pause mid-thought ("I built a service that… "). Smart
Turn classifies the utterance-so-far as complete or incomplete, and the pipeline
only calls the LLM on a complete turn (ARCHITECTURE §4). This is what lets us run
a short VAD stop_secs without cutting people off.

The model is a Whisper-Tiny encoder + a shallow classifier head. Its input is
Whisper's 80-bin log-mel spectrogram over the last 8 s of audio, right-aligned
(the turn ends at the window's end); its `logits` output is already a sigmoid
probability of "complete". We reproduce the reference preprocessing exactly with
`transformers.WhisperFeatureExtractor` so the features match what the model was
trained on — a hand-rolled mel would risk subtle, silent misclassification.
Reference: github.com/pipecat-ai/smart-turn (inference.py, audio_utils.py).

If the weights or transformers are unavailable we degrade gracefully: every VAD
pause is treated as a complete turn — exactly the Phase-1 behaviour, so the
interview still runs. Get the weights with `python models/download_smart_turn.py`.
"""
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MAX_SECONDS = 8
MAX_SAMPLES = SAMPLE_RATE * MAX_SECONDS  # 128000 -> 800 mel frames


class SmartTurn:
    def __init__(self, model_path: str, threshold: float = 0.5):
        self.threshold = threshold
        self._session = None
        self._feature_extractor = None

        path = Path(model_path)
        if not path.is_absolute():
            # Config paths are relative to the repo root (this file lives in
            # media-worker/, so parents[1] is the repo root).
            path = Path(__file__).resolve().parents[1] / path

        if not path.exists():
            logger.warning(
                "Smart Turn model not found at %s — end-of-turn detection DISABLED "
                "(every VAD pause treated as a complete turn). "
                "Run `python models/download_smart_turn.py` to enable it.",
                path,
            )
            return

        try:
            import onnxruntime as ort
            from transformers import WhisperFeatureExtractor

            # chunk_length=8 → 8 s windows (800 frames), matching the export.
            self._feature_extractor = WhisperFeatureExtractor(chunk_length=MAX_SECONDS)
            self._session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            logger.info(
                "Smart Turn loaded from %s (input=%s output=%s)",
                path.name,
                self._session.get_inputs()[0].name,
                self._session.get_outputs()[0].name,
            )
        except Exception:
            logger.exception("Failed to load Smart Turn; disabling it")
            self._session = None
            self._feature_extractor = None

    @property
    def enabled(self) -> bool:
        return self._session is not None

    def is_complete(self, pcm_bytes: bytes) -> bool:
        """True if the utterance sounds finished (or if the model is disabled)."""
        if self._session is None:
            return True
        try:
            prob = self._infer(pcm_bytes)
        except Exception:
            logger.exception("Smart Turn inference failed; treating turn as complete")
            return True
        logger.info("Smart Turn p(complete)=%.3f (threshold %.2f)", prob, self.threshold)
        return prob >= self.threshold

    def _infer(self, pcm_bytes: bytes) -> float:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio = _truncate_to_last_n_seconds(audio, MAX_SAMPLES)

        inputs = self._feature_extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=MAX_SAMPLES,
            truncation=True,
            do_normalize=True,
        )
        input_features = inputs.input_features.squeeze(0).astype(np.float32)[None, ...]
        outputs = self._session.run(None, {"input_features": input_features})
        # Output is already a sigmoid probability of "complete".
        return float(np.asarray(outputs[0]).ravel()[0])


def _truncate_to_last_n_seconds(audio: np.ndarray, max_samples: int) -> np.ndarray:
    """Keep the most recent `max_samples`; left-pad shorter clips with zeros so
    the turn's end sits at the window's end (matches the reference)."""
    if len(audio) > max_samples:
        return audio[-max_samples:]
    if len(audio) < max_samples:
        return np.pad(audio, (max_samples - len(audio), 0), mode="constant")
    return audio
