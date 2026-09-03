"""Central configuration. Every service imports settings from here.

Values come from environment variables (loaded from a .env file in dev).
Keeping this in one place means we never scatter os.getenv() calls around.
"""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Agora
    agora_app_id: str = ""
    agora_app_certificate: str = ""

    # Speech providers
    sarvam_api_key: str = ""
    deepgram_api_key: str = ""

    # LLM. Every model call goes through the shared router in
    # shared/llm_router.py; never hardcode a model at a call site. Model ids may
    # carry a provider prefix ("groq:" / "openrouter:"); unprefixed ids default
    # to OpenRouter. Both providers are OpenAI-compatible.
    openrouter_api_key: str = ""
    groq_api_key: str = ""
    # Optional pool of Groq keys (comma-separated) from separate accounts. The
    # router round-robins across them so the per-account tokens-per-minute limit
    # is shared out — same model/quality, ~N× the free budget. Falls back to the
    # single groq_api_key if unset.
    groq_api_keys: str = ""
    llm_fast_model: str = "groq:openai/gpt-oss-120b"      # candidate is waiting
    llm_reasoning_model: str = "openrouter:minimax/minimax-m3:free"
    llm_fallback_chain: str = ""                          # comma-separated ids
    # Coding round (§8) vision models — gpt-oss is text-only, so these must be
    # image-capable. Two tiers: a FAST one for live screen reads mid-interview, and
    # a bigger one for the scoring pass at conclude (latency-insensitive). Both
    # verified free + vision-capable on OpenRouter (Gemma 4, no reasoning overhead).
    # No fallback chain (text models can't see) — coding degrades to verbal on fail.
    llm_vision_fast: str = "openrouter:google/gemma-4-26b-a4b-it:free"
    llm_vision_model: str = "openrouter:google/gemma-4-31b-it:free"
    # Gemini Live (§8): realtime screen-watch + voice for the coding round, run in
    # the browser via a server-minted ephemeral token. If the key is unset the
    # coding round falls back to the snapshot-vision path above.
    gemini_api_key: str = ""
    gemini_live_model: str = "gemini-3.1-flash-live-preview"
    # Free vision tiers rate-limit (429) often, so a vision read falls through this
    # chain before giving up (all verified free + image-capable on OpenRouter).
    llm_vision_fallbacks: str = ("openrouter:google/gemma-4-31b-it:free,"
                                 "openrouter:minimax/minimax-m3:free")

    # Smart Turn v3.1 end-of-turn detection
    smart_turn_model_path: str = "models/smart-turn-v3.1.onnx"
    smart_turn_threshold: float = 0.5

    # TTS: streaming (Deepgram WebSocket) for low first-audio latency (~1.3s). The
    # socket is occasionally slow to connect; we fail fast to REST on those, so
    # streaming stays the default and a bad moment just degrades, never stalls.
    tts_use_streaming: bool = True

    # Barge-in (interrupt the agent by talking over it). Needs the candidate on
    # headphones — otherwise the agent's own voice echoes into the mic and
    # false-triggers it. Set false for robust half-duplex on speakers: the agent
    # can't be interrupted, but echo can't derail the interview either.
    allow_bargein: bool = True

    # Floor control (ARCHITECTURE §5). λ (coverage weight) ramps from start→end
    # across the interview so late questions chase what we still don't know.
    interview_time_budget_s: int = 1200
    coverage_lambda_start: float = 0.5
    coverage_lambda_end: float = 1.5

    # Conviction (ARCHITECTURE §6): STRONG if |score-0.5| > margin AND
    # evidence_count >= min_evidence, else NEUTRAL. Computed in code, not asked.
    conviction_margin: float = 0.25
    conviction_min_evidence: int = 3

    # Phase 1 tunables
    vad_stop_secs: float = 0.2
    audio_sample_rate: int = 16000
    media_worker_port: int = 8080

    # Phase 8: where locked interviews are persisted (one JSON per interview), so
    # the recruiter dashboard can list past candidates and rejoin. Mounted as a
    # volume in docker-compose so it survives container rebuilds.
    data_dir: str = "/app/data"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def fallback_models(self) -> list[str]:
        """Parsed fallback chain: primary tried first, then these in order."""
        return [m.strip() for m in self.llm_fallback_chain.split(",") if m.strip()]

    @property
    def vision_fallbacks(self) -> list[str]:
        """Extra vision models tried (in order) if the primary vision model fails."""
        return [m.strip() for m in self.llm_vision_fallbacks.split(",") if m.strip()]

    @property
    def groq_keys(self) -> list[str]:
        """All Groq keys to round-robin over (pool, else the single key)."""
        pool = [k.strip() for k in self.groq_api_keys.split(",") if k.strip()]
        if pool:
            return pool
        return [self.groq_api_key] if self.groq_api_key else []


@lru_cache
def get_settings() -> Settings:
    """Cached so we parse the environment only once per process."""
    return Settings()
