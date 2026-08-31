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

    # Phase 1 tunables
    vad_stop_secs: float = 0.6
    audio_sample_rate: int = 16000
    media_worker_port: int = 8080

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Cached so we parse the environment only once per process."""
    return Settings()
