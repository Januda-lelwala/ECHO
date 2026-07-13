from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    REDIS_URL: str = "redis://localhost:6379/0"
    MODEL_DEVICE: str = "auto"
    ALLOWED_ORIGINS: str = ""
    SESSION_COOKIE_NAME: str = "sid"
    SESSION_TTL_SECONDS: int = 24 * 60 * 60
    COOKIE_SECURE: bool = False
    COOKIE_SAMESITE: str = "lax"  # use "none" on cross-site + https
    COOKIE_DOMAIN: str | None = None

settings = Settings()
