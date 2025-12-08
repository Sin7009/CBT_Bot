from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    TELEGRAM_TOKEN: str
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str = "https://openrouter.ai/api/v1"
    REDIS_URL: str = "redis://redis:6379/0"

    # Models
    MODEL_THERAPIST: str = "google/gemini-2.5-flash"
    MODEL_SUPERVISOR: str = "deepseek/deepseek-v3.2-speciale"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
