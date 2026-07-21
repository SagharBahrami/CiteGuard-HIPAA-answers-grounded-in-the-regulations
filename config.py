from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    openai_api_key: str
    generation_model: str = "gpt-5.6-terra"
    guardrail_model: str = "gpt-5.6-luna"
    embedding_model: str = "text-embedding-3-small"
    redis_url: str = "redis://localhost:6379/0"
    chroma_dir: str = "./chroma_db"
    chroma_collection: str = "hipaa_regs"
    similarity_threshold: float = 0.35
    hipaa_source_url: str


settings = Settings()
