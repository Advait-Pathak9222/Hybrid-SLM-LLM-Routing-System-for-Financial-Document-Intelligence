"""Application configuration loaded from environment variables."""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for the finance pipeline.

    Values are loaded from a `.env` file and can be overridden by
    real environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # --- Ollama (local SLM) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "phi3:mini"

    # --- Groq (cloud LLM) ---
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # --- Routing ---
    confidence_threshold: float = 0.6

    # --- Timeouts ---
    slm_timeout_sec: int = 30
    llm_timeout_sec: int = 60

    # --- Cache ---
    cache_max_size: int = 128
    cache_ttl_seconds: int = 300

    # --- RAG / ChromaDB ---
    chroma_persist_dir: str = "./chroma_data"
    rag_default_top_k: int = 3
    rag_enabled: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance.

    Using ``lru_cache`` ensures the `.env` file is read only once
    per process lifetime.
    """
    return Settings()
