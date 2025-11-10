from pathlib import Path
from typing import List

from pydantic import AnyUrl, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # MongoDB connection
    mongo_url: AnyUrl = Field(..., env="MONGO_URL")
    db_name: str = Field(..., env="DB_NAME")
    collection_name: str = Field(..., env="COLLECTION_NAME")
    bm25_index_name: str = Field(..., env="BM25_INDEX_NAME")

    # Search tuning
    search_fields: List[str] = Field(default_factory=lambda: ["text", "summary", "content"], env="SEARCH_FIELDS")
    default_limit: int = Field(10, env="DEFAULT_LIMIT")
    max_limit: int = Field(100, env="MAX_LIMIT")

    # Mongo connection pool tuning
    mongo_max_pool_size: int = Field(50, env="MONGO_MAX_POOL_SIZE")
    mongo_server_selection_timeout_ms: int = Field(5000, env="MONGO_SERVER_SELECTION_TIMEOUT_MS")
    langchain_timeout_seconds: float = Field(2.0, env="LANGCHAIN_TIMEOUT_SECONDS")

    # Vector search / embedding settings
    vector_index_name: str = Field(None, env="VECTOR_INDEX_NAME")
    embedding_api_base: str = Field(None, env="EMBEDDING_API_BASE")
    embedding_model: str = Field("text-embedding-3-small", env="EMBEDDING_MODEL")
    embedding_user_email: str = Field(None, env="EMBEDDING_USER_EMAIL")
    embedding_auth_token: str = Field(None, env="EMBEDDING_AUTH_TOKEN")

    # Groq re-ranking
    groq_api_base: str = Field("https://api.groq.com/openai/v1", env="GROQ_API_BASE")
    groq_api_key: str = Field(None, env="GROQ_API_KEY")
    groq_model: str = Field(None, env="GROQ_MODEL")


settings = Settings()
