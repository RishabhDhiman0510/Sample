"""Configuration management using Pydantic BaseSettings."""

from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Model paths
    MODEL_ADAPTER_PATH: str = Field(
        default="./apollo2b-medical-qa-final",
        description="Path to PEFT adapter"
    )
    BASE_MODEL_PATH: Optional[str] = Field(
        default=None,
        description="Override base model path"
    )

    # RAG library
    RAG_LIBRARY_DIR: str = Field(
        default="./medical_rag_library",
        description="Directory containing RAG documents and indices"
    )
    RAG_LIBRARY_NAME: str = Field(
        default="medical_rag",
        description="Name prefix for RAG files"
    )

    # Feedback storage
    FEEDBACK_FILE: str = Field(
        default="./medical_rag_library/user_feedback.json",
        description="Path to feedback JSON file"
    )
    FEEDBACK_ENCRYPTION_KEY: Optional[str] = Field(
        default=None,
        description="Fernet encryption key for feedback (base64)"
    )

    # Embedding models
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    RERANKER_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking"
    )

    # Retrieval parameters
    TOP_K_RETRIEVAL: int = Field(default=10, ge=1, le=50)
    TOP_K_RERANK: int = Field(default=5, ge=1, le=20)
    MIN_SIMILARITY: float = Field(default=0.20, ge=0.0, le=1.0)
    CHUNK_SIZE: int = Field(default=300, ge=100, le=1000)
    CHUNK_OVERLAP: int = Field(default=150, ge=0, le=500)

    # Generation parameters
    MAX_NEW_TOKENS: int = Field(default=120, ge=10, le=500)
    TEMPERATURE: float = Field(default=0.6, ge=0.0, le=2.0)
    TOP_P: float = Field(default=0.9, ge=0.0, le=1.0)
    REPETITION_PENALTY: float = Field(default=1.3, ge=1.0, le=2.0)

    # Model loading
    LOAD_IN_8BIT: bool = Field(default=True)
    DEVICE_MAP: str = Field(default="auto")

    # Web search
    ENABLE_WEB_SEARCH: bool = Field(default=True)
    PUBMED_EMAIL: Optional[str] = Field(default=None)
    WEB_SEARCH_TIMEOUT: int = Field(default=5, ge=1, le=30)
    WEB_SEARCH_MAX_RETRIES: int = Field(default=3, ge=1, le=10)

    # Redis cache (optional)
    REDIS_URL: Optional[str] = Field(
        default=None,
        description="Redis URL for caching (e.g., redis://localhost:6379/0)"
    )
    CACHE_TTL: int = Field(default=3600, ge=60)

    # API server
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000, ge=1024, le=65535)
    API_WORKERS: int = Field(default=1, ge=1, le=8)
    LOG_LEVEL: str = Field(default="INFO")

    # Observability
    ENABLE_METRICS: bool = Field(default=True)
    METRICS_PORT: int = Field(default=9090, ge=1024, le=65535)

    # PII detection
    ENABLE_PII_DETECTION: bool = Field(default=True)
    ENABLE_PRESIDIO: bool = Field(
        default=False,
        description="Use Presidio for advanced PII detection"
    )


settings = Settings()
