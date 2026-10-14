"""Application settings loaded from environment variables."""

import os

from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Settings used by the API and database service."""

    mongodb_uri: str = os.getenv("MONGODB_URI", "")
    mongodb_database: str = os.getenv("MONGODB_DATABASE", "")
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )


settings = Settings()
