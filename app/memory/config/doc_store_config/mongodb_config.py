import os
from typing import Optional

from pydantic import Field, field_validator

from app.memory.config.base import BaseStoreConfig


class MongoDBConfig(BaseStoreConfig):
    """MongoDB document store configuration"""

    connection_string: str = Field(
        default_factory=lambda: os.getenv(
            "MONGODB_CONNECTION_STRING", "mongodb://localhost:27017"
        ),
        description="MongoDB connection string",
    )

    database_name: str = Field(
        default_factory=lambda: os.getenv("MONGODB_DATABASE", "memory_system"),
        description="Database name",
    )

    collection_name: str = Field(
        default_factory=lambda: os.getenv("MONGODB_COLLECTION", "memories"),
        description="Collection name for memories",
    )

    timeout_seconds: int = Field(
        default=30, description="Connection timeout in seconds"
    )

    max_pool_size: int = Field(default=100, description="Maximum connection pool size")

    @field_validator("connection_string")
    @classmethod
    def validate_connection_string(cls, v: str) -> str:
        """Validate MongoDB connection string format"""
        if not v.startswith(("mongodb://", "mongodb+srv://")):
            raise ValueError(
                "MongoDB connection string must start with 'mongodb://' or 'mongodb+srv://'"
            )
        return v
