import os
from typing import Optional

from pydantic import Field, field_validator

from app.memory.config.base import BaseStoreConfig


class TimescaleDBConfig(BaseStoreConfig):
    """TimescaleDB relational store configuration"""

    connection_string: str = Field(
        default_factory=lambda: os.getenv(
            "TIMESCALEDB_CONNECTION_STRING", "postgresql://localhost:5432/memory_system"
        ),
        description="TimescaleDB connection string",
    )

    database_name: str = Field(
        default_factory=lambda: os.getenv("TIMESCALEDB_DATABASE", "memory_system"),
        description="Database name",
    )

    table_name: str = Field(
        default_factory=lambda: os.getenv("TIMESCALEDB_TABLE", "memories"),
        description="Table name for memories",
    )

    timeout_seconds: int = Field(
        default=30, description="Connection timeout in seconds"
    )

    max_pool_size: int = Field(default=20, description="Maximum connection pool size")

    enable_ssl: bool = Field(default=True, description="Enable SSL connection")

    @field_validator("connection_string")
    @classmethod
    def validate_connection_string(cls, v: str) -> str:
        """Validate TimescaleDB connection string format"""
        if not v.startswith("postgresql://"):
            raise ValueError(
                "TimescaleDB connection string must start with 'postgresql://'"
            )
        return v
