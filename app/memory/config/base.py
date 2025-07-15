from abc import ABC

from pydantic import BaseModel, Field

from app.common.enum.persistence import PersistenceType

# ===== BASE CONFIGURATIONS =====


class BaseStoreConfig(BaseModel, ABC):
    """Base configuration for all store types"""

    timeout: int = Field(default=30, description="Connection timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    persistence: PersistenceType = Field(
        default=PersistenceType.DISK, description="Persistence mode"
    )

    class Config:
        extra = "allow"  # Allow additional fields for provider-specific configs
