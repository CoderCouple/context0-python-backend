from typing import Optional

from pydantic import Field

from app.memory.config.base import BaseStoreConfig


class PineconeConfig(BaseStoreConfig):
    """Pinecone vector store configuration"""

    api_key: Optional[str] = Field(default=None, description="Pinecone API key")
    environment: Optional[str] = Field(default=None, description="Pinecone environment")
    index_name: str = Field(default="memories", description="Index name")
    metric: str = Field(default="cosine", description="Distance metric")
    pods: int = Field(default=1, description="Number of pods")
    dimension: int = Field(default=1536, description="Vector dimensions")

    class Config:
        env_prefix = "PINECONE_"
