from typing import Optional

from pydantic import BaseModel, Field


class OpenAIEmbedderConfig(BaseModel):
    """OpenAI embedder configuration"""

    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    model: str = Field(default="text-embedding-ada-002", description="Embedding model")
    organization: Optional[str] = Field(default=None, description="Organization ID")

    class Config:
        env_prefix = "OPENAI_"
