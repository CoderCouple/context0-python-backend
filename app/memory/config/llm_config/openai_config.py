from typing import Optional

from pydantic import BaseModel, Field


class OpenAIConfig(BaseModel):
    """OpenAI LLM configuration"""

    api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    model: str = Field(default="gpt-4o-mini", description="Model name")
    temperature: float = Field(default=0.1, description="Temperature for generation")
    max_tokens: int = Field(default=1000, description="Maximum tokens")
    organization: Optional[str] = Field(default=None, description="Organization ID")

    class Config:
        env_prefix = "OPENAI_"
