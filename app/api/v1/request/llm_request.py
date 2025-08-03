"""LLM API request models"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from app.llm.types import CategoryConfig


class CreateLLMPresetRequest(BaseModel):
    """Request to create a new LLM preset"""

    name: str = Field(..., description="Preset name", min_length=1, max_length=100)
    description: str = Field(..., description="Preset description", max_length=500)

    # Provider settings
    provider: str = Field(..., description="LLM provider (openai, anthropic, google)")
    model: str = Field(..., description="Model to use")

    # Generation parameters
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32000)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1, le=100)

    # Prompts
    system_prompt: str = Field(default="You are a helpful assistant.", max_length=2000)
    custom_instructions: Optional[str] = Field(default=None, max_length=2000)

    # Memory settings
    use_memory_context: bool = Field(default=True)
    extract_memories: bool = Field(default=True)
    memory_threshold: float = Field(default=0.3, ge=0.0, le=1.0)
    force_add_only: bool = Field(default=False)
    memory_extraction_types: List[str] = Field(
        default_factory=lambda: ["semantic_memory", "episodic_memory"]
    )

    # Reranking
    reranking_enabled: bool = Field(default=True)
    rerank_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # Categories
    categories: List[CategoryConfig] = Field(default_factory=list)

    # Additional settings
    conversation_history_limit: int = Field(default=10, ge=1, le=50)
    include_timestamps: bool = Field(default=False)
    response_format_preference: Optional[str] = Field(default=None)

    # Cost controls
    daily_token_limit: Optional[int] = Field(default=None, ge=1)
    monthly_budget: Optional[float] = Field(default=None, ge=0.01)

    # Sharing
    is_shared: bool = Field(
        default=False, description="Allow other users to use this preset"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for organization")


class UpdateLLMPresetRequest(BaseModel):
    """Request to update an LLM preset"""

    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)

    # Provider settings (can't change provider/model after creation)
    # provider: str - Not allowed to change
    # model: str - Not allowed to change

    # Generation parameters
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=32000)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=100)

    # Prompts
    system_prompt: Optional[str] = Field(None, max_length=2000)
    custom_instructions: Optional[str] = Field(None, max_length=2000)

    # Memory settings
    use_memory_context: Optional[bool] = None
    extract_memories: Optional[bool] = None
    memory_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)
    force_add_only: Optional[bool] = None
    memory_extraction_types: Optional[List[str]] = None

    # Reranking
    reranking_enabled: Optional[bool] = None
    rerank_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)

    # Categories
    categories: Optional[List[CategoryConfig]] = None

    # Additional settings
    conversation_history_limit: Optional[int] = Field(None, ge=1, le=50)
    include_timestamps: Optional[bool] = None
    response_format_preference: Optional[str] = None

    # Cost controls
    daily_token_limit: Optional[int] = Field(None, ge=1)
    monthly_budget: Optional[float] = Field(None, ge=0.01)

    # Sharing
    is_shared: Optional[bool] = None
    tags: Optional[List[str]] = None


class TestLLMPresetRequest(BaseModel):
    """Request to test an LLM preset"""

    message: str = Field(..., description="Test message", min_length=1, max_length=1000)
    use_memory_context: bool = Field(
        default=False, description="Include memory context in test"
    )
    include_system_prompt: bool = Field(
        default=True, description="Include system prompt in response"
    )

    # Optional overrides for testing
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4000)


class SetDefaultPresetRequest(BaseModel):
    """Request to set a preset as default"""

    preset_id: str = Field(..., description="Preset ID to set as default")
