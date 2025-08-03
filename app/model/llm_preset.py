"""Database models for LLM presets and configurations"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from app.llm.types import CategoryConfig


class LLMPreset(BaseModel):
    """Database model for LLM presets"""

    # Identity
    id: str = Field(..., description="Unique preset ID")
    user_id: str = Field(..., description="Owner user ID")
    name: str = Field(..., description="Preset name")
    description: str = Field(..., description="Preset description")

    # Provider settings
    provider: str = Field(..., description="LLM provider (openai, anthropic, google)")
    model: str = Field(..., description="Model identifier")

    # Generation parameters
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1)

    # Prompts
    system_prompt: str = Field(default="You are a helpful assistant.")
    custom_instructions: Optional[str] = Field(default=None)

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
    conversation_history_limit: int = Field(default=10)
    include_timestamps: bool = Field(default=False)
    response_format_preference: Optional[str] = Field(default=None)

    # Cost controls
    daily_token_limit: Optional[int] = Field(default=None)
    monthly_budget: Optional[float] = Field(default=None)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_default: bool = Field(
        default=False, description="Is this the user's default preset"
    )
    is_shared: bool = Field(
        default=False, description="Can other users use this preset"
    )
    tags: List[str] = Field(default_factory=list)

    # Usage statistics
    usage_count: int = Field(default=0)
    last_used_at: Optional[datetime] = Field(default=None)
    total_tokens_used: int = Field(default=0)
    total_cost: float = Field(default=0.0)

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}


class PromptConfiguration(BaseModel):
    """Database model for custom prompt configurations"""

    id: str = Field(..., description="Unique configuration ID")
    user_id: str = Field(..., description="Owner user ID")
    name: str = Field(..., description="Configuration name")
    description: str = Field(..., description="Configuration description")

    # Template details
    memory_type: str = Field(..., description="Type of memory this extracts")
    template: str = Field(..., description="Jinja2 template string")
    variables: List[Dict[str, Any]] = Field(..., description="Template variables")
    output_schema: Dict[str, Any] = Field(..., description="JSON schema for output")

    # Validation rules
    min_confidence: float = Field(default=0.5)
    require_source_quotes: bool = Field(default=True)
    max_items_per_extraction: int = Field(default=50)

    # Testing
    test_cases: List[Dict[str, Any]] = Field(default_factory=list)
    validation_rules: List[str] = Field(default_factory=list)

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = Field(default=True)

    # Usage
    usage_count: int = Field(default=0)
    success_rate: float = Field(default=0.0)
    average_extraction_time: float = Field(default=0.0)


class LLMUsageRecord(BaseModel):
    """Record of LLM usage for tracking and billing"""

    id: str = Field(..., description="Usage record ID")
    user_id: str = Field(..., description="User ID")
    preset_id: str = Field(..., description="Preset used")
    session_id: Optional[str] = Field(
        default=None, description="Chat session ID if applicable"
    )

    # Provider details
    provider: str
    model: str

    # Token usage
    input_tokens: int
    output_tokens: int
    total_tokens: int

    # Cost calculation
    input_cost: float = Field(default=0.0)
    output_cost: float = Field(default=0.0)
    total_cost: float = Field(default=0.0)

    # Request details
    request_type: str = Field(
        ..., description="Type of request (chat, extraction, etc)"
    )
    success: bool = Field(default=True)
    error_message: Optional[str] = Field(default=None)

    # Timing
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    response_time_ms: int = Field(..., description="Response time in milliseconds")

    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)
