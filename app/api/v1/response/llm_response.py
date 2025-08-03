"""LLM API response models"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

from app.llm.types import CategoryConfig


class LLMPresetResponse(BaseModel):
    """Response for LLM preset"""

    id: str
    user_id: str
    name: str
    description: str

    # Provider settings
    provider: str
    model: str

    # Generation parameters
    temperature: float
    max_tokens: int
    top_p: float
    top_k: int

    # Prompts
    system_prompt: str
    custom_instructions: Optional[str]

    # Memory settings
    use_memory_context: bool
    extract_memories: bool
    memory_threshold: float
    force_add_only: bool
    memory_extraction_types: List[str]

    # Reranking
    reranking_enabled: bool
    rerank_threshold: float

    # Categories
    categories: List[CategoryConfig]

    # Additional settings
    conversation_history_limit: int
    include_timestamps: bool
    response_format_preference: Optional[str]

    # Cost controls
    daily_token_limit: Optional[int]
    monthly_budget: Optional[float]

    # Metadata
    created_at: datetime
    updated_at: datetime
    is_default: bool
    is_shared: bool
    tags: List[str]

    # Usage statistics
    usage_count: int
    last_used_at: Optional[datetime]
    total_tokens_used: int
    total_cost: float


class CreateLLMPresetResponse(BaseModel):
    """Response for creating LLM preset"""

    preset_id: str
    name: str
    provider: str
    model: str
    created_at: datetime


class ProviderInfoResponse(BaseModel):
    """Information about an LLM provider"""

    provider: str
    name: str
    models: List[str]
    supports_tools: bool
    supports_vision: bool
    supports_streaming: bool
    supports_json_mode: bool
    pricing: Dict[str, Dict[str, float]]  # model -> {input: cost, output: cost}


class TestLLMPresetResponse(BaseModel):
    """Response for testing LLM preset"""

    # Request echo
    test_message: str

    # Generated response
    response: str

    # Token usage
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float

    # Timing
    response_time_ms: int

    # Context used (if applicable)
    memory_context_used: Optional[List[str]] = None
    system_prompt_used: Optional[str] = None

    # Model info
    provider: str
    model: str
    temperature_used: float
    max_tokens_used: int


class LLMUsageStatsResponse(BaseModel):
    """Response for LLM usage statistics"""

    period: Dict[str, str]  # {start: ISO date, end: ISO date}

    # Total stats
    total: Dict[str, Any]  # {requests: int, tokens: int, cost: float}

    # By model breakdown
    by_model: List[Dict[str, Any]]  # List of usage by provider/model

    # Daily breakdown (optional)
    daily_breakdown: Optional[List[Dict[str, Any]]] = None

    # Cost projections
    projected_monthly_cost: Optional[float] = None
    remaining_budget: Optional[float] = None


class PromptTemplateResponse(BaseModel):
    """Response for prompt template"""

    id: str
    name: str
    description: str
    memory_type: str
    variables: List[Dict[str, Any]]
    output_schema: Dict[str, Any]

    # Validation settings
    min_confidence: float
    require_source_quotes: bool
    max_items_per_extraction: int

    # Metadata
    created_at: datetime
    updated_at: datetime
    is_active: bool

    # Usage stats
    usage_count: int
    success_rate: float
    average_extraction_time: float
