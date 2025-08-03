"""LLM configuration models"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator

from app.llm.types import CategoryConfig


class BaseLLMConfig(BaseModel):
    """Base configuration for all LLM providers"""

    provider: str = Field(default="openai", description="LLM provider name")
    model: str = Field(default="gpt-4o-mini", description="Model identifier")
    temperature: float = Field(
        default=0.1, ge=0.0, le=2.0, description="Sampling temperature"
    )
    max_tokens: int = Field(
        default=2048, ge=1, description="Maximum tokens to generate"
    )
    top_p: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter"
    )
    top_k: int = Field(default=10, ge=1, description="Top-k sampling parameter")
    api_key: Optional[str] = Field(default=None, description="API key (if not in env)")
    api_base: Optional[str] = Field(default=None, description="Custom API endpoint")
    timeout: int = Field(default=60, description="Request timeout in seconds")

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider is supported"""
        supported = ["openai", "anthropic", "google"]
        if v not in supported:
            raise ValueError(f"Provider must be one of {supported}")
        return v


class OpenAIConfig(BaseLLMConfig):
    """OpenAI-specific configuration"""

    provider: str = "openai"
    model: str = "gpt-4o-mini"
    organization: Optional[str] = None
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    seed: Optional[int] = None


class AnthropicConfig(BaseLLMConfig):
    """Anthropic Claude-specific configuration"""

    provider: str = "anthropic"
    model: str = "claude-3-sonnet-20240229"
    max_tokens: int = 4096  # Claude has higher default


class GoogleConfig(BaseLLMConfig):
    """Google Gemini-specific configuration"""

    provider: str = "google"
    model: str = "gemini-1.5-flash"
    harm_block_threshold: str = "BLOCK_MEDIUM_AND_ABOVE"
    safety_settings: Optional[List[Dict]] = None


class MockConfig(BaseModel):
    """Configuration for Mock LLM provider"""

    model: str = "mock-model"
    response_delay: float = 0.1  # Simulate API delay
    stream_delay: float = 0.01  # Delay between stream chunks
    always_succeed: bool = True
    mock_errors: bool = False


class LLMPresetConfig(BaseModel):
    """Complete LLM preset configuration for users"""

    # Identity
    name: str = Field(..., description="Preset name")
    description: str = Field(..., description="Preset description")

    # Provider settings
    provider: str = Field(default="openai", description="LLM provider")
    model: str = Field(default="gpt-4o-mini", description="Model to use")

    # Generation parameters
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=10, ge=1)

    # Prompts
    system_prompt: str = Field(default="You are a helpful assistant.")
    custom_instructions: Optional[str] = None

    # Memory settings
    use_memory_context: bool = Field(
        default=True, description="Include relevant memories in context"
    )
    extract_memories: bool = Field(
        default=True, description="Extract memories from conversation"
    )
    memory_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Relevance threshold"
    )
    force_add_only: bool = Field(
        default=False, description="Only add new memories, don't update"
    )
    memory_extraction_types: List[str] = Field(
        default_factory=lambda: ["semantic_memory", "episodic_memory"],
        description="Types of memories to extract",
    )

    # Reranking
    reranking_enabled: bool = Field(
        default=True, description="Enable response reranking"
    )
    rerank_threshold: float = Field(default=0.3, ge=0.0, le=1.0)

    # Categories
    categories: List[CategoryConfig] = Field(
        default_factory=list, description="Custom categories"
    )

    # Additional settings
    conversation_history_limit: int = Field(
        default=10, description="Messages to include from history"
    )
    include_timestamps: bool = Field(
        default=False, description="Include timestamps in context"
    )
    response_format_preference: Optional[str] = Field(
        default=None, description="Preferred format: 'markdown', 'plain', 'json'"
    )

    # Cost controls
    daily_token_limit: Optional[int] = Field(
        default=None, description="Daily token limit"
    )
    monthly_budget: Optional[float] = Field(
        default=None, description="Monthly cost budget in USD"
    )

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, v: str) -> str:
        """Validate provider is supported"""
        supported = ["openai", "anthropic", "google"]
        if v not in supported:
            raise ValueError(f"Provider must be one of {supported}")
        return v

    @field_validator("memory_extraction_types")
    @classmethod
    def validate_memory_types(cls, v: List[str]) -> List[str]:
        """Validate memory types"""
        valid_types = [
            "semantic_memory",
            "episodic_memory",
            "procedural_memory",
            "graph_memory",
            "chat_memory",
        ]
        # Also accept short names for backward compatibility
        short_names = {
            "semantic": "semantic_memory",
            "episodic": "episodic_memory",
            "procedural": "procedural_memory",
            "graph": "graph_memory",
            "chat": "chat_memory",
        }

        validated = []
        for mem_type in v:
            if mem_type in valid_types:
                validated.append(mem_type)
            elif mem_type in short_names:
                validated.append(short_names[mem_type])
            elif mem_type.startswith("category_"):
                validated.append(mem_type)
            else:
                raise ValueError(f"Invalid memory type: {mem_type}")
        return validated
