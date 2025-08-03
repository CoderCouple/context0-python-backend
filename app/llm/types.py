"""Type definitions for LLM module"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    """Token usage information"""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def estimated_cost(self) -> float:
        """Estimate cost based on token usage (override in subclasses)"""
        return 0.0


class ToolCall(BaseModel):
    """Represents a tool/function call from the LLM"""

    id: str
    name: str
    arguments: Dict[str, Any]


class LLMResponse(BaseModel):
    """Standard response from any LLM provider"""

    content: str
    usage: TokenUsage
    model: str
    provider: str
    tool_calls: Optional[List[ToolCall]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # For streaming responses
    is_complete: bool = True
    chunk_index: Optional[int] = None


class ValidationResult(BaseModel):
    """Result of JSON validation"""

    is_valid: bool
    data: Optional[Dict[str, Any]] = None
    errors: List[str] = Field(default_factory=list)


class CategoryConfig(BaseModel):
    """Configuration for custom memory categories"""

    name: str = Field(..., description="Category name")
    description: str = Field(..., description="Category description")
    includes_prompt: str = Field(..., description="Keywords/patterns to include")
    excludes_prompt: str = Field(..., description="Keywords/patterns to exclude")
    priority: int = Field(default=1, description="Priority for extraction (1-10)")
    max_items: int = Field(default=10, description="Max items to extract")


class PromptVariable(BaseModel):
    """Definition of a prompt template variable"""

    name: str
    description: str
    required: bool = True
    default_value: Optional[Any] = None
    validation_regex: Optional[str] = None
