"""Request models for Q&A endpoints"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QuestionRequest(BaseModel):
    """Request model for asking questions about memories"""

    question: str = Field(
        ...,
        description="The question to ask about stored memories",
        min_length=1,
        max_length=500,
    )
    user_id: str = Field(..., description="User ID to search memories for")
    session_id: Optional[str] = Field(
        None, description="Optional session ID for context"
    )
    max_memories: int = Field(
        default=20,
        description="Maximum number of memories to consider for reasoning",
        ge=1,
        le=100,
    )
    memory_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by specific memory types (semantic, episodic, etc.)",
    )
    time_range: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional time range filter with 'start' and 'end' keys",
    )
    include_meta_memories: bool = Field(
        default=True, description="Whether to include meta-memories in search"
    )
    search_depth: str = Field(
        default="semantic",
        description="Search depth: 'semantic', 'hybrid', or 'comprehensive'",
        pattern="^(semantic|hybrid|comprehensive)$",
    )


class ConversationRequest(BaseModel):
    """Request model for conversational Q&A with context"""

    messages: List[Dict[str, str]] = Field(
        ..., description="Conversation history with 'role' and 'content' keys"
    )
    user_id: str = Field(..., description="User ID to search memories for")
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation context"
    )
    max_memories: int = Field(
        default=15,
        description="Maximum number of memories to consider for context",
        ge=1,
        le=50,
    )
    conversation_context_window: int = Field(
        default=5,
        description="Number of previous messages to consider for context",
        ge=1,
        le=20,
    )
