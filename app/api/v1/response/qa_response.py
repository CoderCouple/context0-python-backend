"""Response models for Q&A endpoints"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MemoryContext(BaseModel):
    """Memory context used in generating the answer"""

    memory_id: str = Field(..., description="Unique memory identifier")
    content: str = Field(..., description="Memory content")
    summary: Optional[str] = Field(None, description="Memory summary")
    memory_type: str = Field(..., description="Type of memory")
    relevance_score: float = Field(..., description="Relevance score (0.0 to 1.0)")
    created_at: datetime = Field(..., description="When the memory was created")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    source: str = Field(..., description="Which store provided this memory")


class QuestionResponse(BaseModel):
    """Response model for question answering"""

    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="Generated answer based on memories")
    confidence: float = Field(..., description="Confidence in the answer (0.0 to 1.0)")
    memories_found: int = Field(..., description="Number of relevant memories found")
    memories_used: int = Field(
        ..., description="Number of memories used to generate answer"
    )
    memory_contexts: List[MemoryContext] = Field(
        default_factory=list, description="Memories used as context for the answer"
    )
    search_strategy: str = Field(..., description="Search strategy used")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    suggestions: List[str] = Field(
        default_factory=list, description="Suggested follow-up questions"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the response"
    )


class ConversationResponse(BaseModel):
    """Response model for conversational Q&A"""

    response: str = Field(..., description="Conversational response")
    confidence: float = Field(..., description="Confidence in the response")
    context_memories: List[MemoryContext] = Field(
        default_factory=list, description="Memories that provided context"
    )
    conversation_context: List[Dict[str, str]] = Field(
        default_factory=list, description="Relevant conversation history used"
    )
    follow_up_suggestions: List[str] = Field(
        default_factory=list, description="Suggested follow-up questions or topics"
    )
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")


class QAHealthResponse(BaseModel):
    """Health check response for Q&A system"""

    status: str = Field(..., description="Overall system status")
    memory_stores: Dict[str, bool] = Field(
        ..., description="Status of each memory store"
    )
    embedding_service: bool = Field(..., description="Embedding service status")
    llm_service: bool = Field(..., description="LLM service status")
    total_memories: Optional[int] = Field(None, description="Total memories available")
    last_updated: datetime = Field(..., description="Last system update time")
