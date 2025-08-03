"""Chat API response models"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from app.common.enum.chat import ChatRole
from app.common.enum.memory import MemoryType


class MemoryContextItem(BaseModel):
    """Memory item used as context in chat"""

    id: str
    content: str
    summary: Optional[str] = None
    memory_type: MemoryType
    score: float
    tags: List[str] = Field(default_factory=list)
    created_at: datetime


class ExtractedMemoryItem(BaseModel):
    """Memory item extracted from conversation"""

    id: str
    content: str
    memory_type: MemoryType
    tags: List[str] = Field(default_factory=list)
    confidence: float = 1.0


class ChatMessageResponse(BaseModel):
    """Chat message response"""

    id: str
    role: ChatRole
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None
    memories_extracted: Optional[List[ExtractedMemoryItem]] = None
    context_used: Optional[List[MemoryContextItem]] = None


class ChatSessionResponse(BaseModel):
    """Chat session response"""

    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime
    last_message: Optional[str] = None
    message_count: int = 0
    total_memories_extracted: int = 0


class ChatSessionDetailResponse(BaseModel):
    """Detailed chat session response with messages"""

    session: ChatSessionResponse
    messages: List[ChatMessageResponse]


class CreateChatSessionResponse(BaseModel):
    """Response after creating a chat session"""

    session_id: str
    title: str
    created_at: datetime


class SendMessageResponse(BaseModel):
    """Response after sending a message"""

    user_message: ChatMessageResponse
    assistant_message: ChatMessageResponse
    memories_extracted: List[ExtractedMemoryItem] = Field(default_factory=list)
    context_used: List[MemoryContextItem] = Field(default_factory=list)


class ExtractMemoriesResponse(BaseModel):
    """Response after extracting memories from chat"""

    session_id: str
    messages_processed: int
    memories_extracted: List[str]
    extraction_summary: Dict[str, Any]
