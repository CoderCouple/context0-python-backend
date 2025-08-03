"""Chat models for database storage"""
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from app.common.enum.chat import ChatRole, ChatSessionStatus


class ChatMessage(BaseModel):
    """Chat message model"""

    id: str
    session_id: str
    role: ChatRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    # Memory extraction tracking
    memories_extracted: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)
    context_used: List[Union[str, Dict[str, Any]]] = Field(default_factory=list)

    @field_validator("role", mode="before")
    @classmethod
    def validate_role(cls, v):
        """Convert string to ChatRole enum if needed"""
        if isinstance(v, str):
            return ChatRole(v)
        return v


class ChatSession(BaseModel):
    """Chat session model"""

    id: str
    user_id: str
    title: str
    status: ChatSessionStatus = ChatSessionStatus.ACTIVE
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Summary fields
    last_message: Optional[str] = None
    message_count: int = 0
    total_memories_extracted: int = 0

    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    @field_validator("status", mode="before")
    @classmethod
    def validate_status(cls, v):
        """Convert string to ChatSessionStatus enum if needed"""
        if isinstance(v, str):
            return ChatSessionStatus(v)
        return v
