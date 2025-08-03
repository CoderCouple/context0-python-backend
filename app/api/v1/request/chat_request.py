"""Chat API request models"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class CreateChatSessionRequest(BaseModel):
    """Request to create a new chat session"""

    title: Optional[str] = Field(
        None, description="Session title, auto-generated if not provided"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SendMessageRequest(BaseModel):
    """Request to send a message in a chat session"""

    content: str = Field(..., description="Message content", min_length=1)
    extract_memories: bool = Field(
        default=True, description="Whether to extract memories from this message"
    )
    use_memory_context: bool = Field(
        default=True, description="Whether to use memory context for AI response"
    )


class UpdateChatSessionRequest(BaseModel):
    """Request to update a chat session"""

    title: Optional[str] = Field(None, description="New session title")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")


class ExtractMemoriesRequest(BaseModel):
    """Request to extract memories from a chat session"""

    message_ids: Optional[List[str]] = Field(
        None, description="Specific message IDs to extract from, or all if not provided"
    )
    force: bool = Field(
        default=False, description="Force re-extraction even if already extracted"
    )
