"""Chat extracted memory model for storing memory snapshots"""
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from app.common.enum.memory import MemoryType


class ChatExtractedMemory(BaseModel):
    """
    Stores a snapshot of memories extracted during chat conversations.
    This is separate from the main memory store to allow:
    1. Fast retrieval for chat history
    2. Deletion with chat without affecting original memories
    3. Historical record of what was extracted at the time
    """

    id: str  # Same as the original memory ID
    chat_message_id: str  # The message that extracted this memory
    session_id: str
    user_id: str

    # Memory snapshot data
    content: str
    memory_type: MemoryType
    tags: List[str] = Field(default_factory=list)
    confidence: float = 1.0

    # Metadata
    extracted_at: datetime = Field(default_factory=datetime.utcnow)
    original_memory_id: str  # Reference to the actual memory

    # Optional fields from original memory
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

    class Config:
        indexes = [
            ("session_id", 1),
            ("chat_message_id", 1),
            ("user_id", 1),
            [("session_id", 1), ("extracted_at", -1)],
            [("chat_message_id", 1), ("original_memory_id", 1)],  # Compound index
        ]
