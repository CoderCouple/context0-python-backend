"""Request models for LLM Memory API"""

from typing import Dict, List, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from app.common.enum.llm_vendor import LLMVendor
from app.common.enum.memory import MemoryType
from app.common.enum.memory_category import MemoryCategory
from app.common.enum.memory_emotion import MemoryEmotion


class DeleteMemoryRequest(BaseModel):
    """Request to delete memory from LLM vendor"""

    vendor: LLMVendor = Field(..., description="LLM vendor (openai, anthropic, google)")
    conversation_id: Optional[str] = Field(
        None, description="Conversation/thread ID to delete"
    )
    message_ids: Optional[List[str]] = Field(
        None, description="Specific message IDs to delete"
    )
    delete_all: bool = Field(default=False, description="Delete all conversations")
    api_key: Optional[str] = Field(None, description="API key (optional if set in env)")


class InjectMemoryRequest(BaseModel):
    """Request to inject memories into LLM conversation"""

    vendor: LLMVendor = Field(..., description="LLM vendor")
    message: str = Field(
        ..., min_length=1, max_length=10000, description="User's message/prompt"
    )
    memory_types: Optional[List[MemoryType]] = Field(
        None, description="Filter memories by type"
    )
    categories: Optional[List[MemoryCategory]] = Field(
        None, description="Filter memories by category"
    )
    emotions: Optional[List[MemoryEmotion]] = Field(
        None, description="Filter memories by emotion"
    )
    tags: Optional[List[str]] = Field(None, description="Filter memories by tags")
    limit: int = Field(
        default=10, ge=1, le=50, description="Number of memories to inject"
    )
    include_summary: bool = Field(default=True, description="Include memory summaries")
    include_timeline: bool = Field(
        default=False, description="Include temporal context"
    )
    api_key: Optional[str] = Field(None, description="API key (optional if set in env)")
    session_id: Optional[str] = Field(
        None, description="Session ID for conversation continuity"
    )

    # Model configuration
    model: Optional[str] = Field(
        None, description="Specific model to use (e.g., gpt-4, claude-3)"
    )
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="Model temperature"
    )
    max_tokens: int = Field(
        default=1000, ge=1, le=4000, description="Maximum response tokens"
    )


class GetFormattedMemoriesRequest(BaseModel):
    """Request to get formatted memories for manual injection"""

    vendor: LLMVendor = Field(..., description="LLM vendor")
    query: str = Field(
        ..., min_length=1, max_length=1000, description="Query for memory search"
    )
    memory_types: Optional[List[MemoryType]] = Field(
        None, description="Filter by memory types"
    )
    categories: Optional[List[MemoryCategory]] = Field(
        None, description="Filter by categories"
    )
    emotions: Optional[List[MemoryEmotion]] = Field(
        None, description="Filter by emotions"
    )
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(
        default=10, ge=1, le=50, description="Maximum memories to retrieve"
    )
    include_metadata: bool = Field(default=False, description="Include full metadata")


class ClearContextRequest(BaseModel):
    """Request to clear LLM context"""

    vendor: LLMVendor = Field(..., description="LLM vendor")
    session_id: Optional[str] = Field(None, description="Session ID to clear")
    api_key: Optional[str] = Field(None, description="API key (optional if set in env)")


class ListConversationsRequest(BaseModel):
    """Request to list LLM conversations"""

    vendor: LLMVendor = Field(..., description="LLM vendor")
    api_key: Optional[str] = Field(None, description="API key (optional if set in env)")
    limit: int = Field(
        default=100, ge=1, le=500, description="Maximum conversations to retrieve"
    )


class SyncMemoriesRequest(BaseModel):
    """Request to sync memories from external LLM to Context Zero"""

    vendors: List[LLMVendor] = Field(..., description="LLM vendors to sync from")
    api_keys: Optional[Dict[LLMVendor, str]] = Field(
        None, description="API keys per vendor"
    )
    sync_mode: str = Field(
        default="full", description="Sync mode: 'full' or 'incremental'"
    )
    start_date: Optional[datetime] = Field(
        None, description="Start date for incremental sync"
    )
    end_date: Optional[datetime] = Field(
        None, description="End date for incremental sync"
    )
    auto_categorize: bool = Field(
        default=True, description="Auto-categorize imported memories"
    )
    detect_emotions: bool = Field(
        default=True, description="Detect emotions in imported memories"
    )
    conversation_ids: Optional[Dict[LLMVendor, List[str]]] = Field(
        None, description="Specific conversations to sync"
    )


class MemorySyncSettings(BaseModel):
    """Settings for memory synchronization"""

    chunk_size: int = Field(
        default=50, ge=1, le=200, description="Number of memories to process at once"
    )
    deduplicate: bool = Field(default=True, description="Remove duplicate memories")
    merge_similar: bool = Field(default=False, description="Merge similar memories")
    similarity_threshold: float = Field(
        default=0.85, ge=0.0, le=1.0, description="Threshold for similarity"
    )
    preserve_metadata: bool = Field(
        default=True, description="Preserve original LLM metadata"
    )
