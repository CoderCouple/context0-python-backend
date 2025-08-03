"""Response models for LLM Memory API"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pydantic import BaseModel, Field
from app.common.enum.llm_vendor import LLMVendor


class LLMMemoryResponse(BaseModel):
    """Base response for LLM memory operations"""

    success: bool = Field(..., description="Operation success status")
    vendor: LLMVendor = Field(..., description="LLM vendor")
    operation: str = Field(..., description="Operation performed")
    message: str = Field(..., description="Status message")
    deleted_count: Optional[int] = Field(None, description="Number of items deleted")
    details: Optional[Dict[str, Any]] = Field(
        None, description="Additional operation details"
    )


class ConversationResponse(BaseModel):
    """Response from LLM with injected memories"""

    success: bool = Field(..., description="Operation success status")
    vendor: LLMVendor = Field(..., description="LLM vendor")
    response: str = Field(..., description="LLM's response")
    memories_used: int = Field(..., description="Number of memories injected")
    session_id: Optional[str] = Field(None, description="Session ID for continuity")
    model_used: Optional[str] = Field(None, description="Model that was used")
    usage: Optional[Dict[str, int]] = Field(None, description="Token usage information")
    processing_time_ms: Optional[int] = Field(
        None, description="Processing time in milliseconds"
    )


class FormattedMemoriesResponse(BaseModel):
    """Response containing formatted memories"""

    success: bool = Field(..., description="Operation success status")
    memory_count: int = Field(..., description="Number of memories retrieved")
    formatted_context: str = Field(..., description="Formatted memory context")
    raw_memories: List[Dict[str, Any]] = Field(..., description="Raw memory data")
    total_tokens: Optional[int] = Field(None, description="Estimated token count")
    vendor_format: LLMVendor = Field(..., description="Format optimized for vendor")


class ConversationInfo(BaseModel):
    """Information about a conversation/thread"""

    id: str = Field(..., description="Conversation/thread ID")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    message_count: Optional[int] = Field(None, description="Number of messages")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    title: Optional[str] = Field(None, description="Conversation title")


class ListConversationsResponse(BaseModel):
    """Response containing list of conversations"""

    success: bool = Field(..., description="Operation success status")
    conversations: List[ConversationInfo] = Field(
        ..., description="List of conversations"
    )
    count: int = Field(..., description="Total number of conversations")
    vendor: LLMVendor = Field(..., description="LLM vendor")
    has_more: bool = Field(
        default=False, description="Whether more conversations exist"
    )


class MemoryInjectionStats(BaseModel):
    """Statistics about memory injection"""

    total_memories_searched: int = Field(..., description="Total memories searched")
    memories_matched: int = Field(..., description="Memories that matched criteria")
    memories_injected: int = Field(..., description="Memories actually injected")
    categories_used: List[str] = Field(
        default_factory=list, description="Categories of injected memories"
    )
    emotions_present: List[str] = Field(
        default_factory=list, description="Emotions in injected memories"
    )
    time_range: Optional[Dict[str, datetime]] = Field(
        None, description="Time range of memories"
    )


class DetailedConversationResponse(ConversationResponse):
    """Extended conversation response with memory details"""

    injection_stats: MemoryInjectionStats = Field(
        ..., description="Memory injection statistics"
    )
    memories_detail: List[Dict[str, Any]] = Field(
        default_factory=list, description="Detailed memory information"
    )


class SyncedMemory(BaseModel):
    """Information about a synced memory"""

    vendor: LLMVendor = Field(..., description="Source LLM vendor")
    original_id: str = Field(..., description="Original ID from LLM vendor")
    context_zero_id: str = Field(..., description="New ID in Context Zero")
    timestamp: datetime = Field(..., description="Original timestamp")
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    category: Optional[str] = Field(None, description="Detected category")
    emotion: Optional[str] = Field(None, description="Detected emotion")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class VendorSyncResult(BaseModel):
    """Sync result for a single vendor"""

    vendor: LLMVendor = Field(..., description="LLM vendor")
    success: bool = Field(..., description="Whether sync was successful")
    total_conversations: int = Field(..., description="Total conversations found")
    total_messages: int = Field(..., description="Total messages found")
    synced_count: int = Field(..., description="Number of memories synced")
    skipped_count: int = Field(..., description="Number of memories skipped")
    error_count: int = Field(..., description="Number of errors")
    errors: List[str] = Field(default_factory=list, description="Error messages")
    synced_memories: List[SyncedMemory] = Field(
        default_factory=list, description="Successfully synced memories"
    )


class MemorySyncResponse(BaseModel):
    """Response from memory sync operation"""

    success: bool = Field(..., description="Overall sync success")
    sync_id: str = Field(..., description="Unique sync operation ID")
    started_at: datetime = Field(..., description="Sync start time")
    completed_at: datetime = Field(..., description="Sync completion time")
    duration_seconds: float = Field(..., description="Total sync duration")
    vendor_results: List[VendorSyncResult] = Field(
        ..., description="Results per vendor"
    )
    total_synced: int = Field(
        ..., description="Total memories synced across all vendors"
    )
    total_errors: int = Field(..., description="Total errors across all vendors")
    sync_mode: str = Field(..., description="Sync mode used")


class SyncStatusResponse(BaseModel):
    """Response for sync status check"""

    sync_id: str = Field(..., description="Sync operation ID")
    status: str = Field(
        ..., description="Current status: pending, running, completed, failed"
    )
    progress: float = Field(..., description="Progress percentage (0-100)")
    current_vendor: Optional[LLMVendor] = Field(
        None, description="Currently processing vendor"
    )
    processed_count: int = Field(..., description="Memories processed so far")
    total_count: Optional[int] = Field(None, description="Total memories to process")
    errors: List[str] = Field(default_factory=list, description="Current errors")
