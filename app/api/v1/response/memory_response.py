# Response Models
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.common.enum.memory import MemoryOperation, MemoryType


class MemoryEntry(BaseModel):
    """Complete memory entry response model"""

    id: str = Field(..., description="Unique memory identifier")
    cid: str = Field(..., description="Content identifier for deduplication")
    input: str = Field(..., description="Original input text")
    summary: Optional[str] = Field(None, description="AI-generated summary")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    scope: str = Field(..., description="Memory scope/namespace")
    memory_type: MemoryType = Field(..., description="Classified memory type")
    confidence: float = Field(..., description="Classification confidence score")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    last_accessed: Optional[datetime] = Field(None, description="Last access timestamp")
    access_count: int = Field(default=0, description="Number of times accessed")
    is_deleted: bool = Field(default=False, description="Soft deletion flag")
    meta: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class SearchResult(BaseModel):
    """Search result item model"""

    id: str = Field(..., description="Memory identifier")
    summary: str = Field(..., description="Memory summary")
    memory_type: MemoryType = Field(..., description="Memory type")
    score: float = Field(..., description="Relevance score")
    tags: List[str] = Field(default_factory=list, description="Memory tags")
    created_at: datetime = Field(..., description="Creation timestamp")
    confidence: float = Field(..., description="Classification confidence")
    content: Optional[str] = Field(None, description="Full content (if requested)")
    scope: Optional[str] = Field(None, description="Memory scope")


class MemoryResponse(BaseModel):
    """Response for memory creation operations"""

    success: bool = Field(..., description="Operation success flag")
    memory_id: str = Field(..., description="Created memory identifier")
    operation: MemoryOperation = Field(..., description="Performed operation")
    confidence: float = Field(..., description="Processing confidence")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    message: str = Field(..., description="Human-readable message")
    memory_type: Optional[MemoryType] = Field(None, description="Inferred memory type")


class SearchResponse(BaseModel):
    """Response for memory search operations"""

    success: bool = Field(..., description="Search success flag")
    results: List[SearchResult] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total matching memories")
    query_time_ms: int = Field(..., description="Query execution time")
    query: str = Field(..., description="Original search query")
    filters_applied: Dict[str, Any] = Field(
        default_factory=dict, description="Applied filters"
    )


class BulkMemoryResponse(BaseModel):
    """Response for bulk memory operations"""

    success: bool = Field(..., description="Overall operation success")
    processed: int = Field(..., description="Total memories processed")
    successful: int = Field(..., description="Successfully processed memories")
    failed: int = Field(..., description="Failed memory operations")
    processing_time_ms: int = Field(..., description="Total processing time")
    results: List[Dict[str, Any]] = Field(
        ..., description="Individual operation results"
    )
    errors: List[str] = Field(default_factory=list, description="Error messages")


class MemoryAnalyticsResponse(BaseModel):
    """Response for memory analytics queries"""

    success: bool = Field(..., description="Analytics success flag")
    period: str = Field(..., description="Analytics period")
    user_id: Optional[str] = Field(None, description="User filter applied")
    metrics: Dict[str, Any] = Field(..., description="Analytics metrics")
    trends: Dict[str, List[Any]] = Field(..., description="Trending data")
    summary: Dict[str, Any] = Field(..., description="Summary statistics")


class TimelineResponse(BaseModel):
    """Response for timeline/evolution queries"""

    success: bool = Field(..., description="Timeline query success")
    memory_id: Optional[str] = Field(
        None, description="Memory ID (for specific evolution)"
    )
    target_time: Optional[datetime] = Field(
        None, description="Target time (for time travel)"
    )
    timeline: List[Dict[str, Any]] = Field(..., description="Timeline entries")
    total_events: int = Field(..., description="Total timeline events")


class HealthResponse(BaseModel):
    """System health response model"""

    status: str = Field(..., description="Overall system status")
    version: str = Field(..., description="System version")
    uptime_seconds: float = Field(..., description="System uptime")
    stores: Dict[str, bool] = Field(..., description="Store health status")
    memory_count: int = Field(..., description="Total memory count")
    processing_stats: Dict[str, Any] = Field(
        default_factory=dict, description="Processing statistics"
    )


class ExportResponse(BaseModel):
    """Response for memory export operations"""

    success: bool = Field(..., description="Export success flag")
    format: str = Field(..., description="Export format")
    total_memories: int = Field(..., description="Total exported memories")
    file_size_bytes: int = Field(..., description="Export file size")
    download_url: Optional[str] = Field(
        None, description="Download URL (if applicable)"
    )
    export_id: str = Field(..., description="Export operation identifier")
