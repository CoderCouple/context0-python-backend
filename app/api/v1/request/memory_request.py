from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator

from app.common.enum.memory import MemoryType
from app.common.enum.memory_category import MemoryCategory
from app.common.enum.memory_emotion import MemoryEmotion, EmotionIntensity


# Request Models
class MemoryRecordInput(BaseModel):
    """Input model for creating a new memory"""

    user_id: str = Field(
        ..., min_length=1, max_length=100, description="User identifier"
    )
    session_id: str = Field(
        ..., min_length=1, max_length=100, description="Session identifier"
    )
    text: str = Field(..., min_length=1, max_length=50000, description="Memory content")
    memory_type: Optional[MemoryType] = Field(
        None, description="Optional memory type hint"
    )
    tags: List[str] = Field(
        default_factory=list, max_items=20, description="Tags for categorization"
    )
    category: Optional[Union[MemoryCategory, str]] = Field(
        None, description="Primary memory category"
    )
    emotion: Optional[Union[MemoryEmotion, str]] = Field(
        None, description="Primary emotional context of the memory"
    )
    emotion_intensity: Optional[Union[EmotionIntensity, str]] = Field(
        None, description="Intensity of the emotion"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )
    scope: Optional[str] = Field(None, description="Memory scope/namespace")

    @validator("tags")
    def validate_tags(cls, v):
        return [tag.strip().lower()[:50] for tag in v if tag.strip()]

    @validator("text")
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

    @validator("category", pre=True)
    def validate_category(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return MemoryCategory(v.lower())
            except ValueError:
                # If not a valid enum value, return None or a default
                return None
        return v

    @validator("emotion", pre=True)
    def validate_emotion(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return MemoryEmotion(v.lower())
            except ValueError:
                # If not a valid enum value, return None
                return None
        return v

    @validator("emotion_intensity", pre=True)
    def validate_emotion_intensity(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            try:
                return EmotionIntensity(v.lower())
            except ValueError:
                # If not a valid enum value, return None
                return None
        return v


class SearchQuery(BaseModel):
    """Query model for searching memories"""

    user_id: str = Field(..., min_length=1, description="User ID for filtering")
    query: str = Field(
        ..., min_length=1, max_length=1000, description="Search query text"
    )
    memory_types: Optional[List[MemoryType]] = Field(
        None, description="Filter by memory types"
    )
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    category: Optional[MemoryCategory] = Field(
        None, description="Filter by primary category"
    )
    emotion: Optional[MemoryEmotion] = Field(
        None, description="Filter by primary emotion"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Maximum results")
    threshold: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Similarity threshold"
    )
    include_content: bool = Field(
        default=False, description="Include full content in results"
    )
    scope: Optional[str] = Field(None, description="Filter by scope")
    start_date: Optional[datetime] = Field(
        None, description="Filter by creation date range"
    )
    end_date: Optional[datetime] = Field(
        None, description="Filter by creation date range"
    )


class MemoryUpdate(BaseModel):
    """Model for updating an existing memory"""

    text: Optional[str] = Field(
        None, max_length=50000, description="Updated memory content"
    )
    tags: Optional[List[str]] = Field(None, max_items=20, description="Updated tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    scope: Optional[str] = Field(None, description="Updated scope")

    @validator("tags")
    def validate_tags(cls, v):
        if v is not None:
            return [tag.strip().lower()[:50] for tag in v if tag.strip()]
        return v

    @validator("text")
    def validate_text(cls, v):
        if v is not None and (not v or not v.strip()):
            raise ValueError("Text cannot be empty")
        return v.strip() if v else v


class TimeRangeQuery(BaseModel):
    """Query model for time-based memory queries"""

    user_id: str = Field(..., description="User ID")
    start_time: datetime = Field(..., description="Start time for query range")
    end_time: datetime = Field(..., description="End time for query range")
    memory_types: Optional[List[MemoryType]] = Field(
        None, description="Filter by memory types"
    )
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    limit: int = Field(default=50, ge=1, le=200, description="Maximum results")


class BulkMemoryInput(BaseModel):
    """Model for bulk memory operations"""

    memories: List[MemoryRecordInput] = Field(
        ..., max_items=50, description="List of memories to process"
    )
    batch_size: int = Field(
        default=10, ge=1, le=20, description="Processing batch size"
    )
    fail_fast: bool = Field(default=False, description="Stop processing on first error")


class MemoryAnalyticsQuery(BaseModel):
    """Query model for memory analytics"""

    user_id: Optional[str] = Field(None, description="Filter by user")
    start_date: Optional[datetime] = Field(None, description="Analytics start date")
    end_date: Optional[datetime] = Field(None, description="Analytics end date")
    group_by: str = Field(
        default="day", description="Grouping period (hour, day, week, month)"
    )
    metrics: List[str] = Field(
        default=["count", "types"], description="Metrics to include"
    )


class MemoryExportRequest(BaseModel):
    """Request model for exporting memories"""

    user_id: str = Field(..., description="User ID")
    format: str = Field(default="json", description="Export format (json, csv, txt)")
    memory_types: Optional[List[MemoryType]] = Field(
        None, description="Filter by memory types"
    )
    tags: Optional[List[str]] = Field(None, description="Filter by tags")
    start_date: Optional[datetime] = Field(None, description="Export date range start")
    end_date: Optional[datetime] = Field(None, description="Export date range end")
    include_deleted: bool = Field(
        default=False, description="Include soft-deleted memories"
    )
