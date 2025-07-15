from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.common.enum.memory import MemoryType


class GraphLink(BaseModel):
    """Represents a link to another memory or entity in the graph"""

    target_id: str = Field(..., description="ID of the linked memory/entity")
    relationship_type: str = Field(
        ...,
        description="Type of relationship (e.g., 'relates_to', 'causes', 'part_of')",
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict, description="Additional properties of the relationship"
    )


class MemoryPermissions(BaseModel):
    """Access control for memory entries"""

    owner_id: str = Field(..., description="Owner of the memory")
    read_access: List[str] = Field(
        default_factory=list, description="List of user/group IDs with read access"
    )
    write_access: List[str] = Field(
        default_factory=list, description="List of user/group IDs with write access"
    )
    is_public: bool = Field(
        False, description="Whether this memory is publicly accessible"
    )


class MemoryMeta(BaseModel):
    """Metadata about how the memory was processed"""

    classifier_used: Optional[str] = Field(
        None, description="Which classifier was used if type was inferred"
    )
    llm_version: Optional[str] = Field(
        None, description="LLM version used for processing"
    )
    handler_version: Optional[str] = Field(
        None, description="Version of the handler that processed this"
    )
    processing_time_ms: Optional[float] = Field(
        None, description="Time taken to process in milliseconds"
    )
    confidence_score: Optional[float] = Field(
        None, description="Confidence in the classification/processing"
    )


class MemoryEntry(BaseModel):
    """Canonical persisted memory object"""

    # Core identifiers
    id: str = Field(..., description="Unique memory ID")
    cid: str = Field(..., description="Content ID for deduplication")
    scope: str = Field(..., description="Scope/namespace for the memory")

    # Content
    input: str = Field(..., description="Original input text")
    summary: Optional[str] = Field(
        None, description="Summarized version of the content"
    )
    memory_type: MemoryType = Field(..., description="Type of memory")

    # Access control
    permissions: MemoryPermissions = Field(..., description="Access permissions")

    # Vector data
    embedding: Optional[List[float]] = Field(
        None, description="Vector embedding of the content"
    )
    embedding_model: Optional[str] = Field(None, description="Model used for embedding")

    # Graph relationships
    graph_links: List[GraphLink] = Field(
        default_factory=list, description="Links to other memories/entities"
    )

    # Metadata
    meta: MemoryMeta = Field(..., description="Processing metadata")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    custom_metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional custom metadata"
    )

    # Timestamps
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="When the memory was created"
    )
    updated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Last update time"
    )
    accessed_at: Optional[datetime] = Field(None, description="Last access time")

    # Source tracking
    source_session_id: str = Field(..., description="Session that created this memory")
    source_user_id: str = Field(..., description="User who created this memory")

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
