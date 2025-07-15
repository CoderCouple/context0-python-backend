from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from app.common.enum.memory import MemoryOperation, MemoryType


class AuditLogEntry(BaseModel):
    """Audit log entry for tracking all memory operations"""

    # Core fields
    id: str = Field(..., description="Unique audit log ID")
    action: MemoryOperation = Field(
        ..., description="Operation performed (ADD/UPDATE/DELETE)"
    )
    memory_id: str = Field(..., description="ID of the memory that was operated on")
    cid: str = Field(..., description="Content ID of the memory")

    # Context
    user_id: str = Field(..., description="User who performed the operation")
    session_id: str = Field(..., description="Session in which operation was performed")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the operation occurred"
    )

    # Operation details
    inferred_type: Optional[MemoryType] = Field(
        None, description="Memory type if it was inferred"
    )
    handler_used: Optional[str] = Field(
        None, description="Which handler processed the memory"
    )

    # Change tracking
    before_state: Optional[Dict[str, Any]] = Field(
        None, description="State before the operation (for UPDATE/DELETE)"
    )
    after_state: Optional[Dict[str, Any]] = Field(
        None, description="State after the operation (for ADD/UPDATE)"
    )
    changes: Optional[Dict[str, Any]] = Field(
        None, description="Specific fields that changed (for UPDATE)"
    )

    # Performance metrics
    processing_time_ms: Optional[float] = Field(
        None, description="Time taken for the operation"
    )

    # Additional context
    ip_address: Optional[str] = Field(None, description="IP address of the request")
    user_agent: Optional[str] = Field(None, description="User agent string")
    api_version: Optional[str] = Field(None, description="API version used")

    # Error tracking
    error: Optional[str] = Field(None, description="Error message if operation failed")
    error_type: Optional[str] = Field(
        None, description="Type of error if operation failed"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
