from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.common.enum.memory import MemoryType


class MemoryRecord(BaseModel):
    """Runtime context model for incoming memory observations"""

    user_id: str = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    memory_type: Optional[MemoryType] = Field(
        None, description="Type of memory (optional, will be inferred if missing)"
    )
    raw_text: str = Field(..., description="Raw text content of the observation")
    timestamp: datetime = Field(
        default_factory=datetime.utcnow, description="When the observation was made"
    )
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    class Config:
        json_encoders = {datetime: lambda v: v.isoformat()}
