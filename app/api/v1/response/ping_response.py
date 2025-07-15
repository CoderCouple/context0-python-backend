from typing import Dict

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    stores: Dict[str, bool]
    memory_count: int
