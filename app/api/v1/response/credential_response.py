from datetime import datetime
from typing import List

from pydantic import BaseModel


class CredentialResponse(BaseModel):
    id: str
    user_id: str
    name: str
    value: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}  # replaces orm_mode=True


class CredentialListResponse(BaseModel):
    credentials: List[CredentialResponse]
