from typing import Optional

from pydantic import BaseModel


class CreateCredentialRequest(BaseModel):
    user_id: str
    name: str
    value: str


class DeleteCredentialRequest(BaseModel):
    name: str
    user_id: str
    is_soft_delete: Optional[bool] = True
