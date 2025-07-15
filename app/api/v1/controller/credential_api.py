"""Credential API."""
import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.tags import Tags
from app.api.v1.request.credential_request import (
    CreateCredentialRequest,
    DeleteCredentialRequest,
)
from app.api.v1.response.base_response import BaseResponse, success_response
from app.api.v1.response.credential_response import (
    CredentialListResponse,
    CredentialResponse,
)
from app.common.auth.auth import UserContext
from app.common.auth.role_decorator import require_roles
from app.db.session import get_db
from app.service.credential_service import CredentialService

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.Credential])


@router.get("/credential", response_model=BaseResponse[CredentialListResponse])
def get_credentials(
    user_id: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    context: UserContext = Depends(require_roles(["admin", "builder"])),
):
    actual_user_id = user_id or context.user_id
    creds = CredentialService(db, context).get_credentials_for_user(actual_user_id)
    return success_response(creds, "Credentials fetched successfully")


@router.post("/credential", response_model=BaseResponse[CredentialResponse])
def create_credential(
    body: CreateCredentialRequest,
    db: Session = Depends(get_db),
    context: UserContext = Depends(require_roles(["admin", "builder"])),
):
    credential = CredentialService(db, context).create_credential(body)
    return success_response(credential, "Credential created successfully")


@router.delete("/credential", response_model=BaseResponse[CredentialResponse])
def delete_credential(
    body: DeleteCredentialRequest,
    db: Session = Depends(get_db),
    context: UserContext = Depends(require_roles(["admin", "builder"])),
):
    credential = CredentialService(db, context).delete_credential(body)
    return success_response(credential, "Credential deleted successfully")
