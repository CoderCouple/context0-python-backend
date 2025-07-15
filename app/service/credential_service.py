from datetime import datetime

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.api.v1.request.credential_request import (
    CreateCredentialRequest,
    DeleteCredentialRequest,
)
from app.api.v1.response.credential_response import (
    CredentialListResponse,
    CredentialResponse,
)
from app.common.auth.auth import UserContext
from app.model.credential_model import Credential


class CredentialService:
    def __init__(self, db: Session, context: UserContext):
        self.db = db
        self.context = context

    def get_credentials_for_user(self, user_id: str) -> CredentialListResponse:
        creds = (
            self.db.query(Credential)
            .filter(Credential.user_id == user_id, Credential.is_deleted == False)
            .all()
        )
        return CredentialListResponse(credentials=creds)

    def create_credential(self, body: CreateCredentialRequest) -> CredentialResponse:
        credential = Credential(
            user_id=body.user_id,
            name=body.name,
            value=body.value,
            created_by=self.context.user_id,
            updated_by=self.context.user_id,
        )
        self.db.add(credential)
        self.db.commit()
        self.db.refresh(credential)
        return CredentialResponse.from_orm(credential)

    def delete_credential(self, body: DeleteCredentialRequest) -> CredentialResponse:
        credential = (
            self.db.query(Credential)
            .filter(
                Credential.user_id == body.user_id,
                Credential.name == body.name,
                Credential.is_deleted == False,
            )
            .first()
        )

        if not credential:
            raise HTTPException(status_code=404, detail="Credential not found")

        if body.is_soft_delete:
            credential.is_deleted = True
            credential.updated_by = self.context.user_id
            credential.updated_at = datetime.utcnow()
        else:
            self.db.delete(credential)

        self.db.commit()
        return CredentialResponse.from_orm(credential)
