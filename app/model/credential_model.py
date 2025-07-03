import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, String
from sqlalchemy.dialects.postgresql import UUID

from app.db.base import Base


def generate_prefixed_uuid():
    return f"cred_{uuid.uuid4()}"


class Credential(Base):
    __tablename__ = "credential"

    id = Column(
        String(), primary_key=True, default=generate_prefixed_uuid, nullable=False
    )  # format: cred_<uuid>
    user_id = Column(String(), nullable=False)
    name = Column(String(255), nullable=False)
    value = Column(String(2048), nullable=False)

    created_by = Column(String(), nullable=False)
    updated_by = Column(String(), nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)
    created_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        DateTime(timezone=True), default=datetime.utcnow, nullable=False
    )
