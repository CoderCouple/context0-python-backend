import uuid
from datetime import datetime

from sqlalchemy import JSON, TIMESTAMP, Boolean, Column, String

from app.db.base import Base


def generate_prefixed_uuid():
    return f"org_{uuid.uuid4()}"


class Organization(Base):
    __tablename__ = "organization"

    id = Column(
        String(), primary_key=True, default=generate_prefixed_uuid, nullable=False
    )  # format: org_<uuid>
    clerk_organization_id = Column(
        String(64), unique=True, nullable=False
    )  # Clerk's organization ID

    name = Column(String(255), nullable=False)
    slug = Column(String(255), unique=True, nullable=True)
    logo_url = Column(String(2048), nullable=True)
    public_metadata = Column(JSON, nullable=True)
    private_metadata = Column(JSON, nullable=True)

    created_at = Column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )
    is_deleted = Column(Boolean, default=False, nullable=False)
