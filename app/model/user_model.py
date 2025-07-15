import uuid
from datetime import datetime

from sqlalchemy import TIMESTAMP, Boolean, Column
from sqlalchemy import Enum as SqlEnum
from sqlalchemy import String

from app.common.enum.user import UserRole
from app.db.base import Base


def generate_prefixed_uuid():
    return f"user_{uuid.uuid4()}"


class User(Base):
    __tablename__ = "user"

    id = Column(
        String(), primary_key=True, default=generate_prefixed_uuid, nullable=False
    )  # format: user_<uuid>
    clerk_user_id = Column(String(64), unique=True, nullable=False)  # Clerk's user ID

    name = Column(String(256), nullable=False)
    email = Column(String(320), nullable=False, unique=True)
    password = Column(String(256), nullable=False)
    role = Column(
        SqlEnum(UserRole, name="user_role_enum"), nullable=False, default=UserRole.USER
    )
    phone = Column(String(256), nullable=True)
    email_verified = Column(TIMESTAMP(timezone=True), nullable=True)
    avatar = Column(String(2048), nullable=False)

    organization_id = Column(String(64), nullable=True)
    clerk_organization_id = Column(String(64), nullable=True)  # Clerk's organization ID

    is_deleted = Column(Boolean, default=False, nullable=False)
    created_at = Column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )
    updated_at = Column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, nullable=False
    )
