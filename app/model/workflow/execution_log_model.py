import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, String

from app.db.base import Base


def generate_prefixed_uuid():
    return f"cred_{uuid.uuid4()}"


class ExecutionLog(Base):
    __tablename__ = "execution_log"

    id = Column(
        String(), primary_key=True, default=generate_prefixed_uuid, nullable=False
    )  # format: execlog_<uuid>
    execution_phase_id = Column(String, nullable=False)
    log_level = Column(String(20), nullable=False)
    message = Column(String(2048), nullable=False)
    timestamp = Column(DateTime, nullable=False)

    created_by = Column(String, nullable=False)
    updated_by = Column(String, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)
