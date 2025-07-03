import uuid
from datetime import datetime
from decimal import Decimal

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.common.enums import ExecutionPhaseStatus
from app.db.base import Base


def generate_prefixed_uuid():
    return f"phase_{uuid.uuid4()}"


class ExecutionPhase(Base):
    __tablename__ = "execution_phase"

    id = Column(
        String(), primary_key=True, default=generate_prefixed_uuid, nullable=False
    )  # format: phase_<uuid>
    user_id = Column(String, nullable=False)
    workflow_execution_id = Column(
        String, ForeignKey("workflow_execution.id"), nullable=False
    )

    status = Column(
        Enum(ExecutionPhaseStatus), default=ExecutionPhaseStatus.CREATED, nullable=False
    )
    number = Column(Integer, nullable=False)
    node = Column(String(255))
    name = Column(String(255))
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    inputs = Column(Text)
    outputs = Column(Text)
    credits_consumed = Column(Numeric)

    created_by = Column(String, nullable=False)
    updated_by = Column(String, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    is_deleted = Column(Boolean, default=False, nullable=False)

    logs = relationship("ExecutionLog", backref="phase")
