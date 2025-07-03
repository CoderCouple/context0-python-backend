import uuid
from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Enum, ForeignKey, Numeric, String
from sqlalchemy.orm import relationship

from app.common.enums import ExecutionStatus, ExecutionTrigger
from app.db.base import Base


def generate_prefixed_uuid():
    return f"exec_{uuid.uuid4()}"


class WorkflowExecution(Base):
    __tablename__ = "workflow_execution"

    id = Column(
        String(), primary_key=True, default=generate_prefixed_uuid, nullable=False
    )  # format: exec_<uuid>
    workflow_id = Column(String, ForeignKey("workflow.id"), nullable=False)
    user_id = Column(String, nullable=False)
    trigger = Column(
        Enum(ExecutionTrigger), nullable=False, default=ExecutionTrigger.MANUAL
    )
    status = Column(
        Enum(ExecutionStatus), nullable=False, default=ExecutionStatus.PENDING
    )
    credits_consumed = Column(Numeric)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime)
    completed_at = Column(DateTime)

    created_by = Column(String, nullable=False)
    updated_by = Column(String, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    is_deleted = Column(Boolean, default=False, nullable=False)

    phases = relationship("ExecutionPhase", back_populates="execution")
