from datetime import datetime
from typing import Optional

from pydantic import UUID4, BaseModel, condecimal, constr

from app.common.enums import WorkflowStatus


class CreateWorkflowRequest(BaseModel):
    name: constr(max_length=255)
    description: Optional[constr(max_length=1024)] = None
    status: constr(max_length=50) = WorkflowStatus.DRAFT


class UpdateWorkflowRequest(BaseModel):
    name: Optional[constr(max_length=255)] = None
    description: Optional[constr(max_length=1024)] = None
    definition: Optional[str] = None
    execution_plan: Optional[str] = None
    cron: Optional[constr(max_length=100)] = None
    status: Optional[constr(max_length=50)] = None
    credits_cost: Optional[condecimal(max_digits=10, decimal_places=2)] = None
    last_run_at: Optional[datetime] = None
    last_run_id: Optional[UUID4] = None
    last_run_status: Optional[constr(max_length=50)] = None
    next_run_at: Optional[datetime] = None
    updated_by: Optional[UUID4] = None


class DeleteWorkflowRequest(BaseModel):
    id: UUID4
    soft: bool = True
