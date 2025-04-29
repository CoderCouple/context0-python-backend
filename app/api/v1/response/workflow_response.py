from datetime import datetime
from typing import List, Optional

from pydantic import UUID4, BaseModel, RootModel, condecimal

from app.model.workflow_model import Workflow


class WorkflowResponse(BaseModel):
    id: UUID4
    user_id: UUID4
    name: Optional[str] = None
    description: Optional[str] = None
    definition: Optional[str] = None
    execution_plan: Optional[str] = None
    cron: Optional[str] = None
    status: str
    credits_cost: Optional[condecimal(max_digits=10, decimal_places=2)] = None
    last_run_at: Optional[datetime] = None
    last_run_id: Optional[UUID4] = None
    last_run_status: Optional[str] = None
    next_run_at: Optional[datetime] = None
    created_by: UUID4
    updated_by: UUID4
    created_at: datetime
    updated_at: datetime
    is_deleted: bool

    model_config = {"from_attributes": True}  # ✅ replaces orm_mode=True


class WorkflowListResponse(RootModel[List[WorkflowResponse]]):
    @classmethod
    def from_orm_list(cls, workflows: list["Workflow"]) -> "WorkflowListResponse":
        return cls([WorkflowResponse.from_orm(w) for w in workflows])


class WorkflowListResult(BaseModel):
    result: List[WorkflowResponse]

    @classmethod
    def from_orm_list(cls, workflows: list["Workflow"]) -> "WorkflowListResult":
        return cls(result=[WorkflowResponse.from_orm(w) for w in workflows])
