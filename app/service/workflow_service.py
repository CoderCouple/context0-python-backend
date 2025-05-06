from datetime import datetime
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.api.v1.request.workflow_request import CreateWorkflowRequest
from app.api.v1.response.workflow_response import WorkflowListResponse, WorkflowResponse
from app.common.enums import WorkflowStatus
from app.core.auth import UserContext
from app.model.workflow_model import Workflow


class WorkflowService:
    def __init__(self, db: Session, context: UserContext):
        self.db = db
        self.context = context

    def fetch_workflow_by_id(self, workflow_id: str) -> WorkflowResponse:
        workflow = (
            self.db.query(Workflow)
            .filter(Workflow.id == workflow_id, not Workflow.is_deleted)
            .first()
        )

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # Optional org-level access check
        # if workflow.organization_id != self.context.organization_id:
        #     raise HTTPException(status_code=403, detail="Access denied")

        return WorkflowResponse.from_orm(workflow)

    def fetch_workflows(self) -> WorkflowListResponse:
        workflows = (
            self.db.query(Workflow)
            .filter(
                Workflow.user_id == self.context.user_id,
                not Workflow.is_deleted,
            )
            .all()
        )

        if not workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")

        return WorkflowListResponse.from_orm_list(workflows)

    def create_workflow(self, body: CreateWorkflowRequest) -> WorkflowResponse:
        if body.status not in [
            workflow_status.value for workflow_status in WorkflowStatus
        ]:
            raise HTTPException(status_code=400, detail="Invalid status")

        workflow = Workflow(
            id=f"workflow_{uuid4()}",
            user_id=self.context.user_id,
            # organization_id=self.context.organization_id,
            created_by=self.context.user_id,
            updated_by=self.context.user_id,
            **body.dict(),
        )
        self.db.add(workflow)
        self.db.commit()
        self.db.refresh(workflow)

        return WorkflowResponse.from_orm(workflow)

    def delete_workflow(
        self, workflow_id: str, is_soft_delete: bool
    ) -> WorkflowResponse:
        workflow = self.db.query(Workflow).filter_by(id=workflow_id).first()

        if not workflow:
            raise HTTPException(status_code=404, detail="Workflow not found")

        # if workflow.organization_id != self.context.organization_id:
        #     raise HTTPException(
        #         status_code=403,
        #         detail="Access denied: not your organization"
        #     )

        if is_soft_delete:
            workflow.is_deleted = True
            workflow.updated_by = self.context.user_id
            workflow.updated_at = datetime.utcnow()
        else:
            self.db.delete(workflow)

        self.db.commit()
        return WorkflowResponse.from_orm(workflow)
