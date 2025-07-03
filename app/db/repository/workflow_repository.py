from datetime import datetime

from sqlalchemy.orm import Session

from app.model.workflow.workflow_model import Workflow


class WorkflowRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_id(self, workflow_id: str) -> Workflow | None:
        return (
            self.db.query(Workflow)
            .filter(Workflow.id == workflow_id, Workflow.is_deleted == False)
            .first()
        )

    def list_by_user(self, user_id: str) -> list[Workflow]:
        return (
            self.db.query(Workflow)
            .filter(Workflow.user_id == user_id, Workflow.is_deleted == False)
            .all()
        )

    def create(self, workflow: Workflow) -> Workflow:
        self.db.add(workflow)
        self.db.commit()
        self.db.refresh(workflow)
        return workflow

    def update(self, workflow: Workflow) -> Workflow:
        self.db.commit()
        self.db.refresh(workflow)
        return workflow

    def delete(
        self, workflow: Workflow, is_soft_delete: bool, user_id: str
    ) -> Workflow:
        if is_soft_delete:
            workflow.is_deleted = True
            workflow.updated_by = user_id
            workflow.updated_at = datetime.utcnow()
        else:
            self.db.delete(workflow)
        self.db.commit()
        return workflow
