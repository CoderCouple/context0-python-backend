# from fastapi import HTTPException
# from sqlalchemy.orm import Session, selectinload
#
# from app.api.v1.response.workflow_execution_response import WorkflowExecutionWithPhasesResponse, \
#     WorkflowPhaseWithLogsResponse, WorkflowExecutionListResponse, WorkflowExecutionResponse
# from app.common.auth import UserContext
# from app.model.execution_phase_model import ExecutionPhase
# from app.model.workflow_execution_model import WorkflowExecution
#
#
# class WorkflowExecutionService:
#     def __init__(self, db: Session, context: UserContext):
#         self.db = db
#         self.context = context
#
#     def fetch_executions(self, workflow_id: str) -> WorkflowExecutionListResponse:
#         executions = (
#             self.db.query(WorkflowExecution)
#             .filter(
#                 WorkflowExecution.workflow_id == workflow_id,
#                 WorkflowExecution.user_id == self.context.user_id,
#                 WorkflowExecution.is_deleted == False,
#             )
#             .order_by(WorkflowExecution.created_at.desc())
#             .all()
#         )
#         return WorkflowExecutionListResponse.from_orm_list(executions)
#
#     def fetch_execution_with_phases(self, execution_id: str) -> WorkflowExecutionWithPhasesResponse:
#         execution = (
#             self.db.query(WorkflowExecution)
#             .filter(
#                 WorkflowExecution.id == execution_id,
#                 WorkflowExecution.user_id == self.context.user_id,
#             )
#             .options(selectinload(WorkflowExecution.phases))
#             .first()
#         )
#         if not execution:
#             raise HTTPException(status_code=404, detail="Execution not found")
#         return WorkflowExecutionWithPhasesResponse.from_orm(execution)
#
#     def fetch_phase_with_logs(self, phase_id: str) -> WorkflowPhaseWithLogsResponse:
#         phase = (
#             self.db.query(ExecutionPhase)
#             .filter(ExecutionPhase.id == phase_id)
#             .options(selectinload(ExecutionPhase.logs))
#             .first()
#         )
#         if not phase:
#             raise HTTPException(status_code=404, detail="Phase not found")
#
#         # Ensure user owns the parent workflow execution
#         if phase.execution.user_id != self.context.user_id:
#             raise HTTPException(status_code=403, detail="Access denied")
#
#         return WorkflowPhaseWithLogsResponse.from_orm(phase)
#
#     def trigger_executions(self, workflow_id: str) -> WorkflowExecutionResponse:
#         workflow = self.repo.get(workflow_id)
#
#         if not workflow or workflow.is_deleted:
#             raise ValueError("Workflow not found or is deleted")
#
#         # Generate plan if not stored
#         if not workflow.execution_plan:
#             parser = ExecutionPlanBuilder(nodes=workflow.definition["nodes"], edges=workflow.definition["edges"])
#             result = parser.generate()
#
#             if "error" in result:
#                 raise ValueError(f"Workflow validation failed: {result['error']}")
#
#             workflow.execution_plan = result["executionPlan"]
#             self.db.commit()
#
#         # Create workflow run record
#         run_id = str(uuid4())
#         run = WorkflowRun(
#             id=run_id,
#             workflow_id=workflow_id,
#             status="PENDING",
#             triggered_by=self.context.user_id,
#             started_at=datetime.utcnow(),
#             created_by=self.context.user_id,
#             updated_by=self.context.user_id,
#         )
#         self.db.add(run)
#         self.db.commit()
#
#         # Enqueue all nodes in phase 1
#         phase_1 = next((p for p in workflow.execution_plan if p["phase"] == 1), None)
#         if not phase_1:
#             raise ValueError("No entry-point phase found in execution plan")
#
#         for node in phase_1["nodes"]:
#             QueueService.enqueue(workflow_id=workflow_id, node_id=node["id"], run_id=run_id)
#
#         return WorkflowExecutionResponse(
#             id=run_id,
#             workflow_id=workflow_id,
#             status=run.status,
#             started_at=run.started_at
#         )
