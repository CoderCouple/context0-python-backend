# """Workflow Execution API."""
#
# import logging
#
# from fastapi import APIRouter, Depends
# from sqlalchemy.orm import Session
#
# from app.api.base_response import BaseResponse, success_response, error_response
# from app.api.tags import Tags
# from app.api.v1.response.workflow_execution_response import WorkflowExecutionWithPhasesResponse, \
#     WorkflowPhaseWithLogsResponse, WorkflowExecutionListResponse, WorkflowExecutionResponse
# from app.common.auth import UserContext
# from app.common.role_decorator import require_roles
# from app.db.session import get_db
# from app.service.workflow_execution_service import WorkflowExecutionService
#
# logger = logging.getLogger(__name__)
#
# router = APIRouter(tags=[Tags.Workflow_Execution])
#
#
# @router.post("/workflow/{workflow_id}/execute", response_model=BaseResponse[WorkflowExecutionResponse])
# def execute_workflow(
#         workflow_id: str,
#         db: Session = Depends(get_db),
#         context: UserContext = Depends(require_roles(["admin", "builder"])),
# ):
#     try:
#         result = WorkflowExecutionService(db, context).trigger_execution(workflow_id)
#         return success_response(result, "Workflow execution started")
#     except Exception as e:
#         logger.exception("Workflow execution failed", extra={"workflow_id": workflow_id})
#         return error_response(message=f"Execution failed: {str(e)}", status_code=500)
#
#
# @router.get("/workflow/{workflow_id}/executions", response_model=BaseResponse[WorkflowExecutionListResponse])
# def get_workflow_executions(
#         workflow_id: str,
#         db: Session = Depends(get_db),
#         context: UserContext = Depends(require_roles(["admin", "builder"])),
# ):
#     executions = WorkflowExecutionService(db, context).fetch_executions(workflow_id)
#     return success_response(executions, "Executions fetched successfully")
#
#
# @router.get("/execution/{execution_id}", response_model=BaseResponse[WorkflowExecutionWithPhasesResponse])
# def get_workflow_execution_with_phases(
#         execution_id: str,
#         db: Session = Depends(get_db),
#         context: UserContext = Depends(require_roles(["admin", "builder"])),
# ):
#     execution = WorkflowExecutionService(db, context).fetch_execution_with_phases(execution_id)
#     return success_response(execution, "Execution with phases fetched")
#
#
# @router.get("/phase/{phase_id}", response_model=BaseResponse[WorkflowPhaseWithLogsResponse])
# def get_phase_details(
#         phase_id: str,
#         db: Session = Depends(get_db),
#         context: UserContext = Depends(require_roles(["admin", "builder"])),
# ):
#     phase = WorkflowExecutionService(db, context).fetch_phase_with_logs(phase_id)
#     return success_response(phase, "Phase with logs fetched")
