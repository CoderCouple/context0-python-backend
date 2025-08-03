""" Memory API with comprehensive CRUD operations using service layer and BaseResponse."""

import logging
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from starlette import status

from app.api.tags import Tags
from app.api.v1.request.memory_request import (
    MemoryRecordInput,
    MemoryUpdate,
    SearchQuery,
)
from app.api.v1.response.base_response import (
    BaseResponse,
    error_response,
    success_response,
)
from app.api.v1.response.memory_response import (
    BulkMemoryResponse,
    HealthResponse,
    MemoryEntry,
    MemoryResponse,
    SearchResponse,
    TimelineResponse,
)
from app.common.auth.auth import UserContext, get_current_user_context
from app.common.auth.role_decorator import require_roles
from app.common.logging.logging import log_memory_operation
from app.db.session import get_db
from app.service.memory_service import MemoryService

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.Memory])


# Dependency injection for memory service
async def get_memory_service() -> MemoryService:
    """Get memory service instance"""
    return MemoryService()


@router.post("/memories", response_model=BaseResponse[MemoryResponse])
async def create_memory(
    record: MemoryRecordInput,
    background_tasks: BackgroundTasks,
    context: UserContext = Depends(get_current_user_context),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """Create a new memory with enhanced processing"""
    try:
        # Override user_id with the authenticated user's ID
        record.user_id = context.user_id

        response = await memory_service.create_memory(record)

        # Add background audit logging
        if response.success:
            background_tasks.add_task(
                log_memory_operation,
                "create",
                response.memory_id,
                context.user_id,
                response.operation,
            )

        if response.success:
            return success_response(
                result=response, message="Memory created successfully", status_code=201
            )
        else:
            return error_response(message=response.message, status_code=400)

    except Exception as e:
        logger.error(f"Create memory endpoint error: {e}")
        return error_response(
            message="Internal server error processing memory", status_code=500
        )


@router.get("/memories/{memory_id}", response_model=BaseResponse[MemoryEntry])
async def get_memory(
    memory_id: str,
    context: UserContext = Depends(get_current_user_context),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """Get a specific memory by ID with access tracking"""
    try:
        memory = await memory_service.get_memory(memory_id, context.user_id)

        if not memory:
            return error_response(message="Memory not found", status_code=404)

        return success_response(result=memory, message="Memory retrieved successfully")

    except PermissionError:
        return error_response(message="Access denied", status_code=403)
    except Exception as e:
        logger.error(f"Get memory endpoint error: {e}")
        return error_response(
            message="Internal server error retrieving memory", status_code=500
        )


@router.get("/memories", response_model=BaseResponse[List[MemoryEntry]])
async def list_memories(
    memory_type: Optional[str] = Query(None, description="Filter by memory type"),
    tags: Optional[List[str]] = Query(None, description="Filter by tags"),
    limit: int = Query(20, ge=1, le=100, description="Number of memories to return"),
    offset: int = Query(0, ge=0, description="Number of memories to skip"),
    sort_by: str = Query("created_at", description="Sort field"),
    sort_order: str = Query("desc", description="Sort order (asc/desc)"),
    context: UserContext = Depends(get_current_user_context),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """List memories with filtering and pagination

    Uses the authenticated user's ID from context.
    """
    try:
        memories = await memory_service.list_memories(
            user_id=context.user_id,
            memory_type=memory_type,
            tags=tags,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
        )

        return success_response(
            result=memories, message=f"Retrieved {len(memories)} memories"
        )

    except ValueError as e:
        return error_response(message=str(e), status_code=400)
    except Exception as e:
        logger.error(f"List memories error: {e}")
        return error_response(
            message="Internal server error listing memories", status_code=500
        )


@router.post("/memories/search", response_model=BaseResponse[SearchResponse])
async def search_memories(
    query: SearchQuery,
    context: UserContext = Depends(get_current_user_context),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """Advanced semantic search for memories with filtering"""
    try:
        # Override user_id with authenticated user's ID
        query.user_id = context.user_id

        response = await memory_service.search_memories(query)

        if response.success:
            return success_response(
                result=response,
                message=f"Search completed: {len(response.results)} results found",
            )
        else:
            return error_response(message="Search failed", status_code=400)

    except ValueError as e:
        return error_response(message=str(e), status_code=400)
    except Exception as e:
        logger.error(f"Search endpoint error: {e}")
        return error_response(
            message="Internal server error during search", status_code=500
        )


@router.put("/memories/{memory_id}", response_model=BaseResponse[dict])
async def update_memory(
    memory_id: str,
    update: MemoryUpdate,
    background_tasks: BackgroundTasks,
    context: UserContext = Depends(get_current_user_context),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """Update an existing memory with validation"""
    try:
        result = await memory_service.update_memory(memory_id, update, context.user_id)

        if result.get("success"):
            background_tasks.add_task(
                log_memory_operation, "update", memory_id, context.user_id, "UPDATE"
            )

            return success_response(
                result=result, message="Memory updated successfully"
            )
        else:
            return error_response(
                message=result.get("message", "Failed to update memory"),
                status_code=500,
            )

    except PermissionError:
        return error_response(message="Access denied", status_code=403)
    except ValueError as e:
        return error_response(
            message=str(e), status_code=404 if "not found" in str(e).lower() else 400
        )
    except Exception as e:
        logger.error(f"Update memory endpoint error: {e}")
        return error_response(
            message="Internal server error updating memory", status_code=500
        )


@router.delete("/memories/{memory_id}", response_model=BaseResponse[dict])
async def delete_memory(
    memory_id: str,
    background_tasks: BackgroundTasks,
    context: UserContext = Depends(get_current_user_context),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """Soft delete a memory with audit trail"""
    try:
        result = await memory_service.delete_memory(memory_id, context.user_id)

        if result.get("success"):
            background_tasks.add_task(
                log_memory_operation, "delete", memory_id, context.user_id, "DELETE"
            )

            return success_response(
                result=result, message="Memory deleted successfully"
            )
        else:
            return error_response(
                message=result.get("message", "Failed to delete memory"),
                status_code=500,
            )

    except PermissionError:
        return error_response(message="Access denied", status_code=403)
    except ValueError as e:
        return error_response(
            message=str(e), status_code=404 if "not found" in str(e).lower() else 400
        )
    except Exception as e:
        logger.error(f"Delete memory endpoint error: {e}")
        return error_response(
            message="Internal server error deleting memory", status_code=500
        )


@router.post("/memories/bulk", response_model=BaseResponse[BulkMemoryResponse])
async def bulk_create_memories(
    records: List[MemoryRecordInput],
    background_tasks: BackgroundTasks,
    context: UserContext = Depends(get_current_user_context),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """Bulk create multiple memories with batch processing"""
    try:
        if len(records) > 50:
            return error_response(
                message="Bulk operations limited to 50 memories per request",
                status_code=400,
            )

        # Override user_id in all records with authenticated user's ID
        for record in records:
            record.user_id = context.user_id

        response = await memory_service.bulk_create_memories(records)

        # Background audit logging for successful operations
        if response.success:
            for result in response.results:
                if result.get("success"):
                    background_tasks.add_task(
                        log_memory_operation,
                        "bulk_create",
                        result.get("memory_id", ""),
                        context.user_id,
                        "BULK_CREATE",
                    )

        if response.success:
            return success_response(
                result=response,
                message=f"Bulk operation completed: {response.successful}/{response.processed} successful",
            )
        else:
            return error_response(message="Bulk operation failed", status_code=400)

    except Exception as e:
        logger.error(f"Bulk create error: {e}")
        return error_response(
            message="Internal server error during bulk operation", status_code=500
        )


@router.post("/memories/time-travel", response_model=BaseResponse[TimelineResponse])
async def time_travel_query(
    target_time: datetime = Query(..., description="Target timestamp"),
    query: Optional[str] = Query(None, description="Optional search query"),
    context: UserContext = Depends(get_current_user_context),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """Query memory state at a specific point in time"""
    try:
        response = await memory_service.time_travel_query(
            context.user_id, target_time, query
        )

        if response.success:
            return success_response(
                result=response,
                message=f"Time travel query completed for {target_time}",
            )
        else:
            return error_response(message="Time travel query failed", status_code=400)

    except ValueError as e:
        return error_response(message=str(e), status_code=400)
    except Exception as e:
        logger.error(f"Time travel query error: {e}")
        return error_response(
            message="Internal server error during time travel query", status_code=500
        )


@router.get(
    "/memories/{memory_id}/evolution", response_model=BaseResponse[TimelineResponse]
)
async def get_memory_evolution(
    memory_id: str,
    start_time: Optional[datetime] = Query(
        None, description="Start time for evolution"
    ),
    end_time: Optional[datetime] = Query(None, description="End time for evolution"),
    context: UserContext = Depends(get_current_user_context),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """Get the evolution history of a specific memory"""
    try:
        response = await memory_service.get_memory_evolution(
            memory_id=memory_id,
            user_id=context.user_id,
            start_time=start_time,
            end_time=end_time,
        )

        if response.success:
            return success_response(
                result=response,
                message=f"Memory evolution retrieved: {response.total_events} events",
            )
        else:
            return error_response(
                message="Failed to retrieve memory evolution", status_code=400
            )

    except PermissionError:
        return error_response(message="Access denied", status_code=403)
    except ValueError as e:
        return error_response(
            message=str(e), status_code=404 if "not found" in str(e).lower() else 400
        )
    except Exception as e:
        logger.error(f"Memory evolution error: {e}")
        return error_response(
            message="Internal server error retrieving memory evolution", status_code=500
        )


@router.get("/stats", response_model=BaseResponse[HealthResponse])
async def get_memory_stats(
    context: UserContext = Depends(get_current_user_context),
    memory_service: MemoryService = Depends(get_memory_service),
):
    """Get comprehensive memory system statistics for authenticated user"""
    try:
        stats = await memory_service.get_system_stats(context.user_id)

        return success_response(
            result=stats, message="System statistics retrieved successfully"
        )

    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        return error_response(
            message="Internal server error retrieving statistics", status_code=500
        )
