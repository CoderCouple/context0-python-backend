""" Ping API."""

import logging
import time

from fastapi import APIRouter, Depends

from app.api.tags import Tags
from app.api.v1.response.base_response import BaseResponse, success_response
from app.api.v1.response.ping_response import HealthResponse
from app.common.startup import app_start_time
from app.service.memory_service import MemoryService

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.Ping])


# Dependency injection for memory service
async def get_memory_service() -> MemoryService:
    """Get memory service instance"""
    return MemoryService()


@router.get("/ping", response_model=BaseResponse[dict])
async def ping():
    """Basic ping endpoint"""
    result = {
        "status": "OK",
        "timestamp": time.time(),
        "uptime_seconds": time.time() - app_start_time,
    }

    return success_response(
        result=result, message="Context0 Backend API is running. Alive and healthy!"
    )


@router.get("/health", response_model=BaseResponse[HealthResponse])
async def health_check(memory_service: MemoryService = Depends(get_memory_service)):
    """Health check endpoint using memory service"""
    try:
        health_response = await memory_service.get_system_stats()

        return success_response(
            result=health_response, message="System health check completed"
        )

    except Exception as e:
        logger.error(f"Health check error: {e}")
        # Return degraded health status on error
        degraded_health = HealthResponse(
            status="error",
            version="1.0.0",
            uptime_seconds=time.time() - app_start_time,
            stores={},
            memory_count=0,
            processing_stats={},
        )

        return success_response(
            result=degraded_health, message="Health check completed with errors"
        )
