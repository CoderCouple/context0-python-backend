"""Analytics API Controller for usage statistics and metrics"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.tags import Tags
from app.api.v1.response.analytics_response import (
    AnalyticsHealthResponse,
    CreditUsageResponse,
    PeriodsResponse,
    StatsCardsResponse,
    WorkflowExecutionStatsResponse,
)
from app.api.v1.response.base_response import (
    BaseResponse,
    error_response,
    success_response,
)
from app.common.auth.auth import UserContext, get_current_user_context
from app.db.session import get_db
from app.service.analytics_service import AnalyticsService

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.Analytics])


# Dependency injection for analytics service
async def get_analytics_service() -> AnalyticsService:
    """Get analytics service instance"""
    return AnalyticsService()


@router.get("/analytics/credit-usage", response_model=BaseResponse[CreditUsageResponse])
async def get_credit_usage_stats(
    year: int = Query(..., description="Year for analytics data", ge=2020, le=2030),
    month: int = Query(..., description="Month for analytics data", ge=1, le=12),
    user_id: Optional[str] = Query(
        None, description="User ID (defaults to context user)"
    ),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    db: Session = Depends(get_db),
    context: UserContext = Depends(get_current_user_context),
):
    """Get credit usage statistics for a specific period"""
    try:
        # Use provided user_id or fall back to context user_id
        actual_user_id = user_id or context.user_id
        logger.info(
            f"Getting credit usage stats for {year}-{month:02d} for user {actual_user_id}"
        )

        response = await analytics_service.get_credit_usage_stats(
            year=year, month=month, user_id=actual_user_id
        )

        return success_response(
            result=response,
            message=f"Credit usage statistics retrieved for {year}-{month:02d}",
        )

    except ValueError as e:
        return error_response(message=str(e), status_code=400)
    except Exception as e:
        logger.error(f"Credit usage stats endpoint error: {e}")
        return error_response(
            message="Internal server error retrieving credit usage statistics",
            status_code=500,
        )


@router.get("/analytics/stats-cards", response_model=BaseResponse[StatsCardsResponse])
async def get_stats_cards_values(
    year: int = Query(..., description="Year for analytics data", ge=2020, le=2030),
    month: int = Query(..., description="Month for analytics data", ge=1, le=12),
    user_id: Optional[str] = Query(
        None, description="User ID (defaults to context user)"
    ),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    db: Session = Depends(get_db),
    context: UserContext = Depends(get_current_user_context),
):
    """Get values for stats cards dashboard"""
    try:
        # Use provided user_id or fall back to context user_id
        actual_user_id = user_id or context.user_id
        logger.info(
            f"Getting stats cards values for {year}-{month:02d} for user {actual_user_id}"
        )

        response = await analytics_service.get_stats_cards_values(
            year=year, month=month, user_id=actual_user_id
        )

        return success_response(
            result=response,
            message=f"Stats cards values retrieved for {year}-{month:02d}",
        )

    except ValueError as e:
        return error_response(message=str(e), status_code=400)
    except Exception as e:
        logger.error(f"Stats cards endpoint error: {e}")
        return error_response(
            message="Internal server error retrieving stats cards values",
            status_code=500,
        )


@router.get(
    "/analytics/workflow-execution-stats",
    response_model=BaseResponse[WorkflowExecutionStatsResponse],
)
async def get_workflow_execution_stats(
    year: int = Query(..., description="Year for analytics data", ge=2020, le=2030),
    month: int = Query(..., description="Month for analytics data", ge=1, le=12),
    user_id: Optional[str] = Query(
        None, description="User ID (defaults to context user)"
    ),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    db: Session = Depends(get_db),
    context: UserContext = Depends(get_current_user_context),
):
    """Get workflow execution statistics for a specific period"""
    try:
        # Use provided user_id or fall back to context user_id
        actual_user_id = user_id or context.user_id
        logger.info(
            f"Getting workflow execution stats for {year}-{month:02d} for user {actual_user_id}"
        )

        response = await analytics_service.get_workflow_execution_stats(
            year=year, month=month, user_id=actual_user_id
        )

        return success_response(
            result=response,
            message=f"Workflow execution statistics retrieved for {year}-{month:02d}",
        )

    except ValueError as e:
        return error_response(message=str(e), status_code=400)
    except Exception as e:
        logger.error(f"Workflow execution stats endpoint error: {e}")
        return error_response(
            message="Internal server error retrieving workflow execution statistics",
            status_code=500,
        )


@router.get("/analytics/periods", response_model=BaseResponse[PeriodsResponse])
async def get_available_periods(
    user_id: Optional[str] = Query(
        None, description="User ID (defaults to context user)"
    ),
    analytics_service: AnalyticsService = Depends(get_analytics_service),
    db: Session = Depends(get_db),
    context: UserContext = Depends(get_current_user_context),
):
    """Get available periods for analytics"""
    try:
        # Use provided user_id or fall back to context user_id
        actual_user_id = user_id or context.user_id
        logger.info(
            f"Getting available periods for analytics for user {actual_user_id}"
        )

        response = await analytics_service.get_available_periods(user_id=actual_user_id)

        return success_response(
            result=response,
            message="Available periods retrieved successfully",
        )

    except Exception as e:
        logger.error(f"Available periods endpoint error: {e}")
        return error_response(
            message="Internal server error retrieving available periods",
            status_code=500,
        )


@router.get("/analytics/health", response_model=BaseResponse[AnalyticsHealthResponse])
async def get_analytics_health(
    analytics_service: AnalyticsService = Depends(get_analytics_service),
):
    """Get analytics system health information"""
    try:
        logger.info("Getting analytics system health")

        response = await analytics_service.get_system_health()

        return success_response(
            result=response,
            message="Analytics system health retrieved successfully",
        )

    except Exception as e:
        logger.error(f"Analytics health endpoint error: {e}")
        return error_response(
            message="Internal server error retrieving system health",
            status_code=500,
        )
