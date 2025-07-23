"""Analytics Service for generating usage statistics and metrics"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from app.api.v1.request.analytics_request import (
    AnalyticsFilterRequest,
    DateRangeRequest,
)
from app.api.v1.response.analytics_response import (
    AnalyticsHealthResponse,
    CreditUsageResponse,
    CreditUsageStats,
    Period,
    PeriodsResponse,
    StatsCardValues,
    StatsCardsResponse,
    WorkflowExecutionStats,
    WorkflowExecutionStatsResponse,
)

logger = logging.getLogger(__name__)


class AnalyticsService:
    """Service for handling analytics operations"""

    def __init__(self):
        """Initialize analytics service"""
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes cache

    async def get_credit_usage_stats(
        self, year: int, month: int, user_id: Optional[str] = None
    ) -> CreditUsageResponse:
        """Get credit usage statistics for a specific period"""
        try:
            logger.info(f"Getting credit usage stats for {year}-{month:02d}")

            # Generate period info
            period = self._create_period(year, month)

            # Generate sample data for demonstration
            # In production, this would query your actual database
            stats_data = await self._generate_credit_usage_data(year, month, user_id)

            total_success = sum(stat.success for stat in stats_data)
            total_failed = sum(stat.failed for stat in stats_data)

            return CreditUsageResponse(
                data=stats_data,
                period=period,
                total_success=total_success,
                total_failed=total_failed,
            )

        except Exception as e:
            logger.error(f"Error getting credit usage stats: {e}")
            raise

    async def get_stats_cards_values(
        self, year: int, month: int, user_id: Optional[str] = None
    ) -> StatsCardsResponse:
        """Get values for stats cards dashboard"""
        try:
            logger.info(f"Getting stats cards values for {year}-{month:02d}")

            # Generate period info
            period = self._create_period(year, month)

            # Generate sample data for demonstration
            # In production, this would query your actual database
            stats_values = await self._generate_stats_card_data(year, month, user_id)

            return StatsCardsResponse(data=stats_values, period=period)

        except Exception as e:
            logger.error(f"Error getting stats cards values: {e}")
            raise

    async def get_workflow_execution_stats(
        self, year: int, month: int, user_id: Optional[str] = None
    ) -> WorkflowExecutionStatsResponse:
        """Get workflow execution statistics for a specific period"""
        try:
            logger.info(f"Getting workflow execution stats for {year}-{month:02d}")

            # Generate period info
            period = self._create_period(year, month)

            # Generate sample data for demonstration
            # In production, this would query your actual database
            stats_data = await self._generate_workflow_execution_data(
                year, month, user_id
            )

            total_success = sum(stat.success for stat in stats_data)
            total_failed = sum(stat.failed for stat in stats_data)

            return WorkflowExecutionStatsResponse(
                data=stats_data,
                period=period,
                total_success=total_success,
                total_failed=total_failed,
            )

        except Exception as e:
            logger.error(f"Error getting workflow execution stats: {e}")
            raise

    async def get_available_periods(
        self, user_id: Optional[str] = None
    ) -> PeriodsResponse:
        """Get available periods for analytics"""
        try:
            logger.info("Getting available periods for analytics")

            # Generate available periods (last 12 months)
            current_date = datetime.now()
            periods = []

            for i in range(12):
                date = current_date - timedelta(days=i * 30)
                period = self._create_period(date.year, date.month)
                periods.append(period)

            # Current period
            current_period = self._create_period(current_date.year, current_date.month)

            return PeriodsResponse(data=periods, current_period=current_period)

        except Exception as e:
            logger.error(f"Error getting available periods: {e}")
            raise

    async def get_system_health(self) -> AnalyticsHealthResponse:
        """Get analytics system health information"""
        try:
            return AnalyticsHealthResponse(
                status="healthy",
                data_freshness=datetime.now() - timedelta(minutes=5),
                available_periods=12,
                total_records=10000,  # Sample value
            )
        except Exception as e:
            logger.error(f"Error getting system health: {e}")
            raise

    def _create_period(self, year: int, month: int) -> Period:
        """Create a Period object"""
        month_names = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        label = f"{month_names[month - 1]} {year}"
        return Period(year=year, month=month, label=label)

    async def _generate_credit_usage_data(
        self, year: int, month: int, user_id: Optional[str] = None
    ) -> List[CreditUsageStats]:
        """Generate sample credit usage data"""
        import random
        from calendar import monthrange

        days_in_month = monthrange(year, month)[1]
        stats = []

        for day in range(1, days_in_month + 1):
            date_str = f"{year}-{month:02d}-{day:02d}"

            # Generate realistic sample data
            success = random.randint(50, 200)
            failed = random.randint(5, 30)

            stats.append(
                CreditUsageStats(date=date_str, success=success, failed=failed)
            )

        return stats

    async def _generate_stats_card_data(
        self, year: int, month: int, user_id: Optional[str] = None
    ) -> StatsCardValues:
        """Generate sample stats card data"""
        import random

        return StatsCardValues(
            workflow_executions=random.randint(1000, 5000),
            credits_consumed=random.randint(10000, 50000),
            phase_executions=random.randint(5000, 25000),
        )

    async def _generate_workflow_execution_data(
        self, year: int, month: int, user_id: Optional[str] = None
    ) -> List[WorkflowExecutionStats]:
        """Generate sample workflow execution data"""
        import random
        from calendar import monthrange

        days_in_month = monthrange(year, month)[1]
        stats = []

        for day in range(1, days_in_month + 1):
            date_str = f"{year}-{month:02d}-{day:02d}"

            # Generate realistic sample data
            success = random.randint(100, 500)
            failed = random.randint(10, 50)

            stats.append(
                WorkflowExecutionStats(date=date_str, success=success, failed=failed)
            )

        return stats
