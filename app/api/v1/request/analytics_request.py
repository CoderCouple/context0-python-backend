"""Analytics API request models"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class PeriodRequest(BaseModel):
    """Request model for period-based analytics"""

    year: int = Field(..., description="Year for analytics data", ge=2020, le=2030)
    month: int = Field(..., description="Month for analytics data", ge=1, le=12)


class AnalyticsFilterRequest(BaseModel):
    """Base analytics filter request"""

    year: int = Field(..., description="Year for analytics data", ge=2020, le=2030)
    month: int = Field(..., description="Month for analytics data", ge=1, le=12)
    user_id: Optional[str] = Field(None, description="Filter by specific user ID")
    workflow_id: Optional[str] = Field(
        None, description="Filter by specific workflow ID"
    )


class DateRangeRequest(BaseModel):
    """Request model for date range analytics"""

    start_date: datetime = Field(..., description="Start date for analytics")
    end_date: datetime = Field(..., description="End date for analytics")
    user_id: Optional[str] = Field(None, description="Filter by specific user ID")
    workflow_id: Optional[str] = Field(
        None, description="Filter by specific workflow ID"
    )
