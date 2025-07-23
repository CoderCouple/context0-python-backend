"""Analytics API response models"""

from datetime import datetime
from typing import List

from pydantic import BaseModel, Field


class Period(BaseModel):
    """Period model for analytics"""

    year: int = Field(..., description="Year")
    month: int = Field(..., description="Month")
    label: str = Field(..., description="Human readable label (e.g., 'January 2024')")


class CreditUsageStats(BaseModel):
    """Credit usage statistics for a specific date"""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    success: int = Field(..., description="Number of successful credit uses")
    failed: int = Field(..., description="Number of failed credit uses")


class StatsCardValues(BaseModel):
    """Values for stats cards dashboard"""

    workflow_executions: int = Field(..., description="Total workflow executions")
    credits_consumed: int = Field(..., description="Total credits consumed")
    phase_executions: int = Field(..., description="Total phase executions")


class WorkflowExecutionStats(BaseModel):
    """Workflow execution statistics for a specific date"""

    date: str = Field(..., description="Date in YYYY-MM-DD format")
    success: int = Field(..., description="Number of successful executions")
    failed: int = Field(..., description="Number of failed executions")


class CreditUsageResponse(BaseModel):
    """Response for credit usage analytics"""

    data: List[CreditUsageStats] = Field(..., description="Credit usage statistics")
    period: Period = Field(..., description="Period for the data")
    total_success: int = Field(..., description="Total successful uses in period")
    total_failed: int = Field(..., description="Total failed uses in period")


class StatsCardsResponse(BaseModel):
    """Response for stats cards"""

    data: StatsCardValues = Field(..., description="Stats card values")
    period: Period = Field(..., description="Period for the data")


class WorkflowExecutionStatsResponse(BaseModel):
    """Response for workflow execution analytics"""

    data: List[WorkflowExecutionStats] = Field(
        ..., description="Workflow execution statistics"
    )
    period: Period = Field(..., description="Period for the data")
    total_success: int = Field(..., description="Total successful executions in period")
    total_failed: int = Field(..., description="Total failed executions in period")


class PeriodsResponse(BaseModel):
    """Response for available periods"""

    data: List[Period] = Field(..., description="Available periods for analytics")
    current_period: Period = Field(..., description="Current period")


class AnalyticsHealthResponse(BaseModel):
    """Analytics system health response"""

    status: str = Field(..., description="System status")
    data_freshness: datetime = Field(..., description="Last data update timestamp")
    available_periods: int = Field(..., description="Number of available periods")
    total_records: int = Field(..., description="Total analytics records")
