"""LLM Configuration API endpoints"""
import time
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.v1.request.llm_request import (
    CreateLLMPresetRequest,
    UpdateLLMPresetRequest,
    TestLLMPresetRequest,
    SetDefaultPresetRequest,
)
from app.api.v1.response.base_response import BaseResponse
from app.api.v1.response.llm_response import (
    LLMPresetResponse,
    CreateLLMPresetResponse,
    ProviderInfoResponse,
    TestLLMPresetResponse,
    LLMUsageStatsResponse,
)
from app.common.auth.auth import UserContext, get_current_user_context
from app.db.mongodb import get_database
from app.service.llm_service import LLMService
from app.llm.configs import LLMPresetConfig

router = APIRouter()


def get_llm_service(db: AsyncIOMotorDatabase = Depends(get_database)) -> LLMService:
    """Get LLM service instance"""
    return LLMService(db)


@router.get("/llm/providers", response_model=BaseResponse[List[ProviderInfoResponse]])
async def get_available_providers(
    context: UserContext = Depends(get_current_user_context),
):
    """Get information about available LLM providers"""
    providers_info = LLMService.get_available_providers()

    providers = [
        ProviderInfoResponse(provider=key, **value)
        for key, value in providers_info.items()
    ]

    return BaseResponse(
        success=True,
        message=f"Retrieved {len(providers)} providers",
        result=providers,
        status_code=200,
    )


@router.get("/llm/presets", response_model=BaseResponse[List[LLMPresetResponse]])
async def get_llm_presets(
    include_shared: bool = Query(True, description="Include shared presets"),
    context: UserContext = Depends(get_current_user_context),
    llm_service: LLMService = Depends(get_llm_service),
):
    """Get user's LLM presets"""
    try:
        presets = await llm_service.get_user_presets(context.user_id)

        # Filter out shared presets if not requested
        if not include_shared:
            presets = [p for p in presets if p.user_id == context.user_id]

        preset_responses = [LLMPresetResponse(**preset.dict()) for preset in presets]

        return BaseResponse(
            success=True,
            message=f"Retrieved {len(presets)} presets",
            result=preset_responses,
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/presets", response_model=BaseResponse[CreateLLMPresetResponse])
async def create_llm_preset(
    request: CreateLLMPresetRequest,
    context: UserContext = Depends(get_current_user_context),
    llm_service: LLMService = Depends(get_llm_service),
):
    """Create a new LLM preset"""
    try:
        # Convert request to config
        preset_config = LLMPresetConfig(**request.dict())

        # Create preset
        preset = await llm_service.create_preset(
            user_id=context.user_id, preset_config=preset_config
        )

        response = CreateLLMPresetResponse(
            preset_id=preset.id,
            name=preset.name,
            provider=preset.provider,
            model=preset.model,
            created_at=preset.created_at,
        )

        return BaseResponse(
            success=True,
            message="LLM preset created successfully",
            result=response,
            status_code=200,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm/presets/{preset_id}", response_model=BaseResponse[LLMPresetResponse])
async def get_llm_preset(
    preset_id: str,
    context: UserContext = Depends(get_current_user_context),
    llm_service: LLMService = Depends(get_llm_service),
):
    """Get a specific LLM preset"""
    try:
        preset = await llm_service.get_preset(preset_id, context.user_id)

        if not preset:
            raise HTTPException(status_code=404, detail="Preset not found")

        return BaseResponse(
            success=True,
            message="Preset retrieved successfully",
            result=LLMPresetResponse(**preset.dict()),
            status_code=200,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/llm/presets/{preset_id}", response_model=BaseResponse[dict])
async def update_llm_preset(
    preset_id: str,
    request: UpdateLLMPresetRequest,
    context: UserContext = Depends(get_current_user_context),
    llm_service: LLMService = Depends(get_llm_service),
):
    """Update an LLM preset"""
    try:
        # Get only the fields that were actually provided
        updates = request.dict(exclude_unset=True)

        success = await llm_service.update_preset(
            preset_id=preset_id, user_id=context.user_id, updates=updates
        )

        if not success:
            raise HTTPException(
                status_code=404, detail="Preset not found or access denied"
            )

        return BaseResponse(
            success=True,
            message="Preset updated successfully",
            result={"preset_id": preset_id},
            status_code=200,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/llm/presets/{preset_id}", response_model=BaseResponse[dict])
async def delete_llm_preset(
    preset_id: str,
    context: UserContext = Depends(get_current_user_context),
    llm_service: LLMService = Depends(get_llm_service),
):
    """Delete an LLM preset"""
    try:
        success = await llm_service.delete_preset(
            preset_id=preset_id, user_id=context.user_id
        )

        if not success:
            raise HTTPException(
                status_code=404, detail="Preset not found or access denied"
            )

        return BaseResponse(
            success=True,
            message="Preset deleted successfully",
            result={"preset_id": preset_id},
            status_code=200,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/llm/presets/{preset_id}/test", response_model=BaseResponse[TestLLMPresetResponse]
)
async def test_llm_preset(
    preset_id: str,
    request: TestLLMPresetRequest,
    context: UserContext = Depends(get_current_user_context),
    llm_service: LLMService = Depends(get_llm_service),
):
    """Test an LLM preset with a sample message"""
    try:
        start_time = time.time()

        # Build test messages
        messages = []

        # Get preset to include system prompt if requested
        if request.include_system_prompt:
            preset = await llm_service.get_preset(preset_id, context.user_id)
            if not preset:
                raise HTTPException(status_code=404, detail="Preset not found")

            messages.append({"role": "system", "content": preset.system_prompt})

        messages.append({"role": "user", "content": request.message})

        # Generate response
        response = await llm_service.generate_with_preset(
            preset_id=preset_id,
            messages=messages,
            user_id=context.user_id,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
        )

        response_time = int((time.time() - start_time) * 1000)

        # Calculate estimated cost
        preset = await llm_service.get_preset(preset_id, context.user_id)
        pricing = (
            LLMService.PROVIDER_INFO.get(preset.provider, {})
            .get("pricing", {})
            .get(preset.model, {})
        )

        input_cost = (response.usage.prompt_tokens / 1000) * pricing.get("input", 0)
        output_cost = (response.usage.completion_tokens / 1000) * pricing.get(
            "output", 0
        )
        estimated_cost = input_cost + output_cost

        test_response = TestLLMPresetResponse(
            test_message=request.message,
            response=response.content,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            estimated_cost=estimated_cost,
            response_time_ms=response_time,
            system_prompt_used=preset.system_prompt
            if request.include_system_prompt
            else None,
            provider=preset.provider,
            model=preset.model,
            temperature_used=request.temperature or preset.temperature,
            max_tokens_used=request.max_tokens or preset.max_tokens,
        )

        return BaseResponse(
            success=True,
            message="Test completed successfully",
            result=test_response,
            status_code=200,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/llm/presets/default", response_model=BaseResponse[dict])
async def set_default_preset(
    request: SetDefaultPresetRequest,
    context: UserContext = Depends(get_current_user_context),
    llm_service: LLMService = Depends(get_llm_service),
):
    """Set a preset as the user's default"""
    try:
        # Verify preset exists and user has access
        preset = await llm_service.get_preset(request.preset_id, context.user_id)
        if not preset:
            raise HTTPException(status_code=404, detail="Preset not found")

        # Remove default from all user's presets
        await llm_service.presets_collection.update_many(
            {"user_id": context.user_id}, {"$set": {"is_default": False}}
        )

        # Set new default
        await llm_service.presets_collection.update_one(
            {"id": request.preset_id}, {"$set": {"is_default": True}}
        )

        return BaseResponse(
            success=True,
            message="Default preset updated",
            result={"preset_id": request.preset_id},
            status_code=200,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/llm/usage/stats", response_model=BaseResponse[LLMUsageStatsResponse])
async def get_usage_statistics(
    start_date: Optional[datetime] = Query(None, description="Start date for stats"),
    end_date: Optional[datetime] = Query(None, description="End date for stats"),
    include_daily: bool = Query(False, description="Include daily breakdown"),
    context: UserContext = Depends(get_current_user_context),
    llm_service: LLMService = Depends(get_llm_service),
):
    """Get LLM usage statistics for the user"""
    try:
        # Default to last 30 days if not specified
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()

        stats = await llm_service.get_user_usage_stats(
            user_id=context.user_id, start_date=start_date, end_date=end_date
        )

        # Calculate projected monthly cost
        days_in_period = (end_date - start_date).days or 1
        daily_average = stats["total"]["cost"] / days_in_period
        projected_monthly = daily_average * 30

        response = LLMUsageStatsResponse(
            period=stats["period"],
            total=stats["total"],
            by_model=stats["by_model"],
            projected_monthly_cost=projected_monthly,
        )

        # TODO: Add daily breakdown if requested

        return BaseResponse(
            success=True,
            message="Usage statistics retrieved",
            result=response,
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
