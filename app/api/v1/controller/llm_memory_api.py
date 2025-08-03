"""LLM Memory API for managing memory/context in external LLM services like ChatGPT, Claude, and Gemini"""

import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from app.api.tags import Tags
from app.api.v1.request.llm_memory_request import (
    DeleteMemoryRequest,
    InjectMemoryRequest,
    GetFormattedMemoriesRequest,
    ClearContextRequest,
    ListConversationsRequest,
    SyncMemoriesRequest,
    MemorySyncSettings,
)
from app.api.v1.response.llm_memory_response import (
    LLMMemoryResponse,
    ConversationResponse,
    FormattedMemoriesResponse,
    ListConversationsResponse,
    DetailedConversationResponse,
    MemorySyncResponse,
    SyncStatusResponse,
)
from app.api.v1.response.base_response import (
    BaseResponse,
    error_response,
    success_response,
)
from app.common.auth.auth import UserContext, get_current_user_context
from app.common.enum.llm_vendor import LLMVendor
from app.service.llm_memory_service import LLMMemoryService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/llm-memory", tags=[Tags.LLMMemory])


# Dependency injection for LLM memory service
async def get_llm_memory_service(
    context: UserContext = Depends(get_current_user_context),
) -> LLMMemoryService:
    """Get LLM memory service instance with user context"""
    return LLMMemoryService(user_context=context)


@router.delete("/delete", response_model=BaseResponse[LLMMemoryResponse])
async def delete_llm_memory(
    request: DeleteMemoryRequest,
    context: UserContext = Depends(get_current_user_context),
    service: LLMMemoryService = Depends(get_llm_memory_service),
):
    """Delete memory/conversation from external LLM vendor (ChatGPT, Claude, Gemini)

    Note: Direct API deletion capabilities vary by vendor:
    - OpenAI (ChatGPT): Limited API support, mainly for Assistants API
    - Anthropic (Claude): No direct conversation deletion API
    - Google (Gemini): Context management through session control
    """
    try:
        result = await service.delete_memory(
            vendor=request.vendor,
            conversation_id=request.conversation_id,
            message_ids=request.message_ids,
            delete_all=request.delete_all,
            api_key=request.api_key,
        )

        if result["success"]:
            return success_response(
                result=LLMMemoryResponse(**result),
                message=f"Memory deletion completed for {request.vendor.value}",
            )
        else:
            return error_response(
                message=result.get("message", "Failed to delete memory"),
                status_code=400,
            )

    except NotImplementedError as e:
        return error_response(message=str(e), status_code=501)  # Not Implemented
    except Exception as e:
        logger.error(f"Error deleting LLM memory: {e}")
        return error_response(message=f"Internal error: {str(e)}", status_code=500)


@router.post("/sync", response_model=BaseResponse[MemorySyncResponse])
async def sync_llm_memories(
    request: SyncMemoriesRequest,
    settings: MemorySyncSettings = MemorySyncSettings(),
    context: UserContext = Depends(get_current_user_context),
    service: LLMMemoryService = Depends(get_llm_memory_service),
):
    """Sync memories from external LLMs (ChatGPT, Claude, Gemini) to Context Zero

    This endpoint:
    1. Connects to specified LLM vendors
    2. Retrieves conversation history
    3. Transforms messages into Context Zero memories
    4. Auto-categorizes and detects emotions
    5. Imports memories with deduplication
    """
    try:
        result = await service.sync_memories_from_llms(
            user_id=context.user_id,
            vendors=request.vendors,
            api_keys=request.api_keys,
            sync_mode=request.sync_mode,
            start_date=request.start_date,
            end_date=request.end_date,
            auto_categorize=request.auto_categorize,
            detect_emotions=request.detect_emotions,
            conversation_ids=request.conversation_ids,
            settings=settings,
        )

        if result["success"]:
            return success_response(
                result=MemorySyncResponse(**result),
                message=f"Successfully synced {result['total_synced']} memories from {len(request.vendors)} vendors",
            )
        else:
            return error_response(
                message=result.get("message", "Sync failed"), status_code=400
            )

    except Exception as e:
        logger.error(f"Error syncing memories: {e}")
        return error_response(message=f"Internal error: {str(e)}", status_code=500)


@router.get("/sync/{sync_id}/status", response_model=BaseResponse[SyncStatusResponse])
async def get_sync_status(
    sync_id: str,
    context: UserContext = Depends(get_current_user_context),
    service: LLMMemoryService = Depends(get_llm_memory_service),
):
    """Get status of an ongoing sync operation"""
    try:
        result = await service.get_sync_status(sync_id, context.user_id)

        if result:
            return success_response(
                result=SyncStatusResponse(**result),
                message=f"Sync {sync_id} is {result['status']}",
            )
        else:
            return error_response(
                message=f"Sync operation {sync_id} not found", status_code=404
            )

    except Exception as e:
        logger.error(f"Error getting sync status: {e}")
        return error_response(message=f"Internal error: {str(e)}", status_code=500)


@router.get("/list/{vendor}", response_model=BaseResponse[ListConversationsResponse])
async def list_llm_conversations(
    vendor: LLMVendor,
    api_key: Optional[str] = None,
    context: UserContext = Depends(get_current_user_context),
    service: LLMMemoryService = Depends(get_llm_memory_service),
):
    """List conversations/memory from external LLM vendor

    Note: Limited by vendor API capabilities
    """
    try:
        result = await service.list_conversations(vendor=vendor, api_key=api_key)

        return success_response(
            result=ListConversationsResponse(**result),
            message=f"Retrieved conversations from {vendor.value}",
        )

    except NotImplementedError as e:
        return error_response(message=str(e), status_code=501)
    except Exception as e:
        logger.error(f"Error listing LLM conversations: {e}")
        return error_response(message=f"Internal error: {str(e)}", status_code=500)


@router.post("/clear-context", response_model=BaseResponse[LLMMemoryResponse])
async def clear_llm_context(
    request: ClearContextRequest,
    context: UserContext = Depends(get_current_user_context),
    service: LLMMemoryService = Depends(get_llm_memory_service),
):
    """Clear context/memory for a specific session

    This is useful for:
    - ChatGPT: Reset conversation context in API usage
    - Claude: Clear conversation history
    - Gemini: Reset chat session
    """
    try:
        result = await service.clear_context(
            vendor=request.vendor,
            session_id=request.session_id,
            api_key=request.api_key,
        )

        if result["success"]:
            return success_response(
                result=LLMMemoryResponse(**result),
                message=f"Context cleared for {request.vendor.value}",
            )
        else:
            return error_response(
                message=result.get("message", "Failed to clear context"),
                status_code=400,
            )

    except Exception as e:
        logger.error(f"Error clearing LLM context: {e}")
        return error_response(message=f"Internal error: {str(e)}", status_code=500)


@router.post("/inject-and-chat", response_model=BaseResponse[ConversationResponse])
async def inject_memories_and_chat(
    request: InjectMemoryRequest,
    context: UserContext = Depends(get_current_user_context),
    service: LLMMemoryService = Depends(get_llm_memory_service),
):
    """Retrieve relevant memories from Context Zero and inject them into LLM conversation

    This endpoint:
    1. Searches Context Zero for relevant memories based on the user's message
    2. Formats memories appropriately for each LLM
    3. Sends the message with context to the LLM
    4. Returns the LLM's response
    """
    try:
        result = await service.inject_memories_and_chat(
            vendor=request.vendor,
            user_id=context.user_id,  # Use user_id from context
            message=request.message,
            memory_types=request.memory_types,
            limit=request.limit,
            include_summary=request.include_summary,
            include_timeline=request.include_timeline,
            api_key=request.api_key,
            session_id=request.session_id,
        )

        if result["success"]:
            return success_response(
                result=ConversationResponse(**result),
                message="Successfully processed message with injected memories",
            )
        else:
            return error_response(
                message=result.get("message", "Failed to process message"),
                status_code=400,
            )

    except Exception as e:
        logger.error(f"Error in inject and chat: {e}")
        return error_response(message=f"Internal error: {str(e)}", status_code=500)


@router.post("/inject-memories", response_model=BaseResponse[FormattedMemoriesResponse])
async def inject_memories_only(
    request: GetFormattedMemoriesRequest,
    context: UserContext = Depends(get_current_user_context),
    service: LLMMemoryService = Depends(get_llm_memory_service),
):
    """Retrieve and format memories from Context Zero for manual injection

    This endpoint returns formatted memories that you can manually include
    in your LLM API calls.
    """
    try:
        result = await service.get_formatted_memories(
            vendor=request.vendor,
            user_id=context.user_id,  # Use user_id from context
            query=request.query,
            memory_types=request.memory_types,
            categories=request.categories,
            emotions=request.emotions,
            tags=request.tags,
            limit=request.limit,
        )

        return success_response(
            result=FormattedMemoriesResponse(**result),
            message=f"Retrieved {result['memory_count']} memories for {request.vendor.value}",
        )

    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        return error_response(message=f"Internal error: {str(e)}", status_code=500)
