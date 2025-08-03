"""Chat API endpoints"""
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import asyncio
from enum import Enum

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.api.tags import Tags
from app.api.v1.request.chat_request import (
    CreateChatSessionRequest,
    SendMessageRequest,
    UpdateChatSessionRequest,
    ExtractMemoriesRequest,
)
from app.api.v1.response.base_response import BaseResponse
from app.api.v1.response.chat_response import (
    ChatSessionResponse,
    ChatSessionDetailResponse,
    CreateChatSessionResponse,
    SendMessageResponse,
    ExtractMemoriesResponse,
    MemoryContextItem,
    ExtractedMemoryItem,
    ChatMessageResponse,
)
from app.common.auth.auth import UserContext, get_current_user_context
from app.common.enum.memory import MemoryType
from app.db.mongodb import get_database
from app.service.chat_service import ChatService
from app.service.memory_service import MemoryService, get_memory_service
from app.service.llm_service import LLMService
from app.service.memory_extraction_service import MemoryExtractionService

logger = logging.getLogger(__name__)

router = APIRouter(tags=[Tags.Playground])


class EnumEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles Enum values"""

    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super().default(obj)


def _transform_memory_context(
    memories: List[Dict[str, Any]]
) -> List[MemoryContextItem]:
    """Transform memory dictionaries to MemoryContextItem objects"""
    result = []
    for memory in memories:
        if isinstance(memory, dict):
            result.append(
                MemoryContextItem(
                    id=memory.get("id", ""),
                    content=memory.get("content", ""),
                    summary=memory.get("summary"),
                    memory_type=MemoryType(
                        memory.get("memory_type", "semantic_memory")
                    ),
                    score=float(memory.get("score", 0.0)),
                    tags=memory.get("tags", []),
                    created_at=memory.get("created_at", datetime.utcnow()),
                )
            )
    return result


def _transform_extracted_memories(
    memories: List[Dict[str, Any]]
) -> List[ExtractedMemoryItem]:
    """Transform extracted memory dictionaries to ExtractedMemoryItem objects"""
    result = []
    for memory in memories:
        if isinstance(memory, dict):
            result.append(
                ExtractedMemoryItem(
                    id=memory.get("id", ""),
                    content=memory.get("content", ""),
                    memory_type=MemoryType(
                        memory.get("memory_type", "semantic_memory")
                    ),
                    tags=memory.get("tags", []),
                    confidence=float(memory.get("confidence", 1.0)),
                )
            )
    return result


def get_llm_service(db: AsyncIOMotorDatabase = Depends(get_database)) -> LLMService:
    """Get LLM service instance"""
    return LLMService(db)


def get_memory_extraction_service(
    llm_service: LLMService = Depends(get_llm_service),
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> MemoryExtractionService:
    """Get memory extraction service instance"""
    return MemoryExtractionService(llm_service, db)


def get_chat_service(
    db: AsyncIOMotorDatabase = Depends(get_database),
    memory_service: MemoryService = Depends(get_memory_service),
    llm_service: LLMService = Depends(get_llm_service),
    memory_extraction_service: MemoryExtractionService = Depends(
        get_memory_extraction_service
    ),
) -> ChatService:
    """Get chat service instance"""
    return ChatService(db, memory_service, llm_service, memory_extraction_service)


@router.post("/chat/sessions", response_model=BaseResponse[CreateChatSessionResponse])
async def create_chat_session(
    request: CreateChatSessionRequest,
    context: UserContext = Depends(get_current_user_context),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Create a new chat session"""
    try:
        session = await chat_service.create_session(
            user_id=context.user_id, title=request.title, metadata=request.metadata
        )

        response = CreateChatSessionResponse(
            session_id=session.id, title=session.title, created_at=session.created_at
        )

        return BaseResponse(
            success=True,
            message="Chat session created successfully",
            result=response,
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/chat/sessions", response_model=BaseResponse[List[ChatSessionResponse]])
async def get_chat_sessions(
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
    context: UserContext = Depends(get_current_user_context),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Get user's chat sessions"""
    try:
        sessions = await chat_service.get_sessions(
            user_id=context.user_id, limit=limit, skip=skip
        )

        session_responses = [
            ChatSessionResponse(
                id=session.id,
                user_id=session.user_id,
                title=session.title,
                created_at=session.created_at,
                updated_at=session.updated_at,
                last_message=session.last_message,
                message_count=session.message_count,
                total_memories_extracted=session.total_memories_extracted,
            )
            for session in sessions
        ]

        return BaseResponse(
            success=True,
            message=f"Retrieved {len(sessions)} chat sessions",
            result=session_responses,
            status_code=200,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/chat/sessions/{session_id}",
    response_model=BaseResponse[ChatSessionDetailResponse],
)
async def get_chat_session(
    session_id: str,
    message_limit: int = Query(50, ge=1, le=200),
    message_skip: int = Query(0, ge=0),
    context: UserContext = Depends(get_current_user_context),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Get a specific chat session with messages"""
    try:
        session = await chat_service.get_session(session_id, context.user_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        messages = await chat_service.get_messages(
            session_id=session_id, limit=message_limit, skip=message_skip
        )

        session_response = ChatSessionResponse(
            id=session.id,
            user_id=session.user_id,
            title=session.title,
            created_at=session.created_at,
            updated_at=session.updated_at,
            last_message=session.last_message,
            message_count=session.message_count,
            total_memories_extracted=session.total_memories_extracted,
        )

        message_responses = []
        for msg in messages:
            # Debug logging
            if "soy" in msg.content.lower():
                logger.info(f"\n=== DEBUG: Processing soy message ===")
                logger.info(f"Message ID: {msg.id}")
                logger.info(f"msg.memories_extracted: {msg.memories_extracted}")
                logger.info(f"Type: {type(msg.memories_extracted)}")
                logger.info("=====================================")

            # Transform extracted memories if they are stored as IDs
            extracted_memories = []
            if msg.memories_extracted:
                for item in msg.memories_extracted:
                    if isinstance(item, str):
                        # It's just an ID, skip for now (could fetch from memory service)
                        continue
                    elif isinstance(item, dict):
                        # It's already a full memory object
                        try:
                            # Convert memory_type string to MemoryType enum if needed
                            memory_type = item.get("memory_type", "semantic_memory")
                            if isinstance(memory_type, str):
                                memory_type = MemoryType(memory_type)

                            extracted_memory = ExtractedMemoryItem(
                                id=item.get("id", ""),
                                content=item.get("content", ""),
                                memory_type=memory_type,
                                tags=item.get("tags", []),
                                confidence=item.get("confidence", 1.0),
                            )
                            extracted_memories.append(extracted_memory)
                        except Exception as e:
                            print(f"Error converting extracted memory: {e}")
                            continue

            # Transform context memories if they are stored as IDs
            context_memories = []
            if msg.context_used:
                for item in msg.context_used:
                    if isinstance(item, str):
                        # It's just an ID, skip for now
                        continue
                    elif isinstance(item, dict):
                        # It's already a full memory object
                        try:
                            # Convert memory_type string to MemoryType enum if needed
                            memory_type = item.get("memory_type", "semantic_memory")
                            if isinstance(memory_type, str):
                                memory_type = MemoryType(memory_type)

                            # Handle datetime conversion
                            created_at = item.get("created_at", datetime.utcnow())
                            if isinstance(created_at, str):
                                created_at = datetime.fromisoformat(
                                    created_at.replace("Z", "+00:00")
                                )

                            context_memory = MemoryContextItem(
                                id=item.get("id", ""),
                                content=item.get("content", ""),
                                summary=item.get("summary"),
                                memory_type=memory_type,
                                score=item.get("score", 0.0),
                                tags=item.get("tags", []),
                                created_at=created_at,
                            )
                            context_memories.append(context_memory)
                        except Exception as e:
                            print(f"Error converting context memory: {e}")
                            continue

            message_response = ChatMessageResponse(
                id=msg.id,
                role=msg.role,
                content=msg.content,
                timestamp=msg.timestamp,
                metadata=msg.metadata,
                memories_extracted=extracted_memories if extracted_memories else None,
                context_used=context_memories if context_memories else None,
            )
            message_responses.append(message_response)

        response = ChatSessionDetailResponse(
            session=session_response, messages=message_responses
        )

        return BaseResponse(
            success=True,
            message="Session retrieved successfully",
            result=response,
            status_code=200,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat/sessions/{session_id}/messages",
    response_model=BaseResponse[SendMessageResponse],
)
async def send_message(
    session_id: str,
    request: SendMessageRequest,
    context: UserContext = Depends(get_current_user_context),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Send a message in a chat session"""
    try:
        result = await chat_service.send_message(
            session_id=session_id, user_id=context.user_id, request=request
        )

        # Transform memories to proper response format
        user_extracted = _transform_extracted_memories(
            result["user_message"].memories_extracted or []
        )
        assistant_extracted = _transform_extracted_memories(
            result["assistant_message"].memories_extracted or []
        )
        context_used = _transform_memory_context(result["context_used"] or [])

        # Log the transformed data
        print(f"\n=== NON-STREAMING RESPONSE ===")
        print(f"Context Used (raw): {result.get('context_used')}")
        print(f"Context Used (transformed): {context_used}")
        print(
            f"User Memories Extracted (raw): {result['user_message'].memories_extracted}"
        )
        print(f"User Memories Extracted (transformed): {user_extracted}")
        print(
            f"Assistant Memories Extracted (raw): {result['assistant_message'].memories_extracted}"
        )
        print(f"Assistant Memories Extracted (transformed): {assistant_extracted}")
        print("=============================\n")

        response = SendMessageResponse(
            user_message={
                "id": result["user_message"].id,
                "role": result["user_message"].role,
                "content": result["user_message"].content,
                "timestamp": result["user_message"].timestamp,
                "metadata": result["user_message"].metadata,
                "memories_extracted": user_extracted,
                "context_used": context_used,
            },
            assistant_message={
                "id": result["assistant_message"].id,
                "role": result["assistant_message"].role,
                "content": result["assistant_message"].content,
                "timestamp": result["assistant_message"].timestamp,
                "metadata": result["assistant_message"].metadata,
                "memories_extracted": assistant_extracted,
                "context_used": context_used,
            },
            memories_extracted=user_extracted + assistant_extracted,
            context_used=context_used,
        )

        return BaseResponse(
            success=True,
            message="Message sent successfully",
            result=response,
            status_code=200,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat/sessions/{session_id}/messages/stream")
async def send_message_stream(
    session_id: str,
    request: SendMessageRequest,
    context: UserContext = Depends(get_current_user_context),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Send a message in a chat session with streaming response"""

    async def generate():
        try:
            # Send initial message indicating start of stream
            yield f"data: {json.dumps({'type': 'start', 'session_id': session_id}, cls=EnumEncoder)}\n\n"

            # Call the streaming version of send_message
            async for chunk in chat_service.send_message_stream(
                session_id=session_id, user_id=context.user_id, request=request
            ):
                # Log the chunk to see what's being sent
                print(f"\n=== STREAMING CHUNK ===")
                print(f"Type: {chunk.get('type')}")
                if chunk.get("type") == "assistant_message":
                    print(f"Message ID: {chunk.get('message', {}).get('id')}")
                    print(
                        f"Context Used: {chunk.get('message', {}).get('context_used')}"
                    )
                    print(
                        f"Memories Extracted: {chunk.get('message', {}).get('memories_extracted')}"
                    )
                    print(f"Summary: {chunk.get('summary')}")
                print(f"Full chunk: {json.dumps(chunk, cls=EnumEncoder, indent=2)}")
                print("===================\n")

                # Send each chunk as SSE with custom encoder
                yield f"data: {json.dumps(chunk, cls=EnumEncoder)}\n\n"

                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)

            # Send completion message
            yield f"data: {json.dumps({'type': 'done'}, cls=EnumEncoder)}\n\n"

        except ValueError as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'code': 404}, cls=EnumEncoder)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'code': 500}, cls=EnumEncoder)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


@router.put("/chat/sessions/{session_id}", response_model=BaseResponse[dict])
async def update_chat_session(
    session_id: str,
    request: UpdateChatSessionRequest,
    context: UserContext = Depends(get_current_user_context),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Update a chat session"""
    try:
        success = await chat_service.update_session(
            session_id=session_id,
            user_id=context.user_id,
            title=request.title,
            metadata=request.metadata,
        )

        if not success:
            raise HTTPException(status_code=404, detail="Session not found")

        return BaseResponse(
            success=True,
            message="Session updated successfully",
            result={"session_id": session_id},
            status_code=200,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/sessions/{session_id}", response_model=BaseResponse[dict])
async def delete_chat_session(
    session_id: str,
    hard_delete: bool = Query(
        False,
        description="Permanently delete session and messages (requires admin role)",
    ),
    context: UserContext = Depends(get_current_user_context),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Delete a chat session"""
    try:
        if hard_delete:
            # Check if user has admin role
            if context.role != "admin":
                raise HTTPException(
                    status_code=403, detail="Hard delete requires admin role"
                )

            deleted_count = await chat_service.hard_delete_session(
                session_id=session_id, user_id=context.user_id
            )

            if deleted_count["sessions"] == 0:
                raise HTTPException(status_code=404, detail="Session not found")

            return BaseResponse(
                success=True,
                message=f"Permanently deleted session and {deleted_count['messages']} messages",
                result={"session_id": session_id, "deleted_count": deleted_count},
                status_code=200,
            )
        else:
            success = await chat_service.delete_session(
                session_id=session_id, user_id=context.user_id
            )

            if not success:
                raise HTTPException(status_code=404, detail="Session not found")

            return BaseResponse(
                success=True,
                message="Session deleted successfully",
                result={"session_id": session_id},
                status_code=200,
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/chat/sessions", response_model=BaseResponse[dict])
async def delete_all_chat_sessions(
    hard_delete: bool = Query(
        False,
        description="Permanently delete sessions and messages (requires admin role)",
    ),
    context: UserContext = Depends(get_current_user_context),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Delete all chat sessions for the current user"""
    try:
        if hard_delete:
            # Check if user has admin role
            if context.role != "admin":
                raise HTTPException(
                    status_code=403, detail="Hard delete requires admin role"
                )

            deleted_count = await chat_service.hard_delete_all_sessions(
                user_id=context.user_id
            )
            message = f"Permanently deleted {deleted_count['sessions']} sessions and {deleted_count['messages']} messages"
        else:
            deleted_count = await chat_service.delete_all_sessions(
                user_id=context.user_id
            )
            message = f"Deleted {deleted_count} sessions successfully"

        return BaseResponse(
            success=True,
            message=message,
            result={"deleted_count": deleted_count},
            status_code=200,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/chat/sessions/{session_id}/extract-memories",
    response_model=BaseResponse[ExtractMemoriesResponse],
)
async def extract_memories_from_chat(
    session_id: str,
    request: ExtractMemoriesRequest,
    context: UserContext = Depends(get_current_user_context),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Extract memories from a chat session"""
    try:
        result = await chat_service.extract_memories_from_session(
            session_id=session_id,
            user_id=context.user_id,
            message_ids=request.message_ids,
            force=request.force,
        )

        response = ExtractMemoriesResponse(
            session_id=result["session_id"],
            messages_processed=result["messages_processed"],
            memories_extracted=result["memories_extracted"],
            extraction_summary=result["extraction_summary"],
        )

        return BaseResponse(
            success=True,
            message=f"Extracted {len(result['memories_extracted'])} memories",
            result=response,
            status_code=200,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
