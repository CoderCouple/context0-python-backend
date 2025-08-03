"""Chat service for managing chat sessions and messages"""
import asyncio
import json
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any

from motor.motor_asyncio import AsyncIOMotorDatabase
from pymongo import DESCENDING

from app.api.v1.request.chat_request import SendMessageRequest
from app.api.v1.request.memory_request import MemoryRecordInput
from app.common.enum.chat import ChatRole, ChatSessionStatus
from app.common.enum.memory_category import MemoryCategory
from app.model.chat import ChatSession, ChatMessage
from app.model.chat_extracted_memory import ChatExtractedMemory
from app.service.memory_service import MemoryService
from app.service.llm_service import LLMService
from app.service.memory_extraction_service import MemoryExtractionService
from app.service.tag_extraction_service import TagExtractionService
from app.memory.engine.memory_engine import MemoryEngine


class ChatService:
    """Service for managing chat sessions and messages"""

    def __init__(
        self,
        db: AsyncIOMotorDatabase,
        memory_service: MemoryService,
        llm_service: LLMService,
        memory_extraction_service: Optional[MemoryExtractionService] = None,
        tag_extraction_service: Optional[TagExtractionService] = None,
    ):
        self.db = db
        self.memory_service = memory_service
        self.llm_service = llm_service
        self.memory_extraction_service = (
            memory_extraction_service or MemoryExtractionService(llm_service, db)
        )
        self.tag_extraction_service = tag_extraction_service or TagExtractionService(
            llm_service, db
        )
        self.sessions_collection = db.chat_sessions
        self.messages_collection = db.chat_messages
        self.extracted_memories_collection = db.chat_extracted_memories

    async def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ChatSession:
        """Create a new chat session"""
        session_id = str(uuid.uuid4())

        # Auto-generate title if not provided
        if not title:
            title = f"Chat {datetime.utcnow().strftime('%Y-%m-%d %H:%M')}"

        session = ChatSession(
            id=session_id, user_id=user_id, title=title, metadata=metadata or {}
        )

        await self.sessions_collection.insert_one(session.dict())
        return session

    async def get_sessions(
        self, user_id: str, limit: int = 20, skip: int = 0
    ) -> List[ChatSession]:
        """Get user's chat sessions"""
        cursor = (
            self.sessions_collection.find(
                {"user_id": user_id, "status": {"$ne": ChatSessionStatus.DELETED.value}}
            )
            .sort("updated_at", DESCENDING)
            .skip(skip)
            .limit(limit)
        )

        sessions = []
        async for doc in cursor:
            sessions.append(ChatSession(**doc))

        return sessions

    async def get_session(self, session_id: str, user_id: str) -> Optional[ChatSession]:
        """Get a specific chat session"""
        doc = await self.sessions_collection.find_one(
            {
                "id": session_id,
                "user_id": user_id,
                "status": {"$ne": ChatSessionStatus.DELETED.value},
            }
        )

        return ChatSession(**doc) if doc else None

    async def get_messages(
        self, session_id: str, limit: int = 50, skip: int = 0
    ) -> List[ChatMessage]:
        """Get messages for a chat session"""
        cursor = (
            self.messages_collection.find({"session_id": session_id})
            .sort("timestamp", 1)
            .skip(skip)
            .limit(limit)
        )

        messages = []
        async for doc in cursor:
            message = ChatMessage(**doc)

            # Fetch extracted memories from the separate collection
            extracted_memories_cursor = self.extracted_memories_collection.find(
                {"chat_message_id": message.id}
            )

            extracted_memories = []
            async for mem_doc in extracted_memories_cursor:
                extracted_memories.append(
                    {
                        "id": mem_doc["original_memory_id"],
                        "content": mem_doc["content"],
                        "memory_type": mem_doc["memory_type"],
                        "tags": mem_doc.get("tags", []),
                        "confidence": mem_doc.get("confidence", 1.0),
                    }
                )

            if extracted_memories:
                message.memories_extracted = extracted_memories

            messages.append(message)

        return messages

    async def send_message(
        self, session_id: str, user_id: str, request: SendMessageRequest
    ) -> Dict[str, Any]:
        """Send a message and get AI response"""
        # Verify session belongs to user
        session = await self.get_session(session_id, user_id)
        if not session:
            raise ValueError("Session not found")

        # Get LLM preset for session or user default
        preset_id = session.metadata.get("llm_preset_id")
        if (
            not preset_id
            and hasattr(request, "llm_preset_id")
            and request.llm_preset_id
        ):
            preset_id = request.llm_preset_id
        if not preset_id:
            preset_id = await self.llm_service.get_user_default_preset(user_id)

        # Get preset configuration
        preset = None
        if preset_id:
            preset = await self.llm_service.get_preset(preset_id, user_id)
        else:
            # No preset found, get or create user's default preset
            default_preset_id = (
                await self.llm_service.get_or_create_user_default_preset(user_id)
            )
            if default_preset_id:
                preset_id = default_preset_id
                preset = await self.llm_service.get_preset(preset_id, user_id)
                # Update session to use this preset
                await self.sessions_collection.update_one(
                    {"id": session_id}, {"$set": {"metadata.llm_preset_id": preset_id}}
                )

        # Create user message
        user_message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=ChatRole.USER,
            content=request.content,
        )

        # Get memory context if requested
        context_memories = []
        search_results = []
        memory_context_text = ""

        # Use preset settings or request settings
        use_memory_context = preset.use_memory_context if preset else True
        if use_memory_context and request.use_memory_context:
            # Search for relevant memories
            from app.api.v1.request.memory_request import SearchQuery

            search_query = SearchQuery(
                user_id=user_id,
                query=request.content,
                include_content=True,
                limit=10,  # Increase limit to get more relevant memories
            )

            # Apply threshold from preset if available
            if preset and hasattr(preset, "memory_threshold"):
                search_query.threshold = preset.memory_threshold

            search_response = await self.memory_service.search_memories(search_query)
            if search_response.success:
                search_results = search_response.results[:5]  # Get top 5 results
                # Store full memory details for response
                context_memories = []
                for memory in search_results:
                    context_memories.append(
                        {
                            "id": memory.id,
                            "content": memory.content or memory.summary or "",
                            "summary": memory.summary,
                            "memory_type": memory.memory_type.value,
                            "score": memory.score,
                            "tags": memory.tags,
                            "created_at": memory.created_at.isoformat(),
                        }
                    )
                user_message.context_used = context_memories

                # Format memory context
                memory_context_text = self._format_memory_context(search_results)

        # Get conversation history
        history_limit = preset.conversation_history_limit if preset else 10
        recent_messages = await self.get_messages(session_id, limit=history_limit)

        # Build messages
        messages = []

        # Build enhanced system prompt with memory awareness
        base_system_prompt = (
            preset.system_prompt if preset else "You are a helpful assistant."
        )

        # Add memory-aware instructions
        enhanced_prompt = (
            """You are a helpful AI assistant with access to the user's memory system. 
You MUST use the provided memories to personalize your responses. When the user asks for suggestions or recommendations, 
prioritize information from their memories over generic responses.

"""
            + base_system_prompt
        )

        if preset and preset.custom_instructions:
            enhanced_prompt += f"\n\n{preset.custom_instructions}"

        messages.append({"role": "system", "content": enhanced_prompt})

        # Add memory context with clear instructions
        if memory_context_text:
            memory_message = f"""=== USER'S PERSONAL MEMORIES (YOU MUST USE THESE) ===
{memory_context_text}

IMPORTANT: The above are the user's personal memories. You MUST reference and use this information in your response. 
For example, if the user asks for food suggestions and their memories mention they love pizza, you should suggest pizza."""

            messages.append({"role": "system", "content": memory_message})

        # Add conversation history
        for msg in recent_messages[-history_limit:]:
            message_dict = {"role": msg.role.value, "content": msg.content}

            # Add timestamps if requested
            if preset and preset.include_timestamps:
                message_dict["content"] = f"[{msg.timestamp.isoformat()}] {msg.content}"

            messages.append(message_dict)

        # Add current user message
        messages.append({"role": "user", "content": request.content})

        # Get AI response using LLM service
        llm_params = {
            "messages": messages,
            "user_id": user_id,
            "session_id": session_id,
            "stream": getattr(request, "stream", False),
        }

        # Add overrides from request if provided
        if hasattr(request, "temperature"):
            llm_params["temperature"] = request.temperature
        if hasattr(request, "max_tokens"):
            llm_params["max_tokens"] = request.max_tokens

        # Generate response
        if not preset_id:
            raise ValueError("No LLM preset available")

        response = await self.llm_service.generate_with_preset(
            preset_id=preset_id, **llm_params
        )

        assistant_content = response.content

        # Create assistant message
        assistant_message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=ChatRole.ASSISTANT,
            content=assistant_content,
            context_used=context_memories,
        )

        # Extract memories if requested
        extracted_memories = []
        extract_memories = preset.extract_memories if preset else True
        if extract_memories and request.extract_memories:
            # Determine memory types to extract
            memory_types = preset.memory_extraction_types if preset else ["chat_memory"]
            print(f"\n=== PRESET DEBUG ===")
            print(f"Preset: {preset.name if preset else 'None'}")
            print(f"Preset ID: {preset.id if preset else 'None'}")
            print(
                f"Memory types from preset: {preset.memory_extraction_types if preset else 'N/A'}"
            )
            print(f"Memory types to extract: {memory_types}")
            print("===================\n")

            # Build context for extraction
            extraction_context = {
                "user_name": await self._get_user_name(user_id),
                "session_id": session_id,
                "timestamp": datetime.utcnow(),
                "session_context": session.title,
                "domain": "chat_conversation",
                "participants": f"{await self._get_user_name(user_id)}, AI Assistant",
                # Optional fields - provide empty strings to avoid template errors
                "location": "",
                "previous_context": "",
            }

            # Only extract from user messages to avoid duplicates
            # Assistant messages typically just acknowledge or repeat user information
            messages_to_extract = [{"role": "user", "content": user_message.content}]

            for msg in messages_to_extract:
                extraction_result = (
                    await self.memory_extraction_service.extract_and_store_memories(
                        content=msg["content"],
                        memory_types=memory_types,
                        user_id=user_id,
                        session_id=session_id,
                        context={**extraction_context, "role": msg["role"]},
                        preset_id=preset_id,
                    )
                )

                print(f"\n=== EXTRACTION RESULT ===")
                print(f"Role: {msg['role']}")
                print(f"Extracted: {extraction_result.get('extracted', {})}")
                print(f"Stored: {extraction_result.get('stored', {})}")
                print("========================\n")

                # Get stored memory details
                stored_memories = []
                for memory_type, memory_ids in extraction_result.get(
                    "stored", {}
                ).items():
                    # Fetch the actual memory details for each ID
                    for memory_id in memory_ids:
                        try:
                            memory = await self.memory_service.get_memory(
                                memory_id, user_id
                            )
                            if memory:
                                print(f"\n=== FETCHED MEMORY (NON-STREAM) ===")
                                print(f"ID: {memory.id}")
                                print(f"Input: {memory.input}")
                                print(f"Type: {memory.memory_type}")
                                print("==================================\n")
                                memory_snapshot = {
                                    "id": memory.id,
                                    "content": memory.input,
                                    "memory_type": memory.memory_type.value,
                                    "tags": memory.tags,
                                    "confidence": memory.confidence,
                                }
                                stored_memories.append(memory_snapshot)

                                # Store in extracted memories collection
                                message_id = (
                                    user_message.id
                                    if msg["role"] == "user"
                                    else assistant_message.id
                                )
                                extracted_memory_doc = ChatExtractedMemory(
                                    id=str(uuid.uuid4()),  # New ID for the snapshot
                                    chat_message_id=message_id,
                                    session_id=session_id,
                                    user_id=user_id,
                                    content=memory.input,
                                    memory_type=memory.memory_type,
                                    tags=memory.tags,
                                    confidence=memory.confidence,
                                    original_memory_id=memory.id,
                                    summary=getattr(memory, "summary", None),
                                    metadata=getattr(memory, "metadata", {}),
                                )
                                await self.extracted_memories_collection.insert_one(
                                    extracted_memory_doc.dict()
                                )

                        except Exception as e:
                            # If we can't fetch details, just store the ID
                            stored_memories.append(
                                {
                                    "id": memory_id,
                                    "content": extraction_result.get("extracted", {})
                                    .get(memory_type, {})
                                    .get("text", ""),
                                    "memory_type": memory_type,
                                    "tags": [],
                                    "confidence": 1.0,
                                }
                            )

                # Always user message since we only extract from user
                # Don't store full memories in message, they're in extracted_memories collection
                # Just store a marker that memories were extracted
                if stored_memories:
                    user_message.memories_extracted = [
                        "extracted"
                    ]  # Marker for frontend
                extracted_memories.extend(stored_memories)

        # Save messages
        await self.messages_collection.insert_many(
            [user_message.dict(), assistant_message.dict()]
        )

        # Update session
        await self._update_session_summary(
            session_id, assistant_content, len(extracted_memories)
        )

        return {
            "user_message": user_message,
            "assistant_message": assistant_message,
            "memories_extracted": extracted_memories,
            "context_used": context_memories,
        }

    async def _update_session_summary(
        self, session_id: str, last_message: str, new_memories: int
    ):
        """Update session summary after new messages"""
        # Truncate last message for summary
        truncated_message = (
            last_message[:100] + "..." if len(last_message) > 100 else last_message
        )

        await self.sessions_collection.update_one(
            {"id": session_id},
            {
                "$set": {
                    "last_message": truncated_message,
                    "updated_at": datetime.utcnow(),
                },
                "$inc": {
                    "message_count": 2,  # User + assistant
                    "total_memories_extracted": new_memories,
                },
            },
        )

    async def send_message_stream(
        self, session_id: str, user_id: str, request: SendMessageRequest
    ):
        """Send a message and stream AI response"""
        # Similar setup as send_message
        session = await self.get_session(session_id, user_id)
        if not session:
            raise ValueError("Session not found")

        # Get LLM preset
        preset_id = session.metadata.get("llm_preset_id")
        if (
            not preset_id
            and hasattr(request, "llm_preset_id")
            and request.llm_preset_id
        ):
            preset_id = request.llm_preset_id
        if not preset_id:
            preset_id = await self.llm_service.get_user_default_preset(user_id)

        preset = None
        if preset_id:
            preset = await self.llm_service.get_preset(preset_id, user_id)
        else:
            # No preset found, get or create user's default preset
            default_preset_id = (
                await self.llm_service.get_or_create_user_default_preset(user_id)
            )
            if default_preset_id:
                preset_id = default_preset_id
                preset = await self.llm_service.get_preset(preset_id, user_id)
                # Update session to use this preset
                await self.sessions_collection.update_one(
                    {"id": session_id}, {"$set": {"metadata.llm_preset_id": preset_id}}
                )

        # Create and save user message
        user_message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=ChatRole.USER,
            content=request.content,
        )

        # Get memory context (same as send_message)
        context_memories = []
        memory_context_text = ""
        use_memory_context = preset.use_memory_context if preset else True

        if use_memory_context and request.use_memory_context:
            from app.api.v1.request.memory_request import SearchQuery

            search_query = SearchQuery(
                user_id=user_id,
                query=request.content,
                include_content=True,
                limit=10,  # Increase limit to get more relevant memories
            )
            if preset and hasattr(preset, "memory_threshold"):
                search_query.threshold = preset.memory_threshold

            search_response = await self.memory_service.search_memories(search_query)
            if search_response.success:
                search_results = search_response.results[:5]
                # Store full memory details for response
                context_memories = []
                for memory in search_results:
                    context_memories.append(
                        {
                            "id": memory.id,
                            "content": memory.content or memory.summary or "",
                            "summary": memory.summary,
                            "memory_type": memory.memory_type.value,
                            "score": memory.score,
                            "tags": memory.tags,
                            "created_at": memory.created_at.isoformat(),
                        }
                    )
                user_message.context_used = context_memories
                memory_context_text = self._format_memory_context(search_results)
                print(f"\n=== CONTEXT MEMORIES FOUND ===")
                print(f"Found {len(context_memories)} context memories")
                for mem in context_memories[:2]:  # Show first 2
                    print(
                        f"- ID: {mem.get('id')}, Score: {mem.get('score')}, Type: {mem.get('memory_type')}"
                    )
                print("=============================\n")

        # Build messages with enhanced memory-aware prompts
        messages = []

        # Build enhanced system prompt with memory awareness
        base_system_prompt = (
            preset.system_prompt if preset else "You are a helpful assistant."
        )

        # Add memory-aware instructions
        enhanced_prompt = (
            """You are a helpful AI assistant with access to the user's memory system. 
You MUST use the provided memories to personalize your responses. When the user asks for suggestions or recommendations, 
prioritize information from their memories over generic responses.

"""
            + base_system_prompt
        )

        if preset and preset.custom_instructions:
            enhanced_prompt += f"\n\n{preset.custom_instructions}"

        messages.append({"role": "system", "content": enhanced_prompt})

        # Add memory context with clear instructions
        if memory_context_text:
            memory_message = f"""=== USER'S PERSONAL MEMORIES (YOU MUST USE THESE) ===
{memory_context_text}

IMPORTANT: The above are the user's personal memories. You MUST reference and use this information in your response. 
For example, if the user asks for food suggestions and their memories mention they love pizza, you should suggest pizza."""

            messages.append({"role": "system", "content": memory_message})

        # Add conversation history
        history_limit = preset.conversation_history_limit if preset else 10
        recent_messages = await self.get_messages(session_id, limit=history_limit)

        for msg in recent_messages[-history_limit:]:
            message_dict = {
                "role": msg.role.value if hasattr(msg.role, "value") else msg.role,
                "content": msg.content,
            }
            if preset and preset.include_timestamps:
                message_dict["content"] = f"[{msg.timestamp.isoformat()}] {msg.content}"
            messages.append(message_dict)

        messages.append({"role": "user", "content": request.content})

        # Save user message
        await self.messages_collection.insert_one(user_message.dict())

        # Yield user message info
        yield {
            "type": "user_message",
            "message": {
                "id": user_message.id,
                "role": "user",
                "content": user_message.content,
                "timestamp": user_message.timestamp.isoformat(),
                "context_used": context_memories,
            },
        }

        # Create assistant message with placeholder
        assistant_message_id = str(uuid.uuid4())
        assistant_content = ""

        # Stream the response
        llm_params = {
            "messages": messages,
            "user_id": user_id,
            "session_id": session_id,
        }

        if hasattr(request, "temperature"):
            llm_params["temperature"] = request.temperature
        if hasattr(request, "max_tokens"):
            llm_params["max_tokens"] = request.max_tokens

        # Use streaming version of LLM service
        if not preset_id:
            raise ValueError("No LLM preset available")

        stream = self.llm_service.generate_stream_with_preset(
            preset_id=preset_id, **llm_params
        )

        # Yield content chunks as they arrive
        async for chunk in stream:
            assistant_content += chunk
            yield {
                "type": "content",
                "content": chunk,
                "message_id": assistant_message_id,
            }

        # Create and save complete assistant message
        assistant_message = ChatMessage(
            id=assistant_message_id,
            session_id=session_id,
            role=ChatRole.ASSISTANT,
            content=assistant_content,
            context_used=context_memories,
        )

        # Extract memories if requested (same as send_message)
        extracted_memories = []
        extract_memories = preset.extract_memories if preset else True
        print(f"\n=== MEMORY EXTRACTION CHECK ===")
        print(
            f"Preset extract_memories: {preset.extract_memories if preset else 'No preset'}"
        )
        print(f"Request extract_memories: {request.extract_memories}")
        print(f"Will extract: {extract_memories and request.extract_memories}")
        print("==============================\n")

        try:
            if extract_memories and request.extract_memories:
                memory_types = (
                    preset.memory_extraction_types if preset else ["chat_memory"]
                )

                extraction_context = {
                    "user_name": await self._get_user_name(user_id),
                    "session_id": session_id,
                    "timestamp": datetime.utcnow(),
                    "session_context": session.title,
                    "domain": "chat_conversation",
                    "participants": f"{await self._get_user_name(user_id)}, AI Assistant",
                    # Optional fields - provide empty strings to avoid template errors
                    "location": "",
                    "previous_context": "",
                }

                # Only extract from user messages to avoid duplicates
                extraction_tasks = []
                for role, content, message_obj in [
                    ("user", user_message.content, user_message)
                ]:
                    task = self.memory_extraction_service.extract_and_store_memories(
                        content=content,
                        memory_types=memory_types,
                        user_id=user_id,
                        session_id=session_id,
                        context={**extraction_context, "role": role},
                        preset_id=preset_id,
                    )
                    extraction_tasks.append((role, message_obj, task))

                # Execute extractions in parallel
                extraction_results = await asyncio.gather(
                    *[task for _, _, task in extraction_tasks], return_exceptions=True
                )

                # Process results
                for i, (role, message_obj, _) in enumerate(extraction_tasks):
                    extraction_result = extraction_results[i]

                    if isinstance(extraction_result, Exception):
                        print(
                            f"Error extracting memories for {role}: {extraction_result}"
                        )
                        continue

                    print(f"\n=== EXTRACTION RESULT (STREAM) ===")
                    print(f"Role: {role}")
                    print(f"Extracted: {extraction_result.get('extracted', {})}")
                    print(f"Stored: {extraction_result.get('stored', {})}")
                    print("=================================\n")

                    # Get stored memory details
                    stored_memories = []
                    for memory_type, memory_ids in extraction_result.get(
                        "stored", {}
                    ).items():
                        # Fetch the actual memory details for each ID
                        for memory_id in memory_ids:
                            try:
                                memory = await self.memory_service.get_memory(
                                    memory_id, user_id
                                )
                                if memory:
                                    print(f"\n=== FETCHED MEMORY ===")
                                    print(f"ID: {memory.id}")
                                    print(f"Input: {memory.input}")
                                    print(f"Type: {memory.memory_type}")
                                    print("=====================\n")
                                    memory_snapshot = {
                                        "id": memory.id,
                                        "content": memory.input,
                                        "memory_type": memory.memory_type.value,
                                        "tags": memory.tags,
                                        "confidence": memory.confidence,
                                    }
                                    stored_memories.append(memory_snapshot)

                                    # Store in extracted memories collection
                                    extracted_memory_doc = ChatExtractedMemory(
                                        id=str(uuid.uuid4()),  # New ID for the snapshot
                                        chat_message_id=message_obj.id,
                                        session_id=session_id,
                                        user_id=user_id,
                                        content=memory.input,
                                        memory_type=memory.memory_type,
                                        tags=memory.tags,
                                        confidence=memory.confidence,
                                        original_memory_id=memory.id,
                                        summary=getattr(memory, "summary", None),
                                        metadata=getattr(memory, "metadata", {}),
                                    )
                                    await self.extracted_memories_collection.insert_one(
                                        extracted_memory_doc.dict()
                                    )

                            except Exception as e:
                                # If we can't fetch details, just store the ID
                                stored_memories.append(
                                    {
                                        "id": memory_id,
                                        "content": extraction_result.get(
                                            "extracted", {}
                                        )
                                        .get(memory_type, {})
                                        .get("text", ""),
                                        "memory_type": memory_type,
                                        "tags": [],
                                        "confidence": 1.0,
                                    }
                                )

                    # Don't store full memories in message, they're in extracted_memories collection
                    if stored_memories:
                        message_obj.memories_extracted = ["extracted"]  # Marker
                    extracted_memories.extend(stored_memories)

        except Exception as e:
            print(f"\n=== ERROR IN MEMORY EXTRACTION ===")
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            print("==================================\n")
            # Continue without extracted memories

        # Save assistant message
        print("\n=== SAVING ASSISTANT MESSAGE ===")
        print(f"Message ID: {assistant_message.id}")
        print(f"Extracted memories count: {len(extracted_memories)}")
        print("================================\n")
        await self.messages_collection.insert_one(assistant_message.dict())

        # Update session
        await self._update_session_summary(
            session_id, assistant_content, len(extracted_memories)
        )

        # Yield final message info with enhanced memory details
        final_chunk = {
            "type": "assistant_message",
            "message": {
                "id": assistant_message.id,
                "role": "assistant",
                "content": assistant_content,
                "timestamp": assistant_message.timestamp.isoformat(),
                "memories_extracted": assistant_message.memories_extracted,
                "context_used": assistant_message.context_used,
            },
            "summary": {
                "memories_extracted": extracted_memories,
                "context_used": assistant_message.context_used,
            },
        }
        print(f"\n=== YIELDING FINAL CHUNK ===")
        print(f"Type: {final_chunk['type']}")
        print(
            f"Memories extracted: {len(assistant_message.memories_extracted) if assistant_message.memories_extracted else 0}"
        )
        print(
            f"Context used: {len(assistant_message.context_used) if assistant_message.context_used else 0}"
        )
        print("==========================\n")
        yield final_chunk

    def _get_system_prompt(
        self, context_memory_ids: List[str], context_memories: List[Any]
    ) -> str:
        """Get system prompt with memory context"""
        base_prompt = """You are a helpful AI assistant with access to the user's memory system. 
        You MUST use the provided memories to personalize your responses. When the user asks for suggestions or recommendations, 
        prioritize information from their memories over generic responses."""

        if context_memories:
            memory_context = (
                "\n\n=== USER'S PERSONAL MEMORIES (USE THESE IN YOUR RESPONSE) ===\n"
            )
            for i, memory in enumerate(context_memories, 1):
                # Use summary for SearchResult objects, or content if available
                text = memory.summary if hasattr(memory, "summary") else str(memory)
                if hasattr(memory, "content") and memory.content:
                    text = memory.content
                memory_context += f"{i}. {text}\n"

            memory_context += "\nIMPORTANT: Use the above memories to provide personalized responses based on what you know about the user.\n"

            return base_prompt + memory_context

        return base_prompt

    def _format_memory_context(self, search_results: List[Any]) -> str:
        """Format memory search results into context text"""
        if not search_results:
            return ""

        context_parts = []
        for i, result in enumerate(search_results, 1):
            # Extract text content from result
            text = result.summary if hasattr(result, "summary") else str(result)
            if hasattr(result, "content") and result.content:
                text = result.content

            context_parts.append(f"{i}. {text}")

        return "\n".join(context_parts)

    async def _get_user_name(self, user_id: str) -> str:
        """Get user name (placeholder - implement based on your user system)"""
        # TODO: Implement actual user lookup
        return f"User {user_id[:8]}"

    async def delete_session(self, session_id: str, user_id: str) -> bool:
        """Soft delete a chat session"""
        result = await self.sessions_collection.update_one(
            {"id": session_id, "user_id": user_id},
            {"$set": {"status": ChatSessionStatus.DELETED.value}},
        )

        return result.modified_count > 0

    async def hard_delete_session(
        self, session_id: str, user_id: str
    ) -> Dict[str, int]:
        """Permanently delete a single chat session and its messages"""
        # Verify session belongs to user
        session = await self.sessions_collection.find_one(
            {"id": session_id, "user_id": user_id}
        )

        if not session:
            return {"sessions": 0, "messages": 0, "extracted_memories": 0}

        # Delete extracted memories for this session
        extracted_memories_result = (
            await self.extracted_memories_collection.delete_many(
                {"session_id": session_id}
            )
        )

        # Delete all messages for this session
        messages_result = await self.messages_collection.delete_many(
            {"session_id": session_id}
        )

        # Delete the session
        session_result = await self.sessions_collection.delete_one(
            {"id": session_id, "user_id": user_id}
        )

        return {
            "sessions": session_result.deleted_count,
            "messages": messages_result.deleted_count,
            "extracted_memories": extracted_memories_result.deleted_count,
        }

    async def delete_all_sessions(self, user_id: str) -> int:
        """Soft delete all chat sessions for a user"""
        result = await self.sessions_collection.update_many(
            {"user_id": user_id, "status": {"$ne": ChatSessionStatus.DELETED.value}},
            {"$set": {"status": ChatSessionStatus.DELETED.value}},
        )

        return result.modified_count

    async def hard_delete_all_sessions(self, user_id: str) -> Dict[str, int]:
        """Permanently delete all chat sessions and messages for a user"""
        # Get all session IDs for the user
        sessions = await self.sessions_collection.find(
            {"user_id": user_id}, {"id": 1}
        ).to_list(None)

        session_ids = [s["id"] for s in sessions]

        # Delete extracted memories for these sessions
        extracted_memories_result = (
            await self.extracted_memories_collection.delete_many(
                {"session_id": {"$in": session_ids}}
            )
        )

        # Delete all messages for these sessions
        messages_result = await self.messages_collection.delete_many(
            {"session_id": {"$in": session_ids}}
        )

        # Delete all sessions
        sessions_result = await self.sessions_collection.delete_many(
            {"user_id": user_id}
        )

        return {
            "sessions": sessions_result.deleted_count,
            "messages": messages_result.deleted_count,
            "extracted_memories": extracted_memories_result.deleted_count,
        }

    async def extract_memories_from_session(
        self,
        session_id: str,
        user_id: str,
        message_ids: Optional[List[str]] = None,
        force: bool = False,
    ) -> Dict[str, Any]:
        """Extract memories from an entire chat session or specific messages"""
        # Verify session belongs to user
        session = await self.get_session(session_id, user_id)
        if not session:
            raise ValueError("Session not found")

        # Get messages to process
        query = {"session_id": session_id}
        if message_ids:
            query["id"] = {"$in": message_ids}
        if not force:
            query["memories_extracted"] = {
                "$size": 0
            }  # Only messages without extracted memories

        cursor = self.messages_collection.find(query)

        messages_processed = 0
        all_extracted_memories = []

        # Get preset for extraction
        preset_id = session.metadata.get("llm_preset_id")
        preset = None
        if preset_id:
            preset = await self.llm_service.get_preset(preset_id, user_id)

        # Determine memory types to extract
        memory_types = preset.memory_extraction_types if preset else ["chat_memory"]

        async for doc in cursor:
            message = ChatMessage(**doc)

            # Only extract from user messages to avoid duplicates
            if message.role != ChatRole.USER:
                continue

            # Build context for extraction
            extraction_context = {
                "user_name": await self._get_user_name(user_id),
                "session_id": session_id,
                "timestamp": message.timestamp,
                "session_context": session.title,
                "role": message.role.value
                if isinstance(message.role, ChatRole)
                else message.role,
            }

            # Extract and store memories
            extraction_result = (
                await self.memory_extraction_service.extract_and_store_memories(
                    content=message.content,
                    memory_types=memory_types,
                    user_id=user_id,
                    session_id=session_id,
                    context=extraction_context,
                    preset_id=preset_id,
                )
            )

            # Get stored memory IDs
            stored_ids = []
            for memory_type, ids in extraction_result.get("stored", {}).items():
                stored_ids.extend(ids)

            if stored_ids:
                # Update message with extracted memories
                await self.messages_collection.update_one(
                    {"id": message.id}, {"$set": {"memories_extracted": stored_ids}}
                )

                all_extracted_memories.extend(stored_ids)

            messages_processed += 1

        # Update session total
        if all_extracted_memories:
            await self.sessions_collection.update_one(
                {"id": session_id},
                {"$inc": {"total_memories_extracted": len(all_extracted_memories)}},
            )

        return {
            "session_id": session_id,
            "messages_processed": messages_processed,
            "memories_extracted": all_extracted_memories,
            "extraction_summary": {
                "total_memories": len(all_extracted_memories),
                "by_role": {
                    "user": len(
                        [m for m in all_extracted_memories if ":user" in str(m)]
                    ),
                    "assistant": len(
                        [m for m in all_extracted_memories if ":assistant" in str(m)]
                    ),
                },
            },
        }

    async def update_session(
        self,
        session_id: str,
        user_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Update a chat session"""
        update_data = {"updated_at": datetime.utcnow()}

        if title is not None:
            update_data["title"] = title
        if metadata is not None:
            update_data["metadata"] = metadata

        result = await self.sessions_collection.update_one(
            {"id": session_id, "user_id": user_id}, {"$set": update_data}
        )

        return result.modified_count > 0
