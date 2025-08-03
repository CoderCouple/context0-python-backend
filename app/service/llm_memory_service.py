"""Service for managing memory in external LLM services"""

import logging
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from app.common.enum.llm_vendor import LLMVendor
from app.common.enum.memory import MemoryType
from app.common.enum.memory_category import MemoryCategory
from app.common.enum.memory_emotion import MemoryEmotion
from app.memory.engine.memory_engine import MemoryEngine
from app.api.v1.request.memory_request import SearchQuery
from app.common.auth.auth import UserContext

logger = logging.getLogger(__name__)


class LLMMemoryService:
    """Service to handle memory operations for different LLM vendors"""

    def __init__(self, user_context: Optional[UserContext] = None):
        # Load API keys from environment
        self.api_keys = {
            LLMVendor.OPENAI: os.getenv("OPENAI_API_KEY"),
            LLMVendor.ANTHROPIC: os.getenv("ANTHROPIC_API_KEY"),
            LLMVendor.GOOGLE: os.getenv("GOOGLE_API_KEY"),
        }
        # Get memory engine instance
        self.memory_engine = MemoryEngine.get_instance()
        # Store user context
        self.user_context = user_context

    async def delete_memory(
        self,
        vendor: LLMVendor,
        conversation_id: Optional[str] = None,
        message_ids: Optional[List[str]] = None,
        delete_all: bool = False,
        api_key: Optional[str] = None,
    ) -> Dict:
        """Delete memory from LLM vendor"""

        # Use provided API key or fall back to environment
        api_key = api_key or self.api_keys.get(vendor)

        if not api_key:
            return {
                "success": False,
                "vendor": vendor,
                "operation": "delete_memory",
                "message": f"API key required for {vendor.value}",
            }

        if vendor == LLMVendor.OPENAI:
            return await self._delete_openai_memory(
                api_key, conversation_id, message_ids, delete_all
            )
        elif vendor == LLMVendor.ANTHROPIC:
            return await self._delete_anthropic_memory(
                api_key, conversation_id, message_ids, delete_all
            )
        elif vendor == LLMVendor.GOOGLE:
            return await self._delete_google_memory(
                api_key, conversation_id, message_ids, delete_all
            )
        else:
            raise NotImplementedError(f"Vendor {vendor} not supported")

    async def _delete_openai_memory(
        self,
        api_key: str,
        conversation_id: Optional[str],
        message_ids: Optional[List[str]],
        delete_all: bool,
    ) -> Dict:
        """Delete memory from OpenAI/ChatGPT

        Note: OpenAI's API has limited conversation management.
        Main options:
        1. Assistants API - Can delete threads and messages
        2. Chat Completions API - Stateless, no persistent memory to delete
        3. Fine-tuning data - Can be deleted through dashboard
        """
        try:
            import openai

            openai.api_key = api_key

            deleted_count = 0

            # For Assistants API threads
            if conversation_id and conversation_id.startswith("thread_"):
                try:
                    # Delete specific messages if provided
                    if message_ids:
                        for msg_id in message_ids:
                            openai.beta.threads.messages.delete(
                                thread_id=conversation_id, message_id=msg_id
                            )
                            deleted_count += 1
                    else:
                        # Delete entire thread
                        openai.beta.threads.delete(conversation_id)
                        deleted_count = 1

                    return {
                        "success": True,
                        "vendor": LLMVendor.OPENAI,
                        "operation": "delete_memory",
                        "message": f"Deleted {deleted_count} items from ChatGPT",
                        "deleted_count": deleted_count,
                        "details": {
                            "thread_id": conversation_id,
                            "message_ids": message_ids,
                        },
                    }
                except Exception as e:
                    logger.error(f"Error deleting OpenAI thread: {e}")
                    return {
                        "success": False,
                        "vendor": LLMVendor.OPENAI,
                        "operation": "delete_memory",
                        "message": f"Failed to delete thread: {str(e)}",
                    }

            # For delete_all with Assistants API
            if delete_all:
                try:
                    # List and delete all threads (limited to what API returns)
                    threads = openai.beta.threads.list(limit=100)
                    for thread in threads.data:
                        openai.beta.threads.delete(thread.id)
                        deleted_count += 1

                    return {
                        "success": True,
                        "vendor": LLMVendor.OPENAI,
                        "operation": "delete_memory",
                        "message": f"Deleted {deleted_count} threads from ChatGPT",
                        "deleted_count": deleted_count,
                    }
                except Exception as e:
                    logger.error(f"Error deleting all OpenAI threads: {e}")

            return {
                "success": False,
                "vendor": LLMVendor.OPENAI,
                "operation": "delete_memory",
                "message": "ChatGPT web conversations cannot be deleted via API. Use Assistants API for deletable threads.",
            }

        except ImportError:
            return {
                "success": False,
                "vendor": LLMVendor.OPENAI,
                "operation": "delete_memory",
                "message": "OpenAI package not installed. Run: pip install openai",
            }

    async def _delete_anthropic_memory(
        self,
        api_key: str,
        conversation_id: Optional[str],
        message_ids: Optional[List[str]],
        delete_all: bool,
    ) -> Dict:
        """Delete memory from Anthropic/Claude

        Note: Claude's API is stateless - each API call is independent.
        There's no persistent conversation storage in the API to delete.
        Conversations in claude.ai web interface cannot be deleted via API.
        """
        return {
            "success": False,
            "vendor": LLMVendor.ANTHROPIC,
            "operation": "delete_memory",
            "message": "Claude API is stateless - no persistent memory to delete. Web conversations at claude.ai cannot be managed via API.",
            "details": {
                "note": "For Claude API usage, simply don't include previous messages in your API calls to 'forget' context"
            },
        }

    async def _delete_google_memory(
        self,
        api_key: str,
        conversation_id: Optional[str],
        message_ids: Optional[List[str]],
        delete_all: bool,
    ) -> Dict:
        """Delete memory from Google/Gemini

        Note: Gemini API maintains chat sessions that can be managed.
        """
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)

            # Gemini uses chat sessions which are ephemeral
            # There's no persistent storage to delete in the API
            return {
                "success": True,
                "vendor": LLMVendor.GOOGLE,
                "operation": "delete_memory",
                "message": "Gemini chat sessions are ephemeral. Create new chat session to reset context.",
                "details": {
                    "note": "Gemini API doesn't store conversations. Each chat session exists only during runtime."
                },
            }

        except ImportError:
            return {
                "success": False,
                "vendor": LLMVendor.GOOGLE,
                "operation": "delete_memory",
                "message": "Google Generative AI package not installed. Run: pip install google-generativeai",
            }

    async def list_conversations(
        self, vendor: LLMVendor, api_key: Optional[str] = None
    ) -> Dict:
        """List conversations from LLM vendor"""

        api_key = api_key or self.api_keys.get(vendor)

        if not api_key:
            return {
                "success": False,
                "conversations": [],
                "message": f"API key required for {vendor.value}",
            }

        if vendor == LLMVendor.OPENAI:
            return await self._list_openai_conversations(api_key)
        else:
            return {
                "success": False,
                "conversations": [],
                "message": f"{vendor.value} does not support listing conversations via API",
            }

    async def _list_openai_conversations(self, api_key: str) -> Dict:
        """List OpenAI Assistant threads"""
        try:
            import openai

            openai.api_key = api_key

            threads = openai.beta.threads.list(limit=100)
            conversations = []

            for thread in threads.data:
                conversations.append(
                    {
                        "id": thread.id,
                        "created_at": thread.created_at,
                        "metadata": thread.metadata,
                    }
                )

            return {
                "success": True,
                "conversations": conversations,
                "count": len(conversations),
                "message": f"Retrieved {len(conversations)} threads",
            }

        except Exception as e:
            logger.error(f"Error listing OpenAI threads: {e}")
            return {
                "success": False,
                "conversations": [],
                "message": f"Failed to list threads: {str(e)}",
            }

    async def clear_context(
        self,
        vendor: LLMVendor,
        session_id: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict:
        """Clear context for a session"""

        return {
            "success": True,
            "vendor": vendor,
            "operation": "clear_context",
            "message": f"Context cleared for {vendor.value}. Start new conversation without including previous messages.",
            "details": {
                "note": "All major LLM APIs are stateless. To 'clear' context, simply start a new conversation without previous messages."
            },
        }

    async def get_formatted_memories(
        self,
        vendor: LLMVendor,
        user_id: str,
        query: str,
        memory_types: Optional[List[MemoryType]] = None,
        categories: Optional[List[MemoryCategory]] = None,
        emotions: Optional[List[MemoryEmotion]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
    ) -> Dict:
        """Retrieve and format memories from Context Zero for the specified vendor"""
        try:
            # Ensure memory engine is initialized
            if not self.memory_engine._initialized:
                await self.memory_engine.initialize()

            # Create search query
            search_query = SearchQuery(
                query=query,
                user_id=user_id,
                memory_types=memory_types,
                limit=limit,
                tags=tags,
            )

            # Handle categories and emotions
            if categories and len(categories) > 0:
                search_query.category = categories[0]  # Primary category
            if emotions and len(emotions) > 0:
                search_query.emotion = emotions[0]  # Primary emotion

            # Search for relevant memories
            search_response = await self.memory_engine.search_memories(search_query)

            if not search_response.success:
                return {
                    "success": False,
                    "memory_count": 0,
                    "formatted_context": "",
                    "message": "Failed to retrieve memories",
                }

            # Format memories based on vendor
            if vendor == LLMVendor.OPENAI:
                formatted = self._format_memories_for_openai(search_response.results)
            elif vendor == LLMVendor.ANTHROPIC:
                formatted = self._format_memories_for_anthropic(search_response.results)
            elif vendor == LLMVendor.GOOGLE:
                formatted = self._format_memories_for_google(search_response.results)
            else:
                formatted = ""

            return {
                "success": True,
                "memory_count": len(search_response.results),
                "formatted_context": formatted,
                "raw_memories": [
                    {
                        "id": result.memory_id,
                        "text": result.text,
                        "summary": result.summary,
                        "created_at": result.created_at,
                        "score": result.score,
                    }
                    for result in search_response.results
                ],
            }

        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return {
                "success": False,
                "memory_count": 0,
                "formatted_context": "",
                "message": f"Error: {str(e)}",
            }

    def _format_memories_for_openai(self, memories) -> str:
        """Format memories for OpenAI/ChatGPT"""
        if not memories:
            return ""

        context = "## Relevant Context from Previous Conversations:\n\n"
        for i, memory in enumerate(memories, 1):
            context += f"**Memory {i}** (Relevance: {memory.score:.2f}):\n"
            if memory.summary:
                context += f"Summary: {memory.summary}\n"
            context += f"Content: {memory.text}\n"
            context += f"Created: {memory.created_at}\n\n"

        context += "---\n\nPlease consider the above context when responding.\n\n"
        return context

    def _format_memories_for_anthropic(self, memories) -> str:
        """Format memories for Anthropic/Claude"""
        if not memories:
            return ""

        context = "I have access to the following relevant information from our previous interactions:\n\n"
        for i, memory in enumerate(memories, 1):
            context += f'<memory index="{i}" relevance="{memory.score:.2f}">\n'
            if memory.summary:
                context += f"<summary>{memory.summary}</summary>\n"
            context += f"<content>{memory.text}</content>\n"
            context += f"<timestamp>{memory.created_at}</timestamp>\n"
            context += "</memory>\n\n"

        context += "I'll use this context to provide a more informed response.\n\n"
        return context

    def _format_memories_for_google(self, memories) -> str:
        """Format memories for Google/Gemini"""
        if not memories:
            return ""

        context = "Previous conversation context:\n\n"
        for i, memory in enumerate(memories, 1):
            context += f"{i}. "
            if memory.summary:
                context += f"{memory.summary} - "
            context += f"{memory.text}\n"
            context += (
                f"   (Relevance: {memory.score:.2f}, Date: {memory.created_at})\n\n"
            )

        return context

    async def inject_memories_and_chat(
        self,
        vendor: LLMVendor,
        user_id: str,
        message: str,
        memory_types: Optional[List[MemoryType]] = None,
        limit: int = 10,
        include_summary: bool = True,
        include_timeline: bool = False,
        api_key: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict:
        """Retrieve memories and send to LLM with user's message"""

        # Get API key
        api_key = api_key or self.api_keys.get(vendor)
        if not api_key:
            return {
                "success": False,
                "vendor": vendor,
                "response": "",
                "memories_used": 0,
                "message": f"API key required for {vendor.value}",
            }

        # Get formatted memories
        memories_result = await self.get_formatted_memories(
            vendor=vendor,
            user_id=user_id,
            query=message,
            memory_types=memory_types,
            categories=None,  # Could be passed if needed
            emotions=None,  # Could be passed if needed
            tags=None,  # Could be passed if needed
            limit=limit,
        )

        if not memories_result["success"]:
            return {
                "success": False,
                "vendor": vendor,
                "response": "",
                "memories_used": 0,
                "message": memories_result.get(
                    "message", "Failed to retrieve memories"
                ),
            }

        # Call appropriate LLM with context
        if vendor == LLMVendor.OPENAI:
            return await self._chat_with_openai(
                message=message,
                context=memories_result["formatted_context"],
                memory_count=memories_result["memory_count"],
                api_key=api_key,
                session_id=session_id,
            )
        elif vendor == LLMVendor.ANTHROPIC:
            return await self._chat_with_anthropic(
                message=message,
                context=memories_result["formatted_context"],
                memory_count=memories_result["memory_count"],
                api_key=api_key,
                session_id=session_id,
            )
        elif vendor == LLMVendor.GOOGLE:
            return await self._chat_with_google(
                message=message,
                context=memories_result["formatted_context"],
                memory_count=memories_result["memory_count"],
                api_key=api_key,
                session_id=session_id,
            )

    async def _chat_with_openai(
        self,
        message: str,
        context: str,
        memory_count: int,
        api_key: str,
        session_id: Optional[str],
    ) -> Dict:
        """Send message with context to OpenAI"""
        try:
            import openai

            openai.api_key = api_key

            # Combine context and message
            full_message = context + message if context else message

            # Create chat completion
            response = openai.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant with access to the user's previous conversation history.",
                    },
                    {"role": "user", "content": full_message},
                ],
                temperature=0.7,
                max_tokens=1000,
            )

            return {
                "success": True,
                "vendor": LLMVendor.OPENAI,
                "response": response.choices[0].message.content,
                "memories_used": memory_count,
                "session_id": session_id or f"openai_{datetime.now().isoformat()}",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                },
            }

        except ImportError:
            return {
                "success": False,
                "vendor": LLMVendor.OPENAI,
                "response": "",
                "memories_used": 0,
                "message": "OpenAI package not installed. Run: pip install openai",
            }
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            return {
                "success": False,
                "vendor": LLMVendor.OPENAI,
                "response": "",
                "memories_used": 0,
                "message": f"OpenAI error: {str(e)}",
            }

    async def _chat_with_anthropic(
        self,
        message: str,
        context: str,
        memory_count: int,
        api_key: str,
        session_id: Optional[str],
    ) -> Dict:
        """Send message with context to Anthropic"""
        try:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key)

            # Combine context and message
            full_message = context + message if context else message

            # Create completion
            response = client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                messages=[{"role": "user", "content": full_message}],
            )

            return {
                "success": True,
                "vendor": LLMVendor.ANTHROPIC,
                "response": response.content[0].text,
                "memories_used": memory_count,
                "session_id": session_id or f"claude_{datetime.now().isoformat()}",
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                },
            }

        except ImportError:
            return {
                "success": False,
                "vendor": LLMVendor.ANTHROPIC,
                "response": "",
                "memories_used": 0,
                "message": "Anthropic package not installed. Run: pip install anthropic",
            }
        except Exception as e:
            logger.error(f"Anthropic chat error: {e}")
            return {
                "success": False,
                "vendor": LLMVendor.ANTHROPIC,
                "response": "",
                "memories_used": 0,
                "message": f"Anthropic error: {str(e)}",
            }

    async def _chat_with_google(
        self,
        message: str,
        context: str,
        memory_count: int,
        api_key: str,
        session_id: Optional[str],
    ) -> Dict:
        """Send message with context to Google Gemini"""
        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)

            # Create model
            model = genai.GenerativeModel("gemini-pro")

            # Combine context and message
            full_message = context + message if context else message

            # Generate response
            response = model.generate_content(full_message)

            return {
                "success": True,
                "vendor": LLMVendor.GOOGLE,
                "response": response.text,
                "memories_used": memory_count,
                "session_id": session_id or f"gemini_{datetime.now().isoformat()}",
            }

        except ImportError:
            return {
                "success": False,
                "vendor": LLMVendor.GOOGLE,
                "response": "",
                "memories_used": 0,
                "message": "Google Generative AI package not installed. Run: pip install google-generativeai",
            }
        except Exception as e:
            logger.error(f"Google Gemini chat error: {e}")
            return {
                "success": False,
                "vendor": LLMVendor.GOOGLE,
                "response": "",
                "memories_used": 0,
                "message": f"Gemini error: {str(e)}",
            }

    async def sync_memories_from_llms(
        self,
        user_id: str,
        vendors: List[LLMVendor],
        api_keys: Optional[Dict[LLMVendor, str]],
        sync_mode: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        auto_categorize: bool,
        detect_emotions: bool,
        conversation_ids: Optional[Dict[LLMVendor, List[str]]],
        settings: Any,
    ) -> Dict:
        """Sync memories from multiple LLM vendors to Context Zero"""
        import uuid
        import time
        from app.api.v1.request.memory_request import MemoryRecordInput
        from app.service.memory_categorization_service import (
            MemoryCategoricationService,
        )

        sync_id = str(uuid.uuid4())
        started_at = datetime.now()
        vendor_results = []
        total_synced = 0
        total_errors = 0

        # Ensure memory engine is initialized
        if not self.memory_engine._initialized:
            await self.memory_engine.initialize()

        # Store sync status (in production, use Redis or DB)
        self.sync_status = {
            sync_id: {
                "status": "running",
                "progress": 0.0,
                "current_vendor": None,
                "processed_count": 0,
                "total_count": None,
                "errors": [],
            }
        }

        try:
            for vendor in vendors:
                self.sync_status[sync_id]["current_vendor"] = vendor
                vendor_start = time.time()

                # Get API key
                api_key = None
                if api_keys and vendor in api_keys:
                    api_key = api_keys[vendor]
                else:
                    api_key = self.api_keys.get(vendor)

                if not api_key:
                    vendor_results.append(
                        {
                            "vendor": vendor,
                            "success": False,
                            "total_conversations": 0,
                            "total_messages": 0,
                            "synced_count": 0,
                            "skipped_count": 0,
                            "error_count": 1,
                            "errors": [f"API key not provided for {vendor.value}"],
                            "synced_memories": [],
                        }
                    )
                    total_errors += 1
                    continue

                # Extract memories based on vendor
                if vendor == LLMVendor.OPENAI:
                    vendor_result = await self._sync_openai_memories(
                        user_id,
                        api_key,
                        sync_mode,
                        start_date,
                        end_date,
                        auto_categorize,
                        detect_emotions,
                        conversation_ids.get(vendor) if conversation_ids else None,
                        settings,
                    )
                elif vendor == LLMVendor.ANTHROPIC:
                    vendor_result = await self._sync_anthropic_memories(
                        user_id,
                        api_key,
                        sync_mode,
                        start_date,
                        end_date,
                        auto_categorize,
                        detect_emotions,
                        conversation_ids.get(vendor) if conversation_ids else None,
                        settings,
                    )
                elif vendor == LLMVendor.GOOGLE:
                    vendor_result = await self._sync_google_memories(
                        user_id,
                        api_key,
                        sync_mode,
                        start_date,
                        end_date,
                        auto_categorize,
                        detect_emotions,
                        conversation_ids.get(vendor) if conversation_ids else None,
                        settings,
                    )
                else:
                    vendor_result = {
                        "vendor": vendor,
                        "success": False,
                        "error_count": 1,
                        "errors": [f"Vendor {vendor.value} not supported"],
                    }

                vendor_results.append(vendor_result)
                total_synced += vendor_result.get("synced_count", 0)
                total_errors += vendor_result.get("error_count", 0)

                # Update progress
                progress = (vendors.index(vendor) + 1) / len(vendors) * 100
                self.sync_status[sync_id]["progress"] = progress
                self.sync_status[sync_id]["processed_count"] = total_synced

            completed_at = datetime.now()
            duration = (completed_at - started_at).total_seconds()

            # Mark sync as completed
            self.sync_status[sync_id]["status"] = "completed"
            self.sync_status[sync_id]["progress"] = 100.0

            return {
                "success": total_errors == 0,
                "sync_id": sync_id,
                "started_at": started_at,
                "completed_at": completed_at,
                "duration_seconds": duration,
                "vendor_results": vendor_results,
                "total_synced": total_synced,
                "total_errors": total_errors,
                "sync_mode": sync_mode,
            }

        except Exception as e:
            logger.error(f"Sync error: {e}")
            self.sync_status[sync_id]["status"] = "failed"
            self.sync_status[sync_id]["errors"].append(str(e))

            return {
                "success": False,
                "sync_id": sync_id,
                "started_at": started_at,
                "completed_at": datetime.now(),
                "duration_seconds": (datetime.now() - started_at).total_seconds(),
                "vendor_results": vendor_results,
                "total_synced": total_synced,
                "total_errors": total_errors + 1,
                "sync_mode": sync_mode,
                "message": f"Sync failed: {str(e)}",
            }

    async def _sync_openai_memories(
        self,
        user_id: str,
        api_key: str,
        sync_mode: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        auto_categorize: bool,
        detect_emotions: bool,
        conversation_ids: Optional[List[str]],
        settings: Any,
    ) -> Dict:
        """Sync memories from OpenAI/ChatGPT"""
        try:
            import openai

            openai.api_key = api_key

            synced_memories = []
            total_conversations = 0
            total_messages = 0
            synced_count = 0
            skipped_count = 0
            errors = []

            # For OpenAI Assistants API
            try:
                # List threads (conversations)
                threads = openai.beta.threads.list(limit=100)
                total_conversations = len(threads.data)

                for thread in threads.data:
                    if conversation_ids and thread.id not in conversation_ids:
                        continue

                    try:
                        # Get messages from thread
                        messages = openai.beta.threads.messages.list(
                            thread_id=thread.id, limit=100
                        )

                        for message in messages.data:
                            total_messages += 1

                            # Check date filters
                            msg_time = datetime.fromtimestamp(message.created_at)
                            if start_date and msg_time < start_date:
                                skipped_count += 1
                                continue
                            if end_date and msg_time > end_date:
                                skipped_count += 1
                                continue

                            # Extract content
                            content = ""
                            if hasattr(message, "content") and message.content:
                                for content_item in message.content:
                                    if hasattr(content_item, "text"):
                                        content += content_item.text.value + "\n"

                            if not content.strip():
                                skipped_count += 1
                                continue

                            # Create memory record
                            from app.api.v1.request.memory_request import (
                                MemoryRecordInput,
                            )

                            memory_input = MemoryRecordInput(
                                user_id=user_id,
                                session_id=f"openai_{thread.id}",
                                text=content.strip(),
                                tags=[
                                    f"openai",
                                    f"thread:{thread.id}",
                                    f"role:{message.role}",
                                ],
                                metadata={
                                    "source": "openai",
                                    "thread_id": thread.id,
                                    "message_id": message.id,
                                    "role": message.role,
                                    "created_at": message.created_at,
                                    "original_timestamp": msg_time.isoformat(),
                                },
                            )

                            # Auto-categorize if enabled
                            if auto_categorize or detect_emotions:
                                from app.service.memory_categorization_service import (
                                    MemoryCategoricationService,
                                )

                                memory_input = (
                                    MemoryCategoricationService.enrich_memory_input(
                                        memory_input
                                    )
                                )

                            # Add memory to Context Zero
                            response = await self.memory_engine.add_memory(memory_input)

                            if response.success:
                                synced_count += 1
                                synced_memories.append(
                                    {
                                        "vendor": LLMVendor.OPENAI,
                                        "original_id": message.id,
                                        "context_zero_id": response.memory_id,
                                        "timestamp": msg_time,
                                        "role": message.role,
                                        "content": content[:200] + "..."
                                        if len(content) > 200
                                        else content,
                                        "category": memory_input.category.value
                                        if memory_input.category
                                        else None,
                                        "emotion": memory_input.emotion.value
                                        if memory_input.emotion
                                        else None,
                                        "metadata": memory_input.metadata,
                                    }
                                )
                            else:
                                errors.append(
                                    f"Failed to add message {message.id}: {response.message}"
                                )

                    except Exception as e:
                        errors.append(f"Error processing thread {thread.id}: {str(e)}")

            except Exception as e:
                errors.append(f"OpenAI API error: {str(e)}")

            return {
                "vendor": LLMVendor.OPENAI,
                "success": len(errors) == 0,
                "total_conversations": total_conversations,
                "total_messages": total_messages,
                "synced_count": synced_count,
                "skipped_count": skipped_count,
                "error_count": len(errors),
                "errors": errors,
                "synced_memories": synced_memories[
                    :10
                ],  # Limit to first 10 for response
            }

        except ImportError:
            return {
                "vendor": LLMVendor.OPENAI,
                "success": False,
                "error_count": 1,
                "errors": ["OpenAI package not installed"],
            }

    async def _sync_anthropic_memories(
        self,
        user_id: str,
        api_key: str,
        sync_mode: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        auto_categorize: bool,
        detect_emotions: bool,
        conversation_ids: Optional[List[str]],
        settings: Any,
    ) -> Dict:
        """Sync memories from Anthropic/Claude"""
        # Claude doesn't have a conversation history API
        # This is a placeholder for future implementation
        return {
            "vendor": LLMVendor.ANTHROPIC,
            "success": False,
            "total_conversations": 0,
            "total_messages": 0,
            "synced_count": 0,
            "skipped_count": 0,
            "error_count": 1,
            "errors": [
                "Claude API doesn't support conversation history retrieval. Conversations are not stored server-side."
            ],
            "synced_memories": [],
        }

    async def _sync_google_memories(
        self,
        user_id: str,
        api_key: str,
        sync_mode: str,
        start_date: Optional[datetime],
        end_date: Optional[datetime],
        auto_categorize: bool,
        detect_emotions: bool,
        conversation_ids: Optional[List[str]],
        settings: Any,
    ) -> Dict:
        """Sync memories from Google/Gemini"""
        # Gemini doesn't have persistent conversation storage in the API
        # This is a placeholder for future implementation
        return {
            "vendor": LLMVendor.GOOGLE,
            "success": False,
            "total_conversations": 0,
            "total_messages": 0,
            "synced_count": 0,
            "skipped_count": 0,
            "error_count": 1,
            "errors": [
                "Gemini API doesn't support conversation history retrieval. Chat sessions are ephemeral."
            ],
            "synced_memories": [],
        }

    async def get_sync_status(self, sync_id: str, user_id: str) -> Optional[Dict]:
        """Get status of a sync operation"""
        # In production, retrieve from Redis or DB
        if hasattr(self, "sync_status") and sync_id in self.sync_status:
            return {"sync_id": sync_id, **self.sync_status[sync_id]}
        return None
