"""Advanced Q&A Service with multi-hop reasoning capabilities"""

import logging
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from app.api.v1.request.memory_request import SearchQuery
from app.api.v1.request.qa_request import ConversationRequest, QuestionRequest
from app.api.v1.response.qa_response import (
    ConversationResponse,
    MemoryContext,
    QAHealthResponse,
    QuestionResponse,
)
from app.memory.engine.memory_engine import MemoryEngine
from app.memory.utils.embeddings import get_embeddings
from app.reasoning.engine import MultiHopReasoningEngine
from app.reasoning.models import ReasoningChain, ReasoningResult

logger = logging.getLogger(__name__)


class QAService:
    """Advanced Q&A service with multi-hop reasoning capabilities"""

    def __init__(self):
        """Initialize Q&A service with reasoning engine"""
        self._memory_engine: Optional[MemoryEngine] = None
        self._reasoning_engine: Optional[MultiHopReasoningEngine] = None
        # Initialize embeddings for question processing
        api_key = os.getenv("OPENAI_API_KEY")
        self.embedder = get_embeddings(model="text-embedding-3-small", api_key=api_key)

    @property
    def memory_engine(self) -> MemoryEngine:
        """Get memory engine instance (lazy initialization)"""
        if self._memory_engine is None:
            self._memory_engine = MemoryEngine.get_instance()
        return self._memory_engine

    @property
    def reasoning_engine(self) -> MultiHopReasoningEngine:
        """Get reasoning engine instance (lazy initialization)"""
        if self._reasoning_engine is None:
            self._reasoning_engine = MultiHopReasoningEngine()
        return self._reasoning_engine

    async def ask_question(self, request: QuestionRequest) -> QuestionResponse:
        """Answer a question using advanced multi-hop reasoning"""
        start_time = time.time()

        try:
            logger.info(
                f"Processing question with multi-hop reasoning for user {request.user_id}: {request.question}"
            )

            # Use the reasoning engine for advanced multi-hop reasoning
            reasoning_result = await self.reasoning_engine.reason_about_question(
                request.question,
                request.user_id,
                max_memories=request.max_memories,
                memory_types=request.memory_types,
                include_meta_memories=request.include_meta_memories,
                search_depth=request.search_depth,
            )

            # Convert reasoning result to QuestionResponse
            primary_chain = reasoning_result.primary_chain

            # Extract memory contexts from reasoning chain
            memory_contexts = []
            all_memory_refs = set()

            for step in primary_chain.reasoning_steps:
                all_memory_refs.update(step.memory_references)

            # Convert memory references to MemoryContext objects
            for memory_ref in list(all_memory_refs)[:10]:  # Limit to 10 contexts
                # Fetch actual memory data
                actual_memory = await self._get_actual_memory_content(
                    memory_ref, request.user_id
                )
                if actual_memory:
                    memory_contexts.append(
                        MemoryContext(
                            memory_id=memory_ref,
                            content=actual_memory.get(
                                "text", f"Memory content for {memory_ref}"
                            ),
                            summary=actual_memory.get("summary"),
                            memory_type=actual_memory.get("memory_type", "unknown"),
                            relevance_score=actual_memory.get("relevance_score", 0.8),
                            created_at=actual_memory.get("created_at", datetime.now()),
                            tags=actual_memory.get("tags", []),
                            source="multi_hop_reasoning",
                        )
                    )
                else:
                    # Fallback to placeholder if memory not found
                    memory_contexts.append(
                        MemoryContext(
                            memory_id=memory_ref,
                            content=f"Memory content for {memory_ref}",
                            summary=None,
                            memory_type="unknown",
                            relevance_score=0.8,
                            created_at=datetime.now(),
                            tags=[],
                            source="multi_hop_reasoning",
                        )
                    )

            processing_time = int((time.time() - start_time) * 1000)

            return QuestionResponse(
                question=request.question,
                answer=reasoning_result.synthesis,
                confidence=reasoning_result.primary_chain.overall_confidence,
                memories_found=primary_chain.total_memories_used,
                memories_used=len(memory_contexts),
                memory_contexts=memory_contexts,
                search_strategy=f"multi_hop_{request.search_depth}",
                processing_time_ms=processing_time,
                suggestions=reasoning_result.follow_up_questions,
                metadata={
                    "user_id": request.user_id,
                    "session_id": request.session_id,
                    "reasoning_chains": len(
                        [reasoning_result.primary_chain]
                        + reasoning_result.alternative_chains
                    ),
                    "contradictions": len(reasoning_result.contradictions),
                    "gaps": len(reasoning_result.gaps),
                    "confidence_distribution": reasoning_result.confidence_distribution,
                    "reasoning_metadata": reasoning_result.reasoning_metadata,
                },
            )

        except Exception as e:
            logger.error(f"Error processing question with reasoning engine: {e}")
            return QuestionResponse(
                question=request.question,
                answer=f"I'm sorry, I encountered an error while reasoning about your question: {str(e)}",
                confidence=0.0,
                memories_found=0,
                memories_used=0,
                memory_contexts=[],
                search_strategy=request.search_depth,
                processing_time_ms=int((time.time() - start_time) * 1000),
                suggestions=[
                    "Could you rephrase your question?",
                    "Try asking about something more specific.",
                ],
                metadata={"error": str(e)},
            )

    async def _search_memories(self, request: QuestionRequest) -> List[Dict[str, Any]]:
        """Search for relevant memories based on the question"""
        memories = []

        # Generate embedding for the question
        question_embedding = await self.embedder.aembed_query(request.question)

        # Search across different stores based on search depth
        if request.search_depth in ["semantic", "hybrid", "comprehensive"]:
            # Vector store search (semantic similarity)
            if self.memory_engine.vector_store:
                try:
                    vector_results = await self._search_vector_store(
                        question_embedding, request
                    )
                    memories.extend(vector_results)
                except Exception as e:
                    logger.warning(f"Vector store search failed: {e}")

        if request.search_depth in ["hybrid", "comprehensive"]:
            # Document store search (keyword/metadata based)
            if self.memory_engine.doc_store:
                try:
                    doc_results = await self._search_document_store(request)
                    memories.extend(doc_results)
                except Exception as e:
                    logger.warning(f"Document store search failed: {e}")

        if request.search_depth == "comprehensive":
            # Graph store search (relationship based)
            if self.memory_engine.graph_store:
                try:
                    graph_results = await self._search_graph_store(request)
                    memories.extend(graph_results)
                except Exception as e:
                    logger.warning(f"Graph store search failed: {e}")

        # Remove duplicates and sort by relevance
        unique_memories = self._deduplicate_memories(memories)
        sorted_memories = sorted(
            unique_memories, key=lambda x: x.get("relevance_score", 0), reverse=True
        )

        return sorted_memories[: request.max_memories]

    async def _search_vector_store(
        self, question_embedding: List[float], request: QuestionRequest
    ) -> List[Dict[str, Any]]:
        """Search using vector similarity"""
        # This would be implemented based on your vector store's search method
        # For now, returning empty list - this would need to be implemented based on your vector store
        return []

    async def _search_document_store(
        self, request: QuestionRequest
    ) -> List[Dict[str, Any]]:
        """Search using document store"""
        if not self.memory_engine.doc_store:
            return []

        try:
            # Build search query
            search_query = {
                "source_user_id": request.user_id,
            }

            if request.memory_types:
                search_query["memory_type"] = {"$in": request.memory_types}

            if request.session_id:
                search_query["source_session_id"] = request.session_id

            # Search memories
            results = await self.memory_engine.doc_store.search(
                search_query, request.max_memories
            )

            # Convert to standard format and calculate relevance
            formatted_results = []
            for memory in results:
                relevance = self._calculate_text_relevance(
                    request.question, memory.get("input", "")
                )
                formatted_results.append(
                    {
                        "memory_id": memory.get("id"),
                        "content": memory.get("input", ""),
                        "summary": memory.get("summary"),
                        "memory_type": memory.get("memory_type"),
                        "created_at": memory.get("created_at"),
                        "tags": memory.get("tags", []),
                        "relevance_score": relevance,
                        "source": "document_store",
                        "metadata": memory,
                    }
                )

            return formatted_results

        except Exception as e:
            logger.error(f"Document store search error: {e}")
            return []

    async def _search_graph_store(
        self, request: QuestionRequest
    ) -> List[Dict[str, Any]]:
        """Search using graph relationships"""
        # This would search for memories connected to entities mentioned in the question
        return []

    def _deduplicate_memories(
        self, memories: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Remove duplicate memories based on memory_id"""
        seen_ids = set()
        unique_memories = []

        for memory in memories:
            memory_id = memory.get("memory_id")
            if memory_id and memory_id not in seen_ids:
                seen_ids.add(memory_id)
                unique_memories.append(memory)

        return unique_memories

    def _calculate_text_relevance(self, question: str, content: str) -> float:
        """Calculate simple text relevance score"""
        if not content:
            return 0.0

        question_words = set(question.lower().split())
        content_words = set(content.lower().split())

        if not question_words:
            return 0.0

        intersection = question_words.intersection(content_words)
        return len(intersection) / len(question_words)

    async def _generate_answer(
        self, question: str, memories: List[Dict[str, Any]], user_id: str
    ) -> Tuple[str, float, List[MemoryContext]]:
        """Generate answer based on retrieved memories"""

        if not memories:
            return (
                "I don't have any relevant memories to answer that question. Could you provide more context or ask about something specific you've shared with me?",
                0.1,
                [],
            )

        # Convert memories to context objects
        memory_contexts = []
        context_text = ""

        for memory in memories[:5]:  # Use top 5 most relevant
            context = MemoryContext(
                memory_id=memory["memory_id"],
                content=memory["content"][:500]
                + ("..." if len(memory["content"]) > 500 else ""),
                summary=memory.get("summary"),
                memory_type=memory["memory_type"],
                relevance_score=memory["relevance_score"],
                created_at=memory.get("created_at", datetime.now()),
                tags=memory.get("tags", []),
                source=memory["source"],
            )
            memory_contexts.append(context)

            # Build context for answer generation
            context_text += f"Memory: {memory['content']}\n"
            if memory.get("summary"):
                context_text += f"Summary: {memory['summary']}\n"
            context_text += f"Type: {memory['memory_type']}\n\n"

        # For now, generate a simple answer based on the most relevant memory
        # In a production system, you'd use an LLM here
        most_relevant = memories[0]
        confidence = most_relevant["relevance_score"]

        if confidence > 0.7:
            answer = f"Based on your memories, {most_relevant['content']}"
        elif confidence > 0.3:
            answer = f"I found some related information in your memories: {most_relevant['content']}"
        else:
            answer = f"I found this potentially relevant memory: {most_relevant['content']}, though it may not fully answer your question."

        return answer, confidence, memory_contexts

    async def _generate_suggestions(
        self, question: str, memories: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate follow-up question suggestions"""
        if not memories:
            return [
                "What would you like me to remember for you?",
                "Can you tell me more about this topic?",
            ]

        # Extract common themes from memories
        all_tags = []
        for memory in memories[:3]:
            all_tags.extend(memory.get("tags", []))

        # Generate suggestions based on tags and content
        suggestions = []
        if "work" in all_tags:
            suggestions.append("Tell me more about your work experiences")
        if "family" in all_tags or "friend" in all_tags:
            suggestions.append(
                "What other memories do you have about family or friends?"
            )
        if "learning" in all_tags:
            suggestions.append("What other things have you learned recently?")

        # Default suggestions
        if not suggestions:
            suggestions = [
                "Can you tell me more details about this?",
                "What else would you like to know about your memories?",
            ]

        return suggestions[:3]

    async def health_check(self) -> QAHealthResponse:
        """Check the health of the Q&A system"""
        try:
            memory_stores = {}

            # Check memory engine stores
            if self.memory_engine.vector_store:
                memory_stores[
                    "vector_store"
                ] = await self.memory_engine.vector_store.health_check()
            else:
                memory_stores["vector_store"] = False

            if self.memory_engine.doc_store:
                memory_stores[
                    "document_store"
                ] = await self.memory_engine.doc_store.health_check()
            else:
                memory_stores["document_store"] = False

            if self.memory_engine.graph_store:
                memory_stores[
                    "graph_store"
                ] = await self.memory_engine.graph_store.health_check()
            else:
                memory_stores["graph_store"] = False

            # Check embedding service
            try:
                test_embedding = await self.embedder.aembed_query("test")
                embedding_service = len(test_embedding) > 0
            except:
                embedding_service = False

            # Overall status
            any_store_working = any(memory_stores.values())
            status = (
                "healthy" if any_store_working and embedding_service else "degraded"
            )

            return QAHealthResponse(
                status=status,
                memory_stores=memory_stores,
                embedding_service=embedding_service,
                llm_service=True,  # We're not using an external LLM yet
                total_memories=None,  # Would need to count across stores
                last_updated=datetime.now(),
            )

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return QAHealthResponse(
                status="unhealthy",
                memory_stores={},
                embedding_service=False,
                llm_service=False,
                total_memories=None,
                last_updated=datetime.now(),
            )

    async def _get_actual_memory_content(
        self, memory_id: str, user_id: str
    ) -> Optional[Dict[str, Any]]:
        """Retrieve actual memory content by ID using Pinecone metadata filtering"""
        try:
            # Use Pinecone metadata filtering to find exact memory by ID
            if self.memory_engine.vector_store:
                try:
                    # Create a dummy embedding for the query (we'll filter by metadata)
                    from app.common.enum.memory import MemoryType

                    handler = self.memory_engine.router.get_handler(
                        MemoryType.SEMANTIC_MEMORY
                    )
                    dummy_embedding, _ = await handler.extract_embedding("dummy query")

                    # Use Pinecone metadata filter to find exact memory ID
                    search_filter = {
                        "user_id": user_id,
                        "memory_id": memory_id,  # Filter by exact memory ID
                    }

                    vector_results = (
                        await self.memory_engine.vector_store.similarity_search(
                            embedding=dummy_embedding, limit=1, filter=search_filter
                        )
                    )

                    if vector_results:
                        memory_entry, score = vector_results[0]
                        if isinstance(memory_entry, dict):
                            metadata = memory_entry.get("metadata", {})
                            # Parse tags - handle both string and list formats
                            tags_data = metadata.get("tags", [])
                            if isinstance(tags_data, str):
                                tags = tags_data.split(",") if tags_data else []
                                tags = [tag.strip() for tag in tags if tag.strip()]
                            elif isinstance(tags_data, list):
                                tags = tags_data
                            else:
                                tags = []

                            return {
                                "memory_id": memory_entry["id"],
                                "text": metadata.get("input", metadata.get("text", "")),
                                "summary": metadata.get("summary", ""),
                                "memory_type": metadata.get("memory_type", "unknown"),
                                "relevance_score": 1.0,  # Exact match
                                "created_at": metadata.get("created_at"),
                                "tags": tags,
                            }
                        else:
                            return {
                                "memory_id": memory_entry.id,
                                "text": memory_entry.input,
                                "summary": memory_entry.summary,
                                "memory_type": memory_entry.memory_type.value
                                if hasattr(memory_entry.memory_type, "value")
                                else str(memory_entry.memory_type),
                                "relevance_score": 1.0,
                                "created_at": memory_entry.created_at,
                                "tags": memory_entry.tags,
                            }

                except Exception as e:
                    logger.warning(f"Pinecone metadata filter failed: {e}")

            # Fallback: Use broader search and filter results
            search_query = SearchQuery(
                user_id=user_id,
                query="memory",  # Generic query
                limit=100,  # Get many results to find the specific ID
                threshold=0.0,
                include_content=True,
            )

            search_response = await self.memory_engine.search_memories(search_query)

            if search_response and search_response.success and search_response.results:
                # Find exact match by ID
                for result in search_response.results:
                    if result.id == memory_id:
                        return {
                            "memory_id": result.id,
                            "text": result.content or result.summary,
                            "summary": result.summary,
                            "memory_type": result.memory_type.value
                            if hasattr(result.memory_type, "value")
                            else str(result.memory_type),
                            "relevance_score": result.score,
                            "created_at": result.created_at,
                            "tags": result.tags,
                        }

                # If no exact match, return first result as actual content (not placeholder)
                result = search_response.results[0]
                actual_content = (
                    result.content or result.summary or "No content available"
                )
                return {
                    "memory_id": result.id,
                    "text": actual_content,  # Use actual memory content
                    "summary": result.summary,
                    "memory_type": result.memory_type.value
                    if hasattr(result.memory_type, "value")
                    else str(result.memory_type),
                    "relevance_score": result.score,
                    "created_at": result.created_at,
                    "tags": result.tags,
                }

            return None

        except Exception as e:
            logger.warning(f"Failed to retrieve memory content for {memory_id}: {e}")
            return None
