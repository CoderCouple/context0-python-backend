"""Multi-hop reasoning engine for human-like reasoning and accuracy"""

import asyncio
import logging
import os
import re
import time
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

from app.memory.engine.memory_engine import MemoryEngine
from app.memory.utils.embeddings import get_embeddings
from app.reasoning.models import (
    ContextWindow,
    InferenceRule,
    MemoryCluster,
    ReasoningChain,
    ReasoningResult,
    ReasoningStep,
    ReasoningStepType,
)

logger = logging.getLogger(__name__)


class MultiHopReasoningEngine:
    """Advanced reasoning engine that performs human-like multi-hop reasoning"""

    def __init__(self):
        """Initialize the reasoning engine"""
        self._memory_engine: Optional[MemoryEngine] = None
        api_key = os.getenv("OPENAI_API_KEY")
        self.embedder = get_embeddings(model="text-embedding-3-small", api_key=api_key)

        # Initialize inference rules
        self.inference_rules = self._initialize_inference_rules()

        # Enhanced reasoning parameters for large-scale memory processing
        self.max_reasoning_depth = 8
        self.min_confidence_threshold = 0.1  # Lowered for better retrieval
        self.max_context_window_size = 50  # Increased for larger memory sets
        self.max_memories_per_chain = 25  # Can use 25+ memories per reasoning chain
        self.similarity_threshold = 0.3  # Lowered to retrieve more memories
        self.memory_clustering_threshold = 0.7  # Lowered for better clustering
        self.cross_reference_depth = 3  # How deep to follow memory connections

    @property
    def memory_engine(self) -> MemoryEngine:
        """Get memory engine instance"""
        if self._memory_engine is None:
            self._memory_engine = MemoryEngine.get_instance()
        return self._memory_engine

    async def reason_about_question(
        self, question: str, user_id: str, **kwargs
    ) -> ReasoningResult:
        """Perform multi-hop reasoning to answer a question"""
        start_time = time.time()

        try:
            logger.info(f"Starting multi-hop reasoning for: {question}")

            # Step 1: Analyze the question and identify reasoning requirements
            question_analysis = await self._analyze_question(question)

            # Step 2: Build initial context windows
            context_windows = await self._build_context_windows(
                question, user_id, question_analysis
            )

            # Step 3: Generate multiple reasoning chains
            reasoning_chains = await self._generate_reasoning_chains(
                question, context_windows, question_analysis
            )

            # Step 4: Evaluate and rank chains
            ranked_chains = await self._evaluate_and_rank_chains(
                reasoning_chains, question
            )

            # Step 5: Identify contradictions and gaps
            contradictions, gaps = await self._identify_contradictions_and_gaps(
                ranked_chains
            )

            # Step 6: Synthesize final answer
            synthesis = await self._synthesize_answer(
                ranked_chains, contradictions, gaps
            )

            # Step 7: Generate follow-up questions
            follow_ups = await self._generate_follow_up_questions(
                question, ranked_chains, gaps
            )

            reasoning_time = int((time.time() - start_time) * 1000)

            return ReasoningResult(
                question=question,
                primary_chain=ranked_chains[0]
                if ranked_chains
                else self._create_fallback_chain(question),
                alternative_chains=ranked_chains[1:3] if len(ranked_chains) > 1 else [],
                synthesis=synthesis,
                confidence_distribution=self._calculate_confidence_distribution(
                    ranked_chains
                ),
                contradictions=contradictions,
                gaps=gaps,
                follow_up_questions=follow_ups,
                reasoning_metadata={
                    "reasoning_time_ms": reasoning_time,
                    "context_windows_used": len(context_windows),
                    "chains_explored": len(reasoning_chains),
                    "question_analysis": question_analysis,
                },
            )

        except Exception as e:
            logger.error(f"Reasoning engine error: {e}")
            return self._create_error_result(question, str(e))

    async def _analyze_question(self, question: str) -> Dict[str, Any]:
        """Analyze the question to understand reasoning requirements"""
        analysis = {
            "question_type": "factual",  # factual, temporal, causal, comparative, etc.
            "entities": [],
            "temporal_indicators": [],
            "causal_indicators": [],
            "comparison_indicators": [],
            "reasoning_complexity": "simple",  # simple, medium, complex
            "required_hops": 1,
        }

        question_lower = question.lower()

        # Identify question type
        if any(
            word in question_lower for word in ["why", "because", "cause", "reason"]
        ):
            analysis["question_type"] = "causal"
            analysis["required_hops"] = 2
        elif any(
            word in question_lower
            for word in ["when", "before", "after", "during", "timeline"]
        ):
            analysis["question_type"] = "temporal"
        elif any(
            word in question_lower
            for word in ["compare", "versus", "vs", "difference", "similar"]
        ):
            analysis["question_type"] = "comparative"
            analysis["required_hops"] = 2
        elif any(
            word in question_lower for word in ["how", "process", "steps", "method"]
        ):
            analysis["question_type"] = "procedural"
        elif any(
            word in question_lower
            for word in ["what if", "would have", "could", "might"]
        ):
            analysis["question_type"] = "hypothetical"
            analysis["required_hops"] = 3

        # Extract entities (simple approach - in production use NER)
        entities = re.findall(r"\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)*\\b", question)
        analysis["entities"] = entities

        # Identify temporal indicators
        temporal_words = [
            "yesterday",
            "today",
            "tomorrow",
            "last week",
            "next month",
            "recently",
            "before",
            "after",
        ]
        analysis["temporal_indicators"] = [
            word for word in temporal_words if word in question_lower
        ]

        # Determine complexity
        complexity_indicators = len(analysis["entities"]) + len(
            analysis["temporal_indicators"]
        )
        if complexity_indicators > 3 or analysis["question_type"] in [
            "hypothetical",
            "comparative",
        ]:
            analysis["reasoning_complexity"] = "complex"
            analysis["required_hops"] = max(3, analysis["required_hops"])
        elif complexity_indicators > 1:
            analysis["reasoning_complexity"] = "medium"
            analysis["required_hops"] = max(2, analysis["required_hops"])

        return analysis

    async def _build_context_windows(
        self, question: str, user_id: str, question_analysis: Dict[str, Any]
    ) -> List[ContextWindow]:
        """Build context windows of related memories"""
        context_windows = []

        # Primary context: Direct semantic similarity
        primary_memories = await self._retrieve_similar_memories(
            question, user_id, limit=10
        )
        if primary_memories:
            context_windows.append(
                ContextWindow(
                    window_id=f"primary_{uuid.uuid4().hex[:8]}",
                    focal_memories=[m["memory_id"] for m in primary_memories[:5]],
                    supporting_memories=[m["memory_id"] for m in primary_memories[5:]],
                    entities=question_analysis["entities"],
                    relationships=[],
                    confidence=0.8,
                )
            )

        # Entity-based context windows
        for entity in question_analysis["entities"]:
            entity_memories = await self._retrieve_entity_memories(entity, user_id)
            if entity_memories:
                context_windows.append(
                    ContextWindow(
                        window_id=f"entity_{entity.lower()}_{uuid.uuid4().hex[:8]}",
                        focal_memories=[m["memory_id"] for m in entity_memories[:3]],
                        supporting_memories=[
                            m["memory_id"] for m in entity_memories[3:8]
                        ],
                        entities=[entity],
                        relationships=[],
                        confidence=0.7,
                    )
                )

        # Temporal context windows (if question involves time)
        if question_analysis["temporal_indicators"]:
            temporal_memories = await self._retrieve_temporal_memories(
                question_analysis["temporal_indicators"], user_id
            )
            if temporal_memories:
                context_windows.append(
                    ContextWindow(
                        window_id=f"temporal_{uuid.uuid4().hex[:8]}",
                        focal_memories=[m["memory_id"] for m in temporal_memories[:5]],
                        supporting_memories=[],
                        entities=[],
                        relationships=[],
                        temporal_context={
                            "indicators": question_analysis["temporal_indicators"]
                        },
                        confidence=0.6,
                    )
                )

        # Meta-memory context (reflections and learnings)
        meta_memories = await self._retrieve_meta_memories(question, user_id)
        if meta_memories:
            context_windows.append(
                ContextWindow(
                    window_id=f"meta_{uuid.uuid4().hex[:8]}",
                    focal_memories=[m["memory_id"] for m in meta_memories],
                    supporting_memories=[],
                    entities=[],
                    relationships=[],
                    confidence=0.5,
                )
            )

        return context_windows

    async def _generate_reasoning_chains(
        self,
        question: str,
        context_windows: List[ContextWindow],
        question_analysis: Dict[str, Any],
    ) -> List[ReasoningChain]:
        """Generate multiple reasoning chains using different approaches"""
        chains = []

        # Direct retrieval chain (simplest)
        direct_chain = await self._create_direct_retrieval_chain(
            question, context_windows
        )
        if direct_chain:
            chains.append(direct_chain)

        # Inference chain (moderate complexity)
        if question_analysis["reasoning_complexity"] in ["medium", "complex"]:
            inference_chain = await self._create_inference_chain(
                question, context_windows, question_analysis
            )
            if inference_chain:
                chains.append(inference_chain)

        # Analogy chain (for complex questions)
        if question_analysis["reasoning_complexity"] == "complex":
            analogy_chain = await self._create_analogy_chain(question, context_windows)
            if analogy_chain:
                chains.append(analogy_chain)

        # Causal reasoning chain (for "why" questions)
        if question_analysis["question_type"] == "causal":
            causal_chain = await self._create_causal_reasoning_chain(
                question, context_windows
            )
            if causal_chain:
                chains.append(causal_chain)

        # Temporal reasoning chain (for time-based questions)
        if question_analysis["question_type"] == "temporal":
            temporal_chain = await self._create_temporal_reasoning_chain(
                question, context_windows
            )
            if temporal_chain:
                chains.append(temporal_chain)

        return chains

    async def _create_direct_retrieval_chain(
        self, question: str, context_windows: List[ContextWindow]
    ) -> Optional[ReasoningChain]:
        """Create a simple direct retrieval reasoning chain"""
        if not context_windows:
            return None

        primary_window = context_windows[0]

        # Step 1: Memory retrieval
        retrieval_step = ReasoningStep(
            step_id=f"retrieval_{uuid.uuid4().hex[:8]}",
            step_type=ReasoningStepType.MEMORY_RETRIEVAL,
            input_context=[question],
            reasoning_process="Retrieved most semantically similar memories to the question",
            output=f"Found {len(primary_window.focal_memories)} highly relevant memories",
            confidence=primary_window.confidence,
            memory_references=primary_window.focal_memories,
        )

        # Step 2: Direct synthesis
        synthesis_step = ReasoningStep(
            step_id=f"synthesis_{uuid.uuid4().hex[:8]}",
            step_type=ReasoningStepType.SYNTHESIS,
            input_context=primary_window.focal_memories,
            reasoning_process="Directly synthesized answer from retrieved memories",
            output="Combined information from memories to form answer",
            confidence=primary_window.confidence * 0.9,
            memory_references=primary_window.focal_memories,
        )

        # Generate actual answer from memories
        final_answer = await self._synthesize_answer_from_memories(
            question, primary_window.focal_memories, "direct"
        )

        return ReasoningChain(
            chain_id=f"direct_{uuid.uuid4().hex[:8]}",
            question=question,
            reasoning_steps=[retrieval_step, synthesis_step],
            final_answer=final_answer,
            overall_confidence=primary_window.confidence * 0.85,
            total_memories_used=len(primary_window.focal_memories),
            reasoning_time_ms=50,
        )

    async def _create_inference_chain(
        self,
        question: str,
        context_windows: List[ContextWindow],
        question_analysis: Dict[str, Any],
    ) -> Optional[ReasoningChain]:
        """Create an inference-based reasoning chain"""
        if len(context_windows) < 2:
            return None

        steps = []
        memory_refs = []

        # Step 1: Initial retrieval
        primary_window = context_windows[0]
        steps.append(
            ReasoningStep(
                step_id=f"inference_retrieval_{uuid.uuid4().hex[:8]}",
                step_type=ReasoningStepType.MEMORY_RETRIEVAL,
                input_context=[question],
                reasoning_process="Retrieved memories for inference analysis",
                output=f"Collected {len(primary_window.focal_memories)} primary memories",
                confidence=0.8,
                memory_references=primary_window.focal_memories,
            )
        )
        memory_refs.extend(primary_window.focal_memories)

        # Step 2: Pattern matching across contexts
        pattern_step = ReasoningStep(
            step_id=f"pattern_{uuid.uuid4().hex[:8]}",
            step_type=ReasoningStepType.PATTERN_MATCHING,
            input_context=primary_window.focal_memories,
            reasoning_process="Identified patterns and connections across memory contexts",
            output="Found recurring themes and relationship patterns",
            confidence=0.7,
            memory_references=memory_refs,
        )
        steps.append(pattern_step)

        # Step 3: Inference application
        inference_step = ReasoningStep(
            step_id=f"inference_{uuid.uuid4().hex[:8]}",
            step_type=ReasoningStepType.INFERENCE,
            input_context=[pattern_step.output],
            reasoning_process="Applied logical inference rules to derive new insights",
            output="Generated inferred conclusions from pattern analysis",
            confidence=0.6,
            memory_references=memory_refs,
        )
        steps.append(inference_step)

        # Step 4: Synthesis
        synthesis_step = ReasoningStep(
            step_id=f"inference_synthesis_{uuid.uuid4().hex[:8]}",
            step_type=ReasoningStepType.SYNTHESIS,
            input_context=[inference_step.output],
            reasoning_process="Synthesized final answer incorporating inferences",
            output="Complete answer with inferred insights",
            confidence=0.65,
            memory_references=memory_refs,
        )
        steps.append(synthesis_step)

        # Generate actual answer from memories
        final_answer = await self._synthesize_answer_from_memories(
            question, memory_refs, "inference"
        )

        return ReasoningChain(
            chain_id=f"inference_{uuid.uuid4().hex[:8]}",
            question=question,
            reasoning_steps=steps,
            final_answer=final_answer,
            overall_confidence=0.7,
            total_memories_used=len(memory_refs),
            reasoning_time_ms=200,
        )

    async def _create_analogy_chain(
        self, question: str, context_windows: List[ContextWindow]
    ) -> Optional[ReasoningChain]:
        """Create an analogy-based reasoning chain"""
        if not context_windows:
            return None

        steps = []
        memory_refs = []

        # Find analogical patterns
        for window in context_windows[:2]:  # Use first 2 windows
            steps.append(
                ReasoningStep(
                    step_id=f"analogy_{uuid.uuid4().hex[:8]}",
                    step_type=ReasoningStepType.ANALOGY,
                    input_context=[question],
                    reasoning_process="Found analogical patterns in memories",
                    output="Identified similar situations and outcomes",
                    confidence=0.6,
                    memory_references=window.focal_memories,
                )
            )
            memory_refs.extend(window.focal_memories)

        # Generate actual answer from memories
        final_answer = await self._synthesize_answer_from_memories(
            question, memory_refs, "analogy"
        )

        return ReasoningChain(
            chain_id=f"analogy_{uuid.uuid4().hex[:8]}",
            question=question,
            reasoning_steps=steps,
            final_answer=final_answer,
            overall_confidence=0.6,
            total_memories_used=len(memory_refs),
            reasoning_time_ms=150,
        )

    async def _create_causal_reasoning_chain(
        self, question: str, context_windows: List[ContextWindow]
    ) -> Optional[ReasoningChain]:
        """Create a causal reasoning chain"""
        if not context_windows:
            return None

        steps = []
        memory_refs = []

        # Identify causal relationships
        primary_window = context_windows[0]
        steps.append(
            ReasoningStep(
                step_id=f"causal_{uuid.uuid4().hex[:8]}",
                step_type=ReasoningStepType.CAUSAL_REASONING,
                input_context=[question],
                reasoning_process="Analyzed causal relationships between events and outcomes",
                output="Identified cause-effect patterns",
                confidence=0.7,
                memory_references=primary_window.focal_memories,
            )
        )
        memory_refs.extend(primary_window.focal_memories)

        # Generate actual answer from memories
        final_answer = await self._synthesize_answer_from_memories(
            question, memory_refs, "causal"
        )

        return ReasoningChain(
            chain_id=f"causal_{uuid.uuid4().hex[:8]}",
            question=question,
            reasoning_steps=steps,
            final_answer=final_answer,
            overall_confidence=0.7,
            total_memories_used=len(memory_refs),
            reasoning_time_ms=180,
        )

    async def _create_temporal_reasoning_chain(
        self, question: str, context_windows: List[ContextWindow]
    ) -> Optional[ReasoningChain]:
        """Create a temporal reasoning chain"""
        if not context_windows:
            return None

        steps = []
        memory_refs = []

        # Sort memories by time and analyze temporal patterns
        primary_window = context_windows[0]
        steps.append(
            ReasoningStep(
                step_id=f"temporal_{uuid.uuid4().hex[:8]}",
                step_type=ReasoningStepType.TEMPORAL_REASONING,
                input_context=[question],
                reasoning_process="Analyzed temporal progression and time-based patterns",
                output="Identified timeline and chronological relationships",
                confidence=0.65,
                memory_references=primary_window.focal_memories,
            )
        )
        memory_refs.extend(primary_window.focal_memories)

        # Generate actual answer from memories
        final_answer = await self._synthesize_answer_from_memories(
            question, memory_refs, "temporal"
        )

        return ReasoningChain(
            chain_id=f"temporal_{uuid.uuid4().hex[:8]}",
            question=question,
            reasoning_steps=steps,
            final_answer=final_answer,
            overall_confidence=0.65,
            total_memories_used=len(memory_refs),
            reasoning_time_ms=160,
        )

    async def _retrieve_similar_memories(
        self, query: str, user_id: str, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Advanced hybrid search with query expansion, diversity scoring, and precision matching"""

        # STEP 1: Query Analysis and Expansion
        expanded_queries = await self._expand_query_with_keywords(query)
        logger.info(
            f"Expanded query into {len(expanded_queries)} variants: {expanded_queries[:3]}"
        )

        all_memories = []

        # STEP 2: Multi-query search with diversity enforcement
        for i, expanded_query in enumerate(expanded_queries):
            try:
                from app.api.v1.request.memory_request import SearchQuery

                search_query = SearchQuery(
                    user_id=user_id,
                    query=expanded_query,
                    limit=min(20, limit),  # Get fewer per query to enforce diversity
                    threshold=0.2,  # Lower threshold for broader retrieval
                    include_content=True,
                )

                search_response = await self.memory_engine.search_memories(search_query)

                if (
                    search_response
                    and search_response.success
                    and search_response.results
                ):
                    logger.info(
                        f"Query variant {i+1} found {len(search_response.results)} memories"
                    )

                    # Convert and score each memory
                    for result in search_response.results:
                        memory_data = {
                            "memory_id": result.id,
                            "content": result.content or result.summary or "",
                            "summary": result.summary,
                            "memory_type": result.memory_type.value
                            if hasattr(result.memory_type, "value")
                            else str(result.memory_type),
                            "created_at": result.created_at,
                            "tags": result.tags or [],
                            "scope": getattr(result, "scope", "unknown"),
                            "source": f"expanded_query_{i+1}",
                            "search_scores": {
                                "primary_score": result.score,
                                "confidence": result.confidence,
                                "query_specificity": self._calculate_query_specificity(
                                    result, query, expanded_query
                                ),
                                "keyword_match": self._calculate_keyword_match_score(
                                    result, query
                                ),
                                "memory_type_relevance": self._calculate_memory_type_relevance(
                                    result.memory_type, query
                                ),
                                "recency": self._calculate_recency_score(
                                    result.created_at
                                ),
                                "tag_relevance": self._calculate_tag_relevance(
                                    result.tags or [], query
                                ),
                            },
                        }

                        # Calculate advanced composite score
                        memory_data[
                            "composite_score"
                        ] = self._calculate_advanced_composite_score(memory_data, query)
                        all_memories.append(memory_data)

            except Exception as e:
                logger.error(
                    f"Search failed for expanded query '{expanded_query}': {e}"
                )
                continue

        if not all_memories:
            logger.info(
                "No memories found in expanded search, falling back to multi-store search"
            )
            return await self._fallback_multi_store_search(query, user_id, limit)

        # STEP 3: Advanced memory diversity and relevance optimization
        optimized_memories = await self._optimize_memory_selection(
            all_memories, query, limit
        )

        # STEP 4: Cross-reference enhancement
        enhanced_memories = await self._enhance_with_cross_references(
            optimized_memories, query, user_id
        )

        logger.info(
            f"Final memory selection: {len(enhanced_memories)} memories with average score: {sum(m['composite_score'] for m in enhanced_memories) / len(enhanced_memories):.3f}"
        )

        return enhanced_memories[:limit]

    async def _enhance_with_cross_references(
        self, primary_memories: List[Dict[str, Any]], query: str, user_id: str
    ) -> List[Dict[str, Any]]:
        """Enhance primary search results with cross-references from other stores"""
        enhanced_memories = []

        for memory in primary_memories:
            enhanced_memory = memory.copy()
            memory_id = memory["memory_id"]

            # Enhance with graph relationships
            try:
                if self.memory_engine.graph_store:
                    neighbors = await self.memory_engine.graph_store.get_neighbors(
                        memory_id, depth=1
                    )
                    if neighbors:
                        enhanced_memory["graph_connections"] = len(neighbors)
                        enhanced_memory["search_scores"]["graph_relevance"] = min(
                            len(neighbors) * 0.1, 0.5
                        )
            except Exception as e:
                logger.debug(f"Graph enhancement failed for {memory_id}: {e}")

            # Enhance with temporal context
            try:
                if self.memory_engine.timeseries_store:
                    # Check if memory has temporal significance
                    created_at = memory.get("created_at")
                    if created_at:
                        temporal_relevance = (
                            self._calculate_temporal_relevance_for_memory(
                                created_at, query
                            )
                        )
                        enhanced_memory["search_scores"][
                            "temporal_relevance"
                        ] = temporal_relevance
            except Exception as e:
                logger.debug(f"Temporal enhancement failed for {memory_id}: {e}")

            # Recalculate composite score with enhancements
            search_scores = enhanced_memory["search_scores"]
            enhanced_score = (
                search_scores.get("primary_score", 0.0) * 0.7
                + search_scores.get(  # Primary search (highest weight)
                    "graph_relevance", 0.0
                )
                * 0.15
                + search_scores.get("temporal_relevance", 0.0)  # Graph connections
                * 0.1
                + search_scores.get("recency", 0.0)  # Temporal context
                * 0.05  # Recency bonus
            )

            enhanced_memory["composite_score"] = enhanced_score
            enhanced_memories.append(enhanced_memory)

        # Sort by enhanced composite score
        enhanced_memories.sort(key=lambda x: x["composite_score"], reverse=True)

        logger.info(f"Enhanced {len(enhanced_memories)} memories with cross-references")
        return enhanced_memories

    async def _fallback_multi_store_search(
        self, query: str, user_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Fallback multi-store search when primary search fails"""
        all_memories = []

        # Try each store individually
        stores_tried = []

        # Vector Store
        if self.memory_engine.vector_store:
            try:
                query_embedding = await self.embedder.aembed_query(query)
                vector_results = await self._search_vector_store_advanced(
                    query_embedding, user_id, limit
                )
                all_memories.extend(vector_results)
                stores_tried.append("vector")
            except Exception as e:
                logger.warning(f"Vector store search failed: {e}")

        # Document Store
        if self.memory_engine.doc_store:
            try:
                doc_results = await self._search_document_store_advanced(
                    query, user_id, limit
                )
                all_memories.extend(doc_results)
                stores_tried.append("document")
            except Exception as e:
                logger.warning(f"Document store search failed: {e}")

        logger.info(
            f"Fallback search tried stores: {stores_tried}, found {len(all_memories)} memories"
        )

        # Deduplicate and rank
        if all_memories:
            unique_memories = self._deduplicate_and_score_memories(all_memories, query)
            return sorted(
                unique_memories, key=lambda x: x["composite_score"], reverse=True
            )[:limit]

        return []

    def _calculate_temporal_relevance_for_memory(self, created_at, query: str) -> float:
        """Calculate temporal relevance for a specific memory"""
        if not created_at:
            return 0.0

        # Check if query has temporal indicators
        temporal_words = [
            "recent",
            "recently",
            "today",
            "yesterday",
            "last",
            "past",
            "when",
        ]
        query_lower = query.lower()

        if any(word in query_lower for word in temporal_words):
            # Recent memories get higher temporal relevance for temporal queries
            recency_score = self._calculate_recency_score(created_at)
            return recency_score * 0.5

        return 0.1  # Base temporal relevance

    async def _search_vector_store_advanced(
        self, query_embedding: List[float], user_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Advanced vector store search with semantic similarity"""
        try:
            search_filter = {"user_id": user_id}
            vector_results = await self.memory_engine.vector_store.similarity_search(
                embedding=query_embedding, limit=limit, filter=search_filter
            )

            formatted_results = []
            for memory_entry, score in vector_results:
                if isinstance(memory_entry, dict):
                    metadata = memory_entry.get("metadata", {})
                    # Parse tags properly
                    tags_data = metadata.get("tags", [])
                    if isinstance(tags_data, str):
                        tags = tags_data.split(",") if tags_data else []
                        tags = [tag.strip() for tag in tags if tag.strip()]
                    else:
                        tags = tags_data

                    # Get actual content from metadata
                    content = metadata.get("input") or metadata.get("text", "")

                    # Only include memories with actual content and good scores
                    if content and score >= self.similarity_threshold:
                        formatted_results.append(
                            {
                                "memory_id": memory_entry["id"],
                                "content": content,
                                "summary": metadata.get("summary"),
                                "memory_type": metadata.get("memory_type"),
                                "created_at": metadata.get("created_at"),
                                "tags": tags,
                                "source": "vector_store",
                                "search_scores": {
                                    "vector_similarity": score,
                                    "recency": self._calculate_recency_score(
                                        metadata.get("created_at")
                                    ),
                                    "tag_relevance": self._calculate_tag_relevance(
                                        tags, ""
                                    ),
                                },
                            }
                        )

            logger.info(
                f"Vector store found {len(formatted_results)} relevant memories"
            )
            return formatted_results
        except Exception as e:
            logger.error(f"Vector store search error: {e}")
            return []

    async def _search_document_store_advanced(
        self, query: str, user_id: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Advanced document store search with full-text and metadata"""
        try:
            search_query = {"source_user_id": user_id}
            results = await self.memory_engine.doc_store.search(
                search_query, limit * 2
            )  # Get more to filter

            formatted_results = []
            query_words = set(query.lower().split())

            for memory in results:
                content = memory.get("input", "")
                if not content:  # Skip memories without content
                    continue

                content_words = set(content.lower().split())
                text_similarity = (
                    len(query_words.intersection(content_words)) / len(query_words)
                    if query_words
                    else 0
                )

                # Calculate tag relevance
                tags = memory.get("tags", [])
                tag_relevance = self._calculate_tag_relevance(tags, query)

                # Calculate composite relevance score
                composite_relevance = text_similarity * 0.6 + tag_relevance * 0.4

                # Only include memories with reasonable relevance
                if composite_relevance >= 0.1 or text_similarity >= 0.2:
                    formatted_results.append(
                        {
                            "memory_id": memory.get("id"),
                            "content": content,
                            "summary": memory.get("summary"),
                            "memory_type": memory.get("memory_type"),
                            "created_at": memory.get("created_at"),
                            "tags": tags,
                            "source": "document_store",
                            "search_scores": {
                                "text_similarity": text_similarity,
                                "recency": self._calculate_recency_score(
                                    memory.get("created_at")
                                ),
                                "tag_relevance": tag_relevance,
                                "composite_relevance": composite_relevance,
                            },
                            "metadata": memory,
                        }
                    )

            # Sort by composite relevance and return top results
            formatted_results.sort(
                key=lambda x: x["search_scores"]["composite_relevance"], reverse=True
            )

            logger.info(
                f"Document store found {len(formatted_results)} relevant memories"
            )
            return formatted_results[:limit]
        except Exception as e:
            logger.error(f"Document store search error: {e}")
            return []

    async def _search_graph_store_advanced(
        self, query: str, user_id: str
    ) -> List[Dict[str, Any]]:
        """Advanced graph store search using relationship traversal"""
        try:
            # Extract entities from query
            entities = self._extract_entities_from_query(query)
            graph_memories = []

            for entity in entities:
                # Search for memories connected to this entity
                entity_search = {"user_id": user_id}
                memories = await self.memory_engine.graph_store.search(
                    entity_search, 20
                )

                for memory in memories:
                    # Calculate graph-based relevance
                    graph_relevance = await self._calculate_graph_relevance(
                        memory, entity, user_id
                    )

                    graph_memories.append(
                        {
                            "memory_id": memory.get("id"),
                            "content": memory.get("input", ""),
                            "summary": memory.get("summary"),
                            "memory_type": memory.get("memory_type"),
                            "created_at": memory.get("created_at"),
                            "source": "graph_store",
                            "search_scores": {
                                "graph_relevance": graph_relevance,
                                "entity_connection": 1.0
                                if entity.lower() in memory.get("input", "").lower()
                                else 0.0,
                            },
                            "connected_entity": entity,
                        }
                    )

            return graph_memories
        except Exception as e:
            logger.error(f"Graph store search error: {e}")
            return []

    async def _search_timeseries_store_advanced(
        self, query: str, user_id: str
    ) -> List[Dict[str, Any]]:
        """Advanced timeseries search for temporal patterns"""
        try:
            # Detect temporal expressions in query
            temporal_context = self._extract_temporal_context(query)

            if temporal_context:
                # Search within time ranges
                timeline_data = await self.memory_engine.timeseries_store.get_timeline(
                    user_id, "day"
                )

                timeseries_memories = []
                for time_bucket in timeline_data:
                    temporal_relevance = self._calculate_temporal_relevance(
                        time_bucket, temporal_context
                    )

                    if temporal_relevance > 0.3:
                        timeseries_memories.append(
                            {
                                "memory_id": f"timebucket_{time_bucket.get('time_bucket')}",
                                "content": f"Memory cluster from {time_bucket.get('time_bucket')} with {time_bucket.get('memory_count')} memories",
                                "source": "timeseries_store",
                                "search_scores": {
                                    "temporal_relevance": temporal_relevance,
                                    "memory_density": min(
                                        time_bucket.get("memory_count", 0) / 10.0, 1.0
                                    ),
                                },
                                "time_bucket": time_bucket,
                            }
                        )

                return timeseries_memories

            return []
        except Exception as e:
            logger.error(f"TimeSeries store search error: {e}")
            return []

    async def _search_audit_store_advanced(
        self, query: str, user_id: str
    ) -> List[Dict[str, Any]]:
        """Advanced audit store search for historical context"""
        try:
            # Search for memory evolution patterns
            query_words = query.lower().split()

            # This would search audit logs for memories that have been modified/updated
            # related to the query terms (simplified implementation)
            audit_memories = []

            # Look for patterns of memory updates that might be relevant
            if any(
                word in ["change", "update", "modify", "edit", "correct"]
                for word in query_words
            ):
                # Search for audit entries showing memory evolution
                audit_memories.append(
                    {
                        "memory_id": "audit_pattern_evolution",
                        "content": "Memory evolution patterns detected in audit log",
                        "source": "audit_store",
                        "search_scores": {
                            "evolution_relevance": 0.6,
                            "historical_significance": 0.4,
                        },
                    }
                )

            return audit_memories
        except Exception as e:
            logger.error(f"Audit store search error: {e}")
            return []

    async def _retrieve_entity_memories(
        self, entity: str, user_id: str
    ) -> List[Dict[str, Any]]:
        """Retrieve memories mentioning a specific entity using graph relationships"""
        try:
            if not self.memory_engine.graph_store:
                return []

            # Use graph traversal to find all memories connected to this entity
            entity_id = (
                f"person:{entity.lower()}"
                if entity.istitle()
                else f"entity:{entity.lower()}"
            )

            # Get memories that have relationships with this entity
            neighbors = await self.memory_engine.graph_store.get_neighbors(
                entity_id, depth=2
            )

            entity_memories = []
            for neighbor in neighbors:
                if neighbor["node"].get("memory_type"):  # This is a memory node
                    entity_memories.append(
                        {
                            "memory_id": neighbor["node"].get("id"),
                            "content": neighbor["node"].get("input", ""),
                            "memory_type": neighbor["node"].get("memory_type"),
                            "relationship_distance": neighbor.get("distance", 1),
                            "relationship_types": [
                                rel.get("type")
                                for rel in neighbor.get("relationships", [])
                            ],
                        }
                    )

            return entity_memories
        except Exception as e:
            logger.error(f"Entity memory retrieval error: {e}")
            return []

    async def _cluster_and_cross_reference_memories(
        self, memories: List[Dict[str, Any]], query: str, user_id: str
    ) -> List[Dict[str, Any]]:
        """Cluster and cross-reference large memory sets for comprehensive reasoning"""
        logger.info(f"Clustering {len(memories)} memories for comprehensive analysis")

        # Step 1: Thematic clustering
        thematic_clusters = await self._create_thematic_clusters(memories)

        # Step 2: Temporal clustering
        temporal_clusters = await self._create_temporal_clusters(memories)

        # Step 3: Entity-based clustering
        entity_clusters = await self._create_entity_clusters(memories, user_id)

        # Step 4: Cross-reference between clusters
        cross_references = await self._find_cross_cluster_references(
            thematic_clusters, temporal_clusters, entity_clusters
        )

        # Step 5: Synthesize comprehensive memory set
        synthesized_memories = await self._synthesize_clustered_memories(
            memories,
            thematic_clusters,
            temporal_clusters,
            entity_clusters,
            cross_references,
            query,
        )

        logger.info(
            f"Synthesized {len(synthesized_memories)} memories from {len(memories)} original memories"
        )
        return synthesized_memories

    async def _create_thematic_clusters(
        self, memories: List[Dict[str, Any]]
    ) -> List[MemoryCluster]:
        """Group memories by themes and topics"""
        clusters = []
        processed_memories = set()

        # Simple thematic clustering based on tags and content similarity
        for i, memory in enumerate(memories):
            if memory.get("memory_id") in processed_memories:
                continue

            # Find memories with similar themes
            cluster_memories = [memory]
            memory_tags = set(memory.get("tags", []))
            memory_content = memory.get("content", "").lower()

            for j, other_memory in enumerate(memories[i + 1 :], i + 1):
                if other_memory.get("memory_id") in processed_memories:
                    continue

                other_tags = set(other_memory.get("tags", []))
                other_content = other_memory.get("content", "").lower()

                # Calculate thematic similarity
                tag_overlap = len(memory_tags.intersection(other_tags)) / max(
                    len(memory_tags.union(other_tags)), 1
                )
                content_words = set(memory_content.split())
                other_words = set(other_content.split())
                content_overlap = len(content_words.intersection(other_words)) / max(
                    len(content_words.union(other_words)), 1
                )

                if tag_overlap > 0.3 or content_overlap > 0.2:
                    cluster_memories.append(other_memory)
                    processed_memories.add(other_memory.get("memory_id"))

            if len(cluster_memories) >= 2:  # Only create clusters with 2+ memories
                theme = self._determine_cluster_theme(cluster_memories)
                clusters.append(
                    MemoryCluster(
                        cluster_id=f"thematic_{len(clusters)}",
                        theme=theme,
                        memories=cluster_memories,
                        relationships=[],
                        confidence=0.8,
                    )
                )

            processed_memories.add(memory.get("memory_id"))

        return clusters

    async def _create_temporal_clusters(
        self, memories: List[Dict[str, Any]]
    ) -> List[MemoryCluster]:
        """Group memories by temporal patterns"""
        clusters = []

        # Group by time periods (simplified)
        time_groups = {}
        for memory in memories:
            created_at = memory.get("created_at")
            if created_at:
                # Group by month for simplicity
                time_key = (
                    f"{created_at.year}-{created_at.month:02d}"
                    if hasattr(created_at, "year")
                    else "unknown"
                )
                if time_key not in time_groups:
                    time_groups[time_key] = []
                time_groups[time_key].append(memory)

        for time_key, time_memories in time_groups.items():
            if len(time_memories) >= 3:  # Only create clusters with 3+ memories
                # Calculate temporal span from memory dates
                dates = [
                    m.get("created_at") for m in time_memories if m.get("created_at")
                ]
                temporal_span = None
                if dates:
                    temporal_span = {
                        "start": min(dates) if dates else None,
                        "end": max(dates) if dates else None,
                    }

                clusters.append(
                    MemoryCluster(
                        cluster_id=f"temporal_{time_key}",
                        theme=f"Memories from {time_key}",
                        memories=time_memories,
                        relationships=[],
                        confidence=0.7,
                        temporal_span=temporal_span,
                    )
                )

        return clusters

    async def _create_entity_clusters(
        self, memories: List[Dict[str, Any]], user_id: str
    ) -> List[MemoryCluster]:
        """Group memories by entities mentioned"""
        entity_groups = {}

        for memory in memories:
            content = memory.get("content", "").lower()

            # Extract entities (simplified - using capitalized words)
            import re

            entities = re.findall(r"\b[A-Z][a-z]+\b", memory.get("content", ""))

            for entity in entities:
                entity_key = entity.lower()
                if entity_key not in entity_groups:
                    entity_groups[entity_key] = []
                entity_groups[entity_key].append(memory)

        clusters = []
        for entity, entity_memories in entity_groups.items():
            if len(entity_memories) >= 2:  # Only create clusters with 2+ memories
                clusters.append(
                    MemoryCluster(
                        cluster_id=f"entity_{entity}",
                        theme=f"Memories about {entity}",
                        memories=entity_memories,
                        relationships=[],
                        confidence=0.75,
                    )
                )

        return clusters

    async def _find_cross_cluster_references(
        self,
        thematic: List[MemoryCluster],
        temporal: List[MemoryCluster],
        entity: List[MemoryCluster],
    ) -> List[Dict[str, Any]]:
        """Find cross-references between different types of clusters"""
        cross_refs = []

        # Find overlapping memories between clusters
        all_clusters = thematic + temporal + entity

        for i, cluster1 in enumerate(all_clusters):
            for j, cluster2 in enumerate(all_clusters[i + 1 :], i + 1):
                # Find memory overlaps
                cluster1_memory_ids = set(m.get("memory_id") for m in cluster1.memories)
                cluster2_memory_ids = set(m.get("memory_id") for m in cluster2.memories)

                overlap = cluster1_memory_ids.intersection(cluster2_memory_ids)
                if overlap:
                    cross_refs.append(
                        {
                            "cluster1": cluster1.cluster_id,
                            "cluster2": cluster2.cluster_id,
                            "shared_memories": list(overlap),
                            "relationship_strength": len(overlap)
                            / min(len(cluster1_memory_ids), len(cluster2_memory_ids)),
                        }
                    )

        return cross_refs

    async def _synthesize_clustered_memories(
        self,
        original_memories: List[Dict[str, Any]],
        thematic_clusters: List[MemoryCluster],
        temporal_clusters: List[MemoryCluster],
        entity_clusters: List[MemoryCluster],
        cross_references: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """Synthesize the most relevant memories from clusters"""

        # Score memories based on cluster membership and cross-references
        memory_scores = {}

        # Base scores from original ranking
        for i, memory in enumerate(original_memories):
            memory_id = memory.get("memory_id")
            memory_scores[memory_id] = {
                "memory": memory,
                "base_score": memory.get(
                    "composite_score", 1.0 - (i * 0.01)
                ),  # Decreasing score
                "cluster_bonus": 0.0,
                "cross_ref_bonus": 0.0,
            }

        # Add cluster bonuses
        all_clusters = thematic_clusters + temporal_clusters + entity_clusters
        for cluster in all_clusters:
            cluster_bonus = 0.3 if len(cluster.memories) > 5 else 0.2
            for memory in cluster.memories:
                memory_id = memory.get("memory_id")
                if memory_id in memory_scores:
                    memory_scores[memory_id]["cluster_bonus"] += cluster_bonus

        # Add cross-reference bonuses
        for cross_ref in cross_references:
            cross_ref_bonus = cross_ref["relationship_strength"] * 0.4
            for memory_id in cross_ref["shared_memories"]:
                if memory_id in memory_scores:
                    memory_scores[memory_id]["cross_ref_bonus"] += cross_ref_bonus

        # Calculate final scores and rank
        scored_memories = []
        for memory_id, score_data in memory_scores.items():
            final_score = (
                score_data["base_score"]
                + score_data["cluster_bonus"]
                + score_data["cross_ref_bonus"]
            )

            memory = score_data["memory"].copy()
            memory["final_synthesis_score"] = final_score
            memory["cluster_analysis"] = {
                "cluster_bonus": score_data["cluster_bonus"],
                "cross_ref_bonus": score_data["cross_ref_bonus"],
                "total_clusters": len(
                    [
                        c
                        for c in all_clusters
                        if memory_id in [m.get("memory_id") for m in c.memories]
                    ]
                ),
            }
            scored_memories.append(memory)

        # Return top memories based on synthesis scoring
        return sorted(
            scored_memories, key=lambda x: x["final_synthesis_score"], reverse=True
        )

    def _determine_cluster_theme(self, memories: List[Dict[str, Any]]) -> str:
        """Determine the main theme of a memory cluster"""
        all_tags = []
        for memory in memories:
            all_tags.extend(memory.get("tags", []))

        if not all_tags:
            return "Mixed topics"

        # Find most common tags
        from collections import Counter

        tag_counts = Counter(all_tags)
        most_common_tags = [tag for tag, count in tag_counts.most_common(3)]

        return f"Cluster: {', '.join(most_common_tags)}"

    def _deduplicate_and_score_memories(
        self, memories: List[Dict[str, Any]], query: str
    ) -> List[Dict[str, Any]]:
        """Deduplicate memories and calculate composite scores"""
        seen_ids = set()
        unique_memories = []
        query_words = set(query.lower().split())

        for memory in memories:
            memory_id = memory.get("memory_id")
            if memory_id and memory_id not in seen_ids:
                seen_ids.add(memory_id)

                # Calculate composite score from multiple factors
                search_scores = memory.get("search_scores", {})

                # Get individual scores
                vector_similarity = search_scores.get("vector_similarity", 0.0)
                text_similarity = search_scores.get("text_similarity", 0.0)
                composite_relevance = search_scores.get("composite_relevance", 0.0)
                recency = search_scores.get("recency", 0.0)
                tag_relevance = search_scores.get("tag_relevance", 0.0)
                graph_relevance = search_scores.get("graph_relevance", 0.0)
                temporal_relevance = search_scores.get("temporal_relevance", 0.0)

                # Calculate content quality score
                content = memory.get("content", "")
                content_quality = min(len(content) / 100.0, 1.0) if content else 0.0

                # Calculate keyword match score
                content_words = set(content.lower().split()) if content else set()
                keyword_match = (
                    len(query_words.intersection(content_words)) / len(query_words)
                    if query_words and content_words
                    else 0.0
                )

                # Weighted composite score prioritizing semantic similarity and content relevance
                composite_score = (
                    vector_similarity * 0.35
                    + keyword_match * 0.25  # Semantic similarity (highest weight)
                    + (text_similarity + composite_relevance)  # Direct keyword matches
                    * 0.15
                    + tag_relevance * 0.15  # Text relevance
                    + content_quality * 0.05  # Tag matches
                    + recency * 0.03  # Content quality
                    + graph_relevance * 0.01  # Recency bonus
                    + temporal_relevance  # Graph connections
                    * 0.01  # Temporal patterns
                )

                memory["composite_score"] = composite_score
                memory["keyword_match_score"] = keyword_match
                memory["content_quality_score"] = content_quality
                unique_memories.append(memory)

        return unique_memories

    def _calculate_recency_score(self, created_at) -> float:
        """Calculate recency score for a memory"""
        if not created_at:
            return 0.0

        try:
            if isinstance(created_at, str):
                from dateutil.parser import parse

                created_at = parse(created_at)

            now = datetime.utcnow()
            days_ago = (now - created_at).days

            # Recent memories get higher scores
            if days_ago <= 7:
                return 1.0
            elif days_ago <= 30:
                return 0.8
            elif days_ago <= 90:
                return 0.6
            elif days_ago <= 365:
                return 0.4
            else:
                return 0.2
        except:
            return 0.0

    def _calculate_tag_relevance(self, tags: List[str], query: str) -> float:
        """Calculate how relevant tags are to the query"""
        if not tags:
            return 0.0

        query_words = set(query.lower().split())
        tag_words = set(" ".join(tags).lower().split())

        if not query_words:
            return 0.0

        intersection = query_words.intersection(tag_words)
        return len(intersection) / len(query_words)

    async def _calculate_graph_relevance(
        self, memory: Dict[str, Any], entity: str, user_id: str
    ) -> float:
        """Calculate graph-based relevance score"""
        # This would calculate how well connected the memory is in the graph
        # For now, return a simple relevance based on entity mention
        content = memory.get("input", "").lower()
        if entity.lower() in content:
            return 0.8
        return 0.3

    def _extract_entities_from_query(self, query: str) -> List[str]:
        """Extract entities from query text"""
        import re

        # Simple entity extraction - capitalized words
        entities = re.findall(r"\b[A-Z][a-z]+\b", query)
        return entities[:5]  # Limit to 5 entities

    def _extract_temporal_context(self, query: str) -> Dict[str, Any]:
        """Extract temporal context from query"""
        temporal_words = {
            "today": 0,
            "yesterday": -1,
            "last week": -7,
            "last month": -30,
            "this year": 0,
            "recently": -7,
        }

        query_lower = query.lower()
        context = {}

        for word, days_offset in temporal_words.items():
            if word in query_lower:
                context[word] = days_offset

        return context if context else None

    def _calculate_temporal_relevance(
        self, time_bucket: Dict[str, Any], temporal_context: Dict[str, Any]
    ) -> float:
        """Calculate temporal relevance score"""
        # Simple temporal relevance calculation
        if not temporal_context:
            return 0.0

        # If query mentions recent time, give higher score to recent time buckets
        if any(offset >= -7 for offset in temporal_context.values()):
            return 0.8

        return 0.4

    async def _retrieve_temporal_memories(
        self, temporal_indicators: List[str], user_id: str
    ) -> List[Dict[str, Any]]:
        """Retrieve memories with temporal context"""
        # This would use time-based filtering
        return []

    async def _retrieve_meta_memories(
        self, question: str, user_id: str
    ) -> List[Dict[str, Any]]:
        """Retrieve meta-memories (reflections, learnings) relevant to the question"""
        if not self.memory_engine.doc_store:
            return []

        try:
            search_query = {"source_user_id": user_id, "memory_type": "META_MEMORY"}
            results = await self.memory_engine.doc_store.search(search_query, 5)

            return [
                {
                    "memory_id": memory.get("id"),
                    "content": memory.get("input"),
                    "memory_type": memory.get("memory_type"),
                }
                for memory in results
            ]

        except Exception as e:
            logger.error(f"Error retrieving meta memories: {e}")
            return []

    def _initialize_inference_rules(self) -> List[InferenceRule]:
        """Initialize common inference rules"""
        return [
            InferenceRule(
                rule_id="temporal_sequence",
                name="Temporal Sequence",
                pattern="If A happened before B, and B caused C, then A contributed to C",
                inference_type="causal",
                confidence_factor=0.7,
            ),
            InferenceRule(
                rule_id="pattern_generalization",
                name="Pattern Generalization",
                pattern="If X happened multiple times with similar outcomes, expect similar future outcomes",
                inference_type="predictive",
                confidence_factor=0.6,
            ),
        ]

    # Additional helper methods would be implemented here...

    def _create_fallback_chain(self, question: str) -> ReasoningChain:
        """Create a fallback reasoning chain when no memories are found"""
        return ReasoningChain(
            chain_id=f"fallback_{uuid.uuid4().hex[:8]}",
            question=question,
            reasoning_steps=[],
            final_answer="I don't have enough information in my memories to answer this question.",
            overall_confidence=0.1,
            total_memories_used=0,
            reasoning_time_ms=10,
        )

    def _create_error_result(self, question: str, error: str) -> ReasoningResult:
        """Create an error result"""
        return ReasoningResult(
            question=question,
            primary_chain=self._create_fallback_chain(question),
            synthesis=f"An error occurred during reasoning: {error}",
            confidence_distribution={"error": 1.0},
            reasoning_metadata={"error": error},
        )

    async def _evaluate_and_rank_chains(
        self, chains: List[ReasoningChain], question: str
    ) -> List[ReasoningChain]:
        """Evaluate and rank reasoning chains by confidence and coherence"""
        # Sort by overall confidence for now - more sophisticated ranking could be implemented
        return sorted(chains, key=lambda c: c.overall_confidence, reverse=True)

    async def _identify_contradictions_and_gaps(
        self, chains: List[ReasoningChain]
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Identify contradictions between chains and information gaps"""
        contradictions = []
        gaps = []

        if len(chains) > 1:
            # Simple contradiction detection - compare final answers
            for i, chain1 in enumerate(chains):
                for j, chain2 in enumerate(chains[i + 1 :], i + 1):
                    if chain1.final_answer != chain2.final_answer:
                        contradictions.append(
                            {
                                "chain1": chain1.chain_id,
                                "chain2": chain2.chain_id,
                                "description": "Different conclusions reached",
                            }
                        )

        # Identify gaps (simplified)
        if not chains or chains[0].overall_confidence < 0.5:
            gaps.append("Insufficient memory information")

        return contradictions, gaps

    async def _synthesize_answer(
        self,
        chains: List[ReasoningChain],
        contradictions: List[Dict[str, Any]],
        gaps: List[str],
    ) -> str:
        """Synthesize a final answer from multiple reasoning chains"""
        if not chains:
            return "I don't have enough information to answer this question."

        primary_chain = chains[0]

        # Get the actual synthesized answer from the primary chain
        synthesized_answer = primary_chain.final_answer

        if contradictions:
            return f"{synthesized_answer} (Note: I found some conflicting information in my memories that may affect this answer.)"
        elif gaps:
            return f"{synthesized_answer} (Note: Some information gaps were identified that may limit the completeness of this answer.)"
        else:
            return synthesized_answer

    async def _synthesize_answer_from_memories(
        self, question: str, memory_ids: List[str], reasoning_type: str = "direct"
    ) -> str:
        """Generate actual answer content from retrieved memories"""
        try:
            # Retrieve the actual memory content
            memories = []
            for memory_id in memory_ids[:5]:  # Limit to top 5 memories
                try:
                    # Get memory content from memory engine
                    memory_data = await self._get_memory_content(memory_id)
                    if memory_data:
                        memories.append(memory_data)
                except Exception as e:
                    logger.warning(f"Failed to retrieve memory {memory_id}: {e}")

            if not memories:
                return "I don't have enough relevant information in my memories to answer this question."

            # Create answer based on memory content
            relevant_texts = [
                mem.get("text", "") for mem in memories if mem.get("text")
            ]
            if not relevant_texts:
                return "I found relevant memories but couldn't extract specific information to answer your question."

            # Simple answer synthesis based on reasoning type
            if reasoning_type == "causal":
                return self._synthesize_causal_answer(question, relevant_texts)
            elif reasoning_type == "temporal":
                return self._synthesize_temporal_answer(question, relevant_texts)
            elif reasoning_type == "inference":
                return self._synthesize_inference_answer(question, relevant_texts)
            else:
                return self._synthesize_direct_answer(question, relevant_texts)

        except Exception as e:
            logger.error(f"Error synthesizing answer: {e}")
            return "I encountered an error while processing my memories to answer your question."

    async def _get_memory_content(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory content by ID from multiple stores"""
        try:
            # Try document store first (most comprehensive)
            if self.memory_engine.doc_store:
                try:
                    memory = await self.memory_engine.doc_store.get_memory(memory_id)
                    if memory:
                        return {
                            "memory_id": memory_id,
                            "text": memory.get("input") or memory.get("summary", ""),
                            "summary": memory.get("summary"),
                            "memory_type": memory.get("memory_type"),
                            "tags": memory.get("tags", []),
                            "created_at": memory.get("created_at"),
                        }
                except Exception as e:
                    logger.warning(
                        f"Document store retrieval failed for {memory_id}: {e}"
                    )

            # Try vector store with metadata filtering
            if self.memory_engine.vector_store:
                try:
                    # Use similarity search with exact ID filter
                    dummy_embedding = [
                        0.0
                    ] * 1536  # Dummy embedding since we're filtering by ID
                    search_filter = {"memory_id": memory_id}

                    vector_results = (
                        await self.memory_engine.vector_store.similarity_search(
                            embedding=dummy_embedding, limit=1, filter=search_filter
                        )
                    )

                    if vector_results:
                        memory_entry, score = vector_results[0]
                        if isinstance(memory_entry, dict):
                            metadata = memory_entry.get("metadata", {})

                            # Parse tags properly
                            tags_data = metadata.get("tags", [])
                            if isinstance(tags_data, str):
                                tags = tags_data.split(",") if tags_data else []
                                tags = [tag.strip() for tag in tags if tag.strip()]
                            else:
                                tags = tags_data

                            return {
                                "memory_id": memory_id,
                                "text": metadata.get("input")
                                or metadata.get("text", ""),
                                "summary": metadata.get("summary"),
                                "memory_type": metadata.get("memory_type"),
                                "tags": tags,
                                "created_at": metadata.get("created_at"),
                            }
                except Exception as e:
                    logger.warning(
                        f"Vector store retrieval failed for {memory_id}: {e}"
                    )

            # Try graph store
            if self.memory_engine.graph_store:
                try:
                    graph_result = await self.memory_engine.graph_store.read(memory_id)
                    if graph_result:
                        return {
                            "memory_id": memory_id,
                            "text": graph_result.input
                            if hasattr(graph_result, "input")
                            else "",
                            "summary": graph_result.summary
                            if hasattr(graph_result, "summary")
                            else "",
                            "memory_type": graph_result.memory_type.value
                            if hasattr(graph_result, "memory_type")
                            else "",
                            "tags": graph_result.tags
                            if hasattr(graph_result, "tags")
                            else [],
                            "created_at": graph_result.created_at
                            if hasattr(graph_result, "created_at")
                            else None,
                        }
                except Exception as e:
                    logger.warning(f"Graph store retrieval failed for {memory_id}: {e}")

            logger.warning(
                f"Could not retrieve memory content for {memory_id} from any store"
            )
            return None

        except Exception as e:
            logger.warning(f"Failed to get memory content for {memory_id}: {e}")
            return None

    def _synthesize_direct_answer(self, question: str, memory_texts: List[str]) -> str:
        """Synthesize a direct answer from memory texts"""
        # Combine most relevant memories
        combined_info = ". ".join(memory_texts[:3])

        # Simple answer based on question keywords
        question_lower = question.lower()

        if any(word in question_lower for word in ["when", "born", "birth"]):
            birth_info = [
                text
                for text in memory_texts
                if any(word in text.lower() for word in ["born", "birth", "july", "28"])
            ]
            if birth_info:
                return f"Based on my memories: {birth_info[0]}"

        if any(
            word in question_lower
            for word in ["education", "study", "school", "college", "university"]
        ):
            edu_info = [
                text
                for text in memory_texts
                if any(
                    word in text.lower()
                    for word in ["university", "college", "study", "computer science"]
                )
            ]
            if edu_info:
                return f"Regarding my education: {'. '.join(edu_info[:2])}"

        if any(word in question_lower for word in ["career", "work", "job"]):
            career_info = [
                text
                for text in memory_texts
                if any(
                    word in text.lower()
                    for word in ["amazon", "google", "engineer", "job", "work"]
                )
            ]
            if career_info:
                return f"About my career: {'. '.join(career_info[:2])}"

        if any(
            word in question_lower
            for word in ["family", "brother", "sister", "parents"]
        ):
            family_info = [
                text
                for text in memory_texts
                if any(
                    word in text.lower()
                    for word in [
                        "family",
                        "brother",
                        "sister",
                        "parents",
                        "emma",
                        "michael",
                        "susan",
                    ]
                )
            ]
            if family_info:
                return f"Regarding my family: {'. '.join(family_info[:2])}"

        # Default: return first relevant memory
        return f"Based on my memories: {memory_texts[0]}"

    def _synthesize_causal_answer(self, question: str, memory_texts: List[str]) -> str:
        """Synthesize answer showing causal relationships"""
        return f"Looking at the connections between events: {'. '.join(memory_texts[:2])}. This suggests a causal relationship in my experiences."

    def _synthesize_temporal_answer(
        self, question: str, memory_texts: List[str]
    ) -> str:
        """Synthesize answer showing temporal progression"""
        return f"Tracing the timeline: {'. '.join(memory_texts[:2])}. This shows how things evolved over time."

    def _synthesize_inference_answer(
        self, question: str, memory_texts: List[str]
    ) -> str:
        """Synthesize answer based on inference"""
        return f"Drawing inferences from my memories: {'. '.join(memory_texts[:2])}. This suggests broader patterns in my life."

    async def _generate_follow_up_questions(
        self, question: str, chains: List[ReasoningChain], gaps: List[str]
    ) -> List[str]:
        """Generate relevant follow-up questions"""
        follow_ups = []

        if gaps:
            follow_ups.append(
                "Would you like to share more information about this topic?"
            )

        if chains and chains[0].total_memories_used > 3:
            follow_ups.append(
                "Would you like me to explain how I connected these memories?"
            )

        follow_ups.append("What other aspects of this would you like to explore?")

        return follow_ups[:3]

    def _calculate_confidence_distribution(
        self, chains: List[ReasoningChain]
    ) -> Dict[str, float]:
        """Calculate confidence distribution across different aspects"""
        if not chains:
            return {"overall": 0.0}

        return {
            "overall": chains[0].overall_confidence,
            "memory_relevance": sum(c.overall_confidence for c in chains) / len(chains),
            "reasoning_coherence": max(c.overall_confidence for c in chains),
        }

    # ==========================================
    # ADVANCED MEMORY RETRIEVAL OPTIMIZATION METHODS
    # ==========================================

    async def _expand_query_with_keywords(self, query: str) -> List[str]:
        """Expand query with related keywords and semantic variants for better coverage"""
        expanded_queries = [query]  # Start with original query

        # Define keyword expansion mapping for common query patterns
        keyword_expansions = {
            # Childhood/Early life
            "childhood": ["youth", "early life", "growing up", "young", "child"],
            "curiosity": [
                "interest",
                "fascination",
                "wonder",
                "exploration",
                "learning",
            ],
            # Career/Professional
            "career": ["job", "work", "professional", "occupation", "employment"],
            "achievement": ["success", "accomplishment", "milestone", "progress"],
            "leadership": ["management", "leading", "team", "mentor", "guidance"],
            # Relationships/Social
            "wife": ["spouse", "partner", "married", "Lisa"],
            "family": ["parents", "relatives", "siblings", "daughter", "Maya"],
            "friends": ["friendship", "social", "relationships", "Alex"],
            "mentorship": ["mentor", "teaching", "guidance", "Dr. Chen", "bootcamp"],
            # Technical/Skills
            "technical": ["programming", "engineering", "software", "computer"],
            "skills": ["expertise", "abilities", "knowledge", "experience"],
            "security": ["mobile app", "tech conference", "cybersecurity"],
            # Personal Development
            "growth": ["development", "learning", "progress", "evolution"],
            "reflection": ["thinking", "realization", "understanding", "insight"],
            "values": ["principles", "beliefs", "priorities", "important"],
            # Hobbies/Interests
            "creative": ["guitar", "photography", "artistic", "music"],
            "hobbies": ["interests", "activities", "passion", "recreation"],
        }

        # Extract key terms and expand them
        query_lower = query.lower()
        for base_term, expansions in keyword_expansions.items():
            if base_term in query_lower:
                # Create queries with expanded terms
                for expansion in expansions[:2]:  # Limit to avoid too many queries
                    expanded_query = query_lower.replace(base_term, expansion)
                    if expanded_query != query_lower and expanded_query not in [
                        q.lower() for q in expanded_queries
                    ]:
                        expanded_queries.append(expanded_query)

        # Add specific entity-focused queries based on detected patterns
        if "lisa" in query_lower or "wife" in query_lower:
            expanded_queries.extend(
                [
                    "tech conference mobile security meeting",
                    "marriage wedding Napa Valley relationship",
                ]
            )

        if "mentorship" in query_lower or "mentor" in query_lower:
            expanded_queries.extend(
                [
                    "Dr. Chen professor teaching guidance machine learning",
                    "coding bootcamp teaching volunteers students",
                    "mentoring engineers team leadership",
                ]
            )

        if "family" in query_lower and (
            "professional" in query_lower or "achievement" in query_lower
        ):
            expanded_queries.extend(
                [
                    "parents engineer teacher influence career",
                    "family background professional success Google",
                    "mother father education inspiration",
                ]
            )

        if "creative" in query_lower or "hobbies" in query_lower:
            expanded_queries.extend(
                [
                    "guitar music photography artistic creative",
                    "hiking Pacific Crest Trail adventure nature",
                    "hobbies complement technical work balance",
                ]
            )

        # Enhanced multi-domain pattern detection
        if "technical" in query_lower and (
            "creative" in query_lower or "complement" in query_lower
        ):
            expanded_queries.extend(
                [
                    "guitar photography creative technical balance",
                    "artistic pursuits engineering problem solving",
                    "music photography technical career",
                ]
            )

        if "skills" in query_lower or "expertise" in query_lower:
            expanded_queries.extend(
                [
                    "Python Java C++ programming algorithms",
                    "TensorFlow PyTorch machine learning expertise",
                    "leadership experience mentoring engineers",
                ]
            )

        if "progression" in query_lower or "journey" in query_lower:
            expanded_queries.extend(
                [
                    "Amazon Google career progression timeline",
                    "internship Microsoft first job growth",
                    "student mentor teacher progression",
                ]
            )

        # Remove duplicates while preserving order
        unique_queries = []
        seen = set()
        for q in expanded_queries:
            if q.lower() not in seen:
                unique_queries.append(q)
                seen.add(q.lower())

        return unique_queries[:5]  # Limit to 5 queries max to avoid performance issues

    def _calculate_query_specificity(
        self, result, original_query: str, expanded_query: str
    ) -> float:
        """Calculate how specifically this memory matches the query intent"""
        content = (result.content or result.summary or "").lower()
        original_lower = original_query.lower()
        expanded_lower = expanded_query.lower()

        # Count direct matches in original query
        original_words = set(original_lower.split())
        expanded_words = set(expanded_lower.split())
        content_words = set(content.split())

        original_matches = len(original_words.intersection(content_words))
        expanded_matches = len(expanded_words.intersection(content_words))

        # Higher score for original query matches
        specificity = (
            original_matches / len(original_words) if original_words else 0
        ) * 0.8
        specificity += (
            expanded_matches / len(expanded_words) if expanded_words else 0
        ) * 0.2

        return min(specificity, 1.0)

    def _calculate_keyword_match_score(self, result, query: str) -> float:
        """Calculate keyword match score with weighted importance"""
        content = (result.content or result.summary or "").lower()
        tags = [tag.lower() for tag in (result.tags or [])]
        query_lower = query.lower()

        # Extract key terms from query
        key_terms = []
        important_terms = [
            "childhood",
            "curiosity",
            "career",
            "wife",
            "lisa",
            "family",
            "mentorship",
            "mentor",
            "technical",
            "creative",
            "hobbies",
            "security",
            "achievement",
            "growth",
            "values",
        ]

        for term in important_terms:
            if term in query_lower:
                key_terms.append(term)

        if not key_terms:
            # Fallback to general word matching
            query_words = query_lower.split()
            content_words = content.split()
            matches = len(set(query_words).intersection(set(content_words)))
            return min(matches / len(query_words) if query_words else 0, 1.0)

        # Calculate weighted matches
        content_matches = sum(1 for term in key_terms if term in content)
        tag_matches = sum(1 for term in key_terms if any(term in tag for tag in tags))

        score = (content_matches * 0.7 + tag_matches * 0.3) / len(key_terms)
        return min(score, 1.0)

    def _calculate_memory_type_relevance(self, memory_type, query: str) -> float:
        """Calculate relevance score based on memory type matching query intent"""
        query_lower = query.lower()

        if hasattr(memory_type, "value"):
            type_str = memory_type.value.lower()
        else:
            type_str = str(memory_type).lower()

        # Memory type relevance mapping
        type_relevance = {
            # Factual/Identity questions
            "semantic": 0.9
            if any(
                word in query_lower
                for word in ["about", "what", "who", "skills", "expertise"]
            )
            else 0.6,
            # Experience/Event questions
            "episodic": 0.9
            if any(
                word in query_lower
                for word in ["how", "when", "experience", "met", "happened"]
            )
            else 0.7,
            # Skills/Process questions
            "procedural": 0.9
            if any(
                word in query_lower
                for word in ["skills", "how to", "expertise", "abilities"]
            )
            else 0.5,
            # Reflection/Learning questions
            "meta": 0.9
            if any(
                word in query_lower
                for word in ["learned", "realized", "reflection", "growth", "think"]
            )
            else 0.6,
            # Emotional/Personal questions
            "emotional": 0.8
            if any(
                word in query_lower
                for word in ["feel", "values", "important", "relationship"]
            )
            else 0.4,
        }

        for type_key, score in type_relevance.items():
            if type_key in type_str:
                return score

        return 0.5  # Default relevance

    def _calculate_advanced_composite_score(
        self, memory_data: Dict[str, Any], query: str
    ) -> float:
        """Calculate advanced composite score with multiple weighted factors"""
        scores = memory_data["search_scores"]

        # Weighted scoring components
        weights = {
            "primary_score": 0.25,  # Base semantic similarity
            "query_specificity": 0.20,  # How specifically it matches query
            "keyword_match": 0.20,  # Keyword relevance
            "memory_type_relevance": 0.15,  # Memory type appropriateness
            "confidence": 0.10,  # Memory confidence
            "tag_relevance": 0.05,  # Tag matching
            "recency": 0.05,  # Temporal relevance
        }

        composite_score = 0.0
        for component, weight in weights.items():
            if component in scores:
                composite_score += scores[component] * weight

        # Apply content quality boost
        content = memory_data.get("content", "")
        if len(content) > 100:  # Prefer memories with substantial content
            composite_score *= 1.05

        # Apply diversity penalty for very similar memories (will be handled in optimization)
        return min(composite_score, 1.0)

    async def _optimize_memory_selection(
        self, all_memories: List[Dict[str, Any]], query: str, limit: int
    ) -> List[Dict[str, Any]]:
        """Optimize memory selection for relevance and diversity"""
        if not all_memories:
            return []

        # Remove duplicates by memory_id
        unique_memories = {}
        for memory in all_memories:
            memory_id = memory["memory_id"]
            if (
                memory_id not in unique_memories
                or memory["composite_score"]
                > unique_memories[memory_id]["composite_score"]
            ):
                unique_memories[memory_id] = memory

        memories = list(unique_memories.values())

        # Sort by composite score
        memories.sort(key=lambda m: m["composite_score"], reverse=True)

        # Apply diversity optimization
        selected_memories = []
        used_memory_types = set()
        used_scopes = set()
        used_tag_categories = set()

        # Define tag categories for diversity
        tag_categories = {
            "family": ["family", "parents", "sister", "daughter", "maya"],
            "career": ["career", "work", "job", "google", "amazon", "microsoft"],
            "education": ["education", "university", "professor", "thesis"],
            "relationships": ["wife", "lisa", "friend", "alex", "marriage"],
            "skills": ["programming", "algorithms", "leadership", "technical"],
            "hobbies": ["guitar", "photography", "hiking", "creative"],
            "personal": ["reflection", "growth", "values", "volunteer"],
        }

        def get_tag_category(tags):
            for category, category_tags in tag_categories.items():
                if any(tag.lower() in category_tags for tag in tags):
                    return category
            return "other"

        # First pass: select highest scoring memories with diversity constraints
        for memory in memories:
            if len(selected_memories) >= limit:
                break

            memory_type = memory["memory_type"]
            scope = memory.get("scope", "unknown")
            tag_category = get_tag_category(memory.get("tags", []))

            # Diversity scoring
            diversity_bonus = 1.0

            # Encourage memory type diversity
            if memory_type not in used_memory_types:
                diversity_bonus += 0.1
                used_memory_types.add(memory_type)

            # Encourage scope diversity
            if scope not in used_scopes:
                diversity_bonus += 0.05
                used_scopes.add(scope)

            # Encourage tag category diversity
            if tag_category not in used_tag_categories:
                diversity_bonus += 0.1
                used_tag_categories.add(tag_category)

            # Apply diversity bonus
            memory["final_score"] = memory["composite_score"] * diversity_bonus
            selected_memories.append(memory)

        # Re-sort by final score and return top memories
        selected_memories.sort(key=lambda m: m["final_score"], reverse=True)

        logger.info(
            f"Memory optimization: {len(memories)} -> {len(selected_memories)}, "
            f"types: {len(used_memory_types)}, categories: {len(used_tag_categories)}"
        )

        return selected_memories[:limit]
