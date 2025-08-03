"""
Optimized memory search with parallel execution and caching
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from app.api.v1.request.memory_request import SearchQuery
from app.api.v1.response.memory_response import SearchResult
from app.common.cache.memory_cache import query_cache, embedding_cache
from app.common.utils.batch_processor import BatchEmbeddingProcessor

logger = logging.getLogger(__name__)


class OptimizedMemorySearch:
    """Optimized search implementation with caching and parallel execution"""

    def __init__(self, memory_engine):
        self.memory_engine = memory_engine
        self.embedding_processor = BatchEmbeddingProcessor(
            self._generate_embeddings_batch, batch_size=50
        )

    async def search_memories(self, query: SearchQuery) -> List[SearchResult]:
        """Optimized memory search with parallel store queries"""

        start_time = datetime.utcnow()

        # Check cache first
        cache_key = self._get_cache_key(query)
        cached_results = await query_cache.get(cache_key)
        if cached_results:
            logger.info(f"Cache hit for query: {query.query[:50]}...")
            return [SearchResult(**r) for r in cached_results["results"]]

        # Generate query embedding (with caching)
        query_embedding = await self._get_query_embedding(query.query)

        # Execute searches in parallel across all stores
        search_tasks = []

        if self.memory_engine.vector_store:
            search_tasks.append(self._search_vector_store(query, query_embedding))

        if self.memory_engine.graph_store:
            search_tasks.append(self._search_graph_store(query))

        if self.memory_engine.doc_store:
            search_tasks.append(self._search_doc_store(query))

        # Wait for all searches to complete
        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Merge and rank results
        all_results = []
        for result in results:
            if not isinstance(result, Exception):
                all_results.extend(result)
            else:
                logger.error(f"Search error: {result}")

        # Deduplicate and sort by score
        unique_results = self._deduplicate_results(all_results)
        sorted_results = sorted(unique_results, key=lambda x: x.score, reverse=True)

        # Apply limit
        final_results = sorted_results[: query.limit]

        # Cache results
        cache_data = {
            "results": [r.dict() for r in final_results],
            "timestamp": datetime.utcnow().isoformat(),
            "query_time_ms": int(
                (datetime.utcnow() - start_time).total_seconds() * 1000
            ),
        }
        await query_cache.set(cache_key, cache_data)

        logger.info(
            f"Search completed in {cache_data['query_time_ms']}ms, found {len(final_results)} results"
        )

        return final_results

    async def _get_query_embedding(self, query_text: str) -> List[float]:
        """Get query embedding with caching"""

        # Check embedding cache
        cached_embedding = await embedding_cache.get_embedding(query_text)
        if cached_embedding:
            return cached_embedding

        # Generate new embedding
        embeddings = await self.embedding_processor.generate_embeddings([query_text])
        return embeddings[0] if embeddings else []

    async def _search_vector_store(
        self, query: SearchQuery, embedding: List[float]
    ) -> List[SearchResult]:
        """Search vector store"""
        try:
            if not embedding:
                return []

            # Prepare filters
            filters = {}
            if query.memory_types:
                filters["memory_type"] = {"$in": [t.value for t in query.memory_types]}
            if query.tags:
                filters["tags"] = {"$in": query.tags}

            # Search with optimized parameters
            results = await self.memory_engine.vector_store.search(
                embedding=embedding,
                filters=filters,
                limit=query.limit * 2,  # Get more results for better ranking
                threshold=query.threshold,
            )

            # Convert to SearchResult
            search_results = []
            for result in results:
                search_results.append(
                    SearchResult(
                        id=result["id"],
                        memory_type=result.get("memory_type", "semantic_memory"),
                        content=result.get("input", "")
                        if query.include_content
                        else None,
                        summary=result.get("summary", ""),
                        score=result.get("score", 0.0),
                        tags=result.get("tags", []),
                        created_at=result.get("created_at", datetime.utcnow()),
                        relevance=result.get("score", 0.0),
                        confidence=result.get("confidence", 1.0),
                    )
                )

            return search_results

        except Exception as e:
            logger.error(f"Vector store search error: {e}")
            return []

    async def _search_graph_store(self, query: SearchQuery) -> List[SearchResult]:
        """Search graph store for relationships"""
        try:
            if not self.memory_engine.graph_store:
                return []

            # Graph search implementation
            # This would use Neo4j queries to find related memories
            return []

        except Exception as e:
            logger.error(f"Graph store search error: {e}")
            return []

    async def _search_doc_store(self, query: SearchQuery) -> List[SearchResult]:
        """Search document store with text search"""
        try:
            if not self.memory_engine.doc_store:
                return []

            # Build MongoDB query
            mongo_query = {
                "$and": [
                    {
                        "$or": [
                            {"source_user_id": query.user_id},
                            {"user_id": query.user_id},
                            {"meta.user_id": query.user_id},
                            {"permissions.owner_id": query.user_id},
                        ]
                    },
                    {"$text": {"$search": query.query}},
                ]
            }

            if query.memory_types:
                mongo_query["$and"].append(
                    {"memory_type": {"$in": [t.value for t in query.memory_types]}}
                )

            if query.tags:
                mongo_query["$and"].append({"tags": {"$in": query.tags}})

            # Execute search with text score
            cursor = (
                self.memory_engine.doc_store.collection.find(
                    mongo_query, {"score": {"$meta": "textScore"}}
                )
                .sort([("score", {"$meta": "textScore"})])
                .limit(query.limit * 2)
            )

            results = []
            async for doc in cursor:
                # Normalize text score to 0-1 range
                text_score = min(doc.get("score", 0) / 10.0, 1.0)

                results.append(
                    SearchResult(
                        id=doc["id"],
                        memory_type=doc.get("memory_type", "semantic_memory"),
                        content=doc.get("input", "") if query.include_content else None,
                        summary=doc.get("summary", ""),
                        score=text_score,
                        tags=doc.get("tags", []),
                        created_at=doc.get("created_at", datetime.utcnow()),
                        relevance=text_score,
                        confidence=doc.get("confidence", 1.0),
                    )
                )

            return results

        except Exception as e:
            logger.error(f"Document store search error: {e}")
            return []

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Deduplicate results by ID, keeping highest score"""

        seen = {}
        for result in results:
            if result.id not in seen or result.score > seen[result.id].score:
                seen[result.id] = result

        return list(seen.values())

    def _get_cache_key(self, query: SearchQuery) -> str:
        """Generate cache key for query"""

        filters = {
            "memory_types": [t.value for t in query.memory_types]
            if query.memory_types
            else None,
            "tags": query.tags,
            "limit": query.limit,
            "threshold": query.threshold,
            "include_content": query.include_content,
        }

        return query_cache.get_key(query.user_id, query.query, filters)

    async def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a batch of texts"""

        # This would call the actual embedding API
        # For now, return mock embeddings
        if hasattr(self.memory_engine, "embedder") and self.memory_engine.embedder:
            return await self.memory_engine.embedder.embed_batch(texts)

        # Fallback mock implementation
        return [[0.1] * 1536 for _ in texts]


async def parallel_memory_search(
    memory_engine, queries: List[SearchQuery]
) -> List[List[SearchResult]]:
    """Execute multiple search queries in parallel"""

    searcher = OptimizedMemorySearch(memory_engine)

    # Create search tasks
    search_tasks = [searcher.search_memories(query) for query in queries]

    # Execute in parallel with timeout
    try:
        results = await asyncio.wait_for(
            asyncio.gather(*search_tasks, return_exceptions=True),
            timeout=10.0,  # 10 second timeout
        )

        # Process results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Search {i} failed: {result}")
                final_results.append([])
            else:
                final_results.append(result)

        return final_results

    except asyncio.TimeoutError:
        logger.error("Parallel search timed out")
        return [[] for _ in queries]
