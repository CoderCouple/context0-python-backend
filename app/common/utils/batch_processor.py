"""
Batch processing utilities for performance optimization
"""
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Callable
import logging
from datetime import datetime

from app.common.cache.memory_cache import embedding_cache

logger = logging.getLogger(__name__)


class BatchEmbeddingProcessor:
    """Batch processor for embedding generation with caching"""

    def __init__(self, embedding_function: Callable, batch_size: int = 50):
        self.embedding_function = embedding_function
        self.batch_size = batch_size

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts with batching and caching"""

        # Step 1: Check cache for existing embeddings
        cache_results = await self._check_cache(texts)

        # Step 2: Identify texts that need embedding
        texts_to_embed = []
        indices_to_embed = []
        results = [None] * len(texts)

        for i, (text, cached_embedding) in enumerate(zip(texts, cache_results)):
            if cached_embedding is not None:
                results[i] = cached_embedding
            else:
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        logger.info(
            f"Found {len(texts) - len(texts_to_embed)} cached embeddings, need to generate {len(texts_to_embed)}"
        )

        # Step 3: Generate embeddings for uncached texts in batches
        if texts_to_embed:
            new_embeddings = await self._batch_generate(texts_to_embed)

            # Step 4: Cache new embeddings and update results
            for idx, text, embedding in zip(
                indices_to_embed, texts_to_embed, new_embeddings
            ):
                await embedding_cache.set_embedding(text, embedding)
                results[idx] = embedding

        return results

    async def _check_cache(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Check cache for existing embeddings"""
        tasks = [embedding_cache.get_embedding(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def _batch_generate(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batches"""
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            logger.info(
                f"Generating embeddings for batch {i//self.batch_size + 1} ({len(batch)} texts)"
            )

            try:
                # Call the embedding function for this batch
                batch_embeddings = await self.embedding_function(batch)
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"Error generating embeddings for batch: {e}")
                # Return zero embeddings as fallback
                all_embeddings.extend(
                    [[0.0] * 1536 for _ in batch]
                )  # Assuming 1536 dimensions

        return all_embeddings


class BatchQueryProcessor:
    """Batch processor for database queries"""

    def __init__(self, collection):
        self.collection = collection

    async def batch_get_by_ids(self, ids: List[str], user_id: str) -> Dict[str, Any]:
        """Batch get documents by IDs"""

        # Execute batch query
        documents = await self.collection.find(
            {
                "id": {"$in": ids},
                "$or": [
                    {"source_user_id": user_id},
                    {"user_id": user_id},
                    {"meta.user_id": user_id},
                    {"permissions.owner_id": user_id},
                ],
            }
        ).to_list(None)

        # Create lookup dict
        return {doc["id"]: doc for doc in documents}

    async def batch_update(self, updates: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Batch update multiple documents"""

        results = {}

        # Group updates by operation type for efficiency
        bulk_operations = []

        for update in updates:
            operation = {
                "update_one": {
                    "filter": {"id": update["id"]},
                    "update": {"$set": update["data"]},
                    "upsert": update.get("upsert", False),
                }
            }
            bulk_operations.append(operation)

        if bulk_operations:
            try:
                result = await self.collection.bulk_write(
                    bulk_operations, ordered=False
                )

                # Map results back to IDs
                for update in updates:
                    results[update["id"]] = True

                logger.info(f"Batch updated {result.modified_count} documents")
            except Exception as e:
                logger.error(f"Error in batch update: {e}")
                # Mark all as failed
                for update in updates:
                    results[update["id"]] = False

        return results


class ParallelTaskExecutor:
    """Execute multiple async tasks in parallel with error handling"""

    @staticmethod
    async def execute_parallel(
        tasks: List[Callable], max_concurrent: int = 10
    ) -> List[Any]:
        """Execute tasks in parallel with concurrency limit"""

        semaphore = asyncio.Semaphore(max_concurrent)

        async def bounded_task(task):
            async with semaphore:
                try:
                    return await task()
                except Exception as e:
                    logger.error(f"Error in parallel task: {e}")
                    return None

        results = await asyncio.gather(
            *[bounded_task(task) for task in tasks], return_exceptions=False
        )

        return results

    @staticmethod
    async def execute_with_timeout(
        tasks: List[Callable], timeout: float = 30.0
    ) -> List[Any]:
        """Execute tasks with timeout"""

        async def timed_task(task):
            try:
                return await asyncio.wait_for(task(), timeout=timeout)
            except asyncio.TimeoutError:
                logger.warning(f"Task timed out after {timeout}s")
                return None
            except Exception as e:
                logger.error(f"Error in timed task: {e}")
                return None

        results = await asyncio.gather(*[timed_task(task) for task in tasks])

        return results


# Utility functions for common batch operations


async def batch_search_memories(
    memory_engine, queries: List[Dict[str, Any]]
) -> List[Any]:
    """Batch search memories for multiple queries"""

    # Create search tasks
    search_tasks = []
    for query_data in queries:
        from app.api.v1.request.memory_request import SearchQuery

        query = SearchQuery(**query_data)
        search_tasks.append(lambda q=query: memory_engine.search_memories(q))

    # Execute in parallel
    executor = ParallelTaskExecutor()
    results = await executor.execute_parallel(search_tasks, max_concurrent=5)

    return results


async def batch_create_memories(
    memory_service, memories: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Batch create multiple memories"""

    success_count = 0
    failed_count = 0
    results = []

    # Process in batches to avoid overwhelming the system
    batch_size = 10

    for i in range(0, len(memories), batch_size):
        batch = memories[i : i + batch_size]

        # Create tasks for this batch
        create_tasks = []
        for memory_data in batch:
            from app.api.v1.request.memory_request import MemoryRecordInput

            memory_input = MemoryRecordInput(**memory_data)
            create_tasks.append(memory_service.create_memory(memory_input))

        # Execute batch
        batch_results = await asyncio.gather(*create_tasks, return_exceptions=True)

        # Process results
        for result in batch_results:
            if isinstance(result, Exception):
                failed_count += 1
                results.append({"success": False, "error": str(result)})
            else:
                if result.success:
                    success_count += 1
                else:
                    failed_count += 1
                results.append(result.dict())

    return {
        "total": len(memories),
        "success": success_count,
        "failed": failed_count,
        "results": results,
    }
