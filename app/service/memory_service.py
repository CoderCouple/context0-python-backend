"""Memory Service Layer - Business logic for memory operations."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.api.v1.request.memory_request import (
    BulkMemoryInput,
    MemoryAnalyticsQuery,
    MemoryExportRequest,
    MemoryRecordInput,
    MemoryUpdate,
    SearchQuery,
    TimeRangeQuery,
)
from app.api.v1.response.memory_response import (
    BulkMemoryResponse,
    ExportResponse,
    HealthResponse,
    MemoryAnalyticsResponse,
    MemoryEntry,
    MemoryResponse,
    SearchResponse,
    TimelineResponse,
)
from app.common.enum.memory import MemoryType
from app.memory.engine.memory_engine import MemoryEngine

logger = logging.getLogger(__name__)


class MemoryService:
    """Service layer for memory operations"""

    def __init__(self):
        """Initialize memory service"""
        self._memory_engine: Optional[MemoryEngine] = None

    @property
    def memory_engine(self) -> MemoryEngine:
        """Get memory engine instance (lazy initialization)"""
        if self._memory_engine is None:
            self._memory_engine = MemoryEngine.get_instance()
        return self._memory_engine

    # ==========================================
    # CORE MEMORY OPERATIONS
    # ==========================================

    async def create_memory(self, record_input: MemoryRecordInput) -> MemoryResponse:
        """Create a new memory"""
        try:
            logger.info(f"Creating memory for user {record_input.user_id}")
            response = await self.memory_engine.add_memory(record_input)

            if response.success:
                logger.info(f"Memory created successfully: {response.memory_id}")
            else:
                logger.warning(f"Failed to create memory: {response.message}")

            return response

        except Exception as e:
            logger.error(f"Error creating memory: {e}")
            return MemoryResponse(
                success=False,
                memory_id="",
                operation="CREATE",
                confidence=0.0,
                processing_time_ms=0,
                message=f"Service error: {str(e)}",
            )

    async def get_memory(self, memory_id: str, user_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID"""
        try:
            logger.info(f"Retrieving memory {memory_id} for user {user_id}")

            if not self.memory_engine.doc_store:
                raise ValueError("Document store not available")

            memory = await self.memory_engine.doc_store.get_memory(memory_id)

            if not memory:
                logger.warning(f"Memory {memory_id} not found")
                return None

            # Check user access
            if memory.get("meta", {}).get("user_id") != user_id:
                logger.warning(
                    f"Access denied for memory {memory_id} to user {user_id}"
                )
                raise PermissionError("Access denied")

            # Update access tracking
            access_count = memory.get("access_count", 0) + 1
            last_accessed = datetime.utcnow()
            await self.memory_engine.doc_store.update_memory(
                memory_id,
                {"access_count": access_count, "last_accessed": last_accessed},
            )

            memory["access_count"] = access_count
            memory["last_accessed"] = last_accessed

            logger.info(f"Memory {memory_id} retrieved successfully")
            return MemoryEntry(**memory)

        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            raise

    async def list_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> List[MemoryEntry]:
        """List memories with filtering and pagination"""
        try:
            logger.info(f"Listing memories for user {user_id}")

            if not self.memory_engine.doc_store:
                raise ValueError("Document store not available")

            memories = await self.memory_engine.doc_store.list_memories(
                user_id=user_id,
                memory_type=memory_type,
                tags=tags,
                limit=limit,
                offset=offset,
                sort_by=sort_by,
                sort_order=sort_order,
            )

            # Transform MongoDB documents to match MemoryEntry model structure
            result = []
            for memory in memories:
                # Extract confidence from meta object and add to top level
                transformed_memory = memory.copy()
                if "meta" in memory and isinstance(memory["meta"], dict):
                    transformed_memory["confidence"] = memory["meta"].get(
                        "confidence_score", 0.0
                    )
                else:
                    transformed_memory["confidence"] = 0.0

                # Ensure required fields have default values
                transformed_memory.setdefault("access_count", 0)
                transformed_memory.setdefault("is_deleted", False)
                transformed_memory.setdefault("tags", [])
                transformed_memory.setdefault("meta", {})

                # Convert memory_type string to enum if needed
                if "memory_type" in transformed_memory and isinstance(
                    transformed_memory["memory_type"], str
                ):
                    from app.common.enum.memory import MemoryType

                    try:
                        transformed_memory["memory_type"] = MemoryType(
                            transformed_memory["memory_type"]
                        )
                    except ValueError:
                        transformed_memory["memory_type"] = MemoryType.SEMANTIC_MEMORY

                try:
                    result.append(MemoryEntry(**transformed_memory))
                except Exception as e:
                    logger.warning(
                        f"Failed to create MemoryEntry from document {memory.get('id', 'unknown')}: {e}"
                    )
                    continue

            logger.info(f"Retrieved {len(result)} memories for user {user_id}")
            return result

        except Exception as e:
            logger.error(f"Error listing memories for user {user_id}: {e}")
            raise

    async def search_memories(self, query: SearchQuery) -> SearchResponse:
        """Search memories using semantic similarity"""
        try:
            logger.info(f"Searching memories for user {query.user_id}: '{query.query}'")
            response = await self.memory_engine.search_memories(query)

            logger.info(
                f"Search completed: {len(response.results)} results in {response.query_time_ms}ms"
            )
            return response

        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            import traceback

            logger.error(f"Full traceback: {traceback.format_exc()}")
            return SearchResponse(
                success=False,
                results=[],
                total_count=0,
                query_time_ms=0,
                query=query.query,
                filters_applied={},
            )

    async def update_memory(
        self, memory_id: str, update: MemoryUpdate, user_id: str
    ) -> Dict[str, Any]:
        """Update an existing memory"""
        try:
            logger.info(f"Updating memory {memory_id} for user {user_id}")

            if not self.memory_engine.doc_store:
                raise ValueError("Document store not available")

            # Get existing memory
            existing = await self.memory_engine.doc_store.get_memory(memory_id)
            if not existing:
                raise ValueError("Memory not found")

            # Check user access
            if existing.get("meta", {}).get("user_id") != user_id:
                raise PermissionError("Access denied")

            # Prepare updates
            updates = {"updated_at": datetime.utcnow()}

            if update.text:
                updates["input"] = update.text
                # Re-process memory if text changed
                from app.common.enum.memory import MemoryType
                from app.memory.models.memory_record import MemoryRecord

                temp_record = MemoryRecord(
                    user_id=existing["meta"]["user_id"],
                    session_id=existing["meta"].get("session_id", "update"),
                    raw_text=update.text,
                    memory_type=MemoryType(existing["memory_type"]),
                )

                handler = self.memory_engine.router.get_handler(temp_record.memory_type)
                processed_entry = await handler.process(temp_record)

                updates.update(
                    {
                        "summary": processed_entry.summary,
                        "embedding": processed_entry.embedding,
                        "graph_links": [
                            link.dict() for link in processed_entry.graph_links
                        ],
                    }
                )

            if update.tags is not None:
                updates["tags"] = update.tags

            if update.metadata:
                existing_meta = existing.get("meta", {})
                existing_meta.update(update.metadata)
                updates["meta"] = existing_meta

            if update.scope:
                updates["scope"] = update.scope

            # Update in stores
            success = await self.memory_engine.doc_store.update_memory(
                memory_id, updates
            )

            if success and "embedding" in updates:
                # Update vector store if embedding changed
                updated_memory = existing.copy()
                updated_memory.update(updates)
                if self.memory_engine.vector_store:
                    await self.memory_engine.vector_store.upsert_memory(updated_memory)

            if success:
                logger.info(f"Memory {memory_id} updated successfully")
                return {"success": True, "message": "Memory updated successfully"}
            else:
                raise ValueError("Failed to update memory in store")

        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Error updating memory {memory_id}: {e}")
            raise

    async def delete_memory(self, memory_id: str, user_id: str) -> Dict[str, Any]:
        """Soft delete a memory"""
        try:
            logger.info(f"Deleting memory {memory_id} for user {user_id}")

            if not self.memory_engine.doc_store:
                raise ValueError("Document store not available")

            # Verify memory exists and user access
            memory = await self.memory_engine.doc_store.get_memory(memory_id)
            if not memory:
                raise ValueError("Memory not found")

            if memory.get("meta", {}).get("user_id") != user_id:
                raise PermissionError("Access denied")

            # Soft delete - mark as deleted but keep for audit
            updates = {
                "deleted_at": datetime.utcnow(),
                "is_deleted": True,
                "updated_at": datetime.utcnow(),
            }

            success = await self.memory_engine.doc_store.update_memory(
                memory_id, updates
            )

            if success:
                # Remove from vector store
                if self.memory_engine.vector_store:
                    await self.memory_engine.vector_store.delete_memory(memory_id)

                logger.info(f"Memory {memory_id} deleted successfully")
                return {"success": True, "message": "Memory deleted successfully"}
            else:
                raise ValueError("Failed to delete memory")

        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Error deleting memory {memory_id}: {e}")
            raise

    # ==========================================
    # ADVANCED OPERATIONS
    # ==========================================

    async def bulk_create_memories(
        self, records: List[MemoryRecordInput]
    ) -> BulkMemoryResponse:
        """Bulk create memories"""
        try:
            logger.info(f"Bulk creating {len(records)} memories")

            if len(records) > 50:
                raise ValueError("Bulk operations limited to 50 memories per request")

            import time

            start_time = time.time()

            results = await self.memory_engine.add_memories_bulk(records)
            processing_time = int((time.time() - start_time) * 1000)

            successful = len([r for r in results if r.get("success")])
            failed = len(results) - successful

            logger.info(
                f"Bulk operation completed: {successful} successful, {failed} failed"
            )

            return BulkMemoryResponse(
                success=True,
                processed=len(results),
                successful=successful,
                failed=failed,
                processing_time_ms=processing_time,
                results=results,
                errors=[r.get("message", "") for r in results if not r.get("success")],
            )

        except Exception as e:
            logger.error(f"Error in bulk create: {e}")
            return BulkMemoryResponse(
                success=False,
                processed=0,
                successful=0,
                failed=len(records),
                processing_time_ms=0,
                results=[],
                errors=[str(e)],
            )

    async def time_travel_query(
        self, user_id: str, target_time: datetime, query: Optional[str] = None
    ) -> TimelineResponse:
        """Query memory state at a specific point in time"""
        try:
            logger.info(f"Time travel query for user {user_id} at {target_time}")

            if not self.memory_engine.audit_store:
                raise ValueError(
                    "Audit store not configured - time travel not available"
                )

            # Get memories that existed at the target time
            memories = await self.memory_engine.audit_store.get_memory_state_at_time(
                user_id=user_id, target_time=target_time
            )

            if query:
                # Filter by semantic similarity if query provided
                handler = self.memory_engine.router.get_handler(
                    MemoryType.SEMANTIC_MEMORY
                )
                query_embedding = await handler._generate_embedding(query)

                scored_memories = []
                for memory in memories:
                    if memory.get("embedding"):
                        similarity = self.memory_engine._calculate_similarity(
                            query_embedding, memory["embedding"]
                        )
                        if similarity > 0.7:
                            memory["similarity_score"] = similarity
                            scored_memories.append(memory)

                memories = sorted(
                    scored_memories, key=lambda x: x["similarity_score"], reverse=True
                )

            timeline_entries = [
                {
                    "timestamp": target_time,
                    "event_type": "time_travel_query",
                    "query": query,
                    "memories_found": len(memories),
                    "memories": memories[:20],  # Limit results
                }
            ]

            logger.info(f"Time travel query completed: {len(memories)} memories found")

            return TimelineResponse(
                success=True,
                target_time=target_time,
                timeline=timeline_entries,
                total_events=len(timeline_entries),
            )

        except Exception as e:
            logger.error(f"Error in time travel query: {e}")
            return TimelineResponse(success=False, timeline=[], total_events=0)

    async def get_memory_evolution(
        self,
        memory_id: str,
        user_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
    ) -> TimelineResponse:
        """Get evolution history of a specific memory"""
        try:
            logger.info(f"Getting evolution for memory {memory_id}")

            # Verify user access to memory
            if not self.memory_engine.doc_store:
                raise ValueError("Document store not available")

            memory = await self.memory_engine.doc_store.get_memory(memory_id)
            if not memory:
                raise ValueError("Memory not found")

            if memory.get("meta", {}).get("user_id") != user_id:
                raise PermissionError("Access denied")

            if not self.memory_engine.audit_store:
                raise ValueError("Audit store not configured")

            # Get evolution from audit store
            evolution = await self.memory_engine.audit_store.get_memory_evolution(
                memory_id=memory_id, start_time=start_time, end_time=end_time
            )

            logger.info(f"Memory evolution retrieved: {len(evolution)} events")

            return TimelineResponse(
                success=True,
                memory_id=memory_id,
                timeline=evolution,
                total_events=len(evolution),
            )

        except PermissionError:
            raise
        except Exception as e:
            logger.error(f"Error getting memory evolution: {e}")
            raise

    async def get_system_stats(self, user_id: Optional[str] = None) -> HealthResponse:
        """Get comprehensive system statistics"""
        try:
            logger.info("Getting system statistics")

            import time

            from app.common.startup import app_start_time

            health = await self.memory_engine.health_check()

            stats = {
                "uptime_seconds": time.time() - app_start_time,
                "processing_stats": self.memory_engine.processing_stats,
                "health": health,
            }

            memory_count = 0
            if self.memory_engine.doc_store:
                try:
                    counts = await self.memory_engine.doc_store.get_memory_counts(
                        user_id
                    )
                    memory_count = counts.get("total", 0)
                    stats["memory_counts"] = counts
                except Exception:
                    pass

            if user_id and self.memory_engine.doc_store:
                try:
                    stats[
                        "user_stats"
                    ] = await self.memory_engine.doc_store.get_user_stats(user_id)
                except Exception:
                    pass

            # Convert store health to boolean values
            stores_health = {}
            for store_name, store_status in health.get("stores", {}).items():
                if store_status is None:
                    stores_health[store_name] = False
                elif isinstance(store_status, bool):
                    stores_health[store_name] = store_status
                else:
                    stores_health[store_name] = bool(store_status)

            return HealthResponse(
                status="healthy" if health.get("engine") else "degraded",
                version="1.0.0",
                uptime_seconds=stats["uptime_seconds"],
                stores=stores_health,
                memory_count=memory_count,
                processing_stats=stats.get("processing_stats", {}),
            )

        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return HealthResponse(
                status="error",
                version="1.0.0",
                uptime_seconds=0,
                stores={},
                memory_count=0,
                processing_stats={},
            )
