import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
from app.api.v1.request.memory_request import MemoryRecordInput, SearchQuery
from app.api.v1.response.memory_response import (
    MemoryResponse,
    SearchResponse,
    SearchResult,
)
from app.common.enum.memory import MemoryOperation, MemoryType
from app.memory.config.context_zero_config import ContextZeroConfig
from app.memory.inferencer.memory_type_inferencer import MemoryTypeInferencer
from app.memory.models.audit_log import AuditLogEntry
from app.memory.models.memory_entry import MemoryEntry
from app.memory.models.memory_record import MemoryRecord
from app.memory.router.memory_router import MemoryRouter
from app.memory.stores.base_store import (
    AuditStore,
    DocumentStore,
    GraphStore,
    TimeSeriesStore,
    VectorStore,
)


class MemoryEngine:
    """Main memory processing engine that orchestrates the entire pipeline"""

    _instance = None

    def __init__(self, config: Optional[ContextZeroConfig] = None):
        """Initialize the memory engine with configuration"""
        self.config = config or ContextZeroConfig()
        self.inferencer = MemoryTypeInferencer(self.config.llm)
        self.router = MemoryRouter(self.inferencer)

        # Stores will be initialized in initialize()
        self.vector_store: Optional[VectorStore] = None
        self.graph_store: Optional[GraphStore] = None
        self.doc_store: Optional[DocumentStore] = None
        self.timeseries_store: Optional[TimeSeriesStore] = None
        self.audit_store: Optional[AuditStore] = None

        # Processing statistics
        self.processing_stats = {
            "total_memories": 0,
            "memories_by_type": {},
            "average_processing_time": 0,
            "last_processed": None,
        }

        self._initialized = False
        MemoryEngine._instance = self

    @classmethod
    def get_instance(cls) -> "MemoryEngine":
        """Get singleton instance of memory engine"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def initialize(self):
        """Initialize all stores and components"""
        if self._initialized:
            return

        print("Initializing Memory Engine...")

        # Load configuration from YAML files
        from app.memory.config.yaml_config_loader import get_config_loader

        config_loader = get_config_loader()

        try:
            # Initialize Vector Store (Pinecone)
            vector_config = config_loader.get_vector_store_config()
            print(f"📊 Vector Store Config: {vector_config}")
            if vector_config.get("provider") == "pinecone":
                await self._initialize_pinecone(vector_config)

            # Initialize Graph Store (Neo4j)
            graph_config = config_loader.get_graph_store_config()
            print(f"📊 Graph Store Config: {graph_config}")
            if graph_config.get("provider") == "neo4j":
                await self._initialize_neo4j(graph_config)

            # Initialize Document Store (MongoDB)
            doc_config = config_loader.get_doc_store_config()
            print(f"📊 Document Store Config: {doc_config}")
            if doc_config.get("provider") == "mongodb":
                await self._initialize_mongodb(doc_config)

            # Initialize Time Series Store (TimescaleDB)
            try:
                time_config = config_loader.get_time_store_config()
                print(f"📊 Time Series Store Config: {time_config}")
                if time_config and time_config.get("provider") == "timescaledb":
                    await self._initialize_timescaledb(time_config)
            except Exception as e:
                print(f"⚠️ TimescaleDB not configured: {e}")
                self.timeseries_store = None

            # Initialize Audit Store (MongoDB)
            audit_config = config_loader.get_audit_store_config()
            print(f"📊 Audit Store Config: {audit_config}")
            if audit_config.get("provider") == "mongodb":
                await self._initialize_audit_store(audit_config)

            # Print LLM and Embedder configs
            try:
                llm_config = config_loader.get_llm_config()
                print(f"📊 LLM Config: {llm_config}")
            except Exception as e:
                print(f"⚠️ LLM config error: {e}")

            try:
                embedder_config = config_loader.get_embedder_config()
                print(f"📊 Embedder Config: {embedder_config}")
            except Exception as e:
                print(f"⚠️ Embedder config error: {e}")

            try:
                system_config = config_loader.get_system_config()
                print(f"📊 System Config: {system_config}")
            except Exception as e:
                print(f"⚠️ System config error: {e}")

            self._initialized = True
            print("Memory Engine initialized successfully")

        except Exception as e:
            print(f"❌ Error initializing Memory Engine: {e}")
            import traceback

            traceback.print_exc()
            # Continue with in-memory stores as fallback

    async def _initialize_pinecone(self, config: dict):
        """Initialize Pinecone vector store"""
        try:
            from app.memory.stores.pinecone_store import PineconeVectorStore

            store_config = config.get("config", {})

            self.vector_store = PineconeVectorStore(
                api_key=store_config.get("api_key"),
                environment=store_config.get("environment", "us-east-1-aws"),
                index_name=store_config.get("index_name", "memory-index"),
            )
            await self.vector_store.initialize()
            print("✅ Pinecone vector store initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Pinecone: {e}")
            print(
                "   Continuing without vector store - will use document store fallback"
            )
            self.vector_store = None

    async def _initialize_neo4j(self, config: dict):
        """Initialize Neo4j graph store"""
        try:
            from app.memory.stores.neo4j_store import Neo4jGraphStore

            store_config = config.get("config", {})

            self.graph_store = Neo4jGraphStore(
                uri=store_config.get("uri"),
                username=store_config.get("username"),
                password=store_config.get("password"),
                database=store_config.get("database", "neo4j"),
            )
            await self.graph_store.initialize()
            print("✅ Neo4j graph store initialized")
        except Exception as e:
            print(f"❌ Failed to initialize Neo4j: {e}")

    async def _initialize_mongodb(self, config: dict):
        """Initialize MongoDB document store"""
        try:
            from app.memory.stores.mongodb_store import MongoDocumentStore

            store_config = config.get("config", {})

            self.doc_store = MongoDocumentStore(
                connection_string=store_config.get("connection_string"),
                database_name=store_config.get("database_name", "memory_system"),
                collection_name=store_config.get("collection_name", "memories"),
            )
            await self.doc_store.initialize()
            print("✅ MongoDB document store initialized")
        except Exception as e:
            print(f"❌ Failed to initialize MongoDB document store: {e}")

    async def _initialize_timescaledb(self, config: dict):
        """Initialize TimescaleDB time series store"""
        try:
            from app.memory.stores.timescale_store import TimescaleTimeSeriesStore

            store_config = config.get("config", {})

            self.timeseries_store = TimescaleTimeSeriesStore(
                connection_string=store_config.get("connection_string"),
                table_name=store_config.get("table_name", "memory_timeseries"),
            )
            await self.timeseries_store.initialize()
            print("✅ TimescaleDB time series store initialized")
        except Exception as e:
            print(f"❌ Failed to initialize TimescaleDB: {e}")

    async def _initialize_audit_store(self, config: dict):
        """Initialize MongoDB audit store"""
        try:
            from app.memory.stores.mongodb_audit_store import MongoAuditStore

            store_config = config.get("config", {})

            self.audit_store = MongoAuditStore(
                connection_string=store_config.get("connection_string"),
                database_name=store_config.get("database_name", "memory_audit"),
                collection_name=store_config.get("collection_name", "audit_log"),
            )
            await self.audit_store.initialize()
            print("✅ MongoDB audit store initialized")
        except Exception as e:
            print(f"❌ Failed to initialize MongoDB audit store: {e}")

    # ==========================================
    # CORE MEMORY OPERATIONS
    # ==========================================

    async def add_memory(self, record_input: MemoryRecordInput) -> MemoryResponse:
        """Add a new memory to the system"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # Convert input to MemoryRecord
            record = MemoryRecord(
                user_id=record_input.user_id,
                session_id=record_input.session_id,
                raw_text=record_input.text,
                memory_type=record_input.memory_type,
                tags=record_input.tags,
                metadata=record_input.metadata,
                scope=record_input.scope or "default",
            )

            # Process through the pipeline
            memory_entry = await self.process_memory(record)

            # Store in all configured stores
            await self._store_memory(memory_entry)

            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(memory_entry.memory_type, processing_time)

            return MemoryResponse(
                success=True,
                memory_id=memory_entry.id,
                operation=MemoryOperation.ADD,
                confidence=memory_entry.meta.confidence_score or 1.0,
                processing_time_ms=int(processing_time * 1000),
                message="Memory processed and stored successfully",
                memory_type=memory_entry.memory_type,
            )

        except Exception as e:
            return MemoryResponse(
                success=False,
                memory_id="",
                operation=MemoryOperation.ADD,
                confidence=0.0,
                processing_time_ms=int((time.time() - start_time) * 1000),
                message=f"Failed to process memory: {str(e)}",
            )

    async def process_memory(self, record: MemoryRecord) -> MemoryEntry:
        """Process a single memory through the entire pipeline"""
        # Route to appropriate handler
        handler = await self.router.route(record)

        # Process the memory
        memory_entry = await handler.process(record)

        return memory_entry

    async def search_memories(self, query: SearchQuery) -> SearchResponse:
        """Search for memories using semantic similarity"""
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        try:
            results = []

            # Use vector store for semantic search if available
            if self.vector_store:
                # Generate query embedding
                try:
                    handler = self.router.get_handler(MemoryType.SEMANTIC_MEMORY)
                    if handler is None:
                        raise ValueError("Semantic memory handler not found")
                    query_embedding, _ = await handler.extract_embedding(query.query)
                except Exception as e:
                    logger.error(f"Error generating query embedding: {e}")
                    # Fall back to document store search
                    handler = None

                if handler is not None:
                    # Search vector store using similarity search
                    search_filter = {"user_id": query.user_id}
                    if query.memory_types:
                        search_filter["memory_type"] = {
                            "$in": [mt.value for mt in query.memory_types]
                        }
                    if query.tags:
                        search_filter["tags"] = {"$in": query.tags}

                    vector_results = await self.vector_store.similarity_search(
                        embedding=query_embedding,
                        limit=query.limit,
                        filter=search_filter,
                    )

                    # Convert to SearchResult objects
                    for memory_entry, score in vector_results:
                        # Filter by threshold
                        if score >= query.threshold:
                            try:
                                # Handle both dict and MemoryEntry formats
                                if isinstance(memory_entry, dict):
                                    # Pinecone returns dict format
                                    metadata = memory_entry.get("metadata", {})

                                    # Parse memory type - handle both string formats
                                    memory_type_str = metadata.get(
                                        "memory_type", "semantic_memory"
                                    )
                                    if "." in memory_type_str:
                                        # Format: "MemoryType.EMOTIONAL_MEMORY"
                                        memory_type_value = memory_type_str.split(".")[
                                            -1
                                        ].lower()
                                    else:
                                        memory_type_value = memory_type_str

                                    try:
                                        memory_type = MemoryType(memory_type_value)
                                    except ValueError:
                                        memory_type = MemoryType.SEMANTIC_MEMORY

                                    # Parse created_at
                                    created_at_str = metadata.get("created_at")
                                    if isinstance(created_at_str, str):
                                        try:
                                            created_at = datetime.fromisoformat(
                                                created_at_str.replace("Z", "+00:00")
                                            )
                                        except:
                                            created_at = datetime.now()
                                    else:
                                        created_at = created_at_str or datetime.now()

                                    # Parse tags - handle both string and list formats
                                    tags_data = metadata.get("tags", [])
                                    if isinstance(tags_data, str):
                                        tags = tags_data.split(",") if tags_data else []
                                        tags = [
                                            tag.strip() for tag in tags if tag.strip()
                                        ]
                                    elif isinstance(tags_data, list):
                                        tags = tags_data
                                    else:
                                        tags = []

                                    # Parse confidence
                                    confidence_data = metadata.get("confidence", 0.8)
                                    if isinstance(confidence_data, str):
                                        try:
                                            confidence = float(confidence_data)
                                        except:
                                            confidence = 0.8
                                    else:
                                        confidence = confidence_data

                                    search_result = SearchResult(
                                        id=memory_entry["id"],
                                        content=metadata.get("input")
                                        if query.include_content
                                        else None,
                                        summary=metadata.get("summary", ""),
                                        memory_type=memory_type,
                                        score=score,
                                        confidence=confidence,
                                        tags=tags,
                                        created_at=created_at,
                                        scope=metadata.get("scope"),
                                    )
                                else:
                                    # MemoryEntry object format
                                    search_result = SearchResult(
                                        id=memory_entry.id,
                                        content=memory_entry.input
                                        if query.include_content
                                        else None,
                                        summary=memory_entry.summary or "",
                                        memory_type=memory_entry.memory_type,
                                        score=score,
                                        confidence=memory_entry.confidence
                                        if hasattr(memory_entry, "confidence")
                                        else 0.8,
                                        tags=memory_entry.tags,
                                        created_at=memory_entry.created_at,
                                        scope=memory_entry.scope,
                                    )
                                results.append(search_result)
                            except Exception as e:
                                logger.error(
                                    f"Error processing search result {memory_entry}: {e}"
                                )
                                continue

            # Fallback to document store text search
            elif self.doc_store:
                doc_results = await self.doc_store.search_memories(
                    user_id=query.user_id,
                    query_text=query.query,
                    memory_types=query.memory_types,
                    tags=query.tags,
                    limit=query.limit,
                )

                for result in doc_results:
                    search_result = SearchResult(
                        id=result["id"],
                        summary=result.get("summary", ""),
                        memory_type=MemoryType(result["memory_type"]),
                        score=0.8,  # Default score for text search
                        tags=result.get("tags", []),
                        created_at=result["created_at"],
                        confidence=result.get("confidence", 0.0),
                        content=result.get("input") if query.include_content else None,
                        scope=result.get("scope"),
                    )
                    results.append(search_result)

            return SearchResponse(
                success=True,
                results=results,
                total_count=len(results),
                query_time_ms=int((time.time() - start_time) * 1000),
                query=query.query,
                filters_applied={
                    "memory_types": query.memory_types,
                    "tags": query.tags,
                    "threshold": query.threshold,
                },
            )

        except Exception as e:
            return SearchResponse(
                success=False,
                results=[],
                total_count=0,
                query_time_ms=int((time.time() - start_time) * 1000),
                query=query.query,
                filters_applied={},
            )

    async def add_memories_bulk(
        self, records: List[MemoryRecordInput]
    ) -> List[Dict[str, Any]]:
        """Bulk add memories with batch processing"""
        if not self._initialized:
            await self.initialize()

        results = []

        for record_input in records:
            try:
                response = await self.add_memory(record_input)
                results.append(
                    {
                        "success": response.success,
                        "memory_id": response.memory_id,
                        "user_id": record_input.user_id,
                        "message": response.message,
                        "processing_time_ms": response.processing_time_ms,
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "success": False,
                        "memory_id": "",
                        "user_id": record_input.user_id,
                        "message": f"Failed to process: {str(e)}",
                        "processing_time_ms": 0,
                    }
                )

        return results

    # ==========================================
    # STORAGE OPERATIONS
    # ==========================================

    async def _store_memory(self, memory_entry: MemoryEntry):
        """Store memory in all configured stores"""
        tasks = []

        # Convert to dict for storage
        memory_dict = memory_entry.dict()
        memory_dict["created_at"] = datetime.utcnow()
        memory_dict["updated_at"] = datetime.utcnow()

        # Store in vector store
        if self.vector_store:
            tasks.append(self.vector_store.add_memory(memory_dict))

        # Store in graph store
        if self.graph_store:
            tasks.append(self.graph_store.add_memory(memory_dict))

        # Store in document store
        if self.doc_store:
            tasks.append(self.doc_store.add_memory(memory_dict))

        # Store in time series store
        if self.timeseries_store:
            tasks.append(self.timeseries_store.add_memory(memory_dict))

        # Store in audit store
        if self.audit_store:
            audit_entry = AuditLogEntry(
                id=f"audit_{memory_entry.id}_{int(datetime.utcnow().timestamp())}",
                action=MemoryOperation.ADD,
                memory_id=memory_entry.id,
                cid=memory_entry.cid,
                user_id=memory_entry.source_user_id or "",
                session_id=memory_entry.source_session_id or "",
                timestamp=datetime.utcnow(),
                after_state=memory_dict,
                inferred_type=memory_entry.memory_type,
                handler_used=memory_entry.meta.handler_version,
                processing_time_ms=memory_entry.meta.processing_time_ms,
            )
            tasks.append(self.audit_store.log_operation(audit_entry))

        # Execute all storage operations in parallel
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Check if at least one store succeeded
            successes = sum(1 for r in results if not isinstance(r, Exception))
            if successes == 0:
                # Log available stores for debugging
                available_stores = []
                if self.vector_store:
                    available_stores.append("vector")
                if self.graph_store:
                    available_stores.append("graph")
                if self.doc_store:
                    available_stores.append("document")
                if self.timeseries_store:
                    available_stores.append("timeseries")
                if self.audit_store:
                    available_stores.append("audit")

                print(f"⚠️  Warning: All stores failed for memory {memory_entry.id}")
                print(f"   Available stores: {available_stores}")
                print(f"   Continuing anyway to prevent blocking...")
            else:
                print(f"✅ Memory stored in {successes}/{len(tasks)} stores")

    # ==========================================
    # UTILITY METHODS
    # ==========================================

    def _update_stats(self, memory_type: MemoryType, processing_time: float):
        """Update processing statistics"""
        self.processing_stats["total_memories"] += 1
        self.processing_stats["last_processed"] = datetime.utcnow().isoformat()

        # Update memory type counts
        type_name = memory_type.value
        if type_name not in self.processing_stats["memories_by_type"]:
            self.processing_stats["memories_by_type"][type_name] = 0
        self.processing_stats["memories_by_type"][type_name] += 1

        # Update average processing time
        current_avg = self.processing_stats["average_processing_time"]
        total = self.processing_stats["total_memories"]
        self.processing_stats["average_processing_time"] = (
            current_avg * (total - 1) + processing_time
        ) / total

    def _calculate_similarity(
        self, embedding1: List[float], embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        import numpy as np

        arr1 = np.array(embedding1)
        arr2 = np.array(embedding2)

        # Cosine similarity
        dot_product = np.dot(arr1, arr2)
        norms = np.linalg.norm(arr1) * np.linalg.norm(arr2)

        if norms == 0:
            return 0.0

        return float(dot_product / norms)

    async def health_check(self) -> Dict[str, Any]:
        """Check health of all components"""
        health = {"engine": True, "initialized": self._initialized, "stores": {}}

        # Check store health
        stores = {
            "vector_store": self.vector_store,
            "graph_store": self.graph_store,
            "doc_store": self.doc_store,
            "timeseries_store": self.timeseries_store,
            "audit_store": self.audit_store,
        }

        for store_name, store in stores.items():
            if store:
                try:
                    store_health = await store.health_check()
                    health["stores"][store_name] = store_health
                except Exception:
                    health["stores"][store_name] = False
            else:
                health["stores"][store_name] = None

        return health

    # ==========================================
    # TIME-TRAVEL DEBUGGING METHODS
    # ==========================================

    async def time_travel_query(
        self, user_id: str, query: str, target_time: datetime
    ) -> Dict[str, Any]:
        """Answer a query based on memory state at a specific point in time"""
        if not self._initialized:
            await self.initialize()

        if not self.audit_store:
            raise ValueError("Audit store not configured - time travel not available")

        # Get memory state at target time
        memory_state = await self.audit_store.get_memory_state_at_time(
            user_id, target_time
        )

        # Generate embedding for query using current handlers
        handler = self.router.get_handler(MemoryType.SEMANTIC_MEMORY)
        query_embedding, _ = await handler.extract_embedding(query)

        # Find relevant memories from the historical state
        relevant_memories = []
        for memory in memory_state:
            if memory.get("embedding"):
                similarity = self._calculate_similarity(
                    query_embedding, memory["embedding"]
                )
                if similarity > 0.7:  # Threshold
                    memory["similarity_score"] = similarity
                    relevant_memories.append(memory)

        # Sort by relevance
        relevant_memories.sort(key=lambda x: x["similarity_score"], reverse=True)

        return {
            "success": True,
            "target_time": target_time,
            "query": query,
            "total_memories_at_time": len(memory_state),
            "relevant_memories": relevant_memories[:10],  # Top 10
            "context": f"Based on memory state at {target_time}",
        }

    async def close(self):
        """Clean shutdown of memory eng
        ine and all stores"""
        print("Closing Memory Engine...")

        # Close all stores
        stores = [
            ("vector_store", self.vector_store),
            ("graph_store", self.graph_store),
            ("doc_store", self.doc_store),
            ("timeseries_store", self.timeseries_store),
            ("audit_store", self.audit_store),
        ]

        for store_name, store in stores:
            if store:
                try:
                    if hasattr(store, "close"):
                        await store.close()
                        print(f"✅ {store_name} closed successfully")
                    elif hasattr(store, "disconnect"):
                        await store.disconnect()
                        print(f"✅ {store_name} disconnected successfully")
                    else:
                        print(f"ℹ️ {store_name} has no close/disconnect method")
                except Exception as e:
                    print(f"⚠️ Error closing {store_name}: {e}")

        # Close thread pool executor if it exists
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
            print("✅ Thread pool executor shutdown")

        print("Memory Engine closed successfully")
