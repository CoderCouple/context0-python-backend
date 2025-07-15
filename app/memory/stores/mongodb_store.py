import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel

from app.memory.models.memory_entry import MemoryEntry
from app.memory.stores.base_store import DocumentStore


class MongoDocumentStore(DocumentStore):
    """MongoDB implementation of DocumentStore"""

    def __init__(
        self,
        connection_string: str,
        database_name: str = "memory_system",
        collection_name: str = "memories",
    ):
        """Initialize MongoDB connection"""
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None

    async def initialize(self):
        """Initialize MongoDB connection and setup indexes"""
        # Add timeout to prevent long hangs
        self.client = AsyncIOMotorClient(
            self.connection_string,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
        )
        self.db = self.client[self.database_name]
        self.collection = self.db[self.collection_name]

        # Create indexes for performance
        await self._create_indexes()

        # Test connection with timeout
        await asyncio.wait_for(self.client.admin.command("ping"), timeout=5.0)

    async def _create_indexes(self):
        """Create MongoDB indexes for efficient querying"""
        indexes = [
            IndexModel([("id", 1)], unique=True),  # Unique memory ID
            IndexModel([("cid", 1)]),  # Content ID for deduplication
            IndexModel([("source_user_id", 1)]),  # User-scoped queries
            IndexModel([("source_session_id", 1)]),  # Session-scoped queries
            IndexModel([("memory_type", 1)]),  # Filter by memory type
            IndexModel([("scope", 1)]),  # Scope-based queries
            IndexModel([("created_at", -1)]),  # Chronological ordering
            IndexModel([("updated_at", -1)]),  # Recently updated
            IndexModel([("tags", 1)]),  # Tag-based queries
            IndexModel([("source_user_id", 1), ("memory_type", 1)]),  # Compound index
            IndexModel([("source_user_id", 1), ("created_at", -1)]),  # User timeline
            IndexModel([("permissions.owner_id", 1)]),  # Permission queries
            # Text index for full-text search
            IndexModel([("input", "text"), ("summary", "text")], name="text_search"),
            # Geospatial index if location data is stored
            # IndexModel([("location", "2dsphere")]) - uncomment if using geospatial data
        ]

        try:
            await self.collection.create_indexes(indexes)
        except Exception as e:
            # Indexes might already exist
            pass

    def _memory_entry_to_dict(self, entry: MemoryEntry) -> Dict[str, Any]:
        """Convert MemoryEntry to MongoDB document"""
        doc = {
            "id": entry.id,
            "cid": entry.cid,
            "scope": entry.scope,
            "input": entry.input,
            "summary": entry.summary,
            "memory_type": entry.memory_type.value,
            # Permissions
            "permissions": {
                "owner_id": entry.permissions.owner_id,
                "read_access": entry.permissions.read_access,
                "write_access": entry.permissions.write_access,
                "is_public": entry.permissions.is_public,
            },
            # Vector data
            "embedding": entry.embedding,
            "embedding_model": entry.embedding_model,
            # Graph relationships
            "graph_links": [
                {
                    "target_id": link.target_id,
                    "relationship_type": link.relationship_type,
                    "properties": link.properties,
                }
                for link in entry.graph_links
            ],
            # Metadata
            "meta": {
                "classifier_used": entry.meta.classifier_used,
                "llm_version": entry.meta.llm_version,
                "handler_version": entry.meta.handler_version,
                "processing_time_ms": entry.meta.processing_time_ms,
                "confidence_score": entry.meta.confidence_score,
            },
            # Tags and custom metadata
            "tags": entry.tags,
            "custom_metadata": entry.custom_metadata,
            # Timestamps
            "created_at": entry.created_at,
            "updated_at": entry.updated_at,
            "accessed_at": entry.accessed_at,
            # Source tracking
            "source_session_id": entry.source_session_id,
            "source_user_id": entry.source_user_id,
        }

        return doc

    def _dict_to_memory_entry(self, doc: Dict[str, Any]) -> MemoryEntry:
        """Convert MongoDB document to MemoryEntry"""
        from app.common.enum.memory import MemoryType
        from app.memory.models.memory_entry import (
            GraphLink,
            MemoryMeta,
            MemoryPermissions,
        )

        return MemoryEntry(
            id=doc["id"],
            cid=doc["cid"],
            scope=doc["scope"],
            input=doc["input"],
            summary=doc.get("summary"),
            memory_type=MemoryType(doc["memory_type"]),
            permissions=MemoryPermissions(**doc["permissions"]),
            embedding=doc.get("embedding"),
            embedding_model=doc.get("embedding_model"),
            graph_links=[GraphLink(**link) for link in doc.get("graph_links", [])],
            meta=MemoryMeta(**doc["meta"]),
            tags=doc.get("tags", []),
            custom_metadata=doc.get("custom_metadata", {}),
            created_at=doc["created_at"],
            updated_at=doc["updated_at"],
            accessed_at=doc.get("accessed_at"),
            source_session_id=doc["source_session_id"],
            source_user_id=doc["source_user_id"],
        )

    async def create(self, entry: MemoryEntry) -> str:
        """Create a new memory entry"""
        doc = self._memory_entry_to_dict(entry)
        result = await self.collection.insert_one(doc)
        return entry.id

    async def add_memory(self, memory_dict: Dict[str, Any]) -> bool:
        """Add memory from dictionary format (used by memory engine)"""
        try:
            # Convert the memory dict to the format expected by MongoDB
            doc = memory_dict.copy()

            # Ensure memory_type is a string if it's an enum
            if hasattr(doc.get("memory_type"), "value"):
                doc["memory_type"] = doc["memory_type"].value

            # Convert datetime objects to MongoDB format
            if "created_at" in doc and hasattr(doc["created_at"], "isoformat"):
                doc["created_at"] = doc["created_at"]
            if "updated_at" in doc and hasattr(doc["updated_at"], "isoformat"):
                doc["updated_at"] = doc["updated_at"]

            # Insert the document
            result = await self.collection.insert_one(doc)
            return result.inserted_id is not None

        except Exception as e:
            print(f"Error adding memory to MongoDB: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if MongoDB store is healthy"""
        try:
            if not self.client:
                return False

            # Test connection by pinging the database
            await self.client.admin.command("ping")
            return True

        except Exception as e:
            print(f"MongoDB health check failed: {e}")
            return False

    async def read(self, memory_id: str) -> Optional[MemoryEntry]:
        """Read a memory entry by ID"""
        doc = await self.collection.find_one({"id": memory_id})
        if doc:
            # Update accessed_at timestamp
            await self.collection.update_one(
                {"id": memory_id}, {"$set": {"accessed_at": datetime.utcnow()}}
            )
            doc["accessed_at"] = datetime.utcnow()
            return self._dict_to_memory_entry(doc)
        return None

    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry"""
        # Add updated timestamp
        update_doc = {**updates, "updated_at": datetime.utcnow()}

        result = await self.collection.update_one(
            {"id": memory_id}, {"$set": update_doc}
        )

        return result.modified_count > 0

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry"""
        result = await self.collection.delete_one({"id": memory_id})
        return result.deleted_count > 0

    async def search(self, query: Dict[str, Any], limit: int = 10) -> List[MemoryEntry]:
        """Search for memory entries using MongoDB query syntax"""
        cursor = self.collection.find(query).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [self._dict_to_memory_entry(doc) for doc in docs]

    async def close(self):
        """Clean shutdown of MongoDB store"""
        if self.client:
            self.client.close()
            print("✅ MongoDB store closed")

    async def query(
        self,
        filter: Dict[str, Any],
        projection: Optional[Dict[str, int]] = None,
        sort: Optional[Dict[str, int]] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Query documents with MongoDB-style syntax"""
        cursor = self.collection.find(filter, projection)

        if sort:
            cursor = cursor.sort(list(sort.items()))

        cursor = cursor.limit(limit)
        docs = await cursor.to_list(length=limit)

        if projection:
            # If projection is used, return raw docs since they might not be complete MemoryEntry objects
            return docs
        else:
            return [self._dict_to_memory_entry(doc) for doc in docs]

    async def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run aggregation pipeline"""
        cursor = self.collection.aggregate(pipeline)
        return await cursor.to_list(length=None)

    async def create_index(
        self, fields: List[str], options: Optional[Dict[str, Any]] = None
    ):
        """Create index on fields"""
        index_spec = [(field, 1) for field in fields]
        await self.collection.create_index(index_spec, **(options or {}))

    async def full_text_search(
        self, query: str, user_id: Optional[str] = None, limit: int = 10
    ) -> List[MemoryEntry]:
        """Perform full-text search"""
        filter_query = {"$text": {"$search": query}}

        if user_id:
            filter_query["source_user_id"] = user_id

        cursor = (
            self.collection.find(filter_query, {"score": {"$meta": "textScore"}})
            .sort([("score", {"$meta": "textScore"})])
            .limit(limit)
        )

        docs = await cursor.to_list(length=limit)
        return [self._dict_to_memory_entry(doc) for doc in docs]

    async def get_user_memories(
        self,
        user_id: str,
        memory_types: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50,
    ) -> List[MemoryEntry]:
        """Get memories for a specific user with filters"""
        filter_query = {"source_user_id": user_id}

        if memory_types:
            filter_query["memory_type"] = {"$in": memory_types}

        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter["$gte"] = start_date
            if end_date:
                date_filter["$lte"] = end_date
            filter_query["created_at"] = date_filter

        cursor = self.collection.find(filter_query).sort("created_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [self._dict_to_memory_entry(doc) for doc in docs]

    async def get_similar_memories(
        self, cid: str, user_id: str, limit: int = 5
    ) -> List[MemoryEntry]:
        """Find memories with similar content ID (for deduplication)"""
        filter_query = {"cid": cid, "source_user_id": user_id}

        cursor = self.collection.find(filter_query).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [self._dict_to_memory_entry(doc) for doc in docs]

    async def get_memory_stats(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Get statistics about memories"""
        match_stage = {}
        if user_id:
            match_stage = {"$match": {"source_user_id": user_id}}

        pipeline = []
        if match_stage:
            pipeline.append(match_stage)

        pipeline.extend(
            [
                {
                    "$group": {
                        "_id": "$memory_type",
                        "count": {"$sum": 1},
                        "avg_confidence": {"$avg": "$meta.confidence_score"},
                        "latest": {"$max": "$created_at"},
                        "earliest": {"$min": "$created_at"},
                    }
                },
                {
                    "$group": {
                        "_id": None,
                        "by_type": {
                            "$push": {
                                "memory_type": "$_id",
                                "count": "$count",
                                "avg_confidence": "$avg_confidence",
                                "latest": "$latest",
                                "earliest": "$earliest",
                            }
                        },
                        "total_memories": {"$sum": "$count"},
                        "overall_avg_confidence": {"$avg": "$avg_confidence"},
                    }
                },
            ]
        )

        result = await self.aggregate(pipeline)
        if result:
            return result[0]
        return {"total_memories": 0, "by_type": [], "overall_avg_confidence": 0}

    async def bulk_insert(self, entries: List[MemoryEntry]) -> List[str]:
        """Bulk insert multiple memory entries"""
        docs = [self._memory_entry_to_dict(entry) for entry in entries]
        result = await self.collection.insert_many(docs)
        return [entry.id for entry in entries]

    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
