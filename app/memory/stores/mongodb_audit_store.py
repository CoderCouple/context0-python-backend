import asyncio
import json
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import IndexModel

from app.common.enum.memory import MemoryOperation, MemoryType
from app.memory.models.audit_log import AuditLogEntry
from app.memory.models.memory_entry import MemoryEntry
from app.memory.stores.base_store import AuditStore


class MongoAuditStore(AuditStore):
    """MongoDB implementation of AuditStore optimized for time-travel debugging"""

    def __init__(
        self,
        connection_string: str,
        database_name: str = "memory_audit",
        collection_name: str = "audit_log",
    ):
        """Initialize MongoDB connection for audit logging"""
        self.connection_string = connection_string
        self.database_name = database_name
        self.collection_name = collection_name
        self.client = None
        self.db = None
        self.collection = None

    async def initialize(self):
        """Initialize MongoDB connection and setup audit indexes"""
        # Add timeout to prevent long hangs
        self.client = AsyncIOMotorClient(
            self.connection_string,
            serverSelectionTimeoutMS=5000,  # 5 second timeout
            connectTimeoutMS=5000,
            socketTimeoutMS=5000,
        )
        self.db = self.client[self.database_name]
        self.collection = self.db[self.collection_name]

        # Create indexes optimized for time-travel queries
        await self._create_indexes()

        # Test connection with timeout
        await asyncio.wait_for(self.client.admin.command("ping"), timeout=5.0)

    async def _create_indexes(self):
        """Create MongoDB indexes optimized for debugging and time-travel"""
        indexes = [
            # Core audit indexes
            IndexModel([("id", 1)], unique=True),
            IndexModel([("memory_id", 1), ("timestamp", -1)]),  # Memory history
            IndexModel([("user_id", 1), ("timestamp", -1)]),  # User timeline
            IndexModel([("session_id", 1), ("timestamp", -1)]),  # Session timeline
            IndexModel([("action", 1), ("timestamp", -1)]),  # Operation timeline
            # Time-travel specific indexes
            IndexModel([("user_id", 1), ("timestamp", 1)]),  # Forward time-travel
            IndexModel(
                [("user_id", 1), ("action", 1), ("timestamp", -1)]
            ),  # User operations
            IndexModel([("cid", 1), ("timestamp", -1)]),  # Content deduplication
            # Debugging indexes
            IndexModel([("error_type", 1), ("timestamp", -1)]),  # Error debugging
            IndexModel([("handler_used", 1), ("timestamp", -1)]),  # Handler performance
            IndexModel(
                [("inferred_type", 1), ("timestamp", -1)]
            ),  # Type inference analysis
            # Performance analysis
            IndexModel([("processing_time_ms", -1)]),  # Slow operations
            IndexModel([("timestamp", -1)]),  # Recent operations
            # Rich querying for debugging
            IndexModel([("debug_context.handler_trace.step", 1)]),  # Handler debugging
            IndexModel([("changes", 1)], sparse=True),  # Change analysis
        ]

        try:
            await self.collection.create_indexes(indexes)
        except Exception as e:
            # Indexes might already exist
            pass

    def _create_rich_audit_document(
        self, audit_entry: AuditLogEntry, memory_context: Optional[MemoryEntry] = None
    ) -> Dict[str, Any]:
        """Create a rich audit document optimized for debugging"""
        doc = {
            # Core audit fields
            "id": audit_entry.id,
            "action": audit_entry.action.value,
            "memory_id": audit_entry.memory_id,
            "cid": audit_entry.cid,
            "user_id": audit_entry.user_id,
            "session_id": audit_entry.session_id,
            "timestamp": audit_entry.timestamp,
            # Operation details
            "inferred_type": audit_entry.inferred_type.value
            if audit_entry.inferred_type
            else None,
            "handler_used": audit_entry.handler_used,
            # Performance metrics
            "processing_time_ms": audit_entry.processing_time_ms,
            # Change tracking (rich documents for debugging)
            "before_state": audit_entry.before_state,
            "after_state": audit_entry.after_state,
            "changes": audit_entry.changes,
            # Request context
            "ip_address": audit_entry.ip_address,
            "user_agent": audit_entry.user_agent,
            "api_version": audit_entry.api_version,
            # Error tracking
            "error": audit_entry.error,
            "error_type": audit_entry.error_type,
            # Debugging context (enhanced)
            "debug_context": {
                "operation_id": audit_entry.id,
                "trace_id": f"{audit_entry.session_id}_{audit_entry.timestamp.isoformat()}",
                "memory_snapshot": memory_context.dict() if memory_context else None,
                "timestamp_iso": audit_entry.timestamp.isoformat(),
                "timestamp_unix": audit_entry.timestamp.timestamp(),
            },
            # Time-travel helpers
            "date_bucket": audit_entry.timestamp.strftime("%Y-%m-%d"),
            "hour_bucket": audit_entry.timestamp.strftime("%Y-%m-%d-%H"),
            "minute_bucket": audit_entry.timestamp.strftime("%Y-%m-%d-%H-%M"),
        }

        return doc

    def _dict_to_audit_entry(self, doc: Dict[str, Any]) -> AuditLogEntry:
        """Convert MongoDB document back to AuditLogEntry"""
        return AuditLogEntry(
            id=doc["id"],
            action=MemoryOperation(doc["action"]),
            memory_id=doc["memory_id"],
            cid=doc["cid"],
            user_id=doc["user_id"],
            session_id=doc["session_id"],
            timestamp=doc["timestamp"],
            inferred_type=MemoryType(doc["inferred_type"])
            if doc.get("inferred_type")
            else None,
            handler_used=doc.get("handler_used"),
            before_state=doc.get("before_state"),
            after_state=doc.get("after_state"),
            changes=doc.get("changes"),
            processing_time_ms=doc.get("processing_time_ms"),
            ip_address=doc.get("ip_address"),
            user_agent=doc.get("user_agent"),
            api_version=doc.get("api_version"),
            error=doc.get("error"),
            error_type=doc.get("error_type"),
        )

    async def log_operation(
        self, audit_entry: AuditLogEntry, memory_context: Optional[MemoryEntry] = None
    ) -> str:
        """Log a memory operation with rich debugging context"""
        doc = self._create_rich_audit_document(audit_entry, memory_context)

        result = await self.collection.insert_one(doc)
        return audit_entry.id

    async def get_audit_trail(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[AuditLogEntry]:
        """Get audit trail for memory operations"""
        filter_query = {}

        if memory_id:
            filter_query["memory_id"] = memory_id

        if user_id:
            filter_query["user_id"] = user_id

        if start_time or end_time:
            time_filter = {}
            if start_time:
                time_filter["$gte"] = datetime.fromisoformat(
                    start_time.replace("Z", "+00:00")
                )
            if end_time:
                time_filter["$lte"] = datetime.fromisoformat(
                    end_time.replace("Z", "+00:00")
                )
            filter_query["timestamp"] = time_filter

        cursor = self.collection.find(filter_query).sort("timestamp", -1).limit(1000)
        docs = await cursor.to_list(length=1000)

        return [self._dict_to_audit_entry(doc) for doc in docs]

    async def health_check(self) -> bool:
        """Check if MongoDB audit store is healthy"""
        try:
            if not self.client:
                return False

            # Test connection by pinging the database
            await self.client.admin.command("ping")
            return True

        except Exception as e:
            print(f"MongoDB audit store health check failed: {e}")
            return False

    async def close(self):
        """Clean shutdown of MongoDB audit store"""
        if self.client:
            self.client.close()
            print("✅ MongoDB audit store closed")

    # ==========================================
    # TIME-TRAVEL DEBUGGING METHODS
    # ==========================================

    async def get_memory_state_at_time(
        self, user_id: str, target_time: datetime
    ) -> List[Dict[str, Any]]:
        """Get all memories that existed at a specific point in time"""
        pipeline = [
            # Get all operations up to target time
            {
                "$match": {
                    "user_id": user_id,
                    "timestamp": {"$lte": target_time},
                    "action": {"$in": ["ADD", "UPDATE", "DELETE"]},
                }
            },
            # Sort by memory_id and timestamp
            {"$sort": {"memory_id": 1, "timestamp": -1}},
            # Group by memory_id to get latest state
            {
                "$group": {
                    "_id": "$memory_id",
                    "latest_action": {"$first": "$action"},
                    "latest_state": {"$first": "$after_state"},
                    "latest_timestamp": {"$first": "$timestamp"},
                    "memory_id": {"$first": "$memory_id"},
                    "cid": {"$first": "$cid"},
                }
            },
            # Filter out deleted memories
            {"$match": {"latest_action": {"$ne": "DELETE"}}},
            # Project final result
            {
                "$project": {
                    "memory_id": 1,
                    "cid": 1,
                    "state": "$latest_state",
                    "last_modified": "$latest_timestamp",
                    "action": "$latest_action",
                }
            },
        ]

        cursor = self.collection.aggregate(pipeline)
        return await cursor.to_list(length=None)

    async def get_memory_timeline(self, memory_id: str) -> List[Dict[str, Any]]:
        """Get complete change history for a specific memory"""
        cursor = self.collection.find({"memory_id": memory_id}).sort("timestamp", 1)

        docs = await cursor.to_list(length=None)

        timeline = []
        for doc in docs:
            timeline.append(
                {
                    "timestamp": doc["timestamp"],
                    "action": doc["action"],
                    "before_state": doc.get("before_state"),
                    "after_state": doc.get("after_state"),
                    "changes": doc.get("changes"),
                    "handler_used": doc.get("handler_used"),
                    "processing_time_ms": doc.get("processing_time_ms"),
                    "debug_context": doc.get("debug_context", {}),
                }
            )

        return timeline

    async def find_memory_at_time(
        self, memory_id: str, target_time: datetime
    ) -> Optional[Dict[str, Any]]:
        """Find the state of a specific memory at a given time"""
        doc = await self.collection.find_one(
            {
                "memory_id": memory_id,
                "timestamp": {"$lte": target_time},
                "action": {"$in": ["ADD", "UPDATE"]},
            },
            sort=[("timestamp", -1)],
        )

        if doc:
            return {
                "memory_id": memory_id,
                "state": doc.get("after_state"),
                "timestamp": doc["timestamp"],
                "action": doc["action"],
            }
        return None

    async def time_travel_query(
        self, user_id: str, query: str, target_time: datetime
    ) -> Dict[str, Any]:
        """Simulate answering a query based on memory state at target time"""
        memories_at_time = await self.get_memory_state_at_time(user_id, target_time)

        return {
            "target_time": target_time.isoformat(),
            "available_memories": len(memories_at_time),
            "memory_states": memories_at_time,
            "query": query,
            "context": f"Knowledge available up to {target_time.isoformat()}",
        }

    async def get_session_timeline(self, session_id: str) -> List[Dict[str, Any]]:
        """Get chronological timeline of all operations in a session"""
        cursor = self.collection.find({"session_id": session_id}).sort("timestamp", 1)

        docs = await cursor.to_list(length=None)

        timeline = []
        for doc in docs:
            timeline.append(
                {
                    "timestamp": doc["timestamp"],
                    "memory_id": doc["memory_id"],
                    "action": doc["action"],
                    "memory_type": doc.get("inferred_type"),
                    "processing_time_ms": doc.get("processing_time_ms"),
                    "handler_used": doc.get("handler_used"),
                    "summary": doc.get("after_state", {}).get("summary")
                    if doc.get("after_state")
                    else None,
                }
            )

        return timeline

    # ==========================================
    # DEBUGGING METHODS
    # ==========================================

    async def debug_memory_processing(self, memory_id: str) -> Dict[str, Any]:
        """Get detailed debugging information for memory processing"""
        operations = (
            await self.collection.find({"memory_id": memory_id})
            .sort("timestamp", 1)
            .to_list(length=None)
        )

        if not operations:
            return {"error": "Memory not found"}

        debug_info = {
            "memory_id": memory_id,
            "total_operations": len(operations),
            "processing_timeline": [],
            "performance_analysis": {},
            "error_analysis": {},
            "state_changes": [],
        }

        total_processing_time = 0
        errors = []

        for i, op in enumerate(operations):
            # Timeline entry
            timeline_entry = {
                "operation_order": i + 1,
                "timestamp": op["timestamp"],
                "action": op["action"],
                "handler_used": op.get("handler_used"),
                "processing_time_ms": op.get("processing_time_ms"),
                "success": op.get("error") is None,
            }

            if op.get("error"):
                timeline_entry["error"] = op["error"]
                timeline_entry["error_type"] = op.get("error_type")
                errors.append(op)

            debug_info["processing_timeline"].append(timeline_entry)

            # Accumulate processing time
            if op.get("processing_time_ms"):
                total_processing_time += op["processing_time_ms"]

            # Track state changes
            if op.get("changes"):
                debug_info["state_changes"].append(
                    {"timestamp": op["timestamp"], "changes": op["changes"]}
                )

        # Performance analysis
        debug_info["performance_analysis"] = {
            "total_processing_time_ms": total_processing_time,
            "average_processing_time_ms": total_processing_time / len(operations)
            if operations
            else 0,
            "slowest_operation": max(
                operations, key=lambda x: x.get("processing_time_ms", 0)
            )
            if operations
            else None,
        }

        # Error analysis
        debug_info["error_analysis"] = {
            "total_errors": len(errors),
            "error_rate": len(errors) / len(operations) if operations else 0,
            "errors": errors,
        }

        return debug_info

    async def debug_handler_performance(
        self, handler_name: str, days: int = 7
    ) -> Dict[str, Any]:
        """Analyze performance of a specific handler"""
        start_time = datetime.utcnow() - timedelta(days=days)

        pipeline = [
            {
                "$match": {
                    "handler_used": handler_name,
                    "timestamp": {"$gte": start_time},
                    "processing_time_ms": {"$exists": True},
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_operations": {"$sum": 1},
                    "avg_processing_time": {"$avg": "$processing_time_ms"},
                    "min_processing_time": {"$min": "$processing_time_ms"},
                    "max_processing_time": {"$max": "$processing_time_ms"},
                    "error_count": {
                        "$sum": {"$cond": [{"$ne": ["$error", None]}, 1, 0]}
                    },
                    "operations_by_action": {"$push": "$action"},
                }
            },
        ]

        result = await self.collection.aggregate(pipeline).to_list(length=1)

        if result:
            analysis = result[0]
            analysis["handler_name"] = handler_name
            analysis["analysis_period_days"] = days
            analysis["error_rate"] = (
                analysis["error_count"] / analysis["total_operations"]
                if analysis["total_operations"] > 0
                else 0
            )
            return analysis

        return {"handler_name": handler_name, "no_data": True}

    async def find_similar_errors(
        self, error_type: str, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar errors for debugging patterns"""
        cursor = (
            self.collection.find({"error_type": error_type})
            .sort("timestamp", -1)
            .limit(limit)
        )

        docs = await cursor.to_list(length=limit)

        return [
            {
                "timestamp": doc["timestamp"],
                "memory_id": doc["memory_id"],
                "user_id": doc["user_id"],
                "handler_used": doc.get("handler_used"),
                "error": doc["error"],
                "debug_context": doc.get("debug_context", {}),
            }
            for doc in docs
        ]

    # ==========================================
    # BASE STORE INTERFACE
    # ==========================================

    async def create(self, entry) -> str:
        """Create method for base store interface"""
        return await self.log_operation(entry)

    async def read(self, audit_id: str) -> Optional[AuditLogEntry]:
        """Read a specific audit log entry"""
        doc = await self.collection.find_one({"id": audit_id})
        if doc:
            return self._dict_to_audit_entry(doc)
        return None

    async def update(self, audit_id: str, updates: Dict[str, Any]) -> bool:
        """Update audit entry - limited updates allowed"""
        # Only allow updating debug context and metadata
        allowed_updates = {}
        if "debug_context" in updates:
            allowed_updates["debug_context"] = updates["debug_context"]

        if not allowed_updates:
            return False

        result = await self.collection.update_one(
            {"id": audit_id}, {"$set": allowed_updates}
        )

        return result.modified_count > 0

    async def delete(self, audit_id: str) -> bool:
        """Delete audit entry - generally not recommended"""
        result = await self.collection.delete_one({"id": audit_id})
        return result.deleted_count > 0

    async def search(
        self, query: Dict[str, Any], limit: int = 10
    ) -> List[AuditLogEntry]:
        """Search audit logs"""
        cursor = self.collection.find(query).sort("timestamp", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [self._dict_to_audit_entry(doc) for doc in docs]

    async def get_total_audit_count(self) -> int:
        """Get total number of audit entries"""
        try:
            return await self.collection.count_documents({})
        except Exception as e:
            print(f"Error counting audit entries: {e}")
            return 0

    async def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
