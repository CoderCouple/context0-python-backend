from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from app.memory.models.audit_log import AuditLogEntry
from app.memory.models.memory_entry import MemoryEntry


class BaseStore(ABC):
    """Abstract base class for all memory stores"""

    @abstractmethod
    async def initialize(self):
        """Initialize the store connection/setup"""
        pass

    @abstractmethod
    async def create(self, entry: MemoryEntry) -> str:
        """Create a new memory entry"""
        pass

    @abstractmethod
    async def read(self, memory_id: str) -> Optional[MemoryEntry]:
        """Read a memory entry by ID"""
        pass

    @abstractmethod
    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry"""
        pass

    @abstractmethod
    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry"""
        pass

    @abstractmethod
    async def search(self, query: Dict[str, Any], limit: int = 10) -> List[MemoryEntry]:
        """Search for memory entries"""
        pass

    @abstractmethod
    async def close(self):
        """Close store connections"""
        pass


class VectorStore(BaseStore):
    """Interface for vector stores (Pinecone, Qdrant, etc.)"""

    @abstractmethod
    async def similarity_search(
        self,
        embedding: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[MemoryEntry, float]]:
        """Search by vector similarity"""
        pass

    @abstractmethod
    async def batch_insert(self, entries: List[MemoryEntry]) -> List[str]:
        """Batch insert memory entries with embeddings"""
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by ID"""
        pass

    @abstractmethod
    async def update_memory(self, memory_id: str, memory_dict: Dict[str, Any]) -> bool:
        """Update a memory entry"""
        pass


class GraphStore(BaseStore):
    """Interface for graph stores (Neo4j, etc.)"""

    @abstractmethod
    async def add_relationship(
        self,
        source_id: str,
        target_id: str,
        relationship_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a relationship between entities"""
        pass

    @abstractmethod
    async def get_neighbors(
        self, node_id: str, relationship_type: Optional[str] = None, depth: int = 1
    ) -> List[Dict[str, Any]]:
        """Get neighboring nodes"""
        pass

    @abstractmethod
    async def shortest_path(
        self, source_id: str, target_id: str
    ) -> Optional[List[str]]:
        """Find shortest path between nodes"""
        pass


class DocumentStore(BaseStore):
    """Interface for document stores (MongoDB, etc.)"""

    @abstractmethod
    async def query(
        self,
        filter: Dict[str, Any],
        projection: Optional[Dict[str, int]] = None,
        sort: Optional[Dict[str, int]] = None,
        limit: int = 10,
    ) -> List[MemoryEntry]:
        """Query documents with MongoDB-style syntax"""
        pass

    @abstractmethod
    async def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run aggregation pipeline"""
        pass

    @abstractmethod
    async def create_index(
        self, fields: List[str], options: Optional[Dict[str, Any]] = None
    ):
        """Create index on fields"""
        pass

    @abstractmethod
    async def list_memories(
        self,
        user_id: str,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = "created_at",
        sort_order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """List memories with filtering and pagination"""
        pass


class TimeSeriesStore(BaseStore):
    """Interface for time series stores (TimescaleDB, InfluxDB)"""

    @abstractmethod
    async def query_time_range(
        self,
        start_time: str,
        end_time: str,
        user_id: Optional[str] = None,
        memory_type: Optional[str] = None,
    ) -> List[MemoryEntry]:
        """Query memories within time range"""
        pass

    @abstractmethod
    async def get_timeline(
        self, user_id: str, granularity: str = "day"  # hour, day, week, month
    ) -> List[Dict[str, Any]]:
        """Get aggregated timeline of memories"""
        pass

    @abstractmethod
    async def retention_policy(self, policy: Dict[str, Any]):
        """Set data retention policy"""
        pass


class AuditStore(BaseStore):
    """Specialized store for audit logs"""

    @abstractmethod
    async def log_operation(self, audit_entry: AuditLogEntry) -> str:
        """Log a memory operation"""
        pass

    @abstractmethod
    async def get_audit_trail(
        self,
        memory_id: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
    ) -> List[AuditLogEntry]:
        """Get audit trail for memory operations"""
        pass
