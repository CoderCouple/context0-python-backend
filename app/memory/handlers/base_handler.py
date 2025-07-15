import hashlib
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.common.enum.memory import MemoryType
from app.memory.models.memory_entry import MemoryEntry, MemoryMeta, MemoryPermissions
from app.memory.models.memory_record import MemoryRecord


class BaseMemoryHandler(ABC):
    """Abstract base class for all memory type handlers"""

    def __init__(self):
        self.memory_type: MemoryType = None
        self.handler_version = "1.0.0"

    @abstractmethod
    async def process(
        self, record: MemoryRecord, confidence_score: float = 1.0
    ) -> MemoryEntry:
        """
        Process a memory record and return a memory entry

        Args:
            record: The incoming memory record
            confidence_score: Confidence in the memory type classification

        Returns:
            Processed MemoryEntry ready for persistence
        """
        pass

    @abstractmethod
    async def extract_embedding(self, text: str) -> tuple[List[float], str]:
        """
        Extract embedding for the text

        Args:
            text: Text to embed

        Returns:
            Tuple of (embedding vector, model name)
        """
        pass

    @abstractmethod
    async def extract_graph_links(self, record: MemoryRecord) -> List[Dict[str, Any]]:
        """
        Extract graph relationships from the memory

        Args:
            record: The memory record

        Returns:
            List of graph links
        """
        pass

    @abstractmethod
    async def generate_summary(self, text: str) -> Optional[str]:
        """
        Generate a summary of the memory content

        Args:
            text: Text to summarize

        Returns:
            Summary or None if not applicable
        """
        pass

    async def batch_process(self, records: List[MemoryRecord]) -> List[MemoryEntry]:
        """
        Batch process multiple memory records
        Default implementation just processes sequentially
        """
        results = []
        for record in records:
            result = await self.process(record)
            results.append(result)
        return results

    def generate_id(self) -> str:
        """Generate a unique ID for the memory"""
        return str(uuid.uuid4())

    def generate_cid(self, content: str) -> str:
        """Generate content ID for deduplication"""
        return hashlib.sha256(content.encode()).hexdigest()

    def create_permissions(self, user_id: str) -> MemoryPermissions:
        """Create default permissions for a memory"""
        return MemoryPermissions(
            owner_id=user_id,
            read_access=[user_id],
            write_access=[user_id],
            is_public=False,
        )

    def create_meta(
        self, confidence_score: float, processing_start: datetime
    ) -> MemoryMeta:
        """Create metadata for the memory processing"""
        processing_time = (datetime.utcnow() - processing_start).total_seconds() * 1000
        return MemoryMeta(
            classifier_used="memory_type_inferencer"
            if confidence_score < 1.0
            else None,
            llm_version=None,  # Will be set by specific handlers
            handler_version=self.handler_version,
            processing_time_ms=processing_time,
            confidence_score=confidence_score,
        )

    async def enrich_with_temporal_info(self, record: MemoryRecord) -> Dict[str, Any]:
        """Extract temporal information from the memory"""
        # This is a placeholder - specific handlers can override
        return {"timestamp": record.timestamp, "session_time": datetime.utcnow()}

    async def validate_record(self, record: MemoryRecord) -> bool:
        """Validate the incoming record"""
        if not record.raw_text or not record.raw_text.strip():
            return False
        if not record.user_id or not record.session_id:
            return False
        return True
