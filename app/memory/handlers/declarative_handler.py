import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.common.enum.memory import MemoryType
from app.memory.handlers.base_handler import BaseMemoryHandler
from app.memory.models.memory_entry import GraphLink, MemoryEntry
from app.memory.models.memory_record import MemoryRecord
from app.memory.utils.embeddings import get_embeddings


class DeclarativeMemoryHandler(BaseMemoryHandler):
    """Handler for declarative memory (explicit facts)"""

    def __init__(self):
        super().__init__()
        self.memory_type = MemoryType.DECLARATIVE_MEMORY
        api_key = os.getenv("OPENAI_API_KEY")
        self.embedder = get_embeddings(model="text-embedding-3-small", api_key=api_key)

    async def process(
        self, record: MemoryRecord, confidence_score: float = 1.0
    ) -> MemoryEntry:
        processing_start = datetime.utcnow()

        embedding, model_name = await self.extract_embedding(record.raw_text)

        return MemoryEntry(
            id=self.generate_id(),
            cid=self.generate_cid(record.raw_text),
            scope=f"user:{record.user_id}",
            input=record.raw_text,
            summary=None,
            memory_type=MemoryType.DECLARATIVE_MEMORY,
            permissions=self.create_permissions(record.user_id),
            embedding=embedding,
            embedding_model=model_name,
            graph_links=[],
            meta=self.create_meta(confidence_score, processing_start),
            tags=record.tags + ["declarative", "fact"],
            custom_metadata=record.metadata,
            source_session_id=record.session_id,
            source_user_id=record.user_id,
        )

    async def extract_embedding(self, text: str) -> tuple[List[float], str]:
        try:
            embedding = await self.embedder.aembed_query(text)
            return embedding, "text-embedding-3-small"
        except Exception:
            return [0.0] * 1536, "error"

    async def extract_graph_links(self, record: MemoryRecord) -> List[Dict[str, Any]]:
        return []

    async def generate_summary(self, text: str) -> Optional[str]:
        return None
