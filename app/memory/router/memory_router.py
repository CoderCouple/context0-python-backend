from typing import Dict, Optional, Type

from app.common.enum.memory import MemoryType
from app.memory.handlers.base_handler import BaseMemoryHandler
from app.memory.inferencer.memory_type_inferencer import MemoryTypeInferencer
from app.memory.models.memory_entry import MemoryEntry
from app.memory.models.memory_record import MemoryRecord


class MemoryRouter:
    """Routes memory records to appropriate handlers based on memory type"""

    def __init__(self, inferencer: Optional[MemoryTypeInferencer] = None):
        """Initialize router with memory type inferencer"""
        self.inferencer = inferencer or MemoryTypeInferencer()
        self.handlers: Dict[MemoryType, BaseMemoryHandler] = {}
        self._initialize_handlers()

    def _initialize_handlers(self):
        """Initialize all memory type handlers"""
        # These will be imported once we create them
        from app.memory.handlers.declarative_handler import DeclarativeMemoryHandler
        from app.memory.handlers.emotional_handler import EmotionalMemoryHandler
        from app.memory.handlers.episodic_handler import EpisodicMemoryHandler
        from app.memory.handlers.meta_handler import MetaMemoryHandler
        from app.memory.handlers.procedural_handler import ProceduralMemoryHandler
        from app.memory.handlers.semantic_handler import SemanticMemoryHandler
        from app.memory.handlers.working_handler import WorkingMemoryHandler

        self.handlers = {
            MemoryType.SEMANTIC_MEMORY: SemanticMemoryHandler(),
            MemoryType.EPISODIC_MEMORY: EpisodicMemoryHandler(),
            MemoryType.PROCEDURAL_MEMORY: ProceduralMemoryHandler(),
            MemoryType.EMOTIONAL_MEMORY: EmotionalMemoryHandler(),
            MemoryType.WORKING_MEMORY: WorkingMemoryHandler(),
            MemoryType.DECLARATIVE_MEMORY: DeclarativeMemoryHandler(),
            MemoryType.META_MEMORY: MetaMemoryHandler(),
        }

    def register_handler(self, memory_type: MemoryType, handler: BaseMemoryHandler):
        """Register a custom handler for a memory type"""
        self.handlers[memory_type] = handler

    async def route(self, record: MemoryRecord) -> BaseMemoryHandler:
        """
        Route a memory record to the appropriate handler

        Args:
            record: The incoming memory record

        Returns:
            The appropriate handler for processing
        """
        # Determine memory type
        memory_type = record.memory_type

        # Infer type if not provided
        if memory_type is None:
            memory_type, confidence_score = await self.inferencer.infer_type(
                record.raw_text,
                context={
                    "session_id": record.session_id,
                    "tags": record.tags,
                    "metadata": record.metadata,
                },
            )
            record.memory_type = memory_type

        # Get appropriate handler
        handler = self.handlers.get(memory_type)
        if handler is None:
            # Fallback to semantic handler
            handler = self.handlers[MemoryType.SEMANTIC_MEMORY]

        return handler

    async def batch_route(self, records: list[MemoryRecord]) -> list[MemoryEntry]:
        """Batch route multiple memory records"""
        # Group by memory type for efficient processing
        grouped: Dict[Optional[MemoryType], list[MemoryRecord]] = {}
        for record in records:
            memory_type = record.memory_type
            if memory_type not in grouped:
                grouped[memory_type] = []
            grouped[memory_type].append(record)

        results = []

        # Process each group
        for memory_type, group_records in grouped.items():
            if memory_type is None:
                # These need inference
                for record in group_records:
                    result = await self.route(record)
                    results.append(result)
            else:
                # These can be batch processed by the handler
                handler = self.handlers.get(
                    memory_type, self.handlers[MemoryType.SEMANTIC_MEMORY]
                )
                group_results = await handler.batch_process(group_records)
                results.extend(group_results)

        return results

    def get_handler(self, memory_type: MemoryType) -> Optional[BaseMemoryHandler]:
        """Get the handler for a specific memory type"""
        return self.handlers.get(memory_type)
