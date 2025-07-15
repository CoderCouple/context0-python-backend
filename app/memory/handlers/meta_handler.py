import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.common.enum.memory import MemoryType
from app.memory.handlers.base_handler import BaseMemoryHandler
from app.memory.models.memory_entry import GraphLink, MemoryEntry
from app.memory.models.memory_record import MemoryRecord
from app.memory.utils.embeddings import get_embeddings


class MetaMemoryHandler(BaseMemoryHandler):
    """Handler for meta memory (memories about memories)"""

    def __init__(self):
        super().__init__()
        self.memory_type = MemoryType.META_MEMORY
        api_key = os.getenv("OPENAI_API_KEY")
        self.embedder = get_embeddings(model="text-embedding-3-small", api_key=api_key)

    async def process(
        self, record: MemoryRecord, confidence_score: float = 1.0
    ) -> MemoryEntry:
        """Process meta memory - memories about memories, reflections, and insights"""
        processing_start = datetime.utcnow()

        # Validate
        if not await self.validate_record(record):
            raise ValueError("Invalid memory record")

        # Extract embedding
        embedding, model_name = await self.extract_embedding(record.raw_text)

        # Generate summary for meta-memories
        summary = await self.generate_summary(record.raw_text)

        # Extract meta-memory relationships
        graph_links = await self.extract_graph_links(record)

        # Determine meta-memory type
        meta_type = self._determine_meta_type(record.raw_text)

        # Create memory entry
        memory_entry = MemoryEntry(
            id=self.generate_id(),
            cid=self.generate_cid(record.raw_text),
            scope=f"user:{record.user_id}",
            input=record.raw_text,
            summary=summary,
            memory_type=MemoryType.META_MEMORY,
            permissions=self.create_permissions(record.user_id),
            embedding=embedding,
            embedding_model=model_name,
            graph_links=[GraphLink(**link) for link in graph_links],
            meta=self.create_meta(confidence_score, processing_start),
            tags=record.tags + ["meta", "reflection", meta_type],
            custom_metadata={
                **record.metadata,
                "meta_type": meta_type,
                "reflection_level": self._assess_reflection_level(record.raw_text),
                "temporal_distance": self._assess_temporal_distance(record.raw_text),
            },
            source_session_id=record.session_id,
            source_user_id=record.user_id,
        )

        return memory_entry

    async def extract_embedding(self, text: str) -> tuple[List[float], str]:
        try:
            embedding = await self.embedder.aembed_query(text)
            return embedding, "text-embedding-3-small"
        except Exception:
            return [0.0] * 1536, "error"

    async def extract_graph_links(self, record: MemoryRecord) -> List[Dict[str, Any]]:
        """Extract relationships for meta-memories"""
        links = []
        text = record.raw_text.lower()

        # References to past memories or experiences
        memory_references = [
            (
                r"\b(?:i remember|i recalled|thinking about|reflecting on)\s+([^.]+)",
                "reflects_on",
            ),
            (r"\b(?:that time when|when i)\s+([^.]+)", "references"),
            (r"\b(?:my experience with|what happened)\s+([^.]+)", "analyzes"),
            (
                r"\b(?:i learned that|i realized that|i discovered)\s+([^.]+)",
                "learned_from",
            ),
            (r"\b(?:i should have|i wish i had|if only i)\s+([^.]+)", "regrets"),
            (r"\b(?:next time|in the future|going forward)\s+([^.]+)", "plans_for"),
        ]

        for pattern, relationship_type in memory_references:
            matches = re.finditer(pattern, record.raw_text, re.IGNORECASE)
            for match in matches:
                referenced_content = match.group(1).strip()
                links.append(
                    {
                        "target_id": f"memory_reference:{referenced_content[:50].lower().replace(' ', '_')}",
                        "relationship_type": relationship_type,
                        "properties": {
                            "entity_type": "memory_reference",
                            "referenced_content": referenced_content,
                            "is_meta_reference": True,
                        },
                    }
                )

        # Emotional or cognitive states about memories
        metacognitive_patterns = [
            (
                r"\b(confident|uncertain|confused|clear|foggy|vivid|vague)\s+about",
                "cognitive_state",
            ),
            (
                r"\b(proud|ashamed|happy|sad|regretful|grateful)\s+(?:about|that)",
                "emotional_state",
            ),
            (r"\b(pattern|trend|habit|recurring)\b", "pattern_recognition"),
            (r"\b(insight|realization|understanding|clarity)\b", "cognitive_insight"),
        ]

        for pattern, meta_type in metacognitive_patterns:
            matches = re.finditer(pattern, record.raw_text, re.IGNORECASE)
            for match in matches:
                state = (
                    match.group(1)
                    if meta_type in ["cognitive_state", "emotional_state"]
                    else match.group(0)
                )
                links.append(
                    {
                        "target_id": f"meta_state:{state.lower()}",
                        "relationship_type": "has_meta_state",
                        "properties": {
                            "entity_type": meta_type,
                            "state_name": state,
                            "is_metacognitive": True,
                        },
                    }
                )

        # Temporal references in meta-memories
        temporal_meta_patterns = [
            (r"\b(looking back|in hindsight|now i see)\b", "retrospective"),
            (r"\b(going forward|next time|in the future)\b", "prospective"),
            (r"\b(at the time|back then|i used to think)\b", "temporal_perspective"),
        ]

        for pattern, temporal_type in temporal_meta_patterns:
            matches = re.finditer(pattern, record.raw_text, re.IGNORECASE)
            for match in matches:
                temporal_ref = match.group(0)
                links.append(
                    {
                        "target_id": f"temporal_meta:{temporal_ref.lower().replace(' ', '_')}",
                        "relationship_type": "has_temporal_perspective",
                        "properties": {
                            "entity_type": "temporal_perspective",
                            "perspective_type": temporal_type,
                            "temporal_reference": temporal_ref,
                        },
                    }
                )

        return links[:8]  # Limit to 8 most relevant meta-links

    async def generate_summary(self, text: str) -> Optional[str]:
        """Generate summary for meta-memories focusing on the reflection/insight"""
        # Extract the key insight or reflection
        insight_patterns = [
            r"\b(?:i learned|i realized|i discovered|i understood)\s+([^.]+)",
            r"\b(?:the insight is|the key point is|what i see now is)\s+([^.]+)",
            r"\b(?:reflecting on this|thinking about it|looking back)\s*[,:]\s*([^.]+)",
        ]

        for pattern in insight_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: return first sentence if it's reflective
        sentences = text.split(".")
        if sentences and len(sentences[0]) > 20:
            first_sentence = sentences[0].strip()
            reflective_indicators = [
                "think",
                "feel",
                "remember",
                "realize",
                "learn",
                "understand",
                "reflect",
            ]
            if any(
                indicator in first_sentence.lower()
                for indicator in reflective_indicators
            ):
                return first_sentence

        return None

    def _determine_meta_type(self, text: str) -> str:
        """Determine the type of meta-memory"""
        text_lower = text.lower()

        if any(
            word in text_lower
            for word in ["learned", "realized", "discovered", "insight", "understand"]
        ):
            return "learning_reflection"
        elif any(
            word in text_lower
            for word in ["regret", "wish", "should have", "mistake", "wrong"]
        ):
            return "regret_analysis"
        elif any(
            word in text_lower
            for word in ["proud", "accomplished", "success", "well", "good choice"]
        ):
            return "success_reflection"
        elif any(
            word in text_lower
            for word in ["pattern", "always", "tend to", "usually", "habit"]
        ):
            return "pattern_recognition"
        elif any(
            word in text_lower
            for word in ["plan", "next time", "future", "going forward", "will"]
        ):
            return "future_planning"
        elif any(
            word in text_lower
            for word in ["remember", "thinking about", "reminds me", "recalls"]
        ):
            return "memory_recall"
        else:
            return "general_reflection"

    def _assess_reflection_level(self, text: str) -> str:
        """Assess the depth of reflection"""
        text_lower = text.lower()

        deep_indicators = [
            "why",
            "because",
            "underlying",
            "root cause",
            "deeper",
            "fundamental",
        ]
        surface_indicators = ["what", "when", "where", "simple", "obvious"]

        deep_count = sum(1 for indicator in deep_indicators if indicator in text_lower)
        surface_count = sum(
            1 for indicator in surface_indicators if indicator in text_lower
        )

        if deep_count > surface_count and deep_count >= 2:
            return "deep"
        elif deep_count > 0:
            return "moderate"
        else:
            return "surface"

    def _assess_temporal_distance(self, text: str) -> str:
        """Assess how far back the memory being reflected upon is"""
        text_lower = text.lower()

        if any(
            word in text_lower for word in ["just", "today", "this morning", "earlier"]
        ):
            return "immediate"
        elif any(word in text_lower for word in ["yesterday", "last week", "recently"]):
            return "recent"
        elif any(
            word in text_lower for word in ["last month", "few months ago", "this year"]
        ):
            return "medium"
        elif any(
            word in text_lower for word in ["years ago", "long time ago", "when i was"]
        ):
            return "distant"
        else:
            return "unspecified"
