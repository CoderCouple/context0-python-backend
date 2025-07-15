import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from dateutil import parser

from app.common.enum.memory import MemoryType
from app.memory.handlers.base_handler import BaseMemoryHandler
from app.memory.models.memory_entry import GraphLink, MemoryEntry
from app.memory.models.memory_record import MemoryRecord
from app.memory.utils.embeddings import get_embeddings


class EpisodicMemoryHandler(BaseMemoryHandler):
    """Handler for episodic memory (events tied to time and context)"""

    def __init__(self):
        super().__init__()
        self.memory_type = MemoryType.EPISODIC_MEMORY
        api_key = os.getenv("OPENAI_API_KEY")
        self.embedder = get_embeddings(model="text-embedding-3-small", api_key=api_key)

    async def process(
        self, record: MemoryRecord, confidence_score: float = 1.0
    ) -> MemoryEntry:
        """Process episodic memory - events with temporal context"""
        processing_start = datetime.utcnow()

        # Validate
        if not await self.validate_record(record):
            raise ValueError("Invalid memory record")

        # Extract embedding
        embedding, model_name = await self.extract_embedding(record.raw_text)

        # Extract temporal information
        temporal_info = await self.extract_temporal_info(record)

        # Generate summary
        summary = await self.generate_summary(record.raw_text)

        # Extract graph links (events, people, places)
        graph_links = await self.extract_graph_links(record)

        # Enhance metadata with temporal info
        enhanced_metadata = {**record.metadata, **temporal_info}

        # Create memory entry
        memory_entry = MemoryEntry(
            id=self.generate_id(),
            cid=self.generate_cid(record.raw_text),
            scope=f"user:{record.user_id}",
            input=record.raw_text,
            summary=summary,
            memory_type=MemoryType.EPISODIC_MEMORY,
            permissions=self.create_permissions(record.user_id),
            embedding=embedding,
            embedding_model=model_name,
            graph_links=[GraphLink(**link) for link in graph_links],
            meta=self.create_meta(confidence_score, processing_start),
            tags=record.tags + ["episodic", "event"],
            custom_metadata=enhanced_metadata,
            source_session_id=record.session_id,
            source_user_id=record.user_id,
        )

        return memory_entry

    async def extract_embedding(self, text: str) -> tuple[List[float], str]:
        """Extract embedding for episodic content"""
        try:
            embedding = await self.embedder.aembed_query(text)
            return embedding, "text-embedding-3-small"
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * 1536, "error"

    async def extract_temporal_info(self, record: MemoryRecord) -> Dict[str, Any]:
        """Extract time-related information from the text"""
        temporal_info = {}
        text = record.raw_text.lower()

        # Look for explicit dates
        date_patterns = [
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
            r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
            r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b",
        ]

        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    parsed_date = parser.parse(matches[0])
                    temporal_info["event_date"] = parsed_date.isoformat()
                    break
                except:
                    pass

        # Look for relative time references
        time_references = {
            "yesterday": -1,
            "today": 0,
            "tomorrow": 1,
            "last week": -7,
            "next week": 7,
            "last month": -30,
            "last year": -365,
        }

        for ref, days_offset in time_references.items():
            if ref in text:
                temporal_info["relative_time"] = ref
                temporal_info["days_offset"] = days_offset
                break

        # Extract time of day
        time_pattern = r"\b(\d{1,2}):(\d{2})(?:\s*(am|pm))?\b"
        time_matches = re.findall(time_pattern, text, re.IGNORECASE)
        if time_matches:
            temporal_info["time_of_day"] = time_matches[0]

        return temporal_info

    async def extract_graph_links(self, record: MemoryRecord) -> List[Dict[str, Any]]:
        """Extract event-related entities for graph relationships"""
        links = []
        text = record.raw_text

        # Extract people (simplified - in production use NER)
        person_patterns = [
            r"\b(?:I|me|my)\s+(?:met|saw|talked to|visited)\s+([A-Z][a-z]+)\b",
            r"\b([A-Z][a-z]+)\s+(?:said|told|asked|invited)\b",
        ]

        people = set()
        for pattern in person_patterns:
            matches = re.findall(pattern, text)
            people.update(matches)

        for person in list(people)[:3]:  # Limit to 3 people
            links.append(
                {
                    "target_id": f"person:{person.lower()}",
                    "relationship_type": "involves",
                    "properties": {"entity_type": "person"},
                }
            )

        # Extract locations
        location_patterns = [
            r"\b(?:at|in|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
            r"\b(?:visited|went to|arrived at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
        ]

        locations = set()
        for pattern in location_patterns:
            matches = re.findall(pattern, text)
            locations.update(matches)

        for location in list(locations)[:2]:  # Limit to 2 locations
            links.append(
                {
                    "target_id": f"location:{location.lower().replace(' ', '_')}",
                    "relationship_type": "occurred_at",
                    "properties": {"entity_type": "location"},
                }
            )

        # Link to temporal context
        if "event_date" in record.metadata or "relative_time" in record.metadata:
            links.append(
                {
                    "target_id": f"timeline:{record.user_id}",
                    "relationship_type": "part_of",
                    "properties": {"timeline_type": "personal"},
                }
            )

        return links

    async def generate_summary(self, text: str) -> Optional[str]:
        """Generate a concise summary of the episodic event"""
        # For episodic memories, extract the main action/event
        action_words = [
            "visited",
            "met",
            "went",
            "saw",
            "attended",
            "completed",
            "finished",
            "started",
        ]

        for action in action_words:
            if action in text.lower():
                # Find the sentence containing this action
                sentences = text.split(".")
                for sentence in sentences:
                    if action in sentence.lower():
                        return sentence.strip() + "."

        # Fallback to first sentence
        sentences = text.split(".")
        if sentences:
            return sentences[0].strip() + "."

        return None
