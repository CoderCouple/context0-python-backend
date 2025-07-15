import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.common.enum.memory import MemoryType
from app.memory.handlers.base_handler import BaseMemoryHandler
from app.memory.models.memory_entry import GraphLink, MemoryEntry
from app.memory.models.memory_record import MemoryRecord
from app.memory.utils.embeddings import get_embeddings


class SemanticMemoryHandler(BaseMemoryHandler):
    """Handler for semantic memory (factual knowledge)"""

    def __init__(self):
        super().__init__()
        self.memory_type = MemoryType.SEMANTIC_MEMORY
        # Use get_embeddings which handles fallback to mock automatically
        import os

        api_key = os.getenv("OPENAI_API_KEY")
        self.embedder = get_embeddings(model="text-embedding-3-small", api_key=api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50
        )

    async def process(
        self, record: MemoryRecord, confidence_score: float = 1.0
    ) -> MemoryEntry:
        """Process semantic memory - facts and knowledge"""
        processing_start = datetime.utcnow()

        # Validate
        if not await self.validate_record(record):
            raise ValueError("Invalid memory record")

        # Extract embedding
        embedding, model_name = await self.extract_embedding(record.raw_text)

        # Generate summary for longer texts
        summary = await self.generate_summary(record.raw_text)

        # Extract graph links (concepts, entities)
        graph_links = await self.extract_graph_links(record)

        # Create memory entry
        memory_entry = MemoryEntry(
            id=self.generate_id(),
            cid=self.generate_cid(record.raw_text),
            scope=f"user:{record.user_id}",
            input=record.raw_text,
            summary=summary,
            memory_type=MemoryType.SEMANTIC_MEMORY,
            permissions=self.create_permissions(record.user_id),
            embedding=embedding,
            embedding_model=model_name,
            graph_links=[GraphLink(**link) for link in graph_links],
            meta=self.create_meta(confidence_score, processing_start),
            tags=record.tags + ["semantic", "knowledge"],
            custom_metadata=record.metadata,
            source_session_id=record.session_id,
            source_user_id=record.user_id,
        )

        return memory_entry

    async def extract_embedding(self, text: str) -> tuple[List[float], str]:
        """Extract embedding using OpenAI"""
        try:
            embedding = await self.embedder.aembed_query(text)
            return embedding, "text-embedding-3-small"
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return zero vector as fallback
            return [0.0] * 1536, "error"

    async def extract_graph_links(self, record: MemoryRecord) -> List[Dict[str, Any]]:
        """Extract entities and concepts for graph relationships"""
        links = []
        text = record.raw_text.lower()

        # Extract people and relationships
        people_patterns = [
            (
                r"\b(?:my|our)\s+(friend|brother|sister|mother|father|mom|dad|son|daughter|cousin|uncle|aunt|grandfather|grandmother|wife|husband|partner|boyfriend|girlfriend)\s+([A-Z][a-z]+)",
                "family_friend",
            ),
            (
                r"\b([A-Z][a-z]+)\s+(?:is|was)\s+(?:my|our)\s+(friend|brother|sister|colleague|boss|teacher|student)",
                "person",
            ),
            (
                r"\b(?:met|know|knows|talked to|spoke with|called|texted)\s+([A-Z][a-z]+)",
                "person",
            ),
            (
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:told me|said|mentioned)",
                "person",
            ),
        ]

        for pattern, entity_type in people_patterns:
            matches = re.finditer(pattern, record.raw_text, re.IGNORECASE)
            for match in matches:
                if entity_type == "family_friend":
                    relationship_type = match.group(1)  # friend, brother, etc.
                    person_name = match.group(2)
                    links.append(
                        {
                            "target_id": f"person:{person_name.lower()}",
                            "relationship_type": f"has_{relationship_type}",
                            "properties": {
                                "entity_type": "person",
                                "relationship_context": relationship_type,
                                "mentioned_in_memory": True,
                            },
                        }
                    )
                else:
                    person_name = match.group(1)
                    links.append(
                        {
                            "target_id": f"person:{person_name.lower()}",
                            "relationship_type": "knows",
                            "properties": {
                                "entity_type": "person",
                                "mentioned_in_memory": True,
                            },
                        }
                    )

        # Extract dates and temporal information
        date_patterns = [
            (
                r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2})",
                "date",
            ),
            (r"\b(\d{1,2})/(\d{1,2})/(\d{4})", "date"),
            (
                r"\b(yesterday|today|tomorrow|last week|next week|last month)",
                "temporal",
            ),
        ]

        for pattern, entity_type in date_patterns:
            matches = re.finditer(pattern, record.raw_text, re.IGNORECASE)
            for match in matches:
                if entity_type == "date":
                    date_str = match.group(0)
                    links.append(
                        {
                            "target_id": f"date:{date_str.lower().replace(' ', '_')}",
                            "relationship_type": "occurred_on",
                            "properties": {
                                "entity_type": "date",
                                "temporal_reference": date_str,
                            },
                        }
                    )
                else:
                    temporal_ref = match.group(0)
                    links.append(
                        {
                            "target_id": f"temporal:{temporal_ref.lower().replace(' ', '_')}",
                            "relationship_type": "occurred_during",
                            "properties": {
                                "entity_type": "temporal",
                                "temporal_reference": temporal_ref,
                            },
                        }
                    )

        # Extract locations
        location_patterns = [
            (r"\b(?:at|in|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", "location"),
            (
                r"\b(home|work|office|school|university|restaurant|park|beach|hospital)",
                "location",
            ),
        ]

        for pattern, entity_type in location_patterns:
            matches = re.finditer(pattern, record.raw_text, re.IGNORECASE)
            for match in matches:
                location = (
                    match.group(1)
                    if pattern.startswith(r"\b(?:at|in|to|from)")
                    else match.group(0)
                )
                links.append(
                    {
                        "target_id": f"location:{location.lower().replace(' ', '_')}",
                        "relationship_type": "happened_at",
                        "properties": {
                            "entity_type": "location",
                            "location_name": location,
                        },
                    }
                )

        # Extract general entities (organizations, brands, etc.)
        entity_patterns = [
            (
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:company|corporation|inc|llc)",
                "organization",
            ),
            (
                r"\b(Apple|Google|Microsoft|Amazon|Meta|Tesla|Netflix)",
                "company",
            ),  # Add more as needed
        ]

        for pattern, entity_type in entity_patterns:
            matches = re.finditer(pattern, record.raw_text)
            for match in matches:
                entity_name = (
                    match.group(1) if entity_type == "organization" else match.group(0)
                )
                links.append(
                    {
                        "target_id": f"entity:{entity_name.lower().replace(' ', '_')}",
                        "relationship_type": "mentions",
                        "properties": {
                            "entity_type": entity_type,
                            "entity_name": entity_name,
                        },
                    }
                )

        # Extract key concepts and topics
        concept_patterns = [
            (
                r"\b(birthday|anniversary|wedding|graduation|meeting|conference|vacation|trip)",
                "event",
            ),
            (
                r"\b(happy|sad|excited|worried|angry|frustrated|proud|grateful)",
                "emotion",
            ),
            (r"\b(work|job|career|project|task|deadline)", "work_related"),
            (r"\b(health|doctor|medicine|exercise|diet|sick)", "health_related"),
        ]

        for pattern, concept_type in concept_patterns:
            matches = re.finditer(pattern, record.raw_text, re.IGNORECASE)
            for match in matches:
                concept = match.group(0)
                links.append(
                    {
                        "target_id": f"concept:{concept.lower()}",
                        "relationship_type": "relates_to",
                        "properties": {
                            "concept_type": concept_type,
                            "concept_name": concept,
                        },
                    }
                )

        return links[:10]  # Limit to 10 most relevant links

    async def generate_summary(self, text: str) -> Optional[str]:
        """Generate summary for longer semantic content"""
        if len(text) > 200:
            # For semantic memory, extract the key fact
            sentences = text.split(".")
            if sentences:
                return sentences[0].strip() + "."
        return None

    async def batch_process(self, records: List[MemoryRecord]) -> List[MemoryEntry]:
        """Optimized batch processing for semantic memories"""
        # Batch embed all texts
        texts = [record.raw_text for record in records]
        embeddings = await self.embedder.aembed_documents(texts)

        results = []
        for record, embedding in zip(records, embeddings):
            processing_start = datetime.utcnow()

            summary = await self.generate_summary(record.raw_text)
            graph_links = await self.extract_graph_links(record)

            memory_entry = MemoryEntry(
                id=self.generate_id(),
                cid=self.generate_cid(record.raw_text),
                scope=f"user:{record.user_id}",
                input=record.raw_text,
                summary=summary,
                memory_type=MemoryType.SEMANTIC_MEMORY,
                permissions=self.create_permissions(record.user_id),
                embedding=embedding,
                embedding_model="text-embedding-3-small",
                graph_links=[GraphLink(**link) for link in graph_links],
                meta=self.create_meta(1.0, processing_start),
                tags=record.tags + ["semantic", "knowledge"],
                custom_metadata=record.metadata,
                source_session_id=record.session_id,
                source_user_id=record.user_id,
            )
            results.append(memory_entry)

        return results
