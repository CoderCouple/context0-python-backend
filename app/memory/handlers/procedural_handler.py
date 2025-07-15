import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.common.enum.memory import MemoryType
from app.memory.handlers.base_handler import BaseMemoryHandler
from app.memory.models.memory_entry import GraphLink, MemoryEntry
from app.memory.models.memory_record import MemoryRecord
from app.memory.utils.embeddings import get_embeddings


class ProceduralMemoryHandler(BaseMemoryHandler):
    """Handler for procedural memory (how-to knowledge, instructions)"""

    def __init__(self):
        super().__init__()
        self.memory_type = MemoryType.PROCEDURAL_MEMORY
        api_key = os.getenv("OPENAI_API_KEY")
        self.embedder = get_embeddings(model="text-embedding-3-small", api_key=api_key)

    async def process(
        self, record: MemoryRecord, confidence_score: float = 1.0
    ) -> MemoryEntry:
        """Process procedural memory - instructions and procedures"""
        processing_start = datetime.utcnow()

        # Validate
        if not await self.validate_record(record):
            raise ValueError("Invalid memory record")

        # Extract embedding
        embedding, model_name = await self.extract_embedding(record.raw_text)

        # Extract steps if present
        steps = await self.extract_steps(record.raw_text)

        # Generate summary
        summary = await self.generate_summary(record.raw_text)

        # Extract graph links
        graph_links = await self.extract_graph_links(record)

        # Enhance metadata with procedural info
        enhanced_metadata = {
            **record.metadata,
            "steps_count": len(steps),
            "steps": steps if steps else None,
        }

        # Create memory entry
        memory_entry = MemoryEntry(
            id=self.generate_id(),
            cid=self.generate_cid(record.raw_text),
            scope=f"user:{record.user_id}",
            input=record.raw_text,
            summary=summary,
            memory_type=MemoryType.PROCEDURAL_MEMORY,
            permissions=self.create_permissions(record.user_id),
            embedding=embedding,
            embedding_model=model_name,
            graph_links=[GraphLink(**link) for link in graph_links],
            meta=self.create_meta(confidence_score, processing_start),
            tags=record.tags + ["procedural", "how-to"],
            custom_metadata=enhanced_metadata,
            source_session_id=record.session_id,
            source_user_id=record.user_id,
        )

        return memory_entry

    async def extract_embedding(self, text: str) -> tuple[List[float], str]:
        """Extract embedding for procedural content"""
        try:
            embedding = await self.embedder.aembed_query(text)
            return embedding, "text-embedding-3-small"
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return [0.0] * 1536, "error"

    async def extract_steps(self, text: str) -> List[str]:
        """Extract numbered or bulleted steps from the text"""
        steps = []

        # Look for numbered steps
        numbered_pattern = r"(?:^|\n)\s*(\d+)[.)\s]+(.+?)(?=\n\s*\d+[.)\s]+|\n\n|$)"
        numbered_matches = re.findall(numbered_pattern, text, re.MULTILINE | re.DOTALL)
        if numbered_matches:
            steps = [match[1].strip() for match in numbered_matches]
            return steps

        # Look for bulleted steps
        bullet_pattern = r"(?:^|\n)\s*[•\-*]\s+(.+?)(?=\n\s*[•\-*]|\n\n|$)"
        bullet_matches = re.findall(bullet_pattern, text, re.MULTILINE | re.DOTALL)
        if bullet_matches:
            steps = [match.strip() for match in bullet_matches]
            return steps

        # Look for imperative sentences (simplified)
        sentences = text.split(".")
        for sentence in sentences:
            sentence = sentence.strip()
            # Simple heuristic: starts with a verb
            if (
                sentence
                and sentence[0].isupper()
                and any(
                    sentence.lower().startswith(verb)
                    for verb in [
                        "add",
                        "remove",
                        "mix",
                        "place",
                        "put",
                        "take",
                        "make",
                        "create",
                        "install",
                    ]
                )
            ):
                steps.append(sentence)

        return steps

    async def extract_graph_links(self, record: MemoryRecord) -> List[Dict[str, Any]]:
        """Extract procedure-related entities"""
        links = []

        # Link to procedure type
        procedure_keywords = {
            "recipe": "cooking",
            "install": "technical",
            "setup": "configuration",
            "build": "construction",
            "fix": "repair",
            "create": "creation",
        }

        text_lower = record.raw_text.lower()
        for keyword, proc_type in procedure_keywords.items():
            if keyword in text_lower:
                links.append(
                    {
                        "target_id": f"procedure_type:{proc_type}",
                        "relationship_type": "type_of",
                        "properties": {"procedure_category": proc_type},
                    }
                )
                break

        # Extract tools/ingredients mentioned
        tool_pattern = (
            r"\b(?:using|with|need|requires?)\s+(?:a\s+)?([a-z]+(?:\s+[a-z]+)?)\b"
        )
        tools = re.findall(tool_pattern, text_lower)
        for tool in tools[:3]:
            links.append(
                {
                    "target_id": f"tool:{tool.replace(' ', '_')}",
                    "relationship_type": "requires",
                    "properties": {"requirement_type": "tool"},
                }
            )

        return links

    async def generate_summary(self, text: str) -> Optional[str]:
        """Generate summary focusing on the procedure goal"""
        # Look for "how to" or "steps to" phrases
        how_to_pattern = (
            r"(?:how to|steps to|guide to|instructions for)\s+(.+?)(?:[.!?]|$)"
        )
        match = re.search(how_to_pattern, text, re.IGNORECASE)
        if match:
            return f"How to {match.group(1).strip()}"

        # Fallback to first sentence
        sentences = text.split(".")
        if sentences:
            return sentences[0].strip() + "."

        return None
