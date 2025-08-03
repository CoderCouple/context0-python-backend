"""
Asynchronous tag extraction service using LLM
"""
import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.service.llm_service import LLMService

logger = logging.getLogger(__name__)


class TagExtractionService:
    """Service for extracting intelligent tags from memories using LLM"""

    def __init__(self, llm_service: LLMService, db: AsyncIOMotorDatabase):
        self.llm_service = llm_service
        self.db = db
        self.memories_collection = db.memories
        self.tag_queue_collection = db.tag_extraction_queue

    async def queue_for_tag_extraction(self, memory_ids: List[str], user_id: str):
        """Queue memories for async tag extraction"""
        try:
            # Add to queue for processing
            queue_items = [
                {
                    "memory_id": memory_id,
                    "user_id": user_id,
                    "queued_at": datetime.utcnow(),
                    "status": "pending",
                }
                for memory_id in memory_ids
            ]

            if queue_items:
                await self.tag_queue_collection.insert_many(queue_items)
                logger.info(f"Queued {len(queue_items)} memories for tag extraction")

                # Start async processing without waiting
                asyncio.create_task(self._process_tag_queue(user_id))

        except Exception as e:
            logger.error(f"Error queuing tags: {e}")

    async def extract_tags_for_memory(
        self, memory_content: str, existing_tags: List[str]
    ) -> List[str]:
        """Extract intelligent tags using LLM"""

        prompt = f"""Analyze this memory and generate relevant tags that would help find it later.

Memory: "{memory_content}"
Current tags: {existing_tags}

Generate 5-10 additional semantic tags that capture:
1. Topics and themes
2. Categories (food, work, personal, etc.)
3. Specific entities mentioned
4. Emotions or sentiments
5. Actions or intentions

Return ONLY a JSON array of tags, no explanation:
["tag1", "tag2", "tag3", ...]"""

        try:
            # Use a fast model for tag extraction
            messages = [
                {
                    "role": "system",
                    "content": "You are a tag extraction specialist. Generate concise, relevant tags.",
                },
                {"role": "user", "content": prompt},
            ]

            # Create a lightweight preset for tagging
            response = await self.llm_service.generate(
                messages=messages,
                provider="openai",
                model="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=200,
            )

            # Parse tags from response
            import json

            try:
                new_tags = json.loads(response.content)
                if isinstance(new_tags, list):
                    # Clean and deduplicate tags
                    all_tags = list(
                        set(existing_tags + [tag.lower().strip() for tag in new_tags])
                    )
                    return all_tags[:20]  # Limit to 20 tags
            except:
                logger.warning(
                    f"Failed to parse tags from LLM response: {response.content}"
                )

        except Exception as e:
            logger.error(f"Error extracting tags: {e}")

        return existing_tags

    async def _process_tag_queue(self, user_id: str):
        """Process queued memories for tag extraction"""
        try:
            # Get pending items from queue
            pending_items = (
                await self.tag_queue_collection.find(
                    {"user_id": user_id, "status": "pending"}
                )
                .limit(10)
                .to_list(None)
            )

            for item in pending_items:
                try:
                    # Get memory
                    memory = await self.memories_collection.find_one(
                        {"id": item["memory_id"]}
                    )

                    if memory:
                        # Extract new tags
                        content = memory.get("input", "") or memory.get("summary", "")
                        current_tags = memory.get("tags", [])

                        new_tags = await self.extract_tags_for_memory(
                            content, current_tags
                        )

                        # Update memory with new tags
                        if len(new_tags) > len(current_tags):
                            await self.memories_collection.update_one(
                                {"id": item["memory_id"]},
                                {
                                    "$set": {
                                        "tags": new_tags,
                                        "tags_enriched": True,
                                        "tags_enriched_at": datetime.utcnow(),
                                    }
                                },
                            )
                            logger.info(
                                f"Updated tags for memory {item['memory_id']}: {len(current_tags)} -> {len(new_tags)}"
                            )

                    # Mark as processed
                    await self.tag_queue_collection.update_one(
                        {"_id": item["_id"]},
                        {
                            "$set": {
                                "status": "completed",
                                "processed_at": datetime.utcnow(),
                            }
                        },
                    )

                except Exception as e:
                    logger.error(
                        f"Error processing memory {item.get('memory_id')}: {e}"
                    )
                    # Mark as failed
                    await self.tag_queue_collection.update_one(
                        {"_id": item["_id"]},
                        {"$set": {"status": "failed", "error": str(e)}},
                    )

        except Exception as e:
            logger.error(f"Error processing tag queue: {e}")

    async def enrich_existing_memories(self, user_id: str, limit: int = 50):
        """Enrich existing memories that don't have LLM-generated tags"""

        # Find memories without enriched tags
        memories = (
            await self.memories_collection.find(
                {"source_user_id": user_id, "tags_enriched": {"$ne": True}}
            )
            .limit(limit)
            .to_list(None)
        )

        logger.info(f"Found {len(memories)} memories to enrich with tags")

        memory_ids = [m["id"] for m in memories]
        if memory_ids:
            await self.queue_for_tag_extraction(memory_ids, user_id)
