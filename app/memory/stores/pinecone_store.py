import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import pinecone
from pinecone import Pinecone, ServerlessSpec

from app.memory.models.memory_entry import MemoryEntry
from app.memory.stores.base_store import VectorStore


class PineconeVectorStore(VectorStore):
    """Pinecone implementation of VectorStore"""

    def __init__(
        self,
        api_key: str,
        environment: str = "us-east-1-aws",
        index_name: str = "memory-index",
    ):
        """Initialize Pinecone client"""
        self.api_key = api_key
        self.environment = environment
        self.index_name = index_name
        self.pc = None
        self.index = None
        self.executor = ThreadPoolExecutor(max_workers=10)

    async def initialize(self):
        """Initialize Pinecone connection and index"""

        def _init_pinecone():
            # Initialize Pinecone
            self.pc = Pinecone(api_key=self.api_key)

            # Check if index exists, create if not
            existing_indexes = [index.name for index in self.pc.list_indexes()]

            if self.index_name not in existing_indexes:
                # Create index with 1536 dimensions (OpenAI embeddings)
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=self.environment),
                )

            # Connect to index
            self.index = self.pc.Index(self.index_name)

        # Run in thread pool since Pinecone client is sync
        await asyncio.get_event_loop().run_in_executor(self.executor, _init_pinecone)

    async def add_memory(self, memory_dict: Dict[str, Any]) -> bool:
        """Add memory from dictionary format (used by memory engine)"""
        if not self.index:
            await self.initialize()

        def _upsert():
            # Prepare metadata (Pinecone has size limits)
            # Store the actual input text for retrieval
            input_text = memory_dict.get("input") or memory_dict.get("text") or ""
            summary_text = memory_dict.get("summary") or ""

            metadata = {
                "memory_id": memory_dict.get("id"),
                "user_id": memory_dict.get("source_user_id")
                or memory_dict.get("meta", {}).get("user_id"),
                "session_id": memory_dict.get("source_session_id")
                or memory_dict.get("meta", {}).get("session_id"),
                "memory_type": memory_dict.get("memory_type"),
                "scope": memory_dict.get("scope"),
                "created_at": memory_dict.get("created_at"),
                "input": input_text[:1000]
                if input_text
                else None,  # Store actual input text (truncated for Pinecone limits)
                "summary": summary_text[:500] if summary_text else None,
                "tags": ",".join(memory_dict.get("tags", [])[:10]),
                "confidence": memory_dict.get("meta", {}).get("confidence_score"),
                "handler_used": memory_dict.get("meta", {}).get("handler_version"),
            }

            # Remove None values and convert datetime to string if needed
            clean_metadata = {}
            for k, v in metadata.items():
                if v is not None:
                    if hasattr(v, "isoformat"):
                        clean_metadata[k] = v.isoformat()
                    else:
                        clean_metadata[k] = str(v)

            # Get embedding
            embedding = memory_dict.get("embedding", [])
            if not embedding:
                print(f"Warning: No embedding found for memory {memory_dict.get('id')}")
                return False

            # Upsert to Pinecone
            self.index.upsert(
                vectors=[
                    {
                        "id": memory_dict.get("id"),
                        "values": embedding,
                        "metadata": clean_metadata,
                    }
                ]
            )
            return True

        try:
            # Run in thread pool since Pinecone client is sync
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, _upsert
            )
            return result
        except Exception as e:
            print(f"Error adding memory to Pinecone: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if Pinecone store is healthy"""
        try:
            if not self.index:
                return False

            # Test connection by getting index stats
            def _check_health():
                stats = self.index.describe_index_stats()
                return stats is not None

            result = await asyncio.get_event_loop().run_in_executor(
                self.executor, _check_health
            )
            return result

        except Exception as e:
            print(f"Pinecone health check failed: {e}")
            return False

    async def create(self, entry: MemoryEntry) -> str:
        """Create a new memory entry in Pinecone"""
        if not self.index:
            await self.initialize()

        def _upsert():
            # Prepare metadata (Pinecone has size limits)
            metadata = {
                "memory_id": entry.id,
                "user_id": entry.source_user_id,
                "session_id": entry.source_session_id,
                "memory_type": entry.memory_type.value,
                "scope": entry.scope,
                "created_at": entry.created_at.isoformat(),
                "cid": entry.cid,
                "summary": entry.summary[:500] if entry.summary else None,  # Limit size
                "tags": ",".join(entry.tags[:10]),  # Limit tags
                "confidence": entry.meta.confidence_score,
                "handler_used": entry.meta.handler_version,
            }

            # Remove None values
            metadata = {k: v for k, v in metadata.items() if v is not None}

            # Upsert to Pinecone
            self.index.upsert(
                vectors=[
                    {"id": entry.id, "values": entry.embedding, "metadata": metadata}
                ]
            )

            return entry.id

        return await asyncio.get_event_loop().run_in_executor(self.executor, _upsert)

    async def read(self, memory_id: str) -> Optional[MemoryEntry]:
        """Read a memory entry by ID"""
        if not self.index:
            await self.initialize()

        def _fetch():
            result = self.index.fetch(ids=[memory_id])
            if memory_id in result.vectors:
                vector_data = result.vectors[memory_id]
                metadata = vector_data.metadata

                # This is a simplified reconstruction - in practice you'd need
                # the full entry from DocumentStore and just use this for vector search
                return {
                    "id": memory_id,
                    "embedding": vector_data.values,
                    "metadata": metadata,
                }
            return None

        result = await asyncio.get_event_loop().run_in_executor(self.executor, _fetch)
        # Note: This returns simplified data. Full MemoryEntry should come from DocumentStore
        return result

    async def update(self, memory_id: str, updates: Dict[str, Any]) -> bool:
        """Update a memory entry"""
        if not self.index:
            await self.initialize()

        def _update():
            # First fetch the existing vector
            result = self.index.fetch(ids=[memory_id])
            if memory_id not in result.vectors:
                return False

            vector_data = result.vectors[memory_id]
            metadata = vector_data.metadata.copy()

            # Update metadata fields
            if "tags" in updates:
                metadata["tags"] = ",".join(updates["tags"][:10])
            if "summary" in updates:
                metadata["summary"] = (
                    updates["summary"][:500] if updates["summary"] else None
                )

            # Re-upsert with updated metadata
            self.index.upsert(
                vectors=[
                    {
                        "id": memory_id,
                        "values": vector_data.values,
                        "metadata": metadata,
                    }
                ]
            )
            return True

        return await asyncio.get_event_loop().run_in_executor(self.executor, _update)

    async def delete(self, memory_id: str) -> bool:
        """Delete a memory entry"""
        if not self.index:
            await self.initialize()

        def _delete():
            self.index.delete(ids=[memory_id])
            return True

        return await asyncio.get_event_loop().run_in_executor(self.executor, _delete)

    async def search(self, query: Dict[str, Any], limit: int = 10) -> List[MemoryEntry]:
        """Search for memory entries - not typically used for vector search"""
        # Vector search should use similarity_search instead
        return []

    async def similarity_search(
        self,
        embedding: List[float],
        limit: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[MemoryEntry, float]]:
        """Search by vector similarity"""
        if not self.index:
            await self.initialize()

        def _query():
            # Build Pinecone filter
            pinecone_filter = {}
            if filter:
                if "user_id" in filter:
                    pinecone_filter["user_id"] = {"$eq": filter["user_id"]}
                if "memory_type" in filter:
                    if "$in" in filter["memory_type"]:
                        pinecone_filter["memory_type"] = {
                            "$in": filter["memory_type"]["$in"]
                        }
                    else:
                        pinecone_filter["memory_type"] = {"$eq": filter["memory_type"]}

            # Query Pinecone
            query_result = self.index.query(
                vector=embedding,
                top_k=limit,
                include_metadata=True,
                filter=pinecone_filter if pinecone_filter else None,
            )

            results = []
            for match in query_result.matches:
                # Create simplified MemoryEntry from metadata
                # In practice, you'd fetch full entry from DocumentStore
                metadata = match.metadata

                # This is a placeholder - you'd reconstruct properly from DocumentStore
                memory_entry = {
                    "id": match.id,
                    "score": match.score,
                    "metadata": metadata,
                }

                results.append((memory_entry, match.score))

            return results

        return await asyncio.get_event_loop().run_in_executor(self.executor, _query)

    async def close(self):
        """Clean shutdown of Pinecone store"""
        if self.executor:
            self.executor.shutdown(wait=True)
            print("✅ Pinecone store executor shutdown")

        # Clear references
        self.index = None
        self.pc = None
        print("✅ Pinecone store closed")

    async def batch_insert(self, entries: List[MemoryEntry]) -> List[str]:
        """Batch insert memory entries with embeddings"""
        if not self.index:
            await self.initialize()

        def _batch_upsert():
            vectors = []
            for entry in entries:
                metadata = {
                    "memory_id": entry.id,
                    "user_id": entry.source_user_id,
                    "session_id": entry.source_session_id,
                    "memory_type": entry.memory_type.value,
                    "scope": entry.scope,
                    "created_at": entry.created_at.isoformat(),
                    "cid": entry.cid,
                    "summary": entry.summary[:500] if entry.summary else None,
                    "tags": ",".join(entry.tags[:10]),
                    "confidence": entry.meta.confidence_score,
                }

                # Remove None values
                metadata = {k: v for k, v in metadata.items() if v is not None}

                vectors.append(
                    {"id": entry.id, "values": entry.embedding, "metadata": metadata}
                )

            # Batch upsert (Pinecone handles batching automatically)
            self.index.upsert(vectors=vectors)

            return [entry.id for entry in entries]

        return await asyncio.get_event_loop().run_in_executor(
            self.executor, _batch_upsert
        )

    async def close(self):
        """Close Pinecone connections"""
        if self.executor:
            self.executor.shutdown(wait=True)
        # Pinecone client doesn't need explicit closing
