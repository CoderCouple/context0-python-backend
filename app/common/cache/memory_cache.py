"""
In-memory cache implementation for frequently accessed data
"""
import asyncio
from typing import Dict, Any, Optional, TypeVar, Generic
from datetime import datetime, timedelta
import hashlib
import json

T = TypeVar("T")


class MemoryCache(Generic[T]):
    """Thread-safe in-memory cache with TTL support"""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self._cache: Dict[str, tuple[T, datetime]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._access_count: Dict[str, int] = {}

    async def get(self, key: str) -> Optional[T]:
        """Get value from cache"""
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if datetime.utcnow() - timestamp < self._ttl:
                    # Update access count for LRU
                    self._access_count[key] = self._access_count.get(key, 0) + 1
                    return value
                else:
                    # Expired, remove it
                    del self._cache[key]
                    if key in self._access_count:
                        del self._access_count[key]
        return None

    async def set(self, key: str, value: T) -> None:
        """Set value in cache with TTL"""
        async with self._lock:
            # Check if we need to evict items
            if len(self._cache) >= self._max_size and key not in self._cache:
                await self._evict_lru()

            self._cache[key] = (value, datetime.utcnow())
            self._access_count[key] = 1

    async def delete(self, key: str) -> bool:
        """Delete specific key from cache"""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                if key in self._access_count:
                    del self._access_count[key]
                return True
        return False

    async def clear(self) -> None:
        """Clear entire cache"""
        async with self._lock:
            self._cache.clear()
            self._access_count.clear()

    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        async with self._lock:
            keys_to_delete = [k for k in self._cache.keys() if pattern in k]
            for key in keys_to_delete:
                del self._cache[key]
                if key in self._access_count:
                    del self._access_count[key]
            return len(keys_to_delete)

    async def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if not self._cache:
            return

        # Find LRU key
        lru_key = min(
            self._access_count.keys(), key=lambda k: self._access_count.get(k, 0)
        )

        # Remove it
        if lru_key in self._cache:
            del self._cache[lru_key]
        if lru_key in self._access_count:
            del self._access_count[lru_key]

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        async with self._lock:
            total_items = len(self._cache)
            expired_items = 0
            now = datetime.utcnow()

            for key, (_, timestamp) in self._cache.items():
                if now - timestamp >= self._ttl:
                    expired_items += 1

            return {
                "total_items": total_items,
                "expired_items": expired_items,
                "max_size": self._max_size,
                "ttl_seconds": self._ttl.total_seconds(),
                "hit_count": sum(self._access_count.values()),
            }


class EmbeddingCache(MemoryCache[list]):
    """Specialized cache for embeddings"""

    def get_key(self, text: str) -> str:
        """Generate cache key for text"""
        return f"emb:{hashlib.md5(text.encode()).hexdigest()}"

    async def get_embedding(self, text: str) -> Optional[list]:
        """Get embedding from cache"""
        key = self.get_key(text)
        return await self.get(key)

    async def set_embedding(self, text: str, embedding: list) -> None:
        """Cache embedding"""
        key = self.get_key(text)
        await self.set(key, embedding)


class QueryResultCache(MemoryCache[dict]):
    """Specialized cache for query results"""

    def get_key(self, user_id: str, query: str, filters: Optional[dict] = None) -> str:
        """Generate cache key for query"""
        key_parts = [user_id, query]
        if filters:
            key_parts.append(json.dumps(filters, sort_keys=True))

        key_str = "|".join(key_parts)
        return f"query:{hashlib.md5(key_str.encode()).hexdigest()}"

    async def get_results(
        self, user_id: str, query: str, filters: Optional[dict] = None
    ) -> Optional[dict]:
        """Get cached query results"""
        key = self.get_key(user_id, query, filters)
        return await self.get(key)

    async def set_results(
        self, user_id: str, query: str, results: dict, filters: Optional[dict] = None
    ) -> None:
        """Cache query results"""
        key = self.get_key(user_id, query, filters)
        await self.set(key, results)

    async def invalidate_user(self, user_id: str) -> int:
        """Invalidate all cached queries for a user"""
        # Since we hash the keys, we can't directly match by user_id
        # In a production system, we'd maintain a reverse index
        # For now, we'll clear queries that might be affected
        return await self.invalidate_pattern(f"query:")


# Global cache instances
embedding_cache = EmbeddingCache(ttl_seconds=3600, max_size=5000)  # 1 hour TTL
query_cache = QueryResultCache(ttl_seconds=300, max_size=1000)  # 5 minute TTL
memory_cache = MemoryCache[dict](ttl_seconds=600, max_size=2000)  # 10 minute TTL
