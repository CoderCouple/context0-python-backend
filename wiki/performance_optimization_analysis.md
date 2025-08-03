# Context0 Memory System - Performance Optimization Analysis

## Executive Summary
This document provides a comprehensive analysis of performance optimization opportunities in the Context0 Memory System, focusing on maintaining correctness while improving response times and throughput.

## Current Performance Bottlenecks

### 1. Database Operations
- **Issue**: Multiple sequential database queries in chat and memory operations
- **Impact**: High latency for memory search and retrieval

### 2. LLM API Calls
- **Issue**: Sequential LLM calls for memory extraction and chat responses
- **Impact**: 2-5 second latency per request

### 3. Embedding Generation
- **Issue**: Real-time embedding generation for search queries
- **Impact**: 100-500ms added latency

### 4. Memory Store Initialization
- **Issue**: All stores initialized on startup regardless of usage
- **Impact**: Slower startup time, unnecessary resource allocation

## Optimization Recommendations

### 1. Database Query Optimizations

#### A. Add MongoDB Indexes
```python
# Add to MongoDB initialization
async def create_indexes():
    # Composite index for user + timestamp queries
    await memories_collection.create_index([
        ("source_user_id", 1),
        ("created_at", -1)
    ])
    
    # Text index for content search
    await memories_collection.create_index([
        ("input", "text"),
        ("summary", "text")
    ])
    
    # Index for memory type filtering
    await memories_collection.create_index([
        ("memory_type", 1),
        ("source_user_id", 1)
    ])
    
    # Chat sessions index
    await chat_sessions.create_index([
        ("user_id", 1),
        ("updated_at", -1)
    ])
```

#### B. Implement Query Batching
```python
# Instead of multiple find_one calls
async def batch_get_memories(memory_ids: List[str], user_id: str):
    memories = await memories_collection.find({
        "id": {"$in": memory_ids},
        "source_user_id": user_id
    }).to_list(None)
    return {m["id"]: m for m in memories}
```

### 2. Caching Layer Implementation

#### A. In-Memory Cache for Frequent Data
```python
from typing import Dict, Any, Optional
import asyncio
from datetime import datetime, timedelta

class MemoryCache:
    def __init__(self, ttl_seconds: int = 300):
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if datetime.utcnow() - timestamp < self._ttl:
                    return value
                else:
                    del self._cache[key]
        return None
    
    async def set(self, key: str, value: Any):
        async with self._lock:
            self._cache[key] = (value, datetime.utcnow())
    
    async def invalidate(self, pattern: str = None):
        async with self._lock:
            if pattern:
                keys_to_delete = [k for k in self._cache.keys() if pattern in k]
                for key in keys_to_delete:
                    del self._cache[key]
            else:
                self._cache.clear()

# Usage in memory service
class MemoryService:
    def __init__(self):
        self._cache = MemoryCache(ttl_seconds=300)  # 5 minute cache
        
    async def get_memory(self, memory_id: str, user_id: str):
        cache_key = f"memory:{user_id}:{memory_id}"
        
        # Check cache first
        cached = await self._cache.get(cache_key)
        if cached:
            return cached
        
        # Fetch from database
        memory = await self._fetch_from_db(memory_id, user_id)
        
        # Cache the result
        if memory:
            await self._cache.set(cache_key, memory)
        
        return memory
```

#### B. Redis Cache for Distributed Systems
```python
import redis.asyncio as redis
import json

class RedisCache:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(redis_url)
    
    async def get_embeddings(self, text: str) -> Optional[List[float]]:
        key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
        cached = await self.redis.get(key)
        if cached:
            return json.loads(cached)
        return None
    
    async def set_embeddings(self, text: str, embeddings: List[float]):
        key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
        await self.redis.setex(key, 3600, json.dumps(embeddings))  # 1 hour TTL
```

### 3. Parallel Processing Enhancements

#### A. Parallel Memory Search
```python
async def enhanced_search_memories(self, query: SearchQuery):
    # Execute all store searches in parallel
    tasks = []
    
    if self.vector_store:
        tasks.append(self._search_vector_store(query))
    
    if self.graph_store:
        tasks.append(self._search_graph_store(query))
    
    if self.doc_store:
        tasks.append(self._search_doc_store(query))
    
    # Wait for all searches to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Merge and rank results
    all_results = []
    for result in results:
        if not isinstance(result, Exception):
            all_results.extend(result)
    
    # Sort by score and deduplicate
    return self._merge_search_results(all_results)
```

#### B. Batch Embedding Generation
```python
async def batch_generate_embeddings(self, texts: List[str]) -> List[List[float]]:
    # Check cache for existing embeddings
    cache_keys = [f"emb:{hashlib.md5(t.encode()).hexdigest()}" for t in texts]
    cached_results = await asyncio.gather(
        *[self._cache.get(key) for key in cache_keys]
    )
    
    # Identify texts that need embedding
    texts_to_embed = []
    indices_to_embed = []
    for i, (text, cached) in enumerate(zip(texts, cached_results)):
        if cached is None:
            texts_to_embed.append(text)
            indices_to_embed.append(i)
    
    # Generate embeddings for uncached texts
    if texts_to_embed:
        new_embeddings = await self._openai_batch_embed(texts_to_embed)
        
        # Cache new embeddings
        cache_tasks = []
        for text, embedding in zip(texts_to_embed, new_embeddings):
            key = f"emb:{hashlib.md5(text.encode()).hexdigest()}"
            cache_tasks.append(self._cache.set(key, embedding))
        await asyncio.gather(*cache_tasks)
    
    # Combine cached and new results
    final_embeddings = list(cached_results)
    for i, idx in enumerate(indices_to_embed):
        final_embeddings[idx] = new_embeddings[i]
    
    return final_embeddings
```

### 4. Connection Pooling and Resource Management

#### A. MongoDB Connection Pool
```python
# Update mongodb.py
async def connect_to_mongodb():
    global _mongodb_client, _mongodb
    
    mongodb_url = get_mongodb_url()
    _mongodb_client = AsyncIOMotorClient(
        mongodb_url,
        maxPoolSize=100,
        minPoolSize=10,
        maxIdleTimeMS=30000,
        waitQueueTimeoutMS=5000,
        serverSelectionTimeoutMS=5000
    )
    _mongodb = _mongodb_client[get_mongodb_name()]
```

#### B. HTTP Connection Pooling for LLM APIs
```python
import httpx

class LLMConnectionPool:
    def __init__(self):
        self._clients = {}
    
    def get_client(self, provider: str) -> httpx.AsyncClient:
        if provider not in self._clients:
            self._clients[provider] = httpx.AsyncClient(
                limits=httpx.Limits(
                    max_connections=100,
                    max_keepalive_connections=20,
                    keepalive_expiry=30
                ),
                timeout=httpx.Timeout(30.0, pool=5.0)
            )
        return self._clients[provider]
    
    async def close_all(self):
        for client in self._clients.values():
            await client.aclose()
```

### 5. Lazy Loading and Initialization

#### A. Lazy Store Initialization
```python
class MemoryEngine:
    def __init__(self):
        self._stores = {}
        self._store_configs = {}
        self._initialized_stores = set()
    
    async def _get_vector_store(self) -> VectorStore:
        if "vector" not in self._initialized_stores:
            await self._initialize_vector_store()
            self._initialized_stores.add("vector")
        return self._stores.get("vector")
    
    async def _initialize_vector_store(self):
        # Only initialize when actually needed
        config = self._store_configs.get("vector")
        if config and config.get("provider") == "pinecone":
            from app.memory.stores.pinecone_store import PineconeVectorStore
            self._stores["vector"] = PineconeVectorStore(**config)
            await self._stores["vector"].initialize()
```

### 6. Streaming and Pagination

#### A. Cursor-based Pagination
```python
async def get_memories_paginated(
    self,
    user_id: str,
    cursor: Optional[str] = None,
    limit: int = 20
) -> Dict[str, Any]:
    query = {"source_user_id": user_id}
    
    if cursor:
        # Decode cursor to get last item's created_at
        last_timestamp = decode_cursor(cursor)
        query["created_at"] = {"$lt": last_timestamp}
    
    memories = await self.memories_collection.find(query)\
        .sort("created_at", -1)\
        .limit(limit + 1)\
        .to_list(None)
    
    has_more = len(memories) > limit
    if has_more:
        memories = memories[:-1]
    
    next_cursor = None
    if has_more and memories:
        next_cursor = encode_cursor(memories[-1]["created_at"])
    
    return {
        "memories": memories,
        "next_cursor": next_cursor,
        "has_more": has_more
    }
```

### 7. Optimize Memory Extraction

#### A. Selective Extraction
```python
async def smart_extract_memories(self, content: str, context: Dict):
    # Quick heuristic check before LLM call
    if len(content.split()) < 10:  # Too short
        return {"extracted": {}, "stored": {}}
    
    # Check if content is likely to contain memorable information
    keywords = ["remember", "favorite", "always", "never", "love", "hate"]
    if not any(keyword in content.lower() for keyword in keywords):
        # Use simpler extraction
        return await self._simple_extraction(content, context)
    
    # Full LLM extraction for complex content
    return await self._full_llm_extraction(content, context)
```

### 8. Background Task Processing

#### A. Async Task Queue
```python
from asyncio import Queue, create_task

class BackgroundTaskProcessor:
    def __init__(self):
        self._queue = Queue()
        self._workers = []
        self._running = False
    
    async def start(self, num_workers: int = 3):
        self._running = True
        self._workers = [
            create_task(self._worker(f"worker-{i}"))
            for i in range(num_workers)
        ]
    
    async def _worker(self, name: str):
        while self._running:
            try:
                task = await asyncio.wait_for(
                    self._queue.get(), 
                    timeout=1.0
                )
                await task()
            except asyncio.TimeoutError:
                continue
    
    async def enqueue(self, task):
        await self._queue.put(task)
    
    async def stop(self):
        self._running = False
        await asyncio.gather(*self._workers)

# Usage for background memory processing
async def process_memory_async(memory_data: Dict):
    processor = BackgroundTaskProcessor()
    await processor.start()
    
    # Enqueue non-critical tasks
    await processor.enqueue(
        lambda: enrich_memory_with_tags(memory_data)
    )
    await processor.enqueue(
        lambda: update_memory_graph(memory_data)
    )
```

## Implementation Priority

1. **High Priority** (Immediate impact, low risk):
   - MongoDB indexes
   - In-memory caching
   - Connection pooling
   - Parallel memory search

2. **Medium Priority** (Good impact, moderate effort):
   - Batch embedding generation
   - Lazy initialization
   - Query batching
   - Background task processing

3. **Low Priority** (Nice to have):
   - Redis caching
   - Cursor pagination
   - Smart extraction heuristics

## Performance Metrics to Monitor

1. **Response Times**:
   - Chat response latency (target: < 2s)
   - Memory search latency (target: < 500ms)
   - Memory extraction time (target: < 1s)

2. **Throughput**:
   - Requests per second
   - Concurrent users supported
   - Memory operations per minute

3. **Resource Usage**:
   - Database connection pool usage
   - Memory consumption
   - CPU utilization

4. **Cache Performance**:
   - Cache hit rate (target: > 80%)
   - Cache eviction rate
   - Average cache entry age

## Testing Strategy

1. **Load Testing**:
   ```bash
   # Using locust
   locust -f load_tests/chat_load_test.py --host=http://localhost:8000
   ```

2. **Performance Profiling**:
   ```python
   import cProfile
   import pstats
   
   profiler = cProfile.Profile()
   profiler.enable()
   # Run operations
   profiler.disable()
   stats = pstats.Stats(profiler)
   stats.sort_stats('cumulative')
   stats.print_stats(20)
   ```

3. **Database Query Analysis**:
   ```python
   # Enable MongoDB profiling
   await db.command("profile", 2, slowms=100)
   ```

## Conclusion

These optimizations can significantly improve system performance while maintaining correctness. Start with high-priority items for immediate impact, then progressively implement medium and low-priority optimizations based on measured performance gains.