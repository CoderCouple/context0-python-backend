# Context0 Memory System - Complete Technical Documentation

## Table of Contents

1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Memory Processing Pipeline](#memory-processing-pipeline)
4. [Multi-Hop Reasoning Engine](#multi-hop-reasoning-engine)
5. [Advanced Query Optimization](#advanced-query-optimization)
6. [Store Integration](#store-integration)
7. [API Reference](#api-reference)
8. [Performance Metrics](#performance-metrics)
9. [Troubleshooting Guide](#troubleshooting-guide)
10. [Development Workflow](#development-workflow)

---

## Overview

The Context0 Memory System is an advanced AI-powered memory management platform that processes, stores, and retrieves human memories using multiple specialized storage systems and sophisticated reasoning capabilities.

### Key Features

- **Multi-Store Architecture**: Vector, Graph, Document, TimeSeries, and Audit stores
- **Advanced Reasoning Engine**: Multi-hop reasoning with query expansion and precision matching
- **Memory Type Classification**: Semantic, Episodic, Procedural, Meta, Emotional, Working, Declarative
- **Real-time Processing**: Async/await architecture with concurrent processing
- **Query Optimization**: Advanced search with diversity scoring and relevance optimization
- **Time-Travel Debugging**: Historical state reconstruction and temporal queries

### Technical Stack

- **Backend**: FastAPI (Python 3.11+)
- **Vector Store**: Pinecone (cloud-based semantic search)
- **Graph Store**: Neo4j (relationship mapping)
- **Document Store**: MongoDB (structured storage)
- **TimeSeries Store**: TimescaleDB (temporal data)
- **Audit Store**: MongoDB (operation logging)
- **LLM Integration**: OpenAI GPT-4 / Anthropic Claude
- **Embeddings**: OpenAI text-embedding-3-small

---

## System Architecture

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    Context0 Memory System                       │
├─────────────────────────────────────────────────────────────────┤
│                        FastAPI Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Memory API  │  │ Reasoning   │  │ Search API  │            │
│  │   Endpoints │  │    API      │  │  Endpoints  │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
├─────────────────────────────────────────────────────────────────┤
│                      Business Logic Layer                      │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │                 Memory Engine                               │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │Memory Type  │  │   Memory    │  │   Memory    │        │ │
│  │  │ Inferencer  │  │   Router    │  │   Handler   │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              Multi-Hop Reasoning Engine                    │ │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │ │
│  │  │   Query     │  │  Advanced   │  │  Memory     │        │ │
│  │  │ Expansion   │  │  Scoring    │  │ Diversity   │        │ │
│  │  └─────────────┘  └─────────────┘  └─────────────┘        │ │
│  └─────────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────────┤
│                        Storage Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐            │
│  │ Vector Store│  │ Graph Store │  │ Document    │            │
│  │  (Pinecone) │  │   (Neo4j)   │  │Store(Mongo) │            │
│  └─────────────┘  └─────────────┘  └─────────────┘            │
│  ┌─────────────┐  ┌─────────────┐                             │
│  │ TimeSeries  │  │ Audit Store │                             │
│  │(TimescaleDB)│  │  (MongoDB)  │                             │
│  └─────────────┘  └─────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

### Component Interaction Flow

```
User Query
    │
    ▼
┌─────────────────┐
│   FastAPI       │
│   Endpoint      │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐     ┌─────────────────┐
│ Multi-Hop       │────▶│   Memory        │
│ Reasoning       │     │   Engine        │
│ Engine          │     │                 │
└─────┬───────────┘     └─────┬───────────┘
      │                       │
      ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Query Expansion │     │ Memory Type     │
│ & Optimization  │     │ Classification  │
└─────┬───────────┘     └─────┬───────────┘
      │                       │
      ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│ Multi-Store     │     │ Store-Specific  │
│ Search          │     │ Handlers        │
└─────┬───────────┘     └─────┬───────────┘
      │                       │
      ▼                       ▼
┌─────────────────────────────────────────┐
│            Storage Layer                │
│  Vector │ Graph │ Document │ TimeSeries │
└─────────┬───────────────────────────────┘
          │
          ▼
    ┌─────────────────┐
    │    Response     │
    │   Synthesis     │
    └─────────────────┘
```

---

## Memory Processing Pipeline

### Memory Lifecycle

```
Raw Memory Input
       │
       ▼
┌─────────────────┐
│  Input          │  ← User provides text, metadata, context
│  Validation     │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│  Memory Type    │  ← AI-powered classification
│  Inference      │    (Semantic, Episodic, etc.)
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│  Memory         │  ← Route to appropriate handler
│  Routing        │    based on type and content
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│  Handler        │  ← Type-specific processing
│  Processing     │    (embeddings, relationships)
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│  Multi-Store    │  ← Concurrent storage across
│  Storage        │    all configured stores
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│  Audit &        │  ← Log operations and
│  Monitoring     │    update statistics
└─────────────────┘
```

### Memory Entry Structure

```json
{
  "id": "memory_uuid",
  "cid": "content_identifier", 
  "input": "original_text",
  "summary": "ai_generated_summary",
  "memory_type": "semantic_memory",
  "tags": ["tag1", "tag2"],
  "scope": "user:john-doe",
  "embedding": [0.1, 0.2, ...],
  "embedding_model": "text-embedding-3-small",
  "graph_links": [
    {
      "target_id": "related_memory_id",
      "relationship_type": "caused_by",
      "properties": {}
    }
  ],
  "created_at": "2025-07-15T12:00:00Z",
  "source_user_id": "john-doe",
  "source_session_id": "session-123",
  "meta": {
    "confidence_score": 0.95,
    "processing_time_ms": 150,
    "handler_version": "1.0.0"
  }
}
```

---

## Multi-Hop Reasoning Engine

### Reasoning Architecture

The Multi-Hop Reasoning Engine is the core intelligence component that connects disparate memories to answer complex questions.

#### Reasoning Pipeline

```
Complex Question Input
         │
         ▼
┌─────────────────┐
│   Question      │  ← Analyze question type, entities,
│   Analysis      │    temporal indicators, complexity
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│   Query         │  ← Expand with synonyms, related
│   Expansion     │    terms, and domain-specific
└─────┬───────────┘    keywords
      │
      ▼
┌─────────────────┐
│   Context       │  ← Build multiple context windows
│   Window        │    with different memory clusters
│   Building      │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│   Reasoning     │  ← Generate multiple reasoning
│   Chain         │    chains with different approaches
│   Generation    │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│   Chain         │  ← Evaluate and rank chains by
│   Evaluation    │    confidence and coherence
│   & Ranking     │
└─────┬───────────┘
      │
      ▼
┌─────────────────┐
│   Answer        │  ← Synthesize final answer from
│   Synthesis     │    best reasoning chains
└─────────────────┘
```

### Query Expansion Strategy

#### Level 1: Keyword Expansion
```python
keyword_expansions = {
    "childhood": ["youth", "early life", "growing up", "young", "child"],
    "curiosity": ["interest", "fascination", "wonder", "exploration"],
    "career": ["job", "work", "professional", "occupation"],
    "mentorship": ["mentor", "teaching", "guidance", "Dr. Chen"]
}
```

#### Level 2: Entity-Specific Expansion
```python
# For Lisa + security questions:
["tech conference mobile security meeting",
 "marriage wedding Napa Valley relationship"]

# For mentorship questions:
["Dr. Chen professor teaching guidance machine learning",
 "coding bootcamp teaching volunteers students"]
```

#### Level 3: Multi-Domain Integration
```python
# For creative + technical questions:
["guitar photography creative technical balance",
 "artistic pursuits engineering problem solving"]
```

### Advanced Scoring Mechanisms

#### Composite Score Calculation
```python
weights = {
    "primary_score": 0.25,        # Base semantic similarity
    "query_specificity": 0.20,    # How specifically it matches query
    "keyword_match": 0.20,        # Keyword relevance
    "memory_type_relevance": 0.15, # Memory type appropriateness
    "confidence": 0.10,           # Memory confidence
    "tag_relevance": 0.05,        # Tag matching
    "recency": 0.05,             # Temporal relevance
}
```

#### Memory Diversity Optimization
```python
tag_categories = {
    "family": ["family", "parents", "sister", "daughter"],
    "career": ["career", "work", "job", "google", "amazon"],
    "education": ["education", "university", "professor"],
    "relationships": ["wife", "lisa", "friend", "alex"],
    "skills": ["programming", "algorithms", "leadership"],
    "hobbies": ["guitar", "photography", "hiking"],
    "personal": ["reflection", "growth", "values"]
}
```

---

## Advanced Query Optimization

### Query Processing Flow

```
Original Query: "How did my childhood curiosity influence my career choice?"
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Query Analysis                           │
│  • Type: causal                                            │
│  • Entities: ["childhood", "curiosity", "career"]         │
│  • Complexity: medium                                      │
│  • Expected hops: 2-3                                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                  Query Expansion                            │
│  1. "How did my youth interest influence my job choice?"    │
│  2. "How did my fascination exploration influence work?"    │
│  3. "electronics curiosity childhood technical career"     │
│  4. "childhood technical curiosity career success"         │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                Multi-Query Search                           │
│  • Query 1 → 5 memories (childhood memories)              │
│  • Query 2 → 3 memories (career progression)              │
│  • Query 3 → 4 memories (technical background)            │
│  • Query 4 → 2 memories (reflection memories)             │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│               Memory Optimization                           │
│  • Remove duplicates by memory_id                         │
│  • Calculate advanced composite scores                     │
│  • Apply diversity constraints                            │
│  • Select top N with memory type diversity                │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
                Final Result:
                "I've learned that my technical curiosity from 
                 childhood directly led to my career success"
```

### Memory Selection Algorithm

```python
def optimize_memory_selection(memories, query, limit):
    # Step 1: Remove duplicates, keep highest scoring
    unique_memories = deduplicate_by_id(memories)
    
    # Step 2: Sort by composite score
    sorted_memories = sort_by_score(unique_memories)
    
    # Step 3: Apply diversity constraints
    selected = []
    used_types = set()
    used_categories = set()
    
    for memory in sorted_memories:
        if len(selected) >= limit:
            break
            
        # Diversity scoring
        diversity_bonus = 1.0
        
        if memory.type not in used_types:
            diversity_bonus += 0.1
            used_types.add(memory.type)
            
        if memory.category not in used_categories:
            diversity_bonus += 0.1
            used_categories.add(memory.category)
        
        memory.final_score = memory.composite_score * diversity_bonus
        selected.append(memory)
    
    return sort_by_final_score(selected)
```

---

## Store Integration

### Vector Store (Pinecone)

**Purpose**: Semantic similarity search using embeddings

```python
class PineconeVectorStore:
    async def similarity_search(self, embedding, limit, filter):
        # Search by cosine similarity
        results = self.index.query(
            vector=embedding,
            top_k=limit,
            include_metadata=True,
            filter=filter
        )
        return [(result.metadata, result.score) for result in results]
```

### Graph Store (Neo4j)

**Purpose**: Relationship mapping and graph traversal

```cypher
-- Create memory nodes and relationships
CREATE (m:Memory {id: $memory_id, type: $memory_type})
CREATE (m1)-[:CAUSED_BY]->(m2)
CREATE (m1)-[:RELATES_TO]->(m3)

-- Find connected memories
MATCH (m:Memory {id: $memory_id})-[r*1..3]-(connected:Memory)
RETURN connected, r
```

### Document Store (MongoDB)

**Purpose**: Structured data storage and text search

```javascript
// Memory document structure
{
  _id: ObjectId,
  id: "memory_uuid",
  user_id: "john-doe",
  input: "memory text content",
  memory_type: "semantic_memory",
  tags: ["family", "career"],
  created_at: ISODate,
  metadata: { ... }
}

// Text search capabilities
db.memories.find({
  $text: { $search: "childhood curiosity career" },
  user_id: "john-doe"
})
```

### TimeSeries Store (TimescaleDB)

**Purpose**: Temporal data analysis and time-travel queries

```sql
-- Hypertable for time-series memory data
CREATE TABLE memory_timeseries (
    time TIMESTAMPTZ NOT NULL,
    memory_id UUID,
    user_id VARCHAR(255),
    memory_type VARCHAR(100),
    tags TEXT[],
    content TEXT
);

-- Time-travel query
SELECT * FROM memory_timeseries 
WHERE user_id = 'john-doe' 
  AND time <= '2025-01-01'
ORDER BY time DESC;
```

### Audit Store (MongoDB)

**Purpose**: Operation logging and system monitoring

```javascript
// Audit log entry
{
  id: "audit_uuid",
  action: "ADD",
  memory_id: "memory_uuid", 
  user_id: "john-doe",
  timestamp: ISODate,
  before_state: null,
  after_state: { ... },
  processing_time_ms: 150
}
```

---

## API Reference

### Memory Operations

#### Create Memory
```http
POST /api/v1/memories
Content-Type: application/json

{
  "user_id": "john-doe",
  "session_id": "session-123",
  "text": "I learned Python programming in college",
  "memory_type": "semantic_memory",
  "tags": ["education", "programming"],
  "scope": "education"
}
```

#### Search Memories
```http
POST /api/v1/memories/search
Content-Type: application/json

{
  "user_id": "john-doe",
  "query": "programming skills",
  "limit": 10,
  "threshold": 0.3,
  "include_content": true
}
```

### Reasoning Operations

#### Ask Question (Multi-Hop Reasoning)
```http
POST /api/v1/ask
Content-Type: application/json

{
  "question": "How did my childhood curiosity influence my career?",
  "user_id": "john-doe",
  "session_id": "session-123",
  "max_memories": 15,
  "search_depth": "comprehensive"
}
```

#### Response Format
```json
{
  "success": true,
  "result": {
    "question": "How did my childhood curiosity influence my career?",
    "answer": "I've learned that my technical curiosity from childhood directly led to my career success",
    "confidence": 0.68,
    "memories_found": 12,
    "memories_used": 5,
    "memory_contexts": [
      {
        "memory_id": "uuid-1",
        "content": "Growing up, I was always fascinated by electronics",
        "relevance_score": 0.92
      }
    ],
    "reasoning_chains": 1,
    "processing_time_ms": 530
  }
}
```

---

## Performance Metrics

### System Performance

```
┌─────────────────────┬─────────────────┬─────────────────┐
│      Metric         │   Before Opt.   │   After Opt.    │
├─────────────────────┼─────────────────┼─────────────────┤
│ Pass Rate (Level 1) │      50%        │      100%       │
│ Pass Rate (Level 2) │       0%        │       0%        │
│ Pass Rate (Overall) │      10%        │       20%       │
│ Avg Response Time   │    2.36s        │     2.61s       │
│ Memory Retrieval    │      5.0        │      5.0        │
│ Confidence Score    │     0.68        │     0.68        │
│ Keyword Coverage    │   0-60%         │   16-100%       │
└─────────────────────┴─────────────────┴─────────────────┘
```

### Memory Store Health

```
Vector Store (Pinecone):    ✅ Connected, 1,000+ vectors
Graph Store (Neo4j):        ✅ Connected, 500+ nodes
Document Store (MongoDB):   ✅ Connected, 111 documents  
TimeSeries (TimescaleDB):   ✅ Connected, time-series data
Audit Store (MongoDB):      ✅ Connected, full audit trail
```

---

## Troubleshooting Guide

### Common Issues

#### 1. Pinecone Connection Issues
**Symptoms**: Vector store showing as `false` in health check
**Solutions**:
- Verify `PINECONE_API_KEY` environment variable
- Check `PINECONE_ENVIRONMENT` setting
- Ensure `pinecone` package is installed (not `pinecone-client`)

#### 2. Memory Retrieval Returns 0 Results
**Symptoms**: "I don't have enough information" responses
**Solutions**:
- Check if memories exist for the user_id
- Verify threshold settings (try lowering to 0.1)
- Test with direct search endpoint first
- Check embedding generation

#### 3. Low Keyword Coverage
**Symptoms**: Relevant memories not found for specific terms
**Solutions**:
- Add terms to keyword expansion mapping
- Implement entity-specific query patterns
- Check tag relevance calculations

### Debug Commands

```bash
# Check environment variables
python -c "from dotenv import load_dotenv; load_dotenv('.env'); import os; print('PINECONE_API_KEY:', 'SET' if os.getenv('PINECONE_API_KEY') else 'NOT_SET')"

# Test Pinecone connection
python -c "from pinecone import Pinecone; pc = Pinecone(api_key='YOUR_KEY'); print([idx.name for idx in pc.list_indexes()])"

# Check memory count
curl -X GET "http://localhost:8000/api/v1/stats" | jq '.result.processing_stats'

# Test sample query
curl -X POST "http://localhost:8000/api/v1/ask" -H "Content-Type: application/json" -d '{"question": "test query", "user_id": "john-doe", "session_id": "test", "max_memories": 5}' | jq '.result.memories_used'
```

---

## Development Workflow

### Setting Up Development Environment

1. **Install Dependencies**
```bash
poetry install
```

2. **Configure Environment Variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Start Storage Services** (Docker)
```bash
docker-compose up -d
```

4. **Run Application**
```bash
poetry run uvicorn app.main:app --reload
```

### Testing Workflow

1. **Create Sample Data**
```bash
python tests/create_sample_data.py
```

2. **Run Reasoning Tests**
```bash
python tests/test_multihop_reasoning.py
```

3. **Run Unit Tests**
```bash
pytest tests/
```

### Deployment Checklist

- [ ] All environment variables configured
- [ ] Storage services healthy
- [ ] Sample data created successfully
- [ ] Multi-hop reasoning tests passing (>60%)
- [ ] API endpoints responding correctly
- [ ] Monitoring and logging configured

---

## Technical Implementation Details

### Memory Type Classification

```python
class MemoryTypeInferencer:
    async def infer_type(self, record: MemoryRecord) -> MemoryType:
        prompt = f"""
        Classify this memory into one of these types:
        - semantic_memory: Facts, knowledge, concepts
        - episodic_memory: Personal experiences, events
        - procedural_memory: Skills, how-to knowledge
        - meta_memory: Reflections about memories/learning
        - emotional_memory: Feelings, emotional responses
        
        Memory: {record.raw_text}
        """
        
        response = await self.llm.generate(prompt)
        return MemoryType.from_string(response)
```

### Embedding Generation

```python
async def generate_embedding(self, text: str) -> List[float]:
    # Clean and prepare text
    cleaned_text = self.clean_text(text)
    
    # Generate embedding using OpenAI
    response = await self.embedder.embed_text(cleaned_text)
    
    # Normalize embedding vector
    embedding = self.normalize_vector(response.embedding)
    
    return embedding
```

### Cross-Store Synchronization

```python
async def store_memory(self, memory_entry: MemoryEntry):
    tasks = []
    
    # Store in all configured stores concurrently
    if self.vector_store:
        tasks.append(self.vector_store.add_memory(memory_entry))
    if self.graph_store:
        tasks.append(self.graph_store.add_memory(memory_entry))
    if self.doc_store:
        tasks.append(self.doc_store.add_memory(memory_entry))
    
    # Execute all storage operations in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify at least one store succeeded
    successes = sum(1 for r in results if not isinstance(r, Exception))
    if successes == 0:
        raise StorageError("All stores failed")
```

---

## Future Enhancements

### Planned Improvements

1. **Advanced Reasoning Chains**
   - Implement causal reasoning chains
   - Add temporal reasoning for time-based queries
   - Develop analogy-based reasoning

2. **Enhanced Memory Clustering**
   - Automatic memory clustering by themes
   - Dynamic relationship discovery
   - Temporal pattern recognition

3. **Performance Optimizations**
   - Query result caching
   - Embedding pre-computation
   - Parallel reasoning chain evaluation

4. **Advanced Analytics**
   - Memory usage patterns
   - Reasoning quality metrics
   - User behavior insights

### Research Directions

- **Memory Consolidation**: Automatic merging of related memories
- **Forgetting Mechanisms**: Implementing memory decay and prioritization
- **Contextual Adaptation**: Learning user preferences and adapting responses
- **Multi-Modal Support**: Integration of images, audio, and video memories

---

*This documentation represents the current state of the Context0 Memory System as of July 2025. For the latest updates and technical details, refer to the source code and API documentation.*