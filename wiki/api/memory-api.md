# Memory API Documentation

[← Back to Index](../index.md) | [← API Overview](../api-overview.md)

## Overview

The Memory API provides comprehensive endpoints for creating, managing, and searching memories with advanced categorization and emotional context.

## Features

- **Memory CRUD Operations**: Create, read, update, delete memories
- **Advanced Search**: Semantic and keyword-based search
- **Categorization**: Automatic category and emotion detection
- **Batch Operations**: Process multiple memories efficiently
- **User Context**: All operations scoped to authenticated user

## API Endpoints

### 1. Create Memory
```
POST /api/v1/memories
```

Request:
```json
{
  "text": "I completed the machine learning course today!",
  "category": "learning",
  "emotion": "excited",
  "emotion_intensity": "high",
  "tags": ["education", "achievement", "ml"],
  "metadata": {
    "course_name": "Deep Learning Specialization",
    "platform": "Coursera"
  }
}
```

Response:
```json
{
  "success": true,
  "message": "Memory created successfully",
  "data": {
    "memory_id": "mem_123456",
    "memory_type": "semantic_memory",
    "processing_time_ms": 150,
    "stores_updated": ["vector", "graph", "document", "timeseries", "audit"]
  }
}
```

### 2. Get Memory by ID
```
GET /api/v1/memories/{memory_id}
```

Response:
```json
{
  "success": true,
  "message": "Memory retrieved successfully",
  "data": {
    "id": "mem_123456",
    "user_id": "user_789",
    "text": "I completed the machine learning course today!",
    "summary": "Completed ML course",
    "category": "learning",
    "emotion": "excited",
    "emotion_intensity": "high",
    "created_at": "2025-07-30T15:00:00Z",
    "tags": ["education", "achievement", "ml"],
    "confidence_score": 0.95,
    "metadata": {
      "course_name": "Deep Learning Specialization",
      "platform": "Coursera"
    }
  }
}
```

### 3. Search Memories
```
POST /api/v1/memories/search
```

Request:
```json
{
  "query": "machine learning achievements",
  "limit": 20,
  "skip": 0,
  "filters": {
    "category": "learning",
    "emotion": "excited",
    "date_range": {
      "start": "2025-07-01",
      "end": "2025-07-31"
    },
    "tags": ["ml", "education"]
  },
  "search_type": "semantic"  // or "keyword" or "hybrid"
}
```

Response:
```json
{
  "success": true,
  "message": "Found 5 memories",
  "data": {
    "results": [
      {
        "id": "mem_123456",
        "text": "I completed the machine learning course today!",
        "summary": "Completed ML course",
        "category": "learning",
        "score": 0.92,
        "created_at": "2025-07-30T15:00:00Z"
      }
    ],
    "total": 5,
    "query_time_ms": 45
  }
}
```

### 4. Update Memory
```
PUT /api/v1/memories/{memory_id}
```

Request:
```json
{
  "tags": ["education", "achievement", "ml", "completed"],
  "metadata": {
    "course_name": "Deep Learning Specialization",
    "platform": "Coursera",
    "completion_date": "2025-07-30"
  }
}
```

### 5. Delete Memory
```
DELETE /api/v1/memories/{memory_id}
```

Response:
```json
{
  "success": true,
  "message": "Memory deleted successfully",
  "data": {
    "memory_id": "mem_123456",
    "deleted_from_stores": ["vector", "graph", "document", "timeseries"]
  }
}
```

### 6. Batch Create Memories
```
POST /api/v1/memories/batch
```

Request:
```json
{
  "memories": [
    {
      "text": "Started new project with React",
      "category": "work",
      "tags": ["project", "react"]
    },
    {
      "text": "Team meeting about Q3 goals",
      "category": "work",
      "tags": ["meeting", "planning"]
    }
  ]
}
```

### 7. Get User Statistics
```
GET /api/v1/memories/stats
```

Response:
```json
{
  "success": true,
  "data": {
    "total_memories": 156,
    "by_category": {
      "work": 45,
      "personal": 38,
      "learning": 73
    },
    "by_emotion": {
      "happy": 62,
      "neutral": 51,
      "excited": 43
    },
    "recent_activity": {
      "last_7_days": 23,
      "last_30_days": 89
    }
  }
}
```

### 8. Get Related Memories
```
GET /api/v1/memories/{memory_id}/related?limit=10
```

Response:
```json
{
  "success": true,
  "data": {
    "memory_id": "mem_123456",
    "related": [
      {
        "id": "mem_789012",
        "text": "Studied neural networks chapter",
        "relationship": "similar_topic",
        "score": 0.87
      }
    ]
  }
}
```

## Memory Categories

Available categories:
- `work` - Professional activities
- `personal` - Personal life events
- `learning` - Educational content
- `health` - Health-related information
- `finance` - Financial matters
- `travel` - Travel experiences
- `social` - Social interactions
- `creative` - Creative projects
- `technical` - Technical knowledge
- `general` - Uncategorized

## Emotions

Available emotions:
- `happy` - Positive, joyful
- `excited` - Enthusiastic, energetic
- `calm` - Peaceful, relaxed
- `sad` - Negative, melancholic
- `anxious` - Worried, stressed
- `angry` - Frustrated, upset
- `neutral` - No strong emotion
- `grateful` - Thankful, appreciative
- `proud` - Achievement-based
- `curious` - Interested, questioning

## Emotion Intensity
- `low` - Mild emotion
- `medium` - Moderate emotion
- `high` - Strong emotion

## Search Types

### Semantic Search
Uses vector embeddings to find memories by meaning:
```json
{
  "search_type": "semantic",
  "query": "times I felt accomplished"
}
```

### Keyword Search
Traditional text matching:
```json
{
  "search_type": "keyword",
  "query": "machine learning course"
}
```

### Hybrid Search
Combines semantic and keyword search:
```json
{
  "search_type": "hybrid",
  "query": "React project meetings"
}
```

## Error Handling

### Common Error Responses

#### 404 - Memory Not Found
```json
{
  "success": false,
  "message": "Memory not found",
  "error": {
    "code": "MEMORY_NOT_FOUND",
    "details": "No memory found with ID: mem_123456"
  }
}
```

#### 400 - Invalid Request
```json
{
  "success": false,
  "message": "Invalid request",
  "error": {
    "code": "INVALID_CATEGORY",
    "details": "Category 'invalid' is not supported"
  }
}
```

## Rate Limiting

- **Create**: 100 memories per minute
- **Search**: 300 requests per minute
- **Batch**: 10 batch operations per minute

## Best Practices

1. **Use Specific Categories**: Help the system organize memories effectively
2. **Include Emotion Context**: Improves search and retrieval
3. **Add Relevant Tags**: Enhance discoverability
4. **Batch Operations**: Use batch endpoints for multiple memories
5. **Search Optimization**: Use filters to narrow results

## Examples

### Example: Daily Journal Entry
```python
import httpx
import asyncio

async def create_journal_memory():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/memories",
            headers={"Authorization": "Bearer <token>"},
            json={
                "text": "Had a productive day. Finished the API documentation and helped a colleague debug their code. Feeling accomplished!",
                "category": "personal",
                "emotion": "proud",
                "emotion_intensity": "medium",
                "tags": ["journal", "daily", "productivity"],
                "metadata": {
                    "type": "journal_entry",
                    "date": "2025-07-30",
                    "mood_score": 8
                }
            }
        )
        print(response.json())

asyncio.run(create_journal_memory())
```

### Example: Search Work Memories
```python
async def search_work_memories():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "http://localhost:8000/api/v1/memories/search",
            headers={"Authorization": "Bearer <token>"},
            json={
                "query": "project deadlines and deliverables",
                "filters": {
                    "category": "work",
                    "date_range": {
                        "start": "2025-07-01",
                        "end": "2025-07-31"
                    }
                },
                "search_type": "hybrid",
                "limit": 10
            }
        )
        
        results = response.json()["data"]["results"]
        for memory in results:
            print(f"- {memory['summary']} ({memory['created_at']})")
```

---

[← Back to Index](../index.md) | [Next: Analytics API →](./analytics-api.md)