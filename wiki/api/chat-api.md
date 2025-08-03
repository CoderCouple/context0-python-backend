# Chat API Documentation

[← Back to Index](../index.md) | [← API Overview](../api-overview.md)

## Overview

The Chat API provides endpoints for managing chat sessions and messages with automatic memory extraction and AI-powered responses.

## Features

- **Session Management**: Create, list, update, and delete chat sessions
- **AI-Powered Conversations**: Integration with OpenAI for intelligent responses
- **Automatic Memory Extraction**: Extracts important information from conversations
- **Memory Context**: Uses relevant memories to enhance AI responses
- **Message History**: Full conversation history with metadata

## API Endpoints

### 1. Create Chat Session
```
POST /api/v1/chat/sessions
```

Request:
```json
{
  "title": "Optional session title",
  "metadata": {
    "custom": "data"
  }
}
```

Response:
```json
{
  "success": true,
  "message": "Chat session created successfully",
  "data": {
    "session_id": "uuid",
    "title": "Chat 2025-07-30 10:00",
    "created_at": "2025-07-30T10:00:00Z"
  }
}
```

### 2. List Chat Sessions
```
GET /api/v1/chat/sessions?limit=20&skip=0
```

Response:
```json
{
  "success": true,
  "message": "Retrieved 5 chat sessions",
  "data": [
    {
      "id": "uuid",
      "user_id": "user123",
      "title": "Project Discussion",
      "created_at": "2025-07-30T10:00:00Z",
      "updated_at": "2025-07-30T11:00:00Z",
      "last_message": "Sure, I can help with that...",
      "message_count": 10,
      "total_memories_extracted": 3
    }
  ]
}
```

### 3. Get Chat Session Details
```
GET /api/v1/chat/sessions/{session_id}?message_limit=50&message_skip=0
```

Response:
```json
{
  "success": true,
  "message": "Session retrieved successfully",
  "data": {
    "session": {
      "id": "uuid",
      "user_id": "user123",
      "title": "Project Discussion",
      "created_at": "2025-07-30T10:00:00Z",
      "updated_at": "2025-07-30T11:00:00Z",
      "last_message": "Sure, I can help with that...",
      "message_count": 10,
      "total_memories_extracted": 3
    },
    "messages": [
      {
        "id": "msg1",
        "role": "user",
        "content": "Hello, I need help with my project",
        "timestamp": "2025-07-30T10:00:00Z",
        "metadata": {},
        "memories_extracted": ["mem1", "mem2"],
        "context_used": null
      },
      {
        "id": "msg2",
        "role": "assistant",
        "content": "I'd be happy to help with your project...",
        "timestamp": "2025-07-30T10:00:05Z",
        "metadata": {},
        "memories_extracted": null,
        "context_used": ["mem3", "mem4"]
      }
    ]
  }
}
```

### 4. Send Message
```
POST /api/v1/chat/sessions/{session_id}/messages
```

Request:
```json
{
  "content": "Can you help me with FastAPI project structure?",
  "extract_memories": true,
  "use_memory_context": true
}
```

Response:
```json
{
  "success": true,
  "message": "Message sent successfully",
  "data": {
    "user_message": {
      "id": "msg1",
      "role": "user",
      "content": "Can you help me with FastAPI project structure?",
      "timestamp": "2025-07-30T10:00:00Z",
      "memories_extracted": ["mem1"],
      "context_used": null
    },
    "assistant_message": {
      "id": "msg2",
      "role": "assistant",
      "content": "I'd be happy to help you structure your FastAPI project...",
      "timestamp": "2025-07-30T10:00:05Z",
      "memories_extracted": null,
      "context_used": ["mem3", "mem4"]
    },
    "memories_extracted": ["mem1"],
    "context_used": ["mem3", "mem4"]
  }
}
```

### 5. Update Chat Session
```
PUT /api/v1/chat/sessions/{session_id}
```

Request:
```json
{
  "title": "New session title",
  "metadata": {
    "updated": true
  }
}
```

### 6. Delete Chat Session
```
DELETE /api/v1/chat/sessions/{session_id}
```

### 7. Extract Memories from Session
```
POST /api/v1/chat/sessions/{session_id}/extract-memories
```

Request:
```json
{
  "message_ids": ["msg1", "msg2"],  // Optional, all if not provided
  "force": false  // Force re-extraction
}
```

Response:
```json
{
  "success": true,
  "message": "Extracted 5 memories",
  "data": {
    "session_id": "uuid",
    "messages_processed": 10,
    "memories_extracted": ["mem1", "mem2", "mem3", "mem4", "mem5"],
    "extraction_summary": {
      "total_memories": 5,
      "by_role": {
        "user": 3,
        "assistant": 2
      }
    }
  }
}
```

## Memory Extraction

The chat system automatically extracts important information from conversations:

1. **User Messages**: Extracts facts, preferences, plans, and decisions
2. **Assistant Messages**: Extracts commitments, insights, and important information
3. **Categorization**: Memories are automatically categorized (work, personal, etc.)
4. **Tags**: Automatic tagging with session ID and role

## Memory Context

When `use_memory_context` is enabled:
1. System searches for relevant memories based on the message content
2. Top 5 most relevant memories are included in the AI prompt
3. AI can reference past conversations and information

## Database Schema

### Chat Sessions Collection (MongoDB)
```javascript
{
  id: String,
  user_id: String,
  title: String,
  status: "active" | "archived" | "deleted",
  created_at: Date,
  updated_at: Date,
  last_message: String,
  message_count: Number,
  total_memories_extracted: Number,
  metadata: Object
}
```

### Chat Messages Collection (MongoDB)
```javascript
{
  id: String,
  session_id: String,
  role: "user" | "assistant" | "system",
  content: String,
  timestamp: Date,
  metadata: Object,
  memories_extracted: [String],
  context_used: [String]
}
```

## Usage Example

```python
import httpx
import asyncio

async def chat_example():
    headers = {"Authorization": "Bearer <token>"}
    base_url = "http://localhost:8000/api/v1"
    
    async with httpx.AsyncClient() as client:
        # Create session
        session_resp = await client.post(
            f"{base_url}/chat/sessions",
            headers=headers,
            json={"title": "Project Help"}
        )
        session_id = session_resp.json()["data"]["session_id"]
        
        # Send message
        message_resp = await client.post(
            f"{base_url}/chat/sessions/{session_id}/messages",
            headers=headers,
            json={
                "content": "I'm building a FastAPI app for managing tasks",
                "extract_memories": True,
                "use_memory_context": True
            }
        )
        
        print(message_resp.json()["data"]["assistant_message"]["content"])

asyncio.run(chat_example())
```