# Chat API Reference

## Overview
All chat endpoints require authentication via Clerk JWT token in the Authorization header.

```
Authorization: Bearer YOUR_CLERK_JWT_TOKEN
```

## Base Response Format
All endpoints return responses in this format:

```typescript
interface BaseResponse<T> {
  result: T | null;
  status_code: number;
  message: string;
  success: boolean;
}
```

---

## 1. Create Chat Session

**POST** `/api/v1/chat/sessions`

### Request Body
```json
{
  "title": "New Chat Session"  // Optional
}
```

### Response
```json
{
  "result": {
    "session_id": "882c6f94-5d9c-4216-88a9-5f29b61ce702",
    "title": "New Chat Session",
    "created_at": "2025-08-01T19:38:40.309183"
  },
  "status_code": 200,
  "message": "Chat session created successfully",
  "success": true
}
```

---

## 2. List Chat Sessions

**GET** `/api/v1/chat/sessions`

### Query Parameters
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| skip | integer | 0 | Number of sessions to skip |
| limit | integer | 20 | Maximum sessions to return |

### Response
```json
{
  "result": [
    {
      "id": "882c6f94-5d9c-4216-88a9-5f29b61ce702",
      "title": "Chat about AI",
      "status": "active",
      "created_at": "2025-08-01T19:38:40.309183",
      "updated_at": "2025-08-01T20:15:23.123456",
      "last_message": "That's a great question about...",
      "message_count": 10,
      "total_memories_extracted": 3
    }
  ],
  "status_code": 200,
  "message": "Sessions retrieved successfully",
  "success": true
}
```

---

## 3. Get Session Details

**GET** `/api/v1/chat/sessions/{session_id}`

### Response
```json
{
  "result": {
    "session": {
      "id": "882c6f94-5d9c-4216-88a9-5f29b61ce702",
      "user_id": "user_2xy5bIyLFPhOjYUsbJgsUpbjZxL",
      "title": "Chat about AI",
      "status": "active",
      "created_at": "2025-08-01T19:38:40.309183",
      "updated_at": "2025-08-01T20:15:23.123456",
      "last_message": "That's a great question about...",
      "message_count": 10,
      "total_memories_extracted": 3,
      "metadata": {}
    },
    "messages": [
      {
        "id": "d6c8ee1f-05f3-4f91-bdf6-bfd10c1b4073",
        "session_id": "882c6f94-5d9c-4216-88a9-5f29b61ce702",
        "role": "user",
        "content": "What is artificial intelligence?",
        "timestamp": "2025-08-01T19:52:16.146378",
        "metadata": {},
        "memories_extracted": [],
        "context_used": []
      }
    ]
  },
  "status_code": 200,
  "message": "Session retrieved successfully",
  "success": true
}
```

---

## 4. Send Message

**POST** `/api/v1/chat/sessions/{session_id}/messages`

### Request Body
```json
{
  "content": "Hello, what is 2+2?",
  "use_memory_context": true,      // Optional, default: true
  "extract_memories": true,        // Optional, default: true
  "llm_preset_id": "preset_123",   // Optional
  "temperature": 0.7,              // Optional
  "max_tokens": 2048              // Optional
}
```

### Response
```json
{
  "result": {
    "user_message": {
      "id": "d6c8ee1f-05f3-4f91-bdf6-bfd10c1b4073",
      "role": "user",
      "content": "Hello, what is 2+2?",
      "timestamp": "2025-08-01T19:52:16.146378",
      "metadata": {},
      "memories_extracted": [],
      "context_used": ["mem_1", "mem_2"]
    },
    "assistant_message": {
      "id": "55001d57-fb5a-41d6-96c1-be0a0910cff5",
      "role": "assistant",
      "content": "2 + 2 equals 4.",
      "timestamp": "2025-08-01T19:52:18.502414",
      "metadata": {},
      "memories_extracted": [],
      "context_used": ["mem_1", "mem_2"]
    },
    "memories_extracted": [],
    "context_used": ["mem_1", "mem_2"]
  },
  "status_code": 200,
  "message": "Message sent successfully",
  "success": true
}
```

---

## 5. Send Message (Streaming)

**POST** `/api/v1/chat/sessions/{session_id}/messages/stream`

### Request Body
Same as non-streaming endpoint

### Response (Server-Sent Events)
```
data: {"type": "start", "session_id": "882c6f94-5d9c-4216-88a9-5f29b61ce702"}

data: {"type": "user_message", "message": {...}}

data: {"type": "content", "content": "2", "message_id": "msg_456"}

data: {"type": "content", "content": " +", "message_id": "msg_456"}

data: {"type": "assistant_message", "message": {...}, "summary": {...}}

data: {"type": "done"}
```

---

## 6. Update Session

**PUT** `/api/v1/chat/sessions/{session_id}`

### Request Body
```json
{
  "title": "Updated Title",
  "metadata": {
    "custom_field": "value"
  }
}
```

---

## 7. Delete Single Session

**DELETE** `/api/v1/chat/sessions/{session_id}`

### Query Parameters
| Parameter | Type | Default | Description | Required Role |
|-----------|------|---------|-------------|---------------|
| hard_delete | boolean | false | Permanently delete session and messages | **admin** |

### Response (Soft Delete)
```json
{
  "result": {
    "session_id": "882c6f94-5d9c-4216-88a9-5f29b61ce702"
  },
  "status_code": 200,
  "message": "Session deleted successfully",
  "success": true
}
```

### Response (Hard Delete - Admin Only)
```json
{
  "result": {
    "session_id": "882c6f94-5d9c-4216-88a9-5f29b61ce702",
    "deleted_count": {
      "sessions": 1,
      "messages": 25
    }
  },
  "status_code": 200,
  "message": "Permanently deleted session and 25 messages",
  "success": true
}
```

---

## 8. Delete All Sessions (Bulk)

**DELETE** `/api/v1/chat/sessions`

### Query Parameters
| Parameter | Type | Default | Description | Required Role |
|-----------|------|---------|-------------|---------------|
| hard_delete | boolean | false | Permanently delete all sessions and messages | **admin** |

### Response (Soft Delete)
```json
{
  "result": {
    "deleted_count": 15
  },
  "status_code": 200,
  "message": "Deleted 15 sessions successfully",
  "success": true
}
```

### Response (Hard Delete - Admin Only)
```json
{
  "result": {
    "deleted_count": {
      "sessions": 15,
      "messages": 127
    }
  },
  "status_code": 200,
  "message": "Permanently deleted 15 sessions and 127 messages",
  "success": true
}
```

### Error Response (Non-Admin Attempting Hard Delete)
```json
{
  "result": null,
  "status_code": 403,
  "message": "Hard delete requires admin role",
  "success": false
}
```

---

## Error Responses

### 403 Forbidden
```json
{
  "result": null,
  "status_code": 403,
  "message": "Hard delete requires admin role",
  "success": false
}
```

### 404 Not Found
```json
{
  "result": null,
  "status_code": 404,
  "message": "Session not found",
  "success": false
}
```

### 401 Unauthorized
```json
{
  "result": null,
  "status_code": 401,
  "message": "Missing Authorization header",
  "success": false
}
```

---

## Frontend Implementation Notes

### Checking Admin Role
The frontend should check the user's role before showing hard delete options:

```typescript
// Get user role from Clerk
const { user } = useUser();
const isAdmin = user?.organizationMemberships?.[0]?.role === 'admin';

// Show hard delete button only for admins
{isAdmin && (
  <button onClick={() => deleteAllSessions(true)}>
    Hard Delete All (Admin Only)
  </button>
)}
```

### Delete Functions
```typescript
// Soft delete (all users)
async function deleteSessions() {
  const response = await fetch('/api/v1/chat/sessions', {
    method: 'DELETE',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
}

// Hard delete (admin only)
async function hardDeleteSessions() {
  const response = await fetch('/api/v1/chat/sessions?hard_delete=true', {
    method: 'DELETE',
    headers: {
      'Authorization': `Bearer ${token}`
    }
  });
  
  if (response.status === 403) {
    alert('Hard delete requires admin role');
  }
}
```