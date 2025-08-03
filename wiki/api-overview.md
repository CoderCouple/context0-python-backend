# API Overview

[← Back to Index](./index.md)

## Introduction

The Context0 API is a RESTful API built with FastAPI that provides comprehensive memory management and AI-powered chat capabilities. All endpoints follow REST conventions and return JSON responses.

## Base URL

```
Development: http://localhost:8000/api/v1
Production: https://api.context0.ai/api/v1
```

## Authentication

All API endpoints (except `/ping`) require JWT authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

See [Authentication Guide](./authentication.md) for details on obtaining tokens.

## API Endpoints

### Core APIs

#### 🧠 [Memory API](./api/memory-api.md)
Manage user memories with categorization and search capabilities.
- `POST /memories` - Create memory
- `GET /memories/{id}` - Get memory
- `PUT /memories/{id}` - Update memory
- `DELETE /memories/{id}` - Delete memory
- `POST /memories/search` - Search memories
- `POST /memories/batch` - Batch operations

#### 💬 [Chat API](./api/chat-api.md)
AI-powered chat sessions with memory integration.
- `POST /chat/sessions` - Create session
- `GET /chat/sessions` - List sessions
- `POST /chat/sessions/{id}/messages` - Send message
- `POST /chat/sessions/{id}/extract-memories` - Extract memories

#### 📊 [Analytics API](./api/analytics-api.md)
Usage statistics and insights.
- `GET /analytics/usage` - Usage statistics
- `GET /analytics/memories` - Memory analytics
- `GET /analytics/trends` - Usage trends

#### 💳 [Billing API](./api/billing-api.md)
Usage tracking and billing information.
- `GET /billing/usage` - Current usage
- `GET /billing/history` - Billing history
- `GET /billing/limits` - Plan limits

#### ❓ [Q&A API](./api/qa-api.md)
Question-answering using memory context.
- `POST /qa/ask` - Ask a question
- `GET /qa/history` - Q&A history

#### 🔔 [Webhook API](./api/webhook-api.md)
Webhook integrations for events.
- `POST /webhooks` - Register webhook
- `GET /webhooks` - List webhooks
- `DELETE /webhooks/{id}` - Remove webhook

### Utility Endpoints

#### Health Check
```
GET /ping
```
No authentication required. Returns:
```json
{
  "success": true,
  "message": "pong",
  "data": {
    "version": "1.0.0"
  }
}
```

## Request Format

### Headers
```http
Content-Type: application/json
Authorization: Bearer <token>
Accept: application/json
```

### Request Body
All POST/PUT requests accept JSON:
```json
{
  "field1": "value1",
  "field2": "value2"
}
```

## Response Format

### Success Response
```json
{
  "success": true,
  "message": "Operation completed successfully",
  "data": {
    // Response data
  }
}
```

### Error Response
```json
{
  "success": false,
  "message": "Error description",
  "error": {
    "code": "ERROR_CODE",
    "details": "Detailed error message",
    "field": "field_name"  // For validation errors
  }
}
```

## Status Codes

- `200 OK` - Successful request
- `201 Created` - Resource created
- `204 No Content` - Successful with no response body
- `400 Bad Request` - Invalid request format
- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `422 Unprocessable Entity` - Validation error
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

## Rate Limiting

Default rate limits per user:
- **Standard endpoints**: 1000 requests/hour
- **Search endpoints**: 300 requests/hour
- **Batch operations**: 100 requests/hour
- **AI operations**: 60 requests/hour

Rate limit headers:
```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1627847483
```

## Pagination

List endpoints support pagination:
```
GET /memories?limit=20&skip=0
```

Response includes pagination info:
```json
{
  "data": {
    "items": [...],
    "total": 100,
    "limit": 20,
    "skip": 0
  }
}
```

## Filtering & Sorting

### Filtering
```
GET /memories?category=work&emotion=happy
```

### Date Range
```
GET /memories?start_date=2025-01-01&end_date=2025-12-31
```

### Sorting
```
GET /memories?sort_by=created_at&order=desc
```

## Error Codes

Common error codes across all APIs:

- `AUTH_REQUIRED` - Authentication required
- `INVALID_TOKEN` - Invalid or expired token
- `INSUFFICIENT_PERMISSIONS` - User lacks required permissions
- `RESOURCE_NOT_FOUND` - Requested resource not found
- `VALIDATION_ERROR` - Request validation failed
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `INTERNAL_ERROR` - Server error

## SDK Support

### Python SDK
```python
from context0 import Client

client = Client(api_key="your-api-key")
memory = client.memories.create(text="Important meeting today")
```

### JavaScript/TypeScript SDK
```typescript
import { Context0Client } from '@context0/sdk';

const client = new Context0Client({ apiKey: 'your-api-key' });
const memory = await client.memories.create({ text: 'Important meeting today' });
```

## API Versioning

The API uses URL versioning:
- Current version: `v1`
- URL format: `/api/v1/endpoint`

Version compatibility:
- Minor updates are backward compatible
- Major version changes may include breaking changes
- Deprecation notices provided 3 months in advance

## Best Practices

1. **Use Proper Authentication**: Always secure your API keys
2. **Handle Rate Limits**: Implement exponential backoff
3. **Batch Operations**: Use batch endpoints for multiple operations
4. **Error Handling**: Always check the `success` field
5. **Pagination**: Use pagination for large datasets
6. **Caching**: Cache responses when appropriate
7. **Compression**: Enable gzip compression

## Testing

### Using cURL
```bash
curl -X POST https://api.context0.ai/api/v1/memories \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "Test memory"}'
```

### Using HTTPie
```bash
http POST api.context0.ai/api/v1/memories \
  Authorization:"Bearer YOUR_TOKEN" \
  text="Test memory"
```

### Postman Collection
Download our [Postman Collection](./assets/postman-collection.json) for easy API testing.

## Support

- **Documentation**: You're reading it!
- **API Status**: [status.context0.ai](https://status.context0.ai)
- **Support Email**: api-support@context0.ai
- **GitHub Issues**: [Report bugs](https://github.com/context0/python-backend/issues)

---

[← Back to Index](./index.md) | [Next: Authentication →](./authentication.md)