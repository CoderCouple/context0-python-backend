# Quick Start Guide

[← Back to Index](./index.md)

Get Context0 up and running in under 10 minutes!

## Prerequisites

- Python 3.11+
- Docker & Docker Compose
- Git

## 1. Clone the Repository

```bash
git clone https://github.com/context0/python-backend.git
cd python-backend
```

## 2. Set Up Environment

### Copy Environment Template
```bash
cp .env.example .env
```

### Edit `.env` with Required Keys
```bash
# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Pinecone
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=memory-index

# MongoDB (using Atlas)
MONGODB_CONNECTION_STRING=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_AUDIT_CONNECTION_STRING=mongodb+srv://username:password@cluster.mongodb.net/

# Neo4j (using Aura)
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-password
NEO4J_DATABASE=neo4j

# TimescaleDB (using Cloud)
TIMESCALE_CONNECTION_STRING=postgresql://user:pass@host:port/db?sslmode=require

# JWT Secret
JWT_SECRET_KEY=your-secret-key-here
```

## 3. Install Dependencies

```bash
# Install Poetry (if not installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install
```

## 4. Start the Application

### Option A: Using Poetry
```bash
poetry run uvicorn app.main:app --reload --port 8000
```

### Option B: Using Docker
```bash
docker-compose up -d
```

## 5. Verify Installation

### Check API Health
```bash
curl http://localhost:8000/api/v1/ping
```

Expected response:
```json
{
  "success": true,
  "message": "pong",
  "data": {
    "version": "1.0.0"
  }
}
```

## 6. Create Your First Memory

### Get JWT Token (for testing)
```bash
# Use the test endpoint or implement your auth flow
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "password": "test"}'
```

### Create a Memory
```bash
curl -X POST http://localhost:8000/api/v1/memories \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "I am learning to use Context0!",
    "category": "learning",
    "emotion": "excited",
    "emotion_intensity": "high"
  }'
```

## 7. Start a Chat Session

```bash
# Create session
curl -X POST http://localhost:8000/api/v1/chat/sessions \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"title": "My First Chat"}'

# Send message (replace SESSION_ID)
curl -X POST http://localhost:8000/api/v1/chat/sessions/SESSION_ID/messages \
  -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "Hello! Tell me about Context0.",
    "extract_memories": true,
    "use_memory_context": true
  }'
```

## Next Steps

✅ You now have Context0 running locally!

### Explore Further:
- [API Documentation](./api-overview.md) - Full API reference
- [Memory System](./memory-system.md) - How memories work
- [Chat System](./features/chat-system.md) - Chat features
- [Development Setup](./dev/setup.md) - Advanced development

### Try These Features:
1. **Search Memories**: Use semantic search to find memories
2. **Chat with Context**: Have conversations that reference past memories
3. **Analytics**: View memory usage statistics
4. **Categories**: Explore automatic categorization

## Troubleshooting

### Common Issues

1. **Port 8000 in use**
   ```bash
   # Use a different port
   poetry run uvicorn app.main:app --reload --port 8001
   ```

2. **Database connection errors**
   - Verify all connection strings in `.env`
   - Check if services are accessible
   - See [Database Setup Guides](./databases/mongodb.md)

3. **Missing dependencies**
   ```bash
   poetry install --no-cache
   ```

### Need Help?
- Check [Common Issues](./troubleshooting/common-issues.md)
- Open an issue on GitHub
- Join our Discord community

---

[← Back to Index](./index.md) | [Next: Installation →](./installation.md)