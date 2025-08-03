make a# Context0 Memory System

A sophisticated AI memory management system built with FastAPI that provides intelligent memory storage, retrieval, and time-travel debugging capabilities.

## Features

### 🧠 Memory Types
- **Semantic**: Knowledge and facts
- **Episodic**: Personal experiences and events
- **Procedural**: Skills and how-to knowledge
- **Emotional**: Feelings and emotional context
- **Working**: Temporary active information
- **Declarative**: Explicit knowledge
- **Meta**: Knowledge about knowledge

### 🚀 Core Capabilities
- **LLM-Powered Classification**: Automatic memory type inference using OpenAI/Anthropic
- **Multi-Store Architecture**: Vector, Graph, Document, TimeSeries, and Audit stores
- **Time-Travel Debugging**: Navigate memory state at any point in time
- **Async Processing**: Full async/await support for high performance
- **Rich Metadata**: Comprehensive tagging, permissions, and relationships

### 🏗️ Architecture

```
MemoryRecord → MemoryTypeInferencer → MemoryRouter → Handler.process() → MemoryEntry → Stores
```

## Quick Start

### Prerequisites
- Python 3.11+
- Poetry
- Required external services (Pinecone, MongoDB, Neo4j, TimescaleDB)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd context0-python-backend

# Install dependencies
poetry install

# Install additional memory system dependencies
poetry add pinecone-client motor pymongo neo4j asyncpg langchain openai python-dateutil
```

### Configuration

Set environment variables for external services:

```bash
# OpenAI
export OPENAI_API_KEY="your-openai-key"

# Pinecone
export PINECONE_API_KEY="your-pinecone-key"
export PINECONE_ENVIRONMENT="your-environment"

# MongoDB
export MONGODB_URL="mongodb://localhost:27017"

# Neo4j
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="password"

# TimescaleDB
export TIMESCALE_URL="postgresql://user:pass@localhost:5432/db"
```

### Running the Application

```bash
# Development server with auto-reload
poetry run python context0_app.py app.main:app --reload

# Or using uvicorn directly
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

## Memory System Architecture

### Models

#### MemoryRecord
Runtime context for incoming observations:
```python
class MemoryRecord(BaseModel):
    user_id: str
    session_id: Optional[str]
    memory_type: Optional[MemoryType]
    raw_text: str
    timestamp: datetime
    tags: List[str]
    metadata: Dict[str, Any]
```

#### MemoryEntry
Canonical persisted memory object with rich metadata:
```python
class MemoryEntry(BaseModel):
    id: str
    cid: str  # Content ID for deduplication
    scope: str
    input: str
    summary: Optional[str]
    memory_type: MemoryType
    permissions: MemoryPermissions
    embedding: Optional[List[float]]
    graph_links: List[GraphLink]
```

### Storage Backends

- **Pinecone**: Vector embeddings for semantic search
- **Neo4j**: Knowledge graphs and relationships
- **MongoDB**: Document storage with full-text search
- **TimescaleDB**: Temporal analysis and time-series data
- **MongoDB Audit**: Time-travel debugging and state snapshots

### Handlers

Specialized processors for each memory type:
- **SemanticHandler**: Extracts embeddings and knowledge
- **EpisodicHandler**: Temporal context and experiences
- **ProceduralHandler**: Step-by-step processes
- **EmotionalHandler**: Sentiment and emotional context
- **WorkingHandler**: Active information management
- **DeclarativeHandler**: Explicit facts
- **MetaHandler**: Self-referential knowledge

## API Endpoints

### Memory Operations
- `POST /api/v1/memory/`: Create new memory
- `GET /api/v1/memory/{memory_id}`: Retrieve memory
- `GET /api/v1/memory/search`: Search memories
- `GET /api/v1/memory/time-travel`: Time-travel queries

### Analytics
- `GET /api/v1/analytics/`: System analytics
- `GET /api/v1/analytics/memory-stats`: Memory statistics

### Health & Monitoring
- `GET /api/v1/ping`: Health check
- `GET /api/v1/webhook`: Webhook management

## Time-Travel Debugging

Query memory state at any point in time:

```python
# Get all memories that existed at a specific time
memories = await audit_store.get_memory_state_at_time(
    user_id="user123",
    target_time=datetime(2024, 1, 15, 10, 30)
)

# Trace memory evolution
evolution = await audit_store.get_memory_evolution(
    memory_id="mem_456",
    start_time=start_time,
    end_time=end_time
)
```

## Development

### Code Style
- **Black**: Code formatting
- **isort**: Import sorting
- **mypy**: Type checking
- **pre-commit**: Git hooks

### Testing
```bash
poetry run pytest
```

### Pre-commit Hooks
```bash
poetry run pre-commit install
poetry run pre-commit run --all-files
```

## Project Structure

```
app/
├── memory/                   # Memory system core
│   ├── models/               # Data models
│   ├── inferencer/           # LLM-based type inference
│   ├── router/               # Memory routing logic
│   ├── handlers/             # Type-specific processors
│   ├── stores/               # Storage backends
│   ├── engine/               # Main orchestrator
│   └── config/               # Configuration management
├── api/                      # FastAPI routes
├── common/                   # Shared utilities
├── service/                  # Business logic
└── model/                    # Database models
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

[License details here]

## Support

For questions and support, please [create an issue](link-to-issues) in the repository.