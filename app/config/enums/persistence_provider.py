# context0-python-backend/app/config/persistence_provider.py
from enum import Enum

# Enum for supported vector store providers
# ┌───────────────┬───────────────────────┬────────────────────────┐
# │   DB Type     │    In-Memory Version  │    Disk-Based Version  │
# ├───────────────┼───────────────────────┼────────────────────────┤
# │ Vector Store  │ FAISS, Qdrant (RAM)   │ Weaviate, Pinecone     │
# │ Structured DB │ SQLite                │ PostgreSQL, MySQL      │
# │ Document DB   │ Mongita, TinyDB       │ MongoDB                │
# │ Graph Store   │ Memgraph              │ Neo4j                  │
# │ Time-Series   │ InfluxDB (mem engine) │ TimescaleDB, InfluxDB  │
# └───────────────┴───────────────────────┴────────────────────────┘


class VectorProvider(str, Enum):
    IN_MEMORY = "in_memory"  # Local RAM-based vector store (dev/test mode)
    QDRANT = "qdrant"  # Disk-based vector DB with REST/GRPC APIs
    WEAVIATE = "weaviate"  # Semantic graph-aware vector DB
    FAISS = "faiss"  # Fast, local similarity search engine
    MILVUS = "milvus"  # Scalable open-source vector DB


# Enum for supported graph memory store backends
class GraphProvider(str, Enum):
    IN_MEMORY = "in_memory"  # RAM-only graph memory (testing)
    NEO4J = "neo4j"  # Disk-backed graph database
    MEMGRAPH = "memgraph"  # In-memory graph DB with Cypher support


# Enum for event/conversation log storage
class EventProvider(str, Enum):
    IN_MEMORY = "in_memory"  # Simple Python dict
    SQLITE = "sqlite"  # Lightweight embedded DB
    MONGO = "mongo"  # MongoDB for document-based event storage
    POSTGRES = "postgres"  # Structured SQL event storage


# Enum for time series data store
class TSDBProvider(str, Enum):
    IN_MEMORY = "in_memory"  # List-based event store in RAM
    TIMESCALEDB = "timescaledb"  # PostgreSQL extension for time series
    INFLUXDB = "influxdb"  # Popular open-source time series DB


# Enum for persistence mode
class PersistenceType(str, Enum):
    MEMORY = "memory"  # Non-persistent (for dev/tests)
    DISK = "disk"  # Persistent across sessions
