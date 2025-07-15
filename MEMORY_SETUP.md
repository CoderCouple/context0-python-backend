# Memory System Setup Guide

This guide helps you set up the complete memory system with all required databases using Docker.

## 🚀 Quick Start

### 1. Start all services
```bash
docker-compose up -d
```

This will start:
- **MongoDB** (port 27017) - Document storage and audit logging
- **TimescaleDB** (port 5432) - Time series data
- **Neo4j** (port 7474/7687) - Graph relationships
- **Qdrant** (port 6333) - Vector storage
- **Redis** (port 6379) - Caching (optional)
- **pgAdmin** (port 8080) - Database management
- **Mongo Express** (port 8081) - MongoDB management

### 2. Copy environment variables
```bash
cp .env.example .env
```

The default configuration is already set up for Docker containers.

### 3. Start your application
```bash
python context0_app.py app.main:app --reload
```

## 📊 Database Management Tools

Access these web interfaces to manage your databases:

### pgAdmin (PostgreSQL/TimescaleDB)
- **URL**: http://localhost:8080
- **Email**: admin@memory.dev
- **Password**: devpassword

### Mongo Express (MongoDB)
- **URL**: http://localhost:8081
- **Username**: admin
- **Password**: devpassword

### Neo4j Browser
- **URL**: http://localhost:7474
- **Username**: neo4j
- **Password**: devpassword

### Qdrant Dashboard
- **URL**: http://localhost:6333/dashboard

## 🗄️ Database Configuration

The memory system uses the following databases:

| Store Type | Database | Port | Purpose |
|------------|----------|------|---------|
| Vector Store | Qdrant | 6333 | Embedding storage and similarity search |
| Graph Store | Neo4j | 7687 | Relationship mapping |
| Document Store | MongoDB | 27017 | Structured memory storage |
| Time Series | TimescaleDB | 5432 | Temporal memory analysis |
| Audit Store | MongoDB | 27017 | Operation logging |

## 🔧 Configuration Files

- **default.yaml** - Development configuration using Docker containers
- **prod.yaml** - Production configuration with environment variables
- **example.yaml** - Complete reference of all options
- **.env.example** - Environment variables template

## 🐳 Docker Services

### Core Memory Databases
```yaml
# Vector storage
qdrant:
  image: qdrant/qdrant:v1.7.0
  ports: ["6333:6333"]

# Graph relationships
neo4j:
  image: neo4j:5.13-community
  ports: ["7474:7474", "7687:7687"]

# Document storage
mongodb:
  image: mongo:7.0
  ports: ["27017:27017"]

# Time series
timescaledb:
  image: timescale/timescaledb:latest-pg15
  ports: ["5432:5432"]
```

### Optional Services
- **Redis** (port 6379) - Caching
- **pgAdmin** (port 8080) - PostgreSQL management
- **Mongo Express** (port 8081) - MongoDB management

## 🔄 Common Commands

### Start all services
```bash
docker-compose up -d
```

### Stop all services
```bash
docker-compose down
```

### View logs
```bash
docker-compose logs -f [service-name]
```

### Reset all data
```bash
docker-compose down -v  # Removes volumes
docker-compose up -d
```

### Check service status
```bash
docker-compose ps
```

## 🌐 Port Mapping

| Service | Host Port | Container Port | Purpose |
|---------|-----------|----------------|---------|
| Web App | 8000 | 8000 | Main application |
| Original DB | 5433 | 5432 | Legacy PostgreSQL |
| TimescaleDB | 5432 | 5432 | Time series database |
| MongoDB | 27017 | 27017 | Document database |
| Neo4j HTTP | 7474 | 7474 | Neo4j browser |
| Neo4j Bolt | 7687 | 7687 | Neo4j driver |
| Qdrant | 6333 | 6333 | Vector database |
| Redis | 6379 | 6379 | Cache |
| pgAdmin | 8080 | 80 | DB management |
| Mongo Express | 8081 | 8081 | MongoDB management |

## 🔐 Default Credentials

All services use `devpassword` as the default password for development:

- **Neo4j**: neo4j / devpassword
- **MongoDB**: admin / devpassword
- **TimescaleDB**: postgres / devpassword
- **Redis**: devpassword
- **pgAdmin**: admin@memory.dev / devpassword
- **Mongo Express**: admin / devpassword

## 🚨 Troubleshooting

### Port conflicts
If you get port conflicts, check what's running:
```bash
lsof -i :6333  # Check Qdrant port
lsof -i :7687  # Check Neo4j port
lsof -i :27017 # Check MongoDB port
```

### Service won't start
Check logs for specific service:
```bash
docker-compose logs neo4j
docker-compose logs mongodb
docker-compose logs timescaledb
```

### Reset specific service
```bash
docker-compose stop mongodb
docker volume rm context0-python-backend_mongodb_data
docker-compose up -d mongodb
```

## 📝 Next Steps

1. Start the services: `docker-compose up -d`
2. Copy environment file: `cp .env.example .env`
3. Run your application: `python context0_app.py app.main:app --reload`
4. Test the memory APIs
5. Explore the database management tools

The memory system is now ready for development with full database persistence!