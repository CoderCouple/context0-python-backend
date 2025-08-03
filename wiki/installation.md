# Installation Guide

[← Back to Index](./index.md)

This guide provides detailed installation instructions for Context0. For a quick overview, see the [Memory Setup Guide](MEMORY_SETUP.md).

## System Requirements

### Minimum Requirements
- **CPU**: 2 cores
- **RAM**: 4GB
- **Storage**: 10GB free space
- **OS**: Linux, macOS, or Windows with WSL2

### Recommended Requirements
- **CPU**: 4+ cores
- **RAM**: 8GB+
- **Storage**: 20GB+ free space
- **OS**: Ubuntu 22.04 LTS or macOS 13+

## Prerequisites

### 1. Python Environment
```bash
# Install Python 3.11+
# On Ubuntu/Debian
sudo apt update
sudo apt install python3.11 python3.11-dev python3.11-venv

# On macOS with Homebrew
brew install python@3.11
```

### 2. Poetry (Package Manager)
```bash
# Install Poetry
curl -sSL https://install.python-poetry.org | python3 -

# Add to PATH (add to .bashrc/.zshrc)
export PATH="$HOME/.local/bin:$PATH"
```

### 3. Docker & Docker Compose
```bash
# Docker installation varies by OS
# See: https://docs.docker.com/get-docker/

# Verify installation
docker --version
docker-compose --version
```

## Cloud Service Setup

Before installation, set up required cloud services:

### 1. OpenAI API
1. Create account at [platform.openai.com](https://platform.openai.com)
2. Generate API key
3. Note: Requires GPT-4 access for best results

### 2. Pinecone Vector Database
1. Sign up at [pinecone.io](https://www.pinecone.io)
2. Create an index named `memory-index`
3. Settings:
   - Dimensions: 1536 (for OpenAI embeddings)
   - Metric: Cosine
   - Pod Type: Starter (free tier) or S1

### 3. MongoDB Atlas
1. Create account at [mongodb.com/atlas](https://www.mongodb.com/atlas)
2. Create a cluster (M0 free tier works)
3. Get connection string
4. Create two databases: `memory_system` and `memory_audit`

### 4. Neo4j Aura
1. Sign up at [neo4j.com/aura](https://neo4j.com/aura)
2. Create a free instance
3. Save credentials and connection URI

### 5. TimescaleDB Cloud
1. Create account at [timescale.com](https://www.timescale.com)
2. Create a service (free tier available)
3. Get PostgreSQL connection string

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/context0/python-backend.git
cd python-backend
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use your preferred editor
```

Required environment variables:
```env
# OpenAI Configuration
OPENAI_API_KEY=sk-...your-key...
OPENAI_MODEL=gpt-4-turbo-preview

# Pinecone Configuration
PINECONE_API_KEY=...your-key...
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=memory-index

# MongoDB Configuration
MONGODB_CONNECTION_STRING=mongodb+srv://...
MONGODB_AUDIT_CONNECTION_STRING=mongodb+srv://...

# Neo4j Configuration
NEO4J_URI=neo4j+s://...databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=...your-password...
NEO4J_DATABASE=neo4j

# TimescaleDB Configuration
TIMESCALE_CONNECTION_STRING=postgresql://...

# JWT Configuration
JWT_SECRET_KEY=your-very-secret-key-min-32-chars
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30

# Application Settings
APP_NAME=Context0
APP_VERSION=1.0.0
DEBUG=false
ENVIRONMENT=production
```

### 3. Install Dependencies
```bash
# Install Python dependencies
poetry install

# For development (includes dev dependencies)
poetry install --with dev
```

### 4. Database Initialization

#### MongoDB Setup
```bash
# Collections are created automatically
# Optionally create indexes for performance
poetry run python scripts/create_indexes.py
```

#### Neo4j Setup
```bash
# Initialize graph schema
poetry run python scripts/init_neo4j.py
```

#### TimescaleDB Setup
```bash
# Run the table creation script
poetry run python recreate_timescale_table.py
```

### 5. Verify Installation
```bash
# Run the application
poetry run uvicorn app.main:app --reload

# Test the API
curl http://localhost:8000/api/v1/ping
```

## Docker Installation (Alternative)

### 1. Build Docker Image
```bash
docker build -t context0-backend .
```

### 2. Run with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f
```

### 3. Docker Compose Services
The `docker-compose.yml` includes:
- Context0 API (port 8000)
- Redis (optional, port 6379)
- Nginx (optional, port 80)

## Production Deployment

### 1. Use Production Settings
```env
DEBUG=false
ENVIRONMENT=production
LOG_LEVEL=INFO
```

### 2. Set Up Process Manager
```bash
# Install PM2
npm install -g pm2

# Start application
pm2 start "poetry run uvicorn app.main:app" --name context0
```

### 3. Configure Nginx (Optional)
```nginx
server {
    listen 80;
    server_name api.context0.ai;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## Troubleshooting

### Common Issues

1. **Poetry command not found**
   ```bash
   export PATH="$HOME/.local/bin:$PATH"
   source ~/.bashrc  # or ~/.zshrc
   ```

2. **Database connection errors**
   - Verify connection strings in `.env`
   - Check firewall/security group settings
   - Ensure IP is whitelisted in cloud services

3. **Import errors**
   ```bash
   # Clear cache and reinstall
   poetry cache clear pypi --all
   poetry install --no-cache
   ```

4. **Port already in use**
   ```bash
   # Find process using port 8000
   lsof -i :8000
   # Kill the process or use different port
   poetry run uvicorn app.main:app --port 8001
   ```

## Next Steps

- Review [Configuration Guide](./config/environment.md)
- Set up [Development Environment](./dev/setup.md)
- Explore [API Documentation](./api-overview.md)
- Read [Memory System Documentation](CONTEXT0_MEMORY_SYSTEM_DOCUMENTATION.md)

---

[← Back to Index](./index.md) | [Next: Configuration →](./config/environment.md)