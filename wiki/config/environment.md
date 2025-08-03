# Environment Configuration Guide

[← Back to index](../index.md)

## Overview

Context0 uses environment-specific configuration files to manage different deployment environments. This approach ensures clear separation between development and production settings.

## Environment Files

### File Structure
```
.env.dev      # Development configuration (default)
.env.prod     # Production configuration
.env.example  # Template with all options
.env          # Legacy (deprecated)
```

### Environment Selection

The application automatically loads the correct .env file based on the `APP_ENV` variable:

- `APP_ENV=development` (default) → loads `.env.dev`
- `APP_ENV=production` → loads `.env.prod`

## Quick Start

### 1. Initial Setup
```bash
# Run the setup helper
python setup_env.py

# Or manually copy the example
cp .env.example .env.dev
```

### 2. Configure Development Environment
Edit `.env.dev` with your local settings:
```env
# Core settings
AUTH_DISABLED=true  # Disable auth for local development
DEBUG=true
LOG_LEVEL=INFO

# Local MongoDB (Docker)
MONGODB_CONNECTION_STRING=mongodb://localhost:27017

# Or use MongoDB Atlas
# MONGODB_CONNECTION_STRING=mongodb+srv://user:pass@cluster.mongodb.net/
```

### 3. Start Development Server
```bash
# Uses .env.dev by default
poetry run uvicorn app.main:app --reload
```

### 4. Switch to Production
```bash
# Set environment variable
export APP_ENV=production

# Start with production config
poetry run uvicorn app.main:app
```

## Configuration Options

### Application Settings
| Variable | Description | Dev Default | Prod Default |
|----------|-------------|-------------|--------------|
| `APP_ENV` | Environment name | development | production |
| `DEBUG` | Debug mode | true | false |
| `LOG_LEVEL` | Logging level | INFO | WARNING |
| `AUTH_DISABLED` | Disable authentication | true | false |
| `CORS_ORIGINS` | Allowed origins | ["http://localhost:3000"] | ["https://app.context0.ai"] |

### Database Configuration

#### MongoDB (Required)
```env
# Local development
MONGODB_CONNECTION_STRING=mongodb://localhost:27017
MONGODB_DATABASE_NAME=context0

# Production (MongoDB Atlas)
MONGODB_CONNECTION_STRING=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DATABASE_NAME=context0_prod
```

#### PostgreSQL (For SQLAlchemy models)
```env
DB_HOST=localhost
DB_PORT=5432
DB_USER=postgres
DB_PASSWORD=password
DB_NAME=context0
```

### Memory System Databases

#### Neo4j (Graph Database)
```env
# Local
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Production (Neo4j Aura)
NEO4J_URI=neo4j+s://your-instance.databases.neo4j.io
```

#### TimescaleDB (Time Series)
```env
TIMESCALE_CONNECTION_STRING=postgresql://user:pass@host:port/db?sslmode=require
```

#### Pinecone (Vector Database)
```env
PINECONE_API_KEY=your-api-key
PINECONE_ENVIRONMENT=us-east-1-aws
PINECONE_INDEX_NAME=memory-index
```

### AI Services
```env
# OpenAI (Required for chat and embeddings)
OPENAI_API_KEY=sk-your-api-key

# Optional
ANTHROPIC_API_KEY=sk-ant-your-key
GOOGLE_API_KEY=your-google-key
```

## Development Setup

### Using Docker Services
1. Start local services:
```bash
docker-compose up -d
```

2. Access management UIs:
- MongoDB Express: http://localhost:8081
- Neo4j Browser: http://localhost:7474
- pgAdmin: http://localhost:8080

### Using Cloud Services in Dev
You can use cloud services even in development by updating `.env.dev`:
```env
# Use production databases in dev
MONGODB_CONNECTION_STRING=mongodb+srv://...
NEO4J_URI=neo4j+s://...
```

## Production Deployment

### 1. Create Production Config
```bash
cp .env.example .env.prod
# Edit with production values
```

### 2. Security Checklist
- [ ] Set `AUTH_DISABLED=false`
- [ ] Set `DEBUG=false`
- [ ] Use strong passwords
- [ ] Enable SSL/TLS for all connections
- [ ] Set proper CORS origins
- [ ] Generate secure JWT secret

### 3. Deploy with Production Settings
```bash
# On production server
export APP_ENV=production
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Environment Variables Reference

See [.env.example](../../.env.example) for a complete list of all available environment variables with descriptions.

## Troubleshooting

### Application using wrong config
```bash
# Check current environment
echo $APP_ENV

# Force reload
export APP_ENV=development
# or
export APP_ENV=production
```

### MongoDB authentication errors
1. Check connection string includes credentials
2. Ensure database user has correct permissions
3. Verify network access (IP whitelist for Atlas)

### Missing .env files
```bash
# Run setup script
python setup_env.py

# Or create manually
cp .env.example .env.dev
```

## Best Practices

1. **Never commit sensitive data**: All .env files are gitignored
2. **Use .env.example as reference**: Keep it updated with new variables
3. **Different API keys**: Use separate keys for dev/prod
4. **Regular backups**: Backup production .env.prod securely
5. **Environment validation**: Test configuration before deployment

---

[← Back to index](../index.md) | [Next: Memory Configuration →](./memory-config.md)