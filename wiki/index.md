# Context0 Python Backend Wiki

Welcome to the Context0 Python Backend documentation wiki. This index provides quick access to all documentation resources.

## 📚 Table of Contents

### 🚀 Getting Started
- [Project Overview](./overview.md) - Introduction to Context0 backend
- [Quick Start Guide](./quickstart.md) - Get up and running quickly
- [Installation & Setup](./installation.md) - Detailed setup instructions

### 🏗️ Architecture & Design
- [System Architecture](./architecture.md) - High-level system design
- [Memory System Documentation](./memory-system.md) - Deep dive into the memory system
- [Database Design](./database-design.md) - MongoDB, Neo4j, TimescaleDB schemas

### 📡 API Documentation
- [API Overview](./api-overview.md) - RESTful API design principles
- [Authentication](./authentication.md) - JWT-based auth system
- **Endpoints:**
  - [Memory API](./api/memory-api.md) - Memory management endpoints
  - [Chat API](./api/chat-api.md) - Chat session management
  - [Analytics API](./api/analytics-api.md) - Usage analytics
  - [Billing API](./api/billing-api.md) - Billing and usage tracking
  - [Q&A API](./api/qa-api.md) - Question-answering system
  - [Webhook API](./api/webhook-api.md) - Webhook integrations

### 🧠 Core Features
- [Memory Engine](./features/memory-engine.md) - Memory processing pipeline
- [Chat System](./features/chat-system.md) - AI-powered conversations
- [Memory Categorization](./features/memory-categorization.md) - Automatic categorization
- [Reasoning Engine](./features/reasoning-engine.md) - Multi-hop reasoning
- [Vector Search](./features/vector-search.md) - Semantic search capabilities

### 🔧 Configuration
- [Environment Variables](./config/environment.md) - Required env vars
- [Memory Configuration](./config/memory-config.md) - YAML configuration
- [Service Configuration](./config/services.md) - External service setup

### 🛠️ Development
- [Development Setup](./dev/setup.md) - Local development environment
- [Testing Guide](./dev/testing.md) - Running tests
- [Code Style Guide](./dev/code-style.md) - Coding standards
- [Contributing](./dev/contributing.md) - How to contribute

### 🚢 Deployment
- [Deployment Guide](./deployment/guide.md) - Production deployment
- [Docker Setup](./deployment/docker.md) - Containerization
- [Scaling Considerations](./deployment/scaling.md) - Performance optimization
- [Monitoring](./deployment/monitoring.md) - Logging and monitoring

### 📊 Database Guides
- [MongoDB Setup](./databases/mongodb.md) - Document store configuration
- [Neo4j Setup](./databases/neo4j.md) - Graph database configuration
- [TimescaleDB Setup](./databases/timescaledb.md) - Time-series configuration
- [Pinecone Setup](./databases/pinecone.md) - Vector database configuration

### 🔍 Troubleshooting
- [Common Issues](./troubleshooting/common-issues.md) - Frequent problems
- [Error Codes](./troubleshooting/error-codes.md) - Error reference
- [Performance Issues](./troubleshooting/performance.md) - Performance debugging

### 📝 Migration Guides
- [Database Migrations](./migrations/database.md) - Schema updates
- [API Migrations](./migrations/api.md) - API version updates

## 📄 Core Documentation

### Existing Documentation Files
These are the comprehensive documentation files in the project root:

1. **[Project README](../readme.md)**
   - Project overview and basic setup
   - Quick installation steps
   - Basic usage examples

2. **[Memory System Documentation](CONTEXT0_MEMORY_SYSTEM_DOCUMENTATION.md)**
   - Comprehensive memory system architecture
   - Detailed component descriptions
   - Advanced configuration options
   - Implementation details

3. **[Memory Setup Guide](MEMORY_SETUP.md)**
   - Step-by-step setup instructions
   - Environment configuration
   - Database initialization
   - Troubleshooting guide

4. **[Chat API Documentation](./api/chat-api.md)**
   - Complete chat endpoint reference
   - Request/response examples
   - Integration guide

## 🔗 Quick Links

### API Testing
- [Postman Collection](./assets/postman-collection.json) *(to be added)*
- [OpenAPI Spec](./assets/openapi.yaml) *(to be added)*

### Important Files
- [`pyproject.toml`](../pyproject.toml) - Python dependencies
- [`docker-compose.yml`](../docker-compose.yml) - Local services
- [`.env.example`](../.env.example) - Environment template

## 🎯 Getting Help

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/context0/python-backend/issues)
- **Discussions**: Join conversations on [GitHub Discussions](https://github.com/context0/python-backend/discussions)
- **Email**: support@context0.ai

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

---

*Last updated: July 2025*