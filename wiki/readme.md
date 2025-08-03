# Context0 Documentation Wiki

Welcome to the Context0 documentation wiki! This folder contains comprehensive documentation for the Context0 Python backend.

## 📍 Start Here

**[index.md](./index.md)** - The main entry point with links to all documentation

## 📚 Documentation Structure

### Core Documentation Files (Project Root)
- [`../readme.md`](../readme.md) - Project overview
- [`../CONTEXT0_MEMORY_SYSTEM_DOCUMENTATION.md`](CONTEXT0_MEMORY_SYSTEM_DOCUMENTATION.md) - Detailed memory system architecture
- [`../MEMORY_SETUP.md`](MEMORY_SETUP.md) - Setup instructions

### Wiki Organization
```
wiki/
├── index.md          # Main table of contents
├── sitemap.md        # Visual site structure
├── readme.md         # This file
├── overview.md       # Project overview
├── quickstart.md     # Quick start guide
├── installation.md   # Detailed installation
├── memory-system.md  # Memory system docs
├── api-overview.md   # API introduction
│
├── api/              # API documentation
│   ├── memory-api.md
│   ├── chat-api.md
│   └── ...
│
├── features/         # Feature documentation
├── config/           # Configuration guides
├── databases/        # Database setup guides
├── deployment/       # Deployment guides
├── dev/              # Developer guides
├── troubleshooting/  # Problem solving
└── migrations/       # Migration guides
```

## 🔍 Finding Information

### By Role
- **New Users**: Start with [Overview](./overview.md) → [Quick Start](./quickstart.md)
- **Developers**: See [API Documentation](./api-overview.md) and [Development Setup](./dev/setup.md)
- **System Admins**: Check [Installation](./installation.md) and [Deployment](./deployment/guide.md)

### By Topic
- **Memory System**: [Memory System Docs](./memory-system.md), [Memory API](./api/memory-api.md)
- **Chat Features**: [Chat API](./api/chat-api.md)
- **Configuration**: [Environment Variables](./config/environment.md)
- **Databases**: Individual guides in [databases/](./databases/)

## 📝 Documentation Standards

### File Naming
- Use lowercase with hyphens: `memory-system.md`
- API docs: `api/{endpoint}-api.md`
- Keep names descriptive but concise

### Content Structure
Each document should include:
1. Navigation links (back to index)
2. Clear headings with hierarchy
3. Code examples where relevant
4. Links to related documents

### Navigation Pattern
```markdown
[← Back to Index](./index.md) | [Next: Topic →](./next-topic.md)
```

## 🤝 Contributing

To add or update documentation:
1. Follow the existing structure
2. Update index.md with new pages
3. Include navigation links
4. Use clear, concise language
5. Add code examples

## 🔗 Quick Links

- **Main Index**: [index.md](./index.md)
- **Visual Sitemap**: [sitemap.md](./sitemap.md)
- **API Overview**: [api-overview.md](./api-overview.md)
- **Quick Start**: [quickstart.md](./quickstart.md)

---

*For the latest updates and to contribute, visit the [GitHub repository](https://github.com/context0/python-backend).*