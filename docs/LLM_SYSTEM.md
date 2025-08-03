# LLM Provider System Documentation

## Overview

The LLM Provider System provides a unified interface for working with multiple language model providers (OpenAI, Anthropic Claude, Google Gemini) with advanced features like:

- **Provider Abstraction**: Switch between providers without changing code
- **Preset Management**: Save and reuse LLM configurations
- **Prompt Templates**: Structured prompts with JSON validation
- **Memory Extraction**: Automatic extraction of structured information
- **Cost Tracking**: Monitor usage and costs across providers
- **Category System**: Custom categories for memory organization

## Architecture

```
app/llm/
├── base.py              # Abstract base class for all providers
├── configs.py           # Configuration models
├── types.py             # Common types and interfaces
├── providers/
│   ├── openai.py        # OpenAI implementation
│   ├── anthropic.py     # Claude implementation
│   └── gemini.py        # Google Gemini implementation
└── prompts/
    ├── base.py          # Prompt template system
    ├── memory_templates.py  # Pre-built memory extraction templates
    └── category_builder.py  # Custom category prompt builder
```

## API Endpoints

### Provider Information
- `GET /api/v1/llm/providers` - List available providers and models

### Preset Management
- `GET /api/v1/llm/presets` - Get user's presets
- `POST /api/v1/llm/presets` - Create new preset
- `GET /api/v1/llm/presets/{id}` - Get specific preset
- `PUT /api/v1/llm/presets/{id}` - Update preset
- `DELETE /api/v1/llm/presets/{id}` - Delete preset
- `POST /api/v1/llm/presets/{id}/test` - Test preset

### Usage Statistics
- `GET /api/v1/llm/usage/stats` - Get usage statistics

## Configuration

### Environment Variables
```bash
# OpenAI
OPENAI_API_KEY=sk-...

# Anthropic Claude
ANTHROPIC_API_KEY=sk-ant-...

# Google Gemini
GOOGLE_API_KEY=...

# Defaults
DEFAULT_LLM_PROVIDER=openai
DEFAULT_LLM_MODEL=gpt-4o-mini
DEFAULT_TEMPERATURE=0.1
DEFAULT_MAX_TOKENS=2048
```

## Usage Examples

### Creating a Preset
```python
POST /api/v1/llm/presets
{
    "name": "Creative Writer",
    "description": "For creative writing tasks",
    "provider": "anthropic",
    "model": "claude-3-sonnet-20240229",
    "temperature": 0.8,
    "system_prompt": "You are a creative writer...",
    "categories": [
        {
            "name": "characters",
            "description": "Character descriptions",
            "includes_prompt": "character, person, protagonist",
            "excludes_prompt": "plot, setting"
        }
    ]
}
```

### Using in Chat
```python
POST /api/v1/chat/sessions/{session_id}/messages
{
    "content": "Tell me a story",
    "llm_preset_id": "preset_123",  // Optional - uses default if not specified
    "extract_memories": true,
    "use_memory_context": true
}
```

### Testing a Preset
```python
POST /api/v1/llm/presets/{preset_id}/test
{
    "message": "Hello, how are you?",
    "use_memory_context": false,
    "temperature": 0.5  // Optional override
}
```

## Memory Extraction

The system can automatically extract different types of memories:

### Built-in Memory Types
- **semantic_memory**: Facts, concepts, knowledge
- **episodic_memory**: Events, experiences
- **procedural_memory**: How-to instructions
- **graph_memory**: Entities and relationships
- **chat_memory**: General conversation memories

### Custom Categories
Users can define custom categories with inclusion/exclusion rules:

```python
{
    "name": "technical_terms",
    "description": "Technical terminology",
    "includes_prompt": "API, framework, library, function",
    "excludes_prompt": "personal, opinion"
}
```

## Prompt Templates

### Using Built-in Templates
```python
from app.llm.prompts import SEMANTIC_MEMORY_TEMPLATE

rendered = SEMANTIC_MEMORY_TEMPLATE.render(
    user_name="John",
    content="FastAPI is a modern web framework...",
    domain="programming"
)
```

### Creating Custom Templates
```python
template = PromptTemplate(
    name="custom_extractor",
    template="Extract {{type}} from: {{content}}",
    variables=[
        PromptVariable(name="type", required=True),
        PromptVariable(name="content", required=True)
    ],
    output_schema={...}  # JSON schema for validation
)
```

## Cost Management

### Pricing Information
The system tracks costs based on provider pricing:

- **OpenAI GPT-4**: $0.01/1K input, $0.03/1K output tokens
- **Claude 3 Sonnet**: $0.003/1K input, $0.015/1K output tokens
- **Gemini 1.5 Flash**: $0.000075/1K input, $0.0003/1K output tokens

### Usage Tracking
```python
GET /api/v1/llm/usage/stats?start_date=2024-01-01

Response:
{
    "total": {
        "requests": 1234,
        "tokens": 567890,
        "cost": 12.34
    },
    "by_model": [...],
    "projected_monthly_cost": 45.67
}
```

## Best Practices

1. **Preset Organization**: Create presets for different use cases
2. **Temperature Settings**: Use low temperature (0.1) for factual tasks, higher (0.7-0.9) for creative
3. **Token Limits**: Set appropriate limits to control costs
4. **Memory Extraction**: Choose relevant memory types for your use case
5. **Category Design**: Create specific categories for domain-specific extraction
6. **Cost Monitoring**: Regularly check usage statistics

## Error Handling

The system handles various error cases:
- Invalid API keys
- Rate limiting
- Token limits exceeded
- Invalid JSON responses
- Network timeouts

All errors are logged and returned with appropriate error messages.

## Testing

Run the test script to verify the system:
```bash
python test_llm_system.py
```

This tests:
- Provider connectivity
- Preset creation
- Basic generation
- Memory extraction
- Prompt templates
- JSON validation
- Usage tracking