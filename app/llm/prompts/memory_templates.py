"""Pre-defined prompt templates for memory extraction"""
from app.llm.prompts.base import PromptTemplate
from app.llm.types import PromptVariable


# Graph Memory Extraction Template
GRAPH_MEMORY_TEMPLATE = PromptTemplate(
    name="graph_memory_extraction",
    description="Extract entities and relationships for graph memory",
    template="""Extract entities and relationships from the following text.

Context:
User: {{ user_name }}
Session: {{ session_id }}
Timestamp: {{ timestamp }}

Text to analyze:
{{ content }}

Extract ALL entities (people, places, concepts) and their relationships.

IMPORTANT: Return ONLY valid JSON matching this exact structure:
{
  "entities": [
    {
      "id": "unique_id",
      "name": "entity name",
      "type": "person|place|concept|event|organization",
      "properties": {}
    }
  ],
  "relationships": [
    {
      "source_id": "entity1_id",
      "target_id": "entity2_id",
      "relationship_type": "knows|located_at|works_for|related_to|etc",
      "properties": {
        "strength": 0.0-1.0,
        "context": "description"
      }
    }
  ],
  "summary": "brief summary of the graph structure"
}""",
    variables=[
        PromptVariable(name="user_name", description="User's name", required=True),
        PromptVariable(name="session_id", description="Chat session ID", required=True),
        PromptVariable(
            name="timestamp", description="Current timestamp", required=True
        ),
        PromptVariable(name="content", description="Content to analyze", required=True),
    ],
    output_schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["entities", "relationships", "summary"],
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["id", "name", "type"],
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": [
                                "person",
                                "place",
                                "concept",
                                "event",
                                "organization",
                            ],
                        },
                        "properties": {"type": "object"},
                    },
                },
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["source_id", "target_id", "relationship_type"],
                    "properties": {
                        "source_id": {"type": "string"},
                        "target_id": {"type": "string"},
                        "relationship_type": {"type": "string"},
                        "properties": {
                            "type": "object",
                            "properties": {
                                "strength": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "context": {"type": "string"},
                            },
                        },
                    },
                },
            },
            "summary": {"type": "string"},
        },
    },
    tags=["memory", "graph", "extraction"],
)

# Semantic Memory Extraction Template
SEMANTIC_MEMORY_TEMPLATE = PromptTemplate(
    name="semantic_memory_extraction",
    description="Extract facts and knowledge for semantic memory",
    template="""Extract key facts and knowledge from the following text.

Context:
User: {{ user_name }}
Domain: {{ domain }}
{% if previous_context %}Previous Context: {{ previous_context }}{% endif %}

Text: {{ content }}

Extract structured information and return ONLY valid JSON:
{
  "facts": [
    {
      "statement": "fact statement",
      "confidence": 0.0-1.0,
      "category": "technical|personal|general|domain-specific",
      "tags": ["tag1", "tag2"],
      "source_quote": "exact quote from text"
    }
  ],
  "concepts": [
    {
      "name": "concept name",
      "definition": "brief definition",
      "related_concepts": ["concept1", "concept2"]
    }
  ],
  "questions_raised": ["question 1", "question 2"],
  "action_items": ["action 1", "action 2"]
}""",
    variables=[
        PromptVariable(name="user_name", description="User's name", required=True),
        PromptVariable(
            name="domain",
            description="Domain context",
            required=True,
            default_value="general",
        ),
        PromptVariable(
            name="previous_context", description="Previous context", required=False
        ),
        PromptVariable(name="content", description="Content to analyze", required=True),
    ],
    output_schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["facts", "concepts", "questions_raised", "action_items"],
        "properties": {
            "facts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["statement", "confidence", "category", "tags"],
                    "properties": {
                        "statement": {"type": "string"},
                        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                        "category": {
                            "type": "string",
                            "enum": [
                                "technical",
                                "personal",
                                "general",
                                "domain-specific",
                            ],
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "source_quote": {"type": "string"},
                    },
                },
            },
            "concepts": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "definition"],
                    "properties": {
                        "name": {"type": "string"},
                        "definition": {"type": "string"},
                        "related_concepts": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                },
            },
            "questions_raised": {"type": "array", "items": {"type": "string"}},
            "action_items": {"type": "array", "items": {"type": "string"}},
        },
    },
    tags=["memory", "semantic", "extraction", "knowledge"],
)

# Episodic Memory Extraction Template
EPISODIC_MEMORY_TEMPLATE = PromptTemplate(
    name="episodic_memory_extraction",
    description="Extract episodic information from conversations",
    template="""Extract episodic memory information from this conversation.

Time: {{ timestamp }}
Participants: {{ participants }}
{% if location %}Location: {{ location }}{% endif %}

Conversation: {{ content }}

Extract episode details and return ONLY valid JSON:
{
  "episode": {
    "title": "brief descriptive title",
    "when": "temporal information",
    "where": "location information",
    "who": ["participants"],
    "what_happened": "narrative summary",
    "emotional_tone": "positive|negative|neutral|mixed",
    "importance": 0.0-1.0
  },
  "key_moments": [
    {
      "timestamp": "when in conversation",
      "description": "what happened",
      "significance": "why it matters"
    }
  ],
  "connections": ["related memory IDs or topics"]
}""",
    variables=[
        PromptVariable(
            name="timestamp", description="Conversation timestamp", required=True
        ),
        PromptVariable(
            name="participants",
            description="Participants in conversation",
            required=True,
        ),
        PromptVariable(
            name="location", description="Location of event", required=False
        ),
        PromptVariable(
            name="content", description="Conversation content", required=True
        ),
    ],
    output_schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["episode", "key_moments", "connections"],
        "properties": {
            "episode": {
                "type": "object",
                "required": [
                    "title",
                    "when",
                    "where",
                    "who",
                    "what_happened",
                    "emotional_tone",
                    "importance",
                ],
                "properties": {
                    "title": {"type": "string"},
                    "when": {"type": "string"},
                    "where": {"type": "string"},
                    "who": {"type": "array", "items": {"type": "string"}},
                    "what_happened": {"type": "string"},
                    "emotional_tone": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral", "mixed"],
                    },
                    "importance": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
            "key_moments": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["timestamp", "description", "significance"],
                    "properties": {
                        "timestamp": {"type": "string"},
                        "description": {"type": "string"},
                        "significance": {"type": "string"},
                    },
                },
            },
            "connections": {"type": "array", "items": {"type": "string"}},
        },
    },
    tags=["memory", "episodic", "extraction", "conversation"],
)

# Procedural Memory Extraction Template
PROCEDURAL_MEMORY_TEMPLATE = PromptTemplate(
    name="procedural_memory_extraction",
    description="Extract step-by-step procedures and instructions",
    template="""Extract step-by-step procedures or instructions.

Context: {{ context }}
Text: {{ content }}

Extract procedures and return ONLY valid JSON:
{
  "procedures": [
    {
      "name": "procedure name",
      "goal": "what it accomplishes",
      "prerequisites": ["requirement1", "requirement2"],
      "steps": [
        {
          "order": 1,
          "action": "what to do",
          "details": "how to do it",
          "warnings": ["potential issues"],
          "expected_result": "what should happen"
        }
      ],
      "tools_required": ["tool1", "tool2"],
      "estimated_time": "duration",
      "difficulty": "easy|medium|hard"
    }
  ]
}""",
    variables=[
        PromptVariable(
            name="context", description="Context for the procedure", required=True
        ),
        PromptVariable(
            name="content", description="Content containing procedures", required=True
        ),
    ],
    output_schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["procedures"],
        "properties": {
            "procedures": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "goal", "steps"],
                    "properties": {
                        "name": {"type": "string"},
                        "goal": {"type": "string"},
                        "prerequisites": {"type": "array", "items": {"type": "string"}},
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["order", "action"],
                                "properties": {
                                    "order": {"type": "integer", "minimum": 1},
                                    "action": {"type": "string"},
                                    "details": {"type": "string"},
                                    "warnings": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "expected_result": {"type": "string"},
                                },
                            },
                        },
                        "tools_required": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "estimated_time": {"type": "string"},
                        "difficulty": {
                            "type": "string",
                            "enum": ["easy", "medium", "hard"],
                        },
                    },
                },
            }
        },
    },
    tags=["memory", "procedural", "extraction", "instructions"],
)

# Chat Memory Extraction Template (General Purpose)
CHAT_MEMORY_TEMPLATE = PromptTemplate(
    name="chat_memory_extraction",
    description="General purpose memory extraction from chat messages",
    template="""Analyze this {{ role }} message and identify any important information that should be remembered.

Message: {{ content }}
{% if session_context %}Session Context: {{ session_context }}{% endif %}

Extract:
1. Facts, preferences, or personal information
2. Important events or plans
3. Decisions or commitments
4. Learning or insights

Return ONLY a JSON array of memory objects:
{
  "memories": [
    {
      "text": "memory content",
      "category": "fact|preference|event|decision|insight|general",
      "tags": ["tag1", "tag2"],
      "importance": 0.0-1.0,
      "entities": ["entity1", "entity2"]
    }
  ]
}

Return {"memories": []} if nothing important to remember.""",
    variables=[
        PromptVariable(
            name="role", description="Message role (user/assistant)", required=True
        ),
        PromptVariable(name="content", description="Message content", required=True),
        PromptVariable(
            name="session_context", description="Session context", required=False
        ),
    ],
    output_schema={
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "required": ["memories"],
        "properties": {
            "memories": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["text", "category", "tags", "importance"],
                    "properties": {
                        "text": {"type": "string"},
                        "category": {
                            "type": "string",
                            "enum": [
                                "fact",
                                "preference",
                                "event",
                                "decision",
                                "insight",
                                "general",
                            ],
                        },
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "importance": {"type": "number", "minimum": 0, "maximum": 1},
                        "entities": {"type": "array", "items": {"type": "string"}},
                    },
                },
            }
        },
    },
    tags=["memory", "chat", "extraction", "general"],
)

# Collection of all built-in templates
MEMORY_EXTRACTION_TEMPLATES = {
    "graph_memory": GRAPH_MEMORY_TEMPLATE,
    "semantic_memory": SEMANTIC_MEMORY_TEMPLATE,
    "episodic_memory": EPISODIC_MEMORY_TEMPLATE,
    "procedural_memory": PROCEDURAL_MEMORY_TEMPLATE,
    "chat_memory": CHAT_MEMORY_TEMPLATE,
}
