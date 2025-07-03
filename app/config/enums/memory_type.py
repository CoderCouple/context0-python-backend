# context0-python-backend/app/config/memory_type.py
from enum import Enum

# ┌──────────────────────┬──────────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┐
# │       Type           │                        Description                           │                      Use Case Example                      │
# ├──────────────────────┼──────────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
# │ semantic_memory      │ Factual knowledge chunks stored by meaning                   │ "Tokyo is in Japan"                                        │
# │ episodic_memory      │ Memories tied to time & events (event_id, timestamp, etc.)   │ "User visited Tokyo on July 3"                             │
# │ procedural_memory    │ Instructions or learned behavior ("how to do X")             │ "Steps to plan a trip"                                     │
# │ declarative_memory   │ General knowledge (explicit facts: semantic + episodic)      │ "I know that I booked a flight to Tokyo"                   │
# │ working_memory       │ Short-term task-relevant memory for a session                │ Temporary info needed during a single chat flow            │
# │ emotional_memory     │ Memory with emotional valence or affective weight            │ "User felt anxious before their trip"                      │
# │ meta_memory          │ Memories about memory itself (confidence, age, source, etc.) │ "This was learned 3 months ago from user chat"             │
# └──────────────────────┴──────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────┘


# Enum for types of memory in the system
class MemoryType(str, Enum):
    # Stores facts or knowledge chunks that are generalized and independent of specific experiences
    SEMANTIC_MEMORY = "semantic_memory"

    # Captures specific events tied to time and context (who, when, where)
    EPISODIC_MEMORY = "episodic_memory"

    # Encodes procedures or steps, i.e., "how-to" knowledge
    PROCEDURAL_MEMORY = "procedural_memory"

    # Stores both episodic and semantic info — broadly declarable facts (platform wrapper)
    DECLARATIVE_MEMORY = "declarative_memory"

    # Temporary, short-lived memory used for active tasks within a session
    WORKING_MEMORY = "working_memory"

    # Tracks emotional states and their associated contexts (for mood tracking, preference modeling)
    EMOTIONAL_MEMORY = "emotional_memory"

    # Meta-information about memories (e.g., source, confidence, recency)
    META_MEMORY = "meta_memory"
