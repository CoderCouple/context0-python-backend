import asyncio
import json
from functools import lru_cache
from typing import Dict, Optional, Tuple

from app.common.enum.memory import MemoryType
from app.memory.config.llm_config.llm_configs import LlmConfig

# Try to import langchain components, provide mocks if not available
try:
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_openai import ChatOpenAI

    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

    # Mock classes for when langchain is not available
    class HumanMessage:
        def __init__(self, content: str):
            self.content = content

    class SystemMessage:
        def __init__(self, content: str):
            self.content = content

    class ChatOpenAI:
        def __init__(self, **kwargs):
            self.model_name = kwargs.get("model_name", "gpt-4o-mini")

        async def apredict_messages(self, messages):
            # Return a default classification
            return "semantic_memory"

        def predict_messages(self, messages):
            return "semantic_memory"


class MemoryTypeInferencer:
    """Infers memory type using LLM classification when not explicitly provided"""

    SYSTEM_PROMPT = """You are a memory type classifier. Analyze the given text and classify it into one of these memory types:

1. semantic_memory: Factual knowledge, general information, concepts
   Examples: "The capital of France is Paris", "Water boils at 100°C"

2. episodic_memory: Specific events tied to time and context
   Examples: "I visited Tokyo last summer", "The meeting yesterday was productive"

3. procedural_memory: Instructions, procedures, how-to knowledge
   Examples: "To make coffee, first boil water", "Steps to reset a password"

4. emotional_memory: Memories with emotional content or feelings
   Examples: "I felt anxious before the presentation", "The sunset made me feel peaceful"

5. working_memory: Temporary, task-specific information for current session
   Examples: "Remember to call John at 3pm", "The code I'm debugging has a null pointer"

6. declarative_memory: Explicit facts combining semantic and episodic elements
   Examples: "I learned Python in 2020", "My flight to Paris is on Friday"

7. meta_memory: Information about memories themselves
   Examples: "I'm not sure if I remember this correctly", "This was told to me last month"

Respond with ONLY the memory type name (e.g., 'semantic_memory'). Do not include any explanation."""

    def __init__(self, llm_config: Optional[LlmConfig] = None):
        """Initialize the inferencer with LLM configuration"""
        self.llm_config = llm_config or LlmConfig(
            provider="openai", config={"model": "gpt-4o-mini", "temperature": 0.1}
        )
        self._llm_client = None

    @property
    def llm_client(self):
        """Lazy initialization of LLM client"""
        if self._llm_client is None:
            self._llm_client = self._create_llm_client()
        return self._llm_client

    def _create_llm_client(self):
        """Create the appropriate LLM client based on provider"""
        if not LANGCHAIN_AVAILABLE:
            # Return mock client if langchain not available
            return ChatOpenAI()

        provider = self.llm_config.provider

        if provider == "openai":
            config = self.llm_config.config
            return ChatOpenAI(
                model_name=config.model,
                temperature=config.temperature,
                openai_api_key=config.api_key if hasattr(config, "api_key") else None,
            )
        else:
            # Default to mock for unsupported providers
            return ChatOpenAI()

    async def infer(self, text: str) -> Tuple[MemoryType, float]:
        """
        Infer the memory type from the given text.
        Returns: (MemoryType, confidence_score)
        """
        try:
            # Use the LLM to classify the text
            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=f"Text to classify: {text}"),
            ]

            response = await self.llm_client.apredict_messages(messages)
            memory_type_str = response.strip().lower()

            # Map string to enum
            memory_type_map = {
                "semantic_memory": MemoryType.SEMANTIC_MEMORY,
                "episodic_memory": MemoryType.EPISODIC_MEMORY,
                "procedural_memory": MemoryType.PROCEDURAL_MEMORY,
                "emotional_memory": MemoryType.EMOTIONAL_MEMORY,
                "working_memory": MemoryType.WORKING_MEMORY,
                "declarative_memory": MemoryType.DECLARATIVE_MEMORY,
                "meta_memory": MemoryType.META_MEMORY,
            }

            if memory_type_str in memory_type_map:
                return memory_type_map[memory_type_str], 0.9
            else:
                # Default to semantic memory if unclear
                return MemoryType.SEMANTIC_MEMORY, 0.5

        except Exception as e:
            # If inference fails, use rule-based fallback
            return self._rule_based_inference(text)

    def _rule_based_inference(self, text: str) -> Tuple[MemoryType, float]:
        """Fallback rule-based inference when LLM is unavailable"""
        text_lower = text.lower()

        # Keywords for each memory type
        emotional_keywords = [
            "feel",
            "felt",
            "emotion",
            "happy",
            "sad",
            "angry",
            "anxious",
            "love",
            "hate",
        ]
        procedural_keywords = [
            "how to",
            "steps",
            "procedure",
            "method",
            "algorithm",
            "recipe",
            "guide",
        ]
        episodic_keywords = [
            "yesterday",
            "last week",
            "remember when",
            "that time",
            "ago",
            "visited",
        ]
        working_keywords = [
            "remind me",
            "don't forget",
            "currently",
            "right now",
            "todo",
            "task",
        ]
        meta_keywords = [
            "remember correctly",
            "forget",
            "memory",
            "recall",
            "not sure if",
        ]

        # Check for keywords
        if any(kw in text_lower for kw in emotional_keywords):
            return MemoryType.EMOTIONAL_MEMORY, 0.7
        elif any(kw in text_lower for kw in procedural_keywords):
            return MemoryType.PROCEDURAL_MEMORY, 0.7
        elif any(kw in text_lower for kw in episodic_keywords):
            return MemoryType.EPISODIC_MEMORY, 0.7
        elif any(kw in text_lower for kw in working_keywords):
            return MemoryType.WORKING_MEMORY, 0.7
        elif any(kw in text_lower for kw in meta_keywords):
            return MemoryType.META_MEMORY, 0.7
        else:
            # Default to semantic memory for factual statements
            return MemoryType.SEMANTIC_MEMORY, 0.6

    async def infer_type(
        self, text: str, context: Optional[Dict] = None
    ) -> Tuple[MemoryType, float]:
        """
        Infer the memory type from the given text with optional context.
        This is the method expected by the memory router.
        """
        return await self.infer(text)

    @lru_cache(maxsize=1000)
    def cached_inference(self, text: str) -> Tuple[MemoryType, float]:
        """Cached synchronous inference for repeated queries"""
        return asyncio.run(self.infer(text))
