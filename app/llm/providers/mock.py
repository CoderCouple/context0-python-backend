"""Mock LLM provider for testing without API keys"""
import asyncio
from typing import List, Dict, Any, Optional, AsyncIterator
import uuid
from datetime import datetime

from app.llm.base import BaseLLM
from app.llm.types import LLMResponse
from app.llm.configs import MockConfig


class MockLLM(BaseLLM):
    """Mock LLM for testing purposes"""

    def __init__(self, config: Optional[MockConfig] = None):
        """Initialize mock LLM"""
        self.config = config or MockConfig()
        self.provider = "mock"
        self.default_model = "mock-model"

    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate a mock response"""
        # Simulate processing delay
        await asyncio.sleep(self.config.response_delay)

        # Get the last user message
        user_message = next(
            (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
            "Hello",
        )

        # Generate mock response based on content
        if "2+2" in user_message:
            content = "4"
        elif "memory" in user_message.lower() or response_format:
            # Return JSON for memory extraction
            content = """{
    "memories": [
        {
            "text": "John Smith is a software engineer at TechCorp",
            "category": "semantic_memory",
            "tags": ["professional", "identity"],
            "importance": 0.9
        },
        {
            "text": "Prefers morning meetings",
            "category": "preferences",
            "tags": ["schedule", "meetings"],
            "importance": 0.7
        }
    ]
}"""
        else:
            content = f"Mock response to: {user_message[:50]}..."

        # Calculate mock token counts
        prompt_tokens = sum(len(msg["content"].split()) * 1.3 for msg in messages)
        completion_tokens = len(content.split()) * 1.3

        return LLMResponse(
            id=f"mock-{uuid.uuid4()}",
            object="chat.completion",
            created=int(datetime.now().timestamp()),
            model=self.config.model or self.default_model,
            content=content,
            finish_reason="stop",
            provider="mock",
            usage={
                "prompt_tokens": int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens": int(prompt_tokens + completion_tokens),
                "prompt_cost": prompt_tokens * 0.00001,
                "completion_cost": completion_tokens * 0.00002,
                "total_cost": (prompt_tokens * 0.00001) + (completion_tokens * 0.00002),
            },
        )

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[Dict[str, Any]]:
        """Generate a mock streaming response"""
        # Get response
        response = await self.generate_response(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            system_prompt=system_prompt,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

        # Simulate streaming by yielding words
        words = response.content.split()
        for i, word in enumerate(words):
            await asyncio.sleep(self.config.stream_delay)

            yield {
                "id": response.id,
                "object": "chat.completion.chunk",
                "created": response.created,
                "model": response.model,
                "delta": {"content": word + (" " if i < len(words) - 1 else "")},
                "finish_reason": "stop" if i == len(words) - 1 else None,
            }

    def count_tokens(self, text: str) -> int:
        """Mock token counting"""
        # Simple approximation: 1 token ≈ 4 characters
        return len(text) // 4

    def get_model_info(self, model: str) -> Dict[str, Any]:
        """Get mock model information"""
        return {
            "name": "Mock Model",
            "description": "A mock model for testing",
            "context_window": 8192,
            "max_output_tokens": 4096,
            "supports_tools": True,
            "supports_streaming": True,
            "supports_json_mode": True,
            "pricing": {"input_cost_per_1k": 0.01, "output_cost_per_1k": 0.02},
        }

    @classmethod
    def get_available_models(cls) -> List[str]:
        """Get list of available mock models"""
        return ["mock-model", "mock-model-fast", "mock-model-smart"]
