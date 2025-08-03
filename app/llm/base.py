"""Base LLM abstraction for all providers"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any, AsyncIterator

from app.llm.configs import BaseLLMConfig
from app.llm.types import LLMResponse


class BaseLLM(ABC):
    """Base class for all LLM providers"""

    def __init__(self, config: Optional[BaseLLMConfig] = None):
        """Initialize with configuration"""
        self.config = config or BaseLLMConfig()

    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        response_format: Optional[Dict] = None,
        stream: bool = False,
    ) -> LLMResponse:
        """
        Generate response from messages

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            tools: Available tools/functions
            response_format: Desired response format (e.g., {"type": "json_object"})
            stream: Whether to stream the response

        Returns:
            LLMResponse object with generated content
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        response_format: Optional[Dict] = None,
    ) -> AsyncIterator[str]:
        """
        Stream response generation

        Args:
            Same as generate_response

        Yields:
            String chunks of the response
        """
        pass

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        """Count tokens in text for this provider's tokenizer"""
        pass

    def _get_param(self, param_name: str, override_value: Optional[Any]) -> Any:
        """Get parameter value with override priority"""
        if override_value is not None:
            return override_value
        return getattr(self.config, param_name, None)
