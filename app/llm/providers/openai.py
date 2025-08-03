"""OpenAI LLM provider implementation"""
import os
import json
from typing import List, Dict, Optional, Any, AsyncIterator
from openai import AsyncOpenAI
import tiktoken

from app.llm.base import BaseLLM
from app.llm.configs import OpenAIConfig, BaseLLMConfig
from app.llm.types import LLMResponse, TokenUsage, ToolCall


class OpenAILLM(BaseLLM):
    """OpenAI provider implementation"""

    def __init__(self, config: Optional[OpenAIConfig] = None):
        """Initialize OpenAI client"""
        if config is None:
            config = OpenAIConfig()
        elif isinstance(config, dict):
            config = OpenAIConfig(**config)
        elif isinstance(config, BaseLLMConfig) and not isinstance(config, OpenAIConfig):
            # Convert base config to OpenAI config
            config = OpenAIConfig(**config.dict())

        super().__init__(config)

        # Initialize client
        api_key = config.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")

        self.client = AsyncOpenAI(
            api_key=api_key,
            organization=config.organization,
            base_url=config.api_base,
            timeout=config.timeout,
        )

        # Initialize tokenizer for token counting
        try:
            self.encoding = tiktoken.encoding_for_model(config.model)
        except:
            self.encoding = tiktoken.get_encoding("cl100k_base")

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
        """Generate response using OpenAI API"""

        if stream:
            # Use streaming method instead
            content = ""
            async for chunk in self.generate_stream(
                messages, temperature, max_tokens, top_p, top_k, tools, response_format
            ):
                content += chunk

            # Estimate tokens for streaming response
            usage = TokenUsage(
                prompt_tokens=await self.count_tokens(str(messages)),
                completion_tokens=await self.count_tokens(content),
                total_tokens=0,
            )
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

            return LLMResponse(
                content=content, usage=usage, model=self.config.model, provider="openai"
            )

        # Build request parameters
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self._get_param("temperature", temperature),
            "max_tokens": self._get_param("max_tokens", max_tokens),
            "top_p": self._get_param("top_p", top_p),
        }

        # Add OpenAI-specific parameters
        if hasattr(self.config, "presence_penalty"):
            params["presence_penalty"] = self.config.presence_penalty
        if hasattr(self.config, "frequency_penalty"):
            params["frequency_penalty"] = self.config.frequency_penalty
        if hasattr(self.config, "logit_bias") and self.config.logit_bias:
            params["logit_bias"] = self.config.logit_bias
        if hasattr(self.config, "seed") and self.config.seed:
            params["seed"] = self.config.seed

        # Add tools if provided
        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        # Add response format if specified
        if response_format:
            params["response_format"] = response_format
            # For JSON mode, add instruction to the last user message
            if response_format.get("type") == "json_object":
                if messages and messages[-1]["role"] == "user":
                    messages[-1][
                        "content"
                    ] += "\n\nIMPORTANT: Respond with valid JSON only."

        # Make API call
        response = await self.client.chat.completions.create(**params)

        # Extract tool calls if present
        tool_calls = None
        if response.choices[0].message.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                )
                for tc in response.choices[0].message.tool_calls
            ]

        # Build response
        return LLMResponse(
            content=response.choices[0].message.content or "",
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
            model=response.model,
            provider="openai",
            tool_calls=tool_calls,
            metadata={
                "finish_reason": response.choices[0].finish_reason,
                "response_id": response.id,
            },
        )

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
        """Stream response generation"""

        # Build request parameters (similar to generate_response)
        params = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self._get_param("temperature", temperature),
            "max_tokens": self._get_param("max_tokens", max_tokens),
            "top_p": self._get_param("top_p", top_p),
            "stream": True,
        }

        # Add OpenAI-specific parameters
        if hasattr(self.config, "presence_penalty"):
            params["presence_penalty"] = self.config.presence_penalty
        if hasattr(self.config, "frequency_penalty"):
            params["frequency_penalty"] = self.config.frequency_penalty

        if tools:
            params["tools"] = tools
            params["tool_choice"] = "auto"

        if response_format:
            params["response_format"] = response_format

        # Stream response
        async for chunk in await self.client.chat.completions.create(**params):
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def count_tokens(self, text: str) -> int:
        """Count tokens using tiktoken"""
        return len(self.encoding.encode(text))
