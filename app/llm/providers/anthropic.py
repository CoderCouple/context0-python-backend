"""Anthropic Claude LLM provider implementation"""
import os
import json
from typing import List, Dict, Optional, Any, AsyncIterator
from anthropic import AsyncAnthropic

from app.llm.base import BaseLLM
from app.llm.configs import AnthropicConfig, BaseLLMConfig
from app.llm.types import LLMResponse, TokenUsage, ToolCall


class AnthropicLLM(BaseLLM):
    """Anthropic Claude provider implementation"""

    def __init__(self, config: Optional[AnthropicConfig] = None):
        """Initialize Anthropic client"""
        if config is None:
            config = AnthropicConfig()
        elif isinstance(config, dict):
            config = AnthropicConfig(**config)
        elif isinstance(config, BaseLLMConfig) and not isinstance(
            config, AnthropicConfig
        ):
            # Convert base config to Anthropic config
            config = AnthropicConfig(**config.dict())

        super().__init__(config)

        # Initialize client
        self.client = AsyncAnthropic(
            api_key=config.api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=config.api_base,
            timeout=config.timeout,
        )

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
        """Generate response using Anthropic API"""

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
                content=content,
                usage=usage,
                model=self.config.model,
                provider="anthropic",
            )

        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                # Claude uses a separate system parameter
                system_message = msg["content"]
            else:
                anthropic_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

        # Add JSON instruction for Claude if needed
        if response_format and response_format.get("type") == "json_object":
            json_instruction = "\n\nYou MUST respond with valid JSON only. No explanations, no markdown code blocks, just pure JSON."
            if system_message:
                system_message += json_instruction
            else:
                system_message = json_instruction

        # Build request parameters
        params = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": self._get_param("max_tokens", max_tokens),
            "temperature": self._get_param("temperature", temperature),
        }

        if system_message:
            params["system"] = system_message

        # Claude uses top_k directly, but not top_p (it uses top_p as default)
        if top_k is not None:
            params["top_k"] = top_k
        elif top_p is not None:
            params["top_p"] = top_p

        # Add tools if provided (Claude's tool format)
        if tools:
            params["tools"] = self._convert_tools_to_claude_format(tools)
            params["tool_choice"] = {"type": "auto"}

        # Make API call
        response = await self.client.messages.create(**params)

        # Extract content and tool calls
        content = ""
        tool_calls = []

        for content_block in response.content:
            if content_block.type == "text":
                content += content_block.text
            elif content_block.type == "tool_use":
                tool_calls.append(
                    ToolCall(
                        id=content_block.id,
                        name=content_block.name,
                        arguments=content_block.input,
                    )
                )

        # Build response
        return LLMResponse(
            content=content,
            usage=TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            ),
            model=response.model,
            provider="anthropic",
            tool_calls=tool_calls if tool_calls else None,
            metadata={"stop_reason": response.stop_reason, "response_id": response.id},
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

        # Convert messages to Anthropic format
        system_message = None
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_message = msg["content"]
            else:
                anthropic_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

        # Add JSON instruction if needed
        if response_format and response_format.get("type") == "json_object":
            json_instruction = "\n\nYou MUST respond with valid JSON only."
            if system_message:
                system_message += json_instruction
            else:
                system_message = json_instruction

        # Build request parameters
        params = {
            "model": self.config.model,
            "messages": anthropic_messages,
            "max_tokens": self._get_param("max_tokens", max_tokens),
            "temperature": self._get_param("temperature", temperature),
            "stream": True,
        }

        if system_message:
            params["system"] = system_message

        if top_k is not None:
            params["top_k"] = top_k
        elif top_p is not None:
            params["top_p"] = top_p

        if tools:
            params["tools"] = self._convert_tools_to_claude_format(tools)
            params["tool_choice"] = {"type": "auto"}

        # Stream response
        async with self.client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                yield text

    async def count_tokens(self, text: str) -> int:
        """Estimate token count for Claude (rough approximation)"""
        # Claude doesn't provide a public tokenizer
        # Use approximation: ~4 characters per token
        return len(text) // 4

    def _convert_tools_to_claude_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI tool format to Claude format"""
        claude_tools = []
        for tool in tools:
            claude_tool = {
                "name": tool["function"]["name"],
                "description": tool["function"]["description"],
                "input_schema": tool["function"]["parameters"],
            }
            claude_tools.append(claude_tool)
        return claude_tools
