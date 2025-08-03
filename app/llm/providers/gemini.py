"""Google Gemini LLM provider implementation"""
import os
import json
from typing import List, Dict, Optional, Any, AsyncIterator
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import asyncio

from app.llm.base import BaseLLM
from app.llm.configs import GoogleConfig, BaseLLMConfig
from app.llm.types import LLMResponse, TokenUsage, ToolCall


class GeminiLLM(BaseLLM):
    """Google Gemini provider implementation"""

    def __init__(self, config: Optional[GoogleConfig] = None):
        """Initialize Gemini client"""
        if config is None:
            config = GoogleConfig()
        elif isinstance(config, dict):
            config = GoogleConfig(**config)
        elif isinstance(config, BaseLLMConfig) and not isinstance(config, GoogleConfig):
            # Convert base config to Google config
            config = GoogleConfig(**config.dict())

        super().__init__(config)

        # Configure API key
        genai.configure(api_key=config.api_key or os.getenv("GOOGLE_API_KEY"))

        # Initialize model
        generation_config = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "max_output_tokens": config.max_tokens,
        }

        # Set up safety settings
        safety_settings = self._get_safety_settings(config)

        self.model = genai.GenerativeModel(
            model_name=config.model,
            generation_config=generation_config,
            safety_settings=safety_settings,
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
        """Generate response using Gemini API"""

        if stream:
            # Use streaming method instead
            content = ""
            async for chunk in self.generate_stream(
                messages, temperature, max_tokens, top_p, top_k, tools, response_format
            ):
                content += chunk

            # Estimate tokens
            usage = TokenUsage(
                prompt_tokens=await self.count_tokens(str(messages)),
                completion_tokens=await self.count_tokens(content),
                total_tokens=0,
            )
            usage.total_tokens = usage.prompt_tokens + usage.completion_tokens

            return LLMResponse(
                content=content, usage=usage, model=self.config.model, provider="google"
            )

        # Convert messages to Gemini format
        gemini_messages = self._convert_messages_to_gemini(messages)

        # Add JSON instruction if needed
        if response_format and response_format.get("type") == "json_object":
            json_instruction = "\n\nIMPORTANT: Respond with valid JSON only. No explanations, no markdown, just JSON."
            if gemini_messages:
                if isinstance(gemini_messages[-1], str):
                    gemini_messages[-1] += json_instruction
                else:
                    gemini_messages[-1]["parts"][-1]["text"] += json_instruction

        # Update generation config with overrides
        generation_config = {
            "temperature": self._get_param("temperature", temperature),
            "top_p": self._get_param("top_p", top_p),
            "top_k": self._get_param("top_k", top_k),
            "max_output_tokens": self._get_param("max_tokens", max_tokens),
        }

        # Create a new model instance with updated config
        model = genai.GenerativeModel(
            model_name=self.config.model,
            generation_config=generation_config,
            safety_settings=self._get_safety_settings(self.config),
        )

        # Add tools if provided
        if tools:
            # Convert tools to Gemini format
            gemini_tools = self._convert_tools_to_gemini_format(tools)
            model = genai.GenerativeModel(
                model_name=self.config.model,
                generation_config=generation_config,
                safety_settings=self._get_safety_settings(self.config),
                tools=gemini_tools,
            )

        # Generate response (run in thread pool for async compatibility)
        response = await asyncio.get_event_loop().run_in_executor(
            None, model.generate_content, gemini_messages
        )

        # Extract content and tool calls
        content = response.text if response.text else ""
        tool_calls = None

        # Check for function calls in response
        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                tool_calls_list = []
                for part in candidate.content.parts:
                    if hasattr(part, "function_call"):
                        tool_calls_list.append(
                            ToolCall(
                                id=f"call_{len(tool_calls_list)}",
                                name=part.function_call.name,
                                arguments=dict(part.function_call.args),
                            )
                        )
                if tool_calls_list:
                    tool_calls = tool_calls_list

        # Get token usage
        if hasattr(response, "usage_metadata"):
            usage = TokenUsage(
                prompt_tokens=response.usage_metadata.prompt_token_count,
                completion_tokens=response.usage_metadata.candidates_token_count,
                total_tokens=response.usage_metadata.total_token_count,
            )
        else:
            # Estimate if not available
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
            provider="google",
            tool_calls=tool_calls,
            metadata={
                "finish_reason": response.candidates[0].finish_reason.name
                if response.candidates
                else "UNKNOWN"
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

        # Convert messages to Gemini format
        gemini_messages = self._convert_messages_to_gemini(messages)

        # Add JSON instruction if needed
        if response_format and response_format.get("type") == "json_object":
            json_instruction = "\n\nIMPORTANT: Respond with valid JSON only."
            if gemini_messages:
                if isinstance(gemini_messages[-1], str):
                    gemini_messages[-1] += json_instruction
                else:
                    gemini_messages[-1]["parts"][-1]["text"] += json_instruction

        # Update generation config
        generation_config = {
            "temperature": self._get_param("temperature", temperature),
            "top_p": self._get_param("top_p", top_p),
            "top_k": self._get_param("top_k", top_k),
            "max_output_tokens": self._get_param("max_tokens", max_tokens),
        }

        # Create model with config
        model = genai.GenerativeModel(
            model_name=self.config.model,
            generation_config=generation_config,
            safety_settings=self._get_safety_settings(self.config),
        )

        if tools:
            gemini_tools = self._convert_tools_to_gemini_format(tools)
            model = genai.GenerativeModel(
                model_name=self.config.model,
                generation_config=generation_config,
                safety_settings=self._get_safety_settings(self.config),
                tools=gemini_tools,
            )

        # Generate streaming response
        response_stream = await asyncio.get_event_loop().run_in_executor(
            None, model.generate_content, gemini_messages, {"stream": True}
        )

        # Yield chunks
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text

    async def count_tokens(self, text: str) -> int:
        """Count tokens using Gemini's count_tokens method"""
        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, self.model.count_tokens, text
            )
            return result.total_tokens
        except:
            # Fallback to approximation
            return len(text) // 4

    def _convert_messages_to_gemini(self, messages: List[Dict[str, str]]) -> List[Any]:
        """Convert OpenAI-style messages to Gemini format"""
        gemini_messages = []
        system_prompt = ""

        for msg in messages:
            if msg["role"] == "system":
                # Gemini doesn't have system role, prepend to first user message
                system_prompt += msg["content"] + "\n\n"
            elif msg["role"] == "user":
                content = (
                    system_prompt + msg["content"] if system_prompt else msg["content"]
                )
                gemini_messages.append({"role": "user", "parts": [{"text": content}]})
                system_prompt = ""  # Reset after using
            elif msg["role"] == "assistant":
                gemini_messages.append(
                    {"role": "model", "parts": [{"text": msg["content"]}]}
                )

        # If there's remaining system prompt, add it as a user message
        if system_prompt:
            gemini_messages.append(
                {"role": "user", "parts": [{"text": system_prompt.strip()}]}
            )

        return gemini_messages

    def _get_safety_settings(self, config: GoogleConfig) -> List[Dict]:
        """Get safety settings for Gemini"""
        if config.safety_settings:
            return config.safety_settings

        # Default safety settings based on harm_block_threshold
        threshold_map = {
            "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
            "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "BLOCK_HIGH": HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        threshold = threshold_map.get(
            config.harm_block_threshold, HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE
        )

        return [
            {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": threshold},
            {
                "category": HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                "threshold": threshold,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                "threshold": threshold,
            },
            {
                "category": HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                "threshold": threshold,
            },
        ]

    def _convert_tools_to_gemini_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI tool format to Gemini format"""
        # Gemini uses a different tool format
        # This is a simplified conversion - may need adjustment based on actual Gemini API
        gemini_tools = []
        for tool in tools:
            if tool["type"] == "function":
                gemini_tools.append(
                    {
                        "function_declarations": [
                            {
                                "name": tool["function"]["name"],
                                "description": tool["function"]["description"],
                                "parameters": tool["function"]["parameters"],
                            }
                        ]
                    }
                )
        return gemini_tools
