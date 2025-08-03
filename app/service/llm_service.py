"""Service for managing LLM providers and presets"""
import uuid
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Type, AsyncIterator
from motor.motor_asyncio import AsyncIOMotorDatabase
import logging

from app.llm.base import BaseLLM
from app.llm.providers import OpenAILLM, AnthropicLLM, GeminiLLM
from app.llm.configs import (
    BaseLLMConfig,
    OpenAIConfig,
    AnthropicConfig,
    GoogleConfig,
    LLMPresetConfig,
)
from app.llm.types import LLMResponse, ValidationResult, TokenUsage
from app.model.llm_preset import LLMPreset, LLMUsageRecord

logger = logging.getLogger(__name__)


class LLMService:
    """Service for managing LLM providers and presets"""

    # Provider registry
    PROVIDERS: Dict[str, Type[BaseLLM]] = {
        "openai": OpenAILLM,
        "anthropic": AnthropicLLM,
        "google": GeminiLLM,
    }

    # Config classes for each provider
    CONFIG_CLASSES: Dict[str, Type[BaseLLMConfig]] = {
        "openai": OpenAIConfig,
        "anthropic": AnthropicConfig,
        "google": GoogleConfig,
    }

    # Provider information
    PROVIDER_INFO = {
        "openai": {
            "name": "OpenAI",
            "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            "supports_tools": True,
            "supports_vision": True,
            "supports_streaming": True,
            "supports_json_mode": True,
            "pricing": {
                "gpt-4o": {"input": 0.005, "output": 0.015},  # per 1K tokens
                "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
                "gpt-4-turbo": {"input": 0.01, "output": 0.03},
                "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            },
        },
        "anthropic": {
            "name": "Anthropic",
            "models": [
                "claude-3-opus-20240229",
                "claude-3-sonnet-20240229",
                "claude-3-haiku-20240307",
            ],
            "supports_tools": True,
            "supports_vision": True,
            "supports_streaming": True,
            "supports_json_mode": False,  # Via prompting only
            "pricing": {
                "claude-3-opus-20240229": {"input": 0.015, "output": 0.075},
                "claude-3-sonnet-20240229": {"input": 0.003, "output": 0.015},
                "claude-3-haiku-20240307": {"input": 0.00025, "output": 0.00125},
            },
        },
        "google": {
            "name": "Google",
            "models": ["gemini-1.5-pro", "gemini-1.5-flash", "gemini-1.0-pro"],
            "supports_tools": True,
            "supports_vision": True,
            "supports_streaming": True,
            "supports_json_mode": False,
            "pricing": {
                "gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
                "gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
                "gemini-1.0-pro": {"input": 0.0005, "output": 0.0015},
            },
        },
    }

    def __init__(self, db: AsyncIOMotorDatabase):
        """Initialize LLM service"""
        self.db = db
        self.presets_collection = db.llm_presets
        self.usage_collection = db.llm_usage
        self.prompt_configs_collection = db.prompt_configurations

        # Cache for frequently used presets
        self._preset_cache: Dict[str, LLMPreset] = {}
        self._cache_ttl = 300  # 5 minutes

    def get_provider(self, provider_name: str, config: Dict[str, Any]) -> BaseLLM:
        """
        Factory method to get LLM provider instance

        Args:
            provider_name: Name of the provider
            config: Configuration dictionary

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider is not supported
        """
        provider_class = self.PROVIDERS.get(provider_name)
        if not provider_class:
            raise ValueError(f"Unknown provider: {provider_name}")

        # Get appropriate config class
        config_class = self.CONFIG_CLASSES.get(provider_name, BaseLLMConfig)

        # Create config instance
        if isinstance(config, dict):
            config_instance = config_class(**config)
        else:
            config_instance = config

        return provider_class(config_instance)

    async def create_preset(
        self, user_id: str, preset_config: LLMPresetConfig
    ) -> LLMPreset:
        """Create a new LLM preset"""
        preset_id = str(uuid.uuid4())

        # Check if user already has a preset with this name
        existing = await self.presets_collection.find_one(
            {"user_id": user_id, "name": preset_config.name}
        )

        if existing:
            raise ValueError(f"Preset with name '{preset_config.name}' already exists")

        # Create preset
        preset = LLMPreset(id=preset_id, user_id=user_id, **preset_config.dict())

        # If this is the first preset, make it default
        user_presets_count = await self.presets_collection.count_documents(
            {"user_id": user_id}
        )
        if user_presets_count == 0:
            preset.is_default = True

        # Save to database
        await self.presets_collection.insert_one(preset.dict())

        return preset

    async def get_preset(self, preset_id: str, user_id: str) -> Optional[LLMPreset]:
        """Get a specific preset"""
        # Check cache first
        cache_key = f"{user_id}:{preset_id}"
        if cache_key in self._preset_cache:
            cached_preset, cached_time = self._preset_cache[cache_key]
            if time.time() - cached_time < self._cache_ttl:
                return cached_preset

        # Load from database
        doc = await self.presets_collection.find_one(
            {"id": preset_id, "$or": [{"user_id": user_id}, {"is_shared": True}]}
        )

        if not doc:
            return None

        preset = LLMPreset(**doc)

        # Update cache
        self._preset_cache[cache_key] = (preset, time.time())

        return preset

    async def get_user_presets(self, user_id: str) -> List[LLMPreset]:
        """Get all presets for a user"""
        cursor = self.presets_collection.find(
            {"$or": [{"user_id": user_id}, {"is_shared": True}]}
        ).sort("created_at", -1)

        presets = []
        async for doc in cursor:
            presets.append(LLMPreset(**doc))

        return presets

    async def get_user_default_preset(self, user_id: str) -> Optional[str]:
        """Get user's default preset ID"""
        doc = await self.presets_collection.find_one(
            {"user_id": user_id, "is_default": True}
        )

        return doc["id"] if doc else None

    async def get_or_create_user_default_preset(self, user_id: str) -> str:
        """Get user's default preset ID, creating one if it doesn't exist"""
        # First check if user has a default preset
        default_id = await self.get_user_default_preset(user_id)
        if default_id:
            return default_id

        # Check if user has any presets
        existing_preset = await self.presets_collection.find_one({"user_id": user_id})
        if existing_preset:
            # User has presets but no default, make the first one default
            await self.presets_collection.update_one(
                {"id": existing_preset["id"]}, {"$set": {"is_default": True}}
            )
            return existing_preset["id"]

        # No presets exist, create a default one
        from app.llm.configs import LLMPresetConfig

        default_config = LLMPresetConfig(
            name="Default",
            description="Default chat preset",
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            max_tokens=2000,
            system_prompt="You are a helpful AI assistant.",
            use_memory_context=True,
            extract_memories=True,
            memory_extraction_types=["semantic_memory", "episodic_memory"],
        )

        preset = await self.create_preset(user_id, default_config)
        return preset.id

    async def update_preset(
        self, preset_id: str, user_id: str, updates: Dict[str, Any]
    ) -> bool:
        """Update a preset"""
        # Remove fields that shouldn't be updated
        updates.pop("id", None)
        updates.pop("user_id", None)
        updates.pop("created_at", None)

        # Set updated timestamp
        updates["updated_at"] = datetime.utcnow()

        # Update in database
        result = await self.presets_collection.update_one(
            {"id": preset_id, "user_id": user_id}, {"$set": updates}
        )

        # Clear cache
        cache_key = f"{user_id}:{preset_id}"
        self._preset_cache.pop(cache_key, None)

        return result.modified_count > 0

    async def delete_preset(self, preset_id: str, user_id: str) -> bool:
        """Delete a preset"""
        # Check if it's the default preset
        preset = await self.get_preset(preset_id, user_id)
        if not preset:
            return False

        if preset.is_default:
            # Find another preset to make default
            other_preset = await self.presets_collection.find_one(
                {"user_id": user_id, "id": {"$ne": preset_id}}
            )

            if other_preset:
                await self.presets_collection.update_one(
                    {"id": other_preset["id"]}, {"$set": {"is_default": True}}
                )

        # Delete preset
        result = await self.presets_collection.delete_one(
            {"id": preset_id, "user_id": user_id}
        )

        # Clear cache
        cache_key = f"{user_id}:{preset_id}"
        self._preset_cache.pop(cache_key, None)

        return result.deleted_count > 0

    async def generate_with_preset(
        self,
        preset_id: str,
        messages: List[Dict[str, str]],
        user_id: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> LLMResponse:
        """Generate response using a preset"""
        start_time = time.time()

        # Get preset
        preset = await self.get_preset(preset_id, user_id)
        if not preset:
            # Try to get default preset
            default_id = await self.get_user_default_preset(user_id)
            if default_id:
                preset = await self.get_preset(default_id, user_id)
            else:
                # Create a default preset
                default_config = LLMPresetConfig(
                    name="Default", description="Default preset"
                )
                preset = await self.create_preset(user_id, default_config)

        # Get provider
        # Pass the preset config, not the full preset dict which includes metadata
        preset_config = {
            "provider": preset.provider,
            "model": preset.model,
            "temperature": preset.temperature,
            "max_tokens": preset.max_tokens,
            "top_p": preset.top_p,
            "top_k": preset.top_k,
            # Don't include API key in preset for security
        }
        provider = self.get_provider(preset.provider, preset_config)

        # Apply preset parameters with kwargs overrides
        generation_params = {
            "temperature": kwargs.get("temperature", preset.temperature),
            "max_tokens": kwargs.get("max_tokens", preset.max_tokens),
            "top_p": kwargs.get("top_p", preset.top_p),
            "top_k": kwargs.get("top_k", preset.top_k),
            "tools": kwargs.get("tools"),
            "response_format": kwargs.get("response_format"),
            "stream": kwargs.get("stream", False),
        }

        try:
            # Generate response
            response = await provider.generate_response(messages, **generation_params)

            # Track usage
            response_time = int((time.time() - start_time) * 1000)
            await self._track_usage(
                user_id=user_id,
                preset=preset,
                response=response,
                session_id=session_id,
                response_time_ms=response_time,
                request_type="chat",
                success=True,
            )

            # Update preset usage stats
            await self._update_preset_stats(preset_id, response)

            # Apply reranking if enabled
            if preset.reranking_enabled and not kwargs.get("skip_reranking"):
                response = await self._rerank_response(
                    response,
                    preset.rerank_threshold,
                    messages[-1]["content"] if messages else "",
                )

            return response

        except Exception as e:
            # Track failed usage
            response_time = int((time.time() - start_time) * 1000)
            await self._track_usage(
                user_id=user_id,
                preset=preset,
                response=None,
                session_id=session_id,
                response_time_ms=response_time,
                request_type="chat",
                success=False,
                error_message=str(e),
            )
            raise

    async def generate_stream_with_preset(
        self,
        preset_id: str,
        messages: List[Dict[str, str]],
        user_id: str,
        session_id: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream response using a preset"""
        start_time = time.time()

        # Get preset
        preset = await self.get_preset(preset_id, user_id)
        if not preset:
            raise ValueError("Preset not found")

        # Get provider
        # Pass the preset config, not the full preset dict which includes metadata
        preset_config = {
            "provider": preset.provider,
            "model": preset.model,
            "temperature": preset.temperature,
            "max_tokens": preset.max_tokens,
            "top_p": preset.top_p,
            "top_k": preset.top_k,
            # Don't include API key in preset for security
        }
        provider = self.get_provider(preset.provider, preset_config)

        # Apply preset parameters
        generation_params = {
            "temperature": kwargs.get("temperature", preset.temperature),
            "max_tokens": kwargs.get("max_tokens", preset.max_tokens),
            "top_p": kwargs.get("top_p", preset.top_p),
            "top_k": kwargs.get("top_k", preset.top_k),
            "tools": kwargs.get("tools"),
            "response_format": kwargs.get("response_format"),
        }

        # Stream response
        total_content = ""
        async for chunk in provider.generate_stream(messages, **generation_params):
            total_content += chunk
            yield chunk

        # Track usage after streaming completes
        response_time = int((time.time() - start_time) * 1000)

        # Create a response object for tracking
        response = LLMResponse(
            content=total_content,
            usage=TokenUsage(
                prompt_tokens=await provider.count_tokens(str(messages)),
                completion_tokens=await provider.count_tokens(total_content),
                total_tokens=0,
            ),
            model=preset.model,
            provider=preset.provider,
        )
        response.usage.total_tokens = (
            response.usage.prompt_tokens + response.usage.completion_tokens
        )

        await self._track_usage(
            user_id=user_id,
            preset=preset,
            response=response,
            session_id=session_id,
            response_time_ms=response_time,
            request_type="chat_stream",
            success=True,
        )

        await self._update_preset_stats(preset_id, response)

    async def _track_usage(
        self,
        user_id: str,
        preset: LLMPreset,
        response: Optional[LLMResponse],
        session_id: Optional[str],
        response_time_ms: int,
        request_type: str,
        success: bool,
        error_message: Optional[str] = None,
    ):
        """Track LLM usage for billing and analytics"""
        # Calculate costs
        input_cost = 0.0
        output_cost = 0.0

        if response and success:
            pricing = (
                self.PROVIDER_INFO.get(preset.provider, {})
                .get("pricing", {})
                .get(preset.model, {})
            )
            if pricing:
                input_cost = (response.usage.prompt_tokens / 1000) * pricing.get(
                    "input", 0
                )
                output_cost = (response.usage.completion_tokens / 1000) * pricing.get(
                    "output", 0
                )

        # Create usage record
        usage_record = LLMUsageRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            preset_id=preset.id,
            session_id=session_id,
            provider=preset.provider,
            model=preset.model,
            input_tokens=response.usage.prompt_tokens if response else 0,
            output_tokens=response.usage.completion_tokens if response else 0,
            total_tokens=response.usage.total_tokens if response else 0,
            input_cost=input_cost,
            output_cost=output_cost,
            total_cost=input_cost + output_cost,
            request_type=request_type,
            success=success,
            error_message=error_message,
            response_time_ms=response_time_ms,
        )

        await self.usage_collection.insert_one(usage_record.dict())

    async def _update_preset_stats(self, preset_id: str, response: LLMResponse):
        """Update preset usage statistics"""
        await self.presets_collection.update_one(
            {"id": preset_id},
            {
                "$inc": {
                    "usage_count": 1,
                    "total_tokens_used": response.usage.total_tokens,
                },
                "$set": {"last_used_at": datetime.utcnow()},
            },
        )

    async def _rerank_response(
        self, response: LLMResponse, threshold: float, original_query: str
    ) -> LLMResponse:
        """Apply reranking to response (placeholder for now)"""
        # TODO: Implement actual reranking logic
        # This could involve:
        # 1. Breaking response into segments
        # 2. Scoring each segment for relevance
        # 3. Reordering or filtering based on scores
        # 4. Reconstructing the response

        logger.info(f"Reranking response with threshold {threshold}")
        return response

    async def get_user_usage_stats(
        self,
        user_id: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """Get usage statistics for a user"""
        if not start_date:
            start_date = datetime.utcnow() - timedelta(days=30)
        if not end_date:
            end_date = datetime.utcnow()

        # Aggregate usage data
        pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "timestamp": {"$gte": start_date, "$lte": end_date},
                }
            },
            {
                "$group": {
                    "_id": {"provider": "$provider", "model": "$model"},
                    "request_count": {"$sum": 1},
                    "total_tokens": {"$sum": "$total_tokens"},
                    "total_cost": {"$sum": "$total_cost"},
                    "success_count": {"$sum": {"$cond": ["$success", 1, 0]}},
                    "avg_response_time": {"$avg": "$response_time_ms"},
                }
            },
        ]

        cursor = self.usage_collection.aggregate(pipeline)
        usage_by_model = []

        async for doc in cursor:
            usage_by_model.append(
                {
                    "provider": doc["_id"]["provider"],
                    "model": doc["_id"]["model"],
                    "request_count": doc["request_count"],
                    "total_tokens": doc["total_tokens"],
                    "total_cost": doc["total_cost"],
                    "success_rate": doc["success_count"] / doc["request_count"]
                    if doc["request_count"] > 0
                    else 0,
                    "avg_response_time_ms": doc["avg_response_time"],
                }
            )

        # Get total stats
        total_pipeline = [
            {
                "$match": {
                    "user_id": user_id,
                    "timestamp": {"$gte": start_date, "$lte": end_date},
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_requests": {"$sum": 1},
                    "total_tokens": {"$sum": "$total_tokens"},
                    "total_cost": {"$sum": "$total_cost"},
                }
            },
        ]

        total_cursor = self.usage_collection.aggregate(total_pipeline)
        total_stats = await total_cursor.next()

        return {
            "period": {"start": start_date.isoformat(), "end": end_date.isoformat()},
            "total": {
                "requests": total_stats["total_requests"] if total_stats else 0,
                "tokens": total_stats["total_tokens"] if total_stats else 0,
                "cost": total_stats["total_cost"] if total_stats else 0.0,
            },
            "by_model": usage_by_model,
        }

    @classmethod
    def get_available_providers(cls) -> Dict[str, Any]:
        """Get information about available providers"""
        return cls.PROVIDER_INFO
