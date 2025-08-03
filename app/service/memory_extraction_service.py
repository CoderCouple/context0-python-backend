"""Service for extracting memories using LLM with prompt templates"""
import asyncio
import uuid
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.service.llm_service import LLMService
from app.llm.prompts import (
    PromptTemplate,
    CategoryPromptBuilder,
    MEMORY_EXTRACTION_TEMPLATES,
)
from app.llm.types import CategoryConfig, ValidationResult
from app.model.llm_preset import PromptConfiguration
from app.api.v1.request.memory_request import MemoryRecordInput
from app.memory.engine.memory_engine import MemoryEngine
from app.common.enum.memory_category import MemoryCategory
from app.memory.extraction.smart_extraction_filter import SmartExtractionFilter

logger = logging.getLogger(__name__)


class MemoryExtractionService:
    """Service for extracting structured memories from text using LLM"""

    def __init__(self, llm_service: LLMService, db: AsyncIOMotorDatabase):
        """Initialize memory extraction service"""
        self.llm_service = llm_service
        self.smart_filter = SmartExtractionFilter(llm_service)
        self.db = db
        self.prompt_configs_collection = db.prompt_configurations

        # Built-in templates
        self.templates = MEMORY_EXTRACTION_TEMPLATES.copy()

        # Custom templates cache
        self._custom_templates: Dict[str, PromptTemplate] = {}

        # Category prompt builder
        self.category_builder = CategoryPromptBuilder()

        # Memory engine for storing extracted memories
        self._memory_engine: Optional[MemoryEngine] = None

    @property
    def memory_engine(self) -> MemoryEngine:
        """Get memory engine instance"""
        if not self._memory_engine:
            self._memory_engine = MemoryEngine.get_instance()
        return self._memory_engine

    async def extract_memories(
        self,
        content: str,
        memory_types: List[str],
        user_id: str,
        context: Dict[str, Any],
        preset_id: Optional[str] = None,
        categories: Optional[List[CategoryConfig]] = None,
    ) -> Dict[str, Any]:
        """
        Extract multiple types of memories from content

        Args:
            content: Text to extract memories from
            memory_types: Types of memories to extract
            user_id: User ID
            context: Context variables for templates
            preset_id: LLM preset to use
            categories: Custom categories to extract

        Returns:
            Dictionary with extracted memories by type
        """
        results = {}

        # Add custom categories to memory types
        if categories:
            for category in categories:
                category_name = f"category_{category.name}"
                # Register custom template
                template = self.category_builder.build_extraction_prompt(category)
                self.register_template(category_name, template)
                if category_name not in memory_types:
                    memory_types.append(category_name)

        # Extract each memory type in parallel
        extraction_tasks = []
        for memory_type in memory_types:
            task = self.extract_single_type(
                content=content,
                memory_type=memory_type,
                user_id=user_id,
                context=context,
                preset_id=preset_id,
            )
            extraction_tasks.append((memory_type, task))

        # Execute all extractions in parallel
        if extraction_tasks:
            task_results = await asyncio.gather(
                *[task for _, task in extraction_tasks], return_exceptions=True
            )

            # Process results
            for i, (memory_type, _) in enumerate(extraction_tasks):
                result = task_results[i]
                if isinstance(result, Exception):
                    logger.error(f"Error extracting {memory_type}: {result}")
                    results[memory_type] = {"error": str(result)}
                elif result and result.get("success"):
                    results[memory_type] = result["data"]
                else:
                    logger.warning(f"Failed to extract {memory_type}: {result}")

        return results

    async def extract_single_type(
        self,
        content: str,
        memory_type: str,
        user_id: str,
        context: Dict[str, Any],
        preset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract a single type of memory"""
        # Get template
        template = self.get_template(memory_type)
        if not template:
            return {
                "success": False,
                "error": f"No template found for memory type: {memory_type}",
            }

        # Add default context values
        default_context = {
            "user_name": context.get("user_name", "User"),
            "timestamp": context.get("timestamp", datetime.utcnow().isoformat()),
            "session_id": context.get("session_id", ""),
            "domain": context.get("domain", "general"),
            "participants": context.get("participants", "User and Assistant"),
            "context": context.get("context", "General conversation"),
            "location": context.get("location", ""),
            "previous_context": context.get("previous_context", ""),
        }

        # Merge contexts
        full_context = {**default_context, **context, "content": content}

        try:
            # Render prompt
            prompt = template.render(**full_context)
        except Exception as e:
            return {"success": False, "error": f"Failed to render template: {e}"}

        # Generate with LLM
        messages = [
            {
                "role": "system",
                "content": """You are a memory extraction specialist. Extract ONLY explicit personal information from the text.

IMPORTANT RULES:
1. Only extract what is EXPLICITLY stated in the text
2. Do NOT infer, assume, or imagine information
3. If the text is just a greeting or acknowledgment, extract NOTHING
4. Only extract facts about the USER, not general knowledge
5. Be conservative - when in doubt, don't extract

Examples:
- "I love pizza" → Extract: User loves pizza
- "Hey" → Extract: NOTHING
- "Tell me about Python" → Extract: NOTHING (just a question)
- "I work as a developer" → Extract: User works as a developer""",
            },
            {"role": "user", "content": prompt},
        ]

        # Use specific preset or default memory extractor preset
        if not preset_id:
            preset_id = await self._get_or_create_extractor_preset(user_id)

        try:
            # Check if we should use JSON mode
            preset = await self.llm_service.get_preset(preset_id, user_id)
            use_json_mode = False

            # Only use JSON mode for models that support it
            if preset and preset.model in [
                "gpt-4-0125-preview",
                "gpt-4-turbo-preview",
                "gpt-4-turbo",
                "gpt-3.5-turbo-0125",
            ]:
                use_json_mode = True

            # Prepare generation params
            gen_params = {
                "preset_id": preset_id,
                "messages": messages,
                "user_id": user_id,
                "temperature": 0.1,  # Low temperature for consistent extraction
                "max_tokens": 2000,
            }

            if use_json_mode:
                gen_params["response_format"] = {"type": "json_object"}

            response = await self.llm_service.generate_with_preset(**gen_params)

            # Validate and parse response
            validation_result = template.extract_and_validate(response.content)

            if validation_result.is_valid:
                return {
                    "success": True,
                    "data": validation_result.data,
                    "memory_type": memory_type,
                }
            else:
                # Retry with stronger prompt
                return await self._retry_with_validation(
                    template, messages, user_id, preset_id, validation_result.errors
                )

        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {"success": False, "error": f"LLM generation failed: {e}"}

    async def _retry_with_validation(
        self,
        template: PromptTemplate,
        messages: List[Dict[str, str]],
        user_id: str,
        preset_id: str,
        errors: List[str],
    ) -> Dict[str, Any]:
        """Retry extraction with schema in prompt"""
        # Add schema and errors to prompt
        schema_prompt = (
            f"\n\nJSON Schema:\n{json.dumps(template.output_schema, indent=2)}"
        )
        error_prompt = f"\n\nPrevious attempt had errors:\n" + "\n".join(errors)
        retry_prompt = f"\n\nPlease fix these errors and return ONLY valid JSON matching the schema."

        messages[-1]["content"] += schema_prompt + error_prompt + retry_prompt

        try:
            # Check if we should use JSON mode
            preset = await self.llm_service.get_preset(preset_id, user_id)
            use_json_mode = False

            # Only use JSON mode for models that support it
            if preset and preset.model in [
                "gpt-4-0125-preview",
                "gpt-4-turbo-preview",
                "gpt-4-turbo",
                "gpt-3.5-turbo-0125",
            ]:
                use_json_mode = True

            # Prepare generation params
            gen_params = {
                "preset_id": preset_id,
                "messages": messages,
                "user_id": user_id,
                "temperature": 0.0,  # Zero temperature for retry
            }

            if use_json_mode:
                gen_params["response_format"] = {"type": "json_object"}

            response = await self.llm_service.generate_with_preset(**gen_params)

            validation_result = template.extract_and_validate(response.content)

            return {
                "success": validation_result.is_valid,
                "data": validation_result.data if validation_result.is_valid else None,
                "errors": validation_result.errors,
            }

        except Exception as e:
            return {"success": False, "error": f"Retry failed: {e}"}

    async def extract_and_store_memories(
        self,
        content: str,
        memory_types: List[str],
        user_id: str,
        session_id: str,
        context: Dict[str, Any],
        preset_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract memories and store them in the memory system"""

        # First check if content has personal information worth extracting
        should_extract, reason, confidence = self.smart_filter.should_extract(
            content, user_id
        )

        if not should_extract:
            logger.info(f"Skipping extraction: {reason} (confidence: {confidence})")
            return {"extracted": {}, "stored": {}, "skipped": True, "reason": reason}

        logger.info(f"Extracting memories: {reason} (confidence: {confidence})")

        # Extract memories
        extracted = await self.extract_memories(
            content=content,
            memory_types=memory_types,
            user_id=user_id,
            context=context,
            preset_id=preset_id,
        )

        stored_memories = {}

        # Store each type
        for memory_type, data in extracted.items():
            if isinstance(data, dict) and "error" not in data:
                stored = await self._store_memory_type(
                    memory_type=memory_type,
                    data=data,
                    user_id=user_id,
                    session_id=session_id,
                )
                stored_memories[memory_type] = stored

        return {"extracted": extracted, "stored": stored_memories}

    async def _store_memory_type(
        self, memory_type: str, data: Dict[str, Any], user_id: str, session_id: str
    ) -> List[str]:
        """Store extracted memories based on type"""
        memory_ids = []

        if memory_type == "graph_memory":
            # Store graph data
            # TODO: Implement graph storage
            pass

        elif memory_type == "semantic_memory":
            # Store facts
            for fact in data.get("facts", []):
                memory_input = MemoryRecordInput(
                    user_id=user_id,
                    session_id=session_id,
                    text=fact["statement"],
                    category=MemoryCategory.EDUCATION,  # Use EDUCATION category for semantic/knowledge facts
                    tags=fact.get("tags", []) + ["semantic", "fact"],
                    metadata={
                        "confidence": fact.get("confidence", 1.0),
                        "source_quote": fact.get("source_quote"),
                        "fact_category": fact.get("category", "general"),
                    },
                )

                result = await self.memory_engine.add_memory(memory_input)
                if result.success:
                    memory_ids.append(result.memory_id)

            # Store concepts
            for concept in data.get("concepts", []):
                memory_input = MemoryRecordInput(
                    user_id=user_id,
                    session_id=session_id,
                    text=f"{concept['name']}: {concept['definition']}",
                    category=MemoryCategory.EDUCATION,  # Use EDUCATION category for concepts
                    tags=["semantic", "concept"] + concept.get("related_concepts", []),
                    metadata={
                        "concept_name": concept["name"],
                        "related_concepts": concept.get("related_concepts", []),
                    },
                )

                result = await self.memory_engine.add_memory(memory_input)
                if result.success:
                    memory_ids.append(result.memory_id)

        elif memory_type == "episodic_memory":
            # Store episode
            episode = data.get("episode", {})
            if episode:
                memory_input = MemoryRecordInput(
                    user_id=user_id,
                    session_id=session_id,
                    text=episode.get("what_happened", ""),
                    category=MemoryCategory.PERSONAL,  # Use PERSONAL category for episodic memories
                    tags=[
                        "episodic",
                        "event",
                        episode.get("emotional_tone", "neutral"),
                    ],
                    metadata={
                        "title": episode.get("title"),
                        "when": episode.get("when"),
                        "where": episode.get("where"),
                        "who": episode.get("who", []),
                        "importance": episode.get("importance", 0.5),
                        "key_moments": data.get("key_moments", []),
                    },
                )

                result = await self.memory_engine.add_memory(memory_input)
                if result.success:
                    memory_ids.append(result.memory_id)

        elif memory_type == "chat_memory":
            # Store general chat memories
            for memory in data.get("memories", []):
                memory_input = MemoryRecordInput(
                    user_id=user_id,
                    session_id=session_id,
                    text=memory["text"],
                    category=MemoryCategory.GENERAL,
                    tags=memory.get("tags", [])
                    + ["chat", memory.get("category", "general")],
                    metadata={
                        "importance": memory.get("importance", 0.5),
                        "entities": memory.get("entities", []),
                        "memory_category": memory.get("category", "general"),
                    },
                )

                result = await self.memory_engine.add_memory(memory_input)
                if result.success:
                    memory_ids.append(result.memory_id)

        elif memory_type.startswith("category_"):
            # Store custom category memories
            for item in data.get("extracted_items", []):
                memory_input = MemoryRecordInput(
                    user_id=user_id,
                    session_id=session_id,
                    text=item["text"],
                    category=MemoryCategory.CUSTOM,
                    tags=[memory_type.replace("category_", ""), "custom"]
                    + item.get("metadata", {}).get("tags", []),
                    metadata={
                        "confidence": item.get("confidence", 1.0),
                        "reason": item.get("reason"),
                        "custom_category": memory_type.replace("category_", ""),
                        **item.get("metadata", {}),
                    },
                )

                result = await self.memory_engine.add_memory(memory_input)
                if result.success:
                    memory_ids.append(result.memory_id)

        return memory_ids

    def get_template(self, memory_type: str) -> Optional[PromptTemplate]:
        """Get template by memory type"""
        # Check built-in templates
        if memory_type in self.templates:
            return self.templates[memory_type]

        # Check custom templates
        if memory_type in self._custom_templates:
            return self._custom_templates[memory_type]

        return None

    def register_template(self, name: str, template: PromptTemplate):
        """Register a custom template"""
        self._custom_templates[name] = template

    async def save_prompt_configuration(
        self, user_id: str, config: PromptConfiguration
    ) -> str:
        """Save a custom prompt configuration"""
        config.id = str(uuid.uuid4())
        config.user_id = user_id

        await self.prompt_configs_collection.insert_one(config.dict())

        # Convert to template and register
        template = PromptTemplate(
            name=config.name,
            description=config.description,
            template=config.template,
            variables=[],  # TODO: Parse from config
            output_schema=config.output_schema,
            tags=[config.memory_type, "custom", "user"],
        )

        self.register_template(config.name, template)

        return config.id

    async def load_user_templates(self, user_id: str):
        """Load all custom templates for a user"""
        cursor = self.prompt_configs_collection.find(
            {"user_id": user_id, "is_active": True}
        )

        async for doc in cursor:
            config = PromptConfiguration(**doc)

            template = PromptTemplate(
                name=config.name,
                description=config.description,
                template=config.template,
                variables=[],  # TODO: Parse from config
                output_schema=config.output_schema,
                tags=[config.memory_type, "custom", "user"],
            )

            self.register_template(config.name, template)

    async def _get_or_create_extractor_preset(self, user_id: str) -> str:
        """Get or create a default memory extractor preset"""
        # Check if user has a memory extractor preset
        preset = await self.llm_service.presets_collection.find_one(
            {"user_id": user_id, "name": "Memory Extractor"}
        )

        if preset:
            return preset["id"]

        # Create default extractor preset
        from app.llm.configs import LLMPresetConfig

        extractor_config = LLMPresetConfig(
            name="Memory Extractor",
            description="Optimized for extracting structured memories",
            provider="openai",
            model="gpt-3.5-turbo-0125",  # Model that supports JSON mode
            temperature=0.1,
            max_tokens=2000,
            system_prompt="You are a memory extraction specialist. Extract structured information exactly as requested.",
            extract_memories=False,  # Don't extract from extractions
            use_memory_context=False,  # Don't use context for extraction
        )

        preset = await self.llm_service.create_preset(user_id, extractor_config)
        return preset.id
