"""Builder for custom category prompt templates"""
from typing import Dict, Any
from app.llm.prompts.base import PromptTemplate
from app.llm.types import CategoryConfig, PromptVariable


class CategoryPromptBuilder:
    """Build custom prompts for user-defined categories"""

    def build_extraction_prompt(self, category: CategoryConfig) -> PromptTemplate:
        """
        Build a custom extraction prompt for a category

        Args:
            category: Category configuration

        Returns:
            PromptTemplate for the category
        """
        template_str = f"""Extract information for category: {category.name}
Description: {category.description}

INCLUDE items matching: {category.includes_prompt}
EXCLUDE items matching: {category.excludes_prompt}

Text: {{{{ content }}}}

Analyze the text and extract items that match the inclusion criteria while avoiding the exclusion criteria.
Maximum items to extract: {category.max_items}

Return ONLY valid JSON:
{{
  "category": "{category.name}",
  "extracted_items": [
    {{
      "text": "extracted text",
      "confidence": 0.0-1.0,
      "reason": "why this matches the category",
      "metadata": {{}}
    }}
  ],
  "summary": "brief summary of extracted items",
  "stats": {{
    "total_found": number,
    "included": number,
    "excluded": number
  }}
}}"""

        return PromptTemplate(
            name=f"category_{category.name}",
            description=f"Extract {category.name} category items",
            template=template_str,
            variables=[
                PromptVariable(
                    name="content", description="Content to analyze", required=True
                )
            ],
            output_schema=self._build_category_schema(category),
            tags=["category", "custom", category.name.lower()],
        )

    def build_classification_prompt(
        self, categories: list[CategoryConfig]
    ) -> PromptTemplate:
        """
        Build a prompt to classify content into multiple categories

        Args:
            categories: List of category configurations

        Returns:
            PromptTemplate for multi-category classification
        """
        category_descriptions = "\n".join(
            [
                f"- {cat.name}: {cat.description} (includes: {cat.includes_prompt}, excludes: {cat.excludes_prompt})"
                for cat in categories
            ]
        )

        template_str = f"""Classify the following content into the appropriate categories.

Available Categories:
{category_descriptions}

Content: {{{{ content }}}}

Analyze the content and determine which categories apply. A piece of content can belong to multiple categories.

Return ONLY valid JSON:
{{
  "classifications": [
    {{
      "category": "category name",
      "confidence": 0.0-1.0,
      "matches": ["specific matches from content"],
      "reasoning": "why this category applies"
    }}
  ],
  "primary_category": "most relevant category name",
  "unmatched_content": ["content that doesn't fit any category"]
}}"""

        return PromptTemplate(
            name="multi_category_classification",
            description="Classify content into multiple custom categories",
            template=template_str,
            variables=[
                PromptVariable(
                    name="content", description="Content to classify", required=True
                )
            ],
            output_schema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": [
                    "classifications",
                    "primary_category",
                    "unmatched_content",
                ],
                "properties": {
                    "classifications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": [
                                "category",
                                "confidence",
                                "matches",
                                "reasoning",
                            ],
                            "properties": {
                                "category": {"type": "string"},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "matches": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "reasoning": {"type": "string"},
                            },
                        },
                    },
                    "primary_category": {"type": "string"},
                    "unmatched_content": {"type": "array", "items": {"type": "string"}},
                },
            },
            tags=["classification", "category", "multi"],
        )

    def build_filtering_prompt(self, category: CategoryConfig) -> PromptTemplate:
        """
        Build a prompt to filter existing memories by category

        Args:
            category: Category configuration

        Returns:
            PromptTemplate for filtering
        """
        template_str = f"""Filter the following memories based on category: {category.name}

Category Rules:
- Include if matching: {category.includes_prompt}
- Exclude if matching: {category.excludes_prompt}

Memories to filter:
{{{{ memories }}}}

Return ONLY valid JSON:
{{
  "category": "{category.name}",
  "included_memories": [
    {{
      "memory_id": "id",
      "match_reason": "why included",
      "confidence": 0.0-1.0
    }}
  ],
  "excluded_memories": [
    {{
      "memory_id": "id",
      "exclusion_reason": "why excluded"
    }}
  ],
  "statistics": {{
    "total_processed": number,
    "included_count": number,
    "excluded_count": number
  }}
}}"""

        return PromptTemplate(
            name=f"category_{category.name}_filter",
            description=f"Filter memories by {category.name} category",
            template=template_str,
            variables=[
                PromptVariable(
                    name="memories",
                    description="JSON array of memories to filter",
                    required=True,
                )
            ],
            output_schema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": [
                    "category",
                    "included_memories",
                    "excluded_memories",
                    "statistics",
                ],
                "properties": {
                    "category": {"type": "string"},
                    "included_memories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["memory_id", "match_reason", "confidence"],
                            "properties": {
                                "memory_id": {"type": "string"},
                                "match_reason": {"type": "string"},
                                "confidence": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                            },
                        },
                    },
                    "excluded_memories": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["memory_id", "exclusion_reason"],
                            "properties": {
                                "memory_id": {"type": "string"},
                                "exclusion_reason": {"type": "string"},
                            },
                        },
                    },
                    "statistics": {
                        "type": "object",
                        "required": [
                            "total_processed",
                            "included_count",
                            "excluded_count",
                        ],
                        "properties": {
                            "total_processed": {"type": "integer"},
                            "included_count": {"type": "integer"},
                            "excluded_count": {"type": "integer"},
                        },
                    },
                },
            },
            tags=["filter", "category", category.name.lower()],
        )

    def _build_category_schema(self, category: CategoryConfig) -> Dict[str, Any]:
        """Build JSON schema for category extraction"""
        return {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["category", "extracted_items", "summary", "stats"],
            "properties": {
                "category": {"type": "string", "const": category.name},
                "extracted_items": {
                    "type": "array",
                    "maxItems": category.max_items,
                    "items": {
                        "type": "object",
                        "required": ["text", "confidence", "reason"],
                        "properties": {
                            "text": {"type": "string"},
                            "confidence": {
                                "type": "number",
                                "minimum": 0,
                                "maximum": 1,
                            },
                            "reason": {"type": "string"},
                            "metadata": {"type": "object"},
                        },
                    },
                },
                "summary": {"type": "string"},
                "stats": {
                    "type": "object",
                    "required": ["total_found", "included", "excluded"],
                    "properties": {
                        "total_found": {"type": "integer", "minimum": 0},
                        "included": {"type": "integer", "minimum": 0},
                        "excluded": {"type": "integer", "minimum": 0},
                    },
                },
            },
        }
