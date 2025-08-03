"""Service for auto-categorizing and detecting emotions in memories"""

import logging
from typing import List, Tuple, Optional
from app.common.enum.memory_category import MemoryCategory, suggest_categories
from app.common.enum.memory_emotion import (
    MemoryEmotion,
    EmotionIntensity,
    detect_emotion,
    get_emotion_valence,
)
from app.api.v1.request.memory_request import MemoryRecordInput

logger = logging.getLogger(__name__)


class MemoryCategoricationService:
    """Service to handle memory categorization and emotion detection"""

    @staticmethod
    def enrich_memory_input(memory_input: MemoryRecordInput) -> MemoryRecordInput:
        """Enrich memory input with auto-detected category and emotion (primary + additional as tags)"""

        # Auto-detect category if not provided
        if not memory_input.category:
            suggested_categories = suggest_categories(memory_input.text)
            if suggested_categories:
                # Set the first as primary category
                memory_input.category = suggested_categories[0]
                logger.info(
                    f"Auto-detected primary category: {memory_input.category.value}"
                )

                # Add other suggested categories as tags
                if len(suggested_categories) > 1:
                    additional_category_tags = [
                        f"category:{cat.value}" for cat in suggested_categories[1:3]
                    ]  # Limit to 2 additional
                    memory_input.tags.extend(additional_category_tags)
                    logger.info(
                        f"Additional categories as tags: {additional_category_tags}"
                    )

        # Auto-detect emotion if not provided
        if not memory_input.emotion:
            emotion, intensity = detect_emotion(memory_input.text)
            memory_input.emotion = emotion
            memory_input.emotion_intensity = intensity
            logger.info(
                f"Auto-detected emotion: {emotion.value} (intensity: {intensity.value})"
            )

        # Add primary category as tag (for easier filtering)
        if memory_input.category:
            memory_input.tags.append(f"primary_category:{memory_input.category.value}")

        # Add primary emotion as tag
        if memory_input.emotion:
            memory_input.tags.append(f"primary_emotion:{memory_input.emotion.value}")

            # Add emotion group as tag
            emotion_group = MemoryCategoricationService.get_emotion_group(
                memory_input.emotion
            )
            if emotion_group:
                memory_input.tags.append(f"emotion_group:{emotion_group.lower()}")

        # Remove duplicates from tags
        memory_input.tags = list(set(memory_input.tags))

        # Store categorization metadata
        memory_input.metadata.update(
            {
                "auto_categorized": True,
                "primary_category": memory_input.category.value
                if memory_input.category
                else None,
                "emotion": memory_input.emotion.value if memory_input.emotion else None,
                "emotion_intensity": memory_input.emotion_intensity.value
                if memory_input.emotion_intensity
                else None,
                "emotion_valence": get_emotion_valence(memory_input.emotion)
                if memory_input.emotion
                else 0.0,
            }
        )

        return memory_input

    @staticmethod
    def get_emotion_group(emotion: MemoryEmotion) -> Optional[str]:
        """Get the emotion group (Positive, Negative, Complex, Neutral)"""
        from app.common.enum.memory_emotion import EMOTION_GROUPS

        for group_name, emotions in EMOTION_GROUPS.items():
            if emotion in emotions:
                return group_name
        return None

    @staticmethod
    def get_emotion_valence(emotion: Optional[MemoryEmotion]) -> float:
        """Get emotional valence score for an emotion"""
        if not emotion:
            return 0.0

        return get_emotion_valence(emotion)

    @staticmethod
    def filter_by_emotion_group(
        emotions: List[MemoryEmotion], group: str
    ) -> List[MemoryEmotion]:
        """Filter emotions by group (Positive, Negative, Complex, Neutral)"""
        from app.common.enum.memory_emotion import EMOTION_GROUPS

        if group not in EMOTION_GROUPS:
            return emotions

        group_emotions = EMOTION_GROUPS[group]
        return [emotion for emotion in emotions if emotion in group_emotions]

    @staticmethod
    def get_category_hierarchy(category: MemoryCategory) -> Tuple[str, MemoryCategory]:
        """Get the parent group for a category"""
        from app.common.enum.memory_category import CATEGORY_GROUPS

        for group_name, categories in CATEGORY_GROUPS.items():
            if category in categories:
                return group_name, category

        return "General", category
