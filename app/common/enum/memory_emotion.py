"""Memory Emotion Tags for emotional context of memories"""

from enum import Enum
from typing import List, Dict


class MemoryEmotion(str, Enum):
    """Emotional states associated with memories"""

    # Primary emotions
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEARFUL = "fearful"
    SURPRISED = "surprised"
    DISGUSTED = "disgusted"

    # Extended emotions
    EXCITED = "excited"
    ANXIOUS = "anxious"
    GRATEFUL = "grateful"
    PROUD = "proud"
    EMBARRASSED = "embarrassed"
    CONFUSED = "confused"
    HOPEFUL = "hopeful"
    DISAPPOINTED = "disappointed"
    CONTENT = "content"
    FRUSTRATED = "frustrated"
    NOSTALGIC = "nostalgic"
    INSPIRED = "inspired"
    LONELY = "lonely"
    LOVED = "loved"

    # General states
    NEUTRAL = "neutral"
    MIXED = "mixed"
    OTHER = "other"


# Emotion intensity levels
class EmotionIntensity(str, Enum):
    """Intensity levels for emotions"""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


# Emotion groupings
EMOTION_GROUPS: Dict[str, List[MemoryEmotion]] = {
    "Positive": [
        MemoryEmotion.HAPPY,
        MemoryEmotion.EXCITED,
        MemoryEmotion.GRATEFUL,
        MemoryEmotion.PROUD,
        MemoryEmotion.CONTENT,
        MemoryEmotion.HOPEFUL,
        MemoryEmotion.INSPIRED,
        MemoryEmotion.LOVED,
    ],
    "Negative": [
        MemoryEmotion.SAD,
        MemoryEmotion.ANGRY,
        MemoryEmotion.FEARFUL,
        MemoryEmotion.DISGUSTED,
        MemoryEmotion.ANXIOUS,
        MemoryEmotion.EMBARRASSED,
        MemoryEmotion.DISAPPOINTED,
        MemoryEmotion.FRUSTRATED,
        MemoryEmotion.LONELY,
    ],
    "Complex": [
        MemoryEmotion.SURPRISED,
        MemoryEmotion.CONFUSED,
        MemoryEmotion.NOSTALGIC,
        MemoryEmotion.MIXED,
    ],
    "Neutral": [MemoryEmotion.NEUTRAL, MemoryEmotion.OTHER],
}


# Keywords for emotion detection
EMOTION_KEYWORDS: Dict[MemoryEmotion, List[str]] = {
    MemoryEmotion.HAPPY: [
        "happy",
        "joy",
        "delighted",
        "pleased",
        "cheerful",
        "glad",
        "thrilled",
        "ecstatic",
    ],
    MemoryEmotion.SAD: [
        "sad",
        "unhappy",
        "depressed",
        "down",
        "miserable",
        "sorrowful",
        "crying",
        "tears",
    ],
    MemoryEmotion.ANGRY: [
        "angry",
        "mad",
        "furious",
        "irritated",
        "annoyed",
        "rage",
        "upset",
        "pissed",
    ],
    MemoryEmotion.FEARFUL: [
        "scared",
        "afraid",
        "frightened",
        "terrified",
        "anxious",
        "worried",
        "nervous",
    ],
    MemoryEmotion.EXCITED: [
        "excited",
        "thrilled",
        "eager",
        "enthusiastic",
        "pumped",
        "can't wait",
    ],
    MemoryEmotion.GRATEFUL: [
        "grateful",
        "thankful",
        "appreciative",
        "blessed",
        "fortunate",
    ],
    MemoryEmotion.PROUD: ["proud", "accomplished", "achieved", "success", "proud of"],
    MemoryEmotion.ANXIOUS: [
        "anxious",
        "worried",
        "stressed",
        "nervous",
        "tense",
        "uneasy",
    ],
    MemoryEmotion.FRUSTRATED: [
        "frustrated",
        "annoyed",
        "irritated",
        "stuck",
        "blocked",
    ],
    MemoryEmotion.CONTENT: [
        "content",
        "satisfied",
        "peaceful",
        "calm",
        "relaxed",
        "comfortable",
    ],
    MemoryEmotion.NOSTALGIC: [
        "nostalgic",
        "remember",
        "memories",
        "reminisce",
        "back then",
        "miss",
    ],
    MemoryEmotion.INSPIRED: [
        "inspired",
        "motivated",
        "energized",
        "creative",
        "innovative",
    ],
    MemoryEmotion.LOVED: [
        "loved",
        "cherished",
        "adored",
        "cared",
        "affection",
        "romance",
    ],
    MemoryEmotion.LONELY: ["lonely", "alone", "isolated", "miss", "solitary"],
}


def detect_emotion(text: str) -> tuple[MemoryEmotion, EmotionIntensity]:
    """Detect emotion from text content"""
    text_lower = text.lower()
    emotion_scores = {}

    # Check for emotion keywords
    for emotion, keywords in EMOTION_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text_lower)
        if score > 0:
            emotion_scores[emotion] = score

    if not emotion_scores:
        return MemoryEmotion.NEUTRAL, EmotionIntensity.LOW

    # Get emotion with highest score
    detected_emotion = max(emotion_scores, key=emotion_scores.get)
    score = emotion_scores[detected_emotion]

    # Determine intensity
    if score >= 4:
        intensity = EmotionIntensity.EXTREME
    elif score >= 3:
        intensity = EmotionIntensity.HIGH
    elif score >= 2:
        intensity = EmotionIntensity.MEDIUM
    else:
        intensity = EmotionIntensity.LOW

    # Check if mixed emotions
    if len(emotion_scores) >= 3:
        return MemoryEmotion.MIXED, intensity

    return detected_emotion, intensity


def get_emotion_valence(emotion: MemoryEmotion) -> float:
    """Get emotional valence score (-1 to 1, where -1 is most negative, 1 is most positive)"""
    valence_map = {
        # Positive emotions
        MemoryEmotion.HAPPY: 0.8,
        MemoryEmotion.EXCITED: 0.9,
        MemoryEmotion.GRATEFUL: 0.7,
        MemoryEmotion.PROUD: 0.8,
        MemoryEmotion.CONTENT: 0.6,
        MemoryEmotion.HOPEFUL: 0.7,
        MemoryEmotion.INSPIRED: 0.8,
        MemoryEmotion.LOVED: 0.9,
        # Negative emotions
        MemoryEmotion.SAD: -0.8,
        MemoryEmotion.ANGRY: -0.7,
        MemoryEmotion.FEARFUL: -0.8,
        MemoryEmotion.DISGUSTED: -0.6,
        MemoryEmotion.ANXIOUS: -0.6,
        MemoryEmotion.EMBARRASSED: -0.5,
        MemoryEmotion.DISAPPOINTED: -0.6,
        MemoryEmotion.FRUSTRATED: -0.5,
        MemoryEmotion.LONELY: -0.7,
        # Complex/Neutral emotions
        MemoryEmotion.SURPRISED: 0.1,
        MemoryEmotion.CONFUSED: -0.2,
        MemoryEmotion.NOSTALGIC: 0.2,
        MemoryEmotion.NEUTRAL: 0.0,
        MemoryEmotion.MIXED: 0.0,
        MemoryEmotion.OTHER: 0.0,
    }

    return valence_map.get(emotion, 0.0)
