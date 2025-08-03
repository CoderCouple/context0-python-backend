"""
Smart hybrid extraction filter using patterns + LLM
"""
import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from datetime import datetime
import hashlib
import json

logger = logging.getLogger(__name__)


class SmartExtractionFilter:
    """Hybrid extraction filter using quick patterns and LLM for complex cases"""

    def __init__(self, llm_service=None):
        self.llm_service = llm_service
        self._decision_cache = {}  # Cache LLM decisions

        # Quick skip patterns - definitely not personal info
        self.SKIP_PATTERNS = {
            # Greetings and acknowledgments
            "greetings": [
                "hey",
                "hi",
                "hello",
                "good morning",
                "good evening",
                "howdy",
            ],
            "acknowledgments": [
                "ok",
                "okay",
                "sure",
                "yes",
                "no",
                "yeah",
                "nope",
                "alright",
                "got it",
                "understood",
            ],
            "thanks": ["thanks", "thank you", "thx", "ty"],
            "goodbyes": ["bye", "goodbye", "see you", "later", "ttyl"],
            # Questions (starting with question words)
            "questions": [
                r"^(what|where|when|who|why|how|which|can|could|would|should|will|do|does|is|are)\s+",
                r"\?$",
            ],
            # Commands
            "commands": [
                r"^(tell|show|explain|help|find|search|give|make|create|write|generate)\s+",
                r"^(please|could you|can you|would you)\s+",
            ],
            # Meta conversation
            "meta": [
                "continue",
                "go on",
                "stop",
                "wait",
                "pause",
                "nevermind",
                "forget it",
            ],
        }

        # Quick extract patterns - definitely personal info
        self.EXTRACT_PATTERNS = {
            "strong_personal": [
                r"\b(i|I)\s+(love|hate|prefer|always|never|enjoy|like|dislike)\s+",
                r"\b(my|My)\s+(favorite|favourite)\s+",
                r"\b(i|I)\s+(am|'m)\s+(?:a|an)?\s*\w+",  # "I am 28", "I'm a developer"
                r"\b(i|I)\s+(live|work|study)\s+",
                r"\b(i|I)\s+have\s+",  # "I have cats", "I have a car"
                r"\b(i|I)'m\s+\d+",  # "I'm 28 years old"
                r"\b(my|My)\s+name\s+is\s+",  # "My name is..."
                r"\b(i|I)('m|am)?\s+(want to|plan to|need to|going to|planning to|trying to)\s+",  # Goals
                r"\b(i|I)\s+(started|joined|began)\s+",  # Temporal events
            ]
        }

        # Minimum requirements
        self.MIN_WORDS = 3  # "I love pizza" has 3 words
        self.MIN_CHARS = 10  # Reduced from 15

    def should_extract(
        self, content: str, user_id: str = None
    ) -> Tuple[bool, str, float]:
        """
        Determine if content should be extracted as memory

        Returns:
            - should_extract: Boolean
            - reason: Explanation
            - confidence: 0.0 to 1.0
        """

        # Clean content
        content = content.strip()
        content_lower = content.lower()
        word_count = len(content.split())

        # Step 1: Quick rejection checks
        if len(content) < self.MIN_CHARS or word_count < self.MIN_WORDS:
            return False, "Content too short", 1.0

        # Check skip patterns
        skip_reason = self._check_skip_patterns(content_lower)
        if skip_reason:
            return False, skip_reason, 0.95

        # Step 2: Quick acceptance checks
        extract_match = self._check_extract_patterns(content)
        if extract_match:
            return True, extract_match, 0.95

        # Step 3: For ambiguous cases, use LLM
        if word_count >= 8:  # Substantial content that didn't match patterns
            return self._llm_check(content, user_id)

        return False, "No personal information detected", 0.8

    def _check_skip_patterns(self, content_lower: str) -> Optional[str]:
        """Check if content matches skip patterns"""

        # Check simple skip words
        for category, patterns in self.SKIP_PATTERNS.items():
            if category in [
                "greetings",
                "acknowledgments",
                "thanks",
                "goodbyes",
                "meta",
            ]:
                if content_lower in patterns:
                    return f"Skip: {category}"

        # Check regex patterns
        for pattern in self.SKIP_PATTERNS.get("questions", []):
            if isinstance(pattern, str) and re.match(pattern, content_lower):
                return "Skip: question"

        for pattern in self.SKIP_PATTERNS.get("commands", []):
            if isinstance(pattern, str) and re.match(pattern, content_lower):
                return "Skip: command"

        return None

    def _check_extract_patterns(self, content: str) -> Optional[str]:
        """Check if content matches extract patterns"""

        for category, patterns in self.EXTRACT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return f"Extract: {category} pattern matched"

        return None

    def _llm_check(self, content: str, user_id: str = None) -> Tuple[bool, str, float]:
        """Use LLM to check for personal information"""

        # Check cache first
        cache_key = hashlib.md5(content.encode()).hexdigest()
        if cache_key in self._decision_cache:
            cached = self._decision_cache[cache_key]
            return cached["should_extract"], cached["reason"], cached["confidence"]

        # If no LLM service, be conservative
        if not self.llm_service:
            return False, "No LLM available for complex check", 0.5

        try:
            # Use asyncio to run async function
            import asyncio

            loop = asyncio.get_event_loop()

            if loop.is_running():
                # We're already in an async context
                future = asyncio.ensure_future(self._async_llm_check(content, user_id))
                # For now, return conservative answer
                # In production, this should be properly awaited
                return False, "LLM check pending", 0.5
            else:
                # Run synchronously
                result = loop.run_until_complete(
                    self._async_llm_check(content, user_id)
                )
                return result

        except Exception as e:
            logger.error(f"LLM check failed: {e}")
            return False, f"LLM check error: {str(e)}", 0.5

    async def _async_llm_check(
        self, content: str, user_id: str = None
    ) -> Tuple[bool, str, float]:
        """Async LLM check for personal information"""

        system_prompt = """You are a privacy-aware assistant that identifies personal information.
        
Analyze if the text contains PERSONAL information about the user that should be remembered.

Personal information includes:
- Preferences (likes, dislikes, favorites)
- Personal facts (location, job, relationships, possessions)
- Goals and intentions
- Biographical information
- Personal experiences or events

NOT personal information:
- General knowledge or facts
- Questions or requests
- Casual responses (ok, yes, no)
- Commands or instructions

Respond with JSON:
{
    "has_personal_info": true/false,
    "confidence": 0.0-1.0,
    "category": "preference|fact|goal|biographical|experience|none",
    "reason": "brief explanation"
}"""

        user_prompt = f'Analyze this text: "{content}"'

        try:
            # Call LLM
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            # Use a fast model for this check
            # Get or create a preset for extraction
            preset_id = await self.llm_service.get_or_create_user_default_preset(
                user_id
            )

            response = await self.llm_service.generate_with_preset(
                preset_id=preset_id,
                messages=messages,
                user_id=user_id,
                temperature=0.1,
                max_tokens=150,
            )

            # Parse response
            result = json.loads(response.content)

            should_extract = result.get("has_personal_info", False)
            confidence = result.get("confidence", 0.5)
            reason = result.get("reason", "LLM analysis")

            # Cache the decision
            self._decision_cache[hashlib.md5(content.encode()).hexdigest()] = {
                "should_extract": should_extract,
                "reason": f"LLM: {reason}",
                "confidence": confidence,
                "category": result.get("category", "none"),
                "timestamp": datetime.utcnow(),
            }

            return should_extract, f"LLM: {reason}", confidence

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return False, f"LLM error: {str(e)}", 0.5

    def get_cached_decisions(self) -> List[Dict]:
        """Get all cached LLM decisions for analysis"""
        return [{"content_hash": k, **v} for k, v in self._decision_cache.items()]

    def learn_from_feedback(self, content: str, should_extract: bool, reason: str):
        """Learn from user feedback to improve decisions"""

        cache_key = hashlib.md5(content.encode()).hexdigest()
        self._decision_cache[cache_key] = {
            "should_extract": should_extract,
            "reason": f"User feedback: {reason}",
            "confidence": 1.0,
            "timestamp": datetime.utcnow(),
            "source": "user_feedback",
        }

        # TODO: In production, this could update pattern rules based on feedback
