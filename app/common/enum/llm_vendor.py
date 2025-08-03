"""LLM Vendor Enum"""

from enum import Enum


class LLMVendor(str, Enum):
    """Supported LLM vendors"""

    OPENAI = "openai"  # ChatGPT
    ANTHROPIC = "anthropic"  # Claude
    GOOGLE = "google"  # Gemini
