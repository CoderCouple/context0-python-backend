from abc import ABC
from typing import Dict, Optional, Union

import httpx


class LlmConfig(ABC):
    """
    Base configuration for Large Language Model (LLM) clients.

    Parameters:
        model (str | dict | None): The model name or a dictionary with model configuration details.
        temperature (float): Controls randomness in output; values closer to 1.0 increase creativity.
        api_key (str | None): API key used for authenticating with the LLM provider.
        max_tokens (int): Maximum number of tokens allowed in the generated response.
        top_p (float): Controls nucleus sampling; higher values allow more diverse token selection.
        top_k (int): Limits token selection to the top-k most probable tokens; higher values increase randomness.
        enable_vision (bool): If True, enables vision (multimodal) capabilities for models that support it.
        vision_details (str | None): Level of visual detail to request; options include "low", "high", or "auto".
        ollama_base_url (str | None): Base URL for Ollama's API (used if Ollama is the selected provider).
        aws_access_key_id (str | None): AWS access key ID for using AWS Bedrock models.
        aws_secret_access_key (str | None): AWS secret access key for AWS Bedrock authentication.
        aws_region (str | None): AWS region to use for Bedrock; defaults to "us-west-2".
        http_client_proxies (dict | str | None): Proxy configuration for HTTP requests; can be a string or a dictionary.
    """

    def __init__(
        self,
        model: Optional[Union[str, Dict]] = None,
        temperature: float = 0.1,
        api_key: Optional[str] = None,
        max_tokens: int = 5000,
        top_p: float = 0.1,
        top_k: int = 1,
        enable_vision: bool = False,
        vision_details: Optional[str] = "auto",
        # OpenAI specific
        http_client_proxies: Optional[Union[Dict, str]] = None,
        # Ollama specific
        ollama_base_url: Optional[str] = None,
        # AWS Bedrock specific
        aws_access_key_id: Optional[str] = None,
        aws_secret_access_key: Optional[str] = None,
        aws_region: Optional[str] = "us-west-2",
    ):
        self.model = model
        self.temperature = temperature
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.top_k = top_k
        self.enable_vision = enable_vision
        self.vision_details = vision_details

        # OpenAI specific
        self.http_client = (
            httpx.Client(proxies=http_client_proxies) if http_client_proxies else None
        )

        # Ollama specific
        self.ollama_base_url = ollama_base_url

        # AWS Bedrock specific
        self.aws_access_key_id = aws_access_key_id
        self.aws_secret_access_key = aws_secret_access_key
        self.aws_region = aws_region
