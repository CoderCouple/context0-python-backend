"""Centralized embeddings utilities for memory handlers"""
from typing import List, Optional

# Try to import OpenAI embeddings, fallback to mock if not available
try:
    from langchain_openai import OpenAIEmbeddings as RealOpenAIEmbeddings

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    RealOpenAIEmbeddings = None


class MockEmbeddings:
    """Mock embeddings for testing and development"""

    def __init__(self, model: str = "text-embedding-3-small", **kwargs):
        self.model = model
        self.dimension = 1536  # OpenAI embedding dimension

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed multiple documents"""
        return [[0.1] * self.dimension for _ in texts]

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed single query"""
        return [0.1] * self.dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Sync embed multiple documents"""
        return [[0.1] * self.dimension for _ in texts]

    def embed_query(self, text: str) -> List[float]:
        """Sync embed single query"""
        return [0.1] * self.dimension


def get_embeddings(
    model: str = "text-embedding-3-small", api_key: Optional[str] = None
) -> object:
    """Get embeddings instance - real or mock based on availability"""
    if OPENAI_AVAILABLE and api_key:
        try:
            return RealOpenAIEmbeddings(model=model, api_key=api_key)
        except Exception:
            pass

    # Fallback to mock embeddings
    return MockEmbeddings(model=model)


# Export the appropriate class
if OPENAI_AVAILABLE:
    OpenAIEmbeddings = RealOpenAIEmbeddings
else:
    OpenAIEmbeddings = MockEmbeddings
