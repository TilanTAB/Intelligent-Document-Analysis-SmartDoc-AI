"""
Test fixtures and shared utilities for DocChat tests.
"""
import pytest
from unittest.mock import MagicMock
from langchain_core.documents import Document


class FakeLLM:
    """Mock LLM for testing without API calls."""
    
    def __init__(self, content: str = "Test response"):
        self.content = content
        self.last_prompt = None
        self.invoke_count = 0

    def invoke(self, prompt: str):
        self.last_prompt = prompt
        self.invoke_count += 1
        return type("Response", (), {"content": self.content})()


class FakeRetriever:
    """Mock retriever for testing without vector store."""
    
    def __init__(self, documents: list = None):
        self.documents = documents or []
        self.invoke_count = 0
        self.last_query = None

    def invoke(self, query: str):
        self.last_query = query
        self.invoke_count += 1
        return self.documents


@pytest.fixture
def sample_documents():
    """Create sample documents for testing."""
    return [
        Document(
            page_content="The data center in Singapore achieved a PUE of 1.12 in 2022.",
            metadata={"source": "test.pdf", "page": 1}
        ),
        Document(
            page_content="Carbon-free energy in Asia Pacific reached 45% in 2023.",
            metadata={"source": "test.pdf", "page": 2}
        ),
        Document(
            page_content="DeepSeek-R1 outperformed o1-mini on coding benchmarks.",
            metadata={"source": "deepseek.pdf", "page": 1}
        ),
    ]


@pytest.fixture
def fake_llm():
    """Create a fake LLM for testing."""
    return FakeLLM("This is a test response.")


@pytest.fixture
def fake_retriever(sample_documents):
    """Create a fake retriever with sample documents."""
    return FakeRetriever(sample_documents)


@pytest.fixture
def empty_retriever():
    """Create a fake retriever that returns no documents."""
    return FakeRetriever([])
