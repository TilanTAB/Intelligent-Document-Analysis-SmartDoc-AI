import unittest

try:
    from langchain_core.documents import Document
    from intelligence.knowledge_synthesizer import ResearchAgent
    LANGCHAIN_AVAILABLE = True
except ImportError:
    Document = None  # type: ignore
    ResearchAgent = None  # type: ignore
    LANGCHAIN_AVAILABLE = False


class FakeLLM:
    """Simple stand-in for ChatGoogleGenerativeAI to avoid network calls."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.last_prompt = None

    def invoke(self, prompt: str):
        self.last_prompt = prompt
        return type("Resp", (), {"content": self.content})


@unittest.skipUnless(LANGCHAIN_AVAILABLE, "langchain not installed in this environment")
class ResearchAgentTests(unittest.TestCase):
    def test_generate_returns_stubbed_content_with_citations(self):
        docs = [
            Document(page_content="Alpha text", metadata={"id": "a1"}),
            Document(page_content="Beta text", metadata={"source": "s1"}),
        ]
        llm = FakeLLM("Answer about alpha")
        agent = ResearchAgent(llm=llm, top_k=1, max_context_chars=200)

        result = agent.generate("What is alpha?", docs)

        self.assertEqual(result["draft_answer"], "Answer about alpha")
        self.assertIn("Alpha text", llm.last_prompt)

    def test_generate_handles_no_documents(self):
        llm = FakeLLM("unused")
        agent = ResearchAgent(llm=llm)

        result = agent.generate("Any question", [])

        self.assertIn("could not find supporting documents", result["draft_answer"])


if __name__ == "__main__":
    unittest.main()
