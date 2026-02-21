from langchain_core.documents import Document

from configuration.parameters import parameters
from intelligence.knowledge_synthesizer import ResearchAgent


class _StubResponse:
    def __init__(self, content: str = "ok"):
        self.content = content


class _StubLLM:
    def invoke(self, _prompt: str):
        return _StubResponse("stub answer")


def test_research_agent_top_k_not_lower_than_relevance_k(monkeypatch):
    monkeypatch.setattr(parameters, "RESEARCH_TOP_K", 15, raising=False)
    monkeypatch.setattr(parameters, "RELEVANCE_CHECK_K", 20, raising=False)

    agent = ResearchAgent(llm=_StubLLM())

    assert agent.top_k == 20


def test_research_agent_prioritizes_high_overlap_evidence():
    agent = ResearchAgent(llm=_StubLLM(), top_k=2, max_context_chars=20_000)
    question = "What is the accuracy of AI models in coding?"

    docs = [
        Document(
            page_content="General discussion about AI models and deployment.",
            metadata={"source": "doc.pdf::hash", "page": 1, "chunk_id": "a"},
        ),
        Document(
            page_content="Benchmark caveats and policy details with no coding metric.",
            metadata={"source": "doc.pdf::hash", "page": 2, "chunk_id": "b"},
        ),
        Document(
            page_content="Figure 1.11: Coding (HumanEval) reaches 75% accuracy.",
            metadata={"source": "doc.pdf::hash", "page": 37, "chunk_id": "c"},
        ),
    ]

    context = agent._build_context(question=question, documents=docs)

    # The high-overlap evidence should be pulled into the selected context.
    assert "HumanEval" in context
    assert "75% accuracy" in context
