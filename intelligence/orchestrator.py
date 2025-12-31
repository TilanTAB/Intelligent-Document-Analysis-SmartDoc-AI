"""
Agent orchestrator orchestration using LangGraph.

Defines the multi-agent orchestrator that:
1. Checks document relevance
2. Generates multiple answer candidates using research agent
3. Selects the best answer through verification
4. Provides feedback loop for iterative improvement
"""
from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
from typing import TypedDict, List, Dict, Any, Optional, Annotated
import operator
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import logging

from .knowledge_synthesizer import ResearchAgent
from .accuracy_verifier import VerificationAgent
from .context_validator import ContextValidator
from langchain_google_genai import ChatGoogleGenerativeAI
from configuration.parameters import parameters

logger = logging.getLogger(__name__)


class SubQResult(TypedDict):
    idx: int
    question: str
    answer: str
    report: str


class AgentState(TypedDict, total=False):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: BaseRetriever
    feedback: Optional[str]
    research_attempts: int
    query_used: str
    candidate_answers: List[str]
    selection_reasoning: str
    is_multi_query: bool
    sub_queries: List[str]
    sub_results: Annotated[List[SubQResult], operator.add]


class AgentWorkflow:
    MAX_RESEARCH_ATTEMPTS: int = parameters.MAX_RESEARCH_ATTEMPTS
    NUM_RESEARCH_CANDIDATES: int = parameters.NUM_RESEARCH_CANDIDATES

    def __init__(self, num_candidates: int = None) -> None:
        logger.info("Initializing AgentWorkflow...")
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.context_validator = ContextValidator()
        self.compiled_single = None
        self.compiled_main = None
        self.llm = ChatGoogleGenerativeAI(
            model=parameters.LLM_MODEL_NAME,
            google_api_key=parameters.GOOGLE_API_KEY,
            temperature=0.1,
            max_output_tokens=256
        )
        if num_candidates is not None:
            self.NUM_RESEARCH_CANDIDATES = num_candidates
        logger.info(f"AgentWorkflow initialized (candidates={self.NUM_RESEARCH_CANDIDATES})")

    def _retrieve_docs(self, state: AgentState) -> Dict[str, Any]:
        docs = state["retriever"].invoke(state["question"])
        return {
            "documents": docs,
            "draft_answer": "",
            "verification_report": "",
            "is_relevant": False,
            "feedback": None,
            "feedback_for_research": None,
            "contradictions_for_research": [],
            "unsupported_claims_for_research": [],
            "research_attempts": 0,
            "candidate_answers": [],
            "selection_reasoning": "",
            "query_used": state["question"],
        }

    def _build_single_question_graph(self):
        g = StateGraph(AgentState)
        g.add_node("retrieve_docs", self._retrieve_docs)
        g.add_node("check_relevance", self._check_relevance_step)
        g.add_node("research", self._research_step)
        g.add_node("verify", self._verification_step)
        g.add_edge(START, "retrieve_docs")
        g.add_edge("retrieve_docs", "check_relevance")
        g.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {"relevant": "research", "irrelevant": END},
        )
        g.add_edge("research", "verify")
        g.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {"re_research": "research", "end": END},
        )
        return g.compile()

    def _assign_workers(self, state: AgentState):
        sends = []
        for i, q in enumerate(state.get("sub_queries", [])):
            sends.append(Send("subq_worker", {"question": q, "subq_idx": i, "retriever": state["retriever"]}))
        return sends

    def _subq_worker(self, state: AgentState) -> Dict[str, Any]:
        subq_idx = state["subq_idx"]
        q = state["question"]
        result_state = self.compiled_single.invoke({
            "question": q,
            "retriever": state["retriever"],
            "research_attempts": 0,
        })
        return {
            "sub_results": [{
                "idx": subq_idx,
                "question": q,
                "answer": result_state.get("draft_answer", ""),
                "report": result_state.get("verification_report", ""),
            }]
        }

    def _combine_answers(self, state: AgentState) -> Dict[str, Any]:
        sub_results = sorted(state.get("sub_results", []), key=lambda r: r["idx"])
        combined = "\n\n".join(
            f"Q{i+1}: {r['question']}\nA: {r['answer']}"
            for i, r in enumerate(sub_results)
        )
        return {
            "draft_answer": combined,
            "verification_report": "Multi-question answer combined."
        }

    def build_orchestrator(self) -> Any:
        self.compiled_single = self._build_single_question_graph()
        g = StateGraph(AgentState)
        g.add_node("detect_query_type", self._detect_query_type)
        g.add_node("subq_worker", self._subq_worker)
        g.add_node("combine_answers", self._combine_answers)
        def run_single(state: AgentState) -> Dict[str, Any]:
            out = self.compiled_single.invoke({
                "question": state["question"],
                "retriever": state["retriever"],
                "research_attempts": 0,
            })
            return {
                "draft_answer": out.get("draft_answer", ""),
                "verification_report": out.get("verification_report", ""),
            }
        g.add_node("run_single", run_single)
        g.set_entry_point("detect_query_type")
        g.add_conditional_edges(
            "detect_query_type",
            lambda s: "multi" if s.get("is_multi_query") else "single",
            {"multi": "fanout", "single": "run_single"},
        )
        g.add_node("fanout", lambda s: {})
        g.add_conditional_edges("fanout", self._assign_workers, ["subq_worker"])
        g.add_edge("subq_worker", "combine_answers")
        g.add_edge("combine_answers", END)
        g.add_edge("run_single", END)
        return g.compile()

    def _detect_query_type(self, state: AgentState) -> Dict[str, Any]:
        prompt = f"""
You are an expert assistant for document Q&A. Analyze the following question and determine:
1. Is it a single question or does it contain multiple sub-questions?
2. If it contains multiple questions, decompose it into a list of clear, standalone sub-questions (no overlap, no ambiguity).

Return your answer as a JSON object with two fields:
- is_multi_query: true or false
- sub_queries: a list of strings (the sub-questions, or a single-item list if only one)

Question: {state['question']}
"""
        try:
            response = self.llm.invoke(prompt)
            import json
            content = response.content if hasattr(response, "content") else str(response)
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                result = json.loads(json_str)
                is_multi = bool(result.get("is_multi_query", False))
                sub_queries = result.get("sub_queries", [])
            else:
                is_multi = False
                sub_queries = [state["question"]]
        except Exception as e:
            logger.error(f"LLM decomposition failed: {e}")
            is_multi = False
            sub_queries = [state["question"]]
        if is_multi:
            logger.info(f"[LLM Decompose] Multi-question detected: {len(sub_queries)} sub-queries")
        else:
            logger.info("[LLM Decompose] Single question detected; no decomposition needed.")
        return {"is_multi_query": is_multi, "sub_queries": sub_queries}

    def _check_relevance_step(self, state: AgentState) -> Dict[str, Any]:
        logger.debug("Checking context relevance...")
        result = self.context_validator.context_validate_with_rewrite(
            question=state["question"],
            retriever=state["retriever"],
            k=parameters.RELEVANCE_CHECK_K,
            max_rewrites=parameters.MAX_QUERY_REWRITES,
        )
        classification = result.get("classification", "NO_MATCH")
        query_used = result.get("query_used", state["question"])
        logger.info(f"Relevance: {classification} (query_used={query_used[:80]})")
        if classification in ("CAN_ANSWER", "PARTIAL"):
            documents = state["retriever"].invoke(query_used)
            return {
                "is_relevant": True,
                "query_used": query_used,
                "documents": documents
            }
        return {
            "is_relevant": False,
            "query_used": query_used,
            "draft_answer": "This question isn't related to the uploaded documents. Please ask another question.",
        }

    def _decide_after_relevance_check(self, state: AgentState) -> str:
        return "relevant" if state["is_relevant"] else "irrelevant"

    def run_workflow(self, question: str, retriever: BaseRetriever) -> Dict[str, str]:
        if self.compiled_main is None:
            self.compiled_main = self.build_orchestrator()
        initial_state: AgentState = {
            "question": question,
            "retriever": retriever,
            "sub_results": [],
            "sub_queries": [],
            "is_multi_query": False,
        }
        final = self.compiled_main.invoke(initial_state)
        return {
            "draft_answer": final.get("draft_answer", ""),
            "verification_report": final.get("verification_report", ""),
        }

    def _verification_step(self, state: AgentState) -> Dict[str, Any]:
        logger.debug("Selecting best answer from candidates...")
        candidate_answers = state.get("candidate_answers", []) or [state.get("draft_answer", "")]
        selection_result = self.verifier.select_best_answer(
            candidate_answers=candidate_answers,
            documents=state["documents"],
            question=state["question"]
        )
        best_answer = selection_result["selected_answer"]
        selection_reasoning = selection_result.get("reasoning", "")
        logger.info(f"Selected candidate {selection_result['selected_index'] + 1} as best answer")
        verification_result = self.verifier.check(
            answer=best_answer,
            documents=state["documents"],
            question=state["question"]
        )
        verification_report = verification_result["verification_report"]
        verification_report = f"**Candidates Evaluated:** {len(candidate_answers)}\n" + \
                             f"**Selected Candidate:** {selection_result['selected_index'] + 1}\n" + \
                             f"**Selection Confidence:** {selection_result.get('confidence', 'N/A')}\n" + \
                             f"**Selection Reasoning:** {selection_reasoning}\n\n" + \
                             verification_report
        feedback_for_research = verification_result.get("feedback")
        return {
            "draft_answer": best_answer,
            "verification_report": verification_report,
            "feedback_for_research": feedback_for_research,
            "selection_reasoning": selection_reasoning,
            "should_retry": verification_result.get("should_retry", False),
        }

    def _decide_next_step(self, state: AgentState) -> str:
        research_attempts = state.get("research_attempts", 1)
        should_retry = bool(state.get("should_retry", False))
        if should_retry and research_attempts < self.MAX_RESEARCH_ATTEMPTS:
            return "re_research"
        return "end"

    def _research_step(self, state: AgentState) -> Dict[str, Any]:
        attempts = state.get("research_attempts", 0) + 1
        feedback_for_research = state.get("feedback_for_research")
        previous_answer = state.get("draft_answer") if feedback_for_research else None
        logger.info(f"Research step (attempt {attempts}/{self.MAX_RESEARCH_ATTEMPTS})")
        logger.info(f"Generating {self.NUM_RESEARCH_CANDIDATES} candidate answers...")
        candidate_answers = []
        for i in range(self.NUM_RESEARCH_CANDIDATES):
            logger.info(f"Generating candidate {i + 1}/{self.NUM_RESEARCH_CANDIDATES}")
            result = self.researcher.generate(
                question=state["question"],
                documents=state["documents"],
                feedback=feedback_for_research,
                previous_answer=previous_answer
            )
            candidate_answers.append(result["draft_answer"])
        logger.info(f"Generated {len(candidate_answers)} candidate answers")
        return {
            "candidate_answers": candidate_answers,
            "research_attempts": attempts,
            "feedback": None
        }
