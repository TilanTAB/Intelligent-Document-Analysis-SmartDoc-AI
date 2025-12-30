"""
Agent orchestrator orchestration using LangGraph.

Defines the multi-agent orchestrator that:
1. Checks document relevance
2. Generates multiple answer candidates using research agent
3. Selects the best answer through verification
4. Provides feedback loop for iterative improvement
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import logging

from .knowledge_synthesizer import ResearchAgent
from .accuracy_verifier import VerificationAgent
from .context_validator import ContextValidator
from langchain_google_genai import ChatGoogleGenerativeAI
from configuration.parameters import parameters

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State object passed between orchestrator nodes."""
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
    # For multi-question support
    is_multi_query: bool
    sub_queries: List[str]
    sub_answers: List[str]


class AgentWorkflow:
    """
    Orchestrates multi-agent orchestrator for document Q&A.
    """
    
    MAX_RESEARCH_ATTEMPTS: int = parameters.MAX_RESEARCH_ATTEMPTS
    NUM_RESEARCH_CANDIDATES: int = parameters.NUM_RESEARCH_CANDIDATES
    
    def __init__(self, num_candidates: int = None) -> None:
        """Initialize orchestrator with required agents."""
        logger.info("Initializing AgentWorkflow...")
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.context_validator = ContextValidator()
        self.compiled_orchestrator = None
        self.llm = ChatGoogleGenerativeAI(
            model=parameters.LLM_MODEL_NAME,
            google_api_key=parameters.GOOGLE_API_KEY,
            temperature=0.1,
            max_output_tokens=256
        )
        if num_candidates is not None:
            self.NUM_RESEARCH_CANDIDATES = num_candidates
        logger.info(f"AgentWorkflow initialized (candidates={self.NUM_RESEARCH_CANDIDATES})")
        
    def build_orchestrator(self) -> Any:
        """Create and compile the orchestrator graph."""
        logger.debug("Building orchestrator graph...")
        orchestrator = StateGraph(AgentState)
        
        orchestrator.add_node("detect_query_type", self._detect_query_type)
        orchestrator.add_node("process_sub_queries", self._process_sub_queries_step)
        orchestrator.add_node("combine_answers", self._combine_answers_step)
        orchestrator.add_node("check_relevance", self._check_relevance_step)
        orchestrator.add_node("research", self._research_step)
        orchestrator.add_node("verify", self._verification_step)
        
        orchestrator.set_entry_point("detect_query_type")
        orchestrator.add_conditional_edges(
            "detect_query_type",
            lambda state: "multi" if state.get("is_multi_query") else "single",
            {"multi": "process_sub_queries", "single": "check_relevance"}
        )
        orchestrator.add_edge("process_sub_queries", "combine_answers")
        orchestrator.add_edge("combine_answers", END)
        orchestrator.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {"relevant": "research", "irrelevant": END}
        )
        orchestrator.add_edge("research", "verify")
        orchestrator.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {"re_research": "research", "end": END}
        )
        
        return orchestrator.compile()
    
    def _detect_query_type(self, state: AgentState) -> Dict[str, Any]:
        """
        Use LLM to detect if the question is multi-part and decompose it if so.
        """
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
            # Try to extract JSON from the response
            start = content.find('{')
            end = content.rfind('}')
            if start != -1 and end != -1:
                json_str = content[start:end+1]
                result = json.loads(json_str)
                is_multi = bool(result.get("is_multi_query", False))
                sub_queries = result.get("sub_queries", [])
            else:
                # Fallback: treat as single question
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

    def _process_sub_queries_step(self, state: AgentState) -> Dict[str, Any]:
        sub_answers = []
        logger.info(f"[Decompose] Processing {len(state['sub_queries'])} sub-queries...")
        for sub_query in state["sub_queries"]:
            logger.info(f"[Decompose] Processing sub-query: {sub_query}")
            sub_state = state.copy()
            sub_state["question"] = sub_query
            rel = self._check_relevance_step(sub_state)
            if not rel.get("is_relevant"):
                logger.warning(f"[Decompose] Sub-query not relevant: {sub_query}")
                sub_answers.append(rel.get("draft_answer", "No answer found."))
                continue
            sub_state.update(rel)
            research = self._research_step(sub_state)
            sub_state.update(research)
            verify = self._verification_step(sub_state)
            sub_state.update(verify)
            sub_answers.append(sub_state["draft_answer"])
        logger.info(f"[Decompose] Sub-query answers: {sub_answers}")
        return {"sub_answers": sub_answers}

    def _combine_answers_step(self, state: AgentState) -> Dict[str, Any]:
        logger.info(f"[Decompose] Combining {len(state['sub_answers'])} sub-answers into final answer.")
        combined = "\n\n".join(f"Q{i+1}: {q}\nA: {a}" for i, (q, a) in enumerate(zip(state["sub_queries"], state["sub_answers"])))
        return {"draft_answer": combined, "verification_report": "Multi-question answer combined."}
    
    def _check_relevance_step(self, state: AgentState) -> Dict[str, Any]:
        logger.debug("Checking context relevance...")

        result = self.context_validator.context_validate_with_rewrite(
            question=state["question"],
            retriever=state["retriever"],
            k=parameters.RELEVANCE_CHECK_K,   # use config instead of hardcoding 20
            max_rewrites=parameters.MAX_QUERY_REWRITES,
        )

        classification = result.get("classification", "NO_MATCH")
        query_used = result.get("query_used", state["question"])

        logger.info(f"Relevance: {classification} (query_used={query_used[:80]})")

        if classification in ("CAN_ANSWER", "PARTIAL"):
            # ? ALWAYS retrieve docs for the query we’re actually going to answer
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
        """Decide next step after relevance check."""
        return "relevant" if state["is_relevant"] else "irrelevant"
    
    def run_workflow(self, question: str, retriever: BaseRetriever) -> Dict[str, str]:
        """
        Execute the full Q&A pipeline.
        
        Args:
            question: The user's question
            retriever: The retriever for document lookup
            
        Returns:
            Dict with 'draft_answer' and 'verification_report'
        """
        try:
            if self.compiled_orchestrator is None:
                self.compiled_orchestrator = self.build_orchestrator()

            initial_state: AgentState = {
                "question": question,
                "documents": [],  # Let _check_relevance_step fill this
                "draft_answer": "",
                "verification_report": "",
                "is_relevant": False,
                "retriever": retriever,
                "feedback": None,
                "research_attempts": 0,
                "query_used": question,
                "candidate_answers": [],
                "selection_reasoning": "",
                "is_multi_query": False,
                "sub_queries": [],
                "sub_answers": []
            }

            final_state = self.compiled_orchestrator.invoke(initial_state)

            logger.info(f"Pipeline completed (attempts: {final_state.get('research_attempts', 1)})")

            return {
                "draft_answer": final_state["draft_answer"],
                "verification_report": final_state["verification_report"]
            }
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise RuntimeError(f"Workflow execution failed: {e}") from e
    
    def _verification_step(self, state: AgentState) -> Dict[str, Any]:
        """Select the best answer from candidates and verify it."""
        logger.debug("Selecting best answer from candidates...")
        
        candidate_answers = state.get("candidate_answers", []) or [state.get("draft_answer", "")]
        
        # Select the best answer from candidates
        selection_result = self.verifier.select_best_answer(
            candidate_answers=candidate_answers,
            documents=state["documents"],
            question=state["question"]            
        )
        
        best_answer = selection_result["selected_answer"]
        selection_reasoning = selection_result.get("reasoning", "")
        
        logger.info(f"Selected candidate {selection_result['selected_index'] + 1} as best answer")
        
        # Verify the selected answer
        verification_result = self.verifier.check(
            answer=best_answer, 
            documents=state["documents"],
            question=state["question"]
        )
        
        # Enhance verification report with selection info
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
        """Decide whether to re-research or end orchestrator."""
        research_attempts = state.get("research_attempts", 1)
        should_retry = bool(state.get("should_retry", False))
        if should_retry and research_attempts < self.MAX_RESEARCH_ATTEMPTS:
            return "re_research"
        return "end"

    def _research_step(self, state: AgentState) -> Dict[str, Any]:
        """Generate multiple answer candidates using the research agent."""
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
