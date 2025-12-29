"""
Verification agent module for answer validation against source documents.
"""
from typing import Dict, List, Optional, Literal
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
import logging

from configuration.parameters import parameters

logger = logging.getLogger(__name__)


class VerificationResult(BaseModel):
    """Structured output model for verification results."""
    
    supported: Literal["YES", "NO", "PARTIAL"] = Field(
        description="Whether the answer is supported by the context"
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        default="MEDIUM",
        description="Confidence level in the verification result"
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="Claims not supported by context"
    )
    contradictions: List[str] = Field(
        default_factory=list,
        description="Contradictions between answer and context"
    )
    relevant: Literal["YES", "NO"] = Field(
        description="Whether the answer is relevant to the question"
    )
    completeness: Literal["COMPLETE", "PARTIAL", "INCOMPLETE"] = Field(
        default="PARTIAL",
        description="How completely the answer addresses the question"
    )
    additional_details: str = Field(
        default="",
        description="Additional explanations and reasoning"
    )


class BestAnswerSelection(BaseModel):
    """Structured output model for selecting the best answer from candidates."""
    
    selected_index: int = Field(
        description="The index (0-based) of the best answer from the candidates list"
    )
    reasoning: str = Field(
        description="Explanation of why this answer was selected as the best"
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        default="MEDIUM",
        description="Confidence level in the selection"
    )
    comparison_summary: str = Field(
        default="",
        description="Brief comparison of the candidate answers"
    )


class VerificationAgent:
    """Agent for verifying answers against source documents."""

    def __init__(
        self,
        llm: Optional[ChatGoogleGenerativeAI] = None,
        max_context_chars: int = None,
        max_output_tokens: int = None,
    ) -> None:
        """Initialize the verification agent."""
        logger.info("Initializing VerificationAgent...")

        self.max_context_chars = max_context_chars or parameters.VERIFICATION_MAX_CONTEXT_CHARS
        self.max_output_tokens = max_output_tokens or parameters.VERIFICATION_MAX_OUTPUT_TOKENS
        
        base_llm = llm or ChatGoogleGenerativeAI(
            model=parameters.VERIFICATION_AGENT_MODEL,
            google_api_key=parameters.GOOGLE_API_KEY,
            temperature=0,
            max_output_tokens=self.max_output_tokens,
        )
        
        self.llm = base_llm
        self.structured_llm = base_llm.with_structured_output(VerificationResult)
        self.selection_llm = base_llm.with_structured_output(BestAnswerSelection)
        
        logger.info(f"VerificationAgent initialized (model={parameters.VERIFICATION_AGENT_MODEL})")

    def generate_prompt(self, answer: str, context: str, question: Optional[str] = None) -> str:
        """Generate verification prompt."""
        question_section = f"\n**Original Question:** {question}\n" if question else ""
        
        return f"""Verify the following answer against the provided context.

**Check for:**
1. Factual support (YES/NO/PARTIAL)
2. Confidence level (HIGH/MEDIUM/LOW)
3. Unsupported claims
4. Contradictions
5. Relevance to question
6. Completeness (COMPLETE/PARTIAL/INCOMPLETE)

**Scoring:**
- HIGH: All claims directly stated, no ambiguity
- MEDIUM: Most claims supported, some inferred
- LOW: Significant claims unsupported
{question_section}
**Answer to Verify:**
{answer}

**Context:**
{context}

Provide your verification analysis."""

    def format_verification_report(self, verification: VerificationResult) -> str:
        """Format verification result into readable report."""
        report = f"**Supported:** {verification.supported}\n"
        report += f"**Confidence:** {verification.confidence}\n"
        report += f"**Unsupported Claims:** {', '.join(verification.unsupported_claims) or 'None'}\n"
        report += f"**Contradictions:** {', '.join(verification.contradictions) or 'None'}\n"
        report += f"**Relevant:** {verification.relevant}\n"
        report += f"**Completeness:** {verification.completeness}\n"
        report += f"**Additional Details:** {verification.additional_details or 'None'}\n"
        return report

    def generate_feedback_for_research(self, verification: VerificationResult) -> Optional[str]:
        """Generate feedback for research agent if improvements needed."""
        feedback_parts = []
        if verification.supported == "NO":
            feedback_parts.append("Answer lacks sufficient support from documents.")
        elif verification.supported == "PARTIAL":
            feedback_parts.append("Some parts are not well supported.")
        if verification.unsupported_claims:
            claims_str = "; ".join(verification.unsupported_claims[:3])
            feedback_parts.append(f"Unsupported: {claims_str}")
        if verification.contradictions:
            contradictions_str = "; ".join(verification.contradictions[:3])
            feedback_parts.append(f"Contradictions: {contradictions_str}")
        if verification.completeness == "INCOMPLETE":
            feedback_parts.append("Answer is incomplete.")
        if verification.confidence == "LOW":
            feedback_parts.append("Focus on directly verifiable claims.")
        # Always add additional_details if present, even if other feedback exists
        if verification.additional_details:
            feedback_parts.append(f"Additional Details: {verification.additional_details}")
        return " | ".join(feedback_parts) if feedback_parts else None

    def should_retry_research(self, verification: VerificationResult) -> bool:
        """Determine if research should be retried."""
        if verification.supported == "NO" or verification.relevant == "NO":
            return True
        
        if verification.confidence == "LOW" and (
            verification.unsupported_claims or verification.contradictions
        ):
            return True
        
        if verification.supported == "PARTIAL" and verification.contradictions:
            return True
        
        return False

    def check(self, answer: str, documents: List[Document], question: Optional[str] = None) -> Dict:
        """
        Verify answer against provided documents.
        
        Args:
            answer: The answer to verify
            documents: Source documents for verification
            question: Optional original question
            
        Returns:
            Dict with verification report, context, and metadata
        """
        logger.info(f"Verifying answer ({len(answer)} chars) against {len(documents)} documents")

        context = "\n\n".join([doc.page_content for doc in documents])
        
        if len(context) > self.max_context_chars:
            logger.debug(f"Context truncated: {len(context)} -> {self.max_context_chars}")
            context = context[:self.max_context_chars]

        prompt = self.generate_prompt(answer, context, question)

        try:
            logger.debug("Calling LLM for verification...")
            verification_result: VerificationResult = self.structured_llm.invoke(prompt)
            logger.info(f"Verification: {verification_result.supported} ({verification_result.confidence})")
            
        except Exception as e:
            logger.error(f"Structured output failed: {e}")
            
            try:
                response = self.llm.invoke(prompt)
                report = response.content if hasattr(response, "content") else str(response)
                verification_result = self._parse_unstructured_response(report.strip())
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {fallback_error}")
                verification_result = VerificationResult(
                    supported="NO",
                    confidence="LOW",
                    relevant="NO",
                    completeness="INCOMPLETE",
                    additional_details=f"Verification failed: {str(e)}"
                )

        verification_report = self.format_verification_report(verification_result)
        feedback = self.generate_feedback_for_research(verification_result)
        
        if feedback:
            logger.debug(f"Generated feedback: {feedback[:80]}...")

        return {
            "verification_report": verification_report,
            "context_used": context,
            "structured_result": verification_result.model_dump(),
            "should_retry": self.should_retry_research(verification_result),
            "feedback": feedback
        }

    def select_best_answer(
        self, 
        candidate_answers: List[str], 
        documents: List[Document], 
        question: str
    ) -> Dict:
        """
        Select the best answer from multiple candidates based on verification criteria.
        
        Args:
            candidate_answers: List of candidate answers to evaluate
            documents: Source documents for verification
            question: The original question
            
        Returns:
            Dict with selected answer, index, reasoning, and verification details
        """
        logger.info(f"Selecting best answer from {len(candidate_answers)} candidates")
        
        if len(candidate_answers) == 0:
            logger.warning("No candidate answers provided")
            return {
                "selected_answer": "No answers were generated.",
                "selected_index": -1,
                "reasoning": "No candidates available",
                "confidence": "LOW"
            }
        
        if len(candidate_answers) == 1:
            logger.info("Only one candidate, returning it directly")
            return {
                "selected_answer": candidate_answers[0],
                "selected_index": 0,
                "reasoning": "Only one candidate answer was provided",
                "confidence": "MEDIUM"
            }
        
        context = "\n\n".join([doc.page_content for doc in documents])
        if len(context) > self.max_context_chars:
            logger.debug(f"Context truncated: {len(context)} -> {self.max_context_chars}")
            context = context[:self.max_context_chars]
        
        candidates_text = ""
        for i, answer in enumerate(candidate_answers):
            candidates_text += f"\n**Candidate {i + 1}:**\n{answer}\n"
        
        prompt = f"""You are evaluating multiple candidate answers to select the best one.

**Original Question:** {question}

**Candidate Answers:**
{candidates_text}

**Source Context:**
{context}

**Evaluation Criteria:**
1. **Factual Accuracy**: Which answer is most accurately supported by the context?
2. **Completeness**: Which answer most thoroughly addresses the question?
3. **Relevance**: Which answer stays most focused on what was asked?
4. **No Contradictions**: Which answer has the fewest contradictions with the source?
5. **Clarity**: Which answer is clearest and most well-structured?

Select the best answer by providing its index (0-based) and explain your reasoning."""

        try:
            logger.debug("Calling LLM for best answer selection...")
            selection_result: BestAnswerSelection = self.selection_llm.invoke(prompt)
            
            selected_index = selection_result.selected_index
            if selected_index < 0 or selected_index >= len(candidate_answers):
                logger.warning(f"Invalid selection index {selected_index}, defaulting to 0")
                selected_index = 0
            
            logger.info(f"Selected candidate {selected_index + 1} with {selection_result.confidence} confidence")
            
            return {
                "selected_answer": candidate_answers[selected_index],
                "selected_index": selected_index,
                "reasoning": selection_result.reasoning,
                "confidence": selection_result.confidence,
                "comparison_summary": selection_result.comparison_summary
            }
            
        except Exception as e:
            logger.error(f"Best answer selection failed: {e}")
            # Fallback: return the first candidate
            return {
                "selected_answer": candidate_answers[0],
                "selected_index": 0,
                "reasoning": f"Selection failed, using first candidate: {str(e)}",
                "confidence": "LOW"
            }

    def _parse_unstructured_response(self, response_text: str) -> VerificationResult:
        """Parse unstructured response into VerificationResult (fallback)."""
        try:
            data = {
                "supported": "NO",
                "confidence": "LOW",
                "unsupported_claims": [],
                "contradictions": [],
                "relevant": "NO",
                "completeness": "INCOMPLETE",
                "additional_details": ""
            }
            
            for line in response_text.split('\n'):
                if ':' not in line:
                    continue
                    
                key, value = line.split(':', 1)
                key = key.strip().lower().replace(' ', '_')
                value = value.strip().upper()
                
                if key == "SUPPORTED":
                    data["supported"] = "YES" if "YES" in value else ("PARTIAL" if "PARTIAL" in value else "NO")
                elif key == "CONFIDENCE":
                    data["confidence"] = "HIGH" if "HIGH" in value else ("MEDIUM" if "MEDIUM" in value else "LOW")
                elif key == "RELEVANT":
                    data["relevant"] = "YES" if "YES" in value else "NO"
                elif key == "COMPLETENESS":
                    if "COMPLETE" in value and "INCOMPLETE" not in value:
                        data["completeness"] = "COMPLETE"
                    elif "PARTIAL" in value:
                        data["completeness"] = "PARTIAL"
            
            return VerificationResult(**data)
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return VerificationResult(
                supported="NO",
                confidence="LOW",
                relevant="NO",
                completeness="INCOMPLETE",
                additional_details="Failed to parse verification response"
            )
