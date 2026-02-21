"""
Verification agent module for answer validation against source documents.
"""
from typing import Dict, List, Optional, Literal
from langchain_core.documents import Document
from core.llm_factory import get_chat_llm, get_structured_llm
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
        llm=None,
        max_context_chars: int = None,
        max_output_tokens: int = None,
    ) -> None:
        """Initialize the verification agent."""
        logger.info("Initializing VerificationAgent...")

        self.max_context_chars = max_context_chars or parameters.VERIFICATION_MAX_CONTEXT_CHARS
        self.max_output_tokens = int(max_output_tokens or parameters.VERIFICATION_MAX_OUTPUT_TOKENS)
        self.retry_max_output_tokens = max(
            self.max_output_tokens,
            int(parameters.VERIFICATION_RETRY_MAX_OUTPUT_TOKENS),
        )
        
        base_llm = llm or get_chat_llm(
            role="verification",
            model_name=parameters.VERIFICATION_AGENT_MODEL,
            temperature=0,
            max_output_tokens=self.max_output_tokens,
        )

        self.llm = base_llm
        self.llm_retry = (
            get_chat_llm(
                role="verification_retry",
                model_name=parameters.VERIFICATION_AGENT_MODEL,
                temperature=0,
                max_output_tokens=self.retry_max_output_tokens,
            )
            if self.retry_max_output_tokens > self.max_output_tokens
            else base_llm
        )
        self.structured_llm = get_structured_llm(
            schema=VerificationResult,
            role="verification_structured",
            model_name=parameters.VERIFICATION_AGENT_MODEL,
            temperature=0,
            max_output_tokens=self.max_output_tokens,
        )
        self.structured_llm_retry = (
            get_structured_llm(
                schema=VerificationResult,
                role="verification_structured_retry",
                model_name=parameters.VERIFICATION_AGENT_MODEL,
                temperature=0,
                max_output_tokens=self.retry_max_output_tokens,
            )
            if self.retry_max_output_tokens > self.max_output_tokens
            else self.structured_llm
        )
        self.selection_llm = get_structured_llm(
            schema=BestAnswerSelection,
            role="verification_selection",
            model_name=parameters.VERIFICATION_AGENT_MODEL,
            temperature=0,
            max_output_tokens=self.max_output_tokens,
        )
        self.selection_llm_retry = (
            get_structured_llm(
                schema=BestAnswerSelection,
                role="verification_selection_retry",
                model_name=parameters.VERIFICATION_AGENT_MODEL,
                temperature=0,
                max_output_tokens=self.retry_max_output_tokens,
            )
            if self.retry_max_output_tokens > self.max_output_tokens
            else self.selection_llm
        )
        
        logger.info(f"VerificationAgent initialized (model={parameters.VERIFICATION_AGENT_MODEL})")

    @staticmethod
    def _is_length_limit_error(error: Exception) -> bool:
        message = str(error).lower()
        return (
            "length limit was reached" in message
            or ("max tokens" in message and "reached" in message)
            or ("max_output_tokens" in message and "reached" in message)
        )

    @staticmethod
    def _parse_verification_result(result) -> Optional[VerificationResult]:
        if isinstance(result, VerificationResult):
            return result
        try:
            return VerificationResult.model_validate(result)
        except Exception:
            return None

    @staticmethod
    def _parse_selection_result(result) -> Optional[BestAnswerSelection]:
        if isinstance(result, BestAnswerSelection):
            return result
        try:
            return BestAnswerSelection.model_validate(result)
        except Exception:
            return None

    def _build_context(
        self,
        documents: List[Document],
    ) -> str:
        blocks: List[str] = []
        for idx, doc in enumerate(documents, start=1):
            text = (doc.page_content or "").strip()
            if not text:
                continue
            page = doc.metadata.get("page", "?") if isinstance(doc.metadata, dict) else "?"
            blocks.append(f"[Doc {idx} | page {page}] {text}")
        context = "\n\n".join(blocks)
        if self.max_context_chars and len(context) > self.max_context_chars:
            context = context[:self.max_context_chars]
        return context

    def _build_candidates(self, candidate_answers: List[str]) -> str:
        items = []
        for i, answer in enumerate(candidate_answers):
            items.append(f"[Candidate {i}] {(answer or '').strip()}")
        return "\n\n".join(items)

    def generate_prompt(self, answer: str, context: str, question: Optional[str] = None) -> str:
        """Generate verification prompt."""
        question_section = f"Question: {question}\n" if question else ""

        return f"""Verify the answer against context.

{question_section}Answer:
{answer}

Context:
{context}

Return only schema fields.
Rules:
- supported: YES/PARTIAL/NO
- relevant: YES/NO
- completeness: COMPLETE/PARTIAL/INCOMPLETE
- list unsupported_claims and contradictions as needed."""

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

    def should_retry_research(self, verification: VerificationResult, verification_report: str = None, feedback: Optional[str] = None) -> bool:
        """Determine if research should be retried."""
        # Use structured fields first
        if verification.supported == "NO" or verification.relevant == "NO":
            return True
        if verification.confidence == "LOW" and (
            verification.unsupported_claims or verification.contradictions
        ):
            return True
        if verification.supported == "PARTIAL" and verification.contradictions:
            return True
        # Also check verification_report string for extra signals (legacy/fallback)
        if verification_report:
            if "Supported: NO" in verification_report:
                logger.warning("[Re-Research] Answer not supported; triggering re-research.")
                return True
            elif "Relevant: NO" in verification_report:
                logger.warning("[Re-Research] Answer not relevant; triggering re-research.")
                return True
            elif "Confidence: LOW" in verification_report and "Supported: PARTIAL" in verification_report:
                logger.warning("[Re-Research] Low confidence with partial support; triggering re-research.")
                return True
            elif "Completeness: INCOMPLETE" in verification_report:
                logger.warning("[Re-Research] Answer is incomplete; triggering re-research.")
                return True
            elif "Completeness: PARTIAL" in verification_report:
                logger.warning("[Re-Research] Answer is partially complete; triggering re-research.")
                return True
        # Check feedback for contradiction/unsupported
        if feedback and ("contradiction" in feedback.lower() or "unsupported" in feedback.lower()):
            logger.warning("[Re-Research] Feedback indicates contradiction/unsupported; triggering re-research.")
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

        context = self._build_context(documents=documents)
        if not context:
            context = "No usable context extracted from documents."

        prompt = self.generate_prompt(answer, context, question)

        try:
            logger.debug("Calling LLM for verification...")
            verification_raw = self.structured_llm.invoke(prompt)
            verification_result = self._parse_verification_result(verification_raw)
            if verification_result is None:
                raise ValueError("Structured verifier returned unexpected output format.")
            logger.info(f"Verification: {verification_result.supported} ({verification_result.confidence})")
            
        except Exception as e:
            if self._is_length_limit_error(e) and self.structured_llm_retry is not self.structured_llm:
                logger.warning(
                    "Verification hit output token limit; retrying with a higher token budget "
                    f"({self.retry_max_output_tokens})."
                )
                try:
                    retry_raw = self.structured_llm_retry.invoke(prompt)
                    retry_result = self._parse_verification_result(retry_raw)
                    if retry_result is not None:
                        verification_result = retry_result
                        logger.info(f"Verification: {verification_result.supported} ({verification_result.confidence}) [retry]")
                        verification_report = self.format_verification_report(verification_result)
                        feedback = self.generate_feedback_for_research(verification_result)
                        return {
                            "verification_report": verification_report,
                            "context_used": context,
                            "structured_result": verification_result.model_dump(),
                            "should_retry": self.should_retry_research(verification_result, verification_report, feedback),
                            "feedback": feedback
                        }
                except Exception as retry_error:
                    logger.error(f"Verification retry failed: {retry_error}")

            logger.error(f"Structured output failed: {e}")
            
            try:
                fallback_llm = self.llm_retry if self._is_length_limit_error(e) else self.llm
                response = fallback_llm.invoke(prompt)
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
            "should_retry": self.should_retry_research(verification_result, verification_report, feedback),
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
        
        context = self._build_context(documents=documents)
        if not context:
            context = "No usable context extracted from documents."
        
        candidates_text = self._build_candidates(candidate_answers)
        
        prompt = f"""Select the best candidate answer.

Question: {question}

Candidates (0-based):
{candidates_text}

Context:
{context}

Select by factual support, completeness, relevance, and clarity.
Return only schema fields with concise text."""

        try:
            logger.debug("Calling LLM for best answer selection...")
            selection_raw = self.selection_llm.invoke(prompt)
            selection_result = self._parse_selection_result(selection_raw)
            if selection_result is None:
                raise ValueError("Selection model returned unexpected output format.")

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
            if self._is_length_limit_error(e) and self.selection_llm_retry is not self.selection_llm:
                logger.warning(
                    "Best-answer selection hit output token limit; retrying with a higher token budget "
                    f"({self.retry_max_output_tokens})."
                )
                try:
                    retry_raw = self.selection_llm_retry.invoke(prompt)
                    retry_result = self._parse_selection_result(retry_raw)
                    if retry_result is not None:
                        selected_index = retry_result.selected_index
                        if selected_index < 0 or selected_index >= len(candidate_answers):
                            selected_index = 0
                        logger.info(f"Selected candidate {selected_index + 1} with {retry_result.confidence} confidence [retry]")
                        return {
                            "selected_answer": candidate_answers[selected_index],
                            "selected_index": selected_index,
                            "reasoning": retry_result.reasoning,
                            "confidence": retry_result.confidence,
                            "comparison_summary": retry_result.comparison_summary
                        }
                except Exception as retry_error:
                    logger.error(f"Best answer selection retry failed: {retry_error}")

            logger.error(f"Best answer selection failed: {e}")
            # Fallback: choose the most substantial candidate instead of always picking index 0.
            fallback_index = max(
                range(len(candidate_answers)),
                key=lambda i: len((candidate_answers[i] or "").strip()),
            )
            fallback_len = len((candidate_answers[fallback_index] or "").strip())
            logger.warning(
                "Falling back to candidate %d based on max content length (%d chars).",
                fallback_index + 1,
                fallback_len,
            )
            return {
                "selected_answer": candidate_answers[fallback_index],
                "selected_index": fallback_index,
                "reasoning": f"Selection failed, using longest candidate: {str(e)}",
                "confidence": "LOW",
                "comparison_summary": ""
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
                
                if key == "supported":
                    data["supported"] = "YES" if "YES" in value else ("PARTIAL" if "PARTIAL" in value else "NO")
                elif key == "confidence":
                    data["confidence"] = "HIGH" if "HIGH" in value else ("MEDIUM" if "MEDIUM" in value else "LOW")
                elif key == "relevant":
                    data["relevant"] = "YES" if "YES" in value else "NO"
                elif key == "completeness":
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
