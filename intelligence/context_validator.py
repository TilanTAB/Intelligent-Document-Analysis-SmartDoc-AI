"""
Relevance checker module for document retrieval quality assessment.
"""
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
import logging
from configuration.parameters import parameters

logger = logging.getLogger(__name__)


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """Estimate token count from text length."""
    return len(text) // chars_per_token


# ============================================================================
# Structured Output Models
# ============================================================================

class ContextValidationClassification(BaseModel):
    """Structured output for context validation classification."""
    
    classification: Literal["CAN_ANSWER", "PARTIAL", "NO_MATCH"] = Field(
        description=(
            "CAN_ANSWER: Passages contain enough info to fully answer. "
            "PARTIAL: Passages mention the topic but incomplete. "
            "NO_MATCH: Passages don't discuss the topic at all."
        )
    )
    confidence: Literal["HIGH", "MEDIUM", "LOW"] = Field(
        default="MEDIUM",
        description="Confidence level in the classification"
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation for the classification"
    )


class ContextQueryExpansion(BaseModel):
    """Structured output for query expansion/rewriting."""
    
    rewritten_query: str = Field(
        description="A rephrased version of the original query"
    )
    key_terms: List[str] = Field(
        default_factory=list,
        description="Key terms and synonyms to search for"
    )
    search_strategy: str = Field(
        default="",
        description="Brief explanation of the search approach"
    )


class ContextValidator:
    """
    Checks context relevance of retrieved documents to a user's question.
    
    Uses Gemini model with structured output to classify coverage
    and provides query rewriting for improved retrieval.
    """
    
    VALID_LABELS = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
    
    def __init__(self):
        """Initialize the context validator."""
        logger.info("Initializing ContextValidator...")
        
        base_llm = ChatGoogleGenerativeAI(
            model=parameters.RELEVANCE_CHECKER_MODEL,
            google_api_key=parameters.GOOGLE_API_KEY,
            temperature=0,
            max_output_tokens=100,
        )
        
        self.llm = base_llm
        self.structured_llm = base_llm.with_structured_output(ContextValidationClassification)
        self.query_expansion_llm = base_llm.with_structured_output(ContextQueryExpansion)
        
        logger.info(f"ContextValidator initialized (model={parameters.RELEVANCE_CHECKER_MODEL})")

    def context_query_rewrite(self, original_query: str, context_hint: Optional[str] = None) -> Optional[ContextQueryExpansion]:
        """
        Rewrite a query to potentially retrieve better results.
        
        Args:
            original_query: The original user query
            context_hint: Optional hint about available documents
            
        Returns:
            ContextQueryExpansion with rewritten query, or None on failure
        """
        logger.debug(f"Rewriting query: {original_query[:80]}...")
        
        context_section = f"\n**Available Context:** {context_hint}\n" if context_hint else ""
        
        prompt = f"""Rewrite this query to improve document retrieval.

**Original Query:** {original_query}
{context_section}
**Instructions:**
1. Rephrase to be more specific and searchable
2. Extract key terms and add synonyms
3. Consider exact phrases in formal documents"""

        try:
            result: ContextQueryExpansion = self.query_expansion_llm.invoke(prompt)
            logger.debug(f"Query rewritten: {result.rewritten_query[:60]}...")
            return result
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            return None

    def context_validate(self, question: str, retriever, k: int = 3) -> str:
        """
        Retrieve top-k passages and classify coverage.
        
        Args:
            question: The user's question
            retriever: The retriever for fetching documents
            k: Number of top documents to consider
            
        Returns:
            Classification: "CAN_ANSWER", "PARTIAL", or "NO_MATCH"
        """
        if not question or not question.strip():
            logger.warning("Empty question provided")
            return "NO_MATCH"
        
        if k < 1:
            k = 3

        logger.info(f"Checking context relevance for: {question[:60]}...")

        # Retrieve documents
        try:
            top_docs = retriever.invoke(question)
        except Exception as e:
            logger.error(f"Retriever invocation failed: {e}")
            return "NO_MATCH"

        if not top_docs:
            logger.info("No documents returned")
            return "NO_MATCH"

        logger.debug(f"Retrieved {len(top_docs)} documents")
        
        passages = "\n\n".join(doc.page_content for doc in top_docs[:k])

        prompt = f"""Classify how well the passages address the question.

**Question:** {question}

**Passages:**
{passages}

Classify as CAN_ANSWER (fully answers), PARTIAL (mentions topic), or NO_MATCH (unrelated)."""

        try:
            result: ContextValidationClassification = self.structured_llm.invoke(prompt)
            logger.info(f"Context relevance: {result.classification} ({result.confidence})")
            return result.classification
            
        except Exception as e:
            logger.error(f"Structured output failed: {e}")
            
            # Fallback to text parsing
            try:
                response = self.llm.invoke(prompt)
                raw_response = response.content if hasattr(response, "content") else str(response)
                llm_response = raw_response.strip().upper()
                
                for label in self.VALID_LABELS:
                    if label in llm_response:
                        logger.info(f"Fallback classification: {label}")
                        return label
                
                return "NO_MATCH"
                
            except Exception as fallback_error:
                logger.error(f"Fallback failed: {fallback_error}")
                return "NO_MATCH"

    def context_validate_with_rewrite(self, question: str, retriever, k: int = 3, max_rewrites: int = 1) -> dict:
        """
        Check relevance with automatic query rewriting if needed.
        
        Args:
            question: The user's question
            retriever: The retriever to use
            k: Number of top documents
            max_rewrites: Maximum rewrite attempts
            
        Returns:
            Dict with classification, query_used, and was_rewritten
        """
        classification = self.context_validate(question, retriever, k)
        
        if classification == "CAN_ANSWER" or max_rewrites <= 0:
            return {
                "classification": classification,
                "query_used": question,
                "was_rewritten": False
            }
        
        # Try query rewriting for poor results
        if classification in ["PARTIAL", "NO_MATCH"]:
            logger.info("Attempting query rewrite...")
            
            expansion = self.context_query_rewrite(question)
            if expansion and expansion.rewritten_query != question:
                new_classification = self.context_validate(expansion.rewritten_query, retriever, k)
                
                if self._is_better_classification(new_classification, classification):
                    logger.info(f"Rewrite improved: {classification} -> {new_classification}")
                    return {
                        "classification": new_classification,
                        "query_used": expansion.rewritten_query,
                        "was_rewritten": True,
                        "key_terms": expansion.key_terms
                    }
        
        return {
            "classification": classification,
            "query_used": question,
            "was_rewritten": False
        }
    
    def _is_better_classification(self, new: str, old: str) -> bool:
        """Check if new classification is better than old."""
        ranking = {"NO_MATCH": 0, "PARTIAL": 1, "CAN_ANSWER": 2}
        return ranking.get(new, 0) > ranking.get(old, 0)
