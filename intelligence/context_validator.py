"""
Relevance checker module for document retrieval quality assessment.
"""
from core.llm_factory import get_chat_llm, get_structured_llm
from pydantic import BaseModel, Field
from typing import Literal, Optional, List
import logging
import os
import re
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
        description="Why this classification was selected"
    )


class ContextQueryExpansion(BaseModel):
    """Structured output for query expansion/rewriting."""
    
    rewritten_query: str = Field(
        description="A concise rephrased version of the original query (max 20 words)"
    )
    key_terms: List[str] = Field(
        default_factory=list,
        description="Up to 6 key terms or synonyms to search for"
    )


class ContextValidator:
    """
    Checks context relevance of retrieved documents to a user's question.
    
    Uses the configured LLM with structured output to classify coverage
    and provides query rewriting for improved retrieval.
    """
    
    VALID_LABELS = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
    STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "has", "have", "had", "what", "which", "when", "where", "how", "why", "who",
        "you", "your", "into", "onto", "over", "under", "using", "used", "about",
        "most", "more", "less", "than", "can", "could", "would", "should", "will",
        "may", "might", "task", "tasks", "question", "answer", "context", "passages",
    }
    def __init__(self):
        """Initialize the context validator."""
        logger.info("Initializing ContextValidator...")

        self.max_output_tokens = int(parameters.RELEVANCE_MAX_OUTPUT_TOKENS)
        self.retry_max_output_tokens = max(
            self.max_output_tokens,
            int(parameters.RELEVANCE_RETRY_MAX_OUTPUT_TOKENS),
        )
        
        base_llm = get_chat_llm(
            role="relevance",
            model_name=parameters.RELEVANCE_CHECKER_MODEL,
            temperature=0,
            max_output_tokens=self.max_output_tokens,
        )

        self.llm = base_llm
        self.llm_retry = (
            get_chat_llm(
                role="relevance_retry",
                model_name=parameters.RELEVANCE_CHECKER_MODEL,
                temperature=0,
                max_output_tokens=self.retry_max_output_tokens,
            )
            if self.retry_max_output_tokens > self.max_output_tokens
            else base_llm
        )
        self.structured_llm = get_structured_llm(
            schema=ContextValidationClassification,
            role="relevance_structured",
            model_name=parameters.RELEVANCE_CHECKER_MODEL,
            temperature=0,
            max_output_tokens=self.max_output_tokens,
        )
        self.structured_llm_retry = (
            get_structured_llm(
                schema=ContextValidationClassification,
                role="relevance_structured_retry",
                model_name=parameters.RELEVANCE_CHECKER_MODEL,
                temperature=0,
                max_output_tokens=self.retry_max_output_tokens,
            )
            if self.retry_max_output_tokens > self.max_output_tokens
            else self.structured_llm
        )
        self.query_expansion_llm = get_structured_llm(
            schema=ContextQueryExpansion,
            role="relevance_qe",
            model_name=parameters.RELEVANCE_CHECKER_MODEL,
            temperature=0,
            max_output_tokens=self.max_output_tokens,
        )
        self.query_expansion_llm_retry = (
            get_structured_llm(
                schema=ContextQueryExpansion,
                role="relevance_qe_retry",
                model_name=parameters.RELEVANCE_CHECKER_MODEL,
                temperature=0,
                max_output_tokens=self.retry_max_output_tokens,
            )
            if self.retry_max_output_tokens > self.max_output_tokens
            else self.query_expansion_llm
        )
        
        logger.info(f"ContextValidator initialized (model={parameters.RELEVANCE_CHECKER_MODEL})")

    @classmethod
    def _keyword_set(cls, text: str) -> set:
        terms = {t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", text or "")}
        return {t for t in terms if t not in cls.STOPWORDS}

    def _trace_doc(self, question_terms: set, doc, rank: int) -> str:
        metadata = getattr(doc, "metadata", {}) or {}
        source = metadata.get("source", "?")
        source_name = os.path.basename(str(source)) if source else "?"
        page = metadata.get("page", "?")
        text = (getattr(doc, "page_content", "") or "")
        doc_terms = self._keyword_set(text[:2500])
        overlap = sorted(question_terms.intersection(doc_terms))
        overlap_preview = ", ".join(overlap[:8]) if overlap else "none"
        excerpt = " ".join(text.split())[:180]
        return (
            f"[RELEVANCE_TRACE] rank={rank} source={source_name} page={page} "
            f"chars={len(text)} overlap={len(overlap)} terms=[{overlap_preview}] "
            f"excerpt=\"{excerpt}\""
        )

    @staticmethod
    def _is_length_limit_error(error: Exception) -> bool:
        message = str(error).lower()
        return (
            "length limit was reached" in message
            or ("max tokens" in message and "reached" in message)
            or ("max_output_tokens" in message and "reached" in message)
        )

    def _parse_classification(self, result) -> Optional[ContextValidationClassification]:
        if isinstance(result, ContextValidationClassification):
            return result
        try:
            return ContextValidationClassification.model_validate(result)
        except Exception:
            return None

    def _parse_query_expansion(self, result) -> Optional[ContextQueryExpansion]:
        if isinstance(result, ContextQueryExpansion):
            return result
        try:
            return ContextQueryExpansion.model_validate(result)
        except Exception:
            return None

    def _compress_passages_for_relevance(self, docs, k: int) -> str:
        """
        Build passage text for relevance checking.
        """
        max_docs = max(1, int(k))
        max_chars_per_passage = 900
        max_total_chars = 18_000
        snippets = []
        total_chars = 0

        for idx, doc in enumerate(docs[:max_docs], start=1):
            content = (doc.page_content or "").strip()
            if not content:
                continue
            snippet = " ".join(content.split())
            if not snippet:
                continue
            if len(snippet) > max_chars_per_passage:
                snippet = snippet[:max_chars_per_passage].rstrip() + " …"
            page = doc.metadata.get("page", "?") if hasattr(doc, "metadata") else "?"
            block = f"[Passage {idx} | page {page}] {snippet}"
            projected = total_chars + len(block) + (2 if snippets else 0)
            if projected > max_total_chars:
                break
            snippets.append(block)
            total_chars = projected

        return "\n\n".join(snippets)

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
        
        hint = (context_hint or "").strip()
        context_section = f"\nAvailable context hint: {hint}\n" if hint else ""

        prompt = f"""Rewrite this query for document retrieval.

Original query: {original_query}
{context_section}
Return only:
1) rewritten_query: concise, specific, <= 20 words
2) key_terms: up to 6 short terms/synonyms
No explanations."""

        try:
            result = self.query_expansion_llm.invoke(prompt)
            parsed = self._parse_query_expansion(result)
            if not parsed:
                logger.debug("Query rewrite fallback parsing failed")
                return None
            logger.debug(f"Query rewritten: {parsed.rewritten_query[:60]}...")
            logger.info(
                f"[RELEVANCE_TRACE] rewrite=\"{original_query[:120]}\" -> "
                f"\"{parsed.rewritten_query[:120]}\" key_terms={parsed.key_terms}"
            )
            return parsed
        except Exception as e:
            if self._is_length_limit_error(e) and self.query_expansion_llm_retry is not self.query_expansion_llm:
                logger.warning(
                    "Query rewrite hit output token limit; retrying with a higher token budget "
                    f"({self.retry_max_output_tokens})."
                )
                try:
                    retry_result = self.query_expansion_llm_retry.invoke(prompt)
                    parsed = self._parse_query_expansion(retry_result)
                    if parsed:
                        logger.debug(f"Query rewritten (retry): {parsed.rewritten_query[:60]}...")
                        logger.info(
                            f"[RELEVANCE_TRACE] rewrite(retry)=\"{original_query[:120]}\" -> "
                            f"\"{parsed.rewritten_query[:120]}\" key_terms={parsed.key_terms}"
                        )
                        return parsed
                except Exception as retry_error:
                    logger.error(f"Query rewrite retry failed: {retry_error}")
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
        question_terms = self._keyword_set(question)
        logger.info(
            f"[RELEVANCE_TRACE] retrieved_docs={len(top_docs)} evaluate_top_k={k} "
            f"question_terms={len(question_terms)}"
        )
        for rank, doc in enumerate(top_docs[:k], start=1):
            logger.info(self._trace_doc(question_terms=question_terms, doc=doc, rank=rank))
        
        passages = self._compress_passages_for_relevance(top_docs, k)
        if not passages:
            logger.info("No usable passage content extracted for relevance check")
            return "NO_MATCH"
        logger.info(
            f"[RELEVANCE_TRACE] passage_chars={len(passages)} "
            f"passage_est_tokens={estimate_tokens(passages)}"
        )

        prompt = f"""Classify passage relevance to the question.

Question: {question}

Passages:
{passages}

Rules:
- CAN_ANSWER: passages contain enough evidence to directly answer.
- PARTIAL: passages are topically related but insufficient.
- NO_MATCH: passages are unrelated.
Return only schema fields with short values."""

        try:
            result = self.structured_llm.invoke(prompt)
            parsed = self._parse_classification(result)
            if parsed:
                logger.info(
                    f"Context relevance: {parsed.classification} ({parsed.confidence}) "
                    f"reasoning={parsed.reasoning[:500] if parsed.reasoning else 'n/a'}"
                )
                return parsed.classification
            raise ValueError("Structured classifier returned an unexpected output format.")
        except Exception as e:
            if self._is_length_limit_error(e) and self.structured_llm_retry is not self.structured_llm:
                logger.warning(
                    "Context classifier hit output token limit; retrying with a higher token budget "
                    f"({self.retry_max_output_tokens})."
                )
                try:
                    retry_result = self.structured_llm_retry.invoke(prompt)
                    parsed_retry = self._parse_classification(retry_result)
                    if parsed_retry:
                        logger.info(
                            f"Context relevance: {parsed_retry.classification} "
                            f"({parsed_retry.confidence}) [retry] "
                            f"reasoning={parsed_retry.reasoning[:500] if parsed_retry.reasoning else 'n/a'}"
                        )
                        return parsed_retry.classification
                    logger.warning("Structured classifier retry returned unexpected output format.")
                except Exception as retry_error:
                    logger.error(f"Structured output retry failed: {retry_error}")
            logger.error(f"Structured output failed: {e}")
            
            # Fallback to text parsing
            try:
                fallback_llm = self.llm_retry if self._is_length_limit_error(e) else self.llm
                response = fallback_llm.invoke(prompt)
                raw_response = response.content if hasattr(response, "content") else str(response)
                llm_response = raw_response.strip().upper()
                
                for label in self.VALID_LABELS:
                    if label in llm_response:
                        logger.info(f"Fallback classification: {label} raw={llm_response[:250]}")
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
                logger.info(
                    f"[RELEVANCE_TRACE] rewrite did not improve classification "
                    f"({classification} -> {new_classification})"
                )
            else:
                logger.info("[RELEVANCE_TRACE] rewrite returned no usable query change")
        
        return {
            "classification": classification,
            "query_used": question,
            "was_rewritten": False
        }
    
    def _is_better_classification(self, new: str, old: str) -> bool:
        """Check if new classification is better than old."""
        ranking = {"NO_MATCH": 0, "PARTIAL": 1, "CAN_ANSWER": 2}
        return ranking.get(new, 0) > ranking.get(old, 0)
