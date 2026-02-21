from typing import Any, Dict, List, Optional
import logging
import re
from langchain_core.documents import Document
from core.llm_factory import get_chat_llm
from configuration.parameters import parameters

logger = logging.getLogger(__name__)


def estimate_tokens(text: str, chars_per_token: int = 4) -> int:
    """
    Estimate token count from text length.
    """
    return len(text) // chars_per_token


class ResearchAgent:
    """
    ResearchAgent generates answers to user questions using Gemini LLM,
    focusing on extracting factual, source-cited information from documents.
    """
    STOPWORDS = {
        "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
        "has", "have", "had", "what", "which", "when", "where", "how", "why", "who",
        "into", "onto", "over", "under", "using", "used", "about", "most", "more",
        "less", "than", "can", "could", "would", "should", "will", "may", "might",
        "models",  # extremely common in this domain; low retrieval value on its own
    }
    CHART_PLACEHOLDER_TEXT = "Analysis unavailable (parsing error)"

    def __init__(
        self,
        llm=None,
        top_k: int = None,
        max_context_chars: int = None,
        max_output_tokens: int = None,
        min_answer_chars: int = None,
    ) -> None:
        """
        Initialize the research agent with the Gemini model and configuration.
        """
        logger.info("[RESEARCH_AGENT] Initializing...")
        configured_top_k = int(top_k if top_k is not None else parameters.RESEARCH_TOP_K)
        self.top_k = max(1, configured_top_k)
        # Keep evidence scope at least as wide as relevance screening unless caller explicitly overrides top_k.
        if top_k is None and self.top_k < int(parameters.RELEVANCE_CHECK_K):
            self.top_k = int(parameters.RELEVANCE_CHECK_K)
            logger.info(
                "[RESEARCH_AGENT] top_k raised to RELEVANCE_CHECK_K=%d to avoid dropping evidence "
                "that passed relevance validation.",
                self.top_k,
            )
        self.max_context_chars = max_context_chars or parameters.RESEARCH_MAX_CONTEXT_CHARS
        self.max_output_tokens = max_output_tokens or parameters.RESEARCH_MAX_OUTPUT_TOKENS
        self.min_answer_chars = int(min_answer_chars or parameters.RESEARCH_MIN_ANSWER_CHARS)
        self.llm = llm or get_chat_llm(
            role="research",
            model_name=parameters.RESEARCH_AGENT_MODEL,
            temperature=0.2,
            max_output_tokens=self.max_output_tokens,
        )
        logger.info(f"[RESEARCH_AGENT] ✓ Initialized (top_k={self.top_k}, model={parameters.RESEARCH_AGENT_MODEL})")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    @classmethod
    def _extract_question_terms(cls, question: str) -> List[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", (question or "").lower())
        return [token for token in tokens if token not in cls.STOPWORDS]

    @staticmethod
    def _doc_key(doc: Document) -> tuple:
        metadata = doc.metadata if isinstance(getattr(doc, "metadata", None), dict) else {}
        return (
            metadata.get("source"),
            str(metadata.get("page")),
            metadata.get("chunk_id"),
            hash(doc.page_content or ""),
        )

    @classmethod
    def _is_placeholder_chart_chunk(cls, doc: Document) -> bool:
        metadata = doc.metadata if isinstance(getattr(doc, "metadata", None), dict) else {}
        if metadata.get("type") != "chart":
            return False
        text = (doc.page_content or "").strip()
        return cls.CHART_PLACEHOLDER_TEXT in text

    def _prioritize_documents(self, question: str, documents: List[Document]) -> List[Document]:
        """
        Re-rank retrieved docs by lexical overlap so exact-answer passages are less likely
        to be excluded by top_k truncation.
        """
        if not documents:
            return []

        # De-prioritize known bad chart placeholders so they don't pollute context.
        non_placeholder_docs = [doc for doc in documents if not self._is_placeholder_chart_chunk(doc)]
        candidate_docs = non_placeholder_docs if non_placeholder_docs else documents

        question_terms = self._extract_question_terms(question)
        if not question_terms:
            return candidate_docs

        scored: List[tuple] = []
        for idx, doc in enumerate(candidate_docs):
            text = (doc.page_content or "").lower()
            overlap = sum(1 for term in question_terms if term in text)
            has_numeric_signal = 1 if re.search(r"\d", text) else 0
            scored.append((overlap, has_numeric_signal, -idx, doc))

        # If overlap cannot discriminate, preserve retriever order.
        if max((item[0] for item in scored), default=0) <= 0:
            return candidate_docs

        overlap_ranked = [item[3] for item in sorted(scored, key=lambda item: item[:3], reverse=True)]

        # Merge overlap-first + original-order for stability/diversity.
        merged: List[Document] = []
        seen = set()
        for candidate in overlap_ranked + candidate_docs:
            key = self._doc_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            merged.append(candidate)
        return merged

    def _build_context(self, question: str, documents: List[Document]) -> str:
        prioritized_docs = self._prioritize_documents(question=question, documents=documents)
        selected_docs = prioritized_docs[: self.top_k]
        max_chars_per_doc = max(500, min(1200, int(self.max_context_chars // max(1, self.top_k * 2))))

        blocks: List[str] = []
        total_chars = 0

        for idx, doc in enumerate(selected_docs, start=1):
            metadata = doc.metadata if isinstance(getattr(doc, "metadata", None), dict) else {}
            source = metadata.get("source", "?")
            page = metadata.get("page", "?")
            content = (doc.page_content or "").strip()
            if not content:
                continue
            if len(content) > max_chars_per_doc:
                content = content[:max_chars_per_doc].rstrip() + " …"

            block = f"[Doc {idx} | source={source} | page={page}]\n{content}"
            projected = total_chars + len(block) + (2 if blocks else 0)
            if projected > self.max_context_chars:
                if not blocks:
                    blocks.append(block[: self.max_context_chars])
                break

            blocks.append(block)
            total_chars = projected

        logger.debug(
            "[RESEARCH_AGENT] Context built with %d docs, chars=%d (top_k=%d)",
            len(blocks),
            len("\n\n".join(blocks)),
            self.top_k,
        )
        return "\n\n".join(blocks)

    def generate_prompt(self, question: str, context: str, feedback: Optional[str] = None) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        Includes special instructions for handling tables, charts, and visualizations.
        """
        base_prompt = f"""You are an AI assistant designed to provide precise and factual answers based on the given context.

**Instructions:**
- Answer the following question using only the provided context.
- Be clear, factual, and sufficiently detailed.
- Return as much information as you can get from the context.
- Only include claims that are directly supported by the context.
- When context supports it, provide at least 4-6 sentences with specific evidence.

**IMPORTANT - Data Consolidation:**
- If multiple charts, tables, or data sources provide similar information, CONSOLIDATE the data and provide a single, unified answer.
- DO NOT list or compare values from multiple versions of the same charts/tables separately.
- Present only the most relevant or consensus value for each data point, unless there is a clear, significant difference that must be explained.
- If there are minor discrepancies, choose the value that appears most frequently or is best supported by the context, and mention only that value.

**IMPORTANT - Chart and Page Reference:**
- When referencing data from a chart, always indicate the chart's heading or title, and also include the page title if available.
- Do NOT use phrases like "another chart" or "a different chart". Always refer to the chart by its heading/title and the page title if you need to mention the source.

**CRITICAL - Table, Chart, and Visualization Handling:**
- Pay VERY CLOSE attention to any tables in the context (formatted with | characters or markdown table format).
- Tables contain structured data - read them carefully row by row, column by column.
- Extract and cite specific numbers, percentages, scores, and ratings from tables.
- If a numbered table (Table 1, Table 4, etc.) is relevant, explicitly mention it and provide the exact values.
- **Analyze complex charts and visualizations** when present in the context:
  - Look for chart descriptions, data points, trends, and patterns
  - Extract specific values from line charts, bar charts, pie charts, and scatter plots
  - Identify trends, correlations, and relationships shown in visualizations
  - Note any zones, quadrants, or regions in complex diagrams
  - Reference chart titles, axis labels, and legends when citing data
  - Compare multiple visualizations if relevant to the question
"""

        if feedback:
            base_prompt += f"""
**IMPORTANT - Previous Answer Feedback:**
Your previous answer had issues that need to be addressed:
{feedback}

Please generate an improved answer that:
1. Addresses the unsupported claims by finding support in the context tables and charts
2. Fixes any contradictions with the source material
3. Ensures all statements are verifiable from the context
4. Look carefully at ALL tables and visualizations - the data you need may be in a numbered table or chart description
5. Read table data and chart descriptions carefully - each row/data point represents specific information
"""

        base_prompt += f"""
**Question:** {question}

**Context (pay special attention to tables marked with ### Table, chart descriptions, and data visualizations):**
{context}

**Provide your answer below (cite specific table numbers, chart references, and exact values from the tables and visualizations):**
"""
        return base_prompt

    @staticmethod
    def _extract_text(response: Any) -> str:
        """
        Normalize LangChain/SDK response content into plain text.
        Handles string content plus list/dict content blocks used by newer providers.
        """
        text_attr = getattr(response, "text", None)
        if isinstance(text_attr, str) and text_attr.strip():
            return text_attr

        content = response.content if hasattr(response, "content") else response

        if isinstance(content, str):
            return content

        if isinstance(content, dict):
            for key in ("text", "content", "output_text"):
                value = content.get(key)
                if isinstance(value, str):
                    return value
            return ""

        if isinstance(content, list):
            chunks: List[str] = []
            for item in content:
                if isinstance(item, str):
                    if item.strip():
                        chunks.append(item.strip())
                    continue

                if isinstance(item, dict):
                    for key in ("text", "content", "output_text"):
                        value = item.get(key)
                        if isinstance(value, str) and value.strip():
                            chunks.append(value.strip())
                            break
                    continue

                text_attr = getattr(item, "text", None)
                if isinstance(text_attr, str) and text_attr.strip():
                    chunks.append(text_attr.strip())

            return "\n".join(chunks)

        return str(content or "")

    def _build_expansion_prompt(self, question: str, context: str, short_answer: str) -> str:
        return f"""The draft answer is too short. Rewrite it into a detailed, evidence-based answer.

Question:
{question}

Current short draft:
{short_answer or "(empty response)"}

Context:
{context}

Requirements:
- Use only the context.
- Provide a complete answer with concrete facts and figures when available.
- Mention specific table/chart/page evidence when present.
- Target at least {self.min_answer_chars} characters.
- If context is insufficient, explicitly state what is missing.

Return only the final answer text."""

    def generate(
        self,
        question: str,
        documents: List[Document],
        feedback: Optional[str] = None,
        previous_answer: Optional[str] = None
    ) -> Dict:
        """
        Generate an initial answer using the provided documents.
        Args:
            question: The user's question
            documents: List of relevant documents
            feedback: Optional feedback from verification agent for re-research
            previous_answer: Optional previous answer that failed verification
        Returns:
            Dict with 'draft_answer' and 'context_used'
        """
        logger.info(f"[RESEARCH_AGENT] Generating answer for: {question[:80]}...")
        logger.debug(f"[RESEARCH_AGENT] Documents: {len(documents)}, Feedback: {feedback is not None}")
        if not documents:
            logger.warning("[RESEARCH_AGENT] No documents provided")
            return {
                "draft_answer": "I could not find supporting documents to answer this question.",
                "context_used": ""
            }
        # Build a focused context bundle instead of taking the first N docs blindly.
        context = self._build_context(question=question, documents=documents)
        # Create a prompt for the LLM (with optional feedback)
        prompt = self.generate_prompt(question, context, feedback)
        # Call the LLM to generate the answer
        try:
            response = self.llm.invoke(prompt)
            answer = self._extract_text(response).strip()
            if not answer:
                metadata = getattr(response, "response_metadata", {}) or {}
                logger.warning(
                    "[RESEARCH_AGENT] Empty model output (finish_reason=%s, usage=%s)",
                    metadata.get("finish_reason", "unknown"),
                    metadata.get("token_usage", metadata.get("usage", "unknown")),
                )

            if len(answer) < self.min_answer_chars:
                logger.warning(
                    "[RESEARCH_AGENT] Short answer detected (%d chars). Retrying with expansion prompt (target >= %d chars).",
                    len(answer),
                    self.min_answer_chars,
                )
                expansion_prompt = self._build_expansion_prompt(question, context, answer)
                expansion_response = self.llm.invoke(expansion_prompt)
                expanded_answer = self._extract_text(expansion_response).strip()
                if len(expanded_answer) >= len(answer):
                    answer = expanded_answer

            logger.info("[RESEARCH_AGENT] Answer generated successfully")
        except Exception as e:
            logger.error(f"[RESEARCH_AGENT] LLM call failed: {e}", exc_info=True)
            raise RuntimeError("Failed to generate answer due to a model error.") from e
        # Sanitize the response
        draft_answer = (
            self.sanitize_response(answer)
            if answer
            else "I could not generate a supported answer from the provided context. Please rephrase the question or provide more relevant source content."
        )
        logger.debug(f"[RESEARCH_AGENT] Answer length: {len(draft_answer)} chars")
        return {
            "draft_answer": draft_answer,
            "context_used": context
        }













