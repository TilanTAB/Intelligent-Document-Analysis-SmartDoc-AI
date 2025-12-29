from typing import Dict, List, Optional
import logging
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
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
    def __init__(
        self,
        llm: Optional[ChatGoogleGenerativeAI] = None,
        top_k: int = None,
        max_context_chars: int = None,
        max_output_tokens: int = None,
    ) -> None:
        """
        Initialize the research agent with the Gemini model and configuration.
        """
        logger.info("[RESEARCH_AGENT] Initializing...")
        self.top_k = top_k or parameters.RESEARCH_TOP_K
        self.max_context_chars = max_context_chars or parameters.RESEARCH_MAX_CONTEXT_CHARS
        self.max_output_tokens = max_output_tokens or parameters.RESEARCH_MAX_OUTPUT_TOKENS
        self.llm = llm or ChatGoogleGenerativeAI(
            model=parameters.RESEARCH_AGENT_MODEL,
            google_api_key=parameters.GOOGLE_API_KEY,
            temperature=0.2,
            max_output_tokens=self.max_output_tokens
        )
        logger.info(f"[RESEARCH_AGENT] ✓ Initialized (top_k={self.top_k}, model={parameters.RESEARCH_AGENT_MODEL})")

    def sanitize_response(self, response_text: str) -> str:
        """
        Sanitize the LLM's response by stripping unnecessary whitespace.
        """
        return response_text.strip()

    def generate_prompt(self, question: str, context: str, feedback: Optional[str] = None) -> str:
        """
        Generate a structured prompt for the LLM to generate a precise and factual answer.
        Includes special instructions for handling tables, charts, and visualizations.
        """
        base_prompt = f"""You are an AI assistant designed to provide precise and factual answers based on the given context.

**Instructions:**
- Answer the following question using only the provided context.
- Be clear, concise, and factual.
- Return as much information as you can get from the context.
- Only include claims that are directly supported by the context.

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
        # Combine the top document contents into one string
        context = "\n\n".join([doc.page_content for doc in documents[:self.top_k]])
        # Truncate context if too long
        if len(context) > self.max_context_chars:
            logger.debug(f"[RESEARCH_AGENT] Context truncated: {len(context)} -> {self.max_context_chars} chars")
            context = context[:self.max_context_chars]
        # Create a prompt for the LLM (with optional feedback)
        prompt = self.generate_prompt(question, context, feedback)
        # Call the LLM to generate the answer
        try:
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, "content") else str(response)
            answer = content.strip()
            logger.info("[RESEARCH_AGENT] Answer generated successfully")
        except Exception as e:
            logger.error(f"[RESEARCH_AGENT] LLM call failed: {e}", exc_info=True)
            raise RuntimeError("Failed to generate answer due to a model error.") from e
        # Sanitize the response
        draft_answer = self.sanitize_response(answer) if answer else "I cannot answer this question based on the provided documents."
        logger.debug(f"[RESEARCH_AGENT] Answer length: {len(draft_answer)} chars")
        return {
            "draft_answer": draft_answer,
            "context_used": context
        }













