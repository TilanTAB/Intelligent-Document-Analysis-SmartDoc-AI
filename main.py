import logging
from configuration.logger_setup import configure_logging

logger = logging.getLogger(__name__)

import hashlib
import socket
from typing import Any, List, Dict, Tuple
import os
import shutil
from pathlib import Path
from datetime import datetime
import time
import random
import re
from collections import defaultdict, deque
import threading
import queue   # B2: streaming chunker pipeline

from content_analyzer.document_parser import DocumentProcessor
from search_engine.indexer import RetrieverBuilder
from intelligence.orchestrator import AgentWorkflow
from configuration import definitions, parameters
from core.telemetry import (
    initialize_telemetry,
    mark_span_error,
    record_request_metrics,
    start_span,
)
from core.user_analytics import get_analytics_logger
                             

# Rate limiting state (configurable via RATE_LIMIT_* settings)
_calls = defaultdict(deque)  # ip -> timestamps
_calls_lock = threading.Lock()  # Thread-safe access to rate limit state

def rate_limit(request):
    """Thread-safe rate limiting per IP address."""
    if not parameters.RATE_LIMIT_ENABLED:
        return

    ip = getattr(request.client, "host", "unknown")
    now = time.time()
    window_s = int(parameters.RATE_LIMIT_WINDOW_S)
    max_calls = int(parameters.RATE_LIMIT_MAX_CALLS)

    with _calls_lock:
        q = _calls[ip]
        # Remove expired entries
        while q and (now - q[0]) > window_s:
            q.popleft()
    
        if len(q) >= max_calls:
            import gradio as gr
            raise gr.Error(f"Rate limit: {max_calls} requests per {window_s//60} minutes. Please wait.")
    
        q.append(now)


# Example data for demo
EXAMPLES = {
    "Generative AI and Jobs": {
        "question": "Which occupations are most likely to be automated by AI?",
        "file_paths": ["samples/OIT-NASK-IAGen_WP140_web.pdf"]  
    },
    "Energy and AI": {
        "question": "What is the accuracy of AI models in coding?",
        "file_paths": ["samples/EnergyandAI.pdf"]
    },
     "Digital Progress and Trends Report 2025": {
        "question": "which country has most Gen Ai patents and which country has most total funding raised by AI start-ups?",
        "file_paths": ["samples/Digital Progress and Trends Report 2025, Strengthening AI Foundations.pdf"]
    },
    "NSF Invention 2024 (INV-2)": {
        "question": "According to Figure INV-2, which foreign inventor location surpassed South Korea in patents granted in 2020?",
        "file_paths": ["samples/NSF_Invention_Knowledge_Transfer_Innovation_2024.pdf"]
    },
    "NSF KTI Industries 2022 (KTI-1)": {
        "question": "According to Figure KTI-1, which KTI industry increased its share of total KTI value added by more than 12 percentage points?",
        "file_paths": ["samples/NSF_Production_Trade_KTI_Industries_2022.pdf"]
    },
    "NSF R&D Trends 2022 (RD-1)": {
        "question": "According to Figure RD-1, what were total U.S. R&D amounts in 2019 and the estimated total for 2020?",
        "file_paths": ["samples/NSF_RnD_Trends_International_Comparisons_2022.pdf"]
    },
    "NSF STEM Labor Force 2024 (LBR-1)": {
        "question": "According to Figure LBR-1, what share of workers with a bachelor's degree or higher are in S&E or S&E-related occupations?",
        "file_paths": ["samples/NSF_STEM_Labor_Force_2024.pdf"]
    },
    "NSF R&D Trends 2024 (RD-1)": {
        "question": "According to Figure RD-1, what is total U.S. R&D in 2022 and which performer category is largest?",
        "file_paths": ["samples/NSF_RnD_Trends_International_Comparisons_2024.pdf"]
    },
    "NSF KTI Industries 2024 (KTI-1)": {
        "question": "According to Figure KTI-1, which KTI industry has the largest U.S. nominal value added in 2022?",
        "file_paths": ["samples/NSF_Production_Trade_KTI_Industries_2024.pdf"]
    }
}


def format_chat_history(history: List[Dict]) -> str:
    """Format chat history as markdown for display."""
    if not history:
        return "*No conversation history yet. Ask a question to get started!*"

    formatted = []
    for i, entry in enumerate(history, 1):
        timestamp = entry.get("timestamp", "")
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        confidence = entry.get("confidence", "N/A")
    
        formatted.append(f"""
---
### 💬 Q{i} ({timestamp})
**Question:** {question}

**Answer:** {answer}

*Confidence: {confidence}*
""")

    return "\n".join(formatted)


def _normalize_chat_messages(chat_history: Any) -> List[Dict[str, str]]:
    """
    Normalize chatbot history into Gradio 'messages' format:
    [{"role": "user|assistant", "content": "..."}]
    Supports legacy tuple/list pairs for backward compatibility.
    """
    normalized: List[Dict[str, str]] = []
    if not chat_history:
        return normalized

    for item in chat_history:
        # New format
        if isinstance(item, dict) and "role" in item and "content" in item:
            normalized.append(
                {
                    "role": str(item["role"]),
                    "content": str(item.get("content", "")),
                }
            )
            continue

        # Legacy format: (user, assistant)
        if isinstance(item, (list, tuple)) and len(item) == 2:
            user_msg, assistant_msg = item
            if user_msg:
                normalized.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                normalized.append({"role": "assistant", "content": str(assistant_msg)})

    return normalized


def _append_chat_exchange(chat_history: List[Dict[str, str]], question_text: str, answer_text: str) -> List[Dict[str, str]]:
    """Append one user+assistant exchange in messages format."""
    if question_text and question_text.strip():
        chat_history.append({"role": "user", "content": question_text})
    chat_history.append({"role": "assistant", "content": answer_text})
    return chat_history


def format_document_context(documents: List, question: str = "") -> str:
    """Format retrieved documents with annotation highlighting."""
    if not documents:
        return "*No documents retrieved yet.*"

    formatted = [f"### 📚 Retrieved Context ({len(documents)} chunks)\n"]

    # Extract key terms from question for highlighting
    key_terms = []
    if question:
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or', 'what', 'how', 'why', 'when', 'where', 'which'}
        key_terms = [word.lower() for word in question.split() if word.lower() not in stopwords and len(word) > 2]

    for i, doc in enumerate(documents[:5], 1):
        content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        source = doc.metadata.get('source', 'Unknown') if hasattr(doc, 'metadata') else 'Unknown'
    
        # Truncate long content
        if len(content) > 500:
            content = content[:500] + "..."

        # Highlight key terms
        highlighted_content = content
        for term in key_terms[:5]:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            highlighted_content = pattern.sub(f"**{term}**", highlighted_content)
    
        formatted.append(f"""
<details>
<summary>📄 Chunk {i} - {os.path.basename(source)}</summary>

{highlighted_content}

</details>
""")

    if len(documents) > 5:
        formatted.append(f"\n*... and {len(documents) - 5} more chunks*")

    return "\n".join(formatted)


def extract_referenced_chart_images(documents: List, max_items: int = 3) -> List[Tuple[str, str]]:
    """Return unique chart/page image paths (with captions) from retrieved docs."""
    gallery_items: List[Tuple[str, str]] = []
    seen_paths = set()

    for doc in documents or []:
        metadata = getattr(doc, "metadata", {}) or {}
        image_path = metadata.get("chart_image_path")
        if not image_path:
            continue

        abs_path = os.path.abspath(str(image_path))
        if abs_path in seen_paths or not os.path.exists(abs_path):
            continue

        seen_paths.add(abs_path)
        page = metadata.get("page", "?")
        doc_type = str(metadata.get("type", "chart")).lower()
        source_raw = str(metadata.get("source", "Unknown"))
        source_name = os.path.basename(source_raw.split("::", 1)[0]) or source_raw
        caption = f"{source_name} • page {page} • {doc_type}"
        gallery_items.append((abs_path, caption))

        if len(gallery_items) >= max_items:
            break

    return gallery_items


def question_terms(question: str) -> List[str]:
    """Extract normalized, non-trivial lexical terms from a question."""
    if not question:
        return []
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or",
        "what", "how", "why", "when", "where", "which", "who", "whom", "whose", "with", "from", "by",
        "about", "into", "over", "after", "before", "between", "during", "under", "above", "below",
        "most", "likely", "does", "did", "can", "could", "would", "should",
    }
    tokens = re.findall(r"[a-zA-Z0-9']+", question.lower())
    return [token for token in tokens if len(token) >= 3 and token not in stopwords]


def rank_chart_chunks_by_question(question: str, chart_chunks: List) -> List:
    """Rank chart/page chunks by lexical overlap with the user question.

    Sort key: (overlap_ratio, overlap_count) — ratio = matched_terms / total_terms.
    This eliminates the previous length-bias where a longer chunk with identical
    raw overlap always beat a shorter, more focused chunk.
    """
    terms = question_terms(question)
    if not terms:
        return []

    n_terms = len(terms)
    scored = []
    for doc in chart_chunks or []:
        metadata = getattr(doc, "metadata", {}) or {}
        if not metadata.get("chart_image_path"):
            continue
        content = str(getattr(doc, "page_content", "") or "").lower()
        if not content:
            continue
        overlap = sum(1 for term in terms if term in content)
        if overlap <= 0:
            continue
        # Option C fix: sort by ratio first so a dense match beats a long chunk
        # with the same raw count; use overlap_count as tiebreaker.
        ratio = overlap / n_terms
        scored.append((ratio, overlap, doc))

    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [doc for _, _, doc in scored]


def extract_related_chart_images(question: str, chart_chunks: List, max_items: int = 3) -> List[Tuple[str, str]]:
    """Fallback chart extraction from question-related chart chunks."""
    ranked_chunks = rank_chart_chunks_by_question(question=question, chart_chunks=chart_chunks)
    return extract_referenced_chart_images(ranked_chunks, max_items=max_items)


def _normalize_page(raw_page):
    """Coerce a page metadata value to a plain int, or None if not parseable.

    Some loaders store page as a list ([37]), string ('37'), or int (37).
    Using raw values as dict keys or set members blows up on lists ('unhashable
    type: list') — this helper normalises all cases to a single int.
    """
    if raw_page is None:
        return None
    if isinstance(raw_page, list):
        raw_page = raw_page[0] if raw_page else None
        if raw_page is None:
            return None
    try:
        return int(raw_page)
    except (TypeError, ValueError):
        return None


def _chart_chunks_for_pages(retrieved_docs: List, chart_chunks: List, question: str = "") -> List:
    """Option A: Return chart_chunks whose page number matches any retrieved_doc's page.

    This bridges the gap where text-chunks and chart-image-chunks live separately:
    the retriever surfaces a text chunk from page 37, so we look up the chart chunk
    that was extracted from that same page 37 and return it for display.

    Ordering: chunks are returned in the order their page number first appeared in
    retrieved_docs (i.e. most-relevant page first).

    Tolerance: if no exact-page chart chunk exists, expands search to adjacent pages
    (±1) to handle charts whose image was captured on a slightly different page
    than the text description (common in multi-column or header-overflow layouts).
    Adjacent-page matches are validated for content relevance: at least 1 question
    term must appear in the chart's page_content, preventing unrelated charts from
    being shown (e.g. 'Traffic per internet user' when the question is about 'GenAI patents').
    """
    # Collect ordered, deduplicated page numbers from retrieved docs.
    # _normalize_page() handles list / string / int variants so set.add() never
    # raises "unhashable type: list".
    retrieved_pages: List[int] = []
    seen_pages: set = set()
    for doc in retrieved_docs or []:
        raw = (getattr(doc, "metadata", {}) or {}).get("page")
        page = _normalize_page(raw)
        if page is not None and page not in seen_pages:
            seen_pages.add(page)
            retrieved_pages.append(page)

    if not retrieved_pages:
        return []

    # Build page → [chart_chunk, ...] index (keys are always plain ints).
    page_to_charts: dict = {}
    for doc in chart_chunks or []:
        metadata = getattr(doc, "metadata", {}) or {}
        if not metadata.get("chart_image_path"):
            continue
        page = _normalize_page(metadata.get("page"))
        if page is None:
            continue
        page_to_charts.setdefault(page, []).append(doc)

    logger.debug(
        "[CHART_PAGES] retrieved_pages=%s chart_pages=%s",
        retrieved_pages[:10],
        sorted(page_to_charts.keys())[:20],
    )

    # Pass 1: exact match — same page number
    matched: List = []
    seen_matched: set = set()
    for page in retrieved_pages:
        for chunk in page_to_charts.get(page, []):
            cid = id(chunk)
            if cid not in seen_matched:
                seen_matched.add(cid)
                matched.append(chunk)

    if matched:
        return matched

    # Pass 2: adjacent-page tolerance (±1) — handles chart captured on
    # a neighbouring page to the text that describes it.
    # Content validation: require ≥2 question-term overlap AND ≥20% ratio
    # to prevent generic single-term matches (e.g. "country") from showing
    # topically unrelated charts from adjacent pages.
    _ADJ_MIN_TERMS = 2          # absolute minimum matching terms
    _ADJ_MIN_RATIO = 0.20       # minimum fraction of question terms matched
    q_terms = question_terms(question) if question else []
    for page in retrieved_pages:
        for adj in (page - 1, page + 1):
            for chunk in page_to_charts.get(adj, []):
                cid = id(chunk)
                if cid in seen_matched:
                    continue
                # If we have question terms, validate the chart content is related
                if q_terms:
                    content = str(getattr(chunk, "page_content", "") or "").lower()
                    hits = [term for term in q_terms if term in content]
                    overlap = len(hits)
                    ratio = overlap / len(q_terms)
                    if overlap < min(_ADJ_MIN_TERMS, len(q_terms)) or ratio < _ADJ_MIN_RATIO:
                        logger.info(
                            "[CHART_PAGES] adjacent page %d chart REJECTED "
                            "(%d/%d terms=%.0f%%, need ≥%d and ≥%.0f%%; matched=%s)",
                            adj, overlap, len(q_terms), ratio * 100,
                            _ADJ_MIN_TERMS, _ADJ_MIN_RATIO * 100, hits[:5],
                        )
                        continue
                    logger.info(
                        "[CHART_PAGES] adjacent page %d chart ACCEPTED "
                        "(%d/%d terms=%.0f%%; matched=%s)",
                        adj, overlap, len(q_terms), ratio * 100, hits[:5],
                    )
                seen_matched.add(cid)
                matched.append(chunk)

    logger.info(
        "[CHART_PAGES] exact=0 adjacent_matched=%d for retrieved_pages=%s",
        len(matched), retrieved_pages[:10],
    )
    return matched


def _chart_chunks_from_answer(answer_text: str, chart_chunks: List) -> List:
    """Tier 0 helper: extract explicit page numbers from the LLM answer and
    return chart_chunks that match those pages.

    Why this beats page-match: the LLM often says 'page 37' in the answer even
    when the text chunk for page 37 scores below the top-K cut-off.  Parsing the
    answer gives us a ground-truth page number directly.

    Pattern: 'page 37', 'page 37)', '(page 37', 'Page 37' — all captured.
    Figure refs like 'Figure 1.11' are also collected and used to cross-check
    figure titles in page_content where possible (best-effort).
    """
    if not answer_text:
        return []

    # Extract all bare page numbers from the answer
    page_refs = set()
    for m in re.finditer(r'[Pp]age\s+(\d+)', answer_text):
        try:
            page_refs.add(int(m.group(1)))
        except ValueError:
            pass

    # Extract figure references (e.g. "Figure RD-1", "Figure 1.11", "Table RD-1")
    # These will be matched against chart chunk page_content when page_refs is empty.
    figure_refs: list = []
    for m in re.finditer(
        r'(?:Figure|Table|Chart|Exhibit)\s+([\w][\w.-]*\d[\w.-]*)',
        answer_text,
        re.IGNORECASE,
    ):
        figure_refs.append(m.group(0).strip())  # full match e.g. "Figure RD-1"

    if not page_refs and not figure_refs:
        return []

    # Build page → chart_chunk index (normalised ints, same as _chart_chunks_for_pages)
    page_to_charts: dict = {}
    for doc in chart_chunks or []:
        metadata = getattr(doc, "metadata", {}) or {}
        if not metadata.get("chart_image_path"):
            continue
        page = _normalize_page(metadata.get("page"))
        if page is None:
            continue
        page_to_charts.setdefault(page, []).append(doc)

    logger.info(
        "[CHART_ANSWER] answer_page_refs=%s figure_refs=%s chart_pages_available=%s",
        sorted(page_refs),
        figure_refs[:5],
        sorted(page_to_charts.keys())[:20],
    )

    # Return chart chunks for exactly the pages the answer cited
    matched: List = []
    for page in sorted(page_refs):          # ascending page order
        matched.extend(page_to_charts.get(page, []))

    if matched:
        return matched

    # ── Figure-name matching ──────────────────────────────────────────
    # When the answer cites "Figure RD-1" or "Table RD-1" but no bare
    # "page N", scan chart chunk page_content for that figure name.
    # This handles NSF/OECD-style naming where figures use alphanumeric
    # IDs (RD-1, A-3, etc.) instead of sequential page numbers.
    if figure_refs and not matched:
        fig_matched: List = []
        seen_fig: set = set()
        # Flatten all chart chunks for content search
        all_chart_list = [
            chunk
            for chunks_for_page in page_to_charts.values()
            for chunk in chunks_for_page
        ]
        for fig_ref in figure_refs:
            fig_lower = fig_ref.lower()
            for chunk in all_chart_list:
                cid = id(chunk)
                if cid in seen_fig:
                    continue
                content = str(getattr(chunk, "page_content", "") or "").lower()
                if fig_lower in content:
                    seen_fig.add(cid)
                    fig_matched.append(chunk)
                    pg = _normalize_page((getattr(chunk, "metadata", {}) or {}).get("page"))
                    logger.info(
                        "[CHART_ANSWER] figure-name match: '%s' found on page %s",
                        fig_ref, pg,
                    )
        if fig_matched:
            return fig_matched

    # No exact match — try adjacent pages (±1) of each answer-cited page.
    # Validate against the ANSWER text (not just question terms) since
    # the answer provides more specific topical context.
    answer_terms = question_terms(answer_text)
    _ANS_ADJ_MIN_TERMS = 3          # require ≥3 answer-terms in chart content
    _ANS_ADJ_MIN_RATIO = 0.10       # answer text is long, so lower ratio is ok
    seen: set = set()
    adj_matched: List = []
    for page in sorted(page_refs):
        for adj in (page - 1, page + 1):
            for chunk in page_to_charts.get(adj, []):
                cid = id(chunk)
                if cid in seen:
                    continue
                content = str(getattr(chunk, "page_content", "") or "").lower()
                if not content:
                    continue
                if answer_terms:
                    hits = [t for t in answer_terms if t in content]
                    overlap = len(hits)
                    ratio = overlap / len(answer_terms) if answer_terms else 0
                    if overlap < _ANS_ADJ_MIN_TERMS or ratio < _ANS_ADJ_MIN_RATIO:
                        logger.info(
                            "[CHART_ANSWER] adjacent page %d REJECTED "
                            "(%d/%d answer-terms=%.0f%%; need ≥%d and ≥%.0f%%)",
                            adj, overlap, len(answer_terms), ratio * 100,
                            _ANS_ADJ_MIN_TERMS, _ANS_ADJ_MIN_RATIO * 100,
                        )
                        continue
                    logger.info(
                        "[CHART_ANSWER] adjacent page %d ACCEPTED "
                        "(%d/%d answer-terms=%.0f%%; hits=%s)",
                        adj, overlap, len(answer_terms), ratio * 100, hits[:6],
                    )
                seen.add(cid)
                adj_matched.append(chunk)

    if adj_matched:
        logger.info(
            "[CHART_ANSWER] exact=0 adjacent_matched=%d for answer_pages=%s",
            len(adj_matched), sorted(page_refs),
        )
    return adj_matched


def build_chart_gallery_payload(
    question: str,
    retrieved_docs: List,
    chart_chunks: List,
    max_items: int = 3,
    answer_text: str = "",
) -> Tuple[List[Tuple[str, str]], str, str]:
    """
    Build gallery payload using a four-tier evidence cascade:

    0. Answer-guided — parse LLM answer for explicit 'page N' references → exact chart lookup
    1. Direct        — retrieved_doc itself carries chart_image_path metadata
    2. Top-page-match — chart_chunk shares page with the TOP-3 retrieved text chunks only
                        (prevents showing unrelated charts from lower-ranked pages)
    3. Lexical       — ratio-ranked term-overlap fallback across all chart_chunks

    Returns: (gallery_items, note_text, mode)
    mode ∈ {'answer_guided', 'direct', 'page_match', 'fallback', 'none'}.
    """
    # Tier 0: answer-guided — most precise; uses what the LLM explicitly cited
    if answer_text:
        answer_chunks = _chart_chunks_from_answer(answer_text, chart_chunks)
        answer_items = extract_referenced_chart_images(answer_chunks, max_items=max_items)
        if answer_items:
            return answer_items, "", "answer_guided"
        logger.info(
            "[CHART_SELECT] tier0_answer_guided: no match (parsed_pages=%s, chart_pages=%s)",
            sorted({_normalize_page((getattr(c, 'metadata', {}) or {}).get('page'))
                    for c in (answer_chunks or [])} - {None}),
            sorted({_normalize_page((getattr(c, 'metadata', {}) or {}).get('page'))
                    for c in (chart_chunks or []) if (getattr(c, 'metadata', {}) or {}).get('chart_image_path')} - {None}),
        )

    # Tier 1: direct evidence — retrieved chunk carries its own image
    direct_items = extract_referenced_chart_images(retrieved_docs, max_items=max_items)
    if direct_items:
        return direct_items, "", "direct"

    # Tier 2: page-number cross-reference restricted to TOP-3 retrieved docs only.
    # Using all retrieved docs caused irrelevant charts (e.g. page 21, 29) to appear
    # when the answer was about a page with no chart image (e.g. page 37).
    # NOTE: Relies on retrieved_docs being in retrieval-relevance order.
    # The orchestrator preserves insertion order from retriever.invoke() → AgentState.
    top_retrieved = (retrieved_docs or [])[:3]
    top_pages = [_normalize_page((getattr(d, 'metadata', {}) or {}).get('page')) for d in top_retrieved]
    page_matched_chunks = _chart_chunks_for_pages(top_retrieved, chart_chunks, question=question)
    page_match_items = extract_referenced_chart_images(page_matched_chunks, max_items=max_items)
    if page_match_items:
        matched_pages = [_normalize_page((getattr(c, 'metadata', {}) or {}).get('page')) for c in page_matched_chunks]
        logger.info(
            "[CHART_SELECT] tier2_page_match: top_retrieved_pages=%s matched_chart_pages=%s",
            top_pages, matched_pages,
        )
        return page_match_items, "", "page_match"
    logger.info(
        "[CHART_SELECT] tier2_page_match: no match (top_retrieved_pages=%s)",
        top_pages,
    )

    # Tier 3: lexical fallback (ratio-ranked, length-bias removed by Option C)
    fallback_items = extract_related_chart_images(question=question, chart_chunks=chart_chunks, max_items=max_items)
    if fallback_items:
        return (
            fallback_items,
            "No direct chart evidence in top retrieved chunks; showing related chart pages.",
            "fallback",
        )

    return (
        [],
        "No chart evidence retrieved for this question. Try asking about a specific figure/chart/table (e.g., 'What does Figure 7 show?').",
        "none",
    )


def _get_file_hashes(uploaded_files: List) -> frozenset:
    """Generate SHA-256 hashes for uploaded files."""
    hashes = set()
    for file in uploaded_files:
        with open(file.name, "rb") as f:
            hashes.add(hashlib.sha256(f.read()).hexdigest())
    return frozenset(hashes)


def _find_open_port(start_port: int, max_attempts: int = 20) -> int:
    """Find an available TCP port starting from start_port."""
    port = start_port
    for _ in range(max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    raise RuntimeError(f"Could not find an open port starting at {start_port}")


def _ensure_hfhub_hffolder_compat():
    """
    Shim for Gradio <5.7.1 with huggingface_hub >=1.0.
    """
    import huggingface_hub
    if hasattr(huggingface_hub, "HfFolder"):
        return
    try:
        from huggingface_hub.utils import get_token
    except Exception:
        return
    class HfFolder:
        @staticmethod
        def get_token():
            return get_token()
    huggingface_hub.HfFolder = HfFolder


def _setup_gradio_shim():
    """Shim Gradio's JSON schema conversion to tolerate boolean additionalProperties values."""
    from gradio_client import utils as grc_utils
    _orig_json_schema_to_python_type = grc_utils._json_schema_to_python_type
    def _json_schema_to_python_type_safe(schema, defs=None):
        if isinstance(schema, bool):
            return "Any" if schema else "Never"
        return _orig_json_schema_to_python_type(schema, defs)
    grc_utils._json_schema_to_python_type = _json_schema_to_python_type_safe


def main():
    # Configure logging explicitly at runtime (no import-time side effects).
    configure_logging()
    initialize_telemetry()

    _ensure_hfhub_hffolder_compat()  # must run before importing gradio
    import gradio as gr
    _setup_gradio_shim()

    logger.info("=" * 60)
    logger.info("Starting SmartDoc AI application...")
    logger.info("=" * 60)

    # Initialize components
    processor = DocumentProcessor()
    retriever_indexer = RetrieverBuilder()
    orchestrator = AgentWorkflow()
    analytics_logger = get_analytics_logger()

    logger.info("All components initialized successfully")

    # CSS styling - Clean, accessible light theme with professional colors
    css = """
    /* Global styling - Light, clean background */
    .gradio-container {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Title styles - Dark text for readability */
    .app-title { 
        font-size: 2.2em !important; 
        text-align: center !important; 
        color: #1e293b !important;
        font-weight: 700 !important;
        margin-bottom: 8px !important;
    }
    .app-subtitle { 
        font-size: 1.1em !important; 
        text-align: center !important; 
        color: #0369a1 !important;
        font-weight: 500 !important;
    }
    .app-description { 
        text-align: center; 
        color: #475569 !important;
        font-size: 0.95em !important;
        line-height: 1.6 !important;
    }

    /* Section headers */
    .section-header {
        color: #1e293b !important;
        font-weight: 600 !important;
        border-bottom: 2px solid #0ea5e9 !important;
        padding-bottom: 8px !important;
        margin-bottom: 16px !important;
    }

    /* Chat history panel - Clean white card with more height */
    .chat-history { 
        min-height: 500px;
        max-height: 600px; 
        overflow-y: auto; 
        border: 1px solid #cbd5e1; 
        border-radius: 12px; 
        padding: 20px; 
        background: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        color: #334155 !important;
    }
    #chat-history {
        min-height: 120px !important;
        max-height: none !important;
        height: auto !important;
    }
    .chat-history h3 {
        color: #0f172a !important;
    }
    .chat-history strong {
        color: #1e293b !important;
    }

    /* Document context panel */
    .doc-context { 
        max-height: 380px; 
        overflow-y: auto; 
        border: 1px solid #cbd5e1; 
        border-radius: 12px; 
        padding: 20px; 
        background: #ffffff;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        color: #334155 !important;
    }
    .doc-context details { 
        margin-bottom: 12px; 
        padding: 14px; 
        background: #f1f5f9; 
        border-radius: 8px; 
        border-left: 4px solid #0ea5e9; 
    }
    .doc-context summary { 
        cursor: pointer; 
        font-weight: 600; 
        color: #0369a1 !important;
    }
    .doc-context p, .doc-context span {
        color: #475569 !important;
    }

    /* Answer box - Success green accent, auto-height */
    .answer-box > div:nth-child(2) { 
        border-left: 4px solid #10b981 !important; 
        padding: 16px 16px 16px 20px !important; 
        background: #f0fdf4 !important;
        border-radius: 8px !important;
        min-height: 100px;
        color: #166534 !important;
    }
    .answer-box p, .answer-box li, .answer-box span {
        color: #166534 !important;
    }
    .answer-box strong {
        color: #14532d !important;
    }
    .answer-box h1, .answer-box h2, .answer-box h3, .answer-box h4 {
        color: #15803d !important;
    }
    .answer-box code {
        background: #dcfce7 !important;
        color: #166534 !important;
        padding: 2px 6px !important;
        border-radius: 4px !important;
    }
    .answer-box pre {
        background: #dcfce7 !important;
        padding: 12px !important;
        border-radius: 6px !important;
        overflow-x: auto !important;
    }

    /* Verification box - Blue accent */
    .verification-box >  div:nth-child(2) {
        border-left: 4px solid #0ea5e9 !important;
        padding: 16px 16px 16px 20px !important;
        background: #f0f9ff !important;
        border-radius: 8px !important;
        min-height: 80px;
        color: #0369a1 !important;
    }
    .verification-box p, .verification-box li, .verification-box span {
        color: #0c4a6e !important;
    }
    .verification-box strong {
        color: #075985 !important;
    }

    /* Stats panel - Professional blue gradient */
    .stats-panel { 
        background: linear-gradient(135deg, #0369a1 0%, #0284c7 50%, #0ea5e9 100%) !important; 
        color: #ffffff !important; 
        padding: 20px !important; 
        border-radius: 12px !important; 
        text-align: center;
        box-shadow: 0 4px 14px rgba(3, 105, 161, 0.3);
    }
    .stats-panel strong {
        color: #ffffff !important;
    }

    /* Info panel */
    .info-panel {
        background: #eff6ff !important;
        border: 1px solid #bfdbfe !important;
        border-radius: 8px !important;
        padding: 12px !important;
        color: #1e40af !important;
    }

    /* Form elements */
    .gr-input, .gr-textbox textarea {
        background: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 8px !important;
        color: #1e293b !important;
    }
    .gr-input:focus, .gr-textbox textarea:focus {
        border-color: #0ea5e9 !important;
        box-shadow: 0 0 0 3px rgba(14, 165, 233, 0.1) !important;
    }

    /* Labels */
    label {
        color: #374151 !important;
        font-weight: 500 !important;
    }   

    /* Dropdown - High contrast with darker background for visibility */
    .gr-dropdown, 
    [data-testid="dropdown"],
    .svelte-dropdown,dropdownExample
    div[class*="dropdown"] {
        background: #e0e7ff !important;
        color: #1e293b !important;
        border: 2px solid #1e40af !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 8px rgba(30, 64, 175, 0.2) !important;
    }
    .gr-dropdown:hover,
    [data-testid="dropdown"]:hover {
        background: #c7d2fe !important;
        border-color: #1d4ed8 !important;
        box-shadow: 0 4px 12px rgba(30, 64, 175, 0.3) !important;
    }
    .gr-dropdown select, 
    .gr-dropdown input,
    [data-testid="dropdown"] input {
        color: #1e293b !important;
        background: transparent !important;
        font-weight: 500 !important;
    }

    /* Dropdown container and options */
    [data-testid="dropdown"] span, 
    .dropdown-container span,
    div[class*="dropdown"] span {
        color: #1e293b !important;
        font-weight: 500 !important;
    }

    /* Dropdown list options */
    .gr-dropdown ul, 
    .dropdown-options,
    ul[class*="dropdown"] {
        background: #ffffff !important;
        border: 2px solid #1e40af !important;
        border-radius: 8px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15) !important;
    }
    .gr-dropdown li, 
    .dropdown-options li,
    ul[class*="dropdown"] li {
        color: #1e293b !important;
        padding: 10px 14px !important;
    }
    .gr-dropdown li:hover,
    ul[class*="dropdown"] li:hover {
        background: #c7d2fe !important;
        color: #1e40af !important;
    }

    /* Dropdown label */
    .gr-dropdown label,
    [data-testid="dropdown"] label {
        color: #1e40af !important;
        font-weight: 600 !important;
    }

    /* Tabs - Clean styling */
    .tab-nav {
        border-bottom: 2px solid #e2e8f0 !important;
    }
    .tab-nav button {
        color: #64748b !important;
        font-weight: 500 !important;
        padding: 12px 20px !important;
        border: none !important;
        background: transparent !important;
    }
    .tab-nav button.selected {
        color: #0369a1 !important;
        border-bottom: 3px solid #0369a1 !important;
        font-weight: 600 !important;
    }

    /* Markdown text */
    .prose, .markdown-text {
        color: #334155 !important;
    }
    .prose h1, .prose h2, .prose h3, 
    .markdown-text h1, .markdown-text h2, .markdown-text h3 {
        color: #1e293b !important;
    }
    .prose strong, .markdown-text strong {
        color: #0f172a !important;
    }

    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb {
        background: #94a3b8;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #64748b;
    }

    button.secondary {
        background: #1e40af !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 6px rgba(30, 64, 175, 0.3) !important;
        padding: 12px 20px !important;
        min-height: 44px !important;
    }
    button.secondary:hover {
        background: #1d4ed8 !important;
        box-shadow: 0 4px 10px rgba(30, 64, 175, 0.4) !important;
    }         

    /* Left side input boxes with borders */
    .left-panel-box {
        background: #fafafa !important;
        border: 2px solid #94a3b8 !important;
        border-radius: 10px !important;
        padding: 14px !important;
        margin-bottom: 8px !important;
    }
    .left-panel-box:hover {
        border-color: #64748b !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1) !important;
    }

    /* File upload box with border */
    .file-upload-box {
        background: #f8fafc !important;
        border: 2px dashed #64748b !important;
        border-radius: 10px !important;
        padding: 14px !important;
    }
    .file-upload-box:hover {
        border-color: #0369a1 !important;
        border-style: solid !important;
        background: #f0f9ff !important;
    }

    /* Question input box with border */
    .question-box {
        background: #fffbeb !important;
        border: 2px solid #f59e0b !important;
        border-radius: 10px !important;
        padding: 14px !important;
    }
    .question-box:hover {
        border-color: #d97706 !important;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.2) !important;
    }

    /* Dropdown Example - Beige background on 3rd parent container */
    .dropdownExample {
        background: #f5f5dc !important;
        padding: 16px !important;
        border-radius: 8px !important;
        border: 2px solid #d1d5db !important;
        margin-bottom: 16px !important;
    }
    """
    js = r'''
const uploadMessages = [
  "Crunching your documents...",
  "Warming up the AI...",
  "Extracting knowledge...",
  "Scanning for insights...",
  "Preparing your data...",
  "Looking for answers...",
  "Analyzing file structure...",
  "Reading your files...",
  "Indexing content...",
  "Almost ready..."
];

let msgInterval = null;
let timerInterval = null;
let startMs = 0;
let lastMsg = null;

function root() {
  return document.getElementById("processing-message");
}
function isVisible(el) {
  return !!(el && (el.offsetWidth || el.offsetHeight || el.getClientRects().length));
}
function pickMsg() {
  if (uploadMessages.length === 0) return "";
  if (uploadMessages.length === 1) return uploadMessages[0];
  let m;
  do { m = uploadMessages[Math.floor(Math.random() * uploadMessages.length)]; }
  while (m === lastMsg);
  lastMsg = m;
  return m;
}
function getMsgSpan() {
  const r = root();
  return r ? r.querySelector("#processing-msg") : null;
}
function getTimerSpan() {
  const r = root();
  return r ? r.querySelector("#processing-timer") : null;
}
function setMsg(t) {
  const s = getMsgSpan();
  if (s) s.textContent = t;
}
function fmtElapsed() {
  return ((Date.now() - startMs) / 1000).toFixed(1) + "s elapsed";
}

function start() {
  if (msgInterval || timerInterval) return;
  startMs = Date.now();
  setMsg(pickMsg());

  msgInterval = setInterval(() => setMsg(pickMsg()), 2000);

  const t = getTimerSpan();
  if (t) {
    t.textContent = fmtElapsed();
    timerInterval = setInterval(() => { t.textContent = fmtElapsed(); }, 200);
  }
}

function stop() {
  if (msgInterval) { clearInterval(msgInterval); msgInterval = null; }
  if (timerInterval) { clearInterval(timerInterval); timerInterval = null; }
  const t = getTimerSpan();
  if (t) t.textContent = "";
}

function tick() {
  const r = root();
  if (isVisible(r)) start();
  else stop();
}

// Observe rerenders / visibility changes
const obs = new MutationObserver(tick);
obs.observe(document.body, { subtree: true, childList: true, attributes: true });

window.addEventListener("load", tick);
setInterval(tick, 500);
'''

    # Launch server - Compatible with both local and Hugging Face Spaces
    # HF Spaces sets SPACE_ID environment variable
    is_hf_space = os.environ.get("SPACE_ID") is not None

    with gr.Blocks(title="SmartDoc AI", theme=gr.themes.Soft(), css=css) as demo:
        gr.Markdown("### SmartDoc AI - Document Q&A", elem_classes="app-title")
        gr.Markdown("Upload your documents and ask questions. Answers will appear below, just like a chat.", elem_classes="app-description")
        gr.Markdown("---")

        llm_provider = parameters.LLM_PROVIDER.lower()
        has_openai_env_key = bool(parameters.OPENAI_API_KEY and parameters.OPENAI_API_KEY.strip())
        has_azure_env_config = bool(
            parameters.AZURE_OPENAI_API_KEY
            and parameters.AZURE_OPENAI_ENDPOINT
            and parameters.AZURE_OPENAI_DEPLOYMENT
        )
        has_google_env_key = bool(parameters.GOOGLE_API_KEY and parameters.GOOGLE_API_KEY.strip())

        show_openai_key_input = llm_provider == "openai" and not has_openai_env_key

        if llm_provider == "openai":
            api_key_status_default = (
                "🟢 Using OpenAI API key from environment"
                if has_openai_env_key
                else "🟡 OPENAI_API_KEY not found in environment. Enter a key below."
            )
        elif llm_provider == "azure":
            api_key_status_default = (
                "🟢 Using Azure OpenAI credentials from environment"
                if has_azure_env_config
                else "🔴 Azure OpenAI credentials are incomplete in environment"
            )
        else:
            api_key_status_default = (
                "🟢 Using Google API key from environment"
                if has_google_env_key
                else "🔴 GOOGLE_API_KEY not found in environment"
            )

        if show_openai_key_input:
            with gr.Row():
                openai_api_key = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="sk-...",
                    type="password",
                    info="Only needed when OPENAI_API_KEY is not set in .env"
                )
                api_key_status = gr.Markdown(api_key_status_default, elem_classes="info-panel")

            def validate_openai_key(api_key):
                """Validate OpenAI API key format and connectivity."""
                if not api_key or not api_key.strip():
                    return "🔴 No key provided"

                if not api_key.startswith("sk-"):
                    return "🔴 Invalid format (must start with sk-)"

                # Set it in environment for the LLM factory to use.
                os.environ["OPENAI_API_KEY"] = api_key
                parameters.OPENAI_API_KEY = api_key

                # Validate by trying to initialize the configured LLM.
                try:
                    from core.llm_factory import get_chat_llm
                    get_chat_llm(
                        role="test",
                        model_name=parameters.LLM_MODEL_NAME,
                        max_output_tokens=parameters.LLM_ROUTER_MAX_OUTPUT_TOKENS,
                    )
                    return "🟢 Authenticated with OpenAI"
                except Exception as e:
                    return f"🔴 Authentication failed: {str(e)[:100]}"

            openai_api_key.change(
                validate_openai_key,
                inputs=[openai_api_key],
                outputs=[api_key_status]
            )
        else:
            gr.Markdown(api_key_status_default, elem_classes="info-panel")

        gr.Markdown("---")

        # Examples dropdown - visible for both local and HF Spaces
        example_dropdown = gr.Dropdown(
            label="Quick Start - Choose an Example",
            choices=list(EXAMPLES.keys()),
            value=None,
            info="Select a pre-loaded example to try"
        )
        loaded_file_info = gr.Markdown("", elem_classes="info-panel", visible=False)

        files = gr.Files(label="Upload your files", file_types=definitions.ALLOWED_TYPES)
        question = gr.Textbox(label="Ask a question", lines=2, placeholder="Type your question here...")
        chat = gr.Chatbot(label="Answers", elem_id="chat-history")
        referenced_charts = gr.Gallery(
            label="Referenced extracted charts/pages",
            visible=False,
            columns=3,
            show_label=True,
            height="auto",
        )
        chart_gallery_note = gr.Markdown("", elem_classes="info-panel", visible=False)
        submit_btn = gr.Button("Get Answer", variant="primary")
        processing_message = gr.HTML("", elem_id="processing-message", visible=False)
        doc_context_display = gr.Markdown("*Submit a question to see which document sections were referenced*", elem_classes="doc-context", visible=False)
        refresh_context_btn = gr.Button("Refresh Sources", variant="secondary", visible=False)
        with gr.Tab("Context"):
            pass  # No .render() calls here; components are already defined and used in outputs

        session_state = gr.State({
            "file_hashes": frozenset(),
            "retriever": None,
            "chart_chunks": [],
            "chat_history": [],
            "last_documents": [],
            "total_questions": 0,
            "session_start": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

        def process_question(question_text, uploaded_files, chat_history, request: gr.Request):
            run_started_at = time.perf_counter()
            metric_attrs = {
                "llm_provider": parameters.LLM_PROVIDER,
                "embedding_provider": parameters.EMBEDDING_PROVIDER,
                "has_uploads": bool(uploaded_files),
            }
            # Get user IP for logging and rate limiting
            ip_address = getattr(request.client, "host", "unknown")
            try:
                rate_limit(request)
            except Exception:
                metric_attrs["failure_reason"] = "rate_limit"
                record_request_metrics(
                    duration_s=time.perf_counter() - run_started_at,
                    success=False,
                    extra_attributes=metric_attrs,
                )
                raise
            chat_history = _normalize_chat_messages(chat_history)
            
            # Get file metadata for logging
            file_types = [Path(f.name).suffix.lower() for f in uploaded_files] if uploaded_files else []
            
            # Log the question attempt
            analytics_logger.log_question(
                ip_address=ip_address,
                question=question_text,
                num_files=len(uploaded_files) if uploaded_files else 0,
                file_types=file_types,
                success=False  # Will update to True on success
            )
            
            yield (
                chat_history,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value=[], visible=False),
                gr.update(value="", visible=False),
                gr.update(value='''<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
  <span id="processing-msg">Processing your request...</span>
  <span id="processing-timer" style="opacity:0.8; margin-left:8px;"></span>
</div>''', visible=True)
            )
            try:
                if not question_text.strip():
                    analytics_logger.log_question(
                        ip_address=ip_address,
                        question=question_text,
                        num_files=len(uploaded_files) if uploaded_files else 0,
                        file_types=file_types,
                        success=False,
                        error="Empty question"
                    )
                    metric_attrs["failure_reason"] = "validation_empty_question"
                    record_request_metrics(
                        duration_s=time.perf_counter() - run_started_at,
                        success=False,
                        extra_attributes=metric_attrs,
                    )
                    chat_history = _append_chat_exchange(chat_history, question_text, "Please enter a question.")
                    yield (
                        chat_history,
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(value=[], visible=False),
                        gr.update(value="", visible=False),
                        gr.update(value="", visible=False)
                    )
                    return
                if not uploaded_files:
                    analytics_logger.log_question(
                        ip_address=ip_address,
                        question=question_text,
                        num_files=0,
                        file_types=[],
                        success=False,
                        error="No files uploaded"
                    )
                    metric_attrs["failure_reason"] = "validation_no_files"
                    record_request_metrics(
                        duration_s=time.perf_counter() - run_started_at,
                        success=False,
                        extra_attributes=metric_attrs,
                    )
                    chat_history = _append_chat_exchange(chat_history, question_text, "Please upload at least one document.")
                    yield (
                        chat_history,
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(value=[], visible=False),
                        gr.update(value="", visible=False),
                        gr.update(value="", visible=False)
                    )
                    return
                if not isinstance(session_state.value, dict):
                    session_state.value = {}
                session_state.value.setdefault("chart_chunks", [])
                session_state.value.setdefault("last_documents", [])
                session_state.value.setdefault("chart_chunks_missing_logged", False)

                # Build a stable signature for the current upload set.
                file_entries = []
                for file in uploaded_files:
                    with open(file.name, "rb") as f:
                        file_content = f.read()
                    file_hash = processor._generate_hash(file_content)
                    file_entries.append((file, file_hash))
                current_file_hashes = tuple(sorted(file_hash for _, file_hash in file_entries))

                cached_hashes = tuple(session_state.value.get("file_hashes") or ())
                cached_retriever = session_state.value.get("retriever")
                retriever = None
                used_cached_retriever = False

                # Stage 2/3 can be skipped when files are unchanged in this session.
                if cached_retriever is not None and cached_hashes == current_file_hashes:
                    retriever = cached_retriever
                    used_cached_retriever = True
                    logger.info("Using session-cached retriever (file set unchanged).")
                    if (
                        not session_state.value.get("chart_chunks")
                        and not session_state.value.get("chart_chunks_missing_logged")
                    ):
                        logger.warning(
                            "Session-cached retriever has no chart chunk candidates; chart fallback gallery is limited for this request."
                        )
                        session_state.value["chart_chunks_missing_logged"] = True

                # Stage 2: Chunking with per-chunk progress and rotating status
                def load_or_process(file, file_hash):
                    cache_path = processor.cache_dir / f"{file_hash}.pkl"
                    if processor._is_cache_valid(cache_path):
                        chunks = processor._load_from_cache(cache_path)
                        if chunks:
                            logger.info(f"Using cached chunks for {file.name}")
                            return chunks
                    chunks = processor._process_file(file)
                    processor._save_to_cache(chunks, cache_path)
                    return chunks

                if retriever is None:
                    all_chunks = []
                    seen_hashes = set()
                    chunks_by_file = []
                    total_chunks = 0

                    def _spinner_yield(msg: str):
                        """Build a 10-element Gradio yield tuple showing a spinner message."""
                        return (
                            chat_history,
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(value=[], visible=False),
                            gr.update(value="", visible=False),
                            gr.update(value=(
                                '<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04); display:flex; align-items:center;">'
                                '<img src="https://media.giphy.com/media/26ufnwz3wDUli7GU0/giphy.gif" alt="AI working" style="height:40px; margin-right:16px;">'
                                f'<span id="processing-msg">{msg}</span>'
                                '</div>'
                            ), visible=True)
                        )

                    # ── B2: Streaming chunker decision ───────────────────────────────
                    # Only activate when: kill-switch on, single file, AND cache miss.
                    _use_streaming = (
                        getattr(parameters, "STREAMING_CHUNKER_ENABLED", True)
                        and len(file_entries) == 1
                    )
                    if _use_streaming:
                        _sf, _sfh = file_entries[0]
                        _cache_path_s = processor.cache_dir / f"{_sfh}.pkl"
                        _use_streaming = not processor._is_cache_valid(_cache_path_s)

                    if _use_streaming:
                        # Producer: parse + chunk → queue
                        # Consumer: queue → ChromaDB (running concurrently)
                        _file, _fhash = file_entries[0]
                        _cache_path_s = processor.cache_dir / f"{_fhash}.pkl"
                        _chroma_dir = parameters.CHROMA_DB_PATH

                        _chunk_queue = queue.Queue(maxsize=500)
                        _thread_exc  = [None]           # cross-thread exception carrier
                        _abort_event = threading.Event()
                        _prod_chunks = [None]           # List[Document] returned by producer
                        _cons_result = [None]           # (chunks, vstore) returned by consumer

                        def _producer():
                            try:
                                _chunks_p = processor._process_file(
                                    _file, chunk_queue=_chunk_queue
                                )
                                _prod_chunks[0] = _chunks_p
                            except Exception as exc:
                                _thread_exc[0] = exc
                                _abort_event.set()
                            finally:
                                # Send emergency sentinel if producer died before its own sentinel
                                if _abort_event.is_set():
                                    try:
                                        _chunk_queue.put(None, timeout=5)
                                    except Exception:
                                        pass

                        def _consumer():
                            try:
                                _result = retriever_indexer.ingest_streaming(
                                    chunk_queue=_chunk_queue,
                                    chroma_dir=_chroma_dir,
                                )
                                _cons_result[0] = _result
                            except Exception as exc:
                                if _thread_exc[0] is None:
                                    _thread_exc[0] = exc
                                _abort_event.set()

                        t_prod = threading.Thread(target=_producer, name="stream-producer", daemon=True)
                        t_cons = threading.Thread(target=_consumer, name="stream-consumer", daemon=True)
                        t_prod.start()
                        t_cons.start()

                        # Poll every 2s while threads run so Gradio doesn't see a frozen
                        # connection — without periodic yields the browser shows a stale/error state.
                        _stream_start = time.time()
                        while t_prod.is_alive() or t_cons.is_alive():
                            _elapsed = int(time.time() - _stream_start)
                            yield _spinner_yield(f"Parsing &amp; indexing... ({_elapsed}s)")
                            t_prod.join(timeout=2)   # wait up to 2s then loop and yield again

                        t_cons.join(timeout=300)   # 5-min hard cap; prevents infinite hang
                        if t_cons.is_alive():
                            logger.error("[STREAMING] Consumer thread still alive after 300s — treating as failure")
                            _thread_exc[0] = _thread_exc[0] or RuntimeError("Consumer thread timed out (300s)")

                        if _thread_exc[0] is not None:
                            logger.warning(
                                "[STREAMING] Streaming path failed (%s: %s) — falling back to sequential",
                                type(_thread_exc[0]).__name__, _thread_exc[0],
                            )
                            _use_streaming = False  # fall through to sequential loop below
                        else:
                            _chunks = _prod_chunks[0]
                            processor._save_to_cache(_chunks, _cache_path_s)

                            # Dedup — mirrors sequential path lines below
                            for chunk in _chunks:
                                chunk_hash = processor._generate_hash(chunk.page_content.encode())
                                if chunk_hash not in seen_hashes:
                                    seen_hashes.add(chunk_hash)
                                    all_chunks.append(chunk)

                            _coll_chunks, _vstore = _cons_result[0]

                            # Stage 3 UI update — same spinner as sequential path
                            yield _spinner_yield("Processing your request...")

                            # BM25 needs all chunks — build after both threads have joined
                            try:
                                retriever = retriever_indexer.build_bm25_and_ensemble(
                                    all_chunks=all_chunks,
                                    vector_store=_vstore,
                                )
                            except Exception as e:
                                logger.warning("[STREAMING] build_bm25_and_ensemble failed: %s; falling back", e)
                                retriever = retriever_indexer.build_hybrid_retriever(all_chunks)

                            session_state.value["retriever"] = retriever
                            session_state.value["file_hashes"] = current_file_hashes
                            session_state.value["chart_chunks"] = [
                                chunk for chunk in all_chunks
                                if (getattr(chunk, "metadata", {}) or {}).get("chart_image_path")
                            ]
                            session_state.value["chart_chunks_missing_logged"] = False

                    if not _use_streaming:
                        # ── Original sequential path (unchanged) ────────────────────
                        for file, file_hash in file_entries:
                            chunks = load_or_process(file, file_hash)
                            chunks_by_file.append(chunks)
                            total_chunks += len(chunks)
                        if total_chunks == 0:
                            total_chunks = 1
                        chunk_idx = 0
                        for chunks in chunks_by_file:
                            for chunk in chunks:
                                chunk_hash = processor._generate_hash(chunk.page_content.encode())
                                if chunk_hash not in seen_hashes:
                                    seen_hashes.add(chunk_hash)
                                    all_chunks.append(chunk)
                                # else: skip duplicate chunk
                                chunk_idx += 1
                                # yield progress here if needed
                                yield (
                                    chat_history,
                                    gr.update(visible=False),
                                    gr.update(visible=False),
                                    gr.update(interactive=False),
                                    gr.update(interactive=False),
                                    gr.update(interactive=False),
                                    gr.update(interactive=False),
                                    gr.update(value=[], visible=False),
                                    gr.update(value="", visible=False),
                                    gr.update(value='''<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
  <span id="processing-msg">Processing your request...</span>
  <span id="processing-timer" style="opacity:0.8; margin-left:8px;"></span>
</div>''', visible=True)
                                )
                        # After all chunks, show 100%
                        yield (
                            chat_history,
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(value=[], visible=False),
                            gr.update(value="", visible=False),
                            gr.update(value='''<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
  <span id="processing-msg">Processing your request...</span>
  <span id="processing-timer" style="opacity:0.8; margin-left:8px;"></span>
</div>''', visible=True)
                        )
                        # Stage 3: Building Retriever
                        yield (
                            chat_history,
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(value=[], visible=False),
                            gr.update(value="", visible=False),
                            gr.update(value=(
                                '<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04); display:flex; align-items:center;">'
                                '<img src="https://media.giphy.com/media/26ufnwz3wDUli7GU0/giphy.gif" alt="AI working" style="height:40px; margin-right:16px;">'
                                '<span id="processing-msg">Processing your request...</span>'
                                '</div>'
                            ), visible=True)
                        )
                        retriever = retriever_indexer.build_hybrid_retriever(all_chunks)
                        session_state.value["retriever"] = retriever
                        session_state.value["file_hashes"] = current_file_hashes
                        session_state.value["chart_chunks"] = [
                            chunk
                            for chunk in all_chunks
                            if (getattr(chunk, "metadata", {}) or {}).get("chart_image_path")
                        ]
                        session_state.value["chart_chunks_missing_logged"] = False
                # Stage 4: Generating Answer
                yield (
                    chat_history,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(value=[], visible=False),
                    gr.update(value="", visible=False),
                    gr.update(value='''<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
  <span id="processing-msg">Processing your request...</span>
  <span id="processing-timer" style="opacity:0.8; margin-left:8px;"></span>
</div>''', visible=True)
                )
                with start_span(
                    "smartdoc.workflow.run",
                    attributes={
                        "smartdoc.question.length": len(question_text.strip()),
                        "smartdoc.cached_retriever": used_cached_retriever,
                        "smartdoc.provider.llm": parameters.LLM_PROVIDER,
                    },
                ) as workflow_span:
                    try:
                        result = orchestrator.run_workflow(question=question_text, retriever=retriever)
                    except Exception as workflow_error:
                        mark_span_error(workflow_span, workflow_error)
                        raise
                answer = result["draft_answer"]
                logger.info(
                    "[ANSWER_LOG] delivered_answer | question=%s | chars=%d\n%s",
                    question_text[:200],
                    len((answer or "").strip()),
                    (answer or "").strip(),
                )
                # Stage 5: Verifying Answer
                yield (
                    chat_history,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(value=[], visible=False),
                    gr.update(value="", visible=False),
                    gr.update(value='''<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
  <span id="processing-msg">Processing your request...</span>
  <span id="processing-timer" style="opacity:0.8; margin-left:8px;"></span>
</div>''', visible=True)
                )
                verification = result.get("verification_report", "No verification details available.")
                logger.info(f"Verification (internal):\n{verification}")
                # Do not display verification to user, only use internally
                
                # Log successful question answering
                analytics_logger.log_question(
                    ip_address=ip_address,
                    question=question_text,
                    num_files=len(uploaded_files),
                    file_types=file_types,
                    success=True
                )
                metric_attrs["cached_retriever"] = used_cached_retriever
                record_request_metrics(
                    duration_s=time.perf_counter() - run_started_at,
                    success=True,
                    extra_attributes=metric_attrs,
                )
           
                chat_history = _append_chat_exchange(chat_history, question_text, f"**Answer:**\n{answer}")
                # Reuse documents from the workflow instead of calling
                # retriever.invoke() again — eliminates inconsistency
                # between the docs used for answer generation and chart selection.
                retrieved_docs = result.get("documents") or []
                if not retrieved_docs:
                    # Fallback: if orchestrator didn't return docs (e.g. irrelevant question),
                    # do a single retrieval for the chart gallery.
                    retrieved_docs = retriever.invoke(question_text)
                session_state.value["last_documents"] = retrieved_docs
                chart_chunks = session_state.value.get("chart_chunks") or []
                chart_gallery_items, chart_gallery_note_text, chart_gallery_mode = build_chart_gallery_payload(
                    question=question_text,
                    retrieved_docs=retrieved_docs,
                    chart_chunks=chart_chunks,
                    max_items=3,
                    answer_text=answer,          # Tier 0: answer-guided page lookup
                )
                answer_guided_count = len(chart_gallery_items) if chart_gallery_mode == "answer_guided" else 0
                direct_count     = len(chart_gallery_items) if chart_gallery_mode == "direct"     else 0
                page_match_count = len(chart_gallery_items) if chart_gallery_mode == "page_match" else 0
                fallback_count   = len(chart_gallery_items) if chart_gallery_mode == "fallback"   else 0
                logger.info(
                    "[CHART_GALLERY] mode=%s answer_guided=%d direct=%d page_match=%d fallback=%d selected=%d candidates=%d",
                    chart_gallery_mode,
                    answer_guided_count,
                    direct_count,
                    page_match_count,
                    fallback_count,
                    len(chart_gallery_items),
                    len(chart_chunks),
                )
                yield (
                    chat_history,
                    gr.update(visible=True),  # doc_context_display
                    gr.update(visible=True),  # refresh_context_btn
                    gr.update(interactive=False),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(value=chart_gallery_items, visible=bool(chart_gallery_items)),
                    gr.update(value=chart_gallery_note_text, visible=bool(chart_gallery_note_text)),
                    gr.update(value='''<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
  <span id="processing-msg">Processing your request...</span>
  <span id="processing-timer" style="opacity:0.8; margin-left:8px;"></span>
</div>''', visible=True)
                )      
                yield (
                    chat_history,
                    gr.update(visible=True),
                    gr.update(visible=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(value=chart_gallery_items, visible=bool(chart_gallery_items)),
                    gr.update(value=chart_gallery_note_text, visible=bool(chart_gallery_note_text)),
                    gr.update(value="", visible=False)
                )
            except Exception as e:
                logger.error(f"Processing error: {e}", exc_info=True)
                
                # Log failed question
                analytics_logger.log_question(
                    ip_address=ip_address,
                    question=question_text,
                    num_files=len(uploaded_files) if uploaded_files else 0,
                    file_types=file_types,
                    success=False,
                    error=str(e)
                )
                metric_attrs["failure_reason"] = "exception"
                record_request_metrics(
                    duration_s=time.perf_counter() - run_started_at,
                    success=False,
                    extra_attributes=metric_attrs,
                )
           
                chat_history = _append_chat_exchange(chat_history, question_text, f"Error: {str(e)}")
                yield (
                    chat_history,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(value=[], visible=False),
                    gr.update(value="", visible=False),
                    gr.update(value="", visible=False)
                )

        submit_btn.click(
            fn=process_question,
            inputs=[question, files, chat],
            outputs=[chat, doc_context_display, refresh_context_btn, submit_btn, question, files, example_dropdown, referenced_charts, chart_gallery_note, processing_message],
            queue=True,
            show_progress=True
        )

        def refresh_context():
            docs = session_state.value.get("last_documents", [])
            last_question = ""
            for msg in reversed(chat.value or []):
                if isinstance(msg, (list, tuple)) and len(msg) >= 1:
                    last_question = msg[0] or ""
                    if last_question:
                        break
                elif isinstance(msg, dict) and msg.get("role") == "user":
                    last_question = msg.get("content", "")
                    if last_question:
                        break
                elif getattr(msg, "role", None) == "user":
                    last_question = getattr(msg, "content", "") or ""
                    if last_question:
                        break
            return format_document_context(docs, last_question)

        refresh_context_btn.click(
            fn=refresh_context,
            inputs=[],
            outputs=[doc_context_display]
        )

        def load_example(example_key):
            if not example_key or example_key not in EXAMPLES:
                return [], "", "Select a valid example from the dropdown above"
            ex_data = EXAMPLES[example_key]
            question_text = ex_data["question"]
            file_names = ex_data["file_paths"]
        
            # Try to download from HF dataset if on Spaces
            if is_hf_space:
                try:
                    from datasets import load_dataset
                    import tempfile
                    
                    copied_files = []
                    file_info_text = f"✅ Loaded: {example_key}\n\n"
                    
                    # Get HF token - REQUIRED for gated datasets
                    hf_token = os.environ.get("HF_TOKEN", None)
                    
                    if not hf_token:
                        logger.warning("HF_TOKEN not set - required for gated datasets")
                        return [], "", (
                            "❌ **Authentication Required**\n\n"
                            "The example dataset is gated and requires authentication.\n\n"
                            "**To fix:**\n"
                            "1. Go to Space Settings → Repository secrets\n"
                            "2. Add secret: `HF_TOKEN` = your Hugging Face token\n"
                            "3. Restart the Space\n\n"
                            "Or make your dataset public at:\n"
                            "https://huggingface.co/datasets/TilanB/smartdoc-samples/settings\n\n"
                            "For now, please **upload files manually**."
                        )
                    
                    try:
                        # Load dataset - uses row-based structure
                        logger.info(f"Loading dataset from HuggingFace: TilanB/smartdoc-samples")
                        ds = load_dataset(
                            "TilanB/smartdoc-samples",
                            split="train",
                            token=hf_token
                        )
                        logger.info(f"Dataset loaded with {len(ds)} rows")
                        
                        # Create temp directory for files
                        temp_dir = tempfile.mkdtemp(prefix='hf_examples_')
                        
                        # Debug: Log first row structure
                        if len(ds) > 0:
                            first_row = ds[0]
                            pdf_data = first_row.get('pdf', None)
                            logger.info(f"Dataset first row 'pdf' type: {type(pdf_data)}")
                            
                            # Handle different types
                            if hasattr(pdf_data, 'stream') and hasattr(pdf_data.stream, 'name'):
                                # pdfplumber PDF object
                                logger.info(f"PDF is pdfplumber object, stream path: {pdf_data.stream.name}")
                            elif isinstance(pdf_data, dict):
                                logger.info(f"PDF dict keys: {list(pdf_data.keys())}")
                                if 'path' in pdf_data:
                                    logger.info(f"PDF path: {pdf_data.get('path', 'N/A')}")
            
                        # Extract requested files from dataset rows
                        for file_path in file_names:
                            filename = os.path.basename(file_path)
                            file_found = False
                            
                            logger.info(f"Looking for file: {filename}")
                            
                            # Search through dataset rows
                            for row_idx, row in enumerate(ds):
                                # The 'pdf' column contains file objects from HF datasets
                                pdf_data = row.get('pdf', None)
                                
                                if pdf_data is None:
                                    continue
                                
                                # Extract the actual filename from the pdf data
                                # HF datasets with PDF files can return different types:
                                # 1. pdfplumber.pdf.PDF objects (when using pdf feature type)
                                # 2. dict with 'path' and 'bytes' keys
                                # 3. str path
                                # 4. bytes directly
                                
                                row_filename = ""
                                
                                # Check for pdfplumber PDF object (has .stream.name attribute)
                                if hasattr(pdf_data, 'stream') and hasattr(pdf_data.stream, 'name'):
                                    row_filename = pdf_data.stream.name
                                    logger.debug(f"Got filename from pdfplumber stream: {row_filename}")
                                # Check for pdfplumber PDF object with path attribute
                                elif hasattr(pdf_data, 'path'):
                                    row_filename = pdf_data.path
                                # Check for dict format
                                elif isinstance(pdf_data, dict):
                                    row_filename = pdf_data.get('path', '')
                                # Check for string path
                                elif isinstance(pdf_data, str):
                                    row_filename = pdf_data
                                
                                row_basename = os.path.basename(str(row_filename))
                                logger.debug(f"Row {row_idx}: checking '{row_basename}' vs '{filename}'")
                                
                                # Match by filename
                                if row_basename == filename:
                                    temp_file_path = os.path.join(temp_dir, filename)
                                    logger.info(f"Found match! Extracting {filename}...")
                                    
                                    try:
                                        extracted = False
                                        
                                        # Handle pdfplumber PDF object
                                        if hasattr(pdf_data, 'stream'):
                                            # Get the file path from pdfplumber's stream
                                            source_path = pdf_data.stream.name
                                            if source_path and os.path.exists(source_path):
                                                shutil.copy2(source_path, temp_file_path)
                                                logger.info(f"Copied from pdfplumber stream: {source_path}")
                                                extracted = True
                                            else:
                                                # Try to read bytes from stream
                                                try:
                                                    pdf_data.stream.seek(0)
                                                    pdf_bytes = pdf_data.stream.read()
                                                    with open(temp_file_path, 'wb') as f:
                                                        f.write(pdf_bytes)
                                                    logger.info(f"Wrote {len(pdf_bytes)} bytes from pdfplumber stream")
                                                    extracted = True
                                                except Exception as stream_err:
                                                    logger.warning(f"Could not read stream: {stream_err}")
                                        
                                        # Handle dict format
                                        elif isinstance(pdf_data, dict):
                                            if 'bytes' in pdf_data and pdf_data['bytes']:
                                                with open(temp_file_path, 'wb') as f:
                                                    f.write(pdf_data['bytes'])
                                                logger.info(f"Wrote {len(pdf_data['bytes'])} bytes")
                                                extracted = True
                                            elif 'path' in pdf_data and pdf_data['path'] and os.path.exists(pdf_data['path']):
                                                shutil.copy2(pdf_data['path'], temp_file_path)
                                                logger.info(f"Copied from dict path: {pdf_data['path']}")
                                                extracted = True
                                        
                                        # Handle bytes directly
                                        elif isinstance(pdf_data, bytes):
                                            with open(temp_file_path, 'wb') as f:
                                                f.write(pdf_data)
                                            extracted = True
                                        
                                        # Handle string path
                                        elif isinstance(pdf_data, str) and os.path.exists(pdf_data):
                                            shutil.copy2(pdf_data, temp_file_path)
                                            extracted = True
                                        
                                        if extracted and os.path.exists(temp_file_path):
                                            copied_files.append(temp_file_path)
                                            file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
                                            file_info_text += f"📄 {filename} ({file_size_mb:.2f} MB)\n"
                                            file_found = True
                                            logger.info(f"✅ Successfully extracted {filename}")
                                            break
                                        else:
                                            logger.error(f"Could not extract file: {type(pdf_data)}")
                                            
                                    except Exception as ex:
                                        logger.error(f"Failed to extract {filename}: {ex}", exc_info=True)
                                        continue
                            
                            if not file_found:
                                logger.warning(f"❌ File {filename} not found in dataset rows")
                                # Debug: show what's available
                                for idx, row in enumerate(ds):
                                    pdf_data = row.get('pdf', None)
                                    if pdf_data and hasattr(pdf_data, 'stream') and hasattr(pdf_data.stream, 'name'):
                                        available_name = os.path.basename(str(pdf_data.stream.name))
                                        logger.info(f"  Available file in row {idx}: '{available_name}'")
                                file_info_text += f"⚠️ {filename} - Not found in dataset\n"

                        if not copied_files:
                            if len(ds) > 0:
                                logger.error(f"Dataset structure: {list(ds[0].keys())}")
                            return [], "", f"❌ Could not find example files in dataset.\n\nDataset has {len(ds)} rows. Please upload files manually."
                        
                        return copied_files, question_text, file_info_text
                        
                    except Exception as e:
                        error_msg = str(e)
                        logger.error(f"Failed to load dataset: {e}", exc_info=True)
                        
                        # Check for gated dataset error
                        if "gated" in error_msg.lower() or "authenticated" in error_msg.lower():
                            return [], "", (
                                "❌ **Dataset Access Denied**\n\n"
                                "The dataset is gated and your token doesn't have access.\n\n"
                                "**To fix:**\n"
                                "1. Visit: https://huggingface.co/datasets/TilanB/smartdoc-samples\n"
                                "2. Accept the access terms (if any)\n"
                                "3. Make sure HF_TOKEN is set in Space secrets\n\n"
                                "Or make your dataset public.\n\n"
                                "For now, please **upload files manually**."
                            )
                        
                        return [], "", f"❌ Failed to load dataset: {error_msg}\n\nPlease upload files manually."
                
                except ImportError as e:
                    logger.error(f"datasets package not installed: {e}")
                    return [], "", "❌ 'datasets' package not installed"
            else:
                # Local mode - use files from samples directory
                import tempfile
                temp_dir = tempfile.mkdtemp()
                copied_files = []
                file_info_text = f"Loaded: {example_key}\n\n"
                for source_file_path in file_names:
                    abs_source = os.path.abspath(source_file_path)
                    if os.path.exists(abs_source):
                        filename = os.path.basename(abs_source)
                        temp_file_path = os.path.join(temp_dir, filename)
                        shutil.copy2(abs_source, temp_file_path)
                        copied_files.append(temp_file_path)
                        file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
                        file_info_text += f"{filename} ({file_size_mb:.2f} MB)\n"
                    else:
                        file_info_text += f"{source_file_path} not found\n"
                if not copied_files:
                    return [], "", "Could not load example files"
                return copied_files, question_text, file_info_text
    
        example_dropdown.change(
            fn=load_example,
            inputs=[example_dropdown],
            outputs=[files, question, loaded_file_info]
        )
    
        # Show loaded_file_info when example is selected
        def show_info(example_key):
            return gr.update(visible=bool(example_key))
    
        example_dropdown.change(
            fn=show_info,
            inputs=[example_dropdown],
            outputs=[loaded_file_info]
        )
    # Launch server - Compatible with both local and Hugging Face Spaces
    # HF Spaces sets SPACE_ID environment variable
    is_hf_space = os.environ.get("SPACE_ID") is not None

    demo.queue()
    if is_hf_space:
        # Hugging Face Spaces configuration
        logger.info("Running on Hugging Face Spaces")
        demo.launch(server_name="0.0.0.0", server_port=7860)
    else:
        # Local development configuration
        configured_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
        server_port = _find_open_port(configured_port)
        logger.info(f"Launching Gradio on port {server_port}")
        logger.info(f"Access the app at: http://127.0.0.1:{server_port}")
        demo.launch(server_name="127.0.0.1", server_port=server_port, share=False)


if __name__ == "__main__":
    main()

