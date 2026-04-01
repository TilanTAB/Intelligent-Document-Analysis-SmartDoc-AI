import os
import gc
import hashlib
import json
import logging
import pickle
import re
import shutil
import struct  # For handling struct.error exceptions
import concurrent.futures
import queue as _queue_module  # B2: aliased to avoid shadowing any local 'queue' variable
import threading as _threading   # per-page timeout guard
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta

# Heavy third-party imports — may fail in ProcessPoolExecutor worker
# subprocesses on Windows (spawn mode re-imports the full module tree
# and google.genai Pydantic schema construction can cause MemoryError).
# stdlib imports above this line are guaranteed available regardless.
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from configuration.parameters import parameters
from configuration.definitions import MAX_TOTAL_SIZE, ALLOWED_TYPES
from PIL import Image
from core.vision_client import (
    get_vision_client,
    analyze_chart_images,
    VisionClientUnavailable,
)
from core.perf_monitor import latency_monitor

logger = logging.getLogger(__name__)


PDF_ANALYSIS_MODES = {"pdf_only", "vision_only", "both"}
CHART_DETECTION_BACKENDS = {"pdfplumber", "opencv_optimized"}


def normalize_pdf_analysis_mode(mode: Optional[str]) -> str:
    """Normalize analysis mode values and aliases into supported canonical values."""
    raw_mode = str(mode or "").strip().lower()
    aliases = {
        "pdf": "pdf_only",
        "text": "pdf_only",
        "vision": "vision_only",
        "image": "vision_only",
        "hybrid": "both",
        "all": "both",
    }
    normalized = aliases.get(raw_mode, raw_mode)
    if normalized not in PDF_ANALYSIS_MODES:
        return "both"
    return normalized


def normalize_chart_detection_backend(backend: Optional[str]) -> str:
    """Normalize chart detection backend values and aliases into canonical names."""
    raw_backend = str(backend or "").strip().lower()
    aliases = {
        "pdf": "pdfplumber",
        "pdf_parser": "pdfplumber",
        "structural": "pdfplumber",
        "opencv": "opencv_optimized",
        "local": "opencv_optimized",
    }
    normalized = aliases.get(raw_backend, raw_backend)
    if normalized not in CHART_DETECTION_BACKENDS:
        return "pdfplumber"
    return normalized


def _is_full_page_vision_detection(detection_result: Dict[str, Any]) -> bool:
    """Return True when detection metadata represents forced full-page vision analysis."""
    result = detection_result or {}
    features = result.get("features") or {}
    chart_types = result.get("chart_types") or []
    return bool(features.get("full_page_vision")) or "full_page_vision" in chart_types


def _detection_method_label(detection_result: Dict[str, Any], vision_provider: str, is_batch: bool) -> str:
    """Human-readable extraction method label for chart/page chunks."""
    result = detection_result or {}
    features = result.get("features") or {}
    chart_types = set(result.get("chart_types") or [])
    if _is_full_page_vision_detection(detection_result):
        mode_label = "Batch" if is_batch else "Sequential"
        return f"Vision-Only ({vision_provider.upper()} {mode_label} page analysis)"
    if features.get("pdfplumber_candidate") or "pdfplumber_candidate" in chart_types:
        mode_label = "Batch" if is_batch else "Sequential"
        return f"PDFPlumber-Triage + {vision_provider.upper()} {mode_label} analysis"
    if is_batch:
        return f"Hybrid (Local OpenCV + {vision_provider.upper()} Batch Analysis)"
    return f"Hybrid (Local OpenCV + {vision_provider.upper()} Sequential)"

def preprocess_image(image, max_dim=1000):
    """Downscale image to max_dim before OpenCV processing."""
    if max(image.size) > max_dim:
        ratio = max_dim / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        return image.resize(new_size, Image.Resampling.LANCZOS)
    return image

def detect_chart_on_page(args):
    """
    Top-level function for parallel local chart detection (required for ProcessPoolExecutor).
    Returns the page number, the PIL image, and the detection result.
    """
    page_num, image = args
    from content_analyzer.visual_detector import LocalChartDetector
    # Downscale image before detection to save memory
    image = preprocess_image(
        image,
        max_dim=max(256, int(getattr(parameters, "CHART_OPENCV_MAX_DIM", 700) or 700)),
    )
    detection_result = LocalChartDetector.detect_charts(image)
    return (page_num, image, detection_result)


def detect_chart_on_page_path(args):
    """
    Process-pool-safe chart detection from an on-disk image path.

    Why this exists:
    - Passing large PIL images through ProcessPool on Windows is memory-heavy and
      can break worker processes (BrokenProcessPool).
    - Sending file paths is cheaper and more stable across spawn-based workers.
    """
    page_num, image_path = args
    from pathlib import Path as _Path
    from PIL import Image as _Image
    from content_analyzer.visual_detector import LocalChartDetector

    try:
        with _Image.open(image_path) as img:
            prepared = preprocess_image(
                img.convert("RGB"),
                max_dim=max(256, int(getattr(parameters, "CHART_OPENCV_MAX_DIM", 700) or 700)),
            )
            detection_result = LocalChartDetector.detect_charts(prepared)
        return (page_num, str(image_path), detection_result)
    except Exception as e:
        return (
            page_num,
            str(image_path),
            {
                "has_chart": False,
                "confidence": 0.0,
                "chart_types": [],
                "description": f"Detection failed on {_Path(image_path).name}: {type(e).__name__}",
                "features": {},
                "error": str(e),
            },
        )


def _table_to_markdown_impl(table: List[List], page_num: int, table_idx: int) -> str:
    """Convert a table (list of rows) to markdown format."""
    if not table or len(table) < 1:
        return ""

    cleaned_table: List[List[str]] = []
    for row in table:
        if not row:
            continue
        cleaned_row: List[str] = []
        for cell in row:
            if cell:
                cell_text = (
                    str(cell)
                    .replace("\n", " ")
                    .replace("\r", " ")
                    .replace("|", "\\|")
                    .strip()
                )
                cleaned_row.append(cell_text)
            else:
                cleaned_row.append("")
        if any(cleaned_row):
            cleaned_table.append(cleaned_row)

    if len(cleaned_table) < 1:
        return ""

    max_cols = max(len(row) for row in cleaned_table)
    for row in cleaned_table:
        while len(row) < max_cols:
            row.append("")

    md_lines = [f"### Table {table_idx} (Page {page_num})"]
    md_lines.append("| " + " | ".join(cleaned_table[0]) + " |")
    md_lines.append("| " + " | ".join(["---"] * max_cols) + " |")
    for row in cleaned_table[1:]:
        md_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(md_lines)


# ── Per-page timeout guard ────────────────────────────────────────────
# pdfminer (used by pdfplumber internally) can enter an infinite loop
# in its C-level layout parser when it encounters corrupted or unusual
# PDF page structures (e.g. malformed CMap tables, broken XObject
# streams).  No Python exception is raised — the thread simply hangs.
#
# This helper runs a callable in a daemon thread with a wall-clock
# timeout.  If the callable doesn't return in time, we raise
# TimeoutError so the caller can skip the page and try a fallback.
#
# The daemon thread may leak if the C code never returns, but that is
# acceptable — the process will reclaim it on exit, and the alternative
# (hanging the entire app for hours) is far worse.
_PAGE_OP_TIMEOUT_S = int(os.environ.get("PDF_PAGE_OP_TIMEOUT_S", "60"))


def _run_with_timeout(func, timeout_s: int = _PAGE_OP_TIMEOUT_S):
    """Run *func()* in a daemon thread; raise TimeoutError if it hangs.

    Returns the function's return value on success.
    Re-raises any exception the function raised.
    """
    result_box: list = [None]
    error_box: list = [None]

    def _worker():
        try:
            result_box[0] = func()
        except Exception as exc:
            error_box[0] = exc

    t = _threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout_s)

    if t.is_alive():
        # Thread leaked — monitor accumulation so repeated hangs
        # are visible in logs before they degrade the entire app.
        active = _threading.active_count()
        logger.warning(
            "[THREAD_LEAK] Daemon thread abandoned after %ds timeout. "
            "Active threads now: %d (>20 indicates progressive leak)",
            timeout_s,
            active,
        )
        raise TimeoutError(
            f"Operation did not complete within {timeout_s}s "
            f"(likely pdfminer C-level hang)"
        )
    if error_box[0] is not None:
        raise error_box[0]
    return result_box[0]


_page_offset_cache: dict = {}  # {file_content_hash: offset} — keyed on content, not path

def detect_page_label_offset(file_path: str, file_hash: str = "") -> int:
    """Detect the front-matter page offset for a PDF.

    Many reports/books have unnumbered or Roman-numeral front matter
    (cover, copyright, TOC, abbreviations) before Arabic page "1" starts.
    PDF parsers count ALL physical pages sequentially, so printed page 12
    may be raw page 35 (offset = 23).

    This function uses pypdf's page_labels (derived from the PDF /PageLabels
    entry) to find where Arabic numbering begins.  If the PDF has no labels,
    it falls back to a heuristic: scan the first ~60 pages' extracted text
    for a standalone "1" at the bottom (common footer page number).

    Args:
        file_path: Path to the PDF file.
        file_hash: Optional content-based hash for caching.  When provided,
                   results are cached by hash (safe across Gradio temp renames).
                   When empty, no caching is applied.

    Returns:
        int: The number of front-matter pages before printed page 1.
             Raw page index (1-based) minus offset = printed page number.
             Returns 0 if no offset is detected or pypdf is unavailable.
    """
    # Cache lookup by content hash (not path) — safe across Gradio temp renames
    if file_hash and file_hash in _page_offset_cache:
        return _page_offset_cache[file_hash]
    def _detect() -> int:
        """Inner detection logic — returns offset or 0."""
        try:
            from pypdf import PdfReader
        except ImportError:
            logger.debug("[PAGE_OFFSET] pypdf not available — offset=0")
            return 0

        try:
            reader = PdfReader(file_path)
        except Exception as e:
            logger.debug("[PAGE_OFFSET] Failed to open PDF with pypdf: %s — offset=0", e)
            return 0

        # ── Strategy 1: PDF /PageLabels (most reliable) ──────────────────
        try:
            labels = reader.page_labels
            if labels:
                for raw_idx, label in enumerate(labels):
                    label_stripped = str(label).strip()
                    if label_stripped == "1":
                        offset = raw_idx  # raw_idx is 0-based
                        if offset > 0:
                            logger.info(
                                "[PAGE_OFFSET] Detected offset=%d from PDF page labels "
                                "(physical page %d has label '1') file=%s",
                                offset, raw_idx + 1, Path(file_path).name,
                            )
                            return offset
        except Exception as e:
            logger.debug("[PAGE_OFFSET] page_labels extraction failed: %s", e)

        # ── Strategy 2: Heuristic — scan footer text for page "1" ────────
        try:
            scan_limit = min(len(reader.pages), 60)
            for raw_idx in range(scan_limit):
                try:
                    text = (reader.pages[raw_idx].extract_text() or "").strip()
                    if not text:
                        continue
                    lines = text.splitlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if re.match(r'^[-–—\s]*1[-–—\s]*$', last_line) or last_line == "1":
                            offset = raw_idx
                            if offset > 0:
                                logger.info(
                                    "[PAGE_OFFSET] Detected offset=%d from footer heuristic "
                                    "(physical page %d has footer '1') file=%s",
                                    offset, raw_idx + 1, Path(file_path).name,
                                )
                                return offset
                except Exception:
                    continue
        except Exception as e:
            logger.debug("[PAGE_OFFSET] Footer heuristic failed: %s", e)

        logger.debug("[PAGE_OFFSET] No offset detected for %s — offset=0", Path(file_path).name)
        return 0

    result = _detect()
    if file_hash:
        _page_offset_cache[file_hash] = result
    return result


def extract_pdf_page_range_payload(
    args: Tuple[str, int, int, Dict[str, Any], Dict[str, Any], Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Process a page range from a PDF file in an isolated worker process.

    Returns a payload with:
    - pages: [{"page_num": int, "tables": list, "text": str}, ...]
    - text_fallback_success / text_fallback_fail counters.
    """
    file_path, page_start, page_end, default_parameters, text_parameters, hybrid_parameters = args

    import pdfplumber

    pages_payload: List[Dict[str, Any]] = []
    text_fallback_success = 0
    text_fallback_fail = 0
    pypdf_reader = None
    pypdf_unavailable = False

    def _maybe_get_pypdf_reader():
        nonlocal pypdf_reader, pypdf_unavailable
        if pypdf_unavailable:
            return None
        if pypdf_reader is not None:
            return pypdf_reader
        try:
            from pypdf import PdfReader  # Optional fallback parser

            pypdf_reader = PdfReader(file_path)
            return pypdf_reader
        except Exception:
            pypdf_unavailable = True
            return None

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        safe_start = max(1, int(page_start))
        safe_end = min(int(page_end), total_pages)
        for page_num in range(safe_start, safe_end + 1):
            page_idx = page_num - 1
            page = pdf.pages[page_idx]
            page_tables: List[List[List[Any]]] = []
            table_hashes = set()

            def add_table_if_unique(table) -> bool:
                if not table or len(table) < 2:
                    return False
                table_hash = hash(str(table))
                if table_hash in table_hashes:
                    return False
                table_hashes.add(table_hash)
                page_tables.append(table)
                return True

            try:
                # Per-page timeout guard: pdfminer C-level layout
                # parser can hang forever on corrupted pages.
                def _worker_extract_page():
                    _tables = []
                    for params in (
                        default_parameters,
                        text_parameters,
                        hybrid_parameters,
                    ):
                        try:
                            extracted = page.extract_tables(params) if params else page.extract_tables()
                            if extracted:
                                for table in extracted:
                                    if table and len(table) >= 2:
                                        _tables.append(table)
                        except (struct.error, Exception):
                            continue

                    try:
                        found_tables = page.find_tables(text_parameters)
                        if found_tables:
                            for ft in found_tables:
                                table = ft.extract()
                                if table and len(table) >= 2:
                                    _tables.append(table)
                    except (struct.error, Exception):
                        pass

                    _text = ""
                    try:
                        _text = page.extract_text() or ""
                    except Exception:
                        _text = ""
                    return _tables, _text

                try:
                    raw_tables, text = _run_with_timeout(
                        _worker_extract_page,
                        timeout_s=_PAGE_OP_TIMEOUT_S,
                    )
                    for t in raw_tables:
                        add_table_if_unique(t)
                except TimeoutError:
                    # Page hung — try pypdf text fallback
                    text = ""
                    fallback_reader = _maybe_get_pypdf_reader()
                    if fallback_reader is not None and page_idx < len(fallback_reader.pages):
                        try:
                            text = fallback_reader.pages[page_idx].extract_text() or ""
                            if text.strip():
                                text_fallback_success += 1
                            else:
                                text_fallback_fail += 1
                                text = ""
                        except Exception:
                            text_fallback_fail += 1
                            text = ""
                    else:
                        text_fallback_fail += 1

                if not text.strip():
                    # pdfplumber returned empty — try pypdf fallback
                    fallback_reader = _maybe_get_pypdf_reader()
                    if fallback_reader is not None and page_idx < len(fallback_reader.pages):
                        try:
                            fallback_text = fallback_reader.pages[page_idx].extract_text() or ""
                            if fallback_text.strip():
                                text = fallback_text
                                text_fallback_success += 1
                            else:
                                text_fallback_fail += 1
                        except Exception:
                            text_fallback_fail += 1
                    else:
                        text_fallback_fail += 1

                pages_payload.append(
                    {
                        "page_num": page_num,
                        "tables": page_tables,
                        "text": (text or "").strip(),
                    }
                )
            except Exception:
                # Skip broken pages in worker and continue others.
                continue

    return {
        "pages": pages_payload,
        "text_fallback_success": text_fallback_success,
        "text_fallback_fail": text_fallback_fail,
    }


def _parse_chart_analyses(response_text: str, expected_count: int) -> List[str]:
    """
    Parse chart-analysis model output robustly.

    Handles:
    - JSON payloads (preferred)
    - Markdown code fences containing JSON
    - Legacy marker format: ---CHART N---
    - Single-chart plain text fallback
    """
    expected = max(1, int(expected_count or 1))
    text = (response_text or "").strip()
    if not text:
        return ["Analysis unavailable (empty response)"] * expected

    def _clean(value: object) -> str:
        if value is None:
            return ""
        return str(value).strip()

    def _from_json_payload(payload: object) -> List[str]:
        items = []
        if isinstance(payload, dict):
            for key in ("charts", "results", "items", "data"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    items = candidate
                    break
            if not items and all(str(k).isdigit() for k in payload.keys()):
                items = [payload[k] for k in sorted(payload.keys(), key=lambda k: int(k))]
        elif isinstance(payload, list):
            items = payload

        analyses: List[str] = []
        for item in items:
            if isinstance(item, dict):
                analysis = ""
                for field in (
                    "analysis",
                    "summary",
                    "description",
                    "details",
                    "content",
                    "text",
                ):
                    analysis = _clean(item.get(field))
                    if analysis:
                        break
                if not analysis:
                    # Keep a serialized object instead of dropping data silently.
                    analysis = _clean(json.dumps(item, ensure_ascii=False))
            else:
                analysis = _clean(item)

            if analysis:
                analyses.append(analysis)
        return analyses

    def _attempt_json_parse(raw_text: str) -> List[str]:
        candidates: List[str] = []
        stripped = raw_text.strip()
        if stripped:
            candidates.append(stripped)

        fence_match = re.search(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.IGNORECASE | re.DOTALL)
        if fence_match:
            fenced = fence_match.group(1).strip()
            if fenced:
                candidates.append(fenced)

        first_json_start = min(
            [idx for idx in (raw_text.find("{"), raw_text.find("[")) if idx != -1],
            default=-1,
        )
        if first_json_start != -1:
            candidates.append(raw_text[first_json_start:].strip())

        seen = set()
        for candidate in candidates:
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            try:
                payload = json.loads(candidate)
            except Exception:
                continue

            analyses = _from_json_payload(payload)
            if analyses:
                return analyses
        return []

    # 1) Preferred: parse JSON.
    parsed = _attempt_json_parse(text)
    if parsed:
        output = parsed[:expected]
        if len(output) < expected:
            output.extend(["Analysis unavailable (missing chart section)"] * (expected - len(output)))
        return output

    # 2) Legacy marker parsing: ---CHART N---
    marker_pattern = re.compile(r"---\s*CHART\s*(\d+)\s*---", flags=re.IGNORECASE)
    matches = list(marker_pattern.finditer(text))
    if matches:
        sections_by_number = {}
        for idx, match in enumerate(matches):
            chart_num = int(match.group(1))
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            section = text[start:end].strip()
            if section:
                sections_by_number[chart_num] = section

        if sections_by_number:
            return [
                sections_by_number.get(i + 1, "Analysis unavailable (missing chart section)")
                for i in range(expected)
            ]

    # 3) Plain text fallback (common when batch size is 1 and model ignores format).
    if expected == 1:
        return [text]

    # 4) Multi-chart last resort.
    return [text] + ["Analysis unavailable (parsing error)"] * (expected - 1)


def analyze_batch(batch_tuple):
    """
    Top-level function for parallel vision batch analysis.
    """
    batch, batch_num, total_batches, vision_client, vision_provider, vision_model, file_path, parameters, stats = batch_tuple
    try:
        import logging
        logger = logging.getLogger(__name__)
        full_page_mode = all(_is_full_page_vision_detection(detection_result or {}) for _, _, detection_result in batch)
        if full_page_mode:
            prompt = f"""
Analyze the following {len(batch)} document page image(s) in order.

Return JSON ONLY (no markdown or prose wrapper) using this schema:
{{
  "charts": [
    {{
      "chart_number": 1,
      "analysis": "..."
    }}
  ]
}}

For each page analysis include:
- Page Summary
- Headings / Sections
- Key Facts and Figures (with units and dates)
- Visible Table-like Data (if any)
- Visible Chart/Graph Insights (if any)
- Entities (organizations, countries, products, people)
- Important caveats or ambiguities in extraction
"""
        else:
            prompt = f"""
Analyze the following {len(batch)} chart(s)/graph(s) in order.

Return JSON ONLY (no markdown or prose wrapper) using this schema:
{{
  "charts": [
    {{
      "chart_number": 1,
      "analysis": "..."
    }}
  ]
}}

For each chart analysis include:
- Chart Type
- Title
- X-axis
- Y-axis
- Data Points (extract visible exact values when possible)
- Legend
- Trends
- Key Values
- Context
"""
        response_text = analyze_chart_images(
            client=vision_client,
            image_paths=[image_path for _, image_path, _ in batch],
            prompt=prompt,
            model_name=vision_model,
            max_output_tokens=parameters.CHART_MAX_TOKENS * len(batch),
            provider=vision_provider,
        )
        stats['batch_api_calls'] += 1
        analyses = _parse_chart_analyses(response_text=response_text, expected_count=len(batch))
        unavailable_count = sum(1 for item in analyses if "Analysis unavailable" in (item or ""))
        if unavailable_count:
            logger.warning(
                "Batch %s/%s chart parsing produced %s unavailable section(s). "
                "response_preview=%s",
                batch_num,
                total_batches,
                unavailable_count,
                (response_text or "").replace("\n", " ")[:240],
            )
        batch_docs = []
        for idx, (page_num, image_path, detection_result) in enumerate(batch):
            analysis = analyses[idx] if idx < len(analyses) else "Analysis unavailable (parsing error)"
            detection_result = detection_result or {}
            chart_types_str = ", ".join(detection_result.get("chart_types", [])) or "Unknown"
            confidence = float(detection_result.get("confidence", 0.0) or 0.0)
            detection_method = _detection_method_label(
                detection_result=detection_result,
                vision_provider=vision_provider,
                is_batch=True,
            )
            is_full_page_mode = _is_full_page_vision_detection(detection_result)
            heading = "### 📄 Vision Page Analysis" if is_full_page_mode else "### 📊 Chart Analysis"
            doc_type = "text" if is_full_page_mode else "chart"
            extraction_method = "vision_only_full_page" if is_full_page_mode else "hybrid_batch"
            chart_doc = Document(
                page_content=f"""{heading} (Page {page_num})\n\n**Detection Method**: {detection_method}\n**Local Confidence**: {confidence:.0%}\n**Detected Types**: {chart_types_str}\n**Batch Size**: {len(batch)} items analyzed together\n\n---\n\n{analysis}\n""",
                metadata={
                    "source": file_path,
                    "page": page_num,
                    "type": doc_type,
                    "extraction_method": extraction_method,
                    "detection_confidence": confidence,
                    "batch_size": len(batch),
                    "chart_image_path": str(image_path),
                }
            )
            batch_docs.append(chart_doc)
            stats['charts_analyzed_vision'] += 1
        logger.info(f"✅ Batch {batch_num} complete ({len(batch)} charts analyzed)")
        return (batch_num - 1, batch_docs)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Batch analysis failed: {e}, trying sequential fallback...")
        return (batch_num - 1, [])

class DocumentProcessor:
    """
    Processes documents by splitting them into manageable chunks and caching
    the results to avoid reprocessing. Handles chart extraction using local
    OpenCV detection and provider-based vision analysis with parallelization for speed.
    """
    # Cache metadata version - increment when cache format changes
    CACHE_VERSION = 7  # page label offset: stored page numbers now match printed page numbers

    def __init__(self):
        """Initialize the document processor with cache directory and splitter configuration."""
        self.cache_dir = Path(parameters.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.chart_assets_dir = self.cache_dir / "chart_assets"
        self.chart_assets_dir.mkdir(parents=True, exist_ok=True)
        self._cleanup_expired_chart_assets()
        # Content-type-aware splitters (adaptive chunking).
        self.splitter_text = RecursiveCharacterTextSplitter(
            chunk_size=max(1, int(parameters.CHUNK_SIZE_TEXT or parameters.CHUNK_SIZE)),
            chunk_overlap=max(0, int(parameters.CHUNK_OVERLAP_TEXT or parameters.CHUNK_OVERLAP)),
            length_function=len,
            is_separator_regex=False,
        )
        self.splitter_table = RecursiveCharacterTextSplitter(
            chunk_size=max(1, int(parameters.CHUNK_SIZE_TABLE or parameters.CHUNK_SIZE)),
            chunk_overlap=max(0, int(parameters.CHUNK_OVERLAP_TABLE or parameters.CHUNK_OVERLAP)),
            length_function=len,
            is_separator_regex=False,
        )
        self.splitter_chart = RecursiveCharacterTextSplitter(
            chunk_size=max(1, int(parameters.CHUNK_SIZE_CHART or parameters.CHUNK_SIZE)),
            chunk_overlap=max(0, int(parameters.CHUNK_OVERLAP_CHART or parameters.CHUNK_OVERLAP)),
            length_function=len,
            is_separator_regex=False,
        )
        # Backward-compatible alias used in older paths.
        self.splitter = self.splitter_text
        self.vision_client = None
        self.vision_provider = parameters.VISION_PROVIDER.lower()
        self.chart_vision_model = parameters.CHART_VISION_MODEL
        if self.vision_provider == "azure" and self.chart_vision_model.lower().startswith("gemini"):
            self.chart_vision_model = parameters.LLM_MODEL_NAME
            logger.info(
                f"VISION_PROVIDER=azure with Gemini chart model configured; "
                f"using {self.chart_vision_model} for chart analysis."
            )
        # Instance-level flag instead of modifying global parameters
        self.chart_extraction_enabled = parameters.ENABLE_CHART_EXTRACTION
        self.default_pdf_analysis_mode = normalize_pdf_analysis_mode(
            getattr(parameters, "PDF_ANALYSIS_MODE", "both")
        )
        self.default_chart_detection_backend = normalize_chart_detection_backend(
            getattr(parameters, "CHART_DETECTION_BACKEND", "pdfplumber")
        )
        if self.chart_extraction_enabled:
            self._init_vision_client()
        logger.debug(f"DocumentProcessor initialized with cache dir: {self.cache_dir}")
        logger.debug(
            "Adaptive chunking configured: text=(%s,%s) table=(%s,%s) chart=(%s,%s)",
            parameters.CHUNK_SIZE_TEXT,
            parameters.CHUNK_OVERLAP_TEXT,
            parameters.CHUNK_SIZE_TABLE,
            parameters.CHUNK_OVERLAP_TABLE,
            parameters.CHUNK_SIZE_CHART,
            parameters.CHUNK_OVERLAP_CHART,
        )
        logger.debug(f"Chart extraction: {'enabled' if self.chart_extraction_enabled else 'disabled'}")
        logger.debug("Default PDF analysis mode: %s", self.default_pdf_analysis_mode)
        logger.debug("Default chart detection backend: %s", self.default_chart_detection_backend)

    def _init_vision_client(self):
        """Initialize vision client for chart analysis based on provider."""
        try:
            client = get_vision_client()
            self.vision_client = client
            logger.info("✅ Vision client initialized")
        except VisionClientUnavailable as e:
            logger.warning(f"Vision client unavailable: {e}")
            self.chart_extraction_enabled = False

    def validate_files(self, files: List) -> bool:
        """
        Validate that uploaded files meet size and type requirements.
        
        Args:
            files: List of uploaded file objects
            
        Returns:
            bool: True if all validations pass
            
        Raises:
            ValueError: If validation fails
        """
        if not files:
            raise ValueError("No files provided")
        
        total_size = 0
        for file in files:
            # Get file size
            if hasattr(file, 'size'):
                file_size = file.size
            else:
                # Fallback: read file to get size
                try:
                    with open(file.name, 'rb') as f:
                        file_size = len(f.read())
                except Exception as e:
                    logger.error(f"Failed to determine file size for {file.name}: {e}")
                    raise ValueError(f"Cannot read file: {file.name}")
            
            # Check individual file size
            if file_size > parameters.MAX_FILE_SIZE:
                raise ValueError(
                    f"File {file.name} exceeds maximum size "
                    f"({file_size / 1024 / 1024:.2f}MB > {parameters.MAX_FILE_SIZE / 1024 / 1024:.2f}MB)"
                )
            
            # Check file type
            file_ext = Path(file.name).suffix.lower()
            if file_ext not in ALLOWED_TYPES:
                raise ValueError(
                    f"File type {file_ext} not supported. Allowed types: {ALLOWED_TYPES}"
                )
            
            total_size += file_size
        
        # Check total size
        if total_size > parameters.MAX_TOTAL_SIZE:
            raise ValueError(
                f"Total file size exceeds maximum "
                f"({total_size / 1024 / 1024:.2f}MB > {parameters.MAX_TOTAL_SIZE / 1024 / 1024:.2f}MB)"
            )
        
        logger.info(f"Validation passed for {len(files)} files (total: {total_size / 1024 / 1024:.2f}MB)")
        return True
    
    def _generate_hash(self, content: bytes) -> str:
        """Generate SHA-256 hash of file content."""
        return hashlib.sha256(content).hexdigest()

    def _generate_file_hash(self, file_path: str) -> str:
        """Generate SHA-256 hash of a file without loading it fully into memory."""
        digest = hashlib.sha256()
        with open(file_path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _cleanup_expired_chart_assets(self) -> None:
        """Delete old persisted chart assets and prune empty directories."""
        chart_assets_dir = getattr(self, "chart_assets_dir", None)
        if not chart_assets_dir or not chart_assets_dir.exists():
            return

        expire_days = max(
            1,
            int(getattr(parameters, "CHART_ASSET_EXPIRE_DAYS", parameters.CACHE_EXPIRE_DAYS)),
        )
        cutoff_ts = (datetime.now() - timedelta(days=expire_days)).timestamp()
        removed_files = 0

        try:
            for asset_file in chart_assets_dir.rglob("*"):
                if not asset_file.is_file():
                    continue
                try:
                    if asset_file.stat().st_mtime < cutoff_ts:
                        asset_file.unlink(missing_ok=True)
                        removed_files += 1
                except Exception:
                    continue

            # Clean up now-empty per-document folders.
            for folder in sorted(chart_assets_dir.rglob("*"), reverse=True):
                if folder.is_dir():
                    try:
                        folder.rmdir()
                    except OSError:
                        pass
        except Exception as e:
            logger.warning("Chart asset cleanup failed: %s", e)
            return

        if removed_files:
            logger.info(
                "Cleaned up %s expired chart assets (retention_days=%s)",
                removed_files,
                expire_days,
            )
    
    def _is_cache_valid(self, cache_path: Path) -> bool:
        """Check if a cache file exists and is still valid (not expired)."""
        if not cache_path.exists():
            logger.debug(f"Cache miss: {cache_path.name}")
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if file_age > timedelta(days=parameters.CACHE_EXPIRE_DAYS):
            logger.info(f"Cache expired (age: {file_age.days} days): {cache_path.name}")
            cache_path.unlink()
            return False
        
        logger.debug(f"Cache hit: {cache_path.name} (age: {file_age.days} days)")
        return True
    
    def _load_from_cache(self, cache_path: Path) -> List:
        """Loads chunks from a pickle file, handling potential corruption."""
        try:
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            
            if "chunks" not in data or "timestamp" not in data:
                raise KeyError("Cache file missing 'chunks' or 'timestamp' key.")
            cache_version = int(data.get("cache_version", 0))
            if cache_version != int(self.CACHE_VERSION):
                raise KeyError(
                    f"Cache version mismatch (found={cache_version}, expected={self.CACHE_VERSION})"
                )

            logger.info(f"Loaded {len(data['chunks'])} chunks from cache: {cache_path.name}")
            return data["chunks"]
        except (pickle.UnpicklingError, KeyError, EOFError) as e:
            logger.warning(f"Cache corruption detected in {cache_path.name}: {e}. Deleting cache.")
            cache_path.unlink()
            return []
        except Exception as e:
            logger.error(f"Unexpected error loading cache {cache_path.name}: {e}", exc_info=True)
            if cache_path.exists():
                cache_path.unlink()
            return []

    def _save_to_cache(self, chunks: List, cache_path: Path):
        """Saves chunks to a pickle file."""
        try:
            with open(cache_path, "wb") as f:
                pickle.dump({
                    "timestamp": datetime.now().timestamp(),
                    "cache_version": int(self.CACHE_VERSION),
                    "chunks": chunks
                }, f)
            logger.info(f"Successfully cached {len(chunks)} chunks to {cache_path.name}")
        except Exception as e:
            logger.error(f"Failed to save cache to {cache_path.name}: {e}", exc_info=True)

    def _load_pdf_by_mode(self, file_path: str) -> List[Document]:
        """
        Route PDF text extraction.

        Note:
        - Fast parsing via PyMuPDF was intentionally removed.
        - We keep PDF_PARSE_MODE values for backward compatibility, but all
          modes now use the pdfplumber fidelity parser.
        """
        mode = parameters.PDF_PARSE_MODE.lower()
        if mode != "fidelity":
            logger.warning(
                "PDF_PARSE_MODE=%s requested, but fast parser support has been removed; "
                "using fidelity parser (pdfplumber).",
                mode,
            )
        logger.info("[PROFILE] stage=parse.router mode=fidelity")
        return self._load_pdf_with_pdfplumber(file_path)

    def _select_splitter_for_doc(self, doc: Document) -> RecursiveCharacterTextSplitter:
        """Choose splitter based on content type."""
        doc_type = str(doc.metadata.get("type", "text")).lower()
        if doc_type == "chart":
            return self.splitter_chart
        if doc_type == "table":
            return self.splitter_table
        # Heuristic: many PDF pages are emitted as type=text even when dominated by tables.
        if self._is_table_like_content(doc.page_content):
            return self.splitter_table
        return self.splitter_text

    @staticmethod
    def _is_table_like_content(text: str) -> bool:
        """Heuristic detector for markdown-like / extracted table fragments."""
        sample = (text or "")[:8000]
        if not sample:
            return False
        lowered = sample.lower()
        if "### table" in lowered:
            return True
        pipe_count = sample.count("|")
        newline_count = sample.count("\n")
        if "| ---" in sample and pipe_count >= 8:
            return True
        # Weak fallback for borderless extraction that still keeps many delimiters.
        return pipe_count >= 30 and newline_count >= 4

    def _compress_table_fragment(self, text: str) -> str:
        """
        Compact repetitive table artifacts while preserving semantic table content.

        This targets repeated markdown separators/headers introduced by OCR/table extraction.
        """
        if not text:
            return ""
        lines = [ln.strip() for ln in text.splitlines() if ln and ln.strip()]
        if not lines:
            return ""

        compact_lines: List[str] = []
        prev_line = ""
        seen_table_header = False
        seen_separator = False
        header_pattern = re.compile(r"^###\s*table\s+\d+\s*\(page\s*\d+\)", re.IGNORECASE)
        separator_pattern = re.compile(r"^\|\s*[-: ]+\|\s*$")

        for line in lines:
            # Normalize pipe spacing and repeated spaces.
            normalized_line = re.sub(r"\s*\|\s*", " | ", line)
            normalized_line = re.sub(r"[ \t]+", " ", normalized_line).strip()

            # Canonicalize table header labels for dedupe stability.
            if header_pattern.match(normalized_line):
                normalized_line = "### Table"
                if seen_table_header:
                    continue
                seen_table_header = True

            # Drop duplicated consecutive lines (very common in extracted table chunks).
            if normalized_line == prev_line:
                continue

            if separator_pattern.match(normalized_line):
                if seen_separator:
                    continue
                seen_separator = True
            else:
                seen_separator = False

            compact_lines.append(normalized_line)
            prev_line = normalized_line

        return "\n".join(compact_lines).strip()

    def _build_chunk_dedupe_fingerprint(self, normalized_chunk: str, doc_type: str) -> str:
        """Build dedupe fingerprint with table-aware canonicalization."""
        if not normalized_chunk:
            return ""
        if doc_type == "table" and parameters.PRE_INGEST_TABLE_CANONICAL_DEDUPE:
            fingerprint = normalized_chunk.lower()
            # Remove table index/page counters from headings.
            fingerprint = re.sub(
                r"###\s*table\s+\d+\s*\(page\s*\d+\)",
                "### table",
                fingerprint,
                flags=re.IGNORECASE,
            )
            # Normalize whitespace around separators.
            fingerprint = re.sub(r"\s*\|\s*", "|", fingerprint)
            fingerprint = re.sub(r"\s+", " ", fingerprint).strip()
            return fingerprint
        return normalized_chunk

    def _normalize_chunk_text(self, text: str, doc_type: str) -> str:
        """
        Normalize chunk text for dedupe/storage efficiency.

        - Optional whitespace compression for text chunks.
        - Optional repetitive-line compression for table chunks.
        - Chart chunks keep line breaks to preserve structure cues.
        """
        normalized = (text or "").strip()
        if not normalized:
            return ""
        if doc_type == "table" and parameters.PRE_INGEST_TABLE_COMPRESS_REPEATS:
            normalized = self._compress_table_fragment(normalized)
        if parameters.PRE_INGEST_COMPRESS_WHITESPACE and doc_type == "text":
            normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized

    def _resolve_pdf_analysis_mode(self, mode_override: Optional[str] = None) -> str:
        """Resolve runtime PDF analysis strategy with safe fallback."""
        if mode_override is not None:
            raw_mode = str(mode_override).strip().lower()
            mode = normalize_pdf_analysis_mode(mode_override)
            known_aliases = {"pdf", "text", "vision", "image", "hybrid", "all"}
            if raw_mode not in PDF_ANALYSIS_MODES and raw_mode not in known_aliases:
                logger.warning(
                    "Unsupported pdf_analysis_mode override='%s'; defaulting to '%s'.",
                    mode_override,
                    mode,
                )
            return mode
        return self.default_pdf_analysis_mode

    def _resolve_chart_detection_backend(self, backend_override: Optional[str] = None) -> str:
        """Resolve chart detection backend with safe fallback."""
        if backend_override is not None:
            raw_backend = str(backend_override).strip().lower()
            backend = normalize_chart_detection_backend(backend_override)
            known_aliases = {"pdf", "pdf_parser", "structural", "opencv", "local"}
            if raw_backend not in CHART_DETECTION_BACKENDS and raw_backend not in known_aliases:
                logger.warning(
                    "Unsupported chart_detection_backend override='%s'; defaulting to '%s'.",
                    backend_override,
                    backend,
                )
            return backend
        return self.default_chart_detection_backend

    @staticmethod
    def _iter_contiguous_page_ranges(page_numbers: List[int]) -> List[Tuple[int, int]]:
        """Convert sorted page numbers into contiguous [start, end] ranges."""
        if not page_numbers:
            return []
        ordered = sorted({int(p) for p in page_numbers if int(p) > 0})
        if not ordered:
            return []
        ranges: List[Tuple[int, int]] = []
        start = prev = ordered[0]
        for page_num in ordered[1:]:
            if page_num == prev + 1:
                prev = page_num
                continue
            ranges.append((start, prev))
            start = prev = page_num
        ranges.append((start, prev))
        return ranges

    def _detect_chart_candidates_with_pdfplumber(
        self, file_path: str, total_pages: int
    ) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
        """
        Fast structural page triage using pdfplumber primitives.

        Returns:
            candidate_pages: pages likely containing charts/figures.
            page_signals: per-page lightweight telemetry for debugging.
        """
        import pdfplumber

        t_start = datetime.now().timestamp()
        candidate_pages: List[int] = []
        page_signals: Dict[int, Dict[str, Any]] = {}

        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    line_count = len(page.lines)
                    rect_count = len(page.rects)
                    curve_count = len(page.curves)
                    image_count = len(page.images)
                    char_count = len(page.chars)

                    table_like = (
                        (line_count >= 180 and rect_count >= 70 and curve_count <= 8)
                        or (line_count >= 260 and curve_count <= 5)
                    )
                    vector_chart_like = (
                        curve_count >= 25
                        or (curve_count >= 10 and line_count >= 20)
                        or ((line_count + curve_count) >= 220 and char_count <= 1400)
                        # Small vector charts (e.g. Figure 1.11 in EnergyandAI.pdf p37):
                        # only ~5 curves for data series + ~13 lines for axes/grid.
                        # Require both curves AND lines to avoid flagging text-only pages.
                        or (curve_count >= 5 and line_count >= 8)
                        # Bar charts rendered as pure rectangles (no curves/images):
                        # e.g. FIGURE 1.2 in Digital Progress Report — 116 rects, 0 curves.
                        # Require high rect count with moderate text — a page dominated
                        # by bars but with axis labels and title text nearby.
                        or (rect_count >= 50 and char_count <= 2900 and line_count < 100)
                    )
                    raster_chart_like = image_count >= 1 and char_count <= 1200

                    is_candidate = (vector_chart_like or raster_chart_like) and not table_like
                    if is_candidate:
                        candidate_pages.append(page_num)

                    page_signals[page_num] = {
                        "lines": line_count,
                        "rects": rect_count,
                        "curves": curve_count,
                        "images": image_count,
                        "chars": char_count,
                        "table_like": table_like,
                        "vector_chart_like": vector_chart_like,
                        "raster_chart_like": raster_chart_like,
                        "is_candidate": is_candidate,
                    }
                except Exception as page_err:
                    # Fail-open on object parsing errors so we don't miss potential charts.
                    candidate_pages.append(page_num)
                    page_signals[page_num] = {
                        "is_candidate": True,
                        "forced_candidate": True,
                        "reason": "pdfplumber_page_parse_error",
                        "error_type": type(page_err).__name__,
                    }

        t_end = datetime.now().timestamp()
        logger.info(
            "[PROFILE] stage=chart.phase1.pdfplumber_triage duration_s=%.3f pages=%s candidates=%s",
            t_end - t_start,
            total_pages,
            len(candidate_pages),
        )
        return sorted(set(candidate_pages)), page_signals
    
    def _process_file(
        self,
        file,
        pdf_analysis_mode: Optional[str] = None,
        chart_detection_backend: Optional[str] = None,
        chunk_queue: Optional["_queue_module.Queue"] = None,  # B2: streaming queue; None = disabled
    ) -> List[Document]:
        file_ext = Path(file.name).suffix.lower()
        if file_ext not in ALLOWED_TYPES:
            logger.warning(f"Skipping unsupported file type: {file.name}")
            return []
        try:
            documents = []
            if file_ext == '.pdf':
                analysis_mode = self._resolve_pdf_analysis_mode(pdf_analysis_mode)
                resolved_chart_backend = self._resolve_chart_detection_backend(chart_detection_backend)
                logger.info(
                    "[PROFILE] stage=parse.strategy mode=%s file=%s",
                    analysis_mode,
                    Path(file.name).name,
                )
                logger.info(
                    "[PROFILE] stage=chart.detect.strategy backend=%s file=%s",
                    resolved_chart_backend,
                    Path(file.name).name,
                )
                import concurrent.futures
                def run_text_parse():
                    t_parse_start = datetime.now().timestamp()
                    docs = self._load_pdf_by_mode(file.name)
                    t_parse_end = datetime.now().timestamp()
                    parse_duration = t_parse_end - t_parse_start
                    logger.info(f"[PROFILE] stage=parse.total duration_s={parse_duration:.3f} file={Path(file.name).name}")
                    latency_monitor.record(
                        stage="parse.total",
                        duration_s=parse_duration,
                        metadata={"file": Path(file.name).name},
                    )
                    return docs
                def run_charts():
                    logger.info(
                        f"chart_extraction_enabled={self.chart_extraction_enabled}, "
                        f"vision_client={self.vision_client is not None}, provider={self.vision_provider}"
                    )
                    if self.chart_extraction_enabled and self.vision_client:
                        return self._extract_charts_from_pdf(
                            file.name,
                            analyze_all_pages=(analysis_mode == "vision_only"),
                            detection_backend=resolved_chart_backend,
                        )
                    if analysis_mode == "vision_only":
                        provider_name = str(self.vision_provider or "none").lower()
                        if provider_name == "none":
                            raise RuntimeError(
                                "VISION_ONLY mode requires a vision provider. "
                                "Set VISION_PROVIDER to 'azure' or 'google'."
                            )
                        raise RuntimeError(
                            "VISION_ONLY mode requires chart/page vision extraction, "
                            "but vision client is unavailable."
                        )
                    return []
                if analysis_mode == "pdf_only":
                    documents = run_text_parse() or []
                elif analysis_mode == "vision_only":
                    documents = run_charts() or []
                else:
                    try:
                        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                            future_pdf = executor.submit(run_text_parse)
                            future_charts = executor.submit(run_charts)
                            try:
                                docs = future_pdf.result()
                            except MemoryError as e:
                                logger.error(f"Out of memory in text parsing thread: {e}. Falling back to fidelity parser.")
                                docs = self._load_pdf_with_pdfplumber(file.name)
                            try:
                                chart_docs = future_charts.result()
                            except MemoryError as e:
                                logger.error(f"Out of memory in chart extraction thread: {e}. Falling back to sequential.")
                                chart_docs = self._extract_charts_from_pdf(
                                    file.name,
                                    detection_backend=resolved_chart_backend,
                                )
                            documents = docs or []
                            if chart_docs:
                                documents.extend(chart_docs)
                                logger.info(f"📊 Added {len(chart_docs)} chart descriptions to {file.name}")
                    except MemoryError as e:
                        logger.error(f"Out of memory in parallel PDF processing: {e}. Falling back to sequential.")
                        documents = self._load_pdf_by_mode(file.name)
                        if self.chart_extraction_enabled and self.vision_client:
                            chart_docs = self._extract_charts_from_pdf(
                                file.name,
                                detection_backend=resolved_chart_backend,
                            )
                            if chart_docs:
                                documents.extend(chart_docs)
                                logger.info(f"📊 Added {len(chart_docs)} chart descriptions to {file.name}")
            else:
                from langchain_community.document_loaders import (
                    Docx2txtLoader,
                    TextLoader,
                )
                loader_map = {
                    '.docx': Docx2txtLoader,
                    '.txt': TextLoader,
                    '.md': TextLoader,
                }
                loader_class = loader_map.get(file_ext)
                if not loader_class:
                    logger.warning(f"No loader found for {file_ext}")
                    return []
                logger.info(f"Loading {file_ext} file: {file.name}")
                loader = loader_class(file.name)
                documents = loader.load()
            if not documents:
                logger.warning(f"No content extracted from {file.name}")
                return []
            all_chunks = []
            # --- STABLE FILE HASHING ---
            file_hash = self._generate_file_hash(file.name)  # Stable hash by file content
            stable_source = f"{Path(file.name).name}::{file_hash}"
            t_chunk_start = datetime.now().timestamp()
            seen_chunk_hashes = set()
            dropped_tiny = 0
            dropped_dupes = 0
            for i, doc in enumerate(documents):
                doc_type = str(doc.metadata.get("type", "text")).lower()
                splitter = self._select_splitter_for_doc(doc)
                page_chunks = splitter.split_text(doc.page_content)
                for j, chunk in enumerate(page_chunks):
                    chunk_doc_type = doc_type
                    if chunk_doc_type == "text" and self._is_table_like_content(chunk):
                        chunk_doc_type = "table"

                    normalized_chunk = self._normalize_chunk_text(chunk, doc_type=chunk_doc_type)
                    min_chars = (
                        int(parameters.PRE_INGEST_MIN_TABLE_CHUNK_CHARS)
                        if chunk_doc_type == "table"
                        else int(parameters.PRE_INGEST_MIN_CHUNK_CHARS)
                    )
                    if len(normalized_chunk) < max(0, min_chars):
                        dropped_tiny += 1
                        continue

                    if parameters.PRE_INGEST_DEDUPE_ENABLED:
                        dedupe_fingerprint = self._build_chunk_dedupe_fingerprint(
                            normalized_chunk,
                            doc_type=chunk_doc_type,
                        )
                        chunk_hash = hashlib.sha256(dedupe_fingerprint.encode("utf-8")).hexdigest()
                        if chunk_hash in seen_chunk_hashes:
                            dropped_dupes += 1
                            continue
                        seen_chunk_hashes.add(chunk_hash)

                    chunk_id = f"txt_{file_hash}_{doc.metadata.get('page', i + 1)}_{j}"
                    chunk_metadata = {
                        "source": stable_source,
                        "page": doc.metadata.get("page", i + 1),
                        "type": chunk_doc_type,
                        "chunk_id": chunk_id,
                    }
                    if doc.metadata.get("chunk_id"):
                        chunk_metadata["origin_chunk_id"] = doc.metadata.get("chunk_id")
                    # Preserve high-value extraction metadata for downstream analytics/benchmarking.
                    for key in (
                        "extraction_method",
                        "detection_confidence",
                        "batch_size",
                        "loader",
                        "tables_count",
                        "chart_image_path",
                    ):
                        if key in doc.metadata:
                            chunk_metadata[key] = doc.metadata.get(key)
                    chunk_doc = Document(
                        page_content=normalized_chunk,
                        metadata=chunk_metadata
                    )
                    all_chunks.append(chunk_doc)
                    if chunk_queue is not None:          # B2: forward chunk to consumer thread
                        try:
                            chunk_queue.put(chunk_doc, timeout=5)
                        except _queue_module.Full:
                            # Consumer thread must have died; abort streaming gracefully
                            logger.warning("[STREAMING] Producer queue full — consumer may be dead; aborting")
                            try:
                                chunk_queue.put(None, timeout=1)   # emergency sentinel
                            except Exception:
                                pass
                            raise RuntimeError("Streaming queue full: consumer thread is not draining")
            t_chunk_end = datetime.now().timestamp()
            logger.info(f"[PROFILE] stage=chunking.total duration_s={t_chunk_end - t_chunk_start:.3f} chunks={len(all_chunks)}")
            if dropped_tiny or dropped_dupes:
                logger.info(
                    "[PROFILE] stage=chunking.optimization dropped_tiny=%d dropped_dupes=%d kept=%d",
                    dropped_tiny,
                    dropped_dupes,
                    len(all_chunks),
                )
                 
            logger.info(f"Processed {file.name}: {len(documents)} page(s) → {len(all_chunks)} chunk(s)")
            if chunk_queue is not None:          # B2: signal consumer that all chunks are sent
                chunk_queue.put(None)
            return all_chunks
        except ImportError as e:
            logger.error(f"Required loader not installed for {file_ext}: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to process {file.name}: {e}", exc_info=True)
            raise

    def _extract_charts_from_pdf(
        self,
        file_path: str,
        analyze_all_pages: bool = False,
        detection_backend: Optional[str] = None,
    ) -> List[Document]:
        """
        Extract and analyze charts/graphs from PDF with true batch processing and parallelism.
        PHASE 1: Candidate-page detection (pdfplumber triage OR optimized OpenCV)
        PHASE 2: Parallel vision batch analysis (I/O-bound, uses ThreadPoolExecutor)

        Args:
            file_path: Path to PDF file.
            analyze_all_pages: When True, bypass local chart detection and send every
                page image to vision analysis. Used by vision-only parsing mode.
            detection_backend: Phase-1 chart candidate backend
                (pdfplumber | opencv_optimized).
        """
        file_bytes = Path(file_path).read_bytes()
        file_hash = self._generate_hash(file_bytes)
        stable_source = f"{Path(file_path).name}::{file_hash}"
        # Detect front-matter page offset (same as text parser) so chart
        # page numbers match the printed numbers in the PDF.
        chart_page_offset = detect_page_label_offset(file_path, file_hash=file_hash)
        def deduplicate_charts_by_title(chart_chunks):
            seen_titles = set()
            unique_chunks = []
            import re
            for chunk in chart_chunks:
                match = re.search(r"\*\*Title\*\*:\s*(.+)", chunk.page_content)
                title = match.group(1).strip() if match else None
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_chunks.append(chunk)
                elif not title:
                    unique_chunks.append(chunk)
            return unique_chunks
        try:
            from pdf2image import convert_from_path
            from PIL import Image
            import pdfplumber
            import tempfile
            import os
            
            resolved_backend = self._resolve_chart_detection_backend(detection_backend)
            if resolved_backend == "opencv_optimized" and not parameters.CHART_USE_LOCAL_DETECTION:
                logger.warning(
                    "CHART_USE_LOCAL_DETECTION=false with backend=opencv_optimized; "
                    "falling back to pdfplumber backend."
                )
                resolved_backend = "pdfplumber"
            use_local = parameters.CHART_USE_LOCAL_DETECTION and resolved_backend == "opencv_optimized"
            if use_local:
                try:
                    from content_analyzer.visual_detector import LocalChartDetector  # noqa: F401
                    logger.info(
                        "📊 [BATCH MODE] OpenCV optimized detection enabled "
                        "(backend=%s, max_dim=%s, workers=%s, process_pool=%s)",
                        resolved_backend,
                        getattr(parameters, "CHART_OPENCV_MAX_DIM", 700),
                        getattr(parameters, "CHART_OPENCV_WORKERS", 4),
                        getattr(parameters, "CHART_OPENCV_USE_PROCESS_POOL", False),
                    )
                except ImportError:
                    logger.warning("OpenCV chart detector unavailable; switching detection backend to pdfplumber")
                    resolved_backend = "pdfplumber"
                    use_local = False
            
            # Track statistics
            stats = {
                'pages_scanned': 0,
                'charts_detected_local': 0,
                'charts_analyzed_vision': 0,
                'api_calls_saved': 0,
                'batch_api_calls': 0,
                'persisted_assets_count': 0,
                'persist_fallback_copy_count': 0,
                'persist_failed_temp_fallback_count': 0,
            }
            
            # Get PDF page count
            with pdfplumber.open(file_path) as pdf:
                total_pages = len(pdf.pages)
            
            logger.info(f"Processing {total_pages} pages for chart detection...")
            
            # Create temp directory for chart images
            temp_dir = tempfile.mkdtemp(prefix='charts_')
            chart_assets_dir = self.chart_assets_dir / file_hash
            chart_assets_dir.mkdir(parents=True, exist_ok=True)
            detected_charts = []  # [(page_num, image_path, detection_result), ...]
            
            try:
                # === PHASE 1: PARALLEL LOCAL CHART DETECTION (CPU-BOUND) ===
                logger.info("Phase 1: Detecting charts and caching to disk...")
                phase1_start = datetime.now().timestamp()
                detection_batch_size = 20  # stream pages to keep memory bounded

                def _safe_delete(path: str) -> None:
                    try:
                        if path and os.path.exists(path):
                            os.remove(path)
                    except Exception:
                        logger.debug(f"Could not delete temporary image: {path}")

                def _load_page_images_as_paths(batch_start: int, batch_end: int):
                    """Convert a PDF page range into on-disk JPEG paths."""
                    page_tasks = []
                    try:
                        # Preferred path: let pdf2image write files directly to disk.
                        image_paths = convert_from_path(
                            file_path,
                            dpi=parameters.CHART_DPI,
                            first_page=batch_start,
                            last_page=batch_end,
                            fmt='jpeg',
                            output_folder=temp_dir,
                            paths_only=True,
                            jpegopt={'quality': 85, 'optimize': True},
                        )
                        for idx, image_path in enumerate(image_paths):
                            page_num = batch_start + idx
                            stats['pages_scanned'] += 1
                            page_tasks.append((page_num, str(image_path)))
                        return page_tasks
                    except TypeError:
                        # Compatibility path for older pdf2image versions without paths_only.
                        images = convert_from_path(
                            file_path,
                            dpi=parameters.CHART_DPI,
                            first_page=batch_start,
                            last_page=batch_end,
                            fmt='jpeg',
                            jpegopt={'quality': 85, 'optimize': True},
                        )
                        for idx, image in enumerate(images):
                            page_num = batch_start + idx
                            stats['pages_scanned'] += 1
                            try:
                                max_dimension = parameters.CHART_MAX_IMAGE_SIZE
                                if max(image.size) > max_dimension:
                                    ratio = max_dimension / max(image.size)
                                    new_size = tuple(int(dim * ratio) for dim in image.size)
                                    image = image.resize(new_size, Image.Resampling.LANCZOS)
                                image_path = os.path.join(temp_dir, f'page_{page_num}.jpg')
                                image.save(image_path, 'JPEG', quality=85)
                                page_tasks.append((page_num, image_path))
                            finally:
                                try:
                                    image.close()
                                except Exception:
                                    pass
                        del images
                        return page_tasks

                def _store_detected_chart(page_num: int, page_image_path: str, detection_result: dict) -> None:
                    has_chart = bool(detection_result.get("has_chart"))
                    confidence = float(detection_result.get("confidence", 0.0) or 0.0)
                    if not has_chart:
                        logger.debug(f"Page {page_num}: No chart detected (skipping)")
                        stats['api_calls_saved'] += 1
                        _safe_delete(page_image_path)
                        return
                    if confidence < parameters.CHART_MIN_CONFIDENCE:
                        logger.debug(f"Page {page_num}: Low confidence ({confidence:.0%}), skipping")
                        stats['api_calls_saved'] += 1
                        _safe_delete(page_image_path)
                        return

                    if _is_full_page_vision_detection(detection_result):
                        logger.debug(f"📄 Vision-only page queued for analysis: page {page_num}")
                    else:
                        logger.info(f"📈 Chart detected on page {page_num} (confidence: {confidence:.0%})")
                        stats['charts_detected_local'] += 1

                    source_path = str(page_image_path)
                    page_mode = "vision_page" if _is_full_page_vision_detection(detection_result) else "chart"
                    asset_path = (chart_assets_dir / f"{page_mode}_page_{int(page_num):04d}.jpg").resolve()
                    persisted_path = str(asset_path)
                    persist_failed = False

                    try:
                        if os.path.abspath(source_path) != os.path.abspath(persisted_path):
                            os.replace(source_path, persisted_path)
                        else:
                            persisted_path = source_path
                    except Exception:
                        try:
                            shutil.copy2(source_path, persisted_path)
                            _safe_delete(source_path)
                            stats['persist_fallback_copy_count'] += 1
                        except Exception as copy_err:
                            logger.warning(
                                "Failed to persist chart asset for page %s (%s). Using temp path.",
                                page_num,
                                copy_err,
                            )
                            persisted_path = source_path
                            persist_failed = True

                    if not persist_failed:
                        persisted_exists = False
                        try:
                            persisted_exists = Path(persisted_path).exists()
                        except Exception:
                            persisted_exists = False
                        if persisted_exists:
                            persisted_path = str(Path(persisted_path).resolve())
                            stats['persisted_assets_count'] += 1
                        else:
                            logger.warning(
                                "Persist target missing after write for page %s (%s). Falling back to source path.",
                                page_num,
                                persisted_path,
                            )
                            persisted_path = source_path
                            stats['persist_failed_temp_fallback_count'] += 1
                    else:
                        stats['persist_failed_temp_fallback_count'] += 1

                    detected_charts.append((page_num, persisted_path, detection_result))

                detected_charts = []
                if analyze_all_pages:
                    logger.info(
                        "Phase 1 (vision-only): bypassing local chart detection and queuing all %s pages for vision analysis.",
                        total_pages,
                    )
                    for batch_start in range(1, total_pages + 1, detection_batch_size):
                        batch_end = min(batch_start + detection_batch_size - 1, total_pages)
                        logger.debug(f"Vision-only batch enqueue: pages {batch_start}-{batch_end}")
                        try:
                            page_image_tuples = _load_page_images_as_paths(batch_start, batch_end)
                        except Exception as e:
                            logger.warning(f"Failed to render pages {batch_start}-{batch_end}: {e}")
                            continue
                        for page_num, page_image_path in page_image_tuples:
                            _store_detected_chart(
                                page_num=page_num,
                                page_image_path=page_image_path,
                                detection_result={
                                    "has_chart": True,
                                    "confidence": 1.0,
                                    "chart_types": ["full_page_vision"],
                                    "description": "Vision-only page analysis",
                                    "features": {"full_page_vision": True},
                                },
                            )
                        del page_image_tuples
                        gc.collect()
                elif resolved_backend == "pdfplumber":
                    logger.info(
                        "Phase 1 (pdfplumber): structural candidate triage enabled "
                        "(backend=%s, pages=%s)",
                        resolved_backend,
                        total_pages,
                    )
                    candidate_pages, page_signals = self._detect_chart_candidates_with_pdfplumber(
                        file_path=file_path,
                        total_pages=total_pages,
                    )
                    stats["api_calls_saved"] += max(0, total_pages - len(candidate_pages))
                    if not candidate_pages:
                        logger.info("No chart candidates selected by pdfplumber triage.")
                    else:
                        logger.info(
                            "pdfplumber triage selected %s/%s pages for vision analysis.",
                            len(candidate_pages),
                            total_pages,
                        )
                        for batch_start, batch_end in self._iter_contiguous_page_ranges(candidate_pages):
                            try:
                                page_image_tuples = _load_page_images_as_paths(batch_start, batch_end)
                            except Exception as e:
                                logger.warning(f"Failed to render candidate pages {batch_start}-{batch_end}: {e}")
                                continue

                            for page_num, page_image_path in page_image_tuples:
                                signal = page_signals.get(page_num, {})
                                chart_types = ["pdfplumber_candidate"]
                                if signal.get("forced_candidate"):
                                    chart_types.append("pdfplumber_parse_error")
                                _store_detected_chart(
                                    page_num=page_num,
                                    page_image_path=page_image_path,
                                    detection_result={
                                        "has_chart": True,
                                        "confidence": float(signal.get("confidence", 0.75)),
                                        "chart_types": chart_types,
                                        "description": str(
                                            signal.get("reason", "pdfplumber structural chart candidate")
                                        ),
                                        "features": {"pdfplumber_candidate": True, **signal},
                                    },
                                )
                            del page_image_tuples
                            gc.collect()
                    # pages_scanned is incremented while rendering candidate pages;
                    # expose full-triage coverage in logs/metrics for observability.
                    stats["pages_scanned"] = total_pages
                elif use_local and parameters.CHART_SKIP_GEMINI_DETECTION:
                    max_workers = max(1, int(getattr(parameters, "CHART_OPENCV_WORKERS", 4) or 4))
                    use_process_pool = bool(getattr(parameters, "CHART_OPENCV_USE_PROCESS_POOL", False))
                    logger.info(
                        "Phase 1 (opencv_optimized): local detection workers=%s process_pool=%s max_dim=%s",
                        max_workers,
                        use_process_pool,
                        getattr(parameters, "CHART_OPENCV_MAX_DIM", 700),
                    )

                    process_pool_available = use_process_pool
                    for batch_start in range(1, total_pages + 1, detection_batch_size):
                        batch_end = min(batch_start + detection_batch_size - 1, total_pages)
                        logger.debug(f"Processing detection batch: pages {batch_start}-{batch_end}")

                        try:
                            page_image_tuples = _load_page_images_as_paths(batch_start, batch_end)
                        except Exception as e:
                            logger.warning(f"Failed to render pages {batch_start}-{batch_end}: {e}")
                            continue

                        if not page_image_tuples:
                            continue

                        results = []
                        # Use ThreadPool directly — ProcessPool on Windows
                        # causes MemoryError due to google.genai Pydantic
                        # schema construction in spawned worker subprocesses.
                        # detect_chart_on_page_path is I/O+OpenCV bound and
                        # releases the GIL during C-level work.
                        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                            results = list(executor.map(detect_chart_on_page_path, page_image_tuples))

                        for page_num, page_image_path, detection_result in results:
                            _store_detected_chart(
                                page_num=page_num,
                                page_image_path=page_image_path,
                                detection_result=detection_result or {},
                            )

                        del page_image_tuples
                        del results
                        gc.collect()
                        logger.debug(f"Batch {batch_start}-{batch_end} complete, memory released")
                else:
                    logger.info(
                        "No valid chart detection backend available "
                        "(backend=%s, CHART_USE_LOCAL_DETECTION=%s); skipping Phase 1 detections.",
                        resolved_backend,
                        parameters.CHART_USE_LOCAL_DETECTION,
                    )

                logger.info(f"Phase 1 complete: {len(detected_charts)} charts detected and cached")
                logger.info(
                    "[PROFILE] stage=chart.assets.persist persisted=%s fallback_copy=%s failed_temp_fallback=%s",
                    stats['persisted_assets_count'],
                    stats['persist_fallback_copy_count'],
                    stats['persist_failed_temp_fallback_count'],
                )
                phase1_duration = datetime.now().timestamp() - phase1_start
                latency_monitor.record(
                    stage="chart.phase1",
                    duration_s=phase1_duration,
                    metadata={"pages": total_pages, "detected_charts": len(detected_charts)},
                )
                
                # === PHASE 2: PARALLEL VISION BATCH ANALYSIS (I/O-BOUND) ===
                if not detected_charts or not self.vision_client:
                    return []
                
                logger.info(f"Phase 2: Batch analyzing {len(detected_charts)} charts...")
                phase2_start = datetime.now().timestamp()
                chart_documents = []
                
                if parameters.CHART_ENABLE_BATCH_ANALYSIS and len(detected_charts) > 1:
                    # Batch processing with parallel vision API calls
                    vision_batch_size = parameters.CHART_GEMINI_BATCH_SIZE
                    batches = [detected_charts[i:i + vision_batch_size] for i in range(0, len(detected_charts), vision_batch_size)]

                    # Prepare batch tuples with batch_num and total_batches
                    batch_tuples = [
                        (
                            batch,
                            idx + 1,
                            len(batches),
                            self.vision_client,
                            self.vision_provider,
                            self.chart_vision_model,
                            stable_source,
                            parameters,
                            stats,
                        )
                        for idx, batch in enumerate(batches)
                    ]
                    results = [None] * len(batches)
                    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
                        future_to_idx = {executor.submit(analyze_batch, batch_tuple): idx for idx, batch_tuple in enumerate(batch_tuples)}
                        for future in concurrent.futures.as_completed(future_to_idx):
                            idx = future_to_idx[future]
                            try:
                                batch_idx, batch_docs = future.result()
                                results[batch_idx] = batch_docs
                            except Exception as exc:
                                logger.error(f"Batch {idx} generated an exception: {exc}")
                    # Flatten results and filter out None
                    chart_index = 0
                    for batch_docs in results:
                        if batch_docs:
                            for doc in batch_docs:
                                doc.metadata["chunk_id"] = f"{file_hash}_{doc.metadata.get('page', 0)}_{chart_index}"
                                chart_documents.append(doc)
                                chart_index += 1
                else:
                    # Sequential processing (batch disabled or single chart)
                    for chart_index, (page_num, image_path, detection_result) in enumerate(detected_charts):
                        try:
                            is_full_page_mode = _is_full_page_vision_detection(detection_result or {})
                            if is_full_page_mode:
                                extraction_prompt = """Analyze this document page in comprehensive detail:
                                **Page Summary**: [what this page is about]
                                **Sections/Headings**: [visible structure]
                                **Key Facts**: [numbers, dates, entities, claims]
                                **Table-like Data**: [rows/columns and values if visible]
                                **Chart/Graphic Insights**: [if any visuals exist]
                                **Important Notes**: [ambiguities / low-confidence reads]
                                """
                            else:
                                extraction_prompt = """Analyze this chart/graph in comprehensive detail:
                                **Chart Type**: [type]
                                **Title**: [title]
                                **Axes**: [X and Y labels/units]
                                **Data Points**: [extract all visible data]
                                **Legend**: [series/categories]
                                **Trends**: [key patterns and insights]
                                **Key Values**: [max, min, significant]
                                **Context**: [annotations or notes]
                                """
                            analysis_text = analyze_chart_images(
                                client=self.vision_client,
                                image_paths=[image_path],
                                prompt=extraction_prompt,
                                model_name=self.chart_vision_model,
                                max_output_tokens=parameters.CHART_MAX_TOKENS,
                                provider=self.vision_provider,
                            )
                            chart_types_str = ", ".join((detection_result or {}).get('chart_types', [])) or "Unknown"
                            detection_confidence = float((detection_result or {}).get('confidence', 0.0) or 0.0)
                            detection_method = _detection_method_label(
                                detection_result=detection_result or {},
                                vision_provider=self.vision_provider,
                                is_batch=False,
                            )
                            heading = "### 📄 Vision Page Analysis" if is_full_page_mode else "### 📊 Chart Analysis"
                            doc_type = "text" if is_full_page_mode else "chart"
                            extraction_method = "vision_only_full_page" if is_full_page_mode else "hybrid_sequential"
                            chart_doc = Document(
                                page_content=f"""{heading} (Page {page_num})\n\n**Detection Method**: {detection_method}\n**Local Confidence**: {detection_confidence:.0%}\n**Detected Types**: {chart_types_str}\n\n---\n\n{analysis_text}\n""",
                                metadata={
                                    "source": stable_source,
                                    "page": page_num,
                                    "type": doc_type,
                                    "extraction_method": extraction_method,
                                    "chart_image_path": str(image_path),
                                    "chunk_id": f"{file_hash}_{page_num}_{chart_index}"
                                }
                            )
                            chart_documents.append(chart_doc)
                            stats['charts_analyzed_vision'] += 1
                            logger.info(f"✅ Analyzed chart on page {page_num}")
                        except Exception as e:
                            logger.error(f"Failed to analyze page {page_num}: {e}")
                
                # Log statistics
                if (resolved_backend in {"pdfplumber", "opencv_optimized"} or analyze_all_pages) and parameters.CHART_SKIP_GEMINI_DETECTION:
                    cost_saved = stats['api_calls_saved'] * 0.0125
                    actual_cost = stats['batch_api_calls'] * 0.0125 if stats['batch_api_calls'] > 0 else stats['charts_analyzed_vision'] * 0.0125
                    
                    if stats['batch_api_calls'] > 0:
                        efficiency = stats['charts_analyzed_vision'] / stats['batch_api_calls']
                    else:
                        efficiency = 1.0
                    
                    logger.info(f"""
📊 Chart Extraction Complete (BATCH MODE):
   Detection backend: {resolved_backend}
   Pages scanned: {stats['pages_scanned']}
   Charts detected (local): {stats['charts_detected_local']}
   Charts analyzed (Vision): {stats['charts_analyzed_vision']}
   Batch API calls: {stats['batch_api_calls']}
   Charts per API call: {efficiency:.1f}
   API calls saved (detection): {stats['api_calls_saved']}
   Estimated cost savings: ${cost_saved:.3f}
   Actual API cost: ${actual_cost:.3f}
""")
                
                # After chart_documents is created (batch or sequential), deduplicate by title:
                chart_documents = deduplicate_charts_by_title(chart_documents)

                # ── Apply front-matter page offset to chart documents ──────
                # Internally, chart detection uses raw PDF page indices.
                # Adjust to printed page numbers so they align with text chunks.
                if chart_page_offset:
                    for cdoc in chart_documents:
                        raw_pg = cdoc.metadata.get("page")
                        if raw_pg is None:
                            continue
                        # Coerce to int safely — raw_pg may be int, str, or list
                        try:
                            raw_pg_int = int(raw_pg[0] if isinstance(raw_pg, list) else raw_pg)
                        except (TypeError, ValueError, IndexError):
                            continue
                        printed_pg = max(1, raw_pg_int - chart_page_offset)
                        # Replace in page_content BEFORE updating metadata so the
                        # f-string matches the original raw value in the text.
                        cdoc.page_content = cdoc.page_content.replace(
                            f"(Page {raw_pg_int})", f"(Page {printed_pg})", 1
                        )
                        cdoc.metadata["page"] = printed_pg
                    logger.info(
                        "[PAGE_OFFSET] Applied offset=%d to %d chart documents",
                        chart_page_offset, len(chart_documents),
                    )

                phase2_duration = datetime.now().timestamp() - phase2_start
                latency_monitor.record(
                    stage="chart.phase2",
                    duration_s=phase2_duration,
                    metadata={"charts_analyzed": len(chart_documents)},
                )
                
                return chart_documents
            
            finally:
                # Only clean up after all analysis is done
                try:
                    shutil.rmtree(temp_dir)
                    logger.debug(f"Cleaned up temp directory: {temp_dir}")
                except Exception as e:
                    logger.warning(f"Failed to clean temp directory {temp_dir}: {e}")
        except ImportError as e:
            logger.warning(f"Dependencies missing for chart extraction: {e}")
            return []
        except MemoryError as e:
            logger.error(f"Out of memory while processing {file_path}. Try reducing DPI or batch size.")
            return []
        except Exception as e:
            logger.error(f"Chart extraction failed for {file_path}: {e}", exc_info=True)
            return []

    def _load_pdf_with_pdfplumber(self, file_path: str) -> List[Document]:
        """
        Load PDF using pdfplumber for text and table extraction.
        
        Uses multiple table detection strategies for complex tables.
        """
        import pdfplumber
        
        t_start = datetime.now().timestamp()
        logger.info(f"[PDFPLUMBER] Processing: {file_path}")
        file_hash = self._generate_file_hash(file_path)
        stable_source = f"{Path(file_path).name}::{file_hash}"

        # Detect front-matter page offset so stored page numbers match
        # the printed page numbers in the PDF (e.g. offset=23 means
        # raw page 35 → printed page 12).
        page_offset = detect_page_label_offset(file_path, file_hash=file_hash)
        
        # Strategy 1: Line-based (default) - for tables with visible borders
        default_parameters = {}
        
        # Strategy 2: Text-based - for borderless tables with aligned text
        text_parameters = {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 5,
            "join_tolerance": 5,
            "edge_min_length": 3,
            "min_words_vertical": 2,
            "min_words_horizontal": 1,
            "text_tolerance": 3,
            "intersection_tolerance": 5,
        }
        
        # Strategy 3: Lines + text hybrid - for complex tables
        hybrid_parameters = {
            "vertical_strategy": "lines_strict",
            "horizontal_strategy": "text",
            "snap_tolerance": 5,
            "join_tolerance": 5,
            "min_words_horizontal": 1,
        }

        parse_workers = max(1, int(parameters.PDF_PARSE_PAGE_RANGE_WORKERS or 1))
        page_range_size = max(1, int(parameters.PDF_PARSE_PAGE_RANGE_SIZE or 24))

        if parse_workers > 1:
            try:
                with pdfplumber.open(file_path) as pdf:
                    total_pages = len(pdf.pages)
                if total_pages > 1:
                    logger.info(
                        "[PDFPLUMBER] Parallel page-range parse enabled (workers=%s, range_size=%s, pages=%s)",
                        parse_workers,
                        page_range_size,
                        total_pages,
                    )

                    ranges: List[Tuple[int, int]] = []
                    start = 1
                    while start <= total_pages:
                        end = min(total_pages, start + page_range_size - 1)
                        ranges.append((start, end))
                        start = end + 1

                    worker_inputs = [
                        (
                            file_path,
                            range_start,
                            range_end,
                            default_parameters,
                            text_parameters,
                            hybrid_parameters,
                        )
                        for range_start, range_end in ranges
                    ]

                    range_payloads: List[Dict[str, Any]] = []
                    # ── Timeout-safe ThreadPoolExecutor ────────────────
                    # Why ThreadPool instead of ProcessPool:
                    #   On Windows, ProcessPoolExecutor uses 'spawn' which
                    #   re-imports the ENTIRE module tree in each worker
                    #   subprocess.  The google.genai.types module has
                    #   ~14k lines of Pydantic models whose schema
                    #   construction causes MemoryError, silently killing
                    #   the worker and hanging the parent process.
                    #   ThreadPool avoids this entirely. pdfplumber is
                    #   I/O-bound and pdfminer releases the GIL during
                    #   C-level work, so threads are appropriate here.
                    #
                    # Why manual lifecycle instead of `with` block:
                    #   The context manager calls shutdown(wait=True) on
                    #   exit, which blocks forever if a worker thread
                    #   hangs in pdfminer's C-level layout parser.
                    #   Manual shutdown(wait=False, cancel_futures=True)
                    #   lets us abandon stuck workers and fall through to
                    #   the sequential fallback path.
                    _PPE_TIMEOUT_S = int(os.environ.get("PDF_PARSE_RANGE_TIMEOUT_S", "300"))
                    executor = concurrent.futures.ThreadPoolExecutor(max_workers=parse_workers)
                    try:
                        futures = [
                            executor.submit(extract_pdf_page_range_payload, payload)
                            for payload in worker_inputs
                        ]

                        # Wait for ALL workers, but cap total wall-clock.
                        # If any worker is stuck in pdfminer C code, no
                        # Python exception is raised — only the timeout
                        # can rescue us.
                        done, not_done = concurrent.futures.wait(
                            futures,
                            timeout=_PPE_TIMEOUT_S,
                            return_when=concurrent.futures.ALL_COMPLETED,
                        )

                        # Collect results from workers that finished.
                        for future in done:
                            try:
                                range_payloads.append(future.result(timeout=0))
                            except Exception as worker_err:
                                logger.warning(
                                    "[PDFPLUMBER] Page-range worker error: %s", worker_err
                                )

                        # If any workers are still stuck, log and raise so
                        # the outer except triggers the sequential fallback.
                        if not_done:
                            logger.error(
                                "[PDFPLUMBER] %d/%d page-range worker(s) timed out after %ds",
                                len(not_done), len(futures), _PPE_TIMEOUT_S,
                            )
                            for f in not_done:
                                f.cancel()  # no-op for running tasks, but marks pending ones
                            raise TimeoutError(
                                f"{len(not_done)} page-range worker(s) hung — "
                                f"falling back to sequential parser"
                            )
                    finally:
                        # shutdown(wait=False) returns immediately even if
                        # workers are stuck; cancel_futures=True (Python 3.9+)
                        # prevents queued-but-not-started tasks from running.
                        executor.shutdown(wait=False, cancel_futures=True)

                    pages_payload: List[Dict[str, Any]] = []
                    text_fallback_success = 0
                    text_fallback_fail = 0
                    for payload in range_payloads:
                        pages_payload.extend(payload.get("pages", []))
                        text_fallback_success += int(payload.get("text_fallback_success", 0) or 0)
                        text_fallback_fail += int(payload.get("text_fallback_fail", 0) or 0)

                    pages_payload.sort(key=lambda item: int(item.get("page_num", 0)))
                    all_content: List[Document] = []
                    total_tables = 0
                    for page_payload in pages_payload:
                        raw_page = int(page_payload.get("page_num", 0) or 0)
                        if raw_page < 1:
                            continue
                        # Apply front-matter offset: raw page 35 → printed page 12
                        page_num = max(1, raw_page - page_offset) if page_offset else raw_page
                        page_content = [f"## Page {page_num}"]
                        for table in page_payload.get("tables", []) or []:
                            total_tables += 1
                            md_table = self._table_to_markdown(table, page_num, total_tables)
                            if md_table:
                                page_content.append(md_table)

                        text = (page_payload.get("text") or "").strip()
                        if text:
                            page_content.append(text)

                        if len(page_content) > 1:
                            combined = "\n\n".join(page_content)
                            chunk_id = f"txt_{file_hash}_{page_num}_0"
                            all_content.append(
                                Document(
                                    page_content=combined,
                                    metadata={
                                        "source": stable_source,
                                        "page": page_num,
                                        "loader": "pdfplumber",
                                        "tables_count": total_tables,
                                        "type": "text",
                                        "chunk_id": chunk_id,
                                    },
                                )
                            )

                    t_end = datetime.now().timestamp()
                    if text_fallback_success or text_fallback_fail:
                        logger.info(
                            "[PDFPLUMBER] Text fallback summary: pypdf_success=%s unrecovered=%s",
                            text_fallback_success,
                            text_fallback_fail,
                        )
                    logger.info(f"[PDFPLUMBER] Extracted {len(all_content)} chunks, {total_tables} tables")
                    logger.info(
                        f"[PROFILE] stage=parse.pdfplumber duration_s={t_end - t_start:.3f} "
                        f"pages={total_pages} docs={len(all_content)} tables={total_tables}"
                    )
                    return all_content
            except Exception as parallel_err:
                logger.warning(
                    "[PDFPLUMBER] Parallel page-range parse failed for %s (%s). Falling back to sequential parser.",
                    file_path,
                    parallel_err,
                )
        
        all_content = []
        total_tables = 0
        text_fallback_success = 0
        text_fallback_fail = 0
        pypdf_reader = None
        pypdf_unavailable_reason = None

        def get_pypdf_reader():
            nonlocal pypdf_reader, pypdf_unavailable_reason
            if pypdf_reader is not None:
                return pypdf_reader
            if pypdf_unavailable_reason is not None:
                return None
            try:
                from pypdf import PdfReader  # Optional fallback parser
                pypdf_reader = PdfReader(file_path)
                logger.debug("[PDFPLUMBER] Enabled pypdf text fallback for problematic pages")
                return pypdf_reader
            except Exception as fallback_init_err:
                pypdf_unavailable_reason = f"{type(fallback_init_err).__name__}: {fallback_init_err}"
                logger.warning(
                    "[PDFPLUMBER] pypdf fallback unavailable for %s (%s)",
                    file_path,
                    pypdf_unavailable_reason,
                )
                return None

        with pdfplumber.open(file_path) as pdf:
            total_pages = len(pdf.pages)
            for raw_page, page in enumerate(pdf.pages, 1):
                # Apply front-matter offset: raw page 35 → printed page 12
                page_num = max(1, raw_page - page_offset) if page_offset else raw_page
                page_content = [f"## Page {page_num}"]
                page_tables = []
                table_hashes = set()  # Track unique tables
                
                def add_table_if_unique(table, strategy_name):
                    """Add table if not already found."""
                    if not table or len(table) < 2:
                        return False
                    # Create hash of table content
                    table_str = str(table)
                    table_hash = hash(table_str)
                    if table_hash not in table_hashes:
                        table_hashes.add(table_hash)
                        page_tables.append((table, strategy_name))
                        return True
                    return False
                
                # --- Robust per-page error handling ---
                # Wrap ALL pdfplumber operations for this page in a
                # timeout guard.  pdfminer's C-level layout parser can
                # enter an infinite loop on corrupted pages — no Python
                # exception is raised, the thread simply never returns.
                # _run_with_timeout() spawns a daemon thread and raises
                # TimeoutError if it doesn't finish in _PAGE_OP_TIMEOUT_S.
                _page_timed_out = False
                try:
                    def _extract_page_tables_and_text():
                        """All pdfplumber work for one page — runs inside timeout guard."""
                        _tables: List[Tuple] = []
                        _text = ""

                        # Strategy 1: Default line-based detection
                        try:
                            default_tables = page.extract_tables()
                            if default_tables:
                                for t in default_tables:
                                    if t and len(t) >= 2:
                                        _tables.append((t, "default"))
                        except (struct.error, Exception):
                            pass

                        # Strategy 2: Text-based detection for borderless tables
                        try:
                            text_tables = page.extract_tables(text_parameters)
                            if text_tables:
                                for t in text_tables:
                                    if t and len(t) >= 2:
                                        _tables.append((t, "text"))
                        except (struct.error, Exception):
                            pass

                        # Strategy 3: Hybrid detection
                        try:
                            hybrid_tables = page.extract_tables(hybrid_parameters)
                            if hybrid_tables:
                                for t in hybrid_tables:
                                    if t and len(t) >= 2:
                                        _tables.append((t, "hybrid"))
                        except (struct.error, Exception):
                            pass

                        # Strategy 4: find_tables() for more control
                        try:
                            found_tables = page.find_tables(text_parameters)
                            if found_tables:
                                for ft in found_tables:
                                    t = ft.extract()
                                    if t and len(t) >= 2:
                                        _tables.append((t, "find_tables"))
                        except (struct.error, Exception):
                            pass

                        # Text extraction
                        try:
                            _text = page.extract_text() or ""
                        except Exception:
                            _text = ""   # caller will attempt pypdf fallback

                        return _tables, _text

                    raw_tables, text = _run_with_timeout(
                        _extract_page_tables_and_text,
                        timeout_s=_PAGE_OP_TIMEOUT_S,
                    )

                    # Deduplicate tables
                    for table, strategy in raw_tables:
                        add_table_if_unique(table, strategy)

                except TimeoutError:
                    # pdfminer hung on this page — skip pdfplumber entirely
                    # and attempt pypdf text fallback below.
                    _page_timed_out = True
                    text = ""
                    logger.warning(
                        "[PDFPLUMBER] Page %s timed out after %ds (likely pdfminer C-level hang) — using pypdf fallback",
                        page_num, _PAGE_OP_TIMEOUT_S,
                    )

                # If pdfplumber failed to extract text (empty string from
                # exception OR from timeout), try pypdf as fallback.
                if not text.strip():
                    primary_reason = "timeout" if _page_timed_out else "extraction_empty_or_failed"
                    fallback_reader = get_pypdf_reader()
                    if fallback_reader is not None and (raw_page - 1) < len(fallback_reader.pages):
                        try:
                            fallback_text = fallback_reader.pages[raw_page - 1].extract_text() or ""
                            if fallback_text.strip():
                                text = fallback_text
                                text_fallback_success += 1
                                logger.debug(
                                    "[PDFPLUMBER] Text fallback used on page %s (%s)",
                                    page_num,
                                    primary_reason,
                                )
                            else:
                                text_fallback_fail += 1
                        except Exception as fallback_err:
                            text_fallback_fail += 1
                            logger.warning(
                                "Text extraction failed on page %s: %s; pypdf fallback failed: %s",
                                page_num,
                                primary_reason,
                                fallback_err,
                            )
                    elif _page_timed_out:
                        text_fallback_fail += 1

                # Convert tables to markdown, assemble page Document.
                # Outer safety net: if anything unexpected blows up
                # (e.g. bad table data, encoding error), skip the page
                # instead of crashing the entire parse run.
                try:
                    for table, strategy in page_tables:
                        total_tables += 1
                        md_table = self._table_to_markdown(table, page_num, total_tables)
                        if md_table:
                            page_content.append(md_table)

                    if text:
                        page_content.append(text.strip())

                    if len(page_content) > 1:
                        combined = "\n\n".join(page_content)
                        chunk_id = f"txt_{file_hash}_{page_num}_0"
                        doc = Document(
                            page_content=combined,
                            metadata={
                                "source": stable_source,
                                "page": page_num,
                                "loader": "pdfplumber",
                                "tables_count": total_tables,
                                "type": "text",
                                "chunk_id": chunk_id,
                            },
                        )
                        all_content.append(doc)
                except Exception as page_err:
                    logger.warning(
                        "[PDFPLUMBER] Skipping page %s due to error: %s",
                        page_num, page_err,
                    )
                    continue
        
        t_end = datetime.now().timestamp()
        if text_fallback_success or text_fallback_fail:
            logger.info(
                "[PDFPLUMBER] Text fallback summary: pypdf_success=%s unrecovered=%s",
                text_fallback_success,
                text_fallback_fail,
            )
        logger.info(f"[PDFPLUMBER] Extracted {len(all_content)} chunks, {total_tables} tables")
        logger.info(
            f"[PROFILE] stage=parse.pdfplumber duration_s={t_end - t_start:.3f} "
            f"pages={total_pages} docs={len(all_content)} tables={total_tables}"
        )
        return all_content
    
    def _table_to_markdown(self, table: List[List], page_num: int, table_idx: int) -> str:
        """Convert a table (list of rows) to markdown format."""
        return _table_to_markdown_impl(table=table, page_num=page_num, table_idx=table_idx)

def run_pdfplumber(file_name):
    from content_analyzer.document_parser import DocumentProcessor
    processor = DocumentProcessor()
    return processor._load_pdf_with_pdfplumber(file_name)

def run_charts(
    file_name,
    enable_chart_extraction,
    vision_client,
    detection_backend: Optional[str] = None,
):
    from content_analyzer.document_parser import DocumentProcessor
    processor = DocumentProcessor()
    processor.vision_client = vision_client
    if enable_chart_extraction and vision_client:
        return processor._extract_charts_from_pdf(file_name, detection_backend=detection_backend)
    return []

