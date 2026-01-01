import configuration.logger_setup
import logging

logger = logging.getLogger(__name__)

import hashlib
import socket
from typing import List, Dict
import os
import shutil
from pathlib import Path
from datetime import datetime
import time
import random
from collections import defaultdict, deque
import threading

from content_analyzer.document_parser import DocumentProcessor
from search_engine.indexer import RetrieverBuilder
from intelligence.orchestrator import AgentWorkflow
from configuration import definitions, parameters
                             

# Rate limiting configuration - 3 requests per hour per IP
WINDOW_S = 3600
MAX_CALLS = 5
_calls = defaultdict(deque)  # ip -> timestamps
_calls_lock = threading.Lock()  # Thread-safe access to rate limit state

def rate_limit(request):
    """Thread-safe rate limiting per IP address."""
    ip = getattr(request.client, "host", "unknown")
    now = time.time()

    with _calls_lock:
        q = _calls[ip]
        # Remove expired entries
        while q and (now - q[0]) > WINDOW_S:
            q.popleft()
    
        if len(q) >= MAX_CALLS:
            import gradio as gr
            raise gr.Error(f"Rate limit: {MAX_CALLS} requests per {WINDOW_S//60} minutes. Please wait.")
    
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
            import re
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

    with gr.Blocks(title="SmartDoc AI") as demo:
        gr.Markdown("### SmartDoc AI - Document Q&A", elem_classes="app-title")
        gr.Markdown("Upload your documents and ask questions. Answers will appear below, just like a chat.", elem_classes="app-description")
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
        submit_btn = gr.Button("Get Answer", variant="primary")
        processing_message = gr.HTML("", elem_id="processing-message", visible=False)
        doc_context_display = gr.Markdown("*Submit a question to see which document sections were referenced*", elem_classes="doc-context", visible=False)
        refresh_context_btn = gr.Button("Refresh Sources", variant="secondary", visible=False)
        with gr.Tab("Context"):
            pass  # No .render() calls here; components are already defined and used in outputs

        session_state = gr.State({
            "file_hashes": frozenset(),
            "retriever": None,
            "chat_history": [],
            "last_documents": [],
            "total_questions": 0,
            "session_start": datetime.now().strftime("%Y-%m-%d %H:%M")
        })

        def process_question(question_text, uploaded_files, chat_history, request: gr.Request):
            rate_limit(request)
            chat_history = chat_history or []            
            yield (
                chat_history,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value='''<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
  <span id="processing-msg"></span>
  <span id="processing-timer" style="opacity:0.8; margin-left:8px;"></span>
</div>''', visible=True)
            )
            try:
                if not question_text.strip():
               
                    chat_history.append({"role": "user", "content": question_text})
                    chat_history.append({"role": "assistant", "content": "Please enter a question."})
                    yield (
                        chat_history,
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(value="", visible=False)
                    )
                    return
                if not uploaded_files:                         
              
                    chat_history.append({"role": "user", "content": question_text})
                    chat_history.append({"role": "assistant", "content": "Please upload at least one document."})
                    yield (
                        chat_history,
                        gr.update(visible=False),
                        gr.update(visible=False),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(interactive=True),
                        gr.update(value="", visible=False)
                    )
                    return
                # Stage 2: Chunking with per-chunk progress and rotating status
                def load_or_process(file):
                    with open(file.name, "rb") as f:
                        file_content = f.read()
                    file_hash = processor._generate_hash(file_content)
                    cache_path = processor.cache_dir / f"{file_hash}.pkl"
                    if processor._is_cache_valid(cache_path):
                        chunks = processor._load_from_cache(cache_path)
                        if chunks:
                            logger.info(f"Using cached chunks for {file.name}")
                            return chunks
                    chunks = processor._process_file(file)
                    processor._save_to_cache(chunks, cache_path)
                    return chunks

                all_chunks = []
                seen_hashes = set()
                chunks_by_file = []
                total_chunks = 0
                for file in uploaded_files:
                    chunks = load_or_process(file)
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
                            gr.update(value='''<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
  <span id="processing-msg"></span>
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
                    gr.update(value='''<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
  <span id="processing-msg"></span>
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
                    gr.update(value=(
                        '<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04); display:flex; align-items:center;">'
                        '<img src="https://media.giphy.com/media/26ufnwz3wDUli7GU0/giphy.gif" alt="AI working" style="height:40px; margin-right:16px;">'
                        '<span id="processing-msg"></span>'
                        '</div>'
                    ), visible=True)
                )
                retriever = retriever_indexer.build_hybrid_retriever(all_chunks)
                # Stage 4: Generating Answer
                yield (
                    chat_history,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(value='''<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
  <span id="processing-msg"></span>
  <span id="processing-timer" style="opacity:0.8; margin-left:8px;"></span>
</div>''', visible=True)
                )
                result = orchestrator.run_workflow(question=question_text, retriever=retriever)
                answer = result["draft_answer"]
                # Stage 5: Verifying Answer
                yield (
                    chat_history,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(value='''<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
  <span id="processing-msg"></span>
  <span id="processing-timer" style="opacity:0.8; margin-left:8px;"></span>
</div>''', visible=True)
                )
                verification = result.get("verification_report", "No verification details available.")
                logger.info(f"Verification (internal):\n{verification}")
                # Do not display verification to user, only use internally
           
                chat_history.append({"role": "user", "content": question_text})
                chat_history.append({"role": "assistant", "content": f"**Answer:**\n{answer}"})
                session_state.value["last_documents"] = retriever.invoke(question_text)
                yield (
                    chat_history,
                    gr.update(visible=True),  # doc_context_display
                    gr.update(visible=True),  # refresh_context_btn
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(value='''<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">
  <span id="processing-msg"></span>
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
                    gr.update(value="", visible=False)
                )
            except Exception as e:
                logger.error(f"Processing error: {e}", exc_info=True)
           
           
                chat_history.append({"role": "user", "content": question_text})
                chat_history.append({"role": "assistant", "content": f"Error: {str(e)}"})
                yield (
                    chat_history,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(value="", visible=False)
                )

        submit_btn.click(
            fn=process_question,
            inputs=[question, files, chat],
            outputs=[chat, doc_context_display, refresh_context_btn, submit_btn, question, files, example_dropdown, processing_message],
            queue=True,
            show_progress=True
        )

        def refresh_context():
            docs = session_state.value.get("last_documents", [])
            last_question = ""
            for msg in reversed(chat.value or []):
                if msg["role"] == "user":
                    last_question = msg["content"]
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
        demo.launch(theme=gr.themes.Soft(), server_name="0.0.0.0", server_port=7860, css=css, js=js)
    else:
        # Local development configuration
        configured_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
        server_port = _find_open_port(configured_port)
        logger.info(f"Launching Gradio on port {server_port}")
        logger.info(f"Access the app at: http://127.0.0.1:{server_port}")
        demo.launch(theme=gr.themes.Soft(), server_name="127.0.0.1", server_port=server_port, share=False, css=css, js=js)


if __name__ == "__main__":
    main()
