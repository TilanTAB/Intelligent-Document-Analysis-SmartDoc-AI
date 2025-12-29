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

from content_analyzer.document_parser import DocumentProcessor
from search_engine.indexer import RetrieverBuilder
from intelligence.orchestrator import AgentWorkflow
from configuration import definitions, parameters
import gradio as gr                                 

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


def _setup_gradio_shim():
    """Shim Gradio's JSON schema conversion to tolerate boolean additionalProperties values."""
    import gradio as gr
    from gradio_client import utils as grc_utils
    _orig_json_schema_to_python_type = grc_utils._json_schema_to_python_type
    def _json_schema_to_python_type_safe(schema, defs=None):
        if isinstance(schema, bool):
            return "Any" if schema else "Never"
        return _orig_json_schema_to_python_type(schema, defs)
    grc_utils._json_schema_to_python_type = _json_schema_to_python_type_safe


def main():
    """Main application entry point."""
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
    js = """
    function createGradioAnimation() {
        var container = document.createElement('div');
        container.id = 'gradio-animation';
        container.style.fontSize = '2.4em';
        container.style.fontWeight = '700';
        container.style.textAlign = 'center';
        container.style.marginBottom = '20px';
        container.style.marginTop = '10px';
        container.style.color = '#0369a1';
        container.style.letterSpacing = '-0.02em';
        var text = '📄 SmartDoc AI';
        for (var i = 0; i < text.length; i++) {
            (function(i){
                setTimeout(function(){
                    var letter = document.createElement('span');
                    letter.style.opacity = '0';
                    letter.style.transition = 'opacity 0.2s ease';
                    letter.innerText = text[i];
                    container.appendChild(letter);
                    setTimeout(function() { letter.style.opacity = '1'; }, 50);
                }, i * 80);
            })(i);
        }
        var gradioContainer = document.querySelector('.gradio-container');
        gradioContainer.insertBefore(container, gradioContainer.firstChild);
        return 'Animation created';
    }
    // Timer logic for processing
    window.processingTimerInterval = null;
    function startProcessingTimer() {
        var timerElem = document.getElementById('processing-timer');
        if (!timerElem) return;
        var start = Date.now();
        window.processingTimerInterval = setInterval(function() {
            var elapsed = (Date.now() - start) / 1000;
            timerElem.textContent = elapsed.toFixed(1) + 's elapsed';
        }, 200);
    }
    function stopProcessingTimer() {
        if (window.processingTimerInterval) {
            clearInterval(window.processingTimerInterval);
            window.processingTimerInterval = null;
        }
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(), title="SmartDoc AI", css=css, js=js) as demo:
        gr.Markdown("### SmartDoc AI - Document Q&A", elem_classes="app-title")
        gr.Markdown("Upload your documents and ask questions. Answers will appear below, just like a chat.", elem_classes="app-description")
        gr.Markdown("---")

        # Examples dropdown
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
        processing_message = gr.HTML("", visible=False)
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

        def process_question(question_text, uploaded_files, chat_history):
            import time
            import random
            chat_history = chat_history or []
            upload_messages = [
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
            ]
            last_msg = None
            start_time = time.time()  
            msg = random.choice([m for m in upload_messages if m != last_msg])
            last_msg = msg
            yield (
                chat_history,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(value=f'<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">{msg}</div>', visible=True)
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
                all_chunks = []
                seen_hashes = set()
                total_chunks = 0
                chunk_counts = []
                for file in uploaded_files:
                    with open(file.name, 'rb') as f:
                        file_content = f.read()
                        file_hash = processor._generate_hash(file_content)
                    cache_path = processor.cache_dir / f"{file_hash}.pkl"
                    if processor._is_cache_valid(cache_path):
                        chunks = processor._load_from_cache(cache_path)
                        if not chunks:
                            chunks = processor._process_file(file)
                            processor._save_to_cache(chunks, cache_path)
                    else:
                        chunks = processor._process_file(file)
                        processor._save_to_cache(chunks, cache_path)
                    chunk_counts.append(len(chunks))
                    total_chunks += len(chunks)
                if total_chunks == 0:
                    total_chunks = 1
                chunk_idx = 0
                msg = random.choice(upload_messages)
                for file, file_chunk_count in zip(uploaded_files, chunk_counts):
                    with open(file.name, 'rb') as f:
                        file_content = f.read()
                        file_hash = processor._generate_hash(file_content)
                    cache_path = processor.cache_dir / f"{file_hash}.pkl"
                    if processor._is_cache_valid(cache_path):
                        chunks = processor._load_from_cache(cache_path)
                        if not chunks:
                            chunks = processor._process_file(file)
                            processor._save_to_cache(chunks, cache_path)
                    else:
                        chunks = processor._process_file(file)
                        processor._save_to_cache(chunks, cache_path)
                    for chunk in chunks:
                        chunk_hash = processor._generate_hash(chunk.page_content.encode())
                        if chunk_hash not in seen_hashes:
                            seen_hashes.add(chunk_hash)
                            all_chunks.append(chunk)
                        chunk_idx += 1
                        percent = int((chunk_idx / total_chunks) * 100)
                        elapsed = time.time() - start_time
                        # Rotate status message every 10 seconds
                        if chunk_idx == 1 or (elapsed // 10) > ((elapsed-1) // 10):
                            msg = random.choice([m for m in upload_messages if m != last_msg])
                            last_msg = msg
                        yield (
                            chat_history,
                            gr.update(visible=False),
                            gr.update(visible=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(interactive=False),
                            gr.update(value=f'<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">{msg} ({percent}% complete, {elapsed:.1f}s elapsed)</div>', visible=True)
                        )
                # After all chunks, show 100%
                elapsed = time.time() - start_time
                yield (
                    chat_history,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(value=f'<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">Analyzing your documents... (100% complete, {elapsed:.1f}s elapsed)</div>', visible=True)
                )
                # Stage 3: Building Retriever
                elapsed = time.time() - start_time
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
                        '<span>Finding the most relevant information in your documents...</span>'
                        '</div>'
                    ), visible=True)
                )
                retriever = retriever_indexer.build_hybrid_retriever(all_chunks)
                # Stage 4: Generating Answer
                elapsed = time.time() - start_time
                yield (
                    chat_history,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(value=f'<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">Processing: Generating Answer ({elapsed:.1f}s elapsed)</div>', visible=True)
                )
                result = orchestrator.full_pipeline(question=question_text, retriever=retriever)
                answer = result["draft_answer"]
                # Stage 5: Verifying Answer
                elapsed = time.time() - start_time
                yield (
                    chat_history,
                    gr.update(visible=False),
                    gr.update(visible=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(interactive=False),
                    gr.update(value=f'<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">Processing: Verifying Answer ({elapsed:.1f}s elapsed)</div>', visible=True)
                )
                verification = result.get("verification_report", "No verification details available.")
                logger.info(f"Verification (internal):\n{verification}")
                # Do not display verification to user, only use internally
                chat_history.append({"role": "user", "content": question_text})
                chat_history.append({"role": "assistant", "content": f"**Answer:**\n{answer}"})

                session_state.value["last_documents"] = retriever.invoke(question_text)
                # Final: Show results and make context tab visible
                total_elapsed = time.time() - start_time
                yield (
                    chat_history,
                    gr.update(visible=True),  # doc_context_display
                    gr.update(visible=True),  # refresh_context_btn
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(interactive=True),
                    gr.update(value=f'<div style="background:#fff; border-radius:8px; padding:18px 24px; margin-top:32px; color:#1e293b; font-size:1.2em; font-weight:600; box-shadow:0 2px 8px rgba(0,0,0,0.04);">Total time elapsed: {total_elapsed:.1f}s</div>', visible=True)
                )
               
                time.sleep(1.5)
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
            file_paths = ex_data["file_paths"]
            import tempfile
            temp_dir = tempfile.mkdtemp()
            copied_files = []
            file_info_text = f"Loaded: {example_key}\n\n"
            for source_file_path in file_paths:
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

        # Remove the Load Example button and related logic
        # Instead, load the example immediately when dropdown changes
        example_dropdown.change(
            fn=load_example,
            inputs=[example_dropdown],
            outputs=[files, question, loaded_file_info]
        )
    # Launch server - Compatible with both local and Hugging Face Spaces
    # HF Spaces sets SPACE_ID environment variable
    is_hf_space = os.environ.get("SPACE_ID") is not None
    
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
