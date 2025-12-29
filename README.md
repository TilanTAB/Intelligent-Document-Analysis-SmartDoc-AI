# SmartDoc AI

SmartDoc AI is an advanced document analysis and question answering system. It allows you to upload documents, ask questions, and receive accurate, source-verified answers. The system uses a multi-agent workflow, hybrid search, and both local and cloud-based chart detection for high performance and cost efficiency.

---

## Features

- **Multi-format Document Support**: PDF, DOCX, TXT, and Markdown
- **Smart Chunking**: Configurable chunk size and overlap for optimal retrieval
- **Intelligent Caching**: Speeds up repeated queries
- **Chart Extraction**: Detects and analyzes charts using OpenCV and Gemini Vision
- **Hybrid Search**: Combines keyword and vector search for best results
- **Multi-Agent Workflow**: Relevance checking, research, and answer verification
- **Production Ready**: Structured logging, environment-based config, and test suite
- **Efficient**: Local chart detection saves up to 95% on API costs

---

## Quick Start

### Prerequisites
- Python 3.11 or higher
- Google API Key for Gemini models ([Get one here](https://ai.google.dev/))

### Installation

1. Clone the repository:
```bash
git clone https://github.com/TilanTAB/Intelligent-Document-Analysis-Q-A-3.git
cd Intelligent-Document-Analysis-Q-A-3
```

2. Activate the virtual environment:
```bash
# Windows PowerShell
.\activate_venv.ps1
# Windows Command Prompt
activate_venv.bat
# Or manually:
.\venv\Scripts\Activate.ps1
```

3. Install dependencies (if needed):
```bash
pip install -r dependencies.txt
```

4. Configure environment variables:
```bash
cp .env.template .env
# Edit .env and set your API key
GOOGLE_API_KEY=your_api_key_here
```

5. (Optional) Verify installation:
```bash
python verify_environment.py
```

6. Run the application:
```bash
python main.py
```

7. Open your browser to [http://localhost:7860](http://localhost:7860)

---

## Configuration

All settings can be configured via environment variables or the `.env` file. Key options include:
- `GOOGLE_API_KEY`: Your Gemini API key (required)
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: Document chunking
- `ENABLE_CHART_EXTRACTION`: Enable/disable chart detection
- `CHART_USE_LOCAL_DETECTION`: Use OpenCV for free chart detection
- `CHART_ENABLE_BATCH_ANALYSIS`: Batch process charts for speed
- `CHART_GEMINI_BATCH_SIZE`: Number of charts per Gemini API call
- `LOG_LEVEL`: Logging verbosity
- `GRADIO_SERVER_PORT`: Web interface port

---

## Project Structure

- `intelligence/` - Multi-agent system (relevance, research, verification)
- `configuration/` - App settings and logging
- `content_analyzer/` - Document and chart processing
- `search_engine/` - Hybrid retriever logic
- `core/` - Utilities and diagnostics
- `tests/` - Test suite
- `main.py` - Application entry point

---

## Troubleshooting

- **API Key Not Found**: Set `GOOGLE_API_KEY` in your `.env` file.
- **Python 3.13 Issues**: Use Python 3.11 or 3.12 for best compatibility.
- **Chart Detection Slow**: Lower `CHART_DPI` or `CHART_MAX_IMAGE_SIZE` in `.env`.
- **ChromaDB Lock Issues**: Stop all instances and remove lock files in `vector_store/`.

---

## Contributing

Contributions are welcome! Please fork the repository, create a feature branch, and submit a pull request with a clear description.

---

## License

This project is licensed under the MIT License.

---

SmartDoc AI is actively maintained and designed for real-world document analysis and Q&A. For updates and support, visit the [GitHub repository](https://github.com/TilanTAB/Intelligent-Document-Analysis-Q-A-3).
