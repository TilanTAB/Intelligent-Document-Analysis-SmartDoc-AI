# SmartDoc AI

SmartDoc AI is an advanced document analysis and question answering system, designed for source-grounded Q&A over complex business and scientific reports—especially where key evidence lives in tables and charts.

---

## Personal Research Update

**SmartDoc AI – Document Q&A + Selective Chart Understanding**

I’ve been developing SmartDoc AI as a technical experiment to improve question answering over complex business/scientific reports—especially where key evidence lives in tables and charts.

### Technical highlights:

- **Multi-format ingestion:** PDF, DOCX, TXT, Markdown
- **LLM-assisted query decomposition:** breaks complex prompts into clearer sub-questions for retrieval + answering
- **Selective chart pipeline (cost-aware):**
  - Local OpenCV heuristics flag pages that likely contain charts
  - Gemini Vision is invoked only for chart pages to generate structured chart analysis (reduces unnecessary vision calls)
- **Table extraction + robust PDF parsing:** pdfplumber strategies for bordered and borderless tables
- **Parallelized processing:** concurrent PDF parsing + chart detection; batch chart analysis where enabled
- **Hybrid retrieval:** BM25 + vector search combined via an ensemble retriever
- **Multi-agent answering:** answer drafting + verification pass, with retrieved context available for inspection (page/source metadata)

**Runtime note:** Large PDFs (many pages/charts) can take minutes depending on DPI, chart volume, and available memory/CPU (HF Spaces limits can be a factor).

---

## Demo Videos

- [SmartDoc AI technical demo #1](https://youtu.be/uVU_sLiJU4w)
- [SmartDoc AI technical demo #2](https://youtu.be/c8CF7-OaKmQ)
- [SmartDoc AI technical demo #3](https://youtu.be/P17SZSQJ6Wc)

---

## Repository
 https://github.com/TilanTAB/Intelligent-Document-Analysis-SmartDoc-AI

---

## Use Cases

- Source-grounded Q&A for business/research documents
- Automated extraction and summarization from tables/charts

If you’re interested in architecture tradeoffs (cost, latency, memory limits, retrieval quality), feel free to connect.

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
git clone https://github.com/TilanTAB/Intelligent-Document-Analysis-SmartDoc-AI.git
cd Intelligent-Document-Analysis-SmartDoc-AI
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
pip install -r requirements.txt
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

SmartDoc AI is actively maintained and designed for real-world document analysis and Q&A. For updates and support, visit the [GitHub repository](https://github.com/TilanTAB/Intelligent-Document-Analysis-SmartDoc-AI).
