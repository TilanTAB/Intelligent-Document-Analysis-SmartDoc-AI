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
  - Vision calls are currently enabled only when Google vision is configured; with the default OpenAI setup they remain off for cost control
- **Table extraction + robust PDF parsing:** pdfplumber strategies for bordered and borderless tables
- **Parallelized processing:** concurrent PDF parsing + chart detection; batch chart analysis where enabled
- **Hybrid retrieval:** BM25 + vector search combined via an ensemble retriever
- **Multi-agent answering:** answer drafting + verification pass, with retrieved context available for inspection (page/source metadata)

**Runtime note:** Large PDFs (many pages/charts) can take minutes depending on DPI, chart volume, and available memory/CPU (HF Spaces limits can be a factor).

---

## Demo Videos

- [SmartDoc AI technical demo #1](https://youtu.be/uVU_sLiJU4w)
- [SmartDoc AI technical demo #2](https://youtu.be/c8CF7-OaKmQ)
- [SmartDoc AI technical demo #3](https://youtu.be/Rg3EGEtbH1E)

---

## Repository
 https://github.com/TilanTAB/Intelligent-Document-Analysis-SmartDoc-AI

---
# Hugging Face Link

https://huggingface.co/spaces/TilanB/SmartDocAI

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
- **Chart Extraction**: Detects charts with OpenCV; optional Gemini vision analysis can be enabled for richer summaries
- **Hybrid Search**: Combines keyword and vector search for best results
- **Multi-Agent Workflow**: Relevance checking, research, and answer verification
- **Production Ready**: Structured logging, environment-based config, and test suite
- **Efficient**: Local chart detection saves up to 95% on API costs

---

## Quick Start

### Prerequisites
- Python 3.11 or higher
- Choose an LLM provider (default is low-cost OpenAI GPT):
  - OpenAI GPT (default): `LLM_PROVIDER=openai` and `OPENAI_API_KEY` ([get one](https://platform.openai.com/api-keys))
  - Azure OpenAI: `LLM_PROVIDER=azure` plus `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`
  - Google Gemini (optional): `LLM_PROVIDER=google` and `GOOGLE_API_KEY` ([get one](https://ai.google.dev/))

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

4. Configure environment variables (OpenAI GPT-4o-mini is the default):
```bash
cp .env.template .env
# Edit .env and set provider + keys
# LLM_PROVIDER=openai or google or azure
OPENAI_API_KEY=your_openai_api_key_here
# Optional if you switch to Google
GOOGLE_API_KEY=your_google_api_key_here
# Optional if you use Azure OpenAI
AZURE_OPENAI_API_KEY=your_azure_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your_chat_deployment
AZURE_OPENAI_API_VERSION=2024-08-01-preview
# If you want embeddings on Azure too
# EMBEDDING_PROVIDER=azure
# Optional: use a dedicated key for embeddings
# AZURE_EMBEDDING_API_KEY=your_embedding_resource_key
# Optional: use a dedicated Azure endpoint for embeddings
# AZURE_EMBEDDING_ENDPOINT=https://your-embedding-resource.openai.azure.com/
# AZURE_EMBEDDING_DEPLOYMENT=your_embedding_deployment
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
- `OPENAI_API_KEY`: Your OpenAI API key (required when LLM_PROVIDER=openai — default)
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION`: Required when LLM_PROVIDER=azure
- `GOOGLE_API_KEY`: Your Gemini API key (required only if you switch to Google)
- `EMBEDDING_PROVIDER`: Set to `azure` if you want embeddings on Azure
- `AZURE_EMBEDDING_API_KEY`: Optional dedicated key for Azure embeddings (falls back to `AZURE_OPENAI_API_KEY` if unset)
- `AZURE_EMBEDDING_DEPLOYMENT`: Azure embedding deployment name when `EMBEDDING_PROVIDER=azure`
- `AZURE_EMBEDDING_ENDPOINT`: Optional dedicated Azure endpoint for embeddings (falls back to `AZURE_OPENAI_ENDPOINT` if unset)
- `EMBEDDING_CACHE_ENABLED`, `EMBEDDING_CACHE_DIR`, `EMBEDDING_CACHE_BATCH_SIZE`: Embedding cache controls
- `CHUNK_SIZE`, `CHUNK_OVERLAP`: Document chunking
- `CHUNK_SIZE_TEXT`, `CHUNK_OVERLAP_TEXT`, `CHUNK_SIZE_TABLE`, `CHUNK_OVERLAP_TABLE`, `CHUNK_SIZE_CHART`, `CHUNK_OVERLAP_CHART`: Adaptive chunking per content type
- `MAX_INDEX_CHUNKS`: Optional hard cap on chunks indexed per build (0 disables cap)
- `PRE_INGEST_DEDUPE_ENABLED`, `PRE_INGEST_COMPRESS_WHITESPACE`, `PRE_INGEST_MIN_CHUNK_CHARS`: Pre-ingest chunk optimization controls
- `PDF_PARSE_MODE`: Backward-compatible flag (`auto|fast|fidelity` accepted), but parsing now always uses fidelity mode (`pdfplumber`)
- `PDF_PARSE_PAGE_RANGE_WORKERS`, `PDF_PARSE_PAGE_RANGE_SIZE`: Optional page-range parallelism for large PDFs (`workers>1` enables parallel parse ranges)
- `CHROMA_INGEST_BATCH_SIZE`: Configurable vector ingest batch size
- `VECTOR_INGEST_PARALLEL_WORKERS`: Parallel vector ingest workers (`1` disables parallel mode)
- `EMBEDDING_INGEST_MAX_RETRIES`, `EMBEDDING_BACKOFF_BASE_S`, `EMBEDDING_BACKOFF_MAX_S`, `EMBEDDING_BACKOFF_JITTER_RATIO`: Rate-limit-aware embedding ingest retries/backoff
- `PERF_METRICS_WINDOW`, `PERF_ALERT_MIN_SAMPLES`, `PERF_ALERT_COOLDOWN_S`, `PERF_ALERT_*_P95_S`: Stage p50/p95 metric windows and alert thresholds
- `CHROMA_HNSW_M`, `CHROMA_HNSW_CONSTRUCTION_EF`, `CHROMA_HNSW_SEARCH_EF`: HNSW latency/recall tuning
- `CHROMA_HNSW_NUM_THREADS`, `CHROMA_HNSW_BATCH_SIZE`, `CHROMA_HNSW_SYNC_THRESHOLD`: Chroma ingest/search tuning
- `ENABLE_CHART_EXTRACTION`: Enable/disable chart detection
- `CHART_USE_LOCAL_DETECTION`: Use OpenCV for free chart detection
- `CHART_ENABLE_BATCH_ANALYSIS`: Batch process charts for speed
- `CHART_GEMINI_BATCH_SIZE`: Number of charts per Gemini API call (Gemini-only)
- `LOG_LEVEL`: Logging verbosity
- `OTEL_ENABLED`, `OTEL_SERVICE_NAME`, `OTEL_SERVICE_NAMESPACE`, `OTEL_SERVICE_VERSION`: OpenTelemetry toggle and service metadata
- `OTEL_EXPORTER_OTLP_ENDPOINT` (or per-signal `OTEL_EXPORTER_OTLP_*_ENDPOINT`): OTLP endpoint(s) for traces/metrics/logs
- `OTEL_EXPORTER_OTLP_HEADERS`: OTLP auth headers (`key=value,key2=value2`)
- `OTEL_TRACES_ENABLED`, `OTEL_METRICS_ENABLED`, `OTEL_LOGS_ENABLED`: Per-signal export toggles
- `OTEL_METRICS_EXPORT_INTERVAL_MS`, `OTEL_EXPORT_TIMEOUT_S`: Export cadence and timeout controls
- `GRADIO_SERVER_PORT`: Web interface port

---

## Live Observability (Logs + Metrics)

SmartDoc now supports OpenTelemetry export for:
- live run metrics (`smartdoc.requests.*`, `smartdoc.stage.*`)
- application traces (workflow spans)
- application logs (exported via OTLP when enabled)

Quick start:

```bash
# .env
OTEL_ENABLED=true
OTEL_SERVICE_NAME=smartdoc-ai
OTEL_EXPORTER_OTLP_ENDPOINT=https://otlp-gateway-prod-us-central-0.grafana.net/otlp
OTEL_EXPORTER_OTLP_HEADERS=Authorization=Basic <base64(instance_id:api_token)>
```

Then run:

```bash
python main.py
```

See `docs/OBSERVABILITY.md` for full setup and free-tier backend options (Grafana Cloud and GCP/AWS/Azure notes).

---

## Performance Benchmarking

Use the benchmark scripts to generate baseline/candidate JSON reports and compare against a latency gate:

```bash
python scripts/benchmark_latency.py --input-dir samples --runs 3 --label baseline
python scripts/benchmark_latency.py --input-dir samples --runs 3 --label candidate
python scripts/compare_benchmarks.py --baseline docs/perf/benchmarks/<baseline>.json --candidate docs/perf/benchmarks/<candidate>.json --target-improvement 35
```

Optional quality guard input:
- `--quality-queries path/to/queries.json`
- Format: `[{"question":"...", "expected_sources":["source-fragment"]}]`

---

## Project Structure
- `intelligence/` - Multi-agent system (relevance, research, verification)
- `configuration/` - App settings and logging
- `content_analyzer/` - Document and chart processing
- `search_engine/` - Hybrid retriever logic
- `core/` - Utilities and diagnostics
- `tests/` - Test suite
- `main.py` - Application entry point

## Architecture Diagrams
- Static markdown (IDE-safe): `docs/architecture.md`
- Self-contained visual HTML: `docs/architecture_visual.html`
- Image assets: `docs/diagrams/`

---

## Troubleshooting
- **API Key / Deployment Not Found**: Set `OPENAI_API_KEY` (or `AZURE_OPENAI_API_KEY` + endpoint/deployment for Azure, or `GOOGLE_API_KEY` for Gemini) in your `.env`; for Azure embeddings set `EMBEDDING_PROVIDER=azure`, `AZURE_EMBEDDING_DEPLOYMENT`, and optionally `AZURE_EMBEDDING_ENDPOINT` + `AZURE_EMBEDDING_API_KEY` when embeddings live in a different Azure resource.
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
