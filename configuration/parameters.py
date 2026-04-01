from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator, model_validator
from typing import Optional
import os
import re
from .definitions import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES


def _default_chroma_path() -> str:
    if os.environ.get("SPACE_ID"):  # Hugging Face Spaces
        return os.environ.get("CHROMA_DB_PATH", "/tmp/chroma_db")
    return os.environ.get("CHROMA_DB_PATH", "./chroma_db")


def _is_hf() -> bool:
    return os.environ.get("SPACE_ID") is not None


class Settings(BaseSettings):
    """
    Application parameters loaded from environment variables.
    
    For local development:
        Create a .env file in the project root with your configuration:
        GOOGLE_API_KEY=your_api_key_here
    
    For Hugging Face Spaces:
        Add GOOGLE_API_KEY as a secret in Space Settings > Repository secrets
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # File upload parameters with defaults from definitions
    MAX_FILE_SIZE: int = MAX_FILE_SIZE
    MAX_TOTAL_SIZE: int = MAX_TOTAL_SIZE
    ALLOWED_TYPES: list = ALLOWED_TYPES

    # API keys / endpoints - REQUIRED based on provider
    GOOGLE_API_KEY: Optional[str] = Field(
        default=None,
        description="Google API key for Gemini models",
    )
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="OpenAI API key for GPT models",
    )
    AZURE_OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description="Azure OpenAI API key",
    )
    AZURE_EMBEDDING_API_KEY: Optional[str] = Field(
        default=None,
        description=(
            "Optional separate Azure OpenAI API key for embeddings. "
            "If unset, AZURE_OPENAI_API_KEY is used."
        ),
    )
    AZURE_OPENAI_ENDPOINT: Optional[str] = Field(
        default=None,
        description="Azure OpenAI endpoint URL (e.g., https://YOUR-RESOURCE.openai.azure.com/)",
    )
    AZURE_OPENAI_API_VERSION: Optional[str] = Field(
        default="2024-08-01-preview",
        description="Azure OpenAI API version",
    )
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = Field(
        default=None,
        description="Azure OpenAI chat deployment name",
    )
    AZURE_EMBEDDING_ENDPOINT: Optional[str] = Field(
        default=None,
        description=(
            "Optional separate Azure OpenAI endpoint for embeddings. "
            "If unset, AZURE_OPENAI_ENDPOINT is used."
        ),
    )

    # Database parameters
    CHROMA_DB_PATH: str = Field(default_factory=_default_chroma_path)
    CHROMA_COLLECTION_NAME: str = "documents"
    CHROMA_INGEST_BATCH_SIZE: int = 100
    CHROMA_HNSW_M: int = 32
    CHROMA_HNSW_CONSTRUCTION_EF: int = 100
    CHROMA_HNSW_SEARCH_EF: int = 80
    CHROMA_HNSW_NUM_THREADS: int = Field(default_factory=lambda: max(1, os.cpu_count() or 1))
    CHROMA_HNSW_BATCH_SIZE: int = 100
    CHROMA_HNSW_SYNC_THRESHOLD: int = 1000
    VECTOR_INGEST_PARALLEL_WORKERS: int = 1
    STREAMING_CHUNKER_ENABLED: bool = True  # B2: overlap parse + vector ingest; set False to disable

    # Chunking parameters
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 100
    CHUNK_SIZE_TEXT: int = 2000
    CHUNK_OVERLAP_TEXT: int = 100
    CHUNK_SIZE_TABLE: int = 1400
    CHUNK_OVERLAP_TABLE: int = 80
    CHUNK_SIZE_CHART: int = 2600
    CHUNK_OVERLAP_CHART: int = 120
    MAX_INDEX_CHUNKS: int = 0
    PRE_INGEST_DEDUPE_ENABLED: bool = True
    PRE_INGEST_COMPRESS_WHITESPACE: bool = True
    PRE_INGEST_MIN_CHUNK_CHARS: int = 40
    PRE_INGEST_MIN_TABLE_CHUNK_CHARS: int = 80
    PRE_INGEST_TABLE_CANONICAL_DEDUPE: bool = True
    PRE_INGEST_TABLE_COMPRESS_REPEATS: bool = True
    # NOTE: fast/auto parser paths were removed; value is kept for backward compatibility.
    PDF_PARSE_MODE: str = "fidelity"  # accepted: auto | fast | fidelity (all treated as fidelity)
    # PDF ingestion strategy:
    # - both: text/tables via pdfplumber + vision chart/page analysis (default)
    # - pdf_only: text/tables only, no vision-based extraction
    # - vision_only: vision-based page analysis only (no pdf text parsing)
    PDF_ANALYSIS_MODE: str = "both"  # pdf_only | vision_only | both
    # Optional page-range parallelism for single large PDFs (disabled by default).
    PDF_PARSE_PAGE_RANGE_WORKERS: int = 1
    PDF_PARSE_PAGE_RANGE_SIZE: int = 24

    # Retriever parameters
    VECTOR_SEARCH_K: int = 25
    VECTOR_SEARCH_K_CHROMA: int = 15  # Fixed typo: was VECTOR_Search_K_CHROMA
    VECTOR_FETCH_K: int = 35
    VECTOR_SCORE_THRESHOLD: float = 0.3
    BM25_SEARCH_K: int = 8
    HYBRID_RETRIEVER_WEIGHTS: list = [0.4, 0.6]  # [BM25 weight, Vector weight]

    # Workflow parameters
    MAX_RESEARCH_ATTEMPTS: int = 5
    ENABLE_QUERY_REWRITING: bool = True
    MAX_QUERY_REWRITES: int = 1
    RELEVANCE_CHECK_K: int = 20

    # Research agent parameters
    RESEARCH_TOP_K: int = 15
    RESEARCH_MAX_CONTEXT_CHARS: int = Field(default_factory=lambda: 800_000 if _is_hf() else 8000000000)
    RESEARCH_MAX_OUTPUT_TOKENS: int = 4000
    RESEARCH_MIN_ANSWER_CHARS: int = 300
    NUM_RESEARCH_CANDIDATES: int = 2  # Number of research questions to generate

    # Verification parameters
    VERIFICATION_MAX_CONTEXT_CHARS: int = Field(default_factory=lambda: 300_000 if _is_hf() else 800000000)
    VERIFICATION_MAX_OUTPUT_TOKENS: int = 128000
    VERIFICATION_RETRY_MAX_OUTPUT_TOKENS: int = 128000
    RELEVANCE_MAX_OUTPUT_TOKENS: int = 4000
    RELEVANCE_RETRY_MAX_OUTPUT_TOKENS: int = 8000

    # Logging parameters
    LOG_LEVEL: str = "INFO"
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_WINDOW_S: int = 3600
    RATE_LIMIT_MAX_CALLS: int = 5
    OTEL_ENABLED: bool = False
    OTEL_SERVICE_NAME: str = "smartdoc-ai"
    OTEL_SERVICE_NAMESPACE: str = "smartrag"
    OTEL_SERVICE_VERSION: str = "dev"
    OTEL_RESOURCE_ATTRIBUTES: str = ""
    OTEL_EXPORTER_OTLP_ENDPOINT: Optional[str] = None
    OTEL_EXPORTER_OTLP_TRACES_ENDPOINT: Optional[str] = None
    OTEL_EXPORTER_OTLP_METRICS_ENDPOINT: Optional[str] = None
    OTEL_EXPORTER_OTLP_LOGS_ENDPOINT: Optional[str] = None
    OTEL_EXPORTER_OTLP_HEADERS: Optional[str] = None  # format: key1=value1,key2=value2
    OTEL_EXPORT_TIMEOUT_S: float = 10.0
    OTEL_TRACES_ENABLED: bool = True
    OTEL_METRICS_ENABLED: bool = True
    OTEL_LOGS_ENABLED: bool = True
    OTEL_METRICS_EXPORT_INTERVAL_MS: int = 10000
    OTEL_CONSOLE_DEBUG_EXPORT: bool = False

    # Cache parameters
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    # LLM parameters
    LLM_PROVIDER: str = "openai"  # google | openai | azure
    LLM_MAX_RETRIES: int = 3
    LLM_RETRY_DELAY: float = 1.0
    LLM_MODEL_NAME: str = "gpt-4o-mini"  # Default model for all agents (low-cost GPT)
    LLM_ROUTER_MAX_OUTPUT_TOKENS: int = 4000

    # Agent-specific LLM models (override LLM_MODEL_NAME if needed)
    RESEARCH_AGENT_MODEL: str = "gpt-4o-mini"
    VERIFICATION_AGENT_MODEL: str = "gpt-4o-mini"
    RELEVANCE_CHECKER_MODEL: str = "gpt-4o-mini"

    # Embeddings
    EMBEDDING_PROVIDER: str = "openai"  # google | openai | azure
    EMBEDDING_MODEL_NAME: str = "models/text-embedding-004"
    OPENAI_EMBEDDING_MODEL_NAME: str = "text-embedding-3-small"
    AZURE_EMBEDDING_DEPLOYMENT: Optional[str] = None
    EMBEDDING_CACHE_ENABLED: bool = True
    EMBEDDING_CACHE_DIR: str = "document_cache/embedding_cache"
    EMBEDDING_CACHE_BATCH_SIZE: int = 128
    EMBEDDING_INGEST_MAX_RETRIES: int = 4
    EMBEDDING_BACKOFF_BASE_S: float = 1.0
    EMBEDDING_BACKOFF_MAX_S: float = 12.0
    EMBEDDING_BACKOFF_JITTER_RATIO: float = 0.25

    # Performance metrics / alerting
    PERF_METRICS_WINDOW: int = 200
    PERF_ALERT_MIN_SAMPLES: int = 5
    PERF_ALERT_COOLDOWN_S: int = 300
    PERF_ALERT_PARSE_P95_S: float = 90.0
    PERF_ALERT_CHART_PHASE1_P95_S: float = 120.0
    PERF_ALERT_CHART_PHASE2_P95_S: float = 45.0
    PERF_ALERT_EMBED_BATCH_P95_S: float = 10.0

    # Vision / chart analysis
    VISION_PROVIDER: str = "none"  # google | azure | none

    # Chart extraction parameters
    ENABLE_CHART_EXTRACTION: bool = True
    CHART_ASSET_EXPIRE_DAYS: int = 7
    CHART_VISION_MODEL: str = "gemini-2.5-flash-lite"  # Use gpt-5-nano for Azure vision
    CHART_MAX_TOKENS: int = 1500
    CHART_DPI: int = Field(default_factory=lambda: 110 if _is_hf() else 110)  # Lower DPI saves memory
    CHART_BATCH_SIZE: int = Field(default_factory=lambda: 1 if _is_hf() else 1)  # Process pages in batches
    CHART_MAX_IMAGE_SIZE: int = Field(default_factory=lambda: 1200 if _is_hf() else 1200)  # Max dimension for images

    # Local chart detection parameters (cost optimization)
    CHART_USE_LOCAL_DETECTION: bool = Field(default_factory=lambda: True if _is_hf() else True)  # Use OpenCV first (FREE)
    CHART_MIN_CONFIDENCE: float = 0.4  # Only analyze charts with confidence > 40%
    CHART_SKIP_GEMINI_DETECTION: bool = True  # Skip Gemini for detection, only use for analysis
    CHART_GEMINI_FALLBACK_ENABLED: bool = False  # Optional: Use Gemini if local fails
    # Detection backend options:
    # - pdfplumber: Use PDF structural signals (lines/curves/images/chars) for fast candidate-page triage.
    # - opencv_optimized: Use local OpenCV detection with conservative performance knobs.
    CHART_DETECTION_BACKEND: str = "pdfplumber"  # pdfplumber | opencv_optimized
    CHART_OPENCV_MAX_DIM: int = 700
    CHART_OPENCV_WORKERS: int = Field(default_factory=lambda: min(max(1, os.cpu_count() or 1), 4))
    CHART_OPENCV_USE_PROCESS_POOL: bool = False
    CHART_OPENCV_FAST_MODE: bool = True
    CHART_OPENCV_MIN_LINES_FOR_CIRCLES: int = 20
    CHART_OPENCV_MIN_DIAG_LINES_FOR_CIRCLES: int = 2

    # Gemini batch processing parameters (speed optimization - 2-3× faster)
    CHART_GEMINI_BATCH_SIZE: int = 1  # Analyze 1 chart per API call (reduced from 2 for reliability)
    CHART_ENABLE_BATCH_ANALYSIS: bool = True  # Enable batch processing for speed

    @field_validator("LLM_PROVIDER")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        supported = {"google", "openai", "azure"}
        v_lower = v.lower()
        if v_lower not in supported:
            raise ValueError(f"LLM_PROVIDER must be one of {supported}")
        return v_lower

    @field_validator("VISION_PROVIDER")
    @classmethod
    def validate_vision_provider(cls, v: str) -> str:
        supported = {"google", "azure", "none"}
        v_lower = v.lower()
        if v_lower not in supported:
            raise ValueError(f"VISION_PROVIDER must be one of {supported}")
        return v_lower

    @field_validator("EMBEDDING_PROVIDER")
    @classmethod
    def validate_embedding_provider(cls, v: str) -> str:
        supported = {"google", "openai", "azure"}
        v_lower = v.lower()
        if v_lower not in supported:
            raise ValueError(f"EMBEDDING_PROVIDER must be one of {supported}")
        return v_lower

    @field_validator("PDF_PARSE_MODE")
    @classmethod
    def validate_pdf_parse_mode(cls, v: str) -> str:
        supported = {"auto", "fast", "fidelity"}
        v_lower = v.lower()
        if v_lower not in supported:
            raise ValueError(f"PDF_PARSE_MODE must be one of {supported}")
        return v_lower

    @field_validator("PDF_ANALYSIS_MODE")
    @classmethod
    def validate_pdf_analysis_mode(cls, v: str) -> str:
        supported = {"pdf_only", "vision_only", "both"}
        v_lower = v.lower()
        if v_lower not in supported:
            raise ValueError(f"PDF_ANALYSIS_MODE must be one of {supported}")
        return v_lower

    @field_validator("CHART_DETECTION_BACKEND")
    @classmethod
    def validate_chart_detection_backend(cls, v: str) -> str:
        supported = {"pdfplumber", "opencv_optimized"}
        v_lower = v.lower()
        if v_lower not in supported:
            raise ValueError(f"CHART_DETECTION_BACKEND must be one of {supported}")
        return v_lower

    @field_validator("PDF_PARSE_PAGE_RANGE_WORKERS", "PDF_PARSE_PAGE_RANGE_SIZE")
    @classmethod
    def validate_positive_pdf_parse_parallel_settings(cls, v: int, info) -> int:
        if int(v) < 1:
            raise ValueError(f"{info.field_name} must be >= 1")
        return int(v)

    @field_validator("PRE_INGEST_MIN_CHUNK_CHARS", "PRE_INGEST_MIN_TABLE_CHUNK_CHARS")
    @classmethod
    def validate_non_negative_pre_ingest_thresholds(cls, v: int, info) -> int:
        if int(v) < 0:
            raise ValueError(f"{info.field_name} must be >= 0")
        return int(v)

    @field_validator("RATE_LIMIT_WINDOW_S", "RATE_LIMIT_MAX_CALLS")
    @classmethod
    def validate_positive_rate_limit_settings(cls, v: int, info) -> int:
        if int(v) < 1:
            raise ValueError(f"{info.field_name} must be >= 1")
        return int(v)

    @field_validator(
        "CHART_OPENCV_MAX_DIM",
        "CHART_OPENCV_WORKERS",
        "CHART_OPENCV_MIN_LINES_FOR_CIRCLES",
        "CHART_OPENCV_MIN_DIAG_LINES_FOR_CIRCLES",
    )
    @classmethod
    def validate_positive_chart_opencv_settings(cls, v: int, info) -> int:
        if int(v) < 1:
            raise ValueError(f"{info.field_name} must be >= 1")
        return int(v)

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def validate_openai_api_key(cls, v: Optional[str], info) -> Optional[str]:
        provider = info.data.get("LLM_PROVIDER")
        embedding_provider = info.data.get("EMBEDDING_PROVIDER")
        # If provider fields are not yet available (validation-order issue), defer to model-level validation.
        if provider is None and embedding_provider is None:
            return v
        provider = (provider or "openai").lower()
        embedding_provider = (embedding_provider or "openai").lower()
        needs_openai = provider == "openai" or embedding_provider == "openai"
        if needs_openai and (not v or not v.strip() or v.startswith("your_") or v == "YOUR_API_KEY_HERE"):
            raise ValueError("OPENAI_API_KEY is required when using OpenAI provider. Set it in your .env file.")
        return v

    @field_validator("AZURE_OPENAI_API_KEY")
    @classmethod
    def validate_azure_api_key(cls, v: Optional[str], info) -> Optional[str]:
        provider = info.data.get("LLM_PROVIDER")
        if provider is None:
            return v
        provider = (provider or "openai").lower()
        if provider == "azure" and (not v or not v.strip() or v.startswith("your_") or v == "YOUR_API_KEY_HERE"):
            raise ValueError("AZURE_OPENAI_API_KEY is required when LLM_PROVIDER=azure. Set it in your .env file.")
        return v

    @field_validator("AZURE_OPENAI_ENDPOINT")
    @classmethod
    def validate_azure_endpoint(cls, v: Optional[str], info) -> Optional[str]:
        provider = info.data.get("LLM_PROVIDER")
        if provider is None:
            return v
        provider = (provider or "openai").lower()
        if provider == "azure" and (not v or not v.strip()):
            raise ValueError("AZURE_OPENAI_ENDPOINT is required when LLM_PROVIDER=azure.")
        return v

    @field_validator("AZURE_OPENAI_DEPLOYMENT")
    @classmethod
    def validate_azure_deployment(cls, v: Optional[str], info) -> Optional[str]:
        provider = info.data.get("LLM_PROVIDER")
        if provider is None:
            return v
        provider = provider.lower()
        if provider == "azure" and (not v or not v.strip()):
            raise ValueError("AZURE_OPENAI_DEPLOYMENT is required when using Azure OpenAI chat.")
        return v

    @field_validator("GOOGLE_API_KEY")
    @classmethod
    def validate_google_api_key(cls, v: Optional[str], info) -> Optional[str]:
        provider = info.data.get("LLM_PROVIDER")
        vision_provider = info.data.get("VISION_PROVIDER")
        embedding_provider = info.data.get("EMBEDDING_PROVIDER")
        if provider is None and vision_provider is None and embedding_provider is None:
            return v
        provider = (provider or "openai").lower()
        vision_provider = (vision_provider or "none").lower()
        embedding_provider = (embedding_provider or "openai").lower()
        needs_google = provider == "google" or vision_provider == "google" or embedding_provider == "google"
        if needs_google and (not v or not v.strip() or v.startswith("your_") or v == "YOUR_API_KEY_HERE"):
            raise ValueError("GOOGLE_API_KEY is required when using Google provider. Set it in your .env file or HF Secrets.")
        return v

    @model_validator(mode="after")
    def validate_provider_credentials(self):
        provider = (self.LLM_PROVIDER or "").lower()
        embedding_provider = (self.EMBEDDING_PROVIDER or "").lower()
        vision_provider = (self.VISION_PROVIDER or "").lower()

        def _is_valid_key(value: Optional[str]) -> bool:
            if not value:
                return False
            cleaned = value.strip()
            if not cleaned:
                return False
            if cleaned.startswith("your_") or cleaned == "YOUR_API_KEY_HERE":
                return False
            return True

        if provider == "openai" or embedding_provider == "openai":
            if not self.OPENAI_API_KEY or not self.OPENAI_API_KEY.strip() or self.OPENAI_API_KEY.startswith("your_") or self.OPENAI_API_KEY == "YOUR_API_KEY_HERE":
                raise ValueError("OPENAI_API_KEY is required when using OpenAI provider. Set it in your .env file.")

        if provider == "azure":
            if not _is_valid_key(self.AZURE_OPENAI_API_KEY):
                raise ValueError("AZURE_OPENAI_API_KEY is required when LLM_PROVIDER=azure. Set it in your .env file.")
            if not self.AZURE_OPENAI_ENDPOINT or not self.AZURE_OPENAI_ENDPOINT.strip():
                raise ValueError("AZURE_OPENAI_ENDPOINT is required when LLM_PROVIDER=azure.")
            if not self.AZURE_OPENAI_DEPLOYMENT or not self.AZURE_OPENAI_DEPLOYMENT.strip():
                raise ValueError("AZURE_OPENAI_DEPLOYMENT is required when using Azure OpenAI chat.")
            api_version = (self.AZURE_OPENAI_API_VERSION or "").strip()
            version_match = re.match(r"^(\d{4})-(\d{2})-(\d{2})", api_version)
            if not version_match:
                raise ValueError(
                    "AZURE_OPENAI_API_VERSION must start with an ISO date (e.g., 2024-08-01-preview) "
                    "when LLM_PROVIDER=azure."
                )
            year, month, day = map(int, version_match.groups())
            if (year, month, day) < (2024, 8, 1):
                raise ValueError(
                    "AZURE_OPENAI_API_VERSION must be 2024-08-01-preview or later when LLM_PROVIDER=azure "
                    "because structured json_schema output is required."
                )

        if embedding_provider == "azure":
            has_embedding_key = _is_valid_key(self.AZURE_EMBEDDING_API_KEY) or _is_valid_key(self.AZURE_OPENAI_API_KEY)
            if not has_embedding_key:
                raise ValueError(
                    "Azure embeddings require AZURE_EMBEDDING_API_KEY or AZURE_OPENAI_API_KEY."
                )
            has_embedding_endpoint = bool(self.AZURE_EMBEDDING_ENDPOINT and self.AZURE_EMBEDDING_ENDPOINT.strip())
            has_chat_endpoint = bool(self.AZURE_OPENAI_ENDPOINT and self.AZURE_OPENAI_ENDPOINT.strip())
            if not (has_embedding_endpoint or has_chat_endpoint):
                raise ValueError(
                    "Azure embeddings require AZURE_EMBEDDING_ENDPOINT or AZURE_OPENAI_ENDPOINT."
                )
            if not self.AZURE_EMBEDDING_DEPLOYMENT or not self.AZURE_EMBEDDING_DEPLOYMENT.strip():
                raise ValueError("AZURE_EMBEDDING_DEPLOYMENT is required when EMBEDDING_PROVIDER=azure.")

        if provider == "google" or vision_provider == "google" or embedding_provider == "google":
            if not self.GOOGLE_API_KEY or not self.GOOGLE_API_KEY.strip() or self.GOOGLE_API_KEY.startswith("your_") or self.GOOGLE_API_KEY == "YOUR_API_KEY_HERE":
                raise ValueError("GOOGLE_API_KEY is required when using Google provider. Set it in your .env file or HF Secrets.")

        return self


def _get_parameters():
    """Get parameters instance with helpful error messages."""
    is_hf_space = os.environ.get("SPACE_ID") is not None
    
    try:
        return Settings()
    except Exception as e:
        import sys
        print(f"⚠️  Configuration Error: {e}", file=sys.stderr)
        
        if is_hf_space:
            print("💡 Tip: Add GOOGLE_API_KEY in Space Settings > Repository secrets", file=sys.stderr)
        else:
            print("💡 Tip: Create a .env file with GOOGLE_API_KEY=your_api_key", file=sys.stderr)
        
        raise


# Create parameters instance
parameters = _get_parameters()
