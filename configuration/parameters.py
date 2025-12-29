from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import Optional
import os
from .definitions import MAX_FILE_SIZE, MAX_TOTAL_SIZE, ALLOWED_TYPES


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

    # API keys - REQUIRED, must be set via environment variable or HF Secrets
    GOOGLE_API_KEY: str = Field(
        ...,  # Required field
        description="Google API key for Gemini models",
    )

    # Database parameters
    CHROMA_DB_PATH: str = "./chroma_db"

    # Chunking parameters
    CHUNK_SIZE: int = 2000
    CHUNK_OVERLAP: int = 100

    # Retriever parameters
    VECTOR_SEARCH_K: int = 25
    VECTOR_Search_K_CHROMA: int = 15
    VECTOR_FETCH_K: int = 35
    VECTOR_SCORE_THRESHOLD: float = 0.3
    BM25_SEARCH_K: int = 8
    HYBRID_RETRIEVER_WEIGHTS: list = [0.4, 0.6]  # [BM25 weight, Vector weight]
    CHROMA_COLLECTION_NAME: str = "documents"

    # Workflow parameters
    MAX_RESEARCH_ATTEMPTS: int = 2
    ENABLE_QUERY_REWRITING: bool = True
    MAX_QUERY_REWRITES: int = 1
    RELEVANCE_CHECK_K: int = 20

    # Research agent parameters
    RESEARCH_TOP_K: int = 15
    RESEARCH_MAX_CONTEXT_CHARS: int = 8000000000
    RESEARCH_MAX_OUTPUT_TOKENS: int = 500

    # Verification parameters
    VERIFICATION_MAX_CONTEXT_CHARS: int = 800000000
    VERIFICATION_MAX_OUTPUT_TOKENS: int = 300

    # Logging parameters
    LOG_LEVEL: str = "INFO"

    # Cache parameters
    CACHE_DIR: str = "document_cache"
    CACHE_EXPIRE_DAYS: int = 7

    # LLM parameters
    LLM_MAX_RETRIES: int = 3
    LLM_RETRY_DELAY: float = 1.0
    LLM_MODEL_NAME: str = "gemini-2.5-flash-lite"  # Default model for all agents
    
    # Agent-specific LLM models (override LLM_MODEL_NAME if needed)
    RESEARCH_AGENT_MODEL: str = "gemini-2.5-flash-lite"
    VERIFICATION_AGENT_MODEL: str = "gemini-2.5-flash-lite"
    RELEVANCE_CHECKER_MODEL: str = "gemini-2.5-flash-lite"

    # Chart extraction parameters
    ENABLE_CHART_EXTRACTION: bool = True
    CHART_VISION_MODEL: str = "gemini-2.5-flash-lite"
    CHART_MAX_TOKENS: int = 1500
    CHART_DPI: int = 150  # Lower DPI saves memory
    CHART_BATCH_SIZE: int = 3  # Process pages in batches
    CHART_MAX_IMAGE_SIZE: int = 1920  # Max dimension for images

    # Local chart detection parameters (cost optimization)
    CHART_USE_LOCAL_DETECTION: bool = True  # Use OpenCV first (FREE)
    CHART_MIN_CONFIDENCE: float = 0.4  # Only analyze charts with confidence > 40%
    CHART_SKIP_GEMINI_DETECTION: bool = True  # Skip Gemini for detection, only use for analysis
    CHART_GEMINI_FALLBACK_ENABLED: bool = False  # Optional: Use Gemini if local fails

    # Gemini batch processing parameters (speed optimization - 2-3× faster)
    CHART_GEMINI_BATCH_SIZE: int = 1  # Analyze 1 chart per API call (reduced from 2 for reliability)
    CHART_ENABLE_BATCH_ANALYSIS: bool = True  # Enable batch processing for speed

    @field_validator("GOOGLE_API_KEY")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate that API key is provided and not a placeholder."""
        if not v or v.strip() == "":
            raise ValueError("GOOGLE_API_KEY is required. Set it in your .env file or HF Secrets.")
        if v.startswith("your_") or v == "YOUR_API_KEY_HERE":
            raise ValueError("Please replace the placeholder GOOGLE_API_KEY with your actual API key.")
        return v


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
