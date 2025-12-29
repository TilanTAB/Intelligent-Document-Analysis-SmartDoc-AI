"""
Health check utilities for DocChat.

This module provides diagnostics check functions that can be used
to verify the application is running correctly.
"""
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def check_diagnostics() -> Dict[str, Any]:
    """
    Perform a comprehensive diagnostics check of the application.
    
    Returns:
        Dict with diagnostics status and component information
    """
    diagnostics_status = {
        "status": "diagnosticsy",
        "timestamp": datetime.utcnow().isoformat(),
        "components": {}
    }
    
    # Check parameters
    try:
        from configuration.parameters import parameters
        diagnostics_status["components"]["parameters"] = {
            "status": "ok",
            "chroma_db_path": parameters.CHROMA_DB_PATH,
            "log_level": parameters.LOG_LEVEL
        }
    except Exception as e:
        diagnostics_status["components"]["parameters"] = {
            "status": "error",
            "error": str(e)
        }
        diagnostics_status["status"] = "undiagnosticsy"
    
    # Check ChromaDB directory
    try:
        from pathlib import Path
        chroma_path = Path(parameters.CHROMA_DB_PATH)
        diagnostics_status["components"]["chroma_db"] = {
            "status": "ok",
            "path_exists": chroma_path.exists(),
            "is_writable": chroma_path.exists() and chroma_path.is_dir()
        }
    except Exception as e:
        diagnostics_status["components"]["chroma_db"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check cache directory
    try:
        cache_path = Path(parameters.CACHE_DIR)
        diagnostics_status["components"]["cache"] = {
            "status": "ok",
            "path_exists": cache_path.exists(),
            "is_writable": cache_path.exists() and cache_path.is_dir()
        }
    except Exception as e:
        diagnostics_status["components"]["cache"] = {
            "status": "error",
            "error": str(e)
        }
    
    # Check if required packages are importable
    required_packages = [
        "langchain",
        "langchain_google_genai",
        "chromadb",
        "gradio"
    ]
    
    packages_status = {}
    for package in required_packages:
        try:
            __import__(package)
            packages_status[package] = "ok"
        except ImportError as e:
            packages_status[package] = f"missing: {e}"
            diagnostics_status["status"] = "degraded"
    
    diagnostics_status["components"]["packages"] = packages_status
    
    return diagnostics_status


def check_api_key() -> Dict[str, Any]:
    """
    Check if the Google API key is configured and valid format.
    
    Returns:
        Dict with API key status (does not expose the key)
    """
    try:
        from configuration.parameters import parameters
        api_key = parameters.GOOGLE_API_KEY
        
        if not api_key:
            return {"status": "missing", "message": "GOOGLE_API_KEY not set"}
        
        if len(api_key) < 20:
            return {"status": "invalid", "message": "API key appears too short"}
        
        # Mask the key for logging (show first 4 and last 4 chars)
        masked = f"{api_key[:4]}...{api_key[-4:]}"
        
        return {
            "status": "configured",
            "masked_key": masked,
            "length": len(api_key)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    # Run diagnostics check when executed directly
    import json
    print(json.dumps(check_diagnostics(), indent=2))
