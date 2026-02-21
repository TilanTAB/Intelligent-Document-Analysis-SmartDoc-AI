"""
User analytics and interaction logging module.

Logs user interactions (IP addresses, questions, timestamps) for analytics
and monitoring purposes. Compatible with both local and Hugging Face Spaces.
"""
import logging
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
import threading

logger = logging.getLogger(__name__)

# Thread lock for file writes
_log_lock = threading.Lock()
_REDACTION_PATTERNS = [
    re.compile(r"(?i)(\b(?:openai|azure_openai|google)_api_key\b\s*[=:]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(\bapi[_-]?key\s*[=:]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(\bkey\b\s*[=:]\s*)([A-Za-z0-9_-]{20,})"),
    re.compile(r"(?i)(\bauthorization\b\s*:\s*bearer\s+)([A-Za-z0-9\-._~+/=]+)"),
    re.compile(r"(?i)(api[-_]?key=)([^&\s]+)"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9\-_]{16,}\b"),
]


def _redact_sensitive_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    redacted = value
    for pattern in _REDACTION_PATTERNS:
        if pattern.groups >= 2:
            redacted = pattern.sub(lambda m: f"{m.group(1)}<REDACTED>", redacted)
        else:
            redacted = pattern.sub("<REDACTED>", redacted)
    return redacted


class UserInteractionLogger:
    """Logs user interactions to JSON file for analytics."""
    
    def __init__(self, log_dir: str = "logs"):
        """
        Initialize the user interaction logger.
        
        Args:
            log_dir: Directory to store log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # For HF Spaces, use /tmp to avoid persistence issues
        is_hf_space = os.environ.get("SPACE_ID") is not None
        if is_hf_space:
            self.log_dir = Path("/tmp/logs")
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Running on HF Spaces - logs will be stored in: {self.log_dir}")
        
        self.log_file = self.log_dir / "user_interactions.jsonl"
        logger.info(f"User interaction logger initialized: {self.log_file}")
    
    def log_question(
        self,
        ip_address: str,
        question: str,
        num_files: int = 0,
        file_types: Optional[list] = None,
        success: bool = True,
        error: Optional[str] = None
    ):
        """
        Log a user question with metadata.
        
        Args:
            ip_address: User's IP address
            question: The question asked
            num_files: Number of files uploaded
            file_types: List of file extensions
            success: Whether the request succeeded
            error: Error message if failed
        """
        sanitized_question = _redact_sensitive_text(question) or ""
        sanitized_error = _redact_sensitive_text(error)

        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "ip": ip_address,
            "question": sanitized_question,
            "question_length": len(question),
            "num_files": num_files,
            "file_types": file_types or [],
            "success": success,
            "error": sanitized_error
        }
        
        # Thread-safe file write
        with _log_lock:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")
                logger.debug(f"Logged interaction from {ip_address}: {sanitized_question[:50]}...")
            except Exception as e:
                logger.error(f"Failed to log user interaction: {e}")
    
    def get_stats(self) -> dict:
        """
        Get usage statistics from logs.
        
        Returns:
            Dictionary with usage statistics
        """
        if not self.log_file.exists():
            return {
                "total_queries": 0,
                "unique_ips": 0,
                "success_rate": 0.0
            }
        
        try:
            unique_ips = set()
            total_queries = 0
            successful_queries = 0
            
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        unique_ips.add(entry.get("ip", "unknown"))
                        total_queries += 1
                        if entry.get("success", True):
                            successful_queries += 1
            
            return {
                "total_queries": total_queries,
                "unique_ips": len(unique_ips),
                "success_rate": (successful_queries / total_queries * 100) if total_queries > 0 else 0.0
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_queries": 0,
                "unique_ips": 0,
                "success_rate": 0.0
            }
    
    def export_logs(self, output_file: Optional[str] = None) -> str:
        """
        Export logs to a JSON file.
        
        Args:
            output_file: Output file path (optional)
        
        Returns:
            Path to the exported file
        """
        if output_file is None:
            output_file = str(self.log_dir / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        try:
            logs = []
            if self.log_file.exists():
                with open(self.log_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            logs.append(json.loads(line))
            
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(logs, f, indent=2)
            
            logger.info(f"Exported {len(logs)} log entries to {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Failed to export logs: {e}")
            raise


# Global instance
_analytics_logger = None


def get_analytics_logger() -> UserInteractionLogger:
    """Get the global analytics logger instance."""
    global _analytics_logger
    if _analytics_logger is None:
        _analytics_logger = UserInteractionLogger()
    return _analytics_logger
