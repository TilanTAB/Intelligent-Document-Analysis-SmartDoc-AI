import atexit
import logging
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
import os
from pathlib import Path
import queue
import re
import sys
import threading
from typing import Optional

# Redact API keys / bearer tokens before logs are written to disk or console.
_SECRET_PATTERNS = [
    re.compile(r"(?i)(\b(?:openai|azure_openai|google)_api_key\b\s*[=:]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(\bapi[_-]?key\s*[=:]\s*)([^\s,;]+)"),
    re.compile(r"(?i)(\bkey\b\s*[=:]\s*)([A-Za-z0-9_-]{20,})"),
    re.compile(r"(?i)(\bauthorization\b\s*:\s*bearer\s+)([A-Za-z0-9\-._~+/=]+)"),
    re.compile(r"(?i)(api[-_]?key=)([^&\s]+)"),
    re.compile(r"\bAIza[0-9A-Za-z\-_]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9\-_]{16,}\b"),
]

_setup_lock = threading.Lock()
_listener: Optional[QueueListener] = None
_configured = False


def _redact_secrets(text: str) -> str:
    redacted = text
    for pattern in _SECRET_PATTERNS:
        if pattern.groups >= 2:
            redacted = pattern.sub(lambda m: f"{m.group(1)}<REDACTED>", redacted)
        else:
            redacted = pattern.sub("<REDACTED>", redacted)
    return redacted


class SafeFormatter(logging.Formatter):
    """Formatter that redacts secrets and strips unsupported Unicode chars."""

    def format(self, record):
        msg = super().format(record)
        msg = _redact_secrets(msg)
        # Remove characters not supported by cp1252 (0-255)
        safe_msg = "".join(c if ord(c) < 256 else "?" for c in msg)
        return safe_msg


def _remove_root_handlers(root_logger: logging.Logger) -> None:
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        # Never close stdio-backed stream handlers.
        if isinstance(handler, logging.StreamHandler):
            continue
        try:
            handler.close()
        except Exception:
            pass


def configure_logging(level: int = logging.INFO, force: bool = False) -> None:
    """
    Configure process-local logging once.

    Why:
    - Avoid import-time side effects in worker processes.
    - Prevent duplicate handlers/listeners when called multiple times.
    """
    global _configured, _listener

    with _setup_lock:
        if _configured and not force:
            return

        if _listener is not None:
            try:
                _listener.stop()
            except Exception:
                pass
            _listener = None

        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        log_file_path = os.path.join("logs", "app.log")

        detailed_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

        log_queue: queue.Queue = queue.Queue(-1)

        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,              # Keep 5 backups
            delay=True,                 # Delay file opening until first log
        )
        file_handler.setFormatter(SafeFormatter(detailed_format))

        queue_handler = QueueHandler(log_queue)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(SafeFormatter(detailed_format))
        console_handler.setLevel(level)

        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        _remove_root_handlers(root_logger)
        root_logger.handlers = [console_handler, queue_handler]

        _listener = QueueListener(log_queue, file_handler)
        _listener.start()
        _configured = True

        # Suppress noisy third-party logs.
        logging.getLogger("langchain").setLevel(logging.WARNING)
        logging.getLogger("langchain_community").setLevel(logging.WARNING)
        logging.getLogger("chromadb").setLevel(logging.WARNING)
        logging.getLogger("google").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)

        root_logger.info("Logging system initialized successfully.")


def shutdown_logging() -> None:
    """Best-effort listener shutdown for clean process exit."""
    global _listener, _configured
    with _setup_lock:
        if _listener is not None:
            try:
                _listener.stop()
            except Exception:
                pass
            _listener = None
        _configured = False


atexit.register(shutdown_logging)

