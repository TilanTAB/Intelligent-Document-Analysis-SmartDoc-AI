import logging
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
import queue
import os
import sys
from pathlib import Path

# Custom formatter to remove unsupported Unicode characters
class SafeFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        # Remove characters not supported by cp1252 (0-255)
        safe_msg = ''.join(c if ord(c) < 256 else '?' for c in msg)
        return safe_msg

# Ensure the logs directory exists
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

# Configure log file path
log_file_path = os.path.join("logs", "app.log")

# Set up a queue for log messages
log_queue = queue.Queue(-1)  # No limit on size (-1)

# Detailed log format with timestamp, level, logger name, and message
detailed_format = "%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"

# Create a rotating file handler for the application logs
file_handler = RotatingFileHandler(
    log_file_path,
    maxBytes=10 * 1024 * 1024,  # 10 MB
    backupCount=5,              # Keep 5 backups
    delay=True                   # Delay file opening until a log message is emitted
)
file_handler.setFormatter(SafeFormatter(detailed_format))

# Create a queue handler to send log messages to the queue
queue_handler = QueueHandler(log_queue)

# Console handler (direct, not via queue)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(SafeFormatter(detailed_format))
console_handler.setLevel(logging.INFO)

# Get the root logger
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
root_logger.handlers = [console_handler, queue_handler]  # Console direct, queue for file

# Create and start a listener for the queue to process log messages in the background
listener = QueueListener(log_queue, file_handler)
listener.start()

# Suppress verbose logs from specific third-party libraries
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_community").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

root_logger.info("Logging system initialized successfully.")
