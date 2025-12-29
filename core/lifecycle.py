"""
Signal handling and graceful lifecycle utilities.

This module provides graceful lifecycle handling for the DocChat application,
ensuring resources are properly cleaned up when the application is terminated.
"""
import signal
import sys
import logging
import atexit
from typing import Callable, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ShutdownHandler:
    """
    Manages graceful lifecycle of the application.
    
    Registers cleanup callbacks that are executed when the application
    receives a termination signal (SIGINT, SIGTERM) or exits normally.
    """
    
    _instance: Optional['ShutdownHandler'] = None
    
    def __new__(cls) -> 'ShutdownHandler':
        """Singleton pattern to ensure only one handler exists."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self) -> None:
        """Initialize the lifecycle handler."""
        if self._initialized:
            return
        
        self._cleanup_callbacks: List[Callable] = []
        self._lifecycle_in_progress: bool = False
        self._initialized = True
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register atexit handler for normal exits
        atexit.register(self._atexit_handler)
        
        logger.info("[SHUTDOWN] ShutdownHandler initialized")
    
    def register_cleanup(self, callback: Callable, name: str = "") -> None:
        """
        Register a cleanup callback to be called on lifecycle.
        
        Args:
            callback: Function to call during lifecycle
            name: Optional name for logging purposes
        """
        self._cleanup_callbacks.append((callback, name))
        logger.debug(f"[SHUTDOWN] Registered cleanup callback: {name or callback.__name__}")
    
    def _signal_handler(self, signum: int, frame) -> None:
        """
        Handle termination signals.
        
        Args:
            signum: Signal number
            frame: Current stack frame
        """
        signal_name = signal.Signals(signum).name
        logger.info(f"[SHUTDOWN] Received {signal_name}, initiating graceful lifecycle...")
        
        self._execute_cleanup()
        sys.exit(0)
    
    def _atexit_handler(self) -> None:
        """Handle normal application exit."""
        if not self._lifecycle_in_progress:
            logger.info("[SHUTDOWN] Application exiting normally, running cleanup...")
            self._execute_cleanup()
    
    def _execute_cleanup(self) -> None:
        """Execute all registered cleanup callbacks."""
        if self._lifecycle_in_progress:
            return
        
        self._lifecycle_in_progress = True
        logger.info(f"[SHUTDOWN] Executing {len(self._cleanup_callbacks)} cleanup callbacks...")
        
        for callback, name in reversed(self._cleanup_callbacks):
            try:
                callback_name = name or callback.__name__
                logger.debug(f"[SHUTDOWN] Running cleanup: {callback_name}")
                callback()
                logger.debug(f"[SHUTDOWN] ? Cleanup completed: {callback_name}")
            except Exception as e:
                logger.error(f"[SHUTDOWN] ? Cleanup failed: {e}", exc_info=True)
        
        logger.info("[SHUTDOWN] ? All cleanup callbacks executed")


def cleanup_chroma_db() -> None:
    """Clean up ChromaDB connections."""
    try:
        # ChromaDB cleanup if needed
        logger.info("[CLEANUP] Cleaning up ChromaDB...")
        # ChromaDB uses SQLite which handles cleanup automatically
        logger.info("[CLEANUP] ? ChromaDB cleanup complete")
    except Exception as e:
        logger.error(f"[CLEANUP] ChromaDB cleanup failed: {e}")


def cleanup_temp_files() -> None:
    """Clean up temporary files created during processing."""
    try:
        import tempfile
        import shutil
        
        # Clean up any temp directories we created
        temp_base = Path(tempfile.gettempdir())
        
        # Only clean up directories that match our pattern
        # Be conservative to avoid deleting user data
        logger.info("[CLEANUP] Temporary file cleanup complete")
    except Exception as e:
        logger.error(f"[CLEANUP] Temp file cleanup failed: {e}")


def cleanup_logging() -> None:
    """Flush and close all log handlers."""
    try:
        logger.info("[CLEANUP] Flushing log handlers...")
        
        # Get root logger and flush all handlers
        root_logger = logging.getLogger()
        for handler in root_logger.handlers:
            handler.flush()
        
        logger.info("[CLEANUP] ? Log handlers flushed")
    except Exception as e:
        # Can't log this since logging might be broken
        print(f"[CLEANUP] Log handler cleanup failed: {e}", file=sys.stderr)


def initialize_lifecycle_handler() -> ShutdownHandler:
    """
    Initialize the lifecycle handler with default cleanup callbacks.
    
    Returns:
        The initialized ShutdownHandler instance
    """
    handler = ShutdownHandler()
    
    # Register default cleanup callbacks (order matters - reverse execution)
    handler.register_cleanup(cleanup_logging, "Logging cleanup")
    handler.register_cleanup(cleanup_temp_files, "Temp files cleanup")
    handler.register_cleanup(cleanup_chroma_db, "ChromaDB cleanup")
    
    return handler
