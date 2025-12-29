"""
Logging utility module.

This module provides a centralized logger instance using the standard library logging.
The logging configuration is handled by config/logger_setup.py which should be
called at application startup.

Usage:
    from core.logging import logger
    logger.info("Your message here")
"""
import logging

# Get a logger for the smartdoc module
# The actual configuration (handlers, formatters) is done in config/logger_setup.py
logger = logging.getLogger("smartdoc")