"""
Centralized logging configuration for the GAIA agent system.

This module provides utilities for consistent logging configuration across the codebase,
including functions for creating loggers with standardized formatting and handlers.
"""

import logging
import os
import sys
from typing import Optional, Dict, Any, List
import json
from datetime import datetime

def configure_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    include_console: bool = True
) -> logging.Logger:
    """
    Configure and return a logger with consistent settings.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to log to
        format_string: Optional format string for log messages
        include_console: Whether to include a console handler
        
    Returns:
        Configured logger
    """
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler if requested
    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name, creating it if it doesn't exist.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

def set_log_level(name: str, level: int) -> None:
    """
    Set the log level for a logger.
    
    Args:
        name: Logger name
        level: Logging level
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

def configure_json_logging(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    include_console: bool = True,
    additional_fields: Optional[Dict[str, Any]] = None
) -> logging.Logger:
    """
    Configure a logger to output JSON-formatted logs.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional file to log to
        include_console: Whether to include a console handler
        additional_fields: Additional fields to include in every log message
        
    Returns:
        Configured logger
    """
    class JsonFormatter(logging.Formatter):
        def format(self, record):
            log_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "name": record.name,
                "level": record.levelname,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add exception info if available
            if record.exc_info:
                log_data["exception"] = {
                    "type": record.exc_info[0].__name__,
                    "message": str(record.exc_info[1]),
                    "traceback": self.formatException(record.exc_info)
                }
            
            # Add additional fields
            if additional_fields:
                log_data.update(additional_fields)
            
            return json.dumps(log_data)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = JsonFormatter()
    
    # Create console handler if requested
    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Create file handler if log_file is specified
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def configure_root_logger(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None
) -> None:
    """
    Configure the root logger.
    
    Args:
        level: Logging level
        log_file: Optional file to log to
        format_string: Optional format string for log messages
    """
    # Configure the root logger
    configure_logging("", level, log_file, format_string)
    
    # Silence noisy loggers
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)