import os
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

# Create logs directory if it doesn't exist
LOGS_DIR = Path(__file__).parent.parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

def get_logger(
    name: str, 
    level: int = logging.INFO,
    log_to_file: bool = True,
    log_to_console: bool = True,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Configure and return a logger with the specified name.
    
    Args:
        name: Name of the logger, typically the module name
        level: Logging level (default: INFO)
        log_to_file: Whether to log to a file (default: True)
        log_to_console: Whether to log to console (default: True)
        log_file: Custom log file name (default: None, will use name)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Clear any existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
    
    # Add console handler if requested
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Add file handler if requested
    if log_to_file:
        if log_file is None:
            # Create a log file name based on the module
            module_name = name.split('.')[-1]
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = f"{module_name}_{timestamp}.log"
        
        file_path = LOGS_DIR / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_config(logger: logging.Logger, config: Dict[str, Any]) -> None:
    """
    Log a configuration dictionary in a readable format.
    
    Args:
        logger: The logger to use
        config: The configuration dictionary to log
    """
    logger.info("Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")

def log_execution_time(logger: logging.Logger, start_time: float, description: str) -> None:
    """
    Log the execution time of an operation.
    
    Args:
        logger: The logger to use
        start_time: The start time from time.time()
        description: Description of the operation
    """
    import time
    execution_time = time.time() - start_time
    logger.info(f"{description} completed in {execution_time:.4f} seconds")
