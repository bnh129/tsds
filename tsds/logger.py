"""
Centralized logging configuration for TSDS.
Provides structured logging with file output and configurable levels.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


class TSDBLogger:
    """Centralized logger for TSDS components."""
    
    _loggers = {}
    _initialized = False
    _log_dir = None
    _log_level = logging.INFO
    
    @classmethod
    def setup(cls, log_dir: str = "./logs", log_level: str = "INFO", console_output: bool = False):
        """Setup logging configuration for all TSDS components."""
        if cls._initialized:
            return
        
        cls._log_dir = Path(log_dir)
        cls._log_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert string level to logging constant
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        cls._log_level = level_map.get(log_level.upper(), logging.INFO)
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        simple_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # Setup root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(cls._log_level)
        root_logger.handlers.clear()
        
        # File handler for all logs
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = cls._log_dir / f"tsds_{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(cls._log_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)
        
        # Console handler (optional)
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.WARNING)  # Only warnings/errors to console
            console_handler.setFormatter(simple_formatter)
            root_logger.addHandler(console_handler)
        
        cls._initialized = True
        
        # Log the initialization
        init_logger = cls.get_logger("TSDBLogger")
        init_logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
        if console_output:
            init_logger.info("Console output enabled for WARNING+ messages")
    
    @classmethod
    def get_logger(cls, name: str) -> logging.Logger:
        """Get a logger for a specific component."""
        if not cls._initialized:
            cls.setup()  # Initialize with defaults if not already done
        
        if name not in cls._loggers:
            logger = logging.getLogger(name)
            logger.setLevel(cls._log_level)
            cls._loggers[name] = logger
        
        return cls._loggers[name]
    
    @classmethod
    def set_level(cls, level: str):
        """Change logging level for all loggers."""
        level_map = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL
        }
        new_level = level_map.get(level.upper(), logging.INFO)
        
        root_logger = logging.getLogger()
        root_logger.setLevel(new_level)
        
        for logger in cls._loggers.values():
            logger.setLevel(new_level)
        
        cls._log_level = new_level
    
    @classmethod
    def get_log_file(cls) -> Optional[Path]:
        """Get the current log file path."""
        if not cls._initialized:
            return None
        
        # Find the most recent log file
        if cls._log_dir and cls._log_dir.exists():
            log_files = sorted(cls._log_dir.glob("tsds_*.log"), key=lambda x: x.stat().st_mtime)
            if log_files:
                return log_files[-1]
        return None


def get_logger(name: str) -> logging.Logger:
    """Convenience function to get a logger."""
    return TSDBLogger.get_logger(name)