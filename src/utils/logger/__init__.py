"""
Smart Logging System for Hyperspectral Quality Prediction

This package provides a comprehensive logging system designed to replace
print statements throughout the codebase with intelligent, structured logging.

Key Features:
- Duplicate message suppression
- Contextual logging with experiment tracking
- Extensible design for future enhancements
- CPU-optimized performance logging
"""

from .base_logger import BaseLogger
from .smart_logger import SmartLogger
from .logger_config import LoggerConfig, LogLevel
from .logger_factory import LoggerFactory

# Default logger instance
_default_logger = None

def get_logger(name: str = None) -> SmartLogger:
    """
    Get a logger instance
    
    Args:
        name: Logger name (optional, uses 'hyperspectral' if None)
        
    Returns:
        SmartLogger instance
    """
    global _default_logger
    
    if _default_logger is None:
        _default_logger = LoggerFactory.create_smart_logger(name or 'hyperspectral')
    
    return _default_logger

def setup_logging(config: LoggerConfig = None) -> SmartLogger:
    """
    Setup logging system with configuration
    
    Args:
        config: Logger configuration
        
    Returns:
        Configured SmartLogger instance
    """
    global _default_logger
    _default_logger = LoggerFactory.create_smart_logger(config=config)
    return _default_logger

# Convenience functions for quick logging
def info(message: str, **kwargs):
    """Log info message"""
    get_logger().info(message, **kwargs)

def warning(message: str, **kwargs):
    """Log warning message"""
    get_logger().warning(message, **kwargs)

def error(message: str, **kwargs):
    """Log error message"""
    get_logger().error(message, **kwargs)

def debug(message: str, **kwargs):
    """Log debug message"""
    get_logger().debug(message, **kwargs)

def experiment(message: str, **kwargs):
    """Log experiment-specific message"""
    get_logger().experiment(message, **kwargs)

def performance(message: str, **kwargs):
    """Log performance-related message"""
    get_logger().performance(message, **kwargs)

def stage(message: str, **kwargs):
    """Log stage progression message"""
    get_logger().stage(message, **kwargs)

__all__ = [
    'BaseLogger', 'SmartLogger', 'LoggerConfig', 
    'LogLevel', 'LoggerFactory', 'get_logger', 'setup_logging',
    'info', 'warning', 'error', 'debug', 'experiment', 'performance', 'stage'
]
