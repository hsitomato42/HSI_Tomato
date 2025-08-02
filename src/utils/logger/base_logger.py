"""
Base Logger Class

Defines the abstract interface for all logger implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from .logger_config import LoggerConfig, LogLevel, LogContext, MessageType


class BaseLogger(ABC):
    """
    Abstract base class for all logger implementations
    
    This class defines the interface that all logger implementations must follow.
    It provides a consistent API while allowing for different implementation strategies.
    """
    
    def __init__(self, config: LoggerConfig):
        """
        Initialize the base logger
        
        Args:
            config: Logger configuration
        """
        self.config = config
        self.context = LogContext()
        self._is_initialized = False
    
    @abstractmethod
    def _initialize(self) -> None:
        """Initialize the logger implementation"""
        pass
    
    @abstractmethod
    def _log(self, level: LogLevel, message: str, message_type: MessageType = MessageType.GENERAL, **kwargs) -> None:
        """
        Internal logging method that must be implemented by subclasses
        
        Args:
            level: Log level
            message: Message to log
            message_type: Type of message for intelligent routing
            **kwargs: Additional context data
        """
        pass
    
    @abstractmethod
    def _should_log(self, level: LogLevel, message: str, message_type: MessageType) -> bool:
        """
        Determine if a message should be logged
        
        Args:
            level: Log level
            message: Message to check
            message_type: Type of message
            
        Returns:
            True if message should be logged
        """
        pass
    
    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered messages"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the logger and release resources"""
        pass
    
    def ensure_initialized(self) -> None:
        """Ensure the logger is initialized"""
        if not self._is_initialized:
            self._initialize()
            self._is_initialized = True
    
    # Public logging methods
    def debug(self, message: str, **kwargs) -> None:
        """Log debug message"""
        self.ensure_initialized()
        if self._should_log(LogLevel.DEBUG, message, MessageType.DEBUG):
            self._log(LogLevel.DEBUG, message, MessageType.DEBUG, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """Log info message"""
        self.ensure_initialized()
        if self._should_log(LogLevel.INFO, message, MessageType.GENERAL):
            self._log(LogLevel.INFO, message, MessageType.GENERAL, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """Log warning message"""
        self.ensure_initialized()
        if self._should_log(LogLevel.WARNING, message, MessageType.WARNING):
            self._log(LogLevel.WARNING, message, MessageType.WARNING, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """Log error message"""
        self.ensure_initialized()
        if self._should_log(LogLevel.ERROR, message, MessageType.ERROR):
            self._log(LogLevel.ERROR, message, MessageType.ERROR, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """Log critical message"""
        self.ensure_initialized()
        if self._should_log(LogLevel.CRITICAL, message, MessageType.ERROR):
            self._log(LogLevel.CRITICAL, message, MessageType.ERROR, **kwargs)
    
    def experiment(self, message: str, **kwargs) -> None:
        """Log experiment-specific message"""
        self.ensure_initialized()
        if self._should_log(LogLevel.EXPERIMENT, message, MessageType.EXPERIMENT):
            self._log(LogLevel.EXPERIMENT, message, MessageType.EXPERIMENT, **kwargs)
    
    def performance(self, message: str, **kwargs) -> None:
        """Log performance-related message"""
        self.ensure_initialized()
        if self._should_log(LogLevel.PERFORMANCE, message, MessageType.PERFORMANCE):
            self._log(LogLevel.PERFORMANCE, message, MessageType.PERFORMANCE, **kwargs)
    
    def stage(self, message: str, **kwargs) -> None:
        """Log stage progression message"""
        self.ensure_initialized()
        if self._should_log(LogLevel.STAGE, message, MessageType.STAGE):
            self._log(LogLevel.STAGE, message, MessageType.STAGE, **kwargs)
    
    # Context management methods
    def set_experiment_context(self, experiment_name: str) -> None:
        """Set experiment context"""
        self.context.set_experiment(experiment_name)
    
    def set_stage_context(self, stage_name: str) -> None:
        """Set stage context"""
        self.context.set_stage(stage_name)
    
    def set_epoch_context(self, epoch: int) -> None:
        """Set epoch context"""
        self.context.set_epoch(epoch)
    
    def set_batch_context(self, batch: int) -> None:
        """Set batch context"""
        self.context.set_batch(batch)
    
    def set_model_context(self, model_type: str) -> None:
        """Set model context"""
        self.context.set_model(model_type)
    
    def set_context(self, **kwargs) -> None:
        """Set additional context"""
        self.context.set_context(**kwargs)
    
    def clear_context(self) -> None:
        """Clear all context"""
        self.context.clear_context()
    
    # Context managers for scoped logging
    class ExperimentContext:
        """Context manager for experiment-scoped logging"""
        
        def __init__(self, logger: 'BaseLogger', experiment_name: str):
            self.logger = logger
            self.experiment_name = experiment_name
            self.previous_experiment = None
        
        def __enter__(self):
            self.previous_experiment = self.logger.context.experiment_name
            self.logger.set_experiment_context(self.experiment_name)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.previous_experiment:
                self.logger.set_experiment_context(self.previous_experiment)
            else:
                self.logger.context.experiment_name = None
    
    class StageContext:
        """Context manager for stage-scoped logging"""
        
        def __init__(self, logger: 'BaseLogger', stage_name: str):
            self.logger = logger
            self.stage_name = stage_name
            self.previous_stage = None
        
        def __enter__(self):
            self.previous_stage = self.logger.context.stage_name
            self.logger.set_stage_context(self.stage_name)
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.previous_stage:
                self.logger.set_stage_context(self.previous_stage)
            else:
                self.logger.context.stage_name = None
    
    def experiment_context(self, experiment_name: str) -> 'BaseLogger.ExperimentContext':
        """Create experiment context manager"""
        return self.ExperimentContext(self, experiment_name)
    
    def stage_context(self, stage_name: str) -> 'BaseLogger.StageContext':
        """Create stage context manager"""
        return self.StageContext(self, stage_name)
    
    # Utility methods
    def get_effective_level(self) -> LogLevel:
        """Get the effective logging level"""
        return self.config.level
    
    def is_enabled_for(self, level: LogLevel) -> bool:
        """Check if logging is enabled for the given level"""
        return level.value >= self.config.level.value
    
    def get_context_string(self) -> str:
        """Get formatted context string"""
        return self.context.get_context_string()
    
    def __enter__(self):
        """Context manager entry"""
        self.ensure_initialized()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.flush()
        if exc_type:
            self.error(f"Exception occurred: {exc_val}")
        self.close() 