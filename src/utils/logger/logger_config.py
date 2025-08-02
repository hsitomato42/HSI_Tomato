"""
Logger Configuration System

Defines configuration classes and enums for the smart logging system.
"""

import os
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


class LogLevel(Enum):
    """Log level enumeration"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50
    
    # Special levels for ML workflows
    EXPERIMENT = 25    # Between INFO and WARNING
    PERFORMANCE = 15   # Between DEBUG and INFO
    STAGE = 22        # Between INFO and EXPERIMENT


@dataclass
class LoggerConfig:
    """
    Configuration for the smart logger system
    """
    
    # Basic settings
    name: str = "hyperspectral"
    level: LogLevel = LogLevel.INFO
    
    # Output settings
    log_to_console: bool = True
    log_to_file: bool = True
    log_file_path: Optional[str] = None
    
    # Formatting
    include_timestamp: bool = True
    include_level: bool = True
    include_name: bool = True
    include_context: bool = True
    
    # Duplicate suppression
    enable_duplicate_suppression: bool = True
    duplicate_window_seconds: float = 5.0
    max_duplicate_count: int = 3
    
    # Performance settings
    buffer_size: int = 100
    flush_interval: float = 1.0
    
    # Experiment-specific settings
    experiment_name: Optional[str] = None
    results_dir: Optional[str] = None
    
    # Context tracking
    track_experiment_context: bool = True
    track_stage_context: bool = True
    track_performance_metrics: bool = True
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        if self.log_file_path is None and self.log_to_file:
            # Import paths config
            from src.config.paths import LOGS_DIR
            
            # Use centralized log directory
            log_dir = Path(LOGS_DIR)
            log_dir.mkdir(parents=True, exist_ok=True)
            
            if self.experiment_name:
                self.log_file_path = str(log_dir / f"{self.experiment_name}.log")
            else:
                self.log_file_path = str(log_dir / f"{self.name}.log")
    
    @classmethod
    def for_experiment(cls, experiment_name: str, results_dir: str = None, **kwargs) -> 'LoggerConfig':
        """
        Create configuration optimized for experiment logging
        
        Args:
            experiment_name: Name of the experiment
            results_dir: Directory to store logs
            **kwargs: Additional configuration options
            
        Returns:
            LoggerConfig instance
        """
        defaults = {
            'name': f"exp_{experiment_name}",
            'experiment_name': experiment_name,
            'results_dir': results_dir,
            'level': LogLevel.INFO,
            'enable_duplicate_suppression': True,
            'track_experiment_context': True,
            'track_stage_context': True,
            'track_performance_metrics': True
        }
        defaults.update(kwargs)
        
        return cls(**defaults)
    
    @classmethod
    def for_development(cls, **kwargs) -> 'LoggerConfig':
        """
        Create configuration optimized for development
        
        Args:
            **kwargs: Additional configuration options
            
        Returns:
            LoggerConfig instance
        """
        defaults = {
            'level': LogLevel.DEBUG,
            'log_to_file': False,
            'enable_duplicate_suppression': False,
            'include_context': True
        }
        defaults.update(kwargs)
        
        return cls(**defaults)
    
    @classmethod
    def for_production(cls, **kwargs) -> 'LoggerConfig':
        """
        Create configuration optimized for production
        
        Args:
            **kwargs: Additional configuration options
            
        Returns:
            LoggerConfig instance
        """
        defaults = {
            'level': LogLevel.INFO,
            'log_to_file': True,
            'enable_duplicate_suppression': True,
            'buffer_size': 200,
            'flush_interval': 0.5
        }
        defaults.update(kwargs)
        
        return cls(**defaults)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'name': self.name,
            'level': self.level.name,
            'log_to_console': self.log_to_console,
            'log_to_file': self.log_to_file,
            'log_file_path': self.log_file_path,
            'include_timestamp': self.include_timestamp,
            'include_level': self.include_level,
            'include_name': self.include_name,
            'include_context': self.include_context,
            'enable_duplicate_suppression': self.enable_duplicate_suppression,
            'duplicate_window_seconds': self.duplicate_window_seconds,
            'max_duplicate_count': self.max_duplicate_count,
            'experiment_name': self.experiment_name,
            'results_dir': self.results_dir
        }


class MessageType(Enum):
    """Message type classification for intelligent routing"""
    GENERAL = "general"
    EXPERIMENT = "experiment"
    STAGE = "stage"
    PERFORMANCE = "performance"
    ERROR = "error"
    WARNING = "warning"
    DEBUG = "debug"


class LogContext:
    """
    Context manager for logger state
    """
    
    def __init__(self):
        self.experiment_name: Optional[str] = None
        self.stage_name: Optional[str] = None
        self.epoch: Optional[int] = None
        self.batch: Optional[int] = None
        self.model_type: Optional[str] = None
        self.additional_context: Dict[str, Any] = {}
    
    def set_experiment(self, name: str):
        """Set experiment context"""
        self.experiment_name = name
    
    def set_stage(self, name: str):
        """Set stage context"""
        self.stage_name = name
    
    def set_epoch(self, epoch: int):
        """Set epoch context"""
        self.epoch = epoch
    
    def set_batch(self, batch: int):
        """Set batch context"""
        self.batch = batch
    
    def set_model(self, model_type: str):
        """Set model context"""
        self.model_type = model_type
    
    def set_context(self, **kwargs):
        """Set additional context"""
        self.additional_context.update(kwargs)
    
    def clear_context(self):
        """Clear all context"""
        self.experiment_name = None
        self.stage_name = None
        self.epoch = None
        self.batch = None
        self.model_type = None
        self.additional_context.clear()
    
    def get_context_string(self) -> str:
        """Get formatted context string"""
        parts = []
        
        if self.experiment_name:
            parts.append(f"exp:{self.experiment_name}")
        if self.stage_name:
            parts.append(f"stage:{self.stage_name}")
        if self.epoch is not None:
            parts.append(f"epoch:{self.epoch}")
        if self.batch is not None:
            parts.append(f"batch:{self.batch}")
        if self.model_type:
            parts.append(f"model:{self.model_type}")
        
        # Add additional context
        for key, value in self.additional_context.items():
            parts.append(f"{key}:{value}")
        
        return f"[{' | '.join(parts)}]" if parts else "" 