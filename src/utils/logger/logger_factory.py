"""
Logger Factory

Factory class for creating different types of loggers with appropriate configurations.
"""

from typing import Optional, Union, Dict, Any

from .base_logger import BaseLogger
from .smart_logger import SmartLogger
from .logger_config import LoggerConfig, LogLevel


class LoggerFactory:
    """
    Factory for creating logger instances with appropriate configurations
    """
    
    @staticmethod
    def create_smart_logger(name: str = None, config: LoggerConfig = None) -> SmartLogger:
        """
        Create a SmartLogger instance
        
        Args:
            name: Logger name
            config: Logger configuration (will create default if None)
            
        Returns:
            SmartLogger instance
        """
        if config is None:
            config = LoggerConfig(name=name or "hyperspectral")
        elif name:
            config.name = name
        
        return SmartLogger(config)
    
    
    @staticmethod
    def create_development_logger(name: str = None) -> SmartLogger:
        """
        Create a logger configured for development
        
        Args:
            name: Logger name
            
        Returns:
            SmartLogger instance configured for development
        """
        config = LoggerConfig.for_development(name=name or "dev")
        return SmartLogger(config)
    
    @staticmethod
    def create_production_logger(name: str = None, 
                               log_file_path: str = None) -> SmartLogger:
        """
        Create a logger configured for production
        
        Args:
            name: Logger name
            log_file_path: Path for log file
            
        Returns:
            SmartLogger instance configured for production
        """
        config = LoggerConfig.for_production(
            name=name or "production",
            log_file_path=log_file_path
        )
        return SmartLogger(config)
    
    @staticmethod
    def create_console_only_logger(name: str = None, 
                                 level: LogLevel = LogLevel.INFO) -> SmartLogger:
        """
        Create a logger that only outputs to console
        
        Args:
            name: Logger name
            level: Log level
            
        Returns:
            SmartLogger instance configured for console-only output
        """
        config = LoggerConfig(
            name=name or "console",
            level=level,
            log_to_console=True,
            log_to_file=False,
            enable_duplicate_suppression=True
        )
        return SmartLogger(config)
    
    @staticmethod
    def create_file_only_logger(name: str = None, 
                              log_file_path: str = None,
                              level: LogLevel = LogLevel.INFO) -> SmartLogger:
        """
        Create a logger that only outputs to file
        
        Args:
            name: Logger name
            log_file_path: Path for log file
            level: Log level
            
        Returns:
            SmartLogger instance configured for file-only output
        """
        config = LoggerConfig(
            name=name or "file",
            level=level,
            log_to_console=False,
            log_to_file=True,
            log_file_path=log_file_path,
            enable_duplicate_suppression=True
        )
        return SmartLogger(config)
    
    @staticmethod
    def create_silent_logger(name: str = None) -> SmartLogger:
        """
        Create a logger that suppresses most output (errors only)
        
        Args:
            name: Logger name
            
        Returns:
            SmartLogger instance configured for minimal output
        """
        config = LoggerConfig(
            name=name or "silent",
            level=LogLevel.ERROR,
            log_to_console=True,
            log_to_file=False,
            enable_duplicate_suppression=True,
            include_timestamp=False,
            include_context=False
        )
        return SmartLogger(config)
    
    @staticmethod
    def create_debug_logger(name: str = None, 
                          log_file_path: str = None) -> SmartLogger:
        """
        Create a logger configured for debugging
        
        Args:
            name: Logger name
            log_file_path: Optional path for log file
            
        Returns:
            SmartLogger instance configured for debugging
        """
        config = LoggerConfig(
            name=name or "debug",
            level=LogLevel.DEBUG,
            log_to_console=True,
            log_to_file=bool(log_file_path),
            log_file_path=log_file_path,
            enable_duplicate_suppression=False,  # Show all debug messages
            include_timestamp=True,
            include_level=True,
            include_name=True,
            include_context=True
        )
        return SmartLogger(config)
    
    @staticmethod
    def create_performance_logger(name: str = None, 
                                results_dir: str = None) -> SmartLogger:
        """
        Create a logger configured for performance monitoring
        
        Args:
            name: Logger name
            results_dir: Directory for storing logs
            
        Returns:
            SmartLogger instance configured for performance logging
        """
        config = LoggerConfig(
            name=name or "performance",
            level=LogLevel.PERFORMANCE,
            log_to_console=True,
            log_to_file=True,
            results_dir=results_dir,
            enable_duplicate_suppression=True,
            duplicate_window_seconds=2.0,  # Shorter window for performance logs
            max_duplicate_count=5,
            track_performance_metrics=True
        )
        return SmartLogger(config)
    
    @staticmethod
    def create_custom_logger(name: str, **config_kwargs) -> SmartLogger:
        """
        Create a logger with custom configuration
        
        Args:
            name: Logger name
            **config_kwargs: Configuration parameters
            
        Returns:
            SmartLogger instance with custom configuration
        """
        config = LoggerConfig(name=name, **config_kwargs)
        return SmartLogger(config)
    
    @staticmethod
    def create_logger_from_dict(config_dict: Dict[str, Any]) -> SmartLogger:
        """
        Create a logger from a configuration dictionary
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            SmartLogger instance
        """
        # Convert level string to enum if needed
        if 'level' in config_dict and isinstance(config_dict['level'], str):
            config_dict['level'] = LogLevel[config_dict['level'].upper()]
        
        config = LoggerConfig(**config_dict)
        return SmartLogger(config)
    
    
    @staticmethod
    def get_available_logger_types() -> Dict[str, str]:
        """
        Get information about available logger types
        
        Returns:
            Dictionary mapping logger type names to descriptions
        """
        return {
            'smart': 'General-purpose logger with duplicate suppression',
            'development': 'Logger configured for development work',
            'production': 'Logger configured for production deployment',
            'console_only': 'Logger that outputs only to console',
            'file_only': 'Logger that outputs only to file',
            'silent': 'Logger that shows only errors',
            'debug': 'Logger configured for debugging with verbose output',
            'performance': 'Logger optimized for performance monitoring',
            'custom': 'Logger with custom configuration'
        }
    
    @staticmethod
    def get_recommended_config(use_case: str) -> LoggerConfig:
        """
        Get recommended configuration for common use cases
        
        Args:
            use_case: Use case identifier
            
        Returns:
            Recommended LoggerConfig
            
        Raises:
            ValueError: If use case is not recognized
        """
        configs = {
            'ml_training': LoggerConfig.for_experiment(
                experiment_name="training",
                level=LogLevel.INFO,
                enable_duplicate_suppression=True,
                track_performance_metrics=True
            ),
            'feature_selection': LoggerConfig(
                name="feature_selection",
                level=LogLevel.STAGE,
                enable_duplicate_suppression=True,
                duplicate_window_seconds=1.0,
                track_stage_context=True
            ),
            'model_evaluation': LoggerConfig(
                name="evaluation",
                level=LogLevel.INFO,
                enable_duplicate_suppression=False,
                track_performance_metrics=True
            ),
            'data_processing': LoggerConfig(
                name="data_processing",
                level=LogLevel.INFO,
                enable_duplicate_suppression=True,
                duplicate_window_seconds=10.0,
                max_duplicate_count=2
            ),
            'hyperparameter_tuning': LoggerConfig.for_experiment(
                experiment_name="hyperparameter_tuning",
                level=LogLevel.EXPERIMENT,
                enable_duplicate_suppression=True,
                track_performance_metrics=True
            )
        }
        
        if use_case not in configs:
            available = ', '.join(configs.keys())
            raise ValueError(f"Unknown use case '{use_case}'. Available: {available}")
        
        return configs[use_case] 