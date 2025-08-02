"""
Central configuration for log rotation across the project.
All log files will be limited to 5MB with 5 backup files.
"""

# Log rotation configuration
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 5  # Keep 5 backup files

def get_rotating_file_handler(filename, formatter=None):
    """
    Create a RotatingFileHandler with standard configuration.
    
    Args:
        filename: Path to the log file
        formatter: Optional logging formatter
        
    Returns:
        RotatingFileHandler instance
    """
    from logging.handlers import RotatingFileHandler
    
    handler = RotatingFileHandler(
        filename,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    
    if formatter:
        handler.setFormatter(formatter)
    
    return handler
