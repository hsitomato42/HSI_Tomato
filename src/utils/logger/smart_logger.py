"""
Smart Logger Implementation

Main logger implementation with duplicate suppression, intelligent formatting,
and comprehensive CPU-optimized logging features.
"""

import time
import threading
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, TextIO
from collections import deque, defaultdict
from pathlib import Path

from .base_logger import BaseLogger
from .logger_config import LoggerConfig, LogLevel, MessageType


class MessageRecord:
    """Record for tracking message details and duplicates"""
    
    def __init__(self, level: LogLevel, message: str, message_type: MessageType, 
                 timestamp: float, context: str, **kwargs):
        self.level = level
        self.message = message
        self.message_type = message_type
        self.timestamp = timestamp
        self.context = context
        self.kwargs = kwargs
        self.count = 1
        self.last_seen = timestamp
        
        # Create hash for duplicate detection
        self.hash = self._create_hash()
    
    def _create_hash(self) -> str:
        """Create hash for duplicate detection"""
        # Include level, message, type, and context for hash
        hash_content = f"{self.level.name}:{self.message}:{self.message_type.value}:{self.context}"
        return hashlib.md5(hash_content.encode()).hexdigest()
    
    def update_duplicate(self, timestamp: float):
        """Update duplicate count and last seen timestamp"""
        self.count += 1
        self.last_seen = timestamp


class SmartLogger(BaseLogger):
    """
    Smart logger implementation with duplicate suppression and intelligent formatting
    """
    
    def __init__(self, config: LoggerConfig):
        """
        Initialize the smart logger
        
        Args:
            config: Logger configuration
        """
        super().__init__(config)
        
        # File handling
        self._file_handle: Optional[TextIO] = None
        
        # Message tracking for duplicate suppression
        self._message_history: Dict[str, MessageRecord] = {}
        self._message_buffer: deque = deque(maxlen=config.buffer_size)
        
        # Threading for async operations
        self._lock = threading.RLock()
        self._flush_timer: Optional[threading.Timer] = None
        
        # Statistics
        self._stats = {
            'messages_logged': 0,
            'messages_suppressed': 0,
            'duplicates_found': 0,
            'start_time': time.time()
        }
    
    def _initialize(self) -> None:
        """Initialize the logger implementation"""
        if self.config.log_to_file and self.config.log_file_path:
            try:
                # Ensure directory exists
                log_path = Path(self.config.log_file_path)
                log_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Open file for append
                self._file_handle = open(self.config.log_file_path, 'a', encoding='utf-8')
                
                # Write session header
                self._write_session_header()
                
            except Exception as e:
                # Fallback to console-only logging
                self.config.log_to_file = False
                if self.config.log_to_console:
                    print(f"Warning: Could not open log file {self.config.log_file_path}: {e}")
        
        # Start flush timer if configured
        if self.config.flush_interval > 0:
            self._schedule_flush()
    
    def _write_session_header(self) -> None:
        """Write session header to log file"""
        if not self._file_handle:
            return
            
        header = [
            "=" * 80,
            f"New logging session started: {datetime.now().isoformat()}",
            f"Logger: {self.config.name}",
            f"Level: {self.config.level.name}",
            f"Duplicate suppression: {'enabled' if self.config.enable_duplicate_suppression else 'disabled'}",
            "=" * 80,
            ""
        ]
        
        for line in header:
            self._file_handle.write(line + "\n")
        self._file_handle.flush()
    
    def _should_log(self, level: LogLevel, message: str, message_type: MessageType) -> bool:
        """
        Determine if a message should be logged based on level and duplicate suppression
        
        Args:
            level: Log level
            message: Message to check
            message_type: Type of message
            
        Returns:
            True if message should be logged
        """
        # Check log level
        if not self.is_enabled_for(level):
            return False
        
        # Check duplicate suppression
        if self.config.enable_duplicate_suppression:
            return self._check_duplicate_suppression(level, message, message_type)
        
        return True
    
    def _check_duplicate_suppression(self, level: LogLevel, message: str, message_type: MessageType) -> bool:
        """
        Check if message should be suppressed due to duplication
        
        Args:
            level: Log level
            message: Message to check
            message_type: Type of message
            
        Returns:
            True if message should be logged (not suppressed)
        """
        with self._lock:
            current_time = time.time()
            context = self.get_context_string()
            
            # Create temporary record for hash generation
            temp_record = MessageRecord(level, message, message_type, current_time, context)
            message_hash = temp_record.hash
            
            # Check if we've seen this message recently
            if message_hash in self._message_history:
                record = self._message_history[message_hash]
                
                # Check if within suppression window
                if current_time - record.timestamp <= self.config.duplicate_window_seconds:
                    record.update_duplicate(current_time)
                    self._stats['duplicates_found'] += 1
                    
                    # Suppress if we've exceeded max count
                    if record.count > self.config.max_duplicate_count:
                        self._stats['messages_suppressed'] += 1
                        return False
                    
                    # Log with duplicate count
                    return True
                else:
                    # Outside window, treat as new message
                    del self._message_history[message_hash]
            
            # New message, add to history
            self._message_history[message_hash] = temp_record
            
            # Clean old messages from history
            self._clean_message_history(current_time)
            
            return True
    
    def _clean_message_history(self, current_time: float) -> None:
        """Clean old messages from history to prevent memory bloat"""
        cutoff_time = current_time - (self.config.duplicate_window_seconds * 2)
        
        # Remove old entries
        to_remove = [
            hash_key for hash_key, record in self._message_history.items()
            if record.last_seen < cutoff_time
        ]
        
        for hash_key in to_remove:
            del self._message_history[hash_key]
    
    def _log(self, level: LogLevel, message: str, message_type: MessageType = MessageType.GENERAL, **kwargs) -> None:
        """
        Internal logging method implementation
        
        Args:
            level: Log level
            message: Message to log
            message_type: Type of message for intelligent routing
            **kwargs: Additional context data
        """
        with self._lock:
            current_time = time.time()
            context = self.get_context_string()
            
            # Create record
            record = MessageRecord(level, message, message_type, current_time, context, **kwargs)
            
            # Check for duplicate count info
            if self.config.enable_duplicate_suppression and record.hash in self._message_history:
                existing_record = self._message_history[record.hash]
                if existing_record.count > 1:
                    record.count = existing_record.count
            
            # Format message
            formatted_message = self._format_message(record)
            
            # Buffer the message
            self._message_buffer.append(formatted_message)
            
            # Output immediately if not buffering or high priority
            if (self.config.buffer_size <= 1 or 
                level.value >= LogLevel.ERROR.value or 
                message_type in [MessageType.ERROR, MessageType.WARNING]):
                self._flush_buffer()
            
            self._stats['messages_logged'] += 1
    
    def _format_message(self, record: MessageRecord) -> str:
        """
        Format message for output
        
        Args:
            record: Message record to format
            
        Returns:
            Formatted message string
        """
        parts = []
        
        # Timestamp
        if self.config.include_timestamp:
            timestamp = datetime.fromtimestamp(record.timestamp).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            parts.append(f"[{timestamp}]")
        
        # Level
        if self.config.include_level:
            level_str = record.level.name
            # Add emoji for visual clarity
            level_emoji = {
                'DEBUG': '🔍',
                'INFO': 'ℹ️',
                'WARNING': '⚠️',
                'ERROR': '❌',
                'CRITICAL': '🚨',
                'EXPERIMENT': '🧪',
                'PERFORMANCE': '⚡',
                'STAGE': '📶'
            }
            emoji = level_emoji.get(level_str, '')
            parts.append(f"[{emoji}{level_str}]")
        
        # Logger name
        if self.config.include_name:
            parts.append(f"[{self.config.name}]")
        
        # Context
        if self.config.include_context and record.context:
            parts.append(record.context)
        
        # Duplicate count
        if record.count > 1:
            parts.append(f"[x{record.count}]")
        
        # Message
        message_part = record.message
        
        # Add kwargs if present
        if record.kwargs:
            kwargs_str = " | ".join([f"{k}={v}" for k, v in record.kwargs.items()])
            message_part += f" | {kwargs_str}"
        
        # Combine all parts
        prefix = " ".join(parts)
        if prefix:
            return f"{prefix} {message_part}"
        else:
            return message_part
    
    def _flush_buffer(self) -> None:
        """Flush buffered messages to outputs"""
        if not self._message_buffer:
            return
        
        messages_to_flush = list(self._message_buffer)
        self._message_buffer.clear()
        
        for message in messages_to_flush:
            # Console output
            if self.config.log_to_console:
                print(message)
            
            # File output
            if self.config.log_to_file and self._file_handle:
                try:
                    self._file_handle.write(message + "\n")
                    self._file_handle.flush()
                except Exception as e:
                    # Fallback to console if file write fails
                    if self.config.log_to_console:
                        print(f"Warning: Could not write to log file: {e}")
                        print(message)
    
    def _schedule_flush(self) -> None:
        """Schedule automatic buffer flush"""
        if self._flush_timer:
            self._flush_timer.cancel()
        
        self._flush_timer = threading.Timer(self.config.flush_interval, self._auto_flush)
        self._flush_timer.daemon = True
        self._flush_timer.start()
    
    def _auto_flush(self) -> None:
        """Automatic flush callback"""
        self.flush()
        if self.config.flush_interval > 0:
            self._schedule_flush()
    
    def flush(self) -> None:
        """Flush any buffered messages"""
        with self._lock:
            self._flush_buffer()
    
    def close(self) -> None:
        """Close the logger and release resources"""
        with self._lock:
            # Cancel flush timer
            if self._flush_timer:
                self._flush_timer.cancel()
                self._flush_timer = None
            
            # Flush any remaining messages
            self.flush()
            
            # Write session footer
            if self._file_handle:
                self._write_session_footer()
                self._file_handle.close()
                self._file_handle = None
            
            # Clear message history
            self._message_history.clear()
            self._message_buffer.clear()
    
    def _write_session_footer(self) -> None:
        """Write session footer to log file"""
        if not self._file_handle:
            return
        
        session_duration = time.time() - self._stats['start_time']
        
        footer = [
            "",
            "=" * 80,
            f"Logging session ended: {datetime.now().isoformat()}",
            f"Session duration: {session_duration:.2f} seconds",
            f"Messages logged: {self._stats['messages_logged']}",
            f"Messages suppressed: {self._stats['messages_suppressed']}",
            f"Duplicates found: {self._stats['duplicates_found']}",
            "=" * 80
        ]
        
        for line in footer:
            self._file_handle.write(line + "\n")
        self._file_handle.flush()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics"""
        with self._lock:
            stats = self._stats.copy()
            stats['session_duration'] = time.time() - stats['start_time']
            stats['message_history_size'] = len(self._message_history)
            stats['buffer_size'] = len(self._message_buffer)
            return stats
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.close()
        except:
            pass  # Ignore errors during cleanup 