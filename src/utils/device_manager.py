#!/usr/bin/env python3
"""
CPU Device Manager for Parallel Processing

This module provides centralized device management focused on CPU computation
with support for parallel processing using multiprocessing and joblib.

Features:
- CPU-only computation (GPU code removed)
- Parallel processing support via multiprocessing
- Process and thread management
- Memory monitoring and optimization
- Configurable CPU core usage
"""

import os
import sys
import psutil
import multiprocessing as mp
from typing import Optional, Dict, Any, List
from src.utils.logger import get_logger


class DeviceManager:
    """
    CPU-focused device management for parallel hyperspectral processing.
    """
    
    def __init__(self):
        self._device_config = {
            'device_type': 'cpu',
            'cpu_count': mp.cpu_count(),
            'available_memory_gb': self._get_available_memory_gb(),
            'max_workers': None,
            'tensorflow_device': '/CPU:0',
            'pytorch_device': 'cpu'
        }
        self._initialized = False
        self._logger = get_logger("device_manager")
    
    def initialize(self, max_workers: Optional[int] = None, force_single_thread: bool = False) -> Dict[str, Any]:
        """
        Initialize CPU device configuration.
        
        Args:
            max_workers: Maximum number of parallel workers (None for auto-detect)
            force_single_thread: Force single-threaded execution
            
        Returns:
            Dictionary with device configuration information
        """
        if self._initialized:
            return self._device_config.copy()
        
        self._logger.info("Initializing CPU device manager...")
        
        # Configure parallel processing
        self._configure_parallel_processing(max_workers, force_single_thread)
        
        # Set environment variables for CPU optimization
        self._configure_cpu_environment()
        
        self._initialized = True
        
        # Log device configuration
        self._log_device_info()
        
        return self._device_config.copy()
    
    def _get_available_memory_gb(self) -> float:
        """Get available system memory in GB."""
        try:
            memory = psutil.virtual_memory()
            return round(memory.available / (1024**3), 2)
        except Exception as e:
            self._logger.warning(f"Could not determine available memory: {e}")
            return 0.0
    
    def _configure_parallel_processing(self, max_workers: Optional[int], force_single_thread: bool):
        """Configure parallel processing parameters."""
        if force_single_thread:
            self._device_config['max_workers'] = 1
            self._logger.info("Forcing single-threaded execution")
            return
        
        cpu_count = self._device_config['cpu_count']
        available_memory = self._device_config['available_memory_gb']
        
        if max_workers is not None:
            # User specified max workers
            self._device_config['max_workers'] = min(max_workers, cpu_count)
            self._logger.info(f"Using user-specified max workers: {self._device_config['max_workers']}")
        else:
            # Auto-determine based on system resources
            # Conservative approach: use 2/3 of available cores for stability
            recommended_workers = max(1, int(cpu_count * 0.67))
            
            # Memory-based limitation (assume ~2GB per worker for deep learning)
            memory_limited_workers = max(1, int(available_memory / 2.0))
            
            # Use the more conservative limit
            self._device_config['max_workers'] = min(recommended_workers, memory_limited_workers)
            
            self._logger.info(f"Auto-configured workers: {self._device_config['max_workers']} "
                            f"(CPU-limited: {recommended_workers}, Memory-limited: {memory_limited_workers})")
    
    def _configure_cpu_environment(self):
        """Configure environment variables for optimal CPU performance."""
        # Disable GPU for TensorFlow
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        # Optimize TensorFlow for CPU
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TF logging
        
        # Configure OpenMP for optimal CPU usage
        max_threads = self._device_config['max_workers']
        os.environ['OMP_NUM_THREADS'] = str(max_threads)
        os.environ['MKL_NUM_THREADS'] = str(max_threads)
        os.environ['NUMEXPR_NUM_THREADS'] = str(max_threads)
        
        # Configure TensorFlow threading
        os.environ['TF_NUM_INTRAOP_THREADS'] = str(max_threads)
        os.environ['TF_NUM_INTEROP_THREADS'] = str(1)  # Typically 1 for single experiments
        
        self._logger.debug(f"Configured CPU environment with {max_threads} threads")
    
    def _log_device_info(self):
        """Log device configuration information."""
        config = self._device_config
        
        self._logger.info("Device Configuration:")
        self._logger.info(f"   Device Type: {config['device_type'].upper()}")
        self._logger.info(f"   CPU Cores: {config['cpu_count']}")
        self._logger.info(f"   Available Memory: {config['available_memory_gb']} GB")
        self._logger.info(f"   Max Workers: {config['max_workers']}")
        self._logger.info(f"   TensorFlow Device: {config['tensorflow_device']}")
        self._logger.info(f"   PyTorch Device: {config['pytorch_device']}")
    
    def get_tensorflow_device(self) -> str:
        """Get TensorFlow device string."""
        return self._device_config['tensorflow_device']
    
    def get_pytorch_device(self) -> str:
        """Get PyTorch device string."""
        return self._device_config['pytorch_device']
    
    def get_max_workers(self) -> int:
        """Get maximum number of parallel workers."""
        return self._device_config['max_workers']
    
    def get_cpu_count(self) -> int:
        """Get total CPU core count."""
        return self._device_config['cpu_count']
    
    def get_available_memory_gb(self) -> float:
        """Get available memory in GB."""
        return self._get_available_memory_gb()  # Get current value
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get complete device configuration."""
        return self._device_config.copy()
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        try:
            memory = psutil.virtual_memory()
            return {
                'total_gb': round(memory.total / (1024**3), 2),
                'available_gb': round(memory.available / (1024**3), 2),
                'used_gb': round(memory.used / (1024**3), 2),
                'percent_used': memory.percent
            }
        except Exception as e:
            self._logger.warning(f"Could not get memory usage: {e}")
            return {'error': str(e)}
    
    def get_cpu_usage(self) -> Dict[str, float]:
        """Get current CPU usage statistics."""
        try:
            # Get CPU usage over a short interval
            cpu_percent = psutil.cpu_percent(interval=1.0)
            cpu_per_core = psutil.cpu_percent(interval=1.0, percpu=True)
            
            return {
                'overall_percent': cpu_percent,
                'per_core_percent': cpu_per_core,
                'core_count': len(cpu_per_core)
            }
        except Exception as e:
            self._logger.warning(f"Could not get CPU usage: {e}")
            return {'error': str(e)}
    
    def check_system_resources(self, min_memory_gb: float = 2.0, max_cpu_usage: float = 80.0) -> Dict[str, Any]:
        """
        Check if system has sufficient resources for processing.
        
        Args:
            min_memory_gb: Minimum required available memory in GB
            max_cpu_usage: Maximum acceptable CPU usage percentage
            
        Returns:
            Dictionary with resource check results
        """
        memory_info = self.get_memory_usage()
        cpu_info = self.get_cpu_usage()
        
        memory_ok = memory_info.get('available_gb', 0) >= min_memory_gb
        cpu_ok = cpu_info.get('overall_percent', 100) <= max_cpu_usage
        
        result = {
            'memory_ok': memory_ok,
            'cpu_ok': cpu_ok,
            'overall_ok': memory_ok and cpu_ok,
            'memory_info': memory_info,
            'cpu_info': cpu_info,
            'requirements': {
                'min_memory_gb': min_memory_gb,
                'max_cpu_usage': max_cpu_usage
            }
        }
        
        if not result['overall_ok']:
            issues = []
            if not memory_ok:
                issues.append(f"Insufficient memory: {memory_info.get('available_gb', 0):.1f}GB < {min_memory_gb}GB")
            if not cpu_ok:
                issues.append(f"High CPU usage: {cpu_info.get('overall_percent', 0):.1f}% > {max_cpu_usage}%")
            
            self._logger.warning(f"System resource issues: {'; '.join(issues)}")
        
        return result
    
    def optimize_for_parallel_processing(self) -> Dict[str, Any]:
        """
        Optimize system configuration for parallel processing.
        
        Returns:
            Dictionary with optimization results
        """
        optimizations = []
        
        # Check and adjust worker count based on current system load
        cpu_info = self.get_cpu_usage()
        memory_info = self.get_memory_usage()
        
        current_workers = self._device_config['max_workers']
        
        # Reduce workers if CPU usage is high
        if cpu_info.get('overall_percent', 0) > 70:
            new_workers = max(1, current_workers - 1)
            self._device_config['max_workers'] = new_workers
            optimizations.append(f"Reduced workers from {current_workers} to {new_workers} due to high CPU usage")
        
        # Reduce workers if memory is low
        if memory_info.get('available_gb', 0) < 4.0:
            new_workers = max(1, int(current_workers * 0.5))
            self._device_config['max_workers'] = new_workers
            optimizations.append(f"Reduced workers to {new_workers} due to low memory")
        
        # Update environment variables if workers changed
        if self._device_config['max_workers'] != current_workers:
            self._configure_cpu_environment()
        
        result = {
            'optimizations_applied': optimizations,
            'final_workers': self._device_config['max_workers'],
            'system_status': {
                'cpu_usage': cpu_info.get('overall_percent', 0),
                'memory_available': memory_info.get('available_gb', 0)
            }
        }
        
        if optimizations:
            self._logger.info(f"Applied optimizations: {'; '.join(optimizations)}")
        
        return result


# Global device manager instance
_device_manager: Optional[DeviceManager] = None


def get_device_manager() -> DeviceManager:
    """Get the global device manager instance."""
    global _device_manager
    if _device_manager is None:
        _device_manager = DeviceManager()
    return _device_manager


def initialize_devices(max_workers: Optional[int] = None, force_single_thread: bool = False) -> Dict[str, Any]:
    """
    Initialize the global device manager.
    
    Args:
        max_workers: Maximum number of parallel workers
        force_single_thread: Force single-threaded execution
        
    Returns:
        Device configuration dictionary
    """
    manager = get_device_manager()
    return manager.initialize(max_workers=max_workers, force_single_thread=force_single_thread)


def get_tensorflow_device() -> str:
    """Get TensorFlow device string."""
    return get_device_manager().get_tensorflow_device()


def get_pytorch_device() -> str:
    """Get PyTorch device string."""
    return get_device_manager().get_pytorch_device()


def get_max_workers() -> int:
    """Get maximum number of parallel workers."""
    return get_device_manager().get_max_workers()


def get_cpu_count() -> int:
    """Get total CPU core count."""
    return get_device_manager().get_cpu_count()


def check_system_resources(min_memory_gb: float = 2.0, max_cpu_usage: float = 80.0) -> Dict[str, Any]:
    """Check system resources for processing readiness."""
    return get_device_manager().check_system_resources(min_memory_gb, max_cpu_usage) 