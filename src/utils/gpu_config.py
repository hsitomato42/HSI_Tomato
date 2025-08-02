"""
GPU Configuration Module

This module provides a unified interface for GPU setup across the project.
It wraps the TensorFlow GPU configuration functionality.
"""

import os
import sys
import logging
from typing import Optional

# Simple GPU configuration functions
import tensorflow as tf

def setup_gpu_environment(gpu_id: Optional[str] = None, 
                         memory_limit_mb: Optional[int] = None) -> bool:
    """
    Setup GPU environment for the application.
    
    This function configures TensorFlow GPU settings to prevent CUDA issues
    and ensures proper GPU initialization for deep learning experiments.
    
    Args:
        gpu_id: GPU ID to use (0, 1, 2, 3, etc.)
        memory_limit_mb: Memory limit in MB (optional)
        
    Returns:
        True if GPU configured successfully, False otherwise
    """
    try:
        # Get list of physical GPUs
        gpus = tf.config.list_physical_devices('GPU')
        
        if not gpus:
            print("⚠️  No GPUs found, running on CPU")
            return False
        
        # Set GPU visibility if specific GPU requested
        if gpu_id is not None:
            gpu_index = int(gpu_id)
            if gpu_index < len(gpus):
                tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
                print(f"✅ Using GPU {gpu_index}: {gpus[gpu_index].name}")
            else:
                print(f"⚠️  GPU {gpu_index} not found, using default GPU")
        
        # Set memory growth to prevent allocation of all GPU memory
        for gpu in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set memory limit if specified
            if memory_limit_mb is not None:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)]
                )
        
        print("✅ GPU environment setup completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up GPU environment: {e}")
        return False

def setup_gpu_for_subprocess(gpu_id: Optional[str] = None) -> bool:
    """
    Setup GPU environment for subprocess execution.
    
    Args:
        gpu_id: GPU ID to use
        
    Returns:
        True if GPU configured successfully, False otherwise
    """
    # Set CUDA_VISIBLE_DEVICES environment variable
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    return setup_gpu_environment(gpu_id=gpu_id)

def get_available_gpus() -> list:
    """
    Get list of available GPU devices.
    
    Returns:
        List of available GPU device names
    """
    try:
        gpus = tf.config.list_physical_devices('GPU')
        return [gpu.name for gpu in gpus]
    except Exception:
        return []

def cleanup_gpu():
    """
    Cleanup GPU resources.
    """
    try:
        # Clear TensorFlow session
        tf.keras.backend.clear_session()
        # Force garbage collection
        import gc
        gc.collect()
    except Exception as e:
        logging.debug(f"GPU cleanup error (non-critical): {e}")
