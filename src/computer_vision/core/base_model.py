"""
Base class for computer vision models.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Optional, Union
from pathlib import Path


class BaseComputerVisionModel(ABC):
    """Abstract base class for computer vision models."""
    
    def __init__(self, model_name: str):
        """
        Initialize the base model.
        
        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.is_initialized = False
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize the model and load weights."""
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> Any:
        """
        Run inference on an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Model predictions
        """
        pass
    
    @abstractmethod
    def visualize(self, image: np.ndarray, predictions: Any) -> np.ndarray:
        """
        Visualize predictions on the image.
        
        Args:
            image: Original image
            predictions: Model predictions
            
        Returns:
            Image with visualized predictions
        """
        pass
    
    @abstractmethod
    def save_predictions(self, predictions: Any, save_path: Union[str, Path]) -> None:
        """
        Save predictions to file.
        
        Args:
            predictions: Model predictions
            save_path: Path to save the predictions
        """
        pass
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        Validate input image.
        
        Args:
            image: Input image to validate
            
        Returns:
            True if image is valid, False otherwise
        """
        if image is None:
            return False
        
        if not isinstance(image, np.ndarray):
            return False
        
        if len(image.shape) not in [2, 3]:
            return False
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            return False
        
        return True
    
    def ensure_initialized(self) -> None:
        """Ensure model is initialized before use."""
        if not self.is_initialized:
            raise RuntimeError(f"{self.model_name} is not initialized. Call initialize() first.")
