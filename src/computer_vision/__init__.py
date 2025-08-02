"""
Computer Vision module for tomato instance segmentation.

This module provides tools and models for computer vision tasks,
specifically focused on tomato instance segmentation using various
state-of-the-art models like Grounding DINO and SAM.

Package Structure:
- core/: Base classes and core functionality
- models/: Model implementations  
- utils/: Utility functions and helpers
- examples/: Example usage and demos
- tests/: Test files
- data/: Example data and outputs
"""

# Import core classes (these should not have external dependencies)
from .core.base_model import BaseComputerVisionModel

# Try to import model implementations (may fail if dependencies not installed)
_available_models = []

try:
    from .models.grounding_dino_sam_model import GroundingDinoSamModel
    _available_models.append("GroundingDinoSamModel")
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import GroundingDinoSamModel: {e}")

# InstanceSegModel removed as it was a duplicate mock of GroundingDinoSamModel

# Try to import utilities (may fail if dependencies not installed)
try:
    from .utils.segmentation_helpers import *
    from .utils.annotation_converter import *
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import utilities: {e}")

__all__ = [
    "BaseComputerVisionModel",
    # Dynamic imports based on what's available
] + _available_models

__version__ = "1.0.0" 