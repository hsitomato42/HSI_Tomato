# Models/dl_models/v2_cnn_models.py
"""Compatibility shim for legacy import paths.

This module re-exports CNNModel from Models.dl_models.CNNModel to maintain
backwards–compatibility with code that expects the `Models.dl_models.v2_cnn_models`
module path.
"""

from .CNNModel import CNNModel  # noqa: F401

__all__: list[str] = ["CNNModel"] 