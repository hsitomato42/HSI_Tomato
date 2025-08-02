# Models/dl_models/v2_multi_head_cnn_models.py
"""Compatibility shim to preserve legacy import path for MultiHeadCNNModel."""

from .MultiHeadCNNModel import MultiHeadCNNModel  # noqa: F401

__all__: list[str] = ["MultiHeadCNNModel"] 