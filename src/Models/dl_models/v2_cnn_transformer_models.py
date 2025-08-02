# Models/dl_models/v2_cnn_transformer_models.py
"""Compatibility shim to preserve legacy import path for CNNTransformerModel."""

from .CNNTransformerModel import CNNTransformerModel  # noqa: F401

__all__: list[str] = ["CNNTransformerModel"] 