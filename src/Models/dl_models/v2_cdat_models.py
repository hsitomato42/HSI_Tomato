# Models/dl_models/v2_cdat_models.py
"""Compatibility shim for legacy import path of CDATModel.

Re-exports CDATProgressive as CDATModel so that existing code expecting
Models.dl_models.v2_cdat_models.CDATModel continues to work.
"""

from .CDATProgressive import CDATProgressive as CDATModel  # noqa: F401

__all__: list[str] = ["CDATModel"] 