# feature_selection/__init__.py

from .abstract_feature_selector import AbstractFeatureSelector
from .implementations.hdfs.hierarchical_differentiable_feature_selection import HierarchicalDifferentiableFeatureSelection
from .implementations.hdfs.utils.visualizations import FeatureSelectionVisualizer

# Direct alias for backward compatibility - no wrapper needed since HDFS already has the methods
FeatureSelectionInterface = HierarchicalDifferentiableFeatureSelection
AttentionBasedFeatureSelector = HierarchicalDifferentiableFeatureSelection

__all__ = [
    'AbstractFeatureSelector',
    'HierarchicalDifferentiableFeatureSelection',
    'FeatureSelectionInterface',  # Direct alias - no wrapper
    'AttentionBasedFeatureSelector',  # Direct alias - no wrapper
    'FeatureSelectionVisualizer'
]
