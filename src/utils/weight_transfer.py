"""
Weight transfer utilities for progressive neural network training.

This module provides functionality to transfer weights between stages in progressive
feature selection training, allowing for efficient knowledge transfer while handling
different input channel dimensions.
"""

import tensorflow as tf
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
import logging

logger = logging.getLogger(__name__)


class WeightTransferManager:
    """Manages weight transfer between progressive training stages."""
    
    def __init__(self, enable_weight_transfer: bool = True):
        """
        Initialize the weight transfer manager.
        
        Args:
            enable_weight_transfer: Whether to enable weight transfer between stages
        """
        self.enable_weight_transfer = enable_weight_transfer
        self.transfer_stats = {}
        
    def supports_layer_wise_copying(self) -> bool:
        """
        Check if the framework supports layer-wise weight copying.
        
        Returns:
            True if TensorFlow supports layer-wise weight copying
        """
        return hasattr(tf.keras.layers.Layer, 'get_weights') and hasattr(tf.keras.layers.Layer, 'set_weights')
    
    def transfer_weights(
        self, 
        source_model: tf.keras.Model, 
        target_model: tf.keras.Model,
        stage_channels: int,
        stage_name: str
    ) -> bool:
        """
        Transfer compatible weights from source model to target model.
        
        If the framework supports layer-wise weight copying, slice the existing weights
        to the first `stage_channels` before assigning them to the new model; otherwise,
        rebuild from scratch for each stage.
        
        Args:
            source_model: Model to transfer weights from
            target_model: Model to transfer weights to  
            stage_channels: Number of input channels for the target stage
            stage_name: Name of the current stage
            
        Returns:
            True if weights were successfully transferred, False otherwise
        """
        if not self.enable_weight_transfer:
            logger.info(f"Weight transfer disabled for stage '{stage_name}'")
            return False
            
        if not self.supports_layer_wise_copying():
            logger.warning(f"Framework does not support layer-wise weight copying for stage '{stage_name}'")
            return False
            
        if source_model is None or target_model is None:
            logger.warning(f"Source or target model is None for stage '{stage_name}'")
            return False
            
        try:
            transfer_count = 0
            skipped_count = 0
            
            # Get source and target layers
            source_layers = {layer.name: layer for layer in source_model.layers}
            target_layers = {layer.name: layer for layer in target_model.layers}
            
            logger.info(f"Starting weight transfer for stage '{stage_name}' with {stage_channels} channels")
            logger.info(f"Source model layers: {len(source_layers)}")
            logger.info(f"Target model layers: {len(target_layers)}")
            
            # Transfer weights layer by layer
            for layer_name, target_layer in target_layers.items():
                if layer_name in source_layers:
                    source_layer = source_layers[layer_name]
                    
                    if self._transfer_layer_weights(source_layer, target_layer, stage_channels, stage_name):
                        transfer_count += 1
                    else:
                        skipped_count += 1
                else:
                    skipped_count += 1
                    logger.debug(f"Layer '{layer_name}' not found in source model")
            
            # Store transfer statistics
            self.transfer_stats[stage_name] = {
                'transferred_layers': transfer_count,
                'skipped_layers': skipped_count,
                'total_target_layers': len(target_layers),
                'stage_channels': stage_channels
            }
            
            logger.info(f"Weight transfer completed for stage '{stage_name}': "
                       f"{transfer_count} layers transferred, {skipped_count} layers skipped")
            
            return transfer_count > 0
            
        except Exception as e:
            logger.error(f"Error during weight transfer for stage '{stage_name}': {e}")
            return False
    
    def _transfer_layer_weights(
        self, 
        source_layer: tf.keras.layers.Layer,
        target_layer: tf.keras.layers.Layer,
        stage_channels: int,
        stage_name: str
    ) -> bool:
        """
        Transfer weights from source layer to target layer, handling channel dimension changes.
        
        Args:
            source_layer: Source layer to transfer weights from
            target_layer: Target layer to transfer weights to
            stage_channels: Number of input channels for the target stage
            stage_name: Name of the current stage
            
        Returns:
            True if weights were successfully transferred, False otherwise
        """
        try:
            # Get layer weights
            source_weights = source_layer.get_weights()
            target_weights = target_layer.get_weights()
            
            # Skip if no weights to transfer
            if not source_weights or not target_weights:
                logger.debug(f"No weights to transfer for layer '{source_layer.name}'")
                return False
                
            # Check if layer shapes are compatible
            if len(source_weights) != len(target_weights):
                logger.debug(f"Weight count mismatch for layer '{source_layer.name}': "
                           f"source={len(source_weights)}, target={len(target_weights)}")
                return False
            
            # Transfer weights with channel slicing if needed
            transferred_weights = []
            for i, (src_weight, tgt_weight) in enumerate(zip(source_weights, target_weights)):
                transferred_weight = self._slice_weight_for_channels(
                    src_weight, tgt_weight, stage_channels, source_layer.name, i
                )
                
                if transferred_weight is not None:
                    transferred_weights.append(transferred_weight)
                else:
                    logger.debug(f"Could not transfer weight {i} for layer '{source_layer.name}'")
                    return False
            
            # Set the transferred weights
            target_layer.set_weights(transferred_weights)
            logger.debug(f"Successfully transferred weights for layer '{source_layer.name}'")
            return True
            
        except Exception as e:
            logger.debug(f"Error transferring weights for layer '{source_layer.name}': {e}")
            return False
    
    def _slice_weight_for_channels(
        self, 
        source_weight: np.ndarray,
        target_weight: np.ndarray,
        stage_channels: int,
        layer_name: str,
        weight_index: int
    ) -> Optional[np.ndarray]:
        """
        Slice source weight to match target weight shape, handling channel dimension changes.
        
        Args:
            source_weight: Source weight array
            target_weight: Target weight array (for shape reference)
            stage_channels: Number of input channels for the target stage
            layer_name: Name of the layer (for logging)
            weight_index: Index of the weight within the layer
            
        Returns:
            Sliced weight array or None if incompatible
        """
        try:
            src_shape = source_weight.shape
            tgt_shape = target_weight.shape
            
            # If shapes are identical, no slicing needed
            if src_shape == tgt_shape:
                return source_weight.copy()
            
            # Handle different layer types
            if len(src_shape) == 4 and len(tgt_shape) == 4:
                # Conv2D layer weights: (height, width, in_channels, out_channels)
                return self._slice_conv2d_weights(source_weight, target_weight, stage_channels, layer_name)
                
            elif len(src_shape) == 2 and len(tgt_shape) == 2:
                # Dense layer weights: (in_features, out_features)
                return self._slice_dense_weights(source_weight, target_weight, layer_name)
                
            elif len(src_shape) == 1 and len(tgt_shape) == 1:
                # Bias weights: (out_features,)
                return self._slice_bias_weights(source_weight, target_weight, layer_name)
                
            else:
                logger.debug(f"Unsupported weight shape combination for layer '{layer_name}': "
                           f"source={src_shape}, target={tgt_shape}")
                return None
                
        except Exception as e:
            logger.debug(f"Error slicing weight for layer '{layer_name}': {e}")
            return None
    
    def _slice_conv2d_weights(
        self,
        source_weight: np.ndarray,
        target_weight: np.ndarray,
        stage_channels: int,
        layer_name: str
    ) -> Optional[np.ndarray]:
        """
        Slice Conv2D weights to match target input channels.
        
        Args:
            source_weight: Source conv2d weight (H, W, in_channels, out_channels)
            target_weight: Target conv2d weight (H, W, stage_channels, out_channels)
            stage_channels: Number of input channels for the target stage
            layer_name: Name of the layer
            
        Returns:
            Sliced weight array or None if incompatible
        """
        src_shape = source_weight.shape  # (H, W, src_in_channels, src_out_channels)
        tgt_shape = target_weight.shape  # (H, W, tgt_in_channels, tgt_out_channels)
        
        # Check if spatial dimensions match
        if src_shape[0] != tgt_shape[0] or src_shape[1] != tgt_shape[1]:
            logger.debug(f"Spatial dimensions mismatch for Conv2D layer '{layer_name}': "
                       f"source=({src_shape[0]},{src_shape[1]}), target=({tgt_shape[0]},{tgt_shape[1]})")
            return None
        
        # Check if output channels match
        if src_shape[3] != tgt_shape[3]:
            logger.debug(f"Output channels mismatch for Conv2D layer '{layer_name}': "
                       f"source={src_shape[3]}, target={tgt_shape[3]}")
            return None
        
        # Slice input channels if source has more channels than target
        if src_shape[2] >= tgt_shape[2]:
            # Take the first stage_channels from source
            sliced_weight = source_weight[:, :, :tgt_shape[2], :]
            logger.debug(f"Sliced Conv2D weights for layer '{layer_name}': "
                       f"{src_shape} -> {sliced_weight.shape}")
            return sliced_weight
        else:
            logger.debug(f"Source has fewer channels than target for Conv2D layer '{layer_name}': "
                       f"source={src_shape[2]}, target={tgt_shape[2]}")
            return None
    
    def _slice_dense_weights(
        self,
        source_weight: np.ndarray,
        target_weight: np.ndarray,
        layer_name: str
    ) -> Optional[np.ndarray]:
        """
        Slice Dense layer weights to match target shape.
        
        Args:
            source_weight: Source dense weight (in_features, out_features)
            target_weight: Target dense weight (in_features, out_features)
            layer_name: Name of the layer
            
        Returns:
            Sliced weight array or None if incompatible
        """
        src_shape = source_weight.shape
        tgt_shape = target_weight.shape
        
        # For Dense layers, we typically want to match output dimensions
        # Input dimensions might differ due to different feature extraction
        if src_shape[1] != tgt_shape[1]:
            logger.debug(f"Output dimensions mismatch for Dense layer '{layer_name}': "
                       f"source={src_shape[1]}, target={tgt_shape[1]}")
            return None
        
        # If input dimensions are compatible, use the weights directly
        if src_shape[0] == tgt_shape[0]:
            return source_weight.copy()
        else:
            logger.debug(f"Input dimensions mismatch for Dense layer '{layer_name}': "
                       f"source={src_shape[0]}, target={tgt_shape[0]}")
            return None
    
    def _slice_bias_weights(
        self,
        source_weight: np.ndarray,
        target_weight: np.ndarray,
        layer_name: str
    ) -> Optional[np.ndarray]:
        """
        Slice bias weights to match target shape.
        
        Args:
            source_weight: Source bias weight (out_features,)
            target_weight: Target bias weight (out_features,)
            layer_name: Name of the layer
            
        Returns:
            Sliced weight array or None if incompatible
        """
        src_shape = source_weight.shape
        tgt_shape = target_weight.shape
        
        # Bias should have same output dimensions
        if src_shape[0] == tgt_shape[0]:
            return source_weight.copy()
        else:
            logger.debug(f"Bias dimensions mismatch for layer '{layer_name}': "
                       f"source={src_shape[0]}, target={tgt_shape[0]}")
            return None
    
    def get_transfer_stats(self, stage_name: str) -> Dict[str, Any]:
        """
        Get transfer statistics for a specific stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Dictionary containing transfer statistics
        """
        return self.transfer_stats.get(stage_name, {})
    
    def get_all_transfer_stats(self) -> Dict[str, Dict[str, Any]]:
        """
        Get transfer statistics for all stages.
        
        Returns:
            Dictionary containing transfer statistics for all stages
        """
        return self.transfer_stats.copy()
    
    def clear_transfer_stats(self):
        """Clear all transfer statistics."""
        self.transfer_stats.clear()
