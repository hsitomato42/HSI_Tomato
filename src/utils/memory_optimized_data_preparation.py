"""
Memory-optimized data preparation for feature selection pipelines.
This module provides utilities to reduce memory usage when working with large hyperspectral data.
"""
import numpy as np
import gc
import tensorflow as tf
from typing import List, Tuple, Optional, Dict, Any
from src.Tomato import Tomato


class MemoryOptimizedDataPreparation:
    """
    Handles memory-efficient data preparation for feature selection.
    
    Key optimizations:
    1. Process data in stages, releasing full spectrum after band selection
    2. Use lazy evaluation where possible
    3. Explicit memory cleanup after each stage
    """
    
    @staticmethod
    def prepare_data_for_feature_selection(
        tomatoes: List[Tomato],
        feature_selector,
        components: Dict[str, bool],
        selected_bands: Optional[List[int]] = None,
        augment_times: int = 0,
        stage: str = 'bands'
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Prepare data with memory optimization for feature selection.
        
        Args:
            tomatoes: List of Tomato objects
            feature_selector: The feature selection interface
            components: Dictionary of components to use
            selected_bands: Pre-selected bands (None for full spectrum in stage 1)
            augment_times: Number of augmentation times
            stage: Current feature selection stage
            
        Returns:
            X: Processed features (reduced dimensionality)
            Y: Target values
            selection_info: Information about selected features
        """
        from utils.data_processing import DataProcessingUtils
        import config
        
        print(f"\n[Memory Optimized] Preparing data for stage: {stage}")
        print(f"[Memory Optimized] Processing {len(tomatoes)} tomatoes")
        
        # Stage 1: Extract only what we need based on the current stage
        if stage == 'bands' and selected_bands is None:
            # Need full spectrum only for band selection stage
            print("[Memory Optimized] Stage 'bands': Loading full spectrum (204 bands)")
            X_full, Y = MemoryOptimizedDataPreparation._extract_full_spectrum(
                tomatoes, augment_times
            )
            
            # Apply feature selection to get reduced data
            print("[Memory Optimized] Applying band selection...")
            X_reduced, selection_info = MemoryOptimizedDataPreparation._apply_band_selection(
                X_full, feature_selector
            )
            
            # Free full spectrum memory immediately
            print(f"[Memory Optimized] Releasing full spectrum data ({X_full.nbytes / 1024 / 1024:.2f} MB)")
            del X_full
            gc.collect()
            
            return X_reduced, Y, selection_info
            
        else:
            # For other stages, we can work with already selected bands
            print(f"[Memory Optimized] Stage '{stage}': Using selected bands only")
            
            # Get the selected band indices from feature selector
            if hasattr(feature_selector, '_preserved_selections'):
                selected_indices = feature_selector._preserved_selections.get('band_indices', list(range(204)))
            else:
                selected_indices = selected_bands if selected_bands is not None else list(range(204))
            
            # Extract only selected bands
            X_selected, Y = MemoryOptimizedDataPreparation._extract_selected_bands(
                tomatoes, selected_indices, augment_times
            )
            
            # Apply feature processing for current stage
            X_processed, selection_info = MemoryOptimizedDataPreparation._apply_stage_processing(
                X_selected, feature_selector, components, stage
            )
            
            return X_processed, Y, selection_info
    
    @staticmethod
    def _extract_full_spectrum(
        tomatoes: List[Tomato],
        augment_times: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract full spectrum data from tomatoes."""
        from utils.data_processing import DataProcessingUtils
        import config
        
        X_features = []
        Y_targets = []
        
        # Create augmented tomatoes
        augmented_tomatoes = DataProcessingUtils.create_augmented_and_padded_tomatoes(
            tomatoes=tomatoes,
            augment_times=augment_times
        )
        
        # Free original tomatoes reference
        del tomatoes
        gc.collect()
        
        # Extract only reflectance data (full spectrum)
        for tomato in augmented_tomatoes:
            reflectance_matrix = tomato.spectral_stats.reflectance_matrix
            X_features.append(reflectance_matrix)
            
            # Build target vector
            quality_assess = tomato.quality_assess
            y_values = [getattr(quality_assess, attr, np.nan) 
                       for attr in config.PREDICTED_QUALITY_ATTRIBUTES]
            Y_targets.append(y_values)
        
        # Free augmented tomatoes
        del augmented_tomatoes
        gc.collect()
        
        X = np.array(X_features)
        Y = np.array(Y_targets, dtype=float)
        
        # Fill missing values
        Y = DataProcessingUtils.fill_missing_values(Y, config.PREDICTED_QUALITY_ATTRIBUTES)
        
        return X, Y
    
    @staticmethod
    def _extract_selected_bands(
        tomatoes: List[Tomato],
        selected_indices: List[int],
        augment_times: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract only selected bands from tomatoes."""
        from utils.data_processing import DataProcessingUtils
        import config
        
        X_features = []
        Y_targets = []
        
        # Create augmented tomatoes
        augmented_tomatoes = DataProcessingUtils.create_augmented_and_padded_tomatoes(
            tomatoes=tomatoes,
            augment_times=augment_times
        )
        
        # Free original tomatoes reference
        del tomatoes
        gc.collect()
        
        # Extract only selected bands
        for tomato in augmented_tomatoes:
            reflectance_matrix = tomato.spectral_stats.reflectance_matrix
            # Extract only selected bands
            selected_reflectance = reflectance_matrix[:, :, selected_indices]
            X_features.append(selected_reflectance)
            
            # Build target vector
            quality_assess = tomato.quality_assess
            y_values = [getattr(quality_assess, attr, np.nan) 
                       for attr in config.PREDICTED_QUALITY_ATTRIBUTES]
            Y_targets.append(y_values)
        
        # Free augmented tomatoes
        del augmented_tomatoes
        gc.collect()
        
        X = np.array(X_features)
        Y = np.array(Y_targets, dtype=float)
        
        # Fill missing values
        Y = DataProcessingUtils.fill_missing_values(Y, config.PREDICTED_QUALITY_ATTRIBUTES)
        
        print(f"[Memory Optimized] Extracted {len(selected_indices)} bands: {X.shape}")
        
        return X, Y
    
    @staticmethod
    def _apply_band_selection(
        X_full: np.ndarray,
        feature_selector
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply band selection and return reduced data."""
        # Convert to TensorFlow tensor
        X_tensor = tf.constant(X_full, dtype=tf.float32)
        
        # Apply feature selection (this should select bands)
        with tf.device('/CPU:0'):  # Force CPU to save GPU memory
            X_reduced, selection_info = feature_selector.process_hyperspectral_data(
                X_tensor, training=False, epoch=None
            )
        
        # Convert back to numpy and ensure we have the reduced data
        X_reduced_np = X_reduced.numpy()
        
        # Store selected band indices for future use
        if hasattr(feature_selector, 'feature_selector') and hasattr(feature_selector.feature_selector, 'selected_indices'):
            selected_indices = feature_selector.feature_selector.selected_indices
            selection_info['band_indices'] = selected_indices
            print(f"[Memory Optimized] Selected bands: {selected_indices}")
        
        return X_reduced_np, selection_info
    
    @staticmethod
    def _apply_stage_processing(
        X_selected: np.ndarray,
        feature_selector,
        components: Dict[str, bool],
        stage: str
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply stage-specific processing."""
        # For stages after band selection, we need to compute additional features
        # based on the already selected bands
        
        # Convert to TensorFlow tensor
        X_tensor = tf.constant(X_selected, dtype=tf.float32)
        
        # Apply feature processing for current stage
        with tf.device('/CPU:0'):  # Force CPU to save GPU memory
            X_processed, selection_info = feature_selector.process_hyperspectral_data(
                X_tensor, training=False, epoch=None
            )
        
        # Convert back to numpy
        X_processed_np = X_processed.numpy()
        
        return X_processed_np, selection_info
    
    @staticmethod
    def prepare_data_in_batches(
        tomatoes: List[Tomato],
        selected_bands: Optional[List[int]],
        batch_size: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data in batches to reduce peak memory usage.
        
        Args:
            tomatoes: List of Tomato objects
            selected_bands: Band indices to extract
            batch_size: Number of tomatoes to process at once
            
        Returns:
            X: Feature matrix
            Y: Target matrix
        """
        from utils.data_processing import DataProcessingUtils
        import config
        
        X_batches = []
        Y_batches = []
        
        # Process in batches
        for i in range(0, len(tomatoes), batch_size):
            batch_tomatoes = tomatoes[i:i + batch_size]
            
            # Extract data for this batch
            X_batch, Y_batch = MemoryOptimizedDataPreparation._extract_selected_bands(
                batch_tomatoes, selected_bands or list(range(204)), augment_times=0
            )
            
            X_batches.append(X_batch)
            Y_batches.append(Y_batch)
            
            # Clean up batch
            del batch_tomatoes
            gc.collect()
            
            print(f"[Memory Optimized] Processed batch {i//batch_size + 1}/{(len(tomatoes) + batch_size - 1)//batch_size}")
        
        # Concatenate all batches
        X = np.concatenate(X_batches, axis=0)
        Y = np.concatenate(Y_batches, axis=0)
        
        # Clean up batches
        del X_batches, Y_batches
        gc.collect()
        
        return X, Y
