# Models/base_classes/BaseDLModel.py

from src.Tomato import Tomato
from .BaseModel import BaseModel
from typing import Tuple, Optional, List, Dict, Union, Any
import numpy as np
import os
import tensorflow as tf
from keras._tf_keras.keras.models import load_model as keras_load_model
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from src.utils.data_processing import DataProcessingUtils
from src.utils.spectral_indexes import SpectralIndexCalculator
from src.utils.target_scaling import TargetScaler
from src.utils.model_config_manager import ModelConfigManager
from src.utils.feature_importance import FeatureImportanceAnalyzer
import src.config as config
from src.config.enums import ModelType, SpectralIndex
from abc import abstractmethod
import sys
import json
import uuid
import shutil
import tempfile
import gc
from pathlib import Path
from src.utils.weight_transfer import WeightTransferManager


class BaseDLModel(BaseModel):
    """
    Base class for all deep learning models that provides common functionality.
    Handles multi-regression vs single prediction modes, data preparation, training, and testing.
    """

    def __init__(
        self,
        model_type: ModelType,
        model_name: str,
        model_filename: str,
        model_shape: tuple[int, int, int],
        components: dict,
        component_dimensions: dict,
        selected_bands: List[int],
        predicted_attributes: list[str] = config.PREDICTED_QUALITY_ATTRIBUTES,
        selected_indexes: Optional[List[SpectralIndex]] = None
    ):
        """
        Initialize BaseDLModel with support for both single and multi-regression modes.
        """
        super().__init__(model_type=model_type, model_name=model_name,
                         model_filename=model_filename, selected_indexes=selected_indexes)
        
        self.model_shape = model_shape
        self.predicted_attributes = predicted_attributes
        self.selected_bands = selected_bands
        self.components = components
        self.component_dimensions = component_dimensions
        self.target_scaler = None  # For multi-regression mode
        self.training_time_minutes = 0.0  # Track training time
        
        # 🎯 FEATURE SELECTION: Initialize the redesigned interface
        self.fs_interface = None
        if getattr(config, 'USE_FEATURE_SELECTION', False):
            from src.feature_selection import FeatureSelectionInterface
            
            # Store original shape for feature selection
            self.original_model_shape = model_shape
            
            # Initialize the redesigned feature selection interface
            self.fs_interface = FeatureSelectionInterface(
                original_shape=model_shape,
                components=components,
                selected_bands=selected_bands,
                selected_indexes=selected_indexes,
                enable_feature_selection=True
            )
            
            # Update model shape to match feature selection output
            fs_dimensions = self.fs_interface.get_component_dimensions()
            new_channels = fs_dimensions['total_channels']
            self.model_shape = (model_shape[0], model_shape[1], new_channels)
            
            print(f"[{self.__class__.__name__}] 🎯 FEATURE SELECTION ENABLED:")
            print(f"[{self.__class__.__name__}]   Original shape: {self.original_model_shape}")
            print(f"[{self.__class__.__name__}]   New shape: {self.model_shape}")
            print(f"[{self.__class__.__name__}]   Expected channels: {new_channels}")
        
        # Regenerate hash and configuration now that all parameters are set
        self._generate_model_hash_and_config()
        
        # Initialize checkpoint manager
        experiment_id = getattr(config, 'EXPERIMENT_ID', None) or f"{model_name}_{uuid.uuid4().hex[:8]}"
        self.checkpoint_manager = ExperimentCheckpointManager(experiment_id)
        
        # Clean up memory after complete model initialization
        gc.collect()
        
        
        # 🎯 FEATURE SELECTION: Track actual vs configured selections
        # These will be populated after feature selection is applied
        self.actual_selected_bands = None  # Actual bands selected by feature selection
        self.actual_ndsi_pairs = None      # Actual NDSI pairs computed by feature selection  
        self.actual_index_names = None     # Actual indexes computed by feature selection
        self.actual_std_bands = None       # Actual STD bands computed by feature selection
        self.fs_selection_applied = False  # Flag to track if feature selection was applied
        
        # Initialize based on regression mode
        if config.MULTI_REGRESSION:
            print(f"[{self.__class__.__name__}] Initializing in MULTI-REGRESSION mode for attributes: {predicted_attributes}")
            
            if config.USE_PREVIOUS_MODEL and self._check_if_model_exists():
                # Load existing multi-regression model
                self._load_multi_regression_model()
                print(f"[{self.__class__.__name__}] Loaded existing multi-regression model.")
            else:
                # Build new multi-regression model
                self.model = self._build_multi_output_model()
                self.target_scaler = TargetScaler(predicted_attributes)
                print(f"[{self.__class__.__name__}] Created new multi-regression model.")
                
            self.models = {}  # Keep empty for compatibility
        else:
            print(f"[{self.__class__.__name__}] Initializing in SINGLE-PREDICTION mode for attributes: {predicted_attributes}")
            self.models: Dict[str, tf.keras.Model] = {}
            self.model = None  # Single model not used in this mode
            
            # Create a model for each quality attribute
            for attr in self.predicted_attributes:
                if config.USE_PREVIOUS_MODEL and self._check_if_model_exists(attr):
                    # If we want to load existing model weights
                    loaded_model = self._load_individual_model(attr)
                    self.models[attr] = loaded_model
                    print(f"[{self.__class__.__name__}] Loaded existing model for '{attr}'.")
                else:
                    # Build a new model for this attribute
                    new_model = self._build_model_for_attr()
                    self.models[attr] = new_model
                    print(f"[{self.__class__.__name__}] Created new model for '{attr}'.")

    @abstractmethod
    def _build_model_for_attr(self) -> tf.keras.Model:
        """
        Build a single-output model for one quality attribute.
        Must be implemented by concrete DL model classes.
        """
        pass
        
    @abstractmethod
    def _build_multi_output_model(self) -> tf.keras.Model:
        """
        Build a multi-output model that predicts all quality attributes simultaneously.
        Must be implemented by concrete DL model classes.
        """
        pass

    def _check_if_model_exists(self, attr: str = None) -> bool:
        """
        Check if a model exists in the organized folder system.
        
        Args:
            attr: Attribute name for single prediction mode, None for multi-regression
        """
        try:
            from utils.model_config_manager import ModelConfigManager
            
            # Generate the same configuration to get the hash
            config_manager = ModelConfigManager()
            ndsi_band_pairs = getattr(config, 'NDSI_BAND_PAIRS', [])
            
            save_info = config_manager.save_model_complete(
                model_instance=self,
                model_type=self.model_type,
                model_shape=self.model_shape,
                components=self.components,
                component_dimensions=self.component_dimensions,
                selected_bands=self.selected_bands,
                selected_indexes=self.selected_indexes,
                predicted_attributes=self.predicted_attributes,
                ndsi_band_pairs=ndsi_band_pairs
            )
            
            # Check if model file exists
            if config.MULTI_REGRESSION:
                model_file = save_info["model_file"]
                scaler_file = os.path.join(save_info["model_dir"], "target_scaler.json")
                exists = os.path.exists(model_file) and os.path.exists(scaler_file)
                if exists:
                    print(f"[{self.__class__.__name__}] Found existing multi-regression model at: {save_info['model_dir']}")
                return exists
            else:
                if attr:
                    model_file = os.path.join(save_info["model_dir"], f"model_{attr}.keras")
                    exists = os.path.exists(model_file)
                    if exists:
                        print(f"[{self.__class__.__name__}] Found existing model for '{attr}' at: {model_file}")
                    return exists
                else:
                    # Check if any attribute model exists
                    found_attrs = []
                    for attr_name in self.predicted_attributes:
                        model_file = os.path.join(save_info["model_dir"], f"model_{attr_name}.keras")
                        if os.path.exists(model_file):
                            found_attrs.append(attr_name)
                    if found_attrs:
                        print(f"[{self.__class__.__name__}] Found existing models for attributes: {found_attrs} at: {save_info['model_dir']}")
                    return len(found_attrs) > 0
                    
        except Exception as e:
            print(f"[{self.__class__.__name__}] Error checking for existing models: {e}")
            return False

    def _load_individual_model(self, attr: str) -> tf.keras.Model:
        """
        Load a single attribute model from the organized system.
        """
        from utils.model_config_manager import ModelConfigManager
        
        config_manager = ModelConfigManager()
        ndsi_band_pairs = getattr(config, 'NDSI_BAND_PAIRS', [])
        
        save_info = config_manager.save_model_complete(
            model_instance=self,
            model_type=self.model_type,
            model_shape=self.model_shape,
            components=self.components,
            component_dimensions=self.component_dimensions,
            selected_bands=self.selected_bands,
            selected_indexes=self.selected_indexes,
            predicted_attributes=self.predicted_attributes,
            ndsi_band_pairs=ndsi_band_pairs
        )
        
        model_file = os.path.join(save_info["model_dir"], f"model_{attr}.keras")
        if os.path.exists(model_file):
            print(f"[{self.__class__.__name__}] Loading existing model for '{attr}' from: {model_file}")
            return keras_load_model(model_file, safe_mode=False)
        else:
            raise FileNotFoundError(f"No model found for '{attr}' at: {model_file}")

    def prepare_data(
        self,
        tomatoes: List[Tomato],
        selected_bands: List[int],
        augment_times: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data by extracting features and targets from tomato objects.

        Args:
            tomatoes (List[Tomato]): List of Tomato objects.
            selected_bands (List[int]): List of selected band indices.
            augment_times (int): Number of times to augment the data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature matrix X and target matrix Y.
        """
        # Check if we should use memory-optimized preparation for feature selection
        if (hasattr(self, 'fs_interface') and self.fs_interface is not None and 
            self.fs_interface.is_enabled() and selected_bands is None):
            print("[BaseDLModel] Using memory-optimized data preparation for feature selection")
            return self._prepare_data_memory_optimized(tomatoes, augment_times)
        
        X_features = []
        Y_targets = []

        # Handle None case for feature selection mode
        effective_bands = selected_bands if selected_bands is not None else list(range(204))
        
        band_pairs = DataProcessingUtils.generate_band_pairs(selected_bands)
        augmented_tomatoes = DataProcessingUtils.create_augmented_and_padded_tomatoes(
            tomatoes=tomatoes,
            augment_times=augment_times
        )
        del tomatoes
        gc.collect()
        all_tomatoes = augmented_tomatoes
        
        for tomato in all_tomatoes:
            channel_list = []
            reflectance_matrix = tomato.spectral_stats.reflectance_matrix

            # Reflectance
            if self.components.get('reflectance', False):
                reflectance_features = reflectance_matrix[:, :, effective_bands]
                channel_list.append(reflectance_features)

            # Special STD Images
            if self.components.get('std', False):
                special_std_channels = []
                for i, band_idx in enumerate(effective_bands):
                    # Get the reflectance for this band (use array index, not wavelength index)
                    I_k = reflectance_matrix[:, :, i]
                    
                    # Create mask for the tomato area (non-zero pixels)
                    mask = I_k > 0
                    
                    # Calculate N (number of pixels in the tomato)
                    N = np.sum(mask)
                    
                    if N > 0:  # Only proceed if there are tomato pixels
                        # Calculate mean (μ_k) of reflectance in the tomato area
                        mu_k = np.sum(I_k * mask) / N
                        
                        # Calculate standard deviation (σ_k)
                        squared_diff = ((I_k - mu_k) ** 2) * mask
                        sigma_k = np.sqrt(np.sum(squared_diff) / N)
                        
                        # Create the special STD image S_k
                        if sigma_k > 0:  # Avoid division by zero
                            S_k = np.zeros_like(I_k)
                            S_k[mask] = squared_diff[mask] #/ (N * sigma_k)
                        else:
                            # If sigma_k is zero, all pixels have the same value
                            S_k = np.zeros_like(I_k)
                    else:
                        # If no tomato pixels, return zeros
                        S_k = np.zeros_like(I_k)
                    
                    special_std_channels.append(S_k)
                
                if special_std_channels:
                    special_std_all = np.stack(special_std_channels, axis=-1)
                    channel_list.append(special_std_all)
                else:
                    print(f"Warning: No STD channels were processed for {len(effective_bands)} bands.")

            # NDSI
            if self.components.get('ndsi', False):
                ndsi_channels = []
                for band_i, band_j in band_pairs:
                    # Map wavelength indices to array positions
                    # try:
                    #     band_i_idx = effective_bands.index(band_i)
                    #     band_j_idx = effective_bands.index(band_j)
                    # except ValueError as e:
                    #     print(f"Warning: Band pair ({band_i}, {band_j}) not found in selected bands {effective_bands}. Skipping.")
                    #     continue
                    
                    R_i = reflectance_matrix[:, :, band_i]
                    R_j = reflectance_matrix[:, :, band_j]
                    mask = (R_i == 0) & (R_j == 0)
                    ## for pixel wise devision (the previous way) delete the lines untill (include) the line 
                    ## "ndsi = diff / total_sum" and uncomment this line: ndsi = (R_i - R_j) / (R_i + R_j + 1e-8)
                    # Calculate the difference matrix
                    diff = R_i - R_j
                    
                    # Calculate the sum of all pixels in (R_i + R_j) as a single scalar value
                    total_sum = np.sum(R_i + R_j)
                    
                    # Add a small epsilon to avoid division by zero
                    total_sum = total_sum + 1e-8 if total_sum == 0 else total_sum
                    
                    # Divide the difference matrix by the scalar sum
                    ndsi = diff / total_sum
                    # # Calculate NDSI pixel-wise (the original way)
                    # ndsi = (R_i - R_j) / (R_i + R_j + 1e-8)
                    # Apply the mask
                    ndsi[mask] = 0
                    ndsi_channels.append(ndsi)

                # Only stack if there are NDSI channels to process
                if ndsi_channels:
                    ndsi_all = np.stack(ndsi_channels, axis=-1)
                    channel_list.append(ndsi_all)
                else:
                    print(f"Warning: No NDSI channels were processed. Expected {len(band_pairs)} channels.")

            # Spectral Indexes
            if self.components.get('indexes', False) and self.selected_indexes:
                spectral_index_channels = []
                for index in self.selected_indexes:
                    index_value = SpectralIndexCalculator.compute_pixel_index(
                        reflectance_matrix=reflectance_matrix,
                        selected_bands=effective_bands,
                        spectral_index=index
                    )
                    spectral_index_channels.append(index_value)
                if spectral_index_channels:
                    index_all = np.stack(spectral_index_channels, axis=-1)
                    channel_list.append(index_all)
                else:
                    print(f"Warning: No spectral index channels were processed from {len(self.selected_indexes) if self.selected_indexes else 0} indexes.")

            if len(channel_list) > 1:
                combined_features = np.concatenate(channel_list, axis=-1)
            elif len(channel_list) == 1:
                combined_features = channel_list[0]
            else:
                raise ValueError("No component chosen! Enable reflectance, std, ndsi or indexes.")

            # Build the Y vector: [pH, TSS, etc.]
            quality_assess = tomato.quality_assess
            y_values = [getattr(quality_assess, attr, np.nan) for attr in config.PREDICTED_QUALITY_ATTRIBUTES]
            X_features.append(combined_features)
            Y_targets.append(y_values)

        X = np.array(X_features)
        Y = np.array(Y_targets, dtype=float)
        Y_filled = DataProcessingUtils.fill_missing_values(Y, config.PREDICTED_QUALITY_ATTRIBUTES)
        Y = Y_filled

        # Verify model shape matches data dimensions
        # Account for feature selection: if FS is enabled, data will be transformed later
        expected_shape = self.model_shape
        data_shape = X.shape[1:]
        
        # Check if feature selection is enabled and will transform the data
        has_feature_selection = hasattr(self, 'fs_interface') and hasattr(self.fs_interface, 'is_enabled') and self.fs_interface.is_enabled()
        
        if has_feature_selection:
            # With feature selection, the data will be transformed, so we need to check the original shape
            original_shape = getattr(self, 'original_model_shape', None)
            if original_shape is not None:
                # We expect the data to match the original shape before feature selection
                expected_shape_for_validation = original_shape
                print(f"[{self.__class__.__name__}] Feature Selection enabled: checking against original shape {expected_shape_for_validation}")
            else:
                # If no original shape stored, skip validation (feature selection will handle it)
                print(f"[{self.__class__.__name__}] Feature Selection enabled: skipping shape validation (will be handled by FS)")
                expected_shape_for_validation = None
        else:
            expected_shape_for_validation = expected_shape
        
        if expected_shape_for_validation is not None and data_shape != expected_shape_for_validation:
            print(f"[{self.__class__.__name__}] Warning: Data shape {data_shape} doesn't match expected shape {expected_shape_for_validation}")
            print(f"[{self.__class__.__name__}] Expected channels: {self.component_dimensions}, got: {X.shape[3]}")
            if has_feature_selection:
                print(f"[{self.__class__.__name__}] Note: Feature selection will transform data from {data_shape} to {expected_shape}")
            else:
                raise ValueError("Model shape does not match data dimensions")
        
        print(f"[{self.__class__.__name__}] The X shape is: {X.shape}")
        if has_feature_selection:
            print(f"[{self.__class__.__name__}] Feature selection will transform to: {expected_shape}")
        
        # Clean up memory after data preparation
        gc.collect()
        
        
        return X, Y
    
    def _prepare_data_memory_optimized(
        self,
        tomatoes: List[Tomato],
        augment_times: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Memory-optimized data preparation for feature selection.
        Only loads full spectrum for band selection, then works with reduced data.
        """
        from utils.memory_optimized_data_preparation import MemoryOptimizedDataPreparation
        
        # Get current stage from feature selector
        current_stage = 'bands'  # Default
        if hasattr(self.fs_interface, 'active_stage'):
            current_stage = self.fs_interface.active_stage or 'bands'
        
        print(f"[BaseDLModel] Memory-optimized preparation for stage: {current_stage}")
        
        # For band selection stage, we need full spectrum
        if current_stage == 'bands' and not hasattr(self.fs_interface, '_preserved_selections'):
            print("[BaseDLModel] Loading full spectrum for band selection...")
            X, Y = MemoryOptimizedDataPreparation._extract_full_spectrum(tomatoes, augment_times)
            print(f"[BaseDLModel] Full spectrum data shape: {X.shape} ({X.nbytes / 1024 / 1024:.2f} MB)")
        else:
            # For other stages, load only selected bands
            if hasattr(self.fs_interface, '_preserved_selections') and 'band_indices' in self.fs_interface._preserved_selections:
                selected_indices = self.fs_interface._preserved_selections['band_indices']
                print(f"[BaseDLModel] Loading only {len(selected_indices)} selected bands...")
            else:
                # Fallback to all bands if no selection available
                selected_indices = list(range(204))
                print("[BaseDLModel] No band selection found, loading all bands...")
            
            X, Y = MemoryOptimizedDataPreparation._extract_selected_bands(
                tomatoes, selected_indices, augment_times
            )
            print(f"[BaseDLModel] Reduced data shape: {X.shape} ({X.nbytes / 1024 / 1024:.2f} MB)")
        
        # Apply feature selection transformations
        if current_stage != 'bands':
            # For stages after band selection, we need to compute additional features
            print(f"[BaseDLModel] Computing features for stage: {current_stage}")
            X = self._compute_stage_features(X, current_stage)
        
        return X, Y
    
    def _compute_stage_features(
        self,
        X: np.ndarray,
        stage: str
    ) -> np.ndarray:
        """
        Compute stage-specific features from selected bands.
        """
        channel_list = []
        
        # Always include reflectance if enabled
        if self.components.get('reflectance', False):
            channel_list.append(X)
        
        # Add STD features if we're in std stage or later
        if self.components.get('std', False) and stage in ['std', 'indexes', 'ndsi', 'finetune']:
            print("[BaseDLModel] Computing STD features...")
            std_features = self._compute_std_features(X)
            channel_list.append(std_features)
        
        # Add index features if we're in indexes stage or later
        if self.components.get('indexes', False) and stage in ['indexes', 'ndsi', 'finetune']:
            print("[BaseDLModel] Computing index features...")
            # For memory optimization, we'll skip complex index computation
            # The feature selector will handle this during training
            pass
        
        # Add NDSI features if we're in ndsi stage or later
        if self.components.get('ndsi', False) and stage in ['ndsi', 'finetune']:
            print("[BaseDLModel] Computing NDSI features...")
            ndsi_features = self._compute_ndsi_features(X)
            channel_list.append(ndsi_features)
        
        # Combine all features
        if len(channel_list) > 1:
            X_combined = np.concatenate(channel_list, axis=-1)
        else:
            X_combined = channel_list[0] if channel_list else X
        
        return X_combined
    
    def _compute_std_features(self, X: np.ndarray) -> np.ndarray:
        """Compute STD features from reflectance data."""
        num_bands = X.shape[-1]
        std_channels = []
        
        for i in range(num_bands):
            I_k = X[:, :, :, i]
            
            # Compute STD for each sample
            std_maps = []
            for j in range(X.shape[0]):
                I_k_sample = I_k[j]
                mask = I_k_sample > 0
                N = np.sum(mask)
                
                if N > 0:
                    mu_k = np.sum(I_k_sample * mask) / N
                    squared_diff = ((I_k_sample - mu_k) ** 2) * mask
                    sigma_k = np.sqrt(np.sum(squared_diff) / N)
                    
                    if sigma_k > 0:
                        S_k = np.zeros_like(I_k_sample)
                        S_k[mask] = squared_diff[mask]
                    else:
                        S_k = np.zeros_like(I_k_sample)
                else:
                    S_k = np.zeros_like(I_k_sample)
                
                std_maps.append(S_k)
            
            std_channels.append(np.array(std_maps))
        
        return np.stack(std_channels, axis=-1)
    
    def _compute_ndsi_features(self, X: np.ndarray) -> np.ndarray:
        """Compute NDSI features from reflectance data."""
        num_bands = X.shape[-1]
        ndsi_channels = []
        
        # Generate pairs from available bands
        pairs = []
        for i in range(num_bands):
            for j in range(i + 1, min(i + 2, num_bands)):
                pairs.append((i, j))
        
        # Limit to configured number of NDSI pairs
        max_pairs = getattr(config, 'FEATURE_SELECTION_A_NDSI', 5)
        pairs = pairs[:max_pairs]
        
        for band_i, band_j in pairs:
            R_i = X[:, :, :, band_i]
            R_j = X[:, :, :, band_j]
            
            # Compute NDSI for each sample
            ndsi_maps = []
            for k in range(X.shape[0]):
                R_i_sample = R_i[k]
                R_j_sample = R_j[k]
                
                mask = (R_i_sample == 0) & (R_j_sample == 0)
                diff = R_i_sample - R_j_sample
                total_sum = np.sum(R_i_sample + R_j_sample)
                total_sum = total_sum + 1e-8 if total_sum == 0 else total_sum
                
                ndsi = diff / total_sum
                ndsi[mask] = 0
                ndsi_maps.append(ndsi)
            
            ndsi_channels.append(np.array(ndsi_maps))
        
        return np.stack(ndsi_channels, axis=-1)

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = None,
        **kwargs
    ) -> None:
        """
        Train the model(s) based on the regression mode.
        """
        import time
        
        # Start timing
        start_time = time.time()
        
        # Use config batch size if none provided
        if batch_size is None:
            batch_size = config.BATCH_SIZE
        
        # Check if feature selection is enabled for custom training loop
        has_feature_selection = hasattr(self, 'fs_interface') and self.fs_interface is not None
        
        # Always use progressive feature selection when FS is enabled
        if has_feature_selection:
            print(f"[{self.__class__.__name__}] Using PROGRESSIVE feature selection training")
            self._train_progressive_feature_selection(X_train, Y_train, X_val, Y_val, epochs, batch_size, **kwargs)
        else:
            print(f"[{self.__class__.__name__}] Using standard training loop")
            if config.MULTI_REGRESSION:
                self._train_multi_regression(X_train, Y_train, X_val, Y_val, epochs, batch_size, **kwargs)
            else:
                self._train_single_prediction(X_train, Y_train, X_val, Y_val, epochs, batch_size, **kwargs)
        
        # Calculate training time in minutes
        self.training_time_minutes = (time.time() - start_time) / 60.0
        print(f"[{self.__class__.__name__}] Training completed in {self.training_time_minutes:.2f} minutes")
        
        # Clean up memory after training completes
        gc.collect()
        
    
    def _train_with_feature_selection_custom(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        **kwargs
    ) -> None:
        """
        Custom training loop with feature selection enabled.
        """
        print(f"\n[{self.__class__.__name__}] Training with custom feature selection loop")
        
        # Initialize model_info dictionary
        model_info = {}
        
        # Filter out samples where ANY target value is NaN
        valid_mask = ~np.isnan(Y_train).any(axis=1)
        X_train_filtered = X_train[valid_mask]
        Y_train_filtered = Y_train[valid_mask]
        
        if X_train_filtered.shape[0] == 0:
            print(f"[{self.__class__.__name__}] No valid training data, skipping training.")
            return
        
        # Fit target scaler and transform targets
        Y_train_scaled = self.target_scaler.fit_transform(Y_train_filtered)
        
        # Free memory consumed by the unscaled targets now that we have the scaled version
        del Y_train_filtered
        gc.collect()
        
        # Prepare validation data if provided
        X_val_filtered, Y_val_scaled = None, None
        if X_val is not None and Y_val is not None:
            val_valid_mask = ~np.isnan(Y_val).any(axis=1)
            X_val_filtered = X_val[val_valid_mask]
            Y_val_filtered = Y_val[val_valid_mask]
            Y_val_scaled = self.target_scaler.transform(Y_val_filtered)
            del Y_val_filtered
            gc.collect()
        
        # Setup optimizer
        initial_lr = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=100,
            decay_rate=0.96,
            staircase=True
        )
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        # Create datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_filtered, Y_train_scaled))
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        if X_val_filtered is not None:
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val_filtered, Y_val_scaled))
            val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        else:
            val_dataset = None
        
        # Training metrics
        train_loss_metric = tf.keras.metrics.Mean()
        val_loss_metric = tf.keras.metrics.Mean()
        
        # Initialize training loop variables
        best_train_loss = float('inf')
        best_val_loss = float('inf')
        patience_counter = 0
        patience = float('inf')  # Disable global patience during feature selection
        
        # Initialize validation tracking
        previous_val_loss = None
        previous_val_attribute_losses = None
        
        # Get stage convergence settings
        use_stage_convergence = getattr(config, 'FEATURE_SELECTION_USE_STAGE_CONVERGENCE', True)
        
        # 🎯 NEW: Update model info with parameter count
        if hasattr(self, 'model') and self.model is not None:
            model_info['total_parameters'] = self.model.count_params()
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Reset metrics
            train_loss_metric.reset_state()
            val_loss_metric.reset_state()
            
            # Training step
            epoch_step = 0
            for step, (x_batch, y_batch) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    # 🎯 NEW: Run hierarchical feature selection if enabled (only once per epoch)
                    hierarchical_info = {}
                    if hasattr(self.fs_interface, 'run_hierarchical_feature_selection') and step == 0:
                        hierarchical_info = self.fs_interface.run_hierarchical_feature_selection(
                            x_batch, self.model, None, None, epoch
                        )
                        # Store hierarchical weights for loss computation
                        if hierarchical_info.get('stage_weights'):
                            self.fs_interface._hierarchical_weights = hierarchical_info['stage_weights']
                        # Store hierarchical info for STD selection
                        self.fs_interface._hierarchical_info = hierarchical_info
                    elif hasattr(self.fs_interface, '_hierarchical_info'):
                        # Use cached hierarchical info for subsequent steps in the same epoch
                        hierarchical_info = getattr(self.fs_interface, '_hierarchical_info', {})
                    
                    # Apply feature selection
                    processed_data, selection_info = self.fs_interface.process_hyperspectral_data(
                        x_batch, training=True, epoch=epoch
                    )
                    
                    # 🎯 MEMORY LEAK FIX: Delete original tensor to free GPU memory
                    del x_batch  # Free 204-band tensor, keep only 5-band processed_data
                    
                    # Capture actual selections on first step of first epoch
                    if epoch == 0 and step == 0:
                        self._capture_feature_selection_results(selection_info, epoch)
                    
                    # Model prediction
                    predictions = self.model(processed_data, training=True)
                    
                    # Compute prediction loss
                    prediction_loss = tf.reduce_mean(tf.keras.losses.mse(y_batch, predictions))
                    
                    # 🎯 NEW: Compute attribute-specific losses for feature selection
                    attribute_losses = None
                    if predictions.shape[-1] == len(self.predicted_attributes):
                        attribute_losses = {}
                        for i, attr in enumerate(self.predicted_attributes):
                            # Compute MSE loss on scaled targets (0-1)
                            scaled_loss = tf.reduce_mean(tf.keras.losses.mse(y_batch[:, i], predictions[:, i]))
                            
                            # 🎯 CRITICAL: Normalize by attribute scale to make losses comparable
                            if hasattr(self, 'target_scaler') and hasattr(self.target_scaler, 'get_scale_info'):
                                scale_info = self.target_scaler.get_scale_info()
                                attr_range = scale_info.get(attr, {}).get('range', 1.0)
                                # Normalize loss by squared range to account for MSE scaling
                                normalized_loss = scaled_loss * (attr_range ** 2)
                                attribute_losses[attr] = normalized_loss
                            else:
                                # Fallback: use scaled loss directly
                                attribute_losses[attr] = scaled_loss
                    
                    # 🎯 NEW: Get strategy weights for loss balancing
                    strategy_weights = self.fs_interface.get_strategy_loss_weights(epoch=epoch)
                    fs_weight = strategy_weights['feature_selection_weight']
                    pred_weight = strategy_weights['prediction_weight']
                    
                    # 🎯 NEW: Get feature selection loss (Recommendation 6: Validation Optimization)
                    fs_loss, loss_breakdown = self.fs_interface.get_prediction_aware_feature_selection_loss(
                        selection_info, prediction_loss, epoch=epoch, attribute_losses=attribute_losses,
                        validation_loss=previous_val_loss, validation_attribute_losses=previous_val_attribute_losses
                    )
                    
                    # 🎯 NEW: Apply strategy weights to balance FS loss vs prediction loss
                    total_loss = fs_weight * fs_loss + pred_weight * prediction_loss
                    
                    # Update loss breakdown with weighted components
                    loss_breakdown['strategy_fs_weight'] = fs_weight
                    loss_breakdown['strategy_pred_weight'] = pred_weight
                    loss_breakdown['weighted_fs_loss'] = fs_weight * fs_loss
                    loss_breakdown['weighted_pred_loss'] = pred_weight * prediction_loss
                    loss_breakdown['final_total_loss'] = total_loss
                
                # Apply gradients
                # 🎯 CRITICAL FIX: Include feature selection parameters in gradient computation
                all_trainable_vars = self.model.trainable_variables
                if hasattr(self.fs_interface, 'get_trainable_variables'):
                    fs_vars = self.fs_interface.get_trainable_variables()
                    if fs_vars:
                        all_trainable_vars = all_trainable_vars + fs_vars
                
                gradients = tape.gradient(total_loss, all_trainable_vars)
                optimizer.apply_gradients(zip(gradients, all_trainable_vars))
                
                train_loss_metric.update_state(total_loss)
                
                # 🎯 NEW: Log training step details
                step_info = {
                    'total_loss': float(total_loss),
                    'prediction_loss': float(prediction_loss),
                    'loss_breakdown': loss_breakdown,
                    'selection_info': selection_info
                }
                
                
                # Print progress every 10 steps
                if step % 10 == 0:
                    # 🎯 UPDATED: Include new loss components
                    quality_loss = loss_breakdown.get('quality_alignment_loss', 0)
                    confidence_loss = loss_breakdown.get('confidence_loss', 0)
                    diversity_loss = loss_breakdown.get('diversity_loss', 0)
                    reinforcement_loss = loss_breakdown.get('gate_reinforcement_loss', 0)
                    sparsity_loss = loss_breakdown.get('sparsity_loss', 0)
                    spectral_region_loss = loss_breakdown.get('spectral_region_loss', 0)
                    attribute_weighted_loss = loss_breakdown.get('attribute_weighted_loss', prediction_loss)
                    
                    print(f"  Step {step}: Total = {total_loss:.4f}, Pred = {prediction_loss:.4f}")
                    print(f"    AttrWeighted = {attribute_weighted_loss:.4f}, Quality = {quality_loss:.4f}")
                    print(f"    Conf = {confidence_loss:.4f}, Div = {diversity_loss:.4f}, Reinf = {reinforcement_loss:.4f}")
                    print(f"    Sparse = {sparsity_loss:.4f}, SpectralDiv = {spectral_region_loss:.4f}")
                    
                    # 🎯 NEW: Show validation optimization status (Recommendation 6)
                    if loss_breakdown.get('validation_optimization_active', False):
                        val_loss_tensor = loss_breakdown.get('validation_loss', None)
                        if val_loss_tensor is not None:
                            val_loss_display = f"{float(val_loss_tensor):.4f}"
                            print(f"    [Validation Optimization] Val Loss: {val_loss_display}, Priority: Val  Train")
                        else:
                            print(f"    [Validation Optimization] Active, Priority: Val  Train")
                    
                    # 🎯 NEW: Show hierarchical stage info (only once per epoch to avoid spam)
                    if hierarchical_info and hierarchical_info.get('stage'):
                        stage = hierarchical_info['stage']
                        if not hasattr(self, '_last_stage_print_epoch') or self._last_stage_print_epoch != epoch:
                            print(f"    [Hierarchical Stage: {stage.upper()}]")
                            self._last_stage_print_epoch = epoch
                
                epoch_step += 1
            
            # Validation step
            val_loss = None
            # 🎯 NEW: Initialize validation tracking for next epoch (Recommendation 6)
            current_val_loss = None
            current_val_attribute_losses = None
            
            if val_dataset is not None:
                val_step = 0
                for x_batch, y_batch in val_dataset:
                    # Apply feature selection
                    processed_data, selection_info = self.fs_interface.process_hyperspectral_data(
                        x_batch, training=False, epoch=epoch
                    )
                    
                    # Model prediction
                    predictions = self.model(processed_data, training=False)
                    
                    # Compute validation loss
                    val_prediction_loss = tf.reduce_mean(tf.keras.losses.mse(y_batch, predictions))
                    
                    # 🎯 NEW: Compute validation attribute-specific losses
                    val_attribute_losses = None
                    if predictions.shape[-1] == len(self.predicted_attributes):
                        val_attribute_losses = {}
                        for i, attr in enumerate(self.predicted_attributes):
                            # Compute MSE loss on scaled targets (0-1)
                            scaled_loss = tf.reduce_mean(tf.keras.losses.mse(y_batch[:, i], predictions[:, i]))
                            
                            # 🎯 CRITICAL: Normalize by attribute scale to make losses comparable
                            if hasattr(self, 'target_scaler') and hasattr(self.target_scaler, 'get_scale_info'):
                                scale_info = self.target_scaler.get_scale_info()
                                attr_range = scale_info.get(attr, {}).get('range', 1.0)
                                # Normalize loss by squared range to account for MSE scaling
                                normalized_loss = scaled_loss * (attr_range ** 2)
                                val_attribute_losses[attr] = normalized_loss
                            else:
                                # Fallback: use scaled loss directly
                                val_attribute_losses[attr] = scaled_loss
                    
                    # 🎯 NEW: Apply strategy weights to validation loss as well
                    strategy_weights = self.fs_interface.get_strategy_loss_weights(epoch=epoch)
                    fs_weight = strategy_weights['feature_selection_weight']
                    pred_weight = strategy_weights['prediction_weight']
                    
                    val_fs_loss, val_loss_breakdown = self.fs_interface.get_prediction_aware_feature_selection_loss(
                        selection_info, val_prediction_loss, epoch=epoch, attribute_losses=val_attribute_losses
                    )
                    
                    # Apply strategy weights to validation loss
                    val_total_loss = fs_weight * val_fs_loss + pred_weight * val_prediction_loss
                    
                    # Update validation loss breakdown
                    val_loss_breakdown['strategy_fs_weight'] = fs_weight
                    val_loss_breakdown['strategy_pred_weight'] = pred_weight
                    val_loss_breakdown['weighted_fs_loss'] = fs_weight * val_fs_loss
                    val_loss_breakdown['weighted_pred_loss'] = pred_weight * val_prediction_loss
                    val_loss_breakdown['final_total_loss'] = val_total_loss
                    
                    val_loss_metric.update_state(val_total_loss)
                    
                    # 🎯 NEW: Store validation information for next epoch (Recommendation 6)
                    if val_step == 0:  # Store only from first validation batch to avoid memory issues
                        current_val_loss = val_prediction_loss
                        current_val_attribute_losses = val_attribute_losses
                    
                    # 🎯 NEW: Log validation step (less frequent)
                    if val_step % 20 == 0:
                        val_step_info = {
                            'total_loss': float(val_total_loss),
                            'prediction_loss': float(val_prediction_loss),
                            'loss_breakdown': val_loss_breakdown,
                            'is_validation': True
                        }
                    
                    val_step += 1
                
                val_loss = val_loss_metric.result()
            
            # Print epoch results
            train_loss = train_loss_metric.result()
            print(f"  Train Loss: {train_loss:.4f}")
            
            # Update best train loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                
                # 🎯 CRITICAL FIX: Save best model checkpoint when training loss improves (no validation)
                if val_loss is None:
                    self._save_best_model_checkpoint(epoch, train_loss)
            
            # 🎯 NEW: Log epoch summary
            train_metrics = {'train_loss': float(train_loss)}
            val_metrics = {'val_loss': float(val_loss)} if val_loss is not None else None
            
            if val_loss is not None:
                print(f"  Val Loss: {val_loss:.4f}")
                
                # 🆕 HYBRID EARLY STOPPING LOGIC
                # Combine stage convergence with global patience for better control
                
                global_patience_triggered = False
                stage_action = 'continue'
                
                # 1. GLOBAL PATIENCE CHECK (always active as safety net)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # 🎯 CRITICAL FIX: Save best model checkpoint when validation loss improves
                    self._save_best_model_checkpoint(epoch, val_loss)
                else:
                    patience_counter += 1
                    
                if patience_counter > patience:
                    global_patience_triggered = True
                    print(f"  🚨 Global patience triggered after {patience} epochs without improvement")
                
                # 2. STAGE CONVERGENCE CHECK (if enabled)
                if use_stage_convergence and hasattr(self.fs_interface, 'update_stage_convergence'):
                    stage_convergence_info = self.fs_interface.update_stage_convergence(float(val_loss))
                    stage_action = stage_convergence_info.get('action', 'continue')
                    
                    if stage_action == 'training_complete':
                        print(f"  ✅ All feature selection stages completed - ending training")
                        break
                    elif stage_action == 'transition_to_next_stage':
                        print(f"  🔄 Stage converged - transitioning to next stage")
                        # Continue training for next stage
                
                # 3. HYBRID DECISION LOGIC
                # Stop training if global patience is exhausted (safety net)
                if global_patience_triggered:
                    # Check if we're in a reasonable stage progression
                    if hasattr(self.fs_interface, '_stage_tracker') and self.fs_interface._stage_tracker:
                        current_stage = self.fs_interface._stage_tracker.get('current_stage', 'unknown')
                        stage_epoch = self.fs_interface._stage_tracker.get('stage_epoch', 0)
                        print(f"  ⚠️  Global patience exhausted during stage '{current_stage}' at stage epoch {stage_epoch}")
                        print(f"  🛑 Stopping training to prevent excessive runtime")
                        break
                    else:
                        # No stage tracking, use standard early stopping
                        print(f"  🛑 Early stopping triggered after {patience} epochs without improvement")
                        break
                
                # 4. STAGE PROGRESSION TRACKING (if stage convergence is enabled)
                if use_stage_convergence and hasattr(self.fs_interface, '_stage_tracker'):
                    tracker = self.fs_interface._stage_tracker
                    if tracker and 'stage_epoch' in tracker:
                        # CRITICAL FIX: Increment stage epoch counter
                        tracker['stage_epoch'] += 1
                        
                        # Log stage progress periodically
                        if tracker['stage_epoch'] % 10 == 0:
                            current_stage = tracker.get('current_stage', 'unknown')
                            stage_patience_counter = tracker.get('stage_patience_counter', 0)
                            stage_patience = tracker.get('stage_patience', 8)
                            print(f"  📊 Stage '{current_stage}': epoch {tracker['stage_epoch']}, patience {stage_patience_counter}/{stage_patience}")
            
            
            # 🎯 NEW: Update validation information for next epoch (Recommendation 6)
            if current_val_loss is not None:
                previous_val_loss = current_val_loss
                previous_val_attribute_losses = current_val_attribute_losses
                if epoch % 20 == 0:  # Log validation optimization status periodically
                    print(f"  [Validation Optimization] Stored validation loss for next epoch: {float(current_val_loss):.4f}")
            
        
        # 🎯 NEW: Log training completion
        final_summary = {
            'final_train_loss': float(train_loss),
            'final_val_loss': float(val_loss) if val_loss is not None else None,
            'total_epochs_trained': epoch + 1,
            'early_stopping_triggered': patience_counter == patience,
            'best_val_loss': float(best_val_loss) if val_loss is not None else None
        }
        
        # 🎯 CRITICAL FIX: Save best model checkpoint during training, not at the end
        # The best model should already be saved by the checkpoint mechanism during training
        # We only need to save additional components (scaler, etc.) here
        self._save_training_checkpoint_components(best_val_loss if val_loss is not None else best_train_loss)
        
    
    def _save_best_model_checkpoint(self, epoch: int, val_loss: float):
        """Save the best model checkpoint during training when validation loss improves."""
        print(f"[{self.__class__.__name__}] 🎯 NEW BEST MODEL at epoch {epoch+1} (val_loss: {val_loss:.4f})")
        
        # Save the main model (best weights)
        if hasattr(self, 'model') and self.model is not None:
            model_path = self.checkpoint_manager.get_model_checkpoint_path()
            self.model.save(model_path)
            print(f"[{self.__class__.__name__}] Saved best main model checkpoint: {model_path}")
        
        # Save feature selection model if it exists and has a save method
        if hasattr(self, 'fs_interface') and self.fs_interface and hasattr(self.fs_interface, 'feature_selector'):
            if hasattr(self.fs_interface.feature_selector, 'save'):
                fs_model_path = self.checkpoint_manager.get_fs_model_checkpoint_path()
                self.fs_interface.feature_selector.save(fs_model_path)
                print(f"[{self.__class__.__name__}] Saved best FS model checkpoint: {fs_model_path}")
        
        print(f"[{self.__class__.__name__}] Best model checkpoint saved successfully")
    
    def _save_training_checkpoint_components(self, best_loss: float):
        """Save training checkpoint components (scaler, etc.) at the end of training."""
        print(f"[{self.__class__.__name__}] Saving training checkpoint components (best loss: {best_loss:.4f})")
        
        # Save target scaler
        if hasattr(self, 'target_scaler') and self.target_scaler is not None:
            self.checkpoint_manager.save_checkpoint_scaler(self.target_scaler)
        
        print(f"[{self.__class__.__name__}] Training checkpoint components saved successfully")
    
    def _load_training_checkpoint(self):
        """Load training checkpoint for testing."""
        print(f"[{self.__class__.__name__}] Loading training checkpoint for testing")
        
        # Load the main model
        try:
            if hasattr(self, 'model') and self.model is not None:
                self.model = self.checkpoint_manager.load_best_model()
                print(f"[{self.__class__.__name__}] Loaded main model from checkpoint")
        except FileNotFoundError:
            print(f"[{self.__class__.__name__}] No main model checkpoint found, using current model")
        
        # Load feature selection model if it exists and was saved
        try:
            if hasattr(self, 'fs_interface') and self.fs_interface and hasattr(self.fs_interface, 'feature_selector'):
                if hasattr(self.fs_interface.feature_selector, 'save'):
                    self.fs_interface.feature_selector = self.checkpoint_manager.load_best_fs_model()
                    print(f"[{self.__class__.__name__}] Loaded FS model from checkpoint")
                else:
                    print(f"[{self.__class__.__name__}] Feature selector doesn't support loading - using current model")
        except FileNotFoundError:
            print(f"[{self.__class__.__name__}] No FS model checkpoint found, using current model")
        
        # Load target scaler
        checkpoint_scaler = self.checkpoint_manager.load_checkpoint_scaler()
        if checkpoint_scaler is not None:
            self.target_scaler = checkpoint_scaler
            print(f"[{self.__class__.__name__}] Loaded target scaler from checkpoint")
        
        print(f"[{self.__class__.__name__}] Training checkpoint loaded successfully")
    

    def _train_multi_regression(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        **kwargs
    ) -> None:
        """
        Train a single multi-output model for all quality attributes.
        """
        print(f"\n[{self.__class__.__name__}] Training multi-regression model for all attributes: {self.predicted_attributes}")
        
        # Filter out samples where ANY target value is NaN (for multi-regression we need complete data)
        valid_mask = ~np.isnan(Y_train).any(axis=1)
        X_train_filtered = X_train[valid_mask]
        Y_train_filtered = Y_train[valid_mask]
        
        print(f"[{self.__class__.__name__}] Filtered out {X_train.shape[0] - X_train_filtered.shape[0]} samples with any NaN values")
        print(f"[{self.__class__.__name__}] Training on {X_train_filtered.shape[0]} samples")
        
        if X_train_filtered.shape[0] == 0:
            print(f"[{self.__class__.__name__}] No valid training data, skipping training.")
            return
            
        # Fit target scaler and transform targets
        Y_train_scaled = self.target_scaler.fit_transform(Y_train_filtered)
        print(f"[{self.__class__.__name__}] Target scaling info:")
        for attr, info in self.target_scaler.get_scale_info().items():
            print(f"  {attr}: [{info['min']:.4f}, {info['max']:.4f}] (range: {info['range']:.4f})")
        
        # Free memory consumed by the unscaled targets now that we have the scaled version
        del Y_train_filtered
        gc.collect()
        
        # Prepare validation data if provided
        X_val_filtered, Y_val_scaled = None, None
        if X_val is not None and Y_val is not None:
            val_valid_mask = ~np.isnan(Y_val).any(axis=1)
            X_val_filtered = X_val[val_valid_mask]
            Y_val_filtered = Y_val[val_valid_mask]
            Y_val_scaled = self.target_scaler.transform(Y_val_filtered)
            del Y_val_filtered
            gc.collect()
            print(f"[{self.__class__.__name__}] Using {X_val_filtered.shape[0]} valid validation samples")
        
        # Setup callbacks
        monitor_metric = 'val_loss' if X_val_filtered is not None else 'loss'
        callbacks = []
        early_stop = EarlyStopping(monitor=monitor_metric, patience=config.PATIENCE, restore_best_weights=True)
        callbacks.append(early_stop)
        
        # Use new system for ModelCheckpoint if saving is enabled
        if config.SAVE_MODEL:
            # Generate the new system paths first
            config_manager = ModelConfigManager()
            ndsi_band_pairs = getattr(config, 'NDSI_BAND_PAIRS', [])
            
            save_info = config_manager.save_model_complete(
                model_instance=self,
                model_type=self.model_type,
                model_shape=self.model_shape,
                components=self.components,
                component_dimensions=self.component_dimensions,
                selected_bands=self.selected_bands,
                selected_indexes=self.selected_indexes,
                predicted_attributes=self.predicted_attributes,
                ndsi_band_pairs=ndsi_band_pairs
            )
            
            # Use the new system path for ModelCheckpoint
            model_checkpoint = ModelCheckpoint(
                filepath=save_info["model_file"],
                monitor=monitor_metric,
                save_best_only=True,
                verbose=1
            )
            callbacks.append(model_checkpoint)
            
            # Store save_info for later scaler saving
            self._save_info = save_info
        
        # Train the model
        if X_val_filtered is not None:
            self.model.fit(
                X_train_filtered, Y_train_scaled,
                validation_data=(X_val_filtered, Y_val_scaled),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
        else:
            self.model.fit(
                X_train_filtered, Y_train_scaled,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=1
            )
            
        # Save target scaler if we used the new system
        if config.SAVE_MODEL and hasattr(self, '_save_info'):
            from sklearn.preprocessing import StandardScaler
            from sklearn.utils._openmp_helpers import _openmp_effective_n_threads
            import json

            model_dir = os.path.dirname(self._save_info["model_file"])
            scaler_path = os.path.join(model_dir, "target_scaler.json")

            # Serialize TargetScaler which contains MinMaxScaler objects
            scaler_data = {
                'attributes': self.target_scaler.attributes,
                'is_fitted': self.target_scaler.is_fitted,
                'scalers': {}
            }
            
            # Serialize each MinMaxScaler
            for attr, scaler in self.target_scaler.scalers.items():
                scaler_data['scalers'][attr] = {
                    'data_min_': scaler.data_min_.tolist(),
                    'data_max_': scaler.data_max_.tolist(),
                    'data_range_': scaler.data_range_.tolist(),
                    'scale_': scaler.scale_.tolist(),
                    'min_': scaler.min_.tolist(),
                    'n_samples_seen_': int(scaler.n_samples_seen_),
                    'feature_range': scaler.feature_range
                }

            with open(scaler_path, 'w') as f:
                json.dump(scaler_data, f)
            print(f"[{self.__class__.__name__}] Saved target scaler to {scaler_path}")
            print(f"[{self.__class__.__name__}] Training completed with new system hash: {self._save_info['config_hash']}")
            
    def _train_single_prediction(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        **kwargs
    ) -> None:
        """
        Train separate models for each quality attribute.
        """
        monitor_metric = 'val_loss' if (X_val is not None and Y_val is not None) else 'loss'
        
        # Set up new system paths if saving is enabled
        save_info = None
        if config.SAVE_MODEL:
            config_manager = ModelConfigManager()
            ndsi_band_pairs = getattr(config, 'NDSI_BAND_PAIRS', [])
            
            save_info = config_manager.save_model_complete(
                model_instance=self,
                model_type=self.model_type,
                model_shape=self.model_shape,
                components=self.components,
                component_dimensions=self.component_dimensions,
                selected_bands=self.selected_bands,
                selected_indexes=self.selected_indexes,
                predicted_attributes=self.predicted_attributes,
                ndsi_band_pairs=ndsi_band_pairs
            )
            print(f"[{self.__class__.__name__}] Using new system with hash: {save_info['config_hash']}")
        
        for idx, attr in enumerate(self.predicted_attributes):
            print(f"\n[{self.__class__.__name__}] Training model for attribute: {attr}")
            
            # Filter out samples where the target value is NaN
            valid_mask = ~np.isnan(Y_train[:, idx])
            X_train_filtered = X_train[valid_mask]
            y_train_attr = Y_train[valid_mask, idx]
            
            print(f"[{self.__class__.__name__}] Filtered out {X_train.shape[0] - X_train_filtered.shape[0]} samples with NaN values for '{attr}'")
            print(f"[{self.__class__.__name__}] Training on {X_train_filtered.shape[0]} samples for '{attr}'")
            
            if X_train_filtered.shape[0] == 0:
                print(f"[{self.__class__.__name__}] No valid training data for '{attr}', skipping training.")
                continue

            callbacks = []
            early_stop = EarlyStopping(monitor=monitor_metric, patience=config.PATIENCE, restore_best_weights=True)
            callbacks.append(early_stop)

            if config.SAVE_MODEL and save_info:
                # Use new system path for each attribute model
                model_path = os.path.join(save_info["model_dir"], f"model_{attr}.keras")
                model_checkpoint = ModelCheckpoint(
                    filepath=model_path,
                    monitor=monitor_metric,
                    save_best_only=True,
                    verbose=1
                )
                callbacks.append(model_checkpoint)

            if X_val is not None and Y_val is not None:
                # Filter validation data as well
                val_valid_mask = ~np.isnan(Y_val[:, idx])
                X_val_filtered = X_val[val_valid_mask]
                y_val_attr = Y_val[val_valid_mask, idx]
                
                print(f"[{self.__class__.__name__}] Using {X_val_filtered.shape[0]} valid validation samples for '{attr}'")
                self.models[attr].fit(
                        X_train_filtered, y_train_attr,
                        validation_data=(X_val_filtered, y_val_attr),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=callbacks,
                        verbose=1
                    )
            else:
                self.models[attr].fit(
                    X_train_filtered, y_train_attr,
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=callbacks,
                    verbose=1
                )
        
        if config.SAVE_MODEL and save_info:
            print(f"[{self.__class__.__name__}] Training completed with new system hash: {save_info['config_hash']}")

    def test(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Test the model(s) based on the regression mode.
        """
        # Load checkpoint before testing to use the best model
        self._load_training_checkpoint()
        
        try:
            # Check if feature selection is enabled for custom testing
            has_feature_selection = hasattr(self, 'fs_interface') and self.fs_interface is not None
            
            
            if has_feature_selection:
                print(f"[{self.__class__.__name__}] Testing with feature selection")
                if config.MULTI_REGRESSION:
                    results = self._test_multi_regression_with_fs(X_test, Y_test)
                else:
                    results = self._test_single_prediction_with_fs(X_test, Y_test)
            else:
                print(f"[{self.__class__.__name__}] Testing without feature selection")
                if config.MULTI_REGRESSION:
                    results = self._test_multi_regression(X_test, Y_test)
                else:
                    results = self._test_single_prediction(X_test, Y_test)
            
            
            return results
        finally:
            # Cleanup checkpoint files after testing
            self.checkpoint_manager.cleanup()
            
    def _test_multi_regression(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Test the multi-output model and return evaluation metrics.
        """
        print(f"\n[{self.__class__.__name__}] Testing multi-regression model for all attributes")
        
        num_samples = X_test.shape[0]
        num_attrs = len(self.predicted_attributes)
        
        # Filter out samples where ANY target value is NaN (for multi-regression we need complete data)
        valid_mask = ~np.isnan(Y_test).any(axis=1)
        X_test_filtered = X_test[valid_mask]
        Y_test_filtered = Y_test[valid_mask]
        
        print(f"[{self.__class__.__name__}] Filtered out {num_samples - X_test_filtered.shape[0]} samples with any NaN values")
        print(f"[{self.__class__.__name__}] Testing on {X_test_filtered.shape[0]} valid samples")
        
        # Initialize predictions with NaN
        Y_pred = np.full((num_samples, num_attrs), np.nan)
        
        if X_test_filtered.shape[0] > 0:
            # Check if feature selection is enabled
            has_feature_selection = hasattr(self, 'fs_interface') and hasattr(self.fs_interface, 'is_enabled') and self.fs_interface.is_enabled()
            
            if has_feature_selection:
                # Apply feature selection during inference
                print(f"[{self.__class__.__name__}] Applying feature selection during inference")
                X_test_tensor = tf.constant(X_test_filtered, dtype=tf.float32)
                processed_data, selection_info = self.fs_interface.process_hyperspectral_data(
                    X_test_tensor, training=False
                )
                X_test_processed = processed_data.numpy()
                
                # Log feature selection results
                if hasattr(self.fs_interface, 'feature_selector') and self.fs_interface.feature_selector is not None:
                    selected_bands = self.fs_interface.get_selected_bands()
                    print(f"[{self.__class__.__name__}] Selected bands during inference: {selected_bands}")
                    
                    # Store actual selections for logging
                    self.actual_selected_bands = selected_bands
                    self.fs_selection_applied = True
            else:
                X_test_processed = X_test_filtered
            
            # Make predictions (scaled outputs)
            Y_pred_scaled = self.model.predict(X_test_processed)
            
            # Inverse transform to original scale
            Y_pred_original = self.target_scaler.inverse_transform(Y_pred_scaled)
            
            # Assign predictions back to full array
            Y_pred[valid_mask] = Y_pred_original
            
        metrics = self.evaluate(Y_test, Y_pred)
        return metrics
        
    def _test_single_prediction(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Test separate models for each quality attribute.
        """
        num_samples = X_test.shape[0]
        num_attrs = len(self.predicted_attributes)
        
        # Initialize with NaN values
        Y_pred = np.full((num_samples, num_attrs), np.nan)
        
        # Check if feature selection is enabled
        has_feature_selection = hasattr(self, 'fs_interface') and hasattr(self.fs_interface, 'is_enabled') and self.fs_interface.is_enabled()
        
        if has_feature_selection:
            print(f"[{self.__class__.__name__}] Applying feature selection during inference")

        for idx, attr in enumerate(self.predicted_attributes):
            print(f"[{self.__class__.__name__}] Predicting {attr}...")
            
            # Filter out samples where the target value is NaN
            valid_mask = ~np.isnan(Y_test[:, idx])
            X_test_filtered = X_test[valid_mask]
            print(f"[{self.__class__.__name__}] Testing on {X_test_filtered.shape[0]} valid samples for '{attr}'")
            
            if X_test_filtered.shape[0] > 0:
                if has_feature_selection:
                    # Apply feature selection during inference for this attribute
                    X_test_tensor = tf.constant(X_test_filtered, dtype=tf.float32)
                    processed_data, selection_info = self.fs_interface.process_hyperspectral_data(
                        X_test_tensor, training=False
                    )
                    X_test_processed = processed_data.numpy()
                    
                    # Log feature selection results (only once)
                    if idx == 0 and hasattr(self.fs_interface, 'feature_selector') and self.fs_interface.feature_selector is not None:
                        selected_bands = self.fs_interface.get_selected_bands()
                        print(f"[{self.__class__.__name__}] Selected bands during inference: {selected_bands}")
                        
                        # Store actual selections for logging
                        self.actual_selected_bands = selected_bands
                        self.fs_selection_applied = True
                else:
                    X_test_processed = X_test_filtered
                
                # Predict and assign to corresponding positions in Y_pred
                y_pred_filtered = self.models[attr].predict(X_test_processed).flatten()  # shape: (filtered_samples,)
                Y_pred[valid_mask, idx] = y_pred_filtered

        metrics = self.evaluate(Y_test, Y_pred)
        return metrics

    def _test_multi_regression_with_fs(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Test multi-regression model with feature selection and detailed logging.
        """
        print(f"[{self.__class__.__name__}] Testing multi-regression model with feature selection")
        
        # Filter out samples where ANY target value is NaN
        valid_mask = ~np.isnan(Y_test).any(axis=1)
        X_test_filtered = X_test[valid_mask]
        Y_test_filtered = Y_test[valid_mask]
        
        print(f"[{self.__class__.__name__}] Testing on {X_test_filtered.shape[0]} valid samples")
        
        if X_test_filtered.shape[0] == 0:
            print(f"[{self.__class__.__name__}] No valid test data")
            return {}
        
        # Transform targets using the fitted scaler
        Y_test_scaled = self.target_scaler.transform(Y_test_filtered)
        
        # Create test dataset
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test_filtered, Y_test_scaled))
        test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
        
        # Collect predictions and targets
        all_predictions = []
        all_targets = []
        test_step = 0
        
        for x_batch, y_batch in test_dataset:
            # Apply feature selection (inference mode)
            processed_data, selection_info = self.fs_interface.process_hyperspectral_data(
                x_batch, training=False, epoch=None
            )
            
            # 🎯 MEMORY LEAK FIX: Delete original tensor to free GPU memory
            del x_batch  # Free 204-band tensor, keep only 5-band processed_data
            
            
            # Model prediction
            predictions = self.model(processed_data, training=False)
            
            all_predictions.append(predictions.numpy())
            all_targets.append(y_batch.numpy())
            test_step += 1
        
        # Concatenate all predictions and targets
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Inverse transform to original scale
        all_predictions_original = self.target_scaler.inverse_transform(all_predictions)
        all_targets_original = self.target_scaler.inverse_transform(all_targets)
        
        # Calculate metrics for each attribute
        results = {}
        for i, attr in enumerate(self.predicted_attributes):
            y_true = all_targets_original[:, i]
            y_pred = all_predictions_original[:, i]
            
            # Filter out any remaining NaN values
            valid_indices = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[valid_indices]
            y_pred_clean = y_pred[valid_indices]
            
            if len(y_true_clean) > 0:
                from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                
                r2 = r2_score(y_true_clean, y_pred_clean)
                mse = mean_squared_error(y_true_clean, y_pred_clean)
                mae = mean_absolute_error(y_true_clean, y_pred_clean)
                rmse = np.sqrt(mse)
                
                results[attr] = {
                    'r2': r2,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'n_samples': len(y_true_clean)
                }
                
                print(f"[{self.__class__.__name__}] {attr}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
            else:
                print(f"[{self.__class__.__name__}] {attr}: No valid predictions")
                results[attr] = {'r2': 0.0, 'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf'), 'n_samples': 0}
        
        # Calculate average metrics
        if results:
            avg_r2 = np.mean([metrics['r2'] for metrics in results.values() if not np.isinf(metrics['r2'])])
            avg_rmse = np.mean([metrics['rmse'] for metrics in results.values() if not np.isinf(metrics['rmse'])])
            avg_mae = np.mean([metrics['mae'] for metrics in results.values() if not np.isinf(metrics['mae'])])
            
            results['average'] = {
                'r2': avg_r2,
                'rmse': avg_rmse,
                'mae': avg_mae,
                'n_attributes': len([r for r in results.values() if r['n_samples'] > 0])
            }
            
            print(f"[{self.__class__.__name__}] Average: R² = {avg_r2:.4f}, RMSE = {avg_rmse:.4f}")
        
        return results
    
    def _train_progressive_feature_selection(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        **kwargs
    ) -> None:
        """
        Train with progressive feature selection - building new models for each stage.
        
        This implements the progressive accumulation strategy where:
        1. Each stage trains a new model from scratch
        2. Feature sets accumulate across stages (bands -> bands+std -> bands+std+indexes -> all)
        3. Feature selectors are frozen after their stage completes
        4. No bias between stages as each gets a fresh model
        """
        print(f"\n[{self.__class__.__name__}] 🚀 PROGRESSIVE FEATURE SELECTION TRAINING")
        print(f"[{self.__class__.__name__}] Stages: {config.FEATURE_SELECTION_STAGE_RUN_ORDER}")
        
        
        # Log experiment start
        config_dict = {
            'MODEL_TYPE': self.__class__.__name__,
            'MULTI_REGRESSION': config.MULTI_REGRESSION,
            'USE_FEATURE_SELECTION': True,
            'PROGRESSIVE_MODE': True,
            'STAGE_RUN_ORDER': config.FEATURE_SELECTION_STAGE_RUN_ORDER,
            'epochs': epochs,
            'batch_size': batch_size,
            'predicted_attributes': self.predicted_attributes,
        }
        
        # Add all feature selection config
        for attr_name in dir(config):
            if attr_name.startswith('FEATURE_SELECTION'):
                config_dict[attr_name] = getattr(config, attr_name, None)
        
        model_info = {
            'model_name': self.model_name,
            'original_model_shape': self.original_model_shape,
            'progressive_stages': config.FEATURE_SELECTION_STAGE_RUN_ORDER
        }
        
        
        # Direct progressive training implementation
        from utils.model_factory import ModelFactory
        
        class DLModelStageExecutor:
            """Direct stage executor for deep learning models."""
            
            def __init__(self, parent_model, feature_selection_interface, train_data, val_data, output_dir, epochs):
                self.parent_model = parent_model
                self.fs_interface = feature_selection_interface
                self.train_data = train_data
                self.val_data = val_data
                self.output_dir = output_dir
                self.stage_epochs = epochs
                self.stage_results = {}
                
            def _get_previous_stage_model(self, current_stage_name: str) -> tf.keras.Model:
                """Get the model from the previous stage for weight transfer."""
                # Get the stage run order from config
                import config
                stage_order = config.FEATURE_SELECTION_STAGE_RUN_ORDER
                
                try:
                    current_index = stage_order.index(current_stage_name)
                    if current_index > 0:
                        previous_stage_name = stage_order[current_index - 1]
                        if previous_stage_name in self.stage_results:
                            return self.stage_results[previous_stage_name]['model']
                except (ValueError, IndexError):
                    pass
                
                return None
                
            def train_model(self, model: tf.keras.Model, data: Any, validation_data: Any, **kwargs) -> None:
                """Train a model for the current stage."""
                # Get current stage from fs_interface
                stage_name = self.fs_interface.active_stage
                stage_epochs = self.stage_epochs.get(stage_name, epochs)
                
                print(f"\n🚀 [Progressive FS] ===== STAGE {stage_name.upper()} =====\n")
                print(f"📋 [Progressive FS] Training for {stage_epochs} epochs")
                print(f"🧠 [Progressive FS] Model input channels: {model.input_shape[-1]}")
                print(f"⚙️  [Progressive FS] Model parameters: {model.count_params():,}")
                print(f"🎯 [Progressive FS] Starting training...\n")
                
                # Build and set the new model for this stage
                # Get the current number of input channels from the FS interface
                current_channels = self.fs_interface.get_current_input_channels()
                
                # Build a new model with the current channels
                print(f"[Progressive FS] Building model with {current_channels} input channels for stage '{stage_name}'")
                
                # Get previous model for weight transfer
                previous_model = self._get_previous_stage_model(stage_name)
                
                # Create a new model using the parent's build method but with adjusted channels
                stage_model = self.parent_model._build_progressive_stage_model(current_channels, stage_name, previous_model)
                
                # Replace the parent's model temporarily
                original_model = self.parent_model.model
                self.parent_model.model = stage_model
                
                try:
                    # Train the stage model directly with feature selection
                    self._train_stage_with_feature_selection(
                        stage_model=stage_model,
                        X_train=data[0], 
                        Y_train=data[1],
                        X_val=validation_data[0] if validation_data else None,
                        Y_val=validation_data[1] if validation_data else None,
                        epochs=stage_epochs,
                        batch_size=batch_size,
                        stage_name=stage_name,
                        experiment_name=f"{experiment_name}_stage_{stage_name}"
                    )
                    
                    # Store stage results
                    self.stage_results[stage_name] = {
                        'model': stage_model,
                        'training_complete': True,
                        'final_epoch': stage_epochs
                    }
                    
                finally:
                    # Restore original model
                    self.parent_model.model = original_model
            
            def run_stages(self):
                """Run all progressive feature selection stages."""
                import config
                stages = config.FEATURE_SELECTION_STAGE_RUN_ORDER
                
                print(f"\n[Progressive FS] Running {len(stages)} stages: {stages}")
                
                for stage_name in stages:
                    print(f"\n[Progressive FS] Setting up stage: {stage_name}")
                    
                    # Set the active stage in the feature selection interface
                    self.fs_interface.set_stage(stage_name)
                    
                    # Train model for this stage
                    self.train_model(
                        model=self.parent_model.model,
                        data=self.train_data,
                        validation_data=self.val_data
                    )
                    
                    print(f"[Progressive FS] Completed stage: {stage_name}")
                
                print(f"\n[Progressive FS] All stages completed successfully!")
            
            def _train_stage_with_feature_selection(
                self,
                stage_model: tf.keras.Model,
                X_train: Union[np.ndarray, np.ndarray],  # Can be mask or data
                Y_train: np.ndarray,
                X_val: Optional[Union[np.ndarray, np.ndarray]] = None,  # Can be mask or data
                Y_val: Optional[np.ndarray] = None,
                epochs: int = 100,
                batch_size: int = 32,
                stage_name: str = "",
                experiment_name: str = ""
            ) -> None:
                """Train a stage model with feature selection enabled."""
                print(f"\n[Stage Training] Training {stage_name} with feature selection")
                
                # Check if X_train is a mask (boolean array) - if so, we need to load data for this stage
                if X_train.dtype == bool:
                    print(f"[Stage Training] Loading data for stage {stage_name}...")
                    # Get tomatoes from parent model's stored reference
                    original_X = self.parent_model._original_X_train
                    original_Y = self.parent_model._original_Y_train
                    
                    # Prepare data for this specific stage
                    from utils.memory_optimized_data_preparation import MemoryOptimizedDataPreparation
                    
                    # Get selected bands from previous stages if available
                    if hasattr(self.fs_interface, '_preserved_selections') and 'band_indices' in self.fs_interface._preserved_selections:
                        selected_indices = self.fs_interface._preserved_selections['band_indices']
                        print(f"[Stage Training] Using {len(selected_indices)} selected bands from previous stages")
                    else:
                        # For bands stage, we need full spectrum
                        if stage_name == 'bands':
                            selected_indices = None
                        else:
                            selected_indices = list(range(204))
                    
                    # Extract data for this stage
                    if selected_indices is None:
                        # Full spectrum for band selection
                        X_train_stage = original_X[X_train]  # Use mask to filter
                    else:
                        # Selected bands only
                        X_train_stage = original_X[X_train][:, :, :, selected_indices]
                    
                    # Note: Stage-specific feature computation will be handled by the feature selection interface
                    # during processing. We pass the original 204-band data for proper band selection.
                    
                    # Now use the filtered data
                    X_train_filtered = X_train_stage
                    Y_train_filtered = Y_train  # Already filtered
                    
                    # Clean up stage data after use
                    del X_train_stage
                else:
                    # X_train is already data
                    X_train_filtered = X_train
                    Y_train_filtered = Y_train
                
                if X_train_filtered.shape[0] == 0:
                    print(f"[Stage Training] No valid training data for {stage_name}")
                    return
                
                # Fit target scaler and transform targets
                Y_train_scaled = self.parent_model.target_scaler.fit_transform(Y_train_filtered)
                
                # Free memory consumed by the unscaled targets now that we have the scaled version
                del Y_train_filtered
                gc.collect()
                
                # Prepare validation data if provided
                X_val_filtered, Y_val_scaled = None, None
                if X_val is not None and Y_val is not None:
                    # Check if X_val is a mask (boolean array) - if so, we need to load data for this stage
                    if X_val.dtype == bool:
                        print(f"[Stage Training] Loading validation data for stage {stage_name}...")
                        # Get original validation data from parent model's stored reference
                        original_X_val = self.parent_model._original_X_val
                        original_Y_val = self.parent_model._original_Y_val
                        
                        # Get selected bands from previous stages if available
                        if hasattr(self.fs_interface, '_preserved_selections') and 'band_indices' in self.fs_interface._preserved_selections:
                            selected_indices = self.fs_interface._preserved_selections['band_indices']
                        else:
                            # For bands stage, we need full spectrum
                            if stage_name == 'bands':
                                selected_indices = None
                            else:
                                selected_indices = list(range(204))
                        
                        # Extract validation data for this stage
                        if selected_indices is None:
                            # Full spectrum for band selection
                            X_val_stage = original_X_val[X_val]  # Use mask to filter
                        else:
                            # Selected bands only
                            X_val_stage = original_X_val[X_val][:, :, :, selected_indices]
                        
                        # Note: Stage-specific feature computation will be handled by the feature selection interface
                        # during processing. We pass the original 204-band data for proper band selection.
                        
                        # Now use the filtered data
                        X_val_filtered = X_val_stage
                        Y_val_filtered = Y_val  # Already filtered
                        
                        # Clean up stage data after use
                        del X_val_stage
                    else:
                        # X_val is already data
                        X_val_filtered = X_val
                        Y_val_filtered = Y_val
                    
                    # Scale targets
                    Y_val_scaled = self.parent_model.target_scaler.transform(Y_val_filtered)
                    del Y_val_filtered
                    gc.collect()
                
                # Setup optimizer
                initial_lr = 0.001
                lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=initial_lr,
                    decay_steps=100,
                    decay_rate=0.96,
                    staircase=True
                )
                optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
                
                # Create datasets
                train_dataset = tf.data.Dataset.from_tensor_slices((X_train_filtered, Y_train_scaled))
                train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                
                if X_val_filtered is not None:
                    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_filtered, Y_val_scaled))
                    val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
                else:
                    val_dataset = None
                
                # Training metrics
                train_loss_metric = tf.keras.metrics.Mean()
                val_loss_metric = tf.keras.metrics.Mean()
                
                # Training loop
                best_val_loss = float('inf')
                patience_counter = 0
                patience = float('inf')  # Disable global patience during feature selection
                
                for epoch in range(epochs):
                    # Always show progress for feature selection training
                    print(f"  🎯 Stage {stage_name.upper()} - Epoch {epoch + 1}/{epochs}")
                    
                    # Reset metrics
                    train_loss_metric.reset_state()
                    val_loss_metric.reset_state()
                    
                    # Training step
                    for step, (x_batch, y_batch) in enumerate(train_dataset):
                        with tf.GradientTape() as tape:
                            # Apply feature selection
                            processed_data, selection_info = self.fs_interface.process_hyperspectral_data(
                                x_batch, training=True, epoch=epoch
                            )
                            
                            # Delete original tensor to free GPU memory
                            del x_batch
                            
                            # Model prediction
                            predictions = stage_model(processed_data, training=True)
                            
                            # Compute prediction loss
                            prediction_loss = tf.reduce_mean(tf.keras.losses.mse(y_batch, predictions))
                            
                            # Get feature selection loss
                            fs_loss, loss_breakdown = self.fs_interface.get_prediction_aware_feature_selection_loss(
                                selection_info, prediction_loss, epoch=epoch
                            )
                            
                            # Get strategy weights
                            strategy_weights = self.fs_interface.get_strategy_loss_weights(epoch=epoch)
                            fs_weight = strategy_weights['feature_selection_weight']
                            pred_weight = strategy_weights['prediction_weight']
                            
                            # Total loss
                            total_loss = fs_weight * fs_loss + pred_weight * prediction_loss
                        
                        # Apply gradients
                        all_trainable_vars = stage_model.trainable_variables
                        if hasattr(self.fs_interface, 'get_trainable_variables'):
                            fs_vars = self.fs_interface.get_trainable_variables()
                            if fs_vars:
                                all_trainable_vars = all_trainable_vars + fs_vars
                        
                        gradients = tape.gradient(total_loss, all_trainable_vars)
                        optimizer.apply_gradients(zip(gradients, all_trainable_vars))
                        
                        train_loss_metric.update_state(total_loss)
                    
                    # Validation step
                    if val_dataset is not None:
                        for x_batch, y_batch in val_dataset:
                            # Apply feature selection
                            processed_data, selection_info = self.fs_interface.process_hyperspectral_data(
                                x_batch, training=False, epoch=epoch
                            )
                            
                            # Model prediction
                            predictions = stage_model(processed_data, training=False)
                            
                            # Compute validation loss
                            val_prediction_loss = tf.reduce_mean(tf.keras.losses.mse(y_batch, predictions))
                            
                            # Get feature selection loss for validation
                            val_fs_loss, _ = self.fs_interface.get_prediction_aware_feature_selection_loss(
                                selection_info, val_prediction_loss, epoch=epoch
                            )
                            
                            # Apply strategy weights
                            val_total_loss = fs_weight * val_fs_loss + pred_weight * val_prediction_loss
                            
                            val_loss_metric.update_state(val_total_loss)
                    
                    # Print epoch results (always show for feature selection)
                    train_loss = train_loss_metric.result()
                    print(f"    📊 Train Loss: {train_loss:.4f}")
                    
                    if val_dataset is not None:
                        val_loss = val_loss_metric.result()
                        print(f"    ✅ Val Loss: {val_loss:.4f}")
                        
                        # Check for best model
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            self.parent_model._save_best_model_checkpoint(epoch, val_loss)
                        else:
                            patience_counter += 1
                
                print(f"[Stage Training] Completed training for stage {stage_name}")
        
        # Store original data references for staged processing
        self._original_X_train = X_train
        self._original_Y_train = Y_train
        self._original_X_val = X_val
        self._original_Y_val = Y_val
        
        # Prepare only Y data initially (much smaller)
        # Filter out samples where ANY target value is NaN
        valid_mask = ~np.isnan(Y_train).any(axis=1)
        Y_train_filtered = Y_train[valid_mask]
        
        print(f"[{self.__class__.__name__}] Training on {valid_mask.sum()} valid samples")
        
        if valid_mask.sum() == 0:
            print(f"[{self.__class__.__name__}] No valid training data, skipping training.")
            return
        
        # Initialize target scaler
        if not hasattr(self, 'target_scaler') or self.target_scaler is None:
            from utils.target_scaling import TargetScaler
            self.target_scaler = TargetScaler(self.predicted_attributes)
        
        # Prepare validation Y data if provided
        Y_val_filtered = None
        val_valid_mask = None
        if Y_val is not None:
            val_valid_mask = ~np.isnan(Y_val).any(axis=1)
            Y_val_filtered = Y_val[val_valid_mask]
            print(f"[{self.__class__.__name__}] Using {val_valid_mask.sum()} valid validation samples")
        
        # Create stage executor with masks instead of full data
        experiment_name = kwargs.get('experiment_name', f"{self.__class__.__name__}_Progressive_FS")
        output_dir = os.path.join('outputs', 'progressive_fs', experiment_name)
        stage_executor = DLModelStageExecutor(
            parent_model=self,
            feature_selection_interface=self.fs_interface,
            train_data=(valid_mask, Y_train_filtered),  # Pass mask instead of X data
            val_data=(val_valid_mask, Y_val_filtered) if Y_val_filtered is not None else None,
            output_dir=output_dir,
            epochs=config.FEATURE_SELECTION_STAGE_EPOCHS
        )
        
        # Run all stages
        stage_executor.run_stages()
        
        # After all stages, keep the final model (from 'finetune' stage)
        final_stage = config.FEATURE_SELECTION_STAGE_RUN_ORDER[-1]
        if final_stage in stage_executor.stage_results:
            self.model = stage_executor.stage_results[final_stage]['model']
            print(f"\n[{self.__class__.__name__}] Progressive training complete - using model from '{final_stage}' stage")
        else:
            print(f"\n[{self.__class__.__name__}] Warning: Final stage '{final_stage}' not found in results")
        
        # Log final summary
        final_summary = {
            'stages_completed': list(stage_executor.stage_results.keys()),
            'total_stages': len(config.FEATURE_SELECTION_STAGE_RUN_ORDER),
            'final_model_params': self.model.count_params() if self.model else 0,
            'progressive_training_complete': True
        }
        
        
        # Clean up GPU memory after progressive training
        try:
            from utils.experiments.gpu.tensorflow_gpu_setup import cleanup_tensorflow_gpu
            cleanup_tensorflow_gpu()
        except ImportError:
            pass
    
    def _test_single_prediction_with_fs(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Test single prediction models with feature selection and detailed logging.
        """
        print(f"[{self.__class__.__name__}] Testing individual models with feature selection")
        
        results = {}
        
        for idx, attr in enumerate(self.predicted_attributes):
            print(f"\n[{self.__class__.__name__}] Testing {attr} model...")
            
            # Get target values for this attribute
            y_test_attr = Y_test[:, idx]
            
            # Filter out NaN values for this attribute
            valid_mask = ~np.isnan(y_test_attr)
            X_test_filtered = X_test[valid_mask]
            y_test_filtered = y_test_attr[valid_mask]
            
            print(f"[{self.__class__.__name__}] {attr}: Testing on {X_test_filtered.shape[0]} valid samples")
            
            if X_test_filtered.shape[0] == 0:
                print(f"[{self.__class__.__name__}] {attr}: No valid test data")
                results[attr] = {'r2': 0.0, 'mse': float('inf'), 'mae': float('inf'), 'rmse': float('inf'), 'n_samples': 0}
                continue
            
            # Scale targets
            y_test_scaled = self.target_scalers[attr].transform(y_test_filtered.reshape(-1, 1)).flatten()
            
            # Create test dataset
            test_dataset = tf.data.Dataset.from_tensor_slices((X_test_filtered, y_test_scaled))
            test_dataset = test_dataset.batch(32).prefetch(tf.data.AUTOTUNE)
            
            # Collect predictions
            all_predictions = []
            all_targets = []
            
            for x_batch, y_batch in test_dataset:
                # Apply feature selection (inference mode)
                processed_data, selection_info = self.fs_interface.process_hyperspectral_data(
                    x_batch, training=False, epoch=None
                )
                
                # 🎯 MEMORY LEAK FIX: Delete original tensor to free GPU memory
                del x_batch  # Free 204-band tensor, keep only 5-band processed_data
                
                # Model prediction for this attribute
                predictions = self.models[attr](processed_data, training=False)
                predictions = tf.squeeze(predictions, -1)  # Remove last dimension if needed
                
                all_predictions.append(predictions.numpy())
                all_targets.append(y_batch.numpy())
            
            # Concatenate predictions
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            
            # Inverse transform to original scale
            all_predictions_original = self.target_scalers[attr].inverse_transform(all_predictions.reshape(-1, 1)).flatten()
            all_targets_original = self.target_scalers[attr].inverse_transform(all_targets.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
            
            r2 = r2_score(all_targets_original, all_predictions_original)
            mse = mean_squared_error(all_targets_original, all_predictions_original)
            mae = mean_absolute_error(all_targets_original, all_predictions_original)
            rmse = np.sqrt(mse)
            
            results[attr] = {
                'r2': r2,
                'mse': mse,
                'mae': mae,
                'rmse': rmse,
                'n_samples': len(all_targets_original)
            }
            
            print(f"[{self.__class__.__name__}] {attr}: R² = {r2:.4f}, RMSE = {rmse:.4f}")
        
        # Calculate average metrics
        if results:
            avg_r2 = np.mean([metrics['r2'] for metrics in results.values() if not np.isinf(metrics['r2'])])
            avg_rmse = np.mean([metrics['rmse'] for metrics in results.values() if not np.isinf(metrics['rmse'])])
            avg_mae = np.mean([metrics['mae'] for metrics in results.values() if not np.isinf(metrics['mae'])])
            
            results['average'] = {
                'r2': avg_r2,
                'rmse': avg_rmse,
                'mae': avg_mae,
                'n_attributes': len([r for r in results.values() if r['n_samples'] > 0])
            }
            
            print(f"[{self.__class__.__name__}] Average: R² = {avg_r2:.4f}, RMSE = {avg_rmse:.4f}")
        
        return results

    def save_model(self) -> None:
        """
        Save the model(s) using the new ModelConfigManager system when config.SAVE_MODEL is True.
        """
        if not config.SAVE_MODEL:
            print(f"[{self.__class__.__name__}] config.SAVE_MODEL is False. Skipping save_model().")
            return
        
        # Check if saving was already done during training
        if hasattr(self, '_save_info'):
            print(f"[{self.__class__.__name__}] Model already saved during training with hash: {self._save_info['config_hash']}")
            return
            
        # Use new ModelConfigManager system
        config_manager = ModelConfigManager()
        
        # Get NDSI band pairs from config
        ndsi_band_pairs = getattr(config, 'NDSI_BAND_PAIRS', [])
        
        # Generate configuration and create directories
        save_info = config_manager.save_model_complete(
            model_instance=self,
            model_type=self.model_type,
            model_shape=self.model_shape,
            components=self.components,
            component_dimensions=self.component_dimensions,
            selected_bands=self.selected_bands,
            selected_indexes=self.selected_indexes,
            predicted_attributes=self.predicted_attributes,
            ndsi_band_pairs=ndsi_band_pairs
        )
        
        # Save the actual model(s) to the generated path
        if config.MULTI_REGRESSION:
            self._save_multi_regression_model_new(save_info["model_file"])
        else:
            self._save_single_prediction_models_new(save_info["model_dir"])
            
        print(f"[{self.__class__.__name__}] Model saved successfully with hash: {save_info['config_hash']}")
        
        # Clean up memory after model saving
        gc.collect()
        
        # Clean up GPU memory after model saving
        try:
            from utils.experiments.gpu.tensorflow_gpu_setup import cleanup_tensorflow_gpu
            cleanup_tensorflow_gpu()
        except ImportError:
            pass
            
    def _save_multi_regression_model_new(self, model_file_path: str) -> None:
        """
        Save the multi-output model and target scaler using new system.
        """
        import joblib
        
        # Save the main model
        self.model.save(model_file_path)
        print(f"[{self.__class__.__name__}] Saved multi-regression model to {model_file_path}")
        
        # Save the target scaler in the same directory
        model_dir = os.path.dirname(model_file_path)
        scaler_path = os.path.join(model_dir, "target_scaler.json")
        
        # Serialize TargetScaler which contains MinMaxScaler objects
        scaler_data = {
            'attributes': self.target_scaler.attributes,
            'is_fitted': self.target_scaler.is_fitted,
            'scalers': {}
        }
        
        # Serialize each MinMaxScaler
        for attr, scaler in self.target_scaler.scalers.items():
            scaler_data['scalers'][attr] = {
                'data_min_': scaler.data_min_.tolist(),
                'data_max_': scaler.data_max_.tolist(),
                'data_range_': scaler.data_range_.tolist(),
                'scale_': scaler.scale_.tolist(),
                'min_': scaler.min_.tolist(),
                'n_samples_seen_': int(scaler.n_samples_seen_),
                'feature_range': scaler.feature_range
            }
        
        with open(scaler_path, 'w') as f:
            json.dump(scaler_data, f)
        print(f"[{self.__class__.__name__}] Saved target scaler to {scaler_path}")
        
    def _save_single_prediction_models_new(self, model_dir: str) -> None:
        """
        Save each attribute's model individually using new system.
        """
        for attr in self.predicted_attributes:
            model_path = os.path.join(model_dir, f"model_{attr}.keras")
            self.models[attr].save(model_path)
            print(f"[{self.__class__.__name__}] Saved model for {attr} to {model_path}")

    def load_model(self) -> None:
        """
        Load saved model(s) based on regression mode and model configuration.
        """
        if config.USE_PREVIOUS_MODEL:
            if config.MULTI_REGRESSION:
                self._load_multi_regression_model()
            else:
                self._load_single_prediction_models()
        else:
            print(f"[{self.__class__.__name__}] USE_PREVIOUS_MODEL is False, skipping model loading")

    def calculate_feature_importance(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        methods: List[str] = ["permutation", "gradient"],
        n_repeats: int = 5,  # Number of permutation repetitions (fewer for DL due to computational cost)
        random_state: int = 42,
        save_results: bool = True,
        save_plots: bool = True,
        plot_top_n: Optional[int] = 20,  # None or -1 to show all channels
        print_top_n: int = 5,  # Number of top channels to print in console
        results_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate feature importance for DL models.
        
        Args:
            X_test: Test feature matrix (H, W, C)
            Y_test: Test target matrix
            methods: List of importance methods to use
            n_repeats: Number of permutation repetitions for stable estimates (NOT number of channels)
            random_state: Random state for reproducibility
            save_results: Whether to save results to file
            save_plots: Whether to save importance plots
            plot_top_n: Number of top channels to plot (None or -1 for all channels)
            print_top_n: Number of top channels to print in console
            results_dir: Directory to save results (auto-generated if None)
            
        Returns:
            Dict containing importance results for each attribute and method
        """
        print(f"\n[{self.__class__.__name__}] Calculating feature importance...")
        print(f"[{self.__class__.__name__}] n_repeats={n_repeats} (permutation repetitions for stability)")
        print(f"[{self.__class__.__name__}] Total channels: {X_test.shape[3]}")
        
        # Generate channel names for interpretability
        channel_names = self._generate_dl_feature_names()
        
        # Initialize feature importance analyzer
        analyzer = FeatureImportanceAnalyzer(
            model_instance=self,
            model_type=self.model_type,
            feature_names=channel_names
        )
        
        # Calculate importance
        importance_results = analyzer.calculate_dl_importance(
            X_test=X_test,
            Y_test=Y_test,
            methods=methods,
            n_repeats=n_repeats,
            random_state=random_state
        )
        
        # Set up results directory
        if results_dir is None:
            results_dir = os.path.join("results", "feature_importance", self.__class__.__name__)
        
        # Save results and plots
        if save_results or save_plots:
            os.makedirs(results_dir, exist_ok=True)
            
        if save_results:
            results_path = os.path.join(results_dir, "importance_results.json")
            analyzer.save_importance_results(results_path)
            
            # Save summary DataFrame
            summary_df = analyzer.create_importance_summary()
            summary_path = os.path.join(results_dir, "importance_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"[{self.__class__.__name__}] Saved importance summary to {summary_path}")
        
        if save_plots:
            analyzer.plot_importance(save_dir=results_dir, top_n=plot_top_n)
        
        # Print top channels
        self._print_top_channels(analyzer, importance_results, print_top_n)
        
        # Clean up memory after feature importance calculation
        gc.collect()
        
        # Clean up GPU memory after feature importance calculation
        try:
            from utils.experiments.gpu.tensorflow_gpu_setup import cleanup_tensorflow_gpu
            cleanup_tensorflow_gpu()
        except ImportError:
            pass
        
        return importance_results
    
    def _generate_dl_feature_names(self) -> List[str]:
        """Generate meaningful channel names for DL models."""
        channel_names = []
        current_channel = 0
        
        # Process each component type
        for component_name in ['reflectance', 'std', 'ndsi', 'indexes']:
            if self.components.get(component_name, False):
                num_channels = self.component_dimensions.get(component_name, 0)
                
                if component_name == 'reflectance':
                    for i, band in enumerate(self.selected_bands[:num_channels]):
                        channel_names.append(f"Reflectance_Band_{band}")
                        
                elif component_name == 'std':
                    for i, band in enumerate(self.selected_bands[:num_channels]):
                        channel_names.append(f"STD_Band_{band}")
                        
                elif component_name == 'ndsi':
                    ndsi_pairs = getattr(config, 'NDSI_BAND_PAIRS', [])
                    for i, (band_i, band_j) in enumerate(ndsi_pairs[:num_channels]):
                        channel_names.append(f"NDSI_{band_i}_{band_j}")
                        
                elif component_name == 'indexes':
                    for i, index in enumerate(self.selected_indexes[:num_channels]):
                        if hasattr(index, 'value'):
                            channel_names.append(f"Index_{index.value}")
                        else:
                            channel_names.append(f"Index_{index}")
                
                current_channel += num_channels
        
        # Fill any remaining channels with generic names
        total_channels = self.model_shape[2]
        while len(channel_names) < total_channels:
            channel_names.append(f"Channel_{len(channel_names)}")
        
        return channel_names
    
    def _print_top_channels(
        self, 
        analyzer: FeatureImportanceAnalyzer, 
        importance_results: Dict[str, Dict[str, np.ndarray]],
        print_top_n: int = 5
    ) -> None:
        """Print top channels for each attribute and method."""
        print(f"\n[{self.__class__.__name__}] Top {print_top_n} Most Important Channels:")
        print("=" * 70)
        
        for method in ["permutation", "gradient"]:
            top_features = analyzer.get_top_features(method=method, top_n=print_top_n)
            if top_features:
                print(f"\n{method.upper()} IMPORTANCE:")
                print("-" * 30)
                
                for attr, features in top_features.items():
                    print(f"\n{attr}:")
                    for rank, (channel_idx, importance) in enumerate(features, 1):
                        channel_name = analyzer.feature_names[channel_idx] if channel_idx < len(analyzer.feature_names) else f"Channel_{channel_idx}"
                        print(f"  {rank}. {channel_name}: {importance:.4f}")
        
        print("\n" + "=" * 70)

    def _load_multi_regression_model(self) -> None:
        """
        Load the multi-output model and target scaler from the organized system.
        """
        import json
        from sklearn.preprocessing import StandardScaler
        
        # Check if model exists in organized system
        if self._check_if_model_exists():
            from utils.model_config_manager import ModelConfigManager
            
            config_manager = ModelConfigManager()
            ndsi_band_pairs = getattr(config, 'NDSI_BAND_PAIRS', [])
            
            save_info = config_manager.save_model_complete(
                model_instance=self,
                model_type=self.model_type,
                model_shape=self.model_shape,
                components=self.components,
                component_dimensions=self.component_dimensions,
                selected_bands=self.selected_bands,
                selected_indexes=self.selected_indexes,
                predicted_attributes=self.predicted_attributes,
                ndsi_band_pairs=ndsi_band_pairs
            )
            
            model_file = save_info["model_file"]
            scaler_file = os.path.join(save_info["model_dir"], "target_scaler.json")
            
            if os.path.exists(model_file) and os.path.exists(scaler_file):
                print(f"[{self.__class__.__name__}] Loading existing multi-regression model from: {model_file}")
                print(f"[{self.__class__.__name__}] Loading target scaler from: {scaler_file}")
                self.model = keras_load_model(model_file, safe_mode=False)
                
                with open(scaler_file, 'r') as f:
                    scaler_data = json.load(f)
                
                # Reconstruct TargetScaler with MinMaxScaler objects
                from utils.target_scaling import TargetScaler
                from sklearn.preprocessing import MinMaxScaler
                
                self.target_scaler = TargetScaler(scaler_data['attributes'])
                self.target_scaler.is_fitted = scaler_data['is_fitted']
                self.target_scaler.scalers = {}
                
                # Reconstruct each MinMaxScaler
                for attr, scaler_info in scaler_data['scalers'].items():
                    scaler = MinMaxScaler(feature_range=tuple(scaler_info['feature_range']))
                    scaler.data_min_ = np.array(scaler_info['data_min_'])
                    scaler.data_max_ = np.array(scaler_info['data_max_'])
                    scaler.data_range_ = np.array(scaler_info['data_range_'])
                    scaler.scale_ = np.array(scaler_info['scale_'])
                    scaler.min_ = np.array(scaler_info['min_'])
                    scaler.n_samples_seen_ = scaler_info['n_samples_seen_']
                    self.target_scaler.scalers[attr] = scaler
                return
        
        print(f"[{self.__class__.__name__}] No pre-trained multi-regression model found in organized system.")
            
    def _load_single_prediction_models(self) -> None:
        """
        Load each attribute's model individually from the organized system.
        """
        for attr in self.predicted_attributes:
            if self._check_if_model_exists(attr):
                try:
                    self.models[attr] = self._load_individual_model(attr)
                    print(f"[{self.__class__.__name__}] Successfully loaded existing model for '{attr}'.")
                except Exception as e:
                    print(f"[{self.__class__.__name__}] Failed to load model for '{attr}': {e}")
            else:
                print(f"[{self.__class__.__name__}] No pre-trained model found for '{attr}' in organized system.")

    def _build_progressive_stage_model(self, stage_channels: int, stage_name: str, previous_model: tf.keras.Model = None) -> tf.keras.Model:
        """
        Build a new model for progressive feature selection with the given input channels.
        
        If the framework supports layer-wise weight copying, slice the existing weights
        to the first `stage_channels` before assigning them to the new model; otherwise,
        rebuild from scratch for each stage.
        
        Args:
            stage_channels: Number of input channels for this stage
            stage_name: Name of the current stage (bands, std, indexes, finetune)
            previous_model: Previous stage model to transfer weights from (optional)
            
        Returns:
            A new Keras model configured for the current stage
        """
        from utils.model_factory import ModelFactory
        
        print(f"[{self.__class__.__name__}] Building progressive stage model for '{stage_name}' with {stage_channels} channels")
        
        # Get stage-specific hyperparameters
        hyperparameters = ModelFactory.get_stage_hyperparameters(stage_name)
        
        # Build model using ModelFactory
        stage_model = ModelFactory.build_model(
            input_channels=stage_channels,
            stage_name=stage_name,
            model_type=self.model_type,
            hyperparameters=hyperparameters
        )
        
        print(f"[{self.__class__.__name__}] Built model with {stage_model.count_params():,} parameters")
        
        # Handle weight transfer between stages
        if previous_model is not None:
            # Initialize weight transfer manager if not exists
            if not hasattr(self, 'weight_transfer_manager'):
                self.weight_transfer_manager = WeightTransferManager(enable_weight_transfer=config.FEATURE_SELECTION_ENABLE_WEIGHT_TRANSFER)
            
            # Attempt to transfer weights from previous model
            transfer_success = self.weight_transfer_manager.transfer_weights(
                source_model=previous_model,
                target_model=stage_model,
                stage_channels=stage_channels,
                stage_name=stage_name
            )
            
            if transfer_success:
                print(f"[{self.__class__.__name__}] Successfully transferred weights from previous stage to '{stage_name}'")
            else:
                print(f"[{self.__class__.__name__}] Could not transfer weights, using fresh model for '{stage_name}'")
                
            # Log transfer statistics
            stats = self.weight_transfer_manager.get_transfer_stats(stage_name)
            if stats:
                print(f"[{self.__class__.__name__}] Transfer stats for '{stage_name}': {stats}")
        else:
            print(f"[{self.__class__.__name__}] No previous model available for weight transfer to '{stage_name}'")
        
        # Clean up memory after building progressive stage model
        gc.collect()
        
        # Clean up GPU memory after building progressive stage model
        try:
            from utils.experiments.gpu.tensorflow_gpu_setup import cleanup_tensorflow_gpu
            cleanup_tensorflow_gpu()
        except ImportError:
            pass
        
        return stage_model

    def _capture_feature_selection_results(self, selection_info, epoch=0):
        """
        Capture and store actual feature selection results for logging.
        This should be called during training to record what was actually selected.
        """
        if not hasattr(self, 'fs_interface') or not self.fs_interface.is_enabled():
            return
        
        # Only capture once (during first epoch) to avoid overwriting
        if hasattr(self, 'fs_selection_applied') and self.fs_selection_applied:
            return
            
        # Only print capture info once to avoid spam
        if not hasattr(self, '_fs_capture_printed'):
            self._fs_capture_printed = True
            print(f"[FS] 🎯 CAPTURING ACTUAL SELECTIONS (Epoch {epoch}):")
        
        # Extract band indices
        if 'selected_band_indices' in selection_info:
            actual_selected_bands = selection_info['selected_band_indices']
            if hasattr(actual_selected_bands, 'numpy'):
                actual_selected_bands = actual_selected_bands.numpy()
            
            # Extract first batch from 2D tensor: (batch_size, k_bands) -> (k_bands,)
            if actual_selected_bands.ndim == 2:
                actual_selected_bands = actual_selected_bands[0]
            
            # Convert to list
            actual_selected_bands = actual_selected_bands.tolist()
            
            # For full spectrum mode, these are the actual band indices (0-203)
            if getattr(config, 'FEATURE_SELECTION_FULL_SPECTRUM', True):
                self.actual_selected_bands = actual_selected_bands
            else:
                # For constrained mode, map back to original band indices
                original_bands = self.original_selected_bands
                self.actual_selected_bands = [original_bands[i] for i in actual_selected_bands if i < len(original_bands)]
        else:
            self.actual_selected_bands = []
        
        # Extract component information from selection_info
        component_info = selection_info.get('component_info', {})
        
        # STD bands (use same as reflectance bands since STD is computed from reflectance)
        if self.components.get('std', False) and self.actual_selected_bands:
            # STD uses the same bands as reflectance, limited by b_std parameter
            b_std = getattr(config, 'FEATURE_SELECTION_B_STD', len(self.actual_selected_bands))
            self.actual_std_bands = self.actual_selected_bands[:min(b_std, len(self.actual_selected_bands))]
        else:
            self.actual_std_bands = []
        
        # NDSI pairs (feature selection generates its own optimized pairs)
        if self.components.get('ndsi', False) and component_info.get('ndsi_channels', 0) > 0:
            # Feature selection creates optimized NDSI pairs from selected bands
            # We'll indicate this with a special notation since exact pairs aren't directly available
            a_ndsi = component_info.get('ndsi_channels', 0)
            if self.actual_selected_bands and len(self.actual_selected_bands) >= 2:
                # Generate pairs from top selected bands (simplified representation)
                pairs = []
                for i in range(min(a_ndsi, len(self.actual_selected_bands) // 2)):
                    if i * 2 + 1 < len(self.actual_selected_bands):
                        pairs.append((self.actual_selected_bands[i * 2], self.actual_selected_bands[i * 2 + 1]))
                self.actual_ndsi_pairs = pairs
            else:
                self.actual_ndsi_pairs = []
        else:
            self.actual_ndsi_pairs = []
        
        # Index names (feature selection may use hybrid strategy)
        if self.components.get('indexes', False) and component_info.get('indexes_channels', 0) > 0:
            index_strategy = getattr(config, 'FEATURE_SELECTION_INDEX_STRATEGY', 'hybrid')
            c_indexes = component_info.get('indexes_channels', 0)
            
            if index_strategy == "existing":
                # Uses existing indexes from config
                self.actual_index_names = self.selected_indexes[:c_indexes] if self.selected_indexes else []
            elif index_strategy == "learned":
                # Uses learned combinations
                self.actual_index_names = [f"FS_Learned_{i+1}" for i in range(c_indexes)]
            else:  # hybrid
                # Mix of existing and learned
                existing_count = min(len(self.selected_indexes) if self.selected_indexes else 0, c_indexes // 2)
                learned_count = c_indexes - existing_count
                
                actual_indexes = []
                if self.selected_indexes:
                    actual_indexes.extend(self.selected_indexes[:existing_count])
                actual_indexes.extend([f"FS_Learned_{i+1}" for i in range(learned_count)])
                self.actual_index_names = actual_indexes
        else:
            self.actual_index_names = []
        
        # Mark that feature selection was applied
        self.fs_selection_applied = True
        
        # Only print selection results once to avoid spam
        if not hasattr(self, '_fs_results_printed'):
            self._fs_results_printed = True
            print(f"[FS]   Reflectance bands: {self.actual_selected_bands}")
            print(f"[FS]   STD bands: {len(self.actual_std_bands)} bands")
            print(f"[FS]   NDSI pairs: {len(self.actual_ndsi_pairs)} pairs")
            print(f"[FS]   Index names: {len(self.actual_index_names)} indexes")
        
        # Clean up memory after capturing feature selection results
        gc.collect()
        
        # Clean up GPU memory after capturing feature selection results
        try:
            from utils.experiments.gpu.tensorflow_gpu_setup import cleanup_tensorflow_gpu
            cleanup_tensorflow_gpu()
        except ImportError:
            pass 


class ExperimentCheckpointManager:
    """
    Manages checkpoints for individual sub-experiments.
    Creates unique checkpoint directories that are cleaned up after experiments complete.
    """
    
    def __init__(self, experiment_id: str = None):
        """Initialize checkpoint manager for a specific experiment."""
        self.experiment_id = experiment_id or str(uuid.uuid4())
        self.checkpoint_dir = None
        self.best_model_path = None
        self.best_fs_model_path = None
        self.scaler_path = None
        self.created_files = []
        
    def setup_checkpoint_dir(self) -> str:
        """Create temporary checkpoint directory for this experiment."""
        self.checkpoint_dir = tempfile.mkdtemp(prefix=f"checkpoint_{self.experiment_id}_")
        return self.checkpoint_dir
    
    def get_model_checkpoint_path(self, suffix: str = "") -> str:
        """Get path for main model checkpoint."""
        if not self.checkpoint_dir:
            self.setup_checkpoint_dir()
        
        checkpoint_name = f"best_model{suffix}.keras"
        self.best_model_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        self.created_files.append(self.best_model_path)
        return self.best_model_path
    
    def get_fs_model_checkpoint_path(self, suffix: str = "") -> str:
        """Get path for feature selection model checkpoint."""
        if not self.checkpoint_dir:
            self.setup_checkpoint_dir()
        
        checkpoint_name = f"best_fs_model{suffix}.keras"
        self.best_fs_model_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        self.created_files.append(self.best_fs_model_path)
        return self.best_fs_model_path
    
    def get_scaler_checkpoint_path(self) -> str:
        """Get path for scaler checkpoint."""
        if not self.checkpoint_dir:
            self.setup_checkpoint_dir()
        
        self.scaler_path = os.path.join(self.checkpoint_dir, "target_scaler.json")
        self.created_files.append(self.scaler_path)
        return self.scaler_path
    
    def load_best_model(self, custom_objects=None) -> tf.keras.Model:
        """Load the best saved model from checkpoint."""
        if not self.best_model_path or not os.path.exists(self.best_model_path):
            raise FileNotFoundError(f"Best model checkpoint not found: {self.best_model_path}")
        
        print(f"[CheckpointManager] Loading best model from checkpoint: {self.best_model_path}")
        return keras_load_model(self.best_model_path, safe_mode=False)
    
    def load_best_fs_model(self, custom_objects=None) -> tf.keras.Model:
        """Load the best feature selection model from checkpoint."""
        if not self.best_fs_model_path or not os.path.exists(self.best_fs_model_path):
            raise FileNotFoundError(f"Best FS model checkpoint not found: {self.best_fs_model_path}")
        
        print(f"[CheckpointManager] Loading best FS model from checkpoint: {self.best_fs_model_path}")
        return keras_load_model(self.best_fs_model_path, safe_mode=False)
    
    def save_checkpoint_scaler(self, target_scaler):
        """Save target scaler to checkpoint."""
        scaler_path = self.get_scaler_checkpoint_path()
        
        # Serialize TargetScaler which contains MinMaxScaler objects
        scaler_data = {
            'attributes': target_scaler.attributes,
            'is_fitted': target_scaler.is_fitted,
            'scalers': {}
        }
        
        # Serialize each MinMaxScaler
        for attr, scaler in target_scaler.scalers.items():
            scaler_data['scalers'][attr] = {
                'data_min_': scaler.data_min_.tolist(),
                'data_max_': scaler.data_max_.tolist(),
                'data_range_': scaler.data_range_.tolist(),
                'scale_': scaler.scale_.tolist(),
                'min_': scaler.min_.tolist(),
                'n_samples_seen_': int(scaler.n_samples_seen_),
                'feature_range': scaler.feature_range
            }

        with open(scaler_path, 'w') as f:
            json.dump(scaler_data, f)
        print(f"[CheckpointManager] Saved target scaler checkpoint to {scaler_path}")
    
    def load_checkpoint_scaler(self):
        """Load target scaler from checkpoint."""
        if not self.scaler_path or not os.path.exists(self.scaler_path):
            return None
        
        print(f"[CheckpointManager] Loading target scaler from checkpoint: {self.scaler_path}")
        with open(self.scaler_path, 'r') as f:
            scaler_data = json.load(f)
        
        # Reconstruct TargetScaler
        from utils.target_scaling import TargetScaler
        from sklearn.preprocessing import MinMaxScaler
        
        target_scaler = TargetScaler(scaler_data['attributes'])
        target_scaler.is_fitted = scaler_data['is_fitted']
        
        # Reconstruct each MinMaxScaler
        for attr, scaler_info in scaler_data['scalers'].items():
            scaler = MinMaxScaler(feature_range=tuple(scaler_info['feature_range']))
            scaler.data_min_ = np.array(scaler_info['data_min_'])
            scaler.data_max_ = np.array(scaler_info['data_max_'])
            scaler.data_range_ = np.array(scaler_info['data_range_'])
            scaler.scale_ = np.array(scaler_info['scale_'])
            scaler.min_ = np.array(scaler_info['min_'])
            scaler.n_samples_seen_ = scaler_info['n_samples_seen_']
            scaler.feature_range = tuple(scaler_info['feature_range'])
            target_scaler.scalers[attr] = scaler
        
        return target_scaler
    
    def cleanup(self):
        """Clean up checkpoint files and directory."""
        if self.checkpoint_dir and os.path.exists(self.checkpoint_dir):
            try:
                shutil.rmtree(self.checkpoint_dir)
                print(f"[CheckpointManager] Cleaned up checkpoint directory: {self.checkpoint_dir}")
            except Exception as e:
                print(f"[CheckpointManager] Warning: Failed to cleanup checkpoint directory {self.checkpoint_dir}: {e}")
        
        self.checkpoint_dir = None
        self.best_model_path = None
        self.best_fs_model_path = None
        self.scaler_path = None
        self.created_files.clear()
        
        # Clean up memory after checkpoint cleanup
        gc.collect()
        
        # Clean up GPU memory after checkpoint cleanup
        try:
            from utils.experiments.gpu.tensorflow_gpu_setup import cleanup_tensorflow_gpu
            cleanup_tensorflow_gpu()
        except ImportError:
            pass