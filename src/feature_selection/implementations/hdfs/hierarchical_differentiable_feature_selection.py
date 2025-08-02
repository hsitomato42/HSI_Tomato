# feature_selection/implementations/hdfs/hierarchical_differentiable_feature_selection.py

import tensorflow as tf
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
import src.config as config
from src.config.enums import FeatureSelectionStrategy

from ...abstract_feature_selector import AbstractFeatureSelector
from .architecture.gumbel_gates import GumbelGates
from .architecture.attention_importance import AttentionImportance
from .architecture.attention_based_selector import AttentionBasedFeatureSelector as AttentionSelector
from .utils.feature_computation import compute_std_maps, compute_ndsi_maps, compute_index_maps_simple
from .utils.progressive_selection import ProgressiveSelectionManager
from .utils.loss_computation import (
    compute_band_quality_scores,
    compute_quality_alignment_loss,
    compute_confidence_loss,
    compute_spectral_diversity_loss,
    compute_gate_reinforcement_loss,
    compute_spectral_region_loss,
    compute_attribute_weighted_loss,
    combine_losses
)


class HierarchicalDifferentiableFeatureSelection(AbstractFeatureSelector):
    """
    HDFS (Hierarchical Differentiable Feature Selection) implementation.
    
    This implementation uses attention mechanisms and Gumbel-Softmax gates
    for differentiable discrete band selection, with support for progressive
    multi-stage training and validation-aware optimization.
    """
    
    def __init__(
        self,
        original_shape: Tuple[int, int, int],
        components: Dict[str, bool],
        selected_bands: List[int],
        selected_indexes: Optional[List] = None,
        enable_feature_selection: bool = True
    ):
        """
        Initialize the HDFS (Hierarchical Differentiable Feature Selection) selector.
        
        Args:
            original_shape: Original model input shape (H, W, C)
            components: Which components are enabled
            selected_bands: Original selected bands
            selected_indexes: Original selected indexes
            enable_feature_selection: Whether to enable feature selection
        """
        # Get component allocations from config
        component_allocations = {
            'k_bands': getattr(config, 'FEATURE_SELECTION_K_BANDS', 5),
            'b_std': getattr(config, 'FEATURE_SELECTION_B_STD', 3),
            'a_ndsi': getattr(config, 'FEATURE_SELECTION_A_NDSI', 3),
            'c_indexes': getattr(config, 'FEATURE_SELECTION_C_INDEXES', 7)
        }
        
        # Initialize base class
        super().__init__(
            original_shape=original_shape,
            components=components,
            component_allocations=component_allocations,
            progressive_mode=True
        )
        
        # Store additional parameters
        self.selected_bands = selected_bands
        self.selected_indexes = selected_indexes
        self.enable_feature_selection = enable_feature_selection
        
        # Get selection strategy from config
        self.selection_strategy = getattr(config, 'FEATURE_SELECTION_STRATEGY', FeatureSelectionStrategy.MODEL_AGNOSTIC)
        
        # Calculate expected output channels
        self.expected_channels = self.get_output_channels()
        
        # Initialize progressive selection manager
        self.progressive_manager = ProgressiveSelectionManager(
            k_bands=component_allocations['k_bands'],
            b_std=component_allocations['b_std'],
            a_ndsi=component_allocations['a_ndsi'],
            c_indexes=component_allocations['c_indexes']
        )
        
        # Initialize components if feature selection is enabled
        if self.enable_feature_selection:
            self._initialize_components()
            
            # Initialize progressive selectors
            progressive_selectors = self.progressive_manager.initialize_progressive_selectors()
            self.std_selector = progressive_selectors.get('std_selector')
            self.index_selector = progressive_selectors.get('index_selector')
            self.ndsi_selector = progressive_selectors.get('ndsi_selector')
        else:
            self.feature_selector = None
            self.attention_module = None
            self.attention_selector = None
            self.std_selector = None
            self.index_selector = None
            self.ndsi_selector = None
        
        # State for tracking
        self._last_print_epoch = None
        self._prev_loss = None
        self._stage_components = None
        self._stage_tracker = None  # Progressive stage tracking
        self._hierarchical_weights = None  # Hierarchical loss weights
        self._hierarchical_info = None  # Cached hierarchical info
        
        print(f"[HDFS] Initialized with {self.expected_channels} output channels")
        print(f"[HDFS] Component allocation: {component_allocations}")
        print(f"[HDFS] Progressive mode: {self.progressive_mode}")
    
    def _initialize_components(self):
        """Initialize the feature selection components."""
        # Initialize Gumbel Gates for band selection
        self.feature_selector = GumbelGates(
            num_bands=204,  # Full spectrum
            k_bands=self.component_allocations['k_bands'],
            initial_temperature=getattr(config, 'FEATURE_SELECTION_GUMBEL_TEMPERATURE', 2.0),
            min_temperature=getattr(config, 'FEATURE_SELECTION_MIN_TEMPERATURE', 0.5),
            temperature_decay=getattr(config, 'FEATURE_SELECTION_TEMPERATURE_DECAY', 0.995),
            exploration_epochs=50,
            random_seed=getattr(config, 'RANDOM_STATE', 42)
        )
        
        # Build the layer with dummy data
        dummy_data = tf.zeros((1, 64, 64, 204))
        _ = self.feature_selector(dummy_data, training=False)
        
        # Initialize attention module
        memory_efficient = getattr(config, 'FEATURE_SELECTION_MEMORY_EFFICIENT_MODE', False)
        
        if memory_efficient:
            attention_heads = getattr(config, 'FEATURE_SELECTION_MEMORY_EFFICIENT_HEADS', 1)
            d_model = getattr(config, 'FEATURE_SELECTION_MEMORY_EFFICIENT_D_MODEL', 32)
            attention_layers = getattr(config, 'FEATURE_SELECTION_MEMORY_EFFICIENT_LAYERS', 1)
        else:
            attention_heads = getattr(config, 'FEATURE_SELECTION_ATTENTION_HEADS', 2)
            d_model = getattr(config, 'FEATURE_SELECTION_D_MODEL', 64)
            attention_layers = getattr(config, 'FEATURE_SELECTION_ATTENTION_LAYERS', 1)
        
        self.attention_module = AttentionImportance(
            num_bands=204,
            num_heads=attention_heads,
            d_model=d_model,
            num_layers=attention_layers,
            random_seed=getattr(config, 'RANDOM_STATE', 42)
        )
        
        # Initialize attention-based selector for unified index selection
        self.attention_selector = AttentionSelector(
            num_bands=204,
            k_bands=self.component_allocations['k_bands'],
            a_ndsi=self.component_allocations['a_ndsi'],
            b_std=self.component_allocations['b_std'],
            c_indexes=self.component_allocations['c_indexes'],
            attention_d_model=d_model,
            attention_heads=attention_heads,
            attention_layers=attention_layers,
            gumbel_temperature=getattr(config, 'FEATURE_SELECTION_GUMBEL_TEMPERATURE', 2.0),
            gumbel_min_temp=getattr(config, 'FEATURE_SELECTION_MIN_TEMPERATURE', 0.5),
            gumbel_decay=getattr(config, 'FEATURE_SELECTION_TEMPERATURE_DECAY', 0.995),
            use_spatial_features=getattr(config, 'FEATURE_SELECTION_USE_SPATIAL_FEATURES', True),
            spatial_kernel_size=getattr(config, 'FEATURE_SELECTION_SPATIAL_KERNEL', 7),
            index_strategy=getattr(config, 'FEATURE_SELECTION_INDEX_STRATEGY', 'competitive')
        )
        
        print(f"[HDFS] Components initialized successfully")
    
    def process_data(
        self,
        data: tf.Tensor,
        training: bool = True,
        epoch: Optional[int] = None
    ) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """
        Process hyperspectral data through feature selection.
        
        Args:
            data: Tensor shape [batch, height, width, bands]
            training: Training mode flag
            epoch: Current training epoch number
            
        Returns:
            processed_data: Feature-selected tensor
            selection_info: Dictionary containing selection results
        """
        if not self.enable_feature_selection or self.feature_selector is None:
            return data, {}
        
        # Update stage components if in progressive mode
        if self.progressive_mode and hasattr(self, 'active_stage') and self.active_stage:
            self._update_stage_components()
        
        # Select bands and get selection info
        selected_bands, selection_info = self.select_bands(data, training)
        
        # Build components
        component_list = []
        component_info = {}
        
        # Use stage-specific components in progressive mode
        active_components = self._stage_components if self._stage_components else self.components
        
        # Add reflectance component
        if active_components.get('reflectance', False):
            component_list.append(selected_bands)
            component_info['reflectance_channels'] = self.component_allocations['k_bands']
        
        # Add STD component
        if active_components.get('std', False):
            std_features = self.compute_std_features(selected_bands, training)
            component_list.append(std_features)
            component_info['std_channels'] = self.component_allocations['b_std']
        
        # Add NDSI component
        if active_components.get('ndsi', False):
            ndsi_features = self.compute_ndsi_features(selected_bands, training)
            component_list.append(ndsi_features)
            component_info['ndsi_channels'] = self.component_allocations['a_ndsi']
        
        # Add indexes component
        if active_components.get('indexes', False):
            index_features = self.compute_index_features(selected_bands, training)
            component_list.append(index_features)
            component_info['indexes_channels'] = self.component_allocations['c_indexes']
        
        # Combine components
        if len(component_list) == 0:
            raise ValueError("No components enabled! At least one component must be enabled.")
        elif len(component_list) == 1:
            processed_data = component_list[0]
        else:
            processed_data = tf.concat(component_list, axis=-1)
        
        # Update selection info
        selection_info['component_info'] = component_info
        
        # Print info once per epoch
        if self._last_print_epoch != epoch:
            print(f"[HDFS] Epoch {epoch}: Processed data shape: {processed_data.shape}")
            self._last_print_epoch = epoch
        
        return processed_data, selection_info
    
    def select_bands(self, data: tf.Tensor, training: bool) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """
        Select optimal spectral bands from input data.
        
        Returns:
            selected_bands: Tensor of selected reflectance bands
            metadata: Band selection metadata
        """
        # Compute attention-based importance scores
        importance_scores = self.attention_module(data, training=training)
        
        # Check if bands are frozen in progressive mode
        if self.progressive_mode and 'bands' in self._frozen_selectors:
            selected_bands, metadata = self._use_preserved_band_selection(data)
        else:
            # Apply Gumbel Gates for band selection
            selection_results = self.feature_selector(
                data,
                importance_scores=importance_scores,
                training=training,
                epoch=self._last_print_epoch
            )
            
            selected_bands = selection_results['selected_cube']
            metadata = selection_results
            
            # Store selected band indices for memory optimization
            if 'selected_indices' in metadata:
                self._preserved_selections['band_indices'] = metadata['selected_indices'].numpy().tolist()
                print(f"[HDFS] Preserved band indices: {self._preserved_selections['band_indices'][:10]}...")
        
        return selected_bands, metadata
    
    def compute_std_features(self, selected_bands: tf.Tensor, training: bool) -> tf.Tensor:
        """
        Compute texture STD features from selected bands.
        """
        if self.progressive_mode and self.active_stage in ['std', 'indexes', 'ndsi', 'finetune']:
            # Progressive mode: Use STD selector if available
            if self.std_selector is not None and self.active_stage == 'std':
                std_bands = self._select_std_bands_progressive(selected_bands, training)
            elif 'std_bands' in self._preserved_selections:
                std_bands = self._apply_preserved_std_selection(selected_bands)
            else:
                std_bands = selected_bands
            
            std_maps = compute_std_maps(std_bands, self.component_allocations['b_std'])
        else:
            # Non-progressive mode: use all selected bands
            std_maps = compute_std_maps(selected_bands, self.component_allocations['b_std'])
        
        return std_maps
    
    def compute_ndsi_features(self, selected_bands: tf.Tensor, training: bool) -> tf.Tensor:
        """
        Compute NDSI features from selected bands.
        """
        if self.progressive_mode and self.active_stage in ['ndsi', 'finetune']:
            # Progressive mode: Use NDSI selector if available
            if self.ndsi_selector is not None and self.active_stage == 'ndsi':
                ndsi_maps = self._select_ndsi_pairs_progressive(selected_bands, training)
            elif 'ndsi_pairs' in self._preserved_selections:
                ndsi_maps = self._apply_preserved_ndsi_selection(selected_bands)
            else:
                ndsi_maps = compute_ndsi_maps(selected_bands, self.component_allocations['a_ndsi'])
        else:
            ndsi_maps = compute_ndsi_maps(selected_bands, self.component_allocations['a_ndsi'])
        
        return ndsi_maps
    
    def compute_index_features(self, selected_bands: tf.Tensor, training: bool) -> tf.Tensor:
        """
        Compute spectral index features based on selected bands.
        """
        # Get selected band indices from the last selection
        if hasattr(self, '_last_selected_indices'):
            selected_indices = self._last_selected_indices
        else:
            # Fallback: create dummy indices
            batch_size = tf.shape(selected_bands)[0]
            selected_indices = tf.tile(
                tf.expand_dims(tf.range(self.component_allocations['k_bands']), 0),
                [batch_size, 1]
            )
        
        # Use attention selector if available
        if hasattr(self, 'attention_selector') and self.attention_selector is not None:
            return self.attention_selector._compute_spectral_indexes(
                selected_bands, selected_indices, training=training
            )
        else:
            return compute_index_maps_simple(
                selected_bands, selected_indices, self.component_allocations['c_indexes']
            )
    
    def get_trainable_variables(self) -> List[tf.Variable]:
        """
        Return trainable variables for the active progressive stage.
        """
        variables = []
        
        if not self.progressive_mode:
            # Non-progressive mode: all variables are trainable
            if self.feature_selector is not None:
                variables.extend(self.feature_selector.trainable_variables)
            if self.attention_module is not None:
                variables.extend(self.attention_module.trainable_variables)
            if self.attention_selector is not None:
                variables.extend(self.attention_selector.trainable_variables)
            return variables
        
        # Progressive mode: only train selectors for current stage
        if self.active_stage == 'bands':
            if self.feature_selector is not None:
                variables.extend(self.feature_selector.trainable_variables)
            if self.attention_module is not None:
                variables.extend(self.attention_module.trainable_variables)
                
        elif self.active_stage == 'std':
            if self.std_selector is not None:
                variables.extend(self.std_selector.trainable_variables)
                
        elif self.active_stage == 'indexes':
            if self.index_selector is not None:
                variables.extend(self.index_selector.trainable_variables)
            # Include attention selector's index selectors
            if self.attention_selector is not None and hasattr(self.attention_selector, 'index_selectors'):
                self._add_index_selector_variables(variables)
                
        elif self.active_stage == 'ndsi':
            if self.ndsi_selector is not None:
                variables.extend(self.ndsi_selector.trainable_variables)
                
        elif self.active_stage == 'finetune':
            # All selectors frozen, no feature selection variables
            pass
        
        return variables
    
    def compute_loss(
        self,
        selection_info: Dict[str, Any],
        prediction_loss: tf.Tensor,
        **kwargs
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute the feature selection loss.
        
        Args:
            selection_info: Metadata from process_data
            prediction_loss: The main prediction loss from the model
            kwargs: Additional inputs (validation losses, attribute losses, etc.)
            
        Returns:
            total_loss: Combined feature selection loss
            loss_breakdown: Dictionary of individual loss terms
        """
        if not selection_info:
            return prediction_loss, {'prediction_loss': prediction_loss}
        
        gates = selection_info.get('gates')
        if gates is None:
            return prediction_loss, {'prediction_loss': prediction_loss}
        
        # Extract optional parameters
        epoch = kwargs.get('epoch')
        attribute_losses = kwargs.get('attribute_losses')
        validation_loss = kwargs.get('validation_loss')
        validation_attribute_losses = kwargs.get('validation_attribute_losses')
        
        # Compute average gates
        avg_gates = tf.reduce_mean(gates, axis=0)
        
        # 1. Band quality scores
        spectral_quality = compute_band_quality_scores()
        
        # 2. Quality alignment loss
        quality_alignment_loss = compute_quality_alignment_loss(avg_gates, spectral_quality)
        
        # 3. Confidence loss
        confidence_loss, confidence_reward, top_k_indices = compute_confidence_loss(
            avg_gates, self.component_allocations['k_bands']
        )
        
        # 4. Spectral diversity loss
        wavelengths = tf.linspace(400.0, 1000.0, 204)
        diversity_loss, diversity_reward = compute_spectral_diversity_loss(wavelengths, top_k_indices)
        
        # 5. Gate reinforcement loss
        gate_reinforcement_loss = compute_gate_reinforcement_loss(
            avg_gates, prediction_loss, validation_loss
        )
        
        # 6. Sparsity loss
        sparsity_loss = selection_info.get('sparsity_loss', 0.0)
        
        # 7. Spectral region loss
        spectral_region_loss = compute_spectral_region_loss(
            wavelengths, avg_gates, self.component_allocations['k_bands']
        )
        
        # 8. Attribute-weighted loss
        attribute_weighted_loss = compute_attribute_weighted_loss(
            prediction_loss, attribute_losses, validation_loss, validation_attribute_losses
        )
        
        # Update temperature decay if needed
        if hasattr(self, 'feature_selector') and self.feature_selector is not None:
            self._update_temperature_decay(epoch, attribute_weighted_loss)
        
        # Get hierarchical weights if available
        hierarchical_weights = self.progressive_manager.get_hierarchical_weights()
        
        # Combine losses
        total_loss = combine_losses(
            quality_alignment_loss=quality_alignment_loss,
            confidence_loss=confidence_loss,
            diversity_loss=diversity_loss,
            gate_reinforcement_loss=gate_reinforcement_loss,
            sparsity_loss=sparsity_loss,
            spectral_region_loss=spectral_region_loss,
            hierarchical_weights=hierarchical_weights,
            epoch=epoch
        )
        
        # Build loss breakdown
        loss_breakdown = {
            'prediction_loss': prediction_loss,
            'attribute_weighted_loss': attribute_weighted_loss,
            'quality_alignment_loss': quality_alignment_loss,
            'confidence_loss': confidence_loss,
            'diversity_loss': diversity_loss,
            'gate_reinforcement_loss': gate_reinforcement_loss,
            'sparsity_loss': sparsity_loss,
            'spectral_region_loss': spectral_region_loss,
            'total_loss': total_loss,
            'selected_quality_score': tf.reduce_sum(tf.gather(spectral_quality, top_k_indices)),
            'selection_confidence': confidence_reward,
            'spectral_diversity': diversity_reward
        }
        
        # Add validation info if available
        if validation_loss is not None:
            loss_breakdown['validation_loss'] = validation_loss
            loss_breakdown['validation_optimization_active'] = True
        else:
            loss_breakdown['validation_optimization_active'] = False
        
        return total_loss, loss_breakdown
    
    def get_current_selection(self) -> Dict[str, Any]:
        """
        Retrieve the current feature selection state and details.
        """
        current_selection = {
            'active_stage': self.active_stage,
            'frozen_selectors': list(self._frozen_selectors),
            'preserved_selections': self._preserved_selections,
            'progressive_mode': self.progressive_mode
        }
        
        # Add current band selection
        if self.feature_selector is not None:
            top_k_indices = self.feature_selector.get_top_k_bands()
            if hasattr(top_k_indices, 'numpy'):
                current_selection['selected_bands'] = top_k_indices.numpy().tolist()
            else:
                current_selection['selected_bands'] = top_k_indices.tolist()
        
        return current_selection
    
    def save_state(self, filepath: str):
        """
        Save the internal selection state to a file.
        """
        state = self.get_current_selection()
        
        # Add gate logits and temperature if available
        if self.feature_selector is not None:
            state['gate_logits'] = self.feature_selector.gate_logits.numpy()
            state['temperature'] = self.feature_selector.temperature.numpy()
        
        # Add component allocations
        state['component_allocations'] = self.component_allocations
        
        # Save with pickle
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        
        print(f"[HDFS] State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """
        Load the internal selection state from a file.
        """
        try:
            with open(filepath, 'rb') as f:
                state = pickle.load(f)
            
            # Restore basic state
            self.active_stage = state.get('active_stage')
            self._frozen_selectors = set(state.get('frozen_selectors', []))
            self._preserved_selections = state.get('preserved_selections', {})
            
            # Restore gate logits and temperature
            if self.feature_selector is not None and 'gate_logits' in state:
                self.feature_selector.gate_logits.assign(state['gate_logits'])
                self.feature_selector.temperature.assign(state['temperature'])
            
            print(f"[HDFS] State loaded from {filepath}")
            
        except Exception as e:
            print(f"[HDFS] Warning: Could not load state from {filepath}: {e}")
    
    # Progressive training methods
    
    def run_hierarchical_feature_selection(
        self,
        hyperspectral_cube: tf.Tensor,
        model: Any,
        training_data: Any,
        validation_data: Any,
        epoch: int
    ) -> Dict[str, Any]:
        """Run hierarchical feature selection for current epoch."""
        result = self.progressive_manager.run_hierarchical_feature_selection(
            components=self.components,
            epoch=epoch
        )
        # Update stage tracker
        self._stage_tracker = self.progressive_manager._stage_tracker
        return result
    
    def update_stage_convergence(self, validation_loss: float) -> Dict[str, Any]:
        """Update stage convergence status."""
        result = self.progressive_manager.update_stage_convergence(validation_loss)
        # Update internal stage tracker
        self._stage_tracker = self.progressive_manager._stage_tracker
        return result
    
    def reset_stage_tracking(self):
        """Reset stage tracking for new training session."""
        self.progressive_manager.reset_stage_tracking()
        self._stage_tracker = self.progressive_manager._stage_tracker
    
    # Backward compatibility methods
    
    def process_hyperspectral_data(self, hyperspectral_cube: tf.Tensor, training: bool = True, epoch: Optional[int] = None) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """
        Backward compatibility wrapper for process_data.
        
        Args:
            hyperspectral_cube: Input tensor
            training: Training mode flag
            epoch: Current epoch
            
        Returns:
            Processed data and selection info
        """
        return self.process_data(hyperspectral_cube, training, epoch)
    
    def get_prediction_aware_feature_selection_loss(
        self,
        selection_info: Dict[str, Any],
        prediction_loss: tf.Tensor,
        epoch: Optional[int] = None,
        attribute_losses: Optional[Dict[str, tf.Tensor]] = None,
        validation_loss: Optional[tf.Tensor] = None,
        validation_attribute_losses: Optional[Dict[str, tf.Tensor]] = None,
        is_validation: bool = False
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Backward compatibility wrapper for compute_loss.
        
        Args:
            selection_info: Selection metadata
            prediction_loss: Prediction loss
            epoch: Current epoch
            attribute_losses: Per-attribute losses
            validation_loss: Validation loss
            validation_attribute_losses: Validation per-attribute losses
            is_validation: Whether this is validation step
            
        Returns:
            Total loss and loss breakdown
        """
        return self.compute_loss(
            selection_info=selection_info,
            prediction_loss=prediction_loss,
            epoch=epoch,
            attribute_losses=attribute_losses,
            validation_loss=validation_loss,
            validation_attribute_losses=validation_attribute_losses
        )
    
    def get_strategy_loss_weights(self, epoch: Optional[int] = None) -> Dict[str, float]:
        """
        Get loss weights for balancing feature selection vs prediction loss.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with 'feature_selection_weight' and 'prediction_weight'
        """
        # Use adaptive weighting based on epoch
        if epoch is None:
            epoch = 0
        
        # Start with more emphasis on feature selection, gradually shift to prediction
        warmup_epochs = getattr(config, 'FEATURE_SELECTION_WARMUP_EPOCHS', 50)
        
        if epoch < warmup_epochs:
            # During warmup: emphasize feature selection
            fs_weight = 1.0
            pred_weight = 0.5
        else:
            # After warmup: gradually reduce FS weight
            decay_rate = getattr(config, 'FEATURE_SELECTION_WEIGHT_DECAY', 0.995)
            epochs_after_warmup = epoch - warmup_epochs
            fs_weight = max(0.1, decay_rate ** epochs_after_warmup)
            pred_weight = 1.0
        
        # Apply strategy-specific adjustments
        if hasattr(self, 'selection_strategy'):
            if self.selection_strategy == FeatureSelectionStrategy.MODEL_AGNOSTIC:
                # Model-agnostic: balanced weights
                pass  # Use default weights
            elif self.selection_strategy == FeatureSelectionStrategy.SEMI_ADAPTIVE:
                # Semi-adaptive: slightly higher FS weight
                fs_weight = max(0.3, fs_weight)
            elif self.selection_strategy == FeatureSelectionStrategy.MODEL_SPECIFIC:
                # Model-specific: emphasize prediction performance
                fs_weight = fs_weight * 0.7
        
        return {
            'feature_selection_weight': fs_weight,
            'prediction_weight': pred_weight
        }
    
    def is_enabled(self) -> bool:
        """
        Check if feature selection is enabled.
        
        Returns:
            True if feature selection is enabled
        """
        return self.enable_feature_selection
    
    def set_stage(self, stage_name: str) -> None:
        """
        Set the active stage for progressive feature selection.
        
        Args:
            stage_name: Name of the stage ('bands', 'std', 'indexes', 'ndsi', 'finetune')
        """
        self.set_active_stage(stage_name)
        # Initialize stage tracker if needed
        if not hasattr(self, '_stage_tracker') or self._stage_tracker is None:
            self._stage_tracker = {
                'current_stage': stage_name,
                'stage_epoch': 0,
                'best_val_loss': float('inf'),
                'stage_patience_counter': 0,
                'stage_patience': 8
            }
    
    def get_current_input_channels(self) -> int:
        """
        Get the current number of input channels based on active stage.
        
        Returns:
            Number of channels for the current stage
        """
        # Return channels based on current stage
        stage = getattr(self, 'active_stage', 'finetune')
        
        if stage == 'bands':
            # Stage 1: Only reflectance bands
            return self.component_allocations.get('k_bands', 5)
        elif stage == 'std':
            # Stage 2: Reflectance + STD
            return (self.component_allocations.get('k_bands', 5) + 
                    self.component_allocations.get('b_std', 5))
        elif stage == 'indexes':
            # Stage 3: Reflectance + STD + Indexes
            return (self.component_allocations.get('k_bands', 5) + 
                    self.component_allocations.get('b_std', 5) + 
                    self.component_allocations.get('c_indexes', 5))
        elif stage == 'ndsi':
            # Stage 4: All components
            return self.get_output_channels()
        elif stage == 'finetune':
            # Stage 5: All components
            return self.get_output_channels()
        else:
            # Default: return total channels
            return self.get_output_channels()
    
    # Helper methods
    
    def _update_stage_components(self):
        """Update component map based on current stage."""
        user_components = self.components
        
        stage_component_maps = {
            'bands': {
                'reflectance': user_components.get('reflectance', False),
                'std': False,
                'ndsi': False,
                'indexes': False
            },
            'std': {
                'reflectance': user_components.get('reflectance', False),
                'std': user_components.get('std', False),
                'ndsi': False,
                'indexes': False
            },
            'indexes': {
                'reflectance': user_components.get('reflectance', False),
                'std': user_components.get('std', False),
                'ndsi': False,
                'indexes': user_components.get('indexes', False)
            },
            'ndsi': {
                'reflectance': user_components.get('reflectance', False),
                'std': user_components.get('std', False),
                'ndsi': user_components.get('ndsi', False),
                'indexes': user_components.get('indexes', False)
            },
            'finetune': user_components.copy()
        }
        
        self._stage_components = stage_component_maps.get(self.active_stage, user_components)
    
    def _use_preserved_band_selection(self, data: tf.Tensor) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """Use preserved band selections instead of running the selector."""
        if 'selected_bands' not in self._preserved_selections:
            # Fallback to normal selection
            return self.select_bands(data, training=False)
        
        preserved_bands = self._preserved_selections['selected_bands']
        batch_size = tf.shape(data)[0]
        
        # Create selection indices
        selected_indices = tf.tile(tf.constant([preserved_bands]), [batch_size, 1])
        
        # Extract selected bands
        selected_bands = tf.gather(data, selected_indices, axis=3, batch_dims=1)
        
        # Create mock metadata
        metadata = {
            'gates': tf.zeros((batch_size, 204)),
            'hard_gates': tf.zeros((batch_size, 204)),
            'selected_band_indices': selected_indices,
            'sparsity_loss': tf.constant(0.0),
            'confidence_loss': tf.constant(0.0),
            'stability_loss': tf.constant(0.0),
            'fs_loss': tf.constant(0.0),
            'temperature': tf.constant(1.0),
            'num_selected': tf.constant(float(self.component_allocations['k_bands'])),
            'selection_confidence': tf.constant(1.0)
        }
        
        return selected_bands, metadata
    
    def _select_std_bands_progressive(self, selected_reflectance: tf.Tensor, training: bool) -> tf.Tensor:
        """Select subset of bands for STD computation in progressive mode."""
        # Global average pooling to get band-level features
        band_features = tf.reduce_mean(selected_reflectance, axis=[1, 2])
        
        # Use STD selector to get selection probabilities
        std_probs = self.std_selector(band_features, training=training)
        
        # Select top-k bands for STD
        _, top_indices = tf.nn.top_k(std_probs, k=self.component_allocations['b_std'])
        
        # Gather selected bands
        std_bands = tf.gather(selected_reflectance, top_indices, batch_dims=1, axis=3)
        
        return std_bands
    
    def _apply_preserved_std_selection(self, selected_reflectance: tf.Tensor) -> tf.Tensor:
        """Apply preserved STD selection."""
        # For now, use all selected bands
        # TODO: Implement actual preservation logic
        return selected_reflectance
    
    def _select_ndsi_pairs_progressive(self, selected_reflectance: tf.Tensor, training: bool) -> tf.Tensor:
        """Select optimal NDSI pairs in progressive mode."""
        k_bands = self.component_allocations['k_bands']
        
        # Generate all possible pairs
        all_pairs = []
        for i in range(k_bands):
            for j in range(i + 1, k_bands):
                all_pairs.append((i, j))
        
        # Compute NDSI for all pairs
        all_ndsi_maps = []
        for i, j in all_pairs:
            band_i = selected_reflectance[:, :, :, i]
            band_j = selected_reflectance[:, :, :, j]
            ndsi = (band_i - band_j) / (band_i + band_j + 1e-8)
            all_ndsi_maps.append(ndsi)
        
        # Stack all NDSI maps
        all_ndsi_tensor = tf.stack(all_ndsi_maps, axis=-1)
        
        # Compute features for pair selection
        pair_features = []
        for pair_idx in range(len(all_pairs)):
            ndsi_map = all_ndsi_tensor[:, :, :, pair_idx]
            variance = tf.reduce_mean(tf.math.reduce_variance(ndsi_map, axis=[1, 2]))
            range_val = tf.reduce_mean(tf.reduce_max(ndsi_map, axis=[1, 2]) - tf.reduce_min(ndsi_map, axis=[1, 2]))
            pair_features.append(variance + range_val)
        
        pair_features = tf.stack(pair_features, axis=-1)
        pair_features = tf.expand_dims(pair_features, 0)
        pair_features = tf.tile(pair_features, [tf.shape(selected_reflectance)[0], 1])
        
        # Use NDSI selector
        ndsi_probs = self.ndsi_selector(pair_features, training=training)
        
        # Select top-k pairs
        _, top_indices = tf.nn.top_k(ndsi_probs, k=self.component_allocations['a_ndsi'])
        
        # Gather selected NDSI maps
        selected_ndsi = tf.gather(all_ndsi_tensor, top_indices, batch_dims=1, axis=3)
        
        return selected_ndsi
    
    def _apply_preserved_ndsi_selection(self, selected_reflectance: tf.Tensor) -> tf.Tensor:
        """Apply preserved NDSI selection."""
        # For now, use default computation
        # TODO: Implement actual preservation logic
        return compute_ndsi_maps(selected_reflectance, self.component_allocations['a_ndsi'])
    
    def _add_index_selector_variables(self, variables: List[tf.Variable]):
        """Add index selector variables based on strategy."""
        index_strategy = getattr(config, 'FEATURE_SELECTION_INDEX_STRATEGY', 'hybrid')
        
        if index_strategy == "learned":
            variables.extend(self.attention_selector.index_selectors['learned'].trainable_variables)
        elif index_strategy == "competitive":
            variables.extend(self.attention_selector.index_selectors['combined'].trainable_variables)
        elif index_strategy == "hybrid":
            variables.extend(self.attention_selector.index_selectors['learned'].trainable_variables)
            variables.extend(self.attention_selector.index_selectors['predefined'].trainable_variables)
        elif index_strategy == "existing":
            variables.extend(self.attention_selector.index_selectors['predefined'].trainable_variables)
        
        # Include learned index networks if applicable
        if hasattr(self.attention_selector, 'learned_index_networks') and self.attention_selector.learned_index_networks:
            for network in self.attention_selector.learned_index_networks:
                variables.extend(network.trainable_variables)
    
    def _update_temperature_decay(self, epoch: Optional[int], attribute_weighted_loss: tf.Tensor):
        """Update temperature decay based on validation improvement."""
        adaptive_temp = getattr(config, 'FEATURE_SELECTION_ADAPTIVE_TEMPERATURE', True)
        
        if adaptive_temp and epoch is not None and epoch > 5:
            if hasattr(self, '_prev_loss') and attribute_weighted_loss < self._prev_loss:
                # Improving: slower temperature decay
                temp_decay = getattr(config, 'FEATURE_SELECTION_TEMPERATURE_DECAY_IMPROVING', 0.99)
            else:
                # Stagnating: faster temperature decay  
                temp_decay = getattr(config, 'FEATURE_SELECTION_TEMPERATURE_DECAY_STAGNATING', 0.995)
            
            self.feature_selector.temperature_decay = temp_decay
        
        self._prev_loss = attribute_weighted_loss
