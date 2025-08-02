# feature_selection/implementations/attention_based/utils/progressive_selection.py

import tensorflow as tf
from typing import Dict, Any, Optional, List, Tuple
import src.config as config


class ProgressiveSelectionManager:
    """Manages progressive selection stages and transitions."""
    
    def __init__(self, k_bands: int, b_std: int, a_ndsi: int, c_indexes: int):
        self.k_bands = k_bands
        self.b_std = b_std
        self.a_ndsi = a_ndsi
        self.c_indexes = c_indexes
        self._stage_tracker = None
        self._hierarchical_weights = None
        
    def initialize_progressive_selectors(self) -> Dict[str, Optional[tf.keras.layers.Layer]]:
        """
        Initialize individual selectors for each progressive stage.
        
        Returns:
            Dict containing initialized selectors
        """
        selectors = {}
        
        # STD Selector: Selects which of the chosen bands to use for STD maps
        if self.b_std < self.k_bands:
            selectors['std_selector'] = tf.keras.layers.Dense(
                self.k_bands,
                activation='softmax',
                name='std_band_selector'
            )
            print(f"[FS Progressive] STD selector initialized: {self.k_bands} -> {self.b_std}")
        else:
            selectors['std_selector'] = None
            print(f"[FS Progressive] STD selector skipped: using all {self.k_bands} bands")
        
        # Index Selector: Selects which indexes to use
        total_available_indexes = len(getattr(config, 'INDEXES', [])) + self.c_indexes
        if total_available_indexes > self.c_indexes:
            selectors['index_selector'] = tf.keras.layers.Dense(
                total_available_indexes,
                activation='softmax',
                name='index_selector'
            )
            print(f"[FS Progressive] Index selector initialized: {total_available_indexes} -> {self.c_indexes}")
        else:
            selectors['index_selector'] = None
            print(f"[FS Progressive] Index selector skipped: using all available indexes")
        
        # NDSI Selector: Selects which bands to use for NDSI pairs
        max_ndsi_pairs = (self.k_bands * (self.k_bands - 1)) // 2
        if max_ndsi_pairs > self.a_ndsi:
            selectors['ndsi_selector'] = tf.keras.layers.Dense(
                max_ndsi_pairs,
                activation='softmax',
                name='ndsi_pair_selector'
            )
            print(f"[FS Progressive] NDSI selector initialized: {max_ndsi_pairs} pairs -> {self.a_ndsi}")
        else:
            selectors['ndsi_selector'] = None
            print(f"[FS Progressive] NDSI selector skipped: using all {max_ndsi_pairs} pairs")
            
        return selectors
    
    def run_hierarchical_feature_selection(
        self,
        components: Dict[str, bool],
        epoch: int
    ) -> Dict[str, Any]:
        """
        Run hierarchical feature selection for current epoch.
        
        Args:
            components: Enabled components
            epoch: Current epoch
            
        Returns:
            Dict with hierarchical selection results and adjusted weights
        """
        hierarchical_enabled = getattr(config, 'FEATURE_SELECTION_HIERARCHICAL', True)
        if not hierarchical_enabled:
            return {}
        
        # Define stage order
        stage_order = self._get_stage_order(components)
        
        # Store stage epochs configuration
        stage_epochs_config = getattr(config, 'FEATURE_SELECTION_STAGE_EPOCHS', {
            'bands': 200, 'std': 30, 'ndsi': 30, 'indexes': 1000
        })
        
        # Initialize stage tracking if needed
        if self._stage_tracker is None:
            self._stage_tracker = self._initialize_stage_tracker(stage_order, stage_epochs_config)
        
        tracker = self._stage_tracker
        current_stage = tracker['current_stage']
        stage_epoch = tracker['stage_epoch']
        
        # Check if current stage should transition
        if current_stage in tracker['completed_stages']:
            current_stage = self._transition_to_next_stage(stage_order, tracker)
        
        # Get stage-specific configuration
        results = self._get_stage_configuration(current_stage, stage_order, stage_epoch, tracker)
        
        # Update stage tracking
        tracker['stage_epoch'] += 1
        
        # Store hierarchical weights for use in loss computation
        self._hierarchical_weights = results.get('stage_weights', {})
        
        return results
    
    def update_stage_convergence(self, validation_loss: float) -> Dict[str, Any]:
        """
        Update stage convergence status.
        
        Args:
            validation_loss: Current validation loss
            
        Returns:
            Dict with convergence status and actions
        """
        if self._stage_tracker is None:
            return {'stage_converged': False, 'action': 'continue'}
        
        tracker = self._stage_tracker
        current_stage = tracker['current_stage']
        
        if current_stage == 'completed':
            return {'stage_converged': True, 'action': 'training_complete'}
        
        # Check convergence
        improvement = tracker['stage_best_loss'] - validation_loss
        convergence_threshold = tracker['stage_convergence_threshold']
        
        # Get max epochs for current stage
        stage_epochs_config = tracker.get('stage_epochs_config', {})
        max_epochs_for_stage = stage_epochs_config.get(current_stage, float('inf'))
        
        convergence_info = {
            'current_stage': current_stage,
            'stage_epoch': tracker['stage_epoch'],
            'validation_loss': validation_loss,
            'stage_best_loss': tracker['stage_best_loss'],
            'improvement': improvement,
            'patience_counter': tracker['stage_patience_counter'],
            'stage_patience': tracker['stage_patience'],
            'max_epochs_for_stage': max_epochs_for_stage
        }
        
        # Check max epochs first
        if tracker['stage_epoch'] >= max_epochs_for_stage:
            tracker['completed_stages'].add(current_stage)
            print(f"[FS Stage Convergence] {current_stage.upper()}: REACHED MAX EPOCHS ({max_epochs_for_stage})")
            return {
                'stage_converged': True,
                'action': 'transition_to_next_stage',
                'convergence_info': convergence_info,
                'convergence_reason': 'max_epochs_reached'
            }
        
        # Check patience-based convergence
        if improvement > convergence_threshold:
            tracker['stage_best_loss'] = validation_loss
            tracker['stage_patience_counter'] = 0
            print(f"[FS Stage Convergence] {current_stage.upper()}: Improvement {improvement:.6f}")
            return {
                'stage_converged': False,
                'action': 'continue',
                'convergence_info': convergence_info
            }
        else:
            tracker['stage_patience_counter'] += 1
            
            if tracker['stage_patience_counter'] >= tracker['stage_patience']:
                tracker['completed_stages'].add(current_stage)
                print(f"[FS Stage Convergence] {current_stage.upper()}: CONVERGED after {tracker['stage_patience']} epochs")
                return {
                    'stage_converged': True,
                    'action': 'transition_to_next_stage',
                    'convergence_info': convergence_info,
                    'convergence_reason': 'patience_exhausted'
                }
            else:
                print(f"[FS Stage Convergence] {current_stage.upper()}: No improvement ({tracker['stage_patience_counter']}/{tracker['stage_patience']})")
                return {
                    'stage_converged': False,
                    'action': 'continue',
                    'convergence_info': convergence_info
                }
    
    def reset_stage_tracking(self):
        """Reset stage tracking for new training session."""
        self._stage_tracker = None
        self._hierarchical_weights = None
        print("[FS Stage Convergence] Stage tracking reset for new training session")
    
    def get_hierarchical_weights(self) -> Optional[Dict[str, float]]:
        """Get current hierarchical weights."""
        return self._hierarchical_weights
    
    def _get_stage_order(self, components: Dict[str, bool]) -> List[str]:
        """Determine stage order based on enabled components."""
        stage_order = []
        
        # Stage 1: Reflectance bands
        if components.get('reflectance', True):
            stage_order.append('bands')
        
        # Stage 2: STD
        std_selection_needed = self.b_std < self.k_bands
        if components.get('std', True) and std_selection_needed:
            stage_order.append('std')
        elif not std_selection_needed and components.get('std', True):
            print(f"[FS Hierarchical] STD selection skipped: B_STD ({self.b_std}) == K_BANDS ({self.k_bands})")
        
        # Stage 3: Indexes
        if components.get('indexes', True):
            stage_order.append('indexes')
        
        # Stage 4: NDSI pairs
        if components.get('ndsi', True):
            stage_order.append('ndsi')
        
        print(f"[FS Hierarchical] Active stages: {stage_order}")
        return stage_order
    
    def _initialize_stage_tracker(self, stage_order: List[str], stage_epochs_config: Dict[str, int]) -> Dict[str, Any]:
        """Initialize stage tracking dictionary."""
        stage_patience_config = getattr(config, 'FEATURE_SELECTION_STAGE_PATIENCE', 8)
        
        if isinstance(stage_patience_config, dict):
            stage_patience_dict = stage_patience_config
        else:
            stage_patience_dict = {stage: stage_patience_config for stage in stage_order}
        
        return {
            'current_stage': stage_order[0] if stage_order else 'completed',
            'stage_epoch': 0,
            'stage_best_loss': float('inf'),
            'stage_patience_counter': 0,
            'stage_patience_dict': stage_patience_dict,
            'stage_patience': stage_patience_dict.get(stage_order[0], 8) if stage_order else 8,
            'completed_stages': set(),
            'stage_convergence_threshold': getattr(config, 'FEATURE_SELECTION_STAGE_CONVERGENCE_THRESHOLD', 0.001),
            'stage_epochs_config': stage_epochs_config
        }
    
    def _transition_to_next_stage(self, stage_order: List[str], tracker: Dict[str, Any]) -> str:
        """Handle transition to next stage."""
        current_stage = tracker['current_stage']
        current_stage_idx = stage_order.index(current_stage)
        
        if current_stage_idx < len(stage_order) - 1:
            next_stage = stage_order[current_stage_idx + 1]
            print(f"[FS Hierarchical] Transitioning from {current_stage} to {next_stage}")
            tracker['current_stage'] = next_stage
            tracker['stage_epoch'] = 0
            tracker['stage_best_loss'] = float('inf')
            tracker['stage_patience_counter'] = 0
            tracker['stage_patience'] = tracker['stage_patience_dict'].get(next_stage, 8)
            print(f"[FS Hierarchical] New stage patience for {next_stage}: {tracker['stage_patience']}")
            return next_stage
        else:
            print(f"[FS Hierarchical] All stages completed: {tracker['completed_stages']}")
            return 'completed'
    
    def _get_stage_configuration(
        self,
        current_stage: str,
        stage_order: List[str],
        stage_epoch: int,
        tracker: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get configuration for current stage."""
        results = {}
        
        stage_configs = {
            'bands': {
                'stage': 'reflectance_bands',
                'focus': 'reflectance_band_selection',
                'weights': {
                    'quality_weight': 3.0,
                    'confidence_weight': 1.5,
                    'diversity_weight': 0.8,
                    'reinforcement_weight': 0.2,
                    'sparsity_weight': 0.1,
                    'spectral_diversity_weight': 0.5
                }
            },
            'std': {
                'stage': 'std_bands',
                'focus': 'std_band_selection',
                'weights': {
                    'quality_weight': 2.5,
                    'confidence_weight': 1.8,
                    'diversity_weight': 0.6,
                    'reinforcement_weight': 0.3,
                    'sparsity_weight': 0.1,
                    'spectral_diversity_weight': 0.2
                }
            },
            'indexes': {
                'stage': 'indexes',
                'focus': 'index_selection',
                'weights': {
                    'quality_weight': 1.5,
                    'confidence_weight': 1.2,
                    'diversity_weight': 0.5,
                    'reinforcement_weight': 1.0,
                    'sparsity_weight': 0.2,
                    'spectral_diversity_weight': 0.2
                }
            },
            'ndsi': {
                'stage': 'ndsi_pairs',
                'focus': 'ndsi_optimization',
                'weights': {
                    'quality_weight': 2.0,
                    'confidence_weight': 1.8,
                    'diversity_weight': 1.0,
                    'reinforcement_weight': 0.5,
                    'sparsity_weight': 0.1,
                    'spectral_diversity_weight': 0.3
                }
            },
            'completed': {
                'stage': 'completed',
                'focus': 'final_optimization',
                'weights': {
                    'quality_weight': 2.0,
                    'confidence_weight': 1.2,
                    'diversity_weight': 0.3,
                    'reinforcement_weight': 1.0,
                    'sparsity_weight': 0.1,
                    'spectral_diversity_weight': 0.2
                }
            }
        }
        
        if current_stage in stage_configs:
            config_info = stage_configs[current_stage]
            results.update(config_info)
            results['stage_weights'] = config_info['weights']
            
            # Print stage info
            if current_stage != 'completed':
                stage_num = stage_order.index(current_stage) + 1 if current_stage in stage_order else 0
                print(f"[FS Hierarchical] Stage {stage_num}: {config_info['stage']} (epoch {stage_epoch})")
                print(f"[FS Hierarchical] Stage patience: {tracker['stage_patience']} epochs")
        
        # Add additional info
        results['stage_tracker'] = tracker
        results['std_selection_needed'] = self.b_std < self.k_bands
        results['k_bands'] = self.k_bands
        results['b_std'] = self.b_std
        results['stage_order'] = stage_order
        results['current_stage'] = current_stage
        results['stage_epoch'] = stage_epoch + 1
        
        return results
