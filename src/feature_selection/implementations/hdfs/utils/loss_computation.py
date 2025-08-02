# feature_selection/implementations/attention_based/utils/loss_computation.py

import tensorflow as tf
from typing import Dict, Optional, Tuple, Any
import src.config as config


def compute_band_quality_scores() -> tf.Tensor:
    """
    Create band quality scores based on spectral importance heuristics.
    
    Returns:
        spectral_quality: (204,) tensor of quality scores
    """
    wavelengths = tf.linspace(400.0, 1000.0, 204)  # Approximate wavelength range
    
    # Red-edge region (700-750nm) - critical for vegetation analysis
    red_edge_importance = tf.exp(-0.5 * tf.square((wavelengths - 725.0) / 25.0))
    
    # NIR region (750-900nm) - important for structural properties  
    nir_importance = tf.exp(-0.5 * tf.square((wavelengths - 825.0) / 75.0))
    
    # Visible red (650-700nm) - important for pigments
    red_importance = tf.exp(-0.5 * tf.square((wavelengths - 675.0) / 25.0))
    
    # Green region (500-600nm) - moderate importance
    green_importance = tf.exp(-0.5 * tf.square((wavelengths - 550.0) / 50.0)) * 0.7
    
    # Combine importance scores
    spectral_quality = (
        3.0 * red_edge_importance + 
        2.5 * nir_importance + 
        2.0 * red_importance + 
        1.5 * green_importance
    )
    
    # Normalize to [0, 1] range
    spectral_quality = spectral_quality / tf.reduce_max(spectral_quality)
    
    return spectral_quality


def compute_quality_alignment_loss(avg_gates: tf.Tensor, spectral_quality: tf.Tensor) -> tf.Tensor:
    """
    Compute loss that rewards selecting bands with high spectral quality.
    
    Args:
        avg_gates: (num_bands,) average gate probabilities
        spectral_quality: (num_bands,) band quality scores
        
    Returns:
        quality_alignment_loss: scalar loss
    """
    return -tf.reduce_sum(avg_gates * spectral_quality)


def compute_confidence_loss(avg_gates: tf.Tensor, k_bands: int) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Compute confidence loss to encourage confident selections.
    
    Args:
        avg_gates: (num_bands,) average gate probabilities
        k_bands: number of bands to select
        
    Returns:
        confidence_loss: scalar loss
        confidence_reward: confidence score
        top_k_indices: indices of top k bands
    """
    top_k_values, top_k_indices = tf.nn.top_k(avg_gates, k=k_bands)
    
    # Reward high confidence in top-k selections
    confidence_reward = tf.reduce_mean(top_k_values)
    confidence_loss = -confidence_reward  # Negative because we want to maximize confidence
    
    return confidence_loss, confidence_reward, top_k_indices


def compute_spectral_diversity_loss(wavelengths: tf.Tensor, top_k_indices: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Compute loss to encourage selecting bands from different spectral regions.
    
    Args:
        wavelengths: (num_bands,) wavelengths
        top_k_indices: (k_bands,) selected band indices
        
    Returns:
        diversity_loss: scalar loss
        diversity_reward: diversity score
    """
    selected_wavelengths = tf.gather(wavelengths, top_k_indices)
    
    # Compute pairwise distances between selected wavelengths
    wl_expanded_i = tf.expand_dims(selected_wavelengths, 1)  # (k_bands, 1)
    wl_expanded_j = tf.expand_dims(selected_wavelengths, 0)  # (1, k_bands)
    pairwise_distances = tf.abs(wl_expanded_i - wl_expanded_j)
    
    # Encourage large spectral distances (diversity)
    diversity_reward = tf.reduce_mean(pairwise_distances)
    diversity_loss = -diversity_reward / 100.0  # Scale and negate
    
    return diversity_loss, diversity_reward


def compute_gate_reinforcement_loss(
    avg_gates: tf.Tensor,
    prediction_loss: tf.Tensor,
    validation_loss: Optional[tf.Tensor] = None
) -> tf.Tensor:
    """
    Compute gate reinforcement loss based on performance.
    
    Args:
        avg_gates: (num_bands,) average gate probabilities
        prediction_loss: training prediction loss
        validation_loss: optional validation loss
        
    Returns:
        gate_reinforcement_loss: scalar loss
    """
    validation_reinforcement = getattr(config, 'FEATURE_SELECTION_VALIDATION_REINFORCEMENT', True)
    
    if validation_reinforcement and validation_loss is not None:
        # Use validation loss for gate reinforcement
        normalized_val_loss = tf.nn.sigmoid(validation_loss / 10.0)
        gate_reinforcement_loss = normalized_val_loss * tf.reduce_mean(tf.square(avg_gates - 0.5))
        
        # Add training loss signal with lower weight
        normalized_train_loss = tf.nn.sigmoid(prediction_loss / 10.0)
        train_reinforcement = normalized_train_loss * tf.reduce_mean(tf.square(avg_gates - 0.5))
        
        # Combine with validation priority
        val_reinforcement_weight = getattr(config, 'FEATURE_SELECTION_VALIDATION_REINFORCEMENT_WEIGHT', 1.0)
        gate_reinforcement_loss = (
            val_reinforcement_weight * gate_reinforcement_loss + 
            0.3 * train_reinforcement
        ) / (val_reinforcement_weight + 0.3)
    else:
        # Use training loss for gate reinforcement
        normalized_pred_loss = tf.nn.sigmoid(prediction_loss / 10.0)
        gate_reinforcement_loss = normalized_pred_loss * tf.reduce_mean(tf.square(avg_gates - 0.5))
    
    return gate_reinforcement_loss


def compute_spectral_region_loss(wavelengths: tf.Tensor, avg_gates: tf.Tensor, k_bands: int) -> tf.Tensor:
    """
    Compute loss to encourage balanced selection across spectral regions.
    
    Args:
        wavelengths: (num_bands,) wavelengths
        avg_gates: (num_bands,) average gate probabilities
        k_bands: number of bands to select
        
    Returns:
        spectral_region_loss: scalar loss
    """
    spectral_diversity_enabled = getattr(config, 'FEATURE_SELECTION_SPECTRAL_DIVERSITY', True)
    
    if not spectral_diversity_enabled:
        return tf.constant(0.0)
    
    spectral_regions = getattr(config, 'FEATURE_SELECTION_SPECTRAL_REGIONS', {
        'visible': (400, 700), 'near_ir': (700, 1000), 'shortwave_ir': (1000, 2500)
    })
    
    # Map bands to spectral regions
    top_k_values, top_k_indices = tf.nn.top_k(avg_gates, k=k_bands)
    selected_wavelengths = tf.gather(wavelengths, top_k_indices)
    
    # Count selections per region
    region_counts = []
    for region_name, (min_wl, max_wl) in spectral_regions.items():
        in_region = tf.logical_and(
            selected_wavelengths >= min_wl,
            selected_wavelengths <= max_wl
        )
        region_count = tf.reduce_sum(tf.cast(in_region, tf.float32))
        region_counts.append(region_count)
    
    # Encourage balanced selection across regions
    region_counts = tf.stack(region_counts)
    ideal_per_region = k_bands / len(spectral_regions)
    spectral_region_loss = tf.reduce_mean(tf.square(region_counts - ideal_per_region))
    
    return spectral_region_loss


def compute_attribute_weighted_loss(
    prediction_loss: tf.Tensor,
    attribute_losses: Optional[Dict[str, tf.Tensor]] = None,
    validation_loss: Optional[tf.Tensor] = None,
    validation_attribute_losses: Optional[Dict[str, tf.Tensor]] = None
) -> tf.Tensor:
    """
    Compute attribute-weighted loss with validation prioritization.
    
    Args:
        prediction_loss: training prediction loss
        attribute_losses: training attribute-specific losses
        validation_loss: validation prediction loss
        validation_attribute_losses: validation attribute-specific losses
        
    Returns:
        attribute_weighted_loss: scalar loss
    """
    validation_optimization = getattr(config, 'FEATURE_SELECTION_VALIDATION_OPTIMIZATION', True)
    
    if validation_optimization and validation_loss is not None:
        # Use validation loss as main performance signal
        validation_weight = getattr(config, 'FEATURE_SELECTION_VALIDATION_WEIGHT', 2.0)
        training_weight = getattr(config, 'FEATURE_SELECTION_TRAINING_WEIGHT', 0.5)
        
        # Process validation attribute losses
        validation_weighted_loss = validation_loss
        if validation_attribute_losses is not None:
            attribute_weights = getattr(config, 'FEATURE_SELECTION_ATTRIBUTE_WEIGHTS', {
                'TSS': 1.5, 'citric_acid': 1.0, 'firmness': 0.8,
                'pH': 1.2, 'weight': 0.8, 'ascorbic_acid': 1.5
            })
            
            val_weighted_losses = []
            for attr, loss in validation_attribute_losses.items():
                weight = attribute_weights.get(attr, 1.0)
                val_weighted_losses.append(weight * loss)
            
            if val_weighted_losses:
                validation_weighted_loss = tf.reduce_mean(val_weighted_losses)
        
        # Process training attribute losses
        training_weighted_loss = prediction_loss
        if attribute_losses is not None:
            attribute_weights = getattr(config, 'FEATURE_SELECTION_ATTRIBUTE_WEIGHTS', {
                'TSS': 1.5, 'citric_acid': 1.0, 'firmness': 0.8,
                'pH': 1.2, 'weight': 0.8, 'ascorbic_acid': 1.5
            })
            
            train_weighted_losses = []
            for attr, loss in attribute_losses.items():
                weight = attribute_weights.get(attr, 1.0)
                train_weighted_losses.append(weight * loss)
            
            if train_weighted_losses:
                training_weighted_loss = tf.reduce_mean(train_weighted_losses)
        
        # Combined validation-prioritized performance signal
        attribute_weighted_loss = (
            validation_weight * validation_weighted_loss + 
            training_weight * training_weighted_loss
        ) / (validation_weight + training_weight)
        
    else:
        # Use training loss only
        attribute_weighted_loss = prediction_loss
        if attribute_losses is not None:
            attribute_weights = getattr(config, 'FEATURE_SELECTION_ATTRIBUTE_WEIGHTS', {
                'TSS': 1.5, 'citric_acid': 1.0, 'firmness': 0.8,
                'pH': 1.2, 'weight': 0.8, 'ascorbic_acid': 1.5
            })
            
            weighted_losses = []
            for attr, loss in attribute_losses.items():
                weight = attribute_weights.get(attr, 1.0)
                weighted_losses.append(weight * loss)
            
            if weighted_losses:
                attribute_weighted_loss = tf.reduce_mean(weighted_losses)
    
    return attribute_weighted_loss


def combine_losses(
    quality_alignment_loss: tf.Tensor,
    confidence_loss: tf.Tensor,
    diversity_loss: tf.Tensor,
    gate_reinforcement_loss: tf.Tensor,
    sparsity_loss: tf.Tensor,
    spectral_region_loss: tf.Tensor,
    hierarchical_weights: Optional[Dict[str, float]] = None,
    epoch: Optional[int] = None
) -> tf.Tensor:
    """
    Combine all losses with appropriate weights.
    
    Args:
        quality_alignment_loss: band quality loss
        confidence_loss: selection confidence loss
        diversity_loss: spectral diversity loss
        gate_reinforcement_loss: performance-based reinforcement loss
        sparsity_loss: sparsity regularization loss
        spectral_region_loss: spectral region diversity loss
        hierarchical_weights: optional hierarchical stage weights
        epoch: current training epoch
        
    Returns:
        total_loss: combined weighted loss
    """
    # Get base weights from config
    quality_weight = getattr(config, 'FEATURE_SELECTION_QUALITY_WEIGHT', 2.0)
    confidence_weight = getattr(config, 'FEATURE_SELECTION_CONFIDENCE_WEIGHT', 1.2)
    diversity_weight = getattr(config, 'FEATURE_SELECTION_DIVERSITY_WEIGHT', 0.3)
    reinforcement_weight = getattr(config, 'FEATURE_SELECTION_REINFORCEMENT_WEIGHT', 0.5)
    sparsity_weight = getattr(config, 'FEATURE_SELECTION_SPARSITY_WEIGHT', 0.1)
    spectral_diversity_weight = getattr(config, 'FEATURE_SELECTION_SPECTRAL_DIVERSITY_WEIGHT', 0.2)
    
    # Apply hierarchical stage weights if available
    if hierarchical_weights:
        quality_weight = hierarchical_weights.get('quality_weight', quality_weight)
        confidence_weight = hierarchical_weights.get('confidence_weight', confidence_weight)
        diversity_weight = hierarchical_weights.get('diversity_weight', diversity_weight)
        reinforcement_weight = hierarchical_weights.get('reinforcement_weight', reinforcement_weight)
        sparsity_weight = hierarchical_weights.get('sparsity_weight', sparsity_weight)
        spectral_diversity_weight = hierarchical_weights.get('spectral_diversity_weight', spectral_diversity_weight)
    
    # Adaptive weighting based on epoch (if not overridden by hierarchical)
    elif epoch is not None and epoch < 30:
        # Early training: Focus on exploration
        reinforcement_weight *= 0.2  # Lower reinforcement initially
    else:
        # Later training: Focus on performance
        reinforcement_weight *= 1.5  # Higher reinforcement later
    
    # Combine all losses
    total_loss = (
        quality_weight * quality_alignment_loss +
        confidence_weight * confidence_loss +
        diversity_weight * diversity_loss +
        reinforcement_weight * gate_reinforcement_loss +
        sparsity_weight * sparsity_loss +
        spectral_diversity_weight * spectral_region_loss
    )
    
    return total_loss
