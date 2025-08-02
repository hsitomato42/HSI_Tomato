# feature_selection/implementations/attention_based/utils/feature_computation.py

import tensorflow as tf
from typing import Tuple, List
import src.config as config
from src.config.enums import NDSISelectionStrategy


def compute_std_maps(selected_bands: tf.Tensor, num_maps: int) -> tf.Tensor:
    """
    Compute STD maps from selected bands using spatial information.
    
    Args:
        selected_bands: (batch_size, height, width, k_bands) - the selected reflectance bands
        num_maps: number of STD maps to generate
        
    Returns:
        std_maps: (batch_size, height, width, num_maps)
    """
    batch_size = tf.shape(selected_bands)[0]
    height = tf.shape(selected_bands)[1]
    width = tf.shape(selected_bands)[2]
    k_bands = tf.shape(selected_bands)[3]
    
    # Kernel for local standard deviation computation
    kernel_size = 9  # Fixed kernel size for local texture
    kernel = tf.ones((kernel_size, kernel_size, 1, 1)) / tf.cast(kernel_size * kernel_size, tf.float32)
    
    std_maps_list = []
    
    # Generate STD maps from available selected bands
    for map_idx in range(min(num_maps, k_bands)):
        # Get the band for this STD map
        band_image = selected_bands[:, :, :, map_idx % k_bands]  # (batch, height, width)
        band_expanded = tf.expand_dims(band_image, -1)  # (batch, height, width, 1)
        
        # Compute local mean using convolution
        local_mean = tf.nn.conv2d(band_expanded, kernel, strides=1, padding='SAME')
        local_mean = tf.squeeze(local_mean, -1)  # (batch, height, width)
        
        # Compute local variance
        squared_diff = tf.square(band_image - local_mean)
        squared_diff_expanded = tf.expand_dims(squared_diff, -1)
        local_var = tf.nn.conv2d(squared_diff_expanded, kernel, strides=1, padding='SAME')
        local_std = tf.sqrt(tf.squeeze(local_var, -1) + 1e-8)
        
        # Apply non-linear transformation for better texture representation
        enhanced_std = tf.nn.tanh(local_std * 2.0)  # Scale and saturate
        
        std_maps_list.append(enhanced_std)
    
    # Pad with zeros if needed
    while len(std_maps_list) < num_maps:
        zero_map = tf.zeros((batch_size, height, width), dtype=tf.float32)
        std_maps_list.append(zero_map)
    
    # Stack STD maps: (batch_size, height, width, num_maps)
    std_maps = tf.stack(std_maps_list[:num_maps], axis=-1)
    
    return std_maps


def compute_ndsi_maps(selected_bands: tf.Tensor, num_pairs: int) -> tf.Tensor:
    """
    Compute NDSI maps from selected reflectance bands using different strategies.
    
    Args:
        selected_bands: (batch_size, height, width, k_bands)
        num_pairs: number of NDSI pairs to compute
        
    Returns:
        ndsi_maps: (batch_size, height, width, num_pairs)
    """
    k_bands = tf.shape(selected_bands)[3]
    
    # Get NDSI selection strategy from config
    ndsi_strategy = getattr(config, 'NDSI_SELECTION_STRATEGY', NDSISelectionStrategy.IMPORTANCE_BASED)
    
    ndsi_maps_list = []
    
    if ndsi_strategy == NDSISelectionStrategy.BRUTE_FORCE:
        # Generate all unique pairs where left band index < right band index
        all_pairs = []
        k_bands_np = k_bands.numpy() if hasattr(k_bands, 'numpy') else int(k_bands)
        
        for i in range(k_bands_np):
            for j in range(i + 1, k_bands_np):
                all_pairs.append((i, j))
        
        # Use the first num_pairs pairs (or all if fewer pairs available)
        pairs_to_use = all_pairs[:num_pairs]
        
        for pair_idx, (i, j) in enumerate(pairs_to_use):
            band_i = selected_bands[:, :, :, i]
            band_j = selected_bands[:, :, :, j]
            
            # Compute NDSI: (band_i - band_j) / (band_i + band_j)
            numerator = band_i - band_j
            denominator = band_i + band_j + 1e-8  # Avoid division by zero
            
            ndsi = numerator / denominator
            
            # Apply mask for zero pixels
            mask = tf.logical_and(tf.equal(band_i, 0), tf.equal(band_j, 0))
            ndsi = tf.where(mask, 0.0, ndsi)
            
            ndsi_maps_list.append(ndsi)
        
        # Pad with zeros if we need more pairs than available
        while len(ndsi_maps_list) < num_pairs:
            if len(all_pairs) > 0:
                # Repeat pairs if needed
                i, j = all_pairs[len(ndsi_maps_list) % len(all_pairs)]
                band_i = selected_bands[:, :, :, i]
                band_j = selected_bands[:, :, :, j]
                
                numerator = band_i - band_j
                denominator = band_i + band_j + 1e-8
                ndsi = numerator / denominator
                mask = tf.logical_and(tf.equal(band_i, 0), tf.equal(band_j, 0))
                ndsi = tf.where(mask, 0.0, ndsi)
                
                ndsi_maps_list.append(ndsi)
            else:
                # Fallback: zero map
                zero_map = tf.zeros_like(selected_bands[:, :, :, 0])
                ndsi_maps_list.append(zero_map)
    
    else:
        # Importance-Based Strategy (Default): Sequential band combinations
        for i in range(num_pairs):
            # Create pairs from selected bands
            if k_bands >= 2:
                # Use different band combinations for each pair
                band_i_idx = i % k_bands
                band_j_idx = (i + 1) % k_bands
                
                band_i = selected_bands[:, :, :, band_i_idx]
                band_j = selected_bands[:, :, :, band_j_idx]
            else:
                # If only one band, use it for both (degenerate case)
                band_i = selected_bands[:, :, :, 0]
                band_j = selected_bands[:, :, :, 0]
            
            # Compute NDSI: (band_i - band_j) / (band_i + band_j)
            numerator = band_i - band_j
            denominator = band_i + band_j + 1e-8  # Avoid division by zero
            
            ndsi = numerator / denominator
            
            # Apply mask for zero pixels
            mask = tf.logical_and(tf.equal(band_i, 0), tf.equal(band_j, 0))
            ndsi = tf.where(mask, 0.0, ndsi)
            
            ndsi_maps_list.append(ndsi)
    
    # Stack NDSI maps
    ndsi_maps = tf.stack(ndsi_maps_list, axis=-1)
    
    return ndsi_maps


def compute_index_maps_simple(
    selected_bands: tf.Tensor, 
    selected_indices: tf.Tensor,
    num_indexes: int
) -> tf.Tensor:
    """
    Simple fallback index computation for backward compatibility.
    
    Args:
        selected_bands: (batch_size, height, width, k_bands) - Only selected bands
        selected_indices: (batch_size, k_bands) indices of selected bands
        num_indexes: number of index maps to compute
        
    Returns:
        index_maps: (batch_size, height, width, num_indexes)
    """
    batch_size = tf.shape(selected_bands)[0]
    height = tf.shape(selected_bands)[1]
    width = tf.shape(selected_bands)[2]
    k_bands = tf.shape(selected_bands)[3]
    
    index_maps_list = []
    
    # Simple fallback: create basic index combinations
    for i in range(num_indexes):
        if k_bands >= 2:
            # Use different band pairs for different indexes
            band_a = selected_bands[:, :, :, i % k_bands]
            band_b = selected_bands[:, :, :, (i + 1) % k_bands]
            
            # Create normalized difference index
            index_map = (band_a - band_b) / (band_a + band_b + 1e-8)
        else:
            # Single band case
            index_map = selected_bands[:, :, :, 0]
        
        index_maps_list.append(index_map)
    
    # Stack all index maps
    index_maps = tf.stack(index_maps_list, axis=-1)
    
    return index_maps
