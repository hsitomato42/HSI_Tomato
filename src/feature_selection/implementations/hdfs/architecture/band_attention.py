# feature_selection/band_attention.py

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from typing import Dict, List, Tuple, Optional
import src.config as config


class MultiHeadBandAttention(layers.Layer):
    """
    Multi-Head Self-Attention mechanism for spectral band analysis.
    
    This module treats each spectral band as a token and uses transformer-style
    self-attention to:
    1. Evaluate band importance for the prediction task
    2. Identify redundant/correlated bands
    3. Capture complex inter-band relationships
    4. Generate diversity-aware importance scores
    """
    
    def __init__(
        self,
        num_bands: int = 204,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dff: int = 256,
        dropout_rate: float = 0.1,
        use_spatial_features: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_bands = num_bands
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dff = dff
        self.dropout_rate = dropout_rate
        self.use_spatial_features = use_spatial_features
        
        # Ensure d_model is divisible by num_heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.head_dim = d_model // num_heads
        
        # Band embedding layer - converts band statistics to embeddings
        self.band_embedding = layers.Dense(d_model, name="band_embedding")
        
        # Positional embedding for bands (wavelength-based)
        self.positional_embedding = self._create_positional_embedding()
        
        # Multi-head self-attention layers
        self.attention_layers = []
        self.norm_layers = []
        self.ffn_layers = []
        self.ffn_norm_layers = []
        
        for i in range(num_layers):
            # Self-attention
            self.attention_layers.append(
                layers.MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=self.head_dim,
                    dropout=dropout_rate,
                    name=f"band_attention_layer_{i}"
                )
            )
            self.norm_layers.append(
                layers.LayerNormalization(name=f"attention_norm_{i}")
            )
            
            # Feed-forward network
            ffn = keras.Sequential([
                layers.Dense(dff, activation='relu', name=f"ffn_dense1_{i}"),
                layers.Dropout(dropout_rate),
                layers.Dense(d_model, name=f"ffn_dense2_{i}"),
                layers.Dropout(dropout_rate)
            ], name=f"ffn_{i}")
            
            self.ffn_layers.append(ffn)
            self.ffn_norm_layers.append(
                layers.LayerNormalization(name=f"ffn_norm_{i}")
            )
        
        # Final projection to importance scores
        self.importance_projection = layers.Dense(1, name="importance_projection")
        
        # Diversity analysis projection (for redundancy detection)
        self.diversity_projection = layers.Dense(self.head_dim, name="diversity_projection")
    
    def _create_positional_embedding(self) -> tf.Variable:
        """
        Create wavelength-based positional embeddings for spectral bands.
        Uses sinusoidal embeddings based on wavelength positions.
        """
        # Approximate wavelength range: 400-1000 nm for 204 bands
        wavelengths = np.linspace(400, 1000, self.num_bands)
        
        # Normalize wavelengths to [0, 1]  
        norm_wavelengths = (wavelengths - wavelengths.min()) / (wavelengths.max() - wavelengths.min())
        
        # Create sinusoidal positional embeddings
        position = norm_wavelengths.reshape(-1, 1)  # (num_bands, 1)
        
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pos_embedding = np.zeros((self.num_bands, self.d_model))
        pos_embedding[:, 0::2] = np.sin(position * div_term)
        pos_embedding[:, 1::2] = np.cos(position * div_term)
        
        return tf.Variable(
            pos_embedding.astype(np.float32),
            trainable=False,
            name="wavelength_positional_embedding"
        )
    
    def _extract_band_features(self, hyperspectral_cube: tf.Tensor) -> tf.Tensor:
        """
        Extract features for each spectral band using VECTORIZED operations.
        
        MEMORY OPTIMIZATION: Replaced individual band processing loop with vectorized operations
        to reduce memory usage from ~500MB to ~50MB.
        
        Args:
            hyperspectral_cube: (batch_size, height, width, num_bands)
            
        Returns:
            band_features: (batch_size, num_bands, feature_dim)
        """
        batch_size = tf.shape(hyperspectral_cube)[0]
        height = tf.shape(hyperspectral_cube)[1]
        width = tf.shape(hyperspectral_cube)[2]
        
        # VECTORIZED global statistics (instead of per-band loop)
        band_mean = tf.reduce_mean(hyperspectral_cube, axis=[1, 2])  # (batch, num_bands)
        band_std = tf.math.reduce_std(hyperspectral_cube, axis=[1, 2])  # (batch, num_bands)
        band_max = tf.reduce_max(hyperspectral_cube, axis=[1, 2])  # (batch, num_bands)
        band_min = tf.reduce_min(hyperspectral_cube, axis=[1, 2])  # (batch, num_bands)
        
        # Start with basic features
        features_list = [band_mean, band_std, band_max, band_min]
        
        if self.use_spatial_features:
            # VECTORIZED spatial features (process all bands simultaneously)
            kernel_size = config.FEATURE_SELECTION_SPATIAL_KERNEL
            
            # Create convolution kernel for all bands
            kernel = tf.ones((kernel_size, kernel_size, 1, 1)) / (kernel_size * kernel_size)
            
            # Process all bands simultaneously using grouped convolution
            # Reshape for grouped convolution: (batch, height, width, num_bands)
            reshaped_cube = hyperspectral_cube  # Already in correct shape
            
            # VECTORIZED local mean computation
            local_means = tf.nn.depthwise_conv2d(
                reshaped_cube, 
                tf.tile(kernel, [1, 1, self.num_bands, 1]),
                strides=[1, 1, 1, 1], 
                padding='SAME'
            )
            
            # VECTORIZED local variance computation
            squared_diff = tf.square(reshaped_cube - local_means)
            local_vars = tf.nn.depthwise_conv2d(
                squared_diff,
                tf.tile(kernel, [1, 1, self.num_bands, 1]),
                strides=[1, 1, 1, 1],
                padding='SAME'
            )
            local_stds = tf.sqrt(local_vars + 1e-8)
            
            # VECTORIZED spatial texture statistics
            spatial_texture_mean = tf.reduce_mean(local_stds, axis=[1, 2])  # (batch, num_bands)
            spatial_texture_std = tf.math.reduce_std(local_stds, axis=[1, 2])  # (batch, num_bands)
            
            # VECTORIZED edge detection
            # Compute gradients for all bands simultaneously
            grad_x = hyperspectral_cube[:, :, 1:, :] - hyperspectral_cube[:, :, :-1, :]
            grad_y = hyperspectral_cube[:, 1:, :, :] - hyperspectral_cube[:, :-1, :, :]
            
            # Pad to maintain dimensions
            grad_x_padded = tf.pad(grad_x, [[0, 0], [0, 0], [0, 1], [0, 0]], mode='CONSTANT')
            grad_y_padded = tf.pad(grad_y, [[0, 0], [0, 1], [0, 0], [0, 0]], mode='CONSTANT')
            
            # VECTORIZED gradient magnitude
            gradient_magnitude = tf.sqrt(tf.square(grad_x_padded) + tf.square(grad_y_padded) + 1e-8)
            edge_strength = tf.reduce_mean(gradient_magnitude, axis=[1, 2])  # (batch, num_bands)
            
            features_list.extend([spatial_texture_mean, spatial_texture_std, edge_strength])
        
        # Stack all features: (batch, num_bands, num_features)
        band_features = tf.stack(features_list, axis=2)
        
        return band_features
    
    def call(self, hyperspectral_cube: tf.Tensor, training: Optional[bool] = None) -> Dict[str, tf.Tensor]:
        """
        Forward pass of the multi-head band attention.
        
        Args:
            hyperspectral_cube: (batch_size, height, width, num_bands)
            training: Training mode flag
            
        Returns:
            Dict containing:
            - band_importance: (batch_size, num_bands) - importance scores
            - attention_weights: (batch_size, num_heads, num_bands, num_bands) - attention matrices
            - band_embeddings: (batch_size, num_bands, d_model) - final band representations
            - diversity_scores: (batch_size, num_bands, head_dim) - for redundancy analysis
        """
        batch_size = tf.shape(hyperspectral_cube)[0]
        
        # Extract band features
        band_features = self._extract_band_features(hyperspectral_cube)  # (batch, num_bands, feature_dim)
        
        # Project to d_model dimension
        band_embeddings = self.band_embedding(band_features)  # (batch, num_bands, d_model)
        
        # Add positional embeddings (wavelength information)
        band_embeddings = band_embeddings + self.positional_embedding[tf.newaxis, :, :]
        
        # Apply transformer layers
        attention_weights_list = []
        
        for i in range(self.num_layers):
            # Self-attention
            attended_output, attention_weights = self.attention_layers[i](
                band_embeddings, band_embeddings, return_attention_scores=True, training=training
            )
            attention_weights_list.append(attention_weights)
            
            # Residual connection and normalization
            band_embeddings = self.norm_layers[i](band_embeddings + attended_output)
            
            # Feed-forward network
            ffn_output = self.ffn_layers[i](band_embeddings, training=training)
            
            # Residual connection and normalization
            band_embeddings = self.ffn_norm_layers[i](band_embeddings + ffn_output)
        
        # Generate importance scores
        importance_scores = self.importance_projection(band_embeddings)  # (batch, num_bands, 1)
        importance_scores = tf.squeeze(importance_scores, -1)  # (batch, num_bands)
        
        # Apply softmax to get probability distribution over bands
        band_importance = tf.nn.softmax(importance_scores, axis=-1)
        
        # Generate diversity features for redundancy analysis
        diversity_features = self.diversity_projection(band_embeddings)  # (batch, num_bands, head_dim)
        
        # Stack attention weights from all layers
        stacked_attention = tf.stack(attention_weights_list, axis=1)  # (batch, num_layers, num_heads, num_bands, num_bands)
        
        # Average across layers for final attention weights
        avg_attention_weights = tf.reduce_mean(stacked_attention, axis=1)  # (batch, num_heads, num_bands, num_bands)
        
        return {
            'band_importance': band_importance,
            'attention_weights': avg_attention_weights,
            'band_embeddings': band_embeddings,
            'diversity_scores': diversity_features,
            'raw_importance_scores': importance_scores
        }
    
    def compute_diversity_loss(self, diversity_scores: tf.Tensor, selected_band_mask: tf.Tensor) -> tf.Tensor:
        """
        Compute diversity loss to encourage selection of non-redundant bands.
        
        Args:
            diversity_scores: (batch_size, num_bands, head_dim)
            selected_band_mask: (batch_size, num_bands) - binary mask for selected bands
            
        Returns:
            diversity_loss: scalar tensor
        """
        batch_size = tf.shape(diversity_scores)[0]
        
        # Get diversity features only for selected bands
        selected_features = diversity_scores * tf.expand_dims(selected_band_mask, -1)
        
        # Compute pairwise similarities between selected bands
        # Normalize features
        normalized_features = tf.nn.l2_normalize(selected_features, axis=-1)
        
        # Compute similarity matrix: (batch, num_bands, num_bands)
        similarity_matrix = tf.matmul(normalized_features, normalized_features, transpose_b=True)
        
        # Create mask for selected band pairs
        selected_mask_expanded = tf.expand_dims(selected_band_mask, -1) * tf.expand_dims(selected_band_mask, -2)
        
        # Zero out diagonal (self-similarity)
        eye_mask = 1.0 - tf.eye(self.num_bands, batch_shape=[batch_size])
        selected_mask_expanded = selected_mask_expanded * eye_mask
        
        # Compute diversity loss (penalize high similarities between selected bands)
        diversity_loss = tf.reduce_sum(similarity_matrix * selected_mask_expanded) / (
            tf.reduce_sum(selected_mask_expanded) + 1e-8
        )
        
        return diversity_loss
    
    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({
            'num_bands': self.num_bands,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'dff': self.dff,
            'dropout_rate': self.dropout_rate,
            'use_spatial_features': self.use_spatial_features,
        })
        return config_dict 