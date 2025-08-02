# feature_selection/attention_importance.py

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from typing import Optional
import src.config as config


class AttentionImportance(layers.Layer):
    """
    🎯 NEW: Attention-based importance scoring for spectral bands.
    
    This module computes importance scores for each spectral band using
    multi-head attention mechanisms. These scores guide the Gumbel Gates
    in selecting the most informative bands.
    
    Key Features:
    - Multi-head attention for band importance
    - Spatial-spectral feature integration
    - Learnable importance bias
    - Efficient computation
    """
    
    def __init__(
        self,
        num_bands: int = 204,
        num_heads: int = 4,
        d_model: int = 128,
        num_layers: int = 2,
        random_seed: Optional[int] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_bands = num_bands
        self.num_heads = num_heads
        self.d_model = d_model
        self.num_layers = num_layers
        self.random_seed = random_seed or getattr(config, 'RANDOM_STATE', 42)
        
        # Set random seed for reproducible initialization
        if self.random_seed is not None:
            tf.random.set_seed(self.random_seed)
        
        # Input projection to d_model dimensions
        self.input_projection = layers.Dense(
            d_model,
            activation='relu',
            name='input_projection'
        )
        
        # Multi-head attention layers
        self.attention_layers = []
        for i in range(num_layers):
            attention_layer = layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads,
                name=f'attention_layer_{i}'
            )
            self.attention_layers.append(attention_layer)
        
        # Layer normalization
        self.layer_norms = []
        for i in range(num_layers):
            layer_norm = layers.LayerNormalization(name=f'layer_norm_{i}')
            self.layer_norms.append(layer_norm)
        
        # Output projection to importance scores
        self.importance_projection = layers.Dense(
            1,
            activation='sigmoid',  # Importance scores between 0 and 1
            name='importance_projection'
        )
        
        # Global average pooling for spatial aggregation
        self.global_avg_pool = layers.GlobalAveragePooling2D()
        
        # Only print initialization info once to avoid spam during multiple object creation
        if not hasattr(AttentionImportance, '_init_printed'):
            AttentionImportance._init_printed = True
            print(f"[AttentionImportance] Initialized with {num_bands} bands")
            print(f"[AttentionImportance] Architecture: {num_heads} heads, {d_model} d_model, {num_layers} layers")
    
    def call(
        self,
        hyperspectral_cube: tf.Tensor,
        training: Optional[bool] = None
    ) -> tf.Tensor:
        """
        Compute importance scores for each spectral band.
        
        Args:
            hyperspectral_cube: (batch_size, height, width, num_bands)
            training: training mode flag
            
        Returns:
            importance_scores: (batch_size, num_bands) - importance score for each band
        """
        batch_size = tf.shape(hyperspectral_cube)[0]
        height = tf.shape(hyperspectral_cube)[1]
        width = tf.shape(hyperspectral_cube)[2]
        
        # 1. Spatial aggregation: average each band across spatial dimensions
        # (batch_size, height, width, num_bands) -> (batch_size, num_bands)
        band_features = self.global_avg_pool(hyperspectral_cube)
        
        # 2. Project to d_model dimensions
        # (batch_size, num_bands) -> (batch_size, num_bands, d_model)
        band_features_expanded = tf.expand_dims(band_features, -1)  # (batch_size, num_bands, 1)
        
        # Project each band feature to d_model dimensions
        projected_features = self.input_projection(band_features_expanded)  # (batch_size, num_bands, d_model)
        
        # 3. Apply multi-head attention layers
        attention_output = projected_features
        
        for i in range(self.num_layers):
            # Self-attention: bands attend to each other
            attended_features = self.attention_layers[i](
                query=attention_output,
                key=attention_output,
                value=attention_output,
                training=training
            )
            
            # Residual connection and layer normalization
            attention_output = self.layer_norms[i](attention_output + attended_features)
        
        # 4. Compute importance scores
        # (batch_size, num_bands, d_model) -> (batch_size, num_bands, 1) -> (batch_size, num_bands)
        importance_scores = self.importance_projection(attention_output)
        importance_scores = tf.squeeze(importance_scores, -1)
        
        return importance_scores
    
    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({
            'num_bands': self.num_bands,
            'num_heads': self.num_heads,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'random_seed': self.random_seed
        })
        return config_dict 