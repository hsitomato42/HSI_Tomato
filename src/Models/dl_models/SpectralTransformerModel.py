# Models/dl_models/SpectralTransformerModel.py

from typing import Optional, List, Tuple, Dict
import tensorflow as tf
from keras import Input, Model
from keras._tf_keras.keras.layers import Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Reshape, Lambda
from keras._tf_keras.keras.models import load_model as keras_load_model
import numpy as np
import os

from ..base_classes.BaseDLModel import BaseDLModel  # Inherit from BaseDLModel
from src.utils.PositionalEncoding import PositionalEncodingForSpectral  # Reuse the custom layer
from src.utils.data_processing import DataProcessingUtils
from src.utils.spectral_indexes import SpectralIndexCalculator
import src.config as config
from src.config.enums import ModelType, SpectralIndex

class SpectralTransformerModel(BaseDLModel):
    def __init__(
        self,
        selected_bands: List[int],
        model_name: str,
        model_filename: str,
        selected_indexes: Optional[List[SpectralIndex]],
        model_shape: Tuple[int, int, int],
        components: dict,
        component_dimensions: dict,
        predicted_attributes: List[str]
    ):
        # Inherit from BaseDLModel for multi-attribute dict logic
        super().__init__(
            model_type=ModelType.SPECTRAL_TRANSFORMER,
            model_name=model_name,
            model_filename=model_filename,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            selected_bands=selected_bands,
            predicted_attributes=predicted_attributes,
            selected_indexes=selected_indexes
        )
        
        # Store Spectral Transformer specific parameters for depth calculation
        self.num_transformer_layers = 2  # Number of transformer encoder layers

    # Removed hardcoded get_architecture_config - now using dynamic extraction


    def build_model(self) -> tf.keras.Model:
        H, W, C = self.model_shape
        print(f"Building SpectralTransformerModel with H={H}, W={W}, C={C}")

        # Input layer
        input_layer = Input(shape=(H, W, C), name="input")  # (batch_size, H, W, C)

        # Reshape to (batch_size, H*W, C)
        x = Reshape((H * W, C), name="reshape_spectral_sequences")(input_layer)  # (batch_size, 2025, 15)
        print(f"After Reshape to (H*W, C): {x.shape}")

        # Apply Positional Encoding
        pos_encoding_layer = PositionalEncodingForSpectral(d_model=C, max_len=H * W)
        x = pos_encoding_layer(x)  # (batch_size, 2025, 15)
        print(f"After PositionalEncoding: {x.shape}")

        # Determine Transformer Parameters
        if C % 4 == 0:
            num_heads = 4
            key_dim = C // num_heads
        else:
            # Adjust num_heads to ensure 'key_dim' is an integer
            for nh in range(8, 0, -1):
                if C % nh == 0:
                    num_heads = nh
                    key_dim = C // nh
                    break
            else:
                raise ValueError(f"Cannot find a suitable number of heads for d_model={C}")

        print(f"Transformer Parameters - num_heads: {num_heads}, key_dim: {key_dim}")

        # Transformer Encoder Layers
        transformer_layer = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,  # key_dim should be divisible by num_heads
            name="spectral_transformer_mha"
        )(x, x)  # (batch_size, 2025, 15)
        print(f"After MultiHeadAttention: {transformer_layer.shape}")

        # Apply LayerNormalization and Residual Connection
        transformer_layer = LayerNormalization(
            epsilon=1e-6,
            name="spectral_transformer_layernorm1"
        )(x + transformer_layer)  # (batch_size, 2025, 15)
        print(f"After LayerNormalization1: {transformer_layer.shape}")

        # Feed-Forward Network
        transformer_feedforward = Dense(
            256,
            activation='relu',
            name="spectral_transformer_ffn1"
        )(transformer_layer)  # (batch_size, 2025, 256)
        print(f"After Dense1 (FFN1): {transformer_feedforward.shape}")

        transformer_feedforward = Dense(
            C,
            name="spectral_transformer_ffn2"
        )(transformer_feedforward)  # (batch_size, 2025, 15)
        print(f"After Dense2 (FFN2): {transformer_feedforward.shape}")

        # Apply LayerNormalization and Residual Connection
        transformer_layer = LayerNormalization(
            epsilon=1e-6,
            name="spectral_transformer_layernorm2"
        )(transformer_layer + transformer_feedforward)  # (batch_size, 2025, 15)
        print(f"After LayerNormalization2: {transformer_layer.shape}")

        # Aggregate spatial dimensions using Global Average Pooling
        x = GlobalAveragePooling1D(name="spectral_transformer_global_avg_pool")(transformer_layer)  # (batch_size, 15)
        print(f"After GlobalAveragePooling1D: {x.shape}")

        # Final Dense Layers
        x = Dense(256, activation='relu', name="spectral_final_dense1")(x)  # (batch_size, 256)
        print(f"After Dense1 (Final): {x.shape}")

        x = Dropout(0.5, name="spectral_final_dropout")(x)  # (batch_size, 256)
        print(f"After Dropout: {x.shape}")

        output_layer = Dense(len(self.predicted_attributes), activation='linear', name="spectral_final_output")(x)  # (batch_size, 6)
        print(f"After Output Dense Layer: {output_layer.shape}")

        model = Model(inputs=input_layer, outputs=output_layer, name="SpectralTransformerModel")
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Print model summary for verification
        model.summary()

        # Calculate channel counts
        reflectance_count = len(self.selected_bands) if self.components.get('reflectance', False) else 0
        band_pairs = DataProcessingUtils.generate_band_pairs(self.selected_bands)
        ndsi_count = len(band_pairs) if self.components.get('ndsi', False) else 0
        indexes_count = len(self.selected_indexes) if (self.selected_indexes and self.components.get('indexes', False)) else 0

        print("H, W, C = ", H, W, C)
        print("reflectance_count:", reflectance_count)
        print("ndsi_count:", ndsi_count)
        print("indexes_count:", indexes_count)
        print("Slicing reflectance channels: 0 to", reflectance_count)
        print("Slicing ndsi channels:", reflectance_count, "to", reflectance_count + ndsi_count)
        print("Slicing indexes channels:", reflectance_count + ndsi_count, "to", reflectance_count + ndsi_count + indexes_count)

        # Dummy prediction for verification
        X_dummy = np.random.random((1, H, W, C)).astype(np.float32)
        Y_dummy_pred = model.predict(X_dummy)
        print("Dummy prediction shape:", Y_dummy_pred.shape)
        print("Expected shape:", (1, len(self.predicted_attributes)))

        return model

    def _build_model_for_attr(self) -> tf.keras.Model:
        """
        Build a single-output Spectral Transformer for one quality attribute.
        """
        H, W, C = self.model_shape
        
        # Input layer
        input_layer = Input(shape=(H, W, C), name="input")
        
        # Reshape to (batch_size, H*W, C)
        x = Reshape((H * W, C), name="reshape_spectral_sequences")(input_layer)
        
        # Apply Positional Encoding
        pos_encoding_layer = PositionalEncodingForSpectral(d_model=C, max_len=H * W)
        x = pos_encoding_layer(x)
        
        # Determine Transformer Parameters
        if C % 4 == 0:
            num_heads = 4
            key_dim = C // num_heads
        else:
            # Adjust num_heads to ensure 'key_dim' is an integer
            for nh in range(8, 0, -1):
                if C % nh == 0:
                    num_heads = nh
                    key_dim = C // nh
                    break
            else:
                raise ValueError(f"Cannot find a suitable number of heads for d_model={C}")
        
        # Transformer Encoder Layers
        transformer_layer = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            name="spectral_transformer_mha"
        )(x, x)
        
        # Apply LayerNormalization and Residual Connection
        transformer_layer = LayerNormalization(
            epsilon=1e-6,
            name="spectral_transformer_layernorm1"
        )(x + transformer_layer)
        
        # Feed-Forward Network
        transformer_feedforward = Dense(
            256,
            activation='relu',
            name="spectral_transformer_ffn1"
        )(transformer_layer)
        
        transformer_feedforward = Dense(
            C,
            name="spectral_transformer_ffn2"
        )(transformer_feedforward)
        
        # Apply LayerNormalization and Residual Connection
        transformer_layer = LayerNormalization(
            epsilon=1e-6,
            name="spectral_transformer_layernorm2"
        )(transformer_layer + transformer_feedforward)
        
        # Aggregate spatial dimensions using Global Average Pooling
        x = GlobalAveragePooling1D(name="spectral_transformer_global_avg_pool")(transformer_layer)
        
        # Final Dense Layers
        x = Dense(256, activation='relu', name="spectral_final_dense1")(x)
        x = Dropout(0.5, name="spectral_final_dropout")(x)
        
        # Single output for one attribute
        output_layer = Dense(1, activation='linear', name="spectral_single_output")(x)
        
        model = Model(inputs=input_layer, outputs=output_layer, name="SpectralTransformer_SingleAttr")
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        return model
        
    def _build_multi_output_model(self) -> tf.keras.Model:
        """
        Build a multi-output Spectral Transformer for all quality attributes.
        """
        H, W, C = self.model_shape
        
        # Input layer
        input_layer = Input(shape=(H, W, C), name="input")
        
        # Reshape to (batch_size, H*W, C)
        x = Reshape((H * W, C), name="reshape_spectral_sequences")(input_layer)
        
        # Apply Positional Encoding
        pos_encoding_layer = PositionalEncodingForSpectral(d_model=C, max_len=H * W)
        x = pos_encoding_layer(x)
        
        # Determine Transformer Parameters
        if C % 4 == 0:
            num_heads = 4
            key_dim = C // num_heads
        else:
            # Adjust num_heads to ensure 'key_dim' is an integer
            for nh in range(8, 0, -1):
                if C % nh == 0:
                    num_heads = nh
                    key_dim = C // nh
                    break
            else:
                raise ValueError(f"Cannot find a suitable number of heads for d_model={C}")
        
        # Transformer Encoder Layers
        transformer_layer = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            name="spectral_transformer_mha"
        )(x, x)
        
        # Apply LayerNormalization and Residual Connection
        transformer_layer = LayerNormalization(
            epsilon=1e-6,
            name="spectral_transformer_layernorm1"
        )(x + transformer_layer)
        
        # Feed-Forward Network
        transformer_feedforward = Dense(
            256,
            activation='relu',
            name="spectral_transformer_ffn1"
        )(transformer_layer)
        
        transformer_feedforward = Dense(
            C,
            name="spectral_transformer_ffn2"
        )(transformer_feedforward)
        
        # Apply LayerNormalization and Residual Connection
        transformer_layer = LayerNormalization(
            epsilon=1e-6,
            name="spectral_transformer_layernorm2"
        )(transformer_layer + transformer_feedforward)
        
        # Aggregate spatial dimensions using Global Average Pooling
        x = GlobalAveragePooling1D(name="spectral_transformer_global_avg_pool")(transformer_layer)
        
        # Final Dense Layers
        x = Dense(256, activation='relu', name="spectral_final_dense1")(x)
        x = Dropout(0.5, name="spectral_final_dropout")(x)
        
        # Multi-output layer - one output for all attributes
        output_layer = Dense(len(self.predicted_attributes), activation='linear', name='multi_output')(x)
        
        model = Model(inputs=input_layer, outputs=output_layer, name="SpectralTransformerMultiRegression")
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print(f"[SpectralTransformerModel] Built multi-output model with {len(self.predicted_attributes)} outputs")
        return model

