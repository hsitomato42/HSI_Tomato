# Models/dl_models/CNNTransformerModel.py

from typing import Optional, List, Tuple, Dict
import tensorflow as tf
from keras import Input, Model
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, Lambda, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Reshape
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.models import load_model as keras_load_model
import numpy as np
import os

from ..base_classes.BaseDLModel import BaseDLModel  # Import BaseDLModel instead of CNNModel
from src.utils.PositionalEncoding import PositionalEncoding
from src.utils.data_processing import DataProcessingUtils
from src.utils.spectral_indexes import SpectralIndexCalculator
import src.config as config
from src.config.enums import ModelType, SpectralIndex

class CNNTransformerModel(BaseDLModel):
    """
    CNNTransformerModel that now has a separate Transformer for each attribute 
    (i.e., each attribute's model is built individually).
    """

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
        # Inherit from BaseDLModel 
        super().__init__(
            model_type=ModelType.CNN_TRANSFORMER,
            model_name=model_name,
            model_filename=model_filename,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            selected_bands=selected_bands,
            predicted_attributes=predicted_attributes,
            selected_indexes=selected_indexes
        )
        
        # Store CNN+Transformer specific parameters for depth calculation
        self.cnn_layers = 3  # Number of CNN layers
        self.transformer_layers = 1  # Number of transformer layers

    # Removed hardcoded get_architecture_config - now using dynamic extraction


    def _build_model_for_attr(self) -> tf.keras.Model:
        """
        Build a single-output CNN+Transformer architecture for a single attribute.
        """
        H, W, C = self.model_shape

        input_layer = Input(shape=(H, W, C), name="input")

        # CNN feature extractor
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name="cnn_conv1")(input_layer)
        x = MaxPooling2D((2, 2), name="cnn_pool1")(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name="cnn_conv2")(x)
        x = MaxPooling2D((2, 2), name="cnn_pool2")(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name="cnn_conv3")(x)
        x = MaxPooling2D((2, 2), name="cnn_pool3")(x)

        # Flatten spatial dimension into a sequence
        shape_after_pool = x.shape[1] * x.shape[2]  # (H/8)*(W/8) if 3 pooling layers
        C_pooled = x.shape[3]  # channels after final conv
        x = Reshape((shape_after_pool, C_pooled), name="reshape_for_transformer")(x)

        # Positional Encoding
        x = PositionalEncoding(sequence_length=shape_after_pool, d_model=C_pooled)(x)

        # Transformer
        transformer_layer = MultiHeadAttention(
            num_heads=8,
            key_dim=C_pooled // 8,
            name="transformer_mha"
        )(x, x)
        transformer_layer = LayerNormalization(epsilon=1e-6, name="transformer_layernorm1")(x + transformer_layer)

        transformer_ffn = Dense(256, activation='relu', name="transformer_ffn1")(transformer_layer)
        transformer_ffn = Dense(C_pooled, name="transformer_ffn2")(transformer_ffn)
        transformer_layer = LayerNormalization(epsilon=1e-6, name="transformer_layernorm2")(transformer_layer + transformer_ffn)

        # Global Average Pool
        x = GlobalAveragePooling1D(name="transformer_global_avg_pool")(transformer_layer)

        # Final dense layers for single attribute
        x = Dense(256, activation='relu', name="final_dense1")(x)
        x = Dropout(0.3, name="final_dropout")(x)
        output_layer = Dense(1, activation='linear', name="final_output")(x)

        model = Model(inputs=input_layer, outputs=output_layer, name="CNNTransformer_SingleAttr")
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _build_multi_output_model(self) -> tf.keras.Model:
        """
        Build a multi-output CNN+Transformer architecture for all quality attributes.
        """
        H, W, C = self.model_shape

        input_layer = Input(shape=(H, W, C), name="input")

        # CNN feature extractor
        x = Conv2D(32, (3, 3), activation='relu', padding='same', name="cnn_conv1")(input_layer)
        x = MaxPooling2D((2, 2), name="cnn_pool1")(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name="cnn_conv2")(x)
        x = MaxPooling2D((2, 2), name="cnn_pool2")(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name="cnn_conv3")(x)
        x = MaxPooling2D((2, 2), name="cnn_pool3")(x)

        # Flatten spatial dimension into a sequence
        shape_after_pool = x.shape[1] * x.shape[2]  # (H/8)*(W/8) if 3 pooling layers
        C_pooled = x.shape[3]  # channels after final conv
        x = Reshape((shape_after_pool, C_pooled), name="reshape_for_transformer")(x)

        # Positional Encoding
        x = PositionalEncoding(sequence_length=shape_after_pool, d_model=C_pooled)(x)

        # Transformer
        transformer_layer = MultiHeadAttention(
            num_heads=8,
            key_dim=C_pooled // 8,
            name="transformer_mha"
        )(x, x)
        transformer_layer = LayerNormalization(epsilon=1e-6, name="transformer_layernorm1")(x + transformer_layer)

        transformer_ffn = Dense(256, activation='relu', name="transformer_ffn1")(transformer_layer)
        transformer_ffn = Dense(C_pooled, name="transformer_ffn2")(transformer_ffn)
        transformer_layer = LayerNormalization(epsilon=1e-6, name="transformer_layernorm2")(transformer_layer + transformer_ffn)

        # Global Average Pool
        x = GlobalAveragePooling1D(name="transformer_global_avg_pool")(transformer_layer)

        # Final dense layers for multi-output
        x = Dense(256, activation='relu', name="final_dense1")(x)
        x = Dropout(0.3, name="final_dropout")(x)
        
        # Multi-output layer - one output for all attributes
        outputs = Dense(len(self.predicted_attributes), activation='linear', name='multi_output')(x)

        model = Model(inputs=input_layer, outputs=outputs, name="CNNTransformerMultiRegression")
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print(f"[CNNTransformerModel] Built multi-output model with {len(self.predicted_attributes)} outputs")
        return model

    # Inherit train, test, save_model, load_model from BaseDLModel
