# Models/MultiHeadCNNModel.py

from typing import Optional, List, Tuple, Dict
import tensorflow as tf
from keras import Input, Model
import numpy as np
import os
import tensorflow as tf
from keras import Input, Model
from keras._tf_keras.keras.models import Sequential, load_model as keras_load_model
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, Lambda
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.optimizers import AdamW
from keras._tf_keras.keras.optimizers.schedules import ExponentialDecay
from src.utils.data_processing import DataProcessingUtils
from src.utils.spectral_indexes import SpectralIndexCalculator
import src.config as config
from src.config.enums import ModelType, SpectralIndex
from ..base_classes.BaseDLModel import BaseDLModel

class MultiHeadCNNModel(BaseDLModel):
    """
    MultiHeadCNNModel that also creates a *separate* model for each attribute.
    The difference from CNNModel is in the architecture (_build_model_for_attr) 
    where multiple "heads" handle reflectance / NDSI / indexes separately
    before merging. Then a single output layer for that attribute.
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
        super().__init__(
            model_type=ModelType.MULTI_HEAD_CNN,
            model_name=model_name,
            model_filename=model_filename,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            selected_bands=selected_bands,
            predicted_attributes=predicted_attributes,
            selected_indexes=selected_indexes
        )
        
        # Store Multi-Head CNN specific parameters for depth calculation
        self.num_heads = 3  # Number of CNN heads
        self.layers_per_head = 3  # Number of layers per head

    # Removed hardcoded get_architecture_config - now using dynamic extraction


    def _build_model_for_attr(self) -> tf.keras.Model:
        """
        Build a single-output multi-head CNN pipeline for a single attribute.
        We slice out reflectance channels, std channels, ndsi channels, indexes channels, 
        pass them through separate sub-conv blocks, then merge.
        """
        H, W, C = self.model_shape

        # Use the component_dimensions directly instead of calculating them
        reflectance_count = self.component_dimensions['reflectance']
        special_std_count = self.component_dimensions['std']
        ndsi_count = self.component_dimensions['ndsi']
        indexes_count = self.component_dimensions['indexes']
        
        # Verify the total matches our model shape's channel dimension
        if self.component_dimensions['total'] != C:
            print(f"Warning: Component dimensions total ({self.component_dimensions['total']}) doesn't match model shape channels ({C})")
            raise ValueError("Component dimensions total doesn't match model shape channels")

        # Print component info for debugging
        print(f"Building model with: reflectance({reflectance_count}), std({special_std_count}), ndsi({ndsi_count}), indexes({indexes_count})")
        
        # Input layer
        input_layer = Input(shape=(H, W, C), name="input")

        # We define lambdas to slice each portion
        heads = []

        def build_head_block(input_tensor, name_prefix, channels):
            print(f"Building {name_prefix} head with {channels} channels")
            x = Conv2D(32, (3, 3), activation='relu', name=f"{name_prefix}_conv1")(input_tensor)
            x = MaxPooling2D((2, 2), name=f"{name_prefix}_pool1")(x)
            x = Conv2D(64, (3, 3), activation='relu', name=f"{name_prefix}_conv2")(x)
            x = MaxPooling2D((2, 2), name=f"{name_prefix}_pool2")(x)
            x = Flatten(name=f"{name_prefix}_flatten")(x)
            return x

        # Calculate starting index for each component
        reflectance_start = 0
        special_std_start = reflectance_start + reflectance_count
        ndsi_start = special_std_start + special_std_count
        indexes_start = ndsi_start + ndsi_count
        
        # Reflectance head - only create if we have channels
        if reflectance_count > 0:
            reflectance_slice = Lambda(
                lambda z: z[:, :, :, reflectance_start:reflectance_start+reflectance_count], 
                name="reflectance_slice"
            )(input_layer)
            reflectance_out = build_head_block(reflectance_slice, "reflectance_head", reflectance_count)
            heads.append(reflectance_out)

        # Special STD head - only create if we have channels
        if special_std_count > 0:
            std_slice = Lambda(
                lambda z: z[:, :, :, special_std_start:special_std_start+special_std_count], 
                name="std_slice"
            )(input_layer)
            std_out = build_head_block(std_slice, "std_head", special_std_count)
            heads.append(std_out)
            
        # NDSI head - only create if we have channels
        if ndsi_count > 0:
            ndsi_slice = Lambda(
                lambda z: z[:, :, :, ndsi_start:ndsi_start+ndsi_count],
                name="ndsi_slice"
            )(input_layer)
            ndsi_out = build_head_block(ndsi_slice, "ndsi_head", ndsi_count)
            heads.append(ndsi_out)

        # Indexes head - only create if we have channels
        if indexes_count > 0:
            indexes_slice = Lambda(
                lambda z: z[:, :, :, indexes_start:indexes_start+indexes_count],
                name="indexes_slice"
            )(input_layer)
            indexes_out = build_head_block(indexes_slice, "indexes_head", indexes_count)
            heads.append(indexes_out)

        # Safety check - we need at least one head
        if not heads:
            raise ValueError("No valid heads were created. Please enable at least one component with data.")
            
        if len(heads) > 1:
            x = Concatenate(name="heads_concatenate")(heads)
        else:
            x = heads[0]  # only one

        # Final Dense for a single attribute
        x = Dense(256, activation='relu', name="final_dense1")(x)
        x = Dropout(0.3, name="final_dropout")(x)
        output = Dense(1, activation='linear', name="final_output")(x)

        model = Model(inputs=input_layer, outputs=output, name="MultiHeadCNN_SingleAttr")
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _build_multi_output_model(self) -> tf.keras.Model:
        """
        Build a multi-output multi-head CNN architecture for all quality attributes.
        """
        H, W, C = self.model_shape
        input_layer = Input(shape=(H, W, C), name="input")

        def build_head_block(input_tensor, name_prefix, channels):
            """Build a single head block."""
            x = Conv2D(channels, (3, 3), activation='relu', padding='same', name=f"{name_prefix}_conv1")(input_tensor)
            x = MaxPooling2D((2, 2), name=f"{name_prefix}_pool1")(x)
            x = Conv2D(channels * 2, (3, 3), activation='relu', padding='same', name=f"{name_prefix}_conv2")(x)
            x = MaxPooling2D((2, 2), name=f"{name_prefix}_pool2")(x)
            x = Conv2D(channels * 4, (3, 3), activation='relu', padding='same', name=f"{name_prefix}_conv3")(x)
            x = MaxPooling2D((2, 2), name=f"{name_prefix}_pool3")(x)
            x = Flatten(name=f"{name_prefix}_flatten")(x)
            x = Dense(128, activation='relu', name=f"{name_prefix}_dense")(x)
            return x

        # Create multiple heads
        heads = []
        head_configs = [
            ("head1", 16),
            ("head2", 32),
            ("head3", 64)
        ]
        
        for head_name, channels in head_configs:
            head_output = build_head_block(input_layer, head_name, channels)
            heads.append(head_output)
            
        if len(heads) > 1:
            x = Concatenate(name="heads_concatenate")(heads)
        else:
            x = heads[0]  # only one

        # Final Dense for multi-output
        x = Dense(256, activation='relu', name="final_dense1")(x)
        x = Dropout(0.3, name="final_dropout")(x)
        
        # Multi-output layer - one output for all attributes
        outputs = Dense(len(self.predicted_attributes), activation='linear', name='multi_output')(x)

        model = Model(inputs=input_layer, outputs=outputs, name="MultiHeadCNNMultiRegression")
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print(f"[MultiHeadCNNModel] Built multi-output model with {len(self.predicted_attributes)} outputs")
        return model