# Models/dl_models/ViTModel.py

import os
from typing import Optional, List, Dict, Tuple
import numpy as np
import tensorflow as tf
from keras import layers, Model, Input
from keras._tf_keras.keras.models import load_model as keras_load_model
from keras._tf_keras.keras.layers import Dense, Dropout, LayerNormalization, GlobalAveragePooling1D
from src.utils.data_processing import DataProcessingUtils
from src.utils.spectral_indexes import SpectralIndexCalculator
import src.config as config
from src.config.enums import ModelType, SpectralIndex
from ..base_classes.BaseDLModel import BaseDLModel  # Import BaseDLModel instead of CNNModel


class Patches(tf.keras.layers.Layer):
    """
    Extracts non-overlapping patches from the input image.
    We override build() and compute_output_shape() so Keras knows shapes.
    """
    def __init__(self, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

    def build(self, input_shape):
        """
        Called once when this layer is first used. input_shape = (batch_size, H, W, C).
        We don't create any variables here, but we check that H, W, C are not None.
        """
        if (input_shape[1] is None) or (input_shape[2] is None) or (input_shape[3] is None):
            raise ValueError(
                "Patches layer requires a fully-known (H, W, C). "
                f"Got input_shape={input_shape}. H, W, C must not be None."
            )
        super().build(input_shape)

    def call(self, images):
        """
        images: (batch_size, H, W, C).
        Use tf.image.extract_patches(...) to get patches of size (patch_size x patch_size).
        """
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding='VALID'
        )
        # patches: (batch_size, H//patch_size, W//patch_size, patch_size*patch_size*C)
        patch_dims = patches.shape[-1]
        num_patches = (patches.shape[1] * patches.shape[2])
        # Reshape to (batch_size, num_patches, patch_dims)
        patches = tf.reshape(patches, [batch_size, num_patches, patch_dims])
        return patches

    def compute_output_shape(self, input_shape):
        """
        input_shape: (batch_size, H, W, C).
        Output shape: (batch_size, num_patches, patch_dims).
          where num_patches = (H//patch_size)*(W//patch_size)
          patch_dims = patch_size*patch_size*C
        """
        H = input_shape[1]
        W = input_shape[2]
        C = input_shape[3]
        if any(dim is None for dim in [H, W, C]):
            # If any dimension is None, we can't compute exact shape
            # but let's just raise an error to ensure clarity
            raise ValueError("Patches layer must have fully known (H,W,C).")
        patch_dims = self.patch_size * self.patch_size * C
        num_patches = (H // self.patch_size) * (W // self.patch_size)
        return (input_shape[0], num_patches, patch_dims)


class PatchEncoder(tf.keras.layers.Layer):
    """
    Projects each patch to a vector of size projection_dim,
    and adds a trainable positional embedding.
    """

    def __init__(self, num_patches: int, projection_dim: int):
        super().__init__()
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        # We will define self.projection, self.position_embedding in build()

    def build(self, input_shape):
        """
        input_shape: (batch_size, num_patches, patch_dim).
        Now we know patch_dim = input_shape[-1].
        """
        patch_dim = input_shape[-1]
        if patch_dim is None:
            raise ValueError("PatchEncoder needs a known 'patch_dim' at build time.")
        # Create our layers once shape is known
        self.projection = tf.keras.layers.Dense(self.projection_dim)
        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=self.num_patches,
            output_dim=self.projection_dim
        )
        super().build(input_shape)

    def call(self, patch_embeddings):
        """
        patch_embeddings: (batch_size, num_patches, patch_dim).
        """
        # project them -> (batch_size, num_patches, projection_dim)
        embedded = self.projection(patch_embeddings)
        # add position embeddings
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        embedded = embedded + self.position_embedding(positions)
        return embedded

    def compute_output_shape(self, input_shape):
        """
        input_shape: (batch_size, num_patches, patch_dim).
        output: (batch_size, num_patches, projection_dim).
        """
        return (input_shape[0], input_shape[1], self.projection_dim)


def create_transformer_block(
    x,
    num_heads: int,
    key_dim: int,
    mlp_units: int,
    dropout_rate: float
):
    """
    A single transformer encoder block:
      - MultiHeadAttention
      - LayerNorm + Residual
      - MLP (Dense -> Dense)
      - LayerNorm + Residual
    """
    # 1) MultiHeadAttention
    attn_output = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=key_dim,
        dropout=dropout_rate
    )(x, x)
    # Residual + LayerNorm
    x = x + attn_output
    x = LayerNormalization(epsilon=1e-6)(x)

    # 2) MLP
    mlp = layers.Dense(mlp_units, activation='relu')(x)
    mlp = layers.Dropout(dropout_rate)(mlp)
    mlp = layers.Dense(x.shape[-1])(mlp)  # project back to hidden dim

    # Residual + LayerNorm
    x = x + mlp
    x = LayerNormalization(epsilon=1e-6)(x)
    return x


class ViTModel(BaseDLModel):
    """
    Vision Transformer Model (ViT) that inherits from the BaseDLModel
    so we can reuse the multi-attribute dictionary logic.

    NOTE: We rely on having a FIXED input shape, e.g. (64,64,C) with
    no None dimensions for H and W. If your data is smaller, you might
    reduce patch_size or pad images to a consistent size.
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
        # Inherit from BaseDLModel for multi-attribute dict logic
        super().__init__(
            model_type=ModelType.VIT,
            model_name=model_name,
            model_filename=model_filename,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            selected_bands=selected_bands,
            predicted_attributes=predicted_attributes,
            selected_indexes=selected_indexes
        )
        
        # Store ViT-specific parameters for depth calculation
        self.transformer_depth = 4  # Number of transformer blocks
        
        # Now self.models[attr] is a dict of Keras sub-models,
        # one per attribute, each built by _build_model_for_attr().

    # Removed hardcoded get_architecture_config - now using dynamic extraction


    def _build_model_for_attr(self) -> tf.keras.Model:
        """
        Build a single-output Vision Transformer for one quality attribute.
        1) Patches -> 2) PatchEncoder -> 3) N x Transformer Blocks -> 4) Pool -> 5) Dense -> 1 output
        """
        # Example hyperparameters
        patch_size = 8                  # must divide (H, W)
        projection_dim = 64
        num_heads = 4
        transformer_depth = 4
        mlp_units = 128
        dropout_rate = 0.1

        # Unpack shape
        H, W, C = self.model_shape
        # Verify H & W are known (not None) here
        if None in (H, W, C):
            raise ValueError(f"ViTModel requires a fully-known (H,W,C). Got {(H,W,C)}")

        # Number of patches
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        if num_patches_h < 1 or num_patches_w < 1:
            raise ValueError(
                f"Patch size {patch_size} too large for input ({H}x{W}). "
                "Reduce patch_size or increase input dims."
            )
        num_patches = num_patches_h * num_patches_w

        # 1) Input
        inputs = Input(shape=(H, W, C))

        # 2) Patch extraction
        x = Patches(patch_size)(inputs)

        # 3) Patch encoding
        x = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim)(x)

        # 4) Several Transformer encoder blocks
        for _ in range(transformer_depth):
            x = create_transformer_block(
                x,
                num_heads=num_heads,
                key_dim=projection_dim // num_heads,
                mlp_units=mlp_units,
                dropout_rate=dropout_rate
            )

        # 5) Pool the final sequence
        x = GlobalAveragePooling1D()(x)  # shape: (batch_size, projection_dim)

        # 6) Final MLP for single-attribute regression
        x = Dense(mlp_units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        outputs = Dense(1, activation='linear')(x)

        model = Model(inputs=inputs, outputs=outputs, name="ViT_SingleAttr")
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def _build_multi_output_model(self) -> tf.keras.Model:
        """
        Build a multi-output Vision Transformer for all quality attributes.
        """
        # Example hyperparameters
        patch_size = 8                  # must divide (H, W)
        projection_dim = 64
        num_heads = 4
        transformer_depth = 4
        mlp_units = 128
        dropout_rate = 0.1

        # Unpack shape
        H, W, C = self.model_shape
        # Verify H & W are known (not None) here
        if None in (H, W, C):
            raise ValueError(f"ViTModel requires a fully-known (H,W,C). Got {(H,W,C)}")

        # Number of patches
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        if num_patches_h < 1 or num_patches_w < 1:
            raise ValueError(
                f"Patch size {patch_size} too large for input ({H}x{W}). "
                "Reduce patch_size or increase input dims."
            )
        num_patches = num_patches_h * num_patches_w

        # 1) Input
        inputs = Input(shape=(H, W, C))

        # 2) Patch extraction
        x = Patches(patch_size)(inputs)

        # 3) Patch encoding
        x = PatchEncoder(num_patches=num_patches, projection_dim=projection_dim)(x)

        # 4) Several Transformer encoder blocks
        for _ in range(transformer_depth):
            x = create_transformer_block(
                x,
                num_heads=num_heads,
                key_dim=projection_dim // num_heads,
                mlp_units=mlp_units,
                dropout_rate=dropout_rate
            )

        # 5) Pool the final sequence
        x = GlobalAveragePooling1D()(x)  # shape: (batch_size, projection_dim)

        # 6) Final MLP for multi-output regression
        x = Dense(mlp_units, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        
        # Multi-output layer - one output for all attributes
        outputs = Dense(len(self.predicted_attributes), activation='linear', name='multi_output')(x)

        model = Model(inputs=inputs, outputs=outputs, name="ViTMultiRegression")
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print(f"[ViTModel] Built multi-output model with {len(self.predicted_attributes)} outputs")
        return model