import tensorflow as tf
from tensorflow import keras
from keras import Input, Model, layers
from keras._tf_keras.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, ReLU, Flatten, Dense, Dropout, Concatenate,
    LayerNormalization, MultiHeadAttention, Add, Embedding, Reshape, GlobalAveragePooling2D
)
from keras._tf_keras.keras.optimizers import AdamW

from ..base_classes.BaseDLModel import BaseDLModel # Assuming BaseDLModel handles data prep, train, test, load, save
from src.config.enums import SpectralIndex, ModelType # For type hinting if used by parent
from typing import List, Optional, Tuple, Dict

class GlobalBranchFusionTransformer(BaseDLModel):
    def __init__(
        self,
        # Parameters from BaseDLModel
        model_name: str,
        model_filename: str,
        model_shape: tuple[int, int, int], # (H, W, C_total) - Initial input shape
        components: dict, # {'reflectance': True, 'std': False, ...}
        component_dimensions: dict, # {'reflectance': N_reflectance_channels, ... , 'total': C_total}
        predicted_attributes: list[str],
        selected_bands: Optional[List[int]] = None, # Used by parent for data prep
        selected_indexes: Optional[List[SpectralIndex]] = None, # Used by parent for data prep
        
        # CNN Branch specific parameters
        cnn_branch_filters_start: int = 32,
        cnn_branch_filters_growth: float = 1.5,
        cnn_branch_kernels: Tuple[int, int] = (3, 3),
        cnn_branch_depth: int = 2,
        cnn_branch_use_pooling: bool = True, # Pooling within the CNN branch itself
        cnn_pool_size: Tuple[int, int] = (2, 2),
        
        # Transformer parameters
        d_model: int = 128,  # Common dimension for tokens
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        transformer_dff: int = 256,  # Dimension of the feed-forward network in Transformer
        transformer_dropout_rate: float = 0.1,
        
        # MLP Head parameters for regression
        mlp_head_units: List[int] = [64],
        mlp_dropout_rate: float = 0.2,
        
        # Optimizer parameters
        optimizer_params: Optional[Dict] = None, # e.g., {'learning_rate': 1e-4, 'weight_decay': 1e-4}
    ):
        # Store architecture-specific parameters
        self.cnn_branch_filters_start = cnn_branch_filters_start
        self.cnn_branch_filters_growth = cnn_branch_filters_growth
        self.cnn_branch_kernels = cnn_branch_kernels
        self.cnn_branch_depth = cnn_branch_depth
        self.cnn_branch_use_pooling = cnn_branch_use_pooling
        self.cnn_pool_size = cnn_pool_size

        self.d_model = d_model
        self.transformer_layers = transformer_layers
        if d_model % transformer_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by transformer_heads ({transformer_heads}).")
        self.transformer_heads = transformer_heads
        self.transformer_dff = transformer_dff
        self.transformer_dropout_rate = transformer_dropout_rate

        self.mlp_head_units = mlp_head_units
        self.mlp_dropout_rate = mlp_dropout_rate
        
        # Store optimizer parameters
        self.optimizer_params = optimizer_params or {'learning_rate': 1e-4, 'weight_decay': 1e-5}

        # Call the parent constructor
        super().__init__(
            model_type=ModelType.GLOBAL_BRANCH_FUSION_TRANSFORMER,
            model_name=model_name,
            model_filename=model_filename,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            selected_bands=selected_bands,
            predicted_attributes=predicted_attributes,
            selected_indexes=selected_indexes
        )

    def _build_cnn_branch(self, input_tensor: tf.Tensor, branch_name: str) -> tf.Tensor:
        """
        Builds a CNN branch for a given input tensor.
        Output is a feature map (before global pooling).
        """
        x = input_tensor
        current_filters = self.cnn_branch_filters_start
        
        for i in range(self.cnn_branch_depth):
            x = Conv2D(
                filters=int(current_filters),
                kernel_size=self.cnn_branch_kernels,
                padding='same',
                name=f"{branch_name}_cnn_conv{i+1}"
            )(x)
            x = BatchNormalization(name=f"{branch_name}_cnn_bn{i+1}")(x)
            x = ReLU(name=f"{branch_name}_cnn_relu{i+1}")(x)
            if i < self.cnn_branch_depth - 1: # Grow filters for next layer
                current_filters *= self.cnn_branch_filters_growth
        
        if self.cnn_branch_use_pooling:
            x = MaxPooling2D(pool_size=self.cnn_pool_size, name=f"{branch_name}_cnn_pool")(x)
        
        # Output: Feature map of size H' x W' x d_branch_cnn_output
        return x

    def _transformer_encoder_block(self, inputs: tf.Tensor, name_prefix: str) -> tf.Tensor:
        """
        Standard Transformer Encoder Block.
        inputs shape: (batch_size, sequence_length, d_model)
        """
        # Multi-Head Self-Attention
        attn_output = MultiHeadAttention(
            num_heads=self.transformer_heads, 
            key_dim=self.d_model // self.transformer_heads, # Dimensionality of each attention head for q, k, v
            dropout=self.transformer_dropout_rate, 
            name=f"{name_prefix}_mha"
        )(query=inputs, value=inputs, key=inputs) # Self-attention
        
        # Add & Norm (Residual connection + Layer Normalization)
        out1 = Add(name=f"{name_prefix}_add1")([inputs, attn_output])
        out1 = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(out1)

        # Feed-Forward Network
        ffn_output = Dense(self.transformer_dff, activation='relu', name=f"{name_prefix}_ffn_dense1")(out1)
        ffn_output = Dense(self.d_model, name=f"{name_prefix}_ffn_dense2")(ffn_output) # Projection back to d_model
        ffn_output = Dropout(self.transformer_dropout_rate, name=f"{name_prefix}_ffn_dropout")(ffn_output)
        
        # Add & Norm
        out2 = Add(name=f"{name_prefix}_add2")([out1, ffn_output])
        out2 = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(out2)
        return out2

    def _build_model_for_attr(self) -> tf.keras.Model:
        """
        Builds the multi-branch CNN with global branch fusion via Transformer for a single attribute.
        """
        H, W, C_total = self.model_shape # Original input shape
        input_layer = Input(shape=(H, W, C_total), name="input_combined_modalities")
        
        active_components = {
            name: dim for name, dim in self.component_dimensions.items()
            if name != 'total' and dim > 0 and self.components.get(name, False)
        }
        sorted_component_names = sorted(active_components.keys())

        if not sorted_component_names:
            raise ValueError("No active components found for GlobalBranchFusionTransformer. Check 'components' and 'component_dimensions'.")
        
        print(f"[GlobalBranchFusionTransformer] Building model with active branches: {sorted_component_names}")

        branch_global_vectors_projected = []
        current_channel_offset = 0
        
        for branch_idx, branch_name in enumerate(sorted_component_names):
            num_channels_for_branch = self.component_dimensions[branch_name]
            
            start_offset = current_channel_offset
            end_offset = current_channel_offset + num_channels_for_branch
            
            branch_input_slice = layers.Lambda(
                lambda x, s=start_offset, e=end_offset: x[:, :, :, s:e],
                name=f"{branch_name}_slice"
            )(input_layer)
            
            current_channel_offset += num_channels_for_branch
            
            # 1. CNN Branch processing
            cnn_branch_output_map = self._build_cnn_branch(branch_input_slice, branch_name) # (B, H', W', d_b_cnn)
            
            # 2. Global Pooling
            global_pooled_vector = GlobalAveragePooling2D(name=f"{branch_name}_global_pool")(cnn_branch_output_map) # (B, d_b_cnn)
            
            # 3. Projection to d_model
            projected_vector = Dense(self.d_model, activation='relu', name=f"{branch_name}_project_to_d_model")(global_pooled_vector) # (B, d_model)
            # Add an explicit Reshape to (B, 1, d_model) to represent it as a single token in a sequence
            projected_vector_token = Reshape((1, self.d_model), name=f"{branch_name}_token_reshape")(projected_vector)
            branch_global_vectors_projected.append(projected_vector_token)

        # CLS Token (learnable weight)
        # Using an Embedding layer that takes a dummy input and produces the CLS token vector.
        # The input_dim=1 means there's only one "word" in our CLS vocabulary.
        # The output_dim=self.d_model is the dimension of the CLS token.
        # We need to provide an input to this Embedding layer. A common way is to use tf.zeros or a constant tensor.
        
        # Get batch_size symbolically
        # OLD: batch_size_symbolic = tf.shape(input_layer)[0]
        # NEW: Use a Lambda layer to get batch_size as a KerasTensor
        batch_size_kt = layers.Lambda(lambda x: tf.shape(x)[0], name="get_batch_size")(input_layer)

        # Create a dummy input for the CLS token embedding layer, matching the batch size
        # OLD: cls_dummy_input = tf.zeros_like(batch_size_symbolic, dtype=tf.int32) # Scalar tensor with value 0
        # OLD: cls_dummy_input = Reshape((1,), name="cls_dummy_input_reshape")(cls_dummy_input) # Shape (batch_size, 1) for Embedding
        # NEW: Create input of shape (batch_size, 1) with zeros for the Embedding layer
        cls_input_for_embedding = layers.Lambda(
            lambda x: tf.zeros((tf.shape(x)[0], 1), dtype=tf.int32),
            name="cls_dummy_input_for_embedding"
        )(input_layer)

        cls_token_embedding_layer = Embedding(
            input_dim=1, # Only one CLS token type
            output_dim=self.d_model,
            name="cls_token_embedder"
        )
        cls_token = cls_token_embedding_layer(cls_input_for_embedding) # Shape: (batch_size, 1, d_model)

        # Concatenate CLS token with branch tokens
        # Sequence: [CLS, Branch1_GlobalToken, Branch2_GlobalToken, ...]
        all_tokens_list = [cls_token] + branch_global_vectors_projected
        
        if not all_tokens_list: # Should not happen if sorted_component_names is not empty
             raise ValueError("Token list for Transformer is empty.")

        concatenated_tokens = Concatenate(axis=1, name="concatenate_cls_and_branch_tokens")(all_tokens_list)
        # Shape: (batch, num_branches + 1, d_model)
        
        num_transformer_tokens = len(sorted_component_names) + 1 # Branches + CLS

        # Add Positional Embeddings for the sequence of tokens (CLS, branch1, branch2, ...)
        positional_embedding_layer = Embedding(
            input_dim=num_transformer_tokens, # Max sequence length (CLS + num_branches)
            output_dim=self.d_model,
            name="sequence_positional_embedding"
        )
        # Create position indices: (0, 1, 2, ..., num_transformer_tokens-1)
        position_indices = tf.range(start=0, limit=num_transformer_tokens, delta=1)
        # Expand dims to be (1, num_transformer_tokens) for broadcasting if needed by Embedding layer
        position_indices_expanded = tf.expand_dims(position_indices, axis=0) # (1, num_transformer_tokens)
        
        # Tile it to match batch size using a Lambda layer
        # OLD: position_indices_batch = tf.tile(position_indices_batch, [batch_size_symbolic, 1])
        # NEW:
        position_indices_tiled = layers.Lambda(
            lambda x: tf.tile(position_indices_expanded, [tf.shape(x)[0], 1]),
            name="tile_position_indices"
        )(input_layer) # Pass any KerasTensor from the graph to make it part of the graph

        pos_embeddings = positional_embedding_layer(position_indices_tiled) # (batch, num_transformer_tokens, d_model)
        
        tokens_with_positional_encoding = Add(name="add_positional_embeddings")([concatenated_tokens, pos_embeddings])
        
        # Transformer Encoder Stack
        transformer_input = tokens_with_positional_encoding
        for i in range(self.transformer_layers):
            transformer_input = self._transformer_encoder_block(
                transformer_input, name_prefix=f"transformer_encoder_layer_{i+1}"
            )
        
        # Regression Head using CLS token output
        # CLS token is at index 0 of the sequence
        cls_token_final_representation = layers.Lambda(
            lambda t: t[:, 0, :], name="extract_cls_token_representation"
        )(transformer_input) # (batch_size, d_model)
        
        mlp_head_input = cls_token_final_representation
        for i, num_units in enumerate(self.mlp_head_units):
            mlp_head_input = Dense(num_units, activation='relu', name=f"mlp_head_dense_{i+1}")(mlp_head_input)
            if self.mlp_dropout_rate > 0:
                 mlp_head_input = Dropout(self.mlp_dropout_rate, name=f"mlp_head_dropout_{i+1}")(mlp_head_input)
        
        output_prediction = Dense(1, activation='linear', name="final_output_regression")(mlp_head_input) # Single output for one attribute
        
        model = Model(inputs=input_layer, outputs=output_prediction, name=self.model_name)
        
        # Optimizer
        lr = self.optimizer_params.get('learning_rate', 1e-4)
        wd = self.optimizer_params.get('weight_decay', 0.0) # AdamW default might be 0.004, check Keras docs. Often set explicitly.
        # Using a small default weight decay if not provided.
        wd = self.optimizer_params.get('weight_decay', 1e-5) if 'weight_decay' not in self.optimizer_params else wd

        optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        # model.summary() # Useful for debugging structure
        return model

    # train, test, prepare_data, load_model, save_model methods will be inherited from BaseDLModel
    # Make sure BaseDLModel's prepare_data is compatible or can be overridden if needed.
    # The key is that BaseDLModel's prepare_data must return X with shape (n_samples, H, W, C_total)
    # and Y with shape (n_samples, num_predicted_attributes).
    # The _build_model_for_attr is called for each attribute, so Y_train[:, idx] will be used.
    
    def _build_multi_output_model(self) -> tf.keras.Model:
        """
        Build a multi-output Global Branch Fusion Transformer for all quality attributes.
        """
        H, W, C_total = self.model_shape
        input_layer = Input(shape=(H, W, C_total), name="input_combined_modalities")
        
        active_components = {
            name: dim for name, dim in self.component_dimensions.items()
            if name != 'total' and dim > 0 and self.components.get(name, False)
        }
        sorted_component_names = sorted(active_components.keys())

        if not sorted_component_names:
            raise ValueError("No active components found for GlobalBranchFusionTransformer. Check 'components' and 'component_dimensions'.")
        
        print(f"[GlobalBranchFusionTransformer] Building multi-output model with active branches: {sorted_component_names}")

        branch_global_vectors_projected = []
        current_channel_offset = 0
        
        for branch_idx, branch_name in enumerate(sorted_component_names):
            num_channels_for_branch = self.component_dimensions[branch_name]
            
            start_offset = current_channel_offset
            end_offset = current_channel_offset + num_channels_for_branch
            
            branch_input_slice = layers.Lambda(
                lambda x, s=start_offset, e=end_offset: x[:, :, :, s:e],
                name=f"{branch_name}_slice"
            )(input_layer)
            
            current_channel_offset += num_channels_for_branch
            
            # 1. CNN Branch processing
            cnn_branch_output_map = self._build_cnn_branch(branch_input_slice, branch_name)
            
            # 2. Global Pooling
            global_pooled_vector = GlobalAveragePooling2D(name=f"{branch_name}_global_pool")(cnn_branch_output_map)
            
            # 3. Projection to d_model
            projected_vector = Dense(self.d_model, activation='relu', name=f"{branch_name}_project_to_d_model")(global_pooled_vector)
            projected_vector_token = Reshape((1, self.d_model), name=f"{branch_name}_token_reshape")(projected_vector)
            branch_global_vectors_projected.append(projected_vector_token)

        # CLS Token
        batch_size_kt = layers.Lambda(lambda x: tf.shape(x)[0], name="get_batch_size")(input_layer)
        cls_input_for_embedding = layers.Lambda(
            lambda x: tf.zeros((tf.shape(x)[0], 1), dtype=tf.int32),
            name="cls_dummy_input_for_embedding"
        )(input_layer)

        cls_embedding_layer = Embedding(
            input_dim=1, output_dim=self.d_model, name="cls_token_embedding"
        )
        cls_token_embedded = cls_embedding_layer(cls_input_for_embedding)

        # Concatenate CLS token with branch tokens
        all_tokens = [cls_token_embedded] + branch_global_vectors_projected
        concatenated_tokens = Concatenate(axis=1, name="concatenate_cls_and_branch_tokens")(all_tokens)

        # Transformer Encoder Stack
        transformer_input = concatenated_tokens
        for i in range(self.transformer_layers):
            transformer_input = self._transformer_encoder_block(
                transformer_input, name_prefix=f"transformer_encoder_layer_{i+1}"
            )
        
        # Regression Head using CLS token output
        cls_token_final_representation = layers.Lambda(
            lambda t: t[:, 0, :], name="extract_cls_token_representation"
        )(transformer_input)
        
        mlp_head_input = cls_token_final_representation
        for i, num_units in enumerate(self.mlp_head_units):
            mlp_head_input = Dense(num_units, activation='relu', name=f"mlp_head_dense_{i+1}")(mlp_head_input)
            if self.mlp_dropout_rate > 0:
                 mlp_head_input = Dropout(self.mlp_dropout_rate, name=f"mlp_head_dropout_{i+1}")(mlp_head_input)
        
        # Multi-output layer - one output for all attributes
        output_prediction = Dense(len(self.predicted_attributes), activation='linear', name="multi_output")(mlp_head_input)
        
        model = Model(inputs=input_layer, outputs=output_prediction, name=f"GlobalBranchFusionTransformerMultiRegression")
        
        # Optimizer
        lr = self.optimizer_params.get('learning_rate', 1e-4)
        wd = self.optimizer_params.get('weight_decay', 1e-5)
        optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        print(f"[GlobalBranchFusionTransformer] Built multi-output model with {len(self.predicted_attributes)} outputs")
        return model

    # Removed hardcoded get_architecture_config - now using dynamic extraction
