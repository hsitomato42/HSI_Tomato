import tensorflow as tf
from tensorflow import keras
from keras import Input, Model, layers
from keras._tf_keras.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, ReLU, Flatten, Dense, Dropout, Concatenate,
    LayerNormalization, MultiHeadAttention, Add, Embedding, Reshape, GlobalAveragePooling2D,
    SeparableConv2D, AveragePooling2D, Multiply, Lambda, GlobalAveragePooling1D
)
from keras._tf_keras.keras.optimizers import AdamW

from ..base_classes.BaseDLModel import BaseDLModel
from src.config.enums import SpectralIndex, ModelType
from typing import List, Optional, Tuple, Dict
import numpy as np
import src.config as config


class ComponentDrivenAttentionTransformerV2(BaseDLModel):
    """
    Component-Driven Attention Transformer V2 (CDAT-V2) - Enhanced version with spatial downsampling.
    
    Key improvements over V1:
    - Uses progressive CNN downsampling to reduce spatial dimensions by 2x (each token represents 2x2 pixels)
    - Simplified processing without pixel masking
    - Fixed token count for easier batching
    - ~4x reduction in attention computation while preserving spatial relationships
    """
    
    def __init__(
        self,
        # Parameters from BaseDLModel
        model_name: str,
        model_filename: str,
        model_shape: tuple[int, int, int],  # (H, W, C_total)
        components: dict,  # {'reflectance': True, 'std': False, ...}
        component_dimensions: dict,  # {'reflectance': N_channels, ..., 'total': C_total}
        predicted_attributes: list[str],
        selected_bands: Optional[List[int]] = None,
        selected_indexes: Optional[List[SpectralIndex]] = None,
        
        # CNN Backbone specific parameters with downsampling
        backbone_filters_before_downsample: List[int] = [32],  # Filters before downsampling
        backbone_filters_after_downsample: List[int] = [48, 64],  # Filters after downsampling
        backbone_kernels: Tuple[int, int] = (3, 3),
        downsampling_factor: int = 2,  # How much to downsample (2 = 2x2 regions)
        downsampling_method: str = "strided_conv",  # "strided_conv", "max_pool", "avg_pool"
        
        # Component attention parameters
        d_model: int = 128,  # Common dimension for tokens and representations
        component_attention_heads: int = 4,  # Number of attention heads per component
        num_attention_layers_per_component: int = 1,
        
        # Advisor component parameters
        advisor_filters_before_downsample: List[int] = [16],  # Lighter processing for advisors
        advisor_filters_after_downsample: List[int] = [16],
        
        # Fusion parameters
        fusion_method: str = "sequential",  # "sequential" or "parallel"
        component_processing_order: Optional[List[str]] = None,  # For sequential fusion
        
        # Final transformer parameters
        final_transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dff: int = 256,
        transformer_dropout_rate: float = 0.1,
        
        # MLP Head parameters for regression
        mlp_head_units: List[int] = [64],
        mlp_dropout_rate: float = 0.2,
        
        # Optimizer parameters
        optimizer_params: Optional[Dict] = None,
    ):
        # Store CDAT-V2 specific parameters
        self.backbone_filters_before_downsample = backbone_filters_before_downsample
        self.backbone_filters_after_downsample = backbone_filters_after_downsample
        self.backbone_kernels = backbone_kernels
        self.downsampling_factor = downsampling_factor
        self.downsampling_method = downsampling_method
        
        self.advisor_filters_before_downsample = advisor_filters_before_downsample
        self.advisor_filters_after_downsample = advisor_filters_after_downsample
        
        self.d_model = d_model
        self.component_attention_heads = component_attention_heads
        self.num_attention_layers_per_component = num_attention_layers_per_component
        
        self.fusion_method = fusion_method
        self.component_processing_order = component_processing_order or config.CDAT_COMPONENT_ORDER
        
        print(f"[CDAT-V2] Component processing order: {self.component_processing_order}")
        print(f"[CDAT-V2] Downsampling factor: {self.downsampling_factor}x using {self.downsampling_method}")
        
        self.final_transformer_layers = final_transformer_layers
        if d_model % transformer_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by transformer_heads ({transformer_heads}).")
        self.transformer_heads = transformer_heads
        self.transformer_dff = transformer_dff
        self.transformer_dropout_rate = transformer_dropout_rate
        
        self.mlp_head_units = mlp_head_units
        self.mlp_dropout_rate = mlp_dropout_rate
        
        self.optimizer_params = optimizer_params if optimizer_params else {}
        
        # Calculate downsampled dimensions
        H, W, _ = model_shape
        self.downsampled_h = H // self.downsampling_factor
        self.downsampled_w = W // self.downsampling_factor
        self.downsampled_spatial_tokens = self.downsampled_h * self.downsampled_w
        
        print(f"[CDAT-V2] Original spatial dims: {H}x{W} ({H*W} positions)")
        print(f"[CDAT-V2] Downsampled dims: {self.downsampled_h}x{self.downsampled_w} ({self.downsampled_spatial_tokens} tokens)")
        print(f"[CDAT-V2] Spatial reduction: {(H*W)/self.downsampled_spatial_tokens:.1f}x")
        
        # Call parent constructor
        super().__init__(
            model_type=ModelType.COMPONENT_DRIVEN_ATTENTION_TRANSFORMER_V2,
            model_name=model_name,
            model_filename=model_filename,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            selected_bands=selected_bands,
            predicted_attributes=predicted_attributes,
            selected_indexes=selected_indexes
        )

    def _get_active_advisor_components(self) -> List[str]:
        """
        Dynamically determine which components should act as advisors based on configuration.
        Reflectance is always processed by the CNN backbone, other components become advisors.
        """
        advisor_components = []
        
        # Check each component (excluding reflectance which goes to CNN backbone)
        for component_name in ['std', 'ndsi', 'indexes']:
            if (self.components.get(component_name, False) and 
                self.component_dimensions.get(component_name, 0) > 0):
                advisor_components.append(component_name)
        
        print(f"[CDAT-V2] Active advisor components: {advisor_components}")
        return advisor_components

    def _slice_input_components(self, input_layer: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Slice the input tensor into individual components based on component dimensions.
        """
        component_slices = {}
        current_offset = 0
        
        # Process components in a fixed order to ensure consistency
        for component_name in ['reflectance', 'std', 'ndsi', 'indexes']:
            if (self.components.get(component_name, False) and 
                self.component_dimensions.get(component_name, 0) > 0):
                
                num_channels = self.component_dimensions[component_name]
                start_offset = current_offset
                end_offset = current_offset + num_channels
                
                component_slice = layers.Lambda(
                    lambda x, s=start_offset, e=end_offset: x[:, :, :, s:e],
                    name=f"{component_name}_slice"
                )(input_layer)
                
                component_slices[component_name] = component_slice
                current_offset += num_channels
                
                print(f"[CDAT-V2] {component_name}: channels {start_offset}:{end_offset} ({num_channels} channels)")
        
        return component_slices

    def _apply_downsampling_layer(self, x: tf.Tensor, layer_name: str) -> tf.Tensor:
        """
        Apply downsampling based on the specified method.
        """
        if self.downsampling_method == "strided_conv":
            # Use strided convolution for learnable downsampling
            current_filters = x.shape[-1]
            x = Conv2D(
                filters=current_filters,
                kernel_size=self.backbone_kernels,
                strides=self.downsampling_factor,
                padding='same',
                name=f"{layer_name}_strided_downsample"
            )(x)
            x = BatchNormalization(name=f"{layer_name}_downsample_bn")(x)
            x = ReLU(name=f"{layer_name}_downsample_relu")(x)
        elif self.downsampling_method == "max_pool":
            x = MaxPooling2D(
                pool_size=self.downsampling_factor, 
                strides=self.downsampling_factor,
                name=f"{layer_name}_max_downsample"
            )(x)
        elif self.downsampling_method == "avg_pool":
            x = AveragePooling2D(
                pool_size=self.downsampling_factor,
                strides=self.downsampling_factor, 
                name=f"{layer_name}_avg_downsample"
            )(x)
        else:
            raise ValueError(f"Unknown downsampling method: {self.downsampling_method}")
        
        return x

    def _build_reflectance_backbone(self, reflectance_input: tf.Tensor) -> tf.Tensor:
        """
        Build a CNN backbone with progressive downsampling that processes reflectance data.
        Returns spatial tokens at downsampled resolution: each token represents a 2x2 region.
        """
        print(f"[CDAT-V2] Building CNN backbone with input shape: {reflectance_input.shape}")
        
        x = reflectance_input
        
        # Phase 1: Feature extraction before downsampling
        for i, num_filters in enumerate(self.backbone_filters_before_downsample):
            x = Conv2D(
                filters=num_filters,
                kernel_size=self.backbone_kernels,
                padding='same',
                name=f"backbone_pre_conv{i+1}"
            )(x)
            x = BatchNormalization(name=f"backbone_pre_bn{i+1}")(x)
            x = ReLU(name=f"backbone_pre_relu{i+1}")(x)
        
        print(f"[CDAT-V2] After pre-downsampling layers: {x.shape}")
        
        # Phase 2: Downsampling
        x = self._apply_downsampling_layer(x, "backbone")
        print(f"[CDAT-V2] After downsampling: {x.shape}")
        
        # Phase 3: Feature refinement after downsampling
        for i, num_filters in enumerate(self.backbone_filters_after_downsample):
            x = Conv2D(
                filters=num_filters,
                kernel_size=self.backbone_kernels,
                padding='same',
                name=f"backbone_post_conv{i+1}"
            )(x)
            x = BatchNormalization(name=f"backbone_post_bn{i+1}")(x)
            x = ReLU(name=f"backbone_post_relu{i+1}")(x)
        
        # Convert spatial feature map to tokens
        B, H_down, W_down, d_backbone = x.shape
        N_down = H_down * W_down
        
        print(f"[CDAT-V2] Final CNN output shape: {x.shape}, creating {N_down} spatial tokens")
        
        # Flatten spatial dimensions: (B, H_down, W_down, d_backbone) → (B, N_down, d_backbone)
        spatial_tokens = Reshape((N_down, d_backbone), name="spatial_tokens")(x)
        
        # Project to common dimension: (B, N_down, d_backbone) → (B, N_down, d_model)
        spatial_tokens = Dense(self.d_model, name="spatial_projection")(spatial_tokens)
        
        return spatial_tokens

    def _add_positional_embeddings(self, spatial_tokens: tf.Tensor) -> tf.Tensor:
        """
        Add learnable positional embeddings to spatial tokens.
        Now works with downsampled spatial dimensions.
        """
        class PositionalEmbeddingLayer(layers.Layer):
            def __init__(self, model_shape, downsampling_factor, d_model, **kwargs):
                super().__init__(**kwargs)
                self.d_model = d_model
                
                # Calculate maximum possible positions more safely
                # Use the original spatial dimensions and add a safety buffer
                H, W, _ = model_shape
                max_h = (H // downsampling_factor) + 1  # +1 for safety
                max_w = (W // downsampling_factor) + 1  # +1 for safety
                max_positions = max_h * max_w
                
                self.pos_embedding = Embedding(
                    input_dim=max_positions,
                    output_dim=d_model,
                    name="spatial_positional_embedding"
                )
            
            def call(self, spatial_tokens):
                N_down = tf.shape(spatial_tokens)[1]
                positions = tf.range(N_down)
                pos_embeddings = self.pos_embedding(positions)  # (N_down, d_model)
                
                # Add positional embeddings to tokens
                enhanced_tokens = spatial_tokens + tf.expand_dims(pos_embeddings, axis=0)
                return enhanced_tokens
        
        pos_layer = PositionalEmbeddingLayer(
            model_shape=self.model_shape,
            downsampling_factor=self.downsampling_factor,
            d_model=self.d_model,
            name="add_positional_embeddings"
        )
        
        return pos_layer(spatial_tokens)

    def _build_component_query_generator(self, component_input: tf.Tensor, component_name: str) -> tf.Tensor:
        """
        Generate spatial queries from advisor components with matching downsampling.
        Each component provides different 'perspectives' on what to focus on.
        """
        print(f"[CDAT-V2] Building spatial query generator for {component_name} with input shape: {component_input.shape}")
        
        x = component_input
        
        # Phase 1: Light feature extraction before downsampling
        for i, num_filters in enumerate(self.advisor_filters_before_downsample):
            x = SeparableConv2D(
                num_filters, self.backbone_kernels,
                padding='same',
                name=f"{component_name}_pre_conv{i+1}"
            )(x)
            x = BatchNormalization(name=f"{component_name}_pre_bn{i+1}")(x)
            x = ReLU(name=f"{component_name}_pre_relu{i+1}")(x)
        
        # Phase 2: Downsampling to match backbone resolution
        x = self._apply_downsampling_layer(x, f"{component_name}_advisor")
        print(f"[CDAT-V2] {component_name} after downsampling: {x.shape}")
        
        # Phase 3: Feature refinement after downsampling
        for i, num_filters in enumerate(self.advisor_filters_after_downsample):
            x = SeparableConv2D(
                num_filters, self.backbone_kernels,
                padding='same',
                name=f"{component_name}_post_conv{i+1}"
            )(x)
            x = BatchNormalization(name=f"{component_name}_post_bn{i+1}")(x)
            x = ReLU(name=f"{component_name}_post_relu{i+1}")(x)
        
        # Flatten spatial dimensions to create spatial queries: (B, H_down, W_down, filters) -> (B, N_down, filters)
        H_down, W_down = x.shape[1], x.shape[2]
        N_down = H_down * W_down
        final_filters = x.shape[3]
        spatial_features = Reshape((N_down, final_filters), name=f"{component_name}_spatial_reshape")(x)
        
        # Generate queries for each spatial location and attention head
        d_k = self.d_model // self.component_attention_heads
        total_query_dim = self.component_attention_heads * d_k
        
        # Project to query dimensions: (B, N_down, final_filters) -> (B, N_down, total_query_dim)
        spatial_queries = Dense(
            total_query_dim,
            activation='relu',
            name=f"{component_name}_spatial_queries"
        )(spatial_features)
        
        # Reshape for multi-head attention: (B, N_down, total_query_dim) -> (B, N_down, num_heads, d_k)
        queries = Reshape(
            (N_down, self.component_attention_heads, d_k),
            name=f"{component_name}_reshape_spatial_queries"
        )(spatial_queries)
        
        print(f"[CDAT-V2] Generated spatial queries for {component_name} with shape: {queries.shape}")
        return queries

    def _component_driven_attention(self, spatial_tokens: tf.Tensor, component_queries: tf.Tensor,
                                   component_name: str) -> tf.Tensor:
        """
        Apply spatial-to-spatial attention where queries come from components and 
        keys/values are the spatial tokens from CNN backbone.
        """
        print(f"[CDAT-V2] Applying spatial {component_name} attention")
        
        class ComponentAttentionLayer(layers.Layer):
            def __init__(self, d_model, num_heads, component_name, **kwargs):
                super().__init__(**kwargs)
                self.d_model = d_model
                self.num_heads = num_heads
                self.component_name = component_name
                self.d_k = d_model // num_heads
                
                # Dense layers for keys and values
                self.keys_dense = Dense(d_model, name=f"{component_name}_keys")
                self.values_dense = Dense(d_model, name=f"{component_name}_values")
                self.final_projection = Dense(d_model, name=f"{component_name}_final_projection")
                
            def call(self, inputs):
                spatial_tokens, component_queries = inputs
                # spatial_tokens: (B, N_down, d_model)
                # component_queries: (B, N_down, num_heads, d_k)
                
                # Generate keys and values from spatial tokens
                keys = self.keys_dense(spatial_tokens)  # (B, N_down, d_model)
                values = self.values_dense(spatial_tokens)  # (B, N_down, d_model)
                
                # Get shapes
                B = tf.shape(spatial_tokens)[0]
                N_down = tf.shape(spatial_tokens)[1]
                
                # Reshape keys and values for multi-head attention
                keys = tf.reshape(keys, [B, N_down, self.num_heads, self.d_k])
                values = tf.reshape(values, [B, N_down, self.num_heads, self.d_k])
                
                # Apply multi-head spatial attention
                attention_outputs = []
                for head in range(self.num_heads):
                    # Extract head-specific tensors
                    q = component_queries[:, :, head, :]  # (B, N_down, d_k) - Query per spatial location
                    k = keys[:, :, head, :]               # (B, N_down, d_k) - Key per spatial location
                    v = values[:, :, head, :]             # (B, N_down, d_k) - Value per spatial location
                    
                    # Spatial-to-spatial attention (each location can attend to all others)
                    attention_scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.d_k, tf.float32))
                    attention_weights = tf.nn.softmax(attention_scores, axis=-1)  # (B, N_down, N_down)
                    
                    # Apply attention to values
                    attended_values = tf.matmul(attention_weights, v)  # (B, N_down, d_k)
                    attention_outputs.append(attended_values)
                
                # Concatenate multi-head outputs: (B, N_down, d_model)
                multi_head_output = tf.concat(attention_outputs, axis=-1)
                
                # Final projection
                component_enhanced_tokens = self.final_projection(multi_head_output)
                
                # Global pooling to get component representation for final token sequence
                component_representation = tf.reduce_mean(component_enhanced_tokens, axis=1)  # (B, d_model)
                
                return component_representation
        
        attention_layer = ComponentAttentionLayer(
            d_model=self.d_model,
            num_heads=self.component_attention_heads,
            component_name=component_name,
            name=f"{component_name}_spatial_attention"
        )
        
        component_representation = attention_layer([spatial_tokens, component_queries])
        
        print(f"[CDAT-V2] {component_name} spatial attention output shape: {component_representation.shape}")
        
        return component_representation

    def _sequential_component_fusion(self, spatial_tokens: tf.Tensor, 
                                   component_inputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Apply component attentions sequentially with spatial queries and stable updates.
        """
        print(f"[CDAT-V2] Applying sequential component fusion")
        
        current_tokens = spatial_tokens
        component_representations = []
        
        # Process components in the specified order
        active_components = [comp for comp in self.component_processing_order 
                           if comp in component_inputs]
        
        print(f"[CDAT-V2] Processing components in order: {active_components}")
        
        for component_name in active_components:
            component_input = component_inputs[component_name]
            
            # Generate spatial queries from current component
            spatial_queries = self._build_component_query_generator(
                component_input, component_name
            )
            
            # Apply spatial component attention
            component_repr = self._component_driven_attention(
                current_tokens, spatial_queries, component_name
            )
            
            component_representations.append(component_repr)
            
            # Stable token update with gating
            class StableTokenUpdater(layers.Layer):
                def __init__(self, d_model, **kwargs):
                    super().__init__(**kwargs)
                    self.d_model = d_model
                    self.gate_dense = Dense(d_model, activation='sigmoid')
                    self.update_scale = 0.1  # Small update to maintain stability
                
                def call(self, inputs):
                    tokens, component_repr = inputs
                    # Learnable gating mechanism
                    gate = self.gate_dense(component_repr)  # (B, d_model)
                    
                    # Expand and broadcast component representation
                    component_info = tf.expand_dims(component_repr, axis=1)  # (B, 1, d_model)
                    N_tokens = tf.shape(tokens)[1]
                    component_broadcast = tf.tile(component_info, [1, N_tokens, 1])  # (B, N_down, d_model)
                    
                    # Gated update with scaling
                    gate_broadcast = tf.tile(tf.expand_dims(gate, axis=1), [1, N_tokens, 1])
                    gated_update = gate_broadcast * component_broadcast
                    
                    return tokens + self.update_scale * gated_update
            
            token_updater = StableTokenUpdater(
                d_model=self.d_model,
                name=f"stable_tokens_after_{component_name}"
            )
            current_tokens = token_updater([current_tokens, component_repr])
        
        return current_tokens, component_representations

    def _parallel_component_fusion(self, spatial_tokens: tf.Tensor, 
                                 component_inputs: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply all component attentions in parallel with spatial queries.
        """
        print(f"[CDAT-V2] Applying parallel component fusion")
        
        component_representations = []
        
        for component_name, component_input in component_inputs.items():
            # Generate spatial queries
            spatial_queries = self._build_component_query_generator(
                component_input, component_name
            )
            # Apply spatial attention
            component_repr = self._component_driven_attention(
                spatial_tokens, spatial_queries, component_name
            )
            component_representations.append(component_repr)
        
        # Fuse all component representations
        if len(component_representations) > 1:
            fused_repr = Add(name="parallel_fusion")(component_representations)
        else:
            fused_repr = component_representations[0]
        
        return spatial_tokens, fused_repr

    def _create_learnable_cls_token(self, input_layer: tf.Tensor) -> tf.Tensor:
        """
        Create a learnable CLS token for final prediction.
        """
        class CLSToken(layers.Layer):
            def __init__(self, d_model, **kwargs):
                super().__init__(**kwargs)
                self.d_model = d_model

            def build(self, input_shape):
                self.cls_embedding = self.add_weight(
                    name="cls_embedding",
                    shape=(1, 1, self.d_model),
                    initializer="random_normal",
                    trainable=True
                )

            def call(self, input_tensor):
                batch_size = tf.shape(input_tensor)[0]
                return tf.tile(self.cls_embedding, [batch_size, 1, 1])

        cls_token_layer = CLSToken(d_model=self.d_model, name="cls_token")
        cls_token = cls_token_layer(input_layer)
        
        return cls_token

    def _transformer_encoder_block(self, inputs: tf.Tensor, name_prefix: str) -> tf.Tensor:
        """
        Standard Transformer encoder block with multi-head self-attention and feed-forward network.
        """
        # Multi-Head Self-Attention
        attn_output = MultiHeadAttention(
            num_heads=self.transformer_heads,
            key_dim=self.d_model // self.transformer_heads,
            dropout=self.transformer_dropout_rate,
            name=f"{name_prefix}_mha"
        )(query=inputs, value=inputs, key=inputs)
        
        # Add & Norm
        out1 = Add(name=f"{name_prefix}_add1")([inputs, attn_output])
        out1 = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(out1)
        
        # Feed-Forward Network
        ffn_output = Dense(self.transformer_dff, activation='relu', name=f"{name_prefix}_ffn1")(out1)
        ffn_output = Dense(self.d_model, name=f"{name_prefix}_ffn2")(ffn_output)
        ffn_output = Dropout(self.transformer_dropout_rate, name=f"{name_prefix}_ffn_dropout")(ffn_output)
        
        # Add & Norm
        out2 = Add(name=f"{name_prefix}_add2")([out1, ffn_output])
        out2 = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(out2)
        
        return out2

    def _build_regression_head(self, cls_representation: tf.Tensor) -> tf.Tensor:
        """
        Build the multi-task regression head.
        """
        x = cls_representation
        
        # MLP layers
        for i, num_units in enumerate(self.mlp_head_units):
            x = Dense(num_units, activation='relu', name=f"mlp_head_dense_{i+1}")(x)
            if self.mlp_dropout_rate > 0:
                x = Dropout(self.mlp_dropout_rate, name=f"mlp_head_dropout_{i+1}")(x)
        
        # Final output layer
        if len(self.predicted_attributes) == 1:
            # Single output for single attribute
            output = Dense(1, activation='linear', name="final_output")(x)
        else:
            # Multi-output for multiple attributes
            output = Dense(len(self.predicted_attributes), activation='linear', name="multi_output")(x)
        
        return output

    def _build_model_for_attr(self) -> tf.keras.Model:
        """
        Build the complete Component-Driven Attention Transformer V2 for single attribute prediction.
        """
        print(f"[CDAT-V2] Building single-attribute model")
        print(f"[CDAT-V2] Model shape: {self.model_shape}")
        print(f"[CDAT-V2] Components: {self.components}")
        print(f"[CDAT-V2] Component dimensions: {self.component_dimensions}")
        
        H, W, C_total = self.model_shape
        input_layer = Input(shape=(H, W, C_total), name="input_combined_special_images")
        
        # 1. Slice input into components
        component_slices = self._slice_input_components(input_layer)
        
        # 2. Verify reflectance component exists
        if 'reflectance' not in component_slices:
            raise ValueError("CDAT-V2 requires reflectance component to be enabled for CNN backbone")
        
        # 3. Process reflectance through CNN backbone with downsampling
        reflectance_input = component_slices['reflectance']
        spatial_tokens = self._build_reflectance_backbone(reflectance_input)
        
        # 4. Add positional embeddings to spatial tokens (now at downsampled resolution)
        spatial_tokens = self._add_positional_embeddings(spatial_tokens)
        
        # 5. Get active advisor components (excluding reflectance)
        advisor_components = {name: component_slices[name] 
                            for name in self._get_active_advisor_components() 
                            if name in component_slices}
        
        if not advisor_components:
            print("[CDAT-V2] Warning: No advisor components active. Using spatial tokens only.")
            enhanced_tokens = spatial_tokens
            component_representations = []
        else:
            # 6. Apply component-driven attention
            if self.fusion_method == "sequential":
                enhanced_tokens, component_representations = self._sequential_component_fusion(
                    spatial_tokens, advisor_components
                )
            else:
                enhanced_tokens, component_representations = self._parallel_component_fusion(
                    spatial_tokens, advisor_components
                )
        
        # 7. Create CLS token
        cls_token = self._create_learnable_cls_token(input_layer)
        
        # 8. Prepare final token sequence
        all_tokens = [cls_token]
        
        if isinstance(component_representations, list):
            # Sequential fusion: add each component representation
            for i, comp_repr in enumerate(component_representations):
                comp_token = Lambda(
                    lambda x: tf.expand_dims(x, axis=1),
                    name=f"comp_token_{i}"
                )(comp_repr)
                all_tokens.append(comp_token)
        elif component_representations is not None:
            # Parallel fusion: add fused representation
            fused_token = Lambda(
                lambda x: tf.expand_dims(x, axis=1),
                name="fused_token"
            )(component_representations)
            all_tokens.append(fused_token)
        
        # Add enhanced spatial tokens (global pooled representation for efficiency)
        spatial_summary = GlobalAveragePooling1D(name="spatial_summary")(enhanced_tokens)
        spatial_token = Lambda(
            lambda x: tf.expand_dims(x, axis=1),
            name="spatial_token"
        )(spatial_summary)
        all_tokens.append(spatial_token)
        
        # 9. Concatenate all tokens
        concatenated_tokens = Concatenate(axis=1, name="final_token_concatenation")(all_tokens)
        
        print(f"[CDAT-V2] Final token sequence length: {len(all_tokens)}")
        
        # 10. Final transformer layers for global integration
        transformer_output = concatenated_tokens
        for i in range(self.final_transformer_layers):
            transformer_output = self._transformer_encoder_block(
                transformer_output, name_prefix=f"final_transformer_{i+1}"
            )
        
        # 11. Extract CLS token for prediction
        cls_final = Lambda(lambda x: x[:, 0, :], name="extract_final_cls")(transformer_output)
        
        # 12. Regression head
        prediction = self._build_regression_head(cls_final)
        
        # 13. Create model
        model = Model(inputs=input_layer, outputs=prediction, name="ComponentDrivenAttentionTransformerV2")
        
        # 14. Compile model
        lr = self.optimizer_params.get('learning_rate', 1e-4)
        wd = self.optimizer_params.get('weight_decay', 1e-4)
        optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        print(f"[CDAT-V2] Model built successfully with {model.count_params()} parameters")
        
        return model

    def _build_multi_output_model(self) -> tf.keras.Model:
        """
        Build the complete Component-Driven Attention Transformer V2 for multi-attribute prediction.
        """
        print(f"[CDAT-V2] Building multi-output model for all attributes: {self.predicted_attributes}")
        
        # The architecture is the same as single attribute, but the final layer outputs multiple values
        # We can reuse the same building logic
        model = self._build_model_for_attr()
        
        # Update the model name
        model._name = "ComponentDrivenAttentionTransformerV2MultiRegression"
        
        print(f"[CDAT-V2] Multi-output model built with {len(self.predicted_attributes)} outputs")
        
        return model 