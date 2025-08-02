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


class ComponentDrivenAttentionTransformer(BaseDLModel):
    """
    Component-Driven Attention Transformer (CDAT) - A novel architecture that uses
    a single CNN backbone for reflectance processing while other spectral components
    act as attention directors that guide where the model should focus.
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
        
        # CNN Backbone specific parameters (single backbone instead of multiple branches)
        backbone_filters_start: int = 32,
        backbone_filters_growth: float = 1.5,
        backbone_kernels: Tuple[int, int] = (3, 3),
        backbone_depth: int = 3,  # Deeper since it's the only CNN
        backbone_use_pooling: bool = False,
        backbone_pool_size: Tuple[int, int] = (2, 2),
        
        # Component attention parameters
        d_model: int = 128,  # Common dimension for tokens and representations
        component_attention_heads: int = 4,  # Number of attention heads per component
        num_attention_layers_per_component: int = 1,
        
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
        # Store CDAT-specific parameters
        self.backbone_filters_start = backbone_filters_start
        self.backbone_filters_growth = backbone_filters_growth
        self.backbone_kernels = backbone_kernels
        self.backbone_depth = backbone_depth
        self.backbone_use_pooling = backbone_use_pooling
        self.backbone_pool_size = backbone_pool_size
        
        self.d_model = d_model
        self.component_attention_heads = component_attention_heads
        self.num_attention_layers_per_component = num_attention_layers_per_component
        
        self.fusion_method = fusion_method
        self.component_processing_order = component_processing_order or config.CDAT_COMPONENT_ORDER
        
        print(f"[CDAT] Component processing order: {self.component_processing_order}")
        
        self.final_transformer_layers = final_transformer_layers
        if d_model % transformer_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by transformer_heads ({transformer_heads}).")
        self.transformer_heads = transformer_heads
        self.transformer_dff = transformer_dff
        self.transformer_dropout_rate = transformer_dropout_rate
        
        self.mlp_head_units = mlp_head_units
        self.mlp_dropout_rate = mlp_dropout_rate
        
        self.optimizer_params = optimizer_params if optimizer_params else {}
        
        # Call parent constructor
        super().__init__(
            model_type=ModelType.COMPONENT_DRIVEN_ATTENTION_TRANSFORMER,
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
        
        print(f"[CDAT] Active advisor components: {advisor_components}")
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
                
                print(f"[CDAT] {component_name}: channels {start_offset}:{end_offset} ({num_channels} channels)")
        
        return component_slices

    def _build_reflectance_backbone(self, reflectance_input: tf.Tensor) -> tf.Tensor:
        """
        Build a single powerful CNN backbone that processes reflectance data.
        Returns spatial features (B, H, W, d_backbone) instead of flattened tokens
        to allow for efficient masking of padding pixels.
        """
        print(f"[CDAT] Building CNN backbone with input shape: {reflectance_input.shape}")
        
        x = reflectance_input
        current_filters = self.backbone_filters_start
        skip_connections = []
        
        for i in range(self.backbone_depth):
            x = Conv2D(
                filters=int(current_filters),
                kernel_size=self.backbone_kernels,
                padding='same',
                name=f"backbone_conv{i+1}"
            )(x)
            x = BatchNormalization(name=f"backbone_bn{i+1}")(x)
            x = ReLU(name=f"backbone_relu{i+1}")(x)
            
            # Add residual connections for deeper network (skip first layer)
            if i > 0 and len(skip_connections) > 0:
                # Ensure compatible dimensions for residual connection
                skip_conn = skip_connections[-1]
                if skip_conn.shape[-1] == x.shape[-1]:
                    x = Add(name=f"backbone_residual{i+1}")([x, skip_conn])
            
            skip_connections.append(x)
            
            # Optionally add pooling (not on last layer)
            if self.backbone_use_pooling and i < self.backbone_depth - 1:
                x = MaxPooling2D(pool_size=self.backbone_pool_size, name=f"backbone_pool{i+1}")(x)
            
            # Grow filters for next layer
            if i < self.backbone_depth - 1:
                current_filters *= self.backbone_filters_growth
        
        print(f"[CDAT] CNN output shape: {x.shape} (keeping spatial dimensions for masking)")
        
        return x

    def _create_tomato_pixel_mask(self, reflectance_input: tf.Tensor, threshold: float = 1e-6) -> tf.Tensor:
        """
        Create a binary mask identifying tomato pixels (non-zero) vs padding pixels (zero).
        Since tomato pixels form a contiguous region, we can filter out padding efficiently.
        
        Args:
            reflectance_input: Reflectance tensor of shape (B, H, W, C_refl)
            threshold: Minimum value to consider as non-padding (default: 1e-6)
            
        Returns:
            Binary mask of shape (B, H, W) where 1 = tomato pixel, 0 = padding pixel
        """
        class TomatoMaskLayer(layers.Layer):
            def __init__(self, threshold=1e-6, **kwargs):
                super().__init__(**kwargs)
                self.threshold = threshold
                
            def call(self, reflectance_input):
                # Sum across all reflectance channels to get total reflectance per pixel
                total_reflectance = tf.reduce_sum(tf.abs(reflectance_input), axis=-1)  # (B, H, W)
                
                # Create binary mask: 1 for tomato pixels, 0 for padding
                tomato_mask = tf.cast(total_reflectance > self.threshold, tf.float32)  # (B, H, W)
                
                return tomato_mask
        
        mask_layer = TomatoMaskLayer(threshold=threshold, name="create_tomato_mask")
        return mask_layer(reflectance_input)

    def _filter_tokens_by_mask(self, spatial_features: tf.Tensor, tomato_mask: tf.Tensor, 
                              layer_suffix: str = "") -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Filter spatial features to keep only tomato pixels, removing padding tokens.
        Uses a more efficient approach that works with symbolic tensors.
        
        Args:
            spatial_features: Features of shape (B, H, W, d_backbone)
            tomato_mask: Binary mask of shape (B, H, W)
            
        Returns:
            Tuple of (filtered_tokens, pixel_positions):
            - filtered_tokens: (B, N_max, d_backbone) where N_max is max tomato pixels across batch
            - pixel_positions: (B, N_max, 2) containing original (row, col) positions
        """
        class EfficientTokenFilterLayer(layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                
            def call(self, inputs):
                spatial_features, tomato_mask = inputs
                
                # Get shapes
                batch_size = tf.shape(spatial_features)[0]
                height = tf.shape(spatial_features)[1]
                width = tf.shape(spatial_features)[2]
                d_backbone = tf.shape(spatial_features)[3]
                
                # Flatten spatial dimensions for easier processing
                # spatial_features: (B, H, W, d_backbone) -> (B, H*W, d_backbone)
                flattened_features = tf.reshape(spatial_features, [batch_size, height * width, d_backbone])
                
                # tomato_mask: (B, H, W) -> (B, H*W)
                flattened_mask = tf.reshape(tomato_mask, [batch_size, height * width])
                
                # Create position indices for each pixel
                # Create coordinate grids
                row_indices = tf.range(height, dtype=tf.float32)
                col_indices = tf.range(width, dtype=tf.float32)
                row_grid, col_grid = tf.meshgrid(row_indices, col_indices, indexing='ij')
                
                # Flatten and stack to get (H*W, 2) position matrix
                positions = tf.stack([
                    tf.reshape(row_grid, [-1]),
                    tf.reshape(col_grid, [-1])
                ], axis=1)  # (H*W, 2)
                
                # Expand to batch dimension: (1, H*W, 2) -> (B, H*W, 2)
                batch_positions = tf.tile(tf.expand_dims(positions, 0), [batch_size, 1, 1])
                
                # Find maximum number of tomato pixels across the batch
                num_tomato_per_sample = tf.reduce_sum(flattened_mask, axis=1)  # (B,)
                max_tomato_pixels = tf.reduce_max(num_tomato_per_sample)
                max_tomato_pixels = tf.cast(max_tomato_pixels, tf.int32)
                
                # For each sample, extract top-k tomato pixels (where k = max_tomato_pixels)
                # Use top_k to get indices of tomato pixels, padding with zeros if needed
                
                # Create large negative values for padding pixels to ensure they're not selected
                mask_scores = flattened_mask * 1e6 + tf.random.uniform(tf.shape(flattened_mask), 0, 1)
                
                # Get top-k indices (these will be the tomato pixel indices)
                _, top_indices = tf.nn.top_k(mask_scores, k=max_tomato_pixels, sorted=False)  # (B, max_tomato_pixels)
                
                # Gather features and positions using these indices
                batch_indices = tf.expand_dims(tf.range(batch_size), 1)  # (B, 1)
                batch_indices = tf.tile(batch_indices, [1, max_tomato_pixels])  # (B, max_tomato_pixels)
                
                # Create gather indices: (B, max_tomato_pixels, 2) where last dim is [batch_idx, spatial_idx]
                gather_indices = tf.stack([batch_indices, top_indices], axis=-1)
                
                # Gather features and positions
                filtered_features = tf.gather_nd(flattened_features, gather_indices)  # (B, max_tomato_pixels, d_backbone)
                filtered_positions = tf.gather_nd(batch_positions, gather_indices)  # (B, max_tomato_pixels, 2)
                
                # Create validity mask (1 for real tomato pixels, 0 for padding)
                gathered_mask_values = tf.gather_nd(flattened_mask, gather_indices)  # (B, max_tomato_pixels)
                
                # Zero out features for padding pixels
                validity_mask = tf.expand_dims(gathered_mask_values, -1)  # (B, max_tomato_pixels, 1)
                filtered_features = filtered_features * validity_mask
                
                # Set positions to -1 for padding pixels
                invalid_positions = tf.ones_like(filtered_positions) * -1.0
                filtered_positions = tf.where(
                    tf.expand_dims(gathered_mask_values > 0.5, -1),
                    filtered_positions,
                    invalid_positions
                )
                
                return filtered_features, filtered_positions
                
            def compute_output_shape(self, input_shape):
                spatial_shape, mask_shape = input_shape
                batch_size = spatial_shape[0]
                d_backbone = spatial_shape[3]
                # We don't know max_tomato_pixels at graph construction time, so use None
                return [(batch_size, None, d_backbone), (batch_size, None, 2)]
        
        filter_layer = EfficientTokenFilterLayer(name=f"efficient_filter_tomato_tokens{layer_suffix}")
        filtered_tokens, pixel_positions = filter_layer([spatial_features, tomato_mask])
        
        # Project filtered tokens to model dimension
        projected_tokens = Dense(self.d_model, name=f"filtered_token_projection{layer_suffix}")(filtered_tokens)
        
        return projected_tokens, pixel_positions

    def _add_positional_embeddings_with_positions(self, spatial_tokens: tf.Tensor, pixel_positions: tf.Tensor) -> tf.Tensor:
        """
        Add learnable positional embeddings based on original 2D positions of tomato pixels.
        """
        class PositionalEmbeddingWithCoords(layers.Layer):
            def __init__(self, max_h, max_w, d_model, **kwargs):
                super().__init__(**kwargs)
                self.max_h = max_h
                self.max_w = max_w
                self.d_model = d_model
                
                # Separate embeddings for row and column positions
                self.row_embedding = Embedding(
                    input_dim=max_h,
                    output_dim=d_model // 2,
                    name="row_positional_embedding"
                )
                self.col_embedding = Embedding(
                    input_dim=max_w,
                    output_dim=d_model // 2,
                    name="col_positional_embedding"
                )
                
            def call(self, inputs):
                spatial_tokens, pixel_positions = inputs
                
                # Extract row and column positions
                row_positions = tf.cast(pixel_positions[:, :, 0], tf.int32)  # (B, N_tomato)
                col_positions = tf.cast(pixel_positions[:, :, 1], tf.int32)  # (B, N_tomato)
                
                # Handle padding positions (marked as -1)
                valid_mask = pixel_positions[:, :, 0] >= 0  # (B, N_tomato)
                row_positions = tf.where(valid_mask, row_positions, 0)  # Use 0 for padding
                col_positions = tf.where(valid_mask, col_positions, 0)  # Use 0 for padding
                
                # Get positional embeddings
                row_embeds = self.row_embedding(row_positions)  # (B, N_tomato, d_model//2)
                col_embeds = self.col_embedding(col_positions)  # (B, N_tomato, d_model//2)
                
                # Concatenate row and column embeddings
                pos_embeddings = tf.concat([row_embeds, col_embeds], axis=-1)  # (B, N_tomato, d_model)
                
                # Add positional embeddings to tokens (zero out for padding positions)
                valid_mask_expanded = tf.expand_dims(tf.cast(valid_mask, tf.float32), axis=-1)
                pos_embeddings = pos_embeddings * valid_mask_expanded
                
                enhanced_tokens = spatial_tokens + pos_embeddings
                return enhanced_tokens
        
        pos_layer = PositionalEmbeddingWithCoords(
            max_h=self.model_shape[0],
            max_w=self.model_shape[1],
            d_model=self.d_model,
            name="add_positional_embeddings_with_coords"
        )
        
        return pos_layer([spatial_tokens, pixel_positions])

    def _create_attention_mask(self, pixel_positions: tf.Tensor) -> tf.Tensor:
        """
        Create attention mask to prevent attending to padding tokens.
        
        Args:
            pixel_positions: (B, N_tomato, 2) where padding positions are marked as -1
            
        Returns:
            attention_mask: (B, N_tomato) where 1 = valid token, 0 = padding token
        """
        class AttentionMaskLayer(layers.Layer):
            def call(self, pixel_positions):
                # Valid tokens have non-negative positions
                attention_mask = tf.cast(pixel_positions[:, :, 0] >= 0, tf.float32)  # (B, N_tomato)
                return attention_mask
        
        mask_layer = AttentionMaskLayer(name="create_attention_mask")
        return mask_layer(pixel_positions)

    def _extract_local_patterns(self, component_input: tf.Tensor, component_name: str) -> tf.Tensor:
        """
        Extract local spatial patterns from component images using efficient operations.
        """
        # Use separable convolutions for efficiency
        x = SeparableConv2D(
            32, (3, 3), 
            padding='same', 
            name=f"{component_name}_local_conv"
        )(component_input)
        x = BatchNormalization(name=f"{component_name}_local_bn")(x)
        x = ReLU(name=f"{component_name}_local_relu")(x)
        
        # Multi-scale pooling to capture different spatial scales
        pool_1 = AveragePooling2D((2, 2), name=f"{component_name}_pool_1")(x)
        pool_2 = AveragePooling2D((4, 4), name=f"{component_name}_pool_2")(x)
        
        # Global average pooling for each scale
        global_1 = GlobalAveragePooling2D(name=f"{component_name}_global_1")(pool_1)
        global_2 = GlobalAveragePooling2D(name=f"{component_name}_global_2")(pool_2)
        
        # Combine multi-scale features
        local_features = Concatenate(name=f"{component_name}_local_concat")([global_1, global_2])
        
        return local_features

    def _build_component_query_generator(self, component_input: tf.Tensor, component_name: str, 
                                         tomato_mask: tf.Tensor) -> tf.Tensor:
        """
        Generate spatial queries that preserve spatial structure instead of global pooling.
        Each component provides different 'perspectives' on what to focus on.
        """
        print(f"[CDAT] Building spatial query generator for {component_name} with input shape: {component_input.shape}")
        
        # Light CNN processing to extract features while maintaining spatial dimensions
        x = SeparableConv2D(
            16, (3, 3), 
            padding='same', 
            name=f"{component_name}_spatial_conv"
        )(component_input)
        x = BatchNormalization(name=f"{component_name}_spatial_bn")(x)
        x = ReLU(name=f"{component_name}_spatial_relu")(x)
        
        # Filter spatial features by tomato mask (reuse the filtering logic)
        filtered_features, _ = self._filter_tokens_by_mask(x, tomato_mask, f"_{component_name}")
        
        # Generate queries for each tomato spatial location and attention head
        d_k = self.d_model // self.component_attention_heads
        total_query_dim = self.component_attention_heads * d_k
        
        # Project to query dimensions: (B, N_tomato, 16) -> (B, N_tomato, total_query_dim)
        spatial_queries = Dense(
            total_query_dim,
            activation='relu',
            name=f"{component_name}_spatial_queries"
        )(filtered_features)
        
        # Reshape for multi-head attention: (B, N_tomato, total_query_dim) -> (B, N_tomato, num_heads, d_k)
        # Use -1 to let TensorFlow infer the sequence length dimension
        queries = Reshape(
            (-1, self.component_attention_heads, d_k),
            name=f"{component_name}_reshape_spatial_queries"
        )(spatial_queries)
        
        print(f"[CDAT] Generated filtered spatial queries for {component_name}")
        return queries

    def _component_driven_attention(self, spatial_tokens: tf.Tensor, component_queries: tf.Tensor, 
                                   component_name: str, attention_mask: tf.Tensor) -> tf.Tensor:
        """
        Apply spatial-to-spatial attention where queries come from components and 
        keys/values are the spatial tokens from CNN backbone.
        """
        print(f"[CDAT] Applying spatial {component_name} attention")
        
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
                spatial_tokens, component_queries, attention_mask = inputs
                # spatial_tokens: (B, N', d_model)
                # component_queries: (B, N', num_heads, d_k)
                
                # Generate keys and values from spatial tokens
                keys = self.keys_dense(spatial_tokens)  # (B, N', d_model)
                values = self.values_dense(spatial_tokens)  # (B, N', d_model)
                
                # Get shapes
                B = tf.shape(spatial_tokens)[0]
                N_prime = tf.shape(spatial_tokens)[1]
                
                # Reshape keys and values for multi-head attention
                keys = tf.reshape(keys, [B, N_prime, self.num_heads, self.d_k])
                values = tf.reshape(values, [B, N_prime, self.num_heads, self.d_k])
                
                # Apply multi-head spatial attention
                attention_outputs = []
                for head in range(self.num_heads):
                    # Extract head-specific tensors
                    q = component_queries[:, :, head, :]  # (B, N', d_k) - Query per spatial location
                    k = keys[:, :, head, :]            # (B, N', d_k) - Key per spatial location
                    v = values[:, :, head, :]          # (B, N', d_k) - Value per spatial location
                    
                    # Spatial-to-spatial attention (each location can attend to all others)
                    attention_scores = tf.matmul(q, k, transpose_b=True) / tf.sqrt(tf.cast(self.d_k, tf.float32))
                    
                    # Apply attention mask to prevent attending to padding tokens
                    mask_expanded = tf.expand_dims(attention_mask, axis=1)  # (B, 1, N_tomato)
                    mask_scores = mask_expanded * (-1e9)  # Large negative for padding
                    masked_scores = attention_scores + mask_scores
                    
                    attention_weights = tf.nn.softmax(masked_scores, axis=-1)  # (B, N_tomato, N_tomato)
                    
                    # Apply attention to values
                    attended_values = tf.matmul(attention_weights, v)  # (B, N_tomato, d_k)
                    attention_outputs.append(attended_values)
                
                # Concatenate multi-head outputs: (B, N', d_model)
                multi_head_output = tf.concat(attention_outputs, axis=-1)
                
                # Final projection
                component_enhanced_tokens = self.final_projection(multi_head_output)
                
                # Masked global pooling to get component representation (exclude padding tokens)
                mask_expanded = tf.expand_dims(attention_mask, axis=-1)  # (B, N_tomato, 1)
                masked_tokens = component_enhanced_tokens * mask_expanded
                
                # Calculate mean only over valid tokens
                valid_token_count = tf.reduce_sum(attention_mask, axis=1, keepdims=True)  # (B, 1)
                component_representation = tf.reduce_sum(masked_tokens, axis=1) / (valid_token_count + 1e-8)  # (B, d_model)
                
                return component_representation
        
        attention_layer = ComponentAttentionLayer(
            d_model=self.d_model,
            num_heads=self.component_attention_heads,
            component_name=component_name,
            name=f"{component_name}_spatial_attention"
        )
        
        component_representation = attention_layer([spatial_tokens, component_queries, attention_mask])
        
        print(f"[CDAT] {component_name} spatial attention output shape: {component_representation.shape}")
        
        return component_representation

    def _sequential_component_fusion(self, spatial_tokens: tf.Tensor, 
                                   component_inputs: Dict[str, tf.Tensor],
                                   tomato_mask: tf.Tensor,
                                   attention_mask: tf.Tensor) -> Tuple[tf.Tensor, List[tf.Tensor]]:
        """
        Apply component attentions sequentially with spatial queries and stable updates.
        """
        print(f"[CDAT] Applying sequential component fusion")
        
        current_tokens = spatial_tokens
        component_representations = []
        
        # Process components in the specified order
        active_components = [comp for comp in self.component_processing_order 
                           if comp in component_inputs]
        
        print(f"[CDAT] Processing components in order: {active_components}")
        
        for component_name in active_components:
            component_input = component_inputs[component_name]
            
            # Generate spatial queries from current component
            spatial_queries = self._build_component_query_generator(
                component_input, component_name, tomato_mask
            )
            
            # Apply spatial component attention
            component_repr = self._component_driven_attention(
                current_tokens, spatial_queries, component_name, attention_mask
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
                    component_broadcast = tf.tile(component_info, [1, N_tokens, 1])  # (B, N', d_model)
                    
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
                                 component_inputs: Dict[str, tf.Tensor],
                                 tomato_mask: tf.Tensor,
                                 attention_mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply all component attentions in parallel with spatial queries.
        """
        print(f"[CDAT] Applying parallel component fusion")
        
        component_representations = []
        
        for component_name, component_input in component_inputs.items():
            # Generate spatial queries
            spatial_queries = self._build_component_query_generator(
                component_input, component_name, tomato_mask
            )
            # Apply spatial attention
            component_repr = self._component_driven_attention(
                spatial_tokens, spatial_queries, component_name, attention_mask
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
        Build the complete Component-Driven Attention Transformer for single attribute prediction.
        """
        print(f"[CDAT] Building single-attribute model")
        print(f"[CDAT] Model shape: {self.model_shape}")
        print(f"[CDAT] Components: {self.components}")
        print(f"[CDAT] Component dimensions: {self.component_dimensions}")
        
        H, W, C_total = self.model_shape
        input_layer = Input(shape=(H, W, C_total), name="input_combined_special_images")
        
        # 1. Slice input into components
        component_slices = self._slice_input_components(input_layer)
        
        # 2. Verify reflectance component exists
        if 'reflectance' not in component_slices:
            raise ValueError("CDAT requires reflectance component to be enabled for CNN backbone")
        
        # 3. Process reflectance through single CNN backbone
        reflectance_input = component_slices['reflectance']
        spatial_features = self._build_reflectance_backbone(reflectance_input)
        
        # 4. Create tomato pixel mask
        tomato_mask = self._create_tomato_pixel_mask(reflectance_input)
        
        # 5. Filter tokens by mask
        filtered_tokens, pixel_positions = self._filter_tokens_by_mask(spatial_features, tomato_mask, "_main")
        
        # 6. Add positional embeddings with positions
        spatial_tokens = self._add_positional_embeddings_with_positions(filtered_tokens, pixel_positions)
        
        # 7. Create attention mask
        attention_mask = self._create_attention_mask(pixel_positions)
        
        # 8. Get active advisor components (excluding reflectance)
        advisor_components = {name: component_slices[name] 
                            for name in self._get_active_advisor_components() 
                            if name in component_slices}
        
        if not advisor_components:
            print("[CDAT] Warning: No advisor components active. Using spatial tokens only.")
            enhanced_tokens = spatial_tokens
            component_representations = []
        else:
            # 9. Apply component-driven attention
            if self.fusion_method == "sequential":
                enhanced_tokens, component_representations = self._sequential_component_fusion(
                    spatial_tokens, advisor_components, tomato_mask, attention_mask
                )
            else:
                enhanced_tokens, component_representations = self._parallel_component_fusion(
                    spatial_tokens, advisor_components, tomato_mask, attention_mask
                )
        
        # 10. Create CLS token
        cls_token = self._create_learnable_cls_token(input_layer)
        
        # 11. Prepare final token sequence
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
        
        # Add enhanced spatial tokens (masked pooling for efficiency)
        # Use masked global pooling to get spatial summary (exclude padding tokens)
        class MaskedGlobalPooling(layers.Layer):
            def call(self, inputs):
                tokens, mask = inputs
                mask_expanded = tf.expand_dims(mask, axis=-1)  # (B, N_tomato, 1)
                masked_tokens = tokens * mask_expanded
                
                # Calculate mean only over valid tokens
                valid_token_count = tf.reduce_sum(mask, axis=1, keepdims=True)  # (B, 1)
                spatial_summary = tf.reduce_sum(masked_tokens, axis=1) / (valid_token_count + 1e-8)  # (B, d_model)
                return spatial_summary
        
        masked_pooling = MaskedGlobalPooling(name="masked_spatial_pooling")
        spatial_summary = masked_pooling([enhanced_tokens, attention_mask])
        spatial_token = Lambda(
            lambda x: tf.expand_dims(x, axis=1),
            name="spatial_token"
        )(spatial_summary)
        all_tokens.append(spatial_token)
        
        # 12. Concatenate all tokens
        concatenated_tokens = Concatenate(axis=1, name="final_token_concatenation")(all_tokens)
        
        print(f"[CDAT] Final token sequence length: {len(all_tokens)}")
        
        # 13. Final transformer layers for global integration
        transformer_output = concatenated_tokens
        for i in range(self.final_transformer_layers):
            transformer_output = self._transformer_encoder_block(
                transformer_output, name_prefix=f"final_transformer_{i+1}"
            )
        
        # 14. Extract CLS token for prediction
        cls_final = Lambda(lambda x: x[:, 0, :], name="extract_final_cls")(transformer_output)
        
        # 15. Regression head
        prediction = self._build_regression_head(cls_final)
        
        # 16. Create model
        model = Model(inputs=input_layer, outputs=prediction, name="ComponentDrivenAttentionTransformer")
        
        # 17. Compile model
        lr = self.optimizer_params.get('learning_rate', 1e-4)
        wd = self.optimizer_params.get('weight_decay', 1e-4)
        optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        print(f"[CDAT] Model built successfully with {model.count_params()} parameters")
        
        return model

    def _build_multi_output_model(self) -> tf.keras.Model:
        """
        Build the complete Component-Driven Attention Transformer for multi-attribute prediction.
        """
        print(f"[CDAT] Building multi-output model for all attributes: {self.predicted_attributes}")
        
        # The architecture is the same as single attribute, but the final layer outputs multiple values
        # We can reuse the same building logic
        model = self._build_model_for_attr()
        
        # Update the model name
        model._name = "ComponentDrivenAttentionTransformerMultiRegression"
        
        print(f"[CDAT] Multi-output model built with {len(self.predicted_attributes)} outputs")
        
        return model 