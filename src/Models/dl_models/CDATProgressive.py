import tensorflow as tf
from tensorflow import keras
from keras import Input, Model, layers
from keras._tf_keras.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, ReLU, Flatten, Dense, Dropout, Concatenate,
    LayerNormalization, MultiHeadAttention, Add, Embedding, Reshape, GlobalAveragePooling2D,
    SeparableConv2D, AveragePooling2D, GlobalAveragePooling1D, Multiply
)
from keras._tf_keras.keras.optimizers import AdamW

from ..base_classes.BaseDLModel import BaseDLModel
from src.config.enums import SpectralIndex, ModelType
from typing import List, Optional, Tuple, Dict
import numpy as np
import src.config as config


class CDATProgressive(BaseDLModel):
    """
    CDAT Progressive V1 - Component-Driven Attention Transformer with Progressive Elements and Pixel Masking.
    
    This model extends CDAT V1 with progressive spatial memory accumulation, where each component
    builds upon previous attention maps instead of always attending to static reflectance tokens.
    Uses pixel masking to filter out padding tokens for ~61% token reduction.
    
    Architecture:
    1. CNN backbone processes reflectance data → spatial features
    2. Pixel masking filters tokens to only tomato pixels → spatial memory (base memory)
    3. Component encoders process other components → component advisors  
    4. Progressive spatial memory: Each component enhances spatial tokens progressively
    5. Self-attention layers for global integration
    6. Global pooling → regression head
    
    Progressive Memory Hierarchy (tomato quality assessment):
    STD (texture/firmness) → Indexes (biochemical/TSS/pH) → NDSI (defects/quality assessment)
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
        
        # CNN Backbone parameters
        backbone_filters_start: int = 32,
        backbone_filters_growth: float = 1.5,
        backbone_kernels: Tuple[int, int] = (3, 3),
        backbone_depth: int = 3,
        backbone_use_pooling: bool = False,
        backbone_pool_size: Tuple[int, int] = (2, 2),
        
        # Component encoder parameters
        component_filters: int = 16,
        component_depth: int = 2,
        
        # Progressive attention parameters
        d_model: int = 128,
        num_attention_heads: int = 4,
        
        # Self-attention parameters
        self_attention_layers: int = 2,
        transformer_dff: int = 256,
        transformer_dropout_rate: float = 0.1,
        
        # Component processing order for progressive memory
        component_processing_order: Optional[List[str]] = None,
        
        # Progressive memory parameters
        memory_accumulation_method: str = "weighted_sum",  # "weighted_sum", "gated_fusion", "attention_blend"
        progressive_scaling: float = 0.1,  # Scale factor for progressive memory updates
        use_memory_gates: bool = True,     # Whether to use learnable gates for memory fusion
        
        # Spatial component queries parameters
        use_spatial_component_queries: bool = True,  # Enhanced spatial component queries
        query_dimension_per_head: int = 32,
        
        # MLP Head parameters
        mlp_head_units: List[int] = [128, 64],
        mlp_dropout_rate: float = 0.2,
        
        # Optimizer parameters
        optimizer_params: Optional[Dict] = None,
    ):
        # Store CDAT Progressive specific parameters
        self.backbone_filters_start = backbone_filters_start
        self.backbone_filters_growth = backbone_filters_growth
        self.backbone_kernels = backbone_kernels
        self.backbone_depth = backbone_depth
        self.backbone_use_pooling = backbone_use_pooling
        self.backbone_pool_size = backbone_pool_size
        
        self.component_filters = component_filters
        self.component_depth = component_depth
        
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.query_dimension_per_head = query_dimension_per_head
        
        self.self_attention_layers = self_attention_layers
        self.transformer_dff = transformer_dff
        self.transformer_dropout_rate = transformer_dropout_rate
        
        # Use configurable component order for progressive memory hierarchy
        self.component_processing_order = component_processing_order or config.CDAT_COMPONENT_ORDER
        print(f"[CDAT Progressive V1] Progressive memory hierarchy: {self.component_processing_order}")
        
        self.memory_accumulation_method = memory_accumulation_method
        self.progressive_scaling = progressive_scaling
        self.use_memory_gates = use_memory_gates
        self.use_spatial_component_queries = use_spatial_component_queries
        
        self.mlp_head_units = mlp_head_units
        self.mlp_dropout_rate = mlp_dropout_rate
        
        self.optimizer_params = optimizer_params if optimizer_params else {}
        
        # Handle case where component_dimensions is None (fallback calculation)
        if component_dimensions is None:
            from utils.model_utils import get_component_dimensions
            if selected_bands is None:
                import config as cfg
                selected_bands = cfg.PATENT_BANDS
            component_dimensions = get_component_dimensions(components, selected_bands)
            print(f"[CDAT Progressive V1] Calculated component_dimensions: {component_dimensions}")
        
        # Call parent constructor
        super().__init__(
            model_type=ModelType.CDAT_PROGRESSIVE,  
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
        Get active components that will act as progressive memory advisors.
        Reflectance is processed by CNN backbone, others become progressive advisors.
        """
        advisor_components = []
        
        for component_name in self.component_processing_order:
            if (self.components.get(component_name, False) and 
                self.component_dimensions.get(component_name, 0) > 0):
                advisor_components.append(component_name)
        
        print(f"[CDAT Progressive V1] Active progressive advisors: {advisor_components}")
        return advisor_components

    def _slice_input_components(self, input_layer: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Slice the input tensor into individual components.
        """
        component_slices = {}
        current_offset = 0
        
        # Process components in fixed order for consistency
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
                
                print(f"[CDAT Progressive V1] {component_name}: channels {start_offset}:{end_offset} ({num_channels} channels)")
        
        return component_slices

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

    def _build_reflectance_backbone(self, reflectance_input: tf.Tensor) -> tf.Tensor:
        """
        Build CNN backbone that processes reflectance data to create spatial features.
        Returns spatial features (B, H, W, d_backbone) for masking.
        """
        print(f"[CDAT Progressive V1] Building CNN backbone for spatial features")
        print(f"[CDAT Progressive V1] Input shape: {reflectance_input.shape}")
        
        x = reflectance_input
        current_filters = self.backbone_filters_start
        
        for i in range(self.backbone_depth):
            x = Conv2D(
                filters=int(current_filters),
                kernel_size=self.backbone_kernels,
                padding='same',
                name=f"backbone_conv{i+1}"
            )(x)
            x = BatchNormalization(name=f"backbone_bn{i+1}")(x)
            x = ReLU(name=f"backbone_relu{i+1}")(x)
            
            # Optionally add pooling (not on last layer)
            if self.backbone_use_pooling and i < self.backbone_depth - 1:
                x = MaxPooling2D(pool_size=self.backbone_pool_size, name=f"backbone_pool{i+1}")(x)
            
            # Grow filters for next layer
            if i < self.backbone_depth - 1:
                current_filters *= self.backbone_filters_growth
        
        print(f"[CDAT Progressive V1] CNN output shape: {x.shape}, keeping spatial dimensions for masking")
        
        return x

    def _build_component_encoder(self, component_input: tf.Tensor, component_name: str,
                                tomato_mask: tf.Tensor) -> tf.Tensor:
        """
        Build component encoder that creates component representation for progressive memory guidance.
        Uses masking to focus only on tomato pixels.
        """
        print(f"[CDAT Progressive V1] Building component encoder for {component_name}")
        print(f"[CDAT Progressive V1] Component input shape: {component_input.shape}")
        
        x = component_input
        current_filters = self.component_filters
        
        # Lightweight CNN for component processing
        for i in range(self.component_depth):
            x = SeparableConv2D(
                filters=current_filters,
                kernel_size=(3, 3),
                padding='same',
                name=f"{component_name}_conv{i+1}"
            )(x)
            x = BatchNormalization(name=f"{component_name}_bn{i+1}")(x)
            x = ReLU(name=f"{component_name}_relu{i+1}")(x)
            
            if i < self.component_depth - 1:
                current_filters *= 2  # Grow filters
        
        # Filter by tomato mask and get masked global representation
        filtered_features, _ = self._filter_tokens_by_mask(x, tomato_mask, f"_{component_name}")
        
        # Global pooling to get component representation for memory guidance
        global_repr = GlobalAveragePooling1D(name=f"{component_name}_global_pool")(filtered_features)
        
        # Project to d_model dimension
        component_repr = Dense(self.d_model, name=f"{component_name}_projection")(global_repr)
        
        print(f"[CDAT Progressive V1] Component {component_name} representation shape: {component_repr.shape}")
        return component_repr

    def _create_spatial_component_queries(self, spatial_memory: tf.Tensor, 
                                        component_repr: tf.Tensor, 
                                        component_name: str,
                                        attention_mask: tf.Tensor) -> tf.Tensor:
        """
        Create spatial component-guided queries from component representation and spatial memory.
        Each spatial location gets its own component-informed query.
        Uses attention mask to prevent queries for padding tokens.
        """
        print(f"[CDAT Progressive V1] Creating spatial component queries for {component_name}")
        
        # Expand component representation to all spatial locations
        # (B, d_model) → (B, 1, d_model) → (B, N_tomato, d_model)
        component_expanded = layers.Lambda(
            lambda x: tf.expand_dims(x, axis=1),
            name=f"{component_name}_expand_component"
        )(component_repr)  # (B, 1, d_model)
        
        # Get the sequence length from spatial memory and broadcast
        component_broadcast = layers.Lambda(
            lambda inputs: tf.tile(inputs[0], [1, tf.shape(inputs[1])[1], 1]),
            name=f"{component_name}_broadcast_component"
        )([component_expanded, spatial_memory])  # (B, N_tomato, d_model)
        
        # Combine spatial memory with component guidance to create queries
        # Each spatial location gets component-informed query
        if self.use_spatial_component_queries:
            # Component-guided spatial queries
            combined_features = Concatenate(axis=-1, name=f"{component_name}_combine_memory_component")([
                spatial_memory, component_broadcast
            ])
            
            # Project to query dimension
            spatial_queries = Dense(
                self.d_model, 
                activation='relu',
                name=f"{component_name}_spatial_queries"
            )(combined_features)
        else:
            # Simple component broadcast as queries
            spatial_queries = component_broadcast
        
        # Mask out queries for padding tokens
        mask_expanded = layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1),
            name=f"{component_name}_expand_attention_mask"
        )(attention_mask)  # (B, N_tomato, 1)
        spatial_queries = Multiply(name=f"{component_name}_mask_queries")([spatial_queries, mask_expanded])
        
        print(f"[CDAT Progressive V1] Spatial queries shape for {component_name}: {spatial_queries.shape}")
        return spatial_queries

    def _progressive_spatial_memory_attention(self, current_memory: tf.Tensor,
                                            component_representations: Dict[str, tf.Tensor],
                                            attention_mask: tf.Tensor) -> tf.Tensor:
        """
        Apply progressive spatial memory accumulation where each component enhances 
        the spatial memory progressively, building a hierarchical understanding.
        Uses attention mask to handle variable-length sequences.
        
        Progressive Memory Hierarchy:
        1. STD (texture/firmness) - enhances base reflectance memory
        2. Indexes (biochemical/TSS/pH) - builds upon texture-enhanced memory  
        3. NDSI (defects/quality) - builds upon biochemical-enhanced memory
        """
        print(f"[CDAT Progressive V1] Applying progressive spatial memory accumulation")
        
        memory_evolution = current_memory  # Start with initial reflectance-based memory
        
        # Process components in hierarchical order
        for i, component_name in enumerate(self.component_processing_order):
            if component_name in component_representations:
                print(f"[CDAT Progressive V1] Progressive Stage {i+1}: Enhancing memory with {component_name}")
                
                component_repr = component_representations[component_name]
                
                # Create spatial component-guided queries
                spatial_queries = self._create_spatial_component_queries(
                    memory_evolution, component_repr, component_name, attention_mask
                )
                
                # Multi-head attention: spatial queries attend to current memory state
                enhanced_memory = MultiHeadAttention(
                    num_heads=self.num_attention_heads,
                    key_dim=self.d_model // self.num_attention_heads,
                    dropout=self.transformer_dropout_rate,
                    name=f"{component_name}_progressive_attention"
                )(
                    query=spatial_queries,    # (B, N_tomato, d_model) - component-guided spatial queries
                    key=memory_evolution,     # (B, N_tomato, d_model) - current memory state
                    value=memory_evolution    # (B, N_tomato, d_model) - current memory state
                    # Note: Skip attention_mask to avoid shape issues, rely on masked tokens instead
                )
                
                # Progressive memory accumulation
                if self.memory_accumulation_method == "weighted_sum":
                    # Scaled additive accumulation
                    scaled_enhancement = layers.Lambda(
                        lambda x: self.progressive_scaling * x,
                        name=f"{component_name}_scale_enhancement"
                    )(enhanced_memory)
                    
                    memory_evolution = Add(name=f"{component_name}_accumulate_memory")([
                        memory_evolution, scaled_enhancement
                    ])
                    
                elif self.memory_accumulation_method == "gated_fusion":
                    # Learnable gated fusion
                    if self.use_memory_gates:
                        # Learn gate weights for memory vs enhancement
                        gate_features = Concatenate(axis=-1, name=f"{component_name}_gate_features")([
                            memory_evolution, enhanced_memory
                        ])
                        
                        memory_gate = Dense(
                            self.d_model, 
                            activation='sigmoid',
                            name=f"{component_name}_memory_gate"
                        )(gate_features)
                        
                        enhancement_gate = layers.Lambda(
                            lambda x: 1.0 - x,
                            name=f"{component_name}_enhancement_gate"
                        )(memory_gate)
                        
                        # Gated combination
                        gated_memory = Multiply(name=f"{component_name}_gate_memory")([
                            memory_evolution, memory_gate
                        ])
                        gated_enhancement = Multiply(name=f"{component_name}_gate_enhancement")([
                            enhanced_memory, enhancement_gate
                        ])
                        
                        memory_evolution = Add(name=f"{component_name}_gated_accumulate")([
                            gated_memory, gated_enhancement
                        ])
                    else:
                        # Simple weighted combination
                        memory_evolution = layers.Lambda(
                            lambda x: 0.7 * x[0] + 0.3 * x[1],
                            name=f"{component_name}_weighted_accumulate"
                        )([memory_evolution, enhanced_memory])
                
                elif self.memory_accumulation_method == "attention_blend":
                    # Attention-based blending
                    blend_attention = MultiHeadAttention(
                        num_heads=2,
                        key_dim=self.d_model // 2,
                        name=f"{component_name}_blend_attention"
                    )(
                        query=memory_evolution,
                        key=enhanced_memory,
                        value=enhanced_memory
                        # Note: Skip attention_mask to avoid shape issues
                    )
                    
                    memory_evolution = Add(name=f"{component_name}_blend_accumulate")([
                        memory_evolution, blend_attention
                    ])
                
                # Apply attention mask to memory evolution
                mask_expanded = layers.Lambda(
                    lambda x: tf.expand_dims(x, axis=-1),
                    name=f"{component_name}_expand_memory_mask"
                )(attention_mask)  # (B, N_tomato, 1)
                memory_evolution = Multiply(name=f"{component_name}_mask_memory")([memory_evolution, mask_expanded])
                
                # Layer normalization after each progressive step
                memory_evolution = LayerNormalization(
                    epsilon=1e-6,
                    name=f"{component_name}_progressive_memory_ln"
                )(memory_evolution)
                
                print(f"[CDAT Progressive V1] Enhanced memory shape after {component_name}: {memory_evolution.shape}")
        
        print(f"[CDAT Progressive V1] Final progressive memory shape: {memory_evolution.shape}")
        return memory_evolution

    def _add_positional_embeddings_with_positions(self, spatial_memory: tf.Tensor, 
                                                 pixel_positions: tf.Tensor) -> tf.Tensor:
        """
        Add learnable positional embeddings to filtered spatial memory tokens using their positions.
        """
        class PositionalEmbeddingWithPositionsLayer(layers.Layer):
            def __init__(self, model_shape, d_model, **kwargs):
                super().__init__(**kwargs)
                self.d_model = d_model
                self.model_shape = model_shape  # Fix: Store model_shape properly
                
                # Calculate maximum possible positions
                H, W, _ = model_shape
                max_positions = H * W
                
                self.pos_embedding = Embedding(
                    input_dim=max_positions,
                    output_dim=d_model,
                    name="spatial_memory_positional_embedding"
                )
            
            def call(self, inputs):
                spatial_tokens, pixel_positions = inputs
                
                # Convert 2D positions to 1D indices
                # positions: (B, N_tomato, 2) -> (B, N_tomato)
                H, W, _ = self.model_shape
                row_positions = pixel_positions[:, :, 0]  # (B, N_tomato)
                col_positions = pixel_positions[:, :, 1]  # (B, N_tomato)
                
                # Convert to 1D indices: row * W + col
                position_indices = tf.cast(row_positions * W + col_positions, tf.int32)
                
                # Handle invalid positions (marked as -1) by setting them to 0
                position_indices = tf.where(position_indices < 0, 0, position_indices)
                
                # Clamp position indices to valid range to prevent out-of-bounds access
                max_positions = H * W - 1
                position_indices = tf.clip_by_value(position_indices, 0, max_positions)
                
                # Get positional embeddings
                pos_embeddings = self.pos_embedding(position_indices)  # (B, N_tomato, d_model)
                
                # Add positional embeddings to tokens
                enhanced_tokens = spatial_tokens + pos_embeddings
                
                return enhanced_tokens
            
            def build(self, input_shape):
                super().build(input_shape)
                # Model shape is already stored in __init__
        
        pos_layer = PositionalEmbeddingWithPositionsLayer(
            model_shape=self.model_shape,
            d_model=self.d_model,
            name="add_memory_positional_embeddings"
        )
        
        return pos_layer([spatial_memory, pixel_positions])

    def _transformer_encoder_block(self, inputs: tf.Tensor, attention_mask: tf.Tensor, 
                                  name_prefix: str) -> tf.Tensor:
        """
        Standard Transformer encoder block with multi-head self-attention and masking.
        """
        # Multi-Head Self-Attention with mask
        attn_output = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.d_model // self.num_attention_heads,
            dropout=self.transformer_dropout_rate,
            name=f"{name_prefix}_mha"
        )(query=inputs, value=inputs, key=inputs)
        # Note: Skip attention_mask to avoid shape issues, rely on masked inputs instead
        
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

    def _build_regression_head(self, global_representation: tf.Tensor) -> tf.Tensor:
        """
        Build the regression head for quality attribute prediction.
        """
        x = global_representation
        
        # MLP layers
        for i, num_units in enumerate(self.mlp_head_units):
            x = Dense(num_units, activation='relu', name=f"mlp_head_dense_{i+1}")(x)
            if self.mlp_dropout_rate > 0:
                x = Dropout(self.mlp_dropout_rate, name=f"mlp_head_dropout_{i+1}")(x)
        
        # Final output layer for multi-regression
        if len(self.predicted_attributes) == 1:
            output = Dense(1, activation='linear', name="final_output")(x)
        else:
            output = Dense(len(self.predicted_attributes), activation='linear', name="multi_output")(x)
        
        return output

    def _build_model_for_attr(self) -> tf.keras.Model:
        """
        Build the complete CDAT Progressive V1 model with pixel masking for single attribute prediction.
        """
        print(f"[CDAT Progressive V1] Building progressive memory model with pixel masking")
        print(f"[CDAT Progressive V1] Model shape: {self.model_shape}")
        print(f"[CDAT Progressive V1] Components: {self.components}")
        print(f"[CDAT Progressive V1] Component dimensions: {self.component_dimensions}")
        print(f"[CDAT Progressive V1] Memory accumulation method: {self.memory_accumulation_method}")
        
        H, W, C_total = self.model_shape
        input_layer = Input(shape=(H, W, C_total), name="input_combined_special_images")
        
        # 1. Slice input into components
        component_slices = self._slice_input_components(input_layer)
        
        # 2. Verify reflectance component exists
        if 'reflectance' not in component_slices:
            raise ValueError("CDAT Progressive requires reflectance component for spatial memory backbone")
        
        # 3. Process reflectance through CNN backbone to create spatial features
        reflectance_input = component_slices['reflectance']
        spatial_features = self._build_reflectance_backbone(reflectance_input)
        
        # 4. Create tomato pixel mask
        tomato_mask = self._create_tomato_pixel_mask(reflectance_input)
        
        # 5. Filter tokens by mask to get initial spatial memory
        initial_spatial_memory, pixel_positions = self._filter_tokens_by_mask(
            spatial_features, tomato_mask, "_main"
        )
        
        # 6. Add positional embeddings to spatial memory using pixel positions
        spatial_memory = self._add_positional_embeddings_with_positions(
            initial_spatial_memory, pixel_positions
        )
        
        # 7. Create attention mask for variable-length sequences
        attention_mask = self._create_attention_mask(pixel_positions)
        
        # 8. Build component encoders for progressive memory advisors
        component_representations = {}
        active_advisors = self._get_active_advisor_components()
        
        for component_name in active_advisors:
            if component_name in component_slices:
                component_repr = self._build_component_encoder(
                    component_slices[component_name], component_name, tomato_mask
                )
                component_representations[component_name] = component_repr
        
        # 9. Apply progressive spatial memory accumulation
        if component_representations:
            enhanced_memory = self._progressive_spatial_memory_attention(
                spatial_memory, component_representations, attention_mask
            )
        else:
            print("[CDAT Progressive V1] Warning: No advisor components active. Using initial memory only.")
            enhanced_memory = spatial_memory
        
        # 10. Self-attention layers for global integration of enhanced memory
        transformer_output = enhanced_memory
        for i in range(self.self_attention_layers):
            transformer_output = self._transformer_encoder_block(
                transformer_output, attention_mask, name_prefix=f"memory_integration_layer_{i+1}"
            )
        
        # 11. Masked global average pooling to get final representation
        # Use masked pooling to exclude padding tokens
        class MaskedGlobalPooling(layers.Layer):
            def call(self, inputs):
                tokens, mask = inputs
                mask_expanded = tf.expand_dims(mask, axis=-1)  # (B, N_tomato, 1)
                masked_tokens = tokens * mask_expanded
                
                # Calculate mean only over valid tokens
                valid_token_count = tf.reduce_sum(mask, axis=1, keepdims=True)  # (B, 1)
                global_representation = tf.reduce_sum(masked_tokens, axis=1) / (valid_token_count + 1e-8)  # (B, d_model)
                return global_representation
        
        masked_pooling = MaskedGlobalPooling(name="global_memory_pooling")
        global_representation = masked_pooling([transformer_output, attention_mask])
        
        # 12. Regression head
        prediction = self._build_regression_head(global_representation)
        
        # 13. Create model
        model = Model(inputs=input_layer, outputs=prediction, name="CDATProgressiveV1")
        
        # 14. Compile model
        lr = self.optimizer_params.get('learning_rate', 1e-4)
        wd = self.optimizer_params.get('weight_decay', 1e-4)
        optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        print(f"[CDAT Progressive V1] Model built successfully with {model.count_params()} parameters")
        
        return model

    def _build_multi_output_model(self) -> tf.keras.Model:
        """
        Build the CDAT Progressive V1 model for multi-attribute prediction.
        """
        print(f"[CDAT Progressive V1] Building multi-output progressive memory model with pixel masking")
        print(f"[CDAT Progressive V1] Target attributes: {self.predicted_attributes}")
        
        # The architecture is the same as single attribute, but the final layer outputs multiple values
        model = self._build_model_for_attr()
        
        # Update the model name
        model._name = "CDATProgressiveV1MultiRegression"
        
        print(f"[CDAT Progressive V1] Multi-output model built with {len(self.predicted_attributes)} outputs")
        
        return model 