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


class CDATProgressiveV2(BaseDLModel):
    """
    CDAT Progressive V2 - Component-Driven Attention Transformer with Progressive Elements and Spatial Downsampling.
    
    This model extends CDAT V2 with progressive spatial memory accumulation, where each component
    builds upon previous attention maps instead of always attending to static reflectance tokens.
    Uses CNN spatial downsampling for 4x attention computation reduction while preserving spatial relationships.
    
    Architecture:
    1. CNN backbone processes reflectance data with downsampling → spatial tokens (base memory)
    2. Component encoders process other components with matching downsampling → component advisors  
    3. Progressive spatial memory: Each component enhances spatial tokens progressively
    4. Self-attention layers for global integration
    5. Global pooling → regression head
    
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
        
        # CNN Backbone parameters with downsampling
        backbone_filters_before_downsample: List[int] = [32],  # Filters before downsampling
        backbone_filters_after_downsample: List[int] = [48, 64],  # Filters after downsampling
        backbone_kernels: Tuple[int, int] = (3, 3),
        downsampling_factor: int = 2,  # How much to downsample (2 = 2x2 regions)
        downsampling_method: str = "strided_conv",  # "strided_conv", "max_pool", "avg_pool"
        
        # Component encoder parameters
        component_filters_before_downsample: List[int] = [16],  # Lighter processing for advisors
        component_filters_after_downsample: List[int] = [16],
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
        # Store CDAT Progressive V2 specific parameters
        self.backbone_filters_before_downsample = backbone_filters_before_downsample
        self.backbone_filters_after_downsample = backbone_filters_after_downsample
        self.backbone_kernels = backbone_kernels
        self.downsampling_factor = downsampling_factor
        self.downsampling_method = downsampling_method
        
        self.component_filters_before_downsample = component_filters_before_downsample
        self.component_filters_after_downsample = component_filters_after_downsample
        self.component_depth = component_depth
        
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.query_dimension_per_head = query_dimension_per_head
        
        self.self_attention_layers = self_attention_layers
        self.transformer_dff = transformer_dff
        self.transformer_dropout_rate = transformer_dropout_rate
        
        # Use configurable component order for progressive memory hierarchy
        self.component_processing_order = component_processing_order or config.CDAT_COMPONENT_ORDER
        print(f"[CDAT Progressive V2] Progressive memory hierarchy: {self.component_processing_order}")
        
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
            print(f"[CDAT Progressive V2] Calculated component_dimensions: {component_dimensions}")
        
        # Calculate downsampled dimensions
        H, W, _ = model_shape
        self.downsampled_h = H // self.downsampling_factor
        self.downsampled_w = W // self.downsampling_factor
        self.downsampled_spatial_tokens = self.downsampled_h * self.downsampled_w
        
        print(f"[CDAT Progressive V2] Original spatial dims: {H}x{W} ({H*W} positions)")
        print(f"[CDAT Progressive V2] Downsampled dims: {self.downsampled_h}x{self.downsampled_w} ({self.downsampled_spatial_tokens} tokens)")
        print(f"[CDAT Progressive V2] Spatial reduction: {(H*W)/self.downsampled_spatial_tokens:.1f}x")
        print(f"[CDAT Progressive V2] Downsampling method: {self.downsampling_method}")
        
        # Call parent constructor
        super().__init__(
            model_type=ModelType.CDAT_PROGRESSIVE_V2,  # V2 uses spatial downsampling
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
        
        print(f"[CDAT Progressive V2] Active progressive advisors: {advisor_components}")
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
                
                print(f"[CDAT Progressive V2] {component_name}: channels {start_offset}:{end_offset} ({num_channels} channels)")
        
        return component_slices

    def _apply_downsampling_layer(self, x: tf.Tensor, layer_prefix: str) -> tf.Tensor:
        """
        Apply downsampling based on the configured method.
        """
        if self.downsampling_method == "strided_conv":
            # Learnable downsampling using strided convolution
            x = Conv2D(
                filters=x.shape[-1],  # Keep same number of filters
                kernel_size=self.downsampling_factor,
                strides=self.downsampling_factor,
                padding='same',
                name=f"{layer_prefix}_strided_downsample"
            )(x)
        elif self.downsampling_method == "max_pool":
            # Fixed downsampling using max pooling
            x = MaxPooling2D(
                pool_size=self.downsampling_factor,
                strides=self.downsampling_factor,
                padding='same',
                name=f"{layer_prefix}_max_downsample"
            )(x)
        elif self.downsampling_method == "avg_pool":
            # Fixed downsampling using average pooling
            x = AveragePooling2D(
                pool_size=self.downsampling_factor,
                strides=self.downsampling_factor,
                padding='same',
                name=f"{layer_prefix}_avg_downsample"
            )(x)
        else:
            raise ValueError(f"Unknown downsampling method: {self.downsampling_method}")
        
        return x

    def _build_reflectance_backbone(self, reflectance_input: tf.Tensor) -> tf.Tensor:
        """
        Build CNN backbone with progressive downsampling that processes reflectance data.
        Returns spatial tokens at downsampled resolution for progressive memory.
        """
        print(f"[CDAT Progressive V2] Building CNN backbone with downsampling")
        print(f"[CDAT Progressive V2] Input shape: {reflectance_input.shape}")
        
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
        
        print(f"[CDAT Progressive V2] After pre-downsampling layers: {x.shape}")
        
        # Phase 2: Downsampling
        x = self._apply_downsampling_layer(x, "backbone")
        print(f"[CDAT Progressive V2] After downsampling: {x.shape}")
        
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
        
        # Convert spatial feature map to tokens for progressive memory
        B, H_down, W_down, d_backbone = x.shape
        N_down = H_down * W_down
        
        print(f"[CDAT Progressive V2] Final CNN output shape: {x.shape}, creating {N_down} spatial memory tokens")
        
        # Flatten spatial dimensions: (B, H_down, W_down, d_backbone) → (B, N_down, d_backbone)
        spatial_tokens = Reshape((N_down, d_backbone), name="spatial_memory_tokens")(x)
        
        # Project to common dimension: (B, N_down, d_backbone) → (B, N_down, d_model)
        spatial_tokens = Dense(self.d_model, name="spatial_memory_projection")(spatial_tokens)
        
        return spatial_tokens

    def _build_component_encoder(self, component_input: tf.Tensor, component_name: str) -> tf.Tensor:
        """
        Build component encoder that creates component representation for progressive memory guidance.
        Uses matching downsampling to align with reflectance backbone resolution.
        """
        print(f"[CDAT Progressive V2] Building component encoder for {component_name}")
        print(f"[CDAT Progressive V2] Component input shape: {component_input.shape}")
        
        x = component_input
        
        # Phase 1: Light feature extraction before downsampling
        for i, num_filters in enumerate(self.component_filters_before_downsample):
            x = SeparableConv2D(
                filters=num_filters,
                kernel_size=(3, 3),
                padding='same',
                name=f"{component_name}_pre_conv{i+1}"
            )(x)
            x = BatchNormalization(name=f"{component_name}_pre_bn{i+1}")(x)
            x = ReLU(name=f"{component_name}_pre_relu{i+1}")(x)
        
        # Phase 2: Downsampling to match backbone resolution
        x = self._apply_downsampling_layer(x, f"{component_name}_component")
        print(f"[CDAT Progressive V2] {component_name} after downsampling: {x.shape}")
        
        # Phase 3: Feature refinement after downsampling
        for i, num_filters in enumerate(self.component_filters_after_downsample):
            x = SeparableConv2D(
                filters=num_filters,
                kernel_size=(3, 3),
                padding='same',
                name=f"{component_name}_post_conv{i+1}"
            )(x)
            x = BatchNormalization(name=f"{component_name}_post_bn{i+1}")(x)
            x = ReLU(name=f"{component_name}_post_relu{i+1}")(x)
        
        # Global pooling to get component representation for memory guidance
        global_repr = GlobalAveragePooling2D(name=f"{component_name}_global_pool")(x)
        
        # Project to d_model dimension
        component_repr = Dense(self.d_model, name=f"{component_name}_projection")(global_repr)
        
        print(f"[CDAT Progressive V2] Component {component_name} representation shape: {component_repr.shape}")
        return component_repr

    def _create_spatial_component_queries(self, spatial_memory: tf.Tensor, 
                                        component_repr: tf.Tensor, 
                                        component_name: str) -> tf.Tensor:
        """
        Create spatial component-guided queries from component representation and spatial memory.
        Each spatial location gets its own component-informed query.
        """
        print(f"[CDAT Progressive V2] Creating spatial component queries for {component_name}")
        
        # Expand component representation to all spatial locations
        # (B, d_model) → (B, 1, d_model) → (B, N_down, d_model)
        component_expanded = layers.Lambda(
            lambda x: tf.expand_dims(x[0], axis=1),
            name=f"{component_name}_expand_component"
        )([component_repr])
        
        component_broadcast = layers.Lambda(
            lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], 1]),
            name=f"{component_name}_broadcast_component"
        )([component_expanded, spatial_memory])
        
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
        
        print(f"[CDAT Progressive V2] Spatial queries shape for {component_name}: {spatial_queries.shape}")
        return spatial_queries

    def _progressive_spatial_memory_attention(self, current_memory: tf.Tensor,
                                            component_representations: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Apply progressive spatial memory accumulation where each component enhances 
        the spatial memory progressively, building a hierarchical understanding.
        
        Progressive Memory Hierarchy:
        1. STD (texture/firmness) - enhances base reflectance memory
        2. Indexes (biochemical/TSS/pH) - builds upon texture-enhanced memory  
        3. NDSI (defects/quality) - builds upon biochemical-enhanced memory
        """
        print(f"[CDAT Progressive V2] Applying progressive spatial memory accumulation")
        
        memory_evolution = current_memory  # Start with initial reflectance-based memory
        
        # Process components in hierarchical order
        for i, component_name in enumerate(self.component_processing_order):
            if component_name in component_representations:
                print(f"[CDAT Progressive V2] Progressive Stage {i+1}: Enhancing memory with {component_name}")
                
                component_repr = component_representations[component_name]
                
                # Create spatial component-guided queries
                spatial_queries = self._create_spatial_component_queries(
                    memory_evolution, component_repr, component_name
                )
                
                # Multi-head attention: spatial queries attend to current memory state
                enhanced_memory = MultiHeadAttention(
                    num_heads=self.num_attention_heads,
                    key_dim=self.d_model // self.num_attention_heads,
                    dropout=self.transformer_dropout_rate,
                    name=f"{component_name}_progressive_attention"
                )(
                    query=spatial_queries,    # (B, N_down, d_model) - component-guided spatial queries
                    key=memory_evolution,     # (B, N_down, d_model) - current memory state
                    value=memory_evolution    # (B, N_down, d_model) - current memory state
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
                    )
                    
                    memory_evolution = Add(name=f"{component_name}_blend_accumulate")([
                        memory_evolution, blend_attention
                    ])
                
                # Layer normalization after each progressive step
                memory_evolution = LayerNormalization(
                    epsilon=1e-6,
                    name=f"{component_name}_progressive_memory_ln"
                )(memory_evolution)
                
                print(f"[CDAT Progressive V2] Enhanced memory shape after {component_name}: {memory_evolution.shape}")
        
        print(f"[CDAT Progressive V2] Final progressive memory shape: {memory_evolution.shape}")
        return memory_evolution

    def _add_positional_embeddings(self, spatial_memory: tf.Tensor) -> tf.Tensor:
        """
        Add learnable positional embeddings to spatial memory tokens.
        Works with downsampled spatial dimensions.
        """
        class PositionalEmbeddingLayer(layers.Layer):
            def __init__(self, model_shape, downsampling_factor, d_model, **kwargs):
                super().__init__(**kwargs)
                self.d_model = d_model
                
                # Calculate maximum possible positions for downsampled dimensions
                H, W, _ = model_shape
                max_h = (H // downsampling_factor) + 1  # +1 for safety
                max_w = (W // downsampling_factor) + 1  # +1 for safety
                max_positions = max_h * max_w
                
                self.pos_embedding = Embedding(
                    input_dim=max_positions,
                    output_dim=d_model,
                    name="spatial_memory_positional_embedding"
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
            name="add_memory_positional_embeddings"
        )
        
        return pos_layer(spatial_memory)

    def _transformer_encoder_block(self, inputs: tf.Tensor, name_prefix: str) -> tf.Tensor:
        """
        Standard Transformer encoder block with multi-head self-attention.
        """
        # Multi-Head Self-Attention
        attn_output = MultiHeadAttention(
            num_heads=self.num_attention_heads,
            key_dim=self.d_model // self.num_attention_heads,
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
        Build the complete CDAT Progressive V2 model with spatial downsampling for single attribute prediction.
        """
        print(f"[CDAT Progressive V2] Building progressive memory model with spatial downsampling")
        print(f"[CDAT Progressive V2] Model shape: {self.model_shape}")
        print(f"[CDAT Progressive V2] Components: {self.components}")
        print(f"[CDAT Progressive V2] Component dimensions: {self.component_dimensions}")
        print(f"[CDAT Progressive V2] Memory accumulation method: {self.memory_accumulation_method}")
        
        H, W, C_total = self.model_shape
        input_layer = Input(shape=(H, W, C_total), name="input_combined_special_images")
        
        # 1. Slice input into components
        component_slices = self._slice_input_components(input_layer)
        
        # 2. Verify reflectance component exists
        if 'reflectance' not in component_slices:
            raise ValueError("CDAT Progressive V2 requires reflectance component for spatial memory backbone")
        
        # 3. Process reflectance through CNN backbone with downsampling to create initial spatial memory
        reflectance_input = component_slices['reflectance']
        initial_spatial_memory = self._build_reflectance_backbone(reflectance_input)
        
        # 4. Add positional embeddings to spatial memory
        spatial_memory = self._add_positional_embeddings(initial_spatial_memory)
        
        # 5. Build component encoders for progressive memory advisors
        component_representations = {}
        active_advisors = self._get_active_advisor_components()
        
        for component_name in active_advisors:
            if component_name in component_slices:
                component_repr = self._build_component_encoder(
                    component_slices[component_name], component_name
                )
                component_representations[component_name] = component_repr
        
        # 6. Apply progressive spatial memory accumulation
        if component_representations:
            enhanced_memory = self._progressive_spatial_memory_attention(
                spatial_memory, component_representations
            )
        else:
            print("[CDAT Progressive V2] Warning: No advisor components active. Using initial memory only.")
            enhanced_memory = spatial_memory
        
        # 7. Self-attention layers for global integration of enhanced memory
        transformer_output = enhanced_memory
        for i in range(self.self_attention_layers):
            transformer_output = self._transformer_encoder_block(
                transformer_output, name_prefix=f"memory_integration_layer_{i+1}"
            )
        
        # 8. Global average pooling to get final representation
        global_representation = GlobalAveragePooling1D(name="global_memory_pooling")(transformer_output)
        
        # 9. Regression head
        prediction = self._build_regression_head(global_representation)
        
        # 10. Create model
        model = Model(inputs=input_layer, outputs=prediction, name="CDATProgressiveV2")
        
        # 11. Compile model
        lr = self.optimizer_params.get('learning_rate', 1e-4)
        wd = self.optimizer_params.get('weight_decay', 1e-4)
        optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        print(f"[CDAT Progressive V2] Model built successfully with {model.count_params()} parameters")
        
        return model

    def _build_multi_output_model(self) -> tf.keras.Model:
        """
        Build the CDAT Progressive V2 model for multi-attribute prediction.
        """
        print(f"[CDAT Progressive V2] Building multi-output progressive memory model with spatial downsampling")
        print(f"[CDAT Progressive V2] Target attributes: {self.predicted_attributes}")
        
        # The architecture is the same as single attribute, but the final layer outputs multiple values
        model = self._build_model_for_attr()
        
        # Update the model name
        model._name = "CDATProgressiveV2MultiRegression"
        
        print(f"[CDAT Progressive V2] Multi-output model built with {len(self.predicted_attributes)} outputs")
        
        return model 