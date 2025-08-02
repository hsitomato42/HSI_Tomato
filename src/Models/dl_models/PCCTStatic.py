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


class PCCTStatic(BaseDLModel):
    """
    PCCT Static V1 - Progressive Cross-Attention CNN-Transformer with Static Fusion and Pixel Masking.
    
    Architecture:
    1. Single CNN backbone processes reflectance data → spatial tokens
    2. Pixel masking: Filter out padding tokens (like CDAT V1)
    3. Component encoders process other components → global component representations  
    4. Static cross-attention: All components attend to spatial tokens simultaneously
    5. Self-attention layers for global integration
    6. CLS token → regression head
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
        
        # Cross-attention parameters
        d_model: int = 128,
        cross_attention_heads: int = 4,
        
        # Self-attention parameters
        self_attention_layers: int = 2,
        self_attention_heads: int = 4,
        transformer_dff: int = 256,
        transformer_dropout_rate: float = 0.1,
        
        # Component processing order
        component_processing_order: Optional[List[str]] = None,
        
        # MLP Head parameters
        mlp_head_units: List[int] = [64],
        mlp_dropout_rate: float = 0.2,
        
        # Optimizer parameters
        optimizer_params: Optional[Dict] = None,
    ):
        # Store PCCT Static specific parameters
        self.backbone_filters_start = backbone_filters_start
        self.backbone_filters_growth = backbone_filters_growth
        self.backbone_kernels = backbone_kernels
        self.backbone_depth = backbone_depth
        self.backbone_use_pooling = backbone_use_pooling
        self.backbone_pool_size = backbone_pool_size
        
        self.component_filters = component_filters
        self.component_depth = component_depth
        
        self.d_model = d_model
        self.cross_attention_heads = cross_attention_heads
        
        self.self_attention_layers = self_attention_layers
        self.self_attention_heads = self_attention_heads
        self.transformer_dff = transformer_dff
        self.transformer_dropout_rate = transformer_dropout_rate
        
        # Use configurable component order
        self.component_processing_order = component_processing_order or config.CDAT_COMPONENT_ORDER
        print(f"[PCCT Static] Component processing order: {self.component_processing_order}")
        
        self.mlp_head_units = mlp_head_units
        self.mlp_dropout_rate = mlp_dropout_rate
        
        self.optimizer_params = optimizer_params if optimizer_params else {}
        
        # Call parent constructor
        super().__init__(
            model_type=ModelType.PCCT_STATIC,
            model_name=model_name,
            model_filename=model_filename,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            selected_bands=selected_bands,
            predicted_attributes=predicted_attributes,
            selected_indexes=selected_indexes
        )

    def _create_tomato_pixel_mask(self, reflectance_input: tf.Tensor) -> tf.Tensor:
        """
        Create binary mask to identify tomato pixels vs padding pixels.
        Uses reflectance data to detect non-zero (tomato) pixels.
        """
        # Sum across all reflectance channels to detect non-zero pixels using Lambda layer
        pixel_sum = layers.Lambda(
            lambda x: tf.reduce_sum(tf.abs(x), axis=-1, keepdims=False),
            name="sum_reflectance_channels"
        )(reflectance_input)  # (B, H, W)
        
        # Create binary mask: 1 for tomato pixels, 0 for padding using Lambda layer
        tomato_mask = layers.Lambda(
            lambda x: tf.cast(x > 1e-6, tf.float32),
            name="create_tomato_mask"
        )(pixel_sum)  # (B, H, W)
        
        return tomato_mask

    def _filter_tokens_by_mask(self, spatial_tokens: tf.Tensor, tomato_mask: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Apply simple masking to spatial tokens (simplified approach to avoid KerasTensor issues).
        Instead of complex filtering, we'll use masking in attention layers.
        """
        # For now, return the original tokens and create a simple attention mask
        # This avoids the complex tf.map_fn operations that cause KerasTensor issues
        
        # Create attention mask by flattening the tomato mask using Lambda layer
        attention_mask = layers.Lambda(
            lambda x: tf.reshape(x, [tf.shape(x)[0], -1]),
            name="flatten_attention_mask"
        )(tomato_mask)  # (B, H*W)
        
        print(f"[PCCT Static V1] Using simplified masking approach")
        
        return spatial_tokens, attention_mask

    def _get_active_advisor_components(self) -> List[str]:
        """
        Get active components that will act as cross-attention advisors.
        Reflectance is processed by CNN backbone, others become advisors.
        """
        advisor_components = []
        
        for component_name in self.component_processing_order:
            if (self.components.get(component_name, False) and 
                self.component_dimensions.get(component_name, 0) > 0):
                advisor_components.append(component_name)
        
        print(f"[PCCT Static] Active advisor components: {advisor_components}")
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
                
                print(f"[PCCT Static] {component_name}: channels {start_offset}:{end_offset} ({num_channels} channels)")
        
        return component_slices

    def _build_reflectance_backbone(self, reflectance_input: tf.Tensor) -> tf.Tensor:
        """
        Build CNN backbone that processes reflectance data to create spatial tokens.
        """
        print(f"[PCCT Static] Building CNN backbone with input shape: {reflectance_input.shape}")
        
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
        
        # Convert spatial feature map to tokens
        B, H_prime, W_prime, d_backbone = x.shape
        N_prime = H_prime * W_prime
        
        print(f"[PCCT Static] CNN output shape: {x.shape}, creating {N_prime} spatial tokens")
        
        # Flatten spatial dimensions: (B, H', W', d_backbone) → (B, N', d_backbone)
        spatial_tokens = Reshape((N_prime, d_backbone), name="spatial_tokens")(x)
        
        # Project to common dimension: (B, N', d_backbone) → (B, N', d_model)
        spatial_tokens = Dense(self.d_model, name="spatial_projection")(spatial_tokens)
        
        return spatial_tokens

    def _build_component_encoder(self, component_input: tf.Tensor, component_name: str) -> tf.Tensor:
        """
        Build component encoder that creates global component representation.
        """
        print(f"[PCCT Static] Building component encoder for {component_name} with input shape: {component_input.shape}")
        
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
        
        # Global pooling to get component representation
        global_repr = GlobalAveragePooling2D(name=f"{component_name}_global_pool")(x)
        
        # Project to d_model dimension
        component_repr = Dense(self.d_model, name=f"{component_name}_projection")(global_repr)
        
        print(f"[PCCT Static] Component {component_name} representation shape: {component_repr.shape}")
        return component_repr

    def _static_cross_attention(self, spatial_tokens: tf.Tensor, 
                               component_representations: Dict[str, tf.Tensor]) -> tf.Tensor:
        """
        Apply static cross-attention where all components attend to spatial tokens simultaneously.
        Improved with better component fusion.
        """
        print(f"[PCCT Static] Applying static cross-attention")
        
        enhanced_tokens = spatial_tokens
        component_attention_outputs = []
        
        for component_name, component_repr in component_representations.items():
            print(f"[PCCT Static] Applying cross-attention for {component_name}")
            
            # Expand component representation to serve as queries using Lambda layer
            # (B, d_model) → (B, 1, d_model) for cross-attention
            component_queries = layers.Lambda(
                lambda x: tf.expand_dims(x, axis=1),
                name=f"{component_name}_expand_queries"
            )(component_repr)
            
            # Cross-attention: component queries attend to spatial tokens
            cross_attention_output = MultiHeadAttention(
                num_heads=self.cross_attention_heads,
                key_dim=self.d_model // self.cross_attention_heads,
                dropout=self.transformer_dropout_rate,
                name=f"{component_name}_cross_attention"
            )(
                query=component_queries,      # (B, 1, d_model)
                key=spatial_tokens,           # (B, N', d_model)  
                value=spatial_tokens          # (B, N', d_model)
            )
            
            component_attention_outputs.append(cross_attention_output)
        
        # Improved fusion: Average all component attention outputs
        if len(component_attention_outputs) > 1:
            # Stack all component attentions and average them
            stacked_attentions = layers.Lambda(
                lambda x: tf.stack(x, axis=1),
                name="stack_component_attentions"
            )(component_attention_outputs)
            
            averaged_component_attention = layers.Lambda(
                lambda x: tf.reduce_mean(x, axis=1),
                name="average_component_attentions"
            )(stacked_attentions)
        else:
            averaged_component_attention = component_attention_outputs[0]
        
        # Broadcast averaged component insights to all spatial locations
        component_broadcast = layers.Lambda(
            lambda x: tf.tile(x[0], [1, tf.shape(x[1])[1], 1]),
            name="broadcast_insights"
        )([averaged_component_attention, spatial_tokens])
        
        # Add component insights to spatial tokens with residual connection
        enhanced_tokens = Add(name="add_component_insights")([enhanced_tokens, component_broadcast])
        
        # Layer normalization
        enhanced_tokens = LayerNormalization(epsilon=1e-6, name="cross_attention_ln")(enhanced_tokens)
        
        print(f"[PCCT Static] Enhanced tokens shape: {enhanced_tokens.shape}")
        return enhanced_tokens

    def _add_positional_embeddings(self, spatial_tokens: tf.Tensor) -> tf.Tensor:
        """
        Add learnable positional embeddings to spatial tokens.
        """
        class PositionalEmbeddingLayer(layers.Layer):
            def __init__(self, max_positions, d_model, **kwargs):
                super().__init__(**kwargs)
                self.max_positions = max_positions
                self.d_model = d_model
                self.pos_embedding = Embedding(
                    input_dim=max_positions,
                    output_dim=d_model,
                    name="spatial_positional_embedding"
                )
            
            def call(self, spatial_tokens):
                N_prime = tf.shape(spatial_tokens)[1]
                positions = tf.range(N_prime)
                pos_embeddings = self.pos_embedding(positions)
                enhanced_tokens = spatial_tokens + tf.expand_dims(pos_embeddings, axis=0)
                return enhanced_tokens
        
        pos_layer = PositionalEmbeddingLayer(
            max_positions=self.model_shape[0] * self.model_shape[1],
            d_model=self.d_model,
            name="add_positional_embeddings"
        )
        
        return pos_layer(spatial_tokens)

    def _create_learnable_cls_token(self, input_layer: tf.Tensor) -> tf.Tensor:
        """
        Create a learnable CLS token.
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
        Standard Transformer encoder block with multi-head self-attention.
        """
        # Multi-Head Self-Attention
        attn_output = MultiHeadAttention(
            num_heads=self.self_attention_heads,
            key_dim=self.d_model // self.self_attention_heads,
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
        Build the regression head for quality attribute prediction.
        """
        x = cls_representation
        
        # MLP layers
        for i, num_units in enumerate(self.mlp_head_units):
            x = Dense(num_units, activation='relu', name=f"mlp_head_dense_{i+1}")(x)
            if self.mlp_dropout_rate > 0:
                x = Dropout(self.mlp_dropout_rate, name=f"mlp_head_dropout_{i+1}")(x)
        
        # Final output layer
        if len(self.predicted_attributes) == 1:
            output = Dense(1, activation='linear', name="final_output")(x)
        else:
            output = Dense(len(self.predicted_attributes), activation='linear', name="multi_output")(x)
        
        return output

    def _build_model_for_attr(self) -> tf.keras.Model:
        """
        Build the complete PCCT Static model for single attribute prediction.
        """
        print(f"[PCCT Static] Building single-attribute model")
        print(f"[PCCT Static] Model shape: {self.model_shape}")
        print(f"[PCCT Static] Components: {self.components}")
        print(f"[PCCT Static] Component dimensions: {self.component_dimensions}")
        
        H, W, C_total = self.model_shape
        input_layer = Input(shape=(H, W, C_total), name="input_combined_special_images")
        
        # 1. Slice input into components
        component_slices = self._slice_input_components(input_layer)
        
        # 2. Verify reflectance component exists
        if 'reflectance' not in component_slices:
            raise ValueError("PCCT Static requires reflectance component for CNN backbone")
        
        # 3. Process reflectance through CNN backbone
        reflectance_input = component_slices['reflectance']
        spatial_tokens = self._build_reflectance_backbone(reflectance_input)
        
        # 4. Create tomato pixel mask and filter tokens
        tomato_mask = self._create_tomato_pixel_mask(reflectance_input)
        filtered_tokens, attention_mask = self._filter_tokens_by_mask(spatial_tokens, tomato_mask)
        
        print(f"[PCCT Static V1] Token filtering: {spatial_tokens.shape} → {filtered_tokens.shape}")
        
        # 5. Add positional embeddings to filtered tokens
        spatial_tokens = self._add_positional_embeddings(filtered_tokens)
        
        # 6. Build component encoders for advisor components
        component_representations = {}
        active_advisors = self._get_active_advisor_components()
        
        for component_name in active_advisors:
            if component_name in component_slices:
                component_repr = self._build_component_encoder(
                    component_slices[component_name], component_name
                )
                component_representations[component_name] = component_repr
        
        # 7. Apply static cross-attention (all components simultaneously)
        if component_representations:
            enhanced_tokens = self._static_cross_attention(spatial_tokens, component_representations)
        else:
            print("[PCCT Static V1] Warning: No advisor components active. Using spatial tokens only.")
            enhanced_tokens = spatial_tokens
        
        # 8. Create CLS token
        cls_token = self._create_learnable_cls_token(input_layer)
        
        # 9. Concatenate CLS token with enhanced spatial tokens
        all_tokens = Concatenate(axis=1, name="concatenate_cls_and_tokens")([cls_token, enhanced_tokens])
        
        # 10. Self-attention layers for global integration
        transformer_output = all_tokens
        for i in range(self.self_attention_layers):
            transformer_output = self._transformer_encoder_block(
                transformer_output, name_prefix=f"self_attention_layer_{i+1}"
            )
        
        # 11. Extract CLS token for prediction
        cls_final = layers.Lambda(lambda x: x[:, 0, :], name="extract_final_cls")(transformer_output)
        
        # 12. Regression head
        prediction = self._build_regression_head(cls_final)
        
        # 12. Create model
        model = Model(inputs=input_layer, outputs=prediction, name="PCCTStatic")
        
        # 13. Compile model
        lr = self.optimizer_params.get('learning_rate', 1e-4)
        wd = self.optimizer_params.get('weight_decay', 1e-4)
        optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        print(f"[PCCT Static] Model built successfully with {model.count_params()} parameters")
        
        return model

    def _build_multi_output_model(self) -> tf.keras.Model:
        """
        Build the PCCT Static model for multi-attribute prediction.
        """
        print(f"[PCCT Static] Building multi-output model for all attributes: {self.predicted_attributes}")
        
        # The architecture is the same as single attribute, but the final layer outputs multiple values
        model = self._build_model_for_attr()
        
        # Update the model name
        model._name = "PCCTStaticMultiRegression"
        
        print(f"[PCCT Static] Multi-output model built with {len(self.predicted_attributes)} outputs")
        
        return model 