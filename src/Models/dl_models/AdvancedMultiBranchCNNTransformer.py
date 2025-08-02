import tensorflow as tf
from tensorflow import keras
from keras import Input, Model, layers
from keras._tf_keras.keras.layers import (
    Conv2D, MaxPooling2D, BatchNormalization, ReLU, Flatten, Dense, Dropout, Concatenate,
    LayerNormalization, MultiHeadAttention, Add, Embedding, Reshape
)
from keras._tf_keras.keras.optimizers import AdamW

from ..base_classes.BaseDLModel import BaseDLModel
from src.config.enums import SpectralIndex, ModelType # For type hinting if used by parent
from typing import List, Optional, Tuple, Dict


class AdvancedMultiBranchCNNTransformer(BaseDLModel):
    def __init__(
        self,
        # Parameters from BaseDLModel
        model_name: str,
        model_filename: str,
        model_shape: tuple[int, int, int], # (H, W, C_total)
        components: dict, # {'reflectance': True, 'std': False, ...}
        component_dimensions: dict, # {'reflectance': N_reflectance_channels, ... , 'total': C_total}
        predicted_attributes: list[str],
        selected_bands: Optional[List[int]] = None,
        selected_indexes: Optional[List[SpectralIndex]] = None,
        
        # CNN Branch specific parameters
        cnn_branch_filters_start: int = 32,
        cnn_branch_filters_growth: float = 1.5, # Factor to grow filters by, can be float if cast to int
        cnn_branch_kernels: Tuple[int, int] = (3, 3),
        cnn_branch_depth: int = 2,  # Number of Conv-BN-ReLU blocks per branch
        cnn_branch_use_pooling: bool = False,
        cnn_pool_size: Tuple[int, int] = (2, 2),
        
        # Projection and Transformer parameters
        d_model: int = 128,  # d_f in paper, common dimension for tokens
        transformer_layers: int = 1,  # L in paper, number of Transformer encoder blocks
        transformer_heads: int = 1,  # Number of attention heads
        transformer_dff: int = 256,  # Dimension of the feed-forward network in Transformer
        transformer_dropout_rate: float = 0.3,
        
        # Cross-Attention parameters
        use_cross_attention: bool = False,
        # List of (query_branch_name, key_value_branch_name)
        cross_attention_pairs: Optional[List[Tuple[str, str]]] = None,
        
        # Token filtering parameters
        use_token_filtering: bool = True,  # Enable/disable token filtering
        tomato_mask_threshold: float = 1e-6,  # Threshold for tomato pixel detection
        
        # MLP Head parameters for regression
        mlp_head_units: List[int] = [64],
        mlp_dropout_rate: float = 0.2,
        
        # Optimizer parameters
        optimizer_params: Optional[Dict] =  None, # e.g., {'learning_rate': 1e-4, 'weight_decay': 1e-4}
    ):
        # Initialize AdvancedMultiBranchCNNTransformer specific attributes FIRST
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

        self.use_cross_attention = use_cross_attention
        self.cross_attention_pairs = cross_attention_pairs if cross_attention_pairs else []

        # Token filtering parameters
        self.use_token_filtering = use_token_filtering
        self.tomato_mask_threshold = tomato_mask_threshold

        self.mlp_head_units = mlp_head_units
        self.mlp_dropout_rate = mlp_dropout_rate
        
        self.optimizer_params = optimizer_params if optimizer_params else {}

        # NOW call super().__init__, which will in turn call the overridden _build_model_for_attr
        super().__init__(
            model_type=ModelType.ADVANCED_MULTI_BRANCH_CNN_TRANSFORMER,
            model_name=model_name,
            model_filename=model_filename,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            selected_bands=selected_bands,
            predicted_attributes=predicted_attributes,
            selected_indexes=selected_indexes
        )
        
        # The super().__init__ calls self._build_model_for_attr() for each attribute,
        # which will now use the overridden version from this class and have access to the above attributes.

    def _create_tomato_pixel_mask(self, reflectance_input: tf.Tensor) -> tf.Tensor:
        """
        Create binary mask to identify tomato pixels vs padding pixels.
        Uses reflectance data to detect non-zero (tomato) pixels.
        
        Args:
            reflectance_input: Reflectance tensor of shape (B, H, W, C_refl)
            
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
        
        mask_layer = TomatoMaskLayer(threshold=self.tomato_mask_threshold, name="create_tomato_mask")
        return mask_layer(reflectance_input)

    def _filter_tokens_by_mask(self, spatial_features: tf.Tensor, tomato_mask: tf.Tensor, 
                              layer_suffix: str = "") -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Filter spatial features to keep only tomato pixels, removing padding tokens.
        Uses efficient approach that works with symbolic tensors.
        
        Args:
            spatial_features: Features of shape (B, H, W, d_backbone)
            tomato_mask: Binary mask of shape (B, H, W)
            layer_suffix: Optional suffix for layer names
            
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
        
        return filtered_tokens, pixel_positions

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

    def _add_positional_embeddings_with_positions(self, spatial_tokens: tf.Tensor, 
                                                 pixel_positions: tf.Tensor) -> tf.Tensor:
        """
        Add positional embeddings to spatial tokens using their actual pixel positions.
        
        Args:
            spatial_tokens: (B, N_tomato, d_model) filtered spatial tokens
            pixel_positions: (B, N_tomato, 2) original pixel positions
            
        Returns:
            enhanced_tokens: (B, N_tomato, d_model) with positional embeddings added
        """
        class PositionalEmbeddingWithPositions(layers.Layer):
            def __init__(self, max_h: int, max_w: int, d_model: int, **kwargs):
                super().__init__(**kwargs)
                self.max_h = max_h
                self.max_w = max_w
                self.d_model = d_model
                
            def build(self, input_shape):
                # Create separate embeddings for row and column positions
                self.row_embedding = Embedding(
                    input_dim=self.max_h,
                    output_dim=self.d_model // 2,
                    name="row_positional_embedding"
                )
                self.col_embedding = Embedding(
                    input_dim=self.max_w,
                    output_dim=self.d_model // 2,
                    name="col_positional_embedding"
                )
                super().build(input_shape)
                
            def call(self, inputs):
                spatial_tokens, pixel_positions = inputs
                
                # Extract row and column positions
                row_positions = tf.cast(pixel_positions[:, :, 0], tf.int32)  # (B, N_tomato)
                col_positions = tf.cast(pixel_positions[:, :, 1], tf.int32)  # (B, N_tomato)
                
                # Handle padding positions (marked as -1)
                valid_mask = pixel_positions[:, :, 0] >= 0  # (B, N_tomato)
                row_positions = tf.where(valid_mask, row_positions, 0)  # Use 0 for padding
                col_positions = tf.where(valid_mask, col_positions, 0)  # Use 0 for padding
                
                # Clamp positions to valid range
                row_positions = tf.clip_by_value(row_positions, 0, self.max_h - 1)
                col_positions = tf.clip_by_value(col_positions, 0, self.max_w - 1)
                
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
        
        pos_layer = PositionalEmbeddingWithPositions(
            max_h=self.model_shape[0],
            max_w=self.model_shape[1],
            d_model=self.d_model,
            name="add_positional_embeddings_with_positions"
        )
        
        return pos_layer([spatial_tokens, pixel_positions])

    def _build_cnn_branch(self, input_tensor: tf.Tensor, branch_name: str) -> tf.Tensor:
        x = input_tensor
        print(f"    Inside _build_cnn_branch for {branch_name}, input_tensor symbolic shape: {x.shape}")
        current_filters = self.cnn_branch_filters_start
        
        for i in range(self.cnn_branch_depth):
            print(f"      Loop {i} for {branch_name}, x symbolic shape before Conv2D: {x.shape}")
            x = Conv2D(
                filters=int(current_filters), # Ensure filters is int
                kernel_size=self.cnn_branch_kernels,
                padding='same',
                name=f"{branch_name}_cnn_conv{i+1}"
            )(x)
            x = BatchNormalization(name=f"{branch_name}_cnn_bn{i+1}")(x)
            x = ReLU(name=f"{branch_name}_cnn_relu{i+1}")(x)
            if i < self.cnn_branch_depth - 1: # Don't grow filters for the last conv layer of the branch
                current_filters *= self.cnn_branch_filters_growth
        
        if self.cnn_branch_use_pooling:
            x = MaxPooling2D(pool_size=self.cnn_pool_size, name=f"{branch_name}_cnn_pool")(x)
        
        # Output: Feature map of size H' x W' x d_b (d_b = int(current_filters))
        return x

    def _transformer_encoder_block(self, inputs: tf.Tensor, name_prefix: str, 
                                  attention_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Transformer encoder block with optional attention mask for token filtering.
        
        Args:
            inputs: Input tokens of shape (B, N, d_model)
            name_prefix: Prefix for layer names
            attention_mask: Optional attention mask of shape (B, N) where 1 = valid token, 0 = padding
            
        Returns:
            Encoded tokens of shape (B, N, d_model)
        """
        # Multi-Head Self-Attention
        if attention_mask is not None:
            # Convert attention mask to the format expected by MultiHeadAttention
            # MultiHeadAttention expects (B, 1, 1, N) or (B, 1, N, N)
            # We'll use causal_mask format: (B, 1, 1, N)
            attention_mask_4d = tf.expand_dims(tf.expand_dims(attention_mask, axis=1), axis=1)
            # Convert to additive mask (0 for valid, -inf for padding)
            attention_mask_4d = (1.0 - attention_mask_4d) * -1e9
            
            attn_output = MultiHeadAttention(
                num_heads=self.transformer_heads, 
                key_dim=self.d_model // self.transformer_heads,
                dropout=self.transformer_dropout_rate, 
                name=f"{name_prefix}_mha"
            )(query=inputs, value=inputs, key=inputs, attention_mask=attention_mask_4d)
        else:
            attn_output = MultiHeadAttention(
                num_heads=self.transformer_heads, 
                key_dim=self.d_model // self.transformer_heads,
                dropout=self.transformer_dropout_rate, 
                name=f"{name_prefix}_mha"
            )(query=inputs, value=inputs, key=inputs)
        
        out1 = Add(name=f"{name_prefix}_add1")([inputs, attn_output])
        out1 = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln1")(out1)

        # Feed-Forward Network
        ffn_output = Dense(self.transformer_dff, activation='relu', name=f"{name_prefix}_ffn_dense1")(out1)
        ffn_output = Dense(self.d_model, name=f"{name_prefix}_ffn_dense2")(ffn_output)
        ffn_output = Dropout(self.transformer_dropout_rate, name=f"{name_prefix}_ffn_dropout")(ffn_output)
        
        out2 = Add(name=f"{name_prefix}_add2")([out1, ffn_output])
        out2 = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_ln2")(out2)
        
        # Apply attention mask to output to zero out padding tokens
        if attention_mask is not None:
            mask_expanded = tf.expand_dims(attention_mask, axis=-1)  # (B, N, 1)
            out2 = out2 * mask_expanded
        
        return out2

    def _cross_attention_block(self, query_tokens: tf.Tensor, key_value_tokens: tf.Tensor, 
                              name_prefix: str, attention_mask: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Cross-attention block with optional attention mask.
        
        Args:
            query_tokens: Query tokens of shape (B, N, d_model)
            key_value_tokens: Key/Value tokens of shape (B, N, d_model)
            name_prefix: Prefix for layer names
            attention_mask: Optional attention mask of shape (B, N)
            
        Returns:
            Updated query tokens of shape (B, N, d_model)
        """
        # Cross-Attention: Query from query_tokens, Key/Value from key_value_tokens
        if attention_mask is not None:
            # Convert attention mask to the format expected by MultiHeadAttention
            attention_mask_4d = tf.expand_dims(tf.expand_dims(attention_mask, axis=1), axis=1)
            # Convert to additive mask (0 for valid, -inf for padding)
            attention_mask_4d = (1.0 - attention_mask_4d) * -1e9
            
            cross_attn_output = MultiHeadAttention(
                num_heads=self.transformer_heads, 
                key_dim=self.d_model // self.transformer_heads,
                dropout=self.transformer_dropout_rate, 
                name=f"{name_prefix}_cross_mha"
            )(query=query_tokens, value=key_value_tokens, key=key_value_tokens, 
              attention_mask=attention_mask_4d)
        else:
            cross_attn_output = MultiHeadAttention(
                num_heads=self.transformer_heads, 
                key_dim=self.d_model // self.transformer_heads,
                dropout=self.transformer_dropout_rate, 
                name=f"{name_prefix}_cross_mha"
            )(query=query_tokens, value=key_value_tokens, key=key_value_tokens)
        
        # Add & Norm for query_tokens
        updated_query_tokens = Add(name=f"{name_prefix}_cross_add")([query_tokens, cross_attn_output])
        updated_query_tokens = LayerNormalization(epsilon=1e-6, name=f"{name_prefix}_cross_ln")(updated_query_tokens)
        
        # Apply attention mask to output
        if attention_mask is not None:
            mask_expanded = tf.expand_dims(attention_mask, axis=-1)  # (B, N, 1)
            updated_query_tokens = updated_query_tokens * mask_expanded
        
        return updated_query_tokens

    def _build_model_for_attr(self) -> tf.keras.Model:
        print(f"--- Building model for attribute ---")
        print(f"Initial self.component_dimensions: {self.component_dimensions}")
        print(f"Initial self.components: {self.components}")
        print(f"Token filtering enabled: {self.use_token_filtering}")

        H, W, C_total = self.model_shape
        input_layer = Input(shape=(H, W, C_total), name="input_combined_special_images")
        
        branch_raw_projected_tokens: Dict[str, tf.Tensor] = {}
        branch_pixel_positions: Dict[str, tf.Tensor] = {}
        branch_names_ordered: List[str] = []
        
        active_components = {
            name: dim for name, dim in self.component_dimensions.items()
            if name != 'total' and dim > 0 and self.components.get(name, False)
        }
        print(f"Active components based on initial dims and self.components: {active_components}")
        sorted_component_names = sorted(active_components.keys())

        if not sorted_component_names:
            raise ValueError("No active components found. Cannot build model with zero branches.")
        print(f"Building AdvancedMultiBranchCNNTransformer. Active branches (sorted): {sorted_component_names}")

        current_channel_offset = 0
        cnn_output_spatial_dims = None # (H', W')
        tomato_mask = None
        
        # Find reflectance component for mask creation
        reflectance_slice = None
        for branch_name in sorted_component_names:
            num_channels_for_branch = self.component_dimensions[branch_name] 
            
            print(f"  Defining branch: {branch_name} ({num_channels_for_branch} channels) at offset {current_channel_offset}")
            
            if num_channels_for_branch == 0:
                print(f"ALERT: num_channels_for_branch is ZERO for {branch_name} before Lambda slice!")

            # FIX for lambda closure issue: capture start and end offsets at definition time
            start_offset = current_channel_offset
            end_offset = current_channel_offset + num_channels_for_branch
            branch_input_slice = layers.Lambda(
                lambda x, s=start_offset, e=end_offset: x[:, :, :, s:e],
                name=f"{branch_name}_slice"
            )(input_layer)
            print(f"    {branch_name} slice tensor shape (symbolic): {branch_input_slice.shape}")

            # Store reflectance slice for mask creation
            if branch_name == 'reflectance':
                reflectance_slice = branch_input_slice

            current_channel_offset += num_channels_for_branch
            
            cnn_branch_output = self._build_cnn_branch(branch_input_slice, branch_name)
            
            _, h_prime, w_prime, d_b = cnn_branch_output.shape
            if cnn_output_spatial_dims is None:
                cnn_output_spatial_dims = (h_prime, w_prime)
            elif cnn_output_spatial_dims != (h_prime, w_prime):
                raise ValueError(f"Mismatch in H'xW' dimensions between CNN branches. Expected {cnn_output_spatial_dims}, got {(h_prime, w_prime)} for branch {branch_name}.")
            
            # Apply token filtering if enabled
            if self.use_token_filtering:
                if reflectance_slice is not None:
                    # Create tomato mask from reflectance component (only once)
                    if tomato_mask is None:
                        tomato_mask = self._create_tomato_pixel_mask(reflectance_slice)
                        print(f"    Created tomato mask from reflectance component")
                    
                    # Apply token filtering to this branch
                    filtered_tokens, pixel_positions = self._filter_tokens_by_mask(
                        cnn_branch_output, tomato_mask, f"_{branch_name}"
                    )
                    
                    # Project filtered tokens to d_model dimension
                    projected_tokens = Dense(self.d_model, name=f"{branch_name}_projection_to_d_model")(filtered_tokens)
                    
                    branch_raw_projected_tokens[branch_name] = projected_tokens
                    branch_pixel_positions[branch_name] = pixel_positions
                    
                    print(f"    {branch_name} tokens after filtering: {filtered_tokens.shape} -> {projected_tokens.shape}")
                else:
                    print(f"    WARNING: Token filtering enabled but no reflectance component found for mask creation")
                    # Fall back to non-filtered approach
                    num_tokens_per_branch = h_prime * w_prime
                    flattened_tokens = Reshape((num_tokens_per_branch, d_b), name=f"{branch_name}_flatten")(cnn_branch_output)
                    projected_tokens = Dense(self.d_model, name=f"{branch_name}_projection_to_d_model")(flattened_tokens)
                    branch_raw_projected_tokens[branch_name] = projected_tokens
                    branch_pixel_positions[branch_name] = None
            else:
                # No token filtering - use original approach
                num_tokens_per_branch = h_prime * w_prime
                flattened_tokens = Reshape((num_tokens_per_branch, d_b), name=f"{branch_name}_flatten")(cnn_branch_output)
                projected_tokens = Dense(self.d_model, name=f"{branch_name}_projection_to_d_model")(flattened_tokens)
                branch_raw_projected_tokens[branch_name] = projected_tokens
                branch_pixel_positions[branch_name] = None
            
            branch_names_ordered.append(branch_name)

        # Create positional embeddings
        if self.use_token_filtering and reflectance_slice is not None:
            # Use position-based embeddings for filtered tokens
            tokens_for_transformer: Dict[str, tf.Tensor] = {}
            attention_mask = None
            
            for branch_name in branch_names_ordered:
                tokens = branch_raw_projected_tokens[branch_name]
                pixel_positions = branch_pixel_positions[branch_name]
                
                if pixel_positions is not None:
                    # Add positional embeddings using actual pixel positions
                    tokens = self._add_positional_embeddings_with_positions(tokens, pixel_positions)
                    
                    # Create attention mask (only need to do this once since all branches have same positions)
                    if attention_mask is None:
                        attention_mask = self._create_attention_mask(pixel_positions)
                        print(f"    Created attention mask for filtered tokens")
                
                tokens_for_transformer[branch_name] = tokens
        else:
            # Use standard spatial positional embeddings
            if self.use_token_filtering:
                num_spatial_tokens = max(cnn_output_spatial_dims[0] * cnn_output_spatial_dims[1], 1)
            else:
                num_spatial_tokens = cnn_output_spatial_dims[0] * cnn_output_spatial_dims[1]
            
            # Shared Positional Embedding Layer (learnable)
            positional_embedding_lookup = Embedding(
                input_dim=num_spatial_tokens,
                output_dim=self.d_model,
                name="spatial_positional_embedding_lookup"
            )
            positions_indices = tf.range(start=0, limit=num_spatial_tokens, delta=1)
            spatial_pos_embed = positional_embedding_lookup(positions_indices) 
            spatial_pos_embed_broadcastable = tf.expand_dims(spatial_pos_embed, axis=0)

            # Shared Branch Identity Embedding Layer (learnable)
            branch_identity_embedding_lookup = Embedding(
                input_dim=len(branch_names_ordered),
                output_dim=self.d_model,
                name="branch_identity_embedding_lookup"
            )

            tokens_for_transformer: Dict[str, tf.Tensor] = {}
            attention_mask = None
            
            for i, branch_name in enumerate(branch_names_ordered):
                tokens = branch_raw_projected_tokens[branch_name]
                
                # Add Spatial Positional Embedding
                tokens = Add(name=f"{branch_name}_add_spatial_pos_embed")([tokens, spatial_pos_embed_broadcastable])
                
                # Add Branch Identity Embedding
                branch_id_tensor = tf.constant(i, dtype=tf.int32)
                branch_embed_vector = branch_identity_embedding_lookup(branch_id_tensor)
                branch_embed_to_add = tf.reshape(branch_embed_vector, (1, 1, self.d_model))
                
                tokens = Add(name=f"{branch_name}_add_branch_embed")([tokens, branch_embed_to_add])
                tokens_for_transformer[branch_name] = tokens
        
        # Apply cross-attention if enabled
        if self.use_cross_attention and self.cross_attention_pairs:
            print(f"  Applying cross-attention for pairs: {self.cross_attention_pairs}")
            for pair_idx, (name_q, name_kv) in enumerate(self.cross_attention_pairs):
                if name_q in tokens_for_transformer and name_kv in tokens_for_transformer:
                    print(f"    Cross-attending: Query={name_q}, KeyValue={name_kv}")
                    # Direction 1: Q from name_q, KV from name_kv
                    updated_q_tokens = self._cross_attention_block(
                        tokens_for_transformer[name_q], tokens_for_transformer[name_kv],
                        name_prefix=f"cross_attn_q_{name_q}_kv_{name_kv}_pair{pair_idx}",
                        attention_mask=attention_mask
                    )
                    tokens_for_transformer[name_q] = updated_q_tokens

                    # Direction 2 (Vice-versa): Q from name_kv, KV from name_q
                    print(f"    Cross-attending: Query={name_kv}, KeyValue={name_q}")
                    updated_kv_tokens = self._cross_attention_block(
                        tokens_for_transformer[name_kv], tokens_for_transformer[name_q],
                        name_prefix=f"cross_attn_q_{name_kv}_kv_{name_q}_pair{pair_idx}",
                        attention_mask=attention_mask
                    )
                    tokens_for_transformer[name_kv] = updated_kv_tokens
                else:
                    print(f"Warning: Branch name in cross_attention_pair ('{name_q}', '{name_kv}') not found in active branches: {list(tokens_for_transformer.keys())}")

        final_branch_token_sequences = [tokens_for_transformer[name] for name in branch_names_ordered]

        # CLS Token (learnable)
        class CLSToken(layers.Layer):
            def __init__(self, d_model, name="cls_token_layer", **kwargs):
                super().__init__(name=name, **kwargs)
                self.d_model = d_model
                self.cls_token_weight = self.add_weight(
                    shape=(1, 1, self.d_model),
                    initializer="random_normal",
                    trainable=True,
                    name="cls_token_value"
                )
            def call(self, inputs_batch_size_tensor):
                return tf.tile(self.cls_token_weight, [inputs_batch_size_tensor, 1, 1])

        # Get batch_size symbolically using a Lambda layer
        batch_size_symbolic_tensor = layers.Lambda(lambda x: tf.shape(x)[0], name="get_batch_size")(input_layer)
        cls_token_layer = CLSToken(self.d_model)
        cls_token_batch = cls_token_layer(batch_size_symbolic_tensor)

        all_tokens_list = [cls_token_batch] + final_branch_token_sequences
        concatenated_tokens = Concatenate(axis=1, name="concatenate_all_tokens")(all_tokens_list)

        # Transformer Encoder Stack
        transformer_input = concatenated_tokens
        for i in range(self.transformer_layers):
            transformer_input = self._transformer_encoder_block(
                transformer_input, 
                name_prefix=f"transformer_encoder_layer_{i+1}",
                attention_mask=attention_mask
            )
        
        # Regression Head using CLS token output
        cls_token_final_representation = layers.Lambda(lambda t: t[:, 0, :], name="extract_cls_token_representation")(transformer_input)
        
        mlp_head_input = cls_token_final_representation
        for i, num_units in enumerate(self.mlp_head_units):
            mlp_head_input = Dense(num_units, activation='relu', name=f"mlp_head_dense_{i+1}")(mlp_head_input)
            if self.mlp_dropout_rate > 0:
                 mlp_head_input = Dropout(self.mlp_dropout_rate, name=f"mlp_head_dropout_{i+1}")(mlp_head_input)
        
        output_prediction = Dense(1, activation='linear', name="final_output_regression")(mlp_head_input)
        
        model = Model(inputs=input_layer, outputs=output_prediction, name=f"AdvMultiBranchCNNTransformer_{'_'.join(self.predicted_attributes)}")
        
        lr = self.optimizer_params.get('learning_rate', 1e-4)
        wd = self.optimizer_params.get('weight_decay', 1e-4)
        optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        return model

    def _build_multi_output_model(self) -> tf.keras.Model:
        """
        Build a multi-output Advanced Multi-Branch CNN+Transformer for all quality attributes.
        """
        print(f"--- Building multi-output model for all attributes ---")
        print(f"Initial self.component_dimensions: {self.component_dimensions}")
        print(f"Initial self.components: {self.components}")
        print(f"Token filtering enabled: {self.use_token_filtering}")

        H, W, C_total = self.model_shape
        input_layer = Input(shape=(H, W, C_total), name="input_combined_special_images")
        
        branch_raw_projected_tokens: Dict[str, tf.Tensor] = {}
        branch_pixel_positions: Dict[str, tf.Tensor] = {}
        branch_names_ordered: List[str] = []
        
        active_components = {
            name: dim for name, dim in self.component_dimensions.items()
            if name != 'total' and dim > 0 and self.components.get(name, False)
        }
        print(f"Active components based on initial dims and self.components: {active_components}")
        sorted_component_names = sorted(active_components.keys())

        if not sorted_component_names:
            raise ValueError("No active components found. Cannot build model with zero branches.")
        print(f"Building AdvancedMultiBranchCNNTransformer Multi-Output. Active branches (sorted): {sorted_component_names}")

        current_channel_offset = 0
        cnn_output_spatial_dims = None # (H', W')
        tomato_mask = None
        
        # Find reflectance component for mask creation
        reflectance_slice = None
        for branch_name in sorted_component_names:
            num_channels_for_branch = self.component_dimensions[branch_name] 
            
            print(f"  Defining branch: {branch_name} ({num_channels_for_branch} channels) at offset {current_channel_offset}")
            
            if num_channels_for_branch == 0:
                print(f"ALERT: num_channels_for_branch is ZERO for {branch_name} before Lambda slice!")

            start_offset = current_channel_offset
            end_offset = current_channel_offset + num_channels_for_branch
            branch_input_slice = layers.Lambda(
                lambda x, s=start_offset, e=end_offset: x[:, :, :, s:e],
                name=f"{branch_name}_slice"
            )(input_layer)
            print(f"    {branch_name} slice tensor shape (symbolic): {branch_input_slice.shape}")

            # Store reflectance slice for mask creation
            if branch_name == 'reflectance':
                reflectance_slice = branch_input_slice

            # CNN Branch Processing
            cnn_feature_map = self._build_cnn_branch(branch_input_slice, branch_name)
            print(f"    {branch_name} CNN output shape (symbolic): {cnn_feature_map.shape}")
            
            if cnn_output_spatial_dims is None:
                cnn_output_spatial_dims = cnn_feature_map.shape[1:3] # (H', W')
                print(f"    Set cnn_output_spatial_dims to: {cnn_output_spatial_dims}")

            # Apply token filtering if enabled
            if self.use_token_filtering:
                if reflectance_slice is not None:
                    # Create tomato mask from reflectance component (only once)
                    if tomato_mask is None:
                        tomato_mask = self._create_tomato_pixel_mask(reflectance_slice)
                        print(f"    Created tomato mask from reflectance component")
                    
                    # Apply token filtering to this branch
                    filtered_tokens, pixel_positions = self._filter_tokens_by_mask(
                        cnn_feature_map, tomato_mask, f"_{branch_name}"
                    )
                    
                    # Project filtered tokens to d_model dimension
                    projected_tokens = Dense(self.d_model, name=f"{branch_name}_project_to_d_model")(filtered_tokens)
                    print(f"    {branch_name} tokens after filtering: {filtered_tokens.shape} -> {projected_tokens.shape}")
                    
                    branch_raw_projected_tokens[branch_name] = projected_tokens
                    branch_pixel_positions[branch_name] = pixel_positions
                else:
                    print(f"    WARNING: Token filtering enabled but no reflectance component found for mask creation")
                    # Fall back to non-filtered approach
                    H_prime, W_prime = cnn_output_spatial_dims
                    N_prime = H_prime * W_prime
                    d_b = cnn_feature_map.shape[-1]
                    
                    tokens = Reshape((N_prime, d_b), name=f"{branch_name}_reshape_to_tokens")(cnn_feature_map)
                    projected_tokens = Dense(self.d_model, name=f"{branch_name}_project_to_d_model")(tokens)
                    
                    branch_raw_projected_tokens[branch_name] = projected_tokens
                    branch_pixel_positions[branch_name] = None
            else:
                # No token filtering - use original approach
                H_prime, W_prime = cnn_output_spatial_dims
                N_prime = H_prime * W_prime
                d_b = cnn_feature_map.shape[-1]
                
                tokens = Reshape((N_prime, d_b), name=f"{branch_name}_reshape_to_tokens")(cnn_feature_map)
                projected_tokens = Dense(self.d_model, name=f"{branch_name}_project_to_d_model")(tokens)
                
                branch_raw_projected_tokens[branch_name] = projected_tokens
                branch_pixel_positions[branch_name] = None

            branch_names_ordered.append(branch_name)
            current_channel_offset += num_channels_for_branch

        # Create positional embeddings
        if self.use_token_filtering and reflectance_slice is not None:
            # Use position-based embeddings for filtered tokens
            tokens_for_transformer: Dict[str, tf.Tensor] = {}
            attention_mask = None
            
            for branch_name in branch_names_ordered:
                tokens = branch_raw_projected_tokens[branch_name]
                pixel_positions = branch_pixel_positions[branch_name]
                
                if pixel_positions is not None:
                    # Add positional embeddings using actual pixel positions
                    tokens = self._add_positional_embeddings_with_positions(tokens, pixel_positions)
                    
                    # Create attention mask (only need to do this once since all branches have same positions)
                    if attention_mask is None:
                        attention_mask = self._create_attention_mask(pixel_positions)
                        print(f"    Created attention mask for filtered tokens")
                
                tokens_for_transformer[branch_name] = tokens
        else:
            # Use standard spatial positional embeddings
            if self.use_token_filtering:
                num_spatial_tokens = max(cnn_output_spatial_dims[0] * cnn_output_spatial_dims[1], 1)
            else:
                num_spatial_tokens = cnn_output_spatial_dims[0] * cnn_output_spatial_dims[1]
            
            # Shared Positional Embedding Layer (learnable)
            positional_embedding_lookup = Embedding(
                input_dim=num_spatial_tokens,
                output_dim=self.d_model,
                name="spatial_positional_embedding_lookup"
            )
            positions_indices = tf.range(start=0, limit=num_spatial_tokens, delta=1)
            spatial_pos_embed = positional_embedding_lookup(positions_indices) 
            spatial_pos_embed_broadcastable = tf.expand_dims(spatial_pos_embed, axis=0)

            # Shared Branch Identity Embedding Layer (learnable)
            branch_identity_embedding_lookup = Embedding(
                input_dim=len(branch_names_ordered),
                output_dim=self.d_model,
                name="branch_identity_embedding_lookup"
            )

            tokens_for_transformer: Dict[str, tf.Tensor] = {}
            attention_mask = None
            
            for i, branch_name in enumerate(branch_names_ordered):
                tokens = branch_raw_projected_tokens[branch_name]
                
                # Add Spatial Positional Embedding
                tokens = Add(name=f"{branch_name}_add_spatial_pos_embed")([tokens, spatial_pos_embed_broadcastable])
                
                # Add Branch Identity Embedding
                branch_id_tensor = tf.constant(i, dtype=tf.int32)
                branch_embed_vector = branch_identity_embedding_lookup(branch_id_tensor)
                branch_embed_to_add = tf.reshape(branch_embed_vector, (1, 1, self.d_model))
                
                tokens = Add(name=f"{branch_name}_add_branch_embed")([tokens, branch_embed_to_add])
                tokens_for_transformer[branch_name] = tokens

        # Cross-Attention (if enabled)
        final_branch_token_sequences = []
        if self.use_cross_attention and self.cross_attention_pairs:
            print(f"  Applying cross-attention with pairs: {self.cross_attention_pairs}")
            branch_tokens_after_cross_attn = dict(tokens_for_transformer)
            
            for query_branch, kv_branch in self.cross_attention_pairs:
                if query_branch in branch_tokens_after_cross_attn and kv_branch in branch_tokens_after_cross_attn:
                    updated_query_tokens = self._cross_attention_block(
                        query_tokens=branch_tokens_after_cross_attn[query_branch],
                        key_value_tokens=branch_tokens_after_cross_attn[kv_branch],
                        name_prefix=f"cross_attn_{query_branch}_to_{kv_branch}",
                        attention_mask=attention_mask
                    )
                    branch_tokens_after_cross_attn[query_branch] = updated_query_tokens
                    print(f"    Applied cross-attention: {query_branch} <- {kv_branch}")
                else:
                    print(f"    WARNING: Cross-attention pair ({query_branch}, {kv_branch}) contains unknown branch names.")
            
            for branch_name in branch_names_ordered:
                final_branch_token_sequences.append(branch_tokens_after_cross_attn[branch_name])
        else:
            for branch_name in branch_names_ordered:
                final_branch_token_sequences.append(tokens_for_transformer[branch_name])

        # CLS Token
        class CLSToken(layers.Layer):
            def __init__(self, d_model, name="cls_token_layer", **kwargs):
                super().__init__(name=name, **kwargs)
                self.d_model = d_model

            def build(self, input_shape):
                self.cls_embedding = self.add_weight(
                    name="cls_embedding", shape=(1, 1, self.d_model), initializer="random_normal", trainable=True
                )
                super().build(input_shape)

            def call(self, input_tensor):
                # Extract batch size from the input tensor within the layer
                batch_size = tf.shape(input_tensor)[0]
                return tf.tile(self.cls_embedding, [batch_size, 1, 1])

        cls_token_layer = CLSToken(d_model=self.d_model, name="cls_token")
        cls_token_batch = cls_token_layer(input_layer)

        all_tokens_list = [cls_token_batch] + final_branch_token_sequences
        concatenated_tokens = Concatenate(axis=1, name="concatenate_all_tokens")(all_tokens_list)

        # Transformer Encoder Stack
        transformer_input = concatenated_tokens
        for i in range(self.transformer_layers):
            transformer_input = self._transformer_encoder_block(
                transformer_input, 
                name_prefix=f"transformer_encoder_layer_{i+1}",
                attention_mask=attention_mask
            )
        
        # Regression Head using CLS token output
        cls_token_final_representation = layers.Lambda(lambda t: t[:, 0, :], name="extract_cls_token_representation")(transformer_input)
        
        mlp_head_input = cls_token_final_representation
        for i, num_units in enumerate(self.mlp_head_units):
            mlp_head_input = Dense(num_units, activation='relu', name=f"mlp_head_dense_{i+1}")(mlp_head_input)
            if self.mlp_dropout_rate > 0:
                 mlp_head_input = Dropout(self.mlp_dropout_rate, name=f"mlp_head_dropout_{i+1}")(mlp_head_input)
        
        # Multi-output layer - one output for all attributes
        output_prediction = Dense(len(self.predicted_attributes), activation='linear', name="multi_output")(mlp_head_input)
        
        model = Model(inputs=input_layer, outputs=output_prediction, name=f"AdvMultiBranchCNNTransformerMultiRegression")
        
        lr = self.optimizer_params.get('learning_rate', 1e-4)
        wd = self.optimizer_params.get('weight_decay', 1e-4)
        optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mse'])
        
        print(f"[AdvancedMultiBranchCNNTransformer] Built multi-output model with {len(self.predicted_attributes)} outputs")
        return model

    # Removed hardcoded get_architecture_config - now using dynamic extraction
