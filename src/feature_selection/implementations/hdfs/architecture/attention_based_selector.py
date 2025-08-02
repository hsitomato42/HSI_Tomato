# feature_selection/attention_based_selector.py

import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import src.config as config

from .band_attention import MultiHeadBandAttention
from .gumbel_gates import GumbelGates
from src.utils.spectral_indexes import SpectralIndexCalculator
from src.utils.data_processing import DataProcessingUtils


class AttentionBasedFeatureSelector(layers.Layer):
    """
    Main Attention-Based Feature Selection module that combines:
    1. Multi-head spectral band attention for importance evaluation
    2. Gumbel-Softmax gates for differentiable discrete selection
    3. Intelligent feature map generation (reflectance, std, NDSI, indexes)
    
    This is the central component that implements the novel feature selection approach
    described in the research article.
    """
    
    def __init__(
        self,
        # Band selection parameters
        num_bands: int = 204,
        k_bands: int = 6,
        a_ndsi: int = 2,
        b_std: int = 4,
        c_indexes: int = 3,
        
        # Multi-head attention parameters
        attention_d_model: int = 128,
        attention_heads: int = 4,
        attention_layers: int = 2,
        
        # Gumbel-Softmax parameters
        gumbel_temperature: float = 1.0,
        gumbel_min_temp: float = 0.1,
        gumbel_decay: float = 0.99,
        
        # Component generation parameters
        use_spatial_features: bool = True,
        spatial_kernel_size: int = 9,
        index_strategy: str = "competitive",  # "existing", "learned", "hybrid", "competitive"
        
        # Loss weights
        diversity_weight: float = 0.1,
        sparsity_weight: float = 1.0,
        
        # Random seed for reproducibility
        random_seed: Optional[int] = None,
        
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # Store parameters
        self.num_bands = num_bands
        self.k_bands = k_bands
        self.a_ndsi = a_ndsi
        self.b_std = b_std
        self.c_indexes = c_indexes
        
        self.use_spatial_features = use_spatial_features
        self.spatial_kernel_size = spatial_kernel_size
        self.index_strategy = index_strategy
        
        self.diversity_weight = diversity_weight
        self.sparsity_weight = sparsity_weight
        self.random_seed = random_seed or getattr(config, 'RANDOM_STATE', 42)
        
        # Set random seed for reproducible initialization
        if self.random_seed is not None:
            tf.random.set_seed(self.random_seed)
        
        # Validate parameters
        if b_std > k_bands:
            raise ValueError(f"b_std ({b_std}) cannot be greater than k_bands ({k_bands})")
        
        # Initialize sub-modules with proper seeding
        self.band_attention = MultiHeadBandAttention(
            num_bands=num_bands,
            d_model=attention_d_model,
            num_heads=attention_heads,
            num_layers=attention_layers,
            use_spatial_features=use_spatial_features,
            name="band_attention"
        )
        
        self.gumbel_gates = GumbelGates(
            num_bands=num_bands,
            k_bands=k_bands,
            initial_temperature=gumbel_temperature,
            min_temperature=gumbel_min_temp,
            temperature_decay=gumbel_decay,
            random_seed=self.random_seed,
            name="gumbel_gates"
        )
        
        # NDSI pair selection network
        self.ndsi_pair_selector = self._build_ndsi_pair_selector(attention_d_model)
        
        # Index selection networks - unified approach for all strategies
        self.index_selectors = self._build_index_selector_networks(attention_d_model)
        
        # Learned index networks (neural networks for learned indexes)
        self.learned_index_networks = self._build_learned_index_networks()
        
        # Band-to-std assignment network (decides which bands get std maps)
        self.std_assignment_network = self._build_std_assignment_network(attention_d_model)
    
    def _build_ndsi_pair_selector(self, d_model: int) -> keras.Model:
        """
        Build a network that selects optimal band pairs for NDSI computation.
        """
        # Input: band embeddings from attention module
        band_embeddings = layers.Input(shape=(None, d_model), name="band_embeddings_ndsi")
        
        # Compute pairwise relationships
        # For each band, compute its importance for NDSI pairing
        x = layers.Dense(d_model // 2, activation='relu', name="pair_projection")(band_embeddings)
        
        # Project to pair importance scores per band (not globally pooled)
        pair_logits = layers.Dense(1, name="pair_importance")(x)  # (batch, k_bands, 1)
        
        model = keras.Model(inputs=band_embeddings, outputs=pair_logits, name="ndsi_pair_selector")
        return model
    
    def _build_learned_index_networks(self) -> List[keras.Model]:
        """
        Build networks that learn optimal spectral index combinations.
        """
        networks = []
        
        for i in range(self.c_indexes):
            # Each network learns a different spectral index
            band_input = layers.Input(shape=(self.k_bands,), name=f"selected_bands_{i}")
            
            # Non-linear combination of bands
            x = layers.Dense(16, activation='relu', name=f"index_hidden1_{i}")(band_input)
            x = layers.Dense(8, activation='relu', name=f"index_hidden2_{i}")(x)
            index_output = layers.Dense(1, activation='tanh', name=f"index_output_{i}")(x)
            
            model = keras.Model(inputs=band_input, outputs=index_output, name=f"learned_index_{i}")
            networks.append(model)
        
        return networks
    
    def _build_index_selector_networks(self, attention_d_model: int) -> Dict[str, keras.Model]:
        """
        Build selector networks for different index strategies
        Similar to band selector but for index selection
        """
        selectors = {}
        
        # Learned index selector
        selectors['learned'] = self._build_single_index_selector(
            attention_d_model, name="learned_index_selector"
        )
        
        # Predefined index selector  
        selectors['predefined'] = self._build_single_index_selector(
            attention_d_model, name="predefined_index_selector"
        )
        
        # Combined selector for competitive strategy
        selectors['combined'] = self._build_single_index_selector(
            attention_d_model, name="combined_index_selector"
        )
        
        return selectors

    def _build_single_index_selector(self, d_model: int, name: str) -> keras.Model:
        """
        Build a single index selector network
        """
        # Input: features from all available indexes
        index_features = layers.Input(shape=(None, d_model), name=f"{name}_features")
        
        # Multi-head attention for index relationships
        attention_output = layers.MultiHeadAttention(
            num_heads=8,
            key_dim=d_model // 8,
            name=f"{name}_attention"
        )(index_features, index_features)
        
        # Add & Norm
        attention_output = layers.Add()([index_features, attention_output])
        attention_output = layers.LayerNormalization()(attention_output)
        
        # Feed-forward network
        ffn_output = layers.Dense(d_model * 4, activation='relu')(attention_output)
        ffn_output = layers.Dense(d_model)(ffn_output)
        
        # Add & Norm
        ffn_output = layers.Add()([attention_output, ffn_output])
        ffn_output = layers.LayerNormalization()(ffn_output)
        
        # Selection probabilities
        selection_probs = layers.Dense(1, activation='sigmoid', name=f"{name}_probs")(ffn_output)
        selection_probs = layers.Reshape((-1,))(selection_probs)  # (batch_size, num_indexes)
        
        return keras.Model(inputs=index_features, outputs=selection_probs, name=name)
    
    def _generate_all_learned_indexes(self, selected_bands: tf.Tensor) -> List[tf.Tensor]:
        """
        Generate all possible learned indexes using multiple strategies:
        1. Neural networks (16→8→1 neurons) - fully flexible
        2. 4 Fixed Mathematical Forms (competitive style)
        3. 2 Simple Mathematical Forms (hybrid style)
        4. 1 Fixed Mathematical Form (progressive style)
        
        Returns: List of all possible learned index tensors
        """
        all_learned_indexes = []
        
        # 1. Neural Network Learned Indexes (most flexible)
        if self.learned_index_networks is not None:
            for network in self.learned_index_networks:
                mean_reflectance = tf.reduce_mean(selected_bands, axis=[1, 2])
                learned_index = network(mean_reflectance)
                # Broadcast to spatial dimensions
                index_map = self._broadcast_to_spatial(learned_index, selected_bands)
                all_learned_indexes.append(index_map)
        
        # 2. 4 Fixed Mathematical Forms (competitive style)
        for i in range(12):  # Generate multiple combinations
            if self.k_bands >= 3:
                band_a = selected_bands[:, :, :, i % self.k_bands]
                band_b = selected_bands[:, :, :, (i + 1) % self.k_bands]
                band_c = selected_bands[:, :, :, (i + 2) % self.k_bands]
                
                combination_type = i % 4
                if combination_type == 0:
                    # Normalized difference with third band
                    index_map = (band_a - band_b) / (band_a + band_b + band_c + 1e-8)
                elif combination_type == 1:
                    # Ratio of ratios
                    index_map = (band_a / (band_b + 1e-8)) / (band_c + 1e-8)
                elif combination_type == 2:
                    # Weighted combination
                    index_map = (2.0 * band_a - band_b - band_c) / (band_a + band_b + band_c + 1e-8)
                else:
                    # Enhanced vegetation index style
                    index_map = (band_a - band_b) / (band_a + 6.0 * band_b - 7.5 * band_c + 1.0)
                
                all_learned_indexes.append(index_map)
        
        # 3. 2 Simple Mathematical Forms (hybrid style)
        for i in range(6):  # Generate multiple combinations
            if self.k_bands >= 2:
                band_a = selected_bands[:, :, :, i % self.k_bands]
                band_b = selected_bands[:, :, :, (i + 1) % self.k_bands]
                
                if i % 2 == 0:
                    # Normalized difference
                    index_map = (band_a - band_b) / (band_a + band_b + 1e-8)
                else:
                    # Simple ratio
                    index_map = band_a / (band_b + 1e-8)
                
                all_learned_indexes.append(index_map)
        
        # 4. 1 Fixed Mathematical Form (progressive style)
        for i in range(self.c_indexes):
            if self.k_bands >= 3:
                a = selected_bands[:, :, :, i % self.k_bands]
                b = selected_bands[:, :, :, (i + 1) % self.k_bands]
                c = selected_bands[:, :, :, (i + 2) % self.k_bands]
                
                # Squared difference form
                learned_index = tf.square(a - b) / (tf.square(a) + c + 1e-8)
                all_learned_indexes.append(learned_index)
        
        return all_learned_indexes

    def _generate_all_predefined_indexes(self, selected_bands: tf.Tensor) -> List[tf.Tensor]:
        """
        Generate all predefined indexes from SpectralIndex enum
        
        MEMORY FIX: Removed full_spectrum parameter to eliminate memory waste.
        
        Returns: List of all predefined index tensors
        """
        all_predefined_indexes = []
        
        # Get predefined indexes from config
        predefined_indexes = getattr(config, 'INDEXES', [])
        
        if predefined_indexes:
            for i, index_type in enumerate(predefined_indexes):
                # Use existing logic to compute predefined indexes
                if self.k_bands >= 2:
                    band_a = selected_bands[:, :, :, i % self.k_bands]
                    band_b = selected_bands[:, :, :, (i + 1) % self.k_bands]
                    
                    if 'NDVI' in str(index_type) or 'GRVI' in str(index_type):
                        # NDVI-like: (NIR - Red) / (NIR + Red)
                        index_map = (band_b - band_a) / (band_b + band_a + 1e-8)
                    elif 'ZMI' in str(index_type):
                        # ZMI-like: Red / Green ratio
                        index_map = band_a / (band_b + 1e-8)
                    else:
                        # Generic normalized difference
                        index_map = (band_a - band_b) / (band_a + band_b + 1e-8)
                else:
                    index_map = selected_bands[:, :, :, 0]
                
                all_predefined_indexes.append(index_map)
        
        return all_predefined_indexes

    def _broadcast_to_spatial(self, learned_index: tf.Tensor, reference_tensor: tf.Tensor) -> tf.Tensor:
        """Helper to broadcast learned index to spatial dimensions"""
        batch_size = tf.shape(reference_tensor)[0]
        height = tf.shape(reference_tensor)[1]
        width = tf.shape(reference_tensor)[2]
        
        index_map = tf.tile(
            tf.reshape(learned_index, (batch_size, 1, 1, 1)),
            (1, height, width, 1)
        )
        return tf.squeeze(index_map, -1)  # (batch, height, width)

    def _extract_index_features(self, index_maps: List[tf.Tensor], d_model: int) -> tf.Tensor:
        """
        Extract features from index maps for selector networks
        Similar to how band features are extracted
        """
        if not index_maps:
            # Return empty tensor if no index maps
            batch_size = 1
            return tf.zeros((batch_size, 0, d_model))
        
        batch_size = tf.shape(index_maps[0])[0]
        
        # Compute statistical features for each index
        features = []
        for i, index_map in enumerate(index_maps):
            # Statistical features
            mean_val = tf.reduce_mean(index_map, axis=[1, 2])  # (batch,)
            std_val = tf.math.reduce_std(index_map, axis=[1, 2])  # (batch,)
            min_val = tf.reduce_min(index_map, axis=[1, 2])  # (batch,)
            max_val = tf.reduce_max(index_map, axis=[1, 2])  # (batch,)
            
            # Combine features
            index_features = tf.stack([mean_val, std_val, min_val, max_val], axis=-1)  # (batch, 4)
            
            # Project to d_model dimensions
            index_features = layers.Dense(d_model, name=f"index_feature_proj_{i}")(index_features)
            features.append(index_features)
        
        # Stack features: (batch, num_indexes, d_model)
        return tf.stack(features, axis=1)

    def _select_indexes_with_exploration_exploitation(
        self, 
        index_maps: List[tf.Tensor], 
        selection_probs: tf.Tensor, 
        k: int, 
        training: bool
    ) -> List[tf.Tensor]:
        """
        Select top-k indexes using exploration-exploitation strategy
        Uses existing temperature configuration from config.py
        """
        if len(index_maps) == 0:
            return [tf.zeros_like(index_maps[0]) for _ in range(k)]
        
        batch_size = tf.shape(selection_probs)[0]
        num_indexes = len(index_maps)
        
        if training:
            # EXPLORATION: Use existing Gumbel temperature configuration
            temperature = config.FEATURE_SELECTION_GUMBEL_TEMPERATURE  # Uses existing 2.0
            
            # Add Gumbel noise
            gumbel_noise = -tf.math.log(-tf.math.log(tf.random.uniform(tf.shape(selection_probs))))
            logits = tf.math.log(selection_probs + 1e-8) + gumbel_noise
            
            # Soft selection with existing temperature
            soft_selection = tf.nn.softmax(logits / temperature, axis=-1)
            
            # Select top-k using soft selection
            _, top_indices = tf.nn.top_k(soft_selection, k=min(k, num_indexes))
            
        else:
            # EXPLOITATION: Use deterministic top-k selection
            _, top_indices = tf.nn.top_k(selection_probs, k=min(k, num_indexes))
        
        # Gather selected indexes
        selected = []
        for i in range(k):
            if i < tf.shape(top_indices)[1]:
                # Use the index from the first batch (assuming consistent selection across batches)
                idx = top_indices[0, i]
                selected_index = index_maps[idx]
                selected.append(selected_index)
            else:
                # Pad with zeros if not enough indexes
                selected.append(tf.zeros_like(index_maps[0]))
        
        return selected
    
    def _build_std_assignment_network(self, d_model: int) -> keras.Model:
        """
        Build a network that decides which selected bands should generate std maps.
        """
        band_embeddings = layers.Input(shape=(None, d_model), name="band_embeddings_std")
        
        # Project to assignment scores
        x = layers.Dense(d_model // 2, activation='relu', name="std_projection")(band_embeddings)
        assignment_logits = layers.Dense(1, activation='sigmoid', name="std_assignment")(x)
        
        model = keras.Model(inputs=band_embeddings, outputs=assignment_logits, name="std_assignment_network")
        return model
    
    def _compute_local_std_maps(self, selected_cube: tf.Tensor) -> tf.Tensor:
        """
        Compute local standard deviation maps for texture analysis.
        
        Args:
            selected_cube: (batch_size, height, width, k_bands)
            
        Returns:
            std_maps: (batch_size, height, width, k_bands)
        """
        batch_size = tf.shape(selected_cube)[0]
        height = tf.shape(selected_cube)[1]
        width = tf.shape(selected_cube)[2]
        
        # Kernel for local std computation
        kernel_size = self.spatial_kernel_size
        kernel = tf.ones((kernel_size, kernel_size, 1, 1)) / (kernel_size * kernel_size)
        
        std_maps_list = []
        
        for band_idx in range(self.k_bands):
            band_image = selected_cube[:, :, :, band_idx:band_idx+1]  # (batch, height, width, 1)
            
            # Compute local mean
            local_mean = tf.nn.conv2d(band_image, kernel, strides=1, padding='SAME')
            
            # Compute local variance
            squared_diff = tf.square(band_image - local_mean)
            local_var = tf.nn.conv2d(squared_diff, kernel, strides=1, padding='SAME')
            local_std = tf.sqrt(local_var + 1e-8)
            
            std_maps_list.append(tf.squeeze(local_std, -1))  # Remove channel dim
        
        # Stack std maps: (batch_size, height, width, k_bands)
        std_maps = tf.stack(std_maps_list, axis=-1)
        
        return std_maps
    
    def _select_optimal_ndsi_pairs(
        self, 
        selected_indices: tf.Tensor, 
        band_embeddings: tf.Tensor
    ) -> List[Tuple[int, int]]:
        """
        Select optimal band pairs for NDSI computation using the learned selector.
        
        Args:
            selected_indices: (batch_size, k_bands) - indices of selected bands
            band_embeddings: (batch_size, k_bands, d_model) - embeddings of selected bands
            
        Returns:
            optimal_pairs: List of (band_i, band_j) tuples
        """
        # Get pair importance scores for selected bands
        try:
            pair_scores = self.ndsi_pair_selector(band_embeddings)  # (batch_size, k_bands, 1)
            pair_scores = tf.squeeze(pair_scores, -1)  # (batch_size, k_bands)
            
            # Average across batch to get global band importance
            avg_pair_scores = tf.reduce_mean(pair_scores, axis=0)  # (k_bands,)
            
            # Ensure we have the right shape
            if len(avg_pair_scores.shape) == 0:  # scalar
                # Fallback to uniform scores
                avg_pair_scores = tf.ones(self.k_bands, dtype=tf.float32)
            
            # Convert to numpy for easier iteration
            avg_pair_scores_np = avg_pair_scores.numpy()
            
            # Ensure it's at least 1D
            if avg_pair_scores_np.ndim == 0:
                avg_pair_scores_np = np.ones(self.k_bands, dtype=np.float32)
            
        except Exception as e:
            # Fallback to uniform importance scores
            avg_pair_scores_np = np.ones(self.k_bands, dtype=np.float32)
        
        # Generate all possible pairs and their combined scores
        pair_combinations = []
        pair_scores_list = []
        
        for i in range(self.k_bands):
            for j in range(i + 1, self.k_bands):
                # Score for this pair is sum of individual band scores
                # Plus a diversity bonus (difference in wavelength positions)
                try:
                    combined_score = avg_pair_scores_np[i] + avg_pair_scores_np[j]
                except (IndexError, TypeError):
                    # Fallback to just diversity bonus
                    combined_score = 0.0
                
                # Add diversity bonus (encourage distant wavelengths)
                wavelength_distance = abs(i - j) / self.k_bands  # Normalized distance
                diversity_bonus = wavelength_distance * 0.1
                
                final_score = combined_score + diversity_bonus
                
                pair_combinations.append((i, j))
                pair_scores_list.append(float(final_score))
        
        # Select top a_ndsi pairs
        if len(pair_scores_list) > 0:
            pair_scores_tensor = tf.constant(pair_scores_list)
            _, top_pair_indices = tf.nn.top_k(pair_scores_tensor, k=min(self.a_ndsi, len(pair_combinations)))
            
            optimal_pairs = [pair_combinations[idx.numpy()] for idx in top_pair_indices]
        else:
            # Fallback: use first available pairs
            optimal_pairs = []
            for i in range(min(self.a_ndsi, self.k_bands - 1)):
                j = i + 1
                if j < self.k_bands:
                    optimal_pairs.append((i, j))
        
        return optimal_pairs
    
    def _compute_ndsi_maps(
        self, 
        selected_cube: tf.Tensor, 
        selected_indices: tf.Tensor,
        band_embeddings: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute NDSI maps from selected optimal band pairs.
        
        Args:
            selected_cube: (batch_size, height, width, k_bands)
            selected_indices: (batch_size, k_bands)
            band_embeddings: (batch_size, k_bands, d_model)
            
        Returns:
            ndsi_maps: (batch_size, height, width, a_ndsi)
        """
        # Select optimal pairs
        optimal_pairs = self._select_optimal_ndsi_pairs(selected_indices, band_embeddings)
        
        ndsi_maps_list = []
        
        for pair_idx, (i, j) in enumerate(optimal_pairs):
            if pair_idx >= self.a_ndsi:
                break
                
            band_i = selected_cube[:, :, :, i]  # (batch, height, width)
            band_j = selected_cube[:, :, :, j]  # (batch, height, width)
            
            # Compute NDSI: (band_i - band_j) / (band_i + band_j)
            ndsi = (band_i - band_j) / (band_i + band_j + 1e-8)
            ndsi_maps_list.append(ndsi)
        
        # Pad with zeros if we have fewer pairs than requested
        while len(ndsi_maps_list) < self.a_ndsi:
            zero_map = tf.zeros_like(selected_cube[:, :, :, 0])
            ndsi_maps_list.append(zero_map)
        
        # Stack NDSI maps: (batch_size, height, width, a_ndsi)
        ndsi_maps = tf.stack(ndsi_maps_list, axis=-1)
        
        return ndsi_maps
    
    def _compute_spectral_indexes(
        self, 
        selected_cube: tf.Tensor, 
        selected_indices: tf.Tensor,
        training: bool = True
    ) -> tf.Tensor:
        """
        Unified index computation with trainable exploration-exploitation selection
        
        Args:
            selected_cube: (batch_size, height, width, k_bands)
            selected_indices: (batch_size, k_bands) - indices of selected bands from full spectrum
            training: Whether in training mode
            
        Returns:
            index_maps: (batch_size, height, width, c_indexes)
        """
        # Generate ALL possible indexes
        all_learned_indexes = self._generate_all_learned_indexes(selected_cube)
        all_predefined_indexes = self._generate_all_predefined_indexes(selected_cube)
        
        # Extract features for selector networks
        d_model = config.FEATURE_SELECTION_D_MODEL  # Use existing 128
        
        # Strategy-specific selection using exploration-exploitation
        if self.index_strategy == "learned":
            # Select from learned pool only
            if len(all_learned_indexes) > 0:
                learned_features = self._extract_index_features(all_learned_indexes, d_model)
                selection_probs = self.index_selectors['learned'](learned_features, training=training)
                selected_indexes = self._select_indexes_with_exploration_exploitation(
                    all_learned_indexes, selection_probs, self.c_indexes, training
                )
            else:
                selected_indexes = [tf.zeros_like(selected_cube[:, :, :, 0]) for _ in range(self.c_indexes)]
        
        elif self.index_strategy == "competitive":
            # Select from combined pool (learned + predefined)
            combined_indexes = all_learned_indexes + all_predefined_indexes
            if len(combined_indexes) > 0:
                combined_features = self._extract_index_features(combined_indexes, d_model)
                selection_probs = self.index_selectors['combined'](combined_features, training=training)
                selected_indexes = self._select_indexes_with_exploration_exploitation(
                    combined_indexes, selection_probs, self.c_indexes, training
                )
            else:
                selected_indexes = [tf.zeros_like(selected_cube[:, :, :, 0]) for _ in range(self.c_indexes)]
        
        elif self.index_strategy == "hybrid":
            # Select half from each pool
            num_learned = self.c_indexes // 2
            num_predefined = self.c_indexes - num_learned
            
            selected_indexes = []
            
            # Select from learned pool
            if len(all_learned_indexes) > 0 and num_learned > 0:
                learned_features = self._extract_index_features(all_learned_indexes, d_model)
                learned_probs = self.index_selectors['learned'](learned_features, training=training)
                learned_selected = self._select_indexes_with_exploration_exploitation(
                    all_learned_indexes, learned_probs, num_learned, training
                )
                selected_indexes.extend(learned_selected)
            
            # Select from predefined pool
            if len(all_predefined_indexes) > 0 and num_predefined > 0:
                predefined_features = self._extract_index_features(all_predefined_indexes, d_model)
                predefined_probs = self.index_selectors['predefined'](predefined_features, training=training)
                predefined_selected = self._select_indexes_with_exploration_exploitation(
                    all_predefined_indexes, predefined_probs, num_predefined, training
                )
                selected_indexes.extend(predefined_selected)
            
            # Pad if needed
            while len(selected_indexes) < self.c_indexes:
                selected_indexes.append(tf.zeros_like(selected_cube[:, :, :, 0]))
        
        elif self.index_strategy == "existing":
            # Use only predefined indexes
            if len(all_predefined_indexes) > 0:
                predefined_features = self._extract_index_features(all_predefined_indexes, d_model)
                selection_probs = self.index_selectors['predefined'](predefined_features, training=training)
                selected_indexes = self._select_indexes_with_exploration_exploitation(
                    all_predefined_indexes, selection_probs, self.c_indexes, training
                )
            else:
                selected_indexes = [tf.zeros_like(selected_cube[:, :, :, 0]) for _ in range(self.c_indexes)]
        
        # Stack and return
        return tf.stack(selected_indexes[:self.c_indexes], axis=-1)
    
    def call(
        self, 
        hyperspectral_cube: tf.Tensor, 
        training: Optional[bool] = None,
        return_attention_analysis: bool = False,
        epoch: Optional[int] = None
    ) -> Dict[str, tf.Tensor]:
        """
        Forward pass of the attention-based feature selector.
        
        Args:
            hyperspectral_cube: (batch_size, height, width, num_bands)
            training: training mode flag
            return_attention_analysis: whether to return detailed attention analysis
            epoch: current epoch for progressive training
            
        Returns:
            Dict containing:
            - selected_reflectance: (batch_size, height, width, k_bands)
            - std_maps: (batch_size, height, width, b_std) 
            - ndsi_maps: (batch_size, height, width, a_ndsi)
            - index_maps: (batch_size, height, width, c_indexes)
            - selected_band_indices: (batch_size, k_bands)
            - total_loss: scalar - combined loss for training
            - attention_analysis: Dict - detailed attention information (if requested)
        """
        batch_size = tf.shape(hyperspectral_cube)[0]
        
        # Step 1: Multi-head band attention analysis
        attention_output = self.band_attention(hyperspectral_cube, training=training)
        
        band_importance = attention_output['band_importance']
        attention_weights = attention_output['attention_weights']
        band_embeddings = attention_output['band_embeddings']
        diversity_scores = attention_output['diversity_scores']
        
        # Step 2: Gumbel-Softmax band selection with progressive training
        selection_output = self.gumbel_gates(
            hyperspectral_cube, 
            importance_scores=band_importance,
            training=training,
            epoch=epoch
        )
        
        selected_cube = selection_output['selected_cube']
        gates = selection_output['gates']
        hard_gates = selection_output['hard_gates']
        selected_indices = selection_output['selected_band_indices']
        sparsity_loss = selection_output['sparsity_loss']
        
        # Step 3: Generate feature maps from selected bands
        
        # 3a. Reflectance maps (direct selected bands)
        selected_reflectance = selected_cube  # (batch, height, width, k_bands)
        
        # 3b. Standard deviation maps (texture)
        if self.b_std > 0:
            all_std_maps = self._compute_local_std_maps(selected_cube)  # (batch, height, width, k_bands)
            
            # Select which bands get std maps using the assignment network
            selected_band_embeddings = tf.gather(
                band_embeddings, selected_indices, batch_dims=1
            )  # (batch, k_bands, d_model)
            
            std_assignment_scores = self.std_assignment_network(selected_band_embeddings)
            std_assignment_scores = tf.squeeze(std_assignment_scores, -1)  # (batch, k_bands)
            
            # Select top b_std bands for std maps
            _, std_band_indices = tf.nn.top_k(std_assignment_scores, k=self.b_std)
            
            # Extract std maps for selected bands
            std_maps_list = []
            for b_idx in range(self.b_std):
                band_indices = std_band_indices[:, b_idx]  # (batch,)
                
                gather_indices = tf.stack([tf.range(batch_size), band_indices], axis=1)
                selected_std_map = tf.gather_nd(
                    tf.transpose(all_std_maps, [0, 3, 1, 2]),
                    gather_indices
                )  # (batch, height, width)
                std_maps_list.append(selected_std_map)
            
            std_maps = tf.stack(std_maps_list, axis=-1)  # (batch, height, width, b_std)
        else:
            std_maps = tf.zeros((batch_size, tf.shape(hyperspectral_cube)[1], 
                               tf.shape(hyperspectral_cube)[2], 1))
        
        # 3c. NDSI maps
        if self.a_ndsi > 0:
            selected_band_embeddings = tf.gather(
                band_embeddings, selected_indices, batch_dims=1
            )
            ndsi_maps = self._compute_ndsi_maps(selected_cube, selected_indices, selected_band_embeddings)
        else:
            ndsi_maps = tf.zeros((batch_size, tf.shape(hyperspectral_cube)[1], 
                               tf.shape(hyperspectral_cube)[2], 1))
        
        # 3d. Spectral index maps
        if self.c_indexes > 0:
            index_maps = self._compute_spectral_indexes(selected_cube, selected_indices, training)
        else:
            index_maps = tf.zeros((batch_size, tf.shape(hyperspectral_cube)[1], 
                                tf.shape(hyperspectral_cube)[2], 1))
        
        # Step 4: Compute combined loss
        # Diversity loss (encourage non-redundant band selection)
        diversity_loss = self.band_attention.compute_diversity_loss(diversity_scores, hard_gates)
        
        # Total loss
        total_loss = (
            self.sparsity_weight * sparsity_loss +
            self.diversity_weight * diversity_loss
        )
        
        # Prepare output
        output = {
            'selected_reflectance': selected_reflectance,
            'std_maps': std_maps,
            'ndsi_maps': ndsi_maps,
            'index_maps': index_maps,
            'selected_band_indices': selected_indices,
            'gates': gates,
            'hard_gates': hard_gates,
            'total_loss': total_loss,
            'sparsity_loss': sparsity_loss,
            'diversity_loss': diversity_loss,
            'band_importance': band_importance
        }
        
        # Add detailed attention analysis if requested
        if return_attention_analysis:
            output['attention_analysis'] = {
                'attention_weights': attention_weights,
                'band_embeddings': band_embeddings,
                'diversity_scores': diversity_scores,
                'raw_importance_scores': attention_output['raw_importance_scores'],
                'temperature': selection_output['temperature'],
                'num_selected': selection_output['num_selected']
            }
        
        return output
    
    def get_selected_bands_analysis(self) -> Dict[str, Union[tf.Tensor, List]]:
        """
        Get analysis of currently selected bands (useful after training).
        
        Returns:
            Dict with selected band information and analysis
        """
        selected_band_indices = self.gumbel_gates.get_top_k_bands()
        
        # Convert to wavelengths (approximate)
        wavelengths = np.linspace(400, 1000, self.num_bands)
        selected_wavelengths = wavelengths[selected_band_indices.numpy()]
        
        return {
            'selected_band_indices': selected_band_indices,
            'selected_wavelengths': selected_wavelengths,
            'gate_logits': self.gumbel_gates.gate_logits,
            'current_temperature': self.gumbel_gates.temperature
        }
    
    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({
            'num_bands': self.num_bands,
            'k_bands': self.k_bands,
            'a_ndsi': self.a_ndsi,
            'b_std': self.b_std,
            'c_indexes': self.c_indexes,
            'use_spatial_features': self.use_spatial_features,
            'spatial_kernel_size': self.spatial_kernel_size,
            'index_strategy': self.index_strategy,
            'diversity_weight': self.diversity_weight,
            'sparsity_weight': self.sparsity_weight,
            'random_seed': self.random_seed
        })
        return config_dict 