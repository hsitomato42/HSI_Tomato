# utils/model_factory.py

import tensorflow as tf
from typing import Dict, Any, Optional, Tuple, List
import src.config as config
from src.config.enums import ModelType


class ModelFactory:
    """Factory class for creating models with progressive feature selection support.
    
    This factory creates fresh models for each stage of progressive feature selection,
    ensuring no bias between stages and allowing for different architectures based on
    the increasing number of features.
    """
    
    # Define valid parameters for each model type
    _MODEL_PARAMETERS = {
        ModelType.CNN: {
            'filters', 'kernel_size', 'dropout_rate', 'learning_rate', 'batch_size'
        },
        ModelType.CNN_TRANSFORMER: {
            'filters', 'kernel_size', 'dropout_rate', 'learning_rate', 'batch_size',
            'd_model', 'transformer_heads', 'transformer_layers'
        },
        ModelType.MULTI_HEAD_CNN: {
            'filters', 'kernel_size', 'dropout_rate', 'learning_rate', 'batch_size'
        },
        ModelType.CDAT_PROGRESSIVE: {
            'dropout_rate',
            'backbone_filters_start', 'backbone_filters_growth', 'backbone_kernels', 
            'backbone_depth', 'backbone_use_pooling', 'backbone_pool_size',
            'component_filters', 'component_depth', 'd_model', 'num_attention_heads',
            'self_attention_layers', 'transformer_dff', 'transformer_dropout_rate',
            'component_processing_order', 'memory_accumulation_method', 
            'progressive_scaling', 'use_memory_gates', 'use_spatial_component_queries',
            'query_dimension_per_head', 'mlp_head_units', 'mlp_dropout_rate',
            'optimizer_params'
        },
        ModelType.COMPONENT_DRIVEN_ATTENTION_TRANSFORMER: {
            'dropout_rate',
            'backbone_filters_start', 'backbone_filters_growth', 'backbone_kernels',
            'backbone_depth', 'backbone_use_pooling', 'backbone_pool_size',
            'd_model', 'component_attention_heads', 'num_attention_layers_per_component',
            'fusion_method', 'component_processing_order', 'final_transformer_layers',
            'transformer_heads', 'transformer_dff', 'transformer_dropout_rate',
            'mlp_head_units', 'mlp_dropout_rate', 'optimizer_params'
        },
        ModelType.COMPONENT_DRIVEN_ATTENTION_TRANSFORMER_V2: {
            'dropout_rate',
            'backbone_filters_before_downsample', 'backbone_filters_after_downsample',
            'backbone_kernels', 'downsampling_factor', 'downsampling_method',
            'advisor_filters_before_downsample', 'advisor_filters_after_downsample',
            'd_model', 'component_attention_heads', 'num_attention_layers_per_component',
            'fusion_method', 'component_processing_order', 'final_transformer_layers',
            'transformer_heads', 'transformer_dff', 'transformer_dropout_rate',
            'mlp_head_units', 'mlp_dropout_rate', 'optimizer_params'
        }
    }
    
    @staticmethod
    def _filter_hyperparameters(
        hyperparameters: Dict[str, Any], 
        model_type: ModelType
    ) -> Dict[str, Any]:
        """Filter hyperparameters to only include those accepted by the model type.
        
        Args:
            hyperparameters: All hyperparameters
            model_type: Target model type
            
        Returns:
            Filtered hyperparameters containing only valid parameters for the model
        """
        valid_params = ModelFactory._MODEL_PARAMETERS.get(model_type, set())
        filtered = {k: v for k, v in hyperparameters.items() if k in valid_params}
        
        # Handle special cases for transformer models
        if model_type in [
            ModelType.COMPONENT_DRIVEN_ATTENTION_TRANSFORMER,
            ModelType.COMPONENT_DRIVEN_ATTENTION_TRANSFORMER_V2,
            ModelType.CDAT_PROGRESSIVE
        ]:
            # Convert learning_rate to optimizer_params if needed
            if 'learning_rate' in hyperparameters:
                learning_rate = hyperparameters['learning_rate']
                if 'optimizer_params' not in filtered:
                    filtered['optimizer_params'] = {}
                filtered['optimizer_params']['learning_rate'] = learning_rate
                
            # Remove batch_size as it's not used in model construction
            filtered.pop('batch_size', None)
        
        # Log filtered out parameters for debugging
        filtered_out = set(hyperparameters.keys()) - valid_params - {'learning_rate', 'batch_size'}
        if filtered_out:
            print(f"[ModelFactory] Filtered out parameters for {model_type.name}: {filtered_out}")
        
        return filtered
    
    @staticmethod
    def build_model(
        input_channels: int, 
        stage_name: str,
        model_type: Optional[ModelType] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> tf.keras.Model:
        """Build a model for a specific progressive feature selection stage.
        
        Args:
            input_channels: Number of input channels for this stage
            stage_name: Name of the current stage ('bands', 'std', 'indexes', 'ndsi', 'finetune')
            model_type: Model type to create (defaults to config.MODEL_TYPE)
            hyperparameters: Model-specific hyperparameters
            **kwargs: Additional arguments passed to the model constructor
            
        Returns:
            A new tf.keras.Model instance configured for the stage
        """
        # Use config model type if not specified
        if model_type is None:
            model_type = config.MODEL_TYPE
            
        # Default hyperparameters if not provided
        if hyperparameters is None:
            hyperparameters = {}
            
        # Get image dimensions from config
        use_patches = getattr(config, 'USE_PATCHES', False)
        if use_patches:
            patch_size = getattr(config, 'PATCH_SIZE', 15)
            height = width = patch_size
        else:
            # Full cube dimensions
            if hasattr(config, 'MAX_2D_DIMENSIONS') and config.MAX_2D_DIMENSIONS is not None:
                # Use global max dimensions computed during data processing
                height, width = config.MAX_2D_DIMENSIONS
            else:
                height = getattr(config, 'IMAGE_HEIGHT', 23)
                width = getattr(config, 'IMAGE_WIDTH', 23)
            
        # Get number of output attributes
        predicted_attributes = getattr(config, 'PREDICTED_QUALITY_ATTRIBUTES', [])
        num_outputs = len(predicted_attributes)
        
        # Stage-specific model configuration
        stage_configs = ModelFactory._get_stage_specific_config(
            stage_name, input_channels, hyperparameters, model_type
        )
        
        # Merge stage config with hyperparameters
        merged_hyperparameters = {**hyperparameters, **stage_configs}
        
        # Filter hyperparameters based on model type
        final_hyperparameters = ModelFactory._filter_hyperparameters(
            merged_hyperparameters, model_type
        )
        
        # Build input shape with dynamic channel count
        input_shape = (height, width, input_channels)
        
        print(f"[ModelFactory] Building {model_type.name} model for stage '{stage_name}'")
        print(f"[ModelFactory] Input shape: {input_shape}")
        print(f"[ModelFactory] Output dimensions: {num_outputs}")
        print(f"[ModelFactory] Filtered hyperparameters: {final_hyperparameters}")
        
        # Create the model based on type
        if model_type == ModelType.CNN:
            model = ModelFactory._build_cnn_model(
                input_shape, num_outputs, final_hyperparameters, **kwargs
            )
        elif model_type == ModelType.CNN_TRANSFORMER:
            model = ModelFactory._build_cnn_transformer_model(
                input_shape, num_outputs, final_hyperparameters, **kwargs
            )
        elif model_type == ModelType.MULTI_HEAD_CNN:
            model = ModelFactory._build_multi_head_cnn_model(
                input_shape, num_outputs, final_hyperparameters, **kwargs
            )
        elif model_type == ModelType.CDAT_PROGRESSIVE:
            model = ModelFactory._build_cdat_model(
                input_shape, num_outputs, final_hyperparameters, **kwargs
            )
        elif model_type == ModelType.COMPONENT_DRIVEN_ATTENTION_TRANSFORMER:
            model = ModelFactory._build_component_driven_attention_transformer_model(
                input_shape, num_outputs, final_hyperparameters, **kwargs
            )
        elif model_type == ModelType.COMPONENT_DRIVEN_ATTENTION_TRANSFORMER_V2:
            model = ModelFactory._build_component_driven_attention_transformer_v2_model(
                input_shape, num_outputs, final_hyperparameters, **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type for progressive selection: {model_type}")
            
        # Log model summary
        print(f"[ModelFactory] Model created with {model.count_params():,} parameters")
        
        return model
    
    @staticmethod
    def _get_stage_specific_config(
        stage_name: str, 
        input_channels: int,
        base_hyperparameters: Dict[str, Any],
        model_type: ModelType
    ) -> Dict[str, Any]:
        """Get stage-specific model configuration.
        
        Different stages may benefit from different model capacities based on
        the number of features available.
        
        Args:
            stage_name: Current stage name
            input_channels: Number of input channels
            base_hyperparameters: Base hyperparameters to modify
            model_type: Type of model being built
            
        Returns:
            Stage-specific configuration updates
        """
        config_updates = {}
        
        # CNN-specific configurations
        if model_type in [ModelType.CNN, ModelType.CNN_TRANSFORMER, ModelType.MULTI_HEAD_CNN]:
            if stage_name == 'bands':
                # Early stage with few features - smaller model
                config_updates['filters'] = base_hyperparameters.get('filters', [32, 64, 128])
                config_updates['dropout_rate'] = base_hyperparameters.get('dropout_rate', 0.3)
                
            elif stage_name == 'std':
                # Adding texture features - slightly larger model
                config_updates['filters'] = base_hyperparameters.get('filters', [32, 64, 128])
                config_updates['dropout_rate'] = base_hyperparameters.get('dropout_rate', 0.3)
                
            elif stage_name == 'indexes':
                # More features available - standard model
                config_updates['filters'] = base_hyperparameters.get('filters', [64, 128, 256])
                config_updates['dropout_rate'] = base_hyperparameters.get('dropout_rate', 0.4)
                
            elif stage_name == 'ndsi':
                # Near full feature set - larger model
                config_updates['filters'] = base_hyperparameters.get('filters', [64, 128, 256])
                config_updates['dropout_rate'] = base_hyperparameters.get('dropout_rate', 0.4)
                
            elif stage_name == 'finetune':
                # Full feature set - full capacity model
                config_updates['filters'] = base_hyperparameters.get('filters', [64, 128, 256, 512])
                config_updates['dropout_rate'] = base_hyperparameters.get('dropout_rate', 0.5)
                
            # Scale learning rate based on model size for CNN models
            if 'learning_rate' not in base_hyperparameters:
                if stage_name in ['bands', 'std']:
                    config_updates['learning_rate'] = 0.001
                else:
                    config_updates['learning_rate'] = 0.0005
                    
        # Transformer-specific configurations
        elif model_type in [
            ModelType.COMPONENT_DRIVEN_ATTENTION_TRANSFORMER,
            ModelType.COMPONENT_DRIVEN_ATTENTION_TRANSFORMER_V2,
            ModelType.CDAT_PROGRESSIVE
        ]:
            # Use transformer-specific parameters
            if stage_name == 'bands':
                config_updates['backbone_filters_start'] = base_hyperparameters.get('backbone_filters_start', 16)
                config_updates['d_model'] = base_hyperparameters.get('d_model', 64)
                config_updates['transformer_dropout_rate'] = base_hyperparameters.get('transformer_dropout_rate', 0.1)
                
            elif stage_name == 'std':
                config_updates['backbone_filters_start'] = base_hyperparameters.get('backbone_filters_start', 24)
                config_updates['d_model'] = base_hyperparameters.get('d_model', 96)
                config_updates['transformer_dropout_rate'] = base_hyperparameters.get('transformer_dropout_rate', 0.1)
                
            elif stage_name == 'indexes':
                config_updates['backbone_filters_start'] = base_hyperparameters.get('backbone_filters_start', 32)
                config_updates['d_model'] = base_hyperparameters.get('d_model', 128)
                config_updates['transformer_dropout_rate'] = base_hyperparameters.get('transformer_dropout_rate', 0.1)
                
            elif stage_name == 'ndsi':
                config_updates['backbone_filters_start'] = base_hyperparameters.get('backbone_filters_start', 32)
                config_updates['d_model'] = base_hyperparameters.get('d_model', 128)
                config_updates['transformer_dropout_rate'] = base_hyperparameters.get('transformer_dropout_rate', 0.1)
                
            elif stage_name == 'finetune':
                config_updates['backbone_filters_start'] = base_hyperparameters.get('backbone_filters_start', 48)
                config_updates['d_model'] = base_hyperparameters.get('d_model', 192)
                config_updates['transformer_dropout_rate'] = base_hyperparameters.get('transformer_dropout_rate', 0.2)
                
            # Learning rate for transformer models - will be converted to optimizer_params in filtering
            if 'learning_rate' not in base_hyperparameters:
                if stage_name in ['bands', 'std']:
                    config_updates['learning_rate'] = 0.0001
                else:
                    config_updates['learning_rate'] = 0.00005
                
        return config_updates
    
    @staticmethod
    def _build_cnn_model(
        input_shape: Tuple[int, int, int],
        num_outputs: int,
        hyperparameters: Dict[str, Any],
        **kwargs
    ) -> tf.keras.Model:
        """Build a CNN model with dynamic input shape.
        
        Args:
            input_shape: Input shape (H, W, C) where C is the dynamic channel count
            num_outputs: Number of output predictions
            hyperparameters: Model hyperparameters
            **kwargs: Additional arguments
            
        Returns:
            CNN model instance
        """
        # Extract hyperparameters
        filters = hyperparameters.get('filters', [64, 128, 256])
        kernel_size = hyperparameters.get('kernel_size', 3)
        dropout_rate = hyperparameters.get('dropout_rate', 0.4)
        learning_rate = hyperparameters.get('learning_rate', 0.001)
        
        # Build model directly with dynamic input shape
        if config.MULTI_REGRESSION:
            return ModelFactory._build_cnn_multi_output_model(
                input_shape, num_outputs, filters, kernel_size, dropout_rate, learning_rate
            )
        else:
            return ModelFactory._build_cnn_single_output_model(
                input_shape, filters, kernel_size, dropout_rate, learning_rate
            )
    
    @staticmethod
    def _build_cnn_multi_output_model(
        input_shape: Tuple[int, int, int],
        num_outputs: int,
        filters: List[int],
        kernel_size: int,
        dropout_rate: float,
        learning_rate: float
    ) -> tf.keras.Model:
        """Build a multi-output CNN model with dynamic input shape."""
        inputs = tf.keras.layers.Input(shape=input_shape)
        
        # CNN feature extraction layers
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        
        # Multi-output layer - one output for all attributes
        outputs = tf.keras.layers.Dense(num_outputs, activation='linear', name='multi_output')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs, name='CNNMultiRegression')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    @staticmethod
    def _build_cnn_single_output_model(
        input_shape: Tuple[int, int, int],
        filters: List[int],
        kernel_size: int,
        dropout_rate: float,
        learning_rate: float
    ) -> tf.keras.Model:
        """Build a single-output CNN model with dynamic input shape."""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            # Single output (one attribute)
            tf.keras.layers.Dense(1, activation='linear')
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )
        return model
    
    @staticmethod
    def _build_cnn_transformer_model(
        input_shape: Tuple[int, int, int],
        num_outputs: int,
        hyperparameters: Dict[str, Any],
        **kwargs
    ) -> tf.keras.Model:
        """Build a CNN-Transformer model with dynamic input shape."""
        # For now, just use the CNN model as a placeholder
        # In a real implementation, you would implement the transformer architecture
        return ModelFactory._build_cnn_model(input_shape, num_outputs, hyperparameters, **kwargs)
    
    @staticmethod
    def _build_multi_head_cnn_model(
        input_shape: Tuple[int, int, int],
        num_outputs: int,
        hyperparameters: Dict[str, Any],
        **kwargs
    ) -> tf.keras.Model:
        """Build a Multi-Head CNN model with dynamic input shape."""
        # For now, just use the CNN model as a placeholder
        # In a real implementation, you would implement the multi-head architecture
        return ModelFactory._build_cnn_model(input_shape, num_outputs, hyperparameters, **kwargs)
    
    @staticmethod
    def _build_cdat_model(
        input_shape: Tuple[int, int, int],
        num_outputs: int,
        hyperparameters: Dict[str, Any],
        **kwargs
    ) -> tf.keras.Model:
        """Build a CDAT model with dynamic input shape."""
        # For now, just use the CNN model as a placeholder
        # In a real implementation, you would implement the CDAT architecture
        return ModelFactory._build_cnn_model(input_shape, num_outputs, hyperparameters, **kwargs)
    
    @staticmethod
    def get_stage_hyperparameters(stage_name: str) -> Dict[str, Any]:
        """Get recommended hyperparameters for a specific stage.
        
        Args:
            stage_name: Stage name
            
        Returns:
            Recommended hyperparameters for the stage
        """
        base_hp = {
            'bands': {
                'filters': [32, 64, 128],
                'learning_rate': 0.001,
                'dropout_rate': 0.3,
                'batch_size': 16
            },
            'std': {
                'filters': [32, 64, 128],
                'learning_rate': 0.001,
                'dropout_rate': 0.3,
                'batch_size': 16
            },
            'indexes': {
                'filters': [64, 128, 256],
                'learning_rate': 0.0005,
                'dropout_rate': 0.4,
                'batch_size': 16
            },
            'ndsi': {
                'filters': [64, 128, 256],
                'learning_rate': 0.0005,
                'dropout_rate': 0.4,
                'batch_size': 16
            },
            'finetune': {
                'filters': [64, 128, 256, 512],
                'learning_rate': 0.0001,
                'dropout_rate': 0.5,
                'batch_size': 8
            }
        }
        
        return base_hp.get(stage_name, base_hp['ndsi'])

    @staticmethod
    def _build_component_driven_attention_transformer_model(
        input_shape: tuple, num_outputs: int, hyperparameters: dict, **kwargs
    ) -> tf.keras.Model:
        """
        Build a Component-Driven Attention Transformer model for progressive selection.
        """
        print(f"[ModelFactory] Building COMPONENT_DRIVEN_ATTENTION_TRANSFORMER model")
        
        from Models.dl_models.ComponentDrivenAttentionTransformer import ComponentDrivenAttentionTransformer
        
        # Extract model parameters
        model_name = kwargs.get('model_name', f'ComponentDrivenAttentionTransformer_{kwargs.get("stage_name", "default")}')
        model_filename = kwargs.get('model_filename', f'cdat_{kwargs.get("stage_name", "default")}.h5')
        components = kwargs.get('components', {'reflectance': True, 'std': True, 'ndsi': True, 'indexes': True})
        component_dimensions = kwargs.get('component_dimensions', {'reflectance': input_shape[2], 'total_channels': input_shape[2]})
        predicted_attributes = kwargs.get('predicted_attributes', ['TSS', 'citric_acid', 'firmness', 'pH', 'weight', 'ascorbic_acid'])
        selected_bands = kwargs.get('selected_bands', None)
        selected_indexes = kwargs.get('selected_indexes', None)
        
        # Create model instance
        model_instance = ComponentDrivenAttentionTransformer(
            model_name=model_name,
            model_filename=model_filename,
            model_shape=input_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes,
            selected_bands=selected_bands,
            selected_indexes=selected_indexes,
            **hyperparameters
        )
        
        # Build the model
        if len(predicted_attributes) == 1:
            model = model_instance._build_model_for_attr()
        else:
            model = model_instance._build_multi_output_model()
        
        return model

    @staticmethod
    def _build_component_driven_attention_transformer_v2_model(
        input_shape: tuple, num_outputs: int, hyperparameters: dict, **kwargs
    ) -> tf.keras.Model:
        """
        Build a Component-Driven Attention Transformer V2 model for progressive selection.
        """
        print(f"[ModelFactory] Building COMPONENT_DRIVEN_ATTENTION_TRANSFORMER_V2 model")
        
        from Models.dl_models.ComponentDrivenAttentionTransformerV2 import ComponentDrivenAttentionTransformerV2
        
        # Extract model parameters
        model_name = kwargs.get('model_name', f'ComponentDrivenAttentionTransformerV2_{kwargs.get("stage_name", "default")}')
        model_filename = kwargs.get('model_filename', f'cdat_v2_{kwargs.get("stage_name", "default")}.h5')
        components = kwargs.get('components', {'reflectance': True, 'std': True, 'ndsi': True, 'indexes': True})
        component_dimensions = kwargs.get('component_dimensions', {'reflectance': input_shape[2], 'total_channels': input_shape[2]})
        predicted_attributes = kwargs.get('predicted_attributes', ['TSS', 'citric_acid', 'firmness', 'pH', 'weight', 'ascorbic_acid'])
        selected_bands = kwargs.get('selected_bands', None)
        selected_indexes = kwargs.get('selected_indexes', None)
        
        # Create model instance
        model_instance = ComponentDrivenAttentionTransformerV2(
            model_name=model_name,
            model_filename=model_filename,
            model_shape=input_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes,
            selected_bands=selected_bands,
            selected_indexes=selected_indexes,
            **hyperparameters
        )
        
        # Build the model
        if len(predicted_attributes) == 1:
            model = model_instance._build_model_for_attr()
        else:
            model = model_instance._build_multi_output_model()
        
        return model
