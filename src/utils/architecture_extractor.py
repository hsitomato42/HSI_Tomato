"""
Dynamic Architecture Configuration Extractor

This utility extracts the actual architecture configuration from built Keras models
to ensure that the configuration hash reflects the real model structure,
rather than relying on hardcoded values that can become outdated.
"""

import tensorflow as tf
from typing import Dict, Any, List, Union
import hashlib
import json
import numpy as np


class ArchitectureExtractor:
    """
    Extracts architecture configuration from Keras models dynamically.
    """
    
    @staticmethod
    def _convert_to_serializable(obj: Any) -> Any:
        """
        Convert numpy types and other non-serializable objects to JSON-serializable types.
        
        Args:
            obj: Object to convert
            
        Returns:
            JSON-serializable version of the object
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (list, tuple)):
            return [ArchitectureExtractor._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: ArchitectureExtractor._convert_to_serializable(v) for k, v in obj.items()}
        else:
            return obj

    @staticmethod
    def extract_model_config(model: tf.keras.Model) -> Dict[str, Any]:
        """
        Extract architecture configuration from a Keras model.
        
        Args:
            model: Keras model to extract configuration from
            
        Returns:
            Dict containing the model's architecture configuration
        """
        config = {
            "model_name": model.name,
            "layers": [],
            "input_shape": None,
            "output_shape": None,
            "total_params": int(model.count_params()),  # Convert to regular int
            "trainable_params": int(sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])),
        }
        
        # Extract input and output shapes (convert to regular lists)
        if hasattr(model, 'input_shape'):
            config["input_shape"] = ArchitectureExtractor._convert_to_serializable(model.input_shape)
        if hasattr(model, 'output_shape'):
            config["output_shape"] = ArchitectureExtractor._convert_to_serializable(model.output_shape)
            
        # Extract layer configurations
        for i, layer in enumerate(model.layers):
            layer_config = ArchitectureExtractor._extract_layer_config(layer, i)
            config["layers"].append(layer_config)
            
        # Extract optimizer configuration
        if hasattr(model, 'optimizer') and model.optimizer:
            config["optimizer"] = ArchitectureExtractor._extract_optimizer_config(model.optimizer)
        
        # Extract loss configuration
        if hasattr(model, 'loss') and model.loss:
            config["loss"] = str(model.loss) if callable(model.loss) else model.loss
            
        # Convert entire config to ensure all values are serializable
        config = ArchitectureExtractor._convert_to_serializable(config)
        
        return config
    
    @staticmethod
    def _extract_layer_config(layer: tf.keras.layers.Layer, index: int) -> Dict[str, Any]:
        """
        Extract configuration from a single layer.
        
        Args:
            layer: Keras layer to extract configuration from
            index: Layer index in the model
            
        Returns:
            Dict containing layer configuration
        """
        layer_config = {
            "index": index,
            "name": layer.name,
            "type": layer.__class__.__name__,
            "output_shape": ArchitectureExtractor._convert_to_serializable(layer.output_shape) if hasattr(layer, 'output_shape') else None,
            "trainable": layer.trainable,
            "params": int(layer.count_params())  # Convert to regular int
        }
        
        # Extract layer-specific configurations
        if hasattr(layer, 'get_config'):
            try:
                # Get the layer's configuration
                layer_keras_config = layer.get_config()
                
                # Extract key parameters based on layer type
                layer_specific_config = ArchitectureExtractor._extract_layer_specific_config(layer, layer_keras_config)
                # Convert to serializable types
                layer_specific_config = ArchitectureExtractor._convert_to_serializable(layer_specific_config)
                layer_config.update(layer_specific_config)
                
            except Exception as e:
                # Some layers might not have get_config or it might fail
                layer_config["config_error"] = str(e)
        
        return layer_config
    
    @staticmethod
    def _extract_layer_specific_config(layer: tf.keras.layers.Layer, keras_config: Dict) -> Dict[str, Any]:
        """
        Extract layer-specific important parameters.
        
        Args:
            layer: Keras layer
            keras_config: Layer's get_config() result
            
        Returns:
            Dict with relevant parameters for this layer type
        """
        config = {}
        layer_type = layer.__class__.__name__
        
        # Convolutional layers
        if layer_type in ['Conv2D', 'Conv1D', 'Conv3D']:
            config.update({
                "filters": keras_config.get("filters"),
                "kernel_size": keras_config.get("kernel_size"),
                "strides": keras_config.get("strides"),
                "padding": keras_config.get("padding"),
                "activation": keras_config.get("activation"),
                "use_bias": keras_config.get("use_bias")
            })
            
        # Dense layers
        elif layer_type == 'Dense':
            config.update({
                "units": keras_config.get("units"),
                "activation": keras_config.get("activation"),
                "use_bias": keras_config.get("use_bias")
            })
            
        # Pooling layers
        elif layer_type in ['MaxPooling2D', 'AveragePooling2D', 'GlobalMaxPooling2D', 'GlobalAveragePooling2D']:
            if 'pool_size' in keras_config:
                config["pool_size"] = keras_config.get("pool_size")
            if 'strides' in keras_config:
                config["strides"] = keras_config.get("strides")
            if 'padding' in keras_config:
                config["padding"] = keras_config.get("padding")
                
        # Dropout layers
        elif layer_type == 'Dropout':
            config["rate"] = keras_config.get("rate")
            
        # Normalization layers
        elif layer_type in ['BatchNormalization', 'LayerNormalization']:
            config.update({
                "axis": keras_config.get("axis"),
                "epsilon": keras_config.get("epsilon"),
                "center": keras_config.get("center", True),
                "scale": keras_config.get("scale", True)
            })
            
        # Attention layers
        elif layer_type in ['MultiHeadAttention']:
            config.update({
                "num_heads": keras_config.get("num_heads"),
                "key_dim": keras_config.get("key_dim"),
                "dropout": keras_config.get("dropout", 0.0)
            })
            
        # Reshape layers
        elif layer_type == 'Reshape':
            config["target_shape"] = keras_config.get("target_shape")
            
        # Lambda layers (custom operations)
        elif layer_type == 'Lambda':
            # Lambda layers can't be fully serialized, but we can note they exist
            config["function_type"] = "lambda_function"
            
        # For other layer types, include basic info
        else:
            # Include any numeric or string parameters that might be relevant
            for key, value in keras_config.items():
                if isinstance(value, (int, float, str, bool, list, tuple)) and key not in ['name', 'trainable']:
                    config[key] = value
                    
        # Remove None values to keep config clean
        config = {k: v for k, v in config.items() if v is not None}
        
        return config
    
    @staticmethod
    def _extract_optimizer_config(optimizer) -> Dict[str, Any]:
        """
        Extract optimizer configuration.
        
        Args:
            optimizer: Keras optimizer
            
        Returns:
            Dict containing optimizer configuration
        """
        config = {
            "type": optimizer.__class__.__name__
        }
        
        # Extract common optimizer parameters
        if hasattr(optimizer, 'learning_rate'):
            # Handle both scalar and Variable learning rates
            lr = optimizer.learning_rate
            if hasattr(lr, 'numpy'):
                config["learning_rate"] = float(lr.numpy())
            else:
                config["learning_rate"] = float(lr)
                
        if hasattr(optimizer, 'beta_1'):
            config["beta_1"] = float(optimizer.beta_1)
        if hasattr(optimizer, 'beta_2'):
            config["beta_2"] = float(optimizer.beta_2)
        if hasattr(optimizer, 'epsilon'):
            config["epsilon"] = float(optimizer.epsilon)
        if hasattr(optimizer, 'weight_decay') and optimizer.weight_decay is not None:
            config["weight_decay"] = float(optimizer.weight_decay)
            
        return config
    
    @staticmethod
    def create_architecture_hash(model: tf.keras.Model) -> str:
        """
        Create a hash of the model's architecture.
        
        Args:
            model: Keras model to hash
            
        Returns:
            SHA256 hash string (first 16 characters)
        """
        config = ArchitectureExtractor.extract_model_config(model)
        
        # Convert to consistent string representation
        config_str = json.dumps(config, sort_keys=True, default=str)
        
        # Generate hash
        hash_obj = hashlib.sha256(config_str.encode('utf-8'))
        return hash_obj.hexdigest()[:16]
    
    @staticmethod
    def compare_models(model1: tf.keras.Model, model2: tf.keras.Model) -> Dict[str, Any]:
        """
        Compare two models' architectures.
        
        Args:
            model1: First model
            model2: Second model
            
        Returns:
            Dict containing comparison results
        """
        config1 = ArchitectureExtractor.extract_model_config(model1)
        config2 = ArchitectureExtractor.extract_model_config(model2)
        
        hash1 = ArchitectureExtractor.create_architecture_hash(model1)
        hash2 = ArchitectureExtractor.create_architecture_hash(model2)
        
        return {
            "models_identical": hash1 == hash2,
            "hash1": hash1,
            "hash2": hash2,
            "param_count_diff": config1["total_params"] - config2["total_params"],
            "layer_count_diff": len(config1["layers"]) - len(config2["layers"])
        } 