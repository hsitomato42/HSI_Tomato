# utils/model_config_manager.py

import os
import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from src.config.enums import ModelType, SpectralIndex
from .architecture_extractor import ArchitectureExtractor


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle numpy types and other non-serializable objects."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            # For complex objects, try to serialize their dict representation
            return str(obj)
        return super().default(obj)


class ModelConfigManager:
    """
    Manages model configuration generation, hashing, and saving for the new model storage system.
    Uses dynamic architecture extraction to ensure configuration reflects actual model structure.
    """
    
    def __init__(self):
        self.base_save_dir = os.path.join("Models", "saved_models")
    
    def generate_model_config(
        self,
        model_instance,
        model_type: ModelType,
        model_shape: Tuple[int, int, int],
        components: Dict[str, bool],
        component_dimensions: Dict[str, int],
        selected_bands: List[int],
        selected_indexes: Optional[List[SpectralIndex]] = None,
        predicted_attributes: List[str] = None,
        ndsi_band_pairs: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, Any]:
        """
        Generate the complete model configuration dictionary for hashing.
        
        Args:
            model_instance: The model object to get architecture config from
            model_type: ModelType enum
            model_shape: (H, W, C) tuple
            components: Dict indicating which components are used
            component_dimensions: Dict with component dimensions info
            selected_bands: List of selected bands for reflectance
            selected_indexes: List of selected spectral indexes
            predicted_attributes: List of quality attributes to predict
            ndsi_band_pairs: List of band pairs for NDSI
            
        Returns:
            Dict containing all model configuration data
        """
        import config
        
        # Core configuration
        model_config = {
            "model_type": model_type.value,
            "regression_mode": "multi" if config.MULTI_REGRESSION else "single",
            "use_validation": config.USE_VALIDATION,
            "model_shape": list(model_shape),  # Convert tuple to list for JSON serialization
            "predicted_attributes": sorted(predicted_attributes or config.PREDICTED_QUALITY_ATTRIBUTES)
        }
        
        # Component configuration
        model_config["components"] = dict(sorted(components.items()))
        model_config["component_dimensions"] = dict(sorted(component_dimensions.items()))
        
        # Component values
        model_config["selected_bands"] = sorted(selected_bands)
        model_config["selected_indexes"] = sorted([idx.value for idx in (selected_indexes or [])])
        model_config["ndsi_band_pairs"] = sorted([list(pair) for pair in (ndsi_band_pairs or [])])
        
        # Dynamic architecture configuration
        model_config["architecture"] = self._extract_dynamic_architecture(model_instance, config.MULTI_REGRESSION)
            
        return model_config
        
    def _extract_dynamic_architecture(self, model_instance, multi_regression: bool) -> Dict[str, Any]:
        """
        Extract architecture configuration dynamically from the actual model(s).
        
        Args:
            model_instance: Model instance with built Keras models
            multi_regression: Whether in multi-regression mode
            
        Returns:
            Dict containing dynamic architecture configuration
        """
        try:
            if multi_regression:
                # Multi-regression: extract from the single model
                if hasattr(model_instance, 'model') and model_instance.model is not None:
                    arch_config = ArchitectureExtractor.extract_model_config(model_instance.model)
                    arch_config["mode"] = "multi_regression"
                    return arch_config
                else:
                    print(f"[ModelConfigManager] Warning: Multi-regression model not built yet, using fallback config")
                    return {"mode": "multi_regression", "status": "not_built"}
            else:
                # Single prediction: extract from all individual models
                arch_configs = {}
                if hasattr(model_instance, 'models') and model_instance.models:
                    for attr, model in model_instance.models.items():
                        if model is not None:
                            arch_configs[attr] = ArchitectureExtractor.extract_model_config(model)
                        else:
                            arch_configs[attr] = {"status": "not_built"}
                    arch_configs["mode"] = "single_prediction"
                    return arch_configs
                else:
                    print(f"[ModelConfigManager] Warning: Single prediction models not built yet, using fallback config")
                    return {"mode": "single_prediction", "status": "not_built"}
                    
        except Exception as e:
            print(f"[ModelConfigManager] Error extracting dynamic architecture: {e}")
            # Fallback to legacy method if dynamic extraction fails
            if hasattr(model_instance, 'get_architecture_config'):
                fallback_config = model_instance.get_architecture_config()
                fallback_config["extraction_method"] = "fallback_legacy"
                return fallback_config
            else:
                return {"extraction_method": "failed", "error": str(e)}
    
    def _serialize_for_hash(self, obj: Any) -> str:
        """
        Recursively serialize an object to a consistent string representation for hashing.
        
        Args:
            obj: Object to serialize
            
        Returns:
            String representation suitable for hashing
        """
        # Handle numpy types first
        if isinstance(obj, np.integer):
            return str(int(obj))
        elif isinstance(obj, np.floating):
            return str(float(obj))
        elif isinstance(obj, np.ndarray):
            return self._serialize_for_hash(obj.tolist())
        elif isinstance(obj, dict):
            # Sort keys for consistent ordering
            items = sorted(obj.items())
            return "{" + ",".join(f'"{k}":{self._serialize_for_hash(v)}' for k, v in items) + "}"
        elif isinstance(obj, (list, tuple)):
            return "[" + ",".join(self._serialize_for_hash(item) for item in obj) + "]"
        elif isinstance(obj, str):
            return f'"{obj}"'
        elif obj is None:
            return "null"
        else:
            return str(obj)
    
    def generate_config_hash(self, model_config: Dict[str, Any]) -> str:
        """
        Generate a deterministic hash from the model configuration.
        
        Args:
            model_config: Model configuration dictionary
            
        Returns:
            SHA256 hash string (first 16 characters for readability)
        """
        # Serialize the configuration to a consistent string
        config_str = self._serialize_for_hash(model_config)
        
        # Generate SHA256 hash
        hash_obj = hashlib.sha256(config_str.encode('utf-8'))
        full_hash = hash_obj.hexdigest()
        
        # Return first 16 characters for readability while maintaining uniqueness
        return full_hash[:16]
    
    def create_model_directory(self, model_type: ModelType, config_hash: str) -> str:
        """
        Create the directory structure for saving the model.
        
        Args:
            model_type: ModelType enum
            config_hash: Configuration hash string
            
        Returns:
            Path to the created model directory
        """
        # Create full path: Models/saved_models/[MODEL_TYPE]/[hash]/
        model_dir = os.path.join(
            self.base_save_dir,
            model_type.value,
            config_hash
        )
        
        # Create directories if they don't exist
        os.makedirs(model_dir, exist_ok=True)
        
        return model_dir
    
    def save_model_config(self, model_config: Dict[str, Any], model_dir: str) -> str:
        """
        Save the model configuration as JSON file.
        
        Args:
            model_config: Model configuration dictionary
            model_dir: Directory to save the config file in
            
        Returns:
            Path to the saved config file
        """
        config_path = os.path.join(model_dir, "model_config.json")
        
        # Save with pretty formatting for readability and custom encoder for numpy types
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(model_config, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        return config_path
    
    def get_model_file_path(self, model_dir: str, model_type: ModelType, regression_mode: str) -> str:
        """
        Generate the model file path within the model directory.
        
        Args:
            model_dir: Model directory path
            model_type: ModelType enum
            regression_mode: "multi" or "single"
            
        Returns:
            Full path for the model file
        """
        if regression_mode == "multi":
            filename = "model.keras"
        else:
            filename = "model.keras"  # For now, use same name. Could be model_{attr}.keras for single mode
        
        return os.path.join(model_dir, filename)
    
    def save_model_complete(
        self,
        model_instance,
        model_type: ModelType,
        model_shape: Tuple[int, int, int],
        components: Dict[str, bool],
        component_dimensions: Dict[str, int],
        selected_bands: List[int],
        selected_indexes: Optional[List[SpectralIndex]] = None,
        predicted_attributes: List[str] = None,
        ndsi_band_pairs: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, str]:
        """
        Complete model saving process: generate config, create directories, save model and config.
        
        Returns:
            Dict with paths to saved files: {"model_dir": path, "config_file": path, "model_file": path}
        """
        import config
        
        # Generate model configuration
        model_config = self.generate_model_config(
            model_instance=model_instance,
            model_type=model_type,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            selected_bands=selected_bands,
            selected_indexes=selected_indexes,
            predicted_attributes=predicted_attributes,
            ndsi_band_pairs=ndsi_band_pairs
        )
        
        # Generate hash and create directory
        config_hash = self.generate_config_hash(model_config)
        model_dir = self.create_model_directory(model_type, config_hash)
        
        # Save configuration file
        config_file = self.save_model_config(model_config, model_dir)
        
        # Get model file path (actual model saving will be done by the model instance)
        regression_mode = "multi" if config.MULTI_REGRESSION else "single"
        model_file = self.get_model_file_path(model_dir, model_type, regression_mode)
        
        print(f"[ModelConfigManager] Model directory created: {model_dir}")
        print(f"[ModelConfigManager] Configuration saved: {config_file}")
        
        return {
            "model_dir": model_dir,
            "config_file": config_file,
            "model_file": model_file,
            "config_hash": config_hash
        }
    
    def save_feature_importance(
        self,
        importance_results: Dict[str, Dict[str, Any]],
        model_dir: str,
        feature_names: Optional[List[str]] = None
    ) -> str:
        """
        Save feature importance results to the model directory.
        
        Args:
            importance_results: Feature importance results dictionary
            model_dir: Model directory path
            feature_names: Optional list of feature/channel names
            
        Returns:
            Path to the saved importance file
        """
        importance_data = {
            "importance_results": importance_results,
            "feature_names": feature_names or [],
            "timestamp": pd.Timestamp.now().isoformat(),
            "num_features": len(feature_names) if feature_names else 0
        }
        
        importance_path = os.path.join(model_dir, "feature_importance.json")
        
        # Save with custom encoder for numpy types
        with open(importance_path, 'w', encoding='utf-8') as f:
            json.dump(importance_data, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
        
        print(f"[ModelConfigManager] Feature importance saved: {importance_path}")
        return importance_path
    
    def load_model_config(self, model_dir: str) -> Dict[str, Any]:
        """
        Load model configuration from a saved model directory.
        
        Args:
            model_dir: Path to model directory
            
        Returns:
            Model configuration dictionary
        """
        config_path = os.path.join(model_dir, "model_config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f) 