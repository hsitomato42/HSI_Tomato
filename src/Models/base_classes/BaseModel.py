# Models/BaseModel.py

from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict
import matplotlib.pyplot as plt
import numpy as np
import hashlib
import json

# Import utility functions
import src.config as config
from src.config.enums import ModelType, SpectralIndex
from src.utils.data_processing import DataProcessingUtils
from src.utils.evaluation import evaluate_metrics, plot_evaluation_metrics


class BaseModel(ABC):
    """
    Abstract base class for all predictive models.
    Defines the essential interface and common functionalities.
    """

    def __init__(self, model_type: ModelType , model_name: str, model_filename: str , selected_indexes: Optional[List[SpectralIndex]] = None):
        """
        Initializes the BaseModel with a model name and selected spectral indexes.

        Args:
            model_name (str): The name of the model.
            selected_indexes (Optional[List[SpectralIndex]], optional): 
                List of selected spectral indexes to be used. Defaults to None.
        """
        self.model_type = model_type
        self.model_name = model_name
        self.model_filename = model_filename
        self.model = None  # To be defined in concrete classes
        self.selected_indexes = selected_indexes
        
        # Generate model hash and configuration on initialization
        self.model_hash = None
        self.model_configuration = None
        self._generate_model_hash_and_config()
        
        # Clean up memory after model initialization
        import gc
        gc.collect()
        

    def _generate_model_hash_and_config(self):
        """
        Generate a unique hash and configuration JSON for this model instance.
        This should be called after all model parameters are set.
        """
        # Collect model configuration
        config_dict = {
            "model_type": str(self.model_type),
            "model_name": self.model_name,
            "selected_indexes": [str(idx) for idx in self.selected_indexes] if self.selected_indexes else None,
            # Add common configuration parameters
            "multi_regression": getattr(config, 'MULTI_REGRESSION', None),
            "predicted_attributes": getattr(config, 'PREDICTED_QUALITY_ATTRIBUTES', None),
            "augment_times": getattr(config, 'AUGMENT_TIMES', None),
            "patience": getattr(config, 'PATIENCE', None),
            "epochs": getattr(config, 'EPOCHS', None),
            "batch_size": getattr(config, 'BATCH_SIZE', None),
            "random_state": getattr(config, 'RANDOM_STATE', None),
        }
        
        # Add model-specific configuration if available
        if hasattr(self, 'components'):
            config_dict["components"] = getattr(self, 'components', None)
        if hasattr(self, 'component_dimensions'):
            config_dict["component_dimensions"] = getattr(self, 'component_dimensions', None)
        if hasattr(self, 'selected_bands'):
            config_dict["selected_bands"] = getattr(self, 'selected_bands', None)
        if hasattr(self, 'model_shape'):
            config_dict["model_shape"] = getattr(self, 'model_shape', None)
            
        # Add feature selection parameters only if feature selection is enabled
        if getattr(config, 'USE_FEATURE_SELECTION', False):
            feature_selection_attrs = [
                'USE_FEATURE_SELECTION', 'FEATURE_SELECTION_FULL_SPECTRUM', 'FEATURE_SELECTION_STRATEGY',
                'FEATURE_SELECTION_K_BANDS', 'FEATURE_SELECTION_A_NDSI', 'FEATURE_SELECTION_B_STD', 
                'FEATURE_SELECTION_C_INDEXES', 'FEATURE_SELECTION_GUMBEL_TEMPERATURE', 
                'FEATURE_SELECTION_TEMPERATURE_DECAY', 'FEATURE_SELECTION_MIN_TEMPERATURE',
                'FEATURE_SELECTION_DIVERSITY_WEIGHT', 'FEATURE_SELECTION_ATTENTION_HEADS', 
                'FEATURE_SELECTION_D_MODEL', 'FEATURE_SELECTION_ATTENTION_LAYERS',
                'FEATURE_SELECTION_INDEX_STRATEGY', 'FEATURE_SELECTION_SAVE_VISUALIZATIONS',
                'FEATURE_SELECTION_SAVE_REPORTS'
            ]
            for attr in feature_selection_attrs:
                if hasattr(config, attr):
                    config_dict[attr] = getattr(config, attr, None)
        
        # Add V2 DL model parameters only for V2 models
        model_type_str = str(self.model_type)
        is_v2_model = ('V2' in model_type_str or 'PROGRESSIVE' in model_type_str)
        if is_v2_model:
            v2_attrs = [
                'V2_DL_FUSION_METHOD', 'V2_DL_DOWNSAMPLING_METHOD', 'V2_DL_DOWNSAMPLING_FACTOR',
                'CDAT_V2_FUSION_METHOD', 'CDAT_V2_DOWNSAMPLING_METHOD', 'CDAT_V2_DOWNSAMPLING_FACTOR'
            ]
            for attr in v2_attrs:
                if hasattr(config, attr):
                    config_dict[attr] = getattr(config, attr, None)
        
        # Add CDAT_COMPONENT_ORDER only for CDAT models
        is_cdat_model = 'CDAT' in model_type_str or 'COMPONENT_DRIVEN_ATTENTION_TRANSFORMER' in model_type_str
        if is_cdat_model and hasattr(config, 'CDAT_COMPONENT_ORDER'):
            config_dict['CDAT_COMPONENT_ORDER'] = getattr(config, 'CDAT_COMPONENT_ORDER', None)
        
        # Store configuration as JSON string
        self.model_configuration = json.dumps(config_dict, sort_keys=True, default=str)
        
        # Generate hash from configuration
        config_string = self.model_configuration
        self.model_hash = hashlib.md5(config_string.encode()).hexdigest()[:16]  # Use first 16 characters
        
        # Clean up memory after model hash and config generation
        import gc
        gc.collect()
        

    @abstractmethod
    def load_model(self) -> None:
        """
        Loads the model weights and configurations from a file.
        """
        pass

    @abstractmethod
    def save_model(self) -> None:
        """
        Saves the model weights and configurations to a file.
        """
        pass

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """
        Trains the model on the provided dataset.

        Args:
            X_train (np.ndarray): Training feature matrix.
            Y_train (np.ndarray): Training target matrix.
            X_val (Optional[np.ndarray], optional): Validation feature matrix. Defaults to None.
            Y_val (Optional[np.ndarray], optional): Validation target matrix. Defaults to None.
            **kwargs: Additional keyword arguments for training.
        """
        pass

    @abstractmethod
    def test(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Tests the model on the test dataset.
        """
        pass

    @abstractmethod
    def prepare_data(
        self,
        tomatoes: List,
        selected_bands: List[int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepares the dataset by selecting bands and normalizing.
        """
        pass

    def filter_nan_values(
        self, 
        X: np.ndarray, 
        Y: np.ndarray, 
        axis_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Filters out rows in X and Y where Y has NaN values at the specified axis index.
        
        Args:
            X (np.ndarray): Feature matrix.
            Y (np.ndarray): Target matrix.
            axis_idx (int): Index of the axis in Y to check for NaN values.
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: 
                - X with rows filtered out
                - Y with rows filtered out
                - Boolean mask of valid rows
        """
        # Create mask for non-NaN values at the specified axis
        valid_mask = ~np.isnan(Y[:, axis_idx])
        
        # Apply mask to filter out rows
        X_filtered = X[valid_mask]
        Y_filtered = Y[valid_mask]
        # print how much data was filtered out
        print(f"Filtered out {len(X) - len(X_filtered)} tomatoes")
        return X_filtered, Y_filtered, valid_mask

    def evaluate(
        self,
        Y_true: np.ndarray,
        Y_pred: np.ndarray,
        metrics_to_compute: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluates the model's predictions against the true values.

        Args:
            Y_true (np.ndarray): True target values.
            Y_pred (np.ndarray): Predicted target values.
            metrics_to_compute (Optional[List[str]]): List of metrics to compute.
                Options include 'accuracy', 'r2', 'rmse', 'std'.
                If None, all metrics are computed.

        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for each attribute.
        """
        attributes = config.PREDICTED_QUALITY_ATTRIBUTES  # Ensure alignment with config
        metrics = evaluate_metrics(Y_true, Y_pred, attributes)
        
        # Clean up memory after evaluation
        import gc
        gc.collect()
        
        
        return metrics

    def plot_evaluate(
        self,
        metrics: Dict[str, Dict[str, float]],
        metrics_to_plot: Optional[List[str]] = None
    ) -> None:
        """
        Plots the evaluation metrics.

        Args:
            metrics (Dict[str, Dict[str, float]]): Evaluation metrics for each attribute.
            metrics_to_plot (Optional[List[str]]): List of metrics to plot.
                Options include 'R²', 'RMSE', 'STD'.
                If None, all metrics are plotted.
        """
        plot_evaluation_metrics(metrics, metrics_to_plot)
