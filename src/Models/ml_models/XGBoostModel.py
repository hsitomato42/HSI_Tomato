# models/XGBoostModel.py

from ..base_classes.BaseMLModel import BaseMLModel
from src.Tomato import Tomato

from typing import Tuple, Optional, List, Dict
import numpy as np
import xgboost as xgb
import os
from src.utils.evaluation import evaluate_metrics, plot_evaluation_metrics
from src.utils.data_processing import DataProcessingUtils
from src.utils.spectral_indexes import SpectralIndexCalculator
import src.config as config
from src.config.enums import AdditionalFeature, ModelType, SpectralIndex, ImagePathKind
import src.utils as utils


class XGBoostModel(BaseMLModel):
    def __init__(self, model_name: str, images_path_kind: ImagePathKind, selected_indexes: List[SpectralIndex]):
        """
        Initializes the XGBoostModel.

        Args:
            model_name (str): Unique model name incorporating type, image path kind, and spectral indexes.
            images_path_kind (ImagePathKind): 'oneSide' or 'twoSides'.
            selected_indexes (List[SpectralIndex]): List of selected spectral indexes to use.
        """
        super().__init__(
            model_type=ModelType.XGBOOST,
            model_name=model_name, 
            model_filename='Models/Best Models/XGBoost/temp',
            selected_indexes=selected_indexes
        )
        self.images_path_kind = images_path_kind.value
        self.model_dir = os.path.join(config.BASE_MODEL_DIR, config.MODEL_TYPE.value)
        os.makedirs(self.model_dir, exist_ok=True)

        # Initialize a model for each quality attribute
        for attr in config.PREDICTED_QUALITY_ATTRIBUTES:
            self.models[attr] = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                random_state=42,
                learning_rate=0.1
            )

    def load_model(self) -> None:
        """
        Loads the XGBoost models from the specified directory.
        """
        for attr in config.PREDICTED_QUALITY_ATTRIBUTES:
            model_path = os.path.join(self.model_dir, f"{self.model_name}_{self.images_path_kind}_{attr}.json")
            if os.path.exists(model_path):
                self.models[attr].load_model(model_path)
                print(f"Model for '{attr}' loaded from {model_path}")
            else:
                print(f"Model file not found for '{attr}' at {model_path}")

    def save_model(self) -> None:
        """
        Saves the XGBoost models to the specified directory.
        """
        for attr in config.PREDICTED_QUALITY_ATTRIBUTES:
            model_path = os.path.join(self.model_dir, f"{self.model_name}_{self.images_path_kind}_{attr}.json")
            self.models[attr].save_model(model_path)
            print(f"Model for '{attr}' saved to {model_path}")

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        **kwargs
    ) -> None:
        """
        Trains separate XGBoost models for each quality attribute.

        Args:
            X_train (np.ndarray): Training feature matrix.
            Y_train (np.ndarray): Training target matrix.
            X_val (Optional[np.ndarray], optional): Validation feature matrix. Defaults to None.
            Y_val (Optional[np.ndarray], optional): Validation target matrix. Defaults to None.
            epochs (int, optional): Number of training epochs. Defaults to 100.
            batch_size (int, optional): Size of training batches. Defaults to 32.
            **kwargs: Additional keyword arguments for training.
        """
        super().train(X_train, Y_train, X_val, Y_val, **kwargs)

    def test(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Tests the trained models on the test data and evaluates their performance.
        Delegates to the BaseMLModel implementation.
        
        Args:
            X_test (np.ndarray): Test feature matrix.
            Y_test (np.ndarray): Test target matrix.

        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for each attribute.
        """
        return super().test(X_test, Y_test)


