from src.config.enums import AdditionalFeature
from src.utils.data_processing import DataProcessingUtils
from src.utils.evaluation import evaluate_metrics
from src.utils.spectral_indexes import SpectralIndexCalculator
from ..base_classes.BaseMLModel import BaseMLModel
from src.Tomato import Tomato
from typing import Tuple, Optional, List, Dict
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os
import src.config as config
import src.utils as utils

class RandomForestModel(BaseMLModel):
    def __init__(
        self,
        model_name: str,
        model_filename: str,
        selected_indexes: Optional[List],
        predicted_attributes: List[str] = config.PREDICTED_QUALITY_ATTRIBUTES
    ) -> None:
        super().__init__(
            model_type=config.ModelType.RANDOM_FOREST,
            model_name=model_name,
            model_filename=model_filename,
            selected_indexes=selected_indexes,
            predicted_attributes=predicted_attributes
        )

        # Initialize a model for each quality attribute
        for attr in self.predicted_attributes:
            self.models[attr] = RandomForestRegressor(
                n_estimators=100,
                random_state=config.RANDOM_STATE
            )
            print(f"[RandomForestModel] Initialized RandomForestRegressor for '{attr}'.")

    
    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """
        Train RandomForest models for each quality attribute.
        Delegates to the BaseMLModel implementation.
        """
        super().train(X_train, Y_train, X_val, Y_val, **kwargs)

    def test(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Test RandomForest models for each quality attribute.
        Delegates to the BaseMLModel implementation.
        """
        return super().test(X_test, Y_test)

    def save_model(self) -> None:
        """
        Save each RandomForest model individually using joblib.
        """
        import joblib

        if not config.SAVE_MODEL:
            print("[RandomForestModel] config.SAVE_MODEL is False. Skipping save_model().")
            return

        for attr in self.predicted_attributes:
            model_path = self._get_model_path_for_attr(attr)
            joblib.dump(self.models[attr], model_path)
            print(f"[RandomForestModel] Saved model for '{attr}' to {model_path}")

    def load_model(self) -> None:
        """
        Load each RandomForest model individually using joblib.
        """
        import joblib

        for attr in self.predicted_attributes:
            model_path = self._get_model_path_for_attr(attr)
            if os.path.exists(model_path):
                self.models[attr] = joblib.load(model_path)
                print(f"[RandomForestModel] Loaded model for '{attr}' from {model_path}.")
            else:
                print(f"[RandomForestModel] No pre-trained model found for '{attr}' at {model_path}.")
