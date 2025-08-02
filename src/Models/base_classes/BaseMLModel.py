from .BaseModel import BaseModel
from src.Tomato import Tomato
from typing import Tuple, Optional, List, Dict
import numpy as np
import os
import src.config as config
from src.config.enums import AdditionalFeature, ModelType, SpectralIndex
from src.utils.data_processing import DataProcessingUtils
from src.utils.spectral_indexes import SpectralIndexCalculator
from src.utils.feature_importance import FeatureImportanceAnalyzer, generate_feature_names
import src.utils as utils


class BaseMLModel(BaseModel):
    """
    Base class for traditional machine learning models that provides common functionality.
    This serves as an intermediate layer between BaseModel and specific ML implementations
    like RandomForest and XGBoost.
    """

    def __init__(
        self,
        model_type: ModelType,
        model_name: str,
        model_filename: str,
        selected_indexes: Optional[List[SpectralIndex]] = None,
        predicted_attributes: List[str] = config.PREDICTED_QUALITY_ATTRIBUTES
    ):
        """
        Initializes the BaseMLModel with common parameters.

        Args:
            model_type (ModelType): The type of the model.
            model_name (str): The name of the model.
            model_filename (str): The filename for saving/loading the model.
            selected_indexes (Optional[List[SpectralIndex]]): List of selected spectral indexes.
            predicted_attributes (List[str]): List of quality attributes to predict.
        """
        super().__init__(
            model_type=model_type,
            model_name=model_name,
            model_filename=model_filename,
            selected_indexes=selected_indexes
        )
        self.predicted_attributes = predicted_attributes
        self.models = {}  # Dictionary to hold a model for each quality attribute

    def prepare_data(
        self,
        tomatoes: List[Tomato],
        selected_bands: List[int],
        augment_times: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare data by extracting features and targets from tomato objects.

        Args:
            tomatoes (List[Tomato]): List of Tomato objects.
            selected_bands (List[int]): List of selected band indices.
            augment_times (int): Number of times to augment the data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Feature matrix X and target matrix Y.
        """
        X_features = []
        Y_targets = []

        for tomato in tomatoes:
            # Extract reflectance_matrix
            reflectance_matrix = tomato.spectral_stats.reflectance_matrix
            mean_reflectance = DataProcessingUtils.compute_mean_reflectance(reflectance_matrix, selected_bands)

            # Initialize feature vector for the current tomato
            feature_vector = []
            if hasattr(config, 'ADDITIONAL_FEATURES'):
                 if AdditionalFeature.STD in config.ADDITIONAL_FEATURES:
                    feature_vector.extend([tomato.statistic_stats.std[i] for i in selected_bands])

            ndsi_values = DataProcessingUtils.compute_ndsi(mean_reflectance, selected_bands)  # Shape: (number_of_band_pairs,)
            feature_vector.extend(ndsi_values)
            

            if hasattr(config, 'ADDITIONAL_FEATURES'):
                if AdditionalFeature.AREA in config.ADDITIONAL_FEATURES:
                    area = tomato.spectral_stats.area
                    feature_vector.append(area)
                if AdditionalFeature.MEAN in config.ADDITIONAL_FEATURES:
                    feature_vector.extend(tomato.statistic_stats.mean)
                if AdditionalFeature.MEDIAN in config.ADDITIONAL_FEATURES:
                    feature_vector.extend(tomato.statistic_stats.median)
               

            if config.USE_COMPONENTS.get('indexes', False):
                # Compute selected spectral indexes
                spectral_indexes = SpectralIndexCalculator.calculate_selected_indexes(
                    reflectance_matrix=reflectance_matrix,  # Pass the full reflectance_matrix
                    selected_indexes=self.selected_indexes,
                )
                # Append spectral index values
                for index_shortcut, index_value in spectral_indexes.items():
                    feature_vector.extend(index_value.flatten())

            # Concatenate all features into a single array
            combined_features = np.array(feature_vector)

            X_features.append(combined_features)

            # Extract Y based on config
            quality_assess = tomato.quality_assess
            y = [getattr(quality_assess, attr, np.nan) for attr in config.PREDICTED_QUALITY_ATTRIBUTES]
            Y_targets.append(y)

        X = np.array(X_features)  # Shape: (num_samples, num_features)
        Y = np.array(Y_targets, dtype=float)  # Shape: (num_samples, num_quality_attributes)

        print(f'The X shape is: {X.shape}')
        # Fill missing values in Y with the mean of each quality attribute
        Y_filled = DataProcessingUtils.fill_missing_values(Y, config.PREDICTED_QUALITY_ATTRIBUTES)
        Y = Y_filled

        return X, Y

    def train(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        Y_val: Optional[np.ndarray] = None,
        **kwargs
    ) -> None:
        """
        Common implementation of the train method for ML models.
        Handles filtering of NaN values and training for each attribute.
        
        Args:
            X_train (np.ndarray): Training feature matrix.
            Y_train (np.ndarray): Training target matrix.
            X_val (Optional[np.ndarray], optional): Validation feature matrix. Defaults to None.
            Y_val (Optional[np.ndarray], optional): Validation target matrix. Defaults to None.
            **kwargs: Additional keyword arguments for training.
        """
        for idx, attr in enumerate(self.predicted_attributes):
            print(f"[{self.__class__.__name__}] Training model for '{attr}'...")
            
            # Filter out samples where the target value is NaN using the base method
            X_train_filtered, Y_train_filtered, _ = self.filter_nan_values(
                X_train, Y_train, idx)

            # Train the model
            self.models[attr].fit(X_train_filtered, Y_train_filtered[:, idx])
            print(f"[{self.__class__.__name__}] Training completed for '{attr}'.")

    def test(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Common implementation of the test method for ML models.
        Handles filtering of NaN values and prediction for each attribute.
        
        Args:
            X_test (np.ndarray): Test feature matrix.
            Y_test (np.ndarray): Test target matrix.
            
        Returns:
            Dict[str, Dict[str, float]]: Evaluation metrics for each attribute.
        """
        # Initialize an array filled with NaN values
        Y_pred = np.full_like(Y_test, np.nan)
        
        for idx, attr in enumerate(self.predicted_attributes):
            print(f"[{self.__class__.__name__}] Predicting '{attr}'...")
            
            # Filter out samples where the target value is NaN using the base method
            X_test_filtered, _, valid_mask = self.filter_nan_values(
                X_test, Y_test, idx
            )
            
            # Make predictions only for valid samples
            predictions = self.models[attr].predict(X_test_filtered)
            Y_pred[valid_mask, idx] = predictions
            print(f"[{self.__class__.__name__}] Made predictions for {X_test_filtered.shape[0]} samples for '{attr}'")
        
        # Evaluate metrics
        metrics = self.evaluate(Y_test, Y_pred)
        return metrics

    def save_model(self) -> None:
        pass

    def load_model(self) -> None:
        pass

    def _get_model_path_for_attr(self, attr: str) -> str:
        """
        Generate the file path for saving/loading a model for a specific attribute.
        
        Args:
            attr (str): The quality attribute name.
            
        Returns:
            str: The file path for the model.
        """
        # Create the model directory if it doesn't exist
        model_dir = os.path.join(config.BASE_MODEL_DIR, self.model_type.value)
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate the model filename
        model_filename = f"{self.model_name}_{attr}.joblib"
        model_path = os.path.join(model_dir, model_filename)
        
        return model_path

    def calculate_feature_importance(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        methods: List[str] = ["built_in", "permutation"],
        n_repeats: int = 10,  # Number of permutation repetitions for stable estimates
        random_state: int = 42,
        save_results: bool = True,
        save_plots: bool = True,
        plot_top_n: Optional[int] = 20,  # None or -1 to show all features
        print_top_n: int = 5,  # Number of top features to print in console
        results_dir: Optional[str] = None
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate feature importance for ML models.
        
        Args:
            X_test: Test feature matrix
            Y_test: Test target matrix  
            methods: List of importance methods to use
            n_repeats: Number of permutation repetitions for stable estimates (NOT number of features)
            random_state: Random state for reproducibility
            save_results: Whether to save results to file
            save_plots: Whether to save importance plots
            plot_top_n: Number of top features to plot (None or -1 for all features)
            print_top_n: Number of top features to print in console
            results_dir: Directory to save results (auto-generated if None)
            
        Returns:
            Dict containing importance results for each attribute and method
        """
        print(f"\n[{self.__class__.__name__}] Calculating feature importance...")
        print(f"[{self.__class__.__name__}] n_repeats={n_repeats} (permutation repetitions for stability)")
        
        # Use config.PATENT_BANDS if selected_bands is not available
        selected_bands = getattr(self, 'selected_bands', []) or config.PATENT_BANDS
        
        # Generate feature names for interpretability
        feature_names = generate_feature_names(
            selected_bands=selected_bands,
            selected_indexes=self.selected_indexes,
            ndsi_band_pairs=getattr(config, 'NDSI_BAND_PAIRS', []),
            additional_features=getattr(config, 'ADDITIONAL_FEATURES', [])
        )
        
        # Initialize feature importance analyzer
        analyzer = FeatureImportanceAnalyzer(
            model_instance=self,
            model_type=self.model_type,
            feature_names=feature_names
        )
        
        # Calculate importance
        importance_results = analyzer.calculate_ml_importance(
            X_test=X_test,
            Y_test=Y_test,
            methods=methods,
            n_repeats=n_repeats,
            random_state=random_state
        )
        
        # Set up results directory
        if results_dir is None:
            results_dir = os.path.join("results", "feature_importance", self.__class__.__name__)
        
        # Save results and plots
        if save_results or save_plots:
            os.makedirs(results_dir, exist_ok=True)
            
        if save_results:
            results_path = os.path.join(results_dir, "importance_results.json")
            analyzer.save_importance_results(results_path)
            
            # Save summary DataFrame
            summary_df = analyzer.create_importance_summary()
            summary_path = os.path.join(results_dir, "importance_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            print(f"[{self.__class__.__name__}] Saved importance summary to {summary_path}")
        
        if save_plots:
            analyzer.plot_importance(save_dir=results_dir, top_n=plot_top_n)
        
        # Print top features
        self._print_top_features(analyzer, importance_results, print_top_n)
        
        return importance_results
    
    def _print_top_features(
        self, 
        analyzer: FeatureImportanceAnalyzer, 
        importance_results: Dict[str, Dict[str, np.ndarray]],
        print_top_n: int = 5
    ) -> None:
        """Print top features for each attribute and method."""
        print(f"\n[{self.__class__.__name__}] Top {print_top_n} Most Important Features:")
        print("=" * 70)
        
        for method in ["built_in", "permutation"]:
            top_features = analyzer.get_top_features(method=method, top_n=print_top_n)
            if top_features:
                print(f"\n{method.upper()} IMPORTANCE:")
                print("-" * 30)
                
                for attr, features in top_features.items():
                    print(f"\n{attr}:")
                    for rank, (feature_idx, importance) in enumerate(features, 1):
                        feature_name = analyzer.feature_names[feature_idx] if feature_idx < len(analyzer.feature_names) else f"Feature_{feature_idx}"
                        print(f"  {rank}. {feature_name}: {importance:.4f}")
        
        print("\n" + "=" * 70)
