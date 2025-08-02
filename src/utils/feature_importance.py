"""
Feature Importance Analysis for ML and DL Models

This module provides comprehensive feature importance analysis capabilities for both
traditional machine learning models and deep learning models in the hyperspectral
quality prediction system.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns  # Optional, not essential for core functionality
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from pathlib import Path
import json
import src.config as config
from src.config.enums import ModelType
import os


class FeatureImportanceAnalyzer:
    """
    Analyzes feature importance for both ML and DL models.
    Supports multiple importance calculation methods and visualization.
    """
    
    def __init__(self, model_instance, model_type: ModelType, feature_names: Optional[List[str]] = None):
        """
        Initialize the feature importance analyzer.
        
        Args:
            model_instance: The trained model instance (ML or DL)
            model_type: Type of the model (ModelType enum)
            feature_names: Optional list of feature names for interpretability
        """
        self.model_instance = model_instance
        self.model_type = model_type
        self.feature_names = feature_names or []
        self.importance_results = {}
        
    def calculate_ml_importance(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        methods: List[str] = ["built_in", "permutation"],
        n_repeats: int = 10,
        random_state: int = 42
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate feature importance for ML models.
        
        Args:
            X_test: Test features
            Y_test: Test targets
            methods: List of importance methods to use
            n_repeats: Number of permutation repeats
            random_state: Random state for reproducibility
            
        Returns:
            Dict containing importance results for each attribute and method
        """
        results = {}
        
        for attr_idx, attr in enumerate(self.model_instance.predicted_attributes):
            results[attr] = {}
            model = self.model_instance.models[attr]
            
            # Filter valid samples for this attribute
            valid_mask = ~np.isnan(Y_test[:, attr_idx])
            X_valid = X_test[valid_mask]
            y_valid = Y_test[valid_mask, attr_idx]
            
            if len(X_valid) == 0:
                print(f"[FeatureImportance] No valid samples for {attr}, skipping")
                continue
            
            # Built-in importance (for tree-based models)
            if "built_in" in methods and hasattr(model, 'feature_importances_'):
                results[attr]["built_in"] = model.feature_importances_
                print(f"[FeatureImportance] Calculated built-in importance for {attr}")
            
            # Permutation importance
            if "permutation" in methods:
                try:
                    perm_importance = permutation_importance(
                        model, X_valid, y_valid,
                        n_repeats=n_repeats,
                        random_state=random_state,
                        scoring='neg_mean_squared_error'
                    )
                    results[attr]["permutation"] = perm_importance.importances_mean
                    results[attr]["permutation_std"] = perm_importance.importances_std
                    print(f"[FeatureImportance] Calculated permutation importance for {attr}")
                except Exception as e:
                    print(f"[FeatureImportance] Error calculating permutation importance for {attr}: {e}")
        
        self.importance_results.update(results)
        return results
    
    def calculate_dl_importance(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        methods: List[str] = ["permutation", "gradient"],
        n_repeats: int = 10,
        random_state: int = 42
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Calculate feature importance for DL models.
        
        Args:
            X_test: Test features
            Y_test: Test targets
            methods: List of importance methods to use
            n_repeats: Number of permutation repeats
            random_state: Random state for reproducibility
            
        Returns:
            Dict containing importance results for each attribute and method
        """
        results = {}
        
        if config.MULTI_REGRESSION:
            results = self._calculate_dl_multi_regression_importance(
                X_test, Y_test, methods, n_repeats, random_state
            )
        else:
            results = self._calculate_dl_single_prediction_importance(
                X_test, Y_test, methods, n_repeats, random_state
            )
        
        self.importance_results.update(results)
        return results
    
    def _calculate_dl_multi_regression_importance(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        methods: List[str],
        n_repeats: int,
        random_state: int
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Calculate importance for multi-regression DL model."""
        results = {}
        model = self.model_instance.model
        
        # Filter out samples with NaN values
        valid_mask = ~np.isnan(Y_test).any(axis=1)
        X_valid = X_test[valid_mask]
        Y_valid = Y_test[valid_mask]
        
        if len(X_valid) == 0:
            print("[FeatureImportance] No valid samples for multi-regression model")
            return results
        
        # Original predictions
        original_pred = model.predict(X_valid, verbose=0)
        
        # For multi-regression, calculate importance per attribute
        for attr_idx, attr in enumerate(self.model_instance.predicted_attributes):
            results[attr] = {}
            y_attr = Y_valid[:, attr_idx]
            pred_attr = original_pred[:, attr_idx]
            
            if "permutation" in methods:
                importance_scores = self._calculate_permutation_importance_dl(
                    model, X_valid, y_attr, pred_attr, attr_idx, n_repeats, random_state
                )
                results[attr]["permutation"] = importance_scores["mean"]
                results[attr]["permutation_std"] = importance_scores["std"]
            
            if "gradient" in methods:
                try:
                    gradient_importance = self._calculate_gradient_importance(
                        model, X_valid, attr_idx
                    )
                    results[attr]["gradient"] = gradient_importance
                except Exception as e:
                    print(f"[FeatureImportance] Error calculating gradient importance for {attr}: {e}")
        
        return results
    
    def _calculate_dl_single_prediction_importance(
        self,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        methods: List[str],
        n_repeats: int,
        random_state: int
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Calculate importance for single prediction DL models."""
        results = {}
        
        for attr_idx, attr in enumerate(self.model_instance.predicted_attributes):
            results[attr] = {}
            model = self.model_instance.models[attr]
            
            # Filter valid samples for this attribute
            valid_mask = ~np.isnan(Y_test[:, attr_idx])
            X_valid = X_test[valid_mask]
            y_valid = Y_test[valid_mask, attr_idx]
            
            if len(X_valid) == 0:
                print(f"[FeatureImportance] No valid samples for {attr}")
                continue
            
            # Original predictions
            original_pred = model.predict(X_valid, verbose=0).flatten()
            
            if "permutation" in methods:
                importance_scores = self._calculate_permutation_importance_dl(
                    model, X_valid, y_valid, original_pred, output_idx=0, 
                    n_repeats=n_repeats, random_state=random_state
                )
                results[attr]["permutation"] = importance_scores["mean"]
                results[attr]["permutation_std"] = importance_scores["std"]
            
            if "gradient" in methods:
                try:
                    gradient_importance = self._calculate_gradient_importance(
                        model, X_valid, output_idx=0
                    )
                    results[attr]["gradient"] = gradient_importance
                except Exception as e:
                    print(f"[FeatureImportance] Error calculating gradient importance for {attr}: {e}")
        
        return results
    
    def _calculate_permutation_importance_dl(
        self,
        model,
        X: np.ndarray,
        y_true: np.ndarray,
        original_pred: np.ndarray,
        output_idx: int,
        n_repeats: int,
        random_state: int
    ) -> Dict[str, np.ndarray]:
        """Calculate permutation importance for DL models."""
        np.random.seed(random_state)
        
        # Original score (negative MSE for consistency with sklearn)
        original_score = -mean_squared_error(y_true, original_pred)
        
        # For DL models, we permute spatial locations rather than individual features
        H, W, C = X.shape[1], X.shape[2], X.shape[3]
        importance_scores = []
        
        # Channel-wise permutation importance
        channel_importance = np.zeros((n_repeats, C))
        
        for repeat in range(n_repeats):
            for channel in range(C):
                X_permuted = X.copy()
                # Permute this channel across all spatial locations
                for sample in range(X.shape[0]):
                    channel_data = X_permuted[sample, :, :, channel].flatten()
                    np.random.shuffle(channel_data)
                    X_permuted[sample, :, :, channel] = channel_data.reshape(H, W)
                
                # Predict with permuted data
                permuted_pred_full = model.predict(X_permuted, verbose=0)
                
                # Extract the correct output for this attribute
                if len(permuted_pred_full.shape) == 2 and permuted_pred_full.shape[1] > 1:  # Multi-output
                    permuted_pred = permuted_pred_full[:, output_idx]
                else:  # Single output
                    permuted_pred = permuted_pred_full.flatten()
                
                # Calculate score decrease
                permuted_score = -mean_squared_error(y_true, permuted_pred)
                channel_importance[repeat, channel] = original_score - permuted_score
        
        return {
            "mean": np.mean(channel_importance, axis=0),
            "std": np.std(channel_importance, axis=0)
        }
    
    def _calculate_gradient_importance(
        self,
        model,
        X: np.ndarray,
        output_idx: int = 0
    ) -> np.ndarray:
        """Calculate gradient-based importance."""
        X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = model(X_tensor, training=False)
            
            if len(predictions.shape) == 2 and predictions.shape[1] > 1:  # Multi-output
                output = predictions[:, output_idx]
            else:  # Single output
                output = tf.squeeze(predictions)
            
            # Use mean of outputs as the scalar to compute gradients
            scalar_output = tf.reduce_mean(output)
        
        gradients = tape.gradient(scalar_output, X_tensor)
        
        # Average absolute gradients across samples and spatial dimensions
        importance = tf.reduce_mean(tf.abs(gradients), axis=[0, 1, 2])
        
        return importance.numpy()
    
    def plot_importance(
        self,
        save_dir: Optional[str] = None,
        top_n: Optional[int] = 20,  # None means show all features
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Plot feature importance results.
        
        Args:
            save_dir: Directory to save plots
            top_n: Number of top features to display (None or -1 for all features)
            figsize: Figure size for plots
        """
        if not self.importance_results:
            print("[FeatureImportance] No importance results to plot. Run calculation first.")
            return
        
        for attr, methods in self.importance_results.items():
            for method_name, importance_values in methods.items():
                if method_name.endswith("_std"):  # Skip std arrays
                    continue
                
                plt.figure(figsize=figsize)
                
                # Get feature names or create generic names
                if self.feature_names and len(self.feature_names) == len(importance_values):
                    feature_names = self.feature_names
                else:
                    if importance_values.ndim == 1 and len(importance_values) > 1:
                        # For channel-wise importance
                        feature_names = [f"Channel_{i}" for i in range(len(importance_values))]
                    else:
                        feature_names = [f"Feature_{i}" for i in range(len(importance_values))]
                
                # Sort features by importance
                sorted_indices = np.argsort(importance_values)[::-1]
                
                # Determine how many features to show
                if top_n is None or top_n == -1:
                    # Show all features
                    n_features_to_show = len(sorted_indices)
                    plot_title_suffix = f"All {n_features_to_show} Features"
                else:
                    # Show top N features
                    n_features_to_show = min(top_n, len(sorted_indices))
                    plot_title_suffix = f"Top {n_features_to_show} Features"
                
                display_indices = sorted_indices[:n_features_to_show]
                sorted_importance = importance_values[display_indices]
                sorted_names = [feature_names[i] for i in display_indices]
                
                # Adjust figure size for many features
                if n_features_to_show > 20:
                    figsize = (max(figsize[0], n_features_to_show * 0.4), figsize[1])
                    plt.figure(figsize=figsize)
                
                # Plot
                bars = plt.bar(range(len(sorted_importance)), sorted_importance)
                plt.xlabel('Features')
                plt.ylabel('Importance Score')
                plt.title(f'{method_name.title()} Feature Importance - {attr} ({plot_title_suffix})')
                plt.xticks(range(len(sorted_names)), sorted_names, rotation=45, ha='right')
                
                # Add error bars if std is available
                std_key = f"{method_name}_std"
                if std_key in methods:
                    std_values = methods[std_key][display_indices]
                    plt.errorbar(range(len(sorted_importance)), sorted_importance, 
                               yerr=std_values, fmt='none', color='black', capsize=3)
                
                plt.tight_layout()
                
                # Save plot if directory specified
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)
                    suffix = "all" if (top_n is None or top_n == -1) else f"top{n_features_to_show}"
                    filename = f"importance_{attr}_{method_name}_{suffix}.png"
                    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
                    print(f"[FeatureImportance] Saved plot: {filename}")
                
                plt.show(block=not config.IMAGES_BLOCKER if hasattr(config, 'IMAGES_BLOCKER') else False)
                plt.close()
    
    def save_importance_results(self, save_path: str) -> None:
        """
        Save importance results to JSON file.
        
        Args:
            save_path: Path to save the results
        """
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {}
        for attr, methods in self.importance_results.items():
            serializable_results[attr] = {}
            for method, values in methods.items():
                if isinstance(values, np.ndarray):
                    serializable_results[attr][method] = values.tolist()
                else:
                    serializable_results[attr][method] = values
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save to JSON
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"[FeatureImportance] Saved importance results to {save_path}")
    
    def get_top_features(self, method: str = "permutation", top_n: Optional[int] = 10) -> Dict[str, List[Tuple[int, float]]]:
        """
        Get top N most important features for each attribute.
        
        Args:
            method: Importance method to use
            top_n: Number of top features to return (None or -1 for all features)
            
        Returns:
            Dict with attribute names and their top features (index, importance) tuples
        """
        top_features = {}
        
        for attr, methods in self.importance_results.items():
            if method in methods:
                importance_values = methods[method]
                sorted_indices = np.argsort(importance_values)[::-1]
                
                # Determine how many features to return
                if top_n is None or top_n == -1:
                    # Return all features
                    n_features_to_return = len(sorted_indices)
                else:
                    # Return top N features
                    n_features_to_return = min(top_n, len(sorted_indices))
                
                selected_indices = sorted_indices[:n_features_to_return]
                top_features[attr] = [
                    (int(idx), float(importance_values[idx])) 
                    for idx in selected_indices
                ]
        
        return top_features
    
    def create_importance_summary(self) -> pd.DataFrame:
        """
        Create a summary DataFrame of feature importance results.
        
        Returns:
            DataFrame with importance summary sorted by attribute and importance value
        """
        summary_data = []
        
        for attr, methods in self.importance_results.items():
            for method, values in methods.items():
                if not method.endswith("_std"):
                    if isinstance(values, np.ndarray) and values.ndim == 1:
                        for i, importance in enumerate(values):
                            # Ensure proper feature naming - avoid Feature_x when possible
                            if i < len(self.feature_names) and self.feature_names[i]:
                                feature_name = self.feature_names[i]
                            else:
                                # Generate a more meaningful generic name based on model type
                                if self.model_type.value in ['RANDOM_FOREST', 'XGBOOST']:
                                    feature_name = f'Feature_{i}'  # ML models
                                else:
                                    feature_name = f'Channel_{i}'  # DL models
                            
                            summary_data.append({
                                'Attribute': attr,
                                'Method': method,
                                'Feature_Index': i,
                                'Feature_Name': feature_name,
                                'Importance': float(importance)
                            })
        
        # Create DataFrame and sort by Attribute and then by Importance (descending)
        df = pd.DataFrame(summary_data)
        if not df.empty:
            df = df.sort_values(['Attribute', 'Importance'], ascending=[True, False])
            df = df.reset_index(drop=True)  # Reset index after sorting
        
        return df


def generate_feature_names(
    selected_bands: List[int],
    selected_indexes: Optional[List] = None,
    ndsi_band_pairs: Optional[List[Tuple[int, int]]] = None,
    additional_features: Optional[List] = None
) -> List[str]:
    """
    Generate meaningful feature names for ML models.
    
    Args:
        selected_bands: List of selected spectral bands
        selected_indexes: List of selected spectral indexes
        ndsi_band_pairs: List of NDSI band pairs
        additional_features: List of additional features (AREA, MEAN, etc.)
        
    Returns:
        List of feature names
    """
    feature_names = []
    
    # NOTE: The order must match exactly with BaseMLModel.prepare_data()
    # ML models do NOT use reflectance features directly, only computed features
    
    # 1. STD features (if enabled in ADDITIONAL_FEATURES)
    if hasattr(config, 'ADDITIONAL_FEATURES') and config.ADDITIONAL_FEATURES:
        from config.enums import AdditionalFeature
        if AdditionalFeature.STD in config.ADDITIONAL_FEATURES:
            for band in selected_bands:
                feature_names.append(f"STD_Band_{band}")
    
    # 2. NDSI features (always computed in ML models)
    if ndsi_band_pairs:
        for band_i, band_j in ndsi_band_pairs:
            feature_names.append(f"NDSI_{band_i}_{band_j}")
    
    # 3. Additional features (AREA, MEAN, MEDIAN) in the order they appear in prepare_data
    if hasattr(config, 'ADDITIONAL_FEATURES') and config.ADDITIONAL_FEATURES:
        from config.enums import AdditionalFeature
        
        # AREA feature
        if AdditionalFeature.AREA in config.ADDITIONAL_FEATURES:
            feature_names.append("Additional_AREA")
        
        # MEAN features (one per band)
        if AdditionalFeature.MEAN in config.ADDITIONAL_FEATURES:
            for band in selected_bands:
                feature_names.append(f"Mean_Band_{band}")
        
        # MEDIAN features (one per band)  
        if AdditionalFeature.MEDIAN in config.ADDITIONAL_FEATURES:
            for band in selected_bands:
                feature_names.append(f"Median_Band_{band}")
    
    # 4. Spectral index features (if enabled)
    if config.USE_COMPONENTS.get('indexes', False) and selected_indexes:
        for index in selected_indexes:
            if hasattr(index, 'value'):
                # Extract just the index name from the tuple
                index_name = index.value[0] if isinstance(index.value, tuple) else str(index.value)
                feature_names.append(f"Index_{index_name}")
            else:
                feature_names.append(f"Index_{index}")
    
    return feature_names 