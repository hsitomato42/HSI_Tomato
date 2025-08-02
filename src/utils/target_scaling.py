# utils/target_scaling.py

import numpy as np
from typing import Dict, Tuple, List
from sklearn.preprocessing import MinMaxScaler
import src.config as config


class TargetScaler:
    """
    Utility class for scaling and inverse scaling of target variables (quality attributes)
    to handle different value ranges in multi-regression models.
    """
    
    def __init__(self, attributes: List[str] = None):
        """
        Initialize the target scaler.
        
        Args:
            attributes: List of quality attribute names
        """
        self.attributes = attributes or config.PREDICTED_QUALITY_ATTRIBUTES
        self.scalers: Dict[str, MinMaxScaler] = {}
        self.is_fitted = False
        
    def fit(self, Y: np.ndarray) -> 'TargetScaler':
        """
        Fit scalers for each quality attribute.
        
        Args:
            Y: Target matrix of shape (n_samples, n_attributes)
            
        Returns:
            self for method chaining
        """
        if Y.shape[1] != len(self.attributes):
            raise ValueError(f"Y has {Y.shape[1]} columns but {len(self.attributes)} attributes expected")
            
        self.scalers = {}
        
        for idx, attr in enumerate(self.attributes):
            # Get values for this attribute, filtering out NaN
            attr_values = Y[:, idx]
            valid_mask = ~np.isnan(attr_values)
            
            if np.sum(valid_mask) == 0:
                print(f"Warning: No valid values found for attribute '{attr}', skipping scaling")
                continue
                
            valid_values = attr_values[valid_mask].reshape(-1, 1)
            
            # Fit MinMaxScaler for this attribute
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(valid_values)
            self.scalers[attr] = scaler
            
            print(f"Fitted scaler for '{attr}': range [{valid_values.min():.4f}, {valid_values.max():.4f}] -> [0, 1]")
            
        self.is_fitted = True
        return self
        
    def transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Scale target values to [0, 1] range.
        
        Args:
            Y: Target matrix of shape (n_samples, n_attributes)
            
        Returns:
            Scaled target matrix
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before transform")
            
        Y_scaled = Y.copy()
        
        for idx, attr in enumerate(self.attributes):
            if attr not in self.scalers:
                continue
                
            attr_values = Y[:, idx]
            valid_mask = ~np.isnan(attr_values)
            
            if np.sum(valid_mask) > 0:
                # Scale only valid values
                valid_values = attr_values[valid_mask].reshape(-1, 1)
                scaled_values = self.scalers[attr].transform(valid_values).flatten()
                Y_scaled[valid_mask, idx] = scaled_values
                
        return Y_scaled
        
    def inverse_transform(self, Y_scaled: np.ndarray) -> np.ndarray:
        """
        Convert scaled predictions back to original scale.
        
        Args:
            Y_scaled: Scaled target matrix of shape (n_samples, n_attributes)
            
        Returns:
            Target matrix in original scale
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted before inverse_transform")
            
        Y_original = Y_scaled.copy()
        
        for idx, attr in enumerate(self.attributes):
            if attr not in self.scalers:
                continue
                
            attr_values = Y_scaled[:, idx]
            valid_mask = ~np.isnan(attr_values) & (attr_values >= 0) & (attr_values <= 1)
            
            if np.sum(valid_mask) > 0:
                # Inverse scale only valid values
                valid_values = attr_values[valid_mask].reshape(-1, 1)
                original_values = self.scalers[attr].inverse_transform(valid_values).flatten()
                Y_original[valid_mask, idx] = original_values
                
        return Y_original
        
    def fit_transform(self, Y: np.ndarray) -> np.ndarray:
        """
        Fit scalers and transform in one step.
        
        Args:
            Y: Target matrix of shape (n_samples, n_attributes)
            
        Returns:
            Scaled target matrix
        """
        return self.fit(Y).transform(Y)
        
    def get_scale_info(self) -> Dict[str, Dict[str, float]]:
        """
        Get scaling information for each attribute.
        
        Returns:
            Dictionary with min/max values for each attribute
        """
        if not self.is_fitted:
            raise ValueError("Scaler must be fitted first")
            
        scale_info = {}
        for attr, scaler in self.scalers.items():
            scale_info[attr] = {
                'min': scaler.data_min_[0],
                'max': scaler.data_max_[0],
                'range': scaler.data_max_[0] - scaler.data_min_[0]
            }
        return scale_info 