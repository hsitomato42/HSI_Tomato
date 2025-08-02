# utils/evaluation.py

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from src.config import dictionaries, enums, config
import os

def evaluate_metrics(Y_true: np.ndarray, Y_pred: np.ndarray, attributes: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Evaluates the model's predictions by calculating R², RRMSE, STD, and MAE for each attribute.
    Filters out any prediction that is NaN and its corresponding true value.

    Args:
        Y_true (np.ndarray): True target values.
        Y_pred (np.ndarray): Predicted target values.
        attributes (List[str]): List of attribute names.

    Returns:
        Dict[str, Dict[str, float]]: Dictionary containing metrics for each attribute.
    """
    metrics = {}
    for i, attr in enumerate(attributes):
        y_true_attr = Y_true[:, i]
        y_pred_attr = Y_pred[:, i]
        
        # Filter out NaN values
        valid_mask = ~np.isnan(y_pred_attr)
        y_true_filtered = y_true_attr[valid_mask]
        y_pred_filtered = y_pred_attr[valid_mask]
        
        # Skip attribute if no valid predictions
        if len(y_true_filtered) == 0:
            metrics[attr] = {'R²': np.nan, 'RRMSE': np.nan, 'STD': np.nan, 'MAE': np.nan}
            continue
        
        # Calculate R²
        r2 = r2_score(y_true_filtered, y_pred_filtered)
        
        # Calculate RRMSE (relative RMSE)
        rmse = np.sqrt(mean_squared_error(y_true_filtered, y_pred_filtered))
        mean_y_true = np.mean(y_true_filtered)
        rrmse = rmse / mean_y_true if mean_y_true != 0 else np.nan

        # Calculate STD of the residuals
        std = np.std(y_true_filtered - y_pred_filtered)
        
        # Calculate MAE (Mean Absolute Error)
        mae = mean_absolute_error(y_true_filtered, y_pred_filtered)

        metrics[attr] = {'R²': r2, 'RRMSE': rrmse, 'STD': std, 'MAE': mae}
    
    return metrics
    ## old code:
    # metrics = {}
    # for i, attr in enumerate(attributes):
    #     r2 = r2_score(Y_true[:, i], Y_pred[:, i])
    #     rmse = np.sqrt(mean_squared_error(Y_true[:, i], Y_pred[:, i]))
    #     rrmse = rmse / np.mean(Y_true[:, i]) # if np.mean(Y_true[:, i]) != 0 else np.nan
    #     std = np.std(Y_true[:, i] - Y_pred[:, i])
    #     metrics[attr] = {'R²': r2, 'RMSE': rrmse, 'STD': std} # notice we use here mean rmse (as the state-of-the-art)

    # return metrics

def plot_evaluation_metrics(metrics: Dict[str, Dict[str, float]], metrics_to_plot: Optional[List[str]] = None) -> None:
    """
    Plots the evaluation metrics.

    Args:
        metrics (Dict[str, Dict[str, float]]): Evaluation metrics for each attribute.
        metrics_to_plot (Optional[List[str]]): List of metrics to plot.
            Options include 'R²', 'RRMSE', 'STD', 'MAE'.
            If None, all metrics are plotted.
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['R²', 'RRMSE', 'STD', 'MAE']

    attributes = list(metrics.keys())
    num_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))

    if num_metrics == 1:
        axes = [axes]

    for ax, metric_name in zip(axes, metrics_to_plot):
        metric_values = [metrics[attr].get(metric_name, np.nan) for attr in attributes]
        ax.bar(attributes, metric_values, color='skyblue')
        ax.set_title(f'{metric_name} by Attribute')
        ax.set_xlabel('Attribute')
        ax.set_ylabel(metric_name)
        ax.set_xticklabels(attributes, rotation=45, ha='right')
        for i, v in enumerate(metric_values):
            if not np.isnan(v):
                ax.text(i, v + 0.01 * max(filter(np.isfinite, metric_values)), f"{v:.2f}", ha='center', va='bottom')

    plt.tight_layout()
    plt.show(block=config.IMAGES_BLOCKER)

