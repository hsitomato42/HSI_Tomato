"""
Centralized path configuration for all output files and directories.
All results, logs, and outputs should be saved under the results directory.
"""

import os
from pathlib import Path

# Base project directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
PROJECT_DATA_ROOT = PROJECT_ROOT / "project_data"
RESULTS_ROOT = PROJECT_DATA_ROOT / "results"

# Data directories
DATA_ROOT = PROJECT_DATA_ROOT / "Data"
DOCS_ROOT = PROJECT_DATA_ROOT / "Docs"

# Ensure directories exist
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
DATA_ROOT.mkdir(parents=True, exist_ok=True)
DOCS_ROOT.mkdir(parents=True, exist_ok=True)

# Main subdirectories
MODELS_DIR = RESULTS_ROOT / "models"
LOGS_DIR = RESULTS_ROOT / "logs"
FEATURE_IMPORTANCE_DIR = RESULTS_ROOT / "feature_importance"
FEATURE_SELECTION_DIR = RESULTS_ROOT / "feature_selection"
VISUALIZATIONS_DIR = RESULTS_ROOT / "visualizations"
DATA_ANALYSIS_DIR = RESULTS_ROOT / "data_analysis"
CSV_OUTPUTS_DIR = RESULTS_ROOT / "csv_outputs"
TEST_RESULTS_DIR = RESULTS_ROOT / "test_results"
CHECKPOINTS_DIR = RESULTS_ROOT / "checkpoints"
EXPERIMENTS_DIR = RESULTS_ROOT / "experiments"

# Create all subdirectories
for directory in [MODELS_DIR, LOGS_DIR, FEATURE_IMPORTANCE_DIR, FEATURE_SELECTION_DIR,
                  VISUALIZATIONS_DIR, DATA_ANALYSIS_DIR, CSV_OUTPUTS_DIR, TEST_RESULTS_DIR,
                  CHECKPOINTS_DIR, EXPERIMENTS_DIR]:
    directory.mkdir(exist_ok=True)

# Specific paths for different outputs
MODEL_SAVE_PATH = str(MODELS_DIR)
LOG_FILE_PATH = str(LOGS_DIR / "hyperspectral.log")
FEATURE_IMPORTANCE_RESULTS_PATH = str(FEATURE_IMPORTANCE_DIR)
FEATURE_SELECTION_RESULTS_PATH = str(FEATURE_SELECTION_DIR)
VISUALIZATION_SAVE_PATH = str(VISUALIZATIONS_DIR)
CSV_RESULTS_PATH = str(CSV_OUTPUTS_DIR / "dl_model_test_results.csv")
COMPARISON_RESULTS_PATH = str(CSV_OUTPUTS_DIR / "model_comparisons.csv")

# Legacy paths (for backward compatibility - will be updated)
LEGACY_MODEL_DIR = "Models/Best Models"

def get_model_save_path(model_type: str, model_name: str) -> str:
    """Get the save path for a specific model."""
    model_dir = MODELS_DIR / model_type / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return str(model_dir)

def get_experiment_path(experiment_name: str) -> str:
    """Get the path for a specific experiment."""
    exp_dir = EXPERIMENTS_DIR / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    return str(exp_dir)

def get_visualization_path(viz_type: str, filename: str) -> str:
    """Get the path for saving visualizations."""
    viz_dir = VISUALIZATIONS_DIR / viz_type
    viz_dir.mkdir(parents=True, exist_ok=True)
    return str(viz_dir / filename)

def get_log_path(log_name: str) -> str:
    """Get the path for a specific log file."""
    return str(LOGS_DIR / log_name)
