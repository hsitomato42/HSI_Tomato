#!/usr/bin/env python3
"""
Example script: Run DL Models with Feature Selection

This script shows how to run deep learning models with the advanced feature selector.
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
os.chdir(project_root)

# Import config and enums
import src.config as config
from src.config.enums import ModelType

def run_cnn_with_feature_selection():
    """Run CNN with feature selection enabled."""
    print("🎯 RUNNING CNN WITH FEATURE SELECTION")
    
    # Configure for feature selection
    config.USE_FEATURE_SELECTION = True
    config.MODEL_TYPE = ModelType.CNN
    config.MULTI_REGRESSION = True
    config.EPOCHS = 100  # Increase for real training
    
    # Enable all components for feature selection
    config.USE_COMPONENTS = {
        'reflectance': True,
        'std': True,
        'ndsi': True, 
        'indexes': True
    }
    
    # Import and run main training
    from main_dl_algorithms import main
    main()

def run_cdat_with_feature_selection():
    """Run Component-Driven Attention Transformer with feature selection."""
    print("🎯 RUNNING CDAT WITH FEATURE SELECTION")
    
    # Configure for feature selection
    config.USE_FEATURE_SELECTION = True
    config.MODEL_TYPE = ModelType.COMPONENT_DRIVEN_ATTENTION_TRANSFORMER
    config.MULTI_REGRESSION = True
    config.EPOCHS = 100
    
    # Enable all components
    config.USE_COMPONENTS = {
        'reflectance': True,
        'std': True,
        'ndsi': True,
        'indexes': True
    }
    
    # Import and run main training
    from main_dl_algorithms import main
    main()

def run_cnn_transformer_with_feature_selection():
    """Run CNN-Transformer with feature selection."""
    print("🎯 RUNNING CNN-TRANSFORMER WITH FEATURE SELECTION")
    
    # Configure for feature selection
    config.USE_FEATURE_SELECTION = True
    config.MODEL_TYPE = ModelType.CNN_TRANSFORMER
    config.MULTI_REGRESSION = True
    config.EPOCHS = 100
    
    # Enable all components
    config.USE_COMPONENTS = {
        'reflectance': True,
        'std': True,
        'ndsi': True,
        'indexes': True
    }
    
    # Import and run main training
    from main_dl_algorithms import main
    main()

if __name__ == "__main__":
    print("🚀 DL Models with Feature Selection")
    print("="*50)
    print("1. CNN with Feature Selection")
    print("2. CDAT with Feature Selection") 
    print("3. CNN-Transformer with Feature Selection")
    print("="*50)
    
    choice = input("Choose option (1-3): ").strip()
    
    if choice == "1":
        run_cnn_with_feature_selection()
    elif choice == "2":
        run_cdat_with_feature_selection()
    elif choice == "3":
        run_cnn_transformer_with_feature_selection()
    else:
        print("❌ Invalid choice. Run script again and choose 1, 2, or 3.")
