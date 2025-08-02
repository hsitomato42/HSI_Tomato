#!/usr/bin/env python3
"""
Multi-Model Deep Learning Pipeline

This script allows you to run multiple DL models in a pipeline, automatically handling:
- Dataset loading and preparation
- Model training with validation
- Model testing and evaluation  
- Feature importance analysis
- Results comparison across models

Usage:
    python main/multi_dl_pipeline.py

Modify the MODELS_TO_RUN list below to specify which models you want to run.
"""

import time
import numpy as np
import pandas as pd
import src.Models as Models
from src.Models.utils import get_model
import Tomato 
import src.CreateDataset as CreateDataset
import src.ImageData as ImageData
import src.config as config
from sklearn.model_selection import train_test_split
from src.utils.model_utils import get_components, get_component_dimensions
from src.utils.data_processing import DataProcessingUtils
from src.utils.comparing import compare_model_metrics_to_best_results
from src.config.enums import ModelType, ImagePathKind
import os
from datetime import datetime

# =============================================================================
# CONFIGURATION - MODIFY THIS SECTION
# =============================================================================

# List of DL models to run - Add/remove models as needed
MODELS_TO_RUN = [
    # ModelType.CNN,
    ModelType.CNN_TRANSFORMER, 
    ModelType.SPECTRAL_TRANSFORMER,
    ModelType.VIT,
    ModelType.MULTI_HEAD_CNN,
    # ModelType.ADVANCED_MULTI_BRANCH_CNN_TRANSFORMER,
    # ModelType.GLOBAL_BRANCH_FUSION_TRANSFORMER,
]

# Pipeline settings
SKIP_EXISTING_RESULTS = True  # Skip models that already have results
SAVE_COMPARISON_REPORT = True  # Generate comparison report at the end
RESULTS_BASE_DIR = "results/multi_model_pipeline"

# =============================================================================
# PIPELINE FUNCTIONS
# =============================================================================

def setup_pipeline():
    """Setup the pipeline by loading dataset and preparing components."""
    print("=" * 80)
    print("🚀 MULTI-MODEL DL PIPELINE STARTING")
    print("=" * 80)
    
    # Create dataset if needed
    if config.CREATE_DATASET:
        print("📦 Creating dataset...")
        tomatoes_dict = CreateDataset.File_helper.parse_custom_dict(config.TOMATOES_DICT_PATH)
        two_side_flag = config.IMAGES_PATH_KIND == ImagePathKind.TWO_SIDES
        tomatoes = CreateDataset.Dataset.create_dataset(
            tomatoes_dict['tomatoes_dict'],
            two_sides_flag=two_side_flag,
            only_first_harvest=config.ONLY_HARVEST_1
        )
        dataset_path = f"Data/saved_datasets/tomatoes_dataset_side-{config.IMAGES_PATH_KIND.value}_objectsNum-{len(tomatoes)}.pkl"
        DataProcessingUtils.assign_area_to_tomatoes(tomatoes=tomatoes, area_excel_path=config.SPECTRAL_STATS_FILENAME)
        CreateDataset.Dataset.save_dataset(tomatoes, filename=dataset_path)
    
    # Load dataset
    print("📁 Loading dataset...")
    num_of_tomatoes = 872 if config.IMAGES_PATH_KIND == ImagePathKind.TWO_SIDES else 1744
    dataset_path = f"Data/saved_datasets/tomatoes_dataset_side-{config.IMAGES_PATH_KIND.value}_objectsNum-{num_of_tomatoes}.pkl"
    tomatoes = CreateDataset.Dataset.load_dataset(dataset_path)
    print(f"✅ Dataset loaded successfully! ({len(tomatoes)} tomatoes)")
    
    # Setup global variables
    selected_bands = config.PATENT_BANDS
    attributes = config.PREDICTED_QUALITY_ATTRIBUTES
    components = get_components()
    component_dimensions = get_component_dimensions(components, selected_bands)
    
    # Determine maximum 2D dimensions
    config.MAX_2D_DIMENSIONS = config.MAX_2D_DIMENSIONS if hasattr(config, 'MAX_2D_DIMENSIONS') else DataProcessingUtils.get_max_dimensions(tomatoes)
    augment_times = getattr(config, 'AUGMENT_TIMES', 0)
    
    # Calculate model shape
    total_channels = component_dimensions['total']
    height, width = config.MAX_2D_DIMENSIONS
    model_shape = (height, width, total_channels)
    
    print(f"📊 Configuration:")
    print(f"   Selected bands: {selected_bands}")
    print(f"   Quality attributes: {attributes}")
    print(f"   Model shape: {model_shape}")
    print(f"   Components: {components}")
    
    return tomatoes, selected_bands, components, component_dimensions, model_shape

def prepare_data_splits(tomatoes, model_shape):
    """Prepare train/validation/test splits."""
    print("🔀 Preparing data splits...")
    
    random_number = config.RANDOM_STATE
    X_dummy = np.arange(len(tomatoes))
    X_train_idx, X_temp_idx = train_test_split(X_dummy, test_size=0.3, random_state=random_number)
    
    if config.USE_VALIDATION:
        # 70% train, 15% val, 15% test
        X_val_idx, X_test_idx = train_test_split(X_temp_idx, test_size=0.5, random_state=random_number)
        train_tomatoes = [tomatoes[i] for i in X_train_idx]
        val_tomatoes = [tomatoes[i] for i in X_val_idx]
        test_tomatoes = [tomatoes[i] for i in X_test_idx]
        print(f"   Train: {len(train_tomatoes)}, Val: {len(val_tomatoes)}, Test: {len(test_tomatoes)}")
    else:
        # 70% train, 30% test
        test_tomatoes = [tomatoes[i] for i in X_temp_idx]
        train_tomatoes = [tomatoes[i] for i in X_train_idx]
        val_tomatoes = []
        print(f"   Train: {len(train_tomatoes)}, Test: {len(test_tomatoes)}")
    
    return train_tomatoes, val_tomatoes, test_tomatoes

def run_single_model(model_type, train_tomatoes, val_tomatoes, test_tomatoes, selected_bands, components, component_dimensions, model_shape):
    """Run training, testing, and feature importance for a single model."""
    start_time = time.time()
    
    print("\n" + "=" * 60)
    print(f"🤖 RUNNING MODEL: {model_type.value}")
    print("=" * 60)
    
    # Check if results already exist
    results_dir = os.path.join(RESULTS_BASE_DIR, model_type.value)
    if SKIP_EXISTING_RESULTS and os.path.exists(results_dir):
        metrics_file = os.path.join(results_dir, "metrics.json")
        if os.path.exists(metrics_file):
            print(f"⏭️  Skipping {model_type.value} - results already exist")
            return None
    
    try:
        # Initialize model
        print(f"🔧 Initializing {model_type.value} model...")
        model = get_model(
            model_type=model_type,
            image_path_kind=config.IMAGES_PATH_KIND,
            selected_bands=selected_bands,
            selected_indexes=config.INDEXES,
            components=components,
            component_dimensions=component_dimensions,
            model_shape=model_shape,
            predicted_attributes=config.PREDICTED_QUALITY_ATTRIBUTES
        )
        
        # Prepare data
        print(f"📊 Preparing data for {model_type.value}...")
        augment_times = getattr(config, 'AUGMENT_TIMES', 0)
        X_train, Y_train = model.prepare_data(train_tomatoes, selected_bands, augment_times)
        X_test, Y_test = model.prepare_data(test_tomatoes, selected_bands)
        
        X_val, Y_val = None, None
        if val_tomatoes:
            X_val, Y_val = model.prepare_data(val_tomatoes, selected_bands)
        
        print(f"   Train shape: X={X_train.shape}, Y={Y_train.shape}")
        print(f"   Test shape: X={X_test.shape}, Y={Y_test.shape}")
        if X_val is not None:
            print(f"   Val shape: X={X_val.shape}, Y={Y_val.shape}")
        
        # Train model
        print(f"🎯 Training {model_type.value}...")
        model.train(X_train, Y_train, X_val, Y_val, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
        
        # Test model
        print(f"📋 Testing {model_type.value}...")
        metrics = model.test(X_test, Y_test)
        print(f"✅ {model_type.value} testing completed!")
        
        # Feature importance analysis
        importance_results = None
        if config.ANALYZE_FEATURE_IMPORTANCE:
            print(f"🔍 Analyzing feature importance for {model_type.value}...")
            importance_results = model.calculate_feature_importance(
                X_test=X_test,
                Y_test=Y_test,
                methods=["permutation", "gradient"],
                n_repeats=5,  # Fewer repeats for pipeline efficiency
                plot_top_n=None,  # Show all channels in plots
                print_top_n=10,   # Print top 10 in console
                save_results=True,
                save_plots=True
            )
            print(f"✅ Feature importance analysis completed for {model_type.value}!")
        
        # Save results
        os.makedirs(results_dir, exist_ok=True)
        
        # Save metrics
        import json
        metrics_file = os.path.join(results_dir, "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save model info
        model_info = {
            "model_type": model_type.value,
            "model_shape": model_shape,
            "selected_bands": selected_bands,
            "components": components,
            "training_time": time.time() - start_time,
            "timestamp": datetime.now().isoformat()
        }
        info_file = os.path.join(results_dir, "model_info.json")
        with open(info_file, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        print(f"💾 Results saved to: {results_dir}")
        
        # Compare to best results
        compare_model_metrics_to_best_results(
            metrics, selected_bands, model.model_name, model_type, model.model_filename
        )
        
        elapsed_time = time.time() - start_time
        print(f"⏱️  {model_type.value} completed in {elapsed_time:.2f} seconds")
        
        return {
            "model_type": model_type.value,
            "metrics": metrics,
            "training_time": elapsed_time,
            "success": True
        }
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ Error running {model_type.value}: {str(e)}")
        print(f"⏱️  Failed after {elapsed_time:.2f} seconds")
        
        return {
            "model_type": model_type.value,
            "error": str(e),
            "training_time": elapsed_time,
            "success": False
        }

def generate_comparison_report(results):
    """Generate a comparison report across all models."""
    if not SAVE_COMPARISON_REPORT:
        return
    
    print("\n" + "=" * 80)
    print("📊 GENERATING COMPARISON REPORT")
    print("=" * 80)
    
    # Create comparison DataFrame
    comparison_data = []
    
    for result in results:
        if result is None:
            continue
            
        if result["success"]:
            metrics = result["metrics"]
            for attr in config.PREDICTED_QUALITY_ATTRIBUTES:
                if attr in metrics:
                    attr_metrics = metrics[attr]
                    comparison_data.append({
                        "Model": result["model_type"],
                        "Attribute": attr,
                        "R²": attr_metrics.get("R²", None),
                        "RMSE": attr_metrics.get("RMSE", None),
                        "MAE": attr_metrics.get("MAE", None),
                        "Training_Time": result["training_time"]
                    })
        else:
            comparison_data.append({
                "Model": result["model_type"],
                "Attribute": "FAILED",
                "R²": None,
                "RMSE": None,
                "MAE": None,
                "Training_Time": result["training_time"]
            })
    
    if comparison_data:
        # Create comparison DataFrame
        df_comparison = pd.DataFrame(comparison_data)
        
        # Save to CSV
        os.makedirs(RESULTS_BASE_DIR, exist_ok=True)
        comparison_file = os.path.join(RESULTS_BASE_DIR, "model_comparison.csv")
        df_comparison.to_csv(comparison_file, index=False)
        
        # Generate summary statistics
        summary_data = []
        successful_results = [r for r in results if r and r["success"]]
        
        for result in successful_results:
            metrics = result["metrics"]
            avg_r2 = np.mean([metrics[attr]["R²"] for attr in config.PREDICTED_QUALITY_ATTRIBUTES if attr in metrics])
            avg_rmse = np.mean([metrics[attr]["RMSE"] for attr in config.PREDICTED_QUALITY_ATTRIBUTES if attr in metrics])
            
            summary_data.append({
                "Model": result["model_type"],
                "Avg_R²": avg_r2,
                "Avg_RMSE": avg_rmse,
                "Training_Time": result["training_time"]
            })
        
        df_summary = pd.DataFrame(summary_data)
        df_summary = df_summary.sort_values("Avg_R²", ascending=False)
        
        summary_file = os.path.join(RESULTS_BASE_DIR, "model_summary.csv")
        df_summary.to_csv(summary_file, index=False)
        
        print(f"📄 Comparison report saved to: {comparison_file}")
        print(f"📄 Summary report saved to: {summary_file}")
        
        print("\n🏆 MODEL RANKING (by Average R²):")
        print("-" * 50)
        for i, row in df_summary.iterrows():
            print(f"{i+1:2d}. {row['Model']:25s} - R²: {row['Avg_R²']:.4f}, Time: {row['Training_Time']:.1f}s")

def main():
    """Main pipeline function."""
    pipeline_start_time = time.time()
    
    # Setup pipeline
    tomatoes, selected_bands, components, component_dimensions, model_shape = setup_pipeline()
    
    # Prepare data splits
    train_tomatoes, val_tomatoes, test_tomatoes = prepare_data_splits(tomatoes, model_shape)
    
    print(f"\n🎯 PIPELINE CONFIGURATION:")
    print(f"   Models to run: {len(MODELS_TO_RUN)}")
    for model_type in MODELS_TO_RUN:
        print(f"   - {model_type.value}")
    print(f"   Feature importance: {'✅ Enabled' if config.ANALYZE_FEATURE_IMPORTANCE else '❌ Disabled'}")
    print(f"   Results directory: {RESULTS_BASE_DIR}")
    
    # Run each model
    results = []
    for i, model_type in enumerate(MODELS_TO_RUN, 1):
        print(f"\n{'='*80}")
        print(f"🔄 PROGRESS: {i}/{len(MODELS_TO_RUN)} - Running {model_type.value}")
        print(f"{'='*80}")
        
        result = run_single_model(
            model_type, train_tomatoes, val_tomatoes, test_tomatoes,
            selected_bands, components, component_dimensions, model_shape
        )
        results.append(result)
    
    # Generate comparison report
    generate_comparison_report(results)
    
    # Pipeline summary
    pipeline_time = time.time() - pipeline_start_time
    successful_models = sum(1 for r in results if r and r["success"])
    failed_models = len(MODELS_TO_RUN) - successful_models
    
    print("\n" + "=" * 80)
    print("🏁 PIPELINE COMPLETED")
    print("=" * 80)
    print(f"✅ Successful models: {successful_models}")
    print(f"❌ Failed models: {failed_models}")
    print(f"⏱️  Total pipeline time: {pipeline_time:.2f} seconds")
    print(f"📁 Results saved in: {RESULTS_BASE_DIR}")
    
    if config.ANALYZE_FEATURE_IMPORTANCE:
        print(f"🔍 Feature importance results available for each model")
    
    print("🎉 Pipeline execution complete!")

if __name__ == "__main__":
    main() 