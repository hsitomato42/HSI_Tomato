# main.py


import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

# Import GPU configuration to fix CUDA issues
from src.utils.gpu_config import setup_gpu_environment

# Import memory profiler
from src.utils.memory_profiler import MemoryProfiler

import time
import numpy as np
# import Models
from src.Models.utils.model_factory import get_model
import src.CreateDataset as CreateDataset
import src.config as config
from sklearn.model_selection import train_test_split
from src.utils.model_utils import get_components, get_component_dimensions
from src.utils.data_processing import DataProcessingUtils
from src.config.enums import *
from src.utils.comparing import compare_model_metrics_to_best_results

def main():
    # Setup GPU environment to prevent CUDA issues
    setup_gpu_environment()
    
    # Initialize memory profiler
    profiler = MemoryProfiler()
    profiler.snapshot("Start of main()")
    
    if config.CREATE_DATASET:
        # # Step 1: create the tomatoes dictionary
        tomatoes_dict = CreateDataset.File_helper.parse_custom_dict(config.TOMATOES_DICT_PATH)

        # # Step 2: Create the dataset
        two_side_flag = config.IMAGES_PATH_KIND == ImagePathKind.TWO_SIDES
        tomatoes = CreateDataset.Dataset.create_dataset(
            tomatoes_dict['tomatoes_dict'],
            two_sides_flag=two_side_flag,
            only_first_harvest=config.ONLY_HARVEST_1
        )
        dataset_path = f"project_data/Data/saved_datasets/tomatoes_dataset_side-{config.IMAGES_PATH_KIND.value}_objectsNum-{len(tomatoes)}.pkl"
        # Assign area to each tomato (part of dataset creation)
        # DataProcessingUtils.assign_area_to_tomatoes( tomatoes=tomatoes, area_excel_path=config.SPECTRAL_STATS_FILENAME )
        
        # Step 3: Save the dataset
        CreateDataset.Dataset.save_dataset(tomatoes, filename=dataset_path)


    # # Don't comment if step 2 is commented (if dataset is not created)
    num_of_tomatoes = 872 if config.IMAGES_PATH_KIND == ImagePathKind.TWO_SIDES else 1744
    dataset_path = f"project_data/Data/saved_datasets/tomatoes_dataset_side-{config.IMAGES_PATH_KIND.value}_objectsNum-{num_of_tomatoes}.pkl"

    # # Step 4: Load the dataset
    tomatoes = CreateDataset.Dataset.load_dataset(dataset_path)
    print('Dataset loaded successfully!')
    profiler.snapshot("After loading dataset")
    
    # Apply band cutting if configured
    if hasattr(config, 'CUT_BANDS_FROM_START') and config.CUT_BANDS_FROM_START > 0:
        DataProcessingUtils.apply_band_cutting_to_dataset(tomatoes, config.CUT_BANDS_FROM_START)
    
    # Clean up memory after dataset loading and processing
    import gc
    gc.collect()
    

    # 🎯 PROGRESSIVE FEATURE SELECTION LOGIC
    if getattr(config, 'USE_FEATURE_SELECTION', False):
        print("\n🎯 PROGRESSIVE FEATURE SELECTION ENABLED")
        print(f"Stages: {config.FEATURE_SELECTION_STAGE_RUN_ORDER}")
        # Run progressive feature selection training
        run_progressive_feature_selection_training(tomatoes)
    else:
        # Standard single run without feature selection
        run_standard_training(tomatoes)
    
    # Clean up main tomatoes list after use
    del tomatoes
    import gc
    gc.collect()
    
    
    # Clean up memory after main function completes
    import gc
    gc.collect()
    
    # Final memory report
    profiler.snapshot("End of main()")
    profiler.report()
    

def run_progressive_feature_selection_training(tomatoes):
    """Run progressive feature selection training with all stages."""
    print("\n=== PROGRESSIVE FEATURE SELECTION TRAINING ===")
    
    # Initialize memory profiler for this function
    profiler = MemoryProfiler()
    profiler.snapshot("Start of progressive feature selection")
    
    selected_bands = None  # Use full spectrum for feature selection
    attributes = config.PREDICTED_QUALITY_ATTRIBUTES
    
    # Use utility functions to get components and component dimensions
    components = get_components()
    component_dimensions = get_component_dimensions(components, selected_bands)
    
    # Determine maximum 2D dimensions and set in config
    config.MAX_2D_DIMENSIONS = config.MAX_2D_DIMENSIONS if hasattr(config, 'MAX_2D_DIMENSIONS') else DataProcessingUtils.get_max_dimensions(tomatoes)
    augment_times = getattr(config, 'AUGMENT_TIMES', 0) # Number of augmented samples per tomato
    
    # Calculate total channels from component dimensions
    total_channels = component_dimensions['total']
    
    height, width = config.MAX_2D_DIMENSIONS
    model_shape = (height, width, total_channels)

    # Initialize the desired model using the factory models
    model = get_model(
        model_type=config.MODEL_TYPE,
        image_path_kind=config.IMAGES_PATH_KIND,
        selected_bands=selected_bands,
        selected_indexes=config.INDEXES,
        components=components,
        component_dimensions=component_dimensions,  # Pass the component dimensions
        model_shape=model_shape,
        predicted_attributes=config.PREDICTED_QUALITY_ATTRIBUTES
    )
    
    # Prepare data splits
    random_number = config.RANDOM_STATE
    X_val, Y_val = None, None
    X_dummy = np.arange(len(tomatoes))  # just indices to split
    X_train_idx, X_temp_idx = train_test_split(X_dummy, test_size=0.3, random_state=random_number)
    
    if config.USE_VALIDATION:
        # 70% train, 15% val, 15% test
        X_val_idx, X_test_idx = train_test_split(X_temp_idx, test_size=0.5, random_state=random_number)
        train_tomatoes = [tomatoes[i] for i in X_train_idx]
        val_tomatoes = [tomatoes[i] for i in X_val_idx]
        test_tomatoes = [tomatoes[i] for i in X_test_idx]
        
        # Clean up memory after data splits created
        del X_train_idx, X_val_idx, X_test_idx, X_temp_idx
        import gc
        gc.collect()
        
        X_train, Y_train = model.prepare_data(train_tomatoes, selected_bands, augment_times)
        del train_tomatoes  # Free train tomatoes after data preparation
        gc.collect()
        profiler.snapshot("After preparing training data")
        
        X_val, Y_val = model.prepare_data(val_tomatoes, selected_bands)
        del val_tomatoes  # Free val tomatoes after data preparation
        gc.collect()
        
        X_test, Y_test = model.prepare_data(test_tomatoes, selected_bands)
        del test_tomatoes  # Free test tomatoes after data preparation
        gc.collect()
        profiler.snapshot("After preparing all data")
        
    else:
        # 70% train, 30% test
        X_dummy = np.arange(len(tomatoes))
        X_train_idx, X_test_idx = train_test_split(X_dummy, test_size=0.3, random_state=random_number)
        train_tomatoes = [tomatoes[i] for i in X_train_idx]
        val_tomatoes = []
        test_tomatoes = [tomatoes[i] for i in X_test_idx]
        
        # Clean up memory after data splits created
        del X_train_idx, X_test_idx, X_dummy
        gc.collect()
        
        X_train, Y_train = model.prepare_data(train_tomatoes, selected_bands, augment_times)
        del train_tomatoes  # Free train tomatoes after data preparation
        gc.collect()
        
        X_test, Y_test = model.prepare_data(test_tomatoes, selected_bands)
        del test_tomatoes  # Free test tomatoes after data preparation
        gc.collect()
        
    
    # Progressive feature selection training
    print("🔍 Training with progressive feature selection...")
    profiler.snapshot("Before training")
    model.train(X_train, Y_train, X_val, Y_val, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
    profiler.snapshot("After training")
    
    # Clean up memory after training
    del X_train, Y_train
    if X_val is not None:
        del X_val, Y_val
    import gc
    gc.collect()
    
    
    # Model test
    metrics = model.test(X_test, Y_test)
    print("Evaluation Metrics:", metrics)
    
    # Clean up memory after testing
    del X_test, Y_test
    gc.collect()
    
    
    # Step: Feature Importance Analysis (if enabled)
    if config.ANALYZE_FEATURE_IMPORTANCE:
        print("\n" + "="*60)
        print("ANALYZING CHANNEL IMPORTANCE")
        print("="*60)
        
        # Analyze channel importance for DL models
        importance_results = model.calculate_feature_importance(
            X_test=X_test,
            Y_test=Y_test,
            methods=["permutation", "gradient"],  # Both methods for DL models
            n_repeats=5,         # Fewer repeats for DL (computationally expensive)
            plot_top_n=None,     # Show ALL channels in plots
            print_top_n=10,      # Print top 10 in console
            save_results=True,   # Save JSON and CSV results
            save_plots=True      # Generate and save plots
        )
        
        print(f"\n✅ Channel importance analysis completed!")
        print(f"📁 Results saved in: results/feature_importance/{model.__class__.__name__}/")
    
    # Compare model metrics to XGBoost metrics_dict
    compare_model_metrics_to_best_results(metrics, selected_bands or config.PATENT_BANDS, model.model_name, model.model_type, model.model_filename )
    
    # Save results to CSV file
    from src.utils.comparing import save_dl_results_to_csv
    save_dl_results_to_csv(metrics, selected_bands or config.PATENT_BANDS, model.model_name, model.model_type)

    print("✅ Progressive feature selection training complete!")
    
    # Clean up memory after function completion
    import gc
    gc.collect()
    
    # Report memory usage
    profiler.snapshot("End of progressive feature selection")
    profiler.report()
    
    return metrics

def run_feature_selection_discovery(tomatoes):
    """Stage 1: Run with feature selection to discover optimal bands and components."""
    print("\n=== STAGE 1: FEATURE SELECTION DISCOVERY ===")
    
    selected_bands = config.PATENT_BANDS  # [104, 113, 170]
    attributes = config.PREDICTED_QUALITY_ATTRIBUTES
    
    # Use utility functions to get components and component dimensions
    components = get_components()
    component_dimensions = get_component_dimensions(components, selected_bands)
    
    # Determine maximum 2D dimensions and set in config
    config.MAX_2D_DIMENSIONS = config.MAX_2D_DIMENSIONS if hasattr(config, 'MAX_2D_DIMENSIONS') else DataProcessingUtils.get_max_dimensions(tomatoes)
    augment_times = getattr(config, 'AUGMENT_TIMES', 0) # Number of augmented samples per tomato
    
    # Calculate total channels from component dimensions
    total_channels = component_dimensions['total']
    
    height, width = config.MAX_2D_DIMENSIONS
    model_shape = (height, width, total_channels)

    # Step 8: Initialize the desired model using the factory models
    model = get_model(
        model_type=config.MODEL_TYPE,
        image_path_kind=config.IMAGES_PATH_KIND,
        selected_bands=selected_bands,
        selected_indexes=config.INDEXES,
        components=components,
        component_dimensions=component_dimensions,  # Pass the component dimensions
        model_shape=model_shape,
        predicted_attributes=config.PREDICTED_QUALITY_ATTRIBUTES
    )
    
    # Prepare data splits
    random_number = config.RANDOM_STATE
    X_val, Y_val = None, None
    X_dummy = np.arange(len(tomatoes))  # just indices to split
    X_train_idx, X_temp_idx = train_test_split(X_dummy, test_size=0.3, random_state=random_number)
    
    if config.USE_VALIDATION:
        # 70% train, 15% val, 15% test
        X_val_idx, X_test_idx = train_test_split(X_temp_idx, test_size=0.5, random_state=random_number)
        train_tomatoes = [tomatoes[i] for i in X_train_idx]
        val_tomatoes = [tomatoes[i] for i in X_val_idx]
        test_tomatoes = [tomatoes[i] for i in X_test_idx]
        
        # Clean up memory after data splits created
        del X_train_idx, X_val_idx, X_test_idx, X_temp_idx
        import gc
        gc.collect()
        
        X_train, Y_train = model.prepare_data(train_tomatoes, selected_bands, augment_times)
        del train_tomatoes  # Free train tomatoes after data preparation
        gc.collect()
        
        X_val, Y_val = model.prepare_data(val_tomatoes, selected_bands)
        del val_tomatoes  # Free val tomatoes after data preparation
        gc.collect()
        
        X_test, Y_test = model.prepare_data(test_tomatoes, selected_bands)
        del test_tomatoes  # Free test tomatoes after data preparation
        gc.collect()
        
    else:
        # 70% train, 30% test
        X_dummy = np.arange(len(tomatoes))
        X_train_idx, X_test_idx = train_test_split(X_dummy, test_size=0.3, random_state=random_number)
        train_tomatoes = [tomatoes[i] for i in X_train_idx]
        val_tomatoes = []
        test_tomatoes = [tomatoes[i] for i in X_test_idx]
        
        # Clean up memory after data splits created
        del X_train_idx, X_test_idx, X_dummy
        gc.collect()
        
        X_train, Y_train = model.prepare_data(train_tomatoes, selected_bands, augment_times)
        del train_tomatoes  # Free train tomatoes after data preparation
        gc.collect()
        
        X_test, Y_test = model.prepare_data(test_tomatoes, selected_bands)
        del test_tomatoes  # Free test tomatoes after data preparation
        gc.collect()
        
    
    # Model train (this will discover optimal bands/components)
    print("🔍 Training with feature selection to discover optimal components...")
    model.train(X_train, Y_train, X_val, Y_val, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
    
    # Clean up memory after training
    del X_train, Y_train
    if X_val is not None:
        del X_val, Y_val
    if X_test is not None:
        del X_test, Y_test
    import gc
    gc.collect()
    
    
    # Extract discovered components from the trained model
    discovered_bands = []
    discovered_components = {}
    
    if hasattr(model, 'actual_selected_bands') and model.actual_selected_bands:
        discovered_bands = model.actual_selected_bands
        print(f"🎯 Discovered optimal bands: {discovered_bands}")
    else:
        discovered_bands = selected_bands
        print(f"⚠️ No bands discovered, using original: {discovered_bands}")
    
    # Extract component information
    if hasattr(model, 'actual_std_bands') and model.actual_std_bands:
        discovered_components['std_bands'] = model.actual_std_bands
        print(f"🎯 Discovered STD bands: {len(model.actual_std_bands)} bands")
    
    if hasattr(model, 'actual_ndsi_pairs') and model.actual_ndsi_pairs:
        discovered_components['ndsi_pairs'] = model.actual_ndsi_pairs
        print(f"🎯 Discovered NDSI pairs: {model.actual_ndsi_pairs}")
    
    if hasattr(model, 'actual_index_names') and model.actual_index_names:
        discovered_components['index_names'] = model.actual_index_names
        print(f"🎯 Discovered index names: {model.actual_index_names}")
    
    print("✅ Stage 1 complete - Components discovered!")
    
    # Clean up memory after function completion
    import gc
    gc.collect()
    
    
    return discovered_bands, discovered_components

def run_with_discovered_components(tomatoes, discovered_bands, discovered_components):
    """Stage 2: Run without feature selection using discovered components."""
    print("\n=== STAGE 2: TRAINING WITH DISCOVERED COMPONENTS ===")
    
    # Temporarily disable feature selection for this run
    original_use_fs = config.USE_FEATURE_SELECTION
    config.USE_FEATURE_SELECTION = False
    
    # Update bands to discovered ones
    original_bands = config.PATENT_BANDS
    config.PATENT_BANDS = discovered_bands
    
    # Update NDSI pairs if discovered
    original_ndsi_pairs = getattr(config, 'NDSI_BAND_PAIRS', [])
    if 'ndsi_pairs' in discovered_components:
        config.NDSI_BAND_PAIRS = discovered_components['ndsi_pairs']
    
    # Update indexes if discovered (handle learned indexes)
    original_indexes = config.INDEXES
    if 'index_names' in discovered_components:
        # Convert discovered index names to proper format
        discovered_indexes = []
        for idx_name in discovered_components['index_names']:
            if isinstance(idx_name, str) and idx_name.startswith('FS_Learned_'):
                # For learned indexes, we'll create a placeholder that the model can recognize
                discovered_indexes.append(idx_name)
            else:
                # For existing SpectralIndex objects, keep as is
                discovered_indexes.append(idx_name)
        config.INDEXES = discovered_indexes
        print(f"🎯 Using discovered indexes: {discovered_indexes}")
    
    try:
        print(f"🎯 Using discovered bands: {discovered_bands}")
        print(f"🎯 Using discovered NDSI pairs: {getattr(config, 'NDSI_BAND_PAIRS', 'default')}")
        
        # Check if we should re-initialize the model or use the trained one
        reinitialize = getattr(config, 'FEATURE_SELECTION_REINITIALIZE_MODEL', True)
        if reinitialize:
            print("🔄 Re-initializing model for fresh training with discovered components")
            # Run standard training with discovered components (new model)
            metrics = run_standard_training(tomatoes, log_as_feature_selection=True)
        else:
            print("🔄 Using trained model from Stage 1 for Stage 2 (fine-tuning approach)")
            # This would require passing the trained model from Stage 1
            # For now, we'll default to re-initialization as it's more reliable
            metrics = run_standard_training(tomatoes, log_as_feature_selection=True)
        
        print("✅ Stage 2 complete - Training with discovered components finished!")
        
        # Clean up memory after function completion
        import gc
        gc.collect()
        
        
        return metrics
        
    finally:
        # Restore original configuration
        config.USE_FEATURE_SELECTION = original_use_fs
        config.PATENT_BANDS = original_bands
        config.NDSI_BAND_PAIRS = original_ndsi_pairs
        config.INDEXES = original_indexes

def run_standard_training(tomatoes, log_as_feature_selection=False):
    """Run standard training without feature selection."""
    selected_bands = config.PATENT_BANDS  # [104, 113, 170]
    attributes = config.PREDICTED_QUALITY_ATTRIBUTES
    
    # Use utility functions to get components and component dimensions
    components = get_components()
    component_dimensions = get_component_dimensions(components, selected_bands)
    
    # Determine maximum 2D dimensions and set in config
    config.MAX_2D_DIMENSIONS = config.MAX_2D_DIMENSIONS if hasattr(config, 'MAX_2D_DIMENSIONS') else DataProcessingUtils.get_max_dimensions(tomatoes)
    augment_times = getattr(config, 'AUGMENT_TIMES', 0) # Number of augmented samples per tomato
    
    # Calculate total channels from component dimensions
    total_channels = component_dimensions['total']
    
    height, width = config.MAX_2D_DIMENSIONS
    model_shape = (height, width, total_channels)

    # Step 8: Initialize the desired model using the factory models
    model = get_model(
        model_type=config.MODEL_TYPE,
        image_path_kind=config.IMAGES_PATH_KIND,
        selected_bands=selected_bands,
        selected_indexes=config.INDEXES,
        components=components,
        component_dimensions=component_dimensions,  # Pass the component dimensions
        model_shape=model_shape,
        predicted_attributes=config.PREDICTED_QUALITY_ATTRIBUTES
    )
    random_number = config.RANDOM_STATE
    X_val, Y_val = None, None
    X_dummy = np.arange(len(tomatoes))  # just indices to split
    X_train_idx, X_temp_idx = train_test_split(X_dummy, test_size=0.3, random_state=random_number) # used to be 42
    if config.USE_VALIDATION:
        # 70% train, 15% val, 15% test
        X_val_idx, X_test_idx = train_test_split(X_temp_idx, test_size=0.5, random_state=random_number)

        train_tomatoes = [tomatoes[i] for i in X_train_idx]
        val_tomatoes = [tomatoes[i] for i in X_val_idx]
        test_tomatoes = [tomatoes[i] for i in X_test_idx]
        
        # Clean up memory after data splits created
        del X_train_idx, X_val_idx, X_test_idx, X_temp_idx
        import gc
        gc.collect()
        
        X_train, Y_train = model.prepare_data(train_tomatoes, selected_bands, augment_times)
        del train_tomatoes  # Free train tomatoes after data preparation
        gc.collect()
        
        X_val, Y_val = model.prepare_data(val_tomatoes, selected_bands)
        del val_tomatoes  # Free val tomatoes after data preparation
        gc.collect()
        
        X_test, Y_test = model.prepare_data(test_tomatoes, selected_bands)
        del test_tomatoes  # Free test tomatoes after data preparation
        gc.collect()
        

    else:
        # 70% train, 30% test
        X_dummy = np.arange(len(tomatoes))
        X_train_idx, X_test_idx = train_test_split(X_dummy, test_size=0.3, random_state=random_number)

        train_tomatoes = [tomatoes[i] for i in X_train_idx]
        val_tomatoes = []
        test_tomatoes = [tomatoes[i] for i in X_test_idx]
        
        # Clean up memory after data splits created
        del X_train_idx, X_test_idx, X_dummy
        gc.collect()
        
        X_train, Y_train = model.prepare_data(train_tomatoes, selected_bands, augment_times)
        del train_tomatoes  # Free train tomatoes after data preparation
        gc.collect()
        
        X_test, Y_test = model.prepare_data(test_tomatoes, selected_bands)
        del test_tomatoes  # Free test tomatoes after data preparation
        gc.collect()
        
    
    # # Model train
    model.train(X_train, Y_train, X_val, Y_val, epochs=config.EPOCHS, batch_size=config.BATCH_SIZE)
    
    # Clean up memory after training
    del X_train, Y_train
    if X_val is not None:
        del X_val, Y_val
    import gc
    gc.collect()
    
    
    # # Model test
    metrics = model.test(X_test, Y_test)
    print("Evaluation Metrics:", metrics)
    
    # Clean up memory after testing
    del X_test, Y_test
    gc.collect()
    

    # Step: Feature Importance Analysis (if enabled)
    if config.ANALYZE_FEATURE_IMPORTANCE:
        print("\n" + "="*60)
        print("ANALYZING CHANNEL IMPORTANCE")
        print("="*60)
        
        # Analyze channel importance for DL models
        importance_results = model.calculate_feature_importance(
            X_test=X_test,
            Y_test=Y_test,
            methods=["permutation", "gradient"],  # Both methods for DL models
            n_repeats=5,         # Fewer repeats for DL (computationally expensive)
            plot_top_n=None,     # Show ALL channels in plots
            print_top_n=10,      # Print top 10 in console
            save_results=True,   # Save JSON and CSV results
            save_plots=True      # Generate and save plots
        )
        
        print(f"\n✅ Channel importance analysis completed!")
        print(f"📁 Results saved in: results/feature_importance/{model.__class__.__name__}/")

    #  Step 10: plot results
    # Plot the evaluation metrics
    # model.plot_evaluate(metrics)
    # Compare model metrics to XGBoost metrics_dict
    compare_model_metrics_to_best_results(metrics, selected_bands, model.model_name, model.model_type, model.model_filename )
    
    # Save results to CSV file
    from src.utils.comparing import save_dl_results_to_csv
    save_dl_results_to_csv(metrics, selected_bands, model.model_name, model.model_type)

    
    # Clean up memory after function completion
    import gc
    gc.collect()
    

    return metrics

if __name__ == "__main__":
    main()
