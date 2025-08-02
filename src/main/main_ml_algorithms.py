# main.py


import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import src.Models as Models
from src.Models.utils import get_model
import src.Tomato as Tomato
import src.CreateDataset as CreateDataset
import src.ImageData as ImageData
import src.config as config
from sklearn.model_selection import train_test_split
import src.utils as utils
from src.config.enums import *
import numpy as np  # Ensure numpy is imported
from src.utils.logger import get_logger
from src.utils.data_processing import DataProcessingUtils
from src.utils.comparing import compare_model_metrics_to_best_results

def main():
    logger = get_logger("main_ml")
    
    if config.CREATE_DATASET:
        logger.info("Creating dataset...")
        
        # # Step 1: create the tomatoes dictionary
        tomatoes_dict = CreateDataset.File_helper.parse_custom_dict(config.TOMATOES_DICT_PATH)

        # # Step 2: Create the dataset
        two_side_flag = config.IMAGES_PATH_KIND == ImagePathKind.ONE_SIDE
        tomatoes = CreateDataset.Dataset.create_dataset(
            tomatoes_dict['tomatoes_dict'],
            two_sides_flag=two_side_flag,
            only_first_harvest=config.ONLY_HARVEST_1
        )
        dataset_path = f"project_data/Data/saved_datasets/tomatoes_dataset_side-{config.IMAGES_PATH_KIND.value}_objectsNum-{len(tomatoes)}.pkl"
        # Assign area to each tomato (part of dataset creation)
        DataProcessingUtils.assign_area_to_tomatoes( tomatoes=tomatoes, area_excel_path=config.SPECTRAL_STATS_FILENAME )
        

        # # Step 3: Save the dataset
        CreateDataset.Dataset.save_dataset(tomatoes, filename=dataset_path)
        logger.info(f"Dataset created and saved to {dataset_path}")


    # # Don't comment if step 2 is commented (if dataset is not created)
    num_of_tomatoes = 872 if config.IMAGES_PATH_KIND == ImagePathKind.TWO_SIDES else 1744
    dataset_path = f"project_data/Data/saved_datasets/tomatoes_dataset_side-{config.IMAGES_PATH_KIND.value}_objectsNum-{num_of_tomatoes}.pkl"

    # # Step 4: Load the dataset
    logger.info(f"Loading dataset from {dataset_path}")
    tomatoes = CreateDataset.Dataset.load_dataset(dataset_path)
    logger.info(f'Dataset loaded successfully! ({len(tomatoes)} tomatoes)')


    # # Globals and Cross-Variables:
    # Define the spectral bands and quality attributes
    selected_bands = config.PATENT_BANDS  # [104, 113, 170]
    attributes = config.PREDICTED_QUALITY_ATTRIBUTES

    # Determine maximum 2D dimensions and set in config
    config.MAX_2D_DIMENSIONS = config.MAX_2D_DIMENSIONS if hasattr(config, 'MAX_2D_DIMENSIONS') else DataProcessingUtils.get_max_dimensions(tomatoes)

    # Determine which components are chosen
    band_pairs = DataProcessingUtils.generate_band_pairs(selected_bands)
    num_selected_bands = len(selected_bands) if config.USE_COMPONENTS.get('reflectance', False) else 0
    num_ndsi_channels = len(band_pairs) if config.USE_COMPONENTS.get('ndsi', False) else 0 # one channel per band pair
    num_index_channels = len(config.INDEXES) if config.INDEXES and config.USE_COMPONENTS.get('indexes', False) else 0
    total_channels = num_selected_bands + num_ndsi_channels + num_index_channels

    # Set CNN_INPUT_SIZE in config (if applicable)
    height, width = config.MAX_2D_DIMENSIONS
    config.CNN_INPUT_SIZE = (height, width, total_channels)

   # # # Optional Step - Compare the dataset to state-of-the-art dataset and create csv file for the comparison (with the difference paint in red):
    # utils.comparing.verify_lab_results(tomatoes=tomatoes)

    # # # Optional Step: Perform segmentation checks 
    # # To perform segmentation checks, uncomment the following line:
    # utils.segmentation.segmentation_checks(tomatoes)


    # Step 8: Initialize the desired model using the factory models
    # Override MODEL_TYPE for ML algorithms
    ml_model_type = ModelType.RANDOM_FOREST  # Use RANDOM_FOREST for ML script
    config.MODEL_TYPE = ml_model_type  # Set it in config so other modules see it
    logger.info(f"Initializing {ml_model_type} model...")
    model = get_model(
        model_type=ml_model_type,  # Use ML model type instead of config
        image_path_kind=config.IMAGES_PATH_KIND,
        selected_bands=selected_bands,
        selected_indexes=config.INDEXES,
        components=config.USE_COMPONENTS,
        model_shape=[1,1,1]
    )
    logger.info(f"Initialized {ml_model_type} model with image path kind '{config.IMAGES_PATH_KIND}'.")
    
    # Filter tomatoes by specific harvests
    selected_harvests = config.SELECTED_HARVESTS if hasattr(config, "SELECTED_HARVESTS") else None  
    filtered_tomatoes = [tomato for tomato in tomatoes if tomato.harvest in selected_harvests] if selected_harvests is not None else tomatoes
    logger.info(f"Filtered dataset from {len(tomatoes)} to {len(filtered_tomatoes)} tomatoes "
          f"(selected harvests: {', '.join(selected_harvests) if selected_harvests else 'All harvests'})")
    
    # Step 6: Prepare the data
    X, Y = model.prepare_data(filtered_tomatoes, selected_bands)

    # # Step 7: Split the data into training, validation, and testing sets based on USE_VALIDATION flag
    # if config.USE_VALIDATION:
    #     # Split into train (70%), validation (15%), and test (15%)
    #     X_train, X_temp, Y_train, Y_temp = train_test_split(
    #         X, Y, test_size=0.3, random_state=42
    #     )
    #     X_val, X_test, Y_val, Y_test = train_test_split(
    #         X_temp, Y_temp, test_size=0.5, random_state=42
    #     )
    #     print(f"Data split into Train: {X_train.shape[0]} samples, Validation: {X_val.shape[0]} samples, Test: {X_test.shape[0]} samples.")
        
    #     # Train the model with validation data
    #     model.train(X_train, Y_train, X_val, Y_val, epochs=200, batch_size=32)
    # else:
    # Split into train (70%) and test (30%)
    logger.info("Splitting data into train and test sets...")
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=config.RANDOM_STATE
    )
    logger.info(f"Data split into Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples.")
    
    # Train the model without validation data
    logger.info("Starting model training...")
    model.train(X_train, Y_train, epochs=200, batch_size= config.BATCH_SIZE)


    # Step 9: Test the model
    logger.info("Testing model performance...")
    metrics = model.test(X_test, Y_test)
    logger.info("Evaluation Metrics:", metrics=metrics)

    # Step 9.5: Feature Importance Analysis (if enabled)
    if config.ANALYZE_FEATURE_IMPORTANCE:
        logger.info("=" * 60)
        logger.info("ANALYZING FEATURE IMPORTANCE")
        logger.info("=" * 60)
        
        # Analyze feature importance for ML models
        importance_results = model.calculate_feature_importance(
            X_test=X_test,
            Y_test=Y_test,
            methods=["built_in", "permutation"],  # Both methods for ML models
            n_repeats=30,        # Good balance for ML models  
            plot_top_n=None,     # Show ALL features in plots
            print_top_n=10,      # Print top 10 in console
            save_results=True,   # Save JSON and CSV results
            save_plots=True      # Generate and save plots
        )
        
        logger.info("Feature importance analysis completed!")
        logger.info(f"Results saved in: results/feature_importance/{model.__class__.__name__}/")

    # Compare model metrics to XGBoost metrics_dict
    compare_model_metrics_to_best_results(metrics , selected_bands, model.model_name, ml_model_type )

    # Save results to CSV file
    from src.utils.comparing import save_dl_results_to_csv
    save_dl_results_to_csv(metrics, selected_bands, model.model_name, ml_model_type)

    # Step 11: Save the trained model
    model.save_model()

if __name__ == "__main__":
    main()
