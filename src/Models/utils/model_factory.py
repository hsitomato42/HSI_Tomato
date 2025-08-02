# Models/utils/model_factory.py

import os
from typing import List, Optional, Tuple
from ..dl_models.MultiHeadCNNModel import MultiHeadCNNModel
from ..dl_models.CNNTransformerModel import CNNTransformerModel
from ..dl_models.SpectralTransformerModel import SpectralTransformerModel
from ..dl_models.ViTModel import ViTModel
from ..dl_models.AdvancedMultiBranchCNNTransformer import AdvancedMultiBranchCNNTransformer
from ..dl_models.GlobalBranchFusionTransformer import GlobalBranchFusionTransformer
from ..dl_models.ComponentDrivenAttentionTransformer import ComponentDrivenAttentionTransformer
from ..dl_models.ComponentDrivenAttentionTransformerV2 import ComponentDrivenAttentionTransformerV2
from ..dl_models.PCCTStatic import PCCTStatic
from ..dl_models.PCCTProgressive import PCCTProgressive
from ..dl_models.PCCTStaticV2 import PCCTStaticV2
from ..dl_models.PCCTProgressiveV2 import PCCTProgressiveV2
from ..dl_models.CDATProgressive import CDATProgressive
from ..dl_models.CDATProgressiveV2 import CDATProgressiveV2
from src.config.config import MODEL_TYPE, IMAGES_PATH_KIND, INDEXES, BASE_MODEL_DIR, USE_COMPONENTS
from src.config.enums import ModelType, ImagePathKind, SpectralIndex
from ..dl_models.CNNModel import CNNModel
from ..ml_models.XGBoostModel import XGBoostModel
from ..ml_models.RandomForestModel import RandomForestModel
import src.config as config

def get_model(
    model_type: ModelType,
    image_path_kind: ImagePathKind,
    selected_bands: List[int],
    components: dict,  # {'reflectance': bool, 'ndsi': bool, 'indexes': bool}
    model_shape: Tuple[int, int, int],  # (H, W, C)
    component_dimensions: Optional[dict] = None,  # {'reflectance': int, 'std': int, 'ndsi': int, 'indexes': int, 'total': int}
    selected_indexes: Optional[List[SpectralIndex]] = None,
    predicted_attributes: Optional[List[str]] = config.PREDICTED_QUALITY_ATTRIBUTES
):
    """
    Factory function to instantiate and return the desired model.

    Arguments:
    - model_type: ModelType (e.g., ModelType.CNN, ModelType.XGBOOST, ModelType.MULTI_HEAD_CNN)
    - image_path_kind: ImagePathKind (e.g. ImagePathKind.ONE_SIDE)
    - selected_bands: List of band indices
    - selected_indexes: Optional list of spectral indexes
    - components: dict with boolean flags: {'reflectance': bool, 'ndsi': bool, 'indexes': bool}
    - component_dimensions: dict with channel counts: {'reflectance': int, 'std': int, 'ndsi': int, 'indexes': int, 'total': int}
    - model_shape: (H, W, C)
    - predicted_attributes: list of attributes to predict

    The final model naming:
    "modelType-{model_type.value}_selectedBands-(a, b, c...)_shape-(H, W, C)[_reflectance][_ndsi][_indexes-(...)]"

    Saves under: Models/Best Models/{model_type.value}/(a, b, c)/final_model_name.keras
    """
    
    # 🎯 FEATURE SELECTION INTEGRATION
    # Check if feature selection is enabled and update parameters accordingly
    original_components = components.copy()
    original_component_dimensions = component_dimensions.copy() if component_dimensions else {}
    original_model_shape = model_shape
    original_selected_bands = selected_bands.copy() if selected_bands is not None else None
    
    fs_interface = None
    if getattr(config, 'USE_FEATURE_SELECTION', False):
        print("\n🎯 FEATURE SELECTION ENABLED")
        
        # Handle FEATURE_SELECTION_STRATEGY - can be enum, string, or list
        fs_strategy = getattr(config, 'FEATURE_SELECTION_STRATEGY', 'DEFAULT')
        if isinstance(fs_strategy, list):
            # If it's a list, use the first element (enum string)
            strategy_name = fs_strategy[1] if len(fs_strategy) > 1 else fs_strategy[0]
        elif hasattr(fs_strategy, 'display_name'):
            # If it's an enum object
            strategy_name = fs_strategy.display_name
        else:
            # If it's a string
            strategy_name = str(fs_strategy)
        
        print(f"Strategy: {strategy_name}")
        
        # 🔄 FULL SPECTRUM MODE: Use all 204 bands if enabled
        full_spectrum_mode = getattr(config, 'FEATURE_SELECTION_FULL_SPECTRUM', True)
        if full_spectrum_mode:
            # Only print full spectrum mode info once to avoid spam during multiple model creation
            if not hasattr(get_model, '_fs_mode_printed'):
                get_model._fs_mode_printed = True
                print(f"[FS] FULL SPECTRUM MODE: Switching from pre-configured bands {selected_bands} to ALL 204 bands")
                print(f"[FS] Input: ALL 204 spectral bands (0-203)")
            selected_bands = list(range(204))  # Use all bands as input to feature selection
        else:
            if not hasattr(get_model, '_fs_constrained_printed'):
                get_model._fs_constrained_printed = True
                print(f"[FS] CONSTRAINED MODE: Using pre-configured bands {selected_bands}")
        
        # Import feature selection interface
        from src.feature_selection import FeatureSelectionInterface
        
        # Create feature selection interface with redesigned constructor
        fs_interface = FeatureSelectionInterface(
            original_shape=model_shape,
            components=components,
            selected_bands=selected_bands if not full_spectrum_mode else list(range(204)),
            selected_indexes=selected_indexes,
            enable_feature_selection=True
        )
        
        # Get updated dimensions after feature selection
        fs_component_dims = fs_interface.get_component_dimensions()
        
        # Update component dimensions for model creation
        component_dimensions = fs_component_dims
        
        # Update model shape (height, width, total_channels) 
        height, width, _ = model_shape
        total_channels = fs_component_dims['total_channels']
        model_shape = (height, width, total_channels)
        
        # Only print model shape info once to avoid spam during multiple model creation
        if not hasattr(get_model, '_fs_shape_printed'):
            get_model._fs_shape_printed = True
            print(f"[FS] Original shape: {original_model_shape} → New shape: {model_shape}")
            if full_spectrum_mode:
                print(f"[FS] Data reduction: {(1 - total_channels / 204) * 100:.1f}% (204 → {total_channels} channels)")
            else:
                print(f"[FS] Data reduction: {(1 - total_channels / original_model_shape[2]) * 100:.1f}% ({original_model_shape[2]} → {total_channels} channels)")
            print(f"[FS] Component dimensions: {fs_component_dims}")
            print(f"[FS] Strategy: {fs_strategy}")
        
        # Clean up memory after feature selection interface creation
        import gc
        gc.collect()
        
        # Clean up GPU memory after feature selection setup
        try:
            from utils.experiments.gpu.tensorflow_gpu_setup import cleanup_tensorflow_gpu
            cleanup_tensorflow_gpu()
        except ImportError:
            pass

    # Sort bands
    sorted_bands = sorted(selected_bands)
    # Sort indexes if provided - handle both SpectralIndex objects and string learned indexes
    if selected_indexes:
        def get_sort_key(index):
            if hasattr(index, 'shortcut'):
                return index.shortcut
            else:
                return str(index)  # For learned indexes like 'FS_Learned_1'
        
        selected_indexes = sorted(selected_indexes, key=get_sort_key)

    # Build the model name (include feature selection in naming if enabled)
    name_parts = [model_type.value]
    
    # Add multi-regression indicator
    if config.MULTI_REGRESSION:
        name_parts.append("MultiReg")
    else:
        name_parts.append("SinglePred")
    
    # Add image path kind
    name_parts.append(image_path_kind.value)
    
    # Add bands - special handling for full spectrum mode
    if fs_interface is not None and getattr(config, 'FEATURE_SELECTION_FULL_SPECTRUM', True):
        bands_str = "FullSpectrum"
        bands_folder_name = f"(FullSpectrum_204bands)"
    else:
        bands_str = "_".join(map(str, sorted_bands))
        bands_folder_name = f"({bands_str})"
    
    name_parts.append(f"bands_{bands_str}")
    
    # Add components
    components_str = ""
    if components.get('reflectance', False):
        components_str += "R"
    if components.get('std', False):
        components_str += "S"
    if components.get('ndsi', False):
        components_str += "N"
    if components.get('indexes', False):
        components_str += "I"
    
    # Add feature selection indicator
    if fs_interface is not None:
        components_str += f"_FS{config.FEATURE_SELECTION_K_BANDS}"
    
    # Add padding info
    padded_str = f"pad_{config.PADDED_VALUE}" if hasattr(config, 'PADDED_VALUE') and config.PADDED_VALUE > 0 else ""
    
    # Add augmentation info
    augment_str = f"aug_{config.AUGMENT_TIMES}" if hasattr(config, 'AUGMENT_TIMES') and config.AUGMENT_TIMES > 0 else ""
    
    # Add V2-specific info for V2 DL models
    v2_config_str = ""
    v2_model_types = [
        ModelType.COMPONENT_DRIVEN_ATTENTION_TRANSFORMER_V2,
        ModelType.CDAT_PROGRESSIVE_V2,
        ModelType.PCCT_STATIC_V2,
        ModelType.PCCT_PROGRESSIVE_V2
    ]
    
    if model_type in v2_model_types:
        # Use new V2_DL parameters with fallback to legacy CDAT_V2 parameters
        fusion_method = (getattr(config, 'V2_DL_FUSION_METHOD', None) or 
                        getattr(config, 'CDAT_V2_FUSION_METHOD', 'sequential'))
        downsampling_method = (getattr(config, 'V2_DL_DOWNSAMPLING_METHOD', None) or 
                              getattr(config, 'CDAT_V2_DOWNSAMPLING_METHOD', 'strided_conv'))
        downsampling_factor = (getattr(config, 'V2_DL_DOWNSAMPLING_FACTOR', None) or 
                              getattr(config, 'CDAT_V2_DOWNSAMPLING_FACTOR', 2))
        v2_config_str = f"{fusion_method}_{downsampling_method}_{downsampling_factor}x"
    
    if components_str: # Add component string only if it's not empty
        name_parts.append(components_str)
    if padded_str:
        name_parts.append(padded_str)
    if augment_str:
        name_parts.append(augment_str)
    if v2_config_str:
        name_parts.append(v2_config_str)

    # Filter out any empty strings from name_parts before joining, to avoid double underscores
    final_model_name = "_".join(filter(None, name_parts))

    # Create directories
    model_base_path = os.path.join("Models", "Best Models", model_type.value)
    model_band_subfolder = os.path.join(model_base_path, bands_folder_name)
    os.makedirs(model_band_subfolder, exist_ok=True)

    checkpoint_path = os.path.join(model_band_subfolder, final_model_name + ".keras")
    
    # Clean up memory after directory creation and before model instantiation
    import gc
    gc.collect()
    
    # Clean up GPU memory before model instantiation
    try:
        from utils.experiments.gpu.tensorflow_gpu_setup import cleanup_tensorflow_gpu
        cleanup_tensorflow_gpu()
    except ImportError:
        pass

    # Instantiate the model based on model_type
    model = None
    if model_type == ModelType.CNN:
        model = CNNModel(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes
        )
    elif model_type == ModelType.MULTI_HEAD_CNN:
        model = MultiHeadCNNModel(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes
        )
    elif model_type == ModelType.CNN_TRANSFORMER:
        model = CNNTransformerModel(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes
        )
    elif model_type == ModelType.SPECTRAL_TRANSFORMER:  # Add this condition
        model = SpectralTransformerModel(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes
        )
    elif model_type == ModelType.VIT:  # e.g. define new enum in config/enums.py
        model = ViTModel(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes
        )
    elif model_type == ModelType.ADVANCED_MULTI_BRANCH_CNN_TRANSFORMER:
        model = AdvancedMultiBranchCNNTransformer(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes
        )
    elif model_type == ModelType.GLOBAL_BRANCH_FUSION_TRANSFORMER:
        model = GlobalBranchFusionTransformer(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes
        )
    elif model_type == ModelType.COMPONENT_DRIVEN_ATTENTION_TRANSFORMER:
        model = ComponentDrivenAttentionTransformer(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes
        )
    elif model_type == ModelType.COMPONENT_DRIVEN_ATTENTION_TRANSFORMER_V2:
        # Get V2 parameters with fallback to legacy CDAT_V2 parameters
        fusion_method = (getattr(config, 'V2_DL_FUSION_METHOD', None) or 
                        getattr(config, 'CDAT_V2_FUSION_METHOD', 'sequential'))
        downsampling_method = (getattr(config, 'V2_DL_DOWNSAMPLING_METHOD', None) or 
                              getattr(config, 'CDAT_V2_DOWNSAMPLING_METHOD', 'strided_conv'))
        downsampling_factor = (getattr(config, 'V2_DL_DOWNSAMPLING_FACTOR', None) or 
                              getattr(config, 'CDAT_V2_DOWNSAMPLING_FACTOR', 2))
        
        model = ComponentDrivenAttentionTransformerV2(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes,
            # V2-specific configuration
            fusion_method=fusion_method,
            downsampling_method=downsampling_method,
            downsampling_factor=downsampling_factor
        )
    elif model_type == ModelType.XGBOOST:
        model = XGBoostModel(
            model_name=final_model_name,
            images_path_kind=image_path_kind,
            selected_indexes=selected_indexes
        )
    elif model_type == ModelType.RANDOM_FOREST:
        model = RandomForestModel(
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            predicted_attributes=predicted_attributes
        )
    elif model_type == ModelType.PCCT_STATIC:
        model = PCCTStatic(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes
        )
    elif model_type == ModelType.PCCT_PROGRESSIVE:
        model = PCCTProgressive(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes
        )
    elif model_type == ModelType.PCCT_STATIC_V2:
        # Get V2 parameters
        downsampling_method = (getattr(config, 'V2_DL_DOWNSAMPLING_METHOD', None) or 
                              getattr(config, 'CDAT_V2_DOWNSAMPLING_METHOD', 'strided_conv'))
        downsampling_factor = (getattr(config, 'V2_DL_DOWNSAMPLING_FACTOR', None) or 
                              getattr(config, 'CDAT_V2_DOWNSAMPLING_FACTOR', 2))
        
        model = PCCTStaticV2(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes,
            # V2-specific configuration
            downsampling_method=downsampling_method,
            downsampling_factor=downsampling_factor
        )
    elif model_type == ModelType.PCCT_PROGRESSIVE_V2:
        # Get V2 parameters
        downsampling_method = (getattr(config, 'V2_DL_DOWNSAMPLING_METHOD', None) or 
                              getattr(config, 'CDAT_V2_DOWNSAMPLING_METHOD', 'strided_conv'))
        downsampling_factor = (getattr(config, 'V2_DL_DOWNSAMPLING_FACTOR', None) or 
                              getattr(config, 'CDAT_V2_DOWNSAMPLING_FACTOR', 2))
        
        model = PCCTProgressiveV2(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes,
            # V2-specific configuration
            downsampling_method=downsampling_method,
            downsampling_factor=downsampling_factor
        )
    elif model_type == ModelType.CDAT_PROGRESSIVE:
        model = CDATProgressive(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes
        )
    elif model_type == ModelType.CDAT_PROGRESSIVE_V2:
        # Get V2 parameters
        downsampling_method = (getattr(config, 'V2_DL_DOWNSAMPLING_METHOD', None) or 
                              getattr(config, 'CDAT_V2_DOWNSAMPLING_METHOD', 'strided_conv'))
        downsampling_factor = (getattr(config, 'V2_DL_DOWNSAMPLING_FACTOR', None) or 
                              getattr(config, 'CDAT_V2_DOWNSAMPLING_FACTOR', 2))
        
        model = CDATProgressiveV2(
            selected_bands=sorted_bands,
            model_name=final_model_name,
            model_filename=checkpoint_path,
            selected_indexes=selected_indexes,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            predicted_attributes=predicted_attributes,
            # V2-specific configuration
            downsampling_method=downsampling_method,
            downsampling_factor=downsampling_factor
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # 🎯 ADD FEATURE SELECTION INTERFACE TO DL MODELS
    # Add feature selection interface to DL models (those with components attribute)
    if hasattr(model, 'components') and fs_interface is not None:
        model.fs_interface = fs_interface
        
        # Store original dimensions for reference
        model.original_components = original_components
        model.original_component_dimensions = original_component_dimensions
        model.original_model_shape = original_model_shape
        model.original_selected_bands = original_selected_bands
        
        # Only print interface addition once to avoid spam during multiple model creation
        if not hasattr(get_model, '_fs_interface_printed'):
            get_model._fs_interface_printed = True
            print(f"[FS] Added feature selection interface to {model.__class__.__name__}")
        
        # Override the model's prepare_data method to apply feature selection
        original_prepare_data = model.prepare_data
        
        def prepare_data_with_fs(tomatoes, selected_bands_param, augment_times=0):
            # For feature selection models, always prepare RAW REFLECTANCE data
            # Feature selection will be applied during training, not during data preparation
            # Check if full spectrum mode is enabled
            full_spectrum_mode = getattr(config, 'FEATURE_SELECTION_FULL_SPECTRUM', True)
            if full_spectrum_mode:
                # Only print data preparation info once to avoid spam
                if not hasattr(prepare_data_with_fs, '_fs_data_printed'):
                    prepare_data_with_fs._fs_data_printed = True
                    print(f"[FS] FULL SPECTRUM: Preparing RAW reflectance data for ALL 204 bands")
                # Load raw reflectance data only (no pre-computed components)
                X, Y = prepare_raw_reflectance_data(tomatoes, list(range(204)), augment_times)
            else:
                # Use pre-configured bands (constrained mode)
                bands_to_load = model.original_selected_bands
                if not hasattr(prepare_data_with_fs, '_fs_constrained_data_printed'):
                    prepare_data_with_fs._fs_constrained_data_printed = True
                    print(f"[FS] CONSTRAINED: Preparing RAW reflectance data for pre-configured bands: {bands_to_load}")
                X, Y = prepare_raw_reflectance_data(tomatoes, bands_to_load, augment_times)
            
            # Only print shape info on first call to avoid spam
            if not hasattr(prepare_data_with_fs, '_fs_shape_data_printed'):
                prepare_data_with_fs._fs_shape_data_printed = True
                print(f"[FS] Raw data prepared for feature selection: {X.shape}")
                print(f"[FS] Feature selection will be applied during training/inference")
            
            # Initialize placeholders for actual selections (will be set during training)
            model.actual_selected_bands = []
            model.actual_std_bands = []
            model.actual_ndsi_pairs = []
            model.actual_index_names = []
            model.fs_selection_applied = False  # Will be set to True during training
            
            return X, Y
        
        def prepare_raw_reflectance_data(tomatoes, bands_to_load, augment_times=0):
            """Helper function to load only raw reflectance data for feature selection."""
            from utils.data_processing import DataProcessingUtils
            import numpy as np
            
            X_features = []
            Y_targets = []
            
            # Create augmented tomatoes
            augmented_tomatoes = DataProcessingUtils.create_augmented_and_padded_tomatoes(
                tomatoes=tomatoes,
                augment_times=augment_times
            )
            
            for tomato in augmented_tomatoes:
                # Only load raw reflectance matrix for specified bands
                reflectance_matrix = tomato.spectral_stats.reflectance_matrix
                reflectance_features = reflectance_matrix[:, :, bands_to_load]
                
                # Build the Y vector
                quality_assess = tomato.quality_assess
                y_values = [getattr(quality_assess, attr, np.nan) for attr in config.PREDICTED_QUALITY_ATTRIBUTES]
                
                X_features.append(reflectance_features)
                Y_targets.append(y_values)
            
            X = np.array(X_features)
            Y = np.array(Y_targets, dtype=float)
            Y_filled = DataProcessingUtils.fill_missing_values(Y, config.PREDICTED_QUALITY_ATTRIBUTES)
            Y = Y_filled
            
            return X, Y
        
        # Replace the method
        model.prepare_data = prepare_data_with_fs
        
        # Only print enhancement info once to avoid spam
        if not hasattr(get_model, '_fs_enhancement_printed'):
            get_model._fs_enhancement_printed = True
            print(f"[FS] Enhanced data preparation for {model.__class__.__name__}")

    # Clean up memory after model creation and configuration
    import gc
    gc.collect()
    
    # Clean up GPU memory after model instantiation
    try:
        from utils.experiments.gpu.tensorflow_gpu_setup import cleanup_tensorflow_gpu
        cleanup_tensorflow_gpu()
    except ImportError:
        pass

    return model
