# utils/model_utils.py

from typing import Dict, List
import src.config as config
from src.utils.data_processing import DataProcessingUtils

def get_components() -> Dict[str, bool]:
    """
    Create a dictionary indicating which components are enabled based on configuration.
    
    Returns:
        Dict[str, bool]: Dictionary with keys 'reflectance', 'ndsi', 'indexes', 'std'
            and boolean values indicating if each component is enabled.
    """
    components = {
        'reflectance': config.USE_COMPONENTS.get('reflectance', False),
        'ndsi': config.USE_COMPONENTS.get('ndsi', False),
        'indexes': config.USE_COMPONENTS.get('indexes', False),
        'std': config.USE_COMPONENTS.get('std', False)
    }
    return components

def get_component_dimensions(
    components: Dict[str, bool], 
    selected_bands: List[int] = None
) -> Dict[str, int]:
    """
    Calculate the dimensions (number of channels) for each component.
    
    Args:
        components (Dict[str, bool]): Dictionary indicating which components are enabled.
        selected_bands (List[int], optional): List of selected spectral bands. 
                                            If None, feature selection dimensions will be used.
        
    Returns:
        Dict[str, int]: Dictionary with keys for each component and their channel counts:
            - 'reflectance': Number of channels for reflectance component
            - 'std': Number of channels for STD component
            - 'ndsi': Number of channels for NDSI component
            - 'indexes': Number of channels for indexes component
            - 'total': Total number of channels
    """
    # Handle feature selection case
    if selected_bands is None and getattr(config, 'USE_FEATURE_SELECTION', False):
        # Use feature selection dimensions from config
        num_selected_bands = getattr(config, 'FEATURE_SELECTION_K_BANDS', 5) if components['reflectance'] else 0
        num_std_channels = getattr(config, 'FEATURE_SELECTION_B_STD', 5) if components['std'] else 0
        num_ndsi_channels = getattr(config, 'FEATURE_SELECTION_A_NDSI', 3) if components['ndsi'] else 0
        num_index_channels = getattr(config, 'FEATURE_SELECTION_C_INDEXES', 5) if components['indexes'] else 0
    else:
        # Traditional case with pre-selected bands
        if selected_bands is None:
            selected_bands = getattr(config, 'PATENT_BANDS', [])
        
        # Generate band pairs for NDSI
        band_pairs = DataProcessingUtils.generate_band_pairs(selected_bands)
        
        # Calculate number of channels for each component
        num_selected_bands = len(selected_bands) if components['reflectance'] else 0
        num_ndsi_channels = len(band_pairs) if components['ndsi'] else 0  # one channel per band pair
        num_index_channels = len(config.INDEXES) if config.INDEXES and components['indexes'] else 0
        num_std_channels = len(selected_bands) if components['std'] else 0
    
    # Calculate total channels
    total_channels = num_selected_bands + num_ndsi_channels + num_index_channels + num_std_channels
    
    # Create component_dimensions dictionary
    component_dimensions = {
        'reflectance': num_selected_bands,
        'std': num_std_channels,
        'ndsi': num_ndsi_channels,
        'indexes': num_index_channels,
        'total': total_channels
    }
    
    return component_dimensions 