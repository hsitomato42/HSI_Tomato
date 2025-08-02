# utils/spectral_indexes.py

import numpy as np
from typing import Dict, List
from src.config.enums import ModelType, SpectralIndex
from src.config import config
from src.utils.data_processing import DataProcessingUtils
from src.config.enums import ImagePathKind
from src.config.globals import ML_MODELS
class SpectralIndexCalculator:
    """
    A class to compute various spectral indexes based on reflectance data.
    """

    @staticmethod
    def compute_red_index(R: np.ndarray, G: np.ndarray) -> np.ndarray:
        """
        Computes the Red Index: (R - G) / (R + G)
        """
        result = (R - G) / (R + G + 1e-8)  # Adding epsilon to avoid division by zero
        # Check if result is scalar (for ML models)
        if np.isscalar(result) or result.ndim == 0:
            return result
        mask = (R == 0) & (G == 0)  # Mask where both R and G are zero
        result[mask] = 0  # Set result to zero where mask is True
        return result

    @staticmethod
    def compute_lycopene_index(R630: np.ndarray, R570: np.ndarray) -> np.ndarray:
        """
        Computes the Lycopene Index: (630 nm - 570 nm) / (630 nm + 570 nm)
        """
        result = (R630 - R570) / (R630 + R570 + 1e-8)
        # Check if result is scalar (for ML models)
        if np.isscalar(result) or result.ndim == 0:
            return result
        mask = (R630 == 0) & (R570 == 0)
        result[mask] = 0
        return result

    @staticmethod
    def compute_normalized_anthocyanin_index(R760: np.ndarray, R570: np.ndarray) -> np.ndarray:
        """
        Computes the Normalized Anthocyanin Index: (760 nm - 570 nm) / (760 nm + 570 nm)
        """
        result = (R760 - R570) / (R760 + R570 + 1e-8)
        # Check if result is scalar (for ML models)
        if np.isscalar(result) or result.ndim == 0:
            return result
        mask = (R760 == 0) & (R570 == 0)
        result[mask] = 0
        return result

    @staticmethod
    def compute_zarco_tejada_miller_index(R750: np.ndarray, R710: np.ndarray) -> np.ndarray:
        """
        Computes the Zarco-Tejada & Miller Spectral Index: 750 nm / 710 nm
        """
        result = R750 / (R710 + 1e-8)
        # Check if result is scalar (for ML models)
        if np.isscalar(result) or result.ndim == 0:
            return result
        mask = (R750 == 0) & (R710 == 0)
        result[mask] = 0
        return result

    @staticmethod
    def compute_green_leaf_index(G: np.ndarray, R: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Computes the Green Leaf Index: (G - R + (G - B)) / (2 * G + R + B)
        """
        result = (G - R + (G - B)) / (2 * G + R + B + 1e-8)
        # Check if result is scalar (for ML models)
        if np.isscalar(result) or result.ndim == 0:
            return result
        mask = (G == 0) & (R == 0) & (B == 0)
        result[mask] = 0
        return result

    @staticmethod
    def compute_green_red_vegetation_index(G: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Computes the Green-Red Vegetation Index: (G - R) / (G + R)
        """
        result = (G - R) / (G + R + 1e-8)
        # Check if result is scalar (for ML models)
        if np.isscalar(result) or result.ndim == 0:
            return result
        mask = (G == 0) & (R == 0)
        result[mask] = 0
        return result

    @staticmethod
    def compute_learned_index(
        reflectance_matrix: np.ndarray,
        selected_bands: List[int],
        learned_index_name: str
    ) -> np.ndarray:
        """
        Computes a learned spectral index using discovered bands.
        This creates simple but effective combinations from the selected bands.
        
        Args:
            reflectance_matrix (np.ndarray): Reflectance data of shape (H, W, bands).
            selected_bands (List[int]): List of selected band indices.
            learned_index_name (str): Name like 'FS_Learned_1', 'FS_Learned_2', etc.
            
        Returns:
            np.ndarray: Learned index values per pixel of shape (H, W).
        """
        if len(selected_bands) < 2:
            # If only one band, return it directly
            return reflectance_matrix[:, :, selected_bands[0]]
        
        # Extract the index number from the name (e.g., 'FS_Learned_1' -> 1)
        try:
            index_num = int(learned_index_name.split('_')[-1])
        except (ValueError, IndexError):
            index_num = 1
        
        # Create different learned combinations based on the index number
        # This mimics what the neural networks would learn but in a deterministic way
        
        if index_num % 4 == 1:
            # NDVI-like: (band2 - band1) / (band2 + band1)
            band1_idx = 0 % len(selected_bands)
            band2_idx = 1 % len(selected_bands)
            band1 = reflectance_matrix[:, :, selected_bands[band1_idx]]
            band2 = reflectance_matrix[:, :, selected_bands[band2_idx]]
            return (band2 - band1) / (band2 + band1 + 1e-8)
            
        elif index_num % 4 == 2:
            # Ratio index: band1 / band2
            band1_idx = (index_num - 1) % len(selected_bands)
            band2_idx = index_num % len(selected_bands)
            band1 = reflectance_matrix[:, :, selected_bands[band1_idx]]
            band2 = reflectance_matrix[:, :, selected_bands[band2_idx]]
            return band1 / (band2 + 1e-8)
            
        elif index_num % 4 == 3:
            # Three-band combination: (band1 - band2) / (band1 + band2 + band3)
            if len(selected_bands) >= 3:
                band1_idx = (index_num - 1) % len(selected_bands)
                band2_idx = index_num % len(selected_bands)
                band3_idx = (index_num + 1) % len(selected_bands)
                band1 = reflectance_matrix[:, :, selected_bands[band1_idx]]
                band2 = reflectance_matrix[:, :, selected_bands[band2_idx]]
                band3 = reflectance_matrix[:, :, selected_bands[band3_idx]]
                return (band1 - band2) / (band1 + band2 + band3 + 1e-8)
            else:
                # Fallback to two-band ratio
                band1_idx = 0 % len(selected_bands)
                band2_idx = 1 % len(selected_bands)
                band1 = reflectance_matrix[:, :, selected_bands[band1_idx]]
                band2 = reflectance_matrix[:, :, selected_bands[band2_idx]]
                return band1 / (band2 + 1e-8)
                
        else:  # index_num % 4 == 0
            # Difference index: band1 - band2
            band1_idx = (index_num - 1) % len(selected_bands)
            band2_idx = index_num % len(selected_bands)
            band1 = reflectance_matrix[:, :, selected_bands[band1_idx]]
            band2 = reflectance_matrix[:, :, selected_bands[band2_idx]]
            return band1 - band2

    @staticmethod
    def compute_pixel_index(
        reflectance_matrix: np.ndarray,
        selected_bands: List[int],
        spectral_index
    ) -> np.ndarray:
        """
        Computes a specified spectral index on a per-pixel basis.

        Args:
            reflectance_matrix (np.ndarray): Reflectance data of shape (H, W, bands).
            selected_bands (List[int]): List of selected band indices.
            spectral_index: The spectral index to compute (SpectralIndex enum or string).

        Returns:
            np.ndarray: Spectral index values per pixel of shape (H, W).
        """
        # Handle learned indexes (string format)
        if isinstance(spectral_index, str) and spectral_index.startswith('FS_Learned_'):
            return SpectralIndexCalculator.compute_learned_index(
                reflectance_matrix, selected_bands, spectral_index
            )
        
        # Handle regular SpectralIndex enums
        if spectral_index == SpectralIndex.RED_INDEX:
            R = reflectance_matrix[:, :, config.RED_BAND]
            G = reflectance_matrix[:, :, config.GREEN_BAND]
            return SpectralIndexCalculator.compute_red_index(R, G)

        elif spectral_index == SpectralIndex.LYC:
            # Convert wavelengths to band indices
            bands_needed = [630, 570]
            band_indices = [DataProcessingUtils.get_closest_band_index(wl) for wl in bands_needed]
            R630 = reflectance_matrix[:, :, band_indices[0]]
            R570 = reflectance_matrix[:, :, band_indices[1]]
            return SpectralIndexCalculator.compute_lycopene_index(R630, R570)

        elif spectral_index == SpectralIndex.INA:
            bands_needed = [760, 570]
            band_indices = [DataProcessingUtils.get_closest_band_index(wl) for wl in bands_needed]
            R760 = reflectance_matrix[:, :, band_indices[0]]
            R570 = reflectance_matrix[:, :, band_indices[1]]
            return SpectralIndexCalculator.compute_normalized_anthocyanin_index(R760, R570)

        elif spectral_index == SpectralIndex.ZMI:
            bands_needed = [750, 710]
            band_indices = [DataProcessingUtils.get_closest_band_index(wl) for wl in bands_needed]
            R750 = reflectance_matrix[:, :, band_indices[0]]
            R710 = reflectance_matrix[:, :, band_indices[1]]
            return SpectralIndexCalculator.compute_zarco_tejada_miller_index(R750, R710)

        elif spectral_index == SpectralIndex.GLI:
            R = reflectance_matrix[:, :, config.RED_BAND]
            G = reflectance_matrix[:, :, config.GREEN_BAND]
            B = reflectance_matrix[:, :, config.BLUE_BAND]
            return SpectralIndexCalculator.compute_green_leaf_index(G, R, B)

        elif spectral_index == SpectralIndex.GRVI:
            G = reflectance_matrix[:, :, config.GREEN_BAND]
            R = reflectance_matrix[:, :, config.RED_BAND]
            return SpectralIndexCalculator.compute_green_red_vegetation_index(G, R)

        else:
            raise ValueError(f"Unsupported Spectral Index: {spectral_index}")

    @staticmethod
    def calculate_selected_indexes(
        reflectance_matrix: Dict[str, np.ndarray],
        selected_indexes: List,  # Can be SpectralIndex or string
    ) -> Dict[str, np.ndarray]:
        """
        Calculates all selected spectral indexes based on the provided reflectance data.

        Args:
            reflectance_matrix (Dict[str, np.ndarray] or np.ndarray): Reflectance data.
                - If two-sided: Dict with keys 'sideA' and 'sideB' containing np.ndarrays.
                - If one-sided: np.ndarray.
            selected_indexes (List): List of selected spectral indexes to compute (SpectralIndex or string).

        Returns:
            Dict[str, np.ndarray]: Dictionary mapping index shortcuts to their computed values.
        """
        index_values = {}

        for index in selected_indexes:
            # Handle learned indexes (string format)
            if isinstance(index, str) and index.startswith('FS_Learned_'):
                # For learned indexes, we need the selected bands from config
                selected_bands = getattr(config, 'PATENT_BANDS', [])
                
                # Get mean reflectance for selected bands
                mean_reflectance_bands = []
                for band_idx in selected_bands:
                    mean_reflectance = DataProcessingUtils.compute_mean_reflectance(reflectance_matrix, [band_idx])
                    mean_reflectance_bands.extend(mean_reflectance)
                
                # Compute learned index using the simple combinations
                # For ML models, we use mean values; for DL models, this won't be called
                if config.MODEL_TYPE in ML_MODELS:
                    index_value = SpectralIndexCalculator._compute_learned_index_ml(
                        mean_reflectance_bands, index
                    )
                else:
                    # This shouldn't be called for DL models, but provide fallback
                    index_value = np.array([0.0])
                
                index_values[index] = index_value
                continue

            # Handle regular SpectralIndex enums
            if index == SpectralIndex.RED_INDEX:
                # Bands are already indices
                bands_needed = [config.RED_BAND, config.GREEN_BAND]
                mean_reflectance = DataProcessingUtils.compute_mean_reflectance(reflectance_matrix, bands_needed)
                R, G = mean_reflectance
                index_value = SpectralIndexCalculator.compute_red_index(R, G)
                index_values[index.shortcut] = index_value

            elif index == SpectralIndex.LYC:
                # Bands are specified in wavelengths; convert to indices
                wavelengths_needed = [630, 570]
                bands_needed = [DataProcessingUtils.get_closest_band_index(wl) for wl in wavelengths_needed]
                mean_reflectance = DataProcessingUtils.compute_mean_reflectance(reflectance_matrix, bands_needed)
                R630, R570 = mean_reflectance
                index_value = SpectralIndexCalculator.compute_lycopene_index(R630, R570)
                index_values[index.shortcut] = index_value

            elif index == SpectralIndex.INA:
                # Bands are specified in wavelengths; convert to indices
                wavelengths_needed = [760, 570]
                bands_needed = [DataProcessingUtils.get_closest_band_index(wl) for wl in wavelengths_needed]
                mean_reflectance = DataProcessingUtils.compute_mean_reflectance(reflectance_matrix, bands_needed)
                R760, R570 = mean_reflectance
                index_value = SpectralIndexCalculator.compute_normalized_anthocyanin_index(R760, R570)
                index_values[index.shortcut] = index_value

            elif index == SpectralIndex.ZMI:
                # Bands are specified in wavelengths; convert to indices
                wavelengths_needed = [750, 710]
                bands_needed = [DataProcessingUtils.get_closest_band_index(wl) for wl in wavelengths_needed]
                mean_reflectance = DataProcessingUtils.compute_mean_reflectance(reflectance_matrix, bands_needed)
                R750, R710 = mean_reflectance
                index_value = SpectralIndexCalculator.compute_zarco_tejada_miller_index(R750, R710)
                index_values[index.shortcut] = index_value

            elif index == SpectralIndex.GLI:
                # Bands are already indices
                bands_needed = [config.GREEN_BAND, config.RED_BAND, config.BLUE_BAND]
                mean_reflectance = DataProcessingUtils.compute_mean_reflectance(reflectance_matrix, bands_needed)
                G, R, B = mean_reflectance
                index_value = SpectralIndexCalculator.compute_green_leaf_index(G, R, B)
                index_values[index.shortcut] = index_value

            elif index == SpectralIndex.GRVI:
                # Bands are already indices
                bands_needed = [config.GREEN_BAND, config.RED_BAND]
                mean_reflectance = DataProcessingUtils.compute_mean_reflectance(reflectance_matrix, bands_needed)
                G, R = mean_reflectance
                index_value = SpectralIndexCalculator.compute_green_red_vegetation_index(G, R)
                index_values[index.shortcut] = index_value

            else:
                raise ValueError(f"Unsupported Spectral Index: {index}")

        return index_values

    @staticmethod
    def _compute_learned_index_ml(mean_reflectance_bands: List[float], learned_index_name: str) -> np.ndarray:
        """
        Compute learned index for ML models using mean reflectance values.
        
        Args:
            mean_reflectance_bands: List of mean reflectance values for selected bands
            learned_index_name: Name like 'FS_Learned_1'
            
        Returns:
            np.ndarray: Computed index value
        """
        if len(mean_reflectance_bands) < 2:
            return np.array([mean_reflectance_bands[0]] if mean_reflectance_bands else [0.0])
        
        # Extract the index number from the name
        try:
            index_num = int(learned_index_name.split('_')[-1])
        except (ValueError, IndexError):
            index_num = 1
        
        # Create simple combinations
        if index_num % 4 == 1:
            # NDVI-like
            band1 = mean_reflectance_bands[0]
            band2 = mean_reflectance_bands[1]
            return np.array([(band2 - band1) / (band2 + band1 + 1e-8)])
        elif index_num % 4 == 2:
            # Ratio
            band1 = mean_reflectance_bands[(index_num - 1) % len(mean_reflectance_bands)]
            band2 = mean_reflectance_bands[index_num % len(mean_reflectance_bands)]
            return np.array([band1 / (band2 + 1e-8)])
        elif index_num % 4 == 3:
            # Three-band combination
            if len(mean_reflectance_bands) >= 3:
                band1 = mean_reflectance_bands[(index_num - 1) % len(mean_reflectance_bands)]
                band2 = mean_reflectance_bands[index_num % len(mean_reflectance_bands)]
                band3 = mean_reflectance_bands[(index_num + 1) % len(mean_reflectance_bands)]
                return np.array([(band1 - band2) / (band1 + band2 + band3 + 1e-8)])
            else:
                band1 = mean_reflectance_bands[0]
                band2 = mean_reflectance_bands[1]
                return np.array([band1 / (band2 + 1e-8)])
        else:
            # Difference
            band1 = mean_reflectance_bands[(index_num - 1) % len(mean_reflectance_bands)]
            band2 = mean_reflectance_bands[index_num % len(mean_reflectance_bands)]
            return np.array([band1 - band2])



