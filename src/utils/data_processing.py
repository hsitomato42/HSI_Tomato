# utils/data_processing.py

import math
import random
import numpy as np
from typing import List, Tuple, Union
import pandas as pd
import copy

import src.Tomato as Tomato
from src.Tomato import Tomato
from src.Tomato import SpectralStats
import src.config as config
from src.config.enums import FeelingMissingValuesTechnique, ImagePathKind

class DataProcessingUtils:
    """
    Utility class for data processing functions related to spectral reflectance matrices.
    """
    
    @staticmethod
    def cut_bands_from_start(reflectance: Union[np.ndarray, dict], num_bands_to_cut: int) -> Union[np.ndarray, dict]:
        """
        Cuts bands from the beginning of the reflectance matrix to reduce memory usage.
        
        Args:
            reflectance (Union[np.ndarray, dict]): Reflectance matrix (height, width, bands) 
                                                   or dict for two-sided images.
            num_bands_to_cut (int): Number of bands to cut from the beginning (0-204).
        
        Returns:
            Union[np.ndarray, dict]: Reflectance matrix with reduced bands.
        """
        if num_bands_to_cut == 0:
            return reflectance
            
        if num_bands_to_cut < 0 or num_bands_to_cut > 204:
            raise ValueError(f"num_bands_to_cut must be between 0 and 204, got {num_bands_to_cut}")
        
        if isinstance(reflectance, dict):
            # Two-sided images
            cut_reflectance = {}
            for side, matrix in reflectance.items():
                if matrix is not None:
                    cut_reflectance[side] = matrix[:, :, num_bands_to_cut:]
                else:
                    cut_reflectance[side] = None
            return cut_reflectance
        else:
            # Single-sided images
            return reflectance[:, :, num_bands_to_cut:]

    @staticmethod
    def select_bands(reflectance: np.ndarray, selected_bands: List[int]) -> np.ndarray:
        """
        Selects specific spectral bands from the reflectance matrix.

        Args:
            reflectance (np.ndarray): Reflectance matrix (height, width, bands).
            selected_bands (List[int]): Indices of bands to select.

        Returns:
            np.ndarray: Reflectance matrix with selected bands.
        """
        return reflectance[:, :, selected_bands]

    @staticmethod
    def pad_reflectance_matrix(reflectance: np.ndarray, max_height: int, max_width: int) -> Tuple[np.ndarray, int, int]:
        """
        Pads the reflectance matrix with zeros to match the target dimensions, centering
        the original image within the padded matrix.

        The original reflectance matrix is placed in the center of a zero-padded array
        of shape (max_height, max_width, bands). If the difference in dimensions is odd,
        the original image will be as centered as possible.

        Args:
            reflectance (np.ndarray): Reflectance matrix of shape (height, width, bands).
            max_height (int): Target height.
            max_width (int): Target width.

        Returns:
            Tuple[np.ndarray, int, int]: 
                - Padded reflectance matrix
                - top_pad: The vertical padding added above the original image.
                - left_pad: The horizontal padding added to the left of the original image.
        """
        height, width, bands = reflectance.shape

        top_pad = (max_height - height) // 2
        left_pad = (max_width - width) // 2

        padded = np.zeros((max_height, max_width, bands), dtype=reflectance.dtype)
        padded[top_pad:top_pad+height, left_pad:left_pad+width, :] = reflectance

        return padded


    @staticmethod
    def normalize_reflectance(reflectance: np.ndarray) -> np.ndarray:
        """
        Normalizes the reflectance matrix to the range [0, 1].

        Args:
            reflectance (np.ndarray): Reflectance matrix.

        Returns:
            np.ndarray: Normalized reflectance matrix.
        """
        # # other option:
        # return (reflectance - np.min(reflectance)) / (np.max(reflectance) - np.min(reflectance))

        # Avoid division by zero
        image_min = reflectance.min(axis=(0,1), keepdims=True)
        image_max = reflectance.max(axis=(0,1), keepdims=True)
        normalized = (reflectance - image_min) / (image_max - image_min + 1e-8)
        return normalized

    @staticmethod
    def get_max_dimensions(tomatoes: List[Tomato]) -> Tuple[int, int]:
        """
        Determines the maximum height and width among all reflectance matrices.
        Supports both single-sided and two-sided ImagesPath instances.

        Args:
            tomatoes (List): List of Tomato objects.

        Returns:
            Tuple[int, int]: (max_height, max_width)
        """
        max_height = 0
        max_width = 0
        for tomato in tomatoes:
            reflectance_matrix = tomato.spectral_stats.reflectance_matrix
            
            if isinstance(reflectance_matrix, dict):
                # Two-sided images
                for side, matrix in reflectance_matrix.items():
                    height, width, _ = matrix.shape
                    if height == 512 or width == 512:
                        raise ValueError(f'Problematic tomato: {tomato.harvest} ID in photo: {tomato.id_in_photo} Side: {side}')
                    if height > max_height:
                        max_height = height
                    if width > max_width:
                        max_width = width
            else:
                # Single-sided images
                height, width, _ = reflectance_matrix.shape
                if height == 512 or width == 512:
                    raise ValueError(f'Problematic tomato: {tomato.harvest} ID in photo: {tomato.id_in_photo} Side: {side}')
                if height > max_height:
                    max_height = height
                if width > max_width:
                    max_width = width
        return max_height, max_width


    @staticmethod
    def compute_mean_reflectance(reflectance_matrix: Union[np.ndarray, dict], selected_bands: List[int]) -> np.ndarray:
        """
        Computes the mean reflectance for each selected band, excluding zero-padded areas.
    
        Args:
            reflectance_matrix (np.ndarray): Reflectance matrix of shape (H, W, 204).
            selected_bands (List[int]): List of band indices to compute mean reflectance.
    
        Returns:
            np.ndarray: 1D array containing mean reflectance for each selected band.
        """
        if isinstance(reflectance_matrix, dict):
            return ( (DataProcessingUtils.compute_mean_reflectance(reflectance_matrix['sideA'], selected_bands) + DataProcessingUtils.compute_mean_reflectance(reflectance_matrix['sideB'], selected_bands) ) / 2)
        # Select the desired bands
        selected_reflectance = reflectance_matrix[:, :, selected_bands]  # Shape: (H, W, selected_bands)
        
        # Create a mask to exclude zero-padded areas
        # Assuming zero-padded areas have reflectance values of 0 across all selected bands
        # A pixel is considered valid if **any** of the selected bands have a reflectance > 0
        valid_mask = np.any(selected_reflectance > 0, axis=2)  # Shape: (H, W)
        
        mean_reflectance = []
        for b in range(selected_reflectance.shape[2]):
            band_data = selected_reflectance[:, :, b]
            # Mask to exclude zero-padded areas
            valid_band = band_data[valid_mask]
            if valid_band.size == 0:
                raise ValueError(f"No valid pixels found for band {b}.")
            else:
                mean_val = np.mean(valid_band)
            mean_reflectance.append(mean_val)
        
        return np.array(mean_reflectance)  # Shape: (selected_bands,)
    
    @staticmethod
    def generate_band_pairs(selected_bands: List[int]) -> List[Tuple[int, int]]:
        """
        Generates all unique band pairs from the selected bands without considering order,
        or uses predefined band pairs from config if specified.

        Args:
            selected_bands (List[int]): List of selected band indices. Can be None for feature selection mode.

        Returns:
            List[Tuple[int, int]]: List of unique band index pairs.
        """
        # Handle None case (feature selection mode)
        if selected_bands is None:
            # Use all available bands for feature selection mode
            # Account for band cutting if configured
            num_bands_cut = getattr(config, 'CUT_BANDS_FROM_START', 0)
            total_bands = 204 - num_bands_cut
            selected_bands = list(range(total_bands))
            print(f"[DataProcessingUtils] Feature selection mode: using all {total_bands} bands for NDSI pairs (after cutting {num_bands_cut} bands)")
        
        if hasattr(config, 'NDSI_BAND_PAIRS') and config.NDSI_BAND_PAIRS:
            # Validate that all specified band pairs are within selected_bands
            valid_pairs = []
            selected_bands_set = set(selected_bands)
            for pair in config.NDSI_BAND_PAIRS:
                if pair[0] in selected_bands_set and pair[1] in selected_bands_set:
                    valid_pairs.append(pair)
                else:
                    print(f"Warning: Band pair {pair} is not within the selected bands {selected_bands}. Skipping this pair.")
            return valid_pairs
        else:
            # Default behavior: Generate all unique band pairs
            band_pairs = []
            n = len(selected_bands)
            for i in range(n):
                for j in range(i + 1, n):
                    band_pairs.append((selected_bands[i], selected_bands[j]))
            return band_pairs

    @staticmethod
    def compute_ndsi(mean_reflectance: np.ndarray, selected_bands: List[int]) -> np.ndarray:
        """
        Computes the NDSI for specified band pairs or all unique band pairs if none are specified.

        Args:
            mean_reflectance (np.ndarray): 1D array containing mean reflectance for each selected band.
            selected_bands (List[int]): List of selected band indices.

        Returns:
            np.ndarray: 1D array containing NDSI values for each specified band pair.
        """
        band_pairs = DataProcessingUtils.generate_band_pairs(selected_bands)
        ndsi_values = []
        for b1, b2 in band_pairs:
            index_b1 = selected_bands.index(b1)
            index_b2 = selected_bands.index(b2)
            ndsi = (mean_reflectance[index_b1] - mean_reflectance[index_b2]) / (mean_reflectance[index_b1] + mean_reflectance[index_b2] + 1e-8)
            # ndsi_values.append(ndsi)
            ndsi_values.append(float(ndsi))
        return np.array(ndsi_values)
        return ndsi_values
    @staticmethod
    def fill_missing_values(Y: np.ndarray, quality_attributes: List[str]) -> np.ndarray:
        """
        Fills missing values in Y with the mean of each quality attribute and prints the number of imputations.

        Args:
            Y (np.ndarray): Target matrix with possible NaN or None values.
            quality_attributes (List[str]): List of quality attribute names.

        Returns:
            np.ndarray: Y with missing values filled.
        """
        if Y.shape[1] != len(quality_attributes):
            print("Error: The number of quality attributes does not match the number of Y columns.")
            raise ValueError("Mismatch between quality attributes and Y columns.")
        
        Y_filled = Y.copy().astype(float)
        
        for idx, attr in enumerate(quality_attributes):
            # Identify missing values (NaN)
            missing_mask = np.isnan(Y_filled[:, idx])
            num_missing = np.sum(missing_mask)
            if num_missing > 0:
                # Compute mean excluding NaN
                mean_value = np.nanmean(Y_filled[:, idx])
                if np.isnan(mean_value):
                    raise ValueError(f"All values are missing for attribute '{attr}'.")
                # Fill missing values with the mean
                if config.FEELING_MISSING_VALUES_TECHNIQUE == FeelingMissingValuesTechnique.MEAN:
                    Y_filled[missing_mask, idx] = mean_value
                    print(f"Imputed {num_missing} missing values for attribute '{attr}' with mean value {mean_value:.4f}.")
                elif config.FEELING_MISSING_VALUES_TECHNIQUE == FeelingMissingValuesTechnique.NAN:
                    Y_filled[missing_mask, idx] = np.nan
                    print(f"Imputed {num_missing} missing values for attribute '{attr}' with nan values.")
                else:
                    raise ValueError(f"Need to implement the data filling technique for: {config.FEELING_MISSING_VALUES_TECHNIQUE}")
        
        return Y_filled
    
    @staticmethod
    def assign_area_to_tomatoes(tomatoes: List[Tomato], area_excel_path: str) -> None:
        """
        Assigns area values from the 'Area' sheet to each Tomato's spectral_stats.area based on _id.
        Supports both single-sided and two-sided ImagesPath kinds as specified in config.IMAGES_PATH_KIND.

        Args:
            tomatoes (List[Tomato]): List of Tomato objects.
            area_excel_path (str, optional): Path to the Excel file containing area data.
                                            Defaults to config.SPECTRAL_STATS_FILENAME.
        """
        # Read the 'Area' sheet from the Excel file
        try:
            area_df = pd.read_excel(area_excel_path, sheet_name='Area')
        except Exception as e:
            raise FileNotFoundError(f"Failed to read 'Area' sheet from {area_excel_path}: {e}")

        # Validate the structure of the 'Area' sheet
        if area_df.shape[1] < 2:
            raise ValueError("The 'Area' sheet must have at least two columns: '_id' and 'area(cm^2)'.")

        # Rename columns for clarity if necessary
        # Assuming the first column is '_id' and the second is 'area(cm^2)'
        area_df = area_df.iloc[:, :2]  # Consider only the first two columns
        area_df.columns = ['_id', 'area_cm2']

        # Create a mapping from _id to area
        id_to_area = pd.Series(area_df.area_cm2.values, index=area_df._id).to_dict()

        # Determine the ImagesPath kind from config
        images_path_kind = config.IMAGES_PATH_KIND
        if images_path_kind not in  [ImagePathKind.ONE_SIDE, ImagePathKind.TWO_SIDES]:
            raise ValueError("config.IMAGES_PATH_KIND must be either 'oneSide' or 'twoSides'.")

        # Assign area to each tomato based on the ImagesPath kind
        for tomato in tomatoes:
            tomato_id = tomato._id  # Adjust attribute name if different

            if images_path_kind == ImagePathKind.ONE_SIDE:
                # For single-sided images, map _id to ceil(_id / 2)
                mapped_id = math.ceil(tomato_id / 2)
                area = id_to_area.get(mapped_id, None)
                if area is not None:
                    tomato.spectral_stats.area = area
                    print(f"Assigned area {area} to tomato with _id {tomato_id} (mapped_id {mapped_id}).")
                else:
                    print(f"No area found for mapped_id {mapped_id} (tomato _id {tomato_id}).")
                    tomato.spectral_stats.area = None  # Or assign a default value if preferred
            elif images_path_kind == ImagePathKind.TWO_SIDES:
                # For two-sided images, map _id directly
                area = id_to_area.get(tomato_id, None)
                if area is not None:
                    tomato.spectral_stats.area = area
                    print(f"Assigned area {area} to tomato with _id {tomato_id}.")
                else:
                    raise ValueError(f"No area found for tomato with _id {tomato_id}.")
   
    
    @staticmethod
    def get_closest_band_index(wavelength: float) -> int:
        """
        Finds the closest band index for a given wavelength.
        
        Args:
            wavelength (float): The target wavelength in nm.
        
        Returns:
            int: The band index closest to the target wavelength.
        """
        closest_band = min(config.BAND_ID_TO_WAVELENGTH.keys(), key=lambda k: abs(config.BAND_ID_TO_WAVELENGTH[k] - wavelength))
        return closest_band

    
    @staticmethod
    def create_augmented_and_padded_tomatoes(
        tomatoes: List[Tomato],
        augment_times: int
    ) -> List[Tomato]:
        """
        Creates augmented Tomato objects by shifting the reflectance matrix of each original tomato.

        This function:
        1. For each original tomato, retrieves its reflectance matrix.
        2. Determines valid shifting ranges based on the target padded size.
        3. For each augmentation, chooses a (dy, dx) shift.
        - For the first augmentation (if augment_times > 0), we do not shift (dy=0, dx=0),
            meaning we just pad and center the tomato.
        - For subsequent augmentations, random shifts are chosen.
        4. Pads the reflectance matrix directly with the offset applied, so the original reflectance
        is placed not in the exact center, but shifted by (dy, dx).

        Returns:
            List[Tomato]: A list of newly created Tomato objects representing augmented samples.
        """

        max_h, max_w = config.MAX_2D_DIMENSIONS
        augmented_tomatoes = []
        tomatoes_counter = 0

        for tomato in tomatoes:
            reflectance_matrix = tomato.spectral_stats.reflectance_matrix
            if isinstance(reflectance_matrix, dict):
                # For DL models, just use sideA
                if 'sideA' in reflectance_matrix:
                    reflectance_matrix = reflectance_matrix['sideA']
                elif 'sideB' in reflectance_matrix:
                    reflectance_matrix = reflectance_matrix['sideB']
                else:
                    raise ValueError(f"No valid side found in reflectance_matrix for tomato {tomato._id}")
            orig_h, orig_w, _ = reflectance_matrix.shape

            # Center positions if no shift
            base_top_pad = (max_h - orig_h) // 2
            base_left_pad = (max_w - orig_w) // 2

            # Calculate valid shifting ranges
            min_shift_y = -base_top_pad
            max_shift_y = max_h - orig_h - base_top_pad
            min_shift_x = -base_left_pad
            max_shift_x = max_w - orig_w - base_left_pad

            used_shifts = set()

            for i in range(augment_times + 1):
                if i == 0:
                    # For the first augmentation, do not shift at all
                    dy, dx = 0, 0
                else:
                    # Random unique shifts for subsequent augmentations
                    while True:
                        dy = random.randint(min_shift_y, max_shift_y)
                        dx = random.randint(min_shift_x, max_shift_x)
                        if (dy, dx) not in used_shifts:
                            used_shifts.add((dy, dx))
                            break

                # Given (dy, dx), compute the actual top_pad and left_pad
                top_pad = base_top_pad + dy
                left_pad = base_left_pad + dx

                # Create the padded (and possibly shifted) reflectance matrix
                shifted_reflectance = np.zeros((max_h, max_w, reflectance_matrix.shape[2]), dtype=reflectance_matrix.dtype)
                shifted_reflectance[top_pad:top_pad+orig_h, left_pad:left_pad+orig_w, :] = reflectance_matrix

                # Create a shallow copy of the tomato
                new_tomato = copy.copy(tomato)
                # Create a shallow copy of the spectral_stats
                new_spectral_stats = copy.copy(tomato.spectral_stats)
                new_spectral_stats.reflectance_matrix = shifted_reflectance

                new_tomato.spectral_stats = new_spectral_stats

                augmented_tomatoes.append(new_tomato)
                tomatoes_counter += 1
        print(tomatoes_counter)

        return augmented_tomatoes
    
    @staticmethod
    def apply_band_cutting_to_dataset(tomatoes: List[Tomato], num_bands_to_cut: int) -> None:
        """
        Applies band cutting to all tomatoes in the dataset to reduce memory usage.
        This modifies the tomatoes in-place.
        
        Args:
            tomatoes (List[Tomato]): List of Tomato objects.
            num_bands_to_cut (int): Number of bands to cut from the beginning (0-204).
        """
        if num_bands_to_cut == 0:
            return
            
        print(f"\n📉 Applying band cutting: removing first {num_bands_to_cut} bands from all tomatoes...")
        
        for i, tomato in enumerate(tomatoes):
            if tomato.spectral_stats and tomato.spectral_stats.reflectance_matrix is not None:
                # Apply band cutting to reflectance matrix
                tomato.spectral_stats.reflectance_matrix = DataProcessingUtils.cut_bands_from_start(
                    tomato.spectral_stats.reflectance_matrix, 
                    num_bands_to_cut
                )
                
                # Also update statistic stats if they exist
                if hasattr(tomato, 'statistic_stats') and tomato.statistic_stats:
                    if hasattr(tomato.statistic_stats, 'mean') and tomato.statistic_stats.mean is not None:
                        # Cut mean array
                        if isinstance(tomato.statistic_stats.mean, np.ndarray):
                            tomato.statistic_stats.mean = tomato.statistic_stats.mean[num_bands_to_cut:]
                    
                    if hasattr(tomato.statistic_stats, 'std') and tomato.statistic_stats.std is not None:
                        # Cut std array
                        if isinstance(tomato.statistic_stats.std, np.ndarray):
                            tomato.statistic_stats.std = tomato.statistic_stats.std[num_bands_to_cut:]
                    
                    if hasattr(tomato.statistic_stats, 'median') and tomato.statistic_stats.median is not None:
                        # Cut median array
                        if isinstance(tomato.statistic_stats.median, np.ndarray):
                            tomato.statistic_stats.median = tomato.statistic_stats.median[num_bands_to_cut:]
        
        print(f"✅ Band cutting applied. New spectral dimension: {204 - num_bands_to_cut} bands")
    
    @staticmethod
    def adjust_band_indices_after_cutting(band_indices: List[int], num_bands_cut: int) -> List[int]:
        """
        Adjusts band indices after cutting bands from the beginning.
        
        Args:
            band_indices (List[int]): Original band indices (0-based).
            num_bands_cut (int): Number of bands cut from the beginning.
        
        Returns:
            List[int]: Adjusted band indices, filtering out invalid ones.
        """
        if num_bands_cut == 0:
            return band_indices
        
        adjusted_indices = []
        for idx in band_indices:
            if idx >= num_bands_cut:
                # Adjust the index by subtracting the number of cut bands
                adjusted_indices.append(idx - num_bands_cut)
            else:
                print(f"⚠️ Warning: Band index {idx} is invalid after cutting {num_bands_cut} bands")
        
        return adjusted_indices
    
    @staticmethod
    def adjust_band_pairs_after_cutting(band_pairs: List[Tuple[int, int]], num_bands_cut: int) -> List[Tuple[int, int]]:
        """
        Adjusts band pairs after cutting bands from the beginning.
        
        Args:
            band_pairs (List[Tuple[int, int]]): Original band pairs.
            num_bands_cut (int): Number of bands cut from the beginning.
        
        Returns:
            List[Tuple[int, int]]: Valid adjusted band pairs.
        """
        if num_bands_cut == 0:
            return band_pairs
        
        adjusted_pairs = []
        for b1, b2 in band_pairs:
            if b1 >= num_bands_cut and b2 >= num_bands_cut:
                adjusted_pairs.append((b1 - num_bands_cut, b2 - num_bands_cut))
            else:
                print(f"⚠️ Warning: Band pair ({b1}, {b2}) is invalid after cutting {num_bands_cut} bands")
        
        return adjusted_pairs
