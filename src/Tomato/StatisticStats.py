import numpy as np
import pandas as pd
from spectral import open_image
from src.ImageData.MaskHendler import MaskHandler
from typing import Union

class StatisticStats:
    def __init__(self, reflectance_matrix: Union[np.ndarray, dict]) -> None:
        """
        Computes per-band median, mean, and standard deviation values based on valid tomato pixels,
        where valid pixels are determined as those with a non-zero spectrum. If two-sided images are
        provided (as a dict), the valid pixels from each side are pooled together. In the end, for
        each statistical measure an array is produced of length equal to the number of bands, with
        each value corresponding to that band.
        
        Args:
            reflectance_matrix (np.ndarray or dict): For single-sided images, a NumPy array of shape
                (height, width, num_bands). For two-sided images, a dict with keys 'sideA' and 'sideB'
                each containing a corresponding reflectance matrix.
        """
        def _get_valid_pixels_from_matrix(matrix: np.ndarray) -> np.ndarray:
            """
            Given a reflectance matrix, return a 2D array (N_valid x num_bands) containing the spectral
            values of pixels that are valid (i.e. have any non-zero value across the bands).
            
            Args:
                matrix (np.ndarray): A 3D numpy array of shape (height, width, num_bands).
            
            Returns:
                np.ndarray: Array where each row is one pixel's spectrum (for valid pixels).
            """
            # Create a mask for valid pixels: True for any pixel with at least one non-zero band.
            valid_mask = np.any(matrix != 0, axis=2)  # shape: (height, width)
            # Indexing with a boolean mask returns an array of shape (N_valid, num_bands)
            return matrix[valid_mask]

        # Process based on whether we have single- or two-sided images.
        if isinstance(reflectance_matrix, dict):
            valid_pixels_list = []
            for side in ['sideA', 'sideB']:
                side_matrix = reflectance_matrix.get(side)
                if side_matrix is None:
                    continue
                side_valid_pixels = _get_valid_pixels_from_matrix(side_matrix)
                valid_pixels_list.append(side_valid_pixels)
            all_valid_pixels = (
                np.concatenate(valid_pixels_list, axis=0) if valid_pixels_list else np.array([])
            )
            # Get number of bands from any available side
            first_side_matrix = next(iter(reflectance_matrix.values()))
            num_bands = first_side_matrix.shape[2]
        else:
            all_valid_pixels = _get_valid_pixels_from_matrix(reflectance_matrix)
            num_bands = reflectance_matrix.shape[2]

        # Initialize statistics arrays with one value per band.
        self.median = []
        self.mean = []
        self.std = []

        if all_valid_pixels.size == 0:
            # No valid pixels, defaulting statistics to 0 for each band.
            self.median = [0.0] * num_bands
            self.mean = [0.0] * num_bands
            self.std = [0.0] * num_bands
        else:
            # Process each band separately.
            for band in range(num_bands):
                band_values = all_valid_pixels[:, band]
                if band_values.size > 0:
                    self.std.append(float(np.nanstd(band_values)))
                    self.median.append(float(np.nanmedian(band_values)))
                    self.mean.append(float(np.nanmean(band_values)))
                 
                else:
                    raise ValueError(f"No valid pixels found for band {band}") # for debugging

    def _extract_valid_pixels_for_side(self, hdr_path: str, mask_csv_path: str, mask_id: int) -> np.ndarray:
        """
        Loads the hyperspectral image and corresponding mask for one side, applies the mask ID,
        and returns a 2D array of valid pixel values (shape: [N_valid, num_bands]).
        
        Args:
            hdr_path (str): Path to the HDR image file.
            mask_csv_path (str): Path to the mask CSV file.
            mask_id (int): The mask ID corresponding to the tomato.
        
        Returns:
            np.ndarray: Array where each row represents the spectral values (across bands) of one valid pixel.
        """
        # Load and process the hyperspectral image.
        img = open_image(hdr_path)
        data = img.load()
        # data = np.rot90(data[:, :, :], 3)  # Rotate if necessary
        # data = MaskHandler.normalize_image(data)
        
        # Load the mask from CSV (assuming index_col=0 format).
        mask_df = pd.read_csv(mask_csv_path, index_col=0)
        mask = mask_df.values

        if mask.shape != data.shape[:2]:
            raise ValueError("Mask and image dimensions do not match")

        # Create a boolean mask with the provided mask_id.
        mask_bool = (mask == mask_id)
        if not np.any(mask_bool):
            raise ValueError(f"Mask ID {mask_id} not found in the mask at {mask_csv_path}.")

        # Determine the bounding box of the valid region.
        rows = np.any(mask_bool, axis=1)
        cols = np.any(mask_bool, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop the image data and mask to the bounding box.
        cropped_data = data[rmin:rmax+1, cmin:cmax+1, :]
        cropped_mask = mask_bool[rmin:rmax+1, cmin:cmax+1]

        # Extract the valid pixels. Indexing the 3D array with the 2D boolean mask returns an array
        # of shape (N_valid, num_bands) where each row corresponds to one pixel's spectrum.
        valid_pixels = cropped_data[cropped_mask]
        return valid_pixels
