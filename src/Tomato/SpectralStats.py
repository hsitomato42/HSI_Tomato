# SpectralStats.py

from typing import Optional
import numpy as np
import pandas as pd
import spectral
import os
from src.ImageData import ImagesPath, SingleSideImagesPath, TwoSideImagesPath, MaskHandler
from spectral import open_image
from src.utils.logger import get_logger


class SpectralStats:
    def __init__(self, images_paths: ImagesPath, id_in_image: int) -> None:
        self.reflectance_matrix: dict = {}  # Changed from np.ndarray to dict
        self.pixel_count: int = 0
        self.area: float = 0.0
        self.compute_reflectance_matrix(images_paths, id_in_image)
        self.compute_pixel_count(images_paths, id_in_image)
        # self.compute_area()



    def compute_reflectance_matrix(self, images_paths: ImagesPath, mask_id: int) -> None:
        if isinstance(images_paths, TwoSideImagesPath):
            # Initialize dictionary to hold reflectance matrices for both sides
            self.reflectance_matrix = {'sideA': None, 'sideB': None}
            
            for side in ['sideA', 'sideB']:
                hdr_path = images_paths.get_hdr_paths()[side]
                mask_csv_path = images_paths.get_mask_paths()[side]
                
                # print(f"Processing {side}...")
                reflectance = self._process_single_side(hdr_path, mask_csv_path, mask_id, side)
                self.reflectance_matrix[side] = reflectance
                # print(f"{side} reflectance matrix shape: {reflectance.shape}")
        elif isinstance(images_paths, SingleSideImagesPath):
            hdr_path = images_paths.get_hdr_paths()
            mask_csv_path = images_paths.get_mask_paths()
            
            # print("Processing single side...")
            reflectance = self._process_single_side(hdr_path, mask_csv_path, mask_id)
            self.reflectance_matrix = reflectance
            # print(f"Reflectance matrix shape: {reflectance.shape}")
        else:
            raise TypeError("Unsupported ImagesPath type provided.")

    def _process_single_side(self, hdr_path: str, mask_csv_path: str, mask_id: int, side: Optional[str] = None) -> np.ndarray:
        """
        Helper method to process a single side of the image.
        
        Args:
            hdr_path (str): Path to the HDR image file.
            mask_csv_path (str): Path to the mask CSV file.
            mask_id (int): ID to filter the mask.
            side (str, optional): Side identifier ('sideA' or 'sideB'). Defaults to None.
        
        Returns:
            np.ndarray: Processed reflectance matrix for the specified side.
        """
        # Load the hyperspectral image
        img = open_image(hdr_path)
        data = img.load()
        data = np.rot90(data[:, :, :], 3)  # Rotate if necessary

        # Normalize the segmented array
        data = MaskHandler.normalize_image(data)

        # Load the mask
        mask_df = pd.read_csv(mask_csv_path, index_col=0)
        mask = mask_df.values

        # Apply mask ID filtering
        mask_bool = mask == mask_id

        # Check if any pixels match the mask_id
        if not np.any(mask_bool):
            raise ValueError(f"Mask ID {mask_id} not found in the mask for {side if side else 'the image'}.")

        # Check dimensions
        if mask.shape != data.shape[:2]:
            raise ValueError("Mask and image dimensions do not match.")

        # Find the bounding box of the masked region
        rows = np.any(mask_bool, axis=1)
        cols = np.any(mask_bool, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop the data to the bounding box of the mask
        data_segment = data[rmin:rmax+1, cmin:cmax+1, :]

        # Apply the mask to zero out non-masked pixels within the cropped segment
        mask_segment = mask_bool[rmin:rmax+1, cmin:cmax+1]
        mask_segment_expanded = np.expand_dims(mask_segment, axis=2)
        data_segment_masked = np.where(mask_segment_expanded, data_segment, 0)
        return data_segment_masked

    def compute_pixel_count(self, images_paths: ImagesPath, mask_id: int) -> None:
        # Initializes pixel_count
        pixel_count = 0

        if isinstance(images_paths, TwoSideImagesPath):
            for side in ['sideA', 'sideB']:
                mask_path = images_paths.get_mask_paths()[side]

                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask file not found: {mask_path}")

                # Load mask
                mask = pd.read_csv(mask_path, header=None).values

                # Apply mask_id
                mask_bool = mask == mask_id

                # Count pixels
                count = np.sum(mask_bool)
                pixel_count += count
                # print(f"Side {side}: Found {count} pixels for mask ID {mask_id}.")
        elif isinstance(images_paths, SingleSideImagesPath):
            mask_path = images_paths.get_mask_paths()

            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Mask file not found: {mask_path}")

            # Load mask
            mask = pd.read_csv(mask_path, header=None).values

            # Apply mask_id
            mask_bool = mask == mask_id

            # Count pixels
            pixel_count = int(np.sum(mask_bool))
            # print(f"Found {pixel_count} pixels for mask ID {mask_id}.")
        else:
            raise TypeError("Unsupported ImagesPath type provided.")

        self.pixel_count = pixel_count

    def compute_area(self, pixel_size: float = 1.0) -> None:
        # Computes the area based on pixel count and pixel size.
        logger = get_logger("spectral_stats")
        self.area = self.pixel_count * pixel_size
        logger.debug(f"Computed area: {self.area} (pixel_size={pixel_size})")

    def __repr__(self) -> str:
        if isinstance(self.reflectance_matrix, dict):
            sides = ', '.join([f"{side}: {mat.shape}" for side, mat in self.reflectance_matrix.items()])
            return (
                f"SpectralStats(reflectance_matrix={{ {sides} }}, "
                f"pixel_count={self.pixel_count}, area={self.area})"
            )
        else:
            return (
                f"SpectralStats(reflectance_matrix_shape={self.reflectance_matrix.shape}, "
                f"pixel_count={self.pixel_count}, area={self.area})"
            )


def print_band(matrix, band):
    logger = get_logger("spectral_stats")
    if band < 0 or band >= matrix.shape[2]:
        logger.error("Invalid band number.")
        return
    band_matrix = matrix[:, :, band].astype(float)
    np.set_printoptions(precision=2, suppress=True)
    logger.debug(np.array2string(band_matrix, separator=', '))