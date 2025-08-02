import os
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from spectral import *
from spectral.image import ImageArray
import src.config as config
from tabulate import tabulate
class MaskHandler:
    @staticmethod
    def apply_mask_on_image(image_path: str, mask_csv_path: str) -> None:
        # Applies a mask to an image and displays the result with masked areas highlighted.
        
        # Load the image
        image = Image.open(image_path)
        image = np.array(image)
        
        # Load the mask, assuming headers are in the first row and first column
        mask = pd.read_csv(mask_csv_path, index_col=0)
        mask = mask.values  # Convert DataFrame to numpy array
        
        # Check if mask and image dimensions match
        if mask.shape != image.shape[:2]:
            raise ValueError("Mask and image dimensions do not match")
        
        # Create an RGBA version of the image
        image_rgba = np.concatenate([image, 255 * np.ones((*image.shape[:2], 1), dtype=np.uint8)], axis=2)
        
        # Create a semi-transparent red mask
        red_mask = np.zeros_like(image_rgba)
        red_mask[..., 0] = 255  # Red channel
        red_mask[..., 3] = 128  # Alpha channel for transparency

        # Apply the red mask where mask values are non-zero
        image_rgba[mask != 0] = red_mask[mask != 0]

        # Display the masked image
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgba)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show(block=config.IMAGES_BLOCKER)
        plt.close()

    @staticmethod
    def show_normal_image(image_path: str) -> None:
        # Displays the original, unmasked image.
        
        # Load the image
        image = Image.open(image_path)
        image = np.array(image)

        # Display the image
        plt.imshow(image)
        plt.axis('off')  # Hide axes
        plt.show(block=config.IMAGES_BLOCKER)
        plt.close()

    @staticmethod
    def get_selected_bands_image(data: ImageArray, selected_bands: tuple[int, int, int]) -> np.ndarray:
        # Extracts specified bands from hyperspectral data.
        
        # Extract the specified bands
        band_image = data[:, :, selected_bands]
        return band_image

    @staticmethod
    def show_hyperspectral_image(hdr_path: str, selected_bands: tuple[int, int, int]) -> None:
        # Displays a hyperspectral image using selected bands.
        
        # Load the hyperspectral image
        img = open_image(hdr_path)
        data = img.load()
        data = np.rot90(data[:, :, :], 3)
        image = MaskHandler.get_selected_bands_image(data, selected_bands)
         # Normalize the RGB image for proper display
        image = MaskHandler.normalize_image(image)
        # Display the image
        plt.figure()
        plt.imshow(image)
        plt.axis('off')  # Hide the axes for a cleaner presentation
        plt.show(block=config.IMAGES_BLOCKER)
        plt.close()

    @staticmethod
    def show_image_segment(image_path: str, mask_csv_path: str, mask_id: int) -> None:
        # Displays a specific segment of an image based on the mask ID.
        
        # Load the image
        image = Image.open(image_path)
        image = np.array(image)
        
        # Load the mask
        mask = pd.read_csv(mask_csv_path, index_col=0)
        mask = mask.values  # Convert DataFrame to numpy array
        
        # Apply mask ID filtering
        mask = mask == mask_id
        
        # Check if mask and image dimensions match
        if mask.shape != image.shape[:2]:
            raise ValueError("Mask and image dimensions do not match")

        # Find the bounding box of the masked region
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows) or not np.any(cols):
            raise ValueError("Mask ID not found in the mask.")
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Crop the image to the bounding box of the mask
        image_segment = image[rmin:rmax+1, cmin:cmax+1]

        # Display the segmented image
        plt.figure(figsize=(10, 10))
        plt.imshow(image_segment)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show(block=config.IMAGES_BLOCKER)
        plt.close()

    @staticmethod
    def show_hyperspectral_image_segment(
        hdr_path: str, mask_csv_path: str, mask_id: int, selected_bands: tuple[int, int, int]
    ) -> None:
        # Displays a segmented hyperspectral image based on the mask ID and selected bands.
        
        # Load the hyperspectral image
        img = open_image(hdr_path)
        data = img.load()
        # Rotate the image to fit
        data = np.rot90(data[:, :, :], 3)  

        # Load the mask
        mask_df = pd.read_csv(mask_csv_path, index_col=0)
        mask = mask_df.values

        # Apply mask ID filtering
        mask_bool = mask == mask_id

        # Check if any pixels match the mask_id
        if not np.any(mask_bool):
            raise ValueError(f"Mask ID {mask_id} not found in the mask.")

        # Check dimensions
        if mask.shape != data.shape[:2]:
            raise ValueError("Mask and image dimensions do not match")

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

        # Get the selected bands for RGB visualization
        rgb_image = MaskHandler.get_selected_bands_image(data_segment_masked, selected_bands)

        # Normalize the RGB image for proper display
        rgb_image_normalized = MaskHandler.normalize_image(rgb_image)

        # Display the segmented hyperspectral image
        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_image_normalized)
        plt.axis('off')  # Hide axes
        plt.show(block=config.IMAGES_BLOCKER)
        plt.close()

    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        # Normalizes the image data to the range [0, 1] for display purposes.
        
        # return image / np.max(image)
        
        ### second version:  
        # Avoid division by zero
        image_min = image.min(axis=(0,1), keepdims=True)
        image_max = image.max(axis=(0,1), keepdims=True)
        # Handle the case where max == min for any channel
        difference = image_max - image_min
        # Replace zeros with small value to avoid division by zero
        difference = np.where(difference == 0, 1e-8, difference)
        normalized = (image - image_min) / difference
        return normalized

    @staticmethod
    def get_segmented_hyperspectral_array(
        hdr_path: str, 
        mask_csv_path: str, 
        mask_id: int, 
        selected_bands: tuple[int, ...] = None
    ) -> np.ndarray:
        # Extracts and returns a NumPy array of a segmented hyperspectral image corresponding to a specific mask ID.
        
        # Load the hyperspectral image
        img = open_image(hdr_path)
        data = img.load()
        data = np.rot90(data[:, :, :], 3)  # Rotate if necessary

        # Load the mask
        mask_df = pd.read_csv(mask_csv_path, index_col=0)
        mask = mask_df.values

        # Apply mask ID filtering
        mask_bool = mask == mask_id

        # Check if any pixels match the mask_id
        if not np.any(mask_bool):
            raise ValueError(f"Mask ID {mask_id} not found in the mask.")

        # Check dimensions
        if mask.shape != data.shape[:2]:
            raise ValueError("Mask and image dimensions do not match")

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

        # Select the specified bands or all bands
        if selected_bands is not None:
            band_indices = selected_bands
            try:
                segmented_array = data_segment_masked[:, :, band_indices]
            except IndexError:
                raise IndexError("One or more selected bands are out of the hyperspectral data range.")
        else:
            segmented_array = data_segment_masked  # All bands

        # Normalize the segmented array
        segmented_array_normalized = MaskHandler.normalize_image(segmented_array)

        return segmented_array_normalized
