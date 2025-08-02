# utils/segmentation.py

import os
from typing import List
from src.Tomato import *
from src.ImageData import *
import src.config as config


def segmentation_checks(loaded_tomatoes: List[Tomato], only_one_from_harvest: bool = True) -> None:
    """
    Performs segmentation checks on the loaded tomatoes dataset.
    
    Args:
        loaded_tomatoes (List[Tomato]): List of Tomato objects to process.
        only_one_from_harvest (bool): If True, processes only one tomato per harvest.
    """
    rgb_bands = (config.RED_BAND, config.GREEN_BAND, config.BLUE_BAND)
    previous_harvest = 'harvest_0'
    # Ensure the directory for saving masked images exists
    masked_images_dir = os.path.join('Data', 'masked_images')
    os.makedirs(masked_images_dir, exist_ok=True)

    # Apply MaskHandler functions to each Tomato object
    for tomato in loaded_tomatoes:
        if only_one_from_harvest:
            if previous_harvest != tomato.harvest:
                previous_harvest = tomato.harvest
                print(f"\n{tomato.harvest}:")
            else:
                continue
        print(f"\nProcessing Tomato ID: {tomato._id}, Name: {tomato.name}")

        # Extract necessary paths
        data_paths = tomato.data_paths
        images_paths = data_paths.images_paths

        # Determine if the Tomato object is single-sided or two-sided
        if isinstance(images_paths, SingleSideImagesPath):
            # For single-sided images
            image_path = images_paths.get_png_paths()
            mask_csv_path = images_paths.get_mask_paths()

            # Show normal image
            print("Displaying normal image...")
            MaskHandler.show_normal_image(image_path)

            # Apply mask on image
            print("Applying mask on image...")
            MaskHandler.apply_mask_on_image(image_path, mask_csv_path)

            # Show image segment for a specific mask ID (e.g., mask_id=1)
            mask_id = tomato.id_in_photo
            print(f"Displaying image segment for mask ID {mask_id}...")
            MaskHandler.show_image_segment(image_path, mask_csv_path, mask_id)

        elif isinstance(images_paths, TwoSideImagesPath):
            # For two-sided images
            sideA_image_path = images_paths.get_png_paths()['sideA']
            sideA_mask_path = images_paths.get_mask_paths()['sideA']
            sideB_image_path = images_paths.get_png_paths()['sideB']
            sideB_mask_path = images_paths.get_mask_paths()['sideB']

            # Show normal images
            print("Displaying normal image for Side A...")
            MaskHandler.show_normal_image(sideA_image_path)
            print("Displaying normal image for Side B...")
            MaskHandler.show_normal_image(sideB_image_path)

            # Apply masks on images
            print("Applying mask on Side A image...")
            MaskHandler.apply_mask_on_image(sideA_image_path, sideA_mask_path)
            print("Applying mask on Side B image...")
            MaskHandler.apply_mask_on_image(sideB_image_path, sideB_mask_path)

            # Show image segments for specific mask IDs (e.g., mask_id=1 for both sides)
            mask_id = tomato.id_in_photo
            print(f"Displaying image segment for mask ID {mask_id} on Side A...")
            MaskHandler.show_image_segment(sideA_image_path, sideA_mask_path, mask_id)
            print(f"Displaying image segment for mask ID {mask_id} on Side B...")
            MaskHandler.show_image_segment(sideB_image_path, sideB_mask_path, mask_id)

        else:
            print("Unknown ImagesPath type. Skipping MaskHandler functions for this Tomato object.")
            continue

        # Show hyperspectral image using selected RGB bands
        hdr_path = data_paths.images_paths.get_hdr_paths()
        print("Displaying hyperspectral image...")
        try:
            MaskHandler.show_hyperspectral_image(hdr_path, rgb_bands)
        except Exception as e:
            print(f"Error displaying hyperspectral image: {e}")

        # Show segmented hyperspectral image for a specific mask ID
        mask_id = tomato.id_in_photo
        print(f"Displaying hyperspectral image segment for mask ID {mask_id}...")
        try:
            MaskHandler.show_hyperspectral_image_segment(
                hdr_path=hdr_path,
                mask_csv_path=images_paths.get_mask_paths(),
                mask_id=mask_id,
                selected_bands=rgb_bands
            )
        except Exception as e:
            print(f"Error displaying hyperspectral image segment: {e}")
