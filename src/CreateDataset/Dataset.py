import pickle
from typing import List
import pandas as pd
from datetime import datetime
import os
import sys
import numpy as np
from src.Tomato import *
from src.ImageData import *
from .File_helper import *


def create_dataset(tomatoes_dict: List[Tomato], two_sides_flag=False, only_first_harvest=False):
    dataset = []
    tomato_id_counter = 1
    for harvest_key, harvest in tomatoes_dict.items():
        if only_first_harvest and harvest_key != 'harvest_1': break
        for part_key, part in harvest.items():
            counter = 0                              # # # # counter - delete later
            data_directory = part['data_directory']
            date_picked = datetime.strptime(part['date_picked'], '%Y_%m_%d').date()
            files_path = part['files_path']
            lab_results_path = files_path['lab_results']

            lab_results = parse_lab_results(lab_results_path, harvest_key, part_key, part['amount_of_tomatoes'])

            for index, row in lab_results.iterrows():
                name = str(row['ID']).strip()
                cultivar = name.split('_')[0]
                id_in_photo = int(re.sub(r'\D', '', name.split('_')[-1]))
                if harvest_key == 'harvest_2': 
                    id_in_photo = ((id_in_photo - 1) % part['amount_of_tomatoes']) + 1
                if not (1 <= id_in_photo <= part['amount_of_tomatoes']):
                    print(f"Error: id_in_photo {id_in_photo} out of range for tomato with _id {tomato['_id']}, harvest {tomato['harvest']}.")
                quality_assess = extract_quality_assess(row)

                if two_sides_flag:
                    # Process each side as a separate tomato object
                    for side in ['sideA', 'sideB']:
                        images_paths = create_images_paths(files_path, two_sides_flag=False, side=side)
                        data_paths = TomatoDataPaths(data_directory, lab_results_path, images_paths)
                        spectral_stats = create_spectral_stats(images_paths, id_in_photo)
                        tomato = create_tomato(tomato_id_counter, name, cultivar, date_picked, id_in_photo, quality_assess, spectral_stats, harvest_key, data_paths)
                        dataset.append(tomato)
                        print (tomato_id_counter - 1) # # # # counter - delete later
                        tomato_id_counter, counter = tomato_id_counter + 1, counter + 1

                else:
                    # Process both sides as a single tomato object
                    images_paths = create_images_paths(files_path, two_sides_flag=True)
                    data_paths = TomatoDataPaths(data_directory, lab_results_path, images_paths)
                    spectral_stats = create_spectral_stats(images_paths, id_in_photo)
                    tomato = create_tomato(tomato_id_counter, name, cultivar, date_picked, id_in_photo, quality_assess, spectral_stats, harvest_key, data_paths)
                    dataset.append(tomato)
                    print (tomato_id_counter - 1) # # # # counter - delete later
                    tomato_id_counter, counter = tomato_id_counter + 1, counter + 1
                    
            
            print(f"The number of tomatoes: {harvest_key}, {part_key}, {counter} from {part['amount_of_tomatoes']}" )
             
    print( f'The number of tomatoes objects in the dataset is: {len(dataset)}')
    return dataset


# Helper Functions

def parse_lab_results(lab_results_path, harvest_key, part_key, amount_of_tomatoes):
    """Reads and parses the lab results file into a DataFrame."""
    if not os.path.exists(lab_results_path):
        raise FileNotFoundError(f"Lab results file not found: {lab_results_path}")
    lab_results = pd.read_csv(lab_results_path)
    lab_results.columns = lab_results.columns.str.strip()
    lab_results = lab_results.dropna(how='all').reset_index(drop=True)

    if harvest_key == 'harvest_2':
        # Extract part number from part_key (e.g., 'part 1' -> 1)
        try:
            part_number = int(part_key.split()[1])
        except (IndexError, ValueError):
            raise ValueError(f"Invalid part_key format: {part_key}. Expected format 'part X'.")

        # Calculate row range
        start_row = (5 * 60) if part_number == 6 else ((part_number - 1) * amount_of_tomatoes)
        end_row = start_row + amount_of_tomatoes

        # Check if the lab_results has enough rows
        if end_row > len(lab_results):
            raise IndexError(f"Lab results file does not have enough rows for {part_key}.")
        
        # Slice the DataFrame for the current part
        lab_results = lab_results.iloc[start_row:end_row].reset_index(drop=True)
    return lab_results

def extract_quality_assess(row):
    """Creates a QualityAssess object from a row in the lab results."""
    def get_value(key):
        value = row.get(key, None)
        if pd.isna(value):
            return None
        return value

    quality_assess = QualityAssess(
        weight=get_value('Weight (g)'),
        firmness=get_value('Firmness'),
        citric_acid=get_value('Citric Acid (%)'),
        pH=get_value('pH'),
        TSS=get_value('TSS'),
        ascorbic_acid=get_value('Ascrobic acid (mg/100g)'),
        lycopene=get_value('Lycopene (mg/100g)')
    )
    return quality_assess

def create_images_paths(files_path, two_sides_flag=True, side=None):
    """Creates an ImagesPath instance (SingleSideImagesPath or TwoSideImagesPath)."""
    if two_sides_flag:
        sideA_paths = {
            'hdr': files_path['sideA']['hdr'],
            'png': files_path['sideA']['png'],
            'mask': files_path['sideA']['mask']
        }
        sideB_paths = {
            'hdr': files_path['sideB']['hdr'],
            'png': files_path['sideB']['png'],
            'mask': files_path['sideB']['mask']
        }
        images_paths = TwoSideImagesPath(sideA_paths, sideB_paths)
    else:
        images_paths = SingleSideImagesPath(
            hdr=files_path[side]['hdr'],
            png=files_path[side]['png'],
            mask=files_path[side]['mask']
        )
    return images_paths

def create_spectral_stats(images_paths: ImagesPath, id_in_image: int):
    """Creates a SpectralStats object using the ImagesPath instance."""
    spectral_stats = SpectralStats(images_paths=images_paths, id_in_image=id_in_image)
    return spectral_stats

def create_tomato(_id, name, cultivar, date_picked, id_in_photo, quality_assess, spectral_stats, harvest, data_paths):
    """Creates a Tomato object using all the extracted information."""
    tomato = Tomato(
        _id=_id,
        name=name,
        cultivar=cultivar,
        date_picked=date_picked,
        id_in_photo=id_in_photo,
        quality_assess=quality_assess,
        spectral_stats=spectral_stats,
        harvest=harvest,
        data_paths=data_paths
    )
    return tomato


def save_dataset(tomatoes, filename='tomatoes_dataset.pkl'):
    """
    Saves the list of Tomato objects to a file using pickle.

    :param tomatoes: List of Tomato objects.
    :param filename: Filename to save the dataset.
    """
    with open(filename, 'wb') as file:
        pickle.dump(tomatoes, file)
    print(f"Dataset saved to {filename}")

def load_dataset(filename='tomatoes_dataset.pkl'):
    """
    Loads the list of Tomato objects from a pickle file.

    :param filename: Filename to load the dataset from.
    :return: List of Tomato objects.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"The dataset file {filename} does not exist.")

    # Handle numpy compatibility issues with pickle loading
    try:
        with open(filename, 'rb') as file:
            tomatoes = pickle.load(file)
    except (ModuleNotFoundError, AttributeError) as e:
        if 'numpy._core' in str(e) or 'numpy.core' in str(e):
            # Create a compatibility mapping for numpy module changes
            import numpy
            if not hasattr(numpy, '_core'):
                # Map numpy._core to numpy.core for backward compatibility
                numpy._core = numpy.core
                numpy._core.numeric = numpy.core.numeric
                numpy._core.multiarray = numpy.core.multiarray
                numpy._core.umath = numpy.core.umath
            elif not hasattr(numpy, 'core'):
                # Map numpy.core to numpy._core for forward compatibility
                numpy.core = numpy._core
                
            # Add to sys.modules for pickle to find
            sys.modules['numpy._core'] = numpy._core
            sys.modules['numpy._core.numeric'] = getattr(numpy._core, 'numeric', numpy)
            sys.modules['numpy._core.multiarray'] = getattr(numpy._core, 'multiarray', numpy)
            sys.modules['numpy._core.umath'] = getattr(numpy._core, 'umath', numpy)
            
            # Try loading again
            with open(filename, 'rb') as file:
                tomatoes = pickle.load(file)
        else:
            raise e
    
    print(f"Dataset loaded from {filename}")
    return tomatoes

if __name__ == "__main__":
    # Uncomment the following lines to create and save the dataset

    # Step 0: create the tomatoes dictionary
    tomatoes_dict = parse_custom_dict(config.TOMATOES_DICT_PATH)

    # Step 1: Create the dataset
    tomatoes = create_dataset(tomatoes_dict['tomatoes_dict'])

    # Step 2: Save the dataset to the root folder
    save_dataset(tomatoes, filename=config.DATASET_PATH)

    # Step 3: Load the dataset
    loaded_tomatoes = load_dataset(config.DATASET_PATH)

