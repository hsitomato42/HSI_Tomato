"""
Test script for Grounding DINO + SAM segmentation model with actual dataset.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import json

# Add computer vision module to path
sys.path.append(str(Path(__file__).parent.parent))

from src.computer_vision.models.grounding_dino_sam_model import GroundingDinoSamModel
from src.computer_vision.utils.segmentation_helpers import load_annotation_json, format_segmentation_annotation
from src.computer_vision.utils.annotation_converter import (
    supervisely_to_coco_format, 
    count_tomato_annotations,
    extract_tomato_classes
)


def test_on_real_dataset():
    """Test the model on the real segmentation dataset."""
    # Define paths
    base_path = Path("../Data/segmentation_datasets/segmentation dataset")
    test_img_path = base_path / "Test" / "img"
    test_ann_path = base_path / "Test" / "ann"
    
    print(f"Looking for images in: {test_img_path}")
    print(f"Looking for annotations in: {test_ann_path}")
    
    # Check if paths exist
    if not test_img_path.exists():
        print(f"Error: Image path does not exist: {test_img_path.absolute()}")
        return
    
    if not test_ann_path.exists():
        print(f"Error: Annotation path does not exist: {test_ann_path.absolute()}")
        return
    
    # Find image files
    image_files = sorted(list(test_img_path.glob("*.jpg")) + list(test_img_path.glob("*.JPG")))
    
    if not image_files:
        print("No image files found!")
        return
    
    print(f"\nFound {len(image_files)} test images")
    
    # Initialize model
    print("\nInitializing Grounding DINO + SAM model...")
    model = GroundingDinoSamModel()
    model.initialize()
    
    # Process first 3 images
    for idx, img_path in enumerate(image_files[:3]):
        print(f"\n{'='*80}")
        print(f"Testing image {idx+1}/3: {img_path.name}")
        print(f"{'='*80}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        print(f"Image shape: {image.shape}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load ground truth annotation
        ann_path = test_ann_path / f"{img_path.name}.json"
        if ann_path.exists():
            with open(ann_path, 'r') as f:
                ground_truth = json.load(f)
            
            gt_count = count_tomato_annotations(ground_truth)
            gt_classes = extract_tomato_classes(ground_truth)
            
            print(f"\nGround Truth:")
            print(f"  Total annotations: {len(ground_truth.get('objects', []))}")
            print(f"  Tomato annotations: {gt_count}")
            print(f"  Classes: {gt_classes}")
        
        # Run prediction
        print("\nRunning segmentation...")
        try:
            predictions = model.predict(image_rgb)
            boxes, masks, scores, class_names = predictions
            
            print(f"\nPredictions:")
            print(f"  Detected {len(boxes)} tomatoes")
            for i, (box, score, name) in enumerate(zip(boxes, scores, class_names)):
                print(f"  {i+1}. {name} (confidence: {score:.2f}) - bbox: {[int(b) for b in box]}")
            
            # Visualize results
            result_image = model.visualize(image_rgb, predictions)
            
            # Save visualization
            output_dir = base_path / "test_outputs"
            output_dir.mkdir(exist_ok=True)
            
            vis_path = output_dir / f"{img_path.stem}_visualization.jpg"
            cv2.imwrite(str(vis_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            print(f"\nSaved visualization to: {vis_path}")
            
            # Save predictions
            pred_path = output_dir / f"{img_path.stem}_predictions.json"
            model.save_predictions(predictions, pred_path)
            
        except Exception as e:
            print(f"Error processing image: {e}")
            import traceback
            traceback.print_exc()


def test_segment_and_save_with_approval():
    """Test the interactive segmentation feature."""
    print("\n" + "="*80)
    print("Testing Interactive Segmentation with Approval")
    print("="*80)
    
    # Get a test image
    base_path = Path("../Data/segmentation_datasets/segmentation dataset")
    test_img_path = base_path / "Test" / "img"
    
    image_files = sorted(list(test_img_path.glob("*.jpg")))
    
    if image_files:
        test_image = image_files[0]
        print(f"\nWould test interactive mode with: {test_image.name}")
        print("In actual use, this would:")
        print("1. Display the segmentation result")
        print("2. Ask for user approval")
        print("3. Save only if approved")
        
        # Create a simple non-interactive test
        model = GroundingDinoSamModel()
        model.initialize()
        
        # Load and process image
        image = cv2.imread(str(test_image))
        if image is not None:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            predictions = model.predict(image_rgb)
            
            # Save without approval for testing
            output_dir = base_path / "test_outputs"
            output_dir.mkdir(exist_ok=True)
            
            test_save_path = output_dir / f"{test_image.stem}_auto_saved.json"
            model.save_predictions(predictions, test_save_path)
            print(f"\nAuto-saved test predictions to: {test_save_path}")


def verify_dependencies():
    """Check if required packages are installed."""
    print("\nChecking dependencies...")
    
    dependencies = {
        "cv2": "opencv-python",
        "numpy": "numpy",
        "torch": "torch",
        "matplotlib": "matplotlib"
    }
    
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\nTo install missing packages, run:")
        print(f"pip install {' '.join(missing)}")
    
    return len(missing) == 0


if __name__ == "__main__":
    print("Computer Vision Segmentation Model Test")
    print("="*80)
    
    # Verify dependencies
    if not verify_dependencies():
        print("\nPlease install missing dependencies before continuing.")
        sys.exit(1)
    
    # Run tests
    test_on_real_dataset()
    test_segment_and_save_with_approval()
    
    print("\n\nAll tests completed!")
