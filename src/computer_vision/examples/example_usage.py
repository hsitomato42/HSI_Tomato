"""
Example usage of the Grounding DINO + SAM tomato segmentation model.
"""

import sys
from pathlib import Path

# Add computer vision module to path
sys.path.append(str(Path(__file__).parent.parent))

from src.computer_vision.models.grounding_dino_sam_model import GroundingDinoSamModel
import cv2


def example_basic_segmentation():
    """Basic example: Load image, run segmentation, save results."""
    print("Example 1: Basic Segmentation")
    print("-" * 40)
    
    # Initialize model
    model = GroundingDinoSamModel()
    model.initialize()
    
    # Example image path - replace with your image
    image_path = "../Data/segmentation_datasets/segmentation dataset/Test/img/IMG_0983.jpg"
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Run segmentation
    print("Running segmentation...")
    boxes, masks, scores, class_names = model.predict(image_rgb)
    
    print(f"Detected {len(boxes)} tomatoes")
    
    # Visualize results
    result_image = model.visualize(image_rgb, (boxes, masks, scores, class_names))
    
    # Save visualization
    output_path = "example_segmentation_result.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    print(f"Saved visualization to: {output_path}")
    
    # Save annotations
    annotation_path = "example_annotations.json"
    model.save_predictions((boxes, masks, scores, class_names), annotation_path)
    print(f"Saved annotations to: {annotation_path}")


def example_with_user_approval():
    """Example with user approval before saving."""
    print("\nExample 2: Segmentation with User Approval")
    print("-" * 40)
    
    # Initialize model
    model = GroundingDinoSamModel()
    model.initialize()
    
    # Example image path
    image_path = "../Data/segmentation_datasets/segmentation dataset/Test/img/IMG_0986.jpg"
    
    # Use the interactive method
    print("This would normally show the segmentation and ask for approval.")
    print("For automated testing, we'll simulate approval.")
    
    # In real usage, this would display the image and ask for approval
    approved = model.segment_and_save_with_approval(
        image_path,
        save_path="example_approved_segmentation.json"
    )
    
    if approved:
        print("Segmentation was approved and saved!")
    else:
        print("Segmentation was not approved.")


def example_batch_processing():
    """Example of processing multiple images."""
    print("\nExample 3: Batch Processing")
    print("-" * 40)
    
    # Initialize model once
    model = GroundingDinoSamModel()
    model.initialize()
    
    # Get list of images
    image_dir = Path("../Data/segmentation_datasets/segmentation dataset/Test/img")
    image_files = list(image_dir.glob("*.jpg"))[:5]  # Process first 5 images
    
    output_dir = Path("batch_results")
    output_dir.mkdir(exist_ok=True)
    
    for img_path in image_files:
        print(f"\nProcessing: {img_path.name}")
        
        # Load and process image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"  Failed to load image")
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run segmentation
        predictions = model.predict(image_rgb)
        boxes, masks, scores, class_names = predictions
        
        print(f"  Detected {len(boxes)} tomatoes")
        
        # Save results
        vis_path = output_dir / f"{img_path.stem}_visualization.jpg"
        result_image = model.visualize(image_rgb, predictions)
        cv2.imwrite(str(vis_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
        
        ann_path = output_dir / f"{img_path.stem}_annotations.json"
        model.save_predictions(predictions, ann_path)
        
        print(f"  Saved results to {output_dir}")
    
    print(f"\nBatch processing complete! Results saved in {output_dir}")


def example_custom_configuration():
    """Example with custom model configuration."""
    print("\nExample 4: Custom Configuration")
    print("-" * 40)
    
    # Initialize model with custom settings
    model = GroundingDinoSamModel(
        device="cpu",  # Force CPU usage
        sam_model_type="vit_h"  # Use largest SAM model
    )
    
    # Customize detection parameters
    model.text_prompt = "tomato"  # What to detect
    model.box_threshold = 0.25  # Lower threshold for more detections
    model.text_threshold = 0.20
    
    model.initialize()
    
    print("Model initialized with custom settings")
    print(f"  Device: {model.device}")
    print(f"  Text prompt: {model.text_prompt}")
    print(f"  Box threshold: {model.box_threshold}")
    print(f"  Text threshold: {model.text_threshold}")


if __name__ == "__main__":
    print("Tomato Segmentation Model - Usage Examples")
    print("=" * 50)
    
    # Run examples
    example_basic_segmentation()
    example_with_user_approval()
    example_batch_processing()
    example_custom_configuration()
    
    print("\n\nAll examples completed!")
    print("\nNote: The model is currently using mock detection/segmentation.")
    print("To use real models, install:")
    print("  - groundingdino: https://github.com/IDEA-Research/GroundingDINO")
    print("  - segment-anything: pip install segment-anything")
