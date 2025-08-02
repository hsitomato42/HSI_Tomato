"""
Demo script for tomato instance segmentation using Grounding DINO + SAM.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add computer vision module to path
sys.path.append(str(Path(__file__).parent.parent))

from src.computer_vision.models.grounding_dino_sam_model import GroundingDinoSamModel


def demo_tomato_segmentation():
    """Demonstrate tomato segmentation on test images."""
    print("🍅 Tomato Instance Segmentation Demo")
    print("=" * 60)
    print("\nThis demo uses:")
    print("- Grounding DINO: For detecting tomatoes using text prompts")
    print("- SAM (Segment Anything Model): For precise segmentation masks")
    print("=" * 60)
    
    # Initialize model
    print("\nInitializing model...")
    model = GroundingDinoSamModel()
    model.initialize()
    
    # Configure for tomato detection
    model.text_prompt = "tomato"
    model.box_threshold = 0.35
    model.text_threshold = 0.25
    
    print(f"\nConfiguration:")
    print(f"- Text prompt: '{model.text_prompt}'")
    print(f"- Box threshold: {model.box_threshold}")
    print(f"- Text threshold: {model.text_threshold}")
    
    # Demo on test images
    test_dir = Path("../Data/segmentation_datasets/segmentation dataset/Test/img")
    demo_images = ["IMG_0983.jpg", "IMG_0987.jpg", "IMG_0991.jpg"]
    
    output_dir = Path("demo_outputs")
    output_dir.mkdir(exist_ok=True)
    
    total_tomatoes = 0
    
    for img_name in demo_images:
        img_path = test_dir / img_name
        
        print(f"\n{'='*60}")
        print(f"Processing: {img_name}")
        print(f"{'='*60}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Error loading image: {img_path}")
            continue
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"Image shape: {image.shape}")
        
        # Run segmentation
        print("\nRunning detection and segmentation...")
        boxes, masks, scores, class_names = model.predict(image_rgb)
        
        num_tomatoes = len(boxes)
        total_tomatoes += num_tomatoes
        
        print(f"\n✓ Found {num_tomatoes} tomatoes!")
        
        if num_tomatoes > 0:
            # Show detection details
            print("\nDetection details:")
            for i, (box, score) in enumerate(zip(boxes, scores)):
                print(f"  Tomato {i+1}:")
                print(f"    - Confidence: {score:.2%}")
                print(f"    - Bounding box: [{int(box[0])}, {int(box[1])}, {int(box[2])}, {int(box[3])}]")
                print(f"    - Size: {int(box[2]-box[0])}x{int(box[3]-box[1])} pixels")
            
            # Visualize results
            result_image = model.visualize(image_rgb, (boxes, masks, scores, class_names))
            
            # Save visualization
            vis_path = output_dir / f"{Path(img_name).stem}_segmentation.jpg"
            cv2.imwrite(str(vis_path), cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
            print(f"\n💾 Saved visualization to: {vis_path}")
            
            # Save annotations
            ann_path = output_dir / f"{Path(img_name).stem}_annotations.json"
            model.save_predictions((boxes, masks, scores, class_names), ann_path)
            print(f"💾 Saved annotations to: {ann_path}")
    
    print(f"\n{'='*60}")
    print(f"✅ Demo Complete!")
    print(f"{'='*60}")
    print(f"\nSummary:")
    print(f"- Processed {len(demo_images)} images")
    print(f"- Detected {total_tomatoes} tomatoes in total")
    print(f"- Results saved in: {output_dir}/")
    
    print("\n📌 Next steps:")
    print("1. Check the visualization images in demo_outputs/")
    print("2. Review the JSON annotations for detailed segmentation data")
    print("3. Use model.segment_and_save_with_approval() for interactive mode")
    
    return model


def demo_custom_detection():
    """Demonstrate custom object detection."""
    print("\n\n🔧 Custom Detection Demo")
    print("=" * 60)
    
    model = GroundingDinoSamModel()
    model.initialize()
    
    # You can change this to detect other objects!
    custom_prompts = ["apple", "banana", "orange", "fruit"]
    
    print("You can modify the text prompt to detect different objects:")
    print(f"Example prompts: {custom_prompts}")
    print("\nTo detect other objects, simply change model.text_prompt")
    print("The model will detect and segment any object described in the prompt!")


if __name__ == "__main__":
    # Run main demo
    model = demo_tomato_segmentation()
    
    # Show custom detection possibilities
    demo_custom_detection()
    
    print("\n\n🎉 Thank you for using the Computer Vision Segmentation System!")
