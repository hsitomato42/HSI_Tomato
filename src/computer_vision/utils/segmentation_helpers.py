"""
Helper functions for instance segmentation.
"""

import json
import numpy as np
import cv2
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime


def format_segmentation_annotation(
    boxes: List[List[float]], 
    masks: List[np.ndarray], 
    scores: List[float], 
    class_names: List[str]
) -> Dict[str, Any]:
    """
    Format segmentation results into a standard annotation format.
    
    Args:
        boxes: List of bounding boxes [x1, y1, x2, y2]
        masks: List of segmentation masks
        scores: List of confidence scores
        class_names: List of class names for each detection
        
    Returns:
        Formatted annotation dictionary
    """
    annotations = []
    
    for idx, (box, mask, score, class_name) in enumerate(zip(boxes, masks, scores, class_names)):
        annotation = {
            "id": idx,
            "class_name": class_name,
            "confidence": float(score),
            "bbox": {
                "x1": float(box[0]),
                "y1": float(box[1]),
                "x2": float(box[2]),
                "y2": float(box[3])
            },
            "segmentation": encode_mask_to_rle(mask)
        }
        annotations.append(annotation)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "annotations": annotations,
        "image_shape": masks[0].shape if masks else None
    }


def encode_mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    """
    Encode binary mask to RLE (Run Length Encoding) format.
    
    Args:
        mask: Binary mask array
        
    Returns:
        RLE encoded mask
    """
    # Flatten the mask
    flat_mask = mask.flatten()
    
    # Find runs
    runs = []
    current_run_start = None
    current_run_value = None
    
    for idx, value in enumerate(flat_mask):
        if value != current_run_value:
            if current_run_start is not None:
                runs.append({
                    "start": current_run_start,
                    "length": idx - current_run_start,
                    "value": int(current_run_value)
                })
            current_run_start = idx
            current_run_value = value
    
    # Add the last run
    if current_run_start is not None:
        runs.append({
            "start": current_run_start,
            "length": len(flat_mask) - current_run_start,
            "value": int(current_run_value)
        })
    
    return {
        "shape": mask.shape,
        "runs": runs
    }


def decode_rle_to_mask(rle: Dict[str, Any]) -> np.ndarray:
    """
    Decode RLE format to binary mask.
    
    Args:
        rle: RLE encoded mask
        
    Returns:
        Binary mask array
    """
    shape = tuple(rle["shape"])
    mask = np.zeros(np.prod(shape), dtype=np.uint8)
    
    for run in rle["runs"]:
        start = run["start"]
        length = run["length"]
        value = run["value"]
        mask[start:start + length] = value
    
    return mask.reshape(shape)


def draw_segmentation_on_image(
    image: np.ndarray,
    boxes: List[List[float]],
    masks: List[np.ndarray],
    scores: List[float],
    class_names: List[str],
    alpha: float = 0.5
) -> np.ndarray:
    """
    Draw segmentation results on image.
    
    Args:
        image: Original image
        boxes: List of bounding boxes
        masks: List of segmentation masks
        scores: List of confidence scores
        class_names: List of class names
        alpha: Transparency for masks
        
    Returns:
        Image with drawn segmentations
    """
    # Create a copy to avoid modifying original
    result_image = image.copy()
    
    # Generate colors for each instance
    colors = generate_instance_colors(len(boxes))
    
    for idx, (box, mask, score, class_name, color) in enumerate(zip(boxes, masks, scores, class_names, colors)):
        # Draw mask
        result_image = apply_mask_to_image(result_image, mask, color, alpha)
        
        # Draw bounding box
        x1, y1, x2, y2 = [int(coord) for coord in box]
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f"{class_name}: {score:.2f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        
        # Draw label background
        cv2.rectangle(
            result_image,
            (x1, y1 - label_size[1] - 4),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            result_image,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
    
    return result_image


def apply_mask_to_image(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float
) -> np.ndarray:
    """
    Apply a colored mask to an image.
    
    Args:
        image: Original image
        mask: Binary mask
        color: RGB color tuple
        alpha: Transparency value
        
    Returns:
        Image with applied mask
    """
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = color
    
    # Blend with original image
    result = cv2.addWeighted(image, 1 - alpha, colored_mask, alpha, 0)
    
    return result


def generate_instance_colors(num_instances: int) -> List[Tuple[int, int, int]]:
    """
    Generate distinct colors for each instance.
    
    Args:
        num_instances: Number of instances
        
    Returns:
        List of RGB color tuples
    """
    colors = []
    
    # Use HSV color space for better color distribution
    for i in range(num_instances):
        hue = int(180 * i / max(num_instances, 1))
        hsv = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0]
        colors.append(tuple(int(c) for c in rgb))
    
    return colors


def save_annotation_json(annotations: Dict[str, Any], save_path: Path) -> None:
    """
    Save annotations to JSON file.
    
    Args:
        annotations: Annotation dictionary
        save_path: Path to save JSON file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(annotations, f, indent=2)


def load_annotation_json(json_path: Path) -> Dict[str, Any]:
    """
    Load annotations from JSON file.
    
    Args:
        json_path: Path to JSON file
        
    Returns:
        Annotation dictionary
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def display_image_with_prompt(image: np.ndarray, title: str = "Segmentation Result") -> bool:
    """
    Display image and prompt user for approval.
    
    Args:
        image: Image to display
        title: Window title
        
    Returns:
        True if user approves, False otherwise
    """
    # Convert BGR to RGB if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        display_image = image
    
    plt.figure(figsize=(12, 8))
    plt.imshow(display_image)
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Get user approval
    response = input("Do you approve this segmentation? (yes/no): ").strip().lower()
    return response in ['yes', 'y']


def filter_detections_by_class(
    boxes: List[List[float]],
    masks: List[np.ndarray],
    scores: List[float],
    class_names: List[str],
    target_classes: List[str]
) -> Tuple[List[List[float]], List[np.ndarray], List[float], List[str]]:
    """
    Filter detections to keep only specified classes.
    
    Args:
        boxes: List of bounding boxes
        masks: List of segmentation masks
        scores: List of confidence scores
        class_names: List of class names
        target_classes: List of target class names to keep
        
    Returns:
        Filtered boxes, masks, scores, and class names
    """
    filtered_boxes = []
    filtered_masks = []
    filtered_scores = []
    filtered_class_names = []
    
    for box, mask, score, class_name in zip(boxes, masks, scores, class_names):
        if any(target.lower() in class_name.lower() for target in target_classes):
            filtered_boxes.append(box)
            filtered_masks.append(mask)
            filtered_scores.append(score)
            filtered_class_names.append(class_name)
    
    return filtered_boxes, filtered_masks, filtered_scores, filtered_class_names
