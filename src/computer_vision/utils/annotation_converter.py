"""
Annotation format converter for different segmentation formats.
"""

import json
import numpy as np
from typing import Dict, List, Any, Tuple
from pathlib import Path


def supervisely_to_coco_format(supervisely_ann: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert Supervisely format annotations to COCO-like format.
    
    Args:
        supervisely_ann: Supervisely format annotation
        
    Returns:
        COCO-like format annotation
    """
    annotations = []
    
    for obj in supervisely_ann.get("objects", []):
        if obj["geometryType"] == "polygon":
            # Get polygon points
            points = obj["points"]["exterior"]
            
            # Calculate bounding box
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            # Create annotation
            annotation = {
                "id": obj["id"],
                "class_name": obj["classTitle"],
                "bbox": {
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                },
                "polygon": points,
                "confidence": 1.0  # Ground truth has confidence 1.0
            }
            annotations.append(annotation)
    
    return {
        "annotations": annotations,
        "image_shape": [supervisely_ann["size"]["height"], supervisely_ann["size"]["width"]]
    }


def polygon_to_mask(polygon: List[List[int]], image_shape: Tuple[int, int]) -> np.ndarray:
    """
    Convert polygon points to binary mask.
    
    Args:
        polygon: List of [x, y] points
        image_shape: (height, width) of the image
        
    Returns:
        Binary mask
    """
    import cv2
    
    mask = np.zeros(image_shape, dtype=np.uint8)
    points = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [points], 1)
    
    return mask


def count_tomato_annotations(annotation: Dict[str, Any]) -> int:
    """
    Count tomato-related annotations in the annotation file.
    
    Args:
        annotation: Annotation dictionary
        
    Returns:
        Number of tomato annotations
    """
    tomato_classes = ["tomato", "b_green", "b_fully_ripened", "b_half_ripened", "l_fully_ripened", "l_green", "l_half_ripened"]
    
    if "objects" in annotation:  # Supervisely format
        count = sum(1 for obj in annotation["objects"] 
                   if any(tc in obj.get("classTitle", "").lower() for tc in tomato_classes))
    elif "annotations" in annotation:  # Our format
        count = sum(1 for ann in annotation["annotations"] 
                   if any(tc in ann.get("class_name", "").lower() for tc in tomato_classes))
    else:
        count = 0
    
    return count


def extract_tomato_classes(annotation: Dict[str, Any]) -> List[str]:
    """
    Extract unique tomato class names from annotation.
    
    Args:
        annotation: Annotation dictionary
        
    Returns:
        List of unique class names
    """
    classes = set()
    
    if "objects" in annotation:  # Supervisely format
        for obj in annotation["objects"]:
            classes.add(obj.get("classTitle", ""))
    elif "annotations" in annotation:  # Our format
        for ann in annotation["annotations"]:
            classes.add(ann.get("class_name", ""))
    
    return sorted(list(classes))
