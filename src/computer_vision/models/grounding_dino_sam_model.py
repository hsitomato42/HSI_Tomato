"""
Simplified instance segmentation model using Grounding DINO and SAM for tomato detection.
"""

import torch
import numpy as np
import cv2
from typing import Any, Dict, List, Tuple, Union
from pathlib import Path
import os
from PIL import Image

from ..core.base_model import BaseComputerVisionModel
from ..utils.segmentation_helpers import (
    format_segmentation_annotation,
    draw_segmentation_on_image,
    save_annotation_json,
    display_image_with_prompt,
    filter_detections_by_class
)

# Grounding DINO imports
import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from groundingdino.util.inference import preprocess_caption

# SAM imports
from segment_anything import sam_model_registry, SamPredictor


class GroundingDinoSamModel(BaseComputerVisionModel):
    """
    Instance segmentation model using Grounding DINO for detection and SAM for segmentation.
    Specifically designed for tomato detection and segmentation.
    """
    
    def __init__(
        self,
        grounding_dino_config_path: str = None,
        grounding_dino_checkpoint_path: str = None,
        sam_checkpoint_path: str = None,
        sam_model_type: str = "vit_h",
        device: str = None
    ):
        """
        Initialize the Grounding DINO + SAM model.
        
        Args:
            grounding_dino_config_path: Path to Grounding DINO config
            grounding_dino_checkpoint_path: Path to Grounding DINO checkpoint
            sam_checkpoint_path: Path to SAM checkpoint
            sam_model_type: SAM model type (vit_h, vit_l, vit_b)
            device: Device to run model on (cuda/cpu)
        """
        super().__init__("GroundingDINO_SAM")
        
        # Set default paths if not provided
        models_dir = Path(__file__).parent / "models"
        
        if grounding_dino_config_path is None:
            # Try to find config in groundingdino package
            import groundingdino
            package_path = Path(groundingdino.__file__).parent
            grounding_dino_config_path = package_path / "config" / "GroundingDINO_SwinT_OGC.py"
        
        if grounding_dino_checkpoint_path is None:
            grounding_dino_checkpoint_path = models_dir / "groundingdino_swint_ogc.pth"
            
        if sam_checkpoint_path is None:
            sam_checkpoint_path = models_dir / "sam_vit_h.pth"
        
        self.grounding_dino_config_path = str(grounding_dino_config_path)
        self.grounding_dino_checkpoint_path = str(grounding_dino_checkpoint_path)
        self.sam_checkpoint_path = str(sam_checkpoint_path)
        self.sam_model_type = sam_model_type
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.grounding_dino_model = None
        self.sam_predictor = None
        self.text_prompt = "tomato"
        self.box_threshold = 0.35
        self.text_threshold = 0.25
        
        # Transform for preprocessing
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def initialize(self) -> None:
        """Initialize Grounding DINO and SAM models."""
        print(f"Initializing {self.model_name}...")
        
        # Check if model files exist
        if not Path(self.grounding_dino_checkpoint_path).exists():
            raise FileNotFoundError(f"Grounding DINO checkpoint not found: {self.grounding_dino_checkpoint_path}")
        
        if not Path(self.sam_checkpoint_path).exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {self.sam_checkpoint_path}")
        
        # Initialize Grounding DINO
        self._initialize_grounding_dino()
        
        # Initialize SAM
        self._initialize_sam()
        
        self.is_initialized = True
        print(f"Successfully initialized {self.model_name} on {self.device}")
    
    def _initialize_grounding_dino(self) -> None:
        """Initialize Grounding DINO model."""
        print("Loading Grounding DINO model...")
        
        # Load config
        args = SLConfig.fromfile(self.grounding_dino_config_path)
        args.device = str(self.device)
        
        # Build model
        self.grounding_dino_model = build_model(args)
        
        # Load checkpoint
        checkpoint = torch.load(self.grounding_dino_checkpoint_path, map_location="cpu")
        self.grounding_dino_model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        
        # Move to device and set to eval mode
        self.grounding_dino_model = self.grounding_dino_model.to(self.device)
        self.grounding_dino_model.eval()
        
        print("Grounding DINO model loaded successfully")
    
    def _initialize_sam(self) -> None:
        """Initialize SAM model."""
        print("Loading SAM model...")
        
        # Load SAM model
        sam = sam_model_registry[self.sam_model_type](checkpoint=self.sam_checkpoint_path)
        sam.to(self.device)
        
        # Create predictor
        self.sam_predictor = SamPredictor(sam)
        
        print("SAM model loaded successfully")
    
    def predict(self, image: np.ndarray) -> Tuple[List, List, List, List]:
        """
        Run detection and segmentation on image.
        
        Args:
            image: Input RGB image
            
        Returns:
            Tuple of (boxes, masks, scores, class_names)
        """
        self.ensure_initialized()
        
        if not self.validate_image(image):
            raise ValueError("Invalid image format")
        
        # Run detection with Grounding DINO
        boxes, scores, phrases = self._detect_with_grounding_dino(image)
        
        if len(boxes) == 0:
            return [], [], [], []
        
        # Run segmentation with SAM
        masks = self._segment_with_sam(image, boxes)
        
        # Extract class names from phrases
        class_names = ["tomato"] * len(boxes)  # All detections are tomatoes
        
        return boxes, masks, scores, class_names
    
    def _detect_with_grounding_dino(
        self, 
        image: np.ndarray
    ) -> Tuple[List[List[float]], List[float], List[str]]:
        """
        Detect objects using Grounding DINO.
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Tuple of (boxes, scores, phrases)
        """
        # Convert to PIL
        image_pil = Image.fromarray(image)
        
        # Transform image
        image_transformed, _ = self.transform(image_pil, None)
        
        # Prepare caption
        caption = preprocess_caption(self.text_prompt)
        
        # Move image to device
        image_transformed = image_transformed.to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.grounding_dino_model(image_transformed[None], captions=[caption])
        
        # Post-process outputs
        prediction_logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        prediction_boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # Filter by threshold
        mask = prediction_logits.max(dim=1)[0] > self.box_threshold
        logits = prediction_logits[mask]  # (n, 256)
        boxes = prediction_boxes[mask]  # (n, 4)
        
        # Get phrases
        tokenizer = self.grounding_dino_model.tokenizer
        tokenized = tokenizer(caption)
        
        phrases = [
            get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenizer).replace('.', '')
            for logit in logits
        ]
        
        # Convert boxes from cxcywh to xyxy format
        h, w = image.shape[:2]
        boxes_xyxy = []
        for box in boxes:
            cx, cy, bw, bh = box
            x1 = (cx - bw/2) * w
            y1 = (cy - bh/2) * h
            x2 = (cx + bw/2) * w
            y2 = (cy + bh/2) * h
            boxes_xyxy.append([x1.item(), y1.item(), x2.item(), y2.item()])
        
        scores = logits.max(dim=1)[0].tolist()
        
        print(f"Detected {len(boxes_xyxy)} objects with Grounding DINO")
        
        return boxes_xyxy, scores, phrases
    
    def _segment_with_sam(
        self, 
        image: np.ndarray, 
        boxes: List[List[float]]
    ) -> List[np.ndarray]:
        """
        Segment objects using SAM.
        
        Args:
            image: Input image (RGB)
            boxes: List of bounding boxes [x1, y1, x2, y2]
            
        Returns:
            List of segmentation masks
        """
        # Set image in SAM predictor
        self.sam_predictor.set_image(image)
        
        masks = []
        
        for box in boxes:
            # Convert box to SAM format
            input_box = np.array(box)
            
            # Predict mask
            masks_pred, scores, logits = self.sam_predictor.predict(
                box=input_box,
                multimask_output=True  # Get multiple mask options
            )
            
            # Select best mask (highest score)
            best_mask_idx = np.argmax(scores)
            mask = masks_pred[best_mask_idx]
            
            masks.append(mask.astype(np.uint8))
        
        print(f"Generated {len(masks)} segmentation masks with SAM")
        
        return masks
    
    def visualize(self, image: np.ndarray, predictions: Any) -> np.ndarray:
        """
        Visualize predictions on image.
        
        Args:
            image: Original image
            predictions: Model predictions
            
        Returns:
            Image with visualizations
        """
        boxes, masks, scores, class_names = predictions
        
        if len(boxes) == 0:
            print("No tomatoes detected in the image.")
            return image
        
        return draw_segmentation_on_image(
            image, boxes, masks, scores, class_names, alpha=0.5
        )
    
    def save_predictions(self, predictions: Any, save_path: Union[str, Path]) -> None:
        """
        Save predictions to JSON file.
        
        Args:
            predictions: Model predictions
            save_path: Path to save file
        """
        boxes, masks, scores, class_names = predictions
        
        annotations = format_segmentation_annotation(
            boxes, masks, scores, class_names
        )
        
        save_annotation_json(annotations, Path(save_path))
        print(f"Saved annotations to {save_path}")
    
    def segment_and_save_with_approval(
        self, 
        image_path: Union[str, Path],
        save_path: Union[str, Path] = None
    ) -> bool:
        """
        Segment image, show results, and save if approved.
        
        Args:
            image_path: Path to input image
            save_path: Path to save annotations (auto-generated if None)
            
        Returns:
            True if approved and saved, False otherwise
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error loading image: {image_path}")
            return False
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run prediction
        print(f"Processing image: {Path(image_path).name}")
        predictions = self.predict(image_rgb)
        
        # Check if any detections
        boxes, masks, scores, class_names = predictions
        if len(boxes) == 0:
            print("No tomatoes detected in this image.")
            return False
        
        # Visualize
        result_image = self.visualize(image_rgb, predictions)
        
        # Show and get approval
        approved = display_image_with_prompt(
            result_image, 
            title=f"Segmentation Result - {Path(image_path).name}"
        )
        
        if approved:
            # Generate save path if not provided
            if save_path is None:
                image_path = Path(image_path)
                save_path = image_path.parent / f"{image_path.stem}_segmentation.json"
            
            self.save_predictions(predictions, save_path)
            return True
        else:
            print("Segmentation not approved. Not saving.")
            return False
