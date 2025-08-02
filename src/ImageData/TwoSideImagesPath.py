
from typing import Dict
from .ImagesPath import *

class TwoSideImagesPath(ImagesPath):
    def __init__(
        self, 
        sideA_paths: Dict[str, str], 
        sideB_paths: Dict[str, str]
    ) -> None:
        """
        Initializes paths for both sides of the image.

        Args:
            sideA_paths (Dict[str, str]): Dictionary with keys 'hdr', 'png', 'mask' for side A.
            sideB_paths (Dict[str, str]): Dictionary with keys 'hdr', 'png', 'mask' for side B.
        """
        self.sideA: Dict[str, str] = sideA_paths  # Dict with 'hdr', 'png', 'mask'
        self.sideB: Dict[str, str] = sideB_paths  # Dict with 'hdr', 'png', 'mask'

    def get_hdr_paths(self) -> Dict[str, str]:
        return {'sideA': self.sideA['hdr'], 'sideB': self.sideB['hdr']}

    def get_png_paths(self) -> Dict[str, str]:
        return {'sideA': self.sideA['png'], 'sideB': self.sideB['png']}

    def get_mask_paths(self) -> Dict[str, str]:
        return {'sideA': self.sideA['mask'], 'sideB': self.sideB['mask']}

    def __repr__(self) -> str:
        return (
            f"TwoSideImagesPath(sideA={self.sideA}, sideB={self.sideB})"
        )
