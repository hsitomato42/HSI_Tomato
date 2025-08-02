# SingleSideImagesPath.py

from .ImagesPath import *

class SingleSideImagesPath(ImagesPath):
    def __init__(self, hdr: str, png: str, mask: str) -> None:
        self.hdr: str = hdr
        self.png: str = png
        self.mask: str = mask

    def get_hdr_paths(self) -> str:
        return self.hdr

    def get_png_paths(self) -> str:
        return self.png

    def get_mask_paths(self) -> str:
        return self.mask

    def __repr__(self) -> str:
        return (
            f"SingleSideImagesPath(hdr={self.hdr}, png={self.png}, mask={self.mask})"
        )
