# ImagesPath.py

from abc import ABC, abstractmethod

class ImagesPath(ABC):
    @abstractmethod
    def get_hdr_paths(self):
        pass

    @abstractmethod
    def get_png_paths(self):
        pass

    @abstractmethod
    def get_mask_paths(self):
        pass

