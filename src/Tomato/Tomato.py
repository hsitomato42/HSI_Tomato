# Tomato.py

from src.Tomato.SpectralStats import SpectralStats
from src.Tomato.StatisticStats import StatisticStats
from src.ImageData import ImagesPath, SingleSideImagesPath, TwoSideImagesPath
from datetime import date

class TomatoDataPaths:
    def __init__(self, data_directory: str, lab_results: str, images_paths: ImagesPath):
        self.data_directory = data_directory
        self.lab_results = lab_results
        self.images_paths = images_paths  # Instance of ImagesPath subclass


class QualityAssess:
    def __init__(self, weight, firmness, citric_acid, pH, TSS, ascorbic_acid, lycopene):
        self.weight = weight
        self.firmness = firmness
        self.citric_acid = citric_acid
        self.pH = pH
        self.TSS = TSS
        self.ascorbic_acid = ascorbic_acid
        self.lycopene = lycopene
        


class Tomato:
    def __init__(
        self,
        _id: int,
        name: str,
        cultivar: str,
        date_picked: date,
        id_in_photo: int,
        quality_assess: QualityAssess,
        spectral_stats: SpectralStats,
        harvest: str,
        data_paths: TomatoDataPaths
    ) -> None:
        self._id = _id
        self.name = name
        self.cultivar = cultivar
        self.date_picked = date_picked
        self.id_in_photo = int(id_in_photo)
        self.quality_assess = quality_assess
        self.spectral_stats = spectral_stats
        self.harvest = harvest
        self.data_paths = data_paths # Contains data_directory, lab_results, images_paths
        self.statistic_stats = StatisticStats(self.spectral_stats.reflectance_matrix)
