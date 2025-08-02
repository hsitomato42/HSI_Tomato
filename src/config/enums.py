# enums.py

from enum import Enum
from typing import List, Optional

class ModelType(Enum):
    XGBOOST = "XGBOOST"
    RANDOM_FOREST = "RANDOM_FOREST"
    CNN = "CNN"
    MULTI_HEAD_CNN = "MULTI_HEAD_CNN"
    CNN_TRANSFORMER = "CNN_TRANSFORMER"
    SPECTRAL_TRANSFORMER = "SPECTRAL_TRANSFORMER"
    VIT = "VIT"
    ADVANCED_MULTI_BRANCH_CNN_TRANSFORMER = "ADVANCED_MULTI_BRANCH_CNN_TRANSFORMER"
    GLOBAL_BRANCH_FUSION_TRANSFORMER = "GLOBAL_BRANCH_FUSION_TRANSFORMER"
    COMPONENT_DRIVEN_ATTENTION_TRANSFORMER = "COMPONENT_DRIVEN_ATTENTION_TRANSFORMER"
    COMPONENT_DRIVEN_ATTENTION_TRANSFORMER_V2 = "COMPONENT_DRIVEN_ATTENTION_TRANSFORMER_V2"
    PCCT_STATIC = "PCCT_STATIC"
    PCCT_PROGRESSIVE = "PCCT_PROGRESSIVE"
    CDAT_PROGRESSIVE = "CDAT_PROGRESSIVE"
    CDAT_PROGRESSIVE_V2 = "CDAT_PROGRESSIVE_V2"
    PCCT_STATIC_V2 = "PCCT_STATIC_V2"
    PCCT_PROGRESSIVE_V2 = "PCCT_PROGRESSIVE_V2"

class ImagePathKind(Enum):
    ONE_SIDE = 'oneSide'
    TWO_SIDES = 'twoSides'


class SpectralIndex(Enum):
    """
    Enumeration of Spectral Indexes with their shortcuts, full names, and formulas.
    """
    RED_INDEX = ('RED_INDEX', 'Red Index', '(R - G) / (R + G)')
    LYC = ('LYC', 'Lycopene Index', '(630 nm - 570 nm) / (630 nm + 570 nm)')
    INA = ('INA', 'Normalized Anthocyanin Index', '(760 nm - 570 nm) / (760 nm + 570 nm)')
    ZMI = ('ZMI', 'Zarco-Tejada & Miller Spectral Index', '750 nm / 710 nm')
    GLI = ('GLI', 'Green Leaf Index', '(G - R + (G - B)) / (2 * G + R + B)')
    GRVI = ('GRVI', 'Green-Red Vegetation Index', '(G - R) / (G + R)')

    def __init__(self, shortcut: str, full_name: str, formula: str):
        """
        Initializes the SpectralIndex enum member.
        
        Args:
            shortcut (str): The abbreviation of the index.
            full_name (str): The full descriptive name of the index.
            formula (str): The formula used to calculate the index.
        """
        self.shortcut = shortcut
        self.full_name = full_name
        self.formula = formula

    @classmethod
    def list_shortcuts(cls) -> List[str]:
        """
        Returns a list of all spectral index shortcuts.
        
        Returns:
            List[str]: Shortcuts of all spectral indexes.
        """
        return [index.shortcut for index in cls]

    @classmethod
    def list_full_names(cls) -> List[str]:
        """
        Returns a list of all spectral index full names.
        
        Returns:
            List[str]: Full names of all spectral indexes.
        """
        return [index.full_name for index in cls]

    @classmethod
    def get_formula(cls, shortcut: str) -> Optional[str]:
        """
        Retrieves the formula for a given spectral index shortcut.
        
        Args:
            shortcut (str): The abbreviation of the index.
        
        Returns:
            Optional[str]: The formula of the index if found, else None.
        """
        for index in cls:
            if index.shortcut == shortcut:
                return index.formula
        return None


class AdditionalFeature(Enum):
    AREA = "area"
    MEAN = "mean"
    MEDIAN = "median"
    STD = "std"


class FeelingMissingValuesTechnique(Enum):
    PREDICT_WITH_MODEL = "predict_with_model"
    MEAN = "mean"
    MEDIAN = "median"
    NAN = "nan"
    NONE = "none"


class FeatureSelectionStrategy(Enum):
    """
    Enumeration of Feature Selection Strategies for hyperspectral band selection.
    
    Controls how the attention-based feature selection component behaves
    when used with different models.
    """
    MODEL_AGNOSTIC = ("MODEL_AGNOSTIC", 
                      "Model-Agnostic", 
                      "Same selected bands for all models. Feature selection is trained once and applied consistently.")
    
    SEMI_ADAPTIVE = ("SEMI_ADAPTIVE", 
                     "Semi-Adaptive", 
                     "Joint training allows adaptation to model differences. Balances consistency and specialization.")
    
    MODEL_SPECIFIC = ("MODEL_SPECIFIC", 
                      "Model-Specific", 
                      "Optimizes band selection specifically for each model's performance. Maximum prediction accuracy.")
    
    def __init__(self, shortcut: str, display_name: str, description: str):
        """
        Initializes the FeatureSelectionStrategy enum member.
        
        Args:
            shortcut (str): The enum identifier.
            display_name (str): Human-readable name of the strategy.
            description (str): Detailed description of the strategy behavior.
        """
        self.shortcut = shortcut
        self.display_name = display_name
        self.description = description

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        Returns a list of all feature selection strategy names.
        
        Returns:
            List[str]: Names of all feature selection strategies.
        """
        return [strategy.display_name for strategy in cls]

    @classmethod
    def get_description(cls, strategy_name: str) -> Optional[str]:
        """
        Retrieves the description for a given strategy.
        
        Args:
            strategy_name (str): The name of the strategy.
        
        Returns:
            Optional[str]: The description if found, else None.
        """
        for strategy in cls:
            if strategy.display_name == strategy_name or strategy.shortcut == strategy_name or strategy.name == strategy_name:
                return strategy.description
        return None


class NDSISelectionStrategy(Enum):
    """
    Enumeration of NDSI pair selection strategies for feature selection.
    
    Controls how NDSI pairs are chosen from the selected reflectance bands.
    """
    IMPORTANCE_BASED = ("IMPORTANCE_BASED", 
                       "Importance-Based", 
                       "Select NDSI pairs based on band importance from attention mechanism. Uses learned selection.")
    
    BRUTE_FORCE = ("BRUTE_FORCE", 
                   "Brute Force", 
                   "Generate all unique band pairs from selected bands. Left band index is always smaller.")
    
    def __init__(self, shortcut: str, display_name: str, description: str):
        """
        Initializes the NDSISelectionStrategy enum member.
        
        Args:
            shortcut (str): The enum identifier.
            display_name (str): Human-readable name of the strategy.
            description (str): Detailed description of the strategy behavior.
        """
        self.shortcut = shortcut
        self.display_name = display_name
        self.description = description

    @classmethod
    def list_strategies(cls) -> List[str]:
        """
        Returns a list of all NDSI selection strategy names.
        
        Returns:
            List[str]: Names of all NDSI selection strategies.
        """
        return [strategy.display_name for strategy in cls]

    @classmethod
    def get_description(cls, strategy_name: str) -> Optional[str]:
        """
        Retrieves the description for a given NDSI strategy.
        
        Args:
            strategy_name (str): The name of the strategy.
        
        Returns:
            Optional[str]: The description if found, else None.
        """
        for strategy in cls:
            if strategy.display_name == strategy_name or strategy.shortcut == strategy_name or strategy.name == strategy_name:
                return strategy.description
        return None

    
    