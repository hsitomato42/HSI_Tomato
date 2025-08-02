from pickle import FALSE
import time
import socket
from src.config.enums import *
from typing import List, Tuple
from src.utils.logger import get_logger

class Config:
    """Main configuration class with hierarchical organization"""
    
    @classmethod
    def print_config(cls):
        """Log current configuration for debugging"""
        logger = get_logger("config")
        
        logger.info("=" * 100)
        logger.info("CONFIGURATION")
        logger.info("=" * 100)
        logger.info(f"Dataset.MODEL_TYPE: {cls.Models.MODEL_TYPE}")
        logger.info(f"Training.EPOCHS: {cls.Training.EPOCHS}")
        logger.info(f"Training.BATCH_SIZE: {cls.Training.BATCH_SIZE}")
        logger.info(f"Spectral.INDEXES: {cls.Spectral.INDEXES}")
        logger.info(f"FeatureSelection.USE_FEATURE_SELECTION: {cls.FeatureSelection.USE_FEATURE_SELECTION}")
        logger.info("=" * 100)
    
    class Dataset:
        """Dataset and data processing configuration"""
        CREATE_DATASET = False  # Enable to recreate dataset and avoid pickle compatibility issues
        IMAGES_PATH_KIND = ImagePathKind.ONE_SIDE  # Options: 'ONE_SIDE', 'TWO_SIDES'
        RANDOM_STATE = 42  # (int(time.time() * 1000) % 100) + 1
        ONLY_HARVEST_1 = False
        IMAGES_BLOCKER = False
        
        # Band cutting configuration to prevent OOM
        CUT_BANDS_FROM_START = 0  # Number of bands to cut from the beginning (0-204)
        
        # Data paths
        TOMATOES_DICT_PATH = 'project_data/Data/tomatoes_dict.txt'
        SPECTRAL_STATS_FILENAME = 'project_data/Data/Other/final_data/spectral_stats.xlsx'
        LAB_RESULT_FILENAME = 'project_data/Data/Other/final_data/lab_results_with_internal_ids.csv'
        
        # Image processing
        RED_BAND = 80
        GREEN_BAND = 45
        BLUE_BAND = 20
        
        class Augmentation:
            """Data augmentation configuration"""
            AUGMENT_TIMES = 3
            
            @classmethod
            def get_padded_value(cls):
                """Calculate padded value based on augmentation times"""
                if 0 <= cls.AUGMENT_TIMES <= 4:
                    return 0
                elif 5 <= cls.AUGMENT_TIMES <= 30:
                    return int(cls.AUGMENT_TIMES / 3)
                else:
                    return 15
        
        # Dimensions - calculate based on augmentation
        @classmethod
        def get_max_2d_dimensions(cls):
            """Get maximum 2D dimensions including padding"""
            padded_value = cls.Augmentation.get_padded_value()
            return (44 + padded_value, 43 + padded_value)
        

        
        # Additional features and missing values
        ADDITIONAL_FEATURES = [AdditionalFeature.STD, AdditionalFeature.AREA]
        FEELING_MISSING_VALUES_TECHNIQUE = FeelingMissingValuesTechnique.NAN
    
    class Models:
        """Model configuration and types"""
        MODEL_TYPE = ModelType.CNN  # options: XGBOOST, RANDOM_FOREST, CNN, MULTI_HEAD_CNN, CNN_TRANSFORMER, etc.
        MULTI_REGRESSION = True  # If True: single model predicts all quality attributes
        USE_PREVIOUS_MODEL = False
        SAVE_MODEL = False
        
        # Model directories
        BASE_MODEL_DIR = 'project_data/results/models'  # Now using centralized results directory
        MODELS_CONFIG_DIR = {
            'XGBoost': 'XGBoost',
            'CNN': 'CNN'
        }
        
        class Components:
            """Model component usage configuration"""
            USE_COMPONENTS = {
                'reflectance': True,
                'std': True,
                'ndsi': True,
                'indexes': True
            }
            
            # Component processing order for CDAT models
            CDAT_COMPONENT_ORDER = ['std', 'indexes', 'ndsi']
        
        class V2DeepLearning:
            """V2 Deep Learning Models Configuration"""
            # Component fusion method: "sequential" or "parallel"
            FUSION_METHOD = "sequential"
            
            # Spatial downsampling configuration
            DOWNSAMPLING_METHOD = "strided_conv"  # "strided_conv", "max_pool", "avg_pool"
            DOWNSAMPLING_FACTOR = 2
    
    class Training:
        """Training configuration and parameters"""
        USE_VALIDATION = True  # Set to False to skip validation and use 70% train / 30% test split
        BATCH_SIZE = 32
        EPOCHS = 200   # Model training epochs - REDUCED FOR TESTING TO PREVENT ENDLESS LOOPS
        PATIENCE = 10  # Model training patience - DISABLED during feature selection, only for standard/post-FS training
        
        # Results and analysis
        COMPARE_TO_BEST_RESULTS = True
        COMPARING_MATRICS = ['R²']  # ['R²', 'RMSE']
        ANALYZE_FEATURE_IMPORTANCE = False
        SKIP_RESULTS_LOGGING = False
    
    
    class Device:
        """Device configuration for CPU processing"""
        # CPU processing configuration
        MAX_WORKERS = None  # Auto-detect based on CPU cores and memory (None = auto)
        FORCE_SINGLE_THREAD = False  # Force single-threaded execution
        # Note: CPU-only processing with intelligent parallel worker allocation
   
    class Spectral:
        """Spectral data and indices configuration"""
        # Spectral indexes to use
        INDEXES = [
            # SpectralIndex.ZMI,   # ZMI is the best index for tomatoes
            SpectralIndex.GRVI,  # GRVI is the second best index for tomatoes
            # SpectralIndex.INA,   # INA is the third best index for tomatoes
            # SpectralIndex.GLI,
            # SpectralIndex.LYC,
        ]
        
        # Patent bands and NDSI pairs
        # [45, 106], [45, 106, 180], [45, 106, 104, 190], [41, 85, 104, 113, 170], [41, 85, 104, 113, 170, 182],
        # [41, 85, 104, 113, 170, 182, 185], [41, 85, 104, 113, 170, 182, 185, 190], [41, 65, 85, 104, 113, 170, 182, 185, 190], [22, 41, 65, 85, 104, 113, 170, 182, 185, 190]
         #[106, 113, 115, 118, 125] # [41, 85, 104, 113, 170] #104, 113, 170] # [104, 113, 170] [41, 85, 104, 160, 170] [41, 85, 104, 160, 170, 182] 
        PATENT_BANDS = [41, 85, 104, 113, 170] #  104, 113, 170] # [104, 113, 170] [41, 85, 104, 160, 170] [41, 85, 104, 160, 170, 182] 
        NDSI_BAND_PAIRS: List[Tuple[int, int]] = [
            # (85, 170),
            # (41, 170),
            # (104, 113),

            # (45, 106),
            # (104, 170)
        ]
    
    class FeatureSelection:
        """Feature selection configuration"""
        USE_FEATURE_SELECTION = False
        FEATURE_SELECTION_FULL_SPECTRUM = True
        FEATURE_SELECTION_STRATEGY = FeatureSelectionStrategy.MODEL_AGNOSTIC
        NDSI_SELECTION_STRATEGY = NDSISelectionStrategy.BRUTE_FORCE
        
        # Progressive feature selection is now the default
        STAGE_RUN_ORDER = ['bands','std','indexes','ndsi','finetune']
        FINETUNE_FROM_PREVIOUS = False      # fresh re-initialisation
        ENABLE_WEIGHT_TRANSFER = True     # enable weight transfer between stages
        
        class Core:
            """Core feature selection parameters"""
            # Optimal parameters (active)
            K_BANDS = 5        # Optimal: covers spectral diversity without dilution
            A_NDSI = 5         # NDSI pairs: vegetation indices
            B_STD = 5          # STD maps: texture from selected bands
            C_INDEXES = 5      # Index maps: spectral ratios
        
        class Temperature:
            """Temperature scheduling configuration"""
            GUMBEL_TEMPERATURE = 2.0      # High for exploration
            TEMPERATURE_DECAY = 0.985     # Faster decay for proper exploitation
            MIN_TEMPERATURE = 0.3         # Avoid collapse, maintain diversity
            
            # Two-phase approach per stage
            EXPLORATION_RATIO = 0.35      # 35% of stage epochs for exploration
            MIN_EXPLORATION_EPOCHS = 12   # Minimum exploration epochs per stage
            EXPLOITATION_DECAY = 0.95     # Faster decay in exploitation phase
            
            # Adaptive temperature scheduling
            ADAPTIVE_TEMPERATURE = True
            TEMPERATURE_DECAY_IMPROVING = 0.99    # Slow decay when improving
            TEMPERATURE_DECAY_STAGNATING = 0.99   # Faster decay when stagnating
            TEMPERATURE_PATIENCE = 15             # INCREASED: More patience for performance assessment
        
        class Loss:
            """Loss function weights configuration"""
            # Loss component flags
            USE_QUALITY_LOSS = True
            USE_CONFIDENCE_LOSS = True
            USE_DIVERSITY_LOSS = True
            USE_REINFORCEMENT_LOSS = True
            USE_SPARSITY_LOSS = True
            USE_SPECTRAL_LOSS = True
            
            # Critical for performance
            QUALITY_WEIGHT = 2.0          # Spectral quality dominates
            CONFIDENCE_WEIGHT = 1.2       # Reward confident selection
            DIVERSITY_WEIGHT = 0.3        # Encourage spectral diversity
            REINFORCEMENT_WEIGHT = 0.5    # Performance-based reinforcement
            SPARSITY_WEIGHT = 0.1         # Light sparsity constraint
            
            # Regularization
            L2_REGULARIZATION = 0.01      # L2 regularization - based on Stage 2 results
        
        class Attention:
            """Multi-head attention parameters for band selection"""
            ATTENTION_HEADS = 1       # Reduced from 4 to 2 for memory efficiency
            D_MODEL = 128             # Reduced from 128 to 64 for memory efficiency  
            ATTENTION_LAYERS = 1     # Reduced from 2 to 1 for memory efficiency
            
            # Memory-efficient mode for low-memory scenarios
            MEMORY_EFFICIENT_MODE = False
            MEMORY_EFFICIENT_HEADS = 1
            MEMORY_EFFICIENT_D_MODEL = 128
            MEMORY_EFFICIENT_LAYERS = 1
        
        class Spatial:
            """Spatial processing parameters"""
            SPATIAL_KERNEL = 9              # 9x9 local texture analysis (optimal)
            USE_SPATIAL_FEATURES = True     # Include spatial variance in band scoring
        
        class Strategy:
            """Feature selection strategies"""
            # 🎯 FIXED: Use consistent naming - removed INDEX_STRATEGY duplication
            # This should be the single source of truth for index strategy
            INDEX_STRATEGY = "competitive"  # "existing", "learned", "hybrid", or "competitive"
        
        class Output:
            """Feature selection output and visualization"""
            SAVE_VISUALIZATIONS = True
            SAVE_REPORTS = True
            RESULTS_DIR = "project_data/results/feature_selection"
        
        class Advanced:
            """Advanced feature selection improvements"""
            
            # Attribute-specific feature selection weights
            ATTRIBUTE_WEIGHTS = {
                'TSS': 1.5,           # Higher weight - challenging attribute
                'citric_acid': 1.0,   # Standard weight - good performance
                'firmness': 0.8,      # Lower weight - already excellent
                'pH': 1.2,            # Higher weight - moderate performance
                'weight': 0.8,        # Lower weight - already excellent
                'ascorbic_acid': 1.5  # Higher weight - challenging attribute
            }
            
            # Hierarchical feature selection
            HIERARCHICAL = True  # DISABLED to prevent conflict with two-stage feature selection in main_dl_algorithms.py
            STAGE_EPOCHS = {
                'bands': 1,   # Stage 1: Select optimal reflectance bands - BALANCED FOR CONVERGENCE
                'std': 1,     # Stage 2: Select STD bands from reflectance - BALANCED FOR CONVERGENCE
                'ndsi': 1,    # Stage 3: Select NDSI pairs from discovered bands - BALANCED FOR CONVERGENCE
                'indexes': 1,  # Stage 4: Select complementary indexes - BALANCED FOR CONVERGENCE
                'finetune': 1  # Stage 5: Finetune all components - BALANCED FOR CONVERGENCE
            }
            # Stage patience - ONLY applies during exploitation phase
            # Dictionary format: {stage_name: patience_epochs}
            STAGE_PATIENCE = {
                'bands': 10,      # Stage 1: Reduced patience for faster convergence
                'std': 3,        # Stage 2: Quick convergence for STD selection
                'ndsi': 3,       # Stage 3: Quick convergence for NDSI selection
                'indexes': 3     # Stage 4: Moderate patience for index selection
            }
            
            # Two-phase patience control
            EXPLORATION_PATIENCE_DISABLED = True  # No patience during exploration
            EXPLOITATION_PATIENCE_ENABLED = True  # Patience only during exploitation
            SMOOTH_WEIGHT_TRANSITIONS = True  # Enable smooth transitions between weight phases
            
            # Spectral region awareness
            SPECTRAL_DIVERSITY = False
            SPECTRAL_DIVERSITY_WEIGHT = 0.2
            SPECTRAL_REGIONS = {
                'visible': (400, 700),        # Visible light - pigments, chlorophyll
                'near_ir': (700, 1000),       # Near-infrared - cellular structure
                'shortwave_ir': (1000, 2500)  # Shortwave infrared - water, biochemicals
            }
            
            # Validation set optimization
            VALIDATION_OPTIMIZATION = False
            VALIDATION_WEIGHT = 2.0
            TRAINING_WEIGHT = 0.5
            VALIDATION_PATIENCE = 30
            VALIDATION_THRESHOLD = 0.01
            VALIDATION_REINFORCEMENT = True
            VALIDATION_REINFORCEMENT_WEIGHT = 1.0
        
        
        class Convergence:
            """Stage convergence configuration"""
            USE_STAGE_CONVERGENCE = True  # Enable stage-specific convergence detection
            STAGE_CONVERGENCE_THRESHOLD = 0.001  # Minimum improvement to consider stage progressing
    
    # Database configuration removed - no longer using database
    
    class QualityAttributes:
        """Quality attributes prediction configuration"""
        PREDICTED_QUALITY_ATTRIBUTES = [
            'TSS', 'citric_acid', 'firmness', 'pH', 'weight', 'ascorbic_acid'
        ]

# Set static values after class definition
Config.Dataset.PADDED_VALUE = Config.Dataset.Augmentation.get_padded_value()
Config.Dataset.MAX_2D_DIMENSIONS = Config.Dataset.get_max_2d_dimensions()

# Backward compatibility - create module-level variables for existing code
# These should be gradually replaced with the new Config class structure

# Dataset Configuration
CREATE_DATASET = Config.Dataset.CREATE_DATASET
IMAGES_PATH_KIND = Config.Dataset.IMAGES_PATH_KIND
RANDOM_STATE = Config.Dataset.RANDOM_STATE
ONLY_HARVEST_1 = Config.Dataset.ONLY_HARVEST_1
IMAGES_BLOCKER = Config.Dataset.IMAGES_BLOCKER
TOMATOES_DICT_PATH = Config.Dataset.TOMATOES_DICT_PATH
SPECTRAL_STATS_FILENAME = Config.Dataset.SPECTRAL_STATS_FILENAME
LAB_RESULT_FILENAME = Config.Dataset.LAB_RESULT_FILENAME
RED_BAND = Config.Dataset.RED_BAND
GREEN_BAND = Config.Dataset.GREEN_BAND
BLUE_BAND = Config.Dataset.BLUE_BAND
AUGMENT_TIMES = Config.Dataset.Augmentation.AUGMENT_TIMES
PADDED_VALUE = Config.Dataset.PADDED_VALUE
MAX_2D_DIMENSIONS = Config.Dataset.MAX_2D_DIMENSIONS
ADDITIONAL_FEATURES = Config.Dataset.ADDITIONAL_FEATURES
FEELING_MISSING_VALUES_TECHNIQUE = Config.Dataset.FEELING_MISSING_VALUES_TECHNIQUE
CUT_BANDS_FROM_START = Config.Dataset.CUT_BANDS_FROM_START

# Model Configuration
MODEL_TYPE = Config.Models.MODEL_TYPE
MULTI_REGRESSION = Config.Models.MULTI_REGRESSION
USE_PREVIOUS_MODEL = Config.Models.USE_PREVIOUS_MODEL
SAVE_MODEL = Config.Models.SAVE_MODEL
BASE_MODEL_DIR = Config.Models.BASE_MODEL_DIR
models_config_dir = Config.Models.MODELS_CONFIG_DIR
USE_COMPONENTS = Config.Models.Components.USE_COMPONENTS
CDAT_COMPONENT_ORDER = Config.Models.Components.CDAT_COMPONENT_ORDER
V2_DL_FUSION_METHOD = Config.Models.V2DeepLearning.FUSION_METHOD
V2_DL_DOWNSAMPLING_METHOD = Config.Models.V2DeepLearning.DOWNSAMPLING_METHOD
V2_DL_DOWNSAMPLING_FACTOR = Config.Models.V2DeepLearning.DOWNSAMPLING_FACTOR

# Training Configuration
USE_VALIDATION = Config.Training.USE_VALIDATION
BATCH_SIZE = Config.Training.BATCH_SIZE
EPOCHS = Config.Training.EPOCHS
PATIENCE = Config.Training.PATIENCE  # DISABLED during FS phases, only for standard/post-FS training
COMPARE_TO_BEST_RESULTS = Config.Training.COMPARE_TO_BEST_RESULTS
COMPARING_MATRICS = Config.Training.COMPARING_MATRICS
ANALYZE_FEATURE_IMPORTANCE = Config.Training.ANALYZE_FEATURE_IMPORTANCE
SKIP_RESULTS_LOGGING = Config.Training.SKIP_RESULTS_LOGGING

# Spectral Configuration
INDEXES = Config.Spectral.INDEXES
PATENT_BANDS = Config.Spectral.PATENT_BANDS
NDSI_BAND_PAIRS = Config.Spectral.NDSI_BAND_PAIRS

# Feature Selection Configuration
USE_FEATURE_SELECTION = Config.FeatureSelection.USE_FEATURE_SELECTION
FEATURE_SELECTION_FULL_SPECTRUM = Config.FeatureSelection.FEATURE_SELECTION_FULL_SPECTRUM
FEATURE_SELECTION_STRATEGY = Config.FeatureSelection.FEATURE_SELECTION_STRATEGY
NDSI_SELECTION_STRATEGY = Config.FeatureSelection.NDSI_SELECTION_STRATEGY

# Feature Selection Core
FEATURE_SELECTION_K_BANDS = Config.FeatureSelection.Core.K_BANDS
FEATURE_SELECTION_A_NDSI = Config.FeatureSelection.Core.A_NDSI
FEATURE_SELECTION_B_STD = Config.FeatureSelection.Core.B_STD
FEATURE_SELECTION_C_INDEXES = Config.FeatureSelection.Core.C_INDEXES

# Feature Selection Temperature
FEATURE_SELECTION_GUMBEL_TEMPERATURE = Config.FeatureSelection.Temperature.GUMBEL_TEMPERATURE
FEATURE_SELECTION_TEMPERATURE_DECAY = Config.FeatureSelection.Temperature.TEMPERATURE_DECAY
FEATURE_SELECTION_MIN_TEMPERATURE = Config.FeatureSelection.Temperature.MIN_TEMPERATURE
FEATURE_SELECTION_EXPLORATION_RATIO = Config.FeatureSelection.Temperature.EXPLORATION_RATIO
FEATURE_SELECTION_MIN_EXPLORATION_EPOCHS = Config.FeatureSelection.Temperature.MIN_EXPLORATION_EPOCHS
FEATURE_SELECTION_EXPLOITATION_DECAY = Config.FeatureSelection.Temperature.EXPLOITATION_DECAY
FEATURE_SELECTION_ADAPTIVE_TEMPERATURE = Config.FeatureSelection.Temperature.ADAPTIVE_TEMPERATURE
FEATURE_SELECTION_TEMPERATURE_DECAY_IMPROVING = Config.FeatureSelection.Temperature.TEMPERATURE_DECAY_IMPROVING
FEATURE_SELECTION_TEMPERATURE_DECAY_STAGNATING = Config.FeatureSelection.Temperature.TEMPERATURE_DECAY_STAGNATING
FEATURE_SELECTION_TEMPERATURE_PATIENCE = Config.FeatureSelection.Temperature.TEMPERATURE_PATIENCE

# Feature Selection Loss
FEATURE_SELECTION_QUALITY_WEIGHT = Config.FeatureSelection.Loss.QUALITY_WEIGHT
FEATURE_SELECTION_CONFIDENCE_WEIGHT = Config.FeatureSelection.Loss.CONFIDENCE_WEIGHT
FEATURE_SELECTION_DIVERSITY_WEIGHT = Config.FeatureSelection.Loss.DIVERSITY_WEIGHT
FEATURE_SELECTION_REINFORCEMENT_WEIGHT = Config.FeatureSelection.Loss.REINFORCEMENT_WEIGHT
FEATURE_SELECTION_SPARSITY_WEIGHT = Config.FeatureSelection.Loss.SPARSITY_WEIGHT

# Feature Selection Loss Component Flags - NEW
USE_QUALITY_LOSS = Config.FeatureSelection.Loss.USE_QUALITY_LOSS
USE_CONFIDENCE_LOSS = Config.FeatureSelection.Loss.USE_CONFIDENCE_LOSS
USE_DIVERSITY_LOSS = Config.FeatureSelection.Loss.USE_DIVERSITY_LOSS
USE_REINFORCEMENT_LOSS = Config.FeatureSelection.Loss.USE_REINFORCEMENT_LOSS
USE_SPARSITY_LOSS = Config.FeatureSelection.Loss.USE_SPARSITY_LOSS
USE_SPECTRAL_LOSS = Config.FeatureSelection.Loss.USE_SPECTRAL_LOSS

# Feature Selection Attention
FEATURE_SELECTION_ATTENTION_HEADS = Config.FeatureSelection.Attention.ATTENTION_HEADS
FEATURE_SELECTION_D_MODEL = Config.FeatureSelection.Attention.D_MODEL
FEATURE_SELECTION_ATTENTION_LAYERS = Config.FeatureSelection.Attention.ATTENTION_LAYERS
FEATURE_SELECTION_MEMORY_EFFICIENT_MODE = Config.FeatureSelection.Attention.MEMORY_EFFICIENT_MODE
FEATURE_SELECTION_MEMORY_EFFICIENT_HEADS = Config.FeatureSelection.Attention.MEMORY_EFFICIENT_HEADS
FEATURE_SELECTION_MEMORY_EFFICIENT_D_MODEL = Config.FeatureSelection.Attention.MEMORY_EFFICIENT_D_MODEL
FEATURE_SELECTION_MEMORY_EFFICIENT_LAYERS = Config.FeatureSelection.Attention.MEMORY_EFFICIENT_LAYERS

# Feature Selection Spatial
FEATURE_SELECTION_SPATIAL_KERNEL = Config.FeatureSelection.Spatial.SPATIAL_KERNEL
FEATURE_SELECTION_USE_SPATIAL_FEATURES = Config.FeatureSelection.Spatial.USE_SPATIAL_FEATURES

# Feature Selection Strategy - 🎯 FIXED: Single source of truth
FEATURE_SELECTION_INDEX_STRATEGY = Config.FeatureSelection.Strategy.INDEX_STRATEGY
INDEX_STRATEGY = Config.FeatureSelection.Strategy.INDEX_STRATEGY  # Backward compatibility

# Feature Selection Output
FEATURE_SELECTION_SAVE_VISUALIZATIONS = Config.FeatureSelection.Output.SAVE_VISUALIZATIONS
FEATURE_SELECTION_SAVE_REPORTS = Config.FeatureSelection.Output.SAVE_REPORTS
FEATURE_SELECTION_RESULTS_DIR = Config.FeatureSelection.Output.RESULTS_DIR

# Feature Selection Advanced
FEATURE_SELECTION_ATTRIBUTE_WEIGHTS = Config.FeatureSelection.Advanced.ATTRIBUTE_WEIGHTS
FEATURE_SELECTION_HIERARCHICAL = Config.FeatureSelection.Advanced.HIERARCHICAL
FEATURE_SELECTION_STAGE_EPOCHS = Config.FeatureSelection.Advanced.STAGE_EPOCHS
FEATURE_SELECTION_STAGE_PATIENCE = Config.FeatureSelection.Advanced.STAGE_PATIENCE
FEATURE_SELECTION_EXPLORATION_PATIENCE_DISABLED = Config.FeatureSelection.Advanced.EXPLORATION_PATIENCE_DISABLED
FEATURE_SELECTION_EXPLOITATION_PATIENCE_ENABLED = Config.FeatureSelection.Advanced.EXPLOITATION_PATIENCE_ENABLED
FEATURE_SELECTION_SMOOTH_WEIGHT_TRANSITIONS = Config.FeatureSelection.Advanced.SMOOTH_WEIGHT_TRANSITIONS
FEATURE_SELECTION_SPECTRAL_DIVERSITY = Config.FeatureSelection.Advanced.SPECTRAL_DIVERSITY
FEATURE_SELECTION_SPECTRAL_DIVERSITY_WEIGHT = Config.FeatureSelection.Advanced.SPECTRAL_DIVERSITY_WEIGHT
FEATURE_SELECTION_SPECTRAL_REGIONS = Config.FeatureSelection.Advanced.SPECTRAL_REGIONS
FEATURE_SELECTION_VALIDATION_OPTIMIZATION = Config.FeatureSelection.Advanced.VALIDATION_OPTIMIZATION
FEATURE_SELECTION_VALIDATION_WEIGHT = Config.FeatureSelection.Advanced.VALIDATION_WEIGHT
FEATURE_SELECTION_TRAINING_WEIGHT = Config.FeatureSelection.Advanced.TRAINING_WEIGHT
FEATURE_SELECTION_VALIDATION_PATIENCE = Config.FeatureSelection.Advanced.VALIDATION_PATIENCE
FEATURE_SELECTION_VALIDATION_THRESHOLD = Config.FeatureSelection.Advanced.VALIDATION_THRESHOLD
FEATURE_SELECTION_VALIDATION_REINFORCEMENT = Config.FeatureSelection.Advanced.VALIDATION_REINFORCEMENT
FEATURE_SELECTION_VALIDATION_REINFORCEMENT_WEIGHT = Config.FeatureSelection.Advanced.VALIDATION_REINFORCEMENT_WEIGHT


# Feature Selection Convergence
FEATURE_SELECTION_USE_STAGE_CONVERGENCE = Config.FeatureSelection.Convergence.USE_STAGE_CONVERGENCE
FEATURE_SELECTION_STAGE_CONVERGENCE_THRESHOLD = Config.FeatureSelection.Convergence.STAGE_CONVERGENCE_THRESHOLD

# Feature Selection Progressive
FEATURE_SELECTION_PROGRESSIVE = True  # Now always enabled
FEATURE_SELECTION_STAGE_RUN_ORDER = Config.FeatureSelection.STAGE_RUN_ORDER
FEATURE_SELECTION_FINETUNE_FROM_PREVIOUS = Config.FeatureSelection.FINETUNE_FROM_PREVIOUS
FEATURE_SELECTION_ENABLE_WEIGHT_TRANSFER = Config.FeatureSelection.ENABLE_WEIGHT_TRANSFER

# Additional backward compatibility for experiments
FINETUNE_FROM_PREVIOUS = Config.FeatureSelection.FINETUNE_FROM_PREVIOUS
ENABLE_WEIGHT_TRANSFER = Config.FeatureSelection.ENABLE_WEIGHT_TRANSFER

# Feature Selection vs Prediction Weight Balance (for multi-stage experiments)
FEATURE_SELECTION_WEIGHT = 1.0  # Weight for feature selection loss component
PREDICTION_WEIGHT = 1.0  # Weight for prediction loss component

# Quality Attributes
PREDICTED_QUALITY_ATTRIBUTES = Config.QualityAttributes.PREDICTED_QUALITY_ATTRIBUTES

# Database configuration removed - no longer using database
