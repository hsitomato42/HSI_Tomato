"""
🎯 OPTIMAL FEATURE SELECTION CONFIGURATION
Optimized parameters based on analysis of poor performance (R² ~0.3)

Key Issues Identified:
1. Temperature scheduling too aggressive
2. Loss weights not optimized for hyperspectral data
3. Component selection parameters may be too restrictive
4. Validation-based optimization not properly configured

This configuration addresses these issues with data-driven parameter tuning.
"""

# 🎯 PERFORMANCE-OPTIMIZED CORE PARAMETERS
# Based on analysis: increase bands, reduce constraints, improve diversity
FEATURE_SELECTION_K_BANDS = 8  # Increased from 5 for better spectral coverage
FEATURE_SELECTION_A_NDSI = 4   # Increased from 2 for more vegetation indices  
FEATURE_SELECTION_B_STD = 4    # Increased from 3 for better texture analysis
FEATURE_SELECTION_C_INDEXES = 4  # Increased from 3 for more spectral ratios

# 🎯 IMPROVED TEMPERATURE SCHEDULING
# Less aggressive decay to maintain exploration longer
FEATURE_SELECTION_GUMBEL_TEMPERATURE = 3.0     # Higher initial temperature
FEATURE_SELECTION_TEMPERATURE_DECAY = 0.998    # Slower decay
FEATURE_SELECTION_MIN_TEMPERATURE = 0.5        # Higher minimum to maintain diversity

# 🎯 OPTIMIZED LOSS FUNCTION WEIGHTS
# Rebalanced based on hyperspectral data characteristics
FEATURE_SELECTION_QUALITY_WEIGHT = 3.0          # Increase emphasis on prediction quality
FEATURE_SELECTION_CONFIDENCE_WEIGHT = 0.8       # Reduce confidence bias
FEATURE_SELECTION_DIVERSITY_WEIGHT = 0.8        # Increase diversity for better generalization
FEATURE_SELECTION_REINFORCEMENT_WEIGHT = 1.2    # Better performance-based learning
FEATURE_SELECTION_SPARSITY_WEIGHT = 0.05        # Reduce sparsity constraint

# 🎯 ENHANCED ATTENTION PARAMETERS
# Improved capacity for complex spectral relationships
FEATURE_SELECTION_ATTENTION_HEADS = 6       # Balanced complexity
FEATURE_SELECTION_D_MODEL = 256            # Increased from 128 for better representation
FEATURE_SELECTION_ATTENTION_LAYERS = 3     # Increased from 2 for deeper relationships

# 🎯 SPATIAL PROCESSING IMPROVEMENTS
FEATURE_SELECTION_SPATIAL_KERNEL = 7       # Optimal kernel size for tomato features
FEATURE_SELECTION_USE_SPATIAL_FEATURES = True  # Enable spatial variance analysis

# 🎯 HYBRID INDEX STRATEGY
FEATURE_SELECTION_INDEX_STRATEGY = "competitive"  # Competitive selection between existing + learned indices

# 🎯 VALIDATION-DRIVEN OPTIMIZATION (CRITICAL)
FEATURE_SELECTION_VALIDATION_OPTIMIZATION = True
FEATURE_SELECTION_VALIDATION_WEIGHT = 3.0    # Strong emphasis on validation performance
FEATURE_SELECTION_TRAINING_WEIGHT = 0.3      # Reduce training bias
FEATURE_SELECTION_VALIDATION_PATIENCE = 10   # More patient for convergence
FEATURE_SELECTION_VALIDATION_THRESHOLD = 0.005  # Stricter improvement threshold

# 🎯 HIERARCHICAL SELECTION IMPROVEMENTS
FEATURE_SELECTION_HIERARCHICAL = True
FEATURE_SELECTION_STAGE_EPOCHS = {
    'bands': 8,     # More epochs for band selection
    'std': 5,       # Adequate for STD selection
    'ndsi': 5,      # More epochs for NDSI optimization
    'indexes': 4    # More epochs for index learning
}

# 🎯 ATTRIBUTE-SPECIFIC OPTIMIZATION
# Adjusted weights based on observed performance patterns
FEATURE_SELECTION_ATTRIBUTE_WEIGHTS = {
    'TSS': 2.0,           # Poor performance - increase weight
    'citric_acid': 1.0,   # Good performance - maintain
    'firmness': 0.6,      # Excellent performance - reduce weight
    'pH': 1.8,            # Poor performance - increase weight
    'weight': 0.6,        # Excellent performance - reduce weight
    'ascorbic_acid': 2.0  # Poor performance - increase weight
}

# 🎯 SPECTRAL DIVERSITY ENHANCEMENT
FEATURE_SELECTION_SPECTRAL_DIVERSITY = True
FEATURE_SELECTION_SPECTRAL_DIVERSITY_WEIGHT = 0.5  # Increased importance
FEATURE_SELECTION_SPECTRAL_REGIONS = {
    'visible': (400, 700),      # Visible light - pigments, chlorophyll
    'red_edge': (700, 750),     # Red edge - critical for vegetation
    'near_ir': (750, 1000),     # Near-infrared - cellular structure  
    'shortwave_ir': (1000, 2500) # Shortwave infrared - water, biochemicals
}

# 🎯 ADAPTIVE TEMPERATURE SCHEDULING
FEATURE_SELECTION_ADAPTIVE_TEMPERATURE = True
FEATURE_SELECTION_TEMPERATURE_DECAY_IMPROVING = 0.999   # Very slow when improving
FEATURE_SELECTION_TEMPERATURE_DECAY_STAGNATING = 0.996  # Faster when stagnating
FEATURE_SELECTION_TEMPERATURE_PATIENCE = 5

# 🎯 REGULARIZATION AND STABILITY
FEATURE_SELECTION_DROPOUT_RATE = 0.1        # Light regularization
FEATURE_SELECTION_L2_REGULARIZATION = 0.01  # Updated based on Stage 2 results - significantly improves performance
FEATURE_SELECTION_GRADIENT_CLIPPING = 1.0   # Stabilize training

# 🎯 OUTPUT AND MONITORING
FEATURE_SELECTION_SAVE_VISUALIZATIONS = True
FEATURE_SELECTION_SAVE_REPORTS = True
FEATURE_SELECTION_RESULTS_DIR = "outputs/results/feature_selection"
FEATURE_SELECTION_VERBOSE_LOGGING = True    # Enable detailed logging

def apply_optimal_config():
    """
    Apply optimal configuration to config module.
    Call this before running experiments.
    """
    import config
    from utils.logger import get_logger
    
    logger = get_logger("optimal_config")
    
    # Core parameters
    config.FEATURE_SELECTION_K_BANDS = FEATURE_SELECTION_K_BANDS
    config.FEATURE_SELECTION_A_NDSI = FEATURE_SELECTION_A_NDSI
    config.FEATURE_SELECTION_B_STD = FEATURE_SELECTION_B_STD
    config.FEATURE_SELECTION_C_INDEXES = FEATURE_SELECTION_C_INDEXES
    
    # Temperature scheduling
    config.FEATURE_SELECTION_GUMBEL_TEMPERATURE = FEATURE_SELECTION_GUMBEL_TEMPERATURE
    config.FEATURE_SELECTION_TEMPERATURE_DECAY = FEATURE_SELECTION_TEMPERATURE_DECAY
    config.FEATURE_SELECTION_MIN_TEMPERATURE = FEATURE_SELECTION_MIN_TEMPERATURE
    
    # Loss weights
    config.FEATURE_SELECTION_QUALITY_WEIGHT = FEATURE_SELECTION_QUALITY_WEIGHT
    config.FEATURE_SELECTION_CONFIDENCE_WEIGHT = FEATURE_SELECTION_CONFIDENCE_WEIGHT
    config.FEATURE_SELECTION_DIVERSITY_WEIGHT = FEATURE_SELECTION_DIVERSITY_WEIGHT
    config.FEATURE_SELECTION_REINFORCEMENT_WEIGHT = FEATURE_SELECTION_REINFORCEMENT_WEIGHT
    config.FEATURE_SELECTION_SPARSITY_WEIGHT = FEATURE_SELECTION_SPARSITY_WEIGHT
    
    # Attention parameters
    config.FEATURE_SELECTION_ATTENTION_HEADS = FEATURE_SELECTION_ATTENTION_HEADS
    config.FEATURE_SELECTION_D_MODEL = FEATURE_SELECTION_D_MODEL
    config.FEATURE_SELECTION_ATTENTION_LAYERS = FEATURE_SELECTION_ATTENTION_LAYERS
    
    # Validation optimization
    config.FEATURE_SELECTION_VALIDATION_OPTIMIZATION = FEATURE_SELECTION_VALIDATION_OPTIMIZATION
    config.FEATURE_SELECTION_VALIDATION_WEIGHT = FEATURE_SELECTION_VALIDATION_WEIGHT
    config.FEATURE_SELECTION_TRAINING_WEIGHT = FEATURE_SELECTION_TRAINING_WEIGHT
    config.FEATURE_SELECTION_VALIDATION_PATIENCE = FEATURE_SELECTION_VALIDATION_PATIENCE
    config.FEATURE_SELECTION_VALIDATION_THRESHOLD = FEATURE_SELECTION_VALIDATION_THRESHOLD
    
    # Hierarchical selection
    config.FEATURE_SELECTION_HIERARCHICAL = FEATURE_SELECTION_HIERARCHICAL
    config.FEATURE_SELECTION_STAGE_EPOCHS = FEATURE_SELECTION_STAGE_EPOCHS
    
    # Attribute weights
    config.FEATURE_SELECTION_ATTRIBUTE_WEIGHTS = FEATURE_SELECTION_ATTRIBUTE_WEIGHTS
    
    # Spectral diversity
    config.FEATURE_SELECTION_SPECTRAL_DIVERSITY = FEATURE_SELECTION_SPECTRAL_DIVERSITY
    config.FEATURE_SELECTION_SPECTRAL_DIVERSITY_WEIGHT = FEATURE_SELECTION_SPECTRAL_DIVERSITY_WEIGHT
    config.FEATURE_SELECTION_SPECTRAL_REGIONS = FEATURE_SELECTION_SPECTRAL_REGIONS
    
    # Adaptive temperature
    config.FEATURE_SELECTION_ADAPTIVE_TEMPERATURE = FEATURE_SELECTION_ADAPTIVE_TEMPERATURE
    config.FEATURE_SELECTION_TEMPERATURE_DECAY_IMPROVING = FEATURE_SELECTION_TEMPERATURE_DECAY_IMPROVING
    config.FEATURE_SELECTION_TEMPERATURE_DECAY_STAGNATING = FEATURE_SELECTION_TEMPERATURE_DECAY_STAGNATING
    config.FEATURE_SELECTION_TEMPERATURE_PATIENCE = FEATURE_SELECTION_TEMPERATURE_PATIENCE
    
    # Additional parameters if they exist
    if hasattr(config, 'FEATURE_SELECTION_DROPOUT_RATE'):
        config.FEATURE_SELECTION_DROPOUT_RATE = FEATURE_SELECTION_DROPOUT_RATE
    if hasattr(config, 'FEATURE_SELECTION_L2_REGULARIZATION'):
        config.FEATURE_SELECTION_L2_REGULARIZATION = FEATURE_SELECTION_L2_REGULARIZATION
    if hasattr(config, 'FEATURE_SELECTION_GRADIENT_CLIPPING'):
        config.FEATURE_SELECTION_GRADIENT_CLIPPING = FEATURE_SELECTION_GRADIENT_CLIPPING
    if hasattr(config, 'FEATURE_SELECTION_VERBOSE_LOGGING'):
        config.FEATURE_SELECTION_VERBOSE_LOGGING = FEATURE_SELECTION_VERBOSE_LOGGING
    
    logger.experiment("Applied optimal feature selection configuration!")
    logger.info(f"   K_BANDS: {FEATURE_SELECTION_K_BANDS}, A_NDSI: {FEATURE_SELECTION_A_NDSI}")
    logger.info(f"   B_STD: {FEATURE_SELECTION_B_STD}, C_INDEXES: {FEATURE_SELECTION_C_INDEXES}")
    logger.info(f"   Temperature: {FEATURE_SELECTION_GUMBEL_TEMPERATURE} -> {FEATURE_SELECTION_MIN_TEMPERATURE}")
    logger.info(f"   Quality weight: {FEATURE_SELECTION_QUALITY_WEIGHT}, Validation weight: {FEATURE_SELECTION_VALIDATION_WEIGHT}")

if __name__ == "__main__":
    apply_optimal_config() 