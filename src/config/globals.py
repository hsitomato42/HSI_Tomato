from src.config.enums import ModelType


ML_MODELS = [ModelType.XGBOOST, ModelType.RANDOM_FOREST]
DL_MODELS = [
    ModelType.CNN,
    ModelType.MULTI_HEAD_CNN,
    ModelType.CNN_TRANSFORMER,
    ModelType.SPECTRAL_TRANSFORMER,
    ModelType.VIT
]
