# Models/dl_models/CNNModel.py

from ..base_classes.BaseDLModel import BaseDLModel
from typing import Optional, List, Dict
import tensorflow as tf
from keras._tf_keras.keras.models import Sequential, Model
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import src.config as config
from src.config.enums import ModelType, SpectralIndex


class CNNModel(BaseDLModel):
    """
    CNN Model that can operate in two modes:
    1. Single prediction mode (MULTI_REGRESSION=False): Separate CNN for each quality attribute
    2. Multi-regression mode (MULTI_REGRESSION=True): Single CNN predicting all quality attributes
    """

    def __init__(
        self,
        model_name: str,
        model_filename: str,
        model_shape: tuple[int, int, int],
        components: dict,
        component_dimensions: dict,
        selected_bands: List[int],
        predicted_attributes: list[str] = config.PREDICTED_QUALITY_ATTRIBUTES,
        selected_indexes: Optional[List[SpectralIndex]] = None
    ):
        """
        Initialize CNNModel with support for both single and multi-regression modes.
        """
        super().__init__(
            model_type=ModelType.CNN,
            model_name=model_name,
            model_filename=model_filename,
            model_shape=model_shape,
            components=components,
            component_dimensions=component_dimensions,
            selected_bands=selected_bands,
            predicted_attributes=predicted_attributes,
            selected_indexes=selected_indexes
        )

    def _build_model_for_attr(self) -> tf.keras.Model:
        """
        Builds a single-output CNN (Dense(1)) for a single attribute.
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.model_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            # Single output (one attribute)
            Dense(1, activation='linear')
        ])

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model
        
    def _build_multi_output_model(self) -> tf.keras.Model:
        """
        Builds a multi-output CNN that predicts all quality attributes simultaneously.
        """
        inputs = Input(shape=self.model_shape)
        
        # CNN feature extraction layers
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Multi-output layer - one output for all attributes
        outputs = Dense(len(self.predicted_attributes), activation='linear', name='multi_output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='CNNMultiRegression')
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print(f"[CNNModel] Built multi-output model with {len(self.predicted_attributes)} outputs")
        return model

    def _build_progressive_stage_model(self, stage_channels: int, stage_name: str, previous_model: tf.keras.Model = None) -> tf.keras.Model:
        """
        Build a new CNN model for progressive feature selection with dynamic channel count.
        
        Args:
            stage_channels: Number of input channels for this stage
            stage_name: Name of the current stage (bands, std, indexes, finetune)
            previous_model: Previous stage model to transfer weights from (optional)
            
        Returns:
            A new Keras model configured for the current stage
        """
        # Get image dimensions from the original model shape
        height, width = self.model_shape[:2]
        
        # Create input shape with dynamic channel count
        input_shape = (height, width, stage_channels)
        
        print(f"[CNNModel] Building progressive stage model for '{stage_name}' with {stage_channels} channels")
        print(f"[CNNModel] Input shape: {input_shape}")
        
        # Build model based on regression mode
        if config.MULTI_REGRESSION:
            return self._build_progressive_multi_output_model(input_shape, stage_name)
        else:
            return self._build_progressive_single_output_model(input_shape, stage_name)
    
    def _build_progressive_multi_output_model(self, input_shape: tuple, stage_name: str) -> tf.keras.Model:
        """
        Build a multi-output CNN model for progressive feature selection.
        
        Args:
            input_shape: Input shape (H, W, C) where C is the dynamic channel count
            stage_name: Name of the current stage
            
        Returns:
            Multi-output CNN model
        """
        inputs = Input(shape=input_shape)
        
        # CNN feature extraction layers
        x = Conv2D(32, (3, 3), activation='relu')(inputs)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Conv2D(128, (3, 3), activation='relu')(x)
        x = MaxPooling2D((2, 2))(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        
        # Multi-output layer - one output for all attributes
        outputs = Dense(len(self.predicted_attributes), activation='linear', name='multi_output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name=f'CNNMultiRegression_{stage_name}')
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print(f"[CNNModel] Built progressive multi-output model for '{stage_name}' with {len(self.predicted_attributes)} outputs")
        return model
    
    def _build_progressive_single_output_model(self, input_shape: tuple, stage_name: str) -> tf.keras.Model:
        """
        Build a single-output CNN model for progressive feature selection.
        
        Args:
            input_shape: Input shape (H, W, C) where C is the dynamic channel count
            stage_name: Name of the current stage
            
        Returns:
            Single-output CNN model
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.3),
            # Single output (one attribute)
            Dense(1, activation='linear')
        ], name=f'CNNSingleOutput_{stage_name}')

        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print(f"[CNNModel] Built progressive single-output model for '{stage_name}'")
        return model
