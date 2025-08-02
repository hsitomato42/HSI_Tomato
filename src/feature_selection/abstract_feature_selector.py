# feature_selection/abstract_feature_selector.py

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import tensorflow as tf


class AbstractFeatureSelector(ABC):
    """
    Abstract base class for all feature selectors.

    Encapsulates the core operations:
    - Progressive multi-stage feature selection support
    - Modular component handling (bands, STD, NDSI, indexes)
    - Trainable variables exposure per progressive stage
    - Feature selection loss computation integrating validation-aware performance
    - State save/load for persistence
    - Data processing to convert input hyperspectral cube into selected features for model input
    """

    def __init__(
        self,
        original_shape: Tuple[int, int, int],
        components: Dict[str, bool],
        component_allocations: Dict[str, int],
        progressive_mode: bool = True,
        stage_order: Optional[List[str]] = None
    ):
        """
        Initialize the feature selector.

        Args:
            original_shape: (height, width, channels) of original input
            components: Dictionary of enabled components (reflectance, std, ndsi, indexes)
            component_allocations: Band counts per component (e.g. k_bands, b_std, a_ndsi, c_indexes)
            progressive_mode: Use progressive multi-stage training
            stage_order: Stage execution order for progressive training (default provided)
        """
        self.original_shape = original_shape
        self.components = components
        self.component_allocations = component_allocations
        self.progressive_mode = progressive_mode
        self.stage_order = stage_order or ['bands', 'std', 'indexes', 'ndsi', 'finetune']

        self.active_stage: Optional[str] = None
        self._frozen_selectors: set = set()  # Tracks frozen stages in progressive mode
        self._preserved_selections: Dict[str, Any] = {}  # Preserved selections per stage

    @abstractmethod
    def process_data(
        self,
        data: tf.Tensor,
        training: bool = True,
        epoch: Optional[int] = None
    ) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """
        Process raw input hyperspectral data, apply feature selection operations,
        and return processed data for model consumption and selection metadata.

        Args:
            data: Tensor shape [batch, height, width, bands]
            training: Training mode flag (affects stochastic operations)
            epoch: Current training epoch number (for scheduling/temp decay)

        Returns:
            processed_data: Feature-selected tensor compatible with model input
            selection_info: Dictionary containing selection results, losses, stats, etc.
        """
        pass

    @abstractmethod
    def get_trainable_variables(self) -> List[tf.Variable]:
        """
        Return trainable variables relevant for the active progressive stage,
        or all variables if progressive mode disabled.

        This controls which parts of feature selectors get optimized per stage.

        Returns:
            List of trainable TensorFlow variables
        """
        pass

    @abstractmethod
    def compute_loss(
        self,
        selection_info: Dict[str, Any],
        prediction_loss: tf.Tensor,
        **kwargs
    ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """
        Compute the feature selection loss, potentially incorporating validation-aware
        losses, sparsity, confidence, and spectral diversity regularizations.

        Args:
            selection_info: Metadata from process_data, including gate probabilities, sparsity, etc.
            prediction_loss: The main prediction or task loss from the model
            kwargs: Additional optional inputs such as validation losses or attribute-level losses

        Returns:
            total_loss: Single scalar tensor combining all feature selection loss components
            loss_breakdown: Dictionary of each individual loss term for logging/debugging
        """
        pass

    # Component-specific abstract methods to modularize selection logic:

    @abstractmethod
    def select_bands(self, data: tf.Tensor, training: bool) -> Tuple[tf.Tensor, Dict[str, Any]]:
        """
        Select optimal spectral bands from input data.

        Returns:
            selected_bands: Tensor of selected reflectance bands
            metadata: Band selection metadata
        """
        pass

    @abstractmethod
    def compute_std_features(self, selected_bands: tf.Tensor, training: bool) -> tf.Tensor:
        """
        Compute texture STD features from selected bands.
        """
        pass

    @abstractmethod
    def compute_ndsi_features(self, selected_bands: tf.Tensor, training: bool) -> tf.Tensor:
        """
        Compute NDSI features from selected bands.
        """
        pass

    @abstractmethod
    def compute_index_features(self, selected_bands: tf.Tensor, training: bool) -> tf.Tensor:
        """
        Compute spectral index features based on selected bands.
        """
        pass

    # Progressive training related methods:

    def set_active_stage(self, stage: str) -> None:
        """
        Define the currently active progressive training stage.
        """
        if stage not in self.stage_order:
            raise ValueError(f"Invalid stage: {stage}")
        self.active_stage = stage

    def freeze_stage(self, stage: str) -> None:
        """
        Freeze selectors from the specified stage to preserve learned selections.
        """
        self._frozen_selectors.add(stage)

    def preserve_stage_selection(self, stage: str, selection_data: Dict[str, Any]) -> None:
        """
        Save selection results from a completed progressive stage.
        """
        if stage not in self._preserved_selections:
            self._preserved_selections[stage] = {}
        self._preserved_selections[stage].update(selection_data)

    # State management:

    @abstractmethod
    def get_current_selection(self) -> Dict[str, Any]:
        """
        Retrieve the current feature selection state and details.
        """
        pass

    @abstractmethod
    def save_state(self, filepath: str) -> None:
        """
        Save the internal selection state to a file.
        """
        pass

    @abstractmethod
    def load_state(self, filepath: str) -> None:
        """
        Load the internal selection state from a file.
        """
        pass

    # Utility:

    def get_output_channels(self) -> int:
        """
        Compute total output channel count after feature selection.
        """
        total = 0
        if self.components.get('reflectance', False):
            total += self.component_allocations.get('k_bands', 0)
        if self.components.get('std', False):
            total += self.component_allocations.get('b_std', 0)
        if self.components.get('ndsi', False):
            total += self.component_allocations.get('a_ndsi', 0)
        if self.components.get('indexes', False):
            total += self.component_allocations.get('c_indexes', 0)
        return total

    def get_component_dimensions(self) -> Dict[str, int]:
        """
        Return separate channel counts per component plus total.
        """
        dims = {}
        if self.components.get('reflectance', False):
            dims['reflectance'] = self.component_allocations.get('k_bands', 0)
        if self.components.get('std', False):
            dims['std'] = self.component_allocations.get('b_std', 0)
        if self.components.get('ndsi', False):
            dims['ndsi'] = self.component_allocations.get('a_ndsi', 0)
        if self.components.get('indexes', False):
            dims['indexes'] = self.component_allocations.get('c_indexes', 0)
        dims['total_channels'] = self.get_output_channels()
        return dims
