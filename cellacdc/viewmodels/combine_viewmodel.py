"""View-model contract for the Combine Channels feature."""

from __future__ import annotations

from dataclasses import dataclass, field
from cellacdc.models.combine_model import CombineModel


@dataclass(frozen=True)
class CombineViewModel:
    """Presentation logic and commands for the Combine Channels feature."""

    model: CombineModel = field(default_factory=CombineModel)

    def initialize_combine_image_data(self, pos_data):
        """Delegate initialization to model."""
        return self.model.initialize_combine_image_data(pos_data)

    def validate_dimensions(self, ndim: int) -> bool:
        """Delegate validation to model."""
        return self.model.validate_dimensions(ndim)

    def group_processed_data_by_pos(self, processed_data, keys):
        """Delegate grouping to model."""
        return self.model.group_processed_data_by_pos(processed_data, keys)

    def update_combine_image_data(self, pos_data, pos_i_data):
        """Delegate combined image data update to model."""
        return self.model.update_combine_image_data(pos_data, pos_i_data)
