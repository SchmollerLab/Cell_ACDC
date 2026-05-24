"""View-model contracts for measurement workflows."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.measurements_model import MeasurementsModel


@dataclass(frozen=True)
class MeasurementsViewModel:
    """Application-facing commands for measurement calculations."""

    model: MeasurementsModel = field(default_factory=MeasurementsModel)

    def rotational_volume(
        self,
        obj,
        physical_size_y=1,
        physical_size_x=1,
        logger=None,
    ):
        return self.model.rotational_volume(
            obj,
            physical_size_y,
            physical_size_x,
            logger=logger,
        )

    def custom_metrics_instructions(self):
        return self.model.custom_metrics_instructions()

    def metrics_examples_path(self):
        return self.model.metrics_examples_path()

    def all_acdc_df_columns(self, all_pos_data):
        return self.model.all_acdc_df_columns(all_pos_data)

    def not_loaded_channels(self, all_channel_names, loaded_channel_names):
        return self.model.not_loaded_channels(
            all_channel_names, loaded_channel_names
        )

    def drop_unchecked_measurements(self, acdc_df, columns, regionprops):
        return self.model.drop_unchecked_measurements(
            acdc_df,
            columns,
            regionprops,
        )
