"""Scriptable model rules for measurement workflows."""

from __future__ import annotations

from cellacdc.cca_functions import _calc_rot_vol
from cellacdc import measurements


class MeasurementsModel:
    """Headless measurement calculation and setup rules."""

    def rotational_volume(
        self,
        obj,
        physical_size_y=1,
        physical_size_x=1,
        logger=None,
    ):
        return _calc_rot_vol(
            obj,
            physical_size_y,
            physical_size_x,
            logger=logger,
        )

    def custom_metrics_instructions(self):
        return measurements.add_metrics_instructions()

    def metrics_examples_path(self):
        return measurements.metrics_path

    def all_acdc_df_columns(self, all_pos_data):
        columns = set()
        for pos_data in all_pos_data:
            for data_dict in pos_data.allData_li:
                acdc_df = data_dict['acdc_df']
                if acdc_df is None:
                    continue
                columns.update(acdc_df.columns)
        return columns

    def not_loaded_channels(self, all_channel_names, loaded_channel_names):
        return [c for c in all_channel_names if c not in loaded_channel_names]

    def drop_unchecked_measurements(self, acdc_df, columns, regionprops):
        if acdc_df is None:
            return None
        acdc_df = acdc_df.drop(columns=columns, errors='ignore')
        for col_rp in regionprops:
            drop_df_rp = acdc_df.filter(regex=fr'{col_rp}.*', axis=1)
            drop_cols_rp = drop_df_rp.columns
            acdc_df = acdc_df.drop(columns=drop_cols_rp, errors='ignore')
        return acdc_df
