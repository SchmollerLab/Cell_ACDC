"""View-model commands for table normalization."""

from __future__ import annotations

import pandas as pd

from cellacdc.myutils import checked_reset_index_Cell_ID, fix_acdc_df_dtypes


class TableMixin:
    """Application-facing commands for dataframe normalization."""

    def checked_reset_index_cell_id(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return checked_reset_index_Cell_ID(dataframe)

    def fix_acdc_df_dtypes(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        return fix_acdc_df_dtypes(dataframe)
