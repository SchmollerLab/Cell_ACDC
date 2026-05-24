"""View-model commands for lineage tree annotations."""

from __future__ import annotations

import pandas as pd

from cellacdc.domain.lineage import (
    LineageAnnotationsRemovalResult,
    LineageFutureRemovalResult,
    has_lineage_tree_annotations,
    remove_future_lineage_tree_annotations,
    remove_lineage_tree_annotations,
)
from cellacdc.myutils import get_obj_by_label, sort_IDs_dist


class LineageViewModel:
    """Application-facing commands for lineage annotation tables."""

    def has_lineage_tree_annotations(
        self,
        acdc_df: pd.DataFrame | None,
        lineage_tree=None,
        *,
        parent_column: str = 'parent_ID_tree',
    ) -> bool:
        return has_lineage_tree_annotations(
            acdc_df,
            lineage_tree,
            parent_column=parent_column,
        )

    def remove_lineage_tree_annotations(
        self,
        acdc_df: pd.DataFrame | None,
        lineage_tree_colnames,
    ) -> LineageAnnotationsRemovalResult:
        return remove_lineage_tree_annotations(acdc_df, lineage_tree_colnames)

    def remove_future_lineage_tree_annotations(
        self,
        frame_records,
        lineage_tree_colnames,
        from_frame_i: int,
        *,
        size_t: int | None = None,
        acdc_key: str = 'acdc_df',
    ) -> LineageFutureRemovalResult:
        return remove_future_lineage_tree_annotations(
            frame_records,
            lineage_tree_colnames,
            from_frame_i,
            size_t=size_t,
            acdc_key=acdc_key,
        )

    def object_by_label(self, regionprops, label):
        return get_obj_by_label(regionprops, label)

    def sort_ids_by_distance(self, regionprops, *, point=None, label=None):
        return sort_IDs_dist(regionprops, point=point, ID=label)
