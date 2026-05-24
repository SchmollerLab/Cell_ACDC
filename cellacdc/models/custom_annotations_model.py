"""Scriptable model rules for custom annotations."""

from __future__ import annotations

import os

import pandas as pd

from cellacdc import load, myutils
from cellacdc.domain.custom_annotations import (
    CustomAnnotationColumnResult,
    CustomAnnotationFrameUpdate,
    custom_annotation_column_exists,
    drop_custom_annotation_column,
    ensure_custom_annotation_column,
    remap_custom_annotation_ids,
    rename_custom_annotation_column,
    update_custom_annotation_frame,
)


class CustomAnnotationsModel:
    """Headless custom annotation table updates."""

    def read_saved_annotations(
        self,
        annotations_path: str,
        *,
        logger_func=None,
    ) -> dict:
        if not os.path.exists(annotations_path):
            return {}
        return load.read_json(annotations_path, logger_func=logger_func)

    def tooltip(self, annotation_state: dict) -> str:
        return myutils.getCustomAnnotTooltip(annotation_state)

    def ensure_column(
        self,
        acdc_df: pd.DataFrame,
        annotation_name: str,
    ) -> CustomAnnotationColumnResult:
        return ensure_custom_annotation_column(acdc_df, annotation_name)

    def column_exists(
        self,
        frame_records,
        annotation_name: str,
        *,
        summary_acdc_df: pd.DataFrame | None = None,
    ) -> bool:
        return custom_annotation_column_exists(
            frame_records,
            annotation_name,
            summary_acdc_df=summary_acdc_df,
        )

    def drop_column(
        self,
        acdc_df: pd.DataFrame | None,
        annotation_name: str,
    ) -> pd.DataFrame | None:
        return drop_custom_annotation_column(acdc_df, annotation_name)

    def rename_column(
        self,
        acdc_df: pd.DataFrame | None,
        old_name: str,
        new_name: str,
    ) -> pd.DataFrame | None:
        return rename_custom_annotation_column(acdc_df, old_name, new_name)

    def remap_ids(self, annotated_ids_by_frame, old_ids, new_ids) -> dict:
        return remap_custom_annotation_ids(
            annotated_ids_by_frame,
            old_ids,
            new_ids,
        )

    def update_frame(
        self,
        acdc_df: pd.DataFrame,
        annotation_name: str,
        annotated_ids,
        *,
        clicked_id: int = 0,
        click_is_active: bool = False,
        existing_ids=None,
    ) -> CustomAnnotationFrameUpdate:
        return update_custom_annotation_frame(
            acdc_df,
            annotation_name,
            annotated_ids,
            clicked_id=clicked_id,
            click_is_active=click_is_active,
            existing_ids=existing_ids,
        )
