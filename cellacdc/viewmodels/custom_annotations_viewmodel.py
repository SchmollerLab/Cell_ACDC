"""View-model contracts for custom annotations."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from cellacdc.domain.custom_annotations import (
    CustomAnnotationColumnResult,
    CustomAnnotationFrameUpdate,
)
from cellacdc.models.custom_annotations_model import CustomAnnotationsModel


@dataclass(frozen=True)
class CustomAnnotationsViewModel:
    """Application-facing custom annotation table commands."""

    model: CustomAnnotationsModel = field(
        default_factory=CustomAnnotationsModel
    )

    def read_saved_annotations(
        self,
        annotations_path: str,
        *,
        logger_func=None,
    ) -> dict:
        return self.model.read_saved_annotations(
            annotations_path,
            logger_func=logger_func,
        )

    def tooltip(self, annotation_state: dict) -> str:
        return self.model.tooltip(annotation_state)

    def ensure_column(
        self,
        acdc_df: pd.DataFrame,
        annotation_name: str,
    ) -> CustomAnnotationColumnResult:
        return self.model.ensure_column(acdc_df, annotation_name)

    def column_exists(
        self,
        frame_records,
        annotation_name: str,
        *,
        summary_acdc_df: pd.DataFrame | None = None,
    ) -> bool:
        return self.model.column_exists(
            frame_records,
            annotation_name,
            summary_acdc_df=summary_acdc_df,
        )

    def drop_column(
        self,
        acdc_df: pd.DataFrame | None,
        annotation_name: str,
    ) -> pd.DataFrame | None:
        return self.model.drop_column(acdc_df, annotation_name)

    def rename_column(
        self,
        acdc_df: pd.DataFrame | None,
        old_name: str,
        new_name: str,
    ) -> pd.DataFrame | None:
        return self.model.rename_column(acdc_df, old_name, new_name)

    def remap_ids(self, annotated_ids_by_frame, old_ids, new_ids) -> dict:
        return self.model.remap_ids(annotated_ids_by_frame, old_ids, new_ids)

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
        return self.model.update_frame(
            acdc_df,
            annotation_name,
            annotated_ids,
            clicked_id=clicked_id,
            click_is_active=click_is_active,
            existing_ids=existing_ids,
        )
