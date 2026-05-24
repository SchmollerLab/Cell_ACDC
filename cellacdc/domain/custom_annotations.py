"""Pure custom annotation table operations."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CustomAnnotationColumnResult:
    """Frame table with a custom annotation column plus annotated IDs."""

    dataframe: pd.DataFrame
    annotated_ids: list[int]


@dataclass(frozen=True)
class CustomAnnotationFrameUpdate:
    """Frame custom annotation update plus IDs present in current labels."""

    dataframe: pd.DataFrame
    annotated_ids: list[int]
    present_annotated_ids: list[int]


def _cell_ids_from_index_or_column(df: pd.DataFrame) -> list[int]:
    if 'Cell_ID' in df.columns:
        return [int(cell_id) for cell_id in df['Cell_ID'].to_list()]
    if isinstance(df.index, pd.MultiIndex) and 'Cell_ID' in df.index.names:
        cell_ids = df.index.get_level_values('Cell_ID')
        return [int(cell_id) for cell_id in cell_ids.to_list()]
    return [int(cell_id) for cell_id in df.index.to_list()]


def ensure_custom_annotation_column(
    acdc_df: pd.DataFrame,
    annotation_name: str,
) -> CustomAnnotationColumnResult:
    """Return ``acdc_df`` with a 0/1 custom annotation column."""
    updated_df = acdc_df.copy()
    if annotation_name not in updated_df.columns:
        updated_df[annotation_name] = 0
        return CustomAnnotationColumnResult(updated_df, [])

    updated_df[annotation_name] = updated_df[annotation_name].astype(int)
    annotated_df = updated_df[updated_df[annotation_name] == 1]
    return CustomAnnotationColumnResult(
        dataframe=updated_df,
        annotated_ids=_cell_ids_from_index_or_column(annotated_df),
    )


def custom_annotation_column_exists(
    frame_records,
    annotation_name: str,
    *,
    summary_acdc_df: pd.DataFrame | None = None,
) -> bool:
    """Return whether a custom annotation column exists in any metadata table."""
    for frame_record in frame_records:
        acdc_df = frame_record['acdc_df']
        if acdc_df is None:
            continue
        if annotation_name in acdc_df.columns:
            return True

    return (
        summary_acdc_df is not None
        and annotation_name in summary_acdc_df.columns
    )


def drop_custom_annotation_column(
    acdc_df: pd.DataFrame | None,
    annotation_name: str,
) -> pd.DataFrame | None:
    """Return ``acdc_df`` without one custom annotation column."""
    if acdc_df is None:
        return None
    return acdc_df.drop(columns=annotation_name, errors='ignore')


def rename_custom_annotation_column(
    acdc_df: pd.DataFrame | None,
    old_name: str,
    new_name: str,
) -> pd.DataFrame | None:
    """Return ``acdc_df`` with one custom annotation column renamed."""
    if acdc_df is None:
        return None
    return acdc_df.rename(columns={old_name: new_name})


def remap_custom_annotation_ids(
    annotated_ids_by_frame,
    old_ids,
    new_ids,
) -> dict:
    """Return custom annotation ID lists remapped after label-ID changes."""
    id_mapper = dict(zip(old_ids, new_ids))
    return {
        frame_i: [id_mapper[cell_id] for cell_id in annotated_ids]
        for frame_i, annotated_ids in annotated_ids_by_frame.items()
    }


def update_custom_annotation_frame(
    acdc_df: pd.DataFrame,
    annotation_name: str,
    annotated_ids,
    *,
    clicked_id: int = 0,
    click_is_active: bool = False,
    existing_ids=None,
) -> CustomAnnotationFrameUpdate:
    """Return frame table and ID list after one custom annotation action."""
    updated_df = acdc_df.copy()
    updated_ids = list(annotated_ids)
    clicked_id = int(clicked_id)

    if click_is_active and clicked_id > 0:
        if clicked_id in updated_ids:
            updated_ids.remove(clicked_id)
            if clicked_id in updated_df.index:
                updated_df.at[clicked_id, annotation_name] = 0
        else:
            updated_ids.append(clicked_id)

    existing_ids = set(updated_ids if existing_ids is None else existing_ids)
    present_annotated_ids = [
        annot_id for annot_id in updated_ids
        if annot_id in existing_ids
    ]
    for annot_id in present_annotated_ids:
        updated_df.at[annot_id, annotation_name] = 1

    return CustomAnnotationFrameUpdate(
        dataframe=updated_df,
        annotated_ids=updated_ids,
        present_annotated_ids=present_annotated_ids,
    )
