"""Qt-free model rules for cell-cycle GUI workflows."""

from __future__ import annotations

from dataclasses import dataclass
import pandas as pd



@dataclass(frozen=True)
class AnnotatedEditWarningPlan:
    """Decision for editing a frame with existing annotations."""

    proceed_without_warning: bool
    update_images: bool = False
    should_prompt: bool = False
    warn_type: str | None = None


class CellCycleModel:
    """Headless cell-cycle workflow rules."""

    def annotated_edit_warning_plan(
        self,
        *,
        is_snapshot: bool,
        acdc_df_missing: bool,
        lineage_tree_missing: bool,
        cell_cycle_stage_present: bool,
        lineage_tree_present: bool,
        remembered_skip_warning: bool,
    ) -> AnnotatedEditWarningPlan:
        if is_snapshot:
            return AnnotatedEditWarningPlan(proceed_without_warning=True)

        no_annotation_source = acdc_df_missing and lineage_tree_missing
        no_annotations = not cell_cycle_stage_present and not lineage_tree_present
        if no_annotation_source or no_annotations or remembered_skip_warning:
            return AnnotatedEditWarningPlan(
                proceed_without_warning=True,
                update_images=True,
            )

        warn_type = (
            'cell cycle annotations'
            if cell_cycle_stage_present
            else 'lineage tree annotations'
        )
        return AnnotatedEditWarningPlan(
            proceed_without_warning=False,
            should_prompt=True,
            warn_type=warn_type,
        )

    def check_mothers_exclusion_or_dead(
        self,
        acdc_df: pd.DataFrame,
        mother_ids: list[int],
    ) -> list[int]:
        """Checks tracking rules for cell exclusions or deaths."""
        if acdc_df is None or not mother_ids:
            return []

        valid_ids = [m_id for m_id in mother_ids if m_id in acdc_df.index]
        if not valid_ids:
            return []

        mothers_df = acdc_df.loc[valid_ids]
        excluded_mask = (
            (mothers_df.get('is_cell_dead', 0) > 0)
            | (mothers_df.get('is_cell_excluded', 0) > 0)
        )
        return mothers_df[excluded_mask].index.tolist()

    def evaluate_sister_relations(
        self,
        prev_cca_df: pd.DataFrame,
        current_ids: set[int],
    ) -> list[int]:
        """Determines S-phase mother-bud dependencies and sister relation tracking rules."""
        if prev_cca_df is None or not current_ids:
            return []

        current_ids_set = set(current_ids)
        disappeared_ids = []
        for cc_series in prev_cca_df.itertuples():
            if getattr(cc_series, 'cell_cycle_stage', None) != 'S':
                continue

            cell_id = cc_series.Index
            relative_id = getattr(cc_series, 'relative_ID', -1)
            if relative_id == -1:
                continue
            if relative_id not in current_ids_set and cell_id in current_ids_set:
                disappeared_ids.append(relative_id)

        return disappeared_ids

