"""View-model composition for cell-cycle annotation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

from cellacdc.models.cell_cycle_model import (
    AnnotatedEditWarningPlan,
    CellCycleModel,
)

from .cca_edits import CcaEditViewModel
from .cca_workflows import CcaWorkflowViewModel
from .lineage import LineageViewModel
from .model_registry import ModelRegistryViewModel


@dataclass(frozen=True)
class CellCycleViewModel:
    """GUI-facing commands for cell-cycle annotation workflows."""

    model: CellCycleModel = field(default_factory=CellCycleModel)
    cca_edits: CcaEditViewModel = field(default_factory=CcaEditViewModel)
    cca_workflows: CcaWorkflowViewModel = field(
        default_factory=CcaWorkflowViewModel
    )
    lineage: LineageViewModel = field(default_factory=LineageViewModel)
    model_registry: ModelRegistryViewModel = field(
        default_factory=ModelRegistryViewModel
    )

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
        return self.model.annotated_edit_warning_plan(
            is_snapshot=is_snapshot,
            acdc_df_missing=acdc_df_missing,
            lineage_tree_missing=lineage_tree_missing,
            cell_cycle_stage_present=cell_cycle_stage_present,
            lineage_tree_present=lineage_tree_present,
            remembered_skip_warning=remembered_skip_warning,
        )

    def check_mothers_exclusion_or_dead(
        self,
        acdc_df: pd.DataFrame,
        mother_ids: list[int],
    ) -> list[int]:
        """Wrap check_mothers_exclusion_or_dead model call."""
        return self.model.check_mothers_exclusion_or_dead(acdc_df, mother_ids)

    def evaluate_sister_relations(
        self,
        prev_cca_df: pd.DataFrame,
        current_ids: set[int],
    ) -> list[int]:
        """Wrap evaluate_sister_relations model call."""
        return self.model.evaluate_sister_relations(prev_cca_df, current_ids)

