"""View-model commands for CCA workflow operations."""

from __future__ import annotations

from cellacdc.domain.cell_cycle import (
    annotate_division,
    base_cell_cycle_annotation_status,
    collect_existing_new_id_cca_rows_from_frames,
    dead_or_excluded_mother_pairs,
    division_undo_blocking_frame,
    extract_cell_cycle_annotations,
    fix_will_divide_without_next_generation,
    missing_cell_cycle_annotation_items,
    overlay_last_annotated_cca,
    propagate_s_phase_disappearance_divisions,
    reset_will_divide_for_generations,
    s_phase_relative_ids_gone,
    undo_bud_mother_assignment,
    undo_division_annotation,
)
from cellacdc.domain.cell_cycle_auto import (
    apply_auto_cca_assignments,
    auto_cca_assignments_from_cost,
    auto_cca_candidate_mother_ids,
    auto_cca_cost_matrix_from_contours,
    auto_cca_cost_matrix_from_distances,
    auto_cca_repeat_frame_state,
    nearest_point_2d_yx,
    prepare_auto_cca_current_frame,
    uncorrected_new_ids_for_auto_cca,
)
from cellacdc.domain.cell_cycle_deletions import (
    propagate_deleted_cell_cycle_ids,
)
from cellacdc.domain.cell_cycle_divisions import (
    bud_mother_change_eligibility,
    mother_assignment_eligibility,
    previous_relative_status_before_bud_emergence,
    propagate_bud_mother_assignment,
    propagate_manual_division_annotation,
    propagate_swap_mothers_assignment,
    propagate_will_divide,
    swap_mothers_eligibility,
)
from cellacdc.domain.cell_cycle_frames import (
    prepare_missing_cell_cycle_frame_annotations,
)
from cellacdc.domain.cell_cycle_history import (
    known_history_status_for_bud,
    propagate_history_knowledge,
)


class CcaWorkflowViewModel:
    """Application-facing commands for CCA workflows and propagation."""

    def base_status(self, base_values=None):
        return base_cell_cycle_annotation_status(base_values)

    def collect_existing_new_id_rows(self, *args, **kwargs):
        return collect_existing_new_id_cca_rows_from_frames(*args, **kwargs)

    def dead_or_excluded_mother_pairs(self, *args, **kwargs):
        return dead_or_excluded_mother_pairs(*args, **kwargs)

    def division_undo_blocking_frame(self, *args, **kwargs):
        return division_undo_blocking_frame(*args, **kwargs)

    def extract_annotations(self, *args, **kwargs):
        return extract_cell_cycle_annotations(*args, **kwargs)

    def fix_will_divide_without_next_generation(self, *args, **kwargs):
        return fix_will_divide_without_next_generation(*args, **kwargs)

    def missing_annotation_items(self, *args, **kwargs):
        return missing_cell_cycle_annotation_items(*args, **kwargs)

    def overlay_last_annotated(self, *args, **kwargs):
        return overlay_last_annotated_cca(*args, **kwargs)

    def propagate_s_phase_disappearance_divisions(self, *args, **kwargs):
        return propagate_s_phase_disappearance_divisions(*args, **kwargs)

    def reset_will_divide_for_generations(self, *args, **kwargs):
        return reset_will_divide_for_generations(*args, **kwargs)

    def s_phase_relative_ids_gone(self, *args, **kwargs):
        return s_phase_relative_ids_gone(*args, **kwargs)

    def annotate_division(self, *args, **kwargs):
        return annotate_division(*args, **kwargs)

    def undo_division_annotation(self, *args, **kwargs):
        return undo_division_annotation(*args, **kwargs)

    def undo_bud_mother_assignment(self, *args, **kwargs):
        return undo_bud_mother_assignment(*args, **kwargs)

    def apply_auto_assignments(self, *args, **kwargs):
        return apply_auto_cca_assignments(*args, **kwargs)

    def auto_assignments_from_cost(self, *args, **kwargs):
        return auto_cca_assignments_from_cost(*args, **kwargs)

    def auto_candidate_mother_ids(self, *args, **kwargs):
        return auto_cca_candidate_mother_ids(*args, **kwargs)

    def auto_cost_matrix_from_contours(self, *args, **kwargs):
        return auto_cca_cost_matrix_from_contours(*args, **kwargs)

    def auto_cost_matrix_from_distances(self, *args, **kwargs):
        return auto_cca_cost_matrix_from_distances(*args, **kwargs)

    def auto_repeat_frame_state(self, *args, **kwargs):
        return auto_cca_repeat_frame_state(*args, **kwargs)

    def nearest_point_2d_yx(self, *args, **kwargs):
        return nearest_point_2d_yx(*args, **kwargs)

    def prepare_auto_current_frame(self, *args, **kwargs):
        return prepare_auto_cca_current_frame(*args, **kwargs)

    def uncorrected_new_ids_for_auto(self, *args, **kwargs):
        return uncorrected_new_ids_for_auto_cca(*args, **kwargs)

    def propagate_deleted_ids(self, *args, **kwargs):
        return propagate_deleted_cell_cycle_ids(*args, **kwargs)

    def prepare_missing_frame_annotations(self, *args, **kwargs):
        return prepare_missing_cell_cycle_frame_annotations(*args, **kwargs)

    def previous_relative_status_before_bud_emergence(self, *args, **kwargs):
        return previous_relative_status_before_bud_emergence(*args, **kwargs)

    def bud_mother_change_eligibility(self, *args, **kwargs):
        return bud_mother_change_eligibility(*args, **kwargs)

    def mother_assignment_eligibility(self, *args, **kwargs):
        return mother_assignment_eligibility(*args, **kwargs)

    def propagate_bud_mother_assignment(self, *args, **kwargs):
        return propagate_bud_mother_assignment(*args, **kwargs)

    def propagate_manual_division_annotation(self, *args, **kwargs):
        return propagate_manual_division_annotation(*args, **kwargs)

    def propagate_swap_mothers_assignment(self, *args, **kwargs):
        return propagate_swap_mothers_assignment(*args, **kwargs)

    def propagate_will_divide(self, *args, **kwargs):
        return propagate_will_divide(*args, **kwargs)

    def swap_mothers_eligibility(self, *args, **kwargs):
        return swap_mothers_eligibility(*args, **kwargs)

    def known_history_status_for_bud(self, *args, **kwargs):
        return known_history_status_for_bud(*args, **kwargs)

    def propagate_history_knowledge(self, *args, **kwargs):
        return propagate_history_knowledge(*args, **kwargs)
