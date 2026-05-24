"""View-model contracts for undo and redo stack handling."""

from __future__ import annotations

from dataclasses import dataclass, field

from cellacdc.models.undo_redo_model import UndoRedoModel


@dataclass(frozen=True)
class UndoRedoViewModel:
    """Application-facing commands for undo/redo stack decisions."""

    model: UndoRedoModel = field(default_factory=UndoRedoModel)

    def empty_frame_stacks(self, size_t: int) -> list[list]:
        return self.model.empty_frame_stacks(size_t)

    def empty_add_point_queue(self):
        return self.model.empty_add_point_queue()

    def trim_label_states(self, states: list) -> None:
        self.model.trim_stack(states, max_size=5)

    def trim_cca_states(self, states: list) -> None:
        self.model.trim_stack(states, max_size=10)

    def can_undo_labels(self, undo_count: int, states: list) -> bool:
        return self.model.can_undo_labels(undo_count, states)

    def can_redo_labels(self, undo_count: int) -> bool:
        return self.model.can_redo_labels(undo_count)

    def should_disable_undo_after_cca(
        self,
        undo_count: int,
        states: list,
    ) -> bool:
        return self.model.should_disable_undo_after_cca(undo_count, states)
