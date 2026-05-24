"""Scriptable model rules for undo and redo stacks."""

from __future__ import annotations

from collections import defaultdict


class UndoRedoModel:
    """Headless undo/redo stack operations."""

    def empty_frame_stacks(self, size_t: int) -> list[list]:
        return [[] for _ in range(size_t)]

    def empty_add_point_queue(self):
        return defaultdict(list)

    def trim_stack(self, states: list, *, max_size: int) -> None:
        if len(states) > max_size:
            states.pop(-1)

    def can_undo_labels(self, undo_count: int, states: list) -> bool:
        return undo_count < len(states) - 1

    def can_redo_labels(self, undo_count: int) -> bool:
        return undo_count > 0

    def should_disable_undo_after_cca(
        self,
        undo_count: int,
        states: list,
    ) -> bool:
        return len(states) > undo_count
