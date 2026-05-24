"""View-model contracts for object search and navigation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from cellacdc.models.object_search_model import ObjectSearchModel


@dataclass(frozen=True)
class ObjectSearchViewModel:
    """Application-facing commands for finding object IDs across frames."""

    model: ObjectSearchModel = field(default_factory=ObjectSearchModel)

    def find_frame_with_id(
        self,
        pos_data,
        searched_id: int,
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> int | None:
        return self.model.find_frame_with_id(
            pos_data,
            searched_id,
            progress_callback=progress_callback,
        )
