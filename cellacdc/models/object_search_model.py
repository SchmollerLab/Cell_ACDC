"""Scriptable model rules for object search."""

from __future__ import annotations

from collections.abc import Callable

from cellacdc.domain.object_search import find_frame_with_id


class ObjectSearchModel:
    """Headless object-search operations."""

    def find_frame_with_id(
        self,
        pos_data,
        searched_id: int,
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> int | None:
        return find_frame_with_id(
            pos_data.segm_data,
            pos_data.allData_li,
            searched_id,
            progress_callback=progress_callback,
        )
