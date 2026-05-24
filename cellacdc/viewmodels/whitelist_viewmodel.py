"""View-model contract for the Whitelist feature."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Set
from cellacdc.models.whitelist_model import WhitelistModel


@dataclass(frozen=True)
class WhitelistViewModel:
    """Presentation logic and commands for Whitelist management."""

    model: WhitelistModel = field(default_factory=WhitelistModel)

    def filter_existing_ids(self, current_whitelist: Set[int], possible_ids: Set[int]) -> tuple[Set[int], bool]:
        """Filters out non-existing IDs from the current whitelist."""
        return self.model.filter_existing_ids(current_whitelist, possible_ids)

    def get_missing_ids(self, current_ids: Set[int], previous_ids: Set[int]) -> Set[int]:
        """Returns the set of IDs present in current frame but missing from previous frame."""
        return self.model.get_missing_ids(current_ids, previous_ids)

    def check_original_labels(self, whitelist_obj, frame_i: int) -> bool:
        """Delegate label check to model."""
        return self.model.check_original_labels(whitelist_obj, frame_i)

    def get_frames_range(self, frame_i: int) -> list[int]:
        """Delegate range calculation to model."""
        return self.model.get_frames_range(frame_i)

    def get_diff_ids(self, old_ids: Set[int], prev_ids: Set[int], new_ids: Set[int]) -> Set[int]:
        """Delegate difference to model."""
        return self.model.get_diff_ids(old_ids, prev_ids, new_ids)

    def get_whitelist_missing_and_removed_ids(self, whitelist: Set[int], current_ids: Set[int]) -> tuple[list[int], list[int]]:
        """Delegate ID evaluation to model."""
        return self.model.get_whitelist_missing_and_removed_ids(whitelist, current_ids)

    def apply_id_mask(self, curr_lab, og_lab, missing_ids, to_be_removed_ids):
        """Delegate mask updates to model."""
        return self.model.apply_id_mask(curr_lab, og_lab, missing_ids, to_be_removed_ids)

    def construct_og_frame(self, pos_lab, og_frame_base, whitelist_ids, og_ids):
        """Delegate overlay construction to model."""
        return self.model.construct_og_frame(pos_lab, og_frame_base, whitelist_ids, og_ids)
