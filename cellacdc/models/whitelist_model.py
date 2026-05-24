"""Scriptable model rules for the Whitelist feature."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Set, List
import numpy as np


@dataclass(frozen=True)
class WhitelistModel:
    """Headless decisions and calculations for Whitelist management."""

    def filter_existing_ids(self, current_whitelist: Set[int], possible_ids: Set[int]) -> tuple[Set[int], bool]:
        """Filters out non-existing IDs from the current whitelist.
        
        Returns a tuple: (filtered_whitelist, is_any_id_non_existing)
        """
        is_any_id_non_existing = False
        filtered_whitelist = set(current_whitelist)
        for ID in current_whitelist:
            if ID not in possible_ids:
                is_any_id_non_existing = True
                filtered_whitelist.discard(ID)
        return filtered_whitelist, is_any_id_non_existing

    def get_missing_ids(self, current_ids: Set[int], previous_ids: Set[int]) -> Set[int]:
        """Returns the set of IDs present in current frame but missing from previous frame."""
        return set(current_ids) - set(previous_ids)

    def check_original_labels(self, whitelist_obj, frame_i: int) -> bool:
        """Checks if original label data is allocated and valid for the frame."""
        if whitelist_obj is None:
            return False
        if whitelist_obj.originalLabsIDs is None:
            return False
        if frame_i >= len(whitelist_obj.originalLabsIDs) or whitelist_obj.originalLabsIDs[frame_i] is None:
            return False
        return True

    def get_frames_range(self, frame_i: int) -> list[int]:
        """Calculates navigation frame ranges for label loading."""
        if frame_i > 0:
            return [frame_i - 1, frame_i]
        return [frame_i]

    def get_diff_ids(self, old_ids: Set[int], prev_ids: Set[int], new_ids: Set[int]) -> Set[int]:
        """Computes tracking difference intersection (new_ids - old_ids) & prev_ids."""
        return (new_ids - old_ids) & prev_ids

    def get_whitelist_missing_and_removed_ids(self, whitelist: Set[int], current_ids: Set[int]) -> tuple[list[int], list[int]]:
        """Finds IDs that are missing from current_ids and IDs to be removed from current_ids."""
        missing_ids = list(whitelist - current_ids)
        to_be_removed_ids = list(current_ids - whitelist)
        return missing_ids, to_be_removed_ids

    def apply_id_mask(
        self,
        curr_lab: np.ndarray,
        og_lab: np.ndarray | None,
        missing_ids: list[int] | np.ndarray,
        to_be_removed_ids: list[int] | np.ndarray
    ) -> np.ndarray:
        """Applies missing and removed ID masks to the label array."""
        updated_lab = curr_lab.copy().astype(np.int32)
        missing_ids = np.array(missing_ids, dtype=np.int32)
        to_be_removed_ids = np.array(to_be_removed_ids, dtype=np.int32)

        if missing_ids.size > 0 and og_lab is not None:
            mask = np.isin(og_lab, missing_ids)
            updated_lab[mask] = og_lab[mask]

        if to_be_removed_ids.size > 0:
            updated_lab[np.isin(updated_lab, to_be_removed_ids)] = 0

        return updated_lab

    def construct_og_frame(
        self,
        pos_lab: np.ndarray,
        og_frame_base: np.ndarray,
        whitelist_ids: Set[int],
        og_ids: Set[int]
    ) -> np.ndarray:
        """Constructs original labels overlay using np.isin masking."""
        og_frame = og_frame_base.copy()
        
        ids_to_update = whitelist_ids & og_ids
        if ids_to_update:
            mask = np.isin(og_frame, list(ids_to_update))
            og_frame[mask] = 0
            mask = np.isin(pos_lab, list(ids_to_update))
            og_frame[mask] = pos_lab[mask]
        
        ids_to_add = whitelist_ids - og_ids
        if ids_to_add:
            mask = np.isin(pos_lab, list(ids_to_add))
            og_frame[mask] = pos_lab[mask]
            
        return og_frame
