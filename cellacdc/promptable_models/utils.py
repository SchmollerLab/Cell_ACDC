from typing import List, Literal

import numpy as np

import skimage.measure


def build_combined_mask(model_out):
    """Build a single labeled mask from model output, sorted by area (largest first)."""
    rp_model_out = skimage.measure.regionprops(model_out)
    # Sort by area descending so smaller objects paint on top
    rp_model_out = sorted(rp_model_out, key=lambda x: x.area, reverse=True)

    combined = np.zeros(model_out.shape, dtype=model_out.dtype)
    for obj in rp_model_out:
        combined[obj.slice][obj.image] = obj.label
    return combined


def _apply_overlap_rule(
        lab_old,
        lab_new,
        mode: Literal['union', 'intersection']
    ):
    """
    Apply overlap rules between old and new label masks.

    For each overlapping pair (old ID p, new ID q):
    - union: p OR q region → all become p (old absorbs new)
    - intersection: only p AND q → p; p XOR q → 0 (deleted)

    Non-overlapping regions:
    - Old-only IDs: preserved in all modes
    - New-only IDs: added in 'union', deleted in 'intersection'
    """
    result = np.zeros_like(lab_old)

    old_ids = set(np.unique(lab_old)) - {0}
    new_ids = set(np.unique(lab_new)) - {0}

    # Track which new IDs overlap with old
    overlapping_new_ids = set()

    # Process each old object
    for p in old_ids:
        p_mask = lab_old == p

        # Find new IDs that overlap with this old ID
        overlapping_q_ids = set(np.unique(lab_new[p_mask])) - {0}
        overlapping_new_ids.update(overlapping_q_ids)

        if not overlapping_q_ids:
            # No overlap - keep old object as is
            result[p_mask] = p
        else:
            # Has overlap with one or more new IDs
            for q in overlapping_q_ids:
                q_mask = lab_new == q

                p_and_q = np.logical_and(p_mask, q_mask)  # Overlap region
                p_only = np.logical_and(p_mask, ~q_mask)  # Old only
                q_only = np.logical_and(q_mask, ~p_mask)  # New only

                if mode == 'union':
                    # p OR q → all become p
                    result[p_and_q] = p
                    result[p_only] = p
                    result[q_only] = p

                elif mode == 'intersection':
                    # Only p AND q → p; p XOR q → 0
                    result[p_and_q] = p
                    # p_only and q_only become 0 (already 0 in result)

    # Handle new IDs that don't overlap with any old ID
    non_overlapping_new_ids = new_ids - overlapping_new_ids
    for q in non_overlapping_new_ids:
        q_mask = lab_new == q
        if mode == 'union':
            result[q_mask] = q
        # In 'intersection' mode, non-overlapping new IDs are not added

    return result


def insert_model_output_into_labels(
        lab,
        model_out,
        edited_IDs: int | List[int] = 0,
    ):
    """
    Combine model output with existing labels using three strategies.

    Parameters
    ----------
    lab : np.ndarray
        Existing label mask
    model_out : np.ndarray
        New label mask from model
    edited_IDs : int or List[int]
        Deprecated, kept for API compatibility. No longer used.

    Returns
    -------
    lab_new, lab_union, lab_intersection : tuple of np.ndarray
        - lab_new: just the model output (new masks only)
        - lab_union: old absorbs new in overlap regions
        - lab_intersection: only overlap regions kept
    """
    # Build combined new mask from model output (sorted by area, smallest on top)
    lab_new = build_combined_mask(model_out)

    # Apply overlap rules for union and intersection
    lab_union = _apply_overlap_rule(lab, lab_new, mode='union')
    lab_intersection = _apply_overlap_rule(lab, lab_new, mode='intersection')

    return lab_new, lab_union, lab_intersection
