"""Scriptable object counting and label-frame helpers."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import skimage.measure


def _record_get(record, key, default=None):
    if hasattr(record, 'get'):
        return record.get(key, default)
    return getattr(record, key, default)


def current_labels(
    pos_data,
    *,
    curr_lab: np.ndarray | None = None,
    frame_i: int | None = None,
) -> np.ndarray | None:
    """Resolve the current labels from live, cached, or persisted frame data."""
    if frame_i is None:
        frame_i = pos_data.frame_i

    if curr_lab is None and frame_i == pos_data.frame_i:
        curr_lab = pos_data.lab

    if curr_lab is None:
        try:
            labels = _record_get(pos_data.allData_li[frame_i], 'labels')
            curr_lab = labels.copy()
        except (AttributeError, IndexError, TypeError):
            pass

    if curr_lab is None:
        try:
            curr_lab = pos_data.segm_data[frame_i].copy()
        except (AttributeError, IndexError, TypeError):
            pass

    return curr_lab


def collect_all_ids(pos_data, *, only_visited: bool = False) -> set[int]:
    """Collect all object IDs across visited or available segmentation frames."""
    all_ids = set()
    for frame_i in range(len(pos_data.segm_data)):
        if frame_i >= len(pos_data.allData_li):
            break

        frame_record = pos_data.allData_li[frame_i]
        lab = _record_get(frame_record, 'labels')
        if lab is None and only_visited:
            break

        if lab is None:
            regionprops = skimage.measure.regionprops(pos_data.segm_data[frame_i])
        else:
            regionprops = _record_get(frame_record, 'regionprops')
            if regionprops is None:
                regionprops = skimage.measure.regionprops(lab)

        all_ids.update(int(obj.label) for obj in regionprops)

    return all_ids


def snapshot_object_counts(
    positions,
    current_pos_i: int,
    *,
    current_lab_2d=None,
    include_current_z_slice: bool = False,
    path_exists: Callable[[str], bool],
) -> dict[str, int]:
    """Count objects across loaded snapshot positions."""
    pos_data = positions[current_pos_i]
    counts = {
        'In current position': len(pos_data.IDs),
        'In all visited positions (current session)': 0,
        'In all visited positions (previous sessions)': 0,
        'In all loaded positions': 0,
    }
    if include_current_z_slice and current_lab_2d is not None:
        counts['In current z-slice'] = len(
            skimage.measure.regionprops(current_lab_2d)
        )

    for position in positions:
        ids = _record_get(position.allData_li[0], 'IDs', [])
        if path_exists(position.acdc_output_csv_path):
            counts['In all visited positions (previous sessions)'] += len(ids)

        if ids:
            num_objects = len(ids)
        else:
            regionprops = skimage.measure.regionprops(position.segm_data[0])
            num_objects = len(regionprops)

        counts['In all loaded positions'] += num_objects

        if position.visited:
            counts['In all visited positions (current session)'] += num_objects

    return counts
