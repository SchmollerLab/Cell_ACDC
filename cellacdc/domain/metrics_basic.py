"""Basic measurements from ``PositionSession`` (no legacy ``loadData`` required)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import skimage.measure


def compute_basic_metrics(session) -> pd.DataFrame:
    """Regionprops table per frame — fallback when legacy kernel is unavailable."""
    labels = session.labels
    if labels is None:
        raise ValueError('MeasureRunnable requires labels in session')

    rows: list[dict] = []
    num_frames = session.num_frames
    for frame_i in range(num_frames):
        lab = session.frame_labels(frame_i)
        if lab is None or not np.any(lab):
            continue
        for rp in skimage.measure.regionprops(lab.astype(np.int32)):
            if rp.label == 0:
                continue
            rows.append({
                'frame_i': frame_i,
                'Cell_ID': rp.label,
                'area': rp.area,
                'centroid_y': rp.centroid[0],
                'centroid_x': rp.centroid[1],
            })

    if not rows:
        return pd.DataFrame(columns=['frame_i', 'Cell_ID', 'area', 'centroid_y', 'centroid_x'])
    return pd.DataFrame(rows).set_index(['frame_i', 'Cell_ID'])
