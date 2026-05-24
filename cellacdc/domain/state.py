"""Mutable in-memory state container for a position."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class PositionState:
    """Raw mutable store backing :class:`PositionSession`."""

    intensity: np.ndarray
    labels: np.ndarray | None = None
    acdc_df: pd.DataFrame | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    frame_i: int = 0
    fluo_data: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        if self.intensity.ndim == 2:
            return 1
        return int(self.intensity.shape[0])

    def frame_intensity(self, frame_i: int | None = None) -> np.ndarray:
        idx = self.frame_i if frame_i is None else frame_i
        if self.intensity.ndim == 2:
            return self.intensity
        return self.intensity[idx]

    def frame_labels(self, frame_i: int | None = None) -> np.ndarray | None:
        if self.labels is None:
            return None
        idx = self.frame_i if frame_i is None else frame_i
        if self.labels.ndim == 2:
            return self.labels
        return self.labels[idx]
