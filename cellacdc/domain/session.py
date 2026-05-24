"""In-memory session objects for scripting and GUI binding."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .events import EventEmitter
from .state import PositionState


@dataclass
class PositionSession:
    """Headless per-position data container (successor to ``loadData``)."""

    _state: PositionState
    events: EventEmitter = field(default_factory=EventEmitter)
    _legacy_pos_data: Any = field(default=None, repr=False)

    @classmethod
    def from_arrays(
        cls,
        intensity: np.ndarray,
        labels: np.ndarray | None = None,
        acdc_df: pd.DataFrame | None = None,
        *,
        pixel_size_um: float | None = None,
        channel_name: str = '',
        basename: str = '',
        **metadata: Any,
    ) -> PositionSession:
        md = dict(metadata)
        if pixel_size_um is not None:
            md['pixel_size_um'] = pixel_size_um
        if channel_name:
            md['channel_name'] = channel_name
        if basename:
            md['basename'] = basename
        state = PositionState(
            intensity=np.asarray(intensity),
            labels=None if labels is None else np.asarray(labels),
            acdc_df=acdc_df,
            metadata=md,
        )
        return cls(_state=state)

    @classmethod
    def from_path(
        cls,
        img_path: str,
        user_ch_name: str = '',
        **metadata: Any,
    ) -> PositionSession:
        from cellacdc.io.adapters.disk_to_session import load_position_from_disk

        return load_position_from_disk(img_path, user_ch_name, **metadata)

    @classmethod
    def from_loadData(cls, pos_data: Any) -> PositionSession:
        from cellacdc.io.adapters.legacy_load_data import session_from_load_data

        return session_from_load_data(pos_data)

    @property
    def intensity(self) -> np.ndarray:
        return self._state.intensity

    @property
    def labels(self) -> np.ndarray | None:
        return self._state.labels

    @property
    def acdc_df(self) -> pd.DataFrame | None:
        return self._state.acdc_df

    @property
    def metadata(self) -> dict[str, Any]:
        return self._state.metadata

    @property
    def frame_i(self) -> int:
        return self._state.frame_i

    @frame_i.setter
    def frame_i(self, value: int) -> None:
        self._state.frame_i = int(value)
        self.events.emit('frame_changed', self._state.frame_i)

    @property
    def num_frames(self) -> int:
        return self._state.num_frames

    @property
    def legacy_pos_data(self) -> Any:
        return self._legacy_pos_data

    def set_labels(self, labels: np.ndarray) -> None:
        self._state.labels = np.asarray(labels)
        self.events.emit('labels_changed', self._state.labels)

    def set_acdc_df(self, acdc_df: pd.DataFrame) -> None:
        self._state.acdc_df = acdc_df
        self.events.emit('acdc_df_changed', acdc_df)

    def frame_intensity(self, frame_i: int | None = None) -> np.ndarray:
        return self._state.frame_intensity(frame_i)

    def frame_labels(self, frame_i: int | None = None) -> np.ndarray | None:
        return self._state.frame_labels(frame_i)

    def save(self, path: str | None = None) -> None:
        from cellacdc.io.adapters.session_to_disk import save_position_session

        save_position_session(self, path)


@dataclass
class ExperimentSession:
    """Collection of positions in an experiment folder."""

    positions: list[PositionSession] = field(default_factory=list)
    exp_path: str = ''
    events: EventEmitter = field(default_factory=EventEmitter)

    @classmethod
    def from_experiment_path(
        cls,
        exp_path: str,
        user_ch_name: str = '',
    ) -> ExperimentSession:
        from cellacdc.io.adapters.disk_to_session import load_experiment_from_disk

        return load_experiment_from_disk(exp_path, user_ch_name)

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, index: int) -> PositionSession:
        return self.positions[index]
