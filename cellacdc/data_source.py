"""Unified experiment data for decoupling the GUI from filesystem loading."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

VolumeAxes = Literal["yx", "zyx", "tyx", "tzyx"]
PathKind = Literal["file", "experiment", "images", "folder"]
DataSourceKind = Literal["memory", "path"]


@dataclass
class ArrayDataSource:
    """Specification for building in-memory position data."""

    image: np.ndarray
    labels: np.ndarray | None = None
    name: str = "data"
    channel_name: str = "cells"
    axes: VolumeAxes = "tyx"
    workspace: str | os.PathLike | None = None
    time_increment: float = 1.0
    physical_size_xy: tuple[float, float] = (1.0, 1.0)
    physical_size_z: float = 1.0
    is_segm_3d: bool = False
    metadata: dict[str, str | float | int] = field(default_factory=dict)


class ExperimentData:
    """Unified dataset handle for the Cell-ACDC script API.

    Use :meth:`from_arrays` or :meth:`from_path` to create instances.
    """

    name: str
    source: DataSourceKind
    path: str | None
    path_kind: PathKind | None
    _positions: list | None

    def __init__(self):
        pass

    @classmethod
    def from_arrays(
        cls,
        image: np.ndarray,
        labels: np.ndarray | None = None,
        **kwargs,
    ) -> ExperimentData:
        """Create dataset data from in-memory arrays."""
        self = cls()
        load_data_cls = kwargs.pop("_load_data_cls", None)
        name = kwargs.get("name", "data")
        pos = pos_data_from_kwargs(
            image,
            labels,
            _load_data_cls=load_data_cls,
            **kwargs,
        )
        self.source = "memory"
        self.name = name
        self.path = None
        self.path_kind = None
        self._positions = [pos]
        return self

    @classmethod
    def from_path(cls, path: str | os.PathLike, **kwargs) -> ExperimentData:
        """Create a dataset handle from a filesystem path."""
        path = os.fspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(path)

        self = cls()
        name = kwargs.get("name", "data")
        self.source = "path"
        self.path = path
        self.path_kind = _detect_path_kind(path)
        self.name = (
            os.path.basename(path.rstrip(os.sep)) if name == "data" else name
        )
        self._positions = None
        return self

    @property
    def is_materialized(self) -> bool:
        return self.source == "memory" and self._positions is not None

    @property
    def positions(self) -> list:
        if not self.is_materialized:
            raise RuntimeError(
                "Path-based ExperimentData is loaded by the viewer on demand. "
                "Use Viewer(data) or data.load_into(window)."
            )
        return self._positions

    def load_into(self, window) -> None:
        if self.source == "memory":
            window.loadFromExperimentData(self)
            return

        if self.path_kind == "file":
            window.openFile(file_path=self.path)
        elif self.path_kind == "images":
            window.openFolder(exp_path=self.path)
        elif self.path_kind == "experiment":
            window.openFolder(exp_path=self.path)
        else:
            if os.path.isdir(self.path):
                window.openFolder(exp_path=self.path)
            else:
                window.openFile(file_path=self.path)


def _detect_path_kind(path: str) -> PathKind:
    if os.path.isfile(path):
        return "file"

    basename = os.path.basename(path.rstrip(os.sep))
    if basename == "Images":
        return "images"

    try:
        entries = os.listdir(path)
    except OSError:
        return "folder"

    if any(entry.startswith("Position") and os.path.isdir(os.path.join(path, entry)) for entry in entries):
        return "experiment"

    return "folder"


def normalize_volume(
    array: np.ndarray,
    *,
    axes: VolumeAxes = "tyx",
) -> tuple[np.ndarray, int, int]:
    """Return (array, SizeT, SizeZ) in Cell-ACDC's pre-finalize layout."""
    arr = np.asarray(array)
    if arr.ndim == 2:
        if axes != "yx":
            raise ValueError(
                f"A 2D array requires axes='yx', got axes={axes!r}."
            )
        return arr, 1, 1

    if arr.ndim == 3:
        if axes == "zyx":
            return arr, 1, arr.shape[0]
        if axes == "tyx":
            return arr, arr.shape[0], 1
        raise ValueError(
            f"A 3D array requires axes='tyx' or 'zyx', got axes={axes!r}."
        )

    if arr.ndim == 4:
        if axes != "tzyx":
            raise ValueError(
                f"A 4D array requires axes='tzyx', got axes={axes!r}."
            )
        return arr, arr.shape[0], arr.shape[1]

    raise ValueError(
        f"Expected a 2D, 3D, or 4D array, got shape {arr.shape}."
    )


def _finalize_pos_data_arrays(pos_data) -> None:
    """Match the array layout produced by ``loadDataWorker``."""
    if pos_data.SizeT == 1:
        pos_data.img_data = pos_data.img_data[np.newaxis]
        if pos_data.segm_data is not None:
            pos_data.segm_data = pos_data.segm_data[np.newaxis]

    pos_data.img_data_shape = pos_data.img_data.shape
    pos_data.dset = pos_data.img_data
    if pos_data.segm_data is not None:
        pos_data.segmSizeT = len(pos_data.segm_data)


def _write_metadata_csv(
    metadata_csv_path: os.PathLike,
    *,
    basename: str,
    size_t: int,
    size_z: int,
    size_y: int,
    size_x: int,
    channel_name: str,
    time_increment: float,
    physical_size_xy: tuple[float, float],
    physical_size_z: float,
    is_segm_3d: bool,
    extra: dict[str, str | float | int],
) -> None:
    rows = {
        "basename": basename,
        "SizeT": size_t,
        "SizeZ": size_z,
        "SizeY": size_y,
        "SizeX": size_x,
        "TimeIncrement": time_increment,
        "PhysicalSizeX": physical_size_xy[0],
        "PhysicalSizeY": physical_size_xy[1],
        "PhysicalSizeZ": physical_size_z,
        "segm_isSegm3D": str(is_segm_3d),
        f"{channel_name}_name": channel_name,
    }
    rows.update(extra)
    df = pd.DataFrame(
        {"Description": list(rows.keys()), "values": [str(v) for v in rows.values()]}
    )
    df.to_csv(metadata_csv_path, index=False)


def pos_data_from_arrays(source: ArrayDataSource, *, _load_data_cls=None):
    """Build a ``loadData`` instance backed by in-memory arrays."""
    if _load_data_cls is None:
        from cellacdc import load

        _load_data_cls = load.loadData

    image, size_t, size_z = normalize_volume(source.image, axes=source.axes)
    size_y, size_x = image.shape[-2:]

    labels = source.labels
    if labels is not None:
        labels, labels_size_t, labels_size_z = normalize_volume(
            labels, axes=source.axes
        )
        if (labels_size_t, labels_size_z, *labels.shape[-2:]) != (
            size_t,
            size_z,
            size_y,
            size_x,
        ):
            raise ValueError(
                "Labels shape must match the image shape for the given axes."
            )
        labels = labels.astype(np.uint32, copy=False)

    if source.workspace is None:
        workspace = tempfile.mkdtemp(prefix="cellacdc_")
    else:
        workspace = os.fspath(source.workspace)
        os.makedirs(workspace, exist_ok=True)

    exp_path = os.path.join(workspace, source.name)
    pos_path = os.path.join(exp_path, "Position_001")
    images_path = os.path.join(pos_path, "Images")
    os.makedirs(images_path, exist_ok=True)

    basename = f"{source.name}_"
    channel_name = source.channel_name
    img_filename = f"{basename}{channel_name}.npz"
    img_path = os.path.join(images_path, img_filename)

    pos = _load_data_cls(img_path, channel_name, log_func=print)
    pos.basename = basename
    pos.chNames = [channel_name]
    pos.filename = f"{basename}{channel_name}"
    pos.filename_ext = img_filename
    pos.ext = ".npz"
    pos.images_folder_files = [img_filename]
    pos.img_data = image
    pos.SizeT = size_t
    pos.SizeZ = size_z
    pos.SizeY = size_y
    pos.SizeX = size_x
    pos.loadSizeS = 1
    pos.loadSizeT = size_t
    pos.loadSizeZ = size_z
    pos.TimeIncrement = source.time_increment
    pos.PhysicalSizeX = source.physical_size_xy[0]
    pos.PhysicalSizeY = source.physical_size_xy[1]
    pos.PhysicalSizeZ = source.physical_size_z
    pos.isSegm3D = source.is_segm_3d
    pos.is_in_memory = True

    pos.buildPaths()
    metadata_csv_path = pos.metadata_csv_path
    _write_metadata_csv(
        metadata_csv_path,
        basename=basename,
        size_t=size_t,
        size_z=size_z,
        size_y=size_y,
        size_x=size_x,
        channel_name=channel_name,
        time_increment=source.time_increment,
        physical_size_xy=source.physical_size_xy,
        physical_size_z=source.physical_size_z,
        is_segm_3d=source.is_segm_3d,
        extra=source.metadata,
    )
    pos.metadataFound = True
    pos.metadata_df = pd.read_csv(metadata_csv_path).set_index("Description")
    pos.extractMetadata()

    if labels is not None:
        pos.segmFound = True
        pos.segm_data = labels
        pos.labelBoolSegm = False
    else:
        pos.segmFound = False
        pos.labelBoolSegm = False
        pos.loadOtherFiles(
            load_segm_data=False,
            create_new_segm=True,
            load_acdc_df=False,
            load_metadata=False,
        )
        pos.setBlankSegmData(pos.SizeT, pos.SizeZ, size_y, size_x)

    pos.acdc_df_found = False
    pos.acdc_df = None
    pos.segmInfo_df = None
    pos.allData_li = [None] * pos.SizeT
    pos.frame_i = 0

    _finalize_pos_data_arrays(pos)
    return pos


def pos_data_from_kwargs(
    image: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    _load_data_cls=None,
    **kwargs,
):
    source = ArrayDataSource(image=image, labels=labels, **kwargs)
    return pos_data_from_arrays(source, _load_data_cls=_load_data_cls)
