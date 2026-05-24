"""Tests for in-memory data loading and array-based viewer API."""

from __future__ import annotations

import importlib
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from cellacdc.data_source import (
    ArrayDataSource,
    ExperimentData,
    normalize_volume,
    pos_data_from_arrays,
    pos_data_from_kwargs,
)


def test_normalize_volume_shapes():
    image = np.zeros((4, 32, 32), dtype=np.uint8)
    arr, size_t, size_z = normalize_volume(image, axes="tyx")
    assert arr.shape == (4, 32, 32)
    assert size_t == 4
    assert size_z == 1

    stack = np.zeros((5, 8, 16, 16), dtype=np.uint8)
    arr, size_t, size_z = normalize_volume(stack, axes="tzyx")
    assert arr.shape == (5, 8, 16, 16)
    assert size_t == 5
    assert size_z == 8


def test_experiment_data_from_arrays(tmp_path):
    class DummyPosData:
        def __init__(self, img_path, channel_name, **kwargs):
            self.imgPath = img_path
            self.user_ch_name = channel_name
            self.images_path = str(tmp_path / "Images")
            self.exp_path = str(tmp_path)

        def buildPaths(self):
            self.metadata_csv_path = str(tmp_path / "metadata.csv")
            self.segm_npz_path = str(tmp_path / "segm.npz")

        def loadOtherFiles(self, **kwargs):
            pass

        def setBlankSegmData(self, size_t, size_z, size_y, size_x):
            self.segm_data = np.zeros((size_y, size_x), dtype=np.uint32)

        def extractMetadata(self):
            pass

    image = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    data = ExperimentData.from_arrays(
        image,
        name="test",
        channel_name="cells",
        axes="yx",
        workspace=tmp_path,
        _load_data_cls=DummyPosData,
    )

    assert data.is_materialized
    assert data.source == "memory"
    pos = data.positions[0]
    assert pos.SizeT == 1
    assert pos.img_data.shape == (1, 32, 32)


def test_experiment_data_from_path(tmp_path):
    exp_path = tmp_path / "my_experiment"
    exp_path.mkdir()
    data = ExperimentData.from_path(exp_path)

    assert data.source == "path"
    assert data.path == str(exp_path)
    assert not data.is_materialized


def test_pos_data_from_arrays_without_labels(tmp_path):
    class DummyPosData:
        def __init__(self, img_path, channel_name, **kwargs):
            self.imgPath = img_path
            self.user_ch_name = channel_name
            self.images_path = str(tmp_path / "Images")
            self.exp_path = str(tmp_path)

        def buildPaths(self):
            self.metadata_csv_path = str(tmp_path / "metadata.csv")
            self.segm_npz_path = str(tmp_path / "segm.npz")

        def loadOtherFiles(self, **kwargs):
            pass

        def setBlankSegmData(self, size_t, size_z, size_y, size_x):
            self.segm_data = np.zeros((size_y, size_x), dtype=np.uint32)

        def extractMetadata(self):
            pass

    image = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    pos = pos_data_from_kwargs(
        image,
        name="test",
        channel_name="cells",
        axes="yx",
        workspace=tmp_path,
        _load_data_cls=DummyPosData,
    )

    assert pos.SizeT == 1
    assert pos.segmFound is False


def test_viewer_accepts_experiment_data():
    viewer_mod = importlib.import_module("cellacdc.viewer")
    viewer_mod = importlib.reload(viewer_mod)
    mock_win = MagicMock()
    data = MagicMock()
    data.is_materialized = True

    with (
        patch("cellacdc._event_loop.get_qapp", return_value=MagicMock()),
        patch.object(viewer_mod, "_read_version", return_value="test"),
        patch.object(viewer_mod, "_create_gui_window", return_value=mock_win),
        patch.object(viewer_mod, "_check_gui_installed"),
    ):
        viewer = viewer_mod.Viewer(data, show=False)

    data.load_into.assert_called_once_with(mock_win)
    assert viewer.data is data


def test_imshow_returns_viewer_and_experiment_data(tmp_path):
    viewer_mod = importlib.import_module("cellacdc.viewer")
    viewer_mod = importlib.reload(viewer_mod)
    data = ExperimentData.from_path(tmp_path)
    mock_viewer = MagicMock()
    mock_viewer.data = data

    with patch.object(viewer_mod, "Viewer", return_value=mock_viewer) as mock_viewer_cls:
        viewer, returned = viewer_mod.imshow(data)

    mock_viewer_cls.assert_called_once_with(
        data,
        show=True,
        mode="Segmentation and Tracking",
    )
    assert viewer is mock_viewer
    assert returned is data


def test_lazy_exports_include_experiment_data():
    import cellacdc

    assert cellacdc.ExperimentData.__name__ == "ExperimentData"
    assert cellacdc.imshow.__name__ == "imshow"
