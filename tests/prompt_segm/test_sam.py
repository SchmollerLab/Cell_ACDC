"""Tests for promptable Segment Anything (SAM)."""

import importlib
import sys
from pathlib import Path

import pytest
import numpy as np
import matplotlib.pyplot as plt


def _ensure_segment_anything():
    try:
        importlib.import_module("segment_anything")
        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[3].parent
        candidate = repo_root / "segment-anything"
        if candidate.exists():
            sys.path.insert(0, str(candidate))

    pytest.importorskip("segment_anything")


_ensure_segment_anything()

from cellacdc import data, myutils


class TestPromptableSAM:
    @pytest.fixture(scope="class", autouse=True)
    def download_models(self):
        """Download SAM models if not present."""
        myutils.download_model("segment_anything")

    @pytest.fixture
    def frame(self):
        """Load a single frame from the test dataset."""
        dataset = data.MIA_KC_htb1_mCitrine()
        posData = dataset.posData()
        posData.loadImgData()
        posData.loadOtherFiles(load_segm_data=False, load_metadata=True)
        posData.buildPaths()

        frame_index = len(posData.img_data) - 1
        frame = posData.img_data[frame_index]
        frame_min = frame.min()
        frame_max = frame.max()
        if frame_max > frame_min:
            frame = ((frame - frame_min) / (frame_max - frame_min) * 65535).astype(
                np.uint16
            )
        return frame, frame_index

    def test_promptable_segmentation_single_frame(self, frame):
        frame, frame_index = frame
        acdcPromptSegment = myutils.import_promptable_segment_module(
            "segment_anything"
        )

        model = acdcPromptSegment.Model(model_type="Small", gpu=False)

        y = frame.shape[0] // 2
        x = frame.shape[1] // 2
        model.add_prompt(
            prompt=(0, y, x),
            prompt_id=1,
            image=frame,
            image_origin=(0, 0, 0),
            prompt_type="point",
        )

        labels = model.segment(frame)

        assert labels is not None, "Segmentation returned None"
        assert isinstance(labels, np.ndarray), (
            f"Expected numpy array, got {type(labels)}"
        )
        assert labels.shape == frame.shape[:2], (
            f"Shape mismatch: {labels.shape} != {frame.shape[:2]}"
        )
        assert np.issubdtype(labels.dtype, np.integer), (
            f"Expected integer dtype, got {labels.dtype}"
        )
        assert labels.min() >= 0, (
            f"Labels should be non-negative, got min={labels.min()}"
        )

        plots_dir = Path(__file__).parent.parent / "_plots" / "prompt_segm" / "sam"
        plots_dir.mkdir(parents=True, exist_ok=True)
        output_path = plots_dir / f"test_promptable_sam_frame_{frame_index:04d}.png"
        plt.imsave(output_path, labels, cmap="tab20")
