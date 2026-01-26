"""Tests for CellSAM segmentation model."""

from pathlib import Path

import pytest

from cellacdc import myutils
from tests.utils import (
    ensure_cellsam,
    get_test_posdata,
    load_normalized_frames,
    validate_labels,
    save_segmentation_overlay,
    print_segmentation_results,
)

ensure_cellsam()


class TestCellSAMAutomaticSegmentation:
    """Test CellSAM model with automatic segmentation."""

    @pytest.fixture
    def posData(self):
        """Get posData object."""
        return get_test_posdata()

    @pytest.fixture
    def test_frames(self, posData):
        """Load timelapse frames (every 20th frame)."""
        return load_normalized_frames(posData, step=20)

    def test_automatic_segmentation_sampled_frames(self, test_frames, posData):
        """Test CellSAM automatic segmentation on sampled frames (every 20th)."""
        frames, frame_indices = test_frames

        acdcSegment = myutils.import_segment_module("cellsam")

        model = acdcSegment.Model(
            model_type="General",
            bbox_threshold=0.4,
            low_contrast_enhancement=False,
            use_wsi=False,
            postprocess=False,
            remove_boundaries=False,
            gpu=True,
        )

        plots_dir = Path(__file__).parent.parent / "_plots" / "segm" / "cellsam"

        for frame, frame_i in zip(frames, frame_indices):
            labels = model.segment(
                frame,
                frame_i=frame_i,
                automatic_removal_of_background=True,
                posData=posData,
            )

            validate_labels(labels, frame.shape)
            print_segmentation_results(labels, frame, frame_i)
            save_segmentation_overlay(
                labels, frame, frame_i,
                plots_dir / f"test_cellsam_segmentation_frame_{frame_i:04d}.png",
            )
