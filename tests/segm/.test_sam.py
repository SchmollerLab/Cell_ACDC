"""Tests for Segment Anything (SAM) segmentation model."""

from pathlib import Path

import pytest

from cellacdc import myutils
from tests.utils import (
    ensure_sam,
    get_test_posdata,
    load_normalized_frames,
    validate_labels,
    save_segmentation_overlay,
    print_segmentation_results,
)

ensure_sam()


class TestSAMAutomaticSegmentation:
    """Test SAM model with automatic segmentation (no input points)."""

    @pytest.fixture(scope="class", autouse=True)
    def download_models(self):
        """Download SAM models if not present."""
        myutils.download_model("segment_anything")

    @pytest.fixture
    def posData(self):
        """Get posData object."""
        return get_test_posdata()

    @pytest.fixture
    def test_frames(self, posData):
        """Load timelapse frames (every 20th frame)."""
        return load_normalized_frames(posData, step=20)

    def test_automatic_segmentation_sampled_frames(self, test_frames, posData):
        """Test SAM automatic segmentation on sampled frames."""
        frames, frame_indices = test_frames

        acdcSegment = myutils.import_segment_module("segment_anything")

        model = acdcSegment.Model(
            model_type="Small",
            input_points_path="",
            input_points_df="None",
            points_per_side=32,
            pred_iou_thresh=0.60,
            stability_score_thresh=0.70,
            crop_n_layers=0,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=1,
            gpu=True,
        )

        plots_dir = Path(__file__).parent.parent / "_plots" / "segm" / "sam"

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
                plots_dir / f"test_sam_segmentation_frame_{frame_i:04d}.png",
            )
