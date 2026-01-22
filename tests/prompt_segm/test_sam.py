"""Tests for promptable Segment Anything (SAM)."""

from pathlib import Path

import pytest

from cellacdc import myutils
from tests.utils import (
    ensure_sam,
    get_test_dataset,
    get_ground_truth_centroids,
    validate_labels,
    save_segmentation_overlay,
)

ensure_sam()


class TestPromptableSAM:
    @pytest.fixture(scope="class", autouse=True)
    def download_models(self):
        """Download SAM models if not present."""
        myutils.download_model("segment_anything")

    @pytest.fixture
    def test_data(self):
        """Load test dataset with ground truth."""
        dataset = get_test_dataset()
        segm_data = dataset.segm_data()
        posData = dataset.posData()
        posData.loadImgData()
        posData.loadOtherFiles(load_segm_data=False, load_metadata=True)
        posData.buildPaths()
        return posData, segm_data

    def test_promptable_segmentation_with_ground_truth_centroids(self, test_data):
        """Test SAM promptable segmentation using ground truth centroids."""
        posData, segm_data = test_data

        # Use last frame
        frame_index = len(posData.img_data) - 1
        frame = posData.img_data[frame_index]
        gt_mask = segm_data[frame_index]

        # Get centroids from ground truth
        centroids = get_ground_truth_centroids(gt_mask)
        assert len(centroids) > 0, "No objects found in ground truth"

        acdcPromptSegment = myutils.import_promptable_segment_module("segment_anything")
        model = acdcPromptSegment.Model(model_type="Small", gpu=True)

        # Add prompts for each ground truth centroid
        for label_id, y, x in centroids:
            model.add_prompt(
                prompt=(0, y, x),
                prompt_id=label_id,
                image=frame,
                image_origin=(0, 0, 0),
                prompt_type="point",
            )

        labels = model.segment(frame)

        validate_labels(labels, frame.shape[:2])

        num_gt_objects = len(centroids)
        num_detected = labels.max()
        print(f"[INFO] Ground truth objects: {num_gt_objects}")
        print(f"[INFO] Detected objects: {num_detected}")

        plots_dir = Path(__file__).parent.parent / "_plots" / "prompt_segm" / "sam"
        save_segmentation_overlay(
            labels, frame, frame_index,
            plots_dir / f"test_promptable_sam_frame_{frame_index:04d}.png",
        )
