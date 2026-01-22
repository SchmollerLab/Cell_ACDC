"""Tests for promptable SAM2."""

from pathlib import Path

import pytest

from cellacdc import myutils
from tests.utils import (
    ensure_sam2,
    get_test_dataset,
    get_ground_truth_centroids,
    validate_labels,
    save_segmentation_overlay,
)

ensure_sam2()


class TestPromptableSAM2:
    @pytest.fixture(scope="class", autouse=True)
    def download_models(self):
        """Download SAM2 models if not present."""
        myutils.download_model("sam2")

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
        """Test SAM2 promptable segmentation using ground truth centroids."""
        posData, segm_data = test_data

        # Use last frame
        frame_index = len(posData.img_data) - 1
        frame = posData.img_data[frame_index]
        gt_mask = segm_data[frame_index]

        # Get centroids from ground truth
        centroids = get_ground_truth_centroids(gt_mask)
        assert len(centroids) > 0, "No objects found in ground truth"

        acdcPromptSegment = myutils.import_promptable_segment_module("sam2")
        model = acdcPromptSegment.Model(model_type="Tiny", gpu=True)

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

        plots_dir = Path(__file__).parent.parent / "_plots" / "prompt_segm" / "sam2"
        save_segmentation_overlay(
            labels, frame, frame_index,
            plots_dir / f"test_promptable_sam2_frame_{frame_index:04d}.png",
        )
