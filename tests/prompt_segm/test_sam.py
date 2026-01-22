"""Tests for promptable Segment Anything (SAM)."""

from pathlib import Path

import pytest

from cellacdc import myutils
from tests.utils import (
    ensure_sam,
    get_test_posdata,
    load_single_normalized_frame,
    validate_labels,
    save_labels_image,
)

ensure_sam()


class TestPromptableSAM:
    @pytest.fixture(scope="class", autouse=True)
    def download_models(self):
        """Download SAM models if not present."""
        myutils.download_model("segment_anything")

    @pytest.fixture
    def frame(self):
        """Load a single frame from the test dataset."""
        return load_single_normalized_frame(get_test_posdata(), frame_index=-1)

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

        validate_labels(labels, frame.shape[:2])

        plots_dir = Path(__file__).parent.parent / "_plots" / "prompt_segm" / "sam"
        save_labels_image(
            labels,
            plots_dir / f"test_promptable_sam_frame_{frame_index:04d}.png",
        )
