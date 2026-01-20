"""Tests for Segment Anything (SAM) segmentation model."""

import importlib
import sys
from pathlib import Path

import pytest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import patheffects
import skimage.measure

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


class TestSAMAutomaticSegmentation:
    """Test SAM model with automatic segmentation (no input points)."""

    @pytest.fixture(scope="class", autouse=True)
    def download_models(self):
        """Download SAM models if not present."""
        myutils.download_model("segment_anything")

    @pytest.fixture
    def test_data(self):
        """Load test data."""
        return data.MIA_KC_htb1_mCitrine()

    @pytest.fixture
    def posData(self, test_data):
        """Get posData object."""
        return test_data.posData()

    @pytest.fixture
    def test_frames(self, posData):
        """Load timelapse frames (every 20th frame)."""
        posData.loadImgData()
        posData.loadOtherFiles(load_segm_data=False, load_metadata=True)
        posData.buildPaths()

        image_data = posData.img_data

        normalized_frames = []
        frame_indices = []
        for i in range(0, len(image_data), 20):
            frame = image_data[i]
            frame_min = frame.min()
            frame_max = frame.max()
            if frame_max > frame_min:
                frame = ((frame - frame_min) / (frame_max - frame_min) * 65535).astype(
                    np.uint16
                )
            normalized_frames.append(frame)
            frame_indices.append(i)

        return np.array(normalized_frames), frame_indices

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
        plots_dir.mkdir(parents=True, exist_ok=True)

        for frame, frame_i in zip(frames, frame_indices):
            labels = model.segment(
                frame,
                frame_i=frame_i,
                automatic_removal_of_background=True,
                posData=posData,
            )

            assert labels is not None, "Segmentation returned None"
            assert isinstance(labels, np.ndarray), (
                f"Expected numpy array, got {type(labels)}"
            )
            assert labels.shape == frame.shape, (
                f"Shape mismatch: {labels.shape} != {frame.shape}"
            )
            assert np.issubdtype(labels.dtype, np.integer), (
                f"Expected integer dtype, got {labels.dtype}"
            )
            assert labels.min() >= 0, (
                f"Labels should be non-negative, got min={labels.min()}"
            )

            num_objects = labels.max()

            print(f"[PASS] Segmentation completed successfully for frame {frame_i}")
            print(f"  Image shape: {frame.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels dtype: {labels.dtype}")
            print(f"  Objects detected: {num_objects}")
            print(f"  Background pixels: {(labels == 0).sum()}")
            print(f"  Foreground pixels: {(labels > 0).sum()}")

            fig, ax = plt.subplots(figsize=(6, 6))

            n_colors = max(20, num_objects + 1)
            base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
            colors = np.zeros((n_colors, 4))
            colors[0] = [0, 0, 0, 0]
            for i in range(1, n_colors):
                colors[i] = base_colors[(i - 1) % 20]
            overlay_cmap = ListedColormap(colors)

            ax.imshow(frame, cmap="gray")
            ax.imshow(labels, cmap=overlay_cmap, vmin=0, vmax=n_colors - 1, alpha=0.5)

            for region in skimage.measure.regionprops(labels):
                cy, cx = region.centroid
                coords = region.coords
                distances = (coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2
                closest_idx = np.argmin(distances)
                y, x = coords[closest_idx]
                ax.text(
                    x, y, str(region.label),
                    color="white", fontsize=8, fontweight="bold",
                    ha="center", va="center",
                    path_effects=[
                        patheffects.withStroke(linewidth=2, foreground="black")
                    ],
                )

            ax.set_title(f"Frame {frame_i} ({num_objects} objects)")
            ax.axis("off")

            plt.tight_layout()
            output_path = plots_dir / f"test_sam_segmentation_frame_{frame_i:04d}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"  Plot saved to: {output_path}")
