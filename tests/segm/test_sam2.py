"""Tests for SAM2 segmentation model."""

import pytest
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Skip entire module if sam2 is not installed
pytest.importorskip("sam2")

from cellacdc import data, myutils


class TestSAM2AutomaticSegmentation:
    """Test SAM2 model with automatic segmentation (no input points)."""

    @pytest.fixture(scope="class", autouse=True)
    def download_models(self):
        """Download SAM2 models if not present."""
        myutils.download_model("sam2")

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
        """Load timelapse frames (every 5th frame)."""
        posData.loadImgData()
        posData.loadOtherFiles(load_segm_data=False, load_metadata=True)
        posData.buildPaths()

        # Get frames from timelapse (shape: T, Y, X)
        image_data = posData.img_data

        # Apply contrast stretching for better visibility
        # Normalize to full dtype range (like Fiji's auto-contrast)
        # Sample every 5th frame
        normalized_frames = []
        frame_indices = []
        for i in range(0, len(image_data), 5):
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
        """Test SAM2 automatic segmentation on sampled frames (every 5th)."""
        frames, frame_indices = test_frames

        # Import SAM2 model module
        acdcSegment = myutils.import_segment_module("sam2")

        # Initialize model with Tiny variant (fastest) and automatic mode
        # Using more points and very relaxed thresholds for better coverage
        model = acdcSegment.Model(
            model_type="Tiny",
            input_points_path="",
            input_points_df="None",
            points_per_side=32,  # More points for better coverage
            pred_iou_thresh=0.60,  # Very relaxed threshold
            stability_score_thresh=0.70,  # Very relaxed threshold
            crop_n_layers=0,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=1,
            gpu=True,
        )

        # Plot and save segmentation overlays
        plots_dir = Path(__file__).parent.parent / "_plots"
        plots_dir.mkdir(exist_ok=True)

        for idx, (frame, frame_i) in enumerate(zip(frames, frame_indices)):
            # Run segmentation
            labels = model.segment(
                frame,
                frame_i=frame_i,
                automatic_removal_of_background=True,
                posData=posData,
            )

            # Validate output
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

            # Note: We don't require objects to be detected, as this depends on image content
            # and segmentation parameters. The key is that the model runs without errors.
            num_objects = labels.max()

            print(f"[PASS] Segmentation completed successfully for frame {frame_i}")
            print(f"  Image shape: {frame.shape}")
            print(f"  Labels shape: {labels.shape}")
            print(f"  Labels dtype: {labels.dtype}")
            print(f"  Objects detected: {num_objects}")
            print(f"  Background pixels: {(labels == 0).sum()}")
            print(f"  Foreground pixels: {(labels > 0).sum()}")

            fig, ax = plt.subplots(figsize=(6, 6))

            # Create colormap with transparent background for overlay
            n_colors = max(20, num_objects + 1)
            base_colors = plt.cm.tab20(np.linspace(0, 1, 20))
            colors = np.zeros((n_colors, 4))
            colors[0] = [0, 0, 0, 0]  # Transparent background
            for i in range(1, n_colors):
                colors[i] = base_colors[(i - 1) % 20]
            overlay_cmap = ListedColormap(colors)

            # Overlay: grayscale image with colored labels
            ax.imshow(frame, cmap="gray")
            ax.imshow(labels, cmap=overlay_cmap, vmin=0, vmax=n_colors - 1, alpha=0.5)

            # Add text annotations at centroid of each mask
            for region in skimage.measure.regionprops(labels):
                y, x = region.centroid
                ax.text(
                    x, y, str(region.label),
                    color='white', fontsize=8, fontweight='bold',
                    ha='center', va='center',
                    path_effects=[
                        plt.matplotlib.patheffects.withStroke(linewidth=2, foreground='black')
                    ]
                )

            ax.set_title(f"Frame {frame_i} ({num_objects} objects)")
            ax.axis("off")

            plt.tight_layout()
            output_path = plots_dir / f"test_sam2_segmentation_frame_{frame_i:04d}.png"
            plt.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close()

            print(f"  Plot saved to: {output_path}")
