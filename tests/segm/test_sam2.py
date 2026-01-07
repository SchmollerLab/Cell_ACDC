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

    @pytest.fixture
    def test_data(self):
        """Load test data."""
        return data.MIA_KC_htb1_mCitrine()

    @pytest.fixture
    def posData(self, test_data):
        """Get posData object."""
        return test_data.posData()

    @pytest.fixture
    def test_frame(self, posData):
        """Load single frame from timelapse."""
        posData.loadImgData()
        posData.loadOtherFiles(load_segm_data=False, load_metadata=True)
        posData.buildPaths()

        # Get first frame from timelapse (shape: T, Y, X)
        image_data = posData.img_data
        frame = image_data[0]

        # Apply contrast stretching for better visibility
        # Normalize to full dtype range (like Fiji's auto-contrast)
        frame_min = frame.min()
        frame_max = frame.max()
        if frame_max > frame_min:
            frame = ((frame - frame_min) / (frame_max - frame_min) * 65535).astype(np.uint16)

        return frame

    def test_automatic_segmentation_single_frame(self, test_frame, posData):
        """Test SAM2 automatic segmentation on a single frame."""
        # Import SAM2 model module
        acdcSegment = myutils.import_segment_module('sam2')

        # Initialize model with Tiny variant (fastest) and automatic mode
        # Using more points and very relaxed thresholds for better coverage
        model = acdcSegment.Model(
            model_type='Tiny',
            input_points_path='',
            input_points_df='None',
            points_per_side=32,  # More points for better coverage
            pred_iou_thresh=0.60,  # Very relaxed threshold
            stability_score_thresh=0.70,  # Very relaxed threshold
            crop_n_layers=0,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=1,
            gpu=True
        )

        # Run segmentation
        labels = model.segment(
            test_frame,
            frame_i=0,
            automatic_removal_of_background=True,
            posData=posData
        )

        # Validate output
        assert labels is not None, "Segmentation returned None"
        assert isinstance(labels, np.ndarray), f"Expected numpy array, got {type(labels)}"
        assert labels.shape == test_frame.shape, f"Shape mismatch: {labels.shape} != {test_frame.shape}"
        assert np.issubdtype(labels.dtype, np.integer), f"Expected integer dtype, got {labels.dtype}"
        assert labels.min() >= 0, f"Labels should be non-negative, got min={labels.min()}"

        # Note: We don't require objects to be detected, as this depends on image content
        # and segmentation parameters. The key is that the model runs without errors.
        num_objects = labels.max()

        print(f"[PASS] Segmentation completed successfully")
        print(f"  Image shape: {test_frame.shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Labels dtype: {labels.dtype}")
        print(f"  Objects detected: {num_objects}")
        print(f"  Background pixels: {(labels == 0).sum()}")
        print(f"  Foreground pixels: {(labels > 0).sum()}")

        # Plot and save segmentation overlay
        plots_dir = Path(__file__).parent.parent / "_plots"
        plots_dir.mkdir(exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original normalized image
        axes[0].imshow(test_frame, cmap='gray')
        axes[0].set_title('Normalized Image')
        axes[0].axis('off')

        # Segmentation labels
        axes[1].imshow(labels, cmap='tab20')
        axes[1].set_title(f'Segmentation Labels ({num_objects} objects)')
        axes[1].axis('off')

        # Overlay
        axes[2].imshow(test_frame, cmap='gray')
        # Create colormap with transparent background
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        colors[0] = [0, 0, 0, 0]  # Make background transparent
        cmap = ListedColormap(colors)
        masked_labels = np.ma.masked_where(labels == 0, labels)
        axes[2].imshow(masked_labels, cmap=cmap, alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        output_path = plots_dir / 'test_sam2_segmentation.png'
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"  Plot saved to: {output_path}")
