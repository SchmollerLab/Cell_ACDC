"""Shared utilities for segmentation tests."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import patheffects
import skimage.measure


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    """Normalize a frame to uint16 range."""
    frame_min = frame.min()
    frame_max = frame.max()
    if frame_max > frame_min:
        frame = ((frame - frame_min) / (frame_max - frame_min) * 65535).astype(
            np.uint16
        )
    return frame


def _load_posdata(posData):
    """Load image data into posData if not already loaded."""
    posData.loadImgData()
    posData.loadOtherFiles(load_segm_data=False, load_metadata=True)
    posData.buildPaths()
    return posData


def load_normalized_frames(posData, step: int = 20):
    """Load and normalize timelapse frames from posData.

    Parameters
    ----------
    posData : cellacdc.load.loadData
        Position data object with loaded image data.
    step : int, optional
        Sample every Nth frame. Default is 20.

    Returns
    -------
    tuple[np.ndarray, list[int]]
        Normalized frames array and list of frame indices.
    """
    _load_posdata(posData)
    image_data = posData.img_data

    normalized_frames = []
    frame_indices = []
    for i in range(0, len(image_data), step):
        normalized_frames.append(_normalize_frame(image_data[i]))
        frame_indices.append(i)

    return np.array(normalized_frames), frame_indices


def load_single_normalized_frame(posData, frame_index: int = -1):
    """Load and normalize a single frame from posData.

    Parameters
    ----------
    posData : cellacdc.load.loadData
        Position data object.
    frame_index : int, optional
        Frame index to load. Default is -1 (last frame).

    Returns
    -------
    tuple[np.ndarray, int]
        Normalized frame and actual frame index.
    """
    _load_posdata(posData)
    image_data = posData.img_data

    if frame_index < 0:
        frame_index = len(image_data) + frame_index

    return _normalize_frame(image_data[frame_index]), frame_index


def validate_labels(labels: np.ndarray, expected_shape: tuple):
    """Validate segmentation labels.

    Parameters
    ----------
    labels : np.ndarray
        Segmentation labels array.
    expected_shape : tuple
        Expected shape of labels array.

    Raises
    ------
    AssertionError
        If validation fails.
    """
    assert labels is not None, "Segmentation returned None"
    assert isinstance(labels, np.ndarray), (
        f"Expected numpy array, got {type(labels)}"
    )
    assert labels.shape == expected_shape, (
        f"Shape mismatch: {labels.shape} != {expected_shape}"
    )
    assert np.issubdtype(labels.dtype, np.integer), (
        f"Expected integer dtype, got {labels.dtype}"
    )
    assert labels.min() >= 0, (
        f"Labels should be non-negative, got min={labels.min()}"
    )


def print_segmentation_results(labels: np.ndarray, frame: np.ndarray, frame_i: int):
    """Print segmentation results summary.

    Parameters
    ----------
    labels : np.ndarray
        Segmentation labels array.
    frame : np.ndarray
        Original frame.
    frame_i : int
        Frame index.
    """
    num_objects = labels.max()
    print(f"[PASS] Segmentation completed successfully for frame {frame_i}")
    print(f"  Image shape: {frame.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Labels dtype: {labels.dtype}")
    print(f"  Objects detected: {num_objects}")
    print(f"  Background pixels: {(labels == 0).sum()}")
    print(f"  Foreground pixels: {(labels > 0).sum()}")


def save_segmentation_overlay(
    labels: np.ndarray,
    frame: np.ndarray,
    frame_i: int,
    output_path: Path,
):
    """Save segmentation overlay plot.

    Parameters
    ----------
    labels : np.ndarray
        Segmentation labels array.
    frame : np.ndarray
        Original frame (grayscale).
    frame_i : int
        Frame index for title.
    output_path : Path
        Path to save the plot.
    """
    num_objects = labels.max()

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

    # Add text annotations inside each mask
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Plot saved to: {output_path}")


def save_labels_image(labels: np.ndarray, output_path: Path):
    """Save labels as a simple colormap image.

    Parameters
    ----------
    labels : np.ndarray
        Segmentation labels array.
    output_path : Path
        Path to save the image.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(output_path, labels, cmap="tab20")
    print(f"  Plot saved to: {output_path}")


def ensure_sam():
    """Ensure segment_anything is importable, checking local repo as fallback."""
    import importlib
    import sys

    try:
        importlib.import_module("segment_anything")
        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[3].parent
        candidate = repo_root / "segment-anything"
        if candidate.exists():
            sys.path.insert(0, str(candidate))

    import pytest
    pytest.importorskip("segment_anything")


def ensure_sam2():
    """Ensure sam2 is importable, checking local repo as fallback."""
    import importlib
    import sys

    try:
        importlib.import_module("sam2")
        return
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[3].parent
        candidate = repo_root / "sam2"
        if candidate.exists():
            sys.path.insert(0, str(candidate))

    import pytest
    pytest.importorskip("sam2")


def ensure_cellsam():
    """Ensure cellSAM is importable."""
    import pytest
    pytest.importorskip("cellSAM")


def get_test_posdata():
    """Get posData for the standard test dataset."""
    from cellacdc import data
    return data.MIA_KC_htb1_mCitrine().posData()
