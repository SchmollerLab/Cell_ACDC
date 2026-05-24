"""UI-neutral image display transforms."""

from __future__ import annotations

import numpy as np
import skimage
import skimage.exposure


def distant_gray(
    desired_gray: float,
    background_gray: float,
    *,
    threshold: float = 0.3,
) -> float:
    """Return a gray value with enough contrast from a background value."""
    if abs(desired_gray - background_gray) < threshold:
        return 1 - desired_gray
    return desired_gray


def rgb_to_gray(red: float, green: float, blue: float) -> float:
    """Convert RGB values in [0, 255] to gamma-corrected grayscale."""
    c_linear = (0.2126 * red + 0.7152 * green + 0.0722 * blue) / 255
    if c_linear <= 0.0031309:
        return 12.92 * c_linear
    return 1.055 * c_linear ** (1 / 2.4) - 0.055


def normalize_display_image(image: np.ndarray, how: str, *, image_to_float):
    """Apply Cell-ACDC display normalization semantics to an image."""
    if how == 'Do not normalize. Display raw image':
        return image
    if how == 'Convert to floating point format with values [0, 1]':
        return image_to_float(image)
    if how == 'Rescale to [0, 1]':
        return skimage.exposure.rescale_intensity(skimage.img_as_float(image))
    if how == 'Normalize by max value':
        return image / np.max(image)
    return image
