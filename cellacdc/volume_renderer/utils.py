import os
from concurrent.futures import ThreadPoolExecutor
from functools import reduce
from typing import Any, Callable, Sequence

import numpy as np

# Max. number of threads used to process the frames of large arrays
_MAX_NUM_THREADS = 8

def voxel_display_scale(dz: float, dy: float, dx: float) -> tuple[float, float, float]:
    """Return vispy ``STTransform`` scale for ``(Z, Y, X)`` data (Cell-ACDC style)."""
    dx_eff = dx if dx > 0 else 1.0
    dy_eff = dy if dy > 0 else 1.0
    dz_eff = dz if dz > 0 else 1.0
    return (1.0, dy_eff / dx_eff, dz_eff / dx_eff)

def map_frames(
        func: Callable[[int], Any],
        num_frames: int,
        max_workers: int | None=None
    ) -> list:
    """Return `[func(frame_i) for frame_i in range(num_frames)]` computed in
    a pool of threads.

    Useful to process large arrays frame by frame with NumPy, which
    releases the GIL in most array operations (e.g., copies, ufuncs and
    reductions on numeric arrays). The results are in frame order.
    """
    if max_workers is None:
        max_workers = min(_MAX_NUM_THREADS, os.cpu_count() or 1)

    max_workers = min(max_workers, num_frames)
    if max_workers < 2:
        return [func(frame_i) for frame_i in range(num_frames)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(func, range(num_frames)))

def stack_frames(frames: Sequence[np.ndarray]) -> np.ndarray:
    """Return `np.asarray(frames)`, with the frames copied into the new
    array in parallel threads.
    """
    shapes = {np.shape(frame) for frame in frames}
    if len(frames) < 2 or len(shapes) != 1:
        # Same result (or error) as before
        return np.asarray(frames)

    dtype = reduce(np.promote_types, {np.asarray(f).dtype for f in frames})
    out = np.empty((len(frames), *shapes.pop()), dtype=dtype)

    def _copy_frame(frame_i):
        out[frame_i] = frames[frame_i]

    map_frames(_copy_frame, len(frames))
    return out

def frames_min_max(arr: np.ndarray) -> tuple[Any, Any]:
    """Return `(arr.min(), arr.max())` computed frame by frame (first axis)
    in parallel threads.

    NaNs propagate as with `arr.min()` and `arr.max()`.
    """
    def _frame_min_max(frame_i):
        frame = arr[frame_i]
        return frame.min(), frame.max()

    mins, maxs = zip(*map_frames(_frame_min_max, len(arr)))
    return np.min(mins), np.max(maxs)
