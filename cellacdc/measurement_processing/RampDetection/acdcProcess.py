import numbers

import numpy as np

name = 'Ramp detection'


def _fill_gaps(mask, gap_frames):
    """Set runs of at most `gap_frames` False values between True runs."""
    mask = mask.copy()
    true_idx = np.flatnonzero(mask)
    for start, end in zip(true_idx[:-1], true_idx[1:]):
        gap = end - start - 1
        if 0 < gap <= gap_frames:
            mask[start + 1:end] = True
    return mask


def _keep_long_runs(mask, min_length):
    """Keep only True runs with at least `min_length` values."""
    result = np.zeros_like(mask)
    padded = np.concatenate(([False], mask, [False])).astype(int)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)
    for start, end in zip(starts, ends):
        if end - start >= min_length:
            result[start:end] = True
    return result


_ZERO_SCALE = 1e-10


def _relative_scale(values):
    """Return the nonzero denominators of the relative changes of `values`.

    The change from observation i to i + 1 is divided by the absolute value
    of observation i + 1. When that value is (close to) zero, the larger
    absolute value of the two observations is used instead, and 1 (i.e. the
    absolute change) when both are (close to) zero. The denominator is
    therefore never zero and the sign of every change is preserved, also for
    changes towards or away from zero. NaN values propagate to the change.
    """
    previous = np.abs(values[:-1])
    following = np.abs(values[1:])
    with np.errstate(invalid='ignore'):
        scale = np.where(
            following > _ZERO_SCALE, following, np.fmax(previous, following)
        )
        scale = np.where(scale > _ZERO_SCALE, scale, 1.0)
    return scale


def _check_int(value, argname, minimum):
    if (isinstance(value, bool) or not isinstance(value, numbers.Integral)
            or value < minimum):
        raise ValueError(f'{argname} must be an integer >= {minimum}.')
    return int(value)


class Model:
    """Detect sustained rising and falling ramps of a cell's measurement.

    The output is +1 where the measurement rises, -1 where it falls and 0
    elsewhere (no sustained trend). A ramp is a run of consecutive
    observations whose (relative) change has the same sign, after short
    interruptions are bridged and short runs are discarded. Use it on a
    smoothed measurement (e.g. after "Smooth measurement"), since noise
    otherwise breaks ramps into short pieces.

    The value at observation i describes the change from observation i to
    i + 1; the last observation of each cell is 0. Windows count
    observations, not elapsed time or absent frames. Ported from the FUCCI
    cell cycle phase classifier (``count_ramps_with_buffer``).
    """

    is_timelapse = True
    is_bulk_processing = False
    numeric_input = True

    @staticmethod
    def run(
            measurement: np.ndarray,
            threshold: float = 0.0,
            gap_frames: int = 10,
            min_length: int = 40,
            relative: bool = True
        ) -> np.ndarray:
        """Return the ramp direction (+1, -1 or 0) of one cell's measurement.

        Parameters
        ----------
        measurement : numpy.ndarray
            One-dimensional measurement values for one cell, sorted by frame.
        threshold : float, optional
            Minimum change between consecutive observations to count as
            rising (change > threshold) or falling (change < -threshold).
            With 0, only the sign of the change matters.
        gap_frames : int, optional
            Interruptions of at most this many observations inside a ramp
            are bridged (e.g. 10 frames = 50 min at 5 min per frame).
        min_length : int, optional
            Minimum number of observations of a ramp. Shorter rising or
            falling runs are set to 0 (e.g. 40 frames = 3.3 h at 5 min per
            frame).
        relative : bool, optional
            If True, the change is divided by the absolute value of the next
            observation (relative change), so the threshold is a fraction
            per observation. If the next value is zero, the change is
            divided by the absolute value of the current observation
            instead (0 to 0 is no change), so changes towards or away from
            zero keep their sign. If False, the absolute change is used.

        Returns
        -------
        numpy.ndarray
            +1 (rising), -1 (falling) or 0 with the same shape as the input.
            Missing input values remain missing.
        """
        if (isinstance(threshold, bool)
                or not isinstance(threshold, numbers.Real) or threshold < 0):
            raise ValueError('threshold must be a number >= 0.')
        gap_frames = _check_int(gap_frames, 'gap_frames', 0)
        min_length = _check_int(min_length, 'min_length', 1)
        values = np.asarray(measurement, dtype=float)
        result = np.zeros(values.shape, dtype=float)
        if values.size > 1:
            change = np.diff(values)
            if relative:
                change = change / _relative_scale(values)
            change = np.nan_to_num(change, nan=0.0)
            up = change > threshold
            down = change < -threshold
            up = _keep_long_runs(_fill_gaps(up, gap_frames), min_length)
            down = _keep_long_runs(_fill_gaps(down, gap_frames), min_length)
            result[:-1] = up.astype(float) - down.astype(float)
        result[np.isnan(values)] = np.nan
        return result
