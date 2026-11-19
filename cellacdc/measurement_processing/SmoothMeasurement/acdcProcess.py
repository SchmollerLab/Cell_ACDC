import numpy as np
import pandas as pd

name = 'Smooth measurement'


class Model:
    """Centered moving average per cell, ignoring missing observations.

    Missing input values remain missing. At the ends, use the available
    observations. Time windows count observations, not elapsed time or
    absent frames.
    """

    is_timelapse = True
    is_bulk_processing = False
    numeric_input = True

    @staticmethod
    def run(measurement: np.ndarray, window: int = 3) -> np.ndarray:
        """Smooth a single cell's measurement in frame order.

        Parameters
        ----------
        measurement : numpy.ndarray
            One-dimensional measurement values for one cell, sorted by frame.
        window : int, optional
            Positive number of observations in the centered moving window.
            At track ends, use the available observations. Missing input
            values remain missing and are ignored when averaging.

        Returns
        -------
        numpy.ndarray
            Smoothed values with the same shape as the input.
        """
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError('The smoothing window must be a positive integer.')
        values = pd.Series(measurement, dtype=float)
        result = values.rolling(window, center=True, min_periods=1).mean()
        return result.mask(values.isna()).to_numpy()
