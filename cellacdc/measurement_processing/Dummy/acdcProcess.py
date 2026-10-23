import numpy as np

name = 'Dummy'


class Model:
    """Return measurements unchanged (per-frame processing)."""

    is_timelapse = False
    is_bulk_processing = False
    numeric_input = True

    @staticmethod
    def run(measurement: np.ndarray) -> np.ndarray:
        """Copy one measurement across the cells in a frame.

        Parameters
        ----------
        measurement : numpy.ndarray
            One-dimensional measurement values for a single frame.

        Returns
        -------
        numpy.ndarray
            An independent copy of the input, including missing values.
        """
        return measurement.copy()
