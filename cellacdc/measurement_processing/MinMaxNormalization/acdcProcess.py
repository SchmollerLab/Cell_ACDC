import numbers

import pandas as pd

name = 'Min-max normalization'


def _as_float(value, argname):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f'{argname} must be a number.')
    return float(value)


class Model:
    """Linear rescaling with fixed constants shared by all cells.

    Computes ``(measurement - background_value - min_value) /
    (max_value - min_value)`` for every value, so that ``min_value`` maps to
    0 and ``max_value`` maps to 1 (after subtracting the background).

    Use the same constants for all positions and conditions of an
    experiment, e.g. the 5th and 95th percentiles of the background-corrected
    intensity pooled over every cell and frame of a reference condition.
    Only then do values remain comparable between positions and conditions,
    which is required when comparing two channels with each other (e.g. FUCCI
    reporters). The defaults (0, 0, 1) leave the values unchanged. Missing
    values remain missing; results are not clipped to [0, 1].
    """

    is_timelapse = False
    is_bulk_processing = True
    numeric_input = True

    @staticmethod
    def run(
            measurement: pd.DataFrame,
            background_value: float = 0.0,
            min_value: float = 0.0,
            max_value: float = 1.0
        ) -> pd.DataFrame:
        """Rescale all selected measurements with fixed constants.

        Parameters
        ----------
        measurement : pandas.DataFrame
            Selected measurements of all cells and frames.
        background_value : float, optional
            Constant background subtracted from every value before
            normalizing, e.g. the median intensity outside of cells measured
            once for the whole experiment. Use 0 when the measurement is
            already background corrected.
        min_value : float, optional
            Background-corrected value that is mapped to 0. Typically a low
            percentile (e.g. 5th) pooled over all cells, frames, positions
            and conditions (or over a reference condition).
        max_value : float, optional
            Background-corrected value that is mapped to 1. Must be larger
            than min_value. Typically a high percentile (e.g. 95th) pooled
            in the same way as min_value.

        Returns
        -------
        pandas.DataFrame
            Normalized values with the same index and columns.
        """
        background_value = _as_float(background_value, 'background_value')
        min_value = _as_float(min_value, 'min_value')
        max_value = _as_float(max_value, 'max_value')
        if max_value <= min_value:
            raise ValueError('max_value must be larger than min_value.')
        return (
            (measurement.astype(float) - background_value - min_value)
            / (max_value - min_value)
        )
