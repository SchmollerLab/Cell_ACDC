"""Test-only measurement processors.

The identity processor ``Dummy`` is not shipped with Cell-ACDC (it would
appear in the users' recipe dropdown). Tests that need a step which leaves
values unchanged register it with the ``dummy_processor`` fixture
(``tests/conftest.py``), which calls :func:`register_dummy`.
"""
import numpy as np

from cellacdc import measurement_processing

DUMMY_NAME = 'Dummy'


class Dummy:
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


def register_dummy(monkeypatch):
    """Register :class:`Dummy` as ``'Dummy'`` for the current test.

    The entry has the same keys as a discovered processor, so recipes,
    selectors and parameter dialogs treat it like a shipped step. The
    registry is restored when ``monkeypatch`` is undone.
    """
    monkeypatch.setitem(
        measurement_processing.MEASUREMENT_PROCESSING_MAPPER, DUMMY_NAME, {
            'class': Dummy,
            'function': Dummy.run,
            'function_name': DUMMY_NAME,
            'docstring': Dummy.__doc__,
        }
    )
    return Dummy
