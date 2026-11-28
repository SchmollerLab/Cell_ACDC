"""Base class of the Cell-ACDC dialogs (from apps.py).

QBaseDialog subclasses the non-modal _base_widgets.py QBaseDialog (exec_
runs a local event loop, Escape is ignored). Foundation leaf: it imports
only _base_widgets.py at module level, because most dialog modules
subclass it at import time. spotMAX subclasses it too.
"""
from cellacdc import _base_widgets


class QBaseDialog(_base_widgets.QBaseDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
