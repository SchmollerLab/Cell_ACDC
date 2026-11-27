"""Fake gui/main_window/toolbars.py: reads QMenu as a global."""
from qtpy.QtWidgets import QMenu


def make():
    return QMenu
