"""Fake shared/widgets/line_edits.py: binds Qt names it imports."""
import qtpy
from qtpy.QtWidgets import QMenu

is_mac = False


def menu():
    return QMenu
