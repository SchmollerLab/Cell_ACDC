"""Deprecated: moved to segmentation_module/main_window.py.

Importing this module gives the module at the new path (the same object)
and emits a DeprecationWarning. See CODEMAP.md.
"""
__cellacdc_compat__ = True

from cellacdc import _compat

_compat.alias_module(__name__)
