"""Deprecated: the contents of workers.py moved to other modules.

Its names now live in 35 modules.

Old names are forwarded lazily, with a DeprecationWarning
(cellacdc/_compat.py). See CODEMAP.md.
"""
__cellacdc_compat__ = True

from cellacdc import _compat

__getattr__, __dir__ = _compat.moved_module(__name__)
