"""Expected behaviour changes of Re-structure compared with main.

Registry used by `test_golden_restructure.py`: one `ExpectedChange` per
change id (GV-R1 ... GV-R5) that the refactor makes visible in a golden
case. `assert_golden` requires every golden diff to be claimed by an entry
listed for that case and every listed entry to claim at least one diff.

Created empty on main (the goldens were captured there); the verbatim move
of the core keeps it empty, the fixes add one entry each.
"""
from __future__ import annotations

from .scenarios import ExpectedChange, assert_golden

__all__ = ['CHANGES', 'ExpectedChange', 'assert_golden']

CHANGES: list[ExpectedChange] = []
