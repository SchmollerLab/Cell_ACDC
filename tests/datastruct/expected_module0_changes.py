"""Expected GUI-visible changes of Module 0 compared with main.

Registry used by `test_golden_module0.py`: one `ExpectedChange` per change
id (GV-01 ... GV-20) that the refactor makes visible in a golden scenario.
`assert_golden` requires every golden diff to be claimed by an entry listed
for that scenario and every listed entry to claim at least one diff, so this
list is the reviewed list of GUI changes of the PR.

Created empty on main (the goldens were captured there); entries are added
together with the change that causes them.
"""
from __future__ import annotations

from .scenarios import ExpectedChange, assert_golden

__all__ = ['CHANGES', 'ExpectedChange', 'assert_golden']

CHANGES: list[ExpectedChange] = []
