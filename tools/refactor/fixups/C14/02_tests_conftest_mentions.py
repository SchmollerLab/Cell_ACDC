"""C14: the docstring and comments of tests/conftest.py that name moved
functions through the old myutils module (C14/01 covers cellacdc/ only).

``myutils.setupLogger`` and ``myutils.get_list_of_promptable_models`` are
module globals of the myutils package now; the conftest names their
modules with the canonical aliases (``app_info.setupLogger``,
``segm_models.get_list_of_promptable_models``), as CLAUDE.md and the code
do. Comments and a docstring only: no behaviour change.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _fixuplib import replace, run  # noqa: E402

CONFTEST = 'tests/conftest.py'
EDITS = [
    ('``logs_path`` (written and pruned by ``myutils.setupLogger``, e.g. when a',
     '``logs_path`` (written and pruned by ``app_info.setupLogger``, e.g. when a'),
    ('    # myutils.setupLogger writes there and deletes logs older than 7 days',
     '    # app_info.setupLogger writes there and deletes logs older than 7 days'),
    ('    # Custom promptable models (read by myutils.get_list_of_promptable_models)',
     '    # Custom promptable models (read by\n'
     '    # segm_models.get_list_of_promptable_models)'),
]


def main(repo):
    for old, new in EDITS:
        replace(repo, CONFTEST, old, new, count=1,
                items=[CONFTEST + '::__doc__'])


run(main)
