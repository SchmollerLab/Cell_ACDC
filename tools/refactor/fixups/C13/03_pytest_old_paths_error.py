"""C13: pytest turns the MovedWarning of old import paths into an error.

pyproject.toml filterwarnings gets
``error:Cell-ACDC import path moved:DeprecationWarning`` after
``ignore::DeprecationWarning`` (pytest applies the last matching entry):
Cell-ACDC's own code and tests must use the new paths. A message filter,
not ``error::cellacdc._compat.MovedWarning``: pytest resolves a dotted
category while reading the configuration, which would import cellacdc
before the conftest sets QT_QPA_PLATFORM. tests/test_compat.py, which uses
old paths on purpose, catches the warnings with pytest.warns /
warnings.catch_warnings and runs its subprocess checks with -W always.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import _fixuplib as F  # noqa: E402

OLD = '''# Ignore DeprecationWarnings
filterwarnings = [
    "ignore::DeprecationWarning",
    "ignore:cannot import name '_C' from 'sam2':UserWarning",
]
'''
NEW = '''# Ignore DeprecationWarnings, except the ones of old Cell-ACDC import paths
# (compat shims, cellacdc/_compat.py): Cell-ACDC's code and tests use the new
# paths (CODEMAP.md). Later entries win.
filterwarnings = [
    "ignore::DeprecationWarning",
    "error:Cell-ACDC import path moved:DeprecationWarning",
    "ignore:cannot import name '_C' from 'sam2':UserWarning",
]
'''


def main(repo):
    F.replace(repo, 'pyproject.toml', OLD, NEW,
              items=['pyproject.toml::tool.pytest.ini_options'])


if __name__ == '__main__':
    F.run(main)
