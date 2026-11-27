"""Codemod for the Cell-ACDC package-layout refactor (stdlib only).

Modules:

- source.py: byte-exact file I/O (encoding, BOM, line endings) and line
  based editing (FileEdit).
- items.py: top-level items and class members with their exact source
  spans (decorators, comment blocks, trailing comments) and mapping keys.
- scopes.py: scope-aware references to module globals (symtable + ast),
  eager (import-time) positions and per-function hazards.
- imports.py: import bindings, relative <-> absolute resolution, import
  retargeting and emission (with guards).
- plan.py: the mapping shards (tools/refactor/mapping/*.json): loading,
  validation, per-commit locations and the name table.
- move.py: applies the moves of one commit to the working tree.
- shims.py: compat shims, compat blocks and cellacdc/_compat_moves.py.
- plan_check.py: static checks of the mapping.

See tools/refactor/README.md.
"""
import os

#: The commit the mapping keys refer to (PR #1328 head). Override with the
#: environment variable ACDC_REFACTOR_BASE (run.py replay sets it).
DEFAULT_BASE_COMMIT = '4c4e498b'

#: The commit sequence of the refactor (plan "Commit sequence").
COMMITS = [
    'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10',
    'C11', 'C12', 'C13', 'C14', 'C15', 'C16'
]

#: Commits whose automatic step is done by move.py / shims.py.
MOVE_COMMITS = ['C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11']

TOOL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_REPO_ROOT = os.path.dirname(os.path.dirname(TOOL_DIR))


class RefactorError(Exception):
    """An error in the mapping or in the tree that stops the codemod."""


def base_commit():
    return os.environ.get('ACDC_REFACTOR_BASE') or DEFAULT_BASE_COMMIT


def commit_index(commit):
    try:
        return COMMITS.index(commit)
    except ValueError:
        raise RefactorError(
            f'Unknown commit id {commit!r} (expected one of {COMMITS})'
        )


def commit_le(a, b):
    """True if commit `a` comes before or is `b`."""
    return commit_index(a) <= commit_index(b)


def commit_lt(a, b):
    return commit_index(a) < commit_index(b)
