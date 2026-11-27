"""Verification of the package-layout refactor (plan section "Verification").

Every script snapshots the tree it is pointed at (``--repo``, default: this
checkout) into a folder and compares two snapshots; each also has a
``run(commit)`` function that snapshots into ``tools/refactor/out/<commit>/``
and compares with ``tools/refactor/out/baseline/`` (the snapshot of the base
commit, before any move), writing ``<script>_compare.json`` next to it.

- fingerprint.py: static AST fingerprints of every class, function, method
  and module-level statement; old keys map to new ones through the name
  table and the mapping shards; references rewritten to the same new home are
  no change. Every other difference must be listed in ``changes.jsonl``.
- mro_snapshot.py: for guiWin and MeasurementsViewerWin, which function every
  attribute resolves to (and the override chains, Qt shadowing, Signals).
- import_all.py: every module imported alone in a fresh interpreter, in GUI,
  NoQt and server modes, with the loaded modules and a global-state snapshot.
- headless.py: the entry points that must work without Qt.
- collect_ids.py: pytest node ids (collection only).
- importtime.py: ``-X importtime`` of the launcher, GUI and viewer modules.
- wheel_diff.py: sdist and wheel member lists from a clean worktree.
- smoke.py: construct, show and close every window and utility offscreen.
- common.py: paths, the one exclusion list of the import sweeps, the name
  table/shard mapping (``Mapping``), child processes, settings isolation.

The scripts use only the standard library and parse on Python 3.10.
``run_all`` runs the fast ones (and optionally all of them) for a commit.
"""


def run_all(commit, repo=None, before='baseline', slow=False):
    """Run the verification scripts for `commit`; returns {script: report}.

    Fast: fingerprint, mro_snapshot, headless. With `slow`: also import_all,
    smoke, collect_ids, importtime and wheel_diff (minutes each; run them in
    the background).
    """
    try:
        from . import (collect_ids, fingerprint, headless, import_all,
                       importtime, mro_snapshot, smoke, wheel_diff)
    except ImportError:  # run as plain scripts from this folder
        import collect_ids
        import fingerprint
        import headless
        import import_all
        import importtime
        import mro_snapshot
        import smoke
        import wheel_diff
    reports = {
        'fingerprint': fingerprint.run(commit, repo, before),
        'mro_snapshot': mro_snapshot.run(commit, repo, before),
    }
    if slow:
        reports['import_all'] = import_all.run(commit, repo, before)
    reports['headless'] = headless.run(commit, repo, before)
    if slow:
        reports['smoke'] = smoke.run(commit, repo, before)
        reports['collect_ids'] = collect_ids.run(commit, repo, before)
        reports['importtime'] = importtime.run(commit, repo, before)
        reports['wheel_diff'] = wheel_diff.run(commit, repo, before)
    return reports
