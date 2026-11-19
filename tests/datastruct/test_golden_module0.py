"""Module 0 goldens.

Each scenario of `scenarios.MODULE0` runs through `harness_module0` (real
`bioFormatsWorker`, real offscreen `QDialogMetadataXML`, real bioio) and is
compared with `golden/module0/<id>.json` (captured on main) under the
registry `expected_module0_changes.CHANGES`:

- in process (`InProcessBackend`): when the worker accepts `backend=`, i.e.
  after the refactor; skipped on main;
- through real child processes: only with ACDC_SLOW_TESTS=1 (main: the
  legacy children; after the refactor: the default subprocess backend).

`test_cascade_golden_reproduces` re-captures the legacy metadata cascade
(`readMetadata`, kept verbatim for the python-bioformats path) in process.
"""
from __future__ import annotations

import glob
import os
import re

import pytest

from . import (
    conditions, expected_module0_changes, harness_module0, manifest, scenarios
)


def _compare(scenario_id: str, tmp_path, backend) -> None:
    golden = scenarios.load_golden('module0', scenario_id)
    actual = harness_module0.run_module0_scenario(
        scenario_id, str(tmp_path / 'work'), backend=backend
    )
    actual[manifest.ENV_KEY] = manifest.env_info()
    scenarios.assert_golden(
        scenario_id, golden, actual, expected_module0_changes.CHANGES
    )


@conditions.requires_qt
@conditions.requires_bioio
@pytest.mark.parametrize('scenario_id', scenarios.MODULE0_IDS)
def test_module0_golden_in_process(scenario_id, tmp_path):
    if not harness_module0.worker_accepts_backend():
        pytest.skip('bioFormatsWorker has no backend= parameter (main): '
                    'compare through real children with ACDC_SLOW_TESTS=1')
    from cellacdc.acdc_datastruct import backends

    _compare(scenario_id, tmp_path, backends.InProcessBackend())


@conditions.slow
@conditions.requires_qt
@conditions.requires_bioio
@pytest.mark.parametrize('scenario_id', scenarios.MODULE0_IDS)
def test_module0_golden_children(scenario_id, tmp_path):
    _compare(scenario_id, tmp_path, None)


def test_module0_registry_is_consistent():
    assert scenarios.check_registry(
        expected_module0_changes.CHANGES, scenarios.MODULE0_IDS
    ) == []


@conditions.requires_qt
def test_cascade_golden_reproduces(tmp_path):
    golden = scenarios.load_golden('cascade')
    actual = harness_module0.capture_cascade(str(tmp_path / 'work'))
    diffs = manifest.diff_manifests(golden, actual)
    assert diffs == [], manifest.format_diffs(diffs)


# A traceback line with a folder, a Windows drive path or a POSIX home folder
_MACHINE_PATH = re.compile(
    r'File "[^"\n]*[\\/]|(?<![A-Za-z])[A-Za-z]:[\\/]|/(?:home|Users)/'
)


def _strings(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, str):
        yield value


def test_goldens_are_in_the_dump_json_format(tmp_path):
    """Every golden is the text `manifest.dump_json` writes for its data,
    so a recapture changes only the lines whose values change."""
    folder = os.path.join(os.path.dirname(__file__), 'golden')
    paths = glob.glob(os.path.join(folder, '**', '*.json'), recursive=True)
    assert paths
    rewritten = str(tmp_path / 'golden.json')
    for path in paths:
        manifest.dump_json(manifest.load_json(path), rewritten)
        with open(path, encoding='utf-8') as file:
            text = file.read()
        with open(rewritten, encoding='utf-8') as file:
            assert file.read() == text, os.path.relpath(path, folder)


def test_goldens_hold_no_machine_paths():
    """Paths are placeholders and tracebacks keep only file names, so no
    golden holds a path of the machine that captured it."""
    folder = os.path.join(os.path.dirname(__file__), 'golden')
    paths = glob.glob(os.path.join(folder, '**', '*.json'), recursive=True)
    assert paths
    found = [
        (os.path.relpath(path, folder), text[:200])
        for path in paths
        for text in _strings(manifest.load_json(path))
        if _MACHINE_PATH.search(text)
    ]
    assert found == []


def test_strip_traceback_paths():
    text = (
        'Traceback (most recent call last):\n'
        '  File "C:\\Users\\me\\repo\\cellacdc\\dataStruct.py", line 1, '
        'in run\n'
        '  File "/home/me/env/site-packages/bioio/bio_image.py", line 2, '
        'in f\n'
        'IndexError: list index out of range\n'
    )
    assert harness_module0.strip_traceback_paths(text) == (
        'Traceback (most recent call last):\n'
        '  File "dataStruct.py", line 1, in run\n'
        '  File "bio_image.py", line 2, in f\n'
        'IndexError: list index out of range\n'
    )
    assert _MACHINE_PATH.search(text)
    assert not _MACHINE_PATH.search(
        harness_module0.strip_traceback_paths(text)
    )
