"""CLI/GUI parity: ``acdc-data convert`` and GUI Module 0 write the same
files for the same inputs.

GUI: ``harness_module0.run_module0_scenario`` (real ``bioFormatsWorker``,
real offscreen metadata dialog with the scenario's scripted answers, real
bioio) with ``InProcessBackend``. CLI: ``data_cli.main`` with the flags that
correspond to the scenario's setup flags and dialog answers, on a fresh copy
of the same synthetic inputs. The destination and source manifests (decoded
pixels and TIFF tags, h5, metadata.csv rows, symbolic links, other files)
must be identical.

S14/S18 edit SizeT/SizeZ, S15 cancels and S16 is unreadable: they have no
CLI equivalent by design and are not compared.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from . import conditions, harness_module0, manifest, scenarios

# Scenario id -> `acdc-data convert` flags equivalent to its GUI answers.
CLI_FLAGS = {
    'S01': ['--layout', 'one-per-pos'],
    'S02': ['--layout', 'one-per-pos'],
    'S03': ['--layout', 'one-per-pos', '--channels', 'GFP,mCherry'],
    'S04': ['--layout', 'one-per-pos', '--save-channels', 'Channel_0_0',
            '--time-range', '2:3', '--image-format', 'h5'],
    'S05': ['--layout', 'one-per-pos', '--use-symlinks'],
    'S06': ['--layout', 'single-multi-pos', '--positions', '1,3'],
    'S07': ['--layout', 'single-multi-pos', '--positions', '1,3',
            '--add-image-name'],
    'S07b': ['--layout', 'single-multi-pos', '--positions', '1,3',
             '--add-image-name'],
    # The setup dialog passes only the image files; the CLI selects them by
    # extension because the source folder also holds the "other files".
    'S08': ['--layout', 'one-per-channel', '--ext', '.tif',
            '--other-files', 'copy', '--no-lazy-load'],
    'S09': ['--layout', 'one-per-channel'],
    'S10': ['--layout', 'one-per-pos', '--on-existing', 'add',
            '--channels', 'GFP,Channel_0_1', '--save-channels', 'GFP'],
    'S11': ['--layout', 'one-per-pos', '--on-existing', 'overwrite'],
    'S12a': ['--layout', 'one-per-pos', '--on-existing', 'create-new'],
    'S12b': ['--layout', 'one-per-channel', '--on-existing', 'create-new'],
    'S13': ['--layout', 'one-per-pos', '--move-raw', '--use-symlinks'],
    'S17': ['--layout', 'one-per-pos'],
}


def _manifests(src: str, dst: str) -> dict:
    placeholders = {'<SRC>': src, '<DST>': dst}
    return {
        part: manifest.build_manifest(
            root, placeholders=placeholders, include_env=False
        )
        for part, root in (('dst', dst), ('src', src))
    }


def _make_inputs(scenario_id: str, workdir) -> tuple[str, str]:
    src = os.path.join(str(workdir), 'src')
    dst = os.path.join(str(workdir), 'dst')
    os.makedirs(src)
    os.makedirs(dst)
    scenarios.MODULE0[scenario_id].make_inputs(src, dst)
    return src, dst


def _convert_argv(scenario_id: str, src: str, dst: str) -> list[str]:
    return [
        'convert', '-i', src, '-o', dst, *CLI_FLAGS[scenario_id],
        '--format', 'json', '--no-log-file',
    ]


def _gui_run(scenario_id: str, workdir) -> dict:
    from cellacdc.acdc_datastruct import backends

    record = harness_module0.run_module0_scenario(
        scenario_id, str(workdir), backend=backends.InProcessBackend()
    )
    assert record['exceptions'] == [], record['exceptions']
    return record


def _assert_same_files(gui: dict, cli: dict) -> None:
    written = [key for key in gui['dst'] if '/Images/' in key]
    assert written, 'the GUI wrote no Images files: nothing to compare'
    for part in ('dst', 'src'):
        diffs = manifest.diff_manifests(gui[part], cli[part])
        assert diffs == [], f'{part}: ' + '\n'.join(map(str, diffs))


def test_every_comparable_scenario_has_cli_flags():
    not_comparable = {'S14', 'S15', 'S16', 'S18a', 'S18b'}
    assert set(CLI_FLAGS) == set(scenarios.MODULE0_IDS) - not_comparable


@conditions.requires_qt
@conditions.requires_bioio
@pytest.mark.parametrize('scenario_id', list(CLI_FLAGS))
def test_cli_writes_the_same_files_as_the_gui(scenario_id, tmp_path, capsys):
    from cellacdc import data_cli
    from cellacdc.acdc_datastruct import backends

    gui = _gui_run(scenario_id, tmp_path / 'gui')
    capsys.readouterr()

    src, dst = _make_inputs(scenario_id, tmp_path / 'cli')
    returncode = data_cli.main(
        [*_convert_argv(scenario_id, src, dst), '-q'],
        backend_factory=backends.InProcessBackend,
    )
    out = capsys.readouterr().out
    assert returncode == 0, out
    assert json.loads(out)['status'] == 'ok'

    _assert_same_files(gui, _manifests(src, dst))


@conditions.subprocess_heavy
@conditions.requires_qt
@conditions.requires_bioio
@pytest.mark.parametrize('scenario_id', ['S01', 'S13'])
def test_cli_child_writes_the_same_files_as_the_gui(scenario_id, tmp_path):
    """`python -m cellacdc.data_cli` with the production backend (bioio in
    worker children), as a user runs it."""
    gui = _gui_run(scenario_id, tmp_path / 'gui')

    src, dst = _make_inputs(scenario_id, tmp_path / 'cli')
    env = conditions.hooks_env()
    env.pop(conditions.TEST_HOOKS_ENV)
    result = subprocess.run(
        [sys.executable, '-m', 'cellacdc.data_cli',
         *_convert_argv(scenario_id, src, dst)],
        env=env, cwd=conditions.REPO_ROOT, stdin=subprocess.DEVNULL,
        capture_output=True, timeout=600,
    )
    stderr = result.stderr.decode('utf-8', errors='replace')
    assert result.returncode == 0, stderr
    assert json.loads(result.stdout.decode('ascii'))['status'] == 'ok'

    _assert_same_files(gui, _manifests(src, dst))
