"""cellacdc._entry_points.parses_own_args and the config.py exemption."""
from __future__ import annotations

import json
import sys

import pytest

from cellacdc import _entry_points

from . import conditions

PYTHON = 'C:\\envs\\acdc\\python.exe'


@pytest.mark.parametrize('argv0, expected', [
    ('acdc-data', 'acdc-data'),
    ('acdc-data.exe', 'acdc-data'),
    ('ACDC-DATA.EXE', 'acdc-data'),
    ('C:\\x\\Scripts\\acdc-data.EXE', 'acdc-data'),
    ('/usr/bin/acdc-data', 'acdc-data'),
    ('acdc-data-script.py', 'acdc-data'),
    ('C:/x/Scripts/acdc-data-script.pyw', 'acdc-data'),
    ('acdc', 'acdc'),
    ('', ''),
])
def test_script_name(argv0, expected):
    assert _entry_points.script_name(argv0) == expected


@pytest.mark.parametrize('argv, orig_argv', [
    (['acdc-data', 'convert', '-h'], []),
    (['acdc-data.exe', 'metadata', '-i', 'x'], []),
    (['ACDC-DATA.EXE'], []),
    (['C:\\x\\Scripts\\acdc-data.EXE', '-h'], []),
    (['/usr/bin/acdc-data', '--help'], []),
    (['acdc-data-script.py', '-h'], []),
    (['acdc-setup-models', '-h'], []),
    (['C:\\x\\Scripts\\acdc-setup-models.exe'], []),
    (['-m', '-h'], [PYTHON, '-m', 'cellacdc.data_cli', '-h']),
    (['-m'], [PYTHON, '-mcellacdc.data_cli', 'metadata']),
    (['-m'], [PYTHON, '-X', 'utf8', '-m', 'cellacdc.data_cli']),
    (['x.py', 'job.json', '--server'],
     [PYTHON, '-m', 'cellacdc.acdc_datastruct._worker', 'job.json',
      '--server']),
    (['-m'], [PYTHON, '-m', 'cellacdc.service.setup_models', '-h']),
])
def test_parses_own_args_true(argv, orig_argv):
    assert _entry_points.parses_own_args(argv, orig_argv) is True


@pytest.mark.parametrize('argv, orig_argv', [
    (['acdc'], [PYTHON, 'acdc']),
    (['acdc', '-p', 'data'], [PYTHON, 'acdc', '-p', 'data']),
    (['acdc.exe', '-p', 'x.ini'], []),
    (['acdc', '--reset'], []),
    (['acdc-database'], []),
    (['acdc-data2'], []),
    (['spotmax', '-h'], []),
    (['-c'], [PYTHON, '-c', 'import cellacdc']),
    (['-m'], [PYTHON, '-m', 'cellacdc', '-p', 'x.ini']),
    (['-m'], [PYTHON, '-m', 'pytest', '-p', 'no:cacheprovider']),
    (['-m'], [PYTHON, '-m', 'cellacdc.data_cli_extra']),
    (['acdc', 'data'], []),
    ([], []),
    (['acdc'], ['cellacdc.data_cli']),
])
def test_parses_own_args_false(argv, orig_argv):
    assert _entry_points.parses_own_args(argv, orig_argv) is False


def test_parses_own_args_defaults_to_sys(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['C:\\x\\Scripts\\acdc-data.exe', '-h'])
    monkeypatch.setattr(sys, 'orig_argv', [PYTHON, 'acdc-data'], raising=False)
    assert _entry_points.parses_own_args() is True
    monkeypatch.setattr(sys, 'argv', ['-m'])
    monkeypatch.setattr(
        sys, 'orig_argv', [PYTHON, '-m', 'cellacdc.data_cli'], raising=False
    )
    assert _entry_points.parses_own_args() is True
    monkeypatch.setattr(sys, 'argv', ['acdc', '-p', 'x.ini'])
    monkeypatch.setattr(sys, 'orig_argv', [PYTHON, 'acdc'], raising=False)
    assert _entry_points.parses_own_args() is False


def test_parses_own_args_tolerates_missing_argv(monkeypatch):
    """Empty or missing sys.argv / sys.orig_argv -> False."""
    monkeypatch.delattr(sys, 'orig_argv', raising=False)
    monkeypatch.setattr(sys, 'argv', [])
    assert _entry_points.parses_own_args() is False
    monkeypatch.delattr(sys, 'argv')
    assert _entry_points.parses_own_args() is False
    assert _entry_points.parses_own_args([], []) is False


def test_own_parser_tables():
    assert _entry_points.OWN_PARSER_SCRIPTS == {
        'acdc-setup-models': 'cellacdc.service.setup_models',
        'acdc-data': 'cellacdc.data_cli',
    }
    assert _entry_points.OWN_PARSER_MODULES == frozenset({
        'cellacdc.service.setup_models', 'cellacdc.data_cli',
        'cellacdc.acdc_datastruct._worker',
    })


# Runs in one child: for every argv, re-execute cellacdc.config (the module
# parses sys.argv at import time) and report parser_args or the SystemExit.
_CONFIG_PROBE = r'''
import contextlib, importlib, io, json, sys
cases = json.loads(sys.argv[1])
sys.argv[:] = ['acdc']
import cellacdc.config as config
out = []
for argv in cases:
    sys.argv[:] = argv
    stdout = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout):
            importlib.reload(config)
    except SystemExit as err:
        out.append({'exit': err.code, 'stdout': stdout.getvalue()})
        continue
    args = config.parser_args
    out.append({key: args.get(key) for key in
                ('params', 'reset', 'info', 'version', 'debug', 'yes')})
print('@@probe@@' + json.dumps(out))
'''


@conditions.subprocess_heavy
def test_config_parser_args_in_child():
    """`acdc`, `acdc -p x.ini`, `acdc --reset` parse as before; own-parser
    entry points skip the acdc parse (`-h`, `-i`, bare `-p` untouched)."""
    cases = [
        ['acdc'],
        ['acdc', '-p', 'x.ini'],
        ['acdc', '--reset'],
        ['acdc', '-v'],
        ['acdc', '-h'],
        ['acdc-data', 'metadata', '-i', 'x', '-p'],
        ['C:\\x\\Scripts\\acdc-data.exe', 'convert', '--help'],
        ['acdc-setup-models', '-h'],
    ]
    result = conditions.run_python(_CONFIG_PROBE, json.dumps(cases))
    assert result.returncode == 0, result.stderr[-3000:]
    line = [
        line for line in result.stdout.splitlines()
        if line.startswith('@@probe@@')
    ][-1]
    out = json.loads(line[len('@@probe@@'):])
    defaults = {
        'params': '', 'reset': False, 'info': False, 'version': False,
        'yes': False,
    }

    def parsed(**changes):
        return {**defaults, **changes}

    def without_debug(entry):
        return {key: value for key, value in entry.items() if key != 'debug'}

    assert without_debug(out[0]) == parsed()
    assert without_debug(out[1]) == parsed(params='x.ini')
    assert without_debug(out[2]) == parsed(reset=True)
    assert without_debug(out[3]) == parsed(version=True)
    assert out[4]['exit'] == 0
    assert out[4]['stdout'].startswith('usage: acdc')
    assert 'acdc-data -h' in out[4]['stdout']
    for entry in out[5:]:
        assert 'exit' not in entry, entry
        assert without_debug(entry) == parsed()
