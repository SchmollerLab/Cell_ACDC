"""Tests of the shared subprocess runner (``runner.py``) and of its child
entry point (``_worker.py``).

In-process part:

- helpers (``child_env``, ``same_path``, ``build_command``, ...);
- ``_worker.main`` with the test ops, always with
  ``event_stream=io.StringIO()`` so no file descriptor is touched;
- ``run_task`` against a scripted stand-in for ``subprocess.Popen``
  (exit-code table, drain order, cancellation by the parent's own flag,
  line cap, timeout, cleanup).

An autouse guard checks that fd 1, ``sys.stdout`` and ``sys.stderr`` are
unchanged after every test.

Subprocess part (``subprocess_heavy``): real children. Package identity,
remote tracebacks and the event channel are tested with the real cellacdc;
transport-only cases (crash, cancel, Ctrl-C, timeout, package mismatch, fd
restore) use a "light" cellacdc (a minimal package plus byte copies of
``runner.py`` and ``_worker.py``), which starts in well under a second.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

import cellacdc
from cellacdc.acdc_datastruct import _worker, errors, runner

from . import conditions, fake_bioio, qt_free_rules, runner_test_ops

OPS = runner_test_ops.OPS_MODULE
HOOKS = {runner.TEST_HOOKS_ENV: '1'}
PACKAGE_DIR = os.path.dirname(os.path.abspath(cellacdc.__file__))
EXPECTED_PACKAGE_DIR = os.path.normcase(os.path.realpath(PACKAGE_DIR))
ODD_NAME = 'a, b \u00b5 dir'
STATUS_CONTROL_C_EXIT = 3221225786
REPORT = {
    'protocol': 1,
    'task': 'save_positions',
    'error_type': 'TaskError',
    'message': 'expected output missing',
    'traceback': 'Traceback (most recent call last):\n  ...\n'
                 'TaskError: expected output missing\n',
    'context': {'path': 'C:/x, y/\u00b5.tif'},
}


TASKS_AVAILABLE = (
    importlib.util.find_spec(_worker.TASKS_MODULE) is not None
)
requires_tasks = pytest.mark.skipif(
    not TASKS_AVAILABLE, reason=f'{_worker.TASKS_MODULE} not there yet'
)


# ------------------------------------------------------------------ guards
def _fd1_identity() -> tuple:
    fd = os.dup(1)
    try:
        stat = os.fstat(fd)
    finally:
        os.close(fd)
    return stat.st_mode, stat.st_dev, stat.st_ino


@pytest.fixture(autouse=True)
def stdio_guard():
    """fd 1 and the sys streams are unchanged after every test."""
    before = (_fd1_identity(), sys.stdout, sys.stderr)
    yield
    assert (_fd1_identity(), sys.stdout, sys.stderr) == before


@pytest.fixture
def hooks(monkeypatch):
    monkeypatch.setenv(runner.TEST_HOOKS_ENV, '1')


# ----------------------------------------------------------------- helpers
def test_protocol_constants():
    assert runner.PROTOCOL == 1
    assert runner.WORKER_MODULE == 'cellacdc.acdc_datastruct._worker'
    assert runner.WORKER_MODULE == _worker.__name__
    assert runner.EVENT_PREFIX == '@@acdc-event@@ '
    assert runner.TEST_HOOKS_ENV == conditions.TEST_HOOKS_ENV
    assert runner.OUTPUT_LINE_MAX == 4096
    assert runner.SERVER_FLAG == cellacdc.SERVER_FLAG
    assert _worker.TASKS_MODULE == 'cellacdc.acdc_datastruct.tasks'


def test_runner_options_defaults():
    options = runner.RunnerOptions()
    assert dataclasses.astuple(options) == (
        None, True, None, False, None, None
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        options.timeout = 1


def test_build_command():
    job_path = os.path.join('C:', os.sep, ODD_NAME, '-job.json')
    assert runner.build_command(job_path, runner.RunnerOptions()) == [
        sys.executable, '-m', runner.WORKER_MODULE, job_path, '--server'
    ]
    options = runner.RunnerOptions(python='py', server_mode=False)
    assert runner.build_command('j', options) == [
        'py', '-m', runner.WORKER_MODULE, 'j'
    ]


def test_package_root():
    root = runner.package_root()
    assert root == os.path.dirname(PACKAGE_DIR)
    assert runner.same_path(os.path.join(root, 'cellacdc'), PACKAGE_DIR)


def test_child_env_prepends_package_root(monkeypatch, tmp_path):
    root = str(tmp_path / ODD_NAME)
    monkeypatch.setattr(runner, 'package_root', lambda: root)
    base = {
        'PYTHONPATH': os.pathsep.join(['a', 'b']), 'KEEP': '1',
        'PYTHONIOENCODING': 'cp1252',
    }
    original = dict(base)
    env = runner.child_env(base, extra={'KEEP': '2', 'NEW': '3'})
    assert env['PYTHONPATH'] == os.pathsep.join([root, 'a', 'b'])
    assert env['PYTHONIOENCODING'] == 'utf-8'
    assert env['PYTHONUNBUFFERED'] == '1'
    assert (env['KEEP'], env['NEW']) == ('2', '3')
    assert base == original
    # No trailing separator: an empty PYTHONPATH entry means the cwd.
    assert runner.child_env({})['PYTHONPATH'] == root
    monkeypatch.setenv('ACDC_RUNNER_TEST_PROBE', 'x')
    assert runner.child_env()['ACDC_RUNNER_TEST_PROBE'] == 'x'


@pytest.mark.parametrize(
    'case', ['site-packages', 'dist-packages', 'getsitepackages', 'usersite']
)
def test_child_env_never_prepends_site_packages(monkeypatch, tmp_path, case):
    """Prepending a site-packages folder could shadow the stdlib."""
    if case == 'site-packages':
        root = str(tmp_path / 'lib' / 'Site-Packages')
    elif case == 'dist-packages':
        root = str(tmp_path / 'lib' / 'dist-packages')
    else:
        root = str(tmp_path / 'custom lib')
        same_root = os.path.join(root, 'x', '..')
        if case == 'getsitepackages':
            monkeypatch.setattr(
                runner.site, 'getsitepackages',
                lambda: [str(tmp_path / 'other'), same_root]
            )
        else:
            monkeypatch.setattr(
                runner.site, 'getusersitepackages', lambda: same_root
            )
    monkeypatch.setattr(runner, 'package_root', lambda: root)
    assert runner.child_env({'PYTHONPATH': 'a'})['PYTHONPATH'] == 'a'
    assert 'PYTHONPATH' not in runner.child_env({})


def test_same_path(tmp_path):
    folder = str(tmp_path / ODD_NAME)
    os.mkdir(folder)
    assert runner.same_path(folder, os.path.join(folder, 'sub', '..'))
    assert runner.same_path(folder, folder + os.sep)
    assert not runner.same_path(folder, str(tmp_path))
    if sys.platform == 'win32':
        # ASCII only: the micro sign upper-cases to the Greek capital Mu.
        plain = str(tmp_path / 'Case Dir, x')
        assert runner.same_path(plain, plain.upper())
        assert runner.same_path(folder, folder.replace('\\', '/'))


def test_same_path_resolves_symlinks(tmp_path):
    folder = tmp_path / 'target'
    folder.mkdir()
    link = tmp_path / 'link'
    try:
        os.symlink(folder, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip('cannot create symlinks here')
    assert runner.same_path(str(link), str(folder))


def test_resolve_reference():
    resolved = runner.resolve_reference(fake_bioio.FAKE_BIOIMAGE_PATH)
    assert resolved is fake_bioio.FakeBioImage
    for spec in ('no_colon', ':attr', 'module:', ''):
        with pytest.raises(ValueError):
            runner.resolve_reference(spec)
    with pytest.raises(ImportError):
        runner.resolve_reference('tests.datastruct.no_such_module:x')
    with pytest.raises(AttributeError):
        runner.resolve_reference('tests.datastruct.fake_bioio:NoSuchName')


def test_worker_error_text_and_attributes():
    context = {'path': 'x.tif'}
    err = runner.WorkerError(
        'save_positions', 'expected output missing', returncode=1,
        error_type='TaskError', child_traceback=REPORT['traceback'],
        output_tail='noise', context=context
    )
    context['other'] = 1
    assert isinstance(err, RuntimeError)
    assert str(err) == (
        'Worker task save_positions failed (exit 1): TaskError: expected '
        'output missing\nTraceback (most recent call last):\n  ...\n'
        'TaskError: expected output missing'
    )
    assert (err.task, err.message, err.returncode, err.error_type) == (
        'save_positions', 'expected output missing', 1, 'TaskError'
    )
    assert (err.output_tail, err.context, err.hint) == (
        'noise', {'path': 'x.tif'}, None
    )
    crash = runner.WorkerError(
        'ping', 'worker exited with code -11 before reporting',
        returncode=-11, output_tail='Fatal Python error\nSegfault\n'
    )
    assert str(crash) == (
        'Worker task ping failed (exit -11): worker exited with code -11 '
        'before reporting\nFatal Python error\nSegfault'
    )
    bare = runner.WorkerError('ping', 'no result')
    assert str(bare) == 'Worker task ping failed: no result'
    assert (bare.context, bare.child_traceback, bare.output_tail) == (
        {}, '', ''
    )


@pytest.mark.parametrize(
    'relpath', ['acdc_datastruct/runner.py', 'acdc_datastruct/_worker.py']
)
def test_qt_free_and_stdlib_only(relpath):
    rules = qt_free_rules.rules_for(relpath)
    assert rules == {
        'qf': True, 'stdlib_only': True, 'new_file': True,
        'parent_never_opens_raw': False,
    }
    path = os.path.join(qt_free_rules.PACKAGE_DIR, *relpath.split('/'))
    violations = qt_free_rules.check_file(path, **rules)
    assert violations == [], qt_free_rules.format_violations(violations)


# ------------------------------------------------- _worker.main in-process
def make_job(folder, *, job_name: str = runner.JOB_FILENAME,
             raw: str | None = None, drop=(), **changes) -> str:
    """Write a job file into `folder` (created) and return its path."""
    os.makedirs(folder, exist_ok=True)
    job = {
        'protocol': runner.PROTOCOL,
        'task': '_echo',
        'params': {},
        'out_dir': os.path.abspath(folder),
        'expected_package_dir': EXPECTED_PACKAGE_DIR,
        'reader_factory': None,
        'ops_module': OPS,
    }
    job.update(changes)
    for key in drop:
        del job[key]
    path = os.path.join(folder, job_name)
    with open(path, 'w', encoding='utf-8') as file:
        file.write(json.dumps(job) if raw is None else raw)
    return path


def run_worker(*argv: str) -> tuple[int, list[dict]]:
    """`_worker.main` in-process; returns (exit code, events)."""
    stream = io.StringIO()
    code = _worker.main(list(argv), event_stream=stream)
    lines = stream.getvalue().splitlines()
    assert all(line.startswith(runner.EVENT_PREFIX) for line in lines)
    prefix_len = len(runner.EVENT_PREFIX)
    return code, [json.loads(line[prefix_len:]) for line in lines]


def read_report(folder, name: str) -> dict | None:
    path = os.path.join(folder, name)
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def test_worker_success_writes_result_and_events(hooks, tmp_path):
    folder = tmp_path / 'run'
    params = {'events': 3, 'marker': 't\u00ebst, \u00b5', 'n': [1, 2.5]}
    job_path = make_job(
        folder, params=params, reader_factory=fake_bioio.FAKE_BIOIMAGE_PATH
    )
    code, events = run_worker(job_path)
    assert code == runner.EXIT_OK
    assert [event['index'] for event in events] == [0, 1, 2]
    assert events[0] == {
        'kind': 'log', 'level': 'INFO', 'message': 'event 0', 'index': 0
    }
    report = read_report(folder, runner.RESULT_FILENAME)
    assert (report['protocol'], report['task']) == (1, '_echo')
    result = report['result']
    assert result['params'] == params
    assert result['out_dir'] == str(folder)
    assert result['reader_factory'] == fake_bioio.FAKE_BIOIMAGE_PATH
    assert (folder / 'marker.txt').read_text('utf-8') == params['marker']
    assert sorted(os.listdir(folder)) == [
        runner.JOB_FILENAME, 'marker.txt', runner.RESULT_FILENAME
    ]
    with open(os.path.join(folder, runner.RESULT_FILENAME), 'rb') as file:
        assert file.read().isascii()


def test_worker_accepts_trailing_server_flag(hooks, tmp_path):
    code, _ = run_worker(make_job(tmp_path), runner.SERVER_FLAG)
    assert code == runner.EXIT_OK


def test_worker_paths_with_commas_spaces_and_micro(hooks, tmp_path):
    folder = tmp_path / ODD_NAME
    job_path = make_job(folder, job_name='j\u00f6b, 1.json', params={'x': 1})
    code, _ = run_worker(job_path, runner.SERVER_FLAG)
    assert code == runner.EXIT_OK
    assert read_report(folder, runner.RESULT_FILENAME)['result']['params'] == {
        'x': 1
    }


def test_worker_relative_job_path_with_leading_dash(
        hooks, tmp_path, monkeypatch):
    folder = tmp_path / ODD_NAME
    make_job(folder, job_name='-job, \u00b5.json')
    monkeypatch.chdir(folder)
    code, _ = run_worker('-job, \u00b5.json')
    assert code == runner.EXIT_OK
    assert read_report(folder, runner.RESULT_FILENAME) is not None


def test_worker_task_exception_writes_error_report(hooks, tmp_path, capsys):
    context = {'path': 'C:/x, y/\u00b5.tif', 'n': 3}
    job_path = make_job(
        tmp_path, task='_raise',
        params={'events': 2, 'message': 'bad, \u00b5', 'context': context}
    )
    code, events = run_worker(job_path)
    assert code == runner.EXIT_TASK_FAILED
    assert len(events) == 2
    report = read_report(tmp_path, runner.ERROR_FILENAME)
    assert {key: report[key] for key in (
        'protocol', 'task', 'error_type', 'message', 'context'
    )} == {
        'protocol': 1, 'task': '_raise', 'error_type': 'OpError',
        'message': 'bad, \u00b5', 'context': context,
    }
    assert 'ValueError: original cause' in report['traceback']
    assert report['traceback'].rstrip().endswith('OpError: bad, \u00b5')
    assert 'OpError: bad, \u00b5' in capsys.readouterr().err
    assert read_report(tmp_path, runner.RESULT_FILENAME) is None


def test_worker_keyboard_interrupt_exits_130(hooks, tmp_path):
    job_path = make_job(tmp_path, task='_raise', params={'keyboard': True})
    code, _ = run_worker(job_path)
    assert code == runner.EXIT_INTERRUPTED
    assert read_report(tmp_path, runner.RESULT_FILENAME) is None


def test_worker_unserialisable_result_is_a_task_failure(
        hooks, tmp_path, monkeypatch):
    monkeypatch.setitem(
        runner_test_ops.TASKS, '_bad_result',
        lambda params, ctx: {'value': object()}
    )
    code, _ = run_worker(make_job(tmp_path, task='_bad_result'))
    assert code == runner.EXIT_TASK_FAILED
    assert read_report(tmp_path, runner.ERROR_FILENAME)['error_type'] == (
        'TypeError'
    )
    assert sorted(os.listdir(tmp_path)) == [
        runner.ERROR_FILENAME, runner.JOB_FILENAME
    ]


BAD_JOBS = {
    'invalid JSON': dict(raw='{"protocol": 1,'),
    'not an object': dict(raw='[1, 2]'),
    'missing key': dict(drop=('params',)),
    'unknown key': dict(extra=1),
    'protocol': dict(protocol=2),
    'params type': dict(params=[1]),
    'task type': dict(task=5),
    'relative out_dir': dict(out_dir='relative/dir'),
    'hook type': dict(ops_module=5),
}


@pytest.mark.parametrize('job', list(BAD_JOBS.values()), ids=list(BAD_JOBS))
def test_worker_rejects_invalid_jobs(hooks, tmp_path, job):
    code, events = run_worker(make_job(tmp_path, **job))
    assert (code, events) == (runner.EXIT_BAD_JOB, [])
    report = read_report(tmp_path, runner.ERROR_FILENAME)
    assert report['error_type'] == 'JobError'
    assert report['message']
    assert read_report(tmp_path, runner.RESULT_FILENAME) is None


def test_worker_missing_job_file(tmp_path, capsys):
    code, _ = run_worker(str(tmp_path / runner.JOB_FILENAME))
    assert code == runner.EXIT_BAD_JOB
    assert 'cannot read the job file' in capsys.readouterr().err
    assert read_report(tmp_path, runner.ERROR_FILENAME)['error_type'] == (
        'JobError'
    )


@pytest.mark.parametrize(
    'argv', [[], ['a.json', 'b.json'], ['--server'], ['a.json', '--verbose']]
)
def test_worker_rejects_bad_command_lines(argv, capsys):
    code, _ = run_worker(*argv)
    assert code == runner.EXIT_BAD_JOB
    assert 'usage:' in capsys.readouterr().err


def test_worker_unknown_task(hooks, tmp_path):
    code, _ = run_worker(make_job(tmp_path, task='_nope'))
    assert code == runner.EXIT_BAD_JOB
    report = read_report(tmp_path, runner.ERROR_FILENAME)
    assert report['message'] == "unknown task '_nope'"
    assert report['task'] == '_nope'


@pytest.mark.parametrize('hook', [
    dict(ops_module=OPS),
    dict(ops_module=None, task='ping',
         reader_factory=fake_bioio.FAKE_BIOIMAGE_PATH),
], ids=['ops_module', 'reader_factory'])
def test_worker_test_hooks_need_the_env_gate(tmp_path, monkeypatch, hook):
    monkeypatch.delenv(runner.TEST_HOOKS_ENV, raising=False)
    code, _ = run_worker(make_job(tmp_path, **hook))
    assert code == runner.EXIT_BAD_JOB
    report = read_report(tmp_path, runner.ERROR_FILENAME)
    assert runner.TEST_HOOKS_ENV in report['message']
    monkeypatch.setenv(runner.TEST_HOOKS_ENV, 'yes')
    code, _ = run_worker(make_job(tmp_path, **hook))
    assert code == runner.EXIT_BAD_JOB


@pytest.mark.parametrize('hook', [
    dict(reader_factory='no_colon'),
    dict(reader_factory='tests.datastruct.no_such_module:X'),
    dict(reader_factory='tests.datastruct.fake_bioio:NoSuchName'),
    dict(ops_module='tests.datastruct.no_such_module'),
])
def test_worker_rejects_unresolvable_hooks(hooks, tmp_path, hook):
    code, _ = run_worker(make_job(tmp_path, **hook))
    assert code == runner.EXIT_BAD_JOB
    report = read_report(tmp_path, runner.ERROR_FILENAME)
    assert report['message'].startswith('cannot load a test hook')


def test_worker_package_mismatch_exits_3(hooks, tmp_path):
    other = str(tmp_path / 'other cellacdc')
    code, events = run_worker(
        make_job(tmp_path / 'run', expected_package_dir=other)
    )
    assert (code, events) == (runner.EXIT_PACKAGE_MISMATCH, [])
    report = read_report(tmp_path / 'run', runner.ERROR_FILENAME)
    assert report['error_type'] == 'PackageMismatchError'
    assert PACKAGE_DIR in report['message'] and other in report['message']


@requires_tasks
def test_worker_production_ping_in_process(tmp_path, monkeypatch):
    monkeypatch.delenv(runner.TEST_HOOKS_ENV, raising=False)
    code, _ = run_worker(make_job(tmp_path, task='ping', ops_module=None))
    assert code == runner.EXIT_OK
    result = read_report(tmp_path, runner.RESULT_FILENAME)['result']
    assert result['protocol'] == runner.PROTOCOL


def test_pytest_stdout_still_works(capfd):
    """After the in-process worker cases pytest's capture still works."""
    print('stdout via print')
    sys.stdout.flush()
    os.write(1, b'stdout via fd 1\n')
    out = capfd.readouterr().out
    assert 'stdout via print' in out and 'stdout via fd 1' in out


# ------------------------------------------- run_task with a scripted child
@dataclasses.dataclass
class ChildScript:
    """What a scripted child does."""
    returncode: int = 0
    stdout: tuple = ()
    stderr: tuple = ()
    files: dict = dataclasses.field(default_factory=dict)
    running: bool = False            # runs until terminated or killed
    ignore_terminate: bool = False
    stopped_returncode: int = 1      # after terminate()/kill() (Windows)


class FakeProcess:
    """Scripted stand-in for the ``subprocess.Popen`` object of runner.

    It writes ``script.files`` into the run folder, serves the scripted
    lines through byte pipes (Windows line ends) and "exits" immediately
    unless ``script.running``: then only terminate()/kill() end it.
    """
    def __init__(self, script: ChildScript, command, *, cwd, env, stdin,
                 stdout, stderr) -> None:
        self.script = script
        self.command = command
        self.cwd = cwd
        self.env = env
        self.stdin = stdin
        self.pipes = (stdout, stderr)
        with open(os.path.join(cwd, runner.JOB_FILENAME), 'rb') as file:
            self.job_bytes = file.read()
        for name, content in script.files.items():
            if not isinstance(content, str):
                content = json.dumps(content)
            with open(os.path.join(cwd, name), 'w', encoding='utf-8') as file:
                file.write(content)
        self.stdout = io.BytesIO(self._encode(script.stdout))
        self.stderr = io.BytesIO(self._encode(script.stderr))
        self.returncode = None
        self.calls = []

    @staticmethod
    def _encode(lines) -> bytes:
        return ''.join(f'{line}\r\n' for line in lines).encode('utf-8')

    def poll(self):
        if self.returncode is None and not self.script.running:
            self.returncode = self.script.returncode
        return self.returncode

    def wait(self, timeout=None):
        if self.poll() is None:
            raise subprocess.TimeoutExpired(self.command, timeout)
        return self.returncode

    def terminate(self) -> None:
        self.calls.append('terminate')
        if not self.script.ignore_terminate:
            self.returncode = self.script.stopped_returncode

    def kill(self) -> None:
        self.calls.append('kill')
        self.returncode = self.script.stopped_returncode


def _patch_popen(monkeypatch, popen) -> None:
    monkeypatch.setattr(runner, 'subprocess', SimpleNamespace(
        Popen=popen, PIPE=subprocess.PIPE, DEVNULL=subprocess.DEVNULL,
        TimeoutExpired=subprocess.TimeoutExpired,
    ))


@pytest.fixture
def fake_child(monkeypatch):
    """``fake_child(**ChildScript fields)`` makes run_task start scripted
    children; returns the list of FakeProcess objects it creates."""
    processes = []

    def install(**script_fields) -> list[FakeProcess]:
        script = ChildScript(**script_fields)

        def popen(command, **kwargs):
            process = FakeProcess(script, command, **kwargs)
            processes.append(process)
            return process

        _patch_popen(monkeypatch, popen)
        return processes
    return install


def event_line(**event) -> str:
    return runner.EVENT_PREFIX + json.dumps(event)


def test_run_task_job_file_and_command(fake_child, tmp_path):
    processes = fake_child(files={runner.RESULT_FILENAME: {'result': 'ok'}})
    temp_root = tmp_path / ODD_NAME
    options = runner.RunnerOptions(
        temp_root=str(temp_root), extra_env={'ACDC_RUNNER_EXTRA': 'y'}
    )
    params = {'path': 'C:/d\u00e0ta, x/\u00b5.tif', 'n': [1, 2]}
    result = runner.run_task(
        'read_metadata', params, options=options, reader_factory='m:f',
        ops_module='m.ops'
    )
    assert result == 'ok'
    (process,) = processes
    run_dir = process.cwd
    assert os.path.isabs(run_dir)
    assert runner.same_path(os.path.dirname(run_dir), str(temp_root))
    assert os.path.basename(run_dir).startswith('acdc_datastruct_')
    job_path = os.path.join(run_dir, runner.JOB_FILENAME)
    assert process.command == runner.build_command(job_path, options)
    assert process.stdin is subprocess.DEVNULL
    assert process.pipes == (subprocess.PIPE, subprocess.PIPE)
    assert process.job_bytes.isascii()
    assert json.loads(process.job_bytes) == {
        'protocol': 1, 'task': 'read_metadata', 'params': params,
        'out_dir': run_dir, 'expected_package_dir': EXPECTED_PACKAGE_DIR,
        'reader_factory': 'm:f', 'ops_module': 'm.ops',
    }
    expected_env = runner.child_env(extra=options.extra_env)
    assert process.env == expected_env
    assert not os.path.exists(run_dir)
    assert os.listdir(temp_root) == []


def test_exit_0_without_result(fake_child):
    fake_child(returncode=0, stderr=['bye'])
    with pytest.raises(runner.WorkerError) as info:
        runner.run_task('ping', {})
    err = info.value
    assert (err.returncode, err.message, err.output_tail) == (
        0, 'no result', 'bye'
    )


def test_exit_1_reports_the_remote_error(fake_child):
    fake_child(returncode=1, stderr=['noise'],
               files={runner.ERROR_FILENAME: REPORT})
    with pytest.raises(runner.WorkerError) as info:
        runner.run_task('save_positions', {})
    err = info.value
    assert (err.task, err.returncode, err.error_type, err.message) == (
        'save_positions', 1, 'TaskError', 'expected output missing'
    )
    assert err.child_traceback == REPORT['traceback']
    assert err.context == REPORT['context']
    assert (err.output_tail, err.hint) == ('noise', None)
    assert str(err).startswith(
        'Worker task save_positions failed (exit 1): TaskError: expected '
        'output missing\nTraceback'
    )


@pytest.mark.parametrize('returncode', [1, 7, -9, STATUS_CONTROL_C_EXIT])
def test_exit_without_report_uses_the_output_tail(fake_child, returncode):
    fake_child(returncode=returncode, stdout=['stdout noise'],
               stderr=['Fatal Python error: boom'])
    with pytest.raises(runner.WorkerError) as info:
        runner.run_task('read_metadata', {})
    err = info.value
    assert err.returncode == returncode
    assert err.message == (
        f'worker exited with code {returncode} before reporting'
    )
    assert err.error_type is None
    assert sorted(err.output_tail.split('\n')) == [
        'Fatal Python error: boom', 'stdout noise'
    ]
    assert 'Fatal Python error: boom' in str(err)


def test_exit_2_bad_job(fake_child):
    report = dict(REPORT, error_type='JobError', message='unknown task',
                  traceback='', context={})
    fake_child(returncode=2, files={runner.ERROR_FILENAME: report})
    with pytest.raises(runner.WorkerError) as info:
        runner.run_task('nope', {})
    assert (info.value.error_type, info.value.message, info.value.hint) == (
        'JobError', 'unknown task', None
    )


def test_exit_3_has_the_package_hint(fake_child):
    report = dict(REPORT, error_type='PackageMismatchError',
                  message='the worker imported cellacdc from "a"', context={})
    fake_child(returncode=3, files={runner.ERROR_FILENAME: report})
    with pytest.raises(runner.WorkerError) as info:
        runner.run_task('ping', {})
    assert info.value.returncode == runner.EXIT_PACKAGE_MISMATCH
    assert info.value.hint == runner.PACKAGE_MISMATCH_HINT


def test_exit_130_is_a_cancellation(fake_child):
    fake_child(returncode=runner.EXIT_INTERRUPTED)
    with pytest.raises(errors.ConversionCancelled):
        runner.run_task('save_positions', {})


@pytest.mark.parametrize('returncode', [0, 1])
def test_everything_is_dispatched_before_return_or_raise(
        fake_child, returncode):
    """The child "exits" at once; every queued line is still delivered,
    in order and on the calling thread, before the result or the error."""
    if returncode == 0:
        files = {runner.RESULT_FILENAME: {'result': 'done'}}
    else:
        files = {runner.ERROR_FILENAME: REPORT}
    fake_child(
        returncode=returncode, files=files,
        stdout=[event_line(kind='log', index=i) for i in range(300)],
        stderr=[f'err {i}' for i in range(50)],
    )
    events, output, threads = [], [], set()

    def on_event(event):
        events.append(event['index'])
        threads.add(threading.get_ident())

    def on_output(line):
        output.append(line)
        threads.add(threading.get_ident())

    def run():
        return runner.run_task(
            'save_positions', {}, on_event=on_event, on_output=on_output
        )

    if returncode == 0:
        assert run() == 'done'
    else:
        with pytest.raises(runner.WorkerError):
            run()
    assert events == list(range(300))
    assert output == [f'err {i}' for i in range(50)]
    assert threads == {threading.get_ident()}


def test_output_lines_are_classified_and_capped(fake_child):
    long_line = 'x' * 10000
    tqdm_line = 'progress 10%\rprogress 20%\rprogress 30%'
    bad_events = [
        runner.EVENT_PREFIX + '{not json', runner.EVENT_PREFIX + '[1]'
    ]
    stderr_event = event_line(kind='log', index=99)
    fake_child(
        files={runner.RESULT_FILENAME: {'result': None}},
        stdout=['import noise', *bad_events, event_line(kind='log', index=0)],
        stderr=[tqdm_line, long_line, stderr_event],
    )
    events, output = [], []
    runner.run_task('t', {}, on_event=events.append, on_output=output.append)
    assert events == [{'kind': 'log', 'index': 0}]
    assert sorted(output) == sorted([
        'import noise', *bad_events, tqdm_line,
        long_line[:runner.OUTPUT_LINE_MAX], stderr_event,
    ])


def test_output_tail_keeps_the_last_lines(fake_child):
    fake_child(returncode=5, stderr=[f'line {i}' for i in range(250)])
    with pytest.raises(runner.WorkerError) as info:
        runner.run_task('t', {})
    assert info.value.output_tail.split('\n') == [
        f'line {i}' for i in range(50, 250)
    ]


def test_collect_runs_before_cleanup(fake_child, tmp_path):
    fake_child(files={
        runner.RESULT_FILENAME: {'result': {'arrays': ['sample_0.npy']}},
        'sample_0.npy': 'pixels',
    })
    seen = {}

    def collect(result, run_dir):
        with open(os.path.join(run_dir, result['arrays'][0])) as file:
            seen.update(run_dir=run_dir, data=file.read())
        return 'collected'

    options = runner.RunnerOptions(temp_root=str(tmp_path))
    assert runner.run_task(
        'read_sample', {}, options=options, collect=collect
    ) == 'collected'
    assert seen['data'] == 'pixels'
    assert not os.path.exists(seen['run_dir'])


def test_keep_run_dir(fake_child, tmp_path, caplog):
    fake_child(files={runner.RESULT_FILENAME: {'result': 1}})
    caplog.set_level(logging.INFO, logger=runner.__name__)
    options = runner.RunnerOptions(keep_run_dir=True, temp_root=str(tmp_path))
    assert runner.run_task('t', {}, options=options) == 1
    (name,) = os.listdir(tmp_path)
    run_dir = os.path.join(tmp_path, name)
    assert sorted(os.listdir(run_dir)) == [
        runner.JOB_FILENAME, runner.RESULT_FILENAME
    ]
    assert run_dir in caplog.text


def test_unserialisable_params_leave_nothing(fake_child, tmp_path):
    processes = fake_child()
    options = runner.RunnerOptions(temp_root=str(tmp_path))
    with pytest.raises(TypeError):
        runner.run_task('t', {'x': object()}, options=options)
    assert processes == []
    assert os.listdir(tmp_path) == []


@pytest.mark.parametrize(
    'stopped_returncode', [1, -15, 0, 130, STATUS_CONTROL_C_EXIT]
)
def test_cancel_is_decided_by_the_parent_flag(
        fake_child, tmp_path, stopped_returncode):
    """Whatever the child's return code after terminate() (Windows: 1),
    and even with a report on disk, a cancel raises ConversionCancelled."""
    processes = fake_child(
        running=True, stdout=[event_line(kind='started')],
        stopped_returncode=stopped_returncode,
        files={runner.ERROR_FILENAME: REPORT,
               runner.RESULT_FILENAME: {'result': 1}},
    )
    events = []
    with pytest.raises(errors.ConversionCancelled):
        runner.run_task(
            't', {}, options=runner.RunnerOptions(temp_root=str(tmp_path)),
            on_event=events.append, is_cancelled=lambda: bool(events)
        )
    assert events == [{'kind': 'started'}]
    assert processes[0].calls == ['terminate']
    assert os.listdir(tmp_path) == []


def test_keyboard_interrupt_stops_the_child_and_delivers_events(
        fake_child, tmp_path):
    """Ctrl-C: the parent's KeyboardInterrupt decides, not the child's
    STATUS_CONTROL_C_EXIT; queued events are still delivered."""
    processes = fake_child(
        running=True, stopped_returncode=STATUS_CONTROL_C_EXIT,
        stdout=[event_line(kind='log', index=i) for i in range(20)],
    )
    events = []

    def interrupt():
        if events:
            raise KeyboardInterrupt
        return False

    with pytest.raises(KeyboardInterrupt):
        runner.run_task(
            't', {}, options=runner.RunnerOptions(temp_root=str(tmp_path)),
            on_event=events.append, is_cancelled=interrupt
        )
    assert [event['index'] for event in events] == list(range(20))
    assert processes[0].calls == ['terminate']
    assert os.listdir(tmp_path) == []


def test_timeout_stops_the_child(fake_child):
    processes = fake_child(running=True, stderr=['still working'])
    start = time.monotonic()
    options = runner.RunnerOptions(timeout=0.3)
    timed_out = pytest.raises(
        runner.WorkerError, match=r'timed out after 0\.3 s'
    )
    with timed_out as info:
        runner.run_task('t', {}, options=options)
    assert time.monotonic() - start < 5
    assert info.value.output_tail == 'still working'
    assert processes[0].calls == ['terminate']


def test_a_child_ignoring_terminate_is_killed(fake_child):
    processes = fake_child(running=True, ignore_terminate=True)
    with pytest.raises(errors.ConversionCancelled):
        runner.run_task('t', {}, is_cancelled=lambda: True)
    assert processes[0].calls == ['terminate', 'kill']


def test_a_callback_error_stops_the_child(fake_child):
    processes = fake_child(
        running=True, stdout=[event_line(index=i) for i in range(5)]
    )
    calls = []

    def on_event(event):
        calls.append(event)
        raise ValueError('callback bug')

    with pytest.raises(ValueError, match='callback bug'):
        runner.run_task('t', {}, on_event=on_event)
    assert len(calls) == 1
    assert processes[0].calls == ['terminate']


# --------------------------------------------------------- real children
@pytest.fixture(scope='module')
def light_root(tmp_path_factory) -> str:
    """Root of a "light" cellacdc: a minimal package plus byte copies of
    runner.py and _worker.py, for children that test the transport only."""
    root = tmp_path_factory.mktemp('light') / 'light root, \u00b5'
    package = root / 'cellacdc'
    (package / 'acdc_datastruct').mkdir(parents=True)
    (package / '__init__.py').write_text(
        "import sys\nRUNNING_AS_SERVER = '--server' in sys.argv[1:]\n",
        encoding='utf-8'
    )
    source_dir = os.path.dirname(os.path.abspath(runner.__file__))
    for name in ('runner.py', '_worker.py'):
        shutil.copyfile(
            os.path.join(source_dir, name),
            package / 'acdc_datastruct' / name
        )
    return str(root)


def light_options(light_root: str, **kwargs) -> runner.RunnerOptions:
    """Children import the light cellacdc first and the test ops from the
    repository."""
    pythonpath = os.pathsep.join([light_root, conditions.REPO_ROOT])
    return runner.RunnerOptions(
        extra_env={**HOOKS, 'PYTHONPATH': pythonpath}, **kwargs
    )


@pytest.fixture
def light_parent(monkeypatch, light_root):
    """The parent expects the light package, so the identity check of the
    light children passes."""
    monkeypatch.setattr(
        runner, '_package_dir', lambda: os.path.join(light_root, 'cellacdc')
    )


@pytest.fixture
def popen_spy(monkeypatch) -> list[subprocess.Popen]:
    """Real children; the Popen objects are recorded."""
    processes = []

    def popen(command, **kwargs):
        process = subprocess.Popen(command, **kwargs)
        processes.append(process)
        return process

    _patch_popen(monkeypatch, popen)
    return processes


@conditions.subprocess_heavy
def test_child_imports_the_parents_cellacdc(tmp_path):
    """Real child: package identity, server mode, cwd, environment, a
    non-ASCII temp root with comma and space, and every event delivered
    in order on the calling thread before the result."""
    temp_root = tmp_path / 't\u00ebmp dir, x'
    params = {
        'events': 200, 'marker': 't\u00ebst \u00b5, ok', 'text': '\u00fc'
    }
    events, threads, seen = [], set(), {}

    def on_event(event):
        events.append(event['index'])
        threads.add(threading.get_ident())

    def collect(result, run_dir):
        marker_path = os.path.join(run_dir, 'marker.txt')
        with open(marker_path, encoding='utf-8') as file:
            seen.update(run_dir=run_dir, marker=file.read())
        return result

    result = runner.run_task(
        '_echo', params,
        options=runner.RunnerOptions(
            temp_root=str(temp_root), extra_env=HOOKS
        ),
        reader_factory=fake_bioio.FAKE_BIOIMAGE_PATH, ops_module=OPS,
        on_event=on_event, collect=collect
    )
    assert events == list(range(200))
    assert threads == {threading.get_ident()}
    run_dir = seen['run_dir']
    assert seen['marker'] == params['marker']
    assert result['params'] == params
    assert runner.same_path(result['cellacdc_file'], cellacdc.__file__)
    assert result['running_as_server'] is True
    assert result['argv'] == [
        os.path.join(run_dir, runner.JOB_FILENAME), runner.SERVER_FLAG
    ]
    assert runner.same_path(result['cwd'], run_dir)
    assert result['out_dir'] == run_dir
    assert runner.same_path(os.path.dirname(run_dir), str(temp_root))
    assert result['pid'] != os.getpid()
    assert result['env'] == {
        'PYTHONIOENCODING': 'utf-8', 'PYTHONUNBUFFERED': '1',
        runner.TEST_HOOKS_ENV: '1',
    }
    assert result['pythonpath'] == runner.child_env()['PYTHONPATH']
    assert result['reader_factory'] == fake_bioio.FAKE_BIOIMAGE_PATH
    assert os.listdir(temp_root) == []


@conditions.subprocess_heavy
def test_child_exception_gives_the_remote_traceback(tmp_path):
    context = {'path': 'C:/x, y/\u00b5.tif', 'n': 3}
    events = []
    with pytest.raises(runner.WorkerError) as info:
        runner.run_task(
            '_raise',
            {'events': 20, 'message': 'bad, \u00b5', 'context': context},
            options=runner.RunnerOptions(
                temp_root=str(tmp_path), extra_env=HOOKS
            ),
            ops_module=OPS, on_event=events.append
        )
    err = info.value
    assert (err.returncode, err.error_type, err.message) == (
        1, 'OpError', 'bad, \u00b5'
    )
    assert err.context == context
    assert 'ValueError: original cause' in err.child_traceback
    assert 'in _raise' in err.child_traceback
    assert str(err).startswith(
        'Worker task _raise failed (exit 1): OpError: bad, \u00b5\nTraceback'
    )
    assert 'OpError: bad, \u00b5' in err.output_tail
    assert [event['index'] for event in events] == list(range(20))
    assert os.listdir(tmp_path) == []


@conditions.subprocess_heavy
def test_event_channel_survives_noise(tmp_path):
    """print, C-level fd-1 writes, tqdm-style '\\r' updates, a 10,000
    character line and a grandchild never corrupt the events."""
    events, output = [], []
    result = runner.run_task(
        '_print_noise', {'lines': 3, 'long_line': 10000, 'grandchild': True},
        options=runner.RunnerOptions(temp_root=str(tmp_path), extra_env=HOOKS),
        ops_module=OPS, on_event=events.append, on_output=output.append
    )
    assert result == {'lines': 3}
    assert [event['message'] for event in events] == [
        'noisy 0', 'noisy 1', 'noisy 2'
    ]
    for index in range(3):
        assert f'NOISE print {index}' in output
        assert f'NOISE fd1 {index}' in output
    assert 'NOISE grandchild' in output
    tqdm_line = ''.join(f'progress {p}%\r' for p in range(0, 101, 25))
    assert output.count(tqdm_line.rstrip('\r')) == 3
    assert 'L' * runner.OUTPUT_LINE_MAX in output
    assert all(len(line) <= runner.OUTPUT_LINE_MAX for line in output)
    assert not any(line.startswith(runner.EVENT_PREFIX) for line in output)


@requires_tasks
@conditions.subprocess_heavy
def test_production_ping_child(tmp_path, monkeypatch):
    monkeypatch.delenv(runner.TEST_HOOKS_ENV, raising=False)
    result = runner.run_task(
        'ping', {}, options=runner.RunnerOptions(temp_root=str(tmp_path))
    )
    assert result['protocol'] == runner.PROTOCOL
    assert runner.same_path(result['cellacdc_file'], cellacdc.__file__)
    assert result['pid'] != os.getpid()


@conditions.subprocess_heavy
def test_crash_reports_the_output_tail(light_root, light_parent, tmp_path):
    events = []
    with pytest.raises(runner.WorkerError) as info:
        runner.run_task(
            '_exit',
            {'events': 5, 'stderr': 'about to crash, \u00b5\n', 'code': 7},
            options=light_options(light_root, temp_root=str(tmp_path)),
            ops_module=OPS, on_event=events.append
        )
    err = info.value
    assert (err.returncode, err.error_type, err.child_traceback) == (
        7, None, ''
    )
    assert err.message == 'worker exited with code 7 before reporting'
    assert 'about to crash, \u00b5' in err.output_tail
    assert 'about to crash, \u00b5' in str(err)
    assert [event['index'] for event in events] == list(range(5))
    assert os.listdir(tmp_path) == []


@conditions.subprocess_heavy
def test_cancel_terminates_the_running_child(
        light_root, light_parent, popen_spy, tmp_path):
    """ConversionCancelled whatever the child's rc; on Windows terminate()
    gives rc 1. The run folder is removed."""
    events = []
    start = time.monotonic()
    with pytest.raises(errors.ConversionCancelled):
        runner.run_task(
            '_sleep', {'seconds': 60},
            options=light_options(light_root, temp_root=str(tmp_path)),
            ops_module=OPS, on_event=events.append,
            is_cancelled=lambda: bool(events)
        )
    assert time.monotonic() - start < 30
    assert [event['kind'] for event in events] == ['started']
    (process,) = popen_spy
    assert process.returncode not in (None, 0)
    if sys.platform == 'win32':
        assert process.returncode == 1
    assert os.listdir(tmp_path) == []


@conditions.subprocess_heavy
def test_keyboard_interrupt_terminates_the_running_child(
        light_root, light_parent, popen_spy, tmp_path):
    events = []

    def interrupt():
        if events:
            raise KeyboardInterrupt
        return False

    with pytest.raises(KeyboardInterrupt):
        runner.run_task(
            '_sleep', {'seconds': 60},
            options=light_options(light_root, temp_root=str(tmp_path)),
            ops_module=OPS, on_event=events.append, is_cancelled=interrupt
        )
    (process,) = popen_spy
    assert process.returncode is not None
    assert os.listdir(tmp_path) == []


@conditions.subprocess_heavy
def test_timeout_kills_the_running_child(
        light_root, light_parent, popen_spy, tmp_path):
    start = time.monotonic()
    with pytest.raises(runner.WorkerError, match='timed out after 3 s'):
        runner.run_task(
            '_sleep', {'seconds': 60},
            options=light_options(
                light_root, timeout=3, temp_root=str(tmp_path)
            ),
            ops_module=OPS
        )
    assert time.monotonic() - start < 30
    assert popen_spy[0].returncode is not None
    assert os.listdir(tmp_path) == []


@conditions.subprocess_heavy
def test_package_mismatch_exit_3(light_root, tmp_path):
    """The child imports the light cellacdc while the parent expects the
    real one."""
    with pytest.raises(runner.WorkerError) as info:
        runner.run_task(
            '_echo', {},
            options=light_options(light_root, temp_root=str(tmp_path)),
            ops_module=OPS
        )
    err = info.value
    assert err.returncode == runner.EXIT_PACKAGE_MISMATCH
    assert err.error_type == 'PackageMismatchError'
    assert err.hint == runner.PACKAGE_MISMATCH_HINT
    assert os.path.join(light_root, 'cellacdc') in err.message
    assert os.listdir(tmp_path) == []


@conditions.subprocess_heavy
def test_worker_main_restores_stdout(light_root, tmp_path):
    """Without ``event_stream`` the worker swaps fd 1 and sys.stdout for
    the run (noise, fd-1 writes and a grandchild go to stderr) and restores
    both afterwards."""
    folder = tmp_path / ODD_NAME
    job_path = make_job(
        folder, task='_print_noise', params={'lines': 2, 'grandchild': True},
        expected_package_dir=os.path.join(light_root, 'cellacdc')
    )
    code = (
        'import os, sys\n'
        'from cellacdc.acdc_datastruct import _worker\n'
        "rc = _worker.main([sys.argv[1], '--server'])\n"
        "print(f'AFTER print {rc}')\n"
        'sys.stdout.flush()\n'
        "os.write(1, b'AFTER fd1\\n')\n"
    )
    env = conditions.hooks_env(
        PYTHONPATH=os.pathsep.join([light_root, conditions.REPO_ROOT]),
        PYTHONIOENCODING='utf-8'
    )
    proc = subprocess.run(
        [sys.executable, '-c', code, job_path], env=env, cwd=str(folder),
        stdin=subprocess.DEVNULL, capture_output=True, timeout=120
    )
    stdout = proc.stdout.decode('utf-8').splitlines()
    stderr = proc.stderr.decode('utf-8', errors='replace')
    assert proc.returncode == 0, stderr
    events = [line for line in stdout if line.startswith(runner.EVENT_PREFIX)]
    assert len(events) == 2
    assert [line for line in stdout if line not in events] == [
        'AFTER print 0', 'AFTER fd1'
    ]
    for text in ('NOISE print 0', 'NOISE fd1 1', 'NOISE grandchild'):
        assert text in stderr
