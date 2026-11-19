"""Child entry point of the shared data-structure runner::

    python -m cellacdc.acdc_datastruct._worker JOB_JSON [--server]

Runs the task described by ``JOB_JSON`` (written by ``runner.run_task``) and
exits with 0 (``result.json`` written), 1 (the task raised; ``error.json``),
2 (invalid command line or job; ``error.json`` when the job folder exists),
3 (this process imported a different cellacdc than the parent;
``error.json``) or 130 (interrupted). ``result.json`` and ``error.json`` are
written next to the job file.

stdout is the event channel: before anything runs, file descriptor 1 is
duplicated for the events and then pointed at stderr, so stray prints,
C-level writes, progress bars and grandchildren cannot corrupt it.

An ops module (production: ``cellacdc.acdc_datastruct.tasks``; with
``ACDC_DATASTRUCT_TEST_HOOKS=1`` the job's ``ops_module``) exposes
``TASKS`` (name -> ``function(params, ctx) -> dict``) and ``TaskContext``
(constructed with ``out_dir=``, ``emit=``, ``reader_factory=``). The
``context`` mapping of an exception (``tasks.TaskError``) goes into
``error.json``.

Standard library only at module level.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import traceback
from collections.abc import Mapping
from types import ModuleType
from typing import Callable, TextIO

TASKS_MODULE = 'cellacdc.acdc_datastruct.tasks'
_JOB_TYPES = {
    'protocol': int,
    'task': str,
    'params': dict,
    'out_dir': str,
    'expected_package_dir': str,
}
_JOB_HOOK_KEYS = ('reader_factory', 'ops_module')


class _Rejected(Exception):
    """The job cannot run: exit with ``code`` and report ``error_type``."""
    def __init__(self, code: int, error_type: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.error_type = error_type


def main(
        argv: list[str] | None = None, *, event_stream: TextIO | None = None
    ) -> int:
    """Run the job named on the command line and return the exit code.

    Parameters
    ----------
    argv : list of str, optional
        ``[job_path]`` plus an optional trailing ``--server`` (default
        ``sys.argv[1:]``).
    event_stream : text stream, optional
        Where events are written (in-process tests). When given, no file
        descriptor is touched. Otherwise stdout (fd 1) becomes the private
        event channel and fd 1 / ``sys.stdout`` are pointed at stderr until
        this function returns.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if event_stream is not None:
        return _run(args, event_stream)
    sys.stdout.flush()
    saved_fd1 = os.dup(1)
    events = os.fdopen(os.dup(1), 'w', encoding='utf-8', buffering=1)
    os.dup2(2, 1)
    saved_stdout, sys.stdout = sys.stdout, sys.stderr
    try:
        return _run(args, events)
    finally:
        sys.stderr.flush()
        os.dup2(saved_fd1, 1)
        os.close(saved_fd1)
        sys.stdout = saved_stdout
        events.close()


def _run(args: list[str], events: TextIO) -> int:
    from cellacdc.acdc_datastruct import runner
    try:
        return _execute(args, events, runner)
    except KeyboardInterrupt:
        return runner.EXIT_INTERRUPTED


def _execute(args: list[str], events: TextIO, runner: ModuleType) -> int:
    if args and args[-1] == runner.SERVER_FLAG:
        args = args[:-1]
    if len(args) != 1:
        _stderr(
            f'usage: python -m {runner.WORKER_MODULE} JOB_JSON '
            f'[{runner.SERVER_FLAG}] (got {args!r})'
        )
        return runner.EXIT_BAD_JOB
    job_path = os.path.abspath(args[0])
    run_dir = os.path.dirname(job_path)
    task = None
    try:
        job = _load_job(job_path, runner)
        task = job['task']
        _check_package(job, runner)
        reader_factory, ops = _test_hooks(job, runner)
    except _Rejected as err:
        return _reject(run_dir, runner, task, err)
    try:
        if ops is None:
            ops = importlib.import_module(TASKS_MODULE)
        function = ops.TASKS.get(task)
        if function is None:
            raise _Rejected(
                runner.EXIT_BAD_JOB, 'JobError', f'unknown task {task!r}'
            )
        ctx = ops.TaskContext(
            out_dir=job['out_dir'], emit=_event_writer(events, runner),
            reader_factory=reader_factory
        )
        result = function(job['params'], ctx)
        _write_json(
            os.path.join(run_dir, runner.RESULT_FILENAME),
            {'protocol': runner.PROTOCOL, 'task': task, 'result': result}
        )
    except _Rejected as err:
        return _reject(run_dir, runner, task, err)
    except Exception as err:
        return _fail(run_dir, runner, task, err)
    return runner.EXIT_OK


def _load_job(job_path: str, runner: ModuleType) -> dict:
    def bad(message: str) -> _Rejected:
        return _Rejected(runner.EXIT_BAD_JOB, 'JobError', message)

    try:
        with open(job_path, 'r', encoding='utf-8') as file:
            job = json.load(file)
    except (OSError, ValueError) as err:
        raise bad(f'cannot read the job file "{job_path}": {err}')
    if not isinstance(job, dict):
        raise bad('the job file does not contain a JSON object')
    missing = sorted(set(_JOB_TYPES) - set(job))
    unknown = sorted(set(job) - set(_JOB_TYPES) - set(_JOB_HOOK_KEYS))
    if missing or unknown:
        raise bad(f'invalid job keys: missing {missing}, unknown {unknown}')
    if job['protocol'] != runner.PROTOCOL:
        raise bad(
            f'unsupported protocol {job["protocol"]!r} '
            f'(this worker speaks {runner.PROTOCOL})'
        )
    for key, kind in _JOB_TYPES.items():
        if not isinstance(job[key], kind):
            raise bad(f'job key {key!r} must be a {kind.__name__}')
    for key in _JOB_HOOK_KEYS:
        if not isinstance(job.get(key), (str, type(None))):
            raise bad(f'job key {key!r} must be a string or null')
    if not os.path.isabs(job['out_dir']):
        raise bad(f'out_dir must be absolute: "{job["out_dir"]}"')
    return job


def _check_package(job: dict, runner: ModuleType) -> None:
    import cellacdc
    actual = os.path.dirname(os.path.abspath(cellacdc.__file__))
    expected = job['expected_package_dir']
    if not runner.same_path(actual, expected):
        raise _Rejected(
            runner.EXIT_PACKAGE_MISMATCH, 'PackageMismatchError',
            f'the worker imported cellacdc from "{actual}" but the parent '
            f'uses "{expected}"'
        )


def _test_hooks(
        job: dict, runner: ModuleType
    ) -> tuple[Callable | None, ModuleType | None]:
    """Reader factory and ops module of the job; both are test hooks,
    honoured only with ``ACDC_DATASTRUCT_TEST_HOOKS=1``."""
    factory_spec = job.get('reader_factory')
    ops_name = job.get('ops_module')
    if factory_spec is None and ops_name is None:
        return None, None
    if os.environ.get(runner.TEST_HOOKS_ENV) != '1':
        raise _Rejected(
            runner.EXIT_BAD_JOB, 'JobError',
            'the job uses test hooks (reader_factory/ops_module) but '
            f'{runner.TEST_HOOKS_ENV}=1 is not set'
        )
    try:
        factory = (
            None if factory_spec is None
            else runner.resolve_reference(factory_spec)
        )
        ops = None if ops_name is None else importlib.import_module(ops_name)
    except (ImportError, AttributeError, ValueError) as err:
        raise _Rejected(
            runner.EXIT_BAD_JOB, 'JobError', f'cannot load a test hook: {err}'
        )
    return factory, ops


def _event_writer(
        events: TextIO, runner: ModuleType
    ) -> Callable[[dict], None]:
    def emit(event: dict) -> None:
        events.write(
            f'{runner.EVENT_PREFIX}{json.dumps(event, ensure_ascii=True)}\n'
        )
        events.flush()
    return emit


def _write_json(
        path: str, data: Mapping, *, default: Callable | None = None
    ) -> None:
    """Write ``data`` atomically (temporary file + ``os.replace``)."""
    text = json.dumps(data, ensure_ascii=True, default=default)
    tmp_path = f'{path}.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as file:
        file.write(text)
    os.replace(tmp_path, path)


def _write_error(
        run_dir: str, runner: ModuleType, task: str | None, error_type: str,
        message: str, traceback_text: str = '',
        context: Mapping[str, object] | None = None
    ) -> None:
    report = {
        'protocol': runner.PROTOCOL,
        'task': task,
        'error_type': error_type,
        'message': message,
        'traceback': traceback_text,
        'context': dict(context or {}),
    }
    try:
        _write_json(
            os.path.join(run_dir, runner.ERROR_FILENAME), report, default=str
        )
    except OSError as err:
        _stderr(
            f'cannot write {runner.ERROR_FILENAME} into "{run_dir}": {err}'
        )


def _reject(
        run_dir: str, runner: ModuleType, task: str | None, err: _Rejected
    ) -> int:
    _stderr(f'{runner.WORKER_MODULE}: {err}')
    _write_error(run_dir, runner, task, err.error_type, str(err))
    return err.code


def _fail(
        run_dir: str, runner: ModuleType, task: str, err: Exception
    ) -> int:
    text = ''.join(
        traceback.format_exception(type(err), err, err.__traceback__)
    )
    _stderr(text.rstrip('\n'))
    context = getattr(err, 'context', None)
    _write_error(
        run_dir, runner, task, type(err).__name__, str(err), text,
        context if isinstance(context, Mapping) else None
    )
    return runner.EXIT_TASK_FAILED


def _stderr(text: str) -> None:
    sys.stderr.write(f'{text}\n')
    sys.stderr.flush()


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
