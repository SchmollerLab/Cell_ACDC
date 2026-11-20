"""Test-only worker ops for ``test_runner.py``.

The worker loads this module only through the job's ``ops_module`` key,
which it honours only with ``ACDC_DATASTRUCT_TEST_HOOKS=1``, so production
jobs can never reach these ops. Same interface as the production
``cellacdc.acdc_datastruct.tasks``: ``TASKS`` and ``TaskContext``.

Ops (``params`` keys in brackets):

- ``_echo`` [events, marker]: emits ``events`` log events, writes
  ``marker`` into ``out_dir/marker.txt`` and returns what the child sees
  (params, cellacdc file, server mode, cwd, environment, reader factory).
- ``_raise`` [events, message, context, keyboard]: emits ``events`` log
  events, then raises ``OpError(message, context=context)`` chained to a
  ``ValueError`` (or ``KeyboardInterrupt`` with ``keyboard``).
- ``_exit`` [events, stderr, code]: emits events, writes ``stderr`` to
  stderr and ends the process with ``os._exit(code)`` (a crash: no report).
  Subprocess only.
- ``_sleep`` [seconds]: emits a ``started`` event, then sleeps.
- ``_print_noise`` [lines, long_line, grandchild]: writes noise to every
  channel a task can reach (``print``, fd 1, stderr with tqdm-style ``\\r``
  updates, a ``long_line``-character line, a grandchild process) around
  ``lines`` log events.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Mapping

OPS_MODULE = 'tests.datastruct.runner_test_ops'
ENV_KEYS = (
    'PYTHONIOENCODING', 'PYTHONUNBUFFERED', 'ACDC_DATASTRUCT_TEST_HOOKS',
)


@dataclass
class TaskContext:
    """Same fields as ``cellacdc.acdc_datastruct.tasks.TaskContext``."""
    out_dir: str
    emit: Callable[[dict], None]
    reader_factory: Callable | None = None


class OpError(Exception):
    """Like ``tasks.TaskError``: carries a ``context`` mapping."""
    def __init__(
            self, message: str, *, context: Mapping[str, object] | None = None
        ) -> None:
        super().__init__(message)
        self.context = dict(context or {})


def _emit_log(ctx: TaskContext, index: int, prefix: str = 'event') -> None:
    ctx.emit({
        'kind': 'log', 'level': 'INFO', 'message': f'{prefix} {index}',
        'index': index,
    })


def _emit_logs(ctx: TaskContext, count: int) -> None:
    for index in range(count):
        _emit_log(ctx, index)


def _echo(params: dict, ctx: TaskContext) -> dict:
    import cellacdc
    _emit_logs(ctx, params.get('events', 0))
    if 'marker' in params:
        marker_path = os.path.join(ctx.out_dir, 'marker.txt')
        with open(marker_path, 'w', encoding='utf-8') as file:
            file.write(params['marker'])
    factory = ctx.reader_factory
    return {
        'params': params,
        'cellacdc_file': cellacdc.__file__,
        'running_as_server': cellacdc.RUNNING_AS_SERVER,
        'argv': sys.argv[1:],
        'cwd': os.getcwd(),
        'out_dir': ctx.out_dir,
        'pid': os.getpid(),
        'env': {key: os.environ.get(key) for key in ENV_KEYS},
        'pythonpath': os.environ.get('PYTHONPATH', ''),
        'reader_factory': (
            None if factory is None
            else f'{factory.__module__}:{factory.__qualname__}'
        ),
    }


def _raise(params: dict, ctx: TaskContext) -> dict:
    _emit_logs(ctx, params.get('events', 0))
    if params.get('keyboard'):
        raise KeyboardInterrupt
    try:
        raise ValueError('original cause')
    except ValueError as err:
        raise OpError(
            params.get('message', 'boom'), context=params.get('context')
        ) from err


def _exit(params: dict, ctx: TaskContext) -> dict:
    _emit_logs(ctx, params.get('events', 0))
    sys.stderr.write(params.get('stderr', ''))
    sys.stderr.flush()
    os._exit(params.get('code', 7))


def _sleep(params: dict, ctx: TaskContext) -> dict:
    ctx.emit({'kind': 'started', 'pid': os.getpid(), 'out_dir': ctx.out_dir})
    seconds = params.get('seconds', 60)
    time.sleep(seconds)
    return {'slept': seconds}


def _print_noise(params: dict, ctx: TaskContext) -> dict:
    lines = params.get('lines', 3)
    for index in range(lines):
        print(f'NOISE print {index}')
        sys.stdout.flush()
        os.write(1, f'NOISE fd1 {index}\n'.encode())
        sys.stderr.write(
            ''.join(f'progress {p}%\r' for p in range(0, 101, 25)) + '\n'
        )
        sys.stderr.flush()
        _emit_log(ctx, index, prefix='noisy')
    sys.stderr.write('L' * params.get('long_line', 0) + '\n')
    sys.stderr.flush()
    if params.get('grandchild'):
        subprocess.run(
            [sys.executable, '-c', "print('NOISE grandchild')"],
            stdin=subprocess.DEVNULL, check=True
        )
    return {'lines': lines}


TASKS = {
    '_echo': _echo,
    '_raise': _raise,
    '_exit': _exit,
    '_sleep': _sleep,
    '_print_noise': _print_noise,
}
