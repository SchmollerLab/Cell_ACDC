"""Parent side of the shared data-structure subprocess runner.

``run_task`` runs one task of ``cellacdc.acdc_datastruct.tasks`` in a child
process (Java/Bio-Formats never runs in the GUI process)::

    python -m cellacdc.acdc_datastruct._worker <run_dir>/job.json [--server]

Protocol (version ``PROTOCOL``):

- The parent creates a private run folder (``tempfile.mkdtemp``), writes
  ``job.json`` into it (ASCII JSON) and starts the child with ``cwd`` = run
  folder, ``stdin`` = DEVNULL and ``PYTHONPATH`` starting with the folder
  that contains the parent's ``cellacdc`` package.
- The child's stdout carries only event lines (``EVENT_PREFIX`` + a JSON
  object); everything else the child, its libraries or its own children
  write goes to stderr. Two threads read the pipes; events and output lines
  are dispatched on the thread that called ``run_task``.
- The child writes ``result.json`` (exit 0) or ``error.json`` (exit 1, 2 or
  3) into the run folder. Exit codes: 0 success, 1 the task raised, 2 the
  job was invalid, 3 the child imported a different cellacdc, 130
  interrupted.

Standard library only at module level, so a broken installation is reported
by the caller instead of failing at import time.
"""
from __future__ import annotations

import collections
import importlib
import json
import logging
import os
import queue
import shutil
import site
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Callable, Mapping

logger = logging.getLogger(__name__)

PROTOCOL = 1
WORKER_MODULE = 'cellacdc.acdc_datastruct._worker'
EVENT_PREFIX = '@@acdc-event@@ '
TEST_HOOKS_ENV = 'ACDC_DATASTRUCT_TEST_HOOKS'
OUTPUT_LINE_MAX = 4096
"""Characters; longer child output lines are truncated before ``on_output``
(tqdm ``\\r`` updates arrive as one long line)."""

SERVER_FLAG = '--server'
"""Same flag as ``cellacdc.SERVER_FLAG`` (read by ``cellacdc/__init__``)."""
JOB_FILENAME = 'job.json'
RESULT_FILENAME = 'result.json'
ERROR_FILENAME = 'error.json'

EXIT_OK = 0
EXIT_TASK_FAILED = 1
EXIT_BAD_JOB = 2
EXIT_PACKAGE_MISMATCH = 3
EXIT_INTERRUPTED = 130

PACKAGE_MISMATCH_HINT = (
    'the child imported a different cellacdc (editable install / PYTHONPATH)'
)
OUTPUT_TAIL_LINES = 200
POLL_INTERVAL_S = 0.2
TERMINATE_WAIT_S = 5.0
READER_JOIN_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class RunnerOptions:
    """How ``run_task`` starts the worker child.

    Attributes
    ----------
    python : str, optional
        Interpreter of the child (default ``sys.executable``).
    server_mode : bool
        Append ``--server`` to the child command line (no Qt in the child).
    timeout : float, optional
        Seconds; the child is killed on expiry and ``WorkerError`` raised.
    keep_run_dir : bool
        Keep the run folder (job, result, arrays) and log its path.
    temp_root : str, optional
        Parent folder of the run folders (default: the system temp folder).
    extra_env : mapping, optional
        Variables added to (or replacing) the child environment.
    """
    python: str | None = None
    server_mode: bool = True
    timeout: float | None = None
    keep_run_dir: bool = False
    temp_root: str | None = None
    extra_env: Mapping[str, str] | None = None


class WorkerError(RuntimeError):
    """A worker child failed.

    The task raised (exit 1, ``error_type``/``message``/``child_traceback``/
    ``context`` come from the child), the job was rejected (exit 2), the
    child imported a different cellacdc (exit 3, ``hint`` set), it crashed
    (``output_tail`` holds its last output lines), timed out or exited
    without a result. ``str(err)`` is
    ``'Worker task <task> failed (exit <rc>): <type>: <message>'`` followed
    by the child traceback (or the output tail) on the next lines.
    """
    def __init__(
            self, task: str, message: str, *, returncode: int | None = None,
            error_type: str | None = None, child_traceback: str = '',
            output_tail: str = '', context: Mapping[str, object] | None = None,
            hint: str | None = None
        ) -> None:
        self.task = task
        self.message = message
        self.returncode = returncode
        self.error_type = error_type
        self.child_traceback = child_traceback or ''
        self.output_tail = output_tail or ''
        self.context = dict(context or {})
        self.hint = hint
        super().__init__(self._describe())

    def _describe(self) -> str:
        head = f'Worker task {self.task} failed'
        if self.returncode is not None:
            head = f'{head} (exit {self.returncode})'
        reason = self.message
        if self.error_type:
            reason = f'{self.error_type}: {reason}'
        details = (self.child_traceback or self.output_tail).rstrip()
        text = f'{head}: {reason}'
        return f'{text}\n{details}' if details else text


def _package_dir() -> str:
    import cellacdc
    return os.path.dirname(os.path.abspath(cellacdc.__file__))


def package_root() -> str:
    """Folder that contains the parent's ``cellacdc`` package."""
    return os.path.dirname(_package_dir())


def same_path(a: str, b: str) -> bool:
    """Whether two paths name the same location (``realpath`` and
    ``normcase`` applied to both sides)."""
    return (
        os.path.normcase(os.path.realpath(a))
        == os.path.normcase(os.path.realpath(b))
    )


def _site_packages_dirs() -> list[str]:
    dirs = []
    get_site_packages = getattr(site, 'getsitepackages', None)
    if get_site_packages is not None:
        dirs.extend(get_site_packages())
    get_user_site = getattr(site, 'getusersitepackages', None)
    if get_user_site is not None:
        dirs.append(get_user_site())
    return dirs


def _is_site_packages(path: str) -> bool:
    name = os.path.basename(os.path.normpath(path)).lower()
    if name in ('site-packages', 'dist-packages'):
        return True
    return any(same_path(path, other) for other in _site_packages_dirs())


def child_env(
        base: Mapping[str, str] | None = None, *,
        extra: Mapping[str, str] | None = None
    ) -> dict[str, str]:
    """Environment of a worker child.

    A copy of ``base`` (default ``os.environ``) with ``package_root()``
    prepended to ``PYTHONPATH`` (so the child imports the parent's
    cellacdc), except when the root is a site-packages folder (prepending
    it could shadow the standard library), plus ``PYTHONIOENCODING=utf-8``,
    ``PYTHONUNBUFFERED=1`` and finally ``extra``.
    """
    env = dict(os.environ if base is None else base)
    root = package_root()
    if not _is_site_packages(root):
        old = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = f'{root}{os.pathsep}{old}' if old else root
    env['PYTHONIOENCODING'] = 'utf-8'
    env['PYTHONUNBUFFERED'] = '1'
    if extra:
        env.update(extra)
    return env


def build_command(job_path: str, options: RunnerOptions) -> list[str]:
    """Command line of the worker child for ``job_path``."""
    command = [options.python or sys.executable, '-m', WORKER_MODULE, job_path]
    if options.server_mode:
        command.append(SERVER_FLAG)
    return command


def resolve_reference(spec: str) -> object:
    """Import ``'package.module:attr'`` and return the attribute.

    Format of the test-hook references (``reader_factory``,
    ``ACDC_DATASTRUCT_READER_FACTORY``). Raises ``ValueError`` for a
    malformed spec, ``ImportError``/``AttributeError`` when it does not
    resolve.
    """
    module_name, sep, attr = spec.partition(':')
    if not (module_name and sep and attr):
        raise ValueError(f"expected 'module:attr', got {spec!r}")
    return getattr(importlib.import_module(module_name), attr)


def run_task(
        task: str, params: Mapping, *, options: RunnerOptions | None = None,
        reader_factory: str | None = None,
        on_event: Callable[[dict], None] | None = None,
        on_output: Callable[[str], None] | None = None,
        is_cancelled: Callable[[], bool] | None = None,
        collect: Callable[[dict, str], object] | None = None,
        ops_module: str | None = None
    ) -> dict | object:
    """Run ``task`` with ``params`` in a worker child and return its result.

    Parameters
    ----------
    task : str
        Name in the worker's task table.
    params : mapping
        JSON-serialisable task parameters.
    options : RunnerOptions, optional
        Interpreter, server mode, timeout, temp folder, environment.
    reader_factory : str, optional
        ``'module:attr'`` of a reader factory (test hook; the child honours
        it only with ``ACDC_DATASTRUCT_TEST_HOOKS=1``).
    on_event : callable, optional
        Called with each event dict the task emits.
    on_output : callable, optional
        Called with each other output line of the child (stdout lines that
        are not events and every stderr line), truncated to
        ``OUTPUT_LINE_MAX`` characters. Default: discarded.
    is_cancelled : callable, optional
        Polled about every 0.2 s; when it returns True the child is
        terminated and ``ConversionCancelled`` raised.
    collect : callable, optional
        ``collect(result, run_dir)`` runs before the run folder is removed
        (e.g. to load arrays the task saved there); its return value is
        returned instead of the result.
    ops_module : str, optional
        Dotted name of a test ops module used instead of the production
        tasks (test hook, same gate as ``reader_factory``).

    Every callback runs on the calling thread. Every event and output line
    the child produced is dispatched before this function returns or
    raises.

    Raises
    ------
    WorkerError
        The child failed, crashed, timed out or exited without a result.
    ConversionCancelled
        ``is_cancelled()`` returned True, or the child was interrupted.
    KeyboardInterrupt
        Re-raised after the child was stopped and the run folder removed.
    """
    options = RunnerOptions() if options is None else options
    if options.temp_root:
        os.makedirs(options.temp_root, exist_ok=True)
    run_dir = os.path.abspath(
        tempfile.mkdtemp(prefix='acdc_datastruct_', dir=options.temp_root)
    )
    try:
        job_path = os.path.join(run_dir, JOB_FILENAME)
        job = {
            'protocol': PROTOCOL,
            'task': task,
            'params': dict(params),
            'out_dir': run_dir,
            'expected_package_dir': os.path.normcase(
                os.path.realpath(_package_dir())
            ),
            'reader_factory': reader_factory,
            'ops_module': ops_module,
        }
        text = json.dumps(job, ensure_ascii=True)
        with open(job_path, 'w', encoding='utf-8') as file:
            file.write(text)
        supervisor = _Supervisor(
            task, options, on_event=on_event, on_output=on_output
        )
        returncode, stopped = supervisor.run(
            build_command(job_path, options), run_dir, is_cancelled
        )
        return _outcome(
            task, returncode, stopped, run_dir, options,
            supervisor.output_tail(), collect
        )
    finally:
        if options.keep_run_dir:
            logger.info('Worker run folder kept: %s', run_dir)
        else:
            shutil.rmtree(run_dir, ignore_errors=True)


class _Supervisor:
    """Starts the child, dispatches its lines and stops it if needed."""
    def __init__(
            self, task: str, options: RunnerOptions, *,
            on_event: Callable[[dict], None] | None,
            on_output: Callable[[str], None] | None
        ) -> None:
        self.task = task
        self.options = options
        self.on_event = on_event
        self.on_output = on_output
        self.lines: queue.Queue = queue.Queue()
        self.tail: collections.deque = collections.deque(
            maxlen=OUTPUT_TAIL_LINES
        )

    def run(
            self, command: list[str], run_dir: str,
            is_cancelled: Callable[[], bool] | None
        ) -> tuple[int, str | None]:
        """Run the child to the end; return ``(returncode, stopped)`` with
        ``stopped`` None (exited by itself), 'cancelled' or 'timeout'."""
        timeout = self.options.timeout
        deadline = None if timeout is None else time.monotonic() + timeout
        logger.debug(
            'Worker task %s: %r (cwd %s)', self.task, command, run_dir
        )
        proc = subprocess.Popen(
            command, cwd=run_dir, env=child_env(extra=self.options.extra_env),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        try:
            readers = [
                self._start_reader(proc.stdout, 'stdout'),
                self._start_reader(proc.stderr, 'stderr'),
            ]
            try:
                stopped = self._pump(proc, is_cancelled, deadline)
            except BaseException as err:
                self._stop(proc)
                # Ctrl-C: still deliver what the child reported; an error
                # raised by a callback: do not call the callbacks again.
                interrupted = isinstance(err, KeyboardInterrupt)
                self._finish(proc, readers, dispatch=interrupted)
                raise
            if stopped is not None:
                self._stop(proc)
            self._finish(proc, readers, dispatch=True)
            return proc.returncode, stopped
        finally:
            if proc.poll() is None:
                # Only reached when stopping was itself interrupted.
                proc.kill()
                proc.wait()

    def output_tail(self) -> str:
        return '\n'.join(self.tail)

    def _start_reader(self, stream, name: str) -> threading.Thread:
        reader = threading.Thread(
            target=self._read_lines, args=(stream, name),
            name=f'acdc-worker-{name}', daemon=True
        )
        reader.start()
        return reader

    def _read_lines(self, stream, name: str) -> None:
        """Reader thread: queue every line of a child pipe until EOF. Lines
        end at '\\n' only, so tqdm '\\r' updates stay in one line."""
        try:
            for raw in stream:
                text = raw.decode('utf-8', errors='replace')
                self.lines.put((name, text.rstrip('\r\n')))
        except (OSError, ValueError):
            # The pipe was closed under the reader.
            pass

    def _pump(
            self, proc: subprocess.Popen,
            is_cancelled: Callable[[], bool] | None, deadline: float | None
        ) -> str | None:
        while True:
            try:
                name, line = self.lines.get(timeout=POLL_INTERVAL_S)
            except queue.Empty:
                pass
            else:
                self._dispatch(name, line)
            if proc.poll() is not None:
                return None
            if is_cancelled is not None and is_cancelled():
                return 'cancelled'
            if deadline is not None and time.monotonic() >= deadline:
                return 'timeout'

    def _dispatch(self, name: str, line: str) -> None:
        if name == 'stdout' and line.startswith(EVENT_PREFIX):
            event = _decode_event(line)
            if event is not None:
                if self.on_event is not None:
                    self.on_event(event)
                return
        line = line[:OUTPUT_LINE_MAX]
        self.tail.append(line)
        if self.on_output is not None:
            self.on_output(line)

    @staticmethod
    def _stop(proc: subprocess.Popen) -> None:
        """Terminate the child (kill it after ``TERMINATE_WAIT_S``)."""
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
        except OSError:
            # It exited in the meantime.
            pass
        try:
            proc.wait(TERMINATE_WAIT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def _finish(
            self, proc: subprocess.Popen, readers: list[threading.Thread], *,
            dispatch: bool
        ) -> None:
        """After the child exited: join the readers, then dispatch (or
        drop) every queued line, before any result or error is read."""
        for reader in readers:
            reader.join(READER_JOIN_TIMEOUT_S)
        if not any(reader.is_alive() for reader in readers):
            # A grandchild that inherited the pipes can keep a reader
            # blocked; such pipes are left to the daemon thread.
            proc.stdout.close()
            proc.stderr.close()
        while True:
            try:
                name, line = self.lines.get_nowait()
            except queue.Empty:
                return
            if dispatch:
                self._dispatch(name, line)


def _decode_event(line: str) -> dict | None:
    try:
        event = json.loads(line[len(EVENT_PREFIX):])
    except ValueError:
        return None
    return event if isinstance(event, dict) else None


def _read_json(path: str) -> dict | None:
    try:
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _outcome(
        task: str, returncode: int, stopped: str | None, run_dir: str,
        options: RunnerOptions, output_tail: str,
        collect: Callable[[dict, str], object] | None
    ) -> dict | object:
    """Result of a finished child, or the matching exception.

    Cancellation is decided by the parent's own flag (``stopped``), never by
    the child's return code: on Windows ``terminate()`` gives 1.
    """
    if stopped == 'cancelled':
        from . import errors
        raise errors.ConversionCancelled(f'Worker task {task} was cancelled.')
    if stopped == 'timeout':
        raise WorkerError(
            task, f'timed out after {options.timeout:g} s',
            returncode=returncode, output_tail=output_tail
        )
    if returncode == EXIT_OK:
        report = _read_json(os.path.join(run_dir, RESULT_FILENAME))
        if report is None or 'result' not in report:
            raise WorkerError(
                task, 'no result', returncode=returncode,
                output_tail=output_tail
            )
        if collect is None:
            return report['result']
        return collect(report['result'], run_dir)
    if returncode == EXIT_INTERRUPTED:
        from . import errors
        raise errors.ConversionCancelled(
            f'Worker task {task} was interrupted (exit {returncode}).'
        )
    report = _read_json(os.path.join(run_dir, ERROR_FILENAME)) or {}
    context = report.get('context')
    raise WorkerError(
        task,
        report.get('message')
        or f'worker exited with code {returncode} before reporting',
        returncode=returncode,
        error_type=report.get('error_type'),
        child_traceback=report.get('traceback') or '',
        output_tail=output_tail,
        context=context if isinstance(context, dict) else None,
        hint=(
            PACKAGE_MISMATCH_HINT if returncode == EXIT_PACKAGE_MISMATCH
            else None
        )
    )
