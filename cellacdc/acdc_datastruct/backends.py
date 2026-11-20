"""Where the tasks that open raw files run.

- ``SubprocessBackend``: every call runs one task in a worker child through
  ``runner.run_task`` (production: the GUI and the command line). The
  parent process never opens a raw microscopy file.
- ``InProcessBackend``: calls ``tasks.TASKS`` directly (tests and the
  golden comparison). Params, events and results still go through JSON and
  arrays through ``.npy`` files, exactly as with a child, so both backends
  give the same results.

``default_backend`` builds the production backend of the GUI or of the
command line.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import tempfile
import traceback
from typing import Callable, Mapping, Protocol, Sequence

import numpy as np

from . import errors, metadata, planning, runner

logger = logging.getLogger(__name__)

READER_FACTORY_ENV = 'ACDC_DATASTRUCT_READER_FACTORY'
"""``module:attr`` of a reader factory used by ``default_backend`` (test hook,
honoured only with ``ACDC_DATASTRUCT_TEST_HOOKS=1``)."""


class ConvertBackend(Protocol):
    """What ``convert`` needs from a backend."""
    def init_reader(self, path: str) -> None:
        """Open and close a reader on ``path``."""
        ...

    def read_metadata(
            self, paths: Sequence[str]
        ) -> list[metadata.RawFileMetadata]:
        """Raw metadata of ``paths`` (one task for all of them).

        Raises ``MetadataReadError`` (``filename`` = base name of the file
        that failed, ``details`` = traceback)."""
        ...

    def read_sample(
            self, path: str, *, size_c: int, size_t: int, size_z: int,
            lazy_load: bool
        ) -> list[np.ndarray]:
        """Sample arrays of the metadata dialog, one per channel."""
        ...

    def check_positions(self, jobs: Sequence[planning.PositionJob]) -> None:
        """Check ``jobs`` against their raw files without writing anything
        (one task).

        Raises ``ValueError`` (the message of the check, naming the file)
        when a job does not fit one of its files."""
        ...

    def save_positions(
            self, jobs: Sequence[planning.PositionJob],
            texts: Mapping[str, str], *, on_event: Callable[[dict], None]
        ) -> list[dict]:
        """Write ``jobs`` (one task); events go to ``on_event``. Returns the
        task's ``written`` list."""
        ...


def _sample_params(
        path: str, size_c: int, size_t: int, size_z: int, lazy_load: bool
    ) -> dict:
    return {
        'path': path, 'size_c': size_c, 'size_t': size_t, 'size_z': size_z,
        'lazy_load': lazy_load,
    }


def _jobs_params(jobs: Sequence[planning.PositionJob]) -> dict:
    return {'jobs': [job.to_dict() for job in jobs]}


def _save_params(
        jobs: Sequence[planning.PositionJob], texts: Mapping[str, str]
    ) -> dict:
    return {**_jobs_params(jobs), 'texts': dict(texts)}


def _load_arrays(result: dict, run_dir: str) -> list[np.ndarray]:
    return [np.load(os.path.join(run_dir, name)) for name in result['arrays']]


def _raw_metadata(result: dict) -> list[metadata.RawFileMetadata]:
    return [
        metadata.RawFileMetadata.from_dict(raw) for raw in result['metadata']
    ]


def _metadata_read_error(
        path: str, reason: str, details: str
    ) -> errors.MetadataReadError:
    filename = os.path.basename(path)
    return errors.MetadataReadError(
        f'Error while reading the metadata of "{filename}": {reason}',
        filename=filename, details=details
    )


class SubprocessBackend:
    """Runs every task in a worker child (``runner.run_task``).

    Parameters
    ----------
    options : RunnerOptions, optional
        Interpreter, server mode, timeout, temp folder, environment.
    reader_factory : str, optional
        ``'module:attr'`` of a reader factory (test hook; the child honours
        it only with ``ACDC_DATASTRUCT_TEST_HOOKS=1`` in its environment).
    on_output : callable, optional
        Called with every output line of the children (default: dropped).
    is_cancelled : callable, optional
        Polled by the runner; True terminates the child and raises
        ``ConversionCancelled``.
    """
    def __init__(
            self, *, options: runner.RunnerOptions | None = None,
            reader_factory: str | None = None,
            on_output: Callable[[str], None] | None = None,
            is_cancelled: Callable[[], bool] | None = None
        ) -> None:
        self.options = options
        self.reader_factory = reader_factory
        self.on_output = on_output
        self.is_cancelled = is_cancelled

    def _run(
            self, task: str, params: Mapping, *,
            on_event: Callable[[dict], None] | None = None,
            collect: Callable[[dict, str], object] | None = None
        ):
        return runner.run_task(
            task, params, options=self.options,
            reader_factory=self.reader_factory, on_event=on_event,
            on_output=self.on_output, is_cancelled=self.is_cancelled,
            collect=collect
        )

    def init_reader(self, path: str) -> None:
        self._run('init_reader', {'path': path})

    def read_metadata(
            self, paths: Sequence[str]
        ) -> list[metadata.RawFileMetadata]:
        try:
            result = self._run('read_metadata', {'paths': list(paths)})
        except runner.WorkerError as err:
            path = err.context.get('path')
            if path is None:
                raise
            raise _metadata_read_error(
                str(path), err.message, err.child_traceback or err.output_tail
            ) from err
        return _raw_metadata(result)

    def read_sample(
            self, path: str, *, size_c: int, size_t: int, size_z: int,
            lazy_load: bool
        ) -> list[np.ndarray]:
        return self._run(
            'read_sample',
            _sample_params(path, size_c, size_t, size_z, lazy_load),
            collect=_load_arrays
        )

    def check_positions(self, jobs: Sequence[planning.PositionJob]) -> None:
        """The child's ``ValueError`` (a job does not fit its files) is
        raised again as a ``ValueError`` with the same message."""
        try:
            self._run('check_positions', _jobs_params(jobs))
        except runner.WorkerError as err:
            if err.error_type != 'ValueError':
                raise
            raise ValueError(err.message) from err

    def save_positions(
            self, jobs: Sequence[planning.PositionJob],
            texts: Mapping[str, str], *, on_event: Callable[[dict], None]
        ) -> list[dict]:
        result = self._run(
            'save_positions', _save_params(jobs, texts), on_event=on_event
        )
        return result['written']


def _json_round_trip(value):
    """What the subprocess transport does to ``value`` (strict JSON: a
    value that is not JSON-serialisable raises ``TypeError``)."""
    return json.loads(json.dumps(value, ensure_ascii=True))


class InProcessBackend:
    """Calls ``tasks.TASKS`` in this process with a ``TaskContext`` whose
    ``emit`` forwards to ``on_event``. Used by tests and by the golden
    comparison; never by the GUI or command line production code.

    Params, events and results go through a JSON round trip and the arrays
    of ``read_sample`` through a temporary folder, as with a child. Errors
    are the original exceptions, except ``read_metadata`` (raises
    ``MetadataReadError`` like ``SubprocessBackend``).

    Parameters
    ----------
    reader_factory : callable or str, optional
        Replaces ``bioio.BioImage`` (test hook): the factory, or its
        ``'module:attr'``. None reads the files with bioio.
    """
    def __init__(
            self, *, reader_factory: Callable | str | None = None
        ) -> None:
        if isinstance(reader_factory, str):
            reader_factory = runner.resolve_reference(reader_factory)
        self.reader_factory = reader_factory

    def _run(
            self, task: str, params: Mapping, *,
            on_event: Callable[[dict], None] | None = None,
            collect: Callable[[dict, str], object] | None = None
        ):
        from . import tasks

        def emit(event: dict) -> None:
            event = _json_round_trip(event)
            if on_event is not None:
                on_event(event)

        params = _json_round_trip(dict(params))
        run_dir = tempfile.mkdtemp(prefix='acdc_datastruct_')
        try:
            ctx = tasks.TaskContext(
                out_dir=run_dir, emit=emit,
                reader_factory=self.reader_factory
            )
            result = _json_round_trip(tasks.TASKS[task](params, ctx))
            return result if collect is None else collect(result, run_dir)
        finally:
            shutil.rmtree(run_dir, ignore_errors=True)

    def init_reader(self, path: str) -> None:
        self._run('init_reader', {'path': path})

    def read_metadata(
            self, paths: Sequence[str]
        ) -> list[metadata.RawFileMetadata]:
        from . import tasks
        try:
            result = self._run('read_metadata', {'paths': list(paths)})
        except tasks.TaskError as err:
            path = err.context.get('path')
            if path is None:
                raise
            raise _metadata_read_error(
                str(path), str(err), traceback.format_exc()
            ) from err
        return _raw_metadata(result)

    def read_sample(
            self, path: str, *, size_c: int, size_t: int, size_z: int,
            lazy_load: bool
        ) -> list[np.ndarray]:
        return self._run(
            'read_sample',
            _sample_params(path, size_c, size_t, size_z, lazy_load),
            collect=_load_arrays
        )

    def check_positions(self, jobs: Sequence[planning.PositionJob]) -> None:
        self._run('check_positions', _jobs_params(jobs))

    def save_positions(
            self, jobs: Sequence[planning.PositionJob],
            texts: Mapping[str, str], *, on_event: Callable[[dict], None]
        ) -> list[dict]:
        result = self._run(
            'save_positions', _save_params(jobs, texts), on_event=on_event
        )
        return result['written']


def _write_to_console(line: str) -> None:
    """Child output of GUI conversions goes to the console, as the output of
    the legacy children did (nothing without a console, e.g. pythonw)."""
    stream = sys.__stderr__
    if stream is None:
        return
    try:
        stream.write(f'{line}\n')
        stream.flush()
    except (OSError, ValueError):
        # Closed or non-encodable console: the output is only informative.
        pass


def _log_output(line: str) -> None:
    logger.debug('%s', line)


def _test_reader_factory() -> str | None:
    if os.environ.get(runner.TEST_HOOKS_ENV) != '1':
        return None
    return os.environ.get(READER_FACTORY_ENV) or None


def default_backend(
        *, for_gui: bool, on_output: Callable[[str], None] | None = None,
        options: runner.RunnerOptions | None = None
    ) -> SubprocessBackend:
    """Production backend of the GUI (``for_gui=True``) or of the command
    line.

    Child output goes to ``on_output``; by default the GUI writes it to the
    console (``sys.__stderr__``, skipped when there is none) and the command
    line logs it at DEBUG. ``options`` sets up the runner (e.g.
    ``keep_run_dir`` for ``--keep-temp``). The reader factory is read from
    ``ACDC_DATASTRUCT_READER_FACTORY`` only when
    ``ACDC_DATASTRUCT_TEST_HOOKS=1`` (test hook), else bioio is used.
    """
    if on_output is None:
        on_output = _write_to_console if for_gui else _log_output
    return SubprocessBackend(
        options=options, reader_factory=_test_reader_factory(),
        on_output=on_output
    )
