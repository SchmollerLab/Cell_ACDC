"""Decide where a segmentation model runs and create it.

Shared by every GUI caller that segments with a model:

* the segmentation module (`segm.segmWin`),
* the main GUI (`gui.guiWin.repeatSegm`, `repeatSegmVideo`, segmentation
  for lost IDs via `workers.SegForLostIDsWorker`, and YeastMate budding
  prediction in `autoAssignBud_YeastMate`).

Execution modes (`ModelExecution.backend_type`):

* ``'local'``: the model runs in its isolated environment ``_acdc_<model>``
  as a worker process (see `service/worker.py`). The returned model is a
  `ModelClient`, which has the same methods as `acdcSegment.Model`.
* ``'in_process'``: the model is imported and run in the Cell-ACDC process.

Errors in the isolated mode are raised; there is no silent fallback to the
in-process mode.

GPU checks in the client must pass ``execution.env_python`` to
``myutils.check_gpu_available`` after ``prepare()``. This inspects the selected
model environment and targets GPU installation instructions to it. In-process
executions leave ``env_python`` as None.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple
import dataclasses
import hashlib
import itertools
import json

from cellacdc import myutils
from cellacdc.service import prompts
from cellacdc.service.client import LocalBackend
from cellacdc.service.environment import (
    ensure_required_env, get_or_create_model_env, model_installed_in_main_env,
)

# False runs every model in-process (previous behaviour)
USE_MODEL_ENV = True

# Unused workers kept running per model, ready to be re-initialized
MAX_IDLE_WORKERS = 1
# Orders releases (time.monotonic() can tie on Windows)
_release_counter = itertools.count(1)

ISOLATED_ENV_CHOICE = 'Use isolated model environment (Recommended)'
MAIN_ENV_CHOICE = 'Use existing main environment'


@dataclasses.dataclass(eq=False)
class _Worker:
    """A running worker process and the parameters its model was created with."""
    backend: LocalBackend
    model: Any
    key: str
    users: int = 0
    released_at: int = 0


def _kwargs_key(init_kwargs: Dict[str, Any]) -> str:
    """Fingerprint of init kwargs, including array and DataFrame contents."""
    import numpy as np

    digest = hashlib.sha256()
    for name in sorted(init_kwargs):
        value = init_kwargs[name]
        digest.update(name.encode())
        if isinstance(value, np.ndarray):
            digest.update(f'{value.dtype}{value.shape}'.encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        elif hasattr(value, 'to_json') and hasattr(value, 'columns'):
            digest.update(value.to_json(orient='split').encode())
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str).encode())
            digest.update(_path_stamp(value))
    return digest.hexdigest()


def _path_stamp(value: Any) -> bytes:
    """Modification time of a file/folder path value.

    Existing paths contribute their mtime, so editing the file changes the
    key. Path-like values whose mtime cannot be read contribute a unique
    token, so the key never matches and the model is always re-initialized.
    """
    import os
    import uuid

    if not isinstance(value, (str, os.PathLike)):
        return b''
    path = os.fspath(value)
    if not isinstance(path, str) or not path:
        return b''
    try:
        return f'|mtime={os.stat(path).st_mtime_ns}'.encode()
    except OSError:
        pass
    if os.sep in path or '/' in path:
        return f'|unavailable={uuid.uuid4().hex}'.encode()
    return b''


def _call_directly(func, *args, **kwargs):
    return func(*args, **kwargs)


def run_in_background(parent, func, *args, log_func=None, **kwargs):
    """Run `func` in a QThread while keeping the GUI responsive.

    Must be called from the GUI thread. If `func` accepts `logger_func` or
    `prompt_func`, they are passed so that logs and prompts are shown safely
    in the GUI. Exceptions are re-raised in the calling thread.
    """
    import inspect
    from qtpy.QtCore import QEventLoop, QObject, QThread, Signal

    class _Task(QObject):
        log = Signal(str)

    class _Thread(QThread):
        result = None
        error = None

        def run(thread):
            try:
                thread.result = func(*args, **kwargs)
            except BaseException as err:
                thread.error = err

    task = _Task()
    if log_func is not None:
        task.log.connect(log_func)
    params = inspect.signature(func).parameters
    if 'logger_func' in params:
        kwargs['logger_func'] = task.log.emit
    if 'prompt_func' in params:
        kwargs['prompt_func'] = prompts.QtHandler(parent).ask

    loop = QEventLoop(parent)
    thread = _Thread(parent)
    thread.finished.connect(loop.quit)
    thread.start()
    loop.exec_()
    thread.wait()
    if thread.error is not None:
        raise thread.error
    return thread.result


class ModelExecution:
    """Execution environment and running instances of one model.

    Usage::

        execution = ModelExecution.for_qt(model_name, parent)
        spec = execution.prepare()   # None if the user cancelled
        init_params, segment_params, url = spec
        model = execution.init_model(init_kwargs, posData)  # None if cancelled
        labels = model.segment(image, **segment_kwargs)
        model = execution.init_model(other_kwargs, posData, replace=model)
        execution.release(model)     # worker stays running for reuse
        execution.stop()             # stops all workers of this model

    `prepare` decides the environment once; later calls return the cached
    parameter spec without asking again. In isolated mode, workers are kept
    running and reused by `init_model` (shared when the parameters are the
    same, re-initialized with `POST /init` otherwise), so the model server is
    not restarted for every segmentation. Different parameters used at the
    same time run in separate workers.

    Parameters
    ----------
    model_name : str
        Folder name in `cellacdc/models`.
    run_in_background : callable, optional
        `run_in_background(func, *args)` used for slow steps (environment
        creation, worker startup). Default: call directly.
    prompt_func : callable, optional
        Asks the user (see `service/prompts.py`). Without it, the isolated
        environment is used and missing required variables cancel `init_model`.
    log_func : callable, optional
        Receives progress messages.
    use_model_env : bool, optional
        Defaults to the module-level `USE_MODEL_ENV`.
    """

    def __init__(
            self,
            model_name: str,
            run_in_background: Optional[Callable] = None,
            prompt_func: Optional[Callable] = None,
            log_func: Callable = print,
            use_model_env: Optional[bool] = None,
        ):
        self.model_name = model_name
        self.run_in_background = run_in_background or _call_directly
        self.prompt_func = prompt_func
        self.log = log_func
        self.use_model_env = (
            USE_MODEL_ENV if use_model_env is None else use_model_env
        )
        self.backend_type: Optional[str] = None
        self.env_python: Optional[str] = None
        self.acdcSegment = None
        self.spec: Optional[Tuple[list, list, Any]] = None
        self._workers: List[_Worker] = []
        self._last_started: Optional[_Worker] = None

    @classmethod
    def for_qt(cls, model_name: str, parent, log_func: Optional[Callable] = None):
        """Create an execution that shows prompts and progress in `parent`."""
        if log_func is None:
            log_func = parent.logger.info
        return cls(
            model_name,
            run_in_background=lambda func, *args, **kwargs: run_in_background(
                parent, func, *args, log_func=log_func, **kwargs
            ),
            prompt_func=lambda prompt: prompts.QtHandler(parent).ask(prompt),
            log_func=log_func,
        )

    @property
    def backend(self) -> Optional[LocalBackend]:
        """Backend of the most recently started worker (None in-process)."""
        worker = self._last_started
        if worker is None or worker not in self._workers:
            return None
        return worker.backend

    @property
    def backend_url(self) -> Optional[str]:
        backend = self.backend
        return backend.url if backend is not None else None

    def _ask_environment(self) -> Optional[str]:
        if self.prompt_func is None:
            return ISOLATED_ENV_CHOICE
        import sys
        return self.prompt_func(prompts.choice(
            'Choose model environment',
            f'Compatible packages for {self.model_name} were found in the '
            f'main Cell-ACDC environment:\n{sys.executable}\n\n'
            'An isolated model environment keeps dependencies separate and '
            'reduces version conflicts. It may require extra downloads and '
            'disk space; an existing isolated environment will be reused.\n\n'
            'Using the main environment avoids a separate installation, '
            'but model dependencies share versions with Cell-ACDC. '
            'Package metadata was checked; runtime compatibility is not '
            'guaranteed.',
            [ISOLATED_ENV_CHOICE, MAIN_ENV_CHOICE],
            default=ISOLATED_ENV_CHOICE,
        ))

    def prepare(self) -> Optional[Tuple[list, list, Any]]:
        """Choose the environment and return `(init_params, segment_params, url)`.

        Returns None if the user cancelled (nothing is cached then, so the
        next call asks again).
        """
        if self.spec is not None:
            return self.spec

        backend_type = 'local' if self.use_model_env else 'in_process'
        if self.use_model_env and self.run_in_background(
            model_installed_in_main_env, self.model_name
        ):
            answer = self._ask_environment()
            if answer is None:
                return None
            if answer == MAIN_ENV_CHOICE:
                backend_type = 'in_process'

        if backend_type == 'local':
            self.log(f'Resolving environment for {self.model_name}...')
            env_python = self.run_in_background(
                get_or_create_model_env, self.model_name
            )
            if env_python is None:
                raise RuntimeError(
                    f'No model environment found for {self.model_name}'
                )
            spec = self.run_in_background(
                myutils.getModelArgSpecFromEnv, self.model_name, env_python
            )
            self.env_python = env_python
        else:
            self.log(f'Importing {self.model_name}...')
            self.acdcSegment = myutils.import_segment_module(self.model_name)
            init_params, segment_params = myutils.getModelArgSpec(
                self.acdcSegment
            )
            url = getattr(self.acdcSegment, 'url_help', lambda: None)()
            spec = (init_params, segment_params, url)

        self.backend_type = backend_type
        self.spec = tuple(spec)
        return self.spec

    def init_model(self, init_kwargs: Dict[str, Any], posData=None, replace=None):
        """Return a model instance; returns None if the user cancelled.

        In isolated mode, data the model needs from the client (e.g. an
        auxiliary segmentation) is loaded here from `posData` and sent to the
        worker. Running workers are reused instead of restarted:

        1. a worker that already has the same parameters (and data) is
           shared, so nothing is reloaded;
        2. otherwise an unused worker (released, or `replace`) receives the
           new parameters with `POST /init` (the process and its imports are
           kept, only the model is re-created);
        3. otherwise a new worker is started.

        Every returned instance must be given back with `release` (or by
        passing it as `replace` to the next call) once it is not used anymore.

        Parameters
        ----------
        replace : model, optional
            Instance (from a previous call) that the caller replaces with the
            new one. It is released after the new instance is ready, and its
            worker may be re-initialized for it. If this fails, `replace`
            stays usable.
        """
        if self.backend_type is None:
            raise RuntimeError(
                f'{self.model_name}: call prepare() before init_model().'
            )
        if not ensure_required_env(
            self.model_name, init_kwargs, prompt_func=self.prompt_func
        ):
            self.log(
                f'{self.model_name} cannot start without the required '
                'environment variables.'
            )
            return None

        if self.backend_type == 'in_process':
            return myutils.init_segm_model(
                self.acdcSegment, posData, init_kwargs
            )

        init_kwargs = dict(init_kwargs)
        segm_data = myutils.resolve_segm_model_init_data(posData, init_kwargs)
        if segm_data is not None:
            init_kwargs['segm_data'] = segm_data
        key = _kwargs_key(init_kwargs)

        self._discard_dead_workers()
        worker = self._find_worker(key=key)
        if worker is None:
            worker = self._find_worker(idle=True, replace=replace)
            if worker is not None:
                self.log(f'Re-initializing {self.model_name} model...')
                # The key is updated only if the worker accepted the kwargs
                self.run_in_background(worker.backend.reinitialize, init_kwargs)
                worker.key = key
            else:
                worker = self._start_worker(init_kwargs, key)
        else:
            self.log(f'Reusing running {self.model_name} model.')
        worker.users += 1
        if replace is not None:
            self.release(replace)
        return worker.model

    def _start_worker(self, init_kwargs, key) -> '_Worker':
        self.log(
            f'Starting model server for {self.model_name} in '
            f'{self.env_python}...'
        )
        backend = LocalBackend(
            model_name=self.model_name,
            env_python=self.env_python,
            init_model_kwargs=init_kwargs,
        )
        try:
            model = self.run_in_background(backend.start)
        except Exception:
            backend.stop()
            raise
        worker = _Worker(backend=backend, model=model, key=key)
        self._workers.append(worker)
        self._last_started = worker
        return worker

    def _find_worker(self, key=None, idle=False, replace=None):
        if key is not None:
            for worker in self._workers:
                if worker.key == key:
                    return worker
            return None
        # Prefer the instance being replaced, then the most recently released
        for worker in self._workers:
            if replace is not None and worker.model is replace:
                if worker.users <= 1:
                    return worker
        idle_workers = [w for w in self._workers if w.users == 0]
        if not idle_workers:
            return None
        return max(idle_workers, key=lambda w: w.released_at)

    def _discard_dead_workers(self):
        for worker in list(self._workers):
            if not worker.backend.is_running():
                self.log(f'{self.model_name} worker is not running anymore.')
                self._workers.remove(worker)
                self._stop_backend(worker.backend)

    def release(self, model) -> None:
        """Give back an instance from `init_model` (no-op in-process).

        The worker keeps running so it can be reused; at most
        `MAX_IDLE_WORKERS` unused workers are kept per model (they hold
        memory, also on the GPU).
        """
        for worker in self._workers:
            if worker.model is model:
                worker.users = max(worker.users - 1, 0)
                if worker.users == 0:
                    worker.released_at = next(_release_counter)
                break
        else:
            return
        idle_workers = sorted(
            (w for w in self._workers if w.users == 0),
            key=lambda w: w.released_at,
        )
        for worker in idle_workers[:-MAX_IDLE_WORKERS or None]:
            self._workers.remove(worker)
            self._stop_backend(worker.backend)

    def stop(self) -> None:
        """Stop all workers started by this execution."""
        workers, self._workers = self._workers, []
        self._last_started = None
        for worker in workers:
            self._stop_backend(worker.backend)

    def _stop_backend(self, backend) -> None:
        try:
            backend.stop()
        except Exception as err:
            self.log(f'Error stopping {self.model_name} worker: {err}')
