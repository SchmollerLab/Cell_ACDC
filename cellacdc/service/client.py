import collections
import gc
import io
import json
import os
import socket
import subprocess
import sys
import threading
import time
from typing import Dict, Any, Optional

import numpy as np
import requests

from cellacdc.service.protocol import PROTOCOL_VERSION, headers as _protocol_headers
from cellacdc.service.schema import InfoResponse, VersionResponse, parse_response
from cellacdc._model_startup import startup_timer

_LOCAL_ONLY_KWARGS = ("posData", "signals", "logger_func", "signals_tqdm")

def _find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        s.listen(1)
        return s.getsockname()[1]

def _cast_value(value: Any):
    if isinstance(value, (int, float, str, bool, type(None))):
        return value
    if isinstance(value, (list, tuple)):
        return type(value)(_cast_value(v) for v in value)
    if isinstance(value, dict):
        return {k: _cast_value(v) for k, v in value.items()}
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)

def _sanitize_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {}
    for k, v in kwargs.items():
        if k in _LOCAL_ONLY_KWARGS:
            continue
        # Check JSON serializability
        try:
            v = _cast_value(v)
            k = _cast_value(k)
            json.dumps(v)
            sanitized[k] = v
        except Exception:
            print(f"Failed to serialize key={k}, value={v}")
            print(f"Type of key={type(k)}, type of value={type(v)}")
    return sanitized

def _attachment(name: str, content: bytes):
    """A multipart file part for `requests` (`files={part: ...}`)."""
    return (name, content, "application/octet-stream")

def _encode_segment_kwargs(kwargs: Dict[str, Any]) -> bytes:
    """Encode segment kwargs as one attachment (see `payload.encode`),
    preserving arrays, DataFrames and files on disk.

    Path-like values are passed to `payload.encode` unchanged: as strings
    (`_sanitize_kwargs`), relative paths without a folder such as
    `Path('weights.pt')` would not be uploaded (see `payload.is_file_path`).
    """
    from cellacdc.service import payload

    data_kwargs = {
        key: value for key, value in kwargs.items()
        if key not in _LOCAL_ONLY_KWARGS and (
            isinstance(value, (np.ndarray, os.PathLike))
            or payload._is_dataframe(value)
        )
    }
    plain = _sanitize_kwargs({
        key: value for key, value in kwargs.items() if key not in data_kwargs
    })
    plain.update(data_kwargs)
    return payload.encode(plain)

def _segment_files(image: np.ndarray, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Multipart file parts of a segmentation request: the image as `.npy`
    and the encoded kwargs (see `worker.segment`)."""
    buf = io.BytesIO()
    np.save(buf, image)
    return {
        "image": _attachment("image.npy", buf.getvalue()),
        "params": _attachment("params.bin", _encode_segment_kwargs(kwargs)),
    }


class WorkerVersionError(RuntimeError):
    """The worker uses another protocol version than this client
    (see `protocol.py`)."""

    def __init__(
            self, base_url: str, worker_protocol: int,
            worker_version: Optional[str] = None,
        ):
        import cellacdc

        self.base_url = base_url
        self.worker_protocol = worker_protocol
        self.worker_version = worker_version
        worker = (
            f'Cell-ACDC {worker_version} (protocol {worker_protocol})'
            if worker_version is not None
            else f'an older Cell-ACDC (protocol {worker_protocol})'
        )
        super().__init__(
            f'The model worker at {base_url} runs {worker}, but this '
            f'Cell-ACDC ({cellacdc.__version__}) uses protocol '
            f'{PROTOCOL_VERSION}.'
        )


def check_worker_version(
        base_url: str, timeout: float = 10.0, session=None,
    ) -> VersionResponse:
    """Return the worker's version, or raise `WorkerVersionError` if it uses
    another protocol than this client (see `protocol.py`).

    Workers with protocol 1 have no `/version` endpoint (HTTP 404).
    """
    res = (session or requests).get(
        f"{base_url.rstrip('/')}/version", timeout=timeout
    )
    if res.status_code == 404:
        raise WorkerVersionError(base_url, 1)
    ModelClient._raise_for_status(res)
    version = parse_response(VersionResponse, res.json())
    if version.protocol_version != PROTOCOL_VERSION:
        raise WorkerVersionError(
            base_url, version.protocol_version, version.cellacdc_version
        )
    return version


def post_init(base_url: str, init_kwargs: Dict[str, Any], timeout=None):
    """Initialize a worker started with `--defer-init`.

    Sends data, not paths: arguments that are paths of files on this
    computer are uploaded (see `payload.encode`).
    """
    from cellacdc.service import payload

    with startup_timer("model initialization payload encoding"):
        body = payload.encode(init_kwargs)
    with startup_timer("worker /init request (model construction and weights)"):
        res = requests.post(
            f"{base_url.rstrip('/')}/init",
            files={"kwargs": _attachment("kwargs.bin", body)},
            headers=_protocol_headers(),
            timeout=timeout,
        )
        ModelClient._raise_for_status(res)

def get_model_arg_spec(base_url: str, timeout: float = 10.0):
    """Fetch and decode model argument specs from a worker."""
    init_params, segment_params, _ = get_model_spec(base_url, timeout=timeout)
    return init_params, segment_params


def get_model_spec(base_url: str, timeout: float = 10.0):
    """Fetch model argument specs and help URL, including before initialization."""
    from cellacdc import myutils

    res = requests.get(
        f"{base_url.rstrip('/')}/spec", headers=_protocol_headers(),
        timeout=timeout,
    )
    ModelClient._raise_for_status(res)
    data = res.json()

    def decode_params(params):
        return [
            myutils.ArgSpec(
                name=name,
                default=default,
                type=myutils._deserialize_model_param_type(param_type),
                desc=desc,
                docstring=docstring,
            )
            for name, default, param_type, desc, docstring in params
        ]

    return (
        decode_params(data['init_params']),
        decode_params(data['segment_params']),
        data.get('url'),
    )


class ModelClient:
    """HTTP client interface for Cell-ACDC segmentation models.
    Presents the same interface as an in-process acdcSegment.Model instance.
    """
    @staticmethod
    def _raise_for_status(res):
        if res.ok:
            return
        try:
            detail = res.json().get("detail", res.text)
        except Exception:
            detail = res.text
        raise RuntimeError(
            f"Model server error {res.status_code} at {res.url}:\n{detail}"
        )

    def __init__(self, base_url: str, timeout: float = None): # allow for infinite timeout
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(_protocol_headers())
        self.worker_version = check_worker_version(
            self.base_url, session=self.session
        )
        self.refresh_info()

    def refresh_info(self):
        """Read the model description again (e.g. after `POST /init`)."""
        self.info = self._fetch_info()
        self.model_name = self.info.name
        self.is_segment3DT_available = self.info.capabilities.segment3DT

    def _fetch_info(self) -> InfoResponse:
        res = self.session.get(f"{self.base_url}/info", timeout=10)
        res.raise_for_status()
        return parse_response(InfoResponse, res.json())

    def segment(
            self, 
            image: np.ndarray, 
            frame_i: Optional[int] = None, 
            posData: Any = None, 
            start_z_slice: Optional[int] = None, 
            **kwargs
        ) -> np.ndarray:
        from cellacdc.service import payload

        persist_sam = (
            getattr(self, 'model_name', None) in ('segment_anything', 'sam2')
            and posData is not None
        )
        if persist_sam:
            kwargs['save_embeddings'] = True
        if posData is not None and kwargs.get('use_loaded_embeddings'):
            load_embeddings = getattr(posData, 'loadSamEmbeddings', None)
            if load_embeddings is not None:
                load_embeddings()
        kwargs["frame_i"] = frame_i
        if start_z_slice is not None:
            kwargs["start_z_slice"] = start_z_slice
            
        files = _segment_files(image, kwargs)
        embedding_operation = any(
            kwargs.get(name)
            for name in (
                'save_embeddings', 'only_embeddings', 'use_loaded_embeddings'
            )
        )
        if embedding_operation:
            frame_embeddings = {}
            if posData is not None and kwargs.get('use_loaded_embeddings'):
                frame_embeddings = dict(getattr(
                    posData, 'sam_embeddings', {}
                ).get(frame_i, {}))
            files['embeddings'] = _attachment(
                'embeddings.bin', payload.encode_value_body(frame_embeddings)
            )
        res = self.session.post(
            f"{self.base_url}/segment",
            files=files,
            timeout=self.timeout
        )
        self._raise_for_status(res)
        if embedding_operation:
            result = payload.decode_value_body(res.content)
            if posData is not None:
                if not hasattr(posData, 'sam_embeddings'):
                    posData.sam_embeddings = {}
                existing = posData.sam_embeddings.setdefault(frame_i, {})
                if any(key in existing for key in ('features', '_features')):
                    existing = {start_z_slice or 0: existing}
                    posData.sam_embeddings[frame_i] = existing
                existing.update(result['embeddings'])
                if persist_sam:
                    save_to_disk = getattr(posData, 'saveSamEmbeddings', None)
                    if save_to_disk is not None:
                        save_to_disk()
            return result['labels']
        return np.load(io.BytesIO(res.content))

    def segment3DT(
            self, 
            video_data: np.ndarray, 
            signals: Any = None, 
            **kwargs
        ) -> np.ndarray:
        if not self.is_segment3DT_available:
            raise NotImplementedError(
                f"Model {self.model_name} does not support segment3DT"
            )
        res = self.session.post(
            f"{self.base_url}/segment3DT",
            files=_segment_files(video_data, kwargs),
            timeout=self.timeout
        )
        self._raise_for_status(res)
        return np.load(io.BytesIO(res.content))

    def add_prompt(self, **kwargs):
        """Forward a promptable model's state update to its worker."""
        from cellacdc.service import payload

        res = self.session.post(
            f'{self.base_url}/add_prompt',
            files={'kwargs': _attachment('kwargs.bin', payload.encode(kwargs))},
            timeout=self.timeout,
        )
        self._raise_for_status(res)

    def second_ch_img_to_stack(
            self, 
            first_ch_data: np.ndarray, 
            second_ch_data: np.ndarray
        ) -> np.ndarray:
        """Run the worker model's own `second_ch_img_to_stack`."""
        if not self.info.capabilities.second_ch_img_to_stack:
            raise NotImplementedError(
                f"Model {self.model_name} does not support "
                "second_ch_img_to_stack"
            )
        files = {}
        for key, arr in (("first", first_ch_data), ("second", second_ch_data)):
            buf = io.BytesIO()
            np.save(buf, np.asarray(arr))
            files[key] = (f"{key}.npy", buf.getvalue(), "application/octet-stream")
        res = self.session.post(
            f"{self.base_url}/second_ch_img_to_stack",
            files=files,
            timeout=self.timeout
        )
        self._raise_for_status(res)
        return np.load(io.BytesIO(res.content))

    def predictCcaState(
            self, image: np.ndarray, precomputed_lab: np.ndarray
        ):
        """Run the worker model's `predictCcaState`; returns the cca_df."""
        import pandas as pd

        if not self.info.capabilities.predictCcaState:
            raise NotImplementedError(
                f"Model {self.model_name} does not support predictCcaState"
            )
        files = {}
        for key, arr in (("image", image), ("precomputed_lab", precomputed_lab)):
            buf = io.BytesIO()
            np.save(buf, np.asarray(arr))
            files[key] = (f"{key}.npy", buf.getvalue(), "application/octet-stream")
        res = self.session.post(
            f"{self.base_url}/predictCcaState",
            files=files,
            timeout=self.timeout
        )
        self._raise_for_status(res)
        return pd.read_json(io.StringIO(res.json()["cca_df"]), orient="table")

    def submit_job(self, image: np.ndarray, **kwargs) -> str:
        res = self.session.post(
            f"{self.base_url}/jobs/segment",
            files=_segment_files(image, kwargs),
            timeout=self.timeout
        )
        self._raise_for_status(res)
        return res.json()["job_id"]

    def get_job_status(self, job_id: str) -> dict:
        res = self.session.get(f"{self.base_url}/jobs/{job_id}", timeout=10)
        res.raise_for_status()
        return res.json()

    def get_job_result(self, job_id: str) -> np.ndarray:
        res = self.session.get(
            f"{self.base_url}/jobs/{job_id}/result", 
            timeout=self.timeout
        )
        res.raise_for_status()
        return np.load(io.BytesIO(res.content))

    def cancel_job(self, job_id: str) -> dict:
        res = self.session.delete(f"{self.base_url}/jobs/{job_id}", timeout=10)
        res.raise_for_status()
        return res.json()

    def setupLogger(self, logger=None):
        pass

    def closeLogger(self):
        pass

    def close(self):
        self.session.close()


class LocalBackend:
    """Manages spawning, communication with, and lifecycle of a local model worker."""
    def __init__(
            self, 
            model_name: str, 
            env_python: Optional[str] = None, 
            port: Optional[int] = None, 
            host: str = "127.0.0.1",
            init_model_kwargs: Optional[Dict[str, Any]] = None,
            model_kind: str = 'segmentation',
        ):
        self.model_name = model_name
        self.env_python = env_python if env_python else sys.executable
        self.host = host
        # A port chosen here can be replaced when the worker is restarted
        self._auto_port = port is None
        self.port = port if port is not None else _find_free_port(host)
        self.init_model_kwargs = init_model_kwargs or {}
        self.model_kind = model_kind
        self.proc: Optional[subprocess.Popen] = None
        self.client: Optional[ModelClient] = None
        self.base_url = f"http://{self.host}:{self.port}"

    @property
    def url(self) -> str:
        return self.base_url

    def prepare(self, timeout: float = 120.0):
        """Start without constructing a model and retain the process for `start`."""
        try:
            if not self.is_running():
                self._start_process(timeout)
            return get_model_spec(self.base_url, timeout=timeout)
        except Exception:
            self.stop()
            raise

    def start(self, timeout: float = 120.0) -> ModelClient:
        t0 = time.monotonic()
        if not self.is_running():
            # The budget counts from the last launch: a repair of the
            # environment (pip) does not shorten the time for /init
            launched = self._start_process(timeout)
            if launched is not None:
                t0 = launched
        cmd = self.proc.args
        with startup_timer(f"{self.model_name}: deferred initialization"):
            self._initialize(cmd, timeout - (time.monotonic() - t0))
        try:
            with startup_timer(f"{self.model_name}: client /info connection"):
                self.client = ModelClient(self.base_url)
        except Exception as err:
            message = self._describe_failure(
                f"Connecting to the {self.model_name} worker "
                f"failed:\n{err}", cmd
            )
            self.stop()
            raise RuntimeError(message) from err
        return self.client

    def _start_process(self, timeout):
        """Start the worker and check that it uses this client's protocol.

        If it does not, Cell-ACDC is reinstalled in the worker's environment
        (only in environments created by Cell-ACDC, see
        `environment.is_managed_model_env`) and the worker is started again.
        Returns the `time.monotonic()` at which the running worker was
        launched.
        """
        launched = time.monotonic()
        cmd = self._launch(timeout)
        mismatch = self._check_version(cmd)
        if mismatch is None:
            return launched
        self.stop()
        self._repair_environment(mismatch)
        if self._auto_port:
            # The previous worker's port may not be reusable right away
            self.port = _find_free_port(self.host)
            self.base_url = f"http://{self.host}:{self.port}"
        launched = time.monotonic()
        cmd = self._launch(timeout)
        mismatch = self._check_version(cmd)
        if mismatch is not None:
            message = self._describe_failure(
                f"{mismatch}\nReinstalling Cell-ACDC in the model environment "
                "did not fix it.", cmd
            )
            self.stop()
            raise RuntimeError(message) from mismatch
        return launched

    def _check_version(self, cmd) -> Optional[WorkerVersionError]:
        """Return the version mismatch of the running worker, or None.

        Other errors stop the worker and are raised.
        """
        try:
            check_worker_version(self.base_url)
        except WorkerVersionError as err:
            return err
        except Exception as err:
            message = self._describe_failure(
                f"Reading the version of the {self.model_name} worker "
                f"failed:\n{err}", cmd
            )
            self.stop()
            raise RuntimeError(message) from err
        return None

    def _repair_environment(self, mismatch: WorkerVersionError):
        """Reinstall this Cell-ACDC build in the worker's environment."""
        from cellacdc.service import environment

        # Compare environment directories, not interpreters: on Linux/macOS a
        # venv's `bin/python` links to the base interpreter, which may also
        # be the client's
        env_dir = environment.managed_model_env_dir(self.env_python)
        own_env = env_dir is not None and environment._same_path(
            env_dir, sys.prefix
        )
        if env_dir is None or own_env:
            raise RuntimeError(
                f"{mismatch}\nInstall the same Cell-ACDC version in "
                f"{self.env_python} (Cell-ACDC repairs only the model "
                "environments that it created)."
            ) from mismatch
        print(
            f"{mismatch}\nReinstalling Cell-ACDC in {self.env_python}...",
            flush=True,
        )
        if not environment.repair_cellacdc_runtime(self.env_python):
            raise RuntimeError(
                f"{mismatch}\nReinstalling Cell-ACDC in {self.env_python} "
                "failed (see the log above)."
            ) from mismatch

    def _launch(self, timeout):
        """Start the worker process and wait until it answers `/health`.

        Returns the command line.
        """
        from cellacdc.config import parser_args

        cmd = [
            self.env_python, "-m", "cellacdc.service.worker", "--server",
            "--model", self.model_name,
            "--port", str(self.port),
            "--host", self.host,
            "--model-kind", self.model_kind,
            "--defer-init",
        ]
        cmd.append("--debug" if parser_args["debug"] else "--no-debug")
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self._output = collections.deque(maxlen=200)
        with startup_timer(f"{self.model_name}: worker process launch"):
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
            )
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

        with startup_timer(f"{self.model_name}: waiting for worker /health"):
            self._wait_for_health(cmd, timeout)
        return cmd

    def _wait_for_health(self, cmd, timeout):
        try:
            self._poll_health(cmd, timeout)
        finally:
            if getattr(self, '_health_poll_failed', False):
                # The connection errors of urllib3 are reference cycles
                # (exception -> traceback -> frame -> exception) and their
                # frames keep the callers' frames alive. When the worker is
                # started inside `SegmKernel.run` (CLI), the arrays of the
                # position would stay in memory after `run` returns until
                # the cyclic garbage collector runs
                gc.collect()

    def _poll_health(self, cmd, timeout):
        self._health_poll_failed = False
        t0 = time.monotonic()
        last_error = None
        while time.monotonic() - t0 < timeout:
            if self.proc.poll() is not None:
                self._reader.join(timeout=2)
                raise RuntimeError(self._describe_failure(
                    f"The {self.model_name} worker exited during startup "
                    f"with code {self.proc.returncode}.",
                    cmd,
                ))
            try:
                res = requests.get(f"{self.base_url}/health", timeout=1.0)
                healthy = res.status_code == 200
                if not healthy:
                    last_error = f"/health returned HTTP {res.status_code}"
            except Exception as err:
                healthy = False
                self._health_poll_failed = True
                last_error = f"{type(err).__name__}: {err}"
            if healthy:
                return
            time.sleep(0.3)

        reason = (
            f"Timed out after {timeout:.0f}s waiting for the {self.model_name} "
            f"worker at {self.base_url}. The process was still running but "
            f"never answered (last connection error: {last_error})."
        )
        message = self._describe_failure(reason, cmd, diagnose=True)
        self.stop()
        raise TimeoutError(message)

    def _initialize(self, cmd, timeout):
        try:
            post_init(
                self.base_url, self.init_model_kwargs,
                timeout=max(timeout, 10.0),
            )
        except Exception as err:
            message = self._describe_failure(
                f"Initializing the {self.model_name} model failed:\n{err}", cmd
            )
            self.stop()
            raise RuntimeError(message) from err

    def _read_output(self):
        """Forward worker output live and retain recent lines for failures."""
        try:
            for line in iter(self.proc.stdout.readline, b""):
                text = line.decode(errors="replace").rstrip()
                self._output.append(text)
                print(f'[{self.model_name} worker] {text}', flush=True)
        except Exception:
            pass

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def reinitialize(self, init_model_kwargs: Dict[str, Any]) -> ModelClient:
        """Replace the model of the running worker (`POST /init`).

        Much faster than a restart: the Python process, its imports (e.g.
        torch) and the HTTP server are kept. If initialization fails, the
        worker keeps its previous model and the error is raised.
        """
        if not self.is_running() or self.client is None:
            raise RuntimeError(f"The {self.model_name} worker is not running.")
        try:
            post_init(self.base_url, init_model_kwargs)
        except Exception as err:
            output = "\n".join(self._output).strip()
            raise RuntimeError(
                f"Re-initializing the {self.model_name} model failed:\n{err}"
                + (f"\nWorker output (last lines):\n{output}" if output else "")
            ) from err
        self.init_model_kwargs = init_model_kwargs
        self.client.refresh_info()
        return self.client

    def _describe_failure(self, reason, cmd, diagnose=False) -> str:
        output = "\n".join(self._output).strip()
        parts = [
            reason,
            f"Python: {self.env_python}",
            f"Command: {' '.join(cmd)}",
        ]
        if output:
            parts.append(f"Worker output (last lines):\n{output}")
        else:
            parts.append("The worker printed no output.")
        if diagnose:
            parts.append(self._diagnose_imports())
        return "\n".join(parts)

    def _diagnose_imports(self) -> str:
        """Check whether the worker module can even be imported (and how long it takes)."""
        if not os.path.isfile(self.env_python):
            return f"Diagnosis: the Python executable does not exist: {self.env_python}"
        t0 = time.time()
        try:
            res = subprocess.run(
                [self.env_python, "-c",
                 "import cellacdc.service.worker, cellacdc.models; print('ok')",
                 "--server"],
                capture_output=True, text=True, timeout=120,
            )
        except subprocess.TimeoutExpired:
            return (
                "Diagnosis: importing the worker module takes more than 120s "
                "(heavy imports, antivirus scanning or a hung import)."
            )
        elapsed = time.time() - t0
        if res.returncode != 0:
            return (
                "Diagnosis: importing the worker module fails in this "
                f"environment:\n{(res.stderr or res.stdout).strip()}"
            )
        return (
            f"Diagnosis: the worker module imports fine ({elapsed:.1f}s). "
            "Startup is probably just slow (loading the model/weights), "
            "or the port is blocked by a firewall/proxy. "
            "Try a larger timeout."
        )

    def stop(self):
        if self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass
            self.client = None

        if self.proc is not None and self.proc.poll() is None:
            try:
                requests.post(f"{self.base_url}/shutdown", timeout=1.0)
            except Exception:
                pass
            try:
                self.proc.terminate()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()


class RemoteBackend:
    """Manages connection to an existing remote model worker."""
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client: Optional[ModelClient] = None

    @property
    def url(self) -> str:
        return self.base_url

    def get_client(
            self, timeout: float = 300.0,
            init_model_kwargs: Optional[Dict[str, Any]] = None,
        ) -> ModelClient:
        try:
            check_worker_version(
                self.base_url,
                timeout=30.0 if timeout is None else min(timeout, 30.0),
            )
        except WorkerVersionError as err:
            raise RuntimeError(
                f"{err}\nInstall the same Cell-ACDC version on the server."
            ) from err
        if init_model_kwargs is not None:
            post_init(self.base_url, init_model_kwargs, timeout=timeout)
        self.client = ModelClient(self.base_url, timeout=timeout)
        return self.client

    def stop(self):
        if self.client is not None:
            self.client.close()
            self.client = None
