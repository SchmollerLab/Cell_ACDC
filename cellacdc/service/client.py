import collections
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

from cellacdc.service.schema import InfoResponse, parse_response

def _find_free_port(host: str = "127.0.0.1") -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        s.listen(1)
        return s.getsockname()[1]

def _sanitize_kwargs(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = {}
    for k, v in kwargs.items():
        if k in ("posData", "signals", "logger_func", "signals_tqdm"):
            continue
        # Check JSON serializability
        try:
            json.dumps(v)
            sanitized[k] = v
        except (TypeError, OverflowError):
            pass
    return sanitized

def post_init(base_url: str, init_kwargs: Dict[str, Any], timeout=None):
    """Initialize a worker started with `--defer-init` (sends data, not paths)."""
    from cellacdc.service import payload

    kwargs_json, data = payload.encode(init_kwargs)
    files = None
    if data is not None:
        files = {"data": ("init.npz", data, "application/octet-stream")}
    res = requests.post(
        f"{base_url.rstrip('/')}/init",
        data={"kwargs": kwargs_json},
        files=files,
        timeout=timeout,
    )
    ModelClient._raise_for_status(res)

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
        p = _sanitize_kwargs(kwargs)
        if frame_i is not None:
            p["frame_i"] = frame_i
        if start_z_slice is not None:
            p["start_z_slice"] = start_z_slice

        buf = io.BytesIO()
        np.save(buf, image)
        buf.seek(0)

        files = {
            "image": ("image.npy", buf.getvalue(), "application/octet-stream")
        }
        data = {
            "params": json.dumps(p)
        }
        res = self.session.post(
            f"{self.base_url}/segment",
            files=files,
            data=data,
            timeout=self.timeout
        )
        self._raise_for_status(res)
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
        p = _sanitize_kwargs(kwargs)
        buf = io.BytesIO()
        np.save(buf, video_data)
        buf.seek(0)

        files = {
            "image": ("image.npy", buf.getvalue(), "application/octet-stream")
        }
        data = {
            "params": json.dumps(p)
        }
        res = self.session.post(
            f"{self.base_url}/segment3DT",
            files=files,
            data=data,
            timeout=self.timeout
        )
        self._raise_for_status(res)
        return np.load(io.BytesIO(res.content))

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
        p = _sanitize_kwargs(kwargs)
        buf = io.BytesIO()
        np.save(buf, image)
        buf.seek(0)
        files = {
            "image": ("image.npy", buf.getvalue(), "application/octet-stream")
        }
        data = {"params": json.dumps(p)}
        res = self.session.post(
            f"{self.base_url}/jobs/segment",
            files=files,
            data=data,
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
            init_model_kwargs: Optional[Dict[str, Any]] = None
        ):
        self.model_name = model_name
        self.env_python = env_python if env_python else sys.executable
        self.host = host
        self.port = port if port is not None else _find_free_port(host)
        self.init_model_kwargs = init_model_kwargs or {}
        self.proc: Optional[subprocess.Popen] = None
        self.client: Optional[ModelClient] = None
        self.base_url = f"http://{self.host}:{self.port}"

    @property
    def url(self) -> str:
        return self.base_url

    def start(self, timeout: float = 120.0) -> ModelClient:
        from cellacdc.service import payload

        defer_init = payload.needs_payload(self.init_model_kwargs)
        cmd = [
            self.env_python, "-m", "cellacdc.service.worker", "--server",
            "--model", self.model_name,
            "--port", str(self.port),
            "--host", self.host,
        ]
        if defer_init:
            cmd.append("--defer-init")
        else:
            cmd += ["--init-kwargs", json.dumps(self.init_model_kwargs)]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        self._output = collections.deque(maxlen=200)
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        self._reader = threading.Thread(target=self._read_output, daemon=True)
        self._reader.start()

        t0 = time.time()
        last_error = None
        while time.time() - t0 < timeout:
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
                last_error = f"{type(err).__name__}: {err}"
            if healthy:
                if defer_init:
                    self._initialize(cmd, timeout - (time.time() - t0))
                try:
                    self.client = ModelClient(self.base_url)
                except Exception as err:
                    message = self._describe_failure(
                        f"Connecting to the {self.model_name} worker "
                        f"failed:\n{err}", cmd
                    )
                    self.stop()
                    raise RuntimeError(message) from err
                return self.client
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
        try:
            for line in iter(self.proc.stdout.readline, b""):
                self._output.append(line.decode(errors="replace").rstrip())
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
        if init_model_kwargs is not None:
            post_init(self.base_url, init_model_kwargs, timeout=timeout)
        self.client = ModelClient(self.base_url, timeout=timeout)
        return self.client

    def stop(self):
        if self.client is not None:
            self.client.close()
            self.client = None
