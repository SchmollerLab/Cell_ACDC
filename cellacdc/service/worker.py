"""HTTP worker that serves one Cell-ACDC segmentation model.

This process runs *inside the model's environment* (e.g. `_acdc_cellpose_v3`).
It loads `cellacdc/models/<model>/acdcSegment.py` and lets the Cell-ACDC GUI
(or CLI) call the model over HTTP. The client side is
`cellacdc/service/client.py` (`ModelClient`, `LocalBackend`, `RemoteBackend`).

Started as::

    <model env python> -m cellacdc.service.worker --server --model <name> --port 8520

FastAPI in a nutshell
---------------------
* `app = FastAPI()` creates the web application. It does not listen on a port
  by itself; `uvicorn.run(app, ...)` in `main()` starts the actual HTTP server
  and keeps it running until the process is stopped.
* A decorator such as `@app.post("/segment")` registers the function below it
  as the handler for HTTP `POST` requests to `http://<host>:<port>/segment`.
  `@app.get` handles `GET` requests (reading), `@app.delete` `DELETE`
  requests. We never call these functions ourselves; FastAPI calls them when a
  request arrives.
* The function arguments describe what the request must contain. FastAPI reads
  and converts them automatically:

  - `image: UploadFile = File(...)`: an uploaded file (here: a `.npy` array).
    `File(...)` means required, `File(None)` means optional.
  - `params: str = Form("{}")`: a text field of the form (here: JSON text with
    the keyword arguments); `"{}"` is the default if it is missing.
  - `job_id: str` matching `{job_id}` in the path: taken from the URL, e.g.
    `/jobs/abc123` -> `job_id = "abc123"`.

  File and form fields are sent by the client as a "multipart" request (like a
  web form with a file upload); parsing them requires `python-multipart`.
* The return value is sent back to the client:

  - a `dict` or a pydantic model (from `schema.py`) is sent as JSON.
    `response_model=...` in the decorator validates/documents its structure.
  - `Response(content=bytes, ...)` sends raw bytes (here: the labels as `.npy`).
* `raise HTTPException(status_code, detail)` sends an error response. The
  client (`ModelClient._raise_for_status`) turns it into a `RuntimeError`
  that includes `detail` (we put the full traceback there). Status codes used:
  400 = bad request, 404 = not found, 500 = error in the model,
  503 = model not initialized yet.
* `async def` handlers can `await` (e.g. reading the uploaded file) without
  blocking the server; plain `def` handlers are run in a thread pool by
  FastAPI. Both behave the same from the client's point of view.
  Async handlers offload model initialization and inference to the thread
  pool, including waiting for the shared model lock, so other requests can
  still be served while a model operation is running.
* While the worker runs, `http://<host>:<port>/docs` shows an interactive page
  listing all endpoints (generated automatically by FastAPI).

Typical sequence of requests (what `LocalBackend.start` + `ModelClient` do)
-------------------------------------------------------------------------
1. `GET /health` repeatedly until the server answers (startup is done).
2. Optionally `POST /init` (only with `--defer-init`, to send data such as an
   auxiliary segmentation that cannot be passed on the command line).
3. `GET /info` to learn the model's capabilities.
4. `POST /segment` (or `/segment3DT`, or the `/jobs/...` endpoints) for each
   image.
5. `POST /shutdown` when done.
"""

import io
import json
import os
import signal
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import Response
from fastapi.concurrency import run_in_threadpool

from cellacdc.service.schema import HealthResponse, InfoResponse, Capabilities, JobStatusResponse

# Background jobs (`/jobs/...`) run in these threads, so that the HTTP request
# that submits a job returns immediately with a job id.
_executor = ThreadPoolExecutor(max_workers=2)
# job_id -> {"status", "progress", "error", "result"}; kept in memory only.
# A completed job is removed when its result is fetched. Of the jobs nobody
# collects, only the newest `MAX_FINISHED_JOBS` finished ones are kept, so a
# long-running worker does not accumulate results.
_jobs: Dict[str, Dict[str, Any]] = {}
MAX_FINISHED_JOBS = 50
_FINISHED_STATUSES = ("completed", "failed", "cancelled")
# Job threads and request handlers both update `_jobs`. All reads and writes go
# through this lock, so that a cancellation cannot be overwritten by a
# concurrent "running"/"completed"/"failed" transition.
_jobs_lock = threading.Lock()


def _prune_finished_jobs():
    """Drop the oldest finished jobs beyond `MAX_FINISHED_JOBS`.

    Call with `_jobs_lock` held. Dicts keep insertion order, so the first
    entries are the oldest submissions.
    """
    finished = [
        job_id for job_id, job in _jobs.items()
        if job["status"] in _FINISHED_STATUSES
    ]
    for job_id in finished[:max(0, len(finished) - MAX_FINISHED_JOBS)]:
        del _jobs[job_id]


def _transition_job(job_id: str, status: str, **fields) -> bool:
    """Set the job's status (and `fields`) unless it was cancelled.

    Returns False (and changes nothing) if the job was cancelled (or already
    removed).
    """
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None or job["status"] == "cancelled":
            return False
        job["status"] = status
        job.update(fields)
        if status in _FINISHED_STATUSES:
            _prune_finished_jobs()
        return True

def _check_gpu() -> bool:
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except Exception:
        pass
    try:
        import tensorflow as tf
        if tf.config.list_physical_devices('GPU'):
            return True
    except Exception:
        pass
    return False

def _filter_kwargs(func, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs the model method doesn't accept (e.g. frame_i, start_z_slice)."""
    import inspect
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


class _EmbeddingPosData:
    """Minimal position-data interface used by SAM models in a worker."""

    def __init__(self, model, frame_i, embeddings):
        self.model = model
        self.frame_i = frame_i
        self.sam_embeddings = {frame_i: embeddings}

    def getSamEmbeddings(self, frame_i=0, z=0):
        embedding = self.sam_embeddings.get(frame_i, {}).get(z)
        if embedding is None:
            return None
        embedding = dict(embedding)
        features = embedding.get('features')
        if isinstance(features, np.ndarray):
            import torch
            device = getattr(self.model, 'device', None)
            embedding['features'] = torch.as_tensor(features, device=device)
        return embedding

    def storeSamEmbeddings(self, sam_model, frame_i=0, z=0):
        predictor = getattr(sam_model.model, 'predictor', sam_model.model)
        embedding = {
            'original_size': predictor.original_size,
            'input_size': predictor.input_size,
            'features': predictor.features,
            'is_image_set': True,
        }
        self.sam_embeddings.setdefault(frame_i, {})[z] = embedding

    @property
    def frame_embeddings(self):
        return self.sam_embeddings.get(self.frame_i, {})

def instantiate_model(acdcSegment, init_kwargs: Dict[str, Any]):
    """Create `acdcSegment.Model` from decoded init kwargs.

    The worker may run on another machine, so it never receives client file
    paths: the client loads files itself and sends the data (`segm_data`,
    `input_points_df`, or uploaded model files). Uploaded files are stored in
    temporary directories until the model is released. Raw client paths and
    directory selections are rejected with an error.
    Models whose `__init__` has a required positional argument receive the
    auxiliary segmentation as that argument.
    """
    from contextlib import ExitStack
    import inspect
    import tempfile
    import weakref
    from cellacdc import myutils
    from cellacdc.service.payload import ModelFile

    init_kwargs = dict(init_kwargs)
    init_params, _ = myutils.getModelArgSpec(acdcSegment)
    path_params = {
        param.name: myutils._serialize_model_param_type(param.type)['kind']
        for param in init_params
    }
    uploaded = {
        key for key, value in init_kwargs.items()
        if isinstance(value, ModelFile)
    }
    invalid_uploads = {
        key for key in uploaded
        if path_params.get(key) not in ('path', 'csv_path')
    }
    if invalid_uploads:
        raise ValueError(
            f'Init kwargs {sorted(invalid_uploads)} do not accept file uploads.'
        )
    unsupported_paths = {
        key for key in (set(path_params) & set(init_kwargs)) - uploaded
        if path_params[key] in ('path', 'csv_path', 'folder')
        if init_kwargs[key] is not None and not (
            isinstance(init_kwargs[key], str)
            and init_kwargs[key] in ('', 'None')
        )
    }
    if unsupported_paths:
        raise ValueError(
            f'Init kwargs {sorted(unsupported_paths)} reference client files. '
            'The worker cannot access client-side paths; send the file data '
            'instead.'
        )
    # Empty selections ('' / 'None') mean "no file" and carry no client path
    for key in ('segm_endname', 'input_points_path'):
        if init_kwargs.get(key) in (None, '', 'None'):
            init_kwargs.pop(key, None)
    unresolved = {'segm_endname', 'input_points_path'} & set(init_kwargs)
    if unresolved:
        raise ValueError(
            f'Init kwargs {sorted(unresolved)} reference client files. The '
            'client must send the data instead (see '
            'myutils.resolve_segm_model_init_data).'
        )
    segm_data = init_kwargs.pop('segm_data', None)
    args = inspect.getfullargspec(acdcSegment.Model.__init__).args
    if 'is_rgb' not in args:
        init_kwargs.pop('is_rgb', None)
    with ExitStack() as files:
        for key in uploaded:
            upload = init_kwargs[key]
            directory = files.enter_context(
                tempfile.TemporaryDirectory(prefix='cellacdc-model-')
            )
            path = os.path.join(directory, upload.name)
            with open(path, 'wb') as stream:
                stream.write(upload.data)
            init_kwargs[key] = path
        if myutils.model_init_requires_segm_data(acdcSegment):
            if segm_data is None:
                raise ValueError(
                    'This model requires an auxiliary segmentation (segm_data), '
                    'but none was sent.'
                )
            model = acdcSegment.Model(segm_data, **init_kwargs)
        else:
            model = acdcSegment.Model(**init_kwargs)
        if uploaded:
            model._acdc_uploaded_files_cleanup = weakref.finalize(
                model, files.pop_all().close
            )
        return model

def create_app(
        model_name: str,
        init_kwargs: Optional[Dict[str, Any]] = None,
        defer_init: bool = False,
        model_kind: str = 'segmentation',
    ) -> FastAPI:
    """Build the FastAPI application for `model_name`.

    The model module is imported once here (so import errors stop the worker
    at startup, with a clear traceback in its output). All endpoint functions
    below are defined inside this function, so they can use `acdcSegment`,
    `state` and `model_name` from here (closures).

    Parameters
    ----------
    model_name : str
        Folder name in `cellacdc/models`.
    init_kwargs : dict, optional
        Keyword arguments for `acdcSegment.Model` (JSON values only).
    defer_init : bool
        If True, do not create the model now: the client sends the init
        kwargs (possibly including arrays or model files) with `POST /init`.
    """
    from cellacdc import myutils
    from cellacdc.service import payload

    importer = (
        myutils.import_promptable_segment_module
        if model_kind == 'promptable' else myutils.import_segment_module
    )
    acdcSegment = importer(model_name)
    # Mutable holder, so `/init` can replace the model for all endpoints
    state = {'model': None}
    model_lock = threading.Lock()

    def initialize_model(kwargs):
        with model_lock:
            state['model'] = instantiate_model(acdcSegment, kwargs)

    if not defer_init:
        initialize_model(init_kwargs or {})

    init_params, segment_params = myutils.getModelArgSpec(acdcSegment)

    def serialize_params(params):
        return [
            [
                param.name,
                json.loads(json.dumps(param.default, default=str)),
                myutils._serialize_model_param_type(param.type),
                param.desc,
                param.docstring,
            ]
            for param in params
        ]

    def call_model_method(method_name, *args, **kwargs):
        with model_lock:
            current_model = state['model']
            if current_model is None:
                raise HTTPException(
                    status_code=503,
                    detail='Model is not initialized yet. POST /init first.'
                )
            method = getattr(current_model, method_name)
            return method(*args, **_filter_kwargs(method, kwargs))

    class _ModelProxy:
        """Stands in for the model in the endpoints below.

        `model.segment(...)` is forwarded to the current `state['model']`, or
        answers with HTTP 503 if `/init` has not been called yet.
        """
        def __getattr__(self, name):
            if state['model'] is None:
                raise HTTPException(
                    status_code=503,
                    detail='Model is not initialized yet. POST /init first.'
                )
            return getattr(state['model'], name)

    model = _ModelProxy()

    # Title/description/version only appear on the automatic /docs page
    app = FastAPI(
        title=f"Cell-ACDC {model_name} Worker",
        description="Worker service serving Cell-ACDC segmentation models.",
        version="1.0.0"
    )

    # ---------------------------------------------------------------- setup

    @app.post("/init")
    async def init(
            kwargs: str = Form("{}"), data: Optional[UploadFile] = File(None)
        ):
        """Create (or re-create) the model from data sent by the client.

        Request (multipart form, see `client.post_init`):
            kwargs: JSON text with the init kwargs; arrays are replaced by
                    placeholders, DataFrames are embedded as JSON (see
                    `payload.py`).
            data:   optional `.npz` file with arrays and model-file bytes.
        Response: `{"status": "initialized"}` or HTTP 500 with the traceback.
        """
        try:
            content = await data.read() if data is not None else None
            decoded = payload.decode(kwargs, content)
            await run_in_threadpool(initialize_model, decoded)
            return {"status": "initialized"}
        except Exception:
            import traceback
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.get("/health", response_model=HealthResponse)
    def health():
        """Liveness check: answers as soon as the server is running.

        `LocalBackend.start` polls this until it succeeds (or the process
        exits / the timeout expires). It does not require an initialized model.
        Response (JSON): `{"status": "ok", "model": "<model_name>"}`.
        """
        return HealthResponse(status="ok", model=model_name)

    @app.get("/spec")
    def spec():
        """Return model argument specs without requiring model initialization."""
        return {
            "init_params": serialize_params(init_params),
            "segment_params": serialize_params(segment_params),
        }

    @app.get("/info", response_model=InfoResponse)
    def info():
        """Describe the loaded model (read by `ModelClient` when connecting).

        Response (JSON, see `schema.InfoResponse`): name, version, whether a
        GPU is available, and which optional methods the model implements
        (`segment3DT`, `second_ch_img_to_stack`, `predictCcaState`). HTTP 503
        before `/init`.
        """
        with model_lock:
            model = state['model']
            if model is None:
                raise HTTPException(
                    status_code=503,
                    detail='Model is not initialized yet. POST /init first.'
                )
            has_segment3DT = hasattr(model, "segment3DT")
            has_second_ch = hasattr(model, "second_ch_img_to_stack")
            return InfoResponse(
                name=model_name,
                version=getattr(model, "version", "1.0.0"),
                gpu=_check_gpu(),
                capabilities=Capabilities(
                    segment=True,
                    segment3DT=has_segment3DT,
                    second_ch_img_to_stack=has_second_ch,
                    predictCcaState=hasattr(model, "predictCcaState"),
                )
            )

    # ---------------------------------------------------- direct segmentation
    # The request waits until the result is ready (used by the GUI/CLI).

    @app.post("/segment")
    async def segment(
            image: UploadFile = File(...), params: str = Form("{}"),
            embedding_meta: str = Form(""),
            embeddings: Optional[UploadFile] = File(None),
        ):
        """Run `model.segment(image, **params)` and return the labels.

        Request (multipart form, see `ModelClient.segment`):
            image:  the image saved with `np.save` (`.npy` bytes).
            params: JSON text with the segment kwargs (e.g. `frame_i`).
                    Kwargs the model's `segment` does not accept are dropped.
            embedding_meta/embeddings: optional safe payload containing SAM
                    embeddings from the client-side position data.
        Response: the labels as `.npy` bytes, or HTTP 500 with the traceback.
                  SAM embedding operations return a safe payload containing
                  both labels and updated embeddings.
        """
        try:
            p = json.loads(params) if params else {}
            # Read the uploaded bytes and turn them back into a numpy array
            content = await image.read()
            arr = np.load(io.BytesIO(content))
            embedding_pos_data = None
            if isinstance(embedding_meta, str) and embedding_meta:
                embedding_content = (
                    await embeddings.read()
                    if embeddings is not None and hasattr(embeddings, 'read')
                    else None
                )
                frame_i = p.get('frame_i')
                frame_embeddings = payload.decode_value(
                    embedding_meta, embedding_content
                )
                embedding_pos_data = _EmbeddingPosData(
                    model, frame_i, frame_embeddings
                )
                p['posData'] = embedding_pos_data
            out = await run_in_threadpool(call_model_method, "segment", arr, **p)
            if embedding_pos_data is not None:
                result_meta, result_data = payload.encode_value({
                    'labels': out,
                    'embeddings': embedding_pos_data.frame_embeddings,
                })
                return Response(
                    content=result_data or b'',
                    media_type='application/octet-stream',
                    headers={'X-Cell-ACDC-Payload': result_meta},
                )
            # Send the result back the same way: as `.npy` bytes
            buf = io.BytesIO()
            np.save(buf, out)
            return Response(
                content=buf.getvalue(), 
                media_type="application/octet-stream"
            )
        except Exception as e:
            import traceback
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.post('/add_prompt')
    async def add_prompt(
            kwargs: str = Form('{}'), data: Optional[UploadFile] = File(None)
        ):
        """Update a promptable model with one serialized prompt."""
        if not hasattr(model, 'add_prompt'):
            raise HTTPException(
                status_code=400,
                detail=f'Model {model_name} does not support add_prompt',
            )
        try:
            content = await data.read() if data is not None else None
            decoded = payload.decode(kwargs, content)
            await run_in_threadpool(
                call_model_method, 'add_prompt', **decoded
            )
            return {'status': 'ok'}
        except Exception:
            import traceback
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.post("/segment3DT")
    async def segment3DT(image: UploadFile = File(...), params: str = Form("{}")):
        """Like `/segment`, but for a whole time-lapse via `model.segment3DT`.

        HTTP 400 if the model has no `segment3DT` method.
        """
        if not hasattr(model, "segment3DT"):
            raise HTTPException(
                status_code=400, 
                detail=f"Model {model_name} does not support segment3DT"
            )
        try:
            p = json.loads(params) if params else {}
            content = await image.read()
            arr = np.load(io.BytesIO(content))
            out = await run_in_threadpool(call_model_method, "segment3DT", arr, **p)
            buf = io.BytesIO()
            np.save(buf, out)
            return Response(
                content=buf.getvalue(), 
                media_type="application/octet-stream"
            )
        except Exception as e:
            import traceback
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.post("/second_ch_img_to_stack")
    async def second_ch_img_to_stack(
            first: UploadFile = File(...), second: UploadFile = File(...)
        ):
        """Combine two channels with the model's own `second_ch_img_to_stack`.

        Each model stacks the channels differently (e.g. RGB for Cellpose,
        a two-channel float stack for InstanSeg), so the client must not
        reimplement it.

        Request (multipart form, see `ModelClient.second_ch_img_to_stack`):
            first, second: the two channel images as `.npy` bytes.
        Response: the stacked image as `.npy` bytes. HTTP 400 if the model has
        no `second_ch_img_to_stack` method, 500 with the traceback on errors.
        """
        if not hasattr(model, "second_ch_img_to_stack"):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model {model_name} does not support "
                    "second_ch_img_to_stack"
                )
            )
        try:
            first_arr = np.load(io.BytesIO(await first.read()))
            second_arr = np.load(io.BytesIO(await second.read()))
            out = await run_in_threadpool(
                call_model_method, "second_ch_img_to_stack", first_arr, second_arr
            )
            buf = io.BytesIO()
            np.save(buf, out)
            return Response(
                content=buf.getvalue(),
                media_type="application/octet-stream"
            )
        except Exception:
            import traceback
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    @app.post("/predictCcaState")
    async def predictCcaState(
            image: UploadFile = File(...),
            precomputed_lab: UploadFile = File(...),
        ):
        """Run `model.predictCcaState(image, precomputed_lab)` (YeastMate).

        Request (multipart form, see `ModelClient.predictCcaState`):
            image, precomputed_lab: `.npy` bytes.
        Response (JSON): `{"cca_df": <DataFrame as JSON, orient="table">}`
        (`table` keeps the index name and dtypes). HTTP 400 if the model has
        no `predictCcaState` method, 500 with the traceback on errors.
        """
        if not hasattr(model, "predictCcaState"):
            raise HTTPException(
                status_code=400,
                detail=f"Model {model_name} does not support predictCcaState"
            )
        try:
            img = np.load(io.BytesIO(await image.read()))
            lab = np.load(io.BytesIO(await precomputed_lab.read()))
            cca_df = await run_in_threadpool(
                call_model_method, "predictCcaState", img, lab
            )
            return {"cca_df": cca_df.to_json(orient="table")}
        except Exception:
            import traceback
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    # ------------------------------------------------------- background jobs
    # Alternative to `/segment` for long runs: submit the image, get a job id
    # immediately, then poll the status and fetch the result when completed.
    # Useful for remote servers, where one long HTTP request may time out.

    def _run_job(job_id: str, arr: np.ndarray, params: dict, is_3dt: bool):
        """Runs in an `_executor` thread (not an endpoint).

        A job cancelled while pending is not started; a job cancelled while
        running keeps the status `cancelled` and its result is discarded.
        """
        if not _transition_job(job_id, "running", progress=0.1):
            return
        try:
            if is_3dt:
                out = call_model_method("segment3DT", arr, **params)
            else:
                out = call_model_method("segment", arr, **params)
        except Exception:
            import traceback
            _transition_job(job_id, "failed", error=traceback.format_exc())
            return
        _transition_job(job_id, "completed", result=out, progress=1.0)

    @app.post("/jobs/segment")
    async def submit_job(image: UploadFile = File(...), params: str = Form("{}")):
        """Start segmenting in the background (same request as `/segment`).

        Response (JSON): `{"job_id": "<id>"}`, returned immediately.
        """
        job_id = uuid.uuid4().hex
        p = json.loads(params) if params else {}
        content = await image.read()
        arr = np.load(io.BytesIO(content))
        new_job = {
            "status": "pending",
            "progress": 0.0,
            "error": None,
            "result": None
        }
        with _jobs_lock:
            _jobs[job_id] = new_job
        _executor.submit(_run_job, job_id, arr, p, False)
        return {"job_id": job_id}

    @app.get("/jobs/{job_id}", response_model=JobStatusResponse)
    def get_job_status(job_id: str):
        """`GET /jobs/<id>`: status (pending/running/completed/failed/
        cancelled), progress (0-1) and the traceback if it failed."""
        with _jobs_lock:
            if job_id not in _jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            job = dict(_jobs[job_id])
        return JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            progress=job["progress"],
            error=job.get("error")
        )

    @app.get("/jobs/{job_id}/result")
    def get_job_result(job_id: str):
        """`GET /jobs/<id>/result`: the labels as `.npy` bytes once completed
        (HTTP 400 before that).

        The result can be fetched only once: afterwards the job is removed
        (a later request answers 404), so its array is freed.
        """
        with _jobs_lock:
            if job_id not in _jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            job = _jobs[job_id]
            if job["status"] != "completed":
                raise HTTPException(
                    status_code=400, 
                    detail=f"Job status is {job['status']}, result not ready"
                )
            out = _jobs.pop(job_id)["result"]
        buf = io.BytesIO()
        np.save(buf, out)
        return Response(
            content=buf.getvalue(), 
            media_type="application/octet-stream"
        )

    @app.delete("/jobs/{job_id}")
    def cancel_job(job_id: str):
        """`DELETE /jobs/<id>`: cancel a pending/running job, or delete a
        finished one.

        A pending job is never started. A running computation is not
        interrupted, but its result is discarded and the status stays
        `cancelled`. A finished (completed, failed or cancelled) job is
        removed together with its result.

        Response (JSON): `{"job_id", "status", "deleted"}`, where `status` is
        the job's status at the time of the request.
        """
        with _jobs_lock:
            if job_id not in _jobs:
                raise HTTPException(status_code=404, detail="Job not found")
            job = _jobs[job_id]
            if job["status"] in _FINISHED_STATUSES:
                del _jobs[job_id]
                return {
                    "job_id": job_id, "status": job["status"], "deleted": True
                }
            job["status"] = "cancelled"
            job["result"] = None
            _prune_finished_jobs()
        return {"job_id": job_id, "status": "cancelled", "deleted": False}

    # ------------------------------------------------------------- shutdown

    @app.post("/shutdown")
    def shutdown(background_tasks: BackgroundTasks):
        """Stop the worker process (called by `LocalBackend.stop`).

        `BackgroundTasks` runs `_kill` *after* the response has been sent, so
        the client still receives `{"status": "shutting down"}`.
        """
        with model_lock:
            cleanup = getattr(state['model'], '_acdc_uploaded_files_cleanup', None)
            if cleanup is not None:
                cleanup()
            state['model'] = None

        def _kill():
            import time
            time.sleep(0.5)
            os.kill(os.getpid(), signal.SIGTERM)
        background_tasks.add_task(_kill)
        return {"status": "shutting down"}

    return app

def main():
    """Command-line entry point: parse arguments, build the app, serve it.

    `uvicorn` is the HTTP server that runs the FastAPI app. `uvicorn.run`
    blocks until the process is stopped (e.g. by `/shutdown`).
    """
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Cell-ACDC Model Worker Service")
    parser.add_argument("--model", required=True, help="Segmentation model name")
    parser.add_argument(
        '--model-kind', choices=('segmentation', 'promptable'),
        default='segmentation',
    )
    parser.add_argument("--port", type=int, default=8520, help="Port to listen on")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind to")
    parser.add_argument("--init-kwargs", default="{}", help="JSON string with init kwargs")
    parser.add_argument(
        "--defer-init", action="store_true",
        help="Start without a model; initialize it via POST /init (needed "
             "when init kwargs contain arrays or uploaded model files)"
    )

    parser.add_argument(
        "--server", action="store_true",
        help="Required. Runs Cell-ACDC in server mode (no GUI, no installs); "
             "read at import time by `cellacdc/__init__.py`."
    )

    args = parser.parse_args()
    import cellacdc
    if not (args.server and cellacdc.RUNNING_AS_SERVER):
        parser.error(
            "the worker must be started with --server, e.g. "
            "`python -m cellacdc.service.worker --server --model <name>`"
        )
    init_kwargs = json.loads(args.init_kwargs) if args.init_kwargs else {}

    app = create_app(
        args.model, init_kwargs=init_kwargs, defer_init=args.defer_init,
        model_kind=args.model_kind,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
