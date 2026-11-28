"""Memory use of the segmentation kernel shared by the GUI segmentation
module (`segm.SegmWorker`) and the CLI workflows (`cli.SegmKernel`).

The GUI used more than twice the memory of the CLI on the same time-lapse:
the GUI parameters dialog gives an empty pre-processing recipe `[]` when
pre-processing is unchecked (the CLI gets None), and `segm_model_segment`
copied the whole video into a float64 array for it before `segment3DT`
(4 times the size of a uint16 video, also sent to the model worker).
"""
import gc
import weakref
from types import SimpleNamespace

import numpy as np
import pytest
import requests

from cellacdc import cli, core
from cellacdc.myutils import segm_models
from cellacdc.service import client as service_client

pytestmark = pytest.mark.models


class _RecordingModel:
    """Model that records the images it receives."""
    def __init__(self):
        self.images = []

    def segment(self, image, frame_i=None, posData=None, start_z_slice=0):
        self.images.append(image)
        return np.zeros(image.shape, dtype=np.uint32)

    def segment3DT(self, video):
        self.images.append(video)
        return np.zeros(video.shape[:3], dtype=np.uint32)


GAUSSIAN_RECIPE = [{'method': 'Gaussian Filter', 'kwargs': {'sigma': 1.0}}]


@pytest.mark.parametrize('recipe', [None, []])
def test_no_preprocessing_sends_the_time_lapse_unchanged(recipe):
    # (T, Y, X, 3) uint16 like the stack of `second_ch_img_to_stack`
    video = np.arange(4*5*6*3, dtype=np.uint16).reshape(4, 5, 6, 3)
    model = _RecordingModel()

    core.segm_model_segment(
        model, video, {}, is_timelapse_model_and_data=True,
        preproc_recipe=recipe
    )

    assert model.images[0] is video


@pytest.mark.parametrize('recipe', [None, []])
def test_no_preprocessing_sends_the_frame_unchanged(recipe):
    frame = np.arange(5*6, dtype=np.uint16).reshape(5, 6)
    model = _RecordingModel()

    core.segm_model_segment(
        model, frame, {}, frame_i=0, preproc_recipe=recipe
    )

    assert model.images[0] is frame


def test_preprocessed_time_lapse_is_not_float64():
    rng = np.random.default_rng(0)
    video = rng.integers(0, 4000, size=(3, 8, 9)).astype(np.uint16)
    model = _RecordingModel()

    core.segm_model_segment(
        model, video, {}, is_timelapse_model_and_data=True,
        preproc_recipe=GAUSSIAN_RECIPE
    )

    expected = np.array([
        core.preprocess_image_from_recipe(frame, GAUSSIAN_RECIPE)
        for frame in video
    ])
    sent = model.images[0]
    assert expected.dtype == np.float64
    assert sent.dtype == np.float32
    np.testing.assert_allclose(sent, expected, rtol=1e-6)


def test_preprocess_frames_keeps_non_float64_dtypes():
    video = np.ones((2, 3, 4), dtype=np.uint8)
    dummy_recipe = [{'method': 'Dummy', 'kwargs': {}}]
    identity = {'Dummy': {'function': lambda image: image}}

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(core, 'PREPROCESS_MAPPER', identity)
        filtered = core.preprocess_frames_from_recipe(video, dummy_recipe)

    assert filtered.dtype == np.uint8
    np.testing.assert_array_equal(filtered, video)


class _Segment3DTModel:
    is_segment3DT_available = True

    def segment(self, image):
        pass

    def segment3DT(self, video):
        pass


def _init_kernel(reduce_memory_usage, backend_type='in_process'):
    kernel = cli.SegmKernel(None, None, is_cli=False)
    kernel.init_args(
        'ch', 'segm', 'fake_model', False, False, False, '', {}, {}, {},
        False, False, None, False, {}, {}, 3, 1,
        reduce_memory_usage=reduce_memory_usage,
        backend_type=backend_type,
        env_python='python' if backend_type == 'local' else None,
    )
    return kernel


@pytest.mark.parametrize('reduce_memory_usage', [False, True])
def test_cli_in_process_model_honours_reduce_memory_usage(
        monkeypatch, reduce_memory_usage
    ):
    # The CLI creates the model in `init_segm_model`, which used to replace
    # the `is_segment3DT_available` set by `init_args` with the model's
    # value, ignoring 'Reduce memory usage' of the workflow file
    monkeypatch.setattr(
        segm_models, 'import_segment_module', lambda name: SimpleNamespace()
    )
    monkeypatch.setattr(segm_models, 'getModelArgSpec', lambda module: ([], []))
    monkeypatch.setattr(
        segm_models, 'init_segm_model',
        lambda module, posData, kwargs: _Segment3DTModel()
    )
    kernel = _init_kernel(reduce_memory_usage)

    kernel.init_segm_model()

    assert isinstance(kernel.model, _Segment3DTModel)
    assert kernel.is_segment3DT_available is not reduce_memory_usage


@pytest.mark.parametrize('reduce_memory_usage', [False, True])
def test_cli_worker_model_honours_reduce_memory_usage(
        monkeypatch, reduce_memory_usage
    ):
    monkeypatch.setattr(
        segm_models, 'getModelArgSpecFromEnv', lambda name, python: ([], [], None)
    )
    kernel = _init_kernel(reduce_memory_usage, backend_type='local')
    monkeypatch.setattr(kernel, 'verify_worker_environment', lambda: None)
    monkeypatch.setattr(
        kernel, 'init_backend', lambda **kwargs: _Segment3DTModel()
    )

    kernel.init_segm_model()

    assert isinstance(kernel.model, _Segment3DTModel)
    assert kernel.is_segment3DT_available is not reduce_memory_usage


def _refused_once_then_healthy_get():
    calls = []

    def get(url, timeout):
        calls.append(url)
        if len(calls) > 1:
            return SimpleNamespace(status_code=200)
        try:
            raise OSError('connection refused')
        except OSError as err:
            # Like urllib3's `connectionpool.urlopen`, which keeps the error
            # in a local: error -> traceback -> this frame -> error
            new_e = err
            raise requests.ConnectionError(new_e) from new_e

    return get


def test_failed_health_polls_do_not_keep_the_callers_data(monkeypatch):
    backend = object.__new__(service_client.LocalBackend)
    backend.proc = SimpleNamespace(poll=lambda: None)
    backend.base_url = 'http://127.0.0.1:1'
    backend.model_name = 'fake_model'
    monkeypatch.setattr(
        service_client.requests, 'get', _refused_once_then_healthy_get()
    )
    monkeypatch.setattr(service_client.time, 'sleep', lambda seconds: None)

    class PositionData:
        pass

    def run_position():
        # Like `SegmKernel.run` from the CLI, which starts the worker while
        # it holds the arrays of the first position
        data = PositionData()
        backend._wait_for_health(cmd=[], timeout=10.0)
        return weakref.ref(data)

    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        data_ref = run_position()
        # Freed without waiting for the cyclic garbage collector
        assert data_ref() is None
    finally:
        if gc_was_enabled:
            gc.enable()
