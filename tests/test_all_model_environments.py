r"""Opt-in integration tests that install and run every public segmentation model.

PowerShell:
    python -X faulthandler -m pytest .\tests\test_all_model_environments.py -v -s --tb=short

These tests can download large dependencies and model weights. They use isolated
model environments only, never fall back to the main interpreter, and default to
CPU execution. Models requiring custom weights fail explicitly with their error.
"""

import os
import traceback
from pprint import pformat
from pathlib import Path

import numpy as np
import pytest
import tifffile

from cellacdc import myutils, _types
from cellacdc.service.client import LocalBackend
from cellacdc.service.environment import (
    get_or_create_model_env, missing_required_env,
)
from cellacdc.service.prompts import AutoHandler


MODEL_ROOT = Path(__file__).resolve().parents[1] / "cellacdc" / "models"
MODEL_NAMES = sorted(
    path.parent.name for path in MODEL_ROOT.glob("*/acdcSegment.py")
    if not path.parent.name.startswith("_")
)
IMAGE_PATH = (
    # r'C:\Users\timon\OneDrive\Helmholtz\Microscopy_Data\sample_data\single_frame.tif'
    r'path\to\your\sample_data\single_frame.tif'
)
pytestmark = pytest.mark.skipif(
    not IMAGE_PATH,
    reason="Set IMAGE_PATH to enable dependency/weight downloads and inference.",
)


@pytest.fixture(scope="module")
def sample_image():
    path = Path(IMAGE_PATH)
    assert path.is_file(), f"Sample TIFF not found: {path}"
    image = tifffile.imread(path)
    assert image.ndim == 2, f"Expected a single grayscale frame, got {image.shape}"
    assert np.isfinite(image).all(), "Sample image contains non-finite values"
    return image


@pytest.fixture(scope="module", params=MODEL_NAMES)
def model_spec(request):
    name = request.param
    print(f"\n[{name}] Preparing isolated environment", flush=True)
    try:
        python = get_or_create_model_env(
            name, auto_install=True, prompt_func=AutoHandler().ask,
        )
        assert python, f"No isolated environment returned for {name}"
        print(f"[{name}] Retrieving argument specs using {python}", flush=True)
        init, segment, url = myutils.getModelArgSpecFromEnv(name, python)
    except Exception:
        print(f"\n[{name}] Environment/spec retrieval failed:\n"
              f"{traceback.format_exc()}", flush=True)
        raise
    return name, python, init, segment, url


def _default_kwargs(params):
    return {
        param.name: param.default for param in params
        if not _types.is_widget_not_required(param)
        and not _types.is_second_channel_type(param.type)
    }


def test_model_argument_specs(model_spec):
    name, _, init, segment, url = model_spec
    for params in (init, segment):
        assert isinstance(params, list)
        assert len({p.name for p in params}) == len(params)
        for param in params:
            assert isinstance(param, myutils.ArgSpec)
            assert not isinstance(param.type, str), (
                f"{name}.{param.name}: type metadata is still plain text"
            )
    assert url is None or isinstance(url, str)
    print(f"[{name}] {len(init)} init and {len(segment)} segment parameters", flush=True)


def _synthetic_segm_data(image):
    """Simple Otsu labelling used as auxiliary segmentation input."""
    from skimage.filters import threshold_otsu
    from skimage.measure import label

    return label(image > threshold_otsu(image)).astype(np.uint32)


def test_model_segments_sample_tiff(model_spec, sample_image):
    name, python, init, segment, _ = model_spec
    if name == "Cellpose_germlineNuclei":
        pytest.skip("Model only supports 3D z-stacks with at least 5 slices.")

    init_kwargs = _default_kwargs(init)
    segment_kwargs = _default_kwargs(segment)
    if init_kwargs.pop("Auxiliary segmentation file", None) is not None:
        segm_data = _synthetic_segm_data(sample_image)
        # skip_segmentation returns segm_data[frame_i] (time-lapse layout)
        init_kwargs["segm_data"] = (
            segm_data[np.newaxis] if name == "skip_segmentation" else segm_data
        )
    missing_env = missing_required_env(name, init_kwargs)
    if missing_env:
        pytest.skip(
            f"{name} requires environment variable(s) "
            f"{', '.join(spec['name'] for spec in missing_env)}"
        )
    if name == "cellpose_v4":
        segment_kwargs["diameter"] = 30.0
    if name == "BABY":
        # The default 5z model cannot run on a single 2D frame.
        init_kwargs["model_name"] = "yeast-alcatras-brightfield-sCMOS-60x-1z"
    for params in (init_kwargs, segment_kwargs):
        if "device_type" in params:
            params["device_type"] = "cpu"
        if "gpu" in params:
            params["gpu"] = False

    if myutils._model_url(name) is not None:
        print(f"[{name}] Downloading weights if needed", flush=True)
        assert myutils.download_model(name), f"Weight download failed for {name}"

    backend = LocalBackend(
        name, env_python=python, init_model_kwargs=init_kwargs,
    )
    print(
        f"[{name}] Python: {python}\n"
        f"[{name}] Image: {IMAGE_PATH} ({sample_image.shape}, {sample_image.dtype})\n"
        f"[{name}] Init kwargs: {pformat(init_kwargs)}\n"
        f"[{name}] Segment kwargs: {pformat(segment_kwargs)}",
        flush=True,
    )
    failed = False
    try:
        print(f"[{name}] Starting worker", flush=True)
        client = backend.start()
        print(f"[{name}] Segmenting {sample_image.shape}", flush=True)
        labels = client.segment(sample_image, frame_i=0, **segment_kwargs)
        assert labels.shape == sample_image.shape, (
            f"{name}: output shape {labels.shape}, expected {sample_image.shape}"
        )
        assert np.issubdtype(labels.dtype, np.integer), (
            f"{name}: expected integer labels, got {labels.dtype}"
        )
        assert (labels >= 0).all(), f"{name}: negative labels"
        print(f"[{name}] OK: {np.unique(labels[labels > 0]).size} objects", flush=True)
    except Exception:
        failed = True
        print(
            f"\n[{name}] Segmentation failed at {backend.url}:\n"
            f"{traceback.format_exc()}",
            flush=True,
        )
        raise
    finally:
        try:
            backend.stop()
        finally:
            if failed:
                output = "\n".join(getattr(backend, "_output", []))
                print(
                    f"[{name}] Worker output (last lines):\n"
                    f"{output or '(no output captured)'}",
                    flush=True,
                )
