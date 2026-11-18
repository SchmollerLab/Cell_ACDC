import io
import json
import os
import sys
import zipfile
import numpy as np
import pytest
import requests

from cellacdc.service.schema import HealthResponse, InfoResponse, Capabilities, JobStatusResponse
from cellacdc.service.client import LocalBackend, ModelClient, RemoteBackend
from cellacdc import cli, myutils_models


def test_local_backend_forwards_worker_output(capsys):
    import collections
    from types import SimpleNamespace

    backend = LocalBackend.__new__(LocalBackend)
    backend.model_name = 'segment_anything'
    backend.proc = SimpleNamespace(stdout=io.BytesIO(
        b'SAM: restored embeddings for frame 0, z-slice 2 on cpu.\n'
        b'SAM: reusing loaded full-image embeddings; skipping image encoder.\n'
    ))
    backend._output = collections.deque(maxlen=200)

    backend._read_output()

    output = capsys.readouterr().out
    assert '[segment_anything worker] SAM: restored embeddings' in output
    assert (
        '[segment_anything worker] SAM: reusing loaded full-image embeddings; '
        'skipping image encoder.'
    ) in output
    assert len(backend._output) == 2
    assert backend._output[-1].endswith('skipping image encoder.')


def test_schema_models():
    caps = Capabilities(segment=True, segment3DT=False, second_ch_img_to_stack=True)
    info = InfoResponse(
        name="test_model",
        version="1.0.0",
        gpu=False,
        capabilities=caps
    )
    assert info.name == "test_model"
    assert info.capabilities.segment is True
    assert info.capabilities.segment3DT is False

    health = HealthResponse(status="ok", model="test_model")
    assert health.status == "ok"

    job = JobStatusResponse(job_id="123", status="completed", progress=1.0)
    assert job.status == "completed"


def test_parse_response_supports_pydantic_1_and_2():
    """Responses parse with Pydantic 2 and with an existing Pydantic 1."""
    from cellacdc.service.schema import parse_response

    data = {
        "name": "m", "capabilities": {"segment3DT": True},
    }
    info = parse_response(InfoResponse, data)
    assert info.capabilities.segment3DT is True

    pydantic_v1 = pytest.importorskip("pydantic.v1")

    class CapabilitiesV1(pydantic_v1.BaseModel):
        segment: bool = True
        segment3DT: bool = False

    class InfoResponseV1(pydantic_v1.BaseModel):
        name: str
        capabilities: CapabilitiesV1 = pydantic_v1.Field(
            default_factory=CapabilitiesV1
        )

    assert not hasattr(InfoResponseV1, "model_validate")
    info_v1 = parse_response(InfoResponseV1, data)
    assert info_v1.name == "m"
    assert info_v1.capabilities.segment3DT is True


def test_local_backend_lifecycle():
    with LocalBackend(model_name="thresholding") as client:
        assert isinstance(client, ModelClient)
        assert client.model_name == "thresholding"
        assert client.is_segment3DT_available is False

        # Create dummy image: black background with a bright square
        img = np.zeros((64, 64), dtype=np.uint8)
        img[20:44, 20:44] = 200

        # Run direct segment
        lab = client.segment(img)
        assert lab.shape == (64, 64)
        assert np.max(lab) > 0

        # thresholding has no `second_ch_img_to_stack`: like in-process,
        # the client does not invent one
        ch2 = np.ones((64, 64), dtype=np.uint8) * 50
        assert client.info.capabilities.second_ch_img_to_stack is False
        with pytest.raises(NotImplementedError):
            client.second_ch_img_to_stack(img, ch2)

        # Test job endpoints
        job_id = client.submit_job(img)
        assert isinstance(job_id, str) and len(job_id) > 0

        # Poll status
        import time
        for _ in range(30):
            status = client.get_job_status(job_id)
            if status["status"] in ("completed", "failed"):
                break
            time.sleep(0.1)

        assert status["status"] == "completed"
        res = client.get_job_result(job_id)
        assert np.array_equal(res, lab)
        # The result is removed from the worker once fetched
        with pytest.raises(requests.HTTPError):
            client.get_job_result(job_id)


def test_segm_kernel_with_local_backend():
    kernel = cli.SegmKernel(logger=None, log_path=None, is_cli=True)
    try:
        kernel.init_args(
            user_ch_name="phase_contrast",
            segm_endname="segm.npz",
            model_name="thresholding",
            do_tracking=False,
            do_postprocess=False,
            do_save=False,
            image_channel_tracker=None,
            standard_postrocess_kwargs={},
            custom_postproc_grouped_features={},
            custom_postproc_features={},
            isSegm3D=False,
            use_ROI=False,
            second_channel_name=None,
            use3DdataFor2Dsegm=False,
            model_kwargs={},
            track_params={},
            SizeT=1,
            SizeZ=1,
            backend_type="local",
            env_python=sys.executable,
        )

        assert kernel.backend_type == "local"
        assert kernel.model is None

        # Call init_segm_model (posData can be None since local backend connects over HTTP)
        kernel.init_segm_model(posData=None)

        assert kernel.model is not None
        assert isinstance(kernel.model, ModelClient)
        assert kernel.backend is not None

        # Verify segmentation works through the kernel's client model
        img = np.zeros((50, 50), dtype=np.uint8)
        img[15:35, 15:35] = 255
        lab = kernel.model.segment(img)
        assert lab.shape == (50, 50)
        assert np.max(lab) > 0
    finally:
        kernel.close()
        assert kernel.backend is None


def test_workflow_kernel_backend_general():
    base_kernel = cli._WorkflowKernel(
        logger=None, 
        log_path=None, 
        is_cli=True,
        backend_type="local",
        env_python=sys.executable
    )
    with base_kernel:
        client = base_kernel.init_backend(module_name="thresholding")
        assert client is not None
        assert isinstance(client, ModelClient)
        assert base_kernel.backend is not None
        img = np.zeros((32, 32), dtype=np.uint8)
        img[10:20, 10:20] = 255
        lab = client.segment(img)
        assert lab.shape == (32, 32)
        assert np.max(lab) > 0
    # Context manager exit should close backend
    assert base_kernel.backend is None

    # Test parse_backend_params
    wf_params = {
        "backend": {
            "type": "remote",
            "url": "http://127.0.0.1:9999",
            "env_python": "some_python"
        }
    }
    parsed = base_kernel.parse_backend_params(wf_params)
    assert parsed["backend_type"] == "remote"
    assert parsed["backend_url"] == "http://127.0.0.1:9999"
    assert parsed["env_python"] == "some_python"


def test_environment_resolution_and_naming():
    from cellacdc.service.environment import (
        get_model_env_name,
        get_install_method,
        get_venv_dir,
        get_model_python_exec,
    )
    from cellacdc import is_conda_env

    assert get_model_env_name("cellpose_v3") == "_acdc_cellpose_v3"
    assert get_model_env_name("Automatic thresholding") == "_acdc_Automatic_thresholding"
    assert get_model_env_name("StarDist") == "_acdc_StarDist"

    expected_method = "conda" if is_conda_env() else "venv"
    assert get_install_method() == expected_method

    venv_dir = get_venv_dir("thresholding", env_root="C:\\fake\\root")
    assert venv_dir == os.path.join("C:\\fake\\root", "_acdc_thresholding")


def test_check_install_package_with_python_exec():
    from cellacdc import myutils

    # numpy is installed in sys.executable
    outcome = myutils.check_install_package(
        "numpy",
        python_exec=sys.executable,
        return_outcome=True
    )
    assert outcome is None or outcome is True

    # Nonexistent package should fail or raise ModuleNotFoundError
    with pytest.raises(ModuleNotFoundError):
        myutils.check_install_package(
            "nonexistent_fake_package_xyz_123",
            python_exec=sys.executable,
            raise_on_cancel=True,
            is_cli=True
        )


@pytest.mark.parametrize("external", [False, True])
def test_install_failure_does_not_open_qt_dialog(monkeypatch, external):
    from unittest.mock import Mock
    from cellacdc import myutils

    error = RuntimeError("pip failed")
    monkeypatch.setattr(myutils, "_check_single_package", lambda *a: (False, False))
    monkeypatch.setattr(myutils, "_install_pip_package", Mock(side_effect=error))
    monkeypatch.setattr(myutils, "is_gui_running", lambda: True)
    dialog = Mock(side_effect=AssertionError("CLI must not open Qt dialogs"))
    monkeypatch.setattr(myutils, "_inform_install_package_failed", dialog)
    logger = Mock()
    with pytest.raises(RuntimeError) as raised:
        myutils.check_install_package(
            "fake_package", auto_install=True, is_cli=not external,
            python_exec="model-python" if external else None,
            logger_func=logger,
        )
    assert raised.value is error
    dialog.assert_not_called()
    assert "Installation of fake_package failed" in logger.call_args.args[0]


@pytest.mark.parametrize("external", [False, True])
@pytest.mark.parametrize("mac_arm", [False, True])
def test_tensorflow_install_preserves_main_numpy(monkeypatch, external, mac_arm):
    from unittest.mock import Mock
    from cellacdc import myutils

    python = "model-python" if external else sys.executable
    install = Mock()
    purge = Mock(side_effect=AssertionError("NumPy must not be reloaded"))
    monkeypatch.setattr(myutils.subprocess, "check_call", install)
    monkeypatch.setattr(myutils, "purge_module", purge)
    monkeypatch.setattr(myutils, "is_mac", mac_arm)
    monkeypatch.setattr(myutils.platform, "processor", lambda: "arm" if mac_arm else "x86")
    monkeypatch.setattr(myutils, "get_pip_conda_prefix", lambda: ("conda install", "pip install"))
    numpy_module = sys.modules["numpy"]
    myutils._install_tensorflow(max_version="2.14", python_exec=python)
    purge.assert_not_called()
    assert sys.modules["numpy"] is numpy_module
    if external or not mac_arm:
        install.assert_called_once_with(
            [python, "-m", "pip", "install", "-U", "tensorflow<2.14"],
            shell=False,
        )
    else:
        assert install.call_args.kwargs["shell"] is True


def test_integration_failure_prints_immediate_diagnostics(monkeypatch, capsys):
    from unittest.mock import Mock
    from tests import test_all_model_environments as integration
    from cellacdc import myutils

    backend = Mock(url="http://test-worker", _output=["worker diagnostic"])
    backend.start.return_value.segment.side_effect = RuntimeError("model traceback")
    monkeypatch.setattr(integration, "LocalBackend", Mock(return_value=backend))
    monkeypatch.setattr(integration.myutils, "_model_url", lambda name: None)
    spec = ("test_model", "model-python", [
        myutils.ArgSpec("batch_size", 8, int, None, None),
    ], [], None)
    with pytest.raises(RuntimeError, match="model traceback"):
        integration.test_model_segments_sample_tiff(spec, np.zeros((8, 8), dtype=np.uint16))
    output = capsys.readouterr().out
    for expected in ("model-python", "batch_size", "uint16", "http://test-worker",
                     "model traceback", "worker diagnostic"):
        assert expected in output
    backend.stop.assert_called_once()


def test_isolated_tensorflow_resolves_in_same_batch(monkeypatch):
    from unittest.mock import Mock
    from cellacdc import myutils

    monkeypatch.setattr(myutils, "_check_single_package", lambda *a: (False, False))
    batch = Mock()
    special = Mock(side_effect=AssertionError("Must resolve together"))
    monkeypatch.setattr(myutils, "_install_pip_package", batch)
    monkeypatch.setitem(myutils._SPECIAL_INSTALLERS, "tensorflow", special)
    myutils.check_install_package(
        [{"pkg_name": "tensorflow", "max_version": "2.13"}, "baby-seg"],
        python_exec="model-python", auto_install=True, is_cli=True,
    )
    assert [arg.strip('"') for arg in batch.call_args.args[0]] == [
        "tensorflow<2.13", "baby-seg",
    ]
    assert batch.call_args.kwargs["python_exec"] == "model-python"
    special.assert_not_called()


@pytest.mark.parametrize("installed, minimum, maximum, lower, upper, expected", [
    ("4.5", "4.16", "", True, False, False),
    ("4.16", "4.16", "", True, False, True),
    ("4.16", "4.16", "", False, False, False),
    ("2.13", "", "2.13", False, False, False),
    ("2.13", "", "2.13", False, True, True),
    ("2.14", "", "2.13", False, True, False),
])
def test_external_package_version_bounds(monkeypatch, installed, minimum, maximum,
                                        lower, upper, expected):
    from types import SimpleNamespace
    from cellacdc import myutils

    monkeypatch.setattr(myutils.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=installed,
    ))
    spec = dict(pkg_name="test", import_pkg_name="test", exact_version="",
                min_version=minimum, max_version=maximum,
                include_lower_version=lower, include_higher_version=upper)
    satisfied, _ = myutils._check_single_package(spec, False, "model-python")
    assert satisfied is expected


def test_parameter_filtering_never_constructs_widgets():
    import typing
    from cellacdc import myutils, _types
    from tests.test_all_model_environments import _default_kwargs

    class WidgetType:
        isWidget = True

        def __init__(self):
            raise AssertionError("Metadata inspection must not create widgets")

    params = [
        myutils.ArgSpec("model_type", "Phase contrast", _types.YeazV2ModelType, None, None),
        myutils.ArgSpec("custom", "default", WidgetType, None, None),
        myutils.ArgSpec("hidden", False, _types.NotGUIParam, None, None),
        myutils.ArgSpec("optional_hidden", None, typing.Optional[_types.NotGUIParam], None, None),
        myutils.ArgSpec("second_channel", None, typing.Optional[_types.SecondChannelImage], None, None),
    ]
    assert _default_kwargs(params) == {
        "model_type": "Phase contrast", "custom": "default",
    }
    assert _types.is_widget_not_required(params[2])
    assert _types.is_widget_not_required(params[3])
    assert not _types.is_widget_not_required(params[0])


@pytest.mark.parametrize("already_complete", [False, True])
def test_model_setup_steps_use_target_python(monkeypatch, already_complete):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from cellacdc.service import environment

    config = {"setup_steps": [{
        "name": "custom setup", "check": "import custom_package",
        "commands": [["-m", "pip", "install", "custom-source"], ["-c", "print('setup')"]],
    }]}
    monkeypatch.setattr(environment, "load_model_env_config", lambda name: config)
    results = [SimpleNamespace(returncode=0 if already_complete else 1, stderr="missing", stdout=""),
               SimpleNamespace(returncode=0, stderr="", stdout="")]
    check = Mock(side_effect=results)
    run = Mock()
    monkeypatch.setattr(environment.subprocess, "run", check)
    monkeypatch.setattr(environment, "run_streaming", run)
    logger = Mock()
    environment.ensure_model_dependencies("test", "model-python", logger_func=logger)
    assert check.call_args.args[0] == [
        "model-python", "-c", "import custom_package", "--server"
    ]
    if already_complete:
        run.assert_not_called()
    else:
        assert [call.args[0] for call in run.call_args_list] == [
            ["model-python", "-m", "pip", "install", "custom-source"],
            ["model-python", "-c", "print('setup')"],
        ]
        assert all(call.kwargs["check"] for call in run.call_args_list)


def test_model_setup_step_failure_propagates(monkeypatch):
    from unittest.mock import Mock
    from cellacdc.service import environment

    monkeypatch.setattr(environment, "load_model_env_config", lambda name: {
        "setup_steps": [{"name": "build", "commands": [["-m", "pip", "install", "source"]]}],
    })
    error = RuntimeError("compiler missing")
    monkeypatch.setattr(environment, "run_streaming", Mock(side_effect=error))
    with pytest.raises(RuntimeError, match="compiler missing"):
        environment.ensure_model_dependencies("test", "model-python")


def test_get_model_arg_spec_from_env():
    from cellacdc import myutils

    init_params, segment_params, url = myutils.getModelArgSpecFromEnv(
        "thresholding", python_exec=sys.executable
    )
    assert isinstance(init_params, list)
    assert isinstance(segment_params, list)
    param_names = [p.name for p in segment_params]
    assert "gauss_sigma" in param_names
    assert "threshold_method" in param_names
    assert "segment_3D_volume" in param_names
    types = {p.name: p.type for p in segment_params}
    assert types["gauss_sigma"] is float
    assert types["threshold_method"] is str
    assert types["segment_3D_volume"] is bool


def _patch_model_execution(monkeypatch, use_model_env=True, **overrides):
    from unittest.mock import Mock
    from cellacdc.service import model_execution

    mocks = dict(
        get_or_create_model_env=Mock(return_value="model-python"),
        getModelArgSpecFromEnv=Mock(return_value=([], [], "help")),
        import_segment_module=Mock(),
        getModelArgSpec=Mock(return_value=([], [])),
        init_segm_model=Mock(),
        LocalBackend=Mock(),
        model_installed_in_main_env=Mock(return_value=False),
        ensure_required_env=Mock(return_value=True),
    )
    mocks.update(overrides)
    backend_factory = mocks["LocalBackend"]
    spec_mock = mocks["getModelArgSpecFromEnv"]
    original_factory = backend_factory.side_effect

    def make_prepared_backend(**kwargs):
        backend = (
            original_factory(**kwargs) if original_factory is not None
            else backend_factory.return_value
        )
        backend.prepare.return_value = spec_mock.return_value
        backend.prepare.side_effect = spec_mock.side_effect
        return backend

    backend_factory.side_effect = make_prepared_backend
    monkeypatch.setattr(model_execution, "USE_MODEL_ENV", use_model_env)
    for name in (
            "get_or_create_model_env", "LocalBackend",
            "model_installed_in_main_env", "ensure_required_env",
        ):
        monkeypatch.setattr(model_execution, name, mocks[name])
    for name in (
            "getModelArgSpecFromEnv", "import_segment_module",
            "getModelArgSpec", "init_segm_model",
        ):
            monkeypatch.setattr(model_execution.myutils, name, mocks[name])
    monkeypatch.setattr(
        model_execution.myutils, "resolve_segm_model_init_data",
        lambda posData, init_kwargs: None,
    )
    return mocks


@pytest.mark.parametrize("use_model_env", [False, True])
def test_gui_model_execution_modes(monkeypatch, use_model_env):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from cellacdc.service.model_execution import ModelExecution

    spec = ([], [], "help")
    module = SimpleNamespace(url_help=lambda: "help")
    model = object()
    backend = Mock(url="worker-url")
    backend.start.return_value = model
    mocks = _patch_model_execution(
        monkeypatch, use_model_env,
        getModelArgSpecFromEnv=Mock(return_value=spec),
        import_segment_module=Mock(return_value=module),
        init_segm_model=Mock(return_value=model),
        LocalBackend=Mock(return_value=backend),
    )
    posData = object()
    execution = ModelExecution(
        "test_model", run_in_background=lambda f, *a: f(*a), log_func=Mock()
    )

    assert execution.prepare() == spec
    assert execution.init_model({}, posData) is model
    if use_model_env:
        mocks["get_or_create_model_env"].assert_called_once_with("test_model")
        mocks["getModelArgSpecFromEnv"].assert_not_called()
        backend.prepare.assert_called_once_with()
        mocks["import_segment_module"].assert_not_called()
        mocks["getModelArgSpec"].assert_not_called()
        mocks["init_segm_model"].assert_not_called()
        backend.start.assert_called_once()
        assert execution.backend_type == "local"
        assert execution.backend_url == "worker-url"
        assert execution.acdcSegment is None
    else:
        mocks["get_or_create_model_env"].assert_not_called()
        mocks["getModelArgSpecFromEnv"].assert_not_called()
        mocks["LocalBackend"].assert_not_called()
        mocks["import_segment_module"].assert_called_once_with("test_model")
        mocks["init_segm_model"].assert_called_once_with(module, posData, {})
        assert execution.backend_type == "in_process"
        assert execution.env_python is None
        assert execution.backend is None
        assert execution.backend_url is None



def test_promptable_model_execution_uses_isolated_worker(monkeypatch):
    from unittest.mock import Mock
    from cellacdc.service.model_execution import ModelExecution

    spec = (['init'], ['segment'], 'help')
    backend = Mock(url='worker-url')
    backend.start.return_value = object()
    mocks = _patch_model_execution(
        monkeypatch,
        getModelArgSpecFromEnv=Mock(return_value=spec),
        LocalBackend=Mock(return_value=backend),
    )
    execution = ModelExecution(
        'segment_anything', model_kind='promptable', log_func=Mock()
    )

    assert execution.prepare() == spec
    model = execution.init_model({'model_type': 'Small'}, posData=object())

    mocks['getModelArgSpecFromEnv'].assert_not_called()
    backend.prepare.assert_called_once_with()
    mocks['LocalBackend'].assert_called_once_with(
        model_name='segment_anything',
        env_python='model-python',
        init_model_kwargs={},
        model_kind='promptable',
    )
    assert model is backend.start.return_value


def test_model_execution_restarts_for_required_env_change(monkeypatch):
    from unittest.mock import Mock
    from cellacdc.service import model_execution
    from cellacdc.service.model_execution import ModelExecution

    backends = []

    def make_backend(**kwargs):
        backend = Mock(url=f'url-{len(backends)}')
        backend.start.return_value = object()
        backend.is_running.return_value = True
        backends.append(backend)
        return backend

    _patch_model_execution(
        monkeypatch, LocalBackend=Mock(side_effect=make_backend)
    )
    env_keys = iter(['first', 'first', 'second'])
    monkeypatch.setattr(
        model_execution, 'required_env_fingerprint',
        lambda model_name: next(env_keys),
    )
    execution = ModelExecution('test_model', log_func=Mock())
    execution.prepare()

    first = execution.init_model({'weights': 'pretrained'})
    execution.release(first)
    second = execution.init_model({'weights': 'pretrained'})

    assert second is not first
    assert len(backends) == 2
    backends[0].reinitialize.assert_not_called()


@pytest.mark.parametrize("reason", ["cancel", "stop", "dead", "env_changed", "init_failed"])
def test_prepared_model_worker_lifecycle(monkeypatch, reason):
    from unittest.mock import Mock
    from cellacdc.service import model_execution

    backends = []

    def make_backend(**kwargs):
        backend = Mock()
        backend.start.return_value = object()
        backend.is_running.return_value = True
        backends.append(backend)
        return backend

    mocks = _patch_model_execution(
        monkeypatch, LocalBackend=Mock(side_effect=make_backend)
    )
    env_key = Mock(return_value="initial")
    monkeypatch.setattr(model_execution, "required_env_fingerprint", env_key)
    execution = model_execution.ModelExecution("test_model", log_func=Mock())
    spec = execution.prepare()
    prepared = backends[0]
    prepared.start.assert_not_called()

    if reason == "stop":
        execution.stop()
        execution.stop()
    elif reason == "cancel":
        execution.cancel_preparation()
        execution.cancel_preparation()
        assert execution.prepare() is spec
        assert len(backends) == 1
        execution.init_model({})
        assert len(backends) == 2
    elif reason == "dead":
        prepared.is_running.return_value = False
        execution.init_model({})
        assert len(backends) == 2
    elif reason == "env_changed":
        env_key.return_value = "new-token"
        execution.init_model({})
        assert len(backends) == 2
    else:
        prepared.start.side_effect = RuntimeError("bad initialization")
        with pytest.raises(RuntimeError, match="bad initialization"):
            execution.init_model({})
        assert execution.backend is None
        prepared.start.side_effect = None
        execution.init_model({})
        assert len(backends) == 2

    prepared.stop.assert_called_once()
    assert execution._prepared_backend is None
    mocks["getModelArgSpecFromEnv"].assert_not_called()
    execution.stop()


def test_cellpose_v3_environment_matches_wrapper_minimum():
    from cellacdc.models._env_config import MODEL_ENVS

    package = MODEL_ENVS['cellpose_v3']['packages'][0]
    assert package['min_version'] == '3.1.1.2'


def test_model_execution_caches_spec_and_manages_workers(monkeypatch):
    from unittest.mock import Mock
    import numpy as np
    from cellacdc.service.model_execution import ModelExecution

    backends = []

    def make_backend(**kwargs):
        backend = Mock(url=f"url-{len(backends)}")
        backend.start.return_value = object()
        backend.is_running.return_value = True
        backends.append(backend)
        return backend

    mocks = _patch_model_execution(
        monkeypatch, LocalBackend=Mock(side_effect=make_backend)
    )
    execution = ModelExecution("test_model", log_func=Mock())
    with pytest.raises(RuntimeError):
        execution.init_model({})

    assert execution.prepare() is execution.prepare()
    mocks["get_or_create_model_env"].assert_called_once()
    mocks["getModelArgSpecFromEnv"].assert_not_called()
    backends[0].prepare.assert_called_once()

    # Same parameters (also array contents) share one worker
    first = execution.init_model({"a": 1, "arr": np.zeros(3)})
    shared = execution.init_model({"arr": np.zeros(3), "a": 1})
    assert shared is first and len(backends) == 1
    # Different parameters at the same time need another worker
    second = execution.init_model({"a": 2})
    assert second is not first and len(backends) == 2
    assert execution.backend_url == "url-1"

    # Released workers keep running and are re-initialized, not restarted
    execution.release(first)
    execution.release(shared)
    backends[0].stop.assert_not_called()
    third = execution.init_model({"a": 3})
    assert third is first and len(backends) == 2
    backends[0].reinitialize.assert_called_once_with({"a": 3})

    # `replace` re-initializes the replaced worker for the new parameters
    fourth = execution.init_model({"a": 4}, replace=second)
    assert fourth is second and len(backends) == 2
    backends[1].reinitialize.assert_called_once_with({"a": 4})

    # Failed re-initialization keeps the replaced instance usable
    backends[1].reinitialize.side_effect = RuntimeError("bad kwargs")
    with pytest.raises(RuntimeError):
        execution.init_model({"a": 5}, replace=fourth)
    assert execution.init_model({"a": 4}) is fourth
    execution.release(fourth)

    # Only MAX_IDLE_WORKERS unused workers are kept
    execution.release(fourth)
    execution.release(third)
    assert [b.stop.call_count for b in backends] == [0, 1]

    # Dead workers are not reused
    backends[0].is_running.return_value = False
    execution.init_model({"a": 3})
    assert len(backends) == 3
    backends[0].stop.assert_called_once()

    execution.stop()
    backends[2].stop.assert_called_once()
    assert execution.backend is None


@pytest.mark.parametrize('reuse', ['released', 'replace', 'concurrent'])
def test_model_execution_channel_mode_isolates_mutable_state(monkeypatch, reuse):
    from unittest.mock import Mock
    from cellacdc.service.model_execution import ModelExecution

    backends = []

    class Model:
        def __init__(self, weights, is_rgb=False):
            self.is_rgb = is_rgb

        def second_ch_img_to_stack(self, first, second):
            self.is_rgb = True
            return np.stack([first, second, np.zeros_like(first)], axis=-1)

        def segment(self, image):
            if self.is_rgb:
                assert image.ndim == 3 and image.shape[-1] == 3
                image = image[..., 0]
            else:
                assert image.ndim == 2
            return np.ones_like(image, dtype=np.uint32)

    def make_backend(**kwargs):
        model = None
        backend = Mock()
        backend.init_model_kwargs = kwargs['init_model_kwargs']

        def start():
            nonlocal model
            model = Model(**backend.init_model_kwargs)
            return model

        def initialize(init_kwargs):
            model.__init__(**init_kwargs)

        backend.start.side_effect = start
        backend.is_running.return_value = True
        backend.reinitialize.side_effect = initialize
        backends.append(backend)
        return backend

    _patch_model_execution(
        monkeypatch, LocalBackend=Mock(side_effect=make_backend)
    )
    execution = ModelExecution('test_model', log_func=Mock())
    execution.prepare()
    kwargs = {'weights': 'pretrained'}
    rgb = execution.init_model(kwargs, is_rgb=True)
    assert rgb.is_rgb is True
    image = np.zeros((8, 8), dtype=np.uint8)
    rgb_image = rgb.second_ch_img_to_stack(image, image)
    np.testing.assert_array_equal(rgb.segment(rgb_image), np.ones((8, 8)))
    if reuse == 'released':
        execution.release(rgb)
    gray = execution.init_model(
        kwargs, is_rgb=False, replace=rgb if reuse == 'replace' else None
    )
    assert gray.is_rgb is False
    np.testing.assert_array_equal(gray.segment(image), np.ones((8, 8)))
    assert kwargs == {'weights': 'pretrained'}
    if reuse == 'concurrent':
        assert gray is not rgb
        assert rgb.is_rgb is True
        assert len(backends) == 2
    else:
        assert gray is rgb
        backends[0].reinitialize.assert_called_once_with({
            'weights': 'pretrained', 'is_rgb': False,
        })
    assert execution.init_model(kwargs, is_rgb=False) is gray
    execution.stop()


@pytest.mark.parametrize('is_rgb', [False, True])
def test_in_process_model_execution_initializes_channel_mode(monkeypatch, is_rgb):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from cellacdc.service.model_execution import ModelExecution

    module = SimpleNamespace(url_help=lambda: None)
    mocks = _patch_model_execution(
        monkeypatch, use_model_env=False,
        import_segment_module=Mock(return_value=module),
    )
    execution = ModelExecution('test_model', log_func=Mock())
    execution.prepare()
    kwargs = {'is_rgb': not is_rgb}
    pos_data = object()
    execution.init_model(kwargs, pos_data, is_rgb=is_rgb)
    mocks['init_segm_model'].assert_called_once_with(
        module, pos_data, {'is_rgb': is_rgb}
    )
    assert kwargs == {'is_rgb': not is_rgb}


@pytest.mark.parametrize('scenario', [
    'changed', 'unchanged', 'cancelled', 'other_consumer', 'stored_slot',
])
def test_magic_labeller_reinitialization_releases_ownership(monkeypatch, scenario):
    from contextlib import nullcontext
    from types import MethodType, SimpleNamespace
    from unittest.mock import Mock
    from cellacdc import gui
    from cellacdc.service.model_execution import ModelExecution

    def make_backend(**kwargs):
        backend = Mock()
        backend.start.return_value = SimpleNamespace(model_name='test_model')
        backend.is_running.return_value = True
        return backend

    _patch_model_execution(
        monkeypatch, LocalBackend=Mock(side_effect=make_backend)
    )
    execution = ModelExecution('test_model', log_func=Mock())
    execution.prepare()
    old = execution.init_model({'value': 0}, is_rgb=False)
    if scenario == 'other_consumer':
        assert execution.init_model({'value': 0}, is_rgb=False) is old
    window = SimpleNamespace(
        app=Mock(), logger=Mock(), labelRoiModel=old, models=[old],
        modelExecutions={'test_model': execution},
        labelRoiViewCurrentModelAction=Mock(),
        getModelExecution=lambda name: execution,
        modelExecutionBusy=lambda text: nullcontext(),
    )
    window.initSegmModel = MethodType(gui.guiWin.initSegmModel, window)
    window.releaseSegmModel = MethodType(gui.guiWin.releaseSegmModel, window)
    dialog = SimpleNamespace(cancel=False, selectedModel='test_model', exec_=Mock())
    monkeypatch.setattr(gui.apps, 'QDialogSelectModel', lambda **kwargs: dialog)

    def repeat(**kwargs):
        if scenario == 'cancelled':
            return None
        value = 0 if scenario == 'unchanged' else len(values)
        values.append(value)
        model = window.initSegmModel(
            'test_model', {'value': value}, posData=object(),
            replace=window.models[0],
        )
        if scenario == 'stored_slot':
            # Switching model names leaves the old model in its original slot.
            window.models.append(model)
        else:
            window.models[0] = model
        return model

    values = [0]
    window.repeatSegm = repeat
    for _ in range(4 if scenario == 'changed' else 1):
        assert gui.guiWin.initLabelRoiModel(window) is (scenario == 'cancelled')
    old_worker = next(
        (w for w in execution._workers if w.model is old), None
    )
    if scenario == 'cancelled':
        assert window.labelRoiModel is old
        assert old_worker.users == 1
    elif scenario in ('unchanged', 'other_consumer', 'stored_slot'):
        assert old_worker.users == 1
    if scenario == 'changed':
        assert len(execution._workers) <= 2
        assert sum(w.users for w in execution._workers) == 1
    if scenario == 'other_consumer':
        execution.release(old)
        assert old_worker.users == 0
    execution.stop()


def test_model_execution_missing_required_env_cancels(monkeypatch):
    from unittest.mock import Mock
    from cellacdc.service.model_execution import ModelExecution

    mocks = _patch_model_execution(
        monkeypatch, ensure_required_env=Mock(return_value=False)
    )
    ask = Mock()
    execution = ModelExecution("test_model", prompt_func=ask, log_func=Mock())
    execution.prepare()
    assert execution.init_model({}) is None
    mocks["ensure_required_env"].assert_called_once_with(
        "test_model", {}, prompt_func=ask
    )
    backend = mocks["LocalBackend"].return_value
    backend.start.assert_not_called()
    backend.stop.assert_called_once()


@pytest.mark.parametrize("installed_version, expected", [
    ("3.1.1.2", True), ("3.1.1.1", False), ("4.0", False), (None, False),
])
def test_main_model_detection_without_imports(monkeypatch, installed_version, expected):
    import importlib.metadata
    from unittest.mock import Mock
    from cellacdc.service import environment

    monkeypatch.setattr(environment, "load_model_env_config", lambda name: {
        "packages": [{
            "pkg_name": "fake_model", "min_version": "3.1.1.2",
            "include_lower_version": True, "max_version": "4.0",
        }],
    })

    def version(name):
        assert name == "fake_model"
        if installed_version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return installed_version

    monkeypatch.setattr(importlib.metadata, "version", version)
    import_model = Mock(side_effect=AssertionError("Must not import model"))
    from cellacdc import myutils
    monkeypatch.setattr(myutils, "import_segment_module", import_model)
    assert environment.model_installed_in_main_env("test") is expected
    import_model.assert_not_called()


@pytest.mark.parametrize("answer", ["isolated", "main", "cancel"])
def test_existing_main_model_environment_choice(monkeypatch, answer):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from cellacdc.service.model_execution import ModelExecution

    seen = []

    def ask(prompt):
        seen.append(prompt)
        if answer == "cancel":
            return None
        return prompt.choices[0 if answer == "isolated" else 1]

    mocks = _patch_model_execution(
        monkeypatch,
        model_installed_in_main_env=Mock(return_value=True),
        getModelArgSpecFromEnv=Mock(return_value=([], [], None)),
        import_segment_module=Mock(
            return_value=SimpleNamespace(url_help=lambda: None)
        ),
    )
    execution = ModelExecution(
        "test_model", run_in_background=lambda f, *a: f(*a),
        prompt_func=ask, log_func=Mock(),
    )
    result = execution.prepare()
    assert seen[0].default == seen[0].choices[0]
    assert "version conflicts" in seen[0].text
    resolve = mocks["get_or_create_model_env"]
    import_model = mocks["import_segment_module"]
    if answer == "isolated":
        resolve.assert_called_once()
        import_model.assert_not_called()
        assert execution.backend_type == "local"
    elif answer == "main":
        resolve.assert_not_called()
        import_model.assert_called_once()
        assert execution.backend_type == "in_process"
    else:
        assert result is None
        assert execution.spec is None
        resolve.assert_not_called()
        import_model.assert_not_called()


def test_main_model_detection_custom_config_and_frameworks(monkeypatch):
    import importlib.metadata
    from unittest.mock import Mock
    from cellacdc.service import environment

    config = {
        "torch": True,
        "packages": ["unused_install_source"],
        "main_env_packages": [{
            "pkg_name": "import_name", "distribution_name": "distribution-name",
        }],
    }
    monkeypatch.setattr(environment, "load_model_env_config", lambda name: config)
    version = Mock(return_value="1.0")
    monkeypatch.setattr(importlib.metadata, "version", version)
    assert environment.model_installed_in_main_env("test")
    assert [call.args[0] for call in version.call_args_list] == [
        "distribution-name", "torch", "torchvision",
    ]
    config.clear()
    version.reset_mock()
    assert not environment.model_installed_in_main_env("test")
    version.assert_not_called()


@pytest.mark.parametrize("failure_stage", ["environment", "spec", "startup"])
def test_gui_isolated_mode_never_falls_back(monkeypatch, failure_stage):
    from unittest.mock import Mock
    from cellacdc.service.model_execution import ModelExecution

    error = RuntimeError("original failure")
    resolve = Mock(return_value="model-python")
    remote_spec = Mock(return_value=([], [], None))
    backend = Mock()
    backend.start.side_effect = error
    if failure_stage == "environment":
        resolve.side_effect = error
    elif failure_stage == "spec":
        remote_spec.side_effect = error
    mocks = _patch_model_execution(
        monkeypatch,
        get_or_create_model_env=resolve,
        getModelArgSpecFromEnv=remote_spec,
        LocalBackend=Mock(return_value=backend),
    )
    execution = ModelExecution(
        "test_model", run_in_background=lambda f, *a: f(*a), log_func=Mock()
    )

    with pytest.raises(RuntimeError) as raised:
        execution.prepare()
        execution.init_model({})
    assert raised.value is error
    mocks["import_segment_module"].assert_not_called()
    mocks["init_segm_model"].assert_not_called()
    if failure_stage == "startup":
        backend.stop.assert_called_once()
        assert execution.backend is None


def test_model_param_types_roundtrip():
    import json
    import typing
    from cellacdc import myutils, _types

    for original in (
            bool, int, float, str, os.PathLike, type(None), _types.Vector,
            _types.FolderPath, _types.SecondChannelImage, _types.NotGUIParam,
            _types.YeazV2ModelType, _types.CsvFilePath,
        ):
        encoded = myutils_models._serialize_model_param_type(original)
        restored = myutils._deserialize_model_param_type(json.loads(json.dumps(encoded)))
        assert restored is original

    class Choices:
        values = ['cyto3', 'nuclei']
        is_exclusive_with = ['model_path']
        default_exclusive = 'Using custom model'

    encoded = myutils_models._serialize_model_param_type(Choices)
    restored = myutils._deserialize_model_param_type(json.loads(json.dumps(encoded)))
    assert restored().values == Choices.values
    assert restored().is_exclusive_with == Choices.is_exclusive_with
    assert restored().default_exclusive == Choices.default_exclusive
    assert restored.cast_dtype('cyto3') == 'cyto3'
    optional = myutils_models._serialize_model_param_type(typing.Optional[_types.SecondChannelImage])
    assert _types.is_second_channel_type(myutils._deserialize_model_param_type(optional))
    literal = myutils_models._serialize_model_param_type(typing.Literal['cpu', 'gpu'])
    assert myutils._deserialize_model_param_type(literal)().values == ['cpu', 'gpu']


def test_model_param_metadata_creates_typed_gui_widgets():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication
    from cellacdc import apps, myutils, widgets, _types

    app = QApplication.instance() or QApplication([])

    class Choices:
        values = ['cpu', 'gpu']

    definitions = [
        ('enabled', False, bool, widgets.Toggle),
        ('batch_size', 8, int, widgets.SpinBox),
        ('diameter', 0.0, float, widgets.FloatLineEdit),
        ('model_path', '', os.PathLike, widgets.filePathControl),
        ('device_type', 'cpu', Choices, widgets.AlphaNumericComboBox),
    ]
    params = [
        myutils.ArgSpec(name, default, myutils._deserialize_model_param_type(
            myutils_models._serialize_model_param_type(param_type)
        ), None, None)
        for name, default, param_type, _ in definitions
    ]
    params.append(myutils.ArgSpec('hidden', False, _types.NotGUIParam, None, None))
    dialog = apps.QDialogModelParams(
        params, [], 'test_model', initLastParams=False,
        addPreProcessParams=False, addPostProcessParams=False,
    )
    try:
        controls = {arg.name: arg.widget for arg in dialog.init_argsWidgets}
        assert 'hidden' not in controls
        for name, _, _, widget_type in definitions:
            assert isinstance(controls[name], widget_type)
        assert controls['device_type'].count() == 2
        assert controls['enabled'].isChecked() is False
        assert controls['batch_size'].value() == 8
    finally:
        dialog.close()


def test_create_and_link_venv(tmp_path):
    import subprocess
    from cellacdc.service.environment import (
        create_model_env,
        get_or_create_model_env,
        get_model_python_exec,
        link_cellacdc_to_env,
    )

    env_root = str(tmp_path)
    model_name = "test_mini_model"

    # Test creating venv without downloading packages
    py_exec = create_model_env(
        model_name,
        install_method="venv",
        env_root=env_root,
        ensure_deps=False,
    )
    assert os.path.isfile(py_exec)

    # Test get_model_python_exec
    discovered = get_model_python_exec(
        model_name,
        install_method="venv",
        env_root=env_root
    )
    assert discovered == py_exec

    # Test that cellacdc module is discoverable inside the created venv via .pth
    res = subprocess.run(
        [py_exec, "-c", "import importlib.util; spec = importlib.util.find_spec('cellacdc'); print(spec is not None)"],
        capture_output=True,
        text=True
    )
    assert res.returncode == 0
    assert res.stdout.strip() == "True"

    # Test get_or_create_model_env reuses the existing env
    reused = get_or_create_model_env(
        model_name,
        install_method="venv",
        env_root=env_root,
        ensure_deps=False,
    )
    assert reused == py_exec


# def test_server_mode_skips_installs_and_torch_prompt(monkeypatch):
#     from cellacdc import myutils
#     from cellacdc.service.environment import ensure_torch

#     import cellacdc

#     monkeypatch.setattr(cellacdc, "RUNNING_AS_SERVER", True)
#     assert myutils.is_server_mode()
#     with pytest.raises(ModuleNotFoundError, match="_env_config.py"):
#         myutils.check_install_package("definitely_not_a_pkg_xyz")

#     # The client asks on behalf of the worker; cancelling installs nothing
#     seen = {}

#     def prompt(p):
#         seen.update(choices=p.choices, default=p.default)
#         return None

#     monkeypatch.setattr(cellacdc, "RUNNING_AS_SERVER", False)
#     fake_python = sys.executable
#     import subprocess
#     has_torch = subprocess.run(
#         [fake_python, "-c", "import torch, torchvision"], capture_output=True
#     ).returncode == 0
#     if not has_torch:
#         assert ensure_torch(fake_python, prompt_func=prompt) is False
#         assert seen["default"] in seen["choices"]
#     assert "CPU" in get_torch_choices()


@pytest.mark.parametrize("server_mode", [False, True])
def test_headless_server_config_import(server_mode):
    import subprocess

    script = """
import sys
from importlib.abc import MetaPathFinder

class NoQt(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'qtpy' or fullname.startswith('qtpy.'):
            raise ModuleNotFoundError('Qt disabled for headless regression')

sys.meta_path.insert(0, NoQt())
import cellacdc
from cellacdc import myutils
assert not cellacdc.GUI_INSTALLED
assert cellacdc.RUNNING_AS_SERVER == ('--server' in sys.argv)
assert hasattr(myutils, 'config') == cellacdc.RUNNING_AS_SERVER
assert 'cellacdc.widgets' not in sys.modules
assert 'cellacdc.apps' not in sys.modules
if cellacdc.RUNNING_AS_SERVER:
    assert myutils.config.ConfigParser().sections() == []
"""
    args = ["--server"] if server_mode else []
    # The old environment variable must have no effect any more
    env = dict(os.environ, ACDC_SERVER_MODE="1")
    result = subprocess.run(
        [sys.executable, "-c", script, *args],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_prompts_handlers_and_roundtrip():
    from cellacdc.service import prompts

    p = prompts.choice("T", "x", ["CPU", "GPU"], default="GPU")
    assert prompts.Prompt.from_dict(p.to_dict()) == p
    auto = prompts.AutoHandler()
    assert auto.ask(p) == "GPU"
    assert auto.ask(prompts.confirm("T", "x", default=False)) is False

    answers = iter(["2", "n", "hello"])
    cli = prompts.CliHandler(input_func=lambda _: next(answers), print_func=lambda *a: None)
    assert cli.ask(p) == "GPU"
    assert cli.ask(prompts.confirm("T", "x")) is False
    assert cli.ask(prompts.text_input("T", "x", default="d")) == "hello"


def _click_after_exec(monkeypatch, cls, action):
    """Replace a blocking `exec_` with a simulated user interaction."""
    monkeypatch.setattr(cls, "exec_", lambda self: action(self))


def test_qt_handler_from_worker_thread(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    import threading
    from qtpy.QtWidgets import QApplication
    from qtpy.QtCore import QEventLoop, QTimer
    from cellacdc import widgets
    from cellacdc.service import prompts

    app = QApplication.instance() or QApplication([])
    _click_after_exec(
        monkeypatch, widgets.myMessageBox,
        lambda self: self.clickButtonFromText("CPU"),
    )
    handler = prompts.QtHandler()
    out = []
    thread = threading.Thread(
        target=lambda: out.append(handler.ask(prompts.choice("T", "x", ["CPU", "GPU"])))
    )
    thread.start()
    loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda: out and loop.quit())
    timer.start(20)
    QTimer.singleShot(5000, loop.quit)
    loop.exec_()
    thread.join()
    assert out == ["CPU"]


def test_qt_handler_uses_acdc_dialogs(monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication, QLineEdit
    from cellacdc import apps, widgets
    from cellacdc.service import prompts

    app = QApplication.instance() or QApplication([])
    handler = prompts.QtHandler()

    # Few choices -> buttons; cancelling returns None
    _click_after_exec(
        monkeypatch, widgets.myMessageBox,
        lambda self: self.clickButtonFromText("Cancel"),
    )
    assert handler.ask(prompts.choice("T", "x", ["A", "B"])) is None

    # Confirm -> ok button
    _click_after_exec(
        monkeypatch, widgets.myMessageBox,
        lambda self: self.clickButtonFromText("Yes"),
    )
    assert handler.ask(prompts.confirm("T", "x", ok_text="Yes")) is True

    # Many choices -> QDialogCombobox, preselecting the default
    seen = {}
    def pick(self):
        seen["default"] = self.ComboBox.currentText()
        self.ComboBox.setCurrentText("D")
        self.ok_cb()
    _click_after_exec(monkeypatch, apps.QDialogCombobox, pick)
    choices = ["A", "B", "C", "D"]
    assert handler.ask(prompts.choice("T", "x", choices, default="C")) == "D"
    assert seen["default"] == "C"

    # Secret text -> QLineEditDialog with masked input
    def type_token(self):
        seen["echo"] = self.entryWidget.echoMode()
        self.entryWidget.setText("tok")
        self.ok_cb(None)
    _click_after_exec(monkeypatch, apps.QLineEditDialog, type_token)
    prompt = prompts.text_input("T", "token:", extra={"secret": True})
    assert handler.ask(prompt) == "tok"
    assert seen["echo"] == QLineEdit.EchoMode.Password

    _click_after_exec(
        monkeypatch, apps.QLineEditDialog, lambda self: self.cancel_cb(None)
    )
    assert handler.ask(prompts.text_input("T", "x")) is None


# def test_cellacdc_cli_check_and_model_deps():
#     from cellacdc.service.environment import (
#         is_cellacdc_cli_installed,
#         ensure_model_dependencies,
#         load_model_env_config,
#         detect_torch_index,
#     )

#     # In the current test environment (where cellacdc is installed), it should return True
#     assert is_cellacdc_cli_installed(sys.executable) is True

#     # Config is read from models/_env_config.py without importing the model
#     assert load_model_env_config("StarDist")["python"] == "3.11"
#     assert load_model_env_config("stardist")["python"] == "3.11"
#     assert load_model_env_config("cellpose_v3")["torch"] is True
#     assert load_model_env_config("nonexistent_model_12345") == {}
#     assert "cellacdc.models.StarDist" not in sys.modules

#     index = detect_torch_index()
#     assert index is None or index.startswith("https://download.pytorch.org/whl/")

#     # Calling ensure_model_dependencies for unknown model should safely no-op
#     ensure_model_dependencies("nonexistent_model_12345", sys.executable)

def test_required_env_prompts_and_sets_session_value(monkeypatch):
    from cellacdc.service import environment
    monkeypatch.setenv("DEEPCELL_ACCESS_TOKEN", "placeholder")
    monkeypatch.delenv("DEEPCELL_ACCESS_TOKEN")
    asked = []

    def fake_prompt(prompt):
        asked.append(prompt)
        return "  secret-token  "

    assert environment.ensure_required_env("cellsam", {}, fake_prompt)
    assert asked[0].kind == "text" and asked[0].extra["secret"] is True
    assert "users.deepcell.org" in asked[0].text
    assert environment.os.environ["DEEPCELL_ACCESS_TOKEN"] == "secret-token"
    asked.clear()
    assert environment.ensure_required_env("cellsam", {}, fake_prompt)
    assert asked == []


def test_required_env_cancel_and_custom_model_path(monkeypatch):
    from cellacdc.service import environment
    monkeypatch.delenv("DEEPCELL_ACCESS_TOKEN", raising=False)
    assert not environment.ensure_required_env("cellsam", {}, lambda p: None)
    assert not environment.ensure_required_env("cellsam", {}, None)
    assert "DEEPCELL_ACCESS_TOKEN" not in environment.os.environ
    assert environment.missing_required_env(
        "cellsam", {"model_path": "C:/model.pt"}
    ) == []
    assert environment.ensure_required_env("thresholding", {}, None)


def _payload_json(body):
    """The JSON part of an attachment encoded by `payload.encode`."""
    size = int.from_bytes(body[:8], 'big')
    return json.loads(body[8:8 + size].decode('utf-8'))


def test_init_payload_roundtrip():
    import pandas as pd
    from cellacdc.service import payload

    segm = np.arange(12, dtype=np.uint32).reshape(3, 4)
    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4], 'id': [1, 0]})
    kwargs = {'segm_data': segm, 'input_points_df': df, 'gpu': False}

    body = payload.encode(kwargs)
    assert isinstance(body, bytes)
    # The array is in the `.npz` part, not in the JSON
    assert not isinstance(_payload_json(body)['segm_data'], list)
    decoded = payload.decode(body)
    np.testing.assert_array_equal(decoded['segm_data'], segm)
    assert decoded['segm_data'].dtype == segm.dtype
    pd.testing.assert_frame_equal(decoded['input_points_df'], df)
    assert decoded['gpu'] is False

    # Without arrays or files: only the JSON, no `.npz` part
    body = payload.encode({'gpu': False})
    assert len(body) == 8 + len(json.dumps({'gpu': False}).encode('utf-8'))
    assert payload.decode(body) == {'gpu': False}


def test_payload_uploads_files_on_disk(tmp_path):
    """Every argument that is the path (str or path-like) of an existing file
    is uploaded: the worker receives a `ModelFile`, never the client's path.
    Other strings (folders, missing files, '' and 'None') are sent as they
    are.
    """
    import pandas as pd
    from cellacdc.service import payload

    path = tmp_path / 'FUCCI (1).txt'
    path.write_bytes(b'\x00custom model\xff')
    points = tmp_path / 'points.csv'
    points.write_bytes(b'x,y\n1,2\n')
    folder = tmp_path / 'weights folder'
    folder.mkdir()
    segm = np.arange(6, dtype=np.uint16).reshape(2, 3)
    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4]})
    kwargs = {
        'model_path': str(path), 'points_path': points, 'label': str(path),
        'segm_data': segm, 'input_points_df': df,
        'folder': str(folder), 'missing': str(tmp_path / 'missing.pt'),
        'denoise_model_path': 'None', 'empty_path': '', 'unset_path': None,
        'model_type': 'cyto3', 'diameter': 30.5,
    }
    for key in ('model_path', 'points_path'):
        assert payload.is_file_path(kwargs[key])
    for key in ('folder', 'missing', 'denoise_model_path', 'empty_path',
                'unset_path', 'model_type', 'diameter'):
        assert not payload.is_file_path(kwargs[key])

    body = payload.encode(kwargs)
    meta = _payload_json(body)
    assert meta['model_path']['name'] == path.name
    assert meta['points_path']['name'] == points.name
    # The client's paths of uploaded files are not sent
    for client_path in (path, points):
        for text in (str(client_path), json.dumps(str(client_path))[1:-1]):
            assert text.encode('utf-8') not in body

    decoded = payload.decode(body)
    assert set(decoded) == set(kwargs)
    model_file = payload.ModelFile(path.name, path.read_bytes())
    assert decoded['model_path'] == model_file
    assert decoded['label'] == model_file
    assert decoded['points_path'] == payload.ModelFile(
        points.name, points.read_bytes()
    )
    np.testing.assert_array_equal(decoded['segm_data'], segm)
    assert decoded['segm_data'].dtype == np.uint16
    pd.testing.assert_frame_equal(decoded['input_points_df'], df)
    for key in ('folder', 'missing', 'denoise_model_path', 'empty_path',
                'model_type', 'diameter'):
        assert decoded[key] == kwargs[key]
        assert type(decoded[key]) is type(kwargs[key])
    assert decoded['unset_path'] is None
    # The client's arguments are not modified
    assert kwargs['model_path'] == str(path)
    assert kwargs['points_path'] == points


@pytest.mark.xfail(
    strict=True, raises=TypeError,
    reason=(
        'BUG (minor, also before protocol 2): payload.encode sends path-like '
        'values that are not existing files unchanged to json.dumps, which '
        'raises TypeError instead of sending them as strings like str paths'
    ),
)
@pytest.mark.parametrize('name', ['folder', 'missing.pt'])
def test_payload_sends_other_path_like_values_as_strings(tmp_path, name):
    from cellacdc.service import payload

    (tmp_path / 'folder').mkdir()
    value = tmp_path / name
    decoded = payload.decode(payload.encode({'path': value}))
    assert decoded == {'path': str(value)}


def test_payload_kwarg_names_never_collide(tmp_path):
    """The `.npz` keys are generated, never kwarg names: kwargs named like
    internal keys do not overwrite each other."""
    from cellacdc.service import payload

    first = tmp_path / 'first.bin'
    first.write_bytes(b'first')
    second = tmp_path / 'second.bin'
    second.write_bytes(b'second')
    kwargs = {
        'array_0': np.array([1, 2], dtype=np.int16),
        'file_0': str(second),
        'array_1': np.array([3.5]),
        'file_1': first,
        'file_2': np.array([7], dtype=np.uint8),
        'array_2': str(first),
        '__acdc_json__': 'json-like name',
        '__acdc_array__': 'array_0',
        '__acdc_file__': 'file_0',
        '__acdc_dataframe__': 1,
        'name': 'first.bin',
    }
    decoded = payload.decode(payload.encode(kwargs))

    assert set(decoded) == set(kwargs)
    np.testing.assert_array_equal(decoded['array_0'], [1, 2])
    assert decoded['array_0'].dtype == np.int16
    np.testing.assert_array_equal(decoded['array_1'], [3.5])
    np.testing.assert_array_equal(decoded['file_2'], [7])
    assert decoded['file_0'] == payload.ModelFile('second.bin', b'second')
    assert decoded['file_1'] == payload.ModelFile('first.bin', b'first')
    assert decoded['array_2'] == payload.ModelFile('first.bin', b'first')
    for key in ('__acdc_json__', '__acdc_array__', '__acdc_file__',
                '__acdc_dataframe__', 'name'):
        assert decoded[key] == kwargs[key]


@pytest.mark.parametrize('text', ['[1, 2]', '"text"', 'null', '3'])
def test_payload_decode_rejects_non_object_json(text):
    from cellacdc.service import payload

    data = text.encode('utf-8')
    with pytest.raises(ValueError, match='JSON object'):
        payload.decode(len(data).to_bytes(8, 'big') + data)


@pytest.mark.parametrize('body', [
    b'', b'\x00' * 7, (100).to_bytes(8, 'big') + b'{}',
    (3).to_bytes(8, 'big') + b'{"a',
])
def test_payload_decode_rejects_truncated_body(body):
    from cellacdc.service import payload

    with pytest.raises(ValueError):
        payload.decode(body)
    if len(body) < 8 or int.from_bytes(body[:8], 'big') > len(body) - 8:
        with pytest.raises(ValueError, match='Truncated payload'):
            payload.decode(body)


@pytest.mark.xfail(
    strict=True, raises=(zipfile.BadZipFile, KeyError),
    reason=(
        'BUG (minor): payload.decode raises zipfile.BadZipFile (truncated '
        '.npz part) or KeyError (.npz part missing) instead of ValueError'
    ),
)
@pytest.mark.parametrize('cut', ['npz_end', 'whole_npz'])
def test_payload_decode_rejects_truncated_npz(cut):
    from cellacdc.service import payload

    body = payload.encode({'segm_data': np.arange(100)})
    json_end = 8 + int.from_bytes(body[:8], 'big')
    assert len(body) > json_end + 10
    truncated = body[:-10] if cut == 'npz_end' else body[:json_end]
    with pytest.raises(ValueError):
        payload.decode(truncated)


def test_nested_array_payload_roundtrip():
    from cellacdc.service import payload

    value = {
        None: {
            2: {
                'original_size': (10, 12),
                'features': np.arange(6, dtype=np.float32).reshape(2, 3),
            }
        }
    }
    value_json, data = payload.encode_value(value)
    decoded = payload.decode_value(value_json, data)

    assert decoded[None][2]['original_size'] == (10, 12)
    np.testing.assert_array_equal(
        decoded[None][2]['features'], value[None][2]['features']
    )


@pytest.mark.parametrize('value', [
    {}, None, {'text': '\u00e9', 'sizes': (2, 3), None: [1, True]},
    {'labels': np.ones((2, 2), dtype=np.uint32), 'embeddings': {}},
])
def test_value_body_roundtrip(value):
    from cellacdc.service import payload

    body = payload.encode_value_body(value)
    metadata, _ = payload.encode_value(value)
    assert int.from_bytes(body[:8], 'big') == len(metadata.encode('utf-8'))
    decoded = payload.decode_value_body(body)
    if isinstance(value, dict) and 'labels' in value:
        np.testing.assert_array_equal(decoded['labels'], value['labels'])
        assert decoded['labels'].dtype == value['labels'].dtype
        assert decoded['embeddings'] == {}
    else:
        assert decoded == value


@pytest.mark.parametrize('body', [
    b'', b'\x00' * 7, (100).to_bytes(8, 'big') + b'{}',
])
def test_value_body_rejects_truncated_metadata(body):
    from cellacdc.service import payload

    with pytest.raises(ValueError, match='Truncated payload metadata'):
        payload.decode_value_body(body)


def _points_df(num_rows):
    import pandas as pd

    # int64, like the columns decoded by `pd.read_json` (on Windows with
    # numpy < 2 the default integer type is int32)
    index = np.arange(num_rows, dtype=np.int64)
    return pd.DataFrame({
        'x': index % 1800, 'y': index % 350, 'id': index + 1,
        'frame_i': index // 25,
    })


_FORM_FIELD_LIMIT = 1024 * 1024  # Starlette's limit for one form field


@pytest.mark.parametrize('with_array', [False, True])
def test_payload_keeps_large_json_in_one_attachment(with_array):
    """The worker (Starlette) limits each form field to 1 MiB, so all
    arguments are sent in one attachment, which has no size limit: JSON far
    larger than a form field (e.g. a large table of point prompts) is sent
    as it is, next to the `.npz` part with the arrays.
    """
    import pandas as pd
    from cellacdc.service import payload

    kwargs = {'input_points_df': _points_df(200_000), 'gpu': False}
    if with_array:
        kwargs['segm_data'] = np.arange(6, dtype=np.uint32).reshape(2, 3)
    body = payload.encode(kwargs)
    json_size = int.from_bytes(body[:8], 'big')
    assert json_size > 4 * _FORM_FIELD_LIMIT
    assert isinstance(_payload_json(body), dict)
    # The `.npz` part only holds arrays (and files)
    assert (len(body) > 8 + json_size) == with_array

    decoded = payload.decode(body)
    pd.testing.assert_frame_equal(
        decoded['input_points_df'], kwargs['input_points_df']
    )
    assert decoded['gpu'] is False
    if with_array:
        np.testing.assert_array_equal(decoded['segm_data'], kwargs['segm_data'])
        assert decoded['segm_data'].dtype == np.uint32


def test_value_payload_keeps_large_json_inline(tmp_path):
    from cellacdc import sam_embeddings
    from cellacdc.service import payload

    # The text is never moved, also when it is larger than a form field: SAM
    # embedding files and worker responses keep their format
    big = {'text': 'x' * (2 * _FORM_FIELD_LIMIT)}
    big_text, big_data = payload.encode_value(big)
    assert big_data is None
    assert payload.decode_value(big_text) == big
    body = payload.encode_value_body(big)
    assert int.from_bytes(body[:8], 'big') == len(big_text.encode('utf-8'))
    assert body[8:] == big_text.encode('utf-8')
    assert payload.decode_value_body(body) == big
    sam_embeddings.save(big, tmp_path / 'embeddings.zip')
    with zipfile.ZipFile(tmp_path / 'embeddings.zip') as archive:
        assert archive.namelist() == [sam_embeddings._METADATA]
        assert archive.read(sam_embeddings._METADATA).decode() == big_text

    value = {
        z: {'original_size': (2, 2), 'features': np.full((1, 2), z)}
        for z in range(50)
    }
    text, data = payload.encode_value(value)
    assert data is not None
    body = payload.encode_value_body(value)
    assert body[8:8 + len(text.encode('utf-8'))] == text.encode('utf-8')
    for decoded in (
            payload.decode_value(text, data), payload.decode_value_body(body),
        ):
        assert set(decoded) == set(value)
        for z, embedding in decoded.items():
            assert embedding['original_size'] == (2, 2)
            np.testing.assert_array_equal(
                embedding['features'], value[z]['features']
            )

    # Values that look like the internal markers are decoded as themselves
    for lookalike in (
            ['__acdc_array__', '__acdc_dict__', '__acdc_tuple__'],
            {'__acdc_array__': 'array_0'}, {'__acdc_dict__': []},
            {'__acdc_tuple__': [1, 2]}, {'__acdc_file__': 'file_0'},
        ):
        assert payload.decode_value(*payload.encode_value(lookalike)) == lookalike
        assert payload.decode_value_body(
            payload.encode_value_body(lookalike)
        ) == lookalike


def _serve_app(app):
    """Serve `app` with uvicorn in a thread; return (url, stop function)."""
    import threading
    import time
    import uvicorn
    from cellacdc.service.client import _find_free_port

    port = _find_free_port()
    # `log_config=None`: do not reconfigure the logging of the test process
    server = uvicorn.Server(uvicorn.Config(
        app, host='127.0.0.1', port=port, lifespan='off', log_config=None,
    ))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    def stop():
        server.should_exit = True
        thread.join(10)

    for _ in range(100):
        if server.started:
            return f'http://127.0.0.1:{port}', stop
        time.sleep(0.1)
    stop()
    raise RuntimeError('The test server did not start')


def test_worker_accepts_arguments_larger_than_form_fields(monkeypatch):
    """Over real HTTP (uvicorn and the real client): the worker (Starlette)
    rejects form fields larger than 1 MiB. A table of point prompts for SAM
    (`input_points_df`) with ~28k rows failed with "Field exceeded maximum
    size of 1024KB." at model init. All arguments are attachments now, so
    there is no size limit for /init (with and without arrays), /segment,
    /add_prompt and SAM embeddings.
    """
    import types
    from cellacdc import myutils_models
    from cellacdc.service import payload, worker
    from cellacdc.service.client import post_init

    received = {}

    class Model:
        def __init__(self, input_points_df=None, text='', mask=None):
            received['init'] = (input_points_df, text, mask)

        def segment(self, image, points_df=None, posData=None):
            received['segment'] = points_df
            if posData is not None:
                received['embeddings'] = posData.frame_embeddings
            return np.ones(image.shape, np.uint32)

        def add_prompt(self, prompt, prompt_id, image, points_df=None):
            received['add_prompt'] = (prompt, prompt_id, image, points_df)

    monkeypatch.setattr(
        myutils_models, 'import_segment_module',
        lambda name: types.SimpleNamespace(Model=Model),
    )
    df = _points_df(200_000)
    assert len(df.to_json(orient='split')) > 4 * _FORM_FIELD_LIMIT
    mask = np.arange(12, dtype=np.uint16).reshape(3, 4)
    url, stop = _serve_app(worker.create_app('fake', defer_init=True))
    try:
        # /init without arrays: only the JSON in the attachment
        post_init(url, {'input_points_df': df}, timeout=60)
        assert received['init'][0].equals(df)
        assert received['init'][2] is None

        # /init with arrays: JSON and `.npz` in the attachment
        post_init(url, {'input_points_df': df, 'mask': mask}, timeout=60)
        assert received['init'][0].equals(df)
        np.testing.assert_array_equal(received['init'][2], mask)
        assert received['init'][2].dtype == np.uint16

        # A plain string larger than one form field
        text = 'x' * (2 * _FORM_FIELD_LIMIT)
        post_init(url, {'text': text}, timeout=60)
        assert received['init'][1] == text

        client = ModelClient(url, timeout=60)
        image = np.zeros((4, 4), np.uint8)
        labels = client.segment(image, points_df=df)
        np.testing.assert_array_equal(labels, np.ones((4, 4)))
        assert received['segment'].equals(df)

        client.add_prompt(prompt=(0, 1, 1), prompt_id=1, image=image, points_df=df)
        prompt, prompt_id, prompt_image, prompt_df = received['add_prompt']
        assert prompt == [0, 1, 1] and prompt_id == 1
        np.testing.assert_array_equal(prompt_image, image)
        assert prompt_df.equals(df)

        embeddings = {
            z: {'original_size': (2, 2), 'features': np.full((1, 2), z)}
            for z in range(20_000)
        }
        embeddings_json, _ = payload.encode_value(embeddings)
        assert len(embeddings_json) > _FORM_FIELD_LIMIT
        pos_data = types.SimpleNamespace(sam_embeddings={0: dict(embeddings)})
        labels = client.segment(
            image, frame_i=0, posData=pos_data, use_loaded_embeddings=True
        )
        np.testing.assert_array_equal(labels, np.ones((4, 4)))
        assert len(received['embeddings']) == 20_000
        assert received['embeddings'][19_999]['original_size'] == (2, 2)
        np.testing.assert_array_equal(
            received['embeddings'][19_999]['features'], [[19_999, 19_999]]
        )
        # The (large) response with the embeddings reaches the client too
        assert set(pos_data.sam_embeddings[0]) == set(embeddings)
        client.close()
    finally:
        stop()


def test_worker_rejects_clients_of_other_protocol_versions(monkeypatch):
    """Over real HTTP: requests without the client's protocol version (e.g.
    from an older Cell-ACDC) or with another version are answered with
    HTTP 426, except the endpoints for checking the version, liveness,
    shutdown and FastAPI's documentation."""
    import threading
    import cellacdc
    from cellacdc import myutils_models
    from cellacdc.service import protocol, worker

    version = protocol.PROTOCOL_VERSION
    killed = threading.Event()
    # /shutdown stops the worker process after answering: not this one
    monkeypatch.setattr(worker.os, 'kill', lambda pid, sig: killed.set())
    monkeypatch.setattr(
        myutils_models, 'import_segment_module',
        lambda name: _fake_segment_module(requires_segm=False),
    )
    url, stop = _serve_app(worker.create_app('fake', defer_init=True))
    try:
        versioned = [
            ('GET', '/info'), ('GET', '/spec'), ('POST', '/init'),
            ('POST', '/segment'), ('POST', '/segment3DT'),
            ('POST', '/add_prompt'), ('POST', '/second_ch_img_to_stack'),
            ('POST', '/predictCcaState'), ('POST', '/jobs/segment'),
            ('GET', '/jobs/abc'), ('GET', '/jobs/abc/result'),
            ('DELETE', '/jobs/abc'),
        ]
        for sent in (None, '1', str(version + 1), 'abc'):
            headers = {} if sent is None else {protocol.PROTOCOL_HEADER: sent}
            for method, path in versioned:
                res = requests.request(
                    method, f'{url}{path}', headers=headers, timeout=10
                )
                assert res.status_code == 426, (sent, path, res.text)
                detail = res.json()['detail']
                assert f'client uses model worker protocol {sent or 1}' in detail
                assert f'uses protocol {version}' in detail
                assert str(cellacdc.__version__) in detail

        # With the client's version, requests reach the endpoints
        headers = protocol.headers()
        assert headers == {protocol.PROTOCOL_HEADER: str(version)}
        res = requests.get(f'{url}/spec', headers=headers, timeout=10)
        assert res.status_code == 200 and 'init_params' in res.json()
        res = requests.get(f'{url}/info', headers=headers, timeout=10)
        assert res.status_code == 503  # not initialized, but not rejected

        # Unversioned endpoints answer clients of any version
        for headers in ({}, {protocol.PROTOCOL_HEADER: '1'}, protocol.headers()):
            res = requests.get(f'{url}/version', headers=headers, timeout=10)
            assert res.status_code == 200
            assert res.json() == {
                'protocol_version': version,
                'cellacdc_version': str(cellacdc.__version__),
            }
            res = requests.get(f'{url}/health', headers=headers, timeout=10)
            assert res.status_code == 200
            assert res.json() == {'status': 'ok', 'model': 'fake'}
            for path in ('/docs', '/openapi.json'):
                res = requests.get(f'{url}{path}', headers=headers, timeout=10)
                assert res.status_code == 200, path
        res = requests.post(f'{url}/shutdown', timeout=10)
        assert res.status_code == 200
        assert res.json() == {'status': 'shutting down'}
        assert killed.wait(5)
    finally:
        stop()


def _version_session(status, data=None):
    """A session whose `get` answers `GET /version` with `status`/`data`."""
    from types import SimpleNamespace

    requested = []

    def get(url, timeout):
        requested.append((url, timeout))
        res = requests.Response()
        res.url, res.status_code = url, status
        res._content = json.dumps(data).encode()
        return res

    return SimpleNamespace(get=get, requested=requested)


def test_check_worker_version():
    from cellacdc.service import client as client_module, protocol
    from cellacdc.service.schema import VersionResponse

    version = protocol.PROTOCOL_VERSION
    WorkerVersionError = client_module.WorkerVersionError

    # Same protocol
    session = _version_session(200, {
        'protocol_version': version, 'cellacdc_version': '1.2.3',
    })
    result = client_module.check_worker_version(
        'http://worker/', timeout=4, session=session
    )
    assert isinstance(result, VersionResponse)
    assert result.protocol_version == version
    assert result.cellacdc_version == '1.2.3'
    assert session.requested == [('http://worker/version', 4)]

    # Protocol 1 workers have no /version endpoint
    session = _version_session(404, {'detail': 'Not Found'})
    with pytest.raises(WorkerVersionError) as raised:
        client_module.check_worker_version('http://worker', session=session)
    assert isinstance(raised.value, RuntimeError)
    assert raised.value.worker_protocol == 1
    assert raised.value.worker_version is None
    assert raised.value.base_url == 'http://worker'
    assert 'an older Cell-ACDC (protocol 1)' in str(raised.value)
    assert f'uses protocol {version}' in str(raised.value)

    # Another protocol
    session = _version_session(200, {
        'protocol_version': version + 1, 'cellacdc_version': '9.0',
    })
    with pytest.raises(WorkerVersionError) as raised:
        client_module.check_worker_version('http://worker', session=session)
    assert raised.value.worker_protocol == version + 1
    assert raised.value.worker_version == '9.0'
    assert f'Cell-ACDC 9.0 (protocol {version + 1})' in str(raised.value)

    # Other errors are not version mismatches
    session = _version_session(500, {'detail': 'worker crashed'})
    with pytest.raises(RuntimeError, match='worker crashed') as raised:
        client_module.check_worker_version('http://worker', session=session)
    assert not isinstance(raised.value, WorkerVersionError)


def _recording_worker_app(requested, worker_protocol=None):
    """A minimal worker app with the protocol `worker_protocol` (None: the
    client's; 1: no `/version` endpoint). Records each request as
    `(method, path, protocol header)` in `requested`, and the parts sent to
    `/init` in `app.state.init_parts` (`(kind, decoded value)`)."""
    from fastapi import FastAPI, Request
    from cellacdc.service import payload, protocol

    app = FastAPI()
    app.state.init_parts = None

    @app.middleware('http')
    async def record(request, call_next):
        requested.append((
            request.method, request.url.path,
            request.headers.get(protocol.PROTOCOL_HEADER),
        ))
        return await call_next(request)

    if worker_protocol != 1:
        @app.get('/version')
        def version():
            return {
                'protocol_version': worker_protocol or protocol.PROTOCOL_VERSION,
                'cellacdc_version': 'test-version',
            }

    @app.post('/init')
    async def init(request: Request):
        parts = {}
        for key, value in (await request.form()).multi_items():
            if isinstance(value, str):
                parts[key] = ('field', value)
            else:
                parts[key] = ('file', payload.decode(await value.read()))
        app.state.init_parts = parts
        return {'status': 'initialized'}

    @app.get('/info')
    def info():
        return {'name': 'fake'}

    return app


@pytest.mark.parametrize('worker_protocol', [None, 1, 3])
def test_model_client_checks_worker_version_before_use(worker_protocol):
    from cellacdc.service import protocol
    from cellacdc.service.client import WorkerVersionError

    assert worker_protocol != protocol.PROTOCOL_VERSION
    version = str(protocol.PROTOCOL_VERSION)
    requested = []
    url, stop = _serve_app(_recording_worker_app(requested, worker_protocol))
    try:
        if worker_protocol is None:
            client = ModelClient(url)
            assert client.worker_version.protocol_version == int(version)
            assert client.worker_version.cellacdc_version == 'test-version'
            assert client.model_name == 'fake'
            # Version first; every request with the client's version
            assert requested == [
                ('GET', '/version', version), ('GET', '/info', version),
            ]
            client.close()
        else:
            with pytest.raises(WorkerVersionError) as raised:
                ModelClient(url)
            assert raised.value.worker_protocol == worker_protocol
            # The worker is not used
            assert requested == [('GET', '/version', version)]
    finally:
        stop()


def test_post_init_sends_arguments_as_one_attachment(tmp_path):
    """`post_init` sends one attachment with the protocol version, and does
    not read the model's argument specs (`/spec`) first."""
    from cellacdc.service import payload, protocol
    from cellacdc.service.client import post_init

    weights = tmp_path / 'weights.pt'
    weights.write_bytes(b'weights')
    requested = []
    app = _recording_worker_app(requested)
    url, stop = _serve_app(app)
    try:
        post_init(url, {
            'diameter': 30, 'model_path': str(weights),
            'segm_data': np.ones((2, 2), dtype=np.uint8),
        }, timeout=30)
    finally:
        stop()
    assert requested == [('POST', '/init', str(protocol.PROTOCOL_VERSION))]
    assert set(app.state.init_parts) == {'kwargs'}
    kind, kwargs = app.state.init_parts['kwargs']
    assert kind == 'file'
    assert set(kwargs) == {'diameter', 'model_path', 'segm_data'}
    assert kwargs['diameter'] == 30
    assert kwargs['model_path'] == payload.ModelFile('weights.pt', b'weights')
    np.testing.assert_array_equal(kwargs['segm_data'], np.ones((2, 2)))


@pytest.mark.parametrize('worker_protocol', [None, 1, 3])
def test_remote_backend_checks_worker_version_before_init(worker_protocol):
    from cellacdc.service import protocol
    from cellacdc.service.client import WorkerVersionError

    version = str(protocol.PROTOCOL_VERSION)
    requested = []
    app = _recording_worker_app(requested, worker_protocol)
    url, stop = _serve_app(app)
    try:
        backend = RemoteBackend(url)
        if worker_protocol is None:
            client = backend.get_client(init_model_kwargs={'diameter': 30})
            assert backend.client is client
            assert app.state.init_parts['kwargs'] == ('file', {'diameter': 30})
            assert requested == [
                ('GET', '/version', None), ('POST', '/init', version),
                ('GET', '/version', version), ('GET', '/info', version),
            ]
            backend.stop()
        else:
            with pytest.raises(
                    RuntimeError,
                    match='Install the same Cell-ACDC version on the server',
                ) as raised:
                backend.get_client(init_model_kwargs={'diameter': 30})
            assert isinstance(raised.value.__cause__, WorkerVersionError)
            assert raised.value.__cause__.worker_protocol == worker_protocol
            # The model is not initialized on the incompatible worker
            assert app.state.init_parts is None
            assert requested == [('GET', '/version', None)]
            assert backend.client is None
    finally:
        stop()


# def test_sam_embeddings_portable_and_legacy_roundtrip(tmp_path, monkeypatch):
#     import builtins
#     import torch
#     from cellacdc import sam_embeddings

#     embeddings = {
#         1: {
#             2: {
#                 'original_size': (10, 12),
#                 'features': torch.arange(6, dtype=torch.float32).reshape(2, 3),
#                 'is_image_set': True,
#             }
#         }
#     }
#     portable_path = tmp_path / 'portable.pt'
#     sam_embeddings.save(embeddings, portable_path)

#     original_import = builtins.__import__

#     def without_torch(name, *args, **kwargs):
#         if name == 'torch' or name.startswith('torch.'):
#             raise ImportError('torch intentionally unavailable')
#         return original_import(name, *args, **kwargs)

#     monkeypatch.setattr(builtins, '__import__', without_torch)
#     portable = sam_embeddings.load(portable_path)
#     np.testing.assert_array_equal(
#         portable[1][2]['features'], np.arange(6).reshape(2, 3)
#     )
#     assert portable[1][2]['original_size'] == (10, 12)

#     monkeypatch.setattr(builtins, '__import__', original_import)
#     legacy_path = tmp_path / 'legacy.pt'
#     torch.save(embeddings, legacy_path)
#     monkeypatch.setattr(builtins, '__import__', without_torch)
#     legacy = sam_embeddings.load(legacy_path)
#     np.testing.assert_array_equal(
#         legacy[1][2]['features'], np.arange(6).reshape(2, 3)
#     )
#     assert legacy[1][2]['original_size'] == (10, 12)


@pytest.mark.parametrize(
    'direct_url, version, expected',
    [
        (
            {
                'url': 'https://github.com/SchmollerLab/Cell_ACDC.git',
                'vcs_info': {'vcs': 'git', 'commit_id': 'abc123'},
            },
            '1.7.0.dev1',
            'cellacdc @ git+https://github.com/SchmollerLab/'
            'Cell_ACDC.git@abc123',
        ),
        (None, '1.6.5', 'cellacdc==1.6.5'),
    ],
)
def test_installed_cellacdc_spec_preserves_provenance(
        monkeypatch, direct_url, version, expected
    ):
    import cellacdc
    from cellacdc.service import environment

    class Distribution:
        def read_text(self, name):
            assert name == 'direct_url.json'
            return json.dumps(direct_url) if direct_url is not None else None

    monkeypatch.setattr(cellacdc, '__version__', version)
    monkeypatch.setattr(environment, 'distribution', lambda name: Distribution())
    assert environment._installed_cellacdc_spec() == expected


def test_unreproducible_development_build_fails(monkeypatch):
    import cellacdc
    from cellacdc.service import environment

    class Distribution:
        def read_text(self, name):
            return None

    monkeypatch.setattr(cellacdc, '__version__', '1.7.0.dev1+local')
    monkeypatch.setattr(environment, 'distribution', lambda name: Distribution())
    with pytest.raises(RuntimeError, match='Cannot reproduce'):
        environment._installed_cellacdc_spec()


def test_runtime_mismatch_detects_other_cellacdc(monkeypatch, tmp_path):
    from unittest.mock import Mock
    from cellacdc.service import environment, protocol

    version = protocol.PROTOCOL_VERSION
    checkout = tmp_path / 'Cell_ACDC'
    other = str(tmp_path / 'env' / 'Lib' / 'site-packages' / 'cellacdc' / '__init__.py')

    def identity(**changes):
        return dict({
            'protocol': version, 'version': '1.7.0',
            'file': str(checkout / 'cellacdc' / '__init__.py'),
            'prefix': str(tmp_path / 'env'),
        }, **changes)

    assert 'cannot be imported' in environment._runtime_mismatch(None)

    # Source checkout: the environment must import this checkout
    monkeypatch.setattr(
        environment, '_get_cellacdc_source_root', lambda: str(checkout)
    )
    monkeypatch.setattr(
        environment, '_installed_cellacdc_version',
        Mock(side_effect=AssertionError('No version check for checkouts')),
    )
    assert environment._runtime_mismatch(identity()) is None
    assert environment._runtime_mismatch(identity(version='0.1')) is None
    reason = environment._runtime_mismatch(identity(file=other))
    assert 'not from this source checkout' in reason
    assert other in reason and str(checkout) in reason
    # Cell-ACDC versions without `service.protocol` use protocol 1
    reason = environment._runtime_mismatch(identity(protocol=1))
    assert f'protocol 1, but this Cell-ACDC uses protocol {version}' in reason

    # Installed package: the environment must have the same version
    monkeypatch.setattr(environment, '_get_cellacdc_source_root', lambda: None)
    monkeypatch.setattr(environment, '_installed_cellacdc_version', lambda: '1.7.0')
    assert environment._runtime_mismatch(identity(file=other)) is None
    reason = environment._runtime_mismatch(identity(file=other, version='1.6.5'))
    assert 'Cell-ACDC 1.6.5' in reason and 'Cell-ACDC 1.7.0' in reason
    reason = environment._runtime_mismatch(identity(file=other, version=None))
    assert 'Cell-ACDC None' in reason
    reason = environment._runtime_mismatch(identity(file=other, protocol=1))
    assert 'protocol 1' in reason


def test_parse_runtime_identity():
    from cellacdc.service import environment

    marker = environment._RUNTIME_MARKER
    identity = {'protocol': 2, 'version': '1.7.0', 'file': 'f', 'prefix': 'p'}
    output = f'[model startup] x\n{marker}{json.dumps(identity)}\n__ACDC_ENV_OK__\n'
    assert environment._parse_runtime_identity(output) == identity
    assert environment._parse_runtime_identity('__ACDC_ENV_OK__\n') is None
    assert environment._parse_runtime_identity(f'{marker}{{broken\n') is None


def test_runtime_identity_of_old_cellacdc_is_protocol_1(tmp_path):
    """A Cell-ACDC without `cellacdc.service.protocol` (protocol 1) is
    identified as such by the code run in model environments."""
    import subprocess
    from cellacdc.service import environment

    package = tmp_path / 'cellacdc'
    (package / 'service').mkdir(parents=True)
    (package / '__init__.py').write_text('')
    (package / 'service' / '__init__.py').write_text('')
    result = subprocess.run(
        [sys.executable, '-c', environment._RUNTIME_IDENTITY_CODE],
        cwd=str(tmp_path), capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    identity = environment._parse_runtime_identity(result.stdout)
    assert identity['protocol'] == 1
    assert os.path.normcase(identity['file']) == os.path.normcase(
        str(package / '__init__.py')
    )
    assert 'protocol 1' in environment._runtime_mismatch(identity)


def test_runtime_of_this_source_checkout_matches():
    """Real subprocess: this interpreter imports this Cell-ACDC."""
    import cellacdc
    from cellacdc.service import environment, protocol

    identity, imports_ok, output = environment._read_cellacdc_runtime(
        sys.executable
    )
    assert imports_ok, output
    assert identity['protocol'] == protocol.PROTOCOL_VERSION
    assert os.path.normcase(identity['file']) == os.path.normcase(
        os.path.abspath(cellacdc.__file__)
    )
    assert os.path.normcase(identity['prefix']) == os.path.normcase(sys.prefix)
    assert environment.cellacdc_runtime_mismatch(sys.executable) is None


@pytest.mark.parametrize('layout', [
    ('venvs', 'Scripts', 'python.exe'),  # venv, Windows
    ('venvs', 'bin', 'python'),  # venv, Linux/macOS
    ('envs', 'python.exe'),  # conda, Windows
    ('envs', 'bin', 'python'),  # conda, Linux/macOS
])
def test_is_managed_model_env(tmp_path, layout):
    from cellacdc.service import environment

    root, *relative = layout
    managed = tmp_path / root / '_acdc_cellpose_v3'
    assert environment.is_managed_model_env(str(managed.joinpath(*relative)))
    user_env = tmp_path / root / 'my_env'
    assert not environment.is_managed_model_env(str(user_env.joinpath(*relative)))
    assert not environment.is_managed_model_env(
        str((tmp_path / root / 'acdc_env').joinpath(*relative))
    )
    if len(relative) > 1:
        # Only the environment's folder counts, not the folders above it
        nested = tmp_path / '_acdc_envs' / 'my_env'
        assert not environment.is_managed_model_env(str(nested.joinpath(*relative)))


@pytest.mark.parametrize('mismatch, repaired', [
    (None, None),
    ('it has Cell-ACDC 1.6.5, but this is Cell-ACDC 1.7.0', True),
    ('Cell-ACDC cannot be imported', False),
])
def test_ensure_cellacdc_cli_repairs_only_other_runtime(
        monkeypatch, mismatch, repaired,
    ):
    from unittest.mock import Mock
    from cellacdc.service import environment

    check = Mock(return_value=mismatch)
    repair = Mock(return_value=repaired)
    monkeypatch.setattr(environment, 'cellacdc_runtime_mismatch', check)
    monkeypatch.setattr(environment, 'repair_cellacdc_runtime', repair)
    monkeypatch.setattr(
        environment, 'install_cellacdc_cli',
        Mock(side_effect=AssertionError('Only via repair_cellacdc_runtime')),
    )
    logger = Mock()

    result = environment.ensure_cellacdc_cli('model-python', logger_func=logger)

    check.assert_called_once_with('model-python')
    if mismatch is None:
        assert result is True
        repair.assert_not_called()
    else:
        assert result is repaired
        repair.assert_called_once_with('model-python', logger_func=logger)
        output = '\n'.join(str(call.args[0]) for call in logger.call_args_list)
        assert mismatch in output


@pytest.mark.parametrize('installed, after, expected', [
    (False, None, False), (True, None, True), (True, 'still different', False),
])
def test_repair_cellacdc_runtime_reinstalls_and_verifies(
        monkeypatch, installed, after, expected,
    ):
    from unittest.mock import Mock
    from cellacdc.service import environment

    install = Mock(return_value=installed)
    check = Mock(return_value=after)
    monkeypatch.setattr(environment, 'install_cellacdc_cli', install)
    monkeypatch.setattr(environment, 'cellacdc_runtime_mismatch', check)
    logger = Mock()

    assert environment.repair_cellacdc_runtime(
        'model-python', logger_func=logger
    ) is expected
    install.assert_called_once_with(
        'model-python', logger_func=logger, reinstall=True
    )
    assert check.call_count == int(installed)
    if after is not None:
        output = '\n'.join(str(call.args[0]) for call in logger.call_args_list)
        assert 'still differs' in output and after in output


@pytest.mark.parametrize('reinstall', [False, True])
def test_install_cellacdc_cli_reinstalls_released_package(monkeypatch, reinstall):
    from unittest.mock import Mock
    from cellacdc.service import environment

    run = Mock(return_value=0)
    monkeypatch.setattr(environment, 'run_streaming', run)
    monkeypatch.setattr(environment, '_get_cellacdc_source_root', lambda: None)
    monkeypatch.setattr(
        environment, '_installed_cellacdc_spec', lambda: 'cellacdc==1.7.0'
    )
    for name in ('link_cellacdc_to_env', '_uninstall_shadowing_cellacdc'):
        monkeypatch.setattr(
            environment, name, Mock(side_effect=AssertionError(name))
        )

    assert environment.install_cellacdc_cli(
        'model-python', logger_func=Mock(), reinstall=reinstall
    )

    pip = ['model-python', '-m', 'pip', 'install', '--disable-pip-version-check']
    assert run.call_count == 2
    install = run.call_args_list[0].args[0]
    assert install[:len(pip)] == pip and install[-1] == 'cellacdc==1.7.0'
    expected = {'--no-deps', '--force-reinstall'} if reinstall else {'--no-deps'}
    assert set(install[len(pip):-1]) == expected
    # Only Cell-ACDC is reinstalled, not its runtime dependencies
    assert run.call_args_list[1].args[0] == (
        pip + environment.WORKER_RUNTIME_DEPENDENCIES
    )


def test_install_cellacdc_cli_reports_failed_package_install(monkeypatch):
    from unittest.mock import Mock
    from cellacdc.service import environment

    run = Mock(return_value=1)
    monkeypatch.setattr(environment, 'run_streaming', run)
    monkeypatch.setattr(environment, '_get_cellacdc_source_root', lambda: None)
    monkeypatch.setattr(
        environment, '_installed_cellacdc_spec', lambda: 'cellacdc==1.7.0'
    )
    assert not environment.install_cellacdc_cli(
        'model-python', logger_func=Mock(), reinstall=True
    )
    run.assert_called_once()


@pytest.mark.parametrize('reinstall', [False, True])
def test_install_cellacdc_cli_relinks_source_checkout(
        monkeypatch, tmp_path, reinstall,
    ):
    from unittest.mock import Mock
    from cellacdc.service import environment

    calls = []
    monkeypatch.setattr(
        environment, '_get_cellacdc_source_root', lambda: str(tmp_path)
    )
    monkeypatch.setattr(
        environment, '_uninstall_shadowing_cellacdc',
        lambda python, logger: calls.append(('uninstall', python)),
    )
    monkeypatch.setattr(
        environment, 'link_cellacdc_to_env',
        lambda python, logger_func: calls.append(('link', python)) or True,
    )
    monkeypatch.setattr(
        environment, 'run_streaming',
        lambda cmd, logger: calls.append(('pip', cmd[5:])) or 0,
    )
    monkeypatch.setattr(
        environment, '_installed_cellacdc_spec',
        Mock(side_effect=AssertionError('A checkout is linked, not installed')),
    )

    assert environment.install_cellacdc_cli(
        'model-python', logger_func=Mock(), reinstall=reinstall
    )
    # A copy installed in the environment would hide the linked checkout
    assert calls == (
        [('uninstall', 'model-python')] if reinstall else []
    ) + [
        ('link', 'model-python'),
        ('pip', environment.WORKER_RUNTIME_DEPENDENCIES),
    ]


@pytest.mark.parametrize('imported', ['env_copy', 'other_checkout', 'this_checkout', 'none'])
def test_uninstall_shadowing_cellacdc_removes_only_copies_in_env(
        monkeypatch, tmp_path, imported,
    ):
    from unittest.mock import Mock
    from cellacdc.service import environment, protocol

    prefix = tmp_path / 'env'
    checkout = tmp_path / 'checkout'
    files = {
        'env_copy': prefix / 'Lib' / 'site-packages' / 'cellacdc' / '__init__.py',
        'other_checkout': tmp_path / 'other' / 'cellacdc' / '__init__.py',
        'this_checkout': checkout / 'cellacdc' / '__init__.py',
    }
    identity = None if imported == 'none' else {
        'protocol': protocol.PROTOCOL_VERSION, 'version': '1.7.0',
        'file': str(files[imported]), 'prefix': str(prefix),
    }
    monkeypatch.setattr(
        environment, '_get_cellacdc_source_root', lambda: str(checkout)
    )
    monkeypatch.setattr(
        environment, '_read_cellacdc_runtime',
        lambda python: (identity, True, ''),
    )
    run = Mock(return_value=0)
    monkeypatch.setattr(environment, 'run_streaming', run)

    environment._uninstall_shadowing_cellacdc('model-python', Mock())

    if imported == 'env_copy':
        run.assert_called_once()
        assert run.call_args.args[0] == [
            'model-python', '-m', 'pip', 'uninstall', '-y',
            '--disable-pip-version-check', 'cellacdc',
        ]
    else:
        run.assert_not_called()


@pytest.mark.parametrize('automatic_save', [False, True])
def test_model_client_preserves_null_frame_and_sam_embeddings(automatic_save):
    from unittest.mock import Mock
    from cellacdc.service import payload

    labels = np.ones((2, 2), dtype=np.uint32)
    returned_embeddings = {
        0: {
            'original_size': (2, 2),
            'input_size': (2, 2),
            'features': np.ones((1, 2), dtype=np.float32),
            'is_image_set': True,
        }
    }
    result_data = payload.encode_value_body({
        'labels': labels, 'embeddings': returned_embeddings,
    })
    posted = {}

    class Response:
        ok = True
        content = result_data
        headers = {}
        url = 'http://worker/segment'

    class Session:
        def post(self, url, files, timeout):
            assert url == 'http://worker/segment'
            posted['parts'] = sorted(files)
            posted['params'] = payload.decode(files['params'][1])
            posted['embeddings'] = payload.decode_value_body(
                files['embeddings'][1]
            )
            return Response()

    class PosData:
        sam_embeddings = {None: dict(returned_embeddings)}
        loadSamEmbeddings = Mock()
        saveSamEmbeddings = Mock()

    client = ModelClient.__new__(ModelClient)
    client.base_url = 'http://worker'
    client.timeout = None
    client.session = Session()
    if automatic_save:
        client.model_name = 'segment_anything'
    pos_data = PosData()

    result = client.segment(
        np.zeros((2, 2)), frame_i=None, posData=pos_data,
        save_embeddings=not automatic_save, use_loaded_embeddings=True,
    )

    # Every argument is an attachment (no form fields)
    assert posted['parts'] == ['embeddings', 'image', 'params']
    assert posted['params']['frame_i'] is None
    assert posted['params']['save_embeddings'] is True
    assert posted['params']['use_loaded_embeddings'] is True
    # The loaded embeddings of the frame are sent to the worker
    assert set(posted['embeddings']) == {0}
    np.testing.assert_array_equal(
        posted['embeddings'][0]['features'], returned_embeddings[0]['features']
    )
    pos_data.loadSamEmbeddings.assert_called_once()
    assert pos_data.saveSamEmbeddings.call_count == int(automatic_save)
    np.testing.assert_array_equal(result, labels)
    np.testing.assert_array_equal(
        pos_data.sam_embeddings[None][0]['features'],
        returned_embeddings[0]['features'],
    )


def test_ensure_torch_build_choice_and_qt_command(monkeypatch):
    from unittest.mock import Mock
    from cellacdc.service import environment

    monkeypatch.setattr(
        environment.subprocess, 'run',
        lambda *args, **kwargs: type('Result', (), {'returncode': 1})(),
    )
    run = Mock(return_value=0)
    monkeypatch.setattr(environment, 'run_streaming', run)
    links = ([('CPU', 'https://example/cpu'), ('CUDA', 'https://example/cu')], 1)

    assert environment.ensure_torch(
        'model-python', prompt_func=lambda prompt: prompt.default,
        possible_links=links,
    )
    run.assert_called_with([
        'model-python', '-m', 'pip', 'install', 'torch', 'torchvision',
        '--index-url', 'https://example/cu',
    ], print, check=True)

    run.reset_mock()
    assert environment.ensure_torch(
        'model-python',
        prompt_func=lambda prompt: '"model python" -m pip install torch',
        possible_links=links,
    )
    command = run.call_args.args[0]
    if os.name == 'nt':
        assert command == '"model python" -m pip install torch'
    else:
        assert command == ['model python', '-m', 'pip', 'install', 'torch']


def test_uploaded_model_files_live_until_model_is_released(monkeypatch):
    import gc
    import types
    from pathlib import Path
    from cellacdc import myutils_models
    from cellacdc.service import payload, worker

    class Model:
        def __init__(
                self, model_path: os.PathLike = '',
                denoise_model_path: os.PathLike = '',
                config: str = '',
            ):
            self.model_path = model_path
            self.denoise_model_path = denoise_model_path
            self.config = config
            assert Path(model_path).read_bytes() == b'model'
            assert Path(denoise_model_path).read_bytes() == b'denoiser'

        def segment(self, image):
            return Path(self.model_path).read_bytes()

    module = types.SimpleNamespace(Model=Model)
    monkeypatch.setattr(
        myutils_models, 'model_init_requires_segm_data', lambda _: False
    )
    model = worker.instantiate_model(module, {
        'model_path': payload.ModelFile('weights.pt', b'model'),
        'denoise_model_path': payload.ModelFile('weights.pt', b'denoiser'),
        # Not path-typed: uploaded files are accepted for every argument
        'config': payload.ModelFile('config (1).yaml', b'cfg'),
    })
    model_path = Path(model.model_path)
    denoise_path = Path(model.denoise_model_path)
    config_path = Path(model.config)
    assert model_path != denoise_path
    assert config_path.name == 'config (1).yaml'
    assert config_path.read_bytes() == b'cfg'
    assert model.segment(None) == b'model'
    gc.collect()
    # The files live as long as the model
    assert model_path.exists() and denoise_path.exists() and config_path.exists()
    assert model.segment(None) == b'model'
    del model
    gc.collect()
    assert not model_path.exists()
    assert not denoise_path.exists()
    assert not config_path.exists()


@pytest.mark.parametrize('name', ['../weights.pt', r'..\weights.pt', 'C:weights.pt', ''])
def test_model_file_rejects_invalid_names(name):
    from cellacdc.service.payload import ModelFile

    with pytest.raises(ValueError, match='Invalid uploaded model filename'):
        ModelFile(name, b'weights')


@pytest.mark.parametrize('file_param', ['model_path', 'input_points_path'])
def test_remote_backend_uploads_and_reinitializes_model_files(
        monkeypatch, tmp_path, file_param,
    ):
    """Over real HTTP: files selected on the client are uploaded at init and
    re-init, and the worker's temporary copies live as long as the model."""
    import gc
    import types
    from pathlib import Path
    from fastapi import BackgroundTasks
    from cellacdc import _types, myutils_models
    from cellacdc.service import client as client_module, worker

    worker_paths = []

    class Model:
        def __init__(
                self, model_path: os.PathLike = '',
                input_points_path: _types.CsvFilePath = '',
            ):
            self.model_path = Path(model_path or input_points_path)
            worker_paths.append(self.model_path)
            if self.model_path.read_bytes() == b'bad':
                raise ValueError('Invalid model weights')

        def segment(self, image):
            return np.full(image.shape, self.model_path.read_bytes()[0])

    monkeypatch.setattr(
        myutils_models, 'import_segment_module',
        lambda _: types.SimpleNamespace(Model=Model)
    )
    monkeypatch.setattr(
        myutils_models, 'model_init_requires_segm_data', lambda _: False
    )
    app = worker.create_app('fake', defer_init=True)
    shutdown = next(
        route.endpoint for route in app.routes
        if getattr(route, 'path', None) == '/shutdown'
    )
    # Record what the client sends to /init (and send it over real HTTP)
    sent = []
    real_post = client_module.requests.post

    def post(url, files=None, **kwargs):
        sent.append((url, sorted(files), files['kwargs'][1]))
        return real_post(url, files=files, **kwargs)

    monkeypatch.setattr(client_module.requests, 'post', post)
    url, stop = _serve_app(app)
    try:
        backend = client_module.RemoteBackend(url)
        path = tmp_path / (
            'points.csv' if file_param == 'input_points_path'
            else 'FUCCI (1).txt'
        )
        path.write_bytes(b'\x01model')
        client = backend.get_client(init_model_kwargs={file_param: path})
        path.unlink()
        assert [(u, parts) for u, parts, _ in sent] == [(f'{url}/init', ['kwargs'])]
        # The worker receives the file's data, never the client's path
        for text in (str(tmp_path), json.dumps(str(tmp_path))[1:-1]):
            assert text.encode('utf-8') not in sent[0][2]
        assert worker_paths[0].name == path.name
        assert worker_paths[0].parent != path.parent

        image = np.zeros((2, 2))
        np.testing.assert_array_equal(client.segment(image), np.ones((2, 2)))
        path.write_bytes(b'\x02model')
        client_module.post_init(backend.url, {file_param: str(path)})
        gc.collect()
        assert not worker_paths[0].exists()
        np.testing.assert_array_equal(client.segment(image), np.full((2, 2), 2))
        path.write_bytes(b'bad')
        with pytest.raises(RuntimeError, match='Invalid model weights'):
            client_module.post_init(backend.url, {file_param: str(path)})
        assert not worker_paths[-1].exists()
        assert worker_paths[1].exists()
        np.testing.assert_array_equal(client.segment(image), np.full((2, 2), 2))
        # Called directly: over HTTP, /shutdown would stop the test process
        result = shutdown(BackgroundTasks())
        assert result == {'status': 'shutting down'}
        assert not worker_paths[1].exists()
        client.close()
        backend.stop()
    finally:
        stop()


def test_worker_passes_uploaded_files_to_model_calls(monkeypatch, tmp_path):
    """Over real HTTP: files on the client's disk in the kwargs of /segment,
    /segment3DT, /add_prompt and background jobs reach the model as
    temporary copies (same name and bytes), deleted after the call. Folders
    are not uploaded."""
    import time
    import types
    from pathlib import Path
    from cellacdc import myutils_models
    from cellacdc.service import worker

    calls = []

    def record(method, weights, folder=None):
        path = Path(weights)
        calls.append({
            'method': method, 'path': path, 'is_file': path.is_file(),
            'data': path.read_bytes(), 'folder': folder,
        })

    class Model:
        def __init__(self):
            pass

        def segment(self, image, weights=None, folder=None):
            record('segment', weights, folder)
            return np.ones(image.shape, np.uint32)

        def segment3DT(self, video, weights=None):
            record('segment3DT', weights)
            return np.ones(video.shape, np.uint32)

        def add_prompt(self, prompt, prompt_id, weights=None):
            record('add_prompt', weights)

    monkeypatch.setattr(
        myutils_models, 'import_segment_module',
        lambda name: types.SimpleNamespace(Model=Model),
    )
    monkeypatch.setattr(worker, '_jobs', {})
    weights = tmp_path / 'my weights (1).pt'
    weights.write_bytes(b'\x00weights\xff')
    image = np.zeros((2, 2), np.uint8)
    url, stop = _serve_app(worker.create_app('fake'))
    try:
        client = ModelClient(url, timeout=60)
        assert client.is_segment3DT_available
        client.segment(image, weights=str(weights), folder=str(tmp_path))
        client.segment3DT(image[None], weights=weights)
        client.add_prompt(prompt=(0, 1, 1), prompt_id=1, weights=weights)
        job_id = client.submit_job(image, weights=str(weights))
        for _ in range(100):
            status = client.get_job_status(job_id)
            if status['status'] in ('completed', 'failed'):
                break
            time.sleep(0.05)
        assert status['status'] == 'completed', status
        np.testing.assert_array_equal(client.get_job_result(job_id), np.ones((2, 2)))
        client.close()
    finally:
        stop()

    assert [call['method'] for call in calls] == [
        'segment', 'segment3DT', 'add_prompt', 'segment',
    ]
    for call in calls:
        assert call['is_file']
        assert call['data'] == weights.read_bytes()
        assert call['path'].name == weights.name
        # A copy on the worker, not the client's file
        assert call['path'].parent != weights.parent
        # Deleted after the call
        assert not call['path'].exists()
        assert not call['path'].parent.exists()
    # Folders are sent as their path
    assert calls[0]['folder'] == str(tmp_path)


def test_instantiate_model_rejects_raw_points_csv_path():
    import types
    from cellacdc import _types
    from cellacdc.service.worker import instantiate_model

    class Model:
        def __init__(self, input_points_path: _types.CsvFilePath = ''):
            raise AssertionError('Raw client paths must not reach the model')

        def segment(self, image):
            return image

    with pytest.raises(ValueError, match=r"\['input_points_path'\].*not uploaded"):
        instantiate_model(
            types.SimpleNamespace(Model=Model),
            {'input_points_path': 'points.csv'},
        )


def _fake_segment_module(requires_segm):
    import types

    class Model:
        if requires_segm:
            def __init__(self, segm_data, offset=0):
                self.segm_data, self.offset = segm_data, offset
        else:
            def __init__(self, offset=0):
                self.segm_data, self.offset = None, offset

        def segment(self, image):
            if self.segm_data is not None:
                return self.segm_data + self.offset
            return np.zeros(image.shape, dtype=np.uint32)

    return types.SimpleNamespace(Model=Model)


def test_instantiate_model_with_segm_data():
    from cellacdc.service.worker import instantiate_model

    module = _fake_segment_module(requires_segm=True)
    segm = np.ones((2, 2), dtype=np.uint32)
    model = instantiate_model(
        module, {'segm_data': segm, 'offset': 1, 'is_rgb': True}
    )
    np.testing.assert_array_equal(model.segm_data, segm)
    assert model.offset == 1
    with pytest.raises(ValueError, match='auxiliary segmentation'):
        instantiate_model(module, {'offset': 1})
    with pytest.raises(ValueError, match='client files'):
        instantiate_model(module, {'segm_endname': 'segm'})
    with pytest.raises(ValueError, match='client files'):
        instantiate_model(
            module, {'segm_data': segm, 'input_points_path': 'pts.csv'}
        )
    # Empty selections mean "no file" and are dropped
    model = instantiate_model(module, {
        'segm_data': segm, 'input_points_path': '', 'segm_endname': 'None'
    })
    assert not hasattr(model, 'input_points_path')
    model = instantiate_model(
        _fake_segment_module(requires_segm=False), {'segm_data': segm}
    )
    assert model.segm_data is None


class _CapturingExecutor:
    """Stands in for the worker's thread pool; jobs run when the test says."""
    def __init__(self):
        self.calls = []

    def submit(self, fn, *args):
        self.calls.append((fn, args))


class _FakeUpload:
    def __init__(self, arr):
        buf = io.BytesIO()
        np.save(buf, arr)
        self._content = buf.getvalue()

    async def read(self):
        return self._content


class _FakeUploadBytes:
    def __init__(self, content):
        self._content = content

    async def read(self):
        return self._content


def _kwargs_upload(kwargs=None):
    """Model kwargs as the attachment sent by the client (`payload.encode`)."""
    from cellacdc.service import payload

    return _FakeUploadBytes(payload.encode(kwargs or {}))


def _segment_request(arr, kwargs=None, path='/segment'):
    """Arguments of a direct call of the `/segment`-like endpoint `path`.

    `embeddings` is passed explicitly for `/segment`: called directly, the
    endpoint would receive FastAPI's `File(None)` default instead of None.
    """
    request = {'image': _FakeUpload(arr), 'params': _kwargs_upload(kwargs)}
    if path == '/segment':
        request['embeddings'] = None
    return request


def _jobs_app(monkeypatch, segment, **extra_methods):
    import types
    from cellacdc import myutils_models
    from cellacdc.service import worker

    class Model:
        def __init__(self):
            pass

        def segment(self, image):
            return segment(image)

    for name, func in extra_methods.items():
        setattr(Model, name, lambda self, *args, _f=func: _f(*args))

    module = types.SimpleNamespace(Model=Model)
    monkeypatch.setattr(
        myutils_models, 'import_segment_module', lambda name: module
    )
    executor = _CapturingExecutor()
    monkeypatch.setattr(worker, '_executor', executor)
    monkeypatch.setattr(worker, '_jobs', {})
    app = worker.create_app('fake')
    endpoints = {
        (route.path, tuple(sorted(route.methods))): route.endpoint
        for route in app.routes if hasattr(route, 'methods')
    }
    return endpoints, executor


def test_worker_rejects_all_nonempty_path_typed_init_kwargs(monkeypatch, tmp_path):
    import types
    from pathlib import Path
    from cellacdc import _types, myutils_models
    from cellacdc.service import payload, worker

    class Model:
        def __init__(
                self, model_path: os.PathLike = '',
                model_folder: _types.FolderPath = '',
            ):
            self.model_path = model_path
            self.model_folder = model_folder

        def segment(self, image):
            return image

    module = types.SimpleNamespace(Model=Model)
    monkeypatch.setattr(
        myutils_models, 'model_init_requires_segm_data', lambda _: False
    )

    with pytest.raises(ValueError, match='model_path'):
        worker.instantiate_model(module, {'model_path': '/client/model.pth'})
    # Folders are never uploaded, so the worker cannot use them
    with pytest.raises(ValueError, match="model_folder.*not uploaded"):
        worker.instantiate_model(module, {'model_folder': str(tmp_path)})
    with pytest.raises(ValueError, match=r"\['model_folder', 'model_path'\]"):
        worker.instantiate_model(module, {
            'model_path': '/client/model.pth', 'model_folder': str(tmp_path),
        })
    assert worker.instantiate_model(module, {}).model_path == ''
    for empty in ('', 'None', None):
        assert worker.instantiate_model(
            module, {'model_path': empty}
        ).model_path == empty
    # Uploaded files are accepted
    model = worker.instantiate_model(module, {
        'model_path': payload.ModelFile('model.pth', b'weights'),
    })
    assert Path(model.model_path).read_bytes() == b'weights'


def _submit_job(endpoints, arr):
    import asyncio
    submit = endpoints[('/jobs/segment', ('POST',))]
    return asyncio.run(submit(
        **_segment_request(arr, path='/jobs/segment')
    ))['job_id']


def test_job_cancelled_while_pending_is_never_started(monkeypatch):
    calls = []
    endpoints, executor = _jobs_app(
        monkeypatch, lambda image: calls.append(image) or image
    )
    job_id = _submit_job(endpoints, np.zeros((2, 2)))
    cancel = endpoints[('/jobs/{job_id}', ('DELETE',))]
    status = endpoints[('/jobs/{job_id}', ('GET',))]

    assert cancel(job_id)['status'] == 'cancelled'
    fn, args = executor.calls[0]
    fn(*args)

    assert calls == []
    assert status(job_id).status == 'cancelled'


@pytest.mark.parametrize('fails', [False, True])
def test_job_cancelled_while_running_stays_cancelled(monkeypatch, fails):
    import threading

    started = threading.Event()
    proceed = threading.Event()

    def segment(image):
        started.set()
        assert proceed.wait(5)
        if fails:
            raise RuntimeError('model failed')
        return image + 1

    endpoints, executor = _jobs_app(monkeypatch, segment)
    job_id = _submit_job(endpoints, np.zeros((2, 2)))
    cancel = endpoints[('/jobs/{job_id}', ('DELETE',))]
    status = endpoints[('/jobs/{job_id}', ('GET',))]
    result = endpoints[('/jobs/{job_id}/result', ('GET',))]

    fn, args = executor.calls[0]
    thread = threading.Thread(target=fn, args=args)
    thread.start()
    assert started.wait(5)
    assert status(job_id).status == 'running'
    assert cancel(job_id)['status'] == 'cancelled'
    proceed.set()
    thread.join(5)

    job = status(job_id)
    assert job.status == 'cancelled'
    assert job.error is None
    from fastapi import HTTPException
    with pytest.raises(HTTPException):
        result(job_id)


def test_cancel_deletes_finished_job(monkeypatch):
    endpoints, executor = _jobs_app(monkeypatch, lambda image: image + 1)
    job_id = _submit_job(endpoints, np.zeros((2, 2)))
    fn, args = executor.calls[0]
    fn(*args)
    cancel = endpoints[('/jobs/{job_id}', ('DELETE',))]
    status = endpoints[('/jobs/{job_id}', ('GET',))]

    out = cancel(job_id)
    assert out['status'] == 'completed' and out['deleted'] is True
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as err:
        status(job_id)
    assert err.value.status_code == 404


def test_job_result_is_fetched_once_and_finished_jobs_are_bounded(monkeypatch):
    from fastapi import HTTPException
    from cellacdc.service import worker

    monkeypatch.setattr(worker, 'MAX_FINISHED_JOBS', 2)
    endpoints, executor = _jobs_app(monkeypatch, lambda image: image + 1)
    status = endpoints[('/jobs/{job_id}', ('GET',))]
    result = endpoints[('/jobs/{job_id}/result', ('GET',))]

    job_ids = [_submit_job(endpoints, np.zeros((2, 2))) for _ in range(4)]
    for fn, args in executor.calls:
        fn(*args)
    # Only the newest MAX_FINISHED_JOBS finished jobs are kept
    assert list(worker._jobs) == job_ids[2:]
    with pytest.raises(HTTPException):
        status(job_ids[0])

    labels = np.load(io.BytesIO(result(job_ids[3]).body))
    np.testing.assert_array_equal(labels, np.ones((2, 2)))
    with pytest.raises(HTTPException) as err:
        result(job_ids[3])
    assert err.value.status_code == 404
    assert list(worker._jobs) == job_ids[2:3]


def test_worker_runs_model_specific_methods(monkeypatch):
    """second_ch_img_to_stack/predictCcaState run the model's own code."""
    import asyncio
    import pandas as pd
    from fastapi import HTTPException
    from cellacdc.core import getBaseCca_df

    def to_stack(first, second):
        return np.stack([first, second]).astype(np.float32)

    def predict(image, lab):
        df = getBaseCca_df([1, 2])
        df.at[2, 'relationship'] = 'bud'
        return df

    endpoints, _ = _jobs_app(
        monkeypatch, lambda image: image,
        second_ch_img_to_stack=to_stack, predictCcaState=predict,
    )
    first, second = np.zeros((3, 3), np.uint8), np.ones((3, 3), np.uint8)
    res = asyncio.run(endpoints[('/second_ch_img_to_stack', ('POST',))](
        first=_FakeUpload(first), second=_FakeUpload(second)
    ))
    stack = np.load(io.BytesIO(res.body))
    assert stack.shape == (2, 3, 3) and stack.dtype == np.float32

    res = asyncio.run(endpoints[('/predictCcaState', ('POST',))](
        image=_FakeUpload(first), precomputed_lab=_FakeUpload(second)
    ))
    cca_df = pd.read_json(io.StringIO(res['cca_df']), orient='table')
    pd.testing.assert_frame_equal(cca_df, predict(None, None))

    info = endpoints[('/info', ('GET',))]()
    assert info.capabilities.second_ch_img_to_stack
    assert info.capabilities.predictCcaState

    endpoints, _ = _jobs_app(monkeypatch, lambda image: image)
    with pytest.raises(HTTPException) as err:
        asyncio.run(endpoints[('/second_ch_img_to_stack', ('POST',))](
            first=_FakeUpload(first), second=_FakeUpload(second)
        ))
    assert err.value.status_code == 400
    assert not endpoints[('/info', ('GET',))]().capabilities.predictCcaState


@pytest.mark.parametrize('method', ['segment', 'segment3DT', 'submit_job'])
def test_model_client_large_array_kwargs_over_http(monkeypatch, method):
    import asyncio
    import types
    from cellacdc import myutils, myutils_models
    from cellacdc.service import worker

    image = np.zeros((1024, 1024), dtype=np.uint8)
    lab = np.arange(image.size, dtype=np.uint32).reshape(image.shape)
    assert len(json.dumps({'lab': lab.tolist()})) > 1024 * 1024
    seen = []

    class Model:
        def segment(self, image, lab, frame_i=None):
            assert isinstance(lab, np.ndarray)
            assert lab.dtype == np.uint32
            np.testing.assert_array_equal(lab, expected_lab)
            seen.append(frame_i)
            return lab

        segment3DT = segment

    expected_lab = lab.copy()
    monkeypatch.setattr(
        myutils_models, 'import_segment_module',
        lambda name: types.SimpleNamespace(Model=Model),
    )
    executor = _CapturingExecutor()
    monkeypatch.setattr(worker, '_executor', executor)
    monkeypatch.setattr(worker, '_jobs', {})
    app = worker.create_app('fake')
    endpoints = {
        route.path: route.endpoint
        for route in app.routes if hasattr(route, 'methods')
    }

    class Session:
        def post(self, url, files, timeout):
            # Only attachments, and no embeddings (not an embedding operation)
            assert sorted(files) == ['image', 'params']
            params = files['params'][1]
            # The array is in the `.npz` part, not in the (small) JSON part
            assert int.from_bytes(params[:8], 'big') < 1024
            assert len(params) > 1024 * 1024
            path = url.removeprefix('http://testserver')
            request = {
                'image': _FakeUploadBytes(files['image'][1]),
                'params': _FakeUploadBytes(params),
            }
            if path == '/segment':
                request['embeddings'] = None
            response = asyncio.run(endpoints[path](**request))
            return self._response(url, response)

        def get(self, url, timeout):
            job_id = url.split('/')[-2]
            response = endpoints['/jobs/{job_id}/result'](job_id)
            return self._response(url, response)

        @staticmethod
        def _response(url, response):
            result = requests.Response()
            result.status_code = getattr(response, 'status_code', 200)
            result.url = url
            if isinstance(response, dict):
                result._content = json.dumps(response).encode()
                result.headers['Content-Type'] = 'application/json'
            else:
                result._content = response.body
                result.headers.update(response.headers)
            return result

    client = ModelClient.__new__(ModelClient)
    client.base_url = 'http://testserver'
    client.timeout = None
    client.info = InfoResponse(
        name='fake', capabilities=Capabilities(segment3DT=True)
    )
    client.is_segment3DT_available = True
    client.model_name = 'fake'
    client.session = Session()
    result = getattr(client, method)(image, lab=lab)
    if method == 'submit_job':
        fn, args = executor.calls.pop()
        fn(*args)
        result = client.get_job_result(result)

    np.testing.assert_array_equal(result, lab)
    assert seen == [None]


def test_worker_runs_promptable_model_protocol(monkeypatch):
    import asyncio
    import types
    from cellacdc import myutils_models
    from cellacdc.service import payload, worker

    models = []

    class Model:
        def __init__(self):
            self.prompts = []
            models.append(self)

        def add_prompt(self, prompt, prompt_id, image, **kwargs):
            self.prompts.append((prompt, prompt_id, image.copy()))

        def segment(self, image, lab=None):
            assert self.prompts
            return np.full_like(image, self.prompts[-1][1], dtype=np.uint32)

    module = types.SimpleNamespace(Model=Model)
    monkeypatch.setattr(
        myutils_models, 'import_promptable_segment_module', lambda name: module
    )
    app = worker.create_app('fake', model_kind='promptable')
    endpoints = {
        route.path: route.endpoint
        for route in app.routes if hasattr(route, 'methods')
    }
    image = np.arange(4, dtype=np.uint8).reshape(2, 2)
    body = payload.encode({
        'prompt': (0, 1, 1), 'prompt_id': 7, 'image': image,
    })

    result = asyncio.run(endpoints['/add_prompt'](
        kwargs=_FakeUploadBytes(body)
    ))
    assert result == {'status': 'ok'}
    response = asyncio.run(endpoints['/segment'](
        **_segment_request(image, {'lab': None})
    ))
    np.testing.assert_array_equal(
        np.load(io.BytesIO(response.body)), np.full((2, 2), 7)
    )
    # The model got the prompt, with its image array
    assert len(models) == 1 and len(models[0].prompts) == 1
    prompt, prompt_id, prompt_image = models[0].prompts[0]
    assert prompt == [0, 1, 1] and prompt_id == 7
    np.testing.assert_array_equal(prompt_image, image)
    assert prompt_image.dtype == np.uint8


@pytest.mark.parametrize('num_slices,single_slice_payload', [
    (1, False), (1, True), (500, False),
])
@pytest.mark.parametrize('start_z_slice', [0, 5])
@pytest.mark.parametrize('operation', [
    'save_embeddings', 'only_embeddings', 'use_loaded_embeddings',
])
def test_worker_roundtrips_sam_embeddings(
        monkeypatch, num_slices, single_slice_payload, start_z_slice, operation,
    ):
    import asyncio
    import http.client
    import types
    from cellacdc import myutils_models
    from cellacdc.service import payload, worker

    class Predictor:
        original_size = (2, 2)
        input_size = (2, 2)
        features = np.ones((1, 2), dtype=np.float32)
        is_image_set = True
        _crop_embeddings = {
            'crop-digest': {
                'original_size': (1, 1),
                'input_size': (1, 1),
                'features': np.full((1, 2), 7, dtype=np.float32),
                'is_image_set': True,
            },
        }

    class Model:
        def __init__(self):
            self.model = Predictor()

        def segment(
                self, image, frame_i, posData, save_embeddings=False,
                only_embeddings=False, use_loaded_embeddings=False,
                start_z_slice=0,
            ):
            assert frame_i is None
            for z in range(start_z_slice, start_z_slice + num_slices):
                if use_loaded_embeddings:
                    loaded = posData.getSamEmbeddings(frame_i, z=z)
                    np.testing.assert_array_equal(
                        loaded['features'], np.zeros((1, 2), dtype=np.float32)
                    )
                if save_embeddings or only_embeddings:
                    posData.storeSamEmbeddings(self, frame_i, z=z)
            return np.ones_like(image, dtype=np.uint32)

    module = types.SimpleNamespace(Model=Model)
    monkeypatch.setattr(myutils_models, 'import_segment_module', lambda name: module)
    monkeypatch.setitem(
        sys.modules, 'torch',
        types.SimpleNamespace(as_tensor=lambda value, device=None: value),
    )
    app = worker.create_app('fake')
    endpoint = next(
        route.endpoint for route in app.routes
        if getattr(route, 'path', None) == '/segment'
    )
    frame_embeddings = {
        z: {
            'original_size': (2, 2),
            'input_size': (2, 2),
            'features': np.zeros((1, 2), dtype=np.float32),
            'is_image_set': True,
            'crops': Predictor._crop_embeddings,
        } for z in range(start_z_slice, start_z_slice + num_slices)
    }
    if single_slice_payload:
        frame_embeddings = frame_embeddings[start_z_slice]

    class Session:
        def post(self, url, files, timeout):
            # Embedding operations always send the (possibly empty)
            # embeddings, as an attachment like every other argument
            assert sorted(files) == ['embeddings', 'image', 'params']
            response = asyncio.run(endpoint(
                image=_FakeUploadBytes(files['image'][1]),
                params=_FakeUploadBytes(files['params'][1]),
                embeddings=_FakeUploadBytes(files['embeddings'][1]),
            ))
            assert 'x-cell-acdc-payload' not in response.headers
            header_bytes = b''.join(
                key + b': ' + value + b'\r\n'
                for key, value in response.raw_headers
            ) + b'\r\n'
            # Exercise the same header-line limit used by Requests.
            parsed_headers = http.client.parse_headers(io.BytesIO(header_bytes))
            result = requests.Response()
            result.status_code = response.status_code
            result.url = url
            result.headers.update(parsed_headers.items())
            result._content = response.body
            return result

    pos_data = types.SimpleNamespace(sam_embeddings={None: frame_embeddings})
    client = ModelClient.__new__(ModelClient)
    client.base_url = 'http://worker'
    client.timeout = None
    client.session = Session()
    shape = (2, 2) if single_slice_payload else (num_slices, 2, 2)
    image = np.zeros(shape, dtype=np.uint8)
    result = client.segment(
        image, frame_i=None, posData=pos_data, start_z_slice=start_z_slice,
        **{operation: True},
    )
    np.testing.assert_array_equal(result, np.ones_like(image, dtype=np.uint32))
    assert result.dtype == np.uint32
    returned_embeddings = pos_data.sam_embeddings[None]
    assert set(returned_embeddings) == set(
        range(start_z_slice, start_z_slice + num_slices)
    )
    for embedding in returned_embeddings.values():
        assert embedding['original_size'] == (2, 2)
        assert embedding['input_size'] == (2, 2)
        assert embedding['is_image_set'] is True
        crop = embedding['crops']['crop-digest']
        assert crop['original_size'] == (1, 1)
        assert crop['is_image_set'] is True
        np.testing.assert_array_equal(crop['features'], np.full((1, 2), 7))
        np.testing.assert_array_equal(
            embedding['features'],
            np.zeros((1, 2), dtype=np.float32)
            if operation == 'use_loaded_embeddings' else Predictor.features,
        )
        assert embedding['features'].dtype == np.float32
    if num_slices == 500:
        metadata, _ = payload.encode_value({
            'labels': result, 'embeddings': returned_embeddings,
        })
        assert len(metadata.encode('utf-8')) > 64 * 1024


@pytest.mark.parametrize('path', [
    '/init', '/segment', '/segment3DT', '/second_ch_img_to_stack',
    '/predictCcaState',
])
def test_worker_model_operations_leave_event_loop_responsive(monkeypatch, path):
    import asyncio
    import threading
    import pandas as pd
    from cellacdc.service import worker

    started = threading.Event()
    release = threading.Event()
    operation_threads = []

    def operation(*args):
        operation_threads.append(threading.get_ident())
        started.set()
        if not release.wait(5):
            raise RuntimeError('Event loop did not release the model operation')
        if path == '/predictCcaState':
            return pd.DataFrame({'state': [1]})
        return args[0]

    endpoints, _ = _jobs_app(
        monkeypatch, operation, segment3DT=operation,
        second_ch_img_to_stack=operation, predictCcaState=operation,
    )
    instantiate = worker.instantiate_model

    def initialize(module, kwargs):
        operation(None)
        return instantiate(module, kwargs)

    monkeypatch.setattr(worker, 'instantiate_model', initialize)
    arr = np.ones((2, 2))
    if path == '/init':
        kwargs = {'kwargs': _kwargs_upload()}
    elif path == '/second_ch_img_to_stack':
        kwargs = {'first': _FakeUpload(arr), 'second': _FakeUpload(arr)}
    elif path == '/predictCcaState':
        kwargs = {
            'image': _FakeUpload(arr), 'precomputed_lab': _FakeUpload(arr)
        }
    else:
        kwargs = _segment_request(arr, path=path)

    async def exercise():
        loop_thread = threading.get_ident()
        task = asyncio.create_task(endpoints[(path, ('POST',))](**kwargs))
        try:
            async def wait_started():
                while not started.is_set():
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_started(), 2)
            assert len(operation_threads) == 1
            assert operation_threads[0] != loop_thread
            assert not task.done()
            assert endpoints[('/health', ('GET',))]().status == 'ok'
        finally:
            release.set()
            response = await task
        if path == '/init':
            assert response == {'status': 'initialized'}
        elif path == '/predictCcaState':
            result = pd.read_json(io.StringIO(response['cca_df']), orient='table')
            pd.testing.assert_frame_equal(result, pd.DataFrame({'state': [1]}))
        else:
            np.testing.assert_array_equal(np.load(io.BytesIO(response.body)), arr)

    asyncio.run(exercise())


@pytest.mark.parametrize('path', ['/init', '/segment'])
def test_worker_lock_wait_leaves_event_loop_responsive(monkeypatch, path):
    import asyncio
    import threading
    from cellacdc.service import worker

    started = threading.Event()
    release = threading.Event()

    def segment(image):
        started.set()
        if not release.wait(5):
            raise RuntimeError('Event loop did not release the background job')
        return image

    endpoints, executor = _jobs_app(monkeypatch, segment)
    arr = np.ones((2, 2))
    job_id = _submit_job(endpoints, arr)
    run_job, job_args = executor.calls[0]
    thread = threading.Thread(target=run_job, args=job_args)
    thread.start()
    try:
        assert started.wait(2)
        instantiate = worker.instantiate_model
        initialized = threading.Event()

        def initialize(module, kwargs):
            initialized.set()
            return instantiate(module, kwargs)

        monkeypatch.setattr(worker, 'instantiate_model', initialize)

        async def exercise():
            kwargs = (
                {'kwargs': _kwargs_upload()} if path == '/init'
                else _segment_request(arr, path=path)
            )
            task = asyncio.create_task(endpoints[(path, ('POST',))](**kwargs))
            try:
                await asyncio.sleep(0.1)
                assert not task.done()
                assert not initialized.is_set()
                assert endpoints[('/health', ('GET',))]().status == 'ok'
                assert endpoints[('/jobs/{job_id}', ('GET',))](job_id).status == 'running'
                assert endpoints[('/jobs/{job_id}', ('DELETE',))](job_id)['status'] == 'cancelled'
            finally:
                release.set()
                await task

        asyncio.run(exercise())
        assert endpoints[('/jobs/{job_id}', ('GET',))](job_id).status == 'cancelled'
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()


def test_worker_serializes_inference_calls(monkeypatch):
    import asyncio
    import threading
    import time

    active = 0
    max_active = 0
    active_lock = threading.Lock()

    def segment(image):
        nonlocal active, max_active
        with active_lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with active_lock:
            active -= 1
        return image

    endpoints, executor = _jobs_app(monkeypatch, segment)
    job_id = _submit_job(endpoints, np.zeros((2, 2)))
    run_job, job_args = executor.calls[0]
    barrier = threading.Barrier(2)

    def run_direct():
        barrier.wait()
        asyncio.run(endpoints[('/segment', ('POST',))](
            **_segment_request(np.zeros((2, 2)))
        ))

    def run_background():
        barrier.wait()
        run_job(*job_args)

    direct_thread = threading.Thread(target=run_direct)
    background_thread = threading.Thread(target=run_background)
    direct_thread.start()
    background_thread.start()
    direct_thread.join(5)
    background_thread.join(5)

    assert not direct_thread.is_alive()
    assert not background_thread.is_alive()
    assert max_active == 1


def test_worker_exposes_argument_specs_to_client(monkeypatch):
    import types
    from cellacdc import myutils_models
    from cellacdc.service import client as client_module
    from cellacdc.service import worker

    class Model:
        def __init__(self, diameter: int = 30):
            pass

        def segment(self, image, gauss_sigma: float = 0.5):
            return image

    module = types.SimpleNamespace(Model=Model, url_help=lambda: "https://model-help")
    monkeypatch.setattr(
        myutils_models, 'import_segment_module', lambda _: module
    )
    app = worker.create_app('fake')
    endpoints = {
        route.path: route.endpoint for route in app.routes
        if hasattr(route, 'methods')
    }
    spec_data = endpoints['/spec']()
    assert spec_data['url'] == "https://model-help"

    class Response:
        ok = True

        def json(self):
            return spec_data

    requested = []

    def get(url, headers=None, timeout=None):
        requested.append((url, headers))
        return Response()

    monkeypatch.setattr(client_module.requests, 'get', get)
    init_params, segment_params = client_module.get_model_arg_spec(
        'http://worker'
    )

    # Sent with the protocol version (the worker answers 426 otherwise)
    from cellacdc.service import protocol
    assert requested == [('http://worker/spec', protocol.headers())]
    assert next(p for p in init_params if p.name == 'diameter').type is int
    assert next(p for p in segment_params if p.name == 'gauss_sigma').type is float


def test_model_client_forwards_model_specific_methods(monkeypatch):
    import pandas as pd
    from cellacdc.core import getBaseCca_df

    cca_df = getBaseCca_df([1, 2])
    posted = []

    class FakeResponse:
        ok = True

        def __init__(self, content=b'', payload=None):
            self.content = content
            self._payload = payload

        def json(self):
            return self._payload

    class FakeSession:
        def post(self, url, files=None, timeout=None, **kwargs):
            posted.append((url.rsplit('/', 1)[1], sorted(files)))
            if url.endswith('/predictCcaState'):
                return FakeResponse(
                    payload={'cca_df': cca_df.to_json(orient='table')}
                )
            buf = io.BytesIO()
            np.save(buf, np.full((2, 3, 3), 7.0))
            return FakeResponse(content=buf.getvalue())

    client = ModelClient.__new__(ModelClient)
    client.base_url = 'http://x'
    client.timeout = None
    client.session = FakeSession()
    client.model_name = 'm'
    client.info = InfoResponse(
        name='m', capabilities=Capabilities(
            second_ch_img_to_stack=True, predictCcaState=True
        )
    )
    img = np.zeros((3, 3))
    assert client.second_ch_img_to_stack(img, img).shape == (2, 3, 3)
    pd.testing.assert_frame_equal(client.predictCcaState(img, img), cca_df)
    assert posted == [
        ('second_ch_img_to_stack', ['first', 'second']),
        ('predictCcaState', ['image', 'precomputed_lab']),
    ]

    client.info = InfoResponse(name='m', capabilities=Capabilities(
        second_ch_img_to_stack=False
    ))
    with pytest.raises(NotImplementedError):
        client.second_ch_img_to_stack(img, img)
    with pytest.raises(NotImplementedError):
        client.predictCcaState(img, img)


@pytest.mark.parametrize('backend_type', ['remote', 'local'])
def test_kernel_sends_data_and_init_kwargs_to_worker(monkeypatch, backend_type):
    """Workers (local and remote) get the loaded data and the init kwargs."""
    from cellacdc import myutils
    from cellacdc.service import client as client_module

    argspecs = [
        myutils.ArgSpec('diameter', 0, int, '', ''),
    ]
    segment_argspecs = [myutils.ArgSpec('threshold', 0.0, float, '', '')]
    monkeypatch.setattr(
        client_module, 'get_model_arg_spec',
        lambda url: (argspecs, segment_argspecs),
    )
    monkeypatch.setattr(
        myutils, 'getModelArgSpecFromEnv',
        lambda *args: (argspecs, segment_argspecs, None),
    )

    segm = np.ones((2, 2), dtype=np.uint32)
    received = {}

    def fake_resolve(posData, init_kwargs):
        init_kwargs.pop('segm_endname')
        return segm

    class FakeRemote:
        def __init__(self, url):
            pass

        def get_client(self, init_model_kwargs=None):
            received['kwargs'] = init_model_kwargs
            return object()

    class FakeLocal:
        def __init__(self, model_name, env_python, init_model_kwargs):
            received['kwargs'] = init_model_kwargs

        def start(self):
            return object()

    monkeypatch.setattr(myutils, 'resolve_segm_model_init_data', fake_resolve)
    monkeypatch.setattr(client_module, 'RemoteBackend', FakeRemote)
    monkeypatch.setattr(client_module, 'LocalBackend', FakeLocal)

    kernel = cli._WorkflowKernel(logger=None, log_path=None)
    kernel.init_backend_args(
        backend_type=backend_type,
        backend_url='http://gpu:8520' if backend_type == 'remote' else None,
        env_python=sys.executable if backend_type == 'local' else None,
    )
    kernel.model = None
    kernel.model_name = 'm'
    kernel.init_model_kwargs = {'segm_endname': 'segm', 'diameter': '30'}
    kernel.model_kwargs = {'threshold': '2.5'}
    kernel.signals = cli.KernelCliSignals(print)
    kernel.logger_func = print
    cli.SegmKernel.init_segm_model.__wrapped__(kernel, posData=object())

    assert received['kwargs']['diameter'] == 30
    assert received['kwargs']['segm_data'] is segm
    assert 'segm_endname' not in received['kwargs']
    assert kernel.model_kwargs['threshold'] == 2.5


def test_pkg_spec_accepts_distribution_name():
    from cellacdc import myutils

    specs = myutils._normalize_pkg_specs(
        [{'pkg_name': 'foo', 'distribution_name': 'foo-dist'}], '', '',
        {key: None for key in myutils._PKG_SPEC_KEYS},
    )
    assert specs[0]['distribution_name'] == 'foo-dist'
    assert specs[0]['pkg_name'] == 'foo'
    with pytest.raises(TypeError, match='Unknown'):
        myutils._normalize_pkg_specs([{'pkg_name': 'foo', 'bad': 1}], '', '', {})


def test_failed_torch_or_cellacdc_install_aborts_setup(monkeypatch):
    from cellacdc.service import environment

    monkeypatch.setattr(
        environment, 'load_model_env_config', lambda name: {'torch': True}
    )
    monkeypatch.setattr(environment, 'ensure_torch', lambda *a, **k: False)
    with pytest.raises(RuntimeError, match='PyTorch is not installed'):
        environment.ensure_model_dependencies('m', sys.executable)

    monkeypatch.setattr(
        environment, 'ensure_cellacdc_cli', lambda *a, **k: False
    )
    with pytest.raises(RuntimeError, match='Cell-ACDC could not be installed'):
        environment.ensure_worker_dependencies(sys.executable)

def test_local_backend_deferred_init_sends_segm_data():
    """Init data (arrays) is sent via POST /init, not as client file paths."""
    segm = np.arange(8, dtype=np.uint32).reshape(2, 2, 2)
    backend = LocalBackend(
        'skip_segmentation', env_python=sys.executable,
        init_model_kwargs={'segm_data': segm},
    )
    try:
        client = backend.start(timeout=60)
        assert '--defer-init' in backend.proc.args
        out = client.segment(np.zeros((2, 2), dtype=np.uint8), frame_i=1)
        np.testing.assert_array_equal(out, segm[1])
    finally:
        backend.stop()


def test_local_backend_deferred_init_error_is_reported():
    backend = LocalBackend(
        'skip_segmentation', env_python=sys.executable,
        init_model_kwargs={'segm_data': None, 'x': np.zeros(1)},
    )
    with pytest.raises(RuntimeError, match='Initializing the skip_segmentation'):
        backend.start(timeout=60)
    assert backend.proc is None or backend.proc.poll() is not None


def test_local_backend_reinitialize_keeps_process():
    """`reinitialize` replaces the model without restarting the worker."""
    first = np.full((1, 2, 2), 1, dtype=np.uint32)
    second = np.full((1, 2, 2), 2, dtype=np.uint32)
    backend = LocalBackend(
        'skip_segmentation', env_python=sys.executable,
        init_model_kwargs={'segm_data': first},
    )
    try:
        client = backend.start(timeout=60)
        pid = backend.proc.pid
        img = np.zeros((2, 2), dtype=np.uint8)
        np.testing.assert_array_equal(client.segment(img, frame_i=0), first[0])

        assert backend.reinitialize({'segm_data': second}) is client
        assert backend.proc.pid == pid and backend.is_running()
        np.testing.assert_array_equal(client.segment(img, frame_i=0), second[0])

        with pytest.raises(RuntimeError, match='Re-initializing'):
            backend.reinitialize({'segm_data': None, 'x': np.zeros(1)})
        # The previous model is still loaded
        np.testing.assert_array_equal(client.segment(img, frame_i=0), second[0])
    finally:
        backend.stop()
    assert not backend.is_running()


def _version_checked_backend(
        monkeypatch, versions, port=None, env_python='model-python',
        managed=True, repaired=True,
    ):
    """A `LocalBackend` whose worker launches, version checks and
    environment repairs are simulated and recorded in `events`.

    `versions` are the results of the successive `/version` checks: None
    (same protocol) or an exception to raise. Free ports are 50001, 50002...
    """
    import collections
    from unittest.mock import Mock
    from cellacdc.service import client as client_module, environment

    events = []
    free_ports = iter(range(50001, 50010))
    find_port = Mock(side_effect=lambda host='127.0.0.1': next(free_ports))
    monkeypatch.setattr(client_module, '_find_free_port', find_port)
    backend = LocalBackend('test_model', env_python=env_python, port=port)
    results = iter(versions)

    def launch(timeout):
        events.append(('launch', backend.port, backend.base_url))
        backend._output = collections.deque(['worker output line'])
        return [backend.env_python, '-m', 'cellacdc.service.worker']

    def check_version(base_url, timeout=10.0, session=None):
        events.append(('version', base_url))
        result = next(results)
        if isinstance(result, BaseException):
            raise result
        return result

    def repair(python, logger_func=print):
        events.append(('repair', python))
        return repaired

    monkeypatch.setattr(backend, '_launch', launch)
    monkeypatch.setattr(backend, 'stop', lambda: events.append(('stop',)))
    monkeypatch.setattr(client_module, 'check_worker_version', check_version)
    monkeypatch.setattr(environment, 'is_managed_model_env', lambda python: managed)
    monkeypatch.setattr(environment, 'repair_cellacdc_runtime', repair)
    return backend, events, find_port


def test_local_backend_starts_worker_of_same_protocol(monkeypatch):
    backend, events, _ = _version_checked_backend(monkeypatch, [None])
    backend._start_process(5)
    url = 'http://127.0.0.1:50001'
    assert events == [('launch', 50001, url), ('version', url)]
    assert backend.url == url


@pytest.mark.parametrize('explicit_port', [False, True])
def test_local_backend_repairs_environment_of_other_protocol(
        monkeypatch, capsys, explicit_port,
    ):
    """A worker with another protocol is stopped, Cell-ACDC is reinstalled
    in its (Cell-ACDC managed) environment, and the worker is started again
    (on a new port if the port was chosen automatically)."""
    from unittest.mock import Mock
    from cellacdc.service import client as client_module
    from cellacdc.service.client import WorkerVersionError

    backend, events, find_port = _version_checked_backend(
        monkeypatch, [WorkerVersionError('http://worker', 1), None],
        port=8600 if explicit_port else None,
    )
    first, second = (8600, 8600) if explicit_port else (50001, 50002)
    spec = ([], [], 'help')
    get_spec = Mock(return_value=spec)
    monkeypatch.setattr(client_module, 'get_model_spec', get_spec)

    assert backend.prepare(timeout=5) == spec

    first_url = f'http://127.0.0.1:{first}'
    second_url = f'http://127.0.0.1:{second}'
    assert events == [
        ('launch', first, first_url), ('version', first_url),
        ('stop',), ('repair', 'model-python'),
        ('launch', second, second_url), ('version', second_url),
    ]
    assert backend.port == second and backend.url == second_url
    get_spec.assert_called_once_with(second_url, timeout=5)
    assert find_port.call_count == (0 if explicit_port else 2)
    assert 'Reinstalling Cell-ACDC in model-python' in capsys.readouterr().out


@pytest.mark.parametrize('scenario', [
    'unmanaged_env', 'gui_python', 'repair_failed', 'still_mismatched',
    'version_error',
])
def test_local_backend_version_mismatch_failures(monkeypatch, scenario):
    from cellacdc.service.client import WorkerVersionError

    mismatch = WorkerVersionError('http://worker', 1)
    versions = {
        'still_mismatched': [mismatch, WorkerVersionError('http://worker', 3, '9.0')],
        'version_error': [RuntimeError('Model server error 500: broken')],
    }.get(scenario, [mismatch])
    backend, events, _ = _version_checked_backend(
        monkeypatch, versions,
        env_python=sys.executable if scenario == 'gui_python' else 'model-python',
        managed=scenario != 'unmanaged_env',
        repaired=scenario != 'repair_failed',
    )

    with pytest.raises(RuntimeError) as raised:
        backend._start_process(5)

    error = raised.value
    message = str(error)
    url = 'http://127.0.0.1:50001'
    started = [('launch', 50001, url), ('version', url), ('stop',)]
    if scenario in ('unmanaged_env', 'gui_python'):
        # Environments that Cell-ACDC did not create are never modified
        assert events == started
        assert 'repairs only the model environments that it created' in message
        assert str(mismatch) in message
        assert error.__cause__ is mismatch
    elif scenario == 'repair_failed':
        assert events == started + [('repair', 'model-python')]
        assert 'Reinstalling Cell-ACDC in model-python failed' in message
        assert error.__cause__ is mismatch
    elif scenario == 'still_mismatched':
        second_url = 'http://127.0.0.1:50002'
        assert events == started + [
            ('repair', 'model-python'),
            ('launch', 50002, second_url), ('version', second_url),
            ('stop',),
        ]
        assert 'did not fix it' in message
        assert 'Cell-ACDC 9.0 (protocol 3)' in message
        assert 'worker output line' in message
        assert isinstance(error.__cause__, WorkerVersionError)
    else:
        # Not a version mismatch: no repair
        assert events == started
        assert 'Reading the version of the test_model worker failed' in message
        assert 'Model server error 500: broken' in message
        assert 'worker output line' in message
        assert not isinstance(error, WorkerVersionError)


def test_find_python_interpreter_checks_all_python_aliases(monkeypatch, tmp_path):
    """`python`/`python3` on PATH are checked, not only `py` and `pythonX.Y`."""
    from cellacdc.service import environment as env

    wrong, right = tmp_path / "wrong", tmp_path / "right"
    wrong.mkdir(); right.mkdir()
    exe = ".exe" if sys.platform == "win32" else ""
    wrong_py, right_py = wrong / f"python{exe}", right / f"python3{exe}"
    wrong_py.write_bytes(b""); right_py.write_bytes(b"")
    for p in (wrong_py, right_py):
        p.chmod(0o755)
    monkeypatch.setenv("PATH", os.pathsep.join([str(wrong), str(right)]))
    monkeypatch.setenv("PYENV_ROOT", str(tmp_path / "no_pyenv"))
    monkeypatch.setattr(env.shutil, "which", lambda name: None)
    versions = {str(wrong_py): "1.8", str(right_py): "1.10"}
    monkeypatch.setattr(env, "_python_version_of", lambda p: versions.get(p))

    assert env.find_python_interpreter("1.10.4") == str(right_py)
    assert env.find_python_interpreter("1.7") is None
