import io
import os
import sys
import numpy as np
import pytest
import requests

from cellacdc.service.schema import HealthResponse, InfoResponse, Capabilities, JobStatusResponse
from cellacdc.service.client import LocalBackend, ModelClient
from cellacdc import cli


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
        mocks["getModelArgSpecFromEnv"].assert_called_once_with(
            "test_model", "model-python"
        )
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
    mocks["getModelArgSpecFromEnv"].assert_called_once()

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
    mocks["LocalBackend"].assert_not_called()


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
        encoded = myutils._serialize_model_param_type(original)
        restored = myutils._deserialize_model_param_type(json.loads(json.dumps(encoded)))
        assert restored is original

    class Choices:
        values = ['cyto3', 'nuclei']
        is_exclusive_with = ['model_path']
        default_exclusive = 'Using custom model'

    encoded = myutils._serialize_model_param_type(Choices)
    restored = myutils._deserialize_model_param_type(json.loads(json.dumps(encoded)))
    assert restored().values == Choices.values
    assert restored().is_exclusive_with == Choices.is_exclusive_with
    assert restored().default_exclusive == Choices.default_exclusive
    assert restored.cast_dtype('cyto3') == 'cyto3'
    optional = myutils._serialize_model_param_type(typing.Optional[_types.SecondChannelImage])
    assert _types.is_second_channel_type(myutils._deserialize_model_param_type(optional))
    literal = myutils._serialize_model_param_type(typing.Literal['cpu', 'gpu'])
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
            myutils._serialize_model_param_type(param_type)
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


def test_server_mode_skips_installs_and_torch_prompt(monkeypatch):
    from cellacdc import myutils
    from cellacdc.service.environment import ensure_torch, get_torch_choices

    import cellacdc

    monkeypatch.setattr(cellacdc, "RUNNING_AS_SERVER", True)
    assert myutils.is_server_mode()
    with pytest.raises(ModuleNotFoundError, match="_env_config.py"):
        myutils.check_install_package("definitely_not_a_pkg_xyz")

    # The client asks on behalf of the worker; cancelling installs nothing
    seen = {}

    def prompt(p):
        seen.update(choices=p.choices, default=p.default)
        return None

    monkeypatch.setattr(cellacdc, "RUNNING_AS_SERVER", False)
    fake_python = sys.executable
    import subprocess
    has_torch = subprocess.run(
        [fake_python, "-c", "import torch, torchvision"], capture_output=True
    ).returncode == 0
    if not has_torch:
        assert ensure_torch(fake_python, prompt_func=prompt) is False
        assert seen["default"] in seen["choices"]
    assert "CPU" in get_torch_choices()


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


def test_cellacdc_cli_check_and_model_deps():
    from cellacdc.service.environment import (
        is_cellacdc_cli_installed,
        ensure_model_dependencies,
        load_model_env_config,
        detect_torch_index,
    )

    # In the current test environment (where cellacdc is installed), it should return True
    assert is_cellacdc_cli_installed(sys.executable) is True

    # Config is read from models/_env_config.py without importing the model
    assert load_model_env_config("StarDist")["python"] == "3.11"
    assert load_model_env_config("stardist")["python"] == "3.11"
    assert load_model_env_config("cellpose_v3")["torch"] is True
    assert load_model_env_config("nonexistent_model_12345") == {}
    assert "cellacdc.models.StarDist" not in sys.modules

    index = detect_torch_index()
    assert index is None or index.startswith("https://download.pytorch.org/whl/")

    # Calling ensure_model_dependencies for unknown model should safely no-op
    ensure_model_dependencies("nonexistent_model_12345", sys.executable)

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


def test_init_payload_roundtrip():
    import pandas as pd
    from cellacdc.service import payload

    segm = np.arange(12, dtype=np.uint32).reshape(3, 4)
    df = pd.DataFrame({'x': [1, 2], 'y': [3, 4], 'id': [1, 0]})
    kwargs = {'segm_data': segm, 'input_points_df': df, 'gpu': False}
    assert payload.needs_payload(kwargs)
    assert not payload.needs_payload({'gpu': False})

    kwargs_json, data = payload.encode(kwargs)
    decoded = payload.decode(kwargs_json, data)
    np.testing.assert_array_equal(decoded['segm_data'], segm)
    pd.testing.assert_frame_equal(decoded['input_points_df'], df)
    assert decoded['gpu'] is False


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


def _jobs_app(monkeypatch, segment, **extra_methods):
    import types
    from cellacdc import myutils
    from cellacdc.service import worker

    class Model:
        def __init__(self):
            pass

        def segment(self, image):
            return segment(image)

    for name, func in extra_methods.items():
        setattr(Model, name, lambda self, *args, _f=func: _f(*args))

    module = types.SimpleNamespace(Model=Model)
    monkeypatch.setattr(myutils, 'import_segment_module', lambda name: module)
    executor = _CapturingExecutor()
    monkeypatch.setattr(worker, '_executor', executor)
    monkeypatch.setattr(worker, '_jobs', {})
    app = worker.create_app('fake')
    endpoints = {
        (route.path, tuple(sorted(route.methods))): route.endpoint
        for route in app.routes if hasattr(route, 'methods')
    }
    return endpoints, executor


def test_worker_rejects_all_nonempty_path_typed_init_kwargs(monkeypatch):
    import types
    from cellacdc import myutils
    from cellacdc.service import worker

    class Model:
        def __init__(self, model_path: os.PathLike = ''):
            self.model_path = model_path

        def segment(self, image):
            return image

    module = types.SimpleNamespace(Model=Model)
    monkeypatch.setattr(myutils, 'model_init_requires_segm_data', lambda _: False)

    with pytest.raises(ValueError, match='model_path'):
        worker.instantiate_model(module, {'model_path': '/client/model.pth'})
    assert worker.instantiate_model(module, {}).model_path == ''


def _submit_job(endpoints, arr):
    import asyncio
    submit = endpoints[('/jobs/segment', ('POST',))]
    return asyncio.run(submit(image=_FakeUpload(arr), params='{}'))['job_id']


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
            image=_FakeUpload(np.zeros((2, 2))), params='{}'
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
    from cellacdc import myutils
    from cellacdc.service import client as client_module
    from cellacdc.service import worker

    class Model:
        def __init__(self, diameter: int = 30):
            pass

        def segment(self, image, gauss_sigma: float = 0.5):
            return image

    module = types.SimpleNamespace(Model=Model)
    monkeypatch.setattr(myutils, 'import_segment_module', lambda _: module)
    app = worker.create_app('fake')
    endpoints = {
        route.path: route.endpoint for route in app.routes
        if hasattr(route, 'methods')
    }
    spec_data = endpoints['/spec']()

    class Response:
        ok = True

        def json(self):
            return spec_data

    monkeypatch.setattr(
        client_module.requests, 'get', lambda *args, **kwargs: Response()
    )
    init_params, segment_params = client_module.get_model_arg_spec(
        'http://worker'
    )

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
