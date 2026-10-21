from unittest.mock import Mock
import sys

import pytest

from cellacdc import config, myutils
from cellacdc.service import environment
from cellacdc.service.client import LocalBackend
from cellacdc import _model_startup


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("needs_torch", [False, True])
def test_environment_only_discovers_torch_options_when_installing(
    monkeypatch, tmp_path, existing, needs_torch
):
    python = tmp_path / "python.exe"
    python.write_text("")
    python = str(python)
    monkeypatch.setattr(
        environment, "get_model_python_exec",
        Mock(return_value=python if existing else None),
    )
    monkeypatch.setattr(environment, "create_venv", Mock())
    monkeypatch.setattr(environment, "get_python_from_env_dir", Mock(return_value=python))
    monkeypatch.setattr(environment, "link_cellacdc_to_env", Mock(return_value=True))
    monkeypatch.setattr(environment, "ensure_worker_dependencies", Mock())
    monkeypatch.setattr(
        environment, "_check_environment_dependencies", Mock(return_value=False)
    )
    monkeypatch.setattr(
        environment, "load_model_env_config", lambda name: {"torch": needs_torch}
    )
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(environment.subprocess, "run", run)
    driver = Mock(side_effect=AssertionError("No CUDA detection needed"))
    options = Mock(side_effect=AssertionError("No network discovery needed"))
    monkeypatch.setattr(myutils, "get_driver_cuda_version", driver)
    monkeypatch.setattr(myutils, "get_torch_options_safe", options)

    result = environment.get_or_create_model_env(
        "test", install_method="venv", env_root=str(tmp_path)
    )

    assert result == python
    assert run.call_count == int(needs_torch)
    driver.assert_not_called()
    options.assert_not_called()


def test_missing_torch_still_discovers_and_installs_preferred_build(monkeypatch):
    monkeypatch.setattr(
        environment.subprocess, "run", Mock(return_value=Mock(returncode=1))
    )
    driver = Mock(return_value=(12, 6))
    links = ([("CPU", "cpu-index"), ("CUDA", "cuda-index")], 1)
    options = Mock(return_value=links)
    install = Mock(return_value=0)
    monkeypatch.setattr(myutils, "get_driver_cuda_version", driver)
    monkeypatch.setattr(myutils, "get_torch_options_safe", options)
    monkeypatch.setattr(environment, "run_streaming", install)
    logger = Mock()
    prompt = Mock(side_effect=lambda request: request.default)

    assert environment.ensure_torch(
        "model-python", logger_func=logger, prompt_func=prompt
    )

    driver.assert_called_once_with()
    options.assert_called_once_with(driver_cuda_version=(12, 6))
    assert prompt.call_args.args[0].extra["found_cuda_version"] == (12, 6)
    install.assert_called_once_with(
        ["model-python", "-m", "pip", "install", "torch", "torchvision",
         "--index-url", "cuda-index"],
        logger, check=True,
    )


def test_external_check_install_torch_defers_discovery(monkeypatch):
    ensure = Mock(return_value=True)
    monkeypatch.setattr(environment, "ensure_torch", ensure)
    monkeypatch.setattr(
        myutils, "get_driver_cuda_version",
        Mock(side_effect=AssertionError("Must be lazy")),
    )
    monkeypatch.setattr(
        myutils, "get_torch_options_safe",
        Mock(side_effect=AssertionError("Must be lazy")),
    )
    monkeypatch.setattr(myutils, "is_server_mode", lambda: False)

    myutils.check_install_torch(is_cli=True, python_exec="model-python")

    assert ensure.call_args.args == ("model-python",)
    assert ensure.call_args.kwargs["found_cuda_version"] is None
    assert ensure.call_args.kwargs["possible_links"] is None


def test_startup_timing_is_disabled_outside_debug_mode(monkeypatch):
    monkeypatch.setitem(config.parser_args, "debug", False)
    monkeypatch.setenv("CELLACDC_MODEL_STARTUP_TIMING", "1")
    clock = Mock(side_effect=AssertionError("Disabled timer must not read the clock"))
    monkeypatch.setattr(_model_startup, "perf_counter", clock)
    logger = Mock()

    with _model_startup.startup_timer("test phase", logger):
        pass

    logger.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_startup_timing_reports_elapsed_and_preserves_errors(monkeypatch, fails):
    monkeypatch.setitem(config.parser_args, "debug", True)
    monkeypatch.delenv("CELLACDC_MODEL_STARTUP_TIMING", raising=False)
    monkeypatch.setattr(_model_startup, "perf_counter", Mock(side_effect=[10.0, 12.345]))
    logger = Mock()
    error = RuntimeError("original failure")

    def execute():
        with _model_startup.startup_timer("test phase", logger):
            if fails:
                raise error

    if fails:
        with pytest.raises(RuntimeError) as raised:
            execute()
        assert raised.value is error
    else:
        execute()

    status = "failed" if fails else "completed"
    logger.assert_called_once_with(
        f"[model startup] test phase: 2.345s ({status})"
    )


@pytest.mark.parametrize("healthy", [False, True])
def test_existing_environment_combined_check_preserves_repair_and_setup(
    monkeypatch, tmp_path, healthy
):
    python = tmp_path / "python.exe"
    python.write_text("")
    monkeypatch.setattr(
        environment, "get_model_python_exec", Mock(return_value=str(python))
    )
    monkeypatch.setattr(environment, "link_cellacdc_to_env", Mock())
    probe = Mock(return_value=healthy)
    worker = Mock()
    model = Mock()
    setup = Mock()
    config = {"setup_steps": [{"name": "custom step"}]}
    monkeypatch.setattr(environment, "load_model_env_config", lambda name: config)
    monkeypatch.setattr(environment, "_check_environment_dependencies", probe)
    monkeypatch.setattr(environment, "ensure_worker_dependencies", worker)
    monkeypatch.setattr(environment, "ensure_model_dependencies", model)
    monkeypatch.setattr(environment, "_ensure_model_setup_steps", setup)
    prompt = Mock()
    logger = Mock()

    assert environment.get_or_create_model_env(
        "test", logger_func=logger, auto_install=False, prompt_func=prompt
    ) == str(python)

    probe.assert_called_once_with("test", str(python), logger)
    assert worker.call_count == int(not healthy)
    assert model.call_count == int(not healthy)
    if healthy:
        setup.assert_called_once_with(
            "test", str(python), config, logger, False, prompt
        )
    else:
        setup.assert_not_called()


@pytest.mark.parametrize("packages, healthy", [
    (["numpy"], True),
    ([{"pkg_name": "numpy", "min_version": "999"}], False),
    (["acdc_nonexistent_test_package"], False),
    ([{"pkg_name": "different-distribution", "import_pkg_name": "numpy",
       "min_version": "1", "include_lower_version": True}], True),
])
def test_combined_check_uses_one_subprocess_and_checks_versions(
    monkeypatch, packages, healthy
):
    monkeypatch.setitem(config.parser_args, "debug", True)
    monkeypatch.setattr(
        environment, "load_model_env_config", lambda name: {"packages": packages}
    )
    run = Mock(wraps=environment.subprocess.run)
    monkeypatch.setattr(environment.subprocess, "run", run)
    logger = Mock()

    assert environment._check_environment_dependencies(
        "test", sys.executable, logger
    ) is healthy

    run.assert_called_once()
    output = "\n".join(call.args[0] for call in logger.call_args_list)
    assert "Cell-ACDC runtime imports:" in output
    assert "combined environment checks:" in output
    assert ("checking dependencies individually for repair" in output) is not healthy


@pytest.mark.parametrize("installed, spec, satisfied", [
    ("2.0", {"exact_version": "2.0"}, True),
    ("2.0", {"exact_version": "2.1"}, False),
    ("2.0", {"min_version": "2.0"}, False),
    ("2.0", {"min_version": "2.0", "include_lower_version": True}, True),
    ("2.0", {"max_version": "2.0"}, False),
    ("2.0", {"max_version": "2.0", "include_higher_version": True}, True),
    ("", {"min_version": "2.0"}, True),
])
def test_shared_version_constraints(installed, spec, satisfied):
    full_spec = dict(
        pkg_name="test", min_version="", max_version="", exact_version="",
        include_lower_version=False, include_higher_version=False,
    )
    full_spec.update(spec)
    if satisfied:
        myutils._check_package_version_constraints(full_spec, installed)
    else:
        with pytest.raises(ModuleNotFoundError):
            myutils._check_package_version_constraints(full_spec, installed)


@pytest.mark.parametrize("debug", [False, True])
def test_server_startup_timings_follow_debug_mode(monkeypatch, capsys, debug):
    monkeypatch.setitem(config.parser_args, "debug", debug)
    backend = LocalBackend("thresholding")
    try:
        client = backend.start()
        assert client.model_name == "thresholding"
        assert ("--debug" in backend.proc.args) is debug
        output = capsys.readouterr().out
        for stage in (
            "worker process launch", "waiting for worker /health",
            "deferred initialization", "client /info connection",
            "worker /spec request", "model initialization payload encoding",
            "worker /init request", "worker runtime imports",
            "worker model module import", "worker model argument inspection",
            "worker model construction",
        ):
            assert (stage in output) is debug
    finally:
        backend.stop()


def test_prepared_worker_is_used_for_specs_and_inference(monkeypatch):
    import numpy as np
    import requests
    from cellacdc.service import model_execution

    monkeypatch.setattr(
        model_execution, "model_installed_in_main_env", lambda name: False
    )
    monkeypatch.setattr(
        model_execution, "get_or_create_model_env", lambda name: sys.executable
    )
    monkeypatch.setattr(
        myutils, "getModelArgSpecFromEnv",
        Mock(side_effect=AssertionError("Must not spawn a spec interpreter")),
    )
    execution = model_execution.ModelExecution("thresholding")
    try:
        spec = execution.prepare()
        backend = execution._prepared_backend
        proc = backend.proc
        assert proc.poll() is None
        assert backend.client is None
        assert requests.get(f"{backend.url}/info", timeout=10).status_code == 503
        assert "gauss_sigma" in [param.name for param in spec[1]]
        assert execution.prepare() is execution.spec
        model = execution.init_model({})
        assert execution.backend is backend
        assert backend.proc is proc
        assert execution._prepared_backend is None
        image = np.zeros((32, 32), dtype=np.uint8)
        image[10:20, 10:20] = 200
        labels = model.segment(image)
        assert labels.shape == image.shape
        assert labels.max() > 0
        execution.release(model)
        assert execution.init_model({}) is model
        assert backend.proc is proc
    finally:
        execution.stop()
    assert proc.poll() is not None
