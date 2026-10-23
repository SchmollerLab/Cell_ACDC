import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from cellacdc import _torch_utils, myutils
from cellacdc._torch_utils import torch_requirement, torch_version_specifier
from cellacdc.service import environment, prompts


def test_legacy_imports_reexport_torch_helpers():
    for name in (
        "check_install_torch", "get_driver_cuda_version", "get_min_compute_cap",
        "get_pytorch_command", "get_torch_device", "get_torch_options",
        "get_torch_options_safe", "resolve_torch_link",
    ):
        assert getattr(myutils, name) is getattr(_torch_utils, name)
    assert environment.ensure_torch is _torch_utils.ensure_torch
    assert environment._torch_import_check is _torch_utils._torch_import_check


@pytest.mark.parametrize("gpu, cuda, mps, expected", [
    (True, True, False, "cuda"),
    (False, True, False, "cpu"),
    (True, False, True, "mps"),
    (False, False, True, "mps"),
    (True, False, False, "cpu"),
])
def test_relocated_device_selection(monkeypatch, gpu, cuda, mps, expected):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps)),
        device=Mock(side_effect=lambda name: name),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert _torch_utils.get_torch_device(gpu=gpu) == expected
    torch.device.assert_called_once_with(expected)


def test_torch_utils_import_does_not_load_gui_or_torch():
    script = """
import sys
from importlib.abc import MetaPathFinder
import cellacdc

sys.modules.pop('cellacdc._torch_utils', None)

class BlockHeavyImports(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'qtpy'} or fullname in {
            'cellacdc.apps', 'cellacdc.widgets', 'cellacdc.myutils',
            'cellacdc.service.environment',
        }:
            raise AssertionError('Unexpected import: ' + fullname)

sys.meta_path.insert(0, BlockHeavyImports())
import cellacdc._torch_utils
"""
    subprocess.run(
        [sys.executable, "-c", script, "--server"],
        capture_output=True, text=True, check=True,
    )


@pytest.mark.parametrize("constraint, installed, expected", [
    ("2.5.1", "2.5.1", True),
    ("2.5.1", "2.5.1+cu124", True),
    ("2.5.1", "2.5.0", False),
    ("2.5.1", "2.6.0", False),
    ("<2", "1.13.1+cu117", True),
    ("<2", "2.0.0", False),
    (">=1.13,<2", "1.12.1", False),
    (">=1.13,<2", "1.13.1", True),
])
def test_torch_import_check_enforces_release(monkeypatch, constraint, installed, expected):
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "torchvision", ModuleType("torchvision"))
    monkeypatch.setattr(importlib.metadata, "version", lambda name: installed)
    script = environment._torch_import_check(constraint)
    if expected:
        exec(script, {})
    else:
        with pytest.raises(AssertionError, match="Expected torch"):
            exec(script, {})


@pytest.mark.parametrize("constraint, installed, expected", [
    ("2.5.1", "2.5.1+cu124", True),
    ("2.5.1", "2.6.0", False),
    ("<2", "1.13.1+cu117", True),
    ("<2", "2.0.0", False),
    (">=1.13,<2", "1.12.1", False),
])
def test_main_environment_detection_honors_torch_version(
    monkeypatch, constraint, installed, expected
):
    monkeypatch.setattr(environment, "load_model_env_config", lambda name: {
        "torch": True, "torch_version": constraint, "packages": ["model"],
    })
    monkeypatch.setattr(
        importlib.metadata, "version",
        lambda name: installed if name == "torch" else "1.0",
    )
    assert environment.model_installed_in_main_env("model") is expected


def test_model_dependencies_forward_torch_version(monkeypatch):
    monkeypatch.setattr(environment, "load_model_env_config", lambda name: {
        "torch": True, "torch_version": "2.5.1",
    })
    ensure = Mock(return_value=True)
    monkeypatch.setattr(environment, "ensure_torch", ensure)
    environment.ensure_model_dependencies("model", "model-python")
    assert ensure.call_args.kwargs["torch_version"] == "2.5.1"


def test_combined_environment_check_includes_torch_version(monkeypatch):
    monkeypatch.setattr(environment, "load_model_env_config", lambda name: {
        "torch": True, "torch_version": "2.5.1",
    })
    run = Mock(return_value=Mock(returncode=0, stdout="__ACDC_ENV_OK__\n"))
    monkeypatch.setattr(environment.subprocess, "run", run)
    assert environment._check_environment_dependencies("model", "model-python")
    script = run.call_args.args[0][2]
    assert repr(environment._torch_import_check("2.5.1")) in script
    compile(script, "<combined environment check>", "exec")


@pytest.mark.parametrize("verification_ok", [False, True])
@pytest.mark.parametrize("gui_command", [False, True])
@pytest.mark.parametrize("constraint", ["2.5.1", "<2", ">=1.13,<2"])
def test_pinned_install_command_and_verification(
    monkeypatch, verification_ok, gui_command, constraint
):
    run = Mock(side_effect=[
        Mock(returncode=1),
        Mock(returncode=0 if verification_ok else 1, stderr="wrong version", stdout=""),
    ])
    monkeypatch.setattr(environment.subprocess, "run", run)
    options = Mock(return_value=([("CPU", "cpu-index"), ("CUDA", "cuda-index")], 1))
    monkeypatch.setattr(_torch_utils, "get_torch_options_safe", options)
    install = Mock()
    monkeypatch.setattr(environment, "run_streaming", install)
    logger = Mock()
    requirement = torch_requirement(constraint)
    command = f'"model-python" -m pip install "{requirement}" torchvision'
    prompt = Mock(side_effect=lambda request: command if gui_command else request.default)

    assert environment.ensure_torch(
        "model-python", torch_version=constraint, found_cuda_version=(12, 4),
        prompt_func=prompt, logger_func=logger,
    ) is verification_ok

    options.assert_called_once_with(
        driver_cuda_version=(12, 4), torch_version=constraint, exec_path="model-python"
    )
    assert prompt.call_args.args[0].extra["torch_version"] == constraint
    if gui_command:
        expected = command if environment.os.name == "nt" else [
            "model-python", "-m", "pip", "install", requirement, "torchvision"
        ]
    else:
        expected = [
            "model-python", "-m", "pip", "install", requirement, "torchvision",
            "--index-url", "cuda-index",
        ]
    install.assert_called_once_with(expected, logger, check=True)
    assert run.call_count == 2
    assert run.call_args_list[0].args[0][2] == run.call_args_list[1].args[0][2]
    if not verification_ok:
        assert "verification failed" in logger.call_args.args[0]


def test_satisfied_pin_does_not_discover_or_install(monkeypatch):
    monkeypatch.setattr(
        environment.subprocess, "run", Mock(return_value=Mock(returncode=0))
    )
    options = Mock(side_effect=AssertionError("Must remain lazy"))
    monkeypatch.setattr(_torch_utils, "get_torch_options_safe", options)
    install = Mock()
    monkeypatch.setattr(environment, "run_streaming", install)
    assert environment.ensure_torch("model-python", torch_version="2.5.1")
    options.assert_not_called()
    install.assert_not_called()


@pytest.mark.parametrize("torch_version, expected_tags", [
    (None, ["cpu", "cu126"]),
    ("2.5.1", ["cpu", "cu124"]),
    ("<2", ["cpu", "cu117"]),
    (">=1.13,<2", ["cpu", "cu117"]),
])
def test_cuda_options_filter_pinned_release(monkeypatch, torch_version, expected_tags):
    monkeypatch.setattr(myutils.sys, "platform", "win32")
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    pages = {
        _torch_utils.BASE_PYTORCH_LINK: "cu117 cu124 cu126",
        f"{_torch_utils.BASE_PYTORCH_LINK}cu117/torch/":
            f"torch-1.13.1%2Bcu117-{py}-{py}-win_amd64.whl",
        f"{_torch_utils.BASE_PYTORCH_LINK}cu124/torch/":
            f"torch-2.5.1%2Bcu124-{py}-{py}-win_amd64.whl",
        f"{_torch_utils.BASE_PYTORCH_LINK}cu126/torch/":
            f"torch-2.6.0%2Bcu126-{py}-{py}-win_amd64.whl",
    }
    monkeypatch.setattr(_torch_utils, "_get", pages.__getitem__)
    options, preferred = _torch_utils.get_torch_options(
        driver_cuda_version=(13, 0), torch_version=torch_version
    )
    assert [url.rsplit("/", 1)[-1] for _, url in options] == expected_tags
    assert preferred == 1


@pytest.mark.parametrize("target_version, target_platform, machine, wheel_platform", [
    ([3, 10], "win32", "AMD64", "win_amd64"),
    ([3, 11], "win32", "AMD64", "win_amd64"),
    ([3, 10], "linux", "x86_64", "linux_x86_64"),
])
def test_cuda_options_use_target_interpreter(
    monkeypatch, target_version, target_platform, machine, wheel_platform
):
    monkeypatch.setattr(myutils.sys, "version_info", (3, 14))
    monkeypatch.setattr(myutils.sys, "platform", "win32")
    run = Mock(return_value=Mock(
        stdout=json.dumps([target_version, target_platform, machine])
    ))
    monkeypatch.setattr(myutils.subprocess, "run", run)
    pages = {
        _torch_utils.BASE_PYTORCH_LINK: "cu117",
        f"{_torch_utils.BASE_PYTORCH_LINK}cu117/torch/":
            f"torch-1.13.1%2Bcu117-cp310-cp310-{wheel_platform}.whl",
    }
    monkeypatch.setattr(_torch_utils, "_get", pages.__getitem__)

    options, preferred = _torch_utils.get_torch_options(
        driver_cuda_version=(12, 4), torch_version="<2", exec_path="model-python"
    )

    expected_tags = ["cpu", "cu117"] if target_version == [3, 10] else ["cpu"]
    assert [url.rsplit("/", 1)[-1] for _, url in options] == expected_tags
    assert preferred == len(expected_tags) - 1
    run.assert_called_once()
    assert run.call_args.args[0][0] == "model-python"
    assert run.call_args.kwargs["check"] is True
    assert run.call_args.kwargs["timeout"] == 20


def test_target_interpreter_probe_script(monkeypatch):
    original_run = subprocess.run
    results = []

    def run_probe(*args, **kwargs):
        result = original_run(*args, **kwargs)
        results.append(result)
        return result

    run = Mock(side_effect=run_probe)
    monkeypatch.setattr(myutils.subprocess, "run", run)
    monkeypatch.setattr(_torch_utils, "_get", lambda url: "")
    _torch_utils.get_torch_options(
        driver_cuda_version=(12, 4), exec_path=Path(sys.executable)
    )
    run.assert_called_once()
    version, target_platform, machine = json.loads(results[0].stdout)
    assert version == list(sys.version_info[:2])
    assert target_platform == sys.platform
    assert machine


@pytest.mark.parametrize("error", [
    FileNotFoundError("missing model Python"),
    subprocess.CalledProcessError(1, ["model-python"]),
    subprocess.TimeoutExpired(["model-python"], 20),
    json.JSONDecodeError("invalid probe output", "", 0),
])
def test_target_interpreter_failure_is_not_hidden(monkeypatch, error):
    monkeypatch.setattr(_torch_utils, "get_min_compute_cap", lambda: None)
    monkeypatch.setattr(myutils.subprocess, "run", Mock(side_effect=error))
    with pytest.raises(type(error)):
        _torch_utils.get_torch_options_safe(
            driver_cuda_version=(12, 4), exec_path="model-python"
        )


@pytest.mark.parametrize("constraint, requirement", [
    ("2.5.1", "torch==2.5.1"),
    ("<2", '"torch<2"'),
    (">=1.13,<2", '"torch>=1.13,<2"'),
])
def test_pytorch_command_pins_torch_only(monkeypatch, constraint, requirement):
    monkeypatch.setattr(
        _torch_utils, "_safe_torch_options",
        Mock(return_value=([("CPU", "cpu-index")], 0)),
    )
    monkeypatch.setattr(
        myutils, "get_pip_conda_prefix", lambda **kwargs: ("", "python -m pip install")
    )
    result = _torch_utils.get_pytorch_command(torch_version=constraint)
    commands = next(iter(result.values()))["Pip"]
    assert commands["CPU"] == (
        f"python -m pip install {requirement} torchvision --index-url cpu-index"
    )


@pytest.mark.parametrize("torch_version, torch_spec", [
    (None, "torch"),
    ("2.5.1", "torch==2.5.1"),
    ("<2", '"torch<2"'),
    (">=1.13,<2", '"torch>=1.13,<2"'),
])
def test_install_dialog_preserves_pin_across_platforms(
    monkeypatch, torch_version, torch_spec
):
    from qtpy.QtWidgets import QApplication, QLabel
    from cellacdc import apps

    app = QApplication.instance() or QApplication([])
    options = Mock(return_value=([("CPU", "cpu-index"), ("CUDA", "cuda-index")], 1))
    monkeypatch.setattr(_torch_utils, "get_torch_options_safe", options)
    monkeypatch.setattr(
        myutils, "get_pip_conda_prefix", lambda **kwargs: ("", "python -m pip install")
    )
    dialog = apps.InstallPyTorchDialog(
        target_python="model-python", platform="win32", found_cuda_version=False,
        torch_version=torch_version,
    )
    try:
        assert dialog.commandWidget.command() == (
            f"python -m pip install {torch_spec} torchvision --index-url cuda-index"
        )
        if torch_version:
            assert torch_version in [label.text() for label in dialog.findChildren(QLabel)]
        dialog.cmptPlatformCombobox.setCurrentIndex(0)
        assert dialog.commandWidget.command().endswith("--index-url cpu-index")
        dialog.osCombobox.setCurrentText("Mac")
        assert dialog.commandWidget.command() == (
            f"python -m pip install {torch_spec} torchvision"
        )
        dialog.ok_cb()
        assert not dialog.cancel
        assert dialog.command == dialog.commandWidget.command()
        assert options.call_args.kwargs.get("torch_version") == torch_version
        assert options.call_args.kwargs["exec_path"] == "model-python"
    finally:
        dialog.close()
        dialog.deleteLater()
        app.processEvents()


@pytest.mark.parametrize("value, expected", [
    (None, ""),
    ("", ""),
    ("2.5.1", "==2.5.1"),
    (" <2 ", "<2"),
    (">=1.13,<2", ">=1.13,<2"),
    ("==2.5.*", "==2.5.*"),
    ("~=2.5", "~=2.5"),
    ("!=2.5.0", "!=2.5.0"),
])
def test_torch_version_specifier(value, expected):
    assert torch_version_specifier(value) == expected


@pytest.mark.parametrize("value", ["not-a-version", "<", "torch<2", " "])
def test_invalid_torch_constraint_is_rejected(value):
    from packaging.specifiers import InvalidSpecifier

    with pytest.raises(InvalidSpecifier):
        torch_version_specifier(value)


def test_qt_prompt_forwards_pin(monkeypatch):
    from cellacdc import apps

    dialog = Mock(cancel=False, command="pinned install")
    constructor = Mock(return_value=dialog)
    monkeypatch.setattr(apps, "InstallPyTorchDialog", constructor)
    handler = object.__new__(prompts.QtHandler)
    handler.parent = None
    request = prompts.choice(
        title="Install PyTorch", text="", choices=["CPU"],
        dialog="pytorch", extra={"torch_version": "2.5.1"},
    )
    assert handler._ask_pytorch(request) == "pinned install"
    assert constructor.call_args.kwargs["torch_version"] == "2.5.1"
