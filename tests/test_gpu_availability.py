import json
import logging
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cellacdc import myutils


@pytest.mark.parametrize("return_types", [False, True])
def test_cpu_does_not_probe_environment(monkeypatch, return_types):
    probe = Mock(side_effect=AssertionError("CPU must not probe GPU"))
    monkeypatch.setattr(myutils, "_available_gpu_frameworks_from_env", probe)
    result = myutils.check_gpu_available(
        "cellpose_v3", False, python_exec="model-python",
        return_available_gpu_type=return_types,
    )
    assert result == ((True, []) if return_types else True)


@pytest.mark.parametrize(
    "available,expected",
    [(["cuda"], (True, ["cuda"])), ([], (False, [])),
     (["directML"], (True, ["directML"]))],
)
def test_gpu_uses_selected_environment(monkeypatch, available, expected):
    probe = Mock(return_value=available)
    monkeypatch.setattr(myutils, "_available_gpu_frameworks_from_env", probe)
    monkeypatch.setattr(
        myutils, "_available_gpu_frameworks",
        Mock(side_effect=AssertionError("Must not inspect main environment")),
    )
    assert myutils.check_gpu_available(
        "cellpose_v3", True, do_not_warn=True,
        return_available_gpu_type=True, python_exec="model-python",
    ) == expected
    probe.assert_called_once_with("model-python", True, True)


@pytest.mark.parametrize("python_exec", [None, sys.executable])
def test_in_process_gpu_check(monkeypatch, python_exec):
    probe = Mock(return_value=["cuda"])
    monkeypatch.setattr(myutils, "_available_gpu_frameworks", probe)
    assert myutils.check_gpu_available(
        "cellpose_v3", True, python_exec=python_exec, do_not_warn=True
    )
    probe.assert_called_once_with(True, True)


@pytest.mark.parametrize("count,available", [(0, False), (1, False), (1, True)])
def test_cuda_requires_available_device(monkeypatch, count, available):
    torch = SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: available, device_count=lambda: count,
    ))
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(myutils, "is_mac_arm64", False)
    assert myutils._available_gpu_frameworks(True, False) == (
        ["cuda"] if count and available else []
    )


def test_missing_torch_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert myutils._available_gpu_frameworks(True, False) == []


def test_mps_requires_available_backend(monkeypatch):
    available = Mock(return_value=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(mps=SimpleNamespace(is_available=available)),
    ))
    monkeypatch.setattr(myutils, "is_mac_arm64", True)
    assert myutils._available_gpu_frameworks(True, False) == []
    available.return_value = True
    assert myutils._available_gpu_frameworks(True, False) == ["mps"]


@pytest.mark.parametrize("directml_available", [False, True])
def test_directml_detection(monkeypatch, directml_available):
    monkeypatch.setattr(myutils, "is_win", True)
    monkeypatch.setitem(sys.modules, "torch_directml", SimpleNamespace(
        is_available=lambda: directml_available,
    ))
    assert myutils._available_gpu_frameworks(False, True) == (
        ["directML"] if directml_available else []
    )


@pytest.mark.parametrize("backend", ["cuda", "directML"])
def test_explicit_backend_is_required(monkeypatch, backend):
    monkeypatch.setattr(myutils, "is_win", True)
    monkeypatch.setattr(
        myutils, "_available_gpu_frameworks",
        lambda *args: ["directML" if backend == "cuda" else "cuda"],
    )
    assert myutils.check_gpu_available(
        "cellpose_v3", True, cuda=backend == "cuda",
        directML=backend == "directML", do_not_warn=True,
    ) is False


def test_environment_probe_parses_marked_output(monkeypatch):
    run = Mock(return_value=subprocess.CompletedProcess(
        [], 0, "library output\n__ACDC_GPU__" + json.dumps(["cuda"]) + "\n", "",
    ))
    monkeypatch.setattr(myutils.subprocess, "run", run)
    assert myutils._available_gpu_frameworks_from_env(
        "model-python", True, False
    ) == ["cuda"]
    command = run.call_args.args[0]
    assert command[0] == "model-python"
    assert command[-1] == "--server"
    assert "_available_gpu_frameworks(True, False)" in command[2]


@pytest.mark.parametrize(
    "returncode,stdout,stderr,match",
    [(1, "", "broken torch", "broken torch"),
     (0, "unmarked output", "", "No GPU availability")],
)
def test_environment_probe_errors_are_explicit(
    monkeypatch, returncode, stdout, stderr, match,
):
    monkeypatch.setattr(myutils.subprocess, "run", Mock(
        return_value=subprocess.CompletedProcess([], returncode, stdout, stderr)
    ))
    with pytest.raises(RuntimeError, match=match):
        myutils._available_gpu_frameworks_from_env("model-python", True, False)


def test_gpu_warning_targets_model_environment(monkeypatch):
    monkeypatch.setattr(
        myutils, "_available_gpu_frameworks_from_env", lambda *a: [],
    )
    monkeypatch.setattr(myutils, "is_server_mode", lambda: False)
    warn = Mock(return_value=(True, True))
    monkeypatch.setattr(myutils, "_warn_install_gpu", warn)
    parent = object()
    assert myutils.check_gpu_available(
        "cellpose_v3", True, qparent=parent, python_exec="model-python",
        return_available_gpu_type=True,
    ) == (True, ["directML"])
    assert warn.call_args.kwargs == {
        "qparent": parent, "python_exec": "model-python",
    }


def test_worker_gpu_check_never_opens_dialog(monkeypatch, caplog):
    monkeypatch.setattr(myutils, "_available_gpu_frameworks", lambda *a: [])
    monkeypatch.setattr(myutils, "is_server_mode", lambda: True)
    monkeypatch.setattr(myutils, "_warn_install_gpu", Mock(
        side_effect=AssertionError("Workers must not open Qt dialogs")
    ))
    assert myutils.check_gpu_available(
        "cellpose_v3", True, return_available_gpu_type=True,
    ) == (True, [])
    assert "GPU unavailable" in caplog.text


@pytest.mark.parametrize("ask_cuda", [False, True])
@pytest.mark.parametrize("version,supported", [("3.12", True), ("3.13", False)])
def test_directml_install_uses_target_python_and_version(
    monkeypatch, version, supported, ask_cuda,
):
    stop, directml, proceed = object(), object(), object()
    message = SimpleNamespace(
        cancel=False, clickedButton=directml, warning=Mock(),
    )
    monkeypatch.setattr(myutils, "widgets", SimpleNamespace(
        myMessageBox=lambda **kwargs: message,
        cancelPushButton=lambda text: stop,
        okPushButton=Mock(side_effect=[directml, proceed]),
    ), raising=False)
    monkeypatch.setattr(myutils, "is_win", True)
    run = Mock(return_value=subprocess.CompletedProcess(
        [], 0, version + "\n", "",
    ))
    monkeypatch.setattr(myutils.subprocess, "run", run)
    install = Mock(return_value=True)
    monkeypatch.setattr(myutils, "check_install_package", install)
    monkeypatch.setattr(myutils, "purge_module", Mock(
        side_effect=AssertionError("Do not purge main-environment torch")
    ))
    python = r"C:\model env\python.exe"
    assert myutils._warn_install_gpu(
        "cellpose_v3", {"cuda", "directML"} if ask_cuda else {"directML"},
        python_exec=python,
    ) == ((True, True) if supported else (False, False))
    assert run.call_args.args[0][0] == python
    text = message.warning.call_args_list[0].args[2]
    assert python in text
    commands = message.warning.call_args_list[0].kwargs["commands"]
    if ask_cuda:
        assert commands == (
            f'"{python}" -m pip uninstall torch torchvision torchaudio',
            f'"{python}" -m pip install torch torchvision torchaudio '
            '--index-url https://download.pytorch.org/whl/cu128',
        )
        assert all("\n" not in command for command in commands)
    else:
        assert commands is None
    if supported:
        assert install.call_args.kwargs["python_exec"] == python
    else:
        install.assert_not_called()


def test_lost_ids_gpu_cache_is_environment_specific(monkeypatch):
    pytest.importorskip("qtpy")
    from cellacdc.gui import guiWin

    execution = SimpleNamespace(env_python="model-python")
    window = SimpleNamespace(
        SegForLostIDsWorker=SimpleNamespace(),
        getModelExecution=Mock(return_value=execution),
        _ackSegForLostIDsWorker=Mock(),
        logger=Mock(),
    )
    check = Mock(return_value=True)
    monkeypatch.setattr(myutils, "check_gpu_available", check)
    for _ in range(2):
        guiWin.SegForLostIDsWorkerAskInstallGPU(window, "cellpose_v3", True)
    assert check.call_count == 2
    assert all(
        call.kwargs["python_exec"] == "model-python"
        for call in check.call_args_list
    )
    execution.env_python = None
    guiWin.SegForLostIDsWorkerAskInstallGPU(window, "cellpose_v3", True)
    assert check.call_count == 4
    assert check.call_args.kwargs["python_exec"] is None
    assert window._ackSegForLostIDsWorker.call_count == 3


def test_lost_ids_gpu_failure_releases_worker(monkeypatch, caplog):
    pytest.importorskip("qtpy")
    from cellacdc.gui import guiWin

    window = SimpleNamespace(
        SegForLostIDsWorker=SimpleNamespace(),
        getModelExecution=Mock(return_value=SimpleNamespace(
            env_python="model-python",
        )),
        _ackSegForLostIDsWorker=Mock(),
        logger=logging.getLogger(__name__),
    )
    monkeypatch.setattr(myutils, "check_gpu_available", Mock(
        side_effect=RuntimeError("GPU probe failed"),
    ))
    guiWin.SegForLostIDsWorkerAskInstallGPU(window, "cellpose_v3", True)
    assert window.SegForLostIDsWorker.gpu_go is False
    assert window.SegForLostIDsWorker.dont_force_cpu is False
    window._ackSegForLostIDsWorker.assert_called_once_with("ask_install_gpu")
    assert "GPU probe failed" in caplog.text


@pytest.mark.parametrize(
    "kwargs,expected",
    [({}, False), ({"gpu": True}, True), ({"use_gpu": True}, True),
     ({"device_type": "GPU"}, True), ({"device_type": "cpu"}, False),
     ({"device_type": "directml_gpu"}, True)],
)
def test_gpu_request_parameters(kwargs, expected):
    assert myutils.check_gpu_requested_segm_model(kwargs) is expected


@pytest.mark.parametrize('python_exec', [None, 'model-python'])
@pytest.mark.parametrize('kwargs,use_gpu,cuda,directml', [
    ({'device_type': 'cpu'}, False, False, False),
    ({'device_type': 'GPU'}, True, True, False),
    ({'device_type': 'directml_gpu'}, True, False, True),
    ({'gpu': True}, True, False, False),
    ({'use_gpu': True}, True, False, False),
])
@pytest.mark.parametrize('proceed', [False, True])
def test_segmentation_module_checks_gpu_before_init(
        monkeypatch, python_exec, kwargs, use_gpu, cuda, directml, proceed,
    ):
    pytest.importorskip('qtpy')
    from cellacdc.segm import segmWin

    execution = SimpleNamespace(
        env_python=python_exec, init_model=Mock(return_value=object()),
        cancel_preparation=Mock(),
    )
    pos_data = object()
    window = SimpleNamespace(
        model_name='cellpose_v3', modelExecution=execution,
        posData=pos_data, _syncModelExecution=Mock(), logger=Mock(),
    )
    events = []

    def check(*args, **kwargs):
        events.append('check')
        return proceed

    execution.init_model.side_effect = lambda *a: events.append('init') or 'model'
    check_gpu = Mock(side_effect=check)
    monkeypatch.setattr(myutils, 'check_gpu_available', check_gpu)
    result = segmWin.initModelExecution(window, kwargs)
    check_gpu.assert_called_once_with(
        'cellpose_v3', use_gpu, qparent=window, cuda=cuda, directML=directml,
        python_exec=python_exec,
    )
    if proceed:
        assert result == 'model'
        assert events == ['check', 'init']
        execution.init_model.assert_called_once_with(kwargs, pos_data)
        window._syncModelExecution.assert_called_once()
    else:
        assert result is None
        assert events == ['check']
        execution.init_model.assert_not_called()
        window._syncModelExecution.assert_not_called()
