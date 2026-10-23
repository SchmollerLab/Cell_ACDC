"""PyTorch installation, wheel discovery, version checks and device helpers.

PyTorch and GUI modules are imported only inside the helpers that need them.
"""

import ctypes
import functools
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import traceback
from typing import Callable, Literal, Optional, Union

import requests
from packaging.specifiers import SpecifierSet

from ._model_startup import startup_timer


def torch_version_specifier(torch_version: Optional[str]) -> str:
    """Accept a plain release or a PEP 440 version constraint."""
    if not torch_version:
        return ""
    constraint = torch_version.strip()
    if not constraint.startswith(("<", ">", "=", "!", "~")):
        constraint = f"=={constraint}"
    SpecifierSet(constraint)
    return constraint


def torch_requirement(torch_version: Optional[str], quoted: bool = False) -> str:
    """Build a pip requirement, quoting shell redirection operators if needed."""
    requirement = f"torch{torch_version_specifier(torch_version)}"
    if quoted and ("<" in requirement or ">" in requirement):
        return f'"{requirement}"'
    return requirement


def _warn_dll_torch(qparent=None):
    from . import html_utils, widgets

    msg = widgets.myMessageBox()
    txt = html_utils.paragraph("""
    <b>An error message will occur after you close this message.</b><br>
    <b>Please save your data and restart Cell-ACDC.</b><br>
    Sorry for the inconvenience!<br>
    This error is not critical for the main functionality of Cell-ACDC,
    and only concerns the segmentation model. Your can save your data without
    a problem.<br>
    The specific reason is that PyTorch and QtPy have weird issues with
    DLL conflicts.
    """)
    msg.information(
    qparent, 'Please restart Cell-ACDC', txt,
    buttonsTexts=('Ok, I will save my data and restart Cell-ACDC'),
    )


def check_install_torch(is_cli=False, caller_name='Cell-ACDC', qparent=None, python_exec=None,
                        platform=None, found_cuda_version=None, possible_links=None):
    from . import _warnings, myutils
    from .myutils import is_gui_running, is_server_mode

    if is_server_mode() and python_exec in (None, sys.executable):
        try:
            import torch
            import torchvision
            return
        except Exception as err:
            raise ModuleNotFoundError(
                f'PyTorch could not be imported in the model worker '
                f'environment ({err}). It is installed by the client when '
                f'`torch: True` is set in cellacdc/models/_env_config.py.'
            ) from err

    if python_exec is not None and python_exec != sys.executable:
        # Isolated model env: single installer that detects CPU/CUDA
        from cellacdc.service.environment import ensure_torch
        from cellacdc.service import prompts
        if is_cli or not is_gui_running():
            handler = prompts.CliHandler()
        else:
            handler = prompts.QtHandler(qparent)
        platform = platform or sys.platform
        if not ensure_torch(python_exec, prompt_func=handler.ask,
                            platform=platform,
                            found_cuda_version=found_cuda_version,
                            possible_links=possible_links,
                            ):
            raise ModuleNotFoundError(
                f'PyTorch is not installed in {python_exec} (installation '
                'cancelled or failed). The model cannot run without it.'
            )
        return

    try:
        import torch
        import torchvision
        return

    except OSError as err:
        if 'dll' in str(err):
            _warn_dll_torch(qparent=qparent)
            raise err
        else:
            traceback.print_exc()
    except Exception as err:
        traceback.print_exc()
        print(f'Failed to import PyTorch in venv ({sys.executable}): {err}')

    if is_cli:
        _install_pytorch_cli(caller_name=caller_name)
        return

    from . import apps

    win = apps.InstallPyTorchDialog(parent=qparent, caller_name=caller_name)
    win.exec_()
    if win.cancel:
        _warnings.log_pytorch_not_installed()
        return

    command = win.command
    print(f'Running command: "{command}"')
    myutils._run_command(command)

    try:
        import torch
    except OSError as e:
        if 'dll' in str(e):
            _warn_dll_torch(qparent=qparent)
        raise e

    myutils.purge_module('torch')


def get_driver_cuda_version():
    """Max CUDA version supported by the driver, as (major, minor), or False."""
    # 1) CUDA driver API (no dependencies, no nvidia-smi parsing)


    def _decode(v: int):
        return v // 1000, (v % 1000) // 10  # 13040 -> (13, 4)

    libname = "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
    try:
        lib = ctypes.CDLL(libname)
        v = ctypes.c_int()
        if lib.cuDriverGetVersion(ctypes.byref(v)) == 0:
            return _decode(v.value)
    except (OSError, AttributeError):
        pass

    # 2) NVML, if pynvml / nvidia-ml-py is installed
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            return _decode(pynvml.nvmlSystemGetCudaDriverVersion())
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        pass

    # 3) Last resort: parse nvidia-smi, tolerant of label changes
    #    ("CUDA Version: 12.4", "CUDA UMD Version: 13.4", ...)
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True,
                             check=True, timeout=15).stdout
        m = re.search(r"CUDA\s+(?:\w+\s+)?Version\s*:\s*(\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass

    return False


@functools.lru_cache(maxsize=None)
def _get(url):
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    return r.text


BASE_PYTORCH_LINK = "https://download.pytorch.org/whl/"


def get_torch_options(driver_cuda_version=None, torch_max="", torch_min="",
                      compute_cap=None, allow_experimental=False, min_cuda=None,
                      torch_version=None, exec_path=None):
    """
    Returns (options, preferred_index).

    options: list of (label, index_url) tuples, e.g.
        [('CPU', '.../cpu'), ('CUDA 12.6 (NVIDIA GPU)', '.../cu126'), ('CUDA 13.0 (NVIDIA GPU)', '.../cu130')]
    index_url is None where the default PyPI install applies (macOS).
    preferred_index: the option that fits this machine (highest CUDA build the driver
        and GPU support, else CPU).

    torch_min / torch_max : inclusive bounds like '2.9' or '2.12'; builds without a release
                            in that range are not listed.
    compute_cap           : (major, minor) of the weakest GPU; < (7, 5) caps the preference at cu126.
    torch_version         : release or PEP 440 constraint (e.g. '<2'); only matching
                            CUDA indexes are listed.
    min_cuda              : oldest CUDA build to list; defaults to (12, 6), or (0, 0)
                            for a pinned release so older builds remain available.
    exec_path             : target Python executable. Wheel compatibility uses its
                            Python version, OS and architecture, not the host's.
                            Failure to query the interpreter is raised to the caller.
    """
    EXPERIMENTAL = {"cu132"}  # only offered if allow_experimental=True
    from packaging.specifiers import SpecifierSet
    from packaging.version import Version

    release_spec = SpecifierSet(torch_version_specifier(torch_version))
    if min_cuda is None:
        min_cuda = (0, 0) if torch_version else (12, 6)

    python_version = sys.version_info[:2]
    target_platform = sys.platform
    machine = platform.machine()
    if exec_path is not None and exec_path != sys.executable:
        result = subprocess.run(
            [exec_path, "-c",
             "import json, platform, sys; "
             "print(json.dumps([sys.version_info[:2], sys.platform, platform.machine()]))"],
            capture_output=True, text=True, check=True, timeout=20,
        )
        python_version, target_platform, machine = json.loads(result.stdout)
    py = f"cp{python_version[0]}{python_version[1]}"
    plat = "win_amd64" if target_platform == "win32" and machine.lower() in (
        "amd64", "x86_64"
    ) else machine

    if target_platform == "darwin":       # no CUDA wheels, default PyPI wheel has MPS
        return [("Default (CPU / Apple GPU)", None)], 0

    def _cuda_tags():
        """CUDA builds on the server as {(12, 6): 'cu126', ...}."""
        return {(int(d[:-1]), int(d[-1])): f"cu{d}"
                for d in set(re.findall(r"\bcu(\d{3})\b", _get(BASE_PYTORCH_LINK)))}


    def _torch_versions(tag):
        """Torch versions on this index with a wheel for the target Python + OS/arch."""
        wheels = re.findall(r"torch-(\d+(?:\.\d+)*)(?:%2B|\+)[\w.]+-(cp\d+)-\w+-([\w.]+?)\.whl",
                            _get(f"{BASE_PYTORCH_LINK}{tag}/torch/"))
        return {tuple(map(int, v.split(".")))
                for v, cp, p in wheels if cp == py and plat in p}

    driver = driver_cuda_version or get_driver_cuda_version()
    lo = tuple(map(int, torch_min.split("."))) if torch_min else ()
    hi = tuple(map(int, torch_max.split("."))) if torch_max else None
    limit = (12, 6) if compute_cap and compute_cap < (7, 5) else driver  # Pascal/Volta

    def has_release(tag):
        return any(v[:len(lo)] >= lo and (hi is None or v[:len(hi)] <= hi)
                   and Version(".".join(map(str, v))) in release_spec
                   for v in _torch_versions(tag))

    options = [("CPU", f"{BASE_PYTORCH_LINK}cpu")]
    preferred = 0
    for cuda, tag in sorted(_cuda_tags().items()):            # ascending, like your example
        if cuda < min_cuda or (tag in EXPERIMENTAL and not allow_experimental):
            continue
        if not has_release(tag):
            continue
        extra = ", experimental" if tag in EXPERIMENTAL else ""
        options.append((f"CUDA {cuda[0]}.{cuda[1]} (NVIDIA GPU{extra})", f"{BASE_PYTORCH_LINK}{tag}"))
        if driver and cuda <= driver and cuda <= limit and tag not in EXPERIMENTAL:
            preferred = len(options) - 1                       # last match = highest

    return options, preferred


def resolve_torch_link(*args, **kwargs):
    """Index URL of the preferred option (None -> default PyPI install)."""
    options, preferred = get_torch_options(*args, **kwargs)
    return options[preferred][1]


def _safe_torch_options(options_kwargs):
    options_kwargs = dict(options_kwargs)
    options_kwargs.setdefault('compute_cap', get_min_compute_cap())
    try:
        return get_torch_options(**options_kwargs)
    except requests.RequestException:
        # Offline / server unreachable: fall back to a plain CPU install
        return [('CPU', f'{BASE_PYTORCH_LINK}cpu')], 0


def get_pytorch_command(**options_kwargs):
    """Get the pip commands to install PyTorch (CPU or CUDA) for this machine.

    Parameters
    ----------
    **options_kwargs
        Passed to `get_torch_options` (torch_min, torch_max, allow_experimental,
        min_cuda, driver_cuda_version, compute_cap, exec_path). `compute_cap` is detected
        automatically if not given.

    Returns
    -------
    dict
        {os_name: {'Pip': {label: command}}} for the OS we are running on.
        Labels are in display order (use `list(...)` for the combobox).
        See `get_preferred_pytorch_platform` for the label to preselect.

    Notes
    -----
    As of Oct 2024, the `pytorch` channel on Anaconda was deprecated.
    See here https://github.com/pytorch/pytorch/issues/138506
    Conda commands are therefore no longer generated.
    """

    from .myutils import get_pip_conda_prefix

    _, pip_prefix = get_pip_conda_prefix(exec_path=options_kwargs.get('exec_path'))
    options, _ = _safe_torch_options(options_kwargs)

    os_name = {'win32': 'Windows', 'darwin': 'Mac'}.get(sys.platform, 'Linux')
    pip_commands = {}
    for label, index_url in options:
        torch_version = options_kwargs.get('torch_version')
        torch_spec = torch_requirement(torch_version, quoted=True)
        command = f'{pip_prefix} {torch_spec} torchvision'
        if index_url is not None:   # None -> default PyPI install (macOS)
            command += f' --index-url {index_url}'
        pip_commands[label] = command

    return {os_name: {'Pip': pip_commands}}


def get_torch_options_safe(**kwargs):
    kwargs.setdefault('compute_cap', get_min_compute_cap())
    try:
        return get_torch_options(**kwargs)
    except requests.RequestException:
        return [('CPU', f'{BASE_PYTORCH_LINK}cpu')], 0   # offline: plain CPU install


def get_min_compute_cap():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=15).stdout
        caps = [tuple(int(x) for x in line.strip().split(".")) for line in out.splitlines() if line.strip()]
        return min(caps) if caps else None
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


def _install_pytorch_cli(
        caller_name='Cell-ACDC', action='install', logger_func=print,
        python_exec=None
    ):
    from .myutils import get_cli_multi_choice_question

    separator = '-'*60
    txt = (
        f'{separator}\n{caller_name} needs to {action} PyTorch\n\n'
        'You can choose to install it now or stop the process and install it '
        'later. To install it correctly, we need to know your preferences.\n'
    )
    logger_func(txt)
    questions = {
        'Choose your OS:': ('Windows', 'Mac', 'Linux'),
        'Package manager:': ('Pip'),
        'Compute platform:': (
            'CPU', 'CUDA 11.8 (NVIDIA GPU)', 'CUDA 12.1 (NVIDIA GPU)'
        )
    }
    selected_command = get_pytorch_command()
    selected_preferences = []
    for question, choices in questions.items():
        input_txt = get_cli_multi_choice_question(question, choices)
        while True:
            answer = input(input_txt)
            if answer.lower() == 'q':
                exit('Execution stopped by the user.')

            try:
                idx = int(answer) - 1
                if idx >= len(choices):
                    raise TypeError('Not a valid answer')
            except Exception as err:
                print('-'*100)
                logger_func(
                    f'"{answer}" is not a valid answer.'
                    'Choose one of the options or "q" to quit.'
                )
                print('^'*100)
                continue

            preference = choices[idx]
            selected_command = selected_command[preference]
            selected_preferences.append(preference)
            print('')
            break

    print('-'*100)
    selected_preferences = ', '.join(selected_preferences)
    logger_func(f'Selected preferences: {selected_preferences}')
    print('-'*100)
    logger_func(f'Command:\n\n{selected_command}\n')
    while True:
        answer = input('Do you want to run the command now ([y]/n)?: ')
        if answer.lower() == 'n':
            exit('Execution stopped by the user.')

        if answer.lower() == 'y' or not answer:
            break

        print('-'*100)
        print(
            f'"{answer}" is not a valid answer. '
            'Choose "y" for yes or "n" for no.'
        )
        print('^'*100)

    if selected_command.startswith('conda'):
        try:
            subprocess.check_call([selected_command], shell=True)
        except Exception as err:
            cmd_list = shlex.split(selected_command)
            cmd_list = [cmd.strip('"') for cmd in cmd_list]
            cmd_list = [cmd.strip("'") for cmd in cmd_list]
            cmd_list = [cmd.lstrip(".") for cmd in cmd_list]
            subprocess.check_call(cmd_list, shell=False)
    else:
        cmd_list = shlex.split(selected_command)
        cmd_list = cmd_list[1:]
        cmd_list = [cmd.strip('"') for cmd in cmd_list]
        cmd_list = [cmd.strip("'") for cmd in cmd_list]
        cmd_list = [cmd.lstrip(".") for cmd in cmd_list]
        py_exec = python_exec or sys.executable
        subprocess.check_call([py_exec, *cmd_list], shell=False)


def get_torch_device(gpu=False):
    import torch
    if torch.cuda.is_available() and gpu:
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


def _torch_import_check(torch_version: Optional[str] = None) -> str:
    script = "import torch, torchvision"
    if torch_version:
        constraint = torch_version_specifier(torch_version)
        script += (
            "\nfrom importlib.metadata import version"
            "\nfrom packaging.specifiers import SpecifierSet"
            f"\nassert version('torch') in SpecifierSet({constraint!r}), "
            f"{('Expected torch' + constraint)!r}"
        )
    return script


def ensure_torch(
    python_exec: str,
    logger_func: Callable = print,
    prompt_func: Optional[Callable] = None,
    platform: str = sys.platform,
    found_cuda_version: Optional[Union[tuple[int, int], Literal[False]]] = None,
    possible_links: Optional[tuple[list[tuple[str, Optional[str]]], int]] = None,
    torch_version: Optional[str] = None,
) -> bool:
    """Install PyTorch into the model environment.

    The build (CPU or CUDA) is detected from `nvidia-smi`. If `prompt_func`
    is given (see `cellacdc.service.prompts`), the user is asked on behalf
    of the worker with the detected build preselected. ``torch_version`` accepts
    a release or PEP 440 constraint; pip resolves a compatible torchvision.
    """
    from .service.environment import run_streaming

    check_script = _torch_import_check(torch_version)
    with startup_timer("torch/torchvision import check", logger_func):
        check = subprocess.run(
            [python_exec, "-c", check_script], capture_output=True
        )
    if check.returncode == 0:
        return True

    if possible_links is None:
        with startup_timer("PyTorch installation options discovery", logger_func):
            if found_cuda_version is None:
                found_cuda_version = get_driver_cuda_version()
            options_kwargs = {
                'driver_cuda_version': found_cuda_version,
                'exec_path': python_exec,
            }
            if torch_version:
                options_kwargs['torch_version'] = torch_version
            possible_links = get_torch_options_safe(**options_kwargs)
    options, preferred = possible_links
    choices = [label for label, _ in options]
    default = choices[preferred]
    answer = default
    if prompt_func is not None:
        from cellacdc.service import prompts
        answer = prompt_func(prompts.choice(
            title='Install PyTorch',
            text=(
                f'Select the PyTorch {torch_version} build to install.'
                if torch_version else 'Select the PyTorch build to install.'
            ),
            choices=choices,
            default=default,
            dialog='pytorch',
            extra={'python': python_exec,
                   'platform': platform,
                   'found_cuda_version': found_cuda_version,
                   'possible_links': possible_links,
                   'torch_version': torch_version,},
        ))
        if answer is None:
            logger_func("PyTorch installation cancelled.")
            return False
    urls = dict(options)
    if answer in urls:
        cmd = [
            python_exec, '-m', 'pip', 'install',
            torch_requirement(torch_version), 'torchvision'
        ]
        index_url = urls[answer]
        if index_url is not None:
            cmd.extend(['--index-url', index_url])
    else:
        cmd = answer
        if isinstance(cmd, str) and os.name != 'nt':
            cmd = shlex.split(cmd)

    try:
        run_streaming(cmd, logger_func, check=True)
        if torch_version:
            check = subprocess.run(
                [python_exec, "-c", check_script], capture_output=True, text=True
            )
            if check.returncode:
                logger_func(
                    f"PyTorch {torch_version} verification failed:\n"
                    f"{check.stderr or check.stdout}"
                )
                return False
        return True
    except Exception as err:
        logger_func(f"PyTorch install failed ({err}), trying fallback...")
        print(f"Command attempted: {cmd}")
    return False
