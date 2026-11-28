"""Whether the environment of a segmentation model or tracker is ready: the
check_install_<model> functions (plugin API), the Cellpose version
helpers, _CONVPAINT_PACKAGES, the GPU framework checks (also run with
the Python of a model environment through a subprocess template, which
works with the myutils package and with the single myutils.py of older
versions), check_gpu_available (plugin API) and the GPU install prompt.
Built on installs.py and on torch_utils.py (check_install_torch). The
message box and push buttons of the GPU install prompt
(shared/widgets/message_box.py, push_buttons.py) are imported under if
GUI_INSTALLED.
"""
import shlex
from typing import Literal
import sys
import logging
import subprocess
import importlib
from importlib import import_module
import json
from cellacdc import GUI_INSTALLED
from cellacdc import html_utils, is_win, is_mac_arm64
from cellacdc import _warnings
from .torch_utils import check_install_torch
from cellacdc.models._cellpose_base import min_target_versions_cp
if GUI_INSTALLED:
    from cellacdc.shared.widgets import (
        message_box as widgets_message_box,
        push_buttons as widgets_push_buttons,
    )
from . import installs


def uninstall_omnipose_acdc():
    """Uninstall omnipose-acdc if present. Since v1.5.0 it is not needed.
    """    
    import json
    pip_list_output = subprocess.check_output(
        [sys.executable, '-m', 'pip', 'list', '--format', 'json']
    )
    installed_packages = json.loads(pip_list_output)
    pkgs_to_uninstall = []
    for package_info in installed_packages:
        if package_info['name'] == 'omnipose-acdc':
            pkgs_to_uninstall.append('omnipose-acdc')
        elif package_info['name'] == 'cellpose-omni-acdc':
            pkgs_to_uninstall.append('cellpose-omni-acdc')

    for pkg_to_uninstall in pkgs_to_uninstall:
        installs.uninstall_pip_package(pkg_to_uninstall)

def get_cellpose_major_version(errors='raise'):
    major_installed = None
    try:
        installed_version = installs.get_package_version('cellpose')
        major_installed = int(installed_version.split('.')[0])
    except Exception as err:
        if errors == 'raise':
            raise err
    
    return major_installed

def check_cellpose_version(version: str):
    if isinstance(version, int):
        version = f'{version}.0'

    major_requested = int(version.split('.')[0])
    cancel = False
    try:
        installed_version = installs.get_package_version('cellpose')
        major_installed = int(installed_version.split('.')[0])
        is_version_correct = major_installed == major_requested
        if not is_version_correct:
            cancel = _warnings.warn_installing_different_cellpose_version(
                version, installed_version
            )
        if not installs.is_second_version_greater(
            min_target_versions_cp[str(major_requested)],
            installed_version
        ):
            is_version_correct = False
    except Exception as err:
        is_version_correct = False
    
    if cancel:
        raise ModuleNotFoundError('Cellpose installation cancelled by the user.')
    return is_version_correct


def check_install_cellpose(
        version: Literal['2.0', '3.0', '4.0', 'any'] = '2.0', 
        version_to_install_if_missing: Literal['2.0', '3.0', '4.0'] = '4.0',
        parent=None
    ):
    if isinstance(version, int):
        version = f'{version}.0'
        
    check_install_torch(qparent=parent)

    if version == 'any':
        try:
            from cellpose import models
            return
        except Exception as err:
            version = version_to_install_if_missing # after this the version will for sure be a valid format and not 'any'
            
    is_version_correct = check_cellpose_version(version)
    if is_version_correct:
        return
    
    major_version = int(version.split('.')[0])

    next_version = major_version+1

    min_version = min_target_versions_cp[str(major_version)]
    
    py_version = f'{sys.version_info.major}.{sys.version_info.minor}'
    install_deps_separately = True if major_version <= 3 and py_version >= '3.13' else False
    
    installs.check_install_package(
        'cellpose', 
        max_version=f'{next_version}.0',
        min_version=min_version,
        include_lower_version=True,
        install_dependencies=not install_deps_separately,
        parent= parent
    )
    
    if install_deps_separately:
        installs.check_install_package(
            'fastremap',
            parent=parent
        )
        installs.check_install_package(
            'numba',
            parent=parent
        )
        installs.check_install_package(
            'roifile',
            parent=parent
        )
        installs.check_install_package(
            'imagecodecs',
            parent=parent
        )
        installs.check_install_package(
            'fill_voids',
            parent=parent
        )

    installs.purge_module('cellpose')

def check_install_baby():
    installs.check_install_package(
        'TensorFlow', 
        pypi_name='tensorflow', 
        import_pkg_name='tensorflow', 
        min_version='2.12',
        include_lower_version=True,
        max_version='2.13'
    )
    # TF 2.13 pins typing-extensions<4.6, incompatible with pydantic/fastapi
    installs.check_install_package(
        'typing-extensions', 
        import_pkg_name='typing_extensions', 
        min_version='4.16',
        include_lower_version=True
    )
    installs.check_install_package('baby', pypi_name='baby-seg', import_pkg_name='baby')

def check_install_nnInteractive():
    installs.check_install_package('huggingface-hub')
    check_install_torch()
    installs.check_install_package('nnInteractive')
    
    installs.purge_module('nnInteractive')

    importlib.invalidate_caches()
    import nnInteractive
    importlib.reload(nnInteractive)

# Dependencies of the ConvPaint model (napari-convpaint without napari/Qt).
# Keep in sync with `MODEL_ENVS['ConvPaint']['packages']` in
# `cellacdc/models/_env_config.py`
_CONVPAINT_PACKAGES = [
    {'pkg_name': 'catboost', 'min_version': '1.2.8',
     'include_lower_version': True},
    {'pkg_name': 'timm', 'min_version': '1.0.20',
     'include_lower_version': True},
    'einops',
    'joblib',
    'pandas',
    {'pkg_name': 'pyyaml', 'import_pkg_name': 'yaml'},
    'requests',
    {'pkg_name': 'scikit-image', 'import_pkg_name': 'skimage'},
    {'pkg_name': 'scikit-learn', 'import_pkg_name': 'sklearn'},
    'scipy',
]

def check_install_convpaint():
    import importlib.util

    check_install_torch()
    installs.check_install_package(_CONVPAINT_PACKAGES)

    # Never import `napari_convpaint` here: its `__init__.py` imports napari
    # and Qt (the model is imported without it, see
    # `promptable_models/ConvPaint/acdcPromptSegment.py`)
    is_installed = importlib.util.find_spec('napari_convpaint') is not None
    if is_installed and installs.check_pkg_version(
            'napari-convpaint', '1.2', True, raise_err=False
        ):
        return

    registered_stub = False
    if is_installed and 'napari_convpaint' not in sys.modules:
        # `check_install_package` imports the package to check its version:
        # let it import an empty package module (the version is then read
        # from the metadata) instead of running the outdated `__init__.py`
        spec = importlib.util.find_spec('napari_convpaint')
        sys.modules['napari_convpaint'] = importlib.util.module_from_spec(spec)
        registered_stub = True
    try:
        installs.check_install_package(
            'napari-convpaint',
            import_pkg_name='napari_convpaint',
            min_version='1.2',
            include_lower_version=True,
            upgrade=is_installed,
            install_dependencies=False,
            note=(
                'napari-convpaint is installed without its dependencies '
                '(e.g., napari), because Cell-ACDC only uses the ConvPaint '
                'model.'
            )
        )
    finally:
        if is_installed or registered_stub:
            # Make sure that the (upgraded) package files are loaded next
            # time (do not use `purge_module`, it re-imports the package)
            for module_name in list(sys.modules):
                is_convpaint_module = (
                    module_name == 'napari_convpaint'
                    or module_name.startswith('napari_convpaint.')
                )
                if is_convpaint_module:
                    sys.modules.pop(module_name, None)
            importlib.invalidate_caches()

def check_install_microsam(min_version='2.0'):
    # Keep in sync with `MODEL_ENVS['micro-sam_v1']` and
    # `MODEL_ENVS['micro-sam_v2']` in `cellacdc/models/_env_config.py` (pip
    # installation as in the official micro-sam documentation). micro-sam
    # >= 2.0 has the SAM1 API in `micro_sam.v1` and the SAM2 API in
    # `micro_sam.v2`. micro-sam v2 needs >= 2.0.1, the first release that
    # declares its `sam2` dependency
    check_install_torch()
    installs.check_install_package(
        'micro_sam',
        pypi_name='micro_sam',
        min_version=min_version,
        include_lower_version=True,
    )

def check_install_yeaz():
    check_install_torch()
    installs.check_install_package('yeaz')

def check_install_segment_anything():
    check_install_torch()
    installs.check_install_package('segment_anything')

def check_install_sam2():
    check_install_torch()
    installs.check_install_package('sam2')


def check_install_cellsam():
    check_install_torch()
    installs.check_install_package(
        'cellSAM',
        pypi_name='git+https://github.com/vanvalenlab/cellSAM.git',
        import_pkg_name='cellSAM',
        note=(
            'CellSAM requires a DeepCell access token to download models.\n'
            'Set the DEEPCELL_ACCESS_TOKEN environment variable before use.\n'
            'Get your token at: https://deepcell.org'
        )
    )


def check_install_omnipose():
    try:
        import_module('omnipose')
        return
    except ModuleNotFoundError:
        if installs.is_server_mode():
            installs._raise_server_missing('omnipose')
    
    try:
        installs.check_install_package('omnipose', pypi_name='omnipose_acdc')
    except Exception as err:
        installs.install_package_conda('mahotas')
        installs._install_pip_package('omnipose-acdc')


def _warn_install_gpu(model_name, ask_installs, qparent=None, python_exec=None):
    
    cellpose_cuda_url = (
        r'https://github.com/mouseland/cellpose#gpu-version-cuda-on-windows-or-linux'
    )
    torch_cuda_url = (
        r'https://pytorch.org/get-started/locally/'
    )
    direct_ml_url = (
         r'https://microsoft.github.io/DirectML/'
    )
    torch_directml_url = (
        r'https://learn.microsoft.com/en-us/windows/ai/directml/pytorch-windows'
        )


    cellpose_href = f'{html_utils.href_tag("here", cellpose_cuda_url)}'
    torch_href = f'{html_utils.href_tag("here", torch_cuda_url)}'
    direct_ml_href = f'{html_utils.href_tag("direct_ml_DirectMLref", direct_ml_url)}'
    torch_directml_href = f'{html_utils.href_tag("directml pytorch", torch_directml_url)}'

    py_exec = python_exec or sys.executable
    pip_args = [py_exec, '-m', 'pip']
    pip_command = (
        subprocess.list2cmdline(pip_args) if is_win
        else ' '.join(shlex.quote(arg) for arg in pip_args)
    )
    cuda_commands = (
        f'{pip_command} uninstall torch torchvision torchaudio',
        f'{pip_command} install torch torchvision torchaudio '
        '--index-url https://download.pytorch.org/whl/cu128',
    )

    msg = widgets_message_box.myMessageBox(showCentered=False, wrapText=False)
    txt = html_utils.paragraph(f"""
        In order to use <code>{model_name}</code> with the GPU you need 
        to install a <b>PyTorch version which can use it</b>.<br>
        We recomment using CUDA over DirectML, but if you are using a Windows
        machine with an AMD GPU, you can use DirectML.<br>
        Model environment: <code>{py_exec}</code><br>
        """)
    txt_cuda_title = html_utils.paragraph(f"<b>CUDA</b>", font_size='18px')

    txt_cuda = html_utils.paragraph(f"""
        Check out these instructions {cellpose_href}, and {torch_href}.<br>
        Run the two copyable commands below in order: first uninstall the CPU
        version of PyTorch, then install the CUDA version required by your GPU
        (CUDA 12.8 in the second command).<br>
        """)
    
    add_info = html_utils.to_admonition(
        f"""
        Pleae use the following table to find the correct link for the command.
        You can check the highest CUDA  <br> version supported on your system with the
        command <code>nvidia-smi</code> in the terminal.<br>

        {html_utils.table_style_header}
            <tr>
                <th>CUDA Version</th>
                <th>PyTorch Installation Link</th>
            </tr>
            <tr>
                <td>CUDA 11.8</td>
                <td><code>https://download.pytorch.org/whl/cu118</code></td>
            </tr>
            <tr>
                <td>CUDA 12.6</td>
                <td><code>https://download.pytorch.org/whl/cu126</code></td>
            </tr>
            <tr>
                <td>CUDA 12.8</td>
                <td><code>https://download.pytorch.org/whl/cu128</code></td>
            </tr>
        </table>
        """,
        "info"
    )
    
    txt_cuda = f'{txt_cuda}{add_info}'
    
    txt_directML_title = html_utils.paragraph(f"<b>DirectML</b>", font_size='18px')
    txt_directML = html_utils.paragraph(f"""
        Check out {direct_ml_href}, and {torch_directml_href} for more info.<br>
        Only supported on Windows 10/11 with Python 3.8-3.12.<br>
        Click the <b>Install DirectML</b> button to install DirectML.
        <br><br>
        """)
    
    txt_end = html_utils.paragraph(f"""
        How do you want to proceed?
    """)

    stopButton = widgets_push_buttons.cancelPushButton('Stop the process')
    directMLButton = widgets_push_buttons.okPushButton('Install DirectML')
    proceedButton = widgets_push_buttons.okPushButton('Proceed without GPU')

    buttons = [stopButton]

    if 'cuda' in ask_installs:
        txt = f'{txt}{txt_cuda_title}{txt_cuda}'
    if 'directML' in ask_installs:
        txt = f'{txt}{txt_directML_title}{txt_directML}'
        buttons.append(directMLButton)
    txt = f'{txt}{txt_end}' 
    buttons.append(proceedButton)

    msg.warning(
        qparent, 'PyTorch GPU version not installed', txt, 
        buttonsTexts=buttons,
        commands=cuda_commands if 'cuda' in ask_installs else None,
    )

    if msg.cancel:
        return False, False
    
    if msg.clickedButton == directMLButton:
        py_ver = sys.version_info
        if python_exec is not None and python_exec != sys.executable:
            proc = subprocess.run(
                [python_exec, '-c',
                 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'],
                capture_output=True, text=True, check=True,
            )
            major, minor = map(int, proc.stdout.strip().split('.'))
        else:
            major, minor = py_ver.major, py_ver.minor
        if is_win and major == 3 and 8 <= minor < 13:
            success = installs.check_install_package(
                pkg_name = 'torch-directml',
                import_pkg_name = 'torch_directml',
                pypi_name = 'torch-directml',
                return_outcome=True,
                python_exec=python_exec,
                qparent=qparent,
            )
            if python_exec in (None, sys.executable):
                installs.purge_module('torch')
            return success, True
        else:
            msg = widgets_message_box.myMessageBox()
            msg.warning(
                qparent, 'DirectML not supported', 
                'DirectML is only supported on Python 3.8-3.12 and Windows 10/11',
            )
            return False, False

    if msg.clickedButton == stopButton:
        return False, False

    if msg.clickedButton == proceedButton:
        return True, False

def check_gpu_requested_segm_model(init_kwargs):
    gpu = init_kwargs.get('gpu', False) or init_kwargs.get('use_gpu', False)
    if gpu:
        return True
    
    device_type = init_kwargs.get('device_type', 'cpu').lower()
    return device_type in ('gpu', 'directml_gpu', '')


def _available_gpu_frameworks(cuda, directML):
    """Inspect GPU backends only in the interpreter that runs the model."""
    available = []
    if cuda:
        try:
            import torch
        except ModuleNotFoundError as err:
            if err.name != 'torch':
                raise
        else:
            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                available.append('cuda')
            if is_mac_arm64 and torch.backends.mps.is_available():
                available.append('mps')
    if directML and is_win:
        try:
            import torch_directml
        except ModuleNotFoundError as err:
            if err.name != 'torch_directml':
                raise
        else:
            if torch_directml.is_available():
                available.append('directML')
    return available


def _available_gpu_frameworks_from_env(python_exec, cuda, directML):
    # Runs with the Cell-ACDC of the model environment, which may still
    # have the single-module myutils.py of older versions
    script = f'''
import json
import cellacdc.myutils
if hasattr(cellacdc.myutils, "__path__"):
    from cellacdc.myutils.model_installs import _available_gpu_frameworks
else:
    from cellacdc.myutils import _available_gpu_frameworks
print("__ACDC_GPU__" + json.dumps(_available_gpu_frameworks({cuda!r}, {directML!r})))
'''
    proc = subprocess.run(
        [python_exec, '-c', script, '--server'],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f'Failed to check GPU availability in {python_exec}:\n{proc.stderr}'
        )
    for line in proc.stdout.splitlines():
        if line.startswith('__ACDC_GPU__'):
            return json.loads(line[len('__ACDC_GPU__'):])
    raise RuntimeError(
        f'No GPU availability returned by {python_exec}:\n{proc.stdout}'
    )

def check_gpu_available(
        model_name, use_gpu, 
        do_not_warn=False, 
        qparent=None, 
        cuda=False, 
        directML=False, 
        return_available_gpu_type=False,
        python_exec=None,
    ):
    """Check the model's GPU backends in `python_exec` (current process by default).

    Pass the interpreter selected by ModelExecution for isolated models. GPU
    warnings and installations stay in the client and target that interpreter.
    `do_not_warn` never opens dialogs or installs packages.
    """
    if not use_gpu:
        if return_available_gpu_type:
            return True, []
        else:
            return True
    
    frameworks = _available_frameworks(model_name)
    check_cuda = cuda or frameworks['cuda']
    check_directML = directML or frameworks['directML']
    if python_exec is not None and python_exec != sys.executable:
        detected = _available_gpu_frameworks_from_env(
            python_exec, check_cuda, check_directML
        )
    else:
        detected = _available_gpu_frameworks(check_cuda, check_directML)
    ask_for_cuda = cuda and 'cuda' not in detected
    ask_for_directML = directML and is_win and 'directML' not in detected
    ask_installs = {
        name for name, check in (
            ('cuda', check_cuda), ('directML', check_directML and is_win)
        ) if check and name not in detected
    }
    available_frameworks_list = detected
    framework_available = bool(available_frameworks_list)
    
    if framework_available and not ask_for_cuda and not ask_for_directML:
        if return_available_gpu_type:
            return True, available_frameworks_list
        else:
            return True

    elif do_not_warn:
        if return_available_gpu_type:
            return False, available_frameworks_list
        else:
            return False
    
    if installs.is_server_mode():
        logging.getLogger(__name__).warning(
            'GPU unavailable for %s in %s; using CPU. '
            'Install GPU support from the Cell-ACDC client.',
            model_name, sys.executable,
        )
        if return_available_gpu_type:
            return True, available_frameworks_list
        return True

    proceed, directML_installed = _warn_install_gpu(
        model_name, ask_installs, qparent=qparent, python_exec=python_exec
    )
    if return_available_gpu_type:
        if directML_installed:
            available_frameworks_list.append('directML')
        return proceed, available_frameworks_list
    else:
        return proceed


def _available_frameworks(model_name):
    frameworks = {

    "cuda":(
        model_name.lower().find('cellpose') != -1
        or model_name.lower().find('omnipose') != -1
        or model_name.lower().find('deepsea') != -1
        or model_name.lower().find('segment_anything') != -1
        or model_name.lower().find('sam2') != -1
        or model_name.lower().find('yeaz') != -1
        or model_name.lower().find('yeaz_v2') != -1
    ),
    "directML":(
        model_name.lower().find('cellpose_v4') != -1
        or model_name.lower().find('cellpose_v3') != -1# has its own way to check

    )
    }
    return frameworks


def check_install_tapir():
    installs.check_install_package(
        'tapnet', pypi_name='git+https://github.com/ElpadoCan/TAPIR.git'
    )


def check_install_trackastra():
    installs.check_install_package(
        'Trackastra', 
        import_pkg_name='trackastra', 
        pypi_name='trackastra'
    )


def check_install_instanseg():
    installs.check_install_package(
        pkg_name='InstanSeg',
        import_pkg_name='instanseg', 
        pypi_name='instanseg-torch'
    )
