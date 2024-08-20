import os
import sys
import shutil
import subprocess

import pathlib
import numpy as np

from typing import Iterable

KNOWN_EXTENSIONS = (
    '.tif', '.npz', '.npy', '.h5', '.json', '.csv', '.txt'
)

def _warn_ask_install_package(commands: Iterable[str], note_txt=''):
    open_str = '='*100
    sep_str = '-'*100
    commands_txt = '\n'.join([f'  {command}' for command in commands])
    text = (
        f'SpotMAX needs to run the following commands{note_txt}:\n\n'
        f'{commands_txt}\n\n'
    )
    question = (
        'How do you want to proceed?: '
        '1) Run the commands now. '
        'q) Quit, I will run the commands myself (1/q): '
    )
    print(open_str)
    print(text)
    
    message_on_exit = (
        '[WARNING]: Execution aborted. Run the following commands before '
        f'running spotMAX again:\n\n{commands_txt}\n'
    )
    msg_on_invalid = (
        '$answer is not a valid answer. '
        'Type "1" to run the commands now or "q" to quit.'
    )
    try:
        while True:
            answer = input(question)
            if answer == 'q':
                print(open_str)
                exit(message_on_exit)
            elif answer == '1':
                break
            else:
                print(sep_str)
                print(msg_on_invalid.replace('$answer', answer))
                print(sep_str)
    except Exception as err:
        traceback.print_exc()
        print(open_str)
        print(message_on_exit)

def _run_pip_commands(commands: Iterable[str]):
    import subprocess
    for command in commands:
        try:
            subprocess.check_call([sys.executable, '-m', *command.split()])
        except Exception as err:
            pass
    
try:
    import requests
except Exception as err:
    import traceback
    traceback.print_exc()
    print('We detected a corrupted library, fixing it now...')
    commands = (
        'pip uninstall -y charset-normalizer', 
        'pip install --upgrade charset-normalizer'
    )
    _warn_ask_install_package(
        commands, note_txt=' (fixing charset-normalizer package)'
    )
    _run_pip_commands(commands)

def user_data_dir():
    r"""
    Get OS specific data directory path for Cell-ACDC.

    Typical user data directories are:
        macOS:    ~/Library/Application Support/
        Unix:     ~/.local/share/   # or in $XDG_DATA_HOME, if defined
        Win 10:   C:\Users\<username>\AppData\Local\
    For Unix, we follow the XDG spec and support $XDG_DATA_HOME if defined.
    :return: full path to the user-specific data dir
    """
    # get os specific path
    if sys.platform.startswith("win"):
        os_path = os.getenv("LOCALAPPDATA")
    elif sys.platform.startswith("darwin"):
        os_path = "~/Library/Application Support"
    else:
        # linux
        os_path = os.getenv("XDG_DATA_HOME", "~/.local/share")

    os_path = os.path.expanduser(os_path)
    return os.path.join(os_path, 'Cell_ACDC')

cellacdc_path = os.path.dirname(os.path.abspath(__file__))
debug_true_filepath = os.path.join(cellacdc_path, '.debug_true')
qrc_resources_path = os.path.join(cellacdc_path, 'qrc_resources.py')
qrc_resources_light_path = os.path.join(cellacdc_path, 'qrc_resources_light.py')
qrc_resources_dark_path = os.path.join(cellacdc_path, 'qrc_resources_dark.py')
old_temp_path = os.path.join(cellacdc_path, 'temp')
tooltips_rst_filepath = os.path.join(
    cellacdc_path, "docs", "source", "tooltips.rst"
)

user_data_folderpath = user_data_dir()
user_profile_path_txt = os.path.join(
    user_data_folderpath, 'acdc_user_profile_location.txt'
)
user_home_path = str(pathlib.Path.home())
user_profile_path = os.path.join(user_home_path, 'acdc-appdata')
if os.path.exists(user_profile_path_txt):
    try:
        with open(user_profile_path_txt, 'r') as txt:
            user_profile_path = fr'{txt.read()}'
    except Exception as e:
        pass

try:
    os.makedirs(user_profile_path, exist_ok=True)
except Exception as e:
    print(
        f'[WARNING]: User profile path was not found "{user_profile_path}". '
        f'Resetting back to default path "{user_home_path}".'
    )
    user_profile_path = user_home_path

# print(f'User profile path: "{user_profile_path}"')

import site
sitepackages = site.getsitepackages()
site_packages = [p for p in sitepackages if p.endswith('site-packages')][0]

cellacdc_path = os.path.dirname(os.path.abspath(__file__))
cellacdc_installation_path = os.path.dirname(cellacdc_path)

if cellacdc_installation_path != site_packages:
    IS_CLONED = True
    settings_folderpath = os.path.join(cellacdc_installation_path, '.acdc-settings')
else:
    IS_CLONED = False
    settings_folderpath = os.path.join(user_profile_path, '.acdc-settings')

def copytree(src, dst):
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(old_temp_path):
        src_filepath = os.path.join(src, name)
        dst_filepath = os.path.join(dst, name)
        if os.path.isdir(src_filepath):
            copytree(src_filepath, dst_filepath)
        elif os.path.isfile(src_filepath):
            shutil.copy2(src_filepath, dst_filepath)

if not os.path.exists(settings_folderpath):
    os.makedirs(settings_folderpath, exist_ok=True)
if os.path.exists(old_temp_path):
    try:
        copytree(old_temp_path, settings_folderpath)
        shutil.rmtree(old_temp_path)
    except Exception as e:
        print('*'*60)
        print(
            '[WARNING]: could not copy settings from previous location. '
            f'Please manually copy the folder "{old_temp_path}" to "{settings_folderpath}"')
        print('^'*60)

import pandas as pd
settings_csv_path = os.path.join(settings_folderpath, 'settings.csv')
if not os.path.exists(settings_csv_path):
    df_settings = pd.DataFrame(
        {'setting': [], 'value': []}).set_index('setting')
    df_settings.to_csv(settings_csv_path)

# Get color scheme
if not os.path.exists(settings_csv_path):
    scheme = 'light'
df_settings = pd.read_csv(settings_csv_path, index_col='setting')
if 'colorScheme' not in df_settings.index:
    scheme = 'light'
else:
    scheme = df_settings.at['colorScheme', 'value']

# Set default qrc resources
if not os.path.exists(qrc_resources_path):
    if scheme == 'light':
        qrc_resources_scheme_path = qrc_resources_light_path
    else:
        qrc_resources_scheme_path = qrc_resources_dark_path
    # Load default light mode
    shutil.copyfile(qrc_resources_scheme_path, qrc_resources_path)

# Replace 'from PyQt5' with 'from qtpy' in qrc_resources.py file
try:
    save_qrc = False
    with open(qrc_resources_path, 'r') as qrc_py:
        text = qrc_py.read()
        if text.find('from PyQt5') != -1:
            text = text.replace('from PyQt5', 'from qtpy')
            save_qrc = True
    if save_qrc:
        with open(qrc_resources_path, 'w') as qrc_py:
            qrc_py.write(text)
except Exception as err:
    raise err

def try_input_install_package(pkg_name, install_command, question=None):
    if question is None:
        question = 'Do you want to install it now ([y]/n)? '
    try:
        answer = input(f'\n{question}')
        return answer
    except Exception as err:
        raise ModuleNotFoundError(
            f'The module "{pkg_name}" is not installed. '
            f'Install it with the command `{install_command}`.'
        )

import os
import inspect
import platform
import traceback
from datetime import datetime
from pprint import pprint

from functools import wraps

try:
    # Force PyQt6 if available
    try:
        from PyQt6 import QtCore
        os.environ["QT_API"] = "pyqt6"
    except Exception as e:
        pass
    from qtpy import QtCore
    import pyqtgraph
    import matplotlib
    GUI_INSTALLED = True
except Exception as e:
    GUI_INSTALLED = False        

import pandas as pd

np.random.seed(3548784512)

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 200)
pd.set_option('display.expand_frame_repr', False)

open_printl_str = '*'*100
close_printl_str = '='*100

def printl(*objects, pretty=False, is_decorator=False, idx=1, **kwargs):
    timestap = datetime.now().strftime('%H:%M:%S')
    currentframe = inspect.currentframe()
    outerframes = inspect.getouterframes(currentframe)
    idx = idx+1 if is_decorator else idx
    callingframe = outerframes[idx].frame
    callingframe_info = inspect.getframeinfo(callingframe)
    filepath = callingframe_info.filename
    fileinfo_str = (
        f'File "{filepath}", line {callingframe_info.lineno} - {timestap}:'
    )
    if pretty:
        print(open_printl_str)
        print(fileinfo_str)
        for o, object in enumerate(objects):
            text = str(object)                
            pprint(text, **kwargs)
        print(close_printl_str)
    else:
        sep = kwargs.get('sep', ', ')
        text = sep.join([str(object) for object in objects])
        text = f'{open_printl_str}\n{fileinfo_str}\n{text}\n{close_printl_str}'
        print(text)

parent_path = os.path.dirname(cellacdc_path)
html_path = os.path.join(cellacdc_path, '_html')
models_path = os.path.join(cellacdc_path, 'models')
data_path = os.path.join(parent_path, 'data')
resources_folderpath = os.path.join(cellacdc_path, 'resources')
resources_filepath = os.path.join(cellacdc_path, 'resources_light.qrc')
logs_path = os.path.join(user_profile_path, '.acdc-logs')
acdc_fiji_path = os.path.join(user_profile_path, 'acdc-fiji')
acdc_ffmpeg_path = os.path.join(user_profile_path, 'acdc-ffmpeg')
resources_path = os.path.join(cellacdc_path, 'resources_light.qrc')
models_list_file_path = os.path.join(settings_folderpath, 'custom_models_paths.ini')
recentPaths_path = os.path.join(settings_folderpath, 'recentPaths.csv')
user_manual_url = 'https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf'
github_home_url = 'https://github.com/SchmollerLab/Cell_ACDC'
data_structure_docs_url = 'https://cell-acdc.readthedocs.io/en/latest/data-structure.html'

# Use to get the acdc_output file name from `segm_filename` as 
# `m = re.sub(segm_re_pattern, '_acdc_output', segm_filename)`
segm_re_pattern = r'_segm(?!.*_segm)'

try:
    from setuptools_scm import get_version
    __version__ = get_version(root='..', relative_to=__file__)
except Exception as e:
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "not-installed"

__author__ = 'Francesco Padovani and Benedikt Mairhoermann'

cite_url = 'https://bmcbiol.biomedcentral.com/articles/10.1186/s12915-022-01372-6'
issues_url = 'https://github.com/SchmollerLab/Cell_ACDC/issues'

# Initialize variables that need to be globally accessible


base_cca_dict = {
    'cell_cycle_stage': 'G1',
    'generation_num': 2,
    'relative_ID': -1,
    'relationship': 'mother',
    'emerg_frame_i': -1,
    'division_frame_i': -1,
    'is_history_known': False,
    'corrected_assignment': False,
    'will_divide': 0,
    'daughter_disappears_before_division': 0,
    'disappears_before_division': 0
}
cca_df_colnames = list(base_cca_dict.keys())

base_cca_tree_dict = {
    'Cell_ID_tree': -1,
    'generation_num_tree': 1,
    'parent_ID_tree': -1,
    'root_ID_tree': -1,
    'sister_ID_tree': -1
}

lineage_tree_cols = list(base_cca_tree_dict.keys())

# lineage_tree_cols = [
#     # 'Cell_ID_tree',
#     'generation_num_tree',
#     'parent_ID_tree',
#     'root_ID_tree',
#     'sister_ID_tree'
# ]

lineage_tree_cols_std_val = [
    -1,
    -1,
    -1,
    -1,
    -1
]

default_annot_df = {
    'is_cell_dead': False,
    'is_cell_excluded': False,
}

base_acdc_df = {
    **default_annot_df,
    'was_manually_edited': 0
}

base_acdc_df_cols = list(base_acdc_df.keys())

sorted_cols = ['time_seconds', 'time_minutes', 'time_hours']
sorted_cols = [
    *sorted_cols, *cca_df_colnames, *lineage_tree_cols, *base_acdc_df_cols
]

cca_df_colnames_with_tree = [*cca_df_colnames, *lineage_tree_cols]

all_non_metrics_cols = [*base_acdc_df_cols, *cca_df_colnames, *lineage_tree_cols]

is_linux = sys.platform.startswith('linux')
is_mac = sys.platform == 'darwin'
is_win = sys.platform.startswith("win")
is_win64 = (is_win and (os.environ["PROCESSOR_ARCHITECTURE"] == "AMD64"))
is_mac_arm64 = is_mac and platform.machine() == 'arm64'

yeaz_weights_filenames = [
    'unet_weights_batchsize_25_Nepochs_100_SJR0_10.hdf5',
    'weights_budding_BF_multilab_0_1.hdf5'
]

yeaz_v2_weights_filenames = [
    'weights_budding_BF_multilab_0_1',
    'weights_budding_PhC_multilab_0_1',
    'weights_fission_multilab_0_2'
]

segment_anything_weights_filenames = [
    'sam_vit_h_4b8939.pth', 
    'sam_vit_l_0b3195.pth', 
    'sam_vit_b_01ec64.pth'
]

deepsea_weights_filenames = [
    'segmentation.pth', 
    'tracker.pth'
]

yeastmate_weights_filenames = [
    'yeastmate_advanced.yaml',
    'yeastmate_weights.pth',
    'yeastmate.yaml'
]

tapir_weights_filenames = [
    'tapir_checkpoint.npy'
]

graphLayoutBkgrColor = (235, 235, 235)
darkBkgrColor = [255-v for v in graphLayoutBkgrColor]

def _critical_exception_gui(self, func_name):
    from . import widgets, html_utils
    result = None
    traceback_str = traceback.format_exc()
    
    if hasattr(self, 'is_error_state') and self.is_error_state:
        printl(traceback_str)
        return
    
    if hasattr(self, 'logger'):
        self.logger.exception(traceback_str)
    else:
        printl(traceback_str)
    
    try:
        self.cleanUpOnError()
    except Exception as e:
        pass
    
    msg = widgets.myMessageBox(wrapText=False, showCentered=False)
    if hasattr(self, 'logs_path'):
        msg.addShowInFileManagerButton(
            self.logs_path, txt='Show log file...'
        )
    if not hasattr(self, 'log_path'):
        log_path = 'NULL'
    else:
        log_path = self.log_path
    
    self.is_error_state = True
    msg.setDetailedText(traceback_str, visible=True)
    href = f'<a href="{issues_url}">GitHub page</a>'
    err_msg = html_utils.paragraph(f"""
        Error in function <code>{func_name}</code>.<br><br>
        More details below or in the terminal/console.<br><br>
        You can <b>report</b> this error by opening an issue
        on our {href}.<br><br>
        Please <b>send the log file</b> when reporting the error, thanks!<br><br>
        NOTE: the <b>log file</b> with the <b>error details</b> can be found 
        here:
    """)

    msg.critical(self, 'Critical error', err_msg, commands=(log_path,))
    
def exception_handler_cli(func):
    @wraps(func)
    def inner_function(self, *args, **kwargs):
        try:
            if func.__code__.co_argcount==1 and func.__defaults__ is None:
                result = func(self)
            elif func.__code__.co_argcount>1 and func.__defaults__ is None:
                result = func(self, *args)
            else:
                result = func(self, *args, **kwargs)
        except Exception as err:
            result = None
            if self.is_cli:
                self.quit(error=err)
            else:
                raise err
        return result
    return inner_function

def exception_handler(func):
    @wraps(func)
    def inner_function(self, *args, **kwargs):
        try:
            if func.__code__.co_argcount==1 and func.__defaults__ is None:
                result = func(self)
            elif func.__code__.co_argcount>1 and func.__defaults__ is None:
                result = func(self, *args)
            else:
                result = func(self, *args, **kwargs)
        except Exception as e:
            try:
                if self.progressWin is not None:
                    self.progressWin.workerFinished = True
                    self.progressWin.close()
            except AttributeError:
                pass
            result = _critical_exception_gui(self, func.__name__)
        return result
    return inner_function

def ignore_exception(func):
    @wraps(func)
    def inner_function(self, *args, **kwargs):
        try:
            if func.__code__.co_argcount==1 and func.__defaults__ is None:
                result = func(self)
            elif func.__code__.co_argcount>1 and func.__defaults__ is None:
                result = func(self, *args)
            else:
                result = func(self, *args, **kwargs)
        except Exception as e:
            pass
        return result
    return inner_function

def is_conda_env():
    python_exec_path = sys.exec_prefix
    is_conda_python = (
        python_exec_path.find('conda') != -1
        or python_exec_path.find('mambaforge') != -1
        or python_exec_path.find('miniforge') != -1
    )
    if not is_conda_python:
        return False
    
    stdout = subprocess.DEVNULL
    try:
        args = ['conda', '-V']
        is_conda_present = subprocess.check_call(
            args, shell=True, stdout=stdout) == 0
        return True
    except Exception as err:
        pass
    
    try:
        args = ['conda -V']
        is_conda_present = subprocess.check_call(
            args, shell=True, stdout=stdout) == 0
        return True
    except Exception as err:
        return False
    
    return True

error_below = f"\n{'*'*50} ERROR {'*'*50}\n"
error_close = f"\n{'^'*(len(error_below)-1)}"

error_up_str = '^'*100
error_up_str = f'\n{error_up_str}'
error_down_str = '^'*100
error_down_str = f'\n{error_down_str}'

pytorch_commands = {
    'Windows': {
        'Conda': {
            'CPU': 'conda install pytorch torchvision cpuonly -c pytorch',
            'CUDA 11.8 (NVIDIA GPU)': 'conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia',
            'CUDA 12.1 (NVIDIA GPU)': 'conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia'
        },
        'Pip': {
            'CPU': 'python -m pip install torch torchvision',
            'CUDA 11.8 (NVIDIA GPU)': 'python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118',
            'CUDA 12.1 (NVIDIA GPU)': 'python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121'
        }
    },
    'Mac': {
        'Conda': {
            'CPU': 'conda install pytorch torchvision cpuonly -c pytorch',
            'CUDA 11.8 (NVIDIA GPU)': '[WARNING]: CUDA is not available on MacOS',
            'CUDA 12.1 (NVIDIA GPU)': '[WARNING]: CUDA is not available on MacOS'
        },
        'Pip': {
            'CPU': 'pip3 install torch torchvision',
            'CUDA 11.8 (NVIDIA GPU)': '[WARNING]: CUDA is not available on MacOS',
            'CUDA 12.1 (NVIDIA GPU)': '[WARNING]: CUDA is not available on MacOS'
        }
    },
    'Linux': {
        'Conda': {
            'CPU': 'conda install pytorch torchvision cpuonly -c pytorch',
            'CUDA 11.8 (NVIDIA GPU)': 'conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia',
            'CUDA 12.1 (NVIDIA GPU)': 'conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia'
        },
        'Pip': {
            'CPU': 'pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu',
            'CUDA 11.8 (NVIDIA GPU)': 'pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118',
            'CUDA 12.1 (NVIDIA GPU)': 'pip3 install torch torchvision'
        }
    }
}