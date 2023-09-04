print('Initialising...')
import os
import sys
import shutil

import pathlib
import numpy as np

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
qrc_resources_path = os.path.join(cellacdc_path, 'qrc_resources.py')
qrc_resources_light_path = os.path.join(cellacdc_path, 'qrc_resources_light.py')
qrc_resources_dark_path = os.path.join(cellacdc_path, 'qrc_resources_dark.py')
old_temp_path = os.path.join(cellacdc_path, 'temp')

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
    
if not os.path.exists(settings_folderpath):
    os.makedirs(settings_folderpath, exist_ok=True)
if os.path.exists(old_temp_path):
    try:
        from distutils.dir_util import copy_tree
        copy_tree(old_temp_path, settings_folderpath)
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
    import seaborn
    GUI_INSTALLED = True
except Exception as e:
    GUI_INSTALLED = False

import pandas as pd

np.random.seed(3548784512)

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 200)
pd.set_option('display.expand_frame_repr', False)

def printl(*objects, pretty=False, is_decorator=False, **kwargs):
    # Copy current stdout, reset to default __stdout__ and then restore current
    current_stdout = sys.stdout
    sys.stdout = sys.__stdout__
    timestap = datetime.now().strftime('%H:%M:%S')
    currentframe = inspect.currentframe()
    outerframes = inspect.getouterframes(currentframe)
    idx = 2 if is_decorator else 1
    callingframe = outerframes[idx].frame
    callingframe_info = inspect.getframeinfo(callingframe)
    filpath = callingframe_info.filename
    filename = os.path.basename(filpath)
    print_func = pprint if pretty else print
    print('*'*30)
    print(f'{timestap} - File "{filename}", line {callingframe_info.lineno}:')
    if 'sep' not in kwargs:
        kwargs['sep'] = ', '
    if pretty:
        del kwargs['sep']
    print_func(*objects, **kwargs)
    print('='*30)
    sys.stdout = current_stdout

parent_path = os.path.dirname(cellacdc_path)
html_path = os.path.join(cellacdc_path, '_html')
models_path = os.path.join(cellacdc_path, 'models')
data_path = os.path.join(parent_path, 'data')
resources_folderpath = os.path.join(cellacdc_path, 'resources')
resources_filepath = os.path.join(cellacdc_path, 'resources_light.qrc')
logs_path = os.path.join(user_profile_path, '.acdc-logs')
resources_path = os.path.join(cellacdc_path, 'resources_light.qrc')
models_list_file_path = os.path.join(settings_folderpath, 'custom_models_paths.ini')
recentPaths_path = os.path.join(settings_folderpath, 'recentPaths.csv')
user_manual_url = 'https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf'
github_home_url = 'https://github.com/SchmollerLab/Cell_ACDC'

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
base_cca_df = {
    'cell_cycle_stage': 'G1',
    'generation_num': 2,
    'relative_ID': -1,
    'relationship': 'mother',
    'emerg_frame_i': -1,
    'division_frame_i': -1,
    'is_history_known': False,
    'corrected_assignment': False,
    'will_divide': 0
}

lineage_tree_cols = [
    'Cell_ID_tree',
    'generation_num_tree',
    'parent_ID_tree',
    'root_ID_tree',
    'sister_ID_tree'
]

base_acdc_df = {
    'is_cell_dead': False,
    'is_cell_excluded': False,
    'was_manually_edited': 0
}

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
    msg.setDetailedText(traceback_str, visible=True)
    href = f'<a href="{issues_url}">GitHub page</a>'
    err_msg = html_utils.paragraph(f"""
        Error in function <code>{func_name}</code>.<br><br>
        More details below or in the terminal/console.<br><br>
        Note that the <b>error details</b> from this session are
        also <b>saved in the following log file</b>:
        <br><br>
        <code>{log_path}</code>
        <br><br>
        You can <b>report</b> this error by opening an issue
        on our {href}.<br><br>
        Please <b>send the log file</b> when reporting a bug, thanks!
    """)

    msg.critical(self, 'Critical error', err_msg)
    self.is_error_state = True

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

error_below = f"\n{'*'*30} ERROR {'*'*30}\n"
error_close = f"\n{'^'*(len(error_below)-1)}"

error_up_str = '^'*50
error_up_str = f'\n{error_up_str}'
error_down_str = '^'*50
error_down_str = f'\n{error_down_str}'