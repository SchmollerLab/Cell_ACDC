import os
import re
import pathlib
import difflib
import sys
import platform
import tempfile
import shutil
import traceback
import logging
import datetime
import time
import subprocess
import importlib
from uuid import uuid4
from importlib import import_module
from math import pow, ceil, floor
from functools import wraps, partial
from collections import namedtuple, Counter
from tqdm import tqdm
import requests
import zipfile
import numpy as np
import pandas as pd
import skimage
import inspect
import typing
from typing import List

from natsort import natsorted

import tifffile
import skimage.io
import skimage.measure

from . import GUI_INSTALLED, KNOWN_EXTENSIONS, is_conda_env

if GUI_INSTALLED:
    from qtpy.QtWidgets import QMessageBox
    from qtpy.QtCore import Signal, QObject, QCoreApplication
    
    from . import widgets, apps
    from . import config
    
from . import core, load
from . import html_utils, is_linux, is_win, is_mac, issues_url, is_mac_arm64
from . import cellacdc_path, printl, acdc_fiji_path, logs_path, acdc_ffmpeg_path
from . import user_profile_path, recentPaths_path
from . import models_list_file_path
from . import github_home_url
from . import try_input_install_package
from . import _warnings
from . import urls

ArgSpec = namedtuple('ArgSpec', ['name', 'default', 'type', 'desc', 'docstring'])

def get_module_name(script_file_path):
    parts = pathlib.Path(script_file_path).parts
    parts = list(parts[parts.index('cellacdc')+1:])
    parts[-1] = os.path.splitext(parts[-1])[0]
    module = '.'.join(parts)
    return module

def get_pos_status(pos_path):
    images_path = os.path.join(pos_path, 'Images')
    ls = listdir(images_path)
    for file in ls:
        if file.endswith('acdc_output.csv'):
            acdc_df_path = os.path.join(images_path, file)
            break
    else:
        return ''
    
    acdc_df = pd.read_csv(acdc_df_path)
    last_tracked_i = acdc_df['frame_i'].max()
    last_cca_i = 0
    if 'cell_cycle_stage' in acdc_df.columns:
        cca_df = acdc_df[['frame_i', 'cell_cycle_stage']].dropna()
        last_cca_i = cca_df['frame_i'].max()
    if last_cca_i > 0:
        return (
            f' (last tracked frame = {last_tracked_i+1}, '
            f'last annotated frame = {last_cca_i+1})'
        )
    else:
        return f' (last tracked frame = {last_tracked_i+1})'

def get_gdrive_path():
    if is_win:
        return os.path.join(f'G:{os.sep}', 'My Drive')
    elif is_mac:
        return os.path.join(
            '/Users/francesco.padovani/Library/CloudStorage/'
            'GoogleDrive-padovaf@tcd.ie/My Drive'
        )

def get_acdc_data_path():
    Cell_ACDC_path = os.path.dirname(cellacdc_path)
    return os.path.join(Cell_ACDC_path, 'data')

def get_open_filemaneger_os_string():
    if is_win:
        return 'Show in Explorer...'
    elif is_mac:
        return 'Reveal in Finder...'
    elif is_linux:
        return 'Show in File Manager...'

def filterCommonStart(images_path):
    startNameLen = 6
    ls = listdir(images_path)
    if not ls:
        return []
    allFilesStartNames = [f[:startNameLen] for f in ls]
    mostCommonStart = Counter(allFilesStartNames).most_common(1)[0][0]
    commonStartFilenames = [f for f in ls if f.startswith(mostCommonStart)]
    return commonStartFilenames

def get_salute_string():
    time_now = datetime.datetime.now().time()
    time_end_morning = datetime.time(12,00,00)
    time_end_lunch = datetime.time(13,00,00)
    time_end_afternoon = datetime.time(15,00,00)
    time_end_evening = datetime.time(20,00,00)
    time_end_night = datetime.time(4,00,00)
    if time_now >= time_end_night and time_now < time_end_morning:
        return 'Have a good day!'
    elif time_now >= time_end_morning and time_now < time_end_lunch:
        return 'Enjoy your lunch!'
    elif time_now >= time_end_lunch and time_now < time_end_afternoon:
        return 'Have a good afternoon!'
    elif time_now >= time_end_afternoon and time_now < time_end_evening:
        return 'Have a good evening!'
    else:
        return 'Have a good night!'

def remove_known_extension(name):
    for ext in KNOWN_EXTENSIONS:
        if name.endswith(ext):
            return name[:-len(ext)], ext

    return name, ''
    
def getCustomAnnotTooltip(annotState):
    toolTip = (
        f'Name: {annotState["name"]}\n\n'
        f'Type: {annotState["type"]}\n\n'
        f'Usage: activate the button and RIGHT-CLICK on cell to annotate\n\n'
        f'Description: {annotState["description"]}\n\n'
        f'SHORTCUT: "{annotState["shortcut"]}"'
    )
    return toolTip

def trim_path(path, depth=3, start_with_dots=True):
    path_li = os.path.abspath(path).split(os.sep)
    rel_path = f'{f"{os.sep}".join(path_li[-depth:])}'
    if start_with_dots:
        return f'...{os.sep}{rel_path}'
    else:
        return rel_path

def get_add_custom_model_instructions():
    url = 'https://github.com/SchmollerLab/Cell_ACDC/issues'
    user_manual_url = 'https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf'
    href_user_manual = f'<a href="{user_manual_url}">user manual</a>'
    href = f'<a href="{url}">here</a>'
    models_path = os.path.join(cellacdc_path, 'models')
    func_color = (111/255,66/255,205/255) # purplish
    kwargs_color = (208/255,88/255,9/255) # reddish/orange
    class_color = (215/255,58/255,73/255) # reddish
    blue_color = (0/255,92/255,197/255) # blueish
    class_sh = html_utils.class_sh
    def_sh = html_utils.def_sh
    kwargs_sh = html_utils.kwargs_sh
    Model_sh = html_utils.Model_sh
    segment_sh = html_utils.segment_sh
    predict_sh = html_utils.predict_sh
    init_sh = html_utils.init_sh
    myModel_sh = html_utils.myModel_sh
    return_sh = html_utils.return_sh
    equal_sh = html_utils.equal_sh
    open_par_sh = html_utils.open_par_sh
    close_par_sh = html_utils.close_par_sh
    image_sh = html_utils.image_sh
    from_sh = html_utils.from_sh
    import_sh = html_utils.import_sh
    s = html_utils.paragraph(f"""
    To use a custom model first <b>create a folder</b> with the same name of your model.<br><br>
    Inside this new folder create a file named <code>acdcSegment.py</code>.<br><br>
    In the <code>acdcSegment.py</code> file you will <b>implement the model class</b>.<br><br>
    Have a look at the other existing models, but essentially you have to create
    a class called <code>Model</code> with at least<br>
    the <code>{init_sh}</code> and the <code>{segment_sh}</code> method.<br><br>
    The <code>{segment_sh}</code> method takes the image (2D or 3D) as an input and return the segmentation mask.<br><br>
    You can find more details in the {href_user_manual} at the section
    called <code>Adding segmentation models to the pipeline</code>.<br><br>
    Pseudo-code for the <code>acdcSegment.py</code> file:
    <pre><code>
    {from_sh} myModel {import_sh} {myModel_sh}

    {class_sh} {Model_sh}:
        {def_sh} {init_sh}(self, {kwargs_sh}):
            self.model {equal_sh} {myModel_sh}{open_par_sh}{close_par_sh}

        {def_sh} {segment_sh}(self, {image_sh}, {kwargs_sh}):
            labels {equal_sh} self.model.{predict_sh}{open_par_sh}{image_sh}{close_par_sh}
            {return_sh} labels
    </code></pre>
    
    If it doesn't work, please report the issue {href} with the
    code you wrote. Thanks.
    """)
    return s, models_path

def is_iterable(item):
     try:
         iter(item)
         return True
     except TypeError as e:
         return False

class utilClass:
    pass

def get_trimmed_list(li: list, max_num_digits=10):
    li_str = li.copy()
    tom_num_digits = sum([len(str(val)) for val in li])
    avg_num_digits = tom_num_digits/len(li)
    max_num_vals = int(round(max_num_digits/avg_num_digits))
    if tom_num_digits>max_num_digits:
        del li_str[max_num_vals:-max_num_vals]
        li_str.insert(max_num_vals, "...")
        li_str = f"[{', '.join(map(str, li_str))}]"
    return li_str

def get_trimmed_dict(di: dict, max_num_digits=10):
    di_str = di.copy()
    total_num_digits = sum([len(str(key)) + len(str(val)) for key, val in di.items()])
    avg_num_digits = total_num_digits / len(di)
    max_num_vals = int(round(max_num_digits / avg_num_digits))
    if total_num_digits > max_num_digits:
        keys = list(di_str.keys())
        for key in keys[max_num_vals:-max_num_vals]:
            del di_str[key]
        di_str[keys[max_num_vals]] = "..."
    return f"[{', '.join([f'{key} -> {val}' for key, val in di_str.items()])}]"

def checked_reset_index(df):
    if df.index.names is None or df.index.names == [None]:
        return df.reset_index(drop=True)
    else:
        return df.reset_index()


def _bytes_to_MB(size_bytes):
    factor = pow(2, -20)
    size_MB = round(size_bytes*factor)
    return size_MB

def _bytes_to_GB(size_bytes):
    factor = pow(2, -30)
    size_GB = round(size_bytes*factor, 2)
    return size_GB

def getMemoryFootprint(files_list):
    required_memory = sum([
        48 if file.endswith('.h5') else os.path.getsize(file)
        for file in files_list
    ])
    return required_memory

def get_logs_path():
    return logs_path

class Logger(logging.Logger):
    def __init__(
            self,
            module='base', 
            name='cellacdc-logger', 
            level=logging.DEBUG
        ):
        super().__init__(f'{name}-{module}', level=level)
        self._stdout = sys.stdout
    
    def write(self, text, log_to_file=True):
        """Capture print statements, print to terminal and log text to 
        the open log file

        Parameters
        ----------
        text : str
            Text to log
        log_to_file : bool, optional
            If True, call `info` method with `text`. Default is True
        """        
        self._stdout.write(text)
        if not log_to_file:
            return
        
        if text == '\n':
            return
        
        if not text:
            return 
        
        self.debug(text)

    def close(self):
        for handler in self.handlers:
            handler.close()
            self.removeHandler(handler)
        sys.stdout = self._stdout
    
    def __del__(self):
        sys.stdout = self._stdout
    
    def info(self, text, *args, **kwargs):
        super().info(text, *args, **kwargs)
        self.write(f'{text}\n', log_to_file=False)
    
    def warning(self, text, *args, **kwargs):
        super().warning(text, *args, **kwargs)
        self.write(f'[WARNING]: {text}\n', log_to_file=False)
    
    def error(self, text, *args, **kwargs):
        super().error(text, *args, **kwargs)
        self.write(traceback.format_exc())
        self.write(f'[ERROR]: {text}\n', log_to_file=False)
    
    def critical(self, text, *args, **kwargs):
        super().critical(text, *args, **kwargs)
        self.write(f'[CRITICAL]: {text}\n', log_to_file=False)
    
    def exception(self, text, *args, **kwargs):
        super().exception(text, *args, **kwargs)
        self.write(traceback.format_exc())
        self.write(f'[ERROR]: {text}\n', log_to_file=False)
    
    def log(self, level, text):
        super().log(level, text)
        levelName = logging.getLevelName(level)
        getattr(self, levelName.lower())(text)
        # self.write(f'[{levelName}]: {text}\n', log_to_file=False)
    
    def flush(self):
        self._stdout.flush()

def delete_older_log_files(logs_path):
    if not os.path.exists(logs_path):
        return
    
    log_files = os.listdir(logs_path)
    for log_file in log_files:
        if not log_file.endswith('.log'):
            continue
        
        log_filepath = os.path.join(logs_path, log_file)
        try:
            mtime = os.path.getmtime(log_filepath)
        except Exception as err:
            continue
        
        mdatetime = datetime.datetime.fromtimestamp(mtime)
        days = (datetime.datetime.now() - mdatetime).days
        if days < 7:
            continue

        try:
            os.remove(log_filepath)
        except Exception as err:
            continue

def _log_system_info(logger, log_path, is_cli=False, also_spotmax=False):
    logger.info(f'Initialized log file "{log_path}"')
    
    py_ver = sys.version_info
    python_version = f'{py_ver.major}.{py_ver.minor}.{py_ver.micro}'
    logger.info(f'Running Python v{python_version} from "{sys.exec_prefix}"')    
    logger.info(f'Cell-ACDC installation directory: "{cellacdc_path}"')
    logger.info(f'System version: {sys.version}')
    logger.info(f'Platform: {platform.platform()}')
    
    if GUI_INSTALLED and not is_cli:
        from qtpy import QtCore
        logger.info(f'Using Qt version {QtCore.__version__}')
    
    if not also_spotmax:
        return
    
    from spotmax import spotmax_path
    logger.info(f'SpotMAX installation directory: "{spotmax_path}"')

def setupLogger(module='base', logs_path=None):
    if logs_path is None:
        logs_path = get_logs_path()
    
    logger = Logger(module=module)
    sys.stdout = logger
    
    delete_older_log_files(logs_path)
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)

    date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    id = uuid4()
    log_filename = f'{date_time}_{module}_{id}_stdout.log'
    log_path = os.path.join(logs_path, log_filename)

    output_file_handler = logging.FileHandler(log_path, mode='w')

    # Format your logs (optional)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s:\n'
        '------------------------\n'
        '%(message)s\n'
        '------------------------\n',
        datefmt='%d-%m-%Y, %H:%M:%S'
    )
    output_file_handler.setFormatter(formatter)

    logger.addHandler(output_file_handler)
    
    _log_system_info(logger, log_path)
    
    # if module == 'gui' and GUI_INSTALLED:
    #     qt_handler = widgets.QtHandler()
    #     qt_handler.setFormatter(logging.Formatter("%(message)s"))
    #     logger.addHandler(qt_handler)

    return logger, logs_path, log_path, log_filename

def get_pos_foldernames(exp_path, check_if_is_sub_folder=False):
    if not check_if_is_sub_folder:
        ls = listdir(exp_path)
        pos_foldernames = [
            pos for pos in ls if is_pos_folderpath(os.path.join(exp_path, pos))
        ]
    else:
        folder_type = determine_folder_type(exp_path)
        is_pos_folder, is_images_folder, _ = folder_type
        if is_pos_folder:
            return [os.path.basename(exp_path)]
        elif is_images_folder:
            pos_path = os.path.dirname(exp_path)
            return [os.path.basename(pos_path)]
        else:
            return get_pos_foldernames(exp_path)
    return pos_foldernames

def getMostRecentPath():
    if os.path.exists(recentPaths_path):
        df = pd.read_csv(recentPaths_path, index_col='index')
        if 'opened_last_on' in df.columns:
            df = df.sort_values('opened_last_on', ascending=False)
        MostRecentPath = ''
        for path in df['path']:
            if os.path.exists(path):
                MostRecentPath = path
                break
    else:
        MostRecentPath = ''
    return MostRecentPath

def addToRecentPaths(exp_path, logger=None):
    if not os.path.exists(exp_path):
        return
    exp_path = exp_path.replace('\\', '/')
    if os.path.exists(recentPaths_path):
        try:
            df = pd.read_csv(recentPaths_path, index_col='index')
            recentPaths = df['path'].to_list()
            if 'opened_last_on' in df.columns:
                openedOn = df['opened_last_on'].to_list()
            else:
                openedOn = [np.nan]*len(recentPaths)
            if exp_path in recentPaths:
                pop_idx = recentPaths.index(exp_path)
                recentPaths.pop(pop_idx)
                openedOn.pop(pop_idx)
            recentPaths.insert(0, exp_path)
            openedOn.insert(0, datetime.datetime.now())
            # Keep max 40 recent paths
            if len(recentPaths) > 40:
                recentPaths.pop(-1)
                openedOn.pop(-1)
        except Exception as e:
            recentPaths = [exp_path]
            openedOn = [datetime.datetime.now()]
    else:
        recentPaths = [exp_path]
        openedOn = [datetime.datetime.now()]
    df = pd.DataFrame({
        'path': recentPaths,
        'opened_last_on': pd.Series(openedOn, dtype='datetime64[ns]')}
    )
    df.index.name = 'index'
    df.to_csv(recentPaths_path)

def checkDataIntegrity(filenames, parent_path, parentQWidget=None):
    char = filenames[0][:2]
    startWithSameChar = all([f.startswith(char) for f in filenames])
    if not startWithSameChar:
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            'Cell-ACDC detected files inside the folder '
            'that <b>do not start with the same, common basename</b>.<br><br>'
            'To ensure correct loading of the data, the folder where '
            'the file(s) is/are should either contain a single image file or'
            'only files that start with the same, common basename.<br><br>'
            'For example the following filenames:<br><br>'
            '<code>F014_s01_phase_contr.tif</code><br>'
            '<code>F014_s01_mCitrine.tif</code><br><br>'
            'are named correctly since they all start with the '
            'the common basename "F014_s01_". After the common basename you '
            'can write whatever text you want. In the example above, "phase_contr" '
            'and "mCitrine" are the channel names.<br><br>'
            'Data loading may still be successfull, so Cell-ACDC will '
            'still try to load data now.<br>'
        )
        filesFormat = [f'    - {file}' for file in filenames]
        filesFormat = "\n".join(filesFormat)
        detailsText = (
            f'Files present in the folder {parent_path}:\n\n'
            f'{filesFormat}'
        )
        msg.addShowInFileManagerButton(parent_path, txt='Open folder...')
        msg.warning(
            parentQWidget, 'Data structure compromised', txt, 
            detailsText=detailsText, buttonsTexts=('Cancel', 'Ok')
        )
        if msg.cancel:
            raise TypeError(
                'Process aborted by the user.'
            )
        return False
    return True

def get_cca_colname_desc():
    desc = {
        'Cell ID': (
            'ID of the segmented cell. All of the other columns '
            'are properties of this ID.'
        ),
        'Cell cycle stage': (
            'G1 if the cell does NOT have a bud. S/G2/M if it does.'
        ),
        'Relative ID': (
            'ID of the bud related to the Cell ID (row). For cells in G1 write the '
            'bud ID it had in the previous cycle.'
        ),
        'Generation number': (
            'Number of times the cell divided from a bud. For cells in the first '
            'frame write any number greater than 1.'
        ),
        'Relationship': (
            'Relationship of the current Cell ID (row). '
            'Either <b>mother</b> or <b>bud</b>. An object is a bud if '
            'it didn\'t divide from the mother yet. All other instances '
            '(e.g., cell in G1) are still labelled as mother.'
        ),
        'Emerging frame num.': (
            'Frame number at which the object emerged/appeared in the scene.'
        ),
        'Division frame num.': (
            'Frame number at which the bud separated from the mother.'
        ),
        'Is history known?': (
            'Cells that are already present in the first frame or appears '
            'from outside of the field of view, have some information missing. '
            'For example, for cells in the first frame we do not know how many '
            'times it budded and divided in the past. '
            'In these cases Is history known? is True.'
        )
    }
    return desc

def testQcoreApp():
    print(QCoreApplication.instance())

def store_custom_model_path(model_file_path):
    model_file_path = model_file_path.replace('\\', '/')
    model_name = os.path.basename(os.path.dirname(model_file_path))
    cp = config.ConfigParser()
    if os.path.exists(models_list_file_path):
        cp.read(models_list_file_path)
    if model_name not in cp:
        cp[model_name] = {}
    cp[model_name]['path'] = model_file_path
    with open(models_list_file_path, 'w') as configFile:
        cp.write(configFile)

def check_git_installed(parent=None):
    try:
        subprocess.check_call(['git', '--version'], shell=True)
        return True
    except Exception as e:
        print('='*20)
        traceback.print_exc()
        print('='*20)
        git_url = 'https://git-scm.com/book/en/v2/Getting-Started-Installing-Git'
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(f"""
            In order to install <code>javabridge</code> you first need to <b>install
            Git</b> (it was not found).<br><br>
            <b>Close Cell-ACDC</b> and follow the instructions
            {html_utils.tag('here', f'a href="{git_url}"')}.<br><br>
            <i><b>NOTE</b>: After installing Git you might need to <b>restart the
            terminal</b></i>.
        """)
        msg.warning(
            parent, 'Git not installed', txt
        )
        return False

def browse_url(url):
    import webbrowser
    webbrowser.open(url)

def browse_docs():
    browse_url(urls.docs_homepage)

def install_java():
    try:
        subprocess.check_call(['javac', '-version'], shell=True)
        return False
    except Exception as e:
        from . import widgets
        win = widgets.installJavaDialog()
        win.exec_()
        return win.clickedButton == win.cancelButton

def install_javabridge(force_compile=False, attempt_uninstall_first=False):
    if attempt_uninstall_first:
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'uninstall', '-y', 'javabridge']
            )
        except Exception as e:
            pass
    if sys.platform.startswith('win'):
        if force_compile:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-U',
                'git+https://github.com/SchmollerLab/python-javabridge-acdc']
            )
        else:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-U',
                'git+https://github.com/SchmollerLab/python-javabridge-windows']
            )
    elif is_mac:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-U',
            'git+https://github.com/SchmollerLab/python-javabridge-acdc']
        )
    elif is_linux:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '-U',
            'git+https://github.com/LeeKamentsky/python-javabridge.git@master']
        )

def is_in_bounds(x,y,X,Y):
    in_bounds = x >= 0 and x < X and y >= 0 and y < Y
    return in_bounds

def read_version(logger=None, return_success=False):
    cellacdc_parent_path = os.path.dirname(cellacdc_path)
    cellacdc_parent_folder = os.path.basename(cellacdc_parent_path)
    if cellacdc_parent_folder == 'site-packages':
        from . import __version__
        version = __version__
        success = True
    else:
        try:
            from setuptools_scm import get_version
            version = get_version(root='..', relative_to=__file__)
            success = True
        except Exception as e:
            if logger is None:
                logger = print
            logger('*'*40)
            logger(traceback.format_exc())
            logger('-'*40)
            logger(
                '[WARNING]: Cell-ACDC could not determine the current version. '
                'Returning the version determined at installation time. '
                'See details above.'
            )
            logger('='*40)
            try:
                from . import _version
                version = _version.version
                success = False
            except Exception as e:
                version = 'ND'
                success = False
    
    if return_success:
        return version, success
    else:
        return version

def get_date_from_version(version: str, package='cellacdc'):
    try:
        res_json = requests.get(f'https://pypi.org/pypi/{package}/json').json()
        pypi_releases_json = res_json['releases']
        version_json = pypi_releases_json[version][0]
        upload_time = version_json['upload_time_iso_8601']
        date = datetime.datetime.strptime(upload_time, r'%Y-%m-%dT%H:%M:%S.%fZ')
        date_str = date.strftime(r'%A %d %B %Y at %H:%M')
        return date_str
    except Exception as err:
        pass
    
    try:
        commit_hash = re.findall(r'\+g([A-Za-z0-9]+)\.d', version)[0]
        commands = ['git', 'show', commit_hash]
        commit_log = subprocess.check_output(commands).decode() 
        date_log = re.findall(r'Date:(.*) \+', commit_log)[0].strip()
        date = datetime.datetime.strptime(date_log, r'%a %b %d %H:%M:%S %Y')
        date_str = date.strftime(r'%A %d %B %Y at %H:%M')
        return date_str
    except Exception as err:
        pass
    
    return 'ND'  

def showInExplorer(path):
    if is_mac:
        os.system(f'open "{path}"')
    elif is_linux:
        os.system(f'xdg-open "{path}"')
    else:
        os.startfile(path)

def exec_time(func):
    @wraps(func)
    def inner_function(self, *args, **kwargs):
        t0 = time.perf_counter()
        if func.__code__.co_argcount==1 and func.__defaults__ is None:
            result = func(self)
        elif func.__code__.co_argcount>1 and func.__defaults__ is None:
            result = func(self, *args)
        else:
            result = func(self, *args, **kwargs)
        t1 = time.perf_counter()
        s = f'{func.__name__} execution time = {(t1-t0)*1000:.3f} ms'
        printl(s, is_decorator=True)
        return result
    return inner_function

def setRetainSizePolicy(widget, retain=True):
    sp = widget.sizePolicy()
    sp.setRetainSizeWhenHidden(retain)
    widget.setSizePolicy(sp)

def getAcdcDfSegmPaths(images_path):
    ls = listdir(images_path)
    basename = getBasename(ls)
    paths = {}
    for file in ls:
        filePath = os.path.join(images_path, file)
        fileName, ext = os.path.splitext(file)
        endName = fileName[len(basename):]
        if endName.find('acdc_output') != -1 and ext=='.csv':
            info_name = endName.replace('acdc_output', '')
            paths.setdefault(info_name, {})
            paths[info_name]['acdc_df_path'] = filePath
            paths[info_name]['acdc_df_filename'] = fileName
        elif endName.find('segm') != -1 and ext=='.npz':
            info_name = endName.replace('segm', '')
            paths.setdefault(info_name, {})
            paths[info_name]['segm_path'] = filePath
            paths[info_name]['segm_filename'] = fileName
    return paths

def getChannelFilePath(images_path, chName):
    file = ''
    alignedFilePath = ''
    tifFilePath = ''
    h5FilePath = ''
    for file in listdir(images_path):
        filePath = os.path.join(images_path, file)
        if file.endswith(f'{chName}_aligned.npz'):
            alignedFilePath = filePath
        elif file.endswith(f'{chName}.tif'):
            tifFilePath = filePath
        elif file.endswith(f'{chName}.h5'):
            h5FilePath = filePath
    if alignedFilePath:
        return alignedFilePath
    elif h5FilePath:
        return h5FilePath
    elif tifFilePath:
        return tifFilePath
    else:
        return ''

def get_number_fstring_formatter(dtype, precision=4):
    if np.issubdtype(dtype, np.integer):
        return 'd'
    else:
        return f'.{precision}f'

def get_chname_from_basename(filename, basename, remove_ext=True):
    if remove_ext:
        filename, ext = os.path.splitext(filename)
    chName = filename[len(basename):]
    aligned_idx = chName.find('_aligned')
    if aligned_idx != -1:
        chName = chName[:aligned_idx]
    return chName

def getBaseAcdcDf(rp):
    zeros_list = [0]*len(rp)
    nones_list = [None]*len(rp)
    minus1_list = [-1]*len(rp)
    IDs = []
    xx_centroid = []
    yy_centroid = []
    zz_centroid = []
    for obj in rp:
        xc, yc = obj.centroid[-2:]
        IDs.append(obj.label)
        xx_centroid.append(xc)
        yy_centroid.append(yc)
        if len(obj.centroid) == 3:
            zc = obj.centroid[0]
            zz_centroid.append(zc)
            
    df = pd.DataFrame(
        {
            'Cell_ID': IDs,
            'is_cell_dead': zeros_list,
            'is_cell_excluded': zeros_list,
            'x_centroid': xx_centroid,
            'y_centroid': yy_centroid,
            'was_manually_edited': minus1_list
        }
    ).set_index('Cell_ID')
    if zz_centroid:
        df['z_centroid'] = zz_centroid
        
    return df

def getBasenameAndChNames(images_path, useExt=None):
    _tempPosData = utilClass()
    _tempPosData.images_path = images_path
    load.loadData.getBasenameAndChNames(_tempPosData, useExt=useExt)
    return _tempPosData.basename, _tempPosData.chNames

def getBasename(files):
    basename = files[0]
    for file in files:
        # Determine the basename based on intersection of all files
        _, ext = os.path.splitext(file)
        sm = difflib.SequenceMatcher(None, file, basename)
        i, j, k = sm.find_longest_match(
            0, len(file), 0, len(basename)
        )
        basename = file[i:i+k]
    return basename

def findalliter(patter, string):
    """Function used to return all re.findall objects in string"""
    m_test = re.findall(r'(\d+)_(.+)', string)
    m_iter = [m_test]
    while m_test:
        m_test = re.findall(r'(\d+)_(.+)', m_test[0][1])
        m_iter.append(m_test)
    return m_iter

def clipSelemMask(mask, shape, Yc, Xc, copy=True):
    if copy:
        mask = mask.copy()
    
    Y, X = shape
    h, w = mask.shape

    # Bottom, Left, Top, Right global coordinates of mask
    Y0, X0, Y1, X1 = Yc-(h/2), Xc-(w/2), Yc+(h/2), Xc+(w/2)
    mask_limits = [floor(Y0)+1, floor(X0)+1, floor(Y1)+1, floor(X1)+1]
    
    if Y0>=0 and X0>=0 and Y1<=Y and X1<=X:
        # Mask is withing shape boundaries, no need to clip
        ystart, xstart, yend, xend = mask_limits
        mask_slice = slice(ystart, yend), slice(xstart, xend)
        return mask, mask_slice

    if Y0<0:
        # Mask is exceeding at the bottom
        ystart = floor(abs(Y0))
        mask_limits[0] = 0
        mask = mask[ystart:]
    if X0<0:
        # Mask is exceeding at the left
        xstart = floor(abs(X0))
        mask_limits[1] = 0
        mask = mask[:, xstart:]
    if Y1>Y:
        # Mask is exceeding at the top
        yend = ceil(abs(Y1)) - Y
        mask_limits[2] = Y
        mask = mask[:-yend]
    if X1>X:
        # Mask is exceeding at the right
        xend = ceil(abs(X1)) - X
        mask_limits[3] = X
        mask = mask[:, :-xend]
    
    ystart, xstart, yend, xend = mask_limits
    mask_slice = slice(ystart, yend), slice(xstart, xend)
    return mask, mask_slice


def listdir(path) -> List[str]:
    return natsorted([
        f for f in os.listdir(path)
        if not f.startswith('.')
        and not f == 'desktop.ini'
        and not f == 'recovery'
    ])

def insertModelArgSpect(
        params, param_name, param_value, param_type=None, desc='',
        docstring=''
    ):
    updated_params = []
    for param in params:
        if param.name == param_name:
            if param_type is None:
                param_type = param.type
            new_param = ArgSpec(
                name=param_name, default=param_value, type=param_type,
                desc=desc, docstring=docstring
            )
            updated_params.append(new_param)
        else:
            updated_params.append(param)
    return updated_params

def getModelArgSpec(acdcSegment):
    init_ArgSpec = inspect.getfullargspec(acdcSegment.Model.__init__)
    init_kwargs_type_hints = typing.get_type_hints(acdcSegment.Model.__init__)
    init_doc = acdcSegment.Model.__init__.__doc__
    init_params = params_to_ArgSpec(
        init_ArgSpec, init_kwargs_type_hints, init_doc
    )
    init_params = add_segm_data_param(init_params, init_ArgSpec)
    
    segment_ArgSpec = inspect.getfullargspec(acdcSegment.Model.segment)
    segment_kwargs_type_hints = typing.get_type_hints(acdcSegment.Model.segment)
    try:
        segment_ArgSpec.args.remove('frame_i')
    except Exception as e:
        pass
    
    segment_doc = acdcSegment.Model.segment.__doc__
    segment_params = params_to_ArgSpec(
        segment_ArgSpec, segment_kwargs_type_hints, segment_doc,
    )
    
    return init_params, segment_params

def _get_doc_stop_idx(docstring, start_idx, next_param_name=None, debug=False):
    if debug:
        import pdb; pdb.set_trace()
    
    if next_param_name is not None:
        doc_stop_idx = docstring.find(f'{next_param_name} : ')
        if doc_stop_idx > 1:
            return doc_stop_idx
    
    docstring_from_start = docstring[start_idx:]
    next_param_searched = re.search(r'\w+ : ', docstring_from_start)
    if next_param_searched is not None:
        return next_param_searched.start(0) + start_idx
    
    doc_stop_idx = docstring.find('Returns')
    if doc_stop_idx > 1:
        return doc_stop_idx
    
    doc_stop_idx = docstring.find('Notes')
    if doc_stop_idx > 1:
        return doc_stop_idx 
    
    return -1

def parse_model_param_doc(name, next_param_name=None, docstring=None):
    if not docstring:
        return ''
    
    try:
        # Extract parameter description from 'param : ...'
        start_text = f'{name} : '
        if docstring.find(start_text) == -1:
            # Parameter not present in docstring
            return ''
        
        doc_start_idx = docstring.find(start_text) + len(start_text)
        
        doc_stop_idx = _get_doc_stop_idx(
            docstring, doc_start_idx, next_param_name=next_param_name
        )
        if doc_stop_idx == -1:
            doc_stop_idx = len(docstring)
        
        param_doc = docstring[doc_start_idx:doc_stop_idx]
        
        # Start at first end of line
        param_doc = param_doc[param_doc.find('\n')+1:]
        
        # Replace multiples spaces with single space
        param_doc = re.sub(' +', ' ', param_doc)
        
        # Remove trailing spaces
        param_doc = param_doc.strip()
    except Exception as err:
        param_doc = ''
    
    param_doc = param_doc.replace(', optional', '')
    
    return param_doc

def add_segm_data_param(init_params, init_argspecs):
    if init_argspecs.defaults is None:
        num_kwargs = 0
    else:
        num_kwargs = len(init_argspecs.defaults)
    
    # Segm model requires segm data --> add it to params
    num_args = len(init_argspecs.args) - num_kwargs
    if num_args == 1:
        # Args is only self --> segm data not needed
        return init_params
    
    desc = (
'This model requires an additional segmentation file as input.\n\n'
'Please, select which segmentation file to provide to the model.'
    )
    
    segm_data_argspec = ArgSpec(
        name='Auxiliary segmentation file', 
        default='', 
        type=str, 
        desc=desc,
        docstring=None
    )
    
    init_params.insert(0, segm_data_argspec)
    return init_params

def params_to_ArgSpec(
        fullargspecs, type_hints, docstring, args_to_skip=None
    ):
    params = []
    
    if fullargspecs.defaults is None:
        return params
    
    if args_to_skip is None:
        args_to_skip = set()
    
    num_params = len(fullargspecs.args)
    ip = num_params - len(fullargspecs.defaults)
    if ip < 0:
        return params
    
    for arg, default in zip(fullargspecs.args[ip:], fullargspecs.defaults):
        if arg in args_to_skip:
            continue
        
        if arg in type_hints:
            _type = type_hints[arg]
        else:
            _type = type(default)
        
        next_param_name = None
        if ip+1 < num_params:
            next_param_name = fullargspecs.args[ip+1]
        
        param_doc = parse_model_param_doc(
            arg, 
            next_param_name=next_param_name,
            docstring=docstring
        )
        param = ArgSpec(
            name=arg, 
            default=default, 
            type=_type, 
            desc=param_doc,
            docstring=docstring
        )
        params.append(param)
        ip += 1
    return params

def getClassArgSpecs(classModule, runMethodName='run'):
    init_ArgSpec = inspect.getfullargspec(classModule.__init__)
    init_kwargs_type_hints = typing.get_type_hints(
        classModule.__init__
    )
    init_doc = classModule.__init__.__doc__
    init_params = params_to_ArgSpec(
        init_ArgSpec, init_kwargs_type_hints, init_doc
    )
    
    run_ArgSpec = inspect.getfullargspec(getattr(classModule, runMethodName))
    run_kwargs_type_hints = typing.get_type_hints(
        getattr(classModule, runMethodName)
    )
    run_doc = getattr(classModule, runMethodName).__doc__
    run_params = params_to_ArgSpec(
        run_ArgSpec, run_kwargs_type_hints, run_doc,
        args_to_skip={'signals', 'export_to'}
    )
    return init_params, run_params

def getTrackerArgSpec(trackerModule, realTime=False):
    init_ArgSpec = inspect.getfullargspec(trackerModule.tracker.__init__)
    init_kwargs_type_hints = typing.get_type_hints(
        trackerModule.tracker.__init__
    )
    init_doc = trackerModule.tracker.__init__.__doc__
    init_params = params_to_ArgSpec(
        init_ArgSpec, init_kwargs_type_hints, init_doc
    )
    if realTime:
        track_ArgSpec = inspect.getfullargspec(trackerModule.tracker.track_frame)
        track_kwargs_type_hints = typing.get_type_hints(
            trackerModule.tracker.track_frame
        )
        track_doc = trackerModule.tracker.track_frame.__doc__
    else:
        track_ArgSpec = inspect.getfullargspec(trackerModule.tracker.track)
        track_kwargs_type_hints = typing.get_type_hints(
            trackerModule.tracker.track
        )
        track_doc = trackerModule.tracker.track.__doc__

    track_params = params_to_ArgSpec(
        track_ArgSpec, track_kwargs_type_hints, track_doc,
        args_to_skip={'signals', 'export_to'}
    )
    return init_params, track_params

def isIntensityImgRequiredForTracker(trackerModule):
    track_ArgSpec = inspect.getfullargspec(trackerModule.tracker.track)
    num_args = len(track_ArgSpec.args) - len(track_ArgSpec.defaults)
    # If the number of args is 3 then we have `self, labels, image` as args 
    # which means the tracker requires the image 
    return num_args == 3                          

def getDefault_SegmInfo_df(posData, filename):
    mid_slice = int(posData.SizeZ/2)
    df = pd.DataFrame({
        'filename': [filename]*posData.SizeT,
        'frame_i': range(posData.SizeT),
        'z_slice_used_dataPrep': [mid_slice]*posData.SizeT,
        'which_z_proj': ['single z-slice']*posData.SizeT,
        'z_slice_used_gui': [mid_slice]*posData.SizeT,
        'which_z_proj_gui': ['single z-slice']*posData.SizeT,
        'resegmented_in_gui': [False]*posData.SizeT,
        'is_from_dataPrep': [False]*posData.SizeT
    }).set_index(['filename', 'frame_i'])
    return df

def get_examples_path(which):
    if which == 'time_lapse_2D':
        foldername = 'TimeLapse_2D'
        url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/CaMdYXiwxxoq3Ts/download/TimeLapse_2D.zip'
        file_size = 45143552
    elif which == 'snapshots_3D':
        foldername = 'Multi_3D_zStack_Analysed'
        url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/CXZDoQMANNrKL7a/download/Yeast_Analysed_multi3D_zStacks.zip'
        file_size = 124822528
    else:
        return ''
    
    examples_path = os.path.join(user_profile_path, 'acdc-examples')
    example_path = os.path.join(examples_path, foldername)
    return examples_path, example_path, url, file_size

def download_examples(which='time_lapse_2D', progress=None):
    examples_path, example_path, url, file_size = get_examples_path(which)
    if os.path.exists(example_path):
        if progress is not None:
            # display 100% progressbar
            progress.emit(0, 0)
        return example_path

    zip_dst = os.path.join(examples_path, 'example_temp.zip')

    if not os.path.exists(examples_path):
        os.makedirs(examples_path, exist_ok=True)

    print(f'Downloading example to {example_path}')

    download_url(
        url, zip_dst, verbose=False, file_size=file_size,
        progress=progress
    )
    exctract_to = examples_path
    extract_zip(zip_dst, exctract_to)

    if progress is not None:
        # display 100% progressbar
        progress.emit(0, 0)

    # Remove downloaded zip archive
    os.remove(zip_dst)
    print('Example downloaded successfully')
    return example_path

def get_acdc_java_path():
    acdc_java_path = os.path.join(user_profile_path, 'acdc-java')
    dot_acdc_java_path = os.path.join(user_profile_path, '.acdc-java')
    return acdc_java_path, dot_acdc_java_path

def get_java_url():
    is_linux = sys.platform.startswith('linux')
    is_mac = sys.platform == 'darwin'
    is_win = sys.platform.startswith("win")
    is_win64 = (is_win and (os.environ["PROCESSOR_ARCHITECTURE"] == "AMD64"))

    # https://drive.google.com/drive/u/0/folders/1MxhySsxB1aBrqb31QmLfVpq8z1vDyLbo
    if is_win64:
        os_foldername = 'win64'
        unzipped_foldername = 'java_portable_windows-0.1'
        file_size = 214798150
        # url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/eMyirTw8qG2wJMt/download/java_portable_windows-0.1.zip'
        url = 'https://github.com/SchmollerLab/java_portable_windows/archive/refs/tags/v0.1.zip'
    elif is_mac:
        os_foldername = 'macOS'
        unzipped_foldername = 'java_portable_macos-0.1'
        url = 'https://github.com/SchmollerLab/java_portable_macos/archive/refs/tags/v0.1.zip'
        # url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/SjZb8aommXgrECq/download/java_portable_macos-0.1.zip'
        file_size = 108478751
    elif is_linux:
        os_foldername = 'linux'
        unzipped_foldername = 'java_portable_linux-0.1'
        url = 'https://github.com/SchmollerLab/java_portable_linux/archive/refs/tags/v0.1.zip'
        # url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/HjeQagixE2cjbZL/download/java_portable_linux-0.1.zip'
        file_size = 92520706
    return url, file_size, os_foldername, unzipped_foldername

def _jdk_exists(jre_path):
    # If jre_path exists and it's windows search for ~/acdc-java/win64/jdk
    # or ~/.acdc-java/win64/jdk. If not Windows return jre_path
    if not jre_path:
        return ''
    os_acdc_java_path = os.path.dirname(jre_path)
    os_foldername = os.path.basename(os_acdc_java_path)
    if not os_foldername.startswith('win'):
        return jre_path
    if os.path.exists(os_acdc_java_path):
        for folder in os.listdir(os_acdc_java_path):
            if not folder.startswith('jdk'):
                continue
            dir_path =  os.path.join(os_acdc_java_path, folder)
            for file in os.listdir(dir_path):
                if file == 'bin':
                    return dir_path
    return ''

def get_package_version(import_pkg_name):
    import importlib.metadata
    version =  importlib.metadata.version(import_pkg_name)
    return version

def check_upgrade_javabridge():
    try:
        version =  get_package_version('javabridge')
    except Exception as e:
        return
    patch = int(version.split('.')[2])
    if patch > 19:
        return
    install_javabridge()

def _java_exists(os_foldername):
    acdc_java_path, dot_acdc_java_path = get_acdc_java_path()
    os_acdc_java_path = os.path.join(acdc_java_path, os_foldername)
    if os.path.exists(os_acdc_java_path):
        for folder in os.listdir(os_acdc_java_path):
            if not folder.startswith('jre'):
                continue
            dir_path =  os.path.join(os_acdc_java_path, folder)
            for file in os.listdir(dir_path):
                if file == 'bin':
                    return dir_path

    # Some users still has the old .acdc folder --> check
    os_dot_acdc_java_path = os.path.join(dot_acdc_java_path, os_foldername)
    if os.path.exists(os_dot_acdc_java_path):
        for folder in os.listdir(os_dot_acdc_java_path):
            if not folder.startswith('jre'):
                continue
            dir_path =  os.path.join(os_dot_acdc_java_path, folder)
            for file in os.listdir(dir_path):
                if file == 'bin':
                    return dir_path
    return ''

    # Check if the user unzipped the javabridge_portable folder and not its content
    os_acdc_java_path = os.path.join(acdc_java_path, os_foldername)
    if os.path.exists(os_acdc_java_path):
        for folder in os.listdir(os_acdc_java_path):
            dir_path =  os.path.join(os_acdc_java_path, folder)
            if folder.startswith('java_portable') and os.path.isdir(dir_path):
                # Move files one level up
                unzipped_path = os.path.join(os_acdc_java_path, folder)
                for name in os.listdir(unzipped_path):
                    # move files up one level
                    src = os.path.join(unzipped_path, name)
                    shutil.move(src, os_acdc_java_path)
                try:
                    shutil.rmtree(unzipped_path)
                except PermissionError as e:
                    pass
        # Check if what we moved one level up was actually java
        for folder in os.listdir(os_acdc_java_path):
            if not folder.startswith('jre'):
                continue
            dir_path =  os.path.join(os_acdc_java_path, folder)
            for file in os.listdir(dir_path):
                if file == 'bin':
                    return dir_path
    return ''

def download_java():
    url, file_size, os_foldername, unzipped_foldername = get_java_url()
    jre_path = _java_exists(os_foldername)
    jdk_path = _jdk_exists(jre_path)
    if os_foldername.startswith('win') and jre_path and jdk_path:
        return jre_path, jdk_path, url

    if jre_path:
        # on macOS jdk is the same as jre
        return jre_path, jre_path, url

    acdc_java_path, _ = get_acdc_java_path()
    os_acdc_java_path = os.path.join(acdc_java_path, os_foldername)
    temp_zip = os.path.join(os_acdc_java_path, 'acdc_java_temp.zip')

    if not os.path.exists(os_acdc_java_path):
        os.makedirs(os_acdc_java_path, exist_ok=True)

    try:
        download_url(url, temp_zip, file_size=file_size, desc='Java')
        extract_zip(temp_zip, os_acdc_java_path)
    except Exception as e:
        print('=======================')
        traceback.print_exc()
        print('=======================')
    finally:
        os.remove(temp_zip)

    # Move files one level up
    unzipped_path = os.path.join(os_acdc_java_path, unzipped_foldername)
    for name in os.listdir(unzipped_path):
        # move files up one level
        src = os.path.join(unzipped_path, name)
        shutil.move(src, os_acdc_java_path)
    try:
        shutil.rmtree(unzipped_path)
    except PermissionError as e:
        pass

    jre_path = _java_exists(os_foldername)
    jdk_path = _jdk_exists(jre_path)
    return jre_path, jdk_path, url

def get_model_path(model_name, create_temp_dir=True):
    if model_name == 'Automatic thresholding':
        model_name == 'thresholding'
        
    model_info_path = os.path.join(cellacdc_path, 'models', model_name, 'model')
    
    if os.path.exists(model_info_path):
        for file in listdir(model_info_path):
            if file != 'weights_location_path.txt':
                continue
            with open(os.path.join(model_info_path, file), 'r') as txt:
                model_path = txt.read()
                model_path = os.path.expanduser(model_path)
            if not os.path.exists(model_path):
                model_path = _write_model_location_to_txt(model_name)
            else:
                break
        else:
            model_path = _write_model_location_to_txt(model_name)
    else:
        os.makedirs(model_info_path, exist_ok=True)
        model_path = _write_model_location_to_txt(model_name)

    model_path = migrate_to_new_user_profile_path(model_path)   
    
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    if not create_temp_dir:
        return '', model_path

    exists = check_model_exists(model_path, model_name)
    if exists:
        return '', model_path

    temp_zip_path = _create_temp_dir()
    return temp_zip_path, model_path

def check_model_exists(model_path, model_name):
    try:
        import cellacdc
        m = model_name.lower()
        weights_filenames = getattr(cellacdc, f'{m}_weights_filenames')
        files_present = listdir(model_path)
        return all([f in files_present for f in weights_filenames])
    except Exception as e:
        return True
    
def _create_temp_dir():
    temp_model_path = tempfile.mkdtemp()
    temp_zip_path = os.path.join(temp_model_path, 'model_temp.zip')
    return temp_zip_path

def _model_url(model_name, return_alternative=False):
    if model_name == 'YeaZ':
        url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/8PMePcwJXmaMMS6/download/YeaZ_weights.zip'
        alternative_url = 'https://zenodo.org/record/6125825/files/YeaZ_weights.zip?download=1'
        file_size = 693685011
    elif model_name == 'YeastMate':
        url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/pMT8pAmMkNtN8BP/download/yeastmate_weights.zip'
        alternative_url = 'https://zenodo.org/record/6140067/files/yeastmate_weights.zip?download=1'
        file_size = 164911104
    elif model_name == 'segment_anything':
        url = [
            'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', 
            'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth',
            'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'
        ]
        file_size = [2564550879, 1249524736, 375042383]
        alternative_url = ''
    elif model_name == 'YeaZ_v2':
        url = [
            'https://hmgubox2.helmholtz-muenchen.de/index.php/s/5PARckkcJcN9D3S/download/weights_budding_BF_multilab_0_1', 
            'https://hmgubox2.helmholtz-muenchen.de/index.php/s/CTHq4HN3adyFbnE/download/weights_budding_PhC_multilab_0_1',
            'https://hmgubox2.helmholtz-muenchen.de/index.php/s/QTtBJycYnLQZsHQ/download/weights_fission_multilab_0_2'
        ]
        file_size = [124142981, 124143031, 124144759]
        alternative_url = 'https://github.com/rahi-lab/YeaZ-GUI#installation'
    elif model_name == 'deepsea':
        url = [
            'https://github.com/abzargar/DeepSea/raw/master/deepsea/trained_models/segmentation.pth',
            'https://github.com/abzargar/DeepSea/raw/master/deepsea/trained_models/tracker.pth'
        ]
        file_size = [7988969, 8637439]
        alternative_url = ''
    elif model_name == 'TAPIR':
        url = [
            'https://storage.googleapis.com/dm-tapnet/tapir_checkpoint.npy'
        ]
        file_size = [124408122]
        alternative_url = ''
    elif model_name == 'Cellpose_germlineNuclei':
        url = [
            'https://hmgubox2.helmholtz-muenchen.de/index.php/s/AXG6fFfD8o5GZ83/download/cellpose_germlineNuclei_2023'
        ]
        file_size = [26570752]
        alternative_url = ''
    elif model_name == 'omnipose':
        url = [
            'https://hmgubox2.helmholtz-muenchen.de/index.php/s/DynLkocWRbQfyRp/download/bact_fluor_cptorch_0'
            'https://hmgubox2.helmholtz-muenchen.de/index.php/s/2248Eoyozp3Ezj2/download/bact_fluor_omnitorch_0',
            'https://hmgubox2.helmholtz-muenchen.de/index.php/s/GiacDfXGerxE7PT/download/bact_phase_omnitorch_0',
            'https://hmgubox2.helmholtz-muenchen.de/index.php/s/DDq8s3CgnG2Yw6H/download/cyto2_omnitorch_0',
            'https://hmgubox2.helmholtz-muenchen.de/index.php/s/MM5meM2J5HbWqXR/download/plant_cptorch_0',
            'https://hmgubox2.helmholtz-muenchen.de/index.php/s/aap7znrWq5sE6JQ/download/plant_omnitorch_0',
            'https://hmgubox2.helmholtz-muenchen.de/index.php/s/w5M46x9qr8zLHZH/download/size_cyto2_omnitorch_0.npy'
        ]
        file_size = [
            26558464,
            26558464,
            26558464,
            26558464,
            26558464,
            75071488,
            4096
        ]
        alternative_url = ''
    else:
        return
    if return_alternative:
        return url, alternative_url
    else:
        return url, file_size

def _download_segment_anything_models():
    urls, file_sizes = _model_url('segment_anything')
    temp_model_path = tempfile.mkdtemp()
    _, final_model_path = (
        get_model_path('segment_anything', create_temp_dir=False)
    )
    for url, file_size in zip(urls, file_sizes):
        filename = url.split('/')[-1]
        final_dst = os.path.join(final_model_path, filename)
        if os.path.exists(final_dst):            
            continue

        temp_dst = os.path.join(temp_model_path, filename)
        download_url(
            url, temp_dst, file_size=file_size, desc='segment_anything',
            verbose=False
        )
        
        shutil.move(temp_dst, final_dst)

def _download_deepsea_models():
    urls, file_sizes = _model_url('deepsea')
    temp_model_path = tempfile.mkdtemp()
    _, final_model_path = (
        get_model_path('deepsea', create_temp_dir=False)
    )
    for url, file_size in zip(urls, file_sizes):
        filename = url.split('/')[-1]
        final_dst = os.path.join(final_model_path, filename)
        if os.path.exists(final_dst):            
            continue

        temp_dst = os.path.join(temp_model_path, filename)
        download_url(
            url, temp_dst, file_size=file_size, desc='deepsea',
            verbose=False
        )
        
        shutil.move(temp_dst, final_dst)

def download_manual():
    manual_folder_path = os.path.join(user_profile_path, 'acdc-manual')
    if not os.path.exists(manual_folder_path):
        os.makedirs(manual_folder_path, exist_ok=True)

    manual_file_path = os.path.join(user_profile_path, 'Cell-ACDC_User_Manual.pdf')
    if not os.path.exists(manual_file_path):
        url = 'https://github.com/SchmollerLab/Cell_ACDC/raw/main/UserManual/Cell-ACDC_User_Manual.pdf'
        download_url(url, manual_file_path, file_size=1727470)
    return manual_file_path

def download_bioformats_jar(
        qparent=None, logger_info=print, logger_exception=print
    ):
    dst_filepath = os.path.join(
        cellacdc_path, 'bioformats', 'jars', 'bioformats_package.jar'
    )
    if os.path.exists(dst_filepath):
        return True, dst_filepath
    urls_to_try = (urls.bioformats_jar_home_url, urls.bioformats_jar_hmgu_url)
    success = False
    for url in urls_to_try:
        try:
            logger_info(
                f'Downloading `bioformats_package.jar`...'
            )
            download_url(url, dst_filepath, file_size=43233280)
            success = True
            break
        except Exception as err:
            success = False
            traceback_str = traceback.format_exc()
            logger_exception(traceback_str)
            continue
    
    if success:
        return True, dst_filepath

    _warnings.warn_download_bioformats_jar_failed(dst_filepath, qparent=qparent)
    raise ModuleNotFoundError(
        'Bioformats package jar could not be downloaded. Please, '
        f'download it from here {urls.bioformats_download_page} and '
        f'place it in the following path "{dst_filepath}". '
        'Thank you for your patience!'
    )
    return False, dst_filepath
        

def showUserManual():
    manual_file_path = download_manual()
    showInExplorer(manual_file_path)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def download_url(
        url, dst, desc='', file_size=None, verbose=True, progress=None
    ):
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    CHUNK_SIZE = 32768
    if verbose:
        print(f'Downloading {desc} to: {os.path.dirname(dst)}')
    response = requests.get(url, stream=True, timeout=20, verify=False)
    if file_size is not None and progress is not None:
        progress.emit(file_size, -1)
    pbar = tqdm(
        total=file_size, unit='B', unit_scale=True,
        unit_divisor=1024, ncols=100
    )
    with open(dst, 'wb') as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            # if chunk:
            f.write(chunk)
            pbar.update(len(chunk))
            if progress is not None:
                progress.emit(-1, len(chunk))
    pbar.close()

def save_response_content(
        response, destination, file_size=None,
        model_name='cellpose', progress=None
    ):
    print(f'Downloading {model_name} to: {os.path.dirname(destination)}')
    CHUNK_SIZE = 32768

    # Download to a temp folder in user path
    temp_folder = pathlib.Path.home().joinpath('.acdc_temp')
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)
    temp_dst = os.path.join(temp_folder, os.path.basename(destination))
    if file_size is not None and progress is not None:
        progress.emit(file_size, -1)
    pbar = tqdm(
        total=file_size, unit='B', unit_scale=True,
        unit_divisor=1024, ncols=100
    )
    with open(temp_dst, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))
                if progress is not None:
                    progress.emit(-1, len(chunk))
    pbar.close()

    # Move to destination and delete temp folder
    destination_dir = os.path.dirname(destination)
    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir, exist_ok=True)
    shutil.move(temp_dst, destination)
    shutil.rmtree(temp_folder)

def extract_zip(zip_path, extract_to_path, verbose=True):
    if verbose:
        print(f'Extracting to {extract_to_path}...')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_path)

def check_v123_model_path(model_name):
    # Cell-ACDC v1.2.3 saved the weights inside the package,
    # while from v1.2.4 we save them on user folder. If we find the
    # weights in the package we move them to user folder without downloading
    # new ones.
    v123_model_path = os.path.join(cellacdc_path, 'models', model_name, 'model')
    exists = check_model_exists(v123_model_path, model_name)
    if exists:
        return v123_model_path
    else:
        return ''

def is_old_user_profile_path(path_to_check: os.PathLike):
    from . import user_data_dir
    user_data_folderpath = user_data_dir()
    user_profile_path_txt = os.path.join(
        user_data_folderpath, 'acdc_user_profile_location.txt'
    )
    if os.path.exists(user_profile_path_txt):
        return False
    
    from . import user_home_path
    user_home_path = user_home_path.replace('\\', '/')
    path_to_check = path_to_check.replace('\\', '/')
    return user_home_path == path_to_check

def migrate_to_new_user_profile_path(path_to_migrate: os.PathLike):
    parent_dir = os.path.dirname(path_to_migrate)
    if not is_old_user_profile_path(parent_dir):
        return path_to_migrate
    folder = os.path.basename(path_to_migrate)
    return os.path.join(user_profile_path, folder)

def _write_model_location_to_txt(model_name):
    model_info_path = os.path.join(cellacdc_path, 'models', model_name, 'model')
    model_path = os.path.join(user_profile_path, f'acdc-{model_name}')
    file = 'weights_location_path.txt'
    with open(os.path.join(model_info_path, file), 'w') as txt:
        txt.write(model_path)
    return os.path.expanduser(model_path)

def determine_folder_type(folder_path):
    is_pos_folder = os.path.basename(folder_path).find('Position_') != -1
    is_images_folder = os.path.basename(folder_path) == 'Images'
    contains_images_folder = os.path.exists(
        os.path.join(folder_path, 'Images')
    )
    contains_pos_folders = len(get_pos_foldernames(folder_path)) > 0
    if contains_pos_folders:
        is_pos_folder = False
        is_images_folder = False
    elif contains_images_folder and not is_pos_folder:
        # Folder created by loading an image
        is_images_folder = True
        folder_path = os.path.join(folder_path, 'Images')
    return is_pos_folder, is_images_folder, folder_path

def download_model(model_name):
    if model_name == 'segment_anything':
        try:
            _download_segment_anything_models()
            return True
        except Exception as e:
            traceback.print_exc()
            return False
    elif model_name == 'DeepSea':
        try:
            _download_deepsea_models()
            return True
        except Exception as e:
            traceback.print_exc()
            return False
    elif model_name == 'TAPIR':
        try:
            _download_tapir_model()
            return True
        except Exception as e:
            traceback.print_exc()
            return False
    elif model_name == 'YeaZ_v2':
        try:
            _download_yeaz_models()
            return True
        except Exception as e:
            traceback.print_exc()
            return False
    elif model_name == 'Cellpose_germlineNuclei':
        try:
            _download_cellpose_germlineNuclei_model()
            return True
        except Exception as e:
            traceback.print_exc()
            return False
    elif model_name == 'omnipose':
        try:
            _download_omnipose_models()
            return True
        except Exception as err:
            return False
    elif model_name != 'YeastMate' and model_name != 'YeaZ':
        # We manage only YeastMate and YeaZ
        return True
    
    try:
        # Check if model exists
        temp_zip_path, model_path = get_model_path(model_name)
        if not temp_zip_path:
            # Model exists return
            return True

        # Check if user has model in the old v1.2.3 location
        v123_model_path = check_v123_model_path(model_name)
        if v123_model_path:
            print(f'Weights files found in {v123_model_path}')
            print(f'--> moving to new location: {model_path}...')
            for file in listdir(v123_model_path):
                src = os.path.join(v123_model_path, file)
                dst = os.path.join(model_path, file)
                shutil.copy(src, dst)
            return True

        # Download model from url to tempDir/model_temp.zip
        temp_dir = os.path.dirname(temp_zip_path)
        url, file_size = _model_url(model_name)
        print(f'Downloading {model_name} to {model_path}')
        download_url(
            url, temp_zip_path, file_size=file_size, desc=model_name,
            verbose=False
        )

        # Extract zip file inside temp dir
        print(f'Extracting model...')
        extract_zip(temp_zip_path, temp_dir, verbose=False)

        # Move unzipped files to ~/acdc-{model_name} folder
        print(f'Moving files from temporary folder to {model_path}...')
        for file in listdir(temp_dir):
            if file.endswith('.zip'):
                continue
            src = os.path.join(temp_dir, file)
            dst = os.path.join(model_path, file)
            shutil.move(src, dst)

        # Remove temp directory
        print(f'Removing temporary folder...')
        shutil.rmtree(temp_dir)
        return True

    except Exception as e:
        traceback.print_exc()
        return False

# def get_tiff_metadata(
#         image_arr,
#         SizeT=None, 
#         SizeZ=None, 
#         PhysicalSizeZ=None,
#         PhysicalSizeX=None, 
#         PhysicalSizeY=None,
#         TimeIncrement=None
#     ):
#     SizeY, SizeX = image_arr.shape[-2:]
#     Type = str(image_arr.dtype)
    
#     metadata = {
#         'SizeX': SizeX,
#         'SizeY': SizeY,
#         'Type': Type
#     }
    
#     axes = 'YX'
#     if SizeZ is not None and SizeZ > 1:
#         axes = f'Z{axes}'
#         metadata['SizeZ'] = SizeZ
        
#     if SizeT is not None and SizeT > 1:
#         axes = f'T{axes}'
#         metadata['SizeT'] = SizeT
        
#     metadata['axes'] = axes
    
#     if PhysicalSizeX is not None:
#         metadata['PhysicalSizeX'] = PhysicalSizeX
    
#     if PhysicalSizeY is not None:
#         metadata['PhysicalSizeY'] = PhysicalSizeY
    
#     if PhysicalSizeZ is not None:
#         metadata['PhysicalSizeZ'] = PhysicalSizeZ
    
#     if TimeIncrement is not None:
#         metadata['TimeIncrement'] = TimeIncrement
    
#     return metadata

def get_tiff_metadata(
        image_arr,
        SizeT=None, 
        SizeZ=None, 
        PhysicalSizeZ=None,
        PhysicalSizeX=None, 
        PhysicalSizeY=None,
        TimeIncrement=None
    ):
    SizeY, SizeX = image_arr.shape[-2:]
    Type = str(image_arr.dtype)
    
    metadata = {
        'Pixels': {
            'SizeX': SizeX,
            'SizeY': SizeY,
            'Type': Type
        }
    }
    
    axes = 'YX'
    if SizeZ is not None and SizeZ > 1:
        axes = f'Z{axes}'
        metadata['Pixels']['SizeZ'] = SizeZ
        
    if SizeT is not None and SizeT > 1:
        axes = f'T{axes}'
        metadata['Pixels']['SizeT'] = SizeT
        
    metadata['axes'] = axes
    
    if PhysicalSizeX is not None:
        metadata['Pixels']['PhysicalSizeX'] = PhysicalSizeX
    
    if PhysicalSizeY is not None:
        metadata['Pixels']['PhysicalSizeY'] = PhysicalSizeY
    
    if PhysicalSizeZ is not None:
        metadata['Pixels']['PhysicalSizeZ'] = PhysicalSizeZ
    
    if TimeIncrement is not None:
        metadata['Pixels']['TimeIncrement'] = TimeIncrement
    
    return metadata

def to_tiff(
        new_path, data, 
        SizeT=None, 
        SizeZ=None, 
        PhysicalSizeZ=None,
        PhysicalSizeX=None, 
        PhysicalSizeY=None,
        TimeIncrement=None
    ):
    valid_dtypes = (
        np.uint8, np.uint16, np.float32
    )
    is_valid_dtype = False
    for valid_dtype in valid_dtypes:
        if np.issubdtype(data.dtype, valid_dtype):
            is_valid_dtype = True
            break
    
    if not is_valid_dtype:
        data = data.astype(np.float32)
    
    metadata = get_tiff_metadata(
        data,
        SizeT=SizeT, 
        SizeZ=SizeZ, 
        PhysicalSizeZ=PhysicalSizeZ,
        PhysicalSizeX=PhysicalSizeX, 
        PhysicalSizeY=PhysicalSizeY,
        TimeIncrement=TimeIncrement
    )
    
    # # Potential alternative 
    # hyperstack = tifffile.memmap(
    #     new_path,
    #     shape=img.shape,
    #     dtype=img.dtype,
    #     imagej=True,
    #     metadata={'axes': 'TZYX'},
    # )
    # hyperstack[:] = img
    # hyperstack.flush()
    
    try:
        tifffile.imwrite(
            new_path, data, metadata=metadata, imagej=True
        )
    except Exception as err:
        tifffile.imwrite(new_path, data)

def from_lab_to_obj_coords(lab):
    rp = skimage.measure.regionprops(lab)
    dfs = []
    keys = []
    for obj in rp:
        keys.append(obj.label)
        obj_coords = obj.coords
        ndim = obj_coords.shape[1]
        if ndim == 3:
            columns = ['z', 'y', 'x']
        else:
            columns = ['y', 'x']
        df_obj = pd.DataFrame(data=obj_coords, columns=columns)
        dfs.append(df_obj)
    df = pd.concat(dfs, keys = keys, names=['Cell_ID', 'idx']).droplevel('idx')
    return df

def lab2d_to_rois(ImagejRoi, lab2D, ndigits, t=None, z=None):
    rp = skimage.measure.regionprops(lab2D)
    rois = []
    for obj in rp:
        cont = core.get_obj_contours(obj)
        yc, xc = obj.centroid
        x_str = str((int(xc))).zfill(ndigits)
        y_str = str((int(yc))).zfill(ndigits)
        name = f'{x_str}-{y_str}'
        if z is not None:
            z_str = str(z).zfill(ndigits)
            name = f'{z_str}-{name}'
        
        if t is not None:
            t_str = str(t).zfill(ndigits)
            name = f'{t_str}-{name}'
        
        name = f'id={obj.label}-{t_str}-{name}'
        
        roi = ImagejRoi.frompoints(
            cont, name=name, t=t, z=z, group=obj.label
        )
        rois.append(roi)
    return rois

def from_lab_to_imagej_rois(lab, ImagejRoi, t=0, SizeT=1, max_ID=None):
    if max_ID is None:
        max_ID = lab.max()
    
    if SizeT == 1:
        t = None
    
    SizeY, SizeX = lab.shape[-2:]
    ndigitsT = len(str(SizeT))
    ndigitsY = len(str(SizeY))
    ndigitsX = len(str(SizeX))
    
    if lab.ndim == 3:
        rois = []
        SizeZ = len(lab)
        ndigitsZ = len(str(SizeZ))
        ndigits = max(ndigitsT, ndigitsZ, ndigitsY, ndigitsX)
        for z, lab2D in enumerate(lab):
            z_rois = lab2d_to_rois(ImagejRoi, lab2D, ndigits, t=t, z=z)
        rois.extend(z_rois)
    else:
        rois = lab2d_to_rois(ImagejRoi, lab2D, ndigits, t=t)
    return rois

def from_imagej_rois_to_segm_data(
        TZYX_shape, ID_to_roi_mapper, rescale_rois_sizes, 
        repeat_2d_rois_zslices_range
    ):
    SizeT, SizeZ, SizeY, SizeX = TZYX_shape
    segm_data = np.zeros(TZYX_shape, dtype=np.uint32)
    for ID, roi in ID_to_roi_mapper.items():
        name = roi.name
        name_parts = name.split('-')
        zz = [0]
        if len(name_parts) == 2 and SizeZ > 1:
            # 2D roi in 3D segm data --> place 2D roi on each z-slice
            zz = range(*repeat_2d_rois_zslices_range)
        
        elif len(name_parts) > 2 and SizeZ > 1:
            # 2D roi from a 3D roi --> place at requested z-slice
            zz = [int(name_parts[-3])]
        
        tt = [0]*len(zz)
        if SizeT > 1:
            tt = [roi.t_position]*len(zz)
        
        y0, x0 = roi.top, roi.left
        contours = roi.integer_coordinates + (x0, y0)
        xx = contours[:, 0]
        yy = contours[:, 1]
        if rescale_rois_sizes is not None:        
            rescale_z = rescale_rois_sizes['Z']
            rescale_y = rescale_rois_sizes['Y']
            rescale_x = rescale_rois_sizes['X']
            
            factor_z = rescale_z[1]/rescale_z[0]
            factor_y = rescale_y[1]/rescale_y[0]
            factor_x = rescale_x[1]/rescale_x[0]
            
            xx = np.clip(np.round(xx * factor_x).astype(int), 0, SizeX-1)
            yy = np.clip(np.round(yy * factor_y).astype(int), 0, SizeY-1)
            
        for t, z in zip(tt, zz):
            if rescale_rois_sizes is not None:
                z = round(z*factor_z)
                z = z if z<SizeZ else SizeZ
                z = z if z>=0 else 0
            
            rr, cc = skimage.draw.polygon(yy, xx)
            segm_data[t, z, rr, cc] = ID
    
    return np.squeeze(segm_data)
            

def get_list_of_real_time_trackers():
    trackers = get_list_of_trackers()
    rt_trackers = []
    for tracker in trackers:
        if tracker == 'CellACDC':
            continue
        if tracker == 'YeaZ':
            continue
        tracker_filename = f'{tracker}_tracker.py'
        tracker_path = os.path.join(
            cellacdc_path, 'trackers', tracker, tracker_filename
        )
        try:
            with open(tracker_path) as file:
                txt = file.read()
            if txt.find('def track_frame') != -1:
                rt_trackers.append(tracker)
        except Exception as e:
            continue
    return rt_trackers

def get_list_of_trackers():
    trackers_path = os.path.join(cellacdc_path, 'trackers')
    trackers = []
    for name in listdir(trackers_path):
        _path = os.path.join(trackers_path, name)
        tracker_script_path = os.path.join(_path, f'{name}_tracker.py')
        is_valid_tracker = (
            os.path.isdir(_path) and os.path.exists(tracker_script_path)
            and not name.endswith('__')
        )
        if is_valid_tracker:
            trackers.append(name)
    return natsorted(trackers)

def get_list_of_models():
    models_path = os.path.join(cellacdc_path, 'models')
    models = set()
    for name in listdir(models_path):
        _path = os.path.join(models_path, name)
        if not os.path.exists(_path):
            continue
        
        if not os.path.isdir(_path):
            continue
        
        if name.endswith('__'):
            continue
        
        if name == 'skip_segmentation':
            continue
        
        if not os.path.exists(os.path.join(_path, 'acdcSegment.py')):
            continue
        
        if name == 'thresholding':
            name = 'Automatic thresholding'
        
        models.add(name)
    if not os.path.exists(models_list_file_path):
        return natsorted(list(models))
    
    cp = config.ConfigParser()
    cp.read(models_list_file_path)
    models.update(cp.sections())
    return natsorted(list(models))

def seconds_to_ETA(seconds):
    seconds = round(seconds)
    ETA = datetime.timedelta(seconds=seconds)
    ETA_split = str(ETA).split(':')
    if seconds < 0:
        ETA = '00h:00m:00s'
    elif seconds >= 86400:
        days, hhmmss = str(ETA).split(',')
        h, m, s = hhmmss.split(':')
        ETA = f'{days}, {int(h):02}h:{int(m):02}m:{int(s):02}s'
    else:
        h, m, s = str(ETA).split(':')
        ETA = f'{int(h):02}h:{int(m):02}m:{int(s):02}s'
    return ETA

def to_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = np.round(img_to_float(img)*255).astype(np.uint8)
    return img

def to_uint16(img):
    if img.dtype == np.uint16:
        return img
    img = np.round(img_to_float(img)*65535).astype(np.uint16)
    return img

def elided_text(text, max_len=50, elid_idx=None):
    if len(text) <= max_len:
        return text

    if elid_idx is None:
        elid_idx = int(max_len/2)
    if elid_idx >= max_len:
        elid_idx = max_len - 1
    idx1 = elid_idx
    idx2 = elid_idx - max_len
    text = f'{text[:idx1]}...{text[idx2:]}'
    return text

def to_relative_path(path, levels=3, prefix='...'):
    path = path.replace('\\', '/')
    parts = path.split('/')
    if levels >= len(parts):
        return path
    parts = parts[-levels:]
    rel_path = '/'.join(parts)
    rel_path.replace('/', os.sep)
    if prefix:
        rel_path = f'{prefix}{os.sep}{rel_path}'
    return rel_path

def img_to_float(img, force_dtype=None, force_missing_dtype=None):
    input_img_dtype = img.dtype
    value = img[(0,) * img.ndim]
    img_max = np.max(img)
    # Check if float outside of -1, 1
    if img_max <= 1.0 and isinstance(value, (np.floating, float)):
        return img

    uint8_max = np.iinfo(np.uint8).max
    uint16_max = np.iinfo(np.uint16).max
    uint32_max = np.iinfo(np.uint32).max
    
    img = img.astype(float)
    
    if force_dtype is not None:
        dtype_max = np.iinfo(force_dtype).max
        img = img/dtype_max
    elif input_img_dtype == np.uint8:
        # Input image is 8-bit
        img = img/uint8_max
    elif input_img_dtype == np.uint16:
        # Input image is 16-bit
        img = img/uint16_max    
    elif input_img_dtype == np.uint32:
        # Input image is 32-bit
        img = img/uint32_max
    elif force_missing_dtype is not None:
        img = img.astype(force_dtype)
    elif img_max <= uint8_max:
        # Input image is probably 8-bit
        _warnings.warn_image_overflow_dtype(input_img_dtype, img_max, '8-bit')
        img = img/uint8_max
    elif img_max <= uint16_max:
        # Input image is probably 16-bit
        _warnings.warn_image_overflow_dtype(input_img_dtype, img_max, '16-bit')
        img = img/uint16_max
    elif img_max <= uint32_max:
        # Input image is probably 32-bit
        _warnings.warn_image_overflow_dtype(input_img_dtype, img_max, '32-bit')
        img = img/uint32_max
    else:
        # Input image is a non-supported data type
        raise TypeError(
            f'The maximum value in the image is {img_max} which is greater than the '
            f'maximum value supported of {uint32_max} (32-bit). '
            'Please consider converting your images to 32-bit or 16-bit first.'
        )
    return img

def float_img_to_dtype(img, dtype):
    if img.dtype == dtype:
        return img
    
    img_max = img.max()
    if img_max > 1.0:
        raise TypeError(
            'Images of float data type with values greater than 1.0 cannot '
            f'be safely casted to {dtype}.'
            f'The max value of the input image is {img_max:.3f}'
        )
    
    img_min = img.min()
    if img_min < -1.0:
        raise TypeError(
            'Images of float data type with values smaller than -1.0 cannot '
            f'be safely casted to {dtype}.'
            f'The minumum value of the input image is {img_min:.3f}'
        )

    if dtype == np.uint8:
        return skimage.img_as_ubyte(img)
    
    if dtype == np.uint16:
        return skimage.img_as_uint(img)
    
    raise TypeError(
        f'Invalid output data type `{dtype}`. '
        'Valid output data types are `np.uin8` and `np.uint16`'
    )

def scale_float(data, force_dtype=None, force_missing_dtype=None):
    val = data[tuple([0]*data.ndim)]
    if isinstance(val, (np.floating, float)):
        data = img_to_float(
            data, 
            force_dtype=force_dtype, 
            force_missing_dtype=force_missing_dtype
        )
    return data

def _install_homebrew_command():
    return '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'

def _brew_install_java_command():
    return 'brew install --cask homebrew/cask-versions/adoptopenjdk8'

def _brew_install_hdf5():
    return 'brew install hdf5'

def _apt_update_command():
    return 'sudo apt-get update'

def _apt_gcc_command():
    return 'sudo apt install python-dev gcc'

def _apt_install_java_command():
    return 'sudo apt-get install openjdk-8-jdk'

def _java_instructions_linux():
    s1 = html_utils.paragraph("""
        Run the following commands<br>
        in the Teminal <b>one by one:</b>
    """)

    s2 = html_utils.paragraph(f"""
        <code>{_apt_gcc_command().replace(' ', '&nbsp;')}</code>
    """)

    s3 = html_utils.paragraph(f"""
        <code>{_apt_update_command().replace(' ', '&nbsp;')}</code>
    """)

    s4 = html_utils.paragraph(f"""
        <code>{_apt_install_java_command().replace(' ', '&nbsp;')}</code>
    """)

    s5 = html_utils.paragraph("""
    The first command is used to install GCC, which is needed later.<br><br>
    The second and third commands are used is used to install
    Java Development Kit 8.<br><br>
    Follow the instructions on the terminal to complete
    installation.<br><br>
    """)
    return s1, s2, s3, s4

def _java_instructions_macOS():
    s1 = html_utils.paragraph("""
        Run the following commands<br>
        in the Teminal <b>one by one:</b>
    """)

    s2 = html_utils.paragraph(f"""
        <code>{_install_homebrew_command()}</code>
    """)

    s3 = html_utils.paragraph(f"""
        <code>{_brew_install_java_command().replace(' ', '&nbsp;')}</code>
    """)

    s4 = html_utils.paragraph("""
    The first command is used to install Homebrew<br>
    a package manager for macOS/Linux.<br><br>
    The second command is used to install Java 8.<br>
    Follow the instructions on the terminal to complete
    installation.<br><br>
    Alternatively,<b> you can install Java as a regular app</b><br>
    by downloading the app from
    <a href="https://hmgubox2.helmholtz-muenchen.de/index.php/s/AWWinWCTXwWTmEi">
        here
    </a>.
    """)
    return s1, s2, s3, s4

def jdk_windows_url():
    return 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/R62Ktcda6jWea2s'

def cpp_windows_url():
    return 'https://visualstudio.microsoft.com/visual-cpp-build-tools/'

def _java_instructions_windows():
    jdk_url = f'"{jdk_windows_url()}"'
    cpp_url = f'"{cpp_windows_url()}"'
    s1 = html_utils.paragraph("""
        Download and install <code>Java Development Kit</code> and<br>
        <b>Microsoft C++ Build Tools</b> for Windows (links below).<br><br>
        <b>IMPORTANT</b>: when installing "Microsoft C++ Build Tools"<br>
        make sure to select <b>"Desktop development with C++"</b>.<br>
        Click "See the screenshot" for more details.<br>
    """)

    s2 = html_utils.paragraph(f"""
        Java Development Kit:
            <a href={jdk_url}>
                here
            </a>
    """)

    s3 = html_utils.paragraph(f"""
        Microsoft C++ Build Tools:
            <a href={cpp_url}>
                here
            </a>
    """)
    return s1, s2, s3

def install_javabridge_instructions_text():
    if is_win:
        return _java_instructions_windows()
    elif is_mac:
        return _java_instructions_macOS()
    elif is_linux:
        return _java_instructions_linux()

def install_javabridge_help(parent=None):
    msg = widgets.myMessageBox()
    txt = html_utils.paragraph(f"""
        Cell-ACDC is going to <b>download and install</b>
        <code>javabridge</code>.<br><br>
        Make sure you have an <b>active internet connection</b>,
        before continuing.
        Progress will be displayed on the terminal<br><br>
        <b>IMPORTANT:</b> If the installation fails, <b>please open an issue</b>
        on our
        <a href="https://github.com/SchmollerLab/Cell_ACDC/issues">
            GitHub page
        </a>.<br><br>
        Alternatively, you can cancel the process and try later.
    """)
    msg.setIcon()
    msg.setWindowTitle('Installing javabridge')
    msg.addText(txt)
    msg.addButton('   Ok   ')
    cancel = msg.addButton(' Cancel ')
    msg.exec_()
    return msg.clickedButton == cancel

def check_napari_plugin(plugin_name, module_name, parent=None):
    try:
        import_module(module_name)
    except ModuleNotFoundError as e:
        url = 'https://napari.org/stable/plugins/find_and_install_plugin.html#find-and-install-plugins'
        href = html_utils.href_tag('this guide', url)
        txt = html_utils.paragraph(f"""
            To correctly use this napari utility you need to <b>install the 
            plugin</b> called <code>{plugin_name}</code>.<br><br>
            Please, read {href} on how to install plugins in napari.<br><br>
            You will need to <b>restart</b> both napari and Cell-ACDC after installing 
            the plugin.<br><br>
            NOTE: in the text box in napari you will need to write the full name 
            <code>{plugin_name}</code> becasue it is NOT A SEARCH BOX.
        """)
        msg = widgets.myMessageBox()
        msg.critical(parent, f'Napari plugin required', txt)
        raise e

def _install_pip_package(pkg_name):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-U', pkg_name])

def uninstall_pip_package(pkg_name):
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', pkg_name]
    )

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
        uninstall_pip_package(pkg_to_uninstall)

def get_cellpose_major_version(errors='raise'):
    major_installed = None
    try:
        installed_version = get_package_version('cellpose')
        major_installed = int(installed_version.split('.')[0])
    except Exception as err:
        if errors == 'raise':
            raise err
    
    return major_installed

def check_cellpose_version(version: str):
    major_requested = int(version.split('.')[0])
    cancel = False
    try:
        installed_version = get_package_version('cellpose')
        major_installed = int(installed_version.split('.')[0])
        is_version_correct = major_installed == major_requested
        if not is_version_correct:
            cancel = _warnings.warn_installing_different_cellpose_version(
                version, installed_version
            )
    except Exception as err:
        is_version_correct = False
    
    if cancel:
        raise ModuleNotFoundError('Cellpose installation cancelled by the user.')
    return is_version_correct

def check_install_cellpose(version: str='2.0'):
    is_version_correct = check_cellpose_version(version)
    if is_version_correct:
        return
    
    next_version = int(version.split('.')[0])+1
    next_version = f'{next_version}.0'
    
    check_install_package(
        'cellpose', 
        pypi_name=f'cellpose>={version},<{next_version}',
        import_pkg_name='cellpose',
        force_upgrade=True
    )

def check_install_baby():
    check_install_package('baby', pypi_name='baby-seg')

def check_install_yeaz():
    check_install_torch()
    check_install_package('yeaz')

def check_install_segment_anything():
    check_install_torch()
    check_install_package('segment_anything')

def is_gui_running():
    if not GUI_INSTALLED:
        return False
    
    return QCoreApplication.instance() is not None

def check_pkg_version(import_pkg_name, min_version, raise_err=True):
    is_version_correct = False
    try:
        from packaging import version
        installed_version = get_package_version(import_pkg_name)  
        if version.parse(installed_version) > version.parse(min_version):
            is_version_correct = True
    except Exception as err:
        is_version_correct = False
    
    if raise_err and not is_version_correct:
        raise ModuleNotFoundError(
            f'{import_pkg_name}>{min_version} not installed.'
        )
    else:
        return is_version_correct

def install_package_conda(conda_pkg_name, channel='conda-forge'):
    try:
        commad = f'conda install -c {channel} -y {conda_pkg_name}'
        subprocess.check_call([commad], shell=True)
    except Exception as err:
        print(
            f'[WARNING]: Installation with command `{[commad]}` failed. '
            f'Trying with `{commad.split()}`...'
        )
    
    subprocess.check_call(commad.split(), shell=True)

def check_install_omnipose():
    try:
        import_module('omnipose')
        return
    except ModuleNotFoundError:
        pass
    
    try:
        check_install_package('omnipose', pypi_name='omnipose_acdc')
    except Exception as err:
        install_package_conda('mahotas')
        _install_pip_package('omnipose-acdc')

def _run_command(command, shell=True):
    if command.find('conda') == -1:
        args = command.split(' ')
    else:
        args = command
    subprocess.check_call(args, shell=shell)

def check_install_torch(is_cli=False, caller_name='Cell-ACDC', qparent=None):
    try:
        import torch
        import torchvision
        return
    except Exception as err:
        traceback.print_exc()
    
    if is_cli:
        _install_pytorch_cli(caller_name=caller_name) 
        return
    
    win = apps.InstallPyTorchDialog(parent=qparent, caller_name=caller_name)
    win.exec_()
    if win.cancel:
        _warnings.log_pytorch_not_installed() 
        return
    
    command = win.command
    print(f'Running command: "{command}"')
    _run_command(command)    

def check_install_package(
        pkg_name: str, 
        import_pkg_name: str='',
        pypi_name='', 
        note='', 
        parent=None, 
        raise_on_cancel=True, 
        logger_func=print, 
        is_cli=False,
        caller_name='Cell-ACDC', 
        force_upgrade=False,
        upgrade=False, 
        min_version=''
    ):
    """Try to import a package. If import fails, ask user to install it 
    automatically.

    Parameters
    ----------
    pkg_name : str
        The name of the package that is displayed to the user.
    import_pkg_name : str, optional
        The name of the package as it should be imported (case sensitive).
        If empty string, `pkg_name` will be imported instead. Default is ''
    pypi_name : str, optional
        The name of the package to be installed with pip.
        If empty string, `pkg_name` will be installed instead. Default is ''
    note : str, optional
        Additional text to display to the user. Default is ''
    parent : _type_, optional
        Calling QtWidget. Default is None
    raise_on_cancel : bool, optional
        Raise exception if processed cancelled. Default is True
    logger_func : _type_, optional
        Function used to log text. Default is print
    is_cli : bool, optional
        If True, message will be displayed in the terminal. 
        If False, message will be displayed in a Qt message box.
        Default is False
    caller_name : str, optional
        Program calling this function. Default is 'Cell-ACDC'
    force_upgrade : bool, optional
        If True, we force the upgrade even if package is installed.
    upgrade : bool, optional
        If True, pip will upgrade the package. This value is True if 
        `force_upgrade` is True. Default is False
    min_version : str, optional
        If not empty it must be a valid version `major[.minor][.patch]` where 
        minor and patch are optional. If the installed package is older the 
        upgrade will be forced. 

    Raises
    ------
    ModuleNotFoundError
        Error raised if process is cancelled and `raise_on_cancel=True`.
    """    
    if not import_pkg_name:
        import_pkg_name = pkg_name
    
    if not is_gui_running():
        is_cli=True
    
    try:
        import_module(import_pkg_name)
        if force_upgrade:
            upgrade = True
            raise ModuleNotFoundError(
                f'User requested to forcefully upgrade the package "{pkg_name}"')
        if min_version:
            check_pkg_version(import_pkg_name, min_version)
    except ModuleNotFoundError:
        proceed = _install_package_msg(
            pkg_name, note=note, parent=parent, upgrade=upgrade,
            is_cli=is_cli, caller_name=caller_name, logger_func=logger_func,
            pkg_command=pypi_name
        )
        if pypi_name:
            pkg_name = pypi_name
        if not proceed:
            if raise_on_cancel:
                raise ModuleNotFoundError(
                    f'User aborted {pkg_name} installation'
                )
            else:
                return traceback.format_exc()
        try:
            if pkg_name == 'tensorflow':
                _install_tensorflow()
            elif pkg_name == 'deepsea':
                _install_deepsea()
            elif pkg_name == 'segment_anything':
                _install_segment_anything()
            else:
                _install_pip_package(pkg_name)
        except Exception as e:
            printl(traceback.format_exc())
            _inform_install_package_failed(
                pkg_name, parent=parent, do_exit=raise_on_cancel
            )

def get_chained_attr(_object, _name):
    for attr in _name.split('.'):
        _object = getattr(_object, attr)
    return _object

def check_matplotlib_version(qparent=None):
    mpl_version = get_package_version('matplotlib')  
    mpl_version_digits = mpl_version.split('.')

    mpl_version = float(f'{mpl_version_digits[0]}.{mpl_version_digits[1]}')
    if mpl_version < 3.5:
        proceed = _install_package_msg('matplotlib', parent=qparent, upgrade=True)
        if not proceed:
            raise ModuleNotFoundError(
                f'User aborted "matplotlib" installation'
            )
        import subprocess
        try:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-U', 'matplotlib']
            )
        except Exception as e:
            printl(traceback.format_exc())
            _inform_install_package_failed(
                'matplotlib', parent=qparent, do_exit=False
            )
            
def _inform_install_package_failed(pkg_name, parent=None, do_exit=True):
    install_command = f'<code>pip install --upgrade {pkg_name}</code>'
    txt = html_utils.paragraph(f"""
        Unfortunately, <b>installation of</b> <code>{pkg_name}</code> <b>returned an error</b>.<br><br>
        Try restarting Cell-ACDC. If it doesn't work, 
        please close Cell-ACDC and, with the <code>acdc</code> <b>environment ACTIVE</b>, 
        install <code>{pkg_name}</code> manually using the follwing command:<br><br>
        {install_command}<br><br>
        Thank you for your patience.
    """)
    msg = widgets.myMessageBox()
    msg.critical(parent, f'{pkg_name} installation failed', txt)
    print('*'*50)
    print(
        f'[ERROR]: Installation of "{pkg_name}" failed. '
        f'Please, close Cell-ACDC and run the command '
        f'`pip install --upgrade {pkg_name}`'
    )
    print('^'*50)

def download_fiji(logger_func=print):
    url = None
    if is_mac:
        url = 'https://downloads.micron.ox.ac.uk/fiji_update/mirrors/fiji-latest/fiji-macosx.zip'
        file_size = 474_525_405
    
    if url is None:
        return

    if os.path.exists(get_fiji_exec_folderpath()):
        return
    
    os.makedirs(acdc_fiji_path)
    
    temp_dir = tempfile.mkdtemp()
    zip_dst = os.path.join(temp_dir, 'fiji-macosx.zip')
    logger_func(f'Downloading Fiji to "{acdc_fiji_path}"...')
    download_url(
        url, zip_dst, verbose=False, file_size=file_size
    )
    extract_zip(zip_dst, acdc_fiji_path)
    
    return acdc_fiji_path

def _install_package_msg(
        pkg_name, note='', parent=None, upgrade=False, caller_name='Cell-ACDC',
        is_cli=False, pkg_command='', logger_func=print
    ):
    if is_cli:
        proceed = _install_package_cli_msg(
            pkg_name, note=note, upgrade=upgrade, caller_name=caller_name,
            pkg_command=pkg_command, logger_func=logger_func
        )
    else:
        proceed = _install_package_gui_msg(
            pkg_name, note=note, parent=parent, upgrade=upgrade, 
            caller_name=caller_name, pkg_command=pkg_command,
            logger_func=logger_func
        )
    return proceed

def get_cli_multi_choice_question(question, choices):
    choices_format = [f'{i+1}) {choice}.' for i, choice in enumerate(choices)]
    choices_format = ' '.join(choices_format)
    choices_opts = '/'.join([str(i) for i in range(1, len(choices)+1)])
    text = f'{question} {choices_format} q) Quit. ({choices_opts})?: '
    return text

def _install_pytorch_cli(
        caller_name='Cell-ACDC', action='install', logger_func=print
    ):
    from cellacdc import pytorch_commands
    separator = '-'*60
    txt = (
        f'{separator}\n{caller_name} needs to {action} PyTorch\n\n'
        'You can choose to install it now or stop the process and install it '
        'later. To install it correctly, we need to know your preferences.\n'
    )
    logger_func(txt)
    questions = {
        'Choose your OS:': ('Windows', 'Mac', 'Linux'), 
        'Package manager:': ('Conda', 'Pip'), 
        'Compute platform:': (
            'CPU', 'CUDA 11.8 (NVIDIA GPU)', 'CUDA 12.1 (NVIDIA GPU)'
        )
    }
    selected_command = pytorch_commands.copy()
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
            subprocess.check_call(selected_command.split(), shell=True)
    else:
        args = selected_command.split()[1:]
        subprocess.check_call([sys.executable, *args], shell=True)

def _install_package_cli_msg(
        pkg_name, note='', upgrade=False, caller_name='Cell-ACDC',
        logger_func=print, pkg_command=''
    ):
    if not pkg_command:
        pkg_command = pkg_name
    
    if upgrade:
        action = 'upgrade'
    else:
        action = 'install'
        
    separator = '-'*60
    txt = (
        f'{separator}\n{caller_name} needs to {action} {pkg_name}\n\n'
        'You can choose to install it now or stop the process and install it '
        'later with the following command:\n\n'
        f'pip install --upgrade {pkg_command}\n'
    )
    logger_func(txt)
    install_command = f'pip install --upgrade {pkg_command}'
    while True:
        answer = try_input_install_package(pkg_name, install_command)
        if not answer or answer.lower() == 'y':
            return True
        
        if answer.lower() == 'n':
            return False
        
        logger_func(
            f'{answer} is not a valid answer. Valid answers are "y" for Yes and '
            '"n" for No.'
        )
        
def _install_package_gui_msg(
        pkg_name, note='', parent=None, upgrade=False, caller_name='Cell-ACDC', 
        pkg_command='', logger_func=None
    ):
    msg = widgets.myMessageBox(parent=parent)
    if upgrade:
        install_text = 'upgrade'
    else:
        install_text = 'install'
    if pkg_name == 'BayesianTracker':
        pkg_name = 'btrack'
    
    if not pkg_command:
        pkg_command = pkg_name
    
    command_html = pkg_command.lower().replace('<', '&lt;').replace('>', '&gt;')
    command = f'pip install --upgrade {command_html}'

    txt = html_utils.paragraph(f"""
        {caller_name} is going to <b>download and {install_text}</b>
        <code>{pkg_name}</code>.<br><br>
        Make sure you have an <b>active internet connection</b>,
        before continuing.<br>
        Progress will be displayed on the terminal<br><br>
        You might have to <b>restart {caller_name}</b>.<br><br>
        Alternatively, you can cancel the process and try later.<br><br>
        To install later, or if the installation fails, run the following 
        command:
    """)
    if note:
        txt = f'{txt}{note}'
    _, okButton = msg.information(
        parent, f'Install {pkg_name}', txt, 
        buttonsTexts=('Cancel', 'Ok'), 
        commands=(command,)
    )
    return msg.clickedButton == okButton

def _install_tensorflow():
    cpu = platform.processor()
    if is_mac and cpu == 'arm':
        args = ['conda install -y -c conda-forge tensorflow']
        shell = True
    else:
        args = [sys.executable, '-m', 'pip', 'install', '-U', 'tensorflow']
        shell = False
    subprocess.check_call(args, shell=shell)

def _install_segment_anything():
    args = [
        sys.executable, '-m', 'pip', 'install', 
        '-U', '--use-pep517', 
        'git+https://github.com/facebookresearch/segment-anything.git'
    ]
    subprocess.check_call(args)

def _install_deepsea():
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'deepsea']
    )

def import_tracker_module(tracker_name):
    module_name =  f'cellacdc.trackers.{tracker_name}.{tracker_name}_tracker'
    tracker_module = import_module(module_name)
    return tracker_module

def download_ffmpeg():    
    ffmpeg_folderpath = acdc_ffmpeg_path
    if is_win:
        url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/rXioWZpwjwn9JTT/download/windows_ffmpeg-7.0-full_build.zip'
        file_size = 173477888
        ffmep_exec_path = os.path.join(ffmpeg_folderpath, 'bin', 'ffmpeg.exe')
    elif is_mac:
        url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/We7rcTLzqAP4zf7/download/mac_ffmpeg.zip'
        file_size = 25288704
        ffmep_exec_path = os.path.join(ffmpeg_folderpath, 'ffmpeg')
    elif is_linux:
        ffmep_exec_path = ''
        return ffmep_exec_path
    
    if os.path.exists(ffmep_exec_path):
        return ffmep_exec_path.replace('\\', os.sep).replace('/', os.sep)
    
    print('Downloading FFMPEG...')
    temp_dir = tempfile.mkdtemp()
    temp_zip_path = os.path.join(temp_dir, 'acdc-ffmpeg.zip')
    
    download_url(
        url, temp_zip_path, verbose=True, file_size=file_size,
    )
    extract_zip(temp_zip_path, ffmpeg_folderpath)
    
    return ffmep_exec_path.replace('\\', os.sep).replace('/', os.sep)

def get_fiji_exec_folderpath():
    if is_mac:
        return os.path.join(
            acdc_fiji_path, 'Fiji.app', 'Contents', 'MacOS', 'ImageJ-macosx'
        )

def get_fiji_base_command():
    if not os.path.exists(acdc_fiji_path):
        return
    
    command = None
    if is_mac:
        command = f'{get_fiji_exec_folderpath()}'

    return command
    
def _init_fiji_cli():
    if not is_win:
        args_add_to_path = [f'chmod 755 {get_fiji_exec_folderpath()}']
        subprocess.check_call(args_add_to_path, shell=True)

def run_fiji_command(command=None, logger_func=print):
    if command is None:
        command = get_fiji_base_command()
    
    if command is None:
        logger_func('[WARNING]: Fiji is not present.')
        return False
    
    _init_fiji_cli()
    
    commands = (command, command.split())
    for args in commands:
        try:
            subprocess.check_call(args, shell=True)
            return True
        except Exception as err:
            continue
    return False

def init_tracker(
        posData, trackerName, realTime=False, qparent=None, 
        return_init_params=False
    ):
    from . import apps
    downloadWin = apps.downloadModel(trackerName, parent=qparent)
    downloadWin.download()

    trackerModule = import_tracker_module(trackerName)
    init_params = {}
    track_params = {}
    paramsWin = None
    if trackerName == 'BayesianTracker':
        Y, X = posData.img_data_shape[-2:]
        if posData.isSegm3D:
            labShape = (posData.SizeZ, Y, X)
        else:
            labShape = (1, Y, X)
        paramsWin = apps.BayesianTrackerParamsWin(
            labShape, parent=qparent, channels=posData.chNames, 
            currentChannelName=posData.user_ch_name
        )
        paramsWin.exec_()
        if not paramsWin.cancel:
            init_params = paramsWin.params
            track_params['export_to'] = posData.get_btrack_export_path()
            if paramsWin.intensityImageChannel is not None:
                chName = paramsWin.intensityImageChannel
                track_params['image'] = posData.loadChannelData(chName)
                track_params['image_channel_name'] = chName
    elif trackerName == 'CellACDC':
        paramsWin = apps.CellACDCTrackerParamsWin(parent=qparent)
        paramsWin.exec_()
        if not paramsWin.cancel:
            init_params = paramsWin.params
    elif trackerName == 'delta':
        paramsWin = apps.DeltaTrackerParamsWin(posData=posData, parent=qparent)
        paramsWin.exec_()
        if not paramsWin.cancel:
            init_params = paramsWin.params
    else:
        init_argspecs, track_argspecs = getTrackerArgSpec(
            trackerModule, realTime=realTime
        )
        intensityImgRequiredForTracker = isIntensityImgRequiredForTracker(
            trackerModule
        )
        if init_argspecs or track_argspecs:
            try:
                url = trackerModule.url_help()
            except AttributeError:
                url = None
            try:
                channels = posData.chNames
            except Exception as e:
                channels = None
            try:
                currentChannelName = posData.user_ch_name
            except Exception as e:
                currentChannelName = None
            try:
                df_metadata = posData.metadata_df
            except Exception as e:
                df_metadata = None
            
            if not intensityImgRequiredForTracker:
                currentChannelName = None
            
            paramsWin = apps.QDialogModelParams(
                init_argspecs, track_argspecs, trackerName, url=url,
                channels=channels, is_tracker=True,
                currentChannelName=currentChannelName,
                df_metadata=df_metadata, posData=posData
            )
            if not intensityImgRequiredForTracker and channels is not None:
                paramsWin.channelCombobox.setDisabled(True)
                
            paramsWin.exec_()
            if not paramsWin.cancel:
                init_params = paramsWin.init_kwargs
                track_params = paramsWin.model_kwargs
                if paramsWin.inputChannelName != 'None':
                    chName = paramsWin.inputChannelName
                    track_params['image'] = posData.loadChannelData(chName)
                    track_params['image_channel_name'] = chName
        if 'export_to_extension' in track_params:
            ext = track_params['export_to_extension']
            track_params['export_to'] = posData.get_tracker_export_path(
                trackerName, ext
            )

    if paramsWin is not None:
        if paramsWin.cancel:
            return None, None
    
    tracker = trackerModule.tracker(**init_params)
    if return_init_params:
        return tracker, track_params, init_params
    else:
        return tracker, track_params

def import_segment_module(model_name):
    try:
        acdcSegment = import_module(f'cellacdc.models.{model_name}.acdcSegment')
    except ModuleNotFoundError as e:
        # Check if custom model
        cp = config.ConfigParser()
        cp.read(models_list_file_path)
        model_path = cp[model_name]['path']
        spec = importlib.util.spec_from_file_location('acdcSegment', model_path)
        acdcSegment = importlib.util.module_from_spec(spec)
        sys.modules['acdcSegment'] = acdcSegment
        spec.loader.exec_module(acdcSegment) 
    return acdcSegment

def _warn_install_torch_cuda(model_name, qparent=None):
    cellpose_cuda_url = (
        r'https://github.com/mouseland/cellpose#gpu-version-cuda-on-windows-or-linux'
    )
    torch_cuda_url = (
        'https://pytorch.org/get-started/locally/'
    )
    cellpose_href = f'{html_utils.href_tag("here", cellpose_cuda_url)}'
    torch_href = f'{html_utils.href_tag("here", torch_cuda_url)}'
    msg = widgets.myMessageBox(showCentered=False, wrapText=False)
    txt = html_utils.paragraph(f"""
        In order to use <code>{model_name}</code> with the GPU you need 
        to install the <b>CUDA version of PyTorch</b>.<br><br>
        Check out these instructions {cellpose_href}, and {torch_href}.<br><br>
        We <b>highly recommend using Anaconda</b> to install PyTorch GPU.<br><br>
        First, uninstall the CPU version of PyTorch with the following command:<br><br>
        <code>pip uninstall torch</code>.<br><br>
        Then, install the CUDA version required by your GPU with the follwing 
        command (which installs version 11.6):<br><br>
        <code>conda install pytorch pytorch-cuda=11.6 -c pytorch -c nvidia</code>
        <br><br>
        How do you want to proceed?
    """)
    proceedButton = widgets.okPushButton('Proceed without GPU')
    stopButton = widgets.cancelPushButton('Stop the process')
    stopButton, proceedButton = msg.warning(
        qparent, 'PyTorch GPU version not installed', txt, 
        buttonsTexts=(stopButton, proceedButton)
    )
    return msg.clickedButton == proceedButton

def check_cuda(model_name, use_gpu, qparent=None):
    if not use_gpu:
        return True
    is_torch_model = (
        model_name.lower().find('cellpose') != -1
        or model_name.lower().find('omnipose') != -1
        or model_name.lower().find('deepsea') != -1
        or model_name.lower().find('segment_anything') != -1
    )
    if is_torch_model and not is_mac_arm64:
        import torch
        if not torch.cuda.is_available():
            proceed = _warn_install_torch_cuda(model_name, qparent=qparent)
            return proceed
        
        if not torch.cuda.device_count() > 0:
            proceed = _warn_install_torch_cuda(model_name, qparent=qparent)
            return proceed
    
    return True

def find_missing_integers(lst, max_range=None):
    if max_range is not None:
        max_range = lst[-1]+1
    return [x for x in range(lst[0], max_range) if x not in lst]

def synthetic_image_geneator(size=(512,512), f_x=1, f_y=1):
    Y, X = size
    x = np.linspace(0, 10, Y)
    y = np.linspace(0, 10, X)
    xx, yy = np.meshgrid(x, y)
    img = np.sin(f_x*xx)*np.cos(f_y*yy)
    return img

def get_show_in_file_manager_text():
    if is_mac:
        return 'Reveal in Finder'
    elif is_linux:
        return 'Show in File Manager'
    elif is_win:
        return 'Show in File Explorer'

def get_slices_local_into_global_arr(bbox_coords, global_shape):
    slice_global_to_local = []
    slice_crop_local = []
    for (_min, _max), _D in zip(bbox_coords, global_shape):
        _min_crop, _max_crop = None, None
        if _min < 0:
            _min_crop = abs(_min)
            _min = 0
        if _max > _D:
            _max_crop = _D - _max
            _max = _D
        
        slice_global_to_local.append(slice(_min, _max))
        slice_crop_local.append(slice(_min_crop, _max_crop))
    
    return tuple(slice_global_to_local), tuple(slice_crop_local)

def get_pip_install_cellacdc_version_command(version=None):
    if version is None:
        version = read_version()
    commit_hash_idx = version.find('+g')
    is_dev_version = commit_hash_idx > 0    
    if is_dev_version:
        commit_hash = version[commit_hash_idx+2:].split('.')[0]
        command = f'pip install --upgrade "git+{github_home_url}.git@{commit_hash}"'
        command_github = None
    else:
        command = f'pip install --upgrade cellacdc=={version}'
        command_github = f'pip install --upgrade "git+{urls.github_url}@{version}"'
    return command, command_github  

def get_git_pull_checkout_cellacdc_version_commands(version=None):
    if version is None:
        version = read_version()
    commit_hash_idx = version.find('+g')
    is_dev_version = commit_hash_idx > 0 
    if not is_dev_version:
        return []
    commit_hash = version[commit_hash_idx+2:].split('.')[0]
    commands = (
        f'cd "{os.path.dirname(cellacdc_path)}"',
        'git pull',
        f'git checkout {commit_hash}'
    )
    return commands

def check_install_tapir():
    check_install_package(
        'tapnet', pypi_name='git+https://github.com/ElpadoCan/TAPIR.git'
    )

def _download_tapir_model():
    urls, file_sizes = _model_url('TAPIR')
    temp_model_path = tempfile.mkdtemp()
    _, final_model_path = (
        get_model_path('TAPIR', create_temp_dir=False)
    )
    for url, file_size in zip(urls, file_sizes):
        filename = url.split('/')[-1]
        final_dst = os.path.join(final_model_path, filename)
        if os.path.exists(final_dst):            
            continue

        temp_dst = os.path.join(temp_model_path, filename)
        download_url(
            url, temp_dst, file_size=file_size, desc='TAPIR',
            verbose=False
        )
        
        shutil.move(temp_dst, final_dst)

def _download_yeaz_models():
    urls, file_sizes = _model_url('YeaZ_v2')
    temp_model_path = tempfile.mkdtemp()
    _, final_model_path = (
        get_model_path('YeaZ_v2', create_temp_dir=False)
    )
    for url, file_size in zip(urls, file_sizes):
        filename = url.split('/')[-1]
        final_dst = os.path.join(final_model_path, filename)
        if os.path.exists(final_dst):            
            continue

        temp_dst = os.path.join(temp_model_path, filename)
        download_url(
            url, temp_dst, file_size=file_size, desc='YeaZ_v2',
            verbose=False
        )
        
        shutil.move(temp_dst, final_dst)

def _download_cellpose_germlineNuclei_model():
    urls, file_sizes = _model_url('Cellpose_germlineNuclei')
    temp_model_path = tempfile.mkdtemp()
    _, final_model_path = (
        get_model_path('Cellpose_germlineNuclei', create_temp_dir=False)
    )
    for url, file_size in zip(urls, file_sizes):
        filename = url.split('/')[-1]
        final_dst = os.path.join(final_model_path, filename)
        if os.path.exists(final_dst):            
            continue

        temp_dst = os.path.join(temp_model_path, filename)
        download_url(
            url, temp_dst, file_size=file_size, desc='Cellpose_germlineNuclei',
            verbose=False
        )
        
        shutil.move(temp_dst, final_dst)

def _download_omnipose_models():
    urls, file_sizes = _model_url('omnipose')
    temp_model_path = tempfile.mkdtemp()
    final_model_path = os.path.expanduser('~\.cellpose\models')
    for url, file_size in zip(urls, file_sizes):
        filename = url.split('/')[-1]
        final_dst = os.path.join(final_model_path, filename)
        if os.path.exists(final_dst):            
            continue

        temp_dst = os.path.join(temp_model_path, filename)
        download_url(
            url, temp_dst, file_size=file_size, desc='omnipose',
            verbose=False
        )
        
        shutil.move(temp_dst, final_dst)

def format_cca_manual_changes(changes: dict):
    txt = ''
    for ID, changes_ID in changes.items():
        txt = f'{txt}* ID {ID}:\n'
        for col, (old_val, new_val) in changes_ID.items():
            txt = f'{txt}    - {col}: {old_val} --> {new_val}\n'
        txt = f'{txt}--------------------------------\n\n'
    return txt

def init_segm_model(acdcSegment, posData, init_kwargs):
    segm_endname = init_kwargs.pop('segm_endname', 'None')
    if segm_endname != 'None':
        load_segm = True
        if not hasattr(posData, 'segm_data'):
            load_segm = True
        elif posData.segm_npz_path.endswith(f'{segm_endname}.npz'):
            load_segm = False
        if not load_segm:
            segm_data = np.squeeze(posData.segm_data)
        else:
            segm_filepath, _ = load.get_path_from_endname(
                segm_endname, posData.images_path
            )
            printl(f'Loading segmentation data from "{segm_filepath}"...')
            segm_data = np.load(segm_filepath)['arr_0']
    else:
        segm_data = None

    # Initialize input_points_df for SAM model
    input_points_filepath = init_kwargs.pop('input_points_path', '')
    if input_points_filepath:
        input_points_df = init_sam_input_points_df(
            posData, input_points_filepath
        )
        init_kwargs['input_points_df'] = input_points_df
    
    try:
        # Models introduced before 1.3.2 do not have the segm_data as input
        model = acdcSegment.Model(**init_kwargs)
    except Exception as e:
        model = acdcSegment.Model(segm_data, **init_kwargs)
    return model

def _parse_bool_str(value):
    if isinstance(value, bool):
        return value
    
    if value == 'True':
        return True
    elif value == 'False':
        return False

def check_install_trackastra():
    check_install_package(
        'Trackastra', 
        import_pkg_name='trackastra', 
        pypi_name='trackastra'
    )

def get_torch_device(gpu=False):
    import torch
    if torch.cuda.is_available() and gpu:
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device

def parse_model_params(model_argspecs, model_params):
    parsed_model_params = {}
    for row, argspec in enumerate(model_argspecs):
        value = model_params.get(argspec.name)
        if value is None:
            continue
        if argspec.type == bool:
            value = _parse_bool_str(value)
        elif argspec.type == int:
            value = int(value)
        elif argspec.type == float:
            value = float(value)
        parsed_model_params[argspec.name] = value
    return parsed_model_params

def init_cellpose_denoise_model():
    from . import apps
    
    from cellacdc.models.cellpose_v3._denoise import (
        CellposeDenoiseModel, url_help
    )

    init_argspecs, run_argspecs = getClassArgSpecs(CellposeDenoiseModel)
    url = url_help()
    
    paramsWin = apps.QDialogModelParams(
        init_argspecs, run_argspecs, 'Cellpose 3.0', 
        url=url, is_tracker=True, action_type='denoising'
    )
    paramsWin.exec_()
    if paramsWin.cancel:
        return
    
    init_params = paramsWin.init_kwargs
    run_params = paramsWin.model_kwargs
    denoise_model = CellposeDenoiseModel(**init_params)
    return denoise_model, init_params, run_params

def init_sam_input_points_df(posData, input_points_filepath):
    input_points_df = None
    if os.path.exists(input_points_filepath):
        input_points_df = pd.read_csv(input_points_filepath)
    else:
        # input_points_filepath is actually and endname
        for file in listdir(posData.images_path):
            if file.endswith(input_points_filepath):
                filepath = os.path.join(posData.images_path, file)
                input_points_df = pd.read_csv(filepath)
                break
    
    if input_points_df is None:
        raise FileNotFoundError(
            f'Could not find input points table from file "input_points_filepath" '
            'Perhaps, you forgot to save the table?'
        )
    
    for col in ('x', 'y', 'id'):
        if col not in input_points_df.columns:
            raise KeyError(
                f'Input points table is missing colum {col}. It must have '
                'the colums (x, y, id)'
            )
    
    return input_points_df

def are_acdc_dfs_equal(df_left, df_right):
    if df_left.shape != df_right.shape:
        return False
    
    try:
        for col in df_left.columns:
            if col not in df_right.columns:
                return False
            
            try:
                eq_mask = np.isclose(df_left[col], df_right[col], equal_nan=True)
            except Exception as err:
                # Data type is string
                eq_mask = df_left[col] == df_right[col]
            
            nan_mask = ((df_left[col].isna()) & (df_right[col].isna()))
            equality_mask = (eq_mask) | (nan_mask)
            if not equality_mask.all():
                return False
    except Exception as err:
        return False
    
    return True

def is_pos_folderpath(folderpath):
    foldername = os.path.basename(folderpath)
    is_valid_pos_folder = (
        re.search('^Position_(\d+)$', foldername) is not None
        and os.path.isdir(folderpath)
        and os.path.exists(os.path.join(folderpath, 'Images'))
    )
    return is_valid_pos_folder

def log_segm_params(
        model_name, init_params, segm_params, logger_func=print, 
        preproc_recipe=None, apply_post_process=False, 
        standard_postprocess_kwargs=None, custom_postprocess_features=None
    ):
    init_params_format = [
        f'  * {option} = {value}' for option, value in init_params.items()
    ]
    init_params_format = '\n'.join(init_params_format)
    
    segm_params_format = [
        f'  * {option} = {value}' for option, value in segm_params.items()
    ]
    segm_params_format = '\n'.join(segm_params_format)
    
    preproc_recipe_format = None
    if preproc_recipe is not None:
        preproc_recipe_format = []
        for s, step in enumerate(preproc_recipe):
            preproc_recipe_format.append(f'  * Step {s+1}')
            method = step['method']
            preproc_recipe_format.append(f'     - Method: {method}')
            for option, value in step['kwargs'].items():
                preproc_recipe_format.append(f'     - {option}: {value}')
        preproc_recipe_format = '\n'.join(preproc_recipe_format)    
    
    standard_postproc_format = None
    if apply_post_process and standard_postprocess_kwargs is not None:
        standard_postproc_format = [
            f'  * {option} = {value}' 
            for option, value in standard_postprocess_kwargs.items()
        ]
        standard_postproc_format = '\n'.join(standard_postproc_format)
    
    custom_postproc_format = None
    if apply_post_process and custom_postprocess_features is not None:
        custom_postproc_format = [
            f'  * {feature} = ({low}, {high})'
            for feature, (low, high) in custom_postprocess_features.items()
        ]
        custom_postproc_format = '\n'.join(custom_postproc_format)
    
    separator = '-'*100
    params_format = (
        f'{separator}\n'
        f'Model name: {model_name}\n\n'
        'Preprocessing recipe:\n\n'
        f'{preproc_recipe_format}\n\n'
        'Initialization parameters:\n\n'
        f'{init_params_format}\n\n'
        'Segmentation parameters:\n\n'
        f'{segm_params_format}\n\n'
        'Post-processing:\n\n'
        f'{standard_postproc_format}\n\n'
        'Custom post-processing:\n\n'
        f'{custom_postproc_format}\n'
        f'{separator}'
    )
    logger_func(params_format)

def pairwise(iterable):
    # pairwise('ABCDEFG') → AB BC CD DE EF FG
    iterator = iter(iterable)
    a = next(iterator, None)
    for b in iterator:
        yield a, b
        a = b

def append_text_filename(filename: str, text_to_append: str):
    filename_noext, ext = os.path.splitext(filename)
    filename_out = f'{filename_noext}{text_to_append}{ext}'
    return filename_out

def validate_images_path(input_path: os.PathLike, create_dirs_tree=False):
    is_images_path = input_path.endswith('Images')
    parent_dir = os.path.dirname(input_path)
    parent_foldername = os.path.basename(parent_dir)
    is_pos_folder = (
        re.search('^Position_(\d+)$', parent_foldername) is not None
        and os.path.isdir(parent_dir)
    )
    if not is_pos_folder:
        existing_pos_foldernames = get_pos_foldernames(input_path)
        pos_n = len(existing_pos_foldernames) + 1
        pos_folderpath = os.path.join(input_path, f'Position_{pos_n}')
        images_path = os.path.join(pos_folderpath, 'Images')
    elif is_images_path:
        pos_folderpath = input_path
        images_path = os.path.join(pos_folderpath, 'Images')
    else:
        images_path = input_path
    
    if create_dirs_tree:
        os.makedirs(images_path, exist_ok=True)
        
    return images_path

def fix_acdc_df_dtypes(acdc_df):
    acdc_df['is_cell_excluded'] = acdc_df['is_cell_excluded'].astype(bool)
    return acdc_df

def _relabel_cca_dfs_and_segm_data(
        cca_dfs,
        IDs_mapper,
        asymm_tracked_segm,
        progressbar=True,
    ):
    # Rename Cell_ID index according to asymmetric cell div convention
    if progressbar:
        pbar = tqdm(
            desc='Applying asymmetric division', 
            total=len(IDs_mapper), ncols=100
        )
    for key, root_ID in IDs_mapper.items():
        div_frame_i, daughter_ID = key
        for frame_i in range(div_frame_i, len(asymm_tracked_segm)):
            cca_dfs[frame_i].rename(
                index={daughter_ID: root_ID}, inplace=True
            )
            lab = asymm_tracked_segm[frame_i]
            rp = skimage.measure.regionprops(lab)
            obj_daught = [obj for obj in rp if obj.label == daughter_ID]
            if not obj_daught:
                continue
            
            obj_daught = obj_daught[0]
            lab[obj_daught.slice][obj_daught.image] = root_ID
        
        if progressbar:
            pbar.update()
    
    if progressbar:
        pbar.close()
    
def df_ctc_to_acdc_df(
        df_ctc, tracked_segm, cell_division_mode='Normal', return_list=False, 
        progressbar=True
    ):
    """Convert Cell Tracking Challenge DataFrame with annotated division to
    Cell-ACDC cell cycle annotations DataFrame.

    Parameters
    ----------
    df_ctc : pd.DataFrame
        DataFrame with {'label', 't1', 't2', 'parent'} columns where 
        't1' is the frame index of cell division.
    tracked_segm : (T, Y, X) array of ints
        Array of tracked segmentation labels.
    cell_division_mode : {'Normal', 'Asymmetric'}, optional
        Type of cell division. `Normal` is the standard cell division, 
        where the mother cell divides into two daughter cells. For the 
        tracking, that means the two daughter cells get a new, unique ID 
        each. 
        
        `Asymmetric` means that the mother cell grows one daughter 
        cell that eventually divides from the mother (e.g., budding yeast). 
        For the tracking, this means that the mother cell ID keeps 
        existing after division and the daughter cell gets a new, unique ID.
        
        If `Asymmetric`, the third returned element is the segmentation data 
        with the asymmetric Cell IDs.  
    return_list : bool, optional
        If `True`, the second returned element is the list of created dataframes, 
        one per frame. Default is False
    progressbar : bool, optional
        If `True`, displays a tqdm progressbar. Default is True
    """    
    cca_dfs = []
    keys = []
    df_ctc = df_ctc.set_index(['t1', 'parent'])
    
    if cell_division_mode == 'Asymmetric':
        asymm_tracked_segm = tracked_segm.copy()
    
    asymmetric_IDs_rename_mapper = {}
    if progressbar:
        pbar = tqdm(
            desc='Converting to Cell-ACDC format', 
            total=len(tracked_segm), ncols=100
        )
    for frame_i, lab in enumerate(tracked_segm):
        rp = skimage.measure.regionprops(lab)
        IDs = [obj.label for obj in rp]
        cca_df = core.getBaseCca_df(IDs, with_tree_cols=True)
        keys.append(frame_i)
        if frame_i == 0:
            cca_dfs.append(cca_df)
            if progressbar:
                pbar.update()
            continue
        
        # Copy annotations from previous frames
        prev_cca_df = cca_dfs[frame_i-1]
        old_IDs = cca_df.index.intersection(prev_cca_df.index)
        cca_df.loc[old_IDs] = prev_cca_df.loc[old_IDs]
        
        try:
            df_ctc_i = df_ctc.loc[frame_i]
        except KeyError as err:
            # No division detected --> nothing to annotate
            cca_dfs.append(cca_df)
            if progressbar:
                pbar.update()
            continue
        
        for parent_ID, df_ctc_i_pID in df_ctc_i.groupby(level=0):
            daughter_IDs = df_ctc_i_pID['label'].to_list()     
            
            if parent_ID == 0:
                continue
            
            cca_df.loc[daughter_IDs, 'parent_ID_tree'] = parent_ID
            cca_df.loc[daughter_IDs, 'emerg_frame_i'] = frame_i
            cca_df.loc[daughter_IDs, 'division_frame_i'] = frame_i
            
            root_ID = prev_cca_df.at[parent_ID, 'root_ID_tree']
            if root_ID == -1:
                root_ID = parent_ID
            cca_df.loc[daughter_IDs, 'root_ID_tree'] = root_ID
            
            cca_df.loc[daughter_IDs[0], 'sister_ID_tree'] = daughter_IDs[1]
            cca_df.loc[daughter_IDs[1], 'sister_ID_tree'] = daughter_IDs[0]
            
            prev_gen_num = prev_cca_df.loc[parent_ID, 'generation_num_tree']
            cca_df.loc[daughter_IDs, 'generation_num_tree'] = prev_gen_num + 1
            
            # Annotate division from df_ctc_i into 
            if cell_division_mode == 'Asymmetric':
                # Recycle the root_ID and assign it to one of the daughters
                replaced_daught_ID = daughter_IDs[1]
                key = (frame_i, replaced_daught_ID)
                asymmetric_IDs_rename_mapper[key] = root_ID    
        
        cca_dfs.append(cca_df)
        
        if progressbar:
            pbar.update()
    
    if progressbar:
        pbar.close()
    if asymmetric_IDs_rename_mapper:
        _relabel_cca_dfs_and_segm_data(
            cca_dfs,
            asymmetric_IDs_rename_mapper,
            asymm_tracked_segm,
            progressbar=True,
        )
    
    cca_df = pd.concat(cca_dfs, keys=keys, names=['frame_i'])
    
    out = [cca_df, None, None]
    
    if return_list:
        out[1] = cca_dfs
        
    if cell_division_mode == 'Asymmetric':
        out[2] = asymm_tracked_segm
    
    return out
        