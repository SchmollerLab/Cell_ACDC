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
import natsort
from tqdm import tqdm
import requests
import zipfile
import numpy as np
import pandas as pd
import skimage
from distutils.dir_util import copy_tree
import inspect
import typing

from natsort import natsorted

from tifffile.tifffile import TiffWriter, TiffFile

from . import GUI_INSTALLED

if GUI_INSTALLED:
    from qtpy.QtWidgets import QMessageBox
    from qtpy.QtCore import Signal, QObject, QCoreApplication
    
    from . import widgets
    from . import config
    
from . import core, load
from . import html_utils, is_linux, is_win, is_mac, issues_url, is_mac_arm64
from . import cellacdc_path, printl, settings_folderpath, logs_path
from . import user_profile_path, recentPaths_path
from . import models_list_file_path
from . import github_home_url

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
    time_end_afternoon = datetime.time(15,00,00)
    time_end_evening = datetime.time(20,00,00)
    time_end_night = datetime.time(4,00,00)
    if time_now >= time_end_night and time_now < time_end_morning:
        return 'Have a good day!'
    elif time_now >= time_end_morning and time_now < time_end_afternoon:
        return 'Have a good afternoon!'
    elif time_now >= time_end_afternoon and time_now < time_end_evening:
        return 'Have a good evening!'
    else:
        return 'Have a good night!'

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

def setupLogger(module='gui', logs_path=None):
    if logs_path is None:
        logs_path = get_logs_path()
    
    logger = logging.getLogger(f'cellacdc-logger-{module}')
    logger.setLevel(logging.INFO)

    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    else:
        # Keep 20 most recent logs
        ls = listdir(logs_path)
        if len(ls)>20:
            ls = [os.path.join(logs_path, f) for f in ls]
            ls.sort(key=lambda x: os.path.getmtime(x))
            for file in ls[:-20]:
                try:
                    os.remove(file)
                except Exception as e:
                    pass
    
    logger.default_stdout = sys.stdout

    date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    id = uuid4()
    log_filename = f'.{date_time}_{module}_{id}_stdout.log'
    log_path = os.path.join(logs_path, log_filename)

    output_file_handler = logging.FileHandler(log_path, mode='w')
    stdout_handler = logging.StreamHandler(sys.stdout)

    # Format your logs (optional)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s:\n'
        '------------------------\n'
        '%(message)s\n'
        '------------------------\n',
        datefmt='%d-%m-%Y, %H:%M:%S')
    output_file_handler.setFormatter(formatter)

    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    
    if module == 'gui':
        qt_handler = widgets.QtHandler()
        qt_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(qt_handler)

    return logger, logs_path, log_path, log_filename

def get_pos_foldernames(exp_path):
    ls = listdir(exp_path)
    pos_foldernames = [
        pos for pos in ls if pos.find('Position_')!=-1
        and os.path.isdir(os.path.join(exp_path, pos))
        and os.path.exists(os.path.join(exp_path, pos, 'Images'))
    ]
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
            'The system detected files inside the folder '
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
        subprocess.check_call(['git', '--version'])
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

def install_java():
    try:
        subprocess.check_call(['javac', '-version'])
        return False
    except Exception as e:
        from . import apps
        win = apps.installJavaDialog()
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
    for obj in rp:
        xc, yc = obj.centroid[-2:]
        IDs.append(obj.label)
        xx_centroid.append(xc)
        yy_centroid.append(yc)
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


def listdir(path):
    return natsorted([
        f for f in os.listdir(path)
        if not f.startswith('.')
        and not f == 'desktop.ini'
        and not f == 'recovery'
    ])

def insertModelArgSpect(params, param_name, param_value, param_type=None):
    ArgSpec = namedtuple('ArgSpec', ['name', 'default', 'type'])
    
    updated_params = []
    for param in params:
        if param.name == param_name:
            if param_type is None:
                param_type = param.type
            new_param = ArgSpec(
                name=param_name, default=param_value, type=param_type
            )
            updated_params.append(new_param)
        else:
            updated_params.append(param)
    return updated_params

def getModelArgSpec(acdcSegment):
    ArgSpec = namedtuple('ArgSpec', ['name', 'default', 'type'])

    init_ArgSpec = inspect.getfullargspec(acdcSegment.Model.__init__)
    init_kwargs_type_hints = typing.get_type_hints(acdcSegment.Model.__init__)
    try:
        init_ArgSpec.args.remove('segm_data')
    except Exception as e:
        pass
    init_params = []
    if len(init_ArgSpec.args)>1:
        for arg, default in zip(init_ArgSpec.args[1:], init_ArgSpec.defaults):
            if arg in init_kwargs_type_hints:
                _type = init_kwargs_type_hints[arg]
            else:
                _type = type(default)
            param = ArgSpec(name=arg, default=default, type=_type)
            init_params.append(param)

    segment_ArgSpec = inspect.getfullargspec(acdcSegment.Model.segment)
    segment_kwargs_type_hints = typing.get_type_hints(acdcSegment.Model.segment)
    try:
        segment_ArgSpec.args.remove('frame_i')
    except Exception as e:
        pass
    
    segment_params = []
    if len(segment_ArgSpec.args)>2:
        iter = zip(segment_ArgSpec.args[2:], segment_ArgSpec.defaults)
        for arg, default in iter:
            if arg in segment_kwargs_type_hints:
                _type = segment_kwargs_type_hints[arg]
            else:
                _type = type(default)
            param = ArgSpec(name=arg, default=default, type=_type)
            segment_params.append(param)
    return init_params, segment_params

def getTrackerArgSpec(trackerModule, realTime=False):
    ArgSpec = namedtuple('ArgSpec', ['name', 'default', 'type'])

    init_ArgSpec = inspect.getfullargspec(trackerModule.tracker.__init__)
    init_kwargs_type_hints = typing.get_type_hints(
        trackerModule.tracker.__init__
    )
    init_params = []
    if len(init_ArgSpec.args)>1 and init_ArgSpec.defaults is not None:
        for arg, default in zip(init_ArgSpec.args[1:], init_ArgSpec.defaults):
            if arg in init_kwargs_type_hints:
                _type = init_kwargs_type_hints[arg]
            else:
                _type = type(default)
            param = ArgSpec(name=arg, default=default, type=_type)
            init_params.append(param)

    if realTime:
        track_ArgSpec = inspect.getfullargspec(trackerModule.tracker.track_frame)
    else:
        track_ArgSpec = inspect.getfullargspec(trackerModule.tracker.track)
    track_params = []
    track_kwargs_type_hints = typing.get_type_hints(
        trackerModule.tracker.track
    )
    # Start at first kwarg
    if len(track_ArgSpec.args)>1 and track_ArgSpec.defaults is not None:
        kwargs_start_idx = len(track_ArgSpec.args) - len(track_ArgSpec.defaults)
        iter = zip(
            track_ArgSpec.args[kwargs_start_idx:], track_ArgSpec.defaults
        )
        for arg, default in iter:
            if arg == 'signals':
                continue
            if arg == 'export_to':
                continue
            if arg in track_kwargs_type_hints:
                _type = track_kwargs_type_hints[arg]
            else:
                _type = type(default)
            param = ArgSpec(name=arg, default=default, type=_type)
            track_params.append(param)
    return init_params, track_params

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

def check_upgrade_javabridge():
    import pkg_resources
    try:
        version = pkg_resources.get_distribution("javabridge").version
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
            url, temp_dst, file_size=file_size, desc='segment_anything',
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

def imagej_tiffwriter(
        new_path, data, metadata: dict=None, SizeT=None, SizeZ=None,
        imagej=True
    ):
    if data.dtype != np.uint8 and data.dtype != np.uint16:
        data = scale_float(data)
        data = skimage.img_as_uint(data)
    with TiffWriter(new_path, bigtiff=True) as new_tif:
        new_tif.save(data)

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
        

def from_lab_to_imagej_rois(lab, ImagejRoi, t=0, SizeT=1, max_ID=None):
    if max_ID is None:
        max_ID = lab.max()
    rois = []
    if lab.ndim == 3:
        SizeZ = len(lab)
        for z, lab2D in enumerate(lab):
            rp = skimage.measure.regionprops(lab2D)
            for obj in rp:
                cont = core.get_obj_contours(obj)
                t_str = str(t).zfill(len(str(SizeT)))
                z_str = str(z).zfill(len(str(SizeZ)))
                id_str = str(obj.label).zfill(len(str(max_ID)))
                name = f't={t_str}-z={z_str}-id={id_str}'
                roi = ImagejRoi.frompoints(
                    cont, name=name, t=t
                )
                rois.append(roi)
    else:
        rp = skimage.measure.regionprops(lab)
        for obj in rp:
            cont = core.get_obj_contours(obj)
            t_str = str(t).zfill(len(str(SizeT)))
            id_str = str(obj.label).zfill(len(str(max_ID)))
            name = f't={t_str}-id={id_str}'
            roi = ImagejRoi.frompoints(
                cont, name=name, t=t
            )
            rois.append(roi)
    return rois

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
        if os.path.isdir(_path) and not name.endswith('__') and name != 'thresholding':
            models.add(name)
        if name == 'thresholding':
            models.add('Automatic thresholding')
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

def img_to_float(img):
    img_max = np.max(img)
    # Check if float outside of -1, 1
    if img_max <= 1:
        return img

    uint8_max = np.iinfo(np.uint8).max
    uint16_max = np.iinfo(np.uint16).max
    if img_max <= uint8_max:
        img = img/uint8_max
    elif img_max <= uint16_max:
        img = img/uint16_max
    else:
        img = img/img_max
    return img

def scale_float(data):
    val = data[tuple([0]*data.ndim)]
    if isinstance(val, (np.floating, float)):
        data = img_to_float(data)
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

def check_install_cellpose():
    check_install_package('cellpose')
    try:
        import pkg_resources
        version = pkg_resources.get_distribution("cellpose").version
        major = int(version.split('.')[0])
        if major < 2:
            _install_pip_package('cellpose')
    except Exception as e:
        printl(traceback.format_exc())
        _inform_install_package_failed('cellpose')

def check_install_yeaz():
    check_install_package('torch')
    check_install_package('yeaz')

def check_install_segment_anything():
    check_install_package('torchvision')
    check_install_package('segment_anything')

def check_install_package(
        pkg_name: str, pypi_name='', note='', parent=None, 
        raise_on_cancel=True
    ):
    try:
        import_module(pkg_name)
    except ModuleNotFoundError:
        if pypi_name:
            pkg_name = pypi_name
        cancel = _install_package_msg(pkg_name, note=note, parent=parent)
        if cancel:
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
    import pkg_resources
    mpl_version = pkg_resources.get_distribution("matplotlib").version
    mpl_version_digits = mpl_version.split('.')

    mpl_version = float(f'{mpl_version_digits[0]}.{mpl_version_digits[1]}')
    if mpl_version < 3.5:
        cancel = _install_package_msg('matplotlib', parent=qparent, upgrade=True)
        if cancel:
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

def _install_package_msg(
        pkg_name, note='', parent=None, upgrade=False, caller_name='Cell-ACDC'
    ):
    msg = widgets.myMessageBox(parent=parent)
    if upgrade:
        install_text = 'upgrade'
    else:
        install_text = 'install'
    if pkg_name == 'BayesianTracker':
        pkg_name = 'btrack'
    txt = html_utils.paragraph(f"""
        {caller_name} is going to <b>download and {install_text}</b>
        <code>{pkg_name}</code>.<br><br>
        Make sure you have an <b>active internet connection</b>,
        before continuing.<br>
        Progress will be displayed on the terminal<br><br>
        You might have to <b>restart {caller_name}</b>.<br><br>
        <b>IMPORTANT:</b> If the installation fails please install
        <code>{pkg_name}</code> manually with the follwing command:<br><br>
        <code>pip install --upgrade {pkg_name.lower()}</code><br><br>
        Alternatively, you can cancel the process and try later.
    """)
    if note:
        txt = f'{txt}{note}'
    msg.setIcon()
    msg.setWindowTitle(f'Install {pkg_name}')
    msg.addText(txt)
    msg.addButton('   Ok   ')
    cancel = msg.addButton(' Cancel ')
    msg.exec_()
    return msg.clickedButton == cancel

def _install_tensorflow():
    cpu = platform.processor()
    if is_mac and cpu == 'arm':
        args = ['conda', 'install', '-y', '-c', 'conda-forge', 'tensorflow']
    else:
        args = [sys.executable, '-m', 'pip', 'install', '-U', 'tensorflow']
    subprocess.check_call(args)

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
            paramsWin = apps.QDialogModelParams(
                init_argspecs, track_argspecs, trackerName, url=url,
                channels=channels, is_tracker=True,
                currentChannelName=currentChannelName,
                df_metadata=df_metadata
            )
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
    else:
        command = f'pip install --upgrade cellacdc=={version}'
    return command

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
            url, temp_dst, file_size=file_size, desc='TAPIR',
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
            segm_data = np.load(segm_filepath)['arr_0']
    else:
        segm_data = None
    
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

def parse_model_params(model_argspecs, model_params):
    parsed_model_params = {}
    for row, argspec in enumerate(model_argspecs):
        value = model_params[argspec.name]
        if argspec.type == bool:
            value = _parse_bool_str(value)
        elif argspec.type == int:
            value = int(value)
        elif argspec.type == float:
            value = float(value)
        parsed_model_params[argspec.name] = value
    return parsed_model_params