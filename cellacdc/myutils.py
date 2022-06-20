import os
import re
import pathlib
import difflib
import sys
import tempfile
import shutil
import traceback
import logging
import datetime
import time
import subprocess
from math import pow
from functools import wraps, partial
from collections import namedtuple, Counter
from collections.abc import Callable, Sequence
from tqdm import tqdm
import requests
import zipfile
import numpy as np
import pandas as pd
import skimage
from distutils.dir_util import copy_tree
from pyqtgraph.colormap import ColorMap
import inspect
import matplotlib.colors
import colorsys

from natsort import natsorted

from tifffile.tifffile import TiffWriter, TiffFile

from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal, QObject, QCoreApplication

from . import prompts, widgets, apps, core, load
from . import html_utils, is_linux, is_win, is_mac

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
            result = None
            traceback_str = traceback.format_exc()
            self.logger.exception(traceback_str)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            msg.addShowInFileManagerButton(self.logs_path, txt='Show log file...')
            msg.setDetailedText(traceback_str, visible=True)
            err_msg = html_utils.paragraph(f"""
                Error in function <code>{func.__name__}</code>.<br><br>
                More details below or in the terminal/console.<br><br>
                Note that the <b>error details</b> from this session are
                also <b>saved in the following file</b>:<br><br>
                {self.log_path}<br><br>
                Please <b>send the log file</b> when reporting a bug, thanks!
            """)

            msg.critical(self, 'Critical error', err_msg)
            self.is_error_state = True
        return result
    return inner_function

def get_gdrive_path():
    if is_win:
        return os.path.join(f'G:{os.sep}', 'My Drive')
    elif is_mac:
        return os.path.join('/Volumes/GoogleDrive/My Drive')

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
    if time_now < time_end_morning:
        return 'Have a good day!'
    elif time_now < time_end_morning:
        return 'Have a good afternoon!'
    elif time_now < time_end_evening:
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

def is_iterable(item):
     try:
         iter(item)
         return True
     except TypeError as e:
         return False

__all__ = ['ColorMap']
_mapCache = {}

def lighten_color(color, amount=0.3, hex=True):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """

    try:
        c = matplotlib.colors.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*matplotlib.colors.to_rgb(c))
    lightened_c = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
    if hex:
        lightened_c = tuple([int(round(v*255)) for v in lightened_c])
        lightened_c = '#%02x%02x%02x' % lightened_c
    return lightened_c

class utilClass:
    pass

class signals(QObject):
    progressBar = pyqtSignal(int)
    progress = pyqtSignal(str)

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

def setupLogger(module='gui'):
    logger = logging.getLogger(f'cellacdc-logger-{module}')
    logger.setLevel(logging.INFO)

    user_path = pathlib.Path.home()
    logs_path = os.path.join(user_path, '.acdc-logs')
    if not os.path.exists(logs_path):
        os.mkdir(logs_path)
    else:
        # Keep 20 most recent logs
        ls = listdir(logs_path)
        if len(ls)>20:
            ls = [os.path.join(logs_path, f) for f in ls]
            ls.sort(key=lambda x: os.path.getmtime(x))
            for file in ls[:-20]:
                os.remove(file)

    date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_filename = f'{date_time}_{module}_stdout.log'
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

    return logger, logs_path, log_path, log_filename

def get_pos_foldernames(exp_path):
    ls = listdir(exp_path)
    pos_foldernames = [
        pos for pos in ls if pos.find('Position_')!=-1
        and os.path.isdir(os.path.join(exp_path, pos))
    ]
    return pos_foldernames

def getMostRecentPath():
    cellacdc_path = os.path.dirname(os.path.abspath(__file__))
    recentPaths_path = os.path.join(
        cellacdc_path, 'temp', 'recentPaths.csv'
    )
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
    cellacdc_path = os.path.dirname(os.path.abspath(__file__))
    recentPaths_path = os.path.join(
        cellacdc_path, 'temp', 'recentPaths.csv'
    )
    if os.path.exists(recentPaths_path):
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
    else:
        recentPaths = [exp_path]
        openedOn = [datetime.datetime.now()]
    df = pd.DataFrame({
        'path': recentPaths,
        'opened_last_on': pd.Series(openedOn, dtype='datetime64[ns]')}
    )
    df.index.name = 'index'
    df.to_csv(recentPaths_path)

def rgb_str_to_values(rgbString, errorRgb=(255,255,255)):
    try:
        r, g, b = re.findall(r'(\d+), (\d+), (\d+)', rgbString)[0]
        r, g, b = int(r), int(g), int(b)
    except TypeError:
        try:
            r, g, b = rgbString
        except Exception as e:
            print('======================')
            traceback.print_exc()
            print('======================')
            r, g, b = errorRgb
    return r, g, b

def rgba_str_to_values(rgbaString, errorRgb=(255,255,255,255)):
    try:
        r, g, b, a = re.findall(r'(\d+), (\d+), (\d+), (\d+)', rgbaString)[0]
        r, g, b, a = int(r), int(g), int(b), int(a)
    except TypeError:
        try:
            r, g, b, a = rgbaString
        except Exception as e:
            r, g, b, a = errorRgb
    return r, g, b, a

def checkDataIntegrity(filenames, parent_path, parentQWidget=None):
    char = filenames[0][:2]
    startWithSameChar = all([f.startswith(char) for f in filenames])
    if not startWithSameChar:
        msg = QMessageBox(parentQWidget)
        msg.setWindowTitle('Data structure compromised')
        msg.setIcon(msg.Warning)
        txt = (
            'The system detected files inside the folder '
            'that do not start with the same, common basename.\n\n'
            'To ensure correct loading of the data, the folder where '
            'the file(s) is/are should either contain a single image file or'
            'only files that start with the same, common basename.\n\n'
            'For example the following filenames:\n\n'
            'F014_s01_phase_contr.tif\n'
            'F014_s01_mCitrine.tif\n\n'
            'are named correctly since they all start with the '
            'the common basename "F014_s01_". After the common basename you '
            'can write whatever text you want. In the example above, "phase_contr" '
            'and "mCitrine" are the channel names.\n\n'
            'Data loading may still be successfull, so Cell-ACDC will '
            'still try to load data now.'
        )
        msg.setText(txt)
        _ls = "\n".join(filenames)
        msg.setDetailedText(
            f'Files present in the folder {parent_path}:\n'
            f'{_ls}'
        )
        msg.addButton(msg.Ok)
        openFolderButton = msg.addButton('Open folder...', msg.HelpRole)
        openFolderButton.disconnect()
        slot = partial(showInExplorer, parent_path)
        openFolderButton.clicked.connect(slot)
        msg.exec_()
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
                [sys.executable, '-m', 'pip', 'install',
                'git+https://github.com/SchmollerLab/python-javabridge-acdc']
            )
        else:
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install',
                'git+https://github.com/SchmollerLab/python-javabridge-windows']
            )
    elif is_mac:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/SchmollerLab/python-javabridge-acdc']
        )
    elif is_linux:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/LeeKamentsky/python-javabridge.git@master']
        )

def is_in_bounds(x,y,X,Y):
    in_bounds = x >= 0 and x < X and y >= 0 and y < Y
    return in_bounds

def read_version():
    try:
        from setuptools_scm import get_version
        version = get_version(root='..', relative_to=__file__)
        return version
    except Exception as e:
        try:
            from . import _version
            return _version.version
        except Exception as e:
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
        print(f'{func.__name__} exec time = {(t1-t0)*1000:.3f} ms')
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
    for file in listdir(images_path):
        filePath = os.path.join(images_path, file)
        if re.search(fr'{chName}.(\w+)$', file) is not None:
            return filePath
    return filePath

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

def getBasenameAndChNames(images_path):
    _tempPosData = utilClass()
    _tempPosData.images_path = images_path
    load.loadData.getBasenameAndChNames(_tempPosData)
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


def listdir(path):
    return natsorted([
        f for f in os.listdir(path)
        if not f.startswith('.')
        and not f == 'desktop.ini'
    ])

def getModelArgSpec(acdcSegment):
    ArgSpec = namedtuple('ArgSpec', ['name', 'default', 'type'])

    init_ArgSpec = inspect.getfullargspec(acdcSegment.Model.__init__)
    init_params = []
    if len(init_ArgSpec.args)>1:
        for arg, default in zip(init_ArgSpec.args[1:], init_ArgSpec.defaults):
            param = ArgSpec(name=arg, default=default, type=type(default))
            init_params.append(param)

    segment_ArgSpec = inspect.getfullargspec(acdcSegment.Model.segment)
    segment_params = []
    if len(segment_ArgSpec.args)>1:
        iter = zip(segment_ArgSpec.args[2:], segment_ArgSpec.defaults)
        for arg, default in iter:
            param = ArgSpec(name=arg, default=default, type=type(default))
            segment_params.append(param)
    return init_params, segment_params

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

    user_path = pathlib.Path.home()
    examples_path = os.path.join(user_path, 'acdc-examples')
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
        os.makedirs(examples_path)

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
    user_path = str(pathlib.Path.home())
    acdc_java_path = os.path.join(user_path, 'acdc-java')
    dot_acdc_java_path = os.path.join(user_path, '.acdc-java')
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
        url = 'https://github.com/SchmollerLab/java_portable_windows/archive/refs/tags/v0.1.zip'
    elif is_mac:
        os_foldername = 'macOS'
        unzipped_foldername = 'java_portable_macos-0.1'
        url = 'https://github.com/SchmollerLab/java_portable_macos/archive/refs/tags/v0.1.zip'
        file_size = 108478751
    elif is_linux:
        os_foldername = 'linux'
        unzipped_foldername = 'java_portable_linux-0.1'
        url = 'https://github.com/SchmollerLab/java_portable_linux/archive/refs/tags/v0.1.zip'
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
        os.makedirs(os_acdc_java_path)

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

def getFromMatplotlib(name):
    """
    Added to pyqtgraph 0.12 copied/pasted here to allow pyqtgraph <0.12. Link:
    https://pyqtgraph.readthedocs.io/en/latest/_modules/pyqtgraph/colormap.html#get
    Generates a ColorMap object from a Matplotlib definition.
    Same as ``colormap.get(name, source='matplotlib')``.
    """
    # inspired and informed by "mpl_cmaps_in_ImageItem.py", published by Sebastian Hoefer at
    # https://github.com/honkomonk/pyqtgraph_sandbox/blob/master/mpl_cmaps_in_ImageItem.py
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None
    cm = None
    col_map = plt.get_cmap(name)
    if hasattr(col_map, '_segmentdata'): # handle LinearSegmentedColormap
        data = col_map._segmentdata
        if ('red' in data) and isinstance(data['red'], (Sequence, np.ndarray)):
            positions = set() # super-set of handle positions in individual channels
            for key in ['red','green','blue']:
                for tup in data[key]:
                    positions.add(tup[0])
            col_data = np.zeros((len(positions),4 ))
            col_data[:,-1] = sorted(positions)
            for idx, key in enumerate(['red','green','blue']):
                positions = np.zeros( len(data[key] ) )
                comp_vals = np.zeros( len(data[key] ) )
                for idx2, tup in enumerate( data[key] ):
                    positions[idx2] = tup[0]
                    comp_vals[idx2] = tup[1] # these are sorted in the raw data
                col_data[:,idx] = np.interp(col_data[:,3], positions, comp_vals)
            cm = ColorMap(pos=col_data[:,-1], color=255*col_data[:,:3]+0.5)
        # some color maps (gnuplot in particular) are defined by RGB component functions:
        elif ('red' in data) and isinstance(data['red'], Callable):
            col_data = np.zeros((64, 4))
            col_data[:,-1] = np.linspace(0., 1., 64)
            for idx, key in enumerate(['red','green','blue']):
                col_data[:,idx] = np.clip( data[key](col_data[:,-1]), 0, 1)
            cm = ColorMap(pos=col_data[:,-1], color=255*col_data[:,:3]+0.5)
    elif hasattr(col_map, 'colors'): # handle ListedColormap
        col_data = np.array(col_map.colors)
        cm = ColorMap(pos=np.linspace(0.0, 1.0, col_data.shape[0]),
                      color=255*col_data[:,:3]+0.5 )
    if cm is not None:
        cm.name = name
        _mapCache[name] = cm
    return cm

def get_model_path(model_name, create_temp_dir=True):
    cellacdc_path = os.path.dirname(os.path.realpath(__file__))
    model_info_path = os.path.join(cellacdc_path, 'models', model_name, 'model')

    if os.path.exists(model_info_path):
        for file in listdir(model_info_path):
            if file == 'weights_location_path.txt':
                with open(os.path.join(model_info_path, file), 'r') as txt:
                    model_path = txt.read()
                if not os.path.exists(model_path):
                    model_path = _write_model_location_to_txt(model_name)
                else:
                    break
        else:
            model_path = _write_model_location_to_txt(model_name)
    else:
        os.makedirs(model_info_path)
        model_path = _write_model_location_to_txt(model_name)

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if not create_temp_dir:
        return '', model_path

    exists = check_model_exists(model_path, model_name)
    if exists:
        return '', model_path

    temp_zip_path = _create_temp_dir()
    return temp_zip_path, model_path


def check_model_exists(model_path, model_name):
    import cellacdc
    m = model_name.lower()
    weights_filenames = getattr(cellacdc, f'{m}_weights_filenames')
    files_present = listdir(model_path)
    return all([f in files_present for f in weights_filenames])

def _create_temp_dir():
    temp_model_path = tempfile.mkdtemp()
    temp_zip_path = os.path.join(temp_model_path, 'model_temp.zip')
    return temp_zip_path

def _model_url(model_name, return_alternative=False):
    if model_name == 'YeaZ':
        url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/CnfxkQtdRQm5MrT/download/YeaZ_weights.zip'
        alternative_url = 'https://zenodo.org/record/6127658/files/yeastmate_weights.zip?download=1'
        file_size = 693685011
    elif model_name == 'YeastMate':
        url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/czTkPmZReGjDRjG/download/yeastmate_weights.zip'
        alternative_url = 'https://zenodo.org/record/6140067/files/yeastmate_weights.zip?download=1'
        file_size = 164911104
    if return_alternative:
        return url, alternative_url
    else:
        return url, file_size

def download_manual():
    user_path = pathlib.Path.home()
    manual_folder_path = os.path.join(user_path, 'acdc-manual')
    if not os.path.exists(manual_folder_path):
        os.makedirs(manual_folder_path)

    manual_file_path = os.path.join(user_path, 'Cell-ACDC_User_Manual.pdf')
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
    CHUNK_SIZE = 32768
    if verbose:
        print(f'Downloading {desc} to: {os.path.dirname(dst)}')
    response = requests.get(url, stream=True, timeout=20)
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
        os.makedirs(destination_dir)
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
    cellacdc_path = os.path.dirname(os.path.realpath(__file__))
    v123_model_path = os.path.join(cellacdc_path, 'models', model_name, 'model')
    exists = check_model_exists(v123_model_path, model_name)
    if exists:
        return v123_model_path
    else:
        return ''

def _write_model_location_to_txt(model_name):
    cellacdc_path = os.path.dirname(os.path.realpath(__file__))
    model_info_path = os.path.join(cellacdc_path, 'models', model_name, 'model')
    user_path = pathlib.Path.home()
    model_path = os.path.join(str(user_path), f'acdc-{model_name}')
    file = 'weights_location_path.txt'
    with open(os.path.join(model_info_path, file), 'w') as txt:
        txt.write(model_path)
    return model_path


def download_model(model_name):
    if model_name != 'YeastMate' and model_name != 'YeaZ':
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
        new_path, data, metadata: dict, SizeT, SizeZ,
        imagej=True
    ):
    if data.dtype != np.uint8 and data.dtype != np.uint16:
        data = scale_float(data)
        data = skimage.img_as_uint(data)
    with TiffWriter(new_path, imagej=imagej) as new_tif:
        if not imagej:
            new_tif.save(data)
            return

        if SizeZ > 1 and SizeT > 1:
            # 3D data over time
            T, Z, Y, X = data.shape
        elif SizeZ == 1 and SizeT > 1:
            # 2D data over time
            T, Y, X = data.shape
            Z = 1
        elif SizeZ > 1 and SizeT == 1:
            # Single 3D data
            Z, Y, X = data.shape
            T = 1
        elif SizeZ == 1 and SizeT == 1:
            # Single 2D data
            Y, X = data.shape
            T, Z = 1, 1
        data.shape = T, Z, 1, Y, X, 1  # imageJ format should always have TZCYXS data shape
        if metadata is None:
            metadata = {}
        new_tif.save(data, metadata=metadata)

def get_list_of_trackers():
    cellacdc_path = os.path.dirname(os.path.abspath(__file__))
    trackers_path = os.path.join(cellacdc_path, 'trackers')
    trackers = []
    for name in listdir(trackers_path):
        _path = os.path.join(trackers_path, name)
        if os.path.isdir(_path) and not name.endswith('__'):
            trackers.append(name)
    return trackers

def get_list_of_models():
    cellacdc_path = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(cellacdc_path, 'models')
    models = []
    for name in listdir(models_path):
        _path = os.path.join(models_path, name)
        if os.path.isdir(_path) and not name.endswith('__'):
            models.append(name)
    return models

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
    img = np.round(uint_to_float(img)*255).astype(np.uint8)
    return img

def uint_to_float(img):
    img_max = core.numba_max(img)
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
        data = uint_to_float(data)
    return data

def _install_homebrew_command():
    return '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'

def _brew_install_java_command():
    return 'brew install --cask homebrew/cask-versions/adoptopenjdk8'

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
    <a href="https://hmgubox2.helmholtz-muenchen.de/index.php/s/7xF7YnArwbt9ZqB">
        here
    </a>.
    """)
    return s1, s2, s3, s4

def jdk_windows_url():
    return 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/zocneD2j2wMwbNc'

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

def install_package_msg(pkg_name, note='', parent=None):
    msg = widgets.myMessageBox()
    if pkg_name == 'BayesianTracker':
        pkg_name = 'btrack'
    txt = html_utils.paragraph(f"""
    <p>
        Cell-ACDC is going to <b>download and install</b>
        <code>{pkg_name}</code>.<br><br>
        Make sure you have an <b>active internet connection</b>,
        before continuing.
        Progress will be displayed on the terminal<br><br>
        You might have to <b>restart Cell-ACDC</b>.<br><br>
        <b>IMPORTANT:</b> If the installation fails please install
        <code>{pkg_name}</code> manually with the follwing command:<br><br>
        <code>pip install {pkg_name.lower()}</code><br><br>
        Alternatively, you can cancel the process and try later.
    </p>
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

if __name__ == '__main__':
    print(get_list_of_models())
    # model_name = 'cellpose'
    # download_model(model_name)
    #
    # download_examples()
