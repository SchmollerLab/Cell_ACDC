import os
import re
import pathlib
import difflib
import sys
import tempfile
import shutil
import traceback
import datetime
import time
import subprocess
from functools import wraps, partial
from collections import namedtuple
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

from natsort import natsorted

from tifffile.tifffile import TiffWriter, TiffFile

from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal, QObject, QCoreApplication

from . import prompts

__all__ = ['ColorMap']
_mapCache = {}

class utilClass:
    pass

class signals(QObject):
    progressBar = pyqtSignal(int)
    progress = pyqtSignal(str)

def rgb_str_to_values(rgbString, errorRgb=(255,255,255)):
    try:
        r, g, b = re.findall('(\d+), (\d+), (\d+)', rgbString)[0]
        r, g, b = int(r), int(g), int(b)
    except TypeError:
        traceback.print_exc()
        r, g, b = errorRgb
    return r, g, b

def rgba_str_to_values(rgbString, errorRgb=(255,255,255,255)):
    try:
        r, g, b, a = re.findall('(\d+), (\d+), (\d+), (\d+)', rgbString)[0]
        r, g, b, a = int(r), int(g), int(b), int(a)
    except TypeError:
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

def testQcoreApp():
    print(QCoreApplication.instance())

def install_javabridge():
    if sys.platform.startswith('win'):
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/SchmollerLab/python-javabridge-windows']
        )
    else:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install',
            'git+https://github.com/SchmollerLab/python-javabridge-acdc']
        )

def is_in_bounds(x,y,X,Y):
    in_bounds = x >= 0 and x < X and y >= 0 and y < Y
    return in_bounds

def read_version():
    try:
        from . import _version
        return _version.version
    except Exception as e:
        return 'ND'

def showInExplorer(path):
    if os.name == 'posix' or os.name == 'os2':
        os.system(f'open "{path}"')
    elif os.name == 'nt':
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

def setRetainSizePolicy(widget):
    sp = widget.sizePolicy()
    sp.setRetainSizeWhenHidden(True)
    widget.setSizePolicy(sp)

def getBasename(files):
    basename = files[0]
    for file in files:
        # Determine the basename based on intersection of all .tif
        _, ext = os.path.splitext(file)
        sm = difflib.SequenceMatcher(None, file, basename)
        i, j, k = sm.find_longest_match(
            0, len(file), 0, len(basename)
        )
        basename = file[i:i+k]
    return basename

def findalliter(patter, string):
    """Function used to return all re.findall objects in string"""
    m_test = re.findall(f'(\d+)_(.+)', string)
    m_iter = [m_test]
    while m_test:
        m_test = re.findall(f'(\d+)_(.+)', m_test[0][1])
        m_iter.append(m_test)
    return m_iter


def listdir(path):
    return natsorted([
        f for f in os.listdir(path)
        if not f.startswith('.')
        and not f.endswith('.ini')
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

def download_examples(which='time_lapse_2D', progress=None):
    # https://drive.google.com/drive/u/0/folders/1OgUgp_HuYsZlDg_TVWPuhT4OdZXJHbAg
    if which == 'time_lapse_2D':
        foldername = 'TimeLapse_2D'
        url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/CaMdYXiwxxoq3Ts/download/TimeLapse_2D.zip'
        file_size = 45143552
    elif which == 'snapshots_3D':
        foldername = 'Multi_3D_zStacks'
        url = 'https://hmgubox2.helmholtz-muenchen.de/index.php/s/CXZDoQMANNrKL7a/download/Yeast_Analysed_multi3D_zStacks.zip'
        file_size = 124822528
    else:
        return ''

    user_path = pathlib.Path.home()
    examples_path = os.path.join(user_path, 'acdc-examples')
    example_path = os.path.join(examples_path, 'examples')

    if os.path.exists(example_path):
        return example_path

    zip_dst = os.path.join(examples_path, 'example_temp.zip')

    if not os.path.exists(examples_path):
        os.makedirs(examples_path)

    download_url(
        url, zip_dst, desc=f'example "{foldername}"', file_size=file_size,
        progress=progress
    )
    exctract_to = examples_path
    extract_zip(zip_dst, exctract_to)
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
        return jre_path, jdk_path

    if jre_path:
        return jre_path, jre_path

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

def download_from_gdrive(id, destination, file_size=None,
                         model_name='cellpose', progress=None):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    if token is not None:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params=params, stream=True)

    save_response_content(
        response, destination, file_size=file_size, model_name=model_name,
        progress=progress
    )

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

def check_v1_model_path():
    script_dirname = os.path.dirname(os.path.realpath(__file__))
    main_path = os.path.dirname(script_dirname)
    v1_model_path = os.path.join(main_path, 'model')
    print(v1_model_path)
    if os.path.exists(v1_model_path):
        delete = prompts.twobuttonsmessagebox('Delete v1 model folder?',
            'The script detected a "./model" folder.\n\n This is most likely from '
            'Cell-ACDC v1.\n\nThis version will automatically download\n the '
            'neural network models required into "/.models" folder.\n'
            'The "./model" is not required anymore and we suggest deleting it,\n'
            'however you can keep it if you want.\n\n '
            'Do you want to delete it or keep it?',
            'Delete', 'Keep', fs=10,
        ).button_left
        if delete:
            shutil.rmtree(v1_model_path)

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
    if data.dtype != np.uint8 or data.dtype != np.uint16:
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
    if img.max() <= 1:
        return img

    uint8_max = np.iinfo(np.uint8).max
    uint16_max = np.iinfo(np.uint16).max
    if img.max() > uint8_max:
        img = img/uint16_max
    else:
        img = img/uint8_max
    return img

def install_package_msg(pkg_name, parent=None):
    msg = QMessageBox()
    txt = (f"""
    <p>
        Cell-ACDC is going to download and install <code>{pkg_name}</code>.<br><br>
        Make sure you have an <b>active internet connection</b>,
        before continuing.
        Progress will be displayed on the terminal<br><br>
        <b>IMPORTANT:</b> If the installation fails please install
        <code>{pkg_name}</code> manually with the follwing command:<br><br>
        <code>pip install {pkg_name}</code><br><br>
        Alternatively, you can cancel the process and try later.
    </p>
    """)
    answer = msg.information(
        parent, f'Install {pkg_name}', txt, msg.Ok | msg.Cancel
    )
    return answer == msg.Cancel

if __name__ == '__main__':
    print(get_list_of_models())
    # model_name = 'cellpose'
    # download_model(model_name)
    #
    # download_examples()
