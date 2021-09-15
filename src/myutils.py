import os
import pathlib
import sys
import tempfile
import shutil
from tqdm import tqdm
import requests
import zipfile
import numpy as np
import skimage
from distutils.dir_util import copy_tree
from pyqtgraph.colormap import ColorMap
import prompts
from tifffile.tifffile import TiffWriter, TiffFile

__all__ = ['ColorMap']
_mapCache = {}

def download_java():
    is_linux = sys.platform.startswith('linux')
    is_mac = sys.platform == 'darwin'
    is_win = sys.platform.startswith("win")
    is_win64 = (is_win and (os.environ["PROCESSOR_ARCHITECTURE"] == "AMD64"))

    if is_win64:
        foldername = 'win64'
        jre_name = 'jre1.8.0_301'
        file_id = '19KXlsTwDwR7VZDBu2uWO1M3uIRlrPzLU'
        file_size = 78397719
    elif is_mac:
        foldername = 'macOS'
        jre_name = 'jre1.8.0_301'
    elif is_linux:
        foldername = 'linux'
        jre_name = 'jre1.8.0_301'
    elif is_win:
        foldername = 'win'
        jre_name = 'jre1.8.0_301'

    src_path = os.path.dirname(os.path.realpath(__file__))
    java_path = os.path.join(src_path, 'java', foldername)
    jre_path = os.path.join(java_path, jre_name)
    zip_dst = os.path.join(java_path, 'java_temp.zip')

    if os.path.exists(jre_path):
        return

    if not os.path.exists(java_path):
        os.makedirs(java_path)

    download_from_gdrive(
        file_id, zip_dst, file_size=file_size, model_name='Java'
    )
    exctract_to = java_path
    extract_zip(zip_dst, exctract_to)
    # Remove downloaded zip archive
    os.remove(zip_dst)
    print('Java downloaded successfully')

def copyRenameJavabridge():
    src_path = os.path.dirname(os.path.realpath(__file__))
    major, minor = sys.version_info[:2]
    if major != 3:
        raise EnvironmentError(
            'Only Python 3 is supported, '
            f'while you have Python {sys.version} installed')
    javabridge_src = os.path.join(src_path, f'javabridge_cp3{minor}')
    javabridge_dst = os.path.join(src_path, 'javabridge')
    copy_tree(javabridge_src, javabridge_dst)

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

def get_model_path(model_foldername):
    script_dirname = os.path.dirname(os.path.realpath(__file__))
    main_path = os.path.dirname(script_dirname)
    model_path = os.path.join(main_path, 'models', model_foldername)

    if not os.path.exists(model_path):
        model_folder_exists = False
        os.mkdir(model_path)
    else:
        model_folder_exists = True
    models_zip_path = os.path.join(model_path, 'model_temp.zip')
    return models_zip_path, model_folder_exists

def get_file_id(model_name, id=None):
    if model_name == 'YeaZ':
        file_id = '1nmtUHG8JM8Hp1zas2xlXoWLYPqS9psON'
        file_size = 693685011
    elif model_name == 'cellpose':
        file_id = '1nfOwE5UtGwKm4zLgPdzDbkJVG7kZL4Yw'
        file_size = 392564736
    else:
        file_id = id
        file_size = None
    return file_id, file_size

def download_from_gdrive(id, destination, file_size=None,
                         model_name='cellpose'):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)
    save_response_content(response, destination,
                          file_size=file_size, model_name=model_name)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination, file_size=None,
                          model_name='cellpose'):
    print(f'Downloading {model_name} to: {os.path.dirname(destination)}')
    CHUNK_SIZE = 32768
    temp_folder = pathlib.Path.home().joinpath('.cp_temp')
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)
    temp_dst = os.path.join(temp_folder, os.path.basename(destination))
    pbar = tqdm(total=file_size, unit='B', unit_scale=True,
                unit_divisor=1024, ncols=100)
    with open(temp_dst, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                pbar.update(len(chunk))
    pbar.close()
    shutil.move(temp_dst, destination)
    shutil.rmtree(temp_folder)

def extract_zip(zip_path, extract_to_path):
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
            'Yeast_ACDC v1.\n\nThis version will automatically download\n the '
            'neural network models required into "/.models" folder.\n'
            'The "./model" is not required anymore and we suggest deleting it,\n'
            'however you can keep it if you want.\n\n '
            'Do you want to delete it or keep it?',
            'Delete', 'Keep', fs=10,
        ).button_left
        if delete:
            shutil.rmtree(v1_model_path)

def download_model(model_name):
    models_zip_path, model_folder_exists = get_model_path(f'{model_name}_model')
    if not model_folder_exists:
        check_v1_model_path()
        file_id, file_size = get_file_id(model_name)
        # Download zip file
        download_from_gdrive(file_id, models_zip_path,
                             file_size=file_size, model_name=model_name)
        # Extract zip file
        extract_to_path = os.path.dirname(models_zip_path)
        extract_zip(models_zip_path, extract_to_path)
        # Remove downloaded zip archive
        os.remove(models_zip_path)

def imagej_tiffwriter(new_path, data, metadata, SizeT, SizeZ):
    if data.dtype != np.uint8 or data.dtype != np.uint16:
        data = skimage.img_as_uint(data)
    with TiffWriter(new_path, imagej=True) as new_tif:
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
        new_tif.save(data, metadata=metadata)

if __name__ == '__main__':
    model_name = 'cellpose'
    download_model(model_name)

    download_java()
