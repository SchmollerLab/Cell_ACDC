"""Downloads: the generic URL and zip helpers, locating and downloading
model weights (get_model_path is plugin API; download_model, _model_url
and the _download_<model> functions), moving models from the old
user-profile location, the user manual and ffmpeg. Example data:
help/welcome.py; Java and Bio-Formats: data_structure/helpers.py; Fiji:
fiji_macros/__init__.py.
"""
import os
import pathlib
import tempfile
import shutil
import traceback
from tqdm import tqdm
import requests
import zipfile
from cellacdc import is_linux, is_win, is_mac
from cellacdc import cellacdc_path, acdc_ffmpeg_path
from cellacdc import user_profile_path
from . import filesystem


def get_model_path(model_name, create_temp_dir=True):
    if model_name == 'Automatic thresholding':
        model_name = 'thresholding'
        
    model_info_path = os.path.join(cellacdc_path, 'models', model_name, 'model')
    
    if os.path.exists(model_info_path):
        for file in filesystem.listdir(model_info_path):
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
        files_present = filesystem.listdir(model_path)
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
    elif model_name == 'DeepSea':
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
    elif model_name == 'sam2':
        url = [
            'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',
            'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt',
            'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt',
            'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt'
        ]
        file_size = [155233385, 184211977, 319128965, 910600801]
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

def _download_sam2_models():
    urls, file_sizes = _model_url('sam2')
    temp_model_path = tempfile.mkdtemp()
    _, final_model_path = (
        get_model_path('sam2', create_temp_dir=False)
    )
    for url, file_size in zip(urls, file_sizes):
        filename = url.split('/')[-1]
        final_dst = os.path.join(final_model_path, filename)
        if os.path.exists(final_dst):
            continue

        temp_dst = os.path.join(temp_model_path, filename)
        download_url(
            url, temp_dst, file_size=file_size, desc='sam2',
            verbose=False
        )

        shutil.move(temp_dst, final_dst)

def _download_deepsea_models():
    urls, file_sizes = _model_url('DeepSea')
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


def showUserManual():
    manual_file_path = download_manual()
    filesystem.showInExplorer(manual_file_path)

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
    from cellacdc import user_data_dir
    user_data_folderpath = user_data_dir()
    user_profile_path_txt = os.path.join(
        user_data_folderpath, 'acdc_user_profile_location.txt'
    )
    if os.path.exists(user_profile_path_txt):
        return False
    
    from cellacdc import user_home_path
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


def download_model(model_name):
    if model_name == 'segment_anything':
        try:
            _download_segment_anything_models()
            return True
        except Exception as e:
            traceback.print_exc()
            return False
    elif model_name == 'sam2':
        try:
            _download_sam2_models()
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
            for file in filesystem.listdir(v123_model_path):
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
        for file in filesystem.listdir(temp_dir):
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
    final_model_path = os.path.expanduser(
        os.path.join("~", ".cellpose", "models")
    )
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
