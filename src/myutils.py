import os
import pathlib
import sys
import tempfile
import shutil
from tqdm import tqdm
import requests
import zipfile
from lib import twobuttonsmessagebox

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
    print(f'Downloading {model_name} models to: {os.path.dirname(destination)}')
    CHUNK_SIZE = 32768
    temp_folder = pathlib.Path.home().joinpath('.cp_temp')
    if not os.path.exists(temp_folder):
        os.mkdir(temp_folder)
    temp_dst = os.path.join(temp_folder, os.path.basename(destination))
    pbar = tqdm(total=file_size, unit='B', unit_scale=True,
                unit_divisor=1024)
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
        delete = twobuttonsmessagebox('Delete v1 model folder?',
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

if __name__ == '__main__':
    model_name = 'cellpose'
    download_model(model_name)
