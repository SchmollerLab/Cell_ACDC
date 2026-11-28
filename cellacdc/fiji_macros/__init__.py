import os
import datetime
from typing import Iterable
from uuid import uuid4


from .. import acdc_fiji_path
import shlex
import tempfile
import subprocess
from cellacdc import is_win, is_mac
from cellacdc import printl
from cellacdc.myutils import downloads

def init_macro(
        files_folderpath: os.PathLike, 
        is_multiple_files: bool,
        is_separate_channels: bool,
        dst_folderpath: os.PathLike, 
        channels: Iterable[str]
    ):
    macros_folderpath = os.path.join(acdc_fiji_path, 'macros')
    os.makedirs(macros_folderpath, exist_ok=True)
    
    macros_template_folderpath = os.path.dirname(os.path.abspath(__file__))
    if is_separate_channels:
        macro_template_filename = 'multiple_files_separate_channels.ijm'
    elif is_multiple_files:
        macro_template_filename = 'multiple_files.ijm'
    else:
        macro_template_filename = 'single_file.ijm'
        
    macro_template_filepath = os.path.join(
        macros_template_folderpath, macro_template_filename
    )
    with open(macro_template_filepath, 'r') as ijm:
        macro_txt = ijm.read()
    
    channels = [f'"{ch.strip()}"' for ch in channels]
    channels_macro = ', '.join(channels)
    macro_txt = macro_txt.replace(
        'channels = newArray(...)', 
        f'channels = newArray({channels_macro})'
    )
    
    files_path = files_folderpath.replace('\\', '/')
    files_path = f'"{files_path}/"'
    macro_txt = macro_txt.replace('id = ...', f'id = {files_path}')
    
    dst_folderpath = dst_folderpath.replace('\\', '/')
    macro_txt = macro_txt.replace(
        'dst_folderpath = ...', f'dst_folderpath = "{dst_folderpath}"'
    )
    
    date_time = datetime.datetime.now().strftime(r'%Y-%m-%d_%H-%M-%S')
    id = uuid4()
    macro_filename = f'{date_time}_{id}_{macro_template_filename}'
    macro_filepath = os.path.join(macros_folderpath, macro_filename)
    with open(macro_filepath, 'w') as ijm:
        ijm.write(macro_txt)
    
    return macro_filepath

def command_run_macro(macro_filepath):
    exec_path = get_fiji_exec_folderpath()
    command = f'{exec_path} -macro {macro_filepath}'
    return command

def run_macro(macro_command):
    success = run_fiji_command(command=macro_command)
    return success


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
    downloads.download_url(
        url, zip_dst, verbose=False, file_size=file_size
    )
    downloads.extract_zip(zip_dst, acdc_fiji_path)
    
    return acdc_fiji_path


def get_fiji_binary_filepath_mac(fiji_app_filepath):
    if not is_mac:
        return ''

    fiji_binary_path = os.path.join(
        fiji_app_filepath, 'Contents', 'MacOS', 'ImageJ-macosx'
    )
    if os.path.exists(fiji_binary_path):
        return fiji_binary_path
    
    fiji_binary_path = os.path.join(
        fiji_app_filepath, 'Contents', 'MacOS', 'fiji-macos'
    )
    if os.path.exists(fiji_binary_path):
        return fiji_binary_path

    return ''

def get_fiji_exec_folderpath() -> str:
    if not is_mac:
        return ''
    
    from cellacdc import fiji_location_filepath
    
    if os.path.exists(fiji_location_filepath):
        with open(fiji_location_filepath, 'r') as txt:
            fiji_app_filepath = txt.read()

        return get_fiji_binary_filepath_mac(fiji_app_filepath)

    if os.path.exists('/Applications/Fiji.app'):
        return get_fiji_binary_filepath_mac('/Applications/Fiji.app')
    
    acdc_fiji_app_path = os.path.join(acdc_fiji_path, 'Fiji.app')
    acdc_fiji_binary_path = get_fiji_binary_filepath_mac(acdc_fiji_app_path)
    
    return acdc_fiji_binary_path

def get_fiji_base_command():
    command = None
    if is_mac:
        command = get_fiji_exec_folderpath()
    
    return command

def _init_fiji_cli():
    if is_win:
        return True
    
    fiji_app_folderpath = get_fiji_exec_folderpath()
    args_add_to_path = [f'chmod 755 {fiji_app_folderpath}']
    try:
        subprocess.check_call(args_add_to_path, shell=True)
        return True
    except Exception as e:
        printl(f'Error occurred while setting permissions: {e}')
        return False

def test_fiji_base_command(logger_func=print):
    base_command = get_fiji_base_command()

    if base_command is None:
        logger_func('[WARNING]: Fiji is not present.')
        return False
    
    command = f'{base_command} --headless'
    return run_fiji_command(command=command, logger_func=logger_func) 

def run_fiji_command(command=None, logger_func=print):
    if command is None:
        command = f'{get_fiji_base_command()} --headless'
    
    init_success = _init_fiji_cli()
    if not init_success:
        return False

    separator = '-'*100
    commands = (command, shlex.split(command))
    for args in commands:
        logger_func(
            f'{separator}\n'
            f'Trying Fiji command: "{args}"...\n'
            f'{separator}\n'
        )
        try:
            subprocess.check_call(args, shell=True)
            return True
        except Exception as err:
            continue
    return False
