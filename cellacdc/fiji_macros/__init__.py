import os
import datetime
from typing import Iterable
from uuid import uuid4

from cellacdc import myutils

from .. import acdc_fiji_path

def init_macro(
        files_folderpath: os.PathLike, 
        is_multiple_files: bool,
        channels: Iterable[str]
    ):
    macros_folderpath = os.path.join(acdc_fiji_path, 'macros')
    os.makedirs(macros_folderpath, exist_ok=True)
    
    macros_template_folderpath = os.path.dirname(os.path.abspath(__file__))
    macro_template_filename = (
        'multiple_files.ijm' if is_multiple_files else 'single_file.ijm'
    )
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
    
    date_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    id = uuid4()
    macro_filename = f'{date_time}_{id}_{macro_template_filename}'
    macro_filepath = os.path.join(macros_folderpath, macro_filename)
    with open(macro_filepath, 'w') as ijm:
        ijm.write(macro_txt)
    
    return macro_filepath

def command_run_macro(macro_filepath):
    exec_path = myutils.get_fiji_exec_folderpath()
    command = f'{exec_path} -macro {macro_filepath}'
    return command

def run_macro(macro_command):
    success = myutils.run_fiji_command(command=macro_command)
    return success