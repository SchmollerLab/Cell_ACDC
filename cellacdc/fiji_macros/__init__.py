import os
import datetime
from typing import Iterable
from uuid import uuid4

from cellacdc import myutils

from .. import acdc_fiji_path

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
    exec_path = myutils.get_fiji_exec_folderpath()
    command = f'{exec_path} -macro {macro_filepath}'
    return command

def run_macro(macro_command):
    success = myutils.run_fiji_command(command=macro_command)
    return success