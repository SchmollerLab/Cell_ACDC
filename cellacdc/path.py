import os
import sys
import pathlib

import subprocess

from natsort import natsorted

from . import is_mac, is_linux

def listdir(path):
    return natsorted([
        f for f in os.listdir(path)
        if not f.startswith('.')
        and not f == 'desktop.ini'
        and not f == 'recovery'
    ])

def newfilepath(file_path, appended_text: str=None):
    if appended_text is None:
        appended_text=''
    
    if not os.path.exists(file_path):
        return file_path, appended_text
    
    folder_path = os.path.dirname(file_path)
    filename = os.path.basename(file_path)
    filename, ext = os.path.splitext(filename)

    if appended_text:
        if appended_text.startswith('_'):
            appended_text = appended_text.lstrip('_')

    if appended_text:
        new_filename = f'{filename}_{appended_text}{ext}'
        new_filepath = os.path.join(folder_path, new_filename)
        if not os.path.exists(new_filepath):
            return new_filepath, appended_text
    
    i = 0
    while True:
        if appended_text:
            new_filename = f'{filename}_{appended_text}_{i+1}{ext}'
        else:
            new_filename = f'{filename}_{i+1}{ext}'
        new_filepath = os.path.join(folder_path, new_filename)
        if not os.path.exists(new_filepath):
            return new_filepath, f'{appended_text}_{i+1}'
        i += 1

def show_in_file_manager(path):
    if is_mac:
        args = ['open', fr'{path}']
    elif is_linux:
        args = ['xdg-open', fr'{path}']
    else:
        if os.path.isfile(path):
            args = ['explorer', '/select,', os.path.realpath(path)]
        else:
            args = ['explorer', os.path.realpath(path)]
    subprocess.run(args)