from collections import defaultdict
import os
import sys
import shutil

import subprocess

from natsort import natsorted

from . import is_mac, is_linux
from . import printl
from . import myutils

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

def copy_or_move_tree(
        src: os.PathLike, dst: os.PathLike, copy=False,
        sigInitPbar=None, sigUpdatePbar=None
    ):
    if sigInitPbar is not None:
        sigInitPbar.emit(0)
    
    files_failed_move = {}
    files_info = {}
    for root, dirs, files in os.walk(src):
        for file in files:
            rel_path = os.path.relpath(root, src).replace('\\', '/')
            src_filepath = os.path.join(root, file)
            dst_filepath = os.path.join(dst, *rel_path.split('/'), file)
            files_info[src_filepath] = dst_filepath
    
    if sigInitPbar is not None:
        sigInitPbar.emit(len(files_info))
    for src_filepath, dst_filepath in files_info.items():
        os.makedirs(os.path.dirname(dst_filepath), exist_ok=True)
        if copy:
            shutil.copyfile(src_filepath, dst_filepath)
        else:
            try:
                os.rename(src_filepath, dst_filepath)
            except Exception as e:
                shutil.copyfile(src_filepath, dst_filepath)
                files_failed_move[src_filepath] = dst_filepath
        if sigUpdatePbar is not None:
            sigUpdatePbar.emit(1)
    return files_failed_move

def get_posfolderpaths_walk(folderpath):
    pos_folderpaths = defaultdict(set)
    for root, dirs, files in os.walk(folderpath):
        if not root.endswith('Images'):
            continue
        
        pos_folderpath = os.path.dirname(root)
        if not myutils.is_pos_folderpath(pos_folderpath):
            continue
        
        exp_path = os.path.dirname(pos_folderpath).replace('\\', '/')
        pos_foldername = os.path.basename(pos_folderpath)
        pos_folderpaths[exp_path].add(pos_foldername)
    
    for exp_path in pos_folderpaths.keys():
        pos_folderpaths[exp_path] = natsorted(pos_folderpaths[exp_path])
    
    return pos_folderpaths