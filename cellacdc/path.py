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

def copytree(src, dst, symlinks=False, ignore=None):
    """Recursively copy a directory tree using copy2().

    The destination directory must not already exist.
    If exception(s) occur, an Error is raised with a list of reasons.

    If the optional symlinks flag is true, symbolic links in the
    source tree result in symbolic links in the destination tree; if
    it is false, the contents of the files pointed to by symbolic
    links are copied.

    The optional ignore argument is a callable. If given, it
    is called with the `src` parameter, which is the directory
    being visited by copytree(), and `names` which is the list of
    `src` contents, as returned by os.listdir():

        callable(src, names) -> ignored_names

    Since copytree() is called recursively, the callable will be
    called once for each directory that is copied. It returns a
    list of names relative to the `src` directory that should
    not be copied.

    XXX Consider this example code rather than the ultimate tool.

    """
    names = os.listdir(src)
    if ignore is not None:
        ignored_names = ignore(src, names)
    else:
        ignored_names = set()

    os.makedirs(dst, exist_ok=True)
    errors = []
    for name in names:
        if name in ignored_names:
            continue
        srcname = os.path.join(src, name)
        dstname = os.path.join(dst, name)
        try:
            if symlinks and os.path.islink(srcname):
                linkto = os.readlink(srcname)
                os.symlink(linkto, dstname)
            elif os.path.isdir(srcname):
                copytree(srcname, dstname, symlinks, ignore)
            else:
                # Will raise a SpecialFileError for unsupported file types
                shutil.copy2(srcname, dstname)
        # catch the Error from the recursive copytree so that we can
        # continue with other files
        except shutil.Error as err:
            errors.extend(err.args[0])
        except EnvironmentError as why:
            errors.append((srcname, dstname, str(why)))
    try:
        shutil.copystat(src, dst)
    except OSError as why:
        if WindowsError is not None and isinstance(why, WindowsError):
            # Copying file access times may fail on Windows
            pass
        else:
            errors.append((src, dst, str(why)))
    if errors:
        raise shutil.Error, errors