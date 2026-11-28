"""Finding experiments and files: Position and Images folder detection,
listdir, channel and basename lookup, the status of a Position
(Cell-ACDC and spotMAX), the recent-path CSV files (also those of the
GUI's "export current view"), resetting the user settings, and opening
folders or URLs in the operating system. Imports load.py at module level
and the shared message box (shared/widgets/message_box.py) under if
GUI_INSTALLED.
"""
import os
import re
from typing import Literal, List
import pathlib
import difflib
import traceback
import datetime
import subprocess
import numpy as np
import pandas as pd
from natsort import natsorted
from cellacdc import GUI_INSTALLED, KNOWN_EXTENSIONS
from cellacdc import load
from cellacdc import html_utils, is_linux, is_win, is_mac
from cellacdc import cellacdc_path
from cellacdc import recentPaths_path
from cellacdc import export_current_view_gui_recent_paths_csv_path
from cellacdc import urls
from cellacdc import settings_folderpath
if GUI_INSTALLED:
    from cellacdc import widgets


def get_module_name(script_file_path):
    parts = pathlib.Path(script_file_path).parts
    parts = list(parts[parts.index('cellacdc')+1:])
    parts[-1] = os.path.splitext(parts[-1])[0]
    module = '.'.join(parts)
    return module

def get_pos_status_acdc(pos_path):
    images_path = os.path.join(pos_path, 'Images')
    ls = listdir(images_path)
    for file in ls:
        if file.endswith('acdc_output.csv'):
            acdc_df_path = os.path.join(images_path, file)
            break
    else:
        return ''
    
    acdc_df = pd.read_csv(acdc_df_path)
    last_tracked_i = acdc_df['frame_i'].max()
    last_cca_i = 0
    if 'cell_cycle_stage' in acdc_df.columns:
        cca_df = acdc_df[['frame_i', 'cell_cycle_stage']].dropna()
        last_cca_i = cca_df['frame_i'].max()
    if last_cca_i > 0:
        return (
            f' (last tracked frame = {last_tracked_i+1}, '
            f'last annotated frame = {last_cca_i+1})'
        )
    else:
        return f' (last tracked frame = {last_tracked_i+1})'

def get_pos_status_spotmax(pos_path):
    spotmax_out_path = os.path.join(pos_path, 'spotMAX_output')
    is_smax_out_present = 'Yes' if os.path.exists(spotmax_out_path) else 'No'
    if os.path.exists(spotmax_out_path):
        return ' (SpotMAX output exists)'
    else:
        return ''

def get_pos_status(
        pos_path, 
        caller: Literal['Cell-ACDC', 'SpotMAX']='Cell-ACDC'
    ):
    if caller == 'Cell-ACDC':
        return get_pos_status_acdc(pos_path)
    
    if caller == 'SpotMAX':
        return get_pos_status_spotmax(pos_path)

def get_gdrive_path():
    if is_win:
        return os.path.join(f'G:{os.sep}', 'My Drive')
    elif is_mac:
        return os.path.join(
            '/Users/francesco.padovani/Library/CloudStorage/'
            'GoogleDrive-padovaf@tcd.ie/My Drive'
        )

def get_acdc_data_path():
    Cell_ACDC_path = os.path.dirname(cellacdc_path)
    return os.path.join(Cell_ACDC_path, 'data')

def get_open_filemaneger_os_string():
    if is_win:
        return 'Show in Explorer...'
    elif is_mac:
        return 'Reveal in Finder...'
    elif is_linux:
        return 'Show in File Manager...'


def remove_known_extension(name):
    for ext in KNOWN_EXTENSIONS:
        if name.endswith(ext):
            return name[:-len(ext)], ext

    return name, ''


class utilClass:
    pass


def get_pos_foldernames(exp_path, check_if_is_sub_folder=False):
    if not check_if_is_sub_folder:
        ls = listdir(exp_path)
        pos_foldernames = [
            pos for pos in ls if is_pos_folderpath(os.path.join(exp_path, pos))
        ]
    else:
        folder_type = determine_folder_type(exp_path)
        is_pos_folder, is_images_folder, _ = folder_type
        if is_pos_folder:
            return [os.path.basename(exp_path)]
        elif is_images_folder:
            pos_path = os.path.dirname(exp_path)
            if is_pos_folderpath(pos_path):
                return [os.path.basename(pos_path)]
            else:
                return []
        else:
            return get_pos_foldernames(exp_path)
    return pos_foldernames

def get_images_folderpath(folderpath):
    if os.path.isfile(folderpath):
        folderpath = os.path.dirname(folderpath)

    if folderpath.endswith('Images'):
        return folderpath
    
    images_folderpath = os.path.join(folderpath, 'Images')
    if os.path.exists(images_folderpath):
        return images_folderpath
    
    return ''

def checkDataIntegrity(filenames, parent_path, parentQWidget=None):
    if not filenames:
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            'Cell-ACDC could not find any files in the folder '
            f'<b>{parent_path}</b>.<br><br>'
            'Please make sure that the folder contains at least one image file.<br><br>'
            'Thank you for your patience!'
        )
        msg.warning(parentQWidget, 'Selected folder is emppty', txt)
        raise FileNotFoundError(
            f'No files found in the folder {parent_path}. '
        )
            
    char = filenames[0][:2]
    startWithSameChar = all([f.startswith(char) for f in filenames])
    if not startWithSameChar:
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            'Cell-ACDC detected files inside the folder '
            'that <b>do not start with the same, common basename</b>.<br><br>'
            'To ensure correct loading of the data, the folder where '
            'the file(s) is/are should either contain a single image file or'
            'only files that start with the same, common basename.<br><br>'
            'For example the following filenames:<br><br>'
            '<code>F014_s01_phase_contr.tif</code><br>'
            '<code>F014_s01_mCitrine.tif</code><br><br>'
            'are named correctly since they all start with the '
            'the common basename "F014_s01_". After the common basename you '
            'can write whatever text you want. In the example above, "phase_contr" '
            'and "mCitrine" are the channel names.<br><br>'
            'Data loading may still be successfull, so Cell-ACDC will '
            'still try to load data now.<br>'
        )
        filesFormat = [f'    - {file}' for file in filenames]
        filesFormat = "\n".join(filesFormat)
        detailsText = (
            f'Files present in the folder {parent_path}:\n\n'
            f'{filesFormat}'
        )
        msg.addShowInFileManagerButton(parent_path, txt='Open folder...')
        msg.warning(
            parentQWidget, 'Data structure compromised', txt, 
            detailsText=detailsText, buttonsTexts=('Cancel', 'Ok')
        )
        if msg.cancel:
            raise TypeError(
                'Process aborted by the user.'
            )
        return False
    return True


def browse_url(url):
    import webbrowser
    webbrowser.open(url)

def browse_docs():
    browse_url(urls.docs_homepage)


def showInExplorer(path):
    if is_mac:
        subprocess.run(["open", str(path)], check=True)
    elif is_linux:
        subprocess.run(["xdg-open", str(path)], check=True)
    else:
        os.startfile(path)


def getAcdcDfSegmPaths(images_path):
    ls = listdir(images_path)
    basename = getBasename(ls)
    paths = {}
    for file in ls:
        filePath = os.path.join(images_path, file)
        fileName, ext = os.path.splitext(file)
        endName = fileName[len(basename):]
        if endName.find('acdc_output') != -1 and ext=='.csv':
            info_name = endName.replace('acdc_output', '')
            paths.setdefault(info_name, {})
            paths[info_name]['acdc_df_path'] = filePath
            paths[info_name]['acdc_df_filename'] = fileName
        elif endName.find('segm') != -1 and ext=='.npz':
            info_name = endName.replace('segm', '')
            paths.setdefault(info_name, {})
            paths[info_name]['segm_path'] = filePath
            paths[info_name]['segm_filename'] = fileName
    return paths

def getChannelFilePath(images_path, chName):
    channel_filepath = load.get_filename_from_channel(images_path, chName)
    return channel_filepath


def get_chname_from_basename(filename, basename, remove_ext=True):
    if remove_ext:
        filename, ext = os.path.splitext(filename)
    chName = filename[len(basename):]
    aligned_idx = chName.find('_aligned')
    if aligned_idx != -1:
        chName = chName[:aligned_idx]
    
    if ';;' in chName:
        chName = chName.split(';;')[-1]
        
    return chName


def getBasenameAndChNames(images_path, useExt=None):
    _tempPosData = utilClass()
    _tempPosData.images_path = images_path
    load.loadData.getBasenameAndChNames(_tempPosData, useExt=useExt)
    return _tempPosData.basename, _tempPosData.chNames

def getBasename(files):
    basename = files[0]
    for file in files:
        # Determine the basename based on intersection of all files
        _, ext = os.path.splitext(file)
        sm = difflib.SequenceMatcher(None, file, basename)
        i, j, k = sm.find_longest_match(
            0, len(file), 0, len(basename)
        )
        basename = file[i:i+k]
    return basename


def listdir(path, include_directories=True) -> List[str]:
    items_list = natsorted([
        f for f in os.listdir(path)
        if not f.startswith('.')
        and not f == 'desktop.ini'
        and not f == 'recovery'
        and not f.endswith('.new.npz')
        and (include_directories or not os.path.isdir(os.path.join(path, f)))
    ])
    return items_list


def determine_folder_type(folder_path):
    is_pos_folder = is_pos_folderpath(folder_path)
    is_images_folder = folder_path.endswith('Images') and listdir(folder_path)
    contains_images_folder = os.path.exists(
        os.path.join(folder_path, 'Images')
    )
    contains_pos_folders = len(get_pos_foldernames(folder_path)) > 0
    if contains_pos_folders:
        is_pos_folder = False
        is_images_folder = False
    elif contains_images_folder and not is_pos_folder:
        # Folder created by loading an image
        is_images_folder = True
        folder_path = os.path.join(folder_path, 'Images')
    
    return is_pos_folder, is_images_folder, folder_path


def get_show_in_file_manager_text():
    if is_mac:
        return 'Reveal in Finder'
    elif is_linux:
        return 'Show in File Manager'
    elif is_win:
        return 'Show in File Explorer'


def is_pos_folderpath(folderpath):
    """Determine if a path is a valid Cell-ACDC Position folder

    Parameters
    ----------
    folderpath : PathLike
        Path to check

    Returns
    -------
    bool
        True if the path is a valid Cell-ACDC Position folder, False otherwise
    
    Notes
    -----
    A valid Cell-ACDC Position folder must:
        - Have a name matching the pattern 'Position_<number>'
        - Be a directory
        - Contain an 'Images' subdirectory
        - The 'Images' subdirectory must not be empty
    """    
    foldername = os.path.basename(folderpath)
    is_valid_pos_folder = (
        re.search(r'^Position_(\d+)$', foldername) is not None
        and os.path.isdir(folderpath)
        and os.path.exists(os.path.join(folderpath, 'Images'))
        and listdir(os.path.join(folderpath, 'Images'))
    )
    return is_valid_pos_folder


def reset_settings():
    question = (
        'Do you want to reset Cell-ACDC settings'
        '- type "h" for help - (y/[n]/h)? '
    )
    info_txt = (
        'If you reset Cell-ACDC settings, the folder below will be deleted.\n\n'
        'This means deeleting things like custom shortcuts, recent paths, last '
        'selections, and GUI preferences.\n\n'
        f'Settings folder path: "{settings_folderpath}"'
    )
    answer = 'y'
    while True:
        try: 
            answer = input(f'\n{question}')
        except Exception as err:
            break
        
        if answer == 'n':
            print('*'*100)
            return 'Resetting Cell-ACDC settings cancelled.'
        
        if answer == 'y':
            break
        
        if answer == 'h':
            print('-'*100)
            print(f'\n{info_txt}')
            print('='*100)
        
        print(
            f'"{answer}" is not a valid answer. '
            'Type "y" for "yes", "n" for "no", or "h" for help.'
        )
    
    try:
        os.remove(settings_folderpath)
        print('*'*100)
        out_txt = (
            'Cell-ACDC settings have been reset.\n\n'
            'The following folder was deleted:\n\n'
            f'{settings_folderpath}'
        )
    except Exception as err:
        traceback.print_exc()
        print('*'*100)
        out_txt = (
            '**ERROR** occured when trying to remove the settings folder.\n\n'
            'To reset Cell-ACDC settings, please remove this folder:\n\n'
            f'{settings_folderpath}\n'
        )
        return out_txt

def separate_fluo_segment_channels(channels):
    segms_to_load = []
    channels_to_load = []
    current_segm = False
    for ch in channels:
        if ch == 'current segm.':
            current_segm = True
        elif 'segm' in ch:
            segms_to_load.append(ch)
        else:
            channels_to_load.append(ch)
    return segms_to_load, channels_to_load, current_segm


def get_recent_paths(*args, only_existing=True):
    """Return existing unique paths and their aligned most recent dates."""
    if os.path.exists(recentPaths_path):
        df = pd.read_csv(recentPaths_path, index_col='index')
        if only_existing:
            df = df[df['path'].apply(os.path.exists)]
        if 'opened_last_on' in df.columns:
            df = df.sort_values('opened_last_on', ascending=False)
        df['path'] = df['path'].map(str)
        df = df.loc[df['path'].map(os.path.isdir).astype(bool)].copy()
        df['path'] = df['path'].str.replace('\\', '/')
        df = df.drop_duplicates(subset='path', keep='first')
        recent_paths = df['path'].to_list()
        
        dates = (
            df['opened_last_on'].to_list()
            if 'opened_last_on' in df.columns else None
        )
        
    else:
        recent_paths = []
        dates = None

    return recent_paths, dates

def getMostRecentPath():
    if os.path.exists(recentPaths_path):
        df = pd.read_csv(recentPaths_path, index_col='index')
        if 'opened_last_on' in df.columns:
            df = df.sort_values('opened_last_on', ascending=False)
        MostRecentPath = ''
        for path in df['path']:
            if os.path.exists(path):
                MostRecentPath = path
                break
    else:
        MostRecentPath = ''
    return MostRecentPath

def addToRecentPaths(exp_path, logger=None):
    if not os.path.exists(exp_path):
        return
    exp_path = exp_path.replace('\\', '/')
    if os.path.exists(recentPaths_path):
        try:
            df = pd.read_csv(recentPaths_path, index_col='index')
            recentPaths = df['path'].to_list()
            if 'opened_last_on' in df.columns:
                openedOn = df['opened_last_on'].to_list()
            else:
                openedOn = [np.nan]*len(recentPaths)
            if exp_path in recentPaths:
                pop_idx = recentPaths.index(exp_path)
                recentPaths.pop(pop_idx)
                openedOn.pop(pop_idx)
            recentPaths.insert(0, exp_path)
            openedOn.insert(0, datetime.datetime.now())
        except Exception as e:
            recentPaths = [exp_path]
            openedOn = [datetime.datetime.now()]
    else:
        recentPaths = [exp_path]
        openedOn = [datetime.datetime.now()]
    df = pd.DataFrame({
        'path': recentPaths,
        'opened_last_on': pd.Series(openedOn, dtype='datetime64[ns]')}
    )
    df.index.name = 'index'
    df.to_csv(recentPaths_path)

def add_to_export_current_view_gui_recent_paths(folderpath):
    if not os.path.exists(folderpath):
        return

    folderpath = folderpath.replace('\\', '/')
    if os.path.exists(export_current_view_gui_recent_paths_csv_path):
        try:
            df = pd.read_csv(
                export_current_view_gui_recent_paths_csv_path, 
                index_col='index'
            )
            recent_paths = df['path'].to_list()
            if 'opened_last_on' in df.columns:
                openedOn = df['opened_last_on'].to_list()
            else:
                openedOn = [np.nan]*len(recent_paths)
            if folderpath in recent_paths:
                pop_idx = recent_paths.index(folderpath)
                recent_paths.pop(pop_idx)
                openedOn.pop(pop_idx)
            recent_paths.insert(0, folderpath)
            openedOn.insert(0, datetime.datetime.now())
        except Exception as e:
            recent_paths = [folderpath]
            openedOn = [datetime.datetime.now()]
    else:
        recent_paths = [folderpath]
        openedOn = [datetime.datetime.now()]
    df = pd.DataFrame({
        'path': recent_paths,
        'opened_last_on': pd.Series(openedOn, dtype='datetime64[ns]')}
    )
    df.index.name = 'index'
    df.to_csv(export_current_view_gui_recent_paths_csv_path)

def get_export_current_view_gui_most_recent_path():
    if os.path.exists(export_current_view_gui_recent_paths_csv_path):
        df = pd.read_csv(
            export_current_view_gui_recent_paths_csv_path, 
            index_col='index'
        )
        df = df[df['path'].apply(os.path.exists)]
        if 'opened_last_on' in df.columns:
            df = df.sort_values('opened_last_on', ascending=False)
        most_recent_path = ''
        for path in df['path']:
            if os.path.exists(path):
                most_recent_path = path
                break
    else:
        most_recent_path = ''
    return most_recent_path

def get_export_current_view_gui_recent_paths(self):
    """Return existing unique paths and their aligned most recent dates."""
    if os.path.exists(export_current_view_gui_recent_paths_csv_path):
        df = pd.read_csv(
            export_current_view_gui_recent_paths_csv_path, index_col='index'
        )
        if 'opened_last_on' in df.columns:
            df = df.sort_values('opened_last_on', ascending=False)
        df['path'] = df['path'].map(str)
        df = df[df['path'].apply(os.path.exists)]
        df = df.loc[df['path'].map(os.path.isdir).astype(bool)].copy()
        df['path'] = df['path'].str.replace('\\', '/')
        df = df.drop_duplicates(subset='path', keep='first')
        recent_paths = df['path'].to_list()
        
        dates = (
            df['opened_last_on'].to_list()
            if 'opened_last_on' in df.columns else None
        )
        
    else:
        recent_paths = []
        dates = None

    return recent_paths, dates
