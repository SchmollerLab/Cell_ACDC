import os
import re
import time
import traceback
import difflib
import numpy as np
import pandas as pd
import sys

from . import GUI_INSTALLED

if GUI_INSTALLED:
    from qtpy.QtWidgets import (
        QApplication, QPushButton, QHBoxLayout, QLabel, QSizePolicy
    )
    from qtpy.QtCore import Qt
    from qtpy.QtGui import QFont

from . import myutils, printl, html_utils, load
from . import settings_folderpath

class select_channel_name:
    def __init__(self, which_channel=None, allow_abort=True):
        self.is_first_call = True
        self.which_channel = which_channel
        self.last_sel_channel = self._load_last_selection()
        self.was_aborted = False
        self.allow_abort = allow_abort

    def get_available_channels(
            self, filenames, images_path, useExt=None,
            channelExt=('.h5', '.tif', '_aligned.npz'), 
            validEndnames=('aligned.npz', 'acdc_output.csv', 'segm.npz')
        ):
        # First check if metadata.csv already has the channel names
        metadata_csv_path = None
        for file in myutils.listdir(images_path):
            if file.endswith('metadata.csv'):
                metadata_csv_path = os.path.join(images_path, file)
                break
        
        chNames_found = False
        channel_names = set()
        basename = None
        if metadata_csv_path is not None:
            df = pd.read_csv(metadata_csv_path)
            basename = None
            if 'Description' in df.columns:
                channelNamesMask = df.Description.str.contains(r'channel_\d+_name')
                channelNames = df[channelNamesMask]['values'].to_list()
                try:
                    basename = df.set_index('Description').at['basename', 'values']
                except Exception as e:
                    basename = None
                if channelNames:
                    # There are channel names in metadata --> check that they 
                    # are still existing as files
                    channel_names = channelNames.copy()
                    for chName in channelNames:
                        chSaved = []
                        for file in filenames:
                            patterns = (
                                f'{chName}.tif', f'{chName}_aligned.npz'
                            )
                            ends = [p for p in patterns if file.endswith(p)]
                            if ends:
                                pattern = ends[0]
                                chSaved.append(True)
                                m = tuple(re.finditer(pattern, file))[-1]
                                chName_idx = m.start()
                                if basename is None:
                                    basename = file[:chName_idx]
                                break
                        if not any(chSaved):
                            channel_names.remove(chName)

                    if basename is not None:
                        self.basenameNotFound = False
                        self.basename = basename
                elif channelNames and basename is not None:
                    self.basename = basename
                    self.basenameNotFound = False
                    channel_names = channelNames

            if channel_names and basename is not None:
                # Add additional channels existing as file but not in metadata.csv
                for file in filenames:
                    ends = [
                        ext for ext in channelExt if (file.endswith(ext) 
                        and not file.endswith('btrack_tracks.h5'))
                    ]
                    if ends:
                        endName = file[len(basename):]
                        chName = endName.replace(ends[0], '')
                        if chName not in channel_names:
                            channel_names.append(chName)
                return channel_names, False
        
        # Find basename as intersection of filenames
        channel_names = set()
        self.basenameNotFound = False
        isBasenamePresent = myutils.checkDataIntegrity(filenames, images_path)
        if basename is None:
            basename = filenames[0]
        basename = filenames[0]
        for file in filenames:
            # Determine the basename based on intersection of all .tif
            _, ext = os.path.splitext(file)
            validFile = False
            if useExt is None:
                validFile = True
            elif ext in useExt and not file.endswith('btrack_tracks.h5'):
                validFile = True
            elif any([file.endswith(end) for end in validEndnames]):
                validFile = True
            else:
                validFile = (
                    (file.find('_acdc_output_') != -1 and ext == '.csv')
                    or (file.find('_segm_') != -1 and ext == '.npz')
                )
            if not validFile:
                continue
            sm = difflib.SequenceMatcher(None, file, basename)
            i, j, k = sm.find_longest_match(0, len(file), 0, len(basename))
            if i > 0:
                continue
            basename = file[i:i+k]
        self.basename = basename
        basenameNotFound = [False]
        for file in filenames:
            filename, ext = os.path.splitext(file)
            validImageFile = False
            if ext in channelExt:
                validImageFile = True
            elif file.endswith('aligned.npz'):
                validImageFile = True
                filename = filename[:-len('_aligned')]
            if not validImageFile:
                continue

            channel_name = filename.split(basename)[-1]
            channel_names.add(channel_name)
            if channel_name == filename:
                # Warn that an intersection could not be found
                basenameNotFound.append(True)
        channel_names = list(channel_names)
        if any(basenameNotFound):
            self.basenameNotFound = True
            filenameNOext, _ = os.path.splitext(basename)
            self.basename = f'{filenameNOext}_'
        if self.which_channel is not None:
            # Search for "phase" and put that channel first on the list
            if self.which_channel == 'segm':
                is_phase_contr_li = [c.lower().find('phase')!=-1
                                     for c in channel_names]
                if any(is_phase_contr_li):
                    idx = is_phase_contr_li.index(True)
                    channel_names[0], channel_names[idx] = (
                                      channel_names[idx], channel_names[0])
        return channel_names, any(basenameNotFound)

    def _load_last_selection(self):
        last_sel_channel = None
        ch = self.which_channel
        if self.which_channel is not None:
            txt_path = os.path.join(settings_folderpath, f'{ch}_last_sel.txt')
            if os.path.exists(txt_path):
                with open(txt_path) as txt:
                    last_sel_channel = txt.read()
        return last_sel_channel

    def _save_last_selection(self, selection):
        ch = self.which_channel
        if self.which_channel is not None:
            if not os.path.exists(settings_folderpath):
                os.mkdir(settings_folderpath)
            txt_path = os.path.join(settings_folderpath, f'{ch}_last_sel.txt')
            with open(txt_path, 'w') as txt:
                txt.write(selection)
    
    def askChannelName(self, filenames, images_path, ask, ch_names):
        from . import apps
        if not ask:
            return ch_names
        filename = self.basename
        possibleChannelNames = []
        splits = [split for split in filename.split('_') if split]
        possibleChannelNames = []
        for i in range(len(splits)-1):
            possibleChanneName = '_'.join(splits[i+1:])
            possibleChannelNames.append(possibleChanneName)
        possibleChannelNames = possibleChannelNames[::-1]

        txt = html_utils.paragraph(f"""
            Cell-ACDC could <b>not determine the channel names</b>.<br><br>
            Please, <b>select</b> below which part of the filename 
            you want to use as <b>the channel name</b>.<br><br>
            Filename: <code>{filename}</code>
        """)
        win = apps.QDialogCombobox(
            'Select channel name', possibleChannelNames, txt, 
            CbLabel='Select channel name:  ', parent=None, centeredCombobox=True
        )
        win.exec_()
        if win.cancel:
            self.was_aborted = True
            return

        channel_name = win.selectedItemText
        basename_idx = self.basename.find(channel_name)
        basename = self.basename[:basename_idx]
        df_metadata, metadata_csv_path = load.get_posData_metadata(
            images_path, basename
        )
        df_metadata.at['channel_0_name', 'values'] = channel_name
        df_metadata.to_csv(metadata_csv_path)
        ch_names, _ = self.get_available_channels(filenames, images_path)
        printl(ch_names)
        return ch_names
        

    def QtPrompt(self, parent, channel_names, informativeText='',
                 CbLabel='Select channel name:  '):
        from . import apps
        font = QFont()
        font.setPixelSize(13)
        win = apps.QDialogCombobox(
            'Select channel name',
            channel_names,
            informativeText,
            CbLabel=CbLabel,
            parent=parent,
            defaultChannelName=self.last_sel_channel
        )
        win.setFont(font)
        win.exec_()
        if win.cancel:
            self.was_aborted = True
        self.channel_name = win.selectedItemText
        self._save_last_selection(self.channel_name)
        self.is_first_call = False

    def setUserChannelName(self):
        if self.basenameNotFound:
            reverse_ch_name = self.channel_name[::-1]
            idx = reverse_ch_name.find('_')
            if idx != -1:
                self.user_ch_name = self.channel_name[-idx:]
            else:
                self.user_ch_name = self.channel_name[-4:]
        else:
            self.user_ch_name = self.channel_name

    def _test(self, name=None, index=None, mode=None):
        pass

    def _abort(self):
        self.was_aborted = True
        if self.allow_abort:
            exit('Execution aborted by the user')

def exportToVideoFinished(
        preferences, conversion_to_mp4_successful, qparent=None
    ):
    from cellacdc import widgets
    
    txt = 'Exporting to video finished!'
    
    msg_type = 'information'
    if not conversion_to_mp4_successful:
        from . import urls
        github_href = html_utils.href_tag('GitHub page', urls.issues_url)
        msg_type = 'warning'
        txt = (
            f'{txt}<br><br>'
            'WARNING: <b>Conversion to MP4 failed</b>. '
            'Video file was saved as AVI instead. '
            f'Feel free to report the issue on our {github_href}'
        )
    
    txt = f'{txt}<br><br>Files were saved here:'
    
    txt = html_utils.paragraph(txt)
    
    
    folderpath = os.path.dirname(preferences['filepath'])
    commands = [preferences['filepath']]
    if preferences['save_pngs']:
        commands.append(preferences['pngs_folderpath'])
    
    msg = widgets.myMessageBox(wrapText=False)
    getattr(msg, msg_type)(
        qparent, 'Exporting video finished', txt, 
        commands=commands, path_to_browse=folderpath
    )