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
    from . import widgets, apps

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
                        and not file.endswith('edited.h5')
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
            if file.endswith('edited.h5'):
                continue
            
            if file.endswith('btrack_tracks.h5'):
                continue
            
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

def exportToImageFinished(filepath, qparent=None):
    from cellacdc import widgets
    
    txt = 'Exporting to image done!'
    txt = f'{txt}<br><br>Files were saved here:'
    
    txt = html_utils.paragraph(txt)
    msg = widgets.myMessageBox(wrapText=False)
    msg.information(
        qparent, 'Exporting image finished', txt, 
        commands=(filepath,), 
        path_to_browse=os.path.dirname(filepath)
    )

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

def askSamSaveEmbeddings(qparent=None):
    txt = html_utils.paragraph("""
    Segment Anything Model generates image embeddings that you 
    can use later in module 3<br>
    for <b>much faster interactive segmentation</b> (with points or bounding boxes 
    prompts).<br><br>
    Do you want to <b>save the image embeddings</b>?
    """)
    saveOnlyButton = widgets.BedPushButton('Save only embeddings')
    saveButton = widgets.BedPlusLabelPushButton('Save also embeddings')
    saveOnlyButton = widgets.BedPushButton('Save only embeddings')
    msg = widgets.myMessageBox(wrapText=False)
    _, saveOnlyButton, saveButton, _ = msg.question(
        qparent, 'Save SAM Image Embeddings?', txt, 
        buttonsTexts=(
            'Cancel', saveOnlyButton, saveButton, 
            widgets.NoBedPushButton('Do not save embeddings')
        )
    )
    sam_only_embeddings = msg.clickedButton == saveOnlyButton
    sam_also_embeddings = msg.clickedButton == saveButton
    return sam_only_embeddings, sam_also_embeddings, msg.cancel

def askSamLoadEmbeddings(
        sam_embeddings_path, qparent=None, is_gui_caller=False
    ):
    txt = html_utils.paragraph("""
    Cell-ACDC detected <b>previously saved Segment Anything image embeddings</b> 
    (see file path below).<br><br>
    If you load the embeddings, computation time will be much lower.<br><br>
    Do you want to <b>load the image embeddings</b>?
    """)
    msg = widgets.myMessageBox(wrapText=False)
    loadButton = widgets.BedPlusLabelPushButton('Load embeddings and segment')
    doNotLoadButton = widgets.NoBedPushButton('Do not load embeddings')
    buttons = (loadButton, doNotLoadButton)
    if is_gui_caller:
        loadOnlyEmbedButton = widgets.BedPushButton('Only load embeddings')
        buttons = (loadOnlyEmbedButton, *buttons)
    
    msg.question(
        qparent, 'Load SAM Image Embeddings?', txt, 
        buttonsTexts=('Cancel', *buttons), 
        commands=(sam_embeddings_path,), 
        path_to_browse=os.path.dirname(sam_embeddings_path)
    )
    loadEmbed = msg.clickedButton == loadButton
    onlyLoadEmbed = False
    if is_gui_caller:
        onlyLoadEmbed = msg.clickedButton == loadOnlyEmbedButton
    return loadEmbed, onlyLoadEmbed, msg.cancel

def init_segm_model_params(
        posData, model_name, init_params, segment_params, 
        qparent=None, help_url=None, init_last_params=False, 
        check_sam_embeddings=True, is_gui_caller=False
    ):
    out = {}
    
    is_sam_model = (
        model_name == 'segment_anything' and check_sam_embeddings
    )
    
    # If SAM with prompts and embeddings were prev saved, asks to load them
    load_sam_embed = False
    only_load_sam_embed = False
    sam_embeddings_exist = os.path.exists(posData.sam_embeddings_path)
    sam_embeddings_loaded = hasattr(posData, 'sam_embeddings')
    if is_sam_model and sam_embeddings_exist and not sam_embeddings_loaded:
        load_sam_embed, only_load_sam_embed, cancel = askSamLoadEmbeddings(
            posData.sam_embeddings_path, qparent=qparent, 
            is_gui_caller=is_gui_caller
        )
        if cancel:
            return out
    
    out['load_sam_embeddings'] = only_load_sam_embed or load_sam_embed
    if only_load_sam_embed:
        return out
    
    segm_files = load.get_segm_files(posData.images_path)
    existingSegmEndnames = load.get_endnames(
        posData.basename, segm_files
    )
    win = apps.QDialogModelParams(
        init_params,
        segment_params,
        model_name, parent=qparent,
        url=help_url, 
        initLastParams=init_last_params, 
        posData=posData,
        segmFileEndnames=existingSegmEndnames,
        df_metadata=posData.metadata_df,
        force_postprocess_2D=False
    )
    win.setChannelNames(posData.chNames)
    win.exec_()
    if win.cancel:
        return out
    
    if load_sam_embed:
        win.model_kwargs['use_loaded_embeddings'] = True
        posData.loadSamEmbeddings()
    
    ask_sam_embeddings = (
        model_name == 'segment_anything' 
        and not load_sam_embed
        and check_sam_embeddings
    )
    # If SAM and embeddings were not laoded, asks to save them
    if ask_sam_embeddings:
        sam_only_embeddings, sam_also_embeddings, cancel = (
            askSamSaveEmbeddings(qparent=qparent)
        )
        if cancel:
            return out

        win.model_kwargs['only_embeddings'] = sam_only_embeddings
        win.model_kwargs['save_embeddings'] = (
            sam_only_embeddings or sam_also_embeddings
        )
    
    out['win'] = win
    
    return out
    