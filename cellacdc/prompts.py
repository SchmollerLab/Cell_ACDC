import os
import re
import time
import traceback
import difflib
import numpy as np
import pandas as pd
import tkinter as tk
import sys
from tkinter import ttk

from qtpy.QtWidgets import (
    QApplication, QPushButton, QHBoxLayout, QLabel, QSizePolicy
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont

from . import myutils, printl, html_utils, load
from . import settings_folderpath

class RichTextPushButton(QPushButton):
    def __init__(self, parent=None, text=None):
        if parent is not None:
            super().__init__(parent)
        else:
            super().__init__()
        self.__lbl = QLabel(self)
        if text is not None:
            self.__lbl.setText(text)
        self.__lyt = QHBoxLayout()
        self.__lyt.setContentsMargins(5, 0, 0, 0)
        self.__lyt.setSpacing(0)
        self.setLayout(self.__lyt)
        self.__lbl.setAttribute(Qt.WA_TranslucentBackground)
        self.__lbl.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.__lbl.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.__lbl.setTextFormat(Qt.RichText)
        self.__lyt.addWidget(self.__lbl)
        return

    def setText(self, text):
        self.__lbl.setText(text)
        self.updateGeometry()
        return

    def sizeHint(self):
        s = QPushButton.sizeHint(self)
        w = self.__lbl.sizeHint()
        s.setWidth(w.width()+10)
        s.setHeight(w.height()+8)
        return s

def folder_dialog(toplevel=False, **options):
    #Prompt the user to select the image file
    if toplevel:
        root = tk.Toplevel()
    else:
        root = tk.Tk()
        root.withdraw()
    path = tk.filedialog.Directory(**options).show()
    root.destroy()
    return path

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


    def prompt(self, channel_names, message=None, toplevel=False):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        root.lift()
        # root.attributes("-topmost", True)
        root.title('Select channel name')
        root.geometry("+800+400")
        row = 0
        if message is not None:
            tk.Label(root,
                     text=message,
                     font=(None, 11)).grid(row=row, column=0,
                                           columnspan= 2, pady=(10,0),
                                                          padx=10)
            row += 1

        tk.Label(root,
                 text='Select channel name to analyse:'
                 ).grid(row=row, column=0, pady=(10,0), padx=10)

        ch_name_var = tk.StringVar()
        w = max([len(s) for s in channel_names])+4
        ch_name_combob = ttk.Combobox(root, width=w, justify='center',
                                      textvariable=ch_name_var)
        ch_name_combob.option_add('*TCombobox*Listbox.Justify', 'center')
        ch_name_combob['values'] = channel_names
        ch_name_combob.grid(column=1, row=row, padx=10, pady=(10,0))
        if self.last_sel_channel is not None:
            if self.last_sel_channel in channel_names:
                ch_name_combob.current(channel_names.index(self.last_sel_channel))
            else:
                ch_name_combob.current(0)
        else:
            ch_name_combob.current(0)

        ch_name_var.trace_add("write", self._test)

        row += 1
        tk.Button(root, text='Ok', width=20,
                        command=self._tk_close).grid(row=row, column=0,
                                                  columnspan=2,
                                                  pady=10, padx=10)



        root.protocol("WM_DELETE_WINDOW", self._tk_abort)
        self.ch_name_var = ch_name_var
        self.root = root

        root.mainloop()

    def _tk_close(self):
        self.was_aborted = False
        self.channel_name = self.ch_name_var.get()
        self.root.quit()
        self.root.destroy()


    def _tk_abort(self):
        self.was_aborted = True
        self.root.quit()
        self.root.destroy()
        if self.allow_abort:
            exit('Execution aborted by the user')

    def _test(self, name=None, index=None, mode=None):
        pass

    def _abort(self):
        self.was_aborted = True
        if self.allow_abort:
            exit('Execution aborted by the user')

def askyesno(title='tk', message='Yes or no?', toplevel=False):
    if toplevel:
        root = tk.Toplevel()
    else:
        root = tk.Tk()
        root.withdraw()
    yes = tk.messagebox.askyesno(title, message, master=root)
    if not toplevel:
        root.quit()
        root.destroy()
    return yes
