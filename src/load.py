import os
import sys
import traceback
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from skimage import io
import skimage.filters
from datetime import datetime
from tifffile import TiffFile
from natsort import natsorted
import skimage
import skimage.measure
from PyQt5 import QtGui
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication
)
import prompts, apps

class loadData:
    def __init__(self, imgPath, user_ch_name, QParent=None):
        self.fluo_data_dict = {}
        self.parent = QParent
        self.imgPath = imgPath
        self.user_ch_name = user_ch_name
        self.images_path = os.path.dirname(imgPath)
        self.pos_path = os.path.dirname(self.images_path)
        self.pos_foldername = os.path.basename(self.pos_path)
        path_li = os.path.normpath(imgPath).split(os.sep)
        self.relPath = f'.../{"/".join(path_li[-3:])}'
        filename_ext = os.path.basename(imgPath)
        self.filename, self.ext = os.path.splitext(filename_ext)

    def getBasenameAndChNames(self, select_channel_name):
        ls = os.listdir(self.images_path)
        selector = select_channel_name()
        self.chNames, _ = selector.get_available_channels(ls)
        self.basename = selector.basename

    def loadImgData(self):
        if self.ext.find('tif') != -1:
            self.img_data = skimage.io.imread(self.imgPath)
        elif self.ext == '.npz':
            self.img_data = np.load(self.imgPath)['arr_0']
        elif self.ext == '.npy':
            self.img_data = np.load(self.imgPath)
        else:
            self.criticalExtNotValid()

    def loadOtherFiles(self,
                       load_segm_data=True,
                       load_acdc_df=False,
                       load_shifts=False,
                       loadSegmInfo=False,
                       load_delROIsInfo=False,
                       loadDataPrepBkgrVals=False,
                       load_last_tracked_i=False,
                       load_metadata=False,
                       getTifPath=False):
        self.segmFound = False if load_segm_data else None
        self.acd_df_found = False if load_acdc_df else None
        self.shiftsFound = False if load_shifts else None
        self.segmInfoFound = False if loadSegmInfo else None
        self.delROIsInfoFound = False if load_delROIsInfo else None
        self.DataPrepBkgrValsFound = False if loadDataPrepBkgrVals else None
        self.last_tracked_i_found = False if load_last_tracked_i else None
        self.metadataFound = False if load_metadata else None
        self.TifPathFound = False if getTifPath else None
        ls = os.listdir(self.images_path)
        for file in ls:
            filePath = os.path.join(self.images_path, file)
            if load_segm_data and file.find('_segm.np')!=-1:
                self.segmFound = True
                self.segm_npz_path = filePath
                try:
                    self.segm_data = np.load(filePath)['arr_0']
                except Exception as e:
                    self.segm_data = np.load(filePath)
            elif getTifPath and file.find(f'_{self.user_ch_name}.tif')!=-1:
                self.tif_path = filePath
                self.TifPathFound = True
            elif load_acdc_df and file.endswith('_acdc_output.csv'):
                self.acd_df_found = True
                acdc_df = pd.read_csv(
                      filePath, index_col=['frame_i', 'Cell_ID']
                )
                acdc_df = self.BooleansTo0s1s(acdc_df, inplace=True)
                acdc_df = self.intToBoolean(acdc_df)
                self.acdc_df = acdc_df
            elif load_shifts and file.endswith('_shifts.npy'):
                self.shiftsFound = True
                self.loaded_shifts = np.load(filePath)
            elif loadSegmInfo and file.endswith('_segmInfo.csv'):
                self.segmInfoFound = True
                self.segmInfo_df = pd.read_csv(filePath,
                                               index_col='frame_i')
            elif load_delROIsInfo and file.endswith('_delROIsInfo.npz'):
                self.delROIsInfoFound = True
                self.delROIsInfo_npz = np.load(filePath)
            elif loadDataPrepBkgrVals and file.endswith('_dataPrep_bkgrValues.csv'):
                self.DataPrepBkgrValsFound = True
                bkgrValues_df = pd.read_csv(filePath)
                self.bkgrValues_chNames = bkgrValues_df['channel_name'].unique()
                self.bkgrValues_df = bkgrValues_df.set_index(
                                                ['channel_name', 'frame_i'])
            elif load_metadata and file.endswith('_metadata.csv'):
                self.metadataFound = True
                self.metadata_df = pd.read_csv(filePath).set_index('Description')
                self.extractMetadata()
            elif load_last_tracked_i and file.endswith('_last_tracked_i.txt'):
                self.last_tracked_i_found = True
                try:
                    with open(filePath, 'r') as txt:
                        self.last_tracked_i = int(txt.read())
                except Exception as e:
                    self.last_tracked_i = None
        self.setNotFoundData()

    def extractMetadata(self):
        if 'SizeT' in self.metadata_df.index:
            self.SizeT = int(self.metadata_df.at['SizeT', 'values'])
        else:
            self.SizeT = 1

        if 'SizeZ' in self.metadata_df.index:
            self.SizeZ = int(self.metadata_df.at['SizeZ', 'values'])
        else:
            self.SizeZ = 1

        if 'TimeIncrement' in self.metadata_df.index:
            self.TimeIncrement = float(
                self.metadata_df.at['TimeIncrement', 'values']
            )
        else:
            self.TimeIncrement = 1

        if 'PhysicalSizeX' in self.metadata_df.index:
            self.PhysicalSizeX = float(
                self.metadata_df.at['PhysicalSizeX', 'values']
            )
        else:
            self.PhysicalSizeX = 1

        if 'PhysicalSizeY' in self.metadata_df.index:
            self.PhysicalSizeY = float(
                self.metadata_df.at['PhysicalSizeY', 'values']
            )
        else:
            self.PhysicalSizeY = 1

        if 'PhysicalSizeZ' in self.metadata_df.index:
            self.PhysicalSizeZ = float(
                self.metadata_df.at['PhysicalSizeZ', 'values']
            )
        else:
            self.PhysicalSizeZ = 1

    def setNotFoundData(self):
        if self.metadataFound is not None and not self.metadataFound:
            self.SizeT, self.SizeZ = len(self.img_data), 1
            self.TimeIncrement = 1.0
            self.PhysicalSizeX = 1.0
            self.PhysicalSizeY = 1.0
            self.PhysicalSizeZ = 1.0
        if self.segmFound is not None and not self.segmFound:
            self.segm_data = None
        if self.acd_df_found is not None and not self.acd_df_found:
            self.acdc_df = None
        if self.shiftsFound is not None and not self.shiftsFound:
            self.loaded_shifts = None
        if self.segmInfoFound is not None and not self.segmInfoFound:
            self.segmInfo_df = None
        if self.delROIsInfoFound is not None and not self.delROIsInfoFound:
            self.delROIsInfo_npz = None
        if self.DataPrepBkgrValsFound is not None and not self.DataPrepBkgrValsFound:
            self.bkgrValues_df = None
        if self.last_tracked_i_found is not None and not self.last_tracked_i_found:
            self.last_tracked_i = None
        if self.TifPathFound is not None and not self.TifPathFound:
            self.tif_path = None

    def buildPaths(self):
        basename = self.basename
        base_path = f'{self.images_path}/{basename}'
        self.slice_used_align_path = f'{base_path}slice_used_alignment.csv'
        self.slice_used_segm_path = f'{base_path}slice_segm.csv'
        self.align_npz_path = f'{base_path}{self.user_ch_name}_aligned.npz'
        self.align_old_path = f'{base_path}phc_aligned.npy'
        self.align_shifts_path = f'{base_path}align_shift.npy'
        self.segm_npz_path = f'{base_path}segm.npz'
        self.last_tracked_i_path = f'{base_path}last_tracked_i.txt'
        self.acdc_output_csv_path = f'{base_path}acdc_output.csv'
        self.segmInfo_df_csv_path = f'{base_path}segmInfo.csv'
        self.delROIs_info_path = f'{base_path}delROIsInfo.npz'
        self.dataPrepROIs_coords_path = f'{base_path}dataPrepROIs_coords.csv'
        self.dataPrepBkgrValues_path = f'{base_path}dataPrep_bkgrValues.csv'
        self.metadata_csv_path = f'{base_path}metadata.csv'

    def setBlankSegmData(self, SizeT, SizeZ, SizeY, SizeX):
        if self.segmFound is not None and not self.segmFound:
            if SizeT > 1:
                self.segm_data = np.zeros((SizeT, Y, X), int)
            else:
                self.segm_data = np.zeros((Y, X), int)

    def loadAllImgPaths(self):
        tif_paths = []
        npy_paths = []
        npz_paths = []
        for filename in os.listdir(self.images_path):
            file_path = os.path.join(self.images_path, filename)
            f, ext = os.path.splitext(filename)
            m = re.match(f'{self.basename}.*\.tif', filename)
            if m is not None:
                tif_paths.append(file_path)
                # Search for npy fluo data
                npy = f'{f}_aligned.npy'
                npz = f'{f}_aligned.npz'
                npy_found = False
                npz_found = False
                for name in os.listdir(self.images_path):
                    _path = os.path.join(self.images_path, name)
                    if name == npy:
                        npy_paths.append(_path)
                        npy_found = True
                    if name == npz:
                        npz_paths.append(_path)
                        npz_found = True
                if not npy_found:
                    npy_paths.append(None)
                if not npz_found:
                    npz_paths.append(None)
        self.tif_paths = tif_paths
        self.npy_paths = npy_paths
        self.npz_paths = npz_paths

    def askInputMetadata(self,
                         ask_TimeIncrement=False,
                         ask_PhysicalSizes=False,
                         save=False):
        font = QtGui.QFont()
        font.setPointSize(10)
        metadataWin = apps.QDialogMetadata(
            self.SizeT, self.SizeZ, self.TimeIncrement,
            self.PhysicalSizeZ, self.PhysicalSizeY, self.PhysicalSizeX,
            ask_TimeIncrement, ask_PhysicalSizes,
            parent=self.parent, font=font, imgDataShape=self.img_data.shape)
        metadataWin.setFont(font)
        metadataWin.exec_()
        if metadataWin.cancel:
            return False

        self.SizeT = metadataWin.SizeT
        self.SizeZ = metadataWin.SizeZ

        source = metadataWin if ask_TimeIncrement else self
        self.TimeIncrement = source.TimeIncrement

        source = metadataWin if ask_PhysicalSizes else self
        self.PhysicalSizeZ = source.PhysicalSizeZ
        self.PhysicalSizeY = source.PhysicalSizeY
        self.PhysicalSizeX = source.PhysicalSizeX
        if save:
            self.saveMetadata()
        return True

    def saveMetadata(self):
        df = pd.DataFrame({
            'SizeT': self.SizeT,
            'SizeZ': self.SizeZ,
            'TimeIncrement': self.TimeIncrement,
            'PhysicalSizeZ': self.PhysicalSizeZ,
            'PhysicalSizeY': self.PhysicalSizeY,
            'PhysicalSizeX': self.PhysicalSizeX
        }, index=['values']).T
        df.index.name = 'Description'
        df.to_csv(self.metadata_csv_path)



    @staticmethod
    def BooleansTo0s1s(acdc_df, csv_path=None, inplace=True):
        """
        Function used to convert "FALSE" strings and booleans to 0s and 1s
        to avoid pandas interpreting as strings or numbers
        """
        if not inplace:
            acdc_df = acdc_df.copy()
        colsToCast = ['is_cell_dead', 'is_cell_excluded']
        for col in colsToCast:
            isInt = pd.api.types.is_integer_dtype(acdc_df[col])
            isFloat = pd.api.types.is_float_dtype(acdc_df[col])
            isObject = pd.api.types.is_object_dtype(acdc_df[col])
            isString = pd.api.types.is_string_dtype(acdc_df[col])
            isBool = pd.api.types.is_bool_dtype(acdc_df[col])
            if isFloat or isBool:
                acdc_df[col] = acdc_df[col].astype(int)
            elif isString or isObject:
                acdc_df[col] = (acdc_df[col].str.lower() == 'true').astype(int)
        if csv_path is not None:
            acdc_df.to_csv(csv_path)
        return acdc_df

    def intToBoolean(self, acdc_df):
        colsToCast = ['is_cell_dead', 'is_cell_excluded']
        for col in colsToCast:
            acdc_df[col] = acdc_df[col] > 0
        return acdc_df


    def criticalExtNotValid(self):
        err_title = f'File extension {self.ext} not valid.'
        err_msg = (
            f'The requested file {self.relPath}\n'
            'has an invalid extension.\n\n'
            'Valid extensions are .tif, .tiff, .npy or .npz'
        )
        if self.parent is None:
            print('-------------------------')
            print(err_msg)
            print('-------------------------')
            raise FileNotFoundError(err_title)
        else:
            print('-------------------------')
            print(err_msg)
            print('-------------------------')
            msg = QtGui.QMessageBox()
            msg.critical(self.parent, err_title, err_msg, msg.Ok)
            return None



class fix_pos_n_mismatch:
    '''Geometry: "WidthxHeight+Left+Top" '''
    def __init__(self, title, message, button_1_text='Ignore',
                 button_2_text='Fix it!',
                 button_3_text='Show in Explorer', path=None):
        self.path=path
        self.ignore=False
        root = tk.Tk()
        self.root = root
        root.lift()
        # root.attributes("-topmost", True)
        root.title(title)
        # root.geometry(geometry)
        tk.Label(root,
                 text=message,
                 font=(None, 11)).grid(row=0, column=0,
                                       columnspan=3, pady=4, padx=4)

        tk.Button(root,
                  text=button_1_text,
                  command=self.ignore_cb,
                  width=10,).grid(row=4,
                                  column=0,
                                  pady=8, padx=4)

        tk.Button(root,
                  text=button_2_text,
                  command=self.fix_cb,
                  width=15).grid(row=4,
                                 column=1,
                                 pady=8, padx=4)
        tk.Button(root,
                  text=button_3_text,
                  command=self.open_path_explorer,
                  width=25).grid(row=4,
                                 column=2, padx=4)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        root.mainloop()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

    def ignore_cb(self):
        self.ignore=True
        self.root.quit()
        self.root.destroy()

    def open_path_explorer(self):
        subprocess.Popen('explorer "{}"'.format(os.path.normpath(self.path)))

    def fix_cb(self):
        self.root.quit()
        self.root.destroy()

class beyond_listdir_pos:
    def __init__(self, folder_path):
        self.bp = apps.tk_breakpoint()
        self.folder_path = folder_path
        self.TIFFs_path = []
        self.count_recursions = 0
        # self.walk_directories(folder_path)
        self.listdir_recursion(folder_path)
        if not self.TIFFs_path:
            raise FileNotFoundError(f'Path {folder_path} is not valid!')
        self.all_exp_info = self.count_segmented_pos()

    def listdir_recursion(self, folder_path):
        if os.path.isdir(folder_path):
            listdir_folder = natsorted(os.listdir(folder_path))
            contains_pos_folders = any([name.find('Position_')!=-1
                                        for name in listdir_folder])
            if not contains_pos_folders:
                contains_TIFFs = any([name=='TIFFs' for name in listdir_folder])
                contains_CZIs = any([name=='CZIs' for name in listdir_folder])
                contains_czis_files = any([name.find('.czi')!=-1
                                           for name in listdir_folder])
                if contains_TIFFs:
                    self.TIFFs_path.append(f'{folder_path}/TIFFs')
                elif contains_CZIs:
                    self.TIFFs_path.append(f'{folder_path}')
                elif not contains_CZIs and contains_czis_files:
                    self.TIFFs_path.append(f'{folder_path}/CZIs')
                else:
                    for name in listdir_folder:
                        subfolder_path = f'{folder_path}/{name}'
                        self.listdir_recursion(subfolder_path)
            else:
                self.TIFFs_path.append(folder_path)

    def walk_directories(self, folder_path):
        for root, dirs, files in os.walk(folder_path, topdown=True):
            # Avoid scanning TIFFs and CZIs folder
            dirs[:] = [d for d in dirs if d not in ['TIFFs', 'CZIs', 'Original_TIFFs']]
            contains_czis_files = any([name.find('.czi')!=-1 for name in files])
            print(root)
            print(dirs, files)
            self.bp.pausehere()
            for dirname in dirs:
                path = f'{root}/{dirname}'
                listdir_folder = natsorted(os.listdir(path))
                if dirname == 'TIFFs':
                    self.TIFFs_path.append(path)
                    print(self.TIFFs_path)
                    break

    def get_rel_path(self, path):
        rel_path = ''
        images_path = path
        count = 0
        while images_path != self.folder_path or count==10:
            if count > 0:
                rel_path = f'{os.path.basename(images_path)}/{rel_path}'
            images_path = os.path.dirname(images_path)
            count += 1
        rel_path = f'.../{rel_path}'
        return rel_path

    def count_segmented_pos(self):
        all_exp_info = []
        for path in self.TIFFs_path:
            foldername = os.path.basename(path)
            if foldername == 'TIFFs':
                pos_foldernames = natsorted([p for p in os.listdir(path)
                             if os.path.isdir(os.path.join(path, p))
                             and p.find('Position_') != -1])
                num_pos = len(pos_foldernames)
                if num_pos == 0:
                    root = tk.Tk()
                    root.withdraw()
                    delete_empty_TIFFs = messagebox.askyesno(
                    'Folder will be deleted!',
                    f'WARNING: The folder\n\n {path}\n\n'
                    'does not contain any file!\n'
                    'It will be DELETED!\n'
                    'Are you sure you want to continue?\n\n')
                    root.quit()
                    root.destroy()
                    if delete_empty_TIFFs:
                        os.rmdir(path)
                    rel_path = self.get_rel_path(path)
                    exp_info = f'{rel_path} (FIJI macro not executed!)'
                else:
                    rel_path = self.get_rel_path(path)
                    pos_ok = False
                    while not pos_ok:
                        num_segm_pos = 0
                        pos_paths_multi_segm = []
                        tmtimes = []
                        for pos_foldername in pos_foldernames:
                            images_path = f'{path}/{pos_foldername}/Images'
                            filenames = os.listdir(images_path)
                            count = 0
                            m = re.findall('Position_(\d+)', pos_foldername)
                            mismatch_paths = []
                            pos_n = int(m[0])
                            is_mismatch = False
                            for filename in filenames:
                                m = re.findall('_s(\d+)_', filename)
                                if not m:
                                    m = re.findall('_s(\d+)-', filename)
                                if not m:
                                    continue
                                s_n = int(m[0])
                                if s_n == pos_n:
                                    if filename.find('segm.np') != -1:
                                        file_path = f'{images_path}/{filename}'
                                        tmtime = os.path.getmtime(file_path)
                                        tmtimes.append(tmtime)
                                        num_segm_pos += 1
                                        count += 1
                                        if count > 1:
                                            pos_paths_multi_segm.append(
                                                                    images_path)
                                else:
                                    is_mismatch = True
                                    file_path = f'{images_path}/{filename}'
                                    mismatch_paths.append(file_path)
                            if is_mismatch:
                                fix = fix_pos_n_mismatch(
                                      title='Filename mismatch!',
                                      message='The following position contains '
                                      'files that do not belong to the '
                                      f'Position_n folder:\n\n {images_path}\n\n'
                                      f'The Position number according to the folder is {pos_n}\n'
                                      f'while the position number according to the file name is {s_n}'
                                      f' (i.e. ".._s{s_n}_")\n\n'
                                      f'File name: {filename}',
                                      path=images_path)
                                if not fix.ignore:
                                    paths_print = ',\n\n'.join(mismatch_paths)
                                    root = tk.Tk()
                                    root.withdraw()
                                    do_it = messagebox.askyesno(
                                    'Files will be deleted!',
                                    'WARNING: The files below will be DELETED!\n'
                                    'Are you sure you want to continue?\n\n'
                                    f'{paths_print}')
                                    root.quit()
                                    root.destroy()
                                    if do_it:
                                        for mismatch_path in mismatch_paths:
                                            os.remove(mismatch_path)
                                    pos_ok = False
                                else:
                                    pos_ok = True
                            else:
                                pos_ok = True
                if num_segm_pos < num_pos:
                    if num_segm_pos != 0:
                        exp_info = (f'{rel_path} (N. of segmented pos: '
                                    f'{num_segm_pos})')
                    else:
                        exp_info = (f'{rel_path} '
                                     '(NONE of the pos have been segmented)')
                elif num_segm_pos == num_pos:
                    if num_pos != 0:
                        tmtime = max(tmtimes)
                        modified_on = (datetime.utcfromtimestamp(tmtime)
                                               .strftime('%Y/%m/%d'))
                        exp_info = f'{rel_path} (All pos segmented - {modified_on})'
                elif num_segm_pos > num_pos:
                    print('Position_n folders that contain multiple segm.np files:\n'
                          f'{pos_paths_multi_segm}')
                    exp_info = f'{rel_path} (WARNING: multiple "segm.np" files found!)'
                else:
                    exp_info = rel_path
            else:
                rel_path = self.get_rel_path(path)
                exp_info = f'{rel_path} (FIJI macro not executed!)'
            all_exp_info.append(exp_info)
        return all_exp_info

class select_exp_folder:
    def QtPrompt(self, parentQWidget, values,
                       current=0,
                       title='Select Position folder',
                       CbLabel="Select \'Position_n\' folder to analyze:",
                       showinexplorer_button=False,
                       full_paths=None, allow_abort=True,
                       show=False, toggleMulti=False):
        font = QtGui.QFont()
        font.setPointSize(10)
        win = apps.QtSelectItems(title, values, '',
                               CbLabel=CbLabel,
                               parent=parentQWidget)
        win.setFont(font)
        toFront = win.windowState() & ~Qt.WindowMinimized | Qt.WindowActive
        win.setWindowState(toFront)
        win.activateWindow()
        if toggleMulti:
            win.multiPosButton.setChecked(True)
        win.exec_()
        self.was_aborted = win.cancel
        if not win.cancel:
            self.selected_pos = [self.pos_foldernames[idx]
                                 for idx in win.selectedItemsIdx]

    def run_widget(self, values, current=0,
                   title='Select Position folder',
                   label_txt="Select \'Position_n\' folder to analyze:",
                   showinexplorer_button=False,
                   full_paths=None, allow_abort=True,
                   toplevel=False):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        width = max([len(value) for value in values])
        root.geometry('+800+400')
        root.title(title)
        root.lift()
        # root.attributes("-topmost", True)
        self.full_paths=full_paths
        self.was_aborted = False
        self.allow_abort = allow_abort
        # Label
        ttk.Label(root, text = label_txt,
                  font = (None, 10)).grid(column=0, row=0, padx=10, pady=10)

        # Ok button
        ok_b = ttk.Button(root, text='Ok!', comman=self._close)
        ok_b.grid(column=0, row=1, pady=10, sticky=tk.E)
        self.ok_b = ok_b

        # All button
        if len(values) > 1:
            all_b = ttk.Button(root, text='All positions', comman=self._all_cb)
            all_b.grid(column=1, row=1, pady=10)
            self.all_b = all_b

        self.root = root

        # Combobox
        pos_n_sv = tk.StringVar()
        self.pos_n_sv = pos_n_sv
        self.pos_n_sv.trace_add("write", self._check_fiji_macro)
        self.values = values
        pos_b_combob = ttk.Combobox(root, textvariable=pos_n_sv, width=width)
        pos_b_combob['values'] = values
        pos_b_combob.grid(column=1, row=0, padx=10, columnspan=2)
        pos_b_combob.current(current)

        # Show in explorer button
        if showinexplorer_button:
            show_expl_button = ttk.Button(root, text='Show in explorer',
                                          comman=self.open_path_explorer)
            show_expl_button.grid(column=2, row=1, pady=10)

        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        root.mainloop()

    def _check_fiji_macro(self, name=None, index=None, mode=None):
        path_info = self.pos_n_sv.get()
        if path_info.find('FIJI macro not executed') != -1:
            self.ok_b.configure(text='Exit', comman=self.on_closing)
            more_info = ttk.Button(self.root, text='More info',
                                          comman=self._more_info)
            more_info.grid(column=2, row=1, pady=10)

    def _more_info(self):
        tk.messagebox.showwarning(title='FIJI Macro not executed',
            message='The script could not find the "Position_n folders"!\n\n'
            'This is most likely because you did not run the Fiji macro\n'
            'that creates the correct folder structure expected by the GUI loader.\n\n'
            'See the section "Preparing your data" on the GitHub repo for more info.' )

    def _all_cb(self):
        self.selected_pos = self.pos_foldernames
        self.root.quit()
        self.root.destroy()

    def open_path_explorer(self):
        if self.full_paths is None:
            path = self.pos_n_sv.get()
        else:
            sv_txt = self.pos_n_sv.get()
            sv_idx = self.values.index(sv_txt)
            path = self.full_paths[sv_idx]
        systems = {
            'nt': os.startfile,
            'posix': lambda foldername: os.system('xdg-open "%s"' % foldername),
            'os2': lambda foldername: os.system('open "%s"' % foldername)
             }

        systems.get(os.name, os.startfile)(path)

    def get_values_segmGUI(self, exp_path):
        pos_foldernames = natsorted(os.listdir(exp_path))
        pos_foldernames = [f for f in pos_foldernames
                           if f.find('Position_')!=-1
                           and os.path.isdir(f'{exp_path}/{f}')]
        self.pos_foldernames = pos_foldernames
        values = []
        for pos in pos_foldernames:
            last_tracked_i_found = False
            pos_path = f'{exp_path}/{pos}'
            if os.path.isdir(pos_path):
                images_path = f'{exp_path}/{pos}/Images'
                filenames = os.listdir(images_path)
                for filename in filenames:
                    if filename.find('_last_tracked_i.txt') != -1:
                        last_tracked_i_found = True
                        last_tracked_i_path = f'{images_path}/{filename}'
                        try:
                            with open(last_tracked_i_path, 'r') as txt:
                                last_tracked_i = int(txt.read())
                        except Exception as e:
                            last_tracked_i_found = False
                if last_tracked_i_found:
                    values.append(f'{pos} (Last tracked frame: {last_tracked_i+1})')
                else:
                    values.append(pos)
        self.values = values
        return values

    def get_values_cca(self, exp_path):
        pos_foldernames = natsorted(os.listdir(exp_path))
        pos_foldernames = [pos for pos in pos_foldernames
                               if re.match('Position_(\d+)', pos)]
        self.pos_foldernames = pos_foldernames
        values = []
        for pos in pos_foldernames:
            cc_stage_found = False
            pos_path = f'{exp_path}/{pos}'
            if os.path.isdir(pos_path):
                images_path = f'{exp_path}/{pos}/Images'
                filenames = os.listdir(images_path)
                for filename in filenames:
                    if filename.find('cc_stage.csv') != -1:
                        cc_stage_found = True
                        cc_stage_path = f'{images_path}/{filename}'
                        cca_df = pd.read_csv(cc_stage_path,
                                             index_col=['frame_i', 'Cell_ID'])
                        last_analyzed_frame_i = (cca_df.index.
                                                      get_level_values(0).max())
                if cc_stage_found:
                    values.append(f'{pos} (Last analyzed frame: '
                                  f'{last_analyzed_frame_i})')
                else:
                    values.append(pos)
        self.values = values
        return values

    def _close(self):
        val = self.pos_n_sv.get()
        idx = list(self.values).index(val)
        if self.full_paths is None:
            self.selected_pos = [self.pos_foldernames[idx]]
        else:
            self.TIFFs_path = self.full_paths[idx]
        self.root.quit()
        self.root.destroy()

    def on_closing(self):
        self.selected_pos = [None]
        self.was_aborted = True
        self.root.quit()
        self.root.destroy()
        if self.allow_abort:
            exit('Execution aborted by the user')


def get_main_paths(selected_path, vNUM):
    is_pos_path = os.path.basename(selected_path).find('Position_') != -1
    is_TIFFs_path = any([f.find('Position_')!=-1
                         and os.path.isdir(f'{selected_path}/{f}')
                         for f in os.listdir(selected_path)])
    multi_run_msg = ('Multiple runs detected!\n\n'
                     'Select which run number you want to analyse.')
    if not is_pos_path and not is_TIFFs_path:
        selector = select_exp_folder()
        beyond_listdir = beyond_listdir_pos(selected_path)
        main_paths = selector.run_widget(beyond_listdir.all_exp_info,
                             title='Select experiment to segment',
                             label_txt='Select experiment to segment',
                             full_paths=beyond_listdir.TIFFs_path,
                             showinexplorer_button=True)
        prompts_pos_to_analyse = False
        # run_num = beyond_listdir.run_num
    elif is_TIFFs_path:
        # The selected path is already the folder containing Position_n folders
        prompts_pos_to_analyse = True
        main_paths = [selected_path]
        ls_selected_path = os.listdir(selected_path)
        pos_foldername = [p for p in ls_selected_path
                          if p.find('Position_') != -1
                          and os.path.isdir(os.path.join(selected_path, p))][0]
        pos_path = os.path.join(selected_path, pos_foldername)
    elif is_pos_path:
        prompts_pos_to_analyse = False
        main_paths = [selected_path]
        pos_path = selected_path
    else:
        raise FileNotFoundError('Invalid path.'
        f'The selected path {selected_path} is neither a specific position folder '
        'nor the TIFFs folder.')
    run_num = None
    return (main_paths, prompts_pos_to_analyse, run_num, is_pos_path,
            is_TIFFs_path)


def load_shifts(parent_path, basename=None):
    shifts_found = False
    shifts = None
    if basename is None:
        for filename in os.listdir(parent_path):
            if filename.find('align_shift.npy')>0:
                shifts_found = True
                shifts_path = os.path.join(parent_path, filename)
                shifts = np.load(shifts_path)
    else:
        align_shift_fn = f'{basename}_align_shift.npy'
        if align_shift_fn in os.listdir(parent_path):
            shifts_found = True
            shifts_path = os.path.join(parent_path, align_shift_fn)
            shifts = np.load(shifts_path)
        else:
            shifts = None
    return shifts, shifts_found
