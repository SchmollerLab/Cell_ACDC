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
import skimage.measure
from PyQt5 import QtGui
from PyQt5.QtWidgets import (
    QApplication
)
import prompts, apps

class load_frames_data:
    def __init__(self, path, user_ch_name, parentQWidget=None,
                 load_segm_data=True,
                 load_segm_metadata=True):
        self.path = path
        self.fluo_data_dict = {}
        self.images_path = os.path.dirname(path)
        filename_ext = os.path.basename(path)
        self.filename, self.ext = os.path.splitext(filename_ext)
        basename_idx = filename_ext.find(f'_{user_ch_name}')
        self.basename = filename_ext[0:basename_idx]
        if self.ext == '.tif' or self.ext == '.tiff':
            tif_path, img_tif_found = self.substring_path(path,
                                                         f'{user_ch_name}.tif',
                                                          self.images_path)
            if img_tif_found:
                self.tif_path = tif_path
                img_data = io.imread(path)
            else:
                err_title = f'Selected {user_ch_name} file not found!'
                err_msg = (
                    f'{user_ch_name} .tif file not found in the selected path\n'
                    f'{self.images_path}!\n Make sure that the folder contains '
                    f'a file that ends with either "{user_ch_name}"'
                )
                if parentQWidget is None:
                    app = QApplication([])
                    msgBox = QtGui.QMessageBox()
                    msgBox.setWindowTitle(err_title)
                    msgBox.setText(err_msg)
                    msgBox.setIcon(msgBox.Critical)
                    msgBox.setModal(True)
                    msgBox.show()
                    app.exec_()
                    raise FileNotFoundError(err_title)
                else:
                    msg = QtGui.QMessageBox()
                    msg.critical(parentQWidget, err_title, err_msg, msg.Ok)
                    return None
        elif self.ext == '.npy' or self.ext == '.npz':
            if path.find(f'{user_ch_name}_aligned.np') == -1:
                filename = os.path.basename(path)
                err_title = 'Wrong file selected!'
                err_msg = (
                    f'You selected a file called {filename} which is not a valid '
                    'phase contrast image file. Select the file that ends with '
                    f'"{user_ch_name}_aligned.npz" or the .tif phase contrast file'
                )
                if parentQWidget is None:
                    app = QApplication([])
                    msgBox = QtGui.QMessageBox()
                    msgBox.setWindowTitle(err_title)
                    msgBox.setText(err_msg)
                    msgBox.setIcon(msgBox.Critical)
                    msgBox.setModal(True)
                    msgBox.show()
                    app.exec_()
                    raise FileNotFoundError(err_title)
                else:
                    msg = QtGui.QMessageBox()
                    msg.critical(parentQWidget, err_title, err_msg, msg.Ok)
                    return None
            tif_path, img_tif_found = self.substring_path(
                                                  path, f'{user_ch_name}.tif',
                                                  self.images_path)
            if img_tif_found:
                self.tif_path = tif_path
                img_data = np.load(path)
                try:
                    img_data = img_data['arr_0']
                except:
                    img_data = img_data
            else:
                err_title = 'Phase contrast file not found!'
                err_msg = (
                    'Phase contrast .tif file not found in the selected path\n'
                    f'{self.images_path}!\n Make sure that the folder contains '
                    'a file that ends with either \"phase_contr.tif\" or '
                    '\"phase_contrast.tif\"'
                )
                if parentQWidget is None:
                    app = QApplication([])
                    msgBox = QtGui.QMessageBox()
                    msgBox.setWindowTitle(err_title)
                    msgBox.setText(err_msg)
                    msgBox.setIcon(msgBox.Critical)
                    msgBox.setModal(True)
                    msgBox.show()
                    app.exec_()
                    raise FileNotFoundError(err_title)
                else:
                    msg = QtGui.QMessageBox()
                    msg.critical(parentQWidget, err_title, err_msg, msg.Ok)
                    return None
        self.img_data = img_data
        self.info, self.metadata_found = self.metadata(self.tif_path)
        prompt_user = False
        if self.metadata_found:
            try:
                self.SizeT, self.SizeZ = self.data_dimensions(self.info)
            except:
                self.SizeT, self.SizeZ = 1, 1
            try:
                self.zyx_vox_dim = self.zyx_vox_dim()
                zyx_vox_dim_found = True
            except:
                self.zyx_vox_dim = [0.5, 0.01, 0.01]
                zyx_vox_dim_found = False
            (self.SizeT, self.SizeZ,
            self.zyx_vox_dim) = self.inputsWidget(
                SizeT=self.SizeT, SizeZ=self.SizeZ,
                zyx_vox_dim=self.zyx_vox_dim,
                zyx_vox_dim_found=zyx_vox_dim_found,
                parent=parentQWidget
            )
        else:
            (self.SizeT, self.SizeZ,
            self.zyx_vox_dim) = self.inputsWidget(parent=parentQWidget)
        data_T, data_Z = self.img_data.shape[:2]
        if self.SizeZ > 1:
            if data_Z != self.SizeZ:
                root = tk.Tk()
                root.withdraw()
                tk.messagebox.showwarning('Shape mismatch!',
                    'The metadata of the .tif file says that there should be '
                    f'{self.SizeZ} z-slices. However the shape of the data is '
                    f'{self.img_data.shape}!\n\n'
                    'The order of the data dimensions for 3D datasets '
                    'has to be TZYX where T is the number of frames, Z the '
                    'number of slices, and YX the shape of the image.\n\n'
                    'In your case it looks like you either have a single 3D image'
                    ' (no frames), or you have 2D data over time.\n'
                    'In the first case, you should not use this script but the '
                    '"gui_snapshots.py" script. For the second case the '
                    'software will now try to ignore the number '
                    'of slices and it will suppose that your data is 2D.')
                self.SizeZ = 1
                root.quit()
                root.destroy()
            if data_T != self.SizeT:
                root = tk.Tk()
                root.withdraw()
                tk.messagebox.showwarning('Shape mismatch!',
                    'The metadata of the .tif file says that there should be '
                    f'{self.SizeT} frame. However the shape of the data is '
                    f'{self.img_data.shape}!\n\n'
                    'The order of the data dimensions has to be TZYX for '
                    '3D images over time and TYX for 2D images, '
                    'where T is the number of frames, Z the '
                    'number of slices, and YX the shape of the image.\n\n'
                    'In your case it looks like your data contains less/more '
                    'frames than expected.\n\n'
                    f'The software will now try to run with {data_T} '
                    'number of frames.')
                self.SizeT = data_T
                root.quit()
                root.destroy()
        self.segm_data = None
        if load_segm_data:
            segm_npz_path, segm_npy_found = self.substring_path(
                                           path, 'segm.npz', self.images_path)
            if not segm_npy_found:
                segm_npz_path, segm_npy_found = self.substring_path(
                                           path, 'segm.npy', self.images_path)
            self.segm_npy_found = segm_npy_found
            if segm_npy_found:
                segm_data = np.load(segm_npz_path)
                try:
                    self.segm_data = segm_data['arr_0']
                except:
                    self.segm_data = segm_data
            else:
                Y, X = self.img_data.shape[-2:]
                self.segm_data = np.zeros((self.SizeT, Y, X), int)
        # Load last tracked frame
        last_tracked_i_path, last_tracked_i_found = self.substring_path(
                                              path, 'last_tracked_i.txt',
                                              self.images_path)
        if last_tracked_i_found:
            with open(last_tracked_i_path, 'r') as txt:
                self.last_tracked_i = int(txt.read())
        else:
            self.last_tracked_i = None

        self.acdc_df = None
        # Load segmentation metadata
        if load_segm_metadata:
            segm_metadata_path, segm_metadata_found = self.substring_path(
                                              path, '_acdc_output.csv',
                                              self.images_path)
            if segm_metadata_found:
                acdc_df = pd.read_csv(
                    segm_metadata_path, index_col=['frame_i', 'Cell_ID']
                )

                # Keep compatibility with older versions of acdc_df
                if 'Is_dead_cell' in acdc_df.columns:
                    acdc_df.rename(
                        columns={'Is_dead_cell': 'is_cell_dead',
                                 'centroid_x_dead': 'x_centroid',
                                 'centroid_y_dead': 'y_centroid'},
                        inplace=True
                    )
                    acdc_df['is_cell_excluded'] = False

                self.acdc_df = acdc_df

        self.build_paths(self.filename, self.images_path, user_ch_name)

    def zyx_vox_dim(self):
        info = self.info
        try:
            scalint_str = "Scaling|Distance|Value #"
            len_scalin_str = len(scalint_str) + len("1 = ")
            px_x_start_i = info.find(scalint_str + "1 = ") + len_scalin_str
            px_x_end_i = info[px_x_start_i:].find("\n") + px_x_start_i
            px_x = float(info[px_x_start_i:px_x_end_i])*1E6 #convert m to µm
            px_y_start_i = info.find(scalint_str + "2 = ") + len_scalin_str
            px_y_end_i = info[px_y_start_i:].find("\n") + px_y_start_i
            px_y = float(info[px_y_start_i:px_y_end_i])*1E6
            try:
                px_z_start_i = info.find(scalint_str + "3 = ") + len_scalin_str
                px_z_end_i = info[px_z_start_i:].find("\n") + px_z_start_i
                px_z = float(info[px_z_start_i:px_z_end_i])*1E6
            except:
                px_z = 1
        except:
            x_res_match = re.findall('XResolution = ([0-9]*[.]?[0-9]+)', info)
            px_x = 1/float(x_res_match[0])
            y_res_match = re.findall('YResolution = ([0-9]*[.]?[0-9]+)', info)
            px_y = 1/float(y_res_match[0])
            try:
                z_spac_match = re.findall('Spacing = ([0-9]*[.]?[0-9]+)', info)
                px_z = float(z_spac_match[0])
            except:
                px_z = 1
        return [px_z, px_y, px_x]



    def build_paths(self, filename, images_path, user_ch_name):
        basename = self.basename
        base_path = f'{images_path}/{basename}'
        self.slice_used_align_path = f'{base_path}_slice_used_alignment.csv'
        self.slice_used_segm_path = f'{base_path}_slice_segm.csv'
        self.align_npz_path = f'{base_path}_{user_ch_name}_aligned.npz'
        self.align_old_path = f'{base_path}_phc_aligned.npy'
        self.align_shifts_path = f'{base_path}_align_shift.npy'
        self.segm_npz_path = f'{base_path}_segm.npz'
        self.last_tracked_i_path = f'{base_path}_last_tracked_i.txt'
        self.acdc_output_csv_path = f'{base_path}_acdc_output.csv'
        self.benchmarking_df_csv_path = f'{base_path}_benchmarking.csv'

    def substring_path(self, path, substring, images_path):
        substring_found = False
        for filename in os.listdir(images_path):
            if substring == "phase_contr.tif":
                is_match = (filename.find(substring) != -1 or
                            filename.find("phase_contrast.tif") != -1 or
                            filename.find("phase_contrast.tiff") != -1 or
                            filename.find("phase_contr.tiff") != -1)
            else:
                is_match = filename.find(substring) != -1
            if is_match:
                substring_found = True
                break
        substring_path = f'{images_path}/{filename}'
        return substring_path, substring_found


    def metadata(self, tif_path):
        with TiffFile(tif_path) as tif:
            self.metadata = tif.imagej_metadata
        try:
            metadata_found = True
            info = self.metadata['Info']
        except:
            metadata_found = False
            info = []
        return info, metadata_found

    def data_dimensions(self, info):
        SizeT = int(re.findall('SizeT = (\d+)', info)[0])
        SizeZ = int(re.findall('SizeZ = (\d+)', info)[0])
        return SizeT, SizeZ

    def inputsWidget(self, parent=None, SizeZ=1, SizeT=1,
                     zyx_vox_dim=[0.5,0.1,0.1], zyx_vox_dim_found=False):
        src_path = os.path.dirname(os.path.realpath(__file__))
        last_entries_csv_path = os.path.join(
            src_path, 'temp', 'last_entries_metadata.csv'
        )
        if os.path.exists(last_entries_csv_path) and not zyx_vox_dim_found:
            df = pd.read_csv(last_entries_csv_path, index_col='Description')
            z, y, x = df.at[['z_voxSize', 'y_voxSize', 'x_voxSize'], 'values']
            zyx_vox_dim = (z, y, x)

        if parent is None:
            app = QApplication([])
            win = apps.QDialogInputsForm(SizeT, SizeZ, zyx_vox_dim)
            win.show()
            app.setStyle(QtGui.QStyleFactory.create('Fusion'))
            app.exec_()
        else:
            win = apps.QDialogInputsForm(SizeT, SizeZ, zyx_vox_dim, parent=parent)
            win.exec_()
        self.cancel = win.cancel
        return win.SizeT, win.SizeZ, win.zyx_vox_dim


    def dimensions_entry_widget(self, SizeZ=1, SizeT=1,
                                zyx_vox_dim=[0.5,0.1,0.1],
                                zyx_vox_dim_found=False):
        src_path = os.path.dirname(os.path.realpath(__file__))
        last_entries_csv_path = os.path.join(
            src_path, 'temp', 'last_entries_metadata.csv'
        )
        if os.path.exists(last_entries_csv_path) and not zyx_vox_dim_found:
            df = pd.read_csv(last_entries_csv_path, index_col='Description')
            zyx_vox_dim = df.at['zyx_vox_dim', 'values']

        root = tk.Tk()
        root.geometry("+800+400")
        root.title('Provide metadata')
        tk.Label(root,
                 anchor='w',
                 text="Provide the following constants:",
                 font=(None, 12)).grid(row=0, column=0, columnspan=2, pady=4)
        tk.Label(root,
                 anchor='w',
                 text="Number of frames (SizeT)",
                 font=(None, 10)).grid(row=1, pady=4)
        tk.Label(root,
                 anchor='w',
                 text="Number of slices (SizeZ)",
                 font=(None, 10)).grid(row=2, pady=4, padx=8)
        tk.Label(root,
                 anchor='w',
                 text="Z, Y, X voxel size (um/pxl)\n""For 2D images leave Z to 1",
                 font=(None, 10)).grid(row=3, pady=4, padx=8)

        # root.protocol("WM_DELETE_WINDOW", exit)

        SizeT_entry = tk.Entry(root, justify='center')
        SizeZ_entry = tk.Entry(root, justify='center')
        zyx_vox_dim_entry = tk.Entry(root, justify='center')

        # Default texts in entry text box
        SizeT_entry.insert(0, f'{SizeT}')
        SizeZ_entry.insert(0, f'{SizeZ}')
        zyx_vox_dim_entry.insert(0, f'{zyx_vox_dim}')

        SizeT_entry.grid(row=1, column=1, padx=8)
        SizeZ_entry.grid(row=2, column=1, padx=8)
        zyx_vox_dim_entry.grid(row=3, column=1, padx=8)

        tk.Button(root,
                  text='OK',
                  command=root.quit,
                  width=10).grid(row=4,
                                 column=0,
                                 pady=16,
                                 columnspan=2)
        SizeT_entry.focus()

        root.protocol("WM_DELETE_WINDOW", self.do_nothing)

        root.mainloop()

        SizeT = int(SizeT_entry.get())
        SizeZ = int(SizeZ_entry.get())
        re_float = '([0-9]*[.]?[0-9]+)'
        s = zyx_vox_dim_entry.get()
        m = re.findall(f'{re_float}, {re_float}, {re_float}', s)
        zyx_vox_dim = [float(f) for f in m[0]]
        root.destroy()

        # Save values to load them again at the next session
        df = pd.DataFrame(
            {'Description': ['SizeT', 'SizeZ', 'zyx_vox_dim'],
             'values': [SizeT, SizeZ, zyx_vox_dim]}
        ).set_index('Description')
        df.to_csv(last_entries_csv_path)

        return SizeT, SizeZ, zyx_vox_dim

    def do_nothing(self):
        pass

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
        root.attributes("-topmost", True)
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
        root.attributes("-topmost", True)
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
            subprocess.Popen('explorer "{}"'.format(os.path.normpath(path)))
        else:
            sv_txt = self.pos_n_sv.get()
            sv_idx = self.values.index(sv_txt)
            path = self.full_paths[sv_idx]
            subprocess.Popen('explorer "{}"'.format(os.path.normpath(path)))

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
                        with open(last_tracked_i_path, 'r') as txt:
                            last_tracked_i = int(txt.read())
                if last_tracked_i_found:
                    values.append(f'{pos} (Last tracked frame: {last_tracked_i})')
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
        self.selected_pos = [self.pos_foldernames[idx]]
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
