import os
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from ast import literal_eval
from tqdm import tqdm
import numpy as np
from skimage.measure import regionprops
from skimage.transform import rotate
import matplotlib.pyplot as plt


def folder_dialog(**options):
    #Prompt the user to select the image file
    root = tk.Tk()
    root.withdraw()
    path = tk.filedialog.Directory(**options).show()
    root.destroy()
    return path


class single_entry_messagebox:
    def __init__(self, entry_label='Entry 1', input_txt='', toplevel=True):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        root.lift()
        root.attributes("-topmost", True)
        root.geometry("+800+400")
        self._root = root
        tk.Label(root, text=entry_label, font=(None, 10)).grid(row=0)
        e = tk.Entry(root, justify='center', width=40)
        e.grid(row=1, padx=16, pady=4)
        e.focus_force()
        e.insert(0, input_txt)
        tk.Button(root, command=self._quit, text='Ok!', width=10).grid(row=2,
                                                                      pady=4)
        root.bind('<Return>', self._quit)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.e = e
        root.mainloop()
    def on_closing(self):
        self._root.quit()
        self._root.destroy()
        exit('Execution aborted by the user')
    def _quit(self, event=None):
        self.entry_txt = self.e.get()
        self._root.quit()
        self._root.destroy()


class segm_metadata:
    def calc_volume(self, segm_npy_frame, zyx_vox_dim, segm_rp):
        zyx_vox_dim = np.asarray(zyx_vox_dim)
        vox_to_fl = zyx_vox_dim[1]*(zyx_vox_dim[2]**2) #revolution axis = y
        yx_pxl_to_um2 = zyx_vox_dim[1]*zyx_vox_dim[2]
        #print(zyx_vox_dim[1], vox_to_fl, yx_pxl_to_um2)
        cells_props = segm_rp
        cells_areas_pxl = []
        cells_areas_um2 = []
        cells_volumes_fl = []
        cells_volumes_vox = []
        IDs = []
        #Iterate through objects: rotate by orientation and calculate volume
        for cell in cells_props:
            # print('Not rotated cell area = {}'.format(cell.area))
            IDs.append(cell.label)
            orient_deg = cell.orientation*180/np.pi
            cell_img = (segm_npy_frame[cell.slice]==cell.label).astype(int)
            cell_aligned_long_axis = rotate(cell_img,-orient_deg,resize=True,
                                            order=3,preserve_range=True)
            # cell_aligned_area = regionprops(cell_aligned_long_axis)[0].area
            # print('Rotated cell area float = {0:.2f}'.format(np.sum(cell_aligned_long_axis)))
            # print('Rotated cell area int = {0:.2f}'.format(np.sum(np.round(cell_aligned_long_axis))))
            radii = np.sum(cell_aligned_long_axis, axis=1)/2
            cell_volume_vox = np.sum(np.pi*radii**2)
            cell_volume_fl = cell_volume_vox*vox_to_fl
            cells_areas_pxl.append(cell.area)
            cells_areas_um2.append(cell.area*yx_pxl_to_um2)
            cells_volumes_vox.append(cell_volume_vox)
            cells_volumes_fl.append(cell_volume_fl)
            # print('Cell volume = {0:.4e}'.format(cell_volume))
        self.volumes_vox = np.asarray(cells_volumes_vox)
        self.volumes_fl = np.asarray(cells_volumes_fl)
        self.areas_pxl = np.asarray(cells_areas_pxl)
        self.areas_um2 = np.asarray(cells_areas_um2)
        self.IDs = IDs


exp_path = folder_dialog(title='Select folder containing Position_n folders')
t = single_entry_messagebox(
                        entry_label='Enter xy pixel dimensions in pixel/um\n'
                                    '(e.g. [0.07, 0.07])\n\n'
                                    'NOTE: If you don\'t know it leave [1, 1].\n'
                                    'The volume in (fl) will then be equal\n'
                                    'to the volume in (voxels)\n',
                        input_txt='[1, 1]', toplevel=False).entry_txt
zyx_vox_dim = literal_eval(t)
zyx_vox_dim.insert(0, 1)


filenames = os.listdir(exp_path)
metadata_paths = [None]*len(filenames)
segm_npy_paths = [None]*len(filenames)
last_tracked_paths = [None]*len(filenames)
pbar = tqdm(desc='Loading data', total=len(filenames), unit=' pos')
for p, pos in enumerate(filenames):
    pos_path = os.path.join(exp_path, pos)
    if os.path.isdir(pos_path) and pos.find('Position_')!=-1:
        images_path = os.path.join(pos_path, 'Images')
        pos_filenames = os.listdir(images_path)
        for f in pos_filenames:
            if f.find('segm_metadata.csv') != -1:
                metadata_path = os.path.join(images_path, f)
                metadata_paths[p] = metadata_path
            elif f.find('segm.npy') != -1:
                segm_npy_path = os.path.join(images_path, f)
                segm_npy_paths[p] = segm_npy_path
            elif f.find('last_tracked_i.txt') != -1:
                last_tracked_path = os.path.join(images_path, f)
                last_tracked_paths[p] = last_tracked_path
    pbar.update(1)
pbar.close()

meta = segm_metadata()
input = zip(segm_npy_paths, metadata_paths, last_tracked_paths)
# Iterate positions
for segm_npy_path, metadata_path, last_tracked_path in input:
    if segm_npy_path is not None and metadata_path is not None:
        pos_path = os.path.dirname(os.path.dirname(segm_npy_path))
        print(f'Analysing {pos_path}...')
        segm_npy = np.load(segm_npy_path)
        with open(last_tracked_path, 'r') as txt:
            last_tracked_i = int(txt.read())
        metadata_df = pd.read_csv(metadata_path)
        try:
            # Drop previous calculations if present
            metadata_df.drop(['cell_volume_vox', 'cell_volume_fl',
                              'cell_area_pxl', 'cell_area_um2'],
                              axis=1, inplace=True)
        except:
            pass
        metadata_df.set_index(['frame_i', 'Cell_ID'], inplace=True)
        pbar = tqdm(desc='Calculating cell volume', total=last_tracked_i+1,
                    unit=' frame')
        volumes_vox_li = []
        volumes_fl_li = []
        areas_pxl_li = []
        areas_um2_li = []
        IDs = []
        frames = []
        # Iterate frames
        for frame_i in range(last_tracked_i+1):
            segm_npy_frame = segm_npy[frame_i]
            segm_rp = regionprops(segm_npy_frame)
            meta.calc_volume(segm_npy_frame, zyx_vox_dim, segm_rp)
            IDs.extend(meta.IDs)
            frames.extend([frame_i]*len(meta.IDs))
            volumes_vox_li.extend(meta.volumes_vox)
            volumes_fl_li.extend(meta.volumes_fl)
            areas_pxl_li.extend(meta.areas_pxl)
            areas_um2_li.extend(meta.areas_um2)
            pbar.update(1)
        df_other = pd.DataFrame({'cell_volume_vox': volumes_vox_li,
                                 'cell_volume_fl': volumes_fl_li,
                                 'cell_area_pxl': areas_pxl_li,
                                 'cell_area_um2': areas_um2_li,
                                 'Cell_ID': IDs,
                                 'frame_i': frames}
                                 ).set_index(['frame_i', 'Cell_ID'])
        metadata_df = metadata_df.join(df_other, how='outer')
        pbar.close()
        metadata_df.to_csv(metadata_path)
