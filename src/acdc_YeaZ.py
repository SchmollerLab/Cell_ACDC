import sys
import re
import os
import cv2
from skimage import io
from skimage.exposure import equalize_adapthist
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from time import time
from tifffile import TiffFile
from tkinter import messagebox
from myutils import download_model
from natsort import natsorted
from lib import (load_shifts, select_slice_toAlign, align_frames_3D,
               align_frames_2D, single_entry_messagebox, twobuttonsmessagebox,
               auto_select_slice, num_frames_toSegm_tk, draw_ROI_2D_frames,
               text_label_centroid, file_dialog, win_size, dark_mode,
               folder_dialog)

import apps


script_dirname = os.path.dirname(os.path.realpath(__file__))
unet_path = f'{script_dirname}/YeaZ-unet/unet/'

#append all the paths where the modules are stored. Such that this script
#looks into all of these folders when importing modules.
#sys.path.append(unet_path)
# Beno edit: this should be "src"
from YeaZ.unet import neural_network as nn
from YeaZ.unet import segment
from YeaZ.unet import tracking
download_model('YeaZ')

import prompts, load

class load_data:
    def __init__(self, path, user_ch_name):
        self.path = path
        self.parent_path = os.path.dirname(path)
        self.filename, self.ext = os.path.splitext(os.path.basename(path))
        if self.ext == '.tif' or self.ext == '.png' or self.ext == '.jpg':
            self.tif_path = path
            img_data = io.imread(path)
        elif self.ext == '.npy' or self.ext == '.npz':
            tif_path = self.substring_path(path, f'{user_ch_name}.tif',
                                           self.parent_path)[0]
            self.tif_path = tif_path
            img_data = np.load(path)
            try:
                img_data = img_data['arr_0']
            except:
                img_data = img_data
        self.img_data = img_data
        tif_filename = os.path.basename(self.tif_path)
        basename_idx = tif_filename.find(f'{user_ch_name}.tif')
        self.basename = tif_filename[0:basename_idx]
        self.info, self.metadata_found = self.metadata(self.tif_path)
        if self.metadata_found:
            try:
                self.SizeT, self.SizeZ = self.data_dimensions(self.info)
            except:
                self.SizeT, self.SizeZ = self.dimensions_entry_widget()
        elif img_data.ndim > 2:
            self.SizeT, self.SizeZ = self.dimensions_entry_widget()
        else:
            self.SizeT, self.SizeZ = 1, 1

    def build_paths(self, filename, parent_path, user_ch_name, basename=None):
        if basename is None:
            match = re.search('s(\d+)_', filename)
            if match is not None:
                basename = filename[:match.span()[1]-1]
            else:
                basename = single_entry_messagebox(
                     entry_label='Write a common basename for all output files',
                     input_txt=filename,
                     toplevel=False).entry_txt
        self.basename = basename
        base_path = f'{parent_path}/{basename}'
        self.base_path = base_path
        self.slice_used_align_path = f'{base_path}_slice_used_alignment.csv'
        self.slice_used_segm_path = f'{base_path}_slice_segm.csv'
        self.align_npz_path = f'{base_path}_{user_ch_name}_aligned.npz'
        self.align_old_path = f'{base_path}_phc_aligned.npy'
        self.align_shifts_path = f'{base_path}_align_shift.npy'
        self.segm_npy_backup_path = f'{base_path}_segm_YeaZ.npz'
        self.segm_npz_path = f'{base_path}_segm.npz'
        self.pred_npz_path = f'{base_path}_pred.npz'

    def substring_path(self, path, substring, parent_path):
        substring_found = False
        for filename in os.listdir(parent_path):
            if substring == "phase_contr.tif":
                is_match = (filename.find(substring) != -1 or
                            filename.find("phase_contrast.tif") != -1)
            else:
                is_match = filename.find(substring) != -1
            if is_match:
                substring_found = True
                break
        substring_path = f'{parent_path}/{filename}'
        return substring_path, substring_found


    def metadata(self, tif_path):
        try:
            with TiffFile(tif_path) as tif:
                self.metadata = tif.imagej_metadata
            try:
                metadata_found = True
                info = self.metadata['Info']
            except KeyError:
                metadata_found = False
                info = []
        except:
            metadata_found = False
            info = []
        return info, metadata_found

    def data_dimensions(self, info):
        SizeT = int(re.findall('SizeT = (\d+)', info)[0])
        SizeZ = int(re.findall('SizeZ = (\d+)', info)[0])
        return SizeT, SizeZ

    def dimensions_entry_widget(self):
        root = tk.Tk()
        root.geometry("+800+400")
        tk.Label(root,
                 text="Data dimensions not found in metadata.\n"
                      "Provide the following sizes.",
                 font=(None, 12)).grid(row=0, column=0, columnspan=2, pady=4)
        tk.Label(root,
                 text="Number of frames (SizeT)",
                 font=(None, 10)).grid(row=1, pady=4)
        tk.Label(root,
                 text="Number of slices (SizeZ)",
                 font=(None, 10)).grid(row=2, pady=4, padx=8)

        SizeT_entry = tk.Entry(root, justify='center')
        SizeZ_entry = tk.Entry(root, justify='center')

        # Default texts in entry text box
        SizeT_entry.insert(0, '1')
        SizeZ_entry.insert(0, '1')

        SizeT_entry.grid(row=1, column=1, padx=8)
        SizeZ_entry.grid(row=2, column=1, padx=8)

        tk.Button(root,
                  text='OK',
                  command=root.quit,
                  width=10).grid(row=3,
                                 column=0,
                                 pady=16,
                                 columnspan=2)
        SizeT_entry.focus()

        tk.mainloop()

        SizeT = int(SizeT_entry.get())
        SizeZ = int(SizeZ_entry.get())
        root.destroy()
        return SizeT, SizeZ

dark_mode()

root = tk.Tk()
root.withdraw()

selected_path = prompts.folder_dialog(title=
    'Select folder with multiple experiments, the TIFFs folder or '
    'a specific Position_n folder')

if not selected_path:
    exit('Execution aborted.')

selector = load.select_exp_folder()
(main_paths, prompts_pos_to_analyse, run_num,
is_pos_path, is_TIFFs_path) = load.get_main_paths(selected_path, 'v1')

ch_name_selector = prompts.select_channel_name()

is_pc = twobuttonsmessagebox('Img mode', 'Select imaging mode',
                             'Phase contrast', 'Bright-field').button_left

do_tracking = tk.messagebox.askyesno('Track cells?', 'Do you want to track\n'
                                     'the cells?')

save_segm = messagebox.askyesno('Save segmentation?',
                                 'Do you want to save segmentation?',
                                 master=root)

params = apps.YeaZ_Params()
if params.cancel:
    exit('Execution aborted by the user')

thresh_val = params.threshVal
min_distance = params.minDist

ch_name_not_found_msg = (
    'The script could not identify the channel name.\n\n'
    'For automatic loading the file to be segmented MUST have a name like\n'
    '"<name>_s<num>_<channel_name>.tif" e.g. "196_s16_phase_contrast.tif"\n'
    'where "196_s16" is the basename and "phase_contrast"'
    'is the channel name\n\n'
    'Please write here the channel name to be used for automatic loading'
)

all_ROIs = []
all_franges = []
all_paths = []
all_slices = []
all_basenames = []
for exp_idx, main_path in enumerate(main_paths):

    dirname = os.path.basename(main_path)
    is_TIFFs_path = any([f.find('Position_')!=-1
                         and os.path.isdir(f'{main_path}/{f}')
                         for f in os.listdir(main_path)])

    if is_TIFFs_path:
        TIFFs_path = main_path
        print('')
        print('##################################################')
        print('')
        print(f'Analysing experiment: {os.path.dirname(main_path)}')
        folders_main_path = [f for f in natsorted(os.listdir(main_path))
                               if os.path.isdir(f'{main_path}/{f}')
                               and f.find('Position_')!=-1]

        paths = []
        for i, d in enumerate(folders_main_path):
            images_path = f'{main_path}/{d}/Images'
            filenames = os.listdir(images_path)
            if ch_name_selector.is_first_call:
                ch_names, warn = ch_name_selector.get_available_channels(filenames)
                ch_name_selector.prompt(ch_names, toplevel=True)
                if warn:
                    user_ch_name = prompts.single_entry_messagebox(
                        title='Channel name not found',
                        entry_label=ch_name_not_found_msg,
                        input_txt=ch_name_selector.channel_name
                    ).entry_txt
                else:
                    user_ch_name = ch_name_selector.channel_name
            img_ch_aligned_found = False
            tif_found = False
            for j, f in enumerate(filenames):
                if f.find(f'{user_ch_name}_aligned.npy') != -1:
                    img_ch_aligned_found = True
                    aligned_i = j
                elif f.find(f'{user_ch_name}.tif') != -1:
                    tif_i = j
                    tif_found = True
            if img_ch_aligned_found:
                img_ch_path = os.path.join(images_path, filenames[aligned_i])
                paths.append(img_ch_path)
            elif tif_found:
                img_ch_path = os.path.join(images_path, filenames[tif_i])
                paths.append(img_ch_path)
            else:
                print('File not found. '
                'The script could not find a file ending with '
                f'"{user_ch_name}.tif". Skipping {d}')

        ps, pe = 0, len(paths)
        if exp_idx == 0:
            if prompts_pos_to_analyse:
                ps, pe = prompts.num_pos_toSegm_tk(len(paths),
                                                    toplevel=True).frange

    elif is_pos_path:
        TIFFs_path = os.path.dirname(main_path)
        pos_path = main_path
        images_path = f'{pos_path}/Images'
        filenames = os.listdir(images_path)
        if ch_name_selector.is_first_call:
            ch_names, warn = ch_name_selector.get_available_channels(filenames)
            ch_name_selector.prompt(ch_names, toplevel=True)
            if warn:
                user_ch_name = prompts.single_entry_messagebox(
                    title='Channel name not found',
                    entry_label=ch_name_not_found_msg,
                    input_txt='phase_contrast'
                ).entry_txt
            else:
                user_ch_name = ch_name_selector.channel_name
        img_ch_aligned_found = False
        tif_found = False
        for j, f in enumerate(filenames):
            if f.find(f'{user_ch_name}_aligned.npy') != -1:
                img_ch_aligned_found = True
                aligned_i = j
            elif f.find(f'{user_ch_name}.tif') != -1:
                tif_i = j
                tif_found = True
        if img_ch_aligned_found:
            img_ch_path = os.path.join(images_path, filenames[aligned_i])
        elif tif_found:
            img_ch_path = os.path.join(images_path, filenames[tif_i])
        else:
            raise FileNotFoundError('File not found. '
            f'The script could not find a file ending with "{user_ch_name}.tif"')
        paths = [img_ch_path]
        ps, pe = 0, len(paths)
    else:
        raise FileNotFoundError(f'The path {main_path} is not a valid path!')

    """Iterate Position folders"""
    num_pos_total = len(paths[ps:pe])
    for pos_idx, path in enumerate(paths[ps:pe]):

        print('-------------------------')
        print('')
        print('Loading data...')

        data = load_data(path, user_ch_name)
        data.build_paths(data.filename, data.parent_path, user_ch_name)

        basename = data.basename
        parent_path = data.parent_path

        print(f'Image file {path} loaded.')
        print('')

        num_slices = data.SizeZ if data.SizeZ in list(data.img_data.shape) else 1
        if num_slices > 1:
            num_frames = data.img_data.shape[0] if data.img_data.ndim > 3 else 1
        else:
            num_frames = data.img_data.shape[0] if data.img_data.ndim > 2 else 1

        print(f'Image data shape = {data.img_data.shape}')
        print(f'Number of frames = {num_frames}')
        print(f'Number of slices in each z-stack = {num_slices}')
        print('')

        """Check if the file is a split"""
        s = data.filename[:3]
        m = re.findall('(0[1-9]|1[1-9])_', s)
        concat_splits = False
        is_split = False
        split_num = 0
        if m:
            is_split = True
            split_num = m[0]
            if exp_idx == 0 and pos_idx == 0:
                concat_splits = messagebox.askyesno('Concatenate splits?',
                             'It appears you loaded a split file with '
                             f'split number {split_num}.\n'
                             'If true, do you want to concatenate the splits?',
                             master=root)

        prev_last_tracked_frame = None
        if int(split_num) > 1 and concat_splits:
            prev_split = int(split_num)-1
            prev_last_tracked_frame_name = f'{prev_split:02d}_last_tracked_frame.npy'
            prev_last_tracked_frame_path = os.path.join(
                data.parent_path, prev_last_tracked_frame_name
            )
            if os.path.exists(prev_last_tracked_frame_path):
                prev_last_tracked_frame = np.load(prev_last_tracked_frame_path)


        """Pre-align frames if needed if concat_splits"""
        if concat_splits:
            prev_split = int(split_num)-1
            prev_split_basename = f'{prev_split:02d}_{basename[3:]}'
            shifts, shifts_found = load_shifts(data.parent_path,
                                               basename=prev_split_basename)
            shifts = [shifts[-1] for _ in range(len(data.img_data))]
            if shifts_found:
                align_func = align_frames_3D if num_slices>1 else align_frames_2D
                if shifts_found:
                    aligned_frames, shifts = align_func(data.img_data,
                                                        slices=None,
                                                        register=False,
                                                        user_shifts=shifts)

        align = not img_ch_aligned_found

        """Align frames if needed"""
        if num_frames > 1 and data.ext == '.tif' and do_tracking:
            # 2D or 3D frames over time
            if align:
                frames = data.img_data
                print('Aligning frames...')
                loaded_shifts, shifts_found = load_shifts(data.parent_path,
                                                          basename=basename)
                if not shifts_found and num_slices > 1:
                    slices = select_slice_toAlign(frames, num_frames,
                                                  slice_used_for='alignment'
                                                  ).slices
                    df_slices = pd.DataFrame({'Slice used for alignment': slices,
                                              'frame_i': range(num_frames)})
                    df_slices.set_index('frame_i', inplace=True)
                    df_slices.to_csv(data.slice_used_align_path)
                else:
                    slices=None
                align_func = align_frames_3D if num_slices>1 else align_frames_2D
                if shifts_found:
                    print('NOTE: Aligning with saved shifts')
                aligned_frames, shifts = align_func(frames, slices=slices,
                                                  register=not shifts_found,
                                                  user_shifts=loaded_shifts)
                print('Frames aligned!')
                if os.path.exists(data.align_old_path):
                    os.remove(data.align_old_path)
                np.savez_compressed(data.align_npz_path, aligned_frames)
                np.save(data.align_shifts_path, shifts, allow_pickle=False)
                path = data.align_npz_path
                frames = aligned_frames
            else:
                # Aligned file found and already loaded
                frames = data.img_data
        else:
            # Simple 2D image
            frames = data.img_data
            np.savez_compressed(data.align_npz_path, frames)
            shifts = np.array([[0,0]])
            np.save(data.align_shifts_path, shifts, allow_pickle=False)

        """Check img data shape and reshape if needed"""
        print('Checking img data shape and reshaping if needed...')
        ROI_coords = None
        if num_slices == 1:
            slices = None
            if num_frames > 1:
                # 2D frames
                pass
            else:
                # 2D snapshot (no alignment required)
                y, x = frames.shape
                frames = np.reshape(frames, (1,y,x))
            ROI_coords = draw_ROI_2D_frames(
                                frames, num_frames,
                                slice_used_for='segmentation and apply ROI if needed',
                                activate_ROI=True).ROI_coords
        elif num_slices > 1:
            if os.path.exists(data.slice_used_align_path):
                df_slices = pd.read_csv(data.slice_used_align_path)
                slices = df_slices['Slice used for alignment'].to_list()
            else:
                slices = [0]
            if os.path.exists(data.slice_used_segm_path):
                df_slices = pd.read_csv(data.slice_used_segm_path)
                slices = df_slices['Slice used for segmentation'].to_list()
            else:
                slices = [0]
            if num_frames == 1:
                select_slice = auto_select_slice(frames, init_slice=0,
                            slice_used_for='segmentation and apply ROI if needed',
                            activate_ROI=True)
                ROI_coords =  select_slice.ROI_coords
                slices = [select_slice.slice]
                df_slices_path = data.slice_used_segm_path
            else:
                print('Loading slice selector GUI...')
                select_slice = select_slice_toAlign(frames, num_frames,
                            init_slice=slices[0],
                            slice_used_for='segmentation and apply ROI if needed.\n'
                                'Click "help" button for additional info '
                                'on how to select slices',
                            activate_ROI=True,
                            tk_win_title='Select slices to use for segmentation',
                            help_button=True)
                ROI_coords =  select_slice.ROI_coords
                slices = select_slice.slices
                df_slices_path = data.slice_used_segm_path
            df_slices = pd.DataFrame({'Slice used for segmentation': slices,
                                      'frame_i': range(num_frames)})
            df_slices.set_index('frame_i', inplace=True)
            df_slices.to_csv(df_slices_path)

        start = 0
        stop = num_frames
        last_segm_i = None
        if num_frames > 1 and do_tracking:
            start, stop = num_frames_toSegm_tk(num_frames,
                                               last_segm_i=last_segm_i,
                                               toplevel=True,
                                               allow_not_0_start=False
                                                           ).frange
            filenames = os.listdir(parent_path)
            for filename in filenames:
                if filename.find('_last_tracked_i.txt') != -1:
                    last_tracked_i_path = f'{parent_path}/{filename}'
                    with open(last_tracked_i_path, 'w') as txt:
                        txt.write(f'{start-1}')
                    break

        all_ROIs.append(ROI_coords)
        all_franges.append((start, stop))
        all_paths.append(path)
        all_slices.append(slices)
        all_basenames.append(data.basename)

root.destroy()

t0 = time()

inputs = zip(all_paths, all_franges, all_ROIs, all_slices, all_basenames)
for path, frange, ROI_coords, slices, basename in inputs:

    data = load_data(path, user_ch_name)
    data.build_paths(
        data.filename, data.parent_path, user_ch_name, basename=basename
    )

    basename = data.basename
    parent_path = data.parent_path

    num_slices = data.SizeZ if data.SizeZ in list(data.img_data.shape) else 1
    if num_slices > 1:
        num_frames = data.img_data.shape[0] if data.img_data.ndim > 3 else 1
    else:
        num_frames = data.img_data.shape[0] if data.img_data.ndim > 2 else 1

    frames = data.img_data

    # Index the selected frames
    if num_frames > 1:
        frames = frames[start:stop]

    if ROI_coords is not None:
        y_start, y_end, x_start, x_end = ROI_coords
        if num_slices > 1:
            ROI_img = frames[0, slices[start]][y_start:y_end, x_start:x_end]
            if prev_last_tracked_frame is not None:
                ROI_last_tracked_frame = (
                    prev_last_tracked_frame[slices[start]]
                                           [y_start:y_end, x_start:x_end]
                )
        else:
            ROI_img = frames[0][y_start:y_end, x_start:x_end]
            if prev_last_tracked_frame is not None:
                ROI_last_tracked_frame = (
                    prev_last_tracked_frame[y_start:y_end, x_start:x_end]
                )
        print(f'ROI image data shape = {ROI_img.shape}')

    print('')
    # Index the selected slices
    if num_slices > 1:
        frames = frames[range(start, stop), slices[start:stop]]

    r, c = frames.shape[-2], frames.shape[-1]
    if ROI_coords is not None:
        y_start, y_end, x_start, x_end = ROI_coords
        frames = frames[:, y_start:y_end, x_start:x_end]

    if num_frames > 1:
        frames = np.array([equalize_adapthist(f) for f in frames])
    else:
        # Single 2D image
        frames = equalize_adapthist(frames)

    path_weights = nn.determine_path_weights()
    print('Running UNet for Segmentation:')
    if num_frames > 1:
        pred_stack = nn.batch_prediction(frames, is_pc=is_pc,
                                         path_weights=path_weights,
                                         batch_size=1)
    else:
        pred_stack = nn.prediction(frames, is_pc=is_pc,
                                   path_weights=path_weights)
    print('thresholding prediction...')
    thresh_stack = nn.threshold(pred_stack, th=thresh_val)

    print('performing watershed for splitting cells...')
    if num_frames > 1:
        lab_stack = segment.segment_stack(thresh_stack, pred_stack,
                                          min_distance=min_distance
                                          ).astype(np.uint16)
    else:
        lab_stack = segment.segment(thresh_stack, pred_stack,
                                    min_distance=min_distance
                                    ).astype(np.uint16)
    lab_stack = remove_small_objects(lab_stack, min_size=5)
    if do_tracking and num_frames > 1:
        print('performing tracking by hungarian algorithm...')
        if prev_last_tracked_frame is not None:
            lab_stack = np.insert(lab_stack, 0, ROI_last_tracked_frame, axis=0)
        tracked_stack = tracking.correspondence_stack(lab_stack).astype(np.uint16)
        if prev_last_tracked_frame is not None:
            tracked_stack = tracked_stack[1:]
    else:
        tracked_stack = lab_stack

    # for simplicity, pad image back to original shape before saving
    # TODO: save only ROI and ROI borders, to save disk space
    if ROI_coords is not None:
        if num_frames > 1:
            pad_info = ((0, 0), (y_start, r - y_end), (x_start, c - x_end))
        else:
            pad_info = ((y_start, r - y_end), (x_start, c - x_end))
        tracked_stack = np.pad(tracked_stack, pad_info, mode='constant')
        frames = np.pad(frames, pad_info,  mode='constant')

    #save Segmentation results
    if save_segm:
        print('')
        print('Saving...')
        np.savez_compressed(data.segm_npz_path, tracked_stack)
        if concat_splits:
            last_tracked_frame_path = os.path.join(
                data.parent_path,
                f'{split_num}_last_tracked_frame.npz'
            )
            np.savez_compressed(last_tracked_frame_path, tracked_stack[-1])

t_end = time()

print('')
print('************************')
print('Viewing results...')

# View results
fig, ax = plt.subplots(1, 2)

def update_plots(idx):
    t0 = time()
    for a in ax:
        a.clear()
    t1 = time()
    # print(f'Clear axis execution time = {t1-t0:.4f}')
    t0 = time()
    if num_frames > 1:
        lab = tracked_stack[idx]
        img = frames[idx]
    else:
        lab = tracked_stack
        img = frames
    rp = regionprops(lab)
    IDs = [obj.label for obj in rp]

    t1 = time()
    # print(f'Regionprops and find contours = {t1-t0:.4f}')
    t0 = time()
    ax[0].imshow(img)
    ax[1].imshow(lab)
    text_label_centroid(rp, ax[0], 12, 'semibold', 'center', 'center',
                        color='r', clear=True)
    text_label_centroid(rp, ax[1], 12, 'semibold', 'center', 'center',
                        clear=True)
    for obj in rp:
        contours, hierarchy = cv2.findContours(obj.image.astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        min_y, min_x, _, _ = obj.bbox
        cont = np.squeeze(contours[0], axis=1)
        cont = np.vstack((cont, cont[0]))
        cont += [min_x, min_y]
        ax[0].plot(cont[:,0], cont[:,1], c='r')
    # print(f'Plotting contours = {t1-t0:.4f}')
    for a in ax:
        a.axis('off')
    fig.canvas.draw_idle()

idx = 0
idx_txt = fig.text(0.5, 0.15, f'Current index = {idx}/{len(tracked_stack)-1}',
                   color='w', ha='center', fontsize=14)
update_plots(idx)

if do_tracking:
    title = 'Segment&Track'
else:
    title = 'Segmentation'

fig.suptitle(f'{title} overall execution time = {t_end-t0: .3f} s', y=0.9,
             size=18)


def key_down(event):
    global idx
    if event.key == 'right' and idx < len(tracked_stack)-1:
        idx += 1
        idx_txt._text = f'Current index = {idx}/{len(tracked_stack)-1}'
        update_plots(idx)
    elif event.key == 'left' and idx > 0:
        idx -= 1
        idx_txt._text = f'Current index = {idx}/{len(tracked_stack)-1}'
        update_plots(idx)

fig.canvas.mpl_connect('key_press_event', key_down)

#win_size()
plt.show()

print('')
print('************************')
