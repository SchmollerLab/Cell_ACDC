import os
import re
import subprocess
import sys
sys.path.insert(1, "G:/My Drive/01_Postdoc_HMGU/Python_MyScripts/MIA/Nucleoids_finder/OLD")
import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
from copy import deepcopy
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import NavigationToolbar2
from pyglet.canvas import Display
from skimage import io, img_as_float
from skimage.color import label2rgb, gray2rgb
from skimage.filters import (threshold_otsu, threshold_local,
            threshold_multiotsu, gaussian, sobel, apply_hysteresis_threshold,
            try_all_threshold, threshold_minimum, threshold_yen, threshold_li,
            threshold_isodata)
from skimage.morphology import skeletonize, thin
from skimage.measure import label, regionprops
from skimage.draw import circle, line
import scipy.ndimage as nd
from tkinter import Tk, messagebox, simpledialog, Toplevel
from Yeast_ACDC_MyWidgets import Slider, Button, RadioButtons, TextBox
from Yeast_ACDC_FUNCTIONS import (separate_overlapping, text_label_centroid,
        apply_hyst_local_threshold, align_frames_3D, del_min_area_obj,
        load_shifts, cells_tracking, fig_text, sep_overlap_manual_seeds,
        merge_objs, delete_objs, select_slice_toAlign, cc_stage_df_frame0,
        find_contours, twobuttonsmessagebox, single_entry_messagebox,
        twobuttonsmessagebox, CellInt_slideshow, CellInt_slideshow_2D,
        ShowWindow_from_title, select_exp_folder, align_frames_2D, folder_dialog,
        file_dialog, tk_breakpoint)

def set_lims(ax, ax_limits):
    for a, axes in enumerate(ax):
        axes.set_xlim(*ax_limits[a][0])
        axes.set_ylim(*ax_limits[a][1])

def get_overlay(img, ol_img, ol_RGB_val=[1,1,0], ol_brightness=4, ol_alpha=0.5):
    img_rgb = gray2rgb(img_as_float(img))
    ol_img_rgb = gray2rgb(img_as_float(ol_img))*ol_RGB_val
    overlay = (img_rgb*(1.0 - ol_alpha) + ol_img_rgb*ol_alpha)*ol_brightness
    overlay = (np.clip(overlay, 0, 1)*255).astype(np.uint8)
    return overlay

def update_plots(ax, rp, img, segm_npy_frame, cca_df, vmin, vmax, frame_i, fig,
                 frame_text, frameTXT_y, num_frames, ax_limits,
                 cells_slideshow, do_overlay=False, ol_img=None,
                 ol_RGB_val=[1,1,0], ol_brightness=4, ol_alpha=0.5):
    ax[0].clear()
    text_label_centroid(rp, ax[0], 10,
                        'semibold', 'center', 'center', cca_df,
                        color='r', clear=True, apply=True,
                        display_ccStage='Only stage')
    if do_overlay:
        overlay = get_overlay(img, ol_img, ol_RGB_val=ol_RGB_val,
                              ol_brightness=ol_brightness, ol_alpha=ol_alpha)
        ax[0].imshow(overlay)
    else:
        ax[0].imshow(img)
    ax[0].axis('off')
    ax[1].clear()
    text_label_centroid(rp, ax[1], 12,
                        'semibold', 'center', 'center', cca_df,
                        apply=True, display_ccStage='IDs',
                        color='k')
    ax[1].imshow(segm_npy_frame, vmin=vmin, vmax=vmax)
    ax[1].axis('off')
    line_mother_bud(cca_df, frame_i, rp, ax[1])
    line_mother_bud(cca_df, frame_i, rp, ax[0])
    fig_text(fig, '', y=0.92, size=16, color='r')
    frame_text = fig_text(fig, 'Current frame = {}/{}'.format(frame_i,num_frames),
                          y=frameTXT_y, x=0.6, color='w', size=14, clear_all=False,
                          clear_text_ref=True, text_ref=frame_text)
    set_lims(ax, ax_limits)
    fig.canvas.draw_idle()
    try:
        cells_slideshow.sl.set_val(frame_i)
    except:
        pass
    return frame_text

def init_cc_stage_df(all_cells_ids, init_cca_df=None):
    if init_cca_df is None:
        cc_stage = ['S' for ID in all_cells_ids]
        num_cycles = [-1]*len(all_cells_ids)
        relationship = ['mother' for ID in all_cells_ids]
        related_to = [0]*len(all_cells_ids)
        discard = np.zeros(len(all_cells_ids), bool)
        df = pd.DataFrame({'Cell cycle stage': cc_stage,
                           '# of cycles': num_cycles,
                           'Relative\'s ID': all_cells_ids,
                           'Relationship': relationship,
                           'Emerg_frame_i': num_cycles,
                           'Division_frame_i': num_cycles,
                           'Discard': discard},
                            index=all_cells_ids)
        df.index.name = 'Cell_ID'
    else:
        df = init_cca_df
    return df

def nearest_point_2Dyx(points, all_others):
    """
    Given 2D array of [y, x] coordinates points and all_others return the
    [y, x] coordinates of the two points (one from points and one from all_others)
    that have the absolute minimum distance
    """
    # Compute 3D array where each ith row of each kth page is the element-wise
    # difference between kth row of points and ith row in all_others array.
    # (i.e. diff[k,i] = points[k] - all_others[i])
    diff = points[:, np.newaxis] - all_others
    # Compute 2D array of distances where
    # dist[i, j] = euclidean dist (points[i],all_others[j])
    dist = np.linalg.norm(diff, axis=2)
    # Compute i, j indexes of the absolute minimum distance
    i, j = np.unravel_index(dist.argmin(), dist.shape)
    nearest_point = all_others[j]
    point = points[i]
    return point, nearest_point

def auto_assign_bud(segm_npy_frame, new_IDs, old_IDs, cca_df, prev_cca_df, frame_i):
    # Mother and bud IDs are determined as the two objects (old and new) that
    # have the smallest euclidean distance between all pairs of new-old points
    # of the new and old objects' contours. Note that mother has to be in G1
    new_IDs_cont = find_contours(segm_npy_frame, new_IDs, group=True)[0]
    old_IDs_cont = find_contours(segm_npy_frame, old_IDs, concatenate=True)[0]
    IDs_old_cont = np.asarray([segm_npy_frame[y,x] for y,x in old_IDs_cont])
    bud_IDs_not_assigned = []
    for n, new_ID_cont in enumerate(new_IDs_cont):
        moth_ID = 0
        moth_found = False
        bud_ID = segm_npy_frame[new_ID_cont[0,0], new_ID_cont[0,1]]
        cca_df.loc[bud_ID] = ['S', 0, -1, 'bud', frame_i, -1, False]
        all_others_IDs_cont = old_IDs_cont.copy()
        IDs_all_others = IDs_old_cont.copy()
        # iterate old IDs to find the closest ID in 'G1'
        for old_ID in old_IDs:
            # remove moth_ID if it was not in G1 and repeat process
            all_others_IDs_cont = all_others_IDs_cont[IDs_all_others != moth_ID]
            IDs_all_others = IDs_all_others[IDs_all_others != moth_ID]
            new_ID_point, old_ID_point = nearest_point_2Dyx(new_ID_cont,
                                                            all_others_IDs_cont)
            moth_ID = segm_npy_frame[old_ID_point[0], old_ID_point[1]]
            moth_cc_stage = cca_df.at[moth_ID, 'Cell cycle stage']
            if moth_cc_stage == 'G1':
                moth_found = True
                moth_cc_num = prev_cca_df.at[moth_ID, '# of cycles']
                cca_df.at[bud_ID, 'Relative\'s ID'] = moth_ID
                cca_df.at[moth_ID, '# of cycles'] = moth_cc_num
                cca_df.at[moth_ID, 'Relative\'s ID'] = bud_ID
                cca_df.at[moth_ID, 'Relationship'] = 'mother'
                cca_df.at[moth_ID, 'Cell cycle stage'] = 'S'
                break
        if not moth_found:
            bud_IDs_not_assigned.append(bud_ID)
    if bud_IDs_not_assigned:
        warning_txt = ('Bud IDs {} were not assigned because there are no cells'
                      ' in G1 left'.format(bud_IDs_not_assigned))
    else:
        warning_txt = ''
    return cca_df, warning_txt

def manual_division(ID_press, cca_df, prev_cca_df, frame_i, verbose=False):
    rel_ID_press = cca_df.at[ID_press, 'Relative\'s ID']
    cc_stage_ID = cca_df.at[ID_press, 'Cell cycle stage']
    num_cycle_ID = cca_df.at[ID_press, '# of cycles']
    if cc_stage_ID == 'S' and num_cycle_ID == 0:
        prev_num_cycle_ID = prev_cca_df.at[ID_press, '# of cycles']
        prev_num_cycle_rel_ID = prev_cca_df.at[rel_ID_press, '# of cycles']
        cca_df.at[ID_press, 'Relationship'] = 'mother'
        cca_df.at[ID_press, 'Cell cycle stage'] = 'G1'
        cca_df.at[rel_ID_press, 'Cell cycle stage'] = 'G1'
        cca_df.at[rel_ID_press, 'Relationship'] = 'mother'
        cca_df.at[ID_press, '# of cycles'] = prev_num_cycle_ID+1
        cca_df.at[rel_ID_press, '# of cycles'] = prev_num_cycle_rel_ID+1
        cca_df.at[ID_press, 'Division_frame_i'] = frame_i
        cca_df.at[rel_ID_press, 'Division_frame_i'] = frame_i
    # undo manual division
    elif cc_stage_ID == 'G1' and num_cycle_ID == 1:
        prev_num_cycle_ID = cca_df.at[ID_press, '# of cycles']
        prev_num_cycle_rel_ID = cca_df.at[rel_ID_press, '# of cycles']
        cca_df.at[ID_press, 'Relationship'] = 'bud'
        cca_df.at[ID_press, 'Cell cycle stage'] = 'S'
        cca_df.at[rel_ID_press, 'Cell cycle stage'] = 'S'
        cca_df.at[ID_press, '# of cycles'] = prev_num_cycle_ID-1
        cca_df.at[rel_ID_press, '# of cycles'] = prev_num_cycle_rel_ID-1
        cca_df.at[ID_press, 'Division_frame_i'] = -1
        cca_df.at[rel_ID_press, 'Division_frame_i'] = -1
    else:
        messagebox.showwarning('Not a bud!',
        f'You clicked on cell ID {ID_press} which is not a bud.\n'
        'Make sure to click on a cell that has either \'S-0\' (to assign division) '
        'or \'G1-1\' (to undo division) writing on it.')
    if verbose:
        print(cca_df)
    return cca_df, rel_ID_press

def line_mother_bud(cca_df, frame_i, rp, ax):
    IDs = [obj.label for obj in rp]
    bud_IDs_S_frame_i = cca_df.loc[(cca_df['Cell cycle stage'] == 'S') &
                                   (cca_df['Relationship'] == 'bud')]
    for bud_ID, row in bud_IDs_S_frame_i.iterrows():
        emerg_frame_i = row['Emerg_frame_i']
        moth_ID = row['Relative\'s ID']
        if moth_ID > 0:
            if moth_ID in IDs:
                moth_y, moth_x = rp[IDs.index(moth_ID)].centroid
            else:
                print(f'WARNING: Mother cell with ID {moth_ID}'
                              f' disappeared from frame {frame_i}')
            if bud_ID in IDs:
                bud_y, bud_x = rp[IDs.index(bud_ID)].centroid
                if emerg_frame_i == frame_i:
                    ax.plot([bud_x, moth_x], [bud_y, moth_y],
                            color='r', ls='--', lw = 3, dash_capstyle='round')
                else:
                    ax.plot([bud_x, moth_x], [bud_y, moth_y],
                            color='orange', ls='--', lw = 0.7,
                            dash_capstyle='round')
            else:
                print(f'WARNING: Bud with ID {bud_ID}'
                                 f' disappeared from frame {frame_i}')

matplotlib.use("TkAgg")

matplotlib.rcParams['keymap.back'] = ['q', 'backspace', 'MouseButton.BACK']
matplotlib.rcParams['keymap.forward'] = ['v', 'MouseButton.FORWARD']
matplotlib.rcParams['keymap.quit'] = []
matplotlib.rcParams['keymap.quit_all'] = []
plt.ioff()
plt.style.use('dark_background')
plt.rc('axes', edgecolor='0.1')
axcolor = '0.1'
slider_color = '0.2'
hover_color = '0.25'
presscolor = '0.35'
button_true_color = '0.4'
do_overlay = False

bp = tk_breakpoint()

# Folder dialog
exp_path = folder_dialog(
                 title='Select experiment folder containing Position_n folders')

if os.path.basename(exp_path) == 'Images':
    images_path = exp_path
elif os.path.basename(exp_path).find('Position_') != -1:
    images_path = f'{exp_path}/Images'
else:
    select_folder = select_exp_folder()
    values = select_folder.get_values_cca(exp_path)
    pos_foldername = select_folder.run_widget(values)
    images_path = f'{exp_path}/{pos_foldername}/Images'
segm_npy_found = False
last_tracked_i_found = False
for filename in os.listdir(images_path):
    idx = filename.find('_segm.npy')
    if idx != -1:
        fd_segm_npy = f'{images_path}/{filename}'
        basename = filename[:idx]
        segm_npy_found = True
    elif filename.find('_last_tracked_i.txt') != -1:
        last_tracked_i_found = True
        last_tracked_i_path = f'{images_path}/{filename}'
        with open(last_tracked_i_path, mode='r') as txt:
            last_tracked_i = int(txt.read())
if not segm_npy_found:
    raise FileNotFoundError('Phase contrast aligned image file not found!')

#Load files
parent_path = os.path.dirname(fd_segm_npy)
segm_npy_filename = os.path.basename(fd_segm_npy)
segm_npy_filename_noEXT, segm_npy_filename_extension = os.path.splitext(fd_segm_npy)
listdir_parent = os.listdir(parent_path)
phc_found = False
cca_found = False
last_analyzed_frame_i = None
for i, filename in enumerate(listdir_parent):
    if filename.find('phc_aligned.npy')>0:
        phc_found = True
        phc_i = i
    elif filename.find('cc_stage.csv') != -1:
        cca_found = True
        cca_path = f'{parent_path}/{filename}'
        cca_df_csv = pd.read_csv(cca_path, index_col=['frame_i', 'Cell_ID'])
        buds_in_G1 = cca_df_csv[(cca_df_csv['Cell cycle stage']=='G1') &
                                (cca_df_csv['Relationship']=='bud')]
        if not buds_in_G1.empty:
            cca_df_csv.loc[buds_in_G1.index, 'Relationship'] = 'mother'
        grouped = cca_df_csv.groupby('frame_i')
        cca_df_li = [df.loc[frame_i] for frame_i, df in grouped]
        last_analyzed_frame_i = len(cca_df_li)-1
if not phc_found:
    raise FileNotFoundError('Aligned phase contrast (.._phc_aligned.npy) file not found')
phc_aligned_npy_path = parent_path + '/' + listdir_parent[phc_i]

# load image(s) into a 3D array (voxels) where pages are the single Z-stacks.
segm_npy = np.load(fd_segm_npy)
phc_aligned_npy = np.load(phc_aligned_npy_path)

if last_tracked_i_found:
    segm_npy = segm_npy[:last_tracked_i+1]
    phc_aligned_npy = phc_aligned_npy[:last_tracked_i+1]

vmin = 0
vmax = segm_npy.max()

V3D = twobuttonsmessagebox('Image shape', 'Are images 3D or 2D?', '3D', '2D').button_left
expected_shape = 4 if V3D else 3

#Initial variables
frame_text = None
num_frames = len(segm_npy)-1
if not cca_found:
    cca_df_li = [None]*(num_frames+1)
else:
    cca_df_not_analyzed_li = [None]*(num_frames-last_analyzed_frame_i)
    cca_df_li.extend(cca_df_not_analyzed_li)
modified_prev = False
viewing_prev = False
modified_IDs = []
frame_i = 0
if last_analyzed_frame_i is not None:
    frame_i = int(single_entry_messagebox(
               entry_label=f'Last analysed frame: {last_analyzed_frame_i}\n\n'
                            'Start analysing from:',
               input_txt=f'{last_analyzed_frame_i}',
               toplevel=False).entry_txt)
    viewing_prev = frame_i <= last_analyzed_frame_i

#Convert to grayscale if needed and initialize variables
if len(phc_aligned_npy.shape) == expected_shape:
    V = phc_aligned_npy[frame_i]
    segm_npy_frame = segm_npy[frame_i]
else:
    V = phc_aligned_npy
    segm_npy_frame = segm_npy[0]
try:
    V = cv2.cvtColor(V, cv2.COLOR_BGR2GRAY) #try to convert to grayscale if originally RGB (it would give an error if you try to convert a grayscale to grayscale :D)
except:
    V = V

# Read slice used for alignment
if V3D:
    listdir_parent = os.listdir(parent_path)
    listdir_bool = [f.find('slice_used_alignment')!=-1
                    for f in os.listdir(parent_path)]
    slice_used_align_found = True if any(listdir_bool) else False
    if slice_used_align_found:
        idx = [i for i, b in enumerate(listdir_bool) if b][0]
        slice_used_filename = listdir_parent[idx]
        slice_used_align_path = parent_path + '/' + slice_used_filename
        df_slices = pd.read_csv(slice_used_align_path)
        slices = df_slices['Slice used for alignment'].to_list()
    else:
        raise FileNotFoundError('Slice used for alignment file not found')

    init_slice = slices[frame_i]

#Perform z-projection if needed and calculate regionprops
rp_cells_labels_separate = regionprops(segm_npy_frame)

# Initialize cell cycle analysis DataFrame
IDs = [obj.label for obj in rp_cells_labels_separate]
if frame_i == 0:
    cca_df = init_cc_stage_df(IDs, init_cca_df=cca_df_li[0])
    if cca_df_li[0] is None:
        display_ccStage = 'IDs'
    else:
        display_ccStage = 'Only stage'
else:
    cca_df = cca_df_li[frame_i]
    display_ccStage = 'Only stage'


#Initialize plots
if V3D:
    img = V[init_slice]
else:
    img = V

#Determine initial values for z-proj sliders
if V3D:
    num_slices = V.shape[0]
else:
    num_slices = 0
    init_slice = 0

#Create image plot
cells_slideshow = None
sl_top = 0.1
sliders_left = 0.08
buttons_width = 0.1
buttons_height = 0.03
frameTXT_y = 0.15
frameTXT_x = 0.6
buttons_left = frameTXT_x-buttons_width
pltBottom = 0.2
ol_img = None
fig, ax = plt.subplots(1, 2, figsize=[13.66, 7.68])
plt.subplots_adjust(left=sliders_left, bottom=pltBottom)
ax[0].imshow(img)
text_label_centroid(rp_cells_labels_separate, ax[0], 12,
                    'semibold', 'center', 'center', cca_df,
                    color='r', clear=True, apply=True,
                    display_ccStage=display_ccStage)
ax[0].axis('off')
ax[1].imshow(segm_npy_frame, vmin=vmin, vmax=vmax)
text_label_centroid(rp_cells_labels_separate, ax[1], 12,
                    'semibold', 'center', 'center', cca_df,
                    apply=True, display_ccStage='IDs')
ax[1].axis('off')
if cca_found:
    line_mother_bud(cca_df, frame_i, rp_cells_labels_separate, ax[1])
    line_mother_bud(cca_df, frame_i, rp_cells_labels_separate, ax[0])
frame_text = fig_text(fig, 'Current frame = {}/{}'.format(frame_i,num_frames),
                      y=frameTXT_y, x=frameTXT_x, color='w', size=14,
                      clear_all=False, clear_text_ref=True, text_ref=frame_text)

#Position and color of the buttons [left, bottom, width, height]
axcolor = '0.1'
slider_color = '0.2'
hover_color = '0.25'
presscolor = '0.35'

#Buttons axis
ax_prev_button = plt.axes([0.4, 0.1, 0.1, 0.1])
ax_next_button = plt.axes([0.2, 0.1, 0.1, 0.1])
ax_save = plt.axes([0.5, 0.1, 0.1, 0.1])
ax_help = plt.axes([0.7, 0.1, 0.1, 0.1])
ax_slideshow = plt.axes([0.1, 0.1, 0.1, 0.2])
ax_overlay = plt.axes([0.1, 0.35, 0.1, 0.2])
ax_bright_sl = plt.axes([0.1, 0.48, 0.1, 0.2])
ax_alpha_sl = plt.axes([0.1, 0.69, 0.1, 0.2])
ax_rgb = plt.axes([0.1, 0.69, 0.25, 0.2])

#Create buttons
prev_button = Button(ax_prev_button, 'Prev. frame', color=axcolor,
                        hovercolor=hover_color,
                        presscolor=presscolor)
next_button = Button(ax_next_button, 'Next frame', color=axcolor,
                        hovercolor=hover_color,
                        presscolor=presscolor)
save_b = Button(ax_save, 'Save and close', color=axcolor,
                        hovercolor=hover_color,
                        presscolor=presscolor)
help_b = Button(ax_help, 'Help', color=axcolor, hovercolor=hover_color,
                        presscolor=presscolor)
slideshow_b = Button(ax_slideshow, 'Slideshow', color=axcolor,
                hovercolor=hover_color,
                presscolor=presscolor)
overlay_b = Button(ax_overlay, 'Overlay', color=axcolor,
                hovercolor=hover_color,
                presscolor=presscolor)
brightness_slider = Slider(ax_bright_sl, 'Brightness', -1, 30,
                    valinit=4,
                    valstep=1,
                    color=slider_color,
                    #init_val_line_color=hover_color,
                    valfmt='%1.0f')
alpha_slider = Slider(ax_alpha_sl, 'alpha overlay', -0.1, 1.1,
                    valinit=0.5,
                    valstep=0.01,
                    color=slider_color,
                    #init_val_line_color=hover_color,
                    valfmt='%1.2f')

def closest_value_idx(a, val):
    diff = np.abs(a-val).sum(axis=1)
    idx = diff.argmin()
    return idx


# Plot colormap for overlay RGB picker
ol_RGB_val = [1,1,0]
gradient = np.linspace(0, 1, 256)
rgb_gradient = np.vstack((gradient, gradient)).transpose()
ax_rgb.imshow(rgb_gradient, aspect='auto', cmap='hsv')
rgb_cmap_array = np.asarray([plt.cm.hsv(i) for i in gradient])
rgba = ol_RGB_val.copy()
rgba.append(1)
y_rgb = closest_value_idx(rgb_cmap_array, rgba)
x_min, x_max = ax_rgb.get_xlim()
x_rgb = (x_max-x_min)/2+x_min
picked_rgb_marker = ax_rgb.scatter(x_rgb, y_rgb, marker='s', color='k')
ax_rgb.axis('off')


#Create event for next and previous frame
def next_frame(event):
    global frame_i, frame_text, rp_cells_labels_separate,\
           segm_npy_frame, cca_df, cca_df_li, viewing_prev, modified_prev,\
           img, modified_IDs, ol_img
    if frame_i < num_frames:
        cancel = False
        if frame_i == 0:
            IDs = [obj.label for obj in rp_cells_labels_separate]
            init_cca_df_frame0 = cc_stage_df_frame0(IDs, cca_df)
            cca_df = init_cca_df_frame0.df
            cancel = init_cca_df_frame0.cancel
            # Check if we modified previously initialized cca df for frame 0
            if cca_df_li[frame_i] is not None:
                if not cca_df.equals(cca_df_li[frame_i]):
                    modified_prev = True
        if not cancel:
            cca_df_li[frame_i] = cca_df.copy()
            prev_stored_cca_df = cca_df_li[frame_i].copy()
            frame_i += 1
            if not isinstance(cca_df_li[frame_i], pd.DataFrame):
                # Current frame was not analysed
                viewing_prev = False
                modified_prev = False
                modified_IDs = []
            elif modified_prev:
                cca_df = cca_df_li[frame_i].copy()
            if V3D:
                img = phc_aligned_npy[frame_i, slices[frame_i]]
                if do_overlay:
                    ol_img = ol_frames[frame_i, slices[frame_i]]
            else:
                img = phc_aligned_npy[frame_i]
                if do_overlay:
                    ol_img = ol_frames[frame_i]
            segm_npy_frame = segm_npy[frame_i]
            rp_cells_labels_separate = regionprops(segm_npy_frame)
            # Compute new cell cycle analysis table (new_cca_df)
            prev_rp = regionprops(segm_npy[frame_i-1])
            prev_IDs = [obj.label for obj in prev_rp]
            current_IDs = [obj.label for obj in rp_cells_labels_separate]
            new_IDs = [ID for ID in current_IDs if ID not in prev_IDs]
            warn = ''
            if new_IDs:
                new_cca_df, warn = auto_assign_bud(segm_npy_frame, new_IDs,
                                                   prev_IDs, cca_df,
                                                   cca_df_li[frame_i-1],
                                                   frame_i)
            else:
                new_cca_df = cca_df.copy()
            # Drop cells that disappeared in current frame
            new_cca_df = new_cca_df.filter(items=current_IDs, axis=0)
            # Check if we are viewing a previously analysed frame or a new frame
            # and if we modified a previously analysed frame
            if not viewing_prev:
                # we are on a frame that was never analysed
                cca_df = new_cca_df
            # elif modified_prev:
            #     # we modified a previously analysed frame
            #     # reinsert what was manually modified
            #     new_idx = ~new_cca_df.index.isin(modified_IDs)
            #     prev_idx = ~cca_df.index.isin(modified_IDs)
            #     new_cca_df.loc[new_idx] = cca_df.loc[prev_idx]
            #     cca_df = new_cca_df
            #     modified_prev = False
            else:
                # we are on a frame that was already analysed and
                # we didn't modify it
                cca_df = cca_df_li[frame_i]
            frame_text = update_plots(ax, rp_cells_labels_separate, img,
                                      segm_npy_frame, cca_df, vmin, vmax, frame_i,
                                      fig, frame_text, frameTXT_y, num_frames,
                                      ax_limits, cells_slideshow, ol_img=ol_img,
                                      do_overlay=do_overlay,
                                      ol_RGB_val=ol_RGB_val,
                                      ol_brightness=brightness_slider.val,
                                      ol_alpha=alpha_slider.val)
    elif frame_i+1 > num_frames:
        cca_df_li[frame_i] = cca_df.copy()
        print('You reached the last frame')
    connect_axes_cb(ax)


def prev_frame(event):
    global frame_i, frame_text, rp_cells_labels_separate,\
           segm_npy_frame, cca_df, viewing_prev, img, cca_df_li, ol_img
    if frame_i == num_frames:
        cca_df_li[frame_i] = cca_df.copy()
    if frame_i-1 >= 0:
        viewing_prev = True
        frame_i -= 1
        cca_df = cca_df_li[frame_i].copy()
        if V3D:
            img = phc_aligned_npy[frame_i, slices[frame_i]]
            if do_overlay:
                ol_img = ol_frames[frame_i, slices[frame_i]]
        else:
            img = phc_aligned_npy[frame_i]
            if do_overlay:
                ol_img = ol_frames[frame_i]
        segm_npy_frame = segm_npy[frame_i]
        rp_cells_labels_separate = regionprops(segm_npy_frame)
        frame_text = update_plots(ax, rp_cells_labels_separate, img,
                                  segm_npy_frame, cca_df, vmin, vmax, frame_i,
                                  fig, frame_text, frameTXT_y, num_frames,
                                  ax_limits, cells_slideshow, ol_img=ol_img,
                                  do_overlay=do_overlay,
                                  ol_RGB_val=ol_RGB_val,
                                  ol_brightness=brightness_slider.val,
                                  ol_alpha=alpha_slider.val)
    else:
        print('You reached the first frame')
    connect_axes_cb(ax)

def handle_close(event):
    global fd_segm_npy
    save = messagebox.askyesno('Save', 'Do you want to save cell cycle analysis DataFrame?')
    if save:
        save_csv(None)
    if segm_npy_modified:
        save_npy = messagebox.askyesno('Save', 'Do you want to save segmentation file?')
        if save_npy:
            save_newname = twobuttonsmessagebox('Save or replace?',
                                    'Do you want to save segmentation files\n'
                                    'with new name or replace current file?',
                                    'Save with new name', 'Replace').button_left
            if save_newname:
                new_filename = single_entry_messagebox(
                                               entry_label='Write new filename',
                                               input_txt=filename).entry_txt
                fd_segm_npy = '{}/{}'.format(parent_path, new_filename)
            np.save(fd_segm_npy, segm_npy, allow_pickle=False)
    try:
        cells_slideshow.close()
    except:
        pass

def help_cb(event):
    root_info = Toplevel()
    root_info.withdraw()  # hide parent window
    messagebox.showinfo('Cell cycle analysis Help',
        '1. RIGHT click on a bud that shows cell division in the current frame '
        'will assign G1 to both mother and bud.\n'
        'NOTE: if you want to undo the assignment of division simply click again '
        'with the RIGHT button on the same bud\n\n'
        '2. If a bud was assigned incorrectly press \'m\' key and, while keeping'
        ' \'m\' key down, right click on the bud and release on the correct mother\n\n'
        '3. SCROLL WHEEL click on a cell on segmented image to delete from all future frames\n\n'
        'NOTE: If you want to view the cell cycle stage table press \'ctrl+p\'',
        master=root_info)

def slideshow_cb(event):
    global cells_slideshow
    rps = [regionprops(segm_npy_frame) for segm_npy_frame in segm_npy]
    if not ShowWindow_from_title('Cell intensity image slideshow').window_found:
        if V3D:
            cells_slideshow = CellInt_slideshow(phc_aligned_npy,
                                                init_slice, num_frames+1,
                                                frame_i, cca_df_li, rps, False)
            cells_slideshow.run()
        else:
            cells_slideshow = CellInt_slideshow_2D(phc_aligned_npy,
                                                   num_frames+1, frame_i,
                                                   cca_df_li, rps, False)
            cells_slideshow.run()


def save_csv(event):
    print('Saving cell cycle analysis data...')
    if frame_i == num_frames:
        cca_df_li[frame_i] = cca_df.copy()
    keys = [i for i, df in enumerate(cca_df_li) if isinstance(df, pd.DataFrame)]
    df_li = [df for df in cca_df_li if isinstance(df, pd.DataFrame)]
    df = pd.concat(df_li, keys=keys, names=['frame_i', 'Cell_ID'])
    cc_stage_path = f'{parent_path}/{basename}_cc_stage.csv'
    df.to_csv(cc_stage_path)
    fig.canvas.mpl_disconnect(cid_close)
    plt.close()
    print('Saved!')

def overlay_cb(event):
    global ol_frames, do_overlay
    if do_overlay:
        overlay_b.color = axcolor
        overlay_b.hovercolor = hover_color
        overlay_b.label._text = 'Overlay'
        overlay_b.ax.set_facecolor(axcolor)
        fig.canvas.draw_idle()
        do_overlay = False
    else:
        ol_path = file_dialog(title='Select image file to overlay',
                              initialdir=parent_path)
        if ol_path != '':
            do_overlay = True
            overlay_b.color = button_true_color
            overlay_b.hovercolor = button_true_color
            overlay_b.label._text = 'Overlay ON'
            overlay_b.ax.set_facecolor(button_true_color)
            fig.canvas.draw_idle()
            # Load overlay frames and align if needed
            filename = os.path.basename(ol_path)
            filename_noEXT, ext = os.path.splitext(filename)
            print('Loading overlay file...')
            if ext == '.npy':
                ol_frames = np.load(ol_path)
                if filename.find('aligned') != -1:
                    align_ol = False
                else:
                    align_ol = True
            elif ext == '.tif' or ext == '.tiff':
                align_ol = True
                ol_frames = io.imread(ol_path)
            else:
                messagebox.showerror('File Format not supported!',
                    f'File format {ext} is not supported!\n'
                    'Choose either .tif/.tiff or .npy files.')
            if align_ol:
                loaded_shifts, shifts_found = load_shifts(images_path)
                if shifts_found:
                    print('Aligning overlay image frames...')
                    align_func = align_frames_3D if V3D else align_frames_2D
                    aligned_frames, shifts = align_func(ol_frames, slices=None,
                                                      register=False,
                                                      user_shifts=loaded_shifts)
                    aligned_filename = f'{filename_noEXT}_aligned.npy'
                    aligned_path = f'{images_path}/{aligned_filename}'
                    np.save(aligned_path, aligned_frames, allow_pickle=False)
                    print('Overlay image frames aligned!')
                    ol_frames = aligned_frames
                else:
                    messagebox.showerror('Shifts file not found!',
                        f'\"..._align_shift.npy\" file not found!\n'
                        'Overlay images cannot be aligned to the cells image.')
                    raise FileNotFoundError('Shifts file not found!')
            if V3D:
                ol_img = ol_frames[frame_i, slices[frame_i]]
            else:
                ol_img = ol_frames[frame_i]
            ax[0].clear()
            text_label_centroid(rp_cells_labels_separate, ax[0], 10,
                                'semibold', 'center', 'center', cca_df,
                                color='r', clear=True, apply=True,
                                display_ccStage=display_ccStage)
            overlay = get_overlay(img, ol_img, ol_RGB_val=ol_RGB_val,
                                  ol_brightness=brightness_slider.val,
                                  ol_alpha=alpha_slider.val)
            ax[0].imshow(overlay)
            ax[0].axis('off')
            set_lims(ax, ax_limits)
            fig.canvas.draw_idle()
            connect_axes_cb(ax)

def update_overlay_cb(event):
    if do_overlay:
        if V3D:
            ol_img = ol_frames[frame_i, slices[frame_i]]
        else:
            ol_img = ol_frames[frame_i]
        ax[0].clear()
        text_label_centroid(rp_cells_labels_separate, ax[0], 10,
                            'semibold', 'center', 'center', cca_df,
                            color='r', clear=True, apply=True,
                            display_ccStage=display_ccStage)
        overlay = get_overlay(img, ol_img, ol_RGB_val=ol_RGB_val,
                              ol_brightness=brightness_slider.val,
                              ol_alpha=alpha_slider.val)
        ax[0].imshow(overlay)
        ax[0].axis('off')
        set_lims(ax, ax_limits)
        fig.canvas.draw_idle()
    else:
        messagebox.showwarning('Overlay not active', 'Brightness slider, '
            'alpha slider and the vertical color picker all control the '
            'overlay appearance.\n To use them you first need to press on the'
            '"Overlay" button and choose an image to overlay '
            '(typically a fluorescent signal)')

def rgb_cmap_cb(event):
    global ol_RGB_val
    update_overlay_cb(event)


next_button.on_clicked(next_frame)
prev_button.on_clicked(prev_frame)
save_b.on_clicked(save_csv)
help_b.on_clicked(help_cb)
slideshow_b.on_clicked(slideshow_cb)
overlay_b.on_clicked(overlay_cb)
brightness_slider.on_changed(update_overlay_cb)
alpha_slider.on_changed(update_overlay_cb)

assign_bud = False
def key_pressed(event):
    global assign_bud
    if event.key == 'right':
        next_frame(None)
    elif event.key == 'left':
        prev_frame(None)
    elif event.key == 'ctrl+p':
        print(cca_df)
        print('Modified IDs: {}'.format(modified_IDs))
        print(ax_limits)
        print(home_ax_limits)
    elif event.key == 'm':
        assign_bud = True
    else:
        pass

segm_npy_modified = False
dropped_IDs = []
def mouse_down(event):
    global cca_df, modified_prev, yd, xd, frame_text, modified_IDs,\
           segm_npy_frame, segm_npy, segm_npy_modified,\
           rp_cells_labels_separate, cca_df_li, picked_rgb_marker,\
           ol_RGB_val
    right_click = event.button == 3
    scroll_click = event.button == 2
    left_click = event.button == 1
    # Annotate division
    if right_click and event.inaxes == ax[0] and not assign_bud:
        modified_prev = True if viewing_prev else False
        y = int(event.ydata)
        x = int(event.xdata)
        cell_ID = segm_npy_frame[y,x]
        prev_cca_df = cca_df_li[frame_i-1]
        cca_df, rel_ID = manual_division(cell_ID, cca_df, prev_cca_df, frame_i,
                                         verbose=True)
        modified_IDs.extend([cell_ID, rel_ID])
        frame_text = update_plots(ax, rp_cells_labels_separate, img,
                                  segm_npy_frame, cca_df, vmin, vmax, frame_i,
                                  fig, frame_text, frameTXT_y, num_frames,
                                  ax_limits, cells_slideshow, ol_img=ol_img,
                                  do_overlay=do_overlay,
                                  ol_RGB_val=ol_RGB_val,
                                  ol_brightness=brightness_slider.val,
                                  ol_alpha=alpha_slider.val)
        cca_df_li[frame_i] = cca_df.copy()
        cc_stage = cca_df.at[cell_ID, 'Cell cycle stage']
        i = frame_i+1
        if i < num_frames:
            cc_stage_i = cca_df_li[i].at[cell_ID, 'Cell cycle stage']
            # Correct all future frames cell cycle analysis table if present
            while cc_stage_i != cc_stage:
                if i >= num_frames:
                    break
                elif cca_df_li[i] is not None:
                    manual_division(cell_ID, cca_df_li[i], prev_cca_df, i)
                    i += 1
                    cc_stage_i = cca_df_li[i].at[cell_ID, 'Cell cycle stage']
                else:
                    break

    # Correct bud assignment
    if right_click and event.inaxes == ax[0] and assign_bud:
        modified_prev = True if viewing_prev else False
        yd = int(event.ydata)
        xd = int(event.xdata)
    # Delete cell from all future frames
    if scroll_click and event.inaxes == ax[1]:
        segm_npy_modified = True
        y = int(event.ydata)
        x = int(event.xdata)
        cell_ID = segm_npy_frame[y,x]
        frames_mask = np.zeros(segm_npy.shape, bool)
        frames_mask[frame_i:] = True
        drop_ID_mask = np.logical_and(segm_npy == cell_ID, frames_mask)
        segm_npy[drop_ID_mask] = 0
        # If the cell to be deleted is in 'S' phase we need to delete the
        # relative's ID (mother or bud) as well.
        if cca_df.at[cell_ID, 'Cell cycle stage'] == 'S':
            rel_ID = cca_df.at[cell_ID, 'Relative\'s ID']
            drop_ID_mask = np.logical_and(segm_npy == rel_ID, frames_mask)
            segm_npy[drop_ID_mask] = 0
            cca_df.drop(rel_ID, inplace=True)
        segm_npy_frame = segm_npy[frame_i]
        cca_df.drop(cell_ID, inplace=True)
        rp_cells_labels_separate = regionprops(segm_npy_frame)
        frame_text = update_plots(ax, rp_cells_labels_separate, img,
                                  segm_npy_frame, cca_df, vmin, vmax, frame_i,
                                  fig, frame_text, frameTXT_y, num_frames,
                                  ax_limits, cells_slideshow, ol_img=ol_img,
                                  do_overlay=do_overlay,
                                  ol_RGB_val=ol_RGB_val,
                                  ol_brightness=brightness_slider.val,
                                  ol_alpha=alpha_slider.val)
        cca_df_li[frame_i] = cca_df.copy()
    if left_click and event.inaxes == ax_rgb:
        picked_rgb_marker.remove()
        y_rgb = int(round(event.ydata))
        ol_RGB_val = rgb_cmap_array[y_rgb][:3]
        picked_rgb_marker = ax_rgb.scatter(x_rgb, y_rgb, marker='s', color='k')
        update_overlay_cb(event)

def mouse_up(event):
    global cca_df_li, frame_text, modified_IDs, cca_df
    right_click = event.button == 3
    if right_click and event.inaxes == ax[0] and assign_bud:
        # Correct bud assigment
        yu = int(event.ydata)
        xu = int(event.xdata)
        bud_ID = segm_npy_frame[yd,xd]
        moth_ID = segm_npy_frame[yu,xu]
        if bud_ID == 0 or moth_ID == 0:
            action = 'Clicked' if bud_ID==0 else 'Released'
            messagebox.showerror(f'{action} on the background!',
                f'You {action} on the background!')
        bud_relationship = cca_df.at[bud_ID, 'Relationship']
        moth_relationship = cca_df.at[moth_ID, 'Relationship']
        moth_ccs = cca_df.at[moth_ID, 'Cell cycle stage']
        valid_selection = True
        if bud_relationship != 'bud':
            messagebox.showerror('Not a bud!', 'You clicked on a cell '
                'that is not a bud!\n To correct bud assignment you have to '
                'click on the bud and release on the correct mother!\n'
                'If the operation was successful the orange/red line should '
                'move and connect the bud the newly assigned mother.')
            valid_selection = False
        if moth_relationship != 'mother':
            messagebox.showerror('Not a mother!', 'You released on a cell '
                'that is not a mother!\n To correct bud assignment you have to '
                'click on the bud and release on a mother cell!\n'
                'If the operation was successful the orange/red line should '
                'move and connect the bud the newly assigned mother.')
            valid_selection = False
        if moth_ccs != 'G1':
            messagebox.showwarning('Mother not in G1!', 'You released '
                'on a cell that is not in G1!\n The bud can only be assigned '
                'to a cell in G1.\n\n If you still believe that '
                f'cell ID {moth_ID} is the correct mother remember to also '
                'assign its current bud to the right mother.\n'
                'You will have to do the operation three times.')
        if valid_selection:
            emerg_frame_i = cca_df.at[bud_ID, 'Emerg_frame_i']
            df_before_emerg = cca_df_li[emerg_frame_i-1]
            wrong_mothID = cca_df.at[bud_ID, 'Relative\'s ID']
            modified_IDs.extend([bud_ID, moth_ID])
            if wrong_mothID != -1:
                cca_df.loc[wrong_mothID] = df_before_emerg.loc[wrong_mothID]
            else:
                cca_df.at[moth_ID, '# of cycles'] = 1
            cca_df.at[bud_ID, 'Relative\'s ID'] = moth_ID
            cca_df.at[moth_ID, 'Relative\'s ID'] = bud_ID
            cca_df.at[moth_ID, 'Cell cycle stage'] = 'S'
            cca_df.at[moth_ID, 'Relationship'] = 'mother'
            # Correct all previous frames where the bud was assigned to
            # the wrong mother
            if emerg_frame_i != frame_i:
                i = frame_i
                while emerg_frame_i != i:
                    i -=1
                    df = cca_df_li[i]
                    df.loc[wrong_mothID] = df_before_emerg.loc[wrong_mothID]
                    df.at[bud_ID, 'Relative\'s ID'] = moth_ID
                    df.at[moth_ID, 'Relative\'s ID'] = bud_ID
                    df.at[moth_ID, 'Cell cycle stage'] = 'S'
                    df.at[moth_ID, 'Relationship'] = 'mother'
                    cca_df_li[i] = df.copy()
            # Correct all future frames cell cycle analysis table if present
            i = frame_i+1
            if i < num_frames:
                relationship_i = cca_df_li[i].at[bud_ID, 'Relationship']
                print(relationship_i)
                while relationship_i == 'bud':
                    if i >= num_frames:
                        break
                    elif cca_df_li[i] is not None:
                        cca_df_li[i].loc[wrong_mothID] = df_before_emerg.loc[
                                                                   wrong_mothID]
                        cca_df_li[i].at[bud_ID, 'Relative\'s ID'] = moth_ID
                        cca_df_li[i].at[moth_ID, 'Relative\'s ID'] = bud_ID
                        cca_df_li[i].at[moth_ID, 'Cell cycle stage'] = 'S'
                        cca_df_li[i].at[moth_ID, 'Relationship'] = 'mother'
                        i += 1
                        relationship_i = cca_df_li[i].at[bud_ID, 'Relationship']
                    else:
                        break
            frame_text = update_plots(ax, rp_cells_labels_separate, img,
                                      segm_npy_frame, cca_df, vmin, vmax, frame_i,
                                      fig, frame_text, frameTXT_y, num_frames,
                                      ax_limits, cells_slideshow, ol_img=ol_img,
                                      do_overlay=do_overlay,
                                      ol_RGB_val=ol_RGB_val,
                                      ol_brightness=brightness_slider.val,
                                      ol_alpha=alpha_slider.val)
            cca_df_li[frame_i] = cca_df.copy()

def key_up(event):
    global assign_bud
    if event.key == 'm':
        assign_bud = False

def resize_widgets(event):
    """plt.show() triggers a resize event that changes position of images axes
    --> move and resize widgets according to the new images axes position as
    [left, bottom, width, height]"""
    ax1_left, ax1_bottom, ax1_right, ax1_top = ax[1].get_position().get_points().flatten()
    ax0_l, ax0_b, ax0_r, ax0_t = ax[0].get_position().get_points().flatten()
    ax_prev_button.set_position([buttons_left,  sl_top,
                                 buttons_width, buttons_height])
    ax_next_button.set_position([buttons_left+buttons_width,  sl_top,
                                 buttons_width, buttons_height])
    ax_save.set_position([ax1_right-buttons_width, sl_top,
                          buttons_width, buttons_height])
    ax_save.set_position([ax1_right-buttons_width, ax1_bottom-buttons_height-0.01,
                          buttons_width, buttons_height])
    ax_help.set_position([ax0_l, ax0_t+0.01, buttons_width, buttons_height])
    ax_slideshow.set_position([ax0_l+buttons_width+0.005, ax0_t+0.01,
                               buttons_width, buttons_height])
    ax_overlay.set_position([ax0_l+2*(buttons_width+0.005), ax0_t+0.01,
                             buttons_width, buttons_height])
    sl_h = 0.025
    ax_bright_sl.set_position([ax0_l, ax0_b-0.007-sl_h, ax0_r-ax0_l, sl_h])
    ax_alpha_sl.set_position([ax0_l, ax0_b-2*(0.007+sl_h), ax0_r-ax0_l, sl_h])
    ax_rgb.set_position([ax0_r+0.005, ax0_b, sl_h*2/3, ax0_t-ax0_b])

y, x = img.shape
ax_limits = [[(-0.5, x-0.5), (y-0.5, -0.5)] for _ in range(2)]
home_ax_limits = [[(-0.5, x-0.5), (y-0.5, -0.5)] for _ in range(2)]

def on_xlim_changed(axes):
    global ax_limits
    ax_idx = list(ax).index(axes)
    xlim = axes.get_xlim()
    ax_limits[ax_idx][0] = xlim

def on_ylim_changed(axes):
    global ax_limits
    ax_idx = list(ax).index(axes)
    ylim = axes.get_ylim()
    ax_limits[ax_idx][1] = ylim

def connect_axes_cb(ax):
    for i, axes in enumerate(ax):
        axes.callbacks.connect('xlim_changed', on_xlim_changed)
        axes.callbacks.connect('ylim_changed', on_ylim_changed)

home = NavigationToolbar2.home
def new_home(self, *args, **kwargs):
    global ax_limits
    try:
        ax_limits = deepcopy(home_ax_limits)
    except:
        traceback.print_exc()
        pass
    home(self, *args, **kwargs)
NavigationToolbar2.home = new_home

release_zoom = NavigationToolbar2.release_zoom
def my_release_zoom(self, event):
    release_zoom(self, event)
    # Disconnect zoom to rect after having used it once
    self.zoom()
    self.push_current()
    # self.release(event)
NavigationToolbar2.release_zoom = my_release_zoom

#Canvas events
(fig.canvas).mpl_connect('key_press_event', key_pressed)
(fig.canvas).mpl_connect('key_release_event', key_up)
(fig.canvas).mpl_connect('button_press_event', mouse_down)
(fig.canvas).mpl_connect('button_release_event', mouse_up)
(fig.canvas).mpl_connect('resize_event', resize_widgets)
cid_close = (fig.canvas).mpl_connect('close_event', handle_close)
connect_axes_cb(ax)


#Display plots maximized window
mng = plt.get_current_fig_manager()
screens = Display().get_screens()
num_screens = len(screens)
if num_screens==1:
    mng.window.state('zoomed') #display plots window maximized
else:
    width = screens[0].width
    height = screens[0].height - 70
    left = width-7
    geom = "{}x{}+{}+0".format(width,height,left)
    #mng.window.wm_geometry(geom) #move GUI window to second monitor
                                 #with string "widthxheight+x+y"

pos_foldername = os.path.basename(os.path.dirname(parent_path))
fig.canvas.set_window_title('Cell cycle analysis - {}'.format(pos_foldername))
plt.show()
