import os
import subprocess
import numpy as np
import pandas as pd
import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
from tkinter import filedialog as fd
from tkinter import Tk, messagebox, simpledialog
from Yeast_ACDC_MyWidgets import Slider, Button, RadioButtons, MyRadioButtons
from Yeast_ACDC_FUNCTIONS import (separate_overlapping, text_label_centroid,
        apply_hyst_local_threshold, align_frames, del_min_area_obj,
        load_shifts, cells_tracking, fig_text, sep_overlap_manual_seeds,
        merge_objs, delete_objs, select_slice_toAlign, z_proj_max,
        twobuttonsmessagebox, select_exp_folder, folder_dialog, file_dialog)

def line_mother_bud(cca_df, frame_i, rp, ax):
    IDs = [obj.label for obj in rp]
    bud_IDs_S_frame_i = cca_df.loc[(cca_df['Cell cycle stage'] == 'S') &
                                   (cca_df['Relationship'] == 'bud')]
    for bud_ID, row in bud_IDs_S_frame_i.iterrows():
        emerg_frame_i = row['Emerg_frame_i']
        moth_ID = row['Relative\'s ID']
        if moth_ID > 0:
            moth_y, moth_x = rp[IDs.index(moth_ID)].centroid
            bud_y, bud_x = rp[IDs.index(bud_ID)].centroid
            if emerg_frame_i == frame_i:
                ax.plot([bud_x, moth_x], [bud_y, moth_y],
                        color='r', ls=':', lw = 2, dash_capstyle='round')
            else:
                ax.plot([bud_x, moth_x], [bud_y, moth_y],
                        color='orange', ls=':', lw = 0.8, dash_capstyle='round')

def build_cmap(under_vmin_c='0.1', max_ID=100):
    n = max_ID if max_ID<256 else 256
    vals = np.linspace(0,1,n)
    np.random.shuffle(vals)
    my_cmap = plt.cm.colors.ListedColormap(plt.cm.viridis(vals))
    my_cmap.set_under(under_vmin_c)
    return my_cmap

def get_overlay(img, ol_img, ol_RGB_val=[1,1,0], ol_brightness=4, ol_alpha=0.5):
    img_rgb = gray2rgb(img_as_float(img))
    ol_img_rgb = gray2rgb(img_as_float(ol_img))*ol_RGB_val
    overlay = (img_rgb*(1.0 - ol_alpha) + ol_img_rgb*ol_alpha)*ol_brightness
    return overlay

def update_plots(ax, rp, img, segm_npy_frame, cca_df, vmin, vmax, frame_i, fig,
                 frame_text, frameTXT_y, num_frames, display_ccStage,
                 cmap=None, do_overlay=False, ol_img=None,
                 ol_RGB_val=[1,1,0], ol_brightness=4, ol_alpha=0.5):
    ax[0].clear()
    text_label_centroid(rp, ax[0], 12,
                        'semibold', 'center', 'center', cca_df,
                        color='r', clear=True, apply=True,
                        display_ccStage=display_ccStage)
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
                        apply=True, display_ccStage='IDs')
    ax[1].imshow(segm_npy_frame, vmin=1, vmax=vmax, cmap=cmap)
    ax[1].axis('off')
    if cca_df is not None:
        try:
            line_mother_bud(cca_df, frame_i, rp, ax[1])
            line_mother_bud(cca_df, frame_i, rp, ax[0])
            ax[1].set_title('')
        except:
            ax[1].set_title('Cell cycle not analyzed for this frame')
    fig_text(fig, '', y=0.92, size=16, color='r')
    frame_text = fig_text(fig, 'Current frame = {}/{}'.format(frame_i,num_frames),
                          y=frameTXT_y, x=0.6, color='w', size=14,
                          clear_all=False, clear_text_ref=True, text_ref=frame_text)
    fig.canvas.draw_idle()
    return frame_text

plt.ioff()
plt.style.use('dark_background')
plt.rc('axes', edgecolor='0.1')

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
phc_aligned_found = False
ccs_df_found = False
cca_analysed_frames = []
for filename in os.listdir(parent_path):
    if filename.find('phc_aligned.npy')!=-1:
        phc_aligned_found = True
        phc_aligned_npy_path = parent_path + '/' + filename
    if filename.find('_cc_stage.csv')!=-1:
        ccs_df_found = True
        cc_stage_dfs = pd.read_csv('{}/{}'.format(parent_path, filename),
                                   index_col=['frame_i', 'Cell_ID'])
        cc_stage_dfs.sort_index(inplace=True)
        cca_analysed_frames = cc_stage_dfs.index.get_level_values(0)
phc_found = False
if not phc_aligned_found:
    for filename in os.listdir(parent_path):
        if filename.find('phase_contr.tif')>0:
            phc_found = True
            phc_aligned_npy_path = parent_path + '/' + filename
            break
if not phc_aligned_found and not phc_found:
    print('FileNotFoundWarn: Phase contrast file not found!')


# load image(s) into a 3D array (voxels) where pages are the single Z-stacks.
segm_npy = np.load(fd_segm_npy)
if phc_aligned_found:
    phc_aligned_npy = np.load(phc_aligned_npy_path)
elif phc_found:
    phc_aligned_npy = io.imread(phc_aligned_npy_path).astype(np.uint8)
else:
    phc_aligned_npy = np.zeros_like(segm_npy)

vmin = 1
vmax = segm_npy.max()

V3D = twobuttonsmessagebox('Image shape', 'Are images 3D or 2D?', '3D', '2D').button_left
expected_shape = 4 if V3D else 3

#Convert to grayscale if needed and initialize variables
frame_i = 0 #load first frame
display_ccStage = 'Display Cells\' IDs'
if ccs_df_found:
    if frame_i in cca_analysed_frames:
        cca_df = cc_stage_dfs.loc[frame_i]
    else:
        cca_df = []
if len(phc_aligned_npy.shape) == expected_shape:
    V = phc_aligned_npy[frame_i]
    segm_npy_frame = segm_npy[frame_i]
else:
    V = phc_aligned_npy
    segm_npy_frame = segm_npy[0]
try:
    V = cv2.cvtColor(V, cv2.COLOR_BGR2GRAY) # try to convert to grayscale if originally RGB (it would give an error if you try to convert a grayscale to grayscale :D)
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
        slices = select_slice_toAlign(V).slices

    init_slice = slices[0]

#Perform z-projection if needed and calculate regionprops
rp_cells_labels_separate = regionprops(segm_npy_frame)

#Initialize plots
if V3D:
    img = V[init_slice]
else:
    img = V

#Initial variables
frame_text = None
num_frames = len(segm_npy)-1
ol_img = None

#Determine initial values for z-proj sliders
if V3D:
    num_slices = V.shape[0]
else:
    num_slices = 0
    init_slice = 0

#Create image plot
sl_top = 0.1
sliders_left = 0.08
buttons_width = 0.1
buttons_height = 0.03
frameTXT_y = 0.15
frameTXT_x = 0.6
buttons_left =frameTXT_x-buttons_width
my_cmap = build_cmap(max_ID=segm_npy.max())
fig, ax = plt.subplots(1, 2)
plt.subplots_adjust(left=sliders_left, bottom=0.25)
ax[0].imshow(img)
text_label_centroid(rp_cells_labels_separate, ax[0], 12,
                    'semibold', 'center', 'center', color='r',
                    clear=True, apply=True)
ax[0].axis('off')
ax[1].imshow(segm_npy_frame, vmin=1, vmax=vmax, cmap=my_cmap)
text_label_centroid(rp_cells_labels_separate, ax[1], 12,
                    'semibold', 'center', 'center',
                    apply=True)
ax[1].axis('off')
frame_text = fig_text(fig, 'Current frame = {}/{}'.format(frame_i,num_frames),
                      y=frameTXT_y, x=frameTXT_x, color='w', size=14,
                      clear_all=False, clear_text_ref=True, text_ref=frame_text)

#Position and color of the buttons [left, bottom, width, height]
axcolor = '0.1'
slider_color = '0.2'
hover_color = '0.25'
presscolor = '0.35'
button_true_color = '0.5'
do_overlay = False
#Buttons axis
ax_prev_button = plt.axes([buttons_left,  sl_top, buttons_width, buttons_height])
ax_next_button = plt.axes([buttons_left+buttons_width,  sl_top, buttons_width, 0.03])
ax0_left, ax0_bottom, ax0_right, ax0_top = ax[0].get_position().get_points().flatten()
ax0_center = (ax0_right - ax0_left)*(1./2)+ax0_left
radio_buttons_width = buttons_width*3.2
ax_radio = plt.axes([ax0_center-(radio_buttons_width/2),
                     ax0_bottom-(buttons_height*3),
                     radio_buttons_width, buttons_height])
ax_overlay = plt.axes([0.1, 0.35, 0.1, 0.2])
ax_bright_sl = plt.axes([0.1, 0.48, 0.1, 0.2])
ax_alpha_sl = plt.axes([0.1, 0.69, 0.1, 0.2])
ax_rgb = plt.axes([0.1, 0.69, 0.25, 0.2])

#Create buttons
prev_button = Button(ax_prev_button, 'Prev. frame',
                     color=axcolor, hovercolor=hover_color)
next_button = Button(ax_next_button, 'Next frame',
                     color=axcolor, hovercolor=hover_color)
radio_buttons = MyRadioButtons(ax_radio,
                              ('Display Cells\' IDs',
                              'Display Cell Cycle stage',
                              'Display Cell Cycle INFO'),
                              active = 0,
                              activecolor = button_true_color,
                              orientation = 'horizontal',
                              size = 59,
                              circ_p_color = button_true_color)
overlay_b = Button(ax_overlay, 'Overlay', color=axcolor,
                hovercolor=hover_color, presscolor=presscolor)
brightness_slider = Slider(ax_bright_sl, 'Brightness', -1, 30,
                    valinit=4,
                    valstep=1,
                    color=slider_color,
                    init_val_line_color=hover_color,
                    valfmt='%1.0f')
alpha_slider = Slider(ax_alpha_sl, 'alpha overlay', -0.1, 1.1,
                    valinit=0.5,
                    valstep=0.01,
                    color=slider_color,
                    init_val_line_color=hover_color,
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

#Create event for close (required when both Tkinter windows and plt are used in the same script)
def handle_close(evt):
    root.quit()

#Create event for next and previous frame
def next_frame(event):
    global frame_i, frame_text, cca_df, rp_cells_labels_separate, ol_img
    if frame_i < num_frames:
        frame_i += 1
        if frame_i in cca_analysed_frames:
            cca_df = cc_stage_dfs.loc[frame_i]
        else:
            cca_df = []
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
                                  display_ccStage, cmap=my_cmap,
                                  ol_img=ol_img, do_overlay=do_overlay,
                                  ol_RGB_val=ol_RGB_val,
                                  ol_brightness=brightness_slider.val,
                                  ol_alpha=alpha_slider.val)
    elif frame_i+1 > num_frames:
        frame_i = -1
        print('You reached the last frame')
        next_frame(None)

def prev_frame(event):
    global frame_i, frame_text, cca_df, rp_cells_labels_separate, ol_img
    if frame_i > 0:
        frame_i -= 1
        if ccs_df_found:
            if frame_i in cca_analysed_frames:
                cca_df = cc_stage_dfs.loc[frame_i]
            else:
                cca_df = []
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
                                  display_ccStage, cmap=my_cmap,
                                  ol_img=ol_img, do_overlay=do_overlay,
                                  ol_RGB_val=ol_RGB_val,
                                  ol_brightness=brightness_slider.val,
                                  ol_alpha=alpha_slider.val)
    else:
        frame_i = num_frames+1
        print('You reached the first frame')
        prev_frame(None)

def key_pressed(event):
    if event.key == 'right':
        next_frame(None)
    elif event.key == 'left':
        prev_frame(None)
    else:
        pass

def radio_b_cb(label):
    global display_ccStage
    if label == 'Display Cells\' IDs':
        display_ccStage = 'IDs'
    elif label == 'Display Cell Cycle INFO':
        display_ccStage = 'All info'
    else:
        display_ccStage = 'Only stage'
    text_label_centroid(rp_cells_labels_separate, ax[0], 12,
                        'semibold', 'center', 'center', cca_df,
                        display_ccStage=display_ccStage, color='r', clear=True)
    (fig.canvas).draw_idle()

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
            elif ext == '.tif' or ext == '.tif':
                align_ol = True
                ol_frames = io.imread(ol_path)
            else:
                messagebox.showerror('File Format not supported!',
                    f'File format {ext} is not supported!\n'
                    'Choose either .tif or .npy files.')
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
            fig.canvas.draw_idle()

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
radio_buttons.on_clicked(radio_b_cb)
overlay_b.on_clicked(overlay_cb)
brightness_slider.on_changed(update_overlay_cb)
alpha_slider.on_changed(update_overlay_cb)

def resize_widgets(event):
    ax0_l, ax0_b, ax0_r, ax0_t = ax[0].get_position().get_points().flatten()
    ax0_center = (ax0_r - ax0_l)*(1./2)+ax0_l
    ax_prev_button.set_position([buttons_left,  sl_top,
                                 buttons_width, buttons_height])
    ax_next_button.set_position([buttons_left+buttons_width,  sl_top,
                                 buttons_width, buttons_height])
    if ccs_df_found:
        ax0_center = (ax0_r - ax0_l)*(1./2)+ax0_l
        ax_radio.set_position([ax0_center-(radio_buttons_width/2),
                               ax0_b-(buttons_height+0.01),
                               radio_buttons_width,
                               buttons_height])
    else:
        ax_radio.set_visible(False)
    ax_overlay.set_position([ax0_center-(buttons_width/2), ax0_t+0.01,
                             buttons_width, buttons_height])
    sl_h = 0.025
    ax_bright_sl.set_position([ax0_l, ax0_b-0.007-sl_h, ax0_r-ax0_l, sl_h])
    ax_alpha_sl.set_position([ax0_l, ax0_b-2*(0.007+sl_h), ax0_r-ax0_l, sl_h])
    ax_rgb.set_position([ax0_r+0.005, ax0_b, sl_h*2/3, ax0_t-ax0_b])

def mouse_down(event):
    global picked_rgb_marker, ol_RGB_val
    right_click = event.button == 3
    scroll_click = event.button == 2
    left_click = event.button == 1
    if left_click and event.inaxes == ax_rgb:
        picked_rgb_marker.remove()
        y_rgb = int(round(event.ydata))
        ol_RGB_val = rgb_cmap_array[y_rgb][:3]
        picked_rgb_marker = ax_rgb.scatter(x_rgb, y_rgb, marker='s', color='k')
        update_overlay_cb(event)

#Canvas events
(fig.canvas).mpl_connect('resize_event', resize_widgets)
(fig.canvas).mpl_connect('button_press_event', mouse_down)
(fig.canvas).mpl_connect('key_press_event', key_pressed)

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
    mng.window.wm_geometry(geom) #move GUI window to second monitor
                                 #with string "widthxheight+x+y"

pos_path = os.path.dirname(parent_path)
exp_path = os.path.dirname(pos_path)
pos_foldername = os.path.basename(pos_path)
exp_foldername = os.path.basename(exp_path)
fig.text(0.5, 0.95, pos_foldername, color = 'w', fontsize=12, #
         horizontalalignment='center')
fig.canvas.set_window_title(
            f'Cell segmentation slideshow - {exp_foldername}/{pos_foldername}')
plt.show()
