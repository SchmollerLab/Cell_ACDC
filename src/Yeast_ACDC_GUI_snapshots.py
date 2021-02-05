print('Importing modules...')
import os, sys, re, traceback, subprocess, cv2
from time import time
from sys import exit, exc_info
from copy import deepcopy
from natsort import natsorted
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib
from math import atan2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.path import Path
from tkinter import E, S, W, END
import tkinter as tk
import skimage
from skimage import io
from skimage.util import img_as_float, img_as_ubyte
from skimage.feature import peak_local_max
from skimage.filters import (gaussian, sobel, apply_hysteresis_threshold,
                            threshold_otsu, unsharp_mask)
from skimage.measure import (label, regionprops, subdivide_polygon,
                            find_contours, approximate_polygon)
from skimage.morphology import (remove_small_objects, convex_hull_image,
                                dilation, erosion, disk)
from skimage.draw import line, line_aa, polygon
from skimage.exposure import equalize_adapthist, rescale_intensity
from skimage.segmentation import relabel_sequential
from scipy.ndimage.morphology import binary_fill_holes, distance_transform_edt
from tifffile import TiffFile
from Yeast_ACDC_MyWidgets import Slider, Button, MyRadioButtons, TextBox
from Yeast_ACDC_FUNCTIONS import (auto_select_slice, separate_overlapping,
                       text_label_centroid, tk_breakpoint, manual_emerg_bud,
                       CellInt_slideshow, twobuttonsmessagebox,
                       single_entry_messagebox, beyond_listdir_pos,
                       select_exp_folder, expand_labels, tk_breakpoint,
                       folder_dialog, dark_mode, win_size, imshow_tk)

script_dirname = os.path.dirname(os.path.realpath(__file__))
unet_path = f'{script_dirname}/YeaZ-unet/unet/'

#append all the paths where the modules are stored. Such that this script
#looks into all of these folders when importing modules.
sys.path.append(unet_path)

"""
NOTE for implementing a NEW manual feature:
    1. Save the inputs required to perform that manual function
    2. Add the manual function name to .li_manual_func attribute of ia
    3. Add the the new manual function case to repeat_manual_func module of ia
"""

"""                          INSTRUCTIONS
GUI Mouse Events: - Apply lower local threshold with RIGHT-click drawing on
                    thresholded image: freely draw area where you want to apply
                    a lower local threshold. The lower local threshold is
                    calculated as Low T. - Delta Local T. sliders.
                  - Delete portions of thresholded image with MIDDLE-click on
                    thresholded image: click on the portion of thresholded
                    image that you want to delete. The portion is determined
                    by labelling the thresholded image with connectivity = 1
                  - Auto contour with RIGHT-click on edge img: click on the
                    edge image (if not visible press "Switch to edge mode")
                    where you clearly see a dark or bright contour to auto
                    segment the cell along that contour.
                  - Manual contour with RIGHT-click on edge img: roughly follow
                    a bright gradient contour on the edge img with pressing
                    the right button.
                  - Assign bud to mother with RIGHT-click press-release
                    on intensity img: click on the bud and release on mother
                  - Merge IDs with RIGHT-click press-release on label img:
                    draw a line that starts from first ID and ends on last ID.
                    All IDs on this line will be merged to single ID.
                  - Press 'ctrl+p' to print the cc_stage dataframe
                  - Manually separate bud with 'b' plus RIGHT-click on label
                    img
                  - Set which cell is mother cell (without assigning the bud)
                    with WHEEL-click on intensity img.
                  - Delete cell ID from label image with WHEEL-click on label
                  - Delete contour pixel with WHEEL-click on contour image:
                    wheel-click on the pixel you want to remove from contour
                  - Automatic zoom with left double click
                  - Restore original view (no zoom) with right double click
                    anywhere outside axis
                  - Use hull contour with left double click on Cell ID on label
                  - Approx hull contour with *a + left double click* on ID
                  - To remove objects from phase contrast image that do not
                    have to be considered paart of the background you can freely
                    draw around it. First, press the key 'd' to activate
                    the objects's removal mode. Then, with the LEFT button of
                    the mouse draw around any object on the LEFT image.
                    You don't need to close the contour. It will be automatically
                    closed when you release the mouse button. If drawing was
                    successfull an 'x' will appear at the center of the closed
                    drawn contour. The objects's removal mode will then be
                    automatically deactivated.
                  - To divide a cell into multiple cells along an intensity line
                    press 'c' and draw with RIGHT-click on edge where you want to
                    cell to be splitted
                  - Press 'ctrl+z' for undo and 'ctrl+y' for redo
                  - Zoom in with double click (LEFT button) on the RIGHT image
                  - Zoom out with RIGHT button double click anywhere outside
                    the images
                  - To store the current segmentation press 'f' or the buttons
                    "Freeze segmentation". You can now further segment onto
                    another z-slice for example and then press
                    "Release frozen segment" to add the stored segmentation
                    on top of the new one. To cancel press with left button
                    on "Release frozen segment"
                  - Delete all IDs inside a freely drawn rectangle: draw a
                    rectangle with the WHEEL-click on the segmented image
                  - Zoom-in: *shift + scroll* - the amount of zoom is
                    proportional to the speed of scrolling. You can adjust
                    the sensitivity by changing the "sensitivity" variable.
                    By default it is set to 6.
                  - Select labels: *ctrl+left-click* on any image. Click
                    any label's ID to select it (left or right image it doesn't
                    matter). Up to two labels can be selected.
                    Press "escape" to deselect all labels.
                  - Activate brush-mode: *e* key or click on "Brush mode" button
                    to toggle the brush mode on or off
                  - With brush mode ON you can toggle eraser on or off with 'x'
                    key. Change the brush size with "up/down" arrow keys.
                    To freely paint/erase use the left button on the IDs.
                    If you have selected IDs you will paint/erase only the
                    first ID of the selected IDs. If you don't have any
                    selected ID you will paint a new label and erase any label
                    touched by the eraser.
"""

class load_data:
    def __init__(self, path):
        self.path = path
        self.parent_path = os.path.dirname(path)
        self.filename, self.ext = os.path.splitext(os.path.basename(path))
        if self.ext == '.tif':
            self.tif_path = path
            img_data = io.imread(path)
        elif self.ext == '.npy':
            tif_path, _ = self.substring_path(path, 'phase_contr.tif',
                                           self.parent_path)
            self.tif_path = tif_path
            img_data = np.load(path)
        self.img_data = img_data
        self.info, self.metadata_found = self.metadata(self.tif_path)
        if self.metadata_found:
            try:
                self.SizeT, self.SizeZ = self.data_dimensions(self.info)
            except:
                print(exc_info())
                self.SizeT, self.SizeZ = self.dimensions_entry_widget()
        else:
            self.SizeT, self.SizeZ = self.dimensions_entry_widget()
        self.build_paths(self.filename, self.parent_path)

    def build_paths(self, filename, parent_path):
        match = re.search('s(\d+)_', filename)
        if match is not None:
            basename = filename[:match.span()[1]-1]
        else:
            match = re.search('s(\d+)-(\d+)_', filename)
            if match is not None:
                basename = filename[:match.span()[1]-1]
            else:
                basename = single_entry_messagebox(
                         entry_label='Write a common basename for all output files',
                         input_txt=filename,
                         toplevel=False).entry_txt
        base_path = f'{parent_path}/{basename}'
        self.slice_used_align_path = f'{base_path}_slice_used_alignment.csv'
        self.slice_used_segm_path = f'{base_path}_slice_segm.csv'
        self.align_npy_path = f'{base_path}_phc_aligned.npy'
        self.align_shifts_path = f'{base_path}_align_shift.npy'
        self.segm_npy_path = f'{base_path}_segm.npy'

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
        with TiffFile(tif_path) as tif:
            self.metadata = tif.imagej_metadata
        try:
            metadata_found = True
            info = self.metadata['Info']
        except KeyError:
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

        root.protocol("WM_DELETE_WINDOW", exit)

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

class num_pos_toSegm_tk:
    def __init__(self, tot_frames, last_segm_i=None):
        root = tk.Tk()
        self.root = root
        self.tot_frames = tot_frames
        root.geometry('+800+400')
        root.attributes("-topmost", True)
        tk.Label(root,
                 text="How many positions do you want to segment?",
                 font=(None, 12)).grid(row=0, column=0, columnspan=3)
        if last_segm_i is None or last_segm_i==0:
            tk.Label(root,
                     text=f"(There is a total of {tot_frames} positions).",
                     font=(None, 10)).grid(row=1, column=0, columnspan=3)
        elif last_segm_i==tot_frames:
            tk.Label(root,
                 text=f'(There is a total of {tot_frames} positions.\n'
                      f'All positions have already been segmented)',
                 font=(None, 10)).grid(row=1, column=0, columnspan=3)
        else:
            tk.Label(root,
                 text=f'(There is a total of {tot_frames} positions.\n'
                      f'Last segmented position number is {last_segm_i})',
                 font=(None, 10)).grid(row=1, column=0, columnspan=3)

        tk.Label(root,
                 text="Start Position",
                 font=(None, 10, 'bold')).grid(row=2, column=0, sticky=E, padx=4)
        tk.Label(root,
                 text="# of positions to analyze",
                 font=(None, 10, 'bold')).grid(row=3, column=0, padx=4)
        sv_sf = tk.StringVar()
        start_frame = tk.Entry(root, width=10, justify='center',font='None 12',
                            textvariable=sv_sf)
        start_frame.insert(0, '{}'.format(1))
        sv_sf.trace_add("write", self.set_all)
        self.start_frame = start_frame
        start_frame.grid(row=2, column=1, pady=8, sticky=W)
        sv_num = tk.StringVar()
        num_frames = tk.Entry(root, width=10, justify='center',font='None 12',
                                textvariable=sv_num)
        self.num_frames = num_frames
        num_frames.insert(0, '{}'.format(tot_frames))
        sv_num.trace_add("write", self.check_max)
        num_frames.grid(row=3, column=1, pady=8, sticky=W)
        tk.Button(root,
                  text='All',
                  command=self.set_all,
                  width=8).grid(row=3,
                                 column=2,
                                 pady=4, padx=4)
        tk.Button(root,
                  text='OK',
                  command=self.ok,
                  width=12).grid(row=4,
                                 column=0,
                                 pady=8,
                                 columnspan=3)
        root.bind('<Return>', self.ok)
        start_frame.focus_force()
        start_frame.selection_range(0, END)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        root.mainloop()

    def set_all(self, name=None, index=None, mode=None):
        start_frame_str = self.start_frame.get()
        if start_frame_str:
            startf = int(start_frame_str)
            rightRange = self.tot_frames - startf + 1
            self.num_frames.delete(0, END)
            self.num_frames.insert(0, '{}'.format(rightRange))

    def check_max(self, name=None, index=None, mode=None):
        num_frames_str = self.num_frames.get()
        start_frame_str = self.start_frame.get()
        if num_frames_str and start_frame_str:
            startf = int(start_frame_str)
            if startf + int(num_frames_str) > self.tot_frames:
                rightRange = self.tot_frames - startf + 1
                self.num_frames.delete(0, END)
                self.num_frames.insert(0, '{}'.format(rightRange))

    def ok(self, event=None):
        num_frames_str = self.num_frames.get()
        start_frame_str = self.start_frame.get()
        if num_frames_str and start_frame_str:
            startf = int(self.start_frame.get())
            num = int(self.num_frames.get())
            stopf = startf + num
            self.frange = (startf, stopf)
            self.root.quit()
            self.root.destroy()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

"""Classes"""
class app_GUI:
    def __init__(self, TIFFs_path):
        directories = natsorted(os.listdir(TIFFs_path))
        self.pos_foldernames = directories
        TIFFs_parent_path = os.path.dirname(TIFFs_path)
        exp_folder_parent_path = os.path.dirname(TIFFs_parent_path)
        self.exp_name = os.path.basename(TIFFs_parent_path)
        self.exp_parent_foldername = os.path.basename(exp_folder_parent_path)
        self.segm_npy_done = [None]*len(directories)
        self.orig_segm_npy = [None]*len(directories)
        self.masks = [None]*len(directories)
        self.cc_stages = [None]*len(directories)
        self.saved_cc_stages = [None]*len(directories)
        self.slices_used = [None]*len(directories)
        self.saved_slices_used = [None]*len(directories)
        self.is_pc = twobuttonsmessagebox('Img mode', 'Select imaging mode',
                                         'Phase contrast', 'Bright-field'
                                         ).button_left
        print('Loading all images...')

        data_pos, Phc, pos_paths, basenames, slices = self.get_allPos_paths(
                                                                    directories)

        phc_li = [pos.shape for pos in Phc]
        counter = Counter(phc_li)
        phc_shapes = counter.keys()
        if len(phc_shapes) > 1:
            c_li = list(counter)
            common_shape, count = counter.most_common(1)[0]
            c_li.remove(common_shape)
            idx = [phc_li.index(shape) for shape in c_li]
            wrong_shape_pos_li = [os.path.basename(
                                    os.path.dirname(pos_paths[i])) for i in idx]
            tk.messagebox.showerror('Wrong image shape',
                          f'Positions {wrong_shape_pos_li} have non consistent shape'
                          f'(non consistent shapes: {c_li})')
            exit()
        phc = np.asarray(Phc)
        self.all_phc = phc
        last_segm_pos = len(phc)
        for e, elem in enumerate(self.segm_npy_done):
            if elem is None:
                last_segm_pos = e
                break
        start, _ = (num_pos_toSegm_tk(len(phc), last_segm_i=last_segm_pos).frange)
        start -= 1
        self.num_slices = data_pos.SizeZ
        if self.slices_used is not None:
            self.slices = slices
        else:
            self.slices = [1]
        self.pos_paths = pos_paths
        self.basenames = basenames
        self.phc = phc
        self.display_ccStage = 'Only stage'
        self.p = start
        self.reset_view = False
        self.use_unet = True
        self.manual_verts = []
        self.prev_states = []
        self.already_approx_IDs = []
        self.is_undo = False
        self.is_redo = False
        self.do_cut = False
        self.is_lab_frozen = False
        self.do_approx = False
        self.zoom_on = False
        self.scroll_zoom = False
        self.select_ID_on = False
        self.brush_mode_on = False
        self.eraser_on = False
        self.selected_IDs = None
        self.set_labRGB_colors()
        print('Total number of Positions = {}'.format(len(self.phc)))
        self.bp = tk_breakpoint()
        self.init_attr()

    def get_allPos_paths(self, directories):
        Phc = []
        pos_paths = []
        basenames = []
        slices = []
        for i, d in enumerate(directories):
            pos_path = '{}/{}/Images'.format(TIFFs_path, d)
            if os.path.isdir(pos_path):
                pos_paths.append(pos_path)
                filenames = os.listdir(pos_path)
                p_found = False
                slice_found = False
                for j, p in enumerate(filenames):
                    temp_pIDX = p.find('_phase_contr.tif')
                    if temp_pIDX != -1:
                        p_idx = temp_pIDX
                        k = j
                        p_found = True
                    elif p.find('slice_segm.txt') != -1:
                        slIDX = j
                        slice_found = True
                        slice_path = '{}/{}'.format(pos_path, filenames[slIDX])
                        with open(slice_path, 'r') as slice_txt:
                            slice = slice_txt.read()
                            slice = int(slice)
                        self.slices_used[i] = slice
                        self.saved_slices_used[i] = slice
                    elif p.find('_segm.npy') != -1:
                        segm_npy_path = '{}/{}'.format(pos_path, p)
                        self.segm_npy_done[i] = np.load(segm_npy_path)
                        self.orig_segm_npy[i] = self.segm_npy_done[i].copy()
                    elif p.find('_mask.npy') != -1:
                        mask_npy_path = '{}/{}'.format(pos_path, p)
                        self.masks[i] = np.load(mask_npy_path)
                    elif p.find('_cc_stage.csv') != -1:
                        cc_stage_path = '{}/{}'.format(pos_path, p)
                        cc_stage_df = pd.read_csv(cc_stage_path,
                                                  index_col=['Cell_ID'])
                        self.cc_stages[i] = cc_stage_df.copy()
                        self.saved_cc_stages[i] = cc_stage_df.copy()
                if p_found:
                    Phc_path = '{}/{}'.format(pos_path, filenames[k])
                    data_pos = load_data(Phc_path)
                    Phc.append(data_pos.img_data)
                    base_name = p[0:p_idx]
                    basenames.append(base_name)
                else:
                    tk.messagebox.showerror('File not found!',
                        'The script could not find the "..._phase_contr.tif" '
                        f'file in {d} folder.\n To fix this error you need '
                        'to make sure that every Position_n folder contains '
                        'a file that ends in "phase_contr.tif".\n'
                        'To do so you probably have to run the Fiji script again'
                        ' and make sure that the "channels" variable contains '
                        'the value "phase_contr" along with the other channels.')
                    raise FileNotFoundError('phase_contr.tif file not found in '
                                            f'{d} folder')
                    Phc.append(np.zeros((600,600), int))
                if slice_found:
                    slice_path = '{}/{}'.format(pos_path, filenames[slIDX])
                    with open(slice_path, 'r') as slice_txt:
                        slice = slice_txt.read()
                        slice = int(slice)
                    slices.append(slice)
                else:
                    slices.append(-1)
        return data_pos, Phc, pos_paths, basenames, slices

    def preprocess_img_data(self, img):
        # print('Preprocessing image...')
        # gauss_filt = gaussian(img)
        # sharp_img = img_as_ubyte(unsharp_mask(gauss_filt, radius=50, amount=2))
        # return sharp_img
        return img

    def store_state(self, ia):
        if self.is_undo or self.is_redo:
            self.prev_states = []
            self.is_undo = False
        self.prev_states.append(deepcopy(ia))
        self.count_states = len(self.prev_states)-1

    def get_state(self, ia, count):
        if count >= 0 and count < len(self.prev_states):
            return self.prev_states[count]
        else:
            return ia

    def init_attr(self):
        # Local threshold polygon attributes
        # self.s = auto_select_slice(self.phc[self.p], init_slice=self.init_s).slice
        if self.slices[self.p] == -1:
            self.s = auto_select_slice(self.phc[self.p], init_slice=0).slice
        else:
            self.s = self.slices[self.p]
        self.cid2_rc = None  # cid for right click on ax2 (thresh)
        self.Line2_rc = None
        self.xdrc = None
        self.ydrc = None
        self.xdrrc = None
        self.ydrrc = None
        self.xurc = None
        self.yurc = None
        self.key_mode = ''

    def init_plots(self, left=0.08, bottom=0.25):
        fig, ax = plt.subplots(1,3)
        plt.subplots_adjust(left=left, bottom=bottom)
        self.mng = plt.get_current_fig_manager()
        self.fig = fig
        self.ax = ax
        self.cids_lim = [[0,0] for _ in range(3)]
        self.ax_limits = [[(),()] for _ in range(3)]
        self.orig_ax_limits = [[(),()] for _ in range(3)]

    def connect_axes_cb(self):
        ax = self.ax
        for i, axes in enumerate(ax):
            cidx = axes.callbacks.connect('xlim_changed', on_xlim_changed)
            cidy = axes.callbacks.connect('ylim_changed', on_ylim_changed)
            self.cids_lim[i][0] = cidx
            self.cids_lim[i][1] = cidy

    def set_orig_lims(self):
        for a, axes in enumerate(self.ax):
            self.orig_ax_limits[a][0] = axes.get_xlim()
            self.orig_ax_limits[a][1] = axes.get_ylim()

    def set_lims(self):
        if self.reset_view:
            for a, axes in enumerate(self.ax):
                self.ax_limits = deepcopy(self.orig_ax_limits)
                self.ax_limits = deepcopy(self.orig_ax_limits)
                axes.set_xlim(*self.orig_ax_limits[a][0])
                axes.set_ylim(*self.orig_ax_limits[a][1])
        else:
            for a, axes in enumerate(self.ax):
                if self.ax_limits[a][0]:
                    axes.set_xlim(*self.ax_limits[a][0])
                if self.ax_limits[a][1]:
                    axes.set_ylim(*self.ax_limits[a][1])

    def approx_contour(self, lab_ID, method='hull'):
        if method == 'hull':
            approx_lab = convex_hull_image(lab_ID)
        elif method == 'approx':
            approx_lab = np.zeros_like(lab_ID)
            for contour in find_contours(lab_ID, 0):
                coords = approximate_polygon(contour, tolerance=3.5)
                r, c = subdivide_polygon(coords, degree=3).transpose()
                rr, cc = polygon(r, c)
                approx_lab[rr, cc] = 1
        return approx_lab

    def relabel_and_fill_lab(self, lab):
        IDs = [obj.label for obj in regionprops(lab)]
        lab, _, _ = relabel_sequential(lab)
        for ID in IDs:
            mask_ID = lab==ID
            filled_mask_ID = binary_fill_holes(mask_ID)
            lab[filled_mask_ID] = ID
        return lab

    def get_overlay(self, img, ol_img, ol_RGB_val=[1,1,0],
                    ol_brightness=4, ol_alpha=0.5):
        img_rgb = gray2rgb(img_as_float(img))*ol_RGB_val
        ol_img_rgb = gray2rgb(img_as_float(ol_img))
        overlay = (ol_img_rgb*(1.0 - ol_alpha)+img_rgb*ol_alpha)*ol_brightness
        overlay = (np.clip(overlay, 0, 1)*255).astype(np.uint8)
        return overlay

    def update_ax0_plot(self, ia, img, draw=True):
        ax0 = self.ax[0]
        ax0.clear()
        ax0.imshow(img)
        for cont in ia.contours:
            x = cont[:,1]
            y = cont[:,0]
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            ax0.plot(x, y, c='r')
        if ia.manual_mask is not None:
            mask_lab = label(ia.manual_mask)
            mask_rp = regionprops(mask_lab)
            mask_ids = [obj.label for obj in mask_rp]
            mask_contours = ia.find_contours(mask_lab, mask_ids, group=True)
            for obj, cont in zip(mask_rp, mask_contours):
                x = cont[:,1]
                y = cont[:,0]
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                ax0.plot(x, y, c='r')
                yc, xc = obj.centroid
                ax0.scatter(xc, yc, s=72, c='r', marker='x')
        try:
            text_label_centroid(ia.rp, ax0, 12, 'semibold', 'center', 'center',
                                cc_stage_frame=ia.cc_stage_df,
                                display_ccStage=self.display_ccStage, color='r',
                                clear=True)
        except:
            traceback.print_exc()
        ax0.axis('off')
        if draw:
            self.fig.canvas.draw_idle()

    def update_ALLplots(self, ia):
        fig, ax = self.fig, self.ax
        img = ia.img
        edge = ia.edge
        lab = ia.lab
        rp = ia.rp
        self.update_ax0_plot(ia, img)
        self.update_ax1_plot(lab, rp, ia, draw=False)
        self.update_ax2_plot(ia, draw=False)
        fig.canvas.draw_idle()

    def update_ax2_plot(self, ia, draw=True):
        ax = self.ax
        ax[2].clear()
        ax[2].imshow(ia.edge)
        for cont in ia.contours:
            x = cont[:,1]
            y = cont[:,0]
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            ax[2].plot(x, y, c='silver', alpha=0.5)
        xx, yy = ia.contour_plot
        ax[2].scatter(xx, yy, s=1.5, c='r')
        # print('Drawing contour time = {0:6f}'.format(time()-t0))
        ax[2].axis('off')
        if draw:
            self.fig.canvas.draw_idle()

    def set_labRGB_colors(self, start_c=64, max_ID=10):
        # Generate a colormap as sparse as possible given the max ID.
        gradient = np.linspace(start_c, 255, max_ID, dtype=int)
        labelRGB_colors = np.asarray([plt.cm.viridis(i) for i in gradient])
        # Randomly shuffle the colormap to minimize the possibility for two closeby
        # ID of having a similar color.
        np.random.shuffle(labelRGB_colors)
        self.labRGB_colors = labelRGB_colors

    def get_labels_overlay(self, labRGB, img, bg_label=0):
        img_rgb = skimage.color.gray2rgb(skimage.img_as_float(img))
        overlay = (img_rgb*(1.0 - 0.3) + labRGB*0.3)*1
        overlay = np.clip(overlay, 0, 1)
        # overlay = skimage.color.label2rgb(lab, image=img, bg_label=bg_label)
        return overlay

    def update_ax1_plot(self, lab, rp, ia, draw=True,
                        draw_brush_circle=False, event_x=None, event_y=None):
        ia.rp = rp
        ia.cc_stage_df = ia.init_cc_stage_df(rp)
        ia.cc_stage_df = ia.assign_bud(ia.cc_stage_df, rp)
        ia.lab = lab
        labRGB = skimage.color.label2rgb(lab, colors=self.labRGB_colors,
                                             bg_label=0, bg_color=(0.1,0.1,0.1))
        if self.selected_IDs is not None:
            labRGB_opaque = 0.1*labRGB
            bg_mask = lab==0
            labRGB_opaque[bg_mask] = labRGB[bg_mask]
            mask_selected_IDs = np.zeros(lab.shape, bool)
            for ID in self.selected_IDs:
                mask_selected_IDs[lab==ID] = True
            labRGB_opaque[mask_selected_IDs] = labRGB[mask_selected_IDs]
            labRGB = labRGB_opaque
        if self.brush_mode_on:
            ax1_img = self.get_labels_overlay(labRGB, ia.img, bg_label=0)
        else:
            ax1_img = labRGB
        ax = self.ax
        ax[1].clear()
        ax[1].imshow(ax1_img)
        text_label_centroid(ia.rp, ax[1], 12, 'semibold', 'center', 'center',
                            selected_IDs=self.selected_IDs)
        ax[1].axis('off')
        self.set_lims()
        if draw:
            if draw_brush_circle:
                self.draw_brush_circle_cursor(ia, event_x, event_y)
            self.fig.canvas.draw_idle()

    def draw_brush_circle_cursor(self, ia, event_x, event_y):
        x_mot, y_mot = self.ax_transData_and_coerce(self.ax[1],
                                                event_x, event_y, ia.img.shape,
                                                return_int=False)
        try:
            self.brush_circle.set_visible(False)
            self.brush_circle.remove()
            self.ax[1].patches = []
        except:
            pass
        c = 'r' if self.eraser_on else 'w'
        self.brush_circle = matplotlib.patches.Circle((x_mot, y_mot),
                                radius=self.brush_size,
                                fill=False,
                                color=c, alpha=0.7)
        self.ax[1].add_patch(self.brush_circle)
        self.fig.canvas.draw_idle()

    def apply_brush_motion(self, event_x, event_y, ia):
        x_mot, y_mot = self.ax_transData_and_coerce(self.ax[1],
                                                event_x, event_y, ia.img.shape,
                                                return_int=False)

        rr, cc = skimage.draw.disk((y_mot, x_mot), radius=self.brush_size,
                                   shape=ia.lab.shape)
        self.brush_mask[rr, cc] = True
        erased_IDs = np.unique(ia.lab[self.brush_mask])
        erased_IDs = [ID for ID in erased_IDs if ID!=0]
        # Check that we don't erase non selected IDs
        if self.selected_IDs is not None:
            # Check if erased_IDs contains non-selected IDs
            non_selected_brushed_IDs = [ID for ID in erased_IDs
                                           if ID not in self.selected_IDs]
            if non_selected_brushed_IDs:
                for non_selected_ID in non_selected_brushed_IDs:
                    self.brush_mask[ia.lab==non_selected_ID] = False
        # Apply either eraser or painter
        if self.eraser_on:
            ia.lab[self.brush_mask] = 0
        else:
            if self.selected_IDs is not None:
                ia.lab[self.brush_mask] = self.selected_IDs[0]
            else:
                ia.lab[self.brush_mask] = self.new_ID
        self.update_ax1_plot(ia.lab, ia.rp, ia,
                             draw_brush_circle=True,
                             event_x=event_x, event_y=event_y)
        self.fig.canvas.draw_idle()

    def save_pos(self, param, pos_path, i):
        mask_npy_path = f'{pos_path}/{self.basenames[i]}_mask.npy'
        segm_npy_path = '{}/{}_segm.npy'.format(pos_path, self.basenames[i])
        ccstage_path = '{}/{}_cc_stage.csv'.format(pos_path, self.basenames[i])
        slice_path = '{}/{}_slice_segm.txt'.format(pos_path, self.basenames[i])
        pos_foldername = os.path.basename(os.path.dirname(pos_path))
        ia = param[i]
        save_lab = False
        if app.orig_segm_npy[i] is None:
            save_lab = True
        elif not (app.orig_segm_npy[i] == ia.lab).all():
            save_lab = True
        if save_lab:
            print(f'Saving {pos_foldername}..')
            np.save(segm_npy_path, ia.lab, allow_pickle=False)
            print(f'    Segmentation file for {pos_foldername} saved in\n'
                  f'    {segm_npy_path}')
            app.orig_segm_npy[i] = ia.lab.copy()
        save_mask = False
        if app.masks[i] is None:
            save_mask = True
        elif not (app.masks[i] == ia.manual_mask).all():
            save_mask = True
        if save_mask:
            if not save_lab:
                print(f'Saving {pos_foldername}..')
            np.save(mask_npy_path, ia.manual_mask, allow_pickle=False)
            print(f'    Manual mask file for {pos_foldername} saved in\n'
                  f'    {mask_npy_path}')
            app.masks[i] = ia.manual_mask.copy()
        save_cc_stage_df = False
        if app.saved_cc_stages[i] is None:
            save_cc_stage_df = True
        elif not app.saved_cc_stages[i].equals(ia.cc_stage_df):
            save_cc_stage_df = True
        if save_cc_stage_df:
            if not save_lab and not save_mask:
                print(f'Saving {pos_foldername}..')
            ia.cc_stage_df.to_csv(ccstage_path, index=True, mode='w',
                                    encoding='utf-8-sig')
            print(f'    Cell cycle analysis file for {pos_foldername} saved in\n'
                  f'    {ccstage_path}')
            app.saved_cc_stages[i] = ia.cc_stage_df.copy()
        save_used_slice = False
        if app.saved_slices_used[i] is None:
            save_used_slice = True
        elif not app.saved_slices_used[i] == ia.slice_used:
            save_used_slice = True
        if save_used_slice:
            if not save_lab and not save_mask and not save_cc_stage_df:
                print(f'Saving {pos_foldername}..')
            with open(slice_path, 'w') as txt:
                txt.write(str(ia.slice_used))
            print(f'    Slice used for segmentation for {pos_foldername} saved in\n'
                  f'    {slice_path}')
            app.saved_slices_used[i] = ia.slice_used

    def ax_transData_and_coerce(self, ax, event_x, event_y, img_shape,
                                return_int=True):
        x, y = ax.transData.inverted().transform((event_x, event_y))
        ymax, xmax = img_shape
        xmin, ymin = 0, 0
        if x < xmin:
            x_coerced = 0
        elif x > xmax:
            x_coerced = xmax
        else:
            x_coerced = int(round(x)) if return_int else x
        if y < ymin:
            y_coerced = 0
        elif y > ymax:
            y_coerced = ymax
        else:
            y_coerced = int(round(y)) if return_int else y
        return x_coerced, y_coerced


class img_analysis:
    def __init__(self, img):
        lowT_f = 0.1
        highT_f = 0.8
        img_slice_max = img.max()
        img_slice_min = img.min()
        init_lowT = img_slice_min + ((img_slice_max - img_slice_min) * lowT_f)
        init_highT = img_slice_max * highT_f
        self.lowT = init_lowT
        self.highT = init_highT
        self.clos_mode = 'Auto closing'
        self.bp = tk_breakpoint()
        self.modified = False
        self.contours = []
        self.exclude_border = 30
        self.peak_dist = 10
        self.draw_bg_mask_on = False
        self.frozen_lab = None
        y, x = img.shape
        self.home_ax_limits = [[(-0.5, x-0.5), (y-0.5, -0.5)] for _ in range(3)]
        self.init_attr(img)

    def init_attr(self, img):
        # Local threshold polygon attributes
        self.locT_img = np.zeros(img.shape, bool)
        self.auto_edge_img = np.zeros(img.shape, bool)
        self.manual_mask = np.zeros(img.shape, bool)
        self.li_Line2_rc = []
        self.locTval = None
        self.Line_Poly = []
        self.del_yx_thresh = []
        self.li_manual_func = []
        self.li_yx_dir_coords = []
        self.merge_yx = [0, 0, 0, 0]
        self.li_merge_yx = []
        self.sep_bud = False
        self.set_moth = False
        self.contour_plot = [[],[]]
        self.enlarg_first_call = True
        self.reduce_first_call = True
        self.prev_bud_assignment_info = None
        self.prev_cc_stage_df = None

    def nearest_nonzero(self, a, y, x):
        r, c = np.nonzero(a)
        dist = ((r - y)**2 + (c - x)**2)
        min_idx = dist.argmin()
        return a[r[min_idx], c[min_idx]]

    def get_dir_coords(self, alfa_dir, yd, xd, shape, connectivity=1):
        h, w = shape
        y_above = yd+1 if yd+1 < h else yd
        y_below = yd-1 if yd > 0 else yd
        x_right = xd+1 if xd+1 < w else xd
        x_left = xd-1 if xd > 0 else xd
        if alfa_dir == 0:
            yy = [y_below, y_below, yd, y_above, y_above]
            xx = [xd, x_right, x_right, x_right, xd]
        elif alfa_dir == 45:
            yy = [y_below, y_below, y_below, yd, y_above]
            xx = [x_left, xd, x_right, x_right, x_right]
        elif alfa_dir == 90:
            yy = [yd, y_below, y_below, y_below, yd]
            xx = [x_left, x_left, xd, x_right, x_right]
        elif alfa_dir == 135:
            yy = [y_above, yd, y_below, y_below, y_below]
            xx = [x_left, x_left, x_left, xd, x_right]
        elif alfa_dir == -180 or alfa_dir == 180:
            yy = [y_above, y_above, yd, y_below, y_below]
            xx = [xd, x_left, x_left, x_left, xd]
        elif alfa_dir == -135:
            yy = [y_below, yd, y_above, y_above, y_above]
            xx = [x_left, x_left, x_left, xd, x_right]
        elif alfa_dir == -90:
            yy = [yd, y_above, y_above, y_above, yd]
            xx = [x_left, x_left, xd, x_right, x_right]
        else:
            yy = [y_above, y_above, y_above, yd, y_below]
            xx = [x_left, xd, x_right, x_right, x_right]
        if connectivity == 1:
            return yy[1:4], xx[1:4]
        else:
            return yy, xx

    def init_manual_cont(self, app, xd, yd):
        app.xdrc = xd
        app.ydrc = yd
        app.y0x0 = (yd, xd)  # initialize first (y,x)
        self.yx_dir_coords = [(yd, xd)]
        self.auto_edge_img[yd, xd] = True


    def manual_contour(self, app, ydr, xdr, size=20):
        y0, x0 = app.y0x0
        Dy = abs(ydr-y0)
        Dx = abs(xdr-x0)
        if Dy != 0 or Dx != 0:
            iter = int(round(max((Dy, Dx))))
            fig, ax = app.fig, app.ax
            alfa = atan2(y0-ydr, xdr-x0)
            base = np.pi/4
            alfa_dir = round((base * round(alfa/base))*180/np.pi)
            for _ in range(iter):
                y0, x0 = app.y0x0
                yy, xx = self.get_dir_coords(alfa_dir, y0, x0, self.edge.shape)
                a_dir = self.edge[yy, xx]
                min_int = a_dir.max() # if int_val > ta else a_dir.min()
                min_i = list(a_dir).index(min_int)
                y, x = yy[min_i], xx[min_i]
                line_du = Line2D([x0, x], [y0, y], color='r')
                ax[2].add_line(line_du)
                yd, xd = y, x
                app.y0x0 = (yd, xd)
                self.yx_dir_coords.append((yd, xd))
                ylen, xlen = self.auto_edge_img.shape
                if yd < ylen and xd < xlen:
                    self.auto_edge_img[yd, xd] = True
                    self.contour_plot[0].append(xd)
                    self.contour_plot[1].append(yd)
                fig.canvas.draw_idle()

    def reset_auto_edge_img(self, contours):
        self.auto_edge_img = np.zeros_like(self.auto_edge_img)
        self.contour_plot = [[],[]]
        for cont in contours:
            x = cont[:,1]
            y = cont[:,0]
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            self.contour_plot[0].extend(x)
            self.contour_plot[1].extend(y)
            self.auto_edge_img[y, x] = True

    def relabel(self, lab, startID=0):
        IDs = np.unique(lab)
        relabeled_lab = np.zeros_like(lab)
        for i, ID in enumerate(IDs):
            if ID != 0:
                ID_mask = lab==ID
                relabeled_lab[ID_mask] = i+1+startID
        return relabeled_lab

    def close_manual_cont(self):
        self.li_yx_dir_coords.append(self.yx_dir_coords)
        self.auto_edge_img = binary_fill_holes(self.auto_edge_img)
        lab, rp = self.separate_overlap(label(self.auto_edge_img))
        return lab, rp

    def repeat_manual_func(self, thresh):
        del_yx_T_i = 0
        self.thresh = thresh
        for func in self.li_manual_func:
            if func == 'Local T.':
                self.local_thresh(thresh, update_plots=False)
                self.thresh = thresh
            elif func == 'Del. yx thresh':
                y, x = self.del_yx_thresh[del_yx_T_i]
                lab_thresh = label(self.thresh, connectivity=1)
                ID = lab_thresh[y, x]
                self.thresh[lab_thresh == ID] = 0
                del_yx_T_i += 1

    def edge_detector(self, img):
        img_DC = (img.max()-img.min())/2
        img = np.abs(img-img_DC)
        # plt.imshow_tk(img)
        vmin, vmax = np.percentile(img, q=(0.5, 99.5))
        img = rescale_intensity(img,
                                in_range=(vmin, vmax),
                                out_range=np.float32)
        edge = sobel(equalize_adapthist(img))
        edge = gaussian(edge, sigma=1, preserve_range=True)
        edge = unsharp_mask(edge, radius=5, amount=2)
        return edge

    def auto_contour(self, app, start_yx=None, alfa_dir=None,
                     iter=300, draw=True):
        fig, ax = app.fig, app.ax
        if start_yx is None:
            yd, xd = app.ydrc, app.xdrc
        else:
            yd, xd = start_yx
        if alfa_dir is None:
            yu, xu = app.yu2rc, app.xu2rc
            alfa = atan2(yd-yu, xu-xd)
            base = np.pi/4
            alfa_dir = round((base * round(alfa/base))*180/np.pi)
        yx_dir_coords = [(yd, xd)]
        xx_plot = [xd]
        yy_plot = [yd]
        self.auto_edge_img[yd, xd] = True
        int_val = self.edge[yd, xd]
        base = np.pi/4
        for _ in range(iter):
            # Given three neighboring pixels in the direction of the line
            # connecting i-1 and i-2 points determine next landing point
            # as the pixel with the max intensity (between the three pixels)
            yy, xx = self.get_dir_coords(alfa_dir, yd, xd, self.edge.shape)
            a_dir = self.edge[yy, xx]
            min_cost = a_dir.max() # if int_val > ta else a_dir.min()
            min_cost_i = list(a_dir).index(min_cost)
            y, x = yy[min_cost_i], xx[min_cost_i]
            if draw:
                line_du = Line2D([xd, x], [yd, y], color='r')
                ax[2].add_line(line_du)
            alfa = atan2(yd-y, x-xd)
            alfa_dir = round((base * round(alfa/base))*180/np.pi)
            yd, xd = y, x
            yx_dir_coords.append((yd, xd))
            xx_plot.append(xd)
            yy_plot.append(yd)
            ylen, xlen = self.auto_edge_img.shape
            if yd < ylen and xd < xlen:
                self.auto_edge_img[yd, xd] = True
                self.contour_plot[0].append(xd)
                self.contour_plot[1].append(yd)
        if draw:
            fig.canvas.draw_idle()
        else:
            self.contour_plot[0].extend(xx_plot)
            self.contour_plot[1].extend(yy_plot)
        self.li_yx_dir_coords.append(yx_dir_coords)
        self.auto_edge_img = binary_fill_holes(self.auto_edge_img)
        lab, rp = self.separate_overlap(label(self.auto_edge_img))
        return lab, rp

    def find_contours(self, label_img, cells_ids, group=False, concat=False,
                      return_hull=False):
        contours = []
        for id in cells_ids:
            label_only_cells_ids_img = np.zeros_like(label_img)
            label_only_cells_ids_img[label_img == id] = id
            uint8_img = (label_only_cells_ids_img > 0).astype(np.uint8)
            cont, hierarchy = cv2.findContours(uint8_img,cv2.RETR_LIST,
                                               cv2.CHAIN_APPROX_NONE)
            cnt = cont[0]
            if return_hull:
                hull = cv2.convexHull(cnt,returnPoints = True)
                contours.append(hull)
            else:
                contours.append(cnt)
        if concat:
            all_contours = np.zeros((0,2), dtype=int)
            for contour in contours:
                contours_2D_yx = np.fliplr(np.reshape(contour,
                                                     (contour.shape[0],2)))
                all_contours = np.concatenate((all_contours, contours_2D_yx))
        elif group:
            # Return a list of n arrays for n objects. Each array has i rows of
            # [y,x] coords for each ith pixel in the nth object's contour
            all_contours = [[] for _ in range(len(cells_ids))]
            for c in contours:
                c2Dyx = np.fliplr(np.reshape(c, (c.shape[0],2)))
                for y,x in c2Dyx:
                    ID = label_img[y, x]
                    idx = list(cells_ids).index(ID)
                    all_contours[idx].append([y,x])
            all_contours = [np.asarray(li) for li in all_contours]
            IDs = [label_img[c[0,0],c[0,1]] for c in all_contours]
        else:
            all_contours = [np.fliplr(np.reshape(contour,
                            (contour.shape[0],2))) for contour in contours]
        return all_contours

    def thresholding(self, edge, lowT, highT):
        self.lowT = lowT
        self.highT = highT
        thresh = apply_hysteresis_threshold(edge, lowT, highT)
        return thresh

    def local_thresh(self, thresh, app=None, update_plots=True):
        thresh[self.locT_img] = self.edge[self.locT_img] > self.locTval
        if update_plots:
            app.update_ax2_plot(self, thresh)
            lab, rp = self.segmentation(thresh)
            app.update_ax1_plot(lab, rp, self)

    def remove_inner_obj(self, thresh):
        lab = label(thresh)
        rp = regionprops(lab)
        for o in rp:
            ID_lab = np.zeros_like(lab)
            ID_lab[o.coords[:,0], o.coords[:,1]] = o.label
            ID_fill = binary_fill_holes(ID_lab)
            IDs = list(np.unique(lab[ID_fill]))
            IDs.remove(o.label)
            if len(IDs) > 1:
                for ID in IDs:
                    thresh[lab == ID] = False
        return thresh

    def segmentation(self, thresh):
        thresh = self.remove_inner_obj(thresh)
        inv_thresh = np.invert(thresh)
        lab = label(inv_thresh)
        rp = regionprops(lab)
        lab[lab == rp[0].label] = 0  # remove background object
        lab = label(lab)
        rp = regionprops(lab)
        # Fill holes
        for o in rp:
            ID_lab = np.zeros_like(lab)
            ID = o.label
            ID_lab[o.coords[:,0], o.coords[:,1]] = ID
            ID_fill = binary_fill_holes(ID_lab)
            lab[ID_fill] = ID
        lab, rp = self.separate_overlap(lab)
        return lab, rp

    def separate_overlap(self, lab):
        dist = distance_transform_edt(lab)
        try:
            lab, num_cells, rp = separate_overlapping(lab, dist, selem_size=3)
            lab = remove_small_objects(lab, min_size=20, connectivity=2)
            rp = regionprops(lab)
            IDs = [obj.label for obj in rp]
            self.contours = self.find_contours(lab, IDs, group=True)
        except:
            rp = regionprops(lab)
            traceback.print_exc()
        self.rp = rp
        return lab, rp

    def init_cc_stage_df(self, rp, reset=False):
        IDs = [obj.label for obj in rp]
        cc_stage = ['G1' for ID in IDs]
        num_cycles = [-1]*len(IDs)
        relationship = ['mother' for ID in IDs]
        related_to = [0]*len(IDs)
        OF = np.zeros(len(IDs), bool)
        df = pd.DataFrame({
                            'Cell cycle stage': cc_stage,
                            '# of cycles': num_cycles,
                            'Relative\'s ID': related_to,
                            'Relationship': relationship,
                            'OF': OF},
                            index=IDs)
        df.index.name = 'Cell_ID'
        if self.prev_cc_stage_df is not None and not reset:
            prev_df = self.prev_cc_stage_df.filter(items=IDs, axis=0)
            current_df = df.drop(prev_df.index)
            return pd.concat([prev_df, current_df]).sort_index()
        else:
            return df

    def assign_bud(self, cc_stage_df, rp):
        if self.prev_bud_assignment_info is None:
            # If there are only two cells automatically assign bud to mother
            if len(rp) == 2:
                areas = [obj.area for obj in rp]
                IDs = [obj.label for obj in rp]
                min_area_idx = areas.index(min(areas))
                max_area_idx = areas.index(max(areas))
                budID = IDs[min_area_idx]
                mothID = IDs[max_area_idx]
                cc_stage_df.at[budID, 'Cell cycle stage'] = 'S'
                cc_stage_df.at[mothID, 'Cell cycle stage'] = 'S'
                cc_stage_df.at[mothID, '# of cycles'] = -1
                cc_stage_df.at[budID, '# of cycles'] = 0
                cc_stage_df.at[budID, 'Relative\'s ID'] = mothID
                cc_stage_df.at[mothID, 'Relative\'s ID'] = budID
                cc_stage_df.at[mothID, 'Relationship'] = 'mother'
                cc_stage_df.at[budID, 'Relationship'] = 'bud'
        else:
            for assign_click_coords in self.prev_bud_assignment_info:
                y_bud, x_bud = assign_click_coords['bud_coords']
                y_moth, x_moth = assign_click_coords['moth_coords']
                budID = self.lab[y_bud, x_bud]
                mothID = self.lab[y_moth, x_moth]
                if mothID != budID:
                    idx = cc_stage_df.index
                    if (budID in idx) and (mothID in idx):
                        cc_stage_df.at[budID, 'Cell cycle stage'] = 'S'
                        cc_stage_df.at[mothID, 'Cell cycle stage'] = 'S'
                        cc_stage_df.at[budID, '# of cycles'] = 0
                        cc_stage_df.at[budID, 'Relative\'s ID'] = mothID
                        cc_stage_df.at[mothID, 'Relative\'s ID'] = budID
                        cc_stage_df.at[mothID, 'Relationship'] = 'mother'
                        cc_stage_df.at[budID, 'Relationship'] = 'bud'
                elif mothID == budID:
                    bud_ccstage = cc_stage_df.at[budID, 'Cell cycle stage']
                    bud_ccnum = cc_stage_df.at[budID, '# of cycles']
                    if bud_ccstage == 'S' and bud_ccnum == 0:
                        mothID = cc_stage_df.at[budID, 'Relative\'s ID']
                        cc_stage_df.at[budID, 'Cell cycle stage'] = 'G1'
                        cc_stage_df.at[mothID, 'Cell cycle stage'] = 'G1'
                        cc_stage_df.at[budID, '# of cycles'] = -1
                        cc_stage_df.at[budID, 'Relative\'s ID'] = 0
                        cc_stage_df.at[mothID, 'Relative\'s ID'] = 0
                        cc_stage_df.at[mothID, 'Relationship'] = 'mother'
                        cc_stage_df.at[budID, 'Relationship'] = 'mother'
        return cc_stage_df

    def convexity_defects(self, img, eps_percent):
        img = img.astype(np.uint8)
        contours, hierarchy = cv2.findContours(img,2,1)
        cnt = contours[0]
        cnt = cv2.approxPolyDP(cnt,eps_percent*cv2.arcLength(cnt,True),True) # see https://www.programcreek.com/python/example/89457/cv22.convexityDefects
        hull = cv2.convexHull(cnt,returnPoints = False) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        defects = cv2.convexityDefects(cnt,hull) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        return cnt, defects

    def auto_separate_bud_ID(self, ID, lab, rp, max_ID, max_i=1,
                             enforce=False, eps_percent=0.01):
        lab_ID_bool = lab == ID
        cnt, defects = self.convexity_defects(lab_ID_bool, eps_percent)
        if defects is not None:
            if len(defects) == 2:
                if not enforce:
                    dist_watshed = segment(lab_ID_bool, None, merge=False)
                    num_obj_watshed = len(np.unique(dist_watshed))
                else:
                    num_obj_watshed = 0
                # Separate only if it was a separation also with watershed method
                if num_obj_watshed > 2 or enforce:
                    defects_points = [0]*len(defects)
                    for i, defect in enumerate(defects):
                        s,e,f,d = defect[0]
                        x,y = tuple(cnt[f][0])
                        defects_points[i] = (y,x)
                    (r0, c0), (r1, c1) = defects_points
                    rr, cc, _ = line_aa(r0, c0, r1, c1)
                    sep_bud_img = np.copy(lab_ID_bool)
                    sep_bud_img[rr, cc] = False
                    sep_bud_label = label(sep_bud_img, connectivity=2)
                    rp_sep = regionprops(sep_bud_label)
                    IDs_sep = [obj.label for obj in rp_sep]
                    areas = [obj.area for obj in rp_sep]
                    curr_ID_bud = IDs_sep[areas.index(min(areas))]
                    curr_ID_moth = IDs_sep[areas.index(max(areas))]
                    orig_sblab = np.copy(sep_bud_label)
                    # sep_bud_label = np.zeros_like(sep_bud_label)
                    # sep_bud_label[orig_sblab==curr_ID_moth] = ID
                    # sep_bud_label[orig_sblab==curr_ID_bud] = max(IDs)+max_i
                    sep_bud_label *= (max_ID+max_i)
                    temp_sep_bud_lab = sep_bud_label.copy()
                    for r, c in zip(rr, cc):
                        if lab_ID_bool[r, c]:
                            nearest_ID = self.nearest_nonzero(
                                                    sep_bud_label, r, c)
                            temp_sep_bud_lab[r,c] = nearest_ID
                    sep_bud_label = temp_sep_bud_lab
                    sep_bud_label_mask = sep_bud_label != 0
                    # plt.imshow_tk(sep_bud_label, dots_coords=np.asarray(defects_points))
                    lab[sep_bud_label_mask] = sep_bud_label[sep_bud_label_mask]
                    max_i += 1
                lab, _, _ = relabel_sequential(lab)
        return lab

    def auto_separate_bud(self, lab, rp, eps_percent=0.01):
        IDs = [obj.label for obj in rp]
        max_i = 1
        max_ID = max(IDs)
        for ID in IDs:
            lab = self.auto_separate_bud_ID(ID, lab, rp, max_i=max_i,
                                            eps_percent=eps_percent,
                                            enforce=False)
        return lab

    def auto_contour_parallel(self, app, yx):
        for a in (-45,45):
            lab, rp = self.auto_contour(app, start_yx=yx, alfa_dir=a,
                                                iter=300, draw=False)
        return lab, rp

    def full_IAroutine(self, img, app, ia):
        lowT = self.lowT
        highT = self.highT
        edge = self.edge_detector(img)
        thresh = self.thresholding(edge, lowT, highT)
        self.img = img
        self.edge = edge
        self.thresh = thresh
        lab = np.zeros(img.shape, int)
        rp = regionprops(lab)
        self.ta = threshold_otsu(edge)
        peaks = peak_local_max(edge, min_distance=self.peak_dist,
                               threshold_abs=self.ta,
                               exclude_border=self.exclude_border)
        start_t = time()
        for yx in peaks:
            for a in (-45,45):
                lab, rp = self.auto_contour(app, start_yx=yx, alfa_dir=a,
                                                    iter=300, draw=False)
        stop_t = time()
        print('Auto contour execution time: {0:.3f}'.format(stop_t-start_t))
        # lab = np.zeros(thresh.shape, int)
        # rp = regionprops(lab)
        self.cc_stage_df = self.init_cc_stage_df(rp)
        self.cc_stage_df = ia.assign_bud(self.cc_stage_df, rp)
        self.lab, _, _ = relabel_sequential(lab)
        self.rp = rp

    def unet_segmentation(self, img, app, ia):
        edge = self.edge_detector(img)
        self.img = img
        self.edge = edge
        start_t = time()
        img = equalize_adapthist(img)
        img = img*1.0
        print('Neural network is thinking...')
        if app.is_pc:
            pred = nn.prediction(img, is_pc=True)
        else:
            pred = nn.prediction(edge, is_pc=True)
        self.pred = pred
        thresh = nn.threshold(pred)
        lab = segment.segment(thresh, pred, min_distance=5).astype(int)
        lab = remove_small_objects(lab, min_size=20, connectivity=2)
        stop_t = time()
        print('Neural network execution time: {0:.3f}'.format(stop_t-start_t))
        self.lab, _, _ = relabel_sequential(lab)
        self.rp = regionprops(lab)
        self.cc_stage_df = self.init_cc_stage_df(self.rp)
        self.cc_stage_df = self.assign_bud(self.cc_stage_df, self.rp)
        IDs = [obj.label for obj in self.rp]
        ia.contours = self.find_contours(ia.lab, IDs, group=True)
        ia.auto_edge_img = np.zeros(ia.img.shape, bool)
        for cont in ia.contours:
            x = cont[:,1]
            y = cont[:,0]
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            ia.auto_edge_img[y, x] = True
        ia.manual_mask = np.zeros(img.shape, bool)
        ia.contour_plot = [[], []]
        ia.modified = False

matplotlib.use("TkAgg")

matplotlib.rcParams['keymap.back'] = ['q', 'backspace', 'MouseButton.BACK']
matplotlib.rcParams['keymap.forward'] = ['v', 'MouseButton.FORWARD']
matplotlib.rcParams['keymap.quit'] = []
matplotlib.rcParams['keymap.quit_all'] = []

"""Initialize app GUI parameters"""
dark_mode()
axcolor = '0.1'
slider_color = '0.2'
hover_color = '0.25'
presscolor = '0.35'
button_true_color = '0.4'
spF = 0.05  # spacing factor
wH = 0.03  # widgets' height
slW = 0.6  # sliders width
bW = 0.1  # buttons width

# Folder dialog
selected_path = folder_dialog(title = "Select folder containing valid experiments")
beyond_listdir_pos = beyond_listdir_pos(selected_path)
select_widget = select_exp_folder()
TIFFs_path = select_widget.run_widget(beyond_listdir_pos.all_exp_info,
                         title='Select experiment to segment',
                         label_txt='Select experiment to segment',
                         full_paths=beyond_listdir_pos.TIFFs_path,
                         showinexplorer_button=True)

"""Load data"""
app = app_GUI(TIFFs_path)
phc = app.phc
num_pos = len(phc)

"""Load neural network if needed"""
if app.is_pc:
    import neural_network as nn
    from segment import segment

"""Customize toolbar behaviour and initialize plots"""
home = NavigationToolbar2.home
def new_home(self, *args, **kwargs):
    try:
        app.ax_limits = deepcopy(ia.home_ax_limits)
        app.orig_ax_limits = deepcopy(ia.home_ax_limits)
    except:
        traceback.print_exc()
        pass
    home(self, *args, **kwargs)
    app.set_lims()
    app.fig.canvas.draw_idle()
NavigationToolbar2.home = new_home

release_zoom = NavigationToolbar2.release_zoom
def my_release_zoom(self, event):
    release_zoom(self, event)
    # Disconnect zoom to rect after having used it once
    self.zoom()
    self.push_current()
    app.zoom_on = False
    # self.release(event)
NavigationToolbar2.release_zoom = my_release_zoom

zoom = NavigationToolbar2.zoom
def my_zoom(self, *args):
    app.zoom_on = True
    zoom(self, *args)
NavigationToolbar2.zoom = my_zoom

app.init_plots()
sharp_img = app.preprocess_img_data(phc[app.p, app.s])
phc[app.p, app.s] = sharp_img
img = phc[app.p, app.s]


"""Initialize image analysis class"""
param = [None]*num_pos
ia = img_analysis(img)

"""Widgets' axes as [left, bottom, width, height]"""
ax_slice = plt.axes([0.1, 0.3, 0.8, 0.03])
ax_peak_dist = plt.axes([0.1, 0.2, 0.6, 0.03])
ax_exclude_bord = plt.axes([0.1, 0.15, 0.6, 0.03])
ax_next = plt.axes([0.62, 0.2, 0.05, 0.03])
ax_prev = plt.axes([0.67, 0.2, 0.05, 0.03])
# ax_locT = plt.axes([0.8, 0.2, 0.05, 0.03])
ax_enlarge_cells = plt.axes([0.9, 0.2, 0.05, 0.03])
ax_reduce_cells = plt.axes([0.6, 0.8, 0.05, 0.05])
ax_use_unet = plt.axes([0.4, 0.8, 0.05, 0.05])
ax_man_clos = plt.axes([0.7, 0.8, 0.05, 0.05])
ax_brush_mode_b = plt.axes([0.03, 0.8, 0.05, 0.05])
ax_keep_current_lab = plt.axes([0.012, 0.8, 0.05, 0.05])
ax_ccstage_radio = plt.axes([0.015, 0.8, 0.05, 0.05])
ax_save = plt.axes([0.001, 0.8, 0.05, 0.05])
ax_view_slices = plt.axes([0.001, 0.080, 0.05, 0.05])
ax_repeat_segm = plt.axes([0.007, 0.080, 0.05, 0.05])
ax_morph_IDs = plt.axes([0.125, 0.36, 0.007, 0.9])
ax_draw_bg_mask_on = plt.axes([0.125, 0.98756, 0.007, 0.9])
ax_reload_segm = plt.axes([0.007, 0.080, 0.008945, 0.05])
ax_reset_ccstage_df = plt.axes([0.007, 0.09512, 0.4581, 0.05])
ax_segment_z = plt.axes([0.007, 0.09512, 0.158744, 0.05])
ax_show_in_expl = plt.axes([0.007, 0.09512, 0.49864, 0.05])


"""Widgets"""
s_slice = Slider(ax_slice, 'Z-slice', 5, app.num_slices,
                    valinit=app.s,
                    valstep=1,
                    color=slider_color,
                    init_val_line_color=hover_color,
                    valfmt='%1.0f')
s_peak_dist = Slider(ax_peak_dist, 'Peak dist.', 0, 30,
                    valinit=ia.peak_dist,
                    valstep=1,
                    color=slider_color,
                    init_val_line_color=hover_color,
                    valfmt='%1.0f')
s_exclude_bord = Slider(ax_exclude_bord, 'Exclude border', 0, 100,
                    valinit=ia.exclude_border,
                    valstep=1,
                    color=slider_color,
                    init_val_line_color=hover_color,
                    valfmt='%1.0f')
# s_locT = Slider(ax_locT, '$\Delta$Local T.', -5, 20,
#                     valinit=10,
#                     valstep=1,
#                     orientation='vertical',
#                     color=slider_color,
#                     init_val_line_color=hover_color,
#                     valfmt='%1.0f')
next_b = Button(ax_next, 'Next pos.',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)
prev_b = Button(ax_prev, 'Prev. pos.',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)
enlarge_cells = Button(ax_enlarge_cells, 'Enlarge cells',
                     color=axcolor, hovercolor=hover_color,
                     presscolor=presscolor)
reduce_cells = Button(ax_reduce_cells, 'Reduce cells',
                     color=axcolor, hovercolor=hover_color,
                     presscolor=presscolor)
use_unet = Button(ax_use_unet, 'Neural network ACTIVE',
                     color=button_true_color, hovercolor=button_true_color,
                     presscolor=presscolor)
man_clos = Button(ax_man_clos, 'Switch to manual closing',
                     color=axcolor, hovercolor=hover_color,
                     presscolor=presscolor)
brush_mode_b = Button(ax_brush_mode_b, 'Brush mode OFF',
                     color=axcolor, hovercolor=hover_color,
                     presscolor=presscolor)
keep_current_lab_b = Button(ax_keep_current_lab, 'Freeze segmentation',
                     color=axcolor, hovercolor=hover_color,
                     presscolor=presscolor)
radio_b_ccStage = MyRadioButtons(ax_ccstage_radio,
                              ('Cells\' IDs',
                              'Cell Cycle stage',
                              'Cell Cycle INFO'),
                              active = 1,
                              activecolor = button_true_color,
                              orientation = 'horizontal',
                              size = 59,
                              circ_p_color = button_true_color)
repeat_segm_b = Button(ax_repeat_segm, 'Repeat segmentation',
                 color=axcolor, hovercolor=hover_color,
                 presscolor=presscolor)
reload_segm_b = Button(ax_reload_segm, 'Reload segmentation',
                 color=axcolor, hovercolor=hover_color,
                 presscolor=presscolor)
reset_ccstage_df_b = Button(ax_reset_ccstage_df, 'Reset cell cycle analysis',
                 color=axcolor, hovercolor=hover_color,
                 presscolor=presscolor)
save_b = Button(ax_save, 'Save and close',
                 color=axcolor, hovercolor=hover_color,
                 presscolor=presscolor)
view_slices_sl = Slider(ax_view_slices, 'View slice', 0, phc.shape[1],
                    valinit=app.s,
                    valstep=1,
                    orientation='horizontal',
                    color=slider_color,
                    init_val_line_color=hover_color,
                    valfmt='%1.0f')
morhp_ids_tb = TextBox(ax_morph_IDs, 'Enlarge\Reduce only cells IDs:',
                      initial='All',color=axcolor, hovercolor=hover_color)
draw_bg_mask_on_b = Button(ax_draw_bg_mask_on,
                     'Press \'d\' to activate objects\'s removal mode',
                     color=axcolor, hovercolor=axcolor,
                     presscolor=presscolor)
segment_this_z_b = Button(ax_segment_z,
                     'Segment this z-slice',
                     color=axcolor, hovercolor=hover_color,
                     presscolor=presscolor)
show_in_expl_b = Button(ax_show_in_expl,
                     'Show in Explorer',
                     color=axcolor, hovercolor=hover_color,
                     presscolor=presscolor)

app.view_slices_sl = view_slices_sl

"""Functions"""
def analyse_img(img):
    print('Thinking...')
    if app.use_unet:
        ia.unet_segmentation(img, app, ia)
    else:
        ia.full_IAroutine(img, app, ia)
    app.update_ALLplots(ia)

def load_param(param):
    ia = param[app.p]
    return ia

def store_param(ia):
    if app.p >= 0:
        param[app.p] = deepcopy(ia)
    return param

def load_param_from_folder(app):
    global ia
    ia.lab = app.segm_npy_done[app.p]
    ia.rp = regionprops(ia.lab)
    if app.cc_stages[app.p] is not None:
        ia.cc_stage_df = app.cc_stages[app.p]
        ia.prev_cc_stage_df = app.cc_stages[app.p]
    if app.slices_used[app.p] is not None:
        app.s = app.slices_used[app.p]
        ia.slice_used = app.s
    ia.img = app.all_phc[app.p, app.s]
    if app.masks[app.p] is not None:
        ia.manual_mask = app.masks[app.p]
    else:
        ia.manual_mask = np.zeros(ia.img.shape, bool)
    ia.edge = ia.edge_detector(ia.img)
    ia.contour_plot = [[], []]
    app.pos_txt._text = (f'Current position = {app.p+1}/{num_pos}   '
                         f'({app.pos_foldernames[app.p]})')
    ia.modified = False
    IDs = [obj.label for obj in ia.rp]
    ia.contours = ia.find_contours(ia.lab, IDs, group=True)
    ia.reset_auto_edge_img(ia.contours)
    ia.enlarg_first_call = True
    ia.reduce_first_call = True
    ia.prev_bud_assignment_info = None
    param = store_param(ia)
    return param

def store_app_param(ia):
    app.segm_npy_done[app.p] = ia.lab
    app.slices_used[app.p] = int(s_slice.val)
    app.cc_stages[app.p] = ia.cc_stage_df


"""Widgets' functions"""
def next_f(event):
    global param, ia, param
    app.reset_view = True
    app.brush_mode_on = False
    brush_mode_button(False)
    if app.segm_npy_done[app.p] is not None:
        param = store_param(ia)
        store_app_param(ia)
    # last frame reached
    if app.p+1 >= num_pos:
        print('You reached the last position!')
        ia.slice_used = app.s
        param = store_param(ia)
        tk.messagebox.showinfo('Last position', 'This is the last position!')
        p = 0
        if app.segm_npy_done[app.p] is not None:
            app.p = p
            param = load_param_from_folder(app)
            app.update_ALLplots(ia)
        else:
            app.save_pos(param, app.pos_paths[app.p], app.p)
    # Next frame was already segmented in a previous session. Load data from HDD
    elif app.segm_npy_done[app.p+1] is not None:
        ia.slice_used = app.s
        param = store_param(ia)
        app.save_pos(param, app.pos_paths[app.p], app.p)
        app.p += 1
        app.all_phc[app.p, app.s] = app.preprocess_img_data(app.all_phc[
                                                                  app.p, app.s])
        param = load_param_from_folder(app)
        app.update_ALLplots(ia)
    # Next frame was never segmented
    elif app.p+1 < num_pos and param[app.p+1] is None:
        ia.slice_used = app.s
        param = store_param(ia)
        app.save_pos(param, app.pos_paths[app.p], app.p)
        app.p += 1
        app.init_attr()
        phc[app.p, app.s] = app.preprocess_img_data(phc[app.p, app.s])
        img = phc[app.p, app.s]
        ia.init_attr(img)
        app.pos_txt._text = (f'Current position = {app.p+1}/{num_pos}   '
                             f'({app.pos_foldernames[app.p]})')
        analyse_img(img)
    # Next frame was already segmented within this session. Load data
    elif app.p+1 and param[app.p+1] is not None:
        param = store_param(ia)
        # app.save_pos(param, app.pos_paths[app.p], app.p)
        app.p += 1
        ia = load_param(param)
        app.pos_txt._text = (f'Current position = {app.p+1}/{num_pos}   '
                             f'({app.pos_foldernames[app.p]})')
        app.update_ALLplots(ia)
    if ia.clos_mode == 'Auto closing':
        man_clos_cb(None)
    s_slice.set_val(app.s, silent=True)
    view_slices_sl.set_val(app.s, silent=True)
    app.fig.canvas.draw_idle()
    app.connect_axes_cb()
    app.set_orig_lims()
    app.prev_states = []
    app.already_approx_IDs = []
    morhp_ids_tb.text_disp._text = 'All'
    app.store_state(ia)

def prev_f(event):
    global ia, param
    app.reset_view = True
    app.brush_mode_on = False
    brush_mode_button(False)
    if app.segm_npy_done[app.p] is not None and ia.modified:
        param = store_param(ia)
        store_app_param(ia)
    # Next frame was already segmented in a previous session. Load data from HDD
    if app.segm_npy_done[app.p-1] is not None and app.p-1 >= 0:
        app.p -= 1
        param = load_param_from_folder(app)
        app.update_ALLplots(ia)
    elif app.p-1 >= 0:
        app.p -= 1
        ia = load_param(param)
        s_slice.set_val(ia.slice_used, silent=True)
        app.pos_txt._text = (f'Current position = {app.p+1}/{num_pos}   '
                             f'({app.pos_foldernames[app.p]})')
        app.update_ALLplots(ia)
    else:
        print('You reached the first frame!')
        p = num_pos-1
        if app.segm_npy_done[p] is not None:
            app.p = p
            param = load_param_from_folder(app)
            app.update_ALLplots(ia)
    s_slice.set_val(app.s, silent=True)
    view_slices_sl.set_val(app.s, silent=True)
    app.prev_states = []
    app.connect_axes_cb()
    app.set_orig_lims()
    app.store_state(ia)

def check_cca_df_ids_segmIDs(labelled_img, cca_df):
    rp = regionprops(labelled_img)
    segmIDs = [obj.label for obj in rp]
    ccaIDs = cca_df.index.to_list()
    ccaIDs_equal_segmIDs = segmIDs == ccaIDs
    return ccaIDs_equal_segmIDs, segmIDs, ccaIDs

def check_cca_df_bud_mother(cca_df):
    num_cycles = cca_df['# of cycles']
    unique_num_cycles = num_cycles.unique()
    cc_stages = cca_df['# of cycles']
    if len(num_cycles) > 1 and len(unique_num_cycles)==1:
        num_cycles_issue = True
    else:
        num_cycles_issue = False
    return num_cycles_issue


def next_cb(event):
    if app.is_lab_frozen:
        tk.messagebox.showwarning('Frozen segmentation!', 'You still have a '
            'frozen segmentation to release.\n Before going to next position, '
            'you have to release the segmentation.')
    else:
        ccaIDs_equal_segmIDs, segmIDs, ccaIDs = check_cca_df_ids_segmIDs(ia.lab,
                                                                 ia.cc_stage_df)
        num_cycles_issue = check_cca_df_bud_mother(ia.cc_stage_df)
        if not ccaIDs_equal_segmIDs:
            tk.messagebox.showwarning('Cell cycle analysis problem!',
                f'The IDs from the cell cycle analysis are {ccaIDs}\n'
                f'while the IDs of the segmentation are {segmIDs}\n'
                'Press "Reset cell cycle analysis" before going to next position.')
        # elif num_cycles_issue:
        #     tk.messagebox.showwarning('Cell cycle analysis problem!',
        #         f'All the cells have the same number of cycles!'
        #         'Right click on the mother cell and assign the bud again.')
        else:
            next_f(event)

def prev_cb(event):
    if app.is_lab_frozen:
        tk.messagebox.showwarning('Frozen segmentation!', 'You still have a '
            'frozen segmentation to release.\n Before going to prev position, '
            'you have to release the segmentation.')
    else:
        prev_f(event)

def update_segm(val):
    ia.locTval = s_peak_dist.val - s_locT.val
    thresh = ia.thresholding(ia.edge, s_peak_dist.val, s_exclude_bord.val)
    ia.repeat_manual_func(thresh)
    lab, rp = ia.segmentation(ia.thresh)
    app.update_ax0_plot(ia, ia.img)
    app.update_ax2_plot(ia, ia.thresh)
    app.update_ax1_plot(lab, rp, ia)

def morph_IDs_invalid_entry_warning():
    tk.messagebox.showwarning('Invalid entry',
            'Invalid entry for IDs to enlarge/reduce.\n'
            'Valid entries are e.g. "All" to apply to all cells,\n'
            '"1,4" to apply only to 1 and 4 or 1-4 to apply from 1 to 4.')

def morph_IDs_loop_single(IDs_to_apply, IDs_li):
    for ID in IDs_to_apply:
        try:
            IDs_li.append(int(ID))
        except:
            morph_IDs_invalid_entry_warning()
            IDs_li = []
            break
    return IDs_li

def morph_IDs_loop_multi(IDs_to_apply, IDs_li):
    invalid_entry = False
    for ID in IDs_to_apply:
        try:
            IDs_li.append(int(ID))
        except:
            try:
                IDs_range = ID.split('-')
                IDs_li_range = morph_IDs_loop_single(IDs_range, [])
                IDs_li.append(IDs_li_range)
            except:
                morph_IDs_invalid_entry_warning()
                IDs_li = []
                invalid_entry = True
                break
    return IDs_li, invalid_entry

def mask_lab_morph(lab, IDs_li, invalid_entry, is_range):
    lab_not_apply = []
    lab_not_apply_idx = []
    lab_mask = np.ones(lab.shape, bool)
    if not invalid_entry and IDs_li:
        lab_mask = np.zeros(lab.shape, bool)
        if is_range:
            IDs_li = range(IDs_li[0], IDs_li[1]+1)
        for item in IDs_li:
            if isinstance(item, list):
                item = range(item[0], item[1]+1)
                for ID in item:
                    ID_mask = np.logical_and(lab != ID, lab != 0)
                    lab_not_apply.append(lab[ID_mask])
                    lab_not_apply_idx.append(ID_mask)
                    lab_mask[~ID_mask] = True
            else:
                ID_mask = np.logical_and(lab != item, lab != 0)
                lab_not_apply.append(lab[ID_mask])
                lab_not_apply_idx.append(ID_mask)
                lab_mask[~ID_mask] = True
    lab[~lab_mask] = 0
    return lab, lab_not_apply, lab_not_apply_idx

def apply_morph_cells(event):
    if app.is_undo or app.is_redo:
        app.store_state(ia)
    # Initialize on first call
    if event.inaxes == ax_enlarge_cells:
        # morph_func = dilation
        morph_func = expand_labels
        if ia.enlarg_first_call:
            ia.orig_lab = ia.lab.copy()
            ia.enlarg_first_call = False
            ia.reduce_first_call = True
            ia.morph_disk_radius = 0
    if event.inaxes == ax_reduce_cells:
        morph_func = erosion
        if ia.reduce_first_call:
            ia.orig_lab = ia.lab.copy()
            ia.reduce_first_call = False
            ia.enlarg_first_call = True
            ia.morph_disk_radius = 0
    ia.morph_disk_radius += 1
    # Mask only IDs requested to enlarge/reduce
    IDs_txt = morhp_ids_tb.text_disp._text
    IDs_to_apply = IDs_txt.split(',')
    is_range = False
    invalid_entry = False
    IDs_li = []
    if len(IDs_to_apply) == 1:
        if IDs_to_apply[0] != 'All':
            try:
                ID = int(IDs_to_apply[0])
                IDs_li.append(ID)
            except:
                try:
                    IDs_to_apply = IDs_txt.split('-')
                    is_range = True
                    IDs_li = morph_IDs_loop_single(IDs_to_apply, IDs_li)
                except:
                    morph_IDs_invalid_entry_warning()
                    invalid_entry = True
    else:
        IDs_li, invalid_entry = morph_IDs_loop_multi(IDs_to_apply, IDs_li)
        is_range = False
    ia.orig_lab, lab_not_apply, lab_not_apply_idx = mask_lab_morph(
                                                           ia.orig_lab,
                                                           IDs_li,
                                                           invalid_entry,
                                                           is_range)
    # Apply enlarge/reduce function
    if morph_func == expand_labels:
        ia.lab = morph_func(ia.orig_lab, distance=ia.morph_disk_radius)
    else:
        ia.lab = morph_func(ia.orig_lab, selem=disk(ia.morph_disk_radius))
    # Reinsert not dilated/eroded cells
    for ID_mask_not_apply, idx in zip(lab_not_apply, lab_not_apply_idx):
        ia.lab[idx] = ID_mask_not_apply
        ia.orig_lab[idx] = ID_mask_not_apply
    ia.rp = regionprops(ia.lab)
    IDs = [obj.label for obj in ia.rp]
    ia.contours = ia.find_contours(ia.lab, IDs, group=True)
    app.update_ax0_plot(ia, ia.img)
    app.update_ax1_plot(ia.lab, ia.rp, ia)
    app.store_state(ia)

def morph_IDs_on_submit(text):
    ia.orig_lab = ia.lab.copy()
    ia.morph_disk_radius = 1

def locT_cb(val):
    pass


def undo_del_t_cb(event):
    del ia.del_yx_thresh[-1]
    func_found = False
    for i, func in enumerate(reversed(ia.li_manual_func)):
        if func == 'Del. yx thresh':
            j = len(ia.li_manual_func) - 1 - i
            func_found = True
            break
    if func_found:
        del ia.li_manual_func[j]
    update_segm(None)


def brush_mode_cb(event):
    if app.brush_mode_on:
        brush_mode_b.color = axcolor
        brush_mode_b.hovercolor = hover_color
        brush_mode_b.label._text = 'Brush mode OFF'
        brush_mode_b.ax.set_facecolor(axcolor)
        (app.fig.canvas).draw_idle()
        app.brush_mode_on = False
        app.update_ax1_plot(ia.lab, ia.rp, ia)
    else:
        brush_mode_b.color = button_true_color
        brush_mode_b.hovercolor = button_true_color
        brush_mode_b.label._text = 'Brush mode ON'
        brush_mode_b.ax.set_facecolor(button_true_color)
        app.brush_mode_on = True
        app.brush_size = 1
        app.update_ax1_plot(ia.lab, ia.rp, ia)

def brush_mode_button(value):
    if value:
        brush_mode_b.color = button_true_color
        brush_mode_b.hovercolor = button_true_color
        brush_mode_b.label._text = 'Brush mode ON'
        brush_mode_b.ax.set_facecolor(button_true_color)
        (app.fig.canvas).draw_idle()
    else:
        brush_mode_b.color = axcolor
        brush_mode_b.hovercolor = hover_color
        brush_mode_b.label._text = 'Brush mode OFF'
        brush_mode_b.ax.set_facecolor(axcolor)
        (app.fig.canvas).draw_idle()

def switch_use_unet_button(value):
    if value:
        use_unet.color = button_true_color
        use_unet.hovercolor = button_true_color
        use_unet.label._text = 'Neural network ACTIVE'
        use_unet.ax.set_facecolor(button_true_color)
        (app.fig.canvas).draw_idle()
    else:
        use_unet.color = axcolor
        use_unet.hovercolor = hover_color
        use_unet.label._text = 'Use neural network'
        use_unet.ax.set_facecolor(axcolor)
        (app.fig.canvas).draw_idle()


def radio_b_ccStage_cb(label):
    if label == 'Cells\' IDs':
        app.display_ccStage = 'IDs'
    elif label == 'Cell Cycle INFO':
        app.display_ccStage = 'All info'
    else:
        app.display_ccStage = 'Only stage'
    text_label_centroid(ia.rp, app.ax[0], 12, 'semibold', 'center', 'center',
                        cc_stage_frame=ia.cc_stage_df,
                        display_ccStage=app.display_ccStage, color='r',
                        clear=True)
    app.fig.canvas.draw_idle()

def s_slice_cb(val):
    app.s = int(val)
    ia.slice_used = int(val)
    view_slices_sl.set_val(val, silent=True)
    app.phc[app.p, app.s] = app.preprocess_img_data(app.phc[app.p, app.s])
    img = app.phc[app.p, app.s]
    ia.init_attr(img)
    analyse_img(img)
    ia.modified = True

def save_cb(event):
    global ia, param
    save_current = tk.messagebox.askyesno('Save current position',
                    'Do you want to save currently displayed position?')
    print('Saving...')
    if save_current:
        ia.slice_used = app.s
        param = store_param(ia)
    for i, pos_path in enumerate(app.pos_paths):
        if param[i] is not None:
            app.save_pos(param, pos_path, i)
    print('Saved!')
    app.fig.canvas.mpl_disconnect(cid_close)
    plt.close('all')

def man_clos_cb(event):
    if ia.clos_mode == 'Manual closing':
        man_clos.color = axcolor
        man_clos.hovercolor = hover_color
        man_clos.label._text = 'Switch to manual closing'
        man_clos.ax.set_facecolor(axcolor)
        (app.fig.canvas).draw_idle()
        ia.clos_mode = 'Auto closing'
    elif ia.clos_mode == 'Auto closing':
        man_clos.color = button_true_color
        man_clos.hovercolor = button_true_color
        man_clos.label._text = 'Man. clos. ACTIVE'
        man_clos.ax.set_facecolor(button_true_color)
        (app.fig.canvas).draw_idle()
        ia.clos_mode = 'Manual closing'

def view_slice(event):
    img = phc[app.p, int(view_slices_sl.val)]
    app.ax[0].clear()
    app.update_ax0_plot(ia, img)
    app.fig.canvas.draw_idle()

def repeat_segmentation_cb(event):
    app.s = auto_select_slice(app.phc[app.p], init_slice=0).slice
    view_slices_sl.val = float(app.s)
    s_slice.val = float(app.s)
    img = app.phc[app.p, app.s]
    ia.init_attr(img)
    analyse_img(img)
    ia.modified = True

def update_peaks_dist(val):
    ia.peak_dist = int(val)
    peaks = peak_local_max(ia.edge, min_distance=ia.peak_dist,
                           threshold_abs=ia.ta,
                           exclude_border=ia.exclude_border)
    app.ax[2].plot(peaks[:,1], peaks[:,0], 'r.')

def update_peaks_bord(val):
    ia.exclude_border = int(val)
    peaks = peak_local_max(ia.edge, min_distance=ia.peak_dist,
                           threshold_abs=ia.ta,
                           exclude_border=ia.exclude_border)
    app.ax[2].plot(peaks[:,1], peaks[:,0], 'r.')

def use_unet_cb(event):
    global nn, segment
    if app.use_unet:
        use_unet.color = axcolor
        use_unet.hovercolor = hover_color
        use_unet.label._text = 'Use neural network'
        use_unet.ax.set_facecolor(axcolor)
        (app.fig.canvas).draw_idle()
        app.use_unet = False
        analyse_img(ia.img)
    else:
        from YeaZ.unet import neural_network as nn
        from YeaZ.unet import segment
        use_unet.color = button_true_color
        use_unet.hovercolor = button_true_color
        use_unet.label._text = 'Neural network ACTIVE'
        use_unet.ax.set_facecolor(button_true_color)
        (app.fig.canvas).draw_idle()
        app.use_unet = True
        analyse_img(ia.img)

def draw_bg_mask_on_cb(event):
    if ia.draw_bg_mask_on:
        draw_bg_mask_on_b.color = axcolor
        draw_bg_mask_on_b.label._text = 'Press \'d\' to activate objects\'s removal mode'
        draw_bg_mask_on_b.ax.set_facecolor(axcolor)
        (app.fig.canvas).draw_idle()
        ia.draw_bg_mask_on = False
    else:
        draw_bg_mask_on_b.color = button_true_color
        draw_bg_mask_on_b.label._text = 'Objects\'s removal mode ACTIVE'
        draw_bg_mask_on_b.ax.set_facecolor(button_true_color)
        (app.fig.canvas).draw_idle()
        ia.draw_bg_mask_on = True

def reload_segmentation_cb(event):
    global param
    if app.is_undo or app.is_redo:
        app.store_state(ia)
    img = app.all_phc[app.p, app.s]
    ia.init_attr(img)
    orig_segm_npy = app.orig_segm_npy[app.p].copy()
    app.segm_npy_done[app.p] = orig_segm_npy
    param = load_param_from_folder(app)
    app.update_ALLplots(ia)
    app.store_state(ia)

def reset_ccstage_df_cb(event):
    if app.is_undo or app.is_redo:
        app.store_state(ia)
    rp = regionprops(ia.lab)
    ia.cc_stage_df = ia.init_cc_stage_df(rp)
    ia.cc_stage_df = ia.assign_bud(ia.cc_stage_df, rp)
    app.update_ALLplots(ia)
    app.store_state(ia)

def keep_release_current_lab_cb(event):
    if app.is_undo or app.is_redo:
        app.store_state(ia)
    if not app.is_lab_frozen and event.button == 1:
        ia.frozen_lab = ia.lab.copy()
        app.frozen_s = app.s
        ia.frozen_edge = ia.edge.copy()
        ia.frozen_img = ia.img.copy()
        ia.frozen_prev_bud_assignment_info = deepcopy(ia.prev_bud_assignment_info)
        keep_current_lab_b.color = button_true_color
        keep_current_lab_b.hovercolor = button_true_color
        keep_current_lab_b.label._text = 'Release frozen segment.'
        keep_current_lab_b.ax.set_facecolor(button_true_color)
        (app.fig.canvas).draw_idle()
        app.is_lab_frozen = True
    elif event.button == 1:
        # Relabel new segmentation to avoid same IDs of the frozen segmentation
        max_ID = ia.frozen_lab.max()+ia.lab.max()
        lab_mask = ia.frozen_lab > 0
        # Add frozen segmentation on top of the new one
        ia.lab[lab_mask] = ia.frozen_lab[lab_mask]+max_ID
        ia.lab = app.relabel_and_fill_lab(ia.lab)
        # Update all the attributes of the new segmentation
        ia.rp = regionprops(ia.lab)
        ia.cc_stage_df = ia.init_cc_stage_df(ia.rp)
        ia.prev_bud_assignment_info = deepcopy(ia.frozen_prev_bud_assignment_info)
        ia.cc_stage_df = ia.assign_bud(ia.cc_stage_df, ia.rp)
        IDs = [obj.label for obj in ia.rp]
        ia.contours = ia.find_contours(ia.lab, IDs, group=True)
        ia.reset_auto_edge_img(ia.contours)
        ia.edge = ia.frozen_edge.copy()
        ia.img = ia.frozen_img.copy()
        ia.contour_plot = [[], []]
        app.s = app.frozen_s
        app.view_slices_sl.set_val(float(app.frozen_s), silent=True)
        s_slice.set_val(float(app.frozen_s), silent=True)
        keep_current_lab_b.color = axcolor
        keep_current_lab_b.hovercolor = hover_color
        keep_current_lab_b.label._text = 'Freeze segmentation'
        keep_current_lab_b.ax.set_facecolor(axcolor)
        (app.fig.canvas).draw_idle()
        app.update_ax0_plot(ia, ia.img)
        app.update_ax2_plot(ia)
        app.update_ax1_plot(ia.lab, ia.rp, ia)
        app.is_lab_frozen = False
    elif event.button != 1:
        keep_current_lab_b.color = axcolor
        keep_current_lab_b.hovercolor = hover_color
        keep_current_lab_b.label._text = 'Freeze segmentation'
        keep_current_lab_b.ax.set_facecolor(axcolor)
        (app.fig.canvas).draw_idle()
        app.is_lab_frozen = False
    app.store_state(ia)

def segment_this_z_cb(event):
    s_slice.set_val(view_slices_sl.val, silent=True)
    s_slice.set_val(view_slices_sl.val)

def show_in_expl_cb(event):
    pos_path = app.pos_paths[app.p]
    subprocess.Popen(f'explorer "{os.path.normpath(pos_path)}"')

# Connect widgets to callbacks
next_b.on_clicked(next_cb)
prev_b.on_clicked(prev_cb)
repeat_segm_b.on_clicked(repeat_segmentation_cb)
reload_segm_b.on_clicked(reload_segmentation_cb)
s_peak_dist.on_changed(update_peaks_dist)
s_exclude_bord.on_changed(update_peaks_bord)
reduce_cells.on_clicked(apply_morph_cells)
enlarge_cells.on_clicked(apply_morph_cells)
morhp_ids_tb.on_submit(morph_IDs_on_submit)
# s_locT.on_changed(locT_cb)

use_unet.on_clicked(use_unet_cb)
# undo_del_t.on_clicked(undo_del_t_cb)
man_clos.on_clicked(man_clos_cb)
brush_mode_b.on_clicked(brush_mode_cb)
radio_b_ccStage.on_clicked(radio_b_ccStage_cb)
s_slice.on_changed(s_slice_cb)
save_b.on_clicked(save_cb)
view_slices_sl.on_changed(view_slice)
reset_ccstage_df_b.on_clicked(reset_ccstage_df_cb)
keep_current_lab_b.on_clicked(keep_release_current_lab_cb)
segment_this_z_b.on_clicked(segment_this_z_cb)
show_in_expl_b.on_clicked(show_in_expl_cb)

"""Canvas events functions"""
def resize_widgets(event):
    # [left, bottom, width, height]
    ax = app.ax
    ax0_l, ax0_b, ax0_r, ax0_t = ax[0].get_position().get_points().flatten()
    ax1_l, ax1_b, ax1_r, ax1_t = ax[1].get_position().get_points().flatten()
    ax2_l, ax2_b, ax2_r, ax2_t = ax[2].get_position().get_points().flatten()
    slW = ax2_r-bW-ax0_l-spF
    ax_peak_dist.set_position([ax0_l, ax0_b-2*spF-wH, slW, wH])
    ax_exclude_bord.set_position([ax0_l, ax0_b-2*spF-2*wH-(spF/2), slW, wH])
    ax_show_in_expl.set_position([ax2_r-bW, ax0_b-(2*spF)+(spF/3), bW, wH])
    ax_prev.set_position([ax2_r-bW, ax0_b-2*spF-wH, bW/2, wH])
    ax_next.set_position([ax2_r-bW/2, ax0_b-2*spF-wH, bW/2, wH])
    # ax_locT.set_position([ax2_r+(spF/3), ax2_b, 0.02, ax2_t-ax2_b])
    a1_c = ax1_l+((ax1_r-ax1_l)/2)
    ax_reduce_cells.set_position([a1_c-bW-(spF/11), ax1_b-(spF/11)-wH, bW, wH])
    ax_enlarge_cells.set_position([a1_c+(spF/11), ax1_b-(spF/11)-wH, bW, wH])
    _, ax_enl_b, ax_enl_r, _ = ax_enlarge_cells.get_position().get_points().flatten()
    ax_morph_IDs.set_position([ax_enl_r-bW, ax_enl_b-(spF/9)-wH, bW, wH])
    ax_use_unet.set_position([ax1_l, ax1_t+(spF/3), bW, wH])
    ax_man_clos.set_position([ax2_l, ax2_t+(spF/3), bW, wH])
    ax_keep_current_lab.set_position([ax1_r-bW, ax1_t+(spF/3), bW, wH])
    udtL, udtb, udtR, udtT = ax_man_clos.get_position().get_points().flatten()
    uaeW = ax2_r-udtR-(spF/11)
    ax_brush_mode_b.set_position([ax2_r-bW, ax2_t+(spF/3), bW, wH])
    ax_ccstage_radio.set_position([ax0_l, ax0_t+(spF/11), ax0_r-ax0_l, wH])
    ax_radio_l, _, _, ax_radio_t = ax_ccstage_radio.get_position().get_points().flatten()
    ax_draw_bg_mask_on.set_position([ax_radio_l, ax_radio_t+(spF/7), bW*2, wH])
    ax_slice.set_position([ax2_l, ax2_b-(spF/11)-wH, ax2_r-ax2_l, wH])
    _, ax_next_b, _, _ = ax_next.get_position().get_points().flatten()
    ax_repeat_segm_b = ax_next_b-wH-(spF/3)
    ax_repeat_segm.set_position([ax2_r-bW, ax_repeat_segm_b, bW, wH])
    ax_reload_segm_b = ax_repeat_segm_b-wH-(spF/3)
    ax_reload_segm.set_position([ax2_r-bW, ax_reload_segm_b, bW, wH])
    ax_reset_ccstage_b = ax_reload_segm_b-wH-(spF/3)
    ax_reset_ccstage_df.set_position([ax2_r-bW, ax_reset_ccstage_b, bW, wH])
    ax_save_b = ax_reset_ccstage_b-wH-(spF/3)
    ax_save.set_position([ax2_r-bW, ax_save_b, bW, wH])
    ax_view_slices.set_position([ax0_l, ax0_b-(spF/11)-wH, ax0_r-ax0_l, wH])
    _, ax_view_z_b, _, _ = ax_view_slices.get_position().get_points().flatten()
    ax_segment_z.set_position([ax0_r-bW, ax_view_z_b-(spF/11)-wH, bW, wH])
    app.connect_axes_cb()
    for axes in app.ax:
        on_xlim_changed(axes)
        on_ylim_changed(axes)

def key_down(event):
    global ia
    key = event.key
    if key == 'right':
        if app.key_mode == 'Z-slice':
            s_slice.set_val(s_slice.val+1)
        elif app.key_mode == 'view_slice':
            view_slices_sl.set_val(view_slices_sl.val+1)
        else:
            next_cb(None)
    elif key == 'left':
        if app.key_mode == 'Z-slice':
            s_slice.set_val(s_slice.val-1)
        elif app.key_mode == 'view_slice':
            view_slices_sl.set_val(view_slices_sl.val-1)
        else:
            prev_cb(None)
    elif key == 'ctrl+p':
        print(f'Current dataframe:\n {ia.cc_stage_df}')
        if param[app.p] is not None:
            print(f'Stored dataframe:\n {param[app.p].cc_stage_df}')
        print(app.ax_limits)
    elif key == 'b':
        ia.sep_bud = True
    elif key == 'm':
        ia.set_moth = True
    elif key == 'c':
        man_clos_cb(None)
    elif key == 'd':
        draw_bg_mask_on_cb(None)
    elif key == 'ctrl+z':
        app.is_undo = True
        if app.count_states-1 >= 0:
            app.count_states -= 1
            ia = app.get_state(ia, app.count_states)
            app.update_ALLplots(ia)
        else:
            print('There are no previous states to restore!')
    elif key == 'ctrl+y':
        app.is_redo = True
        if app.count_states+1 < len(app.prev_states):
            app.count_states += 1
            ia = app.get_state(ia, app.count_states)
            app.update_ALLplots(ia)
        else:
            print('No next states to restore!')
    elif key == 'c':
        app.do_cut = True
    elif key == 'a':
        app.do_approx = True
    elif key == 'x' and app.brush_mode_on:
        # Switch eraser mode on or off
        app.eraser_on = not app.eraser_on
        app.draw_brush_circle_cursor(ia, event.x, event.y)
    elif key == 'up' and app.brush_mode_on:
        app.brush_size += 1
        app.draw_brush_circle_cursor(ia, event.x, event.y)
    elif key == 'down' and app.brush_mode_on:
        app.brush_size -= 1
        app.draw_brush_circle_cursor(ia, event.x, event.y)
    elif key == 'shift':
        app.scroll_zoom = True
    elif key == 'control':
        app.select_ID_on = True
    elif event.key == 'escape':
        app.selected_IDs = None
        app.update_ax0_plot(ia, ia.img)
        app.update_ax1_plot(ia.lab, ia.rp, ia)
    elif event.key == 'e':
        brush_mode_cb(None)


def key_up(event):
    key = event.key
    if key == 'b':
        ia.sep_bud = False
    elif key == 'm':
        ia.set_moth = False
    elif key == 'c':
        app.do_cut = False
    elif key == 'a':
        app.do_approx = False
    elif key == 'shift':
        app.scroll_zoom = False
    elif key == 'control':
        app.select_ID_on = False

def mouse_down(event):
    right_click = event.button == 3
    left_click = event.button == 1
    wheel_click = event.button == 2
    ax0_click = event.inaxes == app.ax[0]
    ax1_click = event.inaxes == app.ax[1]
    ax2_click = event.inaxes == app.ax[2]
    left_ax1_click = left_click and ax1_click
    not_ax = not any([ax0_click, ax1_click, ax2_click])
    app.connect_axes_cb()
    if event.xdata:
        xd = int(round(event.xdata))
        yd = int(round(event.ydata))
    # Zoom with double click on ax2
    if left_click and ax2_click and event.dblclick:
        # app.fig.canvas.toolbar.zoom()
        xlim_left, xlim_right = app.ax[2].get_xlim()
        ylim_bottom, ylim_top = app.ax[2].get_ylim()
        xrange = xlim_right-xlim_left
        yrange = ylim_top-ylim_bottom
        zoom_factor = 6
        app.ax[2].set_xlim((xd-xrange/zoom_factor, xd+xrange/zoom_factor))
        app.ax[2].set_ylim((yd-yrange/zoom_factor, yd+yrange/zoom_factor))
        app.ax[2].relim()
        app.fig.canvas.draw_idle()
    # Manual correction: replace cell with hull contour with double_click on ID
    if left_click and ax1_click and event.dblclick and not app.select_ID_on:
        ID = ia.lab[yd, xd]
        if ID != 0:
            if app.is_undo or app.is_redo:
                app.store_state(ia)
            labels_onlyID = np.zeros(ia.lab.shape, bool)
            labels_onlyID[ia.lab == ID] = True
            if app.do_approx:
                ia.lab[ia.lab==ID] = 0
                approx_lab = app.approx_contour(labels_onlyID, method='approx')
            else:
                approx_lab = app.approx_contour(labels_onlyID, method='hull')
                app.already_approx_IDs.append(ID)
            ia.lab[approx_lab>0] = ID
            ia.lab = app.relabel_and_fill_lab(ia.lab)
            ia.rp = regionprops(ia.lab)
            IDs = [obj.label for obj in ia.rp]
            ia.contours = ia.find_contours(ia.lab, IDs, group=True)
            ia.reset_auto_edge_img(ia.contours)
            app.update_ax0_plot(ia, ia.img)
            app.update_ax2_plot(ia)
            app.update_ax1_plot(ia.lab, ia.rp, ia)
            ia.modified = True
            app.store_state(ia)
    # Select label with alt+left_click on labels or phase contrast
    if left_click and ax1_click and not event.dblclick and app.select_ID_on:
        clicked_ID = ia.lab[yd, xd]
        if clicked_ID != 0:
            # Allow a maximum of two selected IDs
            if app.selected_IDs is None:
                app.selected_IDs = [clicked_ID]
            elif len(app.selected_IDs) == 1:
                app.selected_IDs.append(clicked_ID)
            else:
                app.selected_IDs = [clicked_ID]
        text_label_centroid(ia.rp, app.ax[0], 12, 'semibold', 'center', 'center',
                            cc_stage_frame=ia.cc_stage_df,
                            display_ccStage=app.display_ccStage, color='r',
                            clear=True, selected_IDs=app.selected_IDs)
        app.update_ax1_plot(ia.lab, ia.rp, ia, draw=False)
        app.fig.canvas.draw_idle()
    # Freely paint/erase with the brush tool
    if left_ax1_click and not app.select_ID_on and app.brush_mode_on:
        if app.is_undo or app.is_redo:
            app.store_state(ia)
        app.new_ID = ia.lab.max()+1
        app.brush_mask = np.zeros(ia.lab.shape, bool)
        app.apply_brush_motion(event.x, event.y, ia)
    # Zoom out to home view
    if right_click and event.dblclick and not_ax:
        for a, axes in enumerate(app.ax):
            app.ax_limits = deepcopy(ia.home_ax_limits)
            axes.set_xlim(*ia.home_ax_limits[a][0])
            axes.set_ylim(*ia.home_ax_limits[a][1])
        app.fig.canvas.draw_idle()
    if right_click:
        ia.modified = True
    if right_click and ax2_click and ia.clos_mode == 'Auto closing':
        app.xdrc = xd
        app.ydrc = yd
    elif right_click and ax2_click and ia.clos_mode == 'Manual closing':
        cid2_rc = app.fig.canvas.mpl_connect('motion_notify_event', mouse_motion)
        app.cid2_rc = cid2_rc
        ia.init_manual_cont(app, xd, yd)
    # Initialize merging IDs
    do_merge = (right_click and ax1_click and not ia.sep_bud
                and not ia.set_moth and not event.dblclick)
    if do_merge:
        ia.merge_yx[0] = yd
        ia.merge_yx[1] = xd
    # Manual correction: separate emerging bud
    elif right_click and ax1_click and ia.sep_bud and not event.dblclick:
        ID = ia.lab[yd, xd]
        meb = manual_emerg_bud(ia.lab, ID, ia.rp, del_small_obj=True)
        ia.sep_bud = False
        if not meb.cancel:
            if app.is_undo or app.is_redo:
                app.store_state(ia)
            meb_lab = remove_small_objects(meb.sep_bud_label,
                                             min_size=20,
                                             connectivity=2)
            ia.lab[meb_lab != 0] = meb_lab[meb_lab != 0]
            ia.lab[meb.small_obj_mask] = 0
            for y, x in meb.coords_delete:
                del_ID = ia.lab[y, x]
                ia.lab[ia.lab == del_ID] = 0
            # ia.lab[meb.rr, meb.cc] = 0  # separate bud with 0s
            ia.lab = app.relabel_and_fill_lab(ia.lab)
            rp = regionprops(ia.lab)
            ia.rp = rp
            IDs = [obj.label for obj in ia.rp]
            ia.contours = ia.find_contours(ia.lab, IDs, group=True)
            ia.reset_auto_edge_img(ia.contours)
            app.update_ax0_plot(ia, ia.img)
            app.update_ax1_plot(ia.lab, rp, ia)
            app.update_ax2_plot(ia)
            app.store_state(ia)
    elif wheel_click and ax0_click:
        mothID = ia.lab[yd, xd]
        df = ia.cc_stage_df
        df.at[mothID, 'Cell cycle stage'] = 'S'
        df.at[mothID, 'Relationship'] = 'mother'
        text_label_centroid(ia.rp, app.ax[0], 12, 'semibold', 'center', 'center',
                            cc_stage_frame=ia.cc_stage_df,
                            display_ccStage=app.display_ccStage, color='r',
                            clear=True)
        app.fig.canvas.draw_idle()
    elif right_click and ax0_click:
        app.xdrc = xd
        app.ydrc = yd
    # Manual correction: delete ID
    elif wheel_click and ax1_click:
        if app.is_undo or app.is_redo:
            app.store_state(ia)
        ID = ia.lab[yd, xd]
        if ID != 0:
            ia.lab[ia.lab == ID] = 0
            ia.lab = app.relabel_and_fill_lab(ia.lab)
            ia.rp = regionprops(ia.lab)
            IDs = [obj.label for obj in ia.rp]
            ia.contours = ia.find_contours(ia.lab, IDs, group=True)
            ia.reset_auto_edge_img(ia.contours)
            app.update_ax0_plot(ia, ia.img)
            app.update_ax1_plot(ia.lab, ia.rp, ia)
            app.update_ax2_plot(ia)
            app.store_state(ia)
        else:
            app.draw_ROI_delete = True
            app.ROI_delete_patch = Rectangle((0, 0), 1, 1, fill=False, color='r')
            app.cid_ROI_delete = app.fig.canvas.mpl_connect('motion_notify_event',
                                                            mouse_motion)
            app.xdwc_ax1 = xd
            app.ydwc_ax1 = yd
    # Manual correction: delete contour pixel
    elif wheel_click and ax2_click:
        if app.is_undo or app.is_redo:
            app.store_state(ia)
        xx, yy = ia.contour_plot
        dist = ((np.asarray(yy) - yd)**2 + (np.asarray(xx) - xd)**2)
        min_idx = dist.argmin()
        point_to_del = yy[min_idx], xx[min_idx]
        cnt_points = [(y, x) for y, x in zip(yy, xx)]
        xx = [cnt_p[1] for cnt_p in cnt_points if cnt_p != point_to_del]
        yy = [cnt_p[0] for cnt_p in cnt_points if cnt_p != point_to_del]
        ia.contour_plot = [xx, yy]
        contour_img = np.zeros_like(ia.auto_edge_img)
        contour_img[yy, xx] = True
        ia.auto_edge_img = binary_fill_holes(contour_img)
        lab, rp = ia.separate_overlap(label(ia.auto_edge_img))
        app.update_ax0_plot(ia, ia.img)
        app.update_ax2_plot(ia)
        app.update_ax1_plot(lab, rp, ia)
        app.store_state(ia)
    # Freely draw contour on ax0 to save a mask used for downstream analysis
    # of amounts from fluorescent signal
    elif left_click and ax0_click and ia.draw_bg_mask_on and not app.brush_mode_on:
        cid0_lc = app.fig.canvas.mpl_connect('motion_notify_event', mouse_motion)
        app.cid0_lc = cid0_lc
        app.xdlc_0 = xd
        app.ydlc_0 = yd
        app.xdlc = xd
        app.ydlc = yd
        app.manual_verts.append((xd, yd))
        app.manual_count = 0
    # Manual correction: enforce automatic bud separation
    elif right_click and ax1_click and event.dblclick:
        ID = ia.lab[yd, xd]
        if ID != 0:
            if app.is_undo or app.is_redo:
                app.store_state(ia)
            max_ID = ia.lab.max()
            ia.lab = ia.auto_separate_bud_ID(ID, ia.lab, ia.rp, max_ID,
                                             enforce=True)
            ia.rp = regionprops(ia.lab)
            IDs = [obj.label for obj in ia.rp]
            ia.contours = ia.find_contours(ia.lab, IDs, group=True)
            ia.reset_auto_edge_img(ia.contours)
            app.update_ax0_plot(ia, ia.img)
            app.update_ax2_plot(ia)
            app.update_ax1_plot(ia.lab, ia.rp, ia)
            app.store_state(ia)


def mouse_motion(event):
    right_motion = event.button == 3
    left_motion = event.button == 1
    wheel_motion = event.button == 2
    ax0_motion = event.inaxes == app.ax[0]
    ax1_motion = event.inaxes == app.ax[1]
    ax2_motion = event.inaxes == app.ax[2]
    event_x, event_y = event.x, event.y
    brush_circle_on = (not ia.draw_bg_mask_on and app.brush_mode_on
                       and ax1_motion and not app.zoom_on)
    if event.xdata:
        xdr = int(round(event.xdata))
        ydr = int(round(event.ydata))
    if right_motion and ax2_motion and ia.clos_mode == 'Manual closing':
        ia.manual_contour(app, ydr, xdr)
    # Freely draw contour on ax0 to save a mask used for downstream analysis
    # of amounts from fluorescent signal
    elif left_motion and ia.draw_bg_mask_on and not app.brush_mode_on:
        x_lc, y_lc = app.ax_transData_and_coerce(app.ax[1], event_x, event_y,
                                                 ia.img.shape)
        line_ld = Line2D([app.xdlc, x_lc], [app.ydlc, y_lc], color='r')
        app.ax[0].add_line(line_ld)
        app.manual_count += 1
        app.xdlc = x_lc
        app.ydlc = y_lc
        app.manual_verts.append((x_lc, y_lc))
        if app.manual_count > 3:
            app.fig.canvas.draw_idle()
            app.manual_count = 0
    # Freely paint/erase with the brush tool
    elif left_motion and not app.select_ID_on and app.brush_mode_on:
        app.apply_brush_motion(event.x, event.y, ia)
    # Draw rectangular ROI to delete all objects inside the rectangle
    elif wheel_motion and app.draw_ROI_delete:
        event_x, event_y = event.x, event.y
        ax = app.ax[1]
        ax.patches = []
        xdr_wc, ydr_wc = app.ax_transData_and_coerce(ax, event_x, event_y,
                                                     ia.img.shape)
        y_start, y_end = sorted([app.ydwc_ax1, ydr_wc])
        x_start, x_end = sorted([app.xdwc_ax1, xdr_wc])
        rect_w = (x_end+1)-x_start
        rect_h = (y_end+1)-y_start
        ROI_delete_patch = Rectangle((x_start, y_start), rect_w, rect_h,
                                        fill=False, color='r')
        ax.add_patch(ROI_delete_patch)
        app.fig.canvas.draw_idle()
    # Overlay a circle with radius=brush_size on the mouse cursor when it is
    # moving on top of labels image (center image)
    elif brush_circle_on:
        app.draw_brush_circle_cursor(ia, event_x, event_y)


def mouse_up(event):
    right_click = event.button == 3
    left_click = event.button == 1
    wheel_click = event.button == 2
    ax0_click = event.inaxes == app.ax[0]
    ax1_click = event.inaxes == app.ax[1]
    ax2_click = event.inaxes == app.ax[2]
    left_ax1_click = left_click and ax1_click
    event_x, event_y = event.x, event.y
    is_brush_on = app.brush_mode_on and not app.zoom_on
    if event.xdata is not None:
        xu = int(round(event.xdata))
        yu = int(round(event.ydata))
    # Manual correction: add automatically drawn contour
    if right_click and ax2_click and ia.clos_mode == 'Auto closing':
        if app.is_undo or app.is_redo:
            app.store_state(ia)
        app.xu2rc = xu
        app.yu2rc = yu
        lab, rp = ia.auto_contour(app)
        lab[lab>0] += ia.lab.max() + 1
        lab_mask = ia.lab>0
        lab[lab_mask] = ia.lab[lab_mask]
        app.update_ax0_plot(ia, ia.img)
        app.update_ax1_plot(ia.lab, ia.rp, ia)
        app.store_state(ia)
    # Manual correction: add manually drawn contour
    if right_click and ax2_click and ia.clos_mode == 'Manual closing':
        if app.is_undo or app.is_redo:
            app.store_state(ia)
        app.fig.canvas.mpl_disconnect(app.cid2_rc)
        IDs = [obj.label for obj in ia.rp]
        lab, rp = ia.close_manual_cont()
        ia.lab = lab
        ia.rp = regionprops(ia.lab)
        IDs = [obj.label for obj in ia.rp]
        ia.contours = ia.find_contours(ia.lab, IDs, group=True)
        app.update_ax0_plot(ia, ia.img)
        app.update_ax1_plot(ia.lab, ia.rp, ia)
        # ia.auto_edge_img = np.zeros_like(ia.auto_edge_img)
        app.store_state(ia)
    # Freely paint/erase with the brush tool
    if left_ax1_click and not app.select_ID_on and is_brush_on:
        app.apply_brush_motion(event.x, event.y, ia)
        # RELABEL LAB because eraser can split object in two
        ia.rp = regionprops(ia.lab)
        IDs = [obj.label for obj in ia.rp]
        ia.contours = ia.find_contours(ia.lab, IDs, group=True)
        ia.reset_auto_edge_img(ia.contours)
        ia.cc_stage_df = ia.init_cc_stage_df(ia.rp)
        ia.cc_stage_df = ia.assign_bud(ia.cc_stage_df, ia.rp)
        app.update_ax0_plot(ia, ia.img)
        app.update_ax1_plot(ia.lab, ia.rp, ia)
        app.store_state(ia)
    # Close freely drawn contour and add to mask points inside contour
    elif left_click and ia.draw_bg_mask_on and not app.brush_mode_on:
        if app.is_undo or app.is_redo:
            app.store_state(ia)
        app.fig.canvas.mpl_disconnect(app.cid0_lc)
        xu, yu = app.ax_transData_and_coerce(app.ax[0], event_x, event_y,
                                             ia.img.shape)
        ymax, xmax = ia.img.shape
        if not ax0_click:
            if xu == 0:
                line_ld = Line2D([xu, 0], [yu, app.ydlc_0], color='r')
                app.ax[0].add_line(line_ld)
                line_ld = Line2D([0, app.xdlc_0], [app.ydlc_0, app.ydlc_0], color='r')
                app.ax[0].add_line(line_ld)
                app.manual_count = 0
                app.manual_verts.append((0, app.ydlc_0))
                app.manual_verts.append((app.xdlc_0, app.ydlc_0))
            elif yu == 0:
                line_ld = Line2D([xu, app.xdlc_0], [yu, 0], color='r')
                app.ax[0].add_line(line_ld)
                line_ld = Line2D([app.xdlc_0, app.xdlc_0], [0, app.ydlc_0], color='r')
                app.ax[0].add_line(line_ld)
                app.manual_count = 0
                app.manual_verts.append((app.xdlc_0, 0))
                app.manual_verts.append((app.xdlc_0, app.ydlc_0))
            elif xu == xmax-1:
                line_ld = Line2D([xu, xu], [yu, app.ydlc_0], color='r')
                app.ax[0].add_line(line_ld)
                line_ld = Line2D([xu, app.xdlc_0], [app.ydlc_0, app.ydlc_0], color='r')
                app.ax[0].add_line(line_ld)
                app.manual_count = 0
                app.manual_verts.append((xu, app.ydlc_0))
                app.manual_verts.append((app.xdlc_0, app.ydlc_0))
            else:
                line_ld = Line2D([xu, app.xdlc_0], [yu, yu], color='r')
                app.ax[0].add_line(line_ld)
                line_ld = Line2D([app.xdlc_0, app.xdlc_0], [yu, app.ydlc_0], color='r')
                app.ax[0].add_line(line_ld)
                app.manual_count = 0
                app.manual_verts.append((app.xdlc_0, yu))
                app.manual_verts.append((app.xdlc_0, app.ydlc_0))
        else:
            line_ld = Line2D([xu, app.xdlc_0], [yu, app.ydlc_0], color='r')
            app.ax[0].add_line(line_ld)
            app.manual_count = 0
            app.manual_verts.append((app.xdlc_0, app.ydlc_0))
        for i, (x0, y0) in enumerate(app.manual_verts):
            try:
                x1, y1 = app.manual_verts[i+1]
                rr, cc = line(y0, x0, y1, x1)
                ia.manual_mask[rr, cc] = True
            except IndexError:
                break
        ia.manual_mask = binary_fill_holes(ia.manual_mask)
        mask_lab = label(ia.manual_mask)
        mask_rp = regionprops(mask_lab)
        for obj in mask_rp:
            yc, xc = obj.centroid
            app.ax[0].scatter(xc, yc, s=72, c='r', marker='x')
        app.manual_verts = []
        app.fig.canvas.draw_idle()
        draw_bg_mask_on_cb(event)
        app.store_state(ia)
    # Manual correction: merge IDs
    elif right_click and ax1_click and not ia.sep_bud:
        if app.is_undo or app.is_redo:
            app.store_state(ia)
        ia.merge_yx[2] = yu
        ia.merge_yx[3] = xu
        r0, c0, r1, c1 = ia.merge_yx
        yy, xx = line(r0, c0, r1, c1)
        temp = np.unique(ia.lab[yy, xx])
        IDs_merge = temp[temp != 0]
        ID = IDs_merge[0]
        IDs_merge = IDs_merge[1:]
        for id in IDs_merge:
            ia.lab[ia.lab == id] = ID
        ia.rp = regionprops(ia.lab)
        IDs = [obj.label for obj in ia.rp]
        ia.contours = ia.find_contours(ia.lab, IDs, group=True)
        app.update_ax0_plot(ia, ia.img)
        app.update_ax1_plot(ia.lab, ia.rp, ia)
        app.update_ax2_plot(ia)
        app.store_state(ia)
    # Manual correction: delete all IDs inside rectangular ROI
    elif wheel_click and ax1_click and app.draw_ROI_delete:
        if app.is_undo or app.is_redo:
            app.store_state(ia)
        app.fig.canvas.mpl_disconnect(app.cid_ROI_delete)
        app.draw_ROI_delete = False
        event_x, event_y = event.x, event.y
        ax = app.ax[1]
        xu_wc, yu_wc = app.ax_transData_and_coerce(ax, event_x, event_y,
                                                   ia.img.shape)
        y_start, y_end = sorted([app.ydwc_ax1, yu_wc])
        x_start, x_end = sorted([app.xdwc_ax1, xu_wc])
        lab_ROI = ia.lab[y_start:y_end, x_start:x_end]
        IDs_delete = np.unique(lab_ROI)
        for ID in IDs_delete:
            ia.lab[ia.lab==ID] = 0
        ia.lab = app.relabel_and_fill_lab(ia.lab)
        ia.rp = regionprops(ia.lab)
        IDs = [obj.label for obj in ia.rp]
        ia.contours = ia.find_contours(ia.lab, IDs, group=True)
        app.update_ax0_plot(ia, ia.img)
        app.update_ax1_plot(ia.lab, ia.rp, ia)
        app.update_ax2_plot(ia)
        app.store_state(ia)
    # Manually assign bud to mother
    elif right_click and ax0_click and not ia.set_moth:
        budID = ia.lab[app.ydrc, app.xdrc]
        mothID = ia.lab[yu, xu]
        if mothID == 0 or budID == 0:
            print('WARNING: You clicked (or released) on background! '
                  'No cell cycle stage can be assigned to background!')
        elif mothID != budID:
            if app.is_undo or app.is_redo:
                app.store_state(ia)
            if ia.prev_bud_assignment_info is None:
                ia.prev_bud_assignment_info = [{
                                             'bud_coords': (app.ydrc, app.xdrc),
                                             'moth_coords': (yu, xu)}]
            else:
                ia.prev_bud_assignment_info.append({
                                             'bud_coords': (app.ydrc, app.xdrc),
                                             'moth_coords': (yu, xu)})
            df = ia.cc_stage_df
            df.at[budID, 'Cell cycle stage'] = 'S'
            df.at[mothID, 'Cell cycle stage'] = 'S'
            df.at[budID, '# of cycles'] = 0
            df.at[budID, 'Relative\'s ID'] = mothID
            df.at[mothID, 'Relative\'s ID'] = budID
            df.at[mothID, 'Relationship'] = 'mother'
            df.at[budID, 'Relationship'] = 'bud'
            text_label_centroid(ia.rp, app.ax[0], 12, 'semibold', 'center',
                                'center', cc_stage_frame=ia.cc_stage_df,
                                display_ccStage=app.display_ccStage,
                                color='r', clear=True)
            app.fig.canvas.draw_idle()
            app.store_state(ia)
            app.key_mode = ''
        elif mothID == budID:
            if app.is_undo or app.is_redo:
                app.store_state(ia)
            if ia.prev_bud_assignment_info is None:
                ia.prev_bud_assignment_info = [{
                                             'bud_coords': (app.ydrc, app.xdrc),
                                             'moth_coords': (yu, xu)}]
            else:
                ia.prev_bud_assignment_info.append({
                                             'bud_coords': (app.ydrc, app.xdrc),
                                             'moth_coords': (yu, xu)})
            df = ia.cc_stage_df
            bud_ccstage = df.at[budID, 'Cell cycle stage']
            bud_ccnum = df.at[budID, '# of cycles']
            if bud_ccstage == 'S' and bud_ccnum == 0:
                mothID = df.at[budID, 'Relative\'s ID']
                df.at[budID, 'Cell cycle stage'] = 'G1'
                df.at[mothID, 'Cell cycle stage'] = 'G1'
                df.at[budID, '# of cycles'] = -1
                df.at[budID, 'Relative\'s ID'] = 0
                df.at[mothID, 'Relative\'s ID'] = 0
                df.at[mothID, 'Relationship'] = 'mother'
                df.at[budID, 'Relationship'] = 'mother'
                text_label_centroid(ia.rp, app.ax[0], 12, 'semibold', 'center',
                                    'center', cc_stage_frame=ia.cc_stage_df,
                                    display_ccStage=app.display_ccStage,
                                    color='r', clear=True)
                app.fig.canvas.draw_idle()
                app.store_state(ia)
            else:
                tk.messagebox.showwarning('Wrong ID',
                              f'You clicked on cell ID {budID} which is not a bud.\n'
                              'To undo bud assignment you have to click '
                              'on a bud (i.e. a cell in S-0)')


def axes_enter(event):
    if event.inaxes == ax_slice:
        app.key_mode = 'Z-slice'
    elif event.inaxes == ax_view_slices or event.inaxes == app.ax[0]:
        app.key_mode = 'view_slice'

def axes_leave(event):
    app.key_mode = ''
    if event.inaxes == app.ax[2] and ia.clos_mode == 'Manual closing':
        app.mng.window.configure(cursor='arrow')

def on_xlim_changed(axes):
    ax_idx = list(app.ax).index(axes)
    xlim = axes.get_xlim()
    app.ax_limits[ax_idx][0] = xlim
    app.reset_view = False

def on_ylim_changed(axes):
    ax_idx = list(app.ax).index(axes)
    ylim = axes.get_ylim()
    app.ax_limits[ax_idx][1] = ylim
    app.reset_view = False

def handle_close(event):
    save = tk.messagebox.askyesno('Save', 'Do you want to save segmented data?')
    if save:
        save_cb(None)

t0 = 0
sensitivity = 6
def scroll_cb(event):
    global t0, t1
    # Scroll zoom (activated with 'control')
    if event.inaxes and app.scroll_zoom:
        t1 = time()
        rate = 1/(t1-t0)
        step = event.step*sensitivity
        step_rate = abs(step*rate)
        # Adjust zoom factor by scrolling rate
        if step_rate > sensitivity:
            if step_rate > 50:
                step = 50*event.step
            else:
                step = step_rate*event.step
        Y, X = ia.img.shape
        xc = event.xdata
        yc = event.ydata
        xl, xr = event.inaxes.get_xlim()
        yb, yt = event.inaxes.get_ylim()
        # Center zoom at mouse cursor position (xc, yc)
        step_left = (xc-xl)/(X/2)*step
        step_right = (xr-xc)/(X/2)*step
        step_bottom = (yb-yc)/(Y/2)*step
        step_top = (yc-yt)/(Y/2)*step
        new_xl = xl+step_left
        new_xr = xr-step_right
        new_yb = yb-step_bottom
        new_yt = yt+step_top
        # Avoid zoomming out more than the image shape
        new_xl = new_xl if new_xl > -0.5 else -0.5
        new_xr = new_xr if new_xr < X-0.5 else X-0.5
        new_yb = new_yb if new_yb < Y-0.5 else Y-0.5
        new_yt = new_yt if new_yt > -0.5 else -0.5
        # Apply zoom
        event.inaxes.set_xlim((new_xl, new_xr))
        event.inaxes.set_ylim((new_yb, new_yt))
        app.fig.canvas.draw_idle()
        app.connect_axes_cb()
        t0 = t1




# Canvas events
(app.fig.canvas).mpl_connect('button_press_event', mouse_down)
(app.fig.canvas).mpl_connect('button_release_event', mouse_up)
(app.fig.canvas).mpl_connect('key_press_event', key_down)
(app.fig.canvas).mpl_connect('key_release_event', key_up)
(app.fig.canvas).mpl_connect('resize_event', resize_widgets)
(app.fig.canvas).mpl_connect('axes_enter_event', axes_enter)
(app.fig.canvas).mpl_connect('axes_leave_event', axes_leave)
cid_close = (app.fig.canvas).mpl_connect('close_event', handle_close)
(app.fig.canvas).mpl_connect('motion_notify_event', mouse_motion)
(app.fig.canvas).mpl_connect('scroll_event', scroll_cb)
# NOTE: axes limit changed event is connected first time in resize_widgets

app.pos_txt = app.fig.text(0.5, 0.9,
        f'Current position = {app.p+1}/{num_pos}   ({app.pos_foldernames[app.p]})',
        color='w', ha='center', fontsize=14)

if app.segm_npy_done[app.p] is not None:
    app.use_unet = app.is_pc
    param = load_param_from_folder(app)
    if not app.use_unet:
        switch_use_unet_button(False)
    app.update_ALLplots(ia)
else:
    app.use_unet = app.is_pc
    if not app.use_unet:
        switch_use_unet_button(False)
    analyse_img(img)
man_clos_cb(None)
app.set_orig_lims()
app.store_state(ia)

win_title = f'{app.exp_parent_foldername}/{app.exp_name}'
try:
    win_size(swap_screen=True)
except:
    pass
app.fig.canvas.set_window_title(f'Cell segmentation GUI - {win_title}')
plt.show()
