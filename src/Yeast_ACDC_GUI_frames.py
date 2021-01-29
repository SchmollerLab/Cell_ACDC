import os, sys
import traceback
from time import time
from sys import exit, exc_info
from copy import deepcopy
from natsort import natsorted
import re
import cv2
import numpy as np
import pandas as pd
import matplotlib
from math import atan2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
from matplotlib.backend_bases import NavigationToolbar2
from tkinter import E, S, W, END
import tkinter as tk
from skimage import io, img_as_float
from skimage.feature import peak_local_max
from skimage.filters import (gaussian, sobel, apply_hysteresis_threshold,
                            threshold_otsu)
from skimage.measure import label, regionprops
from skimage.morphology import (remove_small_objects, convex_hull_image,
                                dilation, erosion, disk)
from skimage.exposure import equalize_adapthist
from skimage.draw import line, line_aa
from skimage.color import gray2rgb
from scipy.ndimage.morphology import binary_fill_holes, distance_transform_edt
from tifffile import TiffFile
from Yeast_ACDC_MyWidgets import (Slider, Button, RadioButtons, TextBox,
                       MyRadioButtons)
from Yeast_ACDC_FUNCTIONS import (auto_select_slice, manual_emerg_bud,
                       separate_overlapping, text_label_centroid, tk_breakpoint,
                       CellInt_slideshow, num_frames_toSegm_tk, newID_app,
                       CellInt_slideshow_2D, ShowWindow_from_title,
                       select_exp_folder, align_frames_3D, align_frames_2D,
                       load_shifts, tk_breakpoint, folder_dialog, file_dialog,
                       win_size, imshow_tk)

# Import YeaZ module
#script_dirname = os.path.dirname(os.path.realpath(__file__))
#unet_path = f'{script_dirname}/YeaZ-unet/unet/'
#sys.path.append(unet_path)
from YeaZ.unet import segment


"""
NOTE for implementing a NEW manual feature:
    1. Save the inputs required to perform that manual function
    2. Add the manual function name to .li_manual_func attribute of ia
    3. Add the the new manual function case to repeat_manual_func module of ia
"""

"""                          INSTRUCTIONS
GUI Mouse Events: - Auto contour with RIGHT-click on edge img: click on the
                    edge image (if not visible press "Switch to contour mode")
                    where you clearly see a bright contour to auto
                    segment the cell along that contour.
                  - Manual contour with RIGHT-click on edge img: roughly follow
                    a bright gradient contour on the edge img with pressing
                    the right button.
                  - Merge IDs with RIGHT-click press-release on label img:
                    draw a line that starts from first ID and ends on last ID.
                    All IDs on this line will be merged to single ID.
                  - Press 'ctrl+p' to print the cc_stage dataframe
                  - Manually separate bud with 'b' plus RIGHT-click on label
                    img
                  - Delete cell ID from label image with WHEEL-click on label
                  - Delete contour pixel with WHEEL-click on contour image:
                    wheel-click on the pixel you want to remove from contour
                  - Automatic zoom with LEFT-double click
                  - Restore original view (no zoom) with RIGHT-double click
                    anywhere outside axis
                  - Use hull contour with LEFT-double click on Cell ID on label
                  - Replace and assign new ID with WHEEL-click on img
                  - Draw an area where to automatically delete objects by
                    pressing 'd' + draw with WHEEL-click and drag on label.
                    Delete this area by double click with WHEEL-button on label.
                  - To divide a cell into multiple cells along an intensity line
                    press 'c' and draw with RIGHT-click on edge where you want to
                    cell to be splitted
                  - Press 'ctrl+z' for undo and 'ctrl+y' for redo
"""

class load_data:
    def __init__(self, path):
        self.path = path
        self.parent_path = os.path.dirname(path)
        self.filename, self.ext = os.path.splitext(os.path.basename(path))
        if self.ext == '.tif' or self.ext == '.tiff':
            tif_path, phc_tif_found = self.substring_path(path,
                                                         'phase_contr.tif',
                                                          self.parent_path)
            if phc_tif_found:
                self.tif_path = tif_path
                img_data = io.imread(path)
            else:
                tk.messagebox.showerror('Phase contrast file not found!',
                'Phase contrast .tif file not found in the selected path\n'
                f'{self.parent_path}!\n Make sure that the folder contains '
                'a file that ends with either \"phase_contr.tif\" or '
                '\"phase_contrast.tif\"')
                raise FileNotFoundError
        elif self.ext == '.npy':
            if path.find('_phc_aligned.npy') == -1:
                filename = os.path.basename(path)
                tk.messagebox.showerror('Wrong file selected!',
                f'You selected a file called {filename} which is not a valid '
                'phase contrast image file. Select the file that ends with '
                '\"phc_aligned.npy\" or the .tif phase contrast file')
                raise FileNotFoundError
            tif_path, phc_tif_found = self.substring_path(path,
                                                         'phase_contr.tif',
                                                          self.parent_path)
            if phc_tif_found:
                self.tif_path = tif_path
                img_data = np.load(path)
            else:
                tk.messagebox.showerror('Phase contrast file not found!',
                'Phase contrast .tif file not found in the selected path\n'
                f'{self.parent_path}!\n Make sure that the folder contains '
                'a file that ends with either \"phase_contr.tif\" or '
                '\"phase_contrast.tif\"')
                raise FileNotFoundError
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
        self.last_tracked_i_path = f'{base_path}_last_tracked_i.txt'

    def substring_path(self, path, substring, parent_path):
        substring_found = False
        for filename in os.listdir(parent_path):
            if substring == "phase_contr.tif":
                is_match = (filename.find(substring) != -1 or
                            filename.find("phase_contrast.tif") != -1 or
                            filename.find("phase_contrast.tiff") != -1or
                            filename.find("phase_contr.tiff") != -1)
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

"""Classes"""
class init_App:
    def __init__(self, img_path):
        data = load_data(img_path)
        self.segm_npy_done = [None]*data.SizeT
        self.orig_segm_npy = [None]*data.SizeT
        self.cc_stages = [None]*data.SizeT
        if os.path.exists(data.slice_used_segm_path):
            df_slices = pd.read_csv(data.slice_used_segm_path)
            self.slices_used = df_slices['Slice used for segmentation'].to_list()
        else:
            self.slices_used = None
        pos_path = os.path.dirname(os.path.dirname(img_path))
        self.pos_foldername = os.path.basename(pos_path)
        self.exp_name = os.path.basename(os.path.dirname(pos_path))
        print('Loading data...')
        parent_path = os.path.dirname(img_path)
        filenames = os.listdir(parent_path)
        self.parent_path = parent_path
        self.last_tracked_i = -1
        segm_npy_found = False
        for j, filename in enumerate(filenames):
            if filename.find('_segm.npy') != -1:
                segm_npy_found = True
                segm_npy_path = f'{parent_path}/{filename}'
                segm_npy = np.load(segm_npy_path)
                if data.SizeZ > 1:
                    if segm_npy.ndim == 4:
                        print('Segmentation file has z-stacks. '
                              'Removing Z dimension...')
                        segm_npy = segm_npy[:,0,:,:]
                        np.save(segm_npy_path, segm_npy, allow_pickle=False)
                        print('Dimension removed and segmentation file overwritten')
                self.segm_npy_done[:len(segm_npy)] = deepcopy(list(segm_npy))
                self.orig_segm_npy[:len(segm_npy)] = deepcopy(list(segm_npy))
            elif filename.find('_cc_stage.csv') != -1:
                cc_stage_path = f'{parent_path}/{filename}'
                cca_df = pd.read_csv(cc_stage_path, index_col=['frame_i', 'Cell_ID'])
                cca_df_li = [df.loc[frame_i] for frame_i, df in cca_df.groupby(level=0)]
                self.cc_stages[:len(cca_df_li)] = cca_df_li
            elif filename.find('_last_tracked_i.txt') != -1:
                last_tracked_i_path = f'{parent_path}/{filename}'
                with open(last_tracked_i_path, 'r') as txt:
                    self.last_tracked_i = int(txt.read())
        frames = data.img_data
        if not segm_npy_found:
            print('WARNING: No pre-computed segmentation file found!')
        self.last_segm_i = data.SizeT
        for frame_i, elem in enumerate(self.segm_npy_done):
            if elem is None:
                self.last_segm_i = frame_i
                break
        start, _ = num_frames_toSegm_tk(data.SizeT,
                                  last_segm_i=self.last_segm_i,
                                  last_tracked_i=self.last_tracked_i).frange
        self.start = start
        self.num_slices = data.SizeZ
        if self.slices_used is not None:
            self.slices = self.slices_used
        else:
            self.slices = [1]
        self.img_data = data.img_data
        self.display_ccStage = 'Only IDs'
        self.display_IDs_cont = 'IDs and contours'
        self.frame_i = start
        self.reset_view = False
        self.frames = data.img_data
        self.auto_save = False
        self.frame_i_done = -1
        print('Total number of Positions = {}'.format(len(self.frames)))
        self.bp = tk_breakpoint()
        self.data = data
        self.unet_first_call = True
        self.draw_ROI_delete = False
        self.ROI_delete_coords = []
        self.do_cut = False
        self.prev_states = []
        self.is_undo = False
        self.is_redo = False
        self.use_unet = True
        self.do_overlay = False
        self.ol_frames = None
        self.ol_RGB_val = [1,1,0]
        self.init_attr()

    def set_state(self, ia):
        if self.is_undo or self.is_redo:
            self.prev_states = []
            self.is_undo = False
        ia.warn_txt_text = app.warn_txt._text
        self.prev_states.append(deepcopy(ia))
        self.count_states = len(self.prev_states)-1

    def get_state(self, ia, count):
        if count >= 0 and count < len(self.prev_states):
            return self.prev_states[count]
        else:
            return ia

    def init_attr(self):
        # Local threshold polygon attributes
        self.s = self.slices[self.frame_i]
        self.cid2_rc = None  # cid for right click on ax2 (thresh)
        self.Line2_rc = None
        self.xdrc = None
        self.ydrc = None
        self.xdrrc = None
        self.ydrrc = None
        self.xurc = None
        self.yurc = None
        self.key_mode = ''
        self.contour_plot = [[],[]]
        self.enlarg_first_call = True
        self.reduce_first_call = True

    def get_img(self, frames, frame_i, num_slices, slice=1):
        if num_slices > 1:
            img = frames[frame_i, slice]
        else:
            img = frames[frame_i]
        return img

    def init_plots(self, left=0.08, bottom=0.2):
        fig, ax = plt.subplots(1,3, figsize=[13.66, 7.68])
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

    def update_IMGplot(self, ia, img, ax_img, clear=False):
        if clear:
            ax_img.clear()
        if self.do_overlay:
            ol_img = self.get_img(self.ol_frames, self.frame_i, self.num_slices,
                                  slice=app.s)
            overlay = self.get_overlay(ol_img, img,
                                      ol_RGB_val=self.ol_RGB_val,
                                      ol_brightness=brightness_slider.val,
                                      ol_alpha=alpha_slider.val)
            ax_img.imshow(overlay)
        else:
            ax_img.imshow(img)
        if self.display_IDs_cont != 'Disable':
            for cont in ia.contours:
                x = cont[:,1]
                y = cont[:,0]
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                ax_img.plot(x, y, c='r', alpha=0.5, lw=1)
        text_label_centroid(ia.rp, ax_img, 12, 'semibold', 'center', 'center',
                            cc_stage_frame=ia.cc_stage_df,
                            display_IDs_cont=self.display_IDs_cont, color='r',
                            clear=True)
        ax_img.axis('off')

    def update_ALLplots(self, ia):
        fig, ax = self.fig, self.ax
        img = ia.img
        edge = ia.edge
        lab = ia.lab
        for a in ax:
            a.clear()
        self.update_IMGplot(ia, img, ax[0])
        ax[1].imshow(lab)
        text_label_centroid(ia.rp, ax[1], 10, 'semibold', 'center', 'center',
                            new_IDs=ia.new_IDs,
                            display_IDs_cont=self.display_IDs_cont)
        self.set_lims()
        if self.ROI_delete_coords:
            for ROI in self.ROI_delete_coords:
                y_start, y_end, x_start, x_end = ROI
                rect_w = (x_end+1)-x_start
                rect_h = (y_end+1)-y_start
                ROI_delete_patch = Rectangle((x_start, y_start), rect_w, rect_h,
                                                fill=False, color='r')
                ax[1].add_patch(ROI_delete_patch)
        if self.display_IDs_cont != 'Disable':
            for cont in ia.contours:
                x = cont[:,1]
                y = cont[:,0]
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                ax[2].plot(x, y, c='silver', alpha=0.5)
        if ia.edge_mode:
            ax[2].imshow(edge)
            # t0 = time()
            # for cont in ia.li_yx_dir_coords:
            #     for i, coords in enumerate(cont):
            #         ys, xs = coords
            #         if i+1 < len(cont):
            #             ye, xe = cont[i+1]
            #             Li = L.ine2D([xs, xe], [ys, ye], color='r')
            #             ax[2].add_line(Li)
            xx, yy = ia.contour_plot
            ax[2].scatter(xx, yy, s=1.5, c='r')
            # print('Drawing contour time = {0:6f}'.format(time()-t0))
        else:
            thresh = ia.thresh
            ax[2].imshow(thresh)
        for a in ax:
            a.axis('off')
        fig.canvas.draw_idle()

    def update_ax2_plot(self, ia, thresh=None):
        if thresh is not None:
            ia.thresh = thresh
        ax = self.ax
        ax[2].clear()
        for cont in ia.contours:
            x = cont[:,1]
            y = cont[:,0]
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            ax[2].plot(x, y, c='silver', alpha=0.5)
        if ia.edge_mode:
            ax[2].imshow(ia.edge)
            # for cont in ia.li_yx_dir_coords:
            #     for i, coords in enumerate(cont):
            #         ys, xs = coords
            #         if i+1 < len(cont):
            #             ye, xe = cont[i+1]
            #             Li = Line2D([xs, xe], [ys, ye], color='r')
            #             ax[2].add_line(Li)
            xx, yy = ia.contour_plot
            ax[2].scatter(xx, yy, s=1.5, c='r')
            # print('Drawing contour time = {0:6f}'.format(time()-t0))
        else:
            ax[2].imshow(thresh)
            for poly in ia.li_Line2_rc:
                for r0, c0, r1, c1 in poly:
                    Li = Line2D([c0, c1], [r0, r1], color='r')
                    ax[2].add_line(Li)
        ax[2].axis('off')
        self.set_lims()
        self.fig.canvas.draw_idle()

    def update_LABplot(self, lab, rp, ia, draw=True):
        ia.rp = rp
        # ia.cc_stage_df = ia.init_cc_stage_df(rp)
        # ia.cc_stage_df = ia.assign_bud(ia.cc_stage_df, rp)
        ia.lab = lab
        ax = self.ax
        ax[0].clear()
        self.update_IMGplot(ia, ia.img, ax[0])
        ax[1].clear()
        ax[1].imshow(lab)
        text_label_centroid(ia.rp, ax[1], 10, 'semibold', 'center', 'center',
                            new_IDs=ia.new_IDs,
                            display_IDs_cont='IDs and contours')
        ax[1].axis('off')
        self.set_lims()
        if self.ROI_delete_coords:
            for ROI in self.ROI_delete_coords:
                y_start, y_end, x_start, x_end = ROI
                rect_w = (x_end+1)-x_start
                rect_h = (y_end+1)-y_start
                ROI_delete_patch = Rectangle((x_start, y_start), rect_w, rect_h,
                                                fill=False, color='r')
                ax[1].add_patch(ROI_delete_patch)
        if draw:
            self.fig.canvas.draw_idle()

    def get_overlay(self, img, ol_img, ol_RGB_val=[1,1,0],
                    ol_brightness=4, ol_alpha=0.5):
        img_rgb = gray2rgb(img_as_float(img))*ol_RGB_val
        ol_img_rgb = gray2rgb(img_as_float(ol_img))
        overlay = (ol_img_rgb*(1.0 - ol_alpha)+img_rgb*ol_alpha)*ol_brightness
        overlay = (np.clip(overlay, 0, 1)*255).astype(np.uint8)
        return overlay

    def closest_value_idx(self, a, val):
        diff = np.abs(a-val).sum(axis=1)
        idx = diff.argmin()
        return idx

    def overlay_cmap_widget(self, ax_rgb, ol_RGB_val=[1,1,0]):
        gradient = np.linspace(0, 1, 256)
        rgb_gradient = np.vstack((gradient, gradient)).transpose()
        ax_rgb.imshow(rgb_gradient, aspect='auto', cmap='hsv')
        rgb_cmap_array = np.asarray([plt.cm.hsv(i) for i in gradient])
        rgba = ol_RGB_val.copy()
        rgba.append(1)
        y_rgb = self.closest_value_idx(rgb_cmap_array, rgba)
        x_min, x_max = ax_rgb.get_xlim()
        x_rgb = (x_max-x_min)/2+x_min
        picked_rgb_marker = ax_rgb.scatter(x_rgb, y_rgb, marker='s',
                                                color='k')
        ax_rgb.axis('off')
        return rgb_cmap_array, x_rgb, picked_rgb_marker

    def update_cells_slideshow(self, ia):
        try:
            self.cells_slideshow.rps[self.frame_i] = regionprops(ia.lab)
            self.cells_slideshow.new_IDs = ia.new_IDs
            self.cells_slideshow.lost_IDs = ia.lost_IDs
            self.cells_slideshow.sl.set_val(app.frame_i)
        except:
            pass

    def save_all(self):
        if self.auto_save:
            segm_npy_path = app.data.segm_npy_path
            slice_path = app.data.slice_used_segm_path
            segm_npy_li = [obj for obj in app.segm_npy_done if obj is not None]
            segm_npy = np.asarray(segm_npy_li)
            if segm_npy.dtype != object:
                np.save(segm_npy_path, segm_npy, allow_pickle=False)
                with open(app.data.last_tracked_i_path, 'w+') as txt:
                    txt.write(str(app.last_segm_i))
            else:
                tk.messagebox.showwarning('Wrong shape!',
                f'Segmentation data has wrong shape {segm_npy.shape} or '
                f'wrong data type {segm_npy.dtype}. Data cannot be saved.')
            # ia.cc_stage_df.to_csv(ccstage_path, index=True, mode='w',
            #                         encoding='utf-8-sig')

    def ax_transData_and_coerce(self, ax, event_x, event_y, img_shape):
        x, y = ax.transData.inverted().transform((event_x, event_y))
        ymax, xmax = img_shape
        xmin, ymin = 0, 0
        if x < xmin:
            x_coerced = 0
        elif x > xmax:
            x_coerced = xmax
        else:
            x_coerced = int(round(x))
        if y < ymin:
            y_coerced = 0
        elif y > ymax:
            y_coerced = ymax
        else:
            y_coerced = int(round(y))
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
        self.edge_mode = False
        self.clos_mode = 'Auto closing'
        self.bp = tk_breakpoint()
        self.modified = False
        self.contours = []
        self.do_tracking = True
        y, x = img.shape
        self.home_ax_limits = [[(-0.5, x-0.5), (y-0.5, -0.5)] for _ in range(3)]
        self.new_IDs = []
        self.lost_IDs = []
        self.manual_newID_coords = []
        self.init_attr(img)

    def init_attr(self, img):
        # Local threshold polygon attributes
        self.locT_img = np.zeros(img.shape, bool)
        self.auto_edge_img = np.zeros(img.shape, bool)
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

    def nearest_nonzero(self, a, y, x):
        r, c = np.nonzero(a)
        dist = ((r - y)**2 + (c - x)**2)
        min_idx = dist.argmin()
        return r[min_idx], c[min_idx]

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

    def close_manual_cont(self):
        self.li_yx_dir_coords.append(self.yx_dir_coords)
        self.auto_edge_img = binary_fill_holes(self.auto_edge_img)
        lab, rp = self.separate_overlap(label(self.auto_edge_img))
        return lab, rp

    def convexity_defects(self, img, eps_percent):
        img = img.astype(np.uint8)
        contours, hierarchy = cv2.findContours(img,2,1)
        cnt = contours[0]
        cnt = cv2.approxPolyDP(cnt,eps_percent*cv2.arcLength(cnt,True),True) # see https://www.programcreek.com/python/example/89457/cv22.convexityDefects
        hull = cv2.convexHull(cnt,returnPoints = False) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        defects = cv2.convexityDefects(cnt,hull) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        return cnt, defects

    def nearest_nonzero(self, a, y, x):
        r, c = np.nonzero(a)
        dist = ((r - y)**2 + (c - x)**2)
        min_idx = dist.argmin()
        return a[r[min_idx], c[min_idx]]

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
        return lab


    def auto_separate_bud(self, lab, rp, eps_percent=0.01):
        IDs = [obj.label for obj in rp]
        max_i = 1
        max_ID = max(IDs)
        for ID in IDs:
            lab = self.auto_separate_bud_ID(ID, lab, rp, max_i=max_i,
                                            eps_percent=eps_percent)
        return lab

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
        gauss = gaussian(img, sigma=0.8, preserve_range=True)
        edge = sobel(gauss)
        return edge

    def auto_contour(self, app, start_yx=None, alfa_dir=None, iter=300, draw=True):
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
        for _ in range(iter):
            yy, xx = self.get_dir_coords(alfa_dir, yd, xd, self.edge.shape)
            a_dir = self.edge[yy, xx]
            min_int = a_dir.max() # if int_val > ta else a_dir.min()
            min_i = list(a_dir).index(min_int)
            y, x = yy[min_i], xx[min_i]
            if draw:
                line_du = Line2D([xd, x], [yd, y], color='r')
                ax[2].add_line(line_du)
            alfa = atan2(yd-y, x-xd)
            base = np.pi/4
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
            cnt_len_li = [len(cnt) for cnt in cont]
            max_len_idx = cnt_len_li.index(max(cnt_len_li))
            cnt = cont[max_len_idx]
            if return_hull:
                hull = cv2.convexHull(cnt,returnPoints = True)
                contours.append(hull)
            else:
                contours.append(cnt)
        if concat:
            all_contours = np.zeros((0,2), dtype=int)
            for contour in contours:
                contours_2D_yx = np.fliplr(np.reshape(contour, (contour.shape[0],2)))
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
            app.update_LABplot(lab, rp, self)

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
            # remove_small_objects(lab, min_size=50, connectivity=2, in_place=True)
            rp = regionprops(lab)
            IDs = [obj.label for obj in rp]
            self.contours = self.find_contours(lab, IDs, group=True)
        except:
            rp = regionprops(lab)
            # traceback.print_exc()
        self.rp = rp
        return lab, rp

    def init_cc_stage_df(self, rp):
        all_cells_IDs = [o.label for o in rp]
        cc_stage = ['G1' for ID in all_cells_IDs]
        num_cycles = [-1]*len(all_cells_IDs)
        relationship = ['mother' for ID in all_cells_IDs]
        related_to = [0]*len(all_cells_IDs)
        OF = np.zeros(len(all_cells_IDs), bool)
        df = pd.DataFrame({
                            'Cell cycle stage': cc_stage,
                            '# of cycles': num_cycles,
                            'Relative\'s ID': related_to,
                            'Relationship': relationship,
                            'OF': OF},
                            index=all_cells_IDs)
        df.index.name = 'Cell_ID'
        return df

    def assign_bud(self, cc_stage_df, rp):
        if len(rp) == 2:
            areas = [obj.area for obj in rp]
            IDs = [obj.label for obj in rp]
            min_area_idx = areas.index(min(areas))
            max_area_idx = areas.index(max(areas))
            budID = IDs[min_area_idx]
            mothID = IDs[max_area_idx]
            cc_stage_df.at[budID, 'Cell cycle stage'] = 'S'
            cc_stage_df.at[mothID, 'Cell cycle stage'] = 'S'
            cc_stage_df.at[budID, '# of cycles'] = 0
            cc_stage_df.at[budID, 'Relative\'s ID'] = mothID
            cc_stage_df.at[mothID, 'Relative\'s ID'] = budID
            cc_stage_df.at[mothID, 'Relationship'] = 'mother'
            cc_stage_df.at[budID, 'Relationship'] = 'bud'
        return cc_stage_df


    def auto_contour_parallel(self, app, yx):
        for a in (-45,45):
            lab, rp = self.auto_contour(app, start_yx=yx, alfa_dir=a,
                                                iter=300, draw=False)
        return lab, rp

    def check_prev_IDs_lost_new(self, prev_IDs, curr_IDs):
        self.lost_IDs = [ID for ID in prev_IDs if ID not in curr_IDs]
        self.new_IDs = [ID for ID in curr_IDs if ID not in prev_IDs]
        warn_txt = ''
        if self.lost_IDs:
            warn_txt = f'Cells IDs lost in current frame: {self.lost_IDs}'
        if self.new_IDs:
            warn_txt = f'{warn_txt}\n\nNew cells IDs in current frame: {self.new_IDs}'
        return warn_txt


    def tracking(self, prev_ia, apply=True):
        IDs_prev = [obj.label for obj in prev_ia.rp]
        IDs_prev.sort()
        IDs_curr_untracked = [obj.label for obj in self.rp]
        IDs_curr_untracked.sort()
        num_IDs_prev = len(IDs_prev)
        num_IDs_curr_untracked = len(IDs_curr_untracked)
        IoU_matrix = np.zeros((num_IDs_curr_untracked, num_IDs_prev))
        for j, ID_prev in enumerate(IDs_prev):
            mask_ID_prev = prev_ia.lab == ID_prev
            A_IDprev = np.count_nonzero(mask_ID_prev)
            IDs_interesect_vect = self.lab[mask_ID_prev]
            IDs, counts = np.unique(IDs_interesect_vect, return_counts=True)
            for ID, intersect in zip(IDs, counts):
                if ID != 0:
                    i = IDs_curr_untracked.index(ID)
                    mask_ID_curr = self.lab == ID
                    xor =  np.logical_xor(mask_ID_prev, mask_ID_curr)
                    union = np.count_nonzero(xor) + intersect
                    IoU = intersect/union
                    IoA_IDprev = intersect/A_IDprev
                    IoU_matrix[i,j] = IoA_IDprev
        IoU_df = pd.DataFrame(data=IoU_matrix, index=IDs_curr_untracked,
                                               columns=IDs_prev)
        # print(IoU_df)
        max_IoU_col_idx = IoU_matrix.argmax(axis=1)
        unique_col_idx, counts = np.unique(max_IoU_col_idx, return_counts=True)
        counts_dict = dict(zip(unique_col_idx, counts))
        tracked_IDs = []
        old_IDs = []
        for i, j in enumerate(max_IoU_col_idx):
            max_IoU = IoU_matrix[i,j]
            count = counts_dict[j]
            if max_IoU > 0.4:
                tracked_ID = IDs_prev[j]
                if count == 1:
                    old_ID = IDs_curr_untracked[i]
                elif count > 1:
                    old_ID_idx = IoU_matrix[:,j].argmax()
                    old_ID = IDs_curr_untracked[old_ID_idx]
                tracked_IDs.append(tracked_ID)
                old_IDs.append(old_ID)
        # print(f'Old untracked IDs: {old_IDs}')
        # print(f'Newly assigned tracked IDs: {tracked_IDs}')
        lost_IDs = [ID for ID in IDs_prev if ID not in tracked_IDs]
        new_untracked_IDs = [ID for ID in IDs_curr_untracked if ID not in old_IDs]
        new_tracked_IDs_2 = []
        tracked_lab = np.copy(ia.lab)
        if new_untracked_IDs:
            max_ID = max(IDs_curr_untracked)
            new_tracked_IDs = [max_ID*(i+2) for i in range(len(new_untracked_IDs))]
            tracked_lab = self.np_replace_values(tracked_lab, new_untracked_IDs,
                                                 new_tracked_IDs)
        if tracked_IDs:
            tracked_lab = self.np_replace_values(tracked_lab, old_IDs, tracked_IDs)
        if new_untracked_IDs:
            max_ID = max(IDs_prev)
            new_tracked_IDs_2 = [max_ID+i+1 for i in range(len(new_untracked_IDs))]
            tracked_lab = self.np_replace_values(tracked_lab, new_tracked_IDs,
                                                 new_tracked_IDs_2)
        # print(f'New tracked cells IDs in current frame: {new_tracked_IDs_2}')
        rp = regionprops(tracked_lab)
        for yd, xd, replacingID in self.manual_newID_coords:
            IDs = [obj.label for obj in rp]
            replaced_ID = tracked_lab[yd, xd]
            if replacingID in IDs:
                tempID = tracked_lab.max() + 1
                tracked_lab[tracked_lab == replaced_ID] = tempID
                tracked_lab[tracked_lab == replacingID] = replaced_ID
                tracked_lab[tracked_lab == tempID] = replacingID
            else:
                tracked_lab[tracked_lab == replaced_ID] = replacingID
        curr_IDs = [obj.label for obj in regionprops(tracked_lab)]
        lost_IDs = [ID for ID in IDs_prev if ID not in curr_IDs]
        new_tracked_IDs_2 = [ID for ID in curr_IDs if ID not in IDs_prev]
        self.lost_IDs = lost_IDs
        self.new_IDs = new_tracked_IDs_2
        # print(f'Cells IDs lost in current frame: {lost_IDs}')
        # print(f'Untracked new IDs in current frame: {new_untracked_IDs}')
        warn_txt = ''
        if lost_IDs:
            warn_txt = f'Cells IDs lost in current frame: {lost_IDs}'
        if new_tracked_IDs_2:
            self.new_IDs = new_tracked_IDs_2
            warn_txt = f'{warn_txt}\n\nNew cells IDs in current frame: {new_tracked_IDs_2}'
        if self.do_tracking and apply:
            return tracked_lab, warn_txt
        elif not apply:
            return self.lab, warn_txt
        else:
            warn_txt = ia.check_prev_IDs_lost_new(IDs_prev, IDs_curr_untracked)
            return self.lab, warn_txt
        # self.bp.pausehere()

    def np_replace_values(self, arr, old_values, tracked_values):
        # See method_jdehesa https://stackoverflow.com/questions/45735230/how-to-replace-a-list-of-values-in-a-numpy-array
        old_values = np.asarray(old_values)
        tracked_values = np.asarray(tracked_values)
        n_min, n_max = arr.min(), arr.max()
        replacer = np.arange(n_min, n_max + 1)
        # Mask replacements out of range
        mask = (old_values >= n_min) & (old_values <= n_max)
        replacer[old_values[mask] - n_min] = tracked_values[mask]
        arr = replacer[arr - n_min]
        return arr

    def full_IAroutine(self, img, app):
        lowT = self.lowT
        highT = self.highT
        edge = self.edge_detector(img)
        thresh = self.thresholding(edge, lowT, highT)
        self.img = img
        self.edge = edge
        self.thresh = thresh
        if self.edge_mode:
            ta = threshold_otsu(edge)
            peaks = peak_local_max(edge, min_distance=20, threshold_abs=ta,
                                   exclude_border=60)
            start_t = time()
            for yx in peaks:
                for a in (-45,45):
                    lab, rp = self.auto_contour(app, start_yx=yx, alfa_dir=a,
                                                        iter=300, draw=False)
            stop_t = time()
            print('Auto contour execution time: {0:.3f}'.format(stop_t-start_t))
            # lab = np.zeros(thresh.shape, int)
            # rp = regionprops(lab)
        else:
            lab, rp = self.segmentation(thresh)
        self.cc_stage_df = self.init_cc_stage_df(rp)
        self.lab = lab
        self.rp = rp

matplotlib.use("TkAgg")

matplotlib.rcParams['keymap.back'] = ['q', 'backspace', 'MouseButton.BACK']
matplotlib.rcParams['keymap.forward'] = ['v', 'MouseButton.FORWARD']
matplotlib.rcParams['keymap.quit'] = []
matplotlib.rcParams['keymap.quit_all'] = []
np.set_printoptions(precision=4, linewidth=200, suppress=True)
pd.set_option('display.max_columns', 40)
pd.set_option('display.precision', 3)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

"""Initialize app GUI parameters"""
plt.style.use('dark_background')
plt.rc('axes', edgecolor='0.1')
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
exp_path = folder_dialog(
                 title='Select experiment folder containing Position_n folders')


select_folder = select_exp_folder()
values = select_folder.get_values_segmGUI(exp_path)
pos_foldername = select_folder.run_widget(values)
images_path = f'{exp_path}/{pos_foldername}/Images'
phc_aligned_found = False
for filename in os.listdir(images_path):
    if filename.find('_phc_aligned.npy') != -1:
        img_path = f'{images_path}/{filename}'
        phc_aligned_found = True
if not phc_aligned_found:
    raise FileNotFoundError('Phase contrast aligned image file not found!')

print(f'Loading {img_path}...')
# img_path = file_dialog(title = "Select phase contrast or bright-field image file")

"""Load data"""
app = init_App(img_path)
frames = app.data.img_data
num_frames = app.data.SizeT
num_slices = app.num_slices

"""Initialize plots"""
home = NavigationToolbar2.home
def new_home(self, *args, **kwargs):
    try:
        app.ax_limits = deepcopy(ia.home_ax_limits)
        app.orig_ax_limits = deepcopy(ia.home_ax_limits)
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

app.init_plots()
img = app.get_img(frames, app.frame_i, num_slices, slice=app.s)

"""Initialize image analysis class"""
param = [None]*num_frames
ia = img_analysis(img)
ia.do_tracking = app.last_tracked_i < app.frame_i

"""Widgets' axes as [left, bottom, width, height]"""
ax_slice = plt.axes([0.1, 0.3, 0.8, 0.03])
# ax_lowT = plt.axes([0.1, 0.2, 0.6, 0.03])
# ax_highT = plt.axes([0.1, 0.15, 0.6, 0.03])
ax_autosave = plt.axes([0.026, 0.01, 0.9, 0.3])
ax_next = plt.axes([0.62, 0.2, 0.05, 0.03])
ax_prev = plt.axes([0.67, 0.2, 0.05, 0.03])
# ax_locT = plt.axes([0.8, 0.2, 0.05, 0.03])
# ax_plus_locT = plt.axes([0.9, 0.2, 0.05, 0.03])
# ax_del_locT = plt.axes([0.6, 0.8, 0.05, 0.05])
ax_frames_slideshow = plt.axes([0.4, 0.8, 0.05, 0.05])
ax_man_clos = plt.axes([0.7, 0.8, 0.05, 0.05])
ax_switch_to_edge = plt.axes([0.03, 0.8, 0.05, 0.05])
ax_undo_auto_edge = plt.axes([0.012, 0.8, 0.05, 0.05])
ax_ccstage_radio = plt.axes([0.015, 0.8, 0.05, 0.05])
ax_save = plt.axes([0.001, 0.8, 0.05, 0.05])
ax_view_slices = plt.axes([0.001, 0.080, 0.05, 0.05])
ax_reload_segm = plt.axes([0.007, 0.080, 0.05, 0.05])
ax_retrack = plt.axes([0.007, 0.0896, 0.05, 0.05])
ax_enlarge_cells = plt.axes([0.9, 0.2, 0.05, 0.03])
ax_reduce_cells = plt.axes([0.6, 0.8, 0.05, 0.05])
ax_morph_IDs = plt.axes([0.125, 0.36, 0.007, 0.9])
ax_track = plt.axes([0.125, 0.48, 0.67, 0.9])
ax_repeat_segm = plt.axes([0.125, 0.74, 0.98, 0.9])
ax_overlay = plt.axes([0.1, 0.35, 0.1, 0.2])
ax_bright_sl = plt.axes([0.1, 0.48, 0.1, 0.2])
ax_alpha_sl = plt.axes([0.1, 0.69, 0.1, 0.2])
ax_rgb = plt.axes([0.1, 0.69, 0.25, 0.2])

"""Widgets"""

# Sliders
sliders = []
s_slice = Slider(ax_slice, 'Z-slice', 5, app.num_slices,
                    valinit=app.s,
                    valstep=1,
                    color=slider_color,
                    init_val_line_color=hover_color,
                    valfmt='%1.0f')
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
view_slices_sl = Slider(ax_view_slices, 'View slice', 0, app.num_slices,
                    valinit=app.s,
                    valstep=1,
                    orientation='horizontal',
                    color=slider_color,
                    init_val_line_color=hover_color,
                    valfmt='%1.0f')
sliders.extend([s_slice, brightness_slider, alpha_slider, view_slices_sl])

# Buttons
buttons = []
autosave_b = Button(ax_autosave, 'Auto-save (slower)',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)
overlay_b = Button(ax_overlay, 'Overlay', color=axcolor,
                hovercolor=hover_color, presscolor=presscolor)
next_b = Button(ax_next, 'Next frame',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)
prev_b = Button(ax_prev, 'Prev. frame',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)
enlarge_cells = Button(ax_enlarge_cells, 'Enlarge cells',
                     color=axcolor, hovercolor=hover_color,
                     presscolor=presscolor)
reduce_cells = Button(ax_reduce_cells, 'Reduce cells',
                     color=axcolor, hovercolor=hover_color,
                     presscolor=presscolor)

buttons.extend([autosave_b, overlay_b, next_b, prev_b,
                enlarge_cells, reduce_cells])

if ia.do_tracking:
    track_b = Button(ax_track, 'Tracking is ENABLED',
                     color=button_true_color, hovercolor=button_true_color,
                     presscolor=presscolor)
else:
    track_b = Button(ax_track, 'Tracking is DISABLED',
                     color=axcolor, hovercolor=hover_color,
                     presscolor=presscolor)

frames_slideshow = Button(ax_frames_slideshow, 'Cells slideshow',
                     color=axcolor, hovercolor=hover_color,
                     presscolor=presscolor)
man_clos = Button(ax_man_clos, 'Switch to manual closing',
                     color=axcolor, hovercolor=hover_color,
                     presscolor=presscolor)
switch_segm_mode = Button(ax_switch_to_edge, 'Switch to contour mode',
                     color=axcolor, hovercolor=hover_color,
                     presscolor=presscolor)
undo_auto_cont = Button(ax_undo_auto_edge, 'Undo auto cont.',
                     color=axcolor, hovercolor=hover_color,
                     presscolor=presscolor)

reload_segm_b = Button(ax_reload_segm, 'Reload segmentation',
                 color=axcolor, hovercolor=hover_color,
                 presscolor=presscolor)
repeat_tracking_b = Button(ax_retrack, 'Repeat tracking',
                 color=axcolor, hovercolor=hover_color,
                 presscolor=presscolor)
repeat_segm_b = Button(ax_repeat_segm, 'Repeat segmentation',
                 color=axcolor, hovercolor=hover_color,
                 presscolor=presscolor)
save_b = Button(ax_save, 'Save and close',
                 color=axcolor, hovercolor=hover_color,
                 presscolor=presscolor)

buttons.extend([track_b, frames_slideshow, man_clos, switch_segm_mode,
                undo_auto_cont, reload_segm_b, repeat_tracking_b,
                repeat_segm_b, save_b])


radio_b_ccStage = MyRadioButtons(ax_ccstage_radio,
                              ('IDs and contours',
                              'Only contours',
                              'Disable'),
                              active = 0,
                              activecolor = button_true_color,
                              orientation = 'horizontal',
                              size = 59,
                              circ_p_color = button_true_color)

morhp_ids_tb = TextBox(ax_morph_IDs, 'Enlarge\Reduce only cells IDs:',
                      initial='All',color=axcolor, hovercolor=hover_color)

(app.rgb_cmap_array, app.x_rgb,
app.picked_rgb_marker) = app.overlay_cmap_widget(ax_rgb)


"""Functions"""
def analyse_img(img):
    if app.use_unet:
        ia.unet_segmentation(img, app)
    else:
        ia.full_IAroutine(img, app)
    app.update_ALLplots(ia)
    app.update_ALLplots(ia)

def tracking_routine(app, ia, prev_ia=None, apply=True):
    if app.frame_i >= 1:
        if prev_ia is None:
            prev_ia = param[app.frame_i-1]
        tracked_lab, warn_txt = ia.tracking(prev_ia, apply=apply)
        app.warn_txt._text = warn_txt
        ia.lab = tracked_lab.copy()
        ia.rp = regionprops(tracked_lab)

def load_param(param):
    ia = param[app.frame_i]
    return ia

def store_param(ia):
    param[app.frame_i] = deepcopy(ia)
    return param

def set_param_from_folder(num_frames, app):
    param = [None]*num_frames
    for i in range(app.start+1):
        img = app.get_img(app.img_data, i, app.num_slices,
                          slice=app.s)
        ia = img_analysis(img)
        ia.init_attr(img)
        ia.lab = app.orig_segm_npy[i]
        ia.rp = regionprops(ia.lab)
        ia.cc_stage_df = ia.init_cc_stage_df(ia.rp)
        param[i] = ia
    return param

def load_param_from_folder(app):
    global ia
    ia.lab = app.segm_npy_done[app.frame_i].copy()
    ia.rp = regionprops(ia.lab)
    if app.cc_stages[app.frame_i] is not None:
        ia.cc_stage_df = app.cc_stages[app.frame_i]
    else:
        ia.cc_stage_df = None
    if app.slices_used is not None:
        if app.slices_used[app.frame_i] is not None:
            app.s = app.slices_used[app.frame_i]
            ia.slice_used = app.s
    else:
        app.s = 1
        ia.slice_used = app.s
    ia.img = app.get_img(app.frames, app.frame_i, app.num_slices, slice=app.s)
    ia.edge = ia.edge_detector(ia.img)
    ia.contour_plot = [[], []]
    ia.auto_edge_img = np.zeros(ia.img.shape, bool)
    app.frame_txt._text = f'Current frame = {app.frame_i}/{num_frames-1}'
    ia.modified = False
    IDs = [obj.label for obj in ia.rp]
    ia.contours = ia.find_contours(ia.lab, IDs, group=True)
    ia.enlarg_first_call = True
    ia.reduce_first_call = True
    param = store_param(ia)
    return param

def store_app_param(ia):
    app.segm_npy_done[app.frame_i] = ia.lab.copy()
    if app.num_slices > 1:
        app.slices_used[app.frame_i] = int(s_slice.val)
    app.cc_stages[app.frame_i] = ia.cc_stage_df

"""Widgets' functions"""
def next_f(event):
    global param, ia, param
    proceed = True
    if 'lost' in app.warn_txt._text:
        s = app.warn_txt._text
        lost_IDs = s[s.find('['):s.find(']')+1]
        proceed = tk.messagebox._show('Cells disappeared from frame!',
        f'Cells with ID {lost_IDs} disappeared from current frame!\n'
        'Are you sure you want to proceed with next frame?',
        "warning", "yesno") == 'yes'
    if proceed:
        if app.last_tracked_i < app.frame_i+1:
            ia.do_tracking = False
            tracking_state_cb(None)
        app.reset_view = False
        # Store parameters if a previously segmented frame has been modified
        if app.segm_npy_done[app.frame_i] is not None and ia.modified:
            param = store_param(ia)
            store_app_param(ia)
            ia.modified = False
            app.frame_i_done = app.frame_i
        # last frame reached
        if app.frame_i+1 >= num_frames:
            print('You reached the last frame!')
            app.last_segm_i = app.frame_i
            ia.slice_used = app.s
            param = store_param(ia)
            app.save_all()
        # Next frame was already segmented in a previous session. Load data from HDD
        elif app.segm_npy_done[app.frame_i+1] is not None:
            prev_ia = param[app.frame_i]
            param = store_param(ia)
            store_app_param(ia)
            ia.manual_newID_coords = []
            app.frame_i += 1
            param = load_param_from_folder(app)
            # We are visiting the next frame for the first time
            if app.frame_i > app.frame_i_done:
                if app.ROI_delete_coords:
                    for ROI in app.ROI_delete_coords:
                        y_start, y_end, x_start, x_end = ROI
                        lab_ROI = ia.lab[y_start:y_end, x_start:x_end]
                        IDs_delete = np.unique(lab_ROI)
                        for ID in IDs_delete:
                            ia.lab[ia.lab==ID] = 0
                try:
                    ia.lab = ia.auto_separate_bud(ia.lab, ia.rp)
                except:
                    pass
                remove_small_objects(ia.lab, min_size=5, in_place=True)
                ia.rp = regionprops(ia.lab)
                IDs = [obj.label for obj in ia.rp]
                ia.contours = ia.find_contours(ia.lab, IDs, group=True)
                tracked_lab, warn_txt = ia.tracking(prev_ia)
                app.warn_txt._text = warn_txt
                ia.lab = tracked_lab.copy()
                ia.rp = regionprops(tracked_lab)
                app.frame_i_done = app.frame_i
            else:
                tracked_lab, warn_txt = ia.tracking(prev_ia)
                app.warn_txt._text = warn_txt
            # param = store_param(ia)
            # store_app_param(ia)
            app.update_ALLplots(ia)
            app.save_all()
            ia.modified = True
        # Next frame was never segmented
        elif app.frame_i+1 < num_frames and app.frame_i+1 > app.last_segm_i:
            app.last_segm_i = app.frame_i
            ia.slice_used = app.s
            param = store_param(ia)
            app.save_all()
            app.frame_i += 1
            app.init_attr()
            img = app.get_img(app.frames, app.frame_i,
                              app.num_slices, slice=app.s)
            ia.init_attr(img)
            app.frame_txt._text = f'Current frame = {app.frame_i}/{num_frames-1}'
            analyse_img(img)
        # Next frame was already segmented within this session. Load data
        elif app.frame_i+1 <= app.last_segm_i:
            param = store_param(ia)
            app.save_all()
            app.frame_i += 1
            ia = load_param(param)
            app.frame_txt._text = f'Current frame = {app.frame_i}/{num_frames-1}'
            app.update_ALLplots(ia)
        if ia.clos_mode == 'Auto closing':
            man_clos_cb(None)
        s_slice.set_val(app.s, silent=True)
        view_slices_sl.set_val(app.s, silent=True)
        app.update_cells_slideshow(ia)
        app.fig.canvas.draw_idle()
        app.connect_axes_cb()
        app.set_orig_lims()
        app.prev_states = []
        app.set_state(ia)

def prev_f(event):
    global ia, param
    app.reset_view = False
    if app.segm_npy_done[app.frame_i] is not None and ia.modified:
        param = store_param(ia)
        store_app_param(ia)
    if app.segm_npy_done[app.frame_i-1] is not None and app.frame_i-1 >= 0:
        app.frame_i -= 1
        prev_ia = param[app.frame_i-1]
        param = load_param_from_folder(app)
        if prev_ia is not None:
            prev_IDs = [obj.label for obj in prev_ia.rp]
            curr_IDs = [obj.label for obj in param[app.frame_i].rp]
            warn_txt = ia.check_prev_IDs_lost_new(prev_IDs, curr_IDs)
            app.warn_txt._text = warn_txt
        app.update_ALLplots(ia)
    elif app.frame_i-1 >= 0:
        app.frame_i -= 1
        ia = load_param(param)
        if app.num_slices > 1:
            s_slice.set_val(ia.slice_used, silent=True)
        app.frame_txt._text = 'Current frame = {}/{}'.format(app.frame_i,
                                                              num_frames-1)
        app.update_ALLplots(ia)
    else:
        print('You reached the first frame!')
    app.connect_axes_cb()
    app.set_orig_lims()
    app.update_cells_slideshow(ia)

def update_segm(val):
    ia.locTval = s_lowT.val - s_locT.val
    thresh = ia.thresholding(ia.edge, s_lowT.val, s_highT.val)
    ia.repeat_manual_func(thresh)
    lab, rp = ia.segmentation(ia.thresh)
    app.update_ax2_plot(ia, ia.thresh)
    app.update_LABplot(lab, rp, ia)

def plus_locT_cb(event):
    current_valmax = s_locT.valmax
    s_locT.valmax = current_valmax+5
    s_locT.ax.set_ylim(s_locT.valmin, s_locT.valmax)
    app.fig.canvas.draw_idle()

def locT_cb(val):
    locTval = s_lowT.val - s_locT.val
    ia.locTval = locTval
    ia.local_thresh(ia.thresh, app=app)

def del_locT_cb(event):
    ia.thresh = ia.thresholding(ia.edge, s_lowT.val, s_highT.val)
    del ia.li_Line2_rc[-1]
    ia.locT_img = np.zeros_like(ia.thresh)
    for poly in ia.li_Line2_rc:
        for r0, c0, r1, c1 in poly:
            ia.locT_img[line(r0, c0, r1, c1)] = True
            ia.locT_img = binary_fill_holes(ia.locT_img)
    locTval = s_lowT.val - s_locT.val
    ia.locTval = locTval
    ia.local_thresh(ia.thresh, app=app)

def frames_slideshow_cb(event):
    if not ShowWindow_from_title('Cell intensity image slideshow').window_found:
        rps = [[] for _ in range(len(app.frames))]
        CCAdfs = [[] for _ in range(len(app.frames))]
        j = 0
        for i in range(len(app.frames)):
            if i >= 0 and j < len(param):
                if param[j] is not None:
                    rps[i] = param[j].rp
                    CCAdfs[i] = param[j].cc_stage_df
                j += 1
            elif i < 0:
                rps[i] = regionprops(app.segm_npy_done[i])
        rps[app.frame_i] = regionprops(ia.lab)
        if app.num_slices > 1:
            app.cells_slideshow = CellInt_slideshow(app.frames,
                                                    int(view_slices_sl.val),
                                                    num_frames,
                                                    app.frame_i,
                                                    CCAdfs, rps, False,
                                                    num_slices=app.num_slices)
            app.cells_slideshow.new_IDs = ia.new_IDs
            app.cells_slideshow.lost_IDs = ia.lost_IDs
            app.cells_slideshow.run()
        else:
            app.cells_slideshow = CellInt_slideshow_2D(app.frames,
                                                       num_frames,
                                                       app.frame_i,
                                                       CCAdfs, rps, False)
            app.cells_slideshow.run()

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


def switch_segm_mode_cb(event):
    if ia.edge_mode:
        switch_segm_mode.color = axcolor
        switch_segm_mode.hovercolor = hover_color
        switch_segm_mode.label._text = 'Switch to contour mode'
        switch_segm_mode.ax.set_facecolor(axcolor)
        (app.fig.canvas).draw_idle()
        ia.edge_mode = False
        app.update_ax2_plot(ia, ia.thresh)
    else:
        switch_segm_mode.color = button_true_color
        switch_segm_mode.hovercolor = button_true_color
        switch_segm_mode.label._text = 'Contour mode ACTIVE'
        switch_segm_mode.ax.set_facecolor(button_true_color)
        (app.fig.canvas).draw_idle()
        ia.edge_mode = True
        app.update_ax2_plot(ia)

def switch_segm_mode_button(value):
    if value:
        switch_segm_mode.color = button_true_color
        switch_segm_mode.hovercolor = button_true_color
        switch_segm_mode.label._text = 'Contour mode ACTIVE'
        switch_segm_mode.ax.set_facecolor(button_true_color)
        (app.fig.canvas).draw_idle()
    else:
        switch_segm_mode.color = axcolor
        switch_segm_mode.hovercolor = hover_color
        switch_segm_mode.label._text = 'Switch to contour mode'
        switch_segm_mode.ax.set_facecolor(axcolor)
        (app.fig.canvas).draw_idle()


def undo_auto_cont_cb(event):
    del ia.li_yx_dir_coords[-1]
    ia.auto_edge_img = np.zeros_like(ia.auto_edge_img)
    for cont in ia.li_yx_dir_coords:
        for y, x in cont:
            ia.auto_edge_img[y, x] = True
    ia.auto_edge_img = binary_fill_holes(ia.auto_edge_img)
    app.update_ax2_plot(ia, ia.thresh)
    lab, rp = ia.separate_overlap(label(ia.auto_edge_img))
    app.update_LABplot(lab, rp, ia)

def radio_b_ccStage_cb(label):
    if label == 'IDs and contours':
        app.display_IDs_cont = 'IDs and contours'
    elif label == 'Only contours':
        app.display_IDs_cont = 'Only contours'
    else:
        app.display_IDs_cont = 'Disable'
    app.update_LABplot(ia.lab, ia.rp, ia)

def s_slice_cb(val):
    app.s = int(val)
    ia.slice_used = int(val)
    view_slices_sl.set_val(val, silent=True)
    ia.img = app.get_img(frames, app.frame_i, num_slices, slice=int(val))
    ia.edge = ia.edge_detector(ia.img)
    repeat_segm_cb(None)
    repeat_tracking_cb(None)
    ia.modified = True

def save_cb(event):
    global ia, param
    save_current = tk.messagebox.askyesno('Save current position',
                    'Do you want to save currently displayed position?')
    print('Saving...')
    if app.num_slices > 1:
        ia.slice_used = app.s
    if save_current:
        app.last_segm_i = app.frame_i
        app.segm_npy_done[app.frame_i] = ia.lab.copy()
    else:
        app.last_segm_i = app.frame_i-1
        if app.num_slices > 1:
            ia.slice_used = app.s
    app.auto_save = True
    app.save_all()
    print('Saved!')
    app.fig.canvas.mpl_disconnect(cid_close)
    plt.close('all')
    try:
        app.cells_slideshow.close()
    except:
        pass

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

def view_slice(val):
    img = app.get_img(frames, app.frame_i, num_slices, slice=int(val))
    app.ax[0].clear()
    app.update_IMGplot(ia, img, app.ax[0])
    app.fig.canvas.draw_idle()

def reload_segmentation_cb(event):
    global param
    if app.is_undo or app.is_redo:
        app.set_state(ia)
    img = app.get_img(frames, app.frame_i, num_slices, slice=app.s)
    ia.init_attr(img)
    orig_segm_npy = app.orig_segm_npy[app.frame_i].copy()
    app.segm_npy_done[app.frame_i] = orig_segm_npy
    param = load_param_from_folder(app)
    app.update_ALLplots(ia)
    app.set_state(ia)

def auto_save_cb(value):
    if not app.auto_save:
        autosave_b.color = button_true_color
        autosave_b.hovercolor = button_true_color
        autosave_b.label._text = 'Auto-saving ACTIVE'
        autosave_b.ax.set_facecolor(button_true_color)
        (app.fig.canvas).draw_idle()
        app.auto_save = True
    else:
        autosave_b.color = axcolor
        autosave_b.hovercolor = hover_color
        autosave_b.label._text = 'Auto-saving'
        autosave_b.ax.set_facecolor(axcolor)
        (app.fig.canvas).draw_idle()
        app.auto_save = False

def repeat_tracking_cb(event):
    if app.is_undo or app.is_redo:
        app.set_state(ia)
    temporarily_activate_tracking = not ia.do_tracking
    if temporarily_activate_tracking:
        ia.do_tracking = True
    tracking_routine(app, ia)
    app.update_LABplot(ia.lab, ia.rp, ia)
    if temporarily_activate_tracking:
        ia.do_tracking = False
    app.set_state(ia)

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
        app.set_state(ia)
    # Initialize on first call
    if event.inaxes == ax_enlarge_cells:
        morph_func = dilation
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
    ia.lab = morph_func(ia.orig_lab, selem=disk(ia.morph_disk_radius))
    # Reinsert not dilated/eroded cells
    for ID_mask_not_apply, idx in zip(lab_not_apply, lab_not_apply_idx):
        ia.lab[idx] = ID_mask_not_apply
        ia.orig_lab[idx] = ID_mask_not_apply
    ia.rp = regionprops(ia.lab)
    IDs = [obj.label for obj in ia.rp]
    ia.contours = ia.find_contours(ia.lab, IDs, group=True)
    app.update_LABplot(ia.lab, ia.rp, ia)
    ia.modified = True
    app.set_state(ia)

def morph_IDs_on_submit(text):
    ia.orig_lab = ia.lab.copy()
    ia.morph_disk_radius = 1

def tracking_state_cb(event):
    if event is not None:
        app.last_tracked_i = len(app.frames)-1
    if ia.do_tracking:
        track_b.color = axcolor
        track_b.hovercolor = hover_color
        track_b.label._text = 'Tracking is DISABLED'
        track_b.ax.set_facecolor(axcolor)
        (app.fig.canvas).draw_idle()
        ia.do_tracking = False
    else:
        track_b.color = button_true_color
        track_b.hovercolor = button_true_color
        track_b.label._text = 'Tracking is ENABLED'
        track_b.ax.set_facecolor(button_true_color)
        (app.fig.canvas).draw_idle()
        ia.do_tracking = True

def repeat_segm_cb(event):
    if app.unet_first_call:
        from YeaZ.unet import neural_network as nn
        app.nn = nn
        app.unet_first_call = False
    nn = app.nn
    path_weights = nn.determine_path_weights()
    start_t = time()
    img = equalize_adapthist(ia.img)
    img = img*1.0
    print('Neural network is thinking...')
    pred = nn.prediction(img, is_pc=True, path_weights=path_weights)
    thresh = nn.threshold(pred)
    lab = segment.segment(thresh, pred, min_distance=5).astype(int)
    stop_t = time()
    print('Neural network execution time: {0:.3f}'.format(stop_t-start_t))
    ia.lab = lab.astype(int)
    ia.rp = regionprops(lab)
    IDs = [obj.label for obj in ia.rp]
    ia.contours = ia.find_contours(ia.lab, IDs, group=True)
    ia.auto_edge_img = np.zeros(ia.img.shape, bool)
    ia.manual_mask = np.zeros(img.shape, bool)
    ia.contour_plot = [[], []]
    app.update_ALLplots(ia)

def overlay_cb(event):
    if app.do_overlay:
        overlay_b.color = axcolor
        overlay_b.hovercolor = hover_color
        overlay_b.label._text = 'Overlay'
        overlay_b.ax.set_facecolor(axcolor)
        app.fig.canvas.draw_idle()
        app.do_overlay = False
    else:
        ol_path = file_dialog(title='Select image file to overlay',
                              initialdir=app.parent_path)
        if ol_path != '':
            fig, ax = app.fig, app.ax
            app.do_overlay = True
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
                app.ol_frames = np.load(ol_path)
                if filename.find('aligned') != -1:
                    align_ol = False
                else:
                    align_ol = True
            elif ext == '.tif' or ext == '.tiff':
                align_ol = True
                app.ol_frames = io.imread(ol_path)
            else:
                messagebox.showerror('File Format not supported!',
                    f'File format {ext} is not supported!\n'
                    'Choose either .tif or .npy files.')
            if align_ol:
                loaded_shifts, shifts_found = load_shifts(images_path)
                if shifts_found:
                    print('Aligning overlay image frames...')
                    is_3D = app.num_slices > 1
                    align_func = align_frames_3D if is_3D else align_frames_2D
                    aligned_frames, shifts = align_func(app.ol_frames,
                                                      slices=None,
                                                      register=False,
                                                      user_shifts=loaded_shifts)
                    aligned_filename = f'{filename_noEXT}_aligned.npy'
                    aligned_path = f'{images_path}/{aligned_filename}'
                    np.save(aligned_path, aligned_frames, allow_pickle=False)
                    print('Overlay image frames aligned!')
                    app.ol_frames = aligned_frames
                else:
                    messagebox.showerror('Shifts file not found!',
                        f'\"..._align_shift.npy\" file not found!\n'
                        'Overlay images cannot be aligned to the cells image.')
                    raise FileNotFoundError('Shifts file not found!')
    app.update_IMGplot(ia, ia.img, app.ax[0], clear=True)
    app.set_lims()
    fig.canvas.draw_idle()
    app.connect_axes_cb()

def update_overlay_cb(event):
    if app.do_overlay:
        ol_img = app.get_img(app.ol_frames, app.frame_i, num_slices,
                             slice=app.s)
        fig, ax = app.fig, app.ax
        app.update_IMGplot(ia, ia.img, app.ax[0])
        app.set_lims()
        fig.canvas.draw_idle()
        app.connect_axes_cb()
    else:
        tk.messagebox.showwarning('Overlay not active', 'Brightness slider, '
            'alpha slider and the vertical color picker all control the '
            'overlay appearance.\n To use them you first need to press on the'
            '"Overlay" button and choose an image to overlay '
            '(typically a fluorescent signal)')


def rgb_cmap_cb(event):
    global ol_RGB_val
    update_overlay_cb(event)

autosave_b.on_clicked(auto_save_cb)
next_b.on_clicked(next_f)
prev_b.on_clicked(prev_f)
reload_segm_b.on_clicked(reload_segmentation_cb)
# s_lowT.on_changed(update_segm)
# s_highT.on_changed(update_segm)
# plus_locT.on_clicked(plus_locT_cb)
# s_locT.on_changed(locT_cb)
# del_locT.on_clicked(del_locT_cb)
frames_slideshow.on_clicked(frames_slideshow_cb)
# undo_del_t.on_clicked(undo_del_t_cb)
man_clos.on_clicked(man_clos_cb)
switch_segm_mode.on_clicked(switch_segm_mode_cb)
undo_auto_cont.on_clicked(undo_auto_cont_cb)
radio_b_ccStage.on_clicked(radio_b_ccStage_cb)
s_slice.on_changed(s_slice_cb)
save_b.on_clicked(save_cb)
view_slices_sl.on_changed(view_slice)
repeat_tracking_b.on_clicked(repeat_tracking_cb)
reduce_cells.on_clicked(apply_morph_cells)
enlarge_cells.on_clicked(apply_morph_cells)
morhp_ids_tb.on_submit(morph_IDs_on_submit)
track_b.on_clicked(tracking_state_cb)
repeat_segm_b.on_clicked(repeat_segm_cb)
overlay_b.on_clicked(overlay_cb)
brightness_slider.on_changed(update_overlay_cb)
alpha_slider.on_changed(update_overlay_cb)


def get_font_size_from_winsize(fig_size_pixels):
    w, h = fig_size_pixels
    return w/1920*11

"""Canvas events functions"""
def resize_widgets(event):
    # [left, bottom, width, height]
    fig = app.fig
    # Resize font size according to figure window size
    fig_size_pixels = fig.get_size_inches()*fig.dpi
    font_size = get_font_size_from_winsize(fig_size_pixels)
    [rb.set_fontsize(font_size) for rb in radio_b_ccStage.labels]
    [button.label.set_fontsize(font_size) for button in buttons]
    morhp_ids_tb.label.set_fontsize(font_size)
    ax = app.ax
    ax0_l, ax0_b, ax0_r, ax0_t = ax[0].get_position().get_points().flatten()
    ax1_l, ax1_b, ax1_r, ax1_t = ax[1].get_position().get_points().flatten()
    ax2_l, ax2_b, ax2_r, ax2_t = ax[2].get_position().get_points().flatten()
    slW = ax2_r-bW-ax0_l-spF
    # ax_lowT.set_position([ax0_l, ax0_b-2*spF-wH, slW, wH])
    # ax_highT.set_position([ax0_l, ax0_b-2*spF-2*wH-(spF/2), slW, wH])
    ax_prev.set_position([ax2_r-bW, ax0_b-2*spF-wH, bW/2, wH])
    ax_next.set_position([ax2_r-bW/2, ax0_b-2*spF-wH, bW/2, wH])
    a1_c = ax1_l+((ax1_r-ax1_l)/2)
    ax_reduce_cells.set_position([a1_c-bW-(spF/11), ax1_b-(spF/11)-wH, bW, wH])
    ax_enlarge_cells.set_position([a1_c+(spF/11), ax1_b-(spF/11)-wH, bW, wH])
    _, ax_enl_b, ax_enl_r, _ = ax_enlarge_cells.get_position().get_points().flatten()
    ax_morph_IDs.set_position([ax_enl_r-bW, ax_enl_b-(spF/9)-wH, bW, wH])
    _, ax_next_b, _, _ = ax_next.get_position().get_points().flatten()
    ax_autosave.set_position([ax2_r-bW, ax_next_b+wH+(spF/3), bW, wH])
    # ax_locT.set_position([ax2_r+(spF/3), ax2_b, 0.02, ax2_t-ax2_b])
    # locT_l, locT_b, locT_r, locT_t = ax_locT.get_position().get_points().flatten()
    # ax_plus_locT.set_position([locT_r+(spF/5), locT_t-(wH*1.5), bW/3, wH*1.5])
    # ax_del_locT.set_position([ax2_l, ax2_t+(spF/3), bW, wH])
    ax_frames_slideshow.set_position([ax1_l, ax1_t+(spF/3), bW, wH])
    ax_man_clos.set_position([ax2_l, ax2_t+(spF/3), bW, wH])
    ax_switch_to_edge.set_position([ax1_r-bW, ax1_t+(spF/3), bW, wH])
    udtL, udtb, udtR, udtT = ax_man_clos.get_position().get_points().flatten()
    uaeW = ax2_r-udtR-(spF/11)
    ax_undo_auto_edge.set_position([ax2_r-bW, ax2_t+(spF/3), bW, wH])
    ax_ccstage_radio.set_position([ax0_l, ax0_t+(spF/11), ax0_r-ax0_l, wH])
    _, _, _, radio_T = ax_ccstage_radio.get_position().get_points().flatten()
    ax_overlay_L = (ax0_l+((ax0_r-ax0_l)/2))-bW/2
    ax_overlay.set_position([ax_overlay_L, radio_T+0.01, bW, wH])
    ax_slice.set_position([ax2_l, ax2_b-(spF/11)-wH, ax2_r-ax2_l, wH])
    ax_reload_segm_b = ax_next_b-wH-(spF/3)
    ax_reload_segm.set_position([ax2_r-2*bW-(spF/11), ax_reload_segm_b, bW, wH])
    ax_track.set_position([ax2_r-2*bW-(spF/11), ax_next_b+wH+(spF/3), bW, wH])
    ax_retrack.set_position([ax2_r-bW, ax_reload_segm_b, bW, wH])
    ax_save_b = ax_reload_segm_b-wH-(spF/3)
    ax_save.set_position([ax2_r-bW, ax_save_b, bW, wH])
    ax_repeat_segm.set_position([ax2_r-2*bW-(spF/11), ax_save_b, bW, wH])
    ax_view_slices.set_position([ax0_l, ax0_b-(spF/11)-wH, ax0_r-ax0_l, wH])

    sl_h = 0.025
    ax_bright_sl.set_position([ax0_l, ax0_b-0.007-sl_h, ax0_r-ax0_l, sl_h])
    ax_alpha_sl.set_position([ax0_l, ax0_b-2*(0.007+sl_h), ax0_r-ax0_l, sl_h])
    ax_rgb.set_position([ax0_r+0.005, ax0_b, sl_h*2/3, ax0_t-ax0_b])

    app.connect_axes_cb()
    if app.num_slices <= 1:
        ax_view_slices.set_visible(False)
        ax_slice.set_visible(False)
    for axes in app.ax:
        on_xlim_changed(axes)
        on_ylim_changed(axes)

def key_down(event):
    global ia
    key = event.key
    if key == 'right':
        if app.key_mode == 'Z-slice' and app.num_slices>1:
            s_slice.set_val(s_slice.val+1)
        elif app.key_mode == 'view_slice' and app.num_slices>1:
            view_slices_sl.set_val(view_slices_sl.val+1)
        else:
            next_f(None)
    elif key == 'left':
        if app.key_mode == 'Z-slice' and app.num_slices>1:
            s_slice.set_val(s_slice.val-1)
        elif app.key_mode == 'view_slice' and app.num_slices>1:
            view_slices_sl.set_val(view_slices_sl.val-1)
        else:
            prev_f(None)
    elif key == 'ctrl+p':
        print(ia.cc_stage_df)
        print(app.ax_limits)
        print(app.orig_ax_limits)
        print(app.frame_i_done)
    elif key == 'b':
        ia.sep_bud = True
    elif key == 'm':
        ia.set_moth = True
    elif key == 'c':
        app.do_cut = True
    elif key == 'd':
        app.draw_ROI_delete = True
    elif key == 'ctrl+z':
        app.is_undo = True
        if app.count_states-1 >= 0:
            app.count_states -= 1
            ia = app.get_state(ia, app.count_states)
            app.warn_txt._text = ia.warn_txt_text
            app.update_ALLplots(ia)
        else:
            print('There are no previous states to restore!')
    elif key == 'ctrl+y':
        app.is_redo = True
        if app.count_states+1 < len(app.prev_states):
            app.count_states += 1
            ia = app.get_state(ia, app.count_states)
            app.warn_txt._text = ia.warn_txt_text
            app.update_ALLplots(ia)
        else:
            print('No next states to restore!')

def key_up(event):
    key = event.key
    if key == 'b':
        ia.sep_bud = False
    elif key == 'm':
        ia.set_moth = False
    elif key == 'd':
        app.draw_ROI_delete = False
    elif key == 'c':
        app.do_cut = False

def mouse_down(event):
    right_click = event.button == 3
    left_click = event.button == 1
    wheel_click = event.button == 2
    ax0_click = event.inaxes == app.ax[0]
    ax1_click = event.inaxes == app.ax[1]
    ax2_click = event.inaxes == app.ax[2]
    ax_click = any([ax0_click, ax1_click, ax2_click])
    not_ax = not ax_click
    app.connect_axes_cb()
    if event.xdata:
        xd = int(round(event.xdata))
        yd = int(round(event.ydata))
    # Zoom with double click on ax2
    if left_click and ax2_click and event.dblclick:
        # app.fig.canvas.toolbar.zoom()
        mean_major_ax_length = np.mean([obj.major_axis_length for obj in ia.rp])
        y, x = ia.img.shape
        xlim_left, xlim_right = app.ax[2].get_xlim()
        ylim_bottom, ylim_top = app.ax[2].get_ylim()
        xrange = xlim_right-xlim_left
        yrange = ylim_top-ylim_bottom
        zoom_factor_x = mean_major_ax_length/(x/xrange)
        zoom_factor_y = mean_major_ax_length/(y/yrange)
        app.ax[2].set_xlim((xd-zoom_factor_x, xd+zoom_factor_x))
        app.ax[2].set_ylim((yd-zoom_factor_y, yd+zoom_factor_y))
        app.ax[1].set_xlim((xd-zoom_factor_x, xd+zoom_factor_x))
        app.ax[1].set_ylim((yd-zoom_factor_y, yd+zoom_factor_y))
        app.fig.canvas.draw_idle()
    # Zoom with double click on ax0
    if left_click and ax0_click and event.dblclick:
        # app.fig.canvas.toolbar.zoom()
        nearest_ID = ia.nearest_nonzero(ia.lab, yd, xd)
        IDs = [obj.label for obj in ia.rp]
        nearest_idx = IDs.index(nearest_ID)
        min_row, min_col, max_row, max_col = ia.rp[nearest_idx].bbox
        nearest_bbox_h = max_row - min_row
        nearest_bbox_w = max_col - min_col
        nearest_bbox_cy = min_row + nearest_bbox_h/2
        nearest_bbox_cx = min_col + nearest_bbox_w/2
        nearest_bottom = int(nearest_bbox_cy - nearest_bbox_h/2)
        nearest_left = int(nearest_bbox_cx - nearest_bbox_w/2)
        nearest_top = nearest_bottom + nearest_bbox_h
        nearest_right = nearest_left + nearest_bbox_w
        app.ax[0].set_xlim(nearest_left-5, nearest_right+5)
        app.ax[0].set_ylim(nearest_top+5, nearest_bottom-5)
        app.fig.canvas.draw_idle()
    # Manual correction: replace cell with hull contour
    if left_click and ax1_click and event.dblclick:
        ID = ia.lab[yd, xd]
        if ID != 0:
            if app.is_undo or app.is_redo:
                app.set_state(ia)
            labels_onlyID = np.zeros(ia.lab.shape, bool)
            labels_onlyID[ia.lab == ID] = True
            hull_img_ID = convex_hull_image(labels_onlyID)
            ia.lab[hull_img_ID] = ID
            ia.rp = regionprops(ia.lab)
            IDs = [obj.label for obj in ia.rp]
            ia.contours = ia.find_contours(ia.lab, IDs, group=True)
            app.update_ax2_plot(ia)
            app.update_LABplot(ia.lab, ia.rp, ia)
            ia.modified = True
            app.set_state(ia)
    # Zoom out to home view
    if right_click and event.dblclick and not_ax:
        for a, axes in enumerate(app.ax):
            app.ax_limits = deepcopy(ia.home_ax_limits)
            axes.set_xlim(*ia.home_ax_limits[a][0])
            axes.set_ylim(*ia.home_ax_limits[a][1])
        app.fig.canvas.draw_idle()
    if ax_click:
        ia.modified = True
    if right_click and ax2_click and not ia.edge_mode:
        cid2_rc = app.fig.canvas.mpl_connect('motion_notify_event', mouse_drag)
        app.cid2_rc = cid2_rc
        app.xdrc = xd
        app.ydrc = yd
        app.xdrrc = xd
        app.ydrrc = yd
        ia.Line_Poly = []
    elif right_click and ax2_click and ia.edge_mode and ia.clos_mode == 'Auto closing':
        app.xdrc = xd
        app.ydrc = yd
    elif right_click and ax2_click and ia.edge_mode and ia.clos_mode == 'Manual closing':
        cid2_rc = app.fig.canvas.mpl_connect('motion_notify_event', mouse_drag)
        app.cid2_rc = cid2_rc
        ia.init_manual_cont(app, xd, yd)
    elif wheel_click and ax2_click and not ia.edge_mode:
        lab_thresh = label(ia.thresh, connectivity=1)
        ID = lab_thresh[yd, xd]
        if ID != 0:
            ia.del_yx_thresh.append((yd, xd))
            ia.li_manual_func.append('Del. yx thresh')
            ia.thresh[lab_thresh == ID] = 0
            lab, rp = ia.segmentation(ia.thresh)
            app.update_ax2_plot(ia, ia.thresh)
            app.update_LABplot(lab, rp, ia)
            ia.modified = True
    elif right_click and ax1_click and not ia.sep_bud and not ia.set_moth and not event.dblclick:
        ia.merge_yx[0] = yd
        ia.merge_yx[1] = xd
    # Manual correction: separate emerging bud
    elif right_click and ax1_click and ia.sep_bud:
        ID = ia.lab[yd, xd]
        meb = manual_emerg_bud(ia.lab, ID, ia.rp)
        ia.sep_bud = False
        if not meb.cancel:
            if app.is_undo or app.is_redo:
                app.set_state(ia)
            meb_lab = remove_small_objects(meb.sep_bud_label,
                                             min_size=20,
                                             connectivity=2)
            ia.lab[meb_lab != 0] = meb_lab[meb_lab != 0]
            ia.lab[meb.small_obj_mask] = 0
            for y, x in meb.coords_delete:
                del_ID = ia.lab[y, x]
                ia.lab[ia.lab == del_ID] = 0
            # ia.lab[meb.rr, meb.cc] = 0  # separate bud with 0s
            rp = regionprops(ia.lab)
            IDs = [obj.label for obj in rp]
            ia.contours = ia.find_contours(ia.lab, IDs, group=True)
            ia.rp = rp
            tracking_routine(app, ia)
            app.update_LABplot(ia.lab, ia.rp, ia)
            app.update_ax2_plot(ia)
            ia.modified = True
            app.set_state(ia)
            app.update_cells_slideshow(ia)
    # Manual correction: replace old ID and assign new ID
    elif wheel_click and ax0_click:
        old_ID = ia.lab[yd, xd]
        if old_ID == 0:
            tk.messagebox.showwarning(title='WARNING: Clicked on background',
                        message='You accidentally clicked on the background. '
                                'You cannot assign a new ID to the background')
        else:
            if app.is_undo or app.is_redo:
                app.set_state(ia)
            prev_ia = param[app.frame_i-1]
            new_ID = newID_app(old_ID=old_ID).new_ID
            IDs = [obj.label for obj in ia.rp]
            if new_ID in IDs:
                tempID = ia.lab.max() + 1
                ia.lab[ia.lab == old_ID] = tempID
                ia.lab[ia.lab == new_ID] = old_ID
                ia.lab[ia.lab == tempID] = new_ID
            else:
                ia.lab[ia.lab == old_ID] = new_ID
            ia.rp = regionprops(ia.lab)
            if prev_ia is not None:
                prev_IDs = [obj.label for obj in prev_ia.rp]
                curr_IDs = [obj.label for obj in ia.rp]
                warn_txt = ia.check_prev_IDs_lost_new(prev_IDs, curr_IDs)
                app.warn_txt._text = warn_txt
            app.update_LABplot(ia.lab, ia.rp, ia)
            ia.modified = True
            ia.manual_newID_coords.append((yd, xd, new_ID))
            app.set_state(ia)
            app.update_cells_slideshow(ia)
    elif right_click and ax0_click:
        app.xdrc = xd
        app.ydrc = yd
    # Manual correction: delete ID
    elif wheel_click and ax1_click and not app.draw_ROI_delete and not event.dblclick:
        if app.is_undo or app.is_redo:
            app.set_state(ia)
        prev_ia = param[app.frame_i-1]
        ID = ia.lab[yd, xd]
        ia.lab[ia.lab == ID] = 0
        ia.rp = regionprops(ia.lab)
        IDs = [obj.label for obj in ia.rp]
        ia.contours = ia.find_contours(ia.lab, IDs, group=True)
        if prev_ia is not None:
            prev_IDs = [obj.label for obj in prev_ia.rp]
            curr_IDs = [obj.label for obj in ia.rp]
            warn_txt = ia.check_prev_IDs_lost_new(prev_IDs, curr_IDs)
            app.warn_txt._text = warn_txt
        app.update_LABplot(ia.lab, ia.rp, ia)
        app.update_ax2_plot(ia)
        ia.modified = True
        app.set_state(ia)
        app.update_cells_slideshow(ia)
    elif wheel_click and ax1_click and app.draw_ROI_delete:
        app.ROI_delete_patch = Rectangle((0, 0), 1, 1, fill=False, color='r')
        app.cid_ROI_delete = app.fig.canvas.mpl_connect('motion_notify_event',
                                                        mouse_drag)
        app.xdwc_ax1 = xd
        app.ydwc_ax1 = yd
    # Manual correction: Delete contour pixel
    elif wheel_click and ax2_click and ia.edge_mode:
        if app.is_undo or app.is_redo:
            app.set_state(ia)
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
        app.update_ax2_plot(ia)
        app.update_LABplot(lab, rp, ia)
        ia.modified = True
        app.set_state(ia)
    elif wheel_click and ax1_click and event.dblclick:
        app.ax[1].patches = []
        app.ROI_delete_coords = []
        app.fig.canvas.draw_idle()
    # Manual correction: enforce automatic bud separation
    elif right_click and ax1_click and event.dblclick:
        ID = ia.lab[yd, xd]
        if ID != 0:
            if app.is_undo or app.is_redo:
                app.set_state(ia)
            prev_ia = param[app.frame_i-1]
            max_ID = ia.lab.max()
            ia.lab = ia.auto_separate_bud_ID(ID, ia.lab, ia.rp, max_ID,
                                             enforce=True)
            ia.rp = regionprops(ia.lab)
            IDs = [obj.label for obj in ia.rp]
            ia.contours = ia.find_contours(ia.lab, IDs, group=True)
            if prev_ia is not None:
                tracked_lab, warn_txt = ia.tracking(prev_ia)
                app.warn_txt._text = warn_txt
                ia.lab = tracked_lab.copy()
                ia.rp = regionprops(tracked_lab)
            app.update_ax2_plot(ia)
            app.update_LABplot(ia.lab, ia.rp, ia)
            app.set_state(ia)
            app.update_cells_slideshow(ia)
    # Change RGB color of the overlay
    if left_click and event.inaxes == ax_rgb and app.do_overlay:
        app.picked_rgb_marker.remove()
        y_rgb = int(round(event.ydata))
        app.ol_RGB_val = app.rgb_cmap_array[y_rgb][:3]
        app.picked_rgb_marker = ax_rgb.scatter(app.x_rgb, y_rgb, marker='s',
                                               color='k')
        update_overlay_cb(event)


def mouse_drag(event):
    right_click = event.button == 3
    left_click = event.button == 1
    wheel_click = event.button == 2
    ax0_click = event.inaxes == app.ax[0]
    ax1_click = event.inaxes == app.ax[1]
    ax2_click = event.inaxes == app.ax[2]
    if event.xdata:
        xdr = int(round(event.xdata))
        ydr = int(round(event.ydata))
    if right_click and ax2_click and not ia.edge_mode:
        app.Line2_rc = Line2D([app.xdrrc, xdr], [app.ydrrc, ydr], color='r')
        app.ax[2].add_line(app.Line2_rc)
        app.fig.canvas.draw_idle()
        r0, c0, r1, c1 = app.ydrrc, app.xdrrc, ydr, xdr
        ia.Line_Poly.append((r0, c0, r1, c1))
        ia.locT_img[line(r0, c0, r1, c1)] = True
        app.xdrrc = xdr
        app.ydrrc = ydr
    elif right_click and ax2_click and ia.clos_mode == 'Manual closing':
        ia.manual_contour(app, ydr, xdr)
    elif wheel_click and app.draw_ROI_delete:
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


def mouse_up(event):
    right_click = event.button == 3
    left_click = event.button == 1
    wheel_click = event.button == 2
    ax0_click = event.inaxes == app.ax[0]
    ax1_click = event.inaxes == app.ax[1]
    ax2_click = event.inaxes == app.ax[2]
    if event.xdata:
        xu = int(round(event.xdata))
        yu = int(round(event.ydata))
    if right_click and ax2_click and not ia.edge_mode:
        app.fig.canvas.mpl_disconnect(app.cid2_rc)
        app.Line2_rc = Line2D([xu, app.xdrc], [yu, app.ydrc], color='r')
        app.ax[2].add_line(app.Line2_rc)
        app.fig.canvas.draw_idle()
        r0, c0, r1, c1 = yu, xu, app.ydrc, app.xdrc
        ia.Line_Poly.append((r0, c0, r1, c1))
        ia.li_Line2_rc.append(ia.Line_Poly)
        ia.li_manual_func.append('Local T.')
        ia.locT_img[line(r0, c0, r1, c1)] = True
        ia.locT_img = binary_fill_holes(ia.locT_img)
        locTval = s_lowT.val - s_locT.val
        ia.locTval = locTval
        ia.local_thresh(ia.thresh, app=app)
    # Manual correction: add automatically drawn contour
    elif right_click and ax2_click and ia.edge_mode and ia.clos_mode == 'Auto closing':
        if app.is_undo or app.is_redo:
            app.set_state(ia)
        app.xu2rc = xu
        app.yu2rc = yu
        lab, rp = ia.auto_contour(app)
        lab[lab>0] += ia.lab.max() + 1
        lab_mask = ia.lab>0
        lab[lab_mask] = ia.lab[lab_mask]
        tracking_routine(app, ia)
        app.update_LABplot(ia.lab, ia.rp, ia)
        ia.modified = True
        app.set_state(ia)
    # Manual correction: add manually drawn contour
    elif right_click and ax2_click and ia.edge_mode and ia.clos_mode == 'Manual closing':
        if app.is_undo or app.is_redo:
            app.set_state(ia)
        app.fig.canvas.mpl_disconnect(app.cid2_rc)
        IDs = [obj.label for obj in ia.rp]
        lab, rp = ia.close_manual_cont()
        # Get IDs overlapping with currently manually drawn object
        cut_IDs = np.unique(ia.lab[lab>0])
        # Remove objects that overlap with currently manually drawn object
        ia.lab[lab>0] = 0
        if app.do_cut:
            lab_only_cut_IDs = ia.lab.copy()
            for ID in cut_IDs:
                if ID != 0:
                    lab_only_cut_IDs[lab_only_cut_IDs != ID] = 0
            lab_only_cut_IDs = label(lab_only_cut_IDs, connectivity=1)
            max_ID = max(IDs)
            lab_only_cut_IDs *= (max_ID+1)
            ia.lab[lab_only_cut_IDs>0] = lab_only_cut_IDs[lab_only_cut_IDs>0]
            remove_small_objects(ia.lab, min_size=5, in_place=True)
            app.do_cut = False
        else:
            # Assign to new object a new ID higher than all existing IDs
            lab[lab>0] = ia.lab.max() + 1
            # Add to manual object all previous objects
            lab_mask = ia.lab>0
            lab[lab_mask] = ia.lab[lab_mask]
            ia.lab = lab.copy()
        ia.rp = regionprops(ia.lab)
        tracking_routine(app, ia)
        app.update_LABplot(ia.lab, ia.rp, ia)
        ia.modified = True
        ia.auto_edge_img = np.zeros_like(ia.auto_edge_img)
        app.set_state(ia)
        app.update_cells_slideshow(ia)
    # Manual correction: merge IDs
    elif right_click and ax1_click and not ia.sep_bud:
        if app.is_undo or app.is_redo:
            app.set_state(ia)
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
        tracking_routine(app, ia)
        app.update_LABplot(ia.lab, ia.rp, ia)
        app.update_ax2_plot(ia)
        ia.modified = True
        app.set_state(ia)
        app.update_cells_slideshow(ia)
    elif right_click and ax0_click and not ia.set_moth:
        budID = ia.lab[app.ydrc, app.xdrc]
        mothID = ia.lab[yu, xu]
        if mothID == 0 or budID == 0:
            print('WARNING: You clicked (or released) on background! '
                  'No cell cycle stage can be assigned to background!')
        else:
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
                                display_IDs_cont=app.display_IDs_cont,
                                color='r', clear=True)
            app.fig.canvas.draw_idle()
    # Draw ROI rectangle for constant delete
    elif wheel_click and app.draw_ROI_delete:
        if app.is_undo or app.is_redo:
            app.set_state(ia)
        prev_ia = param[app.frame_i-1]
        app.fig.canvas.mpl_disconnect(app.cid_ROI_delete)
        event_x, event_y = event.x, event.y
        ax = app.ax[1]
        xu_wc, yu_wc = app.ax_transData_and_coerce(ax, event_x, event_y,
                                                   ia.img.shape)
        y_start, y_end = sorted([app.ydwc_ax1, yu_wc])
        x_start, x_end = sorted([app.xdwc_ax1, xu_wc])
        app.ROI_delete_coords.append([y_start, y_end, x_start, x_end])
        lab_ROI = ia.lab[y_start:y_end, x_start:x_end]
        IDs_delete = np.unique(lab_ROI)
        for ID in IDs_delete:
            ia.lab[ia.lab==ID] = 0
        rp = regionprops(ia.lab)
        IDs = [obj.label for obj in rp]
        ia.contours = ia.find_contours(ia.lab, IDs, group=True)
        ia.rp = rp
        if prev_ia is not None:
            prev_IDs = [obj.label for obj in prev_ia.rp]
            curr_IDs = [obj.label for obj in ia.rp]
            warn_txt = ia.check_prev_IDs_lost_new(prev_IDs, curr_IDs)
            app.warn_txt._text = warn_txt
        app.update_LABplot(ia.lab, ia.rp, ia)
        app.update_ax2_plot(ia)
        app.draw_ROI_delete = False
        app.set_state(ia)


def axes_enter(event):
    if event.inaxes == ax_slice:
        app.key_mode = 'Z-slice'
    elif event.inaxes == ax_view_slices:
        app.key_mode = 'view_slice'
    # elif event.inaxes == app.ax[2] and ia.clos_mode == 'Manual closing':
    #     app.mng.window.configure(cursor='circle')

def axes_leave(event):
    app.key_mode = ''
    if event.inaxes == app.ax[2] and ia.clos_mode == 'Manual closing':
        try:
            app.mng.window.configure(cursor='arrow')
        except:
            # If the backend is not TkAgg then I still don't know how to change
            # the mouse cursor to arrow
            pass

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
    try:
        app.cells_slideshow.close()
    except:
        pass
    save = tk.messagebox.askyesno('Save', 'Do you want to save segmented data?')
    if save:
        save_cb(None)

def figure_leave(event):
    try:
        ShowWindow_from_title('Cell intensity image slideshow')
    except:
        # traceback.print_exc()
        pass
    try:
        ShowWindow_from_title('Mother-bud zoom')
    except:
        # traceback.print_exc()
        pass


def figure_enter(event):
    try:
        show_cell_int = ShowWindow_from_title('Cell intensity image slideshow')
        cell_int_open = show_cell_int.window_found
    except:
        traceback.print_exc()
        cell_int_open = False
        pass
    if cell_int_open:
        try:
            ShowWindow_from_title('Cell segmentation GUI')
        except:
            # traceback.print_exc()
            pass

# Canvas events
(app.fig.canvas).mpl_connect('button_press_event', mouse_down)
(app.fig.canvas).mpl_connect('button_release_event', mouse_up)
(app.fig.canvas).mpl_connect('key_press_event', key_down)
(app.fig.canvas).mpl_connect('key_release_event', key_up)
(app.fig.canvas).mpl_connect('resize_event', resize_widgets)
(app.fig.canvas).mpl_connect('axes_enter_event', axes_enter)
(app.fig.canvas).mpl_connect('axes_leave_event', axes_leave)
cid_close = (app.fig.canvas).mpl_connect('close_event', handle_close)
(app.fig.canvas).mpl_connect('figure_leave_event', figure_leave)
(app.fig.canvas).mpl_connect('figure_enter_event', figure_enter)
# NOTE: axes limit changed event is connected first time in resize_widgets

app.frame_txt = app.fig.text(0.5, 0.15,
        f'Current frame = {app.frame_i}/{num_frames-1}',
        color='w', ha='center', fontsize=14)

app.warn_txt = app.fig.text(0.5, 0.9, '', color='orangered', ha='center',
                                            fontsize=16)

param = set_param_from_folder(num_frames, app)

if app.segm_npy_done[app.frame_i] is not None:
    param = load_param_from_folder(app)
    switch_segm_mode_cb(None)
    app.update_ALLplots(ia)
else:
    ia.edge_mode = True
    switch_segm_mode_button(True)
    analyse_img(img)
man_clos_cb(None)
app.set_orig_lims()
app.set_state(ia)

# win_size(swap_screen=False)
app.fig.canvas.set_window_title(f'Cell segmentation GUI - {app.exp_name}\\{app.pos_foldername}')
plt.show()
