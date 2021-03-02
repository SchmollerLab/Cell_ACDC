import os
import shutil
import traceback
import re
from time import time
from sys import exit
from copy import deepcopy
from natsort import natsorted
from copy import deepcopy
import cv2
import numpy as np
import pandas as pd
import matplotlib as mpl
from math import atan2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, MyRadioButtons
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle
from tkinter import tk_breakpoint, E, S, W, END
import tkinter as tk
from tkinter.filedialog import folder_dialog
from tifffile.tifffile import TiffWriter, TiffFile
from skimage import io
from skimage.feature import peak_local_max
from skimage.filters import (gaussian, sobel, apply_hysteresis_threshold,
                            threshold_otsu)
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects, convex_hull_image
from skimage.draw import line, line_aa
from scipy.ndimage.morphology import binary_fill_holes, distance_transform_edt
from FUNCTIONS import (auto_select_slice, manual_emerg_bud,
                       separate_overlapping, text_label_centroid, tk_breakpoint,
                       manual_emerg_bud, CellInt_slideshow, beyond_listdir_pos,
                       select_exp_folder)

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
                 text=f'(There is a total of {tot_frames} positio50ns.\n'
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
class init_App:
    def __init__(self, TIFFs_path):
        phc_img_li = []
        self.images_pos_paths = []
        self.pos_paths = []
        self.TIFFs_parent_path = os.path.dirname(TIFFs_path)
        self.exp_name = os.path.basename(self.TIFFs_parent_path)
        directories = natsorted(os.listdir(TIFFs_path))
        self.directories = directories
        self.basenames = []
        self.all_others_tifs_li = [[] for _ in range(len(directories))]
        self.all_others_tifs_li_dict = [[] for _ in range(len(directories))]
        print('Loading all images...')
        for i, d in enumerate(directories):
            pos_path = f'{TIFFs_path}/{d}/Images'
            self.pos_paths.append(f'{TIFFs_path}/{d}')
            self.images_pos_paths.append(pos_path)
            filenames = os.listdir(pos_path)
            p_found = False
            slice_found = False
            for j, p in enumerate(filenames):
                temp_pIDX = p.find('_phase_contr.tif')
                ext = os.path.splitext(p)[1]
                if temp_pIDX != -1:
                    p_idx = temp_pIDX
                    k = j
                    p_found = True
                elif ext == '.tif':
                    self.all_others_tifs_li[i].append(p)
            if p_found:
                phc_img_path = f'{pos_path}/{filenames[k]}'
                with TiffFile(phc_img_path) as tif:
                    self._metadata = tif.imagej_metadata
                    phc_img_pos = tif.asarray()
                phc_img_li.append(phc_img_pos)
                for tif_filename in self.all_others_tifs_li[i]:
                    tif_path = f'{pos_path}/{tif_filename}'
                    with TiffFile(tif_path) as tif:
                        tif_data = tif.asarray()
                        match = re.search('s(\d+)_', tif_filename)
                        basename_idx = match.span()[1]
                        tif_basename = tif_filename[:basename_idx-1]
                        rest = tif_filename[basename_idx:]
                        self.all_others_tifs_li_dict[i].append(
                                             {'tif_basename': tif_basename,
                                              'tif_channelname': rest,
                                              'tif_data': tif_data})
                base_name = p[0:p_idx]
                self.basenames.append(base_name)
            else:
                raise FileNotFoundError('phase_contr.tif file not found in '
                                        f'Position_{i+1} folder')
        self.all_phc_img_li = phc_img_li
        self.num_slices = phc_img_li[0].shape[0]
        self.p = 0
        self.s = int(self.num_slices/2)
        self.counter = 0
        self.sub_V_dict_li = []
        self.sub_V_other_tifs_dict_li = []
        self.sub_V_dict_li_li = [None]*len(directories)
        self.sub_V_other_tifs_dict_li_li = [None]*len(directories)
        self.key_mode = ''
        print(f'Total number of Positions = {len(phc_img_li)}')

    def init_plt_GUI(self):
        fig, ax = plt.subplots()
        plt.subplots_adjust(bottom=0.15)
        self.fig = fig
        self.ax = ax

    def update_plot(self):
        fig, ax = self.fig, self.ax
        app.img = self.all_phc_img_li[self.p][self.s]
        ax.clear()
        ax.imshow(app.img)
        ax.set_title('Right-click on a cell to save it as a separate file\n\n'
                     f'{self.directories[self.p]}')
        ax.axis('off')
        fig.canvas.draw_idle()


    def imagej_tiffwriter(self, new_path, data):
        with TiffWriter(new_path, imagej=True) as new_tif:
            Z, Y, X = data.shape
            data.shape = 1, Z, 1, Y, X, 1  # imageJ format should always have TZCYXS data shape
            new_tif.save(data, metadata=self._metadata)

    def save_sub_img(self, TIFFs_path, p, sub_V_dict_li, sub_V_other_tifs_dict_li):
        orig_TIFFs_path = f'{self.TIFFs_parent_path}/Original_TIFFs'
        orig_TIFFs_pos_path = (f'{orig_TIFFs_path}/{self.directories[p]}/Images')
        if sub_V_dict_li:
            shutil.move(self.images_pos_paths[p], orig_TIFFs_pos_path)
            shutil.rmtree(self.pos_paths[p])
        for sub_V_dict in sub_V_dict_li:
            folder_name = sub_V_dict['folder_name']
            sub_V = sub_V_dict['V_data']
            file_name = sub_V_dict['file_name']
            new_folder_path = f'{TIFFs_path}/{folder_name}/Images'
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
            new_path = f'{new_folder_path}/{file_name}'
            self.imagej_tiffwriter(new_path, sub_V)
        for sub_V_other_tifs_dict in sub_V_other_tifs_dict_li:
            folder_name = sub_V_other_tifs_dict['folder_name']
            sub_V = sub_V_other_tifs_dict['V_data']
            file_name = sub_V_other_tifs_dict['file_name']
            new_folder_path = f'{TIFFs_path}/{folder_name}/Images'
            new_path = f'{new_folder_path}/{file_name}'
            self.imagej_tiffwriter(new_path, sub_V)

    def save_all_ROI(self, TIFFs_path, sub_V_dict_li_li,
                           sub_V_other_tifs_dict_li_li):
        _zip = zip(range(self.p+1), sub_V_dict_li_li, sub_V_other_tifs_dict_li_li)
        for p, sub_V_dict_li, sub_V_other_tifs_dict_li in _zip:
            print(f'Saving image {p+1}/{self.p+1}...')
            if sub_V_dict_li is not None:
                self.save_sub_img(TIFFs_path, p, sub_V_dict_li,
                                  sub_V_other_tifs_dict_li)


"""Initialize app GUI parameters"""
plt.dark()
axcolor = '0.1'
slider_color = '0.2'
hover_color = '0.25'
presscolor = '0.35'
button_true_color = '0.4'
spF = 0.01  # spacing factor
wH = 0.04  # widgets' height
slW = 0.6  # sliders width
bW = 0.2  # buttons width

# Folder dialog
selected_path = folder_dialog(title = "Select folder containing valid experiments")
beyond_listdir_pos = beyond_listdir_pos(selected_path)
select_widget = select_exp_folder()
TIFFs_path = select_widget.run_widget(beyond_listdir_pos.all_exp_info,
                         title='Select experiment to split',
                         label_txt='Select experiment to split into ROI',
                         full_paths=beyond_listdir_pos.TIFFs_path,
                         showinexplorer_button=True)

"""Load data"""
app = init_App(TIFFs_path)
app.init_plt_GUI()
app.update_plot()
num_pos = len(app.directories)

"""Widgets' axes as [left, bottom, width, height]"""
ax_slice = plt.axes([0.1, 0.3, 0.8, 0.03])
ax_next = plt.axes([0.62, 0.2, 0.05, 0.03])
ax_prev = plt.axes([0.67, 0.2, 0.05, 0.03])

"""Widgets"""
slider_slice = Slider(ax_slice, 'Z-slice', 5, app.num_slices,
                    valinit=app.num_slices/2,
                    valstep=1,
                    color=slider_color,
                    init_val_line_color=hover_color,
                    valfmt='%1.0f')
next_b = Button(ax_next, 'Next position',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)
prev_b = Button(ax_prev, 'Prev. position',
                color=axcolor, hovercolor=hover_color,
                presscolor=presscolor)

"""Widgets functions"""
def next_pos(event):
    if app.p+1 >= num_pos:
        app.sub_V_dict_li_li[app.p] = deepcopy(app.sub_V_dict_li)
        app.sub_V_other_tifs_dict_li_li[app.p] = deepcopy(app.sub_V_other_tifs_dict_li)
        save = tk.messagebox.askyesno('Save?', 'There are no positions left.\n'
                               'Do you want to save?')
        if save:
            app.fig.canvas.mpl_disconnect(cid_close)
            print('Saving data...')
            app.sub_V_dict_li_li[app.p] = deepcopy(app.sub_V_dict_li)
            app.sub_V_other_tifs_dict_li_li[app.p] = deepcopy(app.sub_V_other_tifs_dict_li)
            app.save_all_ROI(TIFFs_path, app.sub_V_dict_li_li,
                                         app.sub_V_other_tifs_dict_li_li)
            print('Saved!')
            close = tk.messagebox.askyesno('Close?', 'Positions saved!.\n'
                                   'Do you want to close?')
            if close:
                plt.close()
        print('You reached the last position!')
    else:
        app.sub_V_dict_li_li[app.p] = deepcopy(app.sub_V_dict_li)
        app.sub_V_other_tifs_dict_li_li[app.p] = deepcopy(app.sub_V_other_tifs_dict_li)
        app.p += 1
        app.update_plot()
        app.sub_V_dict_li = []
        app.sub_V_other_tifs_dict_li = []
    app.counter = 0
    app.ROIimg_li = []


def prev_pos(event):
    if app.p == 0:
        print('You reached the last position!')
    else:
        app.p -= 1
        app.update_plot()
    app.counter = 0
    app.ROIimg_li = []

def s_slice_cb(val):
    app.s = int(val)
    app.update_plot()

"""Connect widgets to callbacks function"""
next_b.on_clicked(next_pos)
prev_b.on_clicked(prev_pos)
slider_slice.on_changed(s_slice_cb)

"""Canvas events functions"""
def key_down(event):
    key = event.key
    if key == 'right':
        if app.key_mode == 'Z-slice':
            slider_slice.set_val(slider_slice.val+1)
        else:
            next_pos(None)
    elif key == 'left':
        if app.key_mode == 'Z-slice':
            slider_slice.set_val(slider_slice.val-1)
        else:
            prev_pos(None)
    elif key == 'ctrl+z':
        app.update_plot()
        app.counter = 0
        app.ROIimg_li = []
        app.sub_V_dict_li = []
        app.sub_V_other_tifs_dict_li = []

def get_rect_patch(img_shape, x, y):
    img_h, img_w = app.img.shape
    square_halfsize = 150
    left = x - square_halfsize
    bottom = y - square_halfsize
    rect_size = square_halfsize*2
    # Ensure that square stays within img boundaries
    right = x + square_halfsize
    if right >= img_w:
        left = img_w - rect_size - 1
    top = y + square_halfsize
    if top >= img_h:
        bottom = img_h - rect_size - 1
    if left < 0:
        left = 0
    if bottom < 0:
        bottom = 0
    # Draw square
    rect_patch = Rectangle((left, bottom), rect_size, rect_size,
                        color='r', ls='--', fill=False)
    return rect_patch, bottom, left, rect_size

def mouse_down(event):
    right_click = event.button == 3
    ax_click = event.inaxes == app.ax
    img_h, img_w = app.img.shape
    if event.xdata:
        xd = int(round(event.xdata))
        yd = int(round(event.ydata))
    if right_click and ax_click:
        rect_patch, bottom, left, rect_size = get_rect_patch(app.img.shape,
                                                                    xd, yd)
        app.ax.add_patch(rect_patch)
        app.ax.text(xd, yd, 'X', size=28, c='r', ha='center', va='center')
        app.fig.canvas.draw_idle()
        # Append sub img data for saving later
        app.counter += 1
        V = app.all_phc_img_li[app.p]
        sub_V = V[:, bottom:bottom+rect_size, left:left+rect_size]
        folder_name = f'{app.directories[app.p]}-{app.counter}'
        file_name = f'{app.basenames[app.p]}-{app.counter}_phase_contr.tif'
        print(f'New file name: {file_name}')
        app.sub_V_dict_li.append({'folder_name': folder_name,
                                  'file_name': file_name,
                                  'V_data': sub_V})
        # Index also the other channels .tif files in the folder
        for tif_dict in app.all_others_tifs_li_dict[app.p]:
            tif_data = tif_dict['tif_data']
            sub_tif_data = tif_data[:, bottom:bottom+rect_size,
                                       left:left+rect_size]
            tif_basename = tif_dict['tif_basename']
            tif_channelname = tif_dict['tif_channelname']
            file_name = f'{tif_basename}-{app.counter}_{tif_channelname}'
            app.sub_V_other_tifs_dict_li.append({'folder_name': folder_name,
                                                 'file_name': file_name,
                                                 'V_data': sub_tif_data})

rect_patch_motion = Rectangle((1, 1), 1, 1, color='r', ls='--', fill=False)
def mouse_motion(event):
    global rect_patch_motion
    if event.inaxes == app.ax:
        xd = int(round(event.xdata))
        yd = int(round(event.ydata))
        rect_patch_motion.set_visible(False)
        try:
            rect_patch_motion.remove()
        except:
            pass
        rect_patch_motion, _, _, _ = get_rect_patch(app.img.shape, xd, yd)
        app.ax.add_patch(rect_patch_motion)
        app.fig.canvas.draw_idle()




def resize_widgets(event):
    # [left, bottom, width, height]
    ax = app.ax
    ax0_l, ax0_b, ax0_r, ax0_t = ax.get_position().get_points().flatten()
    ax_slice.set_position([ax0_l, ax0_b-spF-wH, ax0_r-ax0_l, wH])
    ax_slice_l, ax_slice_b, ax_slice_r, ax_slice_t = (ax_slice.get_position()
                                                              .get_points()
                                                              .flatten())
    ax_next.set_position([ax_slice_r-bW, ax_slice_b-2*spF-wH, bW, wH])
    ax_prev.set_position([ax_slice_r-bW*2, ax_slice_b-2*spF-wH, bW, wH])

def axes_enter(event):
    if event.inaxes == ax_slice:
        app.key_mode = 'Z-slice'

def axes_leave(event):
    app.key_mode = ''

def handle_close(event):
    save = tk.messagebox.askyesno(title='Save?',
              message='Do you want to save?')
    if save:
        print('Saving data...')
        app.sub_V_dict_li_li[app.p] = deepcopy(app.sub_V_dict_li)
        app.sub_V_other_tifs_dict_li_li[app.p] = deepcopy(app.sub_V_other_tifs_dict_li)
        app.save_all_ROI(TIFFs_path, app.sub_V_dict_li_li,
                                     app.sub_V_other_tifs_dict_li_li)

"""Connect to canvas events"""
(app.fig.canvas).mpl_connect('key_press_event', key_down)
(app.fig.canvas).mpl_connect('button_press_event', mouse_down)
(app.fig.canvas).mpl_connect('motion_notify_event', mouse_motion)
(app.fig.canvas).mpl_connect('resize_event', resize_widgets)
(app.fig.canvas).mpl_connect('axes_enter_event', axes_enter)
(app.fig.canvas).mpl_connect('axes_leave_event', axes_leave)
cid_close = (app.fig.canvas).mpl_connect('close_event', handle_close)

"""Final settings and start GUI"""
plt.win_size(w=0.5, h= 0.8, swap_screen=False)
app.fig.canvas.set_window_title(f'Cell segmentation split ROI - GUI - {app.exp_name}')
plt.show()
