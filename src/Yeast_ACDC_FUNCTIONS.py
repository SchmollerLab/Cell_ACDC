import os, traceback, cv2, subprocess, re, matplotlib
from sys import exit
import statistics as stat
import numpy as np
from numpy.linalg import det, norm
import pandas as pd
import tkinter as tk
from tkinter import N, S, E, W, END
from tkinter import messagebox, filedialog
import scipy.ndimage as nd
from scipy.stats import entropy
from scipy.optimize import curve_fit
from scipy.ndimage import distance_transform_edt
from skimage import io
from skimage.measure import label, regionprops, find_contours
from skimage.morphology import erosion, remove_small_objects
from skimage.morphology import disk as morph_disk
from skimage.segmentation import watershed
from skimage.filters import (apply_hysteresis_threshold, gaussian, sobel)
from skimage.feature import peak_local_max
from skimage.registration import phase_cross_correlation
from skimage.draw import disk, circle_perimeter, line, line_aa, bezier_curve
from skimage.exposure import histogram
from skimage.color import gray2rgb, label2rgb
import matplotlib.pyplot as plt
from Yeast_ACDC_MyWidgets import Slider, Button, RadioButtons
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle, PathPatch
import matplotlib.path as mpath
from natsort import natsorted
from ast import literal_eval
try:
    import ctypes
    from ctypes import wintypes
except:
    pass
from tkinter import ttk
from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from pyglet.canvas import Display


#expand dataframe beyond page width in the terminal
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 150)
pd.set_option('display.precision', 3)
pd.set_option('display.expand_frame_repr', False)

class num_frames_toSegm_tk:
    def __init__(self, tot_frames, last_segm_i=None, last_tracked_i=-1,
                 toplevel=False, allow_not_0_start=True):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        self.allow_not_0_start = allow_not_0_start
        self.root = root
        self.tot_frames = tot_frames
        self.root.title('Number of frames to segment')
        root.geometry('+800+400')
        root.lift()
        root.attributes("-topmost", True)
        # root.focus_force()
        tk.Label(root,
                 text="How many frames do you want to segment?",
                 font=(None, 12)).grid(row=0, column=0, columnspan=3)
        if last_segm_i is not None:
            txt = (f'(there is a total of {tot_frames} frames,\n'
                   f'last segmented frame is index {last_segm_i})')
        else:
            txt = f'(there is a total of {tot_frames} frames)'
        if last_tracked_i != -1:
            txt = f'{txt[:-1]}\nlast tracked frame is index {last_tracked_i})'
        tk.Label(root,
                 text=txt,
                 font=(None, 10)).grid(row=1, column=0, columnspan=3)
        tk.Label(root,
                 text="Start frame",
                 font=(None, 10, 'bold')).grid(row=2, column=0, sticky=E,
                                               padx=4)
        tk.Label(root,
                 text="Number of frames to analyze",
                 font=(None, 10, 'bold')).grid(row=3, column=0, padx=4)
        sv_sf = tk.StringVar()
        start_frame = tk.Entry(root, width=10, justify='center',font='None 12',
                            textvariable=sv_sf)
        start_frame.insert(0, '{}'.format(0))
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
        # root.after(1000, self.set_foreground_window)
        root.mainloop()

    def set_all(self, name=None, index=None, mode=None):
        if self.allow_not_0_start:
            start_frame_str = self.start_frame.get()
            if start_frame_str:
                startf = int(start_frame_str)
                rightRange = self.tot_frames - startf
                self.num_frames.delete(0, END)
                self.num_frames.insert(0, '{}'.format(rightRange))

    def check_max(self, name=None, index=None, mode=None):
        if self.allow_not_0_start:
            num_frames_str = self.num_frames.get()
            start_frame_str = self.start_frame.get()
            if num_frames_str and start_frame_str:
                startf = int(start_frame_str)
                if startf + int(num_frames_str) > self.tot_frames:
                    rightRange = self.tot_frames - startf
                    self.num_frames.delete(0, END)
                    self.num_frames.insert(0, '{}'.format(rightRange))
        else:
            self.start_frame.delete(0, END)
            self.start_frame.insert(0, '{}'.format(0))


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

    def set_foreground_window(self):
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.focus_force()

class tk_breakpoint:
    '''Geometry: "WidthxHeight+Left+Top" '''
    def __init__(self, title='Breakpoint', geometry="+800+400",
                 message='Breakpoint', button_1_text='Continue',
                 button_2_text='Abort', button_3_text='Delete breakpoint'):
        self.abort = False
        self.next_i = False
        self.del_breakpoint = False
        self.title = title
        self.geometry = geometry
        self.message = message
        self.button_1_text = button_1_text
        self.button_2_text = button_2_text
        self.button_3_text = button_3_text

    def pausehere(self):
        global root
        if not self.del_breakpoint:
            root = tk.Tk()
            root.lift()
            root.attributes("-topmost", True)
            root.title(self.title)
            root.geometry(self.geometry)
            tk.Label(root,
                     text=self.message,
                     font=(None, 11)).grid(row=0, column=0,
                                           columnspan=2, pady=4, padx=4)

            tk.Button(root,
                      text=self.button_1_text,
                      command=self.continue_button,
                      width=10,).grid(row=4,
                                      column=0, sticky=tk.W+tk.E,
                                      pady=0, padx=4)

            tk.Button(root,
                      text=self.button_2_text,
                      command=self.abort_button,
                      width=15).grid(row=4,
                                     column=1, sticky=tk.W+tk.E,
                                     pady=0, padx=4)
            tk.Button(root,
                      text=self.button_3_text,
                      command=self.delete_breakpoint,
                      width=20).grid(row=5,
                                     column=0, sticky=tk.N,
                                     columnspan=2, pady=4)

            root.mainloop()

    def continue_button(self):
        self.next_i=True
        root.quit()
        root.destroy()

    def delete_breakpoint(self):
        self.del_breakpoint=True
        root.quit()
        root.destroy()

    def abort_button(self):
        self.abort=True
        exit('Execution aborted by the user')
        root.quit()
        root.destroy()

# Z projection by max intensity
def z_proj_max(z_stacks, dtype=None):
    z_proj = z_stack.max(axis=0)
    return z_proj

#Gaussian function
def gaus(x,a,x0,sigma,b):
    return (a*np.exp(-(x-x0)**2/(2*sigma**2)))+b

#Fit gaussian function to background peak and estimate background threshold
def background(hist, x_hist, fit_width, sigma_plus):
    peak = hist.max()
    x_peak = x_hist[hist == peak][0]
    b_guess = stat.mean(hist)
    left = x_peak-int(fit_width/2)
    right = x_peak+int(fit_width/2)
    print('Gaussian fit range: [{},{},{}]'.format(left,x_peak,right))
    popt,pcov = curve_fit(gaus,x_hist[left:right],hist[left:right],p0=[peak,x_peak,1,b_guess])
    mean, sigma = popt[1], popt[2]
    background_threshold = int(mean+6*sigma)+sigma_plus
    return background_threshold

#Extract all cells as a single block
def extract_cells(gauss_img, background_threshold):
    cells_thresh = gauss_img > background_threshold
    cells_thresh_filled = nd.morphology.binary_fill_holes(cells_thresh)
    cells_label, num_obj = label(cells_thresh_filled,connectivity = 2, return_num=True)
    region_props = regionprops(cells_label)
    sizes, labels = np.zeros(num_obj, int), np.zeros(num_obj, int)
    for i, obj in enumerate(region_props):
        sizes[i] = obj.area
        labels[i] = obj.label
    max_size = sizes.max()
    label_max_size = labels[np.where(sizes == max_size)[0][0]]
    cells_binary, cells_intensity = np.zeros(gauss_img.shape,dtype=np.uint8), np.zeros(gauss_img.shape,dtype=np.uint8)
    cells_binary[cells_label == label_max_size] = 1
    cells_intensity[cells_binary == 1] = gauss_img[cells_binary == 1]
    return cells_intensity

#Perform erosion, but avoid eroding objects completely (stop eroding an object when the next iteration would result a complete erosion)
def erosion_no_deletion(image, selem_size = 1):
    fully_eroded = np.zeros_like(image)
    eroded = image
    previous = image
    all_eroded = [image]
    all_label_eroded = []
    all_num_obj_eroded = np.array([],int)
    fully_eroded_bool = np.array([False])
    current_labels = np.unique(image)
    i = 1
    while not np.all(fully_eroded_bool):
        eroded = erosion(eroded, selem=morph_disk(selem_size))
        fully_eroded_bool = eroded == fully_eroded
        if not np.all(fully_eroded_bool):
            all_eroded.append(eroded)
            previous = all_eroded[i-1]
            previous_labels = np.unique(previous)
            current_labels = np.unique(eroded)
            diff_labels = set(previous_labels)-set(current_labels)
            if diff_labels:
                for diff_label in diff_labels:
                    eroded[previous == diff_label] = diff_label
                all_eroded[-1] = eroded
            label_eroded, num_obj_eroded = label(eroded, connectivity=2, return_num=True) #label eroded image
            all_label_eroded.append(label_eroded) #store labelled eroded image
            all_num_obj_eroded = np.append(all_num_obj_eroded, num_obj_eroded) #store number of objects found per each eroded image
            i += 1
    return all_label_eroded, all_num_obj_eroded

#Separate overlapping objects by successive erosions to obtain markers for watershed on distance transform
def separate_overlapping(cells_labels, distance, selem_size=1):
    watershed_dist_eros = cells_labels
    all_label_eroded, all_num_obj_eroded = erosion_no_deletion(cells_labels, selem_size=selem_size)
    #print(all_num_obj_eroded)
    max_num_eros = all_num_obj_eroded.max() #find maximum number of objects obtained by successive erosions
    max_num_i = np.flip(np.where(all_num_obj_eroded == max_num_eros)[0])[0] #get index of the last maximum number of objects
    label_eroded_max = all_label_eroded[max_num_i] #index labelled eroded image corresponding to the index of the last maximum number of objects
    regionprops_label_eroded = regionprops(label_eroded_max)
    markers_eros = np.zeros_like(cells_labels) #initialize markers for erosion
    for obj in regionprops_label_eroded:
        centroid_coords = tuple(np.asarray(obj.centroid).astype(int)) #convert centroid coordinates to integers and convert the result to a tuple
        markers_eros[centroid_coords] = obj.label #insert label to centroid coords
    watershed_dist_eros = watershed(-distance, markers_eros, mask=cells_labels) #perform watershed on the distance transform image using markers obtained by successive erosions and only where mask=True
    num_obj_after_eros = label(watershed_dist_eros, connectivity=2, return_num=True)[1]
    regionprops_watershed_dist_eros = regionprops(watershed_dist_eros)
    return watershed_dist_eros, num_obj_after_eros, regionprops_watershed_dist_eros

#Function for separating overlapping objects using manually entered seeds
def sep_overlap_manual_seeds(frame_i, cells_labels_separate, num_cells_separate=None,
                rpCsep=None, param_dict=None, seeds_coords=None):
    if param_dict is not None:
        seeds_coords = param_dict['Seeds coords'][frame_i]
    if seeds_coords:
        for seeds in seeds_coords:
            seed_1_coords, seed_2_coords = seeds
            seed_1_value = cells_labels_separate[seed_1_coords]
            max_label = rpCsep[-1].label
            for i, obj in enumerate(rpCsep):
                if obj.label == seed_1_value:
                    break
                else:
                    i = None
            obj_seed = rpCsep[i] #region properties of the selected object
            label_img_only_obj = np.zeros_like(cells_labels_separate)
            ID = obj_seed.label
            label_img_only_obj[cells_labels_separate == ID] = ID
            markers = np.zeros_like(cells_labels_separate) #initialize markers for erosion
            markers[seed_1_coords] = seed_1_value
            markers[seed_2_coords] = max_label+1 #assign a label that is not already used by any other object
            distance = nd.distance_transform_edt(label_img_only_obj)
            watershed_dist_manual_seeds = watershed(-distance, markers,
                                                    mask=label_img_only_obj>0)
            cells_labels_separate[
                cells_labels_separate == ID] = watershed_dist_manual_seeds[
                                                    cells_labels_separate == ID]
        cells_labels_separate, num_cells_separate = label(cells_labels_separate,
                                                            connectivity=2,
                                                            return_num=True)
        return cells_labels_separate, num_cells_separate, regionprops(cells_labels_separate)
    else:
        return cells_labels_separate, num_cells_separate, rpCsep

def crossOF(OF_IDs, regionprops_cells_labels_separate, ax):
    for ID in OF_IDs:
        all_IDs = [obj.label for obj in regionprops_cells_labels_separate]
        ID_idx = all_IDs.index(ID)
        y, x = regionprops_cells_labels_separate[ID_idx].centroid
        ax[1].plot(x, y, 'rx', mew=6, ms=22, solid_capstyle='round')
        for X in ax[0].get_lines():
            X.remove()
        ax[0].plot(x, y, 'rx', mew=6, ms=22, solid_capstyle='round')

def update_plots(regionprops_cells_labels_separate, cells_labels_separate,
                 ax, all_bud_mother_ids, param_dict, frame_i, num_cells_separate,
                 num_cells_list, new_cells_ids, warning_text, fig, num_frames,
                 ax1_center, frame_text, display_ccStage, cc_stage_frame,
                 y_frame_txt, y_warn_txt, ax_init_cc_stage_frame0, ax_limits,
                 pos_foldername):
    text_label_centroid(regionprops_cells_labels_separate, ax[0], 12,
                        'semibold', 'center', 'center',
                        cc_stage_frame=cc_stage_frame,
                        display_ccStage=display_ccStage, color='r',
                        clear=True)
    ax[1].clear()
    ax[1].imshow(cells_labels_separate)
    text_label_centroid(regionprops_cells_labels_separate, ax[1], 12,
                        'semibold', 'center', 'center')
    try:
        param_dict = line_mother_bud(all_bud_mother_ids,
                                     param_dict,
                                     regionprops_cells_labels_separate, ax[1],
                                     cc_stage_frame, frame_i)
    except:
        pass
    ax[1].set_title('N. of cells current frame = {}\n '
                    'N. of cells previous frame = {}\n '
                    'New cells\' IDs: {}'.format(num_cells_separate,
                                                 num_cells_list[frame_i-1],
                                                 new_cells_ids))
    ax[1].axis('off')
    warning_text_color = 'limegreen' if warning_text.find('Looking good!') != -1 else 'r'
    fig_text(fig, warning_text, y=y_warn_txt, size=16, color=warning_text_color)
    frame_text = fig_text(fig,
                          'Current frame = {}/{}'.format(frame_i,num_frames),
                          x=ax1_center, y=y_frame_txt, color='w', size=14,
                          clear_all=False, clear_text_ref=True,
                          text_ref=frame_text)
    fig_text(fig,'Folder name = {}'.format(pos_foldername),
              x=0.1, y=0.95, color='w', size=14,
              clear_all=False, clear_text_ref=False,
              text_ref=None)
    if frame_i == 0:
        ax_init_cc_stage_frame0.set_visible(False)
    else:
        ax_init_cc_stage_frame0.set_visible(False)
    crossOF(param_dict['OF_IDs'][frame_i], regionprops_cells_labels_separate, ax)
    fig.canvas.draw_idle()
    for i, axes in enumerate(ax):
        axes.set_xlim(ax_limits[i][0])
        axes.set_ylim(ax_limits[i][1])
    return param_dict


#Function to place label text at the centroid coordinate of each object
def text_label_centroid(regionprops_label_img, ax, size, weight,
                        ha, va, cc_stage_frame=None, display_ccStage=None,
                        color='k',clear=False, apply=True, red_buds=None,
                        new_IDs=[], display_IDs_cont='IDs and contours',
                        selected_IDs=None):
    if apply:
        if clear:
            for t in ax.texts:
                t.set_visible(False)
        new_size = size
        new_weight = weight
        if display_ccStage is not None:
            for region in regionprops_label_img:
                y, x = region.centroid
                ID = region.label
                if display_ccStage == 'All info':
                    new_size = size-2
                    new_weight = 'normal'
                    ID_info = cc_stage_frame.loc[ID]
                    cc_stage = ID_info['Cell cycle stage']
                    cycle_num = ID_info['# of cycles']
                    cycle_num = 'ND' if cycle_num == -1 else cycle_num
                    rel_ID = ID_info['Relative\'s ID']
                    rel_ID = 'ND' if rel_ID == 0 else rel_ID
                    rel = ID_info['Relationship'][0] #'b'=bud, 'd'=daughter, 'm'=mother
                    cc_txt = f'{rel}-of-{rel_ID}\nin_{cc_stage}{cycle_num}'
                    txt = 'ND' if cc_stage=='ND' else cc_txt
                elif display_ccStage == 'Only stage':
                    new_size = size-2
                    #print(cc_stage_frame)
                    ID_info = cc_stage_frame.loc[ID]
                    cc_stage = ID_info['Cell cycle stage']
                    cycle_num = ID_info['# of cycles']
                    cycle_num = 'ND' if cycle_num == -1 else cycle_num
                    cc_txt = '{}-{}'.format(cc_stage, cycle_num)
                    txt = 'ND' if cc_stage=='ND' else cc_txt
                    color = 'r' if (cc_stage=='S' and cycle_num==0) else '0.8'
                    if (cc_stage=='S' and cycle_num==0):
                        new_weight = 'semibold'
                    else:
                        new_weight = 'normal'
                else:
                    txt = str(ID)
                    if red_buds is not None:
                        color = 'r' if ID in red_buds else '0.8'
                        new_weight = 'semibold' if ID in red_buds else 'normal'
                        new_size = size if ID in red_buds else size-2
                    elif new_IDs:
                        color = 'r' if ID in new_IDs else 'k'
                try:
                    if selected_IDs is not None:
                        if ID in selected_IDs:
                            alpha = 1
                        else:
                            alpha = 0.3
                    else:
                        alpha = 1
                    ax.text(int(x), int(y), txt, fontsize=new_size,
                            fontweight=new_weight, horizontalalignment=ha,
                            verticalalignment=va, color=color, alpha=alpha)
                    # if selected_IDs is not None:
                    #     for selected_ID in selected_IDs:
                    #         if selected_ID == ID:
                    #             circle_ID = Circle((int(x), int(y)), radius=15,
                    #                                fill=False, color='r', lw=2)
                    #             ax.add_patch(circle_ID)
                    # else:
                    #     ax.patches = []
                except:
                    traceback.print_exc()
        elif display_IDs_cont is not None:
            for region in regionprops_label_img:
                y, x = region.centroid
                ID = region.label
                if display_IDs_cont == 'IDs and contours':
                    txt = str(ID)
                    if red_buds is not None:
                        color = 'r' if ID in red_buds else '0.8'
                        new_weight = 'semibold' if ID in red_buds else 'normal'
                        new_size = size if ID in red_buds else size-2
                    elif new_IDs:
                        color = 'r' if ID in new_IDs else 'k'
                    try:
                        if selected_IDs is not None:
                            if ID in selected_IDs:
                                alpha = 1
                            else:
                                alpha = 0.3
                        else:
                            alpha = 1
                        ax.text(int(x), int(y), txt, fontsize=new_size,
                                fontweight=new_weight, horizontalalignment=ha,
                                verticalalignment=va, color=color)
                        # if selected_IDs is not None:
                        #     for selected_ID in selected_IDs:
                        #         if selected_ID == ID:
                        #             circle_ID = Circle((int(x), int(y)),
                        #                                radius=15, fill=False,
                        #                                color='r', lw=2)
                        #             ax.add_patch(circle_ID)
                    except:
                        traceback.print_exc()
                else:
                    txt = ''


#Function that applies Hysteresis threshold plus local threshold on manually selected portion of the image
#rectangles is a list of tuples where each tuple is (y_start, y_end, x_start, x_end) coordinates of each rectangle
#The function extract the rectangular portion from img and apply a selected global threshold on the rectangle and update the final thresholded image
def apply_hyst_local_threshold(img,lowT,highT,arcs,circles,delta_local_T):
    thresh = apply_hysteresis_threshold(img,lowT,highT)
    # for rect in rectangles:
    #     y_start, y_end, x_start, x_end = rect
    #     thresh[y_start:y_end+1,x_start:x_end+1] = img[y_start:y_end+1,x_start:x_end+1] > (lowT-delta_local_T)
    for rr, cc in arcs:
        thresh[rr, cc] = True
    for circ in circles:
        c, r, radius = circ
        rr, cc = disk((r,c),radius)
        thresh[rr, cc] = 0# img[rr,cc] > (lowT+delta_local_T)
    return thresh

# Function that remove objects with area < area_min
def del_min_area_obj(label_img, region_props, area_min):
    for obj in region_props:
        if obj.area <= area_min:
            label_img[label_img == obj.label] = 0
    label_img, num_obj = label(label_img, connectivity=2, return_num=True)
    region_props_min = regionprops(label_img)
    return label_img, num_obj, region_props_min

#Function that align current frame based on previous frame. If register=True the function calculates the shifts
#otherwise it uses shifts from the user (shift, error, diffphase). Typically these shifts come from cell segmentation
#slice=None will result in alignment of each slice of the 3D data
def align_frames(data, slices, register=True, user_shifts=None):
    registered_shifts = np.zeros((len(data),2), int)
    data_aligned = np.copy(data)
    for frame_i, frame_V in enumerate(data):
        slice = slices[frame_i]
        if frame_i != 0:  # skip first frame
            curr_frame_img = frame_V[slice]
            prev_frame_img = data_aligned[frame_i-1, slice] #previously aligned frame, slice
            if register==True:
                shifts = phase_cross_correlation(prev_frame_img, curr_frame_img)[0]
            else:
                shifts = user_shifts[i]
            shifts = shifts.astype(int)
            aligned_frame_V = np.copy(frame_V)
            aligned_frame_V = np.roll(aligned_frame_V, tuple(shifts), axis=(1,2))
            data_aligned[frame_i] = aligned_frame_V
            registered_shifts[frame_i] = shifts
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(z_proj_max(frame_V))
            # ax[1].imshow(z_proj_max(aligned_frame_V))
            # plt.show()
    return data_aligned, registered_shifts

def align_frames_3D(data, slices=None, register=True, user_shifts=None):
    registered_shifts = np.zeros((len(data),2), int)
    data_aligned = np.copy(data)
    for frame_i, frame_V in enumerate(data):
        slice = slices[frame_i]
        if frame_i != 0:  # skip first frame
            curr_frame_img = frame_V[slice]
            prev_frame_img = data_aligned[frame_i-1, slice] #previously aligned frame, slice
            if register==True:
                shifts = phase_cross_correlation(prev_frame_img, curr_frame_img)[0]
            else:
                shifts = user_shifts[frame_i]
            shifts = shifts.astype(int)
            aligned_frame_V = np.copy(frame_V)
            aligned_frame_V = np.roll(aligned_frame_V, tuple(shifts), axis=(1,2))
            data_aligned[frame_i] = aligned_frame_V
            registered_shifts[frame_i] = shifts
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(z_proj_max(frame_V))
            # ax[1].imshow(z_proj_max(aligned_frame_V))
            # plt.show()
    return data_aligned, registered_shifts


def align_frames_2D(data, slices=None, register=True, user_shifts=None):
    registered_shifts = np.zeros((len(data),2), int)
    data_aligned = np.copy(data)
    for frame_i, frame_V in enumerate(data):
        if frame_i != 0:  # skip first frame
            curr_frame_img = frame_V
            prev_frame_img = data_aligned[frame_i-1] #previously aligned frame, slice
            if register==True:
                shifts = phase_cross_correlation(prev_frame_img, curr_frame_img)[0]
            else:
                shifts = user_shifts[frame_i]
            shifts = shifts.astype(int)
            aligned_frame_V = np.copy(frame_V)
            aligned_frame_V = np.roll(aligned_frame_V, tuple(shifts), axis=(0,1))
            data_aligned[frame_i] = aligned_frame_V
            registered_shifts[frame_i] = shifts
            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(z_proj_max(frame_V))
            # ax[1].imshow(z_proj_max(aligned_frame_V))
            # plt.show()
    return data_aligned, registered_shifts


# Load shifts from Position_n_folder
def load_shifts(parent_path):
    substring_found = False
    for filename in os.listdir(parent_path):
        if filename.find('align_shift.npy')>0:
            substring_found = True
            substring_path = parent_path + '/' + filename
            shifts = np.load(substring_path)
            return shifts, substring_found
        else:
            shifts = None
    return shifts, substring_found


#This function takes a 2D array of coordinates and returns a 1D array of (z,y,x) or (y,x) tuples
#If the input array is a 2D array of 3 or 2 columns then transpose is required
def obj_coords2Dto1Dtuples(obj_2Dcoords,V3D,order='zyx'):
    #check if the input obj_2Dcoords is a tuple (such as np.where() output) or a 2D array such as coords attribute of regionprops
    if type(obj_2Dcoords) is tuple:
        array2D = False
    else:
        array2D = True
        #check if transpose is required (number of rows is 3 or 2 depending on V3D)
        if (obj_2Dcoords.shape[1] == 3 and V3D) or (obj_2Dcoords.shape[1] == 2 and not V3D):
            obj_2Dcoords = np.transpose(obj_2Dcoords)

    object_size = obj_2Dcoords[0].size
    if V3D:
        if order == 'zyx':
            Z, Y, X = obj_2Dcoords[0], obj_2Dcoords[1], obj_2Dcoords[2]
        elif order == 'xyz':
            X, Y, Z = obj_2Dcoords[0], obj_2Dcoords[1], obj_2Dcoords[2]
        D = 3
        obj_a = np.transpose(np.stack((Z,Y,X)))
        coord = np.empty((), dtype=object)
        coord[()] = (0, 0, 0)
        obj_tuplecoords = np.full((object_size), coord, dtype=object)
    else:
        if order == 'yx':
            Y, X = obj_2Dcoords[0], obj_2Dcoords[1]
        elif order == 'xy':
            X, Y = obj_2Dcoords[0], obj_2Dcoords[1]
        D = 2
        obj_a = np.transpose(np.stack((Y,X)))
        coord = np.empty((), dtype=object)
        coord[()] = (0, 0)
        obj_tuplecoords = np.full((object_size), coord, dtype=object)
    for c in range(object_size):
         obj_tuplecoords[c] = tuple(obj_a[c])
    return obj_tuplecoords

#Function to add text to the figure panel (outside axes)
def fig_text(fig, text, x=0.5, y=0.9, color='r', ha='center', size=12,
             clear_all=True, clear_text_ref=False, text_ref=None):
    if clear_text_ref and text_ref:
        text_ref.set_visible(False)
    if clear_all:
        for t in fig.texts:
            t.set_visible(False)
    txt = fig.text(x, y, text, color = color, fontsize=size, horizontalalignment=ha)
    return txt

#Given the regionprops of two objects calculate IoU and IoObj1
def calc_IoU_IoObj1(obj_0_regionprops, obj_1_regionprops):
    obj_0_coords = obj_coords2Dto1Dtuples(obj_0_regionprops.coords,False,order='yx')
    obj_1_coords = obj_coords2Dto1Dtuples(obj_1_regionprops.coords,False,order='yx')
    pixels_intersection = np.intersect1d(obj_0_coords, obj_1_coords, assume_unique=True).size
    IoU = pixels_intersection/(obj_0_regionprops.area+obj_1_regionprops.area-pixels_intersection) #Intersection over Union
    IoObj_1 = pixels_intersection/(obj_1_regionprops.area) #ratio=1 if the obj in frame 1 is fully enclosed in obj in frame 2
    return IoU, IoObj_1

#Check if new objects are really new cells or only part of a previously existing cell
def check_new_ids(frame_0, tracked_frame, new_ids, frame_0_rp, tracked_frame_rp, num_cells_0):
    append_warning_text = ''
    not_new_obj_ids = []
    not_new_obj = False
    for new_id in new_ids:
        for i, obj in enumerate(tracked_frame_rp):
            if obj.label == new_id:
                break
            else:
                i = None
        new_obj_rp = tracked_frame_rp[i]
        IoObj_0_new_obj = [0]*num_cells_0
        obj_0_labels = [0]*num_cells_0
        for n, obj_0_rp in enumerate(frame_0_rp):
            IoU, IoObj_0 = calc_IoU_IoObj1(obj_0_rp, new_obj_rp)
            obj_0_labels[n] = obj_0_rp.label
            IoObj_0_new_obj[n] = IoObj_0
        max_IoObj_0_new_obj = max(IoObj_0_new_obj)
        if max_IoObj_0_new_obj > 0.5:
            #the new obj is actually at least 50% part of a previous object --> raise a warning
            ith_max_IoObj_0_new_obj = IoObj_0_new_obj.index(max_IoObj_0_new_obj)
            id_max_IoObj_0_new_obj = obj_0_labels[ith_max_IoObj_0_new_obj]
            not_new_obj_ids.append(new_id)
    if not_new_obj_ids:
        not_new_obj = True
        append_warning_text = ('\n Cells with IDs: {} did not pass the new object test'
                               .format(not_new_obj_ids))
    return not_new_obj_ids, append_warning_text, not_new_obj

#Function that given a frames_labels = [frame_labels_0, frame_labels_1] and frames_num_cells = [num_cells_0, num_cells_1]
#calculate IoU (see https://en.wikipedia.org/wiki/Jaccard_index) of each object in frame 1 and to each obj in frame 1
#assigns the same label as the obj with max IoU in frame 0
def cells_tracking(frames_labels, frames_num_cells, param_dict, frame_i, fig=None,
                    cell_cycle_analysis=False):
    # NOTE: Add possibility to stop tracking specific IDs
    prev_cc_stage = param_dict['Cell Cycle stages'][frame_i-1]
    prev_bud_mother_ids = param_dict['Bud-mother IDs'][0:frame_i]
    prev_frame_regionprops = param_dict['Regionprops'][frame_i-1]
    prev_frame_G1_df = param_dict['automatic_G1_df'][frame_i-1]
    frame_labels_0, frame_labels_1 = frames_labels
    np.save('prev_frame_labels.npy', frame_labels_0, allow_pickle=False)
    np.save('last_frame_labels.npy', frame_labels_1, allow_pickle=False)
    tracked_frame_labels_1 = np.copy(frame_labels_1)
    num_cells_0, num_cells_1 = frames_num_cells
    if num_cells_1 < num_cells_0:
        warning_text = ('WARNING: Current frame contains less cells segmented '
                    'compared to previous frame. Adjust segmentation to allow '
                    'correct tracking')
        print(warning_text)
        append_warning_text = ''
        tracked_frame_1_regionprops = regionprops(frame_labels_1)
        all_bud_mother_ids = []
        new_ids_tracked_frame_1 = []
        old_cells_ids = []
        cc_stage_frame = prev_cc_stage
        cc_warn_txt = ''
        G1_df = prev_frame_G1_df
    else:
        if fig:
            for t in fig.texts:
                t.set_visible(False)
        frame_0_regionprops = regionprops(frame_labels_0)
        frame_1_regionprops = regionprops(frame_labels_1)
        new_objs_ids_frame_1 = [] #this list will contain the id of the new obj in frame 1 (they will need a new assignment at the end)
        lost_objs_ids_frame_1 = [] #this list will contain the id of obj that were present in frame 0 but disappeared in frame 1
        for n, obj_0 in enumerate(frame_0_regionprops):
            obj_1_labels = [0]*num_cells_1
            IoUs_obj_0 = [0]*num_cells_1
            IoObj_1_obj_0 = [0]*num_cells_1
            for i, obj_1 in enumerate(frame_1_regionprops):
                if n == 0:
                    new_objs_ids_frame_1.append(obj_1.label)
                obj_1_labels[i] = obj_1.label
                IoU, IoObj_1 = calc_IoU_IoObj1(obj_0, obj_1)
                IoUs_obj_0[i], IoObj_1_obj_0[i] = IoU, IoObj_1
            max_IoU_obj_0, max_IoObj_1_obj_0 = max(IoUs_obj_0), max(IoObj_1_obj_0) #max IoU between current obj_0 and all obj_1
            ith_max_IoU_obj_0, ith_max_IoObj_1_obj_0 = IoUs_obj_0.index(max_IoU_obj_0), IoObj_1_obj_0.index(max_IoObj_1_obj_0) #index of the max above
            obj_1_label_max_IoU = obj_1_labels[ith_max_IoU_obj_0] #obj_1 label with max IoU from current obj_0. This object is the same as previous frame.
            obj_1_label_max_IoObj1 = obj_1_labels[ith_max_IoObj_1_obj_0]
            valid_IoU = max_IoU_obj_0 > 0
            #print(obj_0.label, obj_1_label_max_IoU, max_IoU_obj_0, valid_IoU)
            if valid_IoU and (obj_1_label_max_IoU in new_objs_ids_frame_1):
                new_objs_ids_frame_1.remove(obj_1_label_max_IoU) #remove obj_1 labels that has been assigned as same object in the previous frame
                if obj_1_label_max_IoU != obj_0.label:
                    tracked_frame_labels_1[frame_labels_1 == obj_1_label_max_IoU] = obj_0.label #assign same labels to same objects (determined by max IoU)
            else:
                lost_objs_ids_frame_1.append(obj_0.label)
        new_ids_tracked_frame_1 = [0]*len(new_objs_ids_frame_1)
        max_ID_prev_frame = frame_labels_0.max()
        for i, new_obj_id_1 in enumerate(new_objs_ids_frame_1):
            new_id = max_ID_prev_frame + i + 1
            tracked_frame_labels_1[frame_labels_1 == new_obj_id_1] = new_id #assign a new id higher than num_cells_0 to new obj in frame 1
            new_ids_tracked_frame_1[i] = new_id
        tracked_frame_1_regionprops = regionprops(tracked_frame_labels_1)
        (not_new_obj_ids,
        append_warning_text, not_new_obj) = check_new_ids(frame_labels_0,
                                                          tracked_frame_labels_1,
                                                          new_ids_tracked_frame_1,
                                                          frame_0_regionprops,
                                                          tracked_frame_1_regionprops,
                                                          num_cells_0)
        all_cells_ids = [obj.label for obj in tracked_frame_1_regionprops]
        old_cells_ids = [id for id in all_cells_ids if id not in new_ids_tracked_frame_1]
        cc_warn_txt = ''
        if cell_cycle_analysis:
            new_contours = find_contours(tracked_frame_labels_1,
                                new_ids_tracked_frame_1, group=True)[0]
            old_contours = find_contours(tracked_frame_labels_1,
                                        old_cells_ids, concatenate=True)[0]
            all_bud_mother_ids, cc_stage_frame, cc_warn_txt = cc_analysis(
                                                    tracked_frame_labels_1,
                                                    new_contours, old_contours,
                                                    all_cells_ids, prev_cc_stage,
                                                    tracked_frame_1_regionprops,
                                                    prev_bud_mother_ids)
            G1_df = automatic_G1(cc_stage_frame, tracked_frame_1_regionprops,
                        all_cells_ids, prev_frame_regionprops, prev_frame_G1_df)
            for ID in param_dict['OF_IDs'][frame_i]:
                cc_stage_frame.at[ID, 'OF'] = True # remember OF cells
        else:
            all_bud_mother_ids, cc_stage_frame, G1_df = [], [], None
        if lost_objs_ids_frame_1:
            warning_text = 'WARNING: Cells with IDs = {} in previous frame disappeared in current frame'.format(lost_objs_ids_frame_1)
            print('WARNING: Cells with IDs = {} in previous frame disappeared in current frame'.format(lost_objs_ids_frame_1))
        elif not_new_obj or cc_warn_txt:
            warning_text = ''
        else:
            warning_text = 'Looking good!'
    return (tracked_frame_labels_1, new_ids_tracked_frame_1, old_cells_ids,
           tracked_frame_1_regionprops, all_bud_mother_ids,
           warning_text+append_warning_text+'\n'+cc_warn_txt,
           cc_stage_frame, G1_df)


def manual_G1(ID_press, cc_stage_frame, prev_cc_stage_frame, increment=True):
    ID_press_info = cc_stage_frame.loc[ID_press]
    rel_ID_press = ID_press_info['Relative\'s ID']
    rel_ID_press_info = cc_stage_frame.loc[rel_ID_press]
    prev_num_cycle_ID_press = prev_cc_stage_frame.loc[ID_press]['# of cycles']
    prev_num_cycle_rel_ID_press = prev_cc_stage_frame.loc[rel_ID_press]['# of cycles']
    cc_stage_frame.at[ID_press, 'Cell cycle stage'] = 'G1'
    cc_stage_frame.at[rel_ID_press, 'Cell cycle stage'] = 'G1'
    if increment:
        cc_stage_frame.at[ID_press, '# of cycles'] = prev_num_cycle_ID_press+1
        cc_stage_frame.at[rel_ID_press, '# of cycles'] = prev_num_cycle_rel_ID_press+1
    return cc_stage_frame, rel_ID_press, prev_num_cycle_ID_press+1, prev_num_cycle_rel_ID_press+1

def update_G1(ID, rel_ID, cc_stage_frame, num_cycle_ID, num_cycle_rel_ID):
    cc_stage_frame.at[ID, 'Cell cycle stage'] = 'G1'
    cc_stage_frame.at[rel_ID, 'Cell cycle stage'] = 'G1'
    cc_stage_frame.at[ID, '# of cycles'] = num_cycle_ID
    cc_stage_frame.at[rel_ID, '# of cycles'] = num_cycle_rel_ID
    return cc_stage_frame

def combine_df_col(res_col_names, col_to_combine, pdf, cdf, func):
    """Combine 'col_to_combine' columns of two DataFrames given the function
    'func' and write the results into 'res_col_names' columns.
    The number of columns resulting from 'func' must match
    the number of 'res_col_names' columns.

    Parameters
    ----------
    res_col_names : list
        list of the column names where to write the result of func.
    col_to_combine : list
        list of the column names where to apply func.
    pdf : pandas DataFrame
        First DataFrame to combine.
    cdf : pandas DataFrame
        DataFrame to combine with pdf.
    func : function
        Function to apply to col_to_combine columns.

    Returns
    -------
    pandas DataFrame
        DataFrame with the results of func combination betwenn col_to_combine
        columns written into res_col_names.

    """
    cdf[res_col_names] = pdf[col_to_combine].combine(cdf[col_to_combine], func)
    cdf.replace(np.inf, 0, inplace=True)
    cdf.replace(-np.inf, 0, inplace=True)
    cdf.fillna(0, inplace=True)
    return cdf

def automatic_G1(cc_stage_frame, tracked_frame_1_regionprops, all_cells_ids,
                 prev_frame_regionprops, prev_frame_G1_df):
    buds_groups = cc_stage_frame.groupby('Relationship').groups
    G1_df = []
    if 'bud' in buds_groups.keys():
        buds_IDs = buds_groups['bud'].to_list() # all buds in current frame
        buds_df = cc_stage_frame.loc[buds_IDs]
        buds_S_groups = buds_df.groupby('Cell cycle stage').groups
        if 'S' in buds_S_groups.keys():
            all_buds_S_IDs = buds_S_groups['S'].to_list() # all buds in current frame that are in S phase
            if all_buds_S_IDs:
                rp = tracked_frame_1_regionprops
                all_buds_rel_IDs = cc_stage_frame.loc[all_buds_S_IDs]['Relative\'s ID'].to_list()
                unknown = np.zeros(len(all_buds_S_IDs))
                unknown_bool = np.zeros(len(all_buds_S_IDs), bool)
                G1_df = pd.DataFrame({'Delta_shift': unknown,
                                      'Delta_b_m': unknown,
                                      'Delta_eccentricity': unknown,
                                      'Centr. shift': unknown,
                                      'Dist_b_m': unknown,
                                      'Eccentricity': unknown,
                                      'Assign_G1': unknown_bool},
                                      index = all_buds_S_IDs)
                G1_df.index.name = 'Bud_ID'
                all_buds_centr = [rp[all_cells_ids.index(ID)].centroid for ID in all_buds_S_IDs]
                all_buds_rel_centr = [rp[all_cells_ids.index(ID)].centroid for ID in all_buds_rel_IDs]
                dist_b_m = np.linalg.norm((np.asarray(all_buds_rel_centr)-
                                           np.asarray(all_buds_centr)),
                                           axis=1)
                all_buds_eccentr = [rp[all_cells_ids.index(ID)].eccentricity for ID in all_buds_S_IDs]
                G1_df.loc[all_buds_S_IDs, 'Dist_b_m'] = dist_b_m
                G1_df.loc[all_buds_S_IDs, 'Eccentricity'] = all_buds_eccentr
                if prev_frame_regionprops:
                    prev_rp = prev_frame_regionprops
                    prev_IDs = [obj.label for obj in prev_rp]
                    old_buds_S_IDs = [ID for ID in prev_IDs if ID in all_buds_S_IDs] # buds that were in S phase also in previous frame
                    new_buds_S_IDs = [ID for ID in all_buds_S_IDs if ID not in old_buds_S_IDs] # new buds in S phase in current frame
                    # print('All buds in S phase in current frame: {}'.format(all_buds_S_IDs))
                    # print('Buds that were in S phase in both previous and current frame (old buds): {}'.format(old_buds_S_IDs))
                    # print('Buds that are in S phase ONLY in current frame (new buds): {}'.format(new_buds_S_IDs))
                    if old_buds_S_IDs:
                        current_old_buds_centroids = [rp[all_cells_ids.index(ID)].centroid for ID in old_buds_S_IDs]
                        prev_old_buds_centroids = [prev_rp[prev_IDs.index(ID)].centroid for ID in old_buds_S_IDs]
                        dist_buds = np.linalg.norm((np.asarray(prev_old_buds_centroids)-
                                                    np.asarray(current_old_buds_centroids)),
                                                    axis=1)
                        G1_df.loc[old_buds_S_IDs, 'Centr. shift'] = dist_buds
                        if isinstance(prev_frame_G1_df, pd.DataFrame):
                            p_G1_df = prev_frame_G1_df.filter(old_buds_S_IDs, axis=0)
                            old_G1_df = G1_df.filter(old_buds_S_IDs, axis=0)
                            res_col_names = ['Delta_shift', 'Delta_b_m', 'Delta_eccentricity']
                            col_to_combine = ['Centr. shift', 'Dist_b_m', 'Eccentricity']
                            old_G1_df = combine_df_col(res_col_names,
                                                       col_to_combine,
                                                       p_G1_df, old_G1_df,
                                        lambda sp, sc: np.divide((sp-sc),sp))
                            new_G1_df = G1_df.filter(new_buds_S_IDs, axis=0)
                            G1_df = pd.concat([old_G1_df, new_G1_df])
                            cond1 = abs(G1_df['Delta_shift']) > 0.6
                            cond2 = abs(G1_df['Delta_eccentricity']) > 0.1
                            cond3 = G1_df['Delta_b_m'] < 0
                            assign_G1 = (cond1 & cond2) & cond3
                            G1_df['Assign_G1'] = assign_G1
    # print('Automatic G1 dataframe for current frame:')
    # print(G1_df)
    return G1_df


#Function that given a label_img returns coords of the contour of the object
def find_contours(label_img, cells_ids, group=False, concatenate=False,
                  return_img=False):
    # label_only_cells_ids_img = np.zeros_like(label_img)
    # for id in cells_ids:
    #     label_only_cells_ids_img[label_img == id] = id
    # uint8_img = (label_only_cells_ids_img > 0).astype(np.uint8)
    # contours, hierarchy = cv2.findContours(uint8_img,cv2.RETR_LIST,
    #                                         cv2.CHAIN_APPROX_NONE)
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
        contours.append(cnt)
    if concatenate:
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
        # contours_img = np.zeros_like(label_img)
        # id = 1
        # for contour in all_contours:
        #     for y, x in contour:
        #         contours_img[y, x] = id
        #     id += 1
        # if all_contours:
        #     fig = plt.Figure()
        #     ax = fig.add_subplot()
        #     ax.imshow(contours_img)
        #     embed_tk('New contours img', [1024,768,400,150], fig)
    if return_img:
        contours_img = np.zeros_like(label_img)
        for contour in contours:
            for coords in contour:
                y = coords[0][1]
                x = coords[0][0]
                contours_img[y, x] = label_img[y, x]
        return all_contours, contours_img
    else:
        return all_contours, []

def init_cc_stage_df(all_cells_ids):
    cc_stage = ['ND' for ID in all_cells_ids]
    num_cycles = [-1]*len(all_cells_ids)
    relationship = ['ND' for ID in all_cells_ids]
    related_to = [0]*len(all_cells_ids)
    OF = np.zeros(len(all_cells_ids), bool)
    df = pd.DataFrame({
                        'Cell cycle stage': cc_stage,
                        '# of cycles': num_cycles,
                        'Relative\'s ID': related_to,
                        'Relationship': relationship,
                        'OF': OF},
                        index=all_cells_ids)
    df.index.name = 'Cell_ID'
    return df

#Assign bud to mother based on the minimum distance between contours
def cc_analysis(label_img, new_contours, old_contours, all_cells_ids,
                prev_cc_stage, region_props, prev_bud_mother_ids):
    """
    Function that assigns new objects (buds) to the mother.
    The mother is defined as the object with minimum distance between
    old objects' contours and new object's contour.

    Returns:
    --------
    all_bud_mother_ids : list of n bud_mother_ids Dict where n is the number
                         of new cells in the current frame.
                         Dict keys --> {'Min distance', 'bud_id', 'mother_id'}
    cc_stage_frame : DataFrame of N columns where N is the number
                     of ALL cells in the current frame.
                     Column names --> {'Cell cycle stage', '# of cycles'.
                                       'Bud or mother', 'Relative\'s ID'}
    """
    df = init_cc_stage_df(all_cells_ids)

    all_bud_mother_ids = []
    ID_mothers = prev_cc_stage[(prev_cc_stage['Relationship']=='mother') &
                     (prev_cc_stage['Cell cycle stage']=='S')].index.to_list()
    still_mother_warning = False
    still_mother_warning_txt = ''
    if new_contours:
        all_bud_mother_ids = [[] for i in range(len(new_contours))]
        """Iterate new objects"""
        for n, new_contour in enumerate(new_contours):
            bud_mother_ids = {'Min distance': np.inf, 'bud_id': 0, 'mother_id': 0}
            """Iterate contour's points coordinates of the current new obj"""
            for new_coords in new_contour:
                dist_new_all_old = np.linalg.norm(np.subtract(new_coords, old_contours),
                                                                                axis=1)
                new_min_dist_i = np.unravel_index(dist_new_all_old.argmin(),
                                                  dist_new_all_old.shape)
                new_min_dist = dist_new_all_old[new_min_dist_i]
                old_contour_min_dist = old_contours[new_min_dist_i]
                if new_min_dist < bud_mother_ids['Min distance']:
                    bud_ID = label_img[new_coords[0], new_coords[1]]
                    mother_ID = label_img[old_contour_min_dist[0],
                                          old_contour_min_dist[1]]
                    if mother_ID not in ID_mothers:
                        bud_mother_ids['Min distance'] = new_min_dist #new minimum found

                        bud_mother_ids['bud_id'] = bud_ID

                        bud_mother_ids['mother_id'] = mother_ID
                    else:
                        previous_bud = prev_cc_stage.at[mother_ID, 'Relative\'s ID']
            if mother_ID in ID_mothers:
                still_mother_warning_txt = (
                      'Cell ID {} should be the mother of cell ID {}, '
                      'but it is still the mother of cell ID {}.\n'
                      'You should probably check previous frames '
                      'and assign G1 (right click on the cell in intensity image)\n'
                      'to cell ID {} when you see cell division'
                      .format(mother_ID, bud_ID, previous_bud, previous_bud))
                print(still_mother_warning_txt)
            """Set values into cc_stage_df"""
            all_bud_mother_ids[n] = bud_mother_ids
            bud_ID = bud_mother_ids['bud_id'] #last ID was the minimum dist. one
            mother_ID = bud_mother_ids['mother_id']
            if mother_ID in prev_cc_stage.index:
                mother_cc_num = prev_cc_stage.at[mother_ID, '# of cycles']
                df.at[bud_ID, 'Cell cycle stage'] = 'S'
                df.at[bud_ID, '# of cycles'] = 0
                df.at[bud_ID, 'Relative\'s ID'] = mother_ID
                df.at[bud_ID, 'Relationship'] = 'bud'
                df.at[mother_ID, 'Cell cycle stage'] = 'S'
                df.at[mother_ID, '# of cycles'] = mother_cc_num
                df.at[mother_ID, 'Relative\'s ID'] = bud_ID
                df.at[mother_ID, 'Relationship'] = 'mother'
    if (prev_cc_stage.index).to_list()[0] != 0:
        for obj in region_props:
            ID = obj.label
            cc_stage = df.loc[ID]['Cell cycle stage']
            if cc_stage == 'ND':
                if ID in df.index and ID in prev_cc_stage.index:
                    df.loc[ID] = prev_cc_stage.loc[ID]
    return all_bud_mother_ids, df, still_mother_warning_txt

def manual_BudAss_df(param_dict, cc_stage_frame, frame_i):
    idxs = param_dict['Manual_BudAss_info'][frame_i]['frames']
    bidxs = param_dict['Manual_BudAss_info'][frame_i]['dictIDX']
    bud_IDs = param_dict['Manual_BudAss_info'][frame_i]['corr_buds']
    moth_IDs = param_dict['Manual_BudAss_info'][frame_i]['corr_moths']
    for i, idx, B, M in zip(idxs, bidxs, bud_IDs, moth_IDs):
        prev_mID = param_dict['Bud-mother IDs'][i][idx]['mother_id']
        df_im1 = param_dict['Cell Cycle stages'][i-1]
        cc_stage_frame.at[bud_IDs, 'Relative\'s ID'] = M
        cc_stage_frame.at[moth_IDs, 'Cell cycle stage'] = 'S'
        cc_stage_frame.at[moth_IDs, 'Relative\'s ID'] = B
        cc_stage_frame.at[moth_IDs, 'Relationship'] = 'mother'
        cc_stage_frame.loc[prev_mID] = df_im1.loc[prev_mID]
        for idx in range(i, frame_i):
            df = param_dict['Cell Cycle stages'][idx]
            df.at[bud_IDs, 'Relative\'s ID'] = M
            df.at[moth_IDs, 'Cell cycle stage'] = 'S'
            df.at[moth_IDs, 'Relative\'s ID'] = B
            df.at[moth_IDs, 'Relationship'] = 'mother'
            df.loc[prev_mID] = df_im1.loc[prev_mID]
            param_dict['Cell Cycle stages'][idx] = df
    return param_dict, cc_stage_frame

def manual_BudAss_func(param_dict, all_bud_mother_ids, frame_i):
    idxs = param_dict['Manual_BudAss_info'][frame_i]['frames']
    bidxs = param_dict['Manual_BudAss_info'][frame_i]['dictIDX']
    bud_IDs = param_dict['Manual_BudAss_info'][frame_i]['corr_buds']
    moth_IDs = param_dict['Manual_BudAss_info'][frame_i]['corr_moths']
    for i, idx, B, M in zip(idxs, bidxs, bud_IDs, moth_IDs):
        if i == frame_i:
            all_bud_mother_ids[idx]['bud_id'] = B
            all_bud_mother_ids[idx]['mother_id'] = M
        else:
            param_dict['Bud-mother IDs'][i][idx]['bud_id'] = B
            param_dict['Bud-mother IDs'][i][idx]['mother_id'] = M
    return param_dict, all_bud_mother_ids


class manual_emerg_bud:
    def __init__(self, label_img, ID, rp, eps_percent=0.01, del_small_obj=False):
        """Initialize attributes"""
        self.cancel = False
        self.ID_bud = label_img.max() + 1
        self.ID_moth = ID
        self.label_img = label_img
        self.coords_delete = []
        """Build image containing only selected ID obj"""
        only_ID_img = np.zeros_like(label_img)
        only_ID_img[label_img == ID] = ID
        all_IDs = [obj.label for obj in rp]
        obj_rp = rp[all_IDs.index(ID)]
        min_row, min_col, max_row, max_col = obj_rp.bbox
        obj_bbox_h = max_row - min_row
        obj_bbox_w = max_col - min_col
        obj_bbox_cy = min_row + obj_bbox_h/2
        obj_bbox_cx = min_col + obj_bbox_w/2
        obj_bottom = int(obj_bbox_cy - obj_bbox_h/2)
        obj_left = int(obj_bbox_cx - obj_bbox_w/2)
        obj_top = obj_bottom + obj_bbox_h
        obj_right = obj_left + obj_bbox_w
        self.xlims = (obj_left-5, obj_right+5)
        self.ylims = (obj_top+5, obj_bottom-5)
        self.only_ID_img = only_ID_img
        self.sep_bud_label = only_ID_img.copy()
        self.small_obj_mask = np.zeros(only_ID_img.shape, bool)

        """generate image plot and connect to events"""
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot()
        self.fig.subplots_adjust(bottom=0.25)
        (self.ax).imshow(self.only_ID_img)
        (self.ax).set_xlim(obj_left-5, obj_right+5)
        (self.ax).set_ylim(obj_top+5, obj_bottom-5)
        (self.ax).axis('off')
        (self.ax).set_title('Draw a line with the right button to separate cell.\n'
                            'Delete object with mouse wheel button')

        """Find convexity defects"""
        try:
            cnt, defects = self.convexity_defects(
                                              self.only_ID_img.astype(np.uint8),
                                              eps_percent)
        except:
            defects = None
        if defects is not None:
            defects_points = [0]*len(defects)
            for i, defect in enumerate(defects):
                s,e,f,d = defect[0]
                x,y = tuple(cnt[f][0])
                defects_points[i] = (y,x)
                self.ax.plot(x,y,'r.')

        """Embed plt window into a tkinter window"""
        sub_win = embed_tk('Mother-bud zoom', [1024,768,400,150], self.fig)


        """Create buttons"""
        self.ax_ok_B = self.fig.add_subplot(position=[0.2, 0.2, 0.1, 0.03])
        self.ax_undo_B = self.fig.add_subplot(position=[0.8, 0.2, 0.1, 0.03])
        self.ok_B = Button(self.ax_ok_B, 'Happy\nwith that', canvas=sub_win.canvas,
                            color='0.1', hovercolor='0.25', presscolor='0.35')
        self.undo_B = Button(self.ax_undo_B, 'Undo', canvas=sub_win.canvas,
                            color='0.1', hovercolor='0.25', presscolor='0.35')
        """Connect to events"""
        (sub_win.canvas).mpl_connect('button_press_event', self.mouse_down)
        (sub_win.canvas).mpl_connect('button_release_event', self.mouse_up)
        (sub_win.canvas).mpl_connect('key_press_event', self.key_down)
        (sub_win.canvas).mpl_connect('resize_event', self.resize)
        (sub_win.root).protocol("WM_DELETE_WINDOW", self.abort_exec)
        self.undo_B.on_clicked(self.undo)
        self.ok_B.on_clicked(self.ok)
        self.sub_win = sub_win
        sub_win.root.wm_attributes('-topmost',True)
        sub_win.root.focus_force()
        sub_win.root.after_idle(sub_win.root.attributes,'-topmost',False)
        sub_win.root.mainloop()

    def key_down(self, event):
        key = event.key
        if key == 'enter':
            self.ok(None)
        elif key == 'ctrl+z':
            self.undo(None)

    def resize(self, event):
        (self.ax_left, self.ax_bottom,
        self.ax_right, self.ax_top) = self.ax.get_position().get_points().flatten()
        B_h = 0.08
        B_w = 0.1
        self.ax_ok_B.set_position([self.ax_right-B_w, self.ax_bottom-B_h-0.01,
                                   B_w, B_h])
        self.ax_undo_B.set_position([self.ax_left, self.ax_bottom-B_h-0.01,
                                   B_w, B_h])

    def mouse_down(self, event):
        if event.inaxes == self.ax and event.button == 3:
            self.xp = int(event.xdata)
            self.yp = int(event.ydata)
            self.cid = (self.sub_win.canvas).mpl_connect('motion_notify_event',
                                                            self.draw_line)
            self.pltLine = Line2D([self.xp, self.xp], [self.yp, self.yp])
        elif event.inaxes == self.ax and event.button == 2:
            xp = int(event.xdata)
            yp = int(event.ydata)
            self.coords_delete.append((yp, xp))
            self.sep_bud_delete_lab = self.sep_bud_label.copy()
            for y, x in self.coords_delete:
                del_ID = self.sep_bud_label[y, x]
                self.sep_bud_delete_lab[self.sep_bud_delete_lab == del_ID] = 0
            rp = regionprops(self.sep_bud_delete_lab)
            self.ax.imshow(self.sep_bud_delete_lab)
            text_label_centroid(rp, self.ax, 18, 'semibold', 'center',
                                'center', None, display_ccStage=False,
                                color='k', clear=True)
            (self.sub_win.canvas).draw_idle()

    def draw_line(self, event):
        if event.inaxes == self.ax and event.button == 3:
            self.yd = int(event.ydata)
            self.xd = int(event.xdata)
            self.pltLine.set_visible(False)
            self.pltLine = Line2D([self.xp, self.xd], [self.yp, self.yd],
                                   color='r', ls='--')
            self.ax.add_line(self.pltLine)
            (self.sub_win.canvas).draw_idle()

    def nearest_nonzero(self, a, y, x):
        r, c = np.nonzero(a)
        dist = ((r - y)**2 + (c - x)**2)
        min_idx = dist.argmin()
        return a[r[min_idx], c[min_idx]]

    def mouse_up(self, event):
        if event.inaxes == self.ax and event.button == 3:
            self.r1 = int(event.ydata)
            self.c1 = int(event.xdata)
            rr, cc, val = line_aa(self.yp, self.xp, self.r1, self.c1)
            sep_bud_img = np.copy(self.sep_bud_label)
            sep_bud_img[rr, cc] = 0
            self.sep_bud_img = sep_bud_img
            (self.sub_win.canvas).draw_idle()
            (self.sub_win.canvas).mpl_disconnect(self.cid)
            sep_bud_label_0 = label(self.sep_bud_img, connectivity=2)
            sep_bud_label = remove_small_objects(sep_bud_label_0,
                                                 min_size=20,
                                                 connectivity=2)
            small_obj_mask = np.logical_xor(sep_bud_label_0>0,
                                            sep_bud_label>0)
            self.small_obj_mask = np.logical_or(small_obj_mask, self.small_obj_mask)
            rp_sep = regionprops(sep_bud_label)
            IDs = [obj.label for obj in rp_sep]
            max_ID = self.ID_bud+len(IDs)
            sep_bud_label[sep_bud_label>0] = sep_bud_label[sep_bud_label>0]+max_ID
            rp_sep = regionprops(sep_bud_label)
            IDs = [obj.label for obj in rp_sep]
            areas = [obj.area for obj in rp_sep]
            curr_ID_bud = IDs[areas.index(min(areas))]
            curr_ID_moth = IDs[areas.index(max(areas))]
            sep_bud_label[sep_bud_label==curr_ID_moth] = self.ID_moth
            # sep_bud_label = np.zeros_like(sep_bud_label)
            sep_bud_label[sep_bud_label==curr_ID_bud] = self.ID_bud+len(IDs)-2
            temp_sep_bud_lab = sep_bud_label.copy()
            self.rr = []
            self.cc = []
            self.val = []
            for r, c in zip(rr, cc):
                if self.only_ID_img[r, c] != 0:
                    ID = self.nearest_nonzero(sep_bud_label, r, c)
                    temp_sep_bud_lab[r,c] = ID
                    self.rr.append(r)
                    self.cc.append(c)
                    self.val.append(ID)
            self.sep_bud_label = temp_sep_bud_lab
            rp_sep = regionprops(sep_bud_label)
            self.rp = rp_sep
            self.num_cells = len(regionprops(sep_bud_label))
            self.ax.imshow(self.sep_bud_label)
            text_label_centroid(rp_sep, self.ax, 18, 'semibold', 'center',
                                'center', None, display_ccStage=False,
                                color='k', clear=True)
            (self.sub_win.canvas).draw_idle()

    def convexity_defects(self, img, eps_percent):
        contours, hierarchy = cv2.findContours(img,2,1)
        cnt = contours[0]
        cnt = cv2.approxPolyDP(cnt,eps_percent*cv2.arcLength(cnt,True),True) # see https://www.programcreek.com/python/example/89457/cv22.convexityDefects
        hull = cv2.convexHull(cnt,returnPoints = False) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        defects = cv2.convexityDefects(cnt,hull) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        return cnt, defects


    def undo(self, event):
        self.coords_delete = []
        sep_bud_img = np.copy(self.only_ID_img)
        self.sep_bud_img = sep_bud_img
        self.sep_bud_label = np.copy(self.only_ID_img)
        self.small_obj_mask = np.zeros(self.only_ID_img.shape, bool)
        rp = regionprops(sep_bud_img)
        self.ax.clear()
        self.ax.imshow(self.sep_bud_img)
        (self.ax).set_xlim(*self.xlims)
        (self.ax).set_ylim(*self.ylims)
        text_label_centroid(rp, self.ax, 18, 'semibold', 'center',
                            'center', None, display_ccStage=False,
                            color='k', clear=True)
        self.ax.axis('off')
        (self.sub_win.canvas).draw_idle()

    def ok(self, event):
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()

    def abort_exec(self):
        self.cancel = True
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()
        abort = messagebox.askyesno('Abort or continue',
                                    'Do you really want to abort execution?')
        if abort:
            exit('Execution aborted by the user')


class auto_emerg_bud:
    """
    Separate emerging bud by finding convexity defects and setting to 0 those
    pixels lying on the arc of a circle perimeter with center in the object
    centroid and passing through the defects.

    Parameters
    ----------
    label_img : 2D ndarray
        Labelled image of the segmeneted cells.
    ID : int
        Label of the mother-bud object.
    eps_percent: float
        tolerance of the contour approximation. This value needs to be adjusted
        until only 2 convexity defects are found at the edge of bud emergence

    """
    def __init__(self, label_img, ID, eps_percent):
        img = np.copy(label_img)
        img[img!=ID] = 0
        img.astype(np.uint8)
        label_img_only_ID = label(img)
        obj_props = regionprops(label_img_only_ID)[0]
        cnt, defects = self.convexity_defects(img, eps_percent)
        if defects is not None:
            radius, yx1, yx2 = self.get_radius_2points(defects, cnt, img, obj_props)
            centers = self.center_from_r_2points(radius, yx1, yx2)
            rr1, cc1, rr2, cc2 = self.circ_coords(centers, obj_props, radius)
            rr1, cc1 = self.arc_coords(yx1, yx2, rr1, cc1)
            rr2, cc2 = self.arc_coords(yx1, yx2, rr2, cc2)
            img[rr1, cc1] = 0
            img[rr2, cc2] = 0

        self.separate_bud_img = img
        label_separate_bud = label(img, connectivity=1)
        region_props_sep_bud = regionprops(label_separate_bud)
        label_separate_bud = self.del_min_area_obj(label_separate_bud,
                                                    region_props_sep_bud, 5)
        ID_max = label_img.max()
        label_img[label_img == ID] = 0 #remove mother-bud from label
        label_img[label_separate_bud == 1] = ID_max+1
        label_img[label_separate_bud == 2] = ID_max+2
        self.label_separate_bud = label_img


    def convexity_defects(self, img, eps_percent):
        contours, hierarchy = cv2.findContours(img,2,1)
        cnt = contours[0]
        cnt = cv2.approxPolyDP(cnt,eps_percent*cv2.arcLength(cnt,True),True) # see https://www.programcreek.com/python/example/89457/cv22.convexityDefects
        hull = cv2.convexHull(cnt,returnPoints = False) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        defects = cv2.convexityDefects(cnt,hull) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        return cnt, defects


    def get_radius_2points(self, defects, cnt, img, obj_props):
        defects_points = [0]*len(defects)
        for i, defect in enumerate(defects):
            s,e,f,d = defect[0]
            x,y = tuple(cnt[f][0])
            defects_points[i] = (y,x)
            #ax[0].plot(x,y,'r.')
        label_img = label(img)
        r, c = np.asarray(obj_props.centroid).astype(int)
        radius = np.linalg.norm(np.asarray([r,c])-np.asarray([int(y),int(x)]))
        yx1 = defects_points[0]
        yx2 = defects_points[1]
        return radius, yx1, yx2


    def center_from_r_2points(self, radius, yx1, yx2):
        """
        Calculate center coordinates of a circle with specific radius and
        passing through 2 points yx1 = (y1, x1) and yx2 = (y2, x2).

        see:
        https://math.stackexchange.com/questions/1781438/finding-the-center-of-a-circle-given-two-points-and-a-radius-algebraically

        Parameters
        ----------
        radius : float or int
            Radius of the circle.
        xy1 : tuple of (y, x) coordinates
            First point lying on the circle.
        xy2 : tuple of (y, x) coordinates
            Second point lying on the circle.

        Returns
        -------
        centers : list of 2 tuples of (y, x) coordinates
            The two possible centers.

        """
        y1, x1 = yx1
        y2, x2 = yx2
        xa = (x2 - x1)/2
        ya = (y2 - y1)/2
        x0 = x1 + xa
        y0 = y1 + ya
        a = np.sqrt(xa**2 + ya**2)
        b = np.sqrt(radius**2 - a**2)

        h1 = x0 + (b*ya)/a
        k1 = y0 - (b*xa)/a

        h2 = x0 - (b*ya)/a
        k2 = y0 + (b*xa)/a

        return [(k1, h1), (k2, h2)]


    def circ_coords(self, centers, obj_props, radius):
        centers = np.asarray(centers)
        centroids = np.asarray([obj_props.centroid, obj_props.centroid])
        dist = np.linalg.norm(centers-centroids,axis=1)
        (k1, h1) = centers[dist.argmin()]
        radius = int(radius)
        rr1, cc1 = circle_perimeter(int(k1), int(h1), radius)
        rr2, cc2 = circle_perimeter(int(k1), int(h1), radius+1)
        return rr1, cc1, rr2, cc2


    def arc_coords(self, yx1, yx2, rr, cc):
        y1, x1 = yx1
        y2, x2 = yx2
        y_li, x_li = [y1, y2], [x1, x2]
        y_li.sort(), x_li.sort()
        y_min, y_max = y_li
        x_min, x_max = x_li
        df = pd.DataFrame({'y': rr, 'x': cc})
        df = df[(df['y'] >= y_min) & (df['y'] <= y_max)]
        df = df[(df['x'] >= x_min) & (df['x'] <= x_max)]
        rr = df['y'].to_numpy()
        cc = df['x'].to_numpy()
        return rr, cc


    def del_min_area_obj(self, label_img, region_props, area_min):
        for obj in region_props:
            if obj.area <= area_min:
                label_img[label_img == obj.label] = 0
        label_img, num_obj = label(label_img, connectivity=2, return_num=True)
        return label_img

#Function to draw a line between bud-mother centroids coordinates
def line_mother_bud(current_bud_mother_ids, param_dict,
                    regionprops_label_img, ax, cc_stage_frame, frame_i):
    prev_bud_mother_ids = param_dict['Bud-mother IDs'][0:frame_i]
    all_ids = [obj.label for obj in regionprops_label_img]
    for bud_mother_ids in current_bud_mother_ids:
        bud_rp_i = all_ids.index(bud_mother_ids['bud_id'])
        mother_rp_i = all_ids.index(bud_mother_ids['mother_id'])
        bud_centroid_y, bud_centroid_x = regionprops_label_img[bud_rp_i].centroid
        mother_centroid_y, mother_centroid_x = regionprops_label_img[mother_rp_i].centroid
        ax.plot([bud_centroid_x, mother_centroid_x],
                [bud_centroid_y, mother_centroid_y],
                color='r', linestyle='--', linewidth = 3,
                dash_capstyle='round')
    for i, frame_bud_mother_ids in enumerate(prev_bud_mother_ids):
        if frame_bud_mother_ids:
            for bud_mother_ids in frame_bud_mother_ids:
                bud_ID = bud_mother_ids['bud_id']
                bud_ID_cc_stage = cc_stage_frame.at[bud_ID, 'Cell cycle stage']
                try:
                    if bud_ID_cc_stage == 'S':
                        bud_rp_i = all_ids.index(bud_mother_ids['bud_id'])
                        mother_rp_i = all_ids.index(bud_mother_ids['mother_id'])
                        bud_centroid_y, bud_centroid_x = regionprops_label_img[bud_rp_i].centroid
                        mother_centroid_y, mother_centroid_x = regionprops_label_img[mother_rp_i].centroid
                        ax.plot([bud_centroid_x, mother_centroid_x],
                                [bud_centroid_y, mother_centroid_y],
                                color='orange', linestyle='--', linewidth = 2,
                                dash_capstyle='round')
                    else:
                        param_dict['Bud-mother IDs'][i] = []
                except ValueError:
                    print('Did a previously assigned bud disappeared from current frame?')
    return param_dict


#Function for merging objects based on list of lists of IDs selected manually by the user
def merge_objs(frame_i, cells_labels_separate, num_cells_separate=None,
                rpCsep=None, param_dict=None, merged_IDs_frame=None,
                relabel=True):
    if param_dict is not None:
        merged_IDs_frame = param_dict['Merged'][frame_i]
    else:
        pass
    if merged_IDs_frame:
        for labels_press_release in merged_IDs_frame:
            min_label = labels_press_release[0]
            for id in labels_press_release[1:]:
                cells_labels_separate[cells_labels_separate == id] = min_label
        if relabel:
            cells_labels_separate, num_cells_separate = label(cells_labels_separate,
                                                              connectivity=2,
                                                              return_num=True)
        regionprops_cells_labels_separate = regionprops(cells_labels_separate)
        return cells_labels_separate, num_cells_separate, regionprops_cells_labels_separate
    else:
        return cells_labels_separate, num_cells_separate, rpCsep


#Function for deleting objects based on list of lists of IDs selected manually by the user
def delete_objs(frame_i, cells_labels_separate, num_cells_separate=None,
                rpCsep=None, param_dict=None, deleted_IDs_frame=None,
                relabel=True):
    if param_dict is not None:
        deleted_IDs_frame = param_dict['Deleted'][frame_i]
    else:
        pass
    if deleted_IDs_frame:
        for id_delete in deleted_IDs_frame:
            cells_labels_separate[cells_labels_separate == id_delete] = 0
        if relabel:
            cells_labels_separate, num_cells_separate = label(cells_labels_separate,
                                                        connectivity=2,
                                                        return_num=True)
        regionprops_cells_labels_separate = regionprops(cells_labels_separate)
        return cells_labels_separate, num_cells_separate, regionprops_cells_labels_separate
    else:
        return cells_labels_separate, num_cells_separate, rpCsep


class select_slice_toAlign:
    def __init__(self, frames, num_frames, init_slice=0, activate_ROI=False,
                 slice_used_for='segmentation',
                 title='Select slices for alignment'):
        self.activate_ROI = activate_ROI
        self.num_frames = num_frames
        self.slice = init_slice
        self.slice_start = init_slice
        self.slice_end = 0
        self.abort = True
        self.key_mode = False
        self.frames = frames
        self.data = frames[0]
        self.current = 'start'
        self.V_start = frames[0]
        self.V_end = frames[-1]
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot()
        img = self.V_start[init_slice]
        self.fig.subplots_adjust(bottom=0.20)
        sl_width = 0.6
        sl_left = 0.5 - (sl_width/2)
        ok_width = 0.13
        ok_left = 0.5 - (ok_width/2)
        (self.ax).imshow(img)
        (self.ax).axis('off')
        self.fig_title = f'Select slice for {slice_used_for}'
        (self.ax).set_title(self.fig_title)
        """Embed plt window into a tkinter window"""
        sub_win = embed_tk(title, [1024,768,400,150], self.fig)
        # [left, bottom, width, height]
        self.ax_sl = self.fig.add_subplot(
                                position=[sl_left, 0.15, sl_width, 0.03],
                                facecolor='0.1')
        self.ax_frame_sl = self.fig.add_subplot(
                                position=[sl_left, 0.15-0.03-0.01, sl_width, 0.03],
                                facecolor='0.1')
        self.sl = Slider(self.ax_sl, 'Slice', -1, len(self.V_start),
                                canvas=sub_win.canvas,
                                valinit=init_slice,
                                valstep=1,
                                color='0.2',
                                init_val_line_color='0.3',
                                valfmt='%1.0f')
        self.frame_sl = Slider(self.ax_frame_sl, 'Frame', -1, self.num_frames-1,
                                canvas=sub_win.canvas,
                                valinit=0,
                                valstep=1,
                                color='0.2',
                                init_val_line_color='0.3',
                                valfmt='%1.0f')
        (self.sl).on_changed(self.update_slice)
        (self.frame_sl).on_changed(self.update_frame)
        self.ax_ok = self.fig.add_subplot(
                                position=[ok_left, 0.04, ok_width, 0.05])
        self.ok_b = Button(self.ax_ok, 'Happy with that', canvas=sub_win.canvas,
                                color='0.1',
                                hovercolor='0.25',
                                presscolor='0.35')
        self.ax_start = self.fig.add_subplot(
                                position=[ok_left-ok_width, 0.04, ok_width, 0.05])
        self.start_b = Button(self.ax_start, 'First frame',
                                canvas=sub_win.canvas,
                                color='0.4',
                                hovercolor='0.4',
                                presscolor='0.4')
        self.is_start_frame = True
        self.ax_end = self.fig.add_subplot(
                                position=[ok_left+ok_width, 0.04, ok_width, 0.05])
        self.end_b = Button(self.ax_end, 'Last frame',
                                canvas=sub_win.canvas,
                                color='0.1',
                                hovercolor='0.4',
                                presscolor='0.35')
        (self.ok_b).on_clicked(self.ok)
        (self.start_b).on_clicked(self.show_start)
        (self.end_b).on_clicked(self.show_end)
        (sub_win.root).protocol("WM_DELETE_WINDOW", self.abort_exec)
        (sub_win.canvas).mpl_connect('key_press_event', self.set_slvalue)
        (sub_win.canvas).mpl_connect('axes_enter_event', self.set_key_mode_enter)
        (sub_win.canvas).mpl_connect('axes_leave_event', self.set_key_mode_leave)
        self.ROI_coords = None
        if activate_ROI:
            self.orig_frames = frames.copy()
            (sub_win.canvas).mpl_connect('button_press_event', self.mouse_down)
            (sub_win.canvas).mpl_connect('button_release_event', self.mouse_up)
        self.sub_win = sub_win
        sub_win.root.wm_attributes('-topmost',True)
        sub_win.root.focus_force()
        sub_win.root.after_idle(sub_win.root.attributes,'-topmost',False)
        sub_win.root.mainloop()

    def ax_transData_and_coerce(self, ax, event_x, event_y):
        x, y = ax.transData.inverted().transform((event_x, event_y))
        xmin, xmax = ax.get_xlim()
        ymax, ymin = ax.get_ylim()
        if x < xmin:
            x_coerced = int(np.ceil(xmin))
        elif x > xmax:
            x_coerced = int(np.ceil(xmax))
        else:
            x_coerced = int(round(x))
        if y < ymin:
            y_coerced = int(np.ceil(ymin))
        elif y > ymax:
            y_coerced = int(np.ceil(ymax))
        else:
            y_coerced = int(round(y))
        return x_coerced, y_coerced

    def mouse_down(self, event):
        if event.button == 3:
            self.ROI_rect_patch = Rectangle((0, 0), 1, 1, fill=False, color='r')
            self.cid_mouse_drag = self.sub_win.canvas.mpl_connect(
                                         'motion_notify_event', self.mouse_drag)
            event_x, event_y = event.x, event.y
            ax = self.ax
            xd_rc, yd_rc = self.ax_transData_and_coerce(ax, event_x, event_y)
            self.xd_rc, self.yd_rc = xd_rc, yd_rc
            self.ax.add_patch(self.ROI_rect_patch)

    def mouse_drag(self, event):
        if event.button == 3:
            self.ax.patches = []
            event_x, event_y = event.x, event.y
            ax = self.ax
            xdr_rc, ydr_rc = self.ax_transData_and_coerce(ax, event_x, event_y)
            y_start, y_end = sorted([self.yd_rc, ydr_rc])
            x_start, x_end = sorted([self.xd_rc, xdr_rc])
            rect_w = (x_end+1)-x_start
            rect_h = (y_end+1)-y_start
            ROI_rect_patch = Rectangle((x_start, y_start), rect_w, rect_h,
                                            fill=False, color='r')
            self.ax.add_patch(ROI_rect_patch)
            self.sub_win.canvas.draw_idle()


    def mouse_up(self, event):
        if event.button == 3:
            (self.sub_win.canvas).mpl_disconnect(self.cid_mouse_drag)
            event_x, event_y = event.x, event.y
            ax = self.ax
            xu_rc, yu_rc = self.ax_transData_and_coerce(ax, event_x, event_y)
            y_start, y_end = sorted([self.yd_rc, yu_rc])
            x_start, x_end = sorted([self.xd_rc, xu_rc])
            self.ROI_coords = y_start, y_end, x_start, x_end
            self.ax.patches = []
            rect_w = (x_end+1)-x_start
            rect_h = (y_end+1)-y_start
            self.ROI_rect_patch = Rectangle((x_start, y_start), rect_w, rect_h,
                                            fill=False, color='r')
            self.apply_ROI()

    def apply_ROI(self):
        self.ax.clear()
        self.ax.add_patch(self.ROI_rect_patch)
        y_start, y_end, x_start, x_end = self.ROI_coords
        frame_i = int(self.frame_sl.val)
        img = self.orig_frames[frame_i, self.slice].copy()
        r, c = img.shape
        rows = [*range(y_start), *range(y_end, r)]
        col = [*range(x_start), *range(x_end, c)]
        img[rows] = 0
        img[:, col] = 0
        self.ax.imshow(img)
        self.ax.axis('off')
        (self.ax).set_title(self.fig_title)
        self.sub_win.canvas.draw_idle()

    def update_frame(self, event):
        frame_i = int(self.frame_sl.val)
        img = self.frames[frame_i, self.slice]
        self.ax.imshow(img)
        self.fig.canvas.draw_idle()

    def show_start(self, event):
        if not self.is_start_frame:
            self.start_b.color = '0.4'
            self.end_b.color = '0.1'
            self.start_b.ax.set_facecolor('0.4')
            self.end_b.ax.set_facecolor('0.1')
            self.is_start_frame = True
            self.fig.canvas.draw_idle()
        self.data = self.V_start
        self.current = 'start'
        img = self.data[self.slice]
        self.frame_sl.set_val(0, silent=True)
        self.ax.imshow(img)
        self.fig.canvas.draw_idle()

    def show_end(self, event):
        if self.is_start_frame:
            self.start_b.color = '0.1'
            self.end_b.color = '0.4'
            self.start_b.ax.set_facecolor('0.1')
            self.end_b.ax.set_facecolor('0.4')
            self.is_start_frame = False
            self.fig.canvas.draw_idle()
        self.data = self.V_end
        self.current = 'end'
        self.slice_end = self.slice
        img = self.data[self.slice]
        self.frame_sl.set_val(self.num_frames-1, silent=True)
        self.ax.imshow(img)
        self.fig.canvas.draw_idle()

    def set_key_mode_enter(self, event):
        if event.inaxes == self.ax_frame_sl:
            self.key_mode = True

    def set_key_mode_leave(self, event):
        if event.inaxes == self.ax_frame_sl:
            self.key_mode = False

    def set_slvalue(self, event):
        if event.key == 'left':
            if self.key_mode:
                self.frame_sl.set_val(self.frame_sl.val - 1)
            else:
                self.sl.set_val(self.sl.val - 1)
        if event.key == 'right':
            if self.key_mode:
                self.frame_sl.set_val(self.frame_sl.val + 1)
            else:
                self.sl.set_val(self.sl.val + 1)
        if event.key == 'enter':
            self.ok(None)

    def update_slice(self, val):
        self.slice = int(val)
        if self.current == 'start':
            self.slice_start = self.slice
        else:
            self.slice_end = self.slice
        frame_i = int(self.frame_sl.val)
        img = self.frames[frame_i, self.slice]
        self.ax.imshow(img)
        self.fig.canvas.draw_idle()

    def ok(self, event):
        close_without_slice_end = True
        if self.slice_end == 0:
            close_without_slice_end = messagebox.askyesno('Close without slice_end',
                          'You didn\'t set any slice number for the last frame.\n'
                          f'Do you want to use the same slice ({self.slice_start}) '
                          'of the first frame?',
                          master=self.sub_win.root)
        m = (self.slice_end - self.slice_start)/(self.num_frames - 1)
        q = self.slice_start
        self.slices = [round(m*x + q) for x in range(self.num_frames)]
        close_without_ROI = True
        if self.activate_ROI and self.ROI_coords is None:
            close_without_ROI = messagebox.askyesno('Close without ROI',
                          'You didn\'t draw any region of interest.\n'
                          'Are you sure you want to segment the entire frame?',
                          master=self.sub_win.root)
        if close_without_ROI and close_without_slice_end:
            plt.close(self.fig)
            self.sub_win.root.quit()
            self.sub_win.root.destroy()

    def abort_exec(self):
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()
        exit('Execution aborted by the user')


class auto_select_slice:
    def __init__(self, frame_V, init_slice=0, activate_ROI=False,
                 slice_used_for='segmentation'):
        init_slice = self.auto_slice(frame_V)
        self.slice = init_slice
        self.abort = True
        self.data = frame_V
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot()
        self.fig.subplots_adjust(bottom=0.20)
        sl_width = 0.6
        sl_left = 0.5 - (sl_width/2)
        ok_width = 0.13
        ok_left = 0.5 - (ok_width/2)
        (self.ax).imshow(frame_V[init_slice])
        (self.ax).axis('off')
        (self.ax).set_title(f'Select slice for {slice_used_for}')
        """Embed plt window into a tkinter window"""
        sub_win = embed_tk('Mother-bud zoom', [1024,768,400,150], self.fig)
        self.ax_sl = self.fig.add_subplot(
                                position=[sl_left, 0.12, sl_width, 0.04],
                                facecolor='0.1')
        self.sl = Slider(self.ax_sl, 'Slice', -1, len(frame_V),
                                canvas=sub_win.canvas,
                                valinit=init_slice,
                                valstep=1,
                                color='0.2',
                                init_val_line_color='0.3',
                                valfmt='%1.0f')
        (self.sl).on_changed(self.update_slice)
        self.ax_ok = self.fig.add_subplot(
                                position=[ok_left, 0.05, ok_width, 0.05],
                                facecolor='0.1')
        self.ok_b = Button(self.ax_ok, 'Happy with that', canvas=sub_win.canvas,
                                color='0.1',
                                hovercolor='0.25',
                                presscolor='0.35')
        (self.ok_b).on_clicked(self.ok)
        (sub_win.root).protocol("WM_DELETE_WINDOW", self.abort_exec)
        (sub_win.canvas).mpl_connect('key_press_event', self.set_slvalue)
        self.ROI_coords = None
        if activate_ROI:
            self.orig_frame_V = frame_V
            (sub_win.canvas).mpl_connect('button_press_event', self.mouse_down)
            (sub_win.canvas).mpl_connect('button_release_event', self.mouse_up)
        self.sub_win = sub_win
        sub_win.root.wm_attributes('-topmost',True)
        sub_win.root.focus_force()
        #sub_win.root.after_idle(sub_win.root.attributes,'-topmost',False)
        sub_win.root.mainloop()

    def ax_transData_and_coerce(self, ax, event_x, event_y):
        x, y = ax.transData.inverted().transform((event_x, event_y))
        xmin, xmax = ax.get_xlim()
        ymax, ymin = ax.get_ylim()
        if x < xmin:
            x_coerced = int(np.ceil(xmin))
        elif x > xmax:
            x_coerced = int(np.floor(xmax))
        else:
            x_coerced = int(round(x))
        if y < ymin:
            y_coerced = int(np.ceil(ymin))
        elif y > ymax:
            y_coerced = int(np.floor(ymax))
        else:
            y_coerced = int(round(y))
        return x_coerced, y_coerced

    def mouse_down(self, event):
        if event.button == 3:
            self.ROI_rect_patch = Rectangle((0, 0), 1, 1, fill=False, color='r')
            self.cid_mouse_drag = self.sub_win.canvas.mpl_connect(
                                         'motion_notify_event', self.mouse_drag)
            event_x, event_y = event.x, event.y
            ax = self.ax
            xd_rc, yd_rc = self.ax_transData_and_coerce(ax, event_x, event_y)
            self.xd_rc, self.yd_rc = xd_rc, yd_rc
            self.ax.add_patch(self.ROI_rect_patch)

    def mouse_drag(self, event):
        if event.button == 3:
            self.ax.patches = []
            event_x, event_y = event.x, event.y
            ax = self.ax
            xdr_rc, ydr_rc = self.ax_transData_and_coerce(ax, event_x, event_y)
            y_start, y_end = sorted([self.yd_rc, ydr_rc])
            x_start, x_end = sorted([self.xd_rc, xdr_rc])
            rect_w = (x_end+1)-x_start
            rect_h = (y_end+1)-y_start
            ROI_rect_patch = Rectangle((x_start, y_start), rect_w, rect_h,
                                            fill=False, color='r')
            self.ax.add_patch(ROI_rect_patch)
            self.sub_win.canvas.draw_idle()


    def mouse_up(self, event):
        if event.button == 3:
            (self.sub_win.canvas).mpl_disconnect(self.cid_mouse_drag)
            event_x, event_y = event.x, event.y
            ax = self.ax
            xu_rc, yu_rc = self.ax_transData_and_coerce(ax, event_x, event_y)
            y_start, y_end = sorted([self.yd_rc, yu_rc])
            x_start, x_end = sorted([self.xd_rc, xu_rc])
            self.ROI_coords = y_start, y_end, x_start, x_end
            self.ax.patches = []
            rect_w = (x_end+1)-x_start
            rect_h = (y_end+1)-y_start
            self.ROI_rect_patch = Rectangle((x_start, y_start), rect_w, rect_h,
                                            fill=False, color='r')
            self.apply_ROI()

    def apply_ROI(self):
        self.ax.clear()
        self.ax.add_patch(self.ROI_rect_patch)
        y_start, y_end, x_start, x_end = self.ROI_coords
        img = self.orig_frame_V[int(self.sl.val)].copy()
        r, c = img.shape
        rows = [*range(y_start), *range(y_end, r)]
        col = [*range(x_start), *range(x_end, c)]
        img[rows] = 0
        img[:, col] = 0
        self.ax.imshow(img)
        self.ax.axis('off')
        self.sub_win.canvas.draw_idle()



    def resize_widgets(self, event):
        # [left, bottom, width, height]
        pass

    def auto_slice(self, frame_V):
        # https://stackoverflow.com/questions/6646371/detect-which-image-is-sharper
        means = []
        for i, img in enumerate(frame_V):
            edge = sobel(img)
            means.append(np.mean(edge))
        slice = means.index(max(means))
        print('Best slice = {}'.format(slice))
        return slice

    def set_slvalue(self, event):
        if event.key == 'left':
            self.sl.set_val(self.sl.val - 1)
        if event.key == 'right':
            self.sl.set_val(self.sl.val + 1)
        if event.key == 'enter':
            self.ok(None)

    def update_slice(self, val):
        self.slice = int(val)
        img = self.data[int(val)]
        self.ax.imshow(img)
        self.fig.canvas.draw_idle()

    def ok(self, event):
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()

    def abort_exec(self):
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()
        exit('Execution aborted by the user')


def manual_emBud_func(frame_i, cells_labels_separate, num_cells_separate=None,
        rpCsep=None, ID=None, param_dict=None):
    if ID is None:
        IDs = param_dict['Manual_emBud_info'][frame_i]['IDs']
        cells_labels_separate = param_dict['Manual_emBud_info'][frame_i]['cells_labels_separate']
    else:
        bud_em = manual_emerg_bud(cells_labels_separate, ID, rpCsep)
        bud_em_IDimg = bud_em.sep_bud_label
        cells_labels_separate[bud_em_IDimg!=0] = bud_em_IDimg[bud_em_IDimg!=0]
    num_cells_separate = len(regionprops(cells_labels_separate))
    return cells_labels_separate, num_cells_separate, rpCsep


class draw_ROI_2D_frames:
    def __init__(self, frames, num_frames, activate_ROI=False,
                 slice_used_for='segmentation'):
        self.num_frames = num_frames
        self.abort = True
        self.key_mode = False
        self.frames = frames
        self.data = frames[0]
        self.current = 'start'
        self.img_frame_0 = frames[0]
        self.img_frame_end = frames[-1]
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot()
        self.fig.subplots_adjust(bottom=0.20)
        sl_width = 0.6
        sl_left = 0.5 - (sl_width/2)
        ok_width = 0.13
        ok_left = 0.5 - (ok_width/2)
        (self.ax).imshow(self.img_frame_0)
        (self.ax).axis('off')
        (self.ax).set_title(f'Draw ROI with mouse right-button for {slice_used_for}')
        """Embed plt window into a tkinter window"""
        sub_win = embed_tk('Select slices for alignment', [1024,768,400,150], self.fig)
        # [left, bottom, width, height]
        self.ax_frame_sl = self.fig.add_subplot(
                                position=[sl_left, 0.15-0.03-0.01, sl_width, 0.03],
                                facecolor='0.1')
        self.frame_sl = Slider(self.ax_frame_sl, 'Frame', -1, self.num_frames-1,
                                #canvas=sub_win.canvas,
                                valinit=0,
                                valstep=1,
                                color='0.2',
                                #init_val_line_color='0.3',
                                valfmt='%1.0f')
        (self.frame_sl).on_changed(self.update_frame)
        self.ax_ok = self.fig.add_subplot(
                                position=[ok_left, 0.04, ok_width, 0.05])
        self.ok_b = Button(self.ax_ok, 'Happy with that',
                           #canvas=sub_win.canvas,
                                color='0.2',
                                hovercolor='0.25')
                                #presscolor='0.35')
        self.ax_start = self.fig.add_subplot(
                                position=[ok_left-ok_width, 0.04, ok_width, 0.05])
        self.start_b = Button(self.ax_start, 'First frame',
                              #canvas=sub_win.canvas,
                                color='0.1',
                                hovercolor='0.25')
                                #presscolor='0.35')
        self.ax_end = self.fig.add_subplot(
                                position=[ok_left+ok_width, 0.04, ok_width, 0.05])
        self.end_b = Button(self.ax_end, 'Last frame',
                            #canvas=sub_win.canvas,
                                color='0.1',
                                hovercolor='0.25')
                             #   presscolor='0.35')
        (self.ok_b).on_clicked(self.ok)
        (self.start_b).on_clicked(self.show_start)
        (self.end_b).on_clicked(self.show_end)
        (sub_win.root).protocol("WM_DELETE_WINDOW", self.abort_exec)
        (sub_win.canvas).mpl_connect('key_press_event', self.set_slvalue)
        (sub_win.canvas).mpl_connect('axes_enter_event', self.set_key_mode_enter)
        (sub_win.canvas).mpl_connect('axes_leave_event', self.set_key_mode_leave)
        self.ROI_coords = None
        if activate_ROI:
            self.orig_frames = frames.copy()
            (sub_win.canvas).mpl_connect('button_press_event', self.mouse_down)
            (sub_win.canvas).mpl_connect('button_release_event', self.mouse_up)
        self.sub_win = sub_win
        sub_win.root.wm_attributes('-topmost',True)
        sub_win.root.focus_force()
        #sub_win.root.after_idle(sub_win.root.attributes,'-topmost',False)
        sub_win.root.mainloop()

    def ax_transData_and_coerce(self, ax, event_x, event_y):
        x, y = ax.transData.inverted().transform((event_x, event_y))
        xmin, xmax = ax.get_xlim()
        ymax, ymin = ax.get_ylim()
        if x < xmin:
            x_coerced = int(np.ceil(xmin))
        elif x > xmax:
            x_coerced = int(np.ceil(xmax))
        else:
            x_coerced = int(round(x))
        if y < ymin:
            y_coerced = int(np.ceil(ymin))
        elif y > ymax:
            y_coerced = int(np.ceil(ymax))
        else:
            y_coerced = int(round(y))
        return x_coerced, y_coerced

    def mouse_down(self, event):
        if event.button == 3:
            self.ROI_rect_patch = Rectangle((0, 0), 1, 1, fill=False, color='r')
            self.cid_mouse_drag = self.sub_win.canvas.mpl_connect(
                                         'motion_notify_event', self.mouse_drag)
            event_x, event_y = event.x, event.y
            ax = self.ax
            xd_rc, yd_rc = self.ax_transData_and_coerce(ax, event_x, event_y)
            self.xd_rc, self.yd_rc = xd_rc, yd_rc
            self.ax.add_patch(self.ROI_rect_patch)

    def mouse_drag(self, event):
        if event.button == 3:
            self.ax.patches = []
            event_x, event_y = event.x, event.y
            ax = self.ax
            xdr_rc, ydr_rc = self.ax_transData_and_coerce(ax, event_x, event_y)
            y_start, y_end = sorted([self.yd_rc, ydr_rc])
            x_start, x_end = sorted([self.xd_rc, xdr_rc])
            rect_w = (x_end+1)-x_start
            rect_h = (y_end+1)-y_start
            ROI_rect_patch = Rectangle((x_start, y_start), rect_w, rect_h,
                                            fill=False, color='r')
            self.ax.add_patch(ROI_rect_patch)
            self.sub_win.canvas.draw_idle()


    def mouse_up(self, event):
        if event.button == 3:
            (self.sub_win.canvas).mpl_disconnect(self.cid_mouse_drag)
            event_x, event_y = event.x, event.y
            ax = self.ax
            xu_rc, yu_rc = self.ax_transData_and_coerce(ax, event_x, event_y)
            y_start, y_end = sorted([self.yd_rc, yu_rc])
            x_start, x_end = sorted([self.xd_rc, xu_rc])
            self.ROI_coords = y_start, y_end, x_start, x_end
            self.ax.patches = []
            rect_w = (x_end+1)-x_start
            rect_h = (y_end+1)-y_start
            self.ROI_rect_patch = Rectangle((x_start, y_start), rect_w, rect_h,
                                            fill=False, color='r')
            self.apply_ROI()

    def apply_ROI(self):
        self.ax.clear()
        self.ax.add_patch(self.ROI_rect_patch)
        y_start, y_end, x_start, x_end = self.ROI_coords
        frame_i = int(self.frame_sl.val)
        img = self.orig_frames[frame_i].copy()
        r, c = img.shape
        rows = [*range(y_start), *range(y_end, r)]
        col = [*range(x_start), *range(x_end, c)]
        img[rows] = 0
        img[:, col] = 0
        self.ax.imshow(img)
        self.ax.axis('off')
        self.sub_win.canvas.draw_idle()

    def update_frame(self, event):
        frame_i = int(self.frame_sl.val)
        img = self.frames[frame_i]
        self.ax.imshow(img)
        self.fig.canvas.draw_idle()

    def show_start(self, event):
        self.data = self.img_frame_0
        self.current = 'start'
        img = self.data.copy()
        self.frame_sl.set_val(0, silent=True)
        self.ax.imshow(img)
        self.fig.canvas.draw_idle()

    def show_end(self, event):
        self.data = self.img_frame_end
        img = self.data.copy()
        self.frame_sl.set_val(self.num_frames-1, silent=True)
        self.ax.imshow(img)
        self.fig.canvas.draw_idle()

    def set_key_mode_enter(self, event):
        if event.inaxes == self.ax_frame_sl:
            self.key_mode = True

    def set_key_mode_leave(self, event):
        if event.inaxes == self.ax_frame_sl:
            self.key_mode = False

    def set_slvalue(self, event):
        if event.key == 'left':
            self.frame_sl.set_val(self.frame_sl.val - 1)
        if event.key == 'right':
            self.frame_sl.set_val(self.frame_sl.val + 1)
        if event.key == 'enter':
            self.ok(None)

    def ok(self, event):
        close_without_ROI = True
        if self.ROI_coords is None:
            close_without_ROI = messagebox.askyesno('Close without ROI',
                          'You didn\'t draw any region of interest.\n'
                          'Are you sure you want to segment the entire frame?',
                          master=self.sub_win.root)
        if close_without_ROI:
            plt.close(self.fig)
            self.sub_win.root.quit()
            self.sub_win.root.destroy()

    def abort_exec(self):
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()
        exit('Execution aborted by the user')

#TODO: evaluate if it's worth to keep this customized function instead of trying to use tk code
def file_dialog(**options):
    #Prompt the user to select the image file
    root = tk.Tk()
    root.withdraw()
    path = tk.askopenfilename(**options)
    root.destroy()
    return path

#TODO: evaluate if it's worth to keep this customized function instead of trying to use tk code
def folder_dialog(**options):
    #Prompt the user to select the image file
    root = tk.Tk()
    root.withdraw()
    path = tk.filedialog.Directory(**options).show()
    root.destroy()
    return path

def dark_mode():
    plt.style.use('dark_background')
    plt.rc('axes', edgecolor='0.1')

#TODO: evaluate if it's worth to keep this customized class instead of trying to use tk code
class win_size:
    def __init__(self, w=1, h=1, swap_screen=False):
        monitor = Display()
        screens = monitor.get_screens()
        num_screens = len(screens)
        displ_w = int(screens[0].width*w)
        displ_h = int(screens[0].height*h)
        x_displ = screens[0].x
        #Display plots maximized window
        mng = plt.get_current_fig_manager()
        if swap_screen:
            geom = "{}x{}+{}+{}".format(displ_w,(displ_h-70),(displ_w-8), 0)
            mng.window.wm_geometry(geom) #move GUI window to second monitor
                                         #with string "widthxheight+x+y"
        else:
            geom = "{}x{}+{}+{}".format(displ_w,(displ_h-70),-8, 0)
            mng.window.wm_geometry(geom) #move GUI window to second monitor
                                         #with string "widthxheight+x+y"

#TODO: evaluate if it's worth to keep this customized class instead of trying to use tk code
class embed_tk:
    """Example:
    -----------
    img = np.ones((600,600))
    fig = plt.Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot()
    ax.imshow(img)

    sub_win = embed_tk('Embeddding in tk', [1024,768,300,100], fig)

    def on_key_event(event):
        print('you pressed %s' % event.key)

    sub_win.canvas.mpl_connect('key_press_event', on_key_event)

    sub_win.root.mainloop()

    print('cazz')
    """
    def __init__(self, win_title, geom, fig):
        root = tk.Tk()
        root.wm_title(win_title)
        root.geometry("{}x{}+{}+{}".format(*geom)) # WidthxHeight+Left+Top
        # a tk.DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas = canvas
        self.toolbar = toolbar
        self.root = root

#TODO: evaluate if it's worth to keep this customized class instead of trying to use tk code
class tk_breakpoint:
    '''Geometry: "WidthxHeight+Left+Top" '''
    def __init__(self, title='Breakpoint', geometry="230x120+2500+400",
                 message='Breakpoint', button_1_text='Continue',
                 button_2_text='Abort', button_3_text='Delete breakpoint'):
        self.abort = False
        self.next_i = False
        self.del_breakpoint = False
        self.title = title
        self.geometry = geometry
        self.message = message
        self.button_1_text = button_1_text
        self.button_2_text = button_2_text
        self.button_3_text = button_3_text

    def pausehere(self):
        global root
        if not self.del_breakpoint:
            root = tk.Tk()
            root.lift()
            root.attributes("-topmost", True)
            root.title(self.title)
            root.geometry(self.geometry)
            tk.Label(root,
                     text=self.message,
                     font=(None, 11)).grid(row=0, column=0,
                                           columnspan=2, pady=4, padx=4)

            tk.Button(root,
                      text=self.button_1_text,
                      command=self.continue_button,
                      width=10,).grid(row=4,
                                      column=0,
                                      pady=8, padx=8)

            tk.Button(root,
                      text=self.button_2_text,
                      command=self.abort_button,
                      width=15).grid(row=4,
                                     column=1,
                                     pady=8, padx=8)
            tk.Button(root,
                      text=self.button_3_text,
                      command=self.delete_breakpoint,
                      width=20).grid(row=5,
                                     column=0,
                                     columnspan=2)

            root.mainloop()

    def continue_button(self):
        self.next_i=True
        root.quit()
        root.destroy()

    def delete_breakpoint(self):
        self.del_breakpoint=True
        root.quit()
        root.destroy()

    def abort_button(self):
        self.abort=True
        exit('Execution aborted by the user')
        root.quit()
        root.destroy()

class cc_stage_df_frame0:
    """Display a tkinter window where the user initializes values of
    the cc_stage_df for frame 0.

    Parameters
    ----------
    cells_IDs : list or ndarray of int
        List of cells IDs.

    Attributes
    ----------
    cell_IDs : list or ndarray of int
        List of cells IDs.
    root : class 'tkinter.Tk'
        tkinter.Tk root window
    """
    def __init__(self, cells_IDs, cc_stage_df):
        self.cancel = False
        self.df = None
        """Options and labels"""
        cc_stage_opt = ["S", "G1"]
        if len(cells_IDs) == 1:
            related_to_opt = [-1]
        else:
            related_to_opt = [str(ID) for ID in cells_IDs]
            related_to_opt.insert(0, -1)
        relationship_opt = ["mother", "bud"]
        self.cell_IDs = cells_IDs
        self.cc_stage_df = cc_stage_df

        """Root window"""
        root = tk.Toplevel()
        root.lift()
        root.attributes("-topmost", True)
        root.title('Cell cycle stage for frame 0')
        root.geometry('+600+500')
        self.root = root

        """Cells IDs label column"""
        col = 0
        cells_IDs_colTitl = tk.Label(root,
                                  text='Cell_ID',
                                  font=(None, 11))
        cells_IDs_colTitl.grid(row=0, column=col, pady=4, padx=4)
        for row, ID in enumerate(cells_IDs):
            cells_IDs_label = tk.Label(root,
                                      text=str(ID),
                                      font=(None, 10))
            cells_IDs_label.grid(row=row+1, column=col, pady=4, padx=8)

        """Cell cycle stage column"""
        col = 1
        cc_stage_colTitl = tk.Label(root,
                                  text='Cell cycle stage',
                                  font=(None, 11))
        cc_stage_colTitl.grid(row=0, column=col, pady=4, padx=4)
        self.cc_stage_list = []
        init_ccs = cc_stage_df.loc[cells_IDs]['Cell cycle stage'].to_list()
        for row, ID in enumerate(cells_IDs):
            cc_stage_var = tk.StringVar(root)
            if len(cells_IDs) == 1:
                cc_stage_var.set(init_ccs[row]) # default value
            else:
                cc_stage_var.set(init_ccs[row]) # default value
            cc_stage = tk.OptionMenu(root, cc_stage_var, *cc_stage_opt)
            cc_stage.grid(row=row+1, column=col, pady=4, padx=4)
            self.cc_stage_list.append(cc_stage_var)

        """# of cycles column"""
        col = 2
        num_cycles_colTitl = tk.Label(root,
                                  text='# of cycles',
                                  font=(None, 11))
        num_cycles_colTitl.grid(row=0, column=col, pady=4, padx=4)
        self.num_cycles_list = []
        init_num_cycles = cc_stage_df.loc[cells_IDs]['# of cycles'].to_list()
        for row, ID in enumerate(cells_IDs):
            num_cycles = tk.Entry(root, width=10, justify='center')
            num_cycles.insert(0, str(init_num_cycles[row]))
            num_cycles.grid(row=row+1, column=col, pady=4, padx=4)
            self.num_cycles_list.append(num_cycles)

        """Relative's ID column"""
        col = 3
        related_to_colTitl = tk.Label(root,
                                  text='Relative\'s ID',
                                  font=(None, 11))
        related_to_colTitl.grid(row=0, column=col, pady=4, padx=4)
        self.related_to_list = []
        self.relto_varNames = []
        IDs = cells_IDs.copy() if len(cells_IDs) > 1 else [-1]
        init_rel_ID = cc_stage_df.loc[cells_IDs]['Relative\'s ID'].to_list()
        for row, ID in enumerate(IDs):
            related_to_var = tk.StringVar(root, name='rel_to_{}'.format(ID))
            temp_cb = related_to_var.trace('w', self.store_varName)
            related_to_var.set(str(init_rel_ID[row])) # default value
            related_to_var.trace_vdelete('w', temp_cb)
            related_to_var.trace('w', self.update_relID)
            related_to = tk.OptionMenu(root, related_to_var, *related_to_opt)
            related_to.grid(row=row+1, column=col, pady=4, padx=4)
            self.related_to_list.append(related_to_var)

        """Relationship column"""
        col = 4
        relationship_colTitl = tk.Label(root,
                                        text='Relationship',
                                        font=(None, 11))
        relationship_colTitl.grid(row=0, column=col, pady=4, padx=4)
        self.relationship_list = []
        init_relationship = cc_stage_df.loc[cells_IDs]['Relationship'].to_list()
        for row, ID in enumerate(cells_IDs):
            relationship_var = tk.StringVar(root)
            relationship_var.trace('w', self.changeNumCycle)
            relationship_var.set(str(init_relationship[row])) # default value
            relationship = tk.OptionMenu(root, relationship_var,
                                         *relationship_opt)
            relationship.grid(row=row+1, column=col, pady=4, padx=4)
            self.relationship_list.append(relationship_var)

        """OK button"""
        col = 2
        ok_b = tk.Button(root, text='OK!', width=10,
                         command=self.return_df)
        ok_b.grid(row=len(cells_IDs)+1, column=col, pady=8)

        self.root.protocol("WM_DELETE_WINDOW", self.handle_close)
        self.root.focus_force()
        self.root.mainloop()

    def store_varName(self, *args):
        self.relto_varNames.append(args[0])

    def update_relID(self, *args):
        var_name = args[0]
        idxID_var = self.relto_varNames.index(var_name)
        ID_var = self.cell_IDs[idxID_var]
        relID = int(self.root.getvar(name=var_name))
        if relID in self.cell_IDs:
            idxID = list(self.cell_IDs).index(relID)
            self.related_to_list[idxID].set(str(ID_var))

    def changeNumCycle(self, *args):
        relationships = [var.get() for var in self.relationship_list]
        idx_bud = [i for i, var in zip(range(len(relationships)), relationships)
                            if var == 'bud']
        for i in idx_bud:
            self.num_cycles_list[i].delete(0, 'end')
            self.num_cycles_list[i].insert(0, '0')
        idx_moth = [i for i, var in zip(range(len(relationships)), relationships)
                            if var == 'mother']
        for i in idx_moth:
            self.num_cycles_list[i].delete(0, 'end')
            self.num_cycles_list[i].insert(0, '1')

    def handle_close(self):
        self.cancel = True
        self.root.quit()
        self.root.destroy()
        # exit('Execution aborted by the user')

    def return_df(self):
        cc_stage = [var.get() for var in self.cc_stage_list]
        num_cycles = [int(var.get()) for var in self.num_cycles_list]
        related_to = [int(var.get()) for var in self.related_to_list]
        relationship = [var.get() for var in self.relationship_list]
        check_rel = [ID == rel for ID, rel in zip(self.cell_IDs, related_to)]
        check_buds_S = [ccs=='S' and rel_ship=='bud' and not numc==0
                        for ccs, rel_ship, numc
                        in zip(cc_stage, relationship, num_cycles)]
        check_mothers = [rel_ship=='mother' and not numc>=1
                         for rel_ship, numc
                         in zip(relationship, num_cycles)]
        check_buds_G1 = [ccs=='G1' and rel_ship=='bud'
                         for ccs, rel_ship
                         in zip(cc_stage, relationship)]
        if any(check_rel):
            messagebox.showerror('Cell ID == Relative\'s ID', 'Some cells are '
                    'mother or bud of itself. Make sure that the Relative\'s ID'
                    ' is different from the Cell ID!')
        elif any(check_buds_S):
            messagebox.showerror('Bud in S not in 0 num. cycles', 'Some buds '
                'in S phase do not have 0 as num. cycles!\n'
                'Buds in S phase must have 0 as "# of cycles"')
        elif any(check_mothers):
            messagebox.showerror('Mother not in >=1 num. cycles',
                'Some mother cells do not have >=1 as "# of cycles"!\n'
                'Mothers MUST have >1 as "# of cycles"')
        elif any(check_buds_G1):
            messagebox.showerror('Buds in G1!',
                'Some buds are in G1 phase!\n'
                'Buds MUST be in S phase')
        else:
            df = pd.DataFrame({
                                'Cell cycle stage': cc_stage,
                                '# of cycles': num_cycles,
                                'Relative\'s ID': related_to,
                                'Relationship': relationship},
                                index=self.cell_IDs)
            df.index.name = 'Cell_ID'
            df = pd.concat([df, self.cc_stage_df], axis=1)
            df = df.loc[:,~df.columns.duplicated()]
            self.df = df
            relationship_groups = df.groupby('Relationship').groups
            if 'bud' in relationship_groups.keys():
                buds_IDs = relationship_groups['bud'].to_list()
                mothers_IDs = relationship_groups['mother'].to_list()
                all_bud_mother_ids = []
                bud_mother_ids = {'Min distance': np.inf, 'bud_id': 0, 'mother_id': 0}
                for bud_ID, mother_ID in zip(buds_IDs, mothers_IDs):
                    bud_mother_ids['bud_id'] = bud_ID
                    bud_mother_ids['mother_id'] = mother_ID
                    all_bud_mother_ids.append(bud_mother_ids.copy())
                self.all_bud_mother_ids = all_bud_mother_ids
            else:
                self.all_bud_mother_ids = []
            self.root.quit()
            self.root.destroy()

class CellInt_slideshow:
    def __init__(self, frames, slice, num_frames, frame_i, CCAdfs, rps,
                 cell_cycle_analysis, num_slices=50):
        self.frames = frames
        self.slice = slice
        self.frame_i = frame_i
        self.do_update_frame = True
        self.new_IDs = []
        self.lost_IDs = []
        self.is_prev_frame = False
        fig = plt.Figure()
        ax = fig.add_subplot()
        fig.subplots_adjust(bottom=0.2)
        inlay = embed_tk('Cell intensity image slideshow',
                                    [900,768,500,70], fig)
        sl_width = 0.6
        sl_left = 0.5 - (sl_width/2)
        sl_height = 0.04
        ax_sl = fig.add_subplot(position=[sl_left, 0.12, sl_width, sl_height],
                                facecolor='0.1')
        ax_slice_sl = fig.add_subplot(position=[
                                sl_left, 0.11-sl_height, sl_width, sl_height],
                                facecolor='0.1')
        self.ax_ccstage_radio = fig.add_subplot(position=[
                                        sl_left, 0.1, 0.11, 0.12],
                                      facecolor='0.1')
        sl = Slider(ax_sl, 'Frame', -1, num_frames-1,
                                    valinit=0,
                                    valstep=1,
                                    color='0.2',
                                    init_val_line_color='0.3',
                                    valfmt='%1.0f')
        slice_sl = Slider(ax_slice_sl, 'Slice', -1, num_slices,
                                    valinit=slice,
                                    valstep=1,
                                    color='0.2',
                                    init_val_line_color='0.3',
                                    valfmt='%1.0f')
        self.radio_b_ccStage = RadioButtons(self.ax_ccstage_radio,
                                      ('show IDs',
                                      'Disable'),
                                      active = 0,
                                      activecolor = '0.4',
                                      orientation = 'horizontal',
                                      size = 59,
                                      circ_p_color = '0.4',
                                      canvas=inlay.canvas)
        self.current_radio_label = 'show IDs'
        cid = inlay.canvas.mpl_connect('pick_event',
                                        self._clicked)
        self.radio_b_ccStage.cids.append(cid)
        sl.on_changed(self.update_frame)
        slice_sl.on_changed(self.update_slice)
        self.radio_b_ccStage.on_clicked(self.update_txt)
        (inlay.canvas).mpl_connect('resize_event', self.resize_widgets)
        (inlay.canvas).mpl_connect('key_press_event', self.key_down)
        (inlay.canvas).mpl_connect('axes_enter_event', self.axes_enter)
        (inlay.canvas).mpl_connect('axes_leave_event', self.axes_leave)
        (inlay.root).protocol("WM_DELETE_WINDOW", self.close)

        self.fig = fig
        self.ax_sl = ax_sl
        self.ax_slice_sl = ax_slice_sl
        self.ax = ax
        self.sl = sl
        self.slice_sl = slice_sl
        self.inlay = inlay
        self.num_frames = num_frames
        self.CCAdfs = CCAdfs
        self.rps = rps
        self.cell_cycle_analysis = cell_cycle_analysis

    def _clicked(self, event):
        self.radio_b_ccStage._clicked(event)

    def run(self):
        self.sl.set_val(self.frame_i, silent=True)
        self.update_frame(self.frame_i)
        self.inlay.root.lift()
        self.inlay.root.attributes('-topmost',True)
        self.inlay.root.after_idle(self.inlay.root.attributes,'-topmost',False)
        self.inlay.root.focus_force()
        self.inlay.root.mainloop()


    def resize_widgets(self, event):
        # [left, bottom, width, height]
        ax_l, ax_b, ax_r, ax_t = self.ax.get_position().get_points().flatten()
        self.ax_ccstage_radio.set_position([ax_l, ax_t+0.008, (ax_r-ax_l)/2, 0.04])


    def update_slice(self, val):
        slice = int(val)
        self.slice = slice
        num_frames = self.num_frames
        frame_i = self.frame_i
        img = self.frames[frame_i, slice]
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.axis('off')
        self.inlay.canvas.draw_idle()

    def update_frame(self, val, update_txt=True):
        num_frames = self.num_frames
        frame_i = int(val)
        self.frame_i = frame_i
        self.rp = self.rps[frame_i]
        self.CCAdf = self.CCAdfs[frame_i]
        img = self.frames[frame_i, self.slice]
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.axis('off')
        if update_txt:
            self.update_txt(self.current_radio_label)
        self.inlay.canvas.draw_idle()

    def update_txt(self, radio_label):
        self.current_radio_label = radio_label
        if radio_label == 'show IDs':
            for t in self.ax.texts:
                t.set_visible(False)
            for region in self.rp:
                y, x = region.centroid
                ID = region.label
                if self.cell_cycle_analysis:
                    if ID in self.CCAdf.index:
                        ID_info = self.CCAdf.loc[ID]
                        cc_stage = ID_info['Cell cycle stage']
                        cycle_num = ID_info['# of cycles']
                        cc_txt = '{}-{}'.format(cc_stage, cycle_num)
                        txt = 'ND' if cc_stage=='ND' else cc_txt
                else:
                    txt = str(ID)
                if ID in self.lost_IDs:
                    color = 'r'
                    fontweight = 'semibold'
                elif ID in self.new_IDs:
                    color = 'limegreen'
                    fontweight = 'semibold'
                else:
                    color = '0.8'
                    fontweight = 'normal'
                self.ax.text(int(x), int(y), txt, fontsize=10,
                        fontweight=fontweight, horizontalalignment='center',
                        verticalalignment='center', color=color)
        else:
            self.update_frame(self.frame_i, update_txt=False)

    def key_down(self, event):
        key = event.key
        if key == 'left':
            if self.do_update_frame:
                self.sl.set_val(self.sl.val-1)
            else:
                self.slice_sl.set_val(self.slice_sl.val-1)
        elif key == 'right':
            if self.do_update_frame:
                self.sl.set_val(self.sl.val+1)
            else:
                self.slice_sl.set_val(self.slice_sl.val+1)
        elif 'enter':
            pass

    def axes_enter(self, event):
        if event.inaxes == self.ax_slice_sl:
            self.do_update_frame = False
        # elif event.inaxes == app.ax[2] and ia.clos_mode == 'Manual closing':
        #     app.mng.window.configure(cursor='circle')

    def axes_leave(self, event):
        if event.inaxes == self.ax_slice_sl:
            self.do_update_frame = True

    def close(self):
        self.inlay.root.quit()
        self.inlay.root.destroy()


class CellInt_slideshow_2D:
    def __init__(self, frames, num_frames, frame_i, CCAdfs, rps,
                 cell_cycle_analysis, ax_limits=None):
        self.frames = frames
        self.frame_i = frame_i
        if ax_limits is not None:
            self.ax_limits = ax_limits[0]
        else:
            self.ax_limits = None
        fig = plt.Figure()
        ax = fig.add_subplot()
        fig.subplots_adjust(bottom=0.2)
        inlay = embed_tk('Cell intensity image slideshow',
                                    [900,768,500,70], fig)
        sl_width = 0.6
        sl_left = 0.5 - (sl_width/2)
        ax_sl = fig.add_subplot(position=[sl_left, 0.12, sl_width, 0.04],
                                facecolor='0.1')
        sl = Slider(ax_sl, 'Frame', -1, num_frames+1,
                                    valinit=0,
                                    valstep=1,
                                    color='0.2',
                                    init_val_line_color='0.3',
                                    valfmt='%1.0f')
        sl.on_changed(self.update_img)
        (inlay.canvas).mpl_connect('resize_event', self.resize)
        (inlay.canvas).mpl_connect('key_press_event', self.key_down)
        (inlay.root).protocol("WM_DELETE_WINDOW", self.close)

        self.fig = fig
        self.ax_sl = ax_sl
        self.ax = ax
        self.sl = sl
        self.inlay = inlay
        self.num_frames = num_frames
        self.CCAdfs = CCAdfs
        self.rps = rps
        self.cell_cycle_analysis = cell_cycle_analysis

    def run(self):
        self.sl.set_val(self.frame_i, silent=True)
        self.update_img(self.frame_i)
        self.inlay.root.lift()
        self.inlay.root.attributes('-topmost',True)
        self.inlay.root.after_idle(self.inlay.root.attributes,'-topmost',False)
        self.inlay.root.focus_force()
        self.inlay.root.mainloop()

    def resize(self, event):
        pass

    def set_lims(self):
        if self.ax_limits is not None:
            self.ax.set_xlim(*self.ax_limits[0])
            self.ax.set_ylim(*self.ax_limits[1])

    def on_xlim_changed(self, axes):
        xlim = self.ax.get_xlim()
        self.ax_limits[0] = xlim

    def on_ylim_changed(self, axes):
        ylim = self.ax.get_ylim()
        self.ax_limits[1] = ylim

    def connect_axes_cb(self):
        self.cidx = self.ax.callbacks.connect('xlim_changed', self.on_xlim_changed)
        self.cidy = self.ax.callbacks.connect('ylim_changed', self.on_ylim_changed)

    def update_img(self, val):
        num_frames = self.num_frames
        frame_i = int(val)
        if frame_i < num_frames:
            rp = self.rps[frame_i]
            CCAdf = self.CCAdfs[frame_i]
            img = self.frames[frame_i]
            self.ax.clear()
            self.ax.imshow(img)
            self.ax.axis('off')
            self.ax.set_title('Current frame: {}/{}'.format(frame_i, num_frames))
            self.update_txt(rp, CCAdf)
            self.set_lims()
            self.connect_axes_cb()
            self.inlay.canvas.draw_idle()
        else:
            self.sl.set_val(self.frame_i, silent=True)

    def update_txt(self, rp, CCAdf):
        for t in self.ax.texts:
            t.set_visible(False)
        for region in rp:
            y, x = region.centroid
            ID = region.label
            if self.cell_cycle_analysis:
                if ID in CCAdf.index:
                    ID_info = CCAdf.loc[ID]
                    cc_stage = ID_info['Cell cycle stage']
                    cycle_num = ID_info['# of cycles']
                    cc_txt = '{}-{}'.format(cc_stage, cycle_num)
                    txt = 'ND' if cc_stage=='ND' else cc_txt
            else:
                txt = str(ID)
            self.ax.text(int(x), int(y), txt, fontsize=10,
                    fontweight='semibold', horizontalalignment='center',
                    verticalalignment='center', color='r')

    def key_down(self, event):
        key = event.key
        val = self.sl.val
        if key == 'left':
            self.sl.set_val(val-1)
        elif key == 'right':
            self.sl.set_val(val+1)
        elif 'enter':
            pass

    def close(self):
        self.inlay.root.quit()
        self.inlay.root.destroy()


class newID_app:
    def __init__(self, old_ID=None, second_button=False):
        root = tk.Toplevel()
        root.lift()
        root.attributes("-topmost", True)
        root.geometry("+800+400")
        self._root = root
        if old_ID is not None:
            label_txt = f'ID = {old_ID} will be replaced\n with new ID'
        else:
            label_txt = 'New ID'
        tk.Label(root, text=label_txt, font=(None, 10)).grid(row=0, columnspan=2)
        ID_strvar = tk.StringVar()
        ID = tk.Entry(root, justify='center', textvariable=ID_strvar)
        ID_strvar.trace_add("write", self._close_brackets)
        ID.grid(row=1, padx=16, pady=4, columnspan=2)
        ID.focus_force()
        if second_button:
            self.ok_for_all_butt = tk.Button(root, command=self._ok_for_all_cb,
                            text='Ok for all next frames',state=tk.DISABLED)
            self.ok_for_all_butt.grid(row=2, pady=4, column=1, padx=4)
            self.ok_butt = tk.Button(root, command=self._quit, text='Ok!',
                                           width=10)
            self.ok_butt.grid(row=2, pady=4, column=0, padx=4)
        else:
            self.ok_butt = tk.Button(root, command=self._quit, text='Ok!',
                                           width=10)
            self.ok_butt.grid(row=2, pady=4, padx=4, columnspan=2)
        tk.Label(root, text='NOTE:\n You can write a list of tuples:\n'
                            '[(old ID, new ID), ...]', font=(None, 10)
                            ).grid(row=3, pady=4, columnspan=2)
        root.bind('<Return>', self._quit)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.entry = ID
        root.mainloop()

    def _close_brackets(self, name=None, index=None, mode=None):
        txt = self.entry.get()
        input_idx = self.entry.index(tk.INSERT)
        input_txt = txt[input_idx-1:input_idx]
        if input_txt == '(':
            self.entry.insert(tk.INSERT, ')')
            self.entry.icursor(self.entry.index(tk.INSERT)-1)
        elif input_txt == '[':
            self.entry.insert(tk.INSERT, ']')
            self.entry.icursor(self.entry.index(tk.INSERT)-1)
        try:
            int(self.entry.get())
            self.ok_for_all_butt['state'] = tk.NORMAL
        except:
            try:
                self.ok_for_all_butt['state'] = tk.DISABLED
            except:
                pass

    def _ok_for_all_cb(self, event=None):
        self.ok_for_all = True
        txt = self.entry.get()
        if txt.find('[') != -1:
            self.new_ID = literal_eval(txt)
        else:
            self.new_ID = int(self.entry.get())
        self._root.quit()
        self._root.destroy()

    def _quit(self, event=None):
        self.ok_for_all = False
        txt = self.entry.get()
        if txt.find('[') != -1:
            self.new_ID = literal_eval(txt)
        else:
            self.new_ID = int(self.entry.get())
        self._root.quit()
        self._root.destroy()

    def on_closing(self):
        self._root.quit()
        self._root.destroy()
        # exit('Execution aborted by the user')

def xyc_r_arc_3points(x1, y1, x2, y2, x3, y3):
    # Calculate center coordinates (xc, yc) of the circle passing through 3 points
    # see http://mathforum.org/library/drmath/view/55239.html
    Nxc = np.array([[x1**2+y1**2, y1, 1],
                    [x2**2+y2**2, y2, 1],
                    [x3**2+y3**2, y3, 1]])
    Nyc = np.array([[x1, x1**2+y1**2, 1],
                    [x2, x2**2+y2**2, 1],
                    [x3, x3**2+y3**2, 1]])
    Dc = np.array([[x1, y1, 1],
                   [x2, y2, 1],
                   [x3, y3, 1]])

    D = 2*det(Dc)
    xc = det(Nxc)/D
    yc = det(Nyc)/D

    r = norm(np.subtract([xc, yc],[x1, y1]))
    return xc, yc, r

def auto_tracking(prev_frame, current_frame, prev_rp, current_rp):
    prev_IDs = [obj.label for obj in prev_rp]
    IDs_left_toassign = prev_IDs.copy()
    for ID in prev_IDs:
        I_Array = current_frame[prev_frame == ID]
        I_IDs = np.unique(I_Array)
        I_counts = [np.count_nonzero(I_Array == I_ID) if I_ID!=0 else 0
                                                      for I_ID in I_IDs]
        if I_counts:
            max_idx = I_counts.index(max(I_counts))
            current_ID = I_IDs[max_idx]
            IDs_left_toassign.remove(ID)
            if current_ID != ID:
                # current_frame[current_frame == current_ID] = ID
                print('Cell ID {} in current frame now has a the new ID {}'
                      .format(current_ID, ID))
    if IDs_left_toassign:
        warning_txt = ('Cells with ID {} disappeared '
                      'in current frame'.format(IDs_left_toassign))
        print(warning_txt)
    return current_frame

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


class threebuttonsmessagebox:
    '''Geometry: "WidthxHeight+Left+Top" '''
    def __init__(self, title, geometry, message, button_1_text,
                 button_2_text, button_3_text, path):
        global root
        self.path=path
        self.append=False
        root = tk.Tk()
        root.lift()
        root.attributes("-topmost", True)
        root.title(title)
        root.geometry(geometry)
        tk.Label(root,
                 text=message,
                 font=(None, 11)).grid(row=0, column=0, columnspan=2, pady=4, padx=4)

        tk.Button(root,
                  text=button_1_text,
                  command=self.append_button,
                  width=10,).grid(row=4,
                                  column=0,
                                  pady=8, padx=8)

        tk.Button(root,
                  text=button_2_text,
                  command=self.close,
                  width=15).grid(row=4,
                                 column=1,
                                 pady=8, padx=8)
        tk.Button(root,
                  text=button_3_text,
                  command=self.open_path_explorer,
                  width=25).grid(row=5,
                                 column=0,
                                 columnspan=2)

        root.mainloop()

    def append_button(self):
        self.append=True
        root.quit()
        root.destroy()

    def open_path_explorer(self):
        subprocess.Popen('explorer "{}"'.format(os.path.normpath(self.path)))

    def close(self):
        root.quit()
        root.destroy()

class twobuttonsmessagebox:
    '''Geometry: "WidthxHeight+Left+Top" '''
    def __init__(self, title, message, button_1_text, button_2_text,
                 geometry="+800+400"):
        self.button_left=False
        root = tk.Tk()
        self.root = root
        root.attributes("-topmost", True)
        root.title(title)
        root.geometry(geometry)
        tk.Label(root,
                 text=message,
                 font=(None, 11)).grid(row=0, column=0, columnspan=2, pady=4,
                                       padx=4)

        tk.Button(root,
                  text=button_1_text,
                  command=self.button_left_cb).grid(row=4,
                                 column=0,
                                 pady=16, padx=16, sticky=tk.W)

        tk.Button(root,
                  text=button_2_text,
                  command=self.button_right_cb).grid(row=4,
                                 column=1,
                                 pady=16, padx=16, sticky=tk.E)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        root.eval('tk::PlaceWindow . center')
        root.mainloop()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

    def button_left_cb(self):
        self.button_left=True
        self.root.quit()
        self.root.destroy()

    def button_right_cb(self):
        self.root.quit()
        self.root.destroy()

class single_entry_messagebox:
    def __init__(self, *, entry_label='Entry 1', input_txt='', toplevel=True):
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

class ShowWindow_from_title:
    def __init__(self, win_title, SW_MODE=9):
        # see here for showing window option
        # https://docs.microsoft.com/en-gb/windows/win32/api/winuser/nf-winuser-showwindow?redirectedfrom=MSDN
        self.WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL,
                                              wintypes.HWND,
                                              wintypes.LPARAM)
        self.user32 = ctypes.windll.user32
        self.user32.EnumWindows.argtypes = [
            self.WNDENUMPROC,
            wintypes.LPARAM]
        self.user32.GetWindowTextLengthW.argtypes = [
            wintypes.HWND]
        self.user32.GetWindowTextW.argtypes = [
            wintypes.HWND,
            wintypes.LPWSTR,
            ctypes.c_int]
        self.win_title = win_title
        self.window_found = False
        self.SW_MODE = SW_MODE
        self._show()

    def _worker(self, hwnd, lParam):
        length = self.user32.GetWindowTextLengthW(hwnd) + 1
        buffer = ctypes.create_unicode_buffer(length)
        self.user32.GetWindowTextW(hwnd, buffer, length)
        if self.win_title in repr(buffer.value):
            # print("Restoring window: ", repr(buffer.value))
            self.user32.ShowWindow(hwnd, self.SW_MODE)
            self.user32.SetForegroundWindow(hwnd)
            self.window_found = True
        return True

    def _show(self):
        cb_worker = self.WNDENUMPROC(self._worker)
        if not self.user32.EnumWindows(cb_worker, 42):
            print(ctypes.WinError())

class select_exp_folder:
    def run_widget(self, values, current=0,
                   title='Select Position folder',
                   label_txt="Select \'Position_n\' folder to analyze:",
                   showinexplorer_button=False,
                   full_paths=None,
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
        # Label
        ttk.Label(root, text = label_txt,
                  font = (None, 10)).grid(column=0, row=0, padx=10, pady=10)

        # Combobox
        pos_n_sv = tk.StringVar()
        self.pos_n_sv = pos_n_sv
        self.values = values
        pos_b_combob = ttk.Combobox(root, textvariable=pos_n_sv, width=width)
        pos_b_combob['values'] = values
        pos_b_combob.grid(column=1, row=0, padx=10)
        pos_b_combob.current(current)


        # Ok button
        ok_b = ttk.Button(root, text='Ok!', comman=self._close)
        ok_b.grid(column=0, row=1, pady=10, sticky=tk.E)


        # Show in explorer button
        if showinexplorer_button:
            show_expl_button = ttk.Button(root, text='Show in explorer',
                                          comman=self.open_path_explorer)
            show_expl_button.grid(column=1, row=1, pady=10)

        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root = root
        if len(values) > 1:
            root.mainloop()
        else:
            self._close()
        try:
            val = pos_n_sv.get()
            idx = list(self.values).index(val)
            return self.pos_foldernames[idx]
        except:
            try:
                sv_txt = self.pos_n_sv.get()
                sv_idx = self.values.index(sv_txt)
                path = self.full_paths[sv_idx]
                return path
            except:
                return pos_n_sv.get()

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
        self.root.quit()
        self.root.destroy()

    def on_closing(self):
        exit('Execution aborted by the user')

class beyond_listdir_pos:
    def __init__(self, folder_path):
        self.bp = tk_breakpoint()
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
        parent_path = path
        count = 0
        while parent_path != self.folder_path or count==10:
            if count > 0:
                rel_path = f'{os.path.basename(parent_path)}/{rel_path}'
            parent_path = os.path.dirname(parent_path)
            count += 1
        rel_path = f'.../{rel_path}'
        return rel_path

    def count_segmented_pos(self):
        all_exp_info = []
        for path in self.TIFFs_path:
            foldername = os.path.basename(path)
            if foldername == 'TIFFs':
                pos_foldernames = os.listdir(path)
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
                                s_n = int(m[0])
                                if s_n == pos_n:
                                    if filename.find('segm.npy') != -1:
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
                                      f'Position_n folder:\n\n {images_path}',
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
                    print('Position_n folders that contain multiple segm.npy files:\n'
                          f'{pos_paths_multi_segm}')
                    exp_info = f'{rel_path} (WARNING: multiple "segm.npy" files found!)'
                else:
                    exp_info = rel_path
            else:
                rel_path = self.get_rel_path(path)
                exp_info = f'{rel_path} (FIJI macro not executed!)'
            all_exp_info.append(exp_info)
        return all_exp_info

def expand_labels(label_image, distance=1):
    """Expand labels in label image by ``distance`` pixels without overlapping.
    Given a label image, ``expand_labels`` grows label regions (connected components)
    outwards by up to ``distance`` pixels without overflowing into neighboring regions.
    More specifically, each background pixel that is within Euclidean distance
    of <= ``distance`` pixels of a connected component is assigned the label of that
    connected component.
    Where multiple connected components are within ``distance`` pixels of a background
    pixel, the label value of the closest connected component will be assigned (see
    Notes for the case of multiple labels at equal distance).
    Parameters
    ----------
    label_image : ndarray of dtype int
        label image
    distance : float
        Euclidean distance in pixels by which to grow the labels. Default is one.
    Returns
    -------
    enlarged_labels : ndarray of dtype int
        Labeled array, where all connected regions have been enlarged
    Notes
    -----
    Where labels are spaced more than ``distance`` pixels are apart, this is
    equivalent to a morphological dilation with a disc or hyperball of radius ``distance``.
    However, in contrast to a morphological dilation, ``expand_labels`` will
    not expand a label region into a neighboring region.
    This implementation of ``expand_labels`` is derived from CellProfiler [1]_, where
    it is known as module "IdentifySecondaryObjects (Distance-N)" [2]_.
    There is an important edge case when a pixel has the same distance to
    multiple regions, as it is not defined which region expands into that
    space. Here, the exact behavior depends on the upstream implementation
    of ``scipy.ndimage.distance_transform_edt``.
    See Also
    --------
    :func:`skimage.measure.label`, :func:`skimage.segmentation.watershed`, :func:`skimage.morphology.dilation`
    References
    ----------
    .. [1] https://cellprofiler.org
    .. [2] https://github.com/CellProfiler/CellProfiler/blob/082930ea95add7b72243a4fa3d39ae5145995e9c/cellprofiler/modules/identifysecondaryobjects.py#L559
    Examples
    --------
    >>> labels = np.array([0, 1, 0, 0, 0, 0, 2])
    >>> expand_labels(labels, distance=1)
    array([1, 1, 1, 0, 0, 2, 2])
    Labels will not overwrite each other:
    >>> expand_labels(labels, distance=3)
    array([1, 1, 1, 1, 2, 2, 2])
    In case of ties, behavior is undefined, but currently resolves to the
    label closest to ``(0,) * ndim`` in lexicographical order.
    >>> labels_tied = np.array([0, 1, 0, 2, 0])
    >>> expand_labels(labels_tied, 1)
    array([1, 1, 1, 2, 2])
    >>> labels2d = np.array(
    ...     [[0, 1, 0, 0],
    ...      [2, 0, 0, 0],
    ...      [0, 3, 0, 0]]
    ... )
    >>> expand_labels(labels2d, 1)
    array([[2, 1, 1, 0],
           [2, 2, 0, 0],
           [2, 3, 3, 0]])
    """

    distances, nearest_label_coords = distance_transform_edt(
        label_image == 0, return_indices=True
    )
    labels_out = np.zeros_like(label_image)
    dilate_mask = distances <= distance
    # build the coordinates to find nearest labels,
    # in contrast to [1] this implementation supports label arrays
    # of any dimension
    masked_nearest_label_coords = [
        dimension_indices[dilate_mask]
        for dimension_indices in nearest_label_coords
    ]
    nearest_labels = label_image[tuple(masked_nearest_label_coords)]
    labels_out[dilate_mask] = nearest_labels
    return labels_out

def file_dialog(**options):
    #Prompt the user to select the image file
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(**options)
    root.destroy()
    return path

class imshow_tk:
    def __init__(self, img, dots_coords=None, x_idx=1, axis=None):
        fig = plt.Figure()
        ax = fig.add_subplot()
        ax.imshow(img)
        if dots_coords is not None:
            ax.plot(dots_coords[:,x_idx], dots_coords[:,x_idx-1], 'r.')
        if axis:
            ax.axis('off')
        sub_win = embed_tk('Imshow embedded in tk', [800,600,400,150], fig)
        sub_win.root.protocol("WM_DELETE_WINDOW", self._close)
        self.sub_win = sub_win
        sub_win.root.wm_attributes('-topmost',True)
        sub_win.root.focus_force()
        sub_win.root.after_idle(sub_win.root.attributes,'-topmost',False)
        sub_win.root.mainloop()

    def _close(self):
        self.sub_win.root.quit()
        self.sub_win.root.destroy()

class my_paint_app:
    def __init__(self, label_img, ID, rp, eps_percent=0.01, del_small_obj=False,
                 overlay_img=None):
        """Initialize attributes"""
        self.cancel = False
        self.ID_bud = label_img.max() + 1
        self.ID_moth = ID
        self.label_img = label_img
        self.coords_delete = []
        self.overlay_img = overlay_img
        self.num_cells = 1
        """Build image containing only selected ID obj"""
        only_ID_img = np.zeros_like(label_img)
        only_ID_img[label_img == ID] = ID
        all_IDs = [obj.label for obj in rp]
        obj_rp = rp[all_IDs.index(ID)]
        min_row, min_col, max_row, max_col = obj_rp.bbox
        obj_bbox_h = max_row - min_row
        obj_bbox_w = max_col - min_col
        side_len = max([obj_bbox_h, obj_bbox_w])
        obj_bbox_cy = min_row + obj_bbox_h/2
        obj_bbox_cx = min_col + obj_bbox_w/2
        obj_bottom = int(obj_bbox_cy - side_len/2)
        obj_left = int(obj_bbox_cx - side_len/2)
        obj_top = obj_bottom + side_len
        obj_right = obj_left + side_len
        self.xlims = (obj_left-5, obj_right+5)
        self.ylims = (obj_top+5, obj_bottom-5)
        self.only_ID_img = only_ID_img
        self.sep_bud_label = only_ID_img.copy()
        self.eraser_mask = np.zeros(self.label_img.shape, bool)
        self.small_obj_mask = np.zeros(only_ID_img.shape, bool)

        """generate image plot and connect to events"""
        self.fig = plt.Figure()
        self.ax = self.fig.add_subplot()
        self.fig.subplots_adjust(bottom=0.25)
        (self.ax).imshow(self.only_ID_img)
        (self.ax).set_xlim(obj_left-5, obj_right+5)
        (self.ax).set_ylim(obj_top+5, obj_bottom-5)
        (self.ax).axis('off')
        (self.fig).suptitle('Draw a curve with the right button to separate cell.\n'
                            'Delete object with mouse wheel button\n'
                            'Erase with mouse left button', y=0.95)

        """Find convexity defects"""
        try:
            cnt, defects = self.convexity_defects(
                                              self.only_ID_img.astype(np.uint8),
                                              eps_percent)
        except:
            defects = None
        if defects is not None:
            defects_points = [0]*len(defects)
            for i, defect in enumerate(defects):
                s,e,f,d = defect[0]
                x,y = tuple(cnt[f][0])
                defects_points[i] = (y,x)
                self.ax.plot(x,y,'r.')

        """Embed plt window into a tkinter window"""
        sub_win = embed_tk('Mother-bud zoom', [1024,768,400,150], self.fig)


        """Create buttons"""
        self.ax_ok_B = self.fig.add_subplot(position=[0.2, 0.2, 0.1, 0.03])
        self.ax_overlay_B = self.fig.add_subplot(position=[0.8, 0.2, 0.1, 0.03])
        self.ok_B = Button(self.ax_ok_B, 'Happy\nwith that', canvas=sub_win.canvas,
                            color='0.1', hovercolor='0.25', presscolor='0.35')
        self.overlay_B = Button(self.ax_overlay_B, 'Overlay',
                            canvas=sub_win.canvas,
                            color='0.1', hovercolor='0.25', presscolor='0.35')
        """Connect to events"""
        (sub_win.canvas).mpl_connect('button_press_event', self.mouse_down)
        (sub_win.canvas).mpl_connect('button_release_event', self.mouse_up)
        self.cid_brush_circle = (sub_win.canvas).mpl_connect(
                                                    'motion_notify_event',
                                                    self.draw_brush_circle)
        (sub_win.canvas).mpl_connect('key_press_event', self.key_down)
        (sub_win.canvas).mpl_connect('resize_event', self.resize)
        (sub_win.root).protocol("WM_DELETE_WINDOW", self.abort_exec)
        self.overlay_B.on_clicked(self.toggle_overlay)
        self.ok_B.on_clicked(self.ok)
        self.sub_win = sub_win
        self.clicks_count = 0
        self.brush_size = 2
        self.eraser_on = True
        self.overlay_on = False
        self.set_labRGB_colors()
        sub_win.root.wm_attributes('-topmost',True)
        sub_win.root.focus_force()
        sub_win.root.after_idle(sub_win.root.attributes,'-topmost',False)
        sub_win.root.mainloop()

    def toggle_overlay(self, event):
        self.overlay_on = not self.overlay_on
        self.update_img(None)

    def set_labRGB_colors(self):
        # Generate a colormap as sparse as possible given the max ID.
        gradient = np.linspace(255, 0, self.num_cells, dtype=int)
        labelRGB_colors = np.asarray([plt.cm.viridis(i) for i in gradient])
        self.labRGB_colors = labelRGB_colors

    def key_down(self, event):
        key = event.key
        if key == 'enter':
            self.ok(None)
        elif key == 'ctrl+z':
            self.undo(None)
        elif key == 'up':
            self.brush_size += 1
            self.draw_brush_circle(event)
        elif key == 'down':
            self.brush_size -= 1
            self.draw_brush_circle(event)
        elif key == 'x':
            # Switch eraser mode on or off
            self.eraser_on = not self.eraser_on
            self.draw_brush_circle(event)

    def resize(self, event):
        # [left, bottom, width, height]
        (self.ax_left, self.ax_bottom,
        self.ax_right, self.ax_top) = self.ax.get_position().get_points().flatten()
        B_h = 0.08
        B_w = 0.1
        self.ax_ok_B.set_position([self.ax_right-B_w, self.ax_bottom-B_h-0.01,
                                   B_w, B_h])
        self.ax_overlay_B.set_position([self.ax_left, self.ax_bottom-B_h-0.01,
                                   B_w*2, B_h])
        if self.overlay_img is None:
            self.ax_overlay_B.set_visible(False)

    def update_img(self, event):
        lab = self.sep_bud_label.copy()
        for y, x in self.coords_delete:
            del_ID = self.sep_bud_label[y, x]
            lab[lab == del_ID] = 0
        rp = regionprops(lab)
        num_cells = len(rp)
        if self.num_cells != num_cells:
            self.set_labRGB_colors()
        if not self.overlay_on:
            img = lab
        else:
            img = label2rgb(lab,image=self.overlay_img,
                                       bg_label=0,
                                       bg_color=(0.1,0.1,0.1),
                                       colors=self.labRGB_colors)
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.set_xlim(*self.xlims)
        self.ax.set_ylim(*self.ylims)
        self.ax.axis('off')
        text_label_centroid(rp, self.ax, 18, 'semibold', 'center',
                            'center', None, display_ccStage=False,
                            color='k', clear=True)
        (self.sub_win.canvas).draw_idle()

    def mouse_down(self, event):
        if event.inaxes == self.ax and event.button == 3:
            x = int(event.xdata)
            y = int(event.ydata)
            if self.clicks_count == 0:
                self.x0 = x
                self.y0 = y
                self.cid_line = (self.sub_win.canvas).mpl_connect(
                                                         'motion_notify_event',
                                                                self.draw_line)
                self.pltLine = Line2D([self.x0, self.x0], [self.y0, self.y0])
                self.clicks_count = 1
            elif self.clicks_count == 1:
                self.x1 = x
                self.y1 = y
                (self.sub_win.canvas).mpl_disconnect(self.cid_line)
                self.cid_bezier = (self.sub_win.canvas).mpl_connect(
                                                         'motion_notify_event',
                                                              self.draw_bezier)
                self.clicks_count = 2
            elif self.clicks_count == 2:
                self.x2 = x
                self.y2 = y
                (self.sub_win.canvas).mpl_disconnect(self.cid_bezier)
                self.separate_cb()
                self.clicks_count = 0

        elif event.inaxes == self.ax and event.button == 2:
            xp = int(event.xdata)
            yp = int(event.ydata)
            self.coords_delete.append((yp, xp))
            self.update_img(None)

        elif event.inaxes == self.ax and event.button == 1:
            (self.sub_win.canvas).mpl_disconnect(self.cid_brush_circle)
            self.cid_brush = (self.sub_win.canvas).mpl_connect(
                                                     'motion_notify_event',
                                                          self.apply_brush)

    def apply_brush(self, event):
        x, y = self.ax_transData_and_coerce(self.ax, event.x, event.y,
                                                    self.label_img.shape)
        rr, cc = disk((y, x), radius=self.brush_size,
                              shape=self.label_img.shape)
        if self.eraser_on:
            self.sep_bud_label[rr, cc] = 0
            self.eraser_mask[rr, cc] = True
        else:
            self.sep_bud_label[rr, cc] = self.ID_moth
        self.update_img(None)
        c = 'r' if self.eraser_on else 'g'
        self.brush_circle = matplotlib.patches.Circle((x, y),
                                radius=self.brush_size,
                                fill=False,
                                color=c, lw=2)
        (self.ax).add_patch(self.brush_circle)
        (self.sub_win.canvas).draw_idle()


    def draw_line(self, event):
        if event.inaxes == self.ax:
            self.yd = int(event.ydata)
            self.xd = int(event.xdata)
            self.pltLine.set_visible(False)
            self.pltLine = Line2D([self.x0, self.xd], [self.y0, self.yd],
                                   color='r', ls='--')
            self.ax.add_line(self.pltLine)
            (self.sub_win.canvas).draw_idle()

    def draw_bezier(self, event):
        self.xd, self.yd = self.ax_transData_and_coerce(self.ax, event.x,
                                                                 event.y,
                                                    self.label_img.shape)
        try:
            self.plt_bezier.set_visible(False)
        except:
            pass
        p0 = (self.x0, self.y0)
        p1 = (self.xd, self.yd)
        p2 = (self.x1, self.y1)
        self.plt_bezier = PathPatch(
                          mpath.Path([p0, p1, p2],
                                     [mpath.Path.MOVETO,
                                      mpath.Path.CURVE3,
                                      mpath.Path.CURVE3]),
                                     fc="none", transform=self.ax.transData,
                                     color='r')
        self.ax.add_patch(self.plt_bezier)
        (self.sub_win.canvas).draw_idle()

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



    def nearest_nonzero(self, a, y, x):
        r, c = np.nonzero(a)
        dist = ((r - y)**2 + (c - x)**2)
        min_idx = dist.argmin()
        return a[r[min_idx], c[min_idx]]

    def separate_cb(self):
        c0, r0 = (self.x0, self.y0)
        c1, r1 = (self.x2, self.y2)
        c2, r2 = (self.x1, self.y1)
        rr, cc = bezier_curve(r0, c0, r1, c1, r2, c2, 1)
        sep_bud_img = np.copy(self.sep_bud_label)
        sep_bud_img[rr, cc] = 0
        self.sep_bud_img = sep_bud_img
        sep_bud_label_0 = label(self.sep_bud_img, connectivity=1)
        sep_bud_label = remove_small_objects(sep_bud_label_0,
                                             min_size=20,
                                             connectivity=2)
        small_obj_mask = np.logical_xor(sep_bud_label_0>0,
                                        sep_bud_label>0)
        self.small_obj_mask = np.logical_or(small_obj_mask,
                                            self.small_obj_mask)
        rp_sep = regionprops(sep_bud_label)
        IDs = [obj.label for obj in rp_sep]
        max_ID = self.ID_bud+len(IDs)
        sep_bud_label[sep_bud_label>0] = sep_bud_label[sep_bud_label>0]+max_ID
        rp_sep = regionprops(sep_bud_label)
        IDs = [obj.label for obj in rp_sep]
        areas = [obj.area for obj in rp_sep]
        curr_ID_bud = IDs[areas.index(min(areas))]
        curr_ID_moth = IDs[areas.index(max(areas))]
        sep_bud_label[sep_bud_label==curr_ID_moth] = self.ID_moth
        # sep_bud_label = np.zeros_like(sep_bud_label)
        sep_bud_label[sep_bud_label==curr_ID_bud] = self.ID_bud+len(IDs)-2
        temp_sep_bud_lab = sep_bud_label.copy()
        self.rr = []
        self.cc = []
        self.val = []
        for r, c in zip(rr, cc):
            if self.only_ID_img[r, c] != 0:
                ID = self.nearest_nonzero(sep_bud_label, r, c)
                temp_sep_bud_lab[r,c] = ID
                self.rr.append(r)
                self.cc.append(c)
                self.val.append(ID)
        self.sep_bud_label = temp_sep_bud_lab
        self.update_img(None)

    def mouse_up(self, event):
        try:
            (self.sub_win.canvas).mpl_disconnect(self.cid_brush)
            self.cid_brush_circle = (self.sub_win.canvas).mpl_connect(
                                                        'motion_notify_event',
                                                        self.draw_brush_circle)
        except:
            pass

    def draw_brush_circle(self, event):
        if event.inaxes == self.ax:
            x, y = self.ax_transData_and_coerce(self.ax, event.x, event.y,
                                                        self.label_img.shape)
            try:
                self.brush_circle.set_visible(False)
            except:
                pass
            c = 'r' if self.eraser_on else 'g'
            self.brush_circle = matplotlib.patches.Circle((x, y),
                                    radius=self.brush_size,
                                    fill=False,
                                    color=c, lw=2)
            self.ax.add_patch(self.brush_circle)
            (self.sub_win.canvas).draw_idle()

    def convexity_defects(self, img, eps_percent):
        contours, hierarchy = cv2.findContours(img,2,1)
        cnt = contours[0]
        cnt = cv2.approxPolyDP(cnt,eps_percent*cv2.arcLength(cnt,True),True) # see https://www.programcreek.com/python/example/89457/cv22.convexityDefects
        hull = cv2.convexHull(cnt,returnPoints = False) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        defects = cv2.convexityDefects(cnt,hull) # see https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html
        return cnt, defects


    def undo(self, event):
        self.coords_delete = []
        sep_bud_img = np.copy(self.only_ID_img)
        self.sep_bud_img = sep_bud_img
        self.sep_bud_label = np.copy(self.only_ID_img)
        self.small_obj_mask = np.zeros(self.only_ID_img.shape, bool)
        self.eraser_mask = np.zeros(self.label_img.shape, bool)
        self.overlay_on = False
        rp = regionprops(sep_bud_img)
        self.ax.clear()
        self.ax.imshow(self.sep_bud_img)
        (self.ax).set_xlim(*self.xlims)
        (self.ax).set_ylim(*self.ylims)
        text_label_centroid(rp, self.ax, 18, 'semibold', 'center',
                            'center', None, display_ccStage=False,
                            color='k', clear=True)
        self.ax.axis('off')
        (self.sub_win.canvas).draw_idle()

    def ok(self, event):
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()

    def abort_exec(self):
        self.cancel = True
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()
        abort = messagebox.askyesno('Abort or continue',
                                    'Do you really want to abort execution?')
        if abort:
            exit('Execution aborted by the user')
