import os
import re
import time
import traceback
import difflib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk
from ast import literal_eval
from skimage.color import label2rgb, gray2rgb
from skimage.measure import regionprops
from skimage import img_as_float
from natsort import natsorted

from PyQt5.QtWidgets import (
    QApplication
)

import apps

def file_dialog(toplevel=False, **options):
    #Prompt the user to select the image file
    if toplevel:
        root = tk.Toplevel()
    else:
        root = tk.Tk()
        root.withdraw()
    path = tk.filedialog.askopenfilename(**options)
    root.destroy()
    return path

def multi_files_dialog(toplevel=False, **options):
    #Prompt the user to select the image file
    if toplevel:
        root = tk.Toplevel()
    else:
        root = tk.Tk()
        root.withdraw()
    files = tk.filedialog.askopenfilenames(**options)
    root.destroy()
    return files

def folder_dialog(toplevel=False, **options):
    #Prompt the user to select the image file
    if toplevel:
        root = tk.Toplevel()
    else:
        root = tk.Tk()
        root.withdraw()
    path = tk.filedialog.Directory(**options).show()
    root.destroy()
    return path

class select_pos_to_segm:
    def __init__(self):
        self.is_first_call = True

    def prompt(self, NONsegm_pos_foldernames, segm_pos_foldernames,
               toplevel=False):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()

        self.NONsegm_pos_foldernames = NONsegm_pos_foldernames
        self.segm_pos_foldernames = segm_pos_foldernames

        root.title('Select positions')
        root.geometry('+800+400')
        root.lift()
        root.attributes("-topmost", True)

        tk.Label(root, text='Select positions to segment.\n\n'
                 'Hold down "Ctrl" for multiple selections,\n'
                 'or "Shift" for group selection.',
                 justify=tk.CENTER, font=(None, 11)
                 ).grid(row=0, column=0, columnspan=4, pady=(10, 0), padx=10)

        tk.Label(root, text='Never segmented positions',
                 font=(None, 10)
                 ).grid(row=1, column=0, pady=(10, 0),
                        padx=(20, 0))
        NONsegm_pos_var = tk.StringVar()
        NONsegm_pos_var.set(NONsegm_pos_foldernames)
        NONsegm_pos_listbox = tk.Listbox(root,
                                        listvariable=NONsegm_pos_var,
                                        width=25,
                                        selectmode='extended',
                                        exportselection=0)
        NONsegm_pos_listbox.grid(column=0, row=2, pady=4, padx=(20, 0))
        NONsegm_pos_scrollbar = tk.Scrollbar(root)
        NONsegm_pos_scrollbar.grid(column=1, row=2, pady=4, padx=0,
                                   sticky=tk.NS)
        NONsegm_pos_listbox.config(yscrollcommand = NONsegm_pos_scrollbar.set)
        NONsegm_pos_scrollbar.config(command = NONsegm_pos_listbox.yview)
        NONsegm_pos_listbox.config(justify=tk.CENTER)

        self.NONsegm_pos_listbox = NONsegm_pos_listbox

        tk.Label(root, text='Already segmented positions',
                 font=(None, 10)
                 ).grid(row=1, column=2, pady=(10, 0),
                        padx=(20, 0))
        segm_pos_var = tk.StringVar()
        segm_pos_var.set(segm_pos_foldernames)
        segm_pos_listbox = tk.Listbox(root,
                                        listvariable=segm_pos_var,
                                        width=25,
                                        selectmode='extended',
                                        exportselection=0)
        segm_pos_listbox.grid(column=2, row=2, pady=4, padx=(20, 0))
        segm_pos_scrollbar = tk.Scrollbar(root)
        segm_pos_scrollbar.grid(column=3, row=2, pady=4, padx=0,
                                   sticky=tk.NS)
        segm_pos_listbox.config(yscrollcommand = segm_pos_scrollbar.set)
        segm_pos_scrollbar.config(command = segm_pos_listbox.yview)
        segm_pos_listbox.config(justify=tk.CENTER)

        self.segm_pos_listbox = segm_pos_listbox

        ttk.Button(root,
                  text='Segment selected',
                  command=self._segment_selected,
                  width=20).grid(row=3, column=0, columnspan=4, pady=(10,0))

        ttk.Button(root,
                  text='Segment ALL',
                  command=self._segment_all,
                  width=20).grid(row=4, column=0, columnspan=4, pady=(4,10))

        root.bind('<Escape>', self._clear_selection)
        root.protocol("WM_DELETE_WINDOW", self._abort)
        self.root = root
        root.mainloop()

    def _segment_selected(self):
        self.pos_to_segm = []
        sel_idx = self.NONsegm_pos_listbox.curselection()
        self.pos_to_segm.extend([self.NONsegm_pos_foldernames[idx]
                                 for idx in sel_idx])
        sel_idx = self.segm_pos_listbox.curselection()
        self.pos_to_segm.extend([self.segm_pos_foldernames[idx]
                                 for idx in sel_idx])
        self._close()

    def _segment_all(self):
        self.pos_to_segm = (self.NONsegm_pos_foldernames
                           +self.segm_pos_foldernames)
        self._close()

    def _close(self):
        self.pos_to_segm = natsorted(self.pos_to_segm)
        self.root.quit()
        self.root.destroy()

    def _clear_selection(self, key_info):
        self.NONsegm_pos_listbox.selection_clear(0, tk.END)
        self.segm_pos_listbox.selection_clear(0, tk.END)

    def _abort(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user.')



class scan_run_nums:
    def __init__(self, vNUM):
        self.vNUM = vNUM
        self.is_first_call = True

    def scan(self, pos_path):
        self.spotmax_path = os.path.join(pos_path, 'NucleoData')
        if not os.path.exists(self.spotmax_path):
            self.spotmax_path = os.path.join(pos_path, 'spotMAX_output')
        if os.path.exists(self.spotmax_path):
            filenames = os.listdir(self.spotmax_path)
            run_nums = [re.findall('(\d+)_(\d)_', f)
                                 for f in filenames]
            run_nums = np.unique(
                       np.array(
                            [int(m[0][0]) for m in run_nums if m], int))
            run_nums = [r for r in run_nums for f in filenames
                        if f.startswith(f'{r}')]
            run_nums = set(run_nums)
            return run_nums
        else:
            return []

    def prompt(self, run_nums, toplevel=False):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        root.lift()
        root.attributes("-topmost", True)
        root.title('Multiple runs detected')
        root.geometry("+800+400")
        # tk.Label(root,
        #          text='Select run number to analyse: ',
        #          font=(None, 11)).grid(row=0, column=0, columnspan=2,
        #                                                 pady=(10,0),
        #                                                 padx=10)
        tk.Label(root,
                 text='Select run number to analyse :',
                 font=(None, 11)).grid(row=1, column=0, pady=10, padx=10)

        tk.Button(root, text='Ok', width=20,
                        command=self._close).grid(row=3, column=1,
                                                  pady=(0,10), padx=10)

        show_b = tk.Button(root, text='Print analysis inputs', width=20,
                           command=self._print_analysis_inputs)
        show_b.grid(row=2, column=1, pady=(0,5), padx=10)
        show_b.config(font=(None, 9, 'italic'))

        run_num_Intvar = tk.IntVar()
        run_num_combob = ttk.Combobox(root, width=15, justify='center',
                                      textvariable=run_num_Intvar)
        run_num_combob.option_add('*TCombobox*Listbox.Justify', 'center')
        run_num_combob['values'] = list(run_nums)
        run_num_combob.grid(column=1, row=1, padx=10, pady=10)
        run_num_combob.current(0)

        root.protocol("WM_DELETE_WINDOW", self._abort)
        self.run_num_Intvar = run_num_Intvar
        self.root = root
        root.mainloop()
        return run_num_Intvar.get()

    def _print_analysis_inputs(self):
        run_num = self.run_num_Intvar.get()
        analysis_inputs_path = os.path.join(
                self.spotmax_path,
                f'{run_num}_{self.vNUM}_analysis_inputs.csv'
        )
        df_inputs = pd.read_csv(analysis_inputs_path)
        df_inputs['Description'] = df_inputs['Description'].str.replace(
                                                                 '\u03bc', 'u')
        df_inputs.set_index('Description', inplace=True)
        print('================================')
        print(f'Analysis inputs for run number {run_num}:')
        print('')
        print(df_inputs)
        print('================================')

    def _close(self):
        self.is_first_call = False
        self.root.quit()
        self.root.destroy()

    def _abort(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

class fourbuttonsmessagebox:
    '''Geometry: "WidthxHeight+Left+Top" '''
    def __init__(self):
        self.prompt = True

    def run(self, title, message, button_1_text,
                 button_2_text, button_3_text, button_4_text,
                 path, geometry="+800+400",auto_close=False, toplevel=False):
        self.do_save = False
        self.replace = False
        self.path = path
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        self.root = root
        root.lift()
        root.attributes("-topmost", True)
        root.title(title)
        root.geometry(geometry)
        tk.Label(root,
                 text=message,
                 font=(None, 11)).grid(row=0, column=0, columnspan=2,
                 pady=4, padx=4)

        do_save_b = tk.Button(root,
                      text=button_1_text,
                      command=self.do_save_cb,
                      width=10).grid(row=4,
                                      column=0,
                                      pady=4, padx=6)

        close = tk.Button(root,
                  text=button_2_text,
                  command=self.close,
                  width=15).grid(row=4,
                                 column=1,
                                 pady=4, padx=6)
        repl = tk.Button(root,
                  text=button_3_text,
                  command=self.replace_cb,
                  width=10,).grid(row=5,
                                  column=0,
                                  pady=4, padx=6)

        expl = tk.Button(root,
                  text=button_4_text,
                  command=self.open_path_explorer,
                  width=15)

        expl.grid(row=5, column=1, pady=4, padx=6)
        expl.config(font=(None, 9, 'italic'))

        self.time_elapsed_sv = tk.StringVar()
        time_elapsed_label = tk.Label(root,
                         textvariable=self.time_elapsed_sv,
                         font=(None, 10)).grid(row=6, column=0,
                                               columnspan=3, padx=4, pady=4)

        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        root.bind('<Enter>', self.stop_timer)
        self.timer_t_final = time.time() + 10
        self.auto_close = True
        if auto_close:
            self.tk_timer()
        self.root.mainloop()

    def stop_timer(self, event):
        self.auto_close = False

    def tk_timer(self):
        if self.auto_close:
            seconds_elapsed = self.timer_t_final - time.time()
            seconds_elapsed = int(round(seconds_elapsed))
            if seconds_elapsed <= 0:
                print('Time elpased. Replacing files')
                self.replace_cb()
            self.time_elapsed_sv.set('Window will close automatically in: {} s'
                                                       .format(seconds_elapsed))
            self.root.after(1000, self.tk_timer)
        else:
            self.time_elapsed_sv.set('')

    def do_save_cb(self):
        self.do_save = True
        self.root.quit()
        self.root.destroy()

    def replace_cb(self):
        self.replace=True
        self.root.quit()
        self.root.destroy()

    def open_path_explorer(self):
        subprocess.Popen('explorer "{}"'.format(os.path.normpath(self.path)))

    def close(self):
        self.root.quit()
        self.root.destroy()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

class num_frames_toQuant:
    def __init__(self):
        self.is_first_call = True

    def prompt(self, tot_frames, last_segm_i=None, last_tracked_i=None,
               toplevel=False):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        self.root = root
        self.tot_frames = tot_frames
        self.root.title('Number of frames to segment')
        root.geometry('+800+400')
        root.lift()
        root.attributes("-topmost", True)
        # root.focus_force()
        tk.Label(root,
                 text="How many frames do you want to analyse?",
                 font=(None, 12)).grid(row=0, column=0, columnspan=3)
        if last_segm_i is not None:
            txt = (f'(there is a total of {tot_frames} frames,\n'
                   f'last segmented frame is index {last_segm_i})')
        else:
            txt = f'(there is a total of {tot_frames} frames)'
        if last_tracked_i is not None:
            txt = f'{txt[:-1]}\nlast tracked frame is index {last_tracked_i})'
        tk.Label(root,
                 text=txt,
                 font=(None, 10)).grid(row=1, column=0, columnspan=3)
        tk.Label(root,
                 text="Start frame",
                 font=(None, 10, 'bold')).grid(row=2, column=0, sticky=tk.E,
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
        start_frame.grid(row=2, column=1, pady=8, sticky=tk.W)
        sv_num = tk.StringVar()
        num_frames = tk.Entry(root, width=10, justify='center',font='None 12',
                                textvariable=sv_num)
        self.num_frames = num_frames
        num_frames.insert(0, '{}'.format(tot_frames))
        sv_num.trace_add("write", self.check_max)
        num_frames.grid(row=3, column=1, pady=8, sticky=tk.W)
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
        start_frame.selection_range(0, tk.END)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        # root.after(1000, self.set_foreground_window)
        root.mainloop()

    def set_all(self, name=None, index=None, mode=None):
        start_frame_str = self.start_frame.get()
        if start_frame_str:
            startf = int(start_frame_str)
            rightRange = self.tot_frames - startf
            self.num_frames.delete(0, tk.END)
            self.num_frames.insert(0, '{}'.format(rightRange))

    def check_max(self, name=None, index=None, mode=None):
        num_frames_str = self.num_frames.get()
        start_frame_str = self.start_frame.get()
        if num_frames_str and start_frame_str:
            startf = int(start_frame_str)
            if startf + int(num_frames_str) > self.tot_frames:
                rightRange = self.tot_frames - startf
                self.num_frames.delete(0, tk.END)
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

    def set_foreground_window(self):
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.focus_force()

class single_combobox_widget:
    def __init__(self):
        self.is_first_call = True


    def prompt(self, values, title='Select value', message=None,
               toplevel=False):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        root.lift()
        root.attributes("-topmost", True)
        root.title(title)
        root.geometry("+800+400")
        row = 0
        if message is not None:
            tk.Label(root,
                     text=message,
                     font=(None, 11)).grid(row=row, column=0,
                                           pady=(10,0), padx=10)
            row += 1

        # tk.Label(root,
        #          text='Select value:',
        #          font=(None, 11)).grid(row=row, column=0, pady=(10,0),
        #                                                 padx=10)
        w = max([len(v) for v in values])+10
        _var = tk.StringVar()
        _combob = ttk.Combobox(root, width=w, justify='center',
                                      textvariable=_var)
        _combob.option_add('*TCombobox*Listbox.Justify', 'center')
        _combob['values'] = values
        _combob.grid(column=0, row=row, padx=10, pady=(10,0))
        _combob.current(0)

        row += 1
        tk.Button(root, text='Ok', width=20,
                        command=self._close).grid(row=row, column=0,
                                                  pady=10, padx=10)



        root.protocol("WM_DELETE_WINDOW", self._abort)
        self._var = _var
        self.root = root
        root.mainloop()

    def _close(self):
        self.selected_val = self._var.get()
        self.is_first_call = False
        self.root.quit()
        self.root.destroy()

    def _abort(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')

class select_channel_name:
    def __init__(self, which_channel=None, allow_abort=True):
        self.is_first_call = True
        self.which_channel = which_channel
        self.last_sel_channel = self._load_last_selection()
        self.was_aborted = False
        self.allow_abort = allow_abort

    def get_available_channels(self, filenames):
        channel_names = []
        basename = filenames[0]
        for file in filenames:
            # Determine the basename based on intersection of all .tif
            _, ext = os.path.splitext(file)
            if ext == '.tif':
                sm = difflib.SequenceMatcher(None, file, basename)
                i, j, k = sm.find_longest_match(0, len(file),
                                                0, len(basename))
                basename = file[i:i+k]
        warn = [False]
        for file in filenames:
            filename, ext = os.path.splitext(file)
            if ext == '.tif':
                channel_name = filename.split(basename)[-1]
                channel_names.append(channel_name)
                if channel_name == filename:
                    # Warn the an intersection could not be found
                    warn.append(True)
        warn = any(warn)
        if self.which_channel is not None:
            # Search for "phase" and put that channel first on the list
            if self.which_channel == 'segm':
                is_phase_contr_li = [c.lower().find('phase')!=-1
                                     for c in channel_names]
                if any(is_phase_contr_li):
                    idx = is_phase_contr_li.index(True)
                    channel_names[0], channel_names[idx] = (
                                      channel_names[idx], channel_names[0])
        return channel_names, warn

    def _load_last_selection(self):
        last_sel_channel = None
        ch = self.which_channel
        if self.which_channel is not None:
            _path = os.path.dirname(os.path.realpath(__file__))
            temp_path = os.path.join(_path, 'temp')
            txt_path = os.path.join(temp_path, f'{ch}_last_sel.txt')
            if os.path.exists(txt_path):
                with open(txt_path) as txt:
                    last_sel_channel = txt.read()
        return last_sel_channel

    def _saved_last_selection(self, selection):
        ch = self.which_channel
        if self.which_channel is not None:
            _path = os.path.dirname(os.path.realpath(__file__))
            temp_path = os.path.join(_path, 'temp')
            if not os.path.exists(temp_path):
                os.mkdir(temp_path)
            txt_path = os.path.join(temp_path, f'{ch}_last_sel.txt')
            with open(txt_path, 'w') as txt:
                txt.write(selection)

    def QtPrompt(self, channel_names, informativeText='', parent=None):
        if parent is None:
            app = QApplication([])
            win = apps.QDialogCombobox(
                                  'Select channel name',
                                  channel_names,
                                  informativeText,
                                  CbLabel='Select channel name:  ',
                                  parent=None)
            win.show()
            app.exec_()
            if self.allow_abort and win.cancel:
                self._abort()
            elif win.cancel:
                self.was_aborted = True
        else:
            win = apps.QDialogCombobox(
                                  'Select channel name',
                                  channel_names,
                                  informativeText,
                                  CbLabel='Select channel name:  ',
                                  parent=None)
            win.exec_()
            if win.cancel:
                self.was_aborted = True

        self.channel_name = win.selectedItemText
        self._saved_last_selection(self.channel_name)
        self.is_first_call = False


    def prompt(self, channel_names, message=None, toplevel=False):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        root.lift()
        root.attributes("-topmost", True)
        root.title('Select channel name')
        root.geometry("+800+400")
        row = 0
        if message is not None:
            tk.Label(root,
                     text=message,
                     font=(None, 11)).grid(row=row, column=0,
                                           columnspan= 2, pady=(10,0),
                                                          padx=10)
            row += 1

        tk.Label(root,
                 text='Select channel name to analyse:'
                 ).grid(row=row, column=0, pady=(10,0), padx=10)

        ch_name_var = tk.StringVar()
        w = max([len(s) for s in channel_names])+4
        ch_name_combob = ttk.Combobox(root, width=w, justify='center',
                                      textvariable=ch_name_var)
        ch_name_combob.option_add('*TCombobox*Listbox.Justify', 'center')
        ch_name_combob['values'] = channel_names
        ch_name_combob.grid(column=1, row=row, padx=10, pady=(10,0))
        if self.last_sel_channel is not None:
            if self.last_sel_channel in channel_names:
                ch_name_combob.current(channel_names.index(self.last_sel_channel))
            else:
                ch_name_combob.current(0)
        else:
            ch_name_combob.current(0)

        ch_name_var.trace_add("write", self._test)

        row += 1
        tk.Button(root, text='Ok', width=20,
                        command=self._tk_close).grid(row=row, column=0,
                                                  columnspan=2,
                                                  pady=10, padx=10)



        root.protocol("WM_DELETE_WINDOW", self._tk_abort)
        self.ch_name_var = ch_name_var
        self.root = root

        root.mainloop()

    def _tk_close(self):
        self.was_aborted = False
        self.channel_name = self.ch_name_var.get()
        self.root.quit()
        self.root.destroy()


    def _tk_abort(self):
        self.was_aborted = True
        self.root.quit()
        self.root.destroy()
        if self.allow_abort:
            exit('Execution aborted by the user')

    def _test(self, name=None, index=None, mode=None):
        pass

    def _abort(self):
        self.was_aborted = True
        if self.allow_abort:
            exit('Execution aborted by the user')

def check_img_shape_vs_metadata(img_shape, num_frames, SizeT, SizeZ):
    msg = ''
    if num_frames > 1 and len(img_shape) > 3:
        data_T, data_Z = img_shape[:2]
        ndim_msg = 'Data is expected to be 4D with TZYX order'
    elif num_frames > 1 and len(img_shape) > 2:
        data_T, data_Z = img_shape[0], 1
        expected_data_ndim = 3
        ndim_msg = 'Data is expected to be 3D with ZYX order'
    else:
        data_T, data_Z = 1, img_shape[0]
        expected_data_ndim = 3
        ndim_msg = 'Data is expected to be 3D with ZYX order'
    if data_T != SizeT:
        msg = (f'{ndim_msg}.\nData shape is {img_shape} '
        f'(i.e. {data_T} frames), but the metadata of the '
        f'.tif file says that there should be {SizeT} frames.\n\n'
        f'Process cannot continue.')
        root = tk.Tk()
        root.withdraw()
        tk.messagebox.showerror('Shape mismatch!', msg, master=root)
        root.quit()
        root.destroy()
    if data_Z != SizeZ:
        msg = (f'{ndim_msg}.\nData shape is {img_shape} '
        f'(i.e. {data_Z} z-slices), but the metadata of the '
        f'.tif file says that there should be {SizeZ} z-slices.\n\n'
        f'Process cannot continue.')
        root = tk.Tk()
        root.withdraw()
        tk.messagebox.showerror('Shape mismatch!', msg, master=root)
        root.quit()
        root.destroy()
    return msg.replace('\n', ' ')

class single_entry_messagebox:
    def __init__(self, title='Entry', entry_label='Entry 1', input_txt='',
                       toplevel=True, allow_abort=True):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        self.was_aborted = False
        self.allow_abort = allow_abort
        root.lift()
        root.title(title)
        root.attributes("-topmost", True)
        root.geometry("+800+400")
        self._root = root
        tk.Label(root, text=entry_label, font=(None, 10)).grid(row=0, padx=8)
        w = len(input_txt)+10
        w = w if w>40 else 40
        e = tk.Entry(root, justify='center', width=w)
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
        if self.allow_abort:
            exit('Execution aborted by the user')
        else:
            self.was_aborted = True

    def _quit(self, event=None):
        self.entry_txt = self.e.get()
        self._root.quit()
        self._root.destroy()

class dual_entry_messagebox:
    def __init__(self, title='Entry',
                       entry_label='Entry 1', input_txt='',
                       entry_label2='Entry 2', input_txt2='',
                       toplevel=True):
        if toplevel:
            root = tk.Toplevel()
        else:
            root = tk.Tk()
        root.lift()
        root.title(title)
        root.attributes("-topmost", True)
        root.geometry("+800+400")
        self._root = root
        tk.Label(root, text=entry_label, font=(None, 10)).grid(row=0, padx=8)
        w = len(input_txt)+10
        w = w if w>40 else 40
        e = tk.Entry(root, justify='center', width=w)
        e.grid(row=1, padx=16, pady=4)
        e.focus_force()
        e.insert(0, input_txt)

        tk.Label(root, text=entry_label2, font=(None, 10)).grid(row=2, padx=8)
        entry2 = tk.Entry(root, justify='center', width=w)
        entry2.grid(row=3, padx=16, pady=4)
        entry2.focus_force()
        entry2.insert(0, input_txt2)

        tk.Button(root, command=self._quit, text='Ok!', width=10).grid(row=4,
                                                                      pady=4)
        root.bind('<Return>', self._quit)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.e = e
        self.entry2 = entry2
        root.mainloop()

    def on_closing(self):
        self._root.quit()
        self._root.destroy()
        exit('Execution aborted by the user')

    def _quit(self, event=None):
        self.entries_txt = (self.e.get(), self.entry2.get())
        self._root.quit()
        self._root.destroy()

def askyesno(title='tk', message='Yes or no?', toplevel=False):
    if toplevel:
        root = tk.Toplevel()
    else:
        root = tk.Tk()
        root.withdraw()
    yes = tk.messagebox.askyesno(title, message, master=root)
    if not toplevel:
        root.quit()
        root.destroy()
    return yes

class num_pos_toSegm_tk:
    def __init__(self, tot_frames, toplevel=False):
        if toplevel:
            root = tk.Tk()
        else:
            root = tk.Toplevel()
        self.root = root
        self.tot_frames = tot_frames
        root.geometry('+800+400')
        root.attributes("-topmost", True)
        tk.Label(root,
                 text="How many positions do you want to segment?",
                 font=(None, 12)).grid(row=0, column=0, columnspan=3)
        tk.Label(root,
                 text="(There are a total of {} positions).".format(tot_frames),
                 font=(None, 10)).grid(row=1, column=0, columnspan=3)
        tk.Label(root,
                 text="Start position",
                 font=(None, 10, 'bold')).grid(row=2, column=0, sticky=tk.E,
                                               padx=4)
        tk.Label(root,
                 text="# of positions to analyze",
                 font=(None, 10, 'bold')).grid(row=3, column=0, padx=4)
        sv_sf = tk.StringVar()
        start_frame = tk.Entry(root, width=10, justify='center',font='None 12',
                               textvariable=sv_sf)
        start_frame.insert(0, '{}'.format(1))
        sv_sf.trace_add("write", self.set_all)
        self.start_frame = start_frame
        start_frame.grid(row=2, column=1, pady=8, sticky=tk.W)
        sv_num = tk.StringVar()
        num_frames = tk.Entry(root, width=10, justify='center',font='None 12',
                                textvariable=sv_num)
        self.num_frames = num_frames
        num_frames.insert(0, '{}'.format(tot_frames))
        sv_num.trace_add("write", self.check_max)
        num_frames.grid(row=3, column=1, pady=8, sticky=tk.W)
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
        start_frame.selection_range(0, tk.END)
        root.protocol("WM_DELETE_WINDOW", self.on_closing)
        root.mainloop()

    def set_all(self, name=None, index=None, mode=None):
        start_frame_str = self.start_frame.get()
        if start_frame_str:
            startf = int(start_frame_str)
            if startf > self.tot_frames:
                self.start_frame.delete(0, tk.END)
                self.start_frame.insert(0, '{}'.format(self.tot_frames))
                startf = self.tot_frames
            rightRange = self.tot_frames - startf + 1
            self.num_frames.delete(0, tk.END)
            self.num_frames.insert(0, '{}'.format(rightRange))

    def check_max(self, name=None, index=None, mode=None):
        num_frames_str = self.num_frames.get()
        start_frame_str = self.start_frame.get()
        if num_frames_str and start_frame_str:
            startf = int(start_frame_str)
            if startf + int(num_frames_str) > self.tot_frames:
                rightRange = self.tot_frames - startf + 1
                self.num_frames.delete(0, tk.END)
                self.num_frames.insert(0, '{}'.format(rightRange))

    def ok(self, event=None):
        num_frames_str = self.num_frames.get()
        start_frame_str = self.start_frame.get()
        if num_frames_str and start_frame_str:
            startf = int(self.start_frame.get())
            num = int(self.num_frames.get())
            stopf = startf + num
            self.frange = (startf-1, stopf-1)
            self.root.quit()
            self.root.destroy()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')


if __name__ == '__main__':
    num_frames_prompt = num_frames_toQuant()
    num_frames_prompt.prompt(100, last_segm_i=100, last_tracked_i=85)
