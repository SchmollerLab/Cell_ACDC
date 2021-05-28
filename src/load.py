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
import prompts, apps


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

        # Ok button
        ok_b = ttk.Button(root, text='Ok!', comman=self._close)
        ok_b.grid(column=0, row=1, pady=10, sticky=tk.E)
        self.ok_b = ok_b

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
            show_expl_button.grid(column=1, row=1, pady=10)

        root.protocol("WM_DELETE_WINDOW", self.on_closing)
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
        pos_foldernames = [pos for pos in pos_foldernames
                               if re.match('Position_(\d+)', pos)]
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
        self.root.quit()
        self.root.destroy()

    def on_closing(self):
        self.root.quit()
        self.root.destroy()
        exit('Execution aborted by the user')


def get_main_paths(selected_path, vNUM):
    selector = select_exp_folder()
    is_pos_path = os.path.basename(selected_path).find('Position_') != -1
    is_TIFFs_path = os.path.basename(selected_path).find('TIFFs') != -1
    multi_run_msg = ('Multiple runs detected!\n\n'
                     'Select which run number you want to analyse.')
    if not is_pos_path and not is_TIFFs_path:
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
    run_num = None
    return (main_paths, prompts_pos_to_analyse, run_num, is_pos_path,
            is_TIFFs_path)
