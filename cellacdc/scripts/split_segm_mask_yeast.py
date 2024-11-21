import os

from tqdm import tqdm

import numpy as np

import qtpy.compat

from cellacdc import printl, myutils, apps, load, core
from cellacdc._run import _setup_app
from cellacdc.utils.base import NewThreadMultipleExpBaseUtil

DEBUG = False

def ask_select_folder():
    selected_path = qtpy.compat.getexistingdirectory(
        caption='Select experiment folder to analyse', 
        basedir=myutils.getMostRecentPath()
    )
    return selected_path

def get_exp_path_pos_foldernames(selected_path):
    folder_type = myutils.determine_folder_type(selected_path)
    is_pos_folder, is_images_folder, exp_path = folder_type
    if is_pos_folder:
        exp_path = os.path.dirname(selected_path)
        pos_foldernames = myutils.get_pos_foldernames(exp_path)
    elif is_images_folder:
        pos_path = os.path.dirname(selected_path)
        exp_path = os.path.dirname(pos_path)
        pos_foldernames = [os.path.basename(pos_path)]
    else:
        exp_path = selected_path
        pos_foldernames = myutils.get_pos_foldernames(exp_path)
    
    return exp_path, pos_foldernames

def select_segm_masks(exp_path, pos_foldernames):
    infoText = 'Select which segmentation file <b>OF THE CELLS</b>:'
    existingEndNames = load.get_segm_endnames_from_exp_path(
        exp_path, pos_foldernames=pos_foldernames
    )
    win = apps.SelectSegmFileDialog(
        existingEndNames, exp_path, 
        infoText=infoText, 
        fileType='segmentation'
    )
    win.exec_()
    if win.cancel:
        return
    
    cells_segm_endname = win.selectedItemText
    
    infoText = 'Select segmentation files <b>to SPLIT</b>:'
    existingEndNames.discard(cells_segm_endname)
    win = apps.SelectSegmFileDialog(
        existingEndNames, exp_path, 
        infoText=infoText, 
        fileType='segmentation',
        allowMultipleSelection=True
    )
    win.exec_()
    if win.cancel:
        return
    
    list_segm_endnames_to_split = win.selectedItemTexts
    return cells_segm_endname, list_segm_endnames_to_split

def run():
    app, splashScreen = _setup_app(splashscreen=True)  
    splashScreen.close()
    
    selected_path = ask_select_folder()
    if not selected_path:
        exit('Execution cancelled')
    
    myutils.addToRecentPaths(selected_path)
    exp_path, pos_foldernames = get_exp_path_pos_foldernames(selected_path)
    
    selected_segm_endnames = select_segm_masks(exp_path, pos_foldernames)
    if selected_segm_endnames is None:
        exit('Execution cancelled')
    
    cells_segm_endname, list_segm_endnames_to_split = selected_segm_endnames
    acdc_df_endname = cells_segm_endname.replace('segm', 'acdc_output')
    pbar = tqdm(total=len(pos_foldernames), ncols=100)
    for pos in pos_foldernames:
        images_path = os.path.join(exp_path, pos, 'Images')
        cells_segm_data = load.load_segm_file(
            images_path, end_name_segm_file=cells_segm_endname
        )
        acdc_df = load.load_acdc_df_file(
            images_path, end_name_acdc_df_file=acdc_df_endname
        )
        for segm_endname in list_segm_endnames_to_split:
            segm_data_to_split, segm_data_to_split_fp = load.load_segm_file(
                images_path, end_name_segm_file=segm_endname,
                return_path=True
            )
            out = core.split_segm_masks_mother_bud_line(
                cells_segm_data, segm_data_to_split, acdc_df, 
                debug=DEBUG
            )
            split_segm_close, split_segm_away = out
            
            segm_data_to_split_fn = os.path.basename(segm_data_to_split_fp)
            
            split_close_filename = segm_data_to_split_fn.replace(
                segm_endname, f'{segm_endname}_split_close.npz'
            ).replace('.npz.npz', '.npz')
            split_close_filepath = os.path.join(
                images_path, split_close_filename
            )
            
            np.savez_compressed(split_close_filepath, split_segm_close)
            
            
            split_away_filename = segm_data_to_split_fn.replace(
                segm_endname, f'{segm_endname}_split_away.npz'
            ).replace('.npz.npz', '.npz')
            split_away_filepath = os.path.join(
                images_path, split_away_filename
            )
            np.savez_compressed(split_away_filepath, split_segm_away)
        pbar.update()
    
    pbar.close()
    

if __name__ == '__main__':
    run()