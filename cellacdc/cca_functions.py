import numpy as np
import traceback
from skimage.measure import label, regionprops, regionprops_table
import skimage.transform
import pandas as pd
from tifffile import imread
import os
import glob
from math import pow, floor
from tqdm import tqdm
import sys
import difflib
from scipy.stats import binned_statistic
import warnings
from typing import Iterable

from . import GUI_INSTALLED
if GUI_INSTALLED:
    from qtpy.QtWidgets import QFileDialog
    from . import widgets
    from . import qrc_resources
    from . import _run

from . import load, cca_df_colnames
from . import myutils, prompts, html_utils, printl

def configuration_dialog():
    app, _ = _run._setup_app(splashscreen=False)
    
    continue_selection = True
    data_dirs = []
    positions = []
    while continue_selection:
        MostRecentPath = myutils.getMostRecentPath()
        data_dir = QFileDialog.getExistingDirectory(
            None, 'Select experiment folder containing Position_n folders ',
            MostRecentPath
        )
        if not data_dir:
            continue_selection = False
            break

        myutils.addToRecentPaths(data_dir)
        foldername = os.path.basename(data_dir)
        if foldername == 'Images':
            pos_path = os.path.dirname(data_dir)
            data_dir = os.path.dirname(pos_path)
            pos = [os.path.basename(pos_path)]
        elif foldername.find('Position_') != -1:
            pos_path = data_dir
            data_dir = os.path.dirname(data_dir)
            pos = [os.path.basename(pos_path)]
        else:
            available_pos = myutils.get_pos_foldernames(data_dir)
            if not available_pos:
                print('******************************')
                print('Selected folder does not contain any Position folders.')
                print(f'Selected folder: "{data_dir}"')
                print('******************************')
                raise FileNotFoundError

            win = widgets.QDialogListbox(
                'Position Selection',
                'Select which position(s) you want to analyse',
                available_pos
            )
            win.show()
            win.exec_()
            if win.cancel:
                print('******************************')
                print('Execution aborted by the user')
                print('******************************')
                raise InterruptedError
            pos = win.selectedItemsText

        data_dirs.append(data_dir)
        positions.append(pos)
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            'Do you wish to select Positions from other experiments?'
        )
        yes, no = msg.question(
            None, 'Continue selection?', txt, buttonsTexts=(' Yes ', ' No ')
        )
        continue_selection = msg.clickedButton == yes
    if len(data_dirs) == 0:
        print('******************************')
        print("No positions selected!")
        print('******************************')
        raise IndexError("No positions selected!")
    return data_dirs, positions, app

def find_available_channels(filenames, first_pos_dir):
    ch_name_selector = prompts.select_channel_name()
    ch_names, warn = ch_name_selector.get_available_channels(
        filenames, first_pos_dir
    )
    return ch_names, ch_name_selector.basename

def get_segm_endname(images_path, basename):
    segm_files = load.get_segm_files(images_path)
    segm_endnames = load.get_endnames(
        basename, segm_files
    )
    if not segm_endnames:
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(f"""
            The following position does not contain valid segmentation files.<br><br>
            <code>{images_path}</code><br>
        """)
        msg.critical(None, 'Segmentation file(s) not found', txt)
        raise FileNotFoundError(f'Segmentation files not found in "{images_path}"')

    if len(segm_endnames) == 1:
        return segm_endnames[0]
    
    selectSegmWin = widgets.QDialogListbox(
        'Select segmentation file',
        'Select segmentation file to use as ROI:\n',
        segm_endnames, multiSelection=False, parent=None
    )
    selectSegmWin.exec_()
    if selectSegmWin.cancel:
        raise FileNotFoundError(f'Segmentation file selection aborted by the user.')
    
    return selectSegmWin.selectedItemsText[0]
    
    
def calculate_downstream_data(
        file_names,
        image_folders,
        positions,
        channels,
        segm_endname,
        force_recalculation=False, 
        calculate_fluo_metrics=True, 
        save_features_to_acdc_df=False,
    ):
    no_of_channels = len(channels)
    overall_df = pd.DataFrame()
    for file_idx, file in enumerate(file_names):
        for pos_idx, pos_dir in enumerate(image_folders[file_idx]):
            channel_data = ('placeholder')*no_of_channels
            print(f'Load files for {file}, {positions[file_idx][pos_idx]}...')
            acdc_df_path = None
            try:
                *channel_data, seg_mask, cc_data, metadata, cc_props, acdc_df_path = (
                    _load_files(
                        pos_dir, channels, segm_endname, 
                        load_channels_data=calculate_fluo_metrics
                    )
                )
            except TypeError:
                print(f'File {file}, position {positions[file_idx][pos_idx]} skipped due to missing segmentation mask/CC annotations.')
                continue
            print(f'Number of cells in position: {len(cc_data.Cell_ID.unique())}')
            print(f'Number of annotated frames in position: {cc_data.frame_i.max()+1}')

            cc_data = _rename_columns(cc_data)
            is_timelapse_data, is_zstack_data = False, False
            if int(metadata.loc['SizeT'])>1:
                is_timelapse_data=True
            if int(metadata.loc['SizeZ'])>1:
                is_zstack_data=True
            if cc_props is not None and not force_recalculation:
                print('Cell Cycle property data already existing, loaded from disk...')
                overall_df = pd.concat([overall_df, cc_props], ignore_index=True).reset_index(drop=True)
            else:
                print(f'Calculate regionprops on each frame based on Segmentation...')
                rp_df = _calculate_rp_df(seg_mask, is_timelapse_data, is_zstack_data, metadata, max_frame=cc_data.frame_i.max()+1)
                print(f'Calculate signal metrics for every channel and cell...')
                flu_signal_df = _calculate_flu_signal(
                    seg_mask,
                    channel_data,
                    channels,
                    cc_data,
                    is_timelapse_data,
                    is_zstack_data
                )
                temp_df = cc_data.merge(
                    rp_df, on=['frame_i', 'Cell_ID'], how='left',
                    suffixes=('_gui', '')
                )
                temp_df = temp_df.merge(
                    flu_signal_df, on=['frame_i', 'Cell_ID'], how='left',
                    suffixes=('_gui', '')
                )
                # calculate amount of corrected signal by multiplying mean with area
                if is_timelapse_data:
                    for channel in channels:
                        temp_df[f'{channel}_corrected_amount'] = (
                            temp_df[f'{channel}_corrected_mean']
                            * temp_df['area']
                        )
                        try:
                            temp_df[f'{channel}_corrected_concentration'] = (
                                temp_df[f'{channel}_corrected_amount']
                                / temp_df['cell_vol_fl']
                            )
                        except KeyError:
                            print(f'Volume is missing in acdc output, NaNs inserted in concentration columns of channel {channel}')
                            temp_df[f'{channel}_corrected_concentration'] = None
                temp_df['max_frame_pos'] = cc_data.frame_i.max()
                temp_df['file'] = file
                temp_df['selection_subset'] = file_idx
                temp_df['position'] = positions[file_idx][pos_idx]
                temp_df['directory'] = pos_dir
                print('Saving calculated data for next time...')
                files_in_curr_dir = myutils.listdir(pos_dir)
                common_prefix = _determine_common_prefix(files_in_curr_dir)
                save_path = os.path.join(pos_dir, f'{common_prefix}cca_properties_downstream.csv')
                temp_df.to_csv(save_path, index=False)
                overall_df = pd.concat([overall_df, temp_df], ignore_index=True).reset_index(drop=True)

            # if save_features_to_acdc_df:
            #     acdc_df = cc_data.set_index(['frame_i', 'Cell_ID'])
            #     missing_cols = overall_df.columns.difference(acdc_df.columns)
            #     acdc_df_path
            #     import pdb; pdb.set_trace()

    print('Done!')
    return overall_df, is_timelapse_data, is_zstack_data


def calculate_relatives_data(overall_df, channels):
    # Join on Cell_ID vs. relative_ID to later calculate columns like "daughter growth" or "mother-bud-signal-combined"
    overall_df_rel = overall_df.copy()
    overall_df = overall_df.merge(
        overall_df_rel,
        how='left',
        left_on=['frame_i', 'relative_ID', 'max_frame_pos', 'file', 'selection_subset', 'position', 'directory'],
        right_on=['frame_i', 'Cell_ID', 'max_frame_pos', 'file', 'selection_subset', 'position', 'directory'],
        suffixes = ('', '_rel')
    )
    # for every channel, calculate amount from mother and bud cells combined
    for ch in channels:
        try:
            overall_df[f'{ch}_combined_amount_mother_bud'] = overall_df.apply(
                lambda x: (
                    x.loc[f'{ch}_corrected_amount']
                    + x.loc[f'{ch}_corrected_amount_rel']
                    if x.loc['cell_cycle_stage']=='S'
                    else x.loc[f'{ch}_corrected_amount']
                    ),
                axis=1
            )
            overall_df[f'{ch}_combined_raw_sum_mother_bud'] = overall_df.apply(
                lambda x: (
                    x.loc[f'{ch}_raw_sum']
                    + x.loc[f'{ch}_raw_sum_rel']
                    if x.loc['cell_cycle_stage']=='S'
                    else x.loc[f'{ch}_raw_sum']
                    ),
                axis=1
            )
        except KeyError:
            continue
    overall_df['combined_mother_bud_volume'] = overall_df.apply(
        lambda x: x.loc['cell_vol_fl']+x.loc['cell_vol_fl_rel'] if\
        x.loc['cell_cycle_stage']=='S' else\
        x.loc['cell_vol_fl'],
        axis=1
    )
    return overall_df


def calculate_per_phase_quantities(overall_df, group_cols, channels):
    # group by group columns, aggregate some other columns
    phase_grouped = overall_df.sort_values(
        'frame_i'
    ).groupby(group_cols).agg(
        # perform some calculations relating to the whole phase:
        phase_area_growth=('cell_area_um2', lambda x: x.iloc[-1]-x.iloc[0]),
        phase_volume_growth=('cell_vol_fl', lambda x: x.iloc[-1]-x.iloc[0]),
        phase_area_at_beginning=('cell_area_um2', 'first'),
        phase_volume_at_beginning=('cell_vol_fl', 'first'),
        phase_volume_at_end=('cell_vol_fl', 'last'),
        phase_daughter_area_growth=('cell_area_um2_rel', lambda x: x.iloc[-1]-x.iloc[0]),
        phase_daughter_volume_growth=('cell_vol_fl_rel', lambda x: x.iloc[-1]-x.iloc[0]),
        phase_length=('frame_i', lambda x: max(x)-min(x)),
        phase_begin = ('frame_i', 'min'),
        phase_end = ('frame_i', 'max'),
        phase_combined_volume_at_end = ('combined_mother_bud_volume','last')
    ).reset_index()
    # calculate some quantities in a for loop for all available channels and merge results.
    phase_grouped_flu = pd.DataFrame(columns=group_cols)
    for ch in channels:
        if f'{ch}_corrected_mean' in overall_df.columns:
            flu_temp = overall_df.sort_values(
                'frame_i'
            ).groupby(group_cols).agg({
                # perform some calculations on flu data:
                f'{ch}_corrected_amount': 'first',
                f'{ch}_corrected_mean': 'first',
                f'{ch}_corrected_concentration': ['first','last'],
                f'{ch}_combined_amount_mother_bud': ['first','last']
            }).reset_index()
            # collapse multiindex into column name with aggregation as suffix
            flu_temp.columns = ['_'.join(col) if col[1]!='' else col[0] for col in flu_temp.columns.values]
            # rename columns into meaningful names
            flu_temp = flu_temp.rename({
                f'{ch}_corrected_amount_first': f'phase_{ch}_amount_at_beginning',
                f'{ch}_corrected_mean_first': f'phase_{ch}_mean_at_beginning',
                f'{ch}_corrected_concentration_first': f'phase_{ch}_concentration_at_beginning',
                f'{ch}_corrected_concentration_last': f'phase_{ch}_concentration_at_end',
                f'{ch}_combined_amount_mother_bud_first': f'phase_{ch}_combined_amount_at_beginning',
                f'{ch}_combined_amount_mother_bud_last': f'phase_{ch}_combined_amount_at_end',
            }, axis=1)
            phase_grouped_flu = phase_grouped_flu.merge(flu_temp, how='right', on=group_cols, suffixes=('',''))

    # detect complete cell cycle phases and complete cell cycles
    temp = np.logical_and(
        phase_grouped.phase_begin > 0,
        phase_grouped.phase_end < phase_grouped.max_frame_pos
    )
    # this or is for disappearing cells
    if 'max_t' in overall_df.columns:
        complete_phase_indices = np.logical_and(
            temp,
            phase_grouped.phase_end < phase_grouped.max_t
        )
    else:
        complete_phase_indices = temp
    phase_grouped['complete_phase'] = complete_phase_indices.astype(int)
    no_of_compl_phases_per_cycle = phase_grouped.groupby(
        ['Cell_ID', 'generation_num', 'position', 'file']
    )['complete_phase'].transform('sum')
    complete_cycle_indices = no_of_compl_phases_per_cycle == 2
    phase_grouped['complete_cycle'] = complete_cycle_indices.astype(int)
    # join phase-grouped data with
    phase_grouped = phase_grouped.merge(phase_grouped_flu, how='left', on=group_cols, suffixes=('',''))
    return phase_grouped


def _determine_common_prefix(filenames):
    basename = filenames[0]
    for file in filenames:
        # Determine the basename based on intersection of all .tif
        _, ext = os.path.splitext(file)
        sm = difflib.SequenceMatcher(None, file, basename)
        i, j, k = sm.find_longest_match(0, len(file),
                                        0, len(basename))
        basename = file[i:i+k]
    return basename


def _auto_rescale_intensity(img, perc=0.01, clip_min=False):
    """
    function to clip outliers to the given percentiles and scale afterwards.

    use skimage.exposure.rescale_intensity for clipping all images in the
    given percentiles to the percentile border.

    Parameters
    ----------
    img: np.array
        image to eliminate outliers from
    perc: float
        percentage of pixels which are considered as outliers.
        Example: given 0.01, the lowest and highest 1% of pixels are considered
        as outliers and clipped to the 1- and 99-percentile respectively.

    Returns
    -------
    scaled_img: np.array
        image, where outlier pixels are set to 1- and 99-percentiles and which is
        scaled to [0,1] afterwards
    """
    if perc > 0:
        vmin, vmax = np.percentile(img, q=(perc, 100-perc))
        clip_min_indices = img < vmin
        clip_max_indices = img > vmax
        if clip_min:
            img[clip_min_indices] = vmin
        img[clip_max_indices] = vmax
    else:
        vmin, vmax = img.min(), img.max()
    scaled_img = (img-vmin)/(vmax-vmin)
    return scaled_img

def load_acdc_output_only(
        file_names,
        image_folders,
        positions,
        segm_endnames
    ):
    """
    Function to load only the acdc output.
    Use when fluorescent file is too big to load into RAM.
    #TODO: move to cca_functions
    """
    
    overall_df = pd.DataFrame()
    for file_idx, file in enumerate(file_names):
        acdc_output_endname = segm_endnames[file_idx].replace('segm', 'acdc_output')
        for pos_idx, pos_dir in enumerate(image_folders[file_idx]):
            try:
                cc_stage_path = glob.glob(
                    os.path.join(f'{pos_dir}', f'*{acdc_output_endname}.csv')
                )[0]
            except IndexError:
                cc_stage_path = glob.glob(os.path.join(f'{pos_dir}', '*cc_stage.csv'))[0]
            temp_df = pd.read_csv(cc_stage_path)
            temp_df['max_frame_pos'] = temp_df.frame_i.max()
            temp_df['file'] = file
            temp_df['selection_subset'] = file_idx
            temp_df['position'] = positions[file_idx][pos_idx]
            temp_df['directory'] = pos_dir
            overall_df = pd.concat([overall_df, temp_df])
    return overall_df

def _load_channels_data(file_dir, channel_names, no_of_aligned_files):
    channel_files = []
    if no_of_aligned_files > 0:
        for channel in channel_names:
            try:
                ch_aligned_path = glob.glob(os.path.join(f'{file_dir}', f'*{channel}_aligned.npz'))[0]
                channel_files.append(np.load(ch_aligned_path)['arr_0'])
            except IndexError:
                try:
                    ch_aligned_path = glob.glob(os.path.join(f'{file_dir}', f'*{channel}_aligned.npy'))[0]
                    channel_files.append(np.load(ch_aligned_path))
                except IndexError:
                    print(f'Could not find an aligned file for channel {channel}')
                    print(f'Resulting data will not contain fluorescent data for this channel')
                    channel_files.append(None)
    else:
        for channel in channel_names:
            try:
                ch_not_aligned_path = (
                    glob.glob(os.path.join(f'{file_dir}', f'*{channel}.tif'))[0]
                )
                channel_files.append(imread(ch_not_aligned_path))
            except IndexError:
                print(f'Could not find any file for channel {channel}')
                print(f'Resulting data will not contain fluorescent data for this channel')
                channel_files.append(None)
    return channel_files

def _load_files(file_dir, channels, segm_endname, load_channels_data=True):
    """
    Function to load files of all given channels and the corresponding segmentation masks.
    Check first if aligned files are available and use them if so.
    """
    acdc_output_endname = segm_endname.replace('segm', 'acdc_output')
    no_of_aligned_files = len(
        glob.glob(os.path.join(f'{file_dir}', '*aligned.npz'))
    )
    seg_mask_available = len(
        glob.glob(os.path.join(f'{file_dir}', f'*_{segm_endname}.npz'))
    ) > 0
    acdc_output_available = (
        len(glob.glob(os.path.join(f'{file_dir}', f'*{acdc_output_endname}.csv')))
        + len(glob.glob(os.path.join(f'{file_dir}', '*cc_stage*'))) > 0
    )
    if not (seg_mask_available and acdc_output_available):
        return None
    channel_files = []
    if load_channels_data:
        channels_data = _load_channels_data(file_dir, channels, no_of_aligned_files)
        channel_files.extend(channels_data)

    # append segmentation file
    try:
        segm_file_path = glob.glob(
            os.path.join(f'{file_dir}', f'*_{segm_endname}.npz')
        )[0]
        channel_files.append(np.load(segm_file_path)['arr_0'])
    except IndexError:
        segm_file_path = glob.glob(os.path.join(f'{file_dir}', '*_segm.npy'))[0]
        # assume segmentation mask to be .npy
        channel_files.append(np.load(segm_file_path))
    # append cc-data
    try:
        cc_stage_path = glob.glob(
            os.path.join(f'{file_dir}', f'*{acdc_output_endname}.csv')
        )[0]
    except IndexError:
        cc_stage_path = glob.glob(os.path.join(f'{file_dir}', '*cc_stage.csv'))[0]
    # assume cell cycle output of ACDC to be .csv
    channel_files.append(pd.read_csv(cc_stage_path))

    # append metadata if available, else append None
    if len(glob.glob(os.path.join(f'{file_dir}', '*metadata*'))) > 0:
        metadata_path = glob.glob(os.path.join(f'{file_dir}', '*metadata.csv'))[0]
        # assume calculated metadata to be .csv
        channel_files.append(pd.read_csv(metadata_path).set_index('Description'))
    else:
        channel_files.append(None)

    # append cc-properties if available, else append None
    if len(glob.glob(os.path.join(f'{file_dir}', '*_downstream*'))) > 0:
        cc_props_path = glob.glob(os.path.join(f'{file_dir}', '*_downstream*'))[0]
        # assume calculated cc properties to be .csv
        channel_files.append(pd.read_csv(cc_props_path))
    else:
        channel_files.append(None)
    return (*channel_files, cc_stage_path)

def _calculate_rp_df(seg_mask, is_timelapse_data, is_zstack_data, metadata, max_frame=1, label_input=False):
    """
    function to calculate regionprops based on a 2D(!) segmentation mask.
    TODO: insert check if 3D segmentation mask is available and calculate more regionprops.
    """
    if label_input:
        #generate labeled video only when input is not labeled yet
        labeled_data = label(seg_mask)
    else:
        labeled_data = seg_mask.copy()
    # calculate rp's for rings
    t_df = pd.DataFrame()
    props = ('label', 'area', 'convex_area', 'filled_area','major_axis_length',
             'minor_axis_length', 'orientation', 'perimeter', 'centroid', 'solidity')
    rename_dict = {'label':'Cell_ID', 'centroid-0':'centroid_y', 'centroid-1':'centroid_x'}
    if is_timelapse_data:
        for t, img in enumerate(tqdm(labeled_data)):
            # build time-dependent dataframes for further use (later for cca)
            if img.max() > 0:
                t_rp_df = pd.DataFrame(regionprops_table(img.astype(int), properties=props)).rename(columns=rename_dict)
                t_rp_df['frame_i'] = t
                # calculate volumes based on regionprops
                if metadata is None:
                    warnings.warn("No metadata available. Volumes are not calculated")
                    t_rp_df['cell_vol_vox_downstream'] = 0
                    t_rp_df['cell_vol_fl_downstream'] = 0
                else:
                    t_rp = regionprops(img.astype(int))
                    vol_vox = [_calc_rot_vol(obj, metadata.loc["PhysicalSizeY"], metadata.loc["PhysicalSizeX"])[0] for obj in t_rp]
                    vol_fl = [_calc_rot_vol(obj, metadata.loc["PhysicalSizeY"], metadata.loc["PhysicalSizeX"])[1] for obj in t_rp]
                    assert len(t_rp_df) == len(vol_vox)
                    t_rp_df['cell_vol_vox_downstream'] = vol_vox
                    t_rp_df['cell_vol_fl_downstream'] = vol_fl
                # determine id's which are falsely merged by 3D-labeling
                for r_id in t_rp_df.Cell_ID.unique():
                    bin_label = label((img==r_id).astype(int))
                    t_rp_df.loc[t_rp_df['Cell_ID']==r_id, '2d_label_count'] = bin_label.max()
                t_df = pd.concat([t_df, t_rp_df], ignore_index=True)
        # calculate global features by grouping
        grouped_df = t_df.groupby('Cell_ID').agg(
            min_t=('frame_i', min),
            max_t=('frame_i', max),
            lifespan=('frame_i', lambda x: max(x)-min(x)+1)
        ).reset_index()
        merged_df = t_df.merge(grouped_df, how='left', on='Cell_ID')
        # calculate further indicators based on merged data
        merged_df['age'] = merged_df['frame_i'] - merged_df['min_t'] + 1
        merged_df['frames_till_gone'] = merged_df['max_t'] - merged_df['frame_i']
        merged_df['elongation'] = merged_df['major_axis_length']/merged_df['minor_axis_length']
        return merged_df
    else:
        rp_df = pd.DataFrame(regionprops_table(labeled_data.astype(int), properties=props)).rename(columns=rename_dict)
        for r_id in rp_df.Cell_ID.unique():
            bin_label = label((labeled_data==r_id).astype(int))
            rp_df.loc[rp_df['Cell_ID']==r_id, '2d_label_count'] = bin_label.max()
        rp_df['elongation'] = rp_df['major_axis_length']/rp_df['minor_axis_length']
        rp_df['frame_i'] = 0
        return rp_df


def _calc_rot_vol(obj, PhysicalSizeY=1, PhysicalSizeX=1, logger=None):
    """Given the region properties of a 2D object (from skimage.measure.regionprops).
    calculate the rotation volume as described in the Supplementary information of
    https://www.nature.com/articles/s41467-020-16764-x

    Parameters
    ----------
    obj : class skimage.measure.RegionProperties
        Single item of the list returned by from skimage.measure.regionprops.
    PhysicalSizeY : type
        Physical size of the pixel in the Y-diretion in micrometer/pixel.
    PhysicalSizeX : type
        Physical size of the pixel in the X-diretion in micrometer/pixel.

    Returns
    -------
    tuple
        Tuple of the calculate volume in voxels and femtoliters.

    Notes
    -------
    For 3D objects we take max projection

    We convert PhysicalSizeY and PhysicalSizeX to float because when they are
    read from csv they might be a string value.

    """
    is3Dobj = False
    try:
        orientation = obj.orientation
    except NotImplementedError as e:
        # if obj.image.ndim != 3:
        #     printl(e, obj.image.ndim, obj.bbox, obj.centroid)
        is3Dobj = True

    try:
        if is3Dobj:
            # For 3D objects we use a max projection for the rotation
            obj_lab = obj.image.max(axis=0).astype(np.uint32)*obj.label
            obj = regionprops(obj_lab)[0]

        vox_to_fl = float(PhysicalSizeY)*pow(float(PhysicalSizeX), 2)
        rotate_ID_img = skimage.transform.rotate(
            obj.image.astype(np.single), -(obj.orientation*180/np.pi),
            resize=True, order=3
        )
        radii = np.sum(rotate_ID_img, axis=1)/2
        vol_vox = np.sum(np.pi*(radii**2))
        if vox_to_fl is not None:
            return vol_vox, float(vol_vox*vox_to_fl)
        else:
            return vol_vox, vol_vox
    except Exception as e:
        if logger is not None:
            logger.exception(traceback.format_exc())
        else:
            printl(traceback.format_exc())
        return np.nan, np.nan


def _calculate_flu_signal(seg_mask, channel_data, channels, cc_data, is_timelapse_data, is_zstack_data):
    """
    function to calculate sum and scaled sum of fluorescence signal per frame and cell.
    channel_data is a list-like of TYX arrays, one for each channel.
    channels are the name of the channels in the tuple.
    cc_data the output of acdc.
    """        
    max_frame = cc_data.frame_i.max()
    df = pd.DataFrame(columns=['frame_i', 'Cell_ID'])
    bg_medians = []
    
    if seg_mask.ndim == 4:
        raise TypeError(
            '4D segmentation masks not supported. '
            'Feel free to request the new feature on our GitHub page '
            'https://github.com/SchmollerLab/Cell_ACDC/issues'
        )
    
    for i, ch_img in enumerate(channel_data):
        if ch_img.ndim == 3:
            continue
        
        # Use sum projections for 4D data
        channel_data[i] = ch_img.sum(axis=1)
    
    for ch_idx, ch_array in enumerate(channel_data):
        if ch_array is None:
            bg_medians.append(None)
        else:
            bg_index = np.logical_and(
                seg_mask[:max_frame+1]==0, ch_array[:max_frame+1]!=0
            )
            ch_medians = [np.median(ch_array[t][bg_index[t]]) for t in range(max_frame+1)]
            bg_medians.append(ch_medians)
    if is_timelapse_data:
        for cell_id in tqdm(cc_data.Cell_ID.unique()):
            temp_df = pd.DataFrame(columns=['frame_i', 'Cell_ID'])
            times = range(max_frame+1)
            temp_df['frame_i'] = times; temp_df['Cell_ID'] = cell_id
            index_array = (seg_mask[:max_frame+1] == cell_id)
            channel_data_cut = [c_arr[:max_frame+1] if c_arr is not None else None for c_arr in channel_data]
            for c_idx, c_array in enumerate(channel_data_cut):
                if c_array is not None:
                    cell_signal = c_array*index_array
                    # cell_signal = c_array[index_array]
                    summed = np.sum(cell_signal, axis=(1,2))
                    # count = np.sum(cell_signal!=0, axis=(1,2))
                    count = np.sum(index_array, axis=(1,2))
                    mean_signal = np.divide(summed, count, where=count!=0)
                    # mean_signal = np.mean(cell_signal, axis=(1,2))
                    corrected_signal = mean_signal - np.array(bg_medians[c_idx])
                    temp_df[f'{channels[c_idx]}_corrected_mean'] = corrected_signal
                    temp_df[f'{channels[c_idx]}_raw_sum'] = summed
                else:
                    temp_df[f'{channels[c_idx]}_corrected_mean'] = 0
                    temp_df[f'{channels[c_idx]}_raw_sum'] = 0
            df = pd.concat([df, temp_df], ignore_index=True)
        signal_indices = np.array(['_corrected_mean' in col for col in df.columns])
        keep_rows = df.loc[:,signal_indices].sum(axis=1) > 0
        df = df[keep_rows]
        df = df.sort_values(['frame_i', 'Cell_ID']).reset_index(drop=True)
    return df


def _rename_columns(cc_data):
    rename_dict = {
        'Cell cycle stage': 'cell_cycle_stage',
        '# of cycles': 'generation_num',
        "Relative's ID": 'relative_ID',
        'Relationship': 'relationship',
        'Emerg_frame_i': 'emerg_frame_i',
        'Division_frame_i': 'division_frame_i',
        'Discard': 'is_cell_excluded'
    }
    cc_data.columns = [rename_dict.get(col, col) for col in cc_data.columns]
    return cc_data


def binned_mean_stats(x, values, nbins, bins_min_count):
    """
    function to calculate binned means and corresponding standard errors for
    evenly spaced bins in the data ("x" gets distributed in bins, stats are calculated on "values")
    """
    bin_counts, _, _ = binned_statistic(x, values, statistic='count', bins=nbins)
    bin_means, bin_edges, _ = binned_statistic(x, values, bins=nbins)
    bin_std, _, _ = binned_statistic(x, values, statistic='std', bins=nbins)
    bin_standard_errors = bin_std/np.sqrt(bin_counts)
    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width/2
    x_errorbar = bin_centers[bin_counts>bins_min_count]
    y_errorbar = bin_means[bin_counts>bins_min_count]
    err_errorbar = 1.96 * bin_standard_errors[bin_counts>bins_min_count]
    return x_errorbar, y_errorbar, err_errorbar


def calculate_effect_size_cohen(data, group1, group2, cat_column='size_category', val_column='Pp38_concentration'):
    assert cat_column in data.columns and val_column in data.columns
    data_gr1 = data[data[cat_column]==group1]
    data_gr2 = data[data[cat_column]==group2]
    n1 = len(data_gr1)
    n2 = len(data_gr2)
    s1 = np.var(data_gr1[val_column])
    s2 = np.var(data_gr2[val_column])
    cohen_s = np.sqrt(
        ((n1-1)*s1+(n2-1)*s2) / (n1+n2-2)
    )
    effect_size = (np.mean(data_gr1[val_column])- np.mean(data_gr2[val_column])) / cohen_s
    return effect_size

def calculate_effect_size_glass(data, group1, group2, cat_column='size_category', val_column='Pp38_concentration'):
    assert cat_column in data.columns and val_column in data.columns
    data_gr1 = data[data[cat_column]==group1]
    data_gr2 = data[data[cat_column]==group2]
    glass_s = np.std(data_gr2[val_column])
    effect_size = (np.mean(data_gr1[val_column])- np.mean(data_gr2[val_column])) / glass_s
    return effect_size

def _add_end_of_frame_i_column(acdc_df):
    cca_df_idx = acdc_df.cell_cycle_stage.dropna().index
    cca_df = acdc_df.loc[cca_df_idx][cca_df_colnames]
    acdc_df['end_of_cell_cycle_frame_i'] = np.nan
    
    will_divice_cca_df_S = cca_df[
        (cca_df.cell_cycle_stage == 'S') & (cca_df.will_divide > 0)
    ].reset_index()
    
    cca_df['end_of_cell_cycle_frame_i'] = -1
    grouped_ID_gen_num = will_divice_cca_df_S.groupby(
        ['Cell_ID', 'generation_num']
    )
                
    end_cc_frame_i_per_cycle = grouped_ID_gen_num.agg(
        end_of_cell_cycle_frame_i=('frame_i', 'max')
    )
    
    cca_df_with_gen_num_idx = (
        cca_df.reset_index()
        .set_index(['Cell_ID', 'generation_num'])
        .sort_index()
    )
    
    for row in end_cc_frame_i_per_cycle.itertuples():
        ID, gen_num = row.Index
        end_cc_frame_i = row.end_of_cell_cycle_frame_i
        idx = (ID, gen_num)
        cca_df_with_gen_num_idx.loc[idx, 'end_of_cell_cycle_frame_i'] = (
            end_cc_frame_i
        )
    
    cca_df = (
        cca_df_with_gen_num_idx.reset_index()
        .set_index(['frame_i', 'Cell_ID'])
        .sort_index()
    )
    
    acdc_df.loc[cca_df_idx, 'end_of_cell_cycle_frame_i'] = (
        cca_df['end_of_cell_cycle_frame_i']
    )
    return acdc_df

def _extend_will_divide_to_G1(acdc_df):
    acdc_df = acdc_df.drop(columns=['level_0', 'index'], errors='ignore')
    acdc_df = acdc_df.reset_index()
    acdc_df_will_divide_true = acdc_df[acdc_df['will_divide'] > 0]
    grouped = acdc_df_will_divide_true.groupby(['Cell_ID', 'generation_num'])
    for (ID, gen_num) in grouped.groups.keys():
        mask = (
            (acdc_df['Cell_ID'] == ID)
            & (acdc_df['generation_num'] == gen_num)
        )
        acdc_df.loc[mask, 'will_divide'] = 1.0
    acdc_df = (
        acdc_df.reset_index()
        .set_index(['frame_i', 'Cell_ID'])
        .sort_index()
    )
    return acdc_df
    
def add_derived_cell_cycle_columns(acdc_df: pd.DataFrame):
    if 'cell_cycle_stage' not in acdc_df.columns:
        return acdc_df
    
    acdc_df = _extend_will_divide_to_G1(acdc_df)
    acdc_df = _add_end_of_frame_i_column(acdc_df)
    
    return acdc_df
    
def add_generation_num_of_relative_ID(
        acdc_df, prefix_index: Iterable[str]=None, reset_index=True
    ):
    relID_index_col = ['frame_i', 'relative_ID', 'Cell_ID']
    ID_index_col = ['frame_i', 'Cell_ID', 'relative_ID']
    
    if prefix_index is not None:
        relID_index_col = [*prefix_index, *relID_index_col]
        ID_index_col = [*prefix_index, *ID_index_col]
    
    if reset_index:
        acdc_df = acdc_df.reset_index()
    
    acdc_df_by_rel_ID = acdc_df.set_index(relID_index_col)
    acdc_df_by_rel_ID.index
    relative_ID_idx = acdc_df_by_rel_ID.index

    acdc_df_by_frame_i = acdc_df.set_index(ID_index_col)
    relative_ID_idx = relative_ID_idx.intersection(acdc_df_by_frame_i.index)
    acdc_df_by_frame_i['generation_num_relID'] = -1

    acdc_df_by_frame_i.loc[relative_ID_idx, 'generation_num_relID'] = (
        acdc_df_by_rel_ID.loc[relative_ID_idx, 'generation_num']
    )

    # Fix where generation_num_relID is still -1
    to_fix_mask = acdc_df_by_frame_i.generation_num_relID == -1
    acdc_df_to_fix = (
        acdc_df_by_frame_i[to_fix_mask]
        .reset_index()
        .set_index([*prefix_index, 'frame_i', 'relative_ID'])
    )

    acdc_df_by_cellID = (
        acdc_df_by_rel_ID.reset_index()
        .set_index([*prefix_index, 'frame_i', 'Cell_ID'])
    )
    # Intersection takes care of disappearing relative_IDs
    fixing_idx = acdc_df_to_fix.index.intersection(acdc_df_by_cellID.index)

    acdc_df_to_fix.loc[fixing_idx, 'generation_num_relID'] = (
        acdc_df_by_cellID.loc[fixing_idx, 'generation_num'].values
    )
    index_to_fix = acdc_df_by_frame_i[to_fix_mask].index
    acdc_df_by_frame_i.loc[index_to_fix, 'generation_num_relID'] = (
        acdc_df_to_fix['generation_num_relID'].values
    )
    
    acdc_df_with_col = acdc_df_by_frame_i.reset_index()
    return acdc_df_with_col
    
def get_IDs_gen_num_will_divide_wrong(global_cca_df):
    """Get a list of (ID, gen_num) of cells whose `will_divide`>0 but the 
    next generation does not exist (i.e., `will_divide` is wrong)

    Parameters
    ----------
    global_cca_df : pd.DataFrame
        DataFrame with cc annotations for every frame and Cell_ID

    Returns
    -------
    list of tuples
        List of (ID, gen_num) of cells whose `will_divide`>0 but the 
        next generation does not exist (i.e., `will_divide` is wrong)
    
    Notes
    -----
    To get the (ID, gen_num) where `will_divide` is wrong we first get an 
    index of (ID, gen_num) where `will_divide`>0. 
     
    Then we get the same index but with (ID, gen_num+1) which is the next 
    generation. 
    
    Finally we check if (ID, gen_num+1) actually exists in the annotations. 
    If not, those are wrongly annotated with `will_divide`>0. To check for 
    the existence we get the difference between the next gen index and the 
    whole DataFrame (i.e., get the (ID, gen_num+1) that do not exist in 
    annotations).
    """    
    global_cca_will_divide = (
        global_cca_df[(global_cca_df['will_divide'] > 0)]
    ).reset_index()
    
    ID_gen_num_index = (
        global_cca_df.reset_index()
        .set_index(['Cell_ID', 'generation_num'])
        .index
    )
    
    # Next generation index
    next_gen_will_divide_cca_df = (
        global_cca_will_divide[['Cell_ID', 'generation_num']].copy()
    )
    next_gen_will_divide_cca_df['generation_num'] += 1
    next_gen_will_divide_index = (
        next_gen_will_divide_cca_df.reset_index()
        .set_index(['Cell_ID', 'generation_num'])
        .index
    )
    
    # (ID, gen_num) list of cells with will_divide>0 but whose next 
    # generation number actually does not exist
    IDs_will_divide_next_gen_does_not_exist = (
        next_gen_will_divide_index.difference(ID_gen_num_index)
        .to_frame().to_numpy() # .to_list()
    )
    IDs_will_divide_next_gen_does_not_exist[:, -1] -= 1
    
    IDs_will_divide_wrong = list(zip(
        IDs_will_divide_next_gen_does_not_exist[:,0], 
        IDs_will_divide_next_gen_does_not_exist[:, 1]
    ))
    return IDs_will_divide_wrong
    
    