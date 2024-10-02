import traceback
import re

import skimage.measure
import pandas as pd
import numpy as np

from . import _core
from . import measurements
from . import printl

def add_rotational_volume_regionprops(
        rp, PhysicalSizeY=1, PhysicalSizeX=1, logger_func=None
    ):
    for obj in rp:
        vol_vox, vol_fl = _core._calc_rotational_vol(
            obj, PhysicalSizeY, PhysicalSizeX, logger=logger_func
        )
        obj.vol_vox, obj.vol_fl = vol_vox, vol_fl
    return rp

def filter_acdc_df_by_features_range(features_range, acdc_df):
    queries = [] 
    for feature_name, thresholds in features_range.items():
        if feature_name not in acdc_df.columns:
            pass
        _min, _max = thresholds
        if _min is not None:
            queries.append(f'({feature_name} > {_min})')
        if _max is not None:
            queries.append(f'({feature_name} < {_max})')
    if not queries:
        return acdc_df
    
    query = ' & '.join(queries)

    return acdc_df.query(query)

def _eval_equation_df(df, new_col_name, expression):
    try:
        df[new_col_name] = df.eval(expression)
    except Exception as error:
        traceback.print_exc()

def _add_combined_metrics_acdc_df(posData, df):
    # Add channel specifc combined metrics (from equations and 
    # from user_path_equations sections)
    config = posData.combineMetricsConfig
    for chName in posData.loadedChNames:
        posDataEquations = config['equations']
        userPathChEquations = config['user_path_equations']
        for newColName, equation in posDataEquations.items():
            _eval_equation_df(df, newColName, equation)
        for newColName, equation in userPathChEquations.items():
            _eval_equation_df(df, newColName, equation)

def get_acdc_df_features(
        posData, grouped_features, lab, foregr_img, frame_i, filename,
        channel, bkgrData, other_channels_foregr_imgs
    ):
    posData.fluo_bkgrData_dict[filename] = bkgrData
    yx_pxl_to_um2 = posData.PhysicalSizeY*posData.PhysicalSizeX
    vox_to_fl_3D = (
        posData.PhysicalSizeY*posData.PhysicalSizeX*posData.PhysicalSizeZ
    )
    
    rp = skimage.measure.regionprops(lab)
    isSegm3D = lab.ndim == 3
    
    # Initialise DataFrame
    IDs = [obj.label for obj in rp]
    columns = []
    for category, metrics_names in grouped_features.items():
        if isinstance(metrics_names, dict):
            for channel, channel_metrics in metrics_names.items():
                columns.extend(channel_metrics)
        else:
            columns.extend(metrics_names)
    data = np.zeros((len(IDs), len(columns)))
    df = pd.DataFrame(columns=columns, index=IDs, data=data)
    df.index.name = 'Cell_ID'
    for category, metrics_names in grouped_features.items():
        if category == 'size':
            df = measurements.add_size_metrics(
                df, rp, metrics_names, isSegm3D, yx_pxl_to_um2, vox_to_fl_3D
            )
        elif category == 'standard':
            metrics_func, _ = measurements.standard_metrics_func()
            custom_func_dict = measurements.get_custom_metrics_func()
            
            # Get metrics to save
            params = measurements.get_metrics_params(
                metrics_names, metrics_func, custom_func_dict
            )
            (bkgr_metrics_params, foregr_metrics_params, 
            concentration_metrics_params, custom_metrics_params) = params
            
            # Get background masks
            autoBkgr_masks = measurements.get_autoBkgr_mask(
                lab, isSegm3D, posData, frame_i
            )
            
            autoBkgr_mask, autoBkgr_mask_proj = autoBkgr_masks
            dataPrepBkgrROI_mask = measurements.get_bkgrROI_mask(
                posData, isSegm3D
            )
            
            # Get the z-slice if we have z-stacks
            z = posData.zSliceSegmentation(filename, frame_i)
            
            # Get the background data
            bkgr_data = measurements.get_bkgr_data(
                foregr_img, posData, filename, frame_i, autoBkgr_mask, z,
                autoBkgr_mask_proj, dataPrepBkgrROI_mask, isSegm3D, lab
            )
            
            # Compute background values
            df = measurements.add_bkgr_values(
                df, bkgr_data, bkgr_metrics_params[channel], metrics_func
            )
            
            foregr_data = measurements.get_foregr_data(foregr_img, isSegm3D, z)
            
            # Iterate objects and compute foreground metrics
            df = measurements.add_foregr_metrics(
                df, rp, channel, foregr_data, foregr_metrics_params[channel], 
                metrics_func, custom_metrics_params[channel], isSegm3D, 
                lab, foregr_img, other_channels_foregr_imgs, 
                customMetricsCritical=None,
                z_slice=z
            )

            df = measurements.add_concentration_metrics(
                df, concentration_metrics_params
            )
        elif category == 'regionprop':
            try:
                df, rp_errors = measurements.add_regionprops_metrics(
                    df, lab, metrics_names, logger_func=print
                )
            except Exception as error:
                traceback.print_exc()
        
    # Remove 0s columns
    df = df.loc[:, (df != -2).any(axis=0)]
    
    return df

def add_background_metrics_names(
        grouped_features, channel, isSegm3D, isZstack, isManualBackgrPresent
    ):
    _, bkgr_val_desc = measurements.standard_metrics_desc(
        isZstack, channel, isSegm3D=isSegm3D, 
        isManualBackgrPresent=isManualBackgrPresent
    )
    backgr_metrics_names = list(bkgr_val_desc.keys())
    backgr_metrics_names = [
        name for name in backgr_metrics_names 
        if (name.find('bkgrVal_median')!=-1 or name.find('bkgrVal_mean')!=-1)
    ]
    if 'standard' not in grouped_features:
        grouped_features['standard'] = {channel: backgr_metrics_names}
    else:
        for backgr_metric_name in backgr_metrics_names:
            if backgr_metric_name in grouped_features['standard'][channel]:
                continue
            grouped_features['standard'][channel].append(backgr_metric_name)
    return grouped_features

def custom_post_process_segm(
        posData, grouped_features, lab, img, frame_i, filename, channel,
        features_range, other_channels_foregr_imgs=None, return_delIDs=False
    ):
    isSegm3D = lab.ndim == 3
    isZstack = posData.SizeZ > 1
    bkgrData = posData.bkgrData
    isManualBackgrPresent = False
    if posData.manualBackgroundLab is not None:
        isManualBackgrPresent = True
    grouped_features = add_background_metrics_names(
        grouped_features, channel, isSegm3D, isZstack, isManualBackgrPresent
    )
    df = get_acdc_df_features(
        posData, grouped_features, lab, img, frame_i, filename, channel, 
        bkgrData, other_channels_foregr_imgs
    )
    try:
        filtered_df = filter_acdc_df_by_features_range(features_range, df)
    except Exception as err:
        return lab
    filtered_lab = np.zeros_like(lab)
    rp = skimage.measure.regionprops(lab)
    for obj in rp:
        if obj.label not in filtered_df.index:
            continue
        filtered_lab[obj.slice][obj.image] = obj.label
    if return_delIDs:
        return filtered_lab, df.index.difference(filtered_df.index).to_list()
    else:
        return filtered_lab