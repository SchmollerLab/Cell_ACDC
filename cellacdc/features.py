import skimage.measure
import pandas as pd
import numpy as np

from . import _core
from . import measurements

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

def get_acdc_df_features(
        posData, grouped_features, lab, foregr_img, frame_i, filename
    ):
    yx_pxl_to_um2 = posData.PhysicalSizeY*posData.PhysicalSizeX
    vox_to_fl_3D = (
        posData.PhysicalSizeY*posData.PhysicalSizeX*posData.PhysicalSizeZ
    )
    
    rp = skimage.measure.regionprops(lab)
    isSegm3D = lab.ndim == 3
    IDs = [obj.label for obj in rp]
    columns = grouped_features.values()
    data = np.zeros((len(IDs), len(columns)))
    df = pd.DataFrame(columns=columns, index=IDs, data=data)
    for category, names in grouped_features.items():
        if category == 'size':
            df = measurements.add_size_metrics(
                df, rp, names, isSegm3D, yx_pxl_to_um2, vox_to_fl_3D
            )
        elif category == 'standard':
            metrics_func, _ = measurements.standard_metrics_func()
            custom_func_dict = measurements.get_custom_metrics_func()
            
            # Get metrics to save
            params = measurements.get_metrics_params(
                names, metrics_func, custom_func_dict
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