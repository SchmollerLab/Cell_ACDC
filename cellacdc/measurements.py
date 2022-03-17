import numpy as np
import pathlib
import sys
import os
import traceback

user_path = pathlib.Path.home()
metrics_path = os.path.join(user_path, 'acdc-metrics')
if not os.path.exists(metrics_path):
    os.makedirs(metrics_path)
sys.path.append(metrics_path)

from .core import numba_max, numba_min

def get_custom_metrics_func():
    scripts = os.listdir(metrics_path)
    custom_func_dict = {}
    for script in scripts:
        try:
            from script import script as func
            custom_func_dict[script] = func
        except Exception:
            traceback.print_exc()
    return custom_func_dict

def standard_metrics_func():
    metrics_func = {
        'mean': lambda arr: arr.mean(),
        'sum': lambda arr: arr.sum(),
        'amount_autoBkgr': lambda arr, bkgr, area: (arr.mean()-bkgr)*area,
        'amount_dataPrepBkgr': lambda arr, bkgr, area: (arr.mean()-bkgr)*area,
        'median': lambda arr: np.median(arr),
        'min': lambda arr: numba_min(arr),
        'max': lambda arr: numba_max(arr),
        'q25': lambda arr: np.quantile(arr, q=0.25),
        'q75': lambda arr: np.quantile(arr, q=0.75),
        'q05': lambda arr: np.quantile(arr, q=0.05),
        'q95': lambda arr: np.quantile(arr, q=0.95)
    }

    bkgr_val_names = (
        'autoBkgr_val_median',
        'autoBkgr_val_mean',
        'autoBkgr_val_q75',
        'autoBkgr_val_q25',
        'autoBkgr_val_q95',
        'autoBkgr_val_q05',
        'dataPrepBkgr_val_median',
        'dataPrepBkgr_val_mean',
        'dataPrepBkgr_val_q75',
        'dataPrepBkgr_val_q25',
        'dataPrepBkgr_val_q95',
        'dataPrepBkgr_val_q05',
    )

    all_metrics_names = list(metrics_func.keys())
    all_metrics_names.extend(bkgr_val_names)
    return metrics_func, all_metrics_names
