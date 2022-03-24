import numpy as np
import pathlib
import sys
import os
import traceback
import shutil
from importlib import import_module

from . import core

user_path = pathlib.Path.home()
acdc_metrics_path = os.path.join(user_path, 'acdc-metrics')
if not os.path.exists(acdc_metrics_path):
    os.makedirs(acdc_metrics_path)
sys.path.append(acdc_metrics_path)

cellacdc_path = os.path.dirname(os.path.abspath(__file__))
metrics_path = os.path.join(cellacdc_path, 'metrics')

# Copy metrics to acdc-metrics user path
for file in os.listdir(metrics_path):
    if not file.endswith('.py'):
        continue
    src = os.path.join(metrics_path, file)
    dst = os.path.join(acdc_metrics_path, file)
    shutil.copy(src, dst)


def get_custom_metrics_func():
    scripts = os.listdir(acdc_metrics_path)
    custom_func_dict = {}
    for file in scripts:
        module_name, ext = os.path.splitext(file)
        if ext != '.py':
            # print(f'The file {file} is not a python file. Ignoring it.')
            continue
        try:
            module = import_module(module_name)
            func = getattr(module, module_name)
            custom_func_dict[module_name] = func
        except Exception:
            traceback.print_exc()
    return custom_func_dict

def _get_custom_metrics_names():
    custom_func_dict = get_custom_metrics_func()
    keys = custom_func_dict.keys()
    custom_metrics_names = {func_name:func_name for func_name in keys}
    return custom_metrics_names

def custom_metrics_desc(isZstack, chName):
    how_3Dto2D = ['_maxProj', '_meanProj', '_zSlice'] if isZstack else ['']
    how_3Dto2D_desc = _get_how_3Dto2D_desc()
    custom_metrics_names = _get_custom_metrics_names()
    custom_metrics_desc = {}
    for how, how_desc in zip(how_3Dto2D, how_3Dto2D_desc):
        for func_name, func_desc in custom_metrics_names.items():
            metric_name = f'{chName}_{func_name}{how}'
            if isZstack:
                note_txt = (f"""
                <p style="font-size:13px">
                    {_get_zStack_note(how_desc)}
                    Example: <code>{metric_name}</code> is the
                    <b>{func_desc.lower()}</b> of the {chName} signal after
                    converting 3D to 2D {how_desc}
                </p>
                """)
            else:
                note_txt = '</p>'

            desc = (f"""
            <p style="font-size:13px">
                <b>{func_desc}</b> is a custom defined measurement.<br>
                {note_txt}
            """)
            custom_metrics_desc[metric_name] = desc
    return custom_metrics_desc

def _get_zStack_note(how_desc):
    s = (f"""
        <i>NOTE: since you loaded <b>3D z-stacks</b>, Cell-ACDC needs
        to convert the z-stacks to 2D images {how_desc}.<br>
        This is specified in the name of the column.<br><br></i>
    """)
    return s

def _get_how_3Dto2D_desc():
    how_3Dto2D_desc = [
        'using a <b>max projection</b>',
        'using a <b>mean projection</b> (recommended for <b>confocal imaging</b>)',
        'using the <b>z-slice you used for segmentation</b> '
        '(recommended for <b>epifluorescence imaging</b>)'
    ]
    return how_3Dto2D_desc

def standard_metrics_desc(isZstack, chName):
    how_3Dto2D = ['_maxProj', '_meanProj', '_zSlice'] if isZstack else ['']
    how_3Dto2D_desc = _get_how_3Dto2D_desc()
    metrics_names = _get_metrics_names()
    bkgr_val_names = _get_bkgr_val_names()
    metrics_desc = {}
    bkgr_val_desc = {}
    for how, how_desc in zip(how_3Dto2D, how_3Dto2D_desc):
        for func_name, func_desc in metrics_names.items():
            metric_name = f'{chName}_{func_name}{how}'
            if isZstack:
                note_txt = (f"""
                {_get_zStack_note(how_desc)}
                Example: <code>{metric_name}</code> is the
                <b>{func_desc.lower()}</b> of the {chName} signal after
                converting 3D to 2D {how_desc}
                </p>
                """)
            else:
                note_txt = '</p>'

            if func_desc == 'Amount':
                amount_desc = ("""
                    Amount is the <b>background corrected total
                    fluorescent intensity</b>, which is usually the best proxy
                    for the amount of the tagged molecule, e.g.,
                    <b>protein amount</b>.
                    <br><br>
                """)
            else:
                amount_desc = ''

            if func_name == 'amount_autoBkgr':
                bkgr_desc = ("""
                    <code>autoBkgr</code> means that the background value
                    used to correct the intensities is computed as the <b>median</b>
                    of ALL the pixels outside of the segmented objects
                    (i.e., pixels with ID 0 in the segmentation mask)
                    <br><br>
                """)
            elif func_name == 'amount_dataPrepBkgr':
                bkgr_desc = ("""
                    <code>dataPrepBkgr</code> means that the background value
                    used to correct the intensities is computed as the <b>median</b>
                    of the pixels from the pixels inside the rectangular
                    <b>background ROIs</b> that you selected in the
                    data prep module (module 1.).<br>
                    If you didn't do this step, then the amount will be set to 0.
                    <br><br>
                """)
            else:
                bkgr_desc = ''

            desc = (f"""
            <p style="font-size:13px">
                <b>{func_desc}</b> of the intensities from all the pixels inside
                each segmented object.<br><br>
                {amount_desc}{bkgr_desc}{note_txt}
            """)
            metrics_desc[metric_name] = desc

        for bkgr_name, bkgr_desc in bkgr_val_names.items():
            bkgr_colname = f'{chName}_{bkgr_name}{how}'
            if isZstack:
                note_txt = (f"""
                {_get_zStack_note(how_desc)}
                Example: <code>{bkgr_colname}</code> is the
                <b>{bkgr_desc.lower()}</b> of the {chName} background after
                converting 3D to 2D conversion {how_desc}
                </p>
                """)
            else:
                note_txt = '</p>'

            if bkgr_name.find('autoBkgr') != -1:
                bkgr_type_desc = ("""
                    <code>autoBkgr</code> means that the background value
                    is computed from ALL the pixels outside
                    of the segmented objects
                    (i.e., pixels with ID 0 in the segmentation mask)
                    <br><br>
                """)
            else:
                bkgr_type_desc = ("""
                    <code>dataPrepBkgr</code> means that the background value
                    is computed from the pixels inside the rectangular
                    <b>background ROIs</b> that you selected in the
                    data prep module (module 1.).<br>
                    If you didn't do this step, then this value will be set to 0.
                    <br><br>
                """)

            bkgr_final_desc = (f"""
            <p style="font-size:13px">
                <b>{bkgr_desc}</b> of the background intensities.<br><br>
                {bkgr_type_desc}{note_txt}
            """)
            bkgr_val_desc[bkgr_colname] = bkgr_final_desc

    return metrics_desc, bkgr_val_desc

def _get_metrics_names():
    metrics_names = {
        'mean': 'Mean',
        'sum': 'Sum',
        'amount_autoBkgr': 'Amount',
        'amount_dataPrepBkgr': 'Amount',
        'median': 'Median',
        'min': 'Minimum',
        'max': 'Maximum',
        'q25': '25 percentile',
        'q75': '75 percentile',
        'q05': '5 percentile',
        'q95': '95 percentile'
    }
    return metrics_names

def _get_bkgr_val_names():
    bkgr_val_names = {
        'autoBkgr_val_median': 'Median',
        'autoBkgr_val_mean': 'Mean',
        'autoBkgr_val_q75': '75 percentile',
        'autoBkgr_val_q25': '25 percentile',
        'autoBkgr_val_q95': '95 percentile',
        'autoBkgr_val_q05': '5 percentile',
        'dataPrepBkgr_val_median': 'Median',
        'dataPrepBkgr_val_mean': 'Mean',
        'dataPrepBkgr_val_q75': '75 percentile',
        'dataPrepBkgr_val_q25': '25 percentile',
        'dataPrepBkgr_val_q95': '95 percentile',
        'dataPrepBkgr_val_q05': '5 percentile',
    }
    return bkgr_val_names

def get_props_info_txt():
    url = 'https://scikit-image.org/docs/0.18.x/api/skimage.measure.html#skimage.measure.regionprops'
    txt = (f"""
    <p style="font-size:13px"
        Morphological properties are calculated using the function
        from the library <code>scikit-image</code> called
        <code>skimage.measure.regionprops</code>.<br><br>
        You can find more details about each one of the properties
        <a href=\"{url}"></a>.
    </p>
    """)
    return txt

def get_props_names():
    props = (
        'label',
        'bbox',
        'bbox_area',
        'eccentricity',
        'equivalent_diameter',
        'euler_number',
        'extent',
        'filled_area',
        'inertia_tensor_eigvals',
        'local_centroid',
        'major_axis_length',
        'minor_axis_length',
        'moments',
        'moments_central',
        'moments_hu',
        'moments_normalized',
        'orientation',
        'perimeter',
        'solidity'
    )
    return props

def standard_metrics_func():
    metrics_func = {
        'mean': lambda arr: arr.mean(),
        'sum': lambda arr: arr.sum(),
        'amount_autoBkgr': lambda arr, bkgr, area: (arr.mean()-bkgr)*area,
        'amount_dataPrepBkgr': lambda arr, bkgr, area: (arr.mean()-bkgr)*area,
        'median': lambda arr: np.median(arr),
        'min': lambda arr: core.numba_min(arr),
        'max': lambda arr: core.numba_max(arr),
        'q25': lambda arr: np.quantile(arr, q=0.25),
        'q75': lambda arr: np.quantile(arr, q=0.75),
        'q05': lambda arr: np.quantile(arr, q=0.05),
        'q95': lambda arr: np.quantile(arr, q=0.95)
    }

    bkgr_val_names = tuple(_get_bkgr_val_names().keys())

    all_metrics_names = list(metrics_func.keys())
    all_metrics_names.extend(bkgr_val_names)
    return metrics_func, all_metrics_names
