from typing import Dict

import numpy as np
import pandas as pd
import pathlib
import re
import sys
import os
import traceback
import shutil
from importlib import import_module
import skimage.measure
from tqdm import tqdm

from . import core, base_cca_dict, cca_df_colnames, html_utils, config, printl
from . import user_profile_path, cca_functions

import warnings
warnings.filterwarnings("ignore", message="Failed to get convex hull image.")
warnings.filterwarnings("ignore", message="divide by zero encountered in long_scalars")
warnings.filterwarnings("ignore", message="Mean of empty slice.")
warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

acdc_metrics_path = os.path.join(user_profile_path, 'acdc-metrics')
if not os.path.exists(acdc_metrics_path):
    os.makedirs(acdc_metrics_path, exist_ok=True)
sys.path.append(acdc_metrics_path)

combine_metrics_ini_path = os.path.join(acdc_metrics_path, 'combine_metrics.ini')

cellacdc_path = os.path.dirname(os.path.abspath(__file__))
metrics_path = os.path.join(cellacdc_path, 'metrics')

# Copy metrics to acdc-metrics user path
for file in os.listdir(metrics_path):
    if not file.endswith('.py'):
        continue
    src = os.path.join(metrics_path, file)
    dst = os.path.join(acdc_metrics_path, file)
    shutil.copy(src, dst)

PROPS_DTYPES = {
    'label': int,
    'major_axis_length': float,
    'minor_axis_length': float,
    'inertia_tensor_eigvals': tuple,
    'equivalent_diameter': float,
    'moments': np.ndarray,
    'area': int,
    'solidity': float,
    'extent': float,
    'inertia_tensor': np.ndarray,
    'filled_area': int,
    'centroid': tuple,
    'bbox_area': int,
    'local_centroid': tuple,
    'convex_area': int,
    'euler_number': int,
    'moments_normalized': np.ndarray,
    'moments_central': np.ndarray,
    'bbox': tuple
}

def getMetricsFunc(posData):
    metrics_func, all_metrics_names = standard_metrics_func()
    total_metrics = len(metrics_func)
    custom_func_dict = get_custom_metrics_func()
    total_metrics += len(custom_func_dict)

    # Here we simply update the total. The combineMetricsConfig is
    # defined in loadData.setCombineMetricsConfig method
    for key, section in posData.combineMetricsConfig.items():
        total_metrics += len(section)
    return metrics_func, all_metrics_names, custom_func_dict, total_metrics

def get_all_metrics_names():
    all_metrics_names = []
    custom_metrics_names = list(_get_custom_metrics_names().keys())
    size_metrics_names = list(get_size_metrics_desc(True, True).keys())
    standard_metrics_names = list(_get_metrics_names().keys())
    bkgr_val_names = list(_get_bkgr_val_names().keys())
    props_names = get_props_names()
    all_metrics_names.extend(custom_metrics_names)
    all_metrics_names.extend(size_metrics_names)
    all_metrics_names.extend(standard_metrics_names)
    all_metrics_names.extend(bkgr_val_names)
    all_metrics_names.extend(props_names)
    return all_metrics_names

def get_all_acdc_df_colnames():
    all_acdc_df_colnames = get_all_metrics_names()
    all_acdc_df_colnames.append('frame_i')
    all_acdc_df_colnames.append('time_seconds')
    all_acdc_df_colnames.append('Cell_ID')
    all_acdc_df_colnames.extend(cca_df_colnames)
    additional_colnames = [
        'is_cell_dead',
        'is_cell_excluded',
        'x_centroid',
        'y_centroid',
        'was_manually_edited'
    ]
    all_acdc_df_colnames.extend(additional_colnames)
    return all_acdc_df_colnames

def get_user_combine_metrics_equations(chName, isSegm3D=False):
    _, equations = channel_combine_metrics_desc(chName, isSegm3D=isSegm3D)
    return equations

def get_custom_metrics_func():
    scripts = os.listdir(acdc_metrics_path)
    custom_func_dict = {}
    for file in scripts:
        if file == '__init__.py':
            continue
        module_name, ext = os.path.splitext(file)
        if ext != '.py':
            # print(f'The file {file} is not a python file. Ignoring it.')
            continue
        if module_name == 'combine_metrics_example':
            # Ignore the example
            continue
        try:
            module = import_module(module_name)
            func = getattr(module, module_name)
            custom_func_dict[module_name] = func
        except Exception:
            traceback.print_exc()
    return custom_func_dict

def read_saved_user_combine_config():
    configPars = _get_saved_user_combine_config()
    if configPars is None:
        configPars = config.ConfigParser()

    if 'equations' not in configPars:
        configPars['equations'] = {}

    if 'mixed_channels_equations' not in configPars:
        configPars['mixed_channels_equations'] = {}

    if 'channelLess_equations' not in configPars:
        configPars['channelLess_equations'] = {}

    return configPars


def _get_saved_user_combine_config():
    files = os.listdir(acdc_metrics_path)
    configPars = None
    for file in files:
        module_name, ext = os.path.splitext(file)
        if ext != '.ini':
            continue

        filePath = os.path.join(acdc_metrics_path, file)
        configPars = config.ConfigParser()
        configPars.read(filePath)
    return configPars

def add_user_combine_metrics(configPars, equation, colName, isMixedChannels):
    section = 'mixed_channels_equations' if isMixedChannels else 'equations'
    if section not in configPars:
        configPars[section] = {}
    configPars[section][colName] = equation
    return configPars

def add_channelLess_combine_metrics(configPars, equation, equation_name, terms):
    if 'channelLess_equations' not in configPars:
        configPars['channelLess_equations'] = {}
    terms = ','.join(terms)
    equation_terms = f'{equation};{terms}'
    configPars['channelLess_equations'][equation_name] = equation_terms
    return configPars

def save_common_combine_metrics(configPars):
    with open(combine_metrics_ini_path, 'w') as configfile:
        configPars.write(configfile)

def _get_custom_metrics_names():
    custom_func_dict = get_custom_metrics_func()
    keys = custom_func_dict.keys()
    custom_metrics_names = {func_name:func_name for func_name in keys}
    return custom_metrics_names

def custom_metrics_desc(
        isZstack, chName, posData=None, isSegm3D=False, 
        return_combine=False
    ):
    how_3Dto2D, how_3Dto2D_desc = get_how_3Dto2D(isZstack, isSegm3D)
    custom_metrics_names = _get_custom_metrics_names()
    custom_metrics_desc = {}
    for how, how_desc in zip(how_3Dto2D, how_3Dto2D_desc):
        for func_name, func_desc in custom_metrics_names.items():
            metric_name = f'{chName}_{func_name}{how}'
            if isZstack:
                note_txt = html_utils.paragraph(f"""
                    {_get_zStack_note(how_desc)}
                    Example: <code>{metric_name}</code> is the
                    <b>{func_desc.lower()}</b> of the {chName} signal after
                    converting 3D to 2D {how_desc}
                """)
            else:
                note_txt = ''

            desc = html_utils.paragraph(f"""
                <b>{func_desc}</b> is a custom defined measurement.<br><br>
                The code for this function is located at the following path:<br><br>
                <code>{os.path.join(acdc_metrics_path, func_desc)}.py</code><br><br>
                {note_txt}
            """)
            custom_metrics_desc[metric_name] = desc

    combine_metrics_desc, _ = channel_combine_metrics_desc(
        chName, posData=posData, isSegm3D=isSegm3D
    )
    custom_metrics_desc = {**custom_metrics_desc, **combine_metrics_desc}

    if return_combine:
        return custom_metrics_desc, combine_metrics_desc
    else:
        return custom_metrics_desc

def channel_combine_metrics_desc(chName, posData=None, isSegm3D=False):
    combine_metrics_configPars = read_saved_user_combine_config()

    how_3Dto2D, how_3Dto2D_desc = get_how_3Dto2D(True, isSegm3D)
    combine_metrics = combine_metrics_configPars['equations']
    if posData is not None:
        posDataEquations = posData.combineMetricsConfig['equations']
        combine_metrics = {**combine_metrics, **posDataEquations}
    combine_metrics_desc = {}
    all_metrics_names = get_all_metrics_names()
    equations = {}
    for name, equation in combine_metrics.items():
        metric_name = name
        if not any([metric in equation for metric in all_metrics_names]):
            # Equation does not contain any of the available metrics --> Skip it
            continue

        if chName not in metric_name:
            # Equation name is not specific to the requested channel --> Skip it
            continue

        how_3Dto2D_present = [how for how in how_3Dto2D if how in equation]
        isZstack = len(how_3Dto2D_present) > 0

        if isZstack:
            how_desc = how_3Dto2D_present[0]
            note_txt = html_utils.paragraph(f"""{_get_zStack_note(how_desc)}""")
        else:
            note_txt = ''

        desc = html_utils.paragraph(f"""
            <b>{metric_name}</b> is a custom combined measurement that is the
            <b>result of the following equation</b>:.<br><br>
            <code>{metric_name} = {equation}</code><br><br>
            {note_txt}
        """)
        combine_metrics_desc[metric_name] = desc
        equations[metric_name] = equation

    channelLess_combine_metrics = combine_metrics_configPars['channelLess_equations']
    for name, equation_terms in channelLess_combine_metrics.items():
        channelLess_equation, terms = equation_terms.split(';')
        _colNames = terms.split(',')
        metric_name = f'{chName}_{name}'
        equation = channelLess_equation
        for _col in _colNames:
            equation = equation.replace(_col, f'{chName}{_col}')

        if not any([metric in equation for metric in all_metrics_names]):
            # Equation does not contain any of the available metrics --> Skip it
            continue

        how_3Dto2D_present = [how for how in how_3Dto2D if how in equation]
        isZstack = len(how_3Dto2D_present) > 0

        if isZstack:
            how_desc = how_3Dto2D_present[0]
            note_txt = html_utils.paragraph(f"""{_get_zStack_note(how_desc)}""")
        else:
            note_txt = ''

        desc = html_utils.paragraph(f"""
            <b>{metric_name}</b> is a custom combined measurement that is the
            <b>result of the following equation</b>:.<br><br>
            <code>{metric_name} = {equation}</code><br><br>
            {note_txt}
        """)
        combine_metrics_desc[metric_name] = desc
        equations[metric_name] = equation

    return combine_metrics_desc, equations

def get_user_combine_mixed_channels_equations(isSegm3D=False):
    _, equations = _combine_mixed_channels_desc(isSegm3D=isSegm3D)
    return equations

def get_combine_mixed_channels_desc(isSegm3D=False):
    desc, _ = _combine_mixed_channels_desc(isSegm3D=isSegm3D)
    return desc

def _combine_mixed_channels_desc(isSegm3D=False, configPars=None):
    if configPars is None:
        configPars = _get_saved_user_combine_config()
        if configPars is None:
            return {}, {}

    equations = {}
    mixed_channels_desc = {}
    how_3Dto2D, how_3Dto2D_desc = get_how_3Dto2D(True, isSegm3D)
    mixed_channels_combine_metrics = configPars['mixed_channels_equations']
    all_metrics_names = get_all_metrics_names()
    equations = {}
    for name, equation in mixed_channels_combine_metrics.items():
        metric_name = name
        if not any([metric in equation for metric in all_metrics_names]):
            # Equation does not contain any of the available metrics --> Skip it
            continue

        how_3Dto2D_present = [how for how in how_3Dto2D if how in equation]
        isZstack = len(how_3Dto2D_present) > 0

        if isZstack:
            how_desc = how_3Dto2D_present[0]
            note_txt = html_utils.paragraph(f"""{_get_zStack_note(how_desc)}""")
        else:
            note_txt = ''

        desc = html_utils.paragraph(f"""
            <b>{metric_name}</b> is a custom combined measurement that is the
            <b>result of the following equation</b>:<br><br>
            <code>{metric_name} = {equation}</code><br><br>
            {note_txt}
        """)
        mixed_channels_desc[metric_name] = desc
        equations[metric_name] = equation
    return mixed_channels_desc, equations

def combine_mixed_channels_desc(posData=None, isSegm3D=False, available_cols=None):
    desc, equations = _combine_mixed_channels_desc(isSegm3D=isSegm3D)
    if posData is None:
        all_desc = desc
        all_equations = equations
    else:
        pos_desc, pos_equations = _combine_mixed_channels_desc(
            isSegm3D=isSegm3D, configPars=posData.combineMetricsConfig
        )
        all_desc = {**desc, **pos_desc}
        all_equations = {**equations, **pos_equations}
    
    if available_cols is not None:
        # Check that user folder combine metrics have the right columns
        available_desc = {}
        available_equations = {}
        for name, equation in all_equations.items():
            cols = re.findall(r'[A-Za-z0-9]+_[A-Za-z0-9_]+', equation)
            if all([col in available_cols for col in cols]):
                available_desc[name] = all_desc[name]
                available_equations[name] = equation
        return available_desc, available_equations
    else:
        return all_desc, all_equations

def _um3():
    return '<code>&micro;m<sup>3</sup></code>'

def _um2():
    return '<code>&micro;m<sup>2</sup></code>'

def _um():
    return '<code>&micro;</code>'

def _fl():
    return '<code>fl</code>'

def _get_zStack_note(how_desc):
    s = (f"""
        <i>NOTE: since you loaded <b>3D z-stacks</b>, Cell-ACDC needs
        to convert the z-stacks to 2D images {how_desc} for this metric.<br>
        This is specified in the name of the column.<br><br></i>
    """)
    return s

def get_size_metrics_desc(isSegm3D, is_timelapse):
    url = 'https://www.nature.com/articles/s41467-020-16764-x#Sec16'
    size_metrics = {
        'cell_area_pxl': html_utils.paragraph("""
            <b>Area of the segmented object</b> in pixels, i.e.,
            total number of pixels in the object.
        """),
        'cell_vol_vox': html_utils.paragraph(f"""
            <b>Estimated volume of the segmented object</b> in voxels.<br><br><br>
            To calculate object volume based on 2D masks, the object is first
            <b>aligned along its major axis</b>.<br><br>
            Next, it is divided into slices perpendicular to the
            major axis, where the height of each slice is one pixel
            (i.e., one row).<br><br>
            
            We then assume <b>rotational symmetry</b> of each slice around
            its middle axis parallel to the object's major axis.<br><br>
            For each slice Cell-ACDC calculates the volume of the resulting
            <b>cylinder</b> with one pixel height and diameter the width
            of the respective slice.<br><br>
            
            Finally, the volumes of each cylinder are added to obtain
            total object volume.<br><br>
            
            <i> Note that in <a href=\"{url}">this</a> publication we
            showed that this method strongly correlates with volume
            computed from a 3D segmentation mask.</i><br><br>
            
            This value might be <b>grayed out</b> because it is <b>required</b> 
            by the concentration metric that you requested to save 
            (see in the <code>Standard measurements</code> group) and <b>it 
            cannot be unchecked</b>.<br><br>
        """),
        'cell_area_um2': html_utils.paragraph(f"""
            <b>Area of the segmented object</b> in {_um2()}, i.e.,
            total number of pixels in the object.<br><br>
            Conversion from pixels to {_um2()} is perfomed using
            the provided pixel size.
        """),
        'cell_vol_fl': html_utils.paragraph(f"""
            <b>Estimated volume of the segmented object</b> in {_um3()}.<br><br><br>

            To calculate object volume based on 2D masks, the object is first
            <b>aligned along its major axis</b>.<br><br>

            Next, it is divided into slices perpendicular to the
            major axis, where the height of each slice is one pixel
            (i.e., one row).<br><br>

            We then assume <b>rotational symmetry</b> of each slice around
            its middle axis parallel to the object's major axis.<br><br>

            For each slice Cell-ACDC calculates the volume of the resulting
            <b>cylinder</b> with one pixel height and diameter the width
            of the respective slice.<br><br>

            Finally, the volumes of each cylinder are added and converted
            to {_fl()} (same as {_um3()}) using the provided pixel size
            to obtain total object volume.<br><br>

            <i> Note that in <a href=\"{url}">this</a> publication we
            showed that this method strongly correlates with volume
            computed from a 3D segmentation mask.</i><br><br>
            
            This value might be <b>grayed out</b> because it is <b>required</b> 
            by the concentration metric that you requested to save 
            (see in the <code>Standard measurements</code> group) and <b>it 
            cannot be unchecked</b>.<br><br>
        """)
    }
    if isSegm3D:
        size_metrics_3D = {
            'cell_vol_vox_3D': html_utils.paragraph(f"""
                <b>Volume</b> of the segmented object in <b>voxels</b>.<br><br>
                This is given by the total number of voxels inside the object.<br><br>
            
                This value might be <b>grayed out</b> because it is <b>required</b> 
                by the concentration metric that you requested to save 
                (see in the <code>Standard measurements</code> group) and <b>it 
                cannot be unchecked</b>.<br><br>
            """),
            'cell_vol_fl_3D': html_utils.paragraph(f"""
                <b>Volume</b> of the segmented object in <b>{_fl()}</b>.<br><br>
                This is given by the total number of voxels inside the object 
                multiplied by the voxel volume.<br><br>
                The voxel volume is given by:<br><br>
                <code>PhysicalSizeZ * PhysicalSizeY * PhysicalSizeX</code><br><br>
                where <code>PhysicalSizeZ</code> is the spacing between z-slices 
                (in {_um()}), while <code>PhysicalSizeY</code> and 
                <code>PhysicalSizeX</code> are the pixel height and pixel width, 
                respectively (in {_um()}).<br><br>
            
                This value might be <b>grayed out</b> because it is <b>required</b> 
                by the concentration metric that you requested to save 
                (see in the <code>Standard measurements</code> group) and <b>it 
                cannot be unchecked</b>.<br><br>
            """),
        }
        size_metrics = {**size_metrics, **size_metrics_3D}
    if is_timelapse:
        velocity_metrics = {
            'velocity_pixel': html_utils.paragraph(f"""
            Velocity in <code>[pixel/frame]</code> of the segmented object 
            between previous and current frame. 
        """),
        'velocity_um': html_utils.paragraph(f"""
            Velocity in <code>[{_um()}/frame]</code> of the segmented object 
            between previous and current frame. 
        """)
        }
        size_metrics = {**size_metrics, **velocity_metrics}
    return size_metrics

def get_how_3Dto2D(isZstack, isSegm3D):
    how_3Dto2D = ['_maxProj', '_meanProj', '_zSlice'] if isZstack else ['']
    if isSegm3D:
        how_3Dto2D.append('_3D')
    how_3Dto2D_desc = [
        'using a <b>max projection</b>',
        'using a <b>mean projection</b> (recommended for <b>confocal imaging</b>)',
        'using the <b>z-slice you used for segmentation</b> '
        '(recommended for <b>epifluorescence imaging</b>)'
        '<i>NOTE: if segmentation mask is <b>3D</b>, Cell-ACDC will use the '
        '<b>center z-slice</b> of each object.</i>',
        'using 3D data'
    ]
    return how_3Dto2D, how_3Dto2D_desc

def standard_metrics_desc(
        isZstack, chName, isManualBackgrPresent=False, isSegm3D=False
    ):
    how_3Dto2D, how_3Dto2D_desc = get_how_3Dto2D(isZstack, isSegm3D)
    metrics_names = _get_metrics_names(
        is_manual_bkgr_present=isManualBackgrPresent
    )
    bkgr_val_names = _get_bkgr_val_names(
        is_manual_bkgr_present=isManualBackgrPresent
    )
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
                """)
            else:
                note_txt = ''

            if func_desc == 'Amount':
                amount_formula = _get_amount_formula_str(func_name)
                amount_desc = (f"""
                    Amount is the <b>background corrected (subtracted) total
                    fluorescence intensity</b>, which is usually the best proxy
                    for the amount of the tagged molecule, e.g.,
                    <b>protein amount</b>.
                    <br><br>
                    The amount is calculated as follows:<br><br>
                    <code>{amount_formula}</code>
                    <br><br>
                    where <code>_obj</code> refers to the pixels inside the 
                    segmented object.
                    <br><br>
                """)
                main_desc = f'<b>{func_desc}</b> computed from'
            elif func_desc == 'Concentration':
                amount_desc = ("""
                    Concentration is given by <code>Amount/cell_volume</code>,
                    where amount is the <b>background corrected (subtracted) total
                    fluorescence intensity</b>. Amount is usually the best proxy
                    for the amount of the tagged molecule, e.g.,
                    <b>protein amount</b>.
                    <br><br>
                """)
                main_desc = f'<b>{func_desc}</b> computed from'
            else:
                amount_desc = ''
                main_desc = f'<b>{func_desc}</b> computed from'

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
                    data prep module (module 1.).<br><br>
                    Note taht this metric is <b>grayed out</b> and it cannot be selected 
                    if the <b>selection of the background ROIs was not performed</b>.
                """)
            elif func_name.find('_manualBkgr') != -1:
                bkgr_desc = ("""
                    <code>manualBkgr</code> means that the background value
                    used to correct the intensities is computed as the <b>mean</b>
                    of the pixels from the pixels inside each 
                    <b>background objects</b> that you selected in the
                    GUI module (module 3).<br><br>
                """)
            else:
                bkgr_desc = ''

            desc = html_utils.paragraph(f"""
                {main_desc} the pixels inside
                each segmented object.<br><br>
                {amount_desc}{bkgr_desc}{note_txt}
            """)
            metrics_desc[metric_name] = desc

        median_note = ("""
            Note that this value might be <b>grayed out</b> because it is <b>required</b> 
            by the corresponding amount metric that you requested to save 
            (see above in the <code>Standard measurements</code> group) and <b>it 
            cannot be unchecked</b>.<br><br>
        """)
        for bkgr_name, bkgr_desc in bkgr_val_names.items():
            bkgr_colname = f'{chName}_{bkgr_name}{how}'
            if isZstack:
                note_txt = (f"""
                {_get_zStack_note(how_desc)}
                Example: <code>{bkgr_colname}</code> is the
                <b>{bkgr_desc.lower()}</b> of the {chName} background after
                converting 3D to 2D {how_desc}
                """)
            else:
                note_txt = ''

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
                    data prep module (module 1.).<br><br>
                    Note taht this metric is <b>grayed out</b> and it cannot be selected 
                    if the <b>selection of the background ROIs was not performed</b>.
                    <br><br>
                """)
            if bkgr_name.find('bkgrVal_median') != -1:
                bkgr_type_desc = f'{bkgr_type_desc}{median_note}'

            bkgr_final_desc = html_utils.paragraph(f"""
                <b>{bkgr_desc}</b> of the background intensities.<br><br>
                {bkgr_type_desc}{note_txt}
            """)
            bkgr_val_desc[bkgr_colname] = bkgr_final_desc

    return metrics_desc, bkgr_val_desc

def get_conc_keys(amount_colname):
    conc_key_vox = re.sub(
        r'amount_([A-Za-z]+)',
        r'concentration_\1_from_vol_vox',
        amount_colname
    )
    conc_key_fl = conc_key_vox.replace('from_vol_vox', 'from_vol_fl')
    return conc_key_vox, conc_key_fl

def classify_acdc_df_colnames(acdc_df, channels):
    standard_funcs = _get_metrics_names()
    size_metrics_desc = get_size_metrics_desc(True, True)
    props_names = get_props_names()

    foregr_metrics = {ch:[] for ch in channels}
    bkgr_metrics = {ch:[] for ch in channels}
    custom_metrics = {ch:[] for ch in channels}
    size_metrics = []
    props_metrics = []

    for col in acdc_df.columns:
        for ch in channels:
            if col.startswith(f'{ch}_'):
                # Channel specific metric
                if col.find('_bkgrVal_') != -1:
                    # Bkgr metric
                    bkgr_metrics[ch].append(col)
                else:
                    # Foregr metric
                    is_standard = any(
                        [col.find(f'_{f}') != -1 for f in standard_funcs]
                    )
                    if is_standard:
                        # Standard metric
                        foregr_metrics[ch].append(col)
                    else:
                        # Custom metric
                        custom_metrics[ch].append(col)
                break
        else:
            # Non-channel specific metric
            if col in size_metrics_desc:
                # Size metric
                size_metrics.append(col)
            elif col in props_names:
                # Regionprop metric
                props_metrics.append(col)
    
    metrics = {
        'foregr': foregr_metrics, 
        'bkgr': bkgr_metrics,
        'custom': custom_metrics, 
        'size': size_metrics,
        'props': props_metrics
    }

    return metrics

def _get_metrics_names(is_manual_bkgr_present=False):
    metrics_names = {
        'mean': 'Mean',
        'sum': 'Sum',
        'amount_autoBkgr': 'Amount',
        'amount_dataPrepBkgr': 'Amount',
        'amount_manualBkgr': 'Amount',
        'mean_manualBkgr': 'Mean',
        'concentration_autoBkgr_from_vol_vox': 'Concentration',
        'concentration_dataPrepBkgr_from_vol_vox': 'Concentration',
        'concentration_autoBkgr_from_vol_fl': 'Concentration',
        'concentration_dataPrepBkgr_from_vol_fl': 'Concentration',
        'median': 'Median',
        'min': 'Minimum',
        'max': 'Maximum',
        'q25': '25 percentile',
        'q75': '75 percentile',
        'q05': '5 percentile',
        'q95': '95 percentile',
    }
    return metrics_names

def _get_amount_formula_str(func_name):
    if func_name.find('manualBkgr') != -1:
        formula = 'amount = (mean_obj - mean_background)*area_obj'
    else:
        formula = 'amount = (mean_obj - median_background)*area_obj'
    return formula

def _get_bkgr_val_names(is_manual_bkgr_present=False):
    bkgr_val_names = {
        'autoBkgr_bkgrVal_median': 'Median',
        'autoBkgr_bkgrVal_mean': 'Mean',
        'autoBkgr_bkgrVal_q75': '75 percentile',
        'autoBkgr_bkgrVal_q25': '25 percentile',
        'autoBkgr_bkgrVal_q95': '95 percentile',
        'autoBkgr_bkgrVal_q05': '5 percentile',
        'dataPrepBkgr_bkgrVal_median': 'Median',
        'dataPrepBkgr_bkgrVal_mean': 'Mean',
        'dataPrepBkgr_bkgrVal_q75': '75 percentile',
        'dataPrepBkgr_bkgrVal_q25': '25 percentile',
        'dataPrepBkgr_bkgrVal_q95': '95 percentile',
        'dataPrepBkgr_bkgrVal_q05': '5 percentile',
    }
    if is_manual_bkgr_present:
        bkgr_val_names['manualBkgr_bkgrVal_median'] = 'Median'
        bkgr_val_names['manualBkgr_bkgrVal_mean'] = 'Mean'
        bkgr_val_names['manualBkgr_bkgrVal_q75'] = '75 percentile'
        bkgr_val_names['manualBkgr_bkgrVal_q25'] = '25 percentile'
        bkgr_val_names['manualBkgr_bkgrVal_q95'] = '95 percentile'
        bkgr_val_names['manualBkgr_bkgrVal_q05'] = '5 percentile'
    return bkgr_val_names

def get_props_info_txt():
    url = 'https://scikit-image.org/docs/0.18.x/api/skimage.measure.html#skimage.measure.regionprops'
    txt = html_utils.paragraph(f"""
        Morphological properties are calculated using the function
        from the library <code>scikit-image</code> called
        <code>skimage.measure.regionprops</code>.<br><br>
        You can find more details about each one of the properties
        <a href=\"{url}"></a>.
    """)
    return txt

def _is_numeric_dtype(dtype):
    is_numeric = (
        dtype is float or dtype is int
    )
    return is_numeric

def get_bkgrROI_mask(posData, isSegm3D):
    if posData.bkgrROIs:
        ROI_bkgrMask = np.zeros(posData.lab.shape, bool)
        if posData.bkgrROIs:
            for roi in posData.bkgrROIs:
                xl, yl = [int(round(c)) for c in roi.pos()]
                w, h = [int(round(c)) for c in roi.size()]
                if isSegm3D:
                    ROI_bkgrMask[:, yl:yl+h, xl:xl+w] = True
                else:
                    ROI_bkgrMask[yl:yl+h, xl:xl+w] = True
    else:
        ROI_bkgrMask = None
    return ROI_bkgrMask

def get_autoBkgr_mask(lab, isSegm3D, posData, frame_i):
    autoBkgr_mask = lab == 0
    autoBkgr_mask = _mask_0valued_pixels_from_alignment(
        autoBkgr_mask, frame_i, posData
    )
    if isSegm3D:
        autoBkgr_mask_proj = lab.max(axis=0) == 0
        autoBkgr_mask_proj = _mask_0valued_pixels_from_alignment(
            autoBkgr_mask_proj, frame_i, posData
        )
    else:
        autoBkgr_mask_proj = autoBkgr_mask
    
    return autoBkgr_mask, autoBkgr_mask_proj

def regionprops_table(labels, props, logger_func=None):
    rp = skimage.measure.regionprops(labels)
    if 'label' not in props:
        props = ('label', *props)
    
    empty_metric = [None]*len(rp)
    rp_table = {}
    error_ids = {}
    pbar = tqdm(total=len(props), ncols=100, leave=False)
    for prop in props:
        pbar.set_description(f'Computing "{prop}"')
        for o, obj in enumerate(rp):
            try:
                metric = getattr(obj, prop)
                _type = PROPS_DTYPES[prop]
                if _type == int or _type == float:
                    if prop not in rp_table:
                        rp_table[prop] = empty_metric.copy()
                    rp_table[prop][o] = metric
                elif _type == tuple:
                    for m, val in enumerate(metric):
                        prop_1d = f'{prop}-{m}'
                        if prop_1d not in rp_table:
                            rp_table[prop_1d] = empty_metric.copy()
                        rp_table[prop_1d][o] = val
                elif _type == np.ndarray:
                    for i, val in enumerate(metric.flatten()):
                        indices = np.unravel_index(i, metric.shape)
                        s = '-'.join([str(idx) for idx in indices])
                        prop_1d = f'{prop}-{s}'
                        if prop_1d not in rp_table:
                            rp_table[prop_1d] = empty_metric.copy()
                        rp_table[prop_1d][o] = val
            except Exception as e:
                format_exception = traceback.format_exc()
                if logger_func is None:
                    printl(format_exception)
                else:
                    logger_func(format_exception)
                
                if prop not in error_ids:
                    error_ids[prop] = {'ids': [obj.label], 'error': e}
                else:
                    error_ids[prop]['ids'].append(obj.label)
        pbar.update(1)
    return rp_table, error_ids

def get_btrack_features():
    features = (
        'area',
        'major_axis_length', 
        'minor_axis_length',
        'equivalent_diameter',
        'solidity',
        'extent',
        'filled_area',
        'bbox_area',
        'convex_area',
        'euler_number',
        'orientation'
    )
    return features

def get_non_measurements_cols(colnames, metrics_colnames):
    non_metrics_colnames = []
    for col in colnames:
        if col in metrics_colnames:
            continue
        non_metrics_colnames.append(col)
    
    non_metrics_non_rp_colnames = []
    props = get_props_names()
    # Remove composite regionprops
    for col in non_metrics_colnames:
        for prop in props:
            match = re.match(rf'{prop}-\d', col)
            if match is not None:
                break
            match = re.match(rf'{col}-\d-\d', col)
            if match is not None:
                break
        else:
            non_metrics_non_rp_colnames.append(col)
    return non_metrics_non_rp_colnames  
            

def get_props_names_3D():
    props = {
        'label': int,
        'major_axis_length': float,
        'minor_axis_length': float,
        'inertia_tensor_eigvals': tuple,
        'equivalent_diameter': float,
        'moments': np.ndarray,
        'area': int,
        'solidity': float,
        'extent': float,
        'inertia_tensor': np.ndarray,
        'filled_area': int,
        'centroid': tuple,
        'bbox_area': int,
        'local_centroid': tuple,
        'convex_area': int,
        'euler_number': int,
        'moments_normalized': np.ndarray,
        'moments_central': np.ndarray,
        'bbox': tuple
    }
    return list(props.keys())

def get_props_names():
    props = {
        'label': int,
        'major_axis_length': float,
        'minor_axis_length': float,
        'inertia_tensor_eigvals': tuple,
        'equivalent_diameter': float,
        'moments': np.ndarray,
        'area': int,
        'solidity': float,
        'extent': float,
        'inertia_tensor': np.ndarray,
        'filled_area': int,
        'centroid': tuple,
        'bbox_area': int,
        'local_centroid': tuple,
        'convex_area': int,
        'euler_number': int,
        'moments_normalized': np.ndarray,
        'moments_central': np.ndarray,
        'bbox': tuple
    }
    return list(props.keys())

def _try_metric_func(func, *args):
    try:
        val = func(*args)
    except Exception as e:
        val = np.nan
    return val

def _quantile(arr, q):
    try:
        val = np.quantile(arr, q=q)
    except Exception as e:
        val = np.nan
    return val

def _amount(arr, bkgr, area):
    try:
        val = (np.mean(arr)-bkgr)*area
    except Exception as e:
        val = np.nan
    return val

def _mean_corrected(arr, bkgr):
    try:
        val = np.mean(arr)-bkgr
    except Exception as e:
        val = np.nan
    return val

def get_obj_size_metric(
        col_name, obj, isSegm3D, yx_pxl_to_um2, vox_to_fl_3D
    ):
    if col_name == 'cell_area_pxl':
        if isSegm3D:
            return np.count_nonzero(obj.image.max(axis=0))
        else:
            return obj.area
    elif col_name == 'cell_area_um2':
        if isSegm3D:
            return np.count_nonzero(obj.image.max(axis=0))*yx_pxl_to_um2
        else:
            return obj.area*yx_pxl_to_um2
    elif col_name == 'cell_vol_vox':
        if not hasattr(obj, 'vol_vox'):
            PhysicalSizeY = PhysicalSizeX = np.sqrt(yx_pxl_to_um2)
            vol_vox, vol_fl = cca_functions._calc_rot_vol(
                obj, PhysicalSizeY, PhysicalSizeX
            )
            obj.vol_vox, obj.vol_fl = vol_vox, vol_fl
        return obj.vol_vox
    elif col_name == 'cell_vol_fl':
        if not hasattr(obj, 'vol_fl'):
            PhysicalSizeY = PhysicalSizeX = np.sqrt(yx_pxl_to_um2)
            vol_vox, vol_fl = cca_functions._calc_rot_vol(
                obj, PhysicalSizeY, PhysicalSizeX
            )
            obj.vol_vox, obj.vol_fl = vol_vox, vol_fl
        return obj.vol_fl
    elif col_name == 'cell_vol_vox_3D':
        return obj.area
    elif col_name == 'cell_vol_fl_3D':
        return obj.area*vox_to_fl_3D

def get_foregr_data(foregr_img, isSegm3D, z):
    isZstack = foregr_img.ndim == 3
    foregr_data = {}
    if isSegm3D:
        foregr_data['3D'] = foregr_img
    
    if isZstack:
        foregr_data['maxProj'] = foregr_img.max(axis=0)
        foregr_data['meanProj'] = foregr_img.mean(axis=0)
        foregr_data['zSlice'] = foregr_img[z]
    foregr_data[''] = foregr_img
    return foregr_data

def get_cell_volumes_areas(df):
    try:
        cell_vol_vox = df.loc['cell_vol_vox']
    except Exception as e:
        cell_vol_vox = [np.nan]*len(df)
    
    try:
        cell_vol_fl = df.loc['cell_vol_fl']
    except Exception as e:
        cell_vol_fl = [np.nan]*len(df)
    
    try:
        cell_vol_vox_3D = df.loc['cell_vol_vox_3D']
    except Exception as e:
        cell_vol_vox_3D = [np.nan]*len(df)
    
    try:
        cell_vol_fl_3D = df.loc['cell_vol_fl_3D']
    except Exception as e:
        cell_vol_fl_3D = [np.nan]*len(df)
    
    try:
        cell_area_pxl = df.loc['cell_area_pxl']
    except Exception as e:
        cell_area_pxl = [np.nan]*len(df)
    
    try:
        cell_area_um2 = df.loc['cell_vol_fl_3D']
    except Exception as e:
        cell_area_um2 = [np.nan]*len(df)
    
    items = (
        cell_vol_vox, cell_vol_fl, cell_vol_vox_3D, cell_vol_fl_3D,
        cell_area_pxl, cell_area_um2
    )
    return items

def get_bkgrVals(df, channel, how, ID, bkgr_type=None):
    try:
        if how:
            autoBkgr_col = f'{channel}_autoBkgr_bkgrVal_median_{how}'
        else:
            autoBkgr_col = f'{channel}_autoBkgr_bkgrVal_median'
        autoBkgrVal = df.at[ID, autoBkgr_col]
    except Exception as e:
        autoBkgrVal = np.nan
    
    try:
        if how:
            dataPrepBkgr_col = f'{channel}_dataPrepBkgr_bkgrVal_median_{how}'
        else:
            dataPrepBkgr_col = f'{channel}_dataPrepBkgr_bkgrVal_median'
        dataPrepBkgrVal = df.at[ID, dataPrepBkgr_col]
    except Exception as e:
        dataPrepBkgrVal = np.nan

    if bkgr_type is None:
        return autoBkgrVal, dataPrepBkgrVal
    
    if bkgr_type.find('dataPrep') != -1:
        return dataPrepBkgrVal
    else:
        return autoBkgrVal

def get_manualBkgr_bkgrVal(df, channel, how, ID):
    try:
        if how:
            bkgr_col = f'{channel}_manualBkgr_bkgrVal_mean_{how}'
        else:
            bkgr_col = f'{channel}_dataPrepBkgr_bkgrVal_mean'
        bkgrVal = df.at[ID, bkgr_col]
    except Exception as e:
        bkgrVal = np.nan
    return bkgrVal

def get_foregr_obj_array(foregr_arr, obj, isSegm3D, z_slice=None, how=None):
    if foregr_arr.ndim == 3 and isSegm3D:
        # 3D mask on 3D data
        return foregr_arr[obj.slice][obj.image], obj.area
    elif foregr_arr.ndim == 2 and isSegm3D:
        # 3D mask on 2D data
        use_proj = (
            z_slice is None or how is None or how != 'zSlice'
        )
        obj_slice = obj.slice[1:3]
        if use_proj:
            obj_image = obj.image.max(axis=0)
            obj_area = np.count_nonzero(obj_image)
            return foregr_arr[obj_slice][obj_image], obj_area
        else:
            min_z = obj.bbox[0]
            z_local = z_slice - min_z
            try:
                obj_image = obj.image[z_local]
                obj_area = np.count_nonzero(obj_image)
                return foregr_arr[obj_slice][obj_image], obj_area
            except Exception as err:
                return np.zeros(3), 0
    else:
        # 2D mask on 2D data
        return foregr_arr[obj.slice][obj.image], obj.area

def _mask_0valued_pixels_from_alignment(bkgr_mask, frame_i, posData):
    if posData.loaded_shifts is None:
        # Not aligned --> there are no 0-valued pixels
        return bkgr_mask
    
    if posData.dataPrep_ROIcoords is not None:
        df_roi = posData.dataPrep_ROIcoords.loc[0]
        is_cropped = int(df_roi.at['cropped', 'value'])
        if is_cropped:
            # Do not mask 0valued pixels if image was cropped
            return bkgr_mask
    
    shifts = posData.loaded_shifts[frame_i]
    dy, dx = shifts
    if dy>0:
        bkgr_mask[..., :dy, :] = False
    elif dy<0:
        bkgr_mask[..., dy:, :] = False
    if dx>0:
        bkgr_mask[..., :dx] = False
    elif dx<0:
        bkgr_mask[..., dx:] = False
    
    return bkgr_mask

def get_bkgr_data(
        foregr_img, posData, filename, frame_i, autoBkgr_mask, z,
        autoBkgr_mask_proj, dataPrepBkgrROI_mask, isSegm3D, lab
    ):
    isZstack = foregr_img.ndim == 3
    bkgr_data = {}

    """Auto Background"""
    bkgr_data['autoBkgr'] =  {
        '': 0, 'maxProj': 0, 'meanProj': 0, 'zSlice': 0, '3D': 0
    }
    if isZstack:
        if isSegm3D:
            autoBkr_3D = foregr_img[autoBkgr_mask]
            bkgr_data['autoBkgr']['3D'] = autoBkr_3D
        autoBkgr_maxP = foregr_img.max(axis=0)[autoBkgr_mask_proj]
        autoBkgr_meanP = foregr_img.mean(axis=0)[autoBkgr_mask_proj]
        autoBkgr_zSlice = foregr_img[z][autoBkgr_mask_proj]
        bkgr_data['autoBkgr']['maxProj'] = autoBkgr_maxP
        bkgr_data['autoBkgr']['meanProj'] = autoBkgr_meanP
        bkgr_data['autoBkgr']['zSlice'] = autoBkgr_zSlice
    else:
        autoBkgr_data = foregr_img[autoBkgr_mask]
        bkgr_data['autoBkgr'][''] = autoBkgr_data

    """DataPrep Background"""
    bkgr_archive = posData.fluo_bkgrData_dict[filename]
    bkgr_data['dataPrepBkgr'] =  {
        '': [], 'maxProj': [], 'meanProj': [], 'zSlice': [], '3D': []
    }
    dataPrepBkgr_present = False
    if bkgr_archive is not None:
        # Background data saved separately after cropping in dataprep
        for file in bkgr_archive.files:
            bkgrRoi_data = bkgr_archive[file]
            if posData.SizeT > 1:
                bkgrRoi_data = bkgrRoi_data[frame_i]
            if isZstack:
                bkgrRoi_maxP = bkgrRoi_data.max(axis=0)
                bkgrRoi_meanP = bkgrRoi_data.mean(axis=0)
                bkgrRoi_zSlice = bkgrRoi_data[z]
                if isSegm3D:
                    bkgrRoi_3D = bkgrRoi_data
            else:
                bkgrRoi = bkgrRoi_data  
            dataPrepBkgr_present = True       
    elif dataPrepBkgrROI_mask is not None:
        # Get background data from the bkgr ROI mask
        dataPrepBkgrROI_mask = np.logical_and(dataPrepBkgrROI_mask, lab==0)
        if isZstack:
            if isSegm3D:
                bkgrRoi_3D = foregr_img[dataPrepBkgrROI_mask]
                dataPrepBkgrROI_mask_2D = dataPrepBkgrROI_mask[0]
            else:
                dataPrepBkgrROI_mask_2D = dataPrepBkgrROI_mask
            bkgrRoi_maxP = foregr_img.max(axis=0)[dataPrepBkgrROI_mask_2D]
            bkgrRoi_meanP = foregr_img.mean(axis=0)[dataPrepBkgrROI_mask_2D]
            bkgrRoi_zSlice = foregr_img[z][dataPrepBkgrROI_mask_2D]      
        else:
            bkgrRoi = foregr_img[dataPrepBkgrROI_mask]
        dataPrepBkgr_present = True 
    
    if isZstack and dataPrepBkgr_present:
        # Note: we do not try to exclude 0-valued pixels, see issue #285
        bkgr_data['dataPrepBkgr']['maxProj'].extend(bkgrRoi_maxP)
        bkgr_data['dataPrepBkgr']['meanProj'].extend(bkgrRoi_meanP)
        bkgr_data['dataPrepBkgr']['zSlice'].extend(bkgrRoi_zSlice)
        if isSegm3D:
            bkgr_data['dataPrepBkgr']['3D'].extend(bkgrRoi_3D)
    elif dataPrepBkgr_present:
        bkgr_data['dataPrepBkgr'][''].extend(bkgrRoi)
    
    return bkgr_data
            

def standard_metrics_func():
    metrics_func = {
        'sum': lambda arr: _try_metric_func(np.sum, arr),
        'amount_autoBkgr': lambda arr, bkgr, area: _amount(arr, bkgr, area),
        'amount_dataPrepBkgr': lambda arr, bkgr, area: _amount(arr, bkgr, area),
        'amount_manualBkgr': lambda arr, bkgr, area: _amount(arr, bkgr, area),
        'mean_manualBkgr': lambda arr, bkgr, area: _mean_corrected(arr, bkgr),
        'mean': lambda arr: _try_metric_func(np.mean, arr),
        'median': lambda arr: _try_metric_func(np.median, arr),
        'min': lambda arr: _try_metric_func(np.min, arr),
        'max': lambda arr: _try_metric_func(np.max, arr),
        'q25': lambda arr: _quantile(arr, 0.25),
        'q75': lambda arr: _quantile(arr, 0.75),
        'q05': lambda arr: _quantile(arr, 0.05),
        'q95': lambda arr: _quantile(arr, 0.95)
    }
    all_metrics_names = list(_get_metrics_names().keys())

    bkgr_val_names = tuple(_get_bkgr_val_names().keys())
    all_metrics_names.extend(bkgr_val_names)

    return metrics_func, all_metrics_names

def add_metrics_instructions():
    url = 'https://github.com/SchmollerLab/Cell_ACDC/issues'
    href = f'<a href="{url}">here</a>'
    rp_url = f'https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops'
    rp_href = f'<a href="{rp_url}">skimage.measure.regionproperties</a>'
    def_sh = html_utils.def_sh
    CV_sh = html_utils.CV_sh
    open_par_sh = html_utils.open_par_sh
    close_par_sh = html_utils.close_par_sh
    if_sh = html_utils.if_sh
    elif_sh = html_utils.elif_sh
    equal_sh = html_utils.equal_sh
    np_mean_sh = html_utils.np_mean_sh
    np_std_sh = html_utils.np_std_sh
    return_sh = html_utils.return_sh
    is_not_sh = html_utils.is_not_sh
    args_sh = html_utils.span(
        'signal, autoBkgr, dataPrepBkgr, objectRp, correct_with_bkgr=False, '
        'which_bkgr="auto"', color=html_utils.kwargs_color
    )
    s = html_utils.paragraph(f"""
    To add custom metrics to the <code>acdc_output.csv</code>
    file you need to <b>create a python script and save it into the
    following folder</b>
    <code>{acdc_metrics_path}</code><br><br>
    Name the file with the name of the column you want to add. Inside the
    file, create a <b>function with the same name of the file</b>
    similar to the example below:<br><br>
    Pseudo-code:
    <pre><code>
    {def_sh} {CV_sh}{open_par_sh}{args_sh}{close_par_sh}:<br>
        {if_sh} correct_with_bkgr:<br>
            {if_sh} which_bkgr {equal_sh}{equal_sh} 'auto':<br>
                signal {equal_sh} signal - autoBkgr<br>
            {elif_sh} dataPrepBkgr {is_not_sh} None:<br>
                signal {equal_sh} signal - dataPrepBkgr<br>

        <i># Here goes your custom metric computation</i><br>
        CV = {np_std_sh}(signal)/{np_mean_sh}(signal)<br>

        {return_sh} CV
    </code></pre>
    where <code>signal</code> is a vector contaning the fluorescence intesities 
    from the segmented object, <code>autoBkgr</code> is the median of the 
    background intensities (all the pixels outside of the objects), 
    <code>dataPrepBkgr</code> is the median of the pixels inside the 
    background ROI drawn in the data prep step, and <code>objectRp</code> 
    are the region properties of the segmented object computed with 
    the function <code>{rp_href}</code>.<br><br>
    Have a look at the <code>combine_metrics_example.py</code> file (click on "Show example..." below)
    for a full example.<br><br>
    <i>If it doesn't work, please report the issue {href} with the
    code you wrote. Thanks.</i>
    """)
    return s

def _get_combine_metrics_examples_list():
    examples = [
        """
        <code>concentration = _amount_autoBkgr/cell_vol_fl</code><br><br>
        --> <b>for each channel</b> create a new column with the name
        <code>channel_name_concentration</code><br>
        with the result of the division between <code>_amount_autoBkgr</code> and
        <code>cell_vol_fl</code><br><br>.
        """,
        """
        <code>autofluo_corrected_mean = _mean-channel_1_mean</code><br><br>
        --> <b>for each channel</b> create a new column with the name
        <code>channel_autofluo_corrected_mean</code><br>
        with the result of the subtraction between each channel
        signal's mean and the channel_1 signal's mean<br><br>.
        """,
        """
        <code>amount = channel_1_mean-channel_1_autoBkgr_bkgrVal_median</code><br><br>
        --> Create a <b>single new column</b> with the name
        <code>channel_1_amount</code><br>
        with the result of the subtraction between the channel_1 signal's mean
        and the channel_1 background's intensities median<br><br>.
        """,
        """
        <code>ch1_minus_ch2_mean = channel_1_mean-channel_2_mean</code><br><br>
        --> Create a <b>single new column</b> with the name
        <code>ch1_minus_ch2_mean</code><br>
        with the result of the subtraction between the channel_1 signal's mean
        and the channel_2 signal's mean.
        """
    ]
    return examples

def get_combine_metrics_help_txt():
    pandas_eval_url = 'https://pandas.pydata.org/docs/reference/api/pandas.eval.html'
    examples = _get_combine_metrics_examples_list()
    txt = html_utils.paragraph(f"""
        This dialog allows you to write an equation that will be used to
        <b>combine multiple measurements</b>.<br><br>

        To <b>add a measurement</b> to the equation <b>double-click</b> on
        any of the available measurements in the list.<br><br>

        To <b>add a mathematical operator</b> click on any
        of the tool buttons.<br><br>

        <b>Write a name</b> for the resulting measurement in the
        "New measurement name" field.<br>
        This field <b>cannot be left empty</b>.<br><br>

        This name will be used as the <b>name of a new column</b>
        in the <code>acdc_output.csv</code> file.<br>
        This column will contain the resulting measurement.<br><br>

        Once you wrote the name and the equation you need to <b>test it</b> by
        clicking on the "Test output" button.<br>
        This will evaluate the equation with random inputs and it will
        display the result below the button.<br>
        <b>Carefully inspect the output</b>, as it could contain an error.<br>
        If there is no error, <b>check that the result is as expected</b>.<br><br>

        Once you are happy with the test, click the "Ok" button.<br><br>

        If you need to calculate the <b>same measurement for each one of the
        channels</b><br>
        select the measurements from the "All channels measurements"
        item in the list.<br><br>

        You can add measurements specific for one channel or all channels
        measurements to the same equation<br>
        (e.g., if you need to divide a quantity of each channel
        with a specific channel measurement).<br><br>

        If you select any of the all channels measurement, Cell-ACDC will
        create <b>one column for each channel</b>.<br>
        The new columns will start with the channel name followed by the
        new measurement name.<br><br>

        Note that you can <b>manually edit the equation</b>, for example, if you
        need a mathematical operator<br>
        that is not available with the buttons.<br><br>

        Cell-ACDC uses the <b>Python package <code>pandas</code></b> to evaluate
        the expression.<br>
        You can <b>read more</b> about it
        {html_utils.href_tag('here', pandas_eval_url)}<br><br>

        The equations will be <b>saved</b> to both the loaded
        </b>Position folder</b><br>
        (file that ends with <code>custom_combine_metrics.ini</code>)
        and the following path:<br><br>
        <code>{combine_metrics_ini_path}</code><br><br>

        Examples:
        {html_utils.to_list(examples)}
    """)
    return txt

def add_concentration_metrics(df, concentration_metrics_params):
    for col, (func_name, how) in concentration_metrics_params.items():
        idx = col.find('_from_vol_')
        amount_col = col[:idx]
        amount_col = amount_col.replace('concentration_', 'amount_')
        if how:
            amount_col = f'{amount_col}_{how}'

        if col.find('from_vol_vox') != -1:
            try:
                if how == '3D':
                    cell_vol_values = df['cell_vol_vox_3D']
                else:
                    cell_vol_values = df['cell_vol_vox']
                concentration_values = df[amount_col]/cell_vol_values
            except Exception as e:
                concentration_values = np.nan
            df[col] = concentration_values
        elif col.find('from_vol_fl') != -1:
            try:
                if how == '3D':
                    cell_vol_values = df['cell_vol_fl_3D']
                else:
                    cell_vol_values = df['cell_vol_fl']
                concentration_values = df[amount_col]/cell_vol_values
            except Exception as e:
                concentration_values = np.nan
            df[col] = concentration_values
    return df

def add_size_metrics(
        df, rp, size_metrics_to_save, isSegm3D, yx_pxl_to_um2, vox_to_fl_3D
    ):
    for o, obj in enumerate(tqdm(rp, ncols=100, leave=False)):
        for col in size_metrics_to_save:
            val = get_obj_size_metric(
                col, obj, isSegm3D, yx_pxl_to_um2, vox_to_fl_3D
            )
            df.at[obj.label, col] = val
    return df

def add_foregr_metrics(
        df, rp, channel, foregr_data, foregr_metrics_params, metrics_func,
        custom_metrics_params, isSegm3D, lab, foregr_img, 
        other_channels_foregr_imgs: Dict[str, np.ndarray], 
        z_slice=None, manualBackgrRp=None, 
        customMetricsCritical=None, 
    ):
    if manualBackgrRp is not None:
        manualBackgrRp = {obj.label for obj in manualBackgrRp}
    custom_errors = ''
    # Iterate objects and compute foreground metrics
    for o, obj in enumerate(tqdm(rp, ncols=100, leave=False)):
        for col, (func_name, how) in foregr_metrics_params.items():
            func_name, how = foregr_metrics_params[col]
            foregr_arr = foregr_data[how]
            foregr_obj_arr, obj_area = get_foregr_obj_array(
                foregr_arr, obj, isSegm3D, z_slice=z_slice, how=how
            )
            is_manual_bkgr_metric = func_name.find('manualBkgr') != -1
            is_amount_metric = func_name.find('amount_') != -1
            if is_amount_metric and not is_manual_bkgr_metric:
                bkgr_type = func_name[len('amount_'):]
                try:
                    bkgr_val = get_bkgrVals(
                        df, channel, how, obj.label, bkgr_type=bkgr_type
                    )
                    func = metrics_func[func_name]
                    val = func(foregr_obj_arr, bkgr_val, obj_area)
                except Exception as e:
                    val = np.nan
            elif is_manual_bkgr_metric:
                bkgr_val = get_manualBkgr_bkgrVal(df, channel, how, obj.label)
                func = metrics_func[func_name]
                val = func(foregr_obj_arr, bkgr_val, obj_area)
            else:
                func = metrics_func[func_name]
                val = func(foregr_obj_arr)
            df.at[obj.label, col] = val

        for col, (custom_func, how) in custom_metrics_params.items():   
            foregr_arr = foregr_data[how]
            foregr_obj_arr, obj_area = get_foregr_obj_array(
                foregr_arr, obj, isSegm3D, z_slice=z_slice, how=how
            )
            ID = obj.label
            autoBkgrVal, dataPrepBkgrVal = get_bkgrVals(df, channel, how, ID)
            metrics_values = df.to_dict('list')
            items = get_cell_volumes_areas(df)
            (cell_vols_vox, cell_vols_fl, cell_vols_vox_3D, cell_vols_fl_3D,
            cell_areas_pxl, cell_areas_um2) = items
            custom_error, custom_val, custom_col_name = get_custom_metric_value(
                custom_func, foregr_obj_arr, autoBkgrVal, dataPrepBkgrVal, obj,
                o, metrics_values, cell_vols_vox, cell_vols_fl, cell_areas_pxl, 
                cell_areas_um2, foregr_img, lab, isSegm3D, 
                other_channels_foregr_imgs, col,
                cell_vols_vox_3D=cell_vols_vox_3D, 
                cell_vols_fl_3D=cell_vols_fl_3D
            )
            if custom_col_name is None:
                df.at[ID, col] = custom_val
            else:
                for custom_col, value in zip(custom_col_name, custom_val):
                    df.at[ID, custom_col] = value
                    
            if customMetricsCritical is not None and custom_error:
                customMetricsCritical.emit(custom_error, col)
    return df

def add_bkgr_values(
        df, bkgr_data, bkgr_metrics_params, metrics_func, 
        manualBackgrRp=None, foregr_data=None
    ):
    # Compute background values
    for col, (bkgr_type, func_name, how) in bkgr_metrics_params.items():
        bkgr_func = metrics_func[func_name]
        if bkgr_type == 'manualBkgr':
            add_manual_bkgr_values(
                manualBackgrRp, foregr_data, df, col, how, bkgr_func
            )
            continue
        bkgr_arr = bkgr_data[bkgr_type][how]
        bkgr_val = bkgr_func(bkgr_arr)
        df[col] = bkgr_val
    return df

def add_manual_bkgr_values(manualBackgrRp, foregr_data, df, col, how, bkgr_func):
    if manualBackgrRp is None:
        return
    if foregr_data is None:
        return
    foregr_img = foregr_data[how]
    for obj in manualBackgrRp:
        bkgr_obj_arr = foregr_img[obj.slice][obj.image]
        bkgr_val = bkgr_func(bkgr_obj_arr)
        df.at[obj.label, col] = bkgr_val

def add_regionprops_metrics(df, lab, regionprops_to_save, logger_func=None):
    if not regionprops_to_save:
        return df, []

    if 'label' not in regionprops_to_save:
        regionprops_to_save = ('label', *regionprops_to_save)

    rp_table, rp_errors = regionprops_table(
        lab, regionprops_to_save, logger_func=logger_func
    )

    df_rp = pd.DataFrame(rp_table).set_index('label')
    df_rp.index.name = 'Cell_ID'

    # Drop regionprops that were already calculated in a prev session
    df = df.drop(columns=df_rp.columns, errors='ignore')
    df = df.join(df_rp)
    return df, rp_errors

def get_custom_metric_value(
        custom_func, foregr_obj_arr, autoBkgrVal, dataPrepBkgrVal, obj,
        i, metrics_values, cell_vols_vox, cell_vols_fl, cell_areas_pxl, 
        cell_areas_um2, foregr_img, lab, isSegm3D, 
        other_channels_foregr_imgs: Dict[str, np.ndarray], col_name,
        cell_vols_vox_3D=None, 
        cell_vols_fl_3D=None
    ):
    base_args = (foregr_obj_arr, autoBkgrVal, dataPrepBkgrVal)
    
    metrics_obj = {key:mm[i] for key, mm in metrics_values.items()}
    metrics_obj['cell_vol_vox'] = cell_vols_vox[i]
    metrics_obj['cell_vol_fl'] = cell_vols_fl[i]
    metrics_obj['cell_area_pxl'] = cell_areas_pxl[i]
    metrics_obj['cell_area_um2'] = cell_areas_um2[i]
    if isSegm3D and cell_vols_vox_3D is not None and cell_vols_fl_3D is not None:
        metrics_obj['cell_vol_vox_3D'] = cell_vols_vox_3D[i]
        metrics_obj['cell_vol_fl_3D'] = cell_vols_fl_3D[i]
    
    additional_args_kwargs = (
        ((), {}),
        ((obj,), {}), 
        ((obj, metrics_obj), {}),
        ((obj, metrics_obj, foregr_img, lab), {'isSegm3D': isSegm3D}),
    )    
    error = None
    for args, kwargs in additional_args_kwargs:
        try:
            custom_val = custom_func(*base_args, *args, **kwargs)
            return '', custom_val, None
        except Exception as error:
            continue
    
    # Test if custom metric function requires the other channels images
    custom_vals_vs_other_ch = []
    col_names = []
    for other_channel, other_ch_img in other_channels_foregr_imgs.items():
        other_channel_foregr_img = {other_channel: other_ch_img}
        try:
            custom_val = custom_func(
                *base_args, obj, metrics_obj, foregr_img, lab, 
                other_channel_foregr_img, isSegm3D=isSegm3D
            )
            custom_vals_vs_other_ch.append(custom_val)
            col_names.append(f'{col_name}_vs_{other_channel}')
        except Exception as error:
            return traceback.format_exc(), np.nan, None
    
    return '', custom_vals_vs_other_ch, col_names

def get_metrics_params(all_channels_metrics, metrics_func, custom_func_dict):
    channel_names = list(all_channels_metrics.keys())
    bkgr_metrics_params = {ch:{} for ch in channel_names}
    foregr_metrics_params = {ch:{} for ch in channel_names}
    concentration_metrics_params = {}
    custom_metrics_params = {ch:{} for ch in channel_names}
    az = r'[A-Za-z0-9]'
    how_3D_to_2D_pattern = r'zSlice|3D|maxProj|meanProj|(?=\s*$)'
    bkgrVal_pattern = fr'_({az}+)_bkgrVal_({az}+)_?({az}*)$'

    for channel_name, columns in all_channels_metrics.items():
        for col in columns:
            m = re.findall(bkgrVal_pattern, col)
            if m:
                # The metric is a bkgrVal metric
                bkgr_type, func_name, how = m[0]
                bkgr_metrics_params[channel_name][col] = (
                    bkgr_type, func_name, how
                )
                continue
            
            is_standard_foregr = False
            for metric in metrics_func:
                foregr_pattern = (
                    rf'{channel_name}_({metric})_?({how_3D_to_2D_pattern}*)$'
                )
                m = re.findall(foregr_pattern, col)
                if m:
                    # Metric is a standard metric 
                    func_name, how = m[0]
                    foregr_metrics_params[channel_name][col] = (func_name, how)
                    is_standard_foregr = True
                    break
            
            if is_standard_foregr:
                continue

            # Metric is concentration
            conc_pattern = rf'concentration_{az}+_from_vol_[a-z]+'
            conc_metric_pattern = (
                rf'{channel_name}_({conc_pattern})_?({how_3D_to_2D_pattern}*)'
            )
            m = re.findall(conc_metric_pattern, col)
            if m:
                func_name, how = m[0]
                concentration_metrics_params[col] = (func_name, how)
                continue

            for metric, custom_func in custom_func_dict.items():
                custom_pattern = rf'{channel_name}_({metric})_?({how_3D_to_2D_pattern}*)'
                m = re.findall(custom_pattern, col)
                if m:
                    # Metric is a standard metric 
                    func_name, how = m[0]
                    custom_metrics_params[channel_name][col] = (custom_func, how)
                    break
    
    params = (
        bkgr_metrics_params, foregr_metrics_params, 
        concentration_metrics_params, custom_metrics_params
    )
    return params

def get_regionprops_columns(existing_colnames, selected_props_names):
    selected_rp_cols = []
    for col in existing_colnames:
        for selected_prop in selected_props_names:
            if selected_prop == col:
                selected_rp_cols.append(col)
                continue
            m = re.match(fr'{selected_prop}-\d', col)
            if m is not None:
                selected_rp_cols.append(col)
    return selected_rp_cols