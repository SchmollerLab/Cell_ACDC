import numpy as np
import pathlib
import re
import sys
import os
import traceback
import shutil
from importlib import import_module
import skimage.measure
from tqdm import tqdm

from . import core, base_cca_df, html_utils, config, printl

import warnings
warnings.filterwarnings("ignore", message="Failed to get convex hull image.")
warnings.filterwarnings("ignore", message="divide by zero encountered in long_scalars")

user_path = pathlib.Path.home()
acdc_metrics_path = os.path.join(user_path, 'acdc-metrics')
if not os.path.exists(acdc_metrics_path):
    os.makedirs(acdc_metrics_path)
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
    size_metrics_names = list(get_size_metrics_desc().keys())
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
    cca_df_colnames = list(base_cca_df.keys())
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
            <b>result of the following equation</b>:.<br><br>
            <code>{metric_name} = {equation}</code><br><br>
            {note_txt}
        """)
        mixed_channels_desc[metric_name] = desc
        equations[metric_name] = equation
    return mixed_channels_desc, equations

def combine_mixed_channels_desc(posData=None, isSegm3D=False):
    desc, equations = _combine_mixed_channels_desc(isSegm3D=isSegm3D)
    if posData is None:
        return desc, equations
    pos_desc, pos_equations = _combine_mixed_channels_desc(
        isSegm3D=isSegm3D, configPars= posData.combineMetricsConfig
    )
    return {**desc, **pos_desc}, {**equations, **pos_equations}

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

def get_size_metrics_desc():
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
            computed from a 3D segmentation mask.</i>
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
            computed from a 3D segmentation mask.</i>
        """),
        'cell_vol_vox_3D': html_utils.paragraph(f"""
            <b>Volume</b> of the segmented object in <b>voxels</b>.<br><br>
            This is given by the total number of voxels inside the object.
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
            respectively (in {_um()}). 
        """),
        'velocity_pixel': html_utils.paragraph(f"""
            Velocity in <code>[pixel/frame]</code> of the segmented object 
            between previous and current frame. 
        """),
        'velocity_um': html_utils.paragraph(f"""
            Velocity in <code>[{_um()}/frame]</code> of the segmented object 
            between previous and current frame. 
        """)
    }
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

def standard_metrics_desc(isZstack, chName, isSegm3D=False):
    how_3Dto2D, how_3Dto2D_desc = get_how_3Dto2D(isZstack, isSegm3D)
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
                """)
            else:
                note_txt = ''

            if func_desc == 'Amount':
                amount_desc = ("""
                    Amount is the <b>background corrected total
                    fluorescent intensity</b>, which is usually the best proxy
                    for the amount of the tagged molecule, e.g.,
                    <b>protein amount</b>.
                    <br><br>
                """)
                main_desc = f'<b>{func_desc}</b> computed from'
            elif func_desc == 'Concentration':
                amount_desc = ("""
                    Concentration is given by <code>Amount/cell_volume</code>,
                    where amount is the <b>background corrected total
                    fluorescent intensity</b>. Amount is usually the best proxy
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
                    data prep module (module 1.).<br>
                    If you didn't do this step, then the amount will be set to 0.
                    <br><br>
                """)
            else:
                bkgr_desc = ''

            desc = html_utils.paragraph(f"""
                {main_desc} the pixels inside
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
                    data prep module (module 1.).<br>
                    If you didn't do this step, then this value will be set to 0.
                    <br><br>
                """)

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
    size_metrics_desc = get_size_metrics_desc()
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
                # regionprop metric
                props_metrics.append(col)
    
    metrics = {
        'foregr': foregr_metrics, 
        'bkgr': bkgr_metrics,
        'custom': custom_metrics, 
        'size': size_metrics,
        'props': props_metrics
    }

    return metrics

def _get_metrics_names():
    metrics_names = {
        'mean': 'Mean',
        'sum': 'Sum',
        'amount_autoBkgr': 'Amount',
        'amount_dataPrepBkgr': 'Amount',
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
        'q95': '95 percentile'
    }
    return metrics_names

def _get_bkgr_val_names():
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

def standard_metrics_func():
    metrics_func = {
        'mean': lambda arr: arr.mean(),
        'sum': lambda arr: arr.sum(),
        'amount_autoBkgr': lambda arr, bkgr, area: (arr.mean()-bkgr)*area,
        'amount_dataPrepBkgr': lambda arr, bkgr, area: (arr.mean()-bkgr)*area,
        'median': lambda arr: np.median(arr),
        'min': lambda arr: np.min(arr),
        'max': lambda arr: np.max(arr),
        'q25': lambda arr: np.quantile(arr, q=0.25),
        'q75': lambda arr: np.quantile(arr, q=0.75),
        'q05': lambda arr: np.quantile(arr, q=0.05),
        'q95': lambda arr: np.quantile(arr, q=0.95)
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
    def <b>CV</b>(signal, autoBkgr, dataPrepBkgr, objectRp, correct_with_bkgr=False, which_bkgr='auto'):
        if correct_with_bkgr:
            if which_bkgr=='auto':
                signal = signal - autoBkgr
            elif dataPrepBkgr is not None:
                signal = signal - dataPrepBkgr

        <i># Here goes your custom metric computation</i>
        CV = np.std(signal)/np.mean(signal)

        return CV
    </code></pre>
    where <code>signal</code> is a vector contaning the fluorescent intesities 
    from the segmented object, <code>autoBkgr</code> is the median of the 
    background intensities (all the pixels outside of the objects), 
    <code>dataPrepBkgr</code> is the median of the pixels inside the 
    background ROI drawn in the data prep step, and <code>objectRp</code> 
    are the region properties of the segmented object computed with 
    the function <code>{rp_href}</code>.<br><br>
    Have a look at the <code>CV.py</code> file (click on "Show example..." below)
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
