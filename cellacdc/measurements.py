import numpy as np
import pathlib
import re
import sys
import os
import traceback
import shutil
from importlib import import_module

from skimage.measure._regionprops import PROPS, COL_DTYPES, RegionProperties

from . import core, base_cca_df, html_utils

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
        'editIDclicked_x',
        'editIDclicked_y',
        'editIDnewID',
        'editIDnewIDs'
    ]
    all_acdc_df_colnames.extend(additional_colnames)
    return all_acdc_df_colnames


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
                note_txt = html_utils.paragraph(f"""
                    {_get_zStack_note(how_desc)}
                    Example: <code>{metric_name}</code> is the
                    <b>{func_desc.lower()}</b> of the {chName} signal after
                    converting 3D to 2D {how_desc}
                """)
            else:
                note_txt = ''

            desc = html_utils.paragraph(f"""
                <b>{func_desc}</b> is a custom defined measurement.<br>
                {note_txt}
            """)
            custom_metrics_desc[metric_name] = desc
    return custom_metrics_desc

def _um3():
    return '<code>&micro;m<sup>3</sup></code>'

def _um2():
    return '<code>&micro;m<sup>2</sup></code>'

def _fl():
    return '<code>fl</code>'

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
        """)
    }
    return size_metrics

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
                converting 3D to 2D conversion {how_desc}
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

def get_props_names():
    props = (
        'label',
        'inertia_tensor_eigvals',
        'major_axis_length',
        'equivalent_diameter',
        'moments',
        'area',
        'solidity',
        'feret_diameter_max',
        'extent',
        'inertia_tensor',
        'filled_area',
        'centroid',
        'bbox_area',
        'local_centroid',
        'convex_area',
        'euler_number',
        'minor_axis_length',
        'moments_normalized',
        'moments_central',
        'bbox'
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
    all_metrics_names = list(_get_metrics_names().keys())

    bkgr_val_names = tuple(_get_bkgr_val_names().keys())
    all_metrics_names.extend(bkgr_val_names)

    return metrics_func, all_metrics_names

def add_metrics_instructions():
    url = 'https://github.com/SchmollerLab/Cell_ACDC/issues'
    href = f'<a href="{url}">here</a>'
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
    def <b>CV</b>(signal, autoBkgr, dataPrepBkgr, correct_with_bkgr=False, which_bkgr='auto'):
        if correct_with_bkgr:
            if which_bkgr=='auto':
                signal = signal - autoBkgr
            elif dataPrepBkgr is not None:
                signal = signal - dataPrepBkgr

        <i># Here goes your custom metric computation</i>
        CV = np.std(signal)/np.mean(signal)

        return CV
    </code></pre>
    Have a look at the <code>CV.py</code> file (click on "Show example..." below)
    for a full example.<br><br>
    <i>If it doesn't work, please report the issue {href} with the
    code you wrote. Thanks.</i>
    """)
    return s, metrics_path
