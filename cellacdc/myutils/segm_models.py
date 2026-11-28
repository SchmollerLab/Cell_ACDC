"""Segmentation models, promptable models and trackers: the custom-model
registry and the instructions for adding models, the lists of available
models and trackers, importing their modules, ArgSpec and parameter
introspection, initialising models and trackers, and logging the
segmentation parameters. Plugin API: getModelArgSpecFromEnv,
resolve_segm_model_init_data. The Qt-free model introspection used by
service/worker.py and the model environments is model_modules.py, whose
ArgSpec is a different namedtuple.
"""
import os
import typing
from typing import List, Dict
import subprocess
from importlib import import_module
from collections import namedtuple
import json
import numpy as np
import pandas as pd
import inspect
from natsort import natsorted
from cellacdc import GUI_INSTALLED, RUNNING_AS_SERVER
from cellacdc import load
from cellacdc import html_utils, issues_url
from cellacdc import cellacdc_path, printl
from cellacdc import models_list_file_path, models_path
from cellacdc import promptable_models_list_file_path, promptable_models_path
from . import model_modules as myutils_models
if GUI_INSTALLED or RUNNING_AS_SERVER:
    from cellacdc import config
from . import filesystem


ArgSpec = namedtuple('ArgSpec', ['name', 'default', 'type', 'desc', 'docstring'])


def get_add_custom_prompt_model_instructions():
    init_sh = html_utils.init_sh
    segment_sh = html_utils.segment_sh
    add_prompt_sh = html_utils.add_prompt_sh
    href = f'<a href="{issues_url}">here</a>'
    text = html_utils.paragraph(f"""
    To use a custom prompt model, you need to create a Python file with the name 
    <code>acdcPromptModel.py</code>.<br>
    Note that the folder name where you place this file will be used as the 
    model name.<br><br>
    In this file, you will implement a class called <code>Model</code> with 
    at least the <code>{init_sh}</code> to initialise the model,<br>
    the <code>{add_prompt_sh}</code> method to add prompts (points, boxes, etc.) 
    to the model, and the <code>{segment_sh}</code> method to run the 
    segmentation.<br><br>
    Have a look at the existing models in the <code>promptable_models</code> 
    folder for examples.<br><br>
    If it doesn't work, please report the issue {href} with the
    code you wrote. Thanks!
    """)
    return text

def get_add_custom_model_instructions():
    user_manual_url = 'https://github.com/SchmollerLab/Cell_ACDC/blob/main/UserManual/Cell-ACDC_User_Manual.pdf'
    href_user_manual = f'<a href="{user_manual_url}">user manual</a>'
    href = f'<a href="{issues_url}">here</a>'
    class_sh = html_utils.class_sh
    def_sh = html_utils.def_sh
    kwargs_sh = html_utils.kwargs_sh
    Model_sh = html_utils.Model_sh
    segment_sh = html_utils.segment_sh
    predict_sh = html_utils.predict_sh
    init_sh = html_utils.init_sh
    myModel_sh = html_utils.myModel_sh
    return_sh = html_utils.return_sh
    equal_sh = html_utils.equal_sh
    open_par_sh = html_utils.open_par_sh
    close_par_sh = html_utils.close_par_sh
    image_sh = html_utils.image_sh
    from_sh = html_utils.from_sh
    import_sh = html_utils.import_sh
    s = html_utils.paragraph(f"""
    To use a custom model first <b>create a folder</b> with the name of your model.<br><br>
    Inside this new folder create a file named <code>acdcSegment.py</code>.<br><br>
    In the <code>acdcSegment.py</code> file you will <b>implement the model class</b>.<br><br>
    Have a look at the other existing models, but essentially you have to create
    a class called <code>Model</code> with at least<br>
    the <code>{init_sh}</code> and the <code>{segment_sh}</code> method.<br><br>
    The <code>{segment_sh}</code> method takes the image (2D or 3D) as an input and return the segmentation mask.<br><br>
    You can find more details in the {href_user_manual} at the section
    called <code>Adding segmentation models to the pipeline</code>.<br><br>
    Pseudo-code for the <code>acdcSegment.py</code> file:
    <pre><code>
    {from_sh} myModel {import_sh} {myModel_sh}

    {class_sh} {Model_sh}:
        {def_sh} {init_sh}(self, {kwargs_sh}):
            self.model {equal_sh} {myModel_sh}{open_par_sh}{close_par_sh}

        {def_sh} {segment_sh}(self, {image_sh}, {kwargs_sh}):
            labels {equal_sh} self.model.{predict_sh}{open_par_sh}{image_sh}{close_par_sh}
            {return_sh} labels
    </code></pre>
    
    If it doesn't work, please report the issue {href} with the
    code you wrote. Thanks.
    """)
    return s


def store_custom_model_path(model_file_path):
    model_file_path = model_file_path.replace('\\', '/')
    model_name = os.path.basename(os.path.dirname(model_file_path))
    cp = config.ConfigParser()
    if os.path.exists(models_list_file_path):
        cp.read(models_list_file_path)
    if model_name not in cp:
        cp[model_name] = {}
    cp[model_name]['path'] = model_file_path
    with open(models_list_file_path, 'w') as configFile:
        cp.write(configFile)

def store_custom_promptable_model_path(promptable_model_file_path):
    model_file_path = promptable_model_file_path.replace('\\', '/')
    model_name = os.path.basename(os.path.dirname(model_file_path))
    cp = config.ConfigParser()
    if os.path.exists(promptable_models_list_file_path):
        cp.read(promptable_models_list_file_path)
    if model_name not in cp:
        cp[model_name] = {}
    cp[model_name]['path'] = model_file_path
    with open(promptable_models_list_file_path, 'w') as configFile:
        cp.write(configFile)


def setDefaultValueArgSpecsFromKwargs(
        params: List[ArgSpec], 
        kwargs: Dict[str, object]
    ):
    new_params = []
    for param in params:
        new_value = kwargs.get(param.name)
        if new_value is None:
            new_params.append(param)
            continue
        
        new_param = ArgSpec(
            name=param.name, 
            default=new_value, 
            type=param.type, 
            desc=param.desc,
            docstring=param.docstring
        )
        new_params.append(new_param)
    return new_params

def insertModelArgSpec(
        params, param_name, param_value, param_type=None, desc='',
        docstring=''
    ):
    updated_params = []
    for param in params:
        if param.name == param_name:
            if param_type is None:
                param_type = param.type
            new_param = ArgSpec(
                name=param_name, default=param_value, type=param_type,
                desc=desc, docstring=docstring
            )
            updated_params.append(new_param)
        else:
            updated_params.append(param)
    return updated_params

def get_function_argspec(function, args_to_skip={'logger_func',}):
    argspecs = inspect.getfullargspec(function)
    kwargs_type_hints = typing.get_type_hints(function)
    docstring = function.__doc__
    params = params_to_ArgSpec(
        argspecs, kwargs_type_hints, docstring, 
        args_to_skip=args_to_skip
    )
    return params

def getModelArgSpec(acdcSegment):
    return myutils_models.getModelArgSpec(acdcSegment)


def _deserialize_model_param_type(data):
    from cellacdc import _types

    if data['kind'] == 'choices':
        return type('ModelParamChoices', (), {
            'values': data['values'],
            'is_exclusive_with': data.get('is_exclusive_with', []),
            'default_exclusive': data.get('default_exclusive', ''),
            'cast_dtype': staticmethod(_types.to_str),
        })
    return {
        'bool': bool, 'int': int, 'float': float, 'str': str,
        'path': os.PathLike, 'none': type(None),
        'vector': _types.Vector, 'folder': _types.FolderPath,
        'second_channel': _types.SecondChannelImage, 'hidden': _types.NotGUIParam,
        'yeaz_model': _types.YeazV2ModelType,
        'csv_path': _types.CsvFilePath,
    }[data['kind']]


def getModelArgSpecFromEnv(
        model_name: str, python_exec: str, model_kind: str = 'segmentation'
    ):
    """Query model argument specification (init_params, segment_params, url)
    by running inspection inside the model's dedicated Python environment.
    GUI types are transferred as metadata and reconstructed on the client.
    """
    # Runs with the Cell-ACDC of the model environment, which may still
    # have the single-module myutils.py of older versions
    script = f'''
import json, sys
import cellacdc.myutils
if hasattr(cellacdc.myutils, "__path__"):
    from cellacdc.myutils import model_modules
else:
    from cellacdc import myutils_models as model_modules
try:
    importer = (
        model_modules.import_promptable_segment_module
        if "{model_kind}" == "promptable"
        else model_modules.import_segment_module
    )
    acdcSegment = importer("{model_name}")
    init_params, segment_params = model_modules.getModelArgSpec(acdcSegment)
    url = getattr(acdcSegment, "url_help", lambda: None)()
    def _serialize_p(p):
        return [p.name, p.default, model_modules._serialize_model_param_type(p.type), p.desc, p.docstring]
    res = {{
        "init_params": [_serialize_p(p) for p in init_params],
        "segment_params": [_serialize_p(p) for p in segment_params],
        "url": url
    }}
    print("__ACDC_SPEC__" + json.dumps(res))
except Exception as e:
    import traceback
    traceback.print_exc()
    sys.exit(1)
'''
    proc = subprocess.run(
        [python_exec, "-c", script, "--server"],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Failed to query model spec for {model_name} from {python_exec}:\n{proc.stderr}"
        )

    for line in proc.stdout.splitlines():
        if line.startswith("__ACDC_SPEC__"):
            data = json.loads(line[len("__ACDC_SPEC__"):])
            init_params = [
                ArgSpec(*p)._replace(type=_deserialize_model_param_type(p[2]))
                for p in data["init_params"]
            ]
            segment_params = [
                ArgSpec(*p)._replace(type=_deserialize_model_param_type(p[2]))
                for p in data["segment_params"]
            ]
            url = data.get("url")
            return init_params, segment_params, url
    raise RuntimeError(f"No spec returned by model worker:\n{proc.stdout}")

def _get_doc_stop_idx(docstring, start_idx, next_param_name=None, debug=False):
    return myutils_models._get_doc_stop_idx(
        docstring, start_idx, next_param_name=next_param_name, debug=debug
    )   

def parse_model_param_doc(name, next_param_name=None, docstring=None):
    return myutils_models.parse_model_param_doc(
        name, next_param_name=next_param_name, docstring=docstring
        )

def add_segm_data_param(init_params, init_argspecs):
    return myutils_models.add_segm_data_param(init_params, init_argspecs)

def params_to_ArgSpec(
        fullargspecs, type_hints, docstring, args_to_skip=None
    ):
    return myutils_models.params_to_ArgSpec(
        fullargspecs, type_hints, docstring, args_to_skip=args_to_skip
    )


def getClassArgSpecs(classModule, runMethodName='run'):
    init_ArgSpec = inspect.getfullargspec(classModule.__init__)
    init_kwargs_type_hints = typing.get_type_hints(
        classModule.__init__
    )
    init_doc = classModule.__init__.__doc__
    init_params = params_to_ArgSpec(
        init_ArgSpec, init_kwargs_type_hints, init_doc
    )
    
    run_ArgSpec = inspect.getfullargspec(getattr(classModule, runMethodName))
    run_kwargs_type_hints = typing.get_type_hints(
        getattr(classModule, runMethodName)
    )
    run_doc = getattr(classModule, runMethodName).__doc__
    run_params = params_to_ArgSpec(
        run_ArgSpec, run_kwargs_type_hints, run_doc,
        args_to_skip={'signals', 'export_to'}
    )
    return init_params, run_params

def getTrackerArgSpec(trackerModule, realTime=False):
    init_ArgSpec = inspect.getfullargspec(trackerModule.tracker.__init__)
    init_kwargs_type_hints = typing.get_type_hints(
        trackerModule.tracker.__init__
    )
    init_doc = trackerModule.tracker.__init__.__doc__
    init_params = params_to_ArgSpec(
        init_ArgSpec, init_kwargs_type_hints, init_doc
    )
    if realTime:
        track_ArgSpec = inspect.getfullargspec(trackerModule.tracker.track_frame)
        track_kwargs_type_hints = typing.get_type_hints(
            trackerModule.tracker.track_frame
        )
        track_doc = trackerModule.tracker.track_frame.__doc__
    else:
        track_ArgSpec = inspect.getfullargspec(trackerModule.tracker.track)
        track_kwargs_type_hints = typing.get_type_hints(
            trackerModule.tracker.track
        )
        track_doc = trackerModule.tracker.track.__doc__

    track_params = params_to_ArgSpec(
        track_ArgSpec, track_kwargs_type_hints, track_doc,
        args_to_skip={'signals', 'export_to'}
    )
    return init_params, track_params

def isIntensityImgRequiredForTracker(trackerModule):
    track_ArgSpec = inspect.getfullargspec(trackerModule.tracker.track)
    num_args = len(track_ArgSpec.args) - len(track_ArgSpec.defaults)
    # If the number of args is 3 then we have `self, labels, image` as args 
    # which means the tracker requires the image 
    return num_args == 3                          


def aliases_real_time_trackers(reverse=False):
    """
    Returns a dictionary with aliases for real-time trackers.
    """

    aliases = {
            'CellACDC_normal_division': 'Cell-ACDC symmetric division',
            'CellACDC_2steps' : 'Cell-ACDC 2 steps',
        }
    
    if reverse:
        aliases = {v: k for k, v in aliases.items()}
    
    return aliases

def get_list_of_real_time_trackers():
    trackers = get_list_of_trackers()
    rt_trackers = []
    aliases = aliases_real_time_trackers()
    for tracker in trackers:
        if tracker == 'CellACDC':
            continue
        if tracker == 'YeaZ':
            continue
        tracker_filename = f'{tracker}_tracker.py'
        tracker_path = os.path.join(
            cellacdc_path, 'trackers', tracker, tracker_filename
        )
        try:
            with open(tracker_path) as file:
                txt = file.read()
            if txt.find('def track_frame') != -1:
                rt_trackers.append(tracker)
        except Exception as e:
            continue
    
    for i, tracker in enumerate(rt_trackers):
        if tracker in aliases:
            rt_trackers[i] = aliases[tracker]

    return natsorted(rt_trackers, key=str.casefold)

def get_list_of_trackers():
    trackers_path = os.path.join(cellacdc_path, 'trackers')
    trackers = []
    for name in filesystem.listdir(trackers_path):
        _path = os.path.join(trackers_path, name)
        tracker_script_path = os.path.join(_path, f'{name}_tracker.py')
        is_valid_tracker = (
            os.path.isdir(_path) and os.path.exists(tracker_script_path)
            and not name.endswith('__')
        )

        if name.startswith('_'):
            continue
        
        if is_valid_tracker:
            trackers.append(name)
    return natsorted(trackers, key=str.casefold)

def get_list_of_models():
    models = set()
    for name in filesystem.listdir(models_path):
        _path = os.path.join(models_path, name)
        if not os.path.exists(_path):
            continue
        
        if not os.path.isdir(_path):
            continue
        
        if name.endswith('__'):
            continue
            
        if name.startswith('_'):
            continue
        
        if name == 'skip_segmentation':
            continue
        
        if not os.path.exists(os.path.join(_path, 'acdcSegment.py')):
            continue
        
        if name == 'thresholding':
            name = 'Automatic thresholding'
        
        models.add(name)
    
    if not os.path.exists(models_list_file_path):
        return natsorted(list(models), key=str.casefold)
    
    cp = config.ConfigParser()
    cp.read(models_list_file_path)
    models.update(cp.sections())
    return natsorted(list(models), key=str.casefold)

def get_list_of_promptable_models():
    models = set()
    for name in filesystem.listdir(promptable_models_path):
        _path = os.path.join(promptable_models_path, name)
        if not os.path.exists(_path):
            continue
        
        if not os.path.isdir(_path):
            continue
        
        if name.endswith('__'):
            continue
        
        if not os.path.exists(os.path.join(_path, 'acdcPromptSegment.py')):
            continue
        
        models.add(name)
    
    if not os.path.exists(promptable_models_list_file_path):
        return natsorted(list(models), key=str.casefold)
    
    cp = config.ConfigParser()
    cp.read(promptable_models_list_file_path)
    models.update(cp.sections())
    return natsorted(list(models), key=str.casefold)


def import_tracker_module(tracker_name):
    module_name =  f'cellacdc.trackers.{tracker_name}.{tracker_name}_tracker'
    tracker_module = import_module(module_name)
    return tracker_module


def import_promptable_segment_module(model_name):
    return myutils_models.import_promptable_segment_module(model_name)


def init_tracker(
        posData, trackerName, realTime=False, qparent=None, 
        return_init_params=False
    ):
    from cellacdc.shared.dialogs import (
        model_params as dialogs_model_params,
        model_selection as dialogs_model_selection,
        tracker_params as dialogs_tracker_params,
    )
    downloadWin = dialogs_model_selection.downloadModel(trackerName, parent=qparent)
    downloadWin.download()

    trackerModule = import_tracker_module(trackerName)
    init_params = {}
    track_params = {}
    paramsWin = None
    if trackerName == 'BayesianTracker':
        Y, X = posData.img_data_shape[-2:]
        if posData.isSegm3D:
            labShape = (posData.SizeZ, Y, X)
        else:
            labShape = (1, Y, X)
        paramsWin = dialogs_tracker_params.BayesianTrackerParamsWin(
            labShape, parent=qparent, channels=posData.chNames, 
            currentChannelName=posData.user_ch_name
        )
        paramsWin.exec_()
        if not paramsWin.cancel:
            init_params = paramsWin.params
            track_params['export_to'] = posData.get_btrack_export_path()
            if paramsWin.intensityImageChannel is not None:
                chName = paramsWin.intensityImageChannel
                track_params['image'] = posData.loadChannelData(chName)
                track_params['image_channel_name'] = chName
    elif trackerName == 'CellACDC':
        paramsWin = dialogs_tracker_params.CellACDCTrackerParamsWin(parent=qparent)
        paramsWin.exec_()
        if not paramsWin.cancel:
            init_params = paramsWin.params
    elif trackerName == 'delta':
        paramsWin = dialogs_tracker_params.DeltaTrackerParamsWin(posData=posData, parent=qparent)
        paramsWin.exec_()
        if not paramsWin.cancel:
            init_params = paramsWin.params
    else:
        init_argspecs, track_argspecs = getTrackerArgSpec(
            trackerModule, realTime=realTime
        )
        intensityImgRequiredForTracker = isIntensityImgRequiredForTracker(
            trackerModule
        )
        if init_argspecs or track_argspecs:
            try:
                url = trackerModule.url_help()
            except AttributeError:
                url = None
            try:
                channels = posData.chNames
            except Exception as e:
                channels = None
            try:
                currentChannelName = posData.user_ch_name
            except Exception as e:
                currentChannelName = None
            try:
                df_metadata = posData.metadata_df
            except Exception as e:
                df_metadata = None
            
            if not intensityImgRequiredForTracker:
                currentChannelName = None
            
            paramsWin = dialogs_model_params.QDialogModelParams(
                init_argspecs, track_argspecs, trackerName, url=url,
                channels=channels, is_tracker=True,
                currentChannelName=currentChannelName,
                df_metadata=df_metadata, posData=posData
            )
            if not intensityImgRequiredForTracker and channels is not None:
                paramsWin.channelCombobox.setDisabled(True)
                
            paramsWin.exec_()
            if not paramsWin.cancel:
                init_params = paramsWin.init_kwargs
                track_params = paramsWin.model_kwargs
                if paramsWin.inputChannelName != 'None':
                    chName = paramsWin.inputChannelName
                    track_params['image'] = posData.loadChannelData(chName)
                    track_params['image_channel_name'] = chName
        if 'export_to_extension' in track_params:
            ext = track_params['export_to_extension']
            track_params['export_to'] = posData.get_tracker_export_path(
                trackerName, ext
            )

    if paramsWin is not None and paramsWin.cancel:
        tracker = None, 
        track_params = None
        init_params = None
    else:
        tracker = trackerModule.tracker(**init_params)
        
    if return_init_params:
        return tracker, track_params, init_params
    else:
        return tracker, track_params

def import_segment_module(model_name, parent=None):
    return myutils_models.import_segment_module(model_name, parent=parent)


def init_prompt_segm_model(acdcPromptSegment, posData, init_kwargs):
    model = acdcPromptSegment.Model(**init_kwargs)
    return model

def resolve_segm_model_init_data(posData, init_kwargs):
    """Replace GUI file selections in `init_kwargs` with the data itself.

    Pops `segm_endname` (returned as loaded `segm_data`, or None) and replaces
    `input_points_path` with `input_points_df`. Modifies `init_kwargs` in place.
    """
    segm_endname = init_kwargs.pop('segm_endname', 'None')
    if segm_endname != 'None':
        load_segm = True
        if not hasattr(posData, 'segm_data'):
            load_segm = True
        elif posData.segm_npz_path.endswith(f'{segm_endname}.npz'):
            load_segm = False
        if not load_segm:
            segm_data = np.squeeze(posData.segm_data)
        else:
            segm_filepath, _ = load.get_path_from_endname(
                segm_endname, posData.images_path
            )
            printl(f'Loading segmentation data from "{segm_filepath}"...')
            segm_data = np.load(segm_filepath)['arr_0']
    else:
        segm_data = None

    # Initialize input_points_df for models promptable with points
    input_points_filepath = init_kwargs.pop('input_points_path', '')
    if input_points_filepath:
        input_points_df = init_input_points_df(
            posData, input_points_filepath
        )
        init_kwargs['input_points_df'] = input_points_df
    return segm_data

def model_init_requires_segm_data(acdcSegment):
    return myutils_models.model_init_requires_segm_data(acdcSegment)

def init_segm_model(acdcSegment, posData, init_kwargs):
    segm_data = resolve_segm_model_init_data(posData, init_kwargs)
    
    try:
        # Models introduced before 1.3.2 do not have the segm_data as input
        kwargs = inspect.getfullargspec(acdcSegment.Model.__init__).args
        if 'is_rgb' not in kwargs and 'is_rgb' in init_kwargs:
            del init_kwargs['is_rgb']
        model = acdcSegment.Model(**init_kwargs)


    except Exception as e:
        model = acdcSegment.Model(segm_data, **init_kwargs)
        
    if hasattr(model, 'init_successful'):
        if not model.init_successful:
            return None
    return model

def _parse_bool_str(value):
    if isinstance(value, bool):
        return value
    
    if value == 'True':
        return True
    elif value == 'False':
        return False


def parse_model_params(model_argspecs, model_params):
    parsed_model_params = {}
    for row, argspec in enumerate(model_argspecs):
        if argspec.name not in model_params:
            # Missing parameter --> the model uses its default value
            continue
        value = model_params[argspec.name]
        if value is None:
            # Keep explicit None values (e.g., `model_type = None` in a
            # workflow config file). Dropping them would replace them with
            # the default value, which can be invalid (e.g., cellpose
            # `model_type='cpsam'` together with a custom `model_path`)
            parsed_model_params[argspec.name] = None
            continue
        if argspec.type == bool:
            value = _parse_bool_str(value)
        elif argspec.type == int:
            value = int(value)
        elif argspec.type == float:
            value = float(value)
        parsed_model_params[argspec.name] = value
    return parsed_model_params

# def init_cellpose_denoise_model():
#     from . import apps
    
#     from cellacdc.models.cellpose_v3._denoise import (
#         CellposeDenoiseModel, url_help
#     )

#     init_argspecs, run_argspecs = getClassArgSpecs(CellposeDenoiseModel)
#     url = url_help()
    
#     paramsWin = apps.QDialogModelParams(
#         init_argspecs, run_argspecs, 'Cellpose 3.0', 
#         url=url, is_tracker=True, action_type='denoising'
#     )
#     paramsWin.exec_()
#     if paramsWin.cancel:
#         return
    
#     init_params = paramsWin.init_kwargs
#     run_params = paramsWin.model_kwargs
#     denoise_model = CellposeDenoiseModel(**init_params)
#     return denoise_model, init_params, run_params

def init_input_points_df(posData, input_points_filepath):
    input_points_df = None
    if os.path.exists(input_points_filepath):
        input_points_df = pd.read_csv(input_points_filepath)
    else:
        # input_points_filepath is actually and endname
        for file in filesystem.listdir(posData.images_path):
            if file.endswith(input_points_filepath):
                filepath = os.path.join(posData.images_path, file)
                input_points_df = pd.read_csv(filepath)
                break
    
    if input_points_df is None:
        raise FileNotFoundError(
            f'Could not find input points table from file "input_points_filepath" '
            'Perhaps, you forgot to save the table?'
        )
    
    for col in ('x', 'y', 'id'):
        if col not in input_points_df.columns:
            raise KeyError(
                f'Input points table is missing colum {col}. It must have '
                'the colums (x, y, id)'
            )
    
    return input_points_df


def log_segm_params(
        model_name, init_params, segm_params, logger_func=print, 
        preproc_recipe=None, apply_post_process=False, 
        standard_postprocess_kwargs=None, custom_postprocess_features=None
    ):
    init_params_format = [
        f'  * {option} = {value}' for option, value in init_params.items()
    ]
    init_params_format = '\n'.join(init_params_format)
    
    segm_params_format = [
        f'  * {option} = {value}' for option, value in segm_params.items()
    ]
    segm_params_format = '\n'.join(segm_params_format)
    
    preproc_recipe_format = None
    if preproc_recipe is not None:
        preproc_recipe_format = []
        for s, step in enumerate(preproc_recipe):
            preproc_recipe_format.append(f'  * Step {s+1}')
            method = step['method']
            preproc_recipe_format.append(f'     - Method: {method}')
            for option, value in step['kwargs'].items():
                preproc_recipe_format.append(f'     - {option}: {value}')
        preproc_recipe_format = '\n'.join(preproc_recipe_format)    
    
    standard_postproc_format = None
    if apply_post_process and standard_postprocess_kwargs is not None:
        standard_postproc_format = [
            f'  * {option} = {value}' 
            for option, value in standard_postprocess_kwargs.items()
        ]
        standard_postproc_format = '\n'.join(standard_postproc_format)
    
    custom_postproc_format = None
    if apply_post_process and custom_postprocess_features is not None:
        custom_postproc_format = [
            f'  * {feature} = ({low}, {high})'
            for feature, (low, high) in custom_postprocess_features.items()
        ]
        custom_postproc_format = '\n'.join(custom_postproc_format)
    
    separator = '-'*100
    params_format = (
        f'{separator}\n'
        f'Model name: {model_name}\n\n'
        'Preprocessing recipe:\n\n'
        f'{preproc_recipe_format}\n\n'
        'Initialization parameters:\n\n'
        f'{init_params_format}\n\n'
        'Segmentation parameters:\n\n'
        f'{segm_params_format}\n\n'
        'Post-processing:\n\n'
        f'{standard_postproc_format}\n\n'
        'Custom post-processing:\n\n'
        f'{custom_postproc_format}\n'
        f'{separator}'
    )
    logger_func(params_format)
