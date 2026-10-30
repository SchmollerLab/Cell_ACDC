

from collections import namedtuple
from importlib import import_module
import inspect
import re
import typing

from . import config, promptable_models_list_file_path, models_list_file_path
from . import cca_df_colnames, printl, base_cca_dict, base_cca_tree_dict
import pandas as pd
import importlib.util
import sys
import os
ArgSpec = namedtuple('ArgSpec', ['name', 'default', 'type', 'desc', 'docstring'])
def import_promptable_segment_module(model_name):
    try:
        acdcPromptSegment = import_module(
            f'cellacdc.promptable_models.{model_name}.acdcPromptSegment'
        )
    except ModuleNotFoundError as e:
        # Check if custom model
        cp = config.ConfigParser()
        cp.read(promptable_models_list_file_path)
        model_path = cp[model_name]['path']
        spec = importlib.util.spec_from_file_location(
            'acdcPromptSegment', model_path
        )
        acdcPromptSegment = importlib.util.module_from_spec(spec)
        sys.modules['acdcPromptSegment'] = acdcPromptSegment
        spec.loader.exec_module(acdcPromptSegment)
    return acdcPromptSegment

def import_segment_module(model_name, parent=None):
    original_model_name = model_name
    if model_name == 'Automatic thresholding':
        model_name = 'thresholding'

    try:
        acdcSegment = import_module(f'cellacdc.models.{model_name}.acdcSegment')
    except ModuleNotFoundError as e:
        # Do not mask missing dependencies imported by the module itself.
        expected_missing_module = f'cellacdc.models.{model_name}'
        if e.name != expected_missing_module:
            raise

        # Check if custom model
        cp = config.ConfigParser()
        cp.read(models_list_file_path)
        model_key = None
        for key in (original_model_name, model_name):
            if key in cp:
                model_key = key
                break

        if model_key is None:
            raise

        model_path = cp[model_key]['path']
        spec = importlib.util.spec_from_file_location('acdcSegment', model_path)
        acdcSegment = importlib.util.module_from_spec(spec)
        sys.modules['acdcSegment'] = acdcSegment
        spec.loader.exec_module(acdcSegment)
    return acdcSegment

def getModelArgSpec(acdcSegment):
    init_ArgSpec = inspect.getfullargspec(acdcSegment.Model.__init__)
    init_kwargs_type_hints = typing.get_type_hints(acdcSegment.Model.__init__)
    init_doc = acdcSegment.Model.__init__.__doc__
    init_params = params_to_ArgSpec(
        init_ArgSpec, init_kwargs_type_hints, init_doc
    )
    init_params = add_segm_data_param(init_params, init_ArgSpec)
    
    segment_ArgSpec = inspect.getfullargspec(acdcSegment.Model.segment)
    segment_kwargs_type_hints = typing.get_type_hints(acdcSegment.Model.segment)
    try:
        segment_ArgSpec.args.remove('frame_i')
    except Exception as e:
        pass
    
    segment_doc = acdcSegment.Model.segment.__doc__
    segment_params = params_to_ArgSpec(
        segment_ArgSpec, segment_kwargs_type_hints, segment_doc,
    )
    
    return init_params, segment_params

def model_init_requires_segm_data(acdcSegment):
    argspec = inspect.getfullargspec(acdcSegment.Model.__init__)
    num_kwargs = len(argspec.defaults or ())
    return len(argspec.args) - num_kwargs > 1


def _serialize_model_param_type(param_type):
    """Transfer GUI type metadata without importing model dependencies on the client side.
    Used in insertModelArgSpec and testing"""
    from . import _types

    if _types.is_optional(param_type):
        param_type = next(t for t in typing.get_args(param_type) if t is not type(None))
    for name, cls in (
            ('bool', bool), ('int', int), ('float', float), ('str', str),
            ('path', os.PathLike), ('none', type(None)),
            ('vector', _types.Vector), ('folder', _types.FolderPath),
            ('second_channel', _types.SecondChannelImage),
            ('hidden', _types.NotGUIParam), ('yeaz_model', _types.YeazV2ModelType),
            ('csv_path', _types.CsvFilePath)
        ):
        if param_type is cls:
            return {'kind': name}
    if getattr(param_type, 'not_a_param', False):
        return {'kind': 'hidden'}
    if typing.get_origin(param_type) is typing.Literal:
        return {'kind': 'choices', 'values': list(typing.get_args(param_type))}
    if hasattr(param_type, 'values'):
        return {
            'kind': 'choices',
            'values': list(param_type.values),
            'is_exclusive_with': getattr(param_type, 'is_exclusive_with', []),
            'default_exclusive': getattr(param_type, 'default_exclusive', ''),
        }
    if getattr(param_type, 'isWidget', False):
        raise TypeError(f'No client-side widget registered for {param_type}')
    # Other annotations (e.g. torch.device unions) use the existing text editor.
    return {'kind': 'str'}


def params_to_ArgSpec(
        fullargspecs, type_hints, docstring, args_to_skip=None
    ):
    params = []
    
    if fullargspecs.defaults is None:
        return params
    
    if args_to_skip is None:
        args_to_skip = set()
    
    num_params = len(fullargspecs.args)
    ip = num_params - len(fullargspecs.defaults)
    if ip < 0:
        return params
    
    for arg, default in zip(fullargspecs.args[ip:], fullargspecs.defaults):
        if arg in args_to_skip:
            continue
        
        if arg in type_hints:
            _type = type_hints[arg]
        else:
            _type = type(default)
        
        next_param_name = None
        if ip+1 < num_params:
            next_param_name = fullargspecs.args[ip+1]
        
        param_doc = parse_model_param_doc(
            arg, 
            next_param_name=next_param_name,
            docstring=docstring
        )
        param = ArgSpec(
            name=arg, 
            default=default, 
            type=_type, 
            desc=param_doc,
            docstring=docstring
        )
        params.append(param)
        ip += 1
    return params

def parse_model_param_doc(name, next_param_name=None, docstring=None):
    if not docstring:
        return ''
    
    try:
        # Extract parameter description from 'param : ...'
        start_text = f'{name} : '
        if docstring.find(start_text) == -1:
            # Parameter not present in docstring
            return ''
        
        doc_start_idx = docstring.find(start_text) + len(start_text)
        
        doc_stop_idx = _get_doc_stop_idx(
            docstring, doc_start_idx, next_param_name=next_param_name
        )
        if doc_stop_idx == -1:
            doc_stop_idx = len(docstring)
        
        param_doc = docstring[doc_start_idx:doc_stop_idx]
        
        # Start at first end of line
        param_doc = param_doc[param_doc.find('\n')+1:]
        
        # Replace multiples spaces with single space
        param_doc = re.sub(' +', ' ', param_doc)
        
        # Remove trailing spaces
        param_doc = param_doc.strip()
    except Exception as err:
        param_doc = ''
    
    param_doc = param_doc.replace(', optional', '')
    
    return param_doc

def _get_doc_stop_idx(docstring, start_idx, next_param_name=None, debug=False):
    if debug:
        import pdb; pdb.set_trace()
    
    if next_param_name is not None:
        doc_stop_idx = docstring.find(f'{next_param_name} : ')
        if doc_stop_idx > 1:
            return doc_stop_idx
    
    docstring_from_start = docstring[start_idx:]
    next_param_searched = re.search(r'\w+ : ', docstring_from_start)
    if next_param_searched is not None:
        return next_param_searched.start(0) + start_idx
    
    doc_stop_idx = docstring.find('Returns')
    if doc_stop_idx > 1:
        return doc_stop_idx
    
    doc_stop_idx = docstring.find('Notes')
    if doc_stop_idx > 1:
        return doc_stop_idx 
    
    return -1

def add_segm_data_param(init_params, init_argspecs):
    if init_argspecs.defaults is None:
        num_kwargs = 0
    else:
        num_kwargs = len(init_argspecs.defaults)
    
    # Segm model requires segm data --> add it to params
    num_args = len(init_argspecs.args) - num_kwargs
    if num_args == 1:
        # Args is only self --> segm data not needed
        return init_params
    
    desc = (
'This model requires an additional segmentation file as input.\n\n'
'Please, select which segmentation file to provide to the model.'
    )
    
    segm_data_argspec = ArgSpec(
        name='Auxiliary segmentation file', 
        default='', 
        type=str, 
        desc=desc,
        docstring=None
    )
    
    init_params.insert(0, segm_data_argspec)
    return init_params


def getBaseCca_df(IDs, with_tree_cols=False):
    row_data = base_cca_dict
    if with_tree_cols:
        row_data = {**base_cca_dict, **base_cca_tree_dict}
    data = [row_data]*len(IDs)
    cca_df = pd.DataFrame(data, index=IDs)    
    if with_tree_cols:
       cca_df['Cell_ID_tree'] = IDs
    cca_df.index.name = 'Cell_ID'
    return cca_df