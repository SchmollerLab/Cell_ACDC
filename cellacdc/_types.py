import typing

from typing import Union, Tuple, Any, List
import numpy as np

class NotGUIParam:
    not_a_param = True

ChannelsDict = dict[str, List[np.ndarray]]

class RescaleIntensitiesInRangeHow:
    values = ['percentage', 'image', 'absolute']

class BaSiCpyResizeModes:
    values = ['jax', 'skimage', 'skimage_dask']

class BaSiCpyFittingModes:
    values = ['ladmap', 'approximate']

class BaSiCpyTimelapse:
    values = ["True", "False", "additive", "multiplicative"]

class Vector:
    """Class used to define model parameter as a vector that will use the 
    cellacdc.widgets.VectorLineEdit widget in the automatic GUI.
    """
    @staticmethod
    def cast_dtype(value: Any) -> Union[Tuple[float], int, float]:
        if isinstance(value, str):
            value = value.lstrip('(').rstrip(')')
            value = value.lstrip('[').rstrip(']')
            values = value.split(',')
            values = tuple([float(val) for val in values])
            return values
        elif isinstance(value, (int, float)):
            return value
        
        raise TypeError(f'Could not convert {value} {(type(value))} to Vector')
        
    def __call__(self, value: Any) -> Union[Tuple[float], int, float]:
        return self.cast_dtype(value)
        
class FolderPath:
    """Class used to define model parameter as a folder path control with a 
    browse button to select a folder in the automatic GUI.
    """
    def cast_dtype(self, value: Any) -> Union[Tuple[float], int, float]:
        return str(value)
    
    def __call__(self, value: Any) -> str:
        return self.cast_dtype(value)

class SecondChannelImage:
    pass

class YeazV2ModelType:
    isWidget = True

    def __init__(self):
        from cellacdc import widgets
        self.widget = widgets.YeazV2SelectModelNameCombobox(
            custom_select_item_text='Select custom weights file...'
        )

class CsvFilePath:
    """Model parameter shown as a CSV file selector in the automatic GUI.

    The widget is created lazily so models can use this annotation without
    importing Qt (e.g. inside a model worker environment).
    """
    isWidget = True

    def __init__(self):
        from cellacdc import widgets
        self.widget = widgets.CsvFilePathControl()

    @staticmethod
    def cast_dtype(value: Any) -> str:
        return str(value)

def is_optional(field):
    return (
        typing.get_origin(field) is Union and 
        type(None) in typing.get_args(field)
    )

def is_second_channel_type(field):
    if is_optional(field):
        field = typing.get_args(field)[0]
    
    return getattr(field, '__name__', None) == 'SecondChannelImage' # avoid union

def is_widget_not_required(ArgSpec):
    field = ArgSpec.type
    if hasattr(field, 'not_a_param'):
        return True
    args = typing.get_args(field)
    return bool(args) and hasattr(args[0], 'not_a_param')

def to_str(*args):
    if len(args) == 2:
        value = args[1]
    else:
        value = args[0]
    
    return str(value)