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
    try:
        not_a_param = ArgSpec.type().not_a_param
        return True
    except Exception as err:
        pass
    
    try:
        # If a parameter if None, python initializes it to 
        # typing.Optional and we need to access the first type
        ArgSpec.type.__args__[0]().not_a_param
        return True
    except Exception as err:
        pass
    
    return False

def to_str(*args):
    if len(args) == 2:
        value = args[1]
    else:
        value = args[0]
    
    return str(value)