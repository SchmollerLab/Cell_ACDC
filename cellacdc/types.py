import typing

from typing import Union, Tuple, Any

class Vector:
    """Class used to define model parameter as a vector that will use the 
    cellacdc.widgets.VectorLineEdit widget in the automatic GUI.
    """
    def __init__(self):
        return 
    
    def __call__(self, value: Any) -> Tuple[float]:
        if isinstance(value, str):
            values = value.split(',')
            values = tuple([float(val) for val in values])
        elif isinstance(value, int):
            return (value,)
        elif isinstance(value, float):
            return (value,)
        
        raise TypeError(f'Could not convert {value} to Vector')
        
class FolderPath:
    """Class used to define model parameter as a folder path control with a 
    browse button to select a folder in the automatic GUI.
    """
    def __init__(self):
        return
    
    def __call__(self, value: Any) -> Tuple[float]:
        return str(value)

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
    
    return field.__name__ == 'SecondChannelImage'

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