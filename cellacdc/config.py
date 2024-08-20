import argparse
import configparser
import pprint
import os

from typing import get_type_hints 

import re

from . import printl, debug_true_filepath

class ConfigParser(configparser.ConfigParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optionxform = str
    
    def __repr__(self) -> str:
        string = pprint.pformat(
            {section: dict(self[section]) for section in self.sections()}
        )
        return string

from . import GUI_INSTALLED

if GUI_INSTALLED:
    from qtpy.QtCore import QObject, Signal, qInstallMessageHandler

    class QtWarningHandler(QObject):
        sigGeometryWarning = Signal(object)

        def _resizeWarningHandler(self, msg_type, msg_log_context, msg_string):
            if msg_string.find('Unable to set geometry') != -1:
                try:
                    self.sigGeometryWarning.emit(msg_string)
                except Exception as e:
                    pass
            elif msg_string:
                print(msg_string)

    warningHandler = QtWarningHandler()
    qInstallMessageHandler(warningHandler._resizeWarningHandler)

help_text = (
    'Welcome to Cell-ACDC!\n\n'
    'You can run Cell-ACDC both as a GUI or in the command line.\n'
    'To run the GUI type `acdc`. To run the command line type `acdc -p <path_to_params_file>`.\n'
    'The `<path_to_params_file>` must be a workflow INI file.\n'
    'If you do not have one, use the GUI to set up the parameters.\n\n'
    'Enjoy!'
)
try:
    ap = argparse.ArgumentParser(
        prog='Cell-ACDC', description=help_text, 
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    ap.add_argument(
        '-p', '--params',
        default='',
        type=str,
        metavar='PATH_TO_PARAMS',
        help=('Path of the ".ini" workflow file')
    )
    
    ap.add_argument(
        '-d', '--debug', action='store_true',
        help=(
            'Used for debugging. Test code with'
            '"from cellacdc.config import parser_args, debug = parser_args["debug"]", '
            'if debug: <debug code here>'
        )
    )

    # Add dummy argument for stupid Jupyter
    # ap.add_argument('-f')

    parser_args, unknown = ap.parse_known_args()
    parser_args = vars(parser_args)
    if os.path.exists(debug_true_filepath):
        parser_args['debug'] = True
except:
    print('Importing from notebook, ignoring Cell-ACDC argument parser...')
    parser_args = {}
    parser_args['debug'] = False

def preprocessing_mapper():
    from cellacdc import preprocess, types
    from inspect import getmembers, isfunction
    functions = getmembers(preprocess, isfunction)
    mapper = {}
    for func_name, func in functions:
        if func_name.startswith('_'):
            continue
        
        method = func_name.title().replace('_', ' ')
        mapper[method] = {
            'widgets': {}, 
            'function': func, 
            'docstring': func.__doc__, 
            'args': []
        }
        type_hints = get_type_hints(func)
        for param, type_hint in type_hints.items():
            # widget must be implemented in cellacdc.widgets module
            if type_hint == types.Vector:
                widget = 'VectorLineEdit'
            elif type_hint == float:
                widget = 'FloatLineEdit'
            elif type_hint == str:
                widget = 'LineEdit'
            elif type_hint == int:
                widget = 'IntLineEdit'
            
            mapper[method]['widgets'][param.capitalize()] = widget
            mapper[method]['args'].append(param)
    
    return mapper

def preprocess_recipe_to_ini_items(preproc_recipe):
    if preproc_recipe is None:
        return {}
    
    ini_items = {}
    for s, step in enumerate(preproc_recipe):
        section = f'preprocess.step{s+1}'
        ini_items[section] = {}
        ini_items[section]['method'] = step['method']
        for option, value in step['kwargs'].items():
            ini_items[section][option] = str(value)
    return ini_items

def preprocess_ini_items_to_recipe(ini_items):
    recipe = {}
    
    for section, section_items in ini_items.items():
        if not section.startswith('preprocess.step'):
            continue
        
        step_n = int(re.findall(r'step(\d+)', section)[0])
        recipe[step_n] = {'method': section_items['method']}
        kwargs = {}
        for option, value_str in section_items.items():
            if option == 'method':
                continue
            
            value = value_str
            if isinstance(value_str, str):
                for _type in (int, float, str):
                    try:
                        value = _type(value_str)
                        break
                    except Exception as e:
                        continue
            
            kwargs[option] = value
            
        recipe[step_n]['kwargs'] = kwargs
    
    recipe = [value for key, value in sorted(recipe.items())]
    
    if not recipe:
        return
    
    return recipe

PREPROCESS_MAPPER = preprocessing_mapper()