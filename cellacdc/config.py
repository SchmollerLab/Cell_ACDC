import argparse
import configparser
import pprint

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
except:
    print('Importing from notebook, ignoring Cell-ACDC argument parser...')
    parser_args = {}
    parser_args['debug'] = False
