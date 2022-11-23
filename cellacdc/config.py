from PyQt5.QtCore import QObject, pyqtSignal, qInstallMessageHandler
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

class QtWarningHandler(QObject):
    sigGeometryWarning = pyqtSignal(object)

    def _resizeWarningHandler(self, msg_type, msg_log_context, msg_string):
        if msg_string.find('Unable to set geometry') != -1:
            try:
                self.sigGeometryWarning.emit(msg_string)
            except:
                pass
        elif msg_string:
            print(msg_string)

warningHandler = QtWarningHandler()
qInstallMessageHandler(warningHandler._resizeWarningHandler)

try:
    ap = argparse.ArgumentParser(description='Cell-ACDC parser')
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
