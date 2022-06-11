from PyQt5.QtCore import QObject, pyqtSignal, qInstallMessageHandler
import argparse
import configparser

class ConfigParser(configparser.ConfigParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.optionxform = str

class QtWarningHandler(QObject):
    sigGeometryWarning = pyqtSignal(object)

    def _resizeWarningHandler(self, msg_type, msg_log_context, msg_string):
        if msg_string.find('Unable to set geometry') != -1:
            self.sigGeometryWarning.emit(msg_type)
        elif msg_string:
            print(msg_string)

warningHandler = QtWarningHandler()
qInstallMessageHandler(warningHandler._resizeWarningHandler)

ap = argparse.ArgumentParser(description='spotMAX inputs')
ap.add_argument(
    '-d', '--debug', action='store_true',
    help=(
        'Used for debugging. Test code with'
        '"from cellacdc.config import parser_args, debug = parser_args["debug"]", '
        'if debug: <debug code here>'
    )
)

# Add dummy argument for stupid Jupyter
ap.add_argument('-f')


parser_args = vars(ap.parse_args())
