import os
import sys
import subprocess

try:
    import cellpose
except ModuleNotFoundError:
    from PyQt5.QtWidgets import QMessageBox, QApplication
    from PyQt5.QtCore import QCoreApplication

    if QCoreApplication.instance() is None:
        app = QApplication(sys.argv)

    from cellacdc import myutils
    cancel = myutils.install_package_msg('cellpose')
    if cancel:
        raise ModuleNotFoundError(
            'User aborted cellpose installation'
        )

    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'cellpose']
    )
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'omnipose']
    )
