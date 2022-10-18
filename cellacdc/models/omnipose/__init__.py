import os
import sys
import subprocess

try:
    import cellpose_omni
except ModuleNotFoundError:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QCoreApplication

    if QCoreApplication.instance() is None:
        app = QApplication(sys.argv)

    from cellacdc import myutils
    cancel = myutils.install_package_msg('omnipose_acdc')
    if cancel:
        raise ModuleNotFoundError(
            'User aborted cellpose installation'
        )

    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'omnipose_acdc']
    )
