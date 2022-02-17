try:
    import btrack
except ModuleNotFoundError:
    pkg_name = 'BayesianTracker'
    import os
    import sys
    import subprocess
    from PyQt5.QtWidgets import QMessageBox
    from cellacdc import myutils
    cancel = myutils.install_package_msg(pkg_name, parent=self)
    if cancel:
        raise ModuleNotFoundError(
            f'User aborted {pkg_name} installation'
        )
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'btrack']
    )
