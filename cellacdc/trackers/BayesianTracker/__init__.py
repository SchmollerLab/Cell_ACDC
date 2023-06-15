INSTALL_BTRACK = False

try:
    import btrack
    import pkg_resources

    version = pkg_resources.get_distribution("btrack").version
    minor = version.split('.')[1]
    if int(minor) < 5:
        INSTALL_BTRACK = True
except Exception as e:
    INSTALL_BTRACK = True

if INSTALL_BTRACK:
    pkg_name = 'BayesianTracker'
    import os
    import sys
    import subprocess
    from qtpy.QtWidgets import QMessageBox
    from cellacdc import myutils
    cancel = myutils._install_package_msg(pkg_name)
    if cancel:
        raise ModuleNotFoundError(
            f'User aborted {pkg_name} installation'
        )
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-U', 'btrack']
    )
