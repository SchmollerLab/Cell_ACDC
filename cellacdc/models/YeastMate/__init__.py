import os
import sys
import subprocess

try:
    import detectron2
except ModuleNotFoundError:
    if sys.platform.startswith("win"):
        from qtpy.QtWidgets import QMessageBox, QApplication
        from qtpy.QtCore import QCoreApplication
        from cellacdc.apps import warnVisualCppRequired

        if QCoreApplication.instance() is None:
            app = QApplication(sys.argv)

        win = warnVisualCppRequired(pkg_name='YeastMate')
        win.exec_()

        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'Cython']
        )
        # subprocess.check_call(
        #     [sys.executable, '-m', 'pip', 'install',
        #     'git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI']
        # )
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install',
        'git+https://github.com/facebookresearch/detectron2.git@v0.5']
    )

try:
    import yeastmatedetector
except ModuleNotFoundError:
    from qtpy.QtWidgets import QMessageBox, QApplication
    from qtpy.QtCore import QCoreApplication

    if QCoreApplication.instance() is None:
        app = QApplication(sys.argv)

    from cellacdc import myutils
    cancel = myutils._install_package_msg('YeastMate')
    if cancel:
        raise ModuleNotFoundError(
            'User aborted YeastMate installation'
        )

    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install',
        'git+https://github.com/hoerlteam/YeastMate.git']
    )
    # YeastMate installs opencv-python which is not functional with PyQt5 on macOS.
    # Uninstall it, and reinstall opencv-python-headless
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', 'opencv-python']
    )
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'uninstall', '-y', 'opencv-python-headless']
    )
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'opencv-python-headless']
    )
