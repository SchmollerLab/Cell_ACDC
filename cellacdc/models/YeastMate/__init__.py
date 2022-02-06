import os
import sys
import subprocess

from PyQt5.QtWidgets import QMessageBox

try:
    import detectron2
except ModuleNotFoundError:
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install',
        'git+https://github.com/facebookresearch/detectron2.git@v0.5']
    )

try:
    import yeastmatedetector
except ModuleNotFoundError:
    msg = QMessageBox()
    txt = (
        'Cell-ACDC is going to download and install "YeastMate".\n\n'
        'Make sure you have an active internet connection, '
        'before continuing. '
        'Progress will be displayed on the terminal\n\n'
        'Alternatively, you can cancel the process and try later.'
    )
    answer = msg.information(
        None, 'Install YeastMate', txt, msg.Ok | msg.Cancel
    )
    if answer == msg.Cancel:
        raise ModuleNotFoundError(
            'User aborted YeastMate installation'
        )

    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install',
        'git+https://github.com/hoerlteam/YeastMate']
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
