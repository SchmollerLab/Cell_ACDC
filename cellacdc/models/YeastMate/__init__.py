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

if sys.platform.startswith("win"):
    try:
        import detectron2
    except ModuleNotFoundError:
        from PyQt5.QtCore import Qt
        msg = QMessageBox()
        msg.setIcon(msg.Critical)
        msg.setWindowTitle('Installation of YeastMate failed')
        msg.setTextFormat(Qt.RichText)
        txt = ("""
        <p style=font-size:12px>
            Installation of YeastMate on Windows requires
            Microsoft Visual C++ 14.0 or higher.<br><br>
            Please <b>close Cell-ACDC</b>, then download and install
            <b>"Microsoft C++ Build Tools"</b> from the link below
            before trying YeastMate again.<br><br>
            <a href='https://visualstudio.microsoft.com/visual-cpp-build-tools/'>
                https://visualstudio.microsoft.com/visual-cpp-build-tools/
            </a>
        </p>
        """)
        msg.setText(txt)
        msg.exec_()
        raise ModuleNotFoundError(
            'Installation of module "detectron2" failed. '
            'Please try by installing "Microsoft C++ Build Tools" from '
            'this link: https://visualstudio.microsoft.com/visual-cpp-build-tools/'
        )
