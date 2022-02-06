import os
import sys
import subprocess

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
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install',
        'git+https://github.com/hoerlteam/YeastMate']
    )
    # YeastMate installs opencv-python which is not functional with PyQt5 on macOS.
    # Uninstall it, and reinstall opencv-python-headless
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'uninstall', 'opencv-python']
    )
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'opencv-python-headless']
    )
