"""Fake myutils/installs.py: reads printl and subprocess as globals."""
import subprocess

from cellacdc import printl


def check_install_package(name):
    printl(name)
    return subprocess.run
