import os
import sys
import subprocess

from cellacdc.myutils import installs

installs.check_install_package('omnipose_acdc', import_pkg_name='omnipose')
