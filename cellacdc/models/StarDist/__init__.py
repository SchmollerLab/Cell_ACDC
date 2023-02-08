import os
import sys
import subprocess

from cellacdc import myutils

note = ''
if sys.platform == 'darwin':
    note = ("""
    <p style="font-size:13px">
        <b>NOTE for M1 mac users</b>: if you are using one of the latest Mac with
        M1 Apple Silicon processor <b>cancel this operation</b> and follow the
        instructions you can find
        <a href="https://github.com/SchmollerLab/Cell_ACDC/issues/8">
            here.
        </a>
    </p>
    """)
myutils.check_install_package('tensorflow', note=note)
myutils.check_install_package('stardist')

import sys
import tensorflow
import h5py
if sys.version_info.minor < 9:
    # Tensorflow > 2.3 has the requirement h5py~=3.1.0,
    # but stardist 0.7.3 with python<3.9 requires h5py<3
    # see issue here https://github.com/stardist/stardist/issues/180
    tf_version = tensorflow.__version__.split('.')
    tf_major, tf_minor = [int(v) for v in tf_version][:2]
    h5py_version = h5py.__version__.split('.')
    h5py_major = int(h5py_version[0])
    if tf_major > 1 and tf_minor > 2 and h5py_major >= 3:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'h5py==2.10.0']
        )
