import os
import sys
import subprocess

try:
    import tensorflow
except ModuleNotFoundError:
    pkg_name = 'tensorflow'
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
    cancel = myutils.install_package_msg(pkg_name, note=note)
    if cancel:
        raise ModuleNotFoundError(
            f'User aborted {pkg_name} installation'
        )
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'tensorflow']
    )

    # numba requires numpy<1.22 but tensorflow might install higher
    # so install numpy less than 1.22 if needed
    import numpy
    np_version = numpy.__version__.split('.')
    np_major, np_minor = [int(v) for v in np_version][:2]
    if np_major >= 1 and np_minor >= 22:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'numpy<1.22']
        )

try:
    import stardist
except ModuleNotFoundError:
    pkg_name = 'StarDist'
    from cellacdc import myutils
    cancel = myutils.install_package_msg(pkg_name)
    if cancel:
        raise ModuleNotFoundError(
            f'User aborted {pkg_name} installation'
        )
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'stardist']
    )

# import sys
# import tensorflow
# import h5py
# if sys.version_info.minor < 9:
#     # Tensorflow > 2.5 has the requirement h5py~=3.1.0,
#     # but stardist 0.7.3 with python<3.9 requires h5py<3
#     # see issue here https://github.com/stardist/stardist/issues/180
#     tf_version = tensorflow.__version__.split('.')
#     tf_major, tf_minor = [int(v) for v in tf_version][:2]
#     h5py_version = h5py_version.__version__.split('.')
#     h5py_major = int(h5py_version[0])
#     if tf_major > 1 and tf_minor > 4 and h5py_major < 3:
#         subprocess.check_call(
#             [sys.executable, '-m', 'pip', 'uninstall', '-y', 'h5py']
#         )
#         subprocess.check_call(
#             [sys.executable, '-m', 'pip', 'install', 'h5py~=3.1.0']
#         )
