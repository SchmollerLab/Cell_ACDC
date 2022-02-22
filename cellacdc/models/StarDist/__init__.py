try:
    import stardist
except ModuleNotFoundError:
    pkg_name = 'StarDist'
    import os
    import sys
    import subprocess
    from cellacdc import myutils
    cancel = myutils.install_package_msg(pkg_name)
    if cancel:
        raise ModuleNotFoundError(
            f'User aborted {pkg_name} installation'
        )
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'stardist']
    )

try:
    import tensorflow
except ModuleNotFoundError:
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

import sys
if sys.version_info.minor < 9:
    tf_version = tensorflow.__version__.split('.')
    tf_major, tf_minor = [int(v) for v in tf_version][:2]
    if tf_major > 1 and tf_minor > 4:
        # Tensorflow > 2.5 has the requirement h5py~=3.1.0,
        # but stardist 0.7.3 with python<3.9 requires h5py<3
        # see issue here https://github.com/stardist/stardist/issues/180
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'uninstall', '-y', 'h5py']
        )
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'h5py~=3.1.0']
        )
