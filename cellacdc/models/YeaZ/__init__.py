try:
    import tensorflow
except ModuleNotFoundError:
    pkg_name = 'tensorflow'
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
