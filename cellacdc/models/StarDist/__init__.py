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

    import tensorflow
    tf_version = tensorflow.__version__.split('.')
    tf_major, tf_minor = [int(v) for v in tf_version][:2]
    if tf_major > 1 and tf_minor > 4:
        # Tensorflow > 2.5 has the requirement h5py~=3.1.0,
        # but stardist 0.7.3 requires h5py<3
        # see issue here https://github.com/stardist/stardist/issues/180
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'uninstall', '-y', 'h5py']
        )
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'h5py~=3.1.0']
        )
