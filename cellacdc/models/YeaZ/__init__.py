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
    myutils.install_tensorflow()
