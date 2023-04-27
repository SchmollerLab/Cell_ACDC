try:
    import trackpy
except ModuleNotFoundError:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QCoreApplication
    import sys
    import subprocess

    if QCoreApplication.instance() is None:
        app = QApplication(sys.argv)

    from cellacdc import myutils
    cancel = myutils._install_package_msg('trackpy')
    if cancel:
        raise ModuleNotFoundError(
            'User aborted trackpy installation'
        )

    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'trackpy']
    )