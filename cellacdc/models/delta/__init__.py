"""
Installs delta2 into acdc.

@author: jroberts / jamesr787
"""

import sys
import subprocess

upgrade_delta_txt = (
    'Cell-ACDC needs to <b>upgrade</b> <code>delta</code>.<br><br>'
    'It is recommended to <b>restart Cell-ACDC</b> after the installation.'
)

try:
    import pkg_resources
    check_upgrade_delta = True
except ModuleNotFoundError:
    check_upgrade_delta = False

try:
    import delta
    if check_upgrade_delta:
        # Upgrade delta to >= 2.0 if needed
        version = pkg_resources.get_distribution("delta2").version
        major = int(version.split('.')[0])
        if major < 2:
            from qtpy.QtWidgets import QApplication
            from qtpy.QtCore import QCoreApplication
            from cellacdc import widgets, html_utils

            if QCoreApplication.instance() is None:
                app = QApplication(sys.argv)

            txt = html_utils.paragraph(upgrade_delta_txt)
            msg = widgets.myMessageBox()
            msg.information(
                None, 'Upgrading delta', txt
            )

            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', 'delta2']
            )

except ModuleNotFoundError:
    from qtpy.QtWidgets import QApplication
    from qtpy.QtCore import QCoreApplication

    if QCoreApplication.instance() is None:
        app = QApplication(sys.argv)

    from cellacdc import myutils
    cancel = myutils._install_package_msg('delta2')
    if cancel:
        raise ModuleNotFoundError(
            'User aborted delta installation'
        )
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '--upgrade', 'delta2']
    )
