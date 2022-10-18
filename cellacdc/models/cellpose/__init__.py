import os
import sys
import subprocess

upgrade_cellpose_txt = (
    'Cell-ACDC needs to <b>upgrade</b> <code>cellpose</code>.<br><br>'
    'It is recommended to <b>restart Cell-ACDC</b> after the installation.'
)

try:
    import pkg_resources
    check_upgrade_cellpose = True
except ModuleNotFoundError:
    check_upgrade_cellpose = False

try:
    import cellpose
    if check_upgrade_cellpose:
        try:
            # Upgrade cellpose to >= 2.0 if needed
            version = pkg_resources.get_distribution("cellpose").version
            major = int(version.split('.')[0])
            if major < 2:
                from PyQt5.QtWidgets import QApplication
                from PyQt5.QtCore import QCoreApplication
                from cellacdc import widgets, html_utils

                if QCoreApplication.instance() is None:
                    app = QApplication(sys.argv)

                txt = html_utils.paragraph(upgrade_cellpose_txt)
                msg = widgets.myMessageBox()
                msg.information(
                    None, 'Upgrading cellpose', txt
                )

                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', '--upgrade', 'cellpose']
                )
        except Exception as e:
            from PyQt5.QtWidgets import QApplication
            from PyQt5.QtCore import QCoreApplication
            from cellacdc import widgets, html_utils

            if QCoreApplication.instance() is None:
                app = QApplication(sys.argv)

            txt = html_utils.paragraph(upgrade_cellpose_txt)
            msg = widgets.myMessageBox()
            msg.information(
                None, 'Upgrading cellpose', txt
            )

            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '--upgrade', 'cellpose']
            )
except ModuleNotFoundError:
    from PyQt5.QtWidgets import QApplication
    from PyQt5.QtCore import QCoreApplication

    if QCoreApplication.instance() is None:
        app = QApplication(sys.argv)

    from cellacdc import myutils
    cancel = myutils.install_package_msg('cellpose')
    if cancel:
        raise ModuleNotFoundError(
            'User aborted cellpose installation'
        )

    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', 'cellpose']
    )
