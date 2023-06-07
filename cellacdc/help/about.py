import os
import re
import cellacdc
from functools import partial

from qtpy.QtWidgets import (
    QDialog, QLabel, QGridLayout, QHBoxLayout, QSpacerItem
)
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt

from ..myutils import read_version
from .. import widgets, myutils
from .. import html_utils
from .. import qrc_resources

class QDialogAbout(QDialog):
    def __init__(self, parent=None):
        cellacdc_path = os.path.dirname(os.path.abspath(cellacdc.__file__))
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog)
        self.setWindowTitle('About Cell-ACDC')

        layout = QGridLayout()
        
        version = read_version()

        titleLabel = QLabel()
        txt = (f"""
        <p style="font-size:20px; font-family:ubuntu">
            <b>Cell-ACDC</b>
            <span style="font-size:12pt; font-family:ubuntu">
                (Analysis of the Cell Division Cycle)
            </span>
        </p>
        <p style="font-size:14px; font-family:ubuntu">
            Version {version}
        </p>
        """)

        titleLabel.setText(txt)
        
        commandWidget = None
        if version.find('.dev') != -1:
            # '{next_version}.dev{distance}+{scm letter}{revision hash}'
            commit_hash = re.findall(r'\+g(.+)\.', version)[0]
            command = f'pip install "git+https://github.com/SchmollerLab/Cell_ACDC.git@{commit_hash}"'
            commandLabel = QLabel(html_utils.paragraph(
                f'To install this specific version run the following command:', 
                font_size='11px'
            ))
            commandWidget = widgets.CopiableCommandWidget(
                command=command, font_size='11px'
            )
            
        iconPixmap = QPixmap(":icon.ico")
        h = 128
        iconPixmap = iconPixmap.scaled(h,h, aspectRatioMode=Qt.KeepAspectRatio)
        iconLabel = QLabel()
        iconLabel.setPixmap(iconPixmap)

        github_url = r'https://github.com/SchmollerLab/Cell_ACDC'
        infoLabel = QLabel()
        infoLabel.setTextInteractionFlags(Qt.TextBrowserInteraction);
        infoLabel.setOpenExternalLinks(True);
        txt = html_utils.paragraph(f"""
            More info on our <a href=\"{github_url}">home page</a>.<br>
        """)
        infoLabel.setText(txt)

        installedLayout = QHBoxLayout()
        installedLabel = QLabel()
        txt = html_utils.paragraph(f"""
            Installed in: {cellacdc_path}
        """, font_size='12px')
        installedLabel.setText(txt)

        button = widgets.showInFileManagerButton(
            myutils.get_open_filemaneger_os_string()
        )
        func = partial(myutils.showInExplorer, cellacdc_path)
        button.clicked.connect(func)
        installedLayout.addWidget(installedLabel)
        installedLayout.addStretch(1)
        installedLayout.addWidget(button)

        layout.addWidget(iconLabel, 0, 0)
        layout.addWidget(titleLabel, 0, 1, alignment=Qt.AlignLeft)
        layout.addWidget(infoLabel, 1, 1, alignment=Qt.AlignLeft)
        if commandWidget is not None:
            layout.addWidget(commandLabel, 2, 1, alignment=Qt.AlignLeft)
            layout.addWidget(commandWidget, 3, 1, alignment=Qt.AlignLeft)
        layout.setColumnStretch(2,1)
        layout.addItem(QSpacerItem(10,20), 4, 1)
        layout.setRowStretch(5,1)
        layout.addLayout(installedLayout, 6, 0, 1, 3)
        

        self.setLayout(layout)

def _test():
    import sys
    from qtpy.QtWidgets import QStyleFactory, QApplication
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    win = QDialogAbout()
    win.show()
    app.exec_()
