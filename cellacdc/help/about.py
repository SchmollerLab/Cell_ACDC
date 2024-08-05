import os
import sys
import cellacdc
import platform
from functools import partial

from qtpy.QtWidgets import (
    QDialog, QLabel, QGridLayout, QHBoxLayout, QSpacerItem, QApplication, 
    QVBoxLayout
)
from qtpy.QtGui import QPixmap
from qtpy.QtCore import Qt
from qtpy import QtCore

from ..myutils import read_version, get_date_from_version
from ..myutils import get_pip_install_cellacdc_version_command
from ..myutils import get_git_pull_checkout_cellacdc_version_commands
from .. import widgets, myutils
from .. import html_utils, printl
from .. import qrc_resources
from .. import cellacdc_path

class QDialogAbout(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog)
        self.setWindowTitle('About Cell-ACDC')

        layout = QGridLayout()
        
        version = read_version()
        release_date = get_date_from_version(version)
        
        py_ver = sys.version_info
        python_version = f'{py_ver.major}.{py_ver.minor}.{py_ver.micro}'

        titleLabel = QLabel()
        txt = (f"""
        <p style="font-size:20px; font-family:ubuntu">
            <b>Cell-ACDC</b>
            <span style="font-size:12pt; font-family:ubuntu">
                (Analysis of the Cell Division Cycle)
            </span>
        </p>
        <p style="font-size:14px; font-family:ubuntu">
            Version {version}<br><br>
            Release date: {release_date}
        </p>
        <p style="font-size:13px; font-family:ubuntu">
            Qt {QtCore.__version__}<br>
            Python {python_version}<br>
        </p>
        <p style="font-size:13px; font-family:ubuntu">
            Platform: {platform.platform()}<br>
        </p>
        """)

        titleLabel.setText(txt)
        
        # '{next_version}.dev{distance}+{scm letter}{revision hash}'
        command, command_github = get_pip_install_cellacdc_version_command(
            version=version
        )
        commandLabel = QLabel(html_utils.paragraph(
            f'<b>To install this specific version</b> '
            f'on a new environment or <b>to upgrade/downgrade</b> in an '
            'environment where you already have Cell-ACDC<br>'
            'installed with pip run the following <b>command</b>:'
        ))
        commandWidget = widgets.CopiableCommandWidget(
            command=command, font_size='11px'
        )
        
        if command_github is not None:
            commandLabelGh = QLabel(html_utils.paragraph(
                f'<b>If the command above fails</b>, it means that this '
                f'specific version was <b>not released on PyPi</b> yet.<br><br>'
                'In that case, you need to run the following command instead:'
            ))
            commandGhWidget = widgets.CopiableCommandWidget(
                command=command_github, font_size='11px'
            )
        
        commandWidgetsGit = []
        git_commands = get_git_pull_checkout_cellacdc_version_commands(version)
        if git_commands:
            commandLabelGit = QLabel(html_utils.paragraph(
                f'<br><br><b>To upgrade/downgrade</b> the Cell-ACDC version in an '
                'environment where you installed it by first cloning with '
                '<code>git</code><br>'
                'run the following <b>commands</b> one by one:'
            ))  
        for command in git_commands:
            commandWidgetsGit.append(
                widgets.CopiableCommandWidget(command=command, font_size='11px')
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
            Installed in: <code>{cellacdc_path}</code>
        """, font_size='12px')
        installedLabel.setText(txt)
        installedLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        
        self.copyCellACDCpathButton = widgets.copyPushButton('Copy path')
        self.copyCellACDCpathButton.clicked.connect(
            self.copyCellACDCpath
        )
        
        self.showHowToInstallButton = widgets.helpPushButton(
            'How to install this version'
        )
        self.showHowToInstallButton.clicked.connect(
            self.showHotToInstallInstructions
        )

        button = widgets.showInFileManagerButton(
            myutils.get_open_filemaneger_os_string()
        )
        func = partial(myutils.showInExplorer, cellacdc_path)
        button.clicked.connect(func)
        installedLayout.addWidget(installedLabel)
        installedLayout.addWidget(self.copyCellACDCpathButton)
        installedLayout.addStretch(1)
        installedLayout.addWidget(self.showHowToInstallButton)
        installedLayout.addWidget(button)

        row = 0
        layout.addWidget(iconLabel, row, 0)
        layout.addWidget(titleLabel, row, 1, alignment=Qt.AlignLeft)
        
        row += 1
        layout.addWidget(infoLabel, row, 1, alignment=Qt.AlignLeft)
        
        row += 1
        layout.setColumnStretch(2,1)
        layout.addItem(QSpacerItem(10,20), row, 1)
        
        row += 1
        layout.setRowStretch(row,1)
        
        row += 1
        layout.addLayout(installedLayout, row, 0, 1, 3)
        
        row += 1
        self.howToInstallDialog = QDialog(self)
        self.howToInstallDialog.setWindowTitle(
            f'How to install Cell-ACDC v{version}'
        )
        howToInstallLayout = QVBoxLayout()
        self.howToInstallDialog.setLayout(howToInstallLayout)
        
        howToInstallOkButton = widgets.okPushButton(' Ok ')
        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(howToInstallOkButton)
        howToInstallOkButton.clicked.connect(self.howToInstallDialog.close)
        
        howToInstallLayout.addWidget(commandLabel, alignment=Qt.AlignLeft)
        howToInstallLayout.addWidget(commandWidget, alignment=Qt.AlignLeft)
        
        if command_github is not None:
            howToInstallLayout.addWidget(commandLabelGh, alignment=Qt.AlignLeft)
            howToInstallLayout.addWidget(commandGhWidget, alignment=Qt.AlignLeft)
        
        if git_commands:
            howToInstallLayout.addWidget(commandLabelGit, alignment=Qt.AlignLeft)
            for widget in commandWidgetsGit:
                howToInstallLayout.addWidget(widget, alignment=Qt.AlignLeft)
        
        howToInstallLayout.addSpacing(20)
        importantText = html_utils.to_admonition("""
            Whenever you run commands with <code>pip</code> <b>make sure to 
            FIRST activate the correct environment</b> (e.g. with 
            <code>conda activate acdc</b>)
        """, admonition_type='important')
        
        howToInstallLayout.addWidget(QLabel(importantText))
        
        # layout.addWidget(self.howToInstallWidget, row, 0, 1, 3)
        howToInstallLayout.addLayout(buttonsLayout)
        self.howToInstallDialog.hide()
        
        self.setLayout(layout)
    
    def copyCellACDCpath(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(cellacdc_path, mode=cb.Clipboard)
    
    def showHotToInstallInstructions(self):
        self.howToInstallDialog.show()

def _test():
    import sys
    from qtpy.QtWidgets import QStyleFactory, QApplication
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    win = QDialogAbout()
    win.show()
    app.exec_()
