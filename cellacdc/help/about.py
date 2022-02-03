from PyQt5.QtWidgets import QDialog, QLabel, QGridLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

from ..myutils import read_version
from .. import qrc_resources

class QDialogAbout(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog)
        self.setWindowTitle('About Cell-ACDC')

        layout = QGridLayout()

        titleLabel = QLabel()
        txt = (f"""
        <p style="font-size:18pt; font-family:ubuntu">
            <b>Cell-ACDC</b>
            <span style="font-size:12pt; font-family:ubuntu">
                (Analysis of the Cell Division Cycle)
            </span>
        </p>
        <p style="font-size:12pt; font-family:ubuntu">
            Version {read_version()}
        </p>
        """)

        titleLabel.setText(txt)

        iconPixmap = QPixmap(":assign-motherbud.svg")
        iconLabel = QLabel()
        iconLabel.setPixmap(iconPixmap)

        github_url = r'https://github.com/SchmollerLab/Cell_ACDC'
        infoLabel = QLabel()
        infoLabel.setTextInteractionFlags(Qt.TextBrowserInteraction);
        infoLabel.setOpenExternalLinks(True);
        txt = (f"""
        <p style="font-size:10pt; font-family:ubuntu">
            More info on our <a href=\"{github_url}">home page</a>.<br>
        </p>
        """)
        infoLabel.setText(txt)

        layout.addWidget(iconLabel, 0, 0)
        layout.addWidget(titleLabel, 0, 1)
        layout.addWidget(infoLabel, 1, 1)

        self.setLayout(layout)

def _test():
    import sys
    from PyQt5.QtWidgets import QStyleFactory, QApplication
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))
    win = QDialogAbout()
    win.show()
    app.exec_()
