from qtpy.QtCore import Signal
from qtpy.QtGui import QShowEvent
from qtpy.QtWidgets import QFrame, QHBoxLayout, QLineEdit

from .buttons import browseFileButton
from .inputs_basic import ElidingLineEdit

class filePathControl(QFrame):
    sigValueChanged = Signal(str)

    def __init__(
        self,
        parent=None,
        browseFolder=False,
        fileManagerTitle="Select file",
        validExtensions=None,
        startFolder="",
        elide=False,
    ):
        super().__init__(parent)

        layout = QHBoxLayout()
        if elide:
            self.le = ElidingLineEdit()
        else:
            self.le = QLineEdit()

        self.browseButton = browseFileButton(
            openFolder=browseFolder,
            title=fileManagerTitle,
            ext=validExtensions,
            start_dir=startFolder,
        )

        layout.addWidget(self.le)
        layout.addWidget(self.browseButton)
        self.setLayout(layout)

        self.le.editingFinished.connect(self.setTextTooltip)
        self.browseButton.sigPathSelected.connect(self.setText)

        self.setFrameStyle(QFrame.Shape.StyledPanel)

    def setText(self, text):
        self.le.setText(text)
        self.le.setToolTip(text)
        self.sigValueChanged.emit(self.le.text())

    def setTextTooltip(self):
        self.le.setToolTip(self.le.text())
        self.sigValueChanged.emit(self.le.text())

    def path(self):
        return self.le.text()

    def showEvent(self, a0: QShowEvent) -> None:
        self.le.setFixedHeight(self.browseButton.height())
        return super().showEvent(a0)


class FolderPathControl(filePathControl):
    def __init__(self, **kwargs):
        super().__init__(browseFolder=True, fileManagerTitle="Select folder", **kwargs)


class CsvFilePathControl(filePathControl):
    def __init__(self, **kwargs):
        super().__init__(
            browseFolder=False,
            fileManagerTitle="Select a CSV file",
            validExtensions={"CSV files": [".csv", ".CSV"]},
            **kwargs,
        )

