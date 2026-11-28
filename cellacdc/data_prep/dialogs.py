"""Data-prep-only dialogs (from apps.py): the sub-crops save paths
DataPrepSubCropsPathsDialog and the frame-range crop tool
QCropTrangeTool.
"""
import os
from collections import defaultdict
from functools import partial
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QLabel, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QGridLayout,
    QScrollArea, QSpacerItem
)
from cellacdc import _window_stacking
from cellacdc import html_utils
from cellacdc import widgets
from cellacdc.shared.dialogs import base as dialogs_base


class DataPrepSubCropsPathsDialog(dialogs_base.QBaseDialog):
    def __init__(self, cropPaths=None, parent=None):
        self.cancel = True
        
        super().__init__(parent=parent)
        
        mainLayout = QVBoxLayout()
        
        gridLayout = QGridLayout()        
        row = 0
        
        if cropPaths is None:
            cropPaths = {os.path.expanduser('~'): 1}
        
        if any([numCrops>1 for numCrops in cropPaths.values()]):
            row += 1
            gridLayout.addWidget(
                QLabel('Same folder for all crops:'), row, 0
            )
            self.sameFolderPathToggle = widgets.Toggle()
            gridLayout.addWidget(
                self.sameFolderPathToggle, row, 1, alignment=Qt.AlignCenter
            )
            self.sameFolderPathToggle.setChecked(True)
            self.sameFolderPathToggle.toggled.connect(self.setSameFolderPath)
        
        self.windowMinWidth = 0
        minWidth = int(self.screen().size().width()/3)
        self.folderPathLineEdits = defaultdict(list)
        for path, numCrops in cropPaths.items():
            row += 1
            gridLayout.addWidget(QLabel('Master Position:'), row, 0)
            masterPathLabel = QLabel(f'<code>{path}</code>')
            gridLayout.addWidget(masterPathLabel, row, 1)
            
            scrollArea = QScrollArea()
            scrollArea.setWidgetResizable(True)
            scrollAreaLayout = QGridLayout()
            for i in range(numCrops):
                label = QLabel(f'<b>Crop {i+1}</b> folder path:')
                scrollAreaLayout.addWidget(label, i, 0)
                folderPathLineEdit = widgets.ElidingLineEdit()
                folderPathLineEdit.label = label
                folderPathLineEdit.setText(path)
                scrollAreaLayout.addWidget(folderPathLineEdit, i, 1)
                browseButton = widgets.browseFileButton(
                    start_dir=path, openFolder=True
                )
                scrollAreaLayout.addWidget(browseButton, i, 2)
                browseButton.sigPathSelected.connect(
                    partial(self.updateFolderPath, lineEdit=folderPathLineEdit)
                )
                self.folderPathLineEdits[path].append(folderPathLineEdit)
                folderPathLineEdit.browseButton = browseButton
            
            scrollAreaLayout.setColumnStretch(0, 0)
            scrollAreaLayout.setColumnStretch(1, 1)
            scrollAreaLayout.setColumnStretch(2, 0)
            container = QWidget()
            container.setLayout(scrollAreaLayout)
            scrollArea.setWidget(container)
            
            row += 1
            gridLayout.addWidget(scrollArea, row, 0, 1, 2)
            noHorizontalScrollbarWidth = (
                container.sizeHint().width() 
                + scrollArea.verticalScrollBar().sizeHint().width() + 20
            )
            if noHorizontalScrollbarWidth > self.windowMinWidth:
                self.windowMinWidth = noHorizontalScrollbarWidth

            row += 1
            gridLayout.addWidget(widgets.QHLine(), row, 0, 1, 2)
            
            row += 1
            gridLayout.addItem(QSpacerItem(10, 10), row, 0, 1, 2)
            
            row += 1
            
        buttonsLayout = widgets.CancelOkButtonsLayout()
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addLayout(gridLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def show(self, block=False):
        self.resize(self.windowMinWidth, self.sizeHint().height())
        super().show(block=block)
    
    def setSameFolderPath(self, checked):
        for masterPath, lineEdits in self.folderPathLineEdits.items():
            referencePath = lineEdits[0].text()
            for lineEdit in lineEdits[1:]:
                if checked:
                    lineEdit.setText(referencePath)
                
                lineEdit.setDisabled(checked)
                lineEdit.browseButton.setDisabled(checked)
                lineEdit.label.setDisabled(checked)
                
    def updateFolderPath(self, path, lineEdit=None):
        lineEdit.setText(path)
        lineEdit.browseButton.setStartPath(path)
    
    def warnFolderPathNotValid(self, cropNum, masterPath, folderPath):
        text = html_utils.paragraph(
            f'The following folder path for crop number {cropNum} '
            'is <b>not a valid folder or does not exist</b>:'
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Not a valid folder', text, commands=(folderPath,))
    
    def askOverwritingPaths(self, overwritingPaths):
        text = html_utils.paragraph(
            'Data in the following paths will be <b>overwritten with '
            'cropped data.</b><br><br>'
            'Are you sure you want to continue?'
        )
        msg = widgets.myMessageBox(wrapText=False)
        _, yesButton = msg.warning(
            self, 'Not a valid folder', text, commands=overwritingPaths, 
            buttonsTexts=('No, let me edit paths', 'Yes, overwrite')
        )
        return msg.clickedButton == yesButton
    
    def validatePaths(self):
        for masterPath, lineEdits in self.folderPathLineEdits.items():
            for i, lineEdit in enumerate(lineEdits):
                path = lineEdit.text()
                if os.path.exists(path) and os.path.isdir(path):
                    continue
                
                self.warnFolderPathNotValid(i+1, masterPath, path)
                return False

        overwritingPaths = []
        for masterPath, lineEdits in self.folderPathLineEdits.items():
            masterPath = masterPath.replace('\\', '/')
            if not masterPath.endswith('Images'):
                continue
            
            for i, lineEdit in enumerate(lineEdits):
                path = lineEdit.text()
                path = path.replace('\\', '/')
                if path == masterPath:
                    overwritingPaths.append(masterPath)
        
        if not overwritingPaths:
            return True
        
        return self.askOverwritingPaths(overwritingPaths)
    
    def paths(self):
        selectedPaths = {}
        for masterPath, lineEdits in self.folderPathLineEdits.items():
            selectedPaths[masterPath] = [le.text() for le in lineEdits]
        return selectedPaths
    
    def ok_cb(self):
        proceed = self.validatePaths()
        if not proceed:
            return

        self.folderPaths = self.paths()
        self.cancel = False
        self.close()


class QCropTrangeTool(dialogs_base.QBaseDialog):
    sigClose = Signal()
    sigTvalueChanged = Signal(int)
    sigReset = Signal()
    sigCrop = Signal(int, int)

    def __init__(
            self, SizeT, 
            cropButtonText='Apply crop', 
            parent=None, 
            addDoNotShowAgain=False, 
            title='Select frames range'
        ):
        super().__init__(parent)

        self.cancel = True

        _window_stacking.set_window_flags_on_top(self, Qt.Tool)

        self.SizeT = SizeT
        self.numDigits = len(str(self.SizeT))

        self.setWindowTitle(title)

        layout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        self.startFrameScrollbar = widgets.sliderWithSpinBox(
            spinbox_loc='left', 
            maximum_on_label=SizeT
        )
        self.startFrameScrollbar.setMaximum(SizeT, including_spinbox=True)
        self.startFrameScrollbar.setMinimum(1, including_spinbox=True)

        self.endFrameScrollbar = widgets.sliderWithSpinBox(
            spinbox_loc='left', 
            maximum_on_label=SizeT
        )
        self.endFrameScrollbar.setMaximum(SizeT, including_spinbox=True)
        self.endFrameScrollbar.setMinimum(1, including_spinbox=True)
        self.endFrameScrollbar.setValue(SizeT)

        cancelButton = widgets.cancelPushButton('Cancel')
        cropButton = widgets.okPushButton(cropButtonText)
        buttonsLayout.addWidget(cropButton)
        buttonsLayout.addWidget(cancelButton)

        row = 0
        layout.addWidget(
            QLabel('Start frame n.  '), row, 0, alignment=Qt.AlignRight
        )
        layout.addWidget(self.startFrameScrollbar, row, 2)

        row += 1
        layout.setRowStretch(row, 5)
        layout.addItem(QSpacerItem(10, 10), row, 0)

        row += 1
        layout.addWidget(
            QLabel('Stop frame n. '), row, 0, alignment=Qt.AlignRight
        )
        layout.addWidget(self.endFrameScrollbar, row, 2)

        row += 1
        if addDoNotShowAgain:
            self.doNotShowAgainCheckbox = QCheckBox('Do not ask again')
            layout.addWidget(
                self.doNotShowAgainCheckbox, row, 2, alignment=Qt.AlignLeft
            )
            row += 1

        layout.addItem(QSpacerItem(10, 20), row, 0)
        layout.addLayout(buttonsLayout, row+1, 2, alignment=Qt.AlignRight)

        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 10)

        self.setLayout(layout)

        # resetButton.clicked.connect(self.emitReset)
        cropButton.clicked.connect(self.emitCrop)
        cancelButton.clicked.connect(self.close)
        self.startFrameScrollbar.sigValueChange.connect(self.TvalueChanged)
        self.endFrameScrollbar.sigValueChange.connect(self.TvalueChanged)

    def emitReset(self):
        self.sigReset.emit()

    def emitCrop(self):
        self.cancel = False
        low_z = self.startFrameScrollbar.value() - 1
        high_z = self.endFrameScrollbar.value() - 1
        self.sigCrop.emit(low_z, high_z)
        self.close()

    def updateScrollbars(self, start_frame_i, lower_frame_i):
        self.startFrameScrollbar.setValue(start_frame_i + 1)
        self.endFrameScrollbar.setValue(lower_frame_i + 1)

    def TvalueChanged(self, value):
        frame_i = value - 1
        self.sigTvalueChanged.emit(frame_i)

    def showEvent(self, event):
        self.resize(int(self.width()*2.0), self.height())

    def closeEvent(self, event):
        super().closeEvent(event)
        self.sigClose.emit()
