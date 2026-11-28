"""Data structure module dialogs for raw-file metadata and conversion (from
apps.py): the OME-XML metadata dialog QDialogMetadataXML and the Fiji
macro setup InitFijiMacroDialog, plus the widgets only they use.
"""
import traceback
from functools import partial
import html
from qtpy.QtCore import QTimer, Qt
from qtpy.QtGui import QIcon
from qtpy.QtWidgets import (
    QLabel, QHBoxLayout, QToolButton, QApplication, QWidget, QVBoxLayout,
    QScrollArea, QComboBox
)
from cellacdc import is_mac, is_win, html_utils, is_linux
from cellacdc.myutils import filesystem
from .. import helpers as data_structure_helpers
from cellacdc.shared.widgets import item_views as widgets_item_views
from cellacdc.shared.widgets import message_box as widgets_message_box
import os
from itertools import combinations
import pandas as pd
from qtpy.QtCore import QEventLoop
from qtpy.QtWidgets import (
    QLineEdit, QPushButton, QDialog, QCheckBox, QGridLayout, QSpinBox,
    QDoubleSpinBox, QRadioButton
)
from cellacdc import _window_stacking
from cellacdc.shared.widgets import (
    line_edits as widgets_line_edits,
    numeric_inputs as widgets_numeric_inputs,
    push_buttons as widgets_push_buttons,
)
from cellacdc.acdc_regex import is_alphanumeric_filename
from cellacdc import io
from cellacdc import fonts
from cellacdc.shared.dialogs import image_tools as dialogs_image_tools
from cellacdc.shared.dialogs import base as dialogs_base


class ExpandableListBox(QComboBox):
    def __init__(self, parent=None, centered=True) -> None:
        super().__init__(parent)

        self.setEditable(True)
        self.lineEdit().setReadOnly(True)

        infoTxt = html_utils.paragraph(
            'Select <b>Positions to save</b><br><br>'
            '<code>Ctrl+Click</code> <i>to select multiple items</i><br>'
            '<code>Shift+Click</code> <i>to select a range of items</i><br>',
            center=True
        )

        self.listW = widgets_item_views.QDialogListbox(
            'Select Positions to save', infoTxt,
            [], multiSelection=True, parent=self
        )

        self.listW.listBox.itemClicked.connect(self.listItemClicked)
        self.listW.sigSelectionConfirmed.connect(self.updateCombobox)

        self.centered = centered 

    def listItemClicked(self, item):
        if item.text().find('All') == -1:
            return
        
        for i in range(self.listW.listBox.count()):
            _item = self.listW.listBox.item(i)
            _item.setSelected(True)

    def clear(self) -> None:
        self.listW.listBox.clear()
        return super().clear()
    
    def setItems(self, items):
        self.clear()
        self.addItems(items)
    
    def addItems(self, items):
        super().addItems(items)
        self.listW.listBox.addItems(items)
        self.listW.listBox.setCurrentRow(self.currentIndex())
        self.listItemClicked(self.listW.listBox.currentItem())
        if self.centered:
            self.centerItems()
    
    def updateCombobox(self, selectedItemsText):
        isAllItem = [
            i for i, t in enumerate(selectedItemsText) if t.find('All') != -1
        ]
        if len(selectedItemsText) == 1:
            self.setCurrentText(selectedItemsText[0])
        elif isAllItem:
            idx = isAllItem[0]
            self.setCurrentText(selectedItemsText[idx])
        else:
            super().clear()
            super().addItems(['Custom selection'])
    
    def centerItems(self, idx=None):
        self.lineEdit().setAlignment(Qt.AlignCenter)
    
    def selectedItems(self):
        return self.listW.listBox.selectedItems()
    
    def selectedItemsText(self):
        return [item.text() for item in self.selectedItems()]
    
    def showPopup(self) -> None:
        self.listW.show()


class installJavaDialog(widgets_message_box.myMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Install Java')
        self.setIcon('SP_MessageBoxWarning')

        txt_macOS = html_utils.paragraph("""
            Your system doesn't have the <code>Java Development Kit</code>
            installed<br> and/or a C++ compiler which is required for the installation of
            <code>javabridge</code><br><br>
            <b>Cell-ACDC is now going to install Java for you</b>.<br><br>
            <i><b>NOTE: After clicking on "Install", follow the instructions<br>
            on the terminal</b>. You will be asked to confirm steps and insert<br>
            your password to allow the installation.</i><br><br>
            If you prefer to do it manually, cancel the process<br>
            and follow the instructions below.
        """)

        txt_windows = html_utils.paragraph("""
            Unfortunately, installing pre-compiled version of
            <code>javabridge</code> <b>failed</b>.<br><br>
            Cell-ACDC is going to <b>try to compile it now</b>.<br><br>
            However, <b>before proceeding</b>, you need to install
            <code>Java Development Kit</code><br> and a <b>C++ compiler</b>.<br><br>
            <b>See instructions below on how to install it.</b>
        """)

        if not is_win:
            self.instructionsButton = self.addButton('Show intructions...')
            self.instructionsButton.setCheckable(True)
            self.instructionsButton.disconnect()
            self.instructionsButton.clicked.connect(self.showInstructions)
            installButton = self.addButton('Install')
            installButton.disconnect()
            installButton.clicked.connect(self.installJava)
            txt = txt_macOS
        else:
            okButton = self.addButton('Ok')
            txt = txt_windows

        self.cancelButton = self.addButton('Cancel')

        label = self.addText(txt)
        label.setWordWrap(False)

        self.resizeCount = 0

    def addInstructionsWindows(self):
        self.scrollArea = QScrollArea()
        _container = QWidget()
        _layout = QVBoxLayout()
        for t, text in enumerate(data_structure_helpers.install_javabridge_instructions_text()):
            label = QLabel()
            label.setText(text)
            if (t == 1 or t == 2):
                label.setOpenExternalLinks(True)
                label.setTextInteractionFlags(Qt.TextBrowserInteraction)
                code_layout = QHBoxLayout()
                code_layout.addWidget(label)
                copyButton = QToolButton()
                copyButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                copyButton.setIcon(QIcon(':edit-copy.svg'))
                copyButton.setText('Copy link')
                if t==1:
                    copyButton.textToCopy = data_structure_helpers.jdk_windows_url()
                    code_layout.addWidget(copyButton, alignment=Qt.AlignLeft)
                else:
                    copyButton.textToCopy = data_structure_helpers.cpp_windows_url()
                    screenshotButton = QToolButton()
                    screenshotButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                    screenshotButton.setIcon(QIcon(':cog.svg'))
                    screenshotButton.setText('See screenshot')
                    code_layout.addWidget(screenshotButton, alignment=Qt.AlignLeft)
                    code_layout.addWidget(copyButton, alignment=Qt.AlignLeft)
                    screenshotButton.clicked.connect(self.viewScreenshot)
                copyButton.clicked.connect(self.copyToClipboard)
                code_layout.setStretch(0, 2)
                code_layout.setStretch(1, 0)
                _layout.addLayout(code_layout)
            else:
                _layout.addWidget(label)


        _container.setLayout(_layout)
        self.scrollArea.setWidget(_container)
        self.currentRow += 1
        self._layout.addWidget(
            self.scrollArea, self.currentRow, 1, alignment=Qt.AlignTop
        )

        # Stretch last row
        self.currentRow += 1
        self._layout.setRowStretch(self.currentRow, 1)

    def viewScreenshot(self, checked=False):
        self.screenShotWin = widgets_message_box.view_visualcpp_screenshot(parent=self)
        self.screenShotWin.show()

    def addInstructionsMacOS(self):
        self.scrollArea = QScrollArea()
        _container = QWidget()
        _layout = QVBoxLayout()
        for t, text in enumerate(data_structure_helpers.install_javabridge_instructions_text()):
            label = QLabel()
            label.setText(text)
            # label.setWordWrap(True)
            if (t == 1 or t == 2):
                label.setWordWrap(True)
                code_layout = QHBoxLayout()
                code_layout.addWidget(label)
                copyButton = QToolButton()
                copyButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                copyButton.setIcon(QIcon(':edit-copy.svg'))
                copyButton.setText('Copy')
                if t==1:
                    copyButton.textToCopy = data_structure_helpers._install_homebrew_command()
                else:
                    copyButton.textToCopy = data_structure_helpers._brew_install_java_command()
                copyButton.clicked.connect(self.copyToClipboard)
                code_layout.addWidget(copyButton, alignment=Qt.AlignLeft)
                # code_layout.addStretch(1)
                code_layout.setStretch(0, 2)
                code_layout.setStretch(1, 0)
                _layout.addLayout(code_layout)
            else:
                _layout.addWidget(label)
        _container.setLayout(_layout)
        self.scrollArea.setWidget(_container)
        self.currentRow += 1
        self._layout.addWidget(
            self.scrollArea, self.currentRow, 1, alignment=Qt.AlignTop
        )

        # Stretch last row
        self.currentRow += 1
        self._layout.setRowStretch(self.currentRow, 1)
        self.scrollArea.hide()

    def addInstructionsLinux(self):
        self.scrollArea = QScrollArea()
        _container = QWidget()
        _layout = QVBoxLayout()
        for t, text in enumerate(data_structure_helpers.install_javabridge_instructions_text()):
            label = QLabel()
            label.setText(text)
            # label.setWordWrap(True)
            if (t == 1 or t == 2 or t==3):
                label.setWordWrap(True)
                code_layout = QHBoxLayout()
                code_layout.addWidget(label)
                copyButton = QToolButton()
                copyButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                copyButton.setIcon(QIcon(':edit-copy.svg'))
                copyButton.setText('Copy')
                if t==1:
                    copyButton.textToCopy = data_structure_helpers._apt_update_command()
                elif t==2:
                    copyButton.textToCopy = data_structure_helpers._apt_install_java_command()
                elif t==3:
                    copyButton.textToCopy = data_structure_helpers._apt_gcc_command()
                copyButton.clicked.connect(self.copyToClipboard)
                code_layout.addWidget(copyButton, alignment=Qt.AlignLeft)
                # code_layout.addStretch(1)
                code_layout.setStretch(0, 2)
                code_layout.setStretch(1, 0)
                _layout.addLayout(code_layout)
            else:
                _layout.addWidget(label)
        _container.setLayout(_layout)
        self.scrollArea.setWidget(_container)
        self.currentRow += 1
        self._layout.addWidget(
            self.scrollArea, self.currentRow, 1, alignment=Qt.AlignTop
        )

        # Stretch last row
        self.currentRow += 1
        self._layout.setRowStretch(self.currentRow, 1)
        self.scrollArea.hide()

    def copyToClipboard(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        plain_text = html.unescape(self.sender().textToCopy).replace("\xa0", " ")
        cb.setText(plain_text, mode=cb.Clipboard)
        print('Command copied!')

    def showInstructions(self, checked):
        if checked:
            self.instructionsButton.setText('Hide instructions')
            self.origHeight = self.height()
            self.resize(self.width(), self.height()+300)
            self.scrollArea.show()
        else:
            self.instructionsButton.setText('Show instructions...')
            self.scrollArea.hide()
            func = partial(self.resize, self.width(), self.origHeight)
            QTimer.singleShot(50, func)

    def installJava(self):
        import subprocess
        try:
            if is_mac:
                try:
                    subprocess.check_call(['brew', 'update'])
                except Exception as e:
                    subprocess.run(
                        data_structure_helpers._install_homebrew_command(),
                        check=True, text=True, shell=True
                    )
                subprocess.run(
                    data_structure_helpers._brew_install_java_command(),
                    check=True, text=True, shell=True
                )
            elif is_linux:
                subprocess.run(
                    data_structure_helpers._apt_gcc_command()(),
                    check=True, text=True, shell=True
                )
                subprocess.run(
                    data_structure_helpers._apt_update_command()(),
                    check=True, text=True, shell=True
                )
                subprocess.run(
                    data_structure_helpers._apt_install_java_command()(),
                    check=True, text=True, shell=True
                )
            self.close()
        except Exception as e:
            print('=======================')
            traceback.print_exc()
            print('=======================')
            msg = widgets_message_box.myMessageBox(wrapText=False)
            err_msg = html_utils.paragraph("""
                Automatic installation of Java failed.<br><br>
                Please, try manually by following the instructions provided
                below (click on "Show instructions..." button). Thanks
            """)
            msg.critical(
               self, 'Java installation failed', err_msg
            )

    def show(self, block=False):
        super().show(block=False)
        print(is_linux)
        if is_win:
            self.addInstructionsWindows()
        elif is_mac:
            self.addInstructionsMacOS()
        elif is_linux:
            self.addInstructionsLinux()
        self.move(self.pos().x(), 20)
        if is_win:
            self.resize(self.width(), self.height()+200)
        if block:
            self._block()

    def exec_(self):
        self.show(block=True)


class QDialogMetadataXML(QDialog):
    def __init__(
            self, title='Metadata',
            LensNA=1.0, rawFilename='test', SizeT=1, SizeZ=1, SizeC=1, SizeS=1,
            TimeIncrement=1.0, TimeIncrementUnit='s',
            PhysicalSizeX=1.0, PhysicalSizeY=1.0, PhysicalSizeZ=1.0,
            PhysicalSizeUnit='μm', ImageName='', chNames=None, emWavelens=None,
            parent=None, rawDataStruct=None, sampleImgData=None,
            rawFilePath=None
        ):
        self.cancel = True
        self.trust = False
        self.overWrite = False
        rawFilename = os.path.splitext(rawFilename)[0]
        self.rawFilename = self.removeInvalidCharacters(rawFilename)
        self.rawFilePath = rawFilePath
        self.sampleImgData = sampleImgData
        self.ImageName = ImageName
        self.rawDataStruct = rawDataStruct
        self.readSampleImgDataAgain = False
        self.requestedReadingSampleImageDataAgain = False
        self.imageViewer = None
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setFont(fonts.font)

        mainLayout = QVBoxLayout()
        entriesLayout = QGridLayout()
        self.channelNameLayouts = (
            QVBoxLayout(), QVBoxLayout(), QVBoxLayout(), QVBoxLayout()
        )
        self.channelEmWLayouts = (
            QVBoxLayout(), QVBoxLayout(), QVBoxLayout(), QVBoxLayout()
        )
        buttonsLayout = QGridLayout()

        infoLabel = QLabel()
        infoTxt = (
            '<b>Confirm/Edit</b> the <b>metadata</b> below.'
        )
        infoLabel.setText(infoTxt)
        # padding: top, left, bottom, right
        infoLabel.setStyleSheet("font-size:12pt; padding:0px 0px 5px 0px;")
        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        noteLabel = QLabel()
        noteLabel.setText(
            f'NOTE: If you are not sure about some of the entries '
            'you can try to click "Ok".\n'
            'If they are wrong you will get '
            'an error message later when trying to read the data.'
        )
        noteLabel.setAlignment(Qt.AlignCenter)
        mainLayout.addWidget(noteLabel, alignment=Qt.AlignCenter)

        row = 0
        to_tif_radiobutton = QRadioButton(".tif")
        to_tif_radiobutton.setChecked(True)
        to_h5_radiobutton = QRadioButton(".h5")
        to_h5_radiobutton.setToolTip(
            '.h5 is highly recommended for big datasets to avoid memory issues.\n'
            'As a rule of thumb, if the single position, single channel file\n'
            'is larger than 1/5 of the available RAM we recommend using .h5 format'
        )
        self.to_h5_radiobutton = to_h5_radiobutton
        txt = 'File format:  '
        label = QLabel(txt)
        fileFormatLayout = QHBoxLayout()
        fileFormatLayout.addStretch(1)
        fileFormatLayout.addWidget(to_tif_radiobutton)
        fileFormatLayout.addStretch(1)
        fileFormatLayout.addWidget(to_h5_radiobutton)
        fileFormatLayout.addStretch(1)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addLayout(fileFormatLayout, row, 1)
        to_h5_radiobutton.toggled.connect(self.updateFileFormat)

        row += 1
        self.SizeS_SB = QSpinBox()
        self.SizeS_SB.setAlignment(Qt.AlignCenter)
        self.SizeS_SB.setMinimum(1)
        self.SizeS_SB.setMaximum(2147483647)
        self.SizeS_SB.setValue(SizeS)
        txt = 'Number of positions (SizeS):  '
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.SizeS_SB, row, 1)
    
        if rawDataStruct == 0:
            row += 1
            # self.SizeS_SB.setValue(1)
            self.SizeS_SB.setDisabled(True)
            self.posSelector = ExpandableListBox()
            positions = ['All positions']
            positions.extend([f'Position_{i+1}' for i in range(SizeS)])
            self.posSelector.addItems(positions)
            txt = 'Positions to save:  '
            label = QLabel(txt)
            entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
            entriesLayout.addWidget(self.posSelector, row, 1)
            self.SizeS_SB.valueChanged.connect(self.SizeSvalueChanged)

        row += 1
        self.LensNA_DSB = QDoubleSpinBox()
        self.LensNA_DSB.setAlignment(Qt.AlignCenter)
        self.LensNA_DSB.setSingleStep(0.1)
        self.LensNA_DSB.setValue(LensNA)
        txt = 'Numerical Aperture Objective Lens:  '
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.LensNA_DSB, row, 1)
        
        row += 1
        self.SizeT_SB = QSpinBox()
        self.SizeT_SB.setAlignment(Qt.AlignCenter)
        self.SizeT_SB.setMinimum(1)
        self.SizeT_SB.setMaximum(2147483647)
        self.SizeT_SB.setValue(SizeT)
        txt = 'Number of frames (SizeT):  '
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.SizeT_SB, row, 1)
        self.SizeT_SB.valueChanged.connect(self.hideShowTimeIncrement)
        
        row += 1
        self.timeRangeToSaveWidget = widgets_numeric_inputs.RangeSelector(integers=True)
        self.timeRangeToSaveWidget.setRange(1, SizeT)
        txt = 'Time range to save:  '
        label = QLabel(txt)
        self.timeRangeToSaveWidget.label = label
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.timeRangeToSaveWidget, row, 1)

        row += 1
        self.SizeZ_SB = QSpinBox()
        self.SizeZ_SB.setAlignment(Qt.AlignCenter)
        self.SizeZ_SB.setMinimum(1)
        self.SizeZ_SB.setMaximum(2147483647)
        self.SizeZ_SB.setValue(SizeZ)
        txt = 'Number of z-slices in the z-stack (SizeZ):  '
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.SizeZ_SB, row, 1)
        self.SizeZ_SB.valueChanged.connect(self.hideShowPhysicalSizeZ)

        row += 1
        self.TimeIncrement_DSB = widgets_line_edits.FloatLineEdit(
            allowNegative=False, warningValues={1.0}
        )
        self.TimeIncrement_DSB.setValue(TimeIncrement)
        self.TimeIncrement_DSB.setMinimum(0.0)
        txt = 'Frame interval:  '
        label = QLabel(txt)
        self.TimeIncrement_Label = label
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.TimeIncrement_DSB, row, 1)

        self.TimeIncrementUnit_CB = QComboBox()
        unitItems = [
            'ms', 'seconds', 'minutes', 'hours'
        ]
        currentTxt = [unit for unit in unitItems
                      if unit.startswith(TimeIncrementUnit)]
        self.TimeIncrementUnit_CB.addItems(unitItems)
        if currentTxt:
            self.TimeIncrementUnit_CB.setCurrentText(currentTxt[0])
        entriesLayout.addWidget(
            self.TimeIncrementUnit_CB, row, 2, alignment=Qt.AlignLeft
        )

        row += 1
        self.PhysicalSizeX_DSB = QDoubleSpinBox()
        self.PhysicalSizeX_DSB.setAlignment(Qt.AlignCenter)
        self.PhysicalSizeX_DSB.setMaximum(2147483647.0)
        self.PhysicalSizeX_DSB.setSingleStep(0.001)
        self.PhysicalSizeX_DSB.setDecimals(7)
        self.PhysicalSizeX_DSB.setValue(PhysicalSizeX)
        txt = 'Pixel width (X):  '
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.PhysicalSizeX_DSB, row, 1)

        self.PhysicalSizeUnit_CB = QComboBox()
        unitItems = [
            'nm', 'μm', 'mm', 'cm'
        ]
        currentTxt = [unit for unit in unitItems
                      if unit.startswith(PhysicalSizeUnit)]
        self.PhysicalSizeUnit_CB.addItems(unitItems)
        if currentTxt:
            self.PhysicalSizeUnit_CB.setCurrentText(currentTxt[0])
        else:
            self.PhysicalSizeUnit_CB.setCurrentText(unitItems[1])
        entriesLayout.addWidget(
            self.PhysicalSizeUnit_CB, row, 2, alignment=Qt.AlignLeft
        )
        self.PhysicalSizeUnit_CB.currentTextChanged.connect(self.updatePSUnit)

        row += 1
        self.PhysicalSizeY_DSB = QDoubleSpinBox()
        self.PhysicalSizeY_DSB.setAlignment(Qt.AlignCenter)
        self.PhysicalSizeY_DSB.setMaximum(2147483647.0)
        self.PhysicalSizeY_DSB.setSingleStep(0.001)
        self.PhysicalSizeY_DSB.setDecimals(7)
        self.PhysicalSizeY_DSB.setValue(PhysicalSizeY)
        txt = 'Pixel height (Y):  '
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.PhysicalSizeY_DSB, row, 1)

        self.PhysicalSizeYUnit_Label = QLabel()
        self.PhysicalSizeYUnit_Label.setStyleSheet(
            'font-size:13px; padding:5px 0px 2px 0px;'
        )
        unit = self.PhysicalSizeUnit_CB.currentText()
        self.PhysicalSizeYUnit_Label.setText(unit)
        entriesLayout.addWidget(self.PhysicalSizeYUnit_Label, row, 2)

        row += 1
        self.PhysicalSizeZ_DSB = QDoubleSpinBox()
        self.PhysicalSizeZ_DSB.setAlignment(Qt.AlignCenter)
        self.PhysicalSizeZ_DSB.setMaximum(2147483647.0)
        self.PhysicalSizeZ_DSB.setSingleStep(0.001)
        self.PhysicalSizeZ_DSB.setDecimals(7)
        self.PhysicalSizeZ_DSB.setValue(PhysicalSizeZ)
        txt = 'Voxel depth (Z):  '
        self.PSZlabel = QLabel(txt)
        entriesLayout.addWidget(self.PSZlabel, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.PhysicalSizeZ_DSB, row, 1)

        self.PhysicalSizeZUnit_Label = QLabel()
        # padding: top, left, bottom, right
        self.PhysicalSizeZUnit_Label.setStyleSheet(
            'font-size:13px; padding:5px 0px 2px 0px;'
        )
        unit = self.PhysicalSizeUnit_CB.currentText()
        self.PhysicalSizeZUnit_Label.setText(unit)
        entriesLayout.addWidget(self.PhysicalSizeZUnit_Label, row, 2)

        if SizeZ == 1:
            self.PSZlabel.hide()
            self.PhysicalSizeZ_DSB.hide()
            self.PhysicalSizeZUnit_Label.hide()

        row += 1
        self.SizeC_SB = QSpinBox()
        self.SizeC_SB.setAlignment(Qt.AlignCenter)
        self.SizeC_SB.setMinimum(1)
        self.SizeC_SB.setMaximum(2147483647)
        self.SizeC_SB.setValue(SizeC)
        txt = 'Number of channels (SizeC):  '
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.SizeC_SB, row, 1)
        self.SizeC_SB.valueChanged.connect(self.addRemoveChannels)

        row += 1
        for j, layout in enumerate(self.channelNameLayouts):
            entriesLayout.addLayout(layout, row, j)

        self.chNames_QLEs = []
        self.saveChannels_QCBs = []
        self.filename_QLabels = []
        self.showChannelDataButtons = []

        ext = 'h5' if self.to_h5_radiobutton.isChecked() else 'tif'
        for c in range(SizeC):
            chName_QLE = QLineEdit()
            chName_QLE.setStyleSheet('')
            chName_QLE.setAlignment(Qt.AlignCenter)
            chName_QLE.textChanged.connect(self.checkChNames)
            if chNames is not None:
                chName_QLE.setText(chNames[c])
            else:
                chName_QLE.setText(f'channel_{c}')
                filename = f''

            txt = f'Channel {c} name:  '
            label = QLabel(txt)

            filenameDescLabel = QLabel(f'<i>e.g., filename for channel {c}:  </i>')

            chName = chName_QLE.text()
            chName = self.removeInvalidCharacters(chName)
            rawFilename = self.elidedRawFilename()
            filenameLabel = QLabel(f"""
                <p style=font-size:10px>{rawFilename}_{chName}.{ext}</p>
            """)
            filenameLabel.setToolTip(f'{self.rawFilename}_{chName}.{ext}')

            checkBox = QCheckBox('Save this channel')
            checkBox.setChecked(True)
            checkBox.stateChanged.connect(self.saveCh_checkBox_cb)

            self.channelNameLayouts[0].addWidget(label, alignment=Qt.AlignRight)
            self.channelNameLayouts[0].addWidget(
                filenameDescLabel, alignment=Qt.AlignRight
            )
            self.channelNameLayouts[1].addWidget(chName_QLE)
            self.channelNameLayouts[1].addWidget(
                filenameLabel, alignment=Qt.AlignCenter
            )

            self.channelNameLayouts[2].addWidget(checkBox)
            if c == 0 and ImageName:
                addImageName_QCB = QCheckBox('Include image name')
                addImageName_QCB.stateChanged.connect(self.addImageName_cb)
                self.addImageName_QCB = addImageName_QCB
                self.channelNameLayouts[2].addWidget(addImageName_QCB)
            else:
                self.addImageName_QCB = QCheckBox('dummy')
                self.addImageName_QCB.hide()
                self.channelNameLayouts[2].addWidget(QLabel())

            showChannelDataButton = QPushButton()
            showChannelDataButton.setIcon(QIcon(":eye-plus.svg"))
            showChannelDataButton.clicked.connect(self.showChannelData)
            self.channelNameLayouts[3].addWidget(showChannelDataButton)
            if self.sampleImgData is None:
                showChannelDataButton.setDisabled(True)

            self.chNames_QLEs.append(chName_QLE)
            self.saveChannels_QCBs.append(checkBox)
            self.filename_QLabels.append(filenameLabel)
            self.showChannelDataButtons.append(showChannelDataButton)

        self.checkChNames()

        row += 1
        for j, layout in enumerate(self.channelEmWLayouts):
            entriesLayout.addLayout(layout, row, j)

        self.emWavelens_DSBs = []
        for c in range(SizeC):
            row += 1
            emWavelen_DSB = QDoubleSpinBox()
            emWavelen_DSB.setAlignment(Qt.AlignCenter)
            emWavelen_DSB.setMaximum(2147483647.0)
            emWavelen_DSB.setSingleStep(0.001)
            emWavelen_DSB.setDecimals(2)
            if emWavelens is not None:
                emWavelen_DSB.setValue(emWavelens[c])
            else:
                emWavelen_DSB.setValue(500.0)

            txt = f'Channel {c} emission wavelength:  '
            label = QLabel(txt)
            self.channelEmWLayouts[0].addWidget(label, alignment=Qt.AlignRight)
            self.channelEmWLayouts[1].addWidget(emWavelen_DSB)
            self.emWavelens_DSBs.append(emWavelen_DSB)

            unit = QLabel('nm')
            unit.setStyleSheet('font-size:13px; padding:5px 0px 2px 0px;')
            self.channelEmWLayouts[2].addWidget(unit)

        entriesLayout.setContentsMargins(0, 15, 0, 0)

        if rawDataStruct is None or rawDataStruct!=-1:
            okButton = widgets_push_buttons.okPushButton(' Ok ')
        elif rawDataStruct==1:
            okButton = QPushButton(' Load next position ')
        buttonsLayout.addWidget(okButton, 0, 1)

        self.trustButton = None
        self.overWriteButton = None
        if rawDataStruct==1:
            trustButton = QPushButton(
                ' Trust metadata reader\n for all next positions ')
            trustButton.setToolTip(
                "If you didn't have to manually modify metadata entries\n"
                "it is very likely that metadata from the metadata reader\n"
                "will be correct also for all the next positions.\n\n"
                "Click this button to stop showing this dialog and use\n"
                "the metadata from the reader for all the next files.\n"
                "(except for channel names, I will use the manually entered)"
            )
            buttonsLayout.addWidget(trustButton, 1, 1)
            self.trustButton = trustButton

            overWriteButton = QPushButton(
                ' Use the above metadata\n for all the next positions ')
            overWriteButton.setToolTip(
                "If you had to manually modify metadata entries\n"
                "AND you know they will be the same for all next positions\n"
                "you can click this button to stop showing this dialog\n"
                "and use the same metadata for all the next positions."
            )
            buttonsLayout.addWidget(overWriteButton, 1, 2)
            self.overWriteButton = overWriteButton

            trustButton.clicked.connect(self.ok_cb)
            overWriteButton.clicked.connect(self.ok_cb)

        cancelButton = widgets_push_buttons.cancelPushButton('Cancel')
        buttonsLayout.addWidget(cancelButton, 0, 2)
        buttonsLayout.setColumnStretch(0, 1)
        buttonsLayout.setColumnStretch(3, 1)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(entriesLayout)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch(1)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)
        
        self.hideShowTimeIncrement(SizeT)
        self.readSampleImgDataAgain = False

        self.setLayout(mainLayout)
        # self.setModal(True)

    def saveCh_checkBox_cb(self, state):
        self.checkChNames()
        idx = self.saveChannels_QCBs.index(self.sender())
        LE = self.chNames_QLEs[idx]
        idx *= 2
        LE.setDisabled(state==0)
        label = self.channelNameLayouts[0].itemAt(idx).widget()
        if state == 0:
            label.setStyleSheet('color: gray; font-size: 10pt')
        else:
            label.setStyleSheet('color: black; font-size: 10pt')

        label = self.channelNameLayouts[0].itemAt(idx+1).widget()
        if state == 0:
            label.setStyleSheet('color: gray; font-size: 10pt')
        else:
            label.setStyleSheet('color: black; font-size: 10pt')

        label = self.channelNameLayouts[1].itemAt(idx+1).widget()
        if state == 0:
            label.setStyleSheet('color: gray; font-size: 10pt')
        else:
            label.setStyleSheet('color: black; font-size: 10pt')

    def addImageName_cb(self, state):
        for idx in range(self.SizeC_SB.value()):
            self.updateFilename(idx)

    def setInvalidChName_StyleSheet(self, LE):
        LE.setStyleSheet(
            'border-radius: 4px;'
            'border: 1.5px solid red;'
            'padding: 1px 0px 1px 0px'
        )

    def removeInvalidCharacters(self, chName):
        # Remove invalid charachters
        chName = "".join(
            c if c.isalnum() or c=='_' or c=='' else '_' for c in chName
        )
        trim_ = chName.endswith('_')
        while trim_:
            chName = chName[:-1]
            trim_ = chName.endswith('_')
        return chName

    def updateFileFormat(self, is_h5):
        for idx in range(len(self.chNames_QLEs)):
            self.updateFilename(idx)
    
    def SizeSvalueChanged(self, SizeS):
        positions = ['All positions']
        positions.extend([f'Position_{i+1}' for i in range(SizeS)])
        self.posSelector.setItems(positions)
    
    def elidedRawFilename(self):
        n = 31
        idx = int((n-3)/2)
        if len(self.rawFilename) > 21:
            elidedText = f'{self.rawFilename[:idx]}...{self.rawFilename[-idx:]}'
        else:
            elidedText = self.rawFilename
        return elidedText

    def updateFilename(self, idx):
        chName = self.chNames_QLEs[idx].text()
        chName = self.removeInvalidCharacters(chName)
        if self.rawDataStruct == 2:
            rawFilename = f'{self.rawFilename}_s{idx+1}'
        else:
            rawFilename = self.rawFilename

        ext = 'h5' if self.to_h5_radiobutton.isChecked() else 'tif'

        rawFilename = self.elidedRawFilename()

        filenameLabel = self.filename_QLabels[idx]
        if self.addImageName_QCB.isChecked():
            self.ImageName = self.removeInvalidCharacters(self.ImageName)
            filename = (f"""
                <p style=font-size:10px>
                    {rawFilename}_{self.ImageName}_{chName}.{ext}
                </p>
            """)
            fullFilename = f'{self.rawFilename}_{self.ImageName}_{chName}.{ext}'
        else:
            filename = (f"""
                <p style=font-size:10px>
                    {rawFilename}_{chName}.{ext}
                </p>
            """)
            fullFilename = f'{self.rawFilename}_{chName}.{ext}'
        filenameLabel.setToolTip(fullFilename)
        filenameLabel.setText(filename)

    def checkChNames(self, text=''):
        if self.sender() in self.chNames_QLEs:
            idx = self.chNames_QLEs.index(self.sender())
            self.updateFilename(idx)
        elif self.sender() in self.saveChannels_QCBs:
            idx = self.saveChannels_QCBs.index(self.sender())
            self.updateFilename(idx)


        areChNamesValid = True
        if len(self.chNames_QLEs) == 1:
            LE1 = self.chNames_QLEs[0]
            saveCh = self.saveChannels_QCBs[0].isChecked()
            if not saveCh:
                LE1.setStyleSheet('')
                return areChNamesValid

            s1 = LE1.text()
            if not s1:
                self.setInvalidChName_StyleSheet(LE1)
                areChNamesValid = False
            else:
                LE1.setStyleSheet('')
            return areChNamesValid

        for LE1, LE2 in combinations(self.chNames_QLEs, 2):
            s1 = LE1.text()
            s2 = LE2.text()
            LE1_idx = self.chNames_QLEs.index(LE1)
            LE2_idx = self.chNames_QLEs.index(LE2)
            saveCh1 = self.saveChannels_QCBs[LE1_idx].isChecked()
            saveCh2 = self.saveChannels_QCBs[LE2_idx].isChecked()
            if not s1 or not s2 or s1==s2:
                if not s1 and saveCh1:
                    self.setInvalidChName_StyleSheet(LE1)
                    areChNamesValid = False
                else:
                    LE1.setStyleSheet('')
                if not s2 and saveCh2:
                    self.setInvalidChName_StyleSheet(LE2)
                    areChNamesValid = False
                else:
                    LE2.setStyleSheet('')
                if s1 == s2 and saveCh1 and saveCh2:
                    self.setInvalidChName_StyleSheet(LE1)
                    self.setInvalidChName_StyleSheet(LE2)
                    areChNamesValid = False
            else:
                LE1.setStyleSheet('')
                LE2.setStyleSheet('')
        return areChNamesValid

    def hideShowTimeIncrement(self, value):
        if self.TimeIncrement_DSB.isVisible() and value == 1:
            self.readSampleImgDataAgain = True
        
        if not self.TimeIncrement_DSB.isVisible() and value > 1:
            self.readSampleImgDataAgain = True
            
        if value > 1:
            self.TimeIncrement_DSB.show()
            self.TimeIncrementUnit_CB.show()
            self.TimeIncrement_Label.show()
            self.timeRangeToSaveWidget.show()
            self.timeRangeToSaveWidget.label.show()
            self.timeRangeToSaveWidget.setRange(1, value)
        else:
            self.TimeIncrement_DSB.hide()
            self.TimeIncrementUnit_CB.hide()
            self.TimeIncrement_Label.hide()
            self.timeRangeToSaveWidget.hide()
            self.timeRangeToSaveWidget.label.hide()

    def hideShowPhysicalSizeZ(self, value):
        if value > 1:
            self.PSZlabel.show()
            self.PhysicalSizeZ_DSB.show()
            self.PhysicalSizeZUnit_Label.show()
        else:
            self.PSZlabel.hide()
            self.PhysicalSizeZ_DSB.hide()
            self.PhysicalSizeZUnit_Label.hide()
        self.readSampleImgDataAgain = True

    def updatePSUnit(self, unit):
        self.PhysicalSizeYUnit_Label.setText(unit)
        self.PhysicalSizeZUnit_Label.setText(unit)
    
    def warnRestart(self):
        msg = widgets_message_box.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph("""
            Since you manually changed some of the metadata, this dialogue will now restart<br>
            because it <b>needs to read the image data again</b>.<br><br>
            Thank you for your patience.
        """)
        msg.warning(self, 'Restart required', txt)

    def showChannelData(self, checked=False, idx=None):
        if self.readSampleImgDataAgain:
            # User changed SizeZ, SizeT, or SizeC --> we need to read sample 
            # image again
            del self.sampleImgData
            self.requestedReadingSampleImageDataAgain = True
            self.sampleImgData = None
            self.warnRestart()
            self.getValues()
            self.cancel = False
            self.close()
            return
        
        if idx is None:
            idx = self.showChannelDataButtons.index(self.sender())
        dimsOrder = 'ctz'
        imgData = self.sampleImgData[dimsOrder][idx]
        posData = filesystem.utilClass()
        posData.frame_i = 0
        sampleSizeT = 4 if self.SizeT_SB.value() >= 4 else self.SizeT_SB.value()
        posData.SizeT = sampleSizeT
        SizeZ = self.SizeZ_SB.value()
        posData.SizeZ = 20 if SizeZ>20 else SizeZ
        posData.filename = f'{self.rawFilename}_C={idx}'
        posData.segmInfo_df = pd.DataFrame({
            'filename': [posData.filename]*sampleSizeT,
            'frame_i': range(sampleSizeT),
            'which_z_proj_gui': ['single z-slice']*sampleSizeT,
            'z_slice_used_gui': [int(posData.SizeZ/2)]*sampleSizeT
        }).set_index(['filename', 'frame_i'])
        path_li = os.path.normpath(self.rawFilePath).split(os.sep)
        posData.relPath = f'{f"{os.sep}".join(path_li[-3:1])}'
        posData.relPath = f'{posData.relPath}{os.sep}{posData.filename}'
        if sampleSizeT == 1:
            posData.img_data = [imgData] # single frame data
        else:
            posData.img_data = imgData

        if self.imageViewer is not None:
            self.imageViewer.close()
        
        self.imageViewer = dialogs_image_tools.imageViewer(
            posData=posData, isSigleFrame=False, enableOverlay=False
        )
        self.imageViewer.channelIndex = idx
        self.imageViewer.update_img()
        self.imageViewer.sigClosed.connect(self.imageViewerClosed)
        self.imageViewer.show()
    
    def imageViewerClosed(self):
        self.imageViewer = None

    def addRemoveChannels(self, value):
        self.readSampleImgDataAgain = True
        currentSizeC = len(self.chNames_QLEs)
        DeltaChannels = abs(value-currentSizeC)
        ext = 'h5' if self.to_h5_radiobutton.isChecked() else 'tif'
        if value > currentSizeC:
            for c in range(currentSizeC, currentSizeC+DeltaChannels):
                chName_QLE = QLineEdit()
                chName_QLE.setStyleSheet('')
                chName_QLE.setAlignment(Qt.AlignCenter)
                chName_QLE.setText(f'channel_{c}')
                chName_QLE.textChanged.connect(self.checkChNames)

                txt = f'Channel {c} name:  '
                label = QLabel(txt)

                filenameDescLabel = QLabel(
                    f'<i>e.g., filename for channel {c}:  </i>'
                )

                chName = chName_QLE.text()
                rawFilename = self.elidedRawFilename()
                filenameLabel = QLabel(f"""
                    <p style=font-size:10px>{rawFilename}_{chName}.{ext}</p>
                """)
                filenameLabel.setToolTip(f'{self.rawFilename}_{chName}.{ext}')

                checkBox = QCheckBox('Save this channel')
                checkBox.setChecked(True)
                checkBox.stateChanged.connect(self.saveCh_checkBox_cb)

                self.channelNameLayouts[0].addWidget(label, alignment=Qt.AlignRight)
                self.channelNameLayouts[0].addWidget(
                    filenameDescLabel, alignment=Qt.AlignRight
                )
                self.channelNameLayouts[1].addWidget(chName_QLE)
                self.channelNameLayouts[1].addWidget(
                    filenameLabel, alignment=Qt.AlignCenter
                )

                self.channelNameLayouts[2].addWidget(checkBox)
                self.channelNameLayouts[2].addWidget(QLabel())

                showChannelDataButton = QPushButton()
                showChannelDataButton.setIcon(QIcon(":eye-plus.svg"))
                showChannelDataButton.clicked.connect(self.showChannelData)
                self.channelNameLayouts[3].addWidget(showChannelDataButton)
                if self.sampleImgData is None:
                    showChannelDataButton.setDisabled(True)

                self.chNames_QLEs.append(chName_QLE)
                self.saveChannels_QCBs.append(checkBox)
                self.filename_QLabels.append(filenameLabel)
                self.showChannelDataButtons.append(showChannelDataButton)

                emWavelen_DSB = QDoubleSpinBox()
                emWavelen_DSB.setAlignment(Qt.AlignCenter)
                emWavelen_DSB.setMaximum(2147483647.0)
                emWavelen_DSB.setSingleStep(0.001)
                emWavelen_DSB.setDecimals(2)
                emWavelen_DSB.setValue(500.0)
                unit = QLabel('nm')
                unit.setStyleSheet('font-size:13px; padding:5px 0px 2px 0px;')

                txt = f'Channel {c} emission wavelength:  '
                label = QLabel(txt)
                self.channelEmWLayouts[0].addWidget(label, alignment=Qt.AlignRight)
                self.channelEmWLayouts[1].addWidget(emWavelen_DSB)
                self.channelEmWLayouts[2].addWidget(unit)
                self.emWavelens_DSBs.append(emWavelen_DSB)
        else:
            for c in range(currentSizeC, currentSizeC+DeltaChannels):
                idx = (c-1)*2
                label1 = self.channelNameLayouts[0].itemAt(idx).widget()
                label2 = self.channelNameLayouts[0].itemAt(idx+1).widget()
                chName_QLE = self.channelNameLayouts[1].itemAt(idx).widget()
                filename_L = self.channelNameLayouts[1].itemAt(idx+1).widget()
                checkBox = self.channelNameLayouts[2].itemAt(idx).widget()
                dummyLabel = self.channelNameLayouts[2].itemAt(idx+1).widget()
                showButton = self.showChannelDataButtons[-1]
                showButton.clicked.disconnect()

                self.channelNameLayouts[0].removeWidget(label1)
                self.channelNameLayouts[0].removeWidget(label2)
                self.channelNameLayouts[1].removeWidget(chName_QLE)
                self.channelNameLayouts[1].removeWidget(filename_L)
                self.channelNameLayouts[2].removeWidget(checkBox)
                self.channelNameLayouts[2].removeWidget(dummyLabel)
                self.channelNameLayouts[3].removeWidget(showButton)

                self.chNames_QLEs.pop(-1)
                self.saveChannels_QCBs.pop(-1)
                self.filename_QLabels.pop(-1)
                self.showChannelDataButtons.pop(-1)

                label = self.channelEmWLayouts[0].itemAt(c-1).widget()
                emWavelen_DSB = self.channelEmWLayouts[1].itemAt(c-1).widget()
                unit = self.channelEmWLayouts[2].itemAt(c-1).widget()
                self.channelEmWLayouts[0].removeWidget(label)
                self.channelEmWLayouts[1].removeWidget(emWavelen_DSB)
                self.channelEmWLayouts[2].removeWidget(unit)
                self.emWavelens_DSBs.pop(-1)

                self.adjustSize()
    
    def ok_cb(self, event):
        areChNamesValid = self.checkChNames()
        if not areChNamesValid:
            err_msg = html_utils.paragraph(
                'Channel names <b>cannot be empty</b> or equal to each other.'
                '<br><br>'
                'Insert a unique text for each channel name.'
            )
            msg = widgets_message_box.myMessageBox()
            msg.critical(
               self, 'Invalid channel names', err_msg
            )
            return
        
        self.getValues()
        self.convertUnits()

        if self.sender() == self.trustButton:
            self.trust = True
        elif self.sender() == self.overWriteButton:
            self.overWrite = True

        self.cancel = False
        self.close()

    def getValues(self):
        self.LensNA = self.LensNA_DSB.value()
        self.SizeT = self.SizeT_SB.value()
        self.SizeZ = self.SizeZ_SB.value()
        self.SizeC = self.SizeC_SB.value()
        self.SizeS = self.SizeS_SB.value()
        self.timeRangeToSave = self.timeRangeToSaveWidget.range()
        self.TimeIncrement = self.TimeIncrement_DSB.value()
        self.PhysicalSizeX = self.PhysicalSizeX_DSB.value()
        self.PhysicalSizeY = self.PhysicalSizeY_DSB.value()
        self.PhysicalSizeZ = self.PhysicalSizeZ_DSB.value()
        self.to_h5 = self.to_h5_radiobutton.isChecked()
        if hasattr(self, 'posSelector'):
            self.selectedPos = self.posSelector.selectedItemsText()
        else:
            self.selectedPos = ['All Positions']
        self.chNames = []
        if hasattr(self, 'addImageName_QCB'):
            self.addImageName = self.addImageName_QCB.isChecked()
        else:
            self.addImageName = False
        self.saveChannels = []
        for LE, QCB in zip(self.chNames_QLEs, self.saveChannels_QCBs):
            s = LE.text()
            s = "".join(c if c.isalnum() or c=='_' or c=='' else '_' for c in s)
            trim_ = s.endswith('_')
            while trim_:
                s = s[:-1]
                trim_ = s.endswith('_')
            self.chNames.append(s)
            self.saveChannels.append(QCB.isChecked())
        self.emWavelens = [DSB.value() for DSB in self.emWavelens_DSBs]

    def convertUnits(self):
        timeUnit = self.TimeIncrementUnit_CB.currentText()
        if timeUnit == 'ms':
            self.TimeIncrement /= 1000
        elif timeUnit == 'minutes':
            self.TimeIncrement *= 60
        elif timeUnit == 'hours':
            self.TimeIncrement *= 3600

        PhysicalSizeUnit = self.PhysicalSizeUnit_CB.currentText()
        if timeUnit == 'nm':
            self.PhysicalSizeX /= 1000
            self.PhysicalSizeY /= 1000
            self.PhysicalSizeZ /= 1000
        elif timeUnit == 'mm':
            self.PhysicalSizeX *= 1000
            self.PhysicalSizeY *= 1000
            self.PhysicalSizeZ *= 1000
        elif timeUnit == 'cm':
            self.PhysicalSizeX *= 1e4
            self.PhysicalSizeY *= 1e4
            self.PhysicalSizeZ *= 1e4

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)
    
    def setSize(self):
        h = self.SizeS_SB.height()
        self.TimeIncrement_DSB.setMinimumHeight(h)

    def show(self, block=False):
        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        super().show()
        self.setSize()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()


class InitFijiMacroDialog(dialogs_base.QBaseDialog):
    def __init__(self, parent=None):
        self.cancel = True
        
        super().__init__(parent=parent)
        
        mainLayout = QVBoxLayout()
        
        infoLabel = QLabel(html_utils.paragraph(
            """ 
            Place all the <b>raw microscopy files in a folder without any other 
            file</b><br>
            and provide the following information:
            """
        ))
        mainLayout.addWidget(infoLabel)
        
        gridLayout = QGridLayout()       
         
        row = 0
        label = QLabel('Files internal structure: ')
        gridLayout.addWidget(label, row, 0)
        self.filesStructureCombobox = QComboBox()
        self.filesStructureCombobox.addItems([
            'Positions (aka "series") embedded in the file',
            'Positions (aka "series") separated, one for each file',
            'Positions (aka "series") and channels separated, one for each file'
        ])
        gridLayout.addWidget(self.filesStructureCombobox, row, 1)
        self.filesStructureCombobox.currentTextChanged.connect(
            self.fileStructureChanged
        )
        infoButton = widgets_push_buttons.infoPushButton()
        gridLayout.addWidget(infoButton, row, 2)
        infoButton.clicked.connect(self.showInfoFileStructure)
        
        row += 1
        label = QLabel('Folder with raw microscopy files: ')
        gridLayout.addWidget(label, row, 0)
        self.folderPathLineEdit = widgets_line_edits.ElidingLineEdit()
        gridLayout.addWidget(self.folderPathLineEdit, row, 1)
        browseButton = widgets_push_buttons.browseFileButton(openFolder=True)
        gridLayout.addWidget(browseButton, row, 2)
        browseButton.sigPathSelected.connect(
            partial(self.updateFolderPath, lineEdit=self.folderPathLineEdit)
        )
        self.folderPathLineEdit.textChanged.connect(self.srcFolderPathChanged)
        
        row += 1
        label = QLabel('Destination folder: ')
        gridLayout.addWidget(label, row, 0)
        self.dstfolderPathLineEdit = widgets_line_edits.ElidingLineEdit()
        gridLayout.addWidget(self.dstfolderPathLineEdit, row, 1)
        browseButton = widgets_push_buttons.browseFileButton(openFolder=True)
        gridLayout.addWidget(browseButton, row, 2)
        browseButton.sigPathSelected.connect(self.dstfolderPathLineEdit.setText)
        
        row += 1
        label = QLabel('Channel(s) name: ')
        gridLayout.addWidget(label, row, 0)
        self.channelNamesLineEdit = widgets_line_edits.alphaNumericLineEdit(
            additionalChars=' ,'
        )
        gridLayout.addWidget(self.channelNamesLineEdit, row, 1)
        checkButton = widgets_push_buttons.TestPushButton('Check')
        gridLayout.addWidget(checkButton, row, 3)
        checkButton.clicked.connect(self.checkChannelNames)
        checkButton.setDisabled(True)
        self.checkButton = checkButton
        infoButton = widgets_push_buttons.infoPushButton()
        gridLayout.addWidget(infoButton, row, 2)
        infoButton.clicked.connect(self.showInfoChannelName)
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        gridLayout.setColumnStretch(0, 0)
        gridLayout.setColumnStretch(1, 1)
        gridLayout.setColumnStretch(2, 0)
        gridLayout.setColumnStretch(3, 0)

        mainLayout.addLayout(gridLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def fileStructureChanged(self, text):
        self.checkButton.setDisabled(not 'channels separated' in text)
    
    def checkChannelNames(self, checked=False):
        proceed = self.validate()
        if not proceed:
            return
        
        src_folderpath = self.folderPath()
        channel_names = self.channelNames()
        extension = os.listdir(src_folderpath)[0].split('.')[-1]
        basenames = io.move_separate_channels_tiffs_to_pos_folders(
            src_folderpath, channel_names, get_only_basenames=True,
            extension=extension
        )
        pos_folders_texts = []
        for p, basename in enumerate(basenames):
            pos_folders_texts.append(f'Position_{p+1}: <code>{basename}</code>')

        pos_folders_html_list = html_utils.to_list(
            pos_folders_texts, ordered=True
        )
        text = html_utils.paragraph(
            'The following Position folders will be created based on the provided channel names:<br>'
            f'{pos_folders_html_list}'
        )
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.information(self, 'Position folders', text)

    def srcFolderPathChanged(self, text):
        if self.dstfolderPathLineEdit.text():
            return
        
        folderPath = self.folderPathLineEdit.text()
        self.dstfolderPathLineEdit.setText(folderPath)
    
    def showInfoFileStructure(self):
        txt = html_utils.paragraph("""
            Select whether the microscopy files contains multiple "series".<br><br>
            This typically depends on how you acquired the images at the 
            microscope, i.e., you generated multiple microscopy files 
            (e.g., snapshots), or you setup automatic acquisition of multiple 
            positions.
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.information(self, 'Files structure info', txt)
    
    def showInfoChannelName(self):
        txt = html_utils.paragraph("""
            Enter the channels name. Separate multiple channels with a comma.<br><br>
            The channel names will be used to name the individual TIFF files 
            (one for each channel).<br><br>
            If multiple channels are embedded in the microscopy file, make sure that you write the <b>channels in the right order</b>.<br> 
            If you are unsure, open the file in Fiji first
            and check the order of channels.<br><br>
            If the channels are already separated, make sure to write the 
            full channel name as it appears in the file, including capitalization and spaces.<br>
            For example, if the files are named "pos1_ch1.tif", "pos1_ch2.tif", etc., the channels names should be "ch1, ch2".<br><br>
            After providing the channel names, you can check that they are correct by clicking on the "Check" button next to the channel names field.<br>
            The number of Positions that will be created will be displayed alongside the basename.
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.information(self, 'Files structure info', txt)
    
    def updateFolderPath(self, path, lineEdit=''):
        for file in os.listdir(path):
            if not is_alphanumeric_filename(file):
                msg = widgets_message_box.myMessageBox(wrapText=False)
                txt = html_utils.paragraph(
                    f"""
                    The filename <b>{file}</b> contains <b>invalid 
                    characters</b>.<br><br>
                    Valid characters are letters, numbers, spaces, underscores 
                    and dashes.<br><br>
                    Please rename the file and try again.<br><br>
                    Thank you for your patience!
                    """
                )
                msg.critical(
                    self, 'Invalid filename', txt, path_to_browse=path
                )
                lineEdit.setText('')
                return
            
        lineEdit.setText(path)

    def warnPathEmpty(self, path_name):
        txt = html_utils.paragraph(f"""
            {path_name} <b>cannot be empty</b>.
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(self, 'Empty folder path', txt)
    
    def warnSelectedPathDoesNotExist(self, path):
        txt = html_utils.paragraph("""
            The selected path <b>does not exist</b>.<br><br>
            Selected path:
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(self, 'Folder path does not exist', txt, commands=(path,))
    
    def warnSelectedPathNotAFolder(self, path):
        txt = html_utils.paragraph("""
            The selected path is <b>not a folder</b>.<br><br>
            Selected path:
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(self, 'Selected path not a folder', txt, commands=(path,))
    
    def warnMultipleExtensionsPresent(self, path, extensions):
        txt = html_utils.paragraph(f"""
            The selected path <b>contains files with different extensions</b>.
            <br><br>
            Extensions present: <code>{extensions}</code><br><br>
            Please, make sure that all the files in the folder have the same 
            extension before proceeding.<br><br>
            Selected path:
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Multiple file extensions detected', txt, commands=(path,)
        )
    
    def warnChannelNamesEmpty(self):
        txt = html_utils.paragraph("""
            Channel(s) name <b>cannot be empty</b>.
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(self, 'Empty channel name', txt)
    
    def validate(self):
        path = self.folderPath()
        dst_path = self.dstfolderPathLineEdit.text()
        paths = {
            'Source folder': path,
            'Destination folder': dst_path,
        }
        for _path_name, _path in paths.items():
            if not _path:
                self.warnPathEmpty(_path_name)
                return False
            
            if not os.path.exists(_path):
                self.warnSelectedPathDoesNotExist(_path)
                return False
            
            if not os.path.isdir(_path):
                self.warnSelectedPathNotAFolder(_path)
                return False
        
        files = filesystem.listdir(path)
        extensions = set([os.path.splitext(file)[1] for file in files])
        if len(extensions) > 1:
            self.warnMultipleExtensionsPresent(path, extensions)
            return False

        if not self.channelNamesLineEdit.text():
            self.warnChannelNamesEmpty()
            return False
        
        return True
    
    def folderPath(self):
        return self.folderPathLineEdit.text()
    
    def channelNames(self):
        channel_names = self.channelNamesLineEdit.text().split(',')
        channel_names = [ch.strip() for ch in channel_names]
        return channel_names
    
    def ok_cb(self):
        proceed = self.validate()
        if not proceed:
            return
        
        self.selectedFolderPath = self.folderPath()
        self.filesStructure = self.filesStructureCombobox.currentText()
        is_multiple_files = self.filesStructure.find('separated') != -1
        is_separate_channels = 'channels separated' in self.filesStructure
        dst_folderpath = self.dstfolderPathLineEdit.text()
        self.init_macro_args = (
            self.folderPath(), 
            is_multiple_files, 
            is_separate_channels,
            dst_folderpath,
            self.channelNames(),
        )
        self.cancel = False
        self.close()
