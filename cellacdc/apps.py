import os
import sys
import re
import ast
from heapq import nlargest
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle, PathPatch, Path
import numpy as np
import scipy.interpolate
import tkinter as tk
import cv2
import traceback
from itertools import combinations
from collections import namedtuple
from natsort import natsorted
# from MyWidgets import Slider, Button, MyRadioButtons
from skimage.measure import label, regionprops
from functools import partial
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.exposure
import skimage.draw
import skimage.registration
import skimage.color
import skimage.segmentation
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import time

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon, QFontMetrics
from PyQt5.QtCore import Qt, QSize, QEvent, pyqtSignal, QEventLoop, QTimer
from PyQt5.QtWidgets import (
    QAction, QApplication, QMainWindow, QMenu, QLabel, QToolBar,
    QScrollBar, QWidget, QVBoxLayout, QLineEdit, QPushButton,
    QHBoxLayout, QDialog, QFormLayout, QListWidget, QAbstractItemView,
    QButtonGroup, QCheckBox, QSizePolicy, QComboBox, QSlider, QGridLayout,
    QSpinBox, QToolButton, QTableView, QTextBrowser, QDoubleSpinBox,
    QScrollArea, QFrame, QProgressBar, QGroupBox, QRadioButton,
    QDockWidget, QMessageBox, QStyle
)

from . import myutils, load, prompts, widgets, core
from . import is_mac
from . import qrc_resources
from . import is_win

pg.setConfigOption('imageAxisOrder', 'row-major') # best performance

class installJavaDialog(widgets.myMessageBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Install Java')
        self.setIcon('SP_MessageBoxWarning')

        txt_macOS = ("""
        <p style="font-size:13px">
            Your system doesn't have the <code>Java Development Kit</code>
            installed<br> and/or a C++ compiler.which is required for the installation of
            <code>javabridge</code><br><br>
            <b>Cell-ACDC is now going to install Java for you</b>.<br><br>
            <i><b>NOTE: After clicking on "Install", follow the instructions<br>
            on the terminal</b>. You will be asked to confirm steps and insert<br>
            your password to allow the installation.</i><br><br>
            If you prefer to do it manually, cancel the process<br>
            and follow the instructions below.
        </p>
        """)

        txt_windows = ("""
        <p style="font-size:13px">
            Unfortunately, installing pre-compiled version of
            <code>javabridge</code> <b>failed</b>.<br><br>
            Cell-ACDC is going to <b>try to compile it now</b>.<br><br>
            However, <b>before proceeding</b>, you need to install
            <code>Java Development Kit</code><br> and a <b>C++ compiler</b>.<br><br>
            <b>See instructions below on how to install it.</b>
        </p>
        """)

        if not is_win:
            self.instructionsButton = self.addButton('Show intructions...')
            self.instructionsButton.setCheckable(True)
            self.instructionsButton.disconnect()
            self.instructionsButton.clicked.connect(self.showInstructions)
            installButton = self.addButton('Install')
            installButton.disconnect()
            installButton.clicked.connect(self.installJavaMacOS)
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
        for t, text in enumerate(myutils.install_javabridge_instructions_text()):
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
                    copyButton.textToCopy = myutils.jdk_windows_url()
                    code_layout.addWidget(copyButton, alignment=Qt.AlignLeft)
                else:
                    copyButton.textToCopy = myutils.cpp_windows_url()
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
        self.layout.addWidget(
            self.scrollArea, self.currentRow, 1, alignment=Qt.AlignTop
        )

        # Stretch last row
        self.currentRow += 1
        self.layout.setRowStretch(self.currentRow, 1)

    def viewScreenshot(self, checked=False):
        self.screenShotWin = widgets.view_visualcpp_screenshot()
        self.screenShotWin.show()

    def addInstructionsMacOS(self):
        self.scrollArea = QScrollArea()
        _container = QWidget()
        _layout = QVBoxLayout()
        for t, text in enumerate(myutils.install_javabridge_instructions_text()):
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
                    copyButton.textToCopy = myutils._install_homebrew_command()
                else:
                    copyButton.textToCopy = myutils._brew_install_java_command()
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
        self.layout.addWidget(
            self.scrollArea, self.currentRow, 1, alignment=Qt.AlignTop
        )

        # Stretch last row
        self.currentRow += 1
        self.layout.setRowStretch(self.currentRow, 1)
        self.scrollArea.hide()

    def copyToClipboard(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(self.sender().textToCopy, mode=cb.Clipboard)
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

    def installJavaMacOS(self):
        import subprocess
        try:
            try:
                subprocess.check_call(['brew', 'update'])
            except Exception as e:
                subprocess.run(
                    myutils._install_homebrew_command(),
                    check=True, text=True, shell=True
                )
            subprocess.run(
                myutils._brew_install_java_command(),
                check=True, text=True, shell=True
            )
            self.close()
        except Exception as e:
            print('=======================')
            traceback.print_exc()
            print('=======================')
            msg = QMessageBox()
            err_msg = ("""
            <p style="font-size:13px">
                Automatic installation of Java failed.<br><br>
                Please, try manually by following the instructions provided
                with the "Show instructions..." button. Thanks
            </p>
            """)
            msg.critical(
               self, 'Java installation failed', err_msg, msg.Ok
            )

    def show(self):
        super().show()
        if not is_win:
            self.addInstructionsMacOS()
        else:
            self.addInstructionsWindows()
        self.move(self.pos().x(), 20)
        if is_win:
            self.resize(self.width(), self.height()+200)

class wandToleranceWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.slider = widgets.sliderWithSpinBox(title='Tolerance')
        self.slider.setMaximum(255)
        self.slider.layout.setColumnStretch(2, 21)

        self.setLayout(self.slider.layout)

class QDialogMetadataXML(QDialog):
    def __init__(
            self, title='Metadata',
            LensNA=1.0, DimensionOrder='', rawFilename='test',
            SizeT=1, SizeZ=1, SizeC=1, SizeS=1,
            TimeIncrement=180.0, TimeIncrementUnit='s',
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
        super().__init__(parent)
        self.setWindowTitle(title)
        font = QtGui.QFont()
        font.setPointSize(11)
        self.setFont(font)

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
        txt = 'Number of positions (S):  '
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.SizeS_SB, row, 1)

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
        self.DimensionOrder_QLE = QLineEdit()
        self.DimensionOrder_QLE.setAlignment(Qt.AlignCenter)
        self.DimensionOrder_QLE.setText(DimensionOrder)
        txt = 'Order of dimensions:  '
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.DimensionOrder_QLE, row, 1)

        row += 1
        self.SizeT_SB = QSpinBox()
        self.SizeT_SB.setAlignment(Qt.AlignCenter)
        self.SizeT_SB.setMinimum(1)
        self.SizeT_SB.setMaximum(2147483647)
        self.SizeT_SB.setValue(SizeT)
        txt = 'Number of frames (T):  '
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.SizeT_SB, row, 1)
        self.SizeT_SB.valueChanged.connect(self.hideShowTimeIncrement)

        row += 1
        self.SizeZ_SB = QSpinBox()
        self.SizeZ_SB.setAlignment(Qt.AlignCenter)
        self.SizeZ_SB.setMinimum(1)
        self.SizeZ_SB.setMaximum(2147483647)
        self.SizeZ_SB.setValue(SizeZ)
        txt = 'Number of z-slices in the z-stack (Z):  '
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.SizeZ_SB, row, 1)
        self.SizeZ_SB.valueChanged.connect(self.hideShowPhysicalSizeZ)

        row += 1
        self.TimeIncrement_DSB = QDoubleSpinBox()
        self.TimeIncrement_DSB.setAlignment(Qt.AlignCenter)
        self.TimeIncrement_DSB.setMaximum(2147483647.0)
        self.TimeIncrement_DSB.setSingleStep(1)
        self.TimeIncrement_DSB.setDecimals(3)
        self.TimeIncrement_DSB.setValue(TimeIncrement)
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

        if SizeT == 1:
            self.TimeIncrement_DSB.hide()
            self.TimeIncrementUnit_CB.hide()
            self.TimeIncrement_Label.hide()

        row += 1
        self.PhysicalSizeX_DSB = QDoubleSpinBox()
        self.PhysicalSizeX_DSB.setAlignment(Qt.AlignCenter)
        self.PhysicalSizeX_DSB.setMaximum(2147483647.0)
        self.PhysicalSizeX_DSB.setSingleStep(0.001)
        self.PhysicalSizeX_DSB.setDecimals(7)
        self.PhysicalSizeX_DSB.setValue(PhysicalSizeX)
        txt = 'Pixel width:  '
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
        txt = 'Pixel height:  '
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.PhysicalSizeY_DSB, row, 1)

        self.PhysicalSizeYUnit_Label = QLabel()
        self.PhysicalSizeYUnit_Label.setStyleSheet(
            'font-size:12px; padding:5px 0px 2px 0px;'
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
        txt = 'Voxel depth:  '
        self.PSZlabel = QLabel(txt)
        entriesLayout.addWidget(self.PSZlabel, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.PhysicalSizeZ_DSB, row, 1)

        self.PhysicalSizeZUnit_Label = QLabel()
        # padding: top, left, bottom, right
        self.PhysicalSizeZUnit_Label.setStyleSheet(
            'font-size:12px; padding:5px 0px 2px 0px;'
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
        txt = 'Number of channels:  '
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
            chName_QLE.setStyleSheet(
                'background: #FEF9C3'
            )
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
            filenameLabel = QLabel(f"""
                <p style=font-size:9pt>
                    {self.rawFilename}_{chName}.{ext}
                </p>
            """)

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
            if c == 0:
                addImageName_QCB = QCheckBox('Include image name')
                addImageName_QCB.stateChanged.connect(self.addImageName_cb)
                self.addImageName_QCB = addImageName_QCB
                self.channelNameLayouts[2].addWidget(addImageName_QCB)
            else:
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
            unit.setStyleSheet('font-size:12px; padding:5px 0px 2px 0px;')
            self.channelEmWLayouts[2].addWidget(unit)

        entriesLayout.setContentsMargins(0, 15, 0, 0)

        if rawDataStruct is None or rawDataStruct!=-1:
            okButton = QPushButton(' Ok ')
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
                "the metadata from the reader\n"
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

        cancelButton = QPushButton('Cancel')
        buttonsLayout.addWidget(cancelButton, 0, 2)
        buttonsLayout.setColumnStretch(0, 1)
        buttonsLayout.setColumnStretch(3, 1)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(entriesLayout)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch(1)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

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
            'background: #FEF9C3;'
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

    def updateFilename(self, idx):
        chName = self.chNames_QLEs[idx].text()
        chName = self.removeInvalidCharacters(chName)
        if self.rawDataStruct == 2:
            rawFilename = f'{self.rawFilename}_s{idx+1}'
        else:
            rawFilename = self.rawFilename

        ext = 'h5' if self.to_h5_radiobutton.isChecked() else 'tif'

        filenameLabel = self.filename_QLabels[idx]
        if self.addImageName_QCB.isChecked():
            self.ImageName = self.removeInvalidCharacters(self.ImageName)
            filename = (f"""
                <p style=font-size:9pt>
                    {rawFilename}_{self.ImageName}_{chName}.{ext}
                </p>
            """)
        else:
            filename = (f"""
                <p style=font-size:9pt>
                    {rawFilename}_{chName}.{ext}
                </p>
            """)
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
                LE1.setStyleSheet('background: #FEF9C3;')
                return areChNamesValid

            s1 = LE1.text()
            if not s1:
                self.setInvalidChName_StyleSheet(LE1)
                areChNamesValid = False
            else:
                LE1.setStyleSheet('background: #FEF9C3;')
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
                    LE1.setStyleSheet('background: #FEF9C3;')
                if not s2 and saveCh2:
                    self.setInvalidChName_StyleSheet(LE2)
                    areChNamesValid = False
                else:
                    LE2.setStyleSheet('background: #FEF9C3;')
                if s1 == s2 and saveCh1 and saveCh2:
                    self.setInvalidChName_StyleSheet(LE1)
                    self.setInvalidChName_StyleSheet(LE2)
                    areChNamesValid = False
            else:
                LE1.setStyleSheet('background: #FEF9C3;')
                LE2.setStyleSheet('background: #FEF9C3;')
        return areChNamesValid

    def hideShowTimeIncrement(self, value):
        if value > 1:
            self.TimeIncrement_DSB.show()
            self.TimeIncrementUnit_CB.show()
            self.TimeIncrement_Label.show()
        else:
            self.TimeIncrement_DSB.hide()
            self.TimeIncrementUnit_CB.hide()
            self.TimeIncrement_Label.hide()

    def hideShowPhysicalSizeZ(self, value):
        if value > 1:
            self.PSZlabel.show()
            self.PhysicalSizeZ_DSB.show()
            self.PhysicalSizeZUnit_Label.show()
        else:
            self.PSZlabel.hide()
            self.PhysicalSizeZ_DSB.hide()
            self.PhysicalSizeZUnit_Label.hide()

    def updatePSUnit(self, unit):
        self.PhysicalSizeYUnit_Label.setText(unit)
        self.PhysicalSizeZUnit_Label.setText(unit)

    def showChannelData(self, checked=False):
        idx = self.showChannelDataButtons.index(self.sender())
        posData = myutils.utilClass()
        posData.frame_i = 0
        posData.SizeT = self.SizeT_SB.value()
        posData.SizeZ = self.SizeZ_SB.value()
        posData.filename = f'{self.rawFilename}_C={idx}'
        posData.segmInfo_df = pd.DataFrame({
            'filename': [posData.filename],
            'frame_i': [0],
            'which_z_proj_gui': ['single z-slice'],
            'z_slice_used_gui': [int(posData.SizeZ/2)]
        }).set_index(['filename', 'frame_i'])
        path_li = os.path.normpath(self.rawFilePath).split(os.sep)
        posData.relPath = f'{f"{os.sep}".join(path_li[-3:1])}'
        posData.relPath = f'{posData.relPath}{os.sep}{posData.filename}'
        try:
            posData.img_data = [self.sampleImgData[idx]] # single frame data
        except Exception as e:
            traceback.print_exc()
            return

        self.imageViewer = imageViewer(posData=posData)
        self.imageViewer.update_img()
        self.imageViewer.show()


    def addRemoveChannels(self, value):
        currentSizeC = len(self.chNames_QLEs)
        DeltaChannels = abs(value-currentSizeC)
        ext = 'h5' if self.to_h5_radiobutton.isChecked() else 'tif'
        if value > currentSizeC:
            for c in range(currentSizeC, currentSizeC+DeltaChannels):
                chName_QLE = QLineEdit()
                chName_QLE.setStyleSheet(
                    'background: #FEF9C3'
                )
                chName_QLE.setAlignment(Qt.AlignCenter)
                chName_QLE.setText(f'channel_{c}')
                chName_QLE.textChanged.connect(self.checkChNames)

                txt = f'Channel {c} name:  '
                label = QLabel(txt)

                filenameDescLabel = QLabel(
                    f'<i>e.g., filename for channel {c}:  </i>'
                )

                chName = chName_QLE.text()
                filenameLabel = QLabel(f"""
                    <p style=font-size:9pt>
                        {self.rawFilename}_{chName}.{ext}
                    </p>
                """)

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
                unit.setStyleSheet('font-size:12px; padding:5px 0px 2px 0px;')

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
        DimensionOrder = self.DimensionOrder_QLE.text()
        m = re.findall('[TZCYXStzcyxs]', DimensionOrder)

        if len(m) != len(DimensionOrder) or not m:
            err_msg = (
                f'"{DimensionOrder}" is not a valid order of dimensions.\n\n'
                f'The letters available are {list("TZCYXS")} without spaces or punctuation.'
                '(e.g. ZYX)'
            )
            msg = QMessageBox()
            msg.critical(
               self, 'Invalid order of dimensions', err_msg, msg.Ok
            )
            return

        areChNamesValid = self.checkChNames()
        if not areChNamesValid:
            err_msg = (
                'Channel names cannot be empty or equal to each other.\n\n'
                'Insert a unique text for each channel name'
            )
            msg = QMessageBox()
            msg.critical(
               self, 'Invalid channel names', err_msg, msg.Ok
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
        self.DimensionOrder = self.DimensionOrder_QLE.text()
        self.SizeT = self.SizeT_SB.value()
        self.SizeZ = self.SizeZ_SB.value()
        self.SizeC = self.SizeC_SB.value()
        self.SizeS = self.SizeS_SB.value()
        self.TimeIncrement = self.TimeIncrement_DSB.value()
        self.PhysicalSizeX = self.PhysicalSizeX_DSB.value()
        self.PhysicalSizeY = self.PhysicalSizeY_DSB.value()
        self.PhysicalSizeZ = self.PhysicalSizeZ_DSB.value()
        self.to_h5 = self.to_h5_radiobutton.isChecked()
        self.chNames = []
        self.addImageName = self.addImageName_QCB.isChecked()
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

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class QDialogWorkerProcess(QDialog):
    def __init__(
            self, title='Progress', infoTxt='',
            showInnerPbar=False, pbarDesc='',
            parent=None
        ):
        self.workerFinished = False
        self.aborted = False
        self.clickCount = 0
        super().__init__(parent)

        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()
        pBarLayout = QGridLayout()

        if infoTxt:
            infoLabel = QLabel(infoTxt)
            mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        self.progressLabel = QLabel(pbarDesc)

        self.mainPbar = widgets.QProgressBarWithETA(self)
        self.mainPbar.setValue(0)
        pBarLayout.addWidget(self.mainPbar, 0, 0)
        pBarLayout.addWidget(self.mainPbar.ETA_label, 0, 1)

        self.innerPbar = widgets.QProgressBarWithETA(self)
        self.innerPbar.setValue(0)
        pBarLayout.addWidget(self.innerPbar, 1, 0)
        pBarLayout.addWidget(self.innerPbar.ETA_label, 1, 1)
        if showInnerPbar:
            self.innerPbar.show()
        else:
            self.innerPbar.hide()

        self.logConsole = widgets.QLogConsole()

        mainLayout.addWidget(self.progressLabel)
        mainLayout.addLayout(pBarLayout)
        mainLayout.addWidget(self.logConsole)

        self.setLayout(mainLayout)
        # self.setModal(True)

    def keyPressEvent(self, event):
        isCtrlAlt = event.modifiers() == (Qt.ControlModifier | Qt.AltModifier)
        if isCtrlAlt and event.key() == Qt.Key_C:
            doAbort = self.askAbort()
            if doAbort:
                self.aborted = True
                self.workerFinished = True
                self.close()

    def askAbort(self):
        msg = QMessageBox()
        txt = ("""
        <p style="font-size:9pt">
            Aborting with "Ctrl+Alt+C" is <b>not safe</b>.<br><br>
            The system status cannot be predicted and
            it will <b>require a restart</b>.<br><br>
            Are you sure you want to abort?
        </p>
        """)
        answer = msg.critical(
            self, 'Are you sure you want to abort?', txt, msg.Yes | msg.No
        )
        return answer == msg.Yes

    def closeEvent(self, event):
        if not self.workerFinished:
            event.ignore()

    def show(self, app):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        QDialog.show(self)
        screen = app.primaryScreen()
        screenWidth = screen.size().width()
        screenHeight = screen.size().height()
        parentGeometry = self.parent().geometry()
        mainWinLeft, mainWinWidth = parentGeometry.left(), parentGeometry.width()
        mainWinTop, mainWinHeight = parentGeometry.top(), parentGeometry.height()
        mainWinCenterX = int(mainWinLeft+mainWinWidth/2)
        mainWinCenterY = int(mainWinTop+mainWinHeight/2)

        width = int(screenWidth/3)
        height = int(screenHeight/3)
        left = int(mainWinCenterX - width/2)
        top = int(mainWinCenterY - height/2)

        self.setGeometry(left, top, width, height)

class QDialogCombobox(QDialog):
    def __init__(self, title, ComboBoxItems, informativeText,
                 CbLabel='Select value:  ', parent=None,
                 defaultChannelName=None, iconPixmap=None):
        self.cancel = True
        self.selectedItemText = ''
        self.selectedItemIdx = None
        super().__init__(parent=parent)
        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()
        infoLayout = QHBoxLayout()
        topLayout = QHBoxLayout()
        bottomLayout = QHBoxLayout()

        self.mainLayout = mainLayout

        if iconPixmap is not None:
            label = QLabel()
            # padding: top, left, bottom, right
            # label.setStyleSheet("padding:5px 0px 12px 0px;")
            label.setPixmap(iconPixmap)
            infoLayout.addWidget(label)

        if informativeText:
            infoLabel = QLabel(informativeText)
            infoLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        if CbLabel:
            label = QLabel(CbLabel)
            topLayout.addWidget(label, alignment=Qt.AlignRight)

        combobox = QComboBox()
        combobox.addItems(ComboBoxItems)
        if defaultChannelName is not None and defaultChannelName in ComboBoxItems:
            combobox.setCurrentText(defaultChannelName)
        self.ComboBox = combobox
        topLayout.addWidget(combobox)
        topLayout.setContentsMargins(0, 10, 0, 0)

        okButton = QPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)
        bottomLayout.addWidget(okButton, alignment=Qt.AlignRight)

        cancelButton = QPushButton('Cancel')
        bottomLayout.addWidget(cancelButton, alignment=Qt.AlignLeft)
        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(infoLayout)
        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        # self.setModal(True)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        self.loop = None

        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)

    def ok_cb(self, event):
        self.cancel = False
        self.selectedItemText = self.ComboBox.currentText()
        self.selectedItemIdx = self.ComboBox.currentIndex()
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        QDialog.show(self)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class QDialogListbox(QDialog):
    def __init__(
            self, title, text, items, cancelText='Cancel',
            multiSelection=True, parent=None,
            additionalButtons=()
        ):
        self.cancel = True
        super().__init__(parent)
        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()
        topLayout = QVBoxLayout()
        bottomLayout = QHBoxLayout()

        self.mainLayout = mainLayout

        label = QLabel(text)
        _font = QtGui.QFont()
        _font.setPointSize(11)
        label.setFont(_font)
        # padding: top, left, bottom, right
        label.setStyleSheet("padding:0px 0px 3px 0px;")
        topLayout.addWidget(label, alignment=Qt.AlignCenter)

        listBox = QListWidget()
        listBox.setFont(_font)
        listBox.addItems(items)
        if multiSelection:
            listBox.setSelectionMode(QAbstractItemView.ExtendedSelection)
        else:
            listBox.setSelectionMode(QAbstractItemView.SingleSelection)
        listBox.setCurrentRow(0)
        self.listBox = listBox
        listBox.itemDoubleClicked.connect(self.ok_cb)
        topLayout.addWidget(listBox)

        bottomLayout.addStretch(1)
        okButton = QPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)
        bottomLayout.addWidget(okButton)

        if additionalButtons:
            self._additionalButtons = []
            for button in additionalButtons:
                _button = QPushButton(button)
                self._additionalButtons.append(_button)
                bottomLayout.addWidget(_button)
                _button.clicked.connect(self.ok_cb)

        cancelButton = QPushButton(cancelText)
        # cancelButton.setShortcut(Qt.Key_Escape)
        bottomLayout.addWidget(cancelButton)
        bottomLayout.addStretch(1)
        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        # self.setModal(True)

    def ok_cb(self, event):
        self.clickedButton = self.sender()
        self.cancel = False
        selectedItems = self.listBox.selectedItems()
        self.selectedItemsText = [item.text() for item in selectedItems]
        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.selectedItemsText = None
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()

        horizontal_sb = self.listBox.horizontalScrollBar()
        while horizontal_sb.isVisible():
            self.resize(self.height(), self.width() + 10)

        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class selectTrackerGUI(QDialogListbox):
    def __init__(
            self, SizeT, currentFrameNo=1, parent=None
        ):
        trackers = myutils.get_list_of_trackers()
        super().__init__(
            'Select tracker', 'Select one of the following trackers',
            trackers, multiSelection=False, parent=parent
        )
        self.setWindowTitle('Select tracker')

        selectFramesContainer = QGroupBox()
        selectFramesLayout = QGridLayout()

        self.startFrame_SB = QSpinBox()
        self.startFrame_SB.setAlignment(Qt.AlignCenter)
        self.startFrame_SB.setMinimum(1)
        self.startFrame_SB.setMaximum(SizeT-1)
        self.startFrame_SB.setValue(currentFrameNo)

        self.stopFrame_SB = QSpinBox()
        self.stopFrame_SB.setAlignment(Qt.AlignCenter)
        self.stopFrame_SB.setMinimum(1)
        self.stopFrame_SB.setMaximum(SizeT)
        self.stopFrame_SB.setValue(SizeT)

        selectFramesLayout.addWidget(QLabel('Start frame n.'), 0, 0)
        selectFramesLayout.addWidget(self.startFrame_SB, 1, 0)

        selectFramesLayout.addWidget(QLabel('Stop frame n.'), 0, 1)
        selectFramesLayout.addWidget(self.stopFrame_SB, 1, 1)

        self.warningLabel = QLabel()
        palette = self.warningLabel.palette();
        palette.setColor(self.warningLabel.backgroundRole(), Qt.red);
        palette.setColor(self.warningLabel.foregroundRole(), Qt.red);
        self.warningLabel.setPalette(palette);
        selectFramesLayout.addWidget(
            self.warningLabel, 2, 0, 1, 2, alignment=Qt.AlignCenter
        )

        selectFramesContainer.setLayout(selectFramesLayout)

        self.mainLayout.insertWidget(1, selectFramesContainer)

        self.stopFrame_SB.valueChanged.connect(self._checkRange)

    def _checkRange(self):
        start = self.startFrame_SB.value()
        stop = self.stopFrame_SB.value()
        if stop <= start:
            self.warningLabel.setText(
                'stop frame smaller than start frame'
            )
        else:
            self.warningLabel.setText('')

    def ok_cb(self, event):
        if self.warningLabel.text():
            return
        else:
            self.startFrame = self.startFrame_SB.value()
            self.stopFrame = self.stopFrame_SB.value()
            QDialogListbox.ok_cb(self, event)


class QDialogAppendTextFilename(QDialog):
    def __init__(self, filename, ext, parent=None, font=None):
        super().__init__(parent)
        self.cancel = True
        filenameNOext, _ = os.path.splitext(filename)
        self.filenameNOext = filenameNOext
        if ext.find('.') == -1:
            ext = f'.{ext}'
        self.ext = ext

        self.setWindowTitle('Append text to file name')

        mainLayout = QVBoxLayout()
        formLayout = QFormLayout()
        buttonsLayout = QHBoxLayout()

        if font is not None:
            self.setFont(font)

        self.LE = QLineEdit()
        self.LE.setAlignment(Qt.AlignCenter)
        formLayout.addRow('Appended text', self.LE)
        self.LE.textChanged.connect(self.updateFinalFilename)

        self.finalName_label = QLabel(
            f'Final file name: "{filenameNOext}_{ext}"'
        )
        # padding: top, left, bottom, right
        self.finalName_label.setStyleSheet(
            'font-size:12px; padding:5px 0px 0px 0px;'
        )

        okButton = QPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = QPushButton('Cancel')

        buttonsLayout.addWidget(okButton, alignment=Qt.AlignRight)
        buttonsLayout.addWidget(cancelButton, alignment=Qt.AlignLeft)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(formLayout)
        mainLayout.addWidget(self.finalName_label, alignment=Qt.AlignCenter)
        mainLayout.addLayout(buttonsLayout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        self.formLayout = formLayout

        self.setLayout(mainLayout)
        # self.setModal(True)

    def updateFinalFilename(self, text):
        finalFilename = f'{self.filenameNOext}_{text}{self.ext}'
        self.finalName_label.setText(f'Final file name: "{finalFilename}"')

    def ok_cb(self, event):
        if not self.LE.text():
            err_msg = (
                'Appended name cannot be empty!'
            )
            msg = QMessageBox()
            msg.critical(
               self, 'Empty name', err_msg, msg.Ok
            )
            return
        self.cancel = False
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class QDialogEntriesWidget(QDialog):
    def __init__(self, entriesLabels, defaultTxts, winTitle='Input',
                 parent=None, font=None):
        self.cancel = True
        self.entriesTxt = []
        self.entriesLabels = entriesLabels
        self.QLEs = []
        super().__init__(parent)
        self.setWindowTitle(winTitle)

        mainLayout = QVBoxLayout()
        formLayout = QFormLayout()
        buttonsLayout = QHBoxLayout()

        if font is not None:
            self.setFont(font)

        for label, txt in zip(entriesLabels, defaultTxts):
            LE = QLineEdit()
            LE.setAlignment(Qt.AlignCenter)
            LE.setText(txt)
            formLayout.addRow(label, LE)
            self.QLEs.append(LE)

        okButton = QPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = QPushButton('Cancel')

        buttonsLayout.addWidget(okButton, alignment=Qt.AlignRight)
        buttonsLayout.addWidget(cancelButton, alignment=Qt.AlignLeft)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(formLayout)
        mainLayout.addLayout(buttonsLayout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        self.formLayout = formLayout

        self.setLayout(mainLayout)
        # self.setModal(True)

    def ok_cb(self, event):
        self.cancel = False
        self.entriesTxt = [self.formLayout.itemAt(i, 1).widget().text()
                           for i in range(len(self.entriesLabels))]
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class QDialogMetadata(QDialog):
    def __init__(self, SizeT, SizeZ, TimeIncrement,
                 PhysicalSizeZ, PhysicalSizeY, PhysicalSizeX,
                 ask_SizeT, ask_TimeIncrement, ask_PhysicalSizes,
                 parent=None, font=None, imgDataShape=None, posData=None,
                 singlePos=False):
        self.cancel = True
        self.ask_TimeIncrement = ask_TimeIncrement
        self.ask_PhysicalSizes = ask_PhysicalSizes
        self.imgDataShape = imgDataShape
        self.posData = posData
        super().__init__(parent)
        self.setWindowTitle('Image properties')

        mainLayout = QVBoxLayout()
        gridLayout = QGridLayout()
        # formLayout = QFormLayout()
        buttonsLayout = QGridLayout()

        if imgDataShape is not None:
            label = QLabel(
                f"""
                <p style="font-size:11pt">
                    <i>Image data shape</i> = <b>{imgDataShape}</b><br>
                </p>
                """)
            mainLayout.addWidget(label, alignment=Qt.AlignCenter)

        row = 0
        gridLayout.addWidget(
            QLabel('Number of frames (SizeT)'), row, 0, alignment=Qt.AlignRight
        )
        self.SizeT_SpinBox = QSpinBox()
        self.SizeT_SpinBox.setMinimum(1)
        self.SizeT_SpinBox.setMaximum(2147483647)
        if ask_SizeT:
            self.SizeT_SpinBox.setValue(SizeT)
        else:
            self.SizeT_SpinBox.setValue(1)
            self.SizeT_SpinBox.setDisabled(True)
        self.SizeT_SpinBox.setAlignment(Qt.AlignCenter)
        self.SizeT_SpinBox.valueChanged.connect(self.TimeIncrementShowHide)
        gridLayout.addWidget(self.SizeT_SpinBox, row, 1)

        row += 1
        gridLayout.addWidget(
            QLabel('Number of z-slices (SizeZ)'), row, 0, alignment=Qt.AlignRight
        )
        self.SizeZ_SpinBox = QSpinBox()
        self.SizeZ_SpinBox.setMinimum(1)
        self.SizeZ_SpinBox.setMaximum(2147483647)
        self.SizeZ_SpinBox.setValue(SizeZ)
        self.SizeZ_SpinBox.setAlignment(Qt.AlignCenter)
        self.SizeZ_SpinBox.valueChanged.connect(self.SizeZvalueChanged)
        gridLayout.addWidget(self.SizeZ_SpinBox, row, 1)

        row += 1
        self.TimeIncrementLabel = QLabel('Time interval (s)')
        gridLayout.addWidget(
            self.TimeIncrementLabel, row, 0, alignment=Qt.AlignRight
        )
        self.TimeIncrementSpinBox = QDoubleSpinBox()
        self.TimeIncrementSpinBox.setDecimals(7)
        self.TimeIncrementSpinBox.setMaximum(2147483647.0)
        self.TimeIncrementSpinBox.setValue(TimeIncrement)
        self.TimeIncrementSpinBox.setAlignment(Qt.AlignCenter)
        gridLayout.addWidget(self.TimeIncrementSpinBox, row, 1)

        if SizeT == 1 or not ask_TimeIncrement:
            self.TimeIncrementSpinBox.hide()
            self.TimeIncrementLabel.hide()

        row += 1
        self.PhysicalSizeZLabel = QLabel('Physical Size Z (um/pixel)')
        gridLayout.addWidget(
            self.PhysicalSizeZLabel, row, 0, alignment=Qt.AlignRight
        )
        self.PhysicalSizeZSpinBox = QDoubleSpinBox()
        self.PhysicalSizeZSpinBox.setDecimals(7)
        self.PhysicalSizeZSpinBox.setMaximum(2147483647.0)
        self.PhysicalSizeZSpinBox.setValue(PhysicalSizeZ)
        self.PhysicalSizeZSpinBox.setAlignment(Qt.AlignCenter)
        gridLayout.addWidget(self.PhysicalSizeZSpinBox, row, 1)

        if SizeZ==1 or not ask_PhysicalSizes:
            self.PhysicalSizeZSpinBox.hide()
            self.PhysicalSizeZLabel.hide()

        row += 1
        self.PhysicalSizeYLabel = QLabel('Physical Size Y (um/pixel)')
        gridLayout.addWidget(
            self.PhysicalSizeYLabel, row, 0, alignment=Qt.AlignRight
        )
        self.PhysicalSizeYSpinBox = QDoubleSpinBox()
        self.PhysicalSizeYSpinBox.setDecimals(7)
        self.PhysicalSizeYSpinBox.setMaximum(2147483647.0)
        self.PhysicalSizeYSpinBox.setValue(PhysicalSizeY)
        self.PhysicalSizeYSpinBox.setAlignment(Qt.AlignCenter)
        gridLayout.addWidget(self.PhysicalSizeYSpinBox, row, 1)

        if not ask_PhysicalSizes:
            self.PhysicalSizeYSpinBox.hide()
            self.PhysicalSizeYLabel.hide()

        row += 1
        self.PhysicalSizeXLabel = QLabel('Physical Size X (um/pixel)')
        gridLayout.addWidget(
            self.PhysicalSizeXLabel, row, 0, alignment=Qt.AlignRight
        )
        self.PhysicalSizeXSpinBox = QDoubleSpinBox()
        self.PhysicalSizeXSpinBox.setDecimals(7)
        self.PhysicalSizeXSpinBox.setMaximum(2147483647.0)
        self.PhysicalSizeXSpinBox.setValue(PhysicalSizeX)
        self.PhysicalSizeXSpinBox.setAlignment(Qt.AlignCenter)
        gridLayout.addWidget(self.PhysicalSizeXSpinBox, row, 1)

        if not ask_PhysicalSizes:
            self.PhysicalSizeXSpinBox.hide()
            self.PhysicalSizeXLabel.hide()

        self.SizeZvalueChanged(SizeZ)

        if singlePos:
            okTxt = 'Apply only to this Position'
        else:
            okTxt = 'Ok for loaded Positions'
        okButton = QPushButton(okTxt)
        okButton.setToolTip(
            'Save metadata only for current positionh'
        )
        okButton.setShortcut(Qt.Key_Enter)
        self.okButton = okButton

        if ask_TimeIncrement or ask_PhysicalSizes:
            okAllButton = QPushButton('Apply to ALL Positions')
            okAllButton.setToolTip(
                'Update existing Physical Sizes, Time interval, cell volume (fl), '
                'cell area (um^2), and time (s) for all the positions '
                'in the experiment folder.'
            )
            self.okAllButton = okAllButton

            selectButton = QPushButton('Select the Positions to be updated')
            selectButton.setToolTip(
                'Ask to select positions then update existing Physical Sizes, '
                'Time interval, cell volume (fl), cell area (um^2), and time (s)'
                'for selected positions.'
            )
            self.selectButton = selectButton
        else:
            self.okAllButton = None
            self.selectButton = None
            okButton.setText('Ok')

        cancelButton = QPushButton('Cancel')

        buttonsLayout.setColumnStretch(0, 1)
        buttonsLayout.addWidget(okButton, 0, 1)
        if ask_TimeIncrement or ask_PhysicalSizes:
            buttonsLayout.addWidget(okAllButton, 0, 2)
            buttonsLayout.addWidget(selectButton, 1, 1)
            buttonsLayout.addWidget(cancelButton, 1, 2)
        else:
            buttonsLayout.addWidget(cancelButton, 0, 2)
        buttonsLayout.setColumnStretch(3, 1)

        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        gridLayout.setColumnMinimumWidth(1, 100)
        mainLayout.addLayout(gridLayout)
        # mainLayout.addLayout(formLayout)
        mainLayout.addLayout(buttonsLayout)

        okButton.clicked.connect(self.ok_cb)
        if ask_TimeIncrement or ask_PhysicalSizes:
            okAllButton.clicked.connect(self.ok_cb)
            selectButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        self.setLayout(mainLayout)
        # self.setModal(True)

    def SizeZvalueChanged(self, val):
        if len(self.imgDataShape) < 3:
            return
        if val > 1 and self.imgDataShape is not None:
            maxSizeZ = self.imgDataShape[-3]
            self.SizeZ_SpinBox.setMaximum(maxSizeZ)
        else:
            self.SizeZ_SpinBox.setMaximum(2147483647)

        if not self.ask_PhysicalSizes:
            return
        if val > 1:
            self.PhysicalSizeZSpinBox.show()
            self.PhysicalSizeZLabel.show()
        else:
            self.PhysicalSizeZSpinBox.hide()
            self.PhysicalSizeZLabel.hide()

    def TimeIncrementShowHide(self, val):
        if not self.ask_TimeIncrement:
            return
        if val > 1:
            self.TimeIncrementSpinBox.show()
            self.TimeIncrementLabel.show()
        else:
            self.TimeIncrementSpinBox.hide()
            self.TimeIncrementLabel.hide()

    def ok_cb(self, event):
        self.cancel = False
        self.SizeT = self.SizeT_SpinBox.value()
        self.SizeZ = self.SizeZ_SpinBox.value()

        self.TimeIncrement = self.TimeIncrementSpinBox.value()
        self.PhysicalSizeX = self.PhysicalSizeXSpinBox.value()
        self.PhysicalSizeY = self.PhysicalSizeYSpinBox.value()
        self.PhysicalSizeZ = self.PhysicalSizeZSpinBox.value()
        valid4D = True
        valid3D = True
        valid2D = True
        if self.imgDataShape is None:
            self.close()
        elif len(self.imgDataShape) == 4:
            T, Z, Y, X = self.imgDataShape
            valid4D = self.SizeT == T and self.SizeZ == Z
        elif len(self.imgDataShape) == 3:
            TZ, Y, X = self.imgDataShape
            valid3D = self.SizeT == TZ or self.SizeZ == TZ
        elif len(self.imgDataShape) == 2:
            valid2D = self.SizeT == 1 and self.SizeZ == 1
        valid = all([valid4D, valid3D, valid2D])
        if not valid4D:
            txt = (f"""
            <p style="font-size:12px">
                You loaded <b>4D data</b>, hence the number of frames MUST be
                <b>{T}</b><br> nd the number of z-slices MUST be <b>{Z}</b>.<br><br>
                What do you want to do?
            </p>
            """)
        if not valid3D:
            txt = (f"""
            <p style="font-size:12px">
                You loaded <b>3D data</b>, hence either the number of frames is
                <b>{TZ}</b><br> or the number of z-slices can be <b>{TZ}</b>.<br><br>
                However, if the number of frames is greater than 1 then the<br>
                number of z-slices MUST be 1, and vice-versa.<br><br>
                What do you want to do?
            </p>
            """)

        if not valid2D:
            txt = (f"""
            <p style="font-size:12px">
                You loaded <b>2D data</b>, hence the number of frames MUST be <b>1</b>
                and the number of z-slices MUST be <b>1</b>.<br><br>
                What do you want to do?
            </p>
            """)

        if not valid:
            msg = QMessageBox(self)
            msg.setIcon(msg.Warning)
            msg.setWindowTitle('Invalid entries')
            msg.setText(txt)
            continueButton = QPushButton(
                f'Continue anyway'
            )
            cancelButton = QPushButton(
                f'Let me correct'
            )
            msg.addButton(continueButton, msg.YesRole)
            msg.addButton(cancelButton, msg.NoRole)
            msg.exec_()
            if msg.clickedButton() == cancelButton:
                return

        if self.posData is not None and self.sender() != self.okButton:
            exp_path = self.posData.exp_path
            pos_foldernames = natsorted(myutils.listdir(exp_path))
            pos_foldernames = [
                pos for pos in pos_foldernames
                if pos.find('Position_')!=-1
                and os.path.isdir(os.path.join(exp_path, pos))
            ]
            if self.sender() == self.selectButton:
                select_folder = load.select_exp_folder()
                select_folder.pos_foldernames = pos_foldernames
                select_folder.QtPrompt(
                    self, pos_foldernames, allow_abort=False, toggleMulti=True
                )
                pos_foldernames = select_folder.selected_pos
            for pos in pos_foldernames:
                images_path = os.path.join(exp_path, pos, 'Images')
                ls = myutils.listdir(images_path)
                search = [file for file in ls if file.find('metadata.csv')!=-1]
                metadata_df = None
                if search:
                    fileName = search[0]
                    metadata_csv_path = os.path.join(images_path, fileName)
                    metadata_df = pd.read_csv(
                        metadata_csv_path
                        ).set_index('Description')
                if metadata_df is not None:
                    metadata_df.at['TimeIncrement', 'values'] = self.TimeIncrement
                    metadata_df.at['PhysicalSizeZ', 'values'] = self.PhysicalSizeZ
                    metadata_df.at['PhysicalSizeY', 'values'] = self.PhysicalSizeY
                    metadata_df.at['PhysicalSizeX', 'values'] = self.PhysicalSizeX
                    metadata_df.to_csv(metadata_csv_path)

                search = [file for file in ls if file.find('acdc_output.csv')!=-1]
                acdc_df = None
                if search:
                    fileName = search[0]
                    acdc_df_path = os.path.join(images_path, fileName)
                    acdc_df = pd.read_csv(acdc_df_path)
                    yx_pxl_to_um2 = self.PhysicalSizeY*self.PhysicalSizeX
                    vox_to_fl = self.PhysicalSizeY*(self.PhysicalSizeX**2)
                    if 'cell_vol_fl' not in acdc_df.columns:
                        continue
                    acdc_df['cell_vol_fl'] = acdc_df['cell_vol_vox']*vox_to_fl
                    acdc_df['cell_area_um2'] = acdc_df['cell_area_pxl']*yx_pxl_to_um2
                    acdc_df['time_seconds'] = acdc_df['frame_i']*self.TimeIncrement
                    try:
                        acdc_df.to_csv(acdc_df_path, index=False)
                    except PermissionError:
                        err_msg = (
                            'The below file is open in another app '
                            '(Excel maybe?).\n\n'
                            f'{acdc_df_path}\n\n'
                            'Close file and then press "Ok".'
                        )
                        msg = QMessageBox()
                        msg.critical(self, 'Permission denied', err_msg, msg.Ok)
                        acdc_df.to_csv(acdc_df_path, index=False)

        elif self.sender() == self.selectButton:
            pass

        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class QCropZtool(QWidget):
    sigClose = pyqtSignal()
    sigZvalueChanged = pyqtSignal(str, int)
    sigReset = pyqtSignal()
    sigCrop = pyqtSignal()

    def __init__(self, SizeZ, parent=None):
        super().__init__(parent)

        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        self.SizeZ = SizeZ
        self.numDigits = len(str(self.SizeZ))

        self.setWindowTitle('Crop Z')

        layout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        self.lowerZscrollbar = QScrollBar(Qt.Horizontal)
        self.lowerZscrollbar.setMaximum(SizeZ)
        s = str(0).zfill(self.numDigits)
        self.lowerZscrollbar.label = QLabel(f'{s}/{SizeZ-1}')

        self.upperZscrollbar = QScrollBar(Qt.Horizontal)
        self.upperZscrollbar.setValue(SizeZ)
        self.upperZscrollbar.setMaximum(SizeZ)
        self.upperZscrollbar.label = QLabel(f'{SizeZ-1}/{SizeZ-1}')

        cancelButton = QPushButton('Cancel')
        cropButton = QPushButton('Crop and save')
        buttonsLayout.addWidget(cropButton)
        buttonsLayout.addWidget(cancelButton)

        layout.addWidget(
            QLabel('Lower z-slice  '), 0, 0, alignment=Qt.AlignRight
        )
        layout.addWidget(
            self.lowerZscrollbar.label, 0, 1, alignment=Qt.AlignRight
        )
        layout.addWidget(self.lowerZscrollbar, 0, 2)

        layout.addWidget(
            QLabel('Upper z-slice  '), 1, 0, alignment=Qt.AlignRight
        )
        layout.addWidget(
            self.upperZscrollbar.label, 1, 1, alignment=Qt.AlignRight
        )
        layout.addWidget(self.upperZscrollbar, 1, 2)

        layout.addLayout(buttonsLayout, 2, 2, alignment=Qt.AlignRight)

        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 0)
        layout.setColumnStretch(2, 10)

        self.setLayout(layout)

        # resetButton.clicked.connect(self.emitReset)
        cropButton.clicked.connect(self.emitCrop)
        cancelButton.clicked.connect(self.close)
        self.lowerZscrollbar.valueChanged.connect(self.ZvalueChanged)
        self.upperZscrollbar.valueChanged.connect(self.ZvalueChanged)

    def emitReset(self):
        self.sigReset.emit()

    def emitCrop(self):
        self.sigCrop.emit()

    def updateScrollbars(self, lower_z, upper_z):
        self.lowerZscrollbar.setValue(lower_z)
        self.upperZscrollbar.setValue(upper_z)

    def ZvalueChanged(self, value):
        which = 'lower' if self.sender() == self.lowerZscrollbar else 'upper'
        if which == 'lower' and value > self.upperZscrollbar.value()-2:
            self.lowerZscrollbar.setValue(self.upperZscrollbar.value()-2)
            return
        if which == 'upper' and value < self.lowerZscrollbar.value()+2:
            self.upperZscrollbar.setValue(self.lowerZscrollbar.value()+2)
            return

        s = str(value).zfill(self.numDigits)
        self.sender().label.setText(f'{s}/{self.SizeZ-1}')
        self.sigZvalueChanged.emit(which, value)

    def show(self):
        super().show()
        self.resize(self.width(), self.height())

    def closeEvent(self, event):
        self.sigClose.emit()

class gaussBlurDialog(QDialog):
    def __init__(self, mainWindow):
        super().__init__(mainWindow)
        self.cancel = True
        self.mainWindow = mainWindow

        posData = mainWindow.data[mainWindow.pos_i]
        items = [posData.filename]
        try:
            items.extend(list(posData.ol_data_dict.keys()))
        except Exception as e:
            pass

        self.keys = items

        self.setWindowTitle('Gaussian blur sigma')
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        mainLayout = QVBoxLayout()
        formLayout = QFormLayout()
        buttonsLayout = QHBoxLayout()

        self.channelsComboBox = QComboBox()
        self.channelsComboBox.addItems(items)
        self.channelsComboBox.setCurrentText(posData.manualContrastKey)
        mainLayout.addWidget(self.channelsComboBox)

        self.sigmaQDSB = QDoubleSpinBox()
        self.sigmaQDSB.setAlignment(Qt.AlignCenter)
        self.sigmaQDSB.setSingleStep(0.5)
        self.sigmaQDSB.setValue(1.0)
        formLayout.addRow('Gaussian filter sigma:  ', self.sigmaQDSB)
        formLayout.setContentsMargins(0, 10, 0, 10)

        self.sigmaSlider = QSlider(Qt.Horizontal)
        self.sigmaSlider.setMinimum(0)
        self.sigmaSlider.setMaximum(100)
        self.sigmaSlider.setValue(20)
        self.sigma = 1.0
        self.sigmaSlider.setTickPosition(QSlider.TicksBelow)
        self.sigmaSlider.setTickInterval(10)

        self.PreviewCheckBox = QCheckBox("Preview")
        self.PreviewCheckBox.setChecked(True)

        mainLayout.addLayout(formLayout)
        mainLayout.addWidget(self.sigmaSlider)
        mainLayout.addWidget(self.PreviewCheckBox)


        closeButton = QPushButton('Close')

        buttonsLayout.addWidget(closeButton, alignment=Qt.AlignCenter)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(buttonsLayout)

        self.PreviewCheckBox.clicked.connect(self.preview_cb)
        self.sigmaSlider.sliderMoved.connect(self.sigmaSliderMoved)
        self.sigmaQDSB.valueChanged.connect(self.sigmaQDSB_valueChanged)
        self.channelsComboBox.currentTextChanged.connect(self.apply)
        closeButton.clicked.connect(self.close)

        self.setLayout(mainLayout)

        self.apply()

    def preview_cb(self, checked):
        if not checked:
            self.restoreNonFiltered()
            self.mainWindow.updateALLimg(only_ax1=True, updateSharp=False)
        else:
            self.getData()
            self.apply()

    def getData(self):
        posData = self.mainWindow.data[self.mainWindow.pos_i]
        key = self.channelsComboBox.currentText()
        if key.find(self.mainWindow.user_ch_name) != -1:
            img = self.mainWindow.getImage()
            data = posData.img_data
        else:
            img = self.mainWindow.getOlImg(key)
            data = posData.ol_data[key]

        self.img = img
        self.frame_i = posData.frame_i
        self.segmSizeT = posData.segmSizeT
        self.imgData = data

    def getFilteredImg(self):
        img = skimage.filters.gaussian(self.img, sigma=self.sigma)
        if self.mainWindow.overlayButton.isChecked():
            key = self.channelsComboBox.currentText()
            img = self.mainWindow.getOverlayImg(
                fluoData=(img, key), setImg=False
            )
        else:
            img = self.mainWindow.getImageWithCmap(img=img)
        # img = self.mainWindow.normalizeIntensities(img)
        return img

    def apply(self):
        self.getData()
        img = self.getFilteredImg()
        if self.PreviewCheckBox.isChecked():
            self.mainWindow.img1.setImage(img)
            # h = self.mainWindow.img1.getHistogram()
            # self.mainWindow.hist.plot.setData(*h)

    def sigmaQDSB_valueChanged(self, val):
        self.sigma = val
        self.sigmaSlider.sliderMoved.disconnect()
        self.sigmaSlider.setSliderPosition(int(val*20))
        self.sigmaSlider.sliderMoved.connect(self.sigmaSliderMoved)
        self.apply()

    def sigmaSliderMoved(self, intVal):
        self.sigma = intVal/20
        self.sigmaQDSB.valueChanged.disconnect()
        self.sigmaQDSB.setValue(self.sigma)
        self.sigmaQDSB.valueChanged.connect(self.sigmaSliderMoved)
        self.apply()

    def closeEvent(self, event):
        self.mainWindow.gaussBlurAction.setChecked(False)
        self.mainWindow.updateALLimg(only_ax1=True, updateFilters=False)

class edgeDetectionDialog(QDialog):
    def __init__(self, mainWindow):
        super().__init__(mainWindow)
        self.cancel = True
        self.mainWindow = mainWindow

        if mainWindow is not None:
            posData = self.mainWindow.data[self.mainWindow.pos_i]
            items = [posData.filename]
        else:
            items = ['test']
        try:
            posData = self.mainWindow.data[self.mainWindow.pos_i]
            items.extend(list(posData.ol_data_dict.keys()))
        except Exception as e:
            pass

        self.keys = items

        self.setWindowTitle('Edge detection')
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        mainLayout = QVBoxLayout()
        paramsLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()


        channelCBLabel = QLabel('Channel:')
        mainLayout.addWidget(channelCBLabel)
        self.channelsComboBox = QComboBox()
        self.channelsComboBox.addItems(items)
        if mainWindow is not None:
            self.channelsComboBox.setCurrentText(posData.manualContrastKey)
        if not self.mainWindow.overlayButton.isChecked():
            self.channelsComboBox.setCurrentIndex(0)
        mainLayout.addWidget(self.channelsComboBox)

        row = 0
        sigmaQSLabel = QLabel('Blur:')
        paramsLayout.addWidget(sigmaQSLabel, row, 0)
        row += 1
        self.sigmaValLabel = QLabel('1.00')
        paramsLayout.addWidget(self.sigmaValLabel, row, 1)
        self.sigmaSlider = QSlider(Qt.Horizontal)
        self.sigmaSlider.setMinimum(1)
        self.sigmaSlider.setMaximum(100)
        self.sigmaSlider.setValue(20)
        self.sigma = 1.0
        self.sigmaSlider.setTickPosition(QSlider.TicksBelow)
        self.sigmaSlider.setTickInterval(10)
        paramsLayout.addWidget(self.sigmaSlider, row, 0)

        row += 1
        sharpQSLabel = QLabel('Sharpen:')
        # padding: top, left, bottom, right
        sharpQSLabel.setStyleSheet("font-size:12px; padding:5px 0px 0px 0px;")
        paramsLayout.addWidget(sharpQSLabel, row, 0)
        row += 1
        self.sharpValLabel = QLabel('5.00')
        paramsLayout.addWidget(self.sharpValLabel, row, 1)
        self.sharpSlider = QSlider(Qt.Horizontal)
        self.sharpSlider.setMinimum(1)
        self.sharpSlider.setMaximum(100)
        self.sharpSlider.setValue(50)
        self.radius = 5.0
        self.sharpSlider.setTickPosition(QSlider.TicksBelow)
        self.sharpSlider.setTickInterval(10)
        paramsLayout.addWidget(self.sharpSlider, row, 0)

        row += 1
        self.PreviewCheckBox = QCheckBox("Preview")
        self.PreviewCheckBox.setChecked(True)
        paramsLayout.addWidget(self.PreviewCheckBox, row, 0, 1, 2,
                               alignment=Qt.AlignCenter)


        closeButton = QPushButton('Close')

        buttonsLayout.addWidget(closeButton, alignment=Qt.AlignCenter)

        paramsLayout.setContentsMargins(0, 10, 0, 0)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(paramsLayout)
        mainLayout.addLayout(buttonsLayout)

        self.PreviewCheckBox.clicked.connect(self.preview_cb)
        self.sigmaSlider.sliderMoved.connect(self.sigmaSliderMoved)
        self.sharpSlider.sliderMoved.connect(self.sharpSliderMoved)
        self.channelsComboBox.currentTextChanged.connect(self.apply)
        closeButton.clicked.connect(self.close)

        self.setLayout(mainLayout)
        self.apply()

    def setSize(self):
        x = self.pos().x()
        y = self.pos().y()
        h = self.size().height()
        self.setGeometry(x, y, 300, h)

    def preview_cb(self, checked):
        if not checked:
            self.restoreNonFiltered()
            self.mainWindow.updateALLimg(only_ax1=True, updateSharp=False)
        else:
            self.getData()
            self.apply()

    def getData(self):
        key = self.channelsComboBox.currentText()
        posData = self.mainWindow.data[self.mainWindow.pos_i]
        if key.find(self.mainWindow.user_ch_name) != -1:
            img = self.mainWindow.getImage(normalizeIntens=False)
            data = posData.img_data
        else:
            img = self.mainWindow.getOlImg(key, normalizeIntens=False)
            data = posData.ol_data[key]

        if self.PreviewCheckBox.isChecked():
            self.img = skimage.exposure.equalize_adapthist(img)
            self.detectEdges()
        self.frame_i = posData.frame_i
        self.imgData = data

    def detectEdges(self):
        self.edge = skimage.filters.sobel(self.img)

    def getFilteredImg(self):
        img = self.edge.copy()
        # Blur
        img = skimage.filters.gaussian(img, sigma=self.sigma)
        # Sharpen
        img = img - skimage.filters.gaussian(img, sigma=self.radius)
        if self.mainWindow.overlayButton.isChecked():
            key = self.channelsComboBox.currentText()
            img = self.mainWindow.getOverlayImg(
                fluoData=(img, key), setImg=False
            )
        else:
            img = self.mainWindow.getImageWithCmap(img=img)
        return img

    def apply(self):
        self.getData()
        img = self.getFilteredImg()
        if self.PreviewCheckBox.isChecked():
            self.mainWindow.img1.setImage(img)
            # h = self.mainWindow.img1.getHistogram()
            # self.mainWindow.hist.plot.setData(*h)

    def sigmaSliderMoved(self, intVal):
        self.sigma = intVal/20
        self.sigmaValLabel.setText(f'{self.sigma:.2f}')
        self.apply()

    def sharpSliderMoved(self, intVal):
        self.radius = 10 - intVal/10
        if self.radius < 0.15:
            self.radius = 0.15
        self.sharpValLabel.setText(f'{intVal/10:.2f}')
        self.apply()

    def closeEvent(self, event):
        self.mainWindow.edgeDetectorAction.setChecked(False)
        self.mainWindow.updateALLimg(only_ax1=True, updateFilters=False)


class entropyFilterDialog(QDialog):
    def __init__(self, mainWindow):
        super().__init__(mainWindow)
        self.cancel = True
        self.mainWindow = mainWindow

        if mainWindow is not None:
            posData = self.mainWindow.data[self.mainWindow.pos_i]
            items = [posData.filename]
        else:
            items = ['test']
        try:
            posData = self.mainWindow.data[self.mainWindow.pos_i]
            items.extend(list(posData.ol_data_dict.keys()))
        except Exception as e:
            pass

        self.keys = items

        self.setWindowTitle('Edge detection')
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        mainLayout = QVBoxLayout()
        paramsLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()


        channelCBLabel = QLabel('Channel:')
        mainLayout.addWidget(channelCBLabel)
        self.channelsComboBox = QComboBox()
        self.channelsComboBox.addItems(items)
        if mainWindow is not None:
            self.channelsComboBox.setCurrentText(posData.manualContrastKey)
        mainLayout.addWidget(self.channelsComboBox)

        row = 0
        sigmaQSLabel = QLabel('Radius: ')
        paramsLayout.addWidget(sigmaQSLabel, row, 0)
        row += 1
        self.radiusValLabel = QLabel('10')
        paramsLayout.addWidget(self.radiusValLabel, row, 1)
        self.radiusSlider = QSlider(Qt.Horizontal)
        self.radiusSlider.setMinimum(1)
        self.radiusSlider.setMaximum(100)
        self.radiusSlider.setValue(10)
        self.radiusSlider.setTickPosition(QSlider.TicksBelow)
        self.radiusSlider.setTickInterval(10)
        paramsLayout.addWidget(self.radiusSlider, row, 0)

        row += 1
        self.PreviewCheckBox = QCheckBox("Preview")
        self.PreviewCheckBox.setChecked(True)
        paramsLayout.addWidget(self.PreviewCheckBox, row, 0, 1, 2,
                               alignment=Qt.AlignCenter)

        closeButton = QPushButton('Close')

        buttonsLayout.addWidget(closeButton, alignment=Qt.AlignCenter)

        paramsLayout.setContentsMargins(0, 10, 0, 0)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(paramsLayout)
        mainLayout.addLayout(buttonsLayout)

        self.PreviewCheckBox.clicked.connect(self.preview_cb)
        self.radiusSlider.sliderMoved.connect(self.radiusSliderMoved)
        self.channelsComboBox.currentTextChanged.connect(self.apply)
        closeButton.clicked.connect(self.close)

        self.setLayout(mainLayout)

        self.apply()

    def setSize(self):
        x = self.pos().x()
        y = self.pos().y()
        h = self.size().height()
        self.setGeometry(x, y, 300, h)

    def preview_cb(self, checked):
        if not checked:
            self.restoreNonFiltered()
            self.mainWindow.updateALLimg(only_ax1=True, updateSharp=False)
        else:
            self.getData()
            self.apply()

    def getData(self):
        key = self.channelsComboBox.currentText()
        posData = self.mainWindow.data[self.mainWindow.pos_i]
        if key.find(self.mainWindow.user_ch_name) != -1:
            img = self.mainWindow.getImage()
            data = posData.img_data
        else:
            img = self.mainWindow.getOlImg(key)
            data = posData.ol_data[key]
        self.img = skimage.img_as_ubyte(img)
        self.frame_i = posData.frame_i
        self.imgData = data

    def getFilteredImg(self):
        radius = self.radiusSlider.sliderPosition()
        selem = skimage.morphology.disk(radius)
        entropyImg = skimage.filters.rank.entropy(self.img, selem)
        if self.mainWindow.overlayButton.isChecked():
            key = self.channelsComboBox.currentText()
            img = self.mainWindow.getOverlayImg(
                fluoData=(entropyImg, key), setImg=False
            )
        else:
            img = self.mainWindow.getImageWithCmap(img=entropyImg)
        return img

    def apply(self):
        self.getData()
        img = self.getFilteredImg()
        if self.PreviewCheckBox.isChecked():
            self.mainWindow.img1.setImage(img)
            # h = self.mainWindow.img1.getHistogram()
            # self.mainWindow.hist.plot.setData(*h)

    def radiusSliderMoved(self, intVal):
        self.radiusValLabel.setText(f'{intVal}')
        self.apply()

    def closeEvent(self, event):
        self.mainWindow.entropyFilterAction.setChecked(False)
        self.mainWindow.updateALLimg(only_ax1=True, updateFilters=False)

class randomWalkerDialog(QDialog):
    def __init__(self, mainWindow):
        super().__init__(mainWindow)
        self.cancel = True
        self.mainWindow = mainWindow

        if mainWindow is not None:
            posData = self.mainWindow.data[self.mainWindow.pos_i]
            items = [posData.filename]
        else:
            items = ['test']
        try:
            posData = self.mainWindow.data[self.mainWindow.pos_i]
            items.extend(list(posData.ol_data_dict.keys()))
        except Exception as e:
            pass

        self.keys = items

        self.setWindowTitle('Random walker segmentation')

        self.colors = [self.mainWindow.RWbkgrColor,
                       self.mainWindow.RWforegrColor]

        mainLayout = QVBoxLayout()
        paramsLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        self.mainWindow.clearAllItems()

        row = 0
        paramsLayout.addWidget(QLabel('Background threshold:'), row, 0)
        row += 1
        self.bkgrThreshValLabel = QLabel('0.05')
        paramsLayout.addWidget(self.bkgrThreshValLabel, row, 1)
        self.bkgrThreshSlider = QSlider(Qt.Horizontal)
        self.bkgrThreshSlider.setMinimum(1)
        self.bkgrThreshSlider.setMaximum(100)
        self.bkgrThreshSlider.setValue(5)
        self.bkgrThreshSlider.setTickPosition(QSlider.TicksBelow)
        self.bkgrThreshSlider.setTickInterval(10)
        paramsLayout.addWidget(self.bkgrThreshSlider, row, 0)

        row += 1
        foregrQSLabel = QLabel('Foreground threshold:')
        # padding: top, left, bottom, right
        foregrQSLabel.setStyleSheet("font-size:12px; padding:5px 0px 0px 0px;")
        paramsLayout.addWidget(foregrQSLabel, row, 0)
        row += 1
        self.foregrThreshValLabel = QLabel('0.95')
        paramsLayout.addWidget(self.foregrThreshValLabel, row, 1)
        self.foregrThreshSlider = QSlider(Qt.Horizontal)
        self.foregrThreshSlider.setMinimum(1)
        self.foregrThreshSlider.setMaximum(100)
        self.foregrThreshSlider.setValue(95)
        self.foregrThreshSlider.setTickPosition(QSlider.TicksBelow)
        self.foregrThreshSlider.setTickInterval(10)
        paramsLayout.addWidget(self.foregrThreshSlider, row, 0)

        # Parameters link label
        row += 1
        url1 = 'https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_random_walker_segmentation.html'
        url2 = 'https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.random_walker'
        htmlTxt1 = f'<a href=\"{url1}">here</a>'
        htmlTxt2 = f'<a href=\"{url2}">here</a>'
        seeHereLabel = QLabel()
        seeHereLabel.setText(f'See {htmlTxt1} and {htmlTxt2} for details '
                              'about Random walker segmentation.')
        seeHereLabel.setTextFormat(Qt.RichText)
        seeHereLabel.setTextInteractionFlags(Qt.TextBrowserInteraction)
        seeHereLabel.setOpenExternalLinks(True)
        font = QtGui.QFont()
        font.setPointSize(11)
        seeHereLabel.setFont(font)
        seeHereLabel.setStyleSheet("padding:12px 0px 0px 0px;")
        paramsLayout.addWidget(seeHereLabel, row, 0, 1, 2)

        computeButton = QPushButton('Compute segmentation')
        closeButton = QPushButton('Close')

        buttonsLayout.addWidget(computeButton, alignment=Qt.AlignRight)
        buttonsLayout.addWidget(closeButton, alignment=Qt.AlignLeft)

        paramsLayout.setContentsMargins(0, 10, 0, 0)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(paramsLayout)
        mainLayout.addLayout(buttonsLayout)

        self.bkgrThreshSlider.sliderMoved.connect(self.bkgrSliderMoved)
        self.foregrThreshSlider.sliderMoved.connect(self.foregrSliderMoved)
        computeButton.clicked.connect(self.computeSegmAndPlot)
        closeButton.clicked.connect(self.close)

        self.setLayout(mainLayout)

        self.getImage()
        self.plotMarkers()

    def getImage(self):
        img = self.mainWindow.getDisplayedCellsImg()
        self.img = img/img.max()
        self.imgRGB = (skimage.color.gray2rgb(self.img)*255).astype(np.uint8)

    def setSize(self):
        x = self.pos().x()
        y = self.pos().y()
        h = self.size().height()
        w = self.size().width()
        if w < 400:
            w = 400
        self.setGeometry(x, y, w, h)

    def plotMarkers(self):
        imgMin, imgMax = self.computeMarkers()

        img = self.img

        imgRGB = self.imgRGB.copy()
        R, G, B = self.colors[0]
        imgRGB[:, :, 0][img < imgMin] = R
        imgRGB[:, :, 1][img < imgMin] = G
        imgRGB[:, :, 2][img < imgMin] = B
        R, G, B = self.colors[1]
        imgRGB[:, :, 0][img > imgMax] = R
        imgRGB[:, :, 1][img > imgMax] = G
        imgRGB[:, :, 2][img > imgMax] = B

        self.mainWindow.img1.setImage(imgRGB)

    def computeMarkers(self):
        bkgrThresh = self.bkgrThreshSlider.sliderPosition()/100
        foregrThresh = self.foregrThreshSlider.sliderPosition()/100
        img = self.img
        self.markers = np.zeros(img.shape, np.uint8)
        imgRange = img.max() - img.min()
        imgMin = img.min() + imgRange*bkgrThresh
        imgMax = img.min() + imgRange*foregrThresh
        self.markers[img < imgMin] = 1
        self.markers[img > imgMax] = 2
        return imgMin, imgMax

    def computeSegm(self, checked=True):
        self.mainWindow.storeUndoRedoStates(False)
        self.mainWindow.titleLabel.setText(
            'Randomly walking around... ', color='w')
        img = self.img
        img = skimage.exposure.rescale_intensity(img)
        t0 = time.time()
        lab = skimage.segmentation.random_walker(img, self.markers, mode='bf')
        lab = skimage.measure.label(lab>1)
        t1 = time.time()
        if len(np.unique(lab)) > 2:
            skimage.morphology.remove_small_objects(lab, min_size=5,
                                                    in_place=True)
        posData = self.mainWindow.data[self.mainWindow.pos_i]
        posData.lab = lab
        return t1-t0

    def computeSegmAndPlot(self):
        deltaT = self.computeSegm()

        posData = self.mainWindow.data[self.mainWindow.pos_i]

        self.mainWindow.update_rp()
        self.mainWindow.tracking(enforce=True)
        self.mainWindow.updateALLimg()
        self.mainWindow.warnEditingWithCca_df('Random Walker segmentation')
        txt = f'Random Walker segmentation computed in {deltaT:.3f} s'
        print('-----------------')
        print(txt)
        print('=================')
        # self.mainWindow.titleLabel.setText(txt, color='g')

    def bkgrSliderMoved(self, intVal):
        self.bkgrThreshValLabel.setText(f'{intVal/100:.2f}')
        self.plotMarkers()

    def foregrSliderMoved(self, intVal):
        self.foregrThreshValLabel.setText(f'{intVal/100:.2f}')
        self.plotMarkers()

    def closeEvent(self, event):
        self.mainWindow.segmModel = ''
        self.mainWindow.updateALLimg()

class FutureFramesAction_QDialog(QDialog):
    def __init__(self, frame_i, last_tracked_i, change_txt,
                 applyTrackingB=False, parent=None):
        self.decision = None
        self.last_tracked_i = last_tracked_i
        super().__init__(parent)
        self.setWindowTitle('Future frames action?')

        mainLayout = QVBoxLayout()
        txtLayout = QVBoxLayout()
        doNotShowLayout = QVBoxLayout()
        buttonsLayout = QVBoxLayout()

        txt = (
            'You already visited/checked future frames '
            f'{frame_i+1}-{last_tracked_i}.\n\n'
            f'The requested "{change_txt}" change might result in\n'
            'NON-correct segmentation/tracking for those frames.\n'
        )

        txtLabel = QLabel(txt)
        _font = QtGui.QFont()
        _font.setPointSize(11)
        _font.setBold(True)
        txtLabel.setFont(_font)
        txtLabel.setAlignment(Qt.AlignCenter)
        # padding: top, left, bottom, right
        txtLabel.setStyleSheet("padding:0px 0px 3px 0px;")
        txtLayout.addWidget(txtLabel, alignment=Qt.AlignCenter)

        infoTxt = (
           f'  Choose one of the following options:\n\n'
           f'      1.  Apply the "{change_txt}" only to this frame and re-initialize\n'
            '          the future frames to the segmentation file present\n'
            '          on the hard drive.\n'
            '      2.  Apply only to this frame and keep the future frames as they are.\n'
            '      3.  Apply the change to ALL visited/checked future frames.\n'
            # '      4.  Apply the change to a specific range of future frames.\n'

        )

        if applyTrackingB:
            infoTxt = (
                f'{infoTxt}'
                '4. Repeat ONLY tracking for all future frames (RECOMMENDED)'
            )

        infotxtLabel = QLabel(infoTxt)
        _font = QtGui.QFont()
        _font.setPointSize(11)
        infotxtLabel.setFont(_font)

        infotxtLabel.setStyleSheet("padding:0px 0px 3px 0px;")
        txtLayout.addWidget(infotxtLabel, alignment=Qt.AlignCenter)

        noteTxt = (
            'NOTE: Only changes applied to current frame can be undone.\n'
            '      Changes applied to future frames CANNOT be UNDONE!\n'
        )

        noteTxtLabel = QLabel(noteTxt)
        _font = QtGui.QFont()
        _font.setPointSize(11)
        _font.setBold(True)
        noteTxtLabel.setFont(_font)
        # padding: top, left, bottom, right
        noteTxtLabel.setStyleSheet("padding:0px 0px 3px 0px;")
        txtLayout.addWidget(noteTxtLabel, alignment=Qt.AlignCenter)

        # Do not show this message again checkbox
        doNotShowCheckbox = QCheckBox(
            'Remember my choice and do not show this message again')
        doNotShowLayout.addWidget(doNotShowCheckbox)
        doNotShowLayout.setContentsMargins(50, 0, 0, 10)
        self.doNotShowCheckbox = doNotShowCheckbox

        apply_and_reinit_b = QPushButton(
                    'Apply only to this frame and re-initialize future frames')

        self.apply_and_reinit_b = apply_and_reinit_b
        buttonsLayout.addWidget(apply_and_reinit_b)

        apply_and_NOTreinit_b = QPushButton(
                'Apply only to this frame and keep future frames as they are')
        self.apply_and_NOTreinit_b = apply_and_NOTreinit_b
        buttonsLayout.addWidget(apply_and_NOTreinit_b)

        apply_to_all_b = QPushButton(
                    'Apply to all future frames')
        self.apply_to_all_b = apply_to_all_b
        buttonsLayout.addWidget(apply_to_all_b)

        self.applyTrackingButton = None
        if applyTrackingB:
            applyTrackingButton = QPushButton(
                        'Repeat ONLY tracking for all future frames')
            self.applyTrackingButton = applyTrackingButton
            buttonsLayout.addWidget(applyTrackingButton)

        apply_to_range_b = QPushButton(
                    'Apply only to a range of future frames')
        self.apply_to_range_b = apply_to_range_b
        # buttonsLayout.addWidget(apply_to_range_b)

        buttonsLayout.setContentsMargins(20, 0, 20, 0)

        self.formLayout = QFormLayout()
        self.OkRangeLayout = QVBoxLayout()
        self.OkRangeButton = QPushButton('Ok')
        self.OkRangeLayout.addWidget(self.OkRangeButton)

        ButtonsGroup = QButtonGroup(self)
        ButtonsGroup.addButton(apply_and_reinit_b)
        ButtonsGroup.addButton(apply_and_NOTreinit_b)
        if applyTrackingB:
            ButtonsGroup.addButton(applyTrackingButton)
        ButtonsGroup.addButton(apply_to_all_b)
        ButtonsGroup.addButton(apply_to_range_b)
        ButtonsGroup.addButton(self.OkRangeButton)

        mainLayout.addLayout(txtLayout)
        mainLayout.addLayout(doNotShowLayout)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addLayout(self.formLayout)
        self.mainLayout = mainLayout
        self.setLayout(mainLayout)

        # Connect events
        ButtonsGroup.buttonClicked.connect(self.buttonClicked)

        # self.setModal(True)

    def buttonClicked(self, button):
        if button == self.apply_and_reinit_b:
            self.decision = 'apply_and_reinit'
            self.endFrame_i = None
            self.close()
        elif button == self.apply_and_NOTreinit_b:
            self.decision = 'apply_and_NOTreinit'
            self.endFrame_i = None
            self.close()
        elif button == self.apply_to_all_b:
            self.decision = 'apply_to_all'
            self.endFrame_i = self.last_tracked_i
            self.close()
        elif button == self.applyTrackingButton:
            self.decision = 'only_tracking'
            self.endFrame_i = self.last_tracked_i
            self.close()
        elif button == self.apply_to_range_b:
            endFrame_LineEntry = QLineEdit()
            self.formLayout.addRow('Apply until frame: ',
                                   endFrame_LineEntry)
            endFrame_LineEntry.setText(f'{self.last_tracked_i}')
            endFrame_LineEntry.setAlignment(Qt.AlignCenter)
            self.formLayout.setContentsMargins(100, 10, 100, 0)

            self.mainLayout.addLayout(self.OkRangeLayout)
            self.OkRangeLayout.setContentsMargins(150, 0, 150, 0)

            self.endRangeFrame_i = int(endFrame_LineEntry.text())
        elif button == self.OkRangeButton:
            self.decision = 'apply_to_range'
            self.endFrame_i = self.endRangeFrame_i
            self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()


class postProcessSegmParams(QGroupBox):
    def __init__(self, title, useSliders=False, parent=None, maxSize=None):
        QGroupBox.__init__(self, title, parent)
        self.useSliders = useSliders

        layout = QGridLayout()

        row = 0
        label = QLabel("Minimum area (pixels): ")
        layout.addWidget(label, row, 0, alignment=Qt.AlignRight)

        if useSliders:
            minSize_SB = widgets.sliderWithSpinBox()
            minSize_SB.setMinimum(1)
            minSize_SB.setMaximum(200)
            minSize_SB.setValue(5)
        else:
            minSize_SB = QSpinBox()
            minSize_SB.setAlignment(Qt.AlignCenter)
            minSize_SB.setMinimum(1)
            minSize_SB.setMaximum(2147483647)
            minSize_SB.setValue(5)

        layout.addWidget(minSize_SB, row, 1)
        self.minSize_SB = minSize_SB

        row += 1
        label = QLabel("Minimum solidity (0-1): ")
        layout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        if useSliders:
            minSolidity_DSB = widgets.sliderWithSpinBox(normalize=True)
            minSolidity_DSB.setMaximum(100)
            minSolidity_DSB.setValue(0.5)
        else:
            minSolidity_DSB = QDoubleSpinBox()
            minSolidity_DSB.setAlignment(Qt.AlignCenter)
            minSolidity_DSB.setMinimum(0)
            minSolidity_DSB.setMaximum(1)
            minSolidity_DSB.setValue(0.5)
            minSolidity_DSB.setSingleStep(0.1)

        minSolidity_DSB.setToolTip(
            'Solidity is a measure of convexity. A solidity of 1 means '
            'that the shape is fully convex (i.e., equal to the convex hull).\n '
            'As solidity approaches 0 the object is more concave.\n'
            'Write 0 for ignoring this parameter.'
        )

        layout.addWidget(minSolidity_DSB, row, 1)
        self.minSolidity_DSB = minSolidity_DSB

        row += 1
        label = QLabel("Max elongation (1=circle):")
        layout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        if useSliders:
            maxElongation_DSB = widgets.sliderWithSpinBox(isFloat=True)
            maxElongation_DSB.setMinimum(0)
            maxElongation_DSB.setMaximum(100)
            maxElongation_DSB.setValue(3)
            maxElongation_DSB.setSingleStep(0.5)
        else:
            maxElongation_DSB = QDoubleSpinBox()
            maxElongation_DSB.setAlignment(Qt.AlignCenter)
            maxElongation_DSB.setMinimum(0)
            maxElongation_DSB.setMaximum(2147483647.0)
            maxElongation_DSB.setValue(3)
            maxElongation_DSB.setDecimals(1)
            maxElongation_DSB.setSingleStep(1.0)

        maxElongation_DSB.setToolTip(
            'Elongation is the ratio between major and minor axis lengths.\n'
            'An elongation of 1 is like a circle.\n'
            'Write 0 for ignoring this parameter.'
        )

        layout.addWidget(maxElongation_DSB, row, 1)
        self.maxElongation_DSB = maxElongation_DSB

        self.setLayout(layout)

class postProcessSegmDialog(QDialog):
    sigClosed = pyqtSignal()

    def __init__(self, mainWin=None):
        super().__init__(mainWin)
        self.cancel = True
        self.mainWin = mainWin

        self.setWindowTitle('Post-processing segmentation parameters')
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        mainLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        artefactsGroupBox = postProcessSegmParams(
            'Post-processing parameters', useSliders=True
        )

        self.minSize_SB = artefactsGroupBox.minSize_SB
        self.minSolidity_DSB = artefactsGroupBox.minSolidity_DSB
        self.maxElongation_DSB = artefactsGroupBox.maxElongation_DSB

        self.minSize_SB.valueChanged.connect(self.applyPostProcessing)
        self.minSolidity_DSB.valueChanged.connect(self.applyPostProcessing)
        self.maxElongation_DSB.valueChanged.connect(self.applyPostProcessing)

        self.minSize_SB.editingFinished.connect(self.onEditingFinished)
        self.minSolidity_DSB.editingFinished.connect(self.onEditingFinished)
        self.maxElongation_DSB.editingFinished.connect(self.onEditingFinished)

        okButton = QPushButton('Ok')
        cancelButton = QPushButton('Cancel')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(okButton)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.setContentsMargins(0,10,0,0)

        mainLayout.addWidget(artefactsGroupBox)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        font = QtGui.QFont()
        font.setPointSize(11)
        self.setFont(font)

        if mainWin is not None:
            self.setPosData()
            self.applyPostProcessing(0)

    def setPosData(self):
        if self.mainWin is None:
            return

        self.mainWin.storeUndoRedoStates(False)
        self.posData = self.mainWin.data[self.mainWin.pos_i]
        self.origLab = self.posData.lab.copy()

    def applyPostProcessing(self, value):
        if self.mainWin is None:
            return

        minSize = self.minSize_SB.value()
        minSolidity = self.minSolidity_DSB.value()
        maxElongation = self.maxElongation_DSB.value()

        self.mainWin.warnEditingWithCca_df('post-processing segmentation mask')

        self.posData.lab, delIDs = core.remove_artefacts(
            self.origLab.copy(),
            min_solidity=minSolidity,
            min_area=minSize,
            max_elongation=maxElongation,
            return_delIDs=True
        )

        self.mainWin.clearItems_IDs(delIDs)
        self.mainWin.setImageImg2()

    def onEditingFinished(self):
        if self.mainWin is None:
            return

        self.mainWin.update_rp()
        self.mainWin.updateALLimg()

    def ok_cb(self):
        self.cancel = False
        self.mainWin.update_rp()
        self.mainWin.store_data()
        self.mainWin.updateALLimg()
        self.close()

    def cancel_cb(self):
        if self.mainWin is not None:
            self.posData.lab = self.origLab
            self.mainWin.update_rp()
            self.mainWin.updateALLimg()
        self.close()

    def show(self):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        QDialog.show(self)
        self.resize(int(self.width()*1.5), self.height())

    def closeEvent(self, event):
        self.sigClosed.emit()

class imageViewer(QMainWindow):
    """Main Window."""

    def __init__(
            self, parent=None, posData=None, button_toUncheck=None,
            spinBox=None
        ):
        self.button_toUncheck = button_toUncheck
        self.parent = parent
        self.posData = posData
        self.spinBox = spinBox
        """Initializer."""
        super().__init__(parent)

        if posData is None:
            posData = self.parent.data[self.parent.pos_i]
        self.posData = posData

        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createToolBars()

        self.gui_createStatusBar()

        self.gui_createGraphics()

        self.gui_connectImgActions()

        self.gui_createImgWidgets()
        self.gui_connectActions()

        mainContainer = QtGui.QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QtGui.QGridLayout()
        mainLayout.addWidget(self.graphLayout, 0, 0, 1, 1)
        mainLayout.addLayout(self.img_Widglayout, 1, 0)

        mainContainer.setLayout(mainLayout)

        self.frame_i = posData.frame_i
        self.num_frames = posData.SizeT
        self.setWindowTitle(f"Cell-ACDC - {posData.relPath}")

    def gui_createActions(self):
        # File actions
        self.exitAction = QAction("&Exit", self)

        # Toolbar actions
        self.prevAction = QAction(QIcon(":arrow-left.svg"),
                                        "Previous frame", self)
        self.nextAction = QAction(QIcon(":arrow-right.svg"),
                                        "Next Frame", self)
        self.jumpForwardAction = QAction(QIcon(":arrow-up.svg"),
                                        "Jump to 10 frames ahead", self)
        self.jumpBackwardAction = QAction(QIcon(":arrow-down.svg"),
                                        "Jump to 10 frames back", self)
        self.prevAction.setShortcut("left")
        self.nextAction.setShortcut("right")
        self.jumpForwardAction.setShortcut("up")
        self.jumpBackwardAction.setShortcut("down")

    def gui_createMenuBar(self):
        menuBar = self.menuBar()
        # File menu
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)
        # fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.exitAction)

    def gui_createToolBars(self):
        toolbarSize = 30

        editToolBar = QToolBar("Edit", self)
        editToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(editToolBar)

        editToolBar.addAction(self.prevAction)
        editToolBar.addAction(self.nextAction)
        editToolBar.addAction(self.jumpBackwardAction)
        editToolBar.addAction(self.jumpForwardAction)

    def gui_connectActions(self):
        self.exitAction.triggered.connect(self.close)
        self.prevAction.triggered.connect(self.prev_frame)
        self.nextAction.triggered.connect(self.next_frame)
        self.jumpForwardAction.triggered.connect(self.skip10ahead_frames)
        self.jumpBackwardAction.triggered.connect(self.skip10back_frames)

    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        # Temporary message
        self.statusbar.showMessage("Ready", 3000)
        # Permanent widget
        self.wcLabel = QLabel(f"")
        self.statusbar.addPermanentWidget(self.wcLabel)

    def gui_createGraphics(self):
        self.graphLayout = pg.GraphicsLayoutWidget()

        # Plot Item container for image
        self.Plot = pg.PlotItem()
        self.Plot.invertY(True)
        self.Plot.setAspectLocked(True)
        self.Plot.hideAxis('bottom')
        self.Plot.hideAxis('left')
        self.graphLayout.addItem(self.Plot, row=1, col=1)

        # Image Item
        self.img = pg.ImageItem(np.zeros((512,512)))
        self.Plot.addItem(self.img)

        #Image histogram
        hist = pg.HistogramLUTItem()
        self.hist = hist
        hist.setImageItem(self.img)
        self.graphLayout.addItem(hist, row=1, col=0)

        # Current frame text
        self.frameLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.frameLabel.setText(' ')
        self.graphLayout.addItem(self.frameLabel, row=2, col=0, colspan=2)

    def gui_connectImgActions(self):
        self.img.hoverEvent = self.gui_hoverEventImg

    def gui_createImgWidgets(self):
        if self.posData is None:
            posData = self.parent.data[self.parent.pos_i]
        else:
            posData = self.posData
        self.img_Widglayout = QtGui.QGridLayout()

        # Frames scrollbar
        self.framesScrollBar = QScrollBar(Qt.Horizontal)
        # self.framesScrollBar.setFixedHeight(20)
        self.framesScrollBar.setMinimum(1)
        self.framesScrollBar.setMaximum(posData.SizeT)
        t_label = QLabel('frame  ')
        _font = QtGui.QFont()
        _font.setPointSize(11)
        t_label.setFont(_font)
        self.img_Widglayout.addWidget(
                t_label, 0, 0, alignment=Qt.AlignRight)
        self.img_Widglayout.addWidget(
                self.framesScrollBar, 0, 1, 1, 20)
        self.framesScrollBar.valueChanged.connect(self.framesScrollBarMoved)

        # z-slice scrollbar
        self.zSliceScrollBar = QScrollBar(Qt.Horizontal)
        # self.zSliceScrollBar.setFixedHeight(20)
        self.zSliceScrollBar.setMaximum(self.posData.SizeZ-1)
        _z_label = QLabel('z-slice  ')
        _font = QtGui.QFont()
        _font.setPointSize(11)
        _z_label.setFont(_font)
        self.z_label = _z_label
        self.img_Widglayout.addWidget(_z_label, 1, 0, alignment=Qt.AlignCenter)
        self.img_Widglayout.addWidget(self.zSliceScrollBar, 1, 1, 1, 20)

        if self.posData.SizeZ == 1:
            self.zSliceScrollBar.setDisabled(True)
            self.zSliceScrollBar.setVisible(False)
            _z_label.setVisible(False)

        self.img_Widglayout.setContentsMargins(100, 0, 50, 0)
        self.zSliceScrollBar.sliderMoved.connect(self.update_z_slice)

    def framesScrollBarMoved(self, frame_n):
        self.frame_i = frame_n-1
        if self.spinBox is not None:
            self.spinBox.setValue(frame_n)
        self.update_img()

    def gui_hoverEventImg(self, event):
        # Update x, y, value label bottom right
        try:
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            _img = self.img.image
            Y, X = _img.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                self.wcLabel.setText(f'(x={x:.2f}, y={y:.2f}, value={val:.2f})')
            else:
                self.wcLabel.setText(f'')
        except Exception as e:
            self.wcLabel.setText(f'')

    def next_frame(self):
        if self.frame_i < self.num_frames-1:
            self.frame_i += 1
        else:
            self.frame_i = 0
        self.update_img()

    def prev_frame(self):
        if self.frame_i > 0:
            self.frame_i -= 1
        else:
            self.frame_i = self.num_frames-1
        self.update_img()

    def skip10ahead_frames(self):
        if self.frame_i < self.num_frames-10:
            self.frame_i += 10
        else:
            self.frame_i = 0
        self.update_img()

    def skip10back_frames(self):
        if self.frame_i > 9:
            self.frame_i -= 10
        else:
            self.frame_i = self.num_frames-1
        self.update_img()

    def update_z_slice(self, z):
        if self.posData is None:
            posData = self.parent.data[self.parent.pos_i]
        else:
            posData = self.posData
            idx = (posData.filename, posData.frame_i)
            posData.segmInfo_df.at[idx, 'z_slice_used_gui'] = z
        self.update_img()

    def getImage(self):
        posData = self.posData
        frame_i = self.frame_i
        if posData.SizeZ > 1:
            idx = (posData.filename, frame_i)
            z = posData.segmInfo_df.at[idx, 'z_slice_used_gui']
            zProjHow = posData.segmInfo_df.at[idx, 'which_z_proj_gui']
            img = posData.img_data[frame_i]
            if zProjHow == 'single z-slice':
                self.zSliceScrollBar.setSliderPosition(z)
                self.z_label.setText(f'z-slice  {z+1:02}/{posData.SizeZ}')
                img = img[z].copy()
            elif zProjHow == 'max z-projection':
                img = img.max(axis=0).copy()
            elif zProjHow == 'mean z-projection':
                img = img.mean(axis=0).copy()
            elif zProjHow == 'median z-proj.':
                img = np.median(img, axis=0).copy()
        else:
            img = posData.img_data[frame_i].copy()
        return img


    def update_img(self):
        self.frameLabel.setText(
                 f'Current frame = {self.frame_i+1}/{self.num_frames}')
        if self.parent is None:
            img = self.getImage()
        else:
            img = self.parent.getImage(frame_i=self.frame_i)
        self.img.setImage(img)
        self.framesScrollBar.setSliderPosition(self.frame_i+1)

    def closeEvent(self, event):
        if self.button_toUncheck is not None:
            self.button_toUncheck.setChecked(False)

    def show(self, left=None, top=None):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        QMainWindow.show(self)
        if left is not None and top is not None:
            self.setGeometry(left, top, 850, 800)

class editCcaTableWidget(QDialog):
    def __init__(
            self, cca_df, title='Edit cell cycle annotations', parent=None
        ):
        self.inputCca_df = cca_df
        self.cancel = True
        self.cca_df = None

        super().__init__(parent)
        self.setWindowTitle(title)

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        # Layouts
        mainLayout = QVBoxLayout()
        tableLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()
        self.scrollArea = QScrollArea()
        self.viewBox = QWidget()

        # Header labels
        col = 0
        row = 0
        IDsLabel = QLabel('Cell ID')
        AC = Qt.AlignCenter
        IDsLabel.setAlignment(AC)
        tableLayout.addWidget(IDsLabel, 0, col, alignment=AC)

        col += 1
        ccsLabel = QLabel('Cell cycle stage')
        ccsLabel.setAlignment(Qt.AlignCenter)
        tableLayout.addWidget(ccsLabel, 0, col, alignment=AC)

        col += 1
        genNumLabel = QLabel('Generation number')
        genNumLabel.setAlignment(Qt.AlignCenter)
        tableLayout.addWidget(genNumLabel, 0, col, alignment=AC)
        genNumColWidth = genNumLabel.sizeHint().width()

        col += 1
        relIDLabel = QLabel('Relative ID')
        relIDLabel.setAlignment(Qt.AlignCenter)
        tableLayout.addWidget(relIDLabel, 0, col, alignment=AC)

        col += 1
        relationshipLabel = QLabel('Relationship')
        relationshipLabel.setAlignment(Qt.AlignCenter)
        tableLayout.addWidget(relationshipLabel, 0, col, alignment=AC)

        col += 1
        emergFrameLabel = QLabel('Emerging frame num.')
        emergFrameLabel.setAlignment(Qt.AlignCenter)
        tableLayout.addWidget(emergFrameLabel, 0, col, alignment=AC)

        col += 1
        divitionFrameLabel = QLabel('Division frame num.')
        divitionFrameLabel.setAlignment(Qt.AlignCenter)
        tableLayout.addWidget(divitionFrameLabel, 0, col, alignment=AC)

        col += 1
        historyKnownLabel = QLabel('Is history known?')
        historyKnownLabel.setAlignment(Qt.AlignCenter)
        tableLayout.addWidget(historyKnownLabel, 0, col, alignment=AC)

        tableLayout.setHorizontalSpacing(20)
        self.tableLayout = tableLayout

        # Add buttons
        okButton = QPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = QPushButton('Cancel')
        buttonsLayout.addWidget(okButton)
        buttonsLayout.addWidget(cancelButton)

        # Scroll area properties
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setFrameStyle(QFrame.NoFrame)
        self.scrollArea.setWidgetResizable(True)

        # Add layouts
        self.viewBox.setLayout(tableLayout)
        self.scrollArea.setWidget(self.viewBox)
        mainLayout.addWidget(self.scrollArea)
        mainLayout.addLayout(buttonsLayout)

        # Populate table Layout
        IDs = cca_df.index
        self.IDs = IDs.to_list()
        relIDsOptions = [str(ID) for ID in IDs]
        relIDsOptions.insert(0, '-1')
        self.IDlabels = []
        self.ccsComboBoxes = []
        self.genNumSpinBoxes = []
        self.relIDComboBoxes = []
        self.relationshipComboBoxes = []
        self.emergFrameSpinBoxes = []
        self.divisFrameSpinBoxes = []
        self.emergFrameSpinPrevValues = []
        self.divisFrameSpinPrevValues = []
        self.historyKnownCheckBoxes = []
        for row, ID in enumerate(IDs):
            col = 0
            IDlabel = QLabel(f'{ID}')
            IDlabel.setAlignment(Qt.AlignCenter)
            tableLayout.addWidget(IDlabel, row+1, col, alignment=AC)
            self.IDlabels.append(IDlabel)

            col += 1
            ccsComboBox = QComboBox()
            ccsComboBox.setFocusPolicy(Qt.StrongFocus)
            ccsComboBox.installEventFilter(self)
            ccsComboBox.addItems(['G1', 'S/G2/M'])
            ccsValue = cca_df.at[ID, 'cell_cycle_stage']
            if ccsValue == 'S':
                ccsValue = 'S/G2/M'
            ccsComboBox.setCurrentText(ccsValue)
            tableLayout.addWidget(ccsComboBox, row+1, col, alignment=AC)
            self.ccsComboBoxes.append(ccsComboBox)
            ccsComboBox.activated.connect(self.clearComboboxFocus)

            col += 1
            genNumSpinBox = QSpinBox()
            genNumSpinBox.setFocusPolicy(Qt.StrongFocus)
            genNumSpinBox.installEventFilter(self)
            genNumSpinBox.setValue(2)
            genNumSpinBox.setAlignment(Qt.AlignCenter)
            genNumSpinBox.setFixedWidth(int(genNumColWidth*2/3))
            genNumSpinBox.setValue(cca_df.at[ID, 'generation_num'])
            tableLayout.addWidget(genNumSpinBox, row+1, col, alignment=AC)
            self.genNumSpinBoxes.append(genNumSpinBox)

            col += 1
            relIDComboBox = QComboBox()
            relIDComboBox.setFocusPolicy(Qt.StrongFocus)
            relIDComboBox.installEventFilter(self)
            relIDComboBox.addItems(relIDsOptions)
            relIDComboBox.setCurrentText(str(cca_df.at[ID, 'relative_ID']))
            tableLayout.addWidget(relIDComboBox, row+1, col)
            self.relIDComboBoxes.append(relIDComboBox)
            relIDComboBox.currentIndexChanged.connect(self.setRelID)
            relIDComboBox.activated.connect(self.clearComboboxFocus)


            col += 1
            relationshipComboBox = QComboBox()
            relationshipComboBox.setFocusPolicy(Qt.StrongFocus)
            relationshipComboBox.installEventFilter(self)
            relationshipComboBox.addItems(['mother', 'bud'])
            relationshipComboBox.setCurrentText(cca_df.at[ID, 'relationship'])
            tableLayout.addWidget(relationshipComboBox, row+1, col)
            self.relationshipComboBoxes.append(relationshipComboBox)
            relationshipComboBox.currentIndexChanged.connect(
                                                self.relationshipChanged_cb)
            relationshipComboBox.activated.connect(self.clearComboboxFocus)

            col += 1
            emergFrameSpinBox = QSpinBox()
            emergFrameSpinBox.setFocusPolicy(Qt.StrongFocus)
            emergFrameSpinBox.installEventFilter(self)
            emergFrameSpinBox.setMinimum(-1)
            emergFrameSpinBox.setValue(-1)
            emergFrameSpinBox.setAlignment(Qt.AlignCenter)
            emergFrameSpinBox.setFixedWidth(int(genNumColWidth*2/3))
            emergFrame_i = cca_df.at[ID, 'emerg_frame_i']
            val = emergFrame_i+1 if emergFrame_i>=0 else -1
            emergFrameSpinBox.setValue(val)
            tableLayout.addWidget(emergFrameSpinBox, row+1, col, alignment=AC)
            self.emergFrameSpinBoxes.append(emergFrameSpinBox)
            self.emergFrameSpinPrevValues.append(emergFrameSpinBox.value())
            emergFrameSpinBox.valueChanged.connect(self.skip0emergFrame)


            col += 1
            divisFrameSpinBox = QSpinBox()
            divisFrameSpinBox.setFocusPolicy(Qt.StrongFocus)
            divisFrameSpinBox.installEventFilter(self)
            divisFrameSpinBox.setMinimum(-1)
            divisFrameSpinBox.setValue(-1)
            divisFrameSpinBox.setAlignment(Qt.AlignCenter)
            divisFrameSpinBox.setFixedWidth(int(genNumColWidth*2/3))
            divisFrame_i = cca_df.at[ID, 'division_frame_i']
            val = divisFrame_i+1 if divisFrame_i>=0 else -1
            divisFrameSpinBox.setValue(val)
            tableLayout.addWidget(divisFrameSpinBox, row+1, col, alignment=AC)
            self.divisFrameSpinBoxes.append(divisFrameSpinBox)
            self.divisFrameSpinPrevValues.append(divisFrameSpinBox.value())
            emergFrameSpinBox.valueChanged.connect(self.skip0divisFrame)

            col += 1
            HistoryCheckBox = QCheckBox()
            HistoryCheckBox.setChecked(bool(cca_df.at[ID, 'is_history_known']))
            tableLayout.addWidget(HistoryCheckBox, row+1, col, alignment=AC)
            self.historyKnownCheckBoxes.append(HistoryCheckBox)

        # Contents margins
        buttonsLayout.setContentsMargins(200, 15, 200, 15)

        self.setLayout(mainLayout)

        # Connect to events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        # self.setModal(True)

    def setRelID(self, itemIndex):
        idx = self.relIDComboBoxes.index(self.sender())
        relID = self.sender().currentText()
        IDofRelID = self.IDs[idx]
        relIDidx = self.IDs.index(int(relID))
        relIDComboBox = self.relIDComboBoxes[relIDidx]
        relIDComboBox.setCurrentText(str(IDofRelID))

    def skip0emergFrame(self, value):
        idx = self.emergFrameSpinBoxes.index(self.sender())
        prevVal = self.emergFrameSpinPrevValues[idx]
        if value == 0 and value > prevVal:
            self.sender().setValue(1)
            self.emergFrameSpinPrevValues[idx] = 1
        elif value == 0 and value < prevVal:
            self.sender().setValue(-1)
            self.emergFrameSpinPrevValues[idx] = -1

    def skip0divisFrame(self, value):
        idx = self.divisFrameSpinBoxes.index(self.sender())
        prevVal = self.divisFrameSpinPrevValues[idx]
        if value == 0 and value > prevVal:
            self.sender().setValue(1)
            self.divisFrameSpinPrevValues[idx] = 1
        elif value == 0 and value < prevVal:
            self.sender().setValue(-1)
            self.divisFrameSpinPrevValues[idx] = -1

    def relationshipChanged_cb(self, itemIndex):
        idx = self.relationshipComboBoxes.index(self.sender())
        ccs = self.sender().currentText()
        if ccs == 'bud':
            self.ccsComboBoxes[idx].setCurrentText('S/G2/M')
            self.genNumSpinBoxes[idx].setValue(0)

    def getCca_df(self):
        ccsValues = [var.currentText() for var in self.ccsComboBoxes]
        ccsValues = [val if val=='G1' else 'S' for val in ccsValues]
        genNumValues = [var.value() for var in self.genNumSpinBoxes]
        relIDValues = [int(var.currentText()) for var in self.relIDComboBoxes]
        relatValues = [var.currentText() for var in self.relationshipComboBoxes]
        emergFrameValues = [var.value()-1 if var.value()>0 else -1
                            for var in self.emergFrameSpinBoxes]
        divisFrameValues = [var.value()-1 if var.value()>0 else -1
                            for var in self.divisFrameSpinBoxes]
        historyValues = [var.isChecked() for var in self.historyKnownCheckBoxes]
        check_rel = [ID == relID for ID, relID in zip(self.IDs, relIDValues)]
        # Buds in S phase must have 0 as number of cycles
        check_buds_S = [ccs=='S' and rel_ship=='bud' and not numc==0
                        for ccs, rel_ship, numc
                        in zip(ccsValues, relatValues, genNumValues)]
        # Mother cells must have at least 1 as number of cycles if history known
        check_mothers = [rel_ship=='mother' and not numc>=1
                         if is_history_known else False
                         for rel_ship, numc, is_history_known
                         in zip(relatValues, genNumValues, historyValues)]
        # Buds cannot be in G1
        check_buds_G1 = [ccs=='G1' and rel_ship=='bud'
                         for ccs, rel_ship
                         in zip(ccsValues, relatValues)]
        # The number of cells in S phase must be half mothers and half buds
        num_moth_S = len([0 for ccs, rel_ship in zip(ccsValues, relatValues)
                            if ccs=='S' and rel_ship=='mother'])
        num_bud_S = len([0 for ccs, rel_ship in zip(ccsValues, relatValues)
                            if ccs=='S' and rel_ship=='bud'])
        # Cells in S phase cannot have -1 as relative's ID
        check_relID_S = [ccs=='S' and relID==-1
                         for ccs, relID
                         in zip(ccsValues, relIDValues)]
        if any(check_rel):
            QMessageBox().critical(self,
                    'Cell ID = Relative\'s ID', 'Some cells are '
                    'mother or bud of itself. Make sure that the Relative\'s ID'
                    ' is different from the Cell ID!',
                    QMessageBox.Ok)
            return None
        elif any(check_buds_S):
            QMessageBox().critical(self,
                'Bud in S/G2/M not in 0 Generation number',
                'Some buds '
                'in S phase do not have 0 as Generation number!\n'
                'Buds in S phase must have 0 as "Generation number"',
                QMessageBox.Ok)
            return None
        elif any(check_mothers):
            QMessageBox().critical(self,
                'Mother not in >=1 Generation number',
                'Some mother cells do not have >=1 as "Generation number"!\n'
                'Mothers MUST have >1 "Generation number"',
                QMessageBox.Ok)
            return None
        elif any(check_buds_G1):
            QMessageBox().critical(self,
                'Buds in G1!',
                'Some buds are in G1 phase!\n'
                'Buds MUST be in S/G2/M phase',
                QMessageBox.Ok)
            return None
        elif num_moth_S != num_bud_S:
            QMessageBox().critical(self,
                'Number of mothers-buds mismatch!',
                f'There are {num_moth_S} mother cells in "S/G2/M" phase,'
                f'but there are {num_bud_S} bud cells.\n\n'
                'The number of mothers and buds in "S/G2/M" '
                'phase must be equal!',
                QMessageBox.Ok)
            return None
        elif any(check_relID_S):
            QMessageBox().critical(self,
                'Relative\'s ID of cells in S/G2/M = -1',
                'Some cells are in "S/G2/M" phase but have -1 as Relative\'s ID!\n'
                'Cells in "S/G2/M" phase must have an existing '
                'ID as Relative\'s ID!',
                QMessageBox.Ok)
            return None
        else:
            corrected_assignment = self.inputCca_df['corrected_assignment']
            cca_df = pd.DataFrame({
                                'cell_cycle_stage': ccsValues,
                                'generation_num': genNumValues,
                                'relative_ID': relIDValues,
                                'relationship': relatValues,
                                'emerg_frame_i': emergFrameValues,
                                'division_frame_i': divisFrameValues,
                                'is_history_known': historyValues,
                                'corrected_assignment': corrected_assignment},
                                index=self.IDs)
            cca_df.index.name = 'Cell_ID'
            d = dict.fromkeys(cca_df.select_dtypes(np.int64).columns, np.int32)
            cca_df = cca_df.astype(d)
            return cca_df

    def ok_cb(self, checked):
        cca_df = self.getCca_df()
        if cca_df is None:
            return
        self.cca_df = cca_df
        self.cancel = False
        self.close()

    def cancel_cb(self, checked):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()
        w = (
            self.viewBox.minimumSizeHint().width()
            + 5*self.tableLayout.columnCount()
        )
        winGeometry = self.geometry()
        l, t, h = winGeometry.left(), winGeometry.top(), winGeometry.height()
        self.setGeometry(l, t, w, h)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def eventFilter(self, object, event):
        # Disable wheel scroll on widgets to allow scroll only on scrollarea
        if event.type() == QEvent.Wheel:
            event.ignore()
            return True
        return False

    def clearComboboxFocus(self):
        self.sender().clearFocus()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class askStopFrameSegm(QDialog):
    def __init__(
            self, user_ch_file_paths, user_ch_name,
            concat_segm=False, parent=None
        ):
        self.parent = parent
        self.cancel = True
        self.concat_segm = concat_segm

        super().__init__(parent)
        self.setWindowTitle('Enter stop frame')

        mainLayout = QVBoxLayout()
        formLayout = QFormLayout()
        buttonsLayout = QHBoxLayout()

        # Message
        infoTxt = (
        """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        p.big {
            line-height: 1.2;
        }
        </style>
        </head>
        <body>
        <p class="big">
            Enter a <b>stop frame number</b> when to stop<br>
            segmentation for each Position loaded:
        </p>
        </body>
        </html>
        """
        )
        infoLabel = QLabel(infoTxt, self)
        _font = QtGui.QFont()
        _font.setPointSize(11)
        infoLabel.setFont(_font)
        infoLabel.setAlignment(Qt.AlignCenter)
        # padding: top, left, bottom, right
        infoLabel.setStyleSheet("padding:0px 0px 8px 0px;")

        self.dataDict = {}

        # Form layout widget
        for img_path in user_ch_file_paths:
            pos_foldername = os.path.basename(
                os.path.dirname(
                    os.path.dirname(img_path)
                )
            )
            spinBox = QSpinBox()
            posData = load.loadData(img_path, user_ch_name, QParent=parent)
            posData.getBasenameAndChNames()
            posData.buildPaths()
            posData.loadImgData()
            posData.loadOtherFiles(
                load_segm_data=False,
                load_metadata=True,
                loadSegmInfo=True,
                )
            spinBox.setMaximum(posData.SizeT)
            if posData.segmSizeT == 1:
                spinBox.setValue(posData.SizeT)
            else:
                if self.concat_segm and posData.segmSizeT < posData.SizeT:
                    spinBox.setMinimum(posData.segmSizeT+1)
                    spinBox.setValue(posData.SizeT)
                else:
                    spinBox.setValue(posData.segmSizeT)
            spinBox.setAlignment(Qt.AlignCenter)
            visualizeButton = QPushButton('Visualize')
            visualizeButton.clicked.connect(self.visualize_cb)
            formLabel = QLabel(f'{pos_foldername}  ')
            layout = QHBoxLayout()
            layout.addWidget(formLabel, alignment=Qt.AlignRight)
            layout.addWidget(spinBox)
            layout.addWidget(visualizeButton)
            self.dataDict[visualizeButton] = (spinBox, posData)
            formLayout.addRow(layout)

        self.formLayout = formLayout
        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        mainLayout.addLayout(formLayout)

        okButton = QPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = QPushButton('Cancel')

        buttonsLayout.addWidget(okButton, alignment=Qt.AlignRight)
        buttonsLayout.addWidget(cancelButton, alignment=Qt.AlignLeft)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        # # self.setModal(True)

    def saveSegmSizeT(self):
        for spinBox, posData in self.dataDict.values():
            posData.segmSizeT = spinBox.value()
            posData.saveMetadata()

    def ok_cb(self, event):
        self.cancel = False
        self.saveSegmSizeT()
        self.close()

    def visualize_cb(self, checked=True):
        spinBox, posData = self.dataDict[self.sender()]
        posData.frame_i = spinBox.value()-1
        self.slideshowWin = imageViewer(
            posData=posData, spinBox=spinBox
        )
        self.slideshowWin.update_img()
        self.slideshowWin.framesScrollBar.setDisabled(True)
        self.slideshowWin.show()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()



class QLineEditDialog(QDialog):
    def __init__(
            self, title='Entry messagebox', msg='Entry value',
            defaultTxt='', parent=None, allowedValues=None,
            warnLastFrame=False
        ):
        QDialog.__init__(self)

        self.loop = None
        self.cancel = True
        self.allowedValues = allowedValues
        self.warnLastFrame = warnLastFrame
        if allowedValues and warnLastFrame:
            self.maxValue = max(allowedValues)

        self.setWindowTitle(title)

        # Layouts
        mainLayout = QVBoxLayout()
        LineEditLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        # Widgets
        msg = QLabel(msg)
        _font = QtGui.QFont()
        _font.setPointSize(11)
        msg.setFont(_font)
        msg.setAlignment(Qt.AlignCenter)
        # padding: top, left, bottom, right
        msg.setStyleSheet("padding:0px 0px 3px 0px;")

        ID_QLineEdit = QLineEdit()
        ID_QLineEdit.setFont(_font)
        ID_QLineEdit.setAlignment(Qt.AlignCenter)
        ID_QLineEdit.setText(defaultTxt)
        self.ID_QLineEdit = ID_QLineEdit

        if allowedValues is not None:
            notValidLabel = QLabel()
            notValidLabel.setStyleSheet('color: red')
            notValidLabel.setFont(_font)
            notValidLabel.setAlignment(Qt.AlignCenter)
            self.notValidLabel = notValidLabel

        okButton = QPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = QPushButton('Cancel')

        # Events
        ID_QLineEdit.textChanged[str].connect(self.ID_LineEdit_cb)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        # Contents margins
        buttonsLayout.setContentsMargins(0,10,0,0)

        # Add widgets to layouts
        LineEditLayout.addWidget(msg, alignment=Qt.AlignCenter)
        LineEditLayout.addWidget(ID_QLineEdit, alignment=Qt.AlignCenter)
        if allowedValues is not None:
            LineEditLayout.addWidget(notValidLabel, alignment=Qt.AlignCenter)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(okButton)
        buttonsLayout.addWidget(cancelButton)

        # Add layouts
        mainLayout.addLayout(LineEditLayout)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        # self.setModal(True)

    def ID_LineEdit_cb(self, text):
        # Get inserted char
        idx = self.ID_QLineEdit.cursorPosition()
        if idx == 0:
            return

        newChar = text[idx-1]

        # Allow only integers
        try:
            val = int(newChar)
            if val > np.iinfo(np.uint16).max:
                self.ID_QLineEdit.setText(str(np.iinfo(np.uint16).max))
            if self.allowedValues is not None:
                currentVal = int(self.ID_QLineEdit.text())
                if currentVal not in self.allowedValues:
                    self.notValidLabel.setText(f'{currentVal} not existing!')
                else:
                    self.notValidLabel.setText('')
        except Exception as e:
            text = text.replace(newChar, '')
            self.ID_QLineEdit.setText(text)
            return

    def warnValLessLastFrame(self, val):
        msg = QMessageBox()
        warn_txt = (f"""
        <p style="font-size:12px">
            WARNING: saving until a frame number below the last visited
            frame ({self.maxValue})<br>
            will result in <b>loss of information
            about any edit or annotation you did on frames
            {val}-{self.maxValue}.</b><br><br>
            Are you sure you want to proceed?
        </p>
        """)
        answer = msg.warning(
           self, 'WARNING: Potential loss of information',
           warn_txt, msg.Yes | msg.Cancel
        )
        return answer == msg.Cancel

    def ok_cb(self, event):
        if self.allowedValues:
            if self.notValidLabel.text():
                return

        val = int(self.ID_QLineEdit.text())
        if self.warnLastFrame and val < self.maxValue:
            cancel = self.warnValLessLastFrame(val)
            if cancel:
                return

        self.cancel = False
        self.EntryID = val
        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()


class editID_QWidget(QDialog):
    def __init__(self, clickedID, IDs, parent=None):
        self.IDs = IDs
        self.clickedID = clickedID
        self.cancel = True
        self.how = None

        super().__init__(parent)
        self.setWindowTitle("Edit ID")
        mainLayout = QVBoxLayout()

        VBoxLayout = QVBoxLayout()
        msg = QLabel(f'Replace ID {clickedID} with:')
        _font = QtGui.QFont()
        _font.setPointSize(11)
        msg.setFont(_font)
        # padding: top, left, bottom, right
        msg.setStyleSheet("padding:0px 0px 3px 0px;")
        VBoxLayout.addWidget(msg, alignment=Qt.AlignCenter)

        ID_QLineEdit = QLineEdit()
        ID_QLineEdit.setFont(_font)
        ID_QLineEdit.setAlignment(Qt.AlignCenter)
        self.ID_QLineEdit = ID_QLineEdit
        VBoxLayout.addWidget(ID_QLineEdit)

        note = QLabel(
            'NOTE: To replace multiple IDs at once\n'
            'write "(old ID, new ID), (old ID, new ID)" etc.'
        )
        note.setFont(_font)
        note.setAlignment(Qt.AlignCenter)
        # padding: top, left, bottom, right
        note.setStyleSheet("padding:12px 0px 0px 0px;")
        VBoxLayout.addWidget(note, alignment=Qt.AlignCenter)
        mainLayout.addLayout(VBoxLayout)

        HBoxLayout = QHBoxLayout()
        okButton = QPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)
        HBoxLayout.addWidget(okButton, alignment=Qt.AlignRight)

        cancelButton = QPushButton('Cancel')
        # cancelButton.setShortcut(Qt.Key_Escape)
        HBoxLayout.addWidget(cancelButton, alignment=Qt.AlignLeft)
        HBoxLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(HBoxLayout)

        self.setLayout(mainLayout)

        # Connect events
        self.prevText = ''
        ID_QLineEdit.textChanged[str].connect(self.ID_LineEdit_cb)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        # self.setModal(True)

    def ID_LineEdit_cb(self, text):
        # Get inserted char
        idx = self.ID_QLineEdit.cursorPosition()
        if idx == 0:
            return

        newChar = text[idx-1]

        # Do nothing if user is deleting text
        if idx == 0 or len(text)<len(self.prevText):
            self.prevText = text
            return

        # Do not allow chars except for "(", ")", "int", ","
        m = re.search(r'\(|\)|\d|,', newChar)
        if m is None:
            self.prevText = text
            text = text.replace(newChar, '')
            self.ID_QLineEdit.setText(text)
            return

        # Cast integers greater than uint16 machine limit
        m_iter = re.finditer(r'\d+', self.ID_QLineEdit.text())
        for m in m_iter:
            val = int(m.group())
            uint16_max = np.iinfo(np.uint16).max
            if val > uint16_max:
                text = self.ID_QLineEdit.text()
                text = f'{text[:m.start()]}{uint16_max}{text[m.end():]}'
                self.ID_QLineEdit.setText(text)

        # Automatically close ( bracket
        if newChar == '(':
            text += ')'
            self.ID_QLineEdit.setText(text)
        self.prevText = text

    def ok_cb(self, event):
        self.cancel = False
        txt = self.ID_QLineEdit.text()
        valid = False

        # Check validity of inserted text
        try:
            ID = int(txt)
            how = [(self.clickedID, ID)]
            if ID in self.IDs:
                warn_msg = (
                    f'ID {ID} is already existing. If you continue ID {ID} '
                    f'will be swapped with ID {self.clickedID}\n\n'
                    'Do you want to continue?'
                )
                msg = QMessageBox()
                do_swap = msg.warning(
                    self, 'Invalid entry', warn_msg, msg.Yes | msg.Cancel
                )
                if do_swap == msg.Yes:
                    valid = True
                else:
                    return
            else:
                valid = True
        except ValueError:
            pattern = r'\((\d+),\s*(\d+)\)'
            fa = re.findall(pattern, txt)
            if fa:
                how = [(int(g[0]), int(g[1])) for g in fa]
                valid = True
            else:
                valid = False

        if valid:
            self.how = how
            self.close()
        else:
            err_msg = (
                'You entered invalid text. Valid text is either a single integer'
                f' ID that will be used to replace ID {self.clickedID} '
                'or a list of elements enclosed in parenthesis separated by a comma\n'
                'such as (5, 10), (8, 27) to replace ID 5 with ID 10 and ID 8 with ID 27'
            )
            msg = QMessageBox()
            msg.critical(
                self, 'Invalid entry', err_msg, msg.Ok
            )

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()


class imshow_tk:
    def __init__(
            self, img, dots_coords=None, x_idx=1, axis=None,
            additional_imgs=[], titles=[], fixed_vrange=False,
            run=True, show_IDs=False
        ):
        if img.ndim == 3:
            if img.shape[-1] > 4:
                img = img.max(axis=0)
                h, w = img.shape
            else:
                h, w, _ = img.shape
        elif img.ndim == 2:
            h, w = img.shape
        elif img.ndim != 2 and img.ndim != 3:
            raise TypeError(f'Invalid shape {img.shape} for image data. '
            'Only 2D or 3D images.')
        for i, im in enumerate(additional_imgs):
            if im.ndim == 3 and im.shape[-1] > 4:
                additional_imgs[i] = im.max(axis=0)
            elif im.ndim != 2 and im.ndim != 3:
                raise TypeError(f'Invalid shape {im.shape} for image data. '
                'Only 2D or 3D images.')
        n_imgs = len(additional_imgs)+1
        if w/h > 2:
            fig, ax = plt.subplots(n_imgs, 1, sharex=True, sharey=True)
        else:
            fig, ax = plt.subplots(1, n_imgs, sharex=True, sharey=True)
        if n_imgs == 1:
            ax = [ax]
        self.ax0img = ax[0].imshow(img)
        if dots_coords is not None:
            ax[0].plot(dots_coords[:,x_idx], dots_coords[:,x_idx-1], 'r.')
        if axis:
            ax[0].axis('off')
        if fixed_vrange:
            vmin, vmax = img.min(), img.max()
        else:
            vmin, vmax = None, None
        self.additional_aximgs = []
        for i, img_i in enumerate(additional_imgs):
            axi_img = ax[i+1].imshow(img_i, vmin=vmin, vmax=vmax)
            self.additional_aximgs.append(axi_img)
            if dots_coords is not None:
                ax[i+1].plot(dots_coords[:,x_idx], dots_coords[:,x_idx-1], 'r.')
            if axis:
                ax[i+1].axis('off')
        for title, a in zip(titles, ax):
            a.set_title(title)

        if show_IDs:
            if issubclass(img.dtype.type, np.integer):
                rp = skimage.measure.regionprops(img)
                for obj in rp:
                    y, x = obj.centroid
                    ID = obj.label
                    ax[0].text(
                        int(x), int(y), str(ID), fontsize=12,
                        fontweight='normal', horizontalalignment='center',
                        verticalalignment='center', color='r'
                    )
            for i, img_i in enumerate(additional_imgs):
                if issubclass(img_i.dtype.type, np.integer):
                    rp = skimage.measure.regionprops(img_i)
                    for obj in rp:
                        y, x = obj.centroid
                        ID = obj.label
                        ax[i+1].text(
                            int(x), int(y), str(ID), fontsize=14,
                            fontweight='normal', horizontalalignment='center',
                            verticalalignment='center', color='r'
                        )
        sub_win = embed_tk('Imshow embedded in tk', [800,600,400,150], fig)
        sub_win.root.protocol("WM_DELETE_WINDOW", self._close)
        self.sub_win = sub_win
        self.fig = fig
        self.ax = ax
        sub_win.root.wm_attributes('-topmost',True)
        sub_win.root.focus_force()
        sub_win.root.after_idle(sub_win.root.attributes,'-topmost',False)
        if run:
            sub_win.root.mainloop()

    def _close(self):
        plt.close(self.fig)
        self.sub_win.root.quit()
        self.sub_win.root.destroy()

class embed_tk:
    """Example:
    -----------
    img = np.ones((600,600))
    fig = plt.Figure(figsize=(5, 4), dpi=100)
    ax = fig.add_subplot()
    ax.imshow(img)

    sub_win = embed_tk('Embeddding in tk', [1024,768,300,100], fig)

    def on_key_event(event):
        print('you pressed %s' % event.key)

    sub_win.canvas.mpl_connect('key_press_event', on_key_event)

    sub_win.root.mainloop()
    """
    def __init__(self, win_title, geom, fig):
        root = tk.Tk()
        root.wm_title(win_title)
        root.geometry("{}x{}+{}+{}".format(*geom)) # WidthxHeight+Left+Top
        # a tk.DrawingArea
        canvas = FigureCanvasTkAgg(fig, master=root)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        toolbar = NavigationToolbar2Tk(canvas, root)
        toolbar.update()
        canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas = canvas
        self.toolbar = toolbar
        self.root = root

class QtSelectItems(QDialog):
    def __init__(self, title, items, informativeText,
                 CbLabel='Select value:  ', parent=None):
        self.cancel = True
        self.selectedItemsText = ''
        self.selectedItemsIdx = None
        self.items = items
        super().__init__(parent)
        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()
        topLayout = QHBoxLayout()
        self.topLayout = topLayout
        bottomLayout = QHBoxLayout()

        if informativeText:
            infoLabel = QLabel(informativeText)
            mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        label = QLabel(CbLabel)
        topLayout.addWidget(label)

        combobox = QComboBox(self)
        combobox.addItems(items)
        self.ComboBox = combobox
        topLayout.addWidget(combobox)
        topLayout.setContentsMargins(0, 10, 0, 0)

        okButton = QPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)
        bottomLayout.addWidget(okButton, alignment=Qt.AlignRight)

        cancelButton = QPushButton('Cancel')
        bottomLayout.addWidget(cancelButton, alignment=Qt.AlignLeft)

        multiPosButton = QPushButton('Multiple selection')
        multiPosButton.setCheckable(True)
        self.multiPosButton = multiPosButton
        bottomLayout.addWidget(multiPosButton, alignment=Qt.AlignLeft)

        listBox = QListWidget()
        listBox.addItems(items)
        listBox.setSelectionMode(QAbstractItemView.ExtendedSelection)
        listBox.setCurrentRow(0)
        topLayout.addWidget(listBox)
        listBox.hide()
        self.ListBox = listBox

        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        # self.setModal(True)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        multiPosButton.toggled.connect(self.toggleMultiSelection)

    def toggleMultiSelection(self, checked):
        if checked:
            self.multiPosButton.setText('Single selection')
            self.ComboBox.hide()
            self.ListBox.show()
            # Show 10 items
            n = self.ListBox.count()
            if n > 10:
                h = sum([self.ListBox.sizeHintForRow(i) for i in range(10)])
            else:
                h = sum([self.ListBox.sizeHintForRow(i) for i in range(n)])
            self.ListBox.setFixedHeight(h+5)
            self.ListBox.setFocusPolicy(Qt.StrongFocus)
            self.ListBox.setFocus(True)
            self.ListBox.setCurrentRow(0)
        else:
            self.multiPosButton.setText('Multiple selection')
            self.ListBox.hide()
            self.ComboBox.show()


    def ok_cb(self, event):
        self.cancel = False
        if self.multiPosButton.isChecked():
            selectedItems = self.ListBox.selectedItems()
            selectedItemsText = [item.text() for item in selectedItems]
            self.selectedItemsText = natsorted(selectedItemsText)
            self.selectedItemsIdx = [self.items.index(txt)
                                     for txt in self.selectedItemsText]
        else:
            self.selectedItemsText = [self.ComboBox.currentText()]
            self.selectedItemsIdx = [self.ComboBox.currentIndex()]
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class manualSeparateGui(QMainWindow):
    def __init__(self, lab, ID, img, fontSize='12pt',
                 IDcolor=[255, 255, 0], parent=None,
                 loop=None):
        super().__init__(parent)
        self.loop = loop
        self.cancel = True
        self.parent = parent
        self.lab = lab.copy()
        self.lab[lab!=ID] = 0
        self.ID = ID
        self.img = skimage.exposure.equalize_adapthist(img/img.max())
        self.IDcolor = IDcolor
        self.countClicks = 0
        self.prevLabs = []
        self.prevAllCutsCoords = []
        self.labelItemsIDs = []
        self.undoIdx = 0
        self.fontSize = fontSize
        self.AllCutsCoords = []
        self.setWindowTitle("Cell-ACDC - Segm&Track")
        # self.setGeometry(Left, Top, 850, 800)

        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createToolBars()

        self.gui_createStatusBar()

        self.gui_createGraphics()
        self.gui_connectImgActions()

        self.gui_createImgWidgets()
        self.gui_connectActions()

        self.updateImg()
        self.zoomToObj()

        mainContainer = QtGui.QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QtGui.QGridLayout()
        mainLayout.addWidget(self.graphLayout, 0, 0, 1, 1)
        mainLayout.addLayout(self.img_Widglayout, 1, 0)

        mainContainer.setLayout(mainLayout)

        self.setWindowModality(Qt.WindowModal)

    def centerWindow(self):
        parent = self.parent
        if parent is not None:
            # Center the window on main window
            mainWinGeometry = parent.geometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinCenterX = int(mainWinLeft + mainWinWidth/2)
            mainWinCenterY = int(mainWinTop + mainWinHeight/2)
            winGeometry = self.geometry()
            winWidth = winGeometry.width()
            winHeight = winGeometry.height()
            winLeft = int(mainWinCenterX - winWidth/2)
            winRight = int(mainWinCenterY - winHeight/2)
            self.move(winLeft, winRight)

    def gui_createActions(self):
        # File actions
        self.exitAction = QAction("&Exit", self)
        self.helpAction = QAction('Help', self)
        self.undoAction = QAction(QIcon(":undo.svg"), "Undo (Ctrl+Z)", self)
        self.undoAction.setEnabled(False)
        self.undoAction.setShortcut("Ctrl+Z")

        self.okAction = QAction(QIcon(":applyCrop.svg"), "Happy with that", self)
        self.cancelAction = QAction(QIcon(":cancel.svg"), "Cancel", self)

    def gui_createMenuBar(self):
        menuBar = self.menuBar()
        style = "QMenuBar::item:selected { background: white; }"
        menuBar.setStyleSheet(style)
        # File menu
        fileMenu = QMenu("&File", self)
        menuBar.addMenu(fileMenu)

        menuBar.addAction(self.helpAction)
        fileMenu.addAction(self.exitAction)

    def gui_createToolBars(self):
        toolbarSize = 30

        editToolBar = QToolBar("Edit", self)
        editToolBar.setIconSize(QSize(toolbarSize, toolbarSize))
        self.addToolBar(editToolBar)

        editToolBar.addAction(self.okAction)
        editToolBar.addAction(self.cancelAction)

        editToolBar.addAction(self.undoAction)

        self.overlayButton = QToolButton(self)
        self.overlayButton.setIcon(QIcon(":overlay.svg"))
        self.overlayButton.setCheckable(True)
        self.overlayButton.setToolTip('Overlay cells image')
        editToolBar.addWidget(self.overlayButton)

    def gui_connectActions(self):
        self.exitAction.triggered.connect(self.close)
        self.helpAction.triggered.connect(self.help)
        self.okAction.triggered.connect(self.ok_cb)
        self.cancelAction.triggered.connect(self.close)
        self.undoAction.triggered.connect(self.undo)
        self.overlayButton.toggled.connect(self.toggleOverlay)
        self.hist.sigLookupTableChanged.connect(self.histLUT_cb)

    def gui_createStatusBar(self):
        self.statusbar = self.statusBar()
        # Temporary message
        self.statusbar.showMessage("Ready", 3000)
        # Permanent widget
        self.wcLabel = QLabel(f"")
        self.statusbar.addPermanentWidget(self.wcLabel)

    def gui_createGraphics(self):
        self.graphLayout = pg.GraphicsLayoutWidget()

        # Plot Item container for image
        self.ax = pg.PlotItem()
        self.ax.invertY(True)
        self.ax.setAspectLocked(True)
        self.ax.hideAxis('bottom')
        self.ax.hideAxis('left')
        self.graphLayout.addItem(self.ax, row=1, col=1)

        # Image Item
        self.imgItem = pg.ImageItem(np.zeros((512,512)))
        self.ax.addItem(self.imgItem)

        #Image histogram
        self.hist = pg.HistogramLUTItem()

        # Curvature items
        self.hoverLinSpace = np.linspace(0, 1, 1000)
        self.hoverLinePen = pg.mkPen(color=(200, 0, 0, 255*0.5),
                                     width=2, style=Qt.DashLine)
        self.hoverCurvePen = pg.mkPen(color=(200, 0, 0, 255*0.5), width=3)
        self.lineHoverPlotItem = pg.PlotDataItem(pen=self.hoverLinePen)
        self.curvHoverPlotItem = pg.PlotDataItem(pen=self.hoverCurvePen)
        self.curvAnchors = pg.ScatterPlotItem(
            symbol='o', size=9,
            brush=pg.mkBrush((255,0,0,50)),
            pen=pg.mkPen((255,0,0), width=2),
            hoverable=True, hoverPen=pg.mkPen((255,0,0), width=3),
            hoverBrush=pg.mkBrush((255,0,0))
        )
        self.ax.addItem(self.curvAnchors)
        self.ax.addItem(self.curvHoverPlotItem)
        self.ax.addItem(self.lineHoverPlotItem)

    def gui_createImgWidgets(self):
        self.img_Widglayout = QtGui.QGridLayout()
        self.img_Widglayout.setContentsMargins(50, 0, 50, 0)

        alphaScrollBar_label = QLabel('Overlay alpha  ')
        alphaScrollBar = QScrollBar(Qt.Horizontal)
        alphaScrollBar.setFixedHeight(20)
        alphaScrollBar.setMinimum(0)
        alphaScrollBar.setMaximum(40)
        alphaScrollBar.setValue(12)
        alphaScrollBar.setToolTip(
            'Control the alpha value of the overlay.\n'
            'alpha=0 results in NO overlay,\n'
            'alpha=1 results in only labels visible'
        )
        alphaScrollBar.sliderMoved.connect(self.alphaScrollBarMoved)
        self.alphaScrollBar = alphaScrollBar
        self.alphaScrollBar_label = alphaScrollBar_label
        self.img_Widglayout.addWidget(
            alphaScrollBar_label, 0, 0, alignment=Qt.AlignCenter
        )
        self.img_Widglayout.addWidget(alphaScrollBar, 0, 1, 1, 20)
        self.alphaScrollBar.hide()
        self.alphaScrollBar_label.hide()

    def gui_connectImgActions(self):
        self.imgItem.hoverEvent = self.gui_hoverEventImg
        self.imgItem.mousePressEvent = self.gui_mousePressEventImg
        self.imgItem.mouseMoveEvent = self.gui_mouseDragEventImg
        self.imgItem.mouseReleaseEvent = self.gui_mouseReleaseEventImg

    def gui_hoverEventImg(self, event):
        # Update x, y, value label bottom right
        try:
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            _img = self.lab
            Y, X = _img.shape
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                val = _img[ydata, xdata]
                self.wcLabel.setText(f'(x={x:.2f}, y={y:.2f}, ID={val:.0f})')
            else:
                self.wcLabel.setText(f'')
        except Exception as e:
            self.wcLabel.setText(f'')

        try:
            if not event.isExit():
                x, y = event.pos()
                if self.countClicks == 1:
                    self.lineHoverPlotItem.setData([self.x0, x], [self.y0, y])
                elif self.countClicks == 2:
                    xx = [self.x0, x, self.x1]
                    yy = [self.y0, y, self.y1]
                    xi, yi = self.getSpline(xx, yy)
                    self.curvHoverPlotItem.setData(xi, yi)
                elif self.countClicks == 0:
                    self.curvHoverPlotItem.setData([], [])
                    self.lineHoverPlotItem.setData([], [])
                    self.curvAnchors.setData([], [])
        except Exception as e:
            traceback.print_exc()
            pass

    def getSpline(self, xx, yy):
        tck, u = scipy.interpolate.splprep([xx, yy], s=0, k=2)
        xi, yi = scipy.interpolate.splev(self.hoverLinSpace, tck)
        return xi, yi


    def gui_mousePressEventImg(self, event):
        right_click = event.button() == Qt.MouseButton.RightButton
        left_click = event.button() == Qt.MouseButton.LeftButton

        dragImg = (left_click)

        if dragImg:
            pg.ImageItem.mousePressEvent(self.imgItem, event)

        if right_click and self.countClicks == 0:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.x0, self.y0 = xdata, ydata
            self.curvAnchors.addPoints([xdata], [ydata])
            self.countClicks = 1
        elif right_click and self.countClicks == 1:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.x1, self.y1 = xdata, ydata
            self.curvAnchors.addPoints([xdata], [ydata])
            self.countClicks = 2
        elif right_click and self.countClicks == 2:
            self.storeUndoState()
            self.countClicks = 0
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            xx = [self.x0, xdata, self.x1]
            yy = [self.y0, ydata, self.y1]
            xi, yi = self.getSpline(xx, yy)
            yy, xx = np.round(yi).astype(int), np.round(xi).astype(int)
            xxCurve, yyCurve = [], []
            for i, (r0, c0) in enumerate(zip(yy, xx)):
                if i == len(yy)-1:
                    break
                r1 = yy[i+1]
                c1 = xx[i+1]
                rr, cc, _ = skimage.draw.line_aa(r0, c0, r1, c1)
                # rr, cc = skimage.draw.line(r0, c0, r1, c1)
                nonzeroMask = self.lab[rr, cc]>0
                xxCurve.extend(cc[nonzeroMask])
                yyCurve.extend(rr[nonzeroMask])
            self.AllCutsCoords.append((yyCurve, xxCurve))
            for rr, cc in self.AllCutsCoords:
                self.lab[rr, cc] = 0
            skimage.morphology.remove_small_objects(self.lab, 5, in_place=True)
            self.splitObjectAlongCurve()


    def histLUT_cb(self, LUTitem):
        if self.overlayButton.isChecked():
            overlay = self.getOverlay()
            self.imgItem.setImage(overlay)

    def updateImg(self):
        self.updateLookuptable()
        rp = skimage.measure.regionprops(self.lab)
        self.rp = rp

        if self.overlayButton.isChecked():
            overlay = self.getOverlay()
            self.imgItem.setImage(overlay)
        else:
            self.imgItem.setImage(self.lab)

        # Draw ID on centroid of each label
        for labelItemID in self.labelItemsIDs:
            self.ax.removeItem(labelItemID)
        self.labelItemsIDs = []
        for obj in rp:
            labelItemID = pg.LabelItem()
            labelItemID.setText(f'{obj.label}', color='r', size=self.fontSize)
            y, x = obj.centroid
            w, h = labelItemID.rect().right(), labelItemID.rect().bottom()
            labelItemID.setPos(x-w/2, y-h/2)
            self.labelItemsIDs.append(labelItemID)
            self.ax.addItem(labelItemID)

    def zoomToObj(self):
        # Zoom to object
        lab_mask = (self.lab>0).astype(np.uint8)
        rp = skimage.measure.regionprops(lab_mask)
        obj = rp[0]
        min_row, min_col, max_row, max_col = obj.bbox
        xRange = min_col-10, max_col+10
        yRange = max_row+10, min_row-10
        self.ax.setRange(xRange=xRange, yRange=yRange)

    def storeUndoState(self):
        self.prevLabs.append(self.lab.copy())
        self.prevAllCutsCoords.append(self.AllCutsCoords.copy())
        self.undoIdx += 1
        self.undoAction.setEnabled(True)

    def undo(self):
        self.undoIdx -= 1
        self.lab = self.prevLabs[self.undoIdx]
        self.AllCutsCoords = self.prevAllCutsCoords[self.undoIdx]
        self.updateImg()
        if self.undoIdx == 0:
            self.undoAction.setEnabled(False)
            self.prevLabs = []
            self.prevAllCutsCoords = []


    def splitObjectAlongCurve(self):
        self.lab = skimage.measure.label(self.lab, connectivity=1)

        # Relabel largest object with original ID
        rp = skimage.measure.regionprops(self.lab)
        areas = [obj.area for obj in rp]
        IDs = [obj.label for obj in rp]
        maxAreaIdx = areas.index(max(areas))
        maxAreaID = IDs[maxAreaIdx]
        if self.ID not in self.lab:
            self.lab[self.lab==maxAreaID] = self.ID

        # Keep only the two largest objects
        larger_areas = nlargest(2, areas)
        larger_ids = [rp[areas.index(area)].label for area in larger_areas]
        for obj in rp:
            if obj.label not in larger_ids:
                print(f'deleting ID {obj.label}')
                self.lab[tuple(obj.coords.T)] = 0

        rp = skimage.measure.regionprops(self.lab)

        if self.parent is not None:
            self.parent.setBrushID()
        # Use parent window setBrushID function for all other IDs
        for i, obj in enumerate(rp):
            if self.parent is None:
                break
            if i == maxAreaIdx:
                continue
            posData = self.parent.data[self.parent.pos_i]
            posData.brushID += 1
            self.lab[obj.slice][obj.image] = posData.brushID

        # Replace 0s on the cutting curve with IDs
        self.cutLab = self.lab.copy()
        for rr, cc in self.AllCutsCoords:
            for y, x in zip(rr, cc):
                top_row = self.cutLab[y+1, x-1:x+2]
                bot_row = self.cutLab[y-1, x-1:x+1]
                left_col = self.cutLab[y-1, x-1]
                right_col = self.cutLab[y:y+2, x+1]
                allNeigh = list(top_row)
                allNeigh.extend(bot_row)
                allNeigh.append(left_col)
                allNeigh.extend(right_col)
                newID = max(allNeigh)
                self.lab[y,x] = newID

        self.rp = skimage.measure.regionprops(self.lab)
        self.updateImg()

    def updateLookuptable(self):
        # Lookup table
        self.cmap = myutils.getFromMatplotlib('viridis')
        self.lut = self.cmap.getLookupTable(0,1,self.lab.max()+1)
        self.lut[0] = [25,25,25]
        self.lut[self.ID] = self.IDcolor
        if self.overlayButton.isChecked():
            self.imgItem.setLookupTable(None)
        else:
            self.imgItem.setLookupTable(self.lut)

    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.countClicks = 0
            self.curvHoverPlotItem.setData([], [])
            self.lineHoverPlotItem.setData([], [])
            self.curvAnchors.setData([], [])
        elif ev.key() == Qt.Key_Enter or ev.key() == Qt.Key_Return:
            self.ok_cb()

    def getOverlay(self):
        # Rescale intensity based on hist ticks values
        min = self.hist.gradient.listTicks()[0][1]
        max = self.hist.gradient.listTicks()[1][1]
        img = skimage.exposure.rescale_intensity(
                                      self.img, in_range=(min, max))
        alpha = self.alphaScrollBar.value()/self.alphaScrollBar.maximum()

        # Convert img and lab to RGBs
        rgb_shape = (self.lab.shape[0], self.lab.shape[1], 3)
        labRGB = np.zeros(rgb_shape)
        labRGB[self.lab>0] = [1, 1, 1]
        imgRGB = skimage.color.gray2rgb(img)
        overlay = imgRGB*(1.0-alpha) + labRGB*alpha

        # Color eaach label
        for obj in self.rp:
            rgb = self.lut[obj.label]/255
            overlay[obj.slice][obj.image] *= rgb

        # Convert (0,1) to (0,255)
        overlay = (np.clip(overlay, 0, 1)*255).astype(np.uint8)
        return overlay


    def gui_mouseDragEventImg(self, event):
        pass

    def gui_mouseReleaseEventImg(self, event):
        pass

    def alphaScrollBarMoved(self, alpha_int):
        overlay = self.getOverlay()
        self.imgItem.setImage(overlay)

    def toggleOverlay(self, checked):
        if checked:
            self.graphLayout.addItem(self.hist, row=1, col=0)
            self.alphaScrollBar.show()
            self.alphaScrollBar_label.show()
        else:
            self.graphLayout.removeItem(self.hist)
            self.alphaScrollBar.hide()
            self.alphaScrollBar_label.hide()
        self.updateImg()

    def help(self):
        msg = QMessageBox()
        msg.information(self, 'Help',
            'Separate object along a curved line.\n\n'
            'To draw a curved line you will need 3 right-clicks:\n\n'
            '1. Right-click outside of the object --> a line appears.\n'
            '2. Right-click to end the line and a curve going through the '
            'mouse cursor will appear.\n'
            '3. Once you are happy with the cutting curve right-click again '
            'and the object will be separated along the curve.\n\n'
            'Note that you can separate as many times as you want.\n\n'
            'Once happy click on the green tick on top-right or '
            'cancel the process with the "X" button')

    def ok_cb(self, checked):
        self.cancel = False
        self.close()

    def closeEvent(self, event):
        if self.loop is not None:
            self.loop.exit()

class DataFrameModel(QtCore.QAbstractTableModel):
    # https://stackoverflow.com/questions/44603119/how-to-display-a-pandas-data-frame-with-pyqt5-pyside2
    DtypeRole = QtCore.Qt.UserRole + 1000
    ValueRole = QtCore.Qt.UserRole + 1001

    def __init__(self, df=pd.DataFrame(), parent=None):
        super(DataFrameModel, self).__init__(parent)
        self._dataframe = df

    def setDataFrame(self, dataframe):
        self.beginResetModel()
        self._dataframe = dataframe.copy()
        self.endResetModel()

    def dataFrame(self):
        return self._dataframe

    dataFrame = QtCore.pyqtProperty(pd.DataFrame, fget=dataFrame,
                                    fset=setDataFrame)

    @QtCore.pyqtSlot(int, QtCore.Qt.Orientation, result=str)
    def headerData(self, section: int,
                   orientation: QtCore.Qt.Orientation,
                   role: int = QtCore.Qt.DisplayRole):
        if role == QtCore.Qt.DisplayRole:
            if orientation == QtCore.Qt.Horizontal:
                return self._dataframe.columns[section]
            else:
                return str(self._dataframe.index[section])
        return QtCore.QVariant()

    def rowCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return len(self._dataframe.index)

    def columnCount(self, parent=QtCore.QModelIndex()):
        if parent.isValid():
            return 0
        return self._dataframe.columns.size

    def data(self, index, role=QtCore.Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < self.rowCount() \
            and 0 <= index.column() < self.columnCount()):
            return QtCore.QVariant()
        row = self._dataframe.index[index.row()]
        col = self._dataframe.columns[index.column()]
        dt = self._dataframe[col].dtype

        if role == Qt.TextAlignmentRole:
            return Qt.AlignCenter

        val = self._dataframe.iloc[row][col]
        if role == QtCore.Qt.DisplayRole:
            return str(val)
        elif role == DataFrameModel.ValueRole:
            return val
        if role == DataFrameModel.DtypeRole:
            return dt
        return QtCore.QVariant()

    def roleNames(self):
        roles = {
            QtCore.Qt.DisplayRole: b'display',
            DataFrameModel.DtypeRole: b'dtype',
            DataFrameModel.ValueRole: b'value'
        }
        return roles

class pdDataFrameWidget(QMainWindow):
    def __init__(self, df, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.setWindowTitle('Cell cycle annotations')

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)



        mainContainer = QtGui.QWidget()
        self.setCentralWidget(mainContainer)

        layout = QVBoxLayout()

        self.tableView = QTableView(self)
        layout.addWidget(self.tableView)
        model = DataFrameModel(df)
        self.tableView.setModel(model)
        for i in range(len(df.columns)):
            self.tableView.resizeColumnToContents(i)
        # layout.addWidget(QPushButton('Ok', self))
        mainContainer.setLayout(layout)

    def updateTable(self, df):
        if df is None:
            df = self.parent.getBaseCca_df()
        df = df.reset_index()
        model = DataFrameModel(df)
        self.tableView.setModel(model)
        for i in range(len(df.columns)):
            self.tableView.resizeColumnToContents(i)

    def setGeometryWindow(self, maxWidth=1024):
        width = self.tableView.verticalHeader().width() + 4
        for j in range(self.tableView.model().columnCount()):
            width += self.tableView.columnWidth(j) + 4
        height = self.tableView.horizontalHeader().height() + 4
        h = height + (self.tableView.rowHeight(0) + 4)*10
        w = width if width<maxWidth else maxWidth
        self.setGeometry(100, 100, w, h)

        # Center window
        parent = self.parent
        if parent is not None:
            # Center the window on main window
            mainWinGeometry = parent.geometry()
            mainWinLeft = mainWinGeometry.left()
            mainWinTop = mainWinGeometry.top()
            mainWinWidth = mainWinGeometry.width()
            mainWinHeight = mainWinGeometry.height()
            mainWinCenterX = int(mainWinLeft + mainWinWidth/2)
            mainWinCenterY = int(mainWinTop + mainWinHeight/2)
            winGeometry = self.geometry()
            winWidth = winGeometry.width()
            winHeight = winGeometry.height()
            winLeft = int(mainWinCenterX - winWidth/2)
            winRight = int(mainWinCenterY - winHeight/2)
            self.move(winLeft, winRight)

    def closeEvent(self, event):
        self.parent.ccaTableWin = None

class QDialogZsliceAbsent(QDialog):
    def __init__(self, filename, SizeZ, filenamesWithInfo, parent=None):
        self.runDataPrep = False
        self.useMiddleSlice = False
        self.useSameAsCh = False

        super().__init__(parent)
        self.setWindowTitle('z-slice info absent!')

        mainLayout = QVBoxLayout()
        buttonsLayout = QGridLayout()

        txt = (
        f"""
        <p style="font-size:14px; text-align: center;">
            You loaded the fluorescent file called<br><br>{filename}<br><br>
            however you <b>never selected which z-slice</b><br> you want to use
            when calculating metrics<br> (e.g., mean, median, amount...etc.)<br><br>
            Choose one of following options:
        <p>
        """
        )
        infoLabel = QLabel(txt)
        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        runDataPrepButton = QPushButton(
            '  Visualize the data now and select a z-slice (RECOMMENDED)  '
        )
        buttonsLayout.addWidget(runDataPrepButton, 0, 1, 1, 2)
        runDataPrepButton.clicked.connect(self.runDataPrep_cb)

        useMiddleSliceButton = QPushButton(
            f'  Use the middle z-slice ({int(SizeZ/2)+1})  '
        )
        buttonsLayout.addWidget(useMiddleSliceButton, 1, 1, 1, 2)
        useMiddleSliceButton.clicked.connect(self.useMiddleSlice_cb)

        useSameAsChButton = QPushButton(
            '  Use the same z-slice used for the channel: '
        )
        useSameAsChButton.clicked.connect(self.useSameAsCh_cb)

        chNameComboBox = QComboBox()
        chNameComboBox.addItems(filenamesWithInfo)
        # chNameComboBox.setEditable(True)
        # chNameComboBox.lineEdit().setAlignment(Qt.AlignCenter)
        # chNameComboBox.lineEdit().setReadOnly(True)
        self.chNameComboBox = chNameComboBox
        buttonsLayout.addWidget(useSameAsChButton, 2, 1)
        buttonsLayout.addWidget(chNameComboBox, 2, 2)

        buttonsLayout.setColumnStretch(0, 1)
        buttonsLayout.setColumnStretch(3, 1)
        buttonsLayout.setContentsMargins(10, 0, 10, 0)

        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        font = QtGui.QFont()
        font.setPointSize(11)
        self.setFont(font)

        # self.setModal(True)

    def useSameAsCh_cb(self, checked):
        self.useSameAsCh = True
        self.selectedChannel = self.chNameComboBox.currentText()
        self.close()

    def useMiddleSlice_cb(self, checked):
        self.useMiddleSlice = True
        self.close()

    def runDataPrep_cb(self, checked):
        self.runDataPrep = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class QDialogMultiSegmNpz(QDialog):
    def __init__(self, images_ls, parent_path, parent=None, multiPos=False):
        self.cancel = True
        self.selectedItemText = ''
        self.selectedItemIdx = None
        self.removeOthers = False
        self.okAllPos = False
        self.images_ls = images_ls
        self.parent_path = parent_path
        super().__init__(parent)

        informativeText = (f"""
        <p style="font-size:12px">
            The folder<br><br>{parent_path}<br><br>
            contains <b>multipe segmentation masks!</b><br>
        </p>
        """)

        self.setWindowTitle('Multiple segm.npz files detected!')
        is_win = sys.platform.startswith("win")

        mainLayout = QVBoxLayout()
        infoLayout = QHBoxLayout()
        selectionLayout = QGridLayout()
        buttonsLayout = QGridLayout()

        label = QLabel()
        # padding: top, left, bottom, right
        # label.setStyleSheet("padding:5px 0px 12px 0px;")
        label.setPixmap(QtGui.QPixmap(':warning.svg'))
        infoLayout.addWidget(label)

        infoLabel = QLabel(informativeText)
        infoLayout.addWidget(infoLabel)
        infoLayout.addStretch(1)
        mainLayout.addLayout(infoLayout)

        label = QLabel('Select which segmentation file to load:')
        combobox = QComboBox()
        combobox.addItems(images_ls)
        self.ComboBox = combobox

        okButton = QPushButton(' Load selected ')
        okButton.setShortcut(Qt.Key_Enter)
        okAndRemoveButton = QPushButton(' Load selected and delete the other files ')
        s = ' Show in Explorer... ' if is_win else ' Reveal in Finder... '
        showInFileManagerButton = QPushButton(s)
        cancelButton = QPushButton(' Cancel ')

        row, col = 0, 0
        buttonsLayout.addWidget(okButton, row, col)
        row, col = 1, 0
        if multiPos:
            row, col = 1, 1
        buttonsLayout.addWidget(showInFileManagerButton, row, col) # 1, 0 --> 1, 1

        row, col = 0, 1
        if multiPos:
            okAllPos = QPushButton(' Load selected for ALL positions ')
            buttonsLayout.addWidget(okAllPos, row, col) # 0, 1
            row, col = 1, 0
            okAllPos.clicked.connect(self.ok_allPos)

        buttonsLayout.addWidget(okAndRemoveButton, row, col) # 0, 1 --> 1, 0

        row, col = 1, 1
        if multiPos:
            row, col = 2, 1
        buttonsLayout.addWidget(cancelButton, row, col) # 1, 1 --> 2, 1


        buttonsLayout.setContentsMargins(0, 10, 0, 10)

        selectionLayout.addWidget(label, 0, 1, alignment=Qt.AlignLeft)
        selectionLayout.addWidget(combobox, 1, 1)
        selectionLayout.setColumnStretch(0, 1)
        selectionLayout.setColumnStretch(2, 1)
        selectionLayout.addLayout(buttonsLayout, 2, 1)

        mainLayout.addLayout(selectionLayout)
        self.setLayout(mainLayout)

        self.okButton = okButton
        self.okAndRemoveButton = okAndRemoveButton

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        okAndRemoveButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        showInFileManagerButton.clicked.connect(self.showInFileManager)

    def showInFileManager(self, checked=True):
        myutils.showInExplorer(self.parent_path)

    def ok_allPos(self, checked=False):
        self.cancel = False
        self.okAllPos = True
        self.selectedItemText = self.ComboBox.currentText()
        self.selectedItemIdx = self.ComboBox.currentIndex()
        self.close()

    def ok_cb(self, event):
        self.removeOthers = self.sender() == self.okAndRemoveButton
        if self.removeOthers:
            msg = QMessageBox()
            err_msg = (f"""
            <p style="font-size:12px">
                Are you sure you want to <b>delete the files</b> below?<br><br>
                {',<br>'.join(self.images_ls)}
            </p>
            """)
            delete_answer = msg.warning(
               self, 'Delete files?', err_msg, msg.Yes | msg.Cancel
            )
            if delete_answer == msg.Cancel:
                self.removeOthers = False
                return
        # self.applyToAll = self.applyToAll_CB.isChecked()
        self.cancel = False
        self.selectedItemText = self.ComboBox.currentText()
        self.selectedItemIdx = self.ComboBox.currentIndex()
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class QDialogPbar(QDialog):
    def __init__(self, title='Progress', infoTxt='', parent=None):
        self.workerFinished = False
        self.aborted = False
        self.clickCount = 0
        super().__init__(parent)

        abort_text = 'Control+Cmd+C to abort' if is_mac else 'Ctrl+Alt+C to abort'

        self.setWindowTitle(f'{title} ({abort_text})')
        self.setWindowFlags(Qt.Window)

        mainLayout = QVBoxLayout()
        pBarLayout = QGridLayout()

        if infoTxt:
            infoLabel = QLabel(infoTxt)
            mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        self.progressLabel = QLabel()

        self.QPbar = QProgressBar(self)
        self.QPbar.setValue(0)
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(207, 235, 155))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
        self.QPbar.setPalette(palette)
        pBarLayout.addWidget(self.QPbar, 0, 0)
        self.ETA_label = QLabel('NDh:NDm:NDs')
        pBarLayout.addWidget(self.ETA_label, 0, 1)

        self.metricsQPbar = QProgressBar(self)
        self.metricsQPbar.setValue(0)
        palette = QtGui.QPalette()
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor(207, 235, 155))
        palette.setColor(QtGui.QPalette.Text, QtGui.QColor(0, 0, 0))
        palette.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor(0, 0, 0))
        self.metricsQPbar.setPalette(palette)
        pBarLayout.addWidget(self.metricsQPbar, 1, 0)

        #pBarLayout.setColumnStretch(2, 1)

        mainLayout.addWidget(self.progressLabel)
        mainLayout.addLayout(pBarLayout)

        self.setLayout(mainLayout)
        # self.setModal(True)

    def keyPressEvent(self, event):
        isCtrlAlt = event.modifiers() == (Qt.ControlModifier | Qt.AltModifier)
        if isCtrlAlt and event.key() == Qt.Key_C:
            doAbort = self.askAbort()
            if doAbort:
                self.aborted = True
                self.workerFinished = True
                self.close()

    def askAbort(self):
        msg = QMessageBox()
        txt = ("""
        <p style="font-size:9pt">
            Aborting with "Ctrl+Alt+C" is <b>not safe</b>.<br><br>
            The system status cannot be predicted and
            it will <b>require a restart</b>.<br><br>
            Are you sure you want to abort?
        </p>
        """)
        answer = msg.critical(
            self, 'Are you sure you want to abort?', txt, msg.Yes | msg.No
        )
        return answer == msg.Yes


    def abort(self):
        self.clickCount += 1
        self.aborted = True
        if self.clickCount > 3:
            self.workerFinished = True
            self.close()

    def closeEvent(self, event):
        if not self.workerFinished:
            event.ignore()

class QDialogModelParams(QDialog):
    def __init__(
            self, init_params, segment_params, model_name,
            url=None, parent=None):
        self.cancel = True
        super().__init__(parent)

        self.setWindowTitle(f'{model_name} parameters')

        mainLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        initGroupBox, self.init_argsWidgets = self.createGroupParams(
            init_params,
            'Parameters for model initialization'
        )

        segmentGroupBox, self.segment2D_argsWidgets = self.createGroupParams(
            segment_params,
            'Parameters for 2D segmentation'
        )

        okButton = QPushButton(' Ok ')
        buttonsLayout.addWidget(okButton)

        infoButton = QPushButton(' More info... ')
        buttonsLayout.addWidget(infoButton)

        cancelButton = QPushButton(' Cancel ')
        buttonsLayout.addWidget(cancelButton)

        buttonsLayout.setContentsMargins(0, 10, 0, 10)

        okButton.clicked.connect(self.ok_cb)
        infoButton.clicked.connect(self.info_params)
        cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(initGroupBox)
        mainLayout.addSpacing(15)
        mainLayout.addStretch(1)
        mainLayout.addWidget(segmentGroupBox)

        # Add minimum size spinbox whihc is valid for all models
        artefactsGroupBox = postProcessSegmParams(
            'Post-processing segmentation parameters'
        )
        artefactsGroupBox.setCheckable(True)
        artefactsGroupBox.setChecked(True)
        self.artefactsGroupBox = artefactsGroupBox

        self.minSize_SB = artefactsGroupBox.minSize_SB
        self.minSolidity_DSB = artefactsGroupBox.minSolidity_DSB
        self.maxElongation_DSB = artefactsGroupBox.maxElongation_DSB

        mainLayout.addSpacing(15)
        mainLayout.addStretch(1)
        mainLayout.addWidget(artefactsGroupBox)

        if url is not None:
            mainLayout.addWidget(
                self.createSeeHereLabel(url),
                alignment=Qt.AlignCenter
            )

        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        font = QtGui.QFont()
        font.setPointSize(11)
        self.setFont(font)

        # self.setModal(True)

    def info_params(self):
        self.infoWin = QMessageBox()
        self.infoWin.setWindowTitle('Model parameters info')
        self.infoWin.setIcon(self.infoWin.Information)
        txt = (
            'Currently Cell-ACDC has three models implemented: '
            'YeaZ, Cellpose and StarDist.\n\n'
            'Cellpose and StarDist have the following default models available:\n\n'
            'Cellpose:\n'
            '   - cyto\n'
            '   - nuclei\n'
            '   - cyto2\n'
            '   - bact\n'
            '   - bact_omni\n'
            '   - cyto2_omni\n\n'
            'StarDist:\n'
            '   - 2D_versatile_fluo\n'
            '   - 2D_versatile_he\n'
            '   - 2D_paper_dsb2018\n'
        )
        self.infoWin.setText(txt)
        self.infoWin.addButton(self.infoWin.Ok)
        self.infoWin.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.infoWin.setModal(False)
        self.infoWin.show()


    def createGroupParams(self, ArgSpecs_list, groupName):
        ArgWidget = namedtuple('ArgsWidgets', ['name', 'type', 'widget'])
        ArgsWidgets_list = []
        groupBox = QGroupBox(groupName)

        groupBoxLayout = QGridLayout()
        for row, ArgSpec in enumerate(ArgSpecs_list):
            var_name = ArgSpec.name.replace('_', ' ').capitalize()
            label = QLabel(f'{var_name}:  ')
            groupBoxLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
            if ArgSpec.type == bool:
                trueRadioButton = QRadioButton('True')
                falseRadioButton = QRadioButton('False')
                if ArgSpec.default:
                    trueRadioButton.setChecked(True)
                else:
                    falseRadioButton.setChecked(True)
                widget = trueRadioButton
                groupBoxLayout.addWidget(trueRadioButton, row, 1)
                groupBoxLayout.addWidget(falseRadioButton, row, 2)
            elif ArgSpec.type == int:
                spinBox = QSpinBox()
                spinBox.setAlignment(Qt.AlignCenter)
                spinBox.setMaximum(2147483647)
                spinBox.setValue(ArgSpec.default)
                widget = spinBox
                groupBoxLayout.addWidget(spinBox, row, 1, 1, 2)
            elif ArgSpec.type == float:
                doubleSpinBox = QDoubleSpinBox()
                doubleSpinBox.setAlignment(Qt.AlignCenter)
                doubleSpinBox.setMaximum(2**32)
                doubleSpinBox.setValue(ArgSpec.default)
                widget = doubleSpinBox
                groupBoxLayout.addWidget(doubleSpinBox, row, 1, 1, 2)
            else:
                lineEdit = QLineEdit()
                lineEdit.setText(str(ArgSpec.default))
                lineEdit.setAlignment(Qt.AlignCenter)
                widget = lineEdit
                groupBoxLayout.addWidget(lineEdit, row, 1, 1, 2)

            argsInfo = ArgWidget(
                name=ArgSpec.name,
                type=ArgSpec.type,
                widget=widget,
            )
            ArgsWidgets_list.append(argsInfo)

        groupBox.setLayout(groupBoxLayout)
        return groupBox, ArgsWidgets_list

    def createSeeHereLabel(self, url):
        htmlTxt = f'<a href=\"{url}">here</a>'
        seeHereLabel = QLabel()
        seeHereLabel.setText(f"""
            <p style="font-size:12px">
                See {htmlTxt} for details on the parameters
            </p>
        """)
        seeHereLabel.setTextFormat(Qt.RichText)
        seeHereLabel.setTextInteractionFlags(Qt.TextBrowserInteraction)
        seeHereLabel.setOpenExternalLinks(True)
        seeHereLabel.setStyleSheet("padding:12px 0px 0px 0px;")
        return seeHereLabel

    def argsWidgets_to_kwargs(self, argsWidgets):
        kwargs_dict = {}
        for argWidget in argsWidgets:
            if argWidget.type == bool:
                kwargs_dict[argWidget.name] = argWidget.widget.isChecked()
            elif argWidget.type == int or argWidget.type == float:
                kwargs_dict[argWidget.name] = argWidget.widget.value()
            elif argWidget.type == str:
                kwargs_dict[argWidget.name] = argWidget.widget.text()
            else:
                to_type = argWidget.type
                s = argWidget.widget.text()
                kwargs_dict[argWidget.name] = eval(s)
        return kwargs_dict

    def ok_cb(self, checked):
        self.cancel = False
        self.init_kwargs = self.argsWidgets_to_kwargs(self.init_argsWidgets)
        self.segment2D_kwargs = self.argsWidgets_to_kwargs(
            self.segment2D_argsWidgets
        )
        self.minSize = self.minSize_SB.value()
        self.minSolidity = self.minSolidity_DSB.value()
        self.maxElongation = self.maxElongation_DSB.value()
        self.applyPostProcessing = self.artefactsGroupBox.isChecked()
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class downloadModel(QMessageBox):
    def __init__(self, model_name, parent=None):
        super().__init__(parent)
        self.loop = None
        self.model_name = model_name

    def download(self):
        success = myutils.download_model(self.model_name)
        if not success:
            self.exec_()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        import cellacdc
        model_name = self.model_name
        m = model_name.lower()
        weights_filenames = getattr(cellacdc, f'{m}_weights_filenames')
        self.setIcon(self.Critical)
        self.setWindowTitle(f'Download of {model_name} failed')
        self.setTextFormat(Qt.RichText)
        url, alternative_url = myutils._model_url(
            model_name, return_alternative=True
        )
        url_href = f'<a href="{url}">this link</a>'
        alternative_url_href = f'<a href="{alternative_url}">this link</a>'
        _, model_path = myutils.get_model_path(model_name, create_temp_dir=False)
        txt = (f"""
        <p style=font-size:13px>
            Automatic download of {model_name} failed.<br><br>
            Please, <b>manually download</b> the model weights from {url_href} or
            {alternative_url_href}.<br><br>
            Next, unzip the content of the downloaded file into the
            following folder:<br><br>
            {model_path}<br>
        </p>
        <p style=font-size:12px>
            <i>NOTE: if clicking on the link above does not work
            copy one of the links below and paste it into the browser</i><br><br>
            {url}<br>{alternative_url}
        </p>
        """)
        self.setText(txt)
        weights_paths = [os.path.join(model_path, f) for f in weights_filenames]
        weights = '\n\n'.join(weights_paths)
        self.setDetailedText(
            f'Files that {model_name} requires:\n\n'
            f'{weights}'
        )
        okButton = QPushButton('Ok')
        self.addButton(okButton, self.YesRole)
        okButton.disconnect()
        okButton.clicked.connect(self.close_)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def close_(self):
        self.hide()
        self.close()
        if self.loop is not None:
            self.loop.exit()

class warnVisualCppRequired(QMessageBox):
    def __init__(self, pkg_name='javabridge', parent=None):
        super().__init__(parent)
        self.loop = None
        self.screenShotWin = None

        self.setModal(False)
        self.setIcon(self.Warning)
        self.setWindowTitle(f'Installation of {pkg_name} info')
        self.setTextFormat(Qt.RichText)
        txt = (f"""
        <p style=font-size:12px>
            Installation of {pkg_name} on Windows requires
            Microsoft Visual C++ 14.0 or higher.<br><br>
            Cell-ACDC will anyway try to install {pkg_name} now.<br><br>
            If the installation fails, please <b>close Cell-ACDC</b>,
            then download and install <b>"Microsoft C++ Build Tools"</b>
            from the link below
            before trying this module again.<br><br>
            <a href='https://visualstudio.microsoft.com/visual-cpp-build-tools/'>
                https://visualstudio.microsoft.com/visual-cpp-build-tools/
            </a><br><br>
            <b>IMPORTANT</b>: when installing "Microsoft C++ Build Tools"
            make sure to select <b>"Desktop development with C++"</b>.
            Click "See the screenshot" for more details.
        </p>
        """)
        seeScreenshotButton = QPushButton('See screenshot...')
        okButton = QPushButton('Ok')
        self.addButton(okButton, self.YesRole)
        okButton.disconnect()
        okButton.clicked.connect(self.close_)
        self.addButton(seeScreenshotButton, self.HelpRole)
        seeScreenshotButton.disconnect()
        seeScreenshotButton.clicked.connect(
            self.viewScreenshot
        )
        self.setText(txt)

    def viewScreenshot(self, checked=False):
        self.screenShotWin = widgets.view_visualcpp_screenshot()
        self.screenShotWin.show()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def close_(self):
        self.hide()
        self.close()
        if self.loop is not None:
            self.loop.exit()
        if self.screenShotWin is not None:
            self.screenShotWin.close()



if __name__ == '__main__':
    # Create the application
    app = QApplication(sys.argv)
    font = QtGui.QFont()
    font.setPointSize(11)
    # title='Select channel name'
    # CbLabel='Select channel name:  '
    # informativeText = ''
    # win = QtSelectItems(title, ['mNeon', 'mKate'],
    #                     informativeText, CbLabel=CbLabel, parent=None)
    # win = edgeDetectionDialog(None)
    # win = QDialogEntriesWidget(entriesLabels=['Input 1'])
    IDs = list(range(1,10))
    cc_stage = ['G1' for ID in IDs]
    num_cycles = [-1]*len(IDs)
    relationship = ['mother' for ID in IDs]
    related_to = [-1]*len(IDs)
    is_history_known = [False]*len(IDs)
    corrected_assignment = [False]*len(IDs)
    cca_df = pd.DataFrame({
                       'cell_cycle_stage': cc_stage,
                       'generation_num': num_cycles,
                       'relative_ID': related_to,
                       'relationship': relationship,
                       'emerg_frame_i': num_cycles,
                       'division_frame_i': num_cycles,
                       'is_history_known': is_history_known,
                       'corrected_assignment': corrected_assignment},
                        index=IDs)
    cca_df.index.name = 'Cell_ID'
    #
    # df = cca_df.reset_index()
    #
    # win = pdDataFrameWidget(df)
    # win = QDialogMetadataXML(
    #     rawDataStruct=1, chNames=[''], ImageName='image'
    # )
    infoTxt = (
    """
        <p style=font-size:12px>
            Saving...<br>
        </p>
    """)
    ArgSpec = namedtuple('ArgSpec', ['name', 'default', 'type'])
    init_params = [ArgSpec(name='is_phase_contrast', default=True, type=bool)]
    segment_params = [
        ArgSpec(name='thresh_val', default=0.0, type=float),
        ArgSpec(name='min_distance', default=10, type=int)
    ]
    # win = QDialogModelParams(init_params, segment_params, 'YeaZ', url='None')
    # win = QDialogPbar(infoTxt=infoTxt)
    win = editID_QWidget(19, [19, 100, 50])
    # win = postProcessSegmDialog()
    # win = QDialogAppendTextFilename('example.npz')
    font = QtGui.QFont()
    font.setPointSize(11)
    filenames = ['test1', 'test2']
    # win = QDialogZsliceAbsent('test3', 30, filenames)
    win = QDialogMultiSegmNpz(filenames, os.path.dirname(__file__))
    # win = QDialogMetadata(
    #     1, 41, 180, 0.5, 0.09, 0.09, False, False, False,
    #     font=font, imgDataShape=(31, 350, 350)
    # )
    # win = cellpose_ParamsDialog()
    # user_ch_file_paths = [
    #     r"G:\My Drive\1_MIA_Data\Beno\test_QtGui\testGuiOnlyTifs\TIFFs\Position_1\Images\19-03-2021_KCY050_SCGE_s02_phase_contr.tif",
    #     r"G:\My Drive\1_MIA_Data\Beno\test_QtGui\testGuiOnlyTifs\TIFFs\Position_2\Images\19-03-2021_KCY050_SCGE_s02_phase_contr.tif"
    # ]
    # user_ch_name = 'phase_contr'
    # win = askStopFrameSegm(user_ch_file_paths, user_ch_name)
    # lab = np.load(r"G:\My Drive\1_MIA_Data\Test_data\Test_Qt_GUI\Position_5\Images\F016_s05_segm.npz")['arr_0'][0]
    # img = np.load(r"G:\My Drive\1_MIA_Data\Test_data\Test_Qt_GUI\Position_5\Images\F016_s05_phase_contr_aligned.npz")['arr_0'][0]
    # win = manualSeparateGui(lab, 2, img)
    # win.setFont(font)
    # win.progressLabel.setText('Preparing data...')
    app.setStyle(QtGui.QStyleFactory.create('Fusion'))
    # win.showAndSetWidth()
    # win.showAndSetFont(font)
    # win.setWidths(font=font)
    # win.setSize()
    # win.setGeometryWindow()
    # win.show()
    win.exec_()
    # print(win.chNames, win.saveChannels)
    # print(win.SizeT, win.SizeZ, win.zyx_vox_dim)
    # print(win.segment2D_kwargs)
