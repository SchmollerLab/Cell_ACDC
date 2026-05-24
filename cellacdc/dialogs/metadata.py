"""Cell-ACDC dialog windows: metadata."""

import os
import sys
import re
from typing import Literal, Callable, Dict, Iterable, List, Tuple
import datetime
import pathlib
from collections import defaultdict
import zipfile
from heapq import nlargest
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle, Circle, PathPatch, Path
import numpy as np
import scipy.interpolate

try:
    import tkinter as tk
except Exception as err:
    pass

import cv2
import traceback
from itertools import combinations, permutations
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import time
import sympy as sp
import json
import html

import pyqtgraph as pg

pg.setConfigOption("imageAxisOrder", "row-major")

from qtpy import QtCore
from qtpy.QtGui import (
    QIcon,
    QFontMetrics,
    QKeySequence,
    QFont,
    QRegularExpressionValidator,
    QCursor,
    QKeyEvent,
    QPixmap,
    QFont,
    QPalette,
    QMouseEvent,
    QColor,
)
from qtpy.QtCore import (
    Qt,
    QSize,
    QEvent,
    Signal,
    QEventLoop,
    QTimer,
    QRegularExpression,
)
from qtpy.QtWidgets import (
    QFileDialog,
    QApplication,
    QMainWindow,
    QMenu,
    QLabel,
    QToolBar,
    QScrollBar,
    QWidget,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QHBoxLayout,
    QDialog,
    QFormLayout,
    QListWidget,
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QSizePolicy,
    QComboBox,
    QSlider,
    QGridLayout,
    QSpinBox,
    QToolButton,
    QTableView,
    QTextBrowser,
    QDoubleSpinBox,
    QScrollArea,
    QFrame,
    QProgressBar,
    QGroupBox,
    QRadioButton,
    QDockWidget,
    QMessageBox,
    QStyle,
    QPlainTextEdit,
    QSpacerItem,
    QTreeWidget,
    QTreeWidgetItem,
    QTextEdit,
    QSplashScreen,
    QAction,
    QListWidgetItem,
    QActionGroup,
    QHeaderView,
    QStyledItemDelegate,
)
import qtpy.compat

from .. import exception_handler
from .. import load, prompts, core, measurements, html_utils
from .. import is_mac, is_win, is_linux, settings_folderpath, config
from .. import preproc_recipes_path, segm_recipes_path, combine_channels_recipes_path
from .. import is_conda_env
from .. import printl
from .. import colors
from .. import issues_url
from .. import myutils
from .. import qutils
from .. import _palettes
from .. import base_cca_dict
from .. import widgets
from .. import user_profile_path, promptable_models_path, models_path
from .. import features
from .. import _core
from .. import _types
from .. import plot
from .. import urls
from ..acdc_regex import float_regex, is_alphanumeric_filename, to_alphanumeric
from .. import _base_widgets
from .. import io
from .. import cca_functions
from .. import path

POSITIVE_FLOAT_REGEX = float_regex(allow_negative=False)
TREEWIDGET_STYLESHEET = _palettes.TreeWidgetStyleSheet()
LISTWIDGET_STYLESHEET = _palettes.ListWidgetStyleSheet()
BACKGROUND_RGBA = _palettes.get_disabled_colors()["Button"]

font = QFont()
font.setPixelSize(12)
italicFont = QFont()
italicFont.setPixelSize(12)
italicFont.setItalic(True)

from ._base import (
    QBaseDialog,
)

class filenameDialog(QDialog):
    def __init__(
        self,
        ext=".npz",
        basename="",
        title="Insert file name",
        hintText="",
        existingNames="",
        parent=None,
        allowEmpty=True,
        helpText="",
        defaultEntry="",
        resizeOnShow=True,
        additionalButtons=None,
        addDoNotSaveButton=False,
    ):
        self.cancel = True
        super().__init__(parent)

        self.resizeOnShow = resizeOnShow

        if hintText.find("segmentation") != -1:
            if helpText:
                helpText = f"{helpText}"
            helpText_loc = """
                With Cell-ACDC you can create as many segmentation files 
                <b>as you want</b>.<br><br>
                If you plan to create <b>only one file</b> then you can leave the 
                text entry <b>empty</b>.<br>
                Cell-ACDC will save the segmentation file with the filename 
                ending with <code>_segm.npz</code>.<br><br>
                However, <b>we recommend to insert some text</b> that will easily 
                allow you <b>to identify</b> what is the segmentation file about.<br><br>
                For example, if you are about to segment the channel 
                <code>phase_contr</code>, you could write 
                <code>phase_contr</code>.<br>
                Cell-ACDC will then save the file with the
                filename ending with <code>_segm_phase_contr.npz</code>.<br><br>
                This way you can create <b>multiple segmentation files</b>, 
                for example one for each channel or one for each segmentation model.<br><br>
                Note that the <b>numerical features and annotations</b> will be saved 
                in a CSV file ending with the same text as the segmentation file,<br> 
                e.g., ending with <code>_acdc_output_phase_contr.csv</code>.
            """
            helpText = f"{helpText}{html_utils.paragraph(helpText_loc)}"

        self.isSegmFile = basename.endswith("_segm")
        self.allowEmpty = allowEmpty
        self.basename = basename
        if ext and not ext.startswith("."):
            ext = f".{ext}"
        self.ext = ext

        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        entryLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        hintLabel = QLabel(hintText)

        basenameLabel = QLabel(basename)

        self.lineEdit = widgets.alphaNumericLineEdit(onlyWarn=True)
        self.lineEdit.setAlignment(Qt.AlignCenter)
        defaultEntry = to_alphanumeric(defaultEntry)
        defaultEntry = defaultEntry.replace(".", "_")
        self.lineEdit.setText(defaultEntry)

        extLabel = QLabel(ext)

        self.filenameLabel = QLabel()
        self.filenameLabel.setText(f"{basename}{ext}")

        entryLayout.addWidget(basenameLabel, 0, 1)
        entryLayout.addWidget(self.lineEdit, 0, 2)
        entryLayout.addWidget(extLabel, 0, 3)
        entryLayout.addWidget(self.filenameLabel, 1, 1, 1, 3, alignment=Qt.AlignCenter)
        # entryLayout.setColumnStretch(0, 1)
        entryLayout.setColumnStretch(2, 1)

        self.warningInvalidCharLabel = QLabel()

        okButton = widgets.okPushButton("Ok")
        cancelButton = widgets.cancelPushButton("Cancel")
        self.okButton = okButton

        buttonsLayout.addStretch()
        buttonsLayout.addWidget(cancelButton)

        if addDoNotSaveButton:
            doNotSaveButton = widgets.noPushButton("Do not save")
            doNotSaveButton.clicked.connect(self.doNotSave_cb)
            buttonsLayout.addWidget(doNotSaveButton)
            self.doNotSave = False

        buttonsLayout.addSpacing(20)
        if helpText:
            helpButton = widgets.helpPushButton("Help...")
            helpButton.clicked.connect(partial(self.showHelp, helpText))
            buttonsLayout.addWidget(helpButton)
        if additionalButtons is not None:
            for button in additionalButtons:
                buttonsLayout.addWidget(button)
        buttonsLayout.addWidget(okButton)

        cancelButton.clicked.connect(self.close)
        okButton.clicked.connect(self.ok_cb)
        self.lineEdit.textChanged.connect(self.updateFilename)
        self.lineEdit.sigInvalidCharactersEntered.connect(
            self.warnInvalidCharactersEntered
        )

        self.existingNames = []
        if existingNames:
            self.existingNames = existingNames
            # self.lineEdit.editingFinished.connect(self.checkExistingNames)

        layout.addWidget(hintLabel)
        layout.addSpacing(20)
        layout.addLayout(entryLayout)
        layout.addSpacing(10)
        layout.addWidget(self.warningInvalidCharLabel)
        layout.addStretch(1)
        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)

        self.setLayout(layout)
        self.setFont(font)

        if defaultEntry:
            self.updateFilename(defaultEntry)

    def doNotSave_cb(self):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            "Are you sure you do <b>not</b> want to save the file?"
        )
        noButton, yesButton = msg.warning(
            self, "Do not save?", txt, buttonsTexts=("No", "Yes")
        )
        if msg.clickedButton == noButton:
            return

        self.doNotSave = True
        self.cancel = False
        self.close()

    def showHelp(self, text):
        text = html_utils.paragraph(text)
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, "Filename help", text)

    def _text(self):
        return self.lineEdit.text()

    def warnInvalidCharactersEntered(self, characters: set[str]):
        statement = "is <b>not a valid</b> character"
        if len(characters) > 1:
            statement = "are <b>not valid</b> characters"

        characters_str = "".join(characters)
        characters_str = html.escape(characters_str)
        warning_text = html_utils.span(f"""
            WARNING: "<code>{characters_str}</code>" {statement}.<br> 
        """)
        warning_text = (
            f"{warning_text}"
            "<i>Valid characters are letters, numbers, underscore, and dash.</i>"
        )
        self.warningInvalidCharLabel.setText(warning_text)

    def checkExistingNames(self):
        is_existing = (
            self._text() in self.existingNames
            or self.filenameLabel.text() in self.existingNames
        )
        if not is_existing:
            return True

        filename = self.filenameLabel.text()
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            "The following file<br><br>"
            f"<code>{filename}</code><br><br>"
            "is <b>already existing</b>.<br><br>"
            "Do you want to <b>overwrite</b> the existing file?"
        )
        noButton, yesButton = msg.warning(
            self, "File name existing", txt, buttonsTexts=("No", "Yes")
        )
        return msg.clickedButton == yesButton

    def updateFilename(self, text):
        if self.lineEdit.invalidCharacters():
            return

        if not text:
            self.filenameLabel.setText(f"{self.basename}{self.ext}")
        else:
            text = text.replace(" ", "_")
            if self.basename:
                if self.basename.endswith("_"):
                    self.filenameLabel.setText(f"{self.basename}{text}{self.ext}")
                else:
                    self.filenameLabel.setText(f"{self.basename}_{text}{self.ext}")
            else:
                self.filenameLabel.setText(f"{text}{self.ext}")

        self.warningInvalidCharLabel.setText("")

    def checkEmptyText(self):
        if self.allowEmpty:
            return True

        if self._text():
            return True

        msg = widgets.myMessageBox()
        msg.critical(
            self,
            "Empty text",
            html_utils.paragraph("Text entry field <b>cannot be empty</b>"),
        )
        return False

    def checkSegmFilename(self):
        if not self.isSegmFile:
            return True

        if "segm" not in self._text():
            return True

        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            "The text appended to the filename cannot contain the text "
            '"segm".<br><br>'
            "Sorry, that would confuse me. Thank you for your patience!"
        )
        msg.critical(self, 'Cannot use "segm" in filename', txt)
        return False

    def ok_cb(self, checked=True):
        if self.warningInvalidCharLabel.text():
            return

        valid = self.checkExistingNames()
        if not valid:
            return

        valid = self.checkEmptyText()
        if not valid:
            return

        valid = self.checkSegmFilename()
        if not valid:
            return

        self.filename = self.filenameLabel.text()
        self.entryText = self._text()
        self.cancel = False
        self.close()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        if self.resizeOnShow:
            self.lineEdit.setMinimumWidth(self.lineEdit.width() * 2)
        self.okButton.setDefault(True)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()


class QDialogMetadataXML(QDialog):
    def __init__(
        self,
        title="Metadata",
        LensNA=1.0,
        rawFilename="test",
        SizeT=1,
        SizeZ=1,
        SizeC=1,
        SizeS=1,
        TimeIncrement=1.0,
        TimeIncrementUnit="s",
        PhysicalSizeX=1.0,
        PhysicalSizeY=1.0,
        PhysicalSizeZ=1.0,
        PhysicalSizeUnit="μm",
        ImageName="",
        chNames=None,
        emWavelens=None,
        parent=None,
        rawDataStruct=None,
        sampleImgData=None,
        rawFilePath=None,
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
        font = QFont()
        font.setPixelSize(12)
        self.setFont(font)

        mainLayout = QVBoxLayout()
        entriesLayout = QGridLayout()
        self.channelNameLayouts = (
            QVBoxLayout(),
            QVBoxLayout(),
            QVBoxLayout(),
            QVBoxLayout(),
        )
        self.channelEmWLayouts = (
            QVBoxLayout(),
            QVBoxLayout(),
            QVBoxLayout(),
            QVBoxLayout(),
        )
        buttonsLayout = QGridLayout()

        infoLabel = QLabel()
        infoTxt = "<b>Confirm/Edit</b> the <b>metadata</b> below."
        infoLabel.setText(infoTxt)
        # padding: top, left, bottom, right
        infoLabel.setStyleSheet("font-size:12pt; padding:0px 0px 5px 0px;")
        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        noteLabel = QLabel()
        noteLabel.setText(
            f"NOTE: If you are not sure about some of the entries "
            'you can try to click "Ok".\n'
            "If they are wrong you will get "
            "an error message later when trying to read the data."
        )
        noteLabel.setAlignment(Qt.AlignCenter)
        mainLayout.addWidget(noteLabel, alignment=Qt.AlignCenter)

        row = 0
        to_tif_radiobutton = QRadioButton(".tif")
        to_tif_radiobutton.setChecked(True)
        to_h5_radiobutton = QRadioButton(".h5")
        to_h5_radiobutton.setToolTip(
            ".h5 is highly recommended for big datasets to avoid memory issues.\n"
            "As a rule of thumb, if the single position, single channel file\n"
            "is larger than 1/5 of the available RAM we recommend using .h5 format"
        )
        self.to_h5_radiobutton = to_h5_radiobutton
        txt = "File format:  "
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
        txt = "Number of positions (SizeS):  "
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.SizeS_SB, row, 1)

        if rawDataStruct == 0:
            row += 1
            self.SizeS_SB.setValue(1)
            self.SizeS_SB.setDisabled(True)
            self.posSelector = widgets.ExpandableListBox()
            positions = ["All positions"]
            positions.extend([f"Position_{i + 1}" for i in range(SizeS)])
            self.posSelector.addItems(positions)
            txt = "Positions to save:  "
            label = QLabel(txt)
            entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
            entriesLayout.addWidget(self.posSelector, row, 1)
            self.SizeS_SB.valueChanged.connect(self.SizeSvalueChanged)

        row += 1
        self.LensNA_DSB = QDoubleSpinBox()
        self.LensNA_DSB.setAlignment(Qt.AlignCenter)
        self.LensNA_DSB.setSingleStep(0.1)
        self.LensNA_DSB.setValue(LensNA)
        txt = "Numerical Aperture Objective Lens:  "
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.LensNA_DSB, row, 1)

        row += 1
        self.SizeT_SB = QSpinBox()
        self.SizeT_SB.setAlignment(Qt.AlignCenter)
        self.SizeT_SB.setMinimum(1)
        self.SizeT_SB.setMaximum(2147483647)
        self.SizeT_SB.setValue(SizeT)
        txt = "Number of frames (SizeT):  "
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.SizeT_SB, row, 1)
        self.SizeT_SB.valueChanged.connect(self.hideShowTimeIncrement)

        row += 1
        self.timeRangeToSaveWidget = widgets.RangeSelector(integers=True)
        self.timeRangeToSaveWidget.setRange(1, SizeT)
        txt = "Time range to save:  "
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
        txt = "Number of z-slices in the z-stack (SizeZ):  "
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.SizeZ_SB, row, 1)
        self.SizeZ_SB.valueChanged.connect(self.hideShowPhysicalSizeZ)

        row += 1
        self.TimeIncrement_DSB = widgets.FloatLineEdit(
            allowNegative=False, warningValues={1.0}
        )
        self.TimeIncrement_DSB.setValue(TimeIncrement)
        self.TimeIncrement_DSB.setMinimum(0.0)
        txt = "Frame interval:  "
        label = QLabel(txt)
        self.TimeIncrement_Label = label
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.TimeIncrement_DSB, row, 1)

        self.TimeIncrementUnit_CB = QComboBox()
        unitItems = ["ms", "seconds", "minutes", "hours"]
        currentTxt = [unit for unit in unitItems if unit.startswith(TimeIncrementUnit)]
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
        txt = "Pixel width (X):  "
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.PhysicalSizeX_DSB, row, 1)

        self.PhysicalSizeUnit_CB = QComboBox()
        unitItems = ["nm", "μm", "mm", "cm"]
        currentTxt = [unit for unit in unitItems if unit.startswith(PhysicalSizeUnit)]
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
        txt = "Pixel height (Y):  "
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.PhysicalSizeY_DSB, row, 1)

        self.PhysicalSizeYUnit_Label = QLabel()
        self.PhysicalSizeYUnit_Label.setStyleSheet(
            "font-size:13px; padding:5px 0px 2px 0px;"
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
        txt = "Voxel depth (Z):  "
        self.PSZlabel = QLabel(txt)
        entriesLayout.addWidget(self.PSZlabel, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.PhysicalSizeZ_DSB, row, 1)

        self.PhysicalSizeZUnit_Label = QLabel()
        # padding: top, left, bottom, right
        self.PhysicalSizeZUnit_Label.setStyleSheet(
            "font-size:13px; padding:5px 0px 2px 0px;"
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
        txt = "Number of channels (SizeC):  "
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

        ext = "h5" if self.to_h5_radiobutton.isChecked() else "tif"
        for c in range(SizeC):
            chName_QLE = QLineEdit()
            chName_QLE.setStyleSheet("")
            chName_QLE.setAlignment(Qt.AlignCenter)
            chName_QLE.textChanged.connect(self.checkChNames)
            if chNames is not None:
                chName_QLE.setText(chNames[c])
            else:
                chName_QLE.setText(f"channel_{c}")
                filename = f""

            txt = f"Channel {c} name:  "
            label = QLabel(txt)

            filenameDescLabel = QLabel(f"<i>e.g., filename for channel {c}:  </i>")

            chName = chName_QLE.text()
            chName = self.removeInvalidCharacters(chName)
            rawFilename = self.elidedRawFilename()
            filenameLabel = QLabel(f"""
                <p style=font-size:10px>{rawFilename}_{chName}.{ext}</p>
            """)
            filenameLabel.setToolTip(f"{self.rawFilename}_{chName}.{ext}")

            checkBox = QCheckBox("Save this channel")
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
                addImageName_QCB = QCheckBox("Include image name")
                addImageName_QCB.stateChanged.connect(self.addImageName_cb)
                self.addImageName_QCB = addImageName_QCB
                self.channelNameLayouts[2].addWidget(addImageName_QCB)
            else:
                self.addImageName_QCB = QCheckBox("dummy")
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

            txt = f"Channel {c} emission wavelength:  "
            label = QLabel(txt)
            self.channelEmWLayouts[0].addWidget(label, alignment=Qt.AlignRight)
            self.channelEmWLayouts[1].addWidget(emWavelen_DSB)
            self.emWavelens_DSBs.append(emWavelen_DSB)

            unit = QLabel("nm")
            unit.setStyleSheet("font-size:13px; padding:5px 0px 2px 0px;")
            self.channelEmWLayouts[2].addWidget(unit)

        entriesLayout.setContentsMargins(0, 15, 0, 0)

        if rawDataStruct is None or rawDataStruct != -1:
            okButton = widgets.okPushButton(" Ok ")
        elif rawDataStruct == 1:
            okButton = QPushButton(" Load next position ")
        buttonsLayout.addWidget(okButton, 0, 1)

        self.trustButton = None
        self.overWriteButton = None
        if rawDataStruct == 1:
            trustButton = QPushButton(
                " Trust metadata reader\n for all next positions "
            )
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
                " Use the above metadata\n for all the next positions "
            )
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

        cancelButton = widgets.cancelPushButton("Cancel")
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
        LE.setDisabled(state == 0)
        label = self.channelNameLayouts[0].itemAt(idx).widget()
        if state == 0:
            label.setStyleSheet("color: gray; font-size: 10pt")
        else:
            label.setStyleSheet("color: black; font-size: 10pt")

        label = self.channelNameLayouts[0].itemAt(idx + 1).widget()
        if state == 0:
            label.setStyleSheet("color: gray; font-size: 10pt")
        else:
            label.setStyleSheet("color: black; font-size: 10pt")

        label = self.channelNameLayouts[1].itemAt(idx + 1).widget()
        if state == 0:
            label.setStyleSheet("color: gray; font-size: 10pt")
        else:
            label.setStyleSheet("color: black; font-size: 10pt")

    def addImageName_cb(self, state):
        for idx in range(self.SizeC_SB.value()):
            self.updateFilename(idx)

    def setInvalidChName_StyleSheet(self, LE):
        LE.setStyleSheet(
            "border-radius: 4px;border: 1.5px solid red;padding: 1px 0px 1px 0px"
        )

    def removeInvalidCharacters(self, chName):
        # Remove invalid charachters
        chName = "".join(
            c if c.isalnum() or c == "_" or c == "" else "_" for c in chName
        )
        trim_ = chName.endswith("_")
        while trim_:
            chName = chName[:-1]
            trim_ = chName.endswith("_")
        return chName

    def updateFileFormat(self, is_h5):
        for idx in range(len(self.chNames_QLEs)):
            self.updateFilename(idx)

    def SizeSvalueChanged(self, SizeS):
        positions = ["All positions"]
        positions.extend([f"Position_{i + 1}" for i in range(SizeS)])
        self.posSelector.setItems(positions)

    def elidedRawFilename(self):
        n = 31
        idx = int((n - 3) / 2)
        if len(self.rawFilename) > 21:
            elidedText = f"{self.rawFilename[:idx]}...{self.rawFilename[-idx:]}"
        else:
            elidedText = self.rawFilename
        return elidedText

    def updateFilename(self, idx):
        chName = self.chNames_QLEs[idx].text()
        chName = self.removeInvalidCharacters(chName)
        if self.rawDataStruct == 2:
            rawFilename = f"{self.rawFilename}_s{idx + 1}"
        else:
            rawFilename = self.rawFilename

        ext = "h5" if self.to_h5_radiobutton.isChecked() else "tif"

        rawFilename = self.elidedRawFilename()

        filenameLabel = self.filename_QLabels[idx]
        if self.addImageName_QCB.isChecked():
            self.ImageName = self.removeInvalidCharacters(self.ImageName)
            filename = f"""
                <p style=font-size:10px>
                    {rawFilename}_{self.ImageName}_{chName}.{ext}
                </p>
            """
            fullFilename = f"{self.rawFilename}_{self.ImageName}_{chName}.{ext}"
        else:
            filename = f"""
                <p style=font-size:10px>
                    {rawFilename}_{chName}.{ext}
                </p>
            """
            fullFilename = f"{self.rawFilename}_{chName}.{ext}"
        filenameLabel.setToolTip(fullFilename)
        filenameLabel.setText(filename)

    def checkChNames(self, text=""):
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
                LE1.setStyleSheet("")
                return areChNamesValid

            s1 = LE1.text()
            if not s1:
                self.setInvalidChName_StyleSheet(LE1)
                areChNamesValid = False
            else:
                LE1.setStyleSheet("")
            return areChNamesValid

        for LE1, LE2 in combinations(self.chNames_QLEs, 2):
            s1 = LE1.text()
            s2 = LE2.text()
            LE1_idx = self.chNames_QLEs.index(LE1)
            LE2_idx = self.chNames_QLEs.index(LE2)
            saveCh1 = self.saveChannels_QCBs[LE1_idx].isChecked()
            saveCh2 = self.saveChannels_QCBs[LE2_idx].isChecked()
            if not s1 or not s2 or s1 == s2:
                if not s1 and saveCh1:
                    self.setInvalidChName_StyleSheet(LE1)
                    areChNamesValid = False
                else:
                    LE1.setStyleSheet("")
                if not s2 and saveCh2:
                    self.setInvalidChName_StyleSheet(LE2)
                    areChNamesValid = False
                else:
                    LE2.setStyleSheet("")
                if s1 == s2 and saveCh1 and saveCh2:
                    self.setInvalidChName_StyleSheet(LE1)
                    self.setInvalidChName_StyleSheet(LE2)
                    areChNamesValid = False
            else:
                LE1.setStyleSheet("")
                LE2.setStyleSheet("")
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
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph("""
            Since you manually changed some of the metadata, this dialogue will now restart<br>
            because it <b>needs to read the image data again</b>.<br><br>
            Thank you for your patience.
        """)
        msg.warning(self, "Restart required", txt)

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
        dimsOrder = "ctz"
        imgData = self.sampleImgData[dimsOrder][idx]
        posData = myutils.utilClass()
        posData.frame_i = 0
        sampleSizeT = 4 if self.SizeT_SB.value() >= 4 else self.SizeT_SB.value()
        posData.SizeT = sampleSizeT
        SizeZ = self.SizeZ_SB.value()
        posData.SizeZ = 20 if SizeZ > 20 else SizeZ
        posData.filename = f"{self.rawFilename}_C={idx}"
        posData.segmInfo_df = pd.DataFrame(
            {
                "filename": [posData.filename] * sampleSizeT,
                "frame_i": range(sampleSizeT),
                "which_z_proj_gui": ["single z-slice"] * sampleSizeT,
                "z_slice_used_gui": [int(posData.SizeZ / 2)] * sampleSizeT,
            }
        ).set_index(["filename", "frame_i"])
        path_li = os.path.normpath(self.rawFilePath).split(os.sep)
        posData.relPath = f"{f'{os.sep}'.join(path_li[-3:1])}"
        posData.relPath = f"{posData.relPath}{os.sep}{posData.filename}"
        if sampleSizeT == 1:
            posData.img_data = [imgData]  # single frame data
        else:
            posData.img_data = imgData

        if self.imageViewer is not None:
            self.imageViewer.close()

        self.imageViewer = imageViewer(
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
        DeltaChannels = abs(value - currentSizeC)
        ext = "h5" if self.to_h5_radiobutton.isChecked() else "tif"
        if value > currentSizeC:
            for c in range(currentSizeC, currentSizeC + DeltaChannels):
                chName_QLE = QLineEdit()
                chName_QLE.setStyleSheet("")
                chName_QLE.setAlignment(Qt.AlignCenter)
                chName_QLE.setText(f"channel_{c}")
                chName_QLE.textChanged.connect(self.checkChNames)

                txt = f"Channel {c} name:  "
                label = QLabel(txt)

                filenameDescLabel = QLabel(f"<i>e.g., filename for channel {c}:  </i>")

                chName = chName_QLE.text()
                rawFilename = self.elidedRawFilename()
                filenameLabel = QLabel(f"""
                    <p style=font-size:10px>{rawFilename}_{chName}.{ext}</p>
                """)
                filenameLabel.setToolTip(f"{self.rawFilename}_{chName}.{ext}")

                checkBox = QCheckBox("Save this channel")
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
                unit = QLabel("nm")
                unit.setStyleSheet("font-size:13px; padding:5px 0px 2px 0px;")

                txt = f"Channel {c} emission wavelength:  "
                label = QLabel(txt)
                self.channelEmWLayouts[0].addWidget(label, alignment=Qt.AlignRight)
                self.channelEmWLayouts[1].addWidget(emWavelen_DSB)
                self.channelEmWLayouts[2].addWidget(unit)
                self.emWavelens_DSBs.append(emWavelen_DSB)
        else:
            for c in range(currentSizeC, currentSizeC + DeltaChannels):
                idx = (c - 1) * 2
                label1 = self.channelNameLayouts[0].itemAt(idx).widget()
                label2 = self.channelNameLayouts[0].itemAt(idx + 1).widget()
                chName_QLE = self.channelNameLayouts[1].itemAt(idx).widget()
                filename_L = self.channelNameLayouts[1].itemAt(idx + 1).widget()
                checkBox = self.channelNameLayouts[2].itemAt(idx).widget()
                dummyLabel = self.channelNameLayouts[2].itemAt(idx + 1).widget()
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

                label = self.channelEmWLayouts[0].itemAt(c - 1).widget()
                emWavelen_DSB = self.channelEmWLayouts[1].itemAt(c - 1).widget()
                unit = self.channelEmWLayouts[2].itemAt(c - 1).widget()
                self.channelEmWLayouts[0].removeWidget(label)
                self.channelEmWLayouts[1].removeWidget(emWavelen_DSB)
                self.channelEmWLayouts[2].removeWidget(unit)
                self.emWavelens_DSBs.pop(-1)

                self.adjustSize()

    def ok_cb(self, event):
        areChNamesValid = self.checkChNames()
        if not areChNamesValid:
            err_msg = html_utils.paragraph(
                "Channel names <b>cannot be empty</b> or equal to each other."
                "<br><br>"
                "Insert a unique text for each channel name."
            )
            msg = widgets.myMessageBox()
            msg.critical(self, "Invalid channel names", err_msg)
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
        if hasattr(self, "posSelector"):
            self.selectedPos = self.posSelector.selectedItemsText()
        else:
            self.selectedPos = ["All Positions"]
        self.chNames = []
        if hasattr(self, "addImageName_QCB"):
            self.addImageName = self.addImageName_QCB.isChecked()
        else:
            self.addImageName = False
        self.saveChannels = []
        for LE, QCB in zip(self.chNames_QLEs, self.saveChannels_QCBs):
            s = LE.text()
            s = "".join(c if c.isalnum() or c == "_" or c == "" else "_" for c in s)
            trim_ = s.endswith("_")
            while trim_:
                s = s[:-1]
                trim_ = s.endswith("_")
            self.chNames.append(s)
            self.saveChannels.append(QCB.isChecked())
        self.emWavelens = [DSB.value() for DSB in self.emWavelens_DSBs]

    def convertUnits(self):
        timeUnit = self.TimeIncrementUnit_CB.currentText()
        if timeUnit == "ms":
            self.TimeIncrement /= 1000
        elif timeUnit == "minutes":
            self.TimeIncrement *= 60
        elif timeUnit == "hours":
            self.TimeIncrement *= 3600

        PhysicalSizeUnit = self.PhysicalSizeUnit_CB.currentText()
        if timeUnit == "nm":
            self.PhysicalSizeX /= 1000
            self.PhysicalSizeY /= 1000
            self.PhysicalSizeZ /= 1000
        elif timeUnit == "mm":
            self.PhysicalSizeX *= 1000
            self.PhysicalSizeY *= 1000
            self.PhysicalSizeZ *= 1000
        elif timeUnit == "cm":
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
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()
        self.setSize()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()


class MultiTimePointFilePattern(QBaseDialog):
    def __init__(self, fileName, folderPath, readPatternFunc=None, parent=None):
        super().__init__(parent)

        self.setWindowTitle("File name pattern")
        self.cancel = True
        self.additionalChannelWidgets = {}

        mainLayout = QVBoxLayout()
        self.readPatternFunc = readPatternFunc

        infoText = html_utils.paragraph("""
            The image files for each time-point <b>must be named with the following pattern:</b><br><br>
            <code>position_channel_timepoint</code>
            <br><br>
            For example a file with name "<code>pos1_GFP_1.tif</code>" would be the first time-point of the channell GFP<br>
            and position called <code>pos1</code>.<br><br>
            The Position number will be determined by <b>alphabetically sorting</b>
            all the image files.<br><br>
            Please, <b>provide the channel names</b> below. 
            Optionally, you can provide a basename<br>
            that will be pre-pended to the name of all created files.<br><br>
            You can also provide a folder path containing the segmentation masks file.<br>
            These files <b>MUST be named exactly as the raw files</b>.
            <br>
        """)

        noteLayout = QHBoxLayout()
        noteText = html_utils.paragraph("""
            Channels <em>do not need to have the same number of frames</em>, 
            however, Cell-ACDC will place<br>
            the frames at the right frame number 
            (given by <code>timepoint</code> number at the end<br>
            of the filename) and it will fill missing frames with zeros.
        """)
        noteLayout.addWidget(
            QLabel(html_utils.to_admonition(noteText)),
            # alignment=(Qt.AlignTop | Qt.AlignRight)
        )

        mainLayout.addWidget(QLabel(infoText))
        mainLayout.addLayout(noteLayout)
        noteLayout.setStretch(0, 0)
        noteLayout.setStretch(1, 1)

        label = QLabel(
            html_utils.paragraph(f"Sample file name: <code>{fileName}</code>")
        )
        mainLayout.addWidget(label, alignment=Qt.AlignCenter)
        mainLayout.addSpacing(5)

        channelName = ""
        posName = ""
        frameNumber = None
        if readPatternFunc is not None:
            posName, frameNumber, channelName = readPatternFunc(fileName)

        formLayout = QGridLayout()

        ncols = 3
        self.vLayouts = [QVBoxLayout() for _ in range(ncols)]
        for j, l in enumerate(self.vLayouts):
            formLayout.addLayout(l, 0, j)

        row = 0
        items = QLabel("Position name: "), widgets.ReadOnlyLineEdit(), QLabel()
        label, self.posNameEntry, button = items
        self.posNameEntry.setAlignment(Qt.AlignCenter)
        self.posNameEntry.setText(str(posName))
        for j, w in enumerate(items):
            self.vLayouts[j].addWidget(w)

        row += 1
        items = (QLabel("Frame number name: "), widgets.ReadOnlyLineEdit(), QLabel())
        self.frameNumberEntry = items[1]
        self.frameNumberEntry.setText(str(frameNumber))
        self.frameNumberEntry.setAlignment(Qt.AlignCenter)
        for j, w in enumerate(items):
            self.vLayouts[j].addWidget(w)

        row += 1
        self.channelNameLE = widgets.alphaNumericLineEdit()
        items = (
            QLabel("Channel_1 name: "),
            self.channelNameLE,
            widgets.addPushButton(" Add channel"),
        )
        self.addChannelButton = items[2]
        self.addChannelButton._row = row
        self.channelNameLE.setAlignment(Qt.AlignCenter)
        self.channelNameLE.setText(channelName)
        for j, w in enumerate(items):
            self.vLayouts[j].addWidget(w)

        row += 1
        items = (
            QLabel("Basename (optional): "),
            widgets.alphaNumericLineEdit(),
            QLabel(),
        )
        label, self.baseNameLE, button = items
        self.baseNameLE.setAlignment(Qt.AlignCenter)
        for j, w in enumerate(items):
            self.vLayouts[j].addWidget(w)

        row += 1
        items = QLabel("File will be saved as: "), QLineEdit(), QLabel()
        label, self.relPathEntry, button = items
        self.relPathEntry.setAlignment(Qt.AlignCenter)
        for j, w in enumerate(items):
            self.vLayouts[j].addWidget(w)

        row += 1
        items = (
            QLabel("Segmentation masks folder path: "),
            widgets.ElidingLineEdit(),
            widgets.browseFileButton(
                "Browse...",
                title="Select folder containing segmentation masks",
                start_dir=folderPath,
                openFolder=True,
            ),
        )
        label, self.segmFolderPathEntry, button = items
        button.sigPathSelected.connect(self.segmFolderpathSelected)
        self.segmFolderPathEntry.setAlignment(Qt.AlignCenter)
        for j, w in enumerate(items):
            self.vLayouts[j].addWidget(w)

        self.formLayout = formLayout

        self.updateRelativePath()

        self.channelNameLE.textChanged.connect(self.updateRelativePath)
        self.baseNameLE.textChanged.connect(self.updateRelativePath)
        self.addChannelButton.clicked.connect(self.addChannel)

        mainLayout.addLayout(formLayout)

        buttonsLayout = widgets.CancelOkButtonsLayout()
        showInFileManagerButton = widgets.showInFileManagerButton(
            myutils.get_open_filemaneger_os_string()
        )
        buttonsLayout.insertWidget(3, showInFileManagerButton)
        func = partial(myutils.showInExplorer, folderPath)
        showInFileManagerButton.clicked.connect(func)
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch()

        self.setLayout(mainLayout)

        self.setFont(font)

    def segmFolderpathSelected(self, path):
        self.segmFolderPathEntry.setText(path)

    def addChannel(self):
        self.addChannelButton._row += 1
        row = self.addChannelButton._row

        channel_idx = len(self.additionalChannelWidgets)
        items = (
            QLabel(f"Channel_{channel_idx + 1} name: "),
            widgets.alphaNumericLineEdit(),
            widgets.subtractPushButton("Remove channel"),
        )
        label, lineEdit, button = items
        lineEdit.setAlignment(Qt.AlignCenter)
        button.clicked.connect(self.removeChannel)
        button._row = row
        for j, w in enumerate(items):
            self.vLayouts[j].insertWidget(row, w)

        self.additionalChannelWidgets[row] = items
        lineEdit.setFocus()

    def removeChannel(self):
        row = self.sender()._row
        for j, w in enumerate(self.additionalChannelWidgets[row]):
            self.vLayouts[j].removeWidget(w)

        self.additionalChannelWidgets.pop(row)
        self.addChannelButton._row -= 1

    def checkChannelNames(self):
        allChannels = [self.channelNameLE.text()]
        allChannels.extend(
            [w[1].text() for w in self.additionalChannelWidgets.values()]
        )
        for ch1, ch2 in combinations(allChannels, 2):
            if ch1 == ch2:
                break
            if not ch1 or not ch2:
                break
        else:
            # Channel names are fine
            return allChannels

        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        txt = html_utils.paragraph("""
            Some channel names are empty or not different from each other.
        """)
        msg.critical(self, "Select two or more items", txt)
        return None

    def updateRelativePath(self, text=""):
        posName = self.posNameEntry.text()
        frameNumber = self.frameNumberEntry.text()
        channelName = self.channelNameLE.text()
        basename = self.baseNameLE.text()
        if basename:
            filename = f"{basename}_{posName}_{channelName}.tif"
        else:
            filename = f"{posName}_{channelName}.tif"
        relPath = f"...{os.sep}Position_1{os.sep}Images{os.sep}{filename}"
        self.relPathEntry.setText(relPath)

    def ok_cb(self):
        allChannels = self.checkChannelNames()
        if allChannels is None:
            return
        self.allChannels = allChannels
        self.basename = self.baseNameLE.text()
        self.segmFolderPath = self.segmFolderPathEntry.text()
        self.cancel = False
        self.close()

    def showEvent(self, event) -> None:
        self.channelNameLE.setFocus()


class OrderableListWidgetDialog(QBaseDialog):
    def __init__(
        self, items, title="Select items", infoTxt="", helpText="", parent=None
    ):
        super().__init__(parent)

        self.selectedItemsText = []

        self.cancel = True
        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()
        self.helpText = helpText

        if infoTxt:
            mainLayout.addWidget(QLabel(html_utils.paragraph(infoTxt)))

        self.listWidget = widgets.OrderableList()
        self.listWidget.addItems(items)

        buttonsLayout = widgets.CancelOkButtonsLayout()
        if helpText:
            helpButton = widgets.helpPushButton("Help...")
            buttonsLayout.insertWidget(3, helpButton)
            helpButton.clicked.connect(self.showHelp)

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(self.listWidget)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def showHelp(self):
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph(self.helpText)
        msg.information(self, "Select tables help", txt)

    def ok_cb(self):
        self.cancel = False
        self.selectedItemsText = [None] * len(self.listWidget.selectedItems())
        for itemW in self.listWidget.selectedItems():
            idx = int(itemW._nrWidget.currentText()) - 1
            if idx >= len(self.selectedItemsText):
                idx = len(self.selectedItemsText) - 1
            self.selectedItemsText[idx] = itemW._text
        self.close()


class QDialogAppendTextFilename(QDialog):
    def __init__(self, filename, ext, parent=None, font=None):
        super().__init__(parent)
        self.cancel = True
        filenameNOext, _ = os.path.splitext(filename)
        self.filenameNOext = filenameNOext
        if ext.find(".") == -1:
            ext = f".{ext}"
        self.ext = ext

        self.setWindowTitle("Append text to file name")

        mainLayout = QVBoxLayout()
        formLayout = QFormLayout()
        buttonsLayout = QHBoxLayout()

        if font is not None:
            self.setFont(font)

        self.LE = QLineEdit()
        self.LE.setAlignment(Qt.AlignCenter)
        formLayout.addRow("Appended text", self.LE)
        self.LE.textChanged.connect(self.updateFinalFilename)

        self.finalName_label = QLabel(f'Final file name: "{filenameNOext}_{ext}"')
        # padding: top, left, bottom, right
        self.finalName_label.setStyleSheet("font-size:13px; padding:5px 0px 0px 0px;")

        okButton = widgets.okPushButton("Ok")
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = widgets.cancelPushButton("Cancel")

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

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
        finalFilename = f"{self.filenameNOext}_{text}{self.ext}"
        self.finalName_label.setText(f'Final file name: "{finalFilename}"')

    def ok_cb(self, event):
        if not self.LE.text():
            err_msg = "Appended name cannot be empty!"
            msg = QMessageBox()
            msg.critical(self, "Empty name", err_msg, msg.Ok)
            return
        self.cancel = False
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
        if hasattr(self, "loop"):
            self.loop.exit()


class QDialogEntriesWidget(QDialog):
    def __init__(
        self, entriesLabels, defaultTxts, winTitle="Input", parent=None, font=None
    ):
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

        okButton = widgets.okPushButton("Ok")
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = widgets.cancelPushButton("Cancel")

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

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
        self.entriesTxt = [
            self.formLayout.itemAt(i, 1).widget().text()
            for i in range(len(self.entriesLabels))
        ]
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
        if hasattr(self, "loop"):
            self.loop.exit()


class QDialogMetadata(QDialog):
    def __init__(
        self,
        SizeT,
        SizeZ,
        TimeIncrement,
        PhysicalSizeZ,
        PhysicalSizeY,
        PhysicalSizeX,
        ask_SizeT,
        ask_TimeIncrement,
        ask_PhysicalSizes,
        parent=None,
        font=None,
        imgDataShape=None,
        posData=None,
        singlePos=False,
        askSegm3D=True,
        additionalValues=None,
        forceEnableAskSegm3D=False,
        SizeT_metadata=None,
        SizeZ_metadata=None,
        basename="",
    ):
        self.cancel = True
        self.ask_TimeIncrement = ask_TimeIncrement
        self.ask_PhysicalSizes = ask_PhysicalSizes
        self.askSegm3D = askSegm3D
        self.imgDataShape = imgDataShape
        self.posData = posData
        self._additionalValues = additionalValues
        self.SizeT_metadata = SizeT_metadata
        self.SizeZ_metadata = SizeZ_metadata
        super().__init__(parent)
        self.setWindowTitle("Image properties")

        mainLayout = QVBoxLayout()
        gridLayout = QGridLayout()
        # formLayout = QFormLayout()
        buttonsLayout = QGridLayout()

        if imgDataShape is not None:
            label = QLabel(
                html_utils.paragraph(
                    f"<i>Image data shape</i> = <b>{imgDataShape}</b><br>"
                )
            )
            mainLayout.addWidget(label, alignment=Qt.AlignCenter)

        row = 0
        self.basenameLineEdit = None
        if basename:
            gridLayout.addWidget(
                QLabel("Basename (read-only)"), row, 0, alignment=Qt.AlignRight
            )
            self.basenameLineEdit = QLineEdit()
            self.basenameLineEdit.setReadOnly(True)
            self.basenameLineEdit.setText(basename)
            minWidth = (
                self.basenameLineEdit.fontMetrics().boundingRect(basename).width() + 10
            )
            self.basenameLineEdit.setMinimumWidth(minWidth)
            self.basenameLineEdit.setAlignment(Qt.AlignCenter)
            gridLayout.addWidget(self.basenameLineEdit, row, 1)
            row += 1

        gridLayout.addWidget(
            QLabel("Number of frames (SizeT)"), row, 0, alignment=Qt.AlignRight
        )
        self.SizeT_SpinBox = QSpinBox()
        self.SizeT_SpinBox.setMinimum(1)
        self.SizeT_SpinBox.setMaximum(2147483647)
        SizeTinfoButton = widgets.infoPushButton()
        self.allowEditSizeTcheckbox = QCheckBox("Let me edit it")
        if ask_SizeT:
            self.SizeT_SpinBox.setValue(SizeT)
            SizeTinfoButton.hide()
            self.allowEditSizeTcheckbox.hide()
        else:
            self.SizeT_SpinBox.setValue(1)
            self.SizeT_SpinBox.setDisabled(True)
            SizeTinfoButton.show()
            SizeTinfoButton.clicked.connect(self.showWhySizeTisGrayed)
            self.allowEditSizeTcheckbox.show()
            self.allowEditSizeTcheckbox.toggled.connect(self.allowEditSizeT)
        self.SizeT_SpinBox.setAlignment(Qt.AlignCenter)
        self.SizeT_SpinBox.valueChanged.connect(self.TimeIncrementShowHide)
        gridLayout.addWidget(self.SizeT_SpinBox, row, 1)
        gridLayout.addWidget(SizeTinfoButton, row, 2)
        gridLayout.setColumnStretch(2, 0)
        gridLayout.addWidget(self.allowEditSizeTcheckbox, row, 3)
        gridLayout.setColumnStretch(3, 0)

        row += 1
        gridLayout.addWidget(
            QLabel("Number of z-slices (SizeZ)"), row, 0, alignment=Qt.AlignRight
        )
        self.SizeZ_SpinBox = QSpinBox()
        self.SizeZ_SpinBox.setMinimum(1)
        self.SizeZ_SpinBox.setMaximum(2147483647)
        self.SizeZ_SpinBox.setValue(SizeZ)
        self.SizeZ_SpinBox.setAlignment(Qt.AlignCenter)
        self.SizeZ_SpinBox.valueChanged.connect(self.SizeZvalueChanged)
        gridLayout.addWidget(self.SizeZ_SpinBox, row, 1)

        row += 1
        self.TimeIncrementLabel = QLabel("Time interval (s)")
        gridLayout.addWidget(self.TimeIncrementLabel, row, 0, alignment=Qt.AlignRight)
        self.TimeIncrementSpinBox = widgets.FloatLineEdit()
        self.TimeIncrementSpinBox.setValue(TimeIncrement)
        gridLayout.addWidget(self.TimeIncrementSpinBox, row, 1)

        if SizeT == 1 or not ask_TimeIncrement:
            self.TimeIncrementSpinBox.hide()
            self.TimeIncrementLabel.hide()

        row += 1
        self.PhysicalSizeZLabel = QLabel("Physical Size Z (um/pixel)")
        gridLayout.addWidget(self.PhysicalSizeZLabel, row, 0, alignment=Qt.AlignRight)
        self.PhysicalSizeZSpinBox = widgets.FloatLineEdit()
        self.PhysicalSizeZSpinBox.setValue(PhysicalSizeZ)
        gridLayout.addWidget(self.PhysicalSizeZSpinBox, row, 1)

        if SizeZ == 1 or not ask_PhysicalSizes:
            self.PhysicalSizeZSpinBox.hide()
            self.PhysicalSizeZLabel.hide()

        row += 1
        self.PhysicalSizeYLabel = QLabel("Physical Size Y (um/pixel)")
        gridLayout.addWidget(self.PhysicalSizeYLabel, row, 0, alignment=Qt.AlignRight)
        self.PhysicalSizeYSpinBox = widgets.FloatLineEdit()
        self.PhysicalSizeYSpinBox.setValue(PhysicalSizeY)
        gridLayout.addWidget(self.PhysicalSizeYSpinBox, row, 1)

        if not ask_PhysicalSizes:
            self.PhysicalSizeYSpinBox.hide()
            self.PhysicalSizeYLabel.hide()

        row += 1
        self.PhysicalSizeXLabel = QLabel("Physical Size X (um/pixel)")
        gridLayout.addWidget(self.PhysicalSizeXLabel, row, 0, alignment=Qt.AlignRight)
        self.PhysicalSizeXSpinBox = widgets.FloatLineEdit()
        self.PhysicalSizeXSpinBox.setValue(PhysicalSizeX)
        gridLayout.addWidget(self.PhysicalSizeXSpinBox, row, 1)

        if not ask_PhysicalSizes:
            self.PhysicalSizeXSpinBox.hide()
            self.PhysicalSizeXLabel.hide()

        row += 1
        self.isSegm3Dtoggle = widgets.Toggle()
        if posData is not None:
            self.isSegm3Dtoggle.setChecked(posData.getIsSegm3D())
            disableToggle = (
                # Disable toggle if not force enable and if
                # segm data was found (we cannot change the shape of
                # loaded segmentation in the GUI)
                posData.segmFound is not None
                and posData.segmFound
                and not forceEnableAskSegm3D
            )
            if disableToggle:
                self.isSegm3Dtoggle.setDisabled(True)
        self.isSegm3DLabel = QLabel("Work with 3D segmentation masks (z-stack)")
        gridLayout.addWidget(self.isSegm3DLabel, row, 0, alignment=Qt.AlignRight)
        gridLayout.addWidget(self.isSegm3Dtoggle, row, 1, alignment=Qt.AlignCenter)
        self.infoButtonSegm3D = QPushButton(self)
        self.infoButtonSegm3D.setCursor(Qt.WhatsThisCursor)
        self.infoButtonSegm3D.setIcon(QIcon(":info.svg"))
        gridLayout.addWidget(self.infoButtonSegm3D, row, 2, alignment=Qt.AlignLeft)
        self.infoButtonSegm3D.clicked.connect(self.infoSegm3D)
        if SizeZ == 1 or not askSegm3D:
            self.isSegm3DLabel.hide()
            self.isSegm3Dtoggle.hide()
            self.infoButtonSegm3D.hide()

        self.SizeZvalueChanged(SizeZ)

        self.additionalFieldsWidgets = []
        addFieldButton = widgets.addPushButton("Add custom field")
        addFieldInfoButton = widgets.infoPushButton()
        addFieldInfoButton.clicked.connect(self.showAddFieldInfo)
        addFieldButton.clicked.connect(self.addField)
        addFieldLayout = QHBoxLayout()
        addFieldLayout.addStretch(1)
        addFieldLayout.addWidget(addFieldButton)
        addFieldLayout.addWidget(addFieldInfoButton)
        addFieldLayout.addStretch(1)

        if singlePos:
            okTxt = "Apply only to this Position"
        else:
            okTxt = "Ok for loaded Positions"
        okButton = widgets.okPushButton(okTxt)
        okButton.setToolTip("Save metadata only for current positionh")
        okButton.setShortcut(Qt.Key_Enter)
        self.okButton = okButton

        if ask_TimeIncrement or ask_PhysicalSizes:
            okAllButton = QPushButton("Apply to ALL Positions")
            okAllButton.setToolTip(
                "Update existing Physical Sizes, Time interval, cell volume (fl), "
                "cell area (um^2), and time (s) for all the positions "
                "in the experiment folder."
            )
            self.okAllButton = okAllButton

            selectButton = QPushButton("Select the Positions to be updated")
            selectButton.setToolTip(
                "Ask to select positions then update existing Physical Sizes, "
                "Time interval, cell volume (fl), cell area (um^2), and time (s)"
                "for selected positions."
            )
            self.selectButton = selectButton
        else:
            self.okAllButton = None
            self.selectButton = None
            okButton.setText("Ok")

        cancelButton = widgets.cancelPushButton("Cancel")

        buttonsLayout.setColumnStretch(0, 1)
        buttonsLayout.addWidget(okButton, 0, 1)
        if ask_TimeIncrement or ask_PhysicalSizes:
            buttonsLayout.addWidget(okAllButton, 0, 2)
            buttonsLayout.addWidget(selectButton, 1, 1)
            buttonsLayout.addWidget(cancelButton, 1, 2)
        else:
            buttonsLayout.addWidget(cancelButton, 0, 2)
        buttonsLayout.setColumnStretch(3, 1)

        gridLayout.setColumnMinimumWidth(1, 100)
        mainLayout.addLayout(gridLayout)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(addFieldLayout)
        # mainLayout.addLayout(formLayout)
        mainLayout.addSpacing(20)
        mainLayout.addStretch(1)
        mainLayout.addLayout(buttonsLayout)
        self.mainLayout = mainLayout

        okButton.clicked.connect(self.ok_cb)
        if ask_TimeIncrement or ask_PhysicalSizes:
            okAllButton.clicked.connect(self.ok_cb)
            selectButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        self.addAdditionalValues(additionalValues)

        self.setLayout(mainLayout)
        self.setFont(font)
        # self.setModal(True)

    def showWhySizeTisGrayed(self):
        txt = html_utils.paragraph(f"""
            The "Number of frames" field is grayed-out because you loaded multiple Positions.<br><br>
            Cell-ACDC <b>cannot load multiple time-lapse Positions</b>, 
            so it is assuming you are loading NON time-lapse data.<br><br>
            To load time-lapse data, load <b>one Position at a time</b>.<br><br>
            Note that you can still edit the number of frames if you need to correct it.<br>
            However, <b>you can only edit the metadata</b>, then the loading process will be stopped.
        """)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.information(self, "Why is the number of frames grayed out?", txt)

    def addAdditionalValues(self, values):
        if values is None:
            return

        for i, (name, value) in enumerate(values.items()):
            self.addField()
            nameWidget = self.additionalFieldsWidgets[i]["nameWidget"]
            valueWidget = self.additionalFieldsWidgets[i]["valueWidget"]
            nameWidget.setText(str(name).strip("__"))
            valueWidget.setText(str(value))

    def addField(self):
        nameWidget = QLineEdit()
        nameWidget.setAlignment(Qt.AlignCenter)
        valueWidget = QLineEdit()
        valueWidget.setAlignment(Qt.AlignCenter)
        removeButton = widgets.delPushButton()

        fieldLayout = QGridLayout()
        fieldLayout.addWidget(QLabel("Name"), 0, 0)
        fieldLayout.addWidget(nameWidget, 1, 0)
        fieldLayout.addWidget(QLabel("Value"), 0, 1)
        fieldLayout.addWidget(valueWidget, 1, 1)
        fieldLayout.addWidget(removeButton, 1, 2)

        self.additionalFieldsWidgets.append(
            {
                "nameWidget": nameWidget,
                "valueWidget": valueWidget,
                "removeButton": removeButton,
                "layout": fieldLayout,
            }
        )

        idx = len(self.additionalFieldsWidgets) - 1
        removeButton.clicked.connect(partial(self.removeField, idx))

        row = self.mainLayout.count() - 3
        self.mainLayout.insertLayout(row, fieldLayout)

    def removeField(self, idx):
        widgets = self.additionalFieldsWidgets[idx]

        layoutToRemove = widgets["layout"]
        for row in range(layoutToRemove.rowCount()):
            for col in range(layoutToRemove.columnCount()):
                item = layoutToRemove.itemAtPosition(row, col)
                if item is not None:
                    widget = item.widget()
                    layoutToRemove.removeWidget(widget)

        self.additionalFieldsWidgets.pop(idx)

        self.mainLayout.removeItem(layoutToRemove)

    def showAddFieldInfo(self):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            Add a <b>field (name and value)</b> that will be saved to the
            <code>metadata.csv</code> file and as a column in the
            <code>acdc_output.csv</code> table.<br><br>
            Example: a strain name or the replicate number.
        """)
        msg.information(self, "Add field info", txt)

    def infoSegm3D(self):
        txt = (
            "Cell-ACDC supports both <b>2D and 3D segmentation</b>. If your data "
            "also have a time dimension, then you can choose to segment "
            "a specific z-slice (2D segmentation mask per frame) or all of them "
            "(3D segmentation mask per frame)<br><br>"
            "In any case, if you choose to activate <b>3D segmentation</b> then the "
            "segmentation mask will have the <b>same number of z-slices "
            "of the image data</b>.<br><br>"
            "Additionally, in the model parameters window, you will be able "
            "to choose if you want to segment the <b>entire 3D volume at once</b> "
            "or use the <b>2D model on each z-slice</b>, one by one.<br><br>"
            "<i>NOTE: if the toggle is disabled it means you already "
            "loaded segmentation data and the shape cannot be changed now.<br>"
            "if you need to start with a blank segmentation, "
            'use the "Create a new segmentation file" button instead of the '
            '"Load folder" button.'
            "</i>"
        )
        msg = widgets.myMessageBox()
        msg.setIcon()
        msg.setWindowTitle(f"3D segmentation info")
        msg.addText(html_utils.paragraph(txt))
        msg.addButton("   Ok   ")
        msg.exec_()

    def SizeZvalueChanged(self, val):
        if len(self.imgDataShape) < 3:
            return

        if val > 1 and self.imgDataShape is not None:
            maxSizeZ = self.imgDataShape[-3]
            self.SizeZ_SpinBox.setMaximum(maxSizeZ)
        else:
            self.SizeZ_SpinBox.setMaximum(2147483647)

        if val > 1:
            if self.ask_PhysicalSizes:
                self.PhysicalSizeZSpinBox.show()
                self.PhysicalSizeZLabel.show()
            if self.askSegm3D:
                self.isSegm3DLabel.show()
                self.isSegm3Dtoggle.show()
                self.infoButtonSegm3D.show()
        else:
            self.PhysicalSizeZSpinBox.hide()
            self.PhysicalSizeZLabel.hide()
            self.isSegm3DLabel.hide()
            self.isSegm3Dtoggle.hide()
            self.infoButtonSegm3D.hide()

        self.checkSegmDataShape()

    def checkSegmDataShape(self):
        if self.posData is None:
            return

        if self.isSegm3Dtoggle.isEnabled():
            return

        SizeT = self.SizeT_SpinBox.value()
        SizeZ = self.SizeZ_SpinBox.value()
        segm_data_ndim = self.posData.segm_data.ndim
        isSegm3D = False
        if segm_data_ndim == 4:
            # Segm data is 4D so it must be 3D over time
            isSegm3D = True
        elif segm_data_ndim == 3 and SizeZ > 1 and SizeT == 1:
            # Segm data is 3D while SizeT == 1 and SizeZ > 1
            # --> also segm is 3D z-stack
            isSegm3D = True

        self.isSegm3Dtoggle.setDisabled(False)
        self.isSegm3Dtoggle.setChecked(isSegm3D)
        self.isSegm3Dtoggle.setDisabled(True)

    def TimeIncrementShowHide(self, val):
        self.checkSegmDataShape()
        if not self.ask_TimeIncrement:
            return

        if val > 1:
            self.TimeIncrementSpinBox.show()
            self.TimeIncrementLabel.show()
        else:
            self.TimeIncrementSpinBox.hide()
            self.TimeIncrementLabel.hide()

    def allowEditSizeT(self, checked):
        if checked:
            self.SizeT_SpinBox.setDisabled(False)
            if self.SizeT_metadata is not None:
                self.SizeT_SpinBox.setValue(self.SizeT_metadata)
        else:
            self.SizeT_SpinBox.setDisabled(True)
            self.SizeT_SpinBox.setValue(1)

    def warnEditingMetadata(self, Size, Size_metadata, which_dim):
        txt = html_utils.paragraph(f"""
            The <b>number of {which_dim} in the saved metadata is {Size_metadata}</b>,  
            but you are requesting to <b>change it to {Size}</b>.<br><br>
            Are you <b>sure you want to proceed</b>?
        """)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        _, noButton, yesButton = msg.warning(
            self,
            "WARNING: Edinting saved metadata",
            txt,
            buttonsTexts=("Cancel", "No", "Yes, edit the metadata"),
        )
        return msg.clickedButton == yesButton

    def ok_cb(self, checked=False):
        self.cancel = False
        self.SizeT = self.SizeT_SpinBox.value()
        self.SizeZ = self.SizeZ_SpinBox.value()

        if self.SizeT_metadata is not None:
            if self.SizeT != self.SizeT_metadata:
                proceed = self.warnEditingMetadata(
                    self.SizeT, self.SizeT_metadata, "frames"
                )
                if not proceed:
                    return

        if self.SizeZ_metadata is not None:
            if self.SizeZ != self.SizeZ_metadata:
                proceed = self.warnEditingMetadata(
                    self.SizeZ, self.SizeZ_metadata, "z-slices"
                )
                if not proceed:
                    return

        self.isSegm3D = self.isSegm3Dtoggle.isChecked()

        self.TimeIncrement = self.TimeIncrementSpinBox.value()
        self.PhysicalSizeX = self.PhysicalSizeXSpinBox.value()
        self.PhysicalSizeY = self.PhysicalSizeYSpinBox.value()
        self.PhysicalSizeZ = self.PhysicalSizeZSpinBox.value()
        self._additionalValues = {
            f"__{field['nameWidget'].text()}": field["valueWidget"].text()
            for field in self.additionalFieldsWidgets
        }
        proceed = self.checkShapeMismatchMetadata()
        if not proceed:
            return

        if self.posData is not None and self.sender() != self.okButton:
            exp_path = self.posData.exp_path
            pos_foldernames = myutils.get_pos_foldernames(exp_path)
            if self.sender() == self.selectButton:
                select_folder = load.select_exp_folder()
                select_folder.pos_foldernames = pos_foldernames
                select_folder.QtPrompt(
                    self, pos_foldernames, allow_cancel=False, toggleMulti=True
                )
                pos_foldernames = select_folder.selected_pos
            for pos in pos_foldernames:
                images_path = os.path.join(exp_path, pos, "Images")
                ls = myutils.listdir(images_path)
                search = [file for file in ls if file.find("metadata.csv") != -1]
                metadata_df = None
                if search:
                    fileName = search[0]
                    metadata_csv_path = os.path.join(images_path, fileName)
                    metadata_df = pd.read_csv(metadata_csv_path).set_index(
                        "Description"
                    )
                if metadata_df is not None:
                    metadata_df.at["TimeIncrement", "values"] = self.TimeIncrement
                    metadata_df.at["PhysicalSizeZ", "values"] = self.PhysicalSizeZ
                    metadata_df.at["PhysicalSizeY", "values"] = self.PhysicalSizeY
                    metadata_df.at["PhysicalSizeX", "values"] = self.PhysicalSizeX
                    metadata_df.to_csv(metadata_csv_path)

                search = [file for file in ls if file.find("acdc_output.csv") != -1]
                acdc_df = None
                if search:
                    fileName = search[0]
                    acdc_df_path = os.path.join(images_path, fileName)
                    acdc_df = pd.read_csv(acdc_df_path)
                    yx_pxl_to_um2 = self.PhysicalSizeY * self.PhysicalSizeX
                    vox_to_fl = self.PhysicalSizeY * (self.PhysicalSizeX**2)
                    if "cell_vol_fl" not in acdc_df.columns:
                        continue
                    acdc_df["cell_vol_fl"] = acdc_df["cell_vol_vox"] * vox_to_fl
                    acdc_df["cell_area_um2"] = acdc_df["cell_area_pxl"] * yx_pxl_to_um2
                    acdc_df["time_seconds"] = acdc_df["frame_i"] * self.TimeIncrement
                    try:
                        acdc_df.to_csv(acdc_df_path, index=False)
                    except PermissionError:
                        err_msg = html_utils.paragraph(
                            "The below file is open in another app "
                            "(Excel maybe?).<br><br>"
                            f"<code>{acdc_df_path}</code><br><br>"
                            'Close file and then press "Ok".'
                        )
                        msg = widgets.myMessageBox()
                        msg.critical(self, "Permission denied", err_msg)
                        acdc_df.to_csv(acdc_df_path, index=False)

        elif self.sender() == self.selectButton:
            pass

        self.close()

    def checkShapeMismatchMetadata(self):
        valid4D = True
        valid3D = True
        valid2D = True
        if self.imgDataShape is None:
            self.close()
        elif len(self.imgDataShape) == 4:
            T, Z, Y, X = self.imgDataShape
            valid4D = self.SizeT == T and self.SizeZ == Z
        elif len(self.imgDataShape) == 3:
            TorZ, Y, X = self.imgDataShape
            valid3D = self.SizeT == TorZ or self.SizeZ == TorZ
        elif len(self.imgDataShape) == 2:
            valid2D = self.SizeT == 1 and self.SizeZ == 1

        valid = all([valid4D, valid3D, valid2D])
        if valid:
            return True

        if not valid4D:
            txt = f"""
                You loaded <b>4D data</b>, hence the number of frames MUST be
                <b>{T}</b><br> and the number of z-slices MUST be <b>{Z}</b>.<br><br>
                What do you want to do?
            """
        if not valid3D:
            txt = f"""
                You loaded <b>3D data</b>, hence either the number of frames or 
                the number of z-slices is <b>{TorZ}</b>.<br><br>
                However, if the number of frames is greater than 1 then the<br>
                number of z-slices MUST be 1, and vice-versa.<br><br>
                What do you want to do?
            """

        if not valid2D:
            txt = f"""
                You loaded <b>2D data</b>, hence the number of frames MUST be <b>1</b>
                and the number of z-slices MUST be <b>1</b>.<br><br>
                What do you want to do?
            """

        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(txt)

        continueButton = widgets.okPushButton("Continue anyway")
        correctButton = widgets.editPushButton("Let me correct")

        msg.warning(
            self,
            "Shape-metadata mismatch",
            txt,
            buttonsTexts=(continueButton, correctButton),
        )
        if msg.cancel or msg.clickedButton == correctButton:
            return False

        return True

    def cancel_cb(self, event):
        self.cancel = True
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
        if hasattr(self, "loop"):
            self.loop.exit()


class QCropZtool(QBaseDialog):
    sigClose = Signal()
    sigZvalueChanged = Signal(str, int)
    sigReset = Signal()
    sigCrop = Signal(int, int)

    def __init__(
        self,
        SizeZ,
        cropButtonText="Apply crop",
        parent=None,
        addDoNotShowAgain=False,
        title="Select z-slices",
    ):
        super().__init__(parent)

        self.cancel = True

        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        self.SizeZ = SizeZ
        self.numDigits = len(str(self.SizeZ))

        self.setWindowTitle(title)

        layout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        self.lowerZscrollbar = widgets.ScrollBarWithNumericControl()
        self.lowerZscrollbar.setMaximum(SizeZ)
        self.lowerZscrollbar.setMinimum(1)
        self.lowerZscrollbar.setValue(1)

        self.upperZscrollbar = widgets.ScrollBarWithNumericControl()
        self.upperZscrollbar.setMaximum(SizeZ)
        self.upperZscrollbar.setValue(SizeZ)

        cancelButton = widgets.cancelPushButton("Cancel")
        cropButton = widgets.okPushButton(cropButtonText)
        buttonsLayout.addWidget(cropButton)
        buttonsLayout.addWidget(cancelButton)

        row = 0
        layout.addWidget(QLabel("Lower z-slice "), row, 0, alignment=Qt.AlignRight)
        layout.addWidget(self.lowerZscrollbar, row, 1)

        row += 1
        layout.setRowStretch(row, 5)

        row += 1
        layout.addWidget(QLabel("Upper z-slice "), row, 0, alignment=Qt.AlignRight)
        layout.addWidget(self.upperZscrollbar, row, 1)

        row += 1
        if addDoNotShowAgain:
            self.doNotShowAgainCheckbox = QCheckBox("Do not ask again")
            layout.addWidget(
                self.doNotShowAgainCheckbox, row, 1, alignment=Qt.AlignLeft
            )
            row += 1

        layout.addLayout(buttonsLayout, row, 1, alignment=Qt.AlignRight)

        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 10)

        self.setLayout(layout)

        # resetButton.clicked.connect(self.emitReset)
        cropButton.clicked.connect(self.emitCrop)
        cancelButton.clicked.connect(self.close)
        self.lowerZscrollbar.sigValueChanged.connect(self.ZvalueChanged)
        self.upperZscrollbar.sigValueChanged.connect(self.ZvalueChanged)

    def emitReset(self):
        self.sigReset.emit()

    def emitCrop(self):
        self.cancel = False
        low_z = self.lowerZscrollbar.value() - 1
        high_z = self.upperZscrollbar.value() - 1
        self.sigCrop.emit(low_z, high_z)
        self.close()

    def updateScrollbars(self, lower_z, upper_z):
        self.lowerZscrollbar.setValue(lower_z + 1)
        self.upperZscrollbar.setValue(upper_z + 1)

    def ZvalueChanged(self, value):
        which = "lower" if self.sender() == self.lowerZscrollbar else "upper"
        if which == "lower" and value > self.upperZscrollbar.value() - 1:
            self.lowerZscrollbar.setValue(self.upperZscrollbar.value() - 1)
            return
        if which == "upper" and value < self.lowerZscrollbar.value() + 1:
            self.upperZscrollbar.setValue(self.lowerZscrollbar.value() + 1)
            return

        z_slice_n = value - 1
        self.sigZvalueChanged.emit(which, z_slice_n)

    def showEvent(self, event):
        self.resize(int(self.width() * 1.5), self.height())

    def closeEvent(self, event):
        super().closeEvent(event)
        self.sigClose.emit()


class TreeSelectorDialog(QBaseDialog):
    sigItemDoubleClicked = Signal(object)

    def __init__(
        self,
        title="Tree selector",
        infoTxt="",
        parent=None,
        multiSelection=True,
        widthFactor=None,
        heightFactor=None,
        expandOnDoubleClick=False,
        isTopLevelSelectable=True,
        allItemsExpanded=True,
        allowNoSelection=True,
    ):
        super().__init__(parent)

        self.setWindowTitle(title)

        self.cancel = True
        self.widthFactor = widthFactor
        self.heightFactor = heightFactor
        self.allItemsExpanded = allItemsExpanded
        self.mainLayout = QVBoxLayout()
        self._isTopLevelSelectable = isTopLevelSelectable
        self.allowNoSelection = allowNoSelection

        if infoTxt:
            self.mainLayout.addWidget(QLabel(html_utils.paragraph(infoTxt)))

        self.treeWidget = widgets.TreeWidget(multiSelection=multiSelection)
        self.treeWidget.setExpandsOnDoubleClick(expandOnDoubleClick)
        self.treeWidget.setHeaderHidden(True)
        self.mainLayout.addWidget(self.treeWidget)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        self.mainLayout.addSpacing(20)
        self.mainLayout.addLayout(buttonsLayout)

        self.buttonsLayout = buttonsLayout

        self.setLayout(self.mainLayout)

        self.treeWidget.itemClicked.connect(self.onItemClicked)
        self.treeWidget.itemDoubleClicked.connect(self.onItemDoubleClicked)

    def onItemDoubleClicked(self, item):
        self.sigItemDoubleClicked.emit(item)

    def onItemClicked(self, item):
        if self._isTopLevelSelectable:
            return
        if item.parent() is None:
            item.setSelected(False)

    def addTree(self, tree: dict):
        for topLevel, children in tree.items():
            topLevelItem = widgets.TreeWidgetItem(self.treeWidget)
            topLevelItem.setText(0, topLevel)
            self.treeWidget.addTopLevelItem(topLevelItem)
            childrenItems = [widgets.TreeWidgetItem([c]) for c in children]
            topLevelItem.addChildren(childrenItems)
            if not self.allItemsExpanded:
                continue
            topLevelItem.setExpanded(True)

    def resizeVertical(self):
        if not self.isVisible():
            self.show()

        currentTreeWidgetHeight = self.treeWidget.height()
        treeWidgetHeight = 0
        for i in range(self.treeWidget.topLevelItemCount()):
            topLevelItem = self.treeWidget.topLevelItem(i)
            rect = self.treeWidget.visualItemRect(topLevelItem)
            treeWidgetHeight += rect.height()
            for j in range(topLevelItem.childCount()):
                childItem = topLevelItem.child(j)
                rect = self.treeWidget.visualItemRect(childItem)
                treeWidgetHeight += rect.height()

        deltaHeight = treeWidgetHeight - currentTreeWidgetHeight + 10
        self.resize(self.width(), self.height() + deltaHeight)
        self.move(self.x(), 20)

    def setCurrentItem(self, itemText: dict):
        if not itemText:
            return
        for i in range(self.treeWidget.topLevelItemCount()):
            topLevelItem = self.treeWidget.topLevelItem(i)
            topLevelName = topLevelItem.text(0)
            childText = itemText.get(topLevelName)
            if childText is None:
                continue
            for j in range(topLevelItem.childCount()):
                childItem = topLevelItem.child(j)
                childItemText = childItem.text(0)
                if childItemText == childText:
                    childItem.setSelected(True)
                    topLevelItem.setExpanded(True)
                    self.treeWidget.scrollToItem(topLevelItem)
                    break

    def selectedItems(self):
        self._selectedItems = {}
        for i in range(self.treeWidget.topLevelItemCount()):
            topLevelItem = self.treeWidget.topLevelItem(i)
            topLevelName = topLevelItem.text(0)
            for j in range(topLevelItem.childCount()):
                childItem = topLevelItem.child(j)
                if not childItem.isSelected():
                    continue
                if topLevelName not in self._selectedItems:
                    self._selectedItems[topLevelName] = [childItem.text(0)]
                else:
                    self._selectedItems[topLevelName].append(childItem.text(0))
        return self._selectedItems

    def warnSelectionIsEmpty(self):
        txt = html_utils.paragraph("""
            You did not select anything :(.<br><br>
            Please press <code>Cancel</code> to exit without selecting items. 
            Thanks! 
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Selection is empty", txt)

    def ok_cb(self):
        if not self.allowNoSelection and not self.selectedItems():
            self.warnSelectionIsEmpty()
            return
        self.cancel = False
        self.close()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.widthFactor is not None:
            self.resize(int(self.width() * self.widthFactor), self.height())
        if self.heightFactor is not None:
            self.resize(self.width(), int(self.height() * self.heightFactor))


class TreesSelectorDialog(QBaseDialog):
    def __init__(
        self, trees, groupsDescr=None, title="Trees selector", infoTxt="", parent=None
    ):
        super().__init__(parent)

        self.setWindowTitle(title)

        self.cancel = True
        self.mainLayout = QVBoxLayout()

        if infoTxt:
            self.mainLayout.addWidget(QLabel(html_utils.paragraph(infoTxt)))

        self.treeWidgets = {}
        self.setLayout(self.mainLayout)

        createdGroupLayouts = {}
        for treeName, tree in trees.items():
            if groupsDescr is None:
                groupName = ""
            else:
                groupName = groupsDescr.get(treeName, "Group info missing")
            groupLayout = createdGroupLayouts.get(groupName, None)
            if groupLayout is None:
                self.mainLayout.addWidget(QLabel(html_utils.paragraph(groupName)))
                groupBox = QGroupBox()
                self.mainLayout.addWidget(groupBox)
                groupLayout = QVBoxLayout()
                groupBox.setLayout(groupLayout)
                createdGroupLayouts[groupName] = groupLayout
            else:
                groupLayout.addSpacing(10)
            groupLayout.addWidget(QLabel(html_utils.paragraph(treeName)))
            treeWidget = widgets.TreeWidget(multiSelection=True)
            treeWidget.setHeaderHidden(True)
            for topLevel, children in tree.items():
                topLevelItem = widgets.TreeWidgetItem(treeWidget)
                topLevelItem.setText(0, topLevel)
                treeWidget.addTopLevelItem(topLevelItem)
                childrenItems = [widgets.TreeWidgetItem([c]) for c in children]
                topLevelItem.addChildren(childrenItems)
                topLevelItem.setExpanded(True)
            self.treeWidgets[treeName] = treeWidget
            groupLayout.addWidget(treeWidget)
            self.mainLayout.addSpacing(20)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        self.mainLayout.addSpacing(10)
        self.mainLayout.addLayout(buttonsLayout)

    def ok_cb(self):
        self.cancel = False
        self.selectedItems = {}
        for treeName, treeWidget in self.treeWidgets.items():
            for i in range(treeWidget.topLevelItemCount()):
                topLevelItem = treeWidget.topLevelItem(i)
                for j in range(topLevelItem.childCount()):
                    childItem = topLevelItem.child(j)
                    if not childItem.isSelected():
                        continue
                    if treeName not in self.selectedItems:
                        self.selectedItems[treeName] = [childItem.text(0)]
                    else:
                        self.selectedItems[treeName].append(childItem.text(0))
        self.close()


class MultiListSelector(QBaseDialog):
    def __init__(
        self,
        lists: dict,
        groupsDescr: dict = None,
        title="Lists selector",
        infoTxt="",
        parent=None,
    ):
        super().__init__(parent)

        self.setWindowTitle(title)

        self.cancel = True
        mainLayout = QVBoxLayout()

        if infoTxt:
            mainLayout.addWidget(QLabel(html_utils.paragraph(infoTxt)))

        self.listWidgets = {}
        createdGroupLayouts = {}
        for listName, listItems in lists.items():
            if groupsDescr is None:
                groupName = ""
            else:
                groupName = groupsDescr.get(listName, "Group info missing")
            groupLayout = createdGroupLayouts.get(listName, None)
            if groupLayout is None:
                mainLayout.addWidget(QLabel(html_utils.paragraph(groupName)))
                groupBox = QGroupBox()
                mainLayout.addWidget(groupBox)
                groupLayout = QVBoxLayout()
                groupBox.setLayout(groupLayout)
                createdGroupLayouts[groupName] = groupLayout
            else:
                groupLayout.addSpacing(10)
            groupLayout.addWidget(QLabel(html_utils.paragraph(listName)))
            listWidget = widgets.listWidget()
            listWidget.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection
            )
            listWidget.addItems(listItems)
            groupLayout.addWidget(listWidget)
            mainLayout.addSpacing(20)
            self.listWidgets[listName] = listWidget

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def ok_cb(self):
        self.cancel = False
        self.selectedItems = {}
        for listName, listWidget in self.listWidgets.items():
            if not listWidget.selectedItems():
                continue
            self.selectedItems[listName] = [
                item.text() for item in listWidget.selectedItems()
            ]
        self.close()


class selectPositionsMultiExp(QBaseDialog):
    def __init__(self, expPaths: dict, infoPaths: dict = None, parent=None):
        super().__init__(parent=parent)

        self.expPaths = expPaths
        self.cancel = True

        mainLayout = QVBoxLayout()

        self.setWindowTitle("Select Positions to process")

        infoTxt = html_utils.paragraph(
            "Select one or more Positions to process<br><br>"
            "<code>Click</code> on experiment path <i>to select all positions</i><br>"
            "<code>Ctrl+Click</code> <i>to select multiple items</i><br>"
            "<code>Shift+Click</code> <i>to select a range of items</i><br>",
            center=True,
        )
        infoLabel = QLabel(infoTxt)

        self.treeWidget = QTreeWidget()
        self.treeWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.treeWidget.setHeaderHidden(True)
        self.treeWidget.setFont(font)
        for exp_path, positions in expPaths.items():
            pathLevels = exp_path.split(os.sep)
            posFoldersInfo = None
            if infoPaths is not None:
                posFoldersInfo = infoPaths.get(exp_path)
            if len(pathLevels) > 4:
                itemText = os.path.join(*pathLevels[-4:])
                itemText = f"...{itemText}"
            else:
                itemText = exp_path
            exp_path_item = QTreeWidgetItem([itemText])
            exp_path_item.setToolTip(0, exp_path)
            exp_path_item.full_path = exp_path
            self.treeWidget.addTopLevelItem(exp_path_item)
            postions_items = []
            for pos in positions:
                if posFoldersInfo is not None:
                    status = posFoldersInfo.get(pos, "")
                else:
                    status = ""
                pos_item_text = f"{pos}{status}"
                pos_item = QTreeWidgetItem(exp_path_item, [pos_item_text])
                pos_item.posFoldername = pos
                postions_items.append(pos_item)
            exp_path_item.addChildren(postions_items)
            exp_path_item.setExpanded(True)

        self.treeWidget.itemClicked.connect(self.selectAllChildren)

        buttonsLayout = QHBoxLayout()
        cancelButton = widgets.cancelPushButton("Cancel")
        okButton = widgets.okPushButton(" Ok ")

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        mainLayout.addWidget(self.treeWidget)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        self.setStyleSheet(TREEWIDGET_STYLESHEET)

    def selectAllChildren(self, item, col):
        if item.parent() is not None:
            return

        for i in range(item.childCount()):
            item.child(i).setSelected(True)

    def ok_cb(self):
        if not self.treeWidget.selectedItems():
            msg = widgets.myMessageBox(wrapText=False)
            txt = "You did not select any experiment/Position folder!"
            msg.warning(self, "Empty selection!", html_utils.paragraph(txt))
            return

        self.cancel = False
        self.selectedPaths = {}
        for item in self.treeWidget.selectedItems():
            if item.parent() is None:
                continue
            parent = item.parent()
            exp_path = parent.full_path
            pos_folder = item.posFoldername
            if exp_path not in self.selectedPaths:
                self.selectedPaths[exp_path] = []
            self.selectedPaths[exp_path].append(pos_folder)

        self.close()

    def showEvent(self, event):
        self.resize(int(self.width() * 2), self.height())


class QDialogZsliceAbsent(QDialog):
    def __init__(self, filename, SizeZ, filenamesWithInfo, parent=None):
        self.runDataPrep = False
        self.useMiddleSlice = False
        self.useSameAsCh = False

        self.cancel = True

        super().__init__(parent)
        self.setWindowTitle("Reference z-slice info absent")

        mainLayout = QVBoxLayout()
        buttonsLayout = QGridLayout()

        txt = html_utils.paragraph(
            f"""
            You loaded the fluorescent file called<br><br>{filename}<br><br>
            however you <b>never selected which z-slice</b><br> you want to use
            when calculating metrics<br> (e.g., mean, median, amount...etc.)<br><br>
            Choose one of following options:
        """,
            center=True,
        )
        infoLabel = QLabel(txt)
        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        runDataPrepButton = QPushButton(
            "   Visualize the data now and select a z-slice    "
        )
        buttonsLayout.addWidget(runDataPrepButton, 0, 1, 1, 2)
        runDataPrepButton.clicked.connect(self.runDataPrep_cb)

        useMiddleSliceButton = QPushButton(
            f"  Use the middle z-slice ({int(SizeZ / 2) + 1})  "
        )
        buttonsLayout.addWidget(useMiddleSliceButton, 1, 1, 1, 2)
        useMiddleSliceButton.clicked.connect(self.useMiddleSlice_cb)

        useSameAsChButton = QPushButton("  Use the same z-slice used for the channel: ")
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

        cancelButtonLayout = QHBoxLayout()
        cancelButton = widgets.cancelPushButton("Cancel")
        cancelButtonLayout.addStretch(1)
        cancelButtonLayout.addWidget(cancelButton)
        cancelButtonLayout.addStretch(1)
        cancelButtonLayout.setStretch(1, 1)
        cancelButton.clicked.connect(self.close)

        mainLayout.addLayout(buttonsLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(cancelButtonLayout)
        mainLayout.addStretch(1)

        self.setLayout(mainLayout)

        font = QFont()
        font.setPixelSize(12)
        self.setFont(font)

        # self.setModal(True)

    def ok_cb(self, checked=True):
        self.cancel = False
        self.close()

    def useSameAsCh_cb(self, checked):
        self.useSameAsCh = True
        self.selectedChannel = self.chNameComboBox.currentText()
        self.ok_cb()

    def useMiddleSlice_cb(self, checked):
        self.useMiddleSlice = True
        self.ok_cb()

    def runDataPrep_cb(self, checked):
        self.runDataPrep = True
        self.ok_cb()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()


class SetColumnNamesDialog(QBaseDialog):
    def __init__(self, columnNames, categories, optionalCategories=None, parent=None):
        super().__init__(parent)

        if not optionalCategories:
            optionalCategories = None

        self.cancel = True

        mainLayout = QVBoxLayout()

        mainLayout.addWidget(
            QLabel(
                html_utils.paragraph("Assign a column to the following categories:<br>")
            )
        )

        self.categoriesWidgets = {}
        formLayout = QFormLayout()
        for row, category in enumerate(categories):
            combobox = widgets.ComboBox()
            combobox.addItems(columnNames)
            if optionalCategories is not None:
                text = f"* {category}"
            else:
                text = category
            formLayout.addRow(text, combobox)
            self.categoriesWidgets[category] = combobox

        if optionalCategories is not None:
            optionalItems = ["None", *columnNames]
            for row, category in enumerate(optionalCategories):
                combobox = widgets.ComboBox()
                combobox.addItems(optionalItems)
                formLayout.addRow(category, combobox)
                self.categoriesWidgets[category] = combobox

        mainLayout.addLayout(formLayout)
        if optionalCategories is not None:
            mainLayout.addSpacing(10)
            mainLayout.addWidget(
                QLabel(html_utils.paragraph("* mandatory", font_size="11px"))
            )

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        self.setFont(font)

    def _warnNonUniqueCategories(self, category_1, category_2):
        txt = html_utils.paragraph(f"""
            The following categories have the same column assigned to it.<br><br>
            Columns assigned to categories <b>must be unique</b>.<br><br>
            Categories with the same column:
            {html_utils.to_list((category_1, category_2))}
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Non-unique columns", txt)

    def _checkUniqueNames(self):
        self.textToCategoryMapper = {}
        for category, combobox in self.categoriesWidgets.items():
            if combobox.text() == "None":
                continue

            if combobox.text() not in self.textToCategoryMapper:
                self.textToCategoryMapper[combobox.text()] = category
                continue

            sameCategory = self.textToCategoryMapper[combobox.text()]
            self._warnNonUniqueCategories(category, sameCategory)
            return False

        return True

    def ok_cb(self):
        proceed = self._checkUniqueNames()
        if not proceed:
            return

        self.selectedColumns = {
            category: combobox.text()
            for category, combobox in self.categoriesWidgets.items()
        }
        self.cancel = False
        self.close()


class QCropTrangeTool(QBaseDialog):
    sigClose = Signal()
    sigTvalueChanged = Signal(int)
    sigReset = Signal()
    sigCrop = Signal(int, int)

    def __init__(
        self,
        SizeT,
        cropButtonText="Apply crop",
        parent=None,
        addDoNotShowAgain=False,
        title="Select frames range",
    ):
        super().__init__(parent)

        self.cancel = True

        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        self.SizeT = SizeT
        self.numDigits = len(str(self.SizeT))

        self.setWindowTitle(title)

        layout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        self.startFrameScrollbar = widgets.sliderWithSpinBox(
            spinbox_loc="left", maximum_on_label=SizeT
        )
        self.startFrameScrollbar.setMaximum(SizeT, including_spinbox=True)
        self.startFrameScrollbar.setMinimum(1, including_spinbox=True)

        self.endFrameScrollbar = widgets.sliderWithSpinBox(
            spinbox_loc="left", maximum_on_label=SizeT
        )
        self.endFrameScrollbar.setMaximum(SizeT, including_spinbox=True)
        self.endFrameScrollbar.setMinimum(1, including_spinbox=True)
        self.endFrameScrollbar.setValue(SizeT)

        cancelButton = widgets.cancelPushButton("Cancel")
        cropButton = widgets.okPushButton(cropButtonText)
        buttonsLayout.addWidget(cropButton)
        buttonsLayout.addWidget(cancelButton)

        row = 0
        layout.addWidget(QLabel("Start frame n.  "), row, 0, alignment=Qt.AlignRight)
        layout.addWidget(self.startFrameScrollbar, row, 2)

        row += 1
        layout.setRowStretch(row, 5)
        layout.addItem(QSpacerItem(10, 10), row, 0)

        row += 1
        layout.addWidget(QLabel("Stop frame n. "), row, 0, alignment=Qt.AlignRight)
        layout.addWidget(self.endFrameScrollbar, row, 2)

        row += 1
        if addDoNotShowAgain:
            self.doNotShowAgainCheckbox = QCheckBox("Do not ask again")
            layout.addWidget(
                self.doNotShowAgainCheckbox, row, 2, alignment=Qt.AlignLeft
            )
            row += 1

        layout.addItem(QSpacerItem(10, 20), row, 0)
        layout.addLayout(buttonsLayout, row + 1, 2, alignment=Qt.AlignRight)

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
        self.resize(int(self.width() * 2.0), self.height())

    def closeEvent(self, event):
        super().closeEvent(event)
        self.sigClose.emit()


class SelectFoldersToAnalyse(QBaseDialog):
    def __init__(
        self,
        parent=None,
        preSelectedPaths=None,
        onlyExpPaths=False,
        scanFolderTree=True,
        instructionsText="Select experiment folders to analyse",
        askSelectPosFolders=False,
    ):
        super().__init__(parent)

        self.cancel = True
        self.onlyExpPaths = onlyExpPaths
        self.setWindowTitle("Select experiments to analyse")
        self.scanTree = scanFolderTree
        self.askSelectPosFolders = askSelectPosFolders

        mainLayout = QVBoxLayout()

        instructionsText = html_utils.paragraph(
            f"{instructionsText}<br><br>"
            "Drag and drop folders or click on <code>Add folder</code> button to "
            "<b>add</b> as many <b>folders</b> "
            "as needed.<br>",
            font_size="14px",
        )
        instructionsLabel = QLabel(instructionsText)
        instructionsLabel.setAlignment(Qt.AlignCenter)

        infoText = html_utils.paragraph(
            "A <b>valid folder</b> is either a <b>Position</b> folder, "
            "or an <b>experiment folder</b> (containing Position_n folders),<br>"
            "or any folder that contains <b>multiple experiment folders</b>.<br><br>"
            "In the last case, Cell-ACDC will automatically scan the entire tree of "
            "sub-directories<br>"
            "and will add all experiments having the right folder structure.<br>",
            font_size="12px",
        )
        infoLabel = QLabel(infoText)
        infoLabel.setAlignment(Qt.AlignCenter)

        self.listWidget = widgets.listWidget()
        self.listWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        if preSelectedPaths is not None:
            self.listWidget.addItems(preSelectedPaths)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        delButton = widgets.delPushButton("Remove selected path(s)")
        browseButton = widgets.browseFileButton(
            "Add folder...", openFolder=True, start_dir=myutils.getMostRecentPath()
        )

        buttonsLayout.insertWidget(3, delButton)
        buttonsLayout.insertWidget(4, browseButton)

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        browseButton.sigPathSelected.connect(self.addFolderPath)
        delButton.clicked.connect(self.removePaths)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(instructionsLabel)
        mainLayout.addWidget(infoLabel)
        mainLayout.addWidget(self.listWidget)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch(1)

        self.setLayout(mainLayout)

        self.setAcceptDrops(True)

        self.setFont(font)

    def dragEnterEvent(self, event):
        event.acceptProposedAction()

    def dropEvent(self, event):
        event.setDropAction(Qt.CopyAction)
        for url in event.mimeData().urls():
            dropped_path = url.toLocalFile()
            if os.path.isfile(dropped_path):
                dropped_path = os.path.dirname(dropped_path)

            QTimer.singleShot(50, partial(self.addFolderPath, dropped_path))

    def pathsList(self):
        return [
            self.listWidget.item(i).text().replace("\\", "/")
            for i in range(self.listWidget.count())
        ]

    def expFolderToPosFoldernamesMapper(self):
        expPathsPosFoldernamesMapper = defaultdict(set)
        for selectedPath in self.pathsList():
            pos_foldernames = myutils.get_pos_foldernames(
                selectedPath, check_if_is_sub_folder=True
            )
            if not pos_foldernames:
                images_path = myutils.get_images_folderpath(selectedPath)
                expPathsPosFoldernamesMapper[selectedPath].add("")
            else:
                expPath = load.get_exp_path(selectedPath)
                expPathsPosFoldernamesMapper[expPath].update(pos_foldernames)

        expPathsPosFoldernamesMapper = {
            expPath: natsorted(pos_foldernames)
            for expPath, pos_foldernames in expPathsPosFoldernamesMapper.items()
        }
        return expPathsPosFoldernamesMapper

    def ok_cb(self):
        self.cancel = False
        self.paths = self.pathsList()
        self.selectedExpFolderToPosFoldernamesMapper = (
            self.expFolderToPosFoldernamesMapper()
        )
        self.close()

    def warnNoValidPathsFound(self, selected_path):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            The selected path (see below) <b>does not contain any valid folder.</b><br><br>
            Please, make sure to select a Position folder, the Images folder 
            inside a Position folder, or any folder containing a Position folder 
            as a sub-directory.<br><br>
            Thank you for your patience!<br><br>
            Selected path:
        """)
        msg.warning(
            self,
            "Training workflow generated",
            txt,
            commands=(f"{selected_path}",),
            path_to_browse=selected_path,
        )

    def warnNoValidExpPaths(self, selected_path):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            The selected folder does 
            <b>not contain any valid experiment folders</b>.
        """)
        command = selected_path.replace("\\", os.sep)
        command = selected_path.replace("/", os.sep)
        msg.warning(
            self,
            "No valid folders found",
            txt,
            commands=(command,),
            path_to_browse=selected_path,
        )

    def parse_select_from_exp_paths(self, exp_paths: dict[os.PathLike, Iterable[str]]):
        if not self.askSelectPosFolders:
            return list(exp_paths.keys())

        paths = []
        for exp_path, pos_foldernames in exp_paths.items():
            if len(pos_foldernames) == 1:
                paths.append(exp_path)
                continue

            informativeText = html_utils.paragraph(
                "The following experiment folder<br><br>"
                f"<code>{exp_path}</code><br><br>"
                "contains multiple Position folders.<br><br>"
                "Please, select which Position folder(s) you want to analyse:<br>"
            )
            select_folder = load.select_exp_folder()
            values = select_folder.get_values_dataprep(exp_path)
            select_folder.QtPrompt(
                self,
                values,
                toggleMulti=True,
                informativeText=informativeText,
                selectedValues=values,
            )
            if select_folder.cancel:
                return

            for pos in select_folder.selected_pos:
                paths.append(os.path.join(exp_path, pos))

        return paths

    def addFolderPath(self, selected_path):
        myutils.addToRecentPaths(selected_path)

        folder_type = myutils.determine_folder_type(selected_path)
        is_pos_folder, is_images_folder, folder_path = folder_type
        if is_pos_folder:
            paths = [selected_path]
        elif is_images_folder:
            paths = [os.path.dirname(selected_path)]
        elif self.scanTree:
            print(f'Scanning selected folder "{selected_path}"...')
            exp_paths = path.get_posfolderpaths_walk(selected_path)
            if not exp_paths:
                self.warnNoValidExpPaths(selected_path)
                return

            paths = self.parse_select_from_exp_paths(exp_paths)
            if paths is None:
                return
        else:
            paths = [selected_path]

        if not paths:
            self.warnNoValidPathsFound(selected_path)

        for selectedPath in paths:
            if self.onlyExpPaths:
                selectedPath = load.get_exp_path(selectedPath)

            selectedPath = selectedPath.replace("\\", "/")
            if selectedPath in self.pathsList():
                print(
                    f"[WARNING]: The following path was already selected: "
                    f'"{selectedPath}"'
                )
                return

            self.listWidget.addItem(selectedPath)

    def removePaths(self):
        for item in self.listWidget.selectedItems():
            row = self.listWidget.row(item)
            self.listWidget.takeItem(row)


class OverlayLabelsAppearanceDialog(QBaseDialog):
    sigValuesChanged = Signal(object)

    def __init__(self, scatterPlotItem: pg.ScatterPlotItem = None, parent=None):
        super().__init__(parent)

        self.cancel = True

        self.setWindowTitle("Overlay contours appearance properties")

        mainLayout = QVBoxLayout()

        formLayout = widgets.FormLayout()

        row = -1

        row += 1
        self.colorButton = widgets.myColorButton(color=(255, 0, 0))
        self.colorButton.clicked.disconnect()
        self.colorButton.clicked.connect(self.selectColor)
        self.colorButton.setCursor(Qt.PointingHandCursor)
        self.colorWidget = widgets.formWidget(
            self.colorButton,
            addInfoButton=False,
            stretchWidget=False,
            labelTextLeft="Symbol color: ",
            parent=self,
            widgetAlignment="left",
        )
        if scatterPlotItem is not None:
            pen = scatterPlotItem.opts["pen"]
            color = pen.color()
            self.colorButton.setColor(color)
        formLayout.addFormWidget(self.colorWidget, row=row)

        row += 1
        self.penWidthSpinBox = widgets.SpinBox()
        self.penWidthSpinBox.setMinimum(0)
        self.penWidthSpinBox.setValue(2)

        self.penWidthWidget = widgets.formWidget(
            self.penWidthSpinBox,
            addInfoButton=False,
            stretchWidget=False,
            labelTextLeft="Symbol weight: ",
            parent=self,
            widgetAlignment="left",
        )
        if scatterPlotItem is not None:
            pen = scatterPlotItem.opts["pen"]
            width = pen.width()
            self.penWidthSpinBox.setValue(width)
        formLayout.addFormWidget(self.penWidthWidget, row=row)

        row += 1
        self.opacitySlider = widgets.sliderWithSpinBox(isFloat=True, normalize=True)
        self.opacitySlider.setMinimum(0)
        self.opacitySlider.setMaximum(100)
        self.opacitySlider.setValue(0.8)

        self.opacityWidget = widgets.formWidget(
            self.opacitySlider,
            addInfoButton=False,
            stretchWidget=True,
            labelTextLeft="Symbol opacity: ",
            parent=self,
        )
        if scatterPlotItem is not None:
            brush = scatterPlotItem.opts["brush"]
            alpha = brush.color().alpha()
            opacity = alpha / 255
            self.opacitySlider.setValue(opacity)
        formLayout.addFormWidget(self.opacityWidget, row=row)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addLayout(formLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def selectColor(self):
        color = self.colorButton.color()
        self.colorButton.origColor = color
        self.colorButton.colorDialog.setCurrentColor(color)
        self.colorButton.colorDialog.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.colorButton.colorDialog.open()
        w = self.width()
        left = self.pos().x()
        colorDialogTop = self.colorButton.colorDialog.pos().y()
        self.colorButton.colorDialog.move(w + left + 10, colorDialogTop)

    def getBrush(self):
        r, g, b, _ = self.colorButton.color().getRgb()
        alpha = round(self.opacitySlider.value() * 255)
        brushColor = (r, g, b, alpha)
        brush = pg.mkBrush(brushColor)
        return brush

    def getPen(self):
        color = self.colorButton.color()
        penWidth = self.penWidthSpinBox.value()
        if penWidth == 0:
            return

        pen = pg.mkPen(color, width=penWidth)
        return pen

    def ok_cb(self):
        self.cancel = False
        self.properties = {"brush": self.getBrush(), "pen": self.getPen()}
        self.close()


class AutoSaveIntervalDialog(QBaseDialog):
    sigValueChanged = Signal(float, str)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.cancel = True

        self.setWindowTitle("Change autosave interval")

        mainLayout = QVBoxLayout()

        self.autoSaveIntervalWidget = widgets.AutoSaveIntervalWidget(parent=self)

        mainLayout.addWidget(QLabel("Autosave interval:"))
        mainLayout.addWidget(self.autoSaveIntervalWidget)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def setValues(self, autoSaveIntevalValue, autoSaveIntervalUnit):
        self.autoSaveIntervalWidget.spinbox.setValue(autoSaveIntevalValue)
        self.autoSaveIntervalWidget.unitCombobox.setCurrentText(autoSaveIntervalUnit)

    def sizeHint(self):
        defaultWidth = super().sizeHint().width()
        defaultHeight = super().sizeHint().height()
        return QSize(defaultWidth * 2, defaultHeight)

    def ok_cb(self):
        self.cancel = False
        self.sigValueChanged.emit(
            self.autoSaveIntervalWidget.spinbox.value(),
            self.autoSaveIntervalWidget.unitCombobox.currentText(),
        )
        self.close()

# Sibling imports (deferred to avoid import cycles)
from .general import (
    imageViewer,
)

