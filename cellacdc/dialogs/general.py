"""Cell-ACDC dialog windows: general."""

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

class customAnnotationDialog(QDialog):
    sigDeleteSelecAnnot = Signal(object)

    def __init__(self, savedCustomAnnot, parent=None, state=None):
        self.cancel = True
        self.loop = None
        self.clickedButton = None
        self.savedCustomAnnot = savedCustomAnnot

        self.internalNames = measurements.get_all_acdc_df_colnames(include_custom=False)

        super().__init__(parent)

        self.setWindowTitle("Custom annotation")
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        layout = widgets.FormLayout()

        row = 0
        typeCombobox = QComboBox()
        typeCombobox.addItems(
            ["Single time-point", "Multiple time-points", "Multiple values class"]
        )
        if state is not None:
            typeCombobox.setCurrentText(state["type"])
        self.typeCombobox = typeCombobox
        body_txt = """
        <b>Single time-point</b> annotation: use this to annotate
        an event that happens on a <b>single frame in time</b>
        (e.g. cell division).
        <br><br>
        <b>Multiple time-points</b> annotation: use this to annotate
        an event that has a <b>duration</b>, i.e., a start frame and a stop
        frame (e.g. cell cycle phase).<br><br>
        <b>Multiple values class</b> annotation: use this to annotate a class
        that has <b>multiple values</b>. An example could be a cell cycle stage
        that can have different values, such as 2-cells division
        or 4-cells division.
        """
        typeInfoTxt = f"{html_utils.paragraph(body_txt)}"
        self.typeWidget = widgets.formWidget(
            typeCombobox,
            addInfoButton=True,
            labelTextLeft="Type: ",
            parent=self,
            infoTxt=typeInfoTxt,
        )
        layout.addFormWidget(self.typeWidget, row=row)
        typeCombobox.currentTextChanged.connect(self.warnType)

        row += 1
        nameInfoTxt = """
        <b>Name of the column</b> that will be saved in the <code>acdc_output.csv</code>
        file.<br><br>
        Valid charachters are letters and numbers separate by underscore
        or dash only.<br><br>
        Additionally, some names are <b>reserved</b> because they are used
        by Cell-ACDC for standard measurements.<br><br>
        Internally reserved names:
        """
        self.nameInfoTxt = f"{html_utils.paragraph(nameInfoTxt)}"
        self.nameWidget = widgets.formWidget(
            widgets.alphaNumericLineEdit(),
            addInfoButton=True,
            labelTextLeft="Name: ",
            parent=self,
            infoTxt=self.nameInfoTxt,
        )
        self.nameWidget.infoButton.disconnect()
        self.nameWidget.infoButton.clicked.connect(self.showNameInfo)
        if state is not None:
            self.nameWidget.widget.setText(state["name"])
        self.nameWidget.widget.textChanged.connect(self.checkName)
        layout.addFormWidget(self.nameWidget, row=row)

        row += 1
        self.nameInfoLabel = QLabel()
        layout.addWidget(self.nameInfoLabel, row, 0, 1, 2, alignment=Qt.AlignCenter)

        row += 1
        spacing = QSpacerItem(10, 10)
        layout.addItem(spacing, row, 0)

        row += 1
        symbolInfoTxt = """
        <b>Symbol</b> that will be drawn on the annotated cell at
        the requested time frame.
        """
        symbolInfoTxt = f"{html_utils.paragraph(symbolInfoTxt)}"
        self.symbolWidget = widgets.formWidget(
            widgets.pgScatterSymbolsCombobox(),
            addInfoButton=True,
            labelTextLeft="Symbol: ",
            parent=self,
            infoTxt=symbolInfoTxt,
        )
        if state is not None:
            self.symbolWidget.widget.setCurrentText(state["symbol"])
        layout.addFormWidget(self.symbolWidget, row=row)

        row += 1
        shortcutInfoTxt = """
        <b>Shortcut</b> that you can use to <b>activate/deactivate</b> annotation
        of this event.<br><br> Leave empty if you don't need a shortcut.
        """
        shortcutInfoTxt = f"{html_utils.paragraph(shortcutInfoTxt)}"
        self.shortcutWidget = widgets.formWidget(
            widgets.ShortcutLineEdit(),
            addInfoButton=True,
            labelTextLeft="Shortcut: ",
            parent=self,
            infoTxt=shortcutInfoTxt,
        )
        if state is not None:
            self.shortcutWidget.widget.setText(state["shortcut"])
        layout.addFormWidget(self.shortcutWidget, row=row)

        row += 1
        descInfoTxt = """
        <b>Description</b> will be used as the <b>tool tip</b> that will be
        displayed when you hover with th mouse cursor on the toolbar button
        specific for this annotation
        """
        descInfoTxt = f"{html_utils.paragraph(descInfoTxt)}"
        self.descWidget = widgets.formWidget(
            QPlainTextEdit(),
            addInfoButton=True,
            labelTextLeft="Description: ",
            parent=self,
            infoTxt=descInfoTxt,
        )
        if state is not None:
            self.descWidget.widget.setPlainText(state["description"])
        layout.addFormWidget(self.descWidget, row=row)

        row += 1
        optionsGroupBox = QGroupBox("Additional options")
        optionsLayout = QGridLayout()
        toggle = widgets.Toggle()
        toggle.setChecked(True)
        self.keepActiveToggle = toggle
        toggleLabel = QLabel("Keep tool active after using it: ")
        colorButtonLabel = QLabel("Symbol color: ")
        self.hideAnnotTooggle = widgets.Toggle()
        self.hideAnnotTooggle.setChecked(True)
        hideAnnotTooggleLabel = QLabel("Hide annotation when button is not active: ")
        self.colorButton = widgets.myColorButton(color=(255, 0, 0))
        self.colorButton.clicked.disconnect()
        self.colorButton.clicked.connect(self.selectColor)

        optionsLayout.setColumnStretch(0, 1)
        optRow = 0
        optionsLayout.addWidget(toggleLabel, optRow, 1)
        optionsLayout.addWidget(toggle, optRow, 2)
        optRow += 1
        optionsLayout.addWidget(hideAnnotTooggleLabel, optRow, 1)
        optionsLayout.addWidget(self.hideAnnotTooggle, optRow, 2)
        optionsLayout.setColumnStretch(3, 1)
        optRow += 1
        optionsLayout.addWidget(colorButtonLabel, optRow, 1)
        optionsLayout.addWidget(self.colorButton, optRow, 2)

        optionsGroupBox.setLayout(optionsLayout)
        layout.addWidget(optionsGroupBox, row, 1, alignment=Qt.AlignCenter)
        optionsInfoButton = QPushButton(self)
        optionsInfoButton.setCursor(Qt.WhatsThisCursor)
        optionsInfoButton.setIcon(QIcon(":info.svg"))
        optionsInfoButton.clicked.connect(self.showOptionsInfo)
        layout.addWidget(optionsInfoButton, row, 3, alignment=Qt.AlignRight)

        row += 1
        layout.addItem(QSpacerItem(5, 5), row, 0)

        row += 1
        noteText = (
            "<i>NOTE: you can change these options later with<br>"
            "<b>RIGHT-click</b> on the associated left-side <b>toolbar button<b>.</i>"
        )
        noteLabel = QLabel(html_utils.paragraph(noteText, font_size="11px"))
        layout.addWidget(noteLabel, row, 1, 1, 3)

        buttonsLayout = QHBoxLayout()

        self.loadSavedAnnotButton = widgets.OpenFilePushButton("  Load annotation...  ")
        if not savedCustomAnnot:
            self.loadSavedAnnotButton.setDisabled(True)
        self.okButton = widgets.okPushButton("  Ok  ")
        cancelButton = widgets.cancelPushButton("Cancel")

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(self.loadSavedAnnotButton)
        buttonsLayout.addWidget(self.okButton)

        cancelButton.clicked.connect(self.cancelCallBack)
        self.cancelButton = cancelButton
        self.loadSavedAnnotButton.clicked.connect(self.loadSavedAnnot)
        self.okButton.clicked.connect(self.ok_cb)
        self.okButton.setFocus()

        mainLayout = QVBoxLayout()

        noteTxt = """
        Custom annotations will be <b>saved in the <code>acdc_output.csv</code></b><br>
        file as a column with the name you write in the field <b>Name</b><br>
        """
        noteTxt = f"{html_utils.paragraph(noteTxt, font_size='15px')}"
        noteLabel = QLabel(noteTxt)
        noteLabel.setAlignment(Qt.AlignCenter)
        mainLayout.addWidget(noteLabel)

        mainLayout.addLayout(layout)
        mainLayout.addStretch(1)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def checkName(self, text):
        if not text:
            txt = "Name cannot be empty"
            self.nameInfoLabel.setText(
                html_utils.paragraph(txt, font_size="11px", font_color="red")
            )
            return
        for name in self.internalNames:
            if name.find(text) != -1:
                txt = f'"{text}" cannot be part of the name, because <b>reserved<b>.'
                self.nameInfoLabel.setText(
                    html_utils.paragraph(txt, font_size="11px", font_color="red")
                )
                break
        else:
            self.nameInfoLabel.setText("")

    def loadSavedAnnot(self):
        items = list(self.savedCustomAnnot.keys())
        self.selectAnnotWin = widgets.QDialogListbox(
            "Load annotation parameters",
            "Select annotation to load:",
            items,
            additionalButtons=("Delete selected annnotations",),
            parent=self,
            multiSelection=False,
        )
        for button in self.selectAnnotWin._additionalButtons:
            button.disconnect()
            button.clicked.connect(self.deleteSelectedAnnot)
        self.selectAnnotWin.exec_()
        if self.selectAnnotWin.cancel:
            return
        if self.selectAnnotWin.listBox.count() == 0:
            return
        if not self.selectAnnotWin.selectedItemsText:
            self.warnNoItemsSelected()
            return
        selectedName = self.selectAnnotWin.selectedItemsText[-1]
        selectedAnnot = self.savedCustomAnnot[selectedName]
        self.typeCombobox.setCurrentText(selectedAnnot["type"])
        self.nameWidget.widget.setText(selectedAnnot["name"])
        self.symbolWidget.widget.setCurrentText(selectedAnnot["symbol"])
        self.shortcutWidget.widget.setText(selectedAnnot["shortcut"])
        self.descWidget.widget.setPlainText(selectedAnnot["description"])
        self.colorButton.setColor(selectedAnnot["symbolColor"])
        keySequence = widgets.macShortcutToWindows(selectedAnnot["shortcut"])
        if keySequence:
            self.shortcutWidget.widget.keySequence = widgets.KeySequenceFromText(
                keySequence
            )

    def warnNoItemsSelected(self):
        msg = widgets.myMessageBox(parent=self)
        msg.setIcon(iconName="SP_MessageBoxWarning")
        msg.setWindowTitle("Delete annotation?")
        msg.addText("You didn't select any annotation!")
        msg.addButton("  Ok  ")
        msg.exec_()

    def deleteSelectedAnnot(self):
        msg = widgets.myMessageBox(parent=self)
        msg.setIcon(iconName="SP_MessageBoxWarning")
        msg.setWindowTitle("Delete annotation?")
        msg.addText("Are you sure you want to delete the selected annotations?")
        msg.addButton("Yes")
        cancelButton = msg.addButton(" Cancel ")
        msg.exec_()
        if msg.clickedButton == cancelButton:
            return
        for item in self.selectAnnotWin.listBox.selectedItems():
            name = item.text()
            self.savedCustomAnnot.pop(name)
        self.sigDeleteSelecAnnot.emit(self.selectAnnotWin.listBox.selectedItems())
        items = list(self.savedCustomAnnot.keys())
        self.selectAnnotWin.listBox.clear()
        self.selectAnnotWin.listBox.addItems(items)

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

    def warnType(self, currentText):
        if currentText == "Single time-point":
            return

        self.typeCombobox.setCurrentIndex(0)

        txt = """
        Unfortunately, the only annotation type that is available so far is
        <b>Single time-point</b>.<br><br>
        We are working on implementing the other types too, so stay tuned!<br><br>
        Thank you for your patience!
        """
        txt = f"{html_utils.paragraph(txt)}"
        msg = widgets.myMessageBox()
        msg.setIcon(iconName="SP_MessageBoxWarning")
        msg.setWindowTitle(f"Feature not implemented yet")
        msg.addText(txt)
        msg.addButton("   Ok   ")
        msg.exec_()

    def showOptionsInfo(self):
        info = """
        <b>Keep tool active after using it</b>: Choose whether the tool
        should stay active or not after annotating.<br><br>
        <b>Hide annotation when button is not active</b>: Choose whether
        annotation on the cell/object should be visible only if the
        button is active or also when it is not active.<br>
        <i>NOTE: annotations are <b>always stored</b> no matter whether
        they are visible or not.</i><br><br>
        <b>Symbol color</b>: Choose color of the symbol that will be used
        to label annotated cell/object.
        """
        info = f"{html_utils.paragraph(info)}"
        msg = widgets.myMessageBox()
        msg.setIcon()
        msg.setWindowTitle(f"Additional options info")
        msg.addText(info)
        msg.addButton("   Ok   ")
        msg.exec_()

    def ok_cb(self, checked=True):
        self.cancel = False
        self.clickedButton = self.okButton
        self.close()

    def cancelCallBack(self, checked=True):
        self.cancel = True
        self.clickedButton = self.cancelButton
        self.close()

    def showNameInfo(self):
        msg = widgets.myMessageBox()
        listView = widgets.readOnlyQList(msg)
        listView.addItems(self.internalNames)
        # listView.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        msg.information(
            self, "Annotation Name info", self.nameInfoTxt, widgets=listView
        )

    def closeEvent(self, event):
        if self.clickedButton is None or self.clickedButton == self.cancelButton:
            # cancel button or closed with 'x' button
            self.cancel = True
            return

        if self.clickedButton == self.okButton and not self.nameWidget.widget.text():
            msg = QMessageBox()
            msg.critical(self, "Empty name", "The name cannot be empty!", msg.Ok)
            event.ignore()
            self.cancel = True
            return

        if self.clickedButton == self.okButton and self.nameInfoLabel.text():
            msg = widgets.myMessageBox()
            listView = widgets.listWidget(msg)
            listView.addItems(self.internalNames)
            listView.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            name = self.nameWidget.widget.text()
            txt = (
                f'"{name}" cannot be part of the name, '
                "because it is <b>reserved</b> for standard measurements "
                "saved by Cell-ACDC.<br><br>"
                "Internally reserved names:"
            )
            msg.critical(
                self, "Not a valid name", html_utils.paragraph(txt), widgets=listView
            )
            event.ignore()
            self.cancel = True
            return

        self.toolTip = (
            f"Name: {self.nameWidget.widget.text()}\n\n"
            f"Type: {self.typeWidget.widget.currentText()}\n\n"
            f"Usage: activate the button and RIGHT-CLICK on cell to annotate\n\n"
            f"Description: {self.descWidget.widget.toPlainText()}\n\n"
            f'SHORTCUT: "{self.shortcutWidget.widget.text()}"'
        )

        symbol = self.symbolWidget.widget.currentText()
        self.symbol = re.findall(r"\'(.+)\'", symbol)[0]

        self.state = {
            "type": self.typeWidget.widget.currentText(),
            "name": self.nameWidget.widget.text(),
            "symbol": self.symbolWidget.widget.currentText(),
            "shortcut": self.shortcutWidget.widget.text(),
            "description": self.descWidget.widget.toPlainText(),
            "keepActive": self.keepActiveToggle.isChecked(),
            "isHideChecked": self.hideAnnotTooggle.isChecked(),
            "symbolColor": self.colorButton.color(),
        }

        if self.loop is not None:
            self.loop.exit()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()


class _PointsLayerAppearanceGroupbox(QGroupBox):
    def __init__(self, *args):
        super().__init__(*args)

        self.setTitle("Points appearance")

        layout = widgets.FormLayout()

        "----------------------------------------------------------------------"
        row = 0
        symbolInfoTxt = """
            <b>Symbol</b> used to draw the points.
        """
        symbolInfoTxt = f"{html_utils.paragraph(symbolInfoTxt)}"
        self.symbolWidget = widgets.formWidget(
            widgets.pgScatterSymbolsCombobox(),
            addInfoButton=True,
            labelTextLeft="Symbol: ",
            parent=self,
            infoTxt=symbolInfoTxt,
            stretchWidget=False,
        )
        layout.addFormWidget(self.symbolWidget, row=row)
        "----------------------------------------------------------------------"

        "----------------------------------------------------------------------"
        row += 1
        self.colorButton = widgets.myColorButton(color=(255, 0, 0))
        self.colorWidget = widgets.formWidget(
            self.colorButton, stretchWidget=True, labelTextLeft="Colour: ", parent=self
        )
        layout.addFormWidget(self.colorWidget, align=Qt.AlignLeft, row=row)
        self.colorButton.clicked.disconnect()
        self.colorButton.clicked.connect(self.selectColor)
        "----------------------------------------------------------------------"

        "----------------------------------------------------------------------"
        row += 1
        self.sizeSpinBox = widgets.SpinBox()
        self.sizeSpinBox.setValue(5)
        self.sizeWidget = widgets.formWidget(
            self.sizeSpinBox, stretchWidget=True, labelTextLeft="Size: ", parent=self
        )
        layout.addFormWidget(self.sizeWidget, row=row)
        "----------------------------------------------------------------------"

        "----------------------------------------------------------------------"
        row += 1
        zHeightTooltip = (
            'If "Z-depth" is greater than 1, the points will be annotated '
            "in all the z-slices in the range `z - (Z-depth/2) < z < z + (Z-depth/2)`\n"
            "where `z` is the center z-slice of the added point."
        )
        self.zHeightSpinBox = widgets.OddSpinBox()
        self.zHeightSpinBox.setValue(1)
        self.zHeightSpinBox.setMinimum(1)
        self.zHeightWidget = widgets.formWidget(
            self.zHeightSpinBox,
            stretchWidget=True,
            labelTextLeft="Z-depth: ",
            parent=self,
            toolTip=zHeightTooltip,
        )
        layout.addFormWidget(self.zHeightWidget, row=row)
        "----------------------------------------------------------------------"

        "----------------------------------------------------------------------"
        row += 1
        shortcutInfoTxt = """
        <b>Shortcut</b> that you can use to <b>hide/show</b> points.
        """
        shortcutInfoTxt = f"{html_utils.paragraph(shortcutInfoTxt)}"
        self.shortcutWidget = widgets.formWidget(
            widgets.ShortcutLineEdit(),
            addInfoButton=True,
            labelTextLeft="Shortcut: ",
            parent=self,
            infoTxt=shortcutInfoTxt,
        )
        layout.addFormWidget(self.shortcutWidget, row=row)
        "----------------------------------------------------------------------"

        self.setLayout(layout)

    def restoreState(self, state):
        self.shortcutWidget.widget.setText(state["shortcut"])
        self.colorButton.setColor(state["color"])
        self.symbolWidget.widget.setCurrentText(state["symbol"])
        self.sizeSpinBox.setValue(state["pointSize"])
        self.zHeightSpinBox.setValue(state["zHeight"])

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

    def state(self):
        r, g, b, a = self.colorButton.color().getRgb()
        _state = {
            "symbol": self.symbolWidget.widget.currentText(),
            "color": (r, g, b),
            "pointSize": self.sizeSpinBox.value(),
            "zHeight": self.zHeightSpinBox.value(),
            "shortcut": self.shortcutWidget.widget.text(),
        }
        return _state


class AddPointsLayerDialog(QBaseDialog):
    sigClosed = Signal()
    sigCriticalReadTable = Signal(str)
    sigLoadedTable = Signal(object, str)
    sigCheckClickEntryTableEndnameExists = Signal(str, bool)

    def __init__(
        self,
        channelNames=None,
        imagesPath="",
        SizeT=1,
        hideCentroidsSection=False,
        hideWeightedCentroidsSection=False,
        hideFromTableSection=False,
        hideManualEntrySection=False,
        hideWithMouseClicksSection=False,
        parent=None,
    ):
        self.cancel = True
        super().__init__(parent)

        self._parent = parent

        self.imagesPath = imagesPath

        self.setWindowTitle("Add points layer")
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        mainLayout = QVBoxLayout()

        scrollArea = widgets.ScrollArea()
        typeGroupbox = QGroupBox("Points to draw")
        typeLayout = QGridLayout()
        typeGroupbox.setLayout(typeLayout)
        typeLayout.addItem(QSpacerItem(10, 1), 0, 0)
        typeLayout.setColumnStretch(0, 0)
        typeLayout.setColumnStretch(2, 1)
        vSpacing = 15

        row = 0

        sections = (
            ("addCentroidsSection", hideCentroidsSection),
            ("addWeightedCentroidsSection", hideWeightedCentroidsSection),
            ("addFromTableSection", hideFromTableSection),
            ("addManualEntrySection", hideManualEntrySection),
            ("addWithMouseClicksSection", hideWithMouseClicksSection),
        )
        radioButtonChecked = False
        for section, hideSection in sections:
            addFunc = getattr(self, section)
            row, sectionWidgets = addFunc(
                row,
                typeLayout,
                imagesPath=imagesPath,
                SizeT=SizeT,
                channelNames=channelNames,
            )
            if not hideSection:
                spacer = QSpacerItem(1, vSpacing)
                typeLayout.addItem(spacer, row, 0)
                row += 1
                if not radioButtonChecked:
                    sectionWidgets[0].setChecked(True)
                    radioButtonChecked = True
                continue

            for widget in sectionWidgets:
                widget.setVisible(False)

        self.scrollArea = scrollArea
        scrollArea.setWidget(typeGroupbox)

        self.appearanceGroupbox = _PointsLayerAppearanceGroupbox()
        self.appearanceGroupbox.sizeSpinBox.setValue(3)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        self.buttonsLayout = buttonsLayout

        mainLayout.addWidget(scrollArea)
        mainLayout.addSpacing(20)
        _layout = QHBoxLayout()
        _layout.addWidget(self.appearanceGroupbox)
        _layout.addStretch(1)
        mainLayout.addLayout(_layout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        self.setFont(font)

    def addCentroidsSection(self, row, layout, **kwargs):
        sectionWidgets = []
        self.centroidsRadiobutton = QRadioButton("Centroids")
        layout.addWidget(self.centroidsRadiobutton, row, 0, 1, 2)
        sectionWidgets.append(self.centroidsRadiobutton)

        self.centroidsRadiobutton.setChecked(True)
        return row + 1, sectionWidgets

    def addWeightedCentroidsSection(self, row, layout, channelNames=None, **kwargs):
        if channelNames is None:
            channelNames = []

        sectionWidgets = []

        self.weightedCentroidsRadiobutton = QRadioButton("Weighted centroids")
        layout.addWidget(self.weightedCentroidsRadiobutton, row, 0, 1, 2)
        sectionWidgets.append(self.weightedCentroidsRadiobutton)

        row += 1
        label = QLabel("Weighing channel: ")
        label.setEnabled(False)
        layout.addWidget(label, row, 1)
        sectionWidgets.append(label)

        self.channelNameForWeightedCentr = widgets.QCenteredComboBox()
        if channelNames:
            self.channelNameForWeightedCentr.addItems(channelNames)
        self.channelNameForWeightedCentr.setDisabled(True)
        layout.addWidget(self.channelNameForWeightedCentr, row, 2)
        sectionWidgets.append(self.channelNameForWeightedCentr)

        self.weightedCentroidsRadiobutton.toggled.connect(label.setEnabled)
        self.weightedCentroidsRadiobutton.toggled.connect(
            self.channelNameForWeightedCentr.setEnabled
        )

        return row + 1, sectionWidgets

    def addFromTableSection(self, row, layout, imagesPath="", SizeT=1, **kwargs):
        sectionWidgets = []

        self.fromTableRadiobutton = QRadioButton("From table")
        layout.addWidget(self.fromTableRadiobutton, row, 0, 1, 2)
        sectionWidgets.append(self.fromTableRadiobutton)
        self.fromTableRadiobutton.widgets = []

        row += 1
        self.tablePath = widgets.ElidingLineEdit()
        self.tablePath.label = QLabel("Table file path: ")
        layout.addWidget(self.tablePath.label, row, 1)
        layout.addWidget(self.tablePath, row, 2)
        self.fromTableRadiobutton.widgets.append(self.tablePath)
        sectionWidgets.append(self.tablePath.label)
        sectionWidgets.append(self.tablePath)

        browseButton = widgets.browseFileButton(
            start_dir=imagesPath, ext={"Table": [".csv", ".h5"]}
        )
        layout.addWidget(browseButton, row, 3)
        browseButton.sigPathSelected.connect(self.tablePathSelected)
        self.browseTableButton = browseButton
        self.fromTableRadiobutton.widgets.append(browseButton)
        sectionWidgets.append(browseButton)

        row += 1
        self.xColName = widgets.QCenteredComboBox()
        self.xColName.addItem("None")
        self.xColName.label = QLabel("X coord. column: ")
        layout.addWidget(self.xColName.label, row, 1)
        layout.addWidget(self.xColName, row, 2)
        self.xColName.currentTextChanged.connect(self.checkColNameX)
        self.fromTableRadiobutton.widgets.append(self.xColName)
        sectionWidgets.append(self.xColName.label)
        sectionWidgets.append(self.xColName)

        row += 1
        self.yColName = widgets.QCenteredComboBox()
        self.yColName.addItem("None")
        self.yColName.label = QLabel("Y coord. column: ")
        layout.addWidget(self.yColName.label, row, 1)
        layout.addWidget(self.yColName, row, 2)
        self.yColName.currentTextChanged.connect(self.checkColNameY)
        self.fromTableRadiobutton.widgets.append(self.yColName)
        sectionWidgets.append(self.yColName.label)
        sectionWidgets.append(self.yColName)

        row += 1
        self.zColName = widgets.QCenteredComboBox()
        self.zColName.addItem("None")
        self.zColName.label = QLabel("Z coord. column: ")
        layout.addWidget(self.zColName.label, row, 1)
        layout.addWidget(self.zColName, row, 2)
        self.zColName.currentTextChanged.connect(self.checkColNameZ)
        self.fromTableRadiobutton.widgets.append(self.zColName)
        sectionWidgets.append(self.zColName.label)
        sectionWidgets.append(self.zColName)

        row += 1
        self.tColName = widgets.QCenteredComboBox()
        self.tColName.addItem("None")
        self.tColName.label = QLabel("Frame index column: ")
        layout.addWidget(self.tColName.label, row, 1)
        layout.addWidget(self.tColName, row, 2)
        self.fromTableRadiobutton.widgets.append(self.tColName)
        sectionWidgets.append(self.tColName.label)
        sectionWidgets.append(self.tColName)

        if SizeT == 1:
            self.tColName.clear()
            self.tColName.addItem("None")
            self.tColName.label.setVisible(False)
            self.tColName.setVisible(False)

        self.fromTableRadiobutton.toggled.connect(self.enableRadioButtonWidgets)
        self.enableRadioButtonWidgets(False, sender=self.fromTableRadiobutton)

        return row + 1, sectionWidgets

    def addManualEntrySection(self, row, layout, SizeT=1, **kwargs):
        sectionWidgets = []

        self.manualEntryRadiobutton = QRadioButton("Manual entry")
        layout.addWidget(self.manualEntryRadiobutton, row, 0, 1, 2)
        self.manualEntryRadiobutton.widgets = []
        sectionWidgets.append(self.manualEntryRadiobutton)

        row += 1
        self.manualXspinbox = widgets.NumericCommaLineEdit()
        self.manualXspinbox.label = QLabel("X coords: ")
        layout.addWidget(self.manualXspinbox.label, row, 1)
        layout.addWidget(self.manualXspinbox, row, 2)
        self.manualEntryRadiobutton.widgets.append(self.manualXspinbox)
        sectionWidgets.append(self.manualXspinbox.label)
        sectionWidgets.append(self.manualXspinbox)

        row += 1
        self.manualYspinbox = widgets.NumericCommaLineEdit()
        self.manualYspinbox.label = QLabel("Y coords: ")
        layout.addWidget(self.manualYspinbox.label, row, 1)
        layout.addWidget(self.manualYspinbox, row, 2)
        self.manualEntryRadiobutton.widgets.append(self.manualYspinbox)
        sectionWidgets.append(self.manualYspinbox.label)
        sectionWidgets.append(self.manualYspinbox)

        row += 1
        self.manualZspinbox = widgets.NumericCommaLineEdit()
        self.manualZspinbox.label = QLabel("Z coords: ")
        layout.addWidget(self.manualZspinbox.label, row, 1)
        layout.addWidget(self.manualZspinbox, row, 2)
        self.manualEntryRadiobutton.widgets.append(self.manualZspinbox)
        sectionWidgets.append(self.manualZspinbox.label)
        sectionWidgets.append(self.manualZspinbox)

        row += 1
        self.manualTspinbox = widgets.NumericCommaLineEdit()
        self.manualTspinbox.label = QLabel("Frame numbers: ")
        layout.addWidget(self.manualTspinbox.label, row, 1)
        layout.addWidget(self.manualTspinbox, row, 2)
        self.manualEntryRadiobutton.widgets.append(self.manualTspinbox)
        sectionWidgets.append(self.manualTspinbox.label)
        sectionWidgets.append(self.manualTspinbox)

        if SizeT == 1:
            self.manualTspinbox.setVisible(False)
            self.manualTspinbox.label.setVisible(False)

        self.manualEntryRadiobutton.toggled.connect(self.enableRadioButtonWidgets)
        self.enableRadioButtonWidgets(False, sender=self.manualEntryRadiobutton)

        return row + 1, sectionWidgets

    def addWithMouseClicksSection(self, row, layout, imagesPath="", **kwargs):
        sectionWidgets = []

        self.clickEntryIsLoadedDf = None

        self.clickEntryRadiobutton = QRadioButton("Add points with mouse clicks")
        layout.addWidget(self.clickEntryRadiobutton, row, 0, 1, 2)
        self.clickEntryRadiobutton.widgets = []
        sectionWidgets.append(self.clickEntryRadiobutton)

        row += 1
        self.snapToMaxToggle = widgets.Toggle()
        self.snapToMaxToggle.label = QLabel("Snap to closest maximum: ")
        layout.addWidget(self.snapToMaxToggle.label, row, 1)
        layout.addWidget(self.snapToMaxToggle, row, 2, alignment=Qt.AlignCenter)
        sectionWidgets.append(self.snapToMaxToggle.label)
        sectionWidgets.append(self.snapToMaxToggle)

        self.snapToMaxInfoButton = widgets.infoPushButton()
        layout.addWidget(self.snapToMaxInfoButton, row, 3)
        sectionWidgets.append(self.snapToMaxInfoButton)

        self.snapToMaxInfoButton.clicked.connect(self.showSnapToMaxButton)
        self.clickEntryRadiobutton.widgets.append(self.snapToMaxToggle)
        self.clickEntryRadiobutton.widgets.append(self.snapToMaxInfoButton)

        row += 1
        self.autoPilotToggle = widgets.Toggle()
        self.autoPilotToggle.label = QLabel("Use auto-pilot: ")
        layout.addWidget(self.autoPilotToggle.label, row, 1)
        layout.addWidget(self.autoPilotToggle, row, 2, alignment=Qt.AlignCenter)
        sectionWidgets.append(self.autoPilotToggle.label)
        sectionWidgets.append(self.autoPilotToggle)
        self.autoPilotInfoButton = widgets.infoPushButton()
        layout.addWidget(self.autoPilotInfoButton, row, 3)
        sectionWidgets.append(self.autoPilotInfoButton)

        self.autoPilotInfoButton.clicked.connect(self.showAutoPilotInfo)
        self.clickEntryRadiobutton.widgets.append(self.autoPilotToggle)
        self.clickEntryRadiobutton.widgets.append(self.autoPilotInfoButton)

        row += 1
        self.clickEntryTableEndname = widgets.alphaNumericLineEdit()
        self.clickEntryTableEndname.setText("points_added_by_clicking")
        self.clickEntryTableEndname.setAlignment(Qt.AlignCenter)
        self.clickEntryTableEndname.label = QLabel("Table endname: ")
        loadButton = widgets.browseFileButton(start_dir=imagesPath, ext={"CSV": ".csv"})
        layout.addWidget(loadButton, row, 3)
        sectionWidgets.append(loadButton)

        loadButton.sigPathSelected.connect(self.loadClickEntryTable)
        self.loadButton = loadButton
        self.clickEntryLoadTableButton = loadButton
        layout.addWidget(self.clickEntryTableEndname.label, row, 1)
        layout.addWidget(self.clickEntryTableEndname, row, 2)
        self.clickEntryRadiobutton.widgets.append(self.clickEntryTableEndname)
        self.clickEntryTableEndname.editingFinished.connect(
            self.emitCheckClickEntryTableEndnameExists
        )
        sectionWidgets.append(self.clickEntryTableEndname)
        sectionWidgets.append(self.clickEntryTableEndname.label)

        row += 1
        instructionsText = html_utils.paragraph(
            "<br><i>Left-click</i> to annotate a new point with a new id.<br><br>"
            "<i>Right-click</i> to annotate a point with the same id<br><br>"
            "<i>Same click used to delete objects</i> to annotate<br>"
            "a point with id = 0 (negative prompt)<br><br>"
            "<i>Click</i> on point to delete it",
            font_size="11px",
        )
        self.instructionsLabel = QLabel(instructionsText)
        self.instructionsLabel.label = QLabel("Instructions")
        layout.addWidget(self.instructionsLabel.label, row, 1)
        layout.addWidget(self.instructionsLabel, row, 2)
        self.clickEntryRadiobutton.widgets.append(self.instructionsLabel)
        sectionWidgets.append(self.instructionsLabel)
        sectionWidgets.append(self.instructionsLabel.label)

        self.clickEntryRadiobutton.toggled.connect(self.enableRadioButtonWidgets)
        self.clickEntryRadiobutton.toggled.connect(
            self.emitCheckClickEntryTableEndnameExists
        )
        self.enableRadioButtonWidgets(False, sender=self.clickEntryRadiobutton)

        return row + 1, sectionWidgets

    def emitCheckClickEntryTableEndnameExists(self, *args, **kwargs):
        if not self.clickEntryRadiobutton.isChecked():
            return
        self.clickEntryIsLoadedDf = None
        tableEndName = self.clickEntryTableEndname.text()
        self.sigCheckClickEntryTableEndnameExists.emit(tableEndName, False)

    def loadClickEntryTable(self, csv_path):
        self.clickEntryIsLoadedDf = None
        posData = load.loadData(csv_path, "points")
        posData.getBasenameAndChNames(qparent=self)
        basename = posData.basename
        filename = os.path.basename(csv_path)
        filename, ext = os.path.splitext(filename)
        if not basename.endswith("_"):
            basename = f"{basename}_"

        endname = filename[len(basename) :]
        self.clickEntryTableEndname.setText(endname)
        self.sigCheckClickEntryTableEndnameExists.emit(endname, True)

    def showAutoPilotInfo(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            With <b>Auto-pilot</b> mode active, Cell-ACDC will <b>automatically zoom</b> on 
            to an object<br>
            to allow you clicking on the points you want to add.<br><br>
            You can then go to the <b>next object</b> by pressing the 
            <code>Enter</code> key or go back to the<br>
            <b>previous object</b> by pressing <code>Backspace</code>.
        """)
        msg.information(self, "Auto-pilot info", txt)

    def showSnapToMaxButton(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            With <bSnap to closest maximum</b> mode active, Cell-ACDC will 
            <b>automatically add the point</b><br>
            to the closest maximum within the point footprint (defined in 
            the appearance settings).
        """)
        msg.information(self, "Snap to closest maximum info", txt)

    def closeEvent(self, event):
        self.sigClosed.emit()

    def enableRadioButtonWidgets(self, enabled, sender=None):
        if sender is None:
            sender = self.sender()
        for widget in sender.widgets:
            widget.setDisabled(not enabled)
            try:
                widget.label.setDisabled(not enabled)
            except:
                pass

    def _readTable(self, path):
        return load.load_df_points_layer(path)

    def tryAutoFillColNames(self, df):
        if "x" in df.columns:
            self.xColName.setCurrentText("x")

        if "y" in df.columns:
            self.yColName.setCurrentText("y")

        if "z" in df.columns:
            self.zColName.setCurrentText("z")

        if "frame_i" in df.columns:
            self.tColName.setCurrentText("frame_i")

    def tablePathSelected(self, path):
        self.tablePath.setText(path)
        try:
            df = self._readTable(path)
            self.xColName.addItems(df.columns)
            self.yColName.addItems(df.columns)
            self.zColName.addItems(df.columns)
            self.tColName.addItems(df.columns)
            self.tryAutoFillColNames(df)
            self.sigLoadedTable.emit(df, os.path.basename(path))
            self.browseTableButton.confirmAction()
        except Exception as e:
            traceback_format = traceback.format_exc()
            self.sigCriticalReadTable.emit(traceback_format)
            self.criticalReadTable(path, traceback_format)
            self.tablePath.setText("")

    def criticalLenMismatchManualEntry(self):
        txt = html_utils.paragraph(f"""
            X coords and Y coords must have the <b>same length</b>.
        """)
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.critical(self, f"X and Y have different length", txt)

    def criticalColNameIsNone(self, axis):
        txt = html_utils.paragraph(f"""
            The "{axis.upper()} coord. column" <b>cannot be "None"</b>
        """)
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.critical(self, f"{axis.upper()} coord. is None", txt)

    def criticalReadTable(self, path, traceback_format):
        txt = html_utils.paragraph(f"""
            Something went <b>wrong when reading the table</b> from the 
            following path:<br><br>
            <code>{path}</code><br><br>
            See the <b>error message below</b>.
        """)
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        detailsText = traceback_format
        msg.critical(self, "Error when reading table", txt, detailsText=detailsText)

    def criticalEmptyTablePath(self):
        txt = html_utils.paragraph(f"""
            The table file path <b>cannot be empty</b>.
        """)
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.critical(self, "Table file path is empty", txt)

    def state(self):
        _state = self.appearanceGroupbox.state()
        return _state

    def _checkSelectedColName(self, colName, label):
        labelsToCheck = ["z", "y", "x"]
        labelsToCheck.remove(label)
        for labelToCheck in labelsToCheck:
            if colName.find(labelToCheck) != -1:
                break
        else:
            return True

        txt = html_utils.paragraph(f"""
            Are you sure that the {label.upper()} coord. column should contain 
            the letter <code>{labelToCheck}<code>?
        """)

        msg = widgets.myMessageBox(wrapText=False)
        _, noButton, yesButton = msg.warning(
            self,
            "Check column name",
            txt,
            buttonsTexts=("Cancel", "No, let me correct it", "Yes, I am"),
        )
        if msg.cancel or msg.clickedButton == noButton:
            return False
        return True

    def checkColNameX(self, text):
        accepted = self._checkSelectedColName(text, "x")
        if accepted:
            return
        self.xColName.setCurrentText("None")

    def checkColNameY(self, text):
        accepted = self._checkSelectedColName(text, "y")
        if accepted:
            return
        self.yColName.setCurrentText("None")

    def checkColNameZ(self, text):
        accepted = self._checkSelectedColName(text, "z")
        if accepted:
            return
        self.zColName.setCurrentText("None")

    def ok_cb(self):
        self.pointsData = {}
        self.loadedDfInfo = None
        self.loadedDf = None
        self.weighingChannel = ""
        if self.fromTableRadiobutton.isChecked():
            tablePath = self.tablePath.text()
            if not tablePath:
                self.criticalEmptyTablePath()
                return

            try:
                df = self._readTable(tablePath)
                tColName = self.tColName.currentText()
                xColName = self.xColName.currentText()
                yColName = self.yColName.currentText()
                zColName = self.zColName.currentText()

                self.loadedDfInfo = {
                    "filepath": tablePath,
                    "t": tColName,
                    "z": zColName,
                    "y": yColName,
                    "x": xColName,
                }

                self._df_to_pointsData(df, tColName, zColName, yColName, xColName)

            except Exception as e:
                traceback_format = traceback.format_exc()
                self.sigCriticalReadTable.emit(traceback_format)
                self.criticalReadTable(tablePath, traceback_format)
                return

            if self.xColName.currentText() == "None":
                self.criticalColNameIsNone("x")
                return
            if self.yColName.currentText() == "None":
                self.criticalColNameIsNone("y")
                return

            self.layerType = os.path.basename(self.tablePath.text())
            self.layerTypeIdx = 2
        elif self.centroidsRadiobutton.isChecked():
            self.layerType = "Centroids"
            self.layerTypeIdx = 0
        elif self.weightedCentroidsRadiobutton.isChecked():
            channel = self.channelNameForWeightedCentr.currentText()
            self.weighingChannel = channel
            self.layerType = f"Centroids weighted by channel {channel}"
            self.layerTypeIdx = 1
        elif self.manualEntryRadiobutton.isChecked():
            xx = self.manualXspinbox.values()
            yy = self.manualYspinbox.values()
            if len(xx) != len(yy):
                self.criticalLenMismatchManualEntry()
                return
            zz = self.manualZspinbox.values()
            tt = [t + 1 for t in self.manualTspinbox.values()]
            df = pd.DataFrame({"x": xx, "y": yy, "id": np.arange(1, len(xx) + 1)})
            if tt:
                df["t"] = tt
                tCol = "t"
            else:
                tCol = "None"
            if zz:
                df["z"] = zz
                zCol = "z"
            else:
                zCol = "None"

            self._df_to_pointsData(df, tCol, zCol, "y", "x")

            self.layerType = "Manual entry"
            self.layerTypeIdx = 3
        elif self.clickEntryRadiobutton.isChecked():
            self.layerType = "Click to annotate point"
            self.description = (
                "Left-click to add a point, click on point to delete it.\n"
                "With auto-pilot you can navigate through object with Up/Down arrows."
            )
            self.clickEntryTableEndnameText = self.clickEntryTableEndname.text()
            self.layerTypeIdx = 4

        self.cancel = False
        symbol = self.appearanceGroupbox.symbolWidget.widget.currentText()
        self.symbol = re.findall(r"\'(.+)\'", symbol)[0]
        self.symbolText = symbol
        self.color = self.appearanceGroupbox.colorButton.color()
        self.pointSize = self.appearanceGroupbox.sizeSpinBox.value()
        self.zHeight = self.appearanceGroupbox.zHeightSpinBox.value()
        shortcutWidget = self.appearanceGroupbox.shortcutWidget
        self.shortcut = shortcutWidget.widget.text()
        self.keySequence = shortcutWidget.widget.keySequence
        self.close()

    def _df_to_pointsData(self, df, tColName, zColName, yColName, xColName):
        self.pointsData = load.loaded_df_to_points_data(
            df, tColName, zColName, yColName, xColName
        )

    def showEvent(self, event) -> None:
        if self._parent is None:
            screen = self.screen()
        else:
            screen = self._parent.screen()
        screenWidth = screen.size().width()
        screenHeight = screen.size().height()

        maxHeight = screenHeight - 100

        buttonHeight = self.buttonsLayout.okButton.minimumSizeHint().height()
        height = (
            self.scrollArea.minimumHeightNoScrollbar()
            + self.appearanceGroupbox.sizeHint().height()
            + buttonHeight
            + 70
        )
        width = self.scrollArea.minimumWidthNoScrollbar() + 50

        height = min(height, maxHeight)

        self.resize(width, height)

        screenLeft = screen.geometry().x()
        screenTop = screen.geometry().y()
        w, h = self.width(), self.height()
        left = int(screenLeft + screenWidth / 2 - w / 2)
        top = int(screenTop + screenHeight / 2 - h / 2 - 20)

        self.move(left, top)


class EditPointsLayerAppearanceDialog(QBaseDialog):
    sigClosed = Signal()

    def __init__(self, parent=None):
        self.cancel = True
        super().__init__(parent)

        self._parent = parent

        self.setWindowTitle("Custom annotation")
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        mainLayout = QVBoxLayout()

        self.appearanceGroupbox = _PointsLayerAppearanceGroupbox()

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(self.appearanceGroupbox)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        self.setFont(font)

    def restoreState(self, state):
        self.appearanceGroupbox.restoreState(state)

    def closeEvent(self, event):
        super().closeEvent(event)
        self.sigClosed.emit()

    def state(self):
        _state = self.appearanceGroupbox.state()
        return _state

    def ok_cb(self):
        self.cancel = False
        symbol = self.appearanceGroupbox.symbolWidget.widget.currentText()
        self.symbol = re.findall(r"\'(.+)\'", symbol)[0]
        self.color = self.appearanceGroupbox.colorButton.color()
        self.pointSize = self.appearanceGroupbox.sizeSpinBox.value()
        self.zHeight = self.appearanceGroupbox.zHeightSpinBox.value()
        shortcutWidget = self.appearanceGroupbox.shortcutWidget
        self.shortcut = shortcutWidget.widget.text()
        self.keySequence = shortcutWidget.widget.keySequence
        self.close()


class QDialogWorkerProgress(QDialog):
    sigClosed = Signal(bool)

    def __init__(
        self,
        title="Progress",
        infoTxt="",
        showInnerPbar=False,
        pbarDesc="",
        parent=None,
    ):
        self.workerFinished = False
        self.aborted = False
        self.clickCount = 0
        super().__init__(parent)

        abort_text = "Option+Command+C" if is_mac else "Ctrl+Alt+C"
        self.abort_text = abort_text

        self.setWindowTitle(f"{title} ({abort_text} to abort)")
        self.setWindowFlags(Qt.Window)

        mainLayout = QVBoxLayout()
        pBarLayout = QGridLayout()

        if infoTxt:
            infoLabel = QLabel(infoTxt)
            mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        self.progressLabel = QLabel(pbarDesc)

        self.mainPbar = widgets.ProgressBarWithETA(self)
        self.mainPbar.setValue(0)
        pBarLayout.addWidget(self.mainPbar, 0, 0)
        pBarLayout.addWidget(self.mainPbar.ETA_label, 0, 1)

        self.innerPbar = widgets.ProgressBarWithETA(self)
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
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(f"""
            Aborting with <code>{self.abort_text}</code> to abort is 
            <b>not safe</b>.<br><br>
            The system status cannot be predicted and
            it will <b>require a restart</b>.<br><br>
            Are you sure you want to abort?
        """)
        yesButton, noButton = msg.critical(
            self, "Are you sure you want to abort?", txt, buttonsTexts=("Yes", "No")
        )
        return msg.clickedButton == yesButton

    def closeEvent(self, event):
        if not self.workerFinished:
            event.ignore()
            return

        self.sigClosed.emit(self.aborted)

    def log(self, text):
        self.logConsole.append(text)

    def show(self, app):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        QDialog.show(self)
        screen = app.primaryScreen()
        screenWidth = screen.size().width()
        screenHeight = screen.size().height()
        parentGeometry = self.parent().geometry()
        mainWinLeft, mainWinWidth = parentGeometry.left(), parentGeometry.width()
        mainWinTop, mainWinHeight = parentGeometry.top(), parentGeometry.height()
        mainWinCenterX = int(mainWinLeft + mainWinWidth / 2)
        mainWinCenterY = int(mainWinTop + mainWinHeight / 2)

        width = int(screenWidth / 3)
        width = width if self.width() < width else self.width()
        height = int(screenHeight / 3)
        left = int(mainWinCenterX - width / 2)
        left = left if left >= 0 else 0
        top = int(mainWinCenterY - height / 2)

        self.setGeometry(left, top, width, height)


class QDialogCombobox(QDialog):
    def __init__(
        self,
        title,
        ComboBoxItems,
        informativeText,
        CbLabel="Select value:  ",
        parent=None,
        defaultChannelName=None,
        iconPixmap=None,
        centeredCombobox=False,
    ):
        self.cancel = True
        self.selectedItemText = ""
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

        if centeredCombobox:
            combobox = widgets.QCenteredComboBox()
        else:
            combobox = QComboBox()
        combobox.addItems(ComboBoxItems)
        if defaultChannelName is not None and defaultChannelName in ComboBoxItems:
            combobox.setCurrentText(defaultChannelName)
        self.ComboBox = combobox
        topLayout.addWidget(combobox)
        topLayout.setContentsMargins(0, 10, 0, 0)

        okButton = widgets.okPushButton("Ok")

        cancelButton = widgets.cancelPushButton("Cancel")

        bottomLayout.addStretch(1)
        bottomLayout.addWidget(cancelButton)
        bottomLayout.addSpacing(20)
        bottomLayout.addWidget(okButton)
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

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setFont(font)

    def ok_cb(self, checked=False):
        self.cancel = False
        self.selectedItemText = self.ComboBox.currentText()
        self.selectedItemIdx = self.ComboBox.currentIndex()
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        QDialog.show(self)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()


class imageViewer(QMainWindow):
    """Main Window."""

    sigClosed = Signal()
    sigHoveringImage = Signal(object, object)

    def __init__(
        self,
        parent=None,
        posData=None,
        button_toUncheck=None,
        spinBox=None,
        linkWindow=None,
        enableOverlay=False,
        isSigleFrame=False,
        enableMirroredCursor=False,
    ):
        self.button_toUncheck = button_toUncheck
        self.parent = parent
        self.posData = posData
        self.spinBox = spinBox
        self.linkWindow = linkWindow
        self.enableMirroredCursor = enableMirroredCursor
        self.isSigleFrame = isSigleFrame
        self.minMaxValuesMapper = None
        """Initializer."""
        super().__init__(parent)

        if posData is None:
            posData = self.parent.data[self.parent.pos_i]
        self.posData = posData
        self.enableOverlay = enableOverlay

        self.gui_createActions()
        self.gui_createMenuBar()
        self.gui_createToolBars()

        self.gui_createStatusBar()

        self.gui_createGraphics()

        self.gui_connectImgActions()

        self.gui_createImgWidgets()
        self.gui_connectActions()

        self.gui_setSingleFrameMode(self.isSigleFrame)

        self.setupMirroredCursor()

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.graphLayout, 0, 0, 1, 1)
        mainLayout.addLayout(self.img_Widglayout, 1, 0)

        mainContainer.setLayout(mainLayout)

        self.frame_i = posData.frame_i
        self.num_frames = posData.SizeT

        version = myutils.read_version()
        self.setWindowTitle(f"Cell-ACDC v{version} - {posData.relPath}")

    def gui_createActions(self):
        # File actions
        self.exitAction = QAction("&Exit", self)

        # Toolbar actions
        self.prevAction = QAction("Previous frame", self)
        self.nextAction = QAction("Next Frame", self)
        self.jumpForwardAction = QAction("Jump to 10 frames ahead", self)
        self.jumpBackwardAction = QAction("Jump to 10 frames back", self)
        self.prevAction.setShortcut("left")
        self.nextAction.setShortcut("right")
        self.jumpForwardAction.setShortcut("up")
        self.jumpBackwardAction.setShortcut("down")
        self.addAction(self.nextAction)
        self.addAction(self.prevAction)
        self.addAction(self.jumpBackwardAction)
        self.addAction(self.jumpForwardAction)
        if self.enableOverlay:
            self.overlayButton = widgets.rightClickToolButton(parent=self)
            self.overlayButton.setIcon(QIcon(":overlay.svg"))
            self.overlayButton.setCheckable(True)

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

        self.editToolBar = editToolBar

        if self.enableOverlay:
            editToolBar.addWidget(self.overlayButton)

        if self.linkWindow:
            # Insert a spacing
            editToolBar.addWidget(QLabel("  "))
            self.linkWindowCheckbox = QCheckBox("Link to main GUI")
            self.linkWindowCheckbox.setChecked(True)
            editToolBar.addWidget(self.linkWindowCheckbox)

        if self.enableMirroredCursor:
            self.showMirroredCursorCheckbox = QCheckBox(
                "Show mirrored cursor from main window"
            )
            self.showMirroredCursorCheckbox.setChecked(True)
            editToolBar.addWidget(self.showMirroredCursorCheckbox)

    def setupMirroredCursor(self):
        self.cursor = pg.ScatterPlotItem(
            symbol="+",
            pxMode=True,
            pen=pg.mkPen("k", width=1),
            brush=pg.mkBrush("w"),
            size=16,
            tip=None,
        )
        self.Plot.addItem(self.cursor)

    def gui_connectActions(self):
        self.exitAction.triggered.connect(self.close)
        self.prevAction.triggered.connect(self.prev_frame)
        self.nextAction.triggered.connect(self.next_frame)
        self.jumpForwardAction.triggered.connect(self.skip10ahead_frames)
        self.jumpBackwardAction.triggered.connect(self.skip10back_frames)
        if self.enableOverlay:
            self.overlayButton.toggled.connect(self.overlay_cb)
            self.overlayButton.sigRightClick.connect(self.showOverlayContextMenu)

    def gui_setSingleFrameMode(self, isSingleFrame: bool):
        if not isSingleFrame:
            return

        self.framesScrollBar.setDisabled(True)
        self.framesScrollBar.setVisible(False)
        self.frameLabel.hide()
        self.t_label.hide()
        self.prevAction.triggered.disconnect()
        self.nextAction.triggered.disconnect()
        self.jumpForwardAction.triggered.disconnect()
        self.jumpBackwardAction.triggered.disconnect()
        self.editToolBar.setVisible(False)

    def showOverlayContextMenu(self, event):
        if not self.overlayButton.isChecked():
            return

        if self.parent is not None:
            self.overlayContextMenu.exec_(QCursor.pos())

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
        self.Plot.hideAxis("bottom")
        self.Plot.hideAxis("left")
        self.graphLayout.addItem(self.Plot, row=1, col=1)

        # Image Item
        self.img = widgets.BaseImageItem()
        self.img.setEnableAutoLevels(True)
        self.Plot.addItem(self.img)

        # Image histogram
        self.imgGrad = widgets.myHistogramLUTitem(isViewer=True)
        self.imgGrad.gradient.showMenu = self.showLutItemOverlayContextMenu
        self.imgGrad.vb.raiseContextMenu = lambda x: None
        self.imgGrad.setImageItem(self.img)
        self.graphLayout.addItem(self.imgGrad, row=1, col=0)

        # Current frame text
        self.frameLabel = pg.LabelItem(justify="center", color="w", size="14pt")
        self.frameLabel.setText(" ")
        self.graphLayout.addItem(self.frameLabel, row=2, col=0, colspan=2)

        if not self.enableOverlay:
            return

    def gui_createOverlayItems(self):
        self.createOverlayChannelsActions()
        self.overlayLayersItems = {}
        for ch in self.posData.chNames:
            if ch == self.parent.user_ch_name:
                continue
            overlayItems = self.getOverlayItems(ch)
            imageItem, lutItem, alphaScrollbar = overlayItems
            lutItem.vb.raiseContextMenu = lambda x: None
            lutItem.gradient.showMenu = self.showLutItemOverlayContextMenu
            lutItem.overlayColorButton.sigColorChanging.connect(self.updateOlColors)
            self.addAlphaScrollbar(ch, imageItem, alphaScrollbar)
            self.overlayLayersItems[ch] = overlayItems
            self.Plot.addItem(imageItem)

    def createOverlayChannelsActions(self):
        self.overlayLutItemAdditionalActions = []
        separator = QAction(self)
        separator.setSeparator(True)
        self.overlayLutItemAdditionalActions.append(separator)
        section = self.imgGrad.gradient.menu.addSection("Select channel to adjust: ")
        self.overlayLutItemAdditionalActions.append(section)
        self.imgGrad.gradient.menu.removeAction(section)

        self.overlayChNamesActionGroup = QActionGroup(self)
        self.overlayChNamesActionGroup.setExclusive(True)
        for chName in self.posData.chNames:
            action = QAction(chName, self)
            action.setCheckable(True)
            if chName == self.parent.user_ch_name:
                action.setChecked(True)
            self.overlayChNamesActionGroup.addAction(action)
        self.overlayChNamesActionGroup.triggered.connect(
            self.chNameGradientActionClicked
        )

    def chNameGradientActionClicked(self, action):
        # Action triggered from lutItem
        self.checkedOverlayChName = action.text()
        if action.text() == self.posData.user_ch_name:
            self.setOverlayItemsVisible("", False)
        else:
            self.setOverlayItemsVisible(action.text(), True)

    def showLutItemOverlayContextMenu(self, event):
        lutItem = self.currentLutItem

        for action in self.overlayLutItemAdditionalActions:
            try:
                lutItem.gradient.menu.removeAction(action)
            except Exception as e:
                pass

        for action in self.overlayChNamesActionGroup.actions():
            try:
                lutItem.gradient.menu.removeAction(action)
            except Exception as e:
                pass

        if self.overlayButton.isChecked():
            for action in self.overlayLutItemAdditionalActions:
                lutItem.gradient.menu.addAction(action)

            for action in self.overlayChNamesActionGroup.actions():
                if action.text() == self.posData.user_ch_name:
                    lutItem.gradient.menu.addAction(action)
                    continue
                for filename in self.posData.ol_data:
                    if filename.endswith(action.text()):
                        lutItem.gradient.menu.addAction(action)
                        break
                    if filename.endswith(f"{action.text()}_aligned"):
                        lutItem.gradient.menu.addAction(action)
                        break

        try:
            # Convert QPointF to QPoint
            lutItem.gradient.menu.popup(event.screenPos().toPoint())
        except AttributeError:
            lutItem.gradient.menu.popup(event.screenPos())

    def gui_connectImgActions(self):
        self.img.hoverEvent = self.gui_hoverEventImg

    def gui_createImgWidgets(self):
        if self.posData is None:
            posData = self.parent.data[self.parent.pos_i]
        else:
            posData = self.posData
        self.img_Widglayout = QGridLayout()

        # Frames scrollbar
        self.framesScrollBar = QScrollBar(Qt.Horizontal)
        # self.framesScrollBar.setFixedHeight(20)
        self.framesScrollBar.setMinimum(1)
        self.framesScrollBar.setMaximum(posData.SizeT)
        t_label = QLabel("frame  ")
        _font = QFont()
        _font.setPixelSize(12)
        t_label.setFont(_font)
        self.img_Widglayout.addWidget(t_label, 0, 0, alignment=Qt.AlignRight)
        self.img_Widglayout.addWidget(self.framesScrollBar, 0, 1, 1, 20)
        self.t_label = t_label
        self.framesScrollBar.valueChanged.connect(self.framesScrollBarMoved)

        # z-slice scrollbar
        self.zSliceScrollBar = QScrollBar(Qt.Horizontal)
        # self.zSliceScrollBar.setFixedHeight(20)
        self.zSliceScrollBar.setMaximum(self.posData.SizeZ - 1)
        _z_label = QLabel("z-slice  ")
        _font = QFont()
        _font.setPixelSize(12)
        _z_label.setFont(_font)
        self.z_label = _z_label
        self.img_Widglayout.addWidget(_z_label, 1, 0, alignment=Qt.AlignCenter)
        self.img_Widglayout.addWidget(self.zSliceScrollBar, 1, 1, 1, 20)

        if self.posData.SizeZ == 1:
            self.zSliceScrollBar.setDisabled(True)
            self.zSliceScrollBar.setVisible(False)
            _z_label.setVisible(False)

        self.img_Widglayout.setContentsMargins(100, 0, 50, 0)
        self.zSliceScrollBar.valueChanged.connect(self.update_z_slice)

        if self.enableOverlay:
            self.setOverlayColors()
            self.gui_createOverlayItems()
            self.createOverlayContextMenu()

            self.img.alphaScrollbar = self.addAlphaScrollbar(
                self.parent.user_ch_name, self.img
            )

    def getOverlayItems(self, channelName):
        imageItem = pg.ImageItem()
        imageItem.setOpacity(0.5)

        lutItem = widgets.myHistogramLUTitem(isViewer=True)

        lutItem.setImageItem(imageItem)
        lutItem.vb.raiseContextMenu = lambda x: None
        initColor = self.overlayRGBs.pop(0)
        self.parent.initColormapOverlayLayerItem(initColor, lutItem)
        lutItem.addOverlayColorButton(initColor, channelName)
        lutItem.initColor = initColor
        lutItem.hide()

        alphaScrollBar = self.addAlphaScrollbar(channelName, imageItem)
        return imageItem, lutItem, alphaScrollBar

    def setMirroredCursorPos(self, x, y):
        if not self.enableMirroredCursor:
            return

        if not self.showMirroredCursorCheckbox.isChecked():
            return

        self.cursor.setData([x], [y])

    def setOverlayColors(self):
        self.overlayRGBs = [
            (255, 255, 0),
            (252, 72, 254),
            (49, 222, 134),
            (22, 108, 27),
        ]
        cmap = matplotlib.colormaps["gist_rainbow"]
        self.overlayRGBs.extend(
            [tuple([round(c * 255) for c in cmap(i)][:3]) for i in np.linspace(0, 1, 8)]
        )

    def setOpacityOverlayLayersItems(self, value, imageItem=None):
        if imageItem is None:
            imageItem = self.sender().imageItem
            alpha = value / self.sender().maximum()
        else:
            alpha = value
        imageItem.setOpacity(alpha)

    def overlay_cb(self, checked):
        if checked:
            if self.posData.ol_data is None:
                selectedChannels = self.askSelectOverlayChannel()
                if selectedChannels is None:
                    self.overlayButton.toggled.disconnect()
                    self.overlayButton.setChecked(False)
                    self.overlayButton.toggled.connect(self.overlay_cb)
                    return
                success = self.parent.loadOverlayData(selectedChannels)
                if not success:
                    return False
                lastChannel = selectedChannels[-1]
                self.checkedOverlayChName = lastChannel
                imageItem = self.overlayLayersItems[lastChannel][0]
                self.setOpacityOverlayLayersItems(0.5, imageItem=imageItem)
                self.img.setOpacity(0.5)
                self.setCheckedOverlayContextMenusActions(selectedChannels)
            else:
                self.checkedOverlayChName = self.parent.imgGrad.checkedChannelname
                selectedChannels = self.parent.checkedOverlayChannels
                self.setCheckedOverlayContextMenusActions(selectedChannels)
            self.setOverlayItemsVisible(self.checkedOverlayChName, True)
        else:
            self.img.setOpacity(1.0)
            self.setOverlayItemsVisible("", False)
            for items in self.overlayLayersItems.values():
                imageItem = items[0]
                imageItem.clear()
        self.update_img()

    def createOverlayContextMenu(self):
        ch_names = [
            ch for ch in self.posData.chNames if ch != self.posData.user_ch_name
        ]
        self.overlayContextMenu = QMenu()
        self.overlayContextMenu.addSeparator()
        self.checkedOverlayChannels = set()
        for chName in ch_names:
            action = QAction(chName, self.overlayContextMenu)
            action.setCheckable(True)
            action.toggled.connect(self.overlayChannelToggled)
            self.overlayContextMenu.addAction(action)

    def setCheckedOverlayContextMenusActions(self, channelNames):
        for action in self.overlayContextMenu.actions():
            if action.text() not in channelNames:
                continue
            action.setChecked(True)
            self.checkedOverlayChannels.add(action.text())

    def overlayChannelToggled(self, checked):
        # Action toggled from overlayButton context menu
        channelName = self.sender().text()
        if checked:
            posData = self.posData
            if channelName not in posData.loadedFluoChannels:
                self.parent.loadOverlayData([channelName], addToExisting=True)
            self.setOverlayItemsVisible(channelName, True)
            self.checkedOverlayChannels.add(channelName)
            self.updateOlColors(None)
        else:
            self.checkedOverlayChannels.remove(channelName)
            imageItem = self.overlayLayersItems[channelName][0]
            imageItem.clear()
            try:
                channelToShow = next(iter(self.checkedOverlayChannels))
                self.setOverlayItemsVisible(channelToShow, True)
            except StopIteration:
                self.setOverlayItemsVisible("", False)
        self.update_img()

    def updateOlColors(self, button):
        lutItem = self.overlayLayersItems[self.checkedOverlayChName][1]
        rgb = lutItem.overlayColorButton.color().getRgb()[:3]
        self.parent.initColormapOverlayLayerItem(rgb, lutItem)
        lutItem.overlayColorButton.setColor(rgb)

    def addAlphaScrollbar(self, channelName, imageItem, alphaScrollBar=None):
        if alphaScrollBar is None:
            alphaScrollBar = QScrollBar(Qt.Horizontal)
        label = QLabel(f"Alpha {channelName}")
        label.setFont(font)
        label.hide()
        alphaScrollBar.imageItem = imageItem
        alphaScrollBar.label = label
        alphaScrollBar.setFixedHeight(self.parent.h)
        alphaScrollBar.hide()
        alphaScrollBar.setMinimum(0)
        alphaScrollBar.setMaximum(40)
        alphaScrollBar.setValue(20)
        alphaScrollBar.setToolTip(
            f"Control the alpha value of the overlaid channel {channelName}.\n"
            "alpha=0 results in NO overlay,\n"
            "alpha=1 results in only fluorescence data visible"
        )
        self.img_Widglayout.addWidget(
            alphaScrollBar.label, 2, 0, alignment=Qt.AlignRight
        )
        self.img_Widglayout.addWidget(alphaScrollBar, 2, 1, 1, 20)
        sp = alphaScrollBar.label.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        alphaScrollBar.label.setSizePolicy(sp)

        sp = alphaScrollBar.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        alphaScrollBar.setSizePolicy(sp)

        alphaScrollBar.valueChanged.connect(self.setOpacityOverlayLayersItems)
        return alphaScrollBar

    def setOverlayItemsVisible(self, channelName, visible):
        if visible:
            self.imgGrad.hide()
            self.img.alphaScrollbar.hide()
            self.img.alphaScrollbar.label.hide()
            try:
                self.graphLayout.removeItem(self.imgGrad)
            except Exception as e:
                pass
            itemsToShow = None
            for name, items in self.overlayLayersItems.items():
                _, lutItem, alphaSB = items
                if name == channelName:
                    itemsToShow = items
                else:
                    lutItem.hide()
                    alphaSB.hide()
                    alphaSB.label.hide()
                    try:
                        self.graphLayout.removeItem(lutItem)
                    except Exception as e:
                        pass

            if itemsToShow is None:
                self.graphLayout.addItem(self.imgGrad, row=1, col=0)
                self.imgGrad.show()
                self.currentLutItem = self.imgGrad
                self.img.alphaScrollbar.show()
                self.img.alphaScrollbar.label.show()
            else:
                _, lutItem, alphaSB = itemsToShow
                lutItem.show()
                alphaSB.show()
                alphaSB.label.show()
                self.currentLutItem = lutItem
                self.graphLayout.addItem(lutItem, row=1, col=0)
        else:
            if self.overlayButton.isChecked():
                self.img.alphaScrollbar.show()
                self.img.alphaScrollbar.label.show()
            else:
                self.img.alphaScrollbar.hide()
                self.img.alphaScrollbar.label.hide()
            for name, items in self.overlayLayersItems.items():
                _, lutItem, alphaSB = items
                lutItem.hide()
                alphaSB.hide()
                alphaSB.label.hide()
                try:
                    self.graphLayout.removeItem(lutItem)
                except Exception as e:
                    pass
            self.graphLayout.addItem(self.imgGrad, row=1, col=0)
            self.imgGrad.show()
            self.currentLutItem = self.imgGrad

    def framesScrollBarMoved(self, frame_n):
        self.frame_i = frame_n - 1
        self.t_label.setText(f"frame n. {self.frame_i + 1}/{self.num_frames}")
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
                self.wcLabel.setText(f"(x={x:.2f}, y={y:.2f}, value={val:.2f})")
            else:
                self.wcLabel.setText(f"")
        except Exception as e:
            self.wcLabel.setText(f"")

        emitHovering = (
            self.enableMirroredCursor and self.showMirroredCursorCheckbox.isChecked()
        )
        if emitHovering:
            if event.isExit():
                x, y = None, None
            else:
                x, y = event.pos()
            self.sigHoveringImage.emit(x, y)
            self.cursor.setData([], [])

    def next_frame(self):
        if self.frame_i < self.num_frames - 1:
            self.frame_i += 1
        else:
            self.frame_i = 0
        self.update_img()

    def prev_frame(self):
        if self.frame_i > 0:
            self.frame_i -= 1
        else:
            self.frame_i = self.num_frames - 1
        self.update_img()

    def skip10ahead_frames(self):
        if self.frame_i < self.num_frames - 10:
            self.frame_i += 10
        else:
            self.frame_i = 0
        self.update_img()

    def skip10back_frames(self):
        if self.frame_i > 9:
            self.frame_i -= 10
        else:
            self.frame_i = self.num_frames - 1
        self.update_img()

    def update_z_slice(self, z):
        if self.posData is None:
            posData = self.parent.data[self.parent.pos_i]
        else:
            posData = self.posData
            idx = (posData.filename, posData.frame_i)
            posData.segmInfo_df.at[idx, "z_slice_used_gui"] = z

        self.z_label.setText(f"z-slice  {z + 1:02}/{posData.SizeZ}")
        self.img.setCurrentZsliceIndex(z)
        self.update_img()

    def getImage(self):
        posData = self.posData
        frame_i = self.frame_i
        if posData.SizeZ > 1:
            idx = (posData.filename, frame_i)
            z = posData.segmInfo_df.at[idx, "z_slice_used_gui"]
            zProjHow = posData.segmInfo_df.at[idx, "which_z_proj_gui"]
            img = posData.img_data[frame_i]
            if zProjHow == "single z-slice":
                self.zSliceScrollBar.setSliderPosition(z)
                self.z_label.setText(f"z-slice  {z + 1:02}/{posData.SizeZ}")
                img = img[z].copy()
            elif zProjHow == "max z-projection":
                img = img.max(axis=0).copy()
            elif zProjHow == "mean z-projection":
                img = img.mean(axis=0).copy()
            elif zProjHow == "median z-proj.":
                img = np.median(img, axis=0).copy()
        else:
            img = posData.img_data[frame_i].copy()
        return img

    def update_img(self):
        self.frameLabel.setText(f"Current frame = {self.frame_i + 1}/{self.num_frames}")
        if self.parent is None:
            img = self.getImage()
        else:
            img = self.parent.getImage(frame_i=self.frame_i, raw=True)

        self.img.setCurrentFrameIndex(self.frame_i)
        self.img.setImage(img)
        self.framesScrollBar.setSliderPosition(self.frame_i + 1)

        if not self.enableOverlay:
            return

        if not self.overlayButton.isChecked():
            return

        self.setOverlayImages(frame_i=self.frame_i)

    def askSelectOverlayChannel(self):
        ch_names = [
            ch for ch in self.posData.chNames if ch != self.posData.user_ch_name
        ]
        selectFluo = widgets.QDialogListbox(
            "Select channel",
            "Select channel names to overlay:\n",
            ch_names,
            multiSelection=True,
            parent=self,
        )
        selectFluo.exec_()
        if selectFluo.cancel:
            return

        return selectFluo.selectedItemsText

    def setOverlayImages(self, frame_i=None):
        posData = self.posData
        for filename in posData.ol_data:
            chName = myutils.get_chname_from_basename(
                filename, posData.basename, remove_ext=False
            )
            if chName not in self.checkedOverlayChannels:
                continue

            imageItem = self.overlayLayersItems[chName][0]
            ol_img = self.parent.getOlImg(filename, frame_i=frame_i)
            imageItem.setImage(ol_img)

    def closeEvent(self, event):
        if self.button_toUncheck is not None:
            self.button_toUncheck.setChecked(False)
        self.sigClosed.emit()

    def show(self, left=None, top=None):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        QMainWindow.show(self)
        try:
            self.framesScrollBar.setFixedHeight(self.parent.h)
        except Exception as e:
            pass
        try:
            self.zSliceScrollBar.setFixedHeight(self.parent.h)
        except Exception as e:
            pass

        try:
            self.img.alphaScrollbar.setFixedHeight(self.parent.h)
        except Exception as e:
            pass
        if left is not None and top is not None:
            self.setGeometry(left, top, 850, 800)


class askStopFrameSegm(QDialog):
    def __init__(self, user_ch_file_paths, user_ch_name, parent=None):
        self.parent = parent
        self.cancel = True

        super().__init__(parent)
        self.setWindowTitle("Enter stop frame")

        self.visualizeWindows = []

        mainLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        # Message
        infoTxt = html_utils.paragraph("""
            Enter a <b>stop frame number</b> when to stop 
            segmentation for each Position loaded:
        """)
        infoLabel = QLabel(infoTxt, self)
        infoLabel.setAlignment(Qt.AlignCenter)
        # padding: top, left, bottom, right
        infoLabel.setStyleSheet("padding:0px 0px 8px 0px;")

        self.dataDict = {}

        exp_path_pos_mapper = path.get_exp_path_pos_foldernames_mapper(
            user_ch_file_paths
        )

        columnsLayout = QHBoxLayout()
        mainScrollArea = widgets.ScrollArea()
        mainScrollAreaWidget = QWidget()
        mainScrollAreaWidget.setLayout(columnsLayout)
        mainScrollArea.setWidget(mainScrollAreaWidget)
        self.mainScrollArea = mainScrollArea

        # Form layout widget
        self.spinBoxes = []
        self.tab_idx = 0
        iter_items = exp_path_pos_mapper.items()
        self.groupboxScrollAreas = []

        for col, (exp_path, pos_folders_files) in enumerate(iter_items):
            groupboxScrollArea = widgets.ScrollArea()
            self.groupboxScrollAreas.append(groupboxScrollArea)
            groupbox = QGroupBox()
            groupbox.setCheckable(False)
            groupbox.setToolTip(exp_path)
            groupboxLayout = QFormLayout()
            groupbox.setLayout(groupboxLayout)
            groupboxScrollArea.setWidget(groupbox)
            columnsLayout.addWidget(groupboxScrollArea)
            pos_folders = pos_folders_files["pos_foldernames"]
            filenames = pos_folders_files["filenames"]
            for i, pos_foldername in enumerate(pos_folders):
                img_filename = filenames[i]
                images_path = os.path.join(exp_path, pos_foldername, "Images")
                img_path = os.path.join(images_path, img_filename)
                spinBox = widgets.mySpinBox()
                spinBox.sigTabEvent.connect(self.keyTabEventSpinbox)
                posData = load.loadData(img_path, user_ch_name, QParent=parent)
                posData.getBasenameAndChNames(qparent=self)
                posData.buildPaths()
                posData.loadOtherFiles(
                    load_segm_data=False,
                    load_metadata=True,
                    loadSegmInfo=True,
                )
                spinBox.setMaximum(posData.SizeT)
                stopFrameNum = posData.readLastUsedStopFrameNumber()
                if stopFrameNum is None:
                    spinBox.setValue(posData.SizeT)
                else:
                    spinBox.setValue(stopFrameNum)
                spinBox.setAlignment(Qt.AlignCenter)
                visualizeButton = widgets.viewPushButton("Visualize")
                visualizeButton.clicked.connect(self.visualize_cb)
                formLabel = QLabel(html_utils.paragraph(f"{pos_foldername}  "))
                layout = QHBoxLayout()
                layout.addWidget(formLabel, alignment=Qt.AlignRight)
                layout.addWidget(spinBox)
                layout.addWidget(visualizeButton)
                self.dataDict[visualizeButton] = (spinBox, posData)
                groupboxLayout.addRow(layout)
                spinBox.idx = i
                self.spinBoxes.append(spinBox)

            fm = QFontMetrics(self.font())
            elidedTitle = fm.elidedText(
                exp_path, Qt.ElideLeft, groupbox.sizeHint().width()
            )
            groupbox.setTitle(elidedTitle)

        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        mainLayout.addWidget(mainScrollArea)

        okButton = widgets.okPushButton("Ok")
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = widgets.cancelPushButton("Cancel")

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        # # self.setModal(True)

    def keyTabEventSpinbox(self, event, sender):
        self.tab_idx += 1
        if self.tab_idx >= len(self.spinBoxes):
            self.tab_idx = 0
        focusSpinbox = self.spinBoxes[self.tab_idx]
        focusSpinbox.setFocus()

    def saveStopFrameNumbers(self):
        for spinBox, posData in self.dataDict.values():
            posData.metadata_df.at["stop_frame_num", "values"] = spinBox.value()
            posData.metadataToCsv()

    def ok_cb(self, event):
        self.cancel = False
        try:
            self.saveStopFrameNumbers()
        except Exception as err:
            printl(traceback.format_exc())
        self.stopFrames = [
            spinBox.value() for spinBox, posData in self.dataDict.values()
        ]
        self.close()

    def closeEvent(self, event):
        for window in self.visualizeWindows:
            window.close()

    def visualize_cb(self, checked=True):
        self.setDisabled(True)
        spinBox, posData = self.dataDict[self.sender()]
        print("Loading image data...")
        posData.loadImgData()
        posData.frame_i = spinBox.value() - 1
        win = plot.imshow(
            posData.img_data, lut="gray", figure_title=posData.relPath, block=False
        )
        self.visualizeWindows.append(win)
        self.setDisabled(False)

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        screenSize = self.screen().size()
        maxWidth = screenSize.width() - 50
        maxHeight = screenSize.height() - 100
        width, height = 0, 0
        for scrollArea in self.groupboxScrollAreas:
            width += scrollArea.minimumWidthNoScrollbar()
            scrollAreaHeight = scrollArea.minimumHeightNoScrollbar()
            if scrollAreaHeight > height:
                height = scrollAreaHeight

        width += 70
        height += self.sizeHint().height() - self.mainScrollArea.sizeHint().height()

        if width > maxWidth:
            width = maxWidth

        if height > maxHeight:
            height = maxHeight

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()
        self.resize(width, height)
        self.move(25, 50)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()


class QLineEditDialog(QDialog):
    def __init__(
        self,
        title="Entry messagebox",
        msg="Entry value",
        defaultTxt="",
        parent=None,
        allowedValues=None,
        warnLastFrame=False,
        isInteger=False,
        isFloat=False,
        stretchEntry=True,
        allowEmpty=True,
        allowedTextEntries=None,
        allowText=False,
        lastVisitedFrame=None,
        allowList=False,
    ):
        QDialog.__init__(self, parent)

        self.loop = None
        self.cancel = True
        self.assignNewID = False
        self.allowedValues = allowedValues
        self.warnLastFrame = warnLastFrame
        self.isFloat = isFloat
        self.allowEmpty = allowEmpty
        self.isInteger = isInteger
        self.allowedTextEntries = allowedTextEntries
        self.allowText = allowText
        self.lastVisitedFrame = lastVisitedFrame
        if allowedValues and warnLastFrame:
            self.maxValue = max(allowedValues)

        self.setWindowTitle(title)

        # Layouts
        mainLayout = QVBoxLayout()
        LineEditLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        # Widgets
        if not msg.startswith("<p"):
            msg = html_utils.paragraph(msg, center=True)
        msg = QLabel(msg)
        msg.setStyleSheet("padding:0px 0px 3px 0px;")

        if isFloat:
            self._type = float
        elif isInteger:
            self._type = int
        else:
            self._type = str

        self.allowList = allowList

        if isFloat and not allowList:
            entryWidget = QDoubleSpinBox()
            if allowedValues is not None:
                _min, _max = min(allowedValues), max(allowedValues)
                entryWidget.setMinimum(_min)
                entryWidget.setMaximum(_max)
            else:
                entryWidget.setMaximum(2147483647)
            if defaultTxt:
                entryWidget.setValue(float(defaultTxt))

        elif isInteger and not allowList:
            entryWidget = QSpinBox()
            if allowedValues is not None:
                _min, _max = min(allowedValues), max(allowedValues)
                entryWidget.setMinimum(_min)
                entryWidget.setMaximum(_max)
            else:
                entryWidget.setMaximum(2147483647)
            if defaultTxt:
                entryWidget.setValue(int(defaultTxt))
        else:
            entryWidget = QLineEdit()
            entryWidget.setText(defaultTxt)
            if not self.allowText:
                entryWidget.textChanged[str].connect(self.onTextChanged)
        entryWidget.setFont(font)
        entryWidget.setAlignment(Qt.AlignCenter)

        self.entryWidget = entryWidget

        if allowedValues is not None:
            notValidLabel = QLabel()
            notValidLabel.setStyleSheet("color: red")
            notValidLabel.setFont(font)
            notValidLabel.setAlignment(Qt.AlignCenter)
            self.notValidLabel = notValidLabel

        okButton = widgets.okPushButton("Ok")
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = widgets.cancelPushButton("Cancel")

        # Events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        # Contents margins
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        # Add widgets to layouts
        LineEditLayout.addWidget(msg, alignment=Qt.AlignCenter)
        if stretchEntry:
            LineEditLayout.addWidget(entryWidget)
        else:
            entryLayout = QHBoxLayout()
            entryLayout.addStretch(1)
            entryLayout.addWidget(entryWidget)
            entryLayout.addStretch(1)
            entryLayout.setStretch(1, 1)
            LineEditLayout.addLayout(entryLayout)
        if allowedValues is not None:
            LineEditLayout.addWidget(notValidLabel, alignment=Qt.AlignCenter)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        self.okButton = okButton
        self.buttonsLayout = buttonsLayout

        # Add layouts
        mainLayout.addLayout(LineEditLayout)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        # self.setModal(True)

    def value(self):
        if self._type == str:
            return self.entryWidget.text()

        if (self.isFloat or self.isInteger) and not self.allowList:
            val = self.entryWidget.value()
        elif not self.allowList:
            val = int(self.entryWidget.text())
        elif self.allowList:
            caster = int if self.isInteger else float
            text = self.entryWidget.text()
            m = re.findall(POSITIVE_FLOAT_REGEX, text)
            val = [caster(val) for val in m]
        return val

    def onTextChanged(self, text):
        # Get inserted char
        idx = self.entryWidget.cursorPosition()
        if idx == 0:
            return

        newChar = text[idx - 1]
        if self.allowList and (newChar == "," or newChar == " "):
            return

        if self._type == str:
            self.entryWidget.setText(text)
            return

        # Allow only integers
        try:
            val = int(newChar)
            if val > np.iinfo(np.uint32).max:
                self.entryWidget.setText(str(np.iinfo(np.uint32).max))
        except Exception as e:
            text = text.replace(newChar, "")
            self.entryWidget.setText(text)
            return

        if self.allowedValues is not None:
            currentVal = self.value()
            if self.allowList:
                currentVal = currentVal[-1]
            if currentVal not in self.allowedValues:
                self.notValidLabel.setText(f"{currentVal} not existing!")
            else:
                self.notValidLabel.setText("")

    def warnValLessLastFrame(self, val):
        msg = widgets.myMessageBox()
        warn_txt = html_utils.paragraph(f"""
            WARNING: saving until a frame number below the last visited
            frame ({self.lastVisitedFrame}) will result in <b>LOSS of information</b>
            about any <b>edit or annotation</b> you did <b>on frames
            {val + 1}-{self.lastVisitedFrame}.</b><br><br>
            Are you sure you want to proceed?
        """)
        msg.warning(
            self,
            "WARNING: Potential loss of information",
            warn_txt,
            buttonsTexts=("Cancel", "Yes, I am sure."),
        )
        return msg.cancel

    def warnValMoreLastVisitedFrame(self, val):
        msg = widgets.myMessageBox()
        warn_txt = html_utils.paragraph(f"""
            The <b>last visited/validated frame is {self.lastVisitedFrame}</b>
            .<br><br>
            Are you sure you want to save until frame n. {val}?<br>
        """)
        msg.warning(
            self,
            "Saving past last visited frame",
            warn_txt,
            buttonsTexts=("Cancel", "Yes, I am sure."),
        )
        return msg.cancel

    def ok_cb(self, event):
        if not self.allowEmpty and not self.entryWidget.text():
            msg = widgets.myMessageBox(showCentered=False, wrapText=False)
            msg.critical(
                self,
                "Empty text",
                html_utils.paragraph("Text entry field <b>cannot be empty</b>"),
            )
            return
        if self.allowedTextEntries is not None:
            if self.entryWidget.text() not in self.allowedTextEntries:
                msg = widgets.myMessageBox(showCentered=False, wrapText=False)
                txt = html_utils.paragraph(
                    f'"{self.entryWidget.text()}" is not a valid entry.<br><br>'
                    "Valid entries are:<br>"
                    f"{html_utils.to_list(self.allowedTextEntries)}"
                )
                msg.critical(self, "Not a valid entry", txt)
                return

        if self.allowedValues:
            if self.notValidLabel.text():
                return

        val = self.value()

        if self.warnLastFrame and self.lastVisitedFrame is not None:
            if val < self.lastVisitedFrame:
                cancel = self.warnValLessLastFrame(val)
                if cancel:
                    return

        if self.lastVisitedFrame is not None:
            if val > self.lastVisitedFrame:
                cancel = self.warnValMoreLastVisitedFrame(val)
                if cancel:
                    return

        self.cancel = False
        try:
            self.EntryID = int(val)
        except Exception as err:
            self.EntryID = val

        self.enteredValue = val
        self.close()

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


class QtSelectItems(QDialog):
    def __init__(
        self,
        title,
        items,
        informativeText,
        CbLabel="Select value:  ",
        parent=None,
        showInFileManagerPath=None,
    ):
        self.cancel = True
        self.selectedItemsText = ""
        self.selectedItemsIdx = None
        self.showInFileManagerPath = showInFileManagerPath
        self.items = items
        super().__init__(parent)
        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()
        topLayout = QHBoxLayout()
        self.topLayout = topLayout
        bottomLayout = QHBoxLayout()

        stretchRow = 0
        if informativeText:
            infoLabel = QLabel(informativeText)
            mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)
            stretchRow = 1

        label = QLabel(CbLabel)
        topLayout.addWidget(label, alignment=Qt.AlignRight)

        combobox = QComboBox(self)
        combobox.addItems(items)
        self.ComboBox = combobox
        topLayout.addWidget(combobox)

        okButton = widgets.okPushButton("Ok")
        cancelButton = widgets.cancelPushButton("Cancel")
        if showInFileManagerPath is not None:
            txt = myutils.get_open_filemaneger_os_string()
            showInFileManagerButton = widgets.showInFileManagerButton(txt)

        bottomLayout.addStretch(1)
        bottomLayout.addWidget(cancelButton)
        bottomLayout.addSpacing(20)
        if showInFileManagerPath is not None:
            bottomLayout.addWidget(showInFileManagerButton)
        bottomLayout.addWidget(okButton)

        multiPosButton = QPushButton("Multiple selection")
        multiPosButton.setCheckable(True)
        self.multiPosButton = multiPosButton
        bottomLayout.addWidget(multiPosButton, alignment=Qt.AlignLeft)

        listBox = widgets.listWidget()
        listBox.addItems(items)
        listBox.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        listBox.setCurrentRow(0)
        listBox.setFont(font)
        topLayout.addWidget(listBox)
        listBox.hide()
        self.ListBox = listBox

        mainLayout.addLayout(topLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(bottomLayout)

        self.setLayout(mainLayout)
        self.mainLayout = mainLayout
        self.topLayout = topLayout

        # self.setModal(True)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        multiPosButton.toggled.connect(self.toggleMultiSelection)
        if showInFileManagerPath is not None:
            showInFileManagerButton.clicked.connect(self.showInFileManager)

        self.setFont(font)

    def setSelectedItems(self, selectedItemsText):
        if self.multiPosButton.isChecked():
            for i in range(self.ListBox.count()):
                item = self.ListBox.item(i)
                if item.text() in selectedItemsText:
                    item.setSelected(True)
        else:
            idx = self.items.index(selectedItemsText[0])
            self.ComboBox.setCurrentIndex(idx)

    def showInFileManager(self):
        selectedTexts, _ = self.getSelectedItems()
        folder = selectedTexts[0].split("(")[0].strip()
        path = os.path.join(self.showInFileManagerPath, folder)
        if os.path.exists(path) and os.path.isdir(path):
            showPath = path
        else:
            showPath = self.showInFileManagerPath
        myutils.showInExplorer(showPath)

    def toggleMultiSelection(self, checked):
        if checked:
            self.multiPosButton.setText("Single selection")
            self.ComboBox.hide()
            self.ListBox.show()
            # Show 10 items
            n = self.ListBox.count()
            if n > 10:
                h = sum([self.ListBox.sizeHintForRow(i) for i in range(10)])
            else:
                h = sum([self.ListBox.sizeHintForRow(i) for i in range(n)])
            self.ListBox.setMinimumHeight(h + 5)
            self.ListBox.setFocusPolicy(Qt.StrongFocus)
            self.ListBox.setFocus()
            self.ListBox.setCurrentRow(0)
            self.mainLayout.setStretchFactor(self.topLayout, 2)
        else:
            self.multiPosButton.setText("Multiple selection")
            self.ListBox.hide()
            self.ComboBox.show()
            self.resize(self.width(), self.singleSelectionHeight)

    def getSelectedItems(self):
        if self.multiPosButton.isChecked():
            selectedItems = self.ListBox.selectedItems()
            selectedItemsText = [item.text() for item in selectedItems]
            selectedItemsText = natsorted(selectedItemsText)
            selectedItemsIdx = [self.items.index(txt) for txt in selectedItemsText]
        else:
            selectedItemsText = [self.ComboBox.currentText()]
            selectedItemsIdx = [self.ComboBox.currentIndex()]
        return selectedItemsText, selectedItemsIdx

    def ok_cb(self, event):
        self.cancel = False
        self.selectedItemsText, self.selectedItemsIdx = self.getSelectedItems()
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()
        self.singleSelectionHeight = self.height()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()


class SelectSegmFileDialog(QDialog):
    def __init__(
        self,
        images_ls,
        parent_path,
        parent=None,
        addNewFileButton=False,
        basename="",
        infoText=None,
        fileType="segmentation",
        allowMultipleSelection=False,
        custom_first=None,
    ):
        self.cancel = True
        self.selectedItemText = ""
        self.selectedItemIdx = None
        self.removeOthers = False
        self.okAllPos = False
        self.newSegmEndName = None
        self.allowMultipleSelection = allowMultipleSelection
        self.basename = basename
        images_ls = sorted(images_ls, key=len)
        if custom_first is not None:
            images_ls.remove(custom_first)
            images_ls.insert(0, custom_first)

        # Remove the 'segm_' part to allow filenameDialog to check if
        # a new file is existing (since we only ask for the part after
        # 'segm_')
        self.existingEndNames = [
            n.replace("segm", "", 1).replace("_", "", 1) for n in images_ls
        ]

        self.images_ls = images_ls
        self.parent_path = parent_path
        super().__init__(parent)

        informativeText = html_utils.paragraph(f"""
            The loaded Position folders already contains
            <b>{len(self.existingEndNames)} {fileType} masks</b><br>
        """)

        self.setWindowTitle(f"{fileType.capitalize()} files detected")
        is_win = sys.platform.startswith("win")

        mainLayout = QVBoxLayout()
        infoLayout = QHBoxLayout()
        selectionLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        # Standard Qt Question icon
        label = QLabel()
        standardIcon = getattr(QStyle, "SP_MessageBoxQuestion")
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        label.setPixmap(pixmap)
        infoLayout.addWidget(label)

        infoLabel = QLabel(informativeText)
        infoLayout.addWidget(infoLabel)
        infoLayout.addStretch(1)
        mainLayout.addLayout(infoLayout)

        if infoText is None:
            infoText = f"Select which {fileType} file to load:"

        questionText = html_utils.paragraph(infoText)
        label = QLabel(questionText)
        listWidget = widgets.listWidget()
        listWidget.addItems(images_ls)
        listWidget.setCurrentRow(0)
        listWidget.itemDoubleClicked.connect(self.listDoubleClicked)
        if allowMultipleSelection:
            listWidget.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection
            )
        self.items = list(images_ls)
        self.listWidget = listWidget

        okButton = widgets.okPushButton(" Load selected ")
        txt = "Reveal in Finder..." if is_mac else "Show in Explorer..."
        showInFileManagerButton = widgets.showInFileManagerButton(txt)
        cancelButton = widgets.cancelPushButton(" Cancel ")

        if addNewFileButton:
            newFileButton = widgets.newFilePushButton("New file...")

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addWidget(showInFileManagerButton)
        buttonsLayout.addSpacing(20)
        if addNewFileButton:
            buttonsLayout.addWidget(newFileButton)
        buttonsLayout.addWidget(okButton)

        buttonsLayout.setContentsMargins(0, 10, 0, 10)

        selectionLayout.addWidget(label, 0, 1, alignment=Qt.AlignLeft)
        selectionLayout.addWidget(listWidget, 1, 1)
        selectionLayout.setColumnStretch(0, 0)
        selectionLayout.setColumnStretch(1, 1)
        selectionLayout.setColumnStretch(2, 0)
        selectionLayout.addLayout(buttonsLayout, 2, 1)

        mainLayout.addLayout(selectionLayout)
        self.setLayout(mainLayout)

        self.okButton = okButton

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        if addNewFileButton:
            newFileButton.clicked.connect(self.newFile_cb)
        cancelButton.clicked.connect(self.close)
        showInFileManagerButton.clicked.connect(self.showInFileManager)

    def listDoubleClicked(self, item):
        self.ok_cb()

    def showInFileManager(self, checked=True):
        myutils.showInExplorer(self.parent_path)

    def newFile_cb(self):
        win = filenameDialog(
            basename=f"{self.basename}segm",
            hintText="Insert a <b>filename</b> for the segmentation file:",
            existingNames=self.existingEndNames,
        )
        win.exec_()
        if win.cancel:
            return
        self.cancel = False
        self.newSegmEndName = win.entryText
        self.close()

    def setSelectedItemFromText(self, itemText):
        for i in range(self.listWidget.count()):
            if self.listWidget.item(i).text() == itemText:
                self.listWidget.setCurrentRow(i)
                break

    def ok_cb(self, event=None):
        self.cancel = False
        try:
            self.selectedItemText = self.listWidget.selectedItems()[0].text()
        except IndexError:
            self.cancel = True
            self.close()
            return
        self.selectedItemIdx = self.items.index(self.selectedItemText)
        self.selectedItemTexts = [
            selectedItem.text() for selectedItem in self.listWidget.selectedItems()
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


class QDialogPbar(QDialog):
    def __init__(self, title="Progress", infoTxt="", parent=None):
        self.workerFinished = False
        self.aborted = False
        self.clickCount = 0
        super().__init__(parent)

        abort_text = "Option+Command+C" if is_mac else "Ctrl+Alt+C"
        self.abort_text = abort_text

        self.setWindowTitle(f"{title} ({abort_text} to abort)")
        self.setWindowFlags(Qt.Window)

        mainLayout = QVBoxLayout()
        pBarLayout = QGridLayout()

        if infoTxt:
            infoLabel = QLabel(infoTxt)
            mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        self.progressLabel = QLabel()

        self.QPbar = widgets.ProgressBar(self)
        pBarLayout.addWidget(self.QPbar, 0, 0)
        self.ETA_label = QLabel("NDh:NDm:NDs")
        pBarLayout.addWidget(self.ETA_label, 0, 1)

        self.metricsQPbar = widgets.ProgressBar(self)
        self.metricsQPbar.setValue(0)
        pBarLayout.addWidget(self.metricsQPbar, 1, 0)

        # pBarLayout.setColumnStretch(2, 1)

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
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(f"""
            Aborting with <code>{self.abort_text}</code> to abort 
            is <b>not safe</b>.<br><br>
            The system status cannot be predicted and
            it will <b>require a restart</b>.<br><br>
            Are you sure you want to abort?
        """)
        yesButton, noButton = msg.critical(
            self, "Are you sure you want to abort?", txt, buttonsTexts=("Yes", "No")
        )
        return msg.clickedButton == yesButton

    def abort(self):
        self.clickCount += 1
        self.aborted = True
        if self.clickCount > 3:
            self.workerFinished = True
            self.close()

    def closeEvent(self, event):
        if not self.workerFinished:
            event.ignore()


class pgTestWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout()

        self.graphLayout = pg.GraphicsLayoutWidget()
        self.ax1 = pg.PlotItem()
        self.ax1.setAspectLocked(True)
        self.graphLayout.addItem(self.ax1)

        layout.addWidget(self.graphLayout)

        self.setLayout(layout)


def get_existing_directory(allow_images_path=True, **kwargs):
    while True:
        folder_path = qtpy.compat.getexistingdirectory(**kwargs)
        if not folder_path:
            return

        if allow_images_path:
            return folder_path

        pos_folderpath = os.path.dirname(folder_path)
        is_images_folder = (
            folder_path.endswith("Images")
            and os.path.basename(pos_folderpath).startswith("Position_")
            and os.path.isdir(folder_path)
        )
        if not is_images_folder:
            return folder_path

        txt = html_utils.paragraph(
            "You <b>cannot save</b> to the <code>Images</code> folder "
            "because it is reserved to files that start with the same "
            "basename.<br><br>Thank you for your patience!"
        )
        msg = widgets.myMessageBox()
        msg.warning(kwargs["parent"], "Cannot save here", txt)


class SetCustomLevelsLut(QBaseDialog):
    sigLevelsChanged = Signal(object)

    def __init__(
        self,
        init_min_value=None,
        init_max_value=None,
        minimum_min_value=0,
        maximum_max_value=None,
        parent=None,
    ):
        super().__init__(parent=parent)

        self.cancel = True

        self.setWindowTitle("Custom LUT levels")

        layout = QVBoxLayout()

        self.minLevelSlider = widgets.sliderWithSpinBox(
            title="Minimum",
            title_loc="top",
        )
        self.minLevelSlider.setMinimum(minimum_min_value)

        if init_min_value is not None:
            self.minLevelSlider.setValue(init_min_value)

        layout.addWidget(self.minLevelSlider)

        self.maxLevelSlider = widgets.sliderWithSpinBox(
            title="Maximum",
            title_loc="top",
        )
        self.maxLevelSlider.setMinimum(minimum_min_value)
        if init_max_value is not None:
            self.maxLevelSlider.setValue(init_max_value)

        if maximum_max_value is not None:
            self.maxLevelSlider.setMaximum(maximum_max_value)
            self.minLevelSlider.setMaximum(maximum_max_value)

        layout.addWidget(self.maxLevelSlider)

        self.minLevelSlider.sigValueChange.connect(self.emitLevelsChanged)
        self.maxLevelSlider.sigValueChange.connect(self.emitLevelsChanged)

        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)

        self.setLayout(layout)

    def sizeHint(self):
        heightHint = super().sizeHint().height()
        widthHint = super().sizeHint().width() * 2
        return QSize(widthHint, heightHint)

    def levels(self):
        levels = (self.minLevelSlider.value(), self.maxLevelSlider.value())
        return levels

    def emitLevelsChanged(self, value):
        self.sigLevelsChanged.emit(self.levels())

    def ok_cb(self):
        self.cancel = False
        self.selectedLevels = self.levels()
        self.close()


class QTreeDialog(QBaseDialog):
    def __init__(
        self,
        items: List[Tuple[str]],
        headerLabels: List[str] = None,
        parent=None,
        infoText="Select item",
        title="Select item",
        path_to_browse=None,
        additional_buttons=None,
    ):
        self.cancel = True
        super().__init__(parent)

        self.setWindowTitle(title)

        mainLayout = QVBoxLayout()

        infoLabel = QLabel(html_utils.paragraph(infoText))

        self.treeWidget = widgets.TreeWidget()
        if headerLabels is not None:
            self.treeWidget.setHeaderLabels(headerLabels)
        else:
            self.treeWidget.setHeaderHidden(True)

        for row, texts in enumerate(items):
            item = widgets.TreeWidgetItem(self.treeWidget)
            for i, text in enumerate(texts):
                item.setText(i, text)
            self.treeWidget.addTopLevelItem(item)

        self.treeWidget.resizeColumnToContents(0)
        self.treeWidget.resizeColumnToContents(1)

        # self.treeWidget.header().setStretchLastSection(False)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        if path_to_browse is not None:
            browseButton = widgets.showInFileManagerButton(setDefaultText=True)
            browseButton.setPathToBrowse(path_to_browse)
            buttonsLayout.insertWidget(3, browseButton)

        if additional_buttons is not None:
            for btn in additional_buttons:
                buttonsLayout.insertWidget(3, btn)

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(infoLabel)
        mainLayout.addWidget(self.treeWidget)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def show(self, block=False):
        w = self.sizeHint().width()
        h = self.sizeHint().height()
        self.resize(int(w * 1.3), h)
        super().show(block=block)

    def ok_cb(self):
        self.clickedButton = self.sender()
        self.cancel = False
        self.selectedItem = self.treeWidget.currentItem()
        self.selectedText = self.selectedItem.text(0)
        self.close()

# Sibling imports (deferred to avoid import cycles)
from .metadata import (
    filenameDialog,
)

