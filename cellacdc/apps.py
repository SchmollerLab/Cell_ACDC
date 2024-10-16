import os
import sys
import re
from typing import Literal
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
import tkinter as tk
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
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import time

import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')

from qtpy import QtCore
from qtpy.QtGui import (
    QIcon, QFontMetrics, QKeySequence, QFont, QRegularExpressionValidator, 
    QCursor, QKeyEvent, QPixmap, QFont, QPalette, QMouseEvent, QColor
)
from qtpy.QtCore import (
    Qt, QSize, QEvent, Signal, QEventLoop, QTimer, QRegularExpression
)
from qtpy.QtWidgets import (
    QFileDialog, QApplication, QMainWindow, QMenu, QLabel, QToolBar,
    QScrollBar, QWidget, QVBoxLayout, QLineEdit, QPushButton,
    QHBoxLayout, QDialog, QFormLayout, QListWidget, QAbstractItemView,
    QButtonGroup, QCheckBox, QSizePolicy, QComboBox, QSlider, QGridLayout,
    QSpinBox, QToolButton, QTableView, QTextBrowser, QDoubleSpinBox,
    QScrollArea, QFrame, QProgressBar, QGroupBox, QRadioButton,
    QDockWidget, QMessageBox, QStyle, QPlainTextEdit, QSpacerItem,
    QTreeWidget, QTreeWidgetItem, QTextEdit, QSplashScreen, QAction,
    QListWidgetItem, QActionGroup, QHeaderView
)
import qtpy.compat

from . import exception_handler
from . import load, prompts, core, measurements, html_utils
from . import is_mac, is_win, is_linux, settings_folderpath, config
from . import is_conda_env, pytorch_commands
from . import qrc_resources, printl
from . import colors
from . import issues_url
from . import myutils
from . import qutils
from . import _palettes
from . import base_cca_dict
from . import widgets
from . import user_profile_path
from . import features
from . import _core
from . import types
from . import plot
from . import urls
from .acdc_regex import float_regex
from . import _base_widgets

POSITIVE_FLOAT_REGEX = float_regex(allow_negative=False)
PRE_PROCESSING_STEPS = [
    'Adjust Brightness/Contrast',
    'Smooth (gaussian filter)', 
    'Sharpen (difference of gaussians filter)'
]

TREEWIDGET_STYLESHEET = _palettes.TreeWidgetStyleSheet()
LISTWIDGET_STYLESHEET = _palettes.ListWidgetStyleSheet()
BACKGROUND_RGBA = _palettes.get_disabled_colors()['Button']

font = QFont()
font.setPixelSize(12)
italicFont = QFont()
italicFont.setPixelSize(12)
italicFont.setItalic(True)

class AcdcSPlashScreen(QSplashScreen):
    def __init__(self):
        super().__init__()
        # self.showMessage('Test', color=Qt.white)
        self.setPixmap(QPixmap(':logo.png'))
    
    def mousePressEvent(self, a0: QMouseEvent) -> None:
        pass


def addCustomModelMessages(QParent=None):
    modelFilePath = None
    msg = widgets.myMessageBox(showCentered=False, wrapText=False)
    txt = html_utils.paragraph("""
    Do you <b>already have</b> the <code>acdcSegment.py</code> file for your code 
    or do you <b>need instructions</b> on how to set-up your custom model?<br>
    """)
    infoButton = widgets.infoPushButton(' I need instructions')
    browseButton = widgets.browseFileButton(' I have the model, let me select it')
    msg.information(
        QParent, 'Add custom model', txt, 
        buttonsTexts=('Cancel', infoButton, browseButton),
        showDialog=False
    )
    browseButton.clicked.disconnect()
    browseButton.clicked.connect(msg.buttonCallBack)
    msg.exec_()
    if msg.cancel:
        return
    if msg.clickedButton == infoButton:           
        txt, models_path = myutils.get_add_custom_model_instructions()
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.addShowInFileManagerButton(models_path, txt='Open models folder...')
        msg.information(
            QParent, 'Custom model instructions', txt, buttonsTexts=('Ok',)
        )
    else:
        homePath = pathlib.Path.home()
        modelFilePath = QFileDialog.getOpenFileName(
            QParent, 'Select the acdcSegment.py file of your model',
            str(homePath), 'acdcSegment.py file (*.py);;All files (*)'
        )[0]
        if not modelFilePath:
            return
    
    return modelFilePath

class QBaseDialog(_base_widgets.QBaseDialog):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
class customAnnotationDialog(QDialog):
    sigDeleteSelecAnnot = Signal(object)

    def __init__(self, savedCustomAnnot, parent=None, state=None):
        self.cancel = True
        self.loop = None
        self.clickedButton = None
        self.savedCustomAnnot = savedCustomAnnot

        self.internalNames = measurements.get_all_acdc_df_colnames()

        super().__init__(parent)

        self.setWindowTitle('Custom annotation')
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        layout = widgets.FormLayout()

        row = 0
        typeCombobox = QComboBox()
        typeCombobox.addItems([
            'Single time-point',
            'Multiple time-points',
            'Multiple values class'
        ])
        if state is not None:
            typeCombobox.setCurrentText(state['type'])
        self.typeCombobox = typeCombobox
        body_txt = ("""
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
        """)
        typeInfoTxt = (f'{html_utils.paragraph(body_txt)}')
        self.typeWidget = widgets.formWidget(
            typeCombobox, addInfoButton=True, labelTextLeft='Type: ',
            parent=self, infoTxt=typeInfoTxt
        )
        layout.addFormWidget(self.typeWidget, row=row)
        typeCombobox.currentTextChanged.connect(self.warnType)

        row += 1
        nameInfoTxt = ("""
        <b>Name of the column</b> that will be saved in the <code>acdc_output.csv</code>
        file.<br><br>
        Valid charachters are letters and numbers separate by underscore
        or dash only.<br><br>
        Additionally, some names are <b>reserved</b> because they are used
        by Cell-ACDC for standard measurements.<br><br>
        Internally reserved names:
        """)
        self.nameInfoTxt = (f'{html_utils.paragraph(nameInfoTxt)}')
        self.nameWidget = widgets.formWidget(
            widgets.alphaNumericLineEdit(), addInfoButton=True,
            labelTextLeft='Name: ', parent=self, infoTxt=self.nameInfoTxt
        )
        self.nameWidget.infoButton.disconnect()
        self.nameWidget.infoButton.clicked.connect(self.showNameInfo)
        if state is not None:
            self.nameWidget.widget.setText(state['name'])
        self.nameWidget.widget.textChanged.connect(self.checkName)
        layout.addFormWidget(self.nameWidget, row=row)

        row += 1
        self.nameInfoLabel = QLabel()
        layout.addWidget(
            self.nameInfoLabel, row, 0, 1, 2, alignment=Qt.AlignCenter
        )

        row += 1
        spacing = QSpacerItem(10, 10)
        layout.addItem(spacing, row, 0)

        row += 1
        symbolInfoTxt = ("""
        <b>Symbol</b> that will be drawn on the annotated cell at
        the requested time frame.
        """)
        symbolInfoTxt = (f'{html_utils.paragraph(symbolInfoTxt)}')
        self.symbolWidget = widgets.formWidget(
            widgets.pgScatterSymbolsCombobox(), addInfoButton=True,
            labelTextLeft='Symbol: ', parent=self, infoTxt=symbolInfoTxt
        )
        if state is not None:
            self.symbolWidget.widget.setCurrentText(state['symbol'])
        layout.addFormWidget(self.symbolWidget, row=row)

        row += 1
        shortcutInfoTxt = ("""
        <b>Shortcut</b> that you can use to <b>activate/deactivate</b> annotation
        of this event.<br><br> Leave empty if you don't need a shortcut.
        """)
        shortcutInfoTxt = (f'{html_utils.paragraph(shortcutInfoTxt)}')
        self.shortcutWidget = widgets.formWidget(
            widgets.ShortcutLineEdit(), addInfoButton=True,
            labelTextLeft='Shortcut: ', parent=self, infoTxt=shortcutInfoTxt
        )
        if state is not None:
            self.shortcutWidget.widget.setText(state['shortcut'])
        layout.addFormWidget(self.shortcutWidget, row=row)

        row += 1
        descInfoTxt = ("""
        <b>Description</b> will be used as the <b>tool tip</b> that will be
        displayed when you hover with th mouse cursor on the toolbar button
        specific for this annotation
        """)
        descInfoTxt = (f'{html_utils.paragraph(descInfoTxt)}')
        self.descWidget = widgets.formWidget(
            QPlainTextEdit(), addInfoButton=True,
            labelTextLeft='Description: ', parent=self, infoTxt=descInfoTxt
        )
        if state is not None:
            self.descWidget.widget.setPlainText(state['description'])
        layout.addFormWidget(self.descWidget, row=row)

        row += 1
        optionsGroupBox = QGroupBox('Additional options')
        optionsLayout = QGridLayout()
        toggle = widgets.Toggle()
        toggle.setChecked(True)
        self.keepActiveToggle = toggle
        toggleLabel = QLabel('Keep tool active after using it: ')
        colorButtonLabel = QLabel('Symbol color: ')
        self.hideAnnotTooggle = widgets.Toggle()
        self.hideAnnotTooggle.setChecked(True)
        hideAnnotTooggleLabel = QLabel(
            'Hide annotation when button is not active: '
        )
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
            '<i>NOTE: you can change these options later with<br>'
            '<b>RIGHT-click</b> on the associated left-side <b>toolbar button<b>.</i>'
        )
        noteLabel = QLabel(html_utils.paragraph(noteText, font_size='11px'))
        layout.addWidget(noteLabel, row, 1, 1, 3)

        buttonsLayout = QHBoxLayout()

        self.loadSavedAnnotButton = widgets.OpenFilePushButton(
            '  Load annotation...  '
        )
        if not savedCustomAnnot:
            self.loadSavedAnnotButton.setDisabled(True)
        self.okButton = widgets.okPushButton('  Ok  ')
        cancelButton = widgets.cancelPushButton('Cancel')

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

        noteTxt = ("""
        Custom annotations will be <b>saved in the <code>acdc_output.csv</code></b><br>
        file as a column with the name you write in the field <b>Name</b><br>
        """)
        noteTxt = (f'{html_utils.paragraph(noteTxt, font_size="15px")}')
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
            txt = 'Name cannot be empty'
            self.nameInfoLabel.setText(
                html_utils.paragraph(
                    txt, font_size='11px', font_color='red'
                )
            )
            return
        for name in self.internalNames:
            if name.find(text) != -1:
                txt = (
                    f'"{text}" cannot be part of the name, '
                    'because <b>reserved<b>.'
                )
                self.nameInfoLabel.setText(
                    html_utils.paragraph(
                        txt, font_size='11px', font_color='red'
                    )
                )
                break
        else:
            self.nameInfoLabel.setText('')

    def loadSavedAnnot(self):
        items = list(self.savedCustomAnnot.keys())
        self.selectAnnotWin = widgets.QDialogListbox(
            'Load annotation parameters',
            'Select annotation to load:', items,
            additionalButtons=('Delete selected annnotations', ),
            parent=self, multiSelection=False
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
        self.typeCombobox.setCurrentText(selectedAnnot['type'])
        self.nameWidget.widget.setText(selectedAnnot['name'])
        self.symbolWidget.widget.setCurrentText(selectedAnnot['symbol'])
        self.shortcutWidget.widget.setText(selectedAnnot['shortcut'])
        self.descWidget.widget.setPlainText(selectedAnnot['description'])
        self.colorButton.setColor(selectedAnnot['symbolColor'])
        keySequence = widgets.macShortcutToWindows(selectedAnnot['shortcut'])
        if keySequence:
            self.shortcutWidget.widget.keySequence = QKeySequence(keySequence)

    def warnNoItemsSelected(self):
        msg = widgets.myMessageBox(parent=self)
        msg.setIcon(iconName='SP_MessageBoxWarning')
        msg.setWindowTitle('Delete annotation?')
        msg.addText('You didn\'t select any annotation!')
        msg.addButton('  Ok  ')
        msg.exec_()

    def deleteSelectedAnnot(self):
        msg = widgets.myMessageBox(parent=self)
        msg.setIcon(iconName='SP_MessageBoxWarning')
        msg.setWindowTitle('Delete annotation?')
        msg.addText('Are you sure you want to delete the selected annotations?')
        msg.addButton('Yes')
        cancelButton = msg.addButton(' Cancel ')
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
        self.colorButton.colorDialog.setWindowFlags(
            Qt.Window | Qt.WindowStaysOnTopHint
        )
        self.colorButton.colorDialog.open()
        w = self.width()
        left = self.pos().x()
        colorDialogTop = self.colorButton.colorDialog.pos().y()
        self.colorButton.colorDialog.move(w+left+10, colorDialogTop)

    def warnType(self, currentText):
        if currentText == 'Single time-point':
            return

        self.typeCombobox.setCurrentIndex(0)

        txt = ("""
        Unfortunately, the only annotation type that is available so far is
        <b>Single time-point</b>.<br><br>
        We are working on implementing the other types too, so stay tuned!<br><br>
        Thank you for your patience!
        """)
        txt = (f'{html_utils.paragraph(txt)}')
        msg = widgets.myMessageBox()
        msg.setIcon(iconName='SP_MessageBoxWarning')
        msg.setWindowTitle(f'Feature not implemented yet')
        msg.addText(txt)
        msg.addButton('   Ok   ')
        msg.exec_()

    def showOptionsInfo(self):
        info = ("""
        <b>Keep tool active after using it</b>: Choose whether the tool
        should stay active or not after annotating.<br><br>
        <b>Hide annotation when button is not active</b>: Choose whether
        annotation on the cell/object should be visible only if the
        button is active or also when it is not active.<br>
        <i>NOTE: annotations are <b>always stored</b> no matter whether
        they are visible or not.</i><br><br>
        <b>Symbol color</b>: Choose color of the symbol that will be used
        to label annotated cell/object.
        """)
        info = (f'{html_utils.paragraph(info)}')
        msg = widgets.myMessageBox()
        msg.setIcon()
        msg.setWindowTitle(f'Additional options info')
        msg.addText(info)
        msg.addButton('   Ok   ')
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
            self, 'Annotation Name info', self.nameInfoTxt,
            widgets=listView
        )

    def closeEvent(self, event):
        if self.clickedButton is None or self.clickedButton==self.cancelButton:
            # cancel button or closed with 'x' button
            self.cancel = True
            return

        if self.clickedButton==self.okButton and not self.nameWidget.widget.text():
            msg = QMessageBox()
            msg.critical(
                self, 'Empty name', 'The name cannot be empty!', msg.Ok
            )
            event.ignore()
            self.cancel = True
            return

        if self.clickedButton==self.okButton and self.nameInfoLabel.text():
            msg = widgets.myMessageBox()
            listView = widgets.listWidget(msg)
            listView.addItems(self.internalNames)
            listView.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
            name = self.nameWidget.widget.text()
            txt = (
                f'"{name}" cannot be part of the name, '
                'because it is <b>reserved</b> for standard measurements '
                'saved by Cell-ACDC.<br><br>'
                'Internally reserved names:'
            )
            msg.critical(
                self, 'Not a valid name', html_utils.paragraph(txt),
                widgets=listView
            )
            event.ignore()
            self.cancel = True
            return

        self.toolTip = (
            f'Name: {self.nameWidget.widget.text()}\n\n'
            f'Type: {self.typeWidget.widget.currentText()}\n\n'
            f'Usage: activate the button and RIGHT-CLICK on cell to annotate\n\n'
            f'Description: {self.descWidget.widget.toPlainText()}\n\n'
            f'SHORTCUT: "{self.shortcutWidget.widget.text()}"'
        )

        symbol = self.symbolWidget.widget.currentText()
        self.symbol = re.findall(r"\'(.+)\'", symbol)[0]

        self.state = {
            'type': self.typeWidget.widget.currentText(),
            'name': self.nameWidget.widget.text(),
            'symbol':  self.symbolWidget.widget.currentText(),
            'shortcut': self.shortcutWidget.widget.text(),
            'description': self.descWidget.widget.toPlainText(),
            'keepActive': self.keepActiveToggle.isChecked(),
            'isHideChecked': self.hideAnnotTooggle.isChecked(),
            'symbolColor': self.colorButton.color()
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

        self.setTitle('Points appearance')

        layout = widgets.FormLayout()

        '----------------------------------------------------------------------' 
        row = 0
        symbolInfoTxt = ("""
            <b>Symbol</b> used to draw the points.
        """)
        symbolInfoTxt = (f'{html_utils.paragraph(symbolInfoTxt)}')
        self.symbolWidget = widgets.formWidget(
            widgets.pgScatterSymbolsCombobox(), addInfoButton=True,
            labelTextLeft='Symbol: ', parent=self, infoTxt=symbolInfoTxt,
            stretchWidget=False
        )
        layout.addFormWidget(self.symbolWidget, row=row)
        '----------------------------------------------------------------------' 

        '----------------------------------------------------------------------' 
        row += 1
        self.colorButton = widgets.myColorButton(color=(255, 0, 0))
        self.colorWidget = widgets.formWidget(
            self.colorButton, stretchWidget=True,
            labelTextLeft='Colour: ', parent=self
        )
        layout.addFormWidget(self.colorWidget, align=Qt.AlignLeft, row=row)
        self.colorButton.clicked.disconnect()
        self.colorButton.clicked.connect(self.selectColor)
        '----------------------------------------------------------------------' 

        '----------------------------------------------------------------------' 
        row += 1
        self.sizeSpinBox = widgets.SpinBox()
        self.sizeSpinBox.setValue(5)
        self.sizeWidget = widgets.formWidget(
            self.sizeSpinBox, stretchWidget=True,
            labelTextLeft='Size: ', parent=self
        )
        layout.addFormWidget(self.sizeWidget, row=row)
        '----------------------------------------------------------------------' 
        
        '----------------------------------------------------------------------' 
        row += 1
        self.zHeightSpinBox = widgets.OddSpinBox()
        self.zHeightSpinBox.setValue(1)
        self.zHeightSpinBox.setMinimum(1)
        self.zHeightWidget = widgets.formWidget(
            self.zHeightSpinBox, stretchWidget=True,
            labelTextLeft='Z- height: ', parent=self
        )
        layout.addFormWidget(self.zHeightWidget, row=row)
        '----------------------------------------------------------------------'

        '----------------------------------------------------------------------' 
        row += 1
        shortcutInfoTxt = ("""
        <b>Shortcut</b> that you can use to <b>hide/show</b> points.
        """)
        shortcutInfoTxt = (f'{html_utils.paragraph(shortcutInfoTxt)}')
        self.shortcutWidget = widgets.formWidget(
            widgets.ShortcutLineEdit(), addInfoButton=True,
            labelTextLeft='Shortcut: ', parent=self, infoTxt=shortcutInfoTxt
        )
        layout.addFormWidget(self.shortcutWidget, row=row)
        '----------------------------------------------------------------------'

        self.setLayout(layout)
    
    def restoreState(self, state):
        self.shortcutWidget.widget.setText(state['shortcut'])
        self.colorButton.setColor(state['color'])
        self.symbolWidget.widget.setCurrentText(state['symbol'])
        self.sizeSpinBox.setValue(state['pointSize'])
        self.zHeightSpinBox.setValue(state['zHeight'])
    
    def selectColor(self):
        color = self.colorButton.color()
        self.colorButton.origColor = color
        self.colorButton.colorDialog.setCurrentColor(color)
        self.colorButton.colorDialog.setWindowFlags(
            Qt.Window | Qt.WindowStaysOnTopHint
        )
        self.colorButton.colorDialog.open()    
        w = self.width()
        left = self.pos().x()
        colorDialogTop = self.colorButton.colorDialog.pos().y()
        self.colorButton.colorDialog.move(w+left+10, colorDialogTop)
    
    def state(self):
        r,g,b,a = self.colorButton.color().getRgb()
        _state = {
            'symbol': self.symbolWidget.widget.currentText(), 
            'color': (r,g,b),
            'pointSize': self.sizeSpinBox.value(),
            'zHeight': self.zHeightSpinBox.value(),
            'shortcut': self.shortcutWidget.widget.text()
        }
        return _state

class AddPointsLayerDialog(QBaseDialog):
    sigClosed = Signal()
    sigCriticalReadTable = Signal(str)
    sigLoadedTable = Signal(object)
    sigCheckClickEntryTableEndnameExists = Signal(str)

    def __init__(self, channelNames=None, imagesPath='', SizeT=1, parent=None):
        self.cancel = True
        super().__init__(parent)

        self._parent = parent

        self.imagesPath = imagesPath

        self.setWindowTitle('Add points layer')
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        mainLayout = QVBoxLayout()

        typeGroupbox = QGroupBox('Points to draw')
        typeLayout = QGridLayout()
        typeGroupbox.setLayout(typeLayout)
        typeLayout.addItem(QSpacerItem(10,1), 0, 0)
        typeLayout.setColumnStretch(0, 0)
        typeLayout.setColumnStretch(2, 1)
        vSpacing = 15

        '----------------------------------------------------------------------'
        row = 0
        self.centroidsRadiobutton = QRadioButton('Centroids')
        typeLayout.addWidget(self.centroidsRadiobutton, row, 0, 1, 2)
        self.centroidsRadiobutton.setChecked(True)

        row += 1
        typeLayout.addItem(QSpacerItem(1,vSpacing), row, 0)
        '----------------------------------------------------------------------'

        '----------------------------------------------------------------------'   
        row += 1
        self.weightedCentroidsRadiobutton = QRadioButton('Weighted centroids')
        typeLayout.addWidget(self.weightedCentroidsRadiobutton, row, 0, 1, 2)

        row += 1
        label = QLabel('Weighing channel: ')
        label.setEnabled(False)
        typeLayout.addWidget(label, row, 1)
        self.channelNameForWeightedCentr = widgets.QCenteredComboBox()
        if channelNames:
            self.channelNameForWeightedCentr.addItems(channelNames)
        self.channelNameForWeightedCentr.setDisabled(True)
        typeLayout.addWidget(self.channelNameForWeightedCentr, row, 2)

        self.weightedCentroidsRadiobutton.toggled.connect(label.setEnabled)
        self.weightedCentroidsRadiobutton.toggled.connect(
            self.channelNameForWeightedCentr.setEnabled
        )

        row += 1
        typeLayout.addItem(QSpacerItem(1,vSpacing), row, 0)
        '----------------------------------------------------------------------'

        '----------------------------------------------------------------------'
        row += 1
        self.fromTableRadiobutton = QRadioButton('From table')
        typeLayout.addWidget(self.fromTableRadiobutton, row, 0, 1, 2)
        self.fromTableRadiobutton.widgets = []
        
        row += 1
        self.tablePath = widgets.ElidingLineEdit()
        self.tablePath.label = QLabel('Table file path: ')
        typeLayout.addWidget(self.tablePath.label, row, 1)
        typeLayout.addWidget(self.tablePath, row, 2)
        self.fromTableRadiobutton.widgets.append(self.tablePath)

        browseButton = widgets.browseFileButton(
            start_dir=imagesPath, ext={'Table': ['.csv', '.h5']}
        )
        typeLayout.addWidget(browseButton, row, 3)
        browseButton.sigPathSelected.connect(self.tablePathSelected)
        self.browseTableButton = browseButton
        self.fromTableRadiobutton.widgets.append(browseButton)

        row += 1
        self.xColName = widgets.QCenteredComboBox()
        self.xColName.addItem('None')
        self.xColName.label = QLabel('X coord. column: ')
        typeLayout.addWidget(self.xColName.label, row, 1)
        typeLayout.addWidget(self.xColName, row, 2)
        self.xColName.currentTextChanged.connect(self.checkColNameX)
        self.fromTableRadiobutton.widgets.append(self.xColName)

        row += 1
        self.yColName = widgets.QCenteredComboBox()
        self.yColName.addItem('None')
        self.yColName.label = QLabel('Y coord. column: ')
        typeLayout.addWidget(self.yColName.label, row, 1)
        typeLayout.addWidget(self.yColName, row, 2)
        self.yColName.currentTextChanged.connect(self.checkColNameY)
        self.fromTableRadiobutton.widgets.append(self.yColName)

        row += 1
        self.zColName = widgets.QCenteredComboBox()
        self.zColName.addItem('None')
        self.zColName.label = QLabel('Z coord. column: ')
        typeLayout.addWidget(self.zColName.label, row, 1)
        typeLayout.addWidget(self.zColName, row, 2)
        self.zColName.currentTextChanged.connect(self.checkColNameZ)
        self.fromTableRadiobutton.widgets.append(self.zColName)

        row += 1
        self.tColName = widgets.QCenteredComboBox()
        self.tColName.addItem('None')
        self.tColName.label = QLabel('Frame index column: ')
        typeLayout.addWidget(self.tColName.label, row, 1)
        typeLayout.addWidget(self.tColName, row, 2)
        self.fromTableRadiobutton.widgets.append(self.tColName)

        if SizeT == 1:
            self.tColName.clear()
            self.tColName.addItem('None')
            self.tColName.label.setVisible(False)
            self.tColName.setVisible(False)
        
        self.fromTableRadiobutton.toggled.connect(self.enableRadioButtonWidgets)
        self.enableRadioButtonWidgets(False, sender=self.fromTableRadiobutton)
        '----------------------------------------------------------------------'

        '----------------------------------------------------------------------'
        row += 1
        self.manualEntryRadiobutton = QRadioButton('Manual entry')
        typeLayout.addWidget(self.manualEntryRadiobutton, row, 0, 1, 2)
        self.manualEntryRadiobutton.widgets = []
        
        row += 1
        self.manualXspinbox = widgets.NumericCommaLineEdit()
        self.manualXspinbox.label = QLabel('X coords: ')
        typeLayout.addWidget(self.manualXspinbox.label, row, 1)
        typeLayout.addWidget(self.manualXspinbox, row, 2)
        self.manualEntryRadiobutton.widgets.append(self.manualXspinbox)

        row += 1
        self.manualYspinbox = widgets.NumericCommaLineEdit()
        self.manualYspinbox.label = QLabel('Y coords: ')
        typeLayout.addWidget(self.manualYspinbox.label, row, 1)
        typeLayout.addWidget(self.manualYspinbox, row, 2)
        self.manualEntryRadiobutton.widgets.append(self.manualYspinbox)

        row += 1
        self.manualZspinbox = widgets.NumericCommaLineEdit()
        self.manualZspinbox.label = QLabel('Z coords: ')
        typeLayout.addWidget(self.manualZspinbox.label, row, 1)
        typeLayout.addWidget(self.manualZspinbox, row, 2)
        self.manualEntryRadiobutton.widgets.append(self.manualZspinbox)

        row += 1
        self.manualTspinbox = widgets.NumericCommaLineEdit()
        self.manualTspinbox.label = QLabel('Frame numbers: ')
        typeLayout.addWidget(self.manualTspinbox.label, row, 1)
        typeLayout.addWidget(self.manualTspinbox, row, 2)
        self.manualEntryRadiobutton.widgets.append(self.manualTspinbox)

        if SizeT == 1:
            self.manualTspinbox.setVisible(False)
            self.manualTspinbox.label.setVisible(False)
        
        self.manualEntryRadiobutton.toggled.connect(self.enableRadioButtonWidgets)
        self.enableRadioButtonWidgets(False, sender=self.manualEntryRadiobutton)
        
        '----------------------------------------------------------------------'
        self.clickEntryIsLoadedDf = None
        row += 1
        self.clickEntryRadiobutton = QRadioButton('Add points with mouse clicks')
        typeLayout.addWidget(self.clickEntryRadiobutton, row, 0, 1, 2) 
        self.clickEntryRadiobutton.widgets = [] 
        
        row += 1
        self.snapToMaxToggle = widgets.Toggle()
        self.snapToMaxToggle.label = QLabel('Snap to closest maximum: ')
        typeLayout.addWidget(self.snapToMaxToggle.label, row, 1)
        typeLayout.addWidget(
            self.snapToMaxToggle, row, 2, alignment=Qt.AlignCenter
        )
        self.snapToMaxInfoButton = widgets.infoPushButton()
        typeLayout.addWidget(self.snapToMaxInfoButton, row, 3)
        
        self.snapToMaxInfoButton.clicked.connect(self.showSnapToMaxButton)
        self.clickEntryRadiobutton.widgets.append(self.snapToMaxToggle)
        self.clickEntryRadiobutton.widgets.append(self.snapToMaxInfoButton)
        
        row += 1
        self.autoPilotToggle = widgets.Toggle()
        self.autoPilotToggle.label = QLabel('Use auto-pilot: ')
        typeLayout.addWidget(self.autoPilotToggle.label, row, 1)
        typeLayout.addWidget(
            self.autoPilotToggle, row, 2, alignment=Qt.AlignCenter
        )
        self.autoPilotInfoButton = widgets.infoPushButton()
        typeLayout.addWidget(self.autoPilotInfoButton, row, 3)
        
        self.autoPilotInfoButton.clicked.connect(self.showAutoPilotInfo)
        self.clickEntryRadiobutton.widgets.append(self.autoPilotToggle)
        self.clickEntryRadiobutton.widgets.append(self.autoPilotInfoButton)
        
        row += 1
        self.clickEntryTableEndname = widgets.alphaNumericLineEdit()
        self.clickEntryTableEndname.setText('points_added_by_clicking')
        self.clickEntryTableEndname.setAlignment(Qt.AlignCenter)
        self.clickEntryTableEndname.label = QLabel('Table endname: ')
        loadButton = widgets.browseFileButton(
            start_dir=imagesPath, ext={'CSV': '.csv'})
        typeLayout.addWidget(loadButton, row, 3)
        loadButton.sigPathSelected.connect(self.loadClickEntryTable)
        self.loadButton = loadButton
        self.clickEntryLoadTableButton = loadButton
        typeLayout.addWidget(self.clickEntryTableEndname.label, row, 1)
        typeLayout.addWidget(self.clickEntryTableEndname, row, 2)
        self.clickEntryRadiobutton.widgets.append(self.clickEntryTableEndname)
        self.clickEntryTableEndname.editingFinished.connect(
            self.emitCheckClickEntryTableEndnameExists
        )
        
        row += 1
        instructionsText = html_utils.paragraph(
            '<br><i>Left-click</i> to annotate a new point with a new id.<br>'
            '<i>Righ-click</i> to annotate a point with the same id<br>'
            '<i>Click</i> on point to delete it', font_size='11px'
        )
        self.instructionsLabel = QLabel(instructionsText)
        self.instructionsLabel.label = QLabel('Instructions')
        typeLayout.addWidget(self.instructionsLabel.label, row, 1)
        typeLayout.addWidget(self.instructionsLabel, row, 2)
        self.clickEntryRadiobutton.widgets.append(self.instructionsLabel)
        
        self.clickEntryRadiobutton.toggled.connect(self.enableRadioButtonWidgets)
        self.clickEntryRadiobutton.toggled.connect(
            self.emitCheckClickEntryTableEndnameExists
        )
        self.enableRadioButtonWidgets(False, sender=self.clickEntryRadiobutton)
        
        '======================================================================'

        self.appearanceGroupbox = _PointsLayerAppearanceGroupbox()
        self.appearanceGroupbox.sizeSpinBox.setValue(3)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(typeGroupbox)
        mainLayout.addSpacing(20)
        _layout = QHBoxLayout()
        _layout.addWidget(self.appearanceGroupbox)
        _layout.addStretch(1)
        mainLayout.addLayout(_layout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        self.setFont(font)
    
    def emitCheckClickEntryTableEndnameExists(self, *args, **kwargs):
        if not self.clickEntryRadiobutton.isChecked():
            return
        self.clickEntryIsLoadedDf = None
        tableEndName = self.clickEntryTableEndname.text()
        self.sigCheckClickEntryTableEndnameExists.emit(tableEndName)
    
    def loadClickEntryTable(self, csv_path):
        self.clickEntryIsLoadedDf = True
        filename = os.path.basename(csv_path)
        filename, ext = os.path.splitext(filename)
        self.clickEntryTableEndname.setText(filename)
        self.loadButton.confirmAction()
        
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
        msg.information(self, 'Auto-pilot info', txt)
    
    def showSnapToMaxButton(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            With <bSnap to closest maximum</b> mode active, Cell-ACDC will 
            <b>automatically add the point</b><br>
            to the closest maximum within the point footprint (defined in 
            the appearance settings).
        """)
        msg.information(self, 'Snap to closest maximum info', txt)
    
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
        if 'x' in df.columns:
            self.xColName.setCurrentText('x')
        
        if 'y' in df.columns:
            self.yColName.setCurrentText('y')
        
        if 'z' in df.columns:
            self.zColName.setCurrentText('z')
        
        if 'frame_i' in df.columns:
            self.tColName.setCurrentText('frame_i')
    
    def tablePathSelected(self, path):
        self.tablePath.setText(path)
        try:
            df = self._readTable(path)
            self.xColName.addItems(df.columns)
            self.yColName.addItems(df.columns)
            self.zColName.addItems(df.columns)
            self.tColName.addItems(df.columns)
            self.tryAutoFillColNames(df)
            self.sigLoadedTable.emit(df)
            self.browseTableButton.confirmAction()
        except Exception as e:
            traceback_format = traceback.format_exc()
            self.sigCriticalReadTable.emit(traceback_format)
            self.criticalReadTable(path, traceback_format)
            self.tablePath.setText('')
        
    
    def criticalLenMismatchManualEntry(self):
        txt = html_utils.paragraph(f"""
            X coords and Y coords must have the <b>same length</b>.
        """)
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.critical(self, f'X and Y have different length', txt)
        
    def criticalColNameIsNone(self, axis):        
        txt = html_utils.paragraph(f"""
            The "{axis.upper()} coord. column" <b>cannot be "None"</b>
        """)
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.critical(self, f'{axis.upper()} coord. is None', txt)
    
    def criticalReadTable(self, path, traceback_format):
        txt = html_utils.paragraph(f"""
            Something went <b>wrong when reading the table</b> from the 
            following path:<br><br>
            <code>{path}</code><br><br>
            See the <b>error message below</b>.
        """)
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        detailsText = traceback_format
        msg.critical(
            self, 'Error when reading table', txt, detailsText=detailsText)

    def criticalEmptyTablePath(self):
        txt = html_utils.paragraph(f"""
            The table file path <b>cannot be empty</b>.
        """)
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.critical(self, 'Table file path is empty', txt)   

    def state(self):
        _state = self.appearanceGroupbox.state() 
        return _state

    def _checkSelectedColName(self, colName, label):
        labelsToCheck = ['z', 'y', 'x']
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
            self, 'Check column name', txt, 
            buttonsTexts=('Cancel', 'No, let me correct it', 'Yes, I am')
        )
        if msg.cancel or msg.clickedButton == noButton:
            return False
        return True
    
    def checkColNameX(self, text):
        accepted = self._checkSelectedColName(text, 'x')
        if accepted: 
            return
        self.xColName.setCurrentText('None')
    
    def checkColNameY(self, text):
        accepted = self._checkSelectedColName(text, 'y')
        if accepted: 
            return
        self.yColName.setCurrentText('None')
    
    def checkColNameZ(self, text):
        accepted = self._checkSelectedColName(text, 'z')
        if accepted: 
            return
        self.zColName.setCurrentText('None')

    def ok_cb(self):
        self.pointsData = {}
        self.loadedDfInfo = None
        self.weighingChannel = ''
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
                    'filepath': tablePath,
                    't': tColName, 
                    'z': zColName, 
                    'y': yColName, 
                    'x': xColName
                }
                
                self._df_to_pointsData(
                    df, tColName, zColName, yColName, xColName
                )
                    
            except Exception as e:
                traceback_format = traceback.format_exc()
                self.sigCriticalReadTable.emit(traceback_format)
                self.criticalReadTable(tablePath, traceback_format)
                return
            
            if self.xColName.currentText() == 'None':
                self.criticalColNameIsNone('x')
                return
            if self.yColName.currentText() == 'None':
                self.criticalColNameIsNone('y')
                return
            
            self.layerType = os.path.basename(self.tablePath.text())
            self.layerTypeIdx = 2
        elif self.centroidsRadiobutton.isChecked():
            self.layerType = 'Centroids'
            self.layerTypeIdx = 0
        elif self.weightedCentroidsRadiobutton.isChecked():
            channel = self.channelNameForWeightedCentr.currentText()
            self.weighingChannel = channel
            self.layerType = f'Centroids weighted by channel {channel}'
            self.layerTypeIdx = 1
        elif self.manualEntryRadiobutton.isChecked():
            xx = self.manualXspinbox.values()
            yy = self.manualYspinbox.values()
            if len(xx) != len(yy):
                self.criticalLenMismatchManualEntry()
                return
            zz = self.manualZspinbox.values()
            tt = [t+1 for t in self.manualTspinbox.values()]
            df = pd.DataFrame({'x': xx, 'y': yy, 'id': np.arange(1, len(xx)+1)})
            if tt:
                df['t'] = tt
                tCol = 't'
            else:
                tCol = 'None'
            if zz:
                df['z'] = zz
                zCol = 'z'
            else:
                zCol = 'None'
            
            self._df_to_pointsData(df, tCol, zCol, 'y', 'x')
            
            self.layerType = 'Manual entry'
            self.layerTypeIdx = 3
        elif self.clickEntryRadiobutton.isChecked():
            self.layerType = ('Click to annotate point')
            self.description = (
                'Left-click to add a point, click on point to delete it.\n'
                'With auto-pilot you can navigate through object with Up/Down arrows.'
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
        self.resize(int(self.width()*1.25), self.height())
        if self._parent is None:
            screen = self.screen()
        else:
            screen = self._parent.screen()
        screenWidth = screen.size().width()
        screenHeight = screen.size().height()
        screenLeft = screen.geometry().x()
        screenTop = screen.geometry().y()
        w, h = self.width(), self.height()
        left = int(screenLeft + screenWidth/2 - w/2)
        top = int(screenTop + screenHeight/2 - h/2)
        self.move(left, top)


class EditPointsLayerAppearanceDialog(QBaseDialog):
    sigClosed = Signal()

    def __init__(self, parent=None):
        self.cancel = True
        super().__init__(parent)

        self._parent = parent

        self.setWindowTitle('Custom annotation')
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

class filenameDialog(QDialog):
    def __init__(
            self, ext='.npz', basename='', title='Insert file name',
            hintText='', existingNames='', parent=None, allowEmpty=True,
            helpText='', defaultEntry='', resizeOnShow=True,
            additionalButtons=None
        ):
        self.cancel = True
        super().__init__(parent)

        self.resizeOnShow = resizeOnShow

        if hintText.find('segmentation') != -1:
            helpText = ("""
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
            """)

        self.allowEmpty = allowEmpty
        self.basename = basename
        if ext.find('.') == -1:
            ext = f'.{ext}'
        self.ext = ext

        self.setWindowTitle(title)
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        entryLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        hintLabel = QLabel(hintText)

        basenameLabel = QLabel(basename)

        self.lineEdit = QLineEdit()
        self.lineEdit.setAlignment(Qt.AlignCenter)
        self.lineEdit.setText(defaultEntry)

        extLabel = QLabel(ext)

        self.filenameLabel = QLabel()
        self.filenameLabel.setText(f'{basename}{ext}')

        entryLayout.addWidget(basenameLabel, 0, 1)
        entryLayout.addWidget(self.lineEdit, 0, 2)
        entryLayout.addWidget(extLabel, 0, 3)
        entryLayout.addWidget(
            self.filenameLabel, 1, 1, 1, 3, alignment=Qt.AlignCenter
        )
        entryLayout.setColumnStretch(0, 1)
        entryLayout.setColumnStretch(4, 1)

        okButton = widgets.okPushButton('Ok')
        cancelButton = widgets.cancelPushButton('Cancel')
        self.okButton = okButton

        buttonsLayout.addStretch()
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        if helpText:
            helpButton = widgets.helpPushButton('Help...')
            helpButton.clicked.connect(partial(self.showHelp, helpText))
            buttonsLayout.addWidget(helpButton)
        if additionalButtons is not None:
            for button in additionalButtons:
                buttonsLayout.addWidget(button)
        buttonsLayout.addWidget(okButton)

        cancelButton.clicked.connect(self.close)
        okButton.clicked.connect(self.ok_cb)
        self.lineEdit.textChanged.connect(self.updateFilename)
        
        self.existingNames = []
        if existingNames:
            self.existingNames = existingNames
            # self.lineEdit.editingFinished.connect(self.checkExistingNames)

        layout.addWidget(hintLabel)
        layout.addSpacing(20)
        layout.addLayout(entryLayout)
        layout.addStretch(1)
        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)

        self.setLayout(layout)
        self.setFont(font)

        if defaultEntry:
            self.updateFilename(defaultEntry)
    
    def showHelp(self, text):
        text = html_utils.paragraph(text)
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, 'Filename help', text)

    def _text(self):
        _text = self.lineEdit.text().replace(' ', '_')
        _text = self.lineEdit.text().replace('.', '_')
        return _text

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
            'The following file<br><br>'
            f'<code>{filename}</code><br><br>'
            'is <b>already existing</b>.<br><br>'
            'Do you want to <b>overwrite</b> the existing file?'
        )
        noButton, yesButton = msg.warning(
            self, 'File name existing', txt, buttonsTexts=('No', 'Yes')
        )
        return msg.clickedButton == yesButton


    def updateFilename(self, text):
        if not text:
            self.filenameLabel.setText(f'{self.basename}{self.ext}')
        else:
            text = text.replace(' ', '_')
            if self.basename:
                if self.basename.endswith('_'):
                    self.filenameLabel.setText(f'{self.basename}{text}{self.ext}')
                else:
                    self.filenameLabel.setText(f'{self.basename}_{text}{self.ext}')
            else:
                self.filenameLabel.setText(f'{text}{self.ext}')

    def ok_cb(self, checked=True):
        valid = self.checkExistingNames()
        if not valid:
            return
        
        if not self.allowEmpty and not self._text():
            msg = widgets.myMessageBox()
            msg.critical(
                self, 'Empty text', 
                html_utils.paragraph('Text entry field <b>cannot be empty</b>')
            )
            return
            
        self.filename = self.filenameLabel.text()
        self.entryText = self._text()
        self.cancel = False
        self.close()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        if self.resizeOnShow:
            self.lineEdit.setMinimumWidth(self.lineEdit.width()*2)
        self.okButton.setDefault(True)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()
        

class wandToleranceWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.slider = widgets.sliderWithSpinBox(title='Tolerance')
        self.slider.setMaximum(255)
        self.slider._layout.setColumnStretch(2, 21)

        self.setLayout(self.slider.layout)

class TrackSubCellObjectsDialog(QBaseDialog):
    def __init__(self, basename='', parent=None):
        self.cancel = True
        super().__init__(parent=parent)
    
        self.setWindowTitle('Track sub-cellular objects parameters')

        mainLayout = QVBoxLayout()
        entriesLayout = widgets.FormLayout()

        row = 0
        infoTxt = html_utils.paragraph("""
            Select <b>behaviour with untracked objects</b>:<br><br>
            NOTE: this utility <b>always create new files</b>.
            Original segmentation masks <br>are not modified</b>.
        """)
        options = (
            'Delete sub-cellular objects that do not belong to any cell',
            'Delete cells that do not have any sub-cellular object',
            'Delete both cells and sub-cellular objects without an assignment',
            'Only track the objects and keep all the non-tracked objects'
        )
        combobox = widgets.QCenteredComboBox()
        combobox.addItems(options)
        self.optionsWidget = widgets.formWidget(
            combobox, addInfoButton=True, labelTextLeft='Tracking mode: ',
            infoTxt=infoTxt
        )
        entriesLayout.addFormWidget(self.optionsWidget, row=row)
        
        row += 1
        infoTxt = html_utils.paragraph("""
            Re-label sub-cellular objects before assigning them to the cell.<br><br> 
            Activate this option if you have <b>merged sub-cellular objects</b>   
            that must be separated, or the segmentation is a <b>boolean mask</b> 
            (i.e., semantic segmentation).
        """)
        self.relabelSubObjLab = widgets.formWidget(
            widgets.Toggle(), addInfoButton=True, stretchWidget=False,
            labelTextLeft='Re-label sub-cellular objects before tracking: ', 
            infoTxt=infoTxt
        )
        entriesLayout.addFormWidget(self.relabelSubObjLab, row=row)

        row += 1
        IoAtext = html_utils.paragraph("""
            Enter a <b>minimum percentage (0-1) of the sub-cellular object's area</b><br>
            that MUST overlap with the parent cell to be considered belonging to a cell:
        """)
        spinbox = widgets.CenteredDoubleSpinbox()
        spinbox.setMaximum(1)
        spinbox.setValue(0.5)
        spinbox.setSingleStep(0.1)
        self.IoAwidget = widgets.formWidget(
            spinbox, addInfoButton=True, labelTextLeft='IoA threshold: ',
            infoTxt=IoAtext
        )
        entriesLayout.addFormWidget(self.IoAwidget, row=row)

        row += 1
        infoTxt = html_utils.paragraph("""
            The third segmentation file is the result of <b>subtracting the 
            sub-cellular objects from the parent objects</b><br><br>
            This is useful if, for example, you need to compute measurements 
            only from the cytoplasm (i.e., the sub-cellular object is the nucleus).
        """)
        self.createThirdSegmWidget = widgets.formWidget(
            widgets.Toggle(), addInfoButton=True, stretchWidget=False,
            labelTextLeft='Create third segmentation: ', infoTxt=infoTxt
        )
        entriesLayout.addFormWidget(self.createThirdSegmWidget, row=row)

        row += 1
        infoTxt = html_utils.paragraph("""
            Text to append at the end of the third segmentation file.<br><br>
            The third segmentation file is the result of <b>subtracting the 
            sub-cellular objects from the parent objects</b><br><br>
            This is useful if, for example, you need to compute measurements 
            only from the cytoplasm (i.e., the sub-cellular object is the nucleus).
        """)
        lineEdit = widgets.alphaNumericLineEdit()
        lineEdit.setText('difference')
        lineEdit.setAlignment(Qt.AlignCenter)
        self.appendTextWidget = widgets.formWidget(
            lineEdit, addInfoButton=True, labelTextLeft='Text to append: ', 
            infoTxt=infoTxt
        )
        entriesLayout.addFormWidget(self.appendTextWidget, row=row)
        self.appendTextWidget.setDisabled(True)
        

        self.createThirdSegmWidget.widget.toggled.connect(
            self.createThirdSegmToggled
        )

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addLayout(entriesLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
        self.setFont(font)
    
    def createThirdSegmToggled(self, checked):
        self.appendTextWidget.setDisabled(not checked)
    
    def ok_cb(self):
        self.cancel = False
        if self.createThirdSegmWidget.widget.isChecked():
            if not self.appendTextWidget.widget.text():
                msg = widgets.myMessageBox(showCentered=False, wrapText=False)
                txt = html_utils.paragraph(
                    'When creating the third segmentation file, '
                    '<b>the name to append cannot be empty!</b>'
                )
                msg.critical(self, 'Empty name', txt)
                return
                
        self.trackSubCellObjParams = {
            'how': self.optionsWidget.widget.currentText(),
            'IoA': self.IoAwidget.widget.value(),
            'createThirdSegm': self.createThirdSegmWidget.widget.isChecked(),
            'relabelSubObjLab': self.relabelSubObjLab.widget.isChecked(),
            'thirdSegmAppendedText': self.appendTextWidget.widget.text()
        }
        self.close()

class SetMeasurementsDialog(QBaseDialog):
    sigClosed = Signal()
    sigCancel = Signal()
    sigRestart = Signal()

    def __init__(
            self, loadedChNames, notLoadedChNames, isZstack, isSegm3D,
            favourite_funcs=None, parent=None, allPos_acdc_df_cols=None,
            acdc_df_path=None, posData=None, addCombineMetricCallback=None,
            allPosData=None, is_concat=False, isSingleSelection=False,
            state=None
        ):
        super().__init__(parent=parent)
        
        self.checkBoxedGroup = QButtonGroup()
        self.checkBoxedGroup.setExclusive(isSingleSelection)

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        self.cancel = True

        self.delExistingCols = False
        self.okClicked = False
        self.is_concat = is_concat
        self.allPos_acdc_df_cols = allPos_acdc_df_cols
        self.acdc_df_path = acdc_df_path
        self.allPosData = allPosData
        self.doNotWarn = False

        self.setWindowTitle('Set measurements')
        # self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        
        searchLayout = QHBoxLayout()
        
        searchLineEdit = widgets.SearchLineEdit()
        searchLayout.addStretch(5)
        searchLayout.addWidget(searchLineEdit)
        searchLayout.setStretch(1, 3)
        
        groupsLayout = QGridLayout()
        self.groupsLayout = groupsLayout
        buttonsLayout = QHBoxLayout()

        self.chNameGroupboxes = []
        self.all_metrics = []

        col = 0
        for col, chName in enumerate(loadedChNames):
            channelGBox = widgets.channelMetricsQGBox(
                isZstack, chName, isSegm3D, favourite_funcs=favourite_funcs,
                posData=posData, is_concat=is_concat
            )
            channelGBox.chName = chName
            groupsLayout.addWidget(channelGBox, 0, col, 3, 1)
            self.chNameGroupboxes.append(channelGBox)
            channelGBox.sigDelClicked.connect(self.delMixedChannelCombineMetric)
            channelGBox.sigCheckboxToggled.connect(self.channelCheckboxToggled)
            groupsLayout.setColumnStretch(col, 5)
            self.all_metrics.extend([c.text() for c in channelGBox.checkBoxes])

        current_col = col+1
        for col, chName in enumerate(notLoadedChNames):
            channelGBox = widgets.channelMetricsQGBox(
                isZstack, chName, isSegm3D, favourite_funcs=favourite_funcs,
                posData=posData, is_concat=is_concat
            )
            channelGBox.setChecked(False)
            channelGBox.chName = chName
            groupsLayout.addWidget(channelGBox, 0, current_col, 3, 1)
            self.chNameGroupboxes.append(channelGBox)
            groupsLayout.setColumnStretch(current_col, 5)
            channelGBox.sigDelClicked.connect(self.delMixedChannelCombineMetric)
            channelGBox.sigCheckboxToggled.connect(self.channelCheckboxToggled)
            current_col += 1
            self.all_metrics.extend([c.text() for c in channelGBox.checkBoxes])

        current_col += 1

        if posData is None:
            isTimelapse = False
        else:
            isTimelapse = posData.SizeT>1
        size_metrics_desc = measurements.get_size_metrics_desc(
            isSegm3D, isTimelapse
        )
        if not isSegm3D:
            size_metrics_desc = {
                key:val for key,val in size_metrics_desc.items()
                if not key.endswith('_3D')
            }
        sizeMetricsQGBox = widgets._metricsQGBox(
            size_metrics_desc, 'Physical measurements',
            favourite_funcs=favourite_funcs, isZstack=isZstack
        )
        self.all_metrics.extend([c.text() for c in sizeMetricsQGBox.checkBoxes])
        self.sizeMetricsQGBox = sizeMetricsQGBox
        for sizeCheckbox in sizeMetricsQGBox.checkBoxes:
            sizeCheckbox.toggled.connect(self.sizeMetricToggled)
        groupsLayout.addWidget(sizeMetricsQGBox, 0, current_col)
        groupsLayout.setRowStretch(0, 1)
        groupsLayout.setColumnStretch(current_col, 3)

        props_info_txt = measurements.get_props_info_txt()
        props_names = measurements.get_props_names()
        rp_desc = {prop_name:props_info_txt for prop_name in props_names}
        regionPropsQGBox = widgets._metricsQGBox(
            rp_desc, 'Morphological properties',
            favourite_funcs=favourite_funcs, isZstack=isZstack
        )
        self.regionPropsQGBox = regionPropsQGBox
        for rpCheckbox in regionPropsQGBox.checkBoxes:
            rpCheckbox.toggled.connect(self.rpMetricToggled)
        groupsLayout.addWidget(regionPropsQGBox, 1, current_col)
        groupsLayout.setRowStretch(1, 2)
        self.all_metrics.extend([c.text() for c in regionPropsQGBox.checkBoxes])

        desc, equations = measurements.combine_mixed_channels_desc(
            isSegm3D=isSegm3D, posData=posData, available_cols=self.all_metrics
        )
        self.mixedChannelsCombineMetricsQGBox = None
        if desc:
            self.mixedChannelsCombineMetricsQGBox = widgets._metricsQGBox(
                desc, 'Mixed channels combined measurements',
                favourite_funcs=favourite_funcs, isZstack=isZstack,
                equations=equations, addDelButton=True
            )
            self.mixedChannelsCombineMetricsQGBox.sigDelClicked.connect(
                self.delMixedChannelCombineMetric
            )
            groupsLayout.addWidget(
                self.mixedChannelsCombineMetricsQGBox, 2, current_col
            )
            groupsLayout.setRowStretch(1, 1)
            if not self.is_concat:
                self.setDisabledMetricsRequestedForCombined(False)
                self.mixedChannelsCombineMetricsQGBox.toggled.connect(
                    self.setDisabledMetricsRequestedForCombined
                )
                for combCheckbox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
                    combCheckbox.toggled.connect(
                        self.setDisabledMetricsRequestedForCombined
                    )
            else:
                for combCheckbox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
                    combCheckbox.toggled.connect(
                        self.mixedChannelsMetricToggled
                    )

        self.numberCols = current_col

        okButton = widgets.okPushButton('   Ok   ')
        cancelButton = widgets.cancelPushButton('Cancel')
        if addCombineMetricCallback is not None:
            addCombineMetricButton = widgets.addPushButton(
                'Add combined measurement...'
            )
            addCombineMetricButton.clicked.connect(addCombineMetricCallback)
        self.okButton = okButton

        loadLastSelButton = widgets.reloadPushButton('Load last selection...')
        self.deselectAllButton = QPushButton('Deselect all')
        self.deselectAllButton.setIcon(QIcon(':deselect_all.svg'))

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        if addCombineMetricCallback is not None:
            buttonsLayout.addWidget(addCombineMetricButton)
        buttonsLayout.addWidget(self.deselectAllButton)
        buttonsLayout.addWidget(loadLastSelButton)
        buttonsLayout.addWidget(okButton)

        self.okButton = okButton

        layout.addLayout(searchLayout)
        layout.addSpacing(10)
        layout.addLayout(groupsLayout)
        layout.addLayout(buttonsLayout)

        self.setLayout(layout)
        
        if state is not None:
            self.setState(state)

        searchLineEdit.textEdited.connect(self.searchAndHighlight)
        self.deselectAllButton.clicked.connect(self.deselectAll)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        loadLastSelButton.clicked.connect(self.loadLastSelection)
        
        self.addCheckboxesToGroup()

        for channelGBox in self.chNameGroupboxes:
            for checkbox in channelGBox.checkBoxes:
                self.channelCheckboxToggled(checkbox)
    
    def allMetricsDict(self):
        all_metrics = {
            'standard': {}, 
            'regionprop': [],
            'size': [],
            'mixed_channels': []
        }
        for chNameGroupbox in self.chNameGroupboxes:
            channel_name = chNameGroupbox.chName
            for checkBox in chNameGroupbox.checkBoxes:
                if channel_name not in all_metrics['standard']:
                    all_metrics['standard'][channel_name] = []
                all_metrics['standard'][channel_name].append(checkBox.text())
        
        for checkBox in self.regionPropsQGBox.checkBoxes:
            all_metrics['regionprop'].append(checkBox.text())
        
        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            all_metrics['size'].append(checkBox.text())
        
        if self.mixedChannelsCombineMetricsQGBox is None:
            return
        
        checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
        for checkBox in checkBoxes:
            all_metrics['mixed_channels'].append(checkBox.text())
        
        return all_metrics
    
    def searchAndHighlight(self, text):
        for chNameGroupbox in self.chNameGroupboxes:
            for groupbox in chNameGroupbox.groupboxes:
                groupbox.highlightCheckboxesFromSearchText(text)
        
        self.regionPropsQGBox.highlightCheckboxesFromSearchText(text)
        self.sizeMetricsQGBox.highlightCheckboxesFromSearchText(text)
        
        if self.mixedChannelsCombineMetricsQGBox is None:
            return
        
        self.mixedChannelsCombineMetricsQGBox.highlightCheckboxesFromSearchText(
            text
        )
    
    def selectedMetricNameAndGroup(self):
        for chNameGroupbox in self.chNameGroupboxes:
            for checkBox in chNameGroupbox.checkBoxes:
                if checkBox.isChecked():
                    return checkBox.text(), {'standard': chNameGroupbox.chName}
        
        for checkBox in self.regionPropsQGBox.checkBoxes:
            if checkBox.isChecked():
                return checkBox.text(), 'regionprop'
        
        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            if checkBox.isChecked():
                return checkBox.text(), 'size'
        
        if self.mixedChannelsCombineMetricsQGBox is None:
            return
        
        checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
        for checkBox in checkBoxes:
            if checkBox.isChecked():
                return checkBox.text(), 'mixed_channels'
    
    def selectedMetricGroup(self):
        for chNameGroupbox in self.chNameGroupboxes:
             for checkBox in chNameGroupbox.checkBoxes:
                if checkBox.isChecked():
                    return checkBox.text()
        
        for checkBox in self.regionPropsQGBox.checkBoxes:
            if checkBox.isChecked():
                return checkBox.text()
        
        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            if checkBox.isChecked():
                return checkBox.text()
        
        if self.mixedChannelsCombineMetricsQGBox is None:
            return
        
        checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
        for checkBox in checkBoxes:
            if checkBox.isChecked():
                return checkBox.text()
    
    def addCheckboxesToGroup(self):
        for chNameGroupbox in self.chNameGroupboxes:
            for checkBox in chNameGroupbox.checkBoxes:
                self.checkBoxedGroup.addButton(checkBox)
        
        for checkBox in self.regionPropsQGBox.checkBoxes:
            self.checkBoxedGroup.addButton(checkBox)
        
        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            self.checkBoxedGroup.addButton(checkBox)
        
        if self.mixedChannelsCombineMetricsQGBox is None:
            return
        
        checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
        for checkBox in checkBoxes:
            self.checkBoxedGroup.addButton(checkBox)
            
    def channelCheckboxToggled(self, checkbox):
        # Make sure to automatically check the requested cell_vol metric for 
        # concentration metrics        
        if checkbox.text().find('concentration_') == -1:
            return
        
        if self.is_concat:
            # When this dialogue is used in concatenate pos utility we do not 
            # need to check that certain metrics are present
            return
        
        pattern = r'.+_from_vol_([a-z]+)(_3D)?(_?[A-Za-z0-9]*)'
        repl = r'cell_vol_\1\2'
        cell_vol_metric_name = re.sub(pattern, repl, checkbox.text())
        for sizeCheckbox in self.sizeMetricsQGBox.checkBoxes:
            if sizeCheckbox.text() == cell_vol_metric_name:
                break
        else:
            # Make sure to not check for similarly named custom metrics
            return 
        
        if checkbox.isChecked():
            sizeCheckbox.setChecked(True)
            sizeCheckbox.isRequired = True
        else:
            # Do not enable cell vol checkbox is any of the other 
            # concentration metrics requiring it is checked
            unit = cell_vol_metric_name[9:]
            is3D = unit.endswith('3D')
            for channelGBox in self.chNameGroupboxes:
                if not channelGBox.isChecked():
                    continue
                for _checkbox in channelGBox.checkBoxes: 
                    if _checkbox.text().find(f'_from_vol_{unit}') == -1: 
                        continue
                    if not is3D and _checkbox.text().find(f'{unit}_3D') != -1:
                        # Metric is 3D but the cell_vol is not 
                        continue
                    if _checkbox.isChecked():
                        return
            sizeCheckbox.isRequired = False
    
    def rpMetricToggled(self, checked):
        pass
    
    def mixedChannelsMetricToggled(self, checked):
        pass
    
    def sizeMetricToggled(self, checked):
        """Method called when a checkbox of a size metric is toggled.
        Check if the size value is required and explain why it cannot be 
        unchecked.

        Parameters
        ----------
        checked : bool
            State of the checkbox toggled
        """
        checkbox = self.sender()
        
        if self.is_concat:
            # When this dialogue is used in concatenate pos utility we do not 
            # need to check that certain metrics are present
            return

        if not hasattr(checkbox, 'isRequired'):
            return
        
        if not checkbox.isRequired:
            return
        
        if checkbox.isChecked():
            return
        
        checkbox.setChecked(True)

        if self.doNotWarn:
            return

        linked_autoBkgr_metric = checkbox.text().replace('cell', '_autoBkgr_from')
        linked_dataPrepBkgr_metric = checkbox.text().replace(
            'cell', '_dataPrepBkgr_from'
        )
        txt = html_utils.paragraph(f"""
            <b>This physical measurement cannot be unchecked</b> 
            because it is required 
            by the <code>{linked_autoBkgr_metric}</code> and 
            <code>{linked_dataPrepBkgr_metric}</code> measurements 
            that you requested to save.<br><br>

            Thank you for you patience!
        """)
        msg = widgets.myMessageBox(showCentered=False)
        msg.warning(self, 'Physical measurement required', txt)

    def deselectAll(self):
        self.doNotWarn = True
        for chNameGroupbox in self.chNameGroupboxes:
            for gb in chNameGroupbox.groupboxes:
                gb.checkAll(False)
            cgb = getattr(chNameGroupbox, 'customMetricsQGBox', None)
            if cgb is not None:
                cgb.checkAll(False)
        
        self.sizeMetricsQGBox.checkAll(False)
        self.regionPropsQGBox.checkAll(False)
        if self.mixedChannelsCombineMetricsQGBox is not None:
            self.mixedChannelsCombineMetricsQGBox.checkAll(False)
        self.doNotWarn = False
    
    def delMixedChannelCombineMetric(self, colname_to_del, hlayout):
        cp = measurements.read_saved_user_combine_config()
        for section in cp.sections():
            cp.remove_option(section, colname_to_del)
        measurements.save_common_combine_metrics(cp)

        for i in range(hlayout.count()):
            item = hlayout.itemAt(i)
            w = item.widget()
            if w is None:
                continue
            w.hide()
               
        if self.allPosData is not None:
            for posData in self.allPosData:
                _config = posData.combineMetricsConfig
                for section in _config.sections():
                    _config.remove_option(section, colname_to_del)
                posData.saveCombineMetrics()
    
    def setState(self, state):
        self.doNotWarn = True
        for chNameGroupbox in self.chNameGroupboxes:
            measurementsInfo = state.get(chNameGroupbox.title())
            if not measurementsInfo:
                chNameGroupbox.setChecked(False)
            else:
                for checkBox in chNameGroupbox.checkBoxes:
                    colname = checkBox.text()
                    checkBox.setChecked(measurementsInfo[colname])

        measurementsInfo = state.get(self.sizeMetricsQGBox.title())
        if not measurementsInfo:
            self.sizeMetricsQGBox.setChecked(False)
        else:
            for checkBox in self.sizeMetricsQGBox.checkBoxes:
                checked = checkBox.isChecked()
                colname = checkBox.text()
                checkBox.setChecked(measurementsInfo[colname])

        measurementsInfo = state.get(self.regionPropsQGBox.title())
        if not measurementsInfo:
            self.regionPropsQGBox.setChecked(False)
        else:
            self.regionPropsToSave = []
            for checkBox in self.regionPropsQGBox.checkBoxes:
                checked = checkBox.isChecked()
                colname = checkBox.text()
                checkBox.setChecked(measurementsInfo[colname])

        if self.mixedChannelsCombineMetricsQGBox is not None:
            measurementsInfo = state.get(self.mixedChannelsCombineMetricsQGBox.title())
            if not measurementsInfo:
                self.mixedChannelsCombineMetricsQGBox.setChecked(False)
            else:
                checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
                for checkBox in checkBoxes:
                    checked = checkBox.isChecked()
                    colname = checkBox.text()
                    key = self.mixedChannelsCombineMetricsQGBox.title()
                    checkBox.setChecked(measurementsInfo[colname])
        
        self.doNotWarn = False
            
    def state(self):
        state = {
            self.sizeMetricsQGBox.title(): {},
            self.regionPropsQGBox.title(): {}
        }
        for chNameGroupbox in self.chNameGroupboxes:
            state[chNameGroupbox.title()] = {}
            if not chNameGroupbox.isChecked():
                # Channel unchecked
                continue
            else:
                for checkBox in chNameGroupbox.checkBoxes:
                    colname = checkBox.text()
                    state[chNameGroupbox.title()][colname] = checkBox.isChecked()

        if not self.sizeMetricsQGBox.isChecked():
            pass
        else:
            for checkBox in self.sizeMetricsQGBox.checkBoxes:
                checked = checkBox.isChecked()
                colname = checkBox.text()
                state[self.sizeMetricsQGBox.title()][colname] = checked

        if not self.regionPropsQGBox.isChecked():
            pass
        else:
            self.regionPropsToSave = []
            for checkBox in self.regionPropsQGBox.checkBoxes:
                checked = checkBox.isChecked()
                colname = checkBox.text()
                state[self.regionPropsQGBox.title()][colname] = checked

        if self.mixedChannelsCombineMetricsQGBox is not None:
            state[self.mixedChannelsCombineMetricsQGBox.title()] = {}
            if self.mixedChannelsCombineMetricsQGBox.isChecked():
                checkBoxes = self.mixedChannelsCombineMetricsQGBox.checkBoxes
                for checkBox in checkBoxes:
                    checked = checkBox.isChecked()
                    key = self.mixedChannelsCombineMetricsQGBox.title()
                    colname = checkBox.text()
                    state[key][colname] = checked
        
        return state
    
    def restoreState(self, state):
        for chNameGroupbox in self.chNameGroupboxes:
            _state = state.get(chNameGroupbox.title())
            if _state is None or not _state:
                continue
            for checkBox in chNameGroupbox.checkBoxes:
                isChecked = _state.get(checkBox.text())
                if isChecked is None:
                    continue
                checkBox.setChecked(isChecked)
        
        _state = state.get(self.sizeMetricsQGBox.title())
        if _state is None or not _state:
            pass
        else:
            for checkBox in self.sizeMetricsQGBox.checkBoxes:
                isChecked = _state.get(checkBox.text())
                if isChecked is None:
                    continue
                checkBox.setChecked(isChecked)
        
        _state = state.get(self.regionPropsQGBox.title())
        if _state is None or not _state:
            pass
        else:
            for checkBox in self.regionPropsQGBox.checkBoxes:
                isChecked = _state.get(checkBox.text())
                if isChecked is None:
                    continue
                checkBox.setChecked(isChecked)
        
        if self.mixedChannelsCombineMetricsQGBox is None:
            return
        
        _state = state.get(self.mixedChannelsCombineMetricsQGBox.title())
        if _state is None or not _state:
            pass
        else:
            for checkBox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
                isChecked = _state.get(checkBox.text())
                if isChecked is None:
                    continue
                checkBox.setChecked(isChecked)
    
    def loadLastSelection(self):
        self.doNotWarn = True
        for chNameGroupbox in self.chNameGroupboxes:
            chNameGroupbox.checkFavouriteFuncs()
            chNameGroupbox.customMetricsQGBox.checkFavouriteFuncs()
        self.sizeMetricsQGBox.checkFavouriteFuncs()
        self.regionPropsQGBox.checkFavouriteFuncs()
        last_sel_gb = load.read_last_selected_gb_meas()
        if not last_sel_gb:
            return
        refChannel = self.chNameGroupboxes[0].chName
        groupboxes = [
            *self.chNameGroupboxes, self.sizeMetricsQGBox, 
            self.regionPropsQGBox
        ]
        for chNameGroupbox in self.chNameGroupboxes:
            chNameGroupbox.setChecked(False)
        self.sizeMetricsQGBox.setChecked(False)
        self.regionPropsQGBox.setChecked(False)
        if refChannel not in last_sel_gb:
            return
        
        for gb_title in last_sel_gb[refChannel]:
            for gb in groupboxes:
                if gb.title() != gb_title:
                    continue
                gb.setChecked(True)
        self.doNotWarn = False
    
    def setDisabledMetricsRequestedForCombined(self, checked):
        checkbox = self.sender()
        
        if self.is_concat:
            # When this dialogue is used in concatenate pos utility we do not 
            # need to check that certain metrics are present
            return
        
        # Set checked and disable those metrics that are requested for 
        # combined measurements
        allCheckboxes = []

        for chNameGroupbox in self.chNameGroupboxes:
            for chCheckBox in chNameGroupbox.checkBoxes:
                chCheckBox.setDisabled(False)
                allCheckboxes.append(chCheckBox)
        
        for sizeCheckBox in self.sizeMetricsQGBox.checkBoxes:
            sizeCheckBox.setDisabled(False)
            allCheckboxes.append(chCheckBox)
        
        for rpCheckBox in self.regionPropsQGBox.checkBoxes:
            rpCheckBox.setDisabled(False)
            allCheckboxes.append(chCheckBox)
        
        if not self.mixedChannelsCombineMetricsQGBox.isChecked():
            return
        
        for cb in allCheckboxes:
            metricName = cb.text()
            for combCheckbox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
                equation = combCheckbox.equation
                if equation.find(metricName) == -1:
                    continue
                elif combCheckbox.isChecked():
                    cb.setChecked(True)
                    cb.setDisabled(True)
                    cb.setToolTip(
                        'This metric cannot be removed because it is required '
                        f'by the combined measurement "{combCheckbox.text()}"'
                    )

    def keyPressEvent(self, a0: QKeyEvent) -> None:
        state = self.state()
        return super().keyPressEvent(a0)
    
    def closeEvent(self, event):
        if self.cancel:
            self.sigCancel.emit()
        super().closeEvent(event)
    
    def restart(self):
        self.cancel = False
        self.close()
        self.sigRestart.emit()
    
    def setDisabledNotExistingMeasurements(self, existing_colnames):
        self.existing_colnames = existing_colnames
        for chNameGroupbox in self.chNameGroupboxes:
            for checkBox in chNameGroupbox.checkBoxes:
                colname = checkBox.text()
                if colname in existing_colnames:
                    checkBox.setChecked(True)
                    continue
                
                checkBox.setChecked(False)
                checkBox.setDisabled(True)
                self.setNotExistingMeasurementTooltip(checkBox)
        
        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            colname = checkBox.text()
            if colname in existing_colnames:
                checkBox.setChecked(True)
                continue
            checkBox.setChecked(False)
            checkBox.setDisabled(True)
            self.setNotExistingMeasurementTooltip(checkBox)
        
        for checkBox in self.regionPropsQGBox.checkBoxes:
            prop_name = checkBox.text()
            for existing_col in existing_colnames:
                if prop_name == existing_col:
                    checkBox.setChecked(True)
                    break
                m = re.match(fr'{prop_name}-\d', existing_col)
                if m is not None:
                    checkBox.setChecked(True)
                    break
            else:
                checkBox.setChecked(False)
                checkBox.setDisabled(True)
                self.setNotExistingMeasurementTooltip(checkBox)
        
        if self.mixedChannelsCombineMetricsQGBox is None:
            return
        
        for combCheckbox in self.mixedChannelsCombineMetricsQGBox.checkBoxes:
            colname = combCheckbox.text()
            if colname in existing_colnames:
                combCheckbox.setChecked(True)
                continue
            combCheckbox.setChecked(False)
            combCheckbox.setDisabled(True)
            self.setNotExistingMeasurementTooltip(combCheckbox)
        
    def addNonMeasurementColumns(self, colnames):
        additionalCols = measurements.get_non_measurements_cols(
            colnames, self.all_metrics
        )
        if not additionalCols:
            return
        self.nonMeasurementsGroupbox = widgets.CheckboxesGroupBox(
            additionalCols, title='Additional columns', checkable=True
        )
        self.groupsLayout.addWidget(
            self.nonMeasurementsGroupbox, 2, self.numberCols
        )
            
    
    def setNotExistingMeasurementTooltip(self, checkBox):
        checkBox.setToolTip(
            'Measurement is disabled because it is not present in selected '
            'acdc_output tables, hence it cannot be addded to concatenated '
            'table. '
        )

    def ok_cb(self):
        if self.allPos_acdc_df_cols is None:
            self.cancel = False
            self.close()
            self.sigClosed.emit()
            return

        self.okClicked = True
        existing_colnames = self.allPos_acdc_df_cols
        unchecked_existing_colnames = []
        unchecked_existing_rps = []
        for chNameGroupbox in self.chNameGroupboxes:
            for checkBox in chNameGroupbox.checkBoxes:
                colname = checkBox.text()
                is_existing = colname in existing_colnames
                if not chNameGroupbox.isChecked() and is_existing:
                    unchecked_existing_colnames.append(colname)
                    continue
                if not checkBox.isChecked() and is_existing:
                    unchecked_existing_colnames.append(colname)
        for checkBox in self.sizeMetricsQGBox.checkBoxes:
            colname = checkBox.text()
            is_existing = colname in existing_colnames
            if not self.sizeMetricsQGBox.isChecked() and is_existing:
                unchecked_existing_colnames.append(colname)
                continue

            if not checkBox.isChecked() and is_existing:
                unchecked_existing_colnames.append(colname)
        for checkBox in self.regionPropsQGBox.checkBoxes:
            colname = checkBox.text()
            is_existing = any([col == colname for col in existing_colnames])
            if not self.regionPropsQGBox.isChecked() and is_existing:
                unchecked_existing_rps.append(colname)
                continue

            if not checkBox.isChecked() and is_existing:
                unchecked_existing_rps.append(colname)

        if unchecked_existing_colnames or unchecked_existing_rps:
            cancel, self.delExistingCols = self.warnUncheckedExistingMeasurements(
                unchecked_existing_colnames, unchecked_existing_rps
            )
            self.existingUncheckedColnames = unchecked_existing_colnames
            self.existingUncheckedRps = unchecked_existing_rps
            if cancel:
                return

        self.cancel = False  
        self.close()
        self.sigClosed.emit()
        
    def warnUncheckedExistingMeasurements(
            self, unchecked_existing_colnames, unchecked_existing_rps
        ):
        msg = widgets.myMessageBox()
        msg.setWidth(500)
        msg.addShowInFileManagerButton(self.acdc_df_path)
        txt = html_utils.paragraph(
            'You chose to <b>not save</b> some measurements that are '
            '<b>already present</b> in the saved <code>acdc_output.csv</code> '
            'file.<br><br>'
            'Do you want to <b>delete</b> these measurements or '
            '<b>keep</b> them?<br><br>'
            'Existing measurements not selected:'
        )
        listView = widgets.readOnlyQList(msg)
        items = unchecked_existing_colnames.copy()
        items.extend(unchecked_existing_rps)
        listView.addItems(items)
        _, delButton, keepButton = msg.warning(
            self, 'Unchecked existing measurements', txt,
            widgets=listView, buttonsTexts=('Cancel', 'Delete', 'Keep')
        )
        return msg.cancel, msg.clickedButton == delButton

    def show(self, block=False):
        super().show(block=False)
        self.deselectAllButton.setMinimumHeight(self.okButton.height())
        screenWidth = self.screen().size().width()
        screenHeight = self.screen().size().height()
        screenLeft = self.screen().geometry().x()
        screenTop = self.screen().geometry().y()
        h = screenHeight-200
        minColWith = screenWidth/5
        w = minColWith*self.numberCols
        xLeft = int((screenWidth-w)/2)
        if w > screenWidth:
            self.move(screenLeft+10, screenTop+50)
            self.resize(screenWidth-20, h)
        else:
            self.move(screenLeft+xLeft, screenTop+50)
            self.resize(int(w), h)
        super().show(block=block)

class QDialogMetadataXML(QDialog):
    sigDimensionOrderEditFinished = Signal(str, int, int, object)

    def __init__(
            self, title='Metadata',
            LensNA=1.0, DimensionOrder=None, rawFilename='test',
            SizeT=1, SizeZ=1, SizeC=1, SizeS=1,
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
        font = QFont()
        font.setPixelSize(12)
        self.setFont(font)
        if DimensionOrder is None:
            DimensionOrder = 'ztc' # default for .czi and .nd2

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
        self.SizeS_SB.valueChanged.connect(self.SizeSvalueChanged)

        if rawDataStruct == 0:
            row += 1
            self.posSelector = widgets.ExpandableListBox()
            positions = ['All positions']
            positions.extend([f'Position_{i+1}' for i in range(SizeS)])
            self.posSelector.addItems(positions)
            txt = 'Positions to save:  '
            label = QLabel(txt)
            entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
            entriesLayout.addWidget(self.posSelector, row, 1)

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
        self.DimensionOrderCombo = widgets.QCenteredComboBox()
        if sampleImgData is None:
            items = [''.join(perm) for perm in permutations('zct', 3)]
        else:
            items = list(sampleImgData.keys())
        self.DimensionOrderCombo.addItems(items)
        self.DimensionOrderCombo.setCurrentText(DimensionOrder[:3].lower())
        txt = 'Order of dimensions:  '
        label = QLabel(txt)
        entriesLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        entriesLayout.addWidget(self.DimensionOrderCombo, row, 1)
        dimensionOrderLayout = QHBoxLayout()
        DimensionOrderHelpButton = widgets.infoPushButton()
        dimensionOrderLayout.addWidget(DimensionOrderHelpButton)
        dimensionOrderLayout.addStretch(1)
        entriesLayout.addLayout(dimensionOrderLayout, row, 2, 1, 2)
        
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
        self.TimeIncrement_DSB = widgets.FloatLineEdit(
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
            okButton = widgets.okPushButton(' Ok ')
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

        cancelButton = widgets.cancelPushButton('Cancel')
        buttonsLayout.addWidget(cancelButton, 0, 2)
        buttonsLayout.setColumnStretch(0, 1)
        buttonsLayout.setColumnStretch(3, 1)
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(entriesLayout)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch(1)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)
        self.DimensionOrderCombo.currentTextChanged.connect(
            self.dimensionOrderChanged
        )
        DimensionOrderHelpButton.clicked.connect(self.dimensionOrderHelp)

        self.setLayout(mainLayout)
        # self.setModal(True)

    def dimensionOrderChanged(self, dimsOrder):
        if self.imageViewer is None:
            return
        
        idx = self.imageViewer.channelIndex
        imgData = self.sampleImgData[dimsOrder][idx] 
        if self.imageViewer.posData.SizeT == 1:
            self.imageViewer.posData.img_data = [imgData] # single frame data
        else:
            self.imageViewer.posData.img_data = imgData
        self.imageViewer.update_img()
    
    def dimensionOrderHelp(self):
        txt = html_utils.paragraph('''
            The "Order of dimensions" is used to get the correct frame given 
            the <b>z-slice</b> (if SizeZ > 1) index, the <b>channel</b> (if SizeC > 1) index, 
            and the <b>frame</b> (if SizeT > 1) index.<br><br>
            Example: "zct" means that the order of dimensions in the image shape  
            is (SizeZ, SizeC, SizeT).<br><br>
            To test this, click on the "eye" button besides the channel name below. For 
            time-lapse data you will be able to visualize the first 4 frames. 
            If the order of dimensions is correct, the displayed image should be 
            the <b>image of the corresponding channel</b>. For time-lapse data check that 
            every frame is correct. Make sure to also check 
            that the z-slices are in the correct order by scrolling with the 
            z-slice scrollbar.
        ''')
        msg = widgets.myMessageBox()
        msg.information(self, 'Order of dimensions help', txt)

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
        if value > 1:
            self.TimeIncrement_DSB.show()
            self.TimeIncrementUnit_CB.show()
            self.TimeIncrement_Label.show()
        else:
            self.TimeIncrement_DSB.hide()
            self.TimeIncrementUnit_CB.hide()
            self.TimeIncrement_Label.hide()
        self.readSampleImgDataAgain = True

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
        dimsOrder = self.DimensionOrderCombo.currentText()
        imgData = self.sampleImgData[dimsOrder][idx]
        posData = myutils.utilClass()
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

    def confirmOrderOfDimensions(self):
        helpButton = widgets.helpPushButton('More details...')
        txt = html_utils.paragraph("""
            Are you sure that the parameter <code>Order of dimensions</code> 
            <b>is correct</b>?<br>
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Double-check Order of dimensions', txt, showDialog=False,
            buttonsTexts=('Cancel', helpButton, 'Yes')
        )
        helpButton.clicked.disconnect()
        helpButton.clicked.connect(self.dimensionOrderHelp)
        msg.exec_()
        return msg.cancel
    
    def ok_cb(self, event):
        areChNamesValid = self.checkChNames()
        if not areChNamesValid:
            err_msg = html_utils.paragraph(
                'Channel names <b>cannot be empty</b> or equal to each other.'
                '<br><br>'
                'Insert a unique text for each channel name.'
            )
            msg = widgets.myMessageBox()
            msg.critical(
               self, 'Invalid channel names', err_msg
            )
            return

        cancel = self.confirmOrderOfDimensions()
        if cancel:
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
        self.DimensionOrder = self.DimensionOrderCombo.currentText()
        self.SizeT = self.SizeT_SB.value()
        self.SizeZ = self.SizeZ_SB.value()
        self.SizeC = self.SizeC_SB.value()
        self.SizeS = self.SizeS_SB.value()
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
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()
        self.setSize()
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class CellACDCTrackerParamsWin(QDialog):
    def __init__(self, parent=None):
        self.cancel = True
        super().__init__(parent)

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setWindowTitle('Cell-ACDC tracker parameters')

        paramsLayout = QGridLayout()
        paramsBox = QGroupBox()

        row = 0
        label = QLabel(html_utils.paragraph(
            'Minimum overlap between objects'
        ))
        paramsLayout.addWidget(label, row, 0)
        maxOverlapSpinbox = QDoubleSpinBox()
        maxOverlapSpinbox.setAlignment(Qt.AlignCenter)
        maxOverlapSpinbox.setMinimum(0)
        maxOverlapSpinbox.setMaximum(1)
        maxOverlapSpinbox.setSingleStep(0.1)
        maxOverlapSpinbox.setValue(0.4)
        self.maxOverlapSpinbox = maxOverlapSpinbox
        paramsLayout.addWidget(maxOverlapSpinbox, row, 1)
        infoButton = widgets.infoPushButton()
        infoButton.clicked.connect(self.showInfo)
        paramsLayout.addWidget(infoButton, row, 2)
        paramsLayout.setColumnStretch(0, 0)
        paramsLayout.setColumnStretch(1, 1)
        paramsLayout.setColumnStretch(2, 0)

        cancelButton = widgets.cancelPushButton('Cancel')
        okButton = widgets.okPushButton(' Ok ')
        cancelButton.clicked.connect(self.cancel_cb)
        okButton.clicked.connect(self.ok_cb)

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        layout = QVBoxLayout()
        infoText = html_utils.paragraph('<b>Cell-ACDC tracker parameters</b>')
        infoLabel = QLabel(infoText)
        layout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        layout.addSpacing(10)
        paramsBox.setLayout(paramsLayout)
        layout.addWidget(paramsBox)
        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)
        layout.addStretch(1)
        self.setLayout(layout)
        self.setFont(font)

    def showInfo(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            'Cell-ACDC tracker computes the percentage of overlap between '
            'all the objects<br> at frame <code>n</code> and all the '
            'objects in previous frame <code>n-1</code>.<br><br>'
            'All objects with <b>overlap less than</b> '
            '<code>Minimum overlap between objects</code><br>are considered '
            '<b>new objects</b>.<br><br>'
            'Set this value to 0 if you want to force tracking of ALL the '
            'objects<br> in the previous frame (e.g., if cells move a lot '
            'between frames)'
        )
        msg.information(self, 'Cell-ACDC tracker info', txt)

    def ok_cb(self, checked=False):
        self.cancel = False
        self.params = {'IoA_thresh': self.maxOverlapSpinbox.value()}
        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        self.resize(int(self.width()*1.3), self.height())
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class BayesianTrackerParamsWin(QDialog):
    def __init__(
            self, segmShape, parent=None, channels=None, 
            currentChannelName=None
        ):
        self.cancel = True
        super().__init__(parent)

        self.channels = channels
        self.currentChannelName = currentChannelName

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        self.setWindowTitle('Bayesian tracker parameters')

        paramsLayout = QGridLayout()
        paramsBox = QGroupBox()

        row = 0
        this_path = os.path.dirname(os.path.abspath(__file__))
        default_model_path = os.path.join(
            this_path, 'trackers', 'BayesianTracker',
            'model', 'cell_config.json'
        )
        label = QLabel(html_utils.paragraph('Model path'))
        paramsLayout.addWidget(label, row, 0)
        modelPathLineEdit = QLineEdit()
        start_dir = ''
        if os.path.exists(default_model_path):
            start_dir = os.path.dirname(default_model_path)
            modelPathLineEdit.setText(default_model_path)
        self.modelPathLineEdit = modelPathLineEdit
        paramsLayout.addWidget(modelPathLineEdit, row, 1)
        browseButton = widgets.browseFileButton(
            title='Select Bayesian Tracker model file',
            ext={'JSON Config': ('.json',)},
            start_dir=start_dir
        )
        browseButton.sigPathSelected.connect(self.onPathSelected)
        paramsLayout.addWidget(browseButton, row, 2, alignment=Qt.AlignLeft)

        if self.channels is not None:
            row += 1
            label = QLabel(html_utils.paragraph('Intensity image channel:  '))
            paramsLayout.addWidget(label, row, 0)
            items = ['None', *self.channels]
            self.channelCombobox = widgets.QCenteredComboBox()
            self.channelCombobox.addItems(items)
            paramsLayout.addWidget(self.channelCombobox, row, 1)
            if self.currentChannelName is not None:
                self.channelCombobox.setCurrentText(self.currentChannelName)

        row += 1
        label = QLabel(html_utils.paragraph('Features'))
        paramsLayout.addWidget(label, row, 0)
        selectFeaturesButton = widgets.setPushButton('Select features')
        paramsLayout.addWidget(selectFeaturesButton, row, 1)
        self.features = []
        selectFeaturesButton.clicked.connect(self.selectFeatures)

        row += 1
        label = QLabel(html_utils.paragraph('Verbose'))
        paramsLayout.addWidget(label, row, 0)
        verboseToggle = widgets.Toggle()
        verboseToggle.setChecked(True)
        self.verboseToggle = verboseToggle
        paramsLayout.addWidget(verboseToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph('Run optimizer'))
        paramsLayout.addWidget(label, row, 0)
        optimizeToggle = widgets.Toggle()
        optimizeToggle.setChecked(True)
        self.optimizeToggle = optimizeToggle
        paramsLayout.addWidget(optimizeToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph('Max search radius'))
        paramsLayout.addWidget(label, row, 0)
        maxSearchRadiusSpinbox = QSpinBox()
        maxSearchRadiusSpinbox.setAlignment(Qt.AlignCenter)
        maxSearchRadiusSpinbox.setMinimum(1)
        maxSearchRadiusSpinbox.setMaximum(2147483647)
        maxSearchRadiusSpinbox.setValue(50)
        self.maxSearchRadiusSpinbox = maxSearchRadiusSpinbox
        self.maxSearchRadiusSpinbox.setDisabled(True)
        paramsLayout.addWidget(maxSearchRadiusSpinbox, row, 1)

        row += 1
        Z, Y, X = segmShape
        label = QLabel(html_utils.paragraph('Tracking volume'))
        paramsLayout.addWidget(label, row, 0)
        volumeLineEdit = QLineEdit()
        defaultVol = f'  (0, {X}), (0, {Y})  '
        if Z > 1:
            defaultVol = f'{defaultVol}, (0, {Z})  '
        volumeLineEdit.setText(defaultVol)
        volumeLineEdit.setAlignment(Qt.AlignCenter)
        self.volumeLineEdit = volumeLineEdit
        paramsLayout.addWidget(volumeLineEdit, row, 1)

        row += 1
        label = QLabel(html_utils.paragraph('Interactive mode step size'))
        paramsLayout.addWidget(label, row, 0)
        stepSizeSpinbox = QSpinBox()
        stepSizeSpinbox.setAlignment(Qt.AlignCenter)
        stepSizeSpinbox.setMinimum(1)
        stepSizeSpinbox.setMaximum(2147483647)
        stepSizeSpinbox.setValue(100)
        self.stepSizeSpinbox = stepSizeSpinbox
        paramsLayout.addWidget(stepSizeSpinbox, row, 1)

        row += 1
        label = QLabel(html_utils.paragraph('Update method'))
        paramsLayout.addWidget(label, row, 0)
        updateMethodCombobox = QComboBox()
        updateMethodCombobox.addItems(['EXACT', 'APPROXIMATE'])
        self.updateMethodCombobox = updateMethodCombobox
        self.updateMethodCombobox.currentTextChanged.connect(self.methodChanged)
        paramsLayout.addWidget(updateMethodCombobox, row, 1)

        cancelButton = widgets.cancelPushButton('Cancel')
        okButton = widgets.okPushButton(' Ok ')
        cancelButton.clicked.connect(self.cancel_cb)
        okButton.clicked.connect(self.ok_cb)

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        layout = QVBoxLayout()
        infoText = html_utils.paragraph('<b>Bayesian Tracker parameters</b>')
        infoLabel = QLabel(infoText)
        layout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        layout.addSpacing(10)
        paramsBox.setLayout(paramsLayout)
        layout.addWidget(paramsBox)

        url = 'https://btrack.readthedocs.io/en/latest/index.html'
        moreInfoText = html_utils.paragraph(
            '<i>Find more info on the Bayesian Tracker\'s '
            f'<a href="{url}">home page</a></i>'
        )
        moreInfoLabel = QLabel(moreInfoText)
        moreInfoLabel.setOpenExternalLinks(True)
        layout.addWidget(moreInfoLabel, alignment=Qt.AlignCenter)

        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)
        layout.addStretch(1)
        self.setLayout(layout)
        self.setFont(font)
    
    def selectFeatures(self):
        features = measurements.get_btrack_features()
        selectWin = widgets.QDialogListbox(
            'Select features',
            'Select features to use for tracking:\n',
            features, multiSelection=True, parent=self, 
            includeSelectionHelp=True
        )
        for i in range(selectWin.listBox.count()):
            item = selectWin.listBox.item(i)
            if item.text() in self.features:
                item.setSelected(True)
        selectWin.exec_()
        if selectWin.cancel:
            return
        self.features = selectWin.selectedItemsText

    def methodChanged(self, method):
        if method == 'APPROXIMATE':
            self.maxSearchRadiusSpinbox.setDisabled(False)
        else:
            self.maxSearchRadiusSpinbox.setDisabled(True)

    def onPathSelected(self, path):
        self.modelPathLineEdit.setText(path)

    def ok_cb(self, checked=False):
        self.cancel = False
        try:
            m = re.findall('\((\d+), *(\d+)\)', self.volumeLineEdit.text())
            if len(m) < 2:
                raise
            self.volume = tuple([(int(start), int(end)) for start, end in m])
            if len(self.volume) == 2:
                self.volume = (self.volume[0], self.volume[1], (-1e5, 1e5))
        except Exception as e:
            self.warnNotAcceptedVolume()
            return

        if not os.path.exists(self.modelPathLineEdit.text()):
            self.warnNotVaidPath()
            return

        self.intensityImageChannel = None
        self.verbose = self.verboseToggle.isChecked()
        self.max_search_radius = self.maxSearchRadiusSpinbox.value()
        self.update_method = self.updateMethodCombobox.currentText()
        self.model_path = os.path.normpath(self.modelPathLineEdit.text())
        self.params = {
            'model_path': self.model_path,
            'verbose': self.verbose,
            'volume': self.volume,
            'max_search_radius': self.max_search_radius,
            'update_method': self.update_method,
            'step_size': self.stepSizeSpinbox.value(),
            'optimize': self.optimizeToggle.isChecked(),
            'features': self.features
        }
        if self.channels is not None:
            if self.channelCombobox.currentText() != 'None':
                self.intensityImageChannel = self.channelCombobox.currentText()
        self.close()

    def warnNotVaidPath(self):
        url = 'https://github.com/lowe-lab-ucl/segment-classify-track/tree/main/models'
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            'The model configuration file path<br><br>'
            f'{self.modelPathLineEdit.text()}<br><br> '
            'does <b>not exist.</b><br><br>'
            'You can find some <b>pre-configured models</b> '
            f'<a href="{url}">here</a>.'
        )
        msg.critical(
            self, 'Invalid volume', txt
        )

    def warnNotAcceptedVolume(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            f'{self.volumeLineEdit.text()} is <b>not a valid volume!</b><br><br>'
            'Valid volume is for example (0, 2048), (0, 2048)<br>'
            'for 2D segmentation or (0, 2048), (0, 2048), (0, 2048)<br>'
            'for 3D segmentation.'
        )
        msg.critical(
            self, 'Invalid volume', txt
        )

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        self.resize(int(self.width()*1.3), self.height())
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class DeltaTrackerParamsWin(QDialog):

    def __init__(self, posData=None, parent=None):
        self.cancel = True
        super().__init__(parent)

        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        self.setWindowTitle('Delta tracker parameters')

        paramsLayout = QGridLayout()
        paramsBox = QGroupBox()

        row = 0
        this_path = os.path.dirname(os.path.abspath(__file__))
        default_model_path = this_path

        label = QLabel(html_utils.paragraph('Original Images path'))
        paramsLayout.addWidget(label, row, 0)
        modelPathLineEdit = QLineEdit()
        start_dir = ''
        if os.path.exists(default_model_path):
            start_dir = os.path.dirname(default_model_path)
            modelPathLineEdit.setText(default_model_path)
        self.modelPathLineEdit = modelPathLineEdit
        paramsLayout.addWidget(modelPathLineEdit, row, 1)
        browseButton = widgets.browseFileButton(
            title='Select Original Images',
            ext={'TIFF': ('.tif',)},
            start_dir=start_dir
        )
        if posData is not None:
            modelPathLineEdit.setText(posData.imgPath)
        browseButton.sigPathSelected.connect(self.onPathSelected)
        paramsLayout.addWidget(browseButton, row, 2, alignment=Qt.AlignLeft)

        row += 1
        label = QLabel(html_utils.paragraph('Model Type'))
        paramsLayout.addWidget(label, row, 0)
        updateMethodCombobox = QComboBox()
        updateMethodCombobox.addItems(['2D', 'mothermachine'])
        self.model_type = '2D'
        self.updateMethodCombobox = updateMethodCombobox
        self.updateMethodCombobox.currentTextChanged.connect(self.methodChanged)
        paramsLayout.addWidget(updateMethodCombobox, row, 1)

        row += 1
        label = QLabel(html_utils.paragraph('Single Mother Machine Chamber?'))
        paramsLayout.addWidget(label, row, 0)
        chamberToggle = widgets.Toggle()
        chamberToggle.setChecked(True)
        self.chamberToggle = chamberToggle
        paramsLayout.addWidget(chamberToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph('Verbose'))
        paramsLayout.addWidget(label, row, 0)
        verboseToggle = widgets.Toggle()
        verboseToggle.setChecked(True)
        self.verboseToggle = verboseToggle
        paramsLayout.addWidget(verboseToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph('Legacy Save (.mat)'))
        paramsLayout.addWidget(label, row, 0)
        legacyToggle = widgets.Toggle()
        legacyToggle.setChecked(False)
        self.legacyToggle = legacyToggle
        paramsLayout.addWidget(legacyToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph('Pickle (.pkl)'))
        paramsLayout.addWidget(label, row, 0)
        pickleToggle = widgets.Toggle()
        pickleToggle.setChecked(False)
        self.pickleToggle = pickleToggle
        paramsLayout.addWidget(pickleToggle, row, 1, alignment=Qt.AlignCenter)

        row += 1
        label = QLabel(html_utils.paragraph('Movie (.mp4) *only for 2D images'))
        paramsLayout.addWidget(label, row, 0)
        movieToggle = widgets.Toggle()
        movieToggle.setChecked(False)
        self.movieToggle = movieToggle
        paramsLayout.addWidget(movieToggle, row, 1, alignment=Qt.AlignCenter)

        cancelButton = widgets.cancelPushButton('Cancel')
        okButton = widgets.okPushButton(' Ok ')
        cancelButton.clicked.connect(self.cancel_cb)
        okButton.clicked.connect(self.ok_cb)

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        layout = QVBoxLayout()
        infoText = html_utils.paragraph('<b>Delta Tracker parameters</b>')
        infoLabel = QLabel(infoText)
        layout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        layout.addSpacing(10)
        paramsBox.setLayout(paramsLayout)
        layout.addWidget(paramsBox)

        url = 'https://delta.readthedocs.io/en/latest/'
        moreInfoText = html_utils.paragraph(
            '<i>Find more info on Delta Tracker\'s '
            f'<a href="{url}">home page</a></i>'
        )
        moreInfoLabel = QLabel(moreInfoText)
        moreInfoLabel.setOpenExternalLinks(True)
        layout.addWidget(moreInfoLabel, alignment=Qt.AlignCenter)

        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)
        layout.addStretch(1)
        self.setLayout(layout)
        self.setFont(font)

    def methodChanged(self, method):
        if method == 'mothermachine':
            self.model_type = 'mothermachine'

    def onPathSelected(self, path):
        self.modelPathLineEdit.setText(path)

    def ok_cb(self, checked=False):
        self.cancel = False

        if not os.path.exists(self.modelPathLineEdit.text()):
            self.warnNotVaidPath()
            return

        self.verbose = self.verboseToggle.isChecked()
        self.legacy = self.legacyToggle.isChecked()
        self.pickle = self.pickleToggle.isChecked()
        self.movie = self.movieToggle.isChecked()
        self.chamber = self.chamberToggle.isChecked()
        self.model_path = os.path.normpath(self.modelPathLineEdit.text())
        self.params = {
            'original_images_path': self.model_path,
            'verbose': self.verbose,
            'legacy': self.legacy,
            'pickle': self.pickle,
            'movie': self.movie,
            'model_type': self.model_type,
            'single mothermachine chamber': self.chamber
        }
        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        self.resize(int(self.width()*1.3), self.height())
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class QDialogWorkerProgress(QDialog):
    sigClosed = Signal(bool)

    def __init__(
            self, title='Progress', infoTxt='',
            showInnerPbar=False, pbarDesc='',
            parent=None
        ):
        self.workerFinished = False
        self.aborted = False
        self.clickCount = 0
        super().__init__(parent)

        abort_text = 'Option+Command+C to abort' if is_mac else 'Ctrl+Alt+C to abort'
        self.abort_text = abort_text

        self.setWindowTitle(f'{title} ({abort_text})')
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
            Aborting with <code>{self.abort_text}</code> is <b>not safe</b>.<br><br>
            The system status cannot be predicted and
            it will <b>require a restart</b>.<br><br>
            Are you sure you want to abort?
        """)
        yesButton, noButton = msg.critical(
            self, 'Are you sure you want to abort?', txt,
            buttonsTexts=('Yes', 'No')
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
        mainWinCenterX = int(mainWinLeft+mainWinWidth/2)
        mainWinCenterY = int(mainWinTop+mainWinHeight/2)

        width = int(screenWidth/3)
        width = width if self.width() < width else self.width()
        height = int(screenHeight/3)
        left = int(mainWinCenterX - width/2)
        left = left if left >= 0 else 0
        top = int(mainWinCenterY - height/2)

        self.setGeometry(left, top, width, height)

class QDialogCombobox(QDialog):
    def __init__(
            self, title, ComboBoxItems, informativeText,
            CbLabel='Select value:  ', parent=None,
            defaultChannelName=None, iconPixmap=None, centeredCombobox=False
        ):
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

        okButton = widgets.okPushButton('Ok')

        cancelButton = widgets.cancelPushButton('Cancel')

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
        if hasattr(self, 'loop'):
            self.loop.exit()

class MultiTimePointFilePattern(QBaseDialog):
    def __init__(self, fileName, folderPath, readPatternFunc=None, parent=None):
        super().__init__(parent)

        self.setWindowTitle('File name pattern')
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
        noteLayout.setStretch(0,0)
        noteLayout.setStretch(1,1)

        label = QLabel(html_utils.paragraph(
            f'Sample file name: <code>{fileName}</code>'
        ))
        mainLayout.addWidget(label, alignment=Qt.AlignCenter)
        mainLayout.addSpacing(5)

        channelName = ''
        posName = ''
        frameNumber = None
        if readPatternFunc is not None:
            posName, frameNumber, channelName = readPatternFunc(fileName)

        formLayout = QGridLayout()

        ncols = 3
        self.vLayouts = [QVBoxLayout() for _ in range(ncols)]
        for j, l in enumerate(self.vLayouts):
            formLayout.addLayout(l, 0, j)

        row = 0
        items = QLabel('Position name: '), widgets.ReadOnlyLineEdit(), QLabel()
        label, self.posNameEntry, button = items
        self.posNameEntry.setAlignment(Qt.AlignCenter)
        self.posNameEntry.setText(str(posName))
        for j, w in enumerate(items):
            self.vLayouts[j].addWidget(w)
        
        row += 1
        items = (
            QLabel('Frame number name: '), widgets.ReadOnlyLineEdit(), QLabel()
        )
        self.frameNumberEntry = items[1]
        self.frameNumberEntry.setText(str(frameNumber))
        self.frameNumberEntry.setAlignment(Qt.AlignCenter)
        for j, w in enumerate(items):
            self.vLayouts[j].addWidget(w)
        
        row += 1
        self.channelNameLE = widgets.alphaNumericLineEdit()
        items = (
            QLabel('Channel_1 name: '), self.channelNameLE, 
            widgets.addPushButton(' Add channel')
        )
        self.addChannelButton = items[2]
        self.addChannelButton._row = row
        self.channelNameLE.setAlignment(Qt.AlignCenter)
        self.channelNameLE.setText(channelName)
        for j, w in enumerate(items):
            self.vLayouts[j].addWidget(w)

        row += 1
        items = (
            QLabel('Basename (optional): '), widgets.alphaNumericLineEdit(), 
            QLabel()
        )
        label, self.baseNameLE, button = items
        self.baseNameLE.setAlignment(Qt.AlignCenter)
        for j, w in enumerate(items):
            self.vLayouts[j].addWidget(w)
        
        row += 1
        items = QLabel('File will be saved as: '), QLineEdit(), QLabel()
        label, self.relPathEntry, button = items
        self.relPathEntry.setAlignment(Qt.AlignCenter)
        for j, w in enumerate(items):
            self.vLayouts[j].addWidget(w)
        
        row += 1
        items = (
            QLabel('Segmentation masks folder path: '), 
            widgets.ElidingLineEdit(), 
            widgets.browseFileButton(
                'Browse...',
                title='Select folder containing segmentation masks',
                start_dir=folderPath, openFolder=True
            )
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
            QLabel(f'Channel_{channel_idx+1} name: '), 
            widgets.alphaNumericLineEdit(), 
            widgets.subtractPushButton('Remove channel')
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
        msg.critical(self, 'Select two or more items', txt)
        return None
    
    def updateRelativePath(self, text=''):
        posName = self.posNameEntry.text()
        frameNumber = self.frameNumberEntry.text()
        channelName = self.channelNameLE.text()
        basename = self.baseNameLE.text()
        if basename:
            filename = f'{basename}_{posName}_{channelName}.tif'
        else:
            filename = f'{posName}_{channelName}.tif'
        relPath = f'...{os.sep}Position_1{os.sep}Images{os.sep}{filename}'
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
            self, items, title='Select items', infoTxt='', helpText='', 
            parent=None
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
            helpButton = widgets.helpPushButton('Help...')
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
        msg.information(self, 'Select tables help', txt)
    
    def ok_cb(self):
        self.cancel = False
        self.selectedItemsText = [None]*len(self.listWidget.selectedItems())
        for itemW in self.listWidget.selectedItems():
            idx = int(itemW._nrWidget.currentText()) - 1
            if idx >= len(self.selectedItemsText):
                idx = len(self.selectedItemsText) - 1
            self.selectedItemsText[idx] = itemW._text
        self.close()


class QDialogAutomaticThresholding(QBaseDialog):
    def __init__(self, parent=None, isSegm3D=True):
        super().__init__(parent)

        self.cancel = True

        self.setWindowTitle('Automatic thresholding parameters')

        layout = QVBoxLayout()
        formLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        row = 0
        self.sigmaGaussSpinbox = QDoubleSpinBox()
        self.sigmaGaussSpinbox.setValue(1)
        self.sigmaGaussSpinbox.setMaximum(2**31)
        self.sigmaGaussSpinbox.setAlignment(Qt.AlignCenter)
        formLayout.addWidget(
            QLabel('Gaussian filter sigma (0 to ignore): '), row, 0,
            alignment=Qt.AlignRight
        )
        formLayout.addWidget(self.sigmaGaussSpinbox, row, 1, 1, 2)

        row += 1
        self.threshMethodCombobox = QComboBox()
        self.threshMethodCombobox.addItems([
            'Isodata', 'Li', 'Mean', 'Minimum', 'Otsu', 'Triangle', 'Yen'
        ])
        formLayout.addWidget(
            QLabel('Thresholding algorithm: '), row, 0,
            alignment=Qt.AlignRight
        )
        formLayout.addWidget(self.threshMethodCombobox, row, 1, 1, 2)

        self.segment3Dcheckbox = None
        if isSegm3D:
            row += 1
            formLayout.addWidget(
                QLabel('Segment 3D volume: '), row, 0, alignment=Qt.AlignRight
            )
            group = QButtonGroup()
            group.setExclusive(True)
            self.segment3Dcheckbox = QRadioButton('Yes')
            segmentSliceBySliceCheckbox = QRadioButton('No, segment slice-by-slice')
            group.addButton(self.segment3Dcheckbox)
            group.addButton(segmentSliceBySliceCheckbox)
            formLayout.addWidget(self.segment3Dcheckbox, row, 1)
            formLayout.addWidget(segmentSliceBySliceCheckbox, row, 2)
            self.segment3Dcheckbox.setChecked(True)

        okButton = widgets.okPushButton('Ok')
        cancelButton = widgets.cancelPushButton('Cancel')
        helpButton = widgets.helpPushButton('Help...')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(helpButton)
        buttonsLayout.addWidget(okButton)

        layout.addLayout(formLayout)
        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)

        okButton.clicked.connect(self.ok_cb)
        helpButton.clicked.connect(self.help_cb)
        cancelButton.clicked.connect(self.close)

        self.setLayout(layout)
        self.setFont(font)

        self.configPars = self.loadLastSelection()

    
    def help_cb(self):
        import webbrowser
        url = 'https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html'
        webbrowser.open(url)

    def ok_cb(self):
        self.cancel = False
        self.gaussSigma = self.sigmaGaussSpinbox.value()
        threshMethod = self.threshMethodCombobox.currentText().lower()
        self.threshMethod = f'threshold_{threshMethod}'
        self.segment_kwargs = {
            'gauss_sigma': self.gaussSigma,
            'threshold_method': self.threshMethod,
            'segment_3D_volume': False
        }
        if self.segment3Dcheckbox is not None:
            doSegm3D = self.segment3Dcheckbox.isChecked()
            self.segment_kwargs['segment_3D_volume'] = doSegm3D
        self.close()
    
    def loadLastSelection(self):
        self.ini_path = os.path.join(settings_folderpath, 'last_params_segm_models.ini')
        if not os.path.exists(self.ini_path):
            return

        configPars = config.ConfigParser()
        configPars.read(self.ini_path)

        if 'thresholding.segment' not in configPars.sections():
            return

        section = configPars['thresholding.segment']
        self.sigmaGaussSpinbox.setValue(float(section['gauss_sigma']))

        threshold_method = section['threshold_method']
        Method = threshold_method[10:].capitalize()
        self.threshMethodCombobox.setCurrentText(Method)
        if self.segment3Dcheckbox is None:
            return
        self.segment3Dcheckbox.setChecked(section.getboolean('segment_3D_volume'))

class ApplyTrackTableSelectColumnsDialog(QBaseDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Select columns containing tracking info')
        
        self.cancel = True
        self.mainLayout = QVBoxLayout()

        options = (
            '"Frame index", "Tracked IDs" and "Segmentation mask IDs"<br>',
            '"Frame index", "Tracked IDs", "X coord. centroid", and "Y coord. centroid"'
        )
        self.instructionsText = html_utils.paragraph(
            f"""
            <b>Select which columns</b> contain the tracking information.<br><br>
            You must choose one of the following combinations:<br> 
            {html_utils.to_list(options)}
            Optionally, you can provide the column name containing the parent ID.<br>
            This will allow you to load lineage information into Cell-ACDC. 
            """
        )
        self.mainLayout.addWidget(QLabel(self.instructionsText))

        formLayout = QFormLayout()

        self.frameIndexCombobox = widgets.QCenteredComboBox()
        self.frameIndexCombobox.addItems(df.columns)
        self.frameIndexCheckbox = QCheckBox('1st frame is index 1')
        frameIndexLayout = QHBoxLayout()
        frameIndexLayout.addWidget(self.frameIndexCombobox)
        frameIndexLayout.addWidget(self.frameIndexCheckbox)
        frameIndexLayout.setStretch(0, 2)
        frameIndexLayout.setStretch(1, 0)
        formLayout.addRow(
            'Frame index: ', frameIndexLayout
        )

        self.trackedIDsCombobox = widgets.QCenteredComboBox()
        self.trackedIDsCombobox.addItems(df.columns)
        formLayout.addRow('Tracked IDs: ', self.trackedIDsCombobox)

        items = df.columns.to_list()
        items.insert(0, 'None')
        self.maskIDsCombobox = widgets.QCenteredComboBox()
        self.maskIDsCombobox.addItems(items)
        formLayout.addRow('Segmentation mask IDs: ', self.maskIDsCombobox)

        self.xCentroidCombobox = widgets.QCenteredComboBox()
        self.xCentroidCombobox.addItems(items)
        formLayout.addRow('X coord. centroid: ', self.xCentroidCombobox)

        self.yCentroidCombobox = widgets.QCenteredComboBox()
        self.yCentroidCombobox.addItems(items)
        formLayout.addRow('Y coord. centroid: ', self.yCentroidCombobox)

        self.parentIDcombobox = widgets.QCenteredComboBox()
        self.parentIDcombobox.addItems(items)
        formLayout.addRow('Parent ID (optional): ', self.parentIDcombobox)

        deleteUntrackedLayout = QHBoxLayout()
        self.deleteUntrackedIDsToggle = widgets.Toggle()
        deleteUntrackedLayout.addStretch(1)
        deleteUntrackedLayout.addWidget(self.deleteUntrackedIDsToggle)
        deleteUntrackedLayout.addStretch(1)
        formLayout.addRow('Delete untracked IDs: ', deleteUntrackedLayout)
        
        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        self.mainLayout.addSpacing(30)
        self.mainLayout.addLayout(formLayout)
        self.mainLayout.addSpacing(20)
        self.mainLayout.addLayout(buttonsLayout)

        self.setLayout(self.mainLayout)
        self.setFont(font)
    
    def ok_cb(self):
        self.cancel = False
        self.frameIndexCol = self.frameIndexCombobox.currentText()
        self.trackedIDsCol = self.trackedIDsCombobox.currentText()
        self.maskIDsCol = self.maskIDsCombobox.currentText()
        self.xCentroidCol = self.xCentroidCombobox.currentText()
        self.yCentroidCol = self.yCentroidCombobox.currentText()
        self.deleteUntrackedIDs = self.deleteUntrackedIDsToggle.isChecked()
        if self.maskIDsCol == 'None':
            if self.xCentroidCol == 'None' or self.yCentroidCol == 'None':
                self.warnInvalidSelection()
                return
        else:
            self.xCentroidCol = 'None'
            self.yCentroidCol = 'None'
        self.parentIDcol = self.parentIDcombobox.currentText()
        self.isFirstFrameOne = self.frameIndexCheckbox.isChecked()
        self.close()
    
    def warnInvalidSelection(self):
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        msg.warning(
            self, 'Invalid selection', html_utils.paragraph(
                f'<b>Invalid selection</b><br> {self.instructionsText}'
            )
        )


class QDialogSelectModel(QDialog):
    def __init__(self, parent=None, addSkipSegmButton=False):
        self.cancel = True
        super().__init__(parent)
        self.setWindowTitle('Select model')

        mainLayout = QVBoxLayout()
        topLayout = QVBoxLayout()
        bottomLayout = QHBoxLayout()

        self.mainLayout = mainLayout

        label = QLabel(html_utils.paragraph(
            'Select model to use for segmentation: '
        ))
        # padding: top, left, bottom, right
        label.setStyleSheet("padding:0px 0px 3px 0px;")
        topLayout.addWidget(label, alignment=Qt.AlignCenter)

        listBox = widgets.listWidget()
        models = myutils.get_list_of_models()
        listBox.setFont(font)
        listBox.addItems(models)
        addCustomModelItem = QListWidgetItem('Add custom model...')
        addCustomModelItem.setFont(italicFont)
        listBox.addItem(addCustomModelItem)
        listBox.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        listBox.setCurrentRow(0)
        self.listBox = listBox
        listBox.itemDoubleClicked.connect(self.ok_cb)
        topLayout.addWidget(listBox)

        cancelButton = widgets.cancelPushButton('Cancel')
        okButton = widgets.okPushButton(' Ok ')
        okButton.setShortcut(Qt.Key_Enter)

        bottomLayout.addStretch(1)
        bottomLayout.addWidget(cancelButton)
        bottomLayout.addSpacing(20)
        if addSkipSegmButton:
            skipSegmButton = widgets.SkipPushButton('Skip segmentation')
            bottomLayout.addWidget(skipSegmButton)
            skipSegmButton.clicked.connect(self.skipSegm)
        bottomLayout.addWidget(okButton)
        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        self.setStyleSheet(LISTWIDGET_STYLESHEET)
    
    def skipSegm(self):
        self.cancel = False
        self.selectedModel = 'skip_segmentation'
        self.close()
    
    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_Escape:
            event.ignore()
            return
            
        super().keyPressEvent(event)

    def ok_cb(self, event):
        self.clickedButton = self.sender()
        self.cancel = False
        item = self.listBox.currentItem()
        model = item.text()
        if model == 'Add custom model...':
            modelFilePath = addCustomModelMessages(self)
            if modelFilePath is None:
                return
            myutils.store_custom_model_path(modelFilePath)
            modelName = os.path.basename(os.path.dirname(modelFilePath))
            item = QListWidgetItem(modelName)
            self.listBox.addItem(item)
            self.listBox.setCurrentItem(item)
        elif model == 'Automatic thresholding':
            self.selectedModel = 'thresholding'
            self.close()
        else:
            self.selectedModel = model
            self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.selectedModel = None
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
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

class ViewTextDialog(QBaseDialog):
    def __init__(self, text, parent=None):
        super().__init__(parent)

        mainLayout = QVBoxLayout()

        textViewWidget = QTextEdit()
        textViewWidget.setReadOnly(True)

        textViewWidget.setText(text)

        buttonsLayout = QHBoxLayout()
        okButton = widgets.okPushButton('Ok')

        okButton.clicked.connect(self.close)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(okButton)

        mainLayout.addWidget(textViewWidget)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)
        self.setFont(font)

class startStopFramesDialog(QBaseDialog):
    def __init__(
            self, SizeT, currentFrameNum=0, parent=None,
            windowTitle='Select frame range to segment'
        ):
        super().__init__(parent=parent)

        self.setWindowTitle(windowTitle)

        self.cancel = True

        layout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        self.selectFramesGroupbox = widgets.selectStartStopFrames(
            SizeT, currentFrameNum=currentFrameNum, parent=parent
        )

        okButton = widgets.okPushButton('Ok')
        cancelButton = widgets.cancelPushButton('Cancel')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        layout.addWidget(self.selectFramesGroupbox)
        layout.addLayout(buttonsLayout)
        self.setLayout(layout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        self.setFont(font)

    def ok_cb(self):
        if self.selectFramesGroupbox.warningLabel.text():
            return
        else:
            self.startFrame = self.selectFramesGroupbox.startFrame_SB.value()
            self.stopFrame = self.selectFramesGroupbox.stopFrame_SB.value()
            self.cancel = False
            self.close()
    
    def show(self, block=False):
        super().show(block=False)

        self.resize(int(self.width()*1.5), self.height())

        if block:
            super().show(block=True)

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
            'font-size:13px; padding:5px 0px 0px 0px;'
        )

        okButton = widgets.okPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = widgets.cancelPushButton('Cancel')

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
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
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

        okButton = widgets.okPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = widgets.cancelPushButton('Cancel')

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
        self.entriesTxt = [self.formLayout.itemAt(i, 1).widget().text()
                           for i in range(len(self.entriesLabels))]
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

class QDialogMetadata(QDialog):
    def __init__(
            self, SizeT, SizeZ, TimeIncrement,
            PhysicalSizeZ, PhysicalSizeY, PhysicalSizeX,
            ask_SizeT, ask_TimeIncrement, ask_PhysicalSizes,
            parent=None, font=None, imgDataShape=None, posData=None,
            singlePos=False, askSegm3D=True, additionalValues=None,
            forceEnableAskSegm3D=False, SizeT_metadata=None, 
            SizeZ_metadata=None, basename=''
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
        self.setWindowTitle('Image properties')

        mainLayout = QVBoxLayout()
        gridLayout = QGridLayout()
        # formLayout = QFormLayout()
        buttonsLayout = QGridLayout()

        if imgDataShape is not None:
            label = QLabel(
                html_utils.paragraph(
                    f'<i>Image data shape</i> = <b>{imgDataShape}</b><br>'
                )
            )
            mainLayout.addWidget(label, alignment=Qt.AlignCenter)

        row = 0
        self.basenameLineEdit = None
        if basename:
            gridLayout.addWidget(
                QLabel('Basename (read-only)'), row, 0, alignment=Qt.AlignRight
            )
            self.basenameLineEdit = QLineEdit()
            self.basenameLineEdit.setReadOnly(True)
            self.basenameLineEdit.setText(basename)
            minWidth = (
                self.basenameLineEdit.fontMetrics()
                .boundingRect(basename).width() + 10
            )
            self.basenameLineEdit.setMinimumWidth(minWidth)
            self.basenameLineEdit.setAlignment(Qt.AlignCenter)
            gridLayout.addWidget(self.basenameLineEdit, row, 1)
            row += 1
        
        gridLayout.addWidget(
            QLabel('Number of frames (SizeT)'), row, 0, alignment=Qt.AlignRight
        )
        self.SizeT_SpinBox = QSpinBox()
        self.SizeT_SpinBox.setMinimum(1)
        self.SizeT_SpinBox.setMaximum(2147483647)
        SizeTinfoButton = widgets.infoPushButton()
        self.allowEditSizeTcheckbox = QCheckBox('Let me edit it')
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
        gridLayout.setColumnStretch(2,0)
        gridLayout.addWidget(self.allowEditSizeTcheckbox, row, 3)
        gridLayout.setColumnStretch(3,0)

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
        self.isSegm3DLabel = QLabel('Work with 3D segmentation masks (z-stack)')
        gridLayout.addWidget(
            self.isSegm3DLabel, row, 0, alignment=Qt.AlignRight
        )
        gridLayout.addWidget(
            self.isSegm3Dtoggle, row, 1, alignment=Qt.AlignCenter
        )
        self.infoButtonSegm3D = QPushButton(self)
        self.infoButtonSegm3D.setCursor(Qt.WhatsThisCursor)
        self.infoButtonSegm3D.setIcon(QIcon(":info.svg"))
        gridLayout.addWidget(
            self.infoButtonSegm3D, row, 2, alignment=Qt.AlignLeft
        )
        self.infoButtonSegm3D.clicked.connect(self.infoSegm3D)
        if SizeZ == 1 or not askSegm3D:
            self.isSegm3DLabel.hide()
            self.isSegm3Dtoggle.hide()
            self.infoButtonSegm3D.hide()

        self.SizeZvalueChanged(SizeZ)

        self.additionalFieldsWidgets = []
        addFieldButton = widgets.addPushButton('Add custom field')
        addFieldInfoButton = widgets.infoPushButton()
        addFieldInfoButton.clicked.connect(self.showAddFieldInfo)
        addFieldButton.clicked.connect(self.addField)
        addFieldLayout = QHBoxLayout()
        addFieldLayout.addStretch(1)
        addFieldLayout.addWidget(addFieldButton)
        addFieldLayout.addWidget(addFieldInfoButton)
        addFieldLayout.addStretch(1)

        if singlePos:
            okTxt = 'Apply only to this Position'
        else:
            okTxt = 'Ok for loaded Positions'
        okButton = widgets.okPushButton(okTxt)
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

        cancelButton = widgets.cancelPushButton('Cancel')

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
        msg.information(
            self, 'Why is the number of frames grayed out?', txt
        )

    def addAdditionalValues(self, values):
        if values is None:
            return

        for i, (name, value) in enumerate(values.items()):
            self.addField()
            nameWidget = self.additionalFieldsWidgets[i]['nameWidget']
            valueWidget = self.additionalFieldsWidgets[i]['valueWidget']
            nameWidget.setText(str(name).strip('__'))
            valueWidget.setText(str(value))

    def addField(self):
        nameWidget = QLineEdit()
        nameWidget.setAlignment(Qt.AlignCenter)
        valueWidget = QLineEdit()
        valueWidget.setAlignment(Qt.AlignCenter)
        removeButton = widgets.delPushButton()

        fieldLayout = QGridLayout()
        fieldLayout.addWidget(QLabel('Name'), 0, 0)
        fieldLayout.addWidget(nameWidget, 1, 0)
        fieldLayout.addWidget(QLabel('Value'), 0, 1)
        fieldLayout.addWidget(valueWidget, 1, 1)
        fieldLayout.addWidget(removeButton, 1, 2)

        self.additionalFieldsWidgets.append({
            'nameWidget': nameWidget,
            'valueWidget': valueWidget,
            'removeButton': removeButton,
            'layout': fieldLayout
        })

        idx = len(self.additionalFieldsWidgets)-1
        removeButton.clicked.connect(partial(self.removeField, idx))

        row = self.mainLayout.count()-3
        self.mainLayout.insertLayout(row, fieldLayout)

    def removeField(self, idx):
        widgets = self.additionalFieldsWidgets[idx]

        layoutToRemove = widgets['layout']
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
        msg.information(self, 'Add field info', txt)

    def infoSegm3D(self):
        txt = (
            'Cell-ACDC supports both <b>2D and 3D segmentation</b>. If your data '
            'also have a time dimension, then you can choose to segment '
            'a specific z-slice (2D segmentation mask per frame) or all of them '
            '(3D segmentation mask per frame)<br><br>'
            'In any case, if you choose to activate <b>3D segmentation</b> then the '
            'segmentation mask will have the <b>same number of z-slices '
            'of the image data</b>.<br><br>'
            'Additionally, in the model parameters window, you will be able '
            'to choose if you want to segment the <b>entire 3D volume at once</b> '
            'or use the <b>2D model on each z-slice</b>, one by one.<br><br>'
            '<i>NOTE: if the toggle is disabled it means you already '
            'loaded segmentation data and the shape cannot be changed now.<br>'
            'if you need to start with a blank segmentation, '
            'use the "Create a new segmentation file" button instead of the '
            '"Load folder" button.'
            '</i>'
        )
        msg = widgets.myMessageBox()
        msg.setIcon()
        msg.setWindowTitle(f'3D segmentation info')
        msg.addText(html_utils.paragraph(txt))
        msg.addButton('   Ok   ')
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
            self, 'WARNING: Edinting saved metadata', txt, 
            buttonsTexts=('Cancel', 'No', 'Yes, edit the metadata')
        )
        return msg.clickedButton == yesButton

    def ok_cb(self, checked=False):
        self.cancel = False
        self.SizeT = self.SizeT_SpinBox.value()
        self.SizeZ = self.SizeZ_SpinBox.value()

        if self.SizeT_metadata is not None:
            if self.SizeT != self.SizeT_metadata:
                proceed = self.warnEditingMetadata(
                    self.SizeT, self.SizeT_metadata, 'frames'
                )
                if not proceed:
                    return
        
        if self.SizeZ_metadata is not None:
            if self.SizeZ != self.SizeZ_metadata:
                proceed = self.warnEditingMetadata(
                    self.SizeZ, self.SizeZ_metadata, 'z-slices'
                )
                if not proceed:
                    return


        self.isSegm3D = self.isSegm3Dtoggle.isChecked()

        self.TimeIncrement = self.TimeIncrementSpinBox.value()
        self.PhysicalSizeX = self.PhysicalSizeXSpinBox.value()
        self.PhysicalSizeY = self.PhysicalSizeYSpinBox.value()
        self.PhysicalSizeZ = self.PhysicalSizeZSpinBox.value()
        self._additionalValues = {
            f"__{field['nameWidget'].text()}":field['valueWidget'].text()
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
                        err_msg = html_utils.paragraph(
                            'The below file is open in another app '
                            '(Excel maybe?).<br><br>'
                            f'<code>{acdc_df_path}</code><br><br>'
                            'Close file and then press "Ok".'
                        )
                        msg = widgets.myMessageBox()
                        msg.critical(self, 'Permission denied', err_msg)
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
            txt = (f"""
                You loaded <b>4D data</b>, hence the number of frames MUST be
                <b>{T}</b><br> and the number of z-slices MUST be <b>{Z}</b>.<br><br>
                What do you want to do?
            """)
        if not valid3D:
            txt = (f"""
                You loaded <b>3D data</b>, hence either the number of frames or 
                the number of z-slices is <b>{TorZ}</b>.<br><br>
                However, if the number of frames is greater than 1 then the<br>
                number of z-slices MUST be 1, and vice-versa.<br><br>
                What do you want to do?
            """)

        if not valid2D:
            txt = (f"""
                You loaded <b>2D data</b>, hence the number of frames MUST be <b>1</b>
                and the number of z-slices MUST be <b>1</b>.<br><br>
                What do you want to do?
            """)
        
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(txt)
        
        continueButton = widgets.okPushButton('Continue anyway')
        correctButton = widgets.editPushButton('Let me correct')
        
        msg.warning(
            self, 'Shape-metadata mismatch', txt,
            buttonsTexts=(continueButton, correctButton)
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
        if hasattr(self, 'loop'):
            self.loop.exit()

class QCropZtool(QBaseDialog):
    sigClose = Signal()
    sigZvalueChanged = Signal(str, int)
    sigReset = Signal()
    sigCrop = Signal(int, int)

    def __init__(
            self, SizeZ, cropButtonText='Apply crop', parent=None, 
            addDoNotShowAgain=False, title='Select z-slices'
        ):
        super().__init__(parent)

        self.cancel = True

        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        self.SizeZ = SizeZ
        self.numDigits = len(str(self.SizeZ))

        self.setWindowTitle(title)

        layout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        self.lowerZscrollbar = QScrollBar(Qt.Horizontal)
        self.lowerZscrollbar.setMaximum(SizeZ-1)
        s = str(1).zfill(self.numDigits)
        self.lowerZscrollbar.label = QLabel(f'{s}/{SizeZ}')

        self.upperZscrollbar = QScrollBar(Qt.Horizontal)
        self.upperZscrollbar.setValue(SizeZ-1)
        self.upperZscrollbar.setMaximum(SizeZ-1)
        self.upperZscrollbar.label = QLabel(f'{SizeZ}/{SizeZ}')

        cancelButton = widgets.cancelPushButton('Cancel')
        cropButton = widgets.okPushButton(cropButtonText)
        buttonsLayout.addWidget(cropButton)
        buttonsLayout.addWidget(cancelButton)

        row = 0
        layout.addWidget(
            QLabel('Lower z-slice  '), row, 0, alignment=Qt.AlignRight
        )
        layout.addWidget(
            self.lowerZscrollbar.label, row, 1, alignment=Qt.AlignRight
        )
        layout.addWidget(self.lowerZscrollbar, row, 2)

        row += 1
        layout.setRowStretch(row, 5)

        row += 1
        layout.addWidget(
            QLabel('Upper z-slice  '), row, 0, alignment=Qt.AlignRight
        )
        layout.addWidget(
            self.upperZscrollbar.label, row, 1, alignment=Qt.AlignRight
        )
        layout.addWidget(self.upperZscrollbar, row, 2)

        row += 1
        if addDoNotShowAgain:
            self.doNotShowAgainCheckbox = QCheckBox('Do not ask again')
            layout.addWidget(
                self.doNotShowAgainCheckbox, row, 2, alignment=Qt.AlignLeft
            )
            row += 1

        layout.addLayout(buttonsLayout, row, 2, alignment=Qt.AlignRight)

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
        self.cancel = False
        low_z = self.lowerZscrollbar.value()
        high_z = self.upperZscrollbar.value()
        self.sigCrop.emit(low_z, high_z)
        self.close()

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

        s = str(value+1).zfill(self.numDigits)
        self.sender().label.setText(f'{s}/{self.SizeZ}')
        self.sigZvalueChanged.emit(which, value)

    def showEvent(self, event):
        self.resize(int(self.width()*1.5), self.height())

    def closeEvent(self, event):
        super().closeEvent(event)
        self.sigClose.emit()

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
        self.bkgrThreshSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.bkgrThreshSlider.setTickInterval(10)
        paramsLayout.addWidget(self.bkgrThreshSlider, row, 0)

        row += 1
        foregrQSLabel = QLabel('Foreground threshold:')
        # padding: top, left, bottom, right
        foregrQSLabel.setStyleSheet("font-size:13px; padding:5px 0px 0px 0px;")
        paramsLayout.addWidget(foregrQSLabel, row, 0)
        row += 1
        self.foregrThreshValLabel = QLabel('0.95')
        paramsLayout.addWidget(self.foregrThreshValLabel, row, 1)
        self.foregrThreshSlider = QSlider(Qt.Horizontal)
        self.foregrThreshSlider.setMinimum(1)
        self.foregrThreshSlider.setMaximum(100)
        self.foregrThreshSlider.setValue(95)
        self.foregrThreshSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
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
        font = QFont()
        font.setPixelSize(12)
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
        img = self.mainWindow.getDisplayedImg1()
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
            lab = skimage.morphology.remove_small_objects(lab, min_size=5)
        posData = self.mainWindow.data[self.mainWindow.pos_i]
        posData.lab = lab
        return t1-t0

    def computeSegmAndPlot(self):
        deltaT = self.computeSegm()

        posData = self.mainWindow.data[self.mainWindow.pos_i]

        self.mainWindow.update_rp()
        self.mainWindow.tracking(enforce=True)
        self.mainWindow.updateAllImages()
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
        self.mainWindow.updateAllImages()

class FutureFramesAction_QDialog(QDialog):
    def __init__(
            self, frame_i, last_tracked_i, change_txt,
            applyTrackingB=False, parent=None, 
            addApplyAllButton=False
        ):
        self.decision = None
        self.last_tracked_i = last_tracked_i
        super().__init__(parent)
        self.setWindowTitle('Future frames action?')

        mainLayout = QVBoxLayout()
        txtLayout = QVBoxLayout()
        doNotShowLayout = QVBoxLayout()
        buttonsLayout = QVBoxLayout()

        txt = html_utils.paragraph(
            'You already visited/checked future frames '
            f'{frame_i+1}-{last_tracked_i+1}.<br><br>'
            f'The requested <b>"{change_txt}"</b> change might result in<br>'
            '<b>NON-correct segmentation/tracking</b> for those frames.<br>'
        )

        txtLabel = QLabel(txt)
        txtLabel.setAlignment(Qt.AlignCenter)
        txtLayout.addWidget(txtLabel, alignment=Qt.AlignCenter)

        options = [
            f'Apply the "{change_txt}" <b>only to current frame and re-initialize</b><br>'
            'the future frames to the segmentation file present<br>'
            'on the hard drive.',
            'Apply <b>only to this frame and keep the future frames</b> as they are.',
            'Apply the change to <b>ALL visited/checked future frames</b>.'
        ]
        if addApplyAllButton:
            options.append('Apply to <b>ALL future frames including unvisited ones</b>.')
        if applyTrackingB:
            options.append('Repeat ONLY tracking for all future frames (RECOMMENDED)')

        infoTxt = html_utils.paragraph(
           f'Choose <b>one of the following options:</b>'
           f'{html_utils.to_list(options, ordered=True)}'
        )

        infotxtLabel = QLabel(infoTxt)
        txtLayout.addWidget(infotxtLabel, alignment=Qt.AlignCenter)

        noteLayout = QHBoxLayout()
        noteTxt = html_utils.paragraph(
            'Only changes applied to current frame can be undone.<br>'
            'Changes applied to <b>future frames CANNOT be UNDONE</b><br>'
        )
        noteLayout.addWidget(
            QLabel(html_utils.paragraph('NOTE:')), alignment=Qt.AlignTop
        )
        noteTxtLabel = QLabel(noteTxt)
        noteLayout.addWidget(noteTxtLabel)
        noteLayout.addStretch(1)
        txtLayout.addSpacing(10)
        txtLayout.addLayout(noteLayout)

        # Do not show this message again checkbox
        doNotShowCheckbox = QCheckBox(
            'Remember my choice and do not show this message again')
        doNotShowLayout.addWidget(doNotShowCheckbox)
        doNotShowLayout.setContentsMargins(50, 0, 0, 10)
        self.doNotShowCheckbox = doNotShowCheckbox

        apply_and_reinit_b = widgets.reloadPushButton(
            ' 1. Apply only to this frame and re-initialize future frames'
        )

        self.apply_and_reinit_b = apply_and_reinit_b
        buttonsLayout.addWidget(apply_and_reinit_b)

        apply_and_NOTreinit_b = widgets.currentPushButton(
            ' 2. Apply only to this frame and keep future frames as they are'
        )
        self.apply_and_NOTreinit_b = apply_and_NOTreinit_b
        buttonsLayout.addWidget(apply_and_NOTreinit_b)

        apply_to_all_visited_b = widgets.futurePushButton(
            ' 3. Apply to all future VISITED frames'
        )
        self.apply_to_all_visited_b = apply_to_all_visited_b
        buttonsLayout.addWidget(apply_to_all_visited_b)

        
        if addApplyAllButton:
            apply_to_all_b = QPushButton(
                ' 4. Apply to ALL future frames (including unvisted)'
            )
            apply_to_all_b.setIcon(QIcon(':arrow_future_all.svg'))
            self.apply_to_all_b = apply_to_all_b
            buttonsLayout.addWidget(apply_to_all_b)

        self.applyTrackingButton = None
        if applyTrackingB:
            n = '5' if addApplyAllButton else '4'
            applyTrackingButton = QPushButton(
                f' {n}. Repeat ONLY tracking for all future frames'
            )
            applyTrackingButton.setIcon(QIcon(':repeat-tracking.svg'))
            self.applyTrackingButton = applyTrackingButton
            buttonsLayout.addWidget(applyTrackingButton)

        buttonsLayout.setContentsMargins(20, 0, 20, 0)

        self.formLayout = QFormLayout()

        ButtonsGroup = QButtonGroup(self)
        ButtonsGroup.addButton(apply_and_reinit_b)
        ButtonsGroup.addButton(apply_and_NOTreinit_b)
        ButtonsGroup.addButton(apply_to_all_visited_b)
        if addApplyAllButton:
            ButtonsGroup.addButton(apply_to_all_b)
        if applyTrackingB:
            ButtonsGroup.addButton(applyTrackingButton)
        
        mainLayout.addLayout(txtLayout)
        mainLayout.addLayout(doNotShowLayout)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addLayout(self.formLayout)
        mainLayout.addStretch(1)
        self.mainLayout = mainLayout
        self.setLayout(mainLayout)

        # Connect events
        ButtonsGroup.buttonClicked.connect(self.buttonClicked)
        self.ButtonsGroup = ButtonsGroup

        # self.setModal(True)

    def buttonClicked(self, button):
        if button == self.apply_and_reinit_b:
            self.decision = 'apply_and_reinit'
            self.endFrame_i = None
        elif button == self.apply_and_NOTreinit_b:
            self.decision = 'apply_and_NOTreinit'
            self.endFrame_i = None
        elif button == self.apply_to_all_visited_b:
            self.decision = 'apply_to_all_visited'
            self.endFrame_i = self.last_tracked_i
        elif button == self.applyTrackingButton:
            self.decision = 'only_tracking'
            self.endFrame_i = self.last_tracked_i
        elif button == self.apply_to_all_b:
            self.decision = 'apply_to_all'
            self.endFrame_i = self.last_tracked_i
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()
        for button in self.ButtonsGroup.buttons():
            button.setMinimumHeight(int(button.height()*1.2))
        if hasattr(self, 'apply_to_all_b'):
            iconHeight = self.apply_to_all_b.iconSize().height()
            self.apply_to_all_b.setIconSize(QSize(iconHeight*2, iconHeight))
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class ComputeMetricsErrorsDialog(QBaseDialog):
    def __init__(
            self, errorsDict, log_path='', parent=None, 
            log_type='custom_metrics'
        ):
        super().__init__(parent)

        self.errorsDict = errorsDict

        layout = QGridLayout()

        self.setWindowTitle('Errors summary')
        
        label = QLabel(self)
        standardIcon = getattr(QStyle, 'SP_MessageBoxWarning')
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        label.setPixmap(pixmap)
        layout.addWidget(label, 0, 0, alignment=Qt.AlignTop)

        if log_type == 'custom_metrics':
            infoText = ("""
                When computing <b>custom metrics</b> the following metrics 
                were <b>ignored</b> because they raised an <b>error</b>.<br><br>
            """)
        elif log_type == 'standard_metrics':
            infoText = ("""
                Some or all of the <b>standard metrics</b> were <b>NOT saved</b> 
                because Cell-ACDC encoutered the following errors.<br><br>
            """)
        elif log_type == 'region_props':
            rp_url = 'https://scikit-image.org/docs/0.18.x/api/skimage.measure.html#skimage.measure.regionprops'
            rp_href = f'<a href="{rp_url}">skimage.measure.regionprops</a>'
            infoText = (f"""
                <b>Region properties</b> were <b>NOT saved</b> because Cell-ACDC 
                encoutered the following errors.<br>
                Region properties are calculated using the <code>scikit-image</code> 
                function called <code>{rp_href}</code>.<br><br>
            """)
        elif log_type == 'missing_annot':
            infoText = ("""
                The following Positions were <b>SKIPPED</b> because they did 
                <b>not have cell cycle annotations</b>.<br><br>
                To add lineage tree information you first need to <b>do the 
                cell cycle analysis</b> in module 3 "Main GUI".<br><br>
            """)
        else:
            infoText = ("""
                Process raised the errors listed below.<br><br>
            """)

        github_issues_href = f'<a href={issues_url}>here</a>'   
        noteText = (f"""
            NOTE: If you <b>need help</b> understanding these errors you can 
            <b>open an issue</b> on our github page {github_issues_href}.
        """)
   
        infoLabel = QLabel(html_utils.paragraph(f'{infoText}{noteText}'))
        infoLabel.setOpenExternalLinks(True)
        layout.addWidget(infoLabel, 0, 1)

        scrollArea = QScrollArea()
        scrollAreaWidget = QWidget()  
        textLayout = QVBoxLayout()
        for func_name, traceback_format in errorsDict.items():
            nameLabel = QLabel(f'<b>{func_name}</b>: ')
            errorMessage = f'\n{traceback_format}'
            errorLabel = QLabel(errorMessage)
            errorLabel.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
            )
            # errorLabel.setStyleSheet("background-color: white")
            errorLabel.setFrameShape(QFrame.Shape.Panel)
            errorLabel.setFrameShadow(QFrame.Shadow.Sunken)
            textLayout.addWidget(nameLabel)
            textLayout.addWidget(errorLabel)
            textLayout.addStretch(1)
        
        scrollAreaWidget.setLayout(textLayout)
        scrollArea.setWidget(scrollAreaWidget)
        
        layout.addWidget(scrollArea, 1, 1)

        buttonsLayout = QHBoxLayout()
        showLogButton = widgets.showInFileManagerButton('Show log file...')
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(showLogButton)

        copyButton = widgets.copyPushButton('Copy error message')
        copyButton.clicked.connect(self.copyErrorMessage)
        buttonsLayout.addWidget(copyButton)
        self.copyButton = copyButton
        self.copyButton.text = 'Copy error message'
        self.copyButton.icon = self.copyButton.icon()
        
        okButton = widgets.okPushButton(' Ok ')
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        showLogButton.clicked.connect(partial(myutils.showInExplorer, log_path))
        okButton.clicked.connect(self.close)
        layout.setVerticalSpacing(10)
        layout.addLayout(buttonsLayout, 2, 1)

        self.setLayout(layout)
        self.setFont(font)
    
    def copyErrorMessage(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        copiedText = ''
        for _, traceback_format in self.errorsDict.items():
            errorBlock = f'{"="*30}\n{traceback_format}{"*"*30}'
            copiedText = f'{copiedText}{errorBlock}'
        cb.setText(copiedText, mode=cb.Clipboard)
        print('Error message copied.')
        self.copyButton.setIcon(QIcon(':okButton.svg'))
        self.copyButton.setText(' Copied to clipboard!')
        QTimer.singleShot(2000, self.restoreCopyButton)
    
    def restoreCopyButton(self):
        self.copyButton.setText(self.copyButton.text)
        self.copyButton.setIcon(self.copyButton.icon)
    
    def showEvent(self, a0) -> None:
        self.copyButton.setFixedWidth(self.copyButton.width())
        return super().showEvent(a0)

class PostProcessSegmParams(QGroupBox):
    valueChanged = Signal(object)
    editingFinished = Signal()

    def __init__(
            self, title, posData, 
            useSliders=False, 
            parent=None, 
            maxSize=None, 
            force_postprocess_2D=False
        ):
        QGroupBox.__init__(self, title, parent)
        SizeZ = posData.SizeZ
        self.isSegm3D = posData.isSegm3D
        self.channelName = posData.user_ch_name
        self.useSliders = useSliders
        self.force_postprocess_2D = force_postprocess_2D
        if maxSize is None:
            maxSize=2147483647

        layout = QGridLayout()

        self.controlWidgets = []

        row = 0
        label = QLabel("Minimum area (pixels) ")
        layout.addWidget(label, row, 0, alignment=Qt.AlignRight)

        minSize_SB = widgets.PostProcessSegmWidget(
            1, 1000, 10, useSliders, label=label
        )
        
        txt = (
            '<b>Area</b> is the total number of pixels in the segmented object.'
        )

        layout.addWidget(minSize_SB, row, 1)
        infoButton = widgets.infoPushButton()
        infoButton.clicked.connect(self.showInfo)
        infoButton.tooltip = txt
        infoButton.name = 'area'
        infoButton.desc = f'less than "{label.text()}"'
        layout.addWidget(infoButton, row, 2)
        self.minSize_SB = minSize_SB
        self.controlWidgets.append(minSize_SB)

        # minSize_SB.disableThisCheckbox = QCheckBox('Disable this filter')
        # layout.addWidget(minSize_SB.disableThisCheckbox, row, 3)

        row += 1
        label = QLabel("Minimum solidity (0-1) ")
        layout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        minSolidity_DSB = widgets.PostProcessSegmWidget(
            0, 1.0, 0.5, useSliders, isFloat=True, normalize=True,
            label=label
        )
        minSolidity_DSB.setValue(0.5)
        minSolidity_DSB.setSingleStep(0.1)
        self.controlWidgets.append(minSolidity_DSB)

        txt = (
            '<b>Solidity</b> is a measure of convexity. A solidity of 1 means '
            'that the shape is fully convex (i.e., equal to the convex hull). '
            'As solidity approaches 0 the object is more concave.<br>'
            'Write 0 for ignoring this parameter.'
        )

        layout.addWidget(minSolidity_DSB, row, 1)
        infoButton = widgets.infoPushButton()
        infoButton.clicked.connect(self.showInfo)
        infoButton.tooltip = txt
        infoButton.name = 'solidity'
        infoButton.desc = f'less than "{label.text()}"'
        layout.addWidget(infoButton, row, 2)
        self.minSolidity_DSB = minSolidity_DSB

        row += 1
        label = QLabel("Max elongation (1=circle) ")
        layout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        maxElongation_DSB = widgets.PostProcessSegmWidget(
            0, 100, 3, useSliders, isFloat=True, normalize=False,
            label=label
        )
        maxElongation_DSB.setDecimals(1)
        maxElongation_DSB.setSingleStep(1.0)

        txt = (
            '<b>Elongation</b> is the ratio between major and minor axis lengths. '
            'An elongation of 1 is like a circle.<br>'
            'Write 0 for ignoring this parameter.'
        )

        layout.addWidget(maxElongation_DSB, row, 1)
        infoButton = widgets.infoPushButton()
        infoButton.clicked.connect(self.showInfo)
        infoButton.tooltip = txt
        infoButton.name = 'elongation'
        infoButton.desc = f'greater than "{label.text()}"'
        layout.addWidget(infoButton, row, 2)
        self.maxElongation_DSB = maxElongation_DSB
        self.controlWidgets.append(maxElongation_DSB)

        if self.isSegm3D:
            row += 1
            label = QLabel("Minimum number of z-slices ")
            layout.addWidget(label, row, 0, alignment=Qt.AlignRight)
            minObjSizeZ_SB = widgets.PostProcessSegmWidget(
                0, SizeZ, 3, useSliders, isFloat=False, normalize=False,
                label=label
            )

            txt = (
                '<b>Minimum number of z-slices</b> per object.'
            )

            layout.addWidget(minObjSizeZ_SB, row, 1)
            infoButton = widgets.infoPushButton()
            infoButton.clicked.connect(self.showInfo)
            infoButton.tooltip = txt
            infoButton.name = 'number of z-slices'
            infoButton.desc = f'less than "{label.text()}"'
            layout.addWidget(infoButton, row, 2)
            self.minObjSizeZ_SB = minObjSizeZ_SB
            self.controlWidgets.append(minObjSizeZ_SB)
        else:
            self.minObjSizeZ_SB = widgets.NoneWidget()

        row += 1
        addCustomFeatureLayout = QHBoxLayout()
        self.addCustomFeaturesButton = widgets.setPushButton(
            'Select custom features for post-processing...',
        )
        addCustomFeatureLayout.addWidget(self.addCustomFeaturesButton)
        addCustomFeatureLayout.addStretch(1)
        self.selectedFeaturesDialog = SelectFeaturesRangeDialog(
            posData=posData, parent=self, 
            force_postprocess_2D=force_postprocess_2D
        )
        self.selectedFeaturesDialog.hide()
        self.addCustomFeaturesButton.clicked.connect(
            self.selectedFeaturesDialog.show
        )
        self.selectedFeaturesDialog.sigValueChanged.connect(self.onValueChanged)
        
        layout.addLayout(addCustomFeatureLayout, row, 0, 1, 2)
        
        layout.setColumnStretch(1, 2)
        layout.setRowStretch(row+1, 1)

        self.setLayout(layout)

        for widget in self.controlWidgets:
            widget.valueChanged.connect(self.onValueChanged)
            widget.editingFinished.connect(self.onEditingFinished)
    
    def selectedFeaturesRange(self):
        return self.selectedFeaturesDialog.groupbox.selectedFeaturesRange()

    def groupedFeatures(self):
        return self.selectedFeaturesDialog.groupbox.groupedFeatures()
    
    def restoreDefault(self):
        self.minSolidity_DSB.setValue(0.5)
        self.minSize_SB.setValue(10)
        self.maxElongation_DSB.setValue(3)
        self.minObjSizeZ_SB.setValue(3)
    
    def restoreFromKwargs(self, kwargs):
        for name, value in kwargs.items():
            if name == 'min_solidity':
                self.minSolidity_DSB.setValue(value)
            elif name == 'min_area':
                self.minSize_SB.setValue(value)
            elif name == 'max_elongation':
                self.maxElongation_DSB.setValue(value)
            elif name == 'min_obj_no_zslices':
                self.minObjSizeZ_SB.setValue(value)
    
    def kwargs(self):
        kwargs = {
            'min_solidity': self.minSolidity_DSB.value(),
            'min_area': self.minSize_SB.value(),
            'max_elongation': self.maxElongation_DSB.value(),
            'min_obj_no_zslices': self.minObjSizeZ_SB.value()
        }
        return kwargs
    
    def onValueChanged(self, value):
        self.valueChanged.emit(value)
    
    def onEditingFinished(self):
        self.editingFinished.emit()
    
    def showInfo(self):
        title = f'{self.sender().text()} info'
        tooltip = self.sender().tooltip
        name = self.sender().name
        desc = self.sender().desc
        txt = (f"""
            The post-processing step is applied to the output of the 
            segmentation model.<br><br>
            During this step, Cell-ACDC will remove all the objects with {name}
            <b>{desc}</b>.<br><br>
            {tooltip}    
        """)
        if self.isCheckable():
            note = f""""
                You can deactivate this step by un-checking the checkbox 
                called "Post-processing parameters".
            """
            txt = f'{txt}{note}'
        msg = widgets.myMessageBox(showCentered=False)
        msg.information(self, title, html_utils.paragraph(txt))

class PostProcessSegmDialog(QBaseDialog):
    sigClosed = Signal()
    sigValueChanged = Signal(object, object)
    sigEditingFinished = Signal()
    sigApplyToAllFutureFrames = Signal(object, object, object)

    def __init__(
            self, posData, 
            mainWin=None, 
            useSliders=True, 
            maxSize=None
        ):
        super().__init__(mainWin)
        self.cancel = True
        self.mainWin = mainWin
        self.isTimelapse = False
        self.isMultiPos = False
        if mainWin is not None:
            self.isMultiPos = len(self.mainWin.data) > 1
            self.isTimelapse = self.mainWin.data[self.mainWin.pos_i].SizeT > 1

        self.setWindowTitle('Post-processing segmentation parameters')
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        mainLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        self.postProcessGroupbox = PostProcessSegmParams(
            'Post-processing parameters', posData,
            useSliders=useSliders, 
            maxSize=maxSize,
            parent=mainWin
        )

        self.postProcessGroupbox.valueChanged.connect(self.valueChanged)
        self.postProcessGroupbox.editingFinished.connect(self.onEditingFinished)

        if self.isTimelapse:
            applyAllButton = widgets.futurePushButton(
                'Apply to all frames...'
            )
            applyAllButton.clicked.connect(self.applyAll_cb)
            applyButton = widgets.okPushButton(
                'Apply', isDefault=False
            )
            applyButton.clicked.connect(self.apply_cb)
        elif self.isMultiPos:
            applyAllButton = widgets.futurePushButton(
                'Apply to all Positions...'
            )
            applyAllButton.clicked.connect(self.applyAll_cb)
            applyButton = widgets.okPushButton('Apply', isDefault=False)
            applyButton.clicked.connect(self.apply_cb)
        else:
            applyAllButton = widgets.okPushButton('Apply', isDefault=False)
            applyAllButton.clicked.connect(self.ok_cb)
            applyButton = None

        cancelButton = widgets.cancelPushButton('Cancel')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        if applyButton is not None:
            buttonsLayout.addWidget(applyButton)
        buttonsLayout.addWidget(applyAllButton)
        
        emitEditingFinishedButton = widgets.okPushButton()
        buttonsLayout.addWidget(emitEditingFinishedButton)
        emitEditingFinishedButton.hide()
        buttonsLayout.setContentsMargins(0,10,0,0)

        mainLayout.addWidget(self.postProcessGroupbox)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        cancelButton.clicked.connect(self.cancel_cb)

        if mainWin is not None:
            self.setPosData()
    
    def keyPressEvent(self, event) -> None:
        return super().keyPressEvent(event)

    def setPosData(self):
        if self.mainWin is None:
            return

        self.mainWin.storeUndoRedoStates(False)
        self.posData = self.mainWin.data[self.mainWin.pos_i]
        self.origLab = self.posData.lab.copy()
        self.origRp = skimage.measure.regionprops(self.origLab)
        self.origObjs = {obj.label:obj for obj in self.origRp}

    def valueChanged(self, value):
        lab, delObjs = self.apply()
        self.sigValueChanged.emit(lab, delObjs)
        
    def apply(self, origLab=None):
        self.mainWin.warnEditingWithCca_df('post-processing segmentation mask')

        if origLab is None:
            origLab = self.origLab.copy()

        lab, delIDs = core.post_process_segm(
            origLab, return_delIDs=True, **self.postProcessGroupbox.kwargs()
        )
        
        if self.postProcessGroupbox.selectedFeaturesRange():
            lab, custom_delIDs = features.custom_post_process_segm(
                self.posData, 
                self.postProcessGroupbox.groupedFeatures(), 
                lab, 
                self.posData.img_data[self.posData.frame_i], 
                self.posData.frame_i, 
                self.posData.filename, 
                self.posData.user_ch_name,
                self.postProcessGroupbox.selectedFeaturesRange(), 
                return_delIDs=True
            )
            delIDs.extend(custom_delIDs)
        
        delObjs = {delID:self.origObjs[delID] for delID in delIDs}
        return lab, delObjs

    def onEditingFinished(self):
        self.sigEditingFinished.emit()

    def ok_cb(self):
        self.cancel = False
        self.apply()
        self.onEditingFinished()
        self.close()

    def apply_cb(self):
        self.cancel = False
        self.apply()
        self.onEditingFinished()

    def applyAll_cb(self):
        self.cancel = False
        self.sigApplyToAllFutureFrames.emit(
            self.postProcessGroupbox.kwargs(),
            self.postProcessGroupbox.groupedFeatures(),
            self.postProcessGroupbox.selectedFeaturesRange()
        )
        self.close()

    def cancel_cb(self):
        self.cancel = True
        self.close()
    
    def undoChanges(self):
        if self.mainWin is not None:
            self.posData.lab = self.origLab
            self.mainWin.update_rp()
            self.mainWin.updateAllImages()

        # Undo if changes were applied to all future frames
        if hasattr(self, 'origSegmData'):
            if self.isTimelapse:
                current_frame_i = self.posData.frame_i
                for frame_i in range(self.posData.segmSizeT):
                    self.posData.frame_i = frame_i
                    origLab = self.origSegmData[frame_i]
                    lab = self.posData.allData_li[frame_i]['labels']
                    if lab is None:
                        # Non-visited frame modify segm_data
                        self.posData.segm_data[frame_i] = origLab
                    else:
                        self.posData.allData_li[frame_i]['labels'] = origLab.copy()
                        self.posData.lab = origLab.copy()
                        self.mainWin.update_rp()
                        # Get the rest of the stored metadata based on the new lab
                        self.mainWin.get_data()
                        self.mainWin.store_data()
                # Back to current frame
                self.posData.frame_i = current_frame_i
                self.mainWin.get_data()
                self.mainWin.updateAllImages()
            elif self.isMultiPos:
                current_pos_i = self.mainWin.pos_i
                # Apply to all future frames or future positions
                for pos_i, posData in enumerate(self.mainWin.data):
                    self.mainWin.pos_i = pos_i
                    origLab = self.origSegmData[pos_i]
                    self.posData.allData_li[0]['labels'] = lab.copy()
                    # Get the rest of the stored metadata based on the new lab
                    self.mainWin.get_data()
                    self.mainWin.store_data()
                # Back to current pos and current frame
                self.mainWin.pos_i = current_pos_i
                self.mainWin.get_data()
                self.mainWin.updateAllImages()

    def show(self, block=False):
        # self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show(block=False)
        self.resize(int(self.width()*1.5), self.height())
        super().show(block=block)

    def closeEvent(self, event):
        self.sigClosed.emit()
        if self.cancel:
            self.undoChanges()
        super().closeEvent(event)

class imageViewer(QMainWindow):
    """Main Window."""
    sigClosed = Signal()

    def __init__(
            self, parent=None, posData=None, button_toUncheck=None,
            spinBox=None, linkWindow=None, enableOverlay=False,
            isSigleFrame=False
        ):
        self.button_toUncheck = button_toUncheck
        self.parent = parent
        self.posData = posData
        self.spinBox = spinBox
        self.linkWindow = linkWindow
        self.isSigleFrame = isSigleFrame
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

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QGridLayout()
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
            editToolBar.addWidget(QLabel('  '))
            self.linkWindowCheckbox = QCheckBox("Link to main GUI")
            self.linkWindowCheckbox.setChecked(True)
            editToolBar.addWidget(self.linkWindowCheckbox)

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
        self.Plot.hideAxis('bottom')
        self.Plot.hideAxis('left')
        self.graphLayout.addItem(self.Plot, row=1, col=1)

        # Image Item
        self.img = pg.ImageItem(np.zeros((512,512)))
        self.Plot.addItem(self.img)

        #Image histogram
        self.imgGrad = widgets.myHistogramLUTitem(isViewer=True)
        self.imgGrad.gradient.showMenu = self.showLutItemOverlayContextMenu 
        self.imgGrad.vb.raiseContextMenu = lambda x: None
        self.imgGrad.setImageItem(self.img)
        self.graphLayout.addItem(self.imgGrad, row=1, col=0)

        # Current frame text
        self.frameLabel = pg.LabelItem(justify='center', color='w', size='14pt')
        self.frameLabel.setText(' ')
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
            lutItem.overlayColorButton.sigColorChanging.connect(
                self.updateOlColors
            )
            self.addAlphaScrollbar(ch, imageItem, alphaScrollbar)
            self.overlayLayersItems[ch] = overlayItems
            self.Plot.addItem(imageItem)
    
    def createOverlayChannelsActions(self):
        self.overlayLutItemAdditionalActions = []
        separator = QAction(self)
        separator.setSeparator(True)
        self.overlayLutItemAdditionalActions.append(separator)
        section = self.imgGrad.gradient.menu.addSection(
            'Select channel to adjust: '
        )
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
            self.setOverlayItemsVisible('', False)
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
                    if filename.endswith(f'{action.text()}_aligned'):
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
        t_label = QLabel('frame  ')
        _font = QFont()
        _font.setPixelSize(12)
        t_label.setFont(_font)
        self.img_Widglayout.addWidget(
                t_label, 0, 0, alignment=Qt.AlignRight)
        self.img_Widglayout.addWidget(
                self.framesScrollBar, 0, 1, 1, 20)
        self.t_label = t_label
        self.framesScrollBar.valueChanged.connect(self.framesScrollBarMoved)

        # z-slice scrollbar
        self.zSliceScrollBar = QScrollBar(Qt.Horizontal)
        # self.zSliceScrollBar.setFixedHeight(20)
        self.zSliceScrollBar.setMaximum(self.posData.SizeZ-1)
        _z_label = QLabel('z-slice  ')
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
    
    def setOverlayColors(self):
        self.overlayRGBs = [
            (255, 255, 0),
            (252, 72, 254),
            (49, 222, 134),
            (22, 108, 27)
        ]
        cmap = matplotlib.colormaps['gist_rainbow']
        self.overlayRGBs.extend(
            [tuple([round(c*255) for c in cmap(i)][:3]) 
            for i in np.linspace(0,1,8)]
        )

    def setOpacityOverlayLayersItems(self, value, imageItem=None):
        if imageItem is None:
            imageItem = self.sender().imageItem
            alpha = value/self.sender().maximum()
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
                self.checkedOverlayChName = (
                    self.parent.imgGrad.checkedChannelname
                )
                selectedChannels = self.parent.checkedOverlayChannels
                self.setCheckedOverlayContextMenusActions(selectedChannels)
            self.setOverlayItemsVisible(self.checkedOverlayChName, True)
        else:
            self.img.setOpacity(1.0)
            self.setOverlayItemsVisible('', False)
            for items in self.overlayLayersItems.values():
                imageItem = items[0]
                imageItem.clear()
        self.update_img()
    
    def createOverlayContextMenu(self):
        ch_names = [
            ch for ch in self.posData.chNames 
            if ch != self.posData.user_ch_name
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
                self.setOverlayItemsVisible('', False)
        self.update_img()
    
    def updateOlColors(self, button):
        lutItem = self.overlayLayersItems[self.checkedOverlayChName][1]
        rgb = lutItem.overlayColorButton.color().getRgb()[:3]
        self.parent.initColormapOverlayLayerItem(rgb, lutItem)
        lutItem.overlayColorButton.setColor(rgb)
    
    def addAlphaScrollbar(self, channelName, imageItem, alphaScrollBar=None):
        if alphaScrollBar is None:
            alphaScrollBar = QScrollBar(Qt.Horizontal)
        label = QLabel(f'Alpha {channelName}')
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
            f'Control the alpha value of the overlaid channel {channelName}.\n'
            'alpha=0 results in NO overlay,\n'
            'alpha=1 results in only fluorescence data visible'
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
        self.frame_i = frame_n-1
        self.t_label.setText(
            f'frame n. {self.frame_i+1}/{self.num_frames}'
        )
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
            f'Current frame = {self.frame_i+1}/{self.num_frames}'
        )
        if self.parent is None:
            img = self.getImage()
        else:
            img = self.parent.getImage(frame_i=self.frame_i)
        self.img.setImage(img)
        self.framesScrollBar.setSliderPosition(self.frame_i+1)
        
        if not self.enableOverlay:
            return

        if not self.overlayButton.isChecked():
            return
        
        self.setOverlayImages(frame_i=self.frame_i)
    
    def askSelectOverlayChannel(self):
        ch_names = [
            ch for ch in self.posData.chNames 
            if ch != self.posData.user_ch_name
        ]
        selectFluo = widgets.QDialogListbox(
            'Select channel',
            'Select channel names to overlay:\n',
            ch_names, multiSelection=True, parent=self
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

class TreeSelectorDialog(QBaseDialog):
    sigItemDoubleClicked = Signal(object)

    def __init__(
            self, title='Tree selector', infoTxt='', parent=None,
            multiSelection=True, widthFactor=None, heightFactor=None,
            expandOnDoubleClick=False, isTopLevelSelectable=True,
            allItemsExpanded=True, allowNoSelection=True
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
        msg.warning(self, 'Selection is empty', txt)
    
    def ok_cb(self):
        if not self.allowNoSelection and not self.selectedItems():
            self.warnSelectionIsEmpty()
            return
        self.cancel = False
        self.close()
    
    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self.widthFactor is not None:
            self.resize(int(self.width()*self.widthFactor), self.height())
        if self.heightFactor is not None:
            self.resize(self.width(), int(self.height()*self.heightFactor))

class TreesSelectorDialog(QBaseDialog):
    def __init__(
            self, trees, groupsDescr=None, title='Trees selector', 
            infoTxt='', parent=None
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
                groupName = ''
            else:
                groupName = groupsDescr.get(treeName, 'Group info missing')
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
            self, lists: dict, groupsDescr: dict=None, 
            title='Lists selector', infoTxt='', parent=None
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
                groupName = ''
            else:
                groupName = groupsDescr.get(listName, 'Group info missing')
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
            listWidget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
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
    def __init__(self, expPaths: dict, infoPaths: dict=None, parent=None):
        super().__init__(parent=parent)

        self.expPaths = expPaths
        self.cancel = True

        mainLayout = QVBoxLayout()

        self.setWindowTitle('Select Positions to process')

        infoTxt = html_utils.paragraph(
            'Select one or more Positions to process<br><br>'
            '<code>Click</code> on experiment path <i>to select all positions</i><br>'
            '<code>Ctrl+Click</code> <i>to select multiple items</i><br>'
            '<code>Shift+Click</code> <i>to select a range of items</i><br>',
            center=True
        )
        infoLabel = QLabel(infoTxt)

        self.treeWidget = QTreeWidget()
        self.treeWidget.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.treeWidget.setHeaderHidden(True)
        self.treeWidget.setFont(font)
        for exp_path, positions in expPaths.items():
            pathLevels = exp_path.split(os.sep)
            posFoldersInfo = None
            if infoPaths is not None:
                posFoldersInfo = infoPaths.get(exp_path)
            if len(pathLevels) > 4:
                itemText = os.path.join(*pathLevels[-4:])
                itemText = f'...{itemText}'
            else:
                itemText = exp_path
            exp_path_item = QTreeWidgetItem([itemText])
            exp_path_item.setToolTip(0, exp_path)
            exp_path_item.full_path = exp_path
            self.treeWidget.addTopLevelItem(exp_path_item)
            postions_items = []
            for pos in positions:
                if posFoldersInfo is not None:
                    status = posFoldersInfo.get(pos, '')
                else:
                    status = ''
                pos_item_text = f'{pos}{status}'
                pos_item = QTreeWidgetItem(exp_path_item, [pos_item_text])
                pos_item.posFoldername = pos
                postions_items.append(pos_item)
            exp_path_item.addChildren(postions_items)
            exp_path_item.setExpanded(True)

        self.treeWidget.itemClicked.connect(self.selectAllChildren)

        buttonsLayout = QHBoxLayout()
        cancelButton = widgets.cancelPushButton('Cancel')
        okButton = widgets.okPushButton(' Ok ')

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
            txt = 'You did not select any experiment/Position folder!'
            msg.warning(self, 'Empty selection!', html_utils.paragraph(txt))
            return

        self.cancel = False
        self.selectedPaths = {}
        for item in self.treeWidget.selectedItems():
            if item.parent() is None:
                exp_path = item.full_path
                self.selectedPaths[exp_path] = self.expPaths[exp_path]
            else:
                parent = item.parent()
                if parent.isSelected():
                    # Already added all children
                    continue
                exp_path = parent.full_path
                pos_folder = item.posFoldername
                if exp_path not in self.selectedPaths:
                    self.selectedPaths[exp_path] = []
                self.selectedPaths[exp_path].append(pos_folder)

        self.close()

    def showEvent(self, event):
        self.resize(int(self.width()*2), self.height())


class editCcaTableWidget(QDialog):
    sigApplyChangesFutureFrames = Signal(object, int)
    
    def __init__(
            self, cca_df, SizeT, title='Edit cell cycle annotations', 
            parent=None, current_frame_i=0
        ):
        self.inputCca_df = cca_df
        self.cancel = True
        self.SizeT = SizeT
        self.cca_df = None
        self.current_frame_i = current_frame_i

        super().__init__(parent)
        self.setWindowTitle(title)

        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        # Layouts
        mainLayout = QVBoxLayout()
        headerLayout = QGridLayout()
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
        headerLayout.addWidget(IDsLabel, 0, col, alignment=AC)

        col += 1
        ccsLabel = QLabel('Cell cycle stage')
        ccsLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(ccsLabel, 0, col, alignment=AC)

        col += 1
        relIDLabel = QLabel('Relative ID')
        relIDLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(relIDLabel, 0, col, alignment=AC)

        col += 1
        genNumLabel = QLabel('Generation number')
        genNumLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(genNumLabel, 0, col, alignment=AC)
        genNumColWidth = genNumLabel.sizeHint().width()

        col += 1
        relationshipLabel = QLabel('Relationship')
        relationshipLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(relationshipLabel, 0, col, alignment=AC)

        col += 1
        emergFrameLabel = QLabel('Emerging frame num.')
        emergFrameLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(emergFrameLabel, 0, col, alignment=AC)

        col += 1
        divitionFrameLabel = QLabel('Division frame num.')
        divitionFrameLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(divitionFrameLabel, 0, col, alignment=AC)

        col += 1
        historyKnownLabel = QLabel('Is history known?')
        historyKnownLabel.setAlignment(Qt.AlignCenter)
        headerLayout.addWidget(historyKnownLabel, 0, col, alignment=AC)
        
        self.headerLayout = headerLayout

        tableLayout.setHorizontalSpacing(20)
        self.tableLayout = tableLayout

        # Add buttons
        cancelButton = widgets.cancelPushButton('Cancel')
        moreInfoButton = widgets.helpPushButton('More info...')
        moreInfoButton.setIcon(QIcon(':info.svg'))
        applyToFutureFramesbutton = widgets.futurePushButton(
            'Apply changes to future frames...'
        )
        okButton = widgets.okPushButton('Ok')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(moreInfoButton)
        buttonsLayout.addWidget(applyToFutureFramesbutton)
        buttonsLayout.addWidget(okButton)

        # Scroll area properties
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.scrollArea.setFrameStyle(QFrame.Shape.NoFrame)
        self.scrollArea.setWidgetResizable(True)

        # Add layouts
        self.viewBox.setLayout(tableLayout)
        self.scrollArea.setWidget(self.viewBox)
        mainLayout.addLayout(headerLayout)
        mainLayout.addWidget(self.scrollArea)
        mainLayout.addSpacing(20)
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
            genNumSpinBox = QSpinBox()
            genNumSpinBox.setFocusPolicy(Qt.StrongFocus)
            genNumSpinBox.installEventFilter(self)
            genNumSpinBox.setValue(2)
            genNumSpinBox.setMaximum(2147483647)
            genNumSpinBox.setAlignment(Qt.AlignCenter)
            genNumSpinBox.setFixedWidth(int(genNumColWidth*2/3))
            genNumSpinBox.setValue(cca_df.at[ID, 'generation_num'])
            tableLayout.addWidget(genNumSpinBox, row+1, col, alignment=AC)
            self.genNumSpinBoxes.append(genNumSpinBox)

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
            emergFrameSpinBox.setMaximum(SizeT)
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
            divisFrameSpinBox.setMaximum(SizeT)
            divisFrameSpinBox.setValue(-1)
            divisFrameSpinBox.setAlignment(Qt.AlignCenter)
            divisFrameSpinBox.setFixedWidth(int(genNumColWidth*2/3))
            divisFrame_i = cca_df.at[ID, 'division_frame_i']
            val = divisFrame_i+1 if divisFrame_i>=0 else -1
            divisFrameSpinBox.setValue(val)
            tableLayout.addWidget(divisFrameSpinBox, row+1, col, alignment=AC)
            self.divisFrameSpinBoxes.append(divisFrameSpinBox)
            self.divisFrameSpinPrevValues.append(divisFrameSpinBox.value())
            divisFrameSpinBox.valueChanged.connect(self.skip0divisFrame)

            col += 1
            HistoryCheckBox = QCheckBox()
            HistoryCheckBox.setChecked(bool(cca_df.at[ID, 'is_history_known']))
            tableLayout.addWidget(HistoryCheckBox, row+1, col, alignment=AC)
            self.historyKnownCheckBoxes.append(HistoryCheckBox)

        self.setLayout(mainLayout)

        # Connect to events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)
        moreInfoButton.clicked.connect(self.moreInfo)
        applyToFutureFramesbutton.clicked.connect(self.applyToFutureFrames)

        # self.setModal(True)
    
    def getChanges(self):
        newCcaDf = self.getCca_df()
        changes = {}
        for row in newCcaDf.itertuples():
            ID = row.Index
            for col in newCcaDf.columns:
                inputValue = self.inputCca_df.at[ID, col]
                newValue = getattr(row, col)
                if newValue == inputValue:
                    continue
                
                if ID not in changes:
                    changes[ID] = {col: (inputValue, newValue)}
                else:
                    changes[ID][col] = (inputValue, newValue)
        return changes

    def applyToFutureFrames(self):        
        txt = 'Enter <b>up to which frame</b> you want to apply the changes<br>'
        win = NumericEntryDialog(
            title='Stop frame', instructions=txt, parent=self, minValue=1, 
            maxValue=self.SizeT, currentValue=self.current_frame_i
        )
        win.exec_()
        if win.cancel:
            return
        
        stop_frame_i = win.value
        changes = self.getChanges()
        changes_format = myutils.format_cca_manual_changes(changes)
        detailsText = (
            f'Changes that will be applied from frame n. {self.current_frame_i+1}'
            f' to frame n. {stop_frame_i+1}:\n\n{changes_format}'
        )
        txt = html_utils.paragraph("""
Use this feature with <b>caution</b>!<br><br>
Before propagating to future frames <b>carefully inspect what changes</b> 
will be applied (see below).<br><br>
""")
        msg = widgets.myMessageBox(wrapText=False)
        msg.setDetailedText(detailsText, visible=True)
        msg.warning(
            self, 'Caution!', txt, buttonsTexts=('Yes, I am sure', 'Cancel')
        )
        if msg.cancel:
            return
        
        self.sigApplyChangesFutureFrames.emit(changes, stop_frame_i)     
    
    def moreInfo(self, checked=True):
        desc = myutils.get_cca_colname_desc()
        msg = widgets.myMessageBox(parent=self)
        msg.setWindowTitle('Cell cycle annotations info')
        msg.setWidth(400)
        msg.setIcon()
        for col, txt in desc.items():
            msg.addText(html_utils.paragraph(f'<b>{col}</b>: {txt}'))
        msg.addButton('  Ok  ')
        msg.exec_()

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
        emergFrameValues = [
            var.value()-1 if var.value()>0 else -1
            for var in self.emergFrameSpinBoxes
        ]
        divisFrameValues = [
            var.value()-1 if var.value()>0 else -1
            for var in self.divisFrameSpinBoxes
        ]
        historyValues = [
            var.isChecked() for var in self.historyKnownCheckBoxes
        ]
        check_rel = [ID == relID for ID, relID in zip(self.IDs, relIDValues)]
        
        # Buds in S phase must have 0 as number of cycles
        check_buds_S = [
            ccs=='S' and rel_ship=='bud' and not numc==0
            for ccs, rel_ship, numc
            in zip(ccsValues, relatValues, genNumValues)
        ]
        
        # Mother cells must have at least 1 as number of cycles if history known
        check_mothers = [
            rel_ship=='mother' and not numc>=1
            if is_history_known else False
            for rel_ship, numc, is_history_known
            in zip(relatValues, genNumValues, historyValues)
        ]
        
        # Buds cannot be in G1
        check_buds_G1 = [
            ccs=='G1' and rel_ship=='bud' for ccs, rel_ship
            in zip(ccsValues, relatValues)
        ]
        
        # The number of cells in S phase must be half mothers and half buds
        num_moth_S = len([
            0 for ccs, rel_ship in zip(ccsValues, relatValues)
            if ccs=='S' and rel_ship=='mother'
        ])
        num_bud_S = len([
            0 for ccs, rel_ship in zip(ccsValues, relatValues)
            if ccs=='S' and rel_ship=='bud'
        ])
        
        # Cells in S phase cannot have -1 as relative's ID
        check_relID_S = [
            ccs=='S' and relID==-1
            for ccs, relID in zip(ccsValues, relIDValues)
        ]
        
        # Mother cells with unknown history at emergence is recommended to have
        # generation number = 2 (easier downstream analysis)
        check_unknown_mothers = [
            rel_ship=='mother' and not is_history_known and gen_num!=2
            and (emerg_frame_i == self.current_frame_i or self.current_frame_i==0)
            for rel_ship, is_history_known, gen_num, emerg_frame_i
            in zip(relatValues, historyValues, genNumValues, emergFrameValues)
        ]
        
        if any(check_rel):
            msg = widgets.myMessageBox(wrapText=False)
            txt = html_utils.paragraph(""" 
                Some cells are mother or bud of itself!<br><br>
                Make sure that the relative ID is different from the Cell ID.
            """)
            msg.critical(self, 'Some IDs are equal to relative ID', txt)
            return None
        elif any(check_unknown_mothers):
            txt = html_utils.paragraph("""
                We recommend to set <b>generation number to 2 for mother cells 
                with unknown history<br>
                that just appeared</b> (i.e., first cell cycle in the video).<br><br>
                While it is allowed to insert any number, knowing that these 
                cells start at generation number 2<br>
                <b>makes downstream analysis easier</b>.<br><br>
                What do you want to do?
            """)
            correctButtonText = ' Fine, let me correct. '
            keepButtonText = ' Keep the generation number that I chose. '
            buttonsTexts = (correctButtonText, keepButtonText)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            msg.warning(self, 'Recommendation', txt, buttonsTexts=buttonsTexts)
            if msg.cancel or msg.clickedButton == correctButtonText:
                return None
        elif any(check_buds_S):
            msg = widgets.myMessageBox(wrapText=False)
            title = (
                'Bud in S/G2/M not in 0 Generation number'
            )
            txt = html_utils.paragraph(
                'Some buds '
                'in S phase do not have 0 as Generation number!<br>'
                'Buds in S phase must have 0 as "Generation number"'
            )
            msg.critical(self, title, txt)
            return None
        elif any(check_mothers):
            msg = widgets.myMessageBox(wrapText=False)
            title = (
                'Mother not in >=1 Generation number'
            )
            txt = html_utils.paragraph(
                'Some mother cells do not have >=1 as "Generation number"!<br>'
                'Mothers MUST have >1 "Generation number"'
            )
            msg.critical(self, title, txt)
            return None
        elif any(check_buds_G1):
            msg = widgets.myMessageBox(wrapText=False)
            title = (
                'Buds in G1!'
            )
            txt = html_utils.paragraph(
                'Some buds are in G1 phase!<br><br>'
                'Buds MUST be in S/G2/M phase'
            )
            msg.critical(self, title, txt)
            return None
        elif num_moth_S != num_bud_S:
            msg = widgets.myMessageBox(wrapText=False)
            title = (
                'Number of mothers-buds mismatch!'
            )
            txt = html_utils.paragraph(
                f'There are {num_moth_S} mother cells in "S/G2/M" phase,'
                f'but there are {num_bud_S} bud cells.<br><br>'
                'The number of mothers and buds in "S/G2/M" '
                'phase must be equal!'
            )
            msg.critical(self, title, txt)
            return None
        elif any(check_relID_S):
            msg = widgets.myMessageBox(wrapText=False)
            title = (
                'Relative\'s ID of cells in S/G2/M = -1'
            )
            txt = html_utils.paragraph(
                'Some cells are in "S/G2/M" phase but have -1 as Relative\'s ID!<br>'
                'Cells in "S/G2/M" phase must have an existing '
                'ID as Relative\'s ID!'
            )
            msg.critical(self, title, txt)
            return None
        
        corrected_assignment = self.inputCca_df['corrected_assignment']
        cca_df = pd.DataFrame({
            'cell_cycle_stage': ccsValues,
            'generation_num': genNumValues,
            'relative_ID': relIDValues,
            'relationship': relatValues,
            'emerg_frame_i': emergFrameValues,
            'division_frame_i': divisFrameValues,
            'is_history_known': historyValues,
            'corrected_assignment': corrected_assignment,
            'will_divide': self.inputCca_df['will_divide'],
            }, index=self.IDs
        )
        cca_df.index.name = 'Cell_ID'
        
        # Add missing columns
        for column, default in base_cca_dict.items():
            if column in cca_df.columns:
                continue
            
            value = self.inputCca_df.get(column, default=default)
            cca_df[column] = value
            
        # Check that every pair of cells in S are relative of each other
        proceed = self.check_ID_rel_ID_mismatches(cca_df)
        if not proceed:
            return None
            
        d = dict.fromkeys(cca_df.select_dtypes(np.int64).columns, np.int32)
        cca_df = cca_df.astype(d)
        return cca_df

    def check_ID_rel_ID_mismatches(self, cca_df):
        ID_rel_ID_mismatches = []
        for row in cca_df.itertuples():
            if row.cell_cycle_stage == 'G1':
                continue
            
            ID = row.Index
            relID = row.relative_ID
            relID_of_relID = cca_df.at[relID, 'relative_ID']
            
            if relID_of_relID != ID:
                ID_rel_ID_mismatches.append((ID, relID, relID_of_relID))
        
        if not ID_rel_ID_mismatches:
            return True
        
        items = [
            f'Cell ID {ID} has relative ID = {relID}, '
            f'while cell ID {relID} has relative ID = {relID_of_relID}'
            for ID, relID, relID_of_relID in ID_rel_ID_mismatches
        ]
        title = '`ID-relative_ID` mismatches'
        txt = html_utils.paragraph(
            f'`ID-relative_ID` mismatches:'
            f'{html_utils.to_list(items)}'
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.critical(self, title, txt)
        return False
    
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
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()
        ncols = self.tableLayout.columnCount()
        maxLabelWidth = max([
            self.headerLayout.itemAt(j).widget().sizeHint().width()
            for j in range(ncols)
        ])
        minWidth = (maxLabelWidth+5)*ncols
        self.setMinimumWidth(minWidth)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def eventFilter(self, object, event):
        # Disable wheel scroll on widgets to allow scroll only on scrollarea
        if event.type() == QEvent.Type.Wheel:
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
            self, user_ch_file_paths, user_ch_name, parent=None
        ):
        self.parent = parent
        self.cancel = True

        super().__init__(parent)
        self.setWindowTitle('Enter stop frame')

        mainLayout = QVBoxLayout()
        formLayout = QFormLayout()
        buttonsLayout = QHBoxLayout()

        # Message
        infoTxt = html_utils.paragraph("""
            Enter a <b>stop frame number</b> when to stop<br>
            segmentation for each Position loaded:
        """)
        infoLabel = QLabel(infoTxt, self)
        _font = QFont()
        _font.setPixelSize(12)
        infoLabel.setFont(_font)
        infoLabel.setAlignment(Qt.AlignCenter)
        # padding: top, left, bottom, right
        infoLabel.setStyleSheet("padding:0px 0px 8px 0px;")

        self.dataDict = {}

        # Form layout widget
        self.spinBoxes = []
        self.tab_idx = 0
        for (i, img_path) in enumerate(user_ch_file_paths):
            pos_foldername = os.path.basename(
                os.path.dirname(
                    os.path.dirname(img_path)
                )
            )
            spinBox = widgets.mySpinBox()
            spinBox.sigTabEvent.connect(self.keyTabEventSpinbox)
            posData = load.loadData(img_path, user_ch_name, QParent=parent)
            posData.getBasenameAndChNames()
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
            visualizeButton = widgets.viewPushButton('Visualize')
            visualizeButton.clicked.connect(self.visualize_cb)
            formLabel = QLabel(html_utils.paragraph(f'{pos_foldername}  '))
            layout = QHBoxLayout()
            layout.addWidget(formLabel, alignment=Qt.AlignRight)
            layout.addWidget(spinBox)
            layout.addWidget(visualizeButton)
            self.dataDict[visualizeButton] = (spinBox, posData)
            formLayout.addRow(layout)
            spinBox.idx = i
            self.spinBoxes.append(spinBox)

        self.formLayout = formLayout
        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)
        mainLayout.addLayout(formLayout)

        okButton = widgets.okPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = widgets.cancelPushButton('Cancel')

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
            posData.metadata_df.at['stop_frame_num', 'values'] = spinBox.value()
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

    def visualize_cb(self, checked=True):
        spinBox, posData = self.dataDict[self.sender()]
        print('Loading image data...')
        posData.loadImgData()
        posData.frame_i = spinBox.value()-1
        plot.imshow(posData.img_data, lut='gray')
        # self.slideshowWin = imageViewer(
        #     posData=posData, spinBox=spinBox
        # )
        # self.slideshowWin.update_img()
        # # self.slideshowWin.framesScrollBar.setDisabled(True)
        # self.slideshowWin.show()

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

class QLineEditDialog(QDialog):
    def __init__(
            self, title='Entry messagebox', msg='Entry value',
            defaultTxt='', parent=None, allowedValues=None,
            warnLastFrame=False, isInteger=False, isFloat=False,
            stretchEntry=True, allowEmpty=True, allowedTextEntries=None, 
            allowText=False, lastVisitedFrame=None, allowList=False
        ):
        QDialog.__init__(self, parent)

        self.loop = None
        self.cancel = True
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
        if not msg.startswith('<p'):
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
                _min, _max = allowedValues
                entryWidget.setMinimum(_min)
                entryWidget.setMaximum(_max)
            else:
                entryWidget.setMaximum(2147483647)
            if defaultTxt:
                entryWidget.setValue(float(defaultTxt))

        elif isInteger and not allowList:
            entryWidget = QSpinBox()
            if allowedValues is not None:
                _min, _max = allowedValues
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
            notValidLabel.setStyleSheet('color: red')
            notValidLabel.setFont(font)
            notValidLabel.setAlignment(Qt.AlignCenter)
            self.notValidLabel = notValidLabel

        okButton = widgets.okPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = widgets.cancelPushButton('Cancel')

        # Events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        # Contents margins
        buttonsLayout.setContentsMargins(0,10,0,0)

        # Add widgets to layouts
        LineEditLayout.addWidget(msg, alignment=Qt.AlignCenter)
        if stretchEntry:
            LineEditLayout.addWidget(entryWidget)
        else:
            entryLayout = QHBoxLayout()
            entryLayout.addStretch(1)
            entryLayout.addWidget(entryWidget)
            entryLayout.addStretch(1)
            entryLayout.setStretch(1,1)
            LineEditLayout.addLayout(entryLayout)
        if allowedValues is not None:
            LineEditLayout.addWidget(notValidLabel, alignment=Qt.AlignCenter)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        # Add layouts
        mainLayout.addLayout(LineEditLayout)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        # self.setModal(True)
    
    def value(self):
        if self.isFloat or self.isInteger:
            val = self.entryWidget.value()
        elif not self.allowList:
            val = int(self.entryWidget.text())
        elif self.allowList:
            text = self.entryWidget.text()
            m = re.findall(POSITIVE_FLOAT_REGEX, text)
            val = [int(val) for val in m]
        return val

    def onTextChanged(self, text):
        # Get inserted char
        idx = self.entryWidget.cursorPosition()
        if idx == 0:
            return

        newChar = text[idx-1]
        if self.allowList and (newChar==',' or newChar == ' '):
            return

        # Allow only integers
        try:
            val = int(newChar)
            if val > np.iinfo(np.uint32).max:
                self.entryWidget.setText(str(np.iinfo(np.uint32).max))
        except Exception as e:
            text = text.replace(newChar, '')
            self.entryWidget.setText(text)
            return
        
        if self.allowedValues is not None:
            currentVal = self.value()
            if self.allowList:
                currentVal = currentVal[-1]
            if currentVal not in self.allowedValues:
                self.notValidLabel.setText(f'{currentVal} not existing!')
            else:
                self.notValidLabel.setText('')

    def warnValLessLastFrame(self, val):
        msg = widgets.myMessageBox()
        warn_txt = html_utils.paragraph(f"""
            WARNING: saving until a frame number below the last visited
            frame ({self.lastVisitedFrame}) will result in <b>LOSS of information</b>
            about any <b>edit or annotation</b> you did <b>on frames
            {val+1}-{self.lastVisitedFrame}.</b><br><br>
            Are you sure you want to proceed?
        """)
        msg.warning(
           self, 'WARNING: Potential loss of information', warn_txt, 
           buttonsTexts=('Cancel', 'Yes, I am sure.')
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
           self, 'Saving past last visited frame', warn_txt, 
           buttonsTexts=('Cancel', 'Yes, I am sure.')
        )
        return msg.cancel

    def ok_cb(self, event):
        if not self.allowEmpty and not self.entryWidget.text():
            msg = widgets.myMessageBox(showCentered=False, wrapText=False)
            msg.critical(
                self, 'Empty text', 
                html_utils.paragraph('Text entry field <b>cannot be empty</b>')
            )
            return
        if self.allowedTextEntries is not None:
            if self.entryWidget.text() not in self.allowedTextEntries:
                msg = widgets.myMessageBox(showCentered=False, wrapText=False)
                txt = html_utils.paragraph(
                    f'"{self.entryWidget.text()}" is not a valid entry.<br><br>'
                    'Valid entries are:<br>'
                    f'{html_utils.to_list(self.allowedTextEntries)}'
                )
                msg.critical(self, 'Not a valid entry', txt)
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
        if hasattr(self, 'loop'):
            self.loop.exit()

class NumericEntryDialog(QBaseDialog):
    def __init__(
            self, title='Entry a value', currentValue=0,
            instructions='Entry value', parent=None, 
            maxValue=None, minValue=None, stretch=False
        ):
        super().__init__(parent=parent)
        self.setWindowTitle(title)
        self.cancel = False
        mainLayout = QVBoxLayout()
        entryLayout = QHBoxLayout()
        cancelOkLayout = widgets.CancelOkButtonsLayout()
        cancelOkLayout.okButton.clicked.connect(self.ok_cb)
        cancelOkLayout.cancelButton.clicked.connect(self.close)
        
        instructionsLabel = QLabel(html_utils.paragraph(instructions))
        mainLayout.addWidget(instructionsLabel)
        
        if type(currentValue) == int:
            self.entryWidget = widgets.SpinBox()
            self.entryWidget.setValue(currentValue)
            self.valueGetter = 'value'
            if maxValue is not None:
                self.entryWidget.setMaximum(maxValue)
            if minValue is not None:
                self.entryWidget.setMinimum(minValue)
        
        if stretch:
            entryLayout.addWidget(self.entryWidget)
        else:
            entryLayout.addStretch(1)
            entryLayout.addWidget(self.entryWidget)
            entryLayout.addStretch(1)
        
        mainLayout.addLayout(entryLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(cancelOkLayout)
        
        self.setLayout(mainLayout)
    
    def ok_cb(self):
        self.cancel = False
        self.value = getattr(self.entryWidget, self.valueGetter)()
        self.close()

class editID_QWidget(QDialog):
    def __init__(
            self, clickedID, IDs, entryID=None, doNotShowAgain=False, 
            parent=None
        ):
        self.IDs = IDs
        self.clickedID = clickedID
        self.cancel = True
        self.how = None
        self.mergeWithExistingID = True
        self.doNotAskAgainExistingID = doNotShowAgain

        super().__init__(parent)
        self.setWindowTitle("Edit ID")
        mainLayout = QVBoxLayout()

        VBoxLayout = QVBoxLayout()
        msg = QLabel(f'Replace ID {clickedID} with:')
        _font = QFont()
        _font.setPixelSize(12)
        msg.setFont(_font)
        # padding: top, left, bottom, right
        msg.setStyleSheet("padding:0px 0px 3px 0px;")
        VBoxLayout.addWidget(msg, alignment=Qt.AlignCenter)

        entryWidget = QLineEdit()
        entryWidget.setFont(_font)
        entryWidget.setAlignment(Qt.AlignCenter)
        self.entryWidget = entryWidget
        VBoxLayout.addWidget(entryWidget)
        if entryID is not None:
            entryWidget.setText(str(entryID))
            entryWidget.selectAll()

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
        okButton = widgets.okPushButton('Ok')
        cancelButton = widgets.cancelPushButton('Cancel')

        HBoxLayout.addWidget(cancelButton)
        HBoxLayout.addSpacing(20)
        HBoxLayout.addWidget(okButton)

        mainLayout.addSpacing(10)
        mainLayout.addLayout(HBoxLayout)

        self.setLayout(mainLayout)

        # Connect events
        self.prevText = ''
        entryWidget.textChanged[str].connect(self.onTextChanged)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)
        # self.setModal(True)

    def onTextChanged(self, text):
        # Get inserted char
        idx = self.entryWidget.cursorPosition()
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
            self.entryWidget.setText(text)
            return

        # Cast integers greater than uint32 machine limit
        m_iter = re.finditer(r'\d+', self.entryWidget.text())
        for m in m_iter:
            val = int(m.group())
            uint32_max = np.iinfo(np.uint32).max
            if val > uint32_max:
                text = self.entryWidget.text()
                text = f'{text[:m.start()]}{uint32_max}{text[m.end():]}'
                self.entryWidget.setText(text)

        # Automatically close ( bracket
        if newChar == '(':
            text += ')'
            self.entryWidget.setText(text)
        self.prevText = text
    
    def _warnExistingID(self, existingID, newID):
        warn_msg = html_utils.paragraph(f"""
            ID {existingID} is <b>already existing</b>.<br><br>
            How do you want to proceed?<br>
        """)
        msg = widgets.myMessageBox()
        doNotAskAgainCheckbox = QCheckBox('Remember my choice and do not ask again')
        swapButton = widgets.reloadPushButton(f'Swap {newID} with {existingID}')
        mergeButton = widgets.mergePushButton(f'Merge {newID} with {existingID}')
        msg.warning(
            self, 'Existing ID', warn_msg, 
            buttonsTexts=('Cancel', mergeButton, swapButton),
            widgets=doNotAskAgainCheckbox
        )
        if msg.cancel:
            return False
        self.doNotAskAgainExistingID = doNotAskAgainCheckbox.isChecked()
        self.mergeWithExistingID = msg.clickedButton ==  mergeButton
        return True

    def ok_cb(self, event):
        self.cancel = False
        txt = self.entryWidget.text()
        valid = False

        # Check validity of inserted text
        try:
            ID = int(txt)
            how = [(self.clickedID, ID)]
            if ID in self.IDs and not self.doNotAskAgainExistingID:
                proceed = self._warnExistingID(self.clickedID, ID)
                if not proceed:
                    return
                valid = True
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
            err_msg = html_utils.paragraph(
                'You entered invalid text. Valid text is either a single integer'
                f' ID that will be used to replace ID {self.clickedID} '
                'or a list of elements enclosed in parenthesis separated by a comma<br>'
                'such as (5, 10), (8, 27) to replace ID 5 with ID 10 and ID 8 with ID 27'
            )
            msg = widgets.myMessageBox()
            msg.critical(
                self, 'Invalid entry', err_msg
            )

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
    def __init__(
            self, title, items, informativeText,
            CbLabel='Select value:  ', parent=None,
            showInFileManagerPath=None
        ):
        self.cancel = True
        self.selectedItemsText = ''
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

        okButton = widgets.okPushButton('Ok')
        cancelButton = widgets.cancelPushButton('Cancel')
        if showInFileManagerPath is not None:
            txt = myutils.get_open_filemaneger_os_string()
            showInFileManagerButton = widgets.showInFileManagerButton(txt)

        bottomLayout.addStretch(1)
        bottomLayout.addWidget(cancelButton)
        bottomLayout.addSpacing(20)
        if showInFileManagerPath is not None:
            bottomLayout.addWidget(showInFileManagerButton)
        bottomLayout.addWidget(okButton)

        multiPosButton = QPushButton('Multiple selection')
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

    def showInFileManager(self):
        selectedTexts, _ = self.getSelectedItems()
        folder = selectedTexts[0].split('(')[0].strip()
        path = os.path.join(self.showInFileManagerPath, folder)
        if os.path.exists(path) and os.path.isdir(path):
            showPath = path
        else:
            showPath = self.showInFileManagerPath
        myutils.showInExplorer(showPath)

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
            self.ListBox.setMinimumHeight(h+5)
            self.ListBox.setFocusPolicy(Qt.StrongFocus)
            self.ListBox.setFocus()
            self.ListBox.setCurrentRow(0)
            self.mainLayout.setStretchFactor(self.topLayout, 2)
        else:
            self.multiPosButton.setText('Multiple selection')
            self.ListBox.hide()
            self.ComboBox.show()
            self.resize(self.width(), self.singleSelectionHeight)

    def getSelectedItems(self):
        if self.multiPosButton.isChecked():
            selectedItems = self.ListBox.selectedItems()
            selectedItemsText = [item.text() for item in selectedItems]
            selectedItemsText = natsorted(selectedItemsText)
            selectedItemsIdx = [
                self.items.index(txt) for txt in selectedItemsText
            ]
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
        if hasattr(self, 'loop'):
            self.loop.exit()


class manualSeparateGui(QMainWindow):
    def __init__(
            self, lab, ID, img, fontSize='12pt', IDcolor=[255, 255, 0],
            parent=None, loop=None, drawMode='threepoints_arc'
        ):
        super().__init__(parent)
        self.loop = loop
        self.cancel = True
        self.drawMode = drawMode
        self._parent = parent
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

        mainContainer = QWidget()
        self.setCentralWidget(mainContainer)

        mainLayout = QGridLayout()
        mainLayout.addWidget(self.graphLayout, 0, 0, 1, 1)
        mainLayout.addLayout(self.img_Widglayout, 1, 0)

        mainContainer.setLayout(mainLayout)

        self.setWindowModality(Qt.WindowModal)

    def centerWindow(self):
        parent = self._parent
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

        self.drawModesActionGroup = QActionGroup(self)

        self.threePointsArcAction = QAction(
            QIcon(":threepoints_arc.svg"), 'Separate with three-points arc', 
            self
        )
        self.threePointsArcAction.setCheckable(True)
        self.threePointsArcAction.drawMode = 'threepoints_arc'
        self.drawModesActionGroup.addAction(self.threePointsArcAction)

        self.freeHandAction = QAction(
            QIcon(":freehand.svg"), 'Separate with freehand line', self
        )
        self.freeHandAction.setCheckable(True)
        self.freeHandAction.drawMode = 'freehand'
        self.drawModesActionGroup.addAction(self.freeHandAction)

        if self.drawMode == 'threepoints_arc':
            self.threePointsArcAction.setChecked(True)
        elif self.drawMode == 'freehand':
            self.freeHandAction.setChecked(True)
    
    def state(self):
        return {
            'is_overlay_active': self.overlayButton.isChecked(),
            'is_three_points_active': self.threePointsArcAction.isChecked(),
            'is_free_hand_active': self.freeHandAction.isChecked()
        }
    
    def show(self, block=False):
        super().show()
        if not block:
            return
        self.loop = QEventLoop(self)
        self.loop.exec_()
    
    def setState(self, state):
        if state is None:
            return
        self.overlayButton.setChecked(state.get('is_overlay_active', False))
        self.threePointsArcAction.setChecked(
            state.get('is_three_points_active', True)
        )
        self.freeHandAction.setChecked(state.get('is_free_hand_active', False))
        
    def gui_storeDrawMode(self):
        self.drawMode = self.sender().drawMode
        
    def gui_createMenuBar(self):
        menuBar = self.menuBar()
        # style = "QMenuBar::item:selected { background: white; }"
        # menuBar.setStyleSheet(style)
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
        self.overlayButton.setToolTip(
            'Overlay channel\'s image'
        )
        editToolBar.addWidget(self.overlayButton)

        editToolBar.addAction(self.threePointsArcAction)
        editToolBar.addAction(self.freeHandAction)

    def gui_connectActions(self):
        self.exitAction.triggered.connect(self.close)
        self.helpAction.triggered.connect(self.help)
        self.okAction.triggered.connect(self.ok_cb)
        self.cancelAction.triggered.connect(self.close)
        self.undoAction.triggered.connect(self.undo)
        self.overlayButton.toggled.connect(self.toggleOverlay)
        self.imgGrad.sigLookupTableChanged.connect(self.histLUT_cb)

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
        self.imgGrad = widgets.myHistogramLUTitem()

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

        self.freeHandItem = widgets.PlotCurveItem(
            pen=pg.mkPen(color='r', width=2)
        )
        self.ax.addItem(self.freeHandItem)

    def gui_createImgWidgets(self):
        self.img_Widglayout = QGridLayout()
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

        if event.isExit():
            return

        self.drawHoverEvent(*event.pos())

    def gui_mousePressEventImg(self, event):
        right_click = event.button() == Qt.MouseButton.RightButton
        left_click = event.button() == Qt.MouseButton.LeftButton

        dragImg = (left_click)

        if dragImg:
            pg.ImageItem.mousePressEvent(self.imgItem, event)
        
        if not right_click:
            return

        self.drawPressEvent(event)
    
    def gui_mouseDragEventImg(self, event):
        pass

    def gui_mouseReleaseEventImg(self, event):
        if self.countClicks == 0:
            return
        if self.freeHandAction.isChecked():
            self.countClicks = 0
            xx, yy = self.freeHandItem.getData()
            self.setSplitCurveCoords(xx, yy)
            self.splitObjectAlongCurve()
            self.freeHandItem.setData([], [])
            self.curvAnchors.setData([], [])
    
    def getSpline(self, xx, yy):
        tck, u = scipy.interpolate.splprep([xx, yy], s=0, k=2)
        xi, yi = scipy.interpolate.splev(self.hoverLinSpace, tck)
        return xi, yi

    def drawPressEvent(self, event):
        if self.freeHandAction.isChecked():
            self.countClicks = 1
            x, y = event.pos().x(), event.pos().y()
            self.curvAnchors.addPoints([x], [y])
        elif self.threePointsArcAction.isChecked():
            self.threePointsArcPressEvent(event)
    
    def drawHoverEvent(self, x, y):
        if self.freeHandAction.isChecked():
            self.freeHandHoverEvent(x, y)
        elif self.threePointsArcAction.isChecked():
            self.threePointsArcHoverEvent(x, y)
        
    def freeHandHoverEvent(self, x, y):
        if self.countClicks == 0:
            return
        self.freeHandItem.addPoint(int(x), int(y))
        _xx, _yy = self.freeHandItem.getData()
        xx = [_xx[0], x]
        yy = [_yy[0], y]
        self.curvAnchors.setData(xx, yy)

    def threePointsArcHoverEvent(self, x, y):
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
    
    def threePointsArcPressEvent(self, event):
        if self.countClicks == 0:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.x0, self.y0 = xdata, ydata
            self.curvAnchors.addPoints([xdata], [ydata])
            self.countClicks = 1
        elif self.countClicks == 1:
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            self.x1, self.y1 = xdata, ydata
            self.curvAnchors.addPoints([xdata], [ydata])
            self.countClicks = 2
        elif self.countClicks == 2:
            self.countClicks = 0
            x, y = event.pos().x(), event.pos().y()
            xdata, ydata = int(x), int(y)
            xx = [self.x0, xdata, self.x1]
            yy = [self.y0, ydata, self.y1]
            xi, yi = self.getSpline(xx, yy)
            yy, xx = np.round(yi).astype(int), np.round(xi).astype(int)
            self.setSplitCurveCoords(xx, yy)
            self.splitObjectAlongCurve()
    
    def setSplitCurveCoords(self, xx, yy):
        self.storeUndoState()
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
        self.lab = skimage.morphology.remove_small_objects(self.lab, 5)

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
            labelItemID = widgets.myLabelItem()
            labelItemID.setText(
                f'{obj.label}', color='r', size=f'{self.fontSize}px'
            )
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
        else:
            tempID = self.lab.max() + 1
            self.lab[self.lab==maxAreaID] = tempID
            self.lab[self.lab==self.ID] = maxAreaID
            self.lab[self.lab==tempID] = self.ID

        # Keep only the two largest objects
        larger_areas = nlargest(2, areas)
        larger_ids = [rp[areas.index(area)].label for area in larger_areas]
        for obj in rp:
            if obj.label not in larger_ids:
                self.lab[tuple(obj.coords.T)] = 0

        rp = skimage.measure.regionprops(self.lab)

        if self._parent is not None:
            self._parent.setBrushID()
        # Use parent window setBrushID function for all other IDs
        for obj in rp:
            if self._parent is None:
                break
            if obj.label == self.ID:
                continue
            posData = self._parent.data[self._parent.pos_i]
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
        self.cmap = colors.getFromMatplotlib('viridis')
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
            self.freeHandItem.setData([], [])
        elif ev.key() == Qt.Key_Enter or ev.key() == Qt.Key_Return:
            self.ok_cb()

    def getOverlay(self):
        # Rescale intensity based on hist ticks values
        min = self.imgGrad.gradient.listTicks()[0][1]
        max = self.imgGrad.gradient.listTicks()[1][1]
        img = skimage.exposure.rescale_intensity(self.img, in_range=(min, max))
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

    def alphaScrollBarMoved(self, alpha_int):
        overlay = self.getOverlay()
        self.imgItem.setImage(overlay)

    def toggleOverlay(self, checked):
        if checked:
            self.graphLayout.addItem(self.imgGrad, row=1, col=0)
            self.alphaScrollBar.show()
            self.alphaScrollBar_label.show()
        else:
            self.graphLayout.removeItem(self.imgGrad)
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

    dataFrame = QtCore.Property(pd.DataFrame, fget=dataFrame,
                                    fset=setDataFrame)

    @QtCore.Slot(int, QtCore.Qt.Orientation, result=str)
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



        mainContainer = QWidget()
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

        self.cancel = True

        super().__init__(parent)
        self.setWindowTitle('Reference z-slice info absent')

        mainLayout = QVBoxLayout()
        buttonsLayout = QGridLayout()

        txt = html_utils.paragraph(f"""
            You loaded the fluorescent file called<br><br>{filename}<br><br>
            however you <b>never selected which z-slice</b><br> you want to use
            when calculating metrics<br> (e.g., mean, median, amount...etc.)<br><br>
            Choose one of following options:
        """, center=True
        )
        infoLabel = QLabel(txt)
        mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        runDataPrepButton = QPushButton(
            '   Visualize the data now and select a z-slice    '
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

        

        cancelButtonLayout = QHBoxLayout()
        cancelButton = widgets.cancelPushButton('Cancel')
        cancelButtonLayout.addStretch(1)
        cancelButtonLayout.addWidget(cancelButton)
        cancelButtonLayout.addStretch(1)
        cancelButtonLayout.setStretch(1,1)
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
        if hasattr(self, 'loop'):
            self.loop.exit()

class SelectSegmFileDialog(QDialog):
    def __init__(
            self, images_ls, parent_path, parent=None, 
            addNewFileButton=False, basename='', infoText=None, 
            fileType='segmentation'
        ):
        self.cancel = True
        self.selectedItemText = ''
        self.selectedItemIdx = None
        self.removeOthers = False
        self.okAllPos = False
        self.newSegmEndName = None
        self.basename = basename
        images_ls = sorted(images_ls, key=len)

        # Remove the 'segm_' part to allow filenameDialog to check if
        # a new file is existing (since we only ask for the part after
        # 'segm_')
        self.existingEndNames = [
            n.replace('segm', '', 1).replace('_', '', 1) for n in images_ls
        ]

        self.images_ls = images_ls
        self.parent_path = parent_path
        super().__init__(parent)

        informativeText = html_utils.paragraph(f"""
            The loaded Position folders already contains
            <b>{len(self.existingEndNames)} {fileType} masks</b><br>
        """)

        self.setWindowTitle(f'{fileType.capitalize()} files detected')
        is_win = sys.platform.startswith("win")

        mainLayout = QVBoxLayout()
        infoLayout = QHBoxLayout()
        selectionLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        # Standard Qt Question icon
        label = QLabel()
        standardIcon = getattr(QStyle, 'SP_MessageBoxQuestion')
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        label.setPixmap(pixmap)
        infoLayout.addWidget(label)

        infoLabel = QLabel(informativeText)
        infoLayout.addWidget(infoLabel)
        infoLayout.addStretch(1)
        mainLayout.addLayout(infoLayout)

        if infoText is None:
            infoText = f'Select which {fileType} file to load:'

        questionText = html_utils.paragraph(infoText)
        label = QLabel(questionText)
        listWidget = widgets.listWidget()
        listWidget.addItems(images_ls)
        listWidget.setCurrentRow(0)
        listWidget.itemDoubleClicked.connect(self.listDoubleClicked)
        self.items = list(images_ls)
        self.listWidget = listWidget

        okButton = widgets.okPushButton(' Load selected ')
        txt = 'Reveal in Finder...' if is_mac else 'Show in Explorer...'
        showInFileManagerButton = widgets.showInFileManagerButton(txt)
        cancelButton = widgets.cancelPushButton(' Cancel ')

        if addNewFileButton:
            newFileButton = widgets.newFilePushButton('New file...')

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
        selectionLayout.setColumnStretch(0, 1)
        selectionLayout.setColumnStretch(1, 3)
        selectionLayout.setColumnStretch(2, 1)
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
            basename=f'{self.basename}segm',
            hintText='Insert a <b>filename</b> for the segmentation file:',
            existingNames=self.existingEndNames
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
        self.selectedItemText = self.listWidget.selectedItems()[0].text()
        self.selectedItemIdx = self.items.index(self.selectedItemText)
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

class QDialogPbar(QDialog):
    def __init__(self, title='Progress', infoTxt='', parent=None):
        self.workerFinished = False
        self.aborted = False
        self.clickCount = 0
        super().__init__(parent)

        abort_text = 'Option+Command+C to abort' if is_mac else 'Ctrl+Alt+C to abort'
        self.abort_text = abort_text

        self.setWindowTitle(f'{title} ({abort_text})')
        self.setWindowFlags(Qt.Window)

        mainLayout = QVBoxLayout()
        pBarLayout = QGridLayout()

        if infoTxt:
            infoLabel = QLabel(infoTxt)
            mainLayout.addWidget(infoLabel, alignment=Qt.AlignCenter)

        self.progressLabel = QLabel()

        self.QPbar = widgets.ProgressBar(self)
        pBarLayout.addWidget(self.QPbar, 0, 0)
        self.ETA_label = QLabel('NDh:NDm:NDs')
        pBarLayout.addWidget(self.ETA_label, 0, 1)

        self.metricsQPbar = widgets.ProgressBar(self)
        self.metricsQPbar.setValue(0)
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
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(f"""
            Aborting with <code>{self.abort_text}</code> is <b>not safe</b>.<br><br>
            The system status cannot be predicted and
            it will <b>require a restart</b>.<br><br>
            Are you sure you want to abort?
        """)
        yesButton, noButton = msg.critical(
            self, 'Are you sure you want to abort?', txt,
            buttonsTexts=('Yes', 'No')
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

class QDialogModelParams(QDialog):
    def __init__(
            self, init_params, segment_params, model_name, is_tracker=False,
            url=None, parent=None, initLastParams=True, posData=None, 
            channels=None, currentChannelName=None, segmFileEndnames=None,
            df_metadata=None, force_postprocess_2D=False, model_module=None,
            action_type='', addPreProcessParams=True
        ):
        self.cancel = True
        super().__init__(parent)
        self.channels = channels
        self.is_tracker = is_tracker
        self.currentChannelName = currentChannelName
        self.channelCombobox = None
        self.segmFileEndnames = segmFileEndnames
        self.df_metadata = df_metadata
        self.force_postprocess_2D = force_postprocess_2D

        if segment_params[0].name.lower().find('skip_segmentation') != -1:
            self.skipSegmentation = True
            addPreProcessParams = False
        else:
            self.skipSegmentation = False
        
        if is_tracker:
            self.ini_filename = 'last_params_trackers.ini'
            addPreProcessParams = False
        else:
            self.ini_filename = 'last_params_segm_models.ini'

        self.addPreProcessParams = addPreProcessParams
        
        self.model_name = model_name

        self.setWindowTitle(f'{model_name} parameters')

        mainLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        loadFunc = self.loadLastSelection

        self.scrollArea = widgets.ScrollArea()
        scrollAreaLayout = QVBoxLayout()
        self.scrollArea.setVerticalLayout(scrollAreaLayout)
        
        self.preProcessParamsGroupbox = None
        if addPreProcessParams:
            self.preProcessParamsGroupbox = PreProcessParamsGroupbox(
                parent=self
            )
            self.preProcessParamsGroupbox.setChecked(False)
            scrollAreaLayout.addWidget(self.preProcessParamsGroupbox)
            self.preProcessParamsGroupbox.sigLoadRecipe.connect(
                self.loadPreprocRecipe
            )
        
        initGroupBox, self.init_argsWidgets = self.createGroupParams(
            init_params, 'Parameters for model initialization'
        )
        initDefaultButton = widgets.reloadPushButton('Restore default')
        initLoadLastSelButton = QPushButton('Load last parameters')
        initLoadLastSelButton.setIcon(QIcon(':folder-open.svg'))
        initButtonsLayout = QHBoxLayout()
        initButtonsLayout.addStretch(1)
        initButtonsLayout.addWidget(initDefaultButton)
        initButtonsLayout.addWidget(initLoadLastSelButton)
        initDefaultButton.clicked.connect(self.restoreDefaultInit)
        initLoadLastSelButton.clicked.connect(
            partial(loadFunc, f'{self.model_name}.init', self.init_argsWidgets)
        )

        if not self.skipSegmentation:
            if action_type:
                runGroupboxTitle = f'Parameters for {action_type}'
            elif is_tracker:
                runGroupboxTitle = 'Parameters for tracking'
            else:
                runGroupboxTitle = 'Parameters for segmentation'
            
            segmentGroupBox, self.argsWidgets = self.createGroupParams(
                segment_params, runGroupboxTitle, 
                addChannelSelector=True
            ) 
            self.segmentGroupBox = segmentGroupBox
            segmentDefaultButton = widgets.reloadPushButton('Restore default')
            segmentLoadLastSelButton = QPushButton('Load last parameters')
            segmentButtonsLayout = QHBoxLayout()
            segmentButtonsLayout.addStretch(1)
            segmentButtonsLayout.addWidget(segmentDefaultButton)
            segmentButtonsLayout.addWidget(segmentLoadLastSelButton)
            segmentDefaultButton.clicked.connect(self.restoreDefaultSegment)
            section = f'{self.model_name}.segment'
            segmentLoadLastSelButton.clicked.connect(
                partial(loadFunc, section, self.argsWidgets)
            )

        cancelButton = widgets.cancelPushButton(' Cancel ')
        okButton = widgets.okPushButton(' Ok ')
            # infoButton = widgets.infoPushButton(' Help... ')
            # restoreDefaultButton = widgets.reloadPushButton('Restore default')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
            # buttonsLayout.addWidget(infoButton)
            # buttonsLayout.addWidget(restoreDefaultButton)
        buttonsLayout.addWidget(okButton)

        buttonsLayout.setContentsMargins(0, 10, 0, 10)

        okButton.clicked.connect(self.ok_cb)
            # infoButton.clicked.connect(self.info_params)
        cancelButton.clicked.connect(self.close)
            # restoreDefaultButton.clicked.connect(self.restoreDefault)

        scrollAreaLayout.addWidget(initGroupBox)
        scrollAreaLayout.addLayout(initButtonsLayout)
        if not self.skipSegmentation:
            scrollAreaLayout.addSpacing(15)
            scrollAreaLayout.addStretch(1)
            scrollAreaLayout.addWidget(segmentGroupBox)
            scrollAreaLayout.addLayout(segmentButtonsLayout)

        self.postProcessGroupbox = None
        
        if not is_tracker:
            # Add minimum size spinbox which is valid for all models
            postProcessGroupbox = PostProcessSegmParams(
                'Post-processing segmentation parameters', posData,
                force_postprocess_2D=force_postprocess_2D
            )
            postProcessGroupbox.setCheckable(True)
            postProcessGroupbox.setChecked(False)
            self.postProcessGroupbox = postProcessGroupbox

            scrollAreaLayout.addSpacing(15)
            scrollAreaLayout.addStretch(1)
            scrollAreaLayout.addWidget(postProcessGroupbox)

            postProcDefaultButton = widgets.reloadPushButton('Restore default')
            postProcLoadLastSelButton = QPushButton('Load last parameters')
            postProcButtonsLayout = QHBoxLayout()
            postProcButtonsLayout.addStretch(1)
            postProcButtonsLayout.addWidget(postProcDefaultButton)
            postProcButtonsLayout.addWidget(postProcLoadLastSelButton)
            postProcDefaultButton.clicked.connect(self.restoreDefaultPostprocess)
            postProcLoadLastSelButton.clicked.connect(
                self.loadLastSelectionPostProcess
            )
            scrollAreaLayout.addLayout(postProcButtonsLayout)

            if url is not None:
                scrollAreaLayout.addWidget(
                    self.createSeeHereLabel(url), alignment=Qt.AlignCenter
                )

        mainLayout.addWidget(self.scrollArea)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.configPars = self.readLastSelection()
        if self.configPars is None:
            initLoadLastSelButton.setDisabled(True)
            segmentLoadLastSelButton.setDisabled(True)
            if not is_tracker:
                postProcLoadLastSelButton.setDisabled(True)

        if initLastParams:
            initLoadLastSelButton.click()
            if not self.skipSegmentation:
                segmentLoadLastSelButton.click()
        
        if not is_tracker and not self.skipSegmentation:
            postProcLoadLastSelButton.click()

        try:
            self.connectCustomSignals(model_module)
        except Exception as e:
            printl(traceback.format_exc())
        
        self.setLayout(mainLayout)
        self.setFont(font)
        # self.setModal(True)

    
    def loadPreprocRecipe(self):
        if self.configPars is None:
            return
        
        preprocConfigPars = {}
        for section in self.configPars.sections():
            if not section.startswith(f'{self.model_name}.preprocess'):
                continue      
            
            preprocConfigPars[section] = self.configPars[section]
        
        if not preprocConfigPars:
            return
        
        self.preProcessParamsGroupbox.loadRecipe(preprocConfigPars)
    
    def connectCustomSignals(self, model_module):
        if model_module is None:
            return

        if not hasattr(model_module, 'CustomSignals'):
            return
        
        customSignals = model_module.CustomSignals()
        for slot_info in customSignals.slots_info:
            group = slot_info['group']
            widget_name = slot_info['widget_name']
            if group == 'init':
                ArgsWidgets_list = self.init_argsWidgets
            else:
                ArgsWidgets_list = self.argsWidgets
            for argwidget in ArgsWidgets_list:
                if argwidget.name == widget_name:
                    signal = getattr(argwidget.widget, slot_info['signal'])
                    signal.connect(partial(slot_info['slot'], self))
                    break
    
    def selectedFeaturesRange(self):
        if self.postProcessGroupbox is None:
            return {}
        return self.postProcessGroupbox.selectedFeaturesRange()

    def groupedFeatures(self):
        if self.postProcessGroupbox is None:
            return {}
        return self.postProcessGroupbox.groupedFeatures()
    
    def setChannelNames(self, chNames):
        if not hasattr(self, 'channelsCombobox'):
            return
        
        items = ['None']
        items.extend(chNames)
        self.channelsCombobox.addItems(items)
    
    def getValueFromMetadata(self, name):
        try:
            value = self.df_metadata.at[name, 'values']
        except Exception as e:
            # traceback.print_exc()
            value = None
        return value

    def criticalSegmFileRequiredButNoneAvailable(self):
        model_name = f'{self.model_name} model'
        action_txt = (
            'Please, segment the correct channel before using '
            f'{self.model_name}.'
        )
        if self.model_name == 'skip_segmentation':
            model_name = 'Skipping the segmentation'
            action_txt = (
                'To be able to skip the segmentation step, you need '
                'create at least one segmentation file.'
            )
        txt = html_utils.paragraph(f"""
            <b>{model_name}</b> 
            <b>requires an additional segmentation file</b> 
            but there are none available!<br><br>
            {action_txt}
            <br><br>Thank you for you patience!
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Segmentation file required', txt)
        raise FileNotFoundError(
            'Model requires segmentation file but none are available.'
        )
    
    
    def checkAddSegmEndnameCombobox(self, ArgSpec, groupBoxLayout, row):
        if ArgSpec.name != 'Auxiliary segmentation file':
            return False
        
        if self.segmFileEndnames is None or not self.segmFileEndnames:
            self.criticalSegmFileRequiredButNoneAvailable()
        
        label = QLabel(f'{ArgSpec.name}:  ')
        groupBoxLayout.addWidget(
            label, row, 0, alignment=Qt.AlignRight
        )
        items = self.segmFileEndnames
        self.segmEndnameCombobox = widgets.QCenteredComboBox()
        self.segmEndnameCombobox.addItems(items)
        groupBoxLayout.addWidget(self.segmEndnameCombobox, row, 1, 1, 2)
        return True
        
    
    def createGroupParams(self, ArgSpecs_list, groupName, addChannelSelector=False):
        ArgWidget = namedtuple(
            'ArgsWidgets',
            ['name', 'type', 'widget', 'defaultVal', 'valueSetter', 'valueGetter']
        )
        ArgsWidgets_list = []
        groupBox = QGroupBox(groupName)
        groupBoxLayout = QGridLayout()

        start_row = 0
        if self.is_tracker and self.channels is not None and addChannelSelector:
            label = QLabel(f'Input image:  ')
            groupBoxLayout.addWidget(
                label, start_row, 0, alignment=Qt.AlignRight
            )
            items = ['None', *self.channels]
            self.channelCombobox = widgets.QCenteredComboBox()
            self.channelCombobox.addItems(items)
            groupBoxLayout.addWidget(self.channelCombobox, start_row, 1, 1, 2)
            if self.currentChannelName is not None:
                self.channelCombobox.setCurrentText(self.currentChannelName)
            infoText = (
                'Some trackers require the intensity image as input.<br><br>'
                'If this one does not require it, leave the selected value '
                'to `None`.'
            )
            infoButton = self.getInfoButton('Input image', infoText)
            groupBoxLayout.addWidget(infoButton, start_row, 3)
            start_row += 1
        
        addSecondChannelSelector = addChannelSelector
        if addSecondChannelSelector and ArgSpecs_list[0].docstring is not None:
            if ArgSpecs_list[0].docstring.lower().find('single channel only') != -1:
                addSecondChannelSelector = False
        
        if self.model_name.find('cellpose') != -1 and addSecondChannelSelector:
            label = QLabel('Second channel (optional):  ')
            groupBoxLayout.addWidget(label, start_row, 0, alignment=Qt.AlignRight)
            self.channelsCombobox = widgets.QCenteredComboBox()
            groupBoxLayout.addWidget(self.channelsCombobox, start_row, 1, 1, 2)
            infoText = (
                'Some cellpose models can merge two channels (e.g., cyto + '
                'nucleus) to obtain better perfomance.\n\n'
                'Select a channel as additional input to the model.'
            )
            infoButton = self.getInfoButton('Second channel', infoText)
            groupBoxLayout.addWidget(infoButton, start_row, 3)
            start_row += 1
        
        for row, ArgSpec in enumerate(ArgSpecs_list):
            try:
                not_a_param = ArgSpec.type().not_a_param
                continue
            except Exception as err:
                pass
            
            try:
                # If a parameter if None, python initializes it to 
                # typing.Optional and we need to access the first type
                ArgSpec.type.__args__[0]().not_a_param
                continue
            except Exception as err:
                pass
            
            row = row + start_row
            skip = self.checkAddSegmEndnameCombobox(
                ArgSpec, groupBoxLayout, row
            )
            if skip:
                continue
            
            arg_name = ArgSpec.name         
            var_name = arg_name.replace('_', ' ')
            var_name = f'{var_name[0].upper()}{var_name[1:]}'
            label = QLabel(f'{var_name}:  ')
            metadata_val = self.getValueFromMetadata(ArgSpec.name)
            groupBoxLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
            try:
                values = ArgSpec.type().values
                isCustomListType = True
            except Exception as err:
                isCustomListType = False
            
            isVectorEntry = False
            try:
                if isinstance(ArgSpec.type(), types.Vector):
                    isVectorEntry = True
            except Exception as err:
                pass
            
            isFolderPath = False
            try:
                if isinstance(ArgSpec.type(), types.FolderPath):
                    isFolderPath = True
            except Exception as err:
                pass
            
            isCustomWidget = hasattr(ArgSpec.type, 'isWidget')
            
            if isCustomWidget:
                widget = ArgSpec.type()
                defaultVal = ArgSpec.default
                valueSetter = ArgSpec.type.setValue
                valueGetter = ArgSpec.type.value
                groupBoxLayout.addWidget(widget, row, 1, 1, 2)
            elif isVectorEntry:
                vectorLineEdit = widgets.VectorLineEdit()
                vectorLineEdit.setValue(ArgSpec.default)
                defaultVal = ArgSpec.default
                valueSetter = widgets.VectorLineEdit.setValue
                valueGetter = widgets.VectorLineEdit.value
                widget = vectorLineEdit
                groupBoxLayout.addWidget(vectorLineEdit, row, 1, 1, 2)
            elif isFolderPath:
                folderPathControl = widgets.FolderPathControl()
                folderPathControl.setText(str(ArgSpec.default))
                widget = folderPathControl
                defaultVal = str(ArgSpec.default)
                valueSetter = widgets.FolderPathControl.setText
                valueGetter = widgets.FolderPathControl.path
                groupBoxLayout.addWidget(folderPathControl, row, 1, 1, 2)
            elif ArgSpec.type == bool:
                booleanGroup = QButtonGroup()
                booleanGroup.setExclusive(True)
                checkBox = widgets.Toggle()
                checkBox.setChecked(ArgSpec.default)
                defaultVal = ArgSpec.default
                valueSetter = widgets.Toggle.setChecked
                valueGetter = widgets.Toggle.isChecked
                widget = checkBox
                groupBoxLayout.addWidget(
                    checkBox, row, 1, 1, 2, alignment=Qt.AlignCenter
                )
            elif ArgSpec.type == int:
                spinBox = widgets.SpinBox()
                if metadata_val is None:
                    spinBox.setValue(ArgSpec.default)
                else:
                    spinBox.setValue(int(metadata_val))
                    spinBox.isMetadataValue = True
                defaultVal = ArgSpec.default
                valueSetter = QSpinBox.setValue
                valueGetter = QSpinBox.value
                widget = spinBox
                groupBoxLayout.addWidget(spinBox, row, 1, 1, 2)
            elif ArgSpec.type == float:
                doubleSpinBox = widgets.FloatLineEdit()
                if metadata_val is None:
                    doubleSpinBox.setValue(ArgSpec.default)
                else:
                    doubleSpinBox.setValue(float(metadata_val))
                    doubleSpinBox.isMetadataValue = True
                widget = doubleSpinBox
                defaultVal = ArgSpec.default
                valueSetter = widgets.FloatLineEdit.setValue
                valueGetter = widgets.FloatLineEdit.value
                groupBoxLayout.addWidget(doubleSpinBox, row, 1, 1, 2)
            elif ArgSpec.type == os.PathLike:
                filePathControl = widgets.filePathControl()
                filePathControl.setText(str(ArgSpec.default))
                widget = filePathControl
                defaultVal = str(ArgSpec.default)
                valueSetter = widgets.filePathControl.setText
                valueGetter = widgets.filePathControl.path
                groupBoxLayout.addWidget(filePathControl, row, 1, 1, 2)
            elif isCustomListType:
                items = ArgSpec.type().values
                defaultVal = str(ArgSpec.default)
                combobox = widgets.AlphaNumericComboBox()
                combobox.addItems(items)
                combobox.setCurrentValue(defaultVal)
                valueSetter = widgets.AlphaNumericComboBox.setCurrentValue
                valueGetter = widgets.AlphaNumericComboBox.currentValue
                widget = combobox
                groupBoxLayout.addWidget(combobox, row, 1, 1, 2)
            else:
                lineEdit = QLineEdit()
                lineEdit.setText(str(ArgSpec.default))
                lineEdit.setAlignment(Qt.AlignCenter)
                widget = lineEdit
                defaultVal = str(ArgSpec.default)
                valueSetter = QLineEdit.setText
                valueGetter = QLineEdit.text
                groupBoxLayout.addWidget(lineEdit, row, 1, 1, 2)
            
            if ArgSpec.desc:
                infoButton = self.getInfoButton(ArgSpec.name, ArgSpec.desc)
                groupBoxLayout.addWidget(infoButton, row, 3)
            
            argsInfo = ArgWidget(
                name=ArgSpec.name,
                type=ArgSpec.type,
                widget=widget,
                defaultVal=defaultVal,
                valueSetter=valueSetter,
                valueGetter=valueGetter
            )
            ArgsWidgets_list.append(argsInfo)

        groupBoxLayout.setColumnStretch(0, 0)
        groupBoxLayout.setColumnStretch(1, 1)
        groupBoxLayout.setColumnStretch(3, 0)
        
        groupBox.setLayout(groupBoxLayout)
        return groupBox, ArgsWidgets_list

    def getInfoButton(self, param_name, infoText):
        infoButton = widgets.infoPushButton()
        infoButton.param_name = param_name
        infoButton.setToolTip(
            f'Click to get more info about `{param_name}` parameter...'
        )
        infoButton.infoText = infoText
        infoButton.clicked.connect(self.showInfoParam)
        return infoButton
    
    def showInfoParam(self):
        text = self.sender().infoText
        text = text.replace('\n', '<br>')
        text = html_utils.rst_urls_to_html(text)
        text = html_utils.rst_to_html(text)
        text = html_utils.paragraph(text)
        param_name = self.sender().param_name
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, f'Info about `{param_name}` parameter', text)
    
    def restoreDefaultInit(self):
        for argWidget in self.init_argsWidgets:
            defaultVal = argWidget.defaultVal
            widget = argWidget.widget
            argWidget.valueSetter(widget, defaultVal)

    def restoreDefaultSegment(self):
        for argWidget in self.argsWidgets:
            defaultVal = argWidget.defaultVal
            widget = argWidget.widget
            argWidget.valueSetter(widget, defaultVal)

    def restoreDefaultPostprocess(self):
        self.postProcessGroupbox.restoreDefault()

    def readLastSelection(self):
        self.ini_path = os.path.join(settings_folderpath, self.ini_filename)
        if not os.path.exists(self.ini_path):
            return None

        configPars = config.ConfigParser()
        configPars.read(self.ini_path)
        return configPars

    def setValuesFromParams(self, init_params, segment_params):
        sections = {
            f'{self.model_name}.init': (init_params, self.init_argsWidgets),
            f'{self.model_name}.segment': (segment_params, self.argsWidgets)
        }
        for section, values in sections.items():
            params, argWidgetList = values
            for argWidget in argWidgetList:
                val = params.get(argWidget.name)
                widget = argWidget.widget
                if val is None:
                    continue
                casters = [lambda x: x, int, float, str, bool]
                for caster in casters:
                    try:
                        argWidget.valueSetter(widget, caster(val))
                        break
                    except Exception as e:
                        continue
    
    def loadLastSelection(self, section, argWidgetList):
        if self.configPars is None:
            return

        getters = ['getboolean', 'getint', 'getfloat', 'get']
        try:
            options = self.configPars.options(section)
        except Exception:
            return

        for argWidget in argWidgetList:
            option = argWidget.name
            val = None
            for getter in getters:
                try:
                    val = getattr(self.configPars, getter)(section, option)
                    break
                except Exception as err:
                    pass
            widget = argWidget.widget
            if hasattr(widget, 'isMetadataValue'):
                continue
            if val is None:
                continue
            casters = [lambda x: x, int, float, str, bool]
            for caster in casters:
                try:
                    argWidget.valueSetter(widget, caster(val))
                    break
                except Exception as e:
                    continue

    def loadLastSelectionPostProcess(self):
        postProcessSection = f'{self.model_name}.postprocess'

        if postProcessSection not in self.configPars.sections():
            return

        try:
            minSize = self.configPars.getint(
                postProcessSection, 'minSize', fallback=10
            )
        except ValueError:
            minSize = 10

        try:
            minSolidity = self.configPars.getfloat(
                postProcessSection, 'minSolidity', fallback=0.5
            )
        except ValueError:
            minSolidity = 0.5

        try: 
            maxElongation = self.configPars.getfloat(
                postProcessSection, 'maxElongation', fallback=3
            )
        except ValueError:
            maxElongation = 3
        
        try:
            minObjSizeZ = self.configPars.getint(
                postProcessSection, 'min_obj_no_zslices', fallback=3
            )
        except ValueError:
            minObjSizeZ = 3
        
        kwargs = {
            'min_solidity': minSolidity,
            'min_area': minSize,
            'max_elongation': maxElongation,
            'min_obj_no_zslices': minObjSizeZ
        }
        self.postProcessGroupbox.restoreFromKwargs(kwargs)

        applyPostProcessing = self.configPars.getboolean(
            postProcessSection, 'applyPostProcessing'
        )
        self.postProcessGroupbox.setChecked(applyPostProcessing)

    def createSeeHereLabel(self, url):
        htmlTxt = f'<a href=\"{url}">here</a>'
        seeHereLabel = QLabel()
        seeHereLabel.setText(f"""
            <p style="font-size:13px">
                See {htmlTxt} for details on the parameters
            </p>
        """)
        seeHereLabel.setTextFormat(Qt.RichText)
        seeHereLabel.setTextInteractionFlags(Qt.TextBrowserInteraction)
        seeHereLabel.setOpenExternalLinks(True)
        seeHereLabel.setStyleSheet("padding:12px 0px 0px 0px;")
        return seeHereLabel

    def argsWidgets_to_kwargs(self, argsWidgets):
        kwargs_dict = {
            argWidget.name:argWidget.valueGetter(argWidget.widget)
            for argWidget in argsWidgets
        }
        return kwargs_dict

    def ok_cb(self, checked):
        self.cancel = False
        self.preproc_recipe = None
        if self.preProcessParamsGroupbox is not None:
            self.preproc_recipe = self.preProcessParamsGroupbox.recipe()
            
        self.init_kwargs = self.argsWidgets_to_kwargs(self.init_argsWidgets)
        if not self.skipSegmentation:
            self.model_kwargs = self.argsWidgets_to_kwargs(
                self.argsWidgets
            )
        else:
            self.model_kwargs = {}
        if hasattr(self, 'segmEndnameCombobox'):
            self.init_kwargs['segm_endname'] = (
                self.segmEndnameCombobox.currentText()
            )
        if self.postProcessGroupbox is not None:
            self.applyPostProcessing = self.postProcessGroupbox.isChecked()
            self.standardPostProcessKwargs = self.postProcessGroupbox.kwargs()
        self.secondChannelName = None
        if hasattr(self, 'channelsCombobox'):
            self.secondChannelName = self.channelsCombobox.currentText()
        if self.secondChannelName == 'None':
            self.secondChannelName = None
        self.inputChannelName = 'None'
        if self.channelCombobox is not None:
            self.inputChannelName = self.channelCombobox.currentText()
        
        self.customPostProcessFeatures = self.selectedFeaturesRange()
        self.customPostProcessGroupedFeatures = self.groupedFeatures()
        self._saveParams()
        self.close()

    def _saveParams(self):
        if self.configPars is None:
            self.configPars = config.ConfigParser()
        
        if self.preProcessParamsGroupbox is not None:
            preprocCp = self.preProcessParamsGroupbox.recipeConfigPars(
                self.model_name
            )
            for section in preprocCp.sections():
                self.configPars[section] = preprocCp[section]
        
        self.configPars[f'{self.model_name}.init'] = {}
        self.configPars[f'{self.model_name}.segment'] = {}
        for key, val in self.init_kwargs.items():
            self.configPars[f'{self.model_name}.init'][key] = str(val)
        for key, val in self.model_kwargs.items():
            self.configPars[f'{self.model_name}.segment'][key] = str(val)

        self.configPars[f'{self.model_name}.postprocess'] = {}
        if self.postProcessGroupbox is not None:
            postProcKwargs = self.postProcessGroupbox.kwargs()
            postProcessConfig = self.configPars[f'{self.model_name}.postprocess']
            postProcessConfig['minSize'] = str(postProcKwargs['min_area'])
            postProcessConfig['minSolidity'] = str(postProcKwargs['min_solidity'])
            postProcessConfig['maxElongation'] = str(
                postProcKwargs['max_elongation']
            )
            postProcessConfig['min_obj_no_zslices'] = str(
                postProcKwargs['min_obj_no_zslices']
            )
            postProcessConfig['applyPostProcessing'] = str(
                self.postProcessGroupbox.isChecked()
            )

        with open(self.ini_path, 'w') as configfile:
            self.configPars.write(configfile)

        print(f'Segmentation parameters saved at "{self.ini_path}"')
        
    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()
        if self.model_name == 'thresholding':
            self.segmentGroupBox.setDisabled(True)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()
    
    def showEvent(self, event) -> None:
        height = self.scrollArea.minimumHeightNoScrollbar() + 70
        self.move(self.pos().x(), 20)
        screenHeight = self.screen().size().height()
        if height >= screenHeight - 150:
            height = screenHeight - 150
        self.resize(self.width(), height)

class downloadModel:
    def __init__(self, model_name, parent=None):
        self.loop = None
        self.model_name = model_name
        self._parent = parent

    def download(self):
        if myutils._model_url(self.model_name) is None:
            return
        _, model_path = myutils.get_model_path(
            self.model_name, create_temp_dir=False
        )
        model_name = self.model_name
        model_exists = myutils.check_model_exists(model_path, model_name)
        if not model_exists:
            self.warnDownloadModel(model_path, self.model_name)
        try:
            self._parent.logger.info(
                f'Downloading {self.model_name} model(s) to "{model_path}"'
            )
        except Exception as err:
            pass
        
        success = myutils.download_model(self.model_name)
        if not success:
            self.criticalDowloadFailed()
        
    def warnDownloadModel(self, model_path, model_name):
        txt = html_utils.paragraph(
            'Cell-ACDC needs to <b>download the model</b> '
            f'<code>{model_name}</code>.<br><br>'
            'The files will be dowloaded into the following folder:<br><br>'
            f'<code>{model_path}</code><br><br>'
            '<b>Progress</b> will be displayed in the <b>terminal</b>.<br>'
        )
        msg = widgets.myMessageBox()
        msg.information(self._parent, 'Download model', txt)

    def criticalDowloadFailed(self):
        import cellacdc
        model_name = self.model_name
        m = model_name.lower()
        weights_filenames = getattr(cellacdc, f'{m}_weights_filenames')
        url, alternative_url = myutils._model_url(
            model_name, return_alternative=True
        )
        url_href = f'<a href="{url}">this link</a>'
        alternative_url_href = f'<a href="{alternative_url}">this link</a>'
        _, model_path = myutils.get_model_path(model_name, create_temp_dir=False)
        txt = html_utils.paragraph(f"""
            Automatic download of {model_name} failed.<br><br>
            Please, <b>manually download</b> the model weights from {url_href} or
            {alternative_url_href}.<br><br>
            Next, unzip the content (or move the files if not a zip archive) 
            of the downloaded file into the following folder:<br><br>
            <code>{model_path}</code><br><br>
            <i>NOTE: if clicking on the link above does not work
            copy one of the links below and paste it into the browser</i><br><br>
            <code>{url}</code>
            <br><br>
            <code>{alternative_url}</code>
        """)
        weights_paths = [os.path.join(model_path, f) for f in weights_filenames]
        weights = '\n\n'.join(weights_paths)
        detailsText = (
            f'Files that {model_name} requires:\n\n'
            f'{weights}'
        )
        msg = widgets.myMessageBox()
        msg.critical(
            self._parent, f'Download of {model_name} failed', txt, 
            detailsText=detailsText
        )
        self.close_()

    def close_(self):
        return
        # self.hide()
        # self.close()
        # if self.loop is not None:
        #     self.loop.exit()

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
        <p style=font-size:13px>
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
        okButton = widgets.okPushButton('Ok')
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

class combineMetricsEquationDialog(QBaseDialog):
    sigOk = Signal(object)

    def __init__(
            self, allChNames, isZstack, isSegm3D, parent=None, debug=False,
            closeOnOk=True
        ):
        super().__init__(parent)

        self.setWindowTitle('Add combined measurement')

        self.initAttributes()
        
        self.allChNames = allChNames

        self.cancel = True
        self.isOperatorMode = False
        self.closeOnOk = closeOnOk

        mainLayout = QVBoxLayout()
        equationLayout = QHBoxLayout()

        metricsTreeWidget = QTreeWidget()
        metricsTreeWidget.setHeaderHidden(True)
        metricsTreeWidget.setFont(font)
        self.metricsTreeWidget = metricsTreeWidget

        for chName in allChNames:
            channelTreeItem = QTreeWidgetItem(metricsTreeWidget)
            channelTreeItem.setText(0, f'{chName} measurements')
            metricsTreeWidget.addTopLevelItem(channelTreeItem)

            metrics_desc, bkgr_val_desc = measurements.standard_metrics_desc(
                isZstack, chName, isSegm3D=isSegm3D
            )
            custom_metrics_desc = measurements.custom_metrics_desc(
                isZstack, chName, isSegm3D=isSegm3D
            )

            foregrMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
            foregrMetricsTreeItem.setText(0, 'Cell signal measurements')
            channelTreeItem.addChild(foregrMetricsTreeItem)

            bkgrMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
            bkgrMetricsTreeItem.setText(0, 'Background values')
            channelTreeItem.addChild(bkgrMetricsTreeItem)

            if custom_metrics_desc:
                customMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
                customMetricsTreeItem.setText(0, 'Custom measurements')
                channelTreeItem.addChild(customMetricsTreeItem)

            self.addTreeItems(
                foregrMetricsTreeItem, metrics_desc.keys(), isCol=True
            )
            self.addTreeItems(
                bkgrMetricsTreeItem, bkgr_val_desc.keys(), isCol=True
            )

            if custom_metrics_desc:
                self.addTreeItems(
                    customMetricsTreeItem, custom_metrics_desc.keys(),
                    isCol=True
                )

        self.addChannelLessItems(isZstack, isSegm3D=isSegm3D)

        sizeMetricsTreeItem = QTreeWidgetItem(metricsTreeWidget)
        sizeMetricsTreeItem.setText(0, 'Size measurements')
        metricsTreeWidget.addTopLevelItem(sizeMetricsTreeItem)

        size_metrics_desc = measurements.get_size_metrics_desc(
            isSegm3D, True
        )
        self.addTreeItems(
            sizeMetricsTreeItem, size_metrics_desc.keys(), isCol=True
        )

        propMetricsTreeItem = QTreeWidgetItem(metricsTreeWidget)
        propMetricsTreeItem.setText(0, 'Region properties')
        metricsTreeWidget.addTopLevelItem(propMetricsTreeItem)

        props_names = measurements.get_props_names()
        self.addTreeItems(
            propMetricsTreeItem, props_names, isCol=True
        )

        operatorsLayout = QHBoxLayout()
        operatorsLayout.addStretch(1)

        iconSize = 24

        self.operatorButtons = []
        self.operators = [
            ('add', '+'),
            ('subtract', '-'),
            ('multiply', '*'),
            ('divide', '/'),
            ('open_bracket', '('),
            ('close_bracket', ')'),
            ('square', '**2'),
            ('pow', '**'),
            ('ln', 'log('),
            ('log10', 'log10('),
        ]
        operatorFont = QFont()
        operatorFont.setPixelSize(16)
        for name, text in self.operators:
            button = QPushButton()
            button.setIcon(QIcon(f':{name}.svg'))
            button.setIconSize(QSize(iconSize,iconSize))
            button.text = text
            operatorsLayout.addWidget(button)
            self.operatorButtons.append(button)
            button.clicked.connect(self.addOperator)
            # button.setFont(operatorFont)

        clearButton = QPushButton()
        clearButton.setIcon(QIcon(':clear.svg'))
        clearButton.setIconSize(QSize(iconSize,iconSize))
        clearButton.setFont(operatorFont)

        clearEntryButton = QPushButton()
        clearEntryButton.setIcon(QIcon(':backspace.svg'))
        clearEntryButton.setFont(operatorFont)
        clearEntryButton.setIconSize(QSize(iconSize,iconSize))

        operatorsLayout.addWidget(clearButton)
        operatorsLayout.addWidget(clearEntryButton)
        operatorsLayout.addStretch(1)

        newColNameLayout = QVBoxLayout()
        newColNameLineEdit = widgets.alphaNumericLineEdit()
        newColNameLineEdit.setAlignment(Qt.AlignCenter)
        self.newColNameLineEdit = newColNameLineEdit
        newColNameLayout.addStretch(1)
        newColNameLayout.addWidget(QLabel('New measurement name:'))
        newColNameLayout.addWidget(newColNameLineEdit)
        newColNameLayout.addStretch(1)

        equationDisplayLayout = QVBoxLayout()
        equationDisplayLayout.addWidget(QLabel('Equation:'))
        equationDisplay = QPlainTextEdit()
        # equationDisplay.setReadOnly(True)
        self.equationDisplay = equationDisplay
        equationDisplayLayout.addWidget(equationDisplay)
        equationDisplayLayout.setStretch(0,0)
        equationDisplayLayout.setStretch(1,1)

        equationLayout.addLayout(newColNameLayout)
        equationLayout.addWidget(QLabel(' = '))
        equationLayout.addLayout(equationDisplayLayout)
        equationLayout.setStretch(0,1)
        equationLayout.setStretch(1,0)
        equationLayout.setStretch(2,2)

        testOutputLayout = QVBoxLayout()
        testOutputLayout.addWidget(QLabel('Result of test with random inputs:'))
        testOutputDisplay = QTextEdit()
        testOutputDisplay.setReadOnly(True)
        self.testOutputDisplay = testOutputDisplay
        testOutputLayout.addWidget(testOutputDisplay)
        testOutputLayout.setStretch(0,0)
        testOutputLayout.setStretch(1,1)

        instructions = html_utils.paragraph("""
            <b>Double-click</b> on any of the <b>available measurements</b>
            to add it to the equation.<br><br>
            <i>NOTE: the result will be saved in the <code>acdc_output.csv</code>
            file as a column with the same name<br>
            you enter in "New measurement name"
            field.</i><br>
        """)

        buttonsLayout = QHBoxLayout()

        cancelButton = widgets.cancelPushButton('Cancel')
        helpButton = widgets.infoPushButton('  Help...')
        testButton = widgets.calcPushButton('Test output')
        okButton = widgets.okPushButton(' Ok ')
        okButton.setDisabled(True)
        self.okButton = okButton

        buttonsLayout.addStretch(1)

        if debug:
            debugButton = QPushButton('Debug')
            debugButton.clicked.connect(self._debug)
            buttonsLayout.addWidget(debugButton)

        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(helpButton)
        buttonsLayout.addWidget(testButton)
        buttonsLayout.addWidget(okButton)

        mainLayout.addWidget(QLabel(instructions))
        mainLayout.addWidget(QLabel('Available measurements:'))
        mainLayout.addWidget(metricsTreeWidget)
        mainLayout.addLayout(operatorsLayout)
        mainLayout.addLayout(equationLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addLayout(testOutputLayout)

        clearButton.clicked.connect(self.clearEquation)
        clearEntryButton.clicked.connect(self.clearEntryEquation)
        metricsTreeWidget.itemDoubleClicked.connect(self.addColname)

        helpButton.clicked.connect(self.showHelp)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        testButton.clicked.connect(self.test_cb)

        self.setLayout(mainLayout)
        self.setFont(font)

        self.setStyleSheet(TREEWIDGET_STYLESHEET)

    def addChannelLessItems(self, isZstack, isSegm3D=False):
        allChannelsTreeItem = QTreeWidgetItem(self.metricsTreeWidget)
        allChannelsTreeItem.setText(0, f'All channels measurements')
        metrics_desc, bkgr_val_desc = measurements.standard_metrics_desc(
            isZstack, '', isSegm3D=isSegm3D
        )
        custom_metrics_desc = measurements.custom_metrics_desc(
            isZstack, '', isSegm3D=isSegm3D
        )

        foregrMetricsTreeItem = QTreeWidgetItem(allChannelsTreeItem)
        foregrMetricsTreeItem.setText(0, 'Cell signal measurements')
        allChannelsTreeItem.addChild(foregrMetricsTreeItem)

        bkgrMetricsTreeItem = QTreeWidgetItem(allChannelsTreeItem)
        bkgrMetricsTreeItem.setText(0, 'Background values')
        allChannelsTreeItem.addChild(bkgrMetricsTreeItem)

        if custom_metrics_desc:
            customMetricsTreeItem = QTreeWidgetItem(allChannelsTreeItem)
            customMetricsTreeItem.setText(0, 'Custom measurements')
            allChannelsTreeItem.addChild(customMetricsTreeItem)

        self.addTreeItems(
            foregrMetricsTreeItem, metrics_desc.keys(), isCol=True,
            isChannelLess=True
        )
        self.addTreeItems(
            bkgrMetricsTreeItem, bkgr_val_desc.keys(), isCol=True,
            isChannelLess=True
        )

        if custom_metrics_desc:
            self.addTreeItems(
                customMetricsTreeItem, custom_metrics_desc.keys(),
                isCol=True, isChannelLess=True
            )

    def addOperator(self):
        button = self.sender()
        text = f'{self.equationDisplay.toPlainText()}{button.text}'
        self.equationDisplay.setPlainText(text)
        self.clearLenghts.append(len(button.text))

    def clearEquation(self):
        self.isOperatorMode = False
        self.equationDisplay.setPlainText('')
        self.initAttributes()

    def initAttributes(self):
        self.clearLenghts = []
        self.equationColNames = []
        self.channelLessColnames = []

    def clearEntryEquation(self):
        if not self.clearLenghts:
            return

        text = self.equationDisplay.toPlainText()
        newText = text[:-self.clearLenghts[-1]]
        clearedText = text[-self.clearLenghts[-1]:]
        self.clearLenghts.pop(-1)
        self.equationDisplay.setPlainText(newText)
        if clearedText in self.equationColNames:
            self.equationColNames.remove(clearedText)
        if clearedText in self.channelLessColnames:
            self.channelLessColnames.remove(clearedText)

    def addTreeItems(
            self, parentItem, itemsText, isCol=False, isChannelLess=False
        ):
        for text in itemsText:
            _item = QTreeWidgetItem(parentItem)
            _item.setText(0, text)
            parentItem.addChild(_item)
            if isCol:
                _item.isCol = True
            _item.isChannelLess = isChannelLess


    def addColname(self, item, column):
        if not hasattr(item, 'isCol'):
            return

        colName = item.text(0)
        text = f'{self.equationDisplay.toPlainText()}{colName}'
        self.equationDisplay.setPlainText(text)
        self.clearLenghts.append(len(colName))
        self.equationColNames.append(colName)
        if item.isChannelLess:
            self.channelLessColnames.append(colName)

    def _debug(self):
        print(self.getEquationsDict())

    def getEquationsDict(self):
        equation = self.equationDisplay.toPlainText()
        newColName = self.newColNameLineEdit.text()
        if not self.channelLessColnames:
            chNamesInTerms = set()
            for term in self.equationColNames:
                for chName in self.allChNames:
                    if chName in term:
                        chNamesInTerms.add(chName)
            if len(chNamesInTerms) == 1:
                # Equation uses metrics from a single channel --> append channel name
                chName = chNamesInTerms.pop()
                chColName = f'{chName}_{newColName}'
                isMixedChannels = False
                return {chColName:equation}, isMixedChannels
            else:
                # Equation doesn't use all channels metrics nor is single channel
                isMixedChannels = True
                return {newColName:equation}, isMixedChannels

        isMixedChannels = False
        equations = {}
        for chName in self.allChNames:
            chEquation = equation
            chEquationName = newColName
            # Append each channel name to channelLess terms
            for colName in self.channelLessColnames:
                chColName = f'{chName}{colName}'
                chEquation = chEquation.replace(colName, chColName)
                chEquationName = f'{chName}_{newColName}'
                equations[chEquationName] = chEquation
        return equations, isMixedChannels

    def ok_cb(self):
        if not self.newColNameLineEdit.text():
            self.warnEmptyEquationName()
            return
        
        self.cancel = False

        # Save equation to "<user_profile_path>/acdc-metrics/combine_metrics.ini" file
        config = measurements.read_saved_user_combine_config()

        equationsDict, isMixedChannels = self.getEquationsDict()
        for newColName, equation in equationsDict.items():
            config = measurements.add_user_combine_metrics(
                config, equation, newColName, isMixedChannels
            )

        isChannelLess = len(self.channelLessColnames) > 0
        if isChannelLess:
            channelLess_equation = self.equationDisplay.toPlainText()
            equation_name = self.newColNameLineEdit.text()
            config = measurements.add_channelLess_combine_metrics(
                config, channelLess_equation, equation_name,
                self.channelLessColnames
            )

        measurements.save_common_combine_metrics(config)

        self.sigOk.emit(self)
        
        if self.closeOnOk:
            self.close()

    def warnEmptyEquationName(self):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            "New measurement name" field <b>cannot be empty</b>!
        """)
        msg.critical(
            self, 'Empty new measurement name', txt
        )

    def showHelp(self):
        txt = measurements.get_combine_metrics_help_txt()
        msg = widgets.myMessageBox(
            showCentered=False, wrapText=False,
            scrollableText=True, enlargeWidthFactor=1.7
        )
        path = measurements.acdc_metrics_path
        msg.addShowInFileManagerButton(path, txt='Show saved file...')
        msg.information(self, 'Combine measurements help', txt)

    def test_cb(self):
        # Evaluate equation with random inputs
        equation = self.equationDisplay.toPlainText()
        random_data = np.random.rand(1, len(self.equationColNames))*5
        df = pd.DataFrame(
            data=random_data,
            columns=self.equationColNames
        ).round(5)
        newColName = self.newColNameLineEdit.text()
        try:
            df[newColName] = df.eval(equation)
        except Exception as e:
            traceback.print_exc()
            self.testOutputDisplay.setHtml(html_utils.paragraph(e))
            self.testOutputDisplay.setStyleSheet("border: 2px solid red")
            return

        self.testOutputDisplay.setStyleSheet("border: 2px solid green")
        self.okButton.setDisabled(False)

        result = df.round(5).iloc[0][newColName]

        # Substitute numbers into equation
        inputs = df.iloc[0]
        equation_numbers = equation
        for c, col in enumerate(self.equationColNames):
            equation_numbers = equation_numbers.replace(col, str(inputs[c]))

        # Format output into html text
        cols = self.equationColNames
        inputs_txt = [f'{col} = {input}' for col, input in zip(cols, inputs)]
        list_html = html_utils.to_list(inputs_txt)
        text = html_utils.paragraph(f"""
            By substituting the following random inputs:
            {list_html}
            we get the equation:<br><br>
            &nbsp;&nbsp;<code>{newColName} = {equation_numbers}</code><br><br>
            that <b>equals to</b>:<br><br>
            &nbsp;&nbsp;<code>{newColName} = {result}</code>
        """)
        self.testOutputDisplay.setHtml(text)

class stopFrameDialog(QBaseDialog):
    def __init__(self, posDatas, parent=None):
        super().__init__(parent=parent)

        self.cancel = True

        self.setWindowTitle('Stop frame')

        mainLayout = QVBoxLayout()

        infoTxt = html_utils.paragraph(
            'Enter a <b>stop frame number</b> for each of the loaded Positions',
            center=True
        )
        exp_path = posDatas[0].exp_path
        exp_path = os.path.normpath(exp_path).split(os.sep)
        exp_path = f'...{f"{os.sep}".join(exp_path[-4:])}'
        subInfoTxt = html_utils.paragraph(
            f'Experiment folder: <code>{exp_path}<code>', font_size='12px',
            center=True
        )
        infoLabel = QLabel(f'{infoTxt}{subInfoTxt}')
        infoLabel.setToolTip(posDatas[0].exp_path)
        mainLayout.addWidget(infoLabel)
        mainLayout.addSpacing(20)

        self.posDatas = posDatas
        for posData in posDatas:
            _layout = QHBoxLayout()
            _layout.addStretch(1)
            _label = QLabel(html_utils.paragraph(f'{posData.pos_foldername}'))
            _layout.addWidget(_label)

            _spinBox = QSpinBox()
            _spinBox.setMaximum(214748364)
            _spinBox.setAlignment(Qt.AlignCenter)
            _spinBox.setFont(font)
            if posData.acdc_df is not None:
                _val = posData.acdc_df.index.get_level_values(0).max()+1
            else:
                _val = posData.readLastUsedStopFrameNumber()
            if _val is None:
                _val = posData.SizeT
            _spinBox.setValue(_val)

            posData.stopFrameSpinbox = _spinBox

            _layout.addWidget(_spinBox)

            viewButton = widgets.viewPushButton('Visualize...')
            viewButton.clicked.connect(
                partial(self.viewChannelData, posData, _spinBox)
            )
            _layout.addWidget(viewButton, alignment=Qt.AlignRight)

            _layout.addStretch(1)

            mainLayout.addLayout(_layout)

        buttonsLayout = QHBoxLayout()

        okButton = widgets.okPushButton(' Ok ')
        cancelButton = widgets.cancelPushButton(' Cancel ')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        self.setLayout(mainLayout)
    
    def viewChannelData(self, posData, spinBox):
        self.sender().setText('Loading...')
        QTimer.singleShot(
            200, partial(self._viewChannelData, posData, spinBox, self.sender())
        )

    def _viewChannelData(self, posData, spinBox, senderButton):      
        chNames = posData.chNames
        if len(chNames) > 1:
            ch_name_selector = prompts.select_channel_name(
                which_channel='segm', allow_abort=False
            )
            ch_name_selector.QtPrompt(
                self, chNames,'Select channel name to visualize: '
            )
            if ch_name_selector.was_aborted:
                return
            chName = ch_name_selector.channel_name
        else:
            chName = chNames[0]
        
        channel_file_path = load.get_filename_from_channel(
            posData.images_path, chName
        )
        posData.frame_i = 0
        posData.loadImgData(imgPath=channel_file_path)
        self.slideshowWin = imageViewer(
            posData=posData, spinBox=spinBox
        )
        self.slideshowWin.update_img()
        self.slideshowWin.show()
        senderButton.setText('Visualize...')

    def ok_cb(self):
        self.cancel = False
        for posData in self.posDatas:
            stopFrameNum = posData.stopFrameSpinbox.value()
            posData.stopFrameNum = stopFrameNum
        self.close()

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


class CombineMetricsMultiDfsDialog(QBaseDialog):
    sigOk = Signal(object, object)
    sigClose = Signal(bool)

    def __init__(self, acdcDfs, allChNames, parent=None, debug=False):
        super().__init__(parent)

        self.setWindowTitle('Add combined measurement')

        self.initAttributes()

        self.acdcDfs = acdcDfs
        self.cancel = True
        self.isOperatorMode = False

        mainLayout = QVBoxLayout()
        equationLayout = QHBoxLayout()

        treesLayout = QHBoxLayout()
        for i, (acdc_df_endname, acdc_df) in enumerate(acdcDfs.items()):
            metricsTreeWidget = QTreeWidget()
            metricsTreeWidget.setHeaderHidden(True)
            metricsTreeWidget.setFont(font)

            classified_metrics = measurements.classify_acdc_df_colnames(
                acdc_df, allChNames
            )

            for chName in allChNames:
                channelTreeItem = QTreeWidgetItem(metricsTreeWidget)
                channelTreeItem.setText(0, f'{chName} measurements')
                metricsTreeWidget.addTopLevelItem(channelTreeItem)

                standard_metrics = classified_metrics['foregr'][chName]
                bkgr_metrics = classified_metrics['bkgr'][chName]
                custom_metrics = classified_metrics['custom'][chName]

                if standard_metrics:
                    foregrMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
                    foregrMetricsTreeItem.setText(0, 'Cell signal measurements')
                    channelTreeItem.addChild(foregrMetricsTreeItem)
                    self.addTreeItems(
                        foregrMetricsTreeItem, standard_metrics, 
                        isCol=True, index=i
                    )

                if bkgr_metrics:
                    bkgrMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
                    bkgrMetricsTreeItem.setText(0, 'Background values')
                    channelTreeItem.addChild(bkgrMetricsTreeItem)
                    self.addTreeItems(
                        bkgrMetricsTreeItem, bkgr_metrics, 
                        isCol=True, index=i
                    )

                if custom_metrics:
                    customMetricsTreeItem = QTreeWidgetItem(channelTreeItem)
                    customMetricsTreeItem.setText(0, 'Custom measurements')
                    channelTreeItem.addChild(customMetricsTreeItem)
                    self.addTreeItems(
                        customMetricsTreeItem, custom_metrics, 
                        isCol=True, index=i
                    )

            if classified_metrics['size']:
                sizeMetricsTreeItem = QTreeWidgetItem(metricsTreeWidget)
                sizeMetricsTreeItem.setText(0, 'Size measurements')
                metricsTreeWidget.addTopLevelItem(sizeMetricsTreeItem)
                self.addTreeItems(
                    sizeMetricsTreeItem, classified_metrics['size'], 
                    isCol=True, index=i
                )

            if classified_metrics['props']:
                propMetricsTreeItem = QTreeWidgetItem(metricsTreeWidget)
                propMetricsTreeItem.setText(0, 'Region properties')
                metricsTreeWidget.addTopLevelItem(propMetricsTreeItem)
                self.addTreeItems(
                    propMetricsTreeItem, classified_metrics['props'], 
                    isCol=True, index=i
                )

            treeLayout = QVBoxLayout()
            treeTitle = QLabel(html_utils.paragraph(
                f'{i+1}. <code>{acdc_df_endname}</code> measurements  '
            ))
            treeLayout.addWidget(treeTitle)       
            treeLayout.addWidget(metricsTreeWidget)
            treesLayout.addLayout(treeLayout)

            metricsTreeWidget.index = i
            metricsTreeWidget.itemDoubleClicked.connect(self.addColname)

        operatorsLayout = QHBoxLayout()
        operatorsLayout.addStretch(1)

        iconSize = 24

        self.operatorButtons = []
        self.operators = [
            ('add', '+'),
            ('subtract', '-'),
            ('multiply', '*'),
            ('divide', '/'),
            ('open_bracket', '('),
            ('close_bracket', ')'),
            ('square', '**2'),
            ('pow', '**'),
            ('ln', 'log('),
            ('log10', 'log10('),
        ]
        operatorFont = QFont()
        operatorFont.setPixelSize(16)
        for name, text in self.operators:
            button = QPushButton()
            button.setIcon(QIcon(f':{name}.svg'))
            button.setIconSize(QSize(iconSize,iconSize))
            button.text = text
            operatorsLayout.addWidget(button)
            self.operatorButtons.append(button)
            button.clicked.connect(self.addOperator)
            # button.setFont(operatorFont)

        clearButton = QPushButton()
        clearButton.setIcon(QIcon(':clear.svg'))
        clearButton.setIconSize(QSize(iconSize,iconSize))
        clearButton.setFont(operatorFont)

        clearEntryButton = QPushButton()
        clearEntryButton.setIcon(QIcon(':backspace.svg'))
        clearEntryButton.setFont(operatorFont)
        clearEntryButton.setIconSize(QSize(iconSize,iconSize))

        operatorsLayout.addWidget(clearButton)
        operatorsLayout.addWidget(clearEntryButton)
        operatorsLayout.addStretch(1)

        newColNameLayout = QVBoxLayout()
        newColNameLineEdit = widgets.alphaNumericLineEdit()
        newColNameLineEdit.setAlignment(Qt.AlignCenter)
        self.newColNameLineEdit = newColNameLineEdit
        newColNameLayout.addStretch(1)
        newColNameLayout.addWidget(QLabel('New measurement name:'))
        newColNameLayout.addWidget(newColNameLineEdit)
        newColNameLayout.addStretch(1)

        equationDisplayLayout = QVBoxLayout()
        equationDisplayLayout.addWidget(QLabel('Equation:'))
        equationDisplay = QPlainTextEdit()
        # equationDisplay.setReadOnly(True)
        self.equationDisplay = equationDisplay
        equationDisplayLayout.addWidget(equationDisplay)
        equationDisplayLayout.setStretch(0,0)
        equationDisplayLayout.setStretch(1,1)

        equationLayout.addLayout(newColNameLayout)
        equationLayout.addWidget(QLabel(' = '))
        equationLayout.addLayout(equationDisplayLayout)
        equationLayout.setStretch(0,1)
        equationLayout.setStretch(1,0)
        equationLayout.setStretch(2,2)

        instructions = html_utils.paragraph("""
            <b>Double-click</b> on any of the <b>available measurements</b>
            to add it to the equation.<br><br>
            <i>NOTE: the result will be saved in a new <code>acdc_output</code>
            file as a column with the same name<br>
            you enter in "New measurement name"
            field.</i><br>
        """)

        buttonsLayout = QHBoxLayout()

        cancelButton = widgets.cancelPushButton('Cancel')
        testButton = widgets.calcPushButton('Test equation')
        okButton = widgets.okPushButton(' Ok ')
        okButton.setDisabled(True)
        self.okButton = okButton

        if debug:
            debugButton = QPushButton('Debug')
            debugButton.clicked.connect(self._debug)
            buttonsLayout.addWidget(debugButton)

        self.statusLabel = QLabel()
        buttonsLayout.addWidget(self.statusLabel)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(testButton)
        buttonsLayout.addWidget(okButton)

        mainLayout.addWidget(QLabel(instructions))
        mainLayout.addLayout(treesLayout)
        mainLayout.addLayout(operatorsLayout)
        mainLayout.addLayout(equationLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        clearButton.clicked.connect(self.clearEquation)
        clearEntryButton.clicked.connect(self.clearEntryEquation)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        testButton.clicked.connect(self.test_cb)

        self.equationDisplay.textChanged.connect(self.equationChanged)
        # self.newColNameLineEdit.editingFinished.connect(self.equationChanged)

        self.setLayout(mainLayout)
        self.setFont(font)

        self.setStyleSheet(TREEWIDGET_STYLESHEET)
    
    def setLogger(self, logger, logs_path, log_path):
        self.logger = logger
        self.logs_path = logs_path
        self.log_path = log_path
    
    def closeEvent(self, event):
        self.sigClose.emit(self.cancel)
        return super().closeEvent(event)
    
    def getCombinedDf(self):
        dfs = []
        for i, acdc_df in enumerate(self.acdcDfs.values()):
            dfs.append(acdc_df.add_suffix(f'_table{i+1}'))
        return pd.concat(dfs, axis=1)
    
    def _log(self, txt):
        if hasattr(self, 'logger'):
            self.logger.info(txt)
        else:
            print(f'[INFO]: {txt}')
    
    def equationChanged(self):
        self.okButton.setDisabled(True)
        self.statusLabel.setText('')

    @exception_handler
    def test_cb(self):
        combined_df = self.getCombinedDf()
        new_df = pd.DataFrame(index=combined_df.index)
        equation = self.equationDisplay.toPlainText()
        newColName = self.newColNameLineEdit.text()
        new_df[newColName] = combined_df.eval(equation)
        self.okButton.setDisabled(False)
        self._log('Equation test was successful.')
        self.statusLabel.setText(
            'Equation test was successful. You can now click OK.'
        )

    def addOperator(self):
        button = self.sender()
        text = f'{self.equationDisplay.toPlainText()}{button.text}'
        self.equationDisplay.setPlainText(text)
        self.clearLenghts.append(len(button.text))

    def clearEquation(self):
        self.isOperatorMode = False
        self.equationDisplay.setPlainText('')
        self.initAttributes()

    def initAttributes(self):
        self.clearLenghts = []
        self.equationColNames = []
        self.channelLessColnames = []

    def clearEntryEquation(self):
        if not self.clearLenghts:
            return

        text = self.equationDisplay.toPlainText()
        newText = text[:-self.clearLenghts[-1]]
        clearedText = text[-self.clearLenghts[-1]:]
        self.clearLenghts.pop(-1)
        self.equationDisplay.setPlainText(newText)
        if clearedText in self.equationColNames:
            self.equationColNames.remove(clearedText)
        if clearedText in self.channelLessColnames:
            self.channelLessColnames.remove(clearedText)

    def addTreeItems(
            self, parentItem, itemsText, isCol=False, isChannelLess=False,
            index=None
        ):
        for text in itemsText:
            _item = QTreeWidgetItem(parentItem)
            _item.setText(0, text)
            parentItem.addChild(_item)
            if isCol:
                _item.isCol = True
            if index is not None:
                _item.index = index
            _item.isChannelLess = isChannelLess

    def addColname(self, item, column):
        if not hasattr(item, 'isCol'):
            return

        colName = f'{item.text(0)}_table{item.index+1}'
        text = f'{self.equationDisplay.toPlainText()}{colName}'

        self.equationDisplay.setPlainText(text)
        self.clearLenghts.append(len(colName))
        self.equationColNames.append(colName)
        if item.isChannelLess:
            self.channelLessColnames.append(colName)

    def _debug(self):
        print(self.getEquationsDict())

    def ok_cb(self):
        if not self.newColNameLineEdit.text():
            self.warnEmptyEquationName()
            return
        if not self.equationDisplay.toPlainText():
            self.warnEmptyEquation()
            return

        self.expression = self.equationDisplay.toPlainText()
        self.newColname = self.newColNameLineEdit.text()
        self.cancel = False
        self.sigOk.emit(self.newColname, self.expression)
        self.close()
    
    def warnEmptyEquation(self):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            "Equation" field <b>cannot be empty</b>!
        """)
        msg.critical(
            self, 'Empty equation', txt
        )

    def warnEmptyEquationName(self):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            "New measurement name" field <b>cannot be empty</b>!
        """)
        msg.critical(
            self, 'Empty new measurement name', txt
        )

class CombineMetricsMultiDfsSummaryDialog(QBaseDialog):
    sigLoadAdditionalAcdcDf = Signal()

    def __init__(
            self, acdcDfs, allChNames, parent=None, debug=False
        ):
        super().__init__(parent)

        self.editedIndex = None
        self.cancel = True
        self.acdcDfs = acdcDfs
        self.allChNames = allChNames

        self.setWindowTitle('Combine measurements summary')
        
        mainLayout = QVBoxLayout()
        viewLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        row = 0
        txt = html_utils.paragraph('Selected acdc_output tables:')
        viewLayout.addWidget(QLabel(txt), row, 0)

        row += 1
        items = [
            f'• <b>Table {i+1}</b>: <code>{e}</code>' 
            for i, e in enumerate(acdcDfs.keys())
        ]
        selectedAcdcDfsList = widgets.readOnlyQList()
        selectedAcdcDfsList.addItems(items)
        self.selectedAcdcDfsList = selectedAcdcDfsList

        tablesButtonsLayout = QVBoxLayout()
        loadAcdcDfButton = widgets.showInFileManagerButton('Load additional tables')
        tablesButtonsLayout.addWidget(loadAcdcDfButton)
        
        loadEquationsButton = widgets.reloadPushButton(
            'Load previously used equations'
        )
        tablesButtonsLayout.addWidget(loadEquationsButton)
        

        tablesButtonsLayout.addStretch(1)

        viewLayout.addWidget(selectedAcdcDfsList, row, 0)
        viewLayout.addLayout(tablesButtonsLayout, row, 1)
        viewLayout.setRowStretch(row, 1)

        row += 1
        txt = html_utils.paragraph('Equations:')
        viewLayout.addWidget(QLabel(txt), row, 0)

        row += 1
        self.equationsList = widgets.TreeWidget()
        self.equationsList.setFont(font)
        self.equationsList.setHeaderLabels(['Metric', 'Expression'])
        self.equationsList.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection)

        equationsButtonsLayout = QVBoxLayout()
        addEquationButton = widgets.addPushButton('Add metric')
        removeEquationButton = widgets.subtractPushButton('Remove metric(s)')
        editEquationButton = widgets.editPushButton('Edit metric')
        removeEquationButton.setDisabled(True)
        editEquationButton.setDisabled(True)
        self.removeEquationButton = removeEquationButton
        self.editEquationButton = editEquationButton

        equationsButtonsLayout.addWidget(addEquationButton)
        equationsButtonsLayout.addWidget(removeEquationButton)
        equationsButtonsLayout.addWidget(editEquationButton)
        equationsButtonsLayout.addStretch(1)
        
        viewLayout.addWidget(self.equationsList, row, 0)
        viewLayout.addLayout(equationsButtonsLayout, row, 1)
        viewLayout.setRowStretch(row, 2)

        cancelButton = widgets.cancelPushButton('Cancel')
        okButton = widgets.okPushButton('Ok')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        viewLayout.setVerticalSpacing(10)
        mainLayout.addLayout(viewLayout)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        addEquationButton.clicked.connect(self.addEquation_cb)
        loadAcdcDfButton.clicked.connect(self.loadButtonClicked)
        loadEquationsButton.clicked.connect(self.loadEquationsButtonClicked)
        removeEquationButton.clicked.connect(self.removeButtonClicked)
        editEquationButton.clicked.connect(self.editButtonClicked)
        self.equationsList.itemSelectionChanged.connect(
            self.onEquationItemSelectionChanged
        )

        self.setLayout(mainLayout)
    
    def setLogger(self, logger, logs_path, log_path):
        self.logger = logger
        self.logs_path = logs_path
        self.log_path = log_path
    
    def loadEquationsButtonClicked(self):
        MostRecentPath = myutils.getMostRecentPath()
        file_path = QFileDialog.getOpenFileName(
            self, 'Select equations file', MostRecentPath, "Config Files (*.ini)"
            ";;All Files (*)")[0]
        if file_path == '':
            return

        cp = config.ConfigParser()
        cp.read(file_path)
        sectionToMatch = [
            f'table{i+1}:{end}' for i, end in enumerate(self.acdcDfs)
        ]
        sectionToMatch = ';'.join(sectionToMatch)
        
        lists = {}
        nonMatchingLists = {}
        groupsDescr = {}
        
        for section in cp.sections():
            # Tag acdc_output names with <code> html and table(\d+) with html bold tag
            listName = ';'.join([
                re.sub(r'table(\d+):(.*)', r'<b>table\g<1></b>: <code>\g<2></code>', s) 
                for s in section.split(';')
            ])
            listName = listName.replace(';', ' ; ')
            children = [f'{opt} = {cp[section][opt]}' for opt in cp[section]]
            if section == sectionToMatch:
                groupsDescr[listName] = (
                    'Equations that were calculated from the <b>same '
                    'table names</b> you loaded'
                )
                lists[listName] = children
            else:
                groupsDescr[listName] = (
                    'Equations that were calculated from <b>table names that '
                    'you did not load</b> now'
                )
                nonMatchingLists[listName] = children
                # # Not implemented yet --> selecting from non matching table names
                # # would require an additional widget where the user sets 
                # # what df1 and df2 are.
                # trees[treeName] = children

        if not lists:
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            txt = html_utils.paragraph("""
                <b>None of the equations</b> in the selected file used the <b>same 
                table names</b> that you loaded.<br><br>
                See below which table names and equations are present in the loaded file.
            """)
            with open(file_path) as iniFile:
                detailedText = iniFile.read()
            
            msg.warning(self, 'Not the same tables', txt, showDialog=False)
            msg.setDetailedText(detailedText, visible=True)
            msg.addShowInFileManagerButton(os.path.dirname(file_path))
            msg.exec_()
            return

        selectWindow = MultiListSelector(
            lists, groupsDescr=groupsDescr, title='Select equations to load',
            infoTxt='Select equations you want to load'
        )
        selectWindow.exec_()
        if selectWindow.cancel or not selectWindow.selectedItems:
            return

        for listName, equations in selectWindow.selectedItems.items():
            for equation in equations:
                metricName, expression = equation.split(' = ')
                self.addEquation(metricName, expression)
        

    def ok_cb(self):
        self.cancel = False
        self.equations = {}
        for i in range(self.equationsList.topLevelItemCount()):
            item = self.equationsList.topLevelItem(i)
            self.equations[item.text(0)] = item.text(1)

        self.close()
    
    def loadButtonClicked(self):
        self.sigLoadAdditionalAcdcDf.emit()
    
    def removeButtonClicked(self):
        for item in self.equationsList.selectedItems():
            self.equationsList.invisibleRootItem().removeChild(item)
        
    def editButtonClicked(self):
        self.editedItem = self.equationsList.selectedItems()[0]
        self.editedIndex = self.equationsList.indexOfTopLevelItem(self.editedItem)
        self.addEquation_cb()
    
    def onEquationItemSelectionChanged(self):
        selectedItems = self.equationsList.selectedItems()
        if len(selectedItems) == 1:
            self.editEquationButton.setDisabled(False)
            self.removeEquationButton.setDisabled(False)
        elif len(selectedItems) > 1:
            self.removeEquationButton.setDisabled(False)
            self.editEquationButton.setDisabled(True)
        else:
            self.removeEquationButton.setDisabled(True)
            self.editEquationButton.setDisabled(True)
    
    def addAcdcDfs(self, acdcDfsDict):
        self.acdcDfs = {**self.acdcDfs, **acdcDfsDict}
        items = [
            f'• <b>Table {i+1}</b>: <code>{e}</code>' 
            for i, e in enumerate(self.acdcDfs.keys())
        ]
        self.selectedAcdcDfsList = widgets.readOnlyQList()
        self.selectedAcdcDfsList.addItems(items)

    def addEquation(self, newColname, expression):
        if self.editedIndex is not None:
            self.equationsList.invisibleRootItem().removeChild(self.editedItem)
        bkgrColor = QColor(*BACKGROUND_RGBA[:3], 200)
        item = widgets.TreeWidgetItem(
            self.equationsList, columnColors=[None, bkgrColor]
        )
        item.setText(0, newColname)
        item.setText(1, expression)
        if self.editedIndex is not None:
            self.equationsList.insertTopLevelItem(self.editedIndex, item)
        else:
            self.equationsList.addTopLevelItem(item)
        self.equationsList.resizeColumnToContents(0)
        self.equationsList.resizeColumnToContents(1)
        self.editedIndex = None
    
    def addEquation_cb(self):
        self.addEquationWin = CombineMetricsMultiDfsDialog(
            self.acdcDfs, self.allChNames, parent=self
        )
        if hasattr(self, 'logger'):
            self.addEquationWin.setLogger(
                self.logger, self.logs_path, self.log_path
            )
        if self.editedIndex is not None:
            editedMetricName = self.editedItem.text(0)
            self.addEquationWin.newColNameLineEdit.setText(editedMetricName)
            editedExpression = self.editedItem.text(1)
            self.addEquationWin.equationDisplay.setPlainText(editedExpression)
        self.addEquationWin.show()
        self.addEquationWin.sigOk.connect(self.addEquation)
        self.addEquationWin.sigClose.connect(self.addEquationClosed)
    
    def addEquationClosed(self, cancelled):
        if cancelled:
            self.editedIndex = None
    
    def showEvent(self, event) -> None:
        self.resize(int(self.width()*2), self.height())

class ShortcutEditorDialog(QBaseDialog):
    def __init__(
            self, widgetsWithShortcut: dict, 
            delObjectKey='',
            delObjectButton: Literal['Middle click', 'Left click']='Middle click',
            parent=None
        ):
        self.cancel = True
        super().__init__(parent)

        self.setWindowTitle('Customize keyboard shortcuts')

        mainLayout = QVBoxLayout()

        self.customShortcuts = {}
        self.shortcutLineEdits = {}

        scrollArea = QScrollArea(self)
        scrollArea.setWidgetResizable(True)
        scrollAreaWidget = QWidget()
        entriesLayout = QGridLayout()
        
        self.delObjShortcutLineEdit = widgets.ShortcutLineEdit(
            allowModifiers=True, notAllowedModifier=Qt.AltModifier
        )
        if delObjectKey is not None:
            self.delObjShortcutLineEdit.setText(delObjectKey)
        self.delObjButtonCombobox = QComboBox()
        self.delObjButtonCombobox.addItems(['Middle click', 'Left click'])
        self.delObjButtonCombobox.setCurrentText(delObjectButton)
        entriesLayout.addWidget(QLabel('Delete object:'), 0, 0)
        entriesLayout.addWidget(self.delObjShortcutLineEdit, 0, 1)
        entriesLayout.addWidget(
            self.delObjButtonCombobox, 0, 2, alignment=Qt.AlignLeft
        )
        
        for row, (name, widget) in enumerate(widgetsWithShortcut.items(), start=1):
            label = QLabel(f'{name}:')
            shortcutLineEdit = widgets.ShortcutLineEdit()
            if hasattr(widget, 'keyPressShortcut'):
                shortcutLineEdit.key = widget.keyPressShortcut
                shortcut = QKeySequence(widget.keyPressShortcut)
                isShortcutKeyPress = True
            else:
                shortcut = widget.shortcut()
                isShortcutKeyPress = False
            shortcutLineEdit.setText(shortcut.toString())
            shortcutLineEdit.textChanged.connect(self.checkDuplicateShortcuts)
            shortcutLineEdit.isShortcutKeyPress = isShortcutKeyPress
            entriesLayout.addWidget(label, row, 0)
            entriesLayout.addWidget(shortcutLineEdit, row, 1)
            self.shortcutLineEdits[name] = shortcutLineEdit
        
        entriesLayout.setColumnStretch(0, 0)
        entriesLayout.setColumnStretch(1, 1)
        entriesLayout.setColumnStretch(2, 0)
        
        scrollAreaWidget.setLayout(entriesLayout)
        scrollArea.setWidget(scrollAreaWidget)
        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addWidget(scrollArea)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setFont(font)
        self.setLayout(mainLayout)
    
    def checkDuplicateShortcuts(self, text):
        for name, shortcutLineEdit in self.shortcutLineEdits.items():
            if shortcutLineEdit == self.sender():
                continue
            if shortcutLineEdit.text() != text:
                continue
            shortcutLineEdit.setText('')
    
    def ok_cb(self):
        self.cancel = False
        for name, shortcutLineEdit in self.shortcutLineEdits.items():
            text = shortcutLineEdit.text()
            if shortcutLineEdit.isShortcutKeyPress:
                self.customShortcuts[name] = (text, shortcutLineEdit.key)
            else:
                self.customShortcuts[name] = (
                    text, shortcutLineEdit.keySequence
                )
        delObjButtonText = self.delObjButtonCombobox.currentText()
        delObjQtButton = (
            Qt.MouseButton.LeftButton if delObjButtonText == 'Left click'
            else Qt.MouseButton.MiddleButton
        )
        self.delObjAction = (
            self.delObjShortcutLineEdit.keySequence, delObjQtButton
        )
        self.close()
    
    def showEvent(self, event) -> None:
        self.resize(int(self.width()*1.2), self.height())
        self.move(self.x(), 100)

class SelectAcdcDfVersionToRestore(QBaseDialog):
    def __init__(self, posData, parent=None):
        super().__init__(parent=parent)
        
        self.cancel = True
        
        self.setWindowTitle('Select annotations table to restore')
        
        mainLayout = QVBoxLayout()
        
        acdc_df_filename = os.path.basename(posData.acdc_output_csv_path)
        instructionsLabel = html_utils.paragraph(
            f'Select an <b>older version</b> of the <code>{acdc_df_filename}</code> '
            'annotations table to load.<br><br>'
            'The datetime refers to the time you replaced the old version with '
            'a newer one.<br><br>'
        )
        mainLayout.addWidget(QLabel(instructionsLabel))
        
        self.savedListBox = None
        if os.path.exists(posData.acdc_output_backup_zip_path):
            zip_path = posData.acdc_output_backup_zip_path
            self.savedArchivefilepath = zip_path
            with zipfile.ZipFile(zip_path, mode='r') as zip:
                csv_names = natsorted(zip.namelist(), reverse=True)
            
            keys = [csv_name[:-4] for csv_name in csv_names]

            self.savedKeys = keys
            f = load.ISO_TIMESTAMP_FORMAT
            timestamps = [datetime.datetime.strptime(key, f) for key in keys]
            items = [date.strftime(r'%d %b %Y, %H:%M:%S') for date in timestamps]
            mainLayout.addWidget(QLabel('Saved annotations:'))
            self.savedListBox = widgets.listWidget()
            self.savedListBox.addItems(items)
            mainLayout.addWidget(self.savedListBox)
            self.savedListBox.itemSelectionChanged.connect(
                self.onItemSelectionChanged
            )
        
        recovery_folderpath = posData.recoveryFolderpath()
        unsaved_recovery_folderpath = os.path.join(
            recovery_folderpath, 'never_saved'
        )
        self.neverSavedFolderpath = unsaved_recovery_folderpath
        files = myutils.listdir(unsaved_recovery_folderpath)
        csv_files = [file for file in files if file.endswith('.csv')]
        self.neverSavedListBox = None
        if csv_files:
            csv_names = natsorted(csv_files)
            keys = [csv_name[:-4] for csv_name in csv_names]
            self.neverSavedKeys = keys
            f = load.ISO_TIMESTAMP_FORMAT
            timestamps = [
                datetime.datetime.strptime(key, f) for key in keys
            ]
            items = [date.strftime(r'%d %b %Y, %H:%M:%S') for date in timestamps]
            mainLayout.addWidget(QLabel('Never saved annotations:'))
            self.neverSavedListBox = widgets.listWidget()
            self.neverSavedListBox.addItems(items)
            mainLayout.addWidget(self.neverSavedListBox)
            self.neverSavedListBox.itemSelectionChanged.connect(
                self.onItemSelectionChanged
            )
                
        cancelOkLayout = widgets.CancelOkButtonsLayout()
        
        cancelOkLayout.okButton.clicked.connect(self.ok_cb)
        cancelOkLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(cancelOkLayout)
        
        self.setLayout(mainLayout)
        
        self.setFont(font)
    
    def ok_cb(self):
        self.cancel = False
        try:
            for i in range(self.savedListBox.count()):
                item = self.savedListBox.item(i)
                if item.isSelected():
                    self.selectedTimestamp = item.text()
                    self.selectedKey = self.savedKeys[i]
                    self.archiveFilePath = self.savedArchivefilepath
                    break
        except Exception as e:
            pass
        
        try:
            for i in range(self.neverSavedListBox.count()):
                item = self.neverSavedListBox.item(i)
                if item.isSelected():
                    self.selectedTimestamp = item.text()
                    self.selectedKey = self.neverSavedKeys[i]
                    self.archiveFilePath = self.neverSavedFolderpath
                    break
        except Exception as e:
            pass
        self.close()
    
    def onItemSelectionChanged(self):
        otherListBox = (
            self.savedListBox if self.sender() == self.neverSavedListBox
            else self.neverSavedListBox
        )
        if otherListBox is None:
            return
        for i in range(otherListBox.count()):
            item = otherListBox.item(i)
            item.setSelected(False)

class ChangeUserProfileFolderPathDialog(QBaseDialog):
    def __init__(self, posData, parent=None):
        super().__init__(parent=parent)
        
        self.cancel = True
        
        self.setWindowTitle('Change user profile folder path')
        
        mainLayout = QVBoxLayout()
        
        acdc_folders = load.get_all_acdc_folders(user_profile_path)
        acdc_folders_format = [f'  - {folder}' for folder in acdc_folders]
        acdc_folders_format = '<br>'.join(acdc_folders_format)
        
        txt = (f"""
            Current user profile path:<br><br>
            <code>{user_profile_path}</code><br><br>
            The user profile contains the following Cell-ACDC folders:<br><br>
            {acdc_folders_format}<br><br>
            After clicking "Ok" you will be <b>asked to select the folder</b> where 
            you want to <b>migrate</b> the user profile data. 
        """)
        
        txt = html_utils.paragraph(txt)
        label = QLabel(txt)
        
        mainLayout.addWidget(label)
        
        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch()
        
        self.setLayout(mainLayout)
    
    def ok_cb(self):
        self.cancel = False
        self.close()
 
class SelectFeaturesRange:
    def __init__(
            self, 
            posData, 
            force_postprocess_2D=False, 
            qparent=None,
            sigValueChanged=None
        ) -> None:
        self.posData = posData
        self.qparent = qparent
        self.force_postprocess_2D = force_postprocess_2D
        self.sigValueChanged = sigValueChanged
        
        self.lowRangeWidgets = widgets.CheckableSpinBoxWidgets()
        self.highRangeWidgets = widgets.CheckableSpinBoxWidgets()        
        
        self.selectButton = widgets.FeatureSelectorButton(
            'Click to select feature...'
        )
        self.selectButton.setSizeLongestText(
            'Spotfit intens. metric, Foregr. integral gauss. peak'
        )
        self.selectButton.clicked.connect(self.selectFeature)
        self.selectButton.setCursor(Qt.PointingHandCursor)

        self.selectedFeatureGroups = {}

        self.widgets = [
            {'pos': (0, 0), 'widget': self.lowRangeWidgets.checkbox}, 
            {'pos': (1, 0), 'widget': self.lowRangeWidgets.spinbox}, 
            {'pos': (1, 1), 'widget': widgets.LessThanPushButton(flat=True)},
            {'pos': (1, 2), 'widget': self.selectButton},
            {'pos': (1, 3), 'widget': widgets.LessThanPushButton(flat=True)},
            {'pos': (0, 4), 'widget': self.highRangeWidgets.checkbox},
            {'pos': (1, 4), 'widget': self.highRangeWidgets.spinbox}, 
            {'pos': (2, 0), 'widget': widgets.VerticalSpacerEmptyWidget(height=10)}
        ]
        self.columnsStretches = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}
    
    def setText(self, text):
        self.selectButton.setText(text)
    
    def selectFeature(self):
        loadedChNames = [self.posData.user_ch_name]
        notLoadedChNames = []
        isZstack = self.posData.SizeZ > 1 and not self.force_postprocess_2D
        isSegm3D = self.posData.isSegm3D and not self.force_postprocess_2D
        self.selectFeatureDialog = SetMeasurementsDialog(
            loadedChNames, notLoadedChNames, isZstack, isSegm3D,
            posData=self.posData, parent=self.qparent,
            isSingleSelection=True, is_concat=True
        )
        # self.selectFeatureDialog.resizeVertical()
        self.selectFeatureDialog.sigClosed.connect(self.setFeatureText)
        self.selectFeatureDialog.show()
    
    def setFeatureText(self):
        if self.selectFeatureDialog.cancel:
            return
        self.selectButton.setFlat(True)
        selectedMetricName, selectedMetricGroup = (
            self.selectFeatureDialog.selectedMetricNameAndGroup()
        )
        self.selectButton.setText(selectedMetricName)
        self.featureGroup = selectedMetricGroup

class SelectFeaturesRangeDialog(QBaseDialog):
    sigValueChanged = Signal(object)
    
    def __init__(self, posData=None, parent=None, force_postprocess_2D=False):
        super().__init__(parent)
        
        self.force_postprocess_2D = force_postprocess_2D
        
        layout = QVBoxLayout()
        self.setWindowTitle('Custom features for post-processing')
        
        self.groupbox = SelectFeaturesRangeGroupbox(
            posData=posData, parent=parent, 
            force_postprocess_2D=force_postprocess_2D
        )
        
        buttonsLayout = QHBoxLayout()
        okPushButton = widgets.okPushButton(' Ok ')
        
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(okPushButton)
        
        okPushButton.clicked.connect(self.ok_cb)

        layout.addWidget(self.groupbox)
        layout.addSpacing(10)
        layout.addLayout(buttonsLayout)
        
        self.setLayout(layout)
    
    def ok_cb(self):
        if self.groupbox.selectedFeaturesRange():
            self.sigValueChanged.emit(None)
        self.hide()

class SelectFeaturesRangeGroupbox(QGroupBox):
    def __init__(
            self, posData=None, parent=None, force_postprocess_2D=False
        ):
        super().__init__(parent)

        self.setTitle('Features and thresholds for filtering segmented objects')
        # self.setCheckable(True)

        self.posData = posData
        self.force_postprocess_2D = force_postprocess_2D
        
        self._layout = QGridLayout()
        self._layout.setVerticalSpacing(0)

        firstSelector = SelectFeaturesRange(
            posData, force_postprocess_2D=force_postprocess_2D
        )
        self.addButton = widgets.addPushButton('  Add feature    ')
        self.addButton.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        for col, widget in enumerate(firstSelector.widgets):
            row, col = widget['pos']
            self._layout.addWidget(widget['widget'], row, col)
        for col, stretch in firstSelector.columnsStretches.items():
            self._layout.setColumnStretch(col, stretch)
            
        lastCol = self._layout.columnCount()
        self._layout.addWidget(self.addButton, 0, lastCol+1, 2, 1)
        self.lastCol = lastCol+1
        self.selectors = [firstSelector]

        self.setLayout(self._layout)

        # self.setFont(font)

        self.addButton.clicked.connect(self.addFeatureField)

    def addFeatureField(self):
        row = self._layout.rowCount()
        selector = SelectFeaturesRange(
            self.posData, force_postprocess_2D=self.force_postprocess_2D
        )
        delButton = widgets.delPushButton('Remove feature')
        delButton.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        delButton.selector = selector
        for col, widget in enumerate(selector.widgets):
            relRow, col = widget['pos']
            self._layout.addWidget(widget['widget'], relRow+row, col)
        self._layout.addWidget(delButton, row, self.lastCol, 2, 1)
        self.selectors.append(selector)
        delButton.clicked.connect(self.removeFeatureField)
    
    def removeFeatureField(self):
        delButton = self.sender()
        for widget in delButton.selector.widgets:
            self._layout.removeWidget(widget['widget'])
        self._layout.removeWidget(delButton)
        self.selectors.remove(delButton.selector)
    
    def selectedFeaturesRange(self):
        featuresRange = {}
        for selector in self.selectors:
            if selector.selectButton.text().find('Click') != -1:
                continue
            featuresRange[selector.selectButton.text()] = (
                selector.lowRangeWidgets.value(), 
                selector.highRangeWidgets.value()
            )
        return featuresRange

    def selectedFeaturesGroup(self):
        featuresGroup = {}
        for selector in self.selectors:
            if selector.selectButton.text().find('Click') != -1:
                continue
            group = selector.featureGroup                
            featuresGroup[selector.selectButton.text()] = group
        return featuresGroup

    def groupedFeatures(self):
        featuresGroup = self.selectedFeaturesGroup()
        groupedFeatures = {}
        for feature, group in featuresGroup.items():
            group = featuresGroup[feature]
            if isinstance(group, str):
                key = group
                if key not in groupedFeatures:
                    groupedFeatures[key] = []
                groupedFeatures[key].append(feature)
            else:
                key, channel = list(group.items())[0]
                if key not in groupedFeatures:
                    groupedFeatures[key] = {}
                if channel not in groupedFeatures[key]:
                    groupedFeatures[key][channel] = []
                groupedFeatures[key][channel].append(feature)
        return groupedFeatures
    
    def setValue(self, value):
        pass

def get_existing_directory(allow_images_path=True, **kwargs):
    while True:
        folder_path = qtpy.compat.getexistingdirectory(**kwargs)
        if not folder_path:
            return
        
        if allow_images_path:
            return folder_path
        
        pos_folderpath = os.path.dirname(folder_path)
        is_images_folder = (
            folder_path.endswith('Images') 
            and os.path.basename(pos_folderpath).startswith('Position_')
            and os.path.isdir(folder_path)
        )
        if not is_images_folder:
            return folder_path
        
        txt = html_utils.paragraph(
            'You <b>cannot save</b> to the <code>Images</code> folder '
            'because it is reserved to files that start with the same '
            'basename.<br><br>Thank you for your patience!'
        )
        msg = widgets.myMessageBox()
        msg.warning(kwargs['parent'], 'Cannot save here', txt)

class ScaleBarPropertiesDialog(QBaseDialog):
    sigValueChanged = Signal(object)
    
    def __init__(
            self, maxLength, maxThickness, PhysicalSizeX, parent=None, 
            **properties
        ):
        super().__init__(parent=parent)
        
        self.cancel = True
        self.setWindowTitle('Scale bar properties')
        
        self.PhysicalSizeX = PhysicalSizeX
        
        mainLayout = QVBoxLayout()
        
        formLayout = widgets.FormLayout()
        formLayout.setVerticalSpacing(10)
        formLayout.setHorizontalSpacing(50)
        
        row = 0
        unitCombobox = QComboBox()
        unitFormWidget = widgets.formWidget(
            unitCombobox, labelTextLeft='Physical unit'
        )
        unitCombobox.addItems(
            ['nm', 'μm', 'mm', 'cm']
        )
        if properties.get('unit') is None:
            unitCombobox.setCurrentIndex(1)
        else:
            unitCombobox.setCurrentText(properties.get('unit'))
        formLayout.addFormWidget(
            unitFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )       
        self.unitCombobox = unitCombobox
        
        row += 1
        lengthDoubleSpinbox = widgets.DoubleSpinBox()
        lengthDoubleSpinbox.setMaximum(maxLength)
        lengthDoubleSpinbox.setMinimum(PhysicalSizeX)
        lengthDoubleSpinbox.setDecimals(1)
        if properties.get('length_unit') is not None:
            lengthDoubleSpinbox.setValue(properties.get('length_unit'))
        else:
            deafultLength = np.ceil(PhysicalSizeX*15)
            lengthDoubleSpinbox.setValue(round(deafultLength))
        lengthFormWidget = widgets.formWidget(
            lengthDoubleSpinbox, labelTextLeft='Length (μm)'
        )
        self.lengthFormWidget = lengthFormWidget
        self.lengthDoubleSpinbox = lengthDoubleSpinbox
        formLayout.addFormWidget(
            lengthFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )       
        
        row += 1
        thicknessSpinbox = widgets.DoubleSpinBox()
        thicknessSpinbox.setMaximum(maxThickness)
        thicknessSpinbox.setMinimum(1)
        if properties.get('thickness') is not None:
            thicknessSpinbox.setValue(properties.get('thickness'))
        else:
            thicknessSpinbox.setValue(round(4, 1))
        thicknessSpinbox.setDecimals(1)
        thicknessFormWidget = widgets.formWidget(
            thicknessSpinbox, labelTextLeft='Thickness (pixel)'
        )
        formLayout.addFormWidget(
            thicknessFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )       
        self.thicknessSpinbox = thicknessSpinbox
        
        row += 1
        locCombobox = QComboBox()
        locFormWidget = widgets.formWidget(
            locCombobox, labelTextLeft='Location'
        )
        locCombobox.addItems(
            ['Bottom-right', 'Bottom-left', 'Top-left', 'Top-right', 'Custom']
        )
        loc = properties.get('loc')
        if isinstance(loc, str):
            locCombobox.setCurrentText(loc)
        formLayout.addFormWidget(
            locFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )       
        self.locCombobox = locCombobox
        
        row += 1
        self.colorButton = widgets.myColorButton(color=(255, 255, 255))
        if properties.get('color') is not None:
            self.colorButton.setColor(properties.get('color'))
        colorFormWidget = widgets.formWidget(
            self.colorButton, labelTextLeft='Color',
            widgetAlignment=Qt.AlignCenter, stretchWidget=False
        )
        formLayout.addFormWidget(
            colorFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )  
        
        row += 1
        displayTextToggle = widgets.Toggle()
        if properties.get('is_text_visible') is not None:
            displayTextToggle.setChecked(properties.get('is_text_visible'))
        else:
            displayTextToggle.setChecked(True)
        displayTextFormWidget = widgets.formWidget(
            displayTextToggle, labelTextLeft='Display text',
            widgetAlignment=Qt.AlignCenter, stretchWidget=False
        )
        formLayout.addFormWidget(
            displayTextFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )       
        self.displayTextToggle = displayTextToggle
        
        row += 1
        fontSizeSpinbox = widgets.SpinBox()
        if properties.get('font_size') is not None:
            fontSizeSpinbox.setValue(int(properties.get('font_size')))
        else:
            fontSizeSpinbox.setValue(12)
        fontSizeFormWidget = widgets.formWidget(
            fontSizeSpinbox, labelTextLeft='Font size (px)'
        )
        self.fontSizeSpinbox = fontSizeSpinbox
        formLayout.addFormWidget(
            fontSizeFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        ) 
        
        row += 1
        decimalsSpinbox = widgets.SpinBox()
        decimalsSpinbox.setMaximum(6)
        decimalsSpinbox.setMinimum(0)
        if properties.get('num_decimals') is not None:
            decimalsSpinbox.setValue(properties.get('num_decimals'))
        else:
            decimalsSpinbox.setValue(0)
        decimalsFormWidget = widgets.formWidget(
            decimalsSpinbox, labelTextLeft='Number of decimals'
        )
        formLayout.addFormWidget(
            decimalsFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )       
        self.decimalsSpinbox = decimalsSpinbox
        
        mainLayout.addLayout(formLayout)
        
        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch()
        
        self.setLayout(mainLayout)
        self.setFont(font)
        
        self.unitCombobox.currentTextChanged.connect(self.updateLengthUnit)
        self.colorButton.clicked.disconnect()
        self.colorButton.clicked.connect(self.selectColor)
        
        self.colorButton.sigColorChanging.connect(self.onValueChanged)
        self.lengthDoubleSpinbox.valueChanged.connect(self.onValueChanged)
        self.thicknessSpinbox.valueChanged.connect(self.onValueChanged)
        self.locCombobox.currentTextChanged.connect(self.onValueChanged)
        self.displayTextToggle.toggled.connect(self.onValueChanged)
        self.fontSizeSpinbox.valueChanged.connect(self.onValueChanged)
        self.decimalsSpinbox.valueChanged.connect(self.onValueChanged)
    
    def onValueChanged(self, object):
        self.sigValueChanged.emit(self.kwargs())
    
    def selectColor(self):
        color = self.colorButton.color()
        self.colorButton.origColor = color
        self.colorButton.colorDialog.setCurrentColor(color)
        self.colorButton.colorDialog.setWindowFlags(
            Qt.Window | Qt.WindowStaysOnTopHint
        )
        self.colorButton.colorDialog.setParent(self)
        self.colorButton.colorDialog.open()
        w = self.width()
        left = self.pos().x()
        colorDialogTop = self.colorButton.colorDialog.pos().y()
        self.colorButton.colorDialog.move(w+left+10, colorDialogTop)
    
    def updateLengthUnit(self, unit):
        newText = re.sub(
            r'\(.*\)', f'({unit})', 
            self.lengthFormWidget.labelLeft.text()
        )
        self.lengthFormWidget.labelLeft.setText(newText)
        self.onValueChanged(self)
    
    def kwargs(self):
        unit = self.unitCombobox.currentText()
        length_unit = self.lengthDoubleSpinbox.value()
        length_um = _core.convert_length(length_unit, unit, 'μm')
        length_pixel = length_um/self.PhysicalSizeX
        kwargs = {
            'thickness': self.thicknessSpinbox.value(),
            'length_pixel': length_pixel,
            'length_unit': length_unit,
            'is_text_visible': self.displayTextToggle.isChecked(),
            'color': self.colorButton.color(),
            'loc': self.locCombobox.currentText().lower(),
            'font_size': self.fontSizeSpinbox.value(),
            'unit': unit,
            'num_decimals': self.decimalsSpinbox.value()
        }
        return kwargs
    
    def ok_cb(self):
        self.cancel = False
        self.close()

class SetColumnNamesDialog(QBaseDialog):
    def __init__(
            self, columnNames, categories, 
            optionalCategories=None, parent=None
        ):
        super().__init__(parent)
        
        if not optionalCategories:
            optionalCategories = None
        
        self.cancel = True
        
        mainLayout = QVBoxLayout()
        
        mainLayout.addWidget(QLabel(html_utils.paragraph(
            'Assign a column to the following categories:<br>'
        )))
        
        self.categoriesWidgets = {}
        formLayout = QFormLayout()
        for row, category in enumerate(categories):
            combobox = widgets.ComboBox()
            combobox.addItems(columnNames)
            if optionalCategories is not None:
                text = f'* {category}'
            else:
                text = category
            formLayout.addRow(text, combobox)
            self.categoriesWidgets[category] = combobox
        
        if optionalCategories is not None:
            optionalItems = ['None', *columnNames]
            for row, category in enumerate(optionalCategories):
                combobox = widgets.ComboBox()
                combobox.addItems(optionalItems)
                formLayout.addRow(category, combobox)
                self.categoriesWidgets[category] = combobox
        
        mainLayout.addLayout(formLayout)
        if optionalCategories is not None:
            mainLayout.addSpacing(10)
            mainLayout.addWidget(QLabel(html_utils.paragraph(
                '* mandatory', font_size='11px'
            )))
        
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
        msg.warning(self, 'Non-unique columns', txt)
    
    def _checkUniqueNames(self):
        self.textToCategoryMapper = {}
        for category, combobox in self.categoriesWidgets.items():
            if combobox.text() == 'None':
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
            category:combobox.text() 
            for category, combobox in self.categoriesWidgets.items() 
        }
        self.cancel = False
        self.close()

class CombineFeaturesCalculator(QBaseDialog):
    sigOk = Signal(object)
    
    def __init__(
            self, features_groups: dict, 
            group_name_to_col_mapper: dict=None,
            title='Combine features calculator',
            parent=None
        ):
        super().__init__(parent)
        
        self.cancel = True
        
        self.setWindowTitle(title)
        self.initAttributes()
        
        mainLayout = QVBoxLayout()
        equationLayout = QHBoxLayout()
        
        metricsTreeWidget = QTreeWidget()
        metricsTreeWidget.setHeaderHidden(True)
        metricsTreeWidget.setFont(font)
        self.metricsTreeWidget = metricsTreeWidget
        
        for groupName, features in features_groups.items():
            topLevelTreeWidgetItem = QTreeWidgetItem(metricsTreeWidget)
            topLevelTreeWidgetItem.setText(0, groupName)
            metricsTreeWidget.addTopLevelItem(topLevelTreeWidgetItem)
            self.addTreeItems(
                topLevelTreeWidgetItem, features, isCol=True, 
                name_to_col_mapper=group_name_to_col_mapper.get(groupName)
            )
            
        operatorsLayout = self.createOperatorsLayout()
        newFeatureNameLayout = self.createNewFeatureNameLayout()
        equationDisplayLayout = self.createEquationDisplayLayout()
        
        equationLayout.addLayout(newFeatureNameLayout)
        equationLayout.addWidget(QLabel(' = '))
        equationLayout.addLayout(equationDisplayLayout)
        equationLayout.setStretch(0,1)
        equationLayout.setStretch(1,0)
        equationLayout.setStretch(2,2)
        
        testOutputLayout = self.createTestOutputLayout()
        buttonsLayout = self.createButtonsOutputLayout()
        
        instructions = html_utils.paragraph("""
            <b>Double-click</b> on any of the <b>available measurements</b>
            to add it to the equation.<br><br>
            Before clicking the `Ok` button, check that the equation returns 
            the expected result by clicking the `Test output` button. 
        """)
        
        mainLayout.addWidget(QLabel(instructions))
        mainLayout.addWidget(QLabel('Available measurements:'))
        mainLayout.addWidget(metricsTreeWidget)
        mainLayout.addLayout(operatorsLayout)
        mainLayout.addLayout(equationLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addLayout(testOutputLayout)

        
        metricsTreeWidget.itemDoubleClicked.connect(self.addFeatureName)
        self.setLayout(mainLayout)
        self.setFont(font)

        self.setStyleSheet(TREEWIDGET_STYLESHEET)
    
    def setExpandedAll(self, expanded):
        if expanded:
            self.expandAll()
        else:
            for i in range(self.metricsTreeWidget.topLevelItemCount()):
                topLevelItem = self.metricsTreeWidget.topLevelItem(i)
                topLevelItem.setExpanded(False)
    
    def expandAll(self):
        for i in range(self.metricsTreeWidget.topLevelItemCount()):
            topLevelItem = self.metricsTreeWidget.topLevelItem(i)
            topLevelItem.setExpanded(True)
    
    def addTreeItems(
            self, parentItem, itemsText, isCol=False, name_to_col_mapper=None
        ):
        for text in itemsText:
            _item = QTreeWidgetItem(parentItem)
            _item.setText(0, text)
            parentItem.addChild(_item)
            if isCol:
                _item.isCol = True
            _item.variable_name = text
            if name_to_col_mapper is None:
                continue
            
            col_name = name_to_col_mapper.get(text, None)
            if col_name is None:
                continue
            
            _item.variable_name = col_name
            
    
    def addFeatureName(self, item, column):
        if not hasattr(item, 'isCol'):
            return

        colName = item.variable_name
        text = f'{self.equationDisplay.toPlainText()}{colName}'
        self.equationDisplay.setPlainText(text)
        self.clearLenghts.append(len(colName))
        self.equationColNames.append(colName)
    
    def clearEquation(self):
        self.isOperatorMode = False
        self.equationDisplay.setPlainText('')
        self.initAttributes()
    
    def createButtonsOutputLayout(self):
        buttonsLayout = QHBoxLayout()

        cancelButton = widgets.cancelPushButton('Cancel')
        helpButton = widgets.infoPushButton('  Help...')
        testButton = widgets.calcPushButton('Test output')
        okButton = widgets.okPushButton(' Ok ')
        okButton.setDisabled(True)
        self.okButton = okButton

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(helpButton)
        buttonsLayout.addWidget(testButton)
        buttonsLayout.addWidget(okButton)
        
        helpButton.clicked.connect(self.showHelp)
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        testButton.clicked.connect(self.test_cb)
        
        return buttonsLayout

    def ok_cb(self):
        if not self.newFeatureNameLineEdit.text():
            self.warnEmptyEquationName()
            return
        
        self.equation = self.equationDisplay.toPlainText()
        self.newFeatureName = self.newFeatureNameLineEdit.text()
        self.cancel = False
        self.close()
        self.sigOk.emit(self)
    
    def test_cb(self):
        # Evaluate equation with random inputs
        equation = self.equationDisplay.toPlainText()
        random_data = np.random.rand(1, len(self.equationColNames))*5
        df = pd.DataFrame(
            data=random_data,
            columns=self.equationColNames
        ).round(5)
        newColName = self.newFeatureNameLineEdit.text()
        try:
            df[newColName] = df.eval(equation)
        except Exception as e:
            traceback.print_exc()
            self.testOutputDisplay.setHtml(html_utils.paragraph(e))
            self.testOutputDisplay.setStyleSheet("border: 2px solid red")
            return

        self.testOutputDisplay.setStyleSheet("border: 2px solid green")
        self.okButton.setDisabled(False)

        result = df.round(5).iloc[0][newColName]

        # Substitute numbers into equation
        inputs = df.iloc[0]
        equation_numbers = equation
        for c, col in enumerate(self.equationColNames):
            equation_numbers = equation_numbers.replace(col, str(inputs[c]))

        # Format output into html text
        cols = self.equationColNames
        inputs_txt = [f'{col} = {input}' for col, input in zip(cols, inputs)]
        list_html = html_utils.to_list(inputs_txt)
        text = html_utils.paragraph(f"""
            By substituting the following random inputs:
            {list_html}
            we get the equation:<br><br>
            &nbsp;&nbsp;<code>{newColName} = {equation_numbers}</code><br><br>
            that <b>equals to</b>:<br><br>
            &nbsp;&nbsp;<code>{newColName} = {result}</code>
        """)
        self.testOutputDisplay.setHtml(text)
    
    def warnEmptyEquationName(self):
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph("""
            "New measurement name" field <b>cannot be empty</b>!
        """)
        msg.critical(
            self, 'Empty new measurement name', txt
        )
    
    def showHelp(self):
        pass
    
    def createTestOutputLayout(self):
        testOutputLayout = QVBoxLayout()
        testOutputLayout.addWidget(QLabel('Result of test with random inputs:'))
        testOutputDisplay = QTextEdit()
        testOutputDisplay.setReadOnly(True)
        self.testOutputDisplay = testOutputDisplay
        testOutputLayout.addWidget(testOutputDisplay)
        testOutputLayout.setStretch(0,0)
        testOutputLayout.setStretch(1,1)
        
        return testOutputLayout
    
    def createEquationDisplayLayout(self):
        equationDisplayLayout = QVBoxLayout()
        equationDisplayLayout.addWidget(QLabel('Equation:'))
        equationDisplay = QPlainTextEdit()
        # equationDisplay.setReadOnly(True)
        self.equationDisplay = equationDisplay
        equationDisplayLayout.addWidget(equationDisplay)
        equationDisplayLayout.setStretch(0,0)
        equationDisplayLayout.setStretch(1,1)
        return equationDisplayLayout
    
    def createNewFeatureNameLayout(self):
        newFeatureNameLayout = QVBoxLayout()
        newFeatureNameLineEdit = widgets.alphaNumericLineEdit()
        newFeatureNameLineEdit.setAlignment(Qt.AlignCenter)
        self.newFeatureNameLineEdit = newFeatureNameLineEdit
        newFeatureNameLayout.addStretch(1)
        newFeatureNameLayout.addWidget(QLabel('New measurement name:'))
        newFeatureNameLayout.addWidget(newFeatureNameLineEdit)
        newFeatureNameLayout.addStretch(1)
        return newFeatureNameLayout
    
    def createOperatorsLayout(self):
        operatorsLayout = QHBoxLayout()
        operatorsLayout.addStretch(1)

        iconSize = 24

        self.operatorButtons = []
        self.operators = [
            ('add', '+'),
            ('subtract', '-'),
            ('multiply', '*'),
            ('divide', '/'),
            ('open_bracket', '('),
            ('close_bracket', ')'),
            ('square', '**2'),
            ('pow', '**'),
            ('ln', 'log('),
            ('log10', 'log10('),
        ]
        operatorFont = QFont()
        operatorFont.setPixelSize(16)
        for name, text in self.operators:
            button = QPushButton()
            button.setIcon(QIcon(f':{name}.svg'))
            button.setIconSize(QSize(iconSize,iconSize))
            button.text = text
            operatorsLayout.addWidget(button)
            self.operatorButtons.append(button)
            button.clicked.connect(self.addOperator)
            # button.setFont(operatorFont)

        clearButton = QPushButton()
        clearButton.setIcon(QIcon(':clear.svg'))
        clearButton.setIconSize(QSize(iconSize,iconSize))
        clearButton.setFont(operatorFont)

        clearEntryButton = QPushButton()
        clearEntryButton.setIcon(QIcon(':backspace.svg'))
        clearEntryButton.setFont(operatorFont)
        clearEntryButton.setIconSize(QSize(iconSize,iconSize))

        operatorsLayout.addWidget(clearButton)
        operatorsLayout.addWidget(clearEntryButton)
        operatorsLayout.addStretch(1)
        
        clearButton.clicked.connect(self.clearEquation)
        clearEntryButton.clicked.connect(self.clearEntryEquation)
        
        return operatorsLayout

    def addOperator(self):
        button = self.sender()
        text = f'{self.equationDisplay.toPlainText()}{button.text}'
        self.equationDisplay.setPlainText(text)
        self.clearLenghts.append(len(button.text))

    def clearEquation(self):
        self.isOperatorMode = False
        self.equationDisplay.setPlainText('')
        self.initAttributes()
    
    def initAttributes(self):
        self.clearLenghts = []
        self.equationColNames = []
        self.channelLessColnames = []
    
    def clearEntryEquation(self):
        if not self.clearLenghts:
            return

        text = self.equationDisplay.toPlainText()
        newText = text[:-self.clearLenghts[-1]]
        clearedText = text[-self.clearLenghts[-1]:]
        self.clearLenghts.pop(-1)
        self.equationDisplay.setPlainText(newText)
        if clearedText in self.equationColNames:
            self.equationColNames.remove(clearedText)
        if clearedText in self.channelLessColnames:
            self.channelLessColnames.remove(clearedText)

class QInput(QBaseDialog):
    def __init__(self, parent=None, title='Input'):
        self.cancel = True
        self.allowEmpty = True

        super().__init__(parent)

        self.setWindowTitle(title)

        self.mainLayout = QVBoxLayout()

        self.infoLabel = QLabel()
        self.mainLayout.addWidget(self.infoLabel)

        promptLayout = QHBoxLayout()
        self.promptLabel = QLabel()
        promptLayout.addWidget(self.promptLabel)
        self.lineEdit = QLineEdit()
        promptLayout.addWidget(self.lineEdit)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        self.mainLayout.addLayout(promptLayout)
        self.mainLayout.addSpacing(20)
        self.mainLayout.addLayout(buttonsLayout)

        self.buttonsLayout = buttonsLayout

        self.setFont(font)
        self.setLayout(self.mainLayout)
    
    def askText(self, prompt, infoText='', allowEmpty=False):
        self.allowEmpty = allowEmpty
        if infoText:
            infoText = f'{infoText}<br>'
            self.infoLabel.setText(html_utils.paragraph(infoText))
        self.promptLabel.setText(prompt)
        self.exec_(resizeWidthFactor=1.5)

    def ok_cb(self):
        self.answer = self.lineEdit.text()
        if not self.allowEmpty and not self.answer:
            msg = widgets.myMessageBox(showCentered=False)
            msg.critical(self, 'Empty', 'Entry cannot be empty.')
            return
        self.cancel = False
        self.close()

class InstallPyTorchDialog(QBaseDialog):
    def __init__(self, parent=None, caller_name='Cell-ACDC'):
        super().__init__(parent=parent)
        
        self.cancel = True
        
        mainLayout = QVBoxLayout()
        
        innerLayout = QGridLayout()
        
        iconLabel = QLabel(self)
        standardIcon = getattr(QStyle, 'SP_MessageBoxInformation')
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        iconLabel.setPixmap(pixmap)
        innerLayout.addWidget(iconLabel, 0, 0, alignment=Qt.AlignTop)
        
        href = html_utils.href_tag('How to install PyTorch', urls.install_pytorch)
        important = html_utils.to_admonition("""
            Should you choose to install PyTorch yourself, <b>make sure to 
            activate<br>
            the correct <code>acdc</code> environment first</b>.
        """, admonition_type='important')
        
        infoText = html_utils.paragraph(f"""
            {caller_name} needs to <b>install the package</b> <code>PyTorch</code>.<br><br>
            Select your preferences and click ok to install it now. 
            You will have to <b>confirm the installation in the terminal</b>.<br><br>
            Alternatively, you can close {caller_name} and run the command 
            yourself.<br><br>
            For more details see this guide: {href}<br>
            {important}
        """)
        innerLayout.addWidget(QLabel(infoText), 0, 1)
        innerLayout.addItem(QSpacerItem(10, 10), 1, 1)
        
        preferencesLayout = QGridLayout()
        
        row = 0
        self.osCombobox = QComboBox()
        self.osCombobox.addItems(['Linux', 'Mac', 'Windows'])
        preferencesLayout.addWidget(QLabel('Your OS'), row, 0)
        preferencesLayout.addWidget(self.osCombobox, row, 1)
        
        if is_mac:
            self.osCombobox.setCurrentText('Mac')
        elif is_win:
            self.osCombobox.setCurrentText('Windows')
        
        row += 1
        self.pkgManagerCombobox = QComboBox()
        self.pkgManagerCombobox.addItems(['Conda', 'Pip'])
        if not is_conda_env():
            self.pkgManagerCombobox.setCurrentText('Pip')
            self.pkgManagerCombobox.setDisabled(True)
        
        preferencesLayout.addWidget(QLabel('Package manager'), row, 0)
        preferencesLayout.addWidget(self.pkgManagerCombobox, row, 1)
        
        row += 1
        self.cmptPlatformCombobox = QComboBox()
        self.cmptPlatformCombobox.addItems(
            ['CPU', 'CUDA 11.8 (NVIDIA GPU)', 'CUDA 12.1 (NVIDIA GPU)']
        )
        
        preferencesLayout.addWidget(QLabel('Compute Platform'), row, 0)
        preferencesLayout.addWidget(self.cmptPlatformCombobox, row, 1)
        
        row += 1
        self.commandWidget = widgets.CopiableCommandWidget(
            command='pip install torch'
        )
        preferencesLayout.addWidget(QLabel('Run this command: '), row, 0)
        preferencesLayout.addWidget(self.commandWidget, row, 1, 1, 2)
        preferencesLayout.setColumnStretch(0, 0)
        preferencesLayout.setColumnStretch(1, 0)
        preferencesLayout.setColumnStretch(2, 1)
        
        innerLayout.addLayout(preferencesLayout, 2, 1)
        
        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addLayout(innerLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
        
        self.osCombobox.currentTextChanged.connect(self.updateCommand)
        self.pkgManagerCombobox.currentTextChanged.connect(self.updateCommand)
        self.cmptPlatformCombobox.currentTextChanged.connect(self.updateCommand)
        
        self.updateCommand()
    
    def updateCommand(self, *args, **kwargs):
        osText = self.osCombobox.currentText()
        pkgManager = self.pkgManagerCombobox.currentText()
        cmptPlatform = self.cmptPlatformCombobox.currentText()
        command = pytorch_commands[osText][pkgManager][cmptPlatform]
        self.commandWidget.setCommand(command)
    
    def ok_cb(self):
        self.command = self.commandWidget.command()
        self.cancel = False
        self.close()

class ExportToVideoParametersDialog(QBaseDialog):
    sigOk = Signal(dict)
    sigAddScaleBar = Signal(bool)
    sigAddTimestamp = Signal(bool)
    sigRescaleIntensLut = Signal(str, str)
    
    def __init__(
            self, channels, parent=None, startFolderpath='', startFilename='', 
            startFrameNum=1, SizeT=1, SizeZ=1, isTimelapseVideo=True, 
            isScaleBarPresent=False, isTimestampPresent=False, 
            rescaleIntensChannelHowMapper=None
        ):
        self.cancel = True
        
        if rescaleIntensChannelHowMapper is None:
            rescaleIntensChannelHowMapper = {}
        
        super().__init__(parent=parent)
        
        self.setWindowTitle('Preferences for output video')
        
        mainLayout = QVBoxLayout()
        
        gridLayout = QGridLayout()
        
        navVar = 'frame number' if isTimelapseVideo else 'z-slice'
        maxNavVar = SizeT if isTimelapseVideo else SizeZ
        
        self.isTimelapseVideo = isTimelapseVideo
        
        row = 0
        gridLayout.addWidget(QLabel(f'Start {navVar}:'), row, 0)
        self.startNavVarNumberEntry = widgets.IntLineEdit(allowNegative=False)
        self.startNavVarNumberEntry.setMinimum(1)
        self.startNavVarNumberEntry.setValue(startFrameNum)
        gridLayout.addWidget(self.startNavVarNumberEntry, row, 1)
        
        row += 1
        gridLayout.addWidget(QLabel(f'Stop {navVar}:'), row, 0)
        self.stopNavVarNumberEntry = widgets.IntLineEdit(allowNegative=False)
        self.stopNavVarNumberEntry.setMaximum(maxNavVar)
        self.stopNavVarNumberEntry.setValue(maxNavVar)
        gridLayout.addWidget(self.stopNavVarNumberEntry, row, 1)
        
        row += 1
        gridLayout.addWidget(QLabel('File format:'), row, 0)
        self.fileFormatCombobox = QComboBox()
        self.fileFormatCombobox.addItems(['MP4', 'AVI'])
        gridLayout.addWidget(self.fileFormatCombobox, row, 1)
        
        row += 1
        gridLayout.addWidget(QLabel('Frame rate (FPS):'), row, 0)
        self.fpsWidget = widgets.FloatLineEdit(allowNegative=False)
        self.fpsWidget.setValue(10.0)
        gridLayout.addWidget(self.fpsWidget, row, 1)
        
        row += 1
        gridLayout.addWidget(QLabel('Folder path:'), row, 0)
        self.folderPathLineEdit = widgets.ElidingLineEdit(minWidth=240)
        self.folderPathLineEdit.setText(startFolderpath)
        gridLayout.addWidget(self.folderPathLineEdit, row, 1)
        self.browseButton = widgets.browseFileButton(
            start_dir=startFolderpath, openFolder=True
        )
        gridLayout.addWidget(self.browseButton, row, 2)
        
        row += 1
        gridLayout.addWidget(QLabel('Filename:'), row, 0)
        self.filenameLineEdit = widgets.alphaNumericLineEdit()
        self.filenameLineEdit.setAlignment(Qt.AlignCenter)
        self.filenameLineEdit.setText(startFilename)
        gridLayout.addWidget(self.filenameLineEdit, row, 1)
        self.fileFormatLabel = QLabel('.mp4')
        gridLayout.addWidget(self.fileFormatLabel, row, 2)
        
        row += 1
        gridLayout.addWidget(QLabel('Add Scale Bar:'), row, 0)
        self.addScaleBarToggle = widgets.Toggle()
        gridLayout.addWidget(
            self.addScaleBarToggle, row, 1, alignment=Qt.AlignCenter
        )
        self.addScaleBarToggle.setChecked(isScaleBarPresent)
        
        if isTimelapseVideo:
            row += 1
            gridLayout.addWidget(QLabel('Add timestamp:'), row, 0)
            self.addTimestampToggle = widgets.Toggle()
            gridLayout.addWidget(
                self.addTimestampToggle, row, 1, alignment=Qt.AlignCenter
            )
            self.addTimestampToggle.setChecked(isTimestampPresent)
        
        for channel in channels:
            row += 1
            labelText = f'Rescale intensities (LUT) <i>{channel}</i>:'
            gridLayout.addWidget(QLabel(labelText), row, 0)
            rescaleItems = ['Rescale each 2D image']
            if SizeZ > 1:
                rescaleItems.append('Rescale across z-stack')
            if isTimelapseVideo:
                rescaleItems.append('Rescale across time frames')
            rescaleItems.append('Choose custom levels...')
            rescaleItems.append('Do no rescale, display raw image')
            rescaleIntensCombobox = QComboBox()
            rescaleIntensCombobox.addItems(rescaleItems)
            rescaleIntensHow = rescaleIntensChannelHowMapper.get(channel)
            if rescaleIntensHow is not None:
                rescaleIntensCombobox.setCurrentText(rescaleIntensHow)
            gridLayout.addWidget(rescaleIntensCombobox, row, 1)
            rescaleIntensCombobox.textActivated.connect(
                partial(self.emitRescaleIntens, channel=channel)
            )
        
        row += 1
        gridLayout.addWidget(QLabel('Save a PNG for each frame:'), row, 0)
        self.saveFramesToggle = widgets.Toggle()
        gridLayout.addWidget(
            self.saveFramesToggle, row, 1, alignment=Qt.AlignCenter
        )
        
        gridLayout.setColumnStretch(0, 0)
        gridLayout.setColumnStretch(1, 1)
        gridLayout.setColumnStretch(2, 0)
        
        self.fileFormatCombobox.currentTextChanged.connect(
            self.updateFileFormat
        )
        self.browseButton.sigPathSelected.connect(self.updateFolderPath)
        self.addScaleBarToggle.toggled.connect(self.addScaleBarToggled)
        if isTimelapseVideo:
            self.addTimestampToggle.toggled.connect(self.addTimestampToggled)
        
        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.setText('Export')
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addLayout(gridLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def emitRescaleIntens(self, how, channel=''):
        self.sigRescaleIntensLut.emit(how, channel)
    
    def addScaleBarToggled(self, checked):
        self.sigAddScaleBar.emit(checked)
    
    def addTimestampToggled(self, checked):
        self.sigAddTimestamp.emit(checked)
    
    def updateFolderPath(self, folderPath):
        self.folderPathLineEdit.setText(folderPath)
        self.browseButton.setStartPath(folderPath)
    
    def updateFileFormat(self, fileFormat):
        self.fileFormatLabel.setText(f'.{fileFormat.lower()}')
    
    def validateFolderPath(self):
        folderPath = self.folderPathLineEdit.text()
        if os.path.exists(folderPath) and os.path.isdir(folderPath):
            return True
        
        text = html_utils.paragraph(
            'The selected folder path is not a valid folder or does not exist'
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Not a valid folder', text)
        return False
    
    def validateFilename(self):
        filename = self.filenameLineEdit.text()
        if filename:
            return True
        
        text = html_utils.paragraph(
            'The filename cannot be empty!'
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Not a valid folder', text)
        return False
    
    def validate(self):
        proceed = self.validateFolderPath()
        if not proceed:
            return False
        
        proceed = self.validateFilename()
        if not proceed:
            return False
        
        return True
    
    def preferences(self, makedirs=True):
        filename = f'{self.filenameLineEdit.text()}{self.fileFormatLabel.text()}'
        avi_filename = f'{self.filenameLineEdit.text()}.avi'
        avi_filepath = os.path.join(self.folderPathLineEdit.text(), avi_filename)
        png_foldername = (
            f'{self.filenameLineEdit.text()}_frames_PNG'
        )
        pngs_folderpath = os.path.join(
            self.folderPathLineEdit.text(), png_foldername
        )
        if makedirs:
            os.makedirs(pngs_folderpath, exist_ok=True)
        preferences = {
            'start_nav_var_num': self.startNavVarNumberEntry.value(),
            'stop_nav_var_num': self.stopNavVarNumberEntry.value(),
            'filepath': os.path.join(self.folderPathLineEdit.text(), filename), 
            'filename': self.filenameLineEdit.text(),
            'avi_filepath': avi_filepath,
            'pngs_folderpath': pngs_folderpath,
            'num_digits': len(str(self.stopNavVarNumberEntry.value())),
            'fps': self.fpsWidget.value(),
            'save_pngs':  self.saveFramesToggle.isChecked(),
            'is_timelapse': self.isTimelapseVideo
        }
        return preferences
    
    def ok_cb(self):
        proceed = self.validate()
        if not proceed:
            return
        self.cancel = False
        self.sigOk.emit(self.preferences())
        self.selected_preferences = self.preferences()
        self.close()
    
class TimestampPropertiesDialog(QBaseDialog):
    sigValueChanged = Signal(object)
    
    def __init__(self, parent=None, **properties):
        super().__init__(parent=parent)
        
        self.cancel = True
        self.setWindowTitle('Timestamp preferences')
        
        mainLayout = QVBoxLayout()
        
        formLayout = widgets.FormLayout()
        formLayout.setVerticalSpacing(10)
        formLayout.setHorizontalSpacing(50)
        
        row = 0
        self.colorButton = widgets.myColorButton(color=(255, 255, 255))
        if properties.get('color') is not None:
            self.colorButton.setColor(properties.get('color'))
        colorFormWidget = widgets.formWidget(
            self.colorButton, labelTextLeft='Color',
            widgetAlignment=Qt.AlignCenter, stretchWidget=False
        )
        formLayout.addFormWidget(
            colorFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        ) 
        
        row += 1
        fontSizeWidget = widgets.FontSizeWidget()
        if properties.get('font_size') is not None:
            fontSizeWidget.setValue(properties.get('font_size'))
        else:
            fontSizeWidget.setValue(12)
        fontSizeFormWidget = widgets.formWidget(
            fontSizeWidget, labelTextLeft='Font size (px)'
        )
        self.fontSizeWidget = fontSizeWidget
        formLayout.addFormWidget(
            fontSizeFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )
        
        row += 1
        locCombobox = QComboBox()
        locFormWidget = widgets.formWidget(
            locCombobox, labelTextLeft='Location'
        )
        locCombobox.addItems(
            ['Top-left', 'Top-right', 'Bottom-left', 'Bottom-right', 'Custom']
        )
        loc = properties.get('loc')
        if isinstance(loc, str):
            locCombobox.setCurrentText(loc)
        formLayout.addFormWidget(
            locFormWidget, row=row, 
            leftLabelAlignment=Qt.AlignLeft
        )       
        self.locCombobox = locCombobox
        
        mainLayout.addLayout(formLayout)
        
        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch()
        
        self.setLayout(mainLayout)
        self.setFont(font)
        
        self.colorButton.clicked.disconnect()
        self.colorButton.clicked.connect(self.selectColor)
        
        self.locCombobox.currentTextChanged.connect(self.onValueChanged)
        self.fontSizeWidget.sigTextChanged.connect(self.onValueChanged)
    
    def onValueChanged(self, object):
        self.sigValueChanged.emit(self.kwargs())
    
    def selectColor(self):
        color = self.colorButton.color()
        self.colorButton.origColor = color
        self.colorButton.colorDialog.setCurrentColor(color)
        self.colorButton.colorDialog.setWindowFlags(
            Qt.Window | Qt.WindowStaysOnTopHint
        )
        self.colorButton.colorDialog.setParent(self)
        self.colorButton.colorDialog.open()
        w = self.width()
        left = self.pos().x()
        colorDialogTop = self.colorButton.colorDialog.pos().y()
        self.colorButton.colorDialog.move(w+left+10, colorDialogTop)
    
    def kwargs(self):
        kwargs = {
            'color': self.colorButton.color(),
            'loc': self.locCombobox.currentText().lower(),
            'font_size': self.fontSizeWidget.text(),
        }
        return kwargs
    
    def ok_cb(self):
        self.cancel = False
        self.close()

class ExportToImageParametersDialog(QBaseDialog):
    sigOk = Signal(dict)
    sigAddScaleBar = Signal(bool)
    sigRangeChanged = Signal(object)
    
    def __init__(
            self, parent=None, startFolderpath='', startFilename='', 
            startViewRange=None, isScaleBarPresent=False
        ):
        self.cancel = True
        
        super().__init__(parent=parent)
        
        self.setWindowTitle('Preferences for output image')
        
        mainLayout = QVBoxLayout()
        
        gridLayout = QGridLayout()
        
        row = 0
        gridLayout.addWidget(QLabel('View range X axis:'), row, 0)
        self.xRangeSelector = widgets.RangeSelector()
        if startViewRange is not None:
            xRange, yRange = startViewRange
            self.xRangeSelector.setRange(*xRange)
        gridLayout.addWidget(self.xRangeSelector, row, 1)
        
        row += 1
        gridLayout.addWidget(QLabel('View range Y axis:'), row, 0)
        self.yRangeSelector = widgets.RangeSelector()
        if startViewRange is not None:
            xRange, yRange = startViewRange
            self.yRangeSelector.setRange(*yRange)
        gridLayout.addWidget(self.yRangeSelector, row, 1)
        
        row += 1
        gridLayout.addWidget(QLabel('File format:'), row, 0)
        self.fileFormatCombobox = QComboBox()
        self.fileFormatCombobox.addItems(['SVG', 'PNG', 'TIF', 'JPEG'])
        gridLayout.addWidget(self.fileFormatCombobox, row, 1)
        
        row += 1
        gridLayout.addWidget(QLabel('Folder path:'), row, 0)
        self.folderPathLineEdit = widgets.ElidingLineEdit(minWidth=240)
        self.folderPathLineEdit.setText(startFolderpath)
        gridLayout.addWidget(self.folderPathLineEdit, row, 1)
        self.browseButton = widgets.browseFileButton(
            start_dir=startFolderpath, openFolder=True
        )
        gridLayout.addWidget(self.browseButton, row, 2)
        
        row += 1
        gridLayout.addWidget(QLabel('Filename:'), row, 0)
        self.filenameLineEdit = widgets.alphaNumericLineEdit()
        self.filenameLineEdit.setAlignment(Qt.AlignCenter)
        self.filenameLineEdit.setText(startFilename)
        gridLayout.addWidget(self.filenameLineEdit, row, 1)
        self.fileFormatLabel = QLabel(
            f'.{self.fileFormatCombobox.currentText().lower()}'
        )
        gridLayout.addWidget(self.fileFormatLabel, row, 2)
        
        row += 1
        gridLayout.addWidget(QLabel('Add Scale Bar:'), row, 0)
        self.addScaleBarToggle = widgets.Toggle()
        gridLayout.addWidget(
            self.addScaleBarToggle, row, 1, alignment=Qt.AlignCenter
        )
        self.addScaleBarToggle.setChecked(isScaleBarPresent)
        
        self.fileFormatCombobox.currentTextChanged.connect(
            self.updateFileFormat
        )
        self.browseButton.sigPathSelected.connect(self.updateFolderPath)
        self.addScaleBarToggle.toggled.connect(self.addScaleBarToggled)
        self.xRangeSelector.sigRangeChanged.connect(self.rangeChanged)
        self.yRangeSelector.sigRangeChanged.connect(self.rangeChanged)
        
        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.setText('Export')
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addLayout(gridLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def updateViewRangeExportToImageDialog(self, viewBox, viewRange, changed):
        xRange, yRange = viewRange
        self.xRangeSelector.setRangeNoEmit(*xRange)
        self.yRangeSelector.setRangeNoEmit(*yRange)
    
    def rangeChanged(self, *args):
        xRange = self.xRangeSelector.range()
        yRange = self.yRangeSelector.range()
        self.sigRangeChanged.emit((xRange, yRange))
    
    def addScaleBarToggled(self, checked):
        self.sigAddScaleBar.emit(checked)
    
    def updateFolderPath(self, folderPath):
        self.folderPathLineEdit.setText(folderPath)
        self.browseButton.setStartPath(folderPath)
    
    def updateFileFormat(self, fileFormat):
        self.fileFormatLabel.setText(f'.{fileFormat.lower()}')
    
    def validateFolderPath(self):
        folderPath = self.folderPathLineEdit.text()
        if os.path.exists(folderPath) and os.path.isdir(folderPath):
            return True
        
        text = html_utils.paragraph(
            'The selected folder path is not a valid folder or does not exist'
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Not a valid folder', text)
        return False
    
    def validateFilename(self):
        filename = self.filenameLineEdit.text()
        if filename:
            return True
        
        text = html_utils.paragraph(
            'The filename cannot be empty!'
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Not a valid folder', text)
        return False
    
    def validate(self):
        proceed = self.validateFolderPath()
        if not proceed:
            return False
        
        proceed = self.validateFilename()
        if not proceed:
            return False
        
        return True
    
    def preferences(self):
        filename = f'{self.filenameLineEdit.text()}{self.fileFormatLabel.text()}'
        preferences = {
            'view_range_x': self.xRangeSelector.range(),
            'view_range_y': self.yRangeSelector.range(),
            'filepath': os.path.join(self.folderPathLineEdit.text(), filename), 
            'filename': self.filenameLineEdit.text(),
        }
        return preferences
    
    def ok_cb(self):
        proceed = self.validate()
        if not proceed:
            return
        self.cancel = False
        self.sigOk.emit(self.preferences())
        self.selected_preferences = self.preferences()
        self.close()
    
class DataPrepSubCropsPathsDialog(QBaseDialog):
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

class PreProcessParamsGroupbox(QWidget):
    sigLoadRecipe = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        mainLayout = QVBoxLayout()
        
        groupbox = QGroupBox()
        self.groupbox = groupbox
        
        groupbox.setTitle('Pre-processing')
        groupbox.setCheckable(True)
        
        self.gridLayout = QGridLayout()        
        self.row = -1
        self.stepsWidgets = []
        
        self.gridLayout.setColumnStretch(0, 0)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 0)
        self.gridLayout.setColumnStretch(3, 0)
        self.gridLayout.setColumnStretch(4, 0)
        groupbox.setLayout(self.gridLayout)
        
        buttonsLayout = QHBoxLayout()
        # saveRecipeButton = widgets.savePushButton()
        loadRecipeButton = widgets.reloadPushButton('Load last parameters')
        
        buttonsLayout.addStretch(1)
        # buttonsLayout.addWidget(saveRecipeButton)
        buttonsLayout.addWidget(loadRecipeButton)
        
        loadRecipeButton.clicked.connect(self.emitLoadRecipe)
        
        mainLayout.addWidget(groupbox)
        mainLayout.addLayout(buttonsLayout)
        
        self.addStep()
        
        mainLayout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(mainLayout)
    
    def setChecked(self, checked):
        self.groupbox.setChecked(checked)
    
    def emitLoadRecipe(self):
        self.sigLoadRecipe.emit()
    
    def loadRecipe(self, configPars: dict):
        for stepWidgets in self.stepsWidgets:
            stepWidgets['delButton'].click()
        
        configPars = self.sortStepsConfigPars(configPars)
        for i in range(1, len(configPars)):
            self.stepsWidgets[i-1]['addButton'].click()
        
        for s, (section, section_items) in enumerate(configPars.items()):
            method = section_items['method']
            self.stepsWidgets[s]['selector'].setCurrentText(method)
            for label, widget in self.stepsWidgets[s]['widgets']:
                option = label.text().lower()
                value = section_items[option]
                widget.setText(value)
        
        self.setChecked(True)
    
    def sortStepsConfigPars(self, configPars: dict):
        sortedConfigPars = {}
        sortedKeys = sorted(
            configPars.keys(), 
            key=lambda key: int(re.findall(r'step(\d+)', key)[0])
        )
        for key in sortedKeys:
            sortedConfigPars[key] = configPars[key]
        return sortedConfigPars
        
    def addStep(self):
        stepWidgets = {}
        
        self.row += 1
        
        label = QLabel(f'Step {len(self.stepsWidgets)+1}: ')
        self.gridLayout.addWidget(label, self.row, 0)
        stepWidgets['stepLabel'] = label
        
        selector = widgets.PreProcessingSelector()
        self.gridLayout.addWidget(selector, self.row, 1)
        stepWidgets['selector'] = selector
        
        infoButton = widgets.infoPushButton()
        self.gridLayout.addWidget(infoButton, self.row, 2)
        infoButton.clicked.connect(partial(self.showInfo, selector=selector))
        stepWidgets['infoButton'] = infoButton
        
        addButton = widgets.addPushButton()
        self.gridLayout.addWidget(addButton, self.row, 3)
        addButton.clicked.connect(self.addStep)
        stepWidgets['addButton'] = addButton

        delButton = widgets.delPushButton()
        self.gridLayout.addWidget(delButton, self.row, 4)
        delButton.clicked.connect(self.removeStep)
        delButton.idx = len(self.stepsWidgets)
        stepWidgets['delButton'] = delButton
        
        self.row += 1
        stepWidgets['widgets'] = []
        for labelText, widgetName in selector.widgets().items():
            label = QLabel(labelText)
            self.gridLayout.addWidget(label, self.row, 0) 
            widget = getattr(widgets, widgetName)()
            self.gridLayout.addWidget(widget, self.row, 1)
            self.row += 1
            stepWidgets['widgets'].append((label, widget))

        hline = widgets.QHLine()
        self.gridLayout.addWidget(hline, self.row, 0, 1, 5)
        stepWidgets['hline'] = hline
        self.row += 1
        
        self.stepsWidgets.append(stepWidgets)
    
    def showInfo(self, checked=False, selector=None):
        if selector is None:
            return
        
        htmlText = selector.htmlInfo()
        htmlText = html_utils.paragraph(htmlText)
        
        method = selector.currentText()
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, f'Info about `{method}`', htmlText)
    
    def removeStep(self, checked=False, idx=None):
        if len(self.stepsWidgets) == 1:
            self.setChecked(False)
            return
        
        if idx is None:
            idx = self.sender().idx
        
        stepWidgets = self.stepsWidgets[idx]
        self.gridLayout.removeWidget(stepWidgets['stepLabel'])
        self.gridLayout.removeWidget(stepWidgets['selector'])
        self.gridLayout.removeWidget(stepWidgets['infoButton'])
        self.gridLayout.removeWidget(stepWidgets['addButton'])
        self.gridLayout.removeWidget(stepWidgets['delButton'])
        self.gridLayout.removeWidget(stepWidgets['hline'])
        for label, widget in stepWidgets['widgets']:
            self.gridLayout.removeWidget(label)    
            self.gridLayout.removeWidget(widget)
        
        for s, stepWidgets in enumerate(self.stepsWidgets):
            label.setText(f'Step {s+1}: ')
        
        del self.stepsWidgets[idx]
    
    def isChecked(self):
        return self.groupbox.isChecked()
    
    def recipe(self):
        recipe = []
        if not self.groupbox.isChecked():
            return recipe
        
        for stepWidgets in self.stepsWidgets:
            method = stepWidgets['selector'].currentText()
            step_kwargs = {}
            for label, widget in stepWidgets['widgets']:
                step_kwargs[label.text().lower()] = widget.value()
            recipe.append({
                'method': method, 'kwargs': step_kwargs
            })
        
        return recipe
    
    def recipeConfigPars(self, model_name):
        cp = config.ConfigParser()
        if not self.groupbox.isChecked():
            return cp
        
        for s, step in enumerate(self.recipe()):
            section = f'{model_name}.preprocess.step{s+1}'
            cp[section] = {}
            cp[section]['method'] = step['method']
            for option, value in step['kwargs'].items():
                cp[section][option] = str(value)
        return cp

class InitFijiMacroDialog(QBaseDialog):
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
            'Positions (aka "series") separated, one for each file'
        ])
        gridLayout.addWidget(self.filesStructureCombobox, row, 1)
        infoButton = widgets.infoPushButton()
        gridLayout.addWidget(infoButton, row, 2)
        infoButton.clicked.connect(self.showInfoFileStructure)
        
        row += 1
        label = QLabel('Folder with raw microscopy files: ')
        gridLayout.addWidget(label, row, 0)
        self.folderPathLineEdit = widgets.ElidingLineEdit()
        gridLayout.addWidget(self.folderPathLineEdit, row, 1)
        browseButton = widgets.browseFileButton(openFolder=True)
        gridLayout.addWidget(browseButton, row, 2)
        browseButton.sigPathSelected.connect(
            partial(self.updateFolderPath, lineEdit=self.folderPathLineEdit)
        )
        
        row += 1
        label = QLabel('Channel(s) name: ')
        gridLayout.addWidget(label, row, 0)
        self.channelNamesLineEdit = widgets.alphaNumericLineEdit(
            additionalChars=' ,'
        )
        gridLayout.addWidget(self.channelNamesLineEdit, row, 1)
        infoButton = widgets.infoPushButton()
        gridLayout.addWidget(infoButton, row, 2)
        infoButton.clicked.connect(self.showInfoChannelName)
        
        
        buttonsLayout = widgets.CancelOkButtonsLayout()
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        gridLayout.setColumnStretch(0, 0)
        gridLayout.setColumnStretch(1, 1)
        gridLayout.setColumnStretch(2, 0)
        
        mainLayout.addLayout(gridLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def showInfoFileStructure(self):
        txt = html_utils.paragraph("""
            Select whether the microscopy files contains multiple "series".<br><br>
            This typically depends on how you acquired the images at the 
            microscope, i.e., you generated multiple microscopy files 
            (e.g., snapshots), or you setup automatic acquisition of multiple 
            positions.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, 'Files structure info', txt)
    
    def showInfoChannelName(self):
        txt = html_utils.paragraph("""
            Enter the channels name. Separate multiple channels with a comma.<br><br>
            The channel names will be used to name the individual TIFF files 
            (one for each channel).
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, 'Files structure info', txt)
    
    def updateFolderPath(self, path, lineEdit=''):
        lineEdit.setText(path)
    
    def warnPathEmpty(self):
        txt = html_utils.paragraph("""
            Folder path <b>cannot be empty</b>.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Empty folder path', txt)
    
    def warnSelectedPathDoesNotExist(self, path):
        txt = html_utils.paragraph("""
            The selected path <b>does not exist</b>.<br><br>
            Selected path:
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Folder path does not exist', txt, commands=(path,))
    
    def warnSelectedPathNotAFolder(self, path):
        txt = html_utils.paragraph("""
            The selected path is <b>not a folder</b>.<br><br>
            Selected path:
        """)
        msg = widgets.myMessageBox(wrapText=False)
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
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Multiple file extensions detected', txt, commands=(path,)
        )
    
    def warnChannelNamesEmpty(self):
        txt = html_utils.paragraph("""
            Channel(s) name <b>cannot be empty</b>.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Empty channel name', txt)
    
    def validate(self):
        path = self.folderPath()
        if not path:
            self.warnPathEmpty()
            return False
        
        if not os.path.exists(path):
            self.warnSelectedPathDoesNotExist(path)
            return False
        
        if not os.path.isdir(path):
            self.warnSelectedPathNotAFolder(path)
            return False
        
        files = os.listdir(path)
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
    
    def ok_cb(self):
        proceed = self.validate()
        if not proceed:
            return
        
        self.selectedFolderPath = self.folderPath()
        self.filesStructure = self.filesStructureCombobox.currentText()
        is_multiple_files = self.filesStructure.find('separated') != -1
        self.init_macro_args = (
            self.folderPath(), is_multiple_files, 
            self.channelNamesLineEdit.text().split(',')
            
        )
        self.cancel = False
        self.close()
    
class ImageJRoisToSegmManager(QBaseDialog):
    def __init__(
            self, rois_filepath, TZYX_shape, 
            addUseSamePropsForNextPosButton=False, parent=None
        ):
        import roifile
        
        self.cancel = True
        super().__init__(parent)
        
        self.setWindowTitle('ROI Manager')
        
        mainLayout = QVBoxLayout()
        
        rois = roifile.roiread(rois_filepath)
        self.rois = {roi.name: roi for roi in rois}
        
        roisNamesTreeWidget = widgets.TreeWidget()
        roisNamesTreeWidget.setHeaderLabels(['ROI name', 'Cell_ID'])
        roisNamesTreeWidget.header().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        # roisNamesTreeWidget.header().setStretchLastSection(False)
        for r, roi in enumerate(rois):
            item = widgets.TreeWidgetItem()
            item.setText(0, roi.name)
            item.setText(1, str(r+1))
            roisNamesTreeWidget.addTopLevelItem(item)
        roisNamesTreeWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        roisNamesTreeWidget.selectAll()
        mainLayout.addWidget(QLabel('Select ROIs to convert'))
        mainLayout.addWidget(roisNamesTreeWidget)
        self.roisNamesTreeWidget = roisNamesTreeWidget
        mainLayout.addSpacing(10)
        mainLayout.addWidget(widgets.QHLine())
        mainLayout.addSpacing(5)
        
        gridLayout = None
        self.lowZspinbox = None
        
        SizeT, SizeZ, SizeY, SizeX = TZYX_shape
        if SizeZ > 1:
            gridLayout = QGridLayout()
            self.lowZspinbox = widgets.SpinBox()
            self.lowZspinbox.setMinimum(0)
            self.lowZspinbox.setMaximum(SizeZ-1)
            
            self.highZspinbox = widgets.SpinBox()
            self.highZspinbox.setMinimum(0)
            self.highZspinbox.setMaximum(SizeZ-1)
            self.highZspinbox.setValue(SizeZ-1)
            
            gridLayout.addWidget(QLabel('Repeat 2D ROIs over z-range: '), 1, 0)
            
            gridLayout.addWidget(QLabel('Start z-slice'), 0, 1)
            gridLayout.addWidget(self.lowZspinbox, 1, 1)
            
            gridLayout.addWidget(QLabel('Stop z-slice'), 0, 2)
            gridLayout.addWidget(self.highZspinbox, 1, 2)
        
        if gridLayout is not None:
            mainLayout.addLayout(gridLayout)
            mainLayout.addSpacing(5)
            mainLayout.addWidget(widgets.QHLine())
            mainLayout.addSpacing(10)
        
        self.rescaleRoisGroupbox = widgets.RescaleImageJroisGroupbox(TZYX_shape)
        self.rescaleRoisGroupbox.setChecked(False)
        mainLayout.addWidget(self.rescaleRoisGroupbox)
        
        buttonsLayout = widgets.CancelOkButtonsLayout()
        
        self.useSamePropsForNextPos = False
        if addUseSamePropsForNextPosButton:
            useSamePropsForNextPosButton = widgets.reloadPushButton(
                'Keep the same preferences for all next Positions'
            )
            buttonsLayout.insertWidget(3, useSamePropsForNextPosButton)
            useSamePropsForNextPosButton.clicked.connect(
                self.useSamePropsForNextPosClicked
            )

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def useSamePropsForNextPosClicked(self):
        self.useSamePropsForNextPos = True
        self.ok_cb()
    
    def warnRoiSelectionEmpty(self):
        txt = html_utils.paragraph(f"""
            You did not select any ROI.<br><br>
            <b>ROIs selection cannot be empty</b>. Thank you for your patience! 
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'ROIs selection empty', txt)
    
    def ok_cb(self):
        selectedRois = self.roisNamesTreeWidget.selectedItems()
        if not selectedRois:
            self.useSamePropsForNextPos = False
            self.warnRoiSelectionEmpty()
            return
        
        self.IDsToRoisMapper = {}
        for item in selectedRois:
            roiName = item.text(0)
            ID = int(item.text(1))
            self.IDsToRoisMapper[ID] = self.rois[roiName]
        
        numRois = self.roisNamesTreeWidget.topLevelItemCount()
        self.areAllRoisSelected = len(self.IDsToRoisMapper) == numRois
        
        self.rescaleSizes = self.rescaleRoisGroupbox.inputOutputSizes()
        self.repeatRoisZslicesRange = None
        if self.lowZspinbox is not None:
            self.repeatRoisZslicesRange = (
                self.lowZspinbox.value(), self.highZspinbox.value()+1
            )
        
        self.cancel = False
        self.close()

class ResizeUtilProps(QBaseDialog):
    def __init__(self, input_path='', parent=None):        
        self.cancel = True
        super().__init__(parent)
        
        self.setWindowTitle('Resize Data Properties')
        
        mainLayout = QVBoxLayout()
        
        paramsLayout = QGridLayout()
        
        self._input_path = input_path
        
        row = 0
        paramsLayout.addWidget(QLabel('Overwrite raw data: '), row, 0)
        self.overwriteToggle = widgets.Toggle()
        self.overwriteToggle.setChecked(True)
        paramsLayout.addWidget(
            self.overwriteToggle, row, 1, 1, 2, alignment=Qt.AlignCenter
        )
        
        row += 1
        paramsLayout.addWidget(
            QLabel('Folder path for resized images: '), row, 0
        )
        self.filepathOutControl = widgets.filePathControl(
            browseFolder=True, 
            fileManagerTitle='Select folder where to save resized data', 
            elide=True, 
            startFolder=myutils.getMostRecentPath()
        )
        self.filepathOutControl.setDisabled(True)
        paramsLayout.addWidget(self.filepathOutControl, row, 1, 1, 2)
        
        row += 1
        paramsLayout.addWidget(QLabel('Text to append to files: '), row, 0)
        self.textToAppendLineEdit = widgets.alphaNumericLineEdit()
        self.textToAppendLineEdit.setAlignment(Qt.AlignCenter)
        paramsLayout.addWidget(self.textToAppendLineEdit, row, 1, 1, 2)
        
        row += 1
        paramsLayout.addWidget(QLabel('Resize mode: '), row, 0)
        self.downScaleRadioButton = QRadioButton('Downscale')
        self.upScaleRadioButton = QRadioButton('Upscale')
        self.downScaleRadioButton.setChecked(True)
        paramsLayout.addWidget(
            self.downScaleRadioButton, row, 1, alignment=Qt.AlignCenter
        )
        paramsLayout.addWidget(
            self.upScaleRadioButton, row, 2, alignment=Qt.AlignCenter
        )
        
        row += 1
        paramsLayout.addWidget(QLabel('Resize factor: '), row, 0)
        self.factorSpinbox = widgets.FloatLineEdit(allowNegative=False)
        self.factorSpinbox.setMinimum(1.0)
        self.factorSpinbox.setValue(2.0)
        paramsLayout.addWidget(self.factorSpinbox, row, 1, 1, 2)
        
        paramsLayout.setColumnStretch(0, 0)
        paramsLayout.setVerticalSpacing(10)
        
        self.overwriteToggle.toggled.connect(self.overwriteToggled)
        
        buttonsLayout = widgets.CancelOkButtonsLayout()
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addLayout(paramsLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addStretch(1)
        
        self.textToAppendLineEdit.setText(self._getDefaultTextToAppend())
        
        self.setLayout(mainLayout)
    
    def _getDefaultTextToAppend(self):
        rescale_mode = 'up' if self.upScaleRadioButton.isChecked() else 'down'
        factor = self.factorSpinbox.value()
        text = f'{rescale_mode}scaled_factor_{factor}'
        return text
    
    def overwriteToggled(self, checked):
        self.filepathOutControl.setDisabled(checked)
    
    def warnFolderPathEmpty(self):
        txt = html_utils.paragraph("""
            To prevent overwriting raw data the <code>Folder path for 
            resized images</code> <b>cannot be empty</b>.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Empty folder path', txt)
    
    def warnTextToAppendEmpty(self):
        txt = html_utils.paragraph("""
            To prevent overwriting raw data the <b>text to append 
            cannot be empty</b>.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Empty text to append', txt)
    
    def ok_cb(self):
        self.expFolderpathOut = self.filepathOutControl.path()
        self.textToAppend = self.textToAppendLineEdit.text()
        isAccidentalOverwrite = (
            not self.overwriteToggle.isChecked()
            and self.expFolderpathOut == self._input_path
            and not self.textToAppend
        )
        if isAccidentalOverwrite:
            self.warnTextToAppendEmpty()
            return
        
        if not self.textToAppend.startswith('_'):
            self.textToAppend = f'_{self.textToAppend}'
            
        if self.overwriteToggle.isChecked():
            self.expFolderpathOut = None
        
        factor = self.factorSpinbox.value()
        self.resizeFactor = (
            factor if self.upScaleRadioButton.isChecked() else 1/factor
        )
        
        self.cancel = False
        self.close()

class LogoDialog(QDialog):
    def __init__(self, logo_path, icon_path, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout()
        
        self.setWindowFlags(Qt.FramelessWindowHint)
        # self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint)
        # self.setAttribute(Qt.WA_TranslucentBackground)
        # self.setWindowIcon(QIcon(icon_path))
        
        labelLogo = QLabel()
        pixmapLogo = QPixmap(logo_path)
        labelLogo.setPixmap(pixmapLogo)
        
        layout.addWidget(labelLogo)
    
        self.setLayout(layout)

class SetCustomLevelsLut(QBaseDialog):
    sigLevelsChanged = Signal(object)
    
    def __init__(
            self, init_min_value=None, 
            init_max_value=None, 
            maximum_max_value=None, 
            parent=None
        ):
        super().__init__(parent=parent)
        
        self.cancel = True
        
        self.setWindowTitle('Custom LUT levels')
        
        layout = QVBoxLayout()
        
        self.minLevelSlider = widgets.sliderWithSpinBox(
            title='Mimimum', 
            title_loc='top', 
        )
        self.minLevelSlider.setMinimum(0)
        if init_max_value is not None:
            self.minLevelSlider.setMaximum(init_max_value)

        if maximum_max_value is not None:
            self.minLevelSlider.setMaximum(maximum_max_value)
            
        if init_min_value is not None:
            self.minLevelSlider.setValue(init_min_value)
        layout.addWidget(self.minLevelSlider)
        
        self.maxLevelSlider = widgets.sliderWithSpinBox(
            title='Maximum', 
            title_loc='top', 
        )
        self.maxLevelSlider.setMinimum(0)
        if init_max_value is not None:
            self.maxLevelSlider.setValue(init_max_value)
        
        if maximum_max_value is not None:
            self.maxLevelSlider.setMaximum(maximum_max_value)
            
        layout.addWidget(self.maxLevelSlider)
        
        self.maxLevelSlider.sigValueChange.connect(self.emitLevelsChanged)
        self.maxLevelSlider.sigValueChange.connect(self.emitLevelsChanged)
        
        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)
        
        self.setLayout(layout)
    
    def sizeHint(self):
        heightHint = super().sizeHint().height()
        widthHint = super().sizeHint().width()*2
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
        