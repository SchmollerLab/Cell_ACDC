from cgitb import html
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
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import math
import time

import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from PyQt5 import QtCore
from PyQt5.QtGui import (
    QIcon, QFontMetrics, QKeySequence, QFont, QGuiApplication, QCursor,
    QKeyEvent  
)
from PyQt5.QtCore import Qt, QSize, QEvent, pyqtSignal, QEventLoop, QTimer
from PyQt5.QtWidgets import (
    QAction, QApplication, QMainWindow, QMenu, QLabel, QToolBar,
    QScrollBar, QWidget, QVBoxLayout, QLineEdit, QPushButton,
    QHBoxLayout, QDialog, QFormLayout, QListWidget, QAbstractItemView,
    QButtonGroup, QCheckBox, QSizePolicy, QComboBox, QSlider, QGridLayout,
    QSpinBox, QToolButton, QTableView, QTextBrowser, QDoubleSpinBox,
    QScrollArea, QFrame, QProgressBar, QGroupBox, QRadioButton,
    QDockWidget, QMessageBox, QStyle, QPlainTextEdit, QSpacerItem,
    QTreeWidget, QTreeWidgetItem, QTextEdit
)

from . import myutils, load, prompts, widgets, core, measurements, html_utils
from . import is_mac, is_win, is_linux, temp_path, config
from . import qrc_resources, printl
from . import colors
from . import issues_url

pg.setConfigOption('imageAxisOrder', 'row-major') # best performance
font = QtGui.QFont()
font.setPixelSize(13)

class QBaseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

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

    def addInstructionsLinux(self):
        self.scrollArea = QScrollArea()
        _container = QWidget()
        _layout = QVBoxLayout()
        for t, text in enumerate(myutils.install_javabridge_instructions_text()):
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
                    copyButton.textToCopy = myutils._apt_update_command()
                elif t==2:
                    copyButton.textToCopy = myutils._apt_install_java_command()
                elif t==3:
                    copyButton.textToCopy = myutils._apt_gcc_command()
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

    def installJava(self):
        import subprocess
        try:
            if is_mac:
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
            elif is_linux:
                subprocess.run(
                    myutils._apt_gcc_command()(),
                    check=True, text=True, shell=True
                )
                subprocess.run(
                    myutils._apt_update_command()(),
                    check=True, text=True, shell=True
                )
                subprocess.run(
                    myutils._apt_install_java_command()(),
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

class customAnnotationDialog(QDialog):
    sigDeleteSelecAnnot = pyqtSignal(object)

    def __init__(self, savedCustomAnnot, parent=None, state=None):
        self.cancel = True
        self.loop = None
        self.clickedButton = None
        self.savedCustomAnnot = savedCustomAnnot

        self.internalNames = measurements.get_all_acdc_df_colnames()

        super().__init__(parent)

        self.setWindowTitle('Custom annotation')
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        layout = widgets.myFormLayout()

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
            widgets.shortCutLineEdit(), addInfoButton=True,
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

        self.loadSavedAnnotButton = QPushButton('Load annotation...')
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
        self.okButton.setFocus(True)

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
        self.selectAnnotWin = QDialogListbox(
            'Load annotation parameters',
            'Select annotation to load:', items,
            additionalButtons=('Delete selected annnotations', ),
            parent=self
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
        keySequence = widgets.macShortcutToQKeySequence(selectedAnnot['shortcut'])
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
        self.sigDeleteSelecAnnot.emit(self.selectAnnotWin.listBox.selectedItems())
        for item in self.selectAnnotWin.listBox.selectedItems():
            name = item.text()
            self.savedCustomAnnot.pop(name)
        items = list(self.savedCustomAnnot.keys())
        self.selectAnnotWin.listBox.clear()
        self.selectAnnotWin.listBox.addItems(items)

    def selectColor(self):
        pg.ColorButton.selectColor(self.colorButton)
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
        # listView.setSelectionMode(QAbstractItemView.NoSelection)
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
            listView.setSelectionMode(QAbstractItemView.NoSelection)
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
        self.symbol = re.findall(r"\'(\w+)\'", symbol)[0]

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

class filenameDialog(QDialog):
    def __init__(
            self, ext='.npz', basename='', title='Insert file name',
            hintText='', existingNames='', parent=None, allowEmpty=True,
            helpText=''
        ):
        self.cancel = True
        super().__init__(parent)

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
        self.existingNames = []
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
        buttonsLayout.addWidget(okButton)

        cancelButton.clicked.connect(self.close)
        okButton.clicked.connect(self.ok_cb)
        self.lineEdit.textChanged.connect(self.updateFilename)
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
    
    def showHelp(self, text):
        text = html_utils.paragraph(text)
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, 'Filename help', text)

    def _text(self):
        return self.lineEdit.text().replace(' ', '_')

    def checkExistingNames(self):
        if self._text() not in self.existingNames:
            return True

        filename = self.filenameLabel.text()
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(
            f'The file <code>{filename}</code> is <b>already existing</b>.<br><br>'
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
            self.filenameLabel.setText(f'{self.basename}_{text}{self.ext}')

    def ok_cb(self, checked=True):
        valid = self.checkExistingNames()
        if not valid:
            return
        
        if not self.allowEmpty and not self._text():
            msg = widgets.myMessageBox()
            msg.warning(
                self, 'Empty text', 
                html_utils.paragraph('Text entry field <b>cannot be empty</b>')
            )
            return
            
        self.filename = self.filenameLabel.text()
        self.entryText = self._text()
        self.cancel = False
        self.close()

    def closeEvent(self, event):
        valid = self.checkExistingNames()
        if not valid:
            event.ignore()
            return

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        super().show()
        self.okButton.setDefault(True)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, 'loop'):
            self.loop.exit()

class wandToleranceWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.slider = widgets.sliderWithSpinBox(title='Tolerance')
        self.slider.setMaximum(255)
        self.slider.layout.setColumnStretch(2, 21)

        self.setLayout(self.slider.layout)

class setMeasurementsDialog(QBaseDialog):
    sigClosed = pyqtSignal()

    def __init__(
            self, loadedChNames, notLoadedChNames, isZstack, isSegm3D,
            favourite_funcs=None, parent=None, acdc_df=None,
            acdc_df_path=None, posData=None, addCombineMetricCallback=None
        ):
        super().__init__(parent=parent)

        self.cancel = True

        self.delExistingCols = False
        self.okClicked = False
        self.acdc_df = acdc_df
        self.acdc_df_path = acdc_df_path

        self.setWindowTitle('Set measurements')
        # self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)

        layout = QVBoxLayout()
        groupsLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        self.chNameGroupboxes = []

        col = 0
        for col, chName in enumerate(loadedChNames):
            channelGBox = widgets.channelMetricsQGBox(
                isZstack, chName, isSegm3D, favourite_funcs=favourite_funcs,
                posData=posData
            )
            channelGBox.chName = chName
            groupsLayout.addWidget(channelGBox, 0, col, 3, 1)
            self.chNameGroupboxes.append(channelGBox)
            groupsLayout.setColumnStretch(col, 5)

        current_col = col+1
        for col, chName in enumerate(notLoadedChNames):
            channelGBox = widgets.channelMetricsQGBox(
                isZstack, chName, isSegm3D, favourite_funcs=favourite_funcs,
                posData=posData
            )
            channelGBox.setChecked(False)
            channelGBox.chName = chName
            groupsLayout.addWidget(channelGBox, 0, current_col, 3, 1)
            self.chNameGroupboxes.append(channelGBox)
            groupsLayout.setColumnStretch(current_col, 5)
            current_col += 1

        current_col += 1

        size_metrics_desc = measurements.get_size_metrics_desc()
        if not isSegm3D:
            size_metrics_desc = {
                key:val for key,val in size_metrics_desc.items()
                if not key.endswith('_3D')
            }
        sizeMetricsQGBox = widgets._metricsQGBox(
            size_metrics_desc, 'Size metrics',
            favourite_funcs=favourite_funcs, isZstack=isZstack
        )
        self.sizeMetricsQGBox = sizeMetricsQGBox
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
        groupsLayout.addWidget(regionPropsQGBox, 1, current_col)
        groupsLayout.setRowStretch(1, 2)

        desc = measurements.get_user_combine_mixed_channels_desc(
            isSegm3D=isSegm3D
        )
        self.mixedChannelsCombineMetricsQGBox = None
        if desc:
            mixedChannelsCombineMetricsQGBox = widgets._metricsQGBox(
                desc, 'Mixed channels combined measurements',
                favourite_funcs=favourite_funcs, isZstack=isZstack
            )
            self.mixedChannelsCombineMetricsQGBox = mixedChannelsCombineMetricsQGBox
            groupsLayout.addWidget(
                mixedChannelsCombineMetricsQGBox, 2, current_col
            )
            groupsLayout.setRowStretch(1, 1)

        self.numberCols = current_col

        okButton = widgets.okPushButton('   Ok   ')
        cancelButton = widgets.cancelPushButton('Cancel')
        if addCombineMetricCallback is not None:
            addCombineMetricButton = widgets.addPushButton(
                'Add combined measurement...'
            )
            addCombineMetricButton.clicked.connect(addCombineMetricCallback)
        self.okButton = okButton

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        if addCombineMetricCallback is not None:
            buttonsLayout.addWidget(addCombineMetricButton)
        buttonsLayout.addWidget(okButton)

        layout.addLayout(groupsLayout)
        layout.addLayout(buttonsLayout)

        self.setLayout(layout)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

    def ok_cb(self):
        if self.acdc_df is None:
            self.cancel = False
            self.close()
            self.sigClosed.emit()
            return

        self.okClicked = True
        existing_colnames = list(self.acdc_df.columns)
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
            is_existing = any([col.find(colname) !=-1 for col in existing_colnames])
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
        screenWidth = self.screen().size().width()
        screenHeight = self.screen().size().height()
        h = screenHeight-200
        minColWith = screenWidth/5
        w = minColWith*self.numberCols
        xLeft = int((screenWidth-w)/2)
        self.move(xLeft, 50)
        self.resize(int(w), h)
        super().show(block=block)

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
        font.setPixelSize(13)
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
        txt = 'Voxel depth:  '
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
        if posData.SizeT > 1:
            self.imageViewer.framesScrollBar.setDisabled(True)
            self.imageViewer.framesScrollBar.setVisible(False)
            self.imageViewer.frameLabel.hide()
            self.imageViewer.t_label.hide()
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

class CellACDCTrackerParamsWin(QDialog):
    def __init__(self, parent=None):
        self.cancel = True
        super().__init__(parent)

        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
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
    def __init__(self, segmShape, parent=None):
        self.cancel = True
        super().__init__(parent)

        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
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
            'optimize': self.optimizeToggle.isChecked()
        }
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

class QDialogWorkerProgress(QDialog):
    sigClosed = pyqtSignal(bool)

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
        width = width if self.width() < width else self.width()
        height = int(screenHeight/3)
        left = int(mainWinCenterX - width/2)
        left = left if left >= 0 else 0
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
        _font.setPixelSize(13)
        label.setFont(_font)
        # padding: top, left, bottom, right
        label.setStyleSheet("padding:0px 0px 3px 0px;")
        topLayout.addWidget(label, alignment=Qt.AlignCenter)

        listBox = widgets.listWidget()
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

        if cancelText.lower().find('cancel') != -1:
            cancelButton = widgets.cancelPushButton(cancelText)
        else:
            cancelButton = QPushButton(cancelText)
        okButton = widgets.okPushButton(' Ok ')
        okButton.setShortcut(Qt.Key_Enter)

        bottomLayout.addStretch(1)
        bottomLayout.addWidget(cancelButton)
        bottomLayout.addSpacing(20)

        if additionalButtons:
            self._additionalButtons = []
            for button in additionalButtons:
                _button = QPushButton(button)
                self._additionalButtons.append(_button)
                bottomLayout.addWidget(_button)
                _button.clicked.connect(self.ok_cb)

        bottomLayout.addWidget(okButton)
        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        self.setStyleSheet("""
            QListWidget::item:hover {background-color:#E6E6E6;}
            QListWidget::item:selected {background-color:#CFEB9B;}
            QListWidget::item:selected {color:black;}
            QListView {
                selection-background-color: #CFEB9B;
                selection-color: white;
                show-decoration-selected: 1;
            }
        """)

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

class QDialogSelectModel(QDialog):
    def __init__(self, parent=None):
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
        models.append('Custom model...')
        listBox.setFont(font)
        listBox.addItems(models)
        listBox.setSelectionMode(QAbstractItemView.SingleSelection)
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
        bottomLayout.addWidget(okButton)
        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        self.setStyleSheet("""
            QListWidget::item:hover {background-color:#E6E6E6;}
            QListWidget::item:selected {background-color:#CFEB9B;}
            QListWidget::item:selected {color:black;}
            QListView {
                selection-background-color: #CFEB9B;
                selection-color: white;
                show-decoration-selected: 1;
            }
        """)

    def ok_cb(self, event):
        self.clickedButton = self.sender()
        self.cancel = False
        item = self.listBox.currentItem()
        model = item.text()
        if model == 'Custom model...':
            txt, models_path = myutils.get_add_custom_model_instructions()
            msg = widgets.myMessageBox(showCentered=False)
            msg.addShowInFileManagerButton(models_path, txt='Open models folder...')
            msg.information(
                self, 'Custom model instructions', txt, buttonsTexts=('Ok',)
            )
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

        self.selectFramesGroupbox = widgets.selectStartStopFrames(
            SizeT, currentFrameNum=currentFrameNo, parent=parent
        )

        self.mainLayout.insertWidget(1, self.selectFramesGroupbox)

    def ok_cb(self, event):
        if self.selectFramesGroupbox.warningLabel.text():
            return
        else:
            self.startFrame = self.selectFramesGroupbox.startFrame_SB.value()
            self.stopFrame = self.selectFramesGroupbox.stopFrame_SB.value()
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
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
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
            forceEnableAskSegm3D=False
        ):
        self.cancel = True
        self.ask_TimeIncrement = ask_TimeIncrement
        self.ask_PhysicalSizes = ask_PhysicalSizes
        self.askSegm3D = askSegm3D
        self.imgDataShape = imgDataShape
        self.posData = posData
        self._additionalValues = additionalValues
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
        self.isSegm3DLabel = QLabel('3D segmentation (z-stacks)')
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
        self.isSegm3D = self.isSegm3Dtoggle.isChecked()

        self.TimeIncrement = self.TimeIncrementSpinBox.value()
        self.PhysicalSizeX = self.PhysicalSizeXSpinBox.value()
        self.PhysicalSizeY = self.PhysicalSizeYSpinBox.value()
        self.PhysicalSizeZ = self.PhysicalSizeZSpinBox.value()
        self._additionalValues = {
            f"__{field['nameWidget'].text()}":field['valueWidget'].text()
            for field in self.additionalFieldsWidgets
        }
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
            <p style="font-size:13px">
                You loaded <b>4D data</b>, hence the number of frames MUST be
                <b>{T}</b><br> nd the number of z-slices MUST be <b>{Z}</b>.<br><br>
                What do you want to do?
            </p>
            """)
        if not valid3D:
            txt = (f"""
            <p style="font-size:13px">
                You loaded <b>3D data</b>, hence either the number of frames is
                <b>{TZ}</b><br> or the number of z-slices can be <b>{TZ}</b>.<br><br>
                However, if the number of frames is greater than 1 then the<br>
                number of z-slices MUST be 1, and vice-versa.<br><br>
                What do you want to do?
            </p>
            """)

        if not valid2D:
            txt = (f"""
            <p style="font-size:13px">
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
            continueButton = widgets.okPushButton(
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

class QCropZtool(QBaseDialog):
    sigClose = pyqtSignal()
    sigZvalueChanged = pyqtSignal(str, int)
    sigReset = pyqtSignal()
    sigCrop = pyqtSignal()

    def __init__(
            self, SizeZ, cropButtonText='Crop and save', parent=None, 
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
        self.bkgrThreshSlider.setTickPosition(QSlider.TicksBelow)
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
        font.setPixelSize(13)
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
        _font.setPixelSize(13)
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
        _font.setPixelSize(13)
        infotxtLabel.setFont(_font)

        infotxtLabel.setStyleSheet("padding:0px 0px 3px 0px;")
        txtLayout.addWidget(infotxtLabel, alignment=Qt.AlignCenter)

        noteTxt = (
            'NOTE: Only changes applied to current frame can be undone.\n'
            '      Changes applied to future frames CANNOT be UNDONE!\n'
        )

        noteTxtLabel = QLabel(noteTxt)
        _font = QtGui.QFont()
        _font.setPixelSize(13)
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

class ComputeMetricsErrorsDialog(QBaseDialog):
    def __init__(
            self, customMetricsErrors, log_path='', parent=None, 
            log_type='custom_metrics'
        ):
        super().__init__(parent)

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
        else:
            infoText = ("""
                <b>Standard metrics</b> were <b>NOT saved</b> because Cell-ACDC 
                encoutered the following errors.<br><br>
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
        for func_name, traceback_format in customMetricsErrors.items():
            nameLabel = QLabel(f'<b>{func_name}</b>: ')
            errorMessage = f'\n{traceback_format}'
            errorLabel = QLabel(errorMessage)
            errorLabel.setTextInteractionFlags(
                Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard
            )
            errorLabel.setStyleSheet("background-color: white")
            errorLabel.setFrameShape(QFrame.Panel)
            errorLabel.setFrameShadow(QFrame.Sunken)
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
        
        okButton = widgets.okPushButton(' Ok ')
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(okButton)

        showLogButton.clicked.connect(partial(myutils.showInExplorer, log_path))
        okButton.clicked.connect(self.close)
        layout.setVerticalSpacing(10)
        layout.addLayout(buttonsLayout, 2, 1)

        self.setLayout(layout)
        self.setFont(font)

class postProcessSegmParams(QGroupBox):
    def __init__(self, title, useSliders=False, parent=None, maxSize=None):
        QGroupBox.__init__(self, title, parent)
        self.useSliders = useSliders

        layout = QGridLayout()

        row = 0
        label = QLabel("Minimum area (pixels) ")
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

        row += 1
        label = QLabel("Minimum solidity (0-1) ")
        layout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        if useSliders:
            minSolidity_DSB = widgets.sliderWithSpinBox(normalize=True)
            minSolidity_DSB.setMaximum(100)
        else:
            minSolidity_DSB = QDoubleSpinBox()
            minSolidity_DSB.setAlignment(Qt.AlignCenter)
            minSolidity_DSB.setMinimum(0)
            minSolidity_DSB.setMaximum(1)
        minSolidity_DSB.setValue(0.5)
        minSolidity_DSB.setSingleStep(0.1)

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

        layout.setColumnStretch(1, 2)

        self.setLayout(layout)
    
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

class postProcessSegmDialog(QBaseDialog):
    sigClosed = pyqtSignal()

    def __init__(self, mainWin=None, useSliders=True):
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

        artefactsGroupBox = postProcessSegmParams(
            'Post-processing parameters', useSliders=useSliders
        )

        self.minSize_SB = artefactsGroupBox.minSize_SB
        self.minSolidity_DSB = artefactsGroupBox.minSolidity_DSB
        self.maxElongation_DSB = artefactsGroupBox.maxElongation_DSB

        self.minSize_SB.valueChanged.connect(self.valueChanged)
        self.minSolidity_DSB.valueChanged.connect(self.valueChanged)
        self.maxElongation_DSB.valueChanged.connect(self.valueChanged)

        self.minSize_SB.editingFinished.connect(self.onEditingFinished)
        self.minSolidity_DSB.editingFinished.connect(self.onEditingFinished)
        self.maxElongation_DSB.editingFinished.connect(self.onEditingFinished)

        if self.isTimelapse:
            applyAllButton = QPushButton('Apply to all frames...')
            applyAllButton.clicked.connect(self.applyAll_cb)
            applyButton = QPushButton('Apply')
            applyButton.clicked.connect(self.apply_cb)
        elif self.isMultiPos:
            applyAllButton = QPushButton('Apply to all Positions...')
            applyAllButton.clicked.connect(self.applyAll_cb)
            applyButton = QPushButton('Apply')
            applyButton.clicked.connect(self.apply_cb)
        else:
            applyAllButton = QPushButton('Apply')
            applyAllButton.clicked.connect(self.ok_cb)
            applyButton = None

        cancelButton = widgets.cancelPushButton('Cancel')

        buttonsLayout.addStretch(1)
        if applyButton is not None:
            buttonsLayout.addWidget(applyButton)
        buttonsLayout.addWidget(applyAllButton)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.setContentsMargins(0,10,0,0)

        mainLayout.addWidget(artefactsGroupBox)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        cancelButton.clicked.connect(self.cancel_cb)

        if mainWin is not None:
            self.setPosData()
            self.apply_cb()

    def setPosData(self):
        if self.mainWin is None:
            return

        self.mainWin.storeUndoRedoStates(False)
        self.posData = self.mainWin.data[self.mainWin.pos_i]
        self.origLab = self.posData.lab.copy()

    def valueChanged(self, value):
        lab, delIDs = self.apply()
        self.mainWin.clearOverlaidMasks(delIDs)
        self.posData.lab = lab   
        self.mainWin.clearItems_IDs(delIDs)
        self.mainWin.setImageImg2()
        self.mainWin.clearOverlaidMasks(delIDs)

    def apply(self, origLab=None):
        if self.mainWin is None:
            return

        minSize = self.minSize_SB.value()
        minSolidity = self.minSolidity_DSB.value()
        maxElongation = self.maxElongation_DSB.value()

        self.mainWin.warnEditingWithCca_df('post-processing segmentation mask')

        if origLab is None:
            origLab = self.origLab.copy()

        lab, delIDs = core.remove_artefacts(
            origLab,
            min_solidity=minSolidity,
            min_area=minSize,
            max_elongation=maxElongation,
            return_delIDs=True
        )

        return lab, delIDs

    def onEditingFinished(self):
        if self.mainWin is None:
            return

        self.mainWin.update_rp()
        self.mainWin.store_data()
        self.mainWin.updateALLimg()

    def ok_cb(self):
        self.apply()
        self.onEditingFinished()
        self.close()

    def apply_cb(self):
        self.apply()
        self.onEditingFinished()

    def applyAll_cb(self):
        if self.mainWin is None:
            return

        if self.isTimelapse:
            current_frame_i = self.posData.frame_i

            self.origSegmData = self.posData.segm_data.copy()

            # Apply to all future frames or future positions
            for frame_i in range(self.posData.segmSizeT):
                self.posData.frame_i = frame_i
                lab = self.posData.allData_li[frame_i]['labels']
                if lab is None:
                    # Non-visited frame modify segm_data
                    origLab = self.posData.segm_data[frame_i].copy()
                    lab, delIDs = self.apply(origLab=origLab)
                    self.posData.segm_data[frame_i] = lab
                else:
                    self.mainWin.get_data()
                    origLab = self.posData.lab.copy()
                    self.origSegmData[frame_i] = origLab
                    lab, delIDs = self.apply(origLab=origLab)
                    self.posData.lab = lab
                    self.posData.allData_li[frame_i]['labels'] = lab.copy()
                    # Get the rest of the stored metadata based on the new lab
                    self.mainWin.get_data()
                    self.mainWin.store_data()

            # Back to current frame
            self.posData.frame_i = current_frame_i
            self.mainWin.get_data()
            self.mainWin.updateALLimg()

            msg = QMessageBox()
            msg.information(
                self, 'Done', 'Post-processing applied to all frames!'
            )

        elif self.isMultiPos:
            self.origSegmData = []
            current_pos_i = self.mainWin.pos_i
            # Apply to all future frames or future positions
            for pos_i, posData in enumerate(self.mainWin.data):
                self.mainWin.pos_i = pos_i
                self.mainWin.get_data()
                origLab = posData.lab.copy()
                self.origSegmData.append(origLab)
                lab, delIDs = self.apply(origLab=origLab)

                self.posData.allData_li[0]['labels'] = lab.copy()
                # Get the rest of the stored metadata based on the new lab
                self.mainWin.get_data()
                self.mainWin.store_data()

            # Back to current pos and current frame
            self.mainWin.pos_i = current_pos_i
            self.mainWin.get_data()
            self.mainWin.updateALLimg()

    def cancel_cb(self):
        if self.mainWin is not None:
            self.posData.lab = self.origLab
            self.mainWin.update_rp()
            self.mainWin.updateALLimg()

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
                self.mainWin.updateALLimg()
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
                self.mainWin.updateALLimg()

        self.close()

    def show(self, block=False):
        # self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show(block=block)
        self.resize(int(self.width()*1.5), self.height())

    def closeEvent(self, event):
        self.sigClosed.emit()
        super().closeEvent(event)

class imageViewer(QMainWindow):
    """Main Window."""

    def __init__(
            self, parent=None, posData=None, button_toUncheck=None,
            spinBox=None, linkWindow=None, enableOverlay=False,
        ):
        self.button_toUncheck = button_toUncheck
        self.parent = parent
        self.posData = posData
        self.spinBox = spinBox
        self.linkWindow = linkWindow
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

        editToolBar.addAction(self.prevAction)
        editToolBar.addAction(self.nextAction)
        editToolBar.addAction(self.jumpBackwardAction)
        editToolBar.addAction(self.jumpForwardAction)

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
            self.overlayButton.toggled.connect(self.update_img)
            self.overlayButton.sigRightClick.connect(self.showOverlayContextMenu)

    def showOverlayContextMenu(self, event):
        if not self.overlayButton.isChecked():
            return

        if self.parent is not None:
            self.parent.overlayContextMenu.exec_(QCursor.pos())

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
        hist = widgets.myHistogramLUTitem()
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
        _font.setPixelSize(13)
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
        _font = QtGui.QFont()
        _font.setPixelSize(13)
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
            if self.overlayButton.isChecked():
                img = self.getOverlayImg()
            else:
                img = self.parent.getImage(frame_i=self.frame_i)
                img = self.parent.getImageWithCmap(img=img)
        self.img.setImage(img)
        self.framesScrollBar.setSliderPosition(self.frame_i+1)

    def getOverlayImg(self):
        try:
            img = self.parent.getOverlayImg(
                setImg=False, frame_i=self.frame_i
            )
        except AttributeError:
            success = self.parent.askSelectOverlayChannel()
            if not success:
                self.overlayButton.toggled.disconnect()
                self.overlayButton.setChecked(False)
                self.overlayButton.toggled.connect(self.update_img)
                img = self.parent.getImage(frame_i=self.frame_i)
                img = self.parent.getImageWithCmap(img=img)
            else:
                self.parent.setCheckedOverlayContextMenusAction()
                img = self.parent.getOverlayImg(
                    setImg=False, frame_i=self.frame_i
                )
        return img

    def closeEvent(self, event):
        if self.button_toUncheck is not None:
            self.button_toUncheck.setChecked(False)

    def show(self, left=None, top=None):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        QMainWindow.show(self)
        if left is not None and top is not None:
            self.setGeometry(left, top, 850, 800)

class selectPositionsMultiExp(QBaseDialog):
    def __init__(self, expPaths: dict, infoPaths: dict=None, parent=None):
        super().__init__(parent=parent)

        self.expPaths = expPaths
        self.cancel = True

        mainLayout = QVBoxLayout()

        self.setWindowTitle('Select Positions to process')

        infoTxt = html_utils.paragraph(
            'Select one or more Positions to process<br><br>'
            '<code>Ctrl+Click</code> <i>to select multiple items</i><br>'
            '<code>Shift+Click</code> <i>to select a range of items</i><br>',
            center=True
        )
        infoLabel = QLabel(infoTxt)

        self.treeWidget = QTreeWidget()
        self.treeWidget.setSelectionMode(QAbstractItemView.ExtendedSelection)
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

        self.setStyleSheet("""
            QTreeWidget::item:hover {background-color:#E6E6E6;}
            QTreeWidget::item:selected {background-color:#CFEB9B;}
            QTreeWidget::item:selected {color:black;}
            QTreeView {
                selection-background-color: #CFEB9B;
                selection-color: white;
                show-decoration-selected: 1;
            }
        """)

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
        relIDLabel = QLabel('Relative ID')
        relIDLabel.setAlignment(Qt.AlignCenter)
        tableLayout.addWidget(relIDLabel, 0, col, alignment=AC)

        col += 1
        genNumLabel = QLabel('Generation number')
        genNumLabel.setAlignment(Qt.AlignCenter)
        tableLayout.addWidget(genNumLabel, 0, col, alignment=AC)
        genNumColWidth = genNumLabel.sizeHint().width()

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
        okButton = widgets.okPushButton('Ok')
        okButton.setShortcut(Qt.Key_Enter)

        cancelButton = widgets.cancelPushButton('Cancel')

        moreInfoButton = QPushButton('More info...')
        moreInfoButton.setIcon(QIcon(':info.svg'))

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(moreInfoButton)
        buttonsLayout.addWidget(okButton)

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
        moreInfoButton.clicked.connect(self.moreInfo)

        # self.setModal(True)

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
        _font.setPixelSize(13)
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
            if posData.segmSizeT == 1:
                spinBox.setValue(posData.SizeT)
            else:
                if self.concat_segm and posData.segmSizeT < posData.SizeT:
                    spinBox.setMinimum(posData.segmSizeT+1)
                    spinBox.setValue(posData.SizeT)
                else:
                    spinBox.setValue(posData.segmSizeT)
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
        focusSpinbox.setFocus(True)

    def saveSegmSizeT(self):
        for spinBox, posData in self.dataDict.values():
            posData.segmSizeT = spinBox.value()
            posData.metadata_df.at['segmSizeT', 'values'] = posData.segmSizeT
            posData.metadataToCsv()

    def ok_cb(self, event):
        self.cancel = False
        self.saveSegmSizeT()
        self.close()

    def visualize_cb(self, checked=True):
        spinBox, posData = self.dataDict[self.sender()]
        print('Loading image data...')
        posData.loadImgData()
        posData.frame_i = spinBox.value()-1
        self.slideshowWin = imageViewer(
            posData=posData, spinBox=spinBox
        )
        self.slideshowWin.update_img()
        # self.slideshowWin.framesScrollBar.setDisabled(True)
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
            warnLastFrame=False, isInteger=False, isFloat=False,
            stretchEntry=True
        ):
        QDialog.__init__(self, parent)

        self.loop = None
        self.cancel = True
        self.allowedValues = allowedValues
        self.warnLastFrame = warnLastFrame
        self.isFloat = isFloat
        self.isInteger = isInteger
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
        _font.setPixelSize(13)
        msg.setFont(_font)
        msg.setAlignment(Qt.AlignCenter)
        # padding: top, left, bottom, right
        msg.setStyleSheet("padding:0px 0px 3px 0px;")

        if isFloat:
            ID_QLineEdit = QDoubleSpinBox()
            if allowedValues is not None:
                _min, _max = allowedValues
                ID_QLineEdit.setMinimum(_min)
                ID_QLineEdit.setMaximum(_max)
            else:
                ID_QLineEdit.setMaximum(2**32)
            if defaultTxt:
                ID_QLineEdit.setValue(float(defaultTxt))

        elif isInteger:
            ID_QLineEdit = QSpinBox()
            if allowedValues is not None:
                _min, _max = allowedValues
                ID_QLineEdit.setMinimum(_min)
                ID_QLineEdit.setMaximum(_max)
            else:
                ID_QLineEdit.setMaximum(2147483647)
            if defaultTxt:
                ID_QLineEdit.setValue(int(defaultTxt))
        else:
            ID_QLineEdit = QLineEdit()
            ID_QLineEdit.setText(defaultTxt)
            ID_QLineEdit.textChanged[str].connect(self.ID_LineEdit_cb)
        ID_QLineEdit.setFont(_font)
        ID_QLineEdit.setAlignment(Qt.AlignCenter)

        self.ID_QLineEdit = ID_QLineEdit

        if allowedValues is not None:
            notValidLabel = QLabel()
            notValidLabel.setStyleSheet('color: red')
            notValidLabel.setFont(_font)
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
            LineEditLayout.addWidget(ID_QLineEdit)
        else:
            entryLayout = QHBoxLayout()
            entryLayout.addStretch(1)
            entryLayout.addWidget(ID_QLineEdit)
            entryLayout.addStretch(1)
            entryLayout.setStretch(1,1)
            LineEditLayout.addLayout(entryLayout)
        if allowedValues is not None:
            LineEditLayout.addWidget(notValidLabel, alignment=Qt.AlignCenter)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.insertSpacing(1, 20)
        buttonsLayout.addWidget(okButton)

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
        msg = widgets.myMessageBox()
        warn_txt = html_utils.paragraph(f"""
            WARNING: saving until a frame number below the last visited
            frame ({self.maxValue}) will result in <b>LOSS of information</b>
            about any <b>edit or annotation</b> you did <b>on frames
            {val}-{self.maxValue}.</b><br><br>
            Are you sure you want to proceed?
        """)
        msg.warning(
           self, 'WARNING: Potential loss of information', warn_txt, 
           buttonsTexts=('Cancel', 'Yes, I am sure.')
        )
        return msg.cancel

    def ok_cb(self, event):
        if self.allowedValues:
            if self.notValidLabel.text():
                return

        if self.isFloat or self.isInteger:
            val = self.ID_QLineEdit.value()
        else:
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
        _font.setPixelSize(13)
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
                warn_msg = html_utils.paragraph(
                    f'ID {ID} is <b>already existing</b>.<br><br>'
                    f'If you continue, ID {ID} will be swapped with '
                    f'ID {self.clickedID}<br><br>'
                    'Do you want to continue?'
                )
                msg = widgets.myMessageBox()
                noButton, yesButton = msg.warning(
                    self, 'Invalid entry', warn_msg, 
                    buttonsTexts=('No', 'Yes')
                )
                if yesButton == msg.clickedButton:
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
        listBox.setSelectionMode(QAbstractItemView.ExtendedSelection)
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
            self.ListBox.setFocus(True)
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
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        super().show()
        self.singleSelectionHeight = self.height()
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
        self.hist = widgets.myHistogramLUTitem()

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
            labelItemID = widgets.myLabelItem()
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
        font.setPixelSize(13)
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
    def __init__(self, images_ls, parent_path, parent=None):
        self.cancel = True
        self.selectedItemText = ''
        self.selectedItemIdx = None
        self.removeOthers = False
        self.okAllPos = False
        self.images_ls = images_ls
        self.parent_path = parent_path
        super().__init__(parent)

        informativeText = html_utils.paragraph(f"""
            The loaded Position folders contains
            <b>multipe segmentation masks</b><br>
        """)

        self.setWindowTitle('Multiple segm.npz files detected')
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

        questionText = html_utils.paragraph(
            'Select which segmentation file to load:'
        )
        label = QLabel(questionText)
        listWidget = widgets.listWidget()
        listWidget.addItems(images_ls)
        listWidget.setCurrentRow(0)
        self.items = list(images_ls)
        self.listWidget = listWidget

        okButton = widgets.okPushButton(' Load selected ')
        okAndRemoveButton = QPushButton(
            'Load selected and delete the other files'
        )
        okAndRemoveButton.setIcon(QIcon(':bin.svg'))
        txt = 'Reveal in Finder...' if is_mac else 'Show in Explorer...'
        showInFileManagerButton = widgets.showInFileManagerButton(txt)
        cancelButton = widgets.cancelPushButton(' Cancel ')


        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addWidget(showInFileManagerButton)
        buttonsLayout.addSpacing(20)
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
        self.okAndRemoveButton = okAndRemoveButton

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        okAndRemoveButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)
        showInFileManagerButton.clicked.connect(self.showInFileManager)

    def showInFileManager(self, checked=True):
        myutils.showInExplorer(self.parent_path)

    def ok_cb(self, event):
        self.cancel = False
        self.selectedItemText = self.listWidget.selectedItems()[0].text()
        self.selectedItemIdx = self.items.index(self.selectedItemText)
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
            self, init_params, segment_params, model_name,
            url=None, parent=None):
        self.cancel = True
        super().__init__(parent)

        self.model_name = model_name

        self.setWindowTitle(f'{model_name} parameters')

        mainLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        loadFunc = self.loadLastSelection

        initGroupBox, self.init_argsWidgets = self.createGroupParams(
            init_params,
            'Parameters for model initialization'
        )
        initDefaultButton = widgets.reloadPushButton('Restore default')
        initLoadLastSelButton = QPushButton('Load last parameters')
        initButtonsLayout = QHBoxLayout()
        initButtonsLayout.addStretch(1)
        initButtonsLayout.addWidget(initDefaultButton)
        initButtonsLayout.addWidget(initLoadLastSelButton)
        initDefaultButton.clicked.connect(self.restoreDefaultInit)
        initLoadLastSelButton.clicked.connect(
            partial(loadFunc, f'{self.model_name}.init', self.init_argsWidgets)
        )

        segmentGroupBox, self.segment2D_argsWidgets = self.createGroupParams(
            segment_params,
            'Parameters for segmentation'
        )
        segmentDefaultButton = widgets.reloadPushButton('Restore default')
        segmentLoadLastSelButton = QPushButton('Load last parameters')
        segmentButtonsLayout = QHBoxLayout()
        segmentButtonsLayout.addStretch(1)
        segmentButtonsLayout.addWidget(segmentDefaultButton)
        segmentButtonsLayout.addWidget(segmentLoadLastSelButton)
        segmentDefaultButton.clicked.connect(self.restoreDefaultSegment)
        section = f'{self.model_name}.segment'
        segmentLoadLastSelButton.clicked.connect(
            partial(loadFunc, section, self.segment2D_argsWidgets)
        )

        cancelButton = widgets.cancelPushButton(' Cancel ')
        okButton = widgets.okPushButton(' Ok ')
        infoButton = widgets.infoPushButton(' Help... ')
        # restoreDefaultButton = widgets.reloadPushButton('Restore default')

        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(infoButton)
        # buttonsLayout.addWidget(restoreDefaultButton)
        buttonsLayout.addWidget(okButton)

        buttonsLayout.setContentsMargins(0, 10, 0, 10)

        okButton.clicked.connect(self.ok_cb)
        infoButton.clicked.connect(self.info_params)
        cancelButton.clicked.connect(self.close)
        # restoreDefaultButton.clicked.connect(self.restoreDefault)

        mainLayout.addWidget(initGroupBox)
        mainLayout.addLayout(initButtonsLayout)
        mainLayout.addSpacing(15)
        mainLayout.addStretch(1)
        mainLayout.addWidget(segmentGroupBox)
        mainLayout.addLayout(segmentButtonsLayout)

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

        self.minSize_SB.default = self.minSize_SB.value()
        self.minSolidity_DSB.default = self.minSolidity_DSB.value()
        self.maxElongation_DSB.default = self.maxElongation_DSB.value()
        self.artefactsGroupBox.default = True

        mainLayout.addSpacing(15)
        mainLayout.addStretch(1)
        mainLayout.addWidget(artefactsGroupBox)

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
        mainLayout.addLayout(postProcButtonsLayout)

        if url is not None:
            mainLayout.addWidget(
                self.createSeeHereLabel(url),
                alignment=Qt.AlignCenter
            )

        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

        font = QtGui.QFont()
        font.setPixelSize(13)
        self.setFont(font)

        self.configPars = self.readLastSelection()
        if self.configPars is None:
            initLoadLastSelButton.setDisabled(True)
            segmentLoadLastSelButton.setDisabled(True)
            postProcLoadLastSelButton.setDisabled(True)

        initLoadLastSelButton.click()
        segmentLoadLastSelButton.click()
        postProcLoadLastSelButton.click()

        # self.setModal(True)

    def createGroupParams(self, ArgSpecs_list, groupName):
        ArgWidget = namedtuple(
            'ArgsWidgets',
            ['name', 'type', 'widget', 'defaultVal', 'valueSetter']
        )
        ArgsWidgets_list = []
        groupBox = QGroupBox(groupName)

        groupBoxLayout = QGridLayout()
        for row, ArgSpec in enumerate(ArgSpecs_list):
            var_name = ArgSpec.name.replace('_', ' ').title()
            label = QLabel(f'{var_name}:  ')
            groupBoxLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
            if ArgSpec.type == bool:
                booleanGroup = QButtonGroup()
                booleanGroup.setExclusive(True)
                trueRadioButton = QRadioButton('True')
                falseRadioButton = QRadioButton('False')
                booleanGroup.addButton(trueRadioButton)
                booleanGroup.addButton(falseRadioButton)
                trueRadioButton.notButton = falseRadioButton
                falseRadioButton.notButton = trueRadioButton
                trueRadioButton.group = booleanGroup
                if ArgSpec.default:
                    trueRadioButton.setChecked(True)
                    defaultVal = True
                else:
                    falseRadioButton.setChecked(True)
                    defaultVal = False
                valueSetter = QRadioButton.setChecked
                widget = trueRadioButton
                groupBoxLayout.addWidget(trueRadioButton, row, 1)
                groupBoxLayout.addWidget(falseRadioButton, row, 2)
            elif ArgSpec.type == int:
                spinBox = QSpinBox()
                spinBox.setAlignment(Qt.AlignCenter)
                spinBox.setMaximum(2147483647)
                spinBox.setValue(ArgSpec.default)
                defaultVal = ArgSpec.default
                valueSetter = QSpinBox.setValue
                widget = spinBox
                groupBoxLayout.addWidget(spinBox, row, 1, 1, 2)
            elif ArgSpec.type == float:
                doubleSpinBox = QDoubleSpinBox()
                doubleSpinBox.setAlignment(Qt.AlignCenter)
                doubleSpinBox.setMaximum(2**32)
                doubleSpinBox.setValue(ArgSpec.default)
                widget = doubleSpinBox
                defaultVal = ArgSpec.default
                valueSetter = QDoubleSpinBox.setValue
                groupBoxLayout.addWidget(doubleSpinBox, row, 1, 1, 2)
            else:
                lineEdit = QLineEdit()
                lineEdit.setText(str(ArgSpec.default))
                lineEdit.setAlignment(Qt.AlignCenter)
                widget = lineEdit
                defaultVal = str(ArgSpec.default)
                valueSetter = QLineEdit.setText
                groupBoxLayout.addWidget(lineEdit, row, 1, 1, 2)

            argsInfo = ArgWidget(
                name=ArgSpec.name,
                type=ArgSpec.type,
                widget=widget,
                defaultVal=defaultVal,
                valueSetter=valueSetter
            )
            ArgsWidgets_list.append(argsInfo)

        groupBox.setLayout(groupBoxLayout)
        return groupBox, ArgsWidgets_list

    def restoreDefaultInit(self):
        for argWidget in self.init_argsWidgets:
            defaultVal = argWidget.defaultVal
            widget = argWidget.widget
            argWidget.valueSetter(widget, defaultVal)
            if defaultVal == False:
                argWidget.valueSetter(widget.notButton, True)

    def restoreDefaultSegment(self):
        for argWidget in self.segment2D_argsWidgets:
            defaultVal = argWidget.defaultVal
            widget = argWidget.widget
            argWidget.valueSetter(widget, defaultVal)
            if defaultVal == False:
                argWidget.valueSetter(widget.notButton, True)

    def restoreDefaultPostprocess(self):
        self.minSize_SB.setValue(self.minSize_SB.default)
        self.minSolidity_DSB.setValue(self.minSolidity_DSB.default)
        self.maxElongation_DSB.setValue(self.maxElongation_DSB.default)
        self.artefactsGroupBox.setChecked(self.artefactsGroupBox.default)

    def readLastSelection(self):
        self.ini_path = os.path.join(temp_path, 'last_params_segm_models.ini')
        if not os.path.exists(self.ini_path):
            return None

        configPars = config.ConfigParser()
        configPars.read(self.ini_path)
        return configPars

    def loadLastSelection(self, section, argWidgetList):
        if self.configPars is None:
            return

        getters = ['getboolean', 'getfloat', 'getint', 'get']
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
                except Exception:
                    pass
            widget = argWidget.widget
            argWidget.valueSetter(widget, val)

    def loadLastSelectionPostProcess(self):
        postProcessSection = f'{self.model_name}.postprocess'

        if postProcessSection not in self.configPars.sections():
            return

        minSize = self.configPars.getint(postProcessSection, 'minSize')
        self.minSize_SB.setValue(minSize)

        minSolidity = self.configPars.getfloat(
            postProcessSection, 'minSolidity'
        )
        self.minSolidity_DSB.setValue(minSolidity)

        maxElongation = self.configPars.getfloat(
            postProcessSection, 'maxElongation'
        )
        self.maxElongation_DSB.setValue(maxElongation)

        applyPostProcessing = self.configPars.getboolean(
            postProcessSection, 'applyPostProcessing'
        )
        self.artefactsGroupBox.setChecked(applyPostProcessing)

    def info_params(self):
        from cellacdc.models import CELLPOSE_MODELS, STARDIST_MODELS
        self.infoWin = widgets.myMessageBox()
        self.infoWin.setWindowTitle('Model parameters info')
        self.infoWin.setIcon()
        cp_models = [f'&nbsp;&nbsp;- {m}'for m in CELLPOSE_MODELS]
        cp_models = '<br>'.join(cp_models)
        stardist_models = [f'  - {m}'for m in STARDIST_MODELS]
        stardist_models = '<br>'.join(stardist_models)
        txt = html_utils.paragraph(
            'Currently Cell-ACDC has <b>four models implemented</b>: '
            'YeaZ, Cellpose, StarDist, and YeastMate.<br><br>'
            'Cellpose and StarDist have the following default models available:<br><br>'
            '<b>Cellpose</b>:<br><br>'
            f'{cp_models}<br><br>'
            '<b>StarDist</b>:<br>'
            f'{stardist_models}'
        )
        self.infoWin.addText(txt)
        self.infoWin.addButton(' Ok ')
        self.infoWin.show()

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
        self._saveParams()
        self.close()

    def _saveParams(self):
        if self.configPars is None:
            self.configPars = config.ConfigParser()
        self.configPars[f'{self.model_name}.init'] = {}
        self.configPars[f'{self.model_name}.segment'] = {}
        for key, val in self.init_kwargs.items():
            self.configPars[f'{self.model_name}.init'][key] = str(val)
        for key, val in self.segment2D_kwargs.items():
            self.configPars[f'{self.model_name}.segment'][key] = str(val)

        self.configPars[f'{self.model_name}.postprocess'] = {}
        postProcessConfig = self.configPars[f'{self.model_name}.postprocess']
        postProcessConfig['minSize'] = str(self.minSize)
        postProcessConfig['minSolidity'] = str(self.minSolidity)
        postProcessConfig['maxElongation'] = str(self.maxElongation)
        postProcessConfig['applyPostProcessing'] = str(self.applyPostProcessing)

        with open(self.ini_path, 'w') as configfile:
            self.configPars.write(configfile)

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
        <p style=font-size:13px>
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
        okButton = widgets.okPushButton('Ok')
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
    sigOk = pyqtSignal(object)

    def __init__(self, allChNames, isZstack, isSegm3D, parent=None, debug=False):
        super().__init__(parent)

        self.initAttributes()

        self.allChNames = allChNames

        self.cancel = True
        self.isOperatorMode = False

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

        size_metrics_desc = measurements.get_size_metrics_desc()
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

        self.setStyleSheet("""
            QTreeWidget::item:hover {background-color:#E6E6E6;}
            QTreeWidget::item:selected {background-color:#CFEB9B;}
            QTreeWidget::item:selected {color:black;}
            QTreeView {
                selection-background-color: #CFEB9B;
                selection-color: white;
                show-decoration-selected: 1;
            }
        """)

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

        # Save equation to "<user_path>/acdc-metrics/combine_metrics.ini" file
        config = measurements.read_saved_user_combine_config()

        equationsDict, isMixedChannels = self.getEquationsDict()
        for newColName, equation in equationsDict.items():
            config = measurements.add_user_combine_metrics(
                config, equation, newColName, isMixedChannels
            )

        self.sigOk.emit(self)

        isChannelLess = len(self.channelLessColnames) > 0
        if isChannelLess:
            channelLess_equation = self.equationDisplay.toPlainText()
            equation_name = self.newColNameLineEdit.text()
            config = measurements.add_channelLess_combine_metrics(
                config, channelLess_equation, equation_name,
                self.channelLessColnames
            )

        measurements.save_common_combine_metrics(config)

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
                _val = posData.segmSizeT
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
