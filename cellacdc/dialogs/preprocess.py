"""Cell-ACDC dialog windows: preprocess."""

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

class wandToleranceWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.slider = widgets.sliderWithSpinBox(title="Tolerance")
        self.slider.setMaximum(255)
        self.slider._layout.setColumnStretch(2, 21)

        self.setLayout(self.slider.layout)


class QDialogAutomaticThresholding(QBaseDialog):
    def __init__(self, parent=None, isSegm3D=True):
        super().__init__(parent)

        self.cancel = True

        self.setWindowTitle("Automatic thresholding parameters")

        layout = QVBoxLayout()
        formLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        row = 0
        self.sigmaGaussSpinbox = QDoubleSpinBox()
        self.sigmaGaussSpinbox.setValue(1)
        self.sigmaGaussSpinbox.setMaximum(2**31)
        self.sigmaGaussSpinbox.setAlignment(Qt.AlignCenter)
        formLayout.addWidget(
            QLabel("Gaussian filter sigma (0 to ignore): "),
            row,
            0,
            alignment=Qt.AlignRight,
        )
        formLayout.addWidget(self.sigmaGaussSpinbox, row, 1, 1, 2)

        row += 1
        self.threshMethodCombobox = QComboBox()
        self.threshMethodCombobox.addItems(
            ["Isodata", "Li", "Mean", "Minimum", "Otsu", "Triangle", "Yen"]
        )
        formLayout.addWidget(
            QLabel("Thresholding algorithm: "), row, 0, alignment=Qt.AlignRight
        )
        formLayout.addWidget(self.threshMethodCombobox, row, 1, 1, 2)

        self.segment3Dcheckbox = None
        if isSegm3D:
            row += 1
            formLayout.addWidget(
                QLabel("Segment 3D volume: "), row, 0, alignment=Qt.AlignRight
            )
            group = QButtonGroup()
            group.setExclusive(True)
            self.segment3Dcheckbox = QRadioButton("Yes")
            segmentSliceBySliceCheckbox = QRadioButton("No, segment slice-by-slice")
            group.addButton(self.segment3Dcheckbox)
            group.addButton(segmentSliceBySliceCheckbox)
            formLayout.addWidget(self.segment3Dcheckbox, row, 1)
            formLayout.addWidget(segmentSliceBySliceCheckbox, row, 2)
            self.segment3Dcheckbox.setChecked(True)

        okButton = widgets.okPushButton("Ok")
        cancelButton = widgets.cancelPushButton("Cancel")
        helpButton = widgets.helpPushButton("Help...")

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

        url = "https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html"
        webbrowser.open(url)

    def ok_cb(self):
        self.cancel = False
        self.gaussSigma = self.sigmaGaussSpinbox.value()
        threshMethod = self.threshMethodCombobox.currentText().lower()
        self.threshMethod = f"threshold_{threshMethod}"
        self.segment_kwargs = {
            "gauss_sigma": self.gaussSigma,
            "threshold_method": self.threshMethod,
            "segment_3D_volume": False,
        }
        self.reduceMemoryUsage = False
        if self.segment3Dcheckbox is not None:
            doSegm3D = self.segment3Dcheckbox.isChecked()
            self.segment_kwargs["segment_3D_volume"] = doSegm3D
        self.close()

    def loadLastSelection(self):
        self.ini_path = os.path.join(settings_folderpath, "last_params_segm_models.ini")
        if not os.path.exists(self.ini_path):
            return

        configPars = config.ConfigParser()
        configPars.read(self.ini_path)

        if "thresholding.segment" not in configPars.sections():
            return

        section = configPars["thresholding.segment"]
        self.sigmaGaussSpinbox.setValue(float(section["gauss_sigma"]))

        threshold_method = section["threshold_method"]
        Method = threshold_method[10:].capitalize()
        self.threshMethodCombobox.setCurrentText(Method)
        if self.segment3Dcheckbox is None:
            return
        self.segment3Dcheckbox.setChecked(section.getboolean("segment_3D_volume"))


class startStopFramesDialog(QBaseDialog):
    def __init__(
        self,
        SizeT,
        currentFrameNum=0,
        parent=None,
        windowTitle="Select frame range to segment",
    ):
        super().__init__(parent=parent)

        self.setWindowTitle(windowTitle)

        self.cancel = True

        layout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        self.selectFramesGroupbox = widgets.selectStartStopFrames(
            SizeT, currentFrameNum=currentFrameNum, parent=parent
        )

        okButton = widgets.okPushButton("Ok")
        cancelButton = widgets.cancelPushButton("Cancel")

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

        self.resize(int(self.width() * 1.5), self.height())

        if block:
            super().show(block=True)


class randomWalkerDialog(QDialog):
    def __init__(self, mainWindow):
        super().__init__(mainWindow)
        self.cancel = True
        self.mainWindow = mainWindow

        if mainWindow is not None:
            posData = self.mainWindow.data[self.mainWindow.pos_i]
            items = [posData.filename]
        else:
            items = ["test"]
        try:
            posData = self.mainWindow.data[self.mainWindow.pos_i]
            items.extend(list(posData.ol_data_dict.keys()))
        except Exception as e:
            pass

        self.keys = items

        self.setWindowTitle("Random walker segmentation")

        self.colors = [self.mainWindow.RWbkgrColor, self.mainWindow.RWforegrColor]

        mainLayout = QVBoxLayout()
        paramsLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        self.mainWindow.clearAllItems()

        row = 0
        paramsLayout.addWidget(QLabel("Background threshold:"), row, 0)
        row += 1
        self.bkgrThreshValLabel = QLabel("0.05")
        paramsLayout.addWidget(self.bkgrThreshValLabel, row, 1)
        self.bkgrThreshSlider = QSlider(Qt.Horizontal)
        self.bkgrThreshSlider.setMinimum(1)
        self.bkgrThreshSlider.setMaximum(100)
        self.bkgrThreshSlider.setValue(5)
        self.bkgrThreshSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.bkgrThreshSlider.setTickInterval(10)
        paramsLayout.addWidget(self.bkgrThreshSlider, row, 0)

        row += 1
        foregrQSLabel = QLabel("Foreground threshold:")
        # padding: top, left, bottom, right
        foregrQSLabel.setStyleSheet("font-size:13px; padding:5px 0px 0px 0px;")
        paramsLayout.addWidget(foregrQSLabel, row, 0)
        row += 1
        self.foregrThreshValLabel = QLabel("0.95")
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
        url1 = "https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_random_walker_segmentation.html"
        url2 = "https://scikit-image.org/docs/dev/api/skimage.segmentation.html#skimage.segmentation.random_walker"
        htmlTxt1 = f'<a href="{url1}">here</a>'
        htmlTxt2 = f'<a href="{url2}">here</a>'
        seeHereLabel = QLabel()
        seeHereLabel.setText(
            f"See {htmlTxt1} and {htmlTxt2} for details "
            "about Random walker segmentation."
        )
        seeHereLabel.setTextFormat(Qt.RichText)
        seeHereLabel.setTextInteractionFlags(Qt.TextBrowserInteraction)
        seeHereLabel.setOpenExternalLinks(True)
        font = QFont()
        font.setPixelSize(12)
        seeHereLabel.setFont(font)
        seeHereLabel.setStyleSheet("padding:12px 0px 0px 0px;")
        paramsLayout.addWidget(seeHereLabel, row, 0, 1, 2)

        computeButton = QPushButton("Compute segmentation")
        closeButton = QPushButton("Close")

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
        self.img = img / img.max()
        self.imgRGB = (skimage.color.gray2rgb(self.img) * 255).astype(np.uint8)

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
        bkgrThresh = self.bkgrThreshSlider.sliderPosition() / 100
        foregrThresh = self.foregrThreshSlider.sliderPosition() / 100
        img = self.img
        self.markers = np.zeros(img.shape, np.uint8)
        imgRange = img.max() - img.min()
        imgMin = img.min() + imgRange * bkgrThresh
        imgMax = img.min() + imgRange * foregrThresh
        self.markers[img < imgMin] = 1
        self.markers[img > imgMax] = 2
        return imgMin, imgMax

    def computeSegm(self, checked=True):
        self.mainWindow.storeUndoRedoStates(False)
        self.mainWindow.titleLabel.setText("Randomly walking around... ", color="w")
        img = self.img
        img = skimage.exposure.rescale_intensity(img)
        t0 = time.time()
        lab = skimage.segmentation.random_walker(img, self.markers, mode="bf")
        lab = skimage.measure.label(lab > 1)
        t1 = time.time()
        if len(np.unique(lab)) > 2:
            lab = skimage.morphology.remove_small_objects(lab, min_size=5)
        posData = self.mainWindow.data[self.mainWindow.pos_i]
        posData.lab = lab
        return t1 - t0

    def computeSegmAndPlot(self):
        deltaT = self.computeSegm()

        posData = self.mainWindow.data[self.mainWindow.pos_i]

        self.mainWindow.update_rp()
        self.mainWindow.tracking(enforce=True)
        self.mainWindow.updateAllImages()
        self.mainWindow.warnEditingWithCca_df("Random Walker segmentation")
        txt = f"Random Walker segmentation computed in {deltaT:.3f} s"
        print("-----------------")
        print(txt)
        print("=================")
        # self.mainWindow.titleLabel.setText(txt, color='g')

    def bkgrSliderMoved(self, intVal):
        self.bkgrThreshValLabel.setText(f"{intVal / 100:.2f}")
        self.plotMarkers()

    def foregrSliderMoved(self, intVal):
        self.foregrThreshValLabel.setText(f"{intVal / 100:.2f}")
        self.plotMarkers()

    def closeEvent(self, event):
        self.mainWindow.segmModel = ""
        self.mainWindow.updateAllImages()


class FutureFramesAction_QDialog(QDialog):
    def __init__(
        self,
        frame_i,
        last_tracked_i,
        change_txt,
        applyTrackingB=False,
        parent=None,
        addApplyAllButton=False,
    ):
        self.decision = None
        self.last_tracked_i = last_tracked_i
        super().__init__(parent)
        self.setWindowTitle("Future frames action?")

        mainLayout = QVBoxLayout()
        txtLayout = QVBoxLayout()
        doNotShowLayout = QVBoxLayout()
        buttonsLayout = QVBoxLayout()

        txt = html_utils.paragraph(
            "You already visited/checked future frames "
            f"{frame_i + 1}-{last_tracked_i + 1}.<br><br>"
            f'The requested <b>"{change_txt}"</b> change might result in<br>'
            "<b>NON-correct segmentation/tracking</b> for those frames.<br>"
        )

        txtLabel = QLabel(txt)
        txtLabel.setAlignment(Qt.AlignCenter)
        txtLayout.addWidget(txtLabel, alignment=Qt.AlignCenter)

        options = [
            f'Apply the "{change_txt}" <b>only to current frame and re-initialize</b><br>'
            "the future frames to the segmentation file present<br>"
            "on the hard drive.",
            "Apply <b>only to this frame and keep the future frames</b> as they are.",
            "Apply the change to <b>ALL visited/checked future frames</b>.",
        ]
        if addApplyAllButton:
            options.append(
                "Apply to <b>ALL future frames including unvisited ones</b>."
            )
        if applyTrackingB:
            options.append("Repeat ONLY tracking for all future frames (RECOMMENDED)")

        infoTxt = html_utils.paragraph(
            f"Choose <b>one of the following options:</b>"
            f"{html_utils.to_list(options, ordered=True)}"
        )

        infotxtLabel = QLabel(infoTxt)
        txtLayout.addWidget(infotxtLabel, alignment=Qt.AlignCenter)

        noteLayout = QHBoxLayout()
        noteTxt = html_utils.paragraph(
            "Only changes applied to current frame can be undone.<br>"
            "Changes applied to <b>future frames CANNOT be UNDONE</b><br>"
        )
        noteLayout.addWidget(
            QLabel(html_utils.paragraph("NOTE:")), alignment=Qt.AlignTop
        )
        noteTxtLabel = QLabel(noteTxt)
        noteLayout.addWidget(noteTxtLabel)
        noteLayout.addStretch(1)
        txtLayout.addSpacing(10)
        txtLayout.addLayout(noteLayout)

        # Do not show this message again checkbox
        doNotShowCheckbox = QCheckBox(
            "Remember my choice and do not show this message again"
        )
        doNotShowLayout.addWidget(doNotShowCheckbox)
        doNotShowLayout.setContentsMargins(50, 0, 0, 10)
        self.doNotShowCheckbox = doNotShowCheckbox

        apply_and_reinit_b = widgets.reloadPushButton(
            " 1. Apply only to this frame and re-initialize future frames"
        )

        self.apply_and_reinit_b = apply_and_reinit_b
        buttonsLayout.addWidget(apply_and_reinit_b)

        apply_and_NOTreinit_b = widgets.currentPushButton(
            " 2. Apply only to this frame and keep future frames as they are"
        )
        self.apply_and_NOTreinit_b = apply_and_NOTreinit_b
        buttonsLayout.addWidget(apply_and_NOTreinit_b)

        apply_to_all_visited_b = widgets.futurePushButton(
            " 3. Apply to all future VISITED frames"
        )
        self.apply_to_all_visited_b = apply_to_all_visited_b
        buttonsLayout.addWidget(apply_to_all_visited_b)

        if addApplyAllButton:
            apply_to_all_b = QPushButton(
                " 4. Apply to ALL future frames (including unvisted)"
            )
            apply_to_all_b.setIcon(QIcon(":arrow_future_all.svg"))
            self.apply_to_all_b = apply_to_all_b
            buttonsLayout.addWidget(apply_to_all_b)

        self.applyTrackingButton = None
        if applyTrackingB:
            n = "5" if addApplyAllButton else "4"
            applyTrackingButton = QPushButton(
                f" {n}. Repeat ONLY tracking for all future frames"
            )
            applyTrackingButton.setIcon(QIcon(":repeat-tracking.svg"))
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
            self.decision = "apply_and_reinit"
            self.endFrame_i = None
        elif button == self.apply_and_NOTreinit_b:
            self.decision = "apply_and_NOTreinit"
            self.endFrame_i = None
        elif button == self.apply_to_all_visited_b:
            self.decision = "apply_to_all_visited"
            self.endFrame_i = self.last_tracked_i
        elif button == self.applyTrackingButton:
            self.decision = "only_tracking"
            self.endFrame_i = self.last_tracked_i
        elif button == self.apply_to_all_b:
            self.decision = "apply_to_all"
            self.endFrame_i = self.last_tracked_i
        self.close()

    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()
        for button in self.ButtonsGroup.buttons():
            button.setMinimumHeight(int(button.height() * 1.2))
        if hasattr(self, "apply_to_all_b"):
            iconHeight = self.apply_to_all_b.iconSize().height()
            self.apply_to_all_b.setIconSize(QSize(iconHeight * 2, iconHeight))
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()


class PostProcessSegmParams(QGroupBox):
    valueChanged = Signal(object)
    editingFinished = Signal()

    def __init__(
        self,
        title,
        posData,
        useSliders=False,
        parent=None,
        maxSize=None,
        force_postprocess_2D=False,
    ):
        QGroupBox.__init__(self, title, parent)
        SizeZ = posData.SizeZ
        self.isSegm3D = posData.isSegm3D
        self.channelName = posData.user_ch_name
        self.useSliders = useSliders
        self.force_postprocess_2D = force_postprocess_2D
        if maxSize is None:
            maxSize = 2147483647

        layout = QGridLayout()

        self.controlWidgets = []

        row = 0
        label = QLabel("Minimum area (pixels) ")
        layout.addWidget(label, row, 0, alignment=Qt.AlignRight)

        minSize_SB = widgets.PostProcessSegmWidget(1, 1000, 10, useSliders, label=label)

        txt = "<b>Area</b> is the total number of pixels in the segmented object."

        layout.addWidget(minSize_SB, row, 1)
        infoButton = widgets.infoPushButton()
        infoButton.clicked.connect(self.showInfo)
        infoButton.tooltip = txt
        infoButton.name = "area"
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
            0, 1.0, 0.5, useSliders, isFloat=True, normalize=True, label=label
        )
        minSolidity_DSB.setValue(0.5)
        minSolidity_DSB.setSingleStep(0.1)
        self.controlWidgets.append(minSolidity_DSB)

        txt = (
            "<b>Solidity</b> is a measure of convexity. A solidity of 1 means "
            "that the shape is fully convex (i.e., equal to the convex hull). "
            "As solidity approaches 0 the object is more concave.<br>"
            "Write 0 for ignoring this parameter."
        )

        layout.addWidget(minSolidity_DSB, row, 1)
        infoButton = widgets.infoPushButton()
        infoButton.clicked.connect(self.showInfo)
        infoButton.tooltip = txt
        infoButton.name = "solidity"
        infoButton.desc = f'less than "{label.text()}"'
        layout.addWidget(infoButton, row, 2)
        self.minSolidity_DSB = minSolidity_DSB

        row += 1
        label = QLabel("Max elongation (1=circle) ")
        layout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        maxElongation_DSB = widgets.PostProcessSegmWidget(
            0, 100, 3, useSliders, isFloat=True, normalize=False, label=label
        )
        maxElongation_DSB.setDecimals(1)
        maxElongation_DSB.setSingleStep(1.0)

        txt = (
            "<b>Elongation</b> is the ratio between major and minor axis lengths. "
            "An elongation of 1 is like a circle.<br>"
            "Write 0 for ignoring this parameter."
        )

        layout.addWidget(maxElongation_DSB, row, 1)
        infoButton = widgets.infoPushButton()
        infoButton.clicked.connect(self.showInfo)
        infoButton.tooltip = txt
        infoButton.name = "elongation"
        infoButton.desc = f'greater than "{label.text()}"'
        layout.addWidget(infoButton, row, 2)
        self.maxElongation_DSB = maxElongation_DSB
        self.controlWidgets.append(maxElongation_DSB)

        if self.isSegm3D:
            row += 1
            label = QLabel("Minimum number of z-slices ")
            layout.addWidget(label, row, 0, alignment=Qt.AlignRight)
            minObjSizeZ_SB = widgets.PostProcessSegmWidget(
                0, SizeZ, 3, useSliders, isFloat=False, normalize=False, label=label
            )

            txt = "<b>Minimum number of z-slices</b> per object."

            layout.addWidget(minObjSizeZ_SB, row, 1)
            infoButton = widgets.infoPushButton()
            infoButton.clicked.connect(self.showInfo)
            infoButton.tooltip = txt
            infoButton.name = "number of z-slices"
            infoButton.desc = f'less than "{label.text()}"'
            layout.addWidget(infoButton, row, 2)
            self.minObjSizeZ_SB = minObjSizeZ_SB
            self.controlWidgets.append(minObjSizeZ_SB)
        else:
            self.minObjSizeZ_SB = widgets.NoneWidget()

        row += 1
        addCustomFeatureLayout = QHBoxLayout()
        self.addCustomFeaturesButton = widgets.setPushButton(
            "Select custom features for post-processing...",
        )
        addCustomFeatureLayout.addWidget(self.addCustomFeaturesButton)
        addCustomFeatureLayout.addStretch(1)
        self.selectedFeaturesDialog = SelectFeaturesRangeDialog(
            posData=posData, parent=self, force_postprocess_2D=force_postprocess_2D
        )
        self.selectedFeaturesDialog.hide()
        self.addCustomFeaturesButton.clicked.connect(self.selectedFeaturesDialog.show)
        self.selectedFeaturesDialog.sigValueChanged.connect(self.onValueChanged)

        layout.addLayout(addCustomFeatureLayout, row, 0, 1, 2)

        layout.setColumnStretch(1, 2)
        # layout.setRowStretch(row+1, 1)

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
        self.selectedFeaturesDialog.groupbox.resetFields()

    def restoreFromKwargs(self, kwargs):
        for name, value in kwargs.items():
            if name == "min_solidity":
                self.minSolidity_DSB.setValue(value)
            elif name == "min_area":
                self.minSize_SB.setValue(value)
            elif name == "max_elongation":
                self.maxElongation_DSB.setValue(value)
            elif name == "min_obj_no_zslices":
                self.minObjSizeZ_SB.setValue(value)

    def kwargs(self):
        kwargs = {
            "min_solidity": self.minSolidity_DSB.value(),
            "min_area": self.minSize_SB.value(),
            "max_elongation": self.maxElongation_DSB.value(),
            "min_obj_no_zslices": self.minObjSizeZ_SB.value(),
        }
        return kwargs

    def onValueChanged(self, value):
        self.valueChanged.emit(value)

    def onEditingFinished(self):
        self.editingFinished.emit()

    def showInfo(self):
        title = f"{self.sender().text()} info"
        tooltip = self.sender().tooltip
        name = self.sender().name
        desc = self.sender().desc
        txt = f"""
            The post-processing step is applied to the output of the 
            segmentation model.<br><br>
            During this step, Cell-ACDC will remove all the objects with {name}
            <b>{desc}</b>.<br><br>
            {tooltip}    
        """
        if self.isCheckable():
            note = f""""
                You can deactivate this step by un-checking the checkbox 
                called "Post-processing parameters".
            """
            txt = f"{txt}{note}"
        msg = widgets.myMessageBox(showCentered=False)
        msg.information(self, title, html_utils.paragraph(txt))


class PostProcessSegmDialog(QBaseDialog):
    sigClosed = Signal()
    sigValueChanged = Signal(object, object)
    sigEditingFinished = Signal()
    sigApplyToAllFutureFrames = Signal(object, object, object)

    def __init__(self, posData, mainWin=None, useSliders=True, maxSize=None):
        super().__init__(mainWin)
        self.cancel = True
        self.mainWin = mainWin
        self.isTimelapse = False
        self.isMultiPos = False
        if mainWin is not None:
            self.isMultiPos = len(self.mainWin.data) > 1
            self.isTimelapse = self.mainWin.data[self.mainWin.pos_i].SizeT > 1

        self.setWindowTitle("Post-processing segmentation parameters")
        self.setWindowFlags(Qt.Tool | Qt.WindowStaysOnTopHint)

        mainLayout = QVBoxLayout()
        buttonsLayout = QHBoxLayout()

        self.postProcessGroupbox = PostProcessSegmParams(
            "Post-processing parameters",
            posData,
            useSliders=useSliders,
            maxSize=maxSize,
            parent=mainWin,
        )

        self.postProcessGroupbox.valueChanged.connect(self.valueChanged)
        self.postProcessGroupbox.editingFinished.connect(self.onEditingFinished)

        if self.isTimelapse:
            applyAllButton = widgets.futurePushButton("Apply to all frames...")
            applyAllButton.clicked.connect(self.applyAll_cb)
            applyButton = widgets.okPushButton("Apply", isDefault=False)
            applyButton.clicked.connect(self.apply_cb)
        elif self.isMultiPos:
            applyAllButton = widgets.futurePushButton("Apply to all Positions...")
            applyAllButton.clicked.connect(self.applyAll_cb)
            applyButton = widgets.okPushButton("Apply", isDefault=False)
            applyButton.clicked.connect(self.apply_cb)
        else:
            applyAllButton = widgets.okPushButton("Apply", isDefault=False)
            applyAllButton.clicked.connect(self.ok_cb)
            applyButton = None

        cancelButton = widgets.cancelPushButton("Cancel")

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        if applyButton is not None:
            buttonsLayout.addWidget(applyButton)
        buttonsLayout.addWidget(applyAllButton)

        emitEditingFinishedButton = widgets.okPushButton()
        buttonsLayout.addWidget(emitEditingFinishedButton)
        emitEditingFinishedButton.hide()
        buttonsLayout.setContentsMargins(0, 10, 0, 0)

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
        # self.img.setCurrentPosIndex(self.pos_i)
        # self.img.minMaxValuesMapper = self.mainWin.img1.minMaxValuesMapper
        self.origLab = self.posData.lab.copy()
        self.origRp = skimage.measure.regionprops(self.origLab)
        self.origObjs = {obj.label: obj for obj in self.origRp}

    def valueChanged(self, value):
        lab, delObjs = self.apply()
        self.sigValueChanged.emit(lab, delObjs)

    def apply(self, origLab=None):
        self.mainWin.warnEditingWithCca_df(
            "post-processing segmentation mask", update_images=False
        )
        ccaAnnotRemoved = self.mainWin.removeCcaAnnotationsCurrentFrame()
        if ccaAnnotRemoved:
            self.mainWin.updateAllImages()

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
                return_delIDs=True,
            )
            delIDs.extend(custom_delIDs)

        delObjs = {delID: self.origObjs[delID] for delID in delIDs}
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
            self.postProcessGroupbox.selectedFeaturesRange(),
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
        if hasattr(self, "origSegmData"):
            if self.isTimelapse:
                current_frame_i = self.posData.frame_i
                for frame_i in range(self.posData.segmSizeT):
                    self.posData.frame_i = frame_i
                    origLab = self.origSegmData[frame_i]
                    lab = self.posData.allData_li[frame_i]["labels"]
                    if lab is None:
                        # Non-visited frame modify segm_data
                        self.posData.segm_data[frame_i] = origLab
                    else:
                        self.posData.allData_li[frame_i]["labels"] = origLab.copy()
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
                    self.posData.allData_li[0]["labels"] = lab.copy()
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
        self.resize(int(self.width() * 1.5), self.height())
        super().show(block=block)

    def closeEvent(self, event):
        self.sigClosed.emit()
        if self.cancel:
            self.undoChanges()
        super().closeEvent(event)


class FunctionParamsDialog(QBaseDialog):
    sigValuesChanged = Signal(dict)

    def __init__(
        self,
        params_argspecs,
        function_name="Function",
        df_metadata=None,
        parent=None,
        addApplyButton=False,
    ):
        self.cancel = True
        self.df_metadata = df_metadata

        super().__init__(parent)

        self.setWindowTitle(f"{function_name} parameters")

        self.mainLayout = QVBoxLayout()

        widgetsLayout, self.argsWidgets = self.getWidgetsLayout(params_argspecs)

        buttonsLayout = widgets.CancelOkButtonsLayout()

        self.buttonsLayout = buttonsLayout
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        if addApplyButton:
            applyButton = widgets.viewPushButton("Apply")
            applyButton.clicked.connect(self.emitValuesChanged)
            buttonsLayout.insertWidget(3, applyButton)
            self.applyButton = applyButton

        self.mainLayout.addLayout(widgetsLayout)
        self.mainLayout.addSpacing(20)
        self.mainLayout.addLayout(buttonsLayout)

        self.setLayout(self.mainLayout)

    def emitValuesChanged(self, *args, **kwargs):
        self.sigValuesChanged.emit(self.functionKwargs())

    def functionKwargs(self):
        function_kwargs = {
            argWidget.name: argWidget.valueGetter(argWidget.widget)
            for argWidget in self.argsWidgets
        }
        return function_kwargs

    def kwargWidgetMapper(self) -> Dict[str, tuple]:
        kwarg_widget_mapper = {
            argWidget.name: (argWidget.widget, argWidget.valueSetter)
            for argWidget in self.argsWidgets
        }
        return kwarg_widget_mapper

    def ok_cb(self):
        self.cancel = False

        self.function_kwargs = self.functionKwargs()

        self.close()

    def getValueFromMetadata(self, name):
        try:
            value = self.df_metadata.at[name, "values"]
        except Exception as e:
            # traceback.print_exc()
            value = None
        return value

    def getWidgetsLayout(self, params_argspecs):
        widgetsLayout = QGridLayout()
        ArgsWidgets_list = []

        for row, ArgSpec in enumerate(params_argspecs):
            if _types.is_widget_not_required(ArgSpec):
                continue

            arg_name = ArgSpec.name
            var_name = arg_name.replace("_", " ")
            var_name = f"{var_name[0].upper()}{var_name[1:]}"
            label = QLabel(f"{var_name}:  ")
            metadata_val = self.getValueFromMetadata(ArgSpec.name)
            widgetsLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)
            try:
                values = ArgSpec.type().values
                isCustomListType = True
            except Exception as err:
                isCustomListType = False

            isVectorEntry = False
            try:
                if isinstance(ArgSpec.type(), _types.Vector):
                    isVectorEntry = True
            except Exception as err:
                pass

            isFolderPath = False
            try:
                if isinstance(ArgSpec.type(), _types.FolderPath):
                    isFolderPath = True
            except Exception as err:
                pass

            isCustomWidget = hasattr(ArgSpec.type, "isWidget")

            if isCustomWidget:
                widget = ArgSpec.type().widget
                self.checkIfTypeCLassHasCastDtype(widget)
                defaultVal = ArgSpec.default
                valueSetter = widget.setValue
                valueGetter = widget.value
                widgetsLayout.addWidget(widget, row, 1, 1, 2)
                try:
                    widget.sigValueChanged.connect(self.emitValuesChanged)
                except Exception as err:
                    pass
            elif isVectorEntry:
                vectorLineEdit = widgets.VectorLineEdit()
                self.checkIfTypeCLassHasCastDtype(ArgSpec.type)
                vectorLineEdit.setValue(ArgSpec.default)
                defaultVal = ArgSpec.default
                valueSetter = widgets.VectorLineEdit.setValue
                valueGetter = widgets.VectorLineEdit.value
                widget = vectorLineEdit
                widgetsLayout.addWidget(vectorLineEdit, row, 1, 1, 2)
                widget.valueChangeFinished.connect(self.emitValuesChanged)
            elif isFolderPath:
                folderPathControl = widgets.FolderPathControl()
                self.checkIfTypeCLassHasCastDtype(ArgSpec.type)
                folderPathControl.setText(str(ArgSpec.default))
                widget = folderPathControl
                defaultVal = str(ArgSpec.default)
                valueSetter = widgets.FolderPathControl.setText
                valueGetter = widgets.FolderPathControl.path
                widgetsLayout.addWidget(folderPathControl, row, 1, 1, 2)
                widget.sigValueChanged.connect(self.emitValuesChanged)
            elif ArgSpec.type == bool:
                booleanGroup = QButtonGroup()
                booleanGroup.setExclusive(True)
                checkBox = widgets.Toggle()
                checkBox.setChecked(ArgSpec.default)
                defaultVal = ArgSpec.default
                valueSetter = widgets.Toggle.setChecked
                valueGetter = widgets.Toggle.isChecked
                widget = checkBox
                widgetsLayout.addWidget(
                    checkBox, row, 1, 1, 2, alignment=Qt.AlignCenter
                )
                widget.toggled.connect(self.emitValuesChanged)
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
                widgetsLayout.addWidget(spinBox, row, 1, 1, 2)
                widget.sigValueChanged.connect(self.emitValuesChanged)
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
                widgetsLayout.addWidget(doubleSpinBox, row, 1, 1, 2)
                widget.valueChanged.connect(self.emitValuesChanged)
            elif ArgSpec.type == os.PathLike:
                filePathControl = widgets.filePathControl()
                filePathControl.setText(str(ArgSpec.default))
                widget = filePathControl
                defaultVal = str(ArgSpec.default)
                valueSetter = widgets.filePathControl.setText
                valueGetter = widgets.filePathControl.path
                widgetsLayout.addWidget(filePathControl, row, 1, 1, 2)
                widget.sigValueChanged.connect(self.emitValuesChanged)
            elif isCustomListType:
                items = ArgSpec.type().values
                ArgSpec.type.cast_dtype = _types.to_str
                defaultVal = str(ArgSpec.default)
                combobox = widgets.AlphaNumericComboBox()
                combobox.addItems(items)
                combobox.setCurrentValue(defaultVal)
                valueSetter = widgets.AlphaNumericComboBox.setCurrentValue
                valueGetter = widgets.AlphaNumericComboBox.currentValue
                widget = combobox
                widgetsLayout.addWidget(combobox, row, 1, 1, 2)
                widget.currentTextChanged.connect(self.emitValuesChanged)
            else:
                lineEdit = QLineEdit()
                lineEdit.setText(str(ArgSpec.default))
                lineEdit.setAlignment(Qt.AlignCenter)
                widget = lineEdit
                defaultVal = str(ArgSpec.default)
                valueSetter = QLineEdit.setText
                valueGetter = QLineEdit.text
                widgetsLayout.addWidget(lineEdit, row, 1, 1, 2)
                widget.editingFinished.connect(self.emitValuesChanged)

            if ArgSpec.desc:
                infoButton = self.getInfoButton(ArgSpec.name, ArgSpec.desc)
                widgetsLayout.addWidget(infoButton, row, 3)

            argsInfo = ArgWidget(
                name=ArgSpec.name,
                type=ArgSpec.type,
                widget=widget,
                defaultVal=defaultVal,
                valueSetter=valueSetter,
                valueGetter=valueGetter,
            )
            ArgsWidgets_list.append(argsInfo)

        widgetsLayout.setColumnStretch(0, 0)
        widgetsLayout.setColumnStretch(1, 1)
        widgetsLayout.setColumnStretch(3, 0)

        return widgetsLayout, ArgsWidgets_list

    def checkIfTypeCLassHasCastDtype(self, cls):
        cast_dtype = getattr(cls, "cast_dtype", None)
        if callable(cast_dtype):
            return

        raise AttributeError(
            "The custom type or widget does not have the `cast_dtype` method. "
            "Please, implement it. The method should cast the value to the "
            "correct type."
        )

    def getInfoButton(self, param_name, infoText):
        infoButton = widgets.infoPushButton()
        infoButton.param_name = param_name
        infoButton.setToolTip(
            f"Click to get more info about `{param_name}` parameter..."
        )
        infoButton.infoText = infoText
        infoButton.clicked.connect(self.showInfoParam)
        return infoButton

    def showInfoParam(self):
        text = self.sender().infoText
        text = html_utils.rst_urls_to_html(text)
        text = html_utils.rst_to_html(text)
        text = html_utils.paragraph(text)
        param_name = self.sender().param_name
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, f"Info about `{param_name}` parameter", text)


class stopFrameDialog(QBaseDialog):
    def __init__(self, posDatas, parent=None):
        super().__init__(parent=parent)

        self.cancel = True

        self.setWindowTitle("Stop frame")

        mainLayout = QVBoxLayout()

        infoTxt = html_utils.paragraph(
            "Enter a <b>stop frame number</b> for each of the loaded Positions",
            center=True,
        )
        exp_path = posDatas[0].exp_path
        exp_path = os.path.normpath(exp_path).split(os.sep)
        exp_path = f"...{f'{os.sep}'.join(exp_path[-4:])}"
        subInfoTxt = html_utils.paragraph(
            f"Experiment folder: <code>{exp_path}<code>", font_size="12px", center=True
        )
        infoLabel = QLabel(f"{infoTxt}{subInfoTxt}")
        infoLabel.setToolTip(posDatas[0].exp_path)
        mainLayout.addWidget(infoLabel)
        mainLayout.addSpacing(20)

        self.posDatas = posDatas
        for posData in posDatas:
            _layout = QHBoxLayout()
            _layout.addStretch(1)
            _label = QLabel(html_utils.paragraph(f"{posData.pos_foldername}"))
            _layout.addWidget(_label)

            _spinBox = QSpinBox()
            _spinBox.setMaximum(214748364)
            _spinBox.setAlignment(Qt.AlignCenter)
            _spinBox.setFont(font)
            if posData.acdc_df is not None:
                _val = posData.acdc_df.index.get_level_values(0).max() + 1
            else:
                _val = posData.readLastUsedStopFrameNumber()
            if _val is None:
                _val = posData.SizeT
            _spinBox.setValue(_val)

            posData.stopFrameSpinbox = _spinBox

            _layout.addWidget(_spinBox)

            viewButton = widgets.viewPushButton("Visualize...")
            viewButton.clicked.connect(partial(self.viewChannelData, posData, _spinBox))
            _layout.addWidget(viewButton, alignment=Qt.AlignRight)

            _layout.addStretch(1)

            mainLayout.addLayout(_layout)

        buttonsLayout = QHBoxLayout()

        okButton = widgets.okPushButton(" Ok ")
        cancelButton = widgets.cancelPushButton(" Cancel ")

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
        self.sender().setText("Loading...")
        QTimer.singleShot(
            200, partial(self._viewChannelData, posData, spinBox, self.sender())
        )

    def _viewChannelData(self, posData, spinBox, senderButton):
        chNames = posData.chNames
        if len(chNames) > 1:
            ch_name_selector = prompts.select_channel_name(
                which_channel="segm", allow_abort=False
            )
            ch_name_selector.QtPrompt(
                self, chNames, "Select channel name to visualize: "
            )
            if ch_name_selector.was_aborted:
                return
            chName = ch_name_selector.channel_name
        else:
            chName = chNames[0]

        channel_file_path = load.get_filename_from_channel(posData.images_path, chName)
        posData.frame_i = 0
        posData.loadImgData(imgPath=channel_file_path)
        self.slideshowWin = imageViewer(posData=posData, spinBox=spinBox)
        self.slideshowWin.update_img()
        self.slideshowWin.show()
        senderButton.setText("Visualize...")

    def ok_cb(self):
        self.cancel = False
        for posData in self.posDatas:
            stopFrameNum = posData.stopFrameSpinbox.value()
            posData.stopFrameNum = stopFrameNum
        self.close()


class DataPrepSubCropsPathsDialog(QBaseDialog):
    def __init__(self, cropPaths=None, parent=None):
        self.cancel = True

        super().__init__(parent=parent)

        mainLayout = QVBoxLayout()

        gridLayout = QGridLayout()
        row = 0

        if cropPaths is None:
            cropPaths = {os.path.expanduser("~"): 1}

        if any([numCrops > 1 for numCrops in cropPaths.values()]):
            row += 1
            gridLayout.addWidget(QLabel("Same folder for all crops:"), row, 0)
            self.sameFolderPathToggle = widgets.Toggle()
            gridLayout.addWidget(
                self.sameFolderPathToggle, row, 1, alignment=Qt.AlignCenter
            )
            self.sameFolderPathToggle.setChecked(True)
            self.sameFolderPathToggle.toggled.connect(self.setSameFolderPath)

        self.windowMinWidth = 0
        minWidth = int(self.screen().size().width() / 3)
        self.folderPathLineEdits = defaultdict(list)
        for path, numCrops in cropPaths.items():
            row += 1
            gridLayout.addWidget(QLabel("Master Position:"), row, 0)
            masterPathLabel = QLabel(f"<code>{path}</code>")
            gridLayout.addWidget(masterPathLabel, row, 1)

            scrollArea = QScrollArea()
            scrollArea.setWidgetResizable(True)
            scrollAreaLayout = QGridLayout()
            for i in range(numCrops):
                label = QLabel(f"<b>Crop {i + 1}</b> folder path:")
                scrollAreaLayout.addWidget(label, i, 0)
                folderPathLineEdit = widgets.ElidingLineEdit()
                folderPathLineEdit.label = label
                folderPathLineEdit.setText(path)
                scrollAreaLayout.addWidget(folderPathLineEdit, i, 1)
                browseButton = widgets.browseFileButton(start_dir=path, openFolder=True)
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
                + scrollArea.verticalScrollBar().sizeHint().width()
                + 20
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
            f"The following folder path for crop number {cropNum} "
            "is <b>not a valid folder or does not exist</b>:"
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Not a valid folder", text, commands=(folderPath,))

    def askOverwritingPaths(self, overwritingPaths):
        text = html_utils.paragraph(
            "Data in the following paths will be <b>overwritten with "
            "cropped data.</b><br><br>"
            "Are you sure you want to continue?"
        )
        msg = widgets.myMessageBox(wrapText=False)
        _, yesButton = msg.warning(
            self,
            "Not a valid folder",
            text,
            commands=overwritingPaths,
            buttonsTexts=("No, let me edit paths", "Yes, overwrite"),
        )
        return msg.clickedButton == yesButton

    def validatePaths(self):
        for masterPath, lineEdits in self.folderPathLineEdits.items():
            for i, lineEdit in enumerate(lineEdits):
                path = lineEdit.text()
                if os.path.exists(path) and os.path.isdir(path):
                    continue

                self.warnFolderPathNotValid(i + 1, masterPath, path)
                return False

        overwritingPaths = []
        for masterPath, lineEdits in self.folderPathLineEdits.items():
            masterPath = masterPath.replace("\\", "/")
            if not masterPath.endswith("Images"):
                continue

            for i, lineEdit in enumerate(lineEdits):
                path = lineEdit.text()
                path = path.replace("\\", "/")
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


class PreProcessParamsWidget(QWidget):
    sigLoadRecipe = Signal()
    sigLoadSavedRecipe = Signal()
    sigValuesChanged = Signal(list)

    def __init__(self, df_metadata=None, addApplyButton=False, parent=None):
        super().__init__(parent)

        mainLayout = QVBoxLayout()

        self.df_metadata = df_metadata
        self.addApplyButton = addApplyButton

        groupbox = QGroupBox()
        self.groupbox = groupbox

        groupbox.setTitle("Pre-processing")
        groupbox.setCheckable(True)

        self.gridLayout = QGridLayout()
        self.row = -1
        self.stepsWidgets = {}

        self.gridLayout.setColumnStretch(0, 0)
        self.gridLayout.setColumnStretch(1, 1)
        self.gridLayout.setColumnStretch(2, 0)
        self.gridLayout.setColumnStretch(3, 0)
        self.gridLayout.setColumnStretch(4, 0)
        groupbox.setLayout(self.gridLayout)

        buttonsLayout = QGridLayout()
        row = 0
        col = 0
        buttonsLayout.setColumnStretch(col, 1)

        loadRecipeButton = widgets.OpenFilePushButton("Load saved recipe...")
        self.loadRecipeButton = loadRecipeButton
        buttonsLayout.addWidget(loadRecipeButton, row, col + 2)

        saveRecipeButton = widgets.savePushButton("Save current recipe...")
        self.saveRecipeButton = saveRecipeButton
        buttonsLayout.addWidget(saveRecipeButton, row + 1, col + 2)

        loadLastRecipeButton = widgets.reloadPushButton("Load last parameters")
        self.loadLastRecipeButton = loadLastRecipeButton
        buttonsLayout.addWidget(loadLastRecipeButton, row, col + 1)

        self.buttonsLayout = buttonsLayout

        loadLastRecipeButton.clicked.connect(self.emitLoadRecipe)
        saveRecipeButton.clicked.connect(self.saveRecipe)
        loadRecipeButton.clicked.connect(self.selectAndLoadRecipe)

        mainLayout.addWidget(groupbox)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)

        self.addStep(is_first=True)

        mainLayout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(mainLayout)

    def stepSizeHeightHint(self):
        stepWidgets = self.stepsWidgets[1]
        height = (
            stepWidgets["stepLabel"].minimumSizeHint().height()
            + stepWidgets["selector"].minimumSizeHint().height()
        )
        return height

    def setChecked(self, checked):
        self.groupbox.setChecked(checked)

    def emitLoadRecipe(self):
        self.sigLoadRecipe.emit()

    def loadRecipe(self, configPars: dict):
        for stepWidgets in list(self.stepsWidgets.values()):
            try:
                stepWidgets["delButton"].click()
            except Exception as err:
                pass

        configPars = self.sortStepsConfigPars(configPars)
        for s in range(1, len(configPars)):
            self.stepsWidgets[1]["addButton"].click()

        for i, (section, section_items) in enumerate(configPars.items()):
            step_n = i + 1
            selector = self.stepsWidgets[step_n]["selector"]
            kwarg_to_value_mapper = {}
            for option, value in section_items.items():
                if option == "method":
                    selector.setCurrentText(value)
                    method = value
                else:
                    kwarg_to_value_mapper[option] = value
            selector.setParams(method, kwarg_to_value_mapper)

        self.setChecked(True)

    def sortStepsConfigPars(self, configPars: dict):
        sortedConfigPars = {}
        sortedKeys = sorted(
            configPars.keys(), key=lambda key: int(re.findall(r"step(\d+)", key)[0])
        )
        for key in sortedKeys:
            sortedConfigPars[key] = configPars[key]
        return sortedConfigPars

    def saveRecipeUI(
        self, folder_path, ext, title, basename, hintText, default_text
    ):  # -> tuple[Literal[False], Literal['']] | tuple[Literal[True], Any]:
        win = filenameDialog(
            title=title,
            basename=basename,
            ext=ext,
            hintText=hintText,
            allowEmpty=False,
            defaultEntry=default_text,
            parent=self,
        )
        win.exec_()
        if win.cancel:
            return False, ""

        self.cancel = False
        filepath = win.filename
        os.makedirs(folder_path, exist_ok=True)
        filepath = os.path.join(folder_path, filepath)

        if os.path.exists(filepath):
            proceed = self.warnExistingRecipeFile(filepath)
            if not proceed:
                return False, ""

        return True, filepath

    def saveRecipe(self):
        recipe = self.recipe()
        if recipe is None:
            return

        default_text = ""
        for step in recipe[:2]:
            method = step["method"]
            func_name = config.PREPROCESS_MAPPER[method]["function_name"]
            default_text = f"{default_text}-{func_name}"
        default_text = default_text.lstrip("-")

        proceed, ini_filepath = self.saveRecipeUI(
            preproc_recipes_path,
            ".ini",
            "Filename for pre-processing recipe",
            "preprocessing_recipe",
            "Insert a <b>filename</b> for the pre-processing recipe:",
            default_text,
        )
        if not proceed:
            return

        cp = self.recipeConfigPars("acdc")
        with open(ini_filepath, "w") as configfile:
            cp.write(configfile)

        self.communicateSavingRecipeFinished(ini_filepath)

    def warnExistingRecipeFile(self, ini_filename):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            "A file with the following name<br><br>"
            f"<code>{ini_filename}</code><br><br>"
            "<b>already exists</b>.<br><br>"
            "Do you want to <b>overwrite</b> the existing file?"
        )
        noButton, yesButton = msg.warning(
            self,
            "File name existing",
            txt,
            buttonsTexts=("No, stop saving process", "Yes, overwrite existing file"),
        )
        return msg.clickedButton == yesButton

    def warnNoAvailableRecipesToLoad(self):
        text = html_utils.paragraph("There are no recipes saved. Sorry about that :(")
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "No recipes saved", text)

    # def selectIniFileToLoadRecipe(self):
    #     import qtpy.compat
    #     ini_filepath = qtpy.compat.getopenfilename(
    #         parent=self,
    #         caption='Select INI file to load pre-processing recipe',
    #         filters='INI (*.ini);;All Files (*)'
    #     )[0]
    #     if not ini_filepath:
    #         return

    #     cp = config.ConfigParser()
    #     cp.read(ini_filepath)
    #     preprocConfigPars = {}
    #     for section in cp.sections():
    #         if not section.startswith('acdc.preprocess'):
    #             continue

    #         preprocConfigPars[section] = cp[section]

    #     if not preprocConfigPars:
    #         return

    #     self.loadRecipe(preprocConfigPars)

    def selectRecipeFilepath(self, recipes_path, recipe_prefix, ext_label, ext):
        availableRecipes = []
        if os.path.exists(recipes_path):
            for file in myutils.listdir(recipes_path):
                if not file.startswith(recipe_prefix):
                    continue
                endname = file.split(f"{recipe_prefix}_")[1]
                availableRecipes.append(endname)

        if not availableRecipes:
            import qtpy.compat

            filepath = qtpy.compat.getopenfilename(
                parent=self,
                caption=f"Select {ext_label} file to load recipe",
                filters=f"{ext_label} (*.{ext});;All Files (*)",
            )[0]
            return filepath or None

        browseButton = widgets.browseFileButton(
            f"Select {ext_label} file...",
            title=f"Select {ext_label} file to load recipe",
            openFolder=False,
            start_dir=myutils.getMostRecentPath(),
            ext={ext_label: f".{ext}"},
        )
        selectRecipeWin = widgets.QDialogListbox(
            "Select recipe",
            "Select recipe to load:\n",
            availableRecipes,
            multiSelection=False,
            allowEmptySelection=False,
            parent=self,
            additionalButtons=(browseButton,),
        )
        browseButton.sigPathSelected.connect(
            partial(
                self.recipeIniFileSelected,
                selectRecipeWin=selectRecipeWin,
                sender=browseButton,
            )
        )
        selectRecipeWin.exec_()
        if selectRecipeWin.cancel:
            return None

        if selectRecipeWin.clickedButton == browseButton:
            return selectRecipeWin.selectedIniFilepath

        selected_endname = selectRecipeWin.selectedItemsText[0]
        filename = f"{recipe_prefix}_{selected_endname}"
        return os.path.join(recipes_path, filename)

    def selectAndLoadRecipe(self):
        filepath = self.selectRecipeFilepath(
            preproc_recipes_path, "preprocessing_recipe", "INI", "ini"
        )
        if filepath is None:
            return
        cp = config.ConfigParser()
        cp.read(filepath)
        preprocConfigPars = {
            s: cp[s] for s in cp.sections() if s.startswith("acdc.preprocess")
        }
        if not preprocConfigPars:
            return
        self.loadRecipe(preprocConfigPars)

    def recipeIniFileSelected(self, ini_filepath, selectRecipeWin=None, sender=None):
        selectRecipeWin.clickedButton = sender
        selectRecipeWin.selectedIniFilepath = ini_filepath
        selectRecipeWin.cancel = False
        selectRecipeWin.close()

    def communicateSavingRecipeFinished(self, ini_filepath):
        text = html_utils.paragraph("Done!<br><br>Pre-processing recipe saved to:")
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(
            self,
            "Pre-processing recipe saved!",
            text,
            commands=(ini_filepath,),
            path_to_browse=os.path.dirname(ini_filepath),
        )

    def addStep(self, is_first=False):
        stepWidgets = {}

        self.row += 1

        step_n = len(self.stepsWidgets) + 1
        label = QLabel(f"Step {step_n}: ")
        self.gridLayout.addWidget(label, self.row, 0)
        stepWidgets["stepLabel"] = label

        selector = widgets.PreProcessingSelector()
        self.gridLayout.addWidget(selector, self.row, 1)
        stepWidgets["selector"] = selector

        setParamsButton = widgets.setPushButton()
        setParamsButton.setToolTip("Set step parameters")
        self.gridLayout.addWidget(setParamsButton, self.row, 2)
        setParamsButton.clicked.connect(partial(self.setParamsStep, selector=selector))
        stepWidgets["setParamsButton"] = setParamsButton

        infoButton = widgets.infoPushButton()
        self.gridLayout.addWidget(infoButton, self.row, 3)
        infoButton.clicked.connect(partial(self.showInfo, selector=selector))
        stepWidgets["infoButton"] = infoButton

        if is_first:
            addButton = widgets.addPushButton()
            self.gridLayout.addWidget(addButton, self.row, 4)
            addButton.clicked.connect(self.addStep)
            stepWidgets["addButton"] = addButton
        else:
            delButton = widgets.delPushButton()
            self.gridLayout.addWidget(delButton, self.row, 4)
            delButton.clicked.connect(self.removeStep)
            delButton.step_n = step_n
            stepWidgets["delButton"] = delButton

        self.row += 1
        selector.row = self.row
        selector.step_n = step_n

        hline = widgets.QHLine()
        self.gridLayout.addWidget(hline, self.row, 0, 1, 6)
        stepWidgets["hline"] = hline
        self.row += 1

        self.stepsWidgets[step_n] = stepWidgets

        selector.sigValuesChanged.connect(self.emitValuesChanged)
        selector.currentTextChanged.connect(
            partial(self.clearInitKwargs, step_n=step_n)
        )

        self.resetStretch()

    def emitValuesChanged(self, functionKwargs, step_n):
        self.stepsWidgets[step_n]["step_kwargs"] = functionKwargs

        recipe = self.recipe(warn=False)
        if recipe is None:
            return

        self.sigValuesChanged.emit(recipe)

    def clearInitKwargs(self, selected_method, step_n=0):
        stepWidgets = self.stepsWidgets[step_n]
        stepWidgets.pop("step_kwargs", None)

    def resetStretch(self):
        for row in range(self.gridLayout.rowCount()):
            self.gridLayout.setRowStretch(row, 0)

        self.gridLayout.setRowStretch(self.gridLayout.rowCount(), 1)
        self.row = self.gridLayout.rowCount() - 1

    def showInfo(self, checked=False, selector=None):
        if selector is None:
            return

        htmlText = selector.htmlInfo()
        htmlText = html_utils.paragraph(htmlText)

        method = selector.currentText()
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, f"Info about `{method}`", htmlText)

    def setParamsStep(
        self, checked=False, selector: "widgets.PreProcessingSelector" = None
    ):
        step_n = selector.step_n
        stepFunctionKwargs = selector.askSetParams(
            df_metadata=self.df_metadata, addApplyButton=self.addApplyButton
        )
        if stepFunctionKwargs is None:
            return

        self.stepsWidgets[step_n]["step_kwargs"] = stepFunctionKwargs

    def removeStep(self, checked=False, step_n=None):
        if step_n is None:
            step_n = self.sender().step_n

        stepWidgets = self.stepsWidgets[step_n]

        stepWidgets["stepLabel"].hide()
        self.gridLayout.removeWidget(stepWidgets["stepLabel"])

        stepWidgets["selector"].hide()
        self.gridLayout.removeWidget(stepWidgets["selector"])

        stepWidgets["infoButton"].hide()
        self.gridLayout.removeWidget(stepWidgets["infoButton"])

        # stepWidgets['addButton'].hide()
        # self.gridLayout.removeWidget(stepWidgets['addButton'])

        stepWidgets["setParamsButton"].hide()
        self.gridLayout.removeWidget(stepWidgets["setParamsButton"])

        stepWidgets["delButton"].hide()
        self.gridLayout.removeWidget(stepWidgets["delButton"])
        self.row -= 1

        stepWidgets["hline"].hide()
        self.gridLayout.removeWidget(stepWidgets["hline"])
        self.row -= 1

        self.stepsWidgets.pop(step_n)

        stepsWidgetsMapper = {1: self.stepsWidgets[1]}
        for i, stepWidgets in enumerate(self.stepsWidgets.values()):
            if i == 0:
                continue
            step_n = i + 1
            label = stepWidgets["stepLabel"]
            label.setText(f"Step {step_n}: ")
            stepWidgets["delButton"].step_n = step_n
            stepWidgets["selector"].step_n = step_n
            stepsWidgetsMapper[step_n] = stepWidgets

        self.stepsWidgets = stepsWidgetsMapper

        self.resetStretch()

    def isChecked(self):
        return self.groupbox.isChecked()

    def warnStepNotInit(self, method):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            f"The parameters for the preprocessing step <b>{method}</b> "
            "were not initialized.<br><br>"
            "Please, click on the corresponding <code>Set step parameters</code> "
            "button to initialize this step (cog icon).<br><br>"
            "Thank you for your patience!"
        )
        msg.warning(self, "Params not initialized!", txt)

    def recipe(self, warn=True):
        recipe = []
        if not self.groupbox.isChecked() and self.groupbox.isCheckable():
            return recipe

        for stepWidgets in self.stepsWidgets.values():
            method = stepWidgets["selector"].currentText()
            step_kwargs = stepWidgets.get("step_kwargs")
            if step_kwargs is None:
                if warn:
                    self.warnStepNotInit(method)
                return

            try:
                init_func = config.PREPROCESS_INIT_MAPPER[method]["function"]
                init_func(**step_kwargs)
            except Exception as err:
                pass

            recipe.append({"method": method, "kwargs": step_kwargs})

        return recipe

    def recipeConfigPars(self, model_name):
        cp = config.ConfigParser()
        if not self.groupbox.isChecked() and self.groupbox.isCheckable():
            return cp

        for s, step in enumerate(self.recipe()):
            section = f"{model_name}.preprocess.step{s + 1}"
            cp[section] = {}
            cp[section]["method"] = step["method"]
            for option, value in step["kwargs"].items():
                cp[section][option] = str(value)
        return cp


class CombineChannelsWidget(PreProcessParamsWidget):
    sigValuesChangedCombineChannels = Signal()

    def __init__(self, channel_names: Iterable[str], parent=None):
        self.channel_names = channel_names

        super().__init__(parent)

        self.parent = parent
        qutils.delete_widget(self.loadLastRecipeButton)
        qutils.delete_widget(self.saveRecipeButton)
        qutils.delete_widget(self.loadRecipeButton)

    def addStep(self, is_first=False):
        stepWidgets = {}

        self.row += 1
        if is_first:
            self.row += 1

        step_n = len(self.stepsWidgets) + 1
        tooltip = "Use this text in the formula"
        if is_first:
            label = QLabel("Formula var")
            label.setToolTip(tooltip)
            self.gridLayout.addWidget(label, self.row - 1, 1)
        name_edit = QLineEdit(text=f"img{step_n}")
        name_edit.setToolTip(tooltip)
        self.gridLayout.addWidget(name_edit, self.row, 1)
        stepWidgets["name_edit"] = name_edit
        name_edit.textChanged.connect(self.emitValuesChanged)

        tooltip = "Select a channel or a segmentation mask"
        if is_first:
            label = QLabel("Channel")
            label.setToolTip(tooltip)
            self.gridLayout.addWidget(label, self.row - 1, 2)
        ch_selector = QComboBox()
        ch_selector.setToolTip(tooltip)
        ch_selector.addItems(self.channel_names)
        self.gridLayout.addWidget(ch_selector, self.row, 2)
        stepWidgets["selector"] = ch_selector
        ch_selector.currentTextChanged.connect(self.setBinarizeCheckableAndNorm)

        # add binarisaion spinbox
        tooltip = (
            "If binarize is selected, the channel will be binarized first, before applying offset and multiplier.\n"
            "If inverse binarize is selected, the channel will be binerized and "
            "then the logical NOT will be applied."
        )
        if is_first:
            label = QLabel("Binarize")
            label.setToolTip(tooltip)
            self.gridLayout.addWidget(label, self.row - 1, 5)
        options = ["No", "binarize", "inverse binarize"]
        self.binarizeCombobox = QComboBox()
        self.binarizeCombobox.addItems(options)
        self.binarizeCombobox.setCurrentIndex(0)
        self.binarizeCombobox.setEnabled(False)
        self.binarizeCombobox.setToolTip(tooltip)
        self.binarizeCombobox.currentIndexChanged.connect(self.emitValuesChanged)
        self.gridLayout.addWidget(self.binarizeCombobox, self.row, 5)
        stepWidgets["binarize"] = self.binarizeCombobox

        tooltip = "Min value of the channel to be normalized to."
        if is_first:
            label = QLabel("Min val")
            label.setToolTip(tooltip)
            self.gridLayout.addWidget(label, self.row - 1, 6)
        self.minValueSpinbox = QDoubleSpinBox()
        self.minValueSpinbox.setRange(-np.inf, np.inf)
        self.minValueSpinbox.setSingleStep(0.1)
        self.minValueSpinbox.setValue(0)
        self.minValueSpinbox.setToolTip(tooltip)

        self.minValueSpinbox.valueChanged.connect(self.emitValuesChanged)
        self.gridLayout.addWidget(self.minValueSpinbox, self.row, 6)
        stepWidgets["minValueSpinbox"] = self.minValueSpinbox

        tooltip = "Max value of the channel to be normalized to."
        if is_first:
            label = QLabel("Max val")
            label.setToolTip(tooltip)
            self.gridLayout.addWidget(label, self.row - 1, 7)
        self.maxValueSpinbox = QDoubleSpinBox()
        self.maxValueSpinbox.setRange(-np.inf, np.inf)
        self.maxValueSpinbox.setSingleStep(0.1)
        self.maxValueSpinbox.setValue(1)
        self.maxValueSpinbox.setToolTip(tooltip)

        self.maxValueSpinbox.valueChanged.connect(self.emitValuesChanged)
        self.gridLayout.addWidget(self.maxValueSpinbox, self.row, 7)
        stepWidgets["maxValueSpinbox"] = self.maxValueSpinbox

        if is_first:
            addButton = widgets.addPushButton()
            self.gridLayout.addWidget(addButton, self.row, 8)
            addButton.clicked.connect(self.addStep)
            stepWidgets["addButton"] = addButton

        else:
            delButton = widgets.delPushButton()
            self.gridLayout.addWidget(delButton, self.row, 8)
            delButton.clicked.connect(self.removeStep)
            delButton.step_n = step_n
            stepWidgets["delButton"] = delButton

        self.row += 1
        ch_selector.row = self.row
        ch_selector.step_n = step_n

        hline = widgets.QHLine()
        self.gridLayout.addWidget(hline, self.row, 0, 1, 8)
        stepWidgets["hline"] = hline
        self.row += 1

        self.stepsWidgets[step_n] = stepWidgets

        self.resetStretch()
        self.sigValuesChangedCombineChannels.emit()
        self.setBinarizeCheckableAndNorm()

    def emitValuesChanged(self, *args):
        self.sigValuesChangedCombineChannels.emit()

    def setBinarizeCheckableAndNorm(self):
        for step_n, stepWidgets in self.stepsWidgets.items():
            binarizeSelector = stepWidgets["binarize"]
            channel = stepWidgets["selector"].currentText()
            if "segm" in channel:
                binarizeSelector.setEnabled(True)
                # set min and max to 0 and 1 and disable
                stepWidgets["minValueSpinbox"].setValue(0)
                stepWidgets["maxValueSpinbox"].setValue(1)
                stepWidgets["minValueSpinbox"].setEnabled(False)
                stepWidgets["maxValueSpinbox"].setEnabled(False)
            else:
                binarizeSelector.setEnabled(False)
                binarizeSelector.setCurrentIndex(0)
                # set min and max to 0 and 1 and enable
                stepWidgets["minValueSpinbox"].setEnabled(True)
                stepWidgets["maxValueSpinbox"].setEnabled(True)

        self.emitValuesChanged()

    def removeStep(self, checked=False, step_n=None):
        if step_n is None:
            step_n = self.sender().step_n

        stepWidgets = self.stepsWidgets[step_n]

        stepWidgets["name_edit"].hide()
        self.gridLayout.removeWidget(stepWidgets["name_edit"])

        stepWidgets["selector"].hide()
        self.gridLayout.removeWidget(stepWidgets["selector"])

        stepWidgets["binarize"].hide()
        self.gridLayout.removeWidget(stepWidgets["binarize"])

        stepWidgets["minValueSpinbox"].hide()
        self.gridLayout.removeWidget(stepWidgets["minValueSpinbox"])

        stepWidgets["maxValueSpinbox"].hide()
        self.gridLayout.removeWidget(stepWidgets["maxValueSpinbox"])

        stepWidgets["delButton"].hide()
        self.gridLayout.removeWidget(stepWidgets["delButton"])

        self.row -= 1

        stepWidgets["hline"].hide()
        self.gridLayout.removeWidget(stepWidgets["hline"])
        self.row -= 1

        self.stepsWidgets.pop(step_n)

        stepsWidgetsMapper = {1: self.stepsWidgets[1]}
        for i, stepWidgets in enumerate(self.stepsWidgets.values()):
            if i == 0:
                continue
            step_n = i + 1
            stepWidgets["delButton"].step_n = step_n
            stepWidgets["selector"].step_n = step_n
            stepsWidgetsMapper[step_n] = stepWidgets

        self.stepsWidgets = stepsWidgetsMapper

        self.resetStretch()
        self.sigValuesChangedCombineChannels.emit()

    def steps(self):
        steps = {}
        if not self.groupbox.isChecked() and self.groupbox.isCheckable():
            return steps

        for step_number, stepWidgets in self.stepsWidgets.items():
            name = stepWidgets["name_edit"].text()
            channel = stepWidgets["selector"].currentText()
            binarize = stepWidgets["binarize"].currentText()
            min_val = stepWidgets["minValueSpinbox"].value()
            max_val = stepWidgets["maxValueSpinbox"].value()
            steps[step_number] = {
                "name": name,
                "channel": channel,
                "binarize": binarize,
                "min_val": min_val,
                "max_val": max_val,
            }

        steps = dict(sorted(steps.items()))
        return steps


class FormulaEditWidget(QWidget):
    sigFormulaChanged = Signal(str, bool)  # formula_str, is_valid

    def __init__(self, variable_names=None, parent=None):
        super().__init__(parent)
        self._variable_names = variable_names or []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self._edit = QLineEdit()
        self._edit.setPlaceholderText("e.g. img1 + img2 * 0.5")
        layout.addWidget(self._edit)

        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setStyleSheet("font-size: 11px;")
        layout.addWidget(self._status_label)

        self._edit.textChanged.connect(self._onTextChanged)
        self._clearStatus()

        self.parent = parent

    def setVariableNames(self, variable_names):
        """Allows setting the variables.

        Parameters
        ----------
        variable_names : list
            list of variable names (strings)
        """

        self._variable_names = variable_names
        self._onTextChanged(self._edit.text())

    def text(self):
        """Returns the current formula text."""
        return self._edit.text()

    def setText(self, text):
        """Sets the formula text."""
        self._edit.setText(text)

    def _clearStatus(self):
        self._status_label.setText("")
        self._status_label.setStyleSheet("font-size: 11px;")

    def _onTextChanged(self, text):
        if not text.strip():
            self._clearStatus()

        success, reconstructed_str = self.checkValidity(self._variable_names)

        if success:
            self._status_label.setText(f"→ {reconstructed_str}")
            self._status_label.setStyleSheet("font-size: 11px; color: green;")
        else:
            self._status_label.setText(reconstructed_str)
            self._status_label.setStyleSheet("font-size: 11px; color: red;")

        self.sigFormulaChanged.emit(text, success)

    def checkValidity(self, variable_names=None):
        if variable_names is None:
            variable_names = self._variable_names
        formula_str = self._edit.text()
        arrays = {name: 1 for name in variable_names}
        success = False
        reconstructed_str = "ERROR"
        forb_ch = self.parent.forbiddenChannels
        if forb_ch:
            stepsWidgets = self.parent.combineChannelsWidget.stepsWidgets
            channels = {
                stepsWidget["selector"].currentText()
                for stepsWidget in stepsWidgets.values()
            }
            if forb_ch.intersection(channels):
                reconstructed_str = (
                    "Channels that are forbidden are not allowed to be used!:\n"
                    f"{forb_ch}"
                )
                return False, reconstructed_str
        if formula_str == "":
            reconstructed_str = "First channel is returned/applied"
            return True, reconstructed_str
        try:
            symbols = {name: sp.Symbol(name) for name in arrays}
            expr = sp.sympify(formula_str, locals=symbols)
            missing = {str(s) for s in expr.free_symbols} - arrays.keys()
            if missing:
                reconstructed_str = f"Missing variables: {missing}"
                return False, reconstructed_str

            if formula_str == "":
                reconstructed_str = ""
                return True, reconstructed_str

            # filter out expressions that have no variables
            if not any(s.is_Symbol for s in expr.free_symbols):
                reconstructed_str = "No variables used"
                return False, reconstructed_str

            reconstructed_str = str(expr)
            success = True
        except Exception as e:
            if "syntax" in str(e):
                reconstructed_str = f"Syntax error"
            else:
                reconstructed_str = str(e)
            success = False
        return success, reconstructed_str


class InitFijiMacroDialog(QBaseDialog):
    def __init__(self, parent=None):
        self.cancel = True

        super().__init__(parent=parent)

        mainLayout = QVBoxLayout()

        infoLabel = QLabel(
            html_utils.paragraph(
                """ 
            Place all the <b>raw microscopy files in a folder without any other 
            file</b><br>
            and provide the following information:
            """
            )
        )
        mainLayout.addWidget(infoLabel)

        gridLayout = QGridLayout()

        row = 0
        label = QLabel("Files internal structure: ")
        gridLayout.addWidget(label, row, 0)
        self.filesStructureCombobox = QComboBox()
        self.filesStructureCombobox.addItems(
            [
                'Positions (aka "series") embedded in the file',
                'Positions (aka "series") separated, one for each file',
                'Positions (aka "series") and channels separated, one for each file',
            ]
        )
        gridLayout.addWidget(self.filesStructureCombobox, row, 1)
        self.filesStructureCombobox.currentTextChanged.connect(
            self.fileStructureChanged
        )
        infoButton = widgets.infoPushButton()
        gridLayout.addWidget(infoButton, row, 2)
        infoButton.clicked.connect(self.showInfoFileStructure)

        row += 1
        label = QLabel("Folder with raw microscopy files: ")
        gridLayout.addWidget(label, row, 0)
        self.folderPathLineEdit = widgets.ElidingLineEdit()
        gridLayout.addWidget(self.folderPathLineEdit, row, 1)
        browseButton = widgets.browseFileButton(openFolder=True)
        gridLayout.addWidget(browseButton, row, 2)
        browseButton.sigPathSelected.connect(
            partial(self.updateFolderPath, lineEdit=self.folderPathLineEdit)
        )
        self.folderPathLineEdit.textChanged.connect(self.srcFolderPathChanged)

        row += 1
        label = QLabel("Destination folder: ")
        gridLayout.addWidget(label, row, 0)
        self.dstfolderPathLineEdit = widgets.ElidingLineEdit()
        gridLayout.addWidget(self.dstfolderPathLineEdit, row, 1)
        browseButton = widgets.browseFileButton(openFolder=True)
        gridLayout.addWidget(browseButton, row, 2)
        browseButton.sigPathSelected.connect(self.dstfolderPathLineEdit.setText)

        row += 1
        label = QLabel("Channel(s) name: ")
        gridLayout.addWidget(label, row, 0)
        self.channelNamesLineEdit = widgets.alphaNumericLineEdit(additionalChars=" ,")
        gridLayout.addWidget(self.channelNamesLineEdit, row, 1)
        checkButton = widgets.TestPushButton("Check")
        gridLayout.addWidget(checkButton, row, 3)
        checkButton.clicked.connect(self.checkChannelNames)
        checkButton.setDisabled(True)
        self.checkButton = checkButton
        infoButton = widgets.infoPushButton()
        gridLayout.addWidget(infoButton, row, 2)
        infoButton.clicked.connect(self.showInfoChannelName)

        buttonsLayout = widgets.CancelOkButtonsLayout()

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
        self.checkButton.setDisabled(not "channels separated" in text)

    def checkChannelNames(self, checked=False):
        proceed = self.validate()
        if not proceed:
            return

        src_folderpath = self.folderPath()
        channel_names = self.channelNames()
        extension = os.listdir(src_folderpath)[0].split(".")[-1]
        basenames = io.move_separate_channels_tiffs_to_pos_folders(
            src_folderpath, channel_names, get_only_basenames=True, extension=extension
        )
        pos_folders_texts = []
        for p, basename in enumerate(basenames):
            pos_folders_texts.append(f"Position_{p + 1}: <code>{basename}</code>")

        pos_folders_html_list = html_utils.to_list(pos_folders_texts, ordered=True)
        text = html_utils.paragraph(
            "The following Position folders will be created based on the provided channel names:<br>"
            f"{pos_folders_html_list}"
        )
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, "Position folders", text)

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
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, "Files structure info", txt)

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
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, "Files structure info", txt)

    def updateFolderPath(self, path, lineEdit=""):
        for file in os.listdir(path):
            if not is_alphanumeric_filename(file):
                msg = widgets.myMessageBox(wrapText=False)
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
                msg.critical(self, "Invalid filename", txt, path_to_browse=path)
                lineEdit.setText("")
                return

        lineEdit.setText(path)

    def warnPathEmpty(self, path_name):
        txt = html_utils.paragraph(f"""
            {path_name} <b>cannot be empty</b>.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Empty folder path", txt)

    def warnSelectedPathDoesNotExist(self, path):
        txt = html_utils.paragraph("""
            The selected path <b>does not exist</b>.<br><br>
            Selected path:
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Folder path does not exist", txt, commands=(path,))

    def warnSelectedPathNotAFolder(self, path):
        txt = html_utils.paragraph("""
            The selected path is <b>not a folder</b>.<br><br>
            Selected path:
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Selected path not a folder", txt, commands=(path,))

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
        msg.warning(self, "Multiple file extensions detected", txt, commands=(path,))

    def warnChannelNamesEmpty(self):
        txt = html_utils.paragraph("""
            Channel(s) name <b>cannot be empty</b>.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Empty channel name", txt)

    def validate(self):
        path = self.folderPath()
        dst_path = self.dstfolderPathLineEdit.text()
        paths = {
            "Source folder": path,
            "Destination folder": dst_path,
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

        files = myutils.listdir(path)
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
        channel_names = self.channelNamesLineEdit.text().split(",")
        channel_names = [ch.strip() for ch in channel_names]
        return channel_names

    def ok_cb(self):
        proceed = self.validate()
        if not proceed:
            return

        self.selectedFolderPath = self.folderPath()
        self.filesStructure = self.filesStructureCombobox.currentText()
        is_multiple_files = self.filesStructure.find("separated") != -1
        is_separate_channels = "channels separated" in self.filesStructure
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


class ImageJRoisToSegmManager(QBaseDialog):
    def __init__(
        self,
        rois_filepath,
        TZYX_shape,
        addUseSamePropsForNextPosButton=False,
        parent=None,
    ):
        import roifile

        self.cancel = True
        super().__init__(parent)

        self.setWindowTitle("ROI Manager")

        mainLayout = QVBoxLayout()

        rois = roifile.roiread(rois_filepath)
        self.rois = {roi.name: roi for roi in rois}

        roisNamesTreeWidget = widgets.TreeWidget()
        roisNamesTreeWidget.setHeaderLabels(["ROI name", "Cell_ID"])
        roisNamesTreeWidget.header().setSectionResizeMode(QHeaderView.ResizeToContents)
        # roisNamesTreeWidget.header().setStretchLastSection(False)
        for r, roi in enumerate(rois):
            item = widgets.TreeWidgetItem()
            item.setText(0, roi.name)
            item.setText(1, str(r + 1))
            roisNamesTreeWidget.addTopLevelItem(item)
        roisNamesTreeWidget.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        roisNamesTreeWidget.selectAll()
        mainLayout.addWidget(QLabel("Select ROIs to convert"))
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
            self.lowZspinbox.setMaximum(SizeZ - 1)

            self.highZspinbox = widgets.SpinBox()
            self.highZspinbox.setMinimum(0)
            self.highZspinbox.setMaximum(SizeZ - 1)
            self.highZspinbox.setValue(SizeZ - 1)

            gridLayout.addWidget(QLabel("Repeat 2D ROIs over z-range: "), 1, 0)

            gridLayout.addWidget(QLabel("Start z-slice"), 0, 1)
            gridLayout.addWidget(self.lowZspinbox, 1, 1)

            gridLayout.addWidget(QLabel("Stop z-slice"), 0, 2)
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
                "Keep the same preferences for all next Positions"
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
        msg.warning(self, "ROIs selection empty", txt)

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
                self.lowZspinbox.value(),
                self.highZspinbox.value() + 1,
            )

        self.cancel = False
        self.close()


class ResizeUtilProps(QBaseDialog):
    def __init__(self, input_path="", parent=None):
        self.cancel = True
        super().__init__(parent)

        self.setWindowTitle("Resize Data Properties")

        mainLayout = QVBoxLayout()

        paramsLayout = QGridLayout()

        self._input_path = input_path

        row = 0
        paramsLayout.addWidget(QLabel("Overwrite raw data: "), row, 0)
        self.overwriteToggle = widgets.Toggle()
        self.overwriteToggle.setChecked(True)
        paramsLayout.addWidget(
            self.overwriteToggle, row, 1, 1, 2, alignment=Qt.AlignCenter
        )

        row += 1
        paramsLayout.addWidget(QLabel("Folder path for resized images: "), row, 0)
        self.folderPathOutControl = widgets.filePathControl(
            browseFolder=True,
            fileManagerTitle="Select folder where to save resized data",
            elide=True,
            startFolder=myutils.getMostRecentPath(),
        )
        self.folderPathOutControl.setDisabled(True)
        paramsLayout.addWidget(self.folderPathOutControl, row, 1, 1, 2)

        row += 1
        paramsLayout.addWidget(QLabel("Text to append to files: "), row, 0)
        self.textToAppendLineEdit = widgets.alphaNumericLineEdit()
        self.textToAppendLineEdit.setAlignment(Qt.AlignCenter)
        self.textToAppendLineEdit.setDisabled(True)
        paramsLayout.addWidget(self.textToAppendLineEdit, row, 1, 1, 2)

        row += 1
        paramsLayout.addWidget(QLabel("Resize mode: "), row, 0)
        self.downScaleRadioButton = QRadioButton("Downscale")
        self.upScaleRadioButton = QRadioButton("Upscale")
        self.downScaleRadioButton.setChecked(True)
        paramsLayout.addWidget(
            self.downScaleRadioButton, row, 1, alignment=Qt.AlignCenter
        )
        paramsLayout.addWidget(
            self.upScaleRadioButton, row, 2, alignment=Qt.AlignCenter
        )

        row += 1
        paramsLayout.addWidget(QLabel("Resize factor: "), row, 0)
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

        # self.textToAppendLineEdit.setText(self._getDefaultTextToAppend())

        self.setLayout(mainLayout)

    def _getDefaultTextToAppend(self):
        rescale_mode = "up" if self.upScaleRadioButton.isChecked() else "down"
        factor = self.factorSpinbox.value()
        text = f"{rescale_mode}scaled_factor_{factor}"
        return text

    def overwriteToggled(self, checked):
        self.folderPathOutControl.setDisabled(checked)
        self.textToAppendLineEdit.setDisabled(checked)
        if checked:
            text = ""
        else:
            text = self._getDefaultTextToAppend()
        self.textToAppendLineEdit.setText(text)

    def warnFolderPathEmpty(self):
        txt = html_utils.paragraph("""
            To prevent overwriting raw data the <code>Folder path for 
            resized images</code> <b>cannot be empty</b>.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Empty folder path", txt)

    def warnTextToAppendEmpty(self):
        txt = html_utils.paragraph("""
            To prevent overwriting raw data the <b>text to append 
            cannot be empty</b>.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, "Empty text to append", txt)

    def ok_cb(self):
        self.expFolderpathOut = self.folderPathOutControl.path()
        self.textToAppend = self.textToAppendLineEdit.text()
        isAccidentalOverwrite = (
            not self.overwriteToggle.isChecked()
            and self.expFolderpathOut == self._input_path
            and not self.textToAppend
        )
        if isAccidentalOverwrite:
            self.warnTextToAppendEmpty()
            return

        if self.textToAppend and not self.textToAppend.startswith("_"):
            self.textToAppend = f"_{self.textToAppend}"

        if self.overwriteToggle.isChecked():
            self.expFolderpathOut = None

        factor = self.factorSpinbox.value()
        self.resizeFactor = (
            factor if self.upScaleRadioButton.isChecked() else 1 / factor
        )

        self.cancel = False
        self.close()


class FucciPreprocessDialog(FunctionParamsDialog):
    def __init__(
        self,
        channel_names,
        df_metadata=None,
        parent=None,
    ):

        from cellacdc.preprocess import fucci_filter

        params_argspecs = myutils.get_function_argspec(fucci_filter)

        super().__init__(
            params_argspecs,
            function_name="FUCCI pre-processing",
            df_metadata=df_metadata,
            parent=parent,
        )

        channelNamesLayout = QGridLayout()

        row = 0
        label = QLabel("First channel name:  ")
        channelNamesLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)
        self.firstChNameWidget = QComboBox()
        self.firstChNameWidget.addItems(channel_names)
        channelNamesLayout.addWidget(self.firstChNameWidget, row, 1)

        row += 1
        label = QLabel("Second channel name:  ")
        channelNamesLayout.addWidget(label, row, 0, alignment=Qt.AlignLeft)
        self.secondChNameWidget = QComboBox()
        self.secondChNameWidget.addItems(channel_names)
        self.secondChNameWidget.setCurrentText(list(channel_names)[1])
        channelNamesLayout.addWidget(self.secondChNameWidget, row, 1)

        channelNamesLayout.setColumnStretch(0, 0)
        channelNamesLayout.setColumnStretch(1, 1)

        self.mainLayout.insertLayout(0, channelNamesLayout)
        self.mainLayout.insertWidget(1, widgets.QHLine())

    def ok_cb(self):
        self.firstChannelName = self.firstChNameWidget.currentText()
        self.secondChannelName = self.secondChNameWidget.currentText()
        super().ok_cb()


class PreProcessRecipeDialog(QBaseDialog):
    sigApplyImage = Signal(object)
    sigApplyZstack = Signal(object)
    sigApplyAllFrames = Signal(object)
    sigApplyAllPos = Signal(object)
    sigPreviewToggled = Signal(bool)
    sigValuesChanged = Signal(list)
    sigSavePreprocData = Signal(object)
    sigClose = Signal(object)

    def __init__(
        self,
        isTimelapse=False,
        isZstack=False,
        isMultiPos=False,
        df_metadata=None,
        addApplyButton=False,
        parent=None,
        hideOnClosing=False,
    ):
        super().__init__(parent=parent)

        self.setWindowTitle("Pre-processing recipe")

        self.cancel = True
        self.hideOnClosing = hideOnClosing

        mainLayout = QVBoxLayout()

        keepInputDataTypeLayout = QHBoxLayout()
        self.keepInputDataTypeToggle = widgets.Toggle()
        self.keepInputDataTypeToggle.setChecked(True)
        self.keepInputDataTypeToggle.toggled.connect(self.emitValuesChanged)

        keepInputDataTypeLayout.addStretch(1)
        keepInputDataTypeLayout.addWidget(QLabel("Keep input data type: "))
        keepInputDataTypeLayout.addWidget(self.keepInputDataTypeToggle)
        keepInputDataTypeInfoButton = widgets.infoPushButton()
        keepInputDataTypeLayout.addWidget(keepInputDataTypeInfoButton)
        keepInputDataTypeInfoButton.clicked.connect(self.showInfoKeepInputDataType)
        self.keepInputDataTypeLayout = keepInputDataTypeLayout

        self.preProcessParamsWidget = PreProcessParamsWidget(
            df_metadata=df_metadata, addApplyButton=addApplyButton, parent=self
        )
        self.preProcessParamsWidget.groupbox.setCheckable(False)

        buttonsLayout = QGridLayout()  # self.preProcessParamsWidget.buttonsLayout
        self.buttonsLayout = buttonsLayout
        self.previewCheckbox = QCheckBox("Preview")
        buttonsLayout.addWidget(self.previewCheckbox, 0, 0)

        # Relocate buttons of PreProcessParamsWidget to this dialog
        pPPWBL = self.preProcessParamsWidget.buttonsLayout
        loadRecipeButtIdx = pPPWBL.indexOf(self.preProcessParamsWidget.loadRecipeButton)
        self.loadRecipeButton = pPPWBL.takeAt(loadRecipeButtIdx).widget()
        buttonsLayout.addWidget(self.loadRecipeButton, 0, 1)

        saveRecipeButtIdx = pPPWBL.indexOf(self.preProcessParamsWidget.saveRecipeButton)
        self.saveRecipeButton = pPPWBL.takeAt(saveRecipeButtIdx).widget()
        buttonsLayout.addWidget(self.saveRecipeButton, 1, 1)

        loadLastRecipeButtIdx = pPPWBL.indexOf(
            self.preProcessParamsWidget.loadLastRecipeButton
        )
        self.loadLastRecipeButton = pPPWBL.takeAt(loadLastRecipeButtIdx).widget()
        buttonsLayout.addWidget(self.loadLastRecipeButton, 1, 0)

        self.loadLastRecipeButton.hide()

        # self.cancelButton = widgets.cancelPushButton('Cancel')
        # buttonsLayout.insertWidget(2, self.cancelButton)
        # buttonsLayout.insertSpacing(3, 20)

        self.allButtons = [
            self.previewCheckbox,
            self.loadRecipeButton,
            self.saveRecipeButton,
        ]
        col = 3
        row = 0
        self.applyCurrentFrameButton = widgets.okPushButton("Apply to displayed image")
        buttonsLayout.addWidget(self.applyCurrentFrameButton, row, col)
        self.applyCurrentFrameButton.clicked.connect(
            partial(self.apply, signal=self.sigApplyImage)
        )
        self.allButtons.append(self.applyCurrentFrameButton)

        infoLayout = QHBoxLayout()
        buttonsHeight = self.applyCurrentFrameButton.sizeHint().height()
        self.loadingCircle = widgets.LoadingCircleAnimation(size=buttonsHeight)
        sp = self.loadingCircle.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.loadingCircle.setSizePolicy(sp)
        self.loadingCircle.setVisible(False)
        infoLayout.addWidget(self.loadingCircle)

        self.infoLabel = QLabel("<i>(Feel free to use Cell-ACDC while waiting)</i>")
        sp = self.infoLabel.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.infoLabel.setSizePolicy(sp)
        self.infoLabel.hide()
        infoLayout.addWidget(self.infoLabel)

        buttonsLayout.addLayout(
            infoLayout, row + 1, 0, 3, 2, alignment=Qt.AlignBottom | Qt.AlignLeft
        )

        if isZstack:
            row += 1
            self.applyAllZslicesButton = widgets.threeDPushButton(
                "Apply to all z-slices of current image"
            )
            buttonsLayout.addWidget(self.applyAllZslicesButton, row, col)
            self.applyAllZslicesButton.clicked.connect(self.applyAllZslices)
            self.allButtons.append(self.applyAllZslicesButton)
        if isTimelapse:
            row += 1
            self.applyAllFramesButton = widgets.futurePushButton("Apply to all frames")
            buttonsLayout.addWidget(self.applyAllFramesButton, row, col)
            self.applyAllFramesButton.clicked.connect(self.applyAllFrames)
            self.allButtons.append(self.applyAllFramesButton)
        if isMultiPos:
            row += 1
            self.applyAllPosButton = widgets.futurePushButton("Apply to all Positions")
            buttonsLayout.addWidget(self.applyAllPosButton, row, col)
            self.applyAllPosButton.clicked.connect(
                partial(self.apply, signal=self.sigApplyAllPos)
            )
            self.allButtons.append(self.applyAllPosButton)

        row += 1
        self.savePreprocButton = widgets.savePushButton("Save pre-processed data...")
        buttonsLayout.addWidget(self.savePreprocButton, row, col)

        self.allButtons.append(self.savePreprocButton)
        self.savePreprocButton.clicked.connect(self.emitSignalSavePreprocData)

        self.previewCheckbox.toggled.connect(self.emitSigPreviewToggled)
        self.preProcessParamsWidget.sigValuesChanged.connect(self.emitValuesChanged)

        # self.cancelButton.clicked.connect(self.close)

        mainLayout.addLayout(keepInputDataTypeLayout)
        mainLayout.addSpacing(20)
        mainLayout.addWidget(self.preProcessParamsWidget)
        mainLayout.addLayout(buttonsLayout)
        self.mainLayout = mainLayout

        self.setLayout(mainLayout)

    def applyAllZslices(self, checked=False):
        # Preview needs to be turned off because we are computing on every
        # z-slice
        self.previewCheckbox.setChecked(False)
        self.apply(signal=self.sigApplyZstack)

    def applyAllFrames(self, checked=False):
        # Preview needs to be turned off because we are computing on all frames
        self.previewCheckbox.setChecked(False)
        self.apply(signal=self.sigApplyAllFrames)

    def emitSigPreviewToggled(self):
        self.sigPreviewToggled.emit(self.previewCheckbox.isChecked())

    def showInfoKeepInputDataType(self):
        txt = html_utils.paragraph("""
            If checked, the data type of the pre-processed data will be 
            the same as the input data type.<br><br>
            This is useful to avoid saving the pre-processed data as 
            floating-point numbers (e.g., 32-bit float) which might 
            increase the file size.<br><br>
            We <b>recommend keeping this option checked</b>.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, "Keep input data type", txt)

    def emitSignalSavePreprocData(self):
        self.sigSavePreprocData.emit(self)

    def emitValuesChanged(self):
        recipe = self.recipe(warn=False)
        if recipe is None:
            return

        self.sigValuesChanged.emit(recipe)

    def setDisabled(self, disabled: bool):
        self.preProcessParamsWidget.setDisabled(disabled)
        self.loadingCircle.setVisible(disabled)
        self.infoLabel.setVisible(disabled)
        for button in self.allButtons:
            try:
                button.setDisabled(disabled)
            except RuntimeError as e:
                printl(traceback.format_exc())
                printl(f"Error: {e}")
                printl(f"Button: {button}")

    def apply(self, checked=False, signal: Signal = None):
        recipe = self.recipe()
        if recipe is None:
            return

        if signal is not None:
            signal.emit(recipe)

        if self.hideOnClosing:
            self.setDisabled(True)
            self.infoLabel.setText(
                f"{self.sender().text().replace('Apply', 'Applying')}...<br>"
                "<i>(Feel free to use Cell-ACDC while waiting)</i>"
            )
        else:
            self.ok_cb()

    def appliedFinished(self):
        self.setDisabled(False)

    def recipe(self, warn=True):
        recipe = self.preProcessParamsWidget.recipe(warn=warn)
        if recipe is None:
            return

        for step in recipe:
            step["keep_input_data_type"] = self.keepInputDataTypeToggle.isChecked()
        return recipe

    def recipeConfigPars(self):
        return self.preProcessParamsWidget.recipeConfigPars("acdc")

    def ok_cb(self):
        if self.hideOnClosing:
            self.hide()
            return

        self.cancel = False
        self.close()

    def close(self):
        super().close()
        self.sigClose.emit(self)


class PreProcessRecipeDialogUtil(PreProcessRecipeDialog):
    def __init__(
        self,
        channel_names: Iterable[str],
        df_metadata=None,
        parent=None,
    ):
        self.cancel = True

        super().__init__(
            isTimelapse=False,
            isZstack=False,
            isMultiPos=False,
            addApplyButton=False,
            df_metadata=df_metadata,
            parent=parent,
            hideOnClosing=False,
        )

        self.listSelector = widgets.listWidget(
            isMultipleSelection=True, minimizeHeight=True
        )
        self.listSelector.addItems(channel_names)
        self.listSelector.setCurrentRow(0)

        self.mainLayout.insertWidget(0, self.listSelector)
        self.mainLayout.insertWidget(0, QLabel("Select channel(s) to pre-process:"))
        self.mainLayout.insertSpacing(2, 10)
        self.mainLayout.insertWidget(2, widgets.QHLine())

        self.savePreprocButton.hide()
        self.previewCheckbox.hide()
        self.applyCurrentFrameButton.setText("Ok")

        buttonsLayout = self.preProcessParamsWidget.buttonsLayout

        saveRecipeButtonIndex = buttonsLayout.indexOf(
            self.preProcessParamsWidget.saveRecipeButton
        )

        if saveRecipeButtonIndex == -1:
            return

        saveRecipeButtonItem = buttonsLayout.takeAt(saveRecipeButtonIndex)

        buttonsLayout.addItem(saveRecipeButtonItem, 0, 2)

    def warnChannelSelectionEmpty(self):
        txt = html_utils.paragraph("""
            You did <b>not select any channel</b>.<br><br>
            <b>Channel selection cannot be empty</b>.<br><br>
            Thank you for your patience!
        """)

    def ok_cb(self):
        selectedChannelItems = self.listSelector.selectedItems()
        if not selectedChannelItems:
            self.warnChannelSelectionEmpty()

        recipe = self.recipe()
        if recipe is None:
            return

        self.selectedRecipe = recipe
        self.selectedChannels = [item.text() for item in selectedChannelItems]

        self.cancel = False
        self.close()


class CombineChannelsSetupDialog(PreProcessRecipeDialog):
    sigApplyImage = Signal(dict, bool, str)
    sigApplyZstack = Signal(dict, bool, str)
    sigApplyAllFrames = Signal(dict, bool, str)
    sigApplyAllPos = Signal(dict, bool, str)
    sigValuesChanged = Signal()
    sigSaveAsSegmCheckboxToggled = Signal(bool)

    # sigApplyAllZslices = Signal(dict, bool, str)
    # sigApplyAllFramesZslices = Signal(dict, bool, str)

    def __init__(
        self,
        channel_names,
        df_metadata=None,
        parent=None,
        hideOnClosing=False,
        isTimelapse=False,
        isZstack=False,
        isMultiPos=False,
    ):

        self.combineChannelsWidget = CombineChannelsWidget(channel_names, parent=self)
        self.warnExistingRecipeFile = self.combineChannelsWidget.warnExistingRecipeFile
        self.communicateSavingRecipeFinished = (
            self.combineChannelsWidget.communicateSavingRecipeFinished
        )
        self.saveRecipeUI = self.combineChannelsWidget.saveRecipeUI
        self.selectRecipeFilepath = self.combineChannelsWidget.selectRecipeFilepath

        super().__init__(
            isTimelapse=isTimelapse,
            isZstack=isZstack,
            isMultiPos=isMultiPos,
            df_metadata=df_metadata,
            parent=parent,
            hideOnClosing=hideOnClosing,
        )

        self.combineChannelsWidget.sigValuesChangedCombineChannels.connect(
            self.emitValuesChangedSteps
        )

        self.segm_blinked = False
        self.validFormula = True  # allow empty formula
        self.forbiddenChannels = set()  # channels that cannot be combined

        self.mainLayout.setSpacing(4)

        self.mainLayout.insertWidget(2, self.combineChannelsWidget)
        self.combineChannelsWidget.groupbox.setCheckable(False)
        self.combineChannelsWidget.groupbox.setTitle(
            "Combine and manipulate channels and/or segmentation files"
        )

        self.formulaEditWidget = FormulaEditWidget(parent=self)
        self._updateFormulaVariableNames()
        self.formulaEditWidget.sigFormulaChanged.connect(self.formulaChanged)
        self.formulaEditWidget.setToolTip(
            'Enter a formula to combine the channels. For example "img1 + img2 * 0.5"'
        )
        self.mainLayout.insertWidget(3, self.formulaEditWidget)

        buttonsLayoutSaveGroup = QGridLayout()

        row = 0
        col = 0
        loadRecipeButton = widgets.OpenFilePushButton("Load saved recipe")
        self.loadRecipeButtonComb = loadRecipeButton
        buttonsLayoutSaveGroup.addWidget(loadRecipeButton, row, col)
        self.loadRecipeButtonComb.clicked.connect(self.selectAndLoadRecipe)

        col += 1
        saveRecipeButton = widgets.savePushButton("Save current recipe")
        self.saveRecipeButtonComb = saveRecipeButton
        buttonsLayoutSaveGroup.addWidget(saveRecipeButton, row, col)
        saveRecipeButton.clicked.connect(self.saveRecipe)
        saveRecipeButton.setToolTip(
            "Save the current recipe to a file\n"
            f"Location: <b>{combine_channels_recipes_path}</b>"
        )

        col += 1
        loadLastRecipeButton = widgets.reloadPushButton("Load last recipe")
        self.loadLastRecipeButtonComb = loadLastRecipeButton
        buttonsLayoutSaveGroup.addWidget(loadLastRecipeButton, row, col)
        self.mainLayout.addLayout(buttonsLayoutSaveGroup)
        loadLastRecipeButton.clicked.connect(self.loadLastRecipe)
        self.setLoadLastRecipe()

        loadLastRecipeButton.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        loadLastRecipeButton.customContextMenuRequested.connect(
            self._showLoadRecipeContextMenu
        )

        self.cancel = True

        self.setWindowTitle("Combine and manipulate channels and/or segmentation files")
        self.preProcessParamsWidget.hide()
        self.mainLayout.removeWidget(self.preProcessParamsWidget)

        self.savePreprocButton.setText("Save combined data...")

        tooltip = (
            "Save as a segmentation file, for example "
            "when combining a binary mask with a segmentation mask."
        )
        label = QLabel("Save as segmentation:")
        self.saveAsSegmlabel = label
        label.setToolTip(tooltip)
        self.saveAsSegmCheckbox = widgets.Toggle()
        self.saveAsSegmCheckbox.setToolTip(tooltip)
        self.saveAsSegmCheckbox.setChecked(False)
        self.saveAsSegmCheckbox.setEnabled(False)
        self.saveAsSegmCheckbox.toggled.connect(self.emitSaveAsSegmCheckboxToggled)

        self.keepInputDataTypeLayout.insertWidget(0, label)
        self.keepInputDataTypeLayout.insertWidget(1, self.saveAsSegmCheckbox)

    def setLoadLastRecipe(self):
        filepath = self._lastRecipePath()
        if not os.path.exists(filepath):
            self.loadLastRecipeButtonComb.setEnabled(False)

    def returLoadSecondLastRecipe(self):
        filepath = self._secondLastRecipePath()
        if not os.path.exists(filepath):
            return False
        return True

    def _showLoadRecipeContextMenu(self, pos):
        menu = QMenu(self)
        action = menu.addAction("Load recipe from before the last one")
        action.triggered.connect(self.loadPreviousRecipe)
        action.setEnabled(self.returLoadSecondLastRecipe())
        menu.exec(self.loadLastRecipeButtonComb.mapToGlobal(pos))

    def loadPreviousRecipe(self):
        filepath = self._secondLastRecipePath()
        if not os.path.exists(filepath):
            return

        self.loadRecipe(filepath)

    def loadLastRecipe(self):
        filepath = self._lastRecipePath()
        if not os.path.exists(filepath):
            return

        self.loadRecipe(filepath)

    def saveLastRecipe(self):
        os.makedirs(combine_channels_recipes_path, exist_ok=True)
        filepath = self._lastRecipePath()

        same = False
        if os.path.exists(filepath):
            steps_curr = self._getSaveRecipyDict()
            with open(filepath, "r") as f:
                steps_prev = json.load(f)
            same = self._recipesMatch(steps_curr, steps_prev)

        if same:
            return

        if os.path.exists(filepath):
            new_filename = self._secondLastRecipePath()
            if os.path.exists(new_filename):
                os.remove(new_filename)
            os.rename(filepath, new_filename)
        self.saveRecipe(filepath=filepath)

    def _recipesMatch(self, steps_curr, steps_prev):
        # Normalize current dict to strings for comparison with JSON-loaded dict
        def normalize(d):
            return {str(k): str(v) for k, v in d.items()}

        for raw_key in steps_curr:
            key = str(raw_key)
            if key not in steps_prev:
                return False
            if key in ("formula", "keep_input_data_type", "save_as_segm"):
                if str(steps_curr[raw_key]) != str(steps_prev[key]):
                    return False
            else:
                step_dict = normalize(steps_curr[raw_key])
                step_dict_prev = steps_prev[key]
                for key2, val2 in step_dict.items():
                    if key2 not in step_dict_prev:
                        return False
                    if val2 != str(step_dict_prev[key2]):
                        return False
        return True

    def _lastRecipePath(self):
        return os.path.join(
            combine_channels_recipes_path, ".last_combine_channels_recipe.json"
        )

    def _secondLastRecipePath(self):
        return os.path.join(
            combine_channels_recipes_path, ".previous_combine_channels_recipe.json"
        )

    def _getSaveRecipyDict(self):
        steps = self.combineChannelsWidget.steps()  # already returns a copy
        formula = self.formulaEditWidget.text()
        steps["formula"] = formula
        steps["keep_input_data_type"] = self.keepInputDataTypeToggle.isChecked()
        steps["save_as_segm"] = self.saveAsSegmCheckbox.isChecked()
        return steps

    def saveRecipe(self, dummy=None, filepath=None):
        os.makedirs(combine_channels_recipes_path, exist_ok=True)

        filepath_provided = filepath is not None
        if not filepath_provided:
            folder_content = myutils.listdir(combine_channels_recipes_path)
            num_recipes = len(folder_content)
            default_text = f"{num_recipes + 1}"
            proceed, filepath = self.saveRecipeUI(
                combine_channels_recipes_path,
                ".json",
                "Save recipe",
                "combine_channels_recipe",
                "Insert a <b>filename</b> for the recipe:",
                default_text,
            )

            if not proceed:
                return

        steps = self._getSaveRecipyDict()

        with open(filepath, "w") as f:
            json.dump(steps, f, indent=2)

        if not filepath_provided:
            self.communicateSavingRecipeFinished(filepath)

    def selectAndLoadRecipe(self):
        filepath = self.selectRecipeFilepath(
            combine_channels_recipes_path, "combine_channels_recipe", "JSON", "json"
        )
        if filepath is None:
            return

        self.loadRecipe(filepath)

    def loadRecipe(self, filepath):
        with open(filepath, "r") as f:
            recipe = json.load(f)

        recipe = dict(sorted(recipe.items()))
        keys_used = set()
        for key, value in recipe.items():
            if key == "formula":
                formula = value
                continue
            if key == "keep_input_data_type":
                self.keepInputDataTypeToggle.setChecked(value)
                continue
            if key == "save_as_segm":
                self.saveAsSegmCheckbox.setChecked(value)
                continue

            name = value["name"]
            channel = value["channel"]
            binarize = value["binarize"]
            min_val = float(value["min_val"])
            max_val = float(value["max_val"])
            key = int(key)
            stepWidgetsNum = len(self.combineChannelsWidget.stepsWidgets)
            if key > stepWidgetsNum:
                self.combineChannelsWidget.addStep()

            stepWidgets = self.combineChannelsWidget.stepsWidgets[key]
            idx = stepWidgets["selector"].findText(channel)
            if idx == -1:
                stepWidgets["selector"].addItem(channel)
                # stepWidgets['selector'].forbiddenItems.add(channel)
                blinker = qutils.QControlBlink(stepWidgets["selector"], qparent=self)
                blinker.start()
                stepWidgets["selector"].blinker = blinker
                self.forbiddenChannels.add(channel)

            stepWidgets["selector"].setCurrentText(channel)
            stepWidgets["name_edit"].setText(name)
            stepWidgets["binarize"].setCurrentText(binarize)
            stepWidgets["minValueSpinbox"].setValue(min_val)
            stepWidgets["maxValueSpinbox"].setValue(max_val)

            keys_used.add(key)

        # remove extra steps
        keys_present = set(range(1, len(self.combineChannelsWidget.stepsWidgets) + 1))
        extra_keys = keys_present - keys_used
        extra_keys = list(extra_keys)
        extra_keys.sort(reverse=True)
        for key in extra_keys:
            self.combineChannelsWidget.removeStep(step_n=key)
            # updates key dynamically so I have to rely that missing indx are always last steps

        # update formula
        self.formulaEditWidget.setText(formula)

        for stepWidgets in self.combineChannelsWidget.stepsWidgets.values():
            combo = stepWidgets["selector"]
            # set forbidden channels red in all steps
            for i in range(combo.count()):
                item = combo.itemText(i)
                if item in self.forbiddenChannels:
                    combo.setItemData(i, QColor("red"), Qt.ForegroundRole)

    def _updateFormulaVariableNames(self):
        names = [
            stepWidgets["name_edit"].text()
            for stepWidgets in self.combineChannelsWidget.stepsWidgets.values()
        ]
        self.formulaEditWidget.setVariableNames(names)

    def formulaChanged(self, formula_str, is_valid):
        self.setButtonsEnabled(is_valid)
        self.validFormula = is_valid
        if is_valid:
            self.sigValuesChanged.emit()

    def setButtonsEnabled(self, enabled):
        for i in range(self.buttonsLayout.count()):
            item = self.buttonsLayout.itemAt(i)
            widget = item.widget()
            if widget is None:
                continue
            if isinstance(widget, QPushButton):
                label = widget.text().lower().rstrip().lstrip()
                if "apply" in label or "save" in label or "ok" in label:
                    if enabled:
                        try:
                            widget.setEnabled(True)
                        except:
                            pass
                    else:
                        try:
                            widget.setDisabled(True)
                        except:
                            pass

    def saveAsSegm(self):
        return self.saveAsSegmCheckbox.isChecked()

    def emitSaveAsSegmCheckboxToggled(self):
        if self.validFormula:
            self.sigSaveAsSegmCheckboxToggled.emit(self.saveAsSegm())

    def autoCheckSaveAsSegmCheckbox(self):
        any_not_seg = False
        for step in self.combineChannelsWidget.steps().values():
            channel = step["channel"]
            if "segm" not in channel:
                any_not_seg = True
                break

        if any_not_seg:
            self.saveAsSegmCheckbox.setChecked(False)
            self.saveAsSegmCheckbox.setEnabled(False)
        else:
            if not self.segm_blinked:
                self.saveAsSegmCheckbox.setEnabled(True)
                self.blinker = qutils.QControlBlink(
                    self.saveAsSegmCheckbox, qparent=self
                )
                self.blinker.start()
                self.segm_blinked = True

    def apply(self, checked=False, signal: Signal = None):
        steps = self.combineChannelsWidget.steps()
        formula = self.formulaEditWidget.text()
        keep_input_dtype = self.keepInputDataTypeToggle.isChecked()
        if not steps or not self.validFormula:
            return

        if signal is not None:
            try:
                signal.emit(steps, formula)
            except TypeError as err:
                signal.emit(steps, keep_input_dtype, formula)

        self.saveLastRecipe()
        if self.hideOnClosing:
            self.setDisabled(True)
            self.infoLabel.setText(
                f"{self.sender().text().replace('Apply', 'Applying')}...<br>"
                "<i>(Feel free to use Cell-ACDC while waiting)</i>"
            )
        else:
            self.ok_cb(saveLastRecipe=False)

    # Not needed anymore since now we funnel all changes to the formulaEditWidget, which then verifies the formula and
    # emits a signal via  formulaChangeda
    # def emitValuesChanged(self):
    #     if not self.validFormula:
    #         return
    #     self.sigValuesChanged.emit()

    def emitValuesChangedSteps(self):
        self.autoCheckSaveAsSegmCheckbox()
        self._updateFormulaVariableNames()

    def ok_cb(self, dummy=None, saveLastRecipe=True):
        if not self.validFormula:
            return

        if saveLastRecipe:
            self.saveLastRecipe()

        self.keepInputDataType = self.keepInputDataTypeToggle.isChecked()
        self.selectedSteps = self.combineChannelsWidget.steps()
        self.formula = self.formulaEditWidget.text()
        self.cancel = False
        self.close()


class CombineChannelsSetupDialogUtil(CombineChannelsSetupDialog):
    def __init__(
        self,
        channel_names,
        df_metadata=None,
        parent=None,
    ):

        super().__init__(channel_names, parent=parent, df_metadata=df_metadata)

        # add int input for number of workers

        self.mainLayout.addSpacing(20)

        qutils.hide_and_delete_layout(self.buttonsLayout)
        buttonsLayout = widgets.CancelOkButtonsLayout()
        self.buttonsLayout = buttonsLayout
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        self.mainLayout.addLayout(buttonsLayout)

        self.nThreadsSpinBox = QSpinBox()
        self.nThreadsSpinBox.setMinimum(1)
        self.nThreadsSpinBox.setValue(4)
        self.nThreadsSpinBox.setToolTip("Number of threads to use for processing")
        self.mainLayout.addWidget(QLabel("Number of threads:"))
        self.mainLayout.addWidget(self.nThreadsSpinBox)


class CombineChannelsSetupDialogGUI(CombineChannelsSetupDialog):
    def __init__(
        self,
        channel_names: Iterable[str],
        df_metadata=None,
        isTimelapse=False,
        isZstack=False,
        isMultiPos=False,
        parent=None,
        hideOnClosing=False,
    ):
        super().__init__(
            channel_names,
            df_metadata=df_metadata,
            isTimelapse=isTimelapse,
            isZstack=isZstack,
            isMultiPos=isMultiPos,
            parent=parent,
            hideOnClosing=hideOnClosing,
        )

        # remove the preprocess buttons, we use the comb version of them
        qutils.delete_widget(self.loadLastRecipeButton)
        qutils.delete_widget(self.saveRecipeButton)
        qutils.delete_widget(self.loadRecipeButton)

        # self.allButtons.remove(self.loadLastRecipeButton)
        self.allButtons.remove(self.saveRecipeButton)
        self.allButtons.remove(self.loadRecipeButton)

        self.previewCheckbox.setChecked(True)
        self.saveAsSegmlabel.setText("Save and view as segmentation")

    def steps(self, return_keepInputDataType=False):
        steps = self.combineChannelsWidget.steps()
        formula = self.formulaEditWidget.text()
        # if not return_keepInputDataType:
        #     return steps, formula

        keep_input_dtype = self.keepInputDataTypeToggle.isChecked()
        return steps, keep_input_dtype, formula


class TestSegmModelInitalDialog(QBaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.cancel = True

        mainLayout = QVBoxLayout()
        entriesLayout = widgets.FormLayout()

        row = 0
        self.startFrameNumberSpinbox = widgets.SpinBox()
        self.startFrameNumberSpinbox.setMinimum(1)

        self.startFrameNumberFormWidget = widgets.formWidget(
            self.startFrameNumberSpinbox,
            labelTextLeft="Start frame number",
            addActivateCheckbox=True,
        )
        entriesLayout.addFormWidget(self.startFrameNumberFormWidget, row=row)

        row += 1
        self.stopFrameNumberSpinbox = widgets.SpinBox()
        self.stopFrameNumberSpinbox.setMinimum(1)

        self.stopFrameNumberFormWidget = widgets.formWidget(
            self.stopFrameNumberSpinbox,
            labelTextLeft="Stop frame number",
            addActivateCheckbox=True,
        )
        entriesLayout.addFormWidget(self.stopFrameNumberFormWidget, row=row)

        row += 1
        self.startZsliceNumberSpinbox = widgets.SpinBox()
        self.startZsliceNumberSpinbox.setMinimum(1)

        self.startZsliceNumberFormWidget = widgets.formWidget(
            self.startZsliceNumberSpinbox,
            labelTextLeft="Start z-slice number",
            addActivateCheckbox=True,
        )
        entriesLayout.addFormWidget(self.startZsliceNumberFormWidget, row=row)

        row += 1
        self.stopZsliceNumberSpinbox = widgets.SpinBox()
        self.stopZsliceNumberSpinbox.setMinimum(1)

        self.stopZsliceNumberFormWidget = widgets.formWidget(
            self.stopZsliceNumberSpinbox,
            labelTextLeft="Stop z-slice number",
            addActivateCheckbox=True,
        )
        entriesLayout.addFormWidget(self.stopZsliceNumberFormWidget, row=row)

        row += 1

        self.isTimelapseToggleFormWidget = widgets.formWidget(
            widgets.Toggle(),
            labelTextLeft="Is timelapse?",
            stretchWidget=False,
            valueGetterName="isChecked",
        )
        entriesLayout.addFormWidget(self.isTimelapseToggleFormWidget, row=row)

        # self.stopFrameNumberSpinbox
        # self.startZsliceNumberSpinbox
        # self.stopZsliceNumberSpinbox
        # self.isTimelapseToggle

        buttonsLayout = widgets.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        mainLayout.addLayout(entriesLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.setLayout(mainLayout)

    def ok_cb(self):
        self.cancel = False

        self.start_frame_n = self.startFrameNumberFormWidget.value()
        self.stop_frame_n = self.stopFrameNumberFormWidget.value()
        self.start_z_slice_n = self.startZsliceNumberFormWidget.value()
        self.stop_z_slice_n = self.stopZsliceNumberFormWidget.value()
        self.is_timelapse = self.isTimelapseToggleFormWidget.value()

        self.close()

# Sibling imports (deferred to avoid import cycles)
from ._base import (
    ArgWidget,
)
from .general import (
    imageViewer,
)
from .measurements import (
    SelectFeaturesRangeDialog,
)
from .metadata import (
    filenameDialog,
)

