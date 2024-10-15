from collections import defaultdict
import os
import sys
import operator
import time
import re
import datetime
import numpy as np
import pandas as pd
import math
import traceback
import logging
import textwrap
import random

from functools import partial
from math import ceil

import skimage.draw
import skimage.morphology

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_agg import FigureCanvasAgg

from qtpy.QtCore import (
    Signal, QTimer, Qt, QPoint, QUrl, Property,
    QPropertyAnimation, QEasingCurve, QLocale,
    QSize, QRect, QPointF, QRect, QPoint, QEasingCurve, QRegularExpression,
    QEvent, QEventLoop, QPropertyAnimation, QObject,
    QItemSelectionModel, QAbstractListModel, QModelIndex,
    QByteArray, QDataStream, QMimeData, QAbstractItemModel, 
    QIODevice, QItemSelection, PYQT6, QRectF
)
from qtpy.QtGui import (
    QFont, QPalette, QColor, QPen, QKeyEvent, QBrush, QPainter,
    QRegularExpressionValidator, QIcon, QPixmap, QKeySequence, QLinearGradient,
    QShowEvent, QDesktopServices, QFontMetrics, QGuiApplication, QLinearGradient,
    QImage, QCursor
)
from qtpy.QtWidgets import (
    QTextEdit, QLabel, QProgressBar, QHBoxLayout, QToolButton, QCheckBox,
    QApplication, QWidget, QVBoxLayout, QMainWindow, QTreeWidgetItemIterator,
    QLineEdit, QSlider, QSpinBox, QGridLayout, QRadioButton,
    QScrollArea, QSizePolicy, QComboBox, QPushButton, QScrollBar,
    QGroupBox, QAbstractSlider, QDoubleSpinBox, QWidgetAction,
    QAction, QTabWidget, QAbstractSpinBox, QToolBar, QStyleOptionSpinBox,
    QStyle, QDialog, QSpacerItem, QFrame, QMenu, QActionGroup,
    QListWidget, QPlainTextEdit, QFileDialog, QListView, QAbstractItemView,
    QTreeWidget, QTreeWidgetItem, QListWidgetItem, QLayout, QStylePainter,
    QGraphicsBlurEffect, QGraphicsProxyWidget, QGraphicsObject
)

import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')

from . import myutils, measurements, is_mac, is_win, html_utils, is_linux
from . import qrc_resources, printl, settings_folderpath
from . import colors, config
from . import html_path
from . import _palettes
from . import load
from . import apps
from . import plot
from . import annotate
from . import urls
from .acdc_regex import float_regex
from .config import PREPROCESS_MAPPER
from . import _base_widgets

LINEEDIT_WARNING_STYLESHEET = _palettes.lineedit_warning_stylesheet()
LINEEDIT_INVALID_ENTRY_STYLESHEET = _palettes.lineedit_invalid_entry_stylesheet()
TREEWIDGET_STYLESHEET = _palettes.TreeWidgetStyleSheet()
LISTWIDGET_STYLESHEET = _palettes.ListWidgetStyleSheet()
BASE_COLOR = _palettes.base_color()
PROGRESSBAR_QCOLOR = _palettes.QProgressBarColor()
PROGRESSBAR_HIGHLIGHTEDTEXT_QCOLOR = _palettes.QProgressBarHighlightedTextColor()
TEXT_COLOR = _palettes.text_float_rgba()

font = QFont()
font.setPixelSize(12)

custom_cmaps_filepath = os.path.join(settings_folderpath, 'custom_colormaps.ini')

def removeHSVcmaps():
    hsv_cmaps = []
    for g, grad in pg.graphicsItems.GradientEditorItem.Gradients.items():
        if grad['mode'] == 'hsv':
            hsv_cmaps.append(g)
    for g in hsv_cmaps:
        del pg.graphicsItems.GradientEditorItem.Gradients[g]

def renamePgCmaps():
    Gradients = pg.graphicsItems.GradientEditorItem.Gradients
    try:
        Gradients['hot'] = Gradients.pop('thermal')
    except KeyError:
        pass
    try:
        Gradients.pop('greyclip')
    except KeyError:
        pass

def _tab20gradient():
    cmap = plt.get_cmap('tab20')
    ticks = [
        (t, tuple([int(v*255) for v in cmap(t)])) for t in np.linspace(0,1,20)
    ]
    gradient = {'ticks': ticks, 'mode': 'rgb'}
    return gradient

def _tab10gradient():
    cmap = plt.get_cmap('tab10')
    ticks = [
        (t, tuple([int(v*255) for v in cmap(t)])) for t in np.linspace(0,1,20)
    ]
    gradient = {'ticks': ticks, 'mode': 'rgb'}
    return gradient

def getCustomGradients(name='image'):
    CustomGradients = {}
    if not os.path.exists(custom_cmaps_filepath):
        return CustomGradients
    
    cp = config.ConfigParser()
    cp.read(custom_cmaps_filepath)
    for section in cp.sections():
        if not section.startswith(f'{name}'):
            continue
        
        cmap_name = section[len(f'{name}.'):]
        CustomGradients[cmap_name] = {'ticks': [], 'mode': 'rgb'}
        for option in cp.options(section):
            value = cp[section][option]
            pos, *rgb = value.split(',')
            rgb = tuple([int(c) for c in rgb])
            pos = float(pos)
            CustomGradients[cmap_name]['ticks'].append((pos, rgb))
    return CustomGradients

def addGradients():
    Gradients = pg.graphicsItems.GradientEditorItem.Gradients
    Gradients['cividis'] = {
        'ticks': [
            (0.0, (0, 34, 78, 255)),
            (0.25, (66, 78, 108, 255)),
            (0.5, (124, 123, 120, 255)),
            (0.75, (187, 173, 108, 255)),
            (1.0, (254, 232, 56, 255))],
        'mode': 'rgb'
    }
    Gradients['cool'] = {
        'ticks': [
            (0.0, (0, 255, 255, 255)),
            (1.0, (255, 0, 255, 255))],
        'mode': 'rgb'
    }
    Gradients['sunset'] = {
        'ticks': [
            (0.0, (71, 118, 148, 255)),
            (0.4, (222, 213, 141, 255)),
            (0.8, (229, 184, 155, 255)),
            (1.0, (240, 127, 97, 255))],
        'mode': 'rgb'
    }
    Gradients['tab20'] = _tab20gradient()
    Gradients['tab10'] = _tab10gradient()
    cmaps = {}
    for name, gradient in Gradients.items():
        ticks = gradient['ticks']
        colors = [tuple([v/255 for v in tick[1]]) for tick in ticks]
        cmaps[name] = LinearSegmentedColormap.from_list(name, colors, N=256)
    return cmaps, Gradients

nonInvertibleCmaps = ['cool', 'sunset', 'bipolar']

renamePgCmaps()
removeHSVcmaps()
cmaps, Gradients = addGradients()
GradientsLabels = Gradients.copy()
GradientsImage = Gradients.copy()

class XStream(QObject):
    _stdout = None
    _stderr = None
    messageWritten = Signal(str)
    
    def flush( self ):
        pass
   
    def fileno( self ):
        return -1
    
    def write(self, msg):
        if not self.signalsBlocked():
            self.messageWritten.emit(msg)
    
    @staticmethod
    def stdout():
        if not XStream._stdout:
            XStream._stdout = XStream()
            sys.stdout = XStream._stdout
        return XStream._stdout
    
    @staticmethod
    def stderr():
        if not XStream._stderr:
            XStream._stderr = XStream()
            sys.stderr = XStream._stderr
        return XStream._stderr

class QtHandler(logging.Handler):
    def __init__(self):
        super().__init__()
    
    def emit(self, record):
        record = self.format(record)
        if record: 
            XStream.stdout().write('%s\n'%record)

class QLog(QPlainTextEdit):
    sigClose = Signal()

    def __init__(self, *args, logger=None):
        super().__init__(*args)
        self.logger = logger
        self.setReadOnly(True)

    def connect(self):
        XStream.stdout().messageWritten.connect(self.writeStdOutput)
        # XStream.stderr().messageWritten.connect(self.writeStdErr)
    
    def writeStdOutput(self, text: str) -> None:
        super().insertPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def writeStdErr(self, text: str) -> None:
        super().insertPlainText(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
        if self.logger is not None:
            self.logger.exception(text)
    
    def insertPlainText(self, text: str) -> None:
        super().insertPlainText(f'{text}\n')
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
    
    def closeEvent(self, event) -> None:
        super().closeEvent(event)
        self.sigClose.emit()

class PushButton(QPushButton):
    def __init__(
            self, *args, icon=None, alignIconLeft=False, 
            flat=False, hoverable=False
        ):
        super().__init__(*args)
        if icon is not None:
            self.setIcon(icon)
        self.alignIconLeft = alignIconLeft
        self._text = None
        if flat:
            self.setFlat(True)
        if hoverable:
            self.installEventFilter(self)
    
    def setRetainSizeWhenHidden(self, retainSize):
        sp = self.sizePolicy()
        sp.setRetainSizeWhenHidden(retainSize)
        self.setSizePolicy(sp)
    
    def eventFilter(self, object, event):
        if event.type() == QEvent.Type.HoverEnter:
            self.setFlat(False)
        elif event.type() == QEvent.Type.HoverLeave:
            self.setFlat(True)
        return False
    
    def show(self):
        text = self.text()
        if not self.alignIconLeft:
            super().show()
            return 

        self._text = text
        self.setStyleSheet('text-align:left;')
        self.setLayout(QGridLayout())
        textLabel = QLabel(self._text)
        textLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        textLabel.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._layout().addWidget(textLabel)
        super().show()
    
    def confirmAction(self):
        self.baseIcon = self.icon()
        self.setIcon(QIcon(':greenTick.svg'))
        QTimer.singleShot(2000, self.resetButton)
    
    def resetButton(self):
        self.setIcon(self.baseIcon)
    
    def setText(self, text):
        if self._text is None:
            super().setText(text)
        else:
            super().setText(self._text)

class LoadPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':fork_lift.svg'))

class mergePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':merge-IDs.svg'))

class okPushButton(PushButton):
    def __init__(self, *args, isDefault=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':yesGray.svg'))
        if isDefault:
            self.setDefault(True)
        # QShortcut(Qt.Key_Return, self, self.click)
        # QShortcut(Qt.Key_Enter, self, self.click)

class SkipPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':skip_arrow.svg'))

class BedPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':bed.svg'))

class BedPlusLabelPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':bed_plus_label.svg'))
        iconH = self.iconSize().height()
        iconW = int(iconH*2.5)
        self.setIconSize(QSize(iconW, iconH))

class NoBedPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':no_bed.svg'))

class NavigatePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':navigate.svg'))

class SwitchPlaneButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':switch_2d_plane.svg'))
        self._planes = ('xy', 'zy', 'zx')
        self._idx = 0
    
    def switchPlane(self):
        self._idx += 1
    
    def setPlane(self, plane):
        self._idx = self._planes.index(plane)
    
    def plane(self):
        return self._planes[self._idx % 3]

    def depthAxes(self):
        plane = self.plane()
        for axes in 'xyz':
            if axes not in plane:
                return axes

class zoomPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':zoom_out.svg'))
    
    def setIconZoomOut(self):
        self.setIcon(QIcon(':zoom_out.svg'))
    
    def setIconZoomIn(self):
        self.setIcon(QIcon(':zoom_in.svg'))

class WarningButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':warning.svg'))

class reloadPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':reload.svg'))

class savePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':file-save.svg'))

class autoPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':cog_play.svg'))

class newFilePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':file-new.svg'))

class helpPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':help.svg'))

class viewPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':eye.svg'))

class infoPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':info.svg'))

class threeDPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':3d.svg'))

class twoDPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':2d.svg'))

class addPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':add.svg'))

class futurePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':arrow_future.svg'))

class currentPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':arrow_current.svg'))

class arrowUpPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        alignIconLeft = kwargs.get('alignIconLeft', False)
        super().__init__(
            *args, icon=QIcon(':arrow-up.svg'), alignIconLeft=alignIconLeft
        )

class arrowDownPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':arrow-down.svg'))

class selectAllPushButton(PushButton):
    sigClicked = Signal(object, bool)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._status = 'deselect'
        self.setIcon(QIcon(':deselect_all.svg'))
        self.setText('Deselect all')
        self.clicked.connect(self.onClicked)
        self.setMinimumWidth(self.sizeHint().width())
    
    def setChecked(self, checked):
        if checked:
            self._status == 'deselect'
        else:
            self._status == 'select'
        self.click()
    
    def onClicked(self):
        if self._status == 'select':
            icon_fn = ':deselect_all.svg'
            self._status = 'deselect'
            checked = True
            text = 'Deselect all'
        else:
            icon_fn = ':select_all.svg'
            text = 'Select all'
            self._status = 'select'
            checked = False
        self.setIcon(QIcon(icon_fn))
        self.setText(text)
        self.sigClicked.emit(self, checked)

class subtractPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':subtract.svg'))

class continuePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':continue.svg'))

class calcPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':calc.svg'))

class playPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':play.svg'))

class stopPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':stop.svg'))

class copyPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':edit-copy.svg'))
        self.clicked.connect(self.onClicked)
        self._text_to_copy = None
    
    def setTextToCopy(self, text):
        self._text_to_copy = text
    
    def onClicked(self):
        self._original_text = self.text()
        if self._text_to_copy is not None:
            cb = QApplication.clipboard()
            cb.clear(mode=cb.Clipboard)
            cb.setText(self._text_to_copy, mode=cb.Clipboard)
            
        super().setText('Copied!')
        self.setIcon(QIcon(':greenTick.svg'))
        QTimer.singleShot(2000, self.resetButton)
    
    def resetButton(self):
        self.setText(self._original_text)
        self.setIcon(QIcon(':edit-copy.svg'))

class OpenFilePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':folder-open.svg'))

class movePushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':folder-move.svg'))

class showInFileManagerButton(PushButton):
    def __init__(self, *args, setDefaultText=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':drawer.svg'))
        self._path_to_browse = None
        if setDefaultText:
            self.setDefaultText()
    
    def setDefaultText(self):
        self._text = myutils.get_show_in_file_manager_text()
        self.setText(self._text)

    def setPathToBrowse(self, path: os.PathLike):
        self._path_to_browse = path
        self.clicked.connect(partial(myutils.showInExplorer, path))
    
        
    
class OpenUrlButton(PushButton):
    def __init__(self, url, *args, **kwargs):
        self._url = url
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':browser.svg'))
        self.clicked.connect(self.openUrl)
    
    def openUrl(self):
        QDesktopServices.openUrl(QUrl(self._url))

class LessThanPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':less_than.svg'))
        flat = kwargs.get('flat')
        if flat is not None:
            self.setFlat(True)

class showDetailsButton(PushButton):
    sigToggled = Signal(bool)
    
    def __init__(self, *args, txt='Show details...', parent=None):
        super().__init__(txt, parent)
        # self.setText(txt)
        self.txt = txt
        self.checkedIcon = QIcon(':hideUp.svg')
        self.uncheckedIcon = QIcon(':showDown.svg')
        self.setIcon(self.uncheckedIcon)
        self.toggled.connect(self.onClicked)
        self.setCheckable(True)
        w = self.sizeHint().width() + 10
        self.setFixedWidth(w)

    def onClicked(self, checked):
        if checked:
            self.setText(self.txt.replace('Show', 'Hide'))
            self.setIcon(self.checkedIcon)
        else:
            self.setText(self.txt)
            self.setIcon(self.uncheckedIcon)
        
        self.sigToggled.emit(checked)

class cancelPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':cancelButton.svg'))

class setPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':cog.svg'))

class TrainPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':train.svg'))

class noPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':no.svg'))

class editPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':edit-id.svg'))

class delPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':bin.svg'))

class eraserPushButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':eraser.svg'))

class CrossCursorPointButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':cross_cursor.svg'))

class browseFileButton(PushButton):
    sigPathSelected = Signal(str)

    def __init__(
            self, *args, ext=None, title='Select file', start_dir='', 
            openFolder=False, **kwargs
        ):
        """PushButton with sigPathSelected Signal to select file or folder

        Parameters
        ----------
        ext : dict or None, optional
            If not None, this is a dictionary of 
            {'FILE NAME': ['.ext1', '.ext2', ...]}. 
            For example, to allow only selection of CSV files, 
            pass {'CSV': ['.csv']}. 
            
            Note that the 'FILE NAME' is arbitrary. Default is None
        title : str, optional
            Title of the File Manager window. Default is 'Select file'
        start_dir : str, optional
            Directory where the File Manager window will initially be open. 
            Default is ''
        openFolder : bool, optional
            If True, allows for selection of folders instead of files. 
            Default is False
        """        
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':folder-open.svg'))
        self.clicked.connect(self.browse)
        
        self._title = title
        self._start_dir = start_dir
        self._openFolder = openFolder
        self._file_types = 'All Files (*)'
        if ext is not None:
            s = ''
            s_li = []
            for name, extensions in ext.items():
                _s = ''
                if isinstance(extensions, str):
                    extensions = [extensions]
                for ext in extensions:
                    _s = f'{_s}*{ext} '
                s_li.append(f'{name} {_s.strip()}')

            self._file_types = ';;'.join(s_li)
            self._file_types = f'{self._file_types};;All Files (*)'

    def setStartPath(self, start_path):
        self._start_dir = start_path
    
    def browse(self):
        if self._openFolder:
            fileDialog = QFileDialog.getExistingDirectory
            args = (self, self._title, self._start_dir)
        else:
            fileDialog = QFileDialog.getOpenFileName
            args = (self, self._title, self._start_dir, self._file_types)
        file_path = fileDialog(*args)
        if not isinstance(file_path, str):
            file_path = file_path[0]
        if file_path:
            self.sigPathSelected.emit(file_path)

def getPushButton(buttonText, qparent=None):
    isCancelButton = (
        buttonText.lower().find('cancel') != -1
        or buttonText.lower().find('abort') != -1
    )
    isYesButton = (
        buttonText.lower().find('yes') != -1
        or buttonText.lower().find('ok') != -1
        or buttonText.lower().find('continue') != -1
        or buttonText.lower().find('recommended') != -1
    )
    isSettingsButton = buttonText.lower().find('set') != -1
    isNoButton = (
        buttonText.replace(' ', '').lower() == 'no'
        or buttonText.lower().find('Do not ') != -1
        or buttonText.lower().find('no, ') != -1
    )
    isDelButton = buttonText.lower().find('delete') != -1
    isAddButton = buttonText.lower().find('add ') != -1
    is3Dbutton = buttonText.find(' 3D ') != -1
    is2Dbutton = buttonText.find(' 2D ') != -1
    isSaveButton = buttonText.lower().find('overwrite') != -1
    isNewFileButton = buttonText.lower().find('rename') != -1
    isTryAgainButton = buttonText.lower().find('try again') != -1

    if isCancelButton:
        button = cancelPushButton(buttonText, qparent)
        if qparent is not None:
            qparent.addCancelButton(button=button)
    elif isYesButton:
        button = okPushButton(buttonText, qparent)
        if qparent is not None:
            qparent.okButton = button
    elif isSettingsButton:
        button = setPushButton(buttonText, qparent)
    elif isNoButton:
        button = noPushButton(buttonText, qparent)
    elif isDelButton:
        button = delPushButton(buttonText, qparent)
    elif isAddButton:
        button = addPushButton(buttonText, qparent)
    elif is3Dbutton:
        button = threeDPushButton(buttonText, qparent)
    elif is2Dbutton:
        button = twoDPushButton(buttonText, qparent)
    elif isSaveButton:
        button = savePushButton(buttonText, qparent)
    elif isNewFileButton:
        button = newFilePushButton(buttonText, qparent)
    elif isTryAgainButton:
        button = reloadPushButton(buttonText, qparent)
    else:
        button = QPushButton(buttonText, qparent)
    
    return button, isCancelButton

def CustomGradientMenuAction(gradient: QLinearGradient, name: str, parent):
    pixmap = QPixmap(100, 15)
    painter = QPainter(pixmap)
    brush = QBrush(gradient)
    painter.fillRect(QRect(0, 0, 100, 15), brush)
    painter.end()
    label = QLabel()
    label.setPixmap(pixmap)
    label.setContentsMargins(1, 1, 1, 1)
    labelName = QLabel(name)
    hbox = QHBoxLayout()
    delButton = delPushButton()
    hbox.addWidget(labelName)
    hbox.addStretch(1)
    hbox.addWidget(label)
    hbox.addWidget(delButton)
    widget = QWidget()
    widget.setLayout(hbox)
    action = QWidgetAction(parent)
    action.name = name
    action.setDefaultWidget(widget)
    action.delButton = delButton
    delButton.action = action
    return action

class ContourItem(pg.PlotCurveItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self._prevData = None
    
    def clear(self):
        try:
            self.setData([], [])
        except AttributeError as e:
            pass
    
    def tempClear(self):
        try:
            self._prevData = [d.copy() for d in self.getData()]
            self.clear()
        except Exception as e:
            pass
    
    def restore(self):
        if self._prevData is not None:
            if self._prevData[0] is not None:
                self.setData(*self._prevData)

class BaseScatterPlotItem(pg.ScatterPlotItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
    
    def tempClear(self):
        try:
            self._prevData = [d.copy() for d in self.getData()]
            self.setData([], [])
        except Exception as e:
            pass
    
    def restore(self):
        if self._prevData is not None:
            if self._prevData[0] is not None:
                self.setData(*self._prevData)

class VerticalSpacerEmptyWidget(QWidget):
    def __init__(self, parent=None, height=5) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self.setFixedHeight(height)

class CustomAnnotationScatterPlotItem(BaseScatterPlotItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

class ElidingLineEdit(QLineEdit):    
    def __init__(self, parent=None, minWidth=None):
        super().__init__(parent)
        self._text = ''
        self._minWidth = minWidth
        if minWidth is not None:
            self.setMinimumWidth(minWidth)
        
        self.textEdited.connect(self.setText)
        self.installEventFilter(self)
        self._elide = True
    
    def setText(self, text: str, width=None, elide=True) -> None:
        if width is None:
            width = self._minWidth
        
        if width is None:
            try:
                textToPrevRatio = len(text)/len(self.text())
                width = round(self.width()*textToPrevRatio)
            except ZeroDivisionError:
                width = self.width()

        if width > self.width():
            width = self.width()
            
        self._text = text
        if not elide or not self._elide:
            super().setText(text)
            return
        
        fm = QFontMetrics(self.font())
        elidedText = fm.elidedText(text, Qt.ElideLeft, width)
        
        super().setText(elidedText)
        self.setToolTip(text)
    
    def text(self):
        return self._text

    def resizeEvent(self, event):
        newWidth = event.size().width()
        self.setText(self._text, width=newWidth)
        event.accept()
    
    def eventFilter(self, a0: 'QObject', a1: 'QEvent') -> bool:
        isFocusIn = a1.type() == QEvent.Type.FocusIn
        if isFocusIn and (self.isReadOnly() or not self.isEnabled()):
            self.clearFocus()
            return True
        return super().eventFilter(a0, a1)
    
    def focusInEvent(self, event):
        super().focusInEvent(event)
        self._elide = False
        self.setText(self._text, elide=False)
        self.setCursorPosition(len(self.text()))

    def focusOutEvent(self, event):
        self._elide = True
        super().focusOutEvent(event)
        self.setText(self._text)

class ValidLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def setInvalidStyleSheet(self):
        self.setStyleSheet(
            'background: #FEF9C3;'
            'border-radius: 4px;'
            'border: 1.5px solid red;'
            'padding: 1px 0px 1px 0px'
        )
    
    def setValidStyleSheet(self):
        self.setStyleSheet('')

class KeepIDsLineEdit(ValidLineEdit):
    sigIDsChanged = Signal(list)
    sigSort = Signal()

    def __init__(self, instructionsLabel, parent=None):
        super().__init__(parent)

        self.validPattern = '^[0-9-, ]+$'
        regExpr = QRegularExpression(self.validPattern)
        self.setValidator(QRegularExpressionValidator(regExpr))

        self.textChanged.connect(self.onTextChanged)
        self.editingFinished.connect(self.onEditingFinished)

        self.instructionsText = instructionsLabel.text()
        self._label = instructionsLabel
    
    def keyPressEvent(self, event) -> None:
        super().keyPressEvent(event)
        if event.text() == ',':
            self.sigSort.emit()
    
    def onTextChanged(self, text):
        IDs = []
        rangesMatch = re.findall('(\d+-\d+)', text)
        if rangesMatch:
            for rangeText in rangesMatch:
                start, stop = rangeText.split('-')
                start, stop = int(start), int(stop)
                IDs.extend(range(start, stop+1))
            text = re.sub('(\d+)-(\d+)', '', text)
        IDsMatch = re.findall('(\d+)', text)
        if IDsMatch:
            for ID in IDsMatch:
                IDs.append(int(ID))
        self.IDs = sorted(list(set(IDs)))
        self.sigIDsChanged.emit(self.IDs)
    
    def onEditingFinished(self):
        self.sigSort.emit()
    
    def warnNotExistingID(self):
        self.setInvalidStyleSheet()
        self._label.setText(
            '  Some of the IDs are not existing --> they will be IGNORED'
        )
        self._label.setStyleSheet('color: red')

    def setInstructionsText(self):
        self.setValidStyleSheet()
        self._label.setText(self.instructionsText)
        self._label.setStyleSheet('')

class ScrollBar(QScrollBar):
    def __init__(self, *args):
        super().__init__(*args)
        self.installEventFilter(self)
        self.setContextMenuPolicy(Qt.NoContextMenu)
    
    def eventFilter(self, object, event) -> bool:
        if event.type() == QEvent.Type.Wheel:
            return True
        elif event.type() == QEvent.Type.MouseButtonPress:
            # Filter right-click to prevent context menu
            return event.button() == Qt.MouseButton.RightButton
        elif event.type() == QEvent.Type.MouseButtonRelease:
            # Filter right-click to prevent context menu
            return event.button() == Qt.MouseButton.RightButton
        return False

class _ReorderableListModel(QAbstractListModel):
    '''
    ReorderableListModel is a list model which implements reordering of its
    items via drag-n-drop
    '''
    dragDropFinished = Signal()

    def __init__(self, items, parent=None):
        QAbstractItemModel.__init__(self, parent)
        self.nodes = items
        self.lastDroppedItems = []
        self.pendingRemoveRowsAfterDrop = False

    def rowForItem(self, text):
        '''
        rowForItem method returns the row corresponding to the passed in item
        or None if no such item exists in the model
        '''
        try:
            row = self.nodes.index(text)
        except ValueError:
            return None
        return row

    def index(self, row, column, parent):
        if row < 0 or row >= len(self.nodes):
            return QModelIndex()
        return self.createIndex(row, column)

    def parent(self, index):
        return QModelIndex()

    def rowCount(self, index):
        if index.isValid():
            return 0
        return len(self.nodes)

    def data(self, index, role):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            row = index.row()
            if row < 0 or row >= len(self.nodes):
                return None
            return self.nodes[row]
        elif role == Qt.SizeHintRole:
            return QSize(48, 32)
        else:
            return None

    def supportedDropActions(self):
        return Qt.MoveAction

    def flags(self, index):
        if not index.isValid():
            return Qt.ItemIsEnabled
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | \
               Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled

    def insertRows(self, row, count, index):
        if index.isValid():
            return False
        if count <= 0:
            return False
        # inserting 'count' empty rows starting at 'row'
        self.beginInsertRows(QModelIndex(), row, row + count - 1)
        for i in range(0, count):
            self.nodes.insert(row + i, '')
        self.endInsertRows()
        return True

    def removeRows(self, row, count, index):
        if index.isValid():
            return False
        if count <= 0:
            return False
        num_rows = self.rowCount(QModelIndex())
        self.beginRemoveRows(QModelIndex(), row, row + count - 1)
        for i in range(count, 0, -1):
            self.nodes.pop(row - i + 1)
        self.endRemoveRows()

        if self.pendingRemoveRowsAfterDrop:
            '''
            If we got here, it means this call to removeRows is the automatic
            'cleanup' action after drag-n-drop performed by Qt
            '''
            self.pendingRemoveRowsAfterDrop = False
            self.dragDropFinished.emit()

        return True

    def setData(self, index, value, role):
        if not index.isValid():
            return False
        if index.row() < 0 or index.row() > len(self.nodes):
            return False
        self.nodes[index.row()] = str(value)
        self.dataChanged.emit(index, index)
        return True

    def mimeTypes(self):
        return ['application/vnd.treeviewdragdrop.list']

    def mimeData(self, indexes):
        mimedata = QMimeData()
        encoded_data = QByteArray()
        stream = QDataStream(encoded_data, QIODevice.WriteOnly)
        for index in indexes:
            if index.isValid():
                text = self.data(index, 0)
        stream << QByteArray(text.encode('utf-8'))
        mimedata.setData('application/vnd.treeviewdragdrop.list', encoded_data)
        return mimedata

    def dropMimeData(self, data, action, row, column, parent):
        if action == Qt.IgnoreAction:
            return True
        if not data.hasFormat('application/vnd.treeviewdragdrop.list'):
            return False
        if column > 0:
            return False

        num_rows = self.rowCount(QModelIndex())
        if num_rows <= 0:
            return False

        if row < 0:
            if parent.isValid():
                row = parent.row()
            else:
                return False

        encoded_data = data.data('application/vnd.treeviewdragdrop.list')
        stream = QDataStream(encoded_data, QIODevice.ReadOnly)

        new_items = []
        rows = 0
        while not stream.atEnd():
            text = QByteArray()
            stream >> text
            text = bytes(text).decode('utf-8')
            index = self.nodes.index(text)
            new_items.append((text, index))
            rows += 1

        self.lastDroppedItems = []
        for (text, index) in new_items:
            target_row = row
            if index < row:
                target_row += 1
            self.beginInsertRows(QModelIndex(), target_row, target_row)
            self.nodes.insert(target_row, self.nodes[index])
            self.endInsertRows()
            self.lastDroppedItems.append(text)
            row += 1

        self.pendingRemoveRowsAfterDrop = True
        return True

class _SelectionModel(QItemSelectionModel):
    def __init__(self, parent=None, isSingleSelection=False):
        QItemSelectionModel.__init__(self, parent)
        self.isSingleSelection = isSingleSelection

    def onModelItemsReordered(self):
        new_selection = QItemSelection()
        new_index = QModelIndex()
        for item in self.model().lastDroppedItems:
            row = self.model().rowForItem(item)
            if row is None:
                continue
            new_index = self.model().index(row, 0, QModelIndex())
            new_selection.select(new_index, new_index)

        self.clearSelection()
        flags = (
            QItemSelectionModel.SelectionFlag.ClearAndSelect 
            | QItemSelectionModel.SelectionFlag.Rows 
            | QItemSelectionModel.SelectionFlag.Current
        )
        self.select(new_selection, flags)
        self.setCurrentIndex(new_index, flags)
        if not self.isSingleSelection:
            self.reset()

class ReorderableListView(QListView):
    def __init__(
            self, items=None, parent=None, isSingleSelection=False
        ) -> None:
        super().__init__(parent)
        if items is None:
            items = []

        self.isSingleSelection = isSingleSelection
        self._model = _ReorderableListModel(items)
        self._selectionModel = _SelectionModel(self._model)
        self._model.dragDropFinished.connect(
            self._selectionModel.onModelItemsReordered
        )
        self.setModel(self._model)
        self.setSelectionModel(self._selectionModel)
        self.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.setDragDropOverwriteMode(False)
        styleSheet = (f"""
            QListView {{
                selection-background-color: rgba(200, 200, 200, 0.30);
                selection-color: black;
                show-decoration-selected: 1;
            }}
            QListView::item {{
                border-bottom: 1px solid rgba(180, 180, 180, 0.5);
            }}
            QListView::item:hover {{
                background-color: rgba(200, 200, 200, 0.30);
            }}
        """)
        self.setStyleSheet(styleSheet)
    
    def setItems(self, items):
        self._model.nodes = items
    
    def items(self):
        return self._model.nodes
    
    # def mouseReleaseEvent(self, e: QMouseEvent) -> None:
    #     super().mouseReleaseEvent(e)
    #     self._selectionModel.reset()

class QDialogListbox(QDialog):
    sigSelectionConfirmed = Signal(list)

    def __init__(
            self, title, text, items, cancelText='Cancel',
            multiSelection=True, parent=None,
            additionalButtons=(), includeSelectionHelp=False,
            allowSingleSelection=True, preSelectedItems=None, 
            allowEmptySelection=True
        ):
        self.cancel = True
        items = list(items)
        
        super().__init__(parent)
        self.setWindowTitle(title)
        
        if preSelectedItems is None:
            preSelectedItems = (items[0],)

        self.allowSingleSelection = allowSingleSelection
        self.allowEmptySelection = allowEmptySelection

        mainLayout = QVBoxLayout()
        topLayout = QVBoxLayout()
        bottomLayout = QHBoxLayout()

        self.mainLayout = mainLayout

        label = QLabel(text)
        _font = QFont()
        _font.setPixelSize(13)
        label.setFont(_font)
        # padding: top, left, bottom, right
        label.setStyleSheet("padding:0px 0px 3px 0px;")
        topLayout.addWidget(label, alignment=Qt.AlignCenter)

        if includeSelectionHelp:
            selectionHelpLabel = QLabel()
            txt = html_utils.paragraph("""<br>
                <code>Ctrl+Click</code> <i>to select multiple items</i><br>
                <code>Shift+Click</code> <i>to select a range of items</i><br>
            """)
            selectionHelpLabel.setText(txt)
            topLayout.addWidget(label, alignment=Qt.AlignCenter)

        listBox = listWidget()
        listBox.setFont(_font)
        listBox.addItems(items)            
        if multiSelection:
            listBox.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        else:
            listBox.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        listBox.setCurrentRow(0)
        for i in range(listBox.count()):
            item = listBox.item(i)
            item.setSelected(item.text() in preSelectedItems)
            
        self.listBox = listBox
        if not multiSelection:
            listBox.itemDoubleClicked.connect(self.ok_cb)
        topLayout.addWidget(listBox)

        if cancelText.lower().find('cancel') != -1:
            cancelButton = cancelPushButton(cancelText)
        else:
            cancelButton = QPushButton(cancelText)
        okButton = okPushButton(' Ok ')

        bottomLayout.addStretch(1)
        bottomLayout.addWidget(cancelButton)
        bottomLayout.addSpacing(20)

        if additionalButtons:
            self._additionalButtons = []
            for button in additionalButtons:
                if isinstance(button, str):
                    _button, isCancelButton = getPushButton(button)
                    self._additionalButtons.append(_button)
                    bottomLayout.addWidget(_button)
                    _button.clicked.connect(self.ok_cb)
                else:
                    bottomLayout.addWidget(button)

        bottomLayout.addWidget(okButton)
        bottomLayout.setContentsMargins(0, 10, 0, 0)

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)
        self.setLayout(mainLayout)

        # Connect events
        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.cancel_cb)

        if multiSelection:
            listBox.itemClicked.connect(self.onItemClicked)
            listBox.itemSelectionChanged.connect(self.onItemSelectionChanged)

        self.setStyleSheet(LISTWIDGET_STYLESHEET)
        self.areItemsSelected = [
            listBox.item(i).isSelected() for i in range(listBox.count())
        ]
        self.setFont(font)
    
    def keyPressEvent(self, event) -> None:
        mod = event.modifiers()
        if mod == Qt.ShiftModifier or mod == Qt.ControlModifier:
            self.listBox.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        elif event.key() == Qt.Key_Escape:
            self.listBox.clearSelection()
            event.ignore()
            return
        super().keyPressEvent(event)
    
    def onItemSelectionChanged(self):
        if not self.listBox.selectedItems():
            self.areItemsSelected = [
                False for i in range(self.listBox.count())
            ]
    
    def onItemClicked(self, item):
        mod = QGuiApplication.keyboardModifiers()
        if mod == Qt.ShiftModifier or mod == Qt.ControlModifier:
            self.listBox.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            return
        
        self.listBox.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        itemIdx = self.listBox.row(item)
        wasSelected = self.areItemsSelected[itemIdx]
        if wasSelected:
            item.setSelected(False)
        
        self.areItemsSelected = [
            self.listBox.item(i).isSelected() 
            for i in range(self.listBox.count())
        ]
        # self.listBox.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # else:
        #     selectedItems.append(item)
        
        # self.listBox.clearSelection()
        # for i in range(self.listBox.count()):
        #     item = self.listBox.item(i).setSelected(True)
        
        # print(self.listBox.selectedItems())
    
    def setSelectedItems(self, itemsTexts):
        for i in range(self.listBox.count()):
            item = self.listBox.item(i)
            if item.text() in itemsTexts:
                item.setSelected(True)
        self.listBox.update()

    def warnSelectionEmpty(self):
        msg = myMessageBox(wrapText=False, showCentered=False)
        txt = html_utils.paragraph(
            'You need to <b>select at least one item!</b>.<br><br>'
            'Use <code>Ctrl+Click</code> to select multiple items<br>, or<br>'
            '<code>Shift+Click</code> to select a range of items'
        )
        msg.warning(self, 'Selection cannot be empty!', txt)
    
    def ok_cb(self, checked=False):
        self.clickedButton = self.sender()
        self.cancel = False
        selectedItems = self.listBox.selectedItems()
        self.selectedItemsText = [item.text() for item in selectedItems]
        if not self.allowSingleSelection and len(self.selectedItemsText) < 2:
            msg = myMessageBox(wrapText=False, showCentered=False)
            txt = html_utils.paragraph(
                'You need to <b>select two or more items</b>.<br><br>'
                'Use <code>Ctrl+Click</code> to select multiple items<br>, or<br>'
                '<code>Shift+Click</code> to select a range of items'
            )
            msg.warning(self, 'Select two or more items', txt)
            return
        
        if not self.allowEmptySelection:
            self.warnSelectionEmpty()
            return
        
        self.sigSelectionConfirmed.emit(self.selectedItemsText)
        self.close()

    def cancel_cb(self, event):
        self.cancel = True
        self.selectedItemsText = None
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

        self.listW = QDialogListbox(
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

class filePathControl(QFrame):
    def __init__(
            self, parent=None, browseFolder=False, 
            fileManagerTitle='Select file', 
            validExtensions=None, 
            startFolder='', 
            elide=False
        ):
        super().__init__(parent)

        layout = QHBoxLayout()
        if elide:
            self.le = ElidingLineEdit()
        else:
            self.le = QLineEdit()
            
        self.browseButton = browseFileButton(
            openFolder=browseFolder, title=fileManagerTitle, 
            ext=validExtensions, start_dir=startFolder
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

    def setTextTooltip(self):
        self.le.setToolTip(self.le.text())
    
    def path(self):
        return self.le.text()
    
    def showEvent(self, a0: QShowEvent) -> None:
        self.le.setFixedHeight(self.browseButton.height())
        return super().showEvent(a0)

class FolderPathControl(filePathControl):
    def __init__(self, **kwargs):
        super().__init__(
            browseFolder=True, 
            fileManagerTitle='Select folder', 
            **kwargs
        )

class QHWidgetSpacer(QWidget):
    def __init__(self, width=10, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(width)

class QVWidgetSpacer(QWidget):
    def __init__(self, height=10, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(height)

class QHLine(QFrame):
    def __init__(self, shadow='Sunken', parent=None, color=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(getattr(QFrame, shadow))
        if color is not None:
            self.setColor(color)
    
    def setColor(self, color):
        qcolor = pg.mkColor(color)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.WindowText, qcolor)
        self.setPalette(pal)

class QVLine(QFrame):
    def __init__(self, shadow='Plain', parent=None, color=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFrameShadow(getattr(QFrame.Shadow, shadow))
        if color is not None:
            self.setColor(color)
    
    def setColor(self, color):
        qcolor = pg.mkColor(color)
        pal = self.palette()
        pal.setColor(QPalette.ColorRole.WindowText, qcolor)
        self.setPalette(pal)

class VerticalResizeHline(QFrame):
    dragged = Signal(object)
    clicked = Signal(object)
    released = Signal(object)

    def __init__(self):
        super().__init__()
        self.setCursor(Qt.SplitVCursor)
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Sunken)
        self.installEventFilter(self)
        self.isMousePressed = False
        self._height = 4
        self.setMinimumHeight(self._height)
    
    def mousePressEvent(self, event) -> None:
        self.isMousePressed = True
        self.clicked.emit(event)
        return super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event) -> None:
        self.dragged.emit(event)
        return super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event) -> None:
        self.isMousePressed = False
        self.released.emit(event)
        return super().mouseReleaseEvent(event)
    
    def eventFilter(self, object, event):
        if event.type() == QEvent.Type.Enter:
            self.setLineWidth(0)
            self.setMidLineWidth(self._height)
            pal = self.palette()
            pal.setColor(QPalette.ColorRole.WindowText, QColor(BASE_COLOR))
            self.setPalette(pal)
            # self.setStyleSheet('background-color: #4d4d4d') 
        elif event.type() == QEvent.Type.Leave:
            self.setMidLineWidth(0)
            self.setLineWidth(1)
        return False

class GroupBox(QGroupBox):
    def __init__(self, *args, keyPressCallback=None):
        super().__init__(*args)
        self.keyPressCallback = None
        self.setFocusPolicy(Qt.NoFocus)
    
    def keyPressEvent(self, event) -> None:
        event.ignore()
        if self.keyPressCallback is None:
            return

        self.keyPressCallback()

class CheckBox(QCheckBox):
    def __init__(self, *args, keyPressCallback=None):
        super().__init__(*args)
        self.keyPressCallback = None
        self.setFocusPolicy(Qt.NoFocus)
    
    def keyPressEvent(self, event) -> None:
        event.ignore()
        if self.keyPressCallback is None:
            return

        self.keyPressCallback()

class ScrollArea(QScrollArea):
    sigLeaveEvent = Signal()

    def __init__(
            self, parent=None, resizeVerticalOnShow=False, 
            dropArrowKeyEvents=False
        ) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.containerWidget = None
        self.resizeVerticalOnShow = resizeVerticalOnShow
        self.isOnlyVertical = False
        self.dropArrowKeyEvents = dropArrowKeyEvents
    
    def setVerticalLayout(self, layout, widget=None):
        if widget is None:
            self.containerWidget = QWidget()
        else:
            self.containerWidget = widget
        self.containerWidget.setLayout(layout)
        self.containerWidget.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.setWidget(self.containerWidget)
        self.containerWidget.installEventFilter(self)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.isOnlyVertical = True
    
    def setWidget(self, widget):
        self.containerWidget = widget
        super().setWidget(widget)
    
    def _resizeHorizontal(self):
        self.setMinimumWidth(
            self.containerWidget.minimumSizeHint().width()
            + self.verticalScrollBar().width()
        )
    
    def minimumHeightNoScrollbar(self) -> int:
        height = (
            self.containerWidget.minimumSizeHint().height()
            + self.horizontalScrollBar().height()
        )
        return height
    
    def _resizeVertical(self):
        height = (
            self.containerWidget.minimumSizeHint().height()
            + self.horizontalScrollBar().height()
        )
        self.containerWidget.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )

        self.setFixedHeight(height)

    def eventFilter(self, object, event: QEvent):
        if event.type() == QEvent.Type.Leave:
            self.sigLeaveEvent.emit()

        if object != self.containerWidget:
            return False
        
        isResize = event.type() == QEvent.Type.Resize
        isShow = event.type() == QEvent.Type.Show
        if isResize and self.isOnlyVertical:
            self._resizeHorizontal()
        elif isShow and self.resizeVerticalOnShow:
            self._resizeVertical()
        return False

class QClickableLabel(QLabel):
    clicked = Signal(object)

    def __init__(self, parent=None):
        self._parent = parent
        super().__init__(parent)
        self._checkableItem = None
    
    def setCheckableItem(self, widget):
        self._checkableItem = widget

    def mousePressEvent(self, event):
        self.clicked.emit(self)
        if self._checkableItem is not None:
            status = not self._checkableItem.isChecked()
            self._checkableItem.setChecked(status)

    def setChecked(self, checked):
        self._checkableItem.setChecked(checked)

class QCenteredComboBox(QComboBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)

        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.lineEdit().setAlignment(Qt.AlignCenter)
        self.lineEdit().installEventFilter(self)

        self.currentIndexChanged.connect(self.centerItems)

        self._isPopupVisibile = False
    
    def centerItems(self, idx):
        for i in range(self.count()):
            self.setItemData(i, Qt.AlignCenter, Qt.TextAlignmentRole)
    
    def eventFilter(self, lineEdit, event):
        # Reimplement show popup on click
        if event.type() == QEvent.Type.MouseButtonPress and self.isEnabled():
            if self._isPopupVisibile:
                self.hidePopup()
                self._isPopupVisibile = False
            else:
                self.showPopup()
                self._isPopupVisibile = True
            return True
        return False

class AlphaNumericComboBox(QCenteredComboBox):
    def __init__(self, parent=None) -> None:
        super().__init__(parent=parent)
    
    def addItems(self, items):
        self._dtype = type(items[0])
        super().addItems([str(item) for item in items])
    
    def setCurrentValue(self, value):
        super().setCurrentText(str(value))
    
    def currentValue(self):
        return self._dtype(super().currentText())

class statusBarPermanentLabel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.rightLabel = QLabel('')
        self.leftLabel = QLabel('')

        layout = QHBoxLayout()
        layout.addWidget(self.leftLabel)
        layout.addStretch(10)
        layout.addWidget(self.rightLabel)

        self.setLayout(layout)

class listWidget(QListWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.itemHeight = None
        self.setStyleSheet(LISTWIDGET_STYLESHEET)
        self.setFont(font)
    
    def setSelectedAll(self, selected):
        for i in range(self.count()):
            self.item(i).setSelected(selected)
    
    def addItems(self, labels) -> None:
        super().addItems(labels)
        if self.itemHeight is None:
            return
        self.setItemHeight()
    
    def addItem(self, text):
        super().addItem(text)
        if self.itemHeight is None:
            return
        self.setItemHeight()
    
    def setItemHeight(self, height=40):
        self.itemHeight = height
        for i in range(self.count()):
            item = self.item(i)
            item.setSizeHint(QSize(0, height))

class OrderableListWidget(QWidget):
    sigEnterEvent = Signal(object)
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._labels = []
    
    def setParentItem(self, item):
        self._item = item
    
    def setLabelsColor(self, selected):
        if selected:
            stylesheet = 'color : black'
        else:
            stylesheet = ''
            
        for label in self._labels:
            label.setStyleSheet(stylesheet)
    
    def enterEvent(self, event):
        super().enterEvent(event)
        self.setLabelsColor(True)
        self.sigEnterEvent.emit(self._item)
    
    # def leaveEvent(self, event):
    #     super().leaveEvent(event)
    #     self.setLabelsColor(self._item.isSelected())
    #     printl('leave', self._item.isSelected())
    
    def addLabel(self, label):
        self._labels.append(label)

class OrderableList(listWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)
        self.itemEntered.connect(self.onItemEntered)
    
    def onItemEntered(self, enteredItem):
        enteredRow = self.row(enteredItem)
        for i in range(self.count()):
            item = self.item(i)
            item._container.setLabelsColor(i == enteredRow or item.isSelected())
    
    def leaveEvent(self, event):
        super().leaveEvent(event)
        for i in range(self.count()):
            item = self.item(i)
            item._container.setLabelsColor(item.isSelected())
    
    def addItems(self, items):
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        nr_items = len(items)
        nn = [str(n) for n in range(1, nr_items+1)]
        for i, item in enumerate(items):
            itemW = QListWidgetItem()
            itemContainer = OrderableListWidget()
            itemContainer.setParentItem(itemW)
            itemText = QLabel(item)
            tableNrLabel = QLabel('| Table nr.')
            itemContainer.addLabel(tableNrLabel)
            itemContainer.addLabel(itemText)
            itemLayout = QHBoxLayout()
            itemNumberWidget = QComboBox()
            itemNumberWidget.addItems(nn)
            itemLayout.addWidget(itemText)
            itemLayout.addWidget(tableNrLabel)
            itemLayout.addWidget(itemNumberWidget)
            itemContainer.setLayout(itemLayout)
            itemLayout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
            itemW.setSizeHint(itemContainer.sizeHint())
            self.addItem(itemW)
            self.setItemWidget(itemW, itemContainer)
            itemW._text = item
            itemW._nrWidget = itemNumberWidget
            itemW._container = itemContainer
            itemNumberWidget.setDisabled(True)
            itemNumberWidget.textActivated.connect(self.onTextActivated)
            itemNumberWidget._currentNr = 1
            itemNumberWidget.row = i
            itemContainer.sigEnterEvent.connect(self.onItemEntered)
        
        self.itemSelectionChanged.connect(self.onItemSelectionChanged)
        
    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            self.clearSelection()
            event.ignore()
            return
        super().keyPressEvent(event)
    
    def updateNr(self):
        for i in range(self.count()):
            item = self.item(i)
            item._currentNr = int(item._nrWidget.currentText())
    
    def onItemSelectionChanged(self):
        for i in range(self.count()):
            item = self.item(i)
            item._container.setLabelsColor(item.isSelected())
            item._nrWidget.setDisabled(not item.isSelected())
            if item._nrWidget.currentText() != '1':
                item._nrWidget.setCurrentText('1')
                item._currentNr = 1
        
        for i, item in enumerate(self.selectedItems()):
            item._nrWidget.setCurrentText(f'{i+1}')
            item._currentNr = i+1
        
    def onTextActivated(self, text):
        changedNr = self.sender()._currentNr
        for item in self.selectedItems():
            row = self.row(item)
            if self.sender().row == row:
                changedNr = item._currentNr
                continue
        
        for item in self.selectedItems():
            row = self.row(item)
            if self.sender().row == row:
                continue
            nr = int(item._nrWidget.currentText())
            if nr == int(text):
                item._nrWidget.setCurrentText(str(changedNr))
                break
        
        self.updateNr()
            

class TreeWidget(QTreeWidget):
    def __init__(self, *args, multiSelection=False):
        super().__init__(*args)    
        self.setStyleSheet(TREEWIDGET_STYLESHEET)
        self.setFont(font)
        if multiSelection:
            self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
            self.itemClicked.connect(self.selectAllChildren)
        
        self.isCtrlDown = False
        self.isShiftDown = False
    
    def keyPressEvent(self, ev):
        if ev.key() == Qt.Key_Escape:
            self.clearSelection()
        elif ev.key() == Qt.Key_Control:
            self.isCtrlDown = True
        elif ev.key() == Qt.Key_Shift:
            self.isShiftDown = True

    def keyReleaseEvent(self, ev):
        if ev.key() == Qt.Key_Control:
            self.isCtrlDown = False
        elif ev.key() == Qt.Key_Shift:
            self.isShiftDown = False
    
    def onFocusChanged(self):
        self.isCtrlDown = False
        self.isShiftDown = False
    
    def selectAllChildren(self, item_or_label):
        label = None
        if isinstance(item_or_label, QLabel):
            label = item_or_label
        else:
            item = item_or_label
            if item.childCount() == 0:
                return

        if label is not None:
            if not self.isCtrlDown and not self.isShiftDown:
                self.clearSelection()
            label.item.setSelected(True)
            if self.isShiftDown:
                selectionStarted = False
                it = QTreeWidgetItemIterator(self)
                while it:
                    item = it.value()
                    if item is None:
                        break
                    if item.isSelected():
                        selectionStarted = not selectionStarted
                    if selectionStarted:
                        item.setSelected(True)
                    it += 1

        for item in self.selectedItems():
            if item.parent() is None:
                for i in range(item.childCount()):
                    item.child(i).setSelected(True)

class CancelOkButtonsLayout(QHBoxLayout):
    def __init__(self, *args, additionalButtons=None):
        super().__init__(*args)

        self.cancelButton = cancelPushButton('Cancel')
        self.okButton = okPushButton(' Ok ')

        self.addStretch(1)
        self.addWidget(self.cancelButton)
        self.addSpacing(20)
        
        if additionalButtons is not None:
            for button in additionalButtons:
                self.addWidget(button)
        
        self.addWidget(self.okButton)

class TreeWidgetItem(QTreeWidgetItem):
    def __init__(self, *args, columnColors=None):
        super().__init__(*args)

        if columnColors is not None:
            for c, color in enumerate(columnColors):
                if color is None:
                    continue
                self.setBackground(c, QBrush(color))
    
class FilterObject(QObject):
    sigFilteredEvent = Signal(object, object)

    def __init__(self) -> None:
        super().__init__()
    
    def eventFilter(self, object, event):
        self.sigFilteredEvent.emit(object, event)
        return super().eventFilter(object, event)

class readOnlyQList(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.items = []

    def addItems(self, items):
        self.items.extend(items)
        items = [str(item) for item in self.items]
        columnList = html_utils.paragraph('<br>'.join(items))
        self.setText(columnList)

class pgScatterSymbolsCombobox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)

        symbols = [
            "'o' circle (default)",
            "'s' square",
            "'t' triangle",
            "'d' diamond",
            "'+' plus",
            "'t1' triangle pointing upwards",
            "'t2' triangle pointing right side",
            "'t3' triangle pointing left side",
            "'p' pentagon",
            "'h' hexagon",
            "'star'",
            "'x' cross",
            "'arrow_up'",
            "'arrow_right'",
            "'arrow_down'",
            "'arrow_left'",
            "'crosshair'"
        ]
        self.addItems(symbols)


class alphaNumericLineEdit(QLineEdit):
    def __init__(self, parent=None, additionalChars=''):
        super().__init__(parent)

        self.validPattern = fr'^[a-zA-Z0-9{additionalChars}_\-]+$'

        regExp = QRegularExpression(self.validPattern)
        self.setValidator(QRegularExpressionValidator(regExp))

        # self.setAlignment(Qt.AlignCenter)

class NumericCommaLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.validPattern = '^[0-9,\.]+$'
        regExp = QRegularExpression(self.validPattern)
        self.setValidator(QRegularExpressionValidator(regExp))
    
    def values(self):
        try:
            vals = [float(c) for c in self.text().split(',')]
        except Exception as e:
            vals = []
        return vals

class mySpinBox(QSpinBox):
    sigTabEvent = Signal(object, object)

    def __init__(self, *args) -> None:
        super().__init__(*args)
    
    def event(self, event):
        if event.type()==QEvent.Type.KeyPress and event.key() == Qt.Key_Tab:
            self.sigTabEvent.emit(event, self)
            return True

        return super().event(event)

class KeptObjectIDsList(list):
    def __init__(self, lineEdit, confirmSelectionAction, *args):
        self.lineEdit = lineEdit
        self.lineEdit.setText('')
        self.confirmSelectionAction = confirmSelectionAction
        confirmSelectionAction.setDisabled(True)
        super().__init__(*args)
    
    def setText(self):
        IDsRange = []
        text = ''
        sorted_vals = sorted(self)
        for i, e in enumerate(sorted_vals):
            # Get previous and next value (if possible)
            if i > 0:
                prevVal = sorted_vals[i-1]
            else:
                prevVal = -1
            if i < len(sorted_vals)-1:
                nextVal = sorted_vals[i+1]
            else:
                nextVal = -1

            if e-prevVal == 1 or nextVal-e == 1:
                if not IDsRange:
                    if nextVal-e == 1 and e-prevVal != 1:
                        # Current value is the first value of a new range
                        IDsRange = [e]
                    else:
                        # Current value is the second element of a new range
                        IDsRange = [prevVal, e]
                else:
                    if e-prevVal == 1:
                        # Current value is part of an ongoing range
                        IDsRange.append(e)
                    else:
                        # Current value is the first element of a new range 
                        # --> create range text and this element will 
                        # be added to the new range at the next iter
                        start, stop = IDsRange[0], IDsRange[-1]
                        if stop-start > 1:
                            sep = '-'
                        else:
                            sep = ','
                        text = f'{text},{start}{sep}{stop}'
                        IDsRange = []
            else:
                # Current value doesn't belong to a range
                if IDsRange:
                    # There was a range not added to text --> add it now
                    start, stop = IDsRange[0], IDsRange[-1]
                    if stop-start > 1:
                        sep = '-'
                    else:
                        sep = ','
                    text = f'{text},{start}{sep}{stop}'
                
                text = f'{text},{e}'    
                IDsRange = []

        if IDsRange:
            # Last range was not added  --> add it now
            start, stop = IDsRange[0], IDsRange[-1]
            text = f'{text},{start}-{stop}'

        text = text[1:]
        
        self.lineEdit.setText(text)
    
    def append(self, element, editText=True):
        super().append(element)
        if editText:
            self.setText()
        if not self.confirmSelectionAction.isEnabled():
            self.confirmSelectionAction.setEnabled(True)

    def remove(self, element, editText=True):
        super().remove(element)
        if editText:
            self.setText()
        if not self:
            self.confirmSelectionAction.setEnabled(False)

class ScatterPlotItem(pg.ScatterPlotItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.updateBrushAndPen(**kwargs)
        
    def updateBrushAndPen(self, **kwargs):
        brush = kwargs.get('brush')
        if brush is not None:
            self._itemBrush = brush
        pen = kwargs.get('pen')
        if pen is not None:
            self._itemPen = pen
    
    def setData(self, *args, **kwargs):
        super().setData(*args, **kwargs)
        self.updateBrushAndPen(**kwargs)        
    
    def itemBrush(self):
        return self._itemBrush
    
    def itemPen(self):
        return self._itemPen
    
    def removePoint(self, index):
        newData = np.delete(self.data, index)
        # Update the index of current points
        for i in range(index, len(newData)):
            spotItem = newData[i]['item']
            spotItem._index = i
            newData[i]['item'] = spotItem
        
        self.data = newData
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.bounds = [None, None]
        self.invalidate()
        self.updateSpots(newData)
        self.sigPlotChanged.emit(self)
    
    def coordsToNumpy(self, includeData=False, rounded=True, decimals=None):
        points = self.points()
        nrows = len(points)
        coords_arr = np.zeros((nrows, 2))
        data_arr = None
        for p, point in enumerate(points):
            pos = point.pos()
            x, y = pos.x(), pos.y()
            if includeData:
                data = point.data()
                if data_arr is None:
                    try:
                        ncols = len(data)
                    except Exception as e:
                        data = [data]
                        ncols = 1
                    data_arr = np.zeros((nrows, ncols))
                for j, data_j in enumerate(data):
                    data_arr[p, j] = data_j
            
            coords_arr[p, 0] = y
            coords_arr[p, 1] = x
        if not includeData:
            out_arr = coords_arr
        elif data_arr is not None:
            out_arr = np.column_stack((data_arr, coords_arr))
        else:
            out_arr = coords_arr
        cast_to_int = decimals is None
        decimals = decimals if decimals is not None else 0
        if rounded:
            out_arr = np.round(out_arr, decimals)
        if cast_to_int:
            out_arr = out_arr.astype(int)
        return out_arr

class myLabelItem(pg.LabelItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prevText = ''

    def setText(self, text, **args):
        self.text = text
        opts = self.opts
        for k in args:
            opts[k] = args[k]
        
        if 'size' in self.opts:
            size = self.opts['size']
            if size == '0pt' or size == '0px':
                self.opts['size'] = '1pt'
                super().setText('', size='1pt')
                return

        optlist = []

        color = self.opts['color']
        if color is None:
            color = pg.getConfigOption('foreground')
        color = pg.functions.mkColor(color)
        optlist.append('color: ' + color.name(QColor.NameFormat.HexArgb))
        if 'size' in opts:
            size = opts['size']
            if not isinstance(size, str):
                size = f'{size}px'
            optlist.append('font-size: ' + size)
        if 'bold' in opts and opts['bold'] in [True, False]:
            optlist.append('font-weight: ' + {True:'bold', False:'normal'}[opts['bold']])
        if 'italic' in opts and opts['italic'] in [True, False]:
            optlist.append('font-style: ' + {True:'italic', False:'normal'}[opts['italic']])
        full = "<span style='%s'>%s</span>" % ('; '.join(optlist), text)
        #print full
        self.item.setHtml(full)
        self.updateMin()
        self.resizeEvent(None)
        self.updateGeometry()
    
    def tempClearText(self):
        if self.text:
            self._prevText = self.text
            self.setText('')
    
    def restoreText(self):
        if self._prevText:
            self.setText(self._prevText)


class myMessageBox(_base_widgets.QBaseDialog):
    def __init__(
            self, parent=None, showCentered=True, wrapText=True,
            scrollableText=False, enlargeWidthFactor=0,
            resizeButtons=True, allowClose=True
        ):
        super().__init__(parent)

        self.wrapText = wrapText
        self.enlargeWidthFactor = enlargeWidthFactor
        self.resizeButtons = resizeButtons

        self.cancel = True
        self.cancelButton = None
        self.okButton = None
        self.clickedButton = None
        self.alreadyShown = False
        self.allowClose = allowClose

        self.showCentered = showCentered

        self.scrollableText = scrollableText

        self._layout = QGridLayout()
        self.commandsLayout = None
        self._layout.setHorizontalSpacing(20)
        self.buttonsLayout = QHBoxLayout()
        self.buttonsLayout.setSpacing(2)
        self.buttons = []
        self.widgets = []
        self.layouts = []
        self.labels = []
        self.labelsWidgets = []
        self._pixmapLabels = []
        self.detailsTextWidget = None
        self.showInFileManagButton = None
        self.visibleDetails = False
        self.doNotShowAgainCheckbox = None

        self.currentRow = 0
        self._w = None

        self._layout.setColumnStretch(1, 1)
        self.setLayout(self._layout)
        
        self.setFont(font)

    def mousePressEvent(self, event):
        for label in self.labels:
            label.setTextInteractionFlags(
                Qt.TextBrowserInteraction | Qt.TextSelectableByKeyboard
            )

    def setIcon(self, iconName='SP_MessageBoxInformation'):
        label = QLabel(self)

        standardIcon = getattr(QStyle, iconName)
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        label.setPixmap(pixmap)

        self._layout.addWidget(label, 0, 0, alignment=Qt.AlignTop)

    def addImage(self, image_path):
        pixmap = QPixmap(image_path)
        label = QLabel()
        label.setPixmap(pixmap)
        self._layout.addWidget(label, self.currentRow, 1)
        self.currentRow += 1
    
    def addShowInFileManagerButton(self, path, txt=None):
        if txt is None:
            txt = 'Reveal in Finder...' if is_mac else 'Show in Explorer...'
        self.showInFileManagButton = showInFileManagerButton(txt)
        self.buttonsLayout.addWidget(self.showInFileManagButton)
        func = partial(myutils.showInExplorer, path)
        self.showInFileManagButton.clicked.connect(func)
    
    def addBrowseUrlButton(self, url, button_text=''):
        self.openUrlButton = OpenUrlButton(url, button_text)
        self.buttonsLayout.addWidget(self.openUrlButton)

    def addCancelButton(self, button=None, connect=False):
        if button is None:
            self.cancelButton = cancelPushButton('Cancel')
        else:
            self.cancelButton = button
            self.cancelButton.setIcon(QIcon(':cancelButton.svg'))

        self.buttonsLayout.insertWidget(0, self.cancelButton)
        self.buttonsLayout.insertSpacing(1, 20)
        if connect:
            self.cancelButton.clicked.connect(self.buttonCallBack)

    def splitLatexBlocks(self, text):
        texts = re.split(r"(<latex.*?>.+?)</latex>", text)
        return texts        

    def addText(self, text):
        texts = self.splitLatexBlocks(text)
        labelsWidget = LabelsWidget(texts, wrapText=self.wrapText)
        self.labelsWidgets.append(labelsWidget)
        self.labels.extend(labelsWidget.labels)
        if self.scrollableText:
            textWidget = QScrollArea()
            textWidget.setFrameStyle(QFrame.Shape.NoFrame)
            textWidget.setWidget(labelsWidget)
        else:
            textWidget = labelsWidget

        self._layout.addWidget(textWidget, self.currentRow, 1)
        self.currentRow += 1
        return labelsWidget
    
    def addCopiableCommand(self, command):
        copiableCommandWidget = CopiableCommandWidget(command)
        screenWidth = self.screen().size().width()
        maxWidth = int(0.75*screenWidth)
        sizeHint = copiableCommandWidget.sizeHint()
        width = sizeHint.width()
        if width > maxWidth:
            copiableCommandWidget = addWidgetToScrollArea(
                copiableCommandWidget, 
                resizeMinHeightNoVerticalScrollbar=True
            )
        self._layout.addWidget(copiableCommandWidget, self.currentRow, 1)
        self.currentRow += 1
    
    def copyToClipboard(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(self.sender()._command, mode=cb.Clipboard)
        print('Command copied!')

    def addButton(self, buttonText):
        if not isinstance(buttonText, str):
            # Passing button directly
            button = buttonText
            self.buttonsLayout.addWidget(button)
            button.clicked.connect(self.buttonCallBack)
            self.buttons.append(button)
            return button
        
        button, isCancelButton = getPushButton(buttonText, qparent=self)
        if not isCancelButton:
            self.buttonsLayout.addWidget(button)

        button.clicked.connect(self.buttonCallBack)
        self.buttons.append(button)
        return button

    def addDoNotShowAgainCheckbox(self, text='Do not show again'):
        self.doNotShowAgainCheckbox = QCheckBox(text)

    def addWidget(self, widget):
        self._layout.addWidget(widget, self.currentRow, 1)
        self.widgets.append(widget)
        self.currentRow += 1

    def addLayout(self, layout):
        self._layout.addLayout(layout, self.currentRow, 1)
        self.layouts.append(layout)
        self.currentRow += 1

    def setWidth(self, w):
        self._w = w

    def show(self, block=False):
        self.endOfScrollableRow = self.currentRow
        
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        # spacer
        spacer = QSpacerItem(10, 10)
        self._layout.addItem(spacer, self.currentRow, 1)
        self._layout.setRowStretch(self.currentRow, 0)

        # buttons
        self.currentRow += 1

        if self.detailsTextWidget is not None:
            self.buttonsLayout.insertWidget(1, self.detailsButton)

        # Do not show again checkbox
        if self.doNotShowAgainCheckbox is not None:
            self._layout.addWidget(
                self.doNotShowAgainCheckbox, self.currentRow, 1, 1, 2
            )
            self.currentRow += 1
        
        # spacer
        self._layout.addItem(QSpacerItem(10, 10), self.currentRow, 1)
        self.currentRow += 1
        
        # buttons
        self._layout.addLayout(
            self.buttonsLayout, self.currentRow, 0, 1, 2,
            alignment=Qt.AlignRight
        )

        # Details
        if self.detailsTextWidget is not None:
            # spacer
            self.currentRow += 1
            self._layout.addItem(QSpacerItem(20, 20), self.currentRow, 1)
            
            # detailsTextWidget
            self.currentRow += 1
            self._layout.addWidget(
                self.detailsTextWidget, self.currentRow, 0, 1, 2
            )

        # spacer
        self.currentRow += 1
        spacer = QSpacerItem(10, 10)
        self._layout.addItem(spacer, self.currentRow, 1)
        self._layout.setRowStretch(self.currentRow, 0)

        super().show()
        QTimer.singleShot(5, self._resize)
        
        self.alreadyShown = True

        if block:
            self._block()

    def setDetailedText(self, text, visible=False, wrap=True):
        text = text.replace('\n', '<br>')
        self.detailsTextWidget = QTextEdit(text)
        self.detailsTextWidget.setReadOnly(True)
        if not wrap:
            self.detailsTextWidget.setLineWrapMode(QTextEdit.NoWrap)
        self.detailsButton = showDetailsButton()
        self.detailsButton.setCheckable(True)
        self.detailsButton.clicked.connect(self._showDetails)
        self.detailsTextWidget.hide()
        self.visibleDetails = visible

    def _showDetails(self, checked):
        if checked:
            self.origHeight = self.height()
            self.resize(self.width(), self.height()+300)
            self.detailsTextWidget.show()
        else:
            self.detailsTextWidget.hide()
            func = partial(self.resize, self.width(), self.origHeight)
            QTimer.singleShot(10, func)

    def _resize(self):
        if self.resizeButtons:
            widths = [button.width() for button in self.buttons]
            if widths:
                max_width = max(widths)
                for button in self.buttons:
                    if button == self.cancelButton:
                        continue
                    button.setMinimumWidth(max_width)

        heights = [button.height() for button in self.buttons]
        if heights:
            max_h = max(heights)
            for button in self.buttons:
                button.setMinimumHeight(max_h)
            if self.detailsTextWidget is not None:
                self.detailsButton.setMinimumHeight(max_h)
            if self.showInFileManagButton is not None:
                self.showInFileManagButton.setMinimumHeight(max_h)

        if self._w is not None and self.width() < self._w:
            self.resize(self._w, self.height())

        if self.width() < 350:
            self.resize(350, self.height())

        if self.enlargeWidthFactor > 0:
            self.resize(int(self.width()*self.enlargeWidthFactor), self.height())

        if self.visibleDetails:
            self.detailsButton.click()

        if self.showCentered:
            screen = self.screen()
            screenWidth = screen.size().width()
            screenHeight = screen.size().height()
            screenLeft = screen.geometry().x()
            screenTop = screen.geometry().y()
            w, h = self.width(), self.height()
            left = int(screenLeft + screenWidth/2 - w/2)
            top = int(screenTop + screenHeight/2 - h/2)
            if top < screenTop:
                top = screenTop
            if left < screenLeft:
                left = screenLeft
            self.move(left, top)

        self._h = self.height()

        if self.okButton is not None:
            self.okButton.setFocus()

        screen = self.screen()
        screenWidth = screen.size().width()
        screenHeight = screen.size().height()
        
        # Check Force wrap Text
        for labelWidget in self.labelsWidgets:
            textWidth = labelWidget.width()
            if not textWidth > screenWidth-10:
                continue
            factor = np.ceil(textWidth/screenWidth)
            lineLength = int(labelWidget.nCharsLongestLine/factor)
            for label in labelWidget.labels:
                text = label.text()
                chunks = textwrap.wrap(text, lineLength)
                text = '<br>'.join(chunks)
                label.setText(text)
            
            QTimer.singleShot(100, self._resizeWrappedText)
        
        if self.widgets:
            return

        if self.layouts:
            return

        # # Start resizing height every 1 ms
        # self.resizeCallsCount = 0
        # self.timer = QTimer()
        # from config import warningHandler
        # warningHandler.sigGeometryWarning.connect(self.timer.stop)
        # self.timer.timeout.connect(self._resizeHeight)
        # self.timer.start(1)

    def _resizeWrappedText(self):
        screenWidth = self.screen().size().width() - 5
        self.resize(screenWidth, self.height())
        screenLeft = self.screen().geometry().left()
        self.move(screenLeft, self.geometry().top())
    
    def _resizeHeight(self):
        try:
            # Resize until a "Unable to set geometry" warning is captured
            # by copnfig.warningHandler._resizeWarningHandler or #
            # height doesn't change anymore
            self.resize(self.width(), self.height()-1)
            if self.height() == self._h or self.resizeCallsCount > 100:
                self.timer.stop()
                return

            self.resizeCallsCount += 1
            self._h = self.height()
        except Exception as e:
            # traceback.format_exc()
            self.timer.stop()

    def _template(
            self, parent, title, message, detailsText=None,
            buttonsTexts=None, layouts=None, widgets=None,
            commands=None, path_to_browse=None, browse_button_text=None,
            url_to_open=None, open_url_button_text='Open url', 
            image_paths=None, wrapDetails=True
        ):
        if parent is not None:
            self.setParent(parent)
        self.setWindowTitle(title)
        self.addText(message)
        if commands is not None:
            if isinstance(commands, str):
                commands = (commands,)
            for command in commands:
                self.addCopiableCommand(command)
        
        if image_paths is not None:
            if isinstance(image_paths, str):
                image_paths = (image_paths,)
            for image_path in image_paths:
                self.addImage(image_path)
        
        if layouts is not None:
            if myutils.is_iterable(layouts):
                for layout in layouts:
                    self.addLayout(layout)
            else:
                self.addLayout(layout)

        if widgets is not None:
            self._layout.addItem(QSpacerItem(20, 20), self.currentRow, 1)
            self.currentRow += 1
            if myutils.is_iterable(widgets):
                for widget in widgets:
                    self.addWidget(widget)
            else:
                self.addWidget(widgets)

        if path_to_browse is not None:
            self.addShowInFileManagerButton(
                path_to_browse, txt=browse_button_text
            )
        
        if url_to_open is not None:
            self.addBrowseUrlButton(
                url_to_open, button_text=open_url_button_text
            )
        
        buttons = []
        if buttonsTexts is None:
            okButton = self.addButton('  Ok  ')
            buttons.append(okButton)
        elif isinstance(buttonsTexts, str):
            button = self.addButton(buttonsTexts)
            buttons.append(button)
        else:
            for buttonText in buttonsTexts:
                button = self.addButton(buttonText)
                buttons.append(button)
        
        if detailsText is not None:
            self.setDetailedText(detailsText, visible=True, wrap=wrapDetails)
        return buttons

    def critical(self, *args, showDialog=True, **kwargs):
        self.setIcon(iconName='SP_MessageBoxCritical')
        buttons = self._template(*args, **kwargs)
        if showDialog:
            self.exec_()
        return buttons

    def information(self, *args, showDialog=True, **kwargs):
        self.setIcon(iconName='SP_MessageBoxInformation')
        buttons = self._template(*args, **kwargs)
        if showDialog:
            self.exec_()
        return buttons

    def warning(self, *args, showDialog=True, **kwargs):
        self.setIcon(iconName='SP_MessageBoxWarning')
        buttons = self._template(*args, **kwargs)
        if showDialog:
            self.exec_()
        return buttons

    def question(self, *args, showDialog=True, **kwargs):
        self.setIcon(iconName='SP_MessageBoxQuestion')
        buttons = self._template(*args, **kwargs)
        if showDialog:
            self.exec_()
        return buttons

    def _block(self):
        self.loop = QEventLoop()
        self.loop.exec_()

    def exec_(self):
        self.show(block=True)
    
    def clickButtonFromText(self, buttonText):
        for button in self.buttons:
            if button.text() == buttonText:
                button.click()
                return

    def buttonCallBack(self, checked=True):
        self.clickedButton = self.sender()
        if self.clickedButton != self.cancelButton:
            self.cancel = False
        self.allowClose = True
        self.close()

    def closeEvent(self, event):
        if not self.allowClose:
            event.ignore()
            return
        super().closeEvent(event)

class FormLayout(QGridLayout):
    def __init__(self):
        QGridLayout.__init__(self)

    def addFormWidget(
            self, formWidget, 
            leftLabelAlignment=Qt.AlignRight, 
            align=None, 
            row=0
        ):
        for col, item in enumerate(formWidget.items):
            if col==0:
                alignment = leftLabelAlignment
            elif col==2:
                alignment = Qt.AlignLeft
            else:
                alignment = align
            try:
                if alignment is None:
                    self.addWidget(item, row, col)
                else:
                    self.addWidget(item, row, col, alignment=alignment)
            except TypeError:
                self.addLayout(item, row, col)

def macShortcutToWindows(shortcut: str):
    if shortcut is None:
        return
    s = shortcut.replace('Control', 'Meta')
    s = shortcut.replace('Option', 'Alt')
    s = shortcut.replace('Command', 'Ctrl')
    return s

class ToolBar(QToolBar):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        for child in self.children(): 
            if child.objectName() == 'qt_toolbar_ext_button':
                self.extendButton = child
                self.extendButton.setIcon(QIcon(":expand.svg"))
                break
    
    def addSeparator(self, width=5):
        self.addWidget(QHWidgetSpacer(width=width))
        self.addWidget(QVLine())
        self.addWidget(QHWidgetSpacer(width=width))
    
    def addSpinBox(self, label=''):
        spinbox = SpinBox(disableKeyPress=True)
        if label:
            spinbox.label = QLabel(label)
            self.addWidget(spinbox.label)
        
        self.addWidget(spinbox)
        return spinbox

class ManualTrackingToolBar(ToolBar):
    sigIDchanged = Signal(int)
    sigDisableGhost = Signal()
    sigClearGhostContour = Signal()
    sigClearGhostMask = Signal()
    sigGhostOpacityChanged = Signal(int)

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.spinboxID = self.addSpinBox(label='ID to track: ')
        self.spinboxID.setMinimum(1)

        self.addSeparator()

        self.showGhostCheckbox = QCheckBox('Show ghost object')
        self.showGhostCheckbox.setChecked(True)
        self.addWidget(self.showGhostCheckbox)

        self.ghostContourRadiobutton = QRadioButton('Contour')
        self.ghostMaskRadiobutton = QRadioButton('Mask ; ')
        self.ghostMaskRadiobutton.setChecked(True)
        self.addWidget(self.ghostContourRadiobutton)
        self.addWidget(self.ghostMaskRadiobutton)

        self.ghostMaskOpacitySpinbox = self.addSpinBox('Mask opacity:  ')
        self.ghostMaskOpacitySpinbox.setMaximum(100)
        self.ghostMaskOpacitySpinbox.setValue(30)

        self.showGhostCheckbox.toggled.connect(self.showGhostCheckboxToggled)
        self.ghostContourRadiobutton.toggled.connect(
            self.ghostContourRadiobuttonToggled
        )
        self.spinboxID.valueChanged.connect(self.IDchanged)

        self.ghostMaskOpacitySpinbox.valueChanged.connect(
            self.ghostOpacityValueChanged
        )

        self.addSeparator()

        self.infoLabel = QLabel('')
        self.addWidget(self.infoLabel)
    
    def showInfo(self, text):
        text = html_utils.paragraph(text, font_color='black')
        self.infoLabel.setText(text)

    def showWarning(self, text):
        text = html_utils.paragraph(f'WARNING: {text}', font_color='red')
        self.infoLabel.setText(text)
    
    def clearInfoText(self):
        self.infoLabel.setText('')
    
    def IDchanged(self, value):
        self.sigIDchanged.emit(value)
    
    def showGhostCheckboxToggled(self, checked):
        disabled = not checked
        self.ghostContourRadiobutton.setDisabled(disabled)
        self.ghostMaskRadiobutton.setDisabled(disabled)
        self.ghostMaskOpacitySpinbox.setDisabled(disabled)
        self.ghostMaskOpacitySpinbox.label.setDisabled(disabled)
        if disabled:
            self.sigDisableGhost.emit()
    
    def ghostContourRadiobuttonToggled(self, checked):
        self.ghostMaskOpacitySpinbox.setDisabled(checked)
        self.ghostMaskOpacitySpinbox.label.setDisabled(checked)
        if checked:
            self.sigClearGhostMask.emit()      
        else:
            self.sigClearGhostContour.emit()
    
    def ghostOpacityValueChanged(self, value):
        self.sigGhostOpacityChanged.emit(value)

class ManualBackgroundToolBar(ToolBar):
    sigIDchanged = Signal(int)

    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.spinboxID = self.addSpinBox(label='Set background of ID ')
        self.spinboxID.setMinimum(1)
        self.spinboxID.valueChanged.connect(self.IDchanged)
        
        self.infoLabel = QLabel('')
        self.addWidget(self.infoLabel)
    
    def IDchanged(self, value):
        self.sigIDchanged.emit(value)
    
    def showWarning(self, text):
        text = html_utils.paragraph(f'WARNING: {text}', font_color='red')
        self.infoLabel.setText(text)
    
    def clearInfoText(self):
        self.infoLabel.setText('')
    

class rightClickToolButton(QToolButton):
    sigRightClick = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
        elif event.button() == Qt.MouseButton.RightButton:
            self.sigRightClick.emit(event)

class ToolButtonCustomColor(rightClickToolButton):
    def __init__(self, symbol, color='r', parent=None):
        super().__init__(parent=parent)
        if not isinstance(color, QColor):
            color = pg.mkColor(color)
        self.symbol = symbol
        self.setColor(color)

    def setColor(self, color):
        self.penColor = color
        self.brushColor = [0, 0, 0, 100]
        self.brushColor[:3] = color.getRgb()[:3]
    
    def updateSymbol(self, symbol, update=True):
        self.symbol = symbol
        if not update:
            return
        self.update()
    
    def updateColor(self, color, update=True):
        self.setColor(color)
        if not update:
            return
        self.update()
    
    def updateIcon(self, symbol, color):
        self.updateSymbol(symbol)
        self.updateColor(color)
        self.update()

    def paintEvent(self, event):
        QToolButton.paintEvent(self, event)
        p = QPainter(self)
        w, h = self.width(), self.height()
        sf = 0.6
        p.scale(w*sf, h*sf)
        p.translate(0.5/sf, 0.5/sf)
        symbol = pg.graphicsItems.ScatterPlotItem.Symbols[self.symbol]
        pen = pg.mkPen(color=self.penColor, width=2)
        brush = pg.mkBrush(color=self.brushColor)
        try:
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(pen)
            p.setBrush(brush)
            p.drawPath(symbol)
        except Exception as e:
            traceback.print_exc()
        finally:
            p.end()

class PointsLayerToolButton(ToolButtonCustomColor):
    sigEditAppearance = Signal(object)
    sigShowIdsToggled = Signal(object, bool)
    sigRemove = Signal(object)

    def __init__(self, symbol, color='r', parent=None):
        super().__init__(symbol, color=color, parent=parent)
        self.sigRightClick.connect(self.showContextMenu)
    
    def showContextMenu(self, event):
        contextMenu = QMenu(self)
        contextMenu.addSeparator()

        editAction = QAction('Edit points appearance...')
        editAction.triggered.connect(self.editAppearance)
        contextMenu.addAction(editAction)
        
        removeAction = QAction('Remove points')
        removeAction.triggered.connect(self.emitRemove)
        contextMenu.addAction(removeAction)
        
        showIdsAction = QAction('Show point ids')
        showIdsAction.setCheckable(True)
        showIdsAction.setChecked(True)
        contextMenu.addAction(showIdsAction)
        showIdsAction.toggled.connect(self.emitShowIdsToggled)

        contextMenu.exec(event.globalPos())
    
    def emitRemove(self):
        self.sigRemove.emit(self)
    
    def emitShowIdsToggled(self, checked):
        self.sigShowIdsToggled.emit(self, checked)
    
    def editAppearance(self):
        self.sigEditAppearance.emit(self)

class customAnnotToolButton(ToolButtonCustomColor):
    sigRemoveAction = Signal(object)
    sigKeepActiveAction = Signal(object)
    sigModifyAction = Signal(object)
    sigHideAction = Signal(object)

    def __init__(
            self, symbol, color, keepToolActive=True, parent=None,
            isHideChecked=True
        ):
        super().__init__(symbol, color=color, parent=parent)
        self.symbol = symbol
        self.keepToolActive = keepToolActive
        self.isHideChecked = isHideChecked
        self.sigRightClick.connect(self.showContextMenu)

    def showContextMenu(self, event):
        contextMenu = QMenu(self)
        contextMenu.addSeparator()

        removeAction = QAction('Remove annotation')
        removeAction.triggered.connect(self.removeAction)
        contextMenu.addAction(removeAction)

        editAction = QAction('Modify annotation parameters...')
        editAction.triggered.connect(self.modifyAction)
        contextMenu.addAction(editAction)

        hideAction = QAction('Hide annotations')
        hideAction.setCheckable(True)
        hideAction.setChecked(self.isHideChecked)
        hideAction.triggered.connect(self.hideAction)
        contextMenu.addAction(hideAction)

        keepActiveAction = QAction('Keep tool active after using it')
        keepActiveAction.setCheckable(True)
        keepActiveAction.setChecked(self.keepToolActive)
        keepActiveAction.triggered.connect(self.keepToolActiveActionToggled)
        contextMenu.addAction(keepActiveAction)

        contextMenu.exec(event.globalPos())

    def keepToolActiveActionToggled(self, checked):
        self.keepToolActive = checked
        self.sigKeepActiveAction.emit(self)

    def modifyAction(self):
        self.sigModifyAction.emit(self)

    def removeAction(self):
        self.sigRemoveAction.emit(self)

    def hideAction(self, checked):
        self.isHideChecked = checked
        self.sigHideAction.emit(self)

class LabelRoiCircularItem(pg.ScatterPlotItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
    
    def setImageShape(self, shape):
        self._shape = shape
    
    def slice(self, zRange=None, tRange=None):
        self.mask()
        if zRange is None:
            _slice = self._slice
        else:
            zmin, zmax = zRange
            _slice = (slice(zmin, zmax), *self._slice)
        
        if tRange is not None:
            tmin, tmax = tRange
            _slice = (slice(tmin, tmax), *_slice)
        
        return _slice

    def mask(self):
        shape = self._shape
        radius = int(self.opts['size']/2)
        mask = skimage.morphology.disk(radius, dtype=bool)
        xx, yy = self.getData()
        Yc, Xc = yy[0], xx[0]
        mask, self._slice = myutils.clipSelemMask(mask, shape, Yc, Xc, copy=False)
        return mask

class Toggle(QCheckBox):
    def __init__(
            self,
            label_text='',
            initial=None,
            width=80,
            bg_color='#b3b3b3',
            circle_color='#ffffff',
            active_color='#26dd66',# '#005ce6',
            animation_curve=QEasingCurve.Type.InOutQuad
        ):
        QCheckBox.__init__(self)

        # self.setFixedSize(width, 28)
        self.setCursor(Qt.PointingHandCursor)

        self._label_text = label_text
        self._bg_color = bg_color
        self._circle_color = circle_color
        self._active_color = active_color
        self._disabled_active_color = colors.lighten_color(active_color)
        self._disabled_circle_color = colors.lighten_color(circle_color)
        self._disabled_bg_color = colors.lighten_color(bg_color, amount=0.5)
        self._circle_margin = 4

        self._circle_position = int(self._circle_margin/2)
        self.animation = QPropertyAnimation(self, b'circle_position', self)
        self.animation.setEasingCurve(animation_curve)
        self.animation.setDuration(200)

        self.stateChanged.connect(self.start_transition)
        self.requestedState = None

        self.installEventFilter(self)
        self._isChecked = False

        if initial is not None:
            self.setChecked(initial)

    def sizeHint(self):
        return QSize(36, 18)

    def eventFilter(self, object, event):
        # To get the actual position of the circle we need to wait that
        # the widget is visible before setting the state
        if event.type() == QEvent.Type.Show and self.requestedState is not None:
            self.setChecked(self.requestedState)
        return False

    def setChecked(self, state):
        # To get the actual position of the circle we need to wait that
        # the widget is visible before setting the state
        self._isChecked = state
        if self.isVisible():
            self.requestedState = None
            QCheckBox.setChecked(self, state>0)
        else:
            self.requestedState = state
    
    def isChecked(self):
        if self.isVisible():
            return super().isChecked()
        else:
            return self._isChecked

    def circlePos(self, state: bool):
        start = int(self._circle_margin/2)
        if state:
            if self.isVisible():
                height, width = self.height(), self.width()
            else:
                sizeHint = self.sizeHint()
                height, width = sizeHint.height(), sizeHint.width()
            circle_diameter = height-self._circle_margin
            pos = width-start-circle_diameter
        else:
            pos = start
        return pos

    @Property(float)
    def circle_position(self):
        return self._circle_position

    @circle_position.setter
    def circle_position(self, pos):
        self._circle_position = pos
        self.update()

    def start_transition(self, state):
        self.animation.stop()
        pos = self.circlePos(state)
        self.animation.setEndValue(pos)
        self.animation.start()

    def hitButton(self, pos: QPoint):
        return self.contentsRect().contains(pos)

    def setDisabled(self, disable):
        QCheckBox.setDisabled(self, disable)
        if hasattr(self, 'label'):
            self.label.setDisabled(disable)
        self.update()

    def paintEvent(self, e):
        circle_color = (
            self._circle_color if self.isEnabled()
            else self._disabled_circle_color
        )
        active_color = (
            self._active_color if self.isEnabled()
            else self._disabled_active_color
        )
        unchecked_color = (
            self._bg_color if self.isEnabled()
            else self._disabled_bg_color
        )

        # set painter
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        # set no pen
        p.setPen(Qt.NoPen)

        # draw rectangle
        rect = QRect(0, 0, self.width(), self.height())

        if not self.isChecked():
            # Draw background
            p.setBrush(QColor(unchecked_color))
            half_h = int(self.height()/2)
            p.drawRoundedRect(
                0, 0, rect.width(), self.height(), half_h, half_h
            )

            # Draw circle
            p.setBrush(QColor(circle_color))
            p.drawEllipse(
                int(self._circle_position), int(self._circle_margin/2),
                self.height()-self._circle_margin,
                self.height()-self._circle_margin
            )
        else:
            # Draw background
            p.setBrush(QColor(active_color))
            half_h = int(self.height()/2)
            p.drawRoundedRect(
                0, 0, rect.width(), self.height(), half_h, half_h
            )

            # Draw circle
            p.setBrush(QColor(circle_color))
            p.drawEllipse(
                int(self._circle_position), int(self._circle_margin/2),
                self.height()-self._circle_margin,
                self.height()-self._circle_margin
            )

        p.end()

def QKeyEventToString(event: QKeyEvent, notAllowedModifier=None):
    isAltKey = event.key()==Qt.Key_Alt
    isCtrlKey = event.key()==Qt.Key_Control
    isShiftKey = event.key()==Qt.Key_Shift
    isModifierKey = isAltKey or isCtrlKey or isShiftKey
    
    modifiers = event.modifiers()
    isNotAllowedMod = (
        notAllowedModifier is not None and modifiers == notAllowedModifier
    )
    if isNotAllowedMod:
        return 
    
    modifers_value = modifiers.value if PYQT6 else modifiers
    if isModifierKey:
        keySequenceText = QKeySequence(modifers_value).toString()
    else:
        keySequenceText = QKeySequence(modifers_value | event.key()).toString()
    
    keySequenceText = keySequenceText.encode('ascii', 'ignore').decode('utf-8')
    
    return keySequenceText

class ShortcutLineEdit(QLineEdit):
    def __init__(
            self, parent=None, allowModifiers=False, notAllowedModifier=None
        ):
        self.keySequence = None
        super().__init__(parent)
        self._allowModifiers = allowModifiers
        self._notAllowedModifier = notAllowedModifier
        self.setAlignment(Qt.AlignCenter)
    
    def setText(self, text):
        super().setText(text)
        if not text:
            self.keySequence = None
            return
        try:
            self.keySequence = QKeySequence(self.text())
        except Exception as e:
            pass

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Backspace or event.key() == Qt.Key_Delete:
            self.setText('')
            return

        keySequenceText = QKeyEventToString(
            event, notAllowedModifier=self._notAllowedModifier
        )
        self.setText(keySequenceText)
        self.key = event.key()
    
    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if self.text().endswith('+'):
            if not self._allowModifiers:
                self.setText('')
            else:
                self.setText(self.text().rstrip('+').strip())
            

class selectStartStopFrames(QGroupBox):
    def __init__(self, SizeT, currentFrameNum=0, parent=None):
        super().__init__(parent)
        selectFramesLayout = QGridLayout()

        self.startFrame_SB = QSpinBox()
        self.startFrame_SB.setAlignment(Qt.AlignCenter)
        self.startFrame_SB.setMinimum(1)
        self.startFrame_SB.setMaximum(SizeT-1)
        self.startFrame_SB.setValue(currentFrameNum)

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

        self.setLayout(selectFramesLayout)

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

class formWidget(QWidget):
    sigApplyButtonClicked = Signal(object)
    sigComputeButtonClicked = Signal(object)

    def __init__(
            self, widget,
            initialVal=None,
            stretchWidget=True,
            widgetAlignment=None,
            labelTextLeft='',
            labelTextRight='',
            font=None,
            addInfoButton=False,
            addApplyButton=False,
            addComputeButton=False,
            key='',
            infoTxt='',
            parent=None
        ):
        QWidget.__init__(self, parent)
        self.widget = widget
        self.key = key
        self.infoTxt = infoTxt
        self.widgetAlignment = widgetAlignment

        widget.setParent(self)

        if isinstance(initialVal, bool):
            widget.setChecked(initialVal)
        elif isinstance(initialVal, str):
            widget.setCurrentText(initialVal)
        elif isinstance(initialVal, float) or isinstance(initialVal, int):
            widget.setValue(initialVal)

        self.items = []

        if font is None:
            font = QFont()
            font.setPixelSize(13)

        self.labelLeft = QClickableLabel(widget)
        self.labelLeft.setText(labelTextLeft)
        self.labelLeft.setFont(font)
        self.items.append(self.labelLeft)

        if not stretchWidget:
            widgetLayout = QHBoxLayout()
            if widgetAlignment != 'left':
                widgetLayout.addStretch(1)
            widgetLayout.addWidget(widget)
            if widgetAlignment != 'right':
                widgetLayout.addStretch(1)
            self.items.append(widgetLayout)
        else:
            self.items.append(widget)

        self.labelRight = QClickableLabel(widget)
        self.labelRight.setText(labelTextRight)
        self.labelRight.setFont(font)
        self.items.append(self.labelRight)

        if addInfoButton:
            infoButton = QPushButton(self)
            infoButton.setCursor(Qt.WhatsThisCursor)
            infoButton.setIcon(QIcon(":info.svg"))
            if labelTextLeft:
                infoButton.setToolTip(
                    f'Info about "{self.labelLeft.text()}" parameter'
                )
            else:
                infoButton.setToolTip(
                    f'Info about "{self.labelRight.text()}" measurement'
                )
            infoButton.clicked.connect(self.showInfo)
            self.infoButton = infoButton
            self.items.append(infoButton)

        if addApplyButton:
            applyButton = QPushButton(self)
            applyButton.setCursor(Qt.PointingHandCursor)
            applyButton.setCheckable(True)
            applyButton.setIcon(QIcon(":apply.svg"))
            applyButton.setToolTip(f'Apply this step and visualize results')
            applyButton.clicked.connect(self.applyButtonClicked)
            self.items.append(applyButton)

        if addComputeButton:
            computeButton = QPushButton(self)
            computeButton.setCursor(Qt.BusyCursor)
            computeButton.setIcon(QIcon(":compute.svg"))
            computeButton.setToolTip(f'Compute this step and visualize results')
            computeButton.clicked.connect(self.computeButtonClicked)
            self.items.append(computeButton)

        self.labelLeft.clicked.connect(self.tryChecking)
        self.labelRight.clicked.connect(self.tryChecking)

    def tryChecking(self, label):
        try:
            self.widget.setChecked(not self.widget.isChecked())
        except AttributeError as e:
            pass

    def applyButtonClicked(self):
        self.sigApplyButtonClicked.emit(self)

    def computeButtonClicked(self):
        self.sigComputeButtonClicked.emit(self)

    def showInfo(self):
        msg = myMessageBox()
        msg.setIcon()
        msg.setWindowTitle(f'{self.labelLeft.text()} info')
        msg.addText(self.infoTxt)
        msg.addButton('   Ok   ')
        msg.exec_()
    
    def setDisabled(self, disabled: bool) -> None:
        for item in self.items:
            try:
                item.setDisabled(disabled)
            except Exception as err:
                pass

class ToggleTerminalButton(PushButton):
    sigClicked = Signal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setIcon(QIcon(':terminal_up.svg'))
        self.setFixedSize(34,18)
        self.setIconSize(QSize(30, 14))
        self.setFlat(True)
        self.terminalVisible = False
        self.clicked.connect(self.mouseClick)
    
    def mouseClick(self):
        if self.terminalVisible:
            self.setIcon(QIcon(':terminal_up.svg'))
            self.terminalVisible = False
        else:
            self.setIcon(QIcon(':terminal_down.svg'))
            self.terminalVisible = True
        self.sigClicked.emit(self.terminalVisible)
    
    def showEvent(self, a0) -> None:
        self.idlePalette = self.palette()
        return super().showEvent(a0)
    
    def enterEvent(self, event) -> None:
        self.setFlat(False)
        # pal = self.palette()
        # pal.setColor(QPalette.ColorRole.Button, QColor(200, 200, 200))
        # self.setAutoFillBackground(True)
        # self.setPalette(pal)
        self.update()
        return super().enterEvent(event)
    
    def leaveEvent(self, event) -> None:
        self.setFlat(True)
        # self.setPalette(self.idlePalette)
        self.update()
        return super().leaveEvent(event)

class CenteredDoubleSpinbox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31-1)

class readOnlyDoubleSpinbox(QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setReadOnly(True)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31-1)
        # self.setStyleSheet('background-color: rgba(240, 240, 240, 200);')

class readOnlySpinbox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setReadOnly(True)
        self.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31-1)
        # self.setStyleSheet('background-color: rgba(240, 240, 240, 200);')

class DoubleSpinBox(QDoubleSpinBox):
    sigValueChanged = Signal(int)

    def __init__(self, parent=None, disableKeyPress=False):
        super().__init__(parent=parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31-1)
        self.setMinimum(-2**31)
        self._valueChangedFunction = None
        self.disableKeyPress = disableKeyPress
    
    def keyPressEvent(self, event) -> None:
        isBackSpaceKey = event.key() == Qt.Key_Backspace
        isDeleteKey = event.key() == Qt.Key_Delete
        try:
            int(event.text())
            isIntegerKey = True
        except:
            isIntegerKey = False
        acceptEvent = isBackSpaceKey or isDeleteKey or isIntegerKey
        if self.disableKeyPress and not acceptEvent:
            event.ignore()
            self.clearFocus()
        else:
            super().keyPressEvent(event)
    
    def textFromValue(self, value: float) -> str:
        text = super().textFromValue(value)
        return text.replace(QLocale().decimalPoint(), '.')

    def valueFromText(self, text: str) -> float:
        text = text.replace('.', QLocale().decimalPoint())
        return super().valueFromText(text)

class SpinBox(QSpinBox):
    sigValueChanged = Signal(int)
    sigUpClicked = Signal()
    sigDownClicked = Signal()

    def __init__(self, parent=None, disableKeyPress=False):
        super().__init__(parent=parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31-1)
        self.setMinimum(-2**31)
        self._valueChangedFunction = None
        self.disableKeyPress = disableKeyPress
    
    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        opt = QStyleOptionSpinBox()
        self.initStyleOption(opt)

        control = self.style().hitTestComplexControl(
            QStyle.ComplexControl.CC_SpinBox, opt, event.pos(), self
        )
        if control == QStyle.SubControl.SC_SpinBoxUp:
            self.sigUpClicked.emit()
        elif control == QStyle.SubControl.SC_SpinBoxDown:
            self.sigDownClicked.emit()

    # def focusOutEvent(self, event):
    #     self.editingFinished.emit()
    #     super().focusOutEvent(event)
    #     printl('emitted')
    
    def keyPressEvent(self, event) -> None:
        isBackSpaceKey = event.key() == Qt.Key_Backspace
        isDeleteKey = event.key() == Qt.Key_Delete
        try:
            int(event.text())
            isIntegerKey = True
        except:
            isIntegerKey = False
        acceptEvent = isBackSpaceKey or isDeleteKey or isIntegerKey
        if self.disableKeyPress and not acceptEvent:
            event.ignore()
            self.clearFocus()
        else:
            super().keyPressEvent(event)
    
    def connectValueChanged(self, function):
        self._valueChangedFunction = function
        self.valueChanged.connect(function)
    
    def setValueNoEmit(self, value):
        if self._valueChangedFunction is None:
            self.setValue(value)
            return
        self.valueChanged.disconnect()
        self.setValue(value)
        self.valueChanged.connect(self._valueChangedFunction)
    
    def wheelEvent(self, event):
        event.ignore()

class ReadOnlyLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setReadOnly(True)
        # self.setStyleSheet(
        #     'background-color: rgba(240, 240, 240, 200);'
        # )
        self.installEventFilter(self)
    
    def eventFilter(self, a0: 'QObject', a1: 'QEvent') -> bool:
        if a1.type() == QEvent.Type.FocusIn:
            return True
        return super().eventFilter(a0, a1)

class FloatLineEdit(QLineEdit):
    valueChanged = Signal(float)

    def __init__(
            self, *args, notAllowed=None, allowNegative=True, initial=None,
            readOnly=False, decimals=6, warningValues=None
        ):
        QLineEdit.__init__(self, *args)
        if readOnly:
            self.setReadOnly(readOnly)
        self.notAllowed = notAllowed
        self.warningValues = warningValues
        self._maximum = np.inf
        self._minimum = -np.inf
        self._decimals = decimals

        self.isNumericRegExp = rf'^{float_regex(allow_negative=allowNegative)}$'
        regExp = QRegularExpression(self.isNumericRegExp)
        self.setValidator(QRegularExpressionValidator(regExp))
        self.setAlignment(Qt.AlignCenter)

        font = QFont()
        font.setPixelSize(11)
        self.setFont(font)

        self.textChanged.connect(self.emitValueChanged)
        if initial is None:
            self.setText('0.0')
    
    def setDecimals(self, decimals):
        self._decimals = 6

    def setValue(self, value: float):
        if value > self._maximum:
            value = self._maximum
        if value < self._minimum:
            value = self._minimum
        self.setText(str(round(value, self._decimals)))

    def value(self):
        m = re.match(self.isNumericRegExp, self.text())
        if m is not None:
            text = m.group(0)
            try:
                val = float(text)
            except ValueError:
                val = 0.0
            return val
        else:
            return 0.0
    
    def setMaximum(self, maximum):
        self._maximum = maximum
    
    def setMinimum(self, minimum):
        self._minimum = minimum

    def emitValueChanged(self, text):
        val = self.value()
        reset_stylesheet = True
        if self.warningValues is not None and val in self.warningValues:
            self.setStyleSheet(LINEEDIT_WARNING_STYLESHEET)
            reset_stylesheet = False
        
        if self.notAllowed is not None and val in self.notAllowed:
            self.setStyleSheet(LINEEDIT_INVALID_ENTRY_STYLESHEET)
            reset_stylesheet = False
        else:
            self.valueChanged.emit(self.value())
        
        if reset_stylesheet:
            self.setStyleSheet('')

class IntLineEdit(QLineEdit):
    valueChanged = Signal(float)

    def __init__(
            self, *args, notAllowed=None, allowNegative=True, initial=None,
            readOnly=False
        ):
        QLineEdit.__init__(self, *args)
        self.notAllowed = notAllowed
        if readOnly:
            self.setReadOnly(readOnly)

        self._maximum = np.inf
        self._minimum = -np.inf
        
        self._regExp = r'\d+'
        if allowNegative:
            self._regExp = r'-?\d+'

        regExp = QRegularExpression(self._regExp)
        self.setValidator(QRegularExpressionValidator(regExp))
        self.setAlignment(Qt.AlignCenter)

        font = QFont()
        font.setPixelSize(11)
        self.setFont(font)

        self.textChanged.connect(self.emitValueChanged)
        if initial is None:
            self.setText('0')
    
    def setMaximum(self, maximum):
        self._maximum = maximum
    
    def setMinimum(self, minimum):
        self._minimum = minimum

    def setValue(self, value: int):
        if value > self._maximum:
            value = self._maximum
        if value < self._minimum:
            value = self._minimum
        self.setText(str(value))

    def value(self):
        m = re.match(self._regExp, self.text())
        if m is not None:
            text = m.group(0)
            try:
                val = int(text)
            except ValueError:
                val = 0
            return val
        else:
            return 0

    def emitValueChanged(self, text):
        val = self.value()
        if self.notAllowed is not None and val in self.notAllowed:
            self.setStyleSheet(LINEEDIT_INVALID_ENTRY_STYLESHEET)
        else:
            self.setStyleSheet('')
            self.valueChanged.emit(self.value())

class CheckboxesGroupBox(QGroupBox):
    def __init__(
            self, texts, title='', checkable=False, parent=None
        ):
        super().__init__(parent)
        
        self.setTitle(title)
        self.setCheckable(checkable)
        layout = QVBoxLayout()

        scrollLayout = QVBoxLayout()
        container = QWidget()
        scrollarea = QScrollArea()
        
        self.checkBoxes = []
        for text in texts:
            checkbox = QCheckBox(text)
            checkbox.setChecked(True)
            scrollLayout.addWidget(checkbox)
            self.checkBoxes.append(checkbox)
        
        container.setLayout(scrollLayout)
        scrollarea.setWidget(container)
        layout.addWidget(scrollarea)
        
        buttonsLayout = QHBoxLayout()
        selectAllButton = selectAllPushButton()
        selectAllButton.sigClicked.connect(self.checkAll)
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(selectAllButton)
        layout.addLayout(buttonsLayout)
        
        self.setLayout(layout)
    
    def checkAll(self, button, checked):
        for checkBox in self.checkBoxes:
            checkBox.setChecked(checked)
            

class _metricsQGBox(QGroupBox):
    sigDelClicked = Signal(str, object)

    def __init__(
            self, desc_dict, title, favourite_funcs=None, isZstack=False,
            equations=None, addDelButton=False, delButtonMetricsDesc=None,
            parent=None
        ):
        QGroupBox.__init__(self, parent)
        
        highlightRgba = _palettes._highlight_rgba()
        r, g, b, a = highlightRgba
        self._highlightStylesheetColor = f'rgb({r}, {g}, {b})'
        
        self._parent = parent
        self.scrollArea = QScrollArea()
        self.scrollAreaWidget = QWidget()
        self.favourite_funcs = favourite_funcs

        self.doNotWarn = False

        layout = QVBoxLayout()
        inner_layout = QVBoxLayout()
        self.inner_layout = inner_layout
        if delButtonMetricsDesc is None:
            delButtonMetricsDesc = []

        self.checkBoxes = []
        self.checkedState = {}
        for metric_colname, metric_desc in desc_dict.items():
            rowLayout = QHBoxLayout()

            checkBox = QCheckBox(metric_colname)
            checkBox.setChecked(True)
            checkBox.scrollArea = self.scrollArea
            self.checkBoxes.append(checkBox)
            self.checkedState[checkBox] = True

            try:
                checkBox.equation = equations[metric_colname]
            except Exception as e:
                pass
            
            if addDelButton or metric_colname in delButtonMetricsDesc:
                delButton = delPushButton()
                delButton.setToolTip('Delete custom combined measurement')
                delButton.colname = metric_colname
                delButton.checkbox = checkBox
                delButton.clicked.connect(self.onDelClicked)
                delButton._layout = rowLayout
                rowLayout.addWidget(delButton)

            infoButton = infoPushButton()
            infoButton.setCursor(Qt.WhatsThisCursor)
            infoButton.info = metric_desc
            infoButton.colname = metric_colname
            infoButton.clicked.connect(self.showInfo)

            rowLayout.addWidget(infoButton)
            rowLayout.addWidget(checkBox)   
            rowLayout.addStretch(1)          

            inner_layout.addLayout(rowLayout)

        self.scrollAreaWidget.setLayout(inner_layout)
        self.scrollArea.setWidget(self.scrollAreaWidget)
        layout.addWidget(self.scrollArea)

        self.selectAllButton = selectAllPushButton()
        self.selectAllButton.sigClicked.connect(self.checkAll)
        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(self.selectAllButton)

        if favourite_funcs is not None:
            self.loadFavouritesButton = reloadPushButton(
                '  Load last selection...  '
            )
            self.loadFavouritesButton.clicked.connect(self.checkFavouriteFuncs)
            # self.checkFavouriteFuncs()
            buttonsLayout.addWidget(self.loadFavouritesButton)

        layout.addLayout(buttonsLayout)

        self.setTitle(title)
        self.setCheckable(True)
        self.setLayout(layout)
        _font = QFont()
        _font.setPixelSize(11)
        self.setFont(_font)

        self.toggled.connect(self.toggled_cb)
    
    def highlightCheckboxesFromSearchText(self, text):
        for checkbox in self.checkBoxes:
            if not text:
                highlighted = False
            else:
                highlighted = checkbox.text().lower().find(text.lower()) != -1
            
            self.setCheckboxHighlighted(highlighted, checkbox)
    
    def setCheckboxHighlighted(self, highlighted, checkbox):
        if highlighted:
            checkbox.setStyleSheet(
                f'background: {self._highlightStylesheetColor}; color: black'
            )
            self.scrollArea.ensureWidgetVisible(checkbox)
        else:
            checkbox.setStyleSheet('')
    
    def onDelClicked(self):
        button = self.sender()
        button.checkbox.setChecked(False)
        self.sigDelClicked.emit(button.colname, button._layout)
    
    def toggled_cb(self, checked):
        for checkbox in self.checkBoxes:
            if not checked:
                self.checkedState[checkbox] = checkbox.isChecked()
                checkbox.setChecked(False)
            else:
                checkbox.setChecked(self.checkedState[checkbox])

    def checkFavouriteFuncs(self, checked=True, isZstack=False):
        self.doNotWarn = True
        if self._parent is not None:
            self._parent.doNotWarn = True
        for checkBox in self.checkBoxes:
            checkBox.setChecked(False)
            for favourite_func in self.favourite_funcs:
                func_name = checkBox.text()
                if func_name.endswith(favourite_func):
                    checkBox.setChecked(True)
                    break
        self.doNotWarn = False
        if self._parent is not None:
            self._parent.doNotWarn = False

    def checkAll(self, button, checked):
        if self._parent is not None:
            self._parent.doNotWarn = True
        for checkBox in self.checkBoxes:
            checkBox.setChecked(checked)
        if self._parent is not None:
            self._parent.doNotWarn = False

    def showInfo(self, checked=False):
        info_txt = self.sender().info
        msg = myMessageBox()
        msg.setWidth(600)
        msg.setIcon()
        msg.setWindowTitle(f'{self.sender().colname} info')
        msg.addText(info_txt)
        msg.addButton('   Ok   ')
        msg.exec_()

    def show(self):
        super().show()
        fw = self.inner_layout.contentsRect().width()
        sw = self.scrollArea.verticalScrollBar().sizeHint().width()
        self.minWidth = fw + sw

class channelMetricsQGBox(QGroupBox):
    sigDelClicked = Signal(str, object)
    sigCheckboxToggled = Signal(object)

    def __init__(
            self, isZstack, chName, isSegm3D, is_concat=False,
            posData=None, favourite_funcs=None
        ):
        QGroupBox.__init__(self)

        self.doNotWarn = False
        self.is_concat = is_concat
        isManualBackgrPresent = False
        if posData is not None:
            if posData.manualBackgroundLab is not None:
                isManualBackgrPresent = True

        layout = QVBoxLayout()
        metrics_desc, bkgr_val_desc = measurements.standard_metrics_desc(
            isZstack, chName, isSegm3D=isSegm3D, 
            isManualBackgrPresent=isManualBackgrPresent
        )

        metricsQGBox = _metricsQGBox(
            metrics_desc, 'Standard measurements',
            favourite_funcs=favourite_funcs, 
            parent=self
        )
        
        bkgrValsQGBox = _metricsQGBox(
            bkgr_val_desc, 'Background values',
            favourite_funcs=favourite_funcs, 
            parent=self
        )

        self.checkBoxes = metricsQGBox.checkBoxes.copy()
        self.checkBoxes.extend(bkgrValsQGBox.checkBoxes)

        self.uncheckAndDisableDataPrepIfPosNotPrepped(posData)

        self.groupboxes = [metricsQGBox, bkgrValsQGBox]

        for checkbox in metricsQGBox.checkBoxes:
            checkbox.toggled.connect(self.standardMetricToggled)
            self.standardMetricToggled(checkbox.isChecked(), checkbox=checkbox)
        
        for bkgrCheckbox in bkgrValsQGBox.checkBoxes:
            bkgrCheckbox.toggled.connect(self.backgroundMetricToggled)

        layout.addWidget(metricsQGBox)
        layout.addWidget(bkgrValsQGBox)

        items = measurements.custom_metrics_desc(
            isZstack, chName, posData=posData, isSegm3D=isSegm3D,
            return_combine=True
        )
        custom_metrics_desc, combine_metrics_desc = items

        if custom_metrics_desc:
            customMetricsQGBox = _metricsQGBox(
                custom_metrics_desc, 'Custom measurements', 
                delButtonMetricsDesc=combine_metrics_desc,
                favourite_funcs=favourite_funcs
            )
            layout.addWidget(customMetricsQGBox)
            self.checkBoxes.extend(customMetricsQGBox.checkBoxes)
            customMetricsQGBox.sigDelClicked.connect(self.onDelClicked)
            self.customMetricsQGBox = customMetricsQGBox

        self.setTitle(f'{chName} metrics')
        self.setCheckable(True)
        self.setLayout(layout)
    
    def uncheckAndDisableDataPrepIfPosNotPrepped(self, posData):
        # Uncheck and disable dataprep metrics if pos is not prepped
        if posData is None:
            return

        if posData.isBkgrROIpresent():
            return

        for checkbox in self.checkBoxes:
            if checkbox.text().find('dataPrep') == -1:
                continue

            checkbox.setChecked(False)
            checkbox.isDataPrepDisabled = True
    
    def _warnDataPrepCannotBeChecked(self):
        if self.doNotWarn:
            return
        txt = html_utils.paragraph("""
            <b>Data prep measurements cannot be saved</b> because you did 
            not select any background ROI at the data prep step.<br><br>

            You can read more details about data prep metrics by clicking 
            on the info button besides the measurement's name.<br><br>

            Thank you for you patience!
        """)
        msg = myMessageBox(showCentered=False)
        msg.warning(self, 'Metric cannot be saved', txt)

    def standardMetricToggled(self, checked, checkbox=None):
        """Method called when a check-box is toggled. It performs the following 
        actions:
            1. If the user try to check a data prep measurement, such as 
            dataPrep_amount, and this cannot be saved (checkbox has the attr 
            `isDataPrepDisabled`) then it warns and explains why it cannot be saved
            2. Make sure that background value median is checked if the user 
            requires amount or concentration metric.
            3. Do not allow unchecking background value median and explain why.

        Parameters
        ----------
        checked : bool
            State of the checkbox toggled
        checkbox : QtWidgets.QCheckBox, optional
            The checkbox that has been toggled. Default is None. If None 
            use `self.sender()`
        """        
        if self.is_concat:
            return
        
        if checkbox is None:
            checkbox = self.sender()

        if hasattr(checkbox, 'isDataPrepDisabled'):
            # Warn that user cannot check data prep metrics and uncheck it
            if not checkbox.isChecked():
                return
            checkbox.setChecked(False)
            self._warnDataPrepCannotBeChecked()
            return

        self.sigCheckboxToggled.emit(checkbox)
        if checkbox.text().find('amount_') == -1:
            return
        pattern = r'amount_([A-Za-z]+)(_?[A-Za-z0-9]*)'
        repl = r'\g<1>_bkgrVal_median\g<2>'
        bkgrValMetric = s1 = re.sub(pattern, repl, checkbox.text())
        for bkgrCheckbox in self.groupboxes[1].checkBoxes:
            if bkgrCheckbox.text() == bkgrValMetric:
                break
        else:
            # Make sure to not check for similarly named custom metrics
            return
        
        if checked:
            bkgrCheckbox.setChecked(True)
            bkgrCheckbox.isRequired = True
        else:
            bkgrCheckbox.setDisabled(False)
            bkgrCheckbox.isRequired = False
    
    def backgroundMetricToggled(self, checked):
        """Method called when a checkbox of a background metric is toggled.
        Check if the background value is required and explain why it cannot be 
        unchecked.

        Parameters
        ----------
        checked : bool
            State of the checkbox toggled
        """
        if self.is_concat:
            return
        
        checkbox = self.sender()
        if not hasattr(checkbox, 'isRequired'):
            return
        
        if not checkbox.isRequired:
            return
        
        if checkbox.isChecked():
            return
        
        if self.doNotWarn:
            return
        
        checkbox.setChecked(True)
        txt = html_utils.paragraph("""
            <b>This background value cannot be unchecked</b> because it is required 
            by the <code>_amount</code> and <code>_concentration</code> measurements 
            that you requested to save.<br><br>

            Thank you for you patience!
        """)
        msg = myMessageBox(showCentered=False)
        msg.warning(self, 'Background value required', txt)
    
    def onDelClicked(self, colname_to_del, hlayout):
        self.sigDelClicked.emit(colname_to_del, hlayout)
    
    def checkFavouriteFuncs(self):
        self.doNotWarn = True
        for groupbox in self.groupboxes:
            groupbox.checkFavouriteFuncs()
        self.doNotWarn = False

class PixelSizeGroupbox(QGroupBox):
    sigValueChanged = Signal(float, float, float)
    sigReset = Signal()
    
    def __init__(self, parent=None):
        super().__init__('Pixel size', parent)
        
        mainLayout = QGridLayout()
        
        row = 0
        label = QLabel('Pixel width (μm): ')
        self.pixelWidthWidget = FloatLineEdit(initial=1.0)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.pixelWidthWidget, row, 1)
        
        row += 1
        label = QLabel('Pixel height (μm): ')
        self.pixelHeightWidget = FloatLineEdit(initial=1.0)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.pixelHeightWidget, row, 1)
        
        row += 1
        label = QLabel('Voxel depth (μm): ')
        self.voxelDepthWidget = FloatLineEdit(initial=1.0)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.voxelDepthWidget, row, 1)
        
        row += 1
        resetButton = reloadPushButton('Reset')
        mainLayout.addWidget(
            resetButton, row, 1, alignment=Qt.AlignRight
        )
        
        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)
        
        mainLayout.setColumnStretch(0, 0)
        mainLayout.setColumnStretch(1, 1)

        self.setLayout(mainLayout)
        
        self.pixelWidthWidget.valueChanged.connect(self.emitValueChanged)
        self.pixelHeightWidget.valueChanged.connect(self.emitValueChanged)
        self.voxelDepthWidget.valueChanged.connect(self.emitValueChanged)
        resetButton.clicked.connect(self.emitReset)
    
    def emitReset(self):
        self.sigReset.emit()
    
    def emitValueChanged(self, value):
        PhysicalSizeX = self.pixelWidthWidget.value()
        PhysicalSizeY = self.pixelHeightWidget.value()
        PhysicalSizeZ = self.voxelDepthWidget.value()
        self.sigValueChanged.emit(PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)

class objPropsQGBox(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, 'Properties', parent)

        mainLayout = QGridLayout()

        row = 0
        label = QLabel('Object ID: ')
        self.idSB = IntLineEdit()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.idSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        self.notExistingIDLabel = QLabel()
        self.notExistingIDLabel.setStyleSheet(
            'font-size:11px; color: rgb(255, 0, 0);'
        )
        mainLayout.addWidget(
            self.notExistingIDLabel, row, 0, 1, 2, alignment=Qt.AlignCenter
        )

        row += 1
        label = QLabel('Area (pixel): ')
        self.cellAreaPxlSB = IntLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellAreaPxlSB, row, 1)

        row += 1
        label = QLabel('Area (<span>&#181;</span>m<sup>2</sup>): ')
        self.cellAreaUm2DSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellAreaUm2DSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        label = QLabel('Rotational volume (voxel): ')
        self.cellVolVoxSB = IntLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolVoxSB, row, 1)

        row += 1
        label = QLabel('3D volume (voxel): ')
        self.cellVolVox3D_SB = IntLineEdit(readOnly=True)
        self.cellVolVox3D_SB.label = label
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolVox3D_SB, row, 1)

        row += 1
        label = QLabel('Rotational volume (fl): ')
        self.cellVolFlDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolFlDSB, row, 1)

        row += 1
        label = QLabel('3D volume (fl): ')
        self.cellVolFl3D_DSB = FloatLineEdit(readOnly=True)
        self.cellVolFl3D_DSB.label = label
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolFl3D_DSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        label = QLabel('Solidity: ')
        self.solidityDSB = FloatLineEdit(readOnly=True)
        self.solidityDSB.setMaximum(1)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.solidityDSB, row, 1)

        row += 1
        label = QLabel('Elongation: ')
        self.elongationDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.elongationDSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        propsNames = measurements.get_props_names()[1:]
        self.additionalPropsCombobox = QComboBox()
        self.additionalPropsCombobox.addItems(propsNames)
        self.additionalPropsCombobox.indicator = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(self.additionalPropsCombobox, row, 0)
        mainLayout.addWidget(self.additionalPropsCombobox.indicator, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)
        
        mainLayout.setColumnStretch(0, 0)
        mainLayout.setColumnStretch(1, 1)

        self.setLayout(mainLayout)

class objIntesityMeasurQGBox(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, 'Intensity measurements', parent)

        mainLayout = QGridLayout()

        row = 0
        label = QLabel('Raw intensity measurements')

        row += 1
        label = QLabel('Channel: ')
        self.channelCombobox = QComboBox()
        self.channelCombobox.addItem('placeholderlong')
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.channelCombobox, row, 1)

        row += 1
        label = QLabel('Minimum: ')
        self.minimumDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.minimumDSB, row, 1)

        row += 1
        label = QLabel('Maximum: ')
        self.maximumDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.maximumDSB, row, 1)

        row += 1
        label = QLabel('Mean: ')
        self.meanDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.meanDSB, row, 1)

        row += 1
        label = QLabel('Median: ')
        self.medianDSB = FloatLineEdit(readOnly=True)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.medianDSB, row, 1)

        row += 1
        metricsDesc = measurements._get_metrics_names()
        metricsFunc, _ = measurements.standard_metrics_func()
        items = list(set([metricsDesc[key] for key in metricsFunc.keys()]))
        items.append('Concentration')
        items.sort()
        nameFuncDict = {}
        for name, desc in metricsDesc.items():
            if name.find('_dataPrepBkgr')!=-1 or  name.find('_manualBkgr')!=-1:
                # Skip dataPrepBkgr and manualBkgr since in the dock widget 
                # we display only autoBkgr metrics
                continue
            if name.startswith('concentration_'):
                # We use amount function because dividing by volume is taken 
                # care in the GUI
                name = 'amount_autoBkgr'
            nameFuncDict[desc] = metricsFunc[name]

        funcionCombobox = QComboBox()
        funcionCombobox.addItems(items)
        self.additionalMeasCombobox = funcionCombobox
        self.additionalMeasCombobox.indicator = FloatLineEdit(readOnly=True)
        self.additionalMeasCombobox.functions = nameFuncDict
        mainLayout.addWidget(funcionCombobox, row, 0)
        mainLayout.addWidget(self.additionalMeasCombobox.indicator, row, 1)

        self.setLayout(mainLayout)

    def addChannels(self, channels):
        self.channelCombobox.clear()
        self.channelCombobox.addItems(channels)

class guiTabControl(QTabWidget):
    def __init__(self, *args):
        super().__init__(args[0])

        self._defaultPixelSize = None
        
        self.propsTab = QScrollArea(self)

        container = QWidget()
        layout = QVBoxLayout()

        self.pixelSizeQGBox = PixelSizeGroupbox(parent=self.propsTab)
        self.propsQGBox = objPropsQGBox(parent=self.propsTab)
        self.intensMeasurQGBox = objIntesityMeasurQGBox(parent=self.propsTab)

        self.highlightCheckbox = QCheckBox('Highlight objects on mouse hover')
        self.highlightCheckbox.setChecked(False)
        
        self.highlightSearchedCheckbox = QCheckBox('Highlight searched object')
        self.highlightSearchedCheckbox.setChecked(True)

        highlightLayout = QHBoxLayout()
        highlightLayout.addWidget(self.highlightCheckbox)
        highlightLayout.addStretch(1)
        highlightLayout.addWidget(QLabel('|'))
        highlightLayout.addStretch(1)
        highlightLayout.addWidget(self.highlightSearchedCheckbox)
        
        layout.addLayout(highlightLayout)
        layout.addWidget(self.pixelSizeQGBox)
        layout.addWidget(self.propsQGBox)
        layout.addWidget(self.intensMeasurQGBox)  
        layout.addStretch(1)     
        container.setLayout(layout)

        self.propsTab.setWidgetResizable(True)
        self.propsTab.setWidget(container)
        self.addTab(self.propsTab, 'Measurements')
        
        self.pixelSizeQGBox.sigValueChanged.connect(self.pixelSizeChanged)
        self.pixelSizeQGBox.sigReset.connect(self.resetPixelSize)
    
    def addChannels(self, channels):
        self.intensMeasurQGBox.addChannels(channels)
    
    def resetPixelSize(self):
        if self._defaultPixelSize is None:
            return
        
        self.initPixelSize(*self._defaultPixelSize)
    
    def initPixelSize(self, PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ):
        self.pixelSizeQGBox.pixelWidthWidget.setValue(PhysicalSizeX)
        self.pixelSizeQGBox.pixelHeightWidget.setValue(PhysicalSizeY)
        self.pixelSizeQGBox.voxelDepthWidget.setValue(PhysicalSizeZ)
        self._defaultPixelSize = (PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)
    
    def pixelSizeChanged(self, PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ):
        propsQGBox = self.propsQGBox
        yx_pxl_to_um2 = PhysicalSizeY*PhysicalSizeX
        vox_rot_to_fl = float(PhysicalSizeY)*pow(float(PhysicalSizeX), 2)
        vox_3D_to_fl = PhysicalSizeZ*PhysicalSizeY*PhysicalSizeX
        
        area_pxl = propsQGBox.cellAreaPxlSB.value()
        area_um2 = area_pxl*yx_pxl_to_um2
        propsQGBox.cellAreaUm2DSB.setValue(area_um2)
        
        vol_rot_vox = propsQGBox.cellVolVoxSB.value()
        vol_rot_fl = vol_rot_vox*vox_rot_to_fl
        propsQGBox.cellVolFlDSB.setValue(vol_rot_fl)
        
        vol_3D_vox = propsQGBox.cellVolVox3D_SB.value()
        vol_3D_fl = vol_3D_vox*vox_3D_to_fl
        propsQGBox.cellVolFl3D_DSB.setValue(vol_3D_fl)
        

class expandCollapseButton(PushButton):
    sigClicked = Signal()

    def __init__(self, parent=None, **kwargs):
        super().__init__(parent, **kwargs)
        self.setIcon(QIcon(":expand.svg"))
        self.setFlat(True)
        self.installEventFilter(self)
        self.isExpand = True
        self.clicked.connect(self.buttonClicked)

    def buttonClicked(self, checked=False):
        if self.isExpand:
            self.setIcon(QIcon(":collapse.svg"))
            self.isExpand = False
            if self.text():
                self.setText(self.text().replace('Hide', 'Show'))
        else:
            self.setIcon(QIcon(":expand.svg"))
            self.isExpand = True
            if self.text():
                self.setText(self.text().replace('Show', 'Hide'))
        self.sigClicked.emit()

    def eventFilter(self, object, event):
        if event.type() == QEvent.Type.HoverEnter:
            self.setFlat(False)
        elif event.type() == QEvent.Type.HoverLeave:
            self.setFlat(True)
        return False

class view_visualcpp_screenshot(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()

        self.setWindowTitle('Visual Studio Builld Tools installation')

        pixmap = QPixmap(':visualcpp.png')
        label = QLabel()
        label.setPixmap(pixmap)

        layout.addWidget(label)
        self.setLayout(layout)

class PolyLineROI(pg.PolyLineROI):
    def __init__(self, positions, closed=False, pos=None, **args):
        super().__init__(positions, closed, pos, **args)

class BaseGradientEditorItemImage(pg.GradientEditorItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
    
    def restoreState(self, state):
        pg.graphicsItems.GradientEditorItem.Gradients = GradientsImage
        return super().restoreState(state)

class MouseCursor(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._x = None
        self._y = None
        self.setMouseTracking(True)
    
    def mouseMoveEvent(self, event) -> None:
        self.move(event.pos())
        self.update()
        return super().mouseMoveEvent(event)
    
    # def drawAtPos(self, x, y):
    #     self._x = x
    #     self._y = y
    #     self.update()
    
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        # p.setPen(QPen(QColor(0,0,0)))
        # p.setBrush(QBrush(QColor(70,70,70,200)))
        p.drawLine(0,0,200,0)
        p.end()

class BaseGradientEditorItemLabels(pg.GradientEditorItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
    
    def restoreState(self, state):
        pg.graphicsItems.GradientEditorItem.Gradients = GradientsLabels
        return super().restoreState(state)

class baseHistogramLUTitem(pg.HistogramLUTItem):
    sigAddColormap = Signal(object, str)
    sigRescaleIntes = Signal(object)

    def __init__(self, name='image', axisLabel='', parent=None, **kwargs):
        pg.GradientEditorItem = BaseGradientEditorItemLabels

        super().__init__(**kwargs)

        self.labelStyle = {'color': '#ffffff', 'font-size': '11px'}

        if axisLabel:
            self.setAxisLabel(axisLabel)

        self.cmaps = cmaps
        self._parent = parent
        self.name = name

        self.gradient.colorDialog.setWindowFlags(
            Qt.Dialog | Qt.WindowStaysOnTopHint
        )
        self.gradient.colorDialog.accepted.disconnect()
        self.gradient.colorDialog.accepted.connect(self.tickColorAccepted)

        self.isInverted = False
        self.lastGradientName = 'grey'
        self.lastGradient = Gradients['grey']

        for action in self.gradient.menu.actions():
            if action.text() == 'HSV':
                HSV_action = action
            elif action.text() == 'RGB':
                RGB_ation = action
        self.gradient.menu.removeAction(HSV_action)
        self.gradient.menu.removeAction(RGB_ation)
        
        # Rescale intensities (LUT)
        rescaleIntensMenu = self.gradient.menu.addMenu(
            'Rescale intensities (LUT)'
        )
        rescaleActionGroup = QActionGroup(self)
        rescaleActionGroup.setExclusive(True)
        
        self.rescaleEach2DimgAction = QAction(
            'Rescale each 2D image', rescaleIntensMenu
        )
        self.rescaleEach2DimgAction.setCheckable(True)
        self.rescaleEach2DimgAction.setChecked(True)
        rescaleActionGroup.addAction(self.rescaleEach2DimgAction)
        rescaleIntensMenu.addAction(self.rescaleEach2DimgAction)
        
        self.rescaleAcrossZstackAction = QAction(
            'Rescale across z-stack', rescaleIntensMenu
        )
        self.rescaleAcrossZstackAction.setCheckable(True)
        self.rescaleAcrossZstackAction.setChecked(False)
        rescaleActionGroup.addAction(self.rescaleAcrossZstackAction)
        rescaleIntensMenu.addAction(self.rescaleAcrossZstackAction)
        
        self.rescaleAcrossTimeAction = QAction(
            'Rescale across time frames', rescaleIntensMenu
        )
        self.rescaleAcrossTimeAction.setCheckable(True)
        self.rescaleAcrossTimeAction.setChecked(False)
        rescaleActionGroup.addAction(self.rescaleAcrossTimeAction)
        rescaleIntensMenu.addAction(self.rescaleAcrossTimeAction)
        
        self.customRescaleAction = QAction(
            'Choose custom levels...', rescaleIntensMenu
        )
        self.customRescaleAction.setCheckable(True)
        rescaleActionGroup.addAction(self.customRescaleAction)
        rescaleIntensMenu.addAction(self.customRescaleAction)
        
        self.doNotRescaleAction = QAction(
            'Do no rescale, display raw image', rescaleIntensMenu
        )
        self.doNotRescaleAction.setCheckable(True)
        rescaleActionGroup.addAction(self.doNotRescaleAction)
        rescaleIntensMenu.addAction(self.doNotRescaleAction)
        
        self.rescaleActionGroup = rescaleActionGroup
        rescaleActionGroup.triggered.connect(self.rescaleActionTriggered)

        # Add custom colormap action
        self.customCmapsMenu = self.gradient.menu.addMenu('Custom colormaps')
        self.customCmapsMenu.aboutToShow.connect(self.onShowCustomCmapsMenu)
        self.customCmapsMenu.triggered.connect(self.customCmapsMenuTriggered)
        
        self.saveColormapAction = QAction(
            'Save current colormap...', self
        )
        self.gradient.menu.addAction(self.saveColormapAction)
        self.saveColormapAction.triggered.connect(
            self.saveColormap
        )

        self.addCustomGradients()

        # Set inverted gradients for invert bw action
        self.addInvertedColorMaps()

        self.gradient.menu.addSeparator()

        # hide histogram tool
        self.vb.hide()

        # Disable histogram default context Menu event
        self.vb.raiseContextMenu = lambda x: None
    
    def rescaleActionTriggered(self, action):
        self.sigRescaleIntes.emit(action)
    
    def onShowCustomCmapsMenu(self):
        self.customCmapsMenu.show()
    
    def customCmapsMenuTriggered(self, action):
        cmap = action.cmap
        self.gradient.colorMapMenuClicked(cmap)
        self.gradient.showTicks(True)
    
    def setAxisLabel(self, text):
        self.labelText = text
        self.axis.setLabel(text, **self.labelStyle)
    
    def updateAxisLabel(self):
        text = self.axis.label.toPlainText()
        if not text:
            return
        self.setAxisLabel(text)
    
    def setGradient(self, gradient):
        self.gradient.restoreState(gradient)
        self.lastGradient = gradient
    
    def colormapClicked(self, checked=False, name=None):
        name = self.sender().name
        self.lastGradientName = name
        if self.isInverted:
            self.setGradient(self.invertedGradients[name])
        else:
            self.setGradient(Gradients[name])

    def sortTicks(self, ticks):
        sortedTicks = sorted(ticks, key=operator.itemgetter(0))
        return sortedTicks
    
    def getInvertedGradients(self):
        invertedGradients = {}
        for name, gradient in Gradients.items():
            ticks = gradient['ticks']
            sortedTicks = self.sortTicks(ticks)
            if name in nonInvertibleCmaps:
                invertedColors = sortedTicks
            else:
                invertedColors = [
                    (t[0], ti[1]) 
                    for t, ti in zip(sortedTicks, sortedTicks[::-1])
                ]
            invertedGradient = {}
            invertedGradient['ticks'] = invertedColors
            invertedGradient['mode'] = gradient['mode']
            invertedGradients[name] = invertedGradient
        return invertedGradients
    
    def addInvertedColorMaps(self):
        self.invertedGradients = self.getInvertedGradients()
        for action in self.gradient.menu.actions():
            if not hasattr(action, 'name'):
                continue
            
            if action.name not in self.cmaps:
                continue
            
            action.triggered.disconnect()
            action.triggered.connect(self.colormapClicked)

            px = QPixmap(100, 15)
            p = QPainter(px)
            invertedGradient = self.invertedGradients[action.name]
            qtGradient = QLinearGradient(QPointF(0,0), QPointF(100,0))
            ticks = self.sortTicks(invertedGradient['ticks'])
            qtGradient.setStops([(x, QColor(*color)) for x,color in ticks])
            brush = QBrush(qtGradient)
            p.fillRect(QRect(0, 0, 100, 15), brush)
            p.end()
            widget = action.defaultWidget()
            hbox = widget.layout()
            rectLabelWidget = QLabel()
            rectLabelWidget.setPixmap(px)
            hbox.addWidget(rectLabelWidget)
            rectLabelWidget.hide()
    
    def setInvertedColorMaps(self, inverted):
        if inverted:
            showIdx = 2
            hideIdx = 1
            self.labelStyle['color'] = '#000000'
        else:
            showIdx = 1
            hideIdx = 2
            self.labelStyle['color'] = '#ffffff'
        
        for action in self.gradient.menu.actions():
            if not hasattr(action, 'name'):
                continue
            
            if action.name not in self.cmaps:
                continue

            widget = action.defaultWidget()
            hbox = widget.layout()
            hideCmapRect = hbox.itemAt(hideIdx).widget()
            showCmapRect = hbox.itemAt(showIdx).widget()
            hideCmapRect.hide()
            showCmapRect.show()
        
        self.updateAxisLabel()
        self.isInverted = inverted
    
    def invertGradient(self, gradient):
        ticks = gradient['ticks']
        sortedTicks = self.sortTicks(ticks)
        invertedColors = [
            (t[0], ti[1]) 
            for t, ti in zip(sortedTicks, sortedTicks[::-1])
        ]
        invertedGradient = {}
        invertedGradient['ticks'] = invertedColors
        invertedGradient['mode'] = gradient['mode']
        return invertedGradient
    
    def invertCurrentColormap(self, inverted, debug=False):
        self.setGradient(self.invertGradient(self.lastGradient))
    
    def addCustomGradient(self, gradient_name, gradient_ticks, restore=True):
        self.originalLength = self.gradient.length
        self.gradient.length = 100
        if restore:
            self.gradient.restoreState(gradient_ticks)
        gradient = self.gradient.getGradient()
        action = CustomGradientMenuAction(gradient, gradient_name, self.gradient)
        # action.triggered.connect(self.gradient.contextMenuClicked)
        action.delButton.clicked.connect(self.removeCustomGradient)
        action.cmap = colors.pg_ticks_to_colormap(gradient_ticks['ticks'])
        # self.gradient.menu.insertAction(self.saveColormapAction, action)
        self.customCmapsMenu.addAction(action)
        self.gradient.length = self.originalLength
        GradientsImage[gradient_name] = gradient_ticks
    
    def removeCustomGradient(self):
        button = self.sender()
        action = button.action
        self.customCmapsMenu.removeAction(action)
        cp = config.ConfigParser()
        cp.read(custom_cmaps_filepath)
        cp.remove_section(f'image.{action.name}')
        with open(custom_cmaps_filepath, mode='w') as file:
            cp.write(file)
    
    def addCustomGradients(self):
        try:
            CustomGradients = getCustomGradients(name='image')
            if not CustomGradients:
                return
            for gradient_name, gradient_ticks in CustomGradients.items():
                self.addCustomGradient(gradient_name, gradient_ticks)
        except Exception as e:
            printl(traceback.format_exc())
            pass
    
    def _askNameColormap(self):
        inputWin = apps.QInput(parent=self._parent, title='Colormap name')
        inputWin.askText('Insert a name for the colormap: ', allowEmpty=False)
        if inputWin.cancel:
            return
        cmapName = inputWin.answer
        return cmapName
    
    def saveColormap(self):
        cmapName = self._askNameColormap()
        if cmapName is None:
            return
        
        cp = config.ConfigParser()
        if os.path.exists(custom_cmaps_filepath):
            cp.read(custom_cmaps_filepath)
        
        SECTION = f'{self.name}.{cmapName}'
        cp[SECTION] = {}

        # gradient_ticks = []
        state = self.gradient.saveState()
        for key, value in state.items():
            if key != 'ticks':
                continue
            for t, tick in enumerate(value):
                pos, rgb = tick
                # gradient_ticks.append((pos, rgb))
                rgb = ','.join([str(c) for c in rgb])
                val = f'{pos},{rgb}'
                cp[SECTION][f'tick_{t}_pos_rgb'] = val
        
        with open(custom_cmaps_filepath, mode='w') as file:
            cp.write(file)
        
        self.addCustomGradient(cmapName, state, restore=False)
    
    def tickColorAccepted(self):
        self.gradient.currentColorAccepted()
        # self.sigTickColorAccepted.emit(self.gradient.colorDialog.color().getRgb())

class ROI(pg.ROI):
    def __init__(
            self, pos, size=..., angle=0, invertible=False, maxBounds=None, 
            snapSize=1, scaleSnap=False, translateSnap=False, rotateSnap=False, 
            parent=None, pen=None, hoverPen=None, handlePen=None, 
            handleHoverPen=None, movable=True, rotatable=True, 
            resizable=True, removable=False, aspectLocked=False
        ):
        super().__init__(
            pos, size, angle, invertible, maxBounds, snapSize, scaleSnap, 
            translateSnap, rotateSnap, parent, pen, hoverPen, handlePen, 
            handleHoverPen, movable, rotatable, resizable, removable, 
            aspectLocked
        )
    
    def slice(self, zRange=None, tRange=None):
        x0, y0 = [int(round(c)) for c in self.pos()]
        w, h = [int(round(c)) for c in self.size()]
        xmin, xmax = x0, x0+w
        if xmin > xmax:
            xmin, xmax = xmax, xmin
        ymin, ymax = y0, y0+h
        if ymin > ymax:
            ymin, ymax = ymax, ymin
        if zRange is not None:
            zmin, zmax = zRange
            _slice = (slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax))
        else:
            _slice = (slice(ymin, ymax), slice(xmin, xmax))
        if tRange is not None:
            tmin, tmax = tRange
            _slice = (slice(tmin, tmax), *_slice)
        return _slice
        
class DelROI(pg.ROI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def clearPoints(self):
        """
        Remove all handles and segments.
        """
        while len(self.handles) > 0:
            self.removeHandle(self.handles[0]['item'])

class PlotCurveItem(pg.PlotCurveItem):    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
    
    def addPoint(self, x, y, **kargs):
        _xx, _yy = self.getData()
        if _xx is None or len(_xx) == 0:
            self.xData = np.array([x], dtype=int)
            self.yData = np.array([y], dtype=int)
            return
        if _xx[-1] == x and _yy[-1] == y:
            # Do not append same point
            return
        
        # Pre-allocate array and insert data (faster than append)
        xx = np.zeros(len(_xx)+1, dtype=_xx.dtype)
        xx[:-1] = _xx
        xx[-1] = x
        yy = np.zeros(len(_yy)+1, dtype=_xx.dtype)
        yy[:-1] = _yy
        yy[-1] = y
        self.setData(xx, yy, **kargs)
    
    def clear(self):
        try:
            self.setData([], [])
        except Exception as e:
            pass
        super().clear()
        
    
    def closeCurve(self):
        _xx, _yy = self.getData()
        self.addPoint(_xx[0], _yy[0])
    
    def mask(self):
        ymin, xmin, ymax, xmax = self.bbox()
        _mask = np.zeros((ymax-ymin+1, xmax-xmin+1), dtype=bool)
        local_xx, local_yy = self.getLocalData()
        rr, cc = skimage.draw.polygon(local_yy, local_xx)
        _mask[rr, cc] = True
        return _mask
    
    def getLocalData(self):
        _xx, _yy = self.getData()
        return _xx - _xx.min(), _yy - _yy.min()

    def slice(self, zRange=None, tRange=None):
        ymin, xmin, ymax, xmax = self.bbox()
        if zRange is not None:
            zmin, zmax = zRange
            _slice = (slice(zmin, zmax), slice(ymin, ymax+1), slice(xmin, xmax+1))
        else:
            _slice = (slice(ymin, ymax+1), slice(xmin, xmax+1))
        if tRange is not None:
            tmin, tmax = tRange
            _slice = (slice(tmin, tmax), *_slice)
        return _slice
    
    def bbox(self):
        _xx, _yy = self.getData()
        return _yy.min(), _xx.min(), _yy.max(), _xx.max()

class ToggleVisibilityButton(PushButton):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFlat(True)
        # self.setCheckable(True)
        self._state = False
        self.setIcon(QIcon(':unchecked.svg'))
        self.clicked.connect(self.onClicked)
        self.setStyleSheet("""
            QPushButton::pressed {
                background-color: none;
                border-style: none;
            }
        """)
        
    def onClicked(self):
        self._state = not self._state
        if self._state:
            self.setIcon(QIcon(':eye-checked.svg'))
        else:
            self.setIcon(QIcon(':unchecked.svg'))

class ToggleVisibilityCheckBox(QCheckBox):
    def __init__(self, *args, pixelSize=24):
        super().__init__(*args)
        self._pixelSize = pixelSize
        self.onToggled(False)
        self.toggled.connect(self.onToggled)
        
    def setPixelSize(self, pixelSize):
        self._pixelSize = pixelSize
        
    def onToggled(self, checked):
        if checked:
            self.setStyleSheet(f"""
                QCheckBox::indicator {{
                    width: {self._pixelSize}px;
                    height: {self._pixelSize}px;
                }}

                QCheckBox::indicator:checked
                {{
                    image: url(:eye-checked.svg);
                }}
            """)
        else:
            self.setStyleSheet(f"""
                QCheckBox::indicator {{
                    width: {self._pixelSize}px;
                    height: {self._pixelSize}px;
                }}
                
                QCheckBox::indicator:unchecked
                {{
                    image: url(:unchecked.svg);
                }}
            """)


class myHistogramLUTitem(baseHistogramLUTitem):
    sigGradientMenuEvent = Signal(object)
    sigTickColorAccepted = Signal(object)
    sigAddScaleBar = Signal(bool)
    sigAddTimestamp = Signal(bool)

    def __init__(
            self, parent=None, name='image', axisLabel='', isViewer=False, 
            **kwargs
        ):
        super().__init__(
            parent=parent, name=name, axisLabel=axisLabel, **kwargs
        )

        self.name = name
        self._parent = parent
        
        self.childLutItem = None

        self.isViewer = isViewer
        if isViewer:
            # In the viewer we don't allow additional settings from the menu
            return
        
        # Add scale bar action
        self.addScaleBarAction = QAction('Add scale bar', self)
        self.addScaleBarAction.setCheckable(True)
        self.addScaleBarAction.triggered.connect(self.emitAddScaleBar)
        self.gradient.menu.addAction(self.addScaleBarAction)
        
        # Add timestamp action
        self.addTimestampAction = QAction('Add timestamp', self)
        self.addTimestampAction.setCheckable(True)
        self.addTimestampAction.triggered.connect(self.emitAddTimestamp)
        self.gradient.menu.addAction(self.addTimestampAction)

        # Invert bw action
        self.invertBwAction = QAction('Invert black/white', self)
        self.invertBwAction.setCheckable(True)
        self.gradient.menu.addAction(self.invertBwAction)

        # Font size menu action
        self.fontSizeMenu =  QMenu('Text font size')
        self.gradient.menu.addMenu(self.fontSizeMenu) 

        # Text color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Text color: '))
        self.textColorButton = myColorButton(color=(255,255,255))
        hbox.addStretch(1)
        hbox.addWidget(self.textColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.textColorButton.click)
        self.gradient.menu.addAction(act)

        # Contours line weight
        contLineWeightMenu = QMenu('Contours line weight', self.gradient.menu)
        self.contLineWightActionGroup = QActionGroup(self)
        self.contLineWightActionGroup.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.Exclusive
        )
        for w in range(1, 11):
            action = QAction(str(w))
            action.setCheckable(True)
            if w == 2:
                action.setChecked(True)
            action.lineWeight = w
            self.contLineWightActionGroup.addAction(action)
            action = contLineWeightMenu.addAction(action)
        self.gradient.menu.addMenu(contLineWeightMenu)

        # Contours color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Contours color: '))
        self.contoursColorButton = myColorButton(color=(25,25,25))
        hbox.addStretch(1)
        hbox.addWidget(self.contoursColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.contoursColorButton.click)
        self.gradient.menu.addAction(act)

        # Mother-bud line weight
        mothBudLineWeightMenu = QMenu('Mother-bud line weight', self.gradient.menu)
        self.mothBudLineWightActionGroup = QActionGroup(self)
        self.mothBudLineWightActionGroup.setExclusionPolicy(
            QActionGroup.ExclusionPolicy.Exclusive
        )
        for w in range(1, 11):
            action = QAction(str(w))
            action.setCheckable(True)
            if w == 2:
                action.setChecked(True)
            action.lineWeight = w
            self.mothBudLineWightActionGroup.addAction(action)
            action = mothBudLineWeightMenu.addAction(action)
        self.gradient.menu.addMenu(mothBudLineWeightMenu)

        # Mother-bud line color
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Mother-bud line color: '))
        self.mothBudLineColorButton = myColorButton(color=(255,0,0))
        hbox.addStretch(1)
        hbox.addWidget(self.mothBudLineColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.mothBudLineColorButton.click)
        self.gradient.menu.addAction(act)

        self.labelsAlphaMenu = self.gradient.menu.addMenu(
            'Segm. masks overlay alpha...'
        )
        # self.labelsAlphaMenu.setDisabled(True)
        hbox = QHBoxLayout()
        self.labelsAlphaSlider = sliderWithSpinBox(
            title='Alpha', title_loc='in_line', isFloat=True,
            normalize=True
        )
        self.labelsAlphaSlider.setMaximum(100)
        self.labelsAlphaSlider.setSingleStep(0.05)
        self.labelsAlphaSlider.setValue(0.3)
        hbox.addWidget(self.labelsAlphaSlider)
        shortCutText = 'Command+Up/Down' if is_mac else 'Ctrl+Up/Down'
        hbox.addWidget(QLabel(f'({shortCutText})'))
        widget = QWidget()
        widget.setLayout(hbox)
        act = QWidgetAction(self)
        act.setDefaultWidget(widget)
        self.labelsAlphaMenu.addSeparator()
        self.labelsAlphaMenu.addAction(act)

        # Default settings
        self.defaultSettingsAction = QAction('Restore default settings...', self)
        self.gradient.menu.addAction(self.defaultSettingsAction)

        self.filterObject = FilterObject()
        self.filterObject.sigFilteredEvent.connect(self.gradientMenuEventFilter)
        self.gradient.menu.installEventFilter(self.filterObject)
        self.highlightedAction = None
        self.lastHoveredAction = None
    
    def setChildLutItem(self, childLutItem):
        self.childLutItem = childLutItem
    
    def removeAddScaleBarAction(self):
        self.gradient.menu.removeAction(self.addScaleBarAction)
    
    def removeAddTimestampAction(self):
        self.gradient.menu.removeAction(self.addTimestampAction)
    
    def emitAddScaleBar(self):
        self.sigAddScaleBar.emit(self.addScaleBarAction.isChecked())
    
    def emitAddTimestamp(self):
        self.sigAddTimestamp.emit(self.addTimestampAction.isChecked())
    
    def gradientMenuEventFilter(self, object, event):
        if event.type() == QEvent.Type.MouseMove:
            hoveredAction = self.gradient.menu.actionAt(event.pos())
            isActionEntered = (
                hoveredAction != self.lastHoveredAction
            )
            if isActionEntered:
                if isinstance(hoveredAction, highlightableQWidgetAction):
                    # print('Entered a custom action')
                    pass
                isActionLeft = (
                    self.highlightedAction is not None
                    and self.highlightedAction != hoveredAction
                ) 
                if isActionLeft:
                    if isinstance(
                            self.highlightedAction, highlightableQWidgetAction
                        ):
                        # print('Left a custom action')
                        pass
                self.highlightedAction = hoveredAction

            self.lastHoveredAction = hoveredAction
    
    def addOverlayColorButton(self, rgbColor, channelName):
        # Overlay color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Overlay color: '))
        self.overlayColorButton = myColorButton(color=rgbColor)
        self.overlayColorButton.channel = channelName
        hbox.addStretch(1)
        hbox.addWidget(self.overlayColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.overlayColorButton.click)
        self.gradient.menu.addAction(act)

    def uncheckContLineWeightActions(self):
        for act in self.contLineWightActionGroup.actions():
            try:
                act.toggled.disconnect()
            except Exception as e:
                pass
            act.setChecked(False)

    def uncheckMothBudLineLineWeightActions(self):
        for act in self.mothBudLineWightActionGroup.actions():
            try:
                act.toggled.disconnect()
            except Exception as e:
                pass
            act.setChecked(False)

    def restoreState(self, df):
        if 'textIDsColor' in df.index:
            rgbString = df.at['textIDsColor', 'value']
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.textColorButton.setColor((r, g, b))

        if 'contLineColor' in df.index:
            rgba_str = df.at['contLineColor', 'value']
            rgb = colors.rgba_str_to_values(rgba_str)[:3]
            self.contoursColorButton.setColor(rgb)
        
        if 'contLineWeight' in df.index:
            w = df.at['contLineWeight', 'value']
            w = int(w)
            for action in self.contLineWightActionGroup.actions():
                if action.lineWeight == w:
                    action.setChecked(True)
                    break
        
        if 'mothBudLineWeight' in df.index:
            w = df.at['mothBudLineWeight', 'value']
            w = int(w)
            for action in self.mothBudLineWightActionGroup.actions():
                if action.lineWeight == w:
                    action.setChecked(True)
                    break

        if 'overlaySegmMasksAlpha' in df.index:
            alpha = df.at['overlaySegmMasksAlpha', 'value']
            self.labelsAlphaSlider.setValue(float(alpha))
        
        if 'mothBudLineColor' in df.index:
            rgba_str = df.at['mothBudLineColor', 'value']
            rgb = colors.rgba_str_to_values(rgba_str)[:3]
            self.mothBudLineColorButton.setColor(rgb)
        
        checked = df.at['is_bw_inverted', 'value'] == 'Yes'
        self.invertBwAction.setChecked(checked)

        self.restoreColormap(df)
    
    def saveState(self, df):
        # remove previous state
        df = df[~df.index.str.contains('img_cmap')].copy()

        state = self.gradient.saveState()
        for key, value in state.items():
            if key == 'ticks':
                for t, tick in enumerate(value):
                    pos, rgb = tick
                    df.at[f'img_cmap_tick{t}_rgb', 'value'] = rgb
                    df.at[f'img_cmap_tick{t}_pos', 'value'] = pos
            else:
                if isinstance(value, bool):
                    value = 'Yes' if value else 'No'
                df.at[f'img_cmap_{key}', 'value'] = value
        return df
    
    def restoreColormap(self, df):
        state = {'mode': 'rgb', 'ticksVisible': True, 'ticks': []}
        ticks_pos = {}
        ticks_rgb = {}
        stateFound = False
        for setting, value in df.itertuples():
            idx = setting.find('img_cmap_')
            if idx == -1:
                continue

            stateFound = True
            m = re.findall(r'tick(\d+)_(\w+)', setting)
            if m:
                tick_idx, tick_type = m[0]
                if tick_type == 'pos':
                    ticks_pos[int(tick_idx)] = float(value)
                elif tick_type == 'rgb':
                    ticks_rgb[int(tick_idx)] = colors.rgba_str_to_values(value)
            else:
                key = setting[9:]
                if value == 'Yes':
                    value = True
                elif value == 'No':
                    value = False
                state[key] = value

        if stateFound:
            ticks = [(0, 0)]*len(ticks_pos)
            for idx, val in ticks_pos.items():
                pos = val
                rgb = ticks_rgb[idx]
                ticks[idx] = (pos, rgb)

            state['ticks'] = ticks
            self.gradient.restoreState(state)

    def regionChanged(self):
        super().regionChanged()
        if self.childLutItem is None:
            return
        
        imageItem = self.imageItem()
        try:
            mn, mx = imageItem.quickMinMax(targetSize=65536)
            # mn and mx can still be NaN if the data is all-NaN
            if mn == mx or imageItem._xp.isnan(mn) or imageItem._xp.isnan(mx):
                mn = 0
                mx = 255
        except AttributeError as err:
            mn, mx = self.getLevels() 
        
        self.childLutItem.setLevels(min=mn, max=mx)
    
    
class labelledQScrollbar(ScrollBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._label = None

    def setLabel(self, label):
        self._label = label

    def updateLabel(self):
        if self._label is not None:
            position = self.sliderPosition()
            s = self._label.text()
            s = re.sub(r'(\d+)/(\d+)', fr'{position+1:02}/\2', s)
            self._label.setText(s)

    def setSliderPosition(self, position):
        QScrollBar.setSliderPosition(self, position)
        self.updateLabel()

    def setValue(self, value):
        QScrollBar.setValue(self, value)
        self.updateLabel()

class navigateQScrollBar(ScrollBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._disableCustomPressEvent = False

    def disableCustomPressEvent(self):
        self._disableCustomPressEvent = True

    def enableCustomPressEvent(self):
        self._disableCustomPressEvent = False

    def setAbsoluteMaximum(self, absoluteMaximum):
        self._absoluteMaximum = absoluteMaximum

    def absoluteMaximum(self):
        return self._absoluteMaximum

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if self.maximum() == self._absoluteMaximum:
            return

        if self._disableCustomPressEvent:
            return

        if self.sliderPosition() == self.maximum():
            # Clicked right arrow of scrollbar with the slider at maximum --> +1
            # self.setMaximum(self.maximum()+1)
            self.triggerAction(QAbstractSlider.SliderAction.SliderSingleStepAdd)

class linkedQScrollbar(ScrollBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._linkedScrollBar = None

    def linkScrollBar(self, scrollbar):
        self._linkedScrollBar = scrollbar
        scrollbar.setSliderPosition(self.sliderPosition())

    def unlinkScrollBar(self):
        self._linkedScrollBar = None

    def setSliderPosition(self, position):
        QScrollBar.setSliderPosition(self, position)
        if self._linkedScrollBar is not None:
            self._linkedScrollBar.setSliderPosition(position)

    def setMaximum(self, max):
        QScrollBar.setMaximum(self, max)
        if self._linkedScrollBar is not None:
            self._linkedScrollBar.setMaximum(max)

class myColorButton(pg.ColorButton):
    def __init__(self, parent=None, color=(128,128,128), padding=5):
        super().__init__(parent=parent, color=color)
        if isinstance(padding, (int, float)):
            self.padding = (padding, padding, -padding, -padding)  
        else:
            self.padding = padding
        self._c = 225
        self._hoverDeltaC = 30
        self._alpha = 100
        self._bkgrColor = QColor(self._c, self._c, self._c, self._alpha) 
        self._borderColor = QColor(171, 171, 171)      
        self._rectBorderPen = QPen(QBrush(QColor(0,0,0)), 0.3)
   
    def paintEvent(self, event):
        # QPushButton.paintEvent(self, ev)
        p = QStylePainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        p.setBrush(QBrush(self._bkgrColor))
        p.setPen(QPen(self._borderColor))
        p.drawRoundedRect(rect, 5, 5)
        # p.fillRect(self.rect(), self._bkgrColor)
        rect = self.rect().adjusted(*self.padding)
        ## draw white base, then texture for indicating transparency, then actual color
        p.setBrush(pg.mkBrush('w'))
        p.drawRect(rect)
        p.setBrush(QBrush(Qt.BrushStyle.DiagCrossPattern))
        p.drawRect(rect)
        p.setPen(self._rectBorderPen)
        p.setBrush(pg.mkBrush(self._color))
        p.drawRect(rect)
        p.end()
    
    def enterEvent(self, event):
        c = self._c + self._hoverDeltaC
        self._bkgrColor = QColor(c, c, c, self._alpha) 
        self.update()
    
    def leaveEvent(self, event):
        c = self._c
        self._bkgrColor = QColor(c, c, c, self._alpha) 
        self.update()

class highlightableQWidgetAction(QWidgetAction):
    def __init__(self, parent) -> None:
        super().__init__(parent)

class overlayLabelsGradientWidget(pg.GradientWidget):
    def __init__(
            self, imageItem, selectActionGroup, segmEndname, 
            parent=None, orientation='right'
        ):
        pg.GradientWidget.__init__(self, parent=parent, orientation=orientation)

        self.imageItem = imageItem
        self.selectActionGroup = selectActionGroup

        for action in self.menu.actions():
            if action.text() == 'HSV':
                HSV_action = action
            elif action.text() == 'RGB':
                RGB_ation = action
        self.menu.removeAction(HSV_action)
        self.menu.removeAction(RGB_ation)

        # Shuffle colors action
        self.shuffleCmapAction =  QAction(
            'Randomly shuffle colormap   (Shift+S)', self
        )
        self.menu.addAction(self.shuffleCmapAction)

        # Drawing mode
        drawModeMenu = QMenu('Drawing mode', self)
        self.drawModeActionGroup = QActionGroup(self)
        contoursDrawModeAction = QAction('Draw contours', drawModeMenu)
        contoursDrawModeAction.setCheckable(True)
        contoursDrawModeAction.setChecked(True)
        contoursDrawModeAction.segmEndname = segmEndname
        self.drawModeActionGroup.addAction(contoursDrawModeAction)
        drawModeMenu.addAction(contoursDrawModeAction)
        olDrawModeAction = QAction('Overlay labels', drawModeMenu)
        olDrawModeAction.setCheckable(True)
        olDrawModeAction.segmEndname = segmEndname
        self.drawModeActionGroup.addAction(olDrawModeAction)
        drawModeMenu.addAction(olDrawModeAction)
        self.menu.addMenu(drawModeMenu)

        self.labelsAlphaMenu = self.menu.addMenu(
            'Overlay labels alpha...'
        )
        hbox = QHBoxLayout()
        self.labelsAlphaSlider = sliderWithSpinBox(
            title='Alpha', title_loc='in_line', isFloat=True,
            normalize=True
        )
        self.labelsAlphaSlider.setMaximum(100)
        self.labelsAlphaSlider.setSingleStep(0.05)
        self.labelsAlphaSlider.setValue(0.3)
        hbox.addWidget(self.labelsAlphaSlider)
        widget = QWidget()
        widget.setLayout(hbox)
        act = QWidgetAction(self)
        act.setDefaultWidget(widget)
        self.labelsAlphaMenu.addSeparator()
        self.labelsAlphaMenu.addAction(act)

        self.menu.addSeparator()
        self.menu.addSection('Select segm. file to adjust:')
        for action in selectActionGroup.actions():
            self.menu.addAction(action)
        
        self.item.loadPreset('viridis')
        self.updateImageLut(None)
        self.updateImageOpacity(0.3)

        # Connect events
        self.sigGradientChangeFinished.connect(self.updateImageLut)
        self.labelsAlphaSlider.valueChanged.connect(self.updateImageOpacity)
        self.shuffleCmapAction.triggered.connect(self.shuffleCmap)
    
    def shuffleCmap(self):
        lut = self.imageItem.lut
        np.random.shuffle(lut)
        lut[0] = [0,0,0,0]
        self.imageItem.setLookupTable(lut)
        self.imageItem.update()
    
    def updateImageLut(self, gradientItem):
        lut = np.zeros((255, 4), dtype=np.uint8)
        lut[:,-1] = 255
        lut[:,:-1] = self.item.colorMap().getLookupTable(0,1,255)
        np.random.shuffle(lut)
        lut[0] = [0,0,0,0]
        self.imageItem.setLookupTable(lut)
        self.imageItem.setLevels([0, 255])
    
    def updateImageOpacity(self, value):
        self.imageItem.setOpacity(value)

class labelsGradientWidget(pg.GradientWidget):
    sigShowRightImgToggled = Signal(bool)
    sigShowLabelsImgToggled = Signal(bool)
    sigShowNextFrameToggled = Signal(bool)

    def __init__( self, *args, parent=None, orientation='right', **kargs):
        pg.GradientEditorItem = BaseGradientEditorItemLabels
        
        pg.GradientWidget.__init__(
            self, *args, parent=parent, orientation=orientation, **kargs
        )

        self._parent = parent
        self.name = 'labels'

        for action in self.menu.actions():
            if action.text() == 'HSV':
                HSV_action = action
            elif action.text() == 'RGB':
                RGB_ation = action
        self.menu.removeAction(HSV_action)
        self.menu.removeAction(RGB_ation)

        # Add custom colormap action
        self.customCmapsMenu = self.menu.addMenu('Custom colormaps')
        self.customCmapsMenu.aboutToShow.connect(self.onShowCustomCmapsMenu)
        self.customCmapsMenu.triggered.connect(self.customCmapsMenuTriggered)
        
        self.saveColormapAction = QAction(
            'Save current colormap...', self
        )
        self.menu.addAction(self.saveColormapAction)
        self.saveColormapAction.triggered.connect(
            self.saveColormap
        )

        self.addCustomGradients()

        # Background color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Background color: '))
        self.colorButton = myColorButton(color=(25,25,25))
        hbox.addStretch(1)
        hbox.addWidget(self.colorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.colorButton.click)
        self.menu.addAction(act)

        # Font size menu action
        self.fontSizeMenu =  QMenu('Text font size', self)
        self.menu.addMenu(self.fontSizeMenu)   

        # IDs color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Text color: '))
        self.textColorButton = myColorButton(color=(25,25,25))
        hbox.addStretch(1)
        hbox.addWidget(self.textColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = highlightableQWidgetAction(self)
        act.setDefaultWidget(widget)
        act.triggered.connect(self.textColorButton.click)
        self.menu.addAction(act)   
        self.menu.addSeparator()  

        # Shuffle colors action
        self.shuffleCmapAction =  QAction(
            'Randomly shuffle colormap   (Shift+S)', self
        )
        self.menu.addAction(self.shuffleCmapAction)

        self.greedyShuffleCmapAction = QAction(
            'Greedily shuffle colormap  (Alt+Shift+S)', self
        )
        self.menu.addAction(self.greedyShuffleCmapAction)

        # Invert bw action
        self.invertBwAction = QAction('Invert black/white', self)
        self.invertBwAction.setCheckable(True)
        self.menu.addAction(self.invertBwAction)

        # Show labels action
        self.showLabelsImgAction = QAction('Show segmentation image', self)
        self.showLabelsImgAction.setCheckable(True)
        self.menu.addAction(self.showLabelsImgAction)

        # Show right image action
        self.showRightImgAction = QAction('Show duplicated left image', self)
        self.showRightImgAction.setCheckable(True)
        self.menu.addAction(self.showRightImgAction)
        
        # Show next frame action
        self.showNextFrameAction = QAction('Show next frame', self)
        self.showNextFrameAction.setCheckable(True)
        self.menu.addAction(self.showNextFrameAction)

        # Default settings
        self.defaultSettingsAction = QAction('Restore default settings...', self)
        self.menu.addAction(self.defaultSettingsAction)

        self.menu.addSeparator()

        self.showRightImgAction.toggled.connect(self.showRightImageToggled)
        self.showLabelsImgAction.toggled.connect(self.showLabelsImageToggled)
        self.showNextFrameAction.toggled.connect(self.showNextFrameToggled)
    
    def onShowCustomCmapsMenu(self):
        self.customCmapsMenu.show()
    
    def customCmapsMenuTriggered(self, action):
        cmap = action.cmap
        self.item.colorMapMenuClicked(cmap)
        self.item.showTicks(True)
    
    def addCustomGradient(self, gradient_name, gradient_ticks, restore=True):
        currentState = self.item.saveState()
        self.originalLength = self.item.length
        self.item.length = 100
        if restore:
            self.item.restoreState(gradient_ticks)
        gradient = self.item.getGradient()
        action = CustomGradientMenuAction(gradient, gradient_name, self.item)
        # action.triggered.connect(self.item.contextMenuClicked)
        action.delButton.clicked.connect(self.removeCustomGradient)
        action.cmap = colors.pg_ticks_to_colormap(gradient_ticks['ticks'])
        # self.item.menu.insertAction(self.saveColormapAction, action)
        self.customCmapsMenu.addAction(action)
        self.item.length = self.originalLength
        self.item.restoreState(currentState)
        GradientsLabels[gradient_name] = gradient_ticks
    
    def removeCustomGradient(self):
        button = self.sender()
        action = button.action
        self.customCmapsMenu.removeAction(action)
        cp = config.ConfigParser()
        cp.read(custom_cmaps_filepath)
        cp.remove_section(f'labels.{action.name}')
        with open(custom_cmaps_filepath, mode='w') as file:
            cp.write(file)
    
    def addCustomGradients(self):
        try:
            CustomGradients = getCustomGradients(name='labels')
            if not CustomGradients:
                return
            for gradient_name, gradient_ticks in CustomGradients.items():
                self.addCustomGradient(gradient_name, gradient_ticks)
        except Exception as e:
            printl(traceback.format_exc())
            pass
    
    def _askNameColormap(self):
        inputWin = apps.QInput(parent=self._parent, title='Colormap name')
        inputWin.askText('Insert a name for the colormap: ', allowEmpty=False)
        if inputWin.cancel:
            return
        cmapName = inputWin.answer
        return cmapName
    
    def saveColormap(self):
        cmapName = self._askNameColormap()
        if cmapName is None:
            return
        
        cp = config.ConfigParser()
        if os.path.exists(custom_cmaps_filepath):
            cp.read(custom_cmaps_filepath)
        
        SECTION = f'{self.name}.{cmapName}'
        cp[SECTION] = {}

        state = self.item.saveState()
        for key, value in state.items():
            if key != 'ticks':
                continue
            for t, tick in enumerate(value):
                pos, rgb = tick
                rgb = ','.join([str(c) for c in rgb])
                val = f'{pos},{rgb}'
                cp[SECTION][f'tick_{t}_pos_rgb'] = val
        
        with open(custom_cmaps_filepath, mode='w') as file:
            cp.write(file)
        
        self.addCustomGradient(cmapName, state, restore=False)
    
    def isRightImageVisible(self):
        return (
            self.showLabelsImgAction.isChecked() 
            or self.showNextFrameAction.isChecked() 
        )

    def showRightImageToggled(self, checked):
        if checked and self.isRightImageVisible():
            # Hide the right labels image before showing right image
            self.showLabelsImgAction.setChecked(False)
            self.showNextFrameAction.setChecked(False)
            self.sigShowLabelsImgToggled.emit(False)
            self.sigShowNextFrameToggled.emit(checked)
        self.sigShowRightImgToggled.emit(checked)
    
    def showLabelsImageToggled(self, checked):
        if checked and self.isRightImageVisible():
            # Hide the right image before showing labels image
            self.showRightImgAction.setChecked(False)
            self.showNextFrameAction.setChecked(False)
            self.sigShowRightImgToggled.emit(False)
            self.sigShowNextFrameToggled.emit(False)
        self.sigShowLabelsImgToggled.emit(checked)
    
    def showNextFrameToggled(self, checked):
        if checked and self.isRightImageVisible():
            # Hide the right image before showing labels image
            self.showRightImgAction.setChecked(False)
            self.showLabelsImgAction.setChecked(False)
            self.sigShowRightImgToggled.emit(False)
            self.sigShowLabelsImgToggled.emit(False)
        self.sigShowNextFrameToggled.emit(checked)

    def saveState(self, df):
        # remove previous state
        df = df[~df.index.str.contains('lab_cmap')].copy()

        state = self.item.saveState()
        for key, value in state.items():
            if key == 'ticks':
                for t, tick in enumerate(value):
                    pos, rgb = tick
                    df.at[f'lab_cmap_tick{t}_rgb', 'value'] = rgb
                    df.at[f'lab_cmap_tick{t}_pos', 'value'] = pos
            else:
                if isinstance(value, bool):
                    value = 'Yes' if value else 'No'
                df.at[f'lab_cmap_{key}', 'value'] = value
        return df

    def restoreState(self, df, loadCmap=True):
        # Insert background color
        if 'labels_bkgrColor' in df.index:
            rgbString = df.at['labels_bkgrColor', 'value']
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.colorButton.setColor((r, g, b))

        if 'labels_text_color' in df.index:
            rgbString = df.at['labels_text_color', 'value']
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.textColorButton.setColor((r, g, b))
        else:
            self.textColorButton.setColor((255, 0, 0))

        checked = df.at['is_bw_inverted', 'value'] == 'Yes'
        self.invertBwAction.setChecked(checked)

        if not loadCmap:
            return

        state = {'mode': 'rgb', 'ticksVisible': True, 'ticks': []}
        ticks_pos = {}
        ticks_rgb = {}
        stateFound = False
        for setting, value in df.itertuples():
            idx = setting.find('lab_cmap_')
            if idx == -1:
                continue

            stateFound = True
            m = re.findall(r'tick(\d+)_(\w+)', setting)
            if m:
                tick_idx, tick_type = m[0]
                if tick_type == 'pos':
                    ticks_pos[int(tick_idx)] = float(value)
                elif tick_type == 'rgb':
                    ticks_rgb[int(tick_idx)] = colors.rgba_str_to_values(value)
            else:
                key = setting[9:]
                if value == 'Yes':
                    value = True
                elif value == 'No':
                    value = False
                state[key] = value

        if stateFound:
            ticks = [(0, 0)]*len(ticks_pos)
            for idx, val in ticks_pos.items():
                pos = val
                rgb = ticks_rgb[idx]
                ticks[idx] = (pos, rgb)

            state['ticks'] = ticks
            self.item.restoreState(state)
        else:
            self.item.loadPreset('viridis')

        return stateFound

    def showMenu(self, ev):
        try:
            # Convert QPointF to QPoint
            self.menu.popup(ev.screenPos().toPoint())
        except AttributeError:
            self.menu.popup(ev.screenPos())

class QLogConsole(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        font = QFont()
        font.setPixelSize(13)
        self.setFont(font)

    def write(self, message):
        # Method required by tqdm pbar
        message = message.replace('\r ', '')
        if message:
            self.apppendText(message)
        
    def append(self, text: str) -> None:
        super().append(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
    
    def insertPlainText(self, text: str) -> None:
        super().append(text)
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

class ProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        palette = self.palette()
        palette.setColor(
            QPalette.ColorRole.Highlight, 
            PROGRESSBAR_QCOLOR
        )
        palette.setColor(
            QPalette.ColorRole.HighlightedText, 
            PROGRESSBAR_HIGHLIGHTEDTEXT_QCOLOR
        )
        self.setPalette(palette)

class ProgressBarWithETA(ProgressBar):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent=parent)
        self.ETA_label = QLabel('NDh:NDm:NDs')

    def update(self, step: int):
        self.setValue(self.value()+step)
        t = time.perf_counter()
        if not hasattr(self, 'last_time_update'):
            self.last_time_update = t
            self.mean_value_duration = None
            return
        seconds_per_value = (t - self.last_time_update)/step
        value_left = self.maximum() - self.value()
        if self.mean_value_duration is None:
            self.mean_value_duration = seconds_per_value
        else:
            self.mean_value_duration = (
                self.mean_value_duration*(self.value()-1) + seconds_per_value
            )/self.value()

        seconds_left = self.mean_value_duration*value_left
        ETA = myutils.seconds_to_ETA(seconds_left)
        self.ETA_label.setText(ETA)
        self.last_time_update = t
        return ETA

    def show(self):
        QProgressBar.show(self)
        self.ETA_label.show()

    def hide(self):
        QProgressBar.hide(self)
        self.ETA_label.hide()

class NoneWidget:
    def __init__(self):
        pass
    
    def value(self):
        return None
    
    def setValue(self, value):
        return

class MainPlotItem(pg.PlotItem):
    def __init__(
            self, parent=None, name=None, labels=None, title=None, 
            viewBox=None, axisItems=None, enableMenu=True, 
            showWelcomeText=False, **kargs
        ):
        super().__init__(
            parent, name, labels, title, viewBox, axisItems, enableMenu, 
            **kargs
        )
        # Overwrite zoom out button behaviour to disable autoRange after
        # clicking it.
        # If autorange is enabled, it is called everytime the brush or eraser 
        # scatter plot items touches the border causing flickering
        self.disableAutoRange()
        self.autoBtn.mode = 'manual'
        if showWelcomeText:
            self.infoTextItem = pg.TextItem()
            self.addItem(self.infoTextItem)
            html_filepath = os.path.join(html_path, 'gui_welcome.html')
            with open(html_filepath) as html_file:
                htmlText = html_file.read()
            self.infoTextItem.setHtml(htmlText)
            self.infoTextItem.setPos(0,0)
    
    def clear(self):
        super().clear()
        try:
            self.removeItem(self.infoTextItem)
        except Exception as e:
            pass
        
    def autoBtnClicked(self):
        self.vb.autoRange()
        self.autoBtn.hide()

class sliderWithSpinBox(QWidget):
    sigValueChange = Signal(object)
    valueChanged = Signal(object)
    editingFinished = Signal()

    def __init__(self, *args, **kwargs):      
        super().__init__(*args)

        layout = QGridLayout()

        title = kwargs.get('title')
        row = 0
        col = 0
        if title is not None:
            titleLabel = QLabel(self)
            titleLabel.setText(title)
            loc = kwargs.get('title_loc', 'top')
            if loc == 'top':
                layout.addWidget(titleLabel, 0, col, alignment=Qt.AlignLeft)
            elif loc=='in_line':
                row = -1
                col = 1
                layout.addWidget(titleLabel, 0, 0, alignment=Qt.AlignLeft)
                layout.setColumnStretch(0, 0)

        self._normalize = False
        normalize = kwargs.get('normalize')
        if normalize is not None and normalize:
            self._normalize = True
            self._isFloat = True

        self._isFloat = False
        isFloat = kwargs.get('isFloat')
        if isFloat is not None and isFloat:
            self._isFloat = True

        self.slider = QSlider(Qt.Horizontal, self)
        layout.addWidget(self.slider, row+1, col)

        if self._normalize or self._isFloat:
            self.spinBox = DoubleSpinBox(self)
        else:
            self.spinBox = SpinBox(self)
        self.spinBox.setAlignment(Qt.AlignCenter)
        self.spinBox.setMaximum(2**31-1)
        layout.addWidget(self.spinBox, row+1, col+1)
        if title is not None:
            layout.setRowStretch(0, 1)
        layout.setRowStretch(row+1, 1)
        layout.setColumnStretch(col, 6)
        layout.setColumnStretch(col+1, 1)

        self._layout = layout
        self.lastCol = col+1
        self.sliderCol = row+1

        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.slider.sliderReleased.connect(self.onEditingFinished)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.spinBox.editingFinished.connect(self.onEditingFinished)
        
        layout.setContentsMargins(5, 0, 5, 0)
        
        self.setLayout(layout)

    def onEditingFinished(self):
        self.editingFinished.emit()

    def maximum(self):
        return self.slider.maximum()

    def minimum(self):
        return self.slider.minimum()

    def setValue(self, value, emitSignal=False):
        valueInt = value
        if self._normalize:
            valueInt = int(value*self.slider.maximum())
        elif self._isFloat:
            valueInt = int(value)

        self.spinBox.valueChanged.disconnect()
        self.spinBox.setValue(value)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)

        self.slider.valueChanged.disconnect()
        if valueInt > self.slider.maximum():
            self.slider.setMaximum(valueInt)
        self.slider.setValue(valueInt)
        self.slider.valueChanged.connect(self.sliderValueChanged)

        if emitSignal:
            self.sigValueChange.emit(self.value())
            self.valueChanged.emit(self.value())

    def setMaximum(self, max):
        self.slider.setMaximum(max)
        # self.spinBox.setMaximum(max)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)

    def setMinimum(self, min):
        self.slider.setMinimum(min)
        # self.spinBox.setMinimum(min)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)

    def setDecimals(self, decimals):
        self.spinBox.setDecimals(decimals)

    def setTickPosition(self, position):
        self.slider.setTickPosition(position)

    def setTickInterval(self, interval):
        self.slider.setTickInterval(interval)

    def sliderValueChanged(self, val):
        self.spinBox.valueChanged.disconnect()
        if self._normalize:
            valF = val/self.slider.maximum()
            self.spinBox.setValue(valF)
        else:
            self.spinBox.setValue(val)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.sigValueChange.emit(self.value())
        self.valueChanged.emit(self.value())

    def spinboxValueChanged(self, val):
        if self._normalize:
            val = int(val*self.slider.maximum())
        elif self._isFloat:
            val = int(val)

        self.slider.valueChanged.disconnect()
        self.slider.setValue(val)
        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.sigValueChange.emit(self.value())
        self.valueChanged.emit(self.value())

    def value(self):
        return self.spinBox.value()
    
    def setDisabled(self, disabled) -> None:
        self.slider.setDisabled(disabled)
        self.spinBox.setDisabled(disabled)

class BaseImageItem(pg.ImageItem):
    def __init__(
            self, image=None, **kargs
        ):
        super().__init__(image, **kargs)
        self.minMaxValuesMapper = None
        self.pos_i = 0
        self.z = 0
        self.frame_i = 0
    
    def preComputedMinMaxValues(self, data: dict):
        self.minMaxValuesMapper = {}
        for pos_i, posData in enumerate(data):
            for frame_i, img_data in enumerate(posData.img_data):
                if img_data.ndim == 2:
                    img_data = (img_data,)
                
                for z, img in enumerate(img_data):
                    self.minMaxValuesMapper[(pos_i, frame_i, z)] = (
                        np.nanmin(img), np.nanmax(img)
                    )
    
    def setCurrentPosIndex(self, pos_i: int):
        self.pos_i = pos_i
    
    def setCurrentFrameIndex(self, frame_i: int):
        self.frame_i = frame_i
    
    def setCurrentZsliceIndex(self, z: int):
        self.z = z
    
    def quickMinMax(self, targetSize=1e6):
        if self.minMaxValuesMapper is None:
            return super().quickMinMax(targetSize=targetSize)
        
        try:
            key = (self.pos_i, self.frame_i, self.z)
            return self.minMaxValuesMapper[key]
        except Exception as err:
            return super().quickMinMax(targetSize=targetSize)
        

class BaseLabelsImageItem(pg.ImageItem):
    def __init__(
            self, image=None, **kargs
        ):
        super().__init__(image, **kargs)
    
    def setImage(self, image=None, **kwargs):
        if image is None:
            return
        autoLevels = kwargs.get('autoLevels')
        if autoLevels is None:
            kwargs['autoLevels'] = False
        super().setImage(image, **kwargs)

class OverlayImageItem(pg.ImageItem):
    def __init__(
            self, image=None, **kargs
        ):
        super().__init__(image, **kargs)
        self.autoLevelsEnabled = None
    
    def setEnableAutoLevels(self, enabled: bool):
        self.autoLevelsEnabled = enabled
    
    def setImage(
            self, image=None, autoLevels=None, **kargs
        ):
        if autoLevels is None:
            autoLevels = self.autoLevelsEnabled
        
        super().setImage(image, autoLevels=autoLevels, **kargs)

class ParentImageItem(BaseImageItem):
    def __init__(
            self, image=None, linkedImageItem=None, activatingActions=None,
            debug=False, **kargs
        ):
        super().__init__(image, **kargs)
        self.linkedImageItem = linkedImageItem
        self.activatingActions = activatingActions
        self.debug = debug
        self._forceDoNotUpdateLinked = False
        self.autoLevelsEnabled = None
    
    def clear(self):
        if self.linkedImageItem is not None:
            self.linkedImageItem.clear()
        return super().clear()
    
    def isLinkedImageItemActive(self):
        if self._forceDoNotUpdateLinked:
            return False
        
        if self.linkedImageItem is None:
            return False
        
        if self.activatingActions is None:
            return False
        for action in self.activatingActions:
            if action.isChecked():
                return True
        return False
    
    # def setLevels(self, levels, **kargs):
    #     if self.linkedImageItem is not None:
    #         self.linkedImageItem.setLevels(levels)
    #     return super().setLevels(levels, **kargs)
    
    def setEnableAutoLevels(self, enabled: bool):
        self.autoLevelsEnabled = enabled
    
    def setImage(
            self, image=None, autoLevels=None, next_frame_image=None, 
            scrollbar_value=None, **kargs
        ):
        if autoLevels is None:
            autoLevels = self.autoLevelsEnabled
        
        super().setImage(image, autoLevels=autoLevels, **kargs)
        
        if not self.isLinkedImageItemActive():
            return
        
        if next_frame_image is not None:
            self.linkedImageItem.setImage(
                next_frame_image, 
                scrollbar_value=scrollbar_value, 
                autoLevels=autoLevels
            )
        elif image is not None:
            self.linkedImageItem.setImage(image)
    
    def updateImage(self, *args, **kargs):
        if self.isLinkedImageItemActive():
            self.linkedImageItem.image = self.image
            self.linkedImageItem.updateImage(*args, **kargs)
        return super().updateImage(*args, **kargs)
    
    def setOpacity(self, value):
        super().setOpacity(value)
        if self.linkedImageItem is not None:
            self.linkedImageItem.setOpacity(value)
    
    def setLookupTable(self, lut):
        super().setLookupTable(lut)
        # if self.linkedImageItem is not None:
        #     self.linkedImageItem.setLookupTable(lut)

class ChildImageItem(BaseImageItem):
    def __init__(self, *args, linkedScrollbar=None, **kwargs):
        BaseImageItem.__init__(self, *args, **kwargs)
        self.linkedScrollbar = linkedScrollbar
    
    def setImage(self, img=None, z=None, scrollbar_value=None, **kargs):
        autoLevels = kargs.get('autoLevels')
        if autoLevels is None:
            kargs['autoLevels'] = False

        if img is None:
            BaseImageItem.setImage(self, img, **kargs)
            return

        if img.ndim == 3 and img.shape[-1] > 4 and z is not None:
            BaseImageItem.setImage(self, img[z], **kargs)
        else:
            BaseImageItem.setImage(self, img, **kargs)
        
        if self.linkedScrollbar is None:
            return
        
        if not self.linkedScrollbar.isEnabled():
            return
        
        if scrollbar_value is None:
            return
        
        self.linkedScrollbar.setValueNoSignal(scrollbar_value)

class labImageItem(pg.ImageItem):
    def __init__(self, *args, **kwargs):
        pg.ImageItem.__init__(self, *args, **kwargs)

    def setImage(self, img=None, z=None, **kargs):
        autoLevels = kargs.get('autoLevels')
        if autoLevels is None:
            kargs['autoLevels'] = False

        if img is None:
            pg.ImageItem.setImage(self, img, **kargs)
            return

        if img.ndim == 3 and img.shape[-1] > 4 and z is not None:
            pg.ImageItem.setImage(self, img[z], **kargs)
        else:
            pg.ImageItem.setImage(self, img, **kargs)
        
            

class PostProcessSegmSlider(sliderWithSpinBox):
    def __init__(self, *args, label=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.label = label
        self.checkbox = QCheckBox('Disable')
        self._layout.addWidget(self.checkbox, self.sliderCol, self.lastCol+1)
        self.checkbox.toggled.connect(self.onCheckBoxToggled)
        self.valueChanged.connect(self.checkExpandRange)
    
    def onCheckBoxToggled(self, checked: bool) -> None:
        super().setDisabled(checked)
        if self.label is not None:
            self.label.setDisabled(checked)
        self.onValueChanged(None)
        self.onEditingFinished()
    
    def onValueChanged(self, value):
        self.valueChanged.emit(value)
    
    def checkExpandRange(self, value):
        if value == self.maximum():
            range = int(self.maximum() - self.minimum())
            half_range = int(range/2)
            newMinimum = self.minimum() + half_range
            newMaximum = self.maximum() + half_range
            self.setMaximum(newMaximum)
            self.setMinimum(newMinimum)
        elif value == self.minimum():
            range = int(self.maximum() - self.minimum())
            half_range = int(range/2)
            newMinimum = self.minimum() - half_range
            newMaximum = self.maximum() - half_range
            self.setMaximum(newMaximum)
            self.setMinimum(newMinimum)
    
    def onEditingFinished(self):
        self.editingFinished.emit()
    
    def value(self):
        if self.checkbox.isChecked():
            return None
        else:
            return super().value()

class GhostContourItem(pg.PlotDataItem):
    def __init__(self, penColor=(245, 184, 0, 100), textColor=(245, 184, 0)):
        super().__init__()
        # Yellow pen
        self.setPen(pg.mkPen(width=2, color=penColor))
        self.label = myLabelItem()
        self.label.setAttr('bold', True)
        self.label.setAttr('color', textColor)
    
    def addToPlotItem(self, PlotItem: MainPlotItem):
        self._plotItem = PlotItem
        PlotItem.addItem(self)
        PlotItem.addItem(self.label)
    
    def removeFromPlotItem(self):
        if not hasattr(self, '_plotItem'):
            return
        self._plotItem.removeItem(self.label)
        self._plotItem.removeItem(self)
    
    def setData(
            self, xx=None, yy=None, fontSize=11, ID=0, 
            y_cursor=None, x_cursor=None
        ):
        if xx is None:
            xx = []
        if yy is None:
            yy = []
        super().setData(xx, yy)
        if not hasattr(self, 'label'):
            return

        if ID == 0:
            self.label.setText('')
        else:
            self.label.setText(f'{ID}', size=fontSize)
            w, h = self.label.itemRect().width(), self.label.itemRect().height()
            self.label.setPos(x_cursor, y_cursor-h)
    
    def clear(self):
        self.setData([], [])

class GhostMaskItem(pg.ImageItem):
    def __init__(self):
        super().__init__()
        self.label = myLabelItem()
        self.label.setAttr('bold', True)
        self.label.setAttr('color', (245, 184, 0))
    
    def initImage(self, imgShape):
        image = np.zeros(imgShape, dtype=np.uint32)
        self.setImage(image)
    
    def initLookupTable(self, rgbaColor):
        lut = np.zeros((2, 4), dtype=np.uint8)
        lut[1,-1] = 255
        lut[1,:-1] = rgbaColor
        self.setLookupTable(lut)
    
    def addToPlotItem(self, PlotItem: MainPlotItem):
        self._plotItem = PlotItem
        PlotItem.addItem(self)
        PlotItem.addItem(self.label)
    
    def removeFromPlotItem(self):
        self._plotItem.removeItem(self.label)
        self._plotItem.removeItem(self)
    
    def updateGhostImage(self, ID=0, y_cursor=None, x_cursor=None, fontSize=None):
        self.setImage(self.image)

        if ID == 0:
            self.label.setText('')
            return
        
        self.label.setText(f'{ID}', size=fontSize)
        w, h = self.label.itemRect().width(), self.label.itemRect().height()
        self.label.item.setPos(x_cursor, y_cursor-h)
    
    def clear(self):
        if hasattr(self, 'label'):
            self.label.setText('')
        if self.image is None:
            return
        self.image[:] = 0
        self.setImage(self.image)

class PostProcessSegmSpinbox(QWidget):
    valueChanged = Signal(int)
    editingFinished = Signal()
    sigCheckboxToggled = Signal()

    def __init__(self, *args, isFloat=False, label=None, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QHBoxLayout()

        if isFloat:
            self.spinBox = DoubleSpinBox()
        else:
            self.spinBox = SpinBox()

        self.spinBox.valueChanged.connect(self.onValueChanged)
        self.spinBox.editingFinished.connect(self.onEditingFinished)

        layout.addWidget(self.spinBox)
        self.checkbox = QCheckBox('Disable')
        layout.addWidget(self.checkbox)
        layout.setStretch(0,1)
        layout.setStretch(1,0)

        self.label = label

        self.checkbox.toggled.connect(self.onCheckBoxToggled)
        
        layout.setContentsMargins(5, 0, 5, 0)
    
        self.setLayout(layout)
    
    def onCheckBoxToggled(self, checked: bool) -> None:
        self.spinBox.setDisabled(checked)
        if self.label is not None:
            self.label.setDisabled(checked)
        self.onValueChanged(None)
        self.onEditingFinished()
    
    def onValueChanged(self, value):
        self.valueChanged.emit(value)
    
    def onEditingFinished(self):
        self.editingFinished.emit()

    def maximum(self):
        return self.spinBox.maximum()
    
    def setValue(self, value):
        self.spinBox.setValue(value)
    
    def sizeHint(self):
        return self.spinBox.sizeHint()
    
    def setMaximum(self, max):
        self.spinBox.setMaximum(max)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)

    def setMinimum(self, min):
        self.spinBox.setMinimum(min)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)
    
    def setDecimals(self, decimals):
        self.spinBox.setDecimals(decimals)
    
    def value(self):
        if self.checkbox.isChecked():
            return None
        else:
            return self.spinBox.value()

class CopiableCommandWidget(QGroupBox):
    def __init__(self, command='', parent=None, font_size='13px'):
        super().__init__(parent)
        
        layout = QHBoxLayout()
        
        label = QLabel(self)
        self.label = label
        self._font_size = font_size
        self.setCommand(command, font_size=font_size)
        label.setTextInteractionFlags(
            Qt.TextBrowserInteraction | Qt.TextSelectableByKeyboard
        )
        layout.addWidget(label)
        layout.addWidget(QVLine(shadow='Plain', color='#4d4d4d'))
        copyButton = copyPushButton('Copy', flat=True, hoverable=True)
        copyButton.clicked.connect(self.copyToClipboard)
        layout.addWidget(copyButton)
        layout.addStretch(1)
        
        self.setLayout(layout)        
    
    def copyToClipboard(self):
        cb = QApplication.clipboard()
        cb.clear(mode=cb.Clipboard)
        cb.setText(self._command, mode=cb.Clipboard)
        print('Command copied!')
    
    def setCommand(self, command, font_size=None):
        if font_size is None:
            font_size = self._font_size
        
        self._command = command
        txt = html_utils.paragraph(
            f'<code>{command}</code>', font_size=font_size
        )
        self.label.setText(txt)
    
    def command(self):
        return self._command
        

def PostProcessSegmWidget(
        minimum, maximum, value, useSliders, isFloat=False, normalize=False,
        label=None
    ):
    if useSliders:
        if normalize:
            maximum = int(maximum*100)
        widget = PostProcessSegmSlider(
            normalize=normalize, isFloat=isFloat, label=label
        )
    else:
        widget = PostProcessSegmSpinbox(label=label, isFloat=isFloat)
    widget.setMinimum(minimum)
    widget.setMaximum(maximum)
    widget.setValue(value)
    return widget

# class Spinner(QLabel):
#     def __init__(self, size=150, parent=None):
#         super().__init__(parent)
#         # layout = QHBoxLayout()
        
#         # self._label = QLabel()
#         self.setAlignment(Qt.AlignCenter)
#         # self._label.setText('Ciao')
#         self._pixmap = QPixmap(':spinner.svg')
        
#         self._pixmapSize = size + size%2
#         self._halfPixmapSize = int(self._pixmapSize/2)
#         printl(self._pixmapSize, self._halfPixmapSize)
#         self.setPixmap(self._pixmap.scaled(self._pixmapSize, self._pixmapSize))

#         # self.setFixedSize(160, 160)
#         self._angle = 0
        
#         blurEffect = QGraphicsBlurEffect()
#         blurEffect.setBlurRadius(1.4)
#         self.setGraphicsEffect(blurEffect)
        
#         # layout.addWidget(self._label)
#         # self.setLayout(layout)

#         self.animation = QPropertyAnimation(self, b"angle", self)
#         self.animation.setStartValue(0)
#         self.animation.setEndValue(360)
#         self.animation.setLoopCount(-1)
#         self.animation.setDuration(1700)
#         self.animation.start()

#     @Property(int)
#     def angle(self):
#         return self._angle

#     @angle.setter
#     def angle(self, value):
#         self._angle = value
#         self.update()

#     def paintEvent(self, ev=None):
#         width, height = self.size().width(), self.size().height()
#         radius_x = int(width/2)
#         radius_y = int(height/2)
#         x = radius_x-self._halfPixmapSize
#         y = radius_y-self._halfPixmapSize
#         painter = QPainter(self)
#         painter.setRenderHint(QPainter.Antialiasing)
#         painter.translate(radius_x, radius_y)
#         painter.rotate(self._angle)
#         painter.translate(-radius_x, -radius_y)
#         painter.drawPixmap(x, y, self._pixmap.scaled(self._pixmapSize, self._pixmapSize))
#         painter.end()

class LoadingCircleAnimation(QLabel):
    def __init__(self, size=32, motionBlur=False, parent=None):
        super().__init__(parent)
        # layout = QHBoxLayout()
        
        # self._label = QLabel()
        self.setAlignment(Qt.AlignCenter)
        self._size = size + size%2
        self._radius = int(self._size/2)
        self.setFixedSize(self._size, self._size)
        self._dotDiameter = int(self._size*0.15)
        self._dotDiameter = self._dotDiameter + self._dotDiameter%2
        self._dotRadius = int(self._dotDiameter/2)
        
        self._rgb = _palettes.getPainterColor()[:3]
        self._index = 0
        
        self.setBrushesAndAngles()
        
        if motionBlur:
            blurEffect = QGraphicsBlurEffect()
            blurRadius = self._size*0.02
            if blurRadius < 1:
                blurRadius = 1
            blurEffect.setBlurRadius(blurRadius)
            self.setGraphicsEffect(blurEffect)

        self.animation = QPropertyAnimation(self, b"index", self)
        self.animation.setStartValue(0)
        self.animation.setEndValue(11)
        self.animation.setLoopCount(-1)
        self.animation.setDuration(1200)
        self.animation.start()
        
        self.update()
    
    def setVisible(self, visible):
        if visible:
            self.animation.start()
        else:
            self.animation.stop()
        super().setVisible(visible)
    
    def setBrushesAndAngles(self):
        self._brushes = []
        self._pens = []
        alphas = np.round(np.linspace(0, 255, 12)).astype(int)
        self._angles = np.arange(0, 360, 30)
        for alpha in alphas:
            color = QColor(*self._rgb, alpha)
            self._brushes.append(pg.mkBrush(color))
            self._pens.append(pg.mkPen(color))

    @Property(int)
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        self._index = value
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.translate(self._radius, self._radius)
        for i in range(12):
            idx = i - self._index
            angle = self._angles[i]
            painter.setBrush(self._brushes[idx])
            painter.setPen(self._pens[idx])
            x = (self._radius-self._dotRadius)*math.cos(angle*math.pi/180)
            y = (self._radius-self._dotRadius)*math.sin(angle*math.pi/180)
            painter.drawEllipse(QPointF(x, y), self._dotRadius, self._dotRadius)
        
        painter.end()

class QBaseWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

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
    
    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            event.ignore()
            return
            
        super().keyPressEvent(event)

class ScrollBarWithNumericControl(QWidget):
    sigValueChanged = Signal(int)
    sigMaxProjToggled = Signal(bool, object)
    
    def __init__(
            self, orientation=Qt.Horizontal, add_max_proj_button=False, 
            parent=None, labelText=''
        ) -> None:
        super().__init__(parent)
        
        self._slot = None
    
        layout = QHBoxLayout()
        self.scrollbar = QScrollBar(orientation, self)
        self.spinbox = QSpinBox(self)
        self.maxLabel = QLabel(self)
        idx = 0
        if labelText:
            layout.addWidget(QLabel(labelText))
            layout.setStretch(idx, 0)
            idx += 1

        layout.addWidget(self.spinbox)
        layout.setStretch(idx,0)
        idx += 1
        
        layout.addWidget(self.maxLabel)
        layout.setStretch(idx,0)
        idx += 1
        
        layout.addWidget(self.scrollbar)
        layout.setStretch(idx,1)
        idx += 1
        
        if add_max_proj_button:
            self.maxProjCheckbox = QCheckBox('MAX')
            self.scrollbar.maxProjCheckbox = self.maxProjCheckbox
            layout.addWidget(self.maxProjCheckbox)
            layout.setStretch(idx,0)
        
        layout.setContentsMargins(5, 0, 5, 0)

        self.setLayout(layout)

        self.spinbox.valueChanged.connect(self.spinboxValueChanged)
        self.scrollbar.valueChanged.connect(self.scrollbarValueChanged)

        if add_max_proj_button:
            self.maxProjCheckbox.toggled.connect(self.maxProjToggled)
    
    def connectValueChanged(self, slot):
        self.sigValueChanged.connect(slot)
        self._slot = slot
    
    def setValueNoSignal(self, value):
        if self._slot is None:
            return
        self.sigValueChanged.disconnect()
        self.setValue(value)
        self.sigValueChanged.connect(self._slot)
    
    def maxProjToggled(self, checked):
        self.scrollbar.setDisabled(checked)
        self.sigMaxProjToggled.emit(checked, self)
    
    def showEvent(self, event) -> None:
        super().showEvent(event)

        self.scrollbar.setMinimumHeight(self.spinbox.height())
    
    def setMaximum(self, maximum):
        self.maxLabel.setText(f'/{maximum}')
        self.scrollbar.setMaximum(maximum)
        self.spinbox.setMaximum(maximum)
    
    def setMinimum(self, minumum):
        self.scrollbar.setMinimum(minumum)
        self.spinbox.setMinimum(minumum)
    
    def spinboxValueChanged(self, value):
        self.scrollbar.setValue(value)
    
    def scrollbarValueChanged(self, value):
        self.spinbox.setValue(value)
        self.sigValueChanged.emit(value)
    
    def setValue(self, value):
        self.scrollbar.setValue(value)
    
    def value(self):
        return self.scrollbar.value()
    
    def maximum(self):
        return self.scrollbar.maximum()

class ImShowPlotItem(pg.PlotItem):
    def __init__(
            self, parent=None, name=None, labels=None, title=None, 
            viewBox=None, axisItems=None, enableMenu=True, **kargs
        ):
        super().__init__(
            parent, name, labels, title, viewBox, axisItems, enableMenu, 
            **kargs
        )
        # Overwrite zoom out button behaviour to disable autoRange after
        # clicking it.
        # If autorange is enabled, it is called everytime the brush or eraser 
        # scatter plot items touches the border causing flickering
        self.autoBtn.mode = 'manual'
        self.invertY(True)
        self.setAspectLocked(True)
        self.addImageItem(kargs.get('imageItem'))
    
    def addImageItem(self, imageItem):
        self.imageItem = imageItem
        if imageItem is None:
            return
        
        self.setupContextMenu()
        self.addItem(imageItem)
    
    def setupContextMenu(self):
        shuffleCmapAction = QAction('Shuffle colormap', self.vb.menu)
        shuffleCmapAction.triggered.connect(self.shuffleColormap)
        self.vb.menu.addAction(shuffleCmapAction)
        
        self.resetCmapAction = QAction('Reset colormap', self.vb.menu)
        self.resetCmapAction.triggered.connect(self.resetColormap)
        self.vb.menu.addAction(self.resetCmapAction)
        self.resetCmapAction.setDisabled(True)
    
    def shuffleColormap(self):
        N = self.imageItem._numLevels
        colors = self.imageItem.lut/255
        cmap = LinearSegmentedColormap.from_list('shuffled', colors, N=N)
        lut = plot.matplotlib_cmap_to_lut(cmap, n_colors=N)
        if not self.resetCmapAction.isEnabled():
            self._defaultLut = lut.copy()
        bkgrColor = lut[0].copy()
        np.random.shuffle(lut)
        lut[0] = bkgrColor
        self.imageItem.setLookupTable(lut)
        self.imageItem.update()
        self.resetCmapAction.setDisabled(False)
    
    def resetColormap(self):
        self.imageItem.setLookupTable(self._defaultLut)
    
    def autoBtnClicked(self):
        self.autoRange()
    
    def autoRange(self):
        self.vb.autoRange()
        self.autoBtn.hide()

class _ImShowImageItem(pg.ImageItem):
    sigDataHover = Signal(str)

    def __init__(self, idx) -> None:
        super().__init__()
        self._idx = idx
        self._cursors = []
    
    def _getHoverImageValue(self, xdata, ydata):
        try:
            value = self.image[ydata, xdata]
            return value
        except Exception as err:
            return
    
    def setOtherImagesCursors(self, cursors):
        self._cursors = cursors
    
    def clearCursors(self):
        for p, cursor in enumerate(self._cursors):
            if p == self._idx:
                continue
            
            cursor.setData([], [])
    
    def setImage(self, *args, **kwargs):
        super().setImage(*args, **kwargs)
        if not args:
            return
        image = args[0]
        self._imageMax = image.max()
        self._imageMin = image.min()
        self._numLevels = self._imageMax - self._imageMin
    
    def hoverEvent(self, event):
        if event.isExit():
            self.clearCursors()
            self.sigDataHover.emit('')
            return
        
        x, y = event.pos()
        xdata, ydata = int(x), int(y)
        value = self._getHoverImageValue(xdata, ydata)
        if value is None:
            self.clearCursors()
            self.sigDataHover.emit('')
            return
        
        try:
            self.sigDataHover.emit(
                f'x={xdata}, y={ydata}, {value = :.4f}'
            )
        except Exception as e:
            self.sigDataHover.emit(
                f'x={xdata}, y={ydata}, {[val for val in value]}'
            )
        
        for p, cursor in enumerate(self._cursors):
            if p == self._idx:
                continue
            
            cursor.setData([x], [y])
            

class ImShow(QBaseWindow):
    def __init__(self, parent=None, link_scrollbars=True, infer_rgb=True):
        super().__init__(parent=parent)
        self._linkedScrollbars = link_scrollbars
        self._infer_rgb = infer_rgb

        self._autoLevels = True

        self.textItems = []
        self.group_to_idx_mapper = {'': 0}
    
    def _getGraphicsScrollbar(self, idx, image, imageItem, maximum):
        proxy = QGraphicsProxyWidget(imageItem)
        scrollbar = ScrollBarWithNumericControl(
            orientation=Qt.Horizontal, add_max_proj_button=True
        )
        scrollbar.sigValueChanged.connect(self.OnScrollbarValueChanged)
        scrollbar.sigMaxProjToggled.connect(self.onMaxProjToggled)
        scrollbar.idx = idx
        scrollbar.image = image
        scrollbar.imageItem = imageItem
        scrollbar.setMaximum(maximum)
        proxy.setWidget(scrollbar)
        proxy.scrollbar = scrollbar
        return proxy
    
    def OnScrollbarValueChanged(self, value):
        scrollbar = self.sender()
        imageItem = scrollbar.imageItem
        img = self._get2Dimg(imageItem, scrollbar.image)
        imageItem.setImage(img, autoLevels=self._autoLevels)
        self.setPointsVisible(imageItem)

        self.updateIDs()


        if not self._linkedScrollbars:
            return
        if len(self.ImageItems) == 1:
            return
        
        self._linkedScrollbars = False
        try:
            idx = scrollbar.idx
            for otherImageItem in self.ImageItems:
                if otherImageItem.gridPos == imageItem.gridPos:
                    continue
                if otherImageItem.image.shape != imageItem.image.shape:
                    continue
                for otherScrollbar in otherImageItem.ScrollBars:
                    if otherScrollbar.idx != idx:
                        continue
                    otherScrollbar.setValue(scrollbar.value())
        except Exception as e:
            pass
        finally:
            self._linkedScrollbars = True
    
    def _get2Dimg(self, imageItem, image):
        for scrollbar in imageItem.ScrollBars:
            if scrollbar.maxProjCheckbox.isChecked():
                image = image.max(axis=0)
            else:
                image = image[scrollbar.value()]
        return image
    
    def onMaxProjToggled(self, checked, scrollbar):
        imageItem = scrollbar.imageItem
        img = self._get2Dimg(imageItem, scrollbar.image)
        imageItem.setImage(img, autoLevels=self._autoLevels)
        self.setPointsVisible(imageItem)
        if not self._linkedScrollbars:
            return
        if len(self.ImageItems) == 1:
            return
        
        self._linkedScrollbars = False
        try:
            idx = scrollbar.idx
            for otherImageItem in self.ImageItems:
                if otherImageItem.gridPos == imageItem.gridPos:
                    continue
                if otherImageItem.image.shape != imageItem.image.shape:
                    continue
                for otherScrollbar in otherImageItem.ScrollBars:
                    if otherScrollbar.idx != idx:
                        continue
                    otherScrollbar.maxProjCheckbox.setChecked(checked)
        except Exception as e:
            pass
        finally:
            self._linkedScrollbars = True

    def setPointsVisible(self, imageItem):
        if not hasattr(imageItem, 'pointsItems'):
            return
        
        first_coord = imageItem.ScrollBars[0].value()
        isMaxProj = imageItem.ScrollBars[0].maxProjCheckbox.isChecked()
        for pointsItems in imageItem.pointsItems.values():
            for p, plotItem in enumerate(pointsItems):
                plotItem.setVisible((isMaxProj) or (p == first_coord))
    
    def setupStatusBar(self):
        self.statusbar = self.statusBar()
        self.wcLabel = QLabel(f"")
        self.statusbar.addPermanentWidget(self.wcLabel)
    
    def setupMainLayout(self):
        self._layout = QHBoxLayout()
        self._container = QWidget()
        self._container.setLayout(self._layout)
        self.setCentralWidget(self._container)
    
    def setupGraphicLayout(
            self, *images, hide_axes=True, max_ncols=4, color_scheme='light'
        ):
        self.graphicLayout = pg.GraphicsLayoutWidget()
        self._colorScheme = color_scheme

        # Set a light background
        if color_scheme == 'light':
            self.graphicLayout.setBackground((235, 235, 235))
        else:
            self.graphicLayout.setBackground((30, 30, 30))

        ncells = max_ncols * ceil(len(images)/max_ncols)

        nrows = ncells // max_ncols
        nrows = nrows if nrows > 0 else 1
        ncols = max_ncols if len(images) > max_ncols else len(images)
        
        # Check if additional rows are needed for the scrollbars
        max_ndim = max([image.ndim for image in images])
        if max_ndim > 4:
            raise TypeError('One or more of the images have more than 4 dimensions.')
        if max_ndim == 4:
            rows_range = range(0, (nrows-1)*3+1, 3)
        elif max_ndim == 3:
            rows_range = range(0, (nrows-1)*2+1, 2)
        else:
            rows_range = range(nrows)
        
        self.PlotItems = []
        self.ImageItems = []
        self.ScrollBars = []
        i = 0
        for row in rows_range:
            for col in range(ncols):
                try:
                    image = images[i]
                except IndexError:
                    break
                plot = ImShowPlotItem()
                if hide_axes:
                    plot.hideAxis('bottom')
                    plot.hideAxis('left')
                self.graphicLayout.addItem(plot, row=row, col=col)
                self.PlotItems.append(plot)

                imageItem = _ImShowImageItem(i)
                plot.addImageItem(imageItem)
                self.ImageItems.append(imageItem)
                imageItem.gridPos = (row, col)
                imageItem.ScrollBars = []
                
                is_rgb = image.shape[-1] == 3 and self._infer_rgb
                is_rgba = image.shape[-1] == 4 and self._infer_rgb
                does_not_require_scrollbars = (
                    image.ndim == 2
                    or (image.ndim == 3 and (is_rgb or is_rgba))
                )
                if does_not_require_scrollbars:
                    i += 1
                    continue
                
                idx_image = 3 if (is_rgb or is_rgba) else 2
                for s in range(image.ndim-idx_image):
                    maximum = image.shape[s]-1
                    scrollbarProxy = self._getGraphicsScrollbar(
                        0, image, imageItem, maximum
                    )
                    self.graphicLayout.addItem(
                        scrollbarProxy, row=row+s+1, col=col
                    )
                    imageItem.ScrollBars.append(scrollbarProxy.scrollbar)

                i += 1
        
        self._layout.addWidget(self.graphicLayout)
    
    def setupTitles(self, *titles):
        color = 'k' if self._colorScheme == 'light' else 'w'
        for plot, title in zip(self.PlotItems, titles):
            plot.setTitle(title, color=color)
    
    def updateStatusBarLabel(self, text):
        self.wcLabel.setText(text)
    
    def autoRange(self):
        for plot in self.PlotItems:
            plot.autoRange()
    
    def showImages(
            self, *images, luts=None, autoLevels=True, 
            autoLevelsOnScroll=False
        ):
        images = [np.squeeze(img) for img in images]
        self.luts = luts
        self._autoLevels = autoLevels
        self._autoLevelsOnScroll = autoLevelsOnScroll
        for image in images:
            if image.ndim > 5 or image.ndim < 2:
                raise TypeError(
                    f'Input image has {image.ndim} dimensions. '
                    'Only 2-D, 3-D, and 4-D images are supported'
                )
        
        for i, (image, imageItem) in enumerate(zip(images, self.ImageItems)):
            if luts is not None:
                imageItem.setLookupTable(luts[i])
                if not autoLevels:
                    imageItem.setLevels([0, len(luts[i])])
            else:
                self._autoLevels = True
            
            is_rgb = image.shape[-1] == 3 and self._infer_rgb
            is_rgba = image.shape[-1] == 4 and self._infer_rgb
            does_not_require_scrollbars = (
                image.ndim == 2
                or (image.ndim == 3 and (is_rgb or is_rgba))
            )

            if does_not_require_scrollbars:
                imageItem.setImage(image, autoLevels=self._autoLevels)
            else:
                if not self._autoLevelsOnScroll:
                    self._autoLevels = False
                    imageItem.setLevels([image.min(), image.max()])
                for scrollbar in imageItem.ScrollBars:
                    scrollbar.setValue(int(scrollbar.maximum()/2))

            imageItem.sigDataHover.connect(self.updateStatusBarLabel)

        # Share axis between images with same X, Y shape
        all_shapes = [image.shape[-2:] for image in images]
        unique_shapes = set(all_shapes)
        shame_shape_plots = []
        for unique_shape in unique_shapes:
            plots = [
                self.PlotItems[i] for i, shape in enumerate(all_shapes) 
                if shape==unique_shape
            ]
            shame_shape_plots.append(plots)
        
        for plots in shame_shape_plots:
            for plot in plots:
                plot.vb.setYLink(plots[0].vb)
                plot.vb.setXLink(plots[0].vb)
    
    def _createPointsScatterItem(self, group, data=None):
        cmap = matplotlib.colormaps['jet_r']
        idx = self.group_to_idx_mapper[group]
        r, g, b = [round(c*255) for c in cmap(idx)][:3]
        item = pg.ScatterPlotItem(
            [], [], symbol='o', pxMode=False, size=3,
            brush=pg.mkBrush(color=(r,g,b,100)),
            pen=pg.mkPen(width=2, color=(r,g,b)),
            hoverable=True, hoverBrush=pg.mkBrush((r,g,b,200)), 
            tip=None, data=data
        ) 
        return item

    def drawPointsFromDf(self, points_df, points_groups=None):
        if isinstance(points_groups, str):
            points_groups = [points_groups]
            
        if points_groups is None:
            grouped = [('', points_df)]
            groups = ['']
        else:
            grouped = points_df.groupby(points_groups)
            groups = grouped.groups.keys()
        
        idxs_space = np.linspace(0, 1, len(groups))
        self.group_to_idx_mapper = dict(zip(groups, idxs_space))

        for group, df in grouped:
            yy = df['y'].values
            xx = df['x'].values
            points_coords = np.column_stack((yy, xx))
            if 'z' in df.columns:
                zz = df['z'].values
                points_coords = np.column_stack((zz, points_coords))
            if len(group) == 1:
                group = group[0]
            self.drawPoints(points_coords, group=group)
    
    def drawPoints(self, points_coords: np.ndarray, group=''):  
        offset = 0.5 if np.issubdtype(points_coords.dtype, np.integer) else 0
        n_dim = points_coords.shape[1]
        if n_dim == 2:
            zz = [0]*len(points_coords)
            self.points_coords = np.column_stack((zz, points_coords))
            for p, plotItem in enumerate(self.PlotItems):
                imageItem = self.ImageItems[p]
                pointsItem = self._createPointsScatterItem(group, data=group)
                pointsItem.z = 0
                plotItem.addItem(pointsItem)
                xx = points_coords[:, 1] + offset
                yy = points_coords[:, 0] + offset  
                pointsItem.setData(xx, yy)
                imageItem.pointsItems = {group: [pointsItem]}
        elif n_dim == 3:
            self.points_coords = points_coords
            for p, plotItem in enumerate(self.PlotItems):
                imageItem = self.ImageItems[p]
                imageItem.pointsItems = defaultdict(list)
                scrollbar = imageItem.ScrollBars[0]
                for first_coord in range(scrollbar.maximum()+1):
                    pointsItem = self._createPointsScatterItem(group, data=group)
                    pointsItem.z = first_coord
                    plotItem.addItem(pointsItem)
                    coords = points_coords[points_coords[:,0] == first_coord]
                    xx = coords[:, 2] + offset
                    yy = coords[:, 1] + offset
                    pointsItem.setData(xx, yy)
                    pointsItem.setVisible(False)
                    imageItem.pointsItems[group].append(pointsItem)
                self.setPointsVisible(imageItem)
    
    def setupDuplicatedCursors(self):
        self.cursors = []
        for p, plotItem in enumerate(self.PlotItems):
            cursor = pg.ScatterPlotItem(
                symbol='+', pxMode=True, pen=pg.mkPen('k', width=1),
                brush=pg.mkBrush('w'), size=16, tip=None
            )
            self.cursors.append(cursor)
            plotItem.addItem(cursor)
        
        for imageItem in self.ImageItems:
            imageItem.setOtherImagesCursors(self.cursors)
    
    def setPointsData(self, points_data):
        points_df = pd.DataFrame({
            'z': self.points_coords[:, 0],
            'y': self.points_coords[:, 1],
            'x': self.points_coords[:, 2]
        })
        if isinstance(points_data, pd.Series):
            points_df[points_data.name] = points_data.values
        elif isinstance(points_data, pd.DataFrame):
            points_df = points_df.join(points_data)
        elif isinstance(points_data, np.ndarray):
            if points_data.ndim == 1:
                points_data = points_data[np.newaxis]
            else:
                points_data = points_data.T
            for i, values in enumerate(points_data):
                points_df[f'col_{i}'] = values

        self.points_df = points_df.set_index(['z', 'y', 'x']).sort_index()
        
        for p, plotItem in enumerate(self.PlotItems):
            imageItem = self.ImageItems[p]
            for pointsItems in imageItem.pointsItems.values():
                for pointsItem in pointsItems:
                    pointsItem.sigClicked.connect(self.pointsClicked)
        
    def pointsClicked(self, item, points, event):
        point = points[0]
        x, y = point.pos()
        coords = (item.z, int(y), int(x))
        point_data = self.points_df.loc[[coords]]
        now = datetime.datetime.now().strftime('%H:%M:%S')
        print('*'*60)
        print(f'Point clicked at {now}. Data:')
        print('-'*60)
        print(point_data)
        print('')
        print('*'*60)

    def annotateObjectIDs(self, annotate_labels_idxs=None, init=False):
        if init:
            self.annotate_labels_idxs = annotate_labels_idxs
            self.textItems = [{} for _ in self.PlotItems] 
        if self.annotate_labels_idxs is None:
            return
        for i, plotItem in enumerate(self.PlotItems):
            if i not in self.annotate_labels_idxs:
                continue
            plotTextItems = self.textItems[i]
            imageItem = self.ImageItems[i]
            lab = imageItem.image
            rp = skimage.measure.regionprops(lab)
            for obj in rp:
                textItem = plotTextItems.get(obj.label)
                yc, xc = obj.centroid[-2:]
                if textItem is None:
                    textItem = pg.TextItem(text='', anchor=(0.5,0.5), color='r')
                    plotItem.addItem(textItem)
                    plotTextItems[obj.label] = textItem
                textItem.setText(str(obj.label))
                textItem.setPos(xc, yc)

    def clearLabels(self):
        for textItems in self.textItems:
            for textItem in textItems.values():
                textItem.setText('') 

    def updateIDs(self):
        self.clearLabels()
        try:
            self.annotateObjectIDs(annotate_labels_idxs=self.annotate_labels_idxs)
        except Exception as err:
            pass

    def show(self, block=False, screenToWindowRatio=None):
        super().show(block=block)
        if screenToWindowRatio is None:
            return
        screenGeometry = self.screen().geometry()
        screenWidth = screenGeometry.width()
        screenHeight = screenGeometry.height()
        finalWidth = int(screenToWindowRatio*screenWidth)
        finalHeight = int(screenToWindowRatio*screenHeight)
        screenTop = screenGeometry.top()
        screenLeft = screenGeometry.left()
        xc, yc = screenLeft + screenWidth/2, screenTop + screenHeight/2
        winLeft = int(xc - finalWidth/2)
        winTop = int(yc - finalHeight/2)
        self.setGeometry(winLeft, winTop, finalWidth, finalHeight)
        
    def run(self, block=False, showMaximised=False, screenToWindowRatio=None):
        if showMaximised:
            self.showMaximized()
        else:
            self.show(screenToWindowRatio=screenToWindowRatio)
        QTimer.singleShot(100, self.autoRange)
        
        if block:
            self.exec_()
    
    def resizeEvent(self, event) -> None:
        self.PlotItems[0].autoRange()       
        return super().resizeEvent(event)

class FeatureSelectorButton(QPushButton):
    def __init__(self, text, parent=None, alignment=''):
        super().__init__(text, parent=parent)
        self._isFeatureSet = False
        self._alignment = alignment
        self.setCursor(Qt.PointingHandCursor)
    
    def setFeatureText(self, text):
        self.setText(text)
        self.setFlat(True)
        self._isFeatureSet = True
        if self._alignment:
            self.setStyleSheet(f'text-align:{self._alignment};')
    
    def enterEvent(self, event) -> None:
        if self._isFeatureSet:
            self.setFlat(False)
        return super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        if self._isFeatureSet:
            self.setFlat(True)
        self.update()
        return super().leaveEvent(event)

    def setSizeLongestText(self, longestText):
        currentText = self.text()
        self.setText(longestText)
        w, h = self.sizeHint().width(), self.sizeHint().height()
        self.setMinimumWidth(w+10)
        # self.setMinimumHeight(h+5)
        self.setText(currentText)

class CheckableSpinBoxWidgets:
    def __init__(self, isFloat=True):
        if isFloat:
            self.spinbox = FloatLineEdit()
        else:
            self.spinbox = SpinBox()
        self.checkbox = QCheckBox('Activate')
        self.spinbox.setEnabled(False)
        self.checkbox.toggled.connect(self.spinbox.setEnabled)
    
    def value(self):
        if not self.checkbox.isChecked():
            return
        return self.spinbox.value()

class Label(QLabel):
    def __init__(self, parent=None, force_html=False):
        super().__init__(parent)
        self._force_html = force_html
        
    def setText(self, text):
        if self._force_html:
            text = html_utils.paragraph(text)
        super().setText(text)
        
        
class LabelItem(pg.LabelItem):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def bbox(self):
        xl, yl = self.pos().x(), self.pos().y()
        wl, hl = self.itemRect().width(), self.itemRect().height()
        return yl, xl, yl+hl, xl+wl
    
    def setBold(self, bold=True):
        self.origPos = self.pos()
        self.setText(self.text, bold=bold)
        self.setPos(self.origPos)

class ScaleBar(QGraphicsObject): 
    sigEditProperties = Signal(object)
    
    def __init__(self, imageShape, parent=None):
        super().__init__(parent)
        self.SizeY, self.SizeX = imageShape
        self.plotItem = PlotCurveItem()
        self.labelItem = LabelItem()
        self._x_pad = 5
        self._y_pad = 3
        self._highlighted = False
        self._parent = parent
        self.clicked = False
        self.createContextMenu()
    
    def createContextMenu(self):
        self.contextMenu = QMenu()
        action = QAction('Edit properties...', self.contextMenu)
        action.triggered.connect(self.emitEditProperties)
        self.contextMenu.addSeparator()
        self.contextMenu.addAction(action)
    
    def emitEditProperties(self):
        self.setHighlighted(False)
        self.sigEditProperties.emit(self.properties())
    
    def isHighlighted(self):
        return self._highlighted
    
    
        
    def setHighlighted(self, highlighted):
        if self._highlighted and highlighted:
            return
        
        if not self._highlighted and not highlighted:
            return
        
        pen = self.highlightPen if highlighted else self.pen
        self.labelItem.setBold(bold=highlighted)
        self.plotItem.setPen(pen)
        
        self._highlighted = highlighted
    
    def showContextMenu(self, x, y):
        self.contextMenu.popup(QPoint(int(x), int(y)))
    
    def properties(self):
        properties = {
            'thickness': self._thickness,
            'length_pixel': self._length,
            'length_unit': self._length_unit,
            'is_text_visible': self._is_text_visible,
            'color': self._color,
            'loc': self._loc,
            'font_size': float(self._font_size[:-2]),
            'unit': self._unit,
            'num_decimals': self._num_decimals
        }
        return properties
    
    def move(self, xm, ym):
        self._loc = 'Custom'
        
        Dy = ym - self.yc
        Dx = xm - self.xc
        
        x0 = self.x0c + Dx
        x1 = x0 + self._length
        y0 = y1 = self.y0c + Dy
        self.plotItem.setData([x0, x1], [y0, y0])
        self.setTextPos()
    
    def paint(self, painter, option, widget):
        pass
    
    def boundingRect(self):
        ymin, xmin, ymax, xmax = self.bbox()
        return QRectF(xmin, ymin, xmax-xmin, ymax-ymin)
    
    def setProperties(
            self, 
            length_pixel, 
            length_unit,
            thickness=3, 
            color='w',
            is_text_visible=True, 
            loc='top-left',
            font_size=12,
            unit='',
            num_decimals=0
        ):
        self._loc = loc
        self._color = color
        self._length = length_pixel
        self._length_unit = length_unit
        self._is_text_visible = is_text_visible
        self._font_size = f'{font_size}px'
        self._unit = unit
        self._num_decimals = num_decimals
        self._thickness = thickness
        self.pen = pg.mkPen(width=thickness, color=color, cosmetic=False)
        self.highlightPen = pg.mkPen(
            width=thickness+2, color=color, cosmetic=False
        )
        self.pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        self.highlightPen.setCapStyle(Qt.PenCapStyle.FlatCap)
        self.plotItem.setPen(self.pen)
    
    def addToAxis(self, ax):
        ax.addItem(self.plotItem)
        ax.addItem(self.labelItem)
    
    def setText(self):
        if self._is_text_visible:
            number = round(self._length_unit, self._num_decimals)
            if self._num_decimals == 0:
                number = int(number)
            text = f'{number} {self._unit}'
        else:
            text = ''
        self.labelItem.setText(
            text, color=self._color, size=self._font_size
        )
    
    def setTextPos(self):
        xx, yy = self.plotItem.getData()
        x0 = xx[0]
        y0 = yy[0]
        xc = x0 + self._length/2
        wl = self.labelItem.itemRect().width()
        hl = self.labelItem.itemRect().height()
        xl = xc-wl/2
        yt = y0-hl    
        self.labelItem.setPos(xl, yt)
    
    def getStartXCoordFromLoc(self, loc):
        if loc == 'custom':
            xx, yy = self.plotItem.getData()
            x0 = xx[0]
            return x0
        
        self.setText()
        wl = self.labelItem.itemRect().width()
        if loc.find('left') != -1:
            x0 = self._x_pad
            xc = x0 + self._length/2
            xl = xc-wl/2
            if xl < x0:
                # Text is larger than line --> move line to the right
                x0 = self._x_pad + abs(xl-self._x_pad)
        else:
            x0 = self.SizeX - self._length - self._x_pad
            xc = x0 + self._length/2
            x1 = x0 + self._length      
            xr = xc+wl/2
            if xr > x1:
                # Text is larger than line --> move line to the left
                delta_overshoot = xr - x1
                x0 = x0 - delta_overshoot
        return x0  
    
    def getStartYCoordFromLoc(self, loc):
        if loc == 'custom':
            xx, yy = self.plotItem.getData()
            y0 = yy[0]
            return y0
        
        self.setText()
        textHeight = self.labelItem.itemRect().height()
        if loc.find('top') != -1:
            return textHeight + self._y_pad
        else:
            return self.SizeY - self._y_pad - self._thickness
    
    def update(self):
        x0 = self.getStartXCoordFromLoc(self._loc) # + self._thickness/2
        y0 = self.getStartYCoordFromLoc(self._loc)
        
        x1 = x0 + self._length # - self._thickness/2
        self.plotItem.setData([x0, x1], [y0, y0])
        
        self.setText()
        self.setTextPos()
    
    def draw(self, length_pixel, length_unit, **kwargs):
        self.setProperties(length_pixel, length_unit, **kwargs)
        self.update()
        
    def bbox(self):
        y_line_min, x_line_min, y_line_max, x_line_max = self.plotItem.bbox()
        y_lab_min, x_lab_min, y_lab_max, x_lab_max = self.labelItem.bbox()
        ymin = min(y_line_min, y_lab_min)
        xmin = min(x_line_min, x_lab_min)
        ymax = max(y_line_max, y_lab_max)
        xmax = max(x_line_max, x_lab_max)
        return ymin, xmin, ymax, xmax
    
    def mousePressed(self, x, y):
        self.clicked = True
        self.xc, self.yc = x, y
        xx, yy = self.plotItem.getData()
        self.x0c = xx[0]
        self.y0c = yy[0]
    
    def removeFromAxis(self, ax):
        ax.removeItem(self.labelItem)
        ax.removeItem(self.plotItem)

class ComboBox(QComboBox):
    sigTextChanged = Signal(str)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._previousText = None
        self._valueChanged = False
        self.currentTextChanged.connect(self.emitTextChanged)
    
    def text(self):
        return self.currentText()
    
    def emitTextChanged(self, text):
        self._valueChanged = True
        self.sigTextChanged.emit(text)
    
    def mousePressEvent(self, event):
        self._previousText = self.currentText()
        super().mousePressEvent(event)
    
    def previousText(self):
        return self._previousText

    def addItems(self, items):
        super().addItems(items)
        self._previousText = items[0]
    
    def itemsText(self):
        return [self.itemText(i) for i in range(self.count())]
    
    def setCurrentIndex(self, idx):
        itemsText = self.itemsText()
        currentText = itemsText[idx]
        self._valueChanged = currentText != self._previousText
        self._previousText = self.currentText()
        super().setCurrentIndex(idx)
    
    def setCurrentText(self, text):
        currentText = text
        self._valueChanged = currentText != self._previousText
        self._previousText = self.currentText()
        super().setCurrentText(text)

class SetMeasurementsGroupBox(QGroupBox):
    def __init__(
            self, title, itemsText, checkable=True, itemsInfo=None, 
            lastSelection=None, itemsInfoUrls=None, parent=None
        ):
        super().__init__(parent)
        
        if itemsInfo is None:
            itemsInfo = {}
        
        if itemsInfo is None:
            itemsInfoUrls = {}
        
        highlightRgba = _palettes._highlight_rgba()
        r, g, b, a = highlightRgba
        self._highlightStylesheetColor = f'rgb({r}, {g}, {b})'
        
        self.setTitle(title)
        self.setCheckable(checkable)
        
        mainLayout = QVBoxLayout()
        
        scrollArea = QScrollArea()
        scrollArea.setWidgetResizable(True)
        scrollAreaLayout = QVBoxLayout()
        scrollAreaWidget = QWidget()
        self.scrollAreaWidget = scrollAreaWidget
        self.scrollAreaLayout = scrollAreaLayout
        
        self.checkboxes = {}
        for text in itemsText:
            rowLayout = QHBoxLayout()
            infoText = itemsInfo.get(text)
            infoUrl = itemsInfoUrls.get(text)
            if infoText is not None or infoUrl is not None:
                infoButton = infoPushButton()
                infoButton.setCursor(Qt.WhatsThisCursor)
                rowLayout.addWidget(infoButton)
            
            if infoText is not None:
                infoButton.itemText = text
                infoButton.infoText = infoText
                infoButton.clicked.connect(self.showInfo)
            
            if infoUrl is not None:
                infoButton.itemText = text
                infoButton.infoUrl = infoUrl
                infoButton.clicked.connect(self.openInfoUrl)
                
            checkbox = QCheckBox(text)
            checkbox.setParent(self.scrollAreaWidget)
            checkbox.setChecked(True)
            rowLayout.addWidget(checkbox)
            rowLayout.addStretch(1) 
            
            self.checkboxes[text] = checkbox
            
            scrollAreaLayout.addLayout(rowLayout)

        scrollAreaLayout.addStretch(1)
        
        scrollAreaWidget.setLayout(scrollAreaLayout)
        scrollArea.setWidget(scrollAreaWidget)
        self.scrollArea = scrollArea
        
        buttonsLayout = QHBoxLayout()
        self.selectAllButton = selectAllPushButton()
        self.selectAllButton.sigClicked.connect(self.setCheckedAll)
        
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(self.selectAllButton)
        self.buttonsLayout = buttonsLayout
        
        if lastSelection is not None:
            self.lastSelection = lastSelection
            self.loadLastSelButton = reloadPushButton(
                '  Load last selection...  '
            )
            self.loadLastSelButton.clicked.connect(self.loadLastSelection)
            buttonsLayout.addWidget(self.loadLastSelButton)
        
        mainLayout.addWidget(scrollArea)
        mainLayout.addSpacing(10)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def openInfoUrl(self):
        url = self.sender().infoUrl
        QDesktopServices.openUrl(QUrl(url))
        # import webbrowser
        # url = self.sender().infoUrl
        # webbrowser.open(url)
    
    def getWidthNoScrollBarNeeded(self):
        width = (
            self.scrollArea.verticalScrollBar().sizeHint().width()
            # self.scrollAreaLayout.contentsRect().width()
            + self.scrollAreaWidget.sizeHint().width() 
            + 30
        )
        buttonsWidth = 0
        for i in range(self.buttonsLayout.count()):
            widget = self.buttonsLayout.itemAt(i).widget()
            if not isinstance(widget, QPushButton):
                continue
            buttonsWidth += widget.sizeHint().width() + 16
        largerWidth = max(width, buttonsWidth)
        return largerWidth
    
    def resizeWidthNoScrollBarNeeded(self):
        width = self.getWidthNoScrollBarNeeded()
        self.setMinimumWidth(width)
        # self.setFixedWidth(width)
    
    def loadLastSelection(self):
        for text, checkbox in self.checkboxes.items():
            checked = self.lastSelection.get(text, False)
            checkbox.setChecked(checked)
            
    def showInfo(self):
        infoText = self.sender().infoText
        itemText = self.sender().itemText
        
        title = f'{itemText} description'
        msg = myMessageBox()
        msg.setWidth(int(self.screen().size().width()/2))
        msg.information(self, title, infoText)
    
    def setCheckedAll(self, button, checked):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(checked)

    def highlightCheckboxesFromSearchText(self, text):
        for checkbox in self.checkboxes.values():
            if not text:
                highlighted = False
            else:
                highlighted = checkbox.text().lower().find(text.lower()) != -1
            
            self.setCheckboxHighlighted(highlighted, checkbox)
    
    def setCheckboxHighlighted(self, highlighted, checkbox):
        if highlighted:
            checkbox.setStyleSheet(
                f'background: {self._highlightStylesheetColor}; color: black'
            )
            self.scrollArea.ensureWidgetVisible(checkbox)
        else:
            checkbox.setStyleSheet('')
    
class SearchLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.initSearch()
        self.setFocusPolicy(Qt.ClickFocus)
        
    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        if super().text() == 'Search...':
            self.setText('')
        self.setStyleSheet('')
    
    def focusOutEvent(self, event) -> None:
        super().focusOutEvent(event)
        if not super().text():
            self.initSearch()
    
    def initSearch(self):
        self.setText('Search...')
        self.setStyleSheet('color: rgb(150, 150, 150)')
        self.clearFocus()
    
    def text(self):
        if super().text() == 'Search...':
            return ''
        return super().text()

class ToolButtonTextIcon(rightClickToolButton):
    def __init__(self, text='', parent=None):
        super().__init__(parent=parent)
        self._text = text
        self._penColor = _palettes.text_pen_color()
    
    def setText(self, text):
        self._text = text
        self.update()
    
    def text(self):
        return self._text
    
    def paintEvent(self, event):
        QToolButton.paintEvent(self, event)
        p = QPainter(self)
        
        pen = pg.mkPen(color=self._penColor, width=2)
        p.setPen(pen)
        
        w, h = self.width(), self.height()
        sf = 0.7
        rect_w = w*sf
        rect_h = h*sf
        x = (w-rect_w)/2
        y = (h-rect_h)/2
        rect = QRectF(x, y, rect_w, rect_h) 
        
        font = p.font()
        font.setBold(True)
        font.setPixelSize(int(h/len(self._text)))
        p.setFont(font)
        
        p.drawText(rect, Qt.AlignCenter, self._text)
        p.end()

class RulerPlotItem(pg.PlotDataItem):
    def __init__(self, *args, **kwargs):
        self.labelItem = pg.LabelItem()
        super().__init__(*args, **kwargs)
        
    def setData(self, *args, lengthText='', **kwargs):
        super().setData(*args, **kwargs)
        self.labelItem.setText('')
        if not lengthText:
            return
        self.setLengthText(lengthText)
    
    def setLengthText(self, lengthText):
        xx, yy = self.getData()
        x0, x1 = sorted(xx)
        y0, y1 = sorted(yy)
        xc = round(x0 + (x1-x0)/2)
        yc = round(y0 + (y1-y0)/2)
        self.labelItem.setText(lengthText, size='11px', color='r')
        # xc = x0 + self._length/2
        wl = self.labelItem.itemRect().width()
        hl = self.labelItem.itemRect().height()
        xl = xc-wl/2
        yt = y0-hl    
        self.labelItem.setPos(xl, yt)

class VectorLineEdit(QLineEdit):
    valueChanged = Signal(object)
    valueChangeFinished = Signal(object)
    
    def __init__(self, parent=None, initial=None):
        super().__init__(parent)
        
        self._minimum = -np.inf
        
        float_re = float_regex()
        vector_regex = fr'\(?\[?{float_re}(,\s?{float_re})+\)?\]?'
        regex = fr'^{vector_regex}$|^{float_re}$'
        self.validRegex = regex
        
        regExp = QRegularExpression(regex)
        self.setValidator(QRegularExpressionValidator(regExp))
        self.setAlignment(Qt.AlignCenter)
        
        self.textChanged.connect(self.emitValueChanged)
        self.editingFinished.connect(self.emitValueChangeFinished)
        if initial is None:
            self.setText('0.0')
        
        font = QFont()
        font.setPixelSize(11)
        self.setFont(font)
    
    def emitValueChangeFinished(self):
        value = self.value()
        self.textChanged.disconnect()
        self.editingFinished.disconnect()
        self.setValue(value)
        self.textChanged.connect(self.emitValueChanged)
        self.editingFinished.connect(self.emitValueChangeFinished)
        
        self.emitValueChanged(self.text(), signal=self.valueChangeFinished)
        
    def emitValueChanged(self, text, signal=None):
        m = re.match(self.validRegex, text)
        if m is None:
            self.setStyleSheet(LINEEDIT_INVALID_ENTRY_STYLESHEET)
            return

        if signal is None:
            signal = self.valueChanged
        
        self.setStyleSheet('')
        signal.emit(self.value())
    
    def increaseValue(self, step):
        value = self.value()
        if isinstance(value, (float, int)):
            value += step
        else:
            value = [val+step for val in value]
            value = str(value).lstrip('[').rstrip(']')
        self.setValue(value)
        self.emitValueChangeFinished()
    
    def decreaseValue(self, step):
        value = self.value()
        if isinstance(value, (float, int)):
            value -= step
        else:
            value = [val-step for val in value]
            value = str(value).lstrip('[').rstrip(']')
        self.setText(value)
        self.emitValueChangeFinished()
    
    def setValue(self, value):
        if isinstance(value, (float, int)):
            if value < self._minimum:
                value = self._minimum
        else:
            clipped = []
            for val in value:
                if val < self._minimum:
                    val = self._minimum
                clipped.append(val)
            value = str(clipped).lstrip('[').rstrip(']')
        self.setText(value)
    
    def setText(self, text):
        super().setText(str(text))
    
    def clipValue(self, val: float):
        if val < self._minimum:
            val = self._minimum
        return val
    
    def value(self):
        m = re.match(self.validRegex, self.text())
        if m is None:
            return 0.0
        
        try: 
            value = self.clipValue(float(self.text()))
            return value
        except Exception as e:
            text = self.text()
            text = text.replace('(', '')
            text = text.replace(')', '')
            text = text.replace('[', '')
            text = text.replace(']', '')
            values = text.split(',')
            return [self.clipValue(float(value)) for value in values]
    
    def setMinimum(self, minimum):
        self._minimum = float(minimum)

class LatexLabel(QLabel):
    def __init__(self, latexText, parent=None):
        super().__init__(parent)
        
        latexText = latexText.replace('<latex>', '$')
        if not latexText.startswith('$'):
            latexText = f'${latexText}'
        
        if not latexText.endswith('$'):
            latexText = f'{latexText}$'
        
        latexText = latexText.replace('<br>', '\n')
        
        pixmap = self.mathTex_to_QPixmap(latexText)
        self.setPixmap(pixmap)
    
    def mathTex_to_QPixmap(self, mathTex):
        #---- set up a mpl figure instance ----

        fig = matplotlib.figure.Figure()
        fig.patch.set_facecolor('none')
        fig.set_canvas(FigureCanvasAgg(fig))
        renderer = fig.canvas.get_renderer()

        #---- plot the mathTex expression ----

        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')
        ax.patch.set_facecolor('none')
        t = ax.text(
            0, 0, mathTex, 
            ha='left', va='bottom', 
            fontsize=13, 
            color=TEXT_COLOR
        )

        #---- fit figure size to text artist ----

        fwidth, fheight = fig.get_size_inches()
        fig_bbox = fig.get_window_extent(renderer)

        text_bbox = t.get_window_extent(renderer)

        tight_fwidth = text_bbox.width * fwidth / fig_bbox.width
        tight_fheight = text_bbox.height * fheight / fig_bbox.height

        fig.set_size_inches(tight_fwidth, tight_fheight)

        #---- convert mpl figure to QPixmap ----

        buf, size = fig.canvas.print_to_buffer()
        qimage = QImage.rgbSwapped(QImage(
            buf, size[0], size[1], QImage.Format_ARGB32)
        )
        qpixmap = QPixmap(qimage)

        return qpixmap
        

class LabelsWidget(QWidget):
    def __init__(self, texts, wrapText=False, parent=None):
        super().__init__(parent=parent)
        
        layout = QVBoxLayout()
        
        texts = self.fixParagraphTags(texts)
        
        self.textLengths = []
        self.labels = []
        for t, text in enumerate(texts):
            if not text:
                continue
            if text.startswith('<latex>'):
                layout.addSpacing(10)
                label = LatexLabel(text)
                layout.addWidget(label, alignment=Qt.AlignCenter)
                try:
                    # Add spacing only if next text is not a formula
                    nextText = texts[t+1]
                    if not nextText.startswith('<latex>'):
                        layout.addSpacing(10)
                except IndexError:
                    layout.addSpacing(10)
            else:
                label = QLabel(text)
                label.setWordWrap(wrapText)
                label.setOpenExternalLinks(True)
                layout.addWidget(label)
                if wrapText:
                    self.textLengths.append(1)
                self.textLengths.extend(
                    [len(line) for line in text.split('<br>')]
                )
            self.labels.append(label)
        
        self.nCharsLongestLine = max(self.textLengths)
        
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
    
    def setWordWrap(self, wordWrap):
        for label in self.labels:
            label.setWordWrap(wordWrap)
    
    def fixParagraphTags(self, texts):
        firstText = texts[0]
        if firstText.find('<p style=') == -1:
            return texts
        
        searched = re.search(r'<p style="[\w\-\:\;]+">', firstText)
        if searched is None:
            openTag = '<p style="font-size:13px;">'
        else:
            openTag = searched.group()
        
        not_allowed = {' ', '\n'}
        
        fixedTexts = []
        for text in texts:
            if text.startswith('<latex>'):
                fixedTexts.append(text)
                continue
            
            if set(text) <= not_allowed:
                # Ignore texts that are made of only \n and spaces
                continue
            
            if text.find('</p>') == -1:
                text = f'{text}<\p>'
            
            if text.find(openTag) == -1:
                text = f'{openTag}{text}'
            
            text = text.replace('\n', '')
            
            fixedTexts.append(text)
        return fixedTexts

class SwitchPlaneCombobox(QComboBox):
    sigPlaneChanged = Signal(str, str)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.addItems(['xy', 'zy', 'zx'])
        self._previousPlane = 'xy'
        self.currentTextChanged.connect(self.emitPlaneChanged)
    
    def emitPlaneChanged(self, plane):
        self.sigPlaneChanged.emit(self._previousPlane, plane)
        self._previousPlane = plane
    
    def setPlane(self, plane):
        self.setCurrentText(plane)
    
    def setCurrentText(self, text):
        self._previousPlane = self.plane()
        super().setCurrentText(text)
    
    def plane(self):
        return self.currentText()

    def depthAxes(self):
        plane = self.plane()
        for axes in 'xyz':
            if axes not in plane:
                return axes

class SamInputPointsWidget(QWidget):
    isWidget = True
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        _layout = QHBoxLayout()
        
        self.lineEntry = ElidingLineEdit(parent=self)
        self.lineEntry.setAlignment(Qt.AlignCenter)
        
        self.editButton = editPushButton()
        self.browseButton = browseFileButton(
            ext={'CSV': '.csv'}, 
            start_dir=myutils.getMostRecentPath()
        )
        
        _layout.addWidget(self.lineEntry)
        _layout.addWidget(self.editButton)
        _layout.addWidget(self.browseButton)
        
        _layout.setStretch(0, 1)
        _layout.setStretch(1, 0)
        _layout.setStretch(1, 0)
        
        self.browseButton.sigPathSelected.connect(self.browseCsvFiles)
        self.editButton.clicked.connect(self.showInfoEditPoints)
        
        _layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(_layout)
    
    def showInfoEditPoints(self):
        note = html_utils.to_note(
            'When adding points with the mouse left button you will create a '
            'new object for each point. To add multiple points for the same '
            'object click the right button.'
        )
        txt = html_utils.paragraph(f"""
            To add input points for Segment Anything open the GUI (module 3), 
            load the data, and then click on the button<br>
            on the top toolbar called <code>Add points layer</code>.<br><br>
            Select the option "Add points by clicking" and click on the image 
            to add points.<br><br>
            Finally, save the table and browse to the saved file on this widget.
            <br>{note}
        """)
        msg = myMessageBox(wrapText=False)
        msg.information(self, 'Info edit points', txt)
    
    def criticalMissingColumn(self, filepath, missing_col):
        txt = html_utils.paragraph(f"""
            [ERROR]: The selected table does not contain the column 
            <code>{missing_col}</code>.<br><br>
            A valid table must contain the columns <code>(x, y, id)</code> 
            with an additional <code>z</code> column for 3D z-stacks data.
        """)
        msg = myMessageBox(wrapText=False)
        msg.critical(self, 'Invalid table', txt)
    
    def setValue(self, value: str):
        self.lineEntry.setText(value)
    
    def value(self):
        return self.lineEntry.text()
    
    def browseCsvFiles(self, filepath):
        # Check if metadata.csv file exists with basename and set only the 
        # endname of the file
        df_points = pd.read_csv(filepath)
        for col in ('x', 'y', 'id'):
            if col not in df_points.columns:
                self.criticalMissingColumn(filepath, col)
                return
        
        # Check if basename is present in metadata
        folderpath = os.path.dirname(filepath)
        basename = None
        for file in myutils.listdir(folderpath):
            if file.endswith('metadata.csv'):
                metadata_csv_path = os.path.join(folderpath, file)
                df = pd.read_csv(metadata_csv_path, index_col='Description')
                try:
                    basename = df.at['basename', 'values']
                except Exception as e:
                    basename = None
                break
        
        # Check if file is inside images folder and get basename
        is_images_folder = folderpath.endswith('Images')
        if is_images_folder:
            images_path = folderpath
            img_filepath = None
            for file in myutils.listdir(images_path):
                if file.endswith('.tif'):
                    img_filepath = os.path.join(images_path, file)
                    break
                
                if file.endswith('aligned.npz'):
                    img_filepath = os.path.join(images_path, file)
                    break
                
            if img_filepath is not None:
                posData = load.loadData(img_filepath, '', QParent=self)
                posData.getBasenameAndChNames()
                filename = os.path.basename(filepath)
                if filename.startswith(posData.basename):
                    basename = posData.basename
        
        if basename is None:
            self.lineEntry.setText(filepath)
        else:
            filename = os.path.basename(filepath)
            endname = filename[len(basename):]
            self.lineEntry.setText(endname)

class PointsScatterPlotItem(pg.ScatterPlotItem):
    def __init__(self, *args, ax=None, **kwargs):
        self.textItem = annotate.TextAnnotationsScatterItem(
            size=12, anchor=(1.0, 1.0)
        )
        self.textItem.createSymbols(
            [str(id) for id in range(200)], includeBold=False
        )
        # self._textItems = {}
        super().__init__(*args, **kwargs)
        self.textItem.setParentItem(self)
        self._font = QFont()
        self._font.setPixelSize(12)
        self.drawIds = True
        self.ax = ax
        
    def setData(self, *args, **kwargs):
        self.clearTextItems()
        super().setData(*args, **kwargs)
        data = kwargs.get('data')
        if data is None:
            return
        if len(data) == 0:
            return
        first_point_data = data[0]
        if not isinstance(first_point_data, (int, str)):
            return
        
        if not self.drawIds:
            return
        
        color = self.opts['brush'].color()
        self.textItem.setColors({'id': color.getRgb()})
        size = self.opts['size']
        radius = size/2
        # xx, yy = args
        # for x, y, point_data in zip(xx, yy, data):
        for point in self.points():
            text = str(point.data())
            if not text:
                continue
            
            x, y = point.pos().x(), point.pos().y()
            xt, yt = x+radius-0.5, y-radius+0.5
            opts = {
                'text': text, 
                'bold': False, 
                'color_name': 'id', 
            }
            data = self.textItem.addObjAnnot(
                (xt, yt), anchor=(-0.3, 1.3), **opts
            )
            self.textItem.appendData(data, opts['text'])
        
        self.textItem.draw()
            # hexColor = color.name()
            # htmlText = html_utils.span(
            #     text, color=hexColor, font_size='13pt', bold=True
            # )
            
            # textItem = self._textItems.get((x, y))
            # if textItem is None:
            #     textItem = pg.TextItem(html=htmlText, anchor=(0, 1))
            #     textItem.setParentItem(self)
            #     self._textItems[(x, y)] = textItem
            #     self.ax.addItem(textItem)
            # else:
            #     textItem.setHtml(htmlText)
            # textItem.setPos(x+radius-0.5, y-radius+0.5)      
        
    def clearTextItems(self):
        self.textItem.clearData()
        # for textItem in self._textItems.values():
        #     textItem.setText('')
    
    def clear(self):
        super().clear()
        self.clearTextItems()
    
    def setVisible(self, visible):
        super().setVisible(visible)
        self.textItem.setVisible(visible)

class installJavaDialog(myMessageBox):
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
        self._layout.addWidget(
            self.scrollArea, self.currentRow, 1, alignment=Qt.AlignTop
        )

        # Stretch last row
        self.currentRow += 1
        self._layout.setRowStretch(self.currentRow, 1)

    def viewScreenshot(self, checked=False):
        self.screenShotWin = view_visualcpp_screenshot(parent=self)
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
            msg = myMessageBox(wrapText=False)
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

        self.selectFramesGroupbox = selectStartStopFrames(
            SizeT, currentFrameNum=currentFrameNo, parent=parent
        )

        self.mainLayout.insertWidget(1, self.selectFramesGroupbox)

    def ok_cb(self, event):
        if self.selectFramesGroupbox.warningLabel.text():
            return
        else:
            self.startFrame = self.selectFramesGroupbox.startFrame_SB.value()
            self.stopFrame = self.selectFramesGroupbox.stopFrame_SB.value()
            super().ok_cb(event)

def addWidgetToScrollArea(
        widget, 
        resizeMinWidthNoHorizontalScrollbar=False, 
        resizeMinHeightNoVerticalScrollbar=False
    ):
    container = QWidget()
    layout = QVBoxLayout()
    layout.addWidget(widget)
    layout.addStretch(1)
    container.setLayout(layout)
    scrollArea = QScrollArea()
    scrollArea.setWidgetResizable(True)
    scrollArea.setWidget(container)
    
    if resizeMinWidthNoHorizontalScrollbar:
        scrollArea.setMinimumWidth(
            container.sizeHint().width()
            + scrollArea.verticalScrollBar().sizeHint().width()
        )
    
    if resizeMinHeightNoVerticalScrollbar:
        scrollArea.setMinimumHeight(
            container.sizeHint().height()
            + scrollArea.horizontalScrollBar().sizeHint().height()
        )
    
    return scrollArea

class CheckableAction(QAction):
    clicked = Signal(bool)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setCheckable(True)
        self.toggled.connect(self.emitClicked)
    
    def emitClicked(self, checked):
        self.clicked.emit(checked)
    
    def setChecked(self, checked):
        self.toggled.disconnect()
        super().setChecked(checked)
        self.toggled.connect(self.emitClicked)

class OddSpinBox(SpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.setSingleStep(2)
        self.editingFinished.connect(self.roundToOdd)
    
    def roundToOdd(self):
        if self.value() % 2 == 1:
            return
        
        self.setValue(self.value()+1)

class TimestampItem(LabelItem):
    sigEditProperties = Signal(object)
    
    def __init__(self, SizeY, SizeX, secondsPerFrame=1, parent=None):
        self._secondsPerFrame = secondsPerFrame
        self._x_pad = 3
        self._y_pad = 2
        self.SizeY = SizeY
        self.SizeX = SizeX
        self._highlighted = False
        self._parent = parent
        self.clicked = False
        super().__init__(self)
        self.createContextMenu()
    
    def createContextMenu(self):
        self.contextMenu = QMenu()
        action = QAction('Edit properties...', self.contextMenu)
        action.triggered.connect(self.emitEditProperties)
        self.contextMenu.addSeparator()
        self.contextMenu.addAction(action)
    
    def mousePressed(self, x, y):
        self.clicked = True
    
    def emitEditProperties(self):
        self.setHighlighted(False)
        self.sigEditProperties.emit(self.properties())
    
    def isHighlighted(self):
        return self._highlighted
        
    def setHighlighted(self, highlighted):
        if self._highlighted and highlighted:
            return
        
        if not self._highlighted and not highlighted:
            return
        
        super().setText(self.text, bold=highlighted)
        
        self._highlighted = highlighted
    
    def showContextMenu(self, x, y):
        self.contextMenu.popup(QPoint(int(x), int(y)))
    
    def properties(self):
        properties = {
            'color': self._color,
            'loc': self._loc,
            'font_size': float(self._font_size[:-2]),
        }
        return properties

    def draw(self, frame_i, **kwargs):
        self.setProperties(**kwargs)
        self.update(frame_i)
    
    def update(self, frame_i):
        self.setPosFromLoc()
        self.setText(frame_i)
    
    def setPosFromLoc(self):
        textHeight = self.itemRect().height()
        textWidth = self.itemRect().width()
        if self._loc == 'custom':
            pos = self.pos()
            x0, y0 = pos.x(), pos.y()
        elif self._loc.find('top') != -1:
            y0 = self._y_pad
        else:
            y0 = self.SizeY - textHeight
        
        if self._loc.find('left') != -1:
            x0 = self._x_pad
        else:
            x0 = self.SizeX - textWidth
        
        self.setPos(x0, y0)
    
    def setProperties(
            self, 
            color=(255, 255, 255), 
            font_size='13px', 
            loc='top-left',
        ):
        self._color = color
        self._loc = loc
        self._font_size = font_size

    def move(self, xm, ym):
        Dy = ym - self.yc
        Dx = xm - self.xc
        x0 = self.x0c + Dx
        y0 = self.y0c + Dy
        self.setPos(x0, y0)
    
    def mousePressed(self, x, y):
        self.clicked = True
        self.xc, self.yc = x, y
        self.x0c = self.pos().x()
        self.y0c = self.pos().y()
    
    def setText(self, frame_i):
        if not isinstance(frame_i, int):
            return
        seconds = frame_i*self._secondsPerFrame
        timedelta = datetime.timedelta(seconds=round(seconds))
        super().setText(
            str(timedelta), color=self._color, size=self._font_size
        )
    
    def addToAxis(self, ax):
        ax.addItem(self)
    
    def removeFromAxis(self, ax):
        ax.removeItem(self)

class FontSizeWidget(QWidget):
    sigTextChanged = Signal(str)
    
    def __init__(self, parent=None, unit='px', initalVal=12):
        super().__init__(parent)
        
        layout = QHBoxLayout()
        
        self.spinbox = SpinBox()    
        self.spinbox.setValue(initalVal)    
        layout.addWidget(self.spinbox)
        
        self.unitLabel = QLabel(unit)
        layout.addWidget(self.unitLabel)
        
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setStretch(0, 1)
        layout.setStretch(1, 0)
        
        self.setLayout(layout)
        
        self.spinbox.valueChanged.connect(self.emitTextChanged)
    
    def emitTextChanged(self, value):
        self.sigTextChanged.emit(self.text())
    
    def setValue(self, value):
        if isinstance(value, str):
            value = int(value.replace(self.unitLabel.text(), '').strip())
        self.spinbox.setValue(value)
    
    def setText(self, text):
        value = int(text.replace(self.unitLabel.text(), '').strip())
        self.setValue(value)
    
    def text(self):
        return f'{self.spinbox.value()}{self.unitLabel.text()}'
    
    def value(self):
        return self.spinbox.value()

class RangeSelector(QWidget):
    sigRangeChanged = Signal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout()
        
        self.lowSpinbox = DoubleSpinBox()
        self.highSpinbox = DoubleSpinBox()
        
        layout.addWidget(self.lowSpinbox)
        layout.addWidget(self.highSpinbox)
        
        layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        
        self.lowSpinbox.valueChanged.connect(self.lowValueChanged)
        self.highSpinbox.valueChanged.connect(self.highValueChanged)
    
    def lowValueChanged(self, value):        
        self.emitRangeChanged()
        
    def highValueChanged(self, value):
        self.emitRangeChanged()
    
    def emitRangeChanged(self):
        self.sigRangeChanged.emit(*self.range())
    
    def setRangeNoEmit(self, lowValue, highValue, decimals=3):
        self.lowSpinbox.valueChanged.disconnect()
        self.highSpinbox.valueChanged.disconnect()
        
        self.setRange(round(lowValue, 3), round(highValue, 3))
        
        self.lowSpinbox.valueChanged.connect(self.lowValueChanged)
        self.highSpinbox.valueChanged.connect(self.highValueChanged)
    
    def setRange(self, lowValue, highValue):
        self.lowSpinbox.setValue(lowValue)
        self.highSpinbox.setValue(highValue)
    
    def range(self):
        return self.lowSpinbox.value(), self.highSpinbox.value()

class LineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
    
    def value(self):
        return self.text()

    def setValue(self, value):
        self.setText(str(value))

class PreProcessingSelector(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.addItems(PREPROCESS_MAPPER.keys())
    
    def htmlInfo(self):
        href = html_utils.href_tag('GitHub page', urls.issues_url)
        docstring = PREPROCESS_MAPPER[self.currentText()]['docstring']
        if docstring is None:
            text = 'This function is not documented, yet.'
        else:
            text = html_utils.rst_docstring_to_html(
                docstring, 
                args_subset=PREPROCESS_MAPPER[self.currentText()]['args']
            )
        text = (
            f'{text}<br><br>'
            f'Feel free to submit an issue on our {href} if you '
            'need help with this filter.'
        )
        return text
        
    def widgets(self):
        return PREPROCESS_MAPPER[self.currentText()]['widgets']

class RescaleImageJroisGroupbox(QGroupBox):
    def __init__(self, TZYX_out_shape, parent=None):
        super().__init__(parent)
        
        self.setTitle('Rescale ROIs')
        self.setCheckable(True)
        
        gridLayout = QGridLayout()
        
        dims = ('Z', 'Y', 'X')
        self.widgets = {}
        for row, SizeD in enumerate(TZYX_out_shape[1:]):
            if SizeD == 1:
                continue
            
            dim = dims[row]
            inputSpinbox = SpinBox()
            inputSpinbox.setMinimum(1)
            inputSpinbox.setValue(SizeD)
            
            outZwidget = QLineEdit()
            outZwidget.setReadOnly(True)
            outZwidget.setAlignment(Qt.AlignCenter)
            # outZwidget.setValue(SizeD)
            outZwidget.setText(str(SizeD))

            row0 = row*2
            row1 = row0+1
            gridLayout.addWidget(QLabel(f'{dim}-dimension: '), row1, 0)
            
            gridLayout.addWidget(QLabel('Input size'), row0, 1)
            gridLayout.addWidget(inputSpinbox, row1, 1)
            
            gridLayout.addWidget(QLabel('Output size'), row0, 2)
            gridLayout.addWidget(outZwidget, row1, 2)
            
            self.widgets[dim] = (inputSpinbox, SizeD)
        
        self.setLayout(gridLayout)
    
    def inputOutputSizes(self):
        if not self.isChecked():
            return
        
        sizes = {
            dim: (spinbox.value(), int(SizeD))
            for dim, (spinbox, SizeD) in self.widgets.items()
        }
        return sizes