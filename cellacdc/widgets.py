import os
import sys
import operator
import time
import re
from matplotlib.pyplot import text
import numpy as np
import string
import traceback
import logging
import difflib
from functools import partial

import skimage.draw
import skimage.morphology

from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.pyplot as plt

from PyQt5.QtCore import (
    pyqtSignal, QTimer, Qt, QPoint, pyqtSlot, pyqtProperty,
    QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup,
    QSize, QRect, QPointF, QRect, QPoint, QEasingCurve, QRegExp,
    QEvent, QEventLoop, QPropertyAnimation, QObject,
    QItemSelectionModel, QAbstractListModel, QModelIndex,
    QByteArray, QDataStream, QMimeData, QAbstractItemModel, 
    QIODevice, QItemSelection
)
from PyQt5.QtGui import (
    QFont, QPalette, QColor, QPen, QKeyEvent, QBrush, QPainter,
    QRegExpValidator, QIcon, QPixmap, QKeySequence, QLinearGradient,
    QShowEvent, QBitmap, QFontMetrics, QGuiApplication, QLinearGradient 
)
from PyQt5.QtWidgets import (
    QTextEdit, QLabel, QProgressBar, QHBoxLayout, QToolButton, QCheckBox,
    QApplication, QWidget, QVBoxLayout, QMainWindow, QStyleFactory,
    QLineEdit, QSlider, QSpinBox, QGridLayout, QRadioButton,
    QScrollArea, QSizePolicy, QComboBox, QPushButton, QScrollBar,
    QGroupBox, QAbstractSlider, QDoubleSpinBox, QWidgetAction,
    QAction, QTabWidget, QAbstractSpinBox, QToolBar, QStyleOptionSpinBox,
    QStyle, QDialog, QSpacerItem, QFrame, QMenu, QActionGroup,
    QListWidget, QPlainTextEdit, QFileDialog, QListView, QAbstractItemView,
    QTreeWidget, QTreeWidgetItem, QListWidgetItem, QLayout, QStylePainter
)

import pyqtgraph as pg

from . import myutils, measurements, is_mac, is_win, html_utils
from . import qrc_resources, printl, temp_path
from . import colors, config

font = QFont()
font.setPixelSize(13)

custom_cmaps_filepath = os.path.join(temp_path, 'custom_colormaps.ini')

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

class QBaseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

    def exec_(self, resizeWidthFactor=None):
        if resizeWidthFactor is not None:
            self.show()
            self.resize(int(self.width()*resizeWidthFactor), self.height())
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

class XStream(QObject):
    _stdout = None
    _stderr = None
    messageWritten = pyqtSignal(str)
    
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
    sigClose = pyqtSignal()

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
    def __init__(self, *args, icon=None, alignIconLeft=False):
        super().__init__(*args)
        if icon is not None:
            self.setIcon(icon)
        self.alignIconLeft = alignIconLeft
        self._text = None
    
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
        self.layout().addWidget(textLabel)
        super().show()
        
    def setText(self, text):
        if self._text is None:
            super().setText(self._text)

class mergePushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':merge-IDs.svg'))

class okPushButton(QPushButton):
    def __init__(self, *args, isDefault=True):
        super().__init__(*args)
        self.setIcon(QIcon(':yesGray.svg'))
        if isDefault:
            self.setDefault(True)
        # QShortcut(Qt.Key_Return, self, self.click)
        # QShortcut(Qt.Key_Enter, self, self.click)

class zoomPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':zoom_out.svg'))
    
    def setIconZoomOut(self):
        self.setIcon(QIcon(':zoom_out.svg'))
    
    def setIconZoomIn(self):
        self.setIcon(QIcon(':zoom_in.svg'))

class reloadPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':reload.svg'))

class savePushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':file-save.svg'))

class autoPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':cog_play.svg'))

class newFilePushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':file-new.svg'))

class helpPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':help.svg'))

class viewPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':eye.svg'))

class infoPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':info.svg'))

class threeDPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':3d.svg'))

class twoDPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':2d.svg'))

class addPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':add.svg'))

class futurePushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':arrow_future.svg'))

class currentPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':arrow_current.svg'))

class arrowUpPushButton(PushButton):
    def __init__(self, *args, alignIconLeft=False):
        super().__init__(
            *args, icon=QIcon(':arrow-up.svg'), alignIconLeft=alignIconLeft
        )

class arrowDownPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':arrow-down.svg'))

class subtractPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':subtract.svg'))

class continuePushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':continue.svg'))

class calcPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':calc.svg'))

class playPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':play.svg'))

class stopPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':stop.svg'))

class copyPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':edit-copy.svg'))

class movePushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':folder-move.svg'))

class showInFileManagerButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':folder-open.svg'))

class showDetailsButton(QPushButton):
    def __init__(self, *args, txt='Show details...'):
        super().__init__(*args)
        self.setText(txt)
        self.txt = txt
        self.checkedIcon = QIcon(':hideUp.svg')
        self.uncheckedIcon = QIcon(':showDown.svg')
        self.setIcon(self.uncheckedIcon)
        self.toggled.connect(self.onClicked)
        w = self.sizeHint().width()
        self.setFixedWidth(w)

    def onClicked(self, checked):
        if checked:
            self.setText(' Hide details   ')
            self.setIcon(self.checkedIcon)
        else:
            self.setText(self.txt)
            self.setIcon(self.uncheckedIcon)

class cancelPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':cancelButton.svg'))

class setPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':cog.svg'))

class noPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':no.svg'))

class editPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':edit-id.svg'))

class delPushButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setIcon(QIcon(':bin.svg'))

class browseFileButton(QPushButton):
    sigPathSelected = pyqtSignal(str)

    def __init__(
            self, *args, ext=None, title='Select file', start_dir='', 
            openFolder=False
        ):
        super().__init__(*args)
        self.setIcon(QIcon(':folder-open.svg'))
        self.clicked.connect(self.browse)
        self._file_types = 'All Files (*)'
        self._title = title
        self._start_dir = start_dir
        self._openFolder = openFolder
        if ext is not None:
            s = ''
            s_li = []
            for name, extensions in ext.items():
                _s = ''
                for ext in extensions:
                    _s = f'{_s}*{ext} '
                s_li.append(f'{name} {_s.strip()}')

            self._file_types = ';;'.join(s_li)
            self._file_types = f'{self._file_types};;All Files (*)'

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

class CustomAnnotationScatterPlotItem(BaseScatterPlotItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)

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

        buttonsLayout = CancelOkButtonsLayout()

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
            msg = myMessageBox(showCentered=False)
            msg.critical(self, 'Empty', 'Entry cannot be empty.')
            return
        self.cancel = False
        self.close()

class ElidingLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._text = ''
    
    def setText(self, text: str, width=None, elide=True) -> None:
        if width is None:
            width = self.width()
        self._text = text
        if not elide:
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
    
    def focusInEvent(self, event):
        super().focusInEvent(event)
        self.setText(self._text, elide=False)
        self.setCursorPosition(len(self.text()))

    def focusOutEvent(self, event):
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
    sigIDsChanged = pyqtSignal(list)
    sigSort = pyqtSignal()

    def __init__(self, instructionsLabel, parent=None):
        super().__init__(parent)

        self.validPattern = '^[0-9-, ]+$'
        self.setValidator(QRegExpValidator(QRegExp(self.validPattern)))

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

class _ReorderableListModel(QAbstractListModel):
    '''
    ReorderableListModel is a list model which implements reordering of its
    items via drag-n-drop
    '''
    dragDropFinished = pyqtSignal()

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
            QItemSelectionModel.ClearAndSelect 
            | QItemSelectionModel.Rows 
            | QItemSelectionModel.Current
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
        self.setDragDropMode(QAbstractItemView.InternalMove)
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
    sigSelectionConfirmed = pyqtSignal(list)

    def __init__(
            self, title, text, items, cancelText='Cancel',
            multiSelection=True, parent=None,
            additionalButtons=(), includeSelectionHelp=False,
            allowSingleSelection=True
        ):
        self.cancel = True
        super().__init__(parent)
        self.setWindowTitle(title)

        self.allowSingleSelection = allowSingleSelection

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
            listBox.setSelectionMode(QAbstractItemView.ExtendedSelection)
        else:
            listBox.setSelectionMode(QAbstractItemView.SingleSelection)
        listBox.setCurrentRow(0)
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
        self.areItemsSelected = [
            listBox.item(i).isSelected() for i in range(listBox.count())
        ]
    
    def keyPressEvent(self, event) -> None:
        mod = event.modifiers()
        if mod == Qt.ShiftModifier or mod == Qt.ControlModifier:
            self.listBox.setSelectionMode(QAbstractItemView.ExtendedSelection)
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
            self.listBox.setSelectionMode(QAbstractItemView.ExtendedSelection)
            return
        
        self.listBox.setSelectionMode(QAbstractItemView.MultiSelection)
        itemIdx = self.listBox.row(item)
        wasSelected = self.areItemsSelected[itemIdx]
        if wasSelected:
            item.setSelected(False)
        
        self.areItemsSelected = [
            self.listBox.item(i).isSelected() 
            for i in range(self.listBox.count())
        ]
        # self.listBox.setSelectionMode(QAbstractItemView.ExtendedSelection)
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
    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout()
        self.le = QLineEdit()
        self.browseButton = browseFileButton()

        layout.addWidget(self.le)
        layout.addWidget(self.browseButton)
        self.setLayout(layout)

        self.le.editingFinished.connect(self.setTextTooltip)
        self.browseButton.sigPathSelected.connect(self.setText)
    
        self.setFrameStyle(QFrame.StyledPanel)

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

class QHWidgetSpacer(QWidget):
    def __init__(self, width=10, parent=None) -> None:
        super().__init__(parent)
        self.setFixedWidth(width)

class QVWidgetSpacer(QWidget):
    def __init__(self, height=10, parent=None) -> None:
        super().__init__(parent)
        self.setFixedHeight(height)

class QHLine(QFrame):
    def __init__(self, shadow='Sunken', parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(getattr(QFrame, shadow))

class QVLine(QFrame):
    def __init__(self, shadow='Plain', parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(getattr(QFrame, shadow))

class VerticalResizeHline(QFrame):
    dragged = pyqtSignal(object)
    clicked = pyqtSignal(object)
    released = pyqtSignal(object)

    def __init__(self):
        super().__init__()
        self.setCursor(Qt.SizeVerCursor)
        self.setFrameShape(QFrame.HLine)
        self.setFrameShadow(QFrame.Sunken)
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
        if event.type() == QEvent.Enter:
            self.setLineWidth(0)
            self.setMidLineWidth(self._height)
            pal = self.palette()
            pal.setColor(QPalette.WindowText, QColor('#4d4d4d'))
            self.setPalette(pal)
            # self.setStyleSheet('background-color: #4d4d4d') 
        elif event.type() == QEvent.Leave:
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
    sigLeaveEvent = pyqtSignal()

    def __init__(
            self, parent=None, resizeVerticalOnShow=False, 
            dropArrowKeyEvents=False
        ) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameStyle(QFrame.NoFrame)
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
            QSizePolicy.Preferred, QSizePolicy.Preferred
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
            QSizePolicy.Preferred, QSizePolicy.Preferred
        )

        self.setFixedHeight(height)

    def eventFilter(self, object, event: QEvent):
        if event.type() == QEvent.Leave:
            self.sigLeaveEvent.emit()

        if object != self.containerWidget:
            return False
        
        isResize = event.type() == QEvent.Resize
        isShow = event.type() == QEvent.Show
        if isResize and self.isOnlyVertical:
            self._resizeHorizontal()
        elif isShow and self.resizeVerticalOnShow:
            self._resizeVertical()
        return False

class QClickableLabel(QLabel):
    clicked = pyqtSignal(object)

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
        if event.type() == QEvent.MouseButtonPress:
            if self._isPopupVisibile:
                self.hidePopup()
                self._isPopupVisibile = False
            else:
                self.showPopup()
                self._isPopupVisibile = True
        return False

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
        self.setFont(font)
    
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

class OrderableList(listWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def addItems(self, items):
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        nr_items = len(items)
        nn = [str(n) for n in range(1, nr_items+1)]
        for i, item in enumerate(items):
            itemW = QListWidgetItem()
            itemContainer = QWidget()
            itemText = QLabel(item)
            itemLayout = QHBoxLayout()
            itemNumberWidget = QComboBox()
            itemNumberWidget.addItems(nn)
            itemLayout.addWidget(itemText)
            itemLayout.addWidget(QLabel('| Table nr.'))
            itemLayout.addWidget(itemNumberWidget)
            itemContainer.setLayout(itemLayout)
            itemLayout.setSizeConstraint(QLayout.SetFixedSize)
            itemW.setSizeHint(itemContainer.sizeHint())
            self.addItem(itemW)
            self.setItemWidget(itemW, itemContainer)
            itemW._text = item
            itemW._nrWidget = itemNumberWidget
            itemNumberWidget.setDisabled(True)
            itemNumberWidget.textActivated.connect(self.onTextActivated)
            itemNumberWidget._currentNr = 1
            itemNumberWidget.row = i
        
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
            if item.isSelected():
                item._nrWidget.setDisabled(False)
            else:
                item._nrWidget.setDisabled(True)
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
        self.setFont(font)
        if multiSelection:
            self.setSelectionMode(QAbstractItemView.ExtendedSelection)
            self.itemClicked.connect(self.selectAllChildren)
    
    def selectAllChildren(self, item):
        if item.childCount() == 0:
            return

        for i in range(item.childCount()):
            item.child(i).setSelected(True)

class CancelOkButtonsLayout(QHBoxLayout):
    def __init__(self, *args):
        super().__init__(*args)

        self.cancelButton = cancelPushButton('Cancel')
        self.okButton = okPushButton(' Ok ')

        self.addStretch(1)
        self.addWidget(self.cancelButton)
        self.addSpacing(20)
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
    sigFilteredEvent = pyqtSignal(object, object)

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
    def __init__(self, parent=None):
        super().__init__(parent)

        self.validPattern = '^[a-zA-Z0-9_-]+$'
        self.setValidator(QRegExpValidator(QRegExp(self.validPattern)))

        # self.setAlignment(Qt.AlignCenter)

class NumericCommaLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.validPattern = '^[0-9,\.]+$'
        self.setValidator(QRegExpValidator(QRegExp(self.validPattern)))
    
    def values(self):
        try:
            vals = [float(c) for c in self.text().split(',')]
        except Exception as e:
            vals = []
        return vals

class mySpinBox(QSpinBox):
    sigTabEvent = pyqtSignal(object, object)

    def __init__(self, *args) -> None:
        super().__init__(*args)
    
    def event(self, event):
        if event.type()==QEvent.KeyPress and event.key() == Qt.Key_Tab:
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


class myMessageBox(QDialog):
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
        self.allowClose = allowClose

        self.showCentered = showCentered

        self.scrollableText = scrollableText

        self.layout = QGridLayout()
        self.layout.setHorizontalSpacing(20)
        self.buttonsLayout = QHBoxLayout()
        self.buttonsLayout.setSpacing(2)
        self.buttons = []
        self.widgets = []
        self.layouts = []
        self.labels = []
        self.detailsTextWidget = None
        self.showInFileManagButton = None
        self.visibleDetails = False
        self.doNotShowAgainCheckbox = None

        self.currentRow = 0
        self._w = None

        self.layout.setColumnStretch(1, 1)
        self.setLayout(self.layout)

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

        self.layout.addWidget(label, 0, 0, alignment=Qt.AlignTop)

    def addShowInFileManagerButton(self, path, txt=None):
        if txt is None:
            txt = 'Reveal in Finder...' if is_mac else 'Show in Explorer...'
        self.showInFileManagButton = showInFileManagerButton(txt)
        self.buttonsLayout.addWidget(self.showInFileManagButton)
        func = partial(myutils.showInExplorer, path)
        self.showInFileManagButton.clicked.connect(func)

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

    def addText(self, text):
        label = QLabel(self)
        label.setText(text)
        label.setWordWrap(self.wrapText)
        label.setOpenExternalLinks(True)
        self.labels.append(label)
        if self.scrollableText:
            textWidget = QScrollArea()
            textWidget.setFrameStyle(QFrame.NoFrame)
            textWidget.setWidget(label)
        else:
            textWidget = label

        self.layout.addWidget(textWidget, self.currentRow, 1)#, alignment=Qt.AlignTop)
        self.currentRow += 1
        return label

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
        self.layout.addWidget(widget, self.currentRow, 1)
        self.widgets.append(widget)
        self.currentRow += 1

    def addLayout(self, layout):
        self.layout.addLayout(layout, self.currentRow, 1)
        self.layouts.append(layout)
        self.currentRow += 1

    def setWidth(self, w):
        self._w = w

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        # spacer
        spacer = QSpacerItem(10, 10)
        self.layout.addItem(spacer, self.currentRow, 1)
        self.layout.setRowStretch(self.currentRow, 0)

        # buttons
        self.currentRow += 1

        if self.detailsTextWidget is not None:
            self.buttonsLayout.insertWidget(1, self.detailsButton)

        # Do not show again checkbox
        if self.doNotShowAgainCheckbox is not None:
            self.layout.addWidget(
                self.doNotShowAgainCheckbox, self.currentRow, 1, 1, 2
            )
            self.currentRow += 1

        self.layout.addLayout(
            self.buttonsLayout, self.currentRow, 0, 1, 2,
            alignment=Qt.AlignRight
        )

        # Details
        if self.detailsTextWidget is not None:
            self.currentRow += 1
            self.layout.addWidget(
                self.detailsTextWidget, self.currentRow, 0, 1, 2
            )

        # spacer
        self.currentRow += 1
        spacer = QSpacerItem(10, 10)
        self.layout.addItem(spacer, self.currentRow, 1)
        self.layout.setRowStretch(self.currentRow, 0)

        super().show()
        QTimer.singleShot(5, self._resize)

        if block:
            self._block()

    def setDetailedText(self, text, visible=False):
        self.detailsTextWidget = QPlainTextEdit(text)
        self.detailsTextWidget.setReadOnly(True)
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
            self.move(left, top)

        self._h = self.height()

        if self.okButton is not None:
            self.okButton.setFocus(True)

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
            buttonsTexts=None, layouts=None, widgets=None
        ):
        if parent is not None:
            self.setParent(parent)
        self.setWindowTitle(title)
        self.addText(message)
        if layouts is not None:
            if myutils.is_iterable(layouts):
                for layout in layouts:
                    self.addLayout(layout)
            else:
                self.addLayout(layout)

        if widgets is not None:
            if myutils.is_iterable(widgets):
                for widget in widgets:
                    self.addWidget(widget)
            else:
                self.addWidget(widgets)

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
            self.setDetailedText(detailsText, visible=True)
        return buttons

    def critical(
            self, parent, title, message,
            buttonsTexts=None, layouts=None, widgets=None,
            showDialog=True, detailsText=None,
        ):
        self.setIcon(iconName='SP_MessageBoxCritical')
        buttons = self._template(
            parent, title, message, detailsText=detailsText,
            buttonsTexts=buttonsTexts, layouts=layouts, widgets=widgets
        )
        if showDialog:
            self.exec_()
        return buttons

    def information(
            self, parent, title, message,
            buttonsTexts=None, layouts=None, widgets=None,
            showDialog=True, detailsText=None
        ):
        self.setIcon(iconName='SP_MessageBoxInformation')
        buttons = self._template(
            parent, title, message, detailsText=detailsText,
            buttonsTexts=buttonsTexts, layouts=layouts, widgets=widgets
        )
        if showDialog:
            self.exec_()
        return buttons

    def warning(
            self, parent, title, message,
            buttonsTexts=None, layouts=None, widgets=None,
            showDialog=True, detailsText=None,
        ):
        self.setIcon(iconName='SP_MessageBoxWarning')
        buttons = self._template(
            parent, title, message, detailsText=detailsText,
            buttonsTexts=buttonsTexts, layouts=layouts, widgets=widgets
        )
        if showDialog:
            self.exec_()
        return buttons

    def question(
            self, parent, title, message,
            buttonsTexts=None, layouts=None, widgets=None,
            showDialog=True, detailsText=None,
        ):
        self.setIcon(iconName='SP_MessageBoxQuestion')
        buttons = self._template(
            parent, title, message, detailsText=detailsText,
            buttonsTexts=buttonsTexts, layouts=layouts, widgets=widgets
        )
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
        if hasattr(self, 'loop'):
            self.loop.exit()

class myFormLayout(QGridLayout):
    def __init__(self):
        QGridLayout.__init__(self)

    def addFormWidget(self, formWidget, align=None, row=0):
        for col, item in enumerate(formWidget.items):
            if col==0:
                alignment = Qt.AlignRight
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
    sigIDchanged = pyqtSignal(int)
    sigDisableGhost = pyqtSignal()
    sigClearGhostContour = pyqtSignal()
    sigClearGhostMask = pyqtSignal()
    sigGhostOpacityChanged = pyqtSignal(int)

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

class rightClickToolButton(QToolButton):
    sigRightClick = pyqtSignal(object)

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
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(pen)
            p.setBrush(brush)
            p.drawPath(symbol)
        except Exception as e:
            traceback.print_exc()
        finally:
            p.end()

class PointsLayerToolButton(ToolButtonCustomColor):
    sigEditAppearance = pyqtSignal(object)

    def __init__(self, symbol, color='r', parent=None):
        super().__init__(symbol, color=color, parent=parent)
        self.sigRightClick.connect(self.showContextMenu)
    
    def showContextMenu(self, event):
        contextMenu = QMenu(self)
        contextMenu.addSeparator()

        editAction = QAction('Edit points appearance...')
        editAction.triggered.connect(self.editAppearance)
        contextMenu.addAction(editAction)

        contextMenu.exec(event.globalPos())
    
    def editAppearance(self):
        self.sigEditAppearance.emit(self)

class customAnnotToolButton(ToolButtonCustomColor):
    sigRemoveAction = pyqtSignal(object)
    sigKeepActiveAction = pyqtSignal(object)
    sigModifyAction = pyqtSignal(object)
    sigHideAction = pyqtSignal(object)

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
        animation_curve=QEasingCurve.InOutQuad
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

        if initial is not None:
            self.setChecked(initial)

    def sizeHint(self):
        return QSize(36, 18)

    def eventFilter(self, object, event):
        # To get the actual position of the circle we need to wait that
        # the widget is visible before setting the state
        if event.type() == QEvent.Show and self.requestedState is not None:
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

    @pyqtProperty(float)
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
        p.setRenderHint(QPainter.Antialiasing)

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

class ShortcutLineEdit(QLineEdit):
    def __init__(self, parent=None):
        self.keySequence = None
        super().__init__(parent)
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

        keySequence = QKeySequence(event.modifiers() | event.key()).toString()
        keySequence = keySequence.encode('ascii', 'ignore').decode('utf-8')
        self.setText(keySequence)
        self.key = event.key()
    
    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if self.text().endswith('+'):
            self.setText('')
            

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
    sigApplyButtonClicked = pyqtSignal(object)
    sigComputeButtonClicked = pyqtSignal(object)

    def __init__(
            self, widget,
            initialVal=None,
            stretchWidget=True,
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
            widgetLayout.addStretch(1)
            widgetLayout.addWidget(widget)
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
            item.setDisabled(disabled)

class ToggleTerminalButton(QPushButton):
    sigClicked = pyqtSignal(bool)

    def __init__(self, *args):
        super().__init__(*args)
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
        # pal.setColor(QPalette.Button, QColor(200, 200, 200))
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
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31-1)
        self.setStyleSheet('background-color: rgba(240, 240, 240, 200);')

class readOnlySpinbox(QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setReadOnly(True)
        self.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31-1)
        self.setStyleSheet('background-color: rgba(240, 240, 240, 200);')

class DoubleSpinBox(QDoubleSpinBox):
    sigValueChanged = pyqtSignal(int)

    def __init__(self, parent=None, disableKeyPress=False):
        super().__init__(parent=parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31-1)
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

class SpinBox(QSpinBox):
    sigValueChanged = pyqtSignal(int)
    sigUpClicked = pyqtSignal()
    sigDownClicked = pyqtSignal()

    def __init__(self, parent=None, disableKeyPress=False):
        super().__init__(parent=parent)
        self.setAlignment(Qt.AlignCenter)
        self.setMaximum(2**31-1)
        self._valueChangedFunction = None
        self.disableKeyPress = disableKeyPress
    
    def mousePressEvent(self, event) -> None:
        super().mousePressEvent(event)
        opt = QStyleOptionSpinBox()
        self.initStyleOption(opt)

        control = self.style().hitTestComplexControl(
            QStyle.CC_SpinBox, opt, event.pos(), self
        )
        if control == QStyle.SC_SpinBoxUp:
            self.sigUpClicked.emit()
        elif control == QStyle.SC_SpinBoxDown:
            self.sigDownClicked.emit()

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

class ReadOnlyLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setReadOnly(True)
        self.setStyleSheet(
            'background-color: rgba(240, 240, 240, 200);'
        )
        self.installEventFilter(self)
    
    def eventFilter(self, a0: 'QObject', a1: 'QEvent') -> bool:
        if a1.type() == QEvent.FocusIn:
            return True
        return super().eventFilter(a0, a1)
        
class _metricsQGBox(QGroupBox):
    sigDelClicked = pyqtSignal(str, object)

    def __init__(
            self, desc_dict, title, favourite_funcs=None, isZstack=False,
            equations=None, addDelButton=False, delButtonMetricsDesc=None
        ):
        QGroupBox.__init__(self)
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

        self.selectAllButton = QPushButton('Deselect all', self)
        self.selectAllButton.setCheckable(True)
        self.selectAllButton.setChecked(True)
        self.selectAllButton.clicked.connect(self.checkAll)
        buttonsLayout = QHBoxLayout()
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(self.selectAllButton)

        if favourite_funcs is not None:
            self.loadFavouritesButton = QPushButton(
                '  Load last selection...  ', self
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
        for checkBox in self.checkBoxes:
            checkBox.setChecked(False)
            for favourite_func in self.favourite_funcs:
                func_name = checkBox.text()
                if func_name.endswith(favourite_func):
                    checkBox.setChecked(True)
                    break
        self.doNotWarn = False

    def checkAll(self, isChecked):
        for checkBox in self.checkBoxes:
            checkBox.setChecked(isChecked)
        if isChecked:
            self.selectAllButton.setText('Deselect all')
        else:
            self.selectAllButton.setText('Select all')

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
    sigDelClicked = pyqtSignal(str, object)
    sigCheckboxToggled = pyqtSignal(object)

    def __init__(
            self, isZstack, chName, isSegm3D,
            posData=None, favourite_funcs=None
        ):
        QGroupBox.__init__(self)

        self.doNotWarn = False

        layout = QVBoxLayout()
        metrics_desc, bkgr_val_desc = measurements.standard_metrics_desc(
            isZstack, chName, isSegm3D=isSegm3D
        )

        metricsQGBox = _metricsQGBox(
            metrics_desc, 'Standard measurements',
            favourite_funcs=favourite_funcs
        )
        
        bkgrValsQGBox = _metricsQGBox(
            bkgr_val_desc, 'Background values',
            favourite_funcs=favourite_funcs
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
            The checkbox that has been toggled, by default None. If None 
            use `self.sender()`
        """        
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

class objPropsQGBox(QGroupBox):
    def __init__(self, parent=None):
        QGroupBox.__init__(self, 'Properties', parent)

        mainLayout = QGridLayout()

        row = 0
        label = QLabel('Object ID: ')
        self.idSB = QSpinBox()
        self.idSB.setMaximum(2**16)
        self.idSB.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.idSB.setAlignment(Qt.AlignCenter)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.idSB, row, 1)

        mainLayout.setColumnStretch(row, 3)

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
        self.cellAreaPxlSB = readOnlySpinbox()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellAreaPxlSB, row, 1)

        row += 1
        label = QLabel('Area (<span>&#181;</span>m<sup>2</sup>): ')
        self.cellAreaUm2DSB = readOnlyDoubleSpinbox()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellAreaUm2DSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        label = QLabel('Rotational volume (voxel): ')
        self.cellVolVoxSB = readOnlySpinbox()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolVoxSB, row, 1)

        row += 1
        label = QLabel('3D volume (voxel): ')
        self.cellVolVox3D_SB = readOnlySpinbox()
        self.cellVolVox3D_SB.label = label
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolVox3D_SB, row, 1)

        row += 1
        label = QLabel('Rotational volume (fl): ')
        self.cellVolFlDSB = readOnlyDoubleSpinbox()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolFlDSB, row, 1)

        row += 1
        label = QLabel('3D volume (fl): ')
        self.cellVolFl3D_DSB = readOnlyDoubleSpinbox()
        self.cellVolFl3D_DSB.label = label
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolFl3D_DSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        label = QLabel('Solidity: ')
        self.solidityDSB = readOnlyDoubleSpinbox()
        self.solidityDSB.setMaximum(1)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.solidityDSB, row, 1)

        row += 1
        label = QLabel('Elongation: ')
        self.elongationDSB = readOnlyDoubleSpinbox()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.elongationDSB, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

        row += 1
        propsNames = measurements.get_props_names()[1:]
        self.additionalPropsCombobox = QComboBox()
        self.additionalPropsCombobox.addItems(propsNames)
        self.additionalPropsCombobox.indicator = readOnlyDoubleSpinbox()
        mainLayout.addWidget(self.additionalPropsCombobox, row, 0)
        mainLayout.addWidget(self.additionalPropsCombobox.indicator, row, 1)

        row += 1
        mainLayout.addWidget(QHLine(), row, 0, 1, 2)

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
        self.minimumDSB = readOnlyDoubleSpinbox()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.minimumDSB, row, 1)

        row += 1
        label = QLabel('Maximum: ')
        self.maximumDSB = readOnlyDoubleSpinbox()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.maximumDSB, row, 1)

        row += 1
        label = QLabel('Mean: ')
        self.meanDSB = readOnlyDoubleSpinbox()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.meanDSB, row, 1)

        row += 1
        label = QLabel('Median: ')
        self.medianDSB = readOnlyDoubleSpinbox()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.medianDSB, row, 1)

        row += 1
        metricsDesc = measurements._get_metrics_names()
        metricsFunc, _ = measurements.standard_metrics_func()
        items = [metricsDesc[key] for key in metricsFunc.keys()]
        items.append('Concentration')
        items.sort()
        nameFuncDict = {}
        for name, desc in metricsDesc.items():
            if name not in metricsFunc.keys():
                continue
            nameFuncDict[desc] = metricsFunc[name]

        funcionCombobox = QComboBox()
        funcionCombobox.addItems(items)
        self.additionalMeasCombobox = funcionCombobox
        self.additionalMeasCombobox.indicator = readOnlyDoubleSpinbox()
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

        self.propsTab = QScrollArea(self)

        container = QWidget()
        layout = QVBoxLayout()

        self.propsQGBox = objPropsQGBox(parent=self.propsTab)
        self.intensMeasurQGBox = objIntesityMeasurQGBox(parent=self.propsTab)

        self.highlightCheckbox = QCheckBox('Highlight objects')
        self.highlightCheckbox.setChecked(True)

        layout.addWidget(self.highlightCheckbox)
        layout.addWidget(self.propsQGBox)
        layout.addWidget(self.intensMeasurQGBox)       
        container.setLayout(layout)

        self.propsTab.setWidget(container)
        self.addTab(self.propsTab, 'Measurements')

    def addChannels(self, channels):
        self.intensMeasurQGBox.addChannels(channels)

class expandCollapseButton(QPushButton):
    def __init__(self, parent=None):
        QPushButton.__init__(self, parent)
        self.setIcon(QIcon(":expand.svg"))
        self.setFlat(True)
        self.installEventFilter(self)
        self.isExpand = True
        self.clicked.connect(self.buttonClicked)

    def buttonClicked(self, checked=False):
        if self.isExpand:
            self.setIcon(QIcon(":collapse.svg"))
            self.isExpand = False
        else:
            self.setIcon(QIcon(":expand.svg"))
            self.isExpand = True

    def eventFilter(self, object, event):
        if event.type() == QEvent.HoverEnter:
            self.setFlat(False)
        elif event.type() == QEvent.HoverLeave:
            self.setFlat(True)
        return False

class view_visualcpp_screenshot(QWidget):
    def __init__(self, parent=None):
        super().__init__()
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

class BaseGradientEditorItemLabels(pg.GradientEditorItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
    
    def restoreState(self, state):
        pg.graphicsItems.GradientEditorItem.Gradients = GradientsLabels
        return super().restoreState(state)

class baseHistogramLUTitem(pg.HistogramLUTItem):
    sigAddColormap = pyqtSignal(object, str)

    def __init__(self, name='image', parent=None, **kwargs):
        pg.GradientEditorItem = BaseGradientEditorItemLabels

        super().__init__(**kwargs)

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

        # Add custom colormap action
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
        else:
            showIdx = 1
            hideIdx = 2
        
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
    
    def invertCurrentColormap(self, debug=False):
        self.setGradient(self.invertGradient(self.lastGradient))
    
    def addCustomGradient(self, gradient_name, gradient_ticks):
        self.originalLength = self.gradient.length
        self.gradient.length = 100
        self.gradient.restoreState(gradient_ticks)
        gradient = self.gradient.getGradient()
        action = CustomGradientMenuAction(gradient, gradient_name, self.gradient)
        action.triggered.connect(self.gradient.contextMenuClicked)
        action.delButton.clicked.connect(self.removeCustomGradient)
        self.gradient.menu.insertAction(self.saveColormapAction, action)
        self.gradient.length = self.originalLength
        GradientsImage[gradient_name] = gradient_ticks
    
    def removeCustomGradient(self):
        button = self.sender()
        action = button.action
        self.gradient.menu.removeAction(action)
        cp = config.ConfigParser()
        cp.read(custom_cmaps_filepath)
        cp.remove_section(f'labels.{action.name}')
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
        inputWin = QInput(parent=self._parent, title='Colormap name')
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

        gradient_ticks = []
        state = self.gradient.saveState()
        for key, value in state.items():
            if key != 'ticks':
                continue
            for t, tick in enumerate(value):
                pos, rgb = tick
                gradient_ticks.append((pos, rgb))
                rgb = ','.join([str(c) for c in rgb])
                val = f'{pos},{rgb}'
                cp[SECTION][f'tick_{t}_pos_rgb'] = val
        
        with open(custom_cmaps_filepath, mode='w') as file:
            cp.write(file)
        
        self.addCustomGradient(SECTION, gradient_ticks)
    
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

class ToggleVisibilityButton(QPushButton):
    def __init__(self, *args):
        super().__init__(*args)
        self.setFlat(True)
        self.setCheckable(True)
        self.toggled.connect(self.onToggled)
    
    def onToggled(self, checked):
        if checked:
            self.setIcon(':eye-checked.svg')
        else:
            self.setIcon('unchecked.svg')

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
    sigGradientMenuEvent = pyqtSignal(object)
    sigTickColorAccepted = pyqtSignal(object)

    def __init__(self, parent=None, name='image', isViewer=False, **kwargs):
        super().__init__(parent=parent, name=name, **kwargs)

        self.name = name
        self._parent = parent

        self.isViewer = isViewer
        if isViewer:
            # In the viewer we don't allow additional settings from the menu
            return

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
                    if isinstance(self.highlightedAction, highlightableQWidgetAction):
                        # print('Left a custom action')
                        pass
                self.highlightedAction = hoveredAction

            self.lastHoveredAction = hoveredAction
    
    def addOverlayColorButton(self, rgbColor):
        # Overlay color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Overlay color: '))
        self.overlayColorButton = myColorButton(color=rgbColor)
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

class labelledQScrollbar(QScrollBar):
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

class navigateQScrollBar(QScrollBar):
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
            self.triggerAction(QAbstractSlider.SliderSingleStepAdd)

class linkedQScrollbar(QScrollBar):
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
        p.setRenderHint(QPainter.Antialiasing)
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
    sigShowRightImgToggled = pyqtSignal(bool)
    sigShowLabelsImgToggled = pyqtSignal(bool)

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

        # Default settings
        self.defaultSettingsAction = QAction('Restore default settings...', self)
        self.menu.addAction(self.defaultSettingsAction)

        self.menu.addSeparator()

        self.showRightImgAction.toggled.connect(self.showRightImageToggled)
        self.showLabelsImgAction.toggled.connect(self.showLabelsImageToggled)
    
    def addCustomGradient(self, gradient_name, gradient_ticks):
        currentState = self.item.saveState()
        self.originalLength = self.item.length
        self.item.length = 100
        self.item.restoreState(gradient_ticks)
        gradient = self.item.getGradient()
        action = CustomGradientMenuAction(gradient, gradient_name, self.item)
        action.triggered.connect(self.item.contextMenuClicked)
        action.delButton.clicked.connect(self.removeCustomGradient)
        self.item.menu.insertAction(self.saveColormapAction, action)
        self.item.length = self.originalLength
        self.item.restoreState(currentState)
        GradientsLabels[gradient_name] = gradient_ticks
    
    def removeCustomGradient(self):
        button = self.sender()
        action = button.action
        self.item.menu.removeAction(action)
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
        inputWin = QInput(parent=self._parent, title='Colormap name')
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
        
        self.addCustomGradient(cmapName, state)

    def showRightImageToggled(self, checked):
        if checked and self.showLabelsImgAction.isChecked():
            # Hide the right labels image before showing right image
            self.showLabelsImgAction.setChecked(False)
            self.sigShowLabelsImgToggled.emit(False)
        self.sigShowRightImgToggled.emit(checked)
    
    def showLabelsImageToggled(self, checked):
        if checked and self.showRightImgAction.isChecked():
            # Hide the right image before showing labels image
            self.showRightImgAction.setChecked(False)
            self.sigShowRightImgToggled.emit(False)
        self.sigShowLabelsImgToggled.emit(checked)

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

class QProgressBarWithETA(QProgressBar):
    def __init__(self, parent=None):
        self.parent = parent
        super().__init__(parent)

        palette = QPalette()
        palette.setColor(QPalette.Highlight, QColor(207, 235, 155))
        palette.setColor(QPalette.Text, QColor(0, 0, 0))
        palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        self.setPalette(palette)
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
        self.disableAutoRange()
        self.autoBtn.mode = 'manual'
    
    def autoBtnClicked(self):
        self.vb.autoRange()
        self.autoBtn.hide()

class sliderWithSpinBox(QWidget):
    sigValueChange = pyqtSignal(object)
    valueChanged = pyqtSignal(object)
    editingFinished = pyqtSignal()

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
            self.spinBox = QDoubleSpinBox(self)
        else:
            self.spinBox = QSpinBox(self)
        self.spinBox.setAlignment(Qt.AlignCenter)
        self.spinBox.setMaximum(2**31-1)
        layout.addWidget(self.spinBox, row+1, col+1)
        if title is not None:
            layout.setRowStretch(0, 1)
        layout.setRowStretch(row+1, 1)
        layout.setColumnStretch(col, 6)
        layout.setColumnStretch(col+1, 1)

        self.layout = layout
        self.lastCol = col+1
        self.sliderCol = row+1

        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.slider.sliderReleased.connect(self.onEditingFinished)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.spinBox.editingFinished.connect(self.onEditingFinished)
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
    
    def setImage(self, image=None, **kwargs):
        if image is None:
            return
        autoLevels = kwargs.get('autoLevels')
        if autoLevels is None:
            kwargs['autoLevels'] = False
        super().setImage(image, **kwargs)

class ParentImageItem(pg.ImageItem):
    def __init__(
            self, image=None, linkedImageItem=None, activatingAction=None,
            debug=False, **kargs
        ):
        super().__init__(image, **kargs)
        self.linkedImageItem = linkedImageItem
        self.activatingAction = activatingAction
        self.debug = debug
    
    def clear(self):
        if self.linkedImageItem is not None:
            self.linkedImageItem.clear()
        return super().clear()
    
    def setLevels(self, levels, **kargs):
        if self.linkedImageItem is not None:
            self.linkedImageItem.setLevels(levels)
        return super().setLevels(levels, **kargs)
    
    def setImage(self, image=None, autoLevels=None, **kargs):
        if self.linkedImageItem is not None:
            if self.activatingAction.isChecked():
                self.linkedImageItem.setImage(image, autoLevels=autoLevels)
        return super().setImage(image, autoLevels, **kargs)
    
    def updateImage(self, *args, **kargs):
        if self.linkedImageItem is not None:
            if self.activatingAction.isChecked():
                self.linkedImageItem.image = self.image
                self.linkedImageItem.updateImage(*args, **kargs)
        return super().updateImage(*args, **kargs)
    
    def setOpacity(self, value):
        super().setOpacity(value)
        if self.linkedImageItem is not None:
            self.linkedImageItem.setOpacity(value)
    
    def setLookupTable(self, lut):
        super().setLookupTable(lut)
        if self.linkedImageItem is not None:
            self.linkedImageItem.setLookupTable(lut)

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
        self.layout.addWidget(self.checkbox, self.sliderCol, self.lastCol+1)
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
    def __init__(self,):
        super().__init__()
        # Yellow pen
        self.setPen(pg.mkPen(width=2, color=(245, 184, 0, 100)))
        self.label = myLabelItem()
        self.label.setAttr('bold', True)
        self.label.setAttr('color', (245, 184, 0))
    
    def addToPlotItem(self, PlotItem: MainPlotItem):
        self._plotItem = PlotItem
        PlotItem.addItem(self)
        PlotItem.addItem(self.label)
    
    def removeFromPlotItem(self):
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
            w, h = self.label.rect().right(), self.label.rect().bottom()
            self.label.setPos(x_cursor-w/2, y_cursor-h/2)
    
    def clear(self):
        self.setData()

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
        w, h = self.label.rect().right(), self.label.rect().bottom()
        self.label.setPos(x_cursor-w/2, y_cursor-h/2)
    
    def clear(self):
        if hasattr(self, 'label'):
            self.label.setText('')
        if self.image is None:
            return
        self.image[:] = 0
        self.updateImage()

class PostProcessSegmSpinbox(QWidget):
    valueChanged = pyqtSignal(int)
    editingFinished = pyqtSignal()
    sigCheckboxToggled = pyqtSignal()

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