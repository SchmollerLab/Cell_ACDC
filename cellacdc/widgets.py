import sys
import time
import re
import numpy as np

from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from PyQt5.QtCore import (
    pyqtSignal, QTimer, Qt, QPoint, pyqtSlot, pyqtProperty,
    QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup,
    QSize, QRectF, QPointF, QRect, QPoint, QEasingCurve, QRegExp,
    QEvent, QEventLoop
)
from PyQt5.QtGui import (
    QFont, QPalette, QColor, QPen, QPaintEvent, QBrush, QPainter,
    QRegExpValidator, QIcon, QPixmap
)
from PyQt5.QtWidgets import (
    QTextEdit, QLabel, QProgressBar, QHBoxLayout, QToolButton, QCheckBox,
    QApplication, QWidget, QVBoxLayout, QMainWindow, QStyleFactory,
    QLineEdit, QSlider, QSpinBox, QGridLayout, QDockWidget,
    QScrollArea, QSizePolicy, QComboBox, QPushButton, QScrollBar,
    QGroupBox, QAbstractSlider, QDoubleSpinBox, QWidgetAction,
    QAction, QTabWidget, QAbstractSpinBox, QMessageBox,
    QStyle, QDialog, QSpacerItem, QFrame, QMenu, QActionGroup
)

import pyqtgraph as pg
from pyqtgraph import QtGui

from . import myutils, apps
from . import qrc_resources

def removeHSVcmaps():
    hsv_cmaps = []
    for g, grad in pg.graphicsItems.GradientEditorItem.Gradients.items():
        if grad['mode'] == 'hsv':
            hsv_cmaps.append(g)
    for g in hsv_cmaps:
        del pg.graphicsItems.GradientEditorItem.Gradients[g]

def renamePgCmaps():
    Gradients = pg.graphicsItems.GradientEditorItem.Gradients
    Gradients['hot'] = Gradients.pop('thermal')
    Gradients.pop('greyclip')

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
    cmaps = {}
    for name, gradient in Gradients.items():
        ticks = gradient['ticks']
        colors = [tuple([v/255 for v in tick[1]]) for tick in ticks]
        cmaps[name] = LinearSegmentedColormap.from_list(name, colors, N=256)
    return cmaps

renamePgCmaps()
removeHSVcmaps()
cmaps = addGradients()

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

class myMessageBox(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.layout = QGridLayout()
        self.layout.setHorizontalSpacing(20)
        self.buttonsLayout = QHBoxLayout()
        self.buttonsLayout.setSpacing(2)
        self.buttons = []

        self.currentRow = 0

        self.layout.setColumnStretch(1, 1)
        self.setLayout(self.layout)

    def setIcon(self, iconName='SP_MessageBoxInformation'):
        label = QLabel(self)

        standardIcon = getattr(QStyle, iconName)
        icon = self.style().standardIcon(standardIcon)
        pixmap = icon.pixmap(60, 60)
        label.setPixmap(pixmap)

        self.layout.addWidget(label, 0, 0, alignment=Qt.AlignTop)

    def addText(self, text):
        label = QLabel(self)
        label.setText(text)
        label.setWordWrap(True)
        label.setOpenExternalLinks(True)
        self.layout.addWidget(label, self.currentRow, 1, alignment=Qt.AlignTop)
        self.currentRow += 1
        return label

    def addButton(self, buttonText):
        button = QPushButton(buttonText, self)
        self.buttonsLayout.addWidget(button)
        button.clicked.connect(self.close)
        self.buttons.append(button)
        return button

    def show(self, block=False):
        self.setWindowFlags(Qt.Dialog | Qt.WindowStaysOnTopHint)
        # spacer
        spacer = QSpacerItem(10, 10)
        self.layout.addItem(spacer, self.currentRow, 1)
        self.layout.setRowStretch(self.currentRow, 0)

        # buttons
        self.currentRow += 1
        self.layout.addLayout(
            self.buttonsLayout, self.currentRow, 0, 1, 2,
            alignment=Qt.AlignRight
        )

        # spacer
        self.currentRow += 1
        spacer = QSpacerItem(10, 10)
        self.layout.addItem(spacer, self.currentRow, 1)
        self.layout.setRowStretch(self.currentRow, 0)

        super().show()
        widths = [button.width() for button in self.buttons]
        if widths:
            max_width = max(widths)
            for button in self.buttons:
                button.setMinimumWidth(max_width)

        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    # def resizeEvent(self, event):
    #     print(self.layout.itemAtPosition(0, 1).widget().sizeHint())
    #     print(self.size())

    def exec_(self):
        self.show()
        super().exec_()

    def close(self):
        self.clickedButton = self.sender()
        super().close()
        if hasattr(self, 'loop'):
            self.loop.exit()

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

class objPropsQGBox(QGroupBox):
    def __init__(self, *args):
        QGroupBox.__init__(self, *args)

        mainLayout = QGridLayout()

        row = 0
        label = QLabel('Object ID: ')
        self.idSB = QSpinBox()
        self.idSB.setMaximum(2**16)
        self.idSB.setButtonSymbols(QAbstractSpinBox.NoButtons)
        self.idSB.setAlignment(Qt.AlignCenter)
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.idSB, row, 1)

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
        label = QLabel('Volume (voxel): ')
        self.cellVolVoxSB = readOnlySpinbox()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolVoxSB, row, 1)

        row += 1
        label = QLabel('Volume (fl): ')
        self.cellVolFlDSB = readOnlyDoubleSpinbox()
        mainLayout.addWidget(label, row, 0)
        mainLayout.addWidget(self.cellVolFlDSB, row, 1)

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

        mainLayout.setColumnStretch(1, 3)
        self.setLayout(mainLayout)

class guiTabControl(QTabWidget):
    def __init__(self, *args):
        super().__init__(args[0])

        self.propsTab = QScrollArea(self)

        container = QWidget()
        layout = QVBoxLayout()

        self.propsQGBox = objPropsQGBox(self.propsTab)

        self.highlightCheckbox = QCheckBox('Highlight objects')
        self.highlightCheckbox.setChecked(True)

        layout.addWidget(self.propsQGBox)
        layout.addWidget(self.highlightCheckbox)
        container.setLayout(layout)

        self.propsTab.setWidget(container)
        self.addTab(self.propsTab, 'Object properties')

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

class myHistogramLUTitem(pg.HistogramLUTItem):
    sigGradientMenuEvent = pyqtSignal(object)

    def __init__(self, **kwargs):
        self.cmaps = cmaps

        super().__init__(**kwargs)

        for action in self.gradient.menu.actions():
            if action.text() == 'HSV':
                HSV_action = action
            elif action.text() == 'RGB':
                RGB_ation = action
        self.gradient.menu.removeAction(HSV_action)
        self.gradient.menu.removeAction(RGB_ation)

        # Invert bw action
        self.invertBwAction = QAction('Invert black/white', self)
        self.invertBwAction.setCheckable(True)
        self.gradient.menu.addAction(self.invertBwAction)
        self.gradient.menu.addSeparator()

        # Contours color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Contours color: '))
        self.contoursColorButton = pg.ColorButton(color=(25,25,25))
        hbox.addWidget(self.contoursColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = QWidgetAction(self)
        act.setDefaultWidget(widget)
        self.gradient.menu.addAction(act)

        # Contours line weight
        contLineWeightMenu = QMenu('Contours line weight', self.gradient.menu)
        self.contLineWightActionGroup = QActionGroup(self)
        for w in range(1, 11):
            action = contLineWeightMenu.addAction(str(w))
            action.setCheckable(True)
            if w == 2:
                action.setChecked(True)
            action.lineWeight = w
            self.contLineWightActionGroup.addAction(action)
        self.gradient.menu.addMenu(contLineWeightMenu)

        self.labelsAlphaMenu = self.gradient.menu.addMenu(
            'Segm. masks overlay alpha...'
        )
        self.labelsAlphaMenu.setDisabled(True)
        hbox = QHBoxLayout()
        self.labelsAlphaSlider = sliderWithSpinBox(
            title='Alpha', title_loc='in_line', is_float=True,
            normalize=True
        )
        self.labelsAlphaSlider.setMaximum(100)
        self.labelsAlphaSlider.setValue(0.3)
        hbox.addWidget(self.labelsAlphaSlider)
        hbox.addWidget(QLabel('(Ctrl + Up/Down)'))
        widget = QWidget()
        widget.setLayout(hbox)
        act = QWidgetAction(self)
        act.setDefaultWidget(widget)
        self.labelsAlphaMenu.addSeparator()
        self.labelsAlphaMenu.addAction(act)

        # Default settings
        self.defaultSettingsAction = QAction('Restore default settings...', self)
        self.gradient.menu.addAction(self.defaultSettingsAction)

        # Select channels section
        self.gradient.menu.addSeparator()
        self.gradient.menu.addSection('Select channel: ')

        # hide histogram tool
        self.vb.hide()

    def restoreState(self, df):
        if 'contLineColor' in df.index:
            rgba_str = df.at['contLineColor', 'value']
            rgb = myutils.rgba_str_to_values(rgba_str)[:3]
            self.contoursColorButton.setColor(rgb)

        if 'contLineWeight' in df.index:
            w = df.at['contLineWeight', 'value']
            w = int(w)
            for action in self.contLineWightActionGroup.actions():
                if action.lineWeight == w:
                    action.setChecked(True)
                    break

        if 'overlaySegmMasksAlpha' in df.index:
            alpha = df.at['overlaySegmMasksAlpha', 'value']
            self.labelsAlphaSlider.setValue(float(alpha))

        checked = df.at['is_bw_inverted', 'value'] == 'Yes'
        self.invertBwAction.setChecked(checked)

class myColorButton(pg.ColorButton):
    sigColorRejected = pyqtSignal(object)

    def __init__(self, parent=None, color=(128,128,128), padding=6):
        pg.ColorButton.__init__(
            self, parent=parent, color=color, padding=padding
        )

    def colorRejected(self):
        self.setColor(self.origColor, finished=False)
        self.sigColorRejected.emit(self)

class labelsGradientWidget(pg.GradientWidget):
    def __init__(self, parent=None, orientation='right',  *args, **kargs):
        pg.GradientWidget.__init__(
            self, parent=parent, orientation=orientation,  *args, **kargs
        )

        for action in self.menu.actions():
            if action.text() == 'HSV':
                HSV_action = action
            elif action.text() == 'RGB':
                RGB_ation = action
        self.menu.removeAction(HSV_action)
        self.menu.removeAction(RGB_ation)

        # Background color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Background color: '))
        self.colorButton = pg.ColorButton(color=(25,25,25))
        hbox.addWidget(self.colorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = QWidgetAction(self)
        act.setDefaultWidget(widget)
        self.menu.addAction(act)

        # IDs color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Text color: '))
        self.textColorButton = pg.ColorButton(color=(25,25,25))
        hbox.addWidget(self.textColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = QWidgetAction(self)
        act.setDefaultWidget(widget)
        self.menu.addAction(act)

        # editFontSizeAction action
        self.editFontSizeAction =  QAction(
            'Text font size...', self
        )
        self.menu.addAction(self.editFontSizeAction)
        self.menu.addSeparator()

        # Shuffle colors action
        self.shuffleCmapAction =  QAction(
            'Shuffle colormap...   (Shift+S)', self
        )
        self.menu.addAction(self.shuffleCmapAction)

        # Invert bw action
        self.invertBwAction = QAction('Invert black/white', self)
        self.invertBwAction.setCheckable(True)
        self.menu.addAction(self.invertBwAction)

        # hide labels action
        self.hideLabelsImgAction = QAction('Hide segmentation image', self)
        self.hideLabelsImgAction.setCheckable(True)
        self.menu.addAction(self.hideLabelsImgAction)

        # Default settings
        self.defaultSettingsAction = QAction('Restore default settings...', self)
        self.menu.addAction(self.defaultSettingsAction)

        self.menu.addSeparator()

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
            r, g, b = myutils.rgb_str_to_values(rgbString)
            self.colorButton.setColor((r, g, b))

        if 'labels_text_color' in df.index:
            rgbString = df.at['labels_text_color', 'value']
            r, g, b = myutils.rgb_str_to_values(rgbString)
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
                    ticks_rgb[int(tick_idx)] = myutils.rgba_str_to_values(value)
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
        self.last_time_update = time.perf_counter()

    def update(self, step):
        t = time.perf_counter()
        self.setValue(self.value()+step)
        elpased_seconds = (t - self.last_time_update)/step
        steps_left = self.maximum() - self.value()
        seconds_left = elpased_seconds*steps_left
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

class sliderWithSpinBox(QWidget):
    sigValueChange = pyqtSignal(object)
    valueChanged = pyqtSignal(object)
    editingFinished = pyqtSignal()

    def __init__(self, *args, **kwargs):
        QWidget.__init__(self, *args)


        layout = QGridLayout()

        title = kwargs.get('title')
        row = 0
        col = 0
        if title is not None:
            titleLabel = QLabel(self)
            titleLabel.setText(title)
            loc = kwargs.get('title_loc')
            loc = loc if loc is not None else 'top'
            if loc == 'top':
                layout.addWidget(titleLabel, 1, col, alignment=Qt.AlignLeft)
            elif loc=='in_line':
                row = -1
                col = 1
                layout.addWidget(titleLabel, 0, 0, alignment=Qt.AlignLeft)
                layout.setColumnStretch(0, 0)

        self._normalize = False
        normalize = kwargs.get('normalize')
        if normalize is not None:
            self._normalize = True

        self._isFloat = False
        isFloat = kwargs.get('isFloat')
        if isFloat is not None:
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

        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.slider.sliderReleased.connect(self.onEditingFinished)
        self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.spinBox.editingFinished.connect(self.onEditingFinished)
        self.setLayout(layout)

    def onEditingFinished(self):
        self.editingFinished.emit()

    def maximum(self):
        return self.slider.maximum()

    def setValue(self, value):
        valueInt = value
        if self._normalize:
            valueInt = int(value*self.slider.maximum())
        if self._isFloat:
            valueInt = int(value)
            self.spinBox.valueChanged.disconnect()
            self.spinBox.setValue(value)
            self.spinBox.valueChanged.connect(self.spinboxValueChanged)
        self.slider.setValue(valueInt)

    def setMaximum(self, max):
        self.slider.setMaximum(max)
        # self.spinBox.setMaximum(max)

    def setMinimum(self, min):
        self.slider.setMinimum(min)
        # self.spinBox.setMinimum(min)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)

    def setDecimals(self, decimals):
        self.spinBox.setDecimals(decimals)

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
        self.slider.valueChanged.disconnect()
        if self._normalize:
            val = int(val*self.slider.maximum())
        if self._isFloat:
            val = int(val)
        self.slider.setValue(val)
        self.slider.valueChanged.connect(self.sliderValueChanged)
        self.sigValueChange.emit(self.value())
        self.valueChanged.emit(self.value())

    def value(self):
        return self.spinBox.value()

if __name__ == '__main__':
    class Window(QMainWindow):
        def __init__(self):
            super().__init__()

            container = QWidget()
            layout = QVBoxLayout()

            # slider = sliderWithSpinBox(isFloat=True)
            # slider.setMaximum(10)
            # slider.setValue(3.2)
            # slider.setSingleStep(0.1)
            # slider.valueChanged.connect(self.sliderValueChanged)
            # slider.slider.sliderReleased.connect(self.sliderReleased)
            # layout.addWidget(slider)
            # self.slider = slider

            okButton = QPushButton('ok')
            layout.addWidget(okButton)
            okButton.clicked.connect(self.okClicked)

            # layout.addStretch(1)
            container.setLayout(layout)
            self.setCentralWidget(container)

            self.setFocus()

        def okClicked(self, checked):
            editID = apps.editID_QWidget(19, [19, 100, 50])
            editID.exec_()
            print('closed')

        def sliderValueChanged(self, value):
            print(value)

        def sliderReleased(self, value):
            print('released')

        def keyPressEvent(self, event):
            if event.key() == Qt.Key_T:
                screens = app.screens()
                current_screen = self.screen()
                num_screens = len(screens)
                if num_screens > 1:
                    other_screen = None
                    for screen in screens:
                        if screen != current_screen:
                            other_screen = screen
                            break
                    print(f'Current screen geometry = {current_screen.geometry()}')
                    print(f'Other screen geometry = {other_screen.geometry()}')
            elif event.key() == Qt.Key_P:
                print(self.slider.value())



    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create('Fusion'))

    w = Window()
    w.show()
    app.exec_()
