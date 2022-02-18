import sys
import time
import re

from PyQt5.QtCore import (
    pyqtSignal, QTimer, Qt, QPoint, pyqtSlot, pyqtProperty,
    QPropertyAnimation, QEasingCurve, QSequentialAnimationGroup,
    QSize, QRectF, QPointF, QRect, QPoint, QEasingCurve, QRegExp,
    QEvent
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
    QAction
)

import pyqtgraph as pg

from . import myutils, apps
from . import qrc_resources

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

class myColorButton(pg.ColorButton):
    sigColorRejected = pyqtSignal(object)

    def __init__(self, parent=None, color=(128,128,128), padding=6):
        pg.ColorButton.__init__(
            self, parent=parent, color=color, padding=padding
        )

    def colorRejected(self):
        self.setColor(self.origColor, finished=False)
        self.sigColorRejected.emit(self)

class myGradientWidget(pg.GradientWidget):
    def __init__(self, parent=None, orientation='right',  *args, **kargs):
        self.removeHSVcmaps()
        self.addGradients()

        pg.GradientWidget.__init__(
            self, parent=parent, orientation=orientation,  *args, **kargs
        )

        for action in self.menu.actions():
            if action.text() == 'HSV':
                HSV_action = action
                break
        self.menu.removeAction(HSV_action)

        # Background color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Background color: '))
        self.colorButton = pg.ColorButton(color=(25,25,25))
        hbox.addWidget(self.colorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = QWidgetAction(self)
        act.setDefaultWidget(widget)
        self.menu.insertAction(self.item.rgbAction, act)

        # IDs color button
        hbox = QHBoxLayout()
        hbox.addWidget(QLabel('Text color: '))
        self.textColorButton = pg.ColorButton(color=(25,25,25))
        hbox.addWidget(self.textColorButton)
        widget = QWidget()
        widget.setLayout(hbox)
        act = QWidgetAction(self)
        act.setDefaultWidget(widget)
        self.menu.insertAction(self.item.rgbAction, act)

        # Shuffle colors action
        self.shuffleCmapAction =  QAction(
            'Shuffle colormap...   (Shift+S)', self
        )
        self.menu.insertAction(self.item.rgbAction, self.shuffleCmapAction)
        self.menu.insertSeparator(self.shuffleCmapAction)

        # Invert bw action
        self.invertBwAction = QAction('Invert black/white', self)
        self.invertBwAction.setCheckable(True)
        self.menu.insertAction(self.item.rgbAction, self.invertBwAction)

        self.menu.insertSeparator(self.item.rgbAction)

    def removeHSVcmaps(self):
        hsv_cmaps = []
        for g, grad in pg.graphicsItems.GradientEditorItem.Gradients.items():
            if grad['mode'] == 'hsv':
                hsv_cmaps.append(g)
        for g in hsv_cmaps:
            del pg.graphicsItems.GradientEditorItem.Gradients[g]

    def addGradients(self):
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

    def restoreState(self, df):
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

        state = {'mode': 'rgb', 'ticksVisible': True, 'ticks': []}
        ticks_pos = {}
        ticks_rgb = {}
        stateFound = False
        for setting, value in df.itertuples():
            idx = setting.find('lab_cmap_')
            if idx == -1:
                continue

            stateFound = True
            m = re.findall('tick(\d+)_(\w+)', setting)
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
        font.setPointSize(9)
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
