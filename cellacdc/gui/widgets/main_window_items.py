"""Graphics items and window parts of the main GUI window only: the ghost
contour and mask of objects being moved (GhostContourItem,
GhostMaskItem), the brush circle (LabelRoiCircularItem, with the
clipSelemMask helper), the delete ROI (DelROI), ChildImageItem,
myLabelItem, GroupBox, the central widget (GuiCentralWidget),
FireworksOverlay and the log widget (XStream, QLog, QtHandler).
"""
import sys
import numpy as np
import math
import logging
import skimage.draw
import skimage.morphology
import skimage.segmentation
from qtpy.QtCore import Signal, QTimer, Qt, QPointF, QObject
from qtpy.QtGui import QColor, QPen, QPainter
from qtpy.QtWidgets import QWidget, QGroupBox, QPlainTextEdit
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import myutils
from cellacdc.shared.widgets import plot_items as widgets_plot_items
from math import ceil, floor


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


class GroupBox(QGroupBox):
    def __init__(self, *args, keyPressCallback=None):
        super().__init__(*args)
        self.keyPressCallback = keyPressCallback
        self.setFocusPolicy(Qt.NoFocus)
    
    def keyPressEvent(self, event) -> None:
        event.ignore()
        if self.keyPressCallback is None:
            return

        self.keyPressCallback()


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


class DelROI(pg.ROI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setCursor(Qt.SizeAllCursor)
    
    def clearPoints(self):
        """
        Remove all handles and segments.
        """
        while len(self.handles) > 0:
            self.removeHandle(self.handles[0]['item'])


class ChildImageItem(widgets_plot_items.BaseImageItem):
    def __init__(self, *args, linkedScrollbar=None, **kwargs):
        widgets_plot_items.BaseImageItem.__init__(self, *args, **kwargs)
        self.linkedScrollbar = linkedScrollbar
    
    def setImage(self, img=None, z=None, scrollbar_value=None, **kargs):
        autoLevels = kargs.get('autoLevels')
        if autoLevels is None:
            kargs['autoLevels'] = False

        if img is None:
            widgets_plot_items.BaseImageItem.setImage(self, img, **kargs)
            return

        if img.ndim == 3 and img.shape[-1] > 4 and z is not None:
            widgets_plot_items.BaseImageItem.setImage(self, img[z], **kargs)
        else:
            widgets_plot_items.BaseImageItem.setImage(self, img, **kargs)
        
        if self.linkedScrollbar is None:
            return
        
        if not self.linkedScrollbar.isEnabled():
            return
        
        if scrollbar_value is None:
            return
        
        self.linkedScrollbar.setValueNoSignal(scrollbar_value)


class GhostContourItem(pg.PlotDataItem):
    def __init__(
            self, ParentPlotItem, penColor=(245, 184, 0, 100), 
            textColor=(245, 184, 0)
        ):
        super().__init__()
        # Yellow pen
        self.setPen(pg.mkPen(width=2, color=penColor))
        self.label = myLabelItem()
        self.label.setAttr('bold', True)
        self.label.setAttr('color', textColor)
        self._ParentPlotItem = ParentPlotItem
    
    def addToPlotItem(self):
        self._ParentPlotItem.addItem(self)
        self._ParentPlotItem.addItem(self.label)
    
    def removeFromPlotItem(self):
        self._ParentPlotItem.removeItem(self.label)
        self._ParentPlotItem.removeItem(self)
    
    def setData(
            self, xx=None, yy=None, fontSize=11, ID=0,
            y_cursor=None, x_cursor=None
        ):
        if xx is None:
            xx = []
        if yy is None:
            yy = []
        self._is_empty = len(xx) == 0 and ID == 0
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
        if getattr(self, '_is_empty', False):
            # Already empty (called at every mouse move)
            return
        self.setData([], [])

class GhostMaskItem(pg.ImageItem):
    def __init__(self, ParentPlotItem):
        super().__init__()
        self.label = myLabelItem()
        self.label.setAttr('bold', True)
        self.label.setAttr('color', (245, 184, 0))
        self._ParentPlotItem = ParentPlotItem
    
    def initImage(self, imgShape):
        image = np.zeros(imgShape, dtype=np.uint32)
        self.setImage(image)
        self._is_cleared = True
    
    def initLookupTable(self, rgbaColor):
        lut = np.zeros((2, 4), dtype=np.uint8)
        lut[1,-1] = 255
        lut[1,:-1] = rgbaColor
        self.setLookupTable(lut)
    
    def addToPlotItem(self):
        self._ParentPlotItem.addItem(self)
        self._ParentPlotItem.addItem(self.label)
    
    def removeFromPlotItem(self):
        self._ParentPlotItem.removeItem(self.label)
        self._ParentPlotItem.removeItem(self)
    
    def updateGhostImage(self, ID=0, y_cursor=None, x_cursor=None, fontSize=None):
        self.setImage(self.image)
        self._is_cleared = False

        if ID == 0:
            self.label.setText('')
            return
        
        self.label.setText(f'{ID}', size=fontSize)
        w, h = self.label.itemRect().width(), self.label.itemRect().height()
        self.label.item.setPos(x_cursor, y_cursor-h)
    
    def clear(self):
        if getattr(self, '_is_cleared', False):
            # Already cleared (called at every mouse move). Zeroing the
            # image and setting it again would redraw the whole image
            return
        if hasattr(self, 'label'):
            self.label.setText('')
        if self.image is None:
            return
        self.image[:] = 0
        self.setImage(self.image)
        self._is_cleared = True


class GuiCentralWidget(QWidget):
    def __init__(self, parent=None, *args):
        super().__init__(parent, *args)


class FireworksOverlay(QWidget):
    COLORS = (
        (255, 80, 80), (255, 205, 70), (80, 220, 255),
        (110, 255, 130), (255, 110, 220), (245, 245, 255),
    )

    def __init__(self, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.hide()
        self.particles = []
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._advance)

    def start(self):
        self.setGeometry(self.parentWidget().rect())
        self.particles.clear()
        rng = np.random.default_rng()
        width = max(1, self.width())
        height = max(1, self.height())
        for burst_i in range(6):
            center_x = rng.uniform(width * 0.12, width * 0.88)
            center_y = rng.uniform(height * 0.12, height * 0.65)
            color = self.COLORS[burst_i % len(self.COLORS)]
            delay = burst_i * 0.12
            for angle in np.linspace(0, 2 * np.pi, 32, endpoint=False):
                speed = rng.uniform(90, 230)
                self.particles.append({
                    'x': center_x,
                    'y': center_y,
                    'vx': math.cos(angle) * speed,
                    'vy': math.sin(angle) * speed,
                    'age': -delay,
                    'life': rng.uniform(0.9, 1.5),
                    'color': color,
                    'size': rng.uniform(1.5, 3.5),
                })
        self.show()
        self.raise_()
        self.timer.start()

    def _advance(self):
        delta_time = self.timer.interval() / 1000
        active = False
        for particle in self.particles:
            particle['age'] += delta_time
            if particle['age'] < 0:
                active = True
                continue
            if particle['age'] >= particle['life']:
                continue
            active = True
            particle['x'] += particle['vx'] * delta_time
            particle['y'] += particle['vy'] * delta_time
            particle['vy'] += 150 * delta_time
            particle['vx'] *= 0.985
        if not active:
            self.timer.stop()
            self.hide()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for particle in self.particles:
            age = particle['age']
            if age < 0 or age >= particle['life']:
                continue
            opacity = int(255 * (1 - age / particle['life']))
            color = QColor(*particle['color'], opacity)
            painter.setPen(QPen(color, particle['size']))
            tail_scale = 0.035
            painter.drawLine(
                QPointF(particle['x'], particle['y']),
                QPointF(
                    particle['x'] - particle['vx'] * tail_scale,
                    particle['y'] - particle['vy'] * tail_scale,
                ),
            )


def clipSelemMask(mask, shape, Yc, Xc, copy=True):
    if copy:
        mask = mask.copy()
    
    Y, X = shape
    h, w = mask.shape

    # Bottom, Left, Top, Right global coordinates of mask
    Y0, X0, Y1, X1 = Yc-(h/2), Xc-(w/2), Yc+(h/2), Xc+(w/2)
    mask_limits = [floor(Y0)+1, floor(X0)+1, floor(Y1)+1, floor(X1)+1]
    
    if Y0>=0 and X0>=0 and Y1<=Y and X1<=X:
        # Mask is withing shape boundaries, no need to clip
        ystart, xstart, yend, xend = mask_limits
        mask_slice = slice(ystart, yend), slice(xstart, xend)
        return mask, mask_slice

    if Y0<0:
        # Mask is exceeding at the bottom
        ystart = floor(abs(Y0))
        mask_limits[0] = 0
        mask = mask[ystart:]
    if X0<0:
        # Mask is exceeding at the left
        xstart = floor(abs(X0))
        mask_limits[1] = 0
        mask = mask[:, xstart:]
    if Y1>Y:
        # Mask is exceeding at the top
        yend = ceil(abs(Y1)) - Y
        mask_limits[2] = Y
        mask = mask[:-yend]
    if X1>X:
        # Mask is exceeding at the right
        xend = ceil(abs(X1)) - X
        mask_limits[3] = X
        mask = mask[:, :-xend]
    
    ystart, xstart, yend, xend = mask_limits
    mask_slice = slice(ystart, yend), slice(xstart, xend)
    return mask, mask_slice
