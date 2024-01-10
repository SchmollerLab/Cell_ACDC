import os
import sys
import time
from tkinter import Y

import numpy as np
import pandas as pd

from qtpy.QtWidgets import QMainWindow, QFrame
from qtpy.QtCore import Qt, QPoint, QRect, QThread
from qtpy.QtGui import QBrush, QColor, QPen, QPainter

from .. import settings_csv_path
from .. import user_data_folderpath
from .. import workers

class ScreenRecorderFrame(QFrame):
    def __init__(self, app, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self._parent = parent
        # Border tolerance to trigger resizing
        self.px = 10
        self.app = app
    
    def mousePressEvent(self, event):
        x, y = event.pos().x(), event.pos().y()
        # x00, y00 = self._parent.x0-self.px, self._parent.y0-self.px
        x01, y01 = self._parent.x0+self.px, self._parent.y0+self.px
        x10, y10 = self._parent.x1-self.px, self._parent.y1-self.px
        # x11, y11 = self._parent.x1+self.px, self._parent.y1+self.px
        if y<y10 and y>y01 and x<x10 and x>x01:
            # Cursor click inside rectangle
            self.app.setOverrideCursor(Qt.ClosedHandCursor)
            self.xc, self.yc = x, y

    def mouseMoveEvent(self, event):
        if event.buttons():
            # Allow mouse move event in main window
            event.ignore()
            return

        x, y = event.pos().x(), event.pos().y()
        x00, y00 = self._parent.x0-self.px, self._parent.y0-self.px
        x01, y01 = self._parent.x0+self.px, self._parent.y0+self.px
        x10, y10 = self._parent.x1-self.px, self._parent.y1-self.px
        x11, y11 = self._parent.x1+self.px, self._parent.y1+self.px
        if y<y10 and y>y01 and x<x10 and x>x01:
            # Cursor inside rectangle
            self.app.setOverrideCursor(Qt.OpenHandCursor)
        elif y<y11 and y>y00 and x<x11 and x>x00:
            # Cursor on border --> determine if ver, hor or diags
            if x<x01 and y<y01:
                # Top left corner
                self.app.setOverrideCursor(Qt.SizeFDiagCursor)
                self.corner = 'topLeft'
            elif x<x01 and y>y10:
                # Bottom left corner
                self.app.setOverrideCursor(Qt.SizeBDiagCursor)
                self.corner = 'bottomLeft'
            elif x>x10 and y<y01:
                # Top right corner
                self.app.setOverrideCursor(Qt.SizeBDiagCursor)
                self.corner = 'topRight'
            elif x>x10 and y>y10:
                # Bottom right corner
                self.app.setOverrideCursor(Qt.SizeFDiagCursor)
                self.corner = 'bottomRight'
            elif x<x01 or x>x10:
                # Left or right side
                self.app.setOverrideCursor(Qt.SizeHorCursor)
                if x<x01:
                    self.corner = 'left'
                else:
                    self.corner = 'right'
            else:
                # Top or bottom side
                self.app.setOverrideCursor(Qt.SizeVerCursor)
                if y<y01:
                    self.corner = 'top'
                else:
                    self.corner = 'bottom'
        elif not event.buttons():
            # Cursor outside rectangle
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()

class ScreenRecorderWindow(QMainWindow):
    def __init__(self, app, parent, parentName):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint
        )
        self.app = app
        self.parentWin = parent
        self.parentWinName = parentName

        self.topLeft_points = []
        self.xymax = None
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        frame = ScreenRecorderFrame(app, parent=self)
        frame.setMouseTracking(True)
        frame.setStyleSheet("background-color: rgba(255, 255, 255, 0)")
        self.setCentralWidget(frame)
        self.frame = frame

    def loadLastRect(self):
        if not os.path.exists(settings_csv_path):
            self.x0, self.y0, self.x1, self.y1 = 100, 100, 400, 300
            return

        idx = f'screenRecorder_rect_{self.parentWinName}'
        self.df_settings = pd.read_csv(
            settings_csv_path, index_col='setting'
        )
        if idx in self.df_settings.index:
            s = self.df_settings.at[idx, 'value']
            coords = [int(d) for d in s.split(',')]
            self.x0, self.y0, self.x1, self.y1 = coords
        else:
            self.x0, self.y0, self.x1, self.y1 = (
                self.xmin, self.ymin, self.xmax, self.ymax
            )

    def getRectPoints(self):
        x1, y1 = self.x1, self.y1
        x0, y0 = self.x0, self.y0
        yy = [0, y0, y1, self.ymax]
        xx = [0, x0, x1, self.xmax]
        self.topLeft_points = []
        self.bottomRight_points = []
        for i, y in enumerate(yy[:3]):
            for j, x in enumerate(xx[:3]):
                self.topLeft_points.append(QPoint(x, y))
                self.bottomRight_points.append(QPoint(xx[j+1], yy[i+1]))

    def paintEvent(self, event):
        if self.xymax is None:
            return

        self.getRectPoints()

        qp = QPainter(self)
        br = QBrush(QColor(20, 20, 20, 140))
        qp.setBrush(br)
        qp.setPen(Qt.NoPen)

        if not self.topLeft_points:
            rect = QRect(self.xymin, self.xymax)
            qp.drawRect(rect)
            return

        iter = enumerate(zip(self.topLeft_points, self.bottomRight_points))
        spotlightWindow = None
        for r, (topLeft, bottomRight) in iter:
            if r == 4:
                spotlightWindow = (topLeft, bottomRight)
                continue
            qp.drawRect(QRect(topLeft, bottomRight))

        if spotlightWindow is not None:
            br = QBrush(QColor(20, 20, 20, 0))
            qp.setBrush(br)
            pen = QPen(QColor(255, 255, 255, 150))
            pen.setWidth(0)
            print(pen.width())
            pen.setStyle(Qt.DashLine)
            qp.setPen(pen)
            qp.drawRect(QRect(*spotlightWindow))

    def mousePressEvent(self, event):
        pass
    
    def boundXYtoScreen(self, x, y):
        xmin = self.xmin
        ymin = self.ymin
        xmax = self.xmax - 1
        ymax = self.ymax - 1
        if x < xmin:
            x = xmin
        elif x > xmax:
            x = xmax
        
        if y < xmin:
            y = ymin
        elif y > ymax:
            y = ymax
        
        return x, y        
    
    def mouseMoveEvent(self, event):
        x, y = event.pos().x(), event.pos().y()
        x, y = self.boundXYtoScreen(x, y)
        if self.app.overrideCursor() == Qt.SizeFDiagCursor:
            if self.frame.corner == 'topLeft':
                self.x0, self.y0 = x, y
                self.update()
            else:
                # bottomRight
                self.x1, self.y1 = x, y
                self.update()
        elif self.app.overrideCursor() == Qt.SizeBDiagCursor:
            if self.frame.corner == 'bottomLeft':
                self.x0, self.y1 = x, y
                self.update()
            else:
                # topRight
                self.x1, self.y0 = x, y
                self.update()
        elif self.app.overrideCursor() == Qt.SizeHorCursor:
            if self.frame.corner == 'left':
                self.x0 = x
                self.update()
            else:
                self.x1 = x
                self.update()
        elif self.app.overrideCursor() == Qt.SizeVerCursor:
            if self.frame.corner == 'top':
                self.y0 = y
                self.update()
            else:
                self.y1 = y
                self.update()
        elif self.app.overrideCursor() == Qt.ClosedHandCursor:
            deltax, deltay = x-self.frame.xc, y-self.frame.yc
            self.x0, self.y0 = self.x0+deltax, self.y0+deltay
            self.x1, self.y1 = self.x1+deltax, self.y1+deltay
            self.frame.xc, self.frame.yc = x, y
            self.update()

    def mouseReleaseEvent(self, event):
        if self.app.overrideCursor() == Qt.ClosedHandCursor:
            self.app.setOverrideCursor(Qt.OpenHandCursor)

        # self.update()

    def startRecorder(self):
        self.thread = QThread()
        self.screenGrabWorker = workers.ScreenRecorderWorker(
            self, user_data_folderpath
        )

        self.screenGrabWorker.moveToThread(self.thread)
        self.screenGrabWorker.finished.connect(self.thread.quit)
        self.screenGrabWorker.finished.connect(self.screenGrabWorker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.started.connect(self.screenGrabWorker.run)
        self.thread.start()
        print('Recording started...')

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_Q:
                self.close()

    def show(self):
        self.showMaximized()
        self.xmax = self.parentWin.geometry().right()
        self.ymax = self.parentWin.geometry().bottom()
        self.xmin = self.parentWin.geometry().left()
        self.ymin = self.parentWin.geometry().top()
        self.xymin = QPoint(self.xmin, self.ymin)
        self.xymax = QPoint(self.xmax, self.ymax)
        self.loadLastRect()
        self.raise_()
        self.update()

    # def mouseReleaseEvent(self, event):
    #     self.begin = event.pos()
    #     self.end = event.pos()
    #     self.update()
