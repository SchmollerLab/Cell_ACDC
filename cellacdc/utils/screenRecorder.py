import os
import sys
import time

import numpy as np
import pandas as pd

from qtpy.QtWidgets import QMainWindow, QApplication, QFrame
from qtpy.QtCore import Qt, QPoint, QRect, QObject, Signal, QThread
from qtpy.QtGui import QBrush, QColor, QPen, QPainter

from cellacdc import cellacdc_path, settings_folderpath, user_profile_path

import pathlib
USER_PATH = user_profile_path

class screenRecorderWorker(QObject):
    sigGrabScreen = Signal()
    finished = Signal()

    def __init__(self):
        QObject.__init__(self)

    def run(self):
        path = os.path.join(USER_PATH, 'Documents', 'acdc_test_grab_screen')
        for i in range(4):
            fn = f'shot_{i:03}.jpg'
            grab_path = os.path.join(path, f'shot_{i:03}.jpg')
            screen = win.screen()
            screenshot = screen.grabWindow(win.winId())
            screenshot.save(grab_path, 'jpg')
            time.sleep(0.2)

        self.finished.emit()

class myFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.parent = parent
        # Border tolerance to trigger resizing
        self.px = 10
        self.app = app

    def mousePressEvent(self, event):
        x, y = event.pos().x(), event.pos().y()
        # x00, y00 = self.parent.x0-self.px, self.parent.y0-self.px
        x01, y01 = self.parent.x0+self.px, self.parent.y0+self.px
        x10, y10 = self.parent.x1-self.px, self.parent.y1-self.px
        # x11, y11 = self.parent.x1+self.px, self.parent.y1+self.px
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
        x00, y00 = self.parent.x0-self.px, self.parent.y0-self.px
        x01, y01 = self.parent.x0+self.px, self.parent.y0+self.px
        x10, y10 = self.parent.x1-self.px, self.parent.y1-self.px
        x11, y11 = self.parent.x1+self.px, self.parent.y1+self.px
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

class screenRecorder(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.setWindowFlags(
            Qt.FramelessWindowHint # | Qt.WindowStaysOnTopHint
        )
        self.app = app

        self.topLeft_points = []
        self.topLeft_screen = QPoint(0, 0)
        self.bottomRight_screen = None
        self.loadLastRect()

        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

        frame = myFrame(self)
        frame.setMouseTracking(True)
        frame.setStyleSheet("background-color: rgba(255, 255, 255, 0)")
        self.setCentralWidget(frame)
        self.frame = frame

    def loadLastRect(self):
        self.settings_csv_path = os.path.join(settings_folderpath, 'settings.csv')
        if not os.path.exists(self.settings_csv_path):
            self.x0, self.y0, self.x1, self.y1 = 100, 100, 400, 300
            return

        self.df_settings = pd.read_csv(
            self.settings_csv_path, index_col='setting'
        )
        if 'screenRecorder_rect' in self.df_settings.index:
            s = self.df_settings.at['screenRecorder_rect', 'value']
            coords = [int(d) for d in s.split(',')]
            self.x0, self.y0, self.x1, self.y1 = coords
        else:
            self.x0, self.y0, self.x1, self.y1 = 100, 100, 400, 300

    def getRectPoints(self):
        x1, y1 = self.x1, self.y1
        x0, y0 = self.x0, self.y0
        yy = [0, y0, y1, self.b]
        xx = [0, x0, x1, self.r]
        self.topLeft_points = []
        self.bottomRight_points = []
        for i, y in enumerate(yy[:3]):
            for j, x in enumerate(xx[:3]):
                self.topLeft_points.append(QPoint(x, y))
                self.bottomRight_points.append(QPoint(xx[j+1], yy[i+1]))

    def paintEvent(self, event):
        if self.bottomRight_screen is None:
            return

        self.getRectPoints()

        qp = QPainter(self)
        br = QBrush(QColor(20, 20, 20, 140))
        qp.setBrush(br)
        qp.setPen(Qt.NoPen)

        if not self.topLeft_points:
            rect = QRect(self.topLeft_screen, self.bottomRight_screen)
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
            pen.setStyle(Qt.DashLine)
            qp.setPen(pen)
            qp.drawRect(QRect(*spotlightWindow))

    def mousePressEvent(self, event):
        pass

    def mouseMoveEvent(self, event):
        if self.app.overrideCursor() == Qt.SizeFDiagCursor:
            if self.frame.corner == 'topLeft':
                self.x0, self.y0 = event.pos().x(), event.pos().y()
                self.update()
            else:
                # bottomRight
                self.x1, self.y1 = event.pos().x(), event.pos().y()
                self.update()
        elif self.app.overrideCursor() == Qt.SizeBDiagCursor:
            if self.frame.corner == 'bottomLeft':
                self.x0, self.y1 = event.pos().x(), event.pos().y()
                self.update()
            else:
                # topRight
                self.x1, self.y0 = event.pos().x(), event.pos().y()
                self.update()
        elif self.app.overrideCursor() == Qt.SizeHorCursor:
            if self.frame.corner == 'left':
                self.x0 = event.pos().x()
                self.update()
            else:
                self.x1 = event.pos().x()
                self.update()
        elif self.app.overrideCursor() == Qt.SizeVerCursor:
            if self.frame.corner == 'top':
                self.y0 = event.pos().y()
                self.update()
            else:
                self.y1 = event.pos().y()
                self.update()
        elif self.app.overrideCursor() == Qt.ClosedHandCursor:
            x, y = event.pos().x(), event.pos().y()
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
        self.screenGrabWorker = screenRecorderWorker()

        self.screenGrabWorker.moveToThread(self.thread)
        self.screenGrabWorker.finished.connect(self.thread.quit)
        self.screenGrabWorker.finished.connect(self.screenGrabWorker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.thread.started.connect(self.screenGrabWorker.run)
        self.thread.start()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()
        elif event.modifiers() == Qt.ControlModifier:
            if event.key() == Qt.Key_Q:
                self.close()

    def show(self):
        self.showMaximized()
        self.r = win.geometry().width()
        self.b = win.geometry().height()
        self.bottomRight_screen = QPoint(self.r, self.b)
        self.raise_()
        self.update()

    # def mouseReleaseEvent(self, event):
    #     self.begin = event.pos()
    #     self.end = event.pos()
    #     self.update()

if __name__ == '__main__':
    app = QApplication([])
    win = screenRecorder(app)
    win.show()
    app.exec_()
