from qtpy.QtCore import QEventLoop, Qt
from qtpy.QtWidgets import QDialog, QMainWindow

from .. import printl


class QBaseDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)

    def exec_(self, resizeWidthFactor=None):
        if resizeWidthFactor is not None:
            self.show()
            self.resize(int(self.width() * resizeWidthFactor), self.height())
        self.show(block=True)

    def show(self, block=False):
        self.setWindowFlags(Qt.Window | Qt.WindowStaysOnTopHint)
        super().show()

        try:
            self.setEnabled(True)
        except Exception as err:
            pass

        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        if hasattr(self, "loop"):
            self.loop.exit()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            event.ignore()
            return

        super().keyPressEvent(event)


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
        if hasattr(self, "loop"):
            self.loop.exit()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            event.ignore()
            return

        super().keyPressEvent(event)
