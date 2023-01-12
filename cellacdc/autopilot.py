import os

from PyQt5.QtCore import (
    QTimer, QThread
)

from . import load, printl, workers

class AutoPilot:
    def __init__(self, guiWin) -> None:
        self.guiWin = guiWin
        self.app = guiWin.app
    
    def _askSelectPos(self):
        posData = self.guiWin.data[self.guiWin.pos_i]
        exp_path = posData.exp_path
        select_folder = load.select_exp_folder()
        values = select_folder.get_values_segmGUI(exp_path)
        select_folder.QtPrompt(self.guiWin, values, allowMultiSelection=False)
        if select_folder.was_aborted:
            return
        
        posPath = os.path.join(exp_path, select_folder.selected_pos[0])
        return posPath

    def startLoadPos(self):
        posPath = self._askSelectPos()
        if posPath is None:
            self.guiWin.logger.info('Loading Position cancelled.')
            return
        
        self.guiWin.openFolder(exp_path=posPath)
        
        self.thread = QThread(self.guiWin)
        self.worker = workers.AutoPilotWorker(self.guiWin)

        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.finished.connect(self.workerFinished)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def loadPosLoop(self):
        openWindows = self.app.topLevelWidgets()
        printl([window.windowTitle() for window in openWindows])
    
    def workerFinished(self):
        pass