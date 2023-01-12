import os

from PyQt5.QtCore import (
    QTimer, QThread, pyqtSignal, QObject
)

from . import load, printl, myutils

class AutoPilot:    
    def __init__(self, guiWin) -> None:
        self.guiWin = guiWin
        self.app = guiWin.app
        self.isFinished = True
        self.loadingProfile = guiWin.lastLoadingProfile.copy()
    
    def _askSelectPos(self):
        posData = self.guiWin.data[self.guiWin.pos_i]
        exp_path = posData.exp_path
        select_folder = load.select_exp_folder()
        values = select_folder.get_values_segmGUI(exp_path)
        # Remove currently loaded position
        values.pop(select_folder.pos_foldernames.index(posData.pos_foldername))
        select_folder.QtPrompt(self.guiWin, values, allowMultiSelection=False)
        if select_folder.was_aborted:
            return
        
        posPath = os.path.join(exp_path, select_folder.selected_pos[0])
        return posPath

    def execLoadPos(self):
        posPath = self._askSelectPos()
        if posPath is None:
            self.guiWin.logger.info('Loading Position cancelled.')
            return
        
        self.isFinished = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.loadPosTimerCallback)
        self.timer.start(100)

        self.guiWin.openFolder(exp_path=posPath)

        if self.timer.isActive():
            self.timer.stop()
    
    def loadPosTimerCallback(self):
        openWindows = self.app.topLevelWidgets()
        if not self.loadingProfile:
            self.timer.stop()
            return
        
        windowTitle = self.loadingProfile[0]['windowTitle']
        for window in openWindows:
            if not window.isVisible():
                continue
            if not windowTitle == window.windowTitle():
                continue
            
            windowActions = self.loadingProfile[0]['windowActions']
            windowActionsArgs = self.loadingProfile[0]['windowActionsArgs']
            for action, args in zip(windowActions, windowActionsArgs):
                printl(action, args)
                func = myutils.get_chained_attr(window, action)
                func(*args)
            
            self.loadingProfile.pop(0)
            break
        