import os

from qtpy.QtCore import (
    QTimer, QThread, Signal, QObject
)

from . import load, printl, myutils

class AutoPilotProfile:
    def __init__(self):
        self.lastLoadingProfile = []

    def storeSelectedChannel(self, user_channel):
        self.lastLoadingProfile.append({
            'windowTitle': 'Select channel name', 
            'windowActions': ('ComboBox.setCurrentText', 'ok_cb'),
            'windowActionsArgs': ((user_channel,), tuple())
        })

    def storeSelectedSegmFile(self, selectedSegmEndName):
        self.lastLoadingProfile.append({
            'windowTitle': 'Multiple segm.npz files detected', 
            'windowActions': ('listWidget.setSelectedItemFromText', 'ok_cb'),
            'windowActionsArgs': ((selectedSegmEndName,), tuple())
        })
    
    def storeOkAskInputMetadata(self):
        self.lastLoadingProfile.append({
            'windowTitle': 'Image properties', 
            'windowActions': ('ok_cb',),
            'windowActionsArgs': (tuple(),)
        })
    
    def storeLoadSavedData(self):
        self.lastLoadingProfile.append({
            'windowTitle': 'Recover unsaved data?', 
            'windowActions': ('clickButtonFromText',),
            'windowActionsArgs': (('Load saved data',),)
        })
    
    def storeClickMessageBox(self, windowTitle, buttonTextToClick):
        self.lastLoadingProfile.append({
            'windowTitle': windowTitle, 
            'windowActions': ('clickButtonFromText',),
            'windowActionsArgs': ((buttonTextToClick,),)
        })
    
    def storeLoadedFluoChannels(self, loadedChannels):
        self.lastLoadingProfile.append({
            'windowTitle': 'Select channel to load', 
            'windowActions': ('setSelectedItems', 'ok_cb'),
            'windowActionsArgs': ((loadedChannels,), tuple())
        })
    
    def getCopy(self):
        return self.lastLoadingProfile.copy()


class AutoPilot:    
    def __init__(self, parentWin) -> None:
        self.parentWin = parentWin
        self.app = parentWin.app
        self.isFinished = True
        self.loadingProfile = parentWin.AutoPilotProfile.getCopy()
    
    def _askSelectPos(self):
        posData = self.parentWin.data[self.parentWin.pos_i]
        exp_path = posData.exp_path
        select_folder = load.select_exp_folder()
        values = select_folder.get_values_segmGUI(exp_path)
        # Remove currently loaded position
        values.pop(select_folder.pos_foldernames.index(posData.pos_foldername))
        select_folder.pos_foldernames.remove(posData.pos_foldername)
        select_folder.QtPrompt(self.parentWin, values, allowMultiSelection=False)
        if select_folder.was_aborted:
            return
        
        posPath = os.path.join(exp_path, select_folder.selected_pos[0])
        return posPath

    def execLoadPos(self):
        posPath = self._askSelectPos()
        if posPath is None:
            self.parentWin.logger.info('Loading Position cancelled.')
            return
        
        self.isFinished = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.loadPosTimerCallback)
        self.timer.start(50)

        self.parentWin.openFolder(exp_path=posPath)

    def loadPosTimerCallback(self):
        openWindows = self.app.topLevelWidgets()
        if not self.loadingProfile:
            self.timer.stop()
            return
        
        windowTitle = self.loadingProfile[0]['windowTitle']
        for window in openWindows:
            if not window.windowTitle():
                continue
            if not window.isVisible():
                continue
            if not windowTitle == window.windowTitle():
                continue
            
            windowActions = self.loadingProfile[0]['windowActions']
            windowActionsArgs = self.loadingProfile[0]['windowActionsArgs']
            for action, args in zip(windowActions, windowActionsArgs):
                func = myutils.get_chained_attr(window, action)
                func(*args)
            
            self.loadingProfile.pop(0)
            break
        