from functools import partial

from qtpy.QtWidgets import QFileDialog

from .. import apps, myutils, workers, widgets, html_utils

from .base import NewThreadMultipleExpBaseUtil

class alignWin(NewThreadMultipleExpBaseUtil):
    def __init__(
            self, expPaths, app, title: str, infoText: str, 
            progressDialogueTitle: str, parent=None
        ):
        module = myutils.get_module_name(__file__)
        super().__init__(
            expPaths, app, title, module, infoText, progressDialogueTitle, 
            parent=parent
        )
        self.expPaths = expPaths
    
    def runWorker(self):
        self.worker = workers.AlignWorker(self)
        self.worker.sigAskUseSavedShifts.connect(self.askUseSavedShifts)
        self.worker.sigAskSelectChannel.connect(self.askSelectChannel)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def showEvent(self, event):
        self.runWorker()
    
    def askUseSavedShifts(self, exp_path, basename):
        txt = html_utils.paragraph(f"""
            Some or all the Positions in this experiment folder<br><br>
            <code>{exp_path}</code><br><br>
            contains <b>saved shifts from a previous alignment process</b>.<br><br>
            What do you want to do?
        """)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        _, useShiftsButton, ignoreShiftsButton, revertButton = msg.question(
            self, 'Select how saved shifts', txt,
            buttonsTexts=(
                'Cancel', 'Apply alignment from saved shifts',
                'Ignore saved shifts and compute alignment',
                'Revert alignment using saved shifts'
            )
        )
        if msg.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
        
        self.worker.revertedAlignEndname = None
        if msg.clickedButton == useShiftsButton:
            savedShiftsHow = 'use_saved_shifts'
        elif msg.clickedButton == ignoreShiftsButton:
            savedShiftsHow = 'ignore_saved_shifts'
        elif msg.clickedButton == revertButton:
            savedShiftsHow = 'rever_alignment'
            txt = html_utils.paragraph(f"""
                How do you want to save the image file with reverted alignment?
            """)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            overWriteButton = widgets.savePushButton('Overwrite existing file')
            saveAsButton = widgets.newFilePushButton('Save as new file...')
            _, overWriteButton, saveAsButton = msg.question(
                self, 'Select how saved shifts', txt,
                buttonsTexts=('Cancel', overWriteButton, saveAsButton),
                showDialog=False
            )
            saveAsButton.clicked.disconnect()
            saveAsCallback = partial(self.askAppendedName, basename, msg)
            saveAsButton.clicked.connect(saveAsCallback)
            msg.exec_()
            if msg.cancel:
                self.worker.abort = True
                self.worker.waitCond.wakeAll()
        
        self.worker.savedShiftsHow = savedShiftsHow

        self.worker.waitCond.wakeAll()
    
    def askAppendedName(self, basename, parent):
        win = apps.filenameDialog(
            ext='.tif', title='Reverted alignment data filename',
            hintText='Insert a text to append to the filename', 
            parent=self, basename=basename, allowEmpty=False
        )
        win.exec_()
        if win.cancel:
            return
        self.worker.revertedAlignEndname = win.entryText
        parent.cancel = False
        parent.close()
    
    def askSelectChannel(self, channels):
        selectChannelWin = apps.QDialogCombobox(
            'Select channel', channels, 'Select reference channel for the aligment',
            CbLabel='Select channel:  ', parent=self
        )
        selectChannelWin.exec_()
        if selectChannelWin.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll() 
        
        self.worker.chName = selectChannelWin.selectedItemText
        self.worker.waitCond.wakeAll()

    def workerAborted(self):
        self.workerFinished(None, aborted=True)
    
    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = 'Aligning frames process CANCELLED.'
        else:
            txt = 'Aligning frames process completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()