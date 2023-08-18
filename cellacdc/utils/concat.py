from qtpy.QtWidgets import QFileDialog

from cellacdc import measurements

from .. import apps, myutils, workers, widgets, html_utils
from .. import printl

from .base import NewThreadMultipleExpBaseUtil

class concatWin(NewThreadMultipleExpBaseUtil):
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
        self.worker = workers.ConcatAcdcDfsWorker(self)
        self.worker.sigAskFolder.connect(self.askFolderWhereToSaveAllExp)
        self.worker.sigAborted.connect(self.workerAborted)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigSetMeasurements.connect(self.askSetMeasurements)
        super().runWorker(self.worker)
    
    def askSetMeasurements(self, kwargs):
        loadedChNames = kwargs['loadedChNames']
        notLoadedChNames = kwargs['notLoadedChNames']
        isZstack = kwargs['isZstack']
        isSegm3D = kwargs['isSegm3D']
        self.setMeasurementsWin = apps.SetMeasurementsDialog(
            loadedChNames, notLoadedChNames, isZstack, isSegm3D,
            is_concat=True, parent=self
        )
        existing_colnames = kwargs['existing_colnames']
        self.setMeasurementsWin.addNonMeasurementColumns(
            existing_colnames
        )
        self.setMeasurementsWin.setDisabledNotExistingMeasurements(
            existing_colnames
        )
        self.setMeasurementsWin.sigClosed.connect(self.setMeasurements)
        self.setMeasurementsWin.sigCancel.connect(self.setMeasurementsCancelled)
        self.setMeasurementsWin.show()
    
    def setMeasurements(self):
        selectedColumns = []
        if hasattr(self.setMeasurementsWin, 'nonMeasurementsGroupbox'):
            if self.setMeasurementsWin.nonMeasurementsGroupbox.isChecked():
                groupbox = self.setMeasurementsWin.nonMeasurementsGroupbox
                for checkBox in groupbox.checkBoxes: 
                    if not checkBox.isChecked():
                        continue
                    colname = checkBox.text()
                    selectedColumns.append(colname) 
                
        for chNameGroupbox in self.setMeasurementsWin.chNameGroupboxes:
            chName = chNameGroupbox.chName
            if not chNameGroupbox.isChecked():
                # Skip entire channel
                continue
            
            for checkBox in chNameGroupbox.checkBoxes:
                if not checkBox.isChecked():
                    continue
                colname = checkBox.text()
                selectedColumns.append(colname)
        
        if self.setMeasurementsWin.sizeMetricsQGBox.isChecked():
            for checkBox in self.setMeasurementsWin.sizeMetricsQGBox.checkBoxes:
                if not checkBox.isChecked():
                    continue
                colname = checkBox.text()
                selectedColumns.append(colname)
        
        selectedPropsNames = []
        if self.setMeasurementsWin.regionPropsQGBox.isChecked():
            for checkBox in self.setMeasurementsWin.regionPropsQGBox.checkBoxes:
                if not checkBox.isChecked():
                    continue
                colname = checkBox.text()
                selectedPropsNames.append(colname)
            selectedRpCols = measurements.get_regionprops_columns(
                self.setMeasurementsWin.existing_colnames, selectedPropsNames
            )
            selectedColumns.extend(selectedRpCols)
        
        checkMixedChannel = (
            self.setMeasurementsWin.mixedChannelsCombineMetricsQGBox is not None
            and self.setMeasurementsWin.mixedChannelsCombineMetricsQGBox.isChecked()
        )
        if checkMixedChannel:
            win = self.setMeasurementsWin
            checkBoxes = win.mixedChannelsCombineMetricsQGBox.checkBoxes
            for checkBox in checkBoxes:
                if not checkBox.isChecked():
                    continue
                colname = checkBox.text()
                selectedColumns.append(colname)
            
        self.worker.selectedColumns = selectedColumns
        self.worker.abort = False
        self.worker.waitCond.wakeAll()
    
    def setMeasurementsCancelled(self):
        self.worker.abort = True
        self.worker.waitCond.wakeAll()
    
    def showEvent(self, event):
        self.runWorker()
    
    def askAppendName(self, basename, existingEndnames):
        win = apps.filenameDialog(
            basename=basename,
            hintText='Insert a name for the <b>concatenated table</b> file:',
            existingNames=existingEndnames, 
            allowEmpty=True,
            ext='.csv'
        )
        win.exec_()
        if win.cancel:
            self.worker.abort = True     
        else:   
            self.worker.concat_df_filename = win.filename
        self.worker.waitCond.wakeAll()
    
    def askFolderWhereToSaveAllExp(self, allExp_filename):
        txt = html_utils.paragraph(f"""
            After clicking "Ok" you will be asked to <b>select a folder where you
            want to save the file</b><br>
            with the <b>concatenated tables</b> from the multiple experiments selected<br>
            (the filename will be <code>{allExp_filename}</code>)
        """)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.information(self, 'Select folder', txt)
        if msg.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
        
    
        mostRecentPath = myutils.getMostRecentPath()
        save_to_dir = QFileDialog.getExistingDirectory(
            self, f'Select folder where to save {allExp_filename}', 
            mostRecentPath
        )
        if not save_to_dir:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
        
        self.worker.allExpSaveFolder = save_to_dir

        self.worker.waitCond.wakeAll()

    def workerAborted(self):
        self.worker.signals.finished.emit(self)
        self.workerFinished(self.worker, aborted=True)
    
    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = 'Concatenating <code>acdc_output</code> tables aborted.'
        else:
            txt = 'Concatenating <code>acdc_output</code> tables completed.'
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if aborted:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()