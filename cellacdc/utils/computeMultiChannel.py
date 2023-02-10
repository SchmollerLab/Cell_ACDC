import pandas as pd

from .. import apps, myutils, workers, widgets, html_utils, load

from .base import NewThreadMultipleExpBaseUtil

class ComputeMetricsMultiChannel(NewThreadMultipleExpBaseUtil):
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
        self.worker = workers.ComputeMetricsMultiChannelWorker(self)
        self.worker.sigAskAppendName.connect(self.askAppendName)
        self.worker.sigCriticalNotEnoughSegmFiles.connect(
            self.criticalNotEnoughSegmFiles
        )
        self.worker.sigHowCombineMetrics.connect(self.showHowCombineMetrics)
        self.worker.sigAborted.connect(self.workerAborted)
        super().runWorker(self.worker)
    
    def showEvent(self, event):
        self.runWorker()
    
    def showHowCombineMetrics(
            self, imagesPath, selectedAcdcOutputEndnames, 
            existingAcdcOutputEndnames, allChNames
        ):
        self.imagesPath = imagesPath
        self.existingAcdcOutputEndnames = existingAcdcOutputEndnames
        acdcDfsDict = {}
        for endname in selectedAcdcOutputEndnames:
            filePath, _ = load.get_path_from_endname(endname, imagesPath)
            acdc_df = pd.read_csv(filePath)
            acdcDfsDict[endname] = acdc_df

        self.combineWindow = apps.CombineMetricsMultiDfsSummaryDialog(
            acdcDfsDict, allChNames, parent=self
        )
        self.combineWindow.setLogger(
            self.logger, self.logs_path, self.log_path
        )
        self.combineWindow.sigLoadAdditionalAcdcDf.connect(
            self.loadAdditionalAcdcDf
        )
        self.combineWindow.exec_()
        if self.combineWindow.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return

        self.worker.equations = self.combineWindow.equations
        self.worker.acdcDfs = self.combineWindow.acdcDfs
        self.worker.waitCond.wakeAll()
    
    def loadAdditionalAcdcDf(self):
        selectWindow = widgets.QDialogListbox(
            'Select acdc_output files',
            f'Select acdc_output files to load\n',
            self.existingAcdcOutputEndnames, multiSelection=True, 
            parent=self, allowSingleSelection=True
        )
        selectWindow.exec_()
        if selectWindow.cancel or not selectWindow.selectedItemsText:
            self.logger.info('Loading additional tables cancelled.')
            return
        
        acdcDfsDict = {}
        for end in selectWindow.selectedItemsText:
            filePath, _ = load.get_path_from_endname(end, self.imagesPath)
            acdc_df = pd.read_csv(filePath)
            acdcDfsDict[end] = acdc_df

    def criticalNotEnoughSegmFiles(self, exp_path):
        text = html_utils.paragraph(f"""
            The following experiment folder<br><br>
            <code>{exp_path}</code><br><br>
            <b>does NOT contain AT LEAST TWO <code>acdc_output</code> 
            table files.</b>.
        """)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        msg.addShowInFileManagerButton(exp_path)
        msg.critical(
            self, 'Not enough segmentation files!', text
        )
        self.worker.abort = True
        self.worker.waitCond.wakeAll()
    
    def askAppendName(self, basename, existingEndnames, selectedEndnames):
        helpText = (
            """
            The CSV table file with the combined measurements 
            will be saved with a different file name.<br><br>
            Insert a name to append to the end of the new name. The rest of 
            the name will have the same basename as all other files.
            """
        )
        channels = [end.replace('acdc_output_', '') for end in selectedEndnames]
        channels = [end.replace('acdc_output', '') for end in channels]
        channels = [end if end else 'refCh' for end in channels]
        defaultEntry = f"{'_'.join(channels)}_combined_metrics"
        win = apps.filenameDialog(
            basename=basename,
            hintText='Insert a name for the <b>new, table</b> file:',
            existingNames=existingEndnames, 
            helpText=helpText, 
            allowEmpty=False,
            ext='.csv',
            defaultEntry=defaultEntry,
            resizeOnShow=True
        )
        win.exec_()
        if win.cancel:
            self.worker.abort = True
            self.worker.waitCond.wakeAll()
            return
        
        self.worker.appendedName = win.entryText
        self.worker.waitCond.wakeAll()
    
    def workerAborted(self):
        self.workerFinished(None, aborted=True)
    
    def workerCritical(self, error):
        super().workerCritical(error)
        self.worker.errors[error] = self.traceback_str
    
    def warnErrors(self, errors):
        win = apps.ComputeMetricsErrorsDialog(
            errors, self.logs_path, log_type='generic', parent=self
        )
        win.exec_()
    
    def workerFinished(self, worker, aborted=False):
        if aborted:
            txt = 'Combining multiple channels measurements aborted.'
            isWarning = True
        elif worker.errors:
            txt = 'Combining multiple channels measurements completed WITH ERRORS.'
            self.warnErrors(worker.errors)
            isWarning = True
        else:
            txt = html_utils.paragraph(
                'Combining multiple channels measurements completed.<br><br>'
                'Results were saved in the respective Position folder(s).'
            )
            isWarning = False
        self.logger.info(txt)
        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if isWarning:
            msg.warning(self, 'Process completed', html_utils.paragraph(txt))
        else:
            msg.information(self, 'Process completed', html_utils.paragraph(txt))
        super().workerFinished(worker)
        self.close()