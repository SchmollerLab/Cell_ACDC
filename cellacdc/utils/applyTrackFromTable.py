import os
from functools import partial

import pandas as pd

from .. import exception_handler
from .. import myutils, apps, widgets, html_utils, printl, workers
from ..utils import base

from qtpy.QtWidgets import QFileDialog

class ApplyTrackingInfoFromTableUtil(base.MainThreadSinglePosUtilBase):
    def __init__(
            self, app, title: str, infoText: str, parent=None,
            callbackOnFinished=None
        ):
        module = myutils.get_module_name(__file__)
        super().__init__(
            app, title, module, infoText, parent
        )

        self.sigClose.connect(self.close)
        self.callbackOnFinished = callbackOnFinished

    @exception_handler
    def run(self, posPath):
        self.logger.info('Reading exisiting segmentation file names...')
        endFilenameSegm = self.selectSegmFileLoadData(posPath)

        if not endFilenameSegm:
            return False
        
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph(
            'After clicking "Ok" you will be asked to <b>select the table '
            'file</b> (.csv) containing the <b>tracking information</b>.'
        )
        msg.information(self, 'Instructions', txt)
        if msg.cancel:
            return False
        
        csvPath = QFileDialog.getOpenFileName(
            self, 'Select table with tracking info', posPath,
            "CSV files (*.csv);;All Files (*)"
        )[0]
        if not csvPath:
            return False

        csvName = os.path.basename(csvPath)
        self.logger.info(f'Reading column names in table "{csvName}"...')

        df = pd.read_csv(csvPath, nrows=2)

        win = apps.ApplyTrackTableSelectColumnsDialog(df, parent=self)
        win.exec_()
        if win.cancel:
            return False
        
        columnsInfo = {
            'frameIndexCol': win.frameIndexCol,
            'trackIDsCol': win.trackedIDsCol,
            'maskIDsCol': win.maskIDsCol,
            'xCentroidCol': win.xCentroidCol,
            'yCentroidCol': win.yCentroidCol,
            'parentIDcol': win.parentIDcol,
            'isFirstFrameOne': win.isFirstFrameOne,
            'deleteUntrackedIDs': win.deleteUntrackedIDs
        }
        
        imagesPath = os.path.join(posPath, 'Images')
        segmFilename = [
            f for f in myutils.listdir(imagesPath) 
            if f.endswith(f'{endFilenameSegm}.npz')
        ][0]
        basename = os.path.splitext(segmFilename)[0]
        overWriteButton = widgets.savePushButton(
            'Overwrite existing segmentation file'
        )
        win = apps.filenameDialog(
            basename=f'{basename}_',
            hintText='Insert a <b>filename for the tracked</b> masks file:',
            allowEmpty=False, defaultEntry='tracked', 
            additionalButtons=(overWriteButton, )
        )
        overWriteButton.clicked.connect(partial(self.overWriteClicked, win))
        win.exec_()
        if win.cancel:
            return False

        self.worker = workers.ApplyTrackInfoWorker(
            self, endFilenameSegm, csvPath, win.filename, columnsInfo, posPath
        )
        if self.callbackOnFinished is not None:
            self.worker.signals.finished.connect(self.callbackOnFinished)
        self.runWorker(self.worker)
        return True
        
    def overWriteClicked(self, win):
        win.cancel = False
        win.filename = ''
        win.close()
        
