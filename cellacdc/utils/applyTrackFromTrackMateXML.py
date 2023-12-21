import os
from functools import partial

import pandas as pd

from .. import exception_handler
from .. import myutils, apps, widgets, html_utils, printl, workers
from .. import transformation, load
from ..utils import base

from qtpy.QtWidgets import QFileDialog

class ApplyTrackingInfoFromTrackMateUtil(base.MainThreadSinglePosUtilBase):
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
            'After clicking "Ok" you will be asked to <b>select the XML '
            'file</b> (.csv) containing the <b>tracking information</b>.'
        )
        msg.information(self, 'Instructions', txt)
        if msg.cancel:
            return False
        
        xmlPath = QFileDialog.getOpenFileName(
            self, 'Select table with tracking info', posPath,
            "XML files (*.xml);;All Files (*)"
        )[0]
        if not xmlPath:
            return False

        xmlName = os.path.basename(xmlPath)
        self.logger.info(f'Parsing XML file "{xmlName}"...')
        
        df = transformation.trackmate_xml_to_df(xmlPath)
        csvName = xmlName.replace('.xml', '.csv')
        csvPath = load.save_df_to_csv_temp_path(df, csvName, index=False)

        deleteUntrackedIDs, proceed = self.askDeleteUntrackedIDs()
        if not proceed:
            return False
        # win = apps.ApplyTrackTableSelectColumnsDialog(df, parent=self)
        # win.exec_()
        # if win.cancel:
        #     return False
        
        columnsInfo = {
            'frameIndexCol': 'frame_i',
            'trackIDsCol': 'ID',
            'maskIDsCol': 'None',
            'xCentroidCol': 'x',
            'yCentroidCol': 'y',
            'parentIDcol': 'None',
            'isFirstFrameOne': False,
            'deleteUntrackedIDs': deleteUntrackedIDs
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
    
    def askDeleteUntrackedIDs(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            'Do you want to remove objects that were not tracked?'
        )
        _, yesButton, noButton = msg.question(
            self, 'Delete untracked objects?', txt, 
            buttonsTexts=('Cancel', 'No', 'Yes')
        )
        if msg.cancel:
            return False, False
        return msg.clickedButton == yesButton, True
