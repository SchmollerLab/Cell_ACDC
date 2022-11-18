import os
from functools import partial

import pandas as pd

from .. import myutils, apps, widgets, html_utils
from ..utils import base

from PyQt5.QtWidgets import QFileDialog

class ApplyTrackingInfoFromTableUtil(base.MainThreadSinglePosUtilBase):
    def __init__(
            self, posPath, app, title: str, infoText: str, parent=None
        ):
        
        module = myutils.get_module_name(__file__)
        super().__init__(
            app, title, module, infoText, parent
        )

        self.sigClose.connect(self.close)

    @myutils.exception_handler
    def run(self, posPath):
        self.logger.info('Loading segmentation file...')
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
        self.logger.info(f'Loading table file {csvName}...')

        df = pd.read_csv(csvPath)

        win = apps.ApplyTrackTableSelectColumnsDialog(df, parent=self)
        win.exec_()
        if win.cancel:
            return False
        
