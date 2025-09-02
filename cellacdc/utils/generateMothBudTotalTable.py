import os
from functools import partial

import pandas as pd

from .. import exception_handler
from .. import myutils, apps, widgets, html_utils, printl, workers
from ..utils import base

from qtpy.QtWidgets import QFileDialog

class GenerateMothBudTotalUtil(base.MainThreadSinglePosUtilBase):
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
    def run(self):        
        msg = widgets.myMessageBox(showCentered=False, wrapText=False)
        txt = html_utils.paragraph(
            'After clicking "Ok" you will be asked to <b>select the input table '
            'file</b> (.csv) containing <b>pedigree information</b>.'
        )
        msg.information(self, 'Instructions', txt)
        if msg.cancel:
            return False
        
        import qtpy.compat
        input_csv_filepath = qtpy.compat.getopenfilename(
            parent=self, 
            caption='Select CSV file to load', 
            filters='CSV (*.csv);;All Files (*)',
            basedir=myutils.getMostRecentPath()
        )[0]
        if input_csv_filepath is None:
            return False

        self.logger.info(f'Reading column names in table "{input_csv_filepath}"...')

        df = pd.read_csv(input_csv_filepath, nrows=2)

        win = apps.GenerateMotherBudTotalTableSelectColumnsDialog(
            df, parent=self
        )
        win.exec_()
        if win.cancel:
            return False
        
        csv_filename = os.path.basename(input_csv_filepath)
        csv_filename_noext, ext = os.path.splitext(csv_filename)[0]
        win = apps.filenameDialog(
            ext='.csv',
            basename=f'{csv_filename_noext}_',
            hintText='Insert a <b>filename for the output table</b> file:',
            allowEmpty=False, 
            defaultEntry='mother_bud_total', 
        )
        win.exec_()
        if win.cancel:
            return False

        selected_options = win.selected_options
        
        self.worker = workers.GenerateMotherBudTotalTableWorker(
            self, input_csv_filepath, selected_options
        )
        if self.callbackOnFinished is not None:
            self.worker.signals.finished.connect(self.callbackOnFinished)
        self.runWorker(self.worker)
        return True
        
    def overWriteClicked(self, win):
        win.cancel = False
        win.filename = ''
        win.close()
        
