import os
from functools import partial

import pandas as pd

from .. import exception_handler
from .. import html_utils, printl
from cellacdc.myutils import filesystem
from cellacdc.shared.dialogs import file_selection as dialogs_file_selection
from cellacdc.shared.widgets import (
    combo_boxes as widgets_combo_boxes,
    message_box as widgets_message_box,
    push_buttons as widgets_push_buttons,
)
from . import segm_workers
from ..utils import base

from qtpy.QtWidgets import QFileDialog
from qtpy.QtWidgets import (
    QLabel, QVBoxLayout, QHBoxLayout, QFormLayout, QCheckBox
)
from cellacdc import fonts
from cellacdc.shared.dialogs import base as dialogs_base

class ApplyTrackingInfoFromTableUtil(base.MainThreadSinglePosUtilBase):
    def __init__(
            self, app, title: str, infoText: str, parent=None,
            callbackOnFinished=None
        ):
        module = filesystem.get_module_name(__file__)
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
        
        msg = widgets_message_box.myMessageBox(showCentered=False, wrapText=False)
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

        win = ApplyTrackTableSelectColumnsDialog(df, parent=self)
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
            f for f in filesystem.listdir(imagesPath) 
            if f.endswith(f'{endFilenameSegm}.npz')
        ][0]
        basename = os.path.splitext(segmFilename)[0]
        overWriteButton = widgets_push_buttons.savePushButton(
            'Overwrite existing segmentation file'
        )
        win = dialogs_file_selection.filenameDialog(
            basename=f'{basename}_',
            hintText='Insert a <b>filename for the tracked</b> masks file:',
            allowEmpty=False, defaultEntry='tracked', 
            additionalButtons=(overWriteButton, )
        )
        overWriteButton.clicked.connect(partial(self.overWriteClicked, win))
        win.exec_()
        if win.cancel:
            return False

        self.worker = segm_workers.ApplyTrackInfoWorker(
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
        


class ApplyTrackTableSelectColumnsDialog(dialogs_base.QBaseDialog):
    def __init__(self, df, parent=None):
        super().__init__(parent)

        self.setWindowTitle('Select columns containing tracking info')
        
        self.cancel = True
        self.mainLayout = QVBoxLayout()

        options = (
            '"Frame index", "Tracked IDs" and "Segmentation mask IDs"<br>',
            '"Frame index", "Tracked IDs", "X coord. centroid", and "Y coord. centroid"'
        )
        self.instructionsText = html_utils.paragraph(
            f"""
            <b>Select which columns</b> contain the tracking information.<br><br>
            You must choose one of the following combinations:<br> 
            {html_utils.to_list(options)}
            Optionally, you can provide the column name containing the parent ID.<br>
            This will allow you to load lineage information into Cell-ACDC. 
            """
        )
        self.mainLayout.addWidget(QLabel(self.instructionsText))

        formLayout = QFormLayout()

        self.frameIndexCombobox = widgets_combo_boxes.QCenteredComboBox()
        self.frameIndexCombobox.addItems(df.columns)
        self.frameIndexCheckbox = QCheckBox('1st frame is index 1')
        frameIndexLayout = QHBoxLayout()
        frameIndexLayout.addWidget(self.frameIndexCombobox)
        frameIndexLayout.addWidget(self.frameIndexCheckbox)
        frameIndexLayout.setStretch(0, 2)
        frameIndexLayout.setStretch(1, 0)
        formLayout.addRow(
            'Frame index: ', frameIndexLayout
        )

        self.trackedIDsCombobox = widgets_combo_boxes.QCenteredComboBox()
        self.trackedIDsCombobox.addItems(df.columns)
        formLayout.addRow('Tracked IDs: ', self.trackedIDsCombobox)

        items = df.columns.to_list()
        items.insert(0, 'None')
        self.maskIDsCombobox = widgets_combo_boxes.QCenteredComboBox()
        self.maskIDsCombobox.addItems(items)
        formLayout.addRow('Segmentation mask IDs: ', self.maskIDsCombobox)

        self.xCentroidCombobox = widgets_combo_boxes.QCenteredComboBox()
        self.xCentroidCombobox.addItems(items)
        formLayout.addRow('X coord. centroid: ', self.xCentroidCombobox)

        self.yCentroidCombobox = widgets_combo_boxes.QCenteredComboBox()
        self.yCentroidCombobox.addItems(items)
        formLayout.addRow('Y coord. centroid: ', self.yCentroidCombobox)

        self.parentIDcombobox = widgets_combo_boxes.QCenteredComboBox()
        self.parentIDcombobox.addItems(items)
        formLayout.addRow('Parent ID (optional): ', self.parentIDcombobox)

        deleteUntrackedLayout = QHBoxLayout()
        self.deleteUntrackedIDsToggle = widgets_push_buttons.Toggle()
        deleteUntrackedLayout.addStretch(1)
        deleteUntrackedLayout.addWidget(self.deleteUntrackedIDsToggle)
        deleteUntrackedLayout.addStretch(1)
        formLayout.addRow('Delete untracked IDs: ', deleteUntrackedLayout)
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)

        self.mainLayout.addSpacing(30)
        self.mainLayout.addLayout(formLayout)
        self.mainLayout.addSpacing(20)
        self.mainLayout.addLayout(buttonsLayout)

        self.setLayout(self.mainLayout)
        self.setFont(fonts.font)
    
    def ok_cb(self):
        self.cancel = False
        self.frameIndexCol = self.frameIndexCombobox.currentText()
        self.trackedIDsCol = self.trackedIDsCombobox.currentText()
        self.maskIDsCol = self.maskIDsCombobox.currentText()
        self.xCentroidCol = self.xCentroidCombobox.currentText()
        self.yCentroidCol = self.yCentroidCombobox.currentText()
        self.deleteUntrackedIDs = self.deleteUntrackedIDsToggle.isChecked()
        if self.maskIDsCol == 'None':
            if self.xCentroidCol == 'None' or self.yCentroidCol == 'None':
                self.warnInvalidSelection()
                return
        else:
            self.xCentroidCol = 'None'
            self.yCentroidCol = 'None'
        self.parentIDcol = self.parentIDcombobox.currentText()
        self.isFirstFrameOne = self.frameIndexCheckbox.isChecked()
        self.close()
    
    def warnInvalidSelection(self):
        msg = widgets_message_box.myMessageBox(showCentered=False, wrapText=False)
        msg.warning(
            self, 'Invalid selection', html_utils.paragraph(
                f'<b>Invalid selection</b><br> {self.instructionsText}'
            )
        )
