"""View adapter for object cleanup workflows."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QThread

from cellacdc import apps, widgets, workers

from .cell_cycle import CellCycle


class ObjectCleanup(CellCycle):
    """Extracted from guiWin."""

    def delObjsOutSegmMaskActionTriggered(self):
        posData = self.data[self.pos_i]
        segm_files = load.get_segm_files(posData.images_path)
        existingSegmEndnames = load.get_endnames(posData.basename, segm_files)
        selectSegmWin = widgets.QDialogListbox(
            "Select segmentation file",
            "Select segmentation file to use as ROI:\n",
            existingSegmEndnames,
            multiSelection=False,
            parent=self,
        )
        selectSegmWin.exec_()
        if selectSegmWin.cancel:
            self.logger.info("Delete objects process cancelled.")
            return

        selectedSegmEndname = selectSegmWin.selectedItemsText[0]

        self.startDelObjsOutSegmMaskWorker(selectedSegmEndname)

    def delObjsOutSegmMaskWorkerFinished(self, result):
        posData = self.data[self.pos_i]
        worker, cleared_segm_data, delIDs = result
        if posData.SizeT == 1:
            cleared_segm_data = cleared_segm_data[np.newaxis]

        self.update_cca_df_deletedIDs(posData, delIDs)

        current_frame_i = posData.frame_i
        for frame_i, cleared_lab in enumerate(cleared_segm_data):
            # Store change
            posData.allData_li[frame_i]["labels"] = cleared_lab
            # Get the rest of the stored metadata based on the new lab
            posData.frame_i = frame_i
            self.get_data()
            self.store_data(autosave=False)

        # Back to current frame
        posData.frame_i = current_frame_i
        self.get_data()

        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        self.logger.info("Deleting objects outside of ROIs finished.")
        self.titleLabel.setText("Deleting objects outside of ROIs finished.", color="w")
        self.updateAllImages()

    def startDelObjsOutSegmMaskWorker(self, selectedSegmEndname):
        self.store_data(autosave=False)
        posData = self.data[self.pos_i]
        segm_data = np.squeeze(self.getStoredSegmData())

        self.progressWin = apps.QDialogWorkerProgress(
            title="Deleting objects outside of ROIs",
            parent=self,
            pbarDesc="Deleting objects outside of ROIs...",
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)

        self.thread = QThread()
        self.worker = workers.DelObjectsOutsideSegmROIWorker(
            selectedSegmEndname, segm_data, posData.images_path
        )
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.progress.connect(self.workerProgress)
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.delObjsOutSegmMaskWorkerFinished)

        self.worker.debug.connect(self.workerDebug)

        self.thread.started.connect(self.worker.run)
        self.thread.start()
