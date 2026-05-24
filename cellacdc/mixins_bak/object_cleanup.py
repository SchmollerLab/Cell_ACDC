"""View adapter for object cleanup workflows."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QThread

from cellacdc import apps, widgets, workers


class ObjectCleanupMixin:
    """Qt-facing adapter around the object-cleanup view-model."""

    """Headless object-cleanup result shaping."""

    def cleared_segmentation_frames(self, cleared_segm_data, *, size_t: int):
        if size_t == 1:
            return cleared_segm_data[np.newaxis]
        return cleared_segm_data

    def delete_objects_outside_mask_action_triggered(self):
        pos_data = self.data[self.pos_i]
        existing_segm_endnames = self.segmentation_roi_endnames(
            basename=pos_data.basename,
            images_path=pos_data.images_path,
        )
        select_segm_win = widgets.QDialogListbox(
            "Select segmentation file",
            "Select segmentation file to use as ROI:\n",
            existing_segm_endnames,
            multiSelection=False,
            parent=self,
        )
        select_segm_win.exec_()
        if select_segm_win.cancel:
            self.logger.info("Delete objects process cancelled.")
            return

        selected_segm_endname = select_segm_win.selectedItemsText[0]
        self.start_delete_objects_outside_mask_worker(selected_segm_endname)

    def delete_objects_outside_mask_worker_finished(self, result):
        pos_data = self.data[self.pos_i]
        worker, cleared_segm_data, del_ids = result
        cleared_segm_data = self.cleared_segmentation_frames(
            cleared_segm_data,
            size_t=pos_data.SizeT,
        )

        self.update_cca_df_deletedIDs(pos_data, del_ids)

        current_frame_i = pos_data.frame_i
        for frame_i, cleared_lab in self.frame_labels(cleared_segm_data):
            pos_data.allData_li[frame_i]["labels"] = cleared_lab
            pos_data.frame_i = frame_i
            self.get_data()
            self.store_data(autosave=False)

        pos_data.frame_i = current_frame_i
        self.get_data()

        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        self.logger.info("Deleting objects outside of ROIs finished.")
        self.titleLabel.setText(
            "Deleting objects outside of ROIs finished.",
            color="w",
        )
        self.updateAllImages()

    def frame_labels(self, cleared_segm_data):
        return list(enumerate(cleared_segm_data))

    def start_delete_objects_outside_mask_worker(self, selected_segm_endname):
        self.store_data(autosave=False)
        pos_data = self.data[self.pos_i]
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
            selected_segm_endname,
            segm_data,
            pos_data.images_path,
        )
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.progress.connect(self.workerProgress)
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.delete_objects_outside_mask_worker_finished)
        self.worker.debug.connect(self.workerDebug)

        self.thread.started.connect(self.worker.run)
        self.thread.start()
