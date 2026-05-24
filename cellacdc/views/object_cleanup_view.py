"""View adapter for object cleanup workflows."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QThread

from cellacdc import apps, widgets, workers
from cellacdc.viewmodels.object_cleanup_viewmodel import (
    ObjectCleanupViewModel,
)


class ObjectCleanupView:
    """Qt-facing adapter around the object-cleanup view-model."""

    def __init__(self, host, view_model: ObjectCleanupViewModel):
        self.host = host
        self.view_model = view_model

    def delete_objects_outside_mask_action_triggered(self):
        pos_data = self.host.data[self.host.pos_i]
        existing_segm_endnames = self.view_model.segmentation_roi_endnames(
            basename=pos_data.basename,
            images_path=pos_data.images_path,
        )
        select_segm_win = widgets.QDialogListbox(
            'Select segmentation file',
            'Select segmentation file to use as ROI:\n',
            existing_segm_endnames,
            multiSelection=False,
            parent=self.host,
        )
        select_segm_win.exec_()
        if select_segm_win.cancel:
            self.host.logger.info('Delete objects process cancelled.')
            return

        selected_segm_endname = select_segm_win.selectedItemsText[0]
        self.start_delete_objects_outside_mask_worker(selected_segm_endname)

    def start_delete_objects_outside_mask_worker(self, selected_segm_endname):
        self.host.store_data(autosave=False)
        pos_data = self.host.data[self.host.pos_i]
        segm_data = np.squeeze(self.host.getStoredSegmData())

        self.host.progressWin = apps.QDialogWorkerProgress(
            title='Deleting objects outside of ROIs',
            parent=self.host,
            pbarDesc='Deleting objects outside of ROIs...',
        )
        self.host.progressWin.show(self.host.app)
        self.host.progressWin.mainPbar.setMaximum(0)

        self.host.thread = QThread()
        self.host.worker = workers.DelObjectsOutsideSegmROIWorker(
            selected_segm_endname,
            segm_data,
            pos_data.images_path,
        )
        self.host.worker.moveToThread(self.host.thread)
        self.host.worker.finished.connect(self.host.thread.quit)
        self.host.worker.finished.connect(self.host.worker.deleteLater)
        self.host.thread.finished.connect(self.host.thread.deleteLater)

        self.host.worker.progress.connect(self.host.workerProgress)
        self.host.worker.critical.connect(self.host.workerCritical)
        self.host.worker.finished.connect(
            self.delete_objects_outside_mask_worker_finished
        )
        self.host.worker.debug.connect(self.host.workerDebug)

        self.host.thread.started.connect(self.host.worker.run)
        self.host.thread.start()

    def delete_objects_outside_mask_worker_finished(self, result):
        pos_data = self.host.data[self.host.pos_i]
        worker, cleared_segm_data, del_ids = result
        cleared_segm_data = self.view_model.cleared_segmentation_frames(
            cleared_segm_data,
            size_t=pos_data.SizeT,
        )

        self.host.update_cca_df_deletedIDs(pos_data, del_ids)

        current_frame_i = pos_data.frame_i
        for frame_i, cleared_lab in self.view_model.frame_labels(
            cleared_segm_data
        ):
            pos_data.allData_li[frame_i]['labels'] = cleared_lab
            pos_data.frame_i = frame_i
            self.host.get_data()
            self.host.store_data(autosave=False)

        pos_data.frame_i = current_frame_i
        self.host.get_data()

        if self.host.progressWin is not None:
            self.host.progressWin.workerFinished = True
            self.host.progressWin.close()
            self.host.progressWin = None
        self.host.logger.info('Deleting objects outside of ROIs finished.')
        self.host.titleLabel.setText(
            'Deleting objects outside of ROIs finished.',
            color='w',
        )
        self.host.updateAllImages()
