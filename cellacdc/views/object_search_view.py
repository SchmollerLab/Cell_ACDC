"""Qt view adapter for object search and navigation."""

from __future__ import annotations

from qtpy.QtCore import QEventLoop, QThread

from cellacdc import apps, html_utils, widgets, workers
from cellacdc.viewmodels.object_search_viewmodel import ObjectSearchViewModel


class ObjectSearchView:
    """Qt-facing adapter around object-search commands."""

    def __init__(self, host, view_model: ObjectSearchViewModel):
        self.host = host
        self.view_model = view_model

    def findID(self, checked=False, ID=None):
        pos_data = self.host.data[self.host.pos_i]
        if ID is None:
            search_id_dialog = apps.FindIDDialog(
                title='Search object by ID',
                msg='Enter object ID to find and highlight',
                parent=self.host,
                isInteger=True,
            )
            search_id_dialog.exec_()
            if search_id_dialog.cancel:
                return

            searched_id = search_id_dialog.EntryID
        else:
            searched_id = ID

        if searched_id in pos_data.IDs:
            self.goToObjectID(searched_id)
            return

        if pos_data.SizeT == 1:
            self.warnIDnotFound(searched_id)
            return

        if searched_id in pos_data.lost_IDs:
            self.goToLostObjectID(searched_id)
            return

        tracked_lost_ids = self.host.getTrackedLostIDs()
        if searched_id in tracked_lost_ids:
            self.goToAcceptedLostObjectID(searched_id)
            return

        self.host.logger.info(f'Searching ID {searched_id} in other frames...')

        frame_i_found = self.startSearchIDworker(searched_id)
        if frame_i_found is None:
            self.warnIDnotFound(searched_id)
            return

        self.host.logger.info(
            f'Object ID {searched_id} found at frame n. {frame_i_found+1}.'
        )
        proceed = self.askGoToFrameFoundID(searched_id, frame_i_found)
        if not proceed:
            return

        pos_data.frame_i = frame_i_found
        self.host.get_data()
        self.host.updateAllImages()
        self.host.updateScrollbars()

        self.goToObjectID(searched_id)

    def startSearchIDworker(self, searchedID):
        self.host.setDisabled(True)
        try:
            return self._startSearchIDworker(searchedID)
        finally:
            self.host.setDisabled(False)
            self.host.activateWindow()

    def _startSearchIDworker(self, searchedID):
        pos_data = self.host.data[self.host.pos_i]

        desc = 'Searching ID in all frames...'

        self.host.progressWin = apps.QDialogWorkerProgress(
            title=desc, parent=self.host.mainWin, pbarDesc=desc
        )
        self.host.progressWin.mainPbar.setMaximum(pos_data.SizeT)
        self.host.progressWin.show(self.host.app)

        self.host.searchIDthread = QThread()
        self.host.searchIDworker = workers.SimpleWorker(
            pos_data,
            self.searchIDworkerCallback,
            func_args=(searchedID,),
        )
        self.host.searchIDworker.frame_i_found = None
        self.host.searchIDworker.moveToThread(self.host.searchIDthread)

        self.host.searchIDworker.signals.finished.connect(
            self.host.searchIDthread.quit
        )
        self.host.searchIDworker.signals.finished.connect(
            self.host.searchIDworker.deleteLater
        )
        self.host.searchIDthread.finished.connect(
            self.host.searchIDthread.deleteLater
        )

        self.host.searchIDworker.signals.critical.connect(
            self.searchIDworkerCritical
        )
        self.host.searchIDworker.signals.initProgressBar.connect(
            self.host.workerInitProgressbar
        )
        self.host.searchIDworker.signals.progressBar.connect(
            self.host.workerUpdateProgressbar
        )
        self.host.searchIDworker.signals.progress.connect(
            self.host.workerProgress
        )
        self.host.searchIDworker.signals.finished.connect(
            self.searchIDworkerFinished
        )

        self.host.searchIDthread.started.connect(self.host.searchIDworker.run)
        self.host.searchIDthread.start()

        self.host.searchIDworkerLoop = QEventLoop()
        self.host.searchIDworkerLoop.exec_()

        return self.host.searchIDworker.frame_i_found

    def searchIDworkerCritical(self, error):
        self.host.searchIDworkerLoop.exit()
        self.host.workerCritical(error)

    def searchIDworkerFinished(self):
        if self.host.progressWin is not None:
            self.host.progressWin.workerFinished = True
            self.host.progressWin.close()
            self.host.progressWin = None

        self.host.searchIDworkerLoop.exit()

    def searchIDworkerCallback(self, posData, searchedID):
        self.host.searchIDworker.signals.initProgressBar.emit(0)
        self.host.setAllIDs()
        self.host.searchIDworker.signals.initProgressBar.emit(posData.SizeT)
        frame_i_found = self.view_model.find_frame_with_id(
            posData,
            searchedID,
            progress_callback=self.host.searchIDworker.signals.progressBar.emit,
        )
        self.host.searchIDworker.frame_i_found = frame_i_found

    def warnIDnotFound(self, searchedID):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            Object ID {searchedID} was not found.<br><br>
        """)
        msg.warning(self.host, f'ID {searchedID} not found', txt)

    def goToObjectID(self, ID):
        pos_data = self.host.data[self.host.pos_i]
        obj_idx = pos_data.IDs_idxs[ID]
        obj = pos_data.rp[obj_idx]
        self.host.goToZsliceSearchedID(obj)

        self.host.highlightSearchedID(ID)
        props_qgbox = self.host.guiTabControl.propsQGBox
        props_qgbox.idSB.setValue(ID)

    def goToLostObjectID(self, lostID, color=(255, 165, 0, 255)):
        pos_data = self.host.data[self.host.pos_i]
        frame_i = pos_data.frame_i
        prev_rp = pos_data.allData_li[frame_i - 1]['regionprops']
        prev_ids_idxs = pos_data.allData_li[frame_i - 1]['IDs_idxs']
        obj = prev_rp[prev_ids_idxs[lostID]]
        self.host.goToZsliceSearchedID(obj)

        image_item = self.host.getLostObjImageItem(0)
        if not hasattr(self.host, 'lostObjContoursImage'):
            self.host.initLostObjContoursImage()
        else:
            self.host.lostObjContoursImage[:] = 0

        contours = []
        obj_contours = self.host.getObjContours(obj, all_external=True)
        contours.extend(obj_contours)

        self.host.addLostObjsToLostObjImage(obj, lostID)
        self.host.drawLostObjContoursImage(
            image_item, contours, thickness=2, color=color
        )

    def goToAcceptedLostObjectID(self, acceptedLostID):
        pos_data = self.host.data[self.host.pos_i]
        frame_i = pos_data.frame_i
        prev_rp = pos_data.allData_li[frame_i - 1]['regionprops']
        prev_ids_idxs = pos_data.allData_li[frame_i - 1]['IDs_idxs']
        obj = prev_rp[prev_ids_idxs[acceptedLostID]]
        self.host.goToZsliceSearchedID(obj)

        self.host.updateLostTrackedContoursImage(
            tracked_lost_IDs=[acceptedLostID]
        )

    def askGoToFrameFoundID(self, searchedID, frame_i_found):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            Object ID {searchedID} was found at frame n. {frame_i_found+1}.<br><br>
            Do you want to go to frame n. {frame_i_found+1}.
        """)
        noButton, yesButton = msg.information(
            self.host,
            f'ID {searchedID} found at frame n. {frame_i_found+1}',
            txt,
            buttonsTexts=(
                'No, stay on current frame',
                f'Yes, go to frame n. {frame_i_found+1}',
            ),
        )
        return msg.clickedButton == yesButton

    def skipForwardToNewID(self):
        self.host.progressWin = apps.QDialogWorkerProgress(
            title='Searching the next frame with a new object',
            parent=self.host,
            pbarDesc='Searching the next frame with a new object...',
        )
        self.host.progressWin.show(self.host.app)
        self.host.progressWin.mainPbar.setMaximum(0)

        self.startFindNextNewIdWorker()

    def startFindNextNewIdWorker(self):
        pos_data = self.host.data[self.host.pos_i]
        self.host._thread = QThread()
        self.host.findNextNewIdWorker = workers.FindNextNewIdWorker(
            pos_data, self.host
        )
        self.host.findNextNewIdWorker.moveToThread(self.host._thread)

        self.host.findNextNewIdWorker.signals.finished.connect(
            self.host._thread.quit
        )
        self.host.findNextNewIdWorker.signals.finished.connect(
            self.host.findNextNewIdWorker.deleteLater
        )
        self.host._thread.finished.connect(self.host._thread.deleteLater)

        self.host.findNextNewIdWorker.signals.finished.connect(
            self.findNextNewIdWorkerFinished
        )
        self.host.findNextNewIdWorker.signals.progress.connect(
            self.host.workerProgress
        )
        self.host.findNextNewIdWorker.signals.initProgressBar.connect(
            self.host.workerInitProgressbar
        )
        self.host.findNextNewIdWorker.signals.progressBar.connect(
            self.host.workerUpdateProgressbar
        )
        self.host.findNextNewIdWorker.signals.critical.connect(
            self.host.workerCritical
        )

        self.host._thread.started.connect(self.host.findNextNewIdWorker.run)
        self.host._thread.start()

    def findNextNewIdWorkerFinished(self, next_frame_i):
        if self.host.progressWin is not None:
            self.host.progressWin.workerFinished = True
            self.host.progressWin.close()
            self.host.progressWin = None

        self.host.navSpinBox.setValue(next_frame_i + 1)
        self.host.framesScrollBarReleased()
