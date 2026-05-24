"""Qt view adapter for object search and navigation."""

from __future__ import annotations

from collections.abc import Callable
from qtpy.QtCore import QEventLoop, QThread

from cellacdc import apps, html_utils, widgets, workers


class ObjectSearchMixin:
    """Qt-facing adapter around object-search commands."""

    """Headless object-search operations."""

    def _startSearchIDworker(self, searchedID):
        pos_data = self.data[self.pos_i]

        desc = "Searching ID in all frames..."

        self.progressWin = apps.QDialogWorkerProgress(
            title=desc, parent=self.mainWin, pbarDesc=desc
        )
        self.progressWin.mainPbar.setMaximum(pos_data.SizeT)
        self.progressWin.show(self.app)

        self.searchIDthread = QThread()
        self.searchIDworker = workers.SimpleWorker(
            pos_data,
            self.searchIDworkerCallback,
            func_args=(searchedID,),
        )
        self.searchIDworker.frame_i_found = None
        self.searchIDworker.moveToThread(self.searchIDthread)

        self.searchIDworker.signals.finished.connect(self.searchIDthread.quit)
        self.searchIDworker.signals.finished.connect(self.searchIDworker.deleteLater)
        self.searchIDthread.finished.connect(self.searchIDthread.deleteLater)

        self.searchIDworker.signals.critical.connect(self.searchIDworkerCritical)
        self.searchIDworker.signals.initProgressBar.connect(self.workerInitProgressbar)
        self.searchIDworker.signals.progressBar.connect(self.workerUpdateProgressbar)
        self.searchIDworker.signals.progress.connect(self.workerProgress)
        self.searchIDworker.signals.finished.connect(self.searchIDworkerFinished)

        self.searchIDthread.started.connect(self.searchIDworker.run)
        self.searchIDthread.start()

        self.searchIDworkerLoop = QEventLoop()
        self.searchIDworkerLoop.exec_()

        return self.searchIDworker.frame_i_found

    def askGoToFrameFoundID(self, searchedID, frame_i_found):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            Object ID {searchedID} was found at frame n. {frame_i_found + 1}.<br><br>
            Do you want to go to frame n. {frame_i_found + 1}.
        """)
        noButton, yesButton = msg.information(
            self,
            f"ID {searchedID} found at frame n. {frame_i_found + 1}",
            txt,
            buttonsTexts=(
                "No, stay on current frame",
                f"Yes, go to frame n. {frame_i_found + 1}",
            ),
        )
        return msg.clickedButton == yesButton

    def findID(self, checked=False, ID=None):
        posData = self.data[self.pos_i]
        if ID is None:
            searchIDdialog = apps.FindIDDialog(
                title="Search object by ID",
                msg="Enter object ID to find and highlight",
                parent=self,
                isInteger=True,
            )
            searchIDdialog.exec_()
            if searchIDdialog.cancel:
                return

            searchedID = searchIDdialog.EntryID
        else:
            searchedID = ID

        if searchedID in posData.IDs:
            self.goToObjectID(searchedID)
            return

        if posData.SizeT == 1:
            self.warnIDnotFound(searchedID)
            return

        if searchedID in posData.lost_IDs:
            self.goToLostObjectID(searchedID)
            return

        tracked_lost_IDs = self.getTrackedLostIDs()
        if searchedID in tracked_lost_IDs:
            self.goToAcceptedLostObjectID(searchedID)
            return

        self.logger.info(f"Searching ID {searchedID} in other frames...")

        frame_i_found = self.startSearchIDworker(searchedID)
        if frame_i_found is None:
            self.warnIDnotFound(searchedID)
            return

        self.logger.info(
            f"Object ID {searchedID} found at frame n. {frame_i_found + 1}."
        )
        proceed = self.askGoToFrameFoundID(searchedID, frame_i_found)
        if not proceed:
            return

        posData.frame_i = frame_i_found
        self.get_data()
        self.updateAllImages()
        self.updateScrollbars()

        self.goToObjectID(searchedID)

    def findNextNewIdWorkerFinished(self, next_frame_i):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None

        self.navSpinBox.setValue(next_frame_i + 1)
        self.framesScrollBarReleased()

    def find_frame_with_id(
        self,
        pos_data,
        searched_id: int,
        *,
        progress_callback: Callable[[int], None] | None = None,
    ) -> int | None:
        return find_frame_with_id(
            pos_data.segm_data,
            pos_data.allData_li,
            searched_id,
            progress_callback=progress_callback,
        )

    def goToAcceptedLostObjectID(self, acceptedLostID):
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        prev_rp = posData.allData_li[frame_i - 1]["regionprops"]
        prev_IDs_idxs = posData.allData_li[frame_i - 1]["IDs_idxs"]
        obj = prev_rp[prev_IDs_idxs[acceptedLostID]]
        self.goToZsliceSearchedID(obj)

        self.updateLostTrackedContoursImage(tracked_lost_IDs=[acceptedLostID])

    def goToLostObjectID(self, lostID, color=(255, 165, 0, 255)):
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        prev_rp = posData.allData_li[frame_i - 1]["regionprops"]
        prev_IDs_idxs = posData.allData_li[frame_i - 1]["IDs_idxs"]
        obj = prev_rp[prev_IDs_idxs[lostID]]
        self.goToZsliceSearchedID(obj)

        imageItem = self.getLostObjImageItem(0)
        if not hasattr(self, "lostObjContoursImage"):
            self.initLostObjContoursImage()
        else:
            self.lostObjContoursImage[:] = 0

        contours = []
        obj_contours = self.getObjContours(obj, all_external=True)
        contours.extend(obj_contours)

        self.addLostObjsToLostObjImage(obj, lostID)
        self.drawLostObjContoursImage(imageItem, contours, thickness=2, color=color)

    def goToObjectID(self, ID):
        posData = self.data[self.pos_i]
        objIdx = posData.IDs_idxs[ID]
        obj = posData.rp[objIdx]
        self.goToZsliceSearchedID(obj)

        self.highlightSearchedID(ID)
        propsQGBox = self.guiTabControl.propsQGBox
        propsQGBox.idSB.setValue(ID)

    def searchIDworkerCallback(self, posData, searchedID):
        self.searchIDworker.signals.initProgressBar.emit(0)
        self.setAllIDs()
        self.searchIDworker.signals.initProgressBar.emit(posData.SizeT)
        frame_i_found = None
        for frame_i in range(len(posData.segm_data)):
            if frame_i >= len(posData.allData_li):
                break
            lab = posData.allData_li[frame_i]["labels"]
            if lab is None:
                rp = skimage.measure.regionprops(posData.segm_data[frame_i])
                IDs = set([obj.label for obj in rp])
            else:
                IDs = posData.allData_li[frame_i]["IDs"]

            if searchedID in IDs:
                frame_i_found = frame_i
                break

            self.searchIDworker.signals.progressBar.emit(1)

        self.searchIDworker.frame_i_found = frame_i_found

    def searchIDworkerCritical(self, error):
        self.searchIDworkerLoop.exit()
        self.workerCritical(error)

    def searchIDworkerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None

        self.searchIDworkerLoop.exit()

    def skipForwardToNewID(self):
        self.progressWin = apps.QDialogWorkerProgress(
            title="Searching the next frame with a new object",
            parent=self,
            pbarDesc="Searching the next frame with a new object...",
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)

        self.startFindNextNewIdWorker()

    def startFindNextNewIdWorker(self):
        posData = self.data[self.pos_i]
        self._thread = QThread()
        self.findNextNewIdWorker = workers.FindNextNewIdWorker(posData, self)
        self.findNextNewIdWorker.moveToThread(self._thread)

        self.findNextNewIdWorker.signals.finished.connect(self._thread.quit)
        self.findNextNewIdWorker.signals.finished.connect(
            self.findNextNewIdWorker.deleteLater
        )
        self._thread.finished.connect(self._thread.deleteLater)

        self.findNextNewIdWorker.signals.finished.connect(
            self.findNextNewIdWorkerFinished
        )
        self.findNextNewIdWorker.signals.progress.connect(self.workerProgress)
        self.findNextNewIdWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.findNextNewIdWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.findNextNewIdWorker.signals.critical.connect(self.workerCritical)

        self._thread.started.connect(self.findNextNewIdWorker.run)
        self._thread.start()

    @disableWindow
    def startSearchIDworker(self, searchedID):
        posData = self.data[self.pos_i]

        desc = "Searching ID in all frames..."

        self.progressWin = apps.QDialogWorkerProgress(
            title=desc, parent=self.mainWin, pbarDesc=desc
        )
        self.progressWin.mainPbar.setMaximum(posData.SizeT)
        self.progressWin.show(self.app)

        self.searchIDthread = QThread()
        self.searchIDworker = workers.SimpleWorker(
            posData, self.searchIDworkerCallback, func_args=(searchedID,)
        )
        self.searchIDworker.frame_i_found = None
        self.searchIDworker.moveToThread(self.searchIDthread)

        self.searchIDworker.signals.finished.connect(self.searchIDthread.quit)
        self.searchIDworker.signals.finished.connect(self.searchIDworker.deleteLater)
        self.searchIDthread.finished.connect(self.searchIDthread.deleteLater)

        self.searchIDworker.signals.critical.connect(self.searchIDworkerCritical)
        self.searchIDworker.signals.initProgressBar.connect(self.workerInitProgressbar)
        self.searchIDworker.signals.progressBar.connect(self.workerUpdateProgressbar)
        self.searchIDworker.signals.progress.connect(self.workerProgress)
        self.searchIDworker.signals.finished.connect(self.searchIDworkerFinished)

        self.searchIDthread.started.connect(self.searchIDworker.run)
        self.searchIDthread.start()

        self.searchIDworkerLoop = QEventLoop()
        self.searchIDworkerLoop.exec_()

        return self.searchIDworker.frame_i_found

    def warnIDnotFound(self, searchedID):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            Object ID {searchedID} was not found.<br><br>
        """)
        msg.warning(self, f"ID {searchedID} not found", txt)
