"""Qt view adapter for GUI worker lifecycle handling."""

from __future__ import annotations

import logging
import traceback
from functools import partial
from typing import Tuple

from qtpy.QtCore import QObject, QMutex, QThread, QTimer, QWaitCondition

from cellacdc import apps, exception_handler, html_utils, issues_url, widgets, workers


class Worker:
    """Extracted from guiWin."""

    def autoSaveWorkerClosed(self, worker):
        if self.autoSaveActiveWorkers:
            self.logger.info('Autosaving worker closed.')
            try:
                self.autoSaveActiveWorkers.remove(worker)
            except Exception as e:
                pass

    def autoSaveWorkerDone(self):
        self.setStatusBarLabel(log=False)

    def autoSaveWorkerStartTimer(self, worker, posData):
        self.autoSaveWorkerTimer = QTimer()
        self.autoSaveWorkerTimer.timeout.connect(
            partial(self.autoSaveWorkerTimerCallback, worker, posData)
        )
        self.autoSaveWorkerTimer.start(150)

    def autoSaveWorkerTimerCallback(self, worker, posData):
        if not self.isSaving:
            self.autoSaveWorkerTimer.stop()
            worker._enqueue(posData)

    def ccaIntegrityWorkerCritical(self, error):
        try:
            raise error
        except Exception as err:
            self.logger.exception(traceback.format_exc())
        
        href = f'<a href="{issues_url}">GitHub page</a>'
        txt = html_utils.paragraph(f"""
            Unfortunately the experimental feature 
            <code>check cell cycle annotations integrity</code> raised a 
            critical error.<br><br>
            Cell-ACDC will now disable this feature to allow you to keep 
            using the software.<br><br>
            However, <b>we kindly ask you to report the issue</b> on our 
            {href}, thank you very much!<br><br>
            Please, <b>include the log file when reporting the issue</b>.<br><br>
            Log file location:
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Experimental feature error', txt,
            commands=(self.log_path,),
            path_to_browse=self.logs_path
        )
        self.disableCcaIntegrityChecker()

    def gui_createAutoSaveWorker(self):        
        if not hasattr(self, 'data'):
            return
        
        if not self.isDataLoaded:
            return 
        
        if self.autoSaveActiveWorkers:
            garbage = self.autoSaveActiveWorkers[-1]
            self.autoSaveGarbageWorkers.append(garbage)
            worker = garbage[0]
            worker._stop()

        posData = self.data[self.pos_i]
        autoSaveThread = QThread()
        self.autoSaveMutex = QMutex()
        self.autoSaveWaitCond = QWaitCondition()

        savedSegmData = posData.segm_data.copy()
        autoSaveWorker = workers.AutoSaveWorker(
            self.autoSaveMutex, self.autoSaveWaitCond, savedSegmData
        )
        autoSaveWorker.isAutoSaveON = self.autoSaveToggle.isChecked()

        autoSaveWorker.moveToThread(autoSaveThread)
        autoSaveWorker.finished.connect(autoSaveThread.quit)
        autoSaveWorker.finished.connect(autoSaveWorker.deleteLater)
        autoSaveThread.finished.connect(autoSaveThread.deleteLater)

        autoSaveWorker.sigDone.connect(self.autoSaveWorkerDone)
        autoSaveWorker.progress.connect(self.workerProgress)
        autoSaveWorker.finished.connect(self.autoSaveWorkerClosed)
        autoSaveWorker.sigAutoSaveCannotProceed.connect(
            self.turnOffAutoSaveWorker
        )
        
        autoSaveThread.started.connect(autoSaveWorker.run)
        autoSaveThread.start()

        self.autoSaveActiveWorkers.append((autoSaveWorker, autoSaveThread))

        self.logger.info('Autosaving worker started.')

    def gui_createLazyLoader(self):
        if not self.lazyLoader is None:
            return

        self.lazyLoaderThread = QThread()
        self.lazyLoaderMutex = QMutex()
        self.lazyLoaderWaitCond = QWaitCondition()
        self.waitReadH5cond = QWaitCondition()
        self.readH5mutex = QMutex()
        self.lazyLoader = workers.LazyLoader(
            self.lazyLoaderMutex, self.lazyLoaderWaitCond, 
            self.waitReadH5cond, self.readH5mutex
        )
        self.lazyLoader.moveToThread(self.lazyLoaderThread)
        self.lazyLoader.wait = True

        self.lazyLoader.signals.finished.connect(self.lazyLoaderThread.quit)
        self.lazyLoader.signals.finished.connect(self.lazyLoader.deleteLater)
        self.lazyLoaderThread.finished.connect(self.lazyLoaderThread.deleteLater)

        self.lazyLoader.signals.progress.connect(self.workerProgress)
        self.lazyLoader.signals.sigLoadingNewChunk.connect(self.loadingNewChunk)
        self.lazyLoader.sigLoadingFinished.connect(self.lazyLoaderFinished)
        self.lazyLoader.signals.critical.connect(self.lazyLoaderCritical)
        self.lazyLoader.signals.finished.connect(self.lazyLoaderWorkerClosed)

        self.lazyLoaderThread.started.connect(self.lazyLoader.run)
        self.lazyLoaderThread.start()

    def gui_createStoreStateWorker(self):
        self.storeStateWorker = None
        return
        self.storeStateThread = QThread()
        self.autoSaveMutex = QMutex()
        self.autoSaveWaitCond = QWaitCondition()

        self.storeStateWorker = workers.StoreGuiStateWorker(
            self.autoSaveMutex, self.autoSaveWaitCond
        )

        self.storeStateWorker.moveToThread(self.storeStateThread)
        self.storeStateWorker.finished.connect(self.storeStateThread.quit)
        self.storeStateWorker.finished.connect(self.storeStateWorker.deleteLater)
        self.storeStateThread.finished.connect(self.storeStateThread.deleteLater)

        self.storeStateWorker.sigDone.connect(self.storeStateWorkerDone)
        self.storeStateWorker.progress.connect(self.workerProgress)
        self.storeStateWorker.finished.connect(self.storeStateWorkerClosed)
        
        self.storeStateThread.started.connect(self.storeStateWorker.run)
        self.storeStateThread.start()

        self.logger.info('Store state worker started.')

    def lazyLoaderCritical(self, error):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
            self.lazyLoader.pause()
        raise error

    def lazyLoaderFinished(self):
        self.logger.info('Load chunk data worker done.')
        if self.lazyLoader.updateImgOnFinished:
            self.updateAllImages()

        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None

    def lazyLoaderWorkerClosed(self):
        if self.lazyLoader.salute:
            self.logger.info('Cell-ACDC GUI closed.')     
            self.sigClosed.emit(self)
        
        self.lazyLoader = None

    def loadingNewChunk(self, chunk_range):
        coord0_chunk, coord1_chunk = chunk_range
        desc = (
            f'Loading new window, range = ({coord0_chunk}, {coord1_chunk})...'
        )
        self.progressWin = apps.QDialogWorkerProgress(
            title='Loading data...', parent=self, pbarDesc=desc
        )
        self.progressWin.mainPbar.setMaximum(0)
        self.progressWin.show(self.app)

    def relabelWorkerFinished(self):
        self.updateAllImages()

    def saveDataWorkerCritical(self, error):
        self.logger.warning(
            'Saving process stopped because of critical error.'
        )
        self.saveWin.aborted = True
        self.worker.finished.emit()
        self.workerCritical(error)

    def savePreprocWorkerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        
        self.setStatusBarLabel()
        self.logger.info('Pre-processed data saved!')
        self.titleLabel.setText('Pre-processed data saved!', color='w')

    def startPostProcessSegmWorker(
            self, postProcessKwargs, customPostProcessGroupedFeatures, 
            customPostProcessFeatures
        ):
        self.thread = QThread()
        self.postProcessWorker = workers.PostProcessSegmWorker(
            postProcessKwargs, customPostProcessGroupedFeatures, 
            customPostProcessFeatures, self
        )
        
        self.postProcessWorker.moveToThread(self.thread)
        self.postProcessWorker.signals.finished.connect(self.thread.quit)
        self.postProcessWorker.signals.finished.connect(
            self.postProcessWorker.deleteLater
        )
        self.thread.finished.connect(self.thread.deleteLater)

        self.postProcessWorker.signals.finished.connect(
            self.postProcessSegmWorkerFinished
        )
        self.postProcessWorker.signals.progress.connect(self.workerProgress)
        self.postProcessWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.postProcessWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.postProcessWorker.signals.critical.connect(
            self.workerCritical
        )

        self.thread.started.connect(self.postProcessWorker.run)
        self.thread.start()

    def startRelabellingWorker(self, posFoldernames):
        self.thread = QThread()
        self.worker = workers.relabelSequentialWorker(self, posFoldernames)
        self.worker.moveToThread(self.thread)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        self.worker.progress.connect(self.workerProgress)
        self.worker.critical.connect(self.workerCritical)
        self.worker.finished.connect(self.workerFinished)
        self.worker.finished.connect(self.relabelWorkerFinished)

        self.worker.debug.connect(self.workerDebug)

        self.thread.started.connect(self.worker.run)
        self.thread.start()

    def startTrackingWorker(self, posData, video_to_track):
        self.thread = QThread()
        self.trackingWorker = workers.trackingWorker(
            posData, self, video_to_track
        )
        self.trackingWorker.moveToThread(self.thread)
        self.trackingWorker.finished.connect(self.thread.quit)
        self.trackingWorker.finished.connect(self.trackingWorker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Custom signals
        self.trackingWorker.signals.progress = self.trackingWorker.progress
        self.trackingWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.trackingWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.trackingWorker.signals.sigInitInnerPbar.connect(
            self.workerInitInnerPbar
        )
        self.trackingWorker.progress.connect(self.workerProgress)
        self.trackingWorker.critical.connect(self.workerCritical)
        self.trackingWorker.finished.connect(self.trackingWorkerFinished)

        self.trackingWorker.debug.connect(self.workerDebug)

        self.thread.started.connect(self.trackingWorker.run)
        self.thread.start()

    def storeStateWorkerClosed(self):
        self.logger.info('Store state worker started.')

    def storeStateWorkerDone(self):
        if self.storeStateWorker.callbackOnDone is not None:
            self.storeStateWorker.callbackOnDone()
        self.storeStateWorker.callbackOnDone = None

    def trackingWorkerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        self.logger.info('Worker process ended.')
        askDisableRealTimeTracking = (
            self.trackingWorker.trackingOnNeverVisitedFrames
            and self.realTimeTrackingToggle.isChecked()
        )
        if askDisableRealTimeTracking:
            msg = widgets.myMessageBox()
            title = 'Disable real-time tracking?'
            txt = (
                'You perfomed tracking on frames that you have '
                '<b>never visited.</b><br><br>'
                'Cell-ACDC default behaviour is to <b>track them again</b> when you '
                'will visit them.<br><br>'
                'However, you can <b>overwrite this behaviour</b> and explicitly '
                'disable tracking for all of the frames you already tracked.<br><br>'
                'NOTE: you can reactivate real-time tracking by clicking on the '
                '"Reset last segmented frame" button on the top toolbar.<br><br>'
                'What do you want me to do?'
            )
            _, disableTrackingButton = msg.information(
                self, title, html_utils.paragraph(txt),
                buttonsTexts=(
                    'Keep real-time tracking active (recommended)',
                    'Disable real-time tracking'
                )
            )
            if msg.clickedButton == disableTrackingButton:
                self.logger.info('Disabling real time tracking...')
                self.realTimeTrackingToggle.setChecked(False)
                # posData = self.data[self.pos_i]
                # current_frame_i = posData.frame_i
                # for frame_i in range(self.start_n-1, self.stop_n):
                #     posData.frame_i = frame_i
                #     self.get_data()
                #     self.store_data(autosave=frame_i==self.stop_n-1)
                # posData.last_tracked_i = frame_i
                # self.setNavigateScrollBarMaximum()

                # # Back to current frame
                # posData.frame_i = current_frame_i
                # self.get_data()
        posData = self.data[self.pos_i]
        self.updateAllImages()
        self.titleLabel.setText('Done', color='w')

    def workerCritical(self, out: Tuple[QObject, Exception]):
        self.setDisabled(False)
        try:
            worker, error = out
        except TypeError as err:
            error = out
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        self.logger.info(error)
        try:
            worker.thread().quit()
            worker.deleteLater()
            worker.thread().deleteLater()
        except Exception as err:
            # Worker already closed
            pass
        raise error

    def workerDebug(self, item):
        tracked_video, worker = item
        from cellacdc.plot import imshow
        imshow(tracked_video)
        worker.waitCond.wakeAll()

    def workerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        self.logger.info('Worker process ended.')
        self.updateAllImages()
        self.titleLabel.setText('Done', color='w')

    def workerInitInnerPbar(self, totalIter):
        self.progressWin.innerPbar.setValue(0)
        if totalIter == 1:
            totalIter = 0
        self.progressWin.innerPbar.setMaximum(totalIter)

    def workerInitProgressbar(self, totalIter):
        self.progressWin.mainPbar.setValue(0)
        if totalIter == 1:
            totalIter = 0
        self.progressWin.mainPbar.setMaximum(totalIter)

    def workerLog(self, text):
        self.logger.info(text)

    def workerProgress(self, text, loggerLevel='INFO'): # used in cca and lin tree
        if self.progressWin is not None:
            self.progressWin.logConsole.append(text)
        self.logger.log(getattr(logging, loggerLevel), text)

    def workerUpdateInnerPbar(self, step):
        self.progressWin.innerPbar.update(step)

    def workerUpdateProgressbar(self, step):
        self.progressWin.mainPbar.update(step)
