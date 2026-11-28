"""Cell cycle annotation tables of the main GUI: automatic annotation,
cca_df bookkeeping, integrity checker and annotations after segmentation
edits. Mixins of guiWin (gui/main_window/window.py) only: the
measurements viewer never composes them.
"""
import traceback
import uuid
from tqdm import tqdm
from typing import Tuple
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.interpolate
import scipy.ndimage
from qtpy.QtCore import QObject, QThread, QMutex, QWaitCondition
from qtpy.QtWidgets import QPushButton, QCheckBox, QMessageBox
from cellacdc import exception_handler, disableWindow
from cellacdc import base_cca_dict
from cellacdc import cca_df_colnames
from cellacdc import load, html_utils
from ..dialogs import editing as gui_dialogs_editing
from ..workers import annotation_checks as gui_workers_annotation_checks
from cellacdc import core
from cellacdc.shared.widgets import message_box as widgets_message_box
from cellacdc import _warnings, issues_url
from cellacdc import printl
from cellacdc.trackers.CellACDC_normal_division.CellACDC_normal_division_tracker import has_lineage_tree_values
from . import constants as gui_main_constants


class CcaIntegrityCheckerGuiElements:
    """Integrity checker worker of the cell cycle annotations (threads that
    did not stop in time are kept in CCA_CHECKER_ORPHAN_THREADS of
    gui/main_window/constants.py).
    """

    def ccaCheckerWorkerDone(self):
        if not hasattr(self, 'data'):
            # Signal delivered after closing the window (data cleared)
            return
        self.setStatusBarLabel(log=False)

    def warnCcaIntegrity(self, txt, category):
        self.logger.warning(f'{html_utils.to_plain_text(txt)}')
        
        if 'disable_all' in self.disabled_cca_warnings:
            return
        
        if category in self.disabled_cca_warnings:
            return
        
        if txt in self.disabled_cca_warnings:
            return
        
        if self.isWarningCcaIntegrity:
            # Some other warning is still open --> avoid opening another one
            return
        
        self.isWarningCcaIntegrity = True
        disabled_warning = _warnings.warn_cca_integrity(
            txt, category, self, 
            go_to_frame_callback=self.goToFrameNumber
        )
        if disabled_warning:
            self.disabled_cca_warnings.add(disabled_warning)
        
        self.isWarningCcaIntegrity = False

    def fixWillDivide(self, warning_txt, IDs_will_divide_wrong):
        self.logger.info(warning_txt)
        self.logger.info('Fixing `will_divide` information...')
        
        global_cca_df = self.getConcatCcaDf()
        global_cca_df = (
            global_cca_df.reset_index()
            .set_index(['Cell_ID', 'generation_num'])
        )
        global_cca_df.loc[IDs_will_divide_wrong, 'will_divide'] = 0
        global_cca_df = (
            global_cca_df.reset_index()
            .set_index(['frame_i', 'Cell_ID'])
        )
        self.storeFromConcatCcaDf(global_cca_df)

    def ccaCheckerWorkerClosed(self, worker):
        self.logger.info('Cell cycle annotations integrity checker stopped.') 
        if worker is not getattr(self, 'ccaIntegrityCheckerWorker', None):
            # Previous worker that finished after a new one was started
            # --> do not mark the new one as not running
            return
        self.ccaCheckerRunning = False           

    def ccaIntegrityWorkerCritical(self, out: Tuple[QObject, Exception]):
        worker, error = out
        traceback_str = ''.join(traceback.format_exception(
            type(error), error, error.__traceback__
        ))
        self.logger.error(traceback_str)

        if worker is not getattr(self, 'ccaIntegrityCheckerWorker', None):
            # Error from a previous worker that was already replaced
            # --> do not stop the current one
            return
        
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
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.warning(
            self, 'Experimental feature error', txt,
            commands=(self.log_path,),
            path_to_browse=self.logs_path
        )
        self.disableCcaIntegrityChecker()

        # Switch the toggle off for this session (without saving the
        # setting), as stated in the message above. Otherwise the checker
        # would be restarted (and fail again) every time the user enters
        # the "Cell cycle analysis" mode
        self.ccaIntegrCheckerToggle.blockSignals(True)
        self.ccaIntegrCheckerToggle.setChecked(False)
        self.ccaIntegrCheckerToggle.blockSignals(False)

    def isCcaCheckerChecking(self):
        if not self.ccaCheckerRunning:
            return False
        
        return self.ccaIntegrityCheckerWorker.isChecking

    def ccaCheckerStopChecking(self):
        if not self.ccaCheckerRunning:
            return
        
        self.ccaIntegrityCheckerWorker.clearQueue()
        
        if self.ccaIntegrityCheckerWorker.isChecking:
            self.ccaIntegrityCheckerWorker.abortChecking = True

    def store_cca_df_checker(self, posData, frame_i, cca_df):
        if not self.ccaCheckerRunning:
            return
        
        if cca_df is None:
            return
        
        posData.allData_li[frame_i]['cca_df_checker'] = cca_df.copy()

    def enqCcaIntegrityChecker(self):
        if not self.ccaCheckerRunning:
            return
        posData = self.data[self.pos_i]  
        self.ccaIntegrityCheckerWorker.enqueue(posData)

    def startCcaIntegrityCheckerWorker(self):
        if not hasattr(self, 'data'):
            return
        
        if not self.isDataLoaded:
            return
        
        if not self.ccaIntegrCheckerToggle.isChecked():
            return
        
        # The previous checker thread must be finished before we replace
        # it, because destroying a running QThread aborts the process
        # (GitHub issue #1299)
        self.stopCcaIntegrityCheckerWorker(wait=True)

        ccaCheckerThread = QThread()
        self.ccaCheckerMutex = QMutex()
        self.ccaCheckerWaitCond = QWaitCondition()
        
        worker = gui_workers_annotation_checks.CcaIntegrityCheckerWorker(
            self.ccaCheckerMutex, self.ccaCheckerWaitCond
        )
        self.ccaIntegrityCheckerWorker = worker
        self.ccaCheckerThread = ccaCheckerThread
        
        worker.moveToThread(ccaCheckerThread)
        worker.finished.connect(ccaCheckerThread.quit)
        worker.finished.connect(worker.deleteLater)
        ccaCheckerThread.finished.connect(ccaCheckerThread.deleteLater)

        worker.sigDone.connect(self.ccaCheckerWorkerDone)
        worker.progress.connect(self.workerProgress)
        worker.critical.connect(self.ccaIntegrityWorkerCritical)
        worker.finished.connect(self.ccaCheckerWorkerClosed)
        worker.sigWarning.connect(self.warnCcaIntegrity)
        worker.sigFixWillDivide.connect(self.fixWillDivide)
        
        ccaCheckerThread.started.connect(worker.run)
        ccaCheckerThread.start()
        
        self.ccaCheckerRunning = True
        
        self.initCcaIntegrityChecker()
        
        self.logger.info('Cell cycle annotations integrity checker started.')

    def initCcaIntegrityChecker(self):
        posData = self.data[self.pos_i]
        for frame_i, data_frame_i in enumerate(posData.allData_li):
            lab = data_frame_i['labels']
            if lab is None:
                break
            
            cca_df = self.get_cca_df(frame_i, return_df=True)
            self.store_cca_df_checker(posData, frame_i, cca_df)
        
        self.enqCcaIntegrityChecker()

    def disableCcaIntegrityChecker(self):
        self.stopCcaIntegrityCheckerWorker()

    def stopCcaIntegrityCheckerWorker(self, wait=False, timeout_ms=5000):
        """Stop the cell cycle annotations integrity checker worker and quit
        its thread.

        Parameters
        ----------
        wait : bool, optional
            If True, block until the thread has finished (at most
            `timeout_ms`). Required before replacing or releasing the
            thread, because destroying a QThread that is still running
            aborts the process (GitHub issue #1299). Default is False
        timeout_ms : int, optional
            Maximum time in milliseconds to wait for the thread. If the
            thread is still running after that, a reference to it is kept
            to prevent its destruction. Default is 5000
        """
        # Do not enqueue new checks to the stopping worker
        self.ccaCheckerRunning = False

        worker = getattr(self, 'ccaIntegrityCheckerWorker', None)
        if worker is not None:
            try:
                worker.abort()
            except Exception as err:
                pass

        if wait:
            self.waitCcaCheckerOrphanThreads(timeout_ms=0)

        thread = getattr(self, 'ccaCheckerThread', None)
        if thread is None:
            return

        try:
            # Quit the thread's event loop also when `worker.finished`
            # (connected to `thread.quit`) was not emitted
            thread.quit()
            if not wait:
                return

            # Always wait, without checking `isRunning` first: `isRunning`
            # is already False while the thread is still finishing and
            # releasing the thread at that moment deadlocks. `wait` releases
            # the GIL and returns immediately if the thread is finished
            if thread.wait(timeout_ms):
                return
        except RuntimeError:
            # The C++ thread was already deleted (finished --> deleteLater)
            return

        self.logger.info(
            '[WARNING]: Cell cycle annotations integrity checker did not '
            f'stop within {timeout_ms} ms.'
        )
        # Keep a reference to prevent destroying a running thread
        gui_main_constants.CCA_CHECKER_ORPHAN_THREADS.append(thread)

    def waitCcaCheckerOrphanThreads(self, timeout_ms=5000):
        """Wait for the checker threads that did not stop in time (see
        `stopCcaIntegrityCheckerWorker`) and release the finished ones.

        Threads are released only after `wait` returned True. Releasing
        them earlier (e.g., from a slot connected to `finished`) could
        destroy them while they are still finishing.
        """
        stillRunning = []
        for thread in gui_main_constants.CCA_CHECKER_ORPHAN_THREADS:
            try:
                thread.quit()
                if not thread.wait(timeout_ms):
                    stillRunning.append(thread)
            except RuntimeError:
                # Thread already deleted
                pass
        gui_main_constants.CCA_CHECKER_ORPHAN_THREADS[:] = stillRunning

    def ccaIntegrCheckerToggled(self, checked):
        self.df_settings.at['is_cca_integrity_checker_activated', 'value'] = (
            int(checked)
        )
        self.df_settings.to_csv(self.settings_csv_path)
        mode = self.modeComboBox.currentText()
        if mode != 'Cell cycle analysis':
            return
        
        if checked:
            self.startCcaIntegrityCheckerWorker()
        else:
            self.disableCcaIntegrityChecker()


class CcaDataGuiElements:
    """Cell cycle annotation tables: automatic annotation (autoCca_df,
    initCca), get/store of cca_df and resets of future annotations.
    """

    def reInitCca(self):
        if not self.isSnapshot:
            txt = html_utils.paragraph(
                'If you decide to continue <b>ALL cell cycle annotations</b> from '
                'this frame to the end will be <b>erased from current session</b> '
                '(saved data is not touched of course).<br><br>'
                'To annotate future frames again you will have to revisit them.<br><br>'
                'Do you want to continue?'
            )
            msg = widgets_message_box.myMessageBox()
            msg.warning(
               self, 'Re-initialize annnotations?', txt, 
               buttonsTexts=('Cancel', 'Yes')
            )
            posData = self.data[self.pos_i]
            if msg.cancel:
                return
            
            # Reset all future frames
            self.resetCcaFuture(posData.frame_i+1)
            self.resetSingleMotherBudPairsCcaInfo(posData.frame_i+1)
            self.resetWillDivideInfo()
            if posData.frame_i == 0:
                # Reset everything since we are on first frame
                posData.cca_df = self.getBaseCca_df()
                self.store_data()
            self.updateAllImages()
            self.navigateScrollBar.setMaximum(posData.frame_i+1)
            self.navSpinBox.setMaximum(posData.frame_i+1)
        else:
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)

            posData = self.data[self.pos_i]
            posData.cca_df = self.getBaseCca_df()
            self.store_data()
            self.updateAllImages()        


    def repeatAutoCca(self):
        # Do not allow automatic bud assignment if there are future
        # frames that already contain anotations
        posData = self.data[self.pos_i]
        next_df = posData.allData_li[posData.frame_i+1]['acdc_df']
        if next_df is not None:
            if 'cell_cycle_stage' in next_df.columns:
                msg = QMessageBox()
                warn_cca = msg.critical(
                    self, 'Future visited frames detected!',
                    'Automatic bud assignment CANNOT be performed becasue '
                    'there are future frames that already contain cell cycle '
                    'annotations. The behaviour in this case cannot be predicted.\n\n'
                    'We suggest assigning the bud manually OR use the '
                    '"Re-initialize cell cycle annotations" button which properly '
                    're-initialize future frames.',
                    msg.Ok
                )
                return

        correctedAssignIDs = (
            posData.cca_df[posData.cca_df['corrected_on_frame_i']>=0].index
        )
        NeverCorrectedAssignIDs = [
            ID for ID in posData.new_IDs if ID not in correctedAssignIDs
        ]

        # Store cca_df temporarily if attempt_auto_cca fails
        posData.cca_df_beforeRepeat = posData.cca_df.copy()

        if not all(NeverCorrectedAssignIDs):
            notEnoughG1Cells, proceed = self.attempt_auto_cca()
            if notEnoughG1Cells or not proceed:
                posData.cca_df = posData.cca_df_beforeRepeat
            else:
                self.updateAllImages()
            return

        msg = QMessageBox()
        msg.setIcon(msg.Question)
        msg.setText(
            'Do you want to automatically assign buds to mother cells for '
            'ALL the new cells in this frame (excluding cells with unknown history) '
            'OR only the cells where you never clicked on?'
        )
        msg.setDetailedText(
            f'New cells that you never touched:\n\n{NeverCorrectedAssignIDs}')
        enforceAllButton = QPushButton('ALL new cells')
        b = QPushButton('Only cells that I never corrected assignment')
        msg.addButton(b, msg.YesRole)
        msg.addButton(enforceAllButton, msg.NoRole)
        msg.exec_()
        if msg.clickedButton() == enforceAllButton:
            notEnoughG1Cells, proceed = self.attempt_auto_cca(enforceAll=True)
        else:
            notEnoughG1Cells, proceed = self.attempt_auto_cca()
        if notEnoughG1Cells or not proceed:
            posData.cca_df = posData.cca_df_beforeRepeat
        else:
            self.updateAllImages()

    def askInitCcaFirstFrame(self):
        mode = str(self.modeComboBox.currentText())
        if mode != 'Cell cycle analysis':
            return True
        
        posData = self.data[self.pos_i]
        if posData.frame_i != 0:
            return True
        
        self.navigateScrollBar.setDisabled(True)
        self.navSpinBox.setDisabled(True)

        self.editCcaWidget = gui_dialogs_editing.editCcaTableWidget(
            posData.cca_df, posData.SizeT, parent=self,
            title='Initialize cell cycle annotations'
        )
        self.editCcaWidget.sigApplyChangesFutureFrames.connect(
            self.applyManualCcaChangesFutureFrames
        )
        self.editCcaWidget.exec_()
        if self.editCcaWidget.cancel:
            self.resetNavigateFramesScrollbar()
            self.editCcaWidget = None
            self.navigateScrollBar.setDisabled(False)
            self.navSpinBox.setDisabled(False)
            return False
        
        if posData.cca_df is not None:
            is_cca_same_as_stored = (
                (posData.cca_df == self.editCcaWidget.cca_df).all(axis=None)
            )
            if not is_cca_same_as_stored:
                reinit_cca = self.warnEditingWithCca_df(
                    'Re-initialize cell cyle annotations first frame',
                    return_answer=True
                )
                if reinit_cca:
                    self.resetCcaFuture(0)
                    
        posData.cca_df = self.editCcaWidget.cca_df
        self.store_cca_df()
        
        self.editCcaWidget = None
        self.navigateScrollBar.setDisabled(False)
        self.navSpinBox.setDisabled(False)

        return True

    def isCurrentFrameCcaVisited(self):
        posData = self.data[self.pos_i]
        curr_df = posData.allData_li[posData.frame_i]['acdc_df']
        return curr_df is not None and 'cell_cycle_stage' in curr_df.columns

    def warnScellsGone(self, ScellsIDsGone, frame_i):
        msg = widgets_message_box.myMessageBox()
        text = html_utils.paragraph(f"""
            In the next frame the followning cells' IDs in S/G2/M
            (highlighted with a yellow contour) <b>will disappear</b>:<br><br>
            {ScellsIDsGone}<br><br>
            If the cell <b>does not exist</b> you might have deleted it at some point. 
            If that's the case, then try to go to some previous frames and reset 
            the cell cycle annotations there (button on the top toolbar).<br><br>
            These cells are either buds or mother whose <b>related IDs will not
            disappear</b>. This is likely due to cell division happening in
            previous frame and the divided bud or mother will be
            washed away.<br><br>
            If you decide to continue these cells will be <b>automatically
            annotated as divided at frame number {frame_i}</b>.<br><br>
            Do you want to continue?
        """)
        _, yesButton, noButton = msg.warning(
           self, 'Cells in "S/G2/M" disappeared!', text,
           buttonsTexts=('Cancel', 'Yes', 'No')
        )
        return msg.clickedButton == yesButton

    def checkScellsGone(self):
        """Check if there are cells in S phase whose relative disappear in
        current frame. Allow user to choose between automatically assign
        division to these cells or cancel and not visit the frame.

        Returns
        -------
        bool
            False if there are no cells disappeared or the user decided
            to accept automatic division.
        """
        automaticallyDividedIDs = []

        mode = str(self.modeComboBox.currentText())
        if mode.find('Cell cycle') == -1:
            # No cell cycle analysis mode --> do nothing
            return False, automaticallyDividedIDs

        posData = self.data[self.pos_i]

        if posData.allData_li[posData.frame_i]['labels'] is None:
            # Frame never visited/checked in segm mode --> autoCca_df will raise
            # a critical message
            return False, automaticallyDividedIDs

        # Check if there are S cells that either only mother or only
        # bud disappeared and automatically assign division to it
        # or abort visiting this frame
        prev_acdc_df = posData.allData_li[posData.frame_i-1]['acdc_df']
        prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
        prev_cca_df = prev_acdc_df[self.cca_df_colnames].copy()

        ScellsIDsGone = []
        for ccSeries in prev_cca_df.itertuples():
            ID = ccSeries.Index
            ccs = ccSeries.cell_cycle_stage
            if ccs != 'S':
                continue

            relID = ccSeries.relative_ID
            if relID == -1:
                continue
            
            # Check is relID is gone while ID stays
            if relID not in posData.IDs and ID in posData.IDs:
                ScellsIDsGone.append(relID)

        if not ScellsIDsGone:
            # No cells in S that disappears --> do nothing
            return False, automaticallyDividedIDs

        self.highlightNewIDs_ccaFailed(ScellsIDsGone, prev_rp=True)
        proceed = self.warnScellsGone(ScellsIDsGone, posData.frame_i)
        self.clearLostObjContoursItems()
        
        if not proceed:
            return True, automaticallyDividedIDs

        for IDgone in ScellsIDsGone:
            moth_bud_pair_cca = (
                posData.allData_li[posData.frame_i-1].get('moth_bud_pairs_cca')
            )
            relID = prev_cca_df.at[IDgone, 'relative_ID']
            self.annotateDisappearedBeforeDivision(relID, IDgone, prev_cca_df)
            self.annotateDivision(
                prev_cca_df, IDgone, relID, moth_bud_pair_cca, 
                frame_i=posData.frame_i-1
            )
            self.annotateDivisionCurrentFrameRelativeIDgone(relID)
            automaticallyDividedIDs.append(relID)
            
        self.store_cca_df(frame_i=posData.frame_i-1, cca_df=prev_cca_df)

        return False, automaticallyDividedIDs

    def annotateDivisionCurrentFrameRelativeIDgone(self, IDwhoseRelativeIsGone):
        posData = self.data[self.pos_i]
        if posData.cca_df is None:
            return
        ID = IDwhoseRelativeIsGone
        posData.cca_df.at[ID, 'generation_num'] += 1
        posData.cca_df.at[ID, 'division_frame_i'] = posData.frame_i-1
        posData.cca_df.at[ID, 'relationship'] = 'mother'

    def annotateDisappearedBeforeDivision(
            self, relID, IDgone, cca_df, frame_i=None
        ):
        posData = self.data[self.pos_i]        
        gen_num = cca_df.at[relID, 'generation_num']
        if frame_i is None:
            frame_i = posData.frame_i
        
        for past_frame_i in range(frame_i-1, -1, -1):
            past_cca_df = self.get_cca_df(frame_i=past_frame_i, return_df=True)
            if past_cca_df is None:
                return
            
            try:
                if past_cca_df.at[relID, 'generation_num'] != gen_num:
                    # ID is a mother and the cell cycle is finished here
                    return
            except Exception as err:
                # Bud stops existing --> stop process
                return
            
            past_cca_df.at[IDgone, 'disappears_before_division'] = 1
            past_cca_df.at[relID, 'daughter_disappears_before_division'] = 1
            
            self.store_cca_df(
                cca_df=past_cca_df, frame_i=past_frame_i, autosave=False
            )

    @exception_handler
    def attempt_auto_cca(self, enforceAll=False):
        mode = str(self.modeComboBox.currentText())
        posData = self.data[self.pos_i]

        if mode == 'Cell cycle analysis':
            notEnoughG1Cells, proceed = self.autoCca_df(
                enforceAll=enforceAll
            )
            if not proceed:
                return notEnoughG1Cells, proceed
            
            # mode = str(self.modeComboBox.currentText())
            if posData.cca_df is None: # ???
                notEnoughG1Cells = False
                proceed = True
                return notEnoughG1Cells, proceed
            if posData.cca_df.isna().any(axis=None):
                raise ValueError('Cell cycle analysis table contains NaNs')
            # self.checkMultiBudMoth()
            proceed = self.checkMothersExcludedOrDead()
            return notEnoughG1Cells, proceed

        elif mode == 'Normal division: Lineage tree':
            self.autoLinTree_df()
            notEnoughG1Cells = False
            proceed = True
            return notEnoughG1Cells, proceed
            
        else:
            notEnoughG1Cells = False
            proceed = True
            return notEnoughG1Cells, proceed

    def checkCcaPastFramesNewIDs(self):
        posData = self.data[self.pos_i]
        if not posData.new_IDs:
            return
        
        found_cca_df_IDs = []
        for frame_i in range(posData.frame_i-2, -1, -1):
            acdc_df = posData.allData_li[frame_i]['acdc_df']
            cca_df_i = acdc_df[self.cca_df_colnames]
            intersect_idx = cca_df_i.index.intersection(posData.new_IDs)
            cca_df_i = cca_df_i.loc[intersect_idx]
            if cca_df_i.empty:
                continue
            found_cca_df_IDs.append(cca_df_i)
            
            # Remove IDs found in past frames from new_IDs list
            newIDs = np.array(posData.new_IDs, dtype=np.uint32)
            mask_index = np.isin(newIDs, cca_df_i.index)
            posData.new_IDs = list(newIDs[~mask_index])
            if not posData.new_IDs:
                return found_cca_df_IDs
        return found_cca_df_IDs

    def initMissingFramesCca(self, last_cca_frame_i, current_frame_i):
        self.logger.info(
            'Initialising cell cycle annotations of missing past frames...'
        )
        posData = self.data[self.pos_i]
        current_frame_i = posData.frame_i
        
        # Collect only the frames that were actually annotated. Frames
        # without annotations are skipped (do not fill them with
        # placeholders, e.g. '' strings, because they would be propagated
        # to the initialized frames and corrupt the annotations)
        annotated_cca_dfs = []
        keys = []
        for frame_i in range(last_cca_frame_i+1):
            acdc_df = posData.allData_li[frame_i]['acdc_df']
            if acdc_df is None:
                continue
            
            if 'cell_cycle_stage' not in acdc_df.columns:
                continue
        
            annotated_cca_dfs.append(
                acdc_df.reindex(columns=self.cca_df_colnames)
            )
            keys.append(frame_i)
        
        last_annotated_cca_df = None
        if annotated_cca_dfs:
            names = ['frame_i', 'Cell_ID']
            annotated_cca_df = (
                pd.concat(annotated_cca_dfs, keys=keys, names=names)
                .reset_index()
                .set_index(['Cell_ID', 'frame_i'])
                .sort_index()
            )
            last_annotated_cca_df = annotated_cca_df.groupby(level=0).last()

        cca_df_colnames = self.cca_df_colnames
        pbar = tqdm(total=current_frame_i-last_cca_frame_i+1, ncols=100)
        for frame_i in range(last_cca_frame_i, current_frame_i+1):
            posData.frame_i = frame_i
            self.get_data()
            cca_df = self.getBaseCca_df()

            if last_annotated_cca_df is not None:
                idx = last_annotated_cca_df.index.intersection(cca_df.index)
                # Values missing in the annotations are taken from the
                # base (default) annotations
                last_annotated = (
                    last_annotated_cca_df.loc[idx, cca_df_colnames]
                    .combine_first(cca_df.loc[idx, cca_df_colnames])
                )
                cca_df.loc[idx, cca_df_colnames] = last_annotated

            self.store_cca_df(cca_df=cca_df, frame_i=frame_i, autosave=False)
            pbar.update()
        pbar.close()

        posData.frame_i = current_frame_i
        self.get_data()

    def _getCcaCostMatrix(
            self, numCellsG1, numNewCells, IDsCellsG1, newIDs_contours
        ):
        posData = self.data[self.pos_i]
        dataDict = posData.allData_li[posData.frame_i]
        dist_matrix_df = dataDict.get('obj_to_obj_dist_cost_matrix_df')
        if dist_matrix_df is None:
            cost = np.full((numCellsG1, numNewCells), np.inf)
            for obj in posData.rp:
                ID = obj.label
                try:
                    i = IDsCellsG1.index(ID)
                except ValueError:
                    continue

                cont = self.getObjContours(obj) # here 3D always is correct!
                i = IDsCellsG1.index(ID)
                
                # Get distance from cell in G1 and all other new cells
                for j, newID_cont in enumerate(newIDs_contours):
                    min_dist, nearest_xy = self.nearest_point_2Dyx(
                        cont, newID_cont
                    )
                    cost[i, j] = min_dist
            
            return cost

        cost = dist_matrix_df.loc[IDsCellsG1, posData.new_IDs].values
        
        return cost

    def autoCca_df(self, enforceAll=False):
        """
        Assign each bud to a mother with scipy linear sum assignment
        (Hungarian or Munkres algorithm). First we build a cost matrix where
        each (i, j) element is the minimum distance between bud i and mother j.
        Then we minimize the cost of assigning each bud to a mother, and finally
        we write the assignment info into cca_df
        """
        proceed = True
        notEnoughG1Cells = False
        ScellsGone = False

        posData = self.data[self.pos_i]

        # Skip cca if not the right mode
        mode = str(self.modeComboBox.currentText())
        if mode.find('Cell cycle') == -1:
            return notEnoughG1Cells, proceed

        # Make sure that this is a visited frame in segmentation tracking mode
        if posData.allData_li[posData.frame_i]['labels'] is None:
            proceed = self.warnFrameNeverVisitedSegmMode()
            return notEnoughG1Cells, proceed
        
        # Get temporary cca_df with single mother-bud pair mode active
        cca_df = self.getSingleMotherBudPairCca_df()
        if cca_df is not None:
            posData.cca_df = cca_df
            self.storeSingleMotherBudPairData(cca_df)
            return notEnoughG1Cells, proceed
        
        # Determine if this is the last visited frame for repeating
        # bud assignment on non manually correct (corrected_on_frame_i>0) buds.
        # The idea is that the user could have assigned division on a cell
        # by going previous and we want to check if this cell could be a
        # "better" mother for those non manually corrected buds
        curr_df = posData.allData_li[posData.frame_i]['acdc_df']
        isLastVisitedAgain = self.isLastVisitedAgainCca(
            curr_df, enforceAll=enforceAll
        )
        
        # Get previous dataframe
        acdc_df = posData.allData_li[posData.frame_i-1]['acdc_df']
        prev_cca_df = acdc_df[self.cca_df_colnames].copy()
        
        # When using "single mother-bud pair annotation" mode, 
        # there might be some IDs that do not have annotations yet
        # --> not fully annotated frame
        missingIDs = []
        if posData.cca_df is not None:
            missingIDs = [
                ID for ID in posData.IDs 
                if ID not in posData.cca_df.index
                and ID in prev_cca_df.index
            ]
        
        frameAlreadyFullyAnnotated = (
            posData.cca_df is not None
            and not enforceAll
            and not isLastVisitedAgain
            and not missingIDs
        )
        # Use stored cca_df and do not modify it with automatic stuff
        if frameAlreadyFullyAnnotated:
            return notEnoughG1Cells, proceed
        
        self.updateLostNewCurrentIDs()
        
        # Keep only correctedAssignIDs if requested
        # For the last visited frame we perform assignment again only on
        # IDs where we didn't manually correct assignment
        correctedAssignIDs = set()
        if isLastVisitedAgain and not enforceAll:
            try:
                correctedAssignIDs = curr_df[
                    curr_df['corrected_on_frame_i']>0
                ].index
            except Exception as e:
                correctedAssignIDs = []
            posData.new_IDs = [
                ID for ID in posData.new_IDs
                if ID not in correctedAssignIDs
            ]
        
        # Check if new IDs exist some time in the past
        found_cca_df_IDs = self.checkCcaPastFramesNewIDs()

        # Check if there are some S cells that disappeared
        abort, automaticallyDividedIDs = self.checkScellsGone()
        if abort:
            notEnoughG1Cells = False
            proceed = False
            return notEnoughG1Cells, proceed
        
        if posData.cca_df is None:
            posData.cca_df = prev_cca_df[self.cca_df_colnames]
        else:
            posData.cca_df = curr_df[self.cca_df_colnames]
        
        if missingIDs:
            # Add missing IDs from previous cca_df
            posData.cca_df.loc[missingIDs] = prev_cca_df.loc[missingIDs]
        
        posData.cca_df = posData.cca_df.dropna()
        posData.cca_df = self.addCcaInfoFromSingleMotherBudPairs(posData.cca_df)
        posData.cca_df = posData.cca_df[self.cca_df_colnames]
        
        # concatenate new IDs found in past frames (before frame_i-1)
        if found_cca_df_IDs is not None:
            cca_df = pd.concat([posData.cca_df, *found_cca_df_IDs])
            unique_idx = ~cca_df.index.duplicated(keep='first')
            posData.cca_df = cca_df[unique_idx]
        
        # If there are no new IDs we are done
        if not posData.new_IDs:
            proceed = True
            self.store_cca_df()
            return notEnoughG1Cells, proceed

        # Get cells in G1 (exclude dead) and check if there are enough cells in G1
        try:
            prev_df_G1 = prev_cca_df[prev_cca_df['cell_cycle_stage']=='G1']
            prev_df_G1 = prev_df_G1[~acdc_df.loc[prev_df_G1.index]['is_cell_dead']]
            IDsCellsG1 = set(prev_df_G1.index)
        except Exception as err:
            IDsCellsG1 = set()
        
        if isLastVisitedAgain or enforceAll:
            # If we are repeating auto cca for last visited frame
            # then we also add the cells in G1 that appears in current frame
            # and we remove the ones that are already in S in current frame 
            # if they were manually corrected (i.e., they cannot be mother).
            # Note that potential mother cells must be either appearing in 
            # current frame or in G1 also at previous frame. 
            # If we would consider cells that are in G1 at current frame 
            # but not in previous frame, assigning a bud to it would 
            # result in no G1 at all for the mother cell.
            df_G1 = posData.cca_df[posData.cca_df['cell_cycle_stage']=='G1']
            current_G1_IDs = df_G1.index
            new_cell_G1 = [
                ID for ID in current_G1_IDs if ID not in prev_cca_df.index
            ]
            IDsCellsG1.update(new_cell_G1)
            cells_S_current = posData.cca_df[
                (posData.cca_df['cell_cycle_stage']=='S')
                & (posData.cca_df['corrected_on_frame_i']==posData.frame_i)
            ].index
            IDsCellsG1 = IDsCellsG1 - set(cells_S_current)

        # Remove cells that disappeared
        IDsCellsG1 = [ID for ID in IDsCellsG1 if ID in posData.IDs]
        
        numCellsG1 = len(IDsCellsG1)
        numNewCells = len(posData.new_IDs)
        if numCellsG1 < numNewCells:
            notEnoughG1Cells, proceed = self.handleNoCellsInG1(
                numCellsG1, numNewCells
            )
            return notEnoughG1Cells, proceed

        # Compute new IDs contours
        newIDs_contours = []
        for obj in posData.rp:
            ID = obj.label
            if ID in posData.new_IDs:
                cont = self.getObjContours(obj) # here 3D always is correct!
                newIDs_contours.append(cont)

        # Compute cost matrix
        cost = self._getCcaCostMatrix(
            numCellsG1, numNewCells, IDsCellsG1, newIDs_contours
        )

        # Run hungarian (munkres) assignment algorithm
        row_idx, col_idx = scipy.optimize.linear_sum_assignment(cost)
        
        # New mother cells
        newMothIDs = {IDsCellsG1[i] for i in row_idx}
        
        # Assign buds to mothers
        for i, j in zip(row_idx, col_idx):
            mothID = IDsCellsG1[i]
            budID = posData.new_IDs[j]
            
            relID = None
            # If we are repeating assignment for the bud then we also have to
            # correct the possibily wrong mother --> it goes back to 
            # G1 if it's not a mother that we assign now
            if budID in posData.cca_df.index:
                relID = posData.cca_df.at[budID, 'relative_ID']
                if relID in prev_cca_df.index and relID not in newMothIDs:
                    posData.cca_df.loc[relID] = prev_cca_df.loc[relID]
            
            posData.cca_df.at[mothID, 'relative_ID'] = budID
            posData.cca_df.at[mothID, 'cell_cycle_stage'] = 'S'

            bud_cca_dict = base_cca_dict.copy()
            bud_cca_dict['cell_cycle_stage'] = 'S'
            bud_cca_dict['generation_num'] = 0
            bud_cca_dict['relative_ID'] = mothID
            bud_cca_dict['relationship'] = 'bud'
            bud_cca_dict['emerg_frame_i'] = posData.frame_i
            bud_cca_dict['is_history_known'] = True
            bud_cca_dict['corrected_on_frame_i'] = -1
            posData.cca_df.loc[budID] = pd.Series(bud_cca_dict)
        
        # Keep only existing IDs
        posData.cca_df = posData.cca_df.loc[posData.IDs]

        self.store_cca_df()
        proceed = True
        return notEnoughG1Cells, proceed

    def addIDBaseCca_df(self, posData, ID):
        if ID <= 0:
            # When calling update_cca_df_deletedIDs we add relative IDs
            # but they could be -1 for cells in G1
            return

        _zip = zip(
            self.cca_df_colnames,
            self.cca_df_default_values,
        )
        if posData.cca_df.empty:
            posData.cca_df = pd.DataFrame(
                {col: val for col, val in _zip},
                index=[ID]
            )
        else:
            for col, val in _zip:
                posData.cca_df.at[ID, col] = val
        self.store_cca_df()

    def getBaseCca_df(self, with_tree_cols=False): 
        posData = self.data[self.pos_i]
        IDs = posData.rp.IDs
        cca_df = core.getBaseCca_df(IDs, with_tree_cols=with_tree_cols)
        return cca_df

    def get_last_cca_frame_i(self):
        posData = self.data[self.pos_i]
        
        # Determine last annotated frame index (0 if no frame is annotated)
        last_cca_frame_i = len(posData.allData_li) - 1
        for i, dict_frame_i in enumerate(posData.allData_li):
            df = dict_frame_i['acdc_df']
            if df is None or 'cell_cycle_stage' not in df.columns:
                last_cca_frame_i = max(i-1, 0)
                break

        return last_cca_frame_i

    @exception_handler
    def initCca(self):
        posData = self.data[self.pos_i]
        last_tracked_i = self.get_last_tracked_i()
        defaultMode = 'Viewer'
        if last_tracked_i == 0:
            txt = html_utils.paragraph(
                'On this dataset either you <b>never checked</b> that the segmentation '
                'and tracking are <b>correct</b> or you did not save yet.<br><br>'
                'If you already visited some frames with "Segmentation and Tracking" '
                'mode save data before switching to "Cell cycle analysis mode".<br><br>'
                'Otherwise you first have to check (and eventually correct) some frames '
                'in "Segmentation and Tracking" mode before proceeding '
                'with cell cycle analysis.')
            msg = widgets_message_box.myMessageBox()
            msg.critical(
                self, 'Tracking was never checked', txt
            )
            self.cancelModeInit(defaultMode)
            return

        proceed = True
        
        last_cca_frame_i = self.get_last_cca_frame_i()
        if last_cca_frame_i == 0:
            # Remove undoable actions from segmentation mode
            posData.UndoRedoStates[0] = []
            self.undoAction.setEnabled(False)
            self.redoAction.setEnabled(False)
        
        if posData.frame_i > last_cca_frame_i:
            # Prompt user to go to last annotated frame
            msg = widgets_message_box.myMessageBox()
            # Cell cycle analysis cannot start from a frame that was never
            # checked in "Segmentation and Tracking" mode (e.g., after
            # visiting it in "Viewer" mode) --> do not offer to stay
            canStayOnCurrentFrame = posData.frame_i <= last_tracked_i
            if canStayOnCurrentFrame:
                txt = html_utils.paragraph(f"""
                    The <b>last annotated frame</b> is frame {last_cca_frame_i+1}.<br><br>
                    Do you want to restart cell cycle analysis from frame
                    {last_cca_frame_i+1}?<br>
                """)
                buttonsTexts = (
                    'Cancel', f'Yes, go to frame {last_cca_frame_i+1}',
                    'No, stay on current frame'
                )
            else:
                txt = html_utils.paragraph(f"""
                    The <b>last annotated frame</b> is frame {last_cca_frame_i+1}.<br><br>
                    The current frame (n. {posData.frame_i+1}) was
                    <b>never checked</b> in "Segmentation and Tracking" mode
                    (last checked frame is n. {last_tracked_i+1}),
                    therefore cell cycle analysis cannot start from the
                    current frame.<br><br>
                    Do you want to restart cell cycle analysis from frame
                    {last_cca_frame_i+1}?<br>
                """)
                buttonsTexts = (
                    'Cancel', f'Yes, go to frame {last_cca_frame_i+1}'
                )
            buttons = msg.warning(
                self, 'Go to last annotated frame?', txt, 
                buttonsTexts=buttonsTexts
            )
            goToFrameButton = buttons[1]
            stayButton = buttons[2] if canStayOnCurrentFrame else None
            if goToFrameButton == msg.clickedButton:
                self.addMissingIDs_cca_df(posData)
                msg = 'Looking good!'
                self.last_cca_frame_i = last_cca_frame_i
                posData.frame_i = last_cca_frame_i
                self.setLookingGoodTitle()
                self.get_data()
                self.addMissingIDs_cca_df(posData)
                self.store_cca_df()
                self.updateAllImages()
                self.updateScrollbars()
                self.updateSingleMotherBudPairLastAnnotatedFrame()
            elif stayButton is not None and stayButton == msg.clickedButton:
                self.addMissingIDs_cca_df(posData)
                self.store_cca_df()
                self.initMissingFramesCca(last_cca_frame_i, posData.frame_i)
                self.updateSingleMotherBudPairLastAnnotatedFrame()
                last_cca_frame_i = posData.frame_i
                msg = 'Cell cycle analysis initialised!'
                self.titleLabel.setText(msg, color='g')
            elif msg.cancel:
                msg = 'Cell cycle analysis cancelled.'
                self.logger.info(msg)
                self.titleLabel.setText(msg, color=self.titleColor)
                self.cancelModeInit(defaultMode)
                proceed = False
                return
        elif posData.frame_i < last_cca_frame_i:
            # Prompt user to go to last annotated frame
            msg = widgets_message_box.myMessageBox()
            txt = html_utils.paragraph(f"""
                The <b>last annotated frame</b> is frame {last_cca_frame_i+1}.<br><br>
                Do you want to restart cell cycle analysis from frame
                {last_cca_frame_i+1}?<br>
            """)
            yesButton, noButton, _ = msg.question(
                self, 'Go to last annotated frame?', txt, 
                buttonsTexts=('Yes', 'No', 'Cancel')
            )
            if msg.cancel:
                msg = 'Cell cycle analysis aborted.'
                self.logger.info(msg)
                self.titleLabel.setText(msg, color=self.titleColor)
                self.cancelModeInit(defaultMode)
                proceed = False
                return
            
            self.addMissingIDs_cca_df(posData)
            if msg.clickedButton == yesButton:
                self.addMissingIDs_cca_df(posData)
                msg = 'Looking good!'
                self.setLookingGoodTitle()
                self.last_cca_frame_i = last_cca_frame_i
                posData.frame_i = last_cca_frame_i
                self.get_data()
                self.addMissingIDs_cca_df(posData)
                self.store_cca_df()
                self.updateAllImages()
                self.updateScrollbars()
        else:
            self.get_data()
            self.addMissingIDs_cca_df(posData)
            self.store_cca_df()

        self.last_cca_frame_i = last_cca_frame_i

        self.navigateScrollBar.setMaximum(last_cca_frame_i+1)
        self.navSpinBox.setMaximum(last_cca_frame_i+1)
        self.lastTrackedFrameLabel.setText(
            f'Last cc annot. frame n. = {last_cca_frame_i+1}'
        )
        
        if posData.cca_df is None:
            posData.cca_df = self.getBaseCca_df()
            self.store_cca_df()
            msg = 'Cell cycle analysis initialized!'
            self.logger.info(msg)
            self.titleLabel.setText(msg, color=self.titleColor)
        else:
            self.get_cca_df()
        
        self.updateSingleMotherBudPairLastAnnotatedFrame()
        self.enqCcaIntegrityChecker()
        
        return proceed

    def getConcatCcaDf(self):
        posData = self.data[self.pos_i]
        cca_dfs = []
        keys = []
        for frame_i in range(0, posData.SizeT):
            cca_df = self.get_cca_df(
                frame_i=frame_i, 
                return_df=True,
            )
            if cca_df is None:
                break

            cca_dfs.append(cca_df)
            keys.append(frame_i)
        
        if not cca_dfs:
            return
        
        global_cca_df = pd.concat(cca_dfs, keys=keys, names=['frame_i'])
        return global_cca_df

    def storeFromConcatCcaDf(self, global_cca_df):
        posData = self.data[self.pos_i]
        for frame_i in range(0, posData.SizeT):
            try:
                cca_df = global_cca_df.loc[frame_i]
            except KeyError as err:
                break
            
            self.store_cca_df(frame_i=frame_i, cca_df=cca_df, autosave=False)
        
        self.get_cca_df()

    def resetWillDivideInfo(self):
        global_cca_df = self.getConcatCcaDf()
        if global_cca_df is None:
            return
        
        posData = self.data[self.pos_i]
        global_cca_df = load._fix_will_divide(global_cca_df)
        global_cca_df = global_cca_df.reset_index().set_index(
            ['Cell_ID', 'generation_num']
        )

        # Get will_divide info from stored single mother-bud pair data
        for frame_i in range(0, posData.SizeT):
            moth_bud_pairs_cca = (
                posData.allData_li[frame_i].get('moth_bud_pairs_cca')
            )
            if moth_bud_pairs_cca is None:
                continue
            
            will_divide_mask = moth_bud_pairs_cca['will_divide'] > 0
            mbp_cca_df_will_divide = moth_bud_pairs_cca[will_divide_mask]
            if mbp_cca_df_will_divide.empty:
                continue
            
            mbp_cca_df_will_divide = (
                mbp_cca_df_will_divide.reset_index()
                .set_index(['Cell_ID', 'generation_num'])
            )
            index_IDs_gen_num_will_divide = (
                global_cca_df.index.intersection(mbp_cca_df_will_divide.index)
            )
            if index_IDs_gen_num_will_divide.empty:
                continue

            global_cca_df.loc[index_IDs_gen_num_will_divide, 'will_divide'] = (
                1
            )
        
        global_cca_df = global_cca_df.reset_index().set_index(
            ['frame_i', 'Cell_ID']
        )

        incomplete_frames_mask = (
            global_cca_df.isna()
                .any(axis=1)
                .groupby(level='frame_i')
                .transform('any')
        )
        global_cca_df = global_cca_df[~incomplete_frames_mask]

        self.storeFromConcatCcaDf(global_cca_df)

    def resetCcaFuture(self, from_frame_i):
        posData = self.data[self.pos_i]
        self.last_cca_frame_i = from_frame_i-1
        self.ccaCheckerStopChecking()
        
        self.setNavigateScrollBarMaximum(last_cca_frame_i=from_frame_i-1) 
        for i in range(from_frame_i, posData.SizeT):
            posData.allData_li[i].pop('cca_df', None)
            posData.allData_li[i].pop('cca_df_checker', None)
            
            df = posData.allData_li[i]['acdc_df']
            if df is None:
                # No more saved info to delete
                break

            if 'cell_cycle_stage' not in df.columns:
                # No cell cycle info present
                continue

            df = df.drop(columns=self.cca_df_colnames)
            posData.allData_li[i]['acdc_df'] = df
        
        if posData.acdc_df is not None:
            frames = posData.acdc_df.index.get_level_values(0)
            if from_frame_i in frames:
                posData.acdc_df = posData.acdc_df.loc[:from_frame_i]
        
        if posData.frame_i == from_frame_i:
            self.removeCcaAnnotationsCurrentFrame()

        self.resetWillDivideInfo()

    def removeCcaAnnotationsCurrentFrame(self):
        posData = self.data[self.pos_i]
        posData.cca_df = None
        
        posData.allData_li[posData.frame_i].pop('cca_df', None)
        posData.allData_li[posData.frame_i].pop('cca_df_checker', None)
        
        df = posData.allData_li[posData.frame_i]['acdc_df']
        if df is None:
            # No more saved info to delete
            return False

        if 'cell_cycle_stage' not in df.columns:
            # No cell cycle info present
            return False

        df = df.drop(columns=self.cca_df_colnames)
        posData.allData_li[posData.frame_i]['acdc_df'] = df
        
        return True

    def resetFutureCcaColCurrentFrame(self):
        posData = self.data[self.pos_i]
        
        cca_df_S_mask = posData.cca_df.cell_cycle_stage == 'S'
        posData.cca_df.loc[cca_df_S_mask, 'will_divide'] = 0
        
        mothers_mask = (
            (posData.cca_df.relationship == 'mother')
            & cca_df_S_mask
        )
        bud_mask = posData.cca_df.relationship == 'bud'
        
        posData.cca_df.loc[mothers_mask, 'daughter_disappears_before_division'] = 0
        posData.cca_df.loc[bud_mask, 'disappears_before_division'] = 0
        
        cca_df = self.get_cca_df(frame_i=posData.frame_i, return_df=True)
        if cca_df is not None:
            cca_df_S_mask = cca_df.cell_cycle_stage == 'S'
            cca_df.loc[cca_df_S_mask, 'will_divide'] = 0
            
            mothers_mask = (
                (cca_df.relationship == 'mother')
                & cca_df_S_mask
            )
            bud_mask = cca_df.relationship == 'bud'
            
            cca_df.loc[mothers_mask, 'daughter_disappears_before_division'] = 0
            cca_df.loc[bud_mask, 'disappears_before_division'] = 0
        
        self.store_data()

    def get_cca_df(
            self, frame_i=None, 
            return_df=False, 
            debug=False, 
        ):
        # cca_df is None unless the metadata contains cell cycle annotations
        # NOTE: cell cycle annotations are either from the current session
        # or loaded from HDD in "initPosAttr" with a .question to the user
        posData = self.data[self.pos_i]
        cca_df = None
        i = posData.frame_i if frame_i is None else frame_i
        df = posData.allData_li[i]['acdc_df']            
        if df is not None and 'cell_cycle_stage' in df.columns:
            cca_df = df[self.cca_df_colnames].copy()
        
        if cca_df is None and self.isSnapshot:
            cca_df = self.getBaseCca_df()
            posData.cca_df = cca_df

        if cca_df is not None:
            cca_df = cca_df.dropna()
        
        if return_df:
            return cca_df
        else:
            posData.cca_df = cca_df

    def unstore_cca_df(self):
        posData = self.data[self.pos_i]
        acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
        for col in self.cca_df_colnames:
            if col not in acdc_df.columns:
                continue
            acdc_df.drop(col, axis=1, inplace=True)

    def store_cca_df(
            self, pos_i=None, frame_i=None, cca_df=None, mainThread=True,
            autosave=True, store_cca_df_copy=False
        ):
        pos_i = self.pos_i if pos_i is None else pos_i
        posData = self.data[pos_i]
        i = posData.frame_i if frame_i is None else frame_i
        if cca_df is None:
            cca_df = posData.cca_df
            if self.ccaTableWin is not None and mainThread:
                zoomIDs = self.getZoomIDs()
                self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)
        
        acdc_df = posData.allData_li[i]['acdc_df']
        if acdc_df is None:
            current_frame_i = None
            if frame_i is not None and frame_i != posData.frame_i:
                current_frame_i = posData.frame_i
                posData.frame_i = frame_i
                self.get_data()
            self.store_data()
            acdc_df = posData.allData_li[i]['acdc_df']
            if current_frame_i is not None:
                # Back to current frame
                posData.frame_i = current_frame_i
                self.get_data(debug=False)
        
        if 'cell_cycle_stage' in acdc_df.columns:
            # Cell cycle info already present --> overwrite with new
            acdc_df[self.cca_df_colnames] = cca_df[self.cca_df_colnames]
            posData.allData_li[i]['acdc_df'] = acdc_df            
        elif cca_df is not None:
            df = acdc_df.drop(cca_df.columns, axis=1, errors='ignore')
            df = df.join(cca_df, how='left')
            posData.allData_li[i]['acdc_df'] = df
        
        # Store copy for cca integrity worker
        self.store_cca_df_checker(posData, i, cca_df)
        
        if store_cca_df_copy and cca_df is not None:
            posData.allData_li[i]['cca_df'] = cca_df.copy()
        
        if self.annotateSingleMotherBudPairButton.isChecked():
            self.storeSingleMotherBudPairData(cca_df)

        if autosave:
            self.enqAutosave()
            self.enqCcaIntegrityChecker()

    def setCcaIssueContour(self, obj, prev_rp=False):
        label = obj.label
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        display_rp = self.get2DRP(frame_i = frame_i if not prev_rp else frame_i-1)
        obj = display_rp.get_obj_from_ID(label)
        objContours = self.getObjContours(
            obj,
            all_external=True,
            include_internal=self.showAllContoursToggle.isChecked()
        )
        for cont in objContours:
            xx = cont[:,0] + 0.5
            yy = cont[:,1] + 0.5
            self.ax1_lostObjScatterItem.addPoints(xx, yy)
            
        self.textAnnot[0].addObjAnnotation(
            obj, 'lost_object', f'{obj.label}?', False,
            rp=display_rp, getObjCentroidFunc=self.getObjCentroid
        )

    def isLastVisitedAgainCca(self, curr_df, enforceAll=False):
        # Determine if this is the last visited frame for repeating
        # bud assignment on non manually corrected_on_frame_i buds.
        # The idea is that the user could have assigned division on a cell
        # by going previous and we want to check if this cell could be a
        # "better" mother for those non manually corrected buds
        posData = self.data[self.pos_i]
        if curr_df is None:
            return False
        
        if 'cell_cycle_stage' not in curr_df.columns:
            return False
        
        if enforceAll:
            return False

        try:
            posData.new_IDs = [
                ID for ID in posData.new_IDs
                if curr_df.at[ID, 'is_history_known']
                and curr_df.at[ID, 'cell_cycle_stage'] == 'S'
            ]
        except Exception as err:
            printl(curr_df)

        lastVisited = False
        posData.new_IDs = [
            ID for ID in posData.new_IDs
            if curr_df.at[ID, 'is_history_known']
            and curr_df.at[ID, 'cell_cycle_stage'] == 'S'
        ]
        if posData.frame_i+1 < posData.SizeT:
            next_df = posData.allData_li[posData.frame_i+1]['acdc_df']
            if next_df is None:
                lastVisited = True
            else:
                if 'cell_cycle_stage' not in next_df.columns:
                    lastVisited = True
        else:
            lastVisited = True
        
        return lastVisited

    def highlightNewCellNotEnoughG1cells(self, IDsCellsG1):
        posData = self.data[self.pos_i]
        
        display_rp = self.get2DRP()
        for obj in display_rp:
            if obj.label not in IDsCellsG1:
                continue
            objContours = self.getObjContours(obj)
            if objContours is not None:
                xx = objContours[:,0] + 0.5
                yy = objContours[:,1] + 0.5
                self.ccaFailedScatterItem.addPoints(xx, yy)
            self.textAnnot[0].addObjAnnotation(
                obj, 'green', f'{obj.label}?', False,
                rp=posData.rp, getObjCentroidFunc=self.getObjCentroid
            )

    def handleNoCellsInG1(self, numCellsG1, numNewCells):
        posData = self.data[self.pos_i]
        self.highlightNewCellNotEnoughG1cells(posData.new_IDs)
        continueAnyway = _warnings.warnNotEnoughG1Cells(
            numCellsG1, posData.frame_i, numNewCells, qparent=self
        )
        if continueAnyway:
            notEnoughG1Cells = False
            proceed = True
            # Annotate the new IDs with unknown history
            for ID in posData.new_IDs:
                posData.cca_df.loc[ID] = pd.Series(base_cca_dict)
                cca_df_ID = self.getStatusKnownHistoryBud(ID)
                posData.ccaStatus_whenEmerged[ID] = cca_df_ID
        else:
            notEnoughG1Cells = True
            proceed = False
        
        # Clear new cells annotations
        self.ccaFailedScatterItem.setData([], [])
        return notEnoughG1Cells, proceed

    def highlightNewIDs_ccaFailed(self, IDsWithIssue, prev_rp=False):
        posData = self.data[self.pos_i]
        if prev_rp is False:
            rp = posData.rp
            frame_i = posData.frame_i
        else:
            frame_i = posData.frame_i - 1
            rp = posData.allData_li[frame_i]['regionprops']
        for obj in rp:
            if obj.label not in IDsWithIssue:
                continue
            self.setCcaIssueContour(obj, prev_rp=prev_rp)

    def setAnnotOptionsCcaMode(self):
        self.storeAnnotOptionsBeforeModeSwitch()
        # Use `setCheckedAlwaysEmit` so that the slots are called (and the
        # annotations updated) even if the checkboxes are already in the
        # requested state. Uncheck first, because the slot of an unchecked
        # text annotation option clears all the text annotations.
        self.annotIDsCheckbox(0).setCheckedAlwaysEmit(False)
        self.annotCellCycleInfoCheckbox(0).setCheckedAlwaysEmit(True)
        self.annotMotherDaughterLineCheckbox(0).setCheckedAlwaysEmit(True)

    def checkMissingCca(self):
        proceed = True
        ignore = False
        doNotShowAgain = False
        if self.doNotShowAgainMissingCca:
            return proceed, ignore, self.doNotShowAgainMissingCca
        
        missing_cca_items = []
        for posData in self.data:
            for frame_i, data_dict in enumerate(posData.allData_li):
                acdc_df = data_dict['acdc_df']
                if acdc_df is None:
                    continue
                
                if 'cell_cycle_stage' not in acdc_df.columns:
                    continue
                
                cca_df = acdc_df[cca_df_colnames]
                if cca_df.isnull().values.any():
                    i = frame_i if not self.isSnapshot else None
                    missing_cca_items.append((cca_df, posData, i))

        if not missing_cca_items:
            return proceed, ignore, doNotShowAgain
        
        proceed = False
        ignore, doNotShowAgain =_warnings.warnMissingCca(
            missing_cca_items, qparent=self
        )
        
        if doNotShowAgain:
            self.df_settings.at['doNotShowAgainMissingCca', 'value'] = 'Yes'
            self.df_settings.to_csv(self.settings_csv_path)
        
        return proceed, ignore, doNotShowAgain


class AnnotationsAfterEditsGuiElements:
    """Annotations after segmentation edits: warnEditingWithCca_df, the
    remembered answer and the updates of the cell cycle table after
    edits.
    """

    def addMissingIDs_cca_df(self, posData):
        base_cca_df = self.getBaseCca_df()
        if posData.cca_df is None:
            posData.cca_df = base_cca_df
            return
        
        posData.cca_df = posData.cca_df.combine_first(base_cca_df)

    def update_cca_df_relabelling(self, posData, oldIDs, newIDs):
        relIDs = posData.cca_df['relative_ID']
        posData.cca_df['relative_ID'] = relIDs.replace(oldIDs, newIDs)
        mapper = dict(zip(oldIDs, newIDs))
        posData.cca_df = posData.cca_df.rename(index=mapper)

    def update_cca_df_deletedIDs(
            self, posData, deletedIDs, dropInPast=True, dropInFuture=True
        ):
        if posData.cca_df is None:
            return
        
        # Store cca_df for undo action
        undoId = uuid.uuid4()
        self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)
        
        try:
            relIDs = (
                posData.cca_df.reindex(deletedIDs, fill_value=-1)
                ['relative_ID']
            )
        except KeyError as err:
            return
        
        posData.cca_df = posData.cca_df.drop(deletedIDs, errors='ignore')
        if self.isSnapshot:
            self.update_cca_df_newIDs(posData, relIDs)
        else:
            self.updateCcaDfDeletedIDsTimelapse(
                posData, relIDs, deletedIDs, undoId, dropInPast, dropInFuture
            )

    @disableWindow
    def updateCcaDfDeletedIDsTimelapse(
            self, posData, relIDsOfDelIDs, deletedIDs, undoId, 
            dropInPast, dropInFuture
        ):
        # Get status of the relIDs (of deleted IDs) to restore
        relIDsCcaStatus = {}
        for relID in relIDsOfDelIDs:
            try:
                ccs = posData.cca_df.at[relID, 'cell_cycle_stage']
                relationship = posData.cca_df.at[relID, 'relationship']
            except Exception as err:
                continue
            
            ccaStatus = core.getBaseCca_df([relID]).loc[relID]
            if relationship == 'mother' and ccs == 'S':
                for past_frame_i in range(posData.frame_i-1, -1, -1):
                    cca_df_i = self.get_cca_df(
                        frame_i=past_frame_i, return_df=True
                    )
                    ccs_past = cca_df_i.at[relID, 'cell_cycle_stage']      
                    if ccs_past == 'G1':
                        ccaStatus = cca_df_i.loc[relID]
                        break
            
            posData.cca_df.loc[relID] = ccaStatus
            self.store_data(autosave=False)
            relIDsCcaStatus[relID] = ccaStatus
            
        for fut_frame_i in range(posData.frame_i+1, posData.SizeT):
            cca_df_i = self.get_cca_df(frame_i=fut_frame_i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break
            
            self.storeUndoRedoCca(fut_frame_i, cca_df_i, undoId)

            if dropInFuture:
                cca_df_i = cca_df_i.drop(deletedIDs, errors='ignore')
            else:
                for delID in deletedIDs:
                    dataDict = posData.allData_li[fut_frame_i]
                    rp = dataDict['regionprops']
                    delIDexists = delID in rp.IDs
                    if not delIDexists:
                        continue
                    
                    cca_df_i.loc[delID] = core.getBaseCca_df([delID]).loc[delID]
            
            areRelIDsPresent = False
            for relID in relIDsOfDelIDs:
                try:
                    ccs = cca_df_i.at[relID, 'cell_cycle_stage']
                    relationship = cca_df_i.at[relID, 'relationship']
                    ccaStatus = relIDsCcaStatus[relID]
                    cca_df_i.loc[relID] = ccaStatus
                    areRelIDsPresent = True
                except Exception as err:
                    continue
            
            if not areRelIDsPresent:
                break
            
            self.store_cca_df(
                frame_i=fut_frame_i, cca_df=cca_df_i, autosave=False
            )
            
        # Correct past frames
        for past_frame_i in range(posData.frame_i-1, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=past_frame_i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break
            
            self.storeUndoRedoCca(past_frame_i, cca_df_i, undoId)
            if dropInPast:
                cca_df_i = cca_df_i.drop(deletedIDs, errors='ignore')
            else:
                for delID in deletedIDs:
                    dataDict = posData.allData_li[past_frame_i]
                    rp = dataDict['regionprops']
                    delIDexists = delID in rp.IDs
                    if not delIDexists:
                        continue
                    
                    cca_df_i.loc[delID] = core.getBaseCca_df([delID]).loc[delID]
            
            areRelIDsPresent = False
            for relID in relIDsOfDelIDs:
                try:
                    ccs = cca_df_i.at[relID, 'cell_cycle_stage']
                    relationship = cca_df_i.at[relID, 'relationship']
                    ccaStatus = relIDsCcaStatus[relID]
                    cca_df_i.loc[relID] = ccaStatus
                    areRelIDsPresent = True
                except Exception as err:
                    continue
            
            if not areRelIDsPresent:
                break
            
            self.store_cca_df(
                frame_i=past_frame_i, cca_df=cca_df_i, autosave=False
            )

    def update_cca_df_newIDs(self, posData, new_IDs):
        for newID in new_IDs:
            self.addIDBaseCca_df(posData, newID)

    def update_cca_df_snapshots(self, editTxt, posData):
        cca_df = posData.cca_df
        cca_df_IDs = cca_df.index
        if editTxt == 'Delete ID':
            deleted_IDs = [ID for ID in cca_df_IDs if ID not in posData.IDs]
            self.update_cca_df_deletedIDs(posData, deleted_IDs)

        elif editTxt == 'Separate IDs':
            new_IDs = [ID for ID in posData.IDs if ID not in cca_df_IDs]
            self.update_cca_df_newIDs(posData, new_IDs)
            deleted_IDs = [ID for ID in cca_df_IDs if ID not in posData.IDs]
            self.update_cca_df_deletedIDs(posData, deleted_IDs)

        elif editTxt == 'Edit ID':
            new_IDs = [ID for ID in posData.IDs if ID not in cca_df_IDs]
            self.update_cca_df_newIDs(posData, new_IDs)
            old_IDs = [ID for ID in cca_df_IDs if ID not in posData.IDs]
            self.update_cca_df_deletedIDs(posData, old_IDs)

        elif editTxt == 'Annotate ID as dead':
            return
        
        elif editTxt == 'Deleted non-selected objects':
            deleted_IDs = [ID for ID in cca_df_IDs if ID not in posData.IDs]
            self.update_cca_df_deletedIDs(posData, deleted_IDs)

        elif editTxt == 'Delete ID with eraser':
            deleted_IDs = [ID for ID in cca_df_IDs if ID not in posData.IDs]
            self.update_cca_df_deletedIDs(posData, deleted_IDs)

        elif editTxt == 'Add new ID with brush tool':
            new_IDs = [ID for ID in posData.IDs if ID not in cca_df_IDs]
            self.update_cca_df_newIDs(posData, new_IDs)

        elif editTxt == 'Merge IDs':
            deleted_IDs = [ID for ID in cca_df_IDs if ID not in posData.IDs]
            self.update_cca_df_deletedIDs(posData, deleted_IDs)

        elif editTxt == 'Add new ID with curvature tool':
            new_IDs = [ID for ID in posData.IDs if ID not in cca_df_IDs]
            self.update_cca_df_newIDs(posData, new_IDs)

        elif editTxt == 'Delete IDs using ROI':
            deleted_IDs = [ID for ID in cca_df_IDs if ID not in posData.IDs]
            self.update_cca_df_deletedIDs(posData, deleted_IDs)

        elif editTxt == 'Repeat segmentation':
            posData.cca_df = self.getBaseCca_df()

    def fixCcaDfAfterEdit(self, editTxt):
        posData = self.data[self.pos_i]
        if posData.cca_df is not None:
            # For snapshot mode we fix or reinit cca_df depending on the edit
            self.update_cca_df_snapshots(editTxt, posData)
            self.store_data()

    def isFrameCcaAnnotated(self):
        posData = self.data[self.pos_i]
        acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
        if acdc_df is None:
            return False

        return 'cell_cycle_stage' in acdc_df.columns

    def getLastAnnotatedFrames(self):
        """Indices of the last frame with cell cycle annotations and of the
        last frame with lineage tree annotations, -1 for none.

        Both annotations start at frame 0 and have no gaps
        (`get_last_cca_frame_i`, `get_last_lin_tree_frame_i`). Unlike
        `get_last_cca_frame_i`, which returns 0 also when no frame is
        annotated, frame 0 counts only if it has cell cycle annotations.

        Returns
        -------
        tuple of int
            (last_cca_frame_i, last_lin_tree_frame_i)
        """
        posData = self.data[self.pos_i]
        first_acdc_df = posData.allData_li[0]['acdc_df']
        last_cca_frame_i = -1
        if (
                first_acdc_df is not None
                and 'cell_cycle_stage' in first_acdc_df.columns
            ):
            last_cca_frame_i = self.get_last_cca_frame_i()
        return last_cca_frame_i, self.get_last_lin_tree_frame_i()

    def warnEditingWithCca_df(
            self, editTxt, return_answer=False, get_answer=False,
            get_cancelled=False, update_images=True
        ):
        """Warn that the segmentation of the current frame was edited while
        the frame has annotations that the edit most likely makes wrong,
        and ask whether to remove them from the current frame to the end.

        Cell cycle annotations are recognized by the `cell_cycle_stage`
        column, lineage tree annotations by their values
        (`has_lineage_tree_values`): tables loaded from file have the
        lineage tree columns on every frame, all NaN where nothing is
        annotated.

        If the user ticked "Remember my choice and do not ask again" for
        this edit (edits of `warnEditingWithAnnotActions`), the remembered
        answer is applied without asking. Both prompted and remembered
        answers are applied by `_applyEditingWithAnnotAnswer`.

        Parameters
        ----------
        editTxt : str
            The edit, shown in the warning.
        return_answer : bool, optional
            If True, only return whether the user chose to remove the
            annotations, without removing or updating anything: the caller
            handles the answer. Only cell cycle annotations are considered
            (the caller re-initializes those of the first frame). Default
            is False.
        get_answer : bool, optional
            If True, return whether the annotations were removed. Default
            is False.
        get_cancelled : bool, optional
            If True, return 'cancelled' when the user cancels. Default is
            False.
        update_images : bool, optional
            If True, update the images also when there is nothing to ask.
            Default is True.

        Returns
        -------
        bool or str
            False (or 'cancelled' with `get_cancelled`) if the user
            cancelled, the answer with `return_answer` or `get_answer`,
            otherwise True.
        """
        if self.isSnapshot:
            return True

        posData = self.data[self.pos_i]
        acdc_df = posData.allData_li[posData.frame_i]['acdc_df']

        cell_cycle_stage_present = (
            acdc_df is not None and 'cell_cycle_stage' in acdc_df.columns
        )
        lineage_tree_present = (
            not return_answer and has_lineage_tree_values(acdc_df)
        )
        if not cell_cycle_stage_present and not lineage_tree_present:
            if update_images:
                self.updateAllImages()
            return True

        dropDelIDsNoteText = (
            '' if editTxt.find('Delete') == -1 else ' (drop removed IDs)'
        )
        action = self.warnEditingWithAnnotActions.get(editTxt, None)
        if action is not None and not action.isChecked():
            # The user ticked "Remember my choice and do not ask again"
            # --> apply the remembered answer without asking
            removeAnnot = action.removeAnnot
            if return_answer:
                return removeAnnot

            self._applyEditingWithAnnotAnswer(
                removeAnnot, cell_cycle_stage_present, lineage_tree_present,
                bool(dropDelIDsNoteText)
            )
            if get_answer:
                return removeAnnot
            return True

        annot_names = []
        if cell_cycle_stage_present:
            annot_names.append('cell cycle')
        if lineage_tree_present:
            annot_names.append('lineage tree')
        warn_type = f'{" and ".join(annot_names)} annotations'

        msg = widgets_message_box.myMessageBox()
        txt = html_utils.paragraph(
            f'You modified a frame that <b>has {warn_type}</b>.<br><br>'
            f'The change <b>"{editTxt}"</b> most likely makes the '
            '<b>annotations wrong</b>.<br><br>'
            'If you really want to apply this change we recommend to remove '
            f'ALL {warn_type}<br>'
            'from current frame to the end.<br><br>'
            'What do you want to do?'
        )
        if action is not None:
            checkBox = QCheckBox('Remember my choice and do not ask again')
        else:
            checkBox = None

        _, removeAnnotButton, _ = msg.warning(
            self, 'Edited segmentation with annotations!', txt,
            buttonsTexts=(
                'Cancel',
                'Remove annotations from future frames (RECOMMENDED)',
                f'Do not remove annotations{dropDelIDsNoteText}'
            ), widgets=checkBox
            )
        if msg.cancel:
            if get_cancelled:
                return 'cancelled'
            removeAnnotations = False
            return removeAnnotations

        removeAnnot = msg.clickedButton == removeAnnotButton
        if action is not None:
            remember = checkBox.isChecked()
            action.setChecked(not remember)
            # Replay only a remembered answer: a warning turned off from
            # the Settings menu keeps the annotations (see
            # `warnEditingWithAnnotActionTriggered`)
            action.removeAnnot = removeAnnot if remember else False

        if return_answer:
            return removeAnnot

        self._applyEditingWithAnnotAnswer(
            removeAnnot, cell_cycle_stage_present, lineage_tree_present,
            bool(dropDelIDsNoteText)
        )

        if get_answer:
            return removeAnnot
        else:
            return True

    def warnEditingWithAnnotActionTriggered(self, action, checked=False):
        """The user toggled one of the "Show warning when ..." items of
        Settings > Warnings and pop-ups (`warnEditingWithAnnotActions`).

        Forget the answer remembered with "Remember my choice": a warning
        turned off from the menu means "do not ask" and keeps the
        annotations, it must not replay an earlier "Remove".

        Parameters
        ----------
        action : QAction
            The toggled item (passed explicitly, not via `sender()`).
        checked : bool, optional
            New checked state (unused).
        """
        action.removeAnnot = False

    def _applyEditingWithAnnotAnswer(
            self, removeAnnot, cell_cycle_stage_present, lineage_tree_present,
            dropDelIDs
        ):
        """Apply the answer to `warnEditingWithCca_df`, prompted or
        remembered.

        Parameters
        ----------
        removeAnnot : bool
            True to remove the annotations of the types present from the
            current frame to the end, False to keep them. Kept cell cycle
            annotations are updated to the edited objects (with
            `dropDelIDs` the deleted IDs are dropped from them); frames
            without cell cycle annotations never get a cell cycle table.
        cell_cycle_stage_present : bool
            Whether the current frame has cell cycle annotations.
        lineage_tree_present : bool
            Whether the current frame has lineage tree annotations.
        dropDelIDs : bool
            Whether the edit deleted objects.
        """
        posData = self.data[self.pos_i]
        if removeAnnot:
            if cell_cycle_stage_present:
                self.resetFutureCcaColCurrentFrame()
                self.resetCcaFuture(posData.frame_i)
            if lineage_tree_present:
                self.resetLin_tree_future()
            self.updateAllImages()
            return

        if cell_cycle_stage_present:
            if dropDelIDs and posData.cca_df is not None:
                delIDs = [
                    ID for ID in posData.cca_df.index if ID not in posData.IDs
                ]
                self.update_cca_df_deletedIDs(
                    posData, delIDs, dropInPast=False
                )
            self.addMissingIDs_cca_df(posData)
        self.updateAllImages()
        self.store_data()
