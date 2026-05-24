"""Qt view adapter for cell-cycle annotation workflows."""

from __future__ import annotations

import traceback
import uuid

from tqdm import tqdm
from qtpy.QtCore import QMutex, QThread, QTimer, QWaitCondition
from qtpy.QtWidgets import QCheckBox, QMessageBox, QPushButton

from cellacdc import (
    apps, _warnings, base_cca_dict, base_cca_tree_dict, disableWindow,
    exception_handler, html_utils,
)
from cellacdc import widgets, workers
from cellacdc.viewmodels.cell_cycle_viewmodel import CellCycleViewModel


class CellCycleView:
    """Qt-facing adapter for cell-cycle annotation workflows."""

    LEGACY_METHODS = (
        'nearest_point_2Dyx',
        'isCurrentFrameCcaVisited',
        'warnScellsGone',
        'checkScellsGone',
        'attempt_auto_cca',
        'highlightIDs',
        'warnFrameNeverVisitedSegmMode',
        'checkCcaPastFramesNewIDs',
        'addIDBaseCca_df',
        'getBaseCca_df',
        'get_last_cca_frame_i',
        'initCca',
        '_getCcaCostMatrix',
        'autoCca_df',
        'initMissingFramesCca',
        'addMissingIDs_cca_df',
        'update_cca_df_relabelling',
        'update_cca_df_deletedIDs',
        'updateCcaDfDeletedIDsTimelapse',
        'update_cca_df_newIDs',
        'update_cca_df_snapshots',
        'fixCcaDfAfterEdit',
        'setCcaIssueContour',
        'isLastVisitedAgainCca',
        'highlightNewCellNotEnoughG1cells',
        'highlightNewIDs_ccaFailed',
        'handleNoCellsInG1',
        'isFrameCcaAnnotated',
        'warnEditingWithCca_df',
        'ccaIntegrCheckerToggled',
        'startCcaIntegrityCheckerWorker',
        'initCcaIntegrityChecker',
        'disableCcaIntegrityChecker',
        'stopCcaIntegrityCheckerWorker',
        'isCcaCheckerChecking',
        'getConcatCcaDf',
        'storeFromConcatCcaDf',
        'resetWillDivideInfo',
        'ccaCheckerStopChecking',
        'enqCcaIntegrityChecker',
        'resetCcaFuture',
        'removeCcaAnnotationsCurrentFrame',
        'resetFutureCcaColCurrentFrame',
        'get_cca_df',
        'unstore_cca_df',
        'store_cca_df_checker',
        'store_cca_df',
        'viewCcaTable',
        'autoAssignBud_YeastMate',
        'reInitCca',
        'repeatAutoCca',
        'manualEditCcaToolbarActionTriggered',
        'manualEditCca',
        'applyManualCcaChangesFutureFrames',
        'ccaCheckerWorkerDone',
        'goToFrameNumber',
        'warnCcaIntegrity',
        'fixWillDivide',
        'ccaCheckerWorkerClosed',
        'updateIsHistoryKnown',
        'annotateIsHistoryKnown',
        'annotateWillDivide',
        'annotateDivision',
        'undoDivisionAnnotation',
        'undoBudMothAssignment',
        'manualCellCycleAnnotation',
        'warnMotherNotEligible',
        'warnSettingHistoryKnownCellsFirstFrame',
        'checkMothEligibility',
        'checkMothersExcludedOrDead',
        'checkDivisionCanBeUndone',
        'stopBlinkingPairItem',
        'warnDeadOrExcludedMothers',
        'startBlinkingPairingItem',
        'blinkPairingItem',
        'annotateBudToDifferentMother',
        'onMotherNotInG1',
        'warnBudAnnotatedDividedInFuture',
        'warnMotherNotAtLeastOneFrameG1',
        'checkChangeMotherBudEligible',
        'checkSwapMothersEligibility',
        'swapMothers',
    )

    def __init__(self, host, view_model: CellCycleViewModel):
        object.__setattr__(self, 'host', host)
        object.__setattr__(self, 'view_model', view_model)

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host', 'view_model'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def bind_legacy_methods(self):
        for name in self.LEGACY_METHODS:
            setattr(self.host, name, getattr(self, name))

    def nearest_point_2Dyx(self, points, all_others):
        """
        Given 2D array of [y, x] coordinates points and all_others return the
        [y, x] coordinates of the two points (one from points and one from all_others)
        that have the absolute minimum distance
        """
        return self.view_model.cca_workflows.nearest_point_2d_yx(points, all_others)

    def isCurrentFrameCcaVisited(self):
        posData = self.data[self.pos_i]
        curr_df = posData.allData_li[posData.frame_i]['acdc_df']
        return self.view_model.cca_edits.has_annotations(curr_df)

    def warnScellsGone(self, ScellsIDsGone, frame_i):
        msg = widgets.myMessageBox()
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
           self.host, 'Cells in "S/G2/M" disappeared!', text,
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

        ScellsIDsGone = self.view_model.evaluate_sister_relations(prev_cca_df, posData.IDs)


        if not ScellsIDsGone:
            # No cells in S that disappears --> do nothing
            return False, automaticallyDividedIDs

        self.highlightNewIDs_ccaFailed(ScellsIDsGone, rp=prev_rp)
        proceed = self.warnScellsGone(ScellsIDsGone, posData.frame_i)
        self.clearLostObjContoursItems()

        if not proceed:
            return True, automaticallyDividedIDs

        past_cca_frames = (
            (frame_i, self.get_cca_df(frame_i=frame_i, return_df=True))
            for frame_i in range(posData.frame_i-2, -1, -1)
        )
        propagation_result = self.view_model.cca_workflows.propagate_s_phase_disappearance_divisions(
            prev_cca_df,
            posData.cca_df,
            posData.frame_i,
            posData.IDs,
            past_cca_frames=past_cca_frames,
            disappeared_ids=ScellsIDsGone,
        )
        if propagation_result.current_cca_df is not None:
            posData.cca_df = propagation_result.current_cca_df

        automaticallyDividedIDs.extend(
            propagation_result.automatically_divided_ids
        )
        updated_cca_dfs = propagation_result.updated_cca_dfs_by_frame
        for frame_i, cca_df_i in updated_cca_dfs.items():
            self.store_cca_df(
                frame_i=frame_i,
                cca_df=cca_df_i,
                autosave=False,
            )

        return False, automaticallyDividedIDs

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

    def highlightIDs(self, IDs, pen):
        pass

    def warnFrameNeverVisitedSegmMode(self):
        msg = widgets.myMessageBox()
        warn_cca = msg.critical(
            self.host, 'Next frame NEVER visited',
            'Next frame was never visited in "Segmentation and Tracking"'
            'mode.\n You cannot perform cell cycle analysis on frames'
            'where segmentation and/or tracking errors were not'
            'checked/corrected.\n\n'
            'Switch to "Segmentation and Tracking" mode '
            'and check/correct next frame,\n'
            'before attempting cell cycle analysis again',
        )
        return False

    def checkCcaPastFramesNewIDs(self):
        posData = self.data[self.pos_i]
        if not posData.new_IDs:
            return

        past_acdc_frames = (
            (frame_i, posData.allData_li[frame_i]['acdc_df'])
            for frame_i in range(posData.frame_i-2, -1, -1)
        )
        result = self.view_model.cca_workflows.collect_existing_new_id_rows(
            posData.new_IDs,
            past_acdc_frames,
            self.cca_df_colnames,
        )
        posData.new_IDs = result.remaining_new_ids
        return result.found_cca_dfs

    def addIDBaseCca_df(self, posData, ID):
        if ID <= 0:
            # When calling update_cca_df_deletedIDs we add relative IDs
            # but they could be -1 for cells in G1
            return

        posData.cca_df = self.view_model.cca_edits.add_base_annotation(
            posData.cca_df,
            ID,
            base_values=base_cca_dict,
        )
        self.store_cca_df()

    def getBaseCca_df(self, with_tree_cols=False):
        posData = self.data[self.pos_i]
        IDs = [obj.label for obj in posData.rp]
        return self.view_model.cca_edits.build_base_annotations(
            IDs,
            with_tree_cols=with_tree_cols,
            base_values=base_cca_dict,
            tree_values=base_cca_tree_dict,
        )

    def get_last_cca_frame_i(self):
        posData = self.data[self.pos_i]
        return self.view_model.cca_edits.last_annotated_frame_index(
            dict_frame_i['acdc_df']
            for dict_frame_i in posData.allData_li
        )

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
            msg = widgets.myMessageBox()
            msg.critical(
                self.host, 'Tracking was never checked', txt
            )
            self.modeComboBox.setCurrentText(defaultMode)
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
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(f"""
                The <b>last annotated frame</b> is frame {last_cca_frame_i+1}.<br><br>
                Do you want to restart cell cycle analysis from frame
                {last_cca_frame_i+1}?<br>
            """)
            _, goToFrameButton, stayButton = msg.warning(
                self.host, 'Go to last annotated frame?', txt,
                buttonsTexts=(
                    'Cancel', f'Yes, go to frame {last_cca_frame_i+1}',
                    'No, stay on current frame')
            )
            if goToFrameButton == msg.clickedButton:
                self.addMissingIDs_cca_df(posData)
                self.store_cca_df()
                msg = 'Looking good!'
                self.last_cca_frame_i = last_cca_frame_i
                posData.frame_i = last_cca_frame_i
                self.titleLabel.setText(msg, color=self.titleColor)
                self.get_data()
                self.addMissingIDs_cca_df(posData)
                self.store_cca_df()
                self.updateAllImages()
                self.updateScrollbars()
            elif stayButton == msg.clickedButton:
                self.addMissingIDs_cca_df(posData)
                self.store_cca_df()
                self.initMissingFramesCca(last_cca_frame_i, posData.frame_i)
                last_cca_frame_i = posData.frame_i
                msg = 'Cell cycle analysis initialised!'
                self.titleLabel.setText(msg, color='g')
            elif msg.cancel:
                msg = 'Cell cycle analysis aborted.'
                self.logger.info(msg)
                self.titleLabel.setText(msg, color=self.titleColor)
                self.modeComboBox.setCurrentText(defaultMode)
                proceed = False
                return
        elif posData.frame_i < last_cca_frame_i:
            # Prompt user to go to last annotated frame
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(f"""
                The <b>last annotated frame</b> is frame {last_cca_frame_i+1}.<br><br>
                Do you want to restart cell cycle analysis from frame
                {last_cca_frame_i+1}?<br>
            """)
            yesButton, noButton, _ = msg.question(
                self.host, 'Go to last annotated frame?', txt,
                buttonsTexts=('Yes', 'No', 'Cancel')
            )
            if msg.cancel:
                msg = 'Cell cycle analysis aborted.'
                self.logger.info(msg)
                self.titleLabel.setText(msg, color=self.titleColor)
                self.modeComboBox.setCurrentText(defaultMode)
                proceed = False
                return

            self.addMissingIDs_cca_df(posData)
            if msg.clickedButton == yesButton:
                self.addMissingIDs_cca_df(posData)
                msg = 'Looking good!'
                self.titleLabel.setText(msg, color=self.titleColor)
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

        self.enqCcaIntegrityChecker()

        return proceed

    def _getCcaCostMatrix(
            self, numCellsG1, numNewCells, IDsCellsG1, newIDs_contours
        ):
        posData = self.data[self.pos_i]
        dataDict = posData.allData_li[posData.frame_i]
        dist_matrix_df = dataDict.get('obj_to_obj_dist_cost_matrix_df')
        if dist_matrix_df is None:
            mother_contours = {}
            for obj in posData.rp:
                ID = obj.label
                if ID not in IDsCellsG1:
                    continue
                mother_contours[ID] = self.getObjContours(obj)

            bud_contours = dict(zip(posData.new_IDs, newIDs_contours))
            return self.view_model.cca_workflows.auto_cost_matrix_from_contours(
                IDsCellsG1,
                posData.new_IDs,
                mother_contours,
                bud_contours,
            )

        return self.view_model.cca_workflows.auto_cost_matrix_from_distances(
            dist_matrix_df,
            IDsCellsG1,
            posData.new_IDs,
        )

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

        # Determine if this is the last visited frame for repeating
        # bud assignment on non manually correct (corrected_on_frame_i>0) buds.
        # The idea is that the user could have assigned division on a cell
        # by going previous and we want to check if this cell could be a
        # "better" mother for those non manually corrected buds
        curr_df = posData.allData_li[posData.frame_i]['acdc_df']
        isLastVisitedAgain = self.isLastVisitedAgainCca(
            curr_df, enforceAll=enforceAll
        )

        frameAlreadyAnnotated = (
            posData.cca_df is not None
            and not enforceAll
            and not isLastVisitedAgain
        )
        # Use stored cca_df and do not modify it with automatic stuff
        if frameAlreadyAnnotated:
            return notEnoughG1Cells, proceed

        # Keep only correctedAssignIDs if requested
        # For the last visited frame we perform assignment again only on
        # IDs where we didn't manually correct assignment
        if isLastVisitedAgain and not enforceAll:
            posData.new_IDs = self.view_model.cca_workflows.uncorrected_new_ids_for_auto(
                posData.new_IDs,
                curr_df,
            )

        # Check if new IDs exist some time in the past
        found_cca_df_IDs = self.checkCcaPastFramesNewIDs()

        # Check if there are some S cells that disappeared
        abort, automaticallyDividedIDs = self.checkScellsGone()
        if abort:
            notEnoughG1Cells = False
            proceed = False
            return notEnoughG1Cells, proceed

        # Get previous dataframe
        acdc_df = posData.allData_li[posData.frame_i-1]['acdc_df']
        prev_cca_df = acdc_df[self.cca_df_colnames].copy()

        init_result = self.view_model.cca_workflows.prepare_auto_current_frame(
            prev_cca_df,
            curr_df,
            self.cca_df_colnames,
            current_cca_df=posData.cca_df,
            found_cca_dfs=found_cca_df_IDs or (),
        )
        posData.cca_df = init_result.cca_df

        # If there are no new IDs we are done
        if not posData.new_IDs:
            proceed = True
            self.store_cca_df()
            return notEnoughG1Cells, proceed

        # Get cells in G1 (exclude dead) and check if there are enough cells in G1
        IDsCellsG1 = self.view_model.cca_workflows.auto_candidate_mother_ids(
            prev_cca_df,
            acdc_df,
            posData.IDs,
            current_cca_df=posData.cca_df,
            include_current_g1=isLastVisitedAgain or enforceAll,
            current_frame_i=posData.frame_i,
        )

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
                cont = self.getObjContours(obj)
                newIDs_contours.append(cont)

        # Compute cost matrix
        cost = self._getCcaCostMatrix(
            numCellsG1, numNewCells, IDsCellsG1, newIDs_contours
        )

        # Assign buds to mothers
        assignments = self.view_model.cca_workflows.auto_assignments_from_cost(
            cost,
            IDsCellsG1,
            posData.new_IDs,
        )
        posData.cca_df = self.view_model.cca_workflows.apply_auto_assignments(
            posData.cca_df,
            assignments,
            posData.frame_i,
            self.view_model.cca_workflows.base_status(base_cca_dict),
            previous_cca_df=prev_cca_df,
            current_ids=posData.IDs,
        )

        self.store_cca_df()
        proceed = True
        return notEnoughG1Cells, proceed

    def initMissingFramesCca(self, last_cca_frame_i, current_frame_i):
        self.logger.info(
            'Initialising cell cycle annotations of missing past frames...'
        )
        posData = self.data[self.pos_i]
        current_frame_i = posData.frame_i

        prep_result = self.view_model.cca_workflows.prepare_missing_frame_annotations(
            posData.allData_li,
            self.cca_df_colnames,
            last_cca_frame_i,
        )
        for frame_i, acdc_df in prep_result.acdc_dfs_by_frame.items():
            posData.allData_li[frame_i]['acdc_df'] = acdc_df

        last_annotated_cca_df = prep_result.last_annotated_cca_df
        cca_df_colnames = self.cca_df_colnames
        pbar = tqdm(total=current_frame_i-last_cca_frame_i+1, ncols=100)
        for frame_i in range(last_cca_frame_i, current_frame_i+1):
            posData.frame_i = frame_i
            self.get_data()
            cca_df = self.getBaseCca_df()
            cca_df = self.view_model.cca_workflows.overlay_last_annotated(
                cca_df,
                last_annotated_cca_df,
                cca_df_colnames,
            )

            self.store_cca_df(cca_df=cca_df, frame_i=frame_i, autosave=False)
            pbar.update()
        pbar.close()

        posData.frame_i = current_frame_i
        self.get_data()

    def addMissingIDs_cca_df(self, posData):
        base_cca_df = self.getBaseCca_df()
        result = self.view_model.cca_edits.add_missing_ids(
            posData.cca_df,
            base_cca_df,
        )
        posData.cca_df = result.cca_df

    def update_cca_df_relabelling(self, posData, oldIDs, newIDs):
        result = self.view_model.cca_edits.relabel_ids(
            posData.cca_df,
            oldIDs,
            newIDs,
        )
        posData.cca_df = result.cca_df

    def update_cca_df_deletedIDs(
            self, posData, deletedIDs, dropInPast=True, dropInFuture=True
        ):
        if posData.cca_df is None:
            return

        # Store cca_df for undo action
        undoId = uuid.uuid4()
        self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)

        try:
            deletion_result = self.view_model.cca_edits.delete_ids(
                posData.cca_df,
                deletedIDs,
            )
        except KeyError:
            return

        posData.cca_df = deletion_result.cca_df
        relIDs = deletion_result.relative_ids
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
        future_cca_frames = (
            (frame_i, self.get_cca_df(frame_i=frame_i, return_df=True))
            for frame_i in range(posData.frame_i + 1, posData.SizeT)
        )
        past_cca_frames = (
            (frame_i, self.get_cca_df(frame_i=frame_i, return_df=True))
            for frame_i in range(posData.frame_i - 1, -1, -1)
        )
        existing_ids_by_frame = None
        if not dropInPast or not dropInFuture:
            existing_ids_by_frame = {}
            for frame_i in range(posData.SizeT):
                dataDict = posData.allData_li[frame_i]
                existingIDs = dataDict.get('IDs_idxs', {})
                if hasattr(existingIDs, 'items'):
                    existing_ids_by_frame[frame_i] = {
                        ID for ID, exists in existingIDs.items() if exists
                    }
                else:
                    existing_ids_by_frame[frame_i] = set(existingIDs)

        propagation_result = self.view_model.cca_workflows.propagate_deleted_ids(
            None,
            posData.frame_i,
            deletedIDs,
            relIDsOfDelIDs,
            current_cca_df=posData.cca_df,
            future_cca_frames=future_cca_frames,
            past_cca_frames=past_cca_frames,
            drop_in_past=dropInPast,
            drop_in_future=dropInFuture,
            existing_ids_by_frame=existing_ids_by_frame,
            base_values=base_cca_dict,
        )
        for frame_i in propagation_result.undo_frame_indices:
            cca_df_i = self.get_cca_df(frame_i=frame_i, return_df=True)
            self.storeUndoRedoCca(frame_i, cca_df_i, undoId)

        updated_cca_dfs = propagation_result.updated_cca_dfs_by_frame
        if posData.frame_i in updated_cca_dfs:
            posData.cca_df = propagation_result.current_cca_df
            self.store_data(autosave=False)

        for frame_i, cca_df_i in updated_cca_dfs.items():
            if frame_i == posData.frame_i:
                continue
            self.store_cca_df(
                frame_i=frame_i, cca_df=cca_df_i, autosave=False
            )

    def update_cca_df_newIDs(self, posData, new_IDs):
        for newID in new_IDs:
            self.addIDBaseCca_df(posData, newID)

    def update_cca_df_snapshots(self, editTxt, posData):
        result = self.view_model.cca_edits.apply_snapshot_id_edits(
            posData.cca_df,
            editTxt,
            posData.IDs,
            self.getBaseCca_df(),
            base_values=base_cca_dict,
        )

        if result.changes.deleted_ids:
            undoId = uuid.uuid4()
            self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)
        posData.cca_df = result.cca_df

    def fixCcaDfAfterEdit(self, editTxt):
        posData = self.data[self.pos_i]
        if posData.cca_df is not None:
            # For snapshot mode we fix or reinit cca_df depending on the edit
            self.update_cca_df_snapshots(editTxt, posData)
            self.store_data()

    def setCcaIssueContour(self, obj):
        objContours = self.getObjContours(obj, all_external=True)
        for cont in objContours:
            xx = cont[:,0] + 0.5
            yy = cont[:,1] + 0.5
            self.ax1_lostObjScatterItem.addPoints(xx, yy)
        self.textAnnot[0].addObjAnnotation(
            obj, 'lost_object', f'{obj.label}?', False
        )

    def isLastVisitedAgainCca(self, curr_df, enforceAll=False):
        # Determine if this is the last visited frame for repeating
        # bud assignment on non manually corrected_on_frame_i buds.
        # The idea is that the user could have assigned division on a cell
        # by going previous and we want to check if this cell could be a
        # "better" mother for those non manually corrected buds
        posData = self.data[self.pos_i]
        next_df = None
        if posData.frame_i+1 < posData.SizeT:
            next_df = posData.allData_li[posData.frame_i+1]['acdc_df']
        result = self.view_model.cca_workflows.auto_repeat_frame_state(
            curr_df,
            next_df,
            posData.new_IDs,
            enforce_all=enforceAll,
        )
        posData.new_IDs = result.new_ids
        return result.is_last_visited_again

    def highlightNewCellNotEnoughG1cells(self, IDsCellsG1):
        posData = self.data[self.pos_i]
        for obj in posData.rp:
            if obj.label not in IDsCellsG1:
                continue
            objContours = self.getObjContours(obj)
            if objContours is not None:
                xx = objContours[:,0] + 0.5
                yy = objContours[:,1] + 0.5
                self.ccaFailedScatterItem.addPoints(xx, yy)
            self.textAnnot[0].addObjAnnotation(
                obj, 'green', f'{obj.label}?', False
            )

    def highlightNewIDs_ccaFailed(self, IDsWithIssue, rp=None):
        if rp is None:
            posData = self.data[self.pos_i]
            rp = posData.rp
        for obj in rp:
            if obj.label not in IDsWithIssue:
                continue
            self.setCcaIssueContour(obj)

    def handleNoCellsInG1(self, numCellsG1, numNewCells):
        posData = self.data[self.pos_i]
        self.highlightNewCellNotEnoughG1cells(posData.new_IDs)
        continueAnyway = _warnings.warnNotEnoughG1Cells(
            numCellsG1, posData.frame_i, numNewCells, qparent=self.host
        )
        if continueAnyway:
            notEnoughG1Cells = False
            proceed = True
            # Annotate the new IDs with unknown history
            for ID in posData.new_IDs:
                posData.cca_df = self.view_model.cca_edits.add_base_annotation(
                    posData.cca_df,
                    ID,
                    base_values=base_cca_dict,
                )
                cca_df_ID = self.view_model.cca_workflows.known_history_status_for_bud(
                    ID,
                    (
                        (i, self.get_cca_df(frame_i=i, return_df=True))
                        for i in range(posData.frame_i-1, -1, -1)
                    ),
                    self.view_model.cca_workflows.base_status(base_cca_dict),
                )
                posData.ccaStatus_whenEmerged[ID] = cca_df_ID
        else:
            notEnoughG1Cells = True
            proceed = False

        # Clear new cells annotations
        self.ccaFailedScatterItem.setData([], [])
        return notEnoughG1Cells, proceed

    def isFrameCcaAnnotated(self):
        posData = self.data[self.pos_i]
        acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
        return self.view_model.cca_edits.has_annotations(acdc_df)

    def warnEditingWithCca_df(
            self, editTxt, return_answer=False, get_answer=False,
            get_cancelled=False, update_images=True
        ):
        # Function used to warn that the user is editing in "Segmentation and
        # Tracking" mode a frame that contains cca annotations.
        # Ask whether to remove annotations from all future frames
        if self.isSnapshot:
            return True

        posData = self.data[self.pos_i]
        acdc_df = posData.allData_li[posData.frame_i]['acdc_df']

        cell_cycle_stage_present = self.view_model.cca_edits.has_annotations(
            acdc_df
        )
        lineage_tree_present = (
            self.view_model.lineage.has_lineage_tree_annotations(
                acdc_df,
                self.lineage_tree,
            )
        )

        action = self.warnEditingWithAnnotActions.get(editTxt, None)
        warning_plan = self.view_model.annotated_edit_warning_plan(
            is_snapshot=self.isSnapshot,
            acdc_df_missing=acdc_df is None,
            lineage_tree_missing=self.lineage_tree is None,
            cell_cycle_stage_present=cell_cycle_stage_present,
            lineage_tree_present=lineage_tree_present,
            remembered_skip_warning=(
                action is not None and not action.isChecked()
            ),
        )
        if warning_plan.proceed_without_warning:
            if update_images:
                if warning_plan.update_images:
                    self.updateAllImages()
            return True

        msg = widgets.myMessageBox()
        warn_type = warning_plan.warn_type
        txt = html_utils.paragraph(
            f'You modified a frame that <b>has {warn_type}</b>.<br><br>'
            f'The change <b>"{editTxt}"</b> most likely makes the '
            '<b>annotations wrong</b>.<br><br>'
            'If you really want to apply this change we reccommend to remove'
            f'ALL {warn_type}<br>'
            'from current frame to the end.<br><br>'
            'What do you want to do?'
        )
        if action is not None:
            checkBox = QCheckBox('Remember my choice and do not ask again')
        else:
            checkBox = None

        dropDelIDsNoteText = (
            '' if editTxt.find('Delete') == -1 else ' (drop removed IDs)'
        )
        _, removeAnnotButton, _ = msg.warning(
            self.host, 'Edited segmentation with annotations!', txt,
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

        if action is not None:
            action.setChecked(not checkBox.isChecked())
            action.removeAnnot = msg.clickedButton == removeAnnotButton

        if return_answer:
            return msg.clickedButton == removeAnnotButton

        if (msg.clickedButton == removeAnnotButton) and cell_cycle_stage_present:
            self.resetFutureCcaColCurrentFrame()
            self.resetCcaFuture(posData.frame_i+1)
            self.updateAllImages()
        elif (msg.clickedButton == removeAnnotButton) and lineage_tree_present:
            self.resetLin_tree_future()
            self.updateAllImages()
        else:
            if dropDelIDsNoteText and posData.cca_df is not None:
                delIDs = [
                    ID for ID in posData.cca_df.index if ID not in posData.IDs
                ]
                self.update_cca_df_deletedIDs(
                    posData, delIDs, dropInPast=False
                )
            self.addMissingIDs_cca_df(posData)
            self.updateAllImages()
            self.store_data()
        # if action is not None:
        #     if action.removeAnnot:
        #         self.store_data()
        #         posData.frame_i -= 1
        #         self.get_data()
        #         if lineage_tree_present:
        #             self.resetLin_tree_future()
        #         self.resetCcaFuture(posData.frame_i)
        #         self.next_frame()

        if get_answer:
            return msg.clickedButton == removeAnnotButton
        else:
            return True

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

    def startCcaIntegrityCheckerWorker(self):
        if not hasattr(self, 'data'):
            return

        if not self.isDataLoaded:
            return

        if not self.ccaIntegrCheckerToggle.isChecked():
            return

        ccaCheckerThread = QThread()
        self.ccaCheckerMutex = QMutex()
        self.ccaCheckerWaitCond = QWaitCondition()

        worker = workers.CcaIntegrityCheckerWorker(
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

    def stopCcaIntegrityCheckerWorker(self):
        try:
            self.ccaIntegrityCheckerWorker._stop()
        except Exception as err:
            pass

    def isCcaCheckerChecking(self):
        if not self.ccaCheckerRunning:
            return False

        return self.ccaIntegrityCheckerWorker.isChecking

    def getConcatCcaDf(self):
        posData = self.data[self.pos_i]
        return self.view_model.cca_edits.concat_annotations(
            posData.allData_li,
            self.cca_df_colnames,
            size_t=posData.SizeT,
        )

    def storeFromConcatCcaDf(self, global_cca_df):
        posData = self.data[self.pos_i]
        for frame_i, cca_df in self.view_model.cca_edits.split_concat_annotations(
            global_cca_df,
            size_t=posData.SizeT,
        ):
            self.store_cca_df(frame_i=frame_i, cca_df=cca_df, autosave=False)

        self.get_cca_df()

    def resetWillDivideInfo(self):
        global_cca_df = self.getConcatCcaDf()
        if global_cca_df is None:
            return

        global_cca_df = self.view_model.cca_workflows.fix_will_divide_without_next_generation(global_cca_df)
        self.storeFromConcatCcaDf(global_cca_df)

    def ccaCheckerStopChecking(self):
        if not self.ccaCheckerRunning:
            return

        self.ccaIntegrityCheckerWorker.clearQueue()

        if self.ccaIntegrityCheckerWorker.isChecking:
            self.ccaIntegrityCheckerWorker.abortChecking = True

    def enqCcaIntegrityChecker(self):
        if not self.ccaCheckerRunning:
            return
        posData = self.data[self.pos_i]
        self.ccaIntegrityCheckerWorker.enqueue(posData)

    def resetCcaFuture(self, from_frame_i):
        posData = self.data[self.pos_i]
        self.last_cca_frame_i = from_frame_i-1
        self.ccaCheckerStopChecking()

        self.setNavigateScrollBarMaximum()
        removal_result = self.view_model.cca_edits.remove_future_annotations(
            posData.allData_li,
            self.cca_df_colnames,
            from_frame_i,
            size_t=posData.SizeT,
            concatenated_acdc_df=posData.acdc_df,
        )
        for i in removal_result.cache_frame_indices:
            posData.allData_li[i].pop('cca_df', None)
            posData.allData_li[i].pop('cca_df_checker', None)
        for i, acdc_df in removal_result.acdc_dfs_by_frame.items():
            posData.allData_li[i]['acdc_df'] = acdc_df
        posData.acdc_df = removal_result.concatenated_acdc_df

        self.resetWillDivideInfo()

    def removeCcaAnnotationsCurrentFrame(self):
        posData = self.data[self.pos_i]
        posData.cca_df = None

        posData.allData_li[posData.frame_i].pop('cca_df', None)
        posData.allData_li[posData.frame_i].pop('cca_df_checker', None)

        df = posData.allData_li[posData.frame_i]['acdc_df']
        result = self.view_model.cca_edits.remove_annotations(
            df, self.cca_df_colnames
        )
        if result.missing_frame or not result.removed:
            return False

        posData.allData_li[posData.frame_i]['acdc_df'] = result.acdc_df
        return True

    def resetFutureCcaColCurrentFrame(self):
        posData = self.data[self.pos_i]
        posData.cca_df = self.view_model.cca_edits.reset_future_flags(
            posData.cca_df
        )
        self.store_data()

    def get_cca_df(self, frame_i=None, return_df=False, debug=False):
        # cca_df is None unless the metadata contains cell cycle annotations
        # NOTE: cell cycle annotations are either from the current session
        # or loaded from HDD in "initPosAttr" with a .question to the user
        posData = self.data[self.pos_i]
        i = posData.frame_i if frame_i is None else frame_i
        df = posData.allData_li[i]['acdc_df']
        result = self.view_model.cca_edits.resolve_annotations(
            df,
            self.cca_df_colnames,
            is_snapshot=self.isSnapshot,
            snapshot_cell_ids=(obj.label for obj in posData.rp),
            base_values=base_cca_dict,
            tree_values=base_cca_tree_dict,
        )
        cca_df = result.cca_df

        if return_df:
            return cca_df
        else:
            posData.cca_df = cca_df

    def unstore_cca_df(self):
        posData = self.data[self.pos_i]
        acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
        result = self.view_model.cca_edits.remove_annotations(
            acdc_df, self.cca_df_colnames
        )
        if result.acdc_df is not None:
            posData.allData_li[posData.frame_i]['acdc_df'] = result.acdc_df

    def store_cca_df_checker(self, posData, frame_i, cca_df):
        checker_cca_df = self.view_model.cca_edits.prepare_checker_annotations(
            cca_df,
            checker_running=self.ccaCheckerRunning,
        )
        if checker_cca_df is None:
            return

        posData.allData_li[frame_i]['cca_df_checker'] = checker_cca_df

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
                zoomIDs = self.exporting_view.getZoomIDs()
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

        store_result = self.view_model.cca_edits.store_frame_annotations(
            acdc_df,
            cca_df,
            self.cca_df_colnames,
            store_checker_copy=self.ccaCheckerRunning,
            store_cca_df_copy=store_cca_df_copy,
        )
        if store_result.acdc_df is not None:
            posData.allData_li[i]['acdc_df'] = store_result.acdc_df

        # Store copy for cca integrity worker
        if store_result.checker_cca_df is not None:
            posData.allData_li[i]['cca_df_checker'] = (
                store_result.checker_cca_df
            )

        if store_result.cached_cca_df is not None:
            posData.allData_li[i]['cca_df'] = store_result.cached_cca_df

        if autosave:
            self.enqAutosave()
            self.enqCcaIntegrityChecker()

    def viewCcaTable(self):
        posData = self.data[self.pos_i]
        zoomIDs = self.exporting_view.getZoomIDs()

        df = posData.allData_li[posData.frame_i]['acdc_df']
        current_cca_df = posData.cca_df
        if zoomIDs is not None:
            df = df.loc[zoomIDs]
            current_cca_df = current_cca_df.loc[zoomIDs]

        for column in current_cca_df.columns:
            header = (
                '================================================\n'
                f'CURRENT vs STORED `{column}` column'
                f'for frame number {posData.frame_i+1}:\n'
            )
            df_compare = current_cca_df[[column]].copy()
            df_compare[f'STORED_{column}'] = df[column]
            text = f'{header}{df_compare}'
            self.logger.info(text)

        if self.view_model.cca_edits.has_annotations(df):
            cca_df = df[self.cca_df_colnames]
            cca_df = cca_df.merge(
                current_cca_df, how='outer', left_index=True, right_index=True,
                suffixes=('_STORED', '_CURRENT')
            )
            cca_df = cca_df.reindex(sorted(cca_df.columns), axis=1)
            num_cols = len(cca_df.columns)
            for j in range(0,num_cols,2):
                df_j_x = cca_df.iloc[:,j]
                df_j_y = cca_df.iloc[:,j+1]
                if any(df_j_x!=df_j_y):
                    self.logger.info('------------------------')
                    self.logger.info('DIFFERENCES:')
                    diff_df = cca_df.iloc[:,j:j+2]
                    diff_mask = diff_df.iloc[:,0]!=diff_df.iloc[:,1]
                    self.logger.info(diff_df[diff_mask])
        else:
            cca_df = None
            self.logger.info(cca_df)
        self.logger.info('========================')
        if current_cca_df is None:
            return
        if current_cca_df.empty:
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                'Cell cycle annotations\' table is <b>empty</b>.<br>'
            )
            msg.warning(self.host, 'Table empty', txt)
            return

        df = posData.add_tree_cols_to_cca_df(
            current_cca_df, frame_i=posData.frame_i
        )
        if self.ccaTableWin is None:
            self.ccaTableWin = apps.ViewCcaTableWindow(df, parent=self.host)
            self.ccaTableWin.show()
            self.ccaTableWin.setGeometryWindow()
            self.ccaTableWin.sigUpdateCcaTable.connect(
                self.exporting_view.onSigUpdateCcaTableWindow
            )
        else:
            self.ccaTableWin.setFocus()
            self.ccaTableWin.activateWindow()
            self.ccaTableWin.updateTable(current_cca_df)

    def autoAssignBud_YeastMate(self):
        if not self.is_win:
            txt = (
                'YeastMate is available only on Windows OS.'
                'We are working on expading support also on macOS and Linux.\n\n'
                'Thank you for your patience!'
            )
            msg = QMessageBox()
            msg.critical(
                self.host, 'Supported only on Windows', txt, msg.Ok
            )
            return

        model_name = 'YeastMate'
        idx = self.modelNames.index(model_name)

        self.titleLabel.setText(
            f'{model_name} is thinking... '
            '(check progress in terminal/console)', color=self.titleColor
        )

        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        posData = self.data[self.pos_i]
        # Check if model needs to be imported
        acdcSegment = self.acdcSegment_li[idx]
        if acdcSegment is None:
            acdcSegment = (
                self.view_model.model_registry.import_segmentation_module(
                    model_name
                )
            )
            self.acdcSegment_li[idx] = acdcSegment

        # Read all models parameters
        init_params, segment_params = (
            self.view_model.model_registry.model_arg_specs(acdcSegment)
        )
        # Prompt user to enter the model parameters
        try:
            url = acdcSegment.url_help()
        except AttributeError:
            url = None

        _SizeZ = None
        if self.isSegm3D:
            _SizeZ = posData.SizeZ
        win = apps.QDialogModelParams(
            init_params,
            segment_params,
            model_name,
            url=url,
            posData=posData,
            df_metadata=posData.metadata_df
        )
        win.exec_()
        if win.cancel:
            self.titleLabel.setText('Segmentation aborted.')
            return

        use_gpu = win.init_kwargs.get('gpu', False)
        proceed = self.view_model.model_registry.check_gpu_available(
            model_name, use_gpu, qparent=self.host
        )
        if not proceed:
            self.logger.info('Segmentation process cancelled.')
            self.titleLabel.setText('Segmentation process cancelled.')
            return

        self.model_kwargs = win.model_kwargs
        model = self.view_model.model_registry.init_segmentation_model(
            acdcSegment, posData, win.init_kwargs
        )
        if model is None:
            self.logger.info('Segmentation process cancelled.')
            self.titleLabel.setText('Segmentation process cancelled.')
            return
        try:
            model.setupLogger(self.logger)
        except Exception as e:
            pass

        self.models[idx] = model

        img = self.getDisplayedImg1()

        posData.cca_df = model.predictCcaState(img, posData.lab)
        self.store_data()
        self.updateAllImages()

        self.titleLabel.setText('Budding event prediction done.', color='g')

    def reInitCca(self):
        if not self.isSnapshot:
            txt = html_utils.paragraph(
                'If you decide to continue <b>ALL cell cycle annotations</b> from '
                'this frame to the end will be <b>erased from current session</b> '
                '(saved data is not touched of course).<br><br>'
                'To annotate future frames again you will have to revisit them.<br><br>'
                'Do you want to continue?'
            )
            msg = widgets.myMessageBox()
            msg.warning(
               self.host, 'Re-initialize annnotations?', txt,
               buttonsTexts=('Cancel', 'Yes')
            )
            posData = self.data[self.pos_i]
            if msg.cancel:
                return

            # Reset all future frames
            self.resetCcaFuture(posData.frame_i+1)
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
        if self.view_model.cca_edits.has_annotations(next_df):
            msg = QMessageBox()
            warn_cca = msg.critical(
                self.host, 'Future visited frames detected!',
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

    def manualEditCcaToolbarActionTriggered(self):
        self.manualEditCca()

    def manualEditCca(self, checked=True):
        posData = self.data[self.pos_i]
        editCcaWidget = apps.editCcaTableWidget(
            posData.cca_df, posData.SizeT, current_frame_i=posData.frame_i,
            parent=self.host
        )
        editCcaWidget.sigApplyChangesFutureFrames.connect(
            self.applyManualCcaChangesFutureFrames
        )
        editCcaWidget.exec_()
        if editCcaWidget.cancel:
            return
        posData.cca_df = editCcaWidget.cca_df
        self.store_cca_df()
        # self.checkMultiBudMoth()
        self.updateAllImages()

    @exception_handler
    def applyManualCcaChangesFutureFrames(self, changes, stop_frame_i):
        self.store_data(autosave=False)
        posData = self.data[self.pos_i]
        undoId = uuid.uuid4()
        for i in range(posData.frame_i, stop_frame_i):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break

            self.storeUndoRedoCca(i, cca_df_i, undoId)

            cca_df_i = self.view_model.cca_edits.apply_manual_changes(
                cca_df_i, changes
            )
            self.store_cca_df(frame_i=i, cca_df=cca_df_i, autosave=False)
        self.get_data()
        self.updateAllImages()

    def ccaCheckerWorkerDone(self):
        self.status_hover_view.set_status_bar_label(log=False)

    def goToFrameNumber(self, frame_n):
        posData = self.data[self.pos_i]
        posData.frame_i = frame_n - 1
        self.get_data()
        self.updateAllImages()
        self.updateScrollbars()

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
            txt, category, self.host,
            go_to_frame_callback=self.goToFrameNumber
        )
        if disabled_warning:
            self.disabled_cca_warnings.add(disabled_warning)

        self.isWarningCcaIntegrity = False

    def fixWillDivide(self, warning_txt, IDs_will_divide_wrong):
        self.logger.info(warning_txt)
        self.logger.info('Fixing `will_divide` information...')

        global_cca_df = self.getConcatCcaDf()
        global_cca_df = self.view_model.cca_workflows.reset_will_divide_for_generations(
            global_cca_df,
            IDs_will_divide_wrong,
        )
        self.storeFromConcatCcaDf(global_cca_df)

    def ccaCheckerWorkerClosed(self, worker):
        self.logger.info('Cell cycle annotations integrity checker stopped.')
        self.ccaCheckerRunning = False

    def updateIsHistoryKnown():
        """
        This function is called every time the user saves and it is used
        for updating the status of cells where we don't know the history

        There are three possibilities:

        1. The cell with unknown history is a BUD
           --> we don't know when that  bud emerged --> 'emerg_frame_i' = -1
        2. The cell with unknown history is a MOTHER cell
           --> we don't know emerging frame --> 'emerg_frame_i' = -1
               AND generation number --> we start from 'generation_num' = 2
        3. The cell with unknown history is a CELL in G1
           --> we don't know emerging frame -->  'emerg_frame_i' = -1
               AND generation number --> we start from 'generation_num' = 2
               AND relative's ID in the previous cell cycle --> 'relative_ID' = -1
        """
        pass

    def annotateIsHistoryKnown(self, ID):
        """
        This function is used for annotating that a cell has unknown or known
        history. Cells with unknown history are for example the cells already
        present in the first frame or cells that appear in the frame from
        outside of the field of view.

        With this function we simply set 'is_history_known' to False.
        When the users saves instead we update the entire staus of the cell
        with unknown history with the function "updateIsHistoryKnown()"
        """
        posData = self.data[self.pos_i]
        is_history_known = posData.cca_df.at[ID, 'is_history_known']
        relID = posData.cca_df.at[ID, 'relative_ID']
        relID_cca = None
        if relID in posData.cca_df.index:
            relID_cca = self.view_model.cca_workflows.previous_relative_status_before_bud_emergence(
                ID,
                relID,
                (
                    (i, self.get_cca_df(frame_i=i, return_df=True))
                    for i in range(posData.frame_i-1, -1, -1)
                ),
                self.view_model.cca_workflows.base_status(base_cca_dict),
            )

        if is_history_known:
            # Save status of ID when emerged to allow undoing
            statusID_whenEmerged = self.view_model.cca_workflows.known_history_status_for_bud(
                ID,
                (
                    (i, self.get_cca_df(frame_i=i, return_df=True))
                    for i in range(posData.frame_i-1, -1, -1)
                ),
                self.view_model.cca_workflows.base_status(base_cca_dict),
            )
            if statusID_whenEmerged is None:
                return
            posData.ccaStatus_whenEmerged[ID] = statusID_whenEmerged

        # Store cca_df for undo action
        undoId = uuid.uuid4()
        self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)

        if ID not in posData.ccaStatus_whenEmerged:
            self.warnSettingHistoryKnownCellsFirstFrame(ID)
            return

        future_cca_frames = (
            (i, self.get_cca_df(frame_i=i, return_df=True))
            for i in range(posData.frame_i+1, posData.SizeT)
        )
        past_cca_frames = (
            (i, self.get_cca_df(frame_i=i, return_df=True))
            for i in range(posData.frame_i-1, -1, -1)
        )
        propagation_result = self.view_model.cca_workflows.propagate_history_knowledge(
            posData.cca_df,
            posData.frame_i,
            ID,
            future_cca_frames=future_cca_frames,
            past_cca_frames=past_cca_frames,
            status_when_emerged=posData.ccaStatus_whenEmerged.get(ID),
            relative_id=relID,
            relative_status=relID_cca,
        )
        posData.cca_df = propagation_result.current_cca_df

        # Update cell cycle info LabelItems
        obj_idx = posData.IDs.index(ID)
        rp_ID = posData.rp[obj_idx]

        if relID in posData.IDs:
            relObj_idx = posData.IDs.index(relID)
            rp_relID = posData.rp[relObj_idx]

        self.setAllTextAnnotations()
        self.drawAllMothBudLines()

        self.store_cca_df()

        if self.ccaTableWin is not None:
            zoomIDs = self.exporting_view.getZoomIDs()
            self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)

        for frame_i in propagation_result.undo_frame_indices:
            cca_df_i = self.get_cca_df(frame_i=frame_i, return_df=True)
            self.storeUndoRedoCca(frame_i, cca_df_i, undoId)

        for frame_i, cca_df_i in propagation_result.updated_cca_dfs_by_frame.items():
            if frame_i == posData.frame_i:
                continue
            self.store_cca_df(frame_i=frame_i, cca_df=cca_df_i, autosave=False)

        self.enqAutosave()

    def annotateWillDivide(self, ID, relID, frame_i=None):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i

        past_cca_frames = (
            (i, self.get_cca_df(frame_i=i, return_df=True))
            for i in range(frame_i-1, -1, -1)
        )
        propagation_result = self.view_model.cca_workflows.propagate_will_divide(
            None,
            frame_i,
            ID,
            relID,
            past_cca_frames=past_cca_frames,
        )
        for past_frame_i, cca_df_i in (
                propagation_result.updated_cca_dfs_by_frame.items()
        ):
            self.store_cca_df(
                cca_df=cca_df_i,
                frame_i=past_frame_i,
                autosave=False,
            )

    def annotateDivision(self, cca_df, ID, relID, frame_i=None):
        # Correct as follows:
        # For frame_i > 0 --> assign to G1 and +1 on generation number
        # For frame == 0 --> reinitialize to unknown cells
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i

        self.annotateWillDivide(ID, relID)
        return self.view_model.cca_workflows.annotate_division(cca_df, ID, relID, frame_i)

    def undoDivisionAnnotation(self, cca_df, ID, relID):
        # Correct as follows:
        # If G1 then correct to S and -1 on generation number
        return self.view_model.cca_workflows.undo_division_annotation(cca_df, ID, relID)

    def undoBudMothAssignment(self, ID):
        posData = self.data[self.pos_i]
        changed = self.view_model.cca_workflows.undo_bud_mother_assignment(posData.cca_df, ID)
        if not changed:
            return

        self.store_cca_df()

        # Update cell cycle info LabelItems
        self.setAllTextAnnotations()

        if self.ccaTableWin is not None:
            zoomIDs = self.exporting_view.getZoomIDs()
            self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)

    @exception_handler
    def manualCellCycleAnnotation(self, ID):
        """
        This function is used for both annotating division or undoing the
        annotation. It can be called on any frame.

        If we annotate division (right click on a cell in S) then it will
        check if there are future frames to correct.
        Frames to correct are those frames where both the mother and the bud
        are annotated as S phase cells.
        In this case we assign all those frames to G1, relationship to mother,
        and +1 generation number

        If we undo the annotation (right click on a cell in G1) then it will
        correct both past and future annotated frames (if present).
        Frames to correct are those frames where both the mother and the bud
        are annotated as G1 phase cells.
        In this case we assign all those frames to G1, relationship back to
        bud, and -1 generation number
        """
        posData = self.data[self.pos_i]

        # Store cca_df for undo action
        undoId = uuid.uuid4()
        self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)

        # Correct current frame
        clicked_ccs = posData.cca_df.at[ID, 'cell_cycle_stage']
        relID = posData.cca_df.at[ID, 'relative_ID']

        if relID not in posData.IDs:
            return

        if clicked_ccs == 'G1' and posData.frame_i == 0:
            # We do not allow undoing division annotation on first frame
            return

        if clicked_ccs == 'G1':
            issue_frame_i = self.checkDivisionCanBeUndone(ID, relID)
            if issue_frame_i is not None:
                _warnings.warnDivisionAnnotationCannotBeUndone(
                    ID, relID, issue_frame_i, qparent=self.host
                )
                return

        future_cca_frames = (
            (frame_i, self.get_cca_df(frame_i=frame_i, return_df=True))
            for frame_i in range(posData.frame_i + 1, posData.SizeT)
        )
        past_cca_frames = (
            (frame_i, self.get_cca_df(frame_i=frame_i, return_df=True))
            for frame_i in range(posData.frame_i - 1, -1, -1)
        )
        propagation_result = self.view_model.cca_workflows.propagate_manual_division_annotation(
            None,
            posData.frame_i,
            ID,
            current_cca_df=posData.cca_df,
            future_cca_frames=future_cca_frames,
            past_cca_frames=past_cca_frames,
        )
        posData.cca_df = propagation_result.current_cca_df
        self.store_cca_df()

        # Update cell cycle info LabelItems
        self.ax1_newMothBudLinesItem.setData([], [])
        self.ax1_oldMothBudLinesItem.setData([], [])
        self.ax2_newMothBudLinesItem.setData([], [])
        self.ax2_oldMothBudLinesItem.setData([], [])
        self.drawAllMothBudLines()
        self.setAllTextAnnotations()

        if self.ccaTableWin is not None:
            zoomIDs = self.exporting_view.getZoomIDs()
            self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)

        for frame_i in propagation_result.undo_frame_indices:
            cca_df_i = self.get_cca_df(frame_i=frame_i, return_df=True)
            self.storeUndoRedoCca(frame_i, cca_df_i, undoId)

        for frame_i, cca_df_i in propagation_result.updated_cca_dfs_by_frame.items():
            if frame_i == posData.frame_i:
                continue
            self.store_cca_df(frame_i=frame_i, cca_df=cca_df_i, autosave=False)

        self.enqAutosave()

    def warnMotherNotEligible(self, new_mothID, budID, i, why):
        if why == 'not_G1_in_the_future':
            err_msg = html_utils.paragraph(f"""
                The requested cell in G1 (ID={new_mothID})
                at future frame {i+1} has a bud assigned to it,
                therefore it cannot be assigned as the mother
                of bud ID {budID}.<br><br>
                You can assign a cell as the mother of bud ID {budID}
                only if this cell is in G1 for the
                entire life of the bud.<br><br>
                One possible solution is to click on "cancel", go to
                frame {i+1} and  assign the bud of cell {new_mothID}
                to another cell.\n'
                A second solution is to assign bud ID {budID} to cell
                {new_mothID} anyway by clicking "Apply".<br><br>
                However to ensure correctness of
                future assignments Cell-ACDC will delete any cell cycle
                information from frame {i+1} to the end. Therefore, you
                will have to visit those frames again.<br><br>
                The deletion of cell cycle information
                <b>CANNOT BE UNDONE!</b>
                Saved data is not changed of course.<br><br>
                Apply assignment or cancel process?
            """)
            applyButton = widgets.okPushButton(isDefault=False)
            applyButton.setText('Apply and remove future annotations')
            msg = widgets.myMessageBox()
            _, applyButton = msg.warning(
               self.host, 'Cell not eligible', err_msg,
               buttonsTexts=('Cancel', applyButton)
            )
            cancel = msg.cancel
            apply = msg.clickedButton == applyButton
        elif why == 'not_G1_in_the_past':
            err_msg = html_utils.paragraph(f"""
                The requested cell in G1
                (ID={new_mothID}) at past frame {i+1}
                has a bud assigned to it, therefore it cannot be
                assigned as mother of bud ID {budID}.<br>
                You can assign a cell as the mother of bud ID {budID}
                only if this cell is in G1 for the entire life of the bud.<br>
                One possible solution is to first go to frame {i+1} and
                assign the bud of cell {new_mothID} to another cell.
            """)
            msg = widgets.myMessageBox()
            msg.warning(
               self.host, 'Cell not eligible', err_msg
            )
            cancel = msg.cancel
            apply = False
        elif why == 'single_frame_G1_duration':
            err_msg = html_utils.paragraph(f"""
                Assigning bud ID {budID} to cell ID {new_mothID} would result
                in <b>no G1 phase at all</b> between previous cell cycle and
                current cell cycle (see frame n. {i+1}).<br><br>

                The solution is to annotate division on cell ID {new_mothID}
                on any frame before the frame number {i+1}, and then
                proceed to correcting the bud assignment.<br><br>

                This will gurantee a G1 duration for the cell {new_mothID}
                of <b>at least 1 frame</b>.<br><br>
                Thank you for your patience!
            """)
            msg = widgets.myMessageBox()
            msg.warning(
               self.host, 'Cell not eligible', err_msg
            )
            cancel = msg.cancel
            apply = False
        return cancel, apply

    def warnSettingHistoryKnownCellsFirstFrame(self, ID):
        txt = html_utils.paragraph(f"""
            Cell ID {ID} is a cell that is <b>present since the first
            frame.</b><br><br>
            These cells already have history UNKNOWN assigned and the
            history status <b>cannot be changed.</b>
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self.host, 'First frame cells', txt
        )

    def checkMothEligibility(self, budID, new_mothID):
        """
        Check that the new mother is in G1 for the entire life of the bud
        and that the G1 duration is > than 1 frame
        """
        last_cca_frame_i = self.navigateScrollBar.maximum()-1
        posData = self.data[self.pos_i]
        future_cca_frames = (
            (future_i, self.get_cca_df(frame_i=future_i, return_df=True))
            for future_i in range(posData.frame_i, posData.SizeT)
        )
        past_cca_frames = (
            (past_i, self.get_cca_df(frame_i=past_i, return_df=True))
            for past_i in range(posData.frame_i-1, -1, -1)
        )
        result = self.view_model.cca_workflows.mother_assignment_eligibility(
            budID,
            new_mothID,
            future_cca_frames,
            past_cca_frames,
            last_cca_frame_i,
        )
        if result.future_issue is not None:
            issue = result.future_issue
            cancel, apply = self.warnMotherNotEligible(
                new_mothID, budID, issue.frame_i, issue.reason
            )
            if apply:
                self.resetCcaFuture(issue.frame_i)
            elif cancel or issue.blocks_assignment:
                return False

        if result.past_issue is not None:
            issue = result.past_issue
            self.warnMotherNotEligible(
                new_mothID, budID, issue.frame_i, issue.reason
            )
            return False

        return True

    def checkMothersExcludedOrDead(self):
        try:
            posData = self.data[self.pos_i]
            acdc_df_i = posData.allData_li[posData.frame_i]['acdc_df']
            
            buds_df = posData.cca_df[
                (posData.cca_df.relationship == 'bud')
                & (posData.cca_df.emerg_frame_i == posData.frame_i)
            ]
            mother_ids = buds_df.relative_ID.to_list() if not buds_df.empty else []
            excluded_mother_ids = self.view_model.check_mothers_exclusion_or_dead(
                acdc_df_i, mother_ids
            )
            
            if not excluded_mother_ids:
                self.stopBlinkingPairItem()
                return True
                
            bud_ids = []
            for m_id in excluded_mother_ids:
                b_id = buds_df[buds_df.relative_ID == m_id].index.tolist()[0]
                bud_ids.append(b_id)
                
            proceed = self.warnDeadOrExcludedMothers(
                bud_ids, excluded_mother_ids
            )
            return proceed
        except Exception as e:
            self.logger.info(traceback.format_exc())
            print('-'*100)
            self.logger.warning(
                'Checking if mother cell is excluded or dead failed.'
            )
            print('^'*100)
            return False

    def checkDivisionCanBeUndone(self, ID, relID):
        """Check that division annotation can be undone (see Notes section)

        Parameters
        ----------
        ID : int
            Cell ID of the clicked cell in G1
        relID : _type_
            Relative ID of the cell that was clicked

        Notes
        -----
        Division annotation can be undone only if `relID` is also in G1 for the
        entire duration of the correction
        """
        posData = self.data[self.pos_i]
        future_cca_frames = (
            (future_i, self.get_cca_df(frame_i=future_i, return_df=True))
            for future_i in range(posData.frame_i+1, posData.SizeT)
        )
        past_cca_frames = (
            (past_i, self.get_cca_df(frame_i=past_i, return_df=True))
            for past_i in range(posData.frame_i-1, -1, -1)
        )
        return self.view_model.cca_workflows.division_undo_blocking_frame(
            ID,
            relID,
            posData.frame_i,
            posData.cca_df,
            future_cca_frames=future_cca_frames,
            past_cca_frames=past_cca_frames,
        )


    def stopBlinkingPairItem(self):
        self.ax1_newMothBudLinesItem.setOpacity(1.0)
        self.ax1_oldMothBudLinesItem.setOpacity(1.0)

        self.warnPairingItem.setData([], [])
        try:
            self.blinkPairingItemTimer.stop()
        except Exception as e:
            pass

    def warnDeadOrExcludedMothers(self, budIDs, mothIDs):
        self.startBlinkingPairingItem(budIDs, mothIDs)
        msg = widgets.myMessageBox(wrapText=False)
        pairings = [
            f'Mother ID {mID} --> bud ID {bID}'
            for mID, bID in zip(mothIDs, budIDs)
        ]
        txt = html_utils.paragraph(f"""
            The <b>mother</b> cell in the following mother-bud pairings
            (blinking line on the image) is<br>
            <b>excluded from the analysis or dead</b>:
            {html_utils.to_list(pairings)}
        """)
        msg.warning(
            self.host, 'Mother cell is excluded or dead', txt,
            buttonsTexts=('Cancel', 'Ok')
        )
        return not msg.cancel

    def startBlinkingPairingItem(self, budIDs, mothIDs):
        self.ax1_newMothBudLinesItem.setOpacity(0.2)
        self.ax1_oldMothBudLinesItem.setOpacity(0.2)

        posData = self.data[self.pos_i]
        acdc_df_i = posData.allData_li[posData.frame_i]['acdc_df']

        # Blink one pairing at the time (the first found)
        xc_b = acdc_df_i.loc[budIDs[0], 'x_centroid']
        yc_b = acdc_df_i.loc[budIDs[0], 'y_centroid']

        xc_m = acdc_df_i.loc[mothIDs[0], 'x_centroid']
        yc_m = acdc_df_i.loc[mothIDs[0], 'y_centroid']

        self.warnPairingItem.setData([xc_b, xc_m], [yc_b, yc_m])

        self.blinkPairingItemTimer = QTimer()
        self.blinkPairingItemTimer.flag = True
        self.blinkPairingItemTimer.timeout.connect(self.blinkPairingItem)
        self.blinkPairingItemTimer.start(300)

    def blinkPairingItem(self):
        if self.blinkPairingItemTimer.flag:
            opacity = 0.3
            self.blinkPairingItemTimer.flag = False
        else:
            opacity = 1.0
            self.blinkPairingItemTimer.flag = True
        self.warnPairingItem.setOpacity(opacity)

    def annotateBudToDifferentMother(self):
        """
        This function is used for correcting automatic mother-bud assignment.

        It can be called at any frame of the bud life.

        There are three cells involved: bud, current mother, new mother.

        Eligibility:
            - User clicked first on a bud (checked at click time)
            - User released mouse button on a cell in G1 (checked at release time)
            - The new mother MUST be in G1 for all the frames of the bud life
              --> if not warn
            - The new mother MUST have appeared in current frame OR be already
              in G1 in previous frame, otherwise there would be no G1 cycle

        Result:
            - The bud only changes relative ID to the new mother
            - The new mother changes relative ID and stage to 'S'
            - The old mother changes its entire status to the status it had
              before being assigned to the clicked bud
        """
        posData = self.data[self.pos_i]
        lab2D = self.get_2Dlab(posData.lab)
        budID = lab2D[self.yClickBud, self.xClickBud]
        new_mothID = lab2D[self.yClickMoth, self.xClickMoth]

        if budID == new_mothID:
            return

        if not self.isSnapshot:
            eligible = self.checkMothEligibility(budID, new_mothID)
            if not eligible:
                return

            budEligible = self.checkChangeMotherBudEligible(
                budID, posData.frame_i
            )
            if not budEligible:
                return

        # Allow partial initialization of cca_df with mouse
        if  posData.frame_i == 0:
            newMothCcs = posData.cca_df.at[new_mothID, 'cell_cycle_stage']
            if not newMothCcs == 'G1':
                err_msg = (
                    'You are assigning the bud to a cell that is not in G1!'
                )
                msg = QMessageBox()
                msg.critical(
                   self.host, 'New mother not in G1!', err_msg, msg.Ok
                )
                return
            # Store cca_df for undo action
            undoId = uuid.uuid4()
            self.storeUndoRedoCca(0, posData.cca_df, undoId)
            propagation_result = self.view_model.cca_workflows.propagate_bud_mother_assignment(
                posData.cca_df,
                posData.frame_i,
                budID,
                new_mothID,
            )
            posData.cca_df = propagation_result.current_cca_df
            self.updateAllImages()
            self.store_cca_df()
            return

        curr_moth_cca = None
        curr_mothID = posData.cca_df.at[budID, 'relative_ID']
        if curr_mothID in posData.cca_df.index:
            curr_moth_cca = self.view_model.cca_workflows.previous_relative_status_before_bud_emergence(
                budID,
                curr_mothID,
                (
                    (i, self.get_cca_df(frame_i=i, return_df=True))
                    for i in range(posData.frame_i-1, -1, -1)
                ),
                self.view_model.cca_workflows.base_status(base_cca_dict),
            )

        # Store cca_df for undo action
        undoId = uuid.uuid4()
        self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)

        future_cca_frames = (
            (frame_i, self.get_cca_df(frame_i=frame_i, return_df=True))
            for frame_i in range(posData.frame_i + 1, posData.SizeT)
        )
        past_cca_frames = (
            (frame_i, self.get_cca_df(frame_i=frame_i, return_df=True))
            for frame_i in range(posData.frame_i - 1, -1, -1)
        )
        propagation_result = self.view_model.cca_workflows.propagate_bud_mother_assignment(
            posData.cca_df,
            posData.frame_i,
            budID,
            new_mothID,
            future_cca_frames=future_cca_frames,
            past_cca_frames=past_cca_frames,
            previous_mother_status=curr_moth_cca,
        )
        posData.cca_df = propagation_result.current_cca_df

        self.updateAllImages()

        # self.checkMultiBudMoth(draw=True)
        self.store_cca_df()
        proceed = self.checkMothersExcludedOrDead()
        if not proceed:
            # User clicked on cancel in the message box
            self.UndoCca()
            return

        if self.ccaTableWin is not None:
            zoomIDs = self.exporting_view.getZoomIDs()
            self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)

        for frame_i in propagation_result.undo_frame_indices:
            cca_df_i = self.get_cca_df(frame_i=frame_i, return_df=True)
            self.storeUndoRedoCca(frame_i, cca_df_i, undoId)

        for frame_i, cca_df_i in propagation_result.updated_cca_dfs_by_frame.items():
            if frame_i == posData.frame_i:
                continue
            self.store_cca_df(frame_i=frame_i, cca_df=cca_df_i, autosave=False)

        self.enqAutosave()

    def onMotherNotInG1(self, mothID):
        txt = html_utils.paragraph(
            f'You clicked on <b>ID={mothID}</b> which is <b>NOT in G1</b><br><br>'
            'Do you want to proceed with <b>swapping the mother cells</b>?<br><br>'
            'NOTE: To assign a bud <b>start by clicking on the bud</b> '
            'and release on a cell in G1'
        )
        msg = widgets.myMessageBox()
        swapMothersButton = widgets.reloadPushButton('Swap mother cells')
        _, swapMothersButton = msg.warning(
            self.host, 'Released on a cell NOT in G1', txt,
            buttonsTexts=('Cancel', swapMothersButton)
        )
        if msg.cancel:
            return

        pairings = self.checkSwapMothersEligibility()
        if pairings is None:
            self.logger.info('Swapping mothers is not possible.')
            return

        self.swapMothers(*pairings)

    def warnBudAnnotatedDividedInFuture(
            self, budID, motherID, future_division_frame_i,
            action='swap mother cells'
        ):
        posData = self.data[self.pos_i]

        txt = html_utils.paragraph(f"""
            Bud ID {budID} is annotated as divided from mother ID {motherID}
            at frame n. {future_division_frame_i+1},<br>
            therefore it is not possible to {action}.<br><br>
            We <b>recommend reinitializing cell cycle annotations</b> on any
            frame<br> between frames number {posData.frame_i+1} and
            {future_division_frame_i} before attempting to {action}.<br><br>
            Thank you for your patience!
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self.host, f'{action} not possible'.title(), txt)
        return

    def warnMotherNotAtLeastOneFrameG1(self, budID, motherID, frame_no_G1):
        posData = self.data[self.pos_i]

        txt = html_utils.paragraph(f"""
            Assigning bud ID {budID} to cell ID {motherID} cannot be
            done because cell ID {motherID} is not in G1 at frame n.
            {frame_no_G1}.<br><br>
            This would result in no G1 phase between previous cell cycle of
            cell ID {motherID} and current one.
            This is unfortunately not allowed.<br><br>
            One possible solution is to annotate division on cell ID
            {motherID} on any frame before frame n. {frame_no_G1}.<br><br>
            Thank you for your patience!
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self.host, 'Swap mothers not possible', txt)
        return

    def checkChangeMotherBudEligible(self, budID, frame_i):
        posData = self.data[self.pos_i]
        future_cca_frames = (
            (future_i, self.get_cca_df(frame_i=future_i, return_df=True))
            for future_i in range(frame_i, posData.SizeT)
        )
        result = self.view_model.cca_workflows.bud_mother_change_eligibility(budID, future_cca_frames)
        if result.can_change:
            return True

        future_division = result.future_division
        self.warnBudAnnotatedDividedInFuture(
            budID,
            future_division.mother_id,
            future_division.frame_i,
            action='change mother cell',
        )
        return False

    def checkSwapMothersEligibility(self):
        posData = self.data[self.pos_i]

        lab2D = self.get_2Dlab(posData.lab)
        budID = lab2D[self.yClickBud, self.xClickBud]
        otherMothID = lab2D[self.yClickMoth, self.xClickMoth]
        mothID = posData.cca_df.at[budID, 'relative_ID']
        otherBudID = posData.cca_df.at[otherMothID, 'relative_ID']

        future_cca_frames = [
            (future_i, self.get_cca_df(frame_i=future_i, return_df=True))
            for future_i in range(posData.frame_i, posData.SizeT)
        ]
        past_cca_frames = [
            (past_i, self.get_cca_df(frame_i=past_i, return_df=True))
            for past_i in range(posData.frame_i, -1, -1)
        ]
        result = self.view_model.cca_workflows.swap_mothers_eligibility(
            budID,
            otherBudID,
            otherMothID,
            mothID,
            future_cca_frames,
            past_cca_frames,
        )
        if result.future_division_frame_i is not None:
            self.warnBudAnnotatedDividedInFuture(
                result.future_division_bud_id,
                result.future_division_mother_id,
                result.future_division_frame_i,
            )
            return

        if result.mother_not_g1_frame_i is not None:
            self.warnMotherNotAtLeastOneFrameG1(
                result.mother_not_g1_bud_id,
                result.mother_not_g1_mother_id,
                result.mother_not_g1_frame_i,
            )
            return

        return budID, otherBudID, otherMothID, mothID

    @exception_handler
    def swapMothers(self, budID, otherBudID, otherMothID, mothID):
        posData = self.data[self.pos_i]

        # Store cca_df for undo action
        undoId = uuid.uuid4()
        self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)

        self.logger.info(
            f'Swapping assignments (requested at frame n. {posData.frame_i+1}):\n'
            f'  * Bud ID {budID} --> mother ID {otherMothID}\n'
            f'  * Bud ID {otherBudID} --> mother ID {mothID}'
        )

        past_cca_frames = [
            (past_i, self.get_cca_df(frame_i=past_i, return_df=True))
            for past_i in range(posData.frame_i-1, -1, -1)
        ]
        future_cca_frames = [
            (future_i, self.get_cca_df(frame_i=future_i, return_df=True))
            for future_i in range(posData.frame_i+1, posData.SizeT)
        ]
        propagation_result = self.view_model.cca_workflows.propagate_swap_mothers_assignment(
            posData.cca_df,
            posData.frame_i,
            budID,
            otherBudID,
            otherMothID,
            mothID,
            past_cca_frames=past_cca_frames,
            future_cca_frames=future_cca_frames,
            base_status=self.view_model.cca_workflows.base_status(base_cca_dict),
        )
        posData.cca_df = propagation_result.current_cca_df
        self.store_cca_df()

        for frame_i, cca_df_i in propagation_result.updated_cca_dfs_by_frame.items():
            if frame_i == posData.frame_i:
                continue
            self.store_cca_df(frame_i=frame_i, cca_df=cca_df_i, autosave=False)

        self.updateAllImages()
