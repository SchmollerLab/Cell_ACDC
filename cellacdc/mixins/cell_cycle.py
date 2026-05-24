"""Qt view adapter for cell-cycle annotation workflows."""

from __future__ import annotations

import traceback
import uuid

from tqdm import tqdm
import pandas as pd
from qtpy.QtCore import QMutex, QThread, QTimer, QWaitCondition
from qtpy.QtWidgets import QCheckBox, QMessageBox, QPushButton

from cellacdc import (
    apps,
    _warnings,
    base_cca_dict,
    disableWindow,
    exception_handler,
    html_utils,
)
from cellacdc import widgets, workers

from .undo_redo import UndoRedo

class CellCycle(UndoRedo):
    """Extracted from guiWin."""

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

                cont = self.getObjContours(obj)
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

    def addMissingIDs_cca_df(self, posData):
        base_cca_df = self.getBaseCca_df()
        if posData.cca_df is None:
            posData.cca_df = base_cca_df
            return
        
        posData.cca_df = posData.cca_df.combine_first(base_cca_df)

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
                   self, 'New mother not in G1!', err_msg, msg.Ok
                )
                return
            # Store cca_df for undo action
            undoId = uuid.uuid4()
            self.storeUndoRedoCca(0, posData.cca_df, undoId)
            currentRelID = posData.cca_df.at[budID, 'relative_ID']
            if currentRelID in posData.cca_df.index:
                posData.cca_df.at[currentRelID, 'relative_ID'] = -1
                posData.cca_df.at[currentRelID, 'generation_num'] = 2
                posData.cca_df.at[currentRelID, 'cell_cycle_stage'] = 'G1'
            posData.cca_df.at[budID, 'relationship'] = 'bud'
            posData.cca_df.at[budID, 'generation_num'] = 0
            posData.cca_df.at[budID, 'relative_ID'] = new_mothID
            posData.cca_df.at[budID, 'cell_cycle_stage'] = 'S'
            posData.cca_df.at[new_mothID, 'relative_ID'] = budID
            posData.cca_df.at[new_mothID, 'generation_num'] = 2
            posData.cca_df.at[new_mothID, 'cell_cycle_stage'] = 'S'
            self.updateAllImages()
            self.store_cca_df()
            return

        curr_mothID = posData.cca_df.at[budID, 'relative_ID']        
        if curr_mothID in posData.cca_df.index:
            curr_moth_cca = self.getStatus_RelID_BeforeEmergence(
                budID, curr_mothID
            )

        # Store cca_df for undo action
        undoId = uuid.uuid4()
        self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)
        
        # Correct current frames and update LabelItems
        posData.cca_df.at[budID, 'relative_ID'] = new_mothID
        posData.cca_df.at[budID, 'generation_num'] = 0
        posData.cca_df.at[budID, 'relative_ID'] = new_mothID
        posData.cca_df.at[budID, 'relationship'] = 'bud'
        posData.cca_df.at[budID, 'corrected_on_frame_i'] = posData.frame_i
        posData.cca_df.at[budID, 'cell_cycle_stage'] = 'S'

        posData.cca_df.at[new_mothID, 'relative_ID'] = budID
        posData.cca_df.at[new_mothID, 'cell_cycle_stage'] = 'S'
        posData.cca_df.at[new_mothID, 'relationship'] = 'mother'

        
        if curr_mothID in posData.cca_df.index:
            # Cells with UNKNOWN history has relative's ID = -1
            # which is not an existing cell
            posData.cca_df.loc[curr_mothID] = curr_moth_cca

        self.updateAllImages()

        # self.checkMultiBudMoth(draw=True)
        self.store_cca_df()
        proceed = self.checkMothersExcludedOrDead()
        if not proceed:
            # User clicked on cancel in the message box
            self.UndoCca()
            return

        if self.ccaTableWin is not None:
            zoomIDs = self.getZoomIDs()
            self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)

        # Correct future frames
        for i in range(posData.frame_i+1, posData.SizeT):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break

            IDs = cca_df_i.index
            if budID not in IDs or new_mothID not in IDs:
                # For some reason ID disappeared from this frame
                continue

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            bud_relationship = cca_df_i.at[budID, 'relationship']
            bud_ccs = cca_df_i.at[budID, 'cell_cycle_stage']

            if bud_relationship == 'mother' and bud_ccs == 'S':
                # The bud at the ith frame budded itself --> stop
                break

            cca_df_i.at[budID, 'relative_ID'] = new_mothID
            cca_df_i.at[budID, 'generation_num'] = 0
            cca_df_i.at[budID, 'relative_ID'] = new_mothID
            cca_df_i.at[budID, 'relationship'] = 'bud'
            cca_df_i.at[budID, 'cell_cycle_stage'] = 'S'

            newMoth_bud_ccs = cca_df_i.at[new_mothID, 'cell_cycle_stage']
            if newMoth_bud_ccs == 'G1':
                # Assign bud to new mother only if the new mother is in G1
                # This can happen if the bud already has a G1 annotated
                cca_df_i.at[new_mothID, 'relative_ID'] = budID
                cca_df_i.at[new_mothID, 'cell_cycle_stage'] = 'S'
                cca_df_i.at[new_mothID, 'relationship'] = 'mother'

            if curr_mothID in cca_df_i.index:
                # Cells with UNKNOWN history has relative's ID = -1
                # which is not an existing cell
                cca_df_i.loc[curr_mothID] = curr_moth_cca

            self.store_cca_df(frame_i=i, cca_df=cca_df_i, autosave=False)

        # Correct past frames
        for i in range(posData.frame_i-1, -1, -1):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)

            is_bud_existing = budID in cca_df_i.index
            if not is_bud_existing:
                # Bud was not emerged yet
                break

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            cca_df_i.at[budID, 'relative_ID'] = new_mothID
            cca_df_i.at[budID, 'generation_num'] = 0
            cca_df_i.at[budID, 'relative_ID'] = new_mothID
            cca_df_i.at[budID, 'relationship'] = 'bud'
            cca_df_i.at[budID, 'cell_cycle_stage'] = 'S'

            cca_df_i.at[new_mothID, 'relative_ID'] = budID
            cca_df_i.at[new_mothID, 'cell_cycle_stage'] = 'S'
            cca_df_i.at[new_mothID, 'relationship'] = 'mother'

            if curr_mothID in cca_df_i.index:
                # Cells with UNKNOWN history has relative's ID = -1
                # which is not an existing cell
                cca_df_i.loc[curr_mothID] = curr_moth_cca

            self.store_cca_df(frame_i=i, cca_df=cca_df_i, autosave=False)
        
        self.enqAutosave()

    def annotateDivision(self, cca_df, ID, relID, frame_i=None):
        # Correct as follows:
        # For frame_i > 0 --> assign to G1 and +1 on generation number
        # For frame == 0 --> reinitialize to unknown cells
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i

        self.annotateWillDivide(ID, relID)

        store = False
        cca_df.at[ID, 'cell_cycle_stage'] = 'G1'
        cca_df.at[relID, 'cell_cycle_stage'] = 'G1'
        
        if frame_i > 0:
            gen_num_clickedID = cca_df.at[ID, 'generation_num']
            cca_df.at[ID, 'generation_num'] += 1
            cca_df.at[ID, 'division_frame_i'] = frame_i    
            gen_num_relID = cca_df.at[relID, 'generation_num']
            cca_df.at[relID, 'generation_num'] = gen_num_relID+1
            cca_df.at[relID, 'division_frame_i'] = frame_i
            if gen_num_clickedID < gen_num_relID:
                cca_df.at[ID, 'relationship'] = 'mother'
            else:
                cca_df.at[relID, 'relationship'] = 'mother'
        else:
            cca_df.at[ID, 'generation_num'] = 2
            cca_df.at[relID, 'generation_num'] = 2

            cca_df.at[ID, 'division_frame_i'] = -1
            cca_df.at[relID, 'division_frame_i'] = -1

            cca_df.at[ID, 'relationship'] = 'mother' 
            cca_df.at[relID, 'relationship'] = 'mother'
        
        store = True
        return store

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
        if relID in posData.cca_df.index:
            relID_cca = self.getStatus_RelID_BeforeEmergence(ID, relID)

        if is_history_known:
            # Save status of ID when emerged to allow undoing
            statusID_whenEmerged = self.getStatusKnownHistoryBud(ID)
            if statusID_whenEmerged is None:
                return
            posData.ccaStatus_whenEmerged[ID] = statusID_whenEmerged

        # Store cca_df for undo action
        undoId = uuid.uuid4()
        self.storeUndoRedoCca(posData.frame_i, posData.cca_df, undoId)

        if ID not in posData.ccaStatus_whenEmerged:
            self.warnSettingHistoryKnownCellsFirstFrame(ID)
            return

        self.setHistoryKnowledge(ID, posData.cca_df)

        if relID in posData.cca_df.index:
            # If the cell with unknown history has a relative ID assigned to it
            # we set the cca of it to the status it had BEFORE the assignment
            posData.cca_df.loc[relID] = relID_cca

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
            zoomIDs = self.getZoomIDs()
            self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)

        # Correct future frames
        for i in range(posData.frame_i+1, posData.SizeT):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            IDs = cca_df_i.index
            if ID not in IDs:
                # For some reason ID disappeared from this frame
                continue
            else:
                self.setHistoryKnowledge(ID, cca_df_i)
                if relID in IDs:
                    cca_df_i.loc[relID] = relID_cca
                self.store_cca_df(frame_i=i, cca_df=cca_df_i, autosave=False)


        # Correct past frames
        for i in range(posData.frame_i-1, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break

            self.storeUndoRedoCca(i, cca_df_i, undoId)
            IDs = cca_df_i.index
            if ID not in IDs:
                # we reached frame where ID was not existing yet
                break
            else:
                relID = cca_df_i.at[ID, 'relative_ID']
                self.setHistoryKnowledge(ID, cca_df_i)
                if relID in IDs:
                    cca_df_i.loc[relID] = relID_cca
                self.store_cca_df(frame_i=i, cca_df=cca_df_i, autosave=False)
        
        self.enqAutosave()

    def annotateWillDivide(self, ID, relID, frame_i=None):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i

        # Store in the past frames that division has been annotated
        for past_frame_i in range(frame_i-1, -1, -1):
            past_cca_df = self.get_cca_df(frame_i=past_frame_i, return_df=True)
            if past_cca_df is None:
                return
            
            if ID not in past_cca_df.index:
                # ID is a bud and is not emerged yet here
                return
            
            if frame_i-1 == past_frame_i:
                # Get generation number at first iteration
                gen_num = past_cca_df.at[ID, 'generation_num']
                
            if past_cca_df.at[ID, 'generation_num'] != gen_num:
                # ID is a mother and the cell cycle is finished here
                return
            
            past_cca_df.at[ID, 'will_divide'] = 1
            past_cca_df.at[relID, 'will_divide'] = 1

            self.store_cca_df(
                cca_df=past_cca_df, frame_i=past_frame_i, autosave=False
            )

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
            
            for ID, changes_ID in changes.items():
                if ID not in cca_df_i.index:
                    continue
                for col, (oldValue, newValue) in changes_ID.items():
                    cca_df_i.at[ID, col] = newValue
            self.store_cca_df(frame_i=i, cca_df=cca_df_i, autosave=False)
        self.get_data()
        self.updateAllImages()

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

    def autoAssignBud_YeastMate(self):
        if not self.is_win:
            txt = (
                'YeastMate is available only on Windows OS.'
                'We are working on expading support also on macOS and Linux.\n\n'
                'Thank you for your patience!'
            )
            msg = QMessageBox()
            msg.critical(
                self, 'Supported only on Windows', txt, msg.Ok
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
            acdcSegment = myutils.import_segment_module(model_name)
            self.acdcSegment_li[idx] = acdcSegment

        # Read all models parameters
        init_params, segment_params = myutils.getModelArgSpec(acdcSegment)
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
        proceed = myutils.check_gpu_available(model_name, use_gpu, qparent=self)
        if not proceed:
            self.logger.info('Segmentation process cancelled.')
            self.titleLabel.setText('Segmentation process cancelled.')
            return
            
        self.model_kwargs = win.model_kwargs
        model = myutils.init_segm_model(acdcSegment, posData, win.init_kwargs)
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

        # Get previous dataframe
        acdc_df = posData.allData_li[posData.frame_i-1]['acdc_df']
        prev_cca_df = acdc_df[self.cca_df_colnames].copy()

        if posData.cca_df is None:
            posData.cca_df = prev_cca_df.copy()
        else:
            posData.cca_df = curr_df[self.cca_df_colnames].copy()
        
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
                cont = self.getObjContours(obj)
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

    def blinkPairingItem(self):
        if self.blinkPairingItemTimer.flag:
            opacity = 0.3
            self.blinkPairingItemTimer.flag = False
        else:
            opacity = 1.0
            self.blinkPairingItemTimer.flag = True
        self.warnPairingItem.setOpacity(opacity)

    def ccaCheckerStopChecking(self):
        if not self.ccaCheckerRunning:
            return
        
        self.ccaIntegrityCheckerWorker.clearQueue()
        
        if self.ccaIntegrityCheckerWorker.isChecking:
            self.ccaIntegrityCheckerWorker.abortChecking = True

    def ccaCheckerWorkerClosed(self, worker):
        self.logger.info('Cell cycle annotations integrity checker stopped.') 
        self.ccaCheckerRunning = False           

    def ccaCheckerWorkerDone(self):
        self.setStatusBarLabel(log=False)

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
            mask_index = np.in1d(newIDs, cca_df_i.index)
            posData.new_IDs = list(newIDs[~mask_index])
            if not posData.new_IDs:
                return found_cca_df_IDs
        return found_cca_df_IDs

    def checkChangeMotherBudEligible(self, budID, frame_i):
        result = self._checkBudFutureNoDivision(budID, frame_i)
        if result is None:
            return True
        
        self.warnBudAnnotatedDividedInFuture(
            budID, *result, action='change mother cell'
        )
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
        
        ccs_relID = posData.cca_df.at[relID, 'cell_cycle_stage']
        if ccs_relID == 'S':
            return posData.frame_i
        
        # Check future frames
        for future_i in range(posData.frame_i+1, posData.SizeT):
            cca_df_i = self.get_cca_df(frame_i=future_i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break 
            
            ccs_relID = cca_df_i.at[relID, 'cell_cycle_stage']
            if ccs_relID == 'S':
                return future_i
        
        # Check past frames
        for past_i in range(posData.frame_i-1, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=past_i, return_df=True)
            if ID not in cca_df_i.index or relID not in cca_df_i.index:
                # Bud did not exist at frame_i = i
                break
            
            ccs = cca_df_i.at[ID, 'cell_cycle_stage']
            if ccs == 'S':
                break
            
            ccs_relID = cca_df_i.at[relID, 'cell_cycle_stage']
            if ccs_relID == 'S':
                return future_i           

    def checkMothEligibility(self, budID, new_mothID):
        """
        Check that the new mother is in G1 for the entire life of the bud
        and that the G1 duration is > than 1 frame
        """
        last_cca_frame_i = self.navigateScrollBar.maximum()-1
        posData = self.data[self.pos_i]
        eligible = True

        # Check future frames
        G1_duration_future = 0
        for future_i in range(posData.frame_i, posData.SizeT):
            cca_df_i = self.get_cca_df(frame_i=future_i, return_df=True)

            if cca_df_i is None:
                # ith frame was not visited yet
                break
            
            if budID not in cca_df_i.index:
                # Bud disappeared
                break

            is_still_bud = cca_df_i.at[budID, 'relationship'] == 'bud'
            if not is_still_bud:
                break

            ccs = cca_df_i.at[new_mothID, 'cell_cycle_stage']
            if ccs != 'G1':
                cancel, apply = self.warnMotherNotEligible(
                    new_mothID, budID, future_i, 'not_G1_in_the_future'
                )
                if apply:
                    self.resetCcaFuture(future_i)
                    break
                isG1singleFrame = G1_duration_future == 1
                isFutureFrameNotLastAnnot = future_i != last_cca_frame_i
                if cancel or (isG1singleFrame and isFutureFrameNotLastAnnot):
                    eligible = False
                    return eligible
            
            G1_duration_future += 1

        # Check past frames
        for past_i in range(posData.frame_i-1, -1, -1):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=past_i, return_df=True)

            is_bud_existing = budID in cca_df_i.index
            is_moth_existing = new_mothID in cca_df_i.index

            if not is_moth_existing:
                # Mother not existing because it appeared from outside FOV
                break

            ccs = cca_df_i.at[new_mothID, 'cell_cycle_stage']
            if ccs != 'G1' and is_bud_existing:
                # Requested mother not in G1 in the past
                # during the life of the bud (is_bud_existing = True)
                self.warnMotherNotEligible(
                    new_mothID, budID, past_i, 'not_G1_in_the_past'
                )
                eligible = False
                return eligible

            if not is_bud_existing:
                # Bud stop existing --> check that mother is still in G1
                if ccs != 'G1':
                    eligible = False
                    self.warnMotherNotEligible(
                        new_mothID, budID, past_i, 'single_frame_G1_duration'
                    )
                break
            
        return eligible

    def checkMothersExcludedOrDead(self):
        try:
            posData = self.data[self.pos_i]
            buds_df = posData.cca_df[
                (posData.cca_df.relationship == 'bud')
                & (posData.cca_df.emerg_frame_i == posData.frame_i)
            ]
            acdc_df_i = posData.allData_li[posData.frame_i]['acdc_df']
            moth_df = acdc_df_i.loc[buds_df.relative_ID.to_list()]
            excluded_df = moth_df[
                (moth_df.is_cell_dead > 0) | (moth_df.is_cell_excluded > 0)
            ]
            excludedMothIDs = excluded_df.index.to_list()
            if not excludedMothIDs:
                self.stopBlinkingPairItem()
                return True
            budIDsOfExcludedMoth = excluded_df.relative_ID.to_list()
            proceed = self.warnDeadOrExcludedMothers(
                budIDsOfExcludedMoth, excludedMothIDs
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

        self.highlightNewIDs_ccaFailed(ScellsIDsGone, rp=prev_rp)
        proceed = self.warnScellsGone(ScellsIDsGone, posData.frame_i)
        self.clearLostObjContoursItems()
        
        if not proceed:
            return True, automaticallyDividedIDs

        for IDgone in ScellsIDsGone:
            relID = prev_cca_df.at[IDgone, 'relative_ID']
            self.annotateDisappearedBeforeDivision(relID, IDgone, prev_cca_df)
            self.annotateDivision(
                prev_cca_df, IDgone, relID, frame_i=posData.frame_i-1
            )
            self.annotateDivisionCurrentFrameRelativeIDgone(relID)
            automaticallyDividedIDs.append(relID)
            
        self.store_cca_df(frame_i=posData.frame_i-1, cca_df=prev_cca_df)

        return False, automaticallyDividedIDs

    def checkSwapMothersEligibility(self):
        posData = self.data[self.pos_i]
        
        lab2D = self.get_2Dlab(posData.lab)
        budID = lab2D[self.yClickBud, self.xClickBud]
        otherMothID = lab2D[self.yClickMoth, self.xClickMoth]
        mothID = posData.cca_df.at[budID, 'relative_ID']
        otherBudID = posData.cca_df.at[otherMothID, 'relative_ID']
        
        for _budID in (budID, otherBudID):
            result = self._checkBudFutureNoDivision(
                _budID, posData.frame_i
            )
            if result is None:
                continue
            
            self.warnBudAnnotatedDividedInFuture(_budID, *result)
            return
        
        correct_pairings = {
            otherBudID: mothID, budID: otherMothID
        }
        wrong_pairings = {
            mothID: budID, otherMothID: otherBudID
        }
        for correctBudID, correctMothID in correct_pairings.items():
            wrongBudID = wrong_pairings[correctMothID]
            frame_no_G1 = self._checkMothInG1beforeBudEmergence(
                correctMothID, correctBudID, wrongBudID, posData.frame_i
            )
            if frame_no_G1 is None:
                continue
            
            self.warnMotherNotAtLeastOneFrameG1(
                correctBudID, correctMothID, frame_no_G1
            )
            return
        
        return budID, otherBudID, otherMothID, mothID

    def disableCcaIntegrityChecker(self):
        self.stopCcaIntegrityCheckerWorker()

    def enqCcaIntegrityChecker(self):
        if not self.ccaCheckerRunning:
            return
        posData = self.data[self.pos_i]  
        self.ccaIntegrityCheckerWorker.enqueue(posData)

    def fixCcaDfAfterEdit(self, editTxt):
        posData = self.data[self.pos_i]
        if posData.cca_df is not None:
            # For snapshot mode we fix or reinit cca_df depending on the edit
            self.update_cca_df_snapshots(editTxt, posData)
            self.store_data()

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

    def getBaseCca_df(self, with_tree_cols=False): 
        posData = self.data[self.pos_i]
        IDs = [obj.label for obj in posData.rp]
        cca_df = core.getBaseCca_df(IDs, with_tree_cols=with_tree_cols)
        return cca_df

    def getConcatCcaDf(self):
        posData = self.data[self.pos_i]
        cca_dfs = []
        keys = []
        for frame_i in range(0, posData.SizeT):
            cca_df = self.get_cca_df(frame_i=frame_i, return_df=True)
            if cca_df is None:
                break
            
            cca_dfs.append(cca_df)
            keys.append(frame_i)
        
        if not cca_dfs:
            return
        
        global_cca_df = pd.concat(cca_dfs, keys=keys, names=['frame_i'])
        return global_cca_df

    def get_cca_df(self, frame_i=None, return_df=False, debug=False):
        # cca_df is None unless the metadata contains cell cycle annotations
        # NOTE: cell cycle annotations are either from the current session
        # or loaded from HDD in "initPosAttr" with a .question to the user
        posData = self.data[self.pos_i]
        cca_df = None
        i = posData.frame_i if frame_i is None else frame_i
        df = posData.allData_li[i]['acdc_df']            
        if df is not None:
            if 'cell_cycle_stage' in df.columns:
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

    def get_last_cca_frame_i(self):
        posData = self.data[self.pos_i]
        
        i = 0
        # Determine last annotated frame index
        for i, dict_frame_i in enumerate(posData.allData_li):
            df = dict_frame_i['acdc_df']
            if df is None:
                break
            elif 'cell_cycle_stage' not in df.columns:
                break
        
        last_cca_frame_i = i if i==0 or i+1==len(posData.allData_li) else i-1  
        
        return last_cca_frame_i

    def goToFrameNumber(self, frame_n):
        posData = self.data[self.pos_i]
        posData.frame_i = frame_n - 1
        self.get_data()
        self.updateAllImages()
        self.updateScrollbars()

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

    def highlightIDs(self, IDs, pen):
        pass

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
                self, 'Tracking was never checked', txt
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
                self, 'Go to last annotated frame?', txt, 
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
                self, 'Go to last annotated frame?', txt, 
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

    def initCcaIntegrityChecker(self):
        posData = self.data[self.pos_i]
        for frame_i, data_frame_i in enumerate(posData.allData_li):
            lab = data_frame_i['labels']
            if lab is None:
                break
            
            cca_df = self.get_cca_df(frame_i, return_df=True)
            self.store_cca_df_checker(posData, frame_i, cca_df)
        
        self.enqCcaIntegrityChecker()

    def initMissingFramesCca(self, last_cca_frame_i, current_frame_i):
        self.logger.info(
            'Initialising cell cycle annotations of missing past frames...'
        )
        posData = self.data[self.pos_i]
        current_frame_i = posData.frame_i
        
        annotated_cca_dfs = []
        for frame_i in range(last_cca_frame_i+1):
            acdc_df = posData.allData_li[frame_i]['acdc_df']
            if 'cell_cycle_stage' in acdc_df.columns:
                continue
            
            acdc_df[self.cca_df_colnames] = ''
        
        annotated_cca_dfs = [
            posData.allData_li[i]['acdc_df'][self.cca_df_colnames]
            for i in range(last_cca_frame_i+1)
        ]
        keys = range(last_cca_frame_i+1)
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

            idx = last_annotated_cca_df.index.intersection(cca_df.index)
            cca_df.loc[idx, cca_df_colnames] = last_annotated_cca_df.loc[idx]

            self.store_cca_df(cca_df=cca_df, frame_i=frame_i, autosave=False)
            pbar.update()
        pbar.close()

        posData.frame_i = current_frame_i
        self.get_data()

    def isCcaCheckerChecking(self):
        if not self.ccaCheckerRunning:
            return False
        
        return self.ccaIntegrityCheckerWorker.isChecking

    def isCurrentFrameCcaVisited(self):
        posData = self.data[self.pos_i]
        curr_df = posData.allData_li[posData.frame_i]['acdc_df']
        return curr_df is not None and 'cell_cycle_stage' in curr_df.columns

    def isFrameCcaAnnotated(self):
        posData = self.data[self.pos_i]
        acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
        if acdc_df is None:
            return False

        return 'cell_cycle_stage' in acdc_df.columns

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
                    ID, relID, issue_frame_i, qparent=self
                )
                return
        
        if clicked_ccs == 'S':
            self.annotateDivision(posData.cca_df, ID, relID)
            self.store_cca_df()
        else:
            self.undoDivisionAnnotation(posData.cca_df, ID, relID)
            self.store_cca_df()

        # Update cell cycle info LabelItems
        self.ax1_newMothBudLinesItem.setData([], [])
        self.ax1_oldMothBudLinesItem.setData([], [])
        self.ax2_newMothBudLinesItem.setData([], [])
        self.ax2_oldMothBudLinesItem.setData([], [])
        self.drawAllMothBudLines()
        self.setAllTextAnnotations()

        if self.ccaTableWin is not None:
            zoomIDs = self.getZoomIDs()
            self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)
        
        # Correct future frames
        for future_i in range(posData.frame_i+1, posData.SizeT):
            cca_df_i = self.get_cca_df(frame_i=future_i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break

            self.storeUndoRedoCca(future_i, cca_df_i, undoId)
            IDs = cca_df_i.index
            if ID not in IDs:
                # For some reason ID disappeared from this frame
                continue

            ccs = cca_df_i.at[ID, 'cell_cycle_stage']
            relID = cca_df_i.at[ID, 'relative_ID']
            if clicked_ccs == 'S':
                if ccs == 'G1':
                    # Cell is in G1 in the future again so stop annotating
                    break
                self.annotateDivision(cca_df_i, ID, relID)
                self.store_cca_df(
                    frame_i=future_i, cca_df=cca_df_i, autosave=False
                )
            elif ccs == 'S':
                # Cell is in S in the future again so stop undoing (break)
                # also leave a 1 frame duration G1 to avoid a continuous
                # S phase
                self.annotateDivision(cca_df_i, ID, relID)
                self.store_cca_df(
                    frame_i=future_i, cca_df=cca_df_i, autosave=False
                )
                break
            else:
                self.undoDivisionAnnotation(cca_df_i, ID, relID)
                self.store_cca_df(
                    frame_i=future_i, cca_df=cca_df_i, autosave=False
                )
        
        # Correct past frames
        for past_i in range(posData.frame_i-1, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=past_i, return_df=True)
            if ID not in cca_df_i.index or relID not in cca_df_i.index:
                # Bud did not exist at frame_i = i
                break

            self.storeUndoRedoCca(past_i, cca_df_i, undoId)
            ccs = cca_df_i.at[ID, 'cell_cycle_stage']
            relID = cca_df_i.at[ID, 'relative_ID']
            if ccs == 'S':
                # We correct only those frames in which the ID was in 'G1'
                break
            else:
                store = self.undoDivisionAnnotation(cca_df_i, ID, relID)
                self.store_cca_df(
                    frame_i=past_i, cca_df=cca_df_i, autosave=False
                )
        
        self.enqAutosave()

    def manualEditCca(self, checked=True):
        posData = self.data[self.pos_i]
        editCcaWidget = apps.editCcaTableWidget(
            posData.cca_df, posData.SizeT, current_frame_i=posData.frame_i,
            parent=self
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

    def manualEditCcaToolbarActionTriggered(self):
        self.manualEditCca()

    def nearest_point_2Dyx(self, points, all_others):
        """
        Given 2D array of [y, x] coordinates points and all_others return the
        [y, x] coordinates of the two points (one from points and one from all_others)
        that have the absolute minimum distance
        """
        # Compute 3D array where each ith row of each kth page is the element-wise
        # difference between kth row of points and ith row in all_others array.
        # (i.e. diff[k,i] = points[k] - all_others[i])
        diff = points[:, np.newaxis] - all_others
        # Compute 2D array of distances where
        # dist[i, j] = euclidean dist (points[i],all_others[j])
        dist = np.linalg.norm(diff, axis=2)
        # Compute i, j indexes of the absolute minimum distance
        i, j = np.unravel_index(dist.argmin(), dist.shape)
        nearest_point = all_others[j]
        point = points[i]
        min_dist = np.min(dist)
        return min_dist, nearest_point

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
            self, 'Released on a cell NOT in G1', txt,
            buttonsTexts=('Cancel', swapMothersButton)
        )
        if msg.cancel:
            return
        
        pairings = self.checkSwapMothersEligibility()
        if pairings is None:
            self.logger.info('Swapping mothers is not possible.')
            return
        
        self.swapMothers(*pairings)

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
               self, 'Re-initialize annnotations?', txt, 
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

    def resetCcaFuture(self, from_frame_i):
        posData = self.data[self.pos_i]
        self.last_cca_frame_i = from_frame_i-1
        self.ccaCheckerStopChecking()
        
        self.setNavigateScrollBarMaximum() 
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
        
        self.resetWillDivideInfo()

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

    def resetWillDivideInfo(self):
        global_cca_df = self.getConcatCcaDf()
        if global_cca_df is None:
            return
        
        global_cca_df = load._fix_will_divide(global_cca_df)
        self.storeFromConcatCcaDf(global_cca_df)

    def setCcaIssueContour(self, obj):
        objContours = self.getObjContours(obj, all_external=True)  
        for cont in objContours:
            xx = cont[:,0] + 0.5
            yy = cont[:,1] + 0.5
            self.ax1_lostObjScatterItem.addPoints(xx, yy)
        self.textAnnot[0].addObjAnnotation(
            obj, 'lost_object', f'{obj.label}?', False
        )

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

    def stopBlinkingPairItem(self):
        self.ax1_newMothBudLinesItem.setOpacity(1.0)
        self.ax1_oldMothBudLinesItem.setOpacity(1.0)
        
        self.warnPairingItem.setData([], [])
        try:
            self.blinkPairingItemTimer.stop()
        except Exception as e:
            pass

    def stopCcaIntegrityCheckerWorker(self):
        try:
            self.ccaIntegrityCheckerWorker._stop()
        except Exception as err:
            pass

    def storeFromConcatCcaDf(self, global_cca_df):
        posData = self.data[self.pos_i]
        for frame_i in range(0, posData.SizeT):
            try:
                cca_df = global_cca_df.loc[frame_i]
            except KeyError as err:
                break
            
            self.store_cca_df(frame_i=frame_i, cca_df=cca_df, autosave=False)
        
        self.get_cca_df()

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
        
        if autosave:
            self.enqAutosave()
            self.enqCcaIntegrityChecker()

    def store_cca_df_checker(self, posData, frame_i, cca_df):
        if not self.ccaCheckerRunning:
            return
        
        if cca_df is None:
            return
        
        posData.allData_li[frame_i]['cca_df_checker'] = cca_df.copy()

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
        
        correct_pairings = {
            otherBudID: mothID,
            budID: otherMothID
        }
        
        for correct_budID, correct_mothID in correct_pairings.items():
            posData.cca_df.at[correct_budID, 'relative_ID'] = correct_mothID
            posData.cca_df.at[correct_mothID, 'relative_ID'] = correct_budID
            posData.cca_df.at[correct_budID, 'corrected_on_frame_i'] = posData.frame_i
            posData.cca_df.at[correct_mothID, 'corrected_on_frame_i'] = posData.frame_i
        self.store_cca_df()
        
        # Correct past frames
        corrected_budIDs_past = set()
        for past_i in range(posData.frame_i-1, -1, -1):
            if len(corrected_budIDs_past) == 2:
                break
            
            for correct_budID, correct_mothID in correct_pairings.items():
                # Get cca_df for ith frame from allData_li
                cca_df_i = self.get_cca_df(frame_i=past_i, return_df=True)
            
                if correct_budID in corrected_budIDs_past:
                    continue

                if correct_budID not in cca_df_i.index:
                    # Bud does not exist anymore in the past
                    corrected_budIDs_past.add(correct_budID)
                    
                    if len(corrected_budIDs_past) < 2:
                        self.restoreMotherToBeforeWrongBudWasAssignedToIt(
                            correct_mothID, cca_df_i, past_i
                        )
                    continue
                
                cca_df_i.at[correct_budID, 'relative_ID'] = correct_mothID
                cca_df_i.at[correct_mothID, 'relative_ID'] = correct_budID
                cca_df_i.at[correct_budID, 'corrected_on_frame_i'] = posData.frame_i
                cca_df_i.at[correct_mothID, 'corrected_on_frame_i'] = posData.frame_i
                
                # Set mother cell cycle stage to S in case it is not
                if cca_df_i.at[correct_mothID, 'cell_cycle_stage'] == 'G1':
                    cca_df_i.at[correct_mothID, 'cell_cycle_stage'] = 'S'
                    # cca_df_i.at[correct_mothID, 'generation_num'] -= 1
            
                self.store_cca_df(
                    frame_i=past_i, cca_df=cca_df_i, autosave=False
                )
        
        # Correct future frames
        corrected_budIDs_future = set()
        for future_i in range(posData.frame_i+1, posData.SizeT):
            if len(corrected_budIDs_future) == 2:
                break
            
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=future_i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break
            
            for correct_budID, correct_mothID in correct_pairings.items():
                if correct_budID in corrected_budIDs_future:
                    # Bud already corrected in the future
                    continue
                
                if correct_budID not in cca_df_i.index:
                    # Bud disappeared in the future
                    corrected_budIDs_future.add(correct_budID)
                    continue
                
                ccs_bud = cca_df_i.at[correct_budID, 'cell_cycle_stage']
                if ccs_bud == 'G1':
                    # Bud divided in the future, annotate division between 
                    # correct mother and wrong bud and then stop correcting
                    if correct_budID not in corrected_budIDs_future:
                        corrected_budIDs_future.add(correct_budID)
                    
                    if len(corrected_budIDs_future) < 2: 
                        self.annotateDivisionFutureFramesSwapMothers(
                            cca_df_i, correct_mothID, future_i
                        )
                    continue
                
                cca_df_i.at[correct_budID, 'relative_ID'] = correct_mothID
                cca_df_i.at[correct_mothID, 'relative_ID'] = correct_budID
                cca_df_i.at[correct_budID, 'corrected_on_frame_i'] = posData.frame_i
                cca_df_i.at[correct_mothID, 'corrected_on_frame_i'] = posData.frame_i
                
                # Set mother cell cycle stage to S in case it is not
                if cca_df_i.at[correct_mothID, 'cell_cycle_stage'] == 'G1':
                    cca_df_i.at[correct_mothID, 'cell_cycle_stage'] = 'S'
                    # cca_df_i.at[correct_mothID, 'generation_num'] -= 1
            
            self.store_cca_df(frame_i=future_i, cca_df=cca_df_i, autosave=False)
        
        self.updateAllImages()

    def undoBudMothAssignment(self, ID):
        posData = self.data[self.pos_i]
        relID = posData.cca_df.at[ID, 'relative_ID']
        ccs = posData.cca_df.at[ID, 'cell_cycle_stage']
        if ccs == 'G1':
            return
        posData.cca_df.at[ID, 'relative_ID'] = -1
        posData.cca_df.at[ID, 'generation_num'] = 2
        posData.cca_df.at[ID, 'cell_cycle_stage'] = 'G1'
        posData.cca_df.at[ID, 'relationship'] = 'mother'
        if relID in posData.cca_df.index:
            posData.cca_df.at[relID, 'relative_ID'] = -1
            posData.cca_df.at[relID, 'generation_num'] = 2
            posData.cca_df.at[relID, 'cell_cycle_stage'] = 'G1'
            posData.cca_df.at[relID, 'relationship'] = 'mother'

        obj_idx = posData.IDs.index(ID)
        relObj_idx = posData.IDs.index(relID)
        rp_ID = posData.rp[obj_idx]
        rp_relID = posData.rp[relObj_idx]

        self.store_cca_df()

        # Update cell cycle info LabelItems
        self.setAllTextAnnotations()

        if self.ccaTableWin is not None:
            zoomIDs = self.getZoomIDs()
            self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)

    def undoDivisionAnnotation(self, cca_df, ID, relID):
        # Correct as follows:
        # If G1 then correct to S and -1 on generation number
        store = False
        cca_df.at[ID, 'cell_cycle_stage'] = 'S'
        gen_num_clickedID = cca_df.at[ID, 'generation_num']
        cca_df.at[ID, 'generation_num'] -= 1
        cca_df.at[ID, 'division_frame_i'] = -1
        cca_df.at[relID, 'cell_cycle_stage'] = 'S'
        gen_num_relID = cca_df.at[relID, 'generation_num']
        cca_df.at[relID, 'generation_num'] -= 1
        cca_df.at[relID, 'division_frame_i'] = -1
        if gen_num_clickedID < gen_num_relID:
            cca_df.at[ID, 'relationship'] = 'bud'
        else:
            cca_df.at[relID, 'relationship'] = 'bud'
        cca_df.at[ID, 'will_divide'] = 0
        cca_df.at[relID, 'will_divide'] = 0
        store = True
        return store

    def unstore_cca_df(self):
        posData = self.data[self.pos_i]
        acdc_df = posData.allData_li[posData.frame_i]['acdc_df']
        for col in self.cca_df_colnames:
            if col not in acdc_df.columns:
                continue
            acdc_df.drop(col, axis=1, inplace=True)

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
                    delIDexists = dataDict['IDs_idxs'].get(delID, False)
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
                    delIDexists = dataDict['IDs_idxs'].get(delID, False)
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

    def update_cca_df_newIDs(self, posData, new_IDs):
        for newID in new_IDs:
            self.addIDBaseCca_df(posData, newID)

    def update_cca_df_relabelling(self, posData, oldIDs, newIDs):
        relIDs = posData.cca_df['relative_ID']
        posData.cca_df['relative_ID'] = relIDs.replace(oldIDs, newIDs)
        mapper = dict(zip(oldIDs, newIDs))
        posData.cca_df = posData.cca_df.rename(index=mapper)

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

    def viewCcaTable(self):
        posData = self.data[self.pos_i]
        zoomIDs = self.getZoomIDs()
        
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
        
        if 'cell_cycle_stage' in df.columns:
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
            msg.warning(self, 'Table empty', txt)
            return
        
        df = posData.add_tree_cols_to_cca_df(
            current_cca_df, frame_i=posData.frame_i
        )
        if self.ccaTableWin is None:
            self.ccaTableWin = apps.ViewCcaTableWindow(df, parent=self)
            self.ccaTableWin.show()
            self.ccaTableWin.setGeometryWindow()
            self.ccaTableWin.sigUpdateCcaTable.connect(
                self.onSigUpdateCcaTableWindow
            )
        else:
            self.ccaTableWin.setFocus()
            self.ccaTableWin.activateWindow()
            self.ccaTableWin.updateTable(current_cca_df)

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
        msg.warning(self, f'{action} not possible'.title(), txt)
        return

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
            self, 'Mother cell is excluded or dead', txt, 
            buttonsTexts=('Cancel', 'Ok')
        )
        return not msg.cancel

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
                
        if acdc_df is None and self.lineage_tree is None:
            if update_images:
                self.updateAllImages()
            return True
        
        cell_cycle_stage_present = (
            acdc_df is not None and 'cell_cycle_stage' in acdc_df.columns
            )
        lineage_tree_present = (
            self.lineage_tree is not None or 'parent_ID_tree' in acdc_df.columns
        )
        if not cell_cycle_stage_present and not lineage_tree_present:
            if update_images:
                self.updateAllImages()
            return True
            
        action = self.warnEditingWithAnnotActions.get(editTxt, None)
        if action is not None and not action.isChecked():
            # user has checked that he does not want to be asked again AND he doesnt want to delete
            if update_images:
                self.updateAllImages()
            return True

        msg = widgets.myMessageBox()
        warn_type = 'cell cycle annotations' if cell_cycle_stage_present else 'lineage tree annotations'
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

    def warnFrameNeverVisitedSegmMode(self):
        msg = widgets.myMessageBox()
        warn_cca = msg.critical(
            self, 'Next frame NEVER visited',
            'Next frame was never visited in "Segmentation and Tracking"'
            'mode.\n You cannot perform cell cycle analysis on frames'
            'where segmentation and/or tracking errors were not'
            'checked/corrected.\n\n'
            'Switch to "Segmentation and Tracking" mode '
            'and check/correct next frame,\n'
            'before attempting cell cycle analysis again',
        )
        return False

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
        msg.warning(self, 'Swap mothers not possible', txt)
        return

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
               self, 'Cell not eligible', err_msg, 
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
               self, 'Cell not eligible', err_msg
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
               self, 'Cell not eligible', err_msg
            )
            cancel = msg.cancel
            apply = False
        return cancel, apply

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
           self, 'Cells in "S/G2/M" disappeared!', text,
           buttonsTexts=('Cancel', 'Yes', 'No')
        )
        return msg.clickedButton == yesButton

    def warnSettingHistoryKnownCellsFirstFrame(self, ID):
        txt = html_utils.paragraph(f"""
            Cell ID {ID} is a cell that is <b>present since the first 
            frame.</b><br><br>
            These cells already have history UNKNOWN assigned and the 
            history status <b>cannot be changed.</b>
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(
            self, 'First frame cells', txt
        )

    def getStatusKnownHistoryBud(self, ID):
        posData = self.data[self.pos_i]
        cca_df_ID = None
        for i in range(posData.frame_i-1, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)
            is_cell_existing = is_bud_existing = ID in cca_df_i.index
            if not is_cell_existing:
                bud_cca_dict = base_cca_dict.copy()
                bud_cca_dict['cell_cycle_stage'] = 'S'
                bud_cca_dict['generation_num'] = 0
                bud_cca_dict['relationship'] = 'bud'
                bud_cca_dict['emerg_frame_i'] = i+1
                bud_cca_dict['is_history_known'] = True
                cca_df_ID = pd.Series(bud_cca_dict)
                return cca_df_ID

    def setHistoryKnowledge(self, ID, cca_df):
        posData = self.data[self.pos_i]
        is_history_known = cca_df.at[ID, 'is_history_known']
        if is_history_known:
            cca_df.at[ID, 'is_history_known'] = False
            cca_df.at[ID, 'cell_cycle_stage'] = 'G1'
            cca_df.at[ID, 'generation_num'] += 2
            cca_df.at[ID, 'emerg_frame_i'] = -1
            cca_df.at[ID, 'relative_ID'] = -1
            cca_df.at[ID, 'relationship'] = 'mother'
        else:
            cca_df.loc[ID] = posData.ccaStatus_whenEmerged[ID]

    def annotateDivisionFutureFramesSwapMothers(
            self, cca_df_at_future_division, mothIDofDisappearedBud, frame_i
        ):
        """This method is called as part of `guiWin.swapMothers`. 
        
        It annotates cell division and propagates that to future frames to the 
        mother cell that stops having the correct bud because division between 
        wrong bud and other wrong mother was annotated in the future. 

        Parameters
        ----------
        cca_df_at_future_division : pd.DataFrame
            _description_
        mothIDofDisappearedBud : int
            Mother ID of the disappeared bud
        frame_i : int
            Frame since when the mother ID stops having the correct bud because 
            the correct bud was assigned as divided from the wrong mother
        """        
        posData = self.data[self.pos_i]
        
        relativeIDofMothID = cca_df_at_future_division.at[
            mothIDofDisappearedBud, 'relative_ID'
        ]
        if relativeIDofMothID not in cca_df_at_future_division.index:
            # Also wrong bud ID disappeared
            return
        
        relativeIDofMothIDrelationship = cca_df_at_future_division.at[
            relativeIDofMothID, 'relationship'
        ]
        if relativeIDofMothIDrelationship != 'bud':
            # The wrong bud ID is a cell in G1 from future cycle --> 
            # the actual wrong bud ID disappeared too.
            return
        
        wrongBudID = relativeIDofMothID
        
        self.annotateDivision(
            cca_df_at_future_division, mothIDofDisappearedBud, wrongBudID, 
            frame_i=frame_i
        )
        cca_df_at_future_division.at[
            mothIDofDisappearedBud, 'corrected_on_frame_i'] = frame_i
        self.store_cca_df(
            frame_i=frame_i, cca_df=cca_df_at_future_division, autosave=False
        )
        
        ccaStatusToRestore = cca_df_at_future_division.loc[mothIDofDisappearedBud]
        for future_i in range(frame_i+1, posData.SizeT):            
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=future_i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                break
            
            ccs = cca_df_i.at[mothIDofDisappearedBud, 'cell_cycle_stage']
            if ccs == 'G1':
                # Mother cell in G1 again, stop correcting
                break
            
            cca_df_i.loc[mothIDofDisappearedBud] = ccaStatusToRestore
            cca_df_i.at[mothIDofDisappearedBud, 'corrected_on_frame_i'] = frame_i
            
            self.store_cca_df(frame_i=future_i, cca_df=cca_df_i, autosave=False)            

    def getStatus_RelID_BeforeEmergence(self, budID, curr_mothID):
        posData = self.data[self.pos_i]
        # Get status of the current mother before it had budID assigned to it
        cca_status_before_bud_emerg = None
        for i in range(posData.frame_i-1, -1, -1):
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=i, return_df=True)

            is_bud_existing = budID in cca_df_i.index
            if not is_bud_existing:
                # Bud was not emerged yet
                if curr_mothID in cca_df_i.index:
                    cca_status_before_bud_emerg = cca_df_i.loc[curr_mothID]
                    return cca_status_before_bud_emerg
                else:
                    # The bud emerged together with the mother because
                    # they appeared together from outside of the fov
                    # and they were trated as new IDs bud in S0
                    bud_cca_dict = base_cca_dict.copy()
                    bud_cca_dict['cell_cycle_stage'] = 'S'
                    bud_cca_dict['generation_num'] = 0
                    bud_cca_dict['relationship'] = 'bud'
                    bud_cca_dict['emerg_frame_i'] = i+1
                    bud_cca_dict['is_history_known'] = True
                    cca_status_before_bud_emerg = pd.Series(bud_cca_dict)
                    return cca_status_before_bud_emerg
        
        # Mother did not have a status before bud emergence because it was
        # already paired with bud at first frame --> reinit to default
        cca_status_before_bud_emerg = (
            core.getBaseCca_df([curr_mothID]).loc[curr_mothID]
        )
        return cca_status_before_bud_emerg

    def _checkBudFutureNoDivision(self, budID, start_frame_i):
        posData = self.data[self.pos_i]
        
        future_i = start_frame_i
        for future_i in range(start_frame_i, posData.SizeT):
            if future_i == 0:
                continue
            
            # Get cca_df for ith frame from allData_li
            cca_df_i = self.get_cca_df(frame_i=future_i, return_df=True)
            if cca_df_i is None:
                # ith frame was not visited yet
                return
            
            if budID not in cca_df_i.index:
                # Bud disappears in the future --> fine
                return
            
            ccs = cca_df_i.at[budID, 'cell_cycle_stage']
            if ccs == 'G1':
                return future_i, cca_df_i.at[budID, 'relative_ID']

    def _checkMothInG1beforeBudEmergence(
            self, motherID, budID, wrongBudID, start_frame_i
        ):
        """Check that mother is in G1 on the frame before bud emergence

        Parameters
        ----------
        motherID : int
            ID of mother cell
        budID : int
            ID of bud
        start_frame_i : int
            Frame index from which to start checking in the past
        """        
        for past_i in range(start_frame_i, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=past_i, return_df=True)            
            if budID not in cca_df_i.index:
                if cca_df_i.at[motherID, 'cell_cycle_stage'] == 'G1':
                    return
                
                budID_prev_cycle = cca_df_i.at[motherID, 'relative_ID']
                if budID_prev_cycle != wrongBudID:
                    return past_i + 1
                
                break

    def restoreMotherToBeforeWrongBudWasAssignedToIt(
            self, mothIDofDisappearedBud, 
            cca_df_at_correct_bud_ID_disappearance, 
            frame_i
        ):
        """This method is called as part of `guiWin.swapMothers`. 

        Parameters
        ----------
        mothIDofDisappearedBud : int
            Mother ID of the disappeared bud
        cca_df_at_correct_bud_ID_disappearance : pd.DataFrame
            Cell cycle annotations DataFrame when the correct bud ID stopped 
            existing (before emergence)
        frame_i : int
            Frame index when the correct bud ID stopped existing 
            (before emergence)
        
        Note
        ----
        It restores the mother cell cycle annotations to the status it had 
        before the wrong bud was assigned to it. 
        
        We need to do it only if the swapMothers past frames loop is still 
        iterating to correct the other bud.
        
        We also need to do this only if the wrong bud ID is actually a bud.
        
        When we swap mothers in the past frames it can be that the correct bud 
        ID stops existing (before emergence). In this case the correct mother 
        still has the wrong bud assigned to ID so we need to restore the status 
        it had before the wrong bud was assigned to it. 
        
        To determine the status we go back until the wrong bud disappear. That 
        is the frame before the wrong bud was assigned to the mother we want to 
        correct. This is the status we want to restore.
        
        When we go back in time it could be that the wrong bud never disappears 
        becuase it is already emerged at frame 0. In this case the status we 
        want to restore at is the default G1 status at frame 0. 
        """      
        relativeIDofMothID = cca_df_at_correct_bud_ID_disappearance.at[
            mothIDofDisappearedBud, 'relative_ID'
        ]
        if relativeIDofMothID not in cca_df_at_correct_bud_ID_disappearance.index:
            # Also wrong bud ID disappeared
            return
        
        relativeIDofMothIDrelationship = cca_df_at_correct_bud_ID_disappearance.at[
            relativeIDofMothID, 'relationship'
        ]
        if relativeIDofMothIDrelationship != 'bud':
            # The wrong bud ID is a cell in G1 from previous cycle --> 
            # the actual wrong bud ID disappeared too.
            return
        
        wrongBudID = relativeIDofMothID
        
        mothCcaBeforeWrongBudID = base_cca_dict
        # Search in the past for status of mother before wrong bud emerged
        for past_i in range(frame_i, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=past_i, return_df=True)
            if wrongBudID not in cca_df_i.index:
                mothCcaBeforeWrongBudID = cca_df_i.loc[mothIDofDisappearedBud]
                break
               
        # Restore in past frames the correct mother status
        for past_i in range(frame_i, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=past_i, return_df=True)
            if wrongBudID in cca_df_i.index:
                cca_df_i.loc[mothIDofDisappearedBud] = mothCcaBeforeWrongBudID
                cca_df_i.at[mothIDofDisappearedBud, 'corrected_on_frame_i'] = frame_i
                self.store_cca_df(
                    frame_i=past_i, cca_df=cca_df_i, autosave=False
                )
            else:
                break

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
