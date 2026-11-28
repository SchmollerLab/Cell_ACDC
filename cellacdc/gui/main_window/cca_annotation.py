"""Manual cell cycle annotation (clicks in the Cell cycle analysis mode) of
the main GUI. Mixins of guiWin (gui/main_window/window.py) only: the
measurements viewer never composes them.
"""
import re
import traceback
import uuid
from typing import Literal
import pandas as pd
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QMessageBox
from cellacdc import exception_handler
from cellacdc import base_cca_dict
from cellacdc import cca_df_colnames
from cellacdc import apps, html_utils
from cellacdc import core, myutils
from cellacdc import widgets
from cellacdc import _warnings
from cellacdc import qutils
from . import constants as gui_main_constants


class CcaManualAnnotationGuiElements:
    """Manual cell cycle annotation in the 'Cell cycle analysis' mode:
    division, bud-mother assignment and swaps, history knowledge,
    will-divide, eligibility warnings, manual cca edits, annotation of
    past frames and of single mother-bud pairs, YeastMate
    auto-assignment.
    """

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

    def _getSMBPCcaDfWillDivide(
            self, 
            past_frame_i: int, 
            current_frame_i: int, 
            ID: int,
            gen_num: int
        ):
        posData = self.data[self.pos_i]
        prev_moth_bud_pairs_cca = (
            posData.allData_li[past_frame_i].get('moth_bud_pairs_cca')
        )
        if prev_moth_bud_pairs_cca is None:
            # There is no stored info before --> stop
            return

        if ID not in prev_moth_bud_pairs_cca.index:
            # Division has been annotated on a cell that does not have 
            # previous info in the past
            return
            
        if prev_moth_bud_pairs_cca.at[ID, 'generation_num'] != gen_num:
            # ID is a mother and the cell cycle is finished here
            return
        
        return prev_moth_bud_pairs_cca

    def _getCcaDfWillDivide(
            self, 
            past_frame_i: int, 
            current_frame_i: int, 
            ID: int,
            prev_moth_bud_pairs_cca: pd.DataFrame | None,
            gen_num: int
        ):
        past_cca_df = self.get_cca_df(
            frame_i=past_frame_i, return_df=True
        )
        if past_cca_df is None:
            return

        if ID not in past_cca_df.index:
            # ID is a bud and is not emerged yet here
            return
        
        if prev_moth_bud_pairs_cca is not None:                
            if prev_moth_bud_pairs_cca.at[ID, 'generation_num'] != gen_num:
                # ID is a mother and the cell cycle is finished here
                return
        else:
            if past_cca_df.at[ID, 'generation_num'] != gen_num:
                # ID is a mother and the cell cycle is finished here
                return
        
        return past_cca_df

    def annotateWillDivide(self, ID, relID, frame_i=None):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i

        if frame_i == 0:
            return

        moth_bud_pairs_cca = (
            posData.allData_li[frame_i-1].get('moth_bud_pairs_cca')
        )
        cca_df = self.get_cca_df(
            frame_i=frame_i-1, return_df=True
        )
        if moth_bud_pairs_cca is None:
            ref_cca_df = cca_df
        elif ID in moth_bud_pairs_cca.index:
            ref_cca_df = moth_bud_pairs_cca
        elif cca_df is not None:
            ref_cca_df = cca_df

        gen_num = ref_cca_df.at[ID, 'generation_num']

        # Store in the past frames that division has been annotated
        for past_frame_i in range(frame_i-1, -1, -1):
            prev_moth_bud_pairs_cca = self._getSMBPCcaDfWillDivide(
                past_frame_i, frame_i, ID, gen_num
            )
            if prev_moth_bud_pairs_cca is not None:
                prev_moth_bud_pairs_cca.at[ID, 'will_divide'] = 1
                prev_moth_bud_pairs_cca.at[relID, 'will_divide'] = 1

            past_cca_df = self._getCcaDfWillDivide(
                past_frame_i, frame_i, ID, prev_moth_bud_pairs_cca, gen_num
            )
            if past_cca_df is not None:
                past_cca_df.at[ID, 'will_divide'] = 1
                past_cca_df.at[relID, 'will_divide'] = 1

                self.store_cca_df(
                    cca_df=past_cca_df, frame_i=past_frame_i, autosave=False
                )

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
        
        moth_bud_pair_cca = (
            posData.allData_li[frame_i].get('moth_bud_pairs_cca')
        )
        
        self.annotateDivision(
            cca_df_at_future_division, 
            mothIDofDisappearedBud, 
            wrongBudID, 
            moth_bud_pair_cca,
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

    def annotateDivision(
            self, 
            cca_df: pd.DataFrame, 
            ID: int, 
            relID: int, 
            moth_bud_pair_cca: pd.DataFrame | None,
            frame_i=None
        ):
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
        
        if moth_bud_pair_cca is None:
            return store
        
        if ID not in moth_bud_pair_cca.index:
            return store
        
        moth_bud_pair_cca.loc[ID] = cca_df.loc[ID]
        moth_bud_pair_cca.loc[relID] = cca_df.loc[relID]
        
        return store

    def annotateDivisionMotherButPairCca(
            self, 
            ID: int, 
            relID: int, 
            frame_i: int,
            division_frame_i: int,
            clicked_ccs: str,
        ):
        posData = self.data[self.pos_i]
        moth_bud_pair_cca = (
            posData.allData_li[frame_i].get('moth_bud_pairs_cca')
        )
        if moth_bud_pair_cca is None:
            return 
        
        if ID not in moth_bud_pair_cca.index:
            return 
        
        current_relID = moth_bud_pair_cca.at[ID, 'relative_ID']
        if current_relID != relID:
            # Cell in the future changes relative ID
            return
        
        ccs = moth_bud_pair_cca.at[ID, 'cell_cycle_stage']
        if clicked_ccs != ccs:
            # Cell in the future changes stage --> this stage is finished
            return
        
        if clicked_ccs == 'S':
            moth_bud_pair_cca.at[ID, 'cell_cycle_stage'] = 'G1'
            moth_bud_pair_cca.at[relID, 'cell_cycle_stage'] = 'G1'
            gen_num_clickedID = moth_bud_pair_cca.at[ID, 'generation_num']
            moth_bud_pair_cca.at[ID, 'generation_num'] += 1
            moth_bud_pair_cca.at[ID, 'division_frame_i'] = division_frame_i    
            gen_num_relID = moth_bud_pair_cca.at[relID, 'generation_num']
            moth_bud_pair_cca.at[relID, 'generation_num'] = gen_num_relID+1
            moth_bud_pair_cca.at[relID, 'division_frame_i'] = division_frame_i
            if gen_num_clickedID < gen_num_relID:
                moth_bud_pair_cca.at[ID, 'relationship'] = 'mother'
            else:
                moth_bud_pair_cca.at[relID, 'relationship'] = 'mother'
        else:
            moth_bud_pair_cca.at[ID, 'cell_cycle_stage'] = 'S'
            gen_num_clickedID = moth_bud_pair_cca.at[ID, 'generation_num']
            moth_bud_pair_cca.at[ID, 'generation_num'] -= 1
            moth_bud_pair_cca.at[ID, 'division_frame_i'] = -1
            moth_bud_pair_cca.at[relID, 'cell_cycle_stage'] = 'S'
            gen_num_relID = moth_bud_pair_cca.at[relID, 'generation_num']
            moth_bud_pair_cca.at[relID, 'generation_num'] -= 1
            moth_bud_pair_cca.at[relID, 'division_frame_i'] = -1
            if gen_num_clickedID < gen_num_relID:
                moth_bud_pair_cca.at[ID, 'relationship'] = 'bud'
            else:
                moth_bud_pair_cca.at[relID, 'relationship'] = 'bud'
            moth_bud_pair_cca.at[ID, 'will_divide'] = 0
            moth_bud_pair_cca.at[relID, 'will_divide'] = 0


    def undoDivisionAnnotation(
            self, 
            cca_df: pd.DataFrame, 
            ID: int, 
            relID: int, 
            moth_bud_pair_cca: pd.DataFrame | None,
            frame_i=None
        ):
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
        
        if moth_bud_pair_cca is None:
            return store
        
        if ID not in moth_bud_pair_cca.index:
            return store
        
        moth_bud_pair_cca.loc[ID] = cca_df.loc[ID]
        moth_bud_pair_cca.loc[relID] = cca_df.loc[relID]
        
        return store

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
            
        self.store_cca_df()

        # Update cell cycle info LabelItems
        self.setAllTextAnnotations()

        if self.ccaTableWin is not None:
            zoomIDs = self.getZoomIDs()
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

        if self.annotateSingleMotherBudPairButton.isChecked():
            mothID = self.annotateSingleMothBudPairState['mother_ID']
            budID = self.annotateSingleMothBudPairState['bud_ID']
            if ID != mothID and ID != budID:
                warn_text = (f'[WARNING]: Only the mother-bud pair {mothID}-{budID} can be annotated')
                htmlTxt = f'<font color="orange">{warn_text}</font>'
                self.titleLabel.setText(htmlTxt)
                self.logger.info(warn_text)
                blinker = qutils.QControlBlink(
                    self.annotateSingleMotherBudPairButton, 
                    qparent=self
                )
                blinker.start()
                return

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
        
        moth_bud_pair_cca = (
            posData.allData_li[posData.frame_i].get('moth_bud_pairs_cca')
        )
        
        if clicked_ccs == 'S':
            self.annotateDivision(posData.cca_df, ID, relID, moth_bud_pair_cca)
            self.store_cca_df()
        else:
            self.undoDivisionAnnotation(
                posData.cca_df, ID, relID, moth_bud_pair_cca
            )
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
                # --> check if moth_bud_pair_cca is present
                self.annotateDivisionMotherButPairCca(
                    ID, relID, future_i, posData.frame_i, clicked_ccs
                )
                continue

            self.storeUndoRedoCca(future_i, cca_df_i, undoId)
            IDs = cca_df_i.index
            if ID not in IDs:
                # For some reason ID disappeared from this frame
                continue
            
            moth_bud_pair_cca = (
                posData.allData_li[future_i].get('moth_bud_pairs_cca')
            )
            
            ccs = cca_df_i.at[ID, 'cell_cycle_stage']
            relID = cca_df_i.at[ID, 'relative_ID']
            if clicked_ccs == 'S':
                if ccs == 'G1':
                    # Cell is in G1 in the future again so stop annotating
                    break
                self.annotateDivision(cca_df_i, ID, relID, moth_bud_pair_cca)
                self.store_cca_df(
                    frame_i=future_i, cca_df=cca_df_i, autosave=False
                )
            elif ccs == 'S':
                # Cell is in S in the future again so stop undoing (break)
                # also leave a 1 frame duration G1 to avoid a continuous
                # S phase
                self.annotateDivision(cca_df_i, ID, relID, moth_bud_pair_cca)
                self.store_cca_df(
                    frame_i=future_i, cca_df=cca_df_i, autosave=False
                )
                break
            else:
                self.undoDivisionAnnotation(
                    cca_df_i, ID, relID, moth_bud_pair_cca
                )
                self.store_cca_df(
                    frame_i=future_i, cca_df=cca_df_i, autosave=False
                )
        
        # Correct past frames
        for past_i in range(posData.frame_i-1, -1, -1):
            cca_df_i = self.get_cca_df(frame_i=past_i, return_df=True)
            if ID not in cca_df_i.index or relID not in cca_df_i.index:
                # Bud did not exist at frame_i = i
                break
            
            moth_bud_pair_cca = (
                posData.allData_li[past_i].get('moth_bud_pairs_cca')
            )
            
            self.storeUndoRedoCca(past_i, cca_df_i, undoId)
            ccs = cca_df_i.at[ID, 'cell_cycle_stage']
            relID = cca_df_i.at[ID, 'relative_ID']
            if ccs == 'S':
                # We correct only those frames in which the ID was in 'G1'
                break
            else:
                store = self.undoDivisionAnnotation(
                    cca_df_i, ID, relID, moth_bud_pair_cca
                )
                self.store_cca_df(
                    frame_i=past_i, cca_df=cca_df_i, autosave=False
                )
        
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
            self, 'Mother cell is excluded or dead', txt, 
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

    def checkChangeMotherBudEligible(self, budID, frame_i):
        result = self._checkBudFutureNoDivision(budID, frame_i)
        if result is None:
            return True
        
        self.warnBudAnnotatedDividedInFuture(
            budID, *result, action='change mother cell'
        )
        return False

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

    def manualEditCcaToolbarActionTriggered(self):
        self.manualEditCca()

    def manualEditCca(self, checked=True):
        if self.annotateSingleMotherBudPairButton.isChecked():
            blinker = qutils.QControlBlink(
                self.annotateSingleMotherBudPairButton, 
                qparent=self
            )
            blinker.start()

            _warnings.warnEditCcaDisabledInAnnotSingleMothBudMode(qparent=self)
            return

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
            
            for ID, changes_ID in changes.items():
                if ID not in cca_df_i.index:
                    continue
                for col, (oldValue, newValue) in changes_ID.items():
                    cca_df_i.at[ID, col] = newValue
            self.store_cca_df(frame_i=i, cca_df=cca_df_i, autosave=False)
        self.get_data()
        self.updateAllImages()

    def getHoveredMotherBudIDs(self):
        posData = self.data[self.pos_i]
        
        hoverID = self.getLastHoveredID()
        try:
            ccs = posData.cca_df.loc[hoverID, 'cell_cycle_stage']
        except Exception as err:
            ccs = 'G1'
        
        cca_df = posData.cca_df
        cca_df_S = cca_df[cca_df['cell_cycle_stage'] == 'S']
        IDs_in_S = cca_df_S.index.to_list()
        
        if not IDs_in_S:
            _warnings.warnNoIDsInS(qparent=self)
            return
        
        while ccs != 'S':
            win = apps.QLineEditDialog(
                title='Selected ID not in S',
                msg='The <b>cell cycle stage</b> of the selected ID '
                    '<b>is not "S"</b>.<br><br>'
                    'Enter the ID (mother or bud) '
                    'that you want to annotate.',
                parent=self, 
                isInteger=True,
                allowedValues=IDs_in_S,
                defaultTxt=IDs_in_S[0]
            )
            win.exec_()
            if win.cancel:
                return
            
            hoverID = win.EntryID
            ccs = cca_df.loc[hoverID, 'cell_cycle_stage']
        
        relationship = posData.cca_df.loc[hoverID, 'relationship']
        relative_ID = posData.cca_df.loc[hoverID, 'relative_ID']
        mothID = hoverID if relationship == 'mother' else relative_ID
        budID = hoverID if relationship == 'bud' else relative_ID
        
        return int(mothID), int(budID)

    def annotateSingleMotherBudPair_cb(self, checked):
        posData = self.data[self.pos_i]
        self.modeComboBox.setDisabled(checked)
        self.assignBudMothButton.setDisabled(checked)
        if checked:
            self.store_data()
            
            mothIDbudID = self.getHoveredMotherBudIDs()
            
            if mothIDbudID is None:
                self.annotateSingleMotherBudPairButton.setChecked(False)
                return
            
            mothID, budID = mothIDbudID
            
            self.logger.info(
                'Setting annotation mode for single mother-bud pair = '
                f'{(mothID, budID)}, at frame n. {posData.frame_i+1}'
            )
            
            self.annotateSingleMothBudPairState['doWarnLostObj'] = (
                self.warnLostCellsAction.isChecked()
            )
            self.annotateSingleMothBudPairState['doAnnotateLostObjs'] = (
                self.annotLostObjsToggle.isChecked()
            )
            
            self.annotateSingleMothBudPairState['mother_ID'] = (
                mothID
            )
            self.annotateSingleMothBudPairState['bud_ID'] = (
                budID
            )
            self.annotateSingleMothBudPairState['frame_i_to_restore'] = (
                posData.frame_i
            )
            self.annotateSingleMothBudPairState['last_cca_frame_i'] = (
                self.navigateScrollBar.maximum()-1
            )
            self.updateSingleMotherBudPairLastAnnotatedFrame()
            
            self.warnLostCellsAction.setChecked(False)
            self.annotLostObjsToggle.setChecked(False)
            
            self.ax1.sigRangeChanged.connect(self.highlightManualAnnotMode)
            self.ax1.setHighlighted(True, color='green')
            
            self.setEnabledCcaToolbar(False)
            self.ccaToolBar.setVisible(True)
            self.annotateSingleMotherBudPairButton.setEnabled(True)
            self.annotateSingleMotherBudPairButton.action.setVisible(True)
        else:
            frame_to_restore = self.annotateSingleMothBudPairState.get(
                'frame_i_to_restore'
            )
            if frame_to_restore is None:
                return            
            
            self.logger.info(
                f'Restoring view to frame n. {frame_to_restore+1}...'
            )
            last_cca_frame_i_to_restore = (
                self.annotateSingleMothBudPairState['last_cca_frame_i']
            )
            
            self.annotateSingleMotherBudPairRestoreLastCcaFrame(
                last_cca_frame_i_to_restore
            )

            posData.frame_i = frame_to_restore
            self.get_data()

            self.updateAllImages()
            self.updateScrollbars()
            # Disconnect only this slot, a bare `disconnect()` would also
            # disconnect `viewRangeChanged`
            try:
                self.ax1.sigRangeChanged.disconnect(
                    self.highlightManualAnnotMode
                )
            except (TypeError, RuntimeError):
                pass
            self.ax1.setHighlighted(False)
            
            self.warnLostCellsAction.setChecked(
                self.annotateSingleMothBudPairState['doWarnLostObj']
            )
            self.annotLostObjsToggle.setChecked(
                self.annotateSingleMothBudPairState['doAnnotateLostObjs']
            )
            
            self.annotateSingleMothBudPairState = {
                'last_annot_frame_i': 
                    self.annotateSingleMothBudPairState['last_annot_frame_i'],
            }
            
            self.setEnabledCcaToolbar(True)
            
            QTimer.singleShot(200, self.autoRange)

    def manualAnnotPast_cb(self, checked):
        posData = self.data[self.pos_i]
        if checked:
            for _ in range(3):
                self.onEscape(
                    buttonsToNotUncheck=[self.manualAnnotPastButton],
                    doAutoRange=False
                )

            self.brushButton.setChecked(True)
            self.store_data()
            self.manualAnnotState = {
                'editID': self.editIDspinbox.value(),
                'isAutoID': self.autoIDcheckbox.isChecked(),
                'doWarnLostObj': self.warnLostCellsAction.isChecked(),
            }
            self.autoIDcheckbox.setChecked(False)
            self.warnLostCellsAction.setChecked(False)
            hoverID = self.getLastHoveredID()
            if hoverID == 0:
                win = apps.QLineEditDialog(
                    title='Not hovering any ID',
                    msg='You are not hovering on any ID.\n'
                        'Enter the ID that you want to lock.',
                    parent=self, 
                    isInteger=True,
                    defaultTxt=self.setBrushID(return_val=True)
                )
                win.exec_()
                if win.cancel:
                    self.manualAnnotPastButton.setChecked(False)
                    return
                hoverID = win.EntryID
            self.logger.info(
                'Setting manual annotation for ID = '
                f'{hoverID}, at frame n. {posData.frame_i+1}'
            )
            self.editIDspinbox.setValue(hoverID)
            try:
                obj = posData.rp.get_obj_from_ID(hoverID)
                radius = 0.9 * obj.minor_axis_length / 2 # math.sqrt(obj.area/math.pi)*0.9
                self.brushSizeSpinbox.setValue(round(radius))
            except Exception as err:
                pass
            
            self.manualAnnotState['frame_i_to_restore'] = posData.frame_i
            self.manualAnnotState['last_tracked_i'] = (
                self.navigateScrollBar.maximum()-1
            )
            self.ax1.sigRangeChanged.connect(self.highlightManualAnnotMode)
            self.ax1.setHighlighted(True, color='green')
        else:
            self.setStatusBarLabel()  
            self.autoIDcheckbox.setChecked(self.manualAnnotState['isAutoID'])
            self.editIDspinbox.setValue(self.manualAnnotState['editID'])
            self.warnLostCellsAction.setChecked(
                self.manualAnnotState['doWarnLostObj']
            )
            frame_to_restore = self.manualAnnotState.get('frame_i_to_restore')
            if frame_to_restore is None:
                return            
            
            self.store_data()
            self.store_manual_annot_data()
            
            last_tracked_i_to_restore = self.manualAnnotState['last_tracked_i']
            self.manualAnnotRestoreLastTrackedFrame(last_tracked_i_to_restore)
            
            self.logger.info(
                f'Restoring view to frame n. {posData.frame_i+1}...'
            )
            posData.frame_i = frame_to_restore
            self.get_data()
            self.updateAllImages()
            self.updateScrollbars()
            # Disconnect only this slot, a bare `disconnect()` would also
            # disconnect `viewRangeChanged`
            try:
                self.ax1.sigRangeChanged.disconnect(
                    self.highlightManualAnnotMode
                )
            except (TypeError, RuntimeError):
                pass
            self.ax1.setHighlighted(False)
            QTimer.singleShot(150, self.autoRange)
        
        self.setManualAnnotModeEnabledTools(checked)

    def highlightManualAnnotMode(self, viewBox, viewRange):
        self.ax1.setHighlighted(True)

    def updateHighlightedAxis(self):
        color = None
        if self.manualAnnotPastButton.isChecked():
            frame_to_restore = self.manualAnnotState.get('frame_i_to_restore')
            posData = self.data[self.pos_i]
            if posData.frame_i == frame_to_restore:
                color = 'green'
            elif posData.frame_i < frame_to_restore:
                color = 'gold'
            else:
                color = 'red'
        
        if self.annotateSingleMotherBudPairButton.isChecked():
            frame_to_restore = self.annotateSingleMothBudPairState.get(
                'frame_i_to_restore'
            )
            posData = self.data[self.pos_i]
            if posData.frame_i == frame_to_restore:
                color = 'green'
            elif posData.frame_i < frame_to_restore:
                color = 'gold'
            else:
                color = 'red'
        
        if color is None:
            return 
                
        self.ax1.setHighlightingRectItemsColor(color)

    @exception_handler
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
        # Read the model parameters (from its isolated env if used)
        model_spec = self.prepareModelExecution(model_name)
        if model_spec is None:
            return
        init_params, segment_params, url = model_spec

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
            self.getModelExecution(model_name).cancel_preparation()
            self.titleLabel.setText('Segmentation aborted.')
            return

        use_gpu = myutils.check_gpu_requested_segm_model(win.init_kwargs)
        proceed = myutils.check_gpu_available(
            model_name, use_gpu, qparent=self,
            python_exec=self.getModelExecution(model_name).env_python,
        )
        if not proceed:
            self.getModelExecution(model_name).cancel_preparation()
            self.logger.info('Segmentation process cancelled.')
            self.titleLabel.setText('Segmentation process cancelled.')
            return
            
        self.model_kwargs = win.model_kwargs
        model = self.initSegmModel(
            model_name, win.init_kwargs, posData, replace=self.models[idx]
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
        model.model_name = model_name

        img = self.getDisplayedImg1()

        with self.modelExecutionBusy(f'{model_name} is predicting budding events...'):
            posData.cca_df = model.predictCcaState(img, posData.lab)
        self.store_data()
        self.updateAllImages()

        self.titleLabel.setText('Budding event prediction done.', color='g')

    def manualAnnotRestoreLastTrackedFrame(self, last_tracked_i_to_restore):
        if self.navigateScrollBar.maximum()-1 <= last_tracked_i_to_restore:
            return
        
        posData = self.data[self.pos_i]
        for frame_i in range(last_tracked_i_to_restore+1, posData.SizeT):
            data_frame_i = myutils.get_empty_stored_data_dict()
            
            data_frame_i['manually_edited_lab'] = (
                posData.allData_li[frame_i]['manually_edited_lab']
            )
            
            posData.allData_li[frame_i] = data_frame_i
        
        posData.last_tracked_i = last_tracked_i_to_restore
        self.setNavigateScrollBarMaximum()

    def annotateSingleMotherBudPairRestoreLastCcaFrame(
            self, cca_frame_i_to_restore
        ):
        if self.navigateScrollBar.maximum()-1 <= cca_frame_i_to_restore:
            return
        
        self.resetCcaFuture(cca_frame_i_to_restore+1)

    def setManualAnnotModeEnabledTools(self, enabled):
        for action in self.editToolBar.actions():
            toolButton = self.editToolBar.widgetForAction(action)
            if toolButton in self.manulAnnotToolButtons:
                continue
            
            toolButton.setDisabled(enabled)  
            action.setDisabled(enabled) 

    def store_manual_annot_data(
            self, posData=None, data_frame_i=None    
        ):
        if posData is None:
            posData = self.data[self.pos_i]
        
        if data_frame_i is None:
            data_frame_i = posData.allData_li[posData.frame_i]
        
        if not self.isSegm3D:
            lab = [posData.lab]
        else:
            lab = posData.lab
            
        for z, lab_2D in enumerate(lab):
            data_frame_i['manually_edited_lab']['lab'][z] = lab_2D
            
        # data_frame_i['manually_edited_lab']['zoom_slice'] = zoom_slice

    def storeSingleMotherBudPairData(self, cca_df):
        if cca_df is None:
            return
        
        posData = self.data[self.pos_i]
        data_frame_i = posData.allData_li[posData.frame_i]        
        mothID = self.annotateSingleMothBudPairState.get('mother_ID')
        if mothID is None:
            return
        
        budID = self.annotateSingleMothBudPairState['bud_ID']
        current_moth_bud_pair_cca = cca_df.loc[[mothID, budID]].copy()
        stored_moth_bud_pairs_cca = data_frame_i.get('moth_bud_pairs_cca')
        if stored_moth_bud_pairs_cca is None:
            data_frame_i['moth_bud_pairs_cca'] = current_moth_bud_pair_cca
        else:
            if mothID in stored_moth_bud_pairs_cca.index:
                stored_moth_bud_pairs_cca = stored_moth_bud_pairs_cca.drop(
                    index=mothID
                )
            
            if budID in stored_moth_bud_pairs_cca.index:
                stored_moth_bud_pairs_cca = stored_moth_bud_pairs_cca.drop(
                    index=budID
                )
                
            data_frame_i['moth_bud_pairs_cca'] = pd.concat(
                [stored_moth_bud_pairs_cca, current_moth_bud_pair_cca],
            ).drop_duplicates()
    
        last_annot_frame_i = (
            self.annotateSingleMothBudPairState.get('last_annot_frame_i', -1)
        )
        if posData.frame_i > last_annot_frame_i:
            self.updateSingleMotherBudPairLastAnnotatedFrame(posData.frame_i)

    def updateSingleMotherBudPairLastAnnotatedFrame(self, frame_i=None):
        if not hasattr(self, 'annotateSingleMothBudPairState'):
            return
        
        last_smbp_annot_frame_i = (
            self.annotateSingleMothBudPairState.get('last_annot_frame_i', -1)
        )

        if frame_i is None:
            frame_i = last_smbp_annot_frame_i

        toolTip = self.annotateSingleMotherBudPairButton.toolTip()
        if frame_i <= 0:
            self.annotateSingleMotherBudPairButton.setStyleSheet(
                'background-color: none'
            )
            toolTip = re.sub(
                r'Last annotated frame n. = [A-Za-z0-9\.]+',
                'Last annotated frame n. = n.a.',
                toolTip
            )
            self.annotateSingleMothBudPairState['last_annot_frame_i'] = (
                -1
            )
        else:
            self.annotateSingleMotherBudPairButton.setStyleSheet(
                f'background-color: {gui_main_constants.ORANGE_HEX}'
            )
            toolTip = re.sub(
                r'Last annotated frame n. = [A-Za-z0-9\.]+',
                f'Last annotated frame n. = {frame_i+1}',
                toolTip
            )
            self.annotateSingleMothBudPairState['last_annot_frame_i'] = (
                frame_i
            )

        self.annotateSingleMotherBudPairButton.setToolTip(toolTip)

    def getSingleMotherBudPairCca_df(self):
        if not self.annotateSingleMotherBudPairButton.isChecked():
            return
        
        posData = self.data[self.pos_i]
        start_frame_i = (
            self.annotateSingleMothBudPairState['frame_i_to_restore']
        )
        
        if posData.frame_i <= start_frame_i:
            # Past frame --> no need to get cca_df
            return
        
        mothID = self.annotateSingleMothBudPairState['mother_ID']
        budID = self.annotateSingleMothBudPairState['bud_ID']
        
        # Get previous dataframe
        acdc_df = posData.allData_li[posData.frame_i-1]['acdc_df']
        prev_cca_df = acdc_df[self.cca_df_colnames].loc[[mothID, budID]].copy()
        
        return prev_cca_df

    def addCcaInfoFromSingleMotherBudPairs(
            self, 
            cca_df: pd.DataFrame, 
            dict_getter: Literal['pop', 'get'] ='pop'
        ):
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        data_dict = posData.allData_li[frame_i]
        getter = getattr(data_dict, dict_getter)
        moth_bud_pairs_cca = getter('moth_bud_pairs_cca', None)
        if moth_bud_pairs_cca is None:
            self.updateSingleMotherBudPairLastAnnotatedFrame(-1)
            return cca_df

        if cca_df is None:
            return moth_bud_pairs_cca

        if 'cell_cycle_stage' not in cca_df.columns:
            return moth_bud_pairs_cca

        # Get the other cells that have not been annotated in the future 
        # with the annotate single mother-bud pair tool from previous frame
        cca_df = pd.concat([cca_df, moth_bud_pairs_cca])
        cca_df = cca_df[~cca_df.index.duplicated(keep="last")]

        return cca_df

    def resetSingleMotherBudPairsCcaInfo(self, from_frame_i):
        posData = self.data[self.pos_i]
        data_dict = posData.allData_li[from_frame_i-1]
        moth_bud_pairs_cca = data_dict.get('moth_bud_pairs_cca', None)
        if moth_bud_pairs_cca is not None:
            moth_bud_pairs_cca['will_divide'] = 0
        
        for i in range(from_frame_i, posData.SizeT):
            data_dict = posData.allData_li[i]
            data_dict.pop('moth_bud_pairs_cca', None)
        
        last_annot_smbp_i = -1
        for i in range(posData.SizeT):
            data_dict = posData.allData_li[i]
            if data_dict.get('moth_bud_pairs_cca', None) is None:
                continue

            last_annot_smbp_i = i
        
        if last_annot_smbp_i <= from_frame_i:
            last_annot_smbp_i = -1

        self.updateSingleMotherBudPairLastAnnotatedFrame(last_annot_smbp_i)

    def initSingleMotherBudPairPartialCcaInfo(self):
        partial_cca_start_frame_i = None
        last_annot_frame_i = -1
        posData = self.data[self.pos_i]
        for frame_i in range(posData.SizeT):
            acdc_df = posData.allData_li[frame_i]['acdc_df']
            if acdc_df is None:
                break
            
            if 'is_single_mother_bud_annotation' not in acdc_df.columns:
                continue
            
            mask = acdc_df['is_single_mother_bud_annotation'] == 1.0
            if not mask.any():
                continue
            
            if 'cell_cycle_stage' not in acdc_df.columns:
                continue
            
            if not acdc_df['cell_cycle_stage'].isna().any():
                continue

            if partial_cca_start_frame_i is None:
                partial_cca_start_frame_i = frame_i

            moth_bud_pairs_cca = acdc_df[mask][cca_df_colnames]

            posData.allData_li[frame_i]['moth_bud_pairs_cca'] = (
                moth_bud_pairs_cca
            )
            last_annot_frame_i = frame_i

        if partial_cca_start_frame_i is None:
            return

        self.updateSingleMotherBudPairLastAnnotatedFrame(last_annot_frame_i)
        self.resetCcaFuture(partial_cca_start_frame_i)

    def detectSingleMotherBudPairsCcaDf(self, isQuickSave=False):
        annotateSingleMotherBudPairButtonColor = (
            self.annotateSingleMotherBudPairButton.palette()
            .button().color().name()
        )
        if annotateSingleMotherBudPairButtonColor != gui_main_constants.ORANGE_HEX:
            return True, False
        
        posData = self.data[self.pos_i]
        last_annot_frame_i = (
            self.annotateSingleMothBudPairState.get('last_annot_frame_i', -1)
        )
        if posData.frame_i >= last_annot_frame_i:
            self.updateSingleMotherBudPairLastAnnotatedFrame(-1)
            return True, False

        if isQuickSave:
            return True, True

        cancel, saveSingleMothBudPairCcaInfo = (
            _warnings.warnAskAboutSaveSingleMotherBudPairsCcaDf(self)
        )
        if cancel:
            return False, False

        return True, saveSingleMothBudPairCcaInfo

    def removeSMBPCcaInfoAfterSaving(self):
        if self._saveDataWorker.saveSMBPCcaInfo:
            return
        
        partial_cca_start_frame_i = None
        posData = self.data[self.pos_i]
        for frame_i in range(posData.SizeT):
            acdc_df = posData.allData_li[frame_i]['acdc_df']
            if acdc_df is None:
                break
            
            if 'cell_cycle_stage' not in acdc_df.columns:
                continue

            if 'is_single_mother_bud_annotation' not in acdc_df.columns:
                continue
            
            if not acdc_df['cell_cycle_stage'].isna().any():
                continue

            mask = acdc_df['is_single_mother_bud_annotation'] == 1.0
            if not mask.any():
                continue

            if partial_cca_start_frame_i is None:
                partial_cca_start_frame_i = frame_i
                break
        
        if partial_cca_start_frame_i is None:
            return

        self.resetCcaFuture(partial_cca_start_frame_i-1)
