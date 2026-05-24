"""Qt view adapter for undo, redo, and future-frame propagation."""

from __future__ import annotations

import uuid

from cellacdc import apps, html_utils, widgets


from collections import defaultdict

from .label_editing import LabelEditing


class UndoRedo(LabelEditing):
    """Extracted from guiWin."""

    def UndoCca(self):
        posData = self.data[self.pos_i]
        # Undo current ccaState
        storeState = False
        if self.UndoCount == 0:
            undoId = uuid.uuid4()
            self.addCcaState(posData.frame_i, posData.cca_df, undoId)
            storeState = True

        # Get previously stored state
        self.UndoCount += 1
        currentCcaStates = posData.UndoRedoCcaStates[posData.frame_i]
        prevCcaState = currentCcaStates[self.UndoCount]
        posData.cca_df = prevCcaState["cca_df"]
        self.store_cca_df()
        self.updateAllImages()

        # Check if we have undone all states
        if len(currentCcaStates) > self.UndoCount:
            # There are no states left to undo for current frame_i
            self.undoAction.setEnabled(False)

        # Undo all past and future frames that has a last status inserted
        # when modyfing current frame
        prevStateId = prevCcaState["id"]
        for frame_i in range(0, posData.SizeT):
            if storeState:
                cca_df_i = self.get_cca_df(frame_i=frame_i, return_df=True)
                if cca_df_i is None:
                    break
                # Store current state to enable redoing it
                self.addCcaState(frame_i, cca_df_i, undoId)

            CcaStates_i = posData.UndoRedoCcaStates[frame_i]
            if len(CcaStates_i) <= self.UndoCount:
                # There are no states to undo for frame_i
                continue

            CcaState_i = CcaStates_i[self.UndoCount]
            id_i = CcaState_i["id"]
            if id_i != prevStateId:
                # The id of the state in frame_i is different from current frame
                continue

            cca_df_i = CcaState_i["cca_df"]
            self.store_cca_df(frame_i=frame_i, cca_df=cca_df_i, autosave=False)

        self.resetWillDivideInfo()
        self.enqAutosave()

    def addCcaState(self, frame_i, cca_df, undoId):
        posData = self.data[self.pos_i]
        posData.UndoRedoCcaStates[frame_i].insert(
            0, {"id": undoId, "cca_df": cca_df.copy()}
        )

    def addCurrentState(self, storeImage=False, storeOnlyZoom=False):
        posData = self.data[self.pos_i]
        if posData.cca_df is not None:
            cca_df = posData.cca_df.copy()
        else:
            cca_df = None

        if storeImage:
            image = self.img1.image.copy()
        else:
            image = None

        if storeOnlyZoom:
            labels, crop_slice = transformation.crop_2D(
                self.currentLab2D, self.ax1.viewRange(), tolerance=10, return_copy=False
            )
            if self.isSegm3D:
                z = self.z_lab(checkIfProj=True)
                if z is None:
                    z_slice = slice(0, len(posData.lab))
                    crop_slice = (z_slice, *crop_slice)
                    labels = posData.lab[crop_slice].copy()
                else:
                    z_slice = z
                    crop_slice = (z_slice, *crop_slice)
                    labels = labels.copy()
            else:
                labels = labels.copy()
        else:
            labels = posData.lab.copy()
            crop_slice = None

        state = {
            "image": image,
            "labels": labels,
            "editID_info": posData.editID_info.copy(),
            "binnedIDs": posData.binnedIDs.copy(),
            "keptObejctsIDs": self.keptObjectsIDs.copy(),
            "ripIDs": posData.ripIDs.copy(),
            "cca_df": cca_df,
            "crop_slice": crop_slice,
        }
        posData.UndoRedoStates[posData.frame_i].insert(0, state)

    def askPropagateChangePast(self, change_txt):
        txt = html_utils.paragraph(f"""
            Do you want to propagate the change "{change_txt}" to the past frames?
        """)
        msg = widgets.myMessageBox(wrapText=False)
        yesButton, _ = msg.question(
            self, "Propagate change to past frames", txt, buttonsTexts=("Yes", "No")
        )
        return msg.clickedButton == yesButton

    def clearUndoQueue(self):
        posData = self.data[self.pos_i]
        self.UndoCount = 0
        self.redoAction.setEnabled(False)
        self.undoAction.setEnabled(False)
        posData.UndoRedoStates = [[] for _ in range(posData.SizeT)]
        posData.UndoRedoCcaStates = [[] for _ in range(posData.SizeT)]
        if hasattr(self, "undoAddPointQueueMapper"):
            self.undoAddPointQueueMapper = defaultdict(list)

    def getCurrentState(self):
        posData = self.data[self.pos_i]
        i = posData.frame_i
        c = self.UndoCount
        state = posData.UndoRedoStates[i][c]
        if state["image"] is None:
            image_left = None
        else:
            image_left = state["image"].copy()

        crop_slice = state["crop_slice"]
        if crop_slice is None:
            posData.lab = state["labels"].copy()
        elif self.isSegm3D:
            z_slice, slice_y, slice_x = crop_slice
            posData.lab[..., z_slice, slice_y, slice_x] = state["labels"].copy()
        else:
            slice_y, slice_x = crop_slice
            posData.lab[..., slice_y, slice_x] = state["labels"].copy()

        posData.editID_info = state["editID_info"].copy()
        posData.binnedIDs = state["binnedIDs"].copy()
        posData.ripIDs = state["ripIDs"].copy()
        self.keptObjectsIDs = state["keptObejctsIDs"].copy()
        cca_df = state["cca_df"]
        if cca_df is not None:
            posData.cca_df = state["cca_df"].copy()
        else:
            posData.cca_df = None
        return image_left

    def propagateChange(
        self,
        modID,
        modTxt,
        doNotShow,
        UndoFutFrames,
        applyFutFrames,
        applyTrackingB=False,
        force=False,
    ):
        """
        This function determines whether there are already visited future frames
        that contains "modID". If so, it triggers a pop-up asking the user
        what to do (propagate change to future frames o not)
        """
        posData = self.data[self.pos_i]
        # Do not check the future for the last frame
        if posData.frame_i + 1 == posData.SizeT:
            # No future frames to propagate the change to
            return False, False, None, doNotShow

        includeUnvisited = posData.includeUnvisitedInfo.get(modTxt, False)
        areFutureIDs_affected = []
        # Get number of future frames already visited and check if future
        # frames has an ID affected by the change
        last_tracked_i_found = False
        segmSizeT = len(posData.segm_data)
        for i in range(posData.frame_i + 1, segmSizeT):
            if posData.allData_li[i]["labels"] is None:
                if not last_tracked_i_found:
                    # We set last tracked frame at -1 first None found
                    last_tracked_i = i - 1
                    last_tracked_i_found = True
                if not includeUnvisited:
                    # Stop at last visited frame since includeUnvisited = False
                    break
                else:
                    lab = posData.segm_data[i]
            else:
                lab = posData.allData_li[i]["labels"]

            if modID in lab:
                areFutureIDs_affected.append(True)

        if not last_tracked_i_found:
            # All frames have been visited in segm&track mode
            last_tracked_i = posData.SizeT - 1

        if last_tracked_i == posData.frame_i and not includeUnvisited:
            # No future frames to propagate the change to
            return False, False, None, doNotShow

        if not areFutureIDs_affected and not force:
            # There are future frames but they are not affected by the change
            return UndoFutFrames, False, None, doNotShow

        # Ask what to do unless the user has previously checked doNotShowAgain
        if doNotShow:
            endFrame_i = last_tracked_i
            if applyFutFrames and not UndoFutFrames and modTxt == "Edit ID":
                self.whitelistSyncIDsOG(frame_is=range(posData.frame_i, endFrame_i + 1))
            return UndoFutFrames, applyFutFrames, endFrame_i, doNotShow
        else:
            addApplyAllButton = (
                modTxt == "Delete ID"
                or modTxt == "Edit ID"
                or modTxt == "Assign new ID"
            )
            ffa = apps.FutureFramesAction_QDialog(
                posData.frame_i + 1,
                last_tracked_i,
                modTxt,
                applyTrackingB=applyTrackingB,
                parent=self,
                addApplyAllButton=addApplyAllButton,
            )
            ffa.exec_()
            decision = ffa.decision

            if decision is None:
                return None, None, None, doNotShow

            endFrame_i = ffa.endFrame_i
            doNotShowAgain = ffa.doNotShowCheckbox.isChecked()
            askAction = self.askHowFutureFramesActions[modTxt]
            askAction.setChecked(not doNotShowAgain)
            askAction.setDisabled(False)

            self.onlyTracking = False
            if decision == "apply_and_reinit":
                UndoFutFrames = True
                applyFutFrames = False
            elif decision == "apply_and_NOTreinit":
                UndoFutFrames = False
                applyFutFrames = False
            elif decision == "apply_to_all_visited":
                UndoFutFrames = False
                applyFutFrames = True
            elif decision == "only_tracking":
                UndoFutFrames = False
                applyFutFrames = True
                self.onlyTracking = True
            elif decision == "apply_to_all":
                UndoFutFrames = False
                applyFutFrames = True
                posData.includeUnvisitedInfo[modTxt] = True

            if applyFutFrames and not UndoFutFrames and modTxt == "Edit ID":
                self.whitelistSyncIDsOG(frame_is=range(posData.frame_i, endFrame_i + 1))
        return UndoFutFrames, applyFutFrames, endFrame_i, doNotShowAgain

    def propagateMergeObjsPast(self, IDs_to_merge):
        self.store_data(autosave=False)
        posData = self.data[self.pos_i]
        current_frame_i = posData.frame_i
        for past_frame_i in range(posData.frame_i - 1, -1, -1):
            posData.frame_i = past_frame_i
            self.get_data()

            IDs = posData.allData_li[past_frame_i]["IDs"]
            stop_loop = False
            for ID in IDs_to_merge:
                if ID not in IDs:
                    stop_loop = True
                    break

                if ID == 0:
                    continue
                posData.lab[posData.lab == ID] = self.firstID
                self.update_rp()

                self.store_data(autosave=False)

            if stop_loop:
                break

        posData.frame_i = current_frame_i
        self.get_data()

    def redo(self):
        posData = self.data[self.pos_i]
        # Get previously stored state
        if self.UndoCount > 0:
            self.UndoCount -= 1
            # Since we have redone then it is possible to undo
            self.undoAction.setEnabled(True)

            # Restore state
            image_left = self.getCurrentState()
            self.update_rp()
            self.updateAllImages(image=image_left)
            self.store_data()

        if not self.UndoCount > 0:
            # We have redone all available states
            self.redoAction.setEnabled(False)

        if self.whitelistIDsButton.isChecked():
            self.whitelistHighlightIDs()

    def storeUndoRedoCca(self, frame_i, cca_df, undoId):
        if self.isSnapshot:
            # For snapshot mode we don't store anything because we have only
            # segmentation undo action active
            return
        """
        Store current cca_df along with a unique id to know which cca_df needs
        to be restored
        """

        posData = self.data[self.pos_i]

        # Restart count from the most recent state (index 0)
        # NOTE: index 0 is most recent state before doing last change
        self.UndoCcaCount = 0
        self.undoAction.setEnabled(True)

        self.addCcaState(frame_i, cca_df, undoId)

        # Keep only 10 Undo/Redo states
        if len(posData.UndoRedoCcaStates[frame_i]) > 10:
            posData.UndoRedoCcaStates[frame_i].pop(-1)

    def storeUndoRedoStates(self, UndoFutFrames, storeImage=False, storeOnlyZoom=False):
        posData = self.data[self.pos_i]
        if UndoFutFrames:
            # Since we modified current frame all future frames that were already
            # visited are not valid anymore. Undo changes there
            self.reInitLastSegmFrame(updateImages=False)

        # Keep only 5 Undo/Redo states
        if len(posData.UndoRedoStates[posData.frame_i]) > 5:
            posData.UndoRedoStates[posData.frame_i].pop(-1)

        # Restart count from the most recent state (index 0)
        # NOTE: index 0 is most recent state before doing last change
        self.UndoCount = 0
        self.undoAction.setEnabled(True)
        self.addCurrentState(storeImage=storeImage, storeOnlyZoom=storeOnlyZoom)

    def undo(self):
        addPointsByClickingButton = self.buttonAddPointsByClickingActive()
        if addPointsByClickingButton is not None:
            done = self.undoAddPoint(addPointsByClickingButton.action)
            if done:
                return

        if self.UndoCount == 0:
            # Store current state to enable redoing it
            self.addCurrentState()

        posData = self.data[self.pos_i]
        # Get previously stored state
        if self.UndoCount < len(posData.UndoRedoStates[posData.frame_i]) - 1:
            self.UndoCount += 1
            # Since we have undone then it is possible to redo
            self.redoAction.setEnabled(True)

            # Restore state
            image_left = self.getCurrentState()
            self.update_rp()
            self.updateAllImages(image=image_left)
            self.store_data()

        if not self.UndoCount < len(posData.UndoRedoStates[posData.frame_i]) - 1:
            # We have undone all available states
            self.undoAction.setEnabled(False)

        if self.whitelistIDsButton.isChecked():
            self.whitelistHighlightIDs()

    def undoCustomAnnotation(self):
        pass
