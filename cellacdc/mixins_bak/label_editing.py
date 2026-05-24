"""Qt view adapter for label-editing workflows."""

from __future__ import annotations

import math

import numpy as np
import skimage.measure
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QAction

from cellacdc import apps, disableWindow, exception_handler


class LabelEditingMixin:
    """Qt-facing adapter around manual label editing."""

    """Headless decisions for manual label editing."""

    # @exec_time

    # @exec_time

    def _get_editID_info(self, df):
        if "was_manually_edited" not in df.columns:
            return []

        if "y_centroid" not in df.columns or "x_centroid" not in df.columns:
            df = self.addYXcentroidToDf(df)

        manually_edited_df = df[df["was_manually_edited"] > 0]
        editID_info = [
            (row.y_centroid, row.x_centroid, row.Index)
            for row in manually_edited_df.itertuples()
        ]
        return editID_info

    def _update_zslices_rp(self):
        if not self.isSegm3D:
            return

        posData = self.data[self.pos_i]
        posData.zSlicesRp = {}
        for z, lab2d in enumerate(posData.lab):
            lab2d_rp = skimage.measure.regionprops(lab2d)
            posData.zSlicesRp[z] = {obj.label: obj for obj in lab2d_rp}

    def addYXcentroidToDf(self, df):
        posData = self.data[self.pos_i]
        for obj in posData.rp:
            y_centroid = int(self.getObjCentroid(obj.centroid)[0])
            x_centroid = int(self.getObjCentroid(obj.centroid)[1])
            df.at[obj.label, "y_centroid"] = y_centroid
            df.at[obj.label, "x_centroid"] = x_centroid
        return df

    def applyEditID(
        self,
        clickedID,
        currentIDs,
        oldIDnewIDMapper,
        clicked_x,
        clicked_y,
        shift=False,
        doPropagateUnvisited=False,
    ):
        posData = self.data[self.pos_i]

        # Ask to propagate change to all future visited frames
        key = "Edit ID"
        askAction = self.askHowFutureFramesActions[key]
        doNotShow = not askAction.isChecked()
        (UndoFutFrames, applyFutFrames, endFrame_i, doNotShowAgain) = (
            self.propagateChange(
                clickedID,
                key,
                doNotShow,
                posData.UndoFutFrames_EditID,
                posData.applyFutFrames_EditID,
                applyTrackingB=True,
            )
        )

        if UndoFutFrames is None:
            return

        if shift and self.isSegm3D:
            lab = self.get_2Dlab(posData.lab)
        else:
            lab = posData.lab

        # Store undo state before modifying stuff
        self.storeUndoRedoStates(UndoFutFrames)
        maxID = max(posData.IDs, default=0)
        for old_ID, new_ID in oldIDnewIDMapper:
            if new_ID in currentIDs and not self.editIDmergeIDs:
                tempID = maxID + 1
                lab[lab == old_ID] = maxID + 1
                lab[lab == new_ID] = old_ID
                lab[lab == tempID] = new_ID
                maxID += 1

                old_ID_idx = currentIDs.index(old_ID)
                new_ID_idx = currentIDs.index(new_ID)

                # Append information for replicating the edit in tracking
                # List of tuples (y, x, replacing ID)
                objo = posData.rp[old_ID_idx]
                yo, xo = self.getObjCentroid(objo.centroid)
                objn = posData.rp[new_ID_idx]
                yn, xn = self.getObjCentroid(objn.centroid)
                if not math.isnan(yo) and not math.isnan(yn):
                    yn, xn = int(yn), int(xn)
                    posData.editID_info.append((yn, xn, new_ID))
                    yo, xo = int(clicked_y), int(clicked_x)
                    posData.editID_info.append((yo, xo, old_ID))
            else:
                lab[lab == old_ID] = new_ID
                if new_ID > maxID:
                    maxID = new_ID
                old_ID_idx = posData.IDs.index(old_ID)

                # Append information for replicating the edit in tracking
                # List of tuples (y, x, replacing ID)
                obj = posData.rp[old_ID_idx]
                y, x = self.getObjCentroid(obj.centroid)
                if not math.isnan(y) and not math.isnan(y):
                    y, x = int(y), int(x)
                    posData.editID_info.append((y, x, new_ID))

            self.updateAssignedObjsAcdcTrackerSecondStep(new_ID)

        if shift and self.isSegm3D:
            self.set_2Dlab(lab)

        # Update rps
        self.update_rp()

        # Since we manually changed an ID we don't want to repeat tracking
        self.setAllTextAnnotations()
        self.highlightLostNew()
        # self.checkIDsMultiContour()

        # Update colors for the edited IDs
        self.updateLookuptable()

        if self.isSnapshot:
            self.fixCcaDfAfterEdit("Edit ID")
            self.updateAllImages()
        else:
            self.warnEditingWithCca_df("Edit ID", update_images=False)

        if not self.editIDbutton.findChild(QAction).isChecked():
            self.editIDbutton.setChecked(False)

        posData.disableAutoActivateViewerWindow = True

        # Perform desired action on future frames
        posData.doNotShowAgain_EditID = doNotShowAgain
        posData.UndoFutFrames_EditID = UndoFutFrames
        posData.applyFutFrames_EditID = applyFutFrames
        includeUnvisited = (
            posData.includeUnvisitedInfo["Edit ID"] or doPropagateUnvisited
        )

        if not applyFutFrames and not doPropagateUnvisited:
            return

        self.changeIDfutureFrames(
            endFrame_i, oldIDnewIDMapper, includeUnvisited, shift=shift
        )

    def apply_manual_edits_to_lab_if_needed(self, lab):
        posData = self.data[self.pos_i]
        data_frame_i = posData.allData_li[posData.frame_i]
        edited_lab_dict = data_frame_i["manually_edited_lab"]["lab"]
        if not edited_lab_dict:
            return lab

        # zoom_slice = data_frame_i['manually_edited_lab']['zoom_slice']
        for z, lab_edited in edited_lab_dict.items():
            if not self.isSegm3D:
                # lab[zoom_slice] = lab_edited
                lab = lab_edited
                break

            lab[z] = lab_edited

            # lab[z, zoom_slice[0], zoom_slice[1]] = zoom_lab

        return lab

    def assignNewIDfromClickedID(self, clickedID: int, event: QGraphicsSceneMouseEvent):
        posData = self.data[self.pos_i]
        x, y = event.pos().x(), event.pos().y()
        newID = self.setBrushID(return_val=True)
        mapper = [(clickedID, newID)]
        self.applyEditID(clickedID, posData.IDs.copy(), mapper, x, y)

    def changeIDfutureFrames(
        self, endFrame_i, oldIDnewIDMapper, includeUnvisited, shift=False
    ):
        posData = self.data[self.pos_i]
        self.current_frame_i = posData.frame_i

        # Store data for current frame
        self.store_data()
        if endFrame_i is None:
            self.app.restoreOverrideCursor()
            return

        segmSizeT = len(posData.segm_data)
        for i in range(posData.frame_i + 1, segmSizeT):
            lab = posData.allData_li[i]["labels"]
            if lab is None and not includeUnvisited:
                self.enqAutosave()
                break

            if lab is not None:
                # Visited frame
                posData.frame_i = i
                self.get_data(lin_tree_init=False)
                if shift and self.isSegm3D:
                    lab = self.get_2Dlab(posData.lab)
                else:
                    lab = posData.lab

                if self.onlyTracking:
                    self.tracking(enforce=True)
                elif not posData.IDs:
                    continue
                else:
                    maxID = max(posData.IDs, default=0) + 1
                    for old_ID, new_ID in oldIDnewIDMapper:
                        if new_ID in lab:
                            tempID = maxID + 1  # lab.max() + 1
                            lab[lab == old_ID] = tempID
                            lab[lab == new_ID] = old_ID
                            lab[lab == tempID] = new_ID
                            maxID += 1
                        else:
                            lab[lab == old_ID] = new_ID

                    if shift and self.isSegm3D:
                        self.set_2Dlab(lab)

                    self.update_rp(draw=False)
                self.store_data(autosave=i == endFrame_i)
            elif includeUnvisited:
                # Unvisited frame (includeUnvisited = True)
                lab = posData.segm_data[i]
                if shift and self.isSegm3D:
                    lab = self.get_2Dlab(lab)
                else:
                    lab = lab

                for old_ID, new_ID in oldIDnewIDMapper:
                    if new_ID in lab:
                        tempID = lab.max() + 1
                        lab[lab == old_ID] = tempID
                        lab[lab == new_ID] = old_ID
                        lab[lab == tempID] = new_ID
                    else:
                        lab[lab == old_ID] = new_ID

                if shift and self.isSegm3D:
                    posData.segm_data[i][self.z_lab()] = lab

        # Back to current frame
        posData.frame_i = self.current_frame_i
        self.get_data()
        self.app.restoreOverrideCursor()

    def delBorderObj(self, checked):
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        posData = self.data[self.pos_i]
        posData.lab = skimage.segmentation.clear_border(posData.lab, buffer_size=1)
        oldIDs = posData.IDs.copy()
        self.update_rp()
        removedIDs = [ID for ID in oldIDs if ID not in posData.IDs]
        if posData.cca_df is not None:
            posData.cca_df = posData.cca_df.drop(index=removedIDs)
        self.store_data()
        self.updateAllImages()

    def delNewObj(self, checked):
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        posData = self.data[self.pos_i]
        frame_i = posData.frame_i

        if frame_i == 0:
            return

        prev_IDs = posData.allData_li[frame_i - 1]["IDs"]
        curr_IDs = posData.IDs
        new_IDs = list(set(curr_IDs) - set(prev_IDs))

        lab = posData.lab
        del_mask = np.isin(lab, new_IDs)
        lab[del_mask] = 0
        posData.lab = lab

        self.update_rp()

        if posData.cca_df is not None:
            posData.cca_df = posData.cca_df.drop(index=new_IDs)
        self.store_data()
        self.updateAllImages()

    def deleteIDFromLab(self, lab, delID, frame_i=None, delMask=None, shift=False):
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i if frame_i is None else frame_i

        if shift and self.isSegm3D:
            lab3D = lab
            delMask3D = delMask
            lab = self.get_2Dlab(lab)
            if delMask is not None:
                delMask = self.get_2Dlab(delMask)
            rp = skimage.measure.regionprops(lab)
            IDs_idxs = {obj.label: idx for idx, obj in enumerate(rp)}
        else:
            if frame_i == posData.frame_i:
                rp = posData.rp
                IDs_idxs = posData.IDs_idxs
            else:
                rp = posData.allData_li[frame_i]["regionprops"]
                IDs_idxs = posData.allData_li[frame_i]["IDs_idxs"]

        if isinstance(delID, int):
            delID = [delID]

        is_any_id_present = False
        for _delID in delID:
            if _delID in IDs_idxs:
                is_any_id_present = True
                break

        if not is_any_id_present:
            return lab, delMask

        if delMask is None:
            delMask = np.zeros(lab.shape, dtype=bool)
        else:
            delMask[:] = False

        for _delID in delID:
            idx = IDs_idxs.get(_delID, None)
            if idx is None:
                continue
            obj = rp[idx]
            delMask[obj.slice][obj.image] = True
        lab[delMask] = 0

        if shift and self.isSegm3D:
            self.set_2Dlab(lab, lab3D=lab3D)
            lab = lab3D
            if delMask3D is not None:
                self.set_2Dlab(delMask, lab3D=delMask3D)
                delMask = delMask3D

        return lab, delMask

    @disableWindow
    def deleteIDmiddleClick(
        self, delIDs: Iterable, applyFutFrames, includeUnvisited, shift=False
    ):
        self.clearHighlightedID()

        posData = self.data[self.pos_i]
        current_frame_i = posData.frame_i

        # Apply Delete ID to future frames if requested
        if applyFutFrames:
            delMask = np.zeros(posData.lab.shape, dtype=bool)
            # Store current data before going to future frames
            self.store_data()
            segmSizeT = len(posData.segm_data)
            for i in range(posData.frame_i + 1, segmSizeT):
                lab = posData.allData_li[i]["labels"]
                if lab is None and not includeUnvisited:
                    self.enqAutosave()
                    break

                if lab is not None:
                    # Visited frame
                    lab, _ = self.deleteIDFromLab(
                        lab, delIDs, frame_i=i, delMask=delMask, shift=shift
                    )

                    # Store change
                    posData.allData_li[i]["labels"] = lab
                    # Get the rest of the stored metadata based on the new lab
                    posData.frame_i = i
                    self.get_data()
                    self.store_data(autosave=False)
                elif includeUnvisited:
                    # Unvisited frame (includeUnvisited = True)
                    lab = posData.segm_data[i]
                    lab, _ = self.deleteIDFromLab(
                        lab, delIDs, frame_i=i, delMask=delMask, shift=shift
                    )

        # Back to current frame
        if applyFutFrames:
            posData.frame_i = current_frame_i
            self.get_data()

        z_slice = None
        if shift and self.isSegm3D:
            z_slice = self.z_lab()

        posData.lab, delID_mask = self.deleteIDFromLab(posData.lab, delIDs, shift=shift)
        for _delID in delIDs:
            self.clearObjContour(ID=_delID, ax=0)
            self.clearObjContour(ID=_delID, ax=1)
            if z_slice is None:
                self.removeObjectFromRp(_delID)
            self.removeStoredContours(_delID, z_slice=z_slice)

        if shift and self.isSegm3D:
            self.update_rp()

        self.store_data(autosave=False)
        self.whitelistPropagateIDs(
            IDs_to_remove=delIDs, curr_frame_only=(not applyFutFrames)
        )
        return delID_mask

    def getClickedID(self, xdata, ydata, text=""):
        posData = self.data[self.pos_i]
        ID = self.get_2Dlab(posData.lab)[ydata, xdata]
        if ID == 0:
            msg = f"You clicked on the background.\nEnter here the ID {text}"
            nearest_ID = core.nearest_nonzero_2D(
                self.get_2Dlab(posData.lab), xdata, ydata
            )
            clickedBkgrID = apps.QLineEditDialog(
                title="Clicked on background",
                msg=msg,
                parent=self,
                allowedValues=posData.IDs,
                defaultTxt=str(nearest_ID),
                isInteger=True,
            )
            clickedBkgrID.exec_()
            if clickedBkgrID.cancel:
                return
            else:
                ID = clickedBkgrID.EntryID
        return ID

    def getHoverID(self, xdata, ydata, byPassShiftCheck=False):
        if not hasattr(self, "diskMask"):
            return 0

        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        if byPassShiftCheck:
            shift = False
        else:
            shift = modifiers == Qt.ShiftModifier

        if self.isPowerBrush() and not ctrl:
            return 0

        if not self.autoIDcheckbox.isChecked():
            return self.editIDspinbox.value()

        ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)
        posData = self.data[self.pos_i]
        lab_2D = self.get_2Dlab(posData.lab)
        ID = lab_2D[ydata, xdata]
        self.isHoverZneighID = False
        if self.isSegm3D:
            z = self.z_lab()
            SizeZ = posData.lab.shape[0]
            doNotLinkThroughZ = self.brushButton.isChecked() and shift
            if doNotLinkThroughZ:
                if self.brushHoverCenterModeAction.isChecked() or ID > 0:
                    hoverID = ID
                else:
                    masked_lab = lab_2D[ymin:ymax, xmin:xmax][diskMask]
                    hoverID = np.bincount(masked_lab).argmax()
            else:
                if z > 0:
                    ID_z_under = posData.lab[z - 1, ydata, xdata]
                    if self.brushHoverCenterModeAction.isChecked() or ID_z_under > 0:
                        hoverIDa = ID_z_under
                    else:
                        lab = posData.lab
                        masked_lab_a = lab[z - 1, ymin:ymax, xmin:xmax][diskMask]
                        hoverIDa = np.bincount(masked_lab_a).argmax()
                else:
                    hoverIDa = 0

                if self.brushHoverCenterModeAction.isChecked() or ID > 0:
                    hoverIDb = lab_2D[ydata, xdata]
                else:
                    masked_lab_b = lab_2D[ymin:ymax, xmin:xmax][diskMask]
                    hoverIDb = np.bincount(masked_lab_b).argmax()

                if z < SizeZ - 1:
                    ID_z_above = posData.lab[z + 1, ydata, xdata]
                    if self.brushHoverCenterModeAction.isChecked() or ID_z_above > 0:
                        hoverIDc = ID_z_above
                    else:
                        lab = posData.lab
                        masked_lab_c = lab[z + 1, ymin:ymax, xmin:xmax][diskMask]
                        hoverIDc = np.bincount(masked_lab_c).argmax()
                else:
                    hoverIDc = 0

                if hoverIDa > 0:
                    hoverID = hoverIDa
                    self.isHoverZneighID = True
                elif hoverIDb > 0:
                    hoverID = hoverIDb
                elif hoverIDc > 0:
                    hoverID = hoverIDc
                    self.isHoverZneighID = True
                else:
                    hoverID = 0
        else:
            if self.brushButton.isChecked() and shift:
                # Force new ID with brush and Shift
                hoverID = 0
            elif self.brushHoverCenterModeAction.isChecked() or ID > 0:
                hoverID = ID
            else:
                masked_lab = lab_2D[ymin:ymax, xmin:xmax][diskMask]
                hoverID = np.bincount(masked_lab).argmax()

        self.editIDspinbox.setValue(hoverID)

        return hoverID

    def getLastHoveredID(self):
        if self.xHoverImg is None:
            return 0

        xdata, ydata = int(self.xHoverImg), int(self.yHoverImg)
        ID = self.currentLab2D[ydata, xdata]
        return ID

    def get_zslices_rp(self):
        if not self.isSegm3D:
            return

        posData = self.data[self.pos_i]
        self.store_zslices_rp()
        posData.zSlicesRp = posData.allData_li[posData.frame_i]["z_slices_rp"]

    def isPowerBrush(self):
        color = self.brushButton.palette().button().color().name()
        return color == self.doublePressKeyButtonColor

    def isPowerButton(self, button):
        color = button.palette().button().color().name()
        return color == self.doublePressKeyButtonColor

    def isPowerEraser(self):
        color = self.eraserButton.palette().button().color().name()
        return color == self.doublePressKeyButtonColor

    def is_power_button_color(
        self,
        *,
        button_color: str,
        power_color: str,
    ) -> bool:
        return button_color == power_color

    def mergeObjs_cb(self, checked):
        if not checked:
            self.mergeObjsTempLine.setData([], [])

    def removeObjectFromRp(self, delID):
        posData = self.data[self.pos_i]
        rp = []
        IDs = []
        IDs_idxs = {}
        idx = 0
        for obj in posData.rp:
            if obj.label == delID:
                continue
            rp.append(obj)
            IDs.append(obj.label)
            IDs_idxs[obj.label] = idx
            idx += 1

        posData.rp = rp
        posData.IDs = IDs
        posData.IDs_idxs = IDs_idxs

        if not self.isSegm3D:
            return

        zSlicesRp = {}
        for z, zSliceRp in posData.zSlicesRp.items():
            if delID in zSliceRp:
                continue

            zSlicesRp[z] = zSlicesRp

        posData.zSlicesRp = zSlicesRp
        self.store_zslices_rp(force_update=True)

    def removeStoredContours(self, delID, frame_i=None, z_slice=None):
        posData = self.data[self.pos_i]

        if frame_i is None:
            frame_i = posData.frame_i

        dataDict = posData.allData_li[posData.frame_i]
        try:
            newContours = {}
            for key, contours in dataDict["contours"].items():
                ID = key[0]
                if ID == delID:
                    continue

                if z_slice is not None:
                    z_slice_i = key[1]
                    if z_slice_i != z_slice:
                        continue

                newContours[key] = contours

            dataDict["contours"] = newContours
        except KeyError:
            pass

    def setHoverToolSymbolColor(
        self,
        xdata,
        ydata,
        pen,
        ScatterItems,
        button,
        brush=None,
        hoverRGB=None,
        ID=None,
        byPassShiftCheck=False,
    ):
        modifiers = QGuiApplication.keyboardModifiers()
        if byPassShiftCheck:
            shift = False
        else:
            shift = modifiers == Qt.ShiftModifier

        posData = self.data[self.pos_i]
        Y, X = self.get_2Dlab(posData.lab).shape
        if not myutils.is_in_bounds(xdata, ydata, X, Y):
            return

        self.isHoverZneighID = False
        if ID is None:
            hoverID = self.getHoverID(xdata, ydata, byPassShiftCheck=byPassShiftCheck)
        else:
            hoverID = ID

        if hoverID == 0:
            for item in ScatterItems:
                item.setPen(pen)
                item.setBrush(brush)
        else:
            try:
                rgb = self.lut[hoverID]
                rgb = rgb if hoverRGB is None else hoverRGB
                rgbPen = np.clip(rgb * 1.1, 0, 255)
                for item in ScatterItems:
                    item.setPen(*rgbPen, width=2)
                    item.setBrush(*rgb, 100)
            except IndexError:
                pass

        checkChangeID = (
            self.isHoverZneighID and not shift and self.lastHoverID != hoverID
        )
        if checkChangeID:
            # We are hovering an ID in z+1 or z-1
            self.restoreBrushID = hoverID
            # self.changeBrushID()

        self.lastHoverID = hoverID

    def should_apply_manual_edits(self, edited_labels_by_z) -> bool:
        return bool(edited_labels_by_z)

    def should_force_new_hover_id(
        self,
        *,
        brush_active: bool,
        shift_pressed: bool,
    ) -> bool:
        return brush_active and shift_pressed

    def should_prompt_for_background_id(self, clicked_id: int) -> bool:
        return clicked_id == 0

    def should_restore_brush_id_from_hover(
        self,
        *,
        is_hover_z_neighbor: bool,
        shift_pressed: bool,
        last_hover_id: int,
        hover_id: int,
    ) -> bool:
        return is_hover_z_neighbor and not shift_pressed and last_hover_id != hover_id

    LEGACY_METHODS = (
        "mergeObjs_cb",
        "assignNewIDfromClickedID",
        "addYXcentroidToDf",
        "_get_editID_info",
        "apply_manual_edits_to_lab_if_needed",
        "store_zslices_rp",
        "removeObjectFromRp",
        "get_zslices_rp",
        "_update_zslices_rp",
        "update_rp",
        "delBorderObj",
        "delNewObj",
        "getClickedID",
        "deleteIDFromLab",
        "removeStoredContours",
        "deleteIDmiddleClick",
        "applyEditID",
        "changeIDfutureFrames",
        "getLastHoveredID",
        "getHoverID",
        "setHoverToolSymbolColor",
        "isPowerBrush",
        "isPowerEraser",
        "isPowerButton",
    )

    def should_store_zslice_regionprops(self, *, is_segm_3d: bool) -> bool:
        return is_segm_3d

    def should_update_zslice_regionprops(
        self,
        *,
        force_update: bool,
        already_stored: bool,
    ) -> bool:
        return force_update or not already_stored

    def store_zslices_rp(self, force_update=False):
        if not self.isSegm3D:
            return

        posData = self.data[self.pos_i]
        are_zslices_rp_stored = (
            posData.allData_li[posData.frame_i].get("z_slices_rp") is not None
        )
        if force_update or not are_zslices_rp_stored:
            self._update_zslices_rp()

        posData.allData_li[posData.frame_i]["z_slices_rp"] = posData.zSlicesRp

    @exception_handler
    def update_rp(
        self,
        draw=True,
        debug=False,
        update_IDs=True,
        wl_update=True,
        wl_track_og_curr=False,
        wl_update_lab=False,
    ):

        posData = self.data[self.pos_i]
        # Update rp for current posData.lab (e.g. after any change)

        if wl_update:
            if self.whitelistOriginalIDs is None:
                old_IDs = posData.allData_li[posData.frame_i][
                    "IDs"
                ].copy()  # for whitelist stuff
            else:
                old_IDs = self.whitelistOriginalIDs.copy()
                self.whitelistOriginalIDs = None
        elif self.whitelistOriginalIDs is None:
            self.whitelist_old_IDs = posData.allData_li[posData.frame_i]["IDs"].copy()

        posData.rp = skimage.measure.regionprops(posData.lab)
        if update_IDs:
            IDs = []
            IDs_idxs = {}
            for idx, obj in enumerate(posData.rp):
                IDs.append(obj.label)
                IDs_idxs[obj.label] = idx
            posData.IDs = IDs
            posData.IDs_idxs = IDs_idxs
        self.update_rp_metadata(draw=draw)
        self.store_zslices_rp(force_update=True)

        if not wl_update:
            return

        # Update tracking whitelist
        accepted_lost_centroids = self.getTrackedLostIDs()
        new_IDs = posData.IDs
        added_IDs = set(new_IDs) - set(old_IDs)
        removed_IDs = set(old_IDs) - set(new_IDs) - set(accepted_lost_centroids)

        self.whitelistPropagateIDs(
            IDs_to_add=added_IDs,
            IDs_to_remove=removed_IDs,
            curr_frame_only=True,
            IDs_curr=new_IDs,
            track_og_curr=wl_track_og_curr,
            curr_lab=posData.lab,
            curr_rp=posData.rp,
            update_lab=wl_update_lab,
        )
