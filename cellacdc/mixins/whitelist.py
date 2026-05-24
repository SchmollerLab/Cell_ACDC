"""Qt view adapter for the Whitelist feature."""

from __future__ import annotations

import os
import numpy as np
from typing import Set, List
import skimage.measure
from typing import Tuple
import time

from cellacdc import printl, html_utils, apps, widgets, exception_handler, disableWindow
from cellacdc.trackers.CellACDC import CellACDC_tracker
from cellacdc.whitelist import Whitelist


class WhitelistMixin:
    """Qt-facing adapter for the Whitelist feature."""

    """Headless decisions and calculations for Whitelist management."""

    def apply_id_mask(
        self,
        curr_lab: np.ndarray,
        og_lab: np.ndarray | None,
        missing_ids: list[int] | np.ndarray,
        to_be_removed_ids: list[int] | np.ndarray,
    ) -> np.ndarray:
        """Applies missing and removed ID masks to the label array."""
        updated_lab = curr_lab.copy().astype(np.int32)
        missing_ids = np.array(missing_ids, dtype=np.int32)
        to_be_removed_ids = np.array(to_be_removed_ids, dtype=np.int32)

        if missing_ids.size > 0 and og_lab is not None:
            mask = np.isin(og_lab, missing_ids)
            updated_lab[mask] = og_lab[mask]

        if to_be_removed_ids.size > 0:
            updated_lab[np.isin(updated_lab, to_be_removed_ids)] = 0

        return updated_lab

    def check_original_labels(self, whitelist_obj, frame_i: int) -> bool:
        """Checks if original label data is allocated and valid for the frame."""
        if whitelist_obj is None:
            return False
        if whitelist_obj.originalLabsIDs is None:
            return False
        if (
            frame_i >= len(whitelist_obj.originalLabsIDs)
            or whitelist_obj.originalLabsIDs[frame_i] is None
        ):
            return False
        return True

    def construct_og_frame(
        self,
        pos_lab: np.ndarray,
        og_frame_base: np.ndarray,
        whitelist_ids: Set[int],
        og_ids: Set[int],
    ) -> np.ndarray:
        """Constructs original labels overlay using np.isin masking."""
        og_frame = og_frame_base.copy()

        ids_to_update = whitelist_ids & og_ids
        if ids_to_update:
            mask = np.isin(og_frame, list(ids_to_update))
            og_frame[mask] = 0
            mask = np.isin(pos_lab, list(ids_to_update))
            og_frame[mask] = pos_lab[mask]

        ids_to_add = whitelist_ids - og_ids
        if ids_to_add:
            mask = np.isin(pos_lab, list(ids_to_add))
            og_frame[mask] = pos_lab[mask]

        return og_frame

    def filter_existing_ids(
        self, current_whitelist: Set[int], possible_ids: Set[int]
    ) -> tuple[Set[int], bool]:
        """Filters out non-existing IDs from the current whitelist.

        Returns a tuple: (filtered_whitelist, is_any_id_non_existing)
        """
        is_any_id_non_existing = False
        filtered_whitelist = set(current_whitelist)
        for ID in current_whitelist:
            if ID not in possible_ids:
                is_any_id_non_existing = True
                filtered_whitelist.discard(ID)
        return filtered_whitelist, is_any_id_non_existing

    def get_diff_ids(
        self, old_ids: Set[int], prev_ids: Set[int], new_ids: Set[int]
    ) -> Set[int]:
        """Computes tracking difference intersection (new_ids - old_ids) & prev_ids."""
        return (new_ids - old_ids) & prev_ids

    def get_frames_range(self, frame_i: int) -> list[int]:
        """Calculates navigation frame ranges for label loading."""
        if frame_i > 0:
            return [frame_i - 1, frame_i]
        return [frame_i]

    def get_missing_ids(
        self, current_ids: Set[int], previous_ids: Set[int]
    ) -> Set[int]:
        """Returns the set of IDs present in current frame but missing from previous frame."""
        return set(current_ids) - set(previous_ids)

    def get_whitelist_missing_and_removed_ids(
        self, whitelist: Set[int], current_ids: Set[int]
    ) -> tuple[list[int], list[int]]:
        """Finds IDs that are missing from current_ids and IDs to be removed from current_ids."""
        missing_ids = list(whitelist - current_ids)
        to_be_removed_ids = list(current_ids - whitelist)
        return missing_ids, to_be_removed_ids

    def whitelistAddNewIDs(self, ignore_not_first_time: bool = False):
        """Function which adds new IDs to the whitelist, based on the original labels.
        It will check if the frame is visited the first time, unless
        ignore_not_first_time is True.
        It does nothing if self.addNewIDsWhitelistToggle is False.
        !!!Careful, does not change the lab, just the whitelist!!!

        Parameters
        ----------
        ignore_not_first_time : bool, optional
            Weather it should be checked if the frame is visited
            the first time, by default False
        """
        mode = self.modeComboBox.currentText()
        if mode != "Segmentation and Tracking":
            return

        if not self.addNewIDsWhitelistToggle:
            return

        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return

        debug = posData.whitelist._debug

        if debug:
            printl("whitelistAddNewIDs")

        posData = self.data[self.pos_i]
        frame_i = posData.frame_i

        if self.get_last_tracked_i() > frame_i and not ignore_not_first_time:
            return

        if frame_i == 0:
            return

        if (
            self.whitelistAddNewIDsFrame is not None
            and frame_i == self.whitelistAddNewIDsFrame
        ):
            return

        self.whitelistAddNewIDsFrame = frame_i

        curr_lab = self.get_curr_lab()

        posData.whitelist.addNewIDs(
            frame_i=frame_i,
            allData_li=posData.allData_li,
            IDs_curr=posData.IDs,
            curr_lab=curr_lab,
        )

    def whitelistAddNewIDsToggled(self, checked: bool):
        """Will set self.addNewIDsWhitelistToggle to checked and call
        whitelistAddNewIDs if checked is True.

        Parameters
        ----------
        checked : bool
            True if the add new IDs toggle is checked, False otherwise.
        """
        self.addNewIDsWhitelistToggle = checked
        if checked:
            self.df_settings.at["addNewIDsWhitelistToggle", "value"] = "Yes"
        else:
            self.df_settings.at["addNewIDsWhitelistToggle", "value"] = "No"
        self.df_settings.to_csv(self.settings_csv_path)
        if checked:
            self.whitelistAddNewIDs(ignore_not_first_time=True)
            self.whitelistPropagateIDs()
            self.updateAllImages()
            self.whitelistIDsUpdateText()

    def whitelistCheckOriginalLabels(self, warning: bool = True, frame_i: int = None):
        """Warns the user that there are no original labels labels are present
        for the frame"""
        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return False

        if frame_i is None:
            frame_i = posData.frame_i

        if posData.whitelist.originalLabsIDs is None:
            return False

        if (
            frame_i >= len(posData.whitelist.originalLabsIDs)
            or posData.whitelist.originalLabsIDs[frame_i] is None
        ):
            txt = """
            No original labels are present for the current frame,
            this action cannot be performed."""
            self.logger.warning(txt)
            if not warning:
                return False
            widgets.myMessageBox.warning(
                self,
                "No original labels",
                txt,
            )

            return False
        else:
            return True

    def whitelistHighlightIDs(self, checked: bool = True):
        """Highlights the IDs in the current frame based on the whitelist.

        Parameters
        ----------
        checked : bool, optional
            If False, will delete all highlights, by default True
        """
        if not checked:
            self.removeHighlightLabelID()
            return

        posData = self.data[self.pos_i]

        if posData.whitelist is None:
            if not hasattr(self, "tempWhitelistIDs"):
                self.tempWhitelistIDs = set()  # not updated, only use in this context
                current_whitelist = self.tempWhitelistIDs
            else:
                current_whitelist = self.tempWhitelistIDs
        else:
            current_whitelist = posData.whitelist.get(frame_i=posData.frame_i)

        for ID in current_whitelist:
            self.highlightLabelID(ID)

    def whitelistIDsAccepted(self, whitelistIDs: Set[int] | List[int]):
        """Function which is called when the user accepts a whitelist.
        Also initializes the whitelist if it is not already initialized. (Aka not loaded)

        Parameters
        ----------
        whitelistIDs : set | list
            The accepted IDs from the whitelist dialog.
        """
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        self.whitelistIDsToolbar.viewOGToggle.setCheckable(True)
        self.whitelistSetViewOGIDsToggle(False)
        self.setFrameNavigationDisabled(False, why="Viewing original labels")

        self.store_data(autosave=False)

        posData = self.data[self.pos_i]

        if not posData.whitelist:
            posData.whitelist = Whitelist(
                total_frames=posData.SizeT,
            )

        if posData.whitelist._debug:
            printl("whitelistIDsAccepted", whitelistIDs)

        whitelistIDs = set(whitelistIDs)

        IDs_curr = set(posData.IDs)

        posData.whitelist.IDsAccepted(
            whitelistIDs,
            segm_data=posData.segm_data,
            frame_i=posData.frame_i,
            allData_li=posData.allData_li,
            IDs_curr=IDs_curr,
            curr_lab=posData.lab,
        )

        # self.whitelistPropagateIDs(new_whitelist=whitelistIDs,
        #                            try_create_new_whitelists=True,
        #                            only_future_frames=True,
        #                            force_not_dynamic_update=True,
        #                            update_lab=True
        #                            )
        self.whitelistUpdateLab(track_og_curr=True)

        self.whitelistIDsUpdateText()
        self.keepIDsTempLayerLeft.clear()

    def whitelistIDsChanged(
        self, whitelistIDs: Set[int] | List[int], debug: bool = False
    ):
        """Callback for when the whitelist IDs are changed.
        This is called when the user changed the IDs in the whitelist IDs toolbar
        (or when its programmatically changed, but if its not
        visible it should return instantly)
        Will update the temp layer and also complain when IDs
        are not valid/present in the current lab

        Parameters
        ----------
        whitelistIDs : set | list
            The IDs that are currently in the whitelist.
        debug : bool, optional
            debug, by default False
        """
        if not self.whitelistIDsButton.isChecked():
            return

        posData = self.data[self.pos_i]

        if posData.whitelist:
            debug = posData.whitelist._debug
        if debug:
            printl("whitelistIDsChanged", whitelistIDs)

        if posData.whitelist is None:
            wl_init = False
            if not hasattr(self, "tempWhitelistIDs"):
                self.tempWhitelistIDs = set()  # not updated, only use in this context
                current_whitelist = self.tempWhitelistIDs
            else:
                current_whitelist = self.tempWhitelistIDs
        else:
            wl_init = True
            current_whitelist = posData.whitelist.get(frame_i=posData.frame_i)

        current_whitelist_copy = current_whitelist.copy()
        if (
            not hasattr(posData, "originalLabsIDs")
            or posData.whitelist.originalLabsIDs is None
        ):
            possible_IDs = posData.IDs.copy()
        else:
            if not self.whitelistCheckOriginalLabels(warning=False):
                possible_IDs = set(posData.IDs)
            else:
                possible_IDs = posData.whitelist.originalLabsIDs[posData.frame_i]
                possible_IDs.update(posData.IDs)

        isAnyIDnotExisting = False
        for ID in whitelistIDs:
            if ID not in possible_IDs:
                isAnyIDnotExisting = True
                continue
            if ID not in current_whitelist_copy:
                current_whitelist.add(ID)
                self.highlightLabelID(ID)

        for ID in current_whitelist_copy:
            if ID not in possible_IDs:
                isAnyIDnotExisting = True
                continue
            if ID not in whitelistIDs:
                current_whitelist.remove(ID)
                self.removeHighlightLabelID(IDs=[ID])

        if wl_init:
            posData.whitelist.whitelistIDs[posData.frame_i] = current_whitelist
        else:
            self.tempWhitelistIDs = current_whitelist

        self.whitelistUpdateTempLayer()
        if isAnyIDnotExisting:
            self.whitelistIDsToolbar.whitelistLineEdit.warnNotExistingID()
        else:
            self.whitelistIDsToolbar.whitelistLineEdit.setInstructionsText()

    def whitelistIDsUpdateText(self):
        """Updates the text. Carefull, triggers whitelistLineEdit.textChanged!"""
        mode = self.modeComboBox.currentText()
        if mode != "Segmentation and Tracking":
            return

        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return

        if posData.whitelist._debug:
            printl("whitelistIDsUpdateText")

        frame_i = posData.frame_i
        whitelist = posData.whitelist.get(frame_i=frame_i)

        self.whitelistIDsToolbar.whitelistLineEdit.setText(whitelist)

    def whitelistIDs_cb(self, checked: bool):
        """Callback for when the whitelist IDs button is checked or unchecked.
        Initialises the pointlayer and the whitelist IDs toolbar if checked.

        Parameters
        ----------
        checked : bool
            True if the whitelist IDs button is checked, False otherwise.
        """
        if checked:
            self.initKeepObjLabelsLayers()
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.whitelistIDsButton)
            self.connectLeftClickButtons()

        self.whitelistIDsToolbar.setVisible(checked)
        self.whitelistHighlightIDs(checked)
        self.whitelistIDsUpdateText()
        self.whitelistUpdateTempLayer()

        if not checked:
            self.setLostNewOldPrevIDs()
            self.updateAllImages()

    def whitelistInitNewFrames(self, frame_i: int = None, force: bool = False):
        """Initialize the whitelist for a new frame. The class whitelist keeps track
        of the init frames and doesnt try to init them again, unless forced.
        Does not init the class!

        Parameters
        ----------
        frame_i : int, optional
            frame_i to be init, posData.frame_i if not provided, by default None
        force : bool, optional
            if the init should be forced, by default False

        Returns
        -------
        bool
            if the frame was new or not
        list
            list of frames that were updated, and info about added/removed IDs
        """

        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return False, []

        if frame_i is None:
            frame_i = posData.frame_i

        if posData.whitelist._debug:
            printl("whitelistInitNewFrames", frame_i, force)

        if frame_i not in posData.whitelist.initialized_i:
            self.whitelistTrackOGCurr(frame_i=frame_i, against_prev=True)

        new_frame, update_frames = posData.whitelist.initNewFrames(
            frame_i=frame_i, force=force
        )

        self.whitelistAddNewIDs()
        return new_frame, update_frames

    @disableWindow
    def whitelistLoadOGLabs(self, selected: str):
        """Loads the original labels from the selected files

        Parameters
        ----------
        selected : str
            Selected file name from the dialog.
        """
        posData = self.data[self.pos_i]
        images_path = posData.images_path

        selected_path = os.path.join(images_path, selected)
        posData.whitelist.loadOGLabs(selected_path)

        self.whitelistIDsToolbar.viewOGToggle.setCheckable(True)

    def whitelistLoadOGLabs_cb(self):
        """Generates a dialog to load the original (not whitelisted) labels"""
        posData = self.data[self.pos_i]
        curr_seg_path = posData.segm_npz_path

        segmFilename = os.path.basename(curr_seg_path)
        custom_first = f"{segmFilename[:-4]}_not_whitelisted.npz"
        images_path = posData.images_path
        existingEndnames = [
            files for files in os.listdir(images_path) if files.endswith(".npz")
        ]
        if custom_first not in existingEndnames:
            custom_first = None

        infoText = html_utils.paragraph(
            "Select the segmentation file containing the original labels "
            'of the objects. Pleae note that the current saved "original" '
            "labels will be replaced with the new ones, but the filtered "
            "labels will be kept."
        )

        win = apps.SelectSegmFileDialog(
            existingEndnames,
            images_path,
            parent=self,
            basename=posData.basename,
            infoText=infoText,
            custom_first=custom_first,
        )
        win.exec_()
        if win.cancel:
            self.logger.info("Loading original labels canceled.")
            return
        selected = win.selectedItemText
        self.logger.info(f"Loading original labels from {selected}...")
        self.whitelistLoadOGLabs(selected)

    def whitelistPropagateIDs(
        self,
        new_whitelist: Set[int] | List[int] = None,
        IDs_to_add: Set[int] = None,
        IDs_to_remove: Set[int] = None,
        frame_i: int = None,
        try_create_new_whitelists: bool = False,
        curr_frame_only: bool = False,
        force_not_dynamic_update: bool = False,
        only_future_frames: bool = True,
        allow_only_current_IDs: bool = False,
        track_og_curr: bool = True,
        IDs_curr: Set[int] | List[int] = None,
        index_lab_combo: Tuple[int, np.ndarray] = None,
        curr_rp: list = None,
        curr_lab: np.ndarray = None,
        store_data: bool = True,
        update_lab: bool = False,
    ):
        """
        Propagates whitelist IDs across frames in the dataset. (Doesnt update labs)
        Should also be called when viewing a new frame!

        This function updates whitelist. If curr_frame_only is True, it only updates the
        whitelist of the current frame. If the frame changes, this function should be called
        again to update the whitelist for the new frame (without this argument).
        It should also handle cases were this is not done, but this is less safe.
        Then, all the additions and removals are propagated to the other frames.
        If force_not_dynamic_update is True, the function will propagate the entire whitelist to
        frames, and not only the IDs which were added or removed.

        Hierarchy of arguments for current_IDs:
        1. IDs_curr (if provided)
        (2. index_lab_combo (if provided) (is also passed to not current frame only
        propagation if that propagation is necessary, and used when the frame_i matches))
        3. curr_rp (if provided)
        4. curr_lab (if provided)
        5. allData_li

        Parameters
        ----------
        new_whitelist : Set[int] | List[int], optional
            A new set of whitelist IDs to replace the current whitelist. Cannot be
            used together with `IDs_to_add` or `IDs_to_remove`, by default None.
        IDs_to_add : Set[int], optional
            A set of IDs to add to the current whitelist, by default None.
        IDs_to_remove : Set[int], optional
            A set of IDs to remove from the current whitelist, by default None.
        frame_i : int, optional
            The frame index for the propagation.
            If None, uses posData.frame_i, by default None.
        try_create_new_whitelists : bool, optional
            If True, creates new whitelist entries for frames that do not already
            have them. Should only be necessary when its initialized, by default False.
        curr_frame_only : bool, optional
            If True, only updates the whitelist for the current frame.
            (See description of function), by default False.
        force_not_dynamic_update : bool, optional
            If True, disables dynamic updates to the whitelist.
            (See description of function), by default False.
        only_future_frames : bool, optional
            If True, propagates changes only to future frames, by default True.
        allow_only_current_IDs : bool, optional
            If True, only allows IDs that are present in the current frame
            to be added to the whitelist, by default True.
        track_og_curr : bool, optional
            If True, tracks the original labels in relation to the current
            (whitelisted) labels. This is done by calling whitelistTrackOGCurr.
            If its a new frame, this is done in whitelistInitNewFrames against the
            previous frame,
            by default True.
        IDs_curr : Set[int] | List[int], optional
            A set of IDs for the current frame, if None,
            will be calculated from other stuff (see description), by default None.
        index_lab_combo : Tuple[int, np.ndarray], optional
            Combination of frame_i and current frame,
            Used to get IDs_curr (see description), when the frame_i matches
            (is also passed to not current frame only
            propagation if that propagation is necessary,
            and used when the frame_i matches), by default None.
        curr_rp : list, optional
            Region properties for the current frame. For IDs_curr. (see description),
            by default None.
        curr_lab : np.ndarray, optional
            Labels for the current frame for IDs_curr. (see description),
            by default None.
        store_data : bool, optional
            If True, stores the data before propagating the IDs.
        update_lab : bool, optional
            If True, updates the labels after propagating the IDs.
            Will always update labels for newly init frames, by default False.

        Raises
        ------
        ValueError
            If both `new_whitelistIDs` and `IDs_to_add`/`IDs_to_remove` are provided.

        Example
        -------
        To add IDs 5 and 6 to the whitelist for the current frame:
        ```python
        self.whitelistPropagateIDs(IDs_to_add={5, 6}, curr_frame_only=True)
        ```
        Then when the frame changes:
        ```python
        self.whitelistPropagateIDs()
        ```

        To replace the whitelist for frame 10 with a new set of IDs:
        ```python
        self.whitelistPropagateIDs(new_whitelistIDs={1, 2, 3}, frame_i=10)
        ```
        This would also propagate the changes to all other frames.

        """
        # doesnt update the frame displayed, only wl
        try:  # safety XD
            IDs_curr = IDs_curr.copy()
        except AttributeError:
            pass

        IDs_curr = set(IDs_curr) if IDs_curr is not None else None

        posData = self.data[self.pos_i]

        debug = posData.whitelist._debug if posData.whitelist is not None else False

        if debug:
            printl("Propagating IDs...")
            from . import debugutils

            debugutils.print_call_stack()
            printl(new_whitelist, IDs_to_add, IDs_to_remove)

        if posData.whitelist is None:
            return

        # og_frame_i = posData.frame_i
        if frame_i is None:
            frame_i = posData.frame_i

        new_frame, update_frames_init = self.whitelistInitNewFrames(frame_i=frame_i)

        if new_frame:
            self.update_rp(wl_update=False)
        # if track_og_curr and not new_frame:
        #     self.whitelistTrackOGCurr(frame_i=frame_i, rp=curr_rp, lab=curr_lab)

        update_frames = posData.whitelist.propagateIDs(
            frame_i,
            posData.allData_li,
            new_whitelist=new_whitelist,
            IDs_to_add=IDs_to_add,
            IDs_to_remove=IDs_to_remove,
            try_create_new_whitelists=try_create_new_whitelists,
            curr_frame_only=curr_frame_only,
            force_not_dynamic_update=force_not_dynamic_update,
            only_future_frames=only_future_frames,
            allow_only_current_IDs=allow_only_current_IDs,
            IDs_curr=IDs_curr,
            index_lab_combo=index_lab_combo,
            curr_rp=curr_rp,
            curr_lab=curr_lab,
        )
        if update_lab:
            update_frames = update_frames_init + update_frames
        else:
            update_frames = update_frames_init
        # printl(posData.whitelistIDs[frame_i])
        # posData.frame_i = og_frame_i
        self.whitelistIDsUpdateText()
        if store_data:
            self.store_data(autosave=False)

        for frame_i, IDs_to_add, IDs_to_remove, new_frame in update_frames:
            self.whitelistUpdateLab(
                frame_i=frame_i,
                track_og_curr=track_og_curr,
                new_frame=new_frame,
                IDs_to_add=IDs_to_add,
                IDs_to_remove=IDs_to_remove,
            )

    def whitelistSetViewOGIDsToggle(self, checked: bool):
        """Set the view original labels toggle button to checked or unchecked.
        This also updates the self.viewOriginalLabels variable.
        !!! Doesn't change the actually displayed labels, use self.whitelistViewOGIDs
        to do that.!!!

        Parameters
        ----------
        checked : bool
            True if the original labels are shown, False otherwise.
        """
        self.viewOriginalLabels = checked
        self.whitelistIDsToolbar.viewOGToggle.blockSignals(True)
        self.whitelistIDsToolbar.viewOGToggle.setChecked(checked)
        self.whitelistIDsToolbar.viewOGToggle.blockSignals(False)

    def whitelistSyncIDsOG(
        self,
        frame_is: List[int] = None,
        against_prev: bool = False,
    ):
        """Interates over the frames and calls whitelistTrackOGCurr for each frame.

        Parameters
        ----------
        frame_is : List[int], optional
            list of frame_i, if None goes through all, by default None
        against_prev : bool, optional
            if the original frame should be tracked against frame_i-1.
        """
        posData = self.data[self.pos_i]
        if frame_is is None:
            frame_is = range(posData.SizeT)

        for frame_i in frame_is:
            self.whitelistTrackOGCurr(frame_i=frame_i, against_prev=against_prev)

    def whitelistTrackCurrOG(self, frame_i: int = None, against_prev: bool = False):
        """Track the current (whitelisted) labels in relation to the original labels.
        Parameters
        ----------
        frame_i : int, optional
            frame_i to be tracked, posData.frame_i if not provided, by default None
        against_prev : bool, optional
            if the original frame should be tracked against frame_i-1.
        """
        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return

        if posData.whitelist._debug:
            printl("whitelistTrackCurrOG", frame_i, against_prev)

        if frame_i is None:
            frame_i = posData.frame_i

        if against_prev and frame_i == 0:
            return

        og_frame = posData.frame_i
        if frame_i != og_frame:
            self.store_data(autosave=False)
            posData.frame_i = frame_i
            self.get_data()

        lab = posData.lab
        rp = posData.rp

        if not self.whitelistCheckOriginalLabels(
            warning=False, frame_i=frame_i if not against_prev else frame_i - 1
        ):
            if posData.whitelist._debug:
                printl("No original labels, cannot track.")
            return

        if against_prev:
            og_lab = posData.whitelist.originalLabs[frame_i - 1]
        else:
            og_lab = posData.whitelist.originalLabs[frame_i]

        og_rp = skimage.measure.regionprops(og_lab)

        denom_overlap_matrix = "union" if not against_prev else "area_prev"

        lab = CellACDC_tracker.track_frame(
            og_lab,
            og_rp,
            lab,
            rp,
            denom_overlap_matrix=denom_overlap_matrix,
            posData=posData,
            setBrushID_func=self.setBrushID,
        )

        posData.lab = lab

        self.update_rp(wl_update=False)
        self.store_data(autosave=False)

        if frame_i != og_frame:
            posData.frame_i = og_frame
            self.get_data()

    def whitelistTrackOGCurr(
        self,
        frame_i: int = None,
        against_prev: bool = False,
        lab: np.ndarray = None,
        rp: list = None,
        IDs: Set[int] | List[int] = None,
    ):
        """Track the original labels in relation to the current (whitelisted)
        labels.
        Parameters

        Parameters
        ----------
        frame_i : int, optional
            frame_i to be tracked, posData.frame_i if not provided,
            by default None
        against_prev : bool, optional
            if the original frame should be tracked against frame_i-1.
            Cannot be used with rp or lab, by default False
        lab : np.ndarray, optional
            lab to be tracked against, by default None
        rp : list, optional
            regionprops for this lab, by default None
        IDs : Set[int] | List[int], optional
            IDs that should be tracked based on og

        Raises
        ------
        ValueError
            Cannot provide both rp and lab when tracking against previous frame.
            Instead only provide rp and lab, and dont set against_prev.
        """
        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return

        debug = posData.whitelist._debug

        if debug:
            from . import debugutils

            debugutils.print_call_stack(depth=2)
            printl("whitelistTrackOGCurr", against_prev)

        if against_prev and (rp is not None or lab is not None):
            raise ValueError(
                "Cannot provide both rp and lab when tracking"
                " against previous frame."
                "Instead only provide rp and lab, and dont set against_prev."
            )

        if frame_i is None:
            frame_i = posData.frame_i

        if against_prev and frame_i == 0:
            return

        if not self.whitelistCheckOriginalLabels(warning=False, frame_i=frame_i):
            if debug:
                printl("No original labels, cannot track.")
            return

        og_frame_i = posData.frame_i
        ### against what should I track?

        if lab is not None and not rp:
            rp = skimage.measure.regionprops(lab)

        changed_frame = False
        if lab is None:
            if debug:
                printl("No lab and no rp provided.")
            if against_prev:
                rp = posData.allData_li[frame_i - 1]["regionprops"]
                lab = posData.allData_li[frame_i - 1]["labels"]
            else:
                if frame_i != og_frame_i:
                    self.store_data(autosave=False)
                    posData.frame_i = frame_i
                    self.get_data()
                    changed_frame = True
                rp = posData.rp
                lab = posData.lab
        og_lab = posData.whitelist.originalLabs[frame_i]
        og_rp = skimage.measure.regionprops(og_lab)
        # lab = lab.copy()

        denom_overlap_matrix = "union" if not against_prev else "area_prev"

        og_lab = CellACDC_tracker.track_frame(
            lab,
            rp,
            og_lab,
            og_rp,
            denom_overlap_matrix=denom_overlap_matrix,
            posData=posData,
            setBrushID_func=self.setBrushID,
            IDs=IDs,
            # assign_unique_new_IDs=False,
        )

        posData.whitelist.originalLabs[frame_i] = og_lab
        posData.whitelist.originalLabsIDs[frame_i] = {
            obj.label for obj in skimage.measure.regionprops(og_lab)
        }

        if changed_frame:
            posData.frame_i = og_frame_i
            self.get_data()

    @disableWindow
    def whitelistTrackOGagainstPreviousFrame_cb(self, signal_slot=None):
        """Tracks the original labels against the previous frame.
        This is used as a callback for sigTrackOGagainstPreviousFrame signal
        """
        posData = self.data[self.pos_i]
        frame_i = posData.frame_i
        if not self.whitelistCheckOriginalLabels():
            return
        old_cell_IDs = posData.whitelist.originalLabsIDs[frame_i]
        prev_cell_IDs = posData.allData_li[frame_i - 1]["IDs"]
        self.whitelistTrackOGCurr(against_prev=True)
        new_cell_IDs = posData.whitelist.originalLabsIDs[frame_i]

        new_IDs = new_cell_IDs - old_cell_IDs
        new_IDs = new_IDs & set(prev_cell_IDs)

        self.whitelistUpdateLab(
            track_og_curr=False,
            IDs_to_add=new_IDs,
        )

    def whitelistUpdateLab(
        self,
        frame_i: int = None,
        track_og_curr=False,
        new_frame: bool = False,
        IDs_to_add: List[int] | Set[int] = None,
        IDs_to_remove: List[int] | Set[int] = None,
    ):
        # this should also work for 3D i think...
        """Updates the displayed lab based on the whitelist.

        Parameters
        ----------
        frame_i : int, optional
            frame which should be updated. If not provided,
            uses posData.frame_i, by default None
        track_og_curr : bool, optional
            if True, will track the original current IDs, by default False
        new_frame : bool, optional
            if True, will set the frame to the new frame, by default False
        IDs_to_add : list, optional
            IDs to add to the whitelist, by default None
        IDs_to_remove : list, optional
            IDs to remove from the whitelist, by default None
        """
        got_data = False
        benchmark = False
        if benchmark:
            ts = [time.perf_counter()]
            titles = [
                "",
                "store_data",
                "whitelistSetViewOGIDsToggle",
                "get_data",
                "get what to add/remove",
                "track_og_curr",
                "get current lab",
                "add/remove IDs",
                "store data",
                "update images",
            ]

        mode = self.modeComboBox.currentText()
        if mode != "Segmentation and Tracking":
            return

        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return

        if frame_i is None:
            frame_i = posData.frame_i
            og_frame_i = frame_i
        else:
            og_frame_i = posData.frame_i
            posData.frame_i = frame_i
            # getting data is handles later in the code

        debug = posData.whitelist._debug
        if debug:
            printl("whitelistUpdateLab", frame_i, og_frame_i)
            from . import debugutils

            debugutils.print_call_stack()

        if benchmark:
            ts.append(time.perf_counter())

        self.whitelistSetViewOGIDsToggle(False)  ###

        if benchmark:
            ts.append(time.perf_counter())

        if self.whitelistCheckOriginalLabels(warning=False, frame_i=frame_i):
            og_lab = posData.whitelist.originalLabs[frame_i]  ###
        else:
            og_lab = None
        if benchmark:
            ts.append(time.perf_counter())

        ####
        whitelist = posData.whitelist.get(frame_i=frame_i)
        IDs_to_add_remove_provided = IDs_to_add is not None or IDs_to_remove is not None
        if not IDs_to_add_remove_provided:
            self.get_data()
            got_data = True
            current_IDs = set(posData.IDs)
            missing_IDs = list(whitelist - current_IDs)
            to_be_removed_IDs = list(current_IDs - whitelist)
        else:
            missing_IDs = list(IDs_to_add) if IDs_to_add is not None else []
            to_be_removed_IDs = list(IDs_to_remove) if IDs_to_remove is not None else []

        ###

        if benchmark:
            ts.append(time.perf_counter())

        ###
        if not missing_IDs and not to_be_removed_IDs:  # nothing to do
            if og_frame_i != frame_i:
                posData.frame_i = og_frame_i
            if got_data and og_frame_i != frame_i:
                self.get_data()
            if benchmark:
                print("No IDs to add/remove")
                ts.append(time.perf_counter())
                indx = titles.index("track_og_curr")
                titles[indx + 1] = "store_data"
                time_taken = time.perf_counter() - ts[0]
                print(f"\nTotal time for whitelistUpdateLab: {time_taken:.2f}s")
                for i in range(1, len(ts)):
                    time_taken = ts[i] - ts[i - 1]
                    print(f"Time taken for {titles[i]}: {time_taken:.2f}s")
                print("")
            return

        if not got_data and og_frame_i != frame_i:
            self.get_data()
            got_data = True

        if benchmark:
            ts.append(time.perf_counter())

        ###
        if missing_IDs and track_og_curr and not new_frame:
            self.whitelistTrackOGCurr(frame_i=frame_i, lab=posData.lab, rp=posData.rp)

        missing_IDs = np.array(missing_IDs, dtype=np.int32)
        to_be_removed_IDs = np.array(to_be_removed_IDs, dtype=np.int32)

        if debug:
            printl(missing_IDs, to_be_removed_IDs)

        curr_lab = posData.lab  # or curr_lab = posData.lab???
        # convert values to int if they are not already
        if curr_lab is None:
            try:
                curr_lab = posData.allData_li[frame_i]["labels"].copy()
            except:
                pass
        if curr_lab is None:
            try:
                curr_lab = posData.segm_data[frame_i].copy()
            except:
                pass
        if curr_lab is None:
            printl("No current lab?")
            curr_lab = np.zeros_like(posData.segm_data[0])
        curr_lab = curr_lab.astype(np.int32)
        if benchmark:
            ts.append(time.perf_counter())

        if missing_IDs.size > 0 and og_lab is not None:
            mask = np.isin(og_lab, missing_IDs)  # add missing_IDs
            curr_lab[mask] = og_lab[mask]

        if to_be_removed_IDs.size > 0:
            curr_lab[np.isin(curr_lab, to_be_removed_IDs)] = (
                0  # remove to_be_removed_IDs
            )

        if benchmark:
            ts.append(time.perf_counter())

        posData.lab = curr_lab

        self.update_rp(wl_update=False)
        self.store_data()

        if benchmark:
            ts.append(time.perf_counter())
        if og_frame_i != frame_i:
            posData.frame_i = og_frame_i
            self.get_data()

        self.updateAllImages()
        self.setAllTextAnnotations()

        if benchmark:
            ts.append(time.perf_counter())
            time_taken = time.perf_counter() - ts[0]
            print(f"\nTotal time for whitelistUpdateLab: {time_taken:.2f}s")
            for i in range(1, len(ts)):
                time_taken = ts[i] - ts[i - 1]
                print(f"Time taken for {titles[i]}: {time_taken:.2f}s")
            print("")

    def whitelistUpdateTempLayer(self):
        """Updates the temp layer with the current whitelist IDs."""
        if not self.whitelistIDsButton.isChecked():
            self.keepIDsTempLayerLeft.clear()
            return

        if not hasattr(self, "keptLab"):
            self.keptLab = np.zeros_like(self.currentLab2D)
            keptLab = self.keptLab
        else:
            keptLab = self.keptLab
            keptLab[:] = 0

        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            if not hasattr(self, "tempWhitelistIDs"):
                self.tempWhitelistIDs = set()  # not updated, only use in this context
                current_whitelist = self.tempWhitelistIDs
            else:
                current_whitelist = self.tempWhitelistIDs
        else:
            current_whitelist = posData.whitelist.get(posData.frame_i)

        for obj in posData.rp:
            if obj.label not in current_whitelist:
                continue

            if not self.isObjVisible(obj.bbox):
                continue

            _slice = self.getObjSlice(obj.slice)
            _objMask = self.getObjImage(obj.image, obj.bbox)

            keptLab[_slice][_objMask] = obj.label

        self.keepIDsTempLayerLeft.setImage(keptLab, autoLevels=False)

    @exception_handler
    @disableWindow
    def whitelistViewOGIDs(self, checked: bool):
        """Switch between selected and original labels.
        Uses self.viewOriginalLabels to see what has to be done.

        Parameters
        ----------
        checked : bool
            True if the original labels have to be shown, False otherwise.
        """
        switch_to_og = checked and not self.viewOriginalLabels
        switch_to_seg = not checked and self.viewOriginalLabels

        if not switch_to_og and not switch_to_seg:
            return

        posData = self.data[self.pos_i]
        if posData.whitelist is None:
            return

        if posData.whitelist._debug:
            printl("whitelistViewOGIDs", checked)

        frame_i = posData.frame_i
        if frame_i > 0:
            frames_range = [frame_i - 1, frame_i]
        else:
            frames_range = [frame_i]

        self.store_data(autosave=False)

        if not self.whitelistCheckOriginalLabels():
            return
        if switch_to_og:
            self.setFrameNavigationDisabled(True, why="Viewing original labels")
            self.viewOriginalLabels = True

            for i in frames_range:
                posData.frame_i = i
                self.get_data()
                self.whitelistTrackOGCurr(frame_i=i)

                IDs = posData.IDs

                og_frame = posData.whitelist.originalLabs[i].copy()
                IDs_to_uppdate = (
                    posData.whitelist.whitelistIDs[i]
                    & posData.whitelist.originalLabsIDs[i]
                )
                if IDs_to_uppdate:
                    mask = np.isin(og_frame, list(IDs_to_uppdate))
                    og_frame[mask] = 0

                    mask = np.isin(posData.lab, list(IDs_to_uppdate))
                    og_frame[mask] = posData.lab[mask]

                IDs_to_add = (
                    posData.whitelist.whitelistIDs[i]
                    - posData.whitelist.originalLabsIDs[i]
                )
                if IDs_to_add:
                    mask = np.isin(posData.lab, list(IDs_to_add))
                    og_frame[mask] = posData.lab[mask]

                posData.lab = og_frame
                self.update_rp(wl_update=False)
                self.store_data(autosave=False)

            if frame_i > 0:
                missing_IDs = set(posData.IDs) - set(
                    posData.allData_li[frame_i - 1]["IDs"]
                )
                self.trackManuallyAddedObject(
                    missing_IDs, isNewID=True, wl_update=False
                )

            self.setAllTextAnnotations()
            self.updateAllImages()

        elif switch_to_seg:
            self.viewOriginalLabels = False
            self.setFrameNavigationDisabled(False, why="Viewing original labels")

            for i in frames_range:
                posData.frame_i = i
                self.get_data()
                try:
                    posData.whitelist.originalLabs[i] = posData.lab.copy()
                    posData.whitelist.originalLabsIDs[i] = set(posData.IDs)
                except AttributeError:
                    lab = posData.segm_data[i].copy()
                    IDs = [obj.label for obj in skimage.measure.regionprops(lab)]
                    posData.whitelist.originalLabs[i] = lab
                    posData.whitelist.originalLabsIDs[i] = set(IDs)

                # self.whitelistTrackCurrOG()
                self.update_rp(wl_update=False)
                self.store_data(autosave=False)
                self.whitelistUpdateLab(frame_i=i)  # has update_rp and store data
                self.setAllTextAnnotations()
                self.updateAllImages()
