"""Qt view adapter for session workflows."""

from __future__ import annotations

import os
from functools import partial

import numpy as np
import skimage.measure
from qtpy.QtWidgets import QAction

from cellacdc import (
    exception_handler,
    html_utils,
    recentPaths_path,
    settings_csv_path,
    widgets,
)
from cellacdc.gui_decorators import get_data_exception_handler

from .worker import Worker


class Session(Worker):
    """Extracted from guiWin."""

    def _get_data_unvisited(
        self,
        posData,
        debug=False,
        lin_tree_init=True,
    ):
        posData.editID_info = []
        proceed_cca = True
        never_visited = True
        if str(self.modeComboBox.currentText()) == "Cell cycle analysis":
            # Warn that we are visiting a frame that was never segm-checked
            # on cell cycle analysis mode
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                "Segmentation and Tracking was <b>never checked from "
                f"frame {posData.frame_i + 1} onwards</b>.<br><br>"
                "To ensure correct cell cell cycle analysis you have to "
                "first visit the frames after "
                f'{posData.frame_i + 1} with "Segmentation and Tracking" mode.'
            )
            warn_cca = msg.critical(
                self, "Never checked segmentation on requested frame", txt
            )
            proceed_cca = False
            return proceed_cca, never_visited

        elif str(self.modeComboBox.currentText()) == "Normal division: Lineage tree":
            # Warn that we are visiting a frame that was never segm-checked
            # on cell cycle analysis mode
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                "Segmentation and Tracking was <b>never checked from "
                f"frame {posData.frame_i + 1} onwards</b>.<br><br>"
                "To ensure correct lineage tree analysis you have to "
                "first visit the frames after "
                f'{posData.frame_i + 1} with "Segmentation and Tracking" mode.'
            )
            warn_cca = msg.critical(  # ???
                self, "Never checked segmentation on requested frame", txt
            )
            proceed_cca = False
            return proceed_cca, never_visited

        # Requested frame was never visited before. Load from HDD
        labels = self.get_labels()
        posData.lab = self.apply_manual_edits_to_lab_if_needed(labels)
        posData.rp = skimage.measure.regionprops(posData.lab)
        self.setManualBackgroundLab()

        if posData.acdc_df is not None:
            frames = posData.acdc_df.index.get_level_values(0)
            if posData.frame_i in frames:
                # Since there was already segmentation metadata from
                # previous closed session add it to current metadata
                df = posData.acdc_df.loc[posData.frame_i].copy()
                binnedIDs_df = df[df["is_cell_excluded"] > 0]
                binnedIDs = set(binnedIDs_df.index).union(posData.binnedIDs)
                posData.binnedIDs = binnedIDs
                ripIDs_df = df[df["is_cell_dead"] > 0]
                ripIDs = set(ripIDs_df.index).union(posData.ripIDs)
                posData.ripIDs = ripIDs
                posData.editID_info.extend(self._get_editID_info(df))
                # Load cca df into current metadata
                if "cell_cycle_stage" in df.columns:
                    cca_cols = df.columns.intersection(self.cca_df_colnames)
                    cca_df = df[cca_cols].dropna()
                    if cca_df.empty:
                        df = df.drop(columns=self.cca_df_colnames, errors="ignore")
                    else:
                        df = df.loc[cca_df.index]
                        cols = self.cca_df_int_cols
                        df[cols] = df[cols].astype("Int64")

                i = posData.frame_i
                posData.allData_li[i]["acdc_df"] = df.copy()

        if self.lineage_tree is None and lin_tree_init:
            self.initLinTree()

        self.get_cca_df()

        return proceed_cca, never_visited

    def _get_data_visited(
        self,
        posData,
        debug=False,
        lin_tree_init=True,
    ):
        # Requested frame was already visited. Load from RAM.
        never_visited = False
        posData.lab = self.get_labels(from_store=True)
        posData.rp = skimage.measure.regionprops(posData.lab)
        df = posData.allData_li[posData.frame_i]["acdc_df"]
        if df is None:
            posData.binnedIDs = set()
            posData.ripIDs = set()
            posData.editID_info = []
        else:
            try:
                binnedIDs_df = df[df["is_cell_excluded"] > 0]
            except Exception as err:
                df = utils.fix_acdc_df_dtypes(df)
                binnedIDs_df = df[df["is_cell_excluded"] > 0]
            posData.binnedIDs = set(binnedIDs_df.index)
            ripIDs_df = df[df["is_cell_dead"] > 0]
            posData.ripIDs = set(ripIDs_df.index)
            posData.editID_info = self._get_editID_info(df)
        self.setManualBackgroundLab(load_from_store=True, debug=debug)
        if self.lineage_tree is None and lin_tree_init:
            self.initLinTree()

        self.get_cca_df(debug=debug)

        return True, never_visited

    def addPathToOpenRecentMenu(self, path):
        for action in self.openRecentMenu.actions():
            if path == action.text():
                break
        else:
            action = QAction(path, self)
            action.triggered.connect(partial(self.openRecentFile, path))

        try:
            firstAction = self.openRecentMenu.actions()[0]
            self.openRecentMenu.insertAction(firstAction, action)
        except Exception as e:
            pass

    def getStoredSegmData(self):
        posData = self.data[self.pos_i]
        segm_data = []
        for data_frame_i in posData.allData_li:
            lab = data_frame_i["labels"]
            if lab is None:
                break
            segm_data.append(lab)
        return np.array(segm_data)

    def get_data(self, debug=False, lin_tree_init=True):
        posData = self.data[self.pos_i]
        proceed_cca = True
        never_visited = False
        if posData.frame_i > 2:
            # Remove undo states from 4 frames back to avoid memory issues
            posData.UndoRedoStates[posData.frame_i - 4] = []
            # Check if current frame contains undo states (not empty list)
            if posData.UndoRedoStates[posData.frame_i]:
                self.undoAction.setDisabled(False)
            elif posData.UndoRedoCcaStates[posData.frame_i]:
                self.undoAction.setDisabled(False)
            else:
                self.undoAction.setDisabled(True)
        self.UndoCount = 0
        # If stored labels is None then it is the first time we visit this frame
        if posData.allData_li[posData.frame_i]["labels"] is None:
            proceed_cca, never_visited = self._get_data_unvisited(
                posData,
                lin_tree_init=lin_tree_init,
            )
            if not proceed_cca:
                return proceed_cca, never_visited
        else:
            proceed_cca, never_visited = self._get_data_visited(
                posData, lin_tree_init=lin_tree_init, debug=debug
            )

        self.update_rp_metadata(draw=False)
        posData.IDs = [obj.label for obj in posData.rp]
        posData.IDs_idxs = {
            ID: i for ID, i in zip(posData.IDs, range(len(posData.IDs)))
        }
        self.get_zslices_rp()
        self.pointsLayerDfsToData(posData)
        return proceed_cca, never_visited

    def get_labels(
        self, from_store=False, frame_i=None, return_existing=False, return_copy=True
    ):
        """Get the labels array.

        Parameters
        ----------
        from_store : bool, optional
            If True load the labels array from the stored posData.allData_li,
            i.e., from RAM. Default is False
        frame_i : int, optional
            If None, use the current frame index. Default is  None
        return_existing : bool, optional
            If True, the second return element will be a boolean that
            is True if the labels array was found stored in `posData.allData_li`.
            Default is  False
        return_copy : bool, optional
            If True returns a copy of the labels array

        Returns
        -------
        numpy.ndarray or tuple of (numpy.ndarray, bool)
            The first element is the labels array requested. If `return_existing`
            is True then this method also returns a second boolean element that
            is True if the labels array was found in in `posData.allData_li`.

        Note
        ----

        If `from_store` is True then this method will try to get the stored
        labels array. If any error occurs then the returned labels are the
        saved ones in the segmentation file (i.e., from hard drive).

        """
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i

        existing = True
        if from_store:
            try:
                labels = posData.allData_li[frame_i]["labels"]
                if labels is None:
                    from_store = False
            except Exception as err:
                from_store = False

        if not from_store:
            try:
                labels = posData.segm_data[frame_i]
            except IndexError:
                existing = False
                # Visting a frame that was not segmented --> empty masks
                if self.isSegm3D:
                    shape = (posData.SizeZ, posData.SizeY, posData.SizeX)
                else:
                    shape = (posData.SizeY, posData.SizeX)
                labels = np.zeros(shape, dtype=np.uint32)
                return_copy = False

        if return_copy:
            labels = labels.copy()

        if return_existing:
            return labels, existing
        else:
            return labels

    def initPosAttr(self):
        exp_path = self.data[self.pos_i].exp_path
        pos_foldernames = utils.get_pos_foldernames(exp_path)
        if len(pos_foldernames) == 1:
            self.loadPosAction.setDisabled(True)
        else:
            self.loadPosAction.setDisabled(False)

        for p, posData in enumerate(self.data):
            self.pos_i = p
            posData.curvPlotItems = []
            posData.curvAnchorsItems = []
            posData.curvHoverItems = []
            posData.trackedLostIDs = set()

            posData.HDDmaxID = np.max(posData.segm_data)

            # Decision on what to do with changes to future frames attr
            posData.doNotShowAgain_EditID = False
            posData.UndoFutFrames_EditID = False
            posData.applyFutFrames_EditID = False

            posData.doNotShowAgain_RipID = False
            posData.UndoFutFrames_RipID = False
            posData.applyFutFrames_RipID = False

            posData.doNotShowAgain_DelID = False
            posData.UndoFutFrames_DelID = False
            posData.applyFutFrames_DelID = False

            posData.doNotShowAgain_keepID = False
            posData.UndoFutFrames_keepID = False
            posData.applyFutFrames_keepID = False

            posData.doNotShowAgainAssignNewID = False
            posData.UndoFutFramesAssignNewID = False
            posData.applyFutFramesAssignNewID = False

            posData.includeUnvisitedInfo = {
                "Delete ID": False,
                "Edit ID": False,
                "Keep ID": False,
            }

            posData.loadTrackedLostCentroids()
            posData.acdcTracker2stepsAnnotInfo = {}

            posData.doNotShowAgain_BinID = False
            posData.UndoFutFrames_BinID = False
            posData.applyFutFrames_BinID = False

            posData.disableAutoActivateViewerWindow = False
            posData.new_IDs = []
            posData.lost_IDs = []
            posData.multiBud_mothIDs = [2]
            posData.UndoRedoStates = [[] for _ in range(posData.SizeT)]
            posData.UndoRedoCcaStates = [[] for _ in range(posData.SizeT)]

            posData.ol_data_dict = {}
            posData.ol_data = None

            posData.ol_labels_data = None

            missing_frames = posData.SizeT - len(posData.allData_li)
            if missing_frames > 0:
                posData.allData_li.extend([None] * missing_frames)
            for i in range(posData.SizeT):
                if posData.allData_li[i] is None:
                    posData.allData_li[i] = utils.get_empty_stored_data_dict()

            posData.lutLevels = {channel: {} for channel in self.ch_names}

            posData.ccaStatus_whenEmerged = {}

            posData.frame_i = 0
            posData.brushID = 0
            posData.binnedIDs = set()
            posData.ripIDs = set()
            posData.cca_df = None
            if posData.last_tracked_i is not None:
                last_tracked_num = posData.last_tracked_i + 1
                # Load previous session data
                # Keep track of which ROIs have already been added
                # in previous frame
                delROIshapes = [[] for _ in range(posData.SizeT)]
                for i in range(last_tracked_num):
                    posData.frame_i = i
                    self.get_data(debug=True)
                    self.store_data(
                        enforce=True, autosave=False, store_cca_df_copy=True
                    )

                # Ask whether to resume from last frame
                if last_tracked_num > 1:
                    msg = widgets.myMessageBox()
                    txt = html_utils.paragraph(
                        "Cell-ACDC detected a previous session ended "
                        f"at frame {last_tracked_num}.<br><br>"
                        f"Do you want to <b>resume from frame "
                        f"{last_tracked_num}?</b>"
                    )
                    noButton, yesButton = msg.question(
                        self,
                        "Start from last session?",
                        txt,
                        buttonsTexts=(" No ", "Yes"),
                    )
                    self.AutoPilotProfile.storeClickMessageBox(
                        "Start from last session?", msg.clickedButton.text()
                    )
                    if msg.clickedButton == yesButton:
                        posData.frame_i = posData.last_tracked_i
                        self.lastFrameRanOnFirstVisitTools = posData.frame_i
                    else:
                        posData.frame_i = 0

            posData.img_data_min_max = (posData.img_data.min(), posData.img_data.max())

        # Back to first position
        self.pos_i = 0
        self.get_data(debug=False)
        self.store_data(autosave=False)
        # self.updateAllImages()

        # Link Y and X axis of both plots to scroll zoom and pan together
        self.ax2.vb.setYLink(self.ax1.vb)
        self.ax2.vb.setXLink(self.ax1.vb)

        self.setAllIDs()

    def loadLastSessionSettings(self):
        self.settings_csv_path = settings_csv_path
        if os.path.exists(settings_csv_path):
            self.df_settings = pd.read_csv(settings_csv_path, index_col="setting")
            if "is_bw_inverted" not in self.df_settings.index:
                self.df_settings.at["is_bw_inverted", "value"] = "No"
            else:
                self.df_settings.loc["is_bw_inverted"] = self.df_settings.loc[
                    "is_bw_inverted"
                ].astype(str)
            if "fontSize" not in self.df_settings.index:
                self.df_settings.at["fontSize", "value"] = 12
            if "overlayColor" not in self.df_settings.index:
                self.df_settings.at["overlayColor", "value"] = "255-255-0"
            if "how_normIntensities" not in self.df_settings.index:
                raw = "Do not normalize. Display raw image"
                self.df_settings.at["how_normIntensities", "value"] = raw
        else:
            idx = ["is_bw_inverted", "fontSize", "overlayColor", "how_normIntensities"]
            values = ["No", 12, "255-255-0", "raw"]
            self.df_settings = pd.DataFrame(
                {"setting": idx, "value": values}
            ).set_index("setting")

        if "isLabelsVisible" not in self.df_settings.index:
            self.df_settings.at["isLabelsVisible", "value"] = "No"

        if "isNextFrameVisible" not in self.df_settings.index:
            self.df_settings.at["isNextFrameVisible", "value"] = "No"

        if "isRightImageVisible" not in self.df_settings.index:
            self.df_settings.at["isRightImageVisible", "value"] = "Yes"

        if "manual_separate_draw_mode" not in self.df_settings.index:
            col = "manual_separate_draw_mode"
            self.df_settings.at[col, "value"] = "threepoints_arc"

        if "colorScheme" in self.df_settings.index:
            col = "colorScheme"
            self._colorScheme = self.df_settings.at[col, "value"]
        else:
            self._colorScheme = "light"

        self.doNotShowAgainMissingCca = False
        if "doNotShowAgainMissingCca" not in self.df_settings.index:
            self.df_settings.at["doNotShowAgainMissingCca", "value"] = "No"
        else:
            val = self.df_settings.at["doNotShowAgainMissingCca", "value"]
            self.doNotShowAgainMissingCca = val == "Yes"

    def reInitGui(self):
        cancel = self.checkAskSavePointsLayers()
        if cancel:
            return False

        if self.overlayToolbar.isTransparent():
            self.overlayToolbar.setTransparent(False)

        self.secondLevelToolbar.setVisible(False)

        self.gui_createLazyLoader()

        try:
            self.navSpinBox.valueChanged.disconnect()
        except Exception as e:
            pass

        try:
            self.scaleBar.removeFromAxis(self.ax1)
        except Exception as e:
            pass

        self.lineage_tree = None
        self.getDistanceListMissingIDsCachedFrame = None
        self.isZmodifier = False
        self.zKeptDown = False
        self.askRepeatSegment3D = True
        self.askZrangeSegm3D = True
        self.isDataLoaded = False
        self.retainSizeLutItems = False
        self.setMeasWinState = None
        self.addPointsWin = None
        self.delRoiLab = None
        self.showPropsDockButton.setDisabled(True)
        self.removeOverlayItems()
        self.lutItemsLayout.addItem(self.imgGrad, row=0, col=0)

        self.reinitWidgetsPos()
        self.removeAllItems()
        self.reinitCustomAnnot()
        self.reinitPointsLayers()
        self.gui_createPlotItems()
        self.setUncheckedAllButtons()
        self.setUncheckedPointsLayers()
        self.restoreDefaultColors()
        self.reinitStoredSegmModels()
        self.removeAxLimits()
        self.curvToolButton.setChecked(False)

        self.wandControlsToolbar.setVisible(False)
        self.wandToolButton.setChecked(False)
        self.segmNdimIndicatorAction.setVisible(False)

        self.navigateToolBar.hide()
        self.ccaToolBar.hide()
        self.editToolBar.hide()
        self.brushEraserToolBar.hide()
        self.modeToolBar.hide()

        self.modeComboBox.setCurrentText("Viewer")

        alpha = self.imgGrad.labelsAlphaSlider.value()
        self.labelsLayerImg1.setOpacity(alpha)
        self.labelsLayerRightImg.setOpacity(alpha)
        self.lastTrackedFrameLabel.setText("")

        self.promptSegmentPointsLayerToolbar.isPointsLayerInit = False

        for action in self.askHowFutureFramesActions.values():
            action.setChecked(True)
            action.setDisabled(True)

        return True

    def readRecentPaths(self, recent_paths_path=None):
        # Step 0. Remove the old options from the menu
        self.openRecentMenu.clear()

        # Step 1. Read recent Paths
        if recent_paths_path is None:
            recent_paths_path = recentPaths_path

        if os.path.exists(recent_paths_path):
            df = pd.read_csv(recent_paths_path, index_col="index")
            df["path"] = df["path"].str.replace("\\", "/")
            df = df.drop_duplicates(subset=["path"])
            df.to_csv(recent_paths_path)
            if "opened_last_on" in df.columns:
                df = df.sort_values("opened_last_on", ascending=False)
            recentPaths = df["path"].to_list()
        else:
            recentPaths = []

        # Step 2. Dynamically create the actions
        actions = []
        for path in recentPaths:
            if not os.path.exists(path):
                continue
            action = QAction(path, self)
            action.triggered.connect(partial(self.openRecentFile, path))
            actions.append(action)

        # Step 3. Add the actions to the menu
        self.openRecentMenu.addActions(actions)

    def reinitWidgetsPos(self):
        pass

    def store_data(
        self,
        pos_i=None,
        enforce=True,
        debug=False,
        mainThread=True,
        autosave=True,
        store_cca_df_copy=False,
    ):
        pos_i = self.pos_i if pos_i is None else pos_i
        posData = self.data[pos_i]
        if posData.frame_i < 0:
            # In some cases we set frame_i = -1 and then call next_frame
            # to visualize frame 0. In that case we don't store data
            # for frame_i = -1
            return

        mode = str(self.modeComboBox.currentText())

        if mode == "Viewer" and not enforce:
            return

        # if not mainThread:
        #     self.lin_tree_ask_changes()

        allData_li = posData.allData_li[posData.frame_i]
        allData_li["regionprops"] = posData.rp.copy()
        allData_li["labels"] = posData.lab.copy()
        allData_li["IDs"] = posData.IDs.copy()
        allData_li["manualBackgroundLab"] = posData.manualBackgroundLab
        allData_li["IDs_idxs"] = posData.IDs_idxs.copy()
        if self.manualAnnotPastButton.isChecked():
            self.store_manual_annot_data(posData=posData, data_frame_i=allData_li)

        self.store_zslices_rp()

        # Store dynamic metadata
        is_cell_dead_li = [False] * len(posData.rp)
        is_cell_excluded_li = [False] * len(posData.rp)
        IDs = [0] * len(posData.rp)
        xx_centroid = [0] * len(posData.rp)
        yy_centroid = [0] * len(posData.rp)
        if self.isSegm3D:
            zz_centroid = [0] * len(posData.rp)
        areManuallyEdited = [0] * len(posData.rp)
        editedNewIDs = [vals[2] for vals in posData.editID_info]
        for i, obj in enumerate(posData.rp):
            is_cell_dead_li[i] = obj.dead
            is_cell_excluded_li[i] = obj.excluded
            IDs[i] = obj.label
            try:
                xx_centroid[i] = int(self.getObjCentroid(obj.centroid)[1])
                yy_centroid[i] = int(self.getObjCentroid(obj.centroid)[0])
            except Exception as err:
                printl(obj, obj.centroid, obj.label, posData.frame_i)
            if self.isSegm3D:
                zz_centroid[i] = int(obj.centroid[0])
            if obj.label in editedNewIDs:
                areManuallyEdited[i] = 1

        posData.STOREDmaxID = max(IDs, default=0)

        acdc_df = allData_li["acdc_df"]
        if acdc_df is None:
            allData_li["acdc_df"] = pd.DataFrame(
                {
                    "Cell_ID": IDs,
                    "is_cell_dead": is_cell_dead_li,
                    "is_cell_excluded": is_cell_excluded_li,
                    "x_centroid": xx_centroid,
                    "y_centroid": yy_centroid,
                    "was_manually_edited": areManuallyEdited,
                }
            ).set_index("Cell_ID")

            if self.isSegm3D:
                allData_li["acdc_df"]["z_centroid"] = zz_centroid
        else:
            # Filter or add IDs that were not stored yet
            acdc_df = acdc_df.drop(columns=["time_seconds"], errors="ignore")
            acdc_df = acdc_df.reindex(IDs, fill_value=0)
            acdc_df["is_cell_dead"] = is_cell_dead_li
            acdc_df["is_cell_excluded"] = is_cell_excluded_li
            acdc_df["x_centroid"] = xx_centroid
            acdc_df["y_centroid"] = yy_centroid
            if self.isSegm3D:
                acdc_df["z_centroid"] = zz_centroid
            acdc_df["was_manually_edited"] = areManuallyEdited
            allData_li["acdc_df"] = acdc_df

        if mainThread:
            self.pointsLayerDataToDf(posData)

        self.store_cca_df(
            pos_i=pos_i,
            mainThread=mainThread,
            autosave=autosave,
            store_cca_df_copy=store_cca_df_copy,
        )

    def store_manual_annot_data(self, posData=None, data_frame_i=None):
        if posData is None:
            posData = self.data[self.pos_i]

        if data_frame_i is None:
            data_frame_i = posData.allData_li[posData.frame_i]

        if not self.isSegm3D:
            lab = [posData.lab]
        else:
            lab = posData.lab

        for z, lab_2D in enumerate(lab):
            data_frame_i["manually_edited_lab"]["lab"][z] = lab_2D

    def unstore_data(self):
        posData = self.data[self.pos_i]
        posData.allData_li[posData.frame_i] = utils.get_empty_stored_data_dict()

    def updateLastVisitedFrame(self, last_visited_frame_i=None):
        if last_visited_frame_i is None:
            posData = self.data[self.pos_i]
            last_visited_frame_i = posData.frame_i

        mode = str(self.modeComboBox.currentText())
        if mode == "Viewer":
            return
        elif mode == "Segmentation and Tracking":
            posData = self.data[self.pos_i]
            if posData.last_tracked_i >= last_visited_frame_i:
                return
            posData.last_tracked_i = last_visited_frame_i
        elif mode == "Cell cycle analysis":
            if self.last_cca_frame_i >= last_visited_frame_i:
                return
            self.last_cca_frame_i = last_visited_frame_i
