"""Qt view adapter for frame and position navigation."""

from __future__ import annotations

from collections import Counter
from functools import partial

import numpy as np
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QAbstractSlider, QCheckBox

from cellacdc import QtScoped, apps, exception_handler, html_utils, printl, widgets


SliderSingleStepAdd = QtScoped.SliderSingleStepAdd()
SliderSingleStepSub = QtScoped.SliderSingleStepSub()
SliderPageStepAdd = QtScoped.SliderPageStepAdd()
SliderPageStepSub = QtScoped.SliderPageStepSub()
SliderMove = QtScoped.SliderMove()

from .graphics import Graphics
from .label_editing import LabelEditing


class FrameNavigation(Graphics, LabelEditing):
    """Extracted from guiWin."""

    def PosScrollBarAction(self, action):
        if action == SliderSingleStepAdd:
            self.next_cb()
        elif action == SliderSingleStepSub:
            self.prev_cb()
        elif action == SliderPageStepAdd:
            self.PosScrollBarReleased()
        elif action == SliderPageStepSub:
            self.PosScrollBarReleased()

    def PosScrollBarMoved(self, pos_n):
        if self.navigateScrollBarStartedMoving:
            self.store_data()

        self.pos_i = pos_n - 1
        self.updateFramePosLabel()
        proceed_cca, never_visited = self.get_data()
        self.updateAllImages()
        self.setStatusBarLabel()
        self.navigateScrollBarStartedMoving = False

    def PosScrollBarReleased(self):
        self.navigateScrollBarStartedMoving = True
        if self.pos_i == self.navigateScrollBar.sliderPosition() - 1:
            # Slider released without changing value --> do nothing
            return

        self.pos_i = self.navigateScrollBar.sliderPosition() - 1
        self.updateFramePosLabel()
        self.updatePos()

    def _setViewRangeSwitchPlane(self, previousPlane):
        posData = self.data[self.pos_i]
        SizeZ = posData.SizeZ
        SizeY, SizeX = self.img1.image.shape[:2]
        currentPlane = self.switchPlaneCombobox.plane()
        if previousPlane == "xy":
            if currentPlane == "zy":
                self.ax1.setRange(xRange=self.yRangePrev)
                unusedRange = np.clip(self.xRangePrev, 0, SizeX)
            elif currentPlane == "zx":
                self.ax1.setRange(xRange=self.xRangePrev)
                unusedRange = np.clip(self.yRangePrev, 0, SizeY)
        elif previousPlane == "zy":
            if currentPlane == "xy":
                self.ax1.setRange(yRange=self.xRangePrev)
                unusedRange = np.clip(self.yRangePrev, 0, SizeZ)
            elif currentPlane == "zx":
                self.ax1.setRange(yRange=self.yRangePrev)
                unusedRange = np.clip(self.xRangePrev, 0, SizeY)
        elif previousPlane == "zx":
            if currentPlane == "xy":
                self.ax1.setRange(xRange=self.xRangePrev)
                unusedRange = np.clip(self.yRangePrev, 0, SizeZ)
            elif currentPlane == "zy":
                self.ax1.setRange(yRange=self.yRangePrev)
                unusedRange = np.clip(self.xRangePrev, 0, SizeX)

        sliceValue = round((unusedRange[0] + unusedRange[1]) / 2)
        self.zSliceScrollBar.setSliderPosition(sliceValue)
        self.update_z_slice(self.zSliceScrollBar.sliderPosition())

    def apply_tools_on_new_frame(self):
        mode = str(self.modeComboBox.currentText())
        if mode != "Segmentation and Tracking":
            return
        posData = self.data[self.pos_i]
        if (
            not (posData.last_tracked_i <= posData.frame_i)
            or posData.frame_i == self.lastFrameRanOnFirstVisitTools
        ):
            return

        self.lastFrameRanOnFirstVisitTools = posData.frame_i
        for name, checkbox in self.applyToolNewFrameActions.items():
            if not checkbox.isChecked():
                continue

            tool_button = self.applyToolNewFrameButtons[name]
            try:
                if hasattr(tool_button, "click"):
                    tool_button.click()
                elif hasattr(tool_button, "trigger"):
                    tool_button.trigger()
                else:
                    printl(f"Warning: {name} has no click or trigger method")
            except Exception as e:
                self.logger.info(f"Error applying tool {name}: {e}")

    def askInitCcaFirstFrame(self):
        mode = str(self.modeComboBox.currentText())
        if mode != "Cell cycle analysis":
            return True

        posData = self.data[self.pos_i]
        if posData.frame_i != 0:
            return True

        editCcaWidget = apps.editCcaTableWidget(
            posData.cca_df,
            posData.SizeT,
            parent=self,
            title="Initialize cell cycle annotations",
        )
        editCcaWidget.sigApplyChangesFutureFrames.connect(
            self.applyManualCcaChangesFutureFrames
        )
        editCcaWidget.exec_()
        if editCcaWidget.cancel:
            self.resetNavigateFramesScrollbar()
            return False

        if posData.cca_df is not None:
            is_cca_same_as_stored = (posData.cca_df == editCcaWidget.cca_df).all(
                axis=None
            )
            if not is_cca_same_as_stored:
                reinit_cca = self.warnEditingWithCca_df(
                    "Re-initialize cell cyle annotations first frame",
                    return_answer=True,
                )
                if reinit_cca:
                    self.resetCcaFuture(0)

        posData.cca_df = editCcaWidget.cca_df
        self.store_cca_df()

        return True

    def askInitLinTreeFirstFrame(self):
        mode = str(self.modeComboBox.currentText())
        if mode != "Normal division: Lineage tree":
            return True

        posData = self.data[self.pos_i]
        if posData.frame_i != 0:
            return True

        if self.lineage_tree is None:
            self.initLinTree()

        return True

    def checkIfFutureFrameManualAnnotPastFrames(self):
        if not self.manualAnnotPastButton.isChecked():
            return True

        posData = self.data[self.pos_i]
        frame_to_restore = self.manualAnnotState.get("frame_i_to_restore")
        if posData.frame_i <= frame_to_restore:
            return True

        warn_txt = (
            "WARNING: Cannot navigate to future frames while in manual annotation mode."
        )
        self.logger.info(warn_txt)
        self.statusBarLabel.setText(f'<p style="color:red;">{warn_txt}</p>')

        return False

    def connectScrollbars(self):
        self.t_label.show()
        self.navigateScrollBar.show()
        self.navigateScrollBar.setDisabled(False)

        if self.data[0].SizeZ > 1:
            self.enableZstackWidgets(True)
            self.zSliceScrollBar.setMaximum(self.data[0].SizeZ - 1)
            self.zSliceSpinbox.setMaximum(self.data[0].SizeZ)
            self.SizeZlabel.setText(f"/{self.data[0].SizeZ}")
            try:
                self.zSliceScrollBar.actionTriggered.disconnect()
                self.zSliceScrollBar.sliderReleased.disconnect()
                self.zProjComboBox.currentTextChanged.disconnect()
                self.zProjComboBox.activated.disconnect()
                self.switchPlaneCombobox.sigPlaneChanged.disconnect()
                self.zProjLockViewButton.toggled.disconnect()
            except Exception as e:
                pass
            self.zSliceScrollBar.actionTriggered.connect(
                self.zSliceScrollBarActionTriggered
            )
            self.zSliceScrollBar.sliderReleased.connect(self.zSliceScrollBarReleased)
            self.zProjComboBox.currentTextChanged.connect(self.updateZproj)
            self.zProjComboBox.activated.connect(self.clearComboBoxFocus)
            self.switchPlaneCombobox.sigPlaneChanged.connect(self.switchViewedPlane)
            self.zProjLockViewButton.toggled.connect(self.zProjLockViewToggled)

        posData = self.data[self.pos_i]
        if posData.SizeT == 1:
            self.t_label.setText("Position n.")
            self.navigateScrollBar.setMinimum(1)
            self.navigateScrollBar.setMaximum(len(self.data))
            self.navigateScrollBar.setAbsoluteMaximum(len(self.data))
            self.navSpinBox.setMaximum(len(self.data))
            self.navigateScrollBar.connectEvents(
                {
                    "sliderMoved": self.PosScrollBarMoved,
                    "sliderReleased": self.PosScrollBarReleased,
                    "actionTriggered": self.PosScrollBarAction,
                }
            )
        else:
            self.navigateScrollBar.setMinimum(1)
            self.navigateScrollBar.setAbsoluteMaximum(posData.SizeT)
            self.rightImageFramesScrollbar.setMinimum(1)
            self.rightImageFramesScrollbar.setMaximum(posData.SizeT)
            if posData.last_tracked_i is not None:
                self.navigateScrollBar.setMaximum(posData.last_tracked_i + 1)
                self.navSpinBox.setMaximum(posData.last_tracked_i + 1)
            self.t_label.setText("Frame n.")
            self.navigateScrollBar.connectEvents(
                {
                    "sliderMoved": self.framesScrollBarMoved,
                    "sliderReleased": self.framesScrollBarReleased,
                    "actionTriggered": self.framesScrollBarActionTriggered,
                }
            )
            self.rightImageFramesScrollbar.connectValueChanged(
                self.rightImageFramesScrollbarValueChanged
            )

    def extendSegmDataIfNeeded(self, stopFrameNum):
        posData = self.data[self.pos_i]
        segmSizeT = len(posData.segm_data)
        if stopFrameNum <= segmSizeT:
            return
        numFramesToAdd = stopFrameNum - segmSizeT
        posData.allData_li.extend(
            [myutils.get_empty_stored_data_dict() for i in range(numFramesToAdd)]
        )
        lab_shape = posData.segm_data[0].shape
        shapeToAdd = (numFramesToAdd, *lab_shape)
        additionalSegmData = np.zeros(shapeToAdd, dtype=posData.segm_data.dtype)
        extendedSegmData = np.concatenate((posData.segm_data, additionalSegmData))
        posData.segm_data = extendedSegmData

    def framesScrollBarActionTriggered(self, action):
        if action == SliderSingleStepAdd:
            # Clicking on dialogs triggered by next_cb might trigger
            # pressEvent of navigateQScrollBar, avoid that
            self.navigateScrollBar.disableCustomPressEvent()
            self.next_cb()
            QTimer.singleShot(100, self.navigateScrollBar.enableCustomPressEvent)
        elif action == SliderSingleStepSub:
            self.prev_cb()
        elif action == SliderPageStepAdd:
            self.framesScrollBarReleased(do_store_data=True)
        elif action == SliderPageStepSub:
            self.framesScrollBarReleased(do_store_data=True)

    def framesScrollBarMoved(self, frame_n):
        if self.navigateScrollBarStartedMoving:
            mode = str(self.modeComboBox.currentText())
            if mode != "Viewer":
                self.store_data(debug=False)

        posData = self.data[self.pos_i]
        posData.frame_i = frame_n - 1
        if posData.allData_li[posData.frame_i]["labels"] is None:
            if posData.frame_i < len(posData.segm_data):
                posData.lab = posData.segm_data[posData.frame_i]
            else:
                posData.lab = np.zeros_like(posData.segm_data[0])
        else:
            posData.lab = posData.allData_li[posData.frame_i]["labels"]

        self.setImageImg1()
        if self.overlayButton.isChecked():
            self.setOverlayImages()

        if self.navigateScrollBarStartedMoving:
            self.clearAllItems()

        self.navSpinBox.setValueNoEmit(posData.frame_i + 1)
        if self.labelsGrad.showLabelsImgAction.isChecked():
            self.img2.setImage(posData.lab, z=self.z_lab(), autoLevels=False)
        self.updateLookuptable()
        self.updateFramePosLabel()
        self.updateViewerWindow()
        self.updateTimestampFrame()
        self.updateHighlightedAxis()
        self.navigateScrollBarStartedMoving = False

    def framesScrollBarReleased(self, do_store_data=False):
        posData = self.data[self.pos_i]
        if posData.frame_i == self.navigateScrollBar.sliderPosition() - 1:
            # Slider released without changing value --> do nothing
            return

        mode = str(self.modeComboBox.currentText())
        if mode != "Viewer" and do_store_data:
            self.store_data(debug=False)

        self.navigateScrollBarStartedMoving = True
        posData.frame_i = self.navigateScrollBar.sliderPosition() - 1
        self.updateFramePosLabel()
        proceed_cca, never_visited = self.get_data()
        self.updateAllImages()

    def goToZsliceSearchedID(self, obj):
        if not self.isSegm3D:
            return

        current_z = self.z_lab()
        nearest_nonzero_z = core.nearest_nonzero_z_idx_from_z_centroid(
            obj, current_z=current_z
        )
        if nearest_nonzero_z == current_z:
            self.drawPointsLayers(computePointsLayers=True)
            return

        self.zSliceScrollBar.setSliderPosition(nearest_nonzero_z)
        self.update_z_slice(nearest_nonzero_z)

    def isNavigateActionOnNextFrame(self):
        posData = self.data[self.pos_i]
        if posData.SizeT == 1:
            return False

        ax1_coords = self.getMouseDataCoordsRightImage()
        if ax1_coords is None:
            return False

        if not self.labelsGrad.showNextFrameAction.isEnabled():
            return False

        if not self.labelsGrad.showNextFrameAction.isChecked():
            return

        # Mouse is on right image and next frame action is checked
        return True

    def manualAnnotRestoreLastTrackedFrame(self, last_tracked_i_to_restore):
        if self.navigateScrollBar.maximum() - 1 <= last_tracked_i_to_restore:
            return

        posData = self.data[self.pos_i]
        for frame_i in range(last_tracked_i_to_restore + 1, posData.SizeT):
            data_frame_i = myutils.get_empty_stored_data_dict()

            data_frame_i["manually_edited_lab"] = posData.allData_li[frame_i][
                "manually_edited_lab"
            ]

            posData.allData_li[frame_i] = data_frame_i

        self.navigateScrollBar.setMaximum(last_tracked_i_to_restore + 1)
        self.navSpinBox.setMaximum(last_tracked_i_to_restore + 1)

    def navigateSpinboxEditingFinished(self):
        if self.isSnapshot:
            self.PosScrollBarReleased()
        else:
            self.framesScrollBarReleased()

    def navigateSpinboxValueChanged(self, value):
        self.navigateScrollBar.setSliderPosition(value)
        if self.isSnapshot:
            self.PosScrollBarMoved(value)
        else:
            self.navigateScrollBarStartedMoving = True
            self.framesScrollBarMoved(value)

    def nextActionTriggered(self):
        if self.isNavigateActionOnNextFrame():
            self.rightImageFramesScrollbar.setValue(
                self.rightImageFramesScrollbar.value() + 1
            )
            return

        stepAddAction = QAbstractSlider.SliderAction.SliderSingleStepAdd
        if self.zKeptDown or self.zSliceCheckbox.isChecked():
            self.zSliceScrollBar.triggerAction(stepAddAction)
        else:
            self.navigateScrollBar.triggerAction(stepAddAction)

    def nextFrameImage(self, current_frame_i=None):
        if not self.labelsGrad.showNextFrameAction.isEnabled():
            return

        if not self.labelsGrad.showNextFrameAction.isChecked():
            return

        posData = self.data[self.pos_i]
        if current_frame_i is None:
            current_frame_i = posData.frame_i

        next_frame_i = current_frame_i + 1
        if next_frame_i >= len(posData.img_data):
            img = posData.img_data[-1]
        else:
            img = posData.img_data[next_frame_i]

        if posData.SizeZ > 1:
            img = self.get_2Dimg_from_3D(img, isLayer0=True)

        # img = self.normalizeIntensities(img)

        return img

    def next_cb(self):
        if self.isSnapshot:
            self.next_pos()
        else:
            self.next_frame()
        if self.curvToolButton.isChecked():
            self.curvTool_cb(True)

        self.updatePropsWidget("")

    def next_frame(self, warn=True):
        proceed = self.checkIfFutureFrameManualAnnotPastFrames()
        if not proceed:
            return

        proceed = self.askInitCcaFirstFrame()
        if not proceed:
            return

        proceed = self.askInitLinTreeFirstFrame()
        if not proceed:
            return

        mode = str(self.modeComboBox.currentText())
        posData = self.data[self.pos_i]

        if posData.frame_i >= posData.SizeT - 1:
            # Store data for current frame
            if mode != "Viewer":
                self.store_data(debug=False)
            msg = "You reached the last segmented frame!"
            self.logger.info(msg)
            self.titleLabel.setText(msg, color=self.titleColor)
            return

        proceed = self.warnLostObjects()
        if not proceed:
            self.resetNavigateScrollbar()
            return

        # Store data for current frame
        if mode != "Viewer":
            self.store_data(debug=False)

        self.askLineageTreeChanges()
        posData.frame_i += 1
        self.removeAlldelROIsCurrentFrame()
        proceed_cca, never_visited = self.get_data()
        if not proceed_cca:
            posData.frame_i -= 1
            self.get_data()
            self.logger.info("No data for current frame. ")
            return

        if mode == "Segmentation and Tracking" or self.isSnapshot:
            self.addExistingDelROIs()

        self.updatePreprocessPreview()
        self.updateCombineChannelsPreview()
        self.postProcessing()
        self.tracking(storeUndo=True, wl_update=False)
        notEnoughG1Cells, proceed = self.attempt_auto_cca()
        if notEnoughG1Cells or not proceed:
            posData.frame_i -= 1
            self.get_data()
            self.setAllTextAnnotations()
            self.logger.info("Not enough G1 cells to compute cell cycle annotations.")
            return

        self.store_zslices_rp()
        self.resetExpandLabel()
        self.updateAllImages()
        self.updateHighlightedAxis()
        self.updateViewerWindow()
        self.updateLastVisitedFrame(last_visited_frame_i=posData.frame_i - 1)
        self.setNavigateScrollBarMaximum()
        self.updateScrollbars()
        self.computeSegm()
        self.initGhostObject()
        self.whitelistPropagateIDs()
        self.zoomToCells()
        self.updateItemsMousePos()
        self.updateObjectCounts()

        self.apply_tools_on_new_frame()

    def next_pos(self):
        self.store_data(debug=True, autosave=False)
        prev_pos_i = self.pos_i
        if self.pos_i < self.num_pos - 1:
            self.pos_i += 1
            self.updateSegmDataAutoSaveWorker()
        else:
            self.logger.info("You reached last position.")
            self.pos_i = 0
        self.updatePos()

    def onZsliceSpinboxValueChange(self, value):
        self.zSliceScrollBar.setSliderPosition(value - 1)

    def prevActionTriggered(self):
        if self.isNavigateActionOnNextFrame():
            self.rightImageFramesScrollbar.setValue(
                self.rightImageFramesScrollbar.value() - 1
            )
            return

        stepSubAction = QAbstractSlider.SliderAction.SliderSingleStepSub
        if self.zKeptDown or self.zSliceCheckbox.isChecked():
            self.zSliceScrollBar.triggerAction(stepSubAction)
        else:
            self.navigateScrollBar.triggerAction(stepSubAction)

    def prev_cb(self):
        if self.isSnapshot:
            self.prev_pos()
        else:
            self.prev_frame()
        if self.curvToolButton.isChecked():
            self.curvTool_cb(True)

        self.updatePropsWidget("")

    def prev_frame(self):
        posData = self.data[self.pos_i]
        if posData.frame_i <= 0:
            msg = "You reached the first frame!"
            self.logger.info(msg)
            self.titleLabel.setText(msg, color=self.titleColor)
            return

        # Store data for current frame
        mode = str(self.modeComboBox.currentText())
        if mode != "Viewer":
            self.store_data(debug=False)

        self.removeAlldelROIsCurrentFrame()
        self.askLineageTreeChanges()
        posData.frame_i -= 1
        _, never_visited = self.get_data()

        if mode == "Segmentation and Tracking" or self.isSnapshot:
            self.addExistingDelROIs()

        self.resetExpandLabel()
        self.updatePreprocessPreview()
        self.updateCombineChannelsPreview()
        self.postProcessing()
        self.tracking()
        self.whitelistPropagateIDs(update_lab=True)
        self.updateAllImages()
        self.updateScrollbars()
        self.updateHighlightedAxis()
        self.zoomToCells()
        self.initGhostObject()
        self.updateViewerWindow()
        self.updateItemsMousePos()
        self.updateObjectCounts()

    def prev_pos(self):
        self.store_data(debug=False, autosave=False)
        prev_pos_i = self.pos_i
        if self.pos_i > 0:
            self.pos_i -= 1
            self.updateSegmDataAutoSaveWorker()
        else:
            self.logger.info("You reached first position.")
            self.pos_i = self.num_pos - 1
        self.updatePos()

    def reInitLastSegmFrame(
        self, checked=True, from_frame_i=None, updateImages=True, force=False
    ):
        if not force:
            cancel = self.warnReinitLastSegmFrame()
            if cancel:
                self.logger.info("Re-initialization of last validated frame cancelled.")
                return

        posData = self.data[self.pos_i]
        if from_frame_i is None:
            from_frame_i = posData.frame_i

        self.lastFrameRanOnFirstVisitTools = posData.frame_i

        self.updateLastCheckedFrameWidgets(from_frame_i)
        posData.last_tracked_i = from_frame_i
        self.navigateScrollBar.setMaximum(from_frame_i + 1)
        self.navSpinBox.setMaximum(from_frame_i + 1)
        # self.navigateScrollBar.setMinimum(1)

        # posData.tracked_lost_centroids[from_frame_i-1] = set()
        for i in range(from_frame_i, posData.SizeT):
            if posData.allData_li[i]["labels"] is None:
                break

            posData.segm_data[i] = posData.allData_li[i]["labels"]
            posData.allData_li[i] = myutils.get_empty_stored_data_dict()

            posData.tracked_lost_centroids[i] = set()
            posData.acdcTracker2stepsAnnotInfo.pop(i, None)

        if posData.acdc_df is not None:
            frames = posData.acdc_df.index.get_level_values(0)
            if from_frame_i in frames:
                posData.acdc_df = posData.acdc_df.loc[:from_frame_i]

        self.removeAlldelROIsCurrentFrame()

        if not updateImages:
            return

        self.updateAllImages()

    def resetAcceptedLostIDs(self, from_frame_i=None):
        posData = self.data[self.pos_i]
        if from_frame_i is None:
            from_frame_i = posData.frame_i

        posData.tracked_lost_centroids[from_frame_i - 1] = set()
        for i in range(from_frame_i, posData.SizeT):
            posData.tracked_lost_centroids[i] = set()

    def resetNavigateFramesScrollbar(self, frame_i=None):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i

        self.navigateScrollBar.setValueNoSignal(frame_i + 1)

    def resetNavigateScrollbar(self):
        try:
            self.navigateScrollBar.blockSignals(True)
            self.navigateScrollBar.actionTriggered.disconnect()
            self.navigateScrollBar.sliderReleased.disconnect()
            self.navigateScrollBar.sliderMoved.disconnect()
            # self.navigateScrollBar.valueChanged.disconnect()
            self.navigateScrollBar.setSliderPosition(self.navSpinBox.value())
        except Exception as e:
            if "disconnect()" not in str(e):
                printl(e)
            pass

        self.navigateScrollBar.blockSignals(False)
        self.navigateScrollBar.actionTriggered.connect(
            self.framesScrollBarActionTriggered
        )
        self.navigateScrollBar.sliderReleased.connect(self.framesScrollBarReleased)
        self.navigateScrollBar.sliderMoved.connect(self.framesScrollBarMoved)

    def rightImageFramesScrollbarValueChanged(self, value):
        img = self.nextFrameImage(current_frame_i=value - 2)
        self.img1.linkedImageItem.frame_i = value
        self.img1.linkedImageItem.setImage(img)

    def setFrameNavigationDisabled(self, disable: bool, why: str):
        """Disables the frame navigation buttons and scrollbar.
        This is used when the user is not allowed to navigate through frames
        Call again to unlock it again. Also sets tooltips to inform the user

        Parameters
        ----------
        disable : bool
            if the navigation should be disabled
        why : str
            the reason for disabeling the navigation.
        """

        if disable:
            self.whyNavigateDisabled.add(why)
        else:
            try:
                self.whyNavigateDisabled.remove(why)
            except KeyError:
                pass

        if len(self.whyNavigateDisabled) == 0:
            disable = False
        else:
            disable = True

        # Apply the disable/enable state
        self.prevAction.setDisabled(disable)
        self.nextAction.setDisabled(disable)
        self.navigateScrollBar.setDisabled(disable)

        # Set appropriate tooltip
        if not disable:
            self.navigateScrollBar.setToolTip(
                "NOTE: The maximum frame number that can be visualized with this "
                "scrollbar\n"
                "is the last visited frame with the selected mode\n"
                '(see "Mode" selector on the top-right).\n\n'
                "If the scrollbar does not move it means that you never visited\n"
                "any frame with current mode.\n\n"
                'Note that the "Viewer" mode allows you to scroll ALL frames.'
            )
            return

        txt = f"Frame navigation disabled: {self.whyNavigateDisabled}"
        self.logger.info(txt)
        self.navigateScrollBar.setToolTip(txt)

    def setNavigateScrollBarMaximum(self):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == "Segmentation and Tracking":
            if posData.last_tracked_i is not None:
                if posData.frame_i > posData.last_tracked_i:
                    self.navigateScrollBar.setMaximum(posData.frame_i + 1)
                    self.navSpinBox.setMaximum(posData.frame_i + 1)
                else:
                    self.navigateScrollBar.setMaximum(posData.last_tracked_i + 1)
                    self.navSpinBox.setMaximum(posData.last_tracked_i + 1)
            else:
                self.navigateScrollBar.setMaximum(posData.frame_i + 1)
                self.navSpinBox.setMaximum(posData.frame_i + 1)

            self.updateLastCheckedFrameWidgets(self.navSpinBox.maximum() - 1)
        elif mode == "Cell cycle analysis":
            if posData.frame_i > self.last_cca_frame_i:
                self.navigateScrollBar.setMaximum(posData.frame_i + 1)
                self.navSpinBox.setMaximum(posData.frame_i + 1)
            else:
                self.navigateScrollBar.setMaximum(self.last_cca_frame_i + 1)
                self.navSpinBox.setMaximum(self.last_cca_frame_i + 1)
            self.lastTrackedFrameLabel.setText(
                f"Last cc annot. frame n. = {self.navSpinBox.maximum()}"
            )
        elif mode == "Normal division: Lineage tree":
            if self.lineage_tree is None:
                self.navigateScrollBar.setMaximum(posData.frame_i + 1)
                self.navSpinBox.setMaximum(posData.frame_i + 1)
            else:
                if self.lineage_tree.frames_for_dfs:
                    i = max(self.lineage_tree.frames_for_dfs)
                else:
                    i = 0
                self.navigateScrollBar.setMaximum(i + 1)
                self.navSpinBox.setMaximum(i + 1)

    def setSwitchViewedPlaneDisabled(self, disabled):
        posData = self.data[self.pos_i]
        if posData.SizeZ == 1:
            return

        self.switchPlaneCombobox.setDisabled(disabled)
        if disabled:
            self.switchPlaneCombobox.setCurrentIndex(0)

    def setViewRangeSwitchPlane(self, previousPlane):
        self.autoRange()
        QTimer.singleShot(100, partial(self._setViewRangeSwitchPlane, previousPlane))

    def setZprojDisabled(self, disabled, storePrevState=False):
        self.combineChannelsAction.setDisabled(disabled)
        for action in self.editToolBar.actions():
            button = self.editToolBar.widgetForAction(action)
            if button == self.eraserButton:
                continue

            if button in self.toolsActiveInProj3Dsegm:
                continue

            try:
                tooltip = button.toolTip()
                prefix = "WARNING: Disabled due to projection mode\n\n"
                if disabled:
                    if not tooltip.startswith(prefix):
                        button.setToolTip(prefix + tooltip)
                else:
                    if tooltip.startswith(prefix):
                        button.setToolTip(tooltip[len(prefix) :])
            except:
                pass
            action.setDisabled(disabled)
            try:
                button.setChecked(False)
            except Exception as err:
                pass

    def switchViewedPlane(self, previousPlane, currentPlane):
        posData = self.data[self.pos_i]
        self.xRangePrev, self.yRangePrev = self.ax1.viewRange()
        self.zSlicePrev = self.zSliceScrollBar.sliderPosition()

        self.zProjComboBox.setCurrentText("single z-slice")
        depthAxes = self.switchPlaneCombobox.depthAxes()
        self.onEscape()
        self.initDelRoiLab()
        if depthAxes != "z":
            # Disable projections on plane that is not xy
            self.zProjComboBox.setCurrentText("single z-slice")
            self.zProjComboBox.setDisabled(True)

            # Clear annotations
            self.clearAllItems()
            self.setHighlightID(False)

            # Disable annotations on a plane that is not yz
            self.setDrawNothingAnnotations()
            self.setDisabledAnnotCheckBoxesLeft(True)
            self.setDisabledAnnotCheckBoxesRight(True)
            self.setEnabledAnnotCheckBoxesLeftZdepthAxes()
            self.overlayButtonPrevState = self.overlayButton.isChecked()
            self.overlayButton.setChecked(False)
            self.overlayButton.setDisabled(True)
        else:
            self.zProjComboBox.setDisabled(False)
            self.restoreAnnotationsOptions()
            self.setDisabledAnnotCheckBoxesLeft(False)
            self.setDisabledAnnotCheckBoxesRight(False)
            self.overlayButton.setDisabled(False)
            if self.overlayButtonPrevState:
                self.overlayButton.setChecked(self.overlayButtonPrevState)
            self.updateZsliceScrollbar(posData.frame_i)

        SizeY, SizeX = posData.img_data[posData.frame_i].shape[-2:]

        if depthAxes != "z" and self.isSnapshot:
            # Disable editing when the plane is not xy
            self.disableEditingViewPlaneNotXY()
        elif self.isSnapshot:
            # Re-enable editing in snapshot mode when the plane is xy
            self.setEnabledSnapshotMode()

        if depthAxes == "z":
            maxSliceNum = posData.SizeZ
        elif depthAxes == "y":
            maxSliceNum = SizeY
        else:
            maxSliceNum = SizeX

        maxSliceText = f"/{maxSliceNum}"
        self.SizeZlabel.setText(maxSliceText)
        self.zSliceCheckbox.setText(f"{depthAxes}-slice")
        self.zSliceScrollBar.setMaximum(maxSliceNum - 1)
        self.zSliceSpinbox.setMaximum(maxSliceNum)

        self.initContoursImage()
        self.updateAllImages()
        QTimer.singleShot(200, partial(self.setViewRangeSwitchPlane, previousPlane))

    def updateFramePosLabel(self):
        if self.isSnapshot:
            posData = self.data[self.pos_i]
            self.navSpinBox.setValueNoEmit(self.pos_i + 1)
        else:
            posData = self.data[0]
            self.navSpinBox.setValueNoEmit(posData.frame_i + 1)

    def updateItemsMousePos(self):
        if self.brushButton.isChecked():
            self.updateBrushCursor(self.xHoverImg, self.yHoverImg)

        if self.eraserButton.isChecked():
            self.updateEraserCursor(self.xHoverImg, self.yHoverImg)

    def updateOverlayZproj(self, how):
        if how.find("max") != -1 or how == "same as above":
            self.overlay_z_label.setDisabled(True)
            self.zSliceOverlay_SB.setDisabled(True)
        else:
            self.overlay_z_label.setDisabled(False)
            self.zSliceOverlay_SB.setDisabled(False)
        self.setOverlayImages()

    def updateOverlayZslice(self, z):
        self.setOverlayImages()

    def updatePos(self):
        self.clearUndoQueue()
        self.setStatusBarLabel()
        self.checkManageVersions()
        self.removeAlldelROIsCurrentFrame()
        self.resetManualBackgroundItems()
        proceed_cca, never_visited = self.get_data(debug=False)
        self.pointsLayerLoadedDfsToData()
        self.flushDirtyPointsLayersAutosave()
        self.initContoursImage()
        self.initDelRoiLab()
        self.initTextAnnot()
        self.postProcessing()
        self.updateScrollbars()
        self.updatePreprocessPreview()
        self.updateCombineChannelsPreview()
        self.updateAllImages()
        self.computeSegm()
        self.zoomOut()
        self.restartZoomAutoPilot()
        self.initManualBackgroundObject()
        self.updateObjectCounts()
        self.updateItemsMousePos()

    def updateScrollbars(self):
        self.updateItemsMousePos()
        self.updateFramePosLabel()
        posData = self.data[self.pos_i]
        navPos = self.pos_i + 1 if self.isSnapshot else posData.frame_i + 1
        self.navigateScrollBar.setSliderPosition(navPos)
        if posData.SizeZ > 1:
            self.updateZsliceScrollbar(posData.frame_i)
            idx = (posData.filename, posData.frame_i)
            self.zSliceScrollBar.setMaximum(posData.SizeZ - 1)
            self.zSliceSpinbox.setMaximum(posData.SizeZ)
            self.SizeZlabel.setText(f"/{posData.SizeZ}")

    def updateViewerWindow(self):
        if self.slideshowWin is None:
            return

        if self.slideshowWin.linkWindow is None:
            return

        if not self.slideshowWin.linkWindowCheckbox.isChecked():
            return

        posData = self.data[self.pos_i]
        self.slideshowWin.frame_i = posData.frame_i
        self.slideshowWin.update_img()

    def updateZproj(self, how):
        for p, posData in enumerate(self.data[self.pos_i :]):
            if self.zProjLockViewButton.isChecked():
                idx = [(posData.filename, frame_i) for frame_i in range(posData.SizeT)]
            else:
                idx = [(posData.filename, posData.frame_i)]
            posData.segmInfo_df.loc[idx, "which_z_proj_gui"] = how
            posData.segmInfo_df.to_csv(posData.segmInfo_df_csv_path)

        posData = self.data[self.pos_i]
        if how == "single z-slice":
            self.zSliceScrollBar.setDisabled(False)
            self.zSliceSpinbox.setDisabled(False)
            self.zSliceCheckbox.setDisabled(False)
            self.setZprojDisabled(False)
            self.update_z_slice(self.zSliceScrollBar.sliderPosition())
        else:
            self.zSliceScrollBar.setDisabled(True)
            self.zSliceSpinbox.setDisabled(True)
            self.zSliceCheckbox.setDisabled(True)
            self.setZprojDisabled(self.isSegm3D)
            self.updateAllImages()

    def update_z_slice(self, z):
        posData = self.data[self.pos_i]
        if self.switchPlaneCombobox.depthAxes() == "z":
            if self.zProjLockViewButton.isChecked():
                idx = [(posData.filename, frame_i) for frame_i in range(posData.SizeT)]
            else:
                idx = [
                    (posData.filename, frame_i)
                    for frame_i in range(posData.frame_i, posData.SizeT)
                ]
            posData.segmInfo_df.loc[idx, "z_slice_used_gui"] = z

        self.updatePreprocessPreview()
        self.updateCombineChannelsPreview()
        self.highlightedID = self.getHighlightedID()
        self.updateAllImages(
            computePointsLayers=False, computeContours=False, updateLookuptable=True
        )
        self.updateItemsMousePos()
        if self.isSegm3D:
            self.updateObjectCounts()

    def warnLostObjects(self, do_warn=True):
        if not do_warn:
            return True

        if not self.warnLostCellsAction.isChecked():
            return True

        mode = str(self.modeComboBox.currentText())
        if not mode == "Segmentation and Tracking":
            return True

        posData = self.data[self.pos_i]
        if not posData.lost_IDs:
            return True

        frame_i = posData.frame_i
        try:
            accepted_lost_IDs = posData.accepted_lost_IDs.get(frame_i, [])
            already_accepted_lost = Counter(accepted_lost_IDs) == Counter(
                posData.lost_IDs
            )
        except AttributeError as err:
            already_accepted_lost = False

        if already_accepted_lost:
            return True

        self.nextAction.setDisabled(True)
        self.prevAction.setDisabled(True)
        self.navigateScrollBar.setDisabled(True)

        msg = widgets.myMessageBox()
        warn_msg = html_utils.paragraph(
            "Current frame (compared to previous frame) "
            "has <b>lost the following cells</b>:<br><br>"
            f"{posData.lost_IDs}<br><br>"
            "Are you <b>sure</b> you want to continue?<br>"
        )
        checkBox = QCheckBox("Do not show again")
        noButton, yesButton = msg.warning(
            self, "Lost cells!", warn_msg, buttonsTexts=("No", "Yes"), widgets=checkBox
        )
        doNotWarnLostCells = not checkBox.isChecked()
        self.warnLostCellsAction.setChecked(doNotWarnLostCells)
        if msg.clickedButton == noButton:
            self.nextAction.setDisabled(False)
            self.prevAction.setDisabled(False)
            self.navigateScrollBar.setDisabled(False)
            return False

        self.nextAction.setDisabled(False)
        self.prevAction.setDisabled(False)
        self.navigateScrollBar.setDisabled(False)
        if not hasattr(posData, "accepted_lost_IDs"):
            posData.accepted_lost_IDs = {}
        if frame_i not in posData.accepted_lost_IDs:
            posData.accepted_lost_IDs[frame_i] = []

        posData.accepted_lost_IDs[frame_i].extend(posData.lost_IDs)
        # This section is adding the lost cells to tracked_lost_centroids... TBH I dont know why this wasnt done in the first place
        prev_rp = posData.allData_li[posData.frame_i - 1]["regionprops"]
        prev_IDs_idxs = posData.allData_li[posData.frame_i - 1]["IDs_idxs"]
        accepted_lost_centroids = {
            tuple(int(val) for val in prev_rp[prev_IDs_idxs[ID]].centroid)
            for ID in posData.lost_IDs
        }
        try:
            posData.tracked_lost_centroids[frame_i] = posData.tracked_lost_centroids[
                frame_i
            ] | (accepted_lost_centroids)
        except KeyError:
            posData.tracked_lost_centroids[frame_i] = accepted_lost_centroids
        return True

    def warnReinitLastSegmFrame(self):
        current_frame_n = self.navigateScrollBar.value()
        msg = widgets.myMessageBox()
        txt = html_utils.paragraph(f"""
            Are you sure you want to <b>re-initialize the last visited and 
            validated</b> frame to number {current_frame_n}?<br><br>
            WARNING: If you save, <b>all annotations after frame number 
            {current_frame_n} will be lost!</b> 
        """)
        msg.warning(
            self,
            "WARNING: Potential loss of data",
            txt,
            buttonsTexts=("Cancel", "Yes, I am sure"),
        )
        return msg.cancel

    def zSliceScrollBarActionTriggered(self, action):
        singleMove = (
            action == SliderSingleStepAdd
            or action == SliderSingleStepSub
            or action == SliderPageStepAdd
            or action == SliderPageStepSub
        )
        if singleMove:
            self.update_z_slice(self.zSliceScrollBar.sliderPosition())
        elif action == SliderMove:
            if self.zSliceScrollBarStartedMoving and self.isSegm3D:
                self.clearAx1Items(onlyHideText=True)
                self.clearAx2Items(onlyHideText=True)
            posData = self.data[self.pos_i]
            idx = (posData.filename, posData.frame_i)
            z = self.zSliceScrollBar.sliderPosition()
            if self.switchPlaneCombobox.depthAxes() == "z":
                posData.segmInfo_df.at[idx, "z_slice_used_gui"] = z
            self.zSliceSpinbox.setValueNoEmit(z + 1)
            img = self._getImageupdateAllImages(None)
            self.img1.setCurrentZsliceIndex(z)
            self.img1.setImage(
                img,
                next_frame_image=self.nextFrameImage(),
                scrollbar_value=posData.frame_i + 2,
            )
            try:
                self.setOverlayImages()
            except Exception as err:
                pass

            if self.labelsGrad.showLabelsImgAction.isChecked():
                self.img2.setImage(posData.lab, z=z, autoLevels=False)
            self.updateViewerWindow()
            self.setTextAnnotZsliceScrolling()
            self.setGraphicalAnnotZsliceScrolling()
            self.setOverlayLabelsItems()
            self.drawPointsLayers(computePointsLayers=False)
            self.zSliceScrollBarStartedMoving = False
            self.highlightSearchedID(self.highlightedID, force=True)

    def zSliceScrollBarReleased(self):
        self.clearTempBrushImage()
        self.zSliceScrollBarStartedMoving = True
        self.update_z_slice(self.zSliceScrollBar.sliderPosition())

    def storeViewRange(self):
        if not hasattr(self, "isRangeReset"):
            return

        if not self.isRangeReset:
            return
        self.ax1_viewRange = self.ax1.viewRange()
        self.isRangeReset = False
