"""The image panel of the measurements viewer
(ViewerImagePanelGuiElements): image updates, text annotations, LUTs,
viewer tools, status bar label, CCA table, objects dock and image mouse
events. Spec: measurements_viewer/__init__.py.
"""
import numpy as np
from qtpy.QtCore import Qt, QTimer, QPoint
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import QAction, QActionGroup
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import load, myutils, transformation, qutils
from cellacdc.config import STANDARD_MOUSE_BUTTONS
from cellacdc.measurements_viewer_utils import (
    get_channels_and_segm_endnames, loaded_data_status
)


class ViewerImagePanelGuiElements:
    """The image panel: updateImages, text annotation options and LUTs,
    viewer tools after loading, the status bar label, the CCA table
    window, the "Cell-ACDC objects" dock, image mouse events and no-op
    stand-ins for main-GUI hooks called by the shared mixins. The
    overrides that call super() (invertBw, addScaleBar, addTimestamp,
    annotateAllObjectTracks, ...) stay in
    measurements_viewer/main_window/window.py.
    """

    def annotatedAxesIndices(self):
        # The viewer has only the left image (no right image like in guiWin)
        return (0,)

    def setDoNotAnnotate(self, checked):
        # Same as `gui.guiWin.setDoNotAnnotate` but only for the left image
        self.annotDoNotAnntoateCheckbox(0).setChecked(checked)

    def launch3dViewerActionToggled(self, checked):
        self.launch3dViewerButton.setChecked(checked)

    def normaliseIntensitiesActionTriggered(self, action):
        how = action.text()
        self.df_settings.at['how_normIntensities', 'value'] = how
        self.df_settings.to_csv(self.settings_csv_path)
        if not self.isDataLoaded:
            return
        self.updateAllImages()
        self.updateImageValueFormatter()

    def uncheckViewerTools(self):
        """Uncheck the viewer tools (with signals, so that their slots run
        their teardown). Called by `setData` while the previous data is
        still loaded. As the main GUI's `reInitGui`, this also removes the
        scale bar and the timestamp: they keep the pixel size and time
        increment of the data they were added to. The 3D viewer closes
        about 300 ms later (`launch3dViewerToggled` defers
        `_launch3dViewer`), which is harmless: closing it does not read
        `posData`."""
        # Their properties dialogs are not modal: close one still open (an
        # `addScaleBar(True)` / `addTimestamp(True)` waiting in it then
        # returns, as the item was removed meanwhile)
        for dialog in (self.scaleBarDialog, self.timestampDialog):
            if dialog is not None:
                dialog.close()
        for button in (
                self.overlayButton, self.overlayLabelsButton,
                self.rulerButton, self.zoomRectButton, self.countObjsButton,
                self.slideshowButton, self.addScaleBarAction,
                self.addTimestampAction
            ):
            if button.isChecked():
                button.setChecked(False)
        if self.launch3dViewerAction.isChecked():
            self.launch3dViewerAction.setChecked(False)

    def resetViewerToolsWidgets(self):
        """Uncheck the tools and remove the overlay items of previously
        loaded data. The tools are normally already unchecked by `setData`
        while the previous data was still loaded (`uncheckViewerTools`)."""
        self.uncheckViewerTools()
        self.overlayToolbar.setVisible(False)
        for widget in self.overlayZsliceWidgets():
            widget.hide()
        
        self.removeOverlayItems()
        # Re-added by `gui_createOverlayItems`
        self.ax1.removeItem(self.rgbaImg1)
        for items in self.overlayLayersItems.values():
            imageItem, lutItem, alphaSB = items[:3]
            self.ax1.removeItem(imageItem)
            for widget in (alphaSB, alphaSB.label):
                self.bottomLeftLayout.removeWidget(widget)
                widget.deleteLater()
        alphaScrollbar = getattr(self.img1, 'alphaScrollbar', None)
        if alphaScrollbar is not None:
            for widget in (alphaScrollbar, alphaScrollbar.label):
                self.bottomLeftLayout.removeWidget(widget)
                widget.deleteLater()
            self.img1.alphaScrollbar = None
        self.alphaScrollbarRow = self.firstAlphaScrollbarRow
        
        for items in self.overlayLabelsItems.values():
            imageItem, contoursItem, gradItem = items
            self.ax1.removeItem(imageItem)
            self.ax1.removeItem(contoursItem)
            self.mainLayout.removeWidget(gradItem)
            gradItem.deleteLater()
        if hasattr(self, 'overlayLabelsContoursImages'):
            del self.overlayLabelsContoursImages
        
        self.overlayLayersItems = {}
        self.overlayLabelsItems = {}
        self.drawModeOverlayLabelsChannels = {}
        self.fluoDataChNameActions = []
        self.checkedOverlayChannels = set()
        self.overlayButton.setStyleSheet('')
        self.lutItemsLayout.addItem(self.imgGrad, row=0, col=0)
        
        if hasattr(self, 'zoomRectItem'):
            self.ax1.removeItem(self.zoomRectItem)
        self.tempSegmentON = False
        self.isMouseDragImg1 = False
        self.ax1_rulerPlotItem.setData([], [])
        self.ax1_rulerAnchorsItem.setData([], [])
        self.clearCursors()

    def initViewerTools(self):
        """Initialize the overlay of other channels and segmentation files 
        as in `gui.guiWin` after loading data"""
        self.resetViewerToolsWidgets()
        
        posData = self.posData()
        self.isSnapshot = posData.SizeT == 1
        try:
            self.ch_names, _ = get_channels_and_segm_endnames(
                posData.images_path
            )
        except Exception as err:
            self.ch_names = [self.user_ch_name]
        if self.user_ch_name not in self.ch_names:
            self.ch_names = [self.user_ch_name, *self.ch_names]
        
        existingSegmEndNames = set()
        for _posData in self.data:
            try:
                segm_files = load.get_segm_files(_posData.images_path)
                existingSegmEndNames.update(
                    load.get_endnames(_posData.basename, segm_files)
                )
            except Exception as err:
                # posData without files on disk --> no overlay labels
                pass
        self.existingSegmEndNames = existingSegmEndNames
        
        for _posData in self.data:
            _posData.ol_data_dict = {}
            _posData.ol_data = None
            _posData.ol_labels_data = None
            for ch in self.ch_names:
                _posData.lutLevels.setdefault(ch, {})
        
        self.setOverlayColors()
        self.createOverlayContextMenu()
        self.userChNameAction = QAction(self)
        self.userChNameAction.setCheckable(True)
        self.userChNameAction.setText(self.user_ch_name)
        self.gui_createOverlayColors()
        self.gui_createOverlayItems()
        
        self.createOverlayLabelsContextMenu(existingSegmEndNames)
        self.createOverlayLabelsItems(existingSegmEndNames)
        self.gui_addOverlayLayerItems()
        
        self.chNamesQActionGroup = QActionGroup(self)
        self.chNamesQActionGroup.addAction(self.userChNameAction)
        self.userChNameAction.setChecked(True)
        self.addActionsLutItemContextMenu(self.imgGrad)
        # Scrollbar for opacity of img1 (when overlaying)
        self.img1.alphaScrollbar = self.addAlphaScrollbar(
            self.user_ch_name, self.img1
        )
        
        self.guiTabControl.addChannels([self.user_ch_name])
        self.showPropsDockButton.setDisabled(False)
        self.setPermanentGreedyCmapPreferences()

    def initViewerToolsAfterLoading(self):
        self.updateImageValueFormatter()
        self.initPixelSizePropsDockWidget()
        # First-load `invertBw`, then the LUT saved for the channel
        self.restoreImageChannelLutAfterLoading()
        self.gui_createZoomRectItem()

    def gui_createAxesTextAnnotItems(self, allIDs):
        if 0 in self.textAnnot:
            self.textAnnot[0].removeFromPlotItem(self.ax1)
        self.gui_createTextAnnotItems(allIDs)
        self.gui_setTextAnnotColors()
        self.textAnnot[0].addToPlotItem(self.ax1)
        self.initTextAnnot()
        self.syncTextAnnotOptions()

    def syncTextAnnotOptions(self):
        """Set the text annotations flags from the annotation checkboxes"""
        for ax in self.annotatedAxesIndices():
            if ax not in self.textAnnot:
                continue
            textAnnot = self.textAnnot[ax]
            textAnnot.setLabelAnnot(self.annotIDsCheckbox(ax).isChecked())
            textAnnot.setCcaAnnot(self.isCellCycleInfoChecked(ax))
            textAnnot.setLineageAnnot(
                self.annotLineageInfoCheckbox(ax).isChecked()
            )
            textAnnot.setScaling(self.scalingLabelsAction.isChecked())

    def restoreViewerAnnotationOptions(self):
        """Restore the saved annotation options without triggering the 
        checkboxes callbacks (data might not be loaded yet)."""
        checkboxes = [
            checkbox for checkbox in self.annotOptionsCheckboxes[0].values()
            if hasattr(checkbox, 'setChecked')
        ]
        for checkbox in checkboxes:
            checkbox.blockSignals(True)
        try:
            self.restoreSavedAnnotationOptions()
        finally:
            for checkbox in checkboxes:
                checkbox.blockSignals(False)
        self.syncTextAnnotOptions()

    def resetFocus(self):
        self.setFocus()

    def setTitleText(self, *args, **kwargs):
        # The viewer does not have the title label of the main GUI
        pass

    def setLookingGoodTitle(self):
        pass

    def changeFontSize(self):
        fontSize = self.fontSizeSpinBox.value()
        if fontSize == self.fontSize:
            return
        
        self.fontSize = fontSize
        self._set_setting_value('fontSize', self.fontSize)
        if not self.isDataLoaded:
            return
        
        for ax in self.annotatedAxesIndices():
            self.textAnnot[ax].changeFontSize(self.fontSize)
        self.updateImages()

    def changeTextResolution(self):
        if not self.isDataLoaded:
            return
        
        mode = 'high' if self.highLowResAction.isChecked() else 'low'
        posData = self.posData()
        img_shape = self.img1.image.shape[:2]
        scalingMode = self.scalingLabelsAction.isChecked()
        for ax in self.annotatedAxesIndices():
            self.textAnnot[ax].changeResolution(
                mode, posData.allIDs, self.ax1, img_shape, 
                scalingMode=scalingMode
            )
        self.updateImages()

    def highLowResToggled(self, checked=True):
        self._set_setting_value('highLowRes', self.highLowResAction.isChecked())
        self.changeTextResolution()

    def scalingLabelsToggled(self, checked=True):
        self._set_setting_value(
            'scalingLabels', self.scalingLabelsAction.isChecked()
        )
        if not self.isDataLoaded:
            return
        for annot in self.textAnnot.values():
            annot.setScaling(self.scalingLabelsAction.isChecked())

    def imgGradLUTfinished_cb(self):
        if self.isSettingChannelLutsFromCode():
            return
        self.df_settings = self.imgGrad.saveState(self.df_settings)
        # LUT of the displayed channel (`gui_navigation.LutGuiElements`)
        self.saveImageChannelLut()
        self.df_settings.to_csv(self.settings_csv_path)

    def initLut(self):
        """Initialize LUT levels as in `gui.guiWin` after loading data. The
        LUT (gradient) saved for the channel is restored at the end of the
        load (`initViewerToolsAfterLoading`)."""
        self.isDataLoaded = True
        posData = self.posData()
        self.resetImageChannelLut()
        self.user_ch_name = posData.user_ch_name
        self.rescaleIntensChannelHowMapper.setdefault(
            self.user_ch_name, 'Rescale each 2D image'
        )
        for _posData in self.data:
            if not hasattr(_posData, 'lutLevels'):
                _posData.lutLevels = {self.user_ch_name: {}}
        self.imgGrad.setAxisLabel(self.user_ch_name)
        self.img1.preComputedMinMaxValues(self.data)
        self.imgGrad.rescaleAcrossZstackAction.setDisabled(
            posData.SizeZ == 1
        )
        self.initImgGradRescaleIntensitiesHowPreference()
        self.rescaleIntensitiesLut(setImage=False)

    def setStatusBarLabel(self, log=True):
        """Show what is displayed in the status bar label, as the main GUI's
        `gui.guiWin.setStatusBarLabel`: experiment and Position folder,
        basename, displayed and overlay channels, segmentation file and
        configuration file name, with the full Images folder and
        configuration paths in its tool tip (`loaded_data_status`; long
        paths broken after separators, other lines by
        `qutils.wrap_tooltip`). Empty
        without data. Called whenever one of them changes; `log` also logs
        the text (once per load or position change)."""
        if not self.data:
            self.statusBarLabel.setText('')
            self.statusBarLabel.setToolTip('')
            return
        text, tooltip = loaded_data_status(
            self.posData(), pos_i=self.pos_i, num_positions=len(self.data),
            channel=self.user_ch_name,
            overlay_channels=self.activeOverlayChannels(),
            configuration_filepath=self.configurationFilepath
        )
        self.statusBarLabel.setText(text)
        self.statusBarLabel.setToolTip(qutils.wrap_tooltip(tooltip))
        if log:
            self.logger.info(text)

    def activeOverlayChannels(self):
        """Overlaid channels (sorted), also while the overlay is turned off
        on a zy/zx plane (`setOverlayEnabledForPlane` restores it on xy)."""
        isOverlayOn = (
            self.overlayButton.isChecked() or bool(self.overlayButtonPrevState)
        )
        if not isOverlayOn:
            return []
        return sorted(self.checkedOverlayChannels)

    def updateCcaTableWin(self):
        """Show the cell cycle annotations of the current frame in the open
        `ccaTableWin`. Frames without annotations show an empty table and
        only the zoomed IDs that are annotated are shown."""
        posData = self.posData()
        cca_df = posData.cca_df
        if cca_df is None:
            self.ccaTableWin.updateTable(self.getBaseCca_df(), IDs=None)
            return

        zoomIDs = self.getZoomIDs()
        if zoomIDs is not None:
            zoomIDs = [ID for ID in zoomIDs if ID in cca_df.index]
        self.ccaTableWin.updateTable(cca_df, IDs=zoomIDs)

    def onSigUpdateCcaTableWindow(self, *args):
        if not self.isDataLoaded or self.ccaTableWin is None:
            return
        self.updateCcaTableWin()

    def getImage(self, frame_i=None, raw=False):
        posData = self.posData()
        if frame_i is None:
            frame_i = self.frame_i
        img = posData.img_data[frame_i]
        if posData.SizeZ > 1:
            img = self.get_2Dimg_from_3D(img)
        return img

    def get_2Dimg_from_3D(self, imgData, isLayer0=True, frame_i=None, yx=None):
        # `yx` not None --> only the value at (y, x) of the 2D image (hover)
        img = self.get_2Dimg_from_switched_plane(imgData, yx=yx)
        if img is not None:
            return img

        zProjHow = self.zProjComboBox.currentText()
        z = self.zSliceScrollBar.sliderPosition()
        if not isLayer0:
            zProjHowOverlay = self.zProjOverlay_CB.currentText()
            if zProjHowOverlay != 'same as above':
                zProjHow = zProjHowOverlay
                z = self.zSliceOverlay_SB.sliderPosition()
        if yx is not None:
            # Only the z-column at (y, x)
            imgData = imgData[:, yx[0], yx[1]]
        return self.zProjectImage(imgData, zProjHow, z)

    def getHighlightedIDfromPropsDock(self):
        if not self.propsDockWidget.isVisible():
            return 0
        guiTabControl = self.guiTabControl
        doHighlight = (
            guiTabControl.highlightCheckbox.isChecked()
            or guiTabControl.highlightSearchedCheckbox.isChecked()
        )
        if not doHighlight:
            return 0
        return guiTabControl.propsQGBox.idSB.value()

    def syncPropsDockToSelectedID(self):
        """Show the properties of the selected cell in the current frame in
        the "Cell-ACDC objects" dock (if visible), so that the dock and the
        viewer always refer to the same cell."""
        if not self.propsDockWidget.isVisible() or not self.selectedID:
            return
        posData = self.posData()
        if getattr(posData, 'rp', None) is None:
            return

        ID = int(self.selectedID)
        propsQGBox = self.guiTabControl.propsQGBox
        wasUpdating = self._isUpdatingPropsDock
        self._isUpdatingPropsDock = True
        try:
            # Force refresh: the values change with frame/z-slice
            self.currentPropsID = -1
            obj = posData.rp.get_obj_from_ID(ID, warn=False)
            propsQGBox.idSB.setValue(ID)
            if obj is None:
                # No values of another (previously shown) cell next to ID
                self.clearPropsDockValues()
                propsQGBox.notExistingIDLabel.setText(
                    f'Object ID {ID:d} does not exist'
                )
            elif not self.isObjInCurrSlice(obj.bbox):
                # 3D segmentation, single z-slice: the mixin's area of the
                # displayed slice would index outside the object
                self.clearPropsDockValues()
                propsQGBox.notExistingIDLabel.setText(
                    f'Object ID {ID:d} is not in this z-slice'
                )
            else:
                self.updatePropsWidget(ID)
        finally:
            self._isUpdatingPropsDock = wasUpdating

    def clearPropsDockValues(self):
        """Empty the value fields of the "Cell-ACDC objects" dock (shown
        when the selected cell has no values in the displayed frame or
        z-slice)."""
        propsQGBox = self.guiTabControl.propsQGBox
        intensMeasurQGBox = self.guiTabControl.intensMeasurQGBox
        fields = (
            propsQGBox.cellAreaPxlSB, propsQGBox.cellAreaUm2DSB,
            propsQGBox.cellVolVoxSB, propsQGBox.cellVolFlDSB,
            propsQGBox.cellVolVox3D_SB, propsQGBox.cellVolFl3D_DSB,
            propsQGBox.elongationDSB, propsQGBox.solidityDSB,
            propsQGBox.additionalPropsCombobox.indicator,
            intensMeasurQGBox.minimumDSB, intensMeasurQGBox.maximumDSB,
            intensMeasurQGBox.meanDSB, intensMeasurQGBox.medianDSB,
            intensMeasurQGBox.additionalMeasCombobox.indicator,
        )
        for field in fields:
            field.setText('')

    def updateImages(self):
        """Update image and annotations as `gui.guiWin.updateAllImages` in
        'Viewer' mode"""
        if not self.data:
            return

        self.get_data()
        self.syncPropsDockToSelectedID()
        self.highlightedID = (
            self.getHighlightedIDfromPropsDock() or self.selectedID or 0
        )
        self.clearAx1Items()
        
        self.rescaleIntensitiesLut(setImage=False)
        self.setImageImg1()
        
        posData = self.posData()
        self.currentLab2D = self.get_2Dlab(posData.lab, force_z=False)
        self.setOverlayImages()
        self.setOverlayLabelsItems()
        self.setOverlaySegmMasks()
        self.updateViewerWindow()
        
        self.setAllTextAnnotations()
        self.setAllContoursImages(compute=False)
        self.drawAllMothBudLines()
        # As in `gui.guiWin.updateAllImages`: lineage tree lines only in
        # frames without cca annotations (they would clear the cca lines)
        if self.isLineageTreeLinesModeActive():
            # Previous frame data required by `drawAllLineageTreeLines`
            self.initLineageTreeFrameData()
            self.drawAllLineageTreeLines()
        
        if self.ccaTableWin is not None:
            self.updateCcaTableWin()
        
        self.highlightSelectedID()
        self.doCustomAnnotation(0)
        self.updateTimestampFrame()
        self.annotateAllObjectTracks()
        posData.visited = True
        self.updateObjectCounts()
        self.updateIDnotPresentLabel()

    def clearAx1Items(self):
        self.labelsLayerImg1.clear()
        self.labelsLayerImg1.setOpacity(self.imgGrad.labelsAlphaSlider.value())
        self.highLightIDLayerImg1.clear()
        self.searchedIDitemLeft.clear()
        self.ax1_contoursImageItem.clear()
        if 0 in self.textAnnot:
            self.textAnnot[0].clear()
        self.ax1_newMothBudLinesItem.setData([], [])
        self.ax1_oldMothBudLinesItem.setData([], [])
        self.ax1_movementAgainstPrevLinesItem.setData([], [])
        self.ax1_movementAgainstPrevScatterItem.setData([], [])
        self.clearObjTracks()

    def highlightSelectedID(self):
        """Highlight the selected ID as the searched ID in `gui.guiWin`"""
        ID = self.highlightedID
        if not ID:
            return
        
        posData = self.posData()
        obj = posData.rp.get_obj_from_ID(ID, warn=False)
        if obj is None:
            return
        
        if not self.isObjInCurrSlice(obj.bbox):
            return
        
        display_rp = self.get2DRP()
        obj = display_rp.get_obj_from_ID(ID, warn=False)
        if obj is None:
            return
        contours = self.getObjContours(
            obj,
            all_external=True,
            include_internal=self.showAllContoursToggle.isChecked()
        )
        for cont in contours:
            self.searchedIDitemLeft.addPoints(cont[:,0]+0.5, cont[:,1]+0.5)

    def updateAllImages(self, *args, **kwargs):
        self.updateImages()

    def setImageImg1(self):
        posData = self.posData()
        self.img1.setCurrentPosIndex(self.pos_i)
        self.img1.setCurrentFrameIndex(posData.frame_i)
        if posData.SizeZ > 1:
            zProjHow = self.zProjComboBox.currentText()
            if self.switchPlaneCombobox.depthAxes() != 'z':
                # Pre-computed min/max values refer to xy planes only
                z = None
            elif zProjHow == 'single z-slice':
                z = self.zSliceScrollBar.sliderPosition()
            else:
                z = zProjHow
            
            self.img1.setCurrentZsliceIndex(z)
        
        self.img1.setImage(self.getImage())

    def updateIDnotPresentLabel(self):
        if self.selectedID is None:
            self.idNotPresentLabel.setText('')
            return
        
        isPresent = np.any(self.getLabels() == self.selectedID)
        if isPresent:
            self.idNotPresentLabel.setText('')
        else:
            self.idNotPresentLabel.setText(
                f'<i>Cell ID {self.selectedID} is not present in frame '
                f'n. {self.frame_i+1}</i>'
            )

    def initLookupTableLab(self):
        self.initLabelsImageItems()

    def updateLookuptable(self, *args, **kwargs):
        self.initLabelsImageItems()

    def clearCursors(self):
        self.ax1_cursor.setData([], [])
        self.update3DViewerCursor(None)

    def connectLeftClickButtons(self):
        # The viewer does not have editing tools that need to be reconnected
        pass

    def disconnectLeftClickButtons(self):
        pass

    def uncheckLeftClickButtons(self, sender):
        for button in self.leftClickButtons:
            if button == sender:
                continue
            if button.isChecked():
                button.setChecked(False)

    def drawPointsLayers(self, **kwargs):
        # Points layers are a main GUI-only feature
        pass

    def addPointsLayersToVolumeViewer(self, *args, **kwargs):
        pass

    def resetCursor(self):
        while self.app.overrideCursor() is not None:
            self.app.restoreOverrideCursor()

    def raiseOnDemandAx1Items(self):
        """Keep the ax1 items created on demand (object tracks, scale bar,
        timestamp) above the overlay channel layers, which are re-added
        (z-value 0) on every load, at the same z-value as the ruler."""
        items = []
        tracksPlotItem = getattr(self, 'tracksPlotItems', {}).get(0)
        if tracksPlotItem is not None:
            items.append(tracksPlotItem)
        scaleBar = getattr(self, 'scaleBar', None)
        if scaleBar is not None:
            items.extend((scaleBar.plotItem, scaleBar.labelItem))
        timestamp = getattr(self, 'timestamp', None)
        if timestamp is not None:
            items.append(timestamp)
        for item in items:
            item.setZValue(0.9)

    def clearComboBoxFocus(self, *args):
        self.setFocus()

    def isScaleBarOrTimestampHighlighted(self):
        for name in ('scaleBar', 'timestamp'):
            item = getattr(self, name, None)
            if item is not None and item.isHighlighted():
                return item

    def resetCountKeyPress(self):
        self.doubleKeyTimeElapsed = True
        self.countKeyPress = 0

    def isHoveringRoiHandle(self, roi):
        # Same as `gui.guiWin.gui_isHoveringRoiHandle`
        if roi is None:
            return False
        try:
            handles = roi.getHandles()
        except Exception:
            return False

        for handle in handles:
            try:
                if handle.currentPen == handle.hoverPen:
                    return True
            except Exception:
                continue
        return False

    def gui_hoverEventImg1(self, event):
        if not self.isDataLoaded:
            return
        
        if event.isExit():
            self.xHoverImg, self.yHoverImg = None, None
            self.wcLabel.setText('')
            self.resetCursor()
            if self.slideshowWin is not None:
                self.slideshowWin.setMirroredCursorPos(None, None)
            self.update3DViewerCursor(None)
            return
        
        x, y = event.pos()
        self.xHoverImg, self.yHoverImg = x, y
        
        if self.slideshowWin is not None:
            self.slideshowWin.setMirroredCursorPos(x, y)

        # Mirrored (depth-aware) cursor in the 3D viewer, if open. Always
        # shown as the slideshow one (guiWin's "Show mirrored cursor on
        # images" setting is a main GUI-only option)
        self.update3DViewerCursor((x, y))
        
        if self.rulerButton.isChecked() and self.tempSegmentON:
            self.drawTempRulerLine(event)
        
        Y, X = self.currentLab2D.shape
        xdata, ydata = int(x), int(y)
        if not myutils.is_in_bounds(xdata, ydata, X, Y):
            self.wcLabel.setText('')
            return
        
        ID = self.currentLab2D[ydata, xdata]
        self.updatePropsWidget(ID, fromHover=True)
        activeToolButton = self.getActiveToolButton()
        txt = self.hoverValuesFormatted(xdata, ydata, activeToolButton, True)
        self.checkHighlightScaleBar(x, y, activeToolButton)
        self.checkHighlightTimestamp(x, y, activeToolButton)
        self.wcLabel.setText(txt)

    def gui_mousePressEventImg1(self, event):
        if not self.isDataLoaded:
            pg.ImageItem.mousePressEvent(self.img1, event)
            return
        
        if event.button() not in STANDARD_MOUSE_BUTTONS:
            # As in guiWin: only bound buttons are forwarded (the window's
            # `QMainWindow.mousePressEvent` does not accept scene events)
            if event.button() in self.mouseBindings.values():
                self.mousePressEvent(event)
            else:
                event.ignore()
            return

        left_click = event.button() == Qt.MouseButton.LeftButton
        right_click = event.button() == Qt.MouseButton.RightButton
        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)

        isZoomRectActive = self.zoomRectButton.isChecked()
        if isZoomRectActive and self.isHoveringRoiHandle(self.zoomRectItem):
            event.ignore()
            return
        
        if left_click and self.rulerButton.isChecked():
            if not self.tempSegmentON:
                self.ax1_rulerAnchorsItem.setData([xdata], [ydata])
                self.tempSegmentON = True
            else:
                modifiers = QGuiApplication.keyboardModifiers()
                ctrl = modifiers == Qt.ControlModifier
                self.tempSegmentON = False
                xxRA, yyRA = self.ax1_rulerAnchorsItem.getData()
                x0, y0 = xxRA[0], yyRA[0]
                if ctrl:
                    x1, y1 = transformation.snap_xy_to_closest_angle(
                        x0, y0, xdata, ydata
                    )
                else:
                    x1, y1 = xdata, ydata
                lengthText = self.getRulerLengthText()
                self.ax1_rulerPlotItem.setData(
                    [x0, x1], [y0, y1], lengthText=lengthText
                )
                self.ax1_rulerAnchorsItem.setData([x0, x1], [y0, y1])
            return
        
        if isZoomRectActive and (left_click or right_click):
            if left_click:
                self.zoomRectItem.setPos((xdata, ydata))
                self.isMouseDragImg1 = True
            else:
                try:
                    xRange, yRange = self.zoomRectItem.getLastRange()
                    self.ax1.setRange(xRange=xRange, yRange=yRange, padding=0)
                except Exception as err:
                    QTimer.singleShot(100, self.autoRange)
            return
        
        highlightedItem = self.isScaleBarOrTimestampHighlighted()
        if left_click and highlightedItem is not None:
            highlightedItem.mousePressed(x, y)
            return
        
        if right_click and self.customAnnotButton is not None:
            self.annotateClickedObject(xdata, ydata)
            return
        
        if right_click:
            screenPos = event.screenPos()
            self.imgGradShowContextMenu(screenPos.x(), screenPos.y())
            return
        
        pg.ImageItem.mousePressEvent(self.img1, event)
        event.ignore()

    def imgGradShowContextMenu(self, x, y):
        # Same as `gui.guiWin.gui_imgGradShowContextMenu`
        highlightedItem = self.isScaleBarOrTimestampHighlighted()
        if highlightedItem is not None:
            highlightedItem.showContextMenu(x, y)
            return
        self.imgGrad.gradient.menu.popup(QPoint(int(x), int(y)))

    def gui_mouseDragEventImg1(self, event):
        if not self.isDataLoaded:
            pg.ImageItem.mouseMoveEvent(self.img1, event)
            return
        
        x, y = event.pos().x(), event.pos().y()
        
        if hasattr(self, 'scaleBar'):
            if self.scaleBarDialog is not None:
                self.scaleBarDialog.locCombobox.setCurrentText('Custom')
            if self.scaleBar.isHighlighted() and self.scaleBar.clicked:
                self.scaleBar.setLocationProperty('custom')
                self.scaleBar.move(x, y)
                return
        
        if hasattr(self, 'timestamp'):
            if self.timestampDialog is not None:
                self.timestampDialog.locCombobox.setCurrentText('Custom')
            if self.timestamp.isHighlighted() and self.timestamp.clicked:
                self.timestamp.setLocationProperty('custom')
                self.timestamp.move(x, y)
                return
        
        if self.isMouseDragImg1 and self.zoomRectButton.isChecked():
            xdata, ydata = int(x), int(y)
            x0, y0 = self.zoomRectItem.pos()
            w, h = (xdata-x0), (ydata-y0)
            self.zoomRectItem.setSize((w, h))
            return
        
        pg.ImageItem.mouseMoveEvent(self.img1, event)

    def gui_mouseReleaseEventImg1(self, event):
        if not self.isDataLoaded:
            pg.ImageItem.mouseReleaseEvent(self.img1, event)
            return
        
        if hasattr(self, 'scaleBar'):
            if self.scaleBar.isHighlighted() and self.scaleBar.clicked:
                self.scaleBar.clicked = False
                return
        
        if hasattr(self, 'timestamp'):
            if self.timestamp.isHighlighted() and self.timestamp.clicked:
                self.timestamp.clicked = False
                return
        
        if self.isMouseDragImg1 and self.zoomRectButton.isChecked():
            self.isMouseDragImg1 = False
            self.zoomRectDone()
            return
        
        pg.ImageItem.mouseReleaseEvent(self.img1, event)

    def gui_mouseClickEventImg1(self, event):
        """Select the clicked cell (pyqtgraph sends click events only if 
        the mouse was not dragged, hence panning does not select)."""
        if not self.isDataLoaded:
            return
        
        if event.button() != Qt.MouseButton.LeftButton:
            return
        
        isToolActive = (
            self.rulerButton.isChecked() 
            or self.zoomRectButton.isChecked()
            or self.isScaleBarOrTimestampHighlighted() is not None
        )
        if isToolActive:
            return
        
        x, y = event.pos().x(), event.pos().y()
        lab2D = self.currentLab2D
        Y, X = lab2D.shape[-2:]
        xdata, ydata = int(x), int(y)
        if not (0 <= xdata < X and 0 <= ydata < Y):
            return
        
        ID = int(lab2D[ydata, xdata])
        if ID == 0:
            return
        
        event.accept()
        self.goToObjectID(ID)
