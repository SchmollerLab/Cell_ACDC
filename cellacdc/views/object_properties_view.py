"""Qt view adapter for object-property workflows."""

from __future__ import annotations

import numpy as np
import skimage.measure
from tqdm import tqdm

from cellacdc import apps, exception_handler, html_utils, widgets
from cellacdc.viewmodels.object_properties_viewmodel import (
    ObjectPropertiesViewModel,
)


class ObjectPropertiesView:
    """Qt-facing adapter around object properties and highlighting."""

    LEGACY_METHODS = (
        'initPixelSizePropsDockWidget',
        'showPropsDockWidget',
        'clearHighlightedID',
        'setAllIDs',
        'countObjectsTimelapse',
        'countObjectsSnapshots',
        'countObjects',
        'updateObjectCounts',
        'countObjectsCb',
        'keepIDs_cb',
        'initKeepObjLabelsLayers',
        'updateTempLayerKeepIDs',
        'highlightLabelID',
        'highlightHoverID',
        'grayOutHighlightedLabels',
        'grayOutOverlaySegm',
        'highlightHoverIDsKeptObj',
        'getHighlightedID',
        'clearHighlightedKeepIDs',
        'setHighlighedIDfromToolbar',
        'highlightSearchedID',
        '_keepObjects',
        'clearHighlightedText',
        'removeHighlightLabelID',
        'updateKeepIDs',
        'applyKeepObjects',
        'get_curr_lab',
        'highlightIDonHoverCheckBoxToggled',
        'highlightSearchedIDcheckBoxToggled',
        'setHighlightID',
        'propsWidgetIDvalueChanged',
        'updatePropsWidget',
    )

    def __init__(self, host, view_model: ObjectPropertiesViewModel):
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

    def initPixelSizePropsDockWidget(self):
        posData = self.data[self.pos_i]
        PhysicalSizeX = posData.PhysicalSizeX
        PhysicalSizeY = posData.PhysicalSizeY
        PhysicalSizeZ = posData.PhysicalSizeZ
        self.guiTabControl.initPixelSize(
            PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ
        )

    def showPropsDockWidget(self, checked=False):
        if self.showPropsDockButton.isExpand:
            self.propsDockWidget.setVisible(False)
            self.setHighlightID(False)
        else:
            self.highlightedID = self.guiTabControl.propsQGBox.idSB.value()
            if self.view_model.should_show_3d_property_controls(
                self.isSegm3D
            ):
                self.guiTabControl.propsQGBox.cellVolVox3D_SB.show()
                self.guiTabControl.propsQGBox.cellVolVox3D_SB.label.show()
                self.guiTabControl.propsQGBox.cellVolFl3D_DSB.show()
                self.guiTabControl.propsQGBox.cellVolFl3D_DSB.label.show()
            else:
                self.guiTabControl.propsQGBox.cellVolVox3D_SB.hide()
                self.guiTabControl.propsQGBox.cellVolVox3D_SB.label.hide()
                self.guiTabControl.propsQGBox.cellVolFl3D_DSB.hide()
                self.guiTabControl.propsQGBox.cellVolFl3D_DSB.label.hide()

            self.propsDockWidget.setVisible(True)
            self.propsDockWidget.setEnabled(True)
        self.updateAllImages()

    def clearHighlightedID(self):
        self.highlightIDToolbar.setVisible(False)

        try:
            self.updateLostContoursImage(ax=0, delROIsIDs=None)
        except Exception as err:
            pass

        if self.highlightedID == 0:
            return

        self.highlightedID = 0
        self.guiTabControl.highlightCheckbox.setChecked(False)
        self.guiTabControl.highlightSearchedCheckbox.setChecked(False)
        self.setHighlightID(False)

    def setAllIDs(self, onlyVisited=False):
        for posData in self.data:
            posData.allIDs = self.view_model.object_counts.collect_all_ids(
                posData,
                only_visited=onlyVisited,
            )

    def countObjectsTimelapse(self):
        if self.countObjsWindow is None:
            activeCategories = self.view_model.timelapse_default_categories()
        else:
            activeCategories = self.countObjsWindow.activeCategories()

        posData = self.data[self.pos_i]
        allCategoryCountMapper = posData.countObjectsInSegmTimelapse(
            activeCategories
        )
        if self.countObjsWindow is None:
            return allCategoryCountMapper

        categoryCountMapper = {}
        for category in activeCategories:
            categoryCountMapper[category] = allCategoryCountMapper[category]

        return categoryCountMapper


    def countObjectsSnapshots(self):
        posData = self.data[self.pos_i]
        if self.countObjsWindow is None:
            activeCategories = self.view_model.snapshot_default_categories(
                is_segm_3d=self.isSegm3D
            )
        else:
            activeCategories = self.countObjsWindow.activeCategories()

        allCategoryCountMapper = self.view_model.object_counts.snapshot_object_counts(
            self.data,
            self.pos_i,
            current_lab_2d=self.currentLab2D,
            include_current_z_slice='In current z-slice' in activeCategories,
        )

        if self.countObjsWindow is None:
            return allCategoryCountMapper

        categoryCountMapper = {}
        for category in activeCategories:
            categoryCountMapper[category] = allCategoryCountMapper[category]

        return categoryCountMapper

    def countObjects(self):
        self.logger.info('Counting objects...')

        posData = self.data[self.pos_i]
        if posData.SizeT > 1:
            return self.countObjectsTimelapse()

        return self.countObjectsSnapshots()


    def updateObjectCounts(self):
        if not self.view_model.should_update_object_counts(
            window_exists=self.countObjsWindow is not None,
            is_visible=(
                self.countObjsWindow.isVisible()
                if self.countObjsWindow is not None else False
            ),
            live_preview_checked=(
                self.countObjsWindow.livePreviewCheckbox.isChecked()
                if self.countObjsWindow is not None else False
            ),
        ):
            return

        categoryCountMapper = self.countObjects()
        self.countObjsWindow.updateCounts(categoryCountMapper)

    def countObjectsCb(self, checked):
        if self.countObjsWindow is None:
            categoryCountMapper = self.countObjects()
            self.countObjsWindow = apps.ObjectCountDialog(
                categoryCountMapper=categoryCountMapper,
                parent=self.host,
                data=self.data
            )
            self.countObjsWindow.sigShowEvent.connect(self.updateObjectCounts)
            self.countObjsWindow.sigUpdateCounts.connect(self.updateObjectCounts)

        if checked:
            self.countObjsWindow.show()
        else:
            self.countObjsWindow.hide()

    def keepIDs_cb(self, checked):
        if checked:
            self.highlightedLab = np.zeros_like(self.currentLab2D)
            if self.annotCcaInfoCheckbox.isChecked():
                self.annotCcaInfoCheckbox.setChecked(False)
                self.annotIDsCheckbox.setChecked(True)
                self.setDrawAnnotComboboxText()
            self.uncheckLeftClickButtons(None)
            self.initKeepObjLabelsLayers()
            self.setAllIDs()
        else:
            # restore items to non-grayed out
            self.clearTempBrushImage()
            alpha = self.imgGrad.labelsAlphaSlider.value()
            self.labelsLayerImg1.setOpacity(alpha)
            self.labelsLayerRightImg.setOpacity(alpha)
            self.ax1_contoursImageItem.setOpacity(1.0)
            self.ax2_contoursImageItem.setOpacity(1.0)
            self.ax1_lostObjImageItem.setOpacity(1.0)
            self.ax2_lostObjImageItem.setOpacity(1.0)
            self.ax1_lostTrackedObjImageItem.setOpacity(1.0)
            self.ax2_lostTrackedObjImageItem.setOpacity(1.0)

        self.keepIDsToolbar.setVisible(checked)
        self.highlightedIDopts = None
        self.keptObjectsIDs = widgets.KeptObjectIDsList(
            self.keptIDsLineEdit, self.keepIDsConfirmAction
        )
        self.updateAllImages()

        # QTimer.singleShot(300, self.autoRange)

    def initKeepObjLabelsLayers(self):
        lut = np.zeros((len(self.lut), 4), dtype=np.uint8)
        lut[:,:-1] = self.lut
        lut[:,-1:] = 255
        lut[0] = [0,0,0,0]
        self.keepIDsTempLayerLeft.setLevels([0, len(lut)])
        self.keepIDsTempLayerLeft.setLookupTable(lut)

    def updateTempLayerKeepIDs(self):
        if not self.keepIDsButton.isChecked():
            return

        keptLab = np.zeros_like(self.currentLab2D)

        posData = self.data[self.pos_i]
        for obj in posData.rp:
            if obj.label not in self.keptObjectsIDs:
                continue

            if not self.isObjVisible(obj.bbox):
                continue

            _slice = self.getObjSlice(obj.slice)
            _objMask = self.getObjImage(obj.image, obj.bbox)

            keptLab[_slice][_objMask] = obj.label

        self.keepIDsTempLayerLeft.setImage(keptLab, autoLevels=False)

    def highlightLabelID(self, ID, ax=0):
        posData = self.data[self.pos_i]
        try:
            obj = posData.rp[posData.IDs_idxs[ID]]
        except KeyError:
            return

        self.textAnnot[ax].highlightObject(obj)

    def highlightHoverID(self, x, y, hoverID=None):
        if hoverID is None:
            try:
                hoverID = self.currentLab2D[int(y), int(x)]
            except IndexError:
                return

        if hoverID == 0:
            return

        posData = self.data[self.pos_i]
        objIdx = posData.IDs_idxs[hoverID]
        obj = posData.rp[objIdx]
        self.goToZsliceSearchedID(obj)
        self.highlightSearchedID(hoverID)

    def grayOutHighlightedLabels(self, nonGrayedIDs=None, alpha=None):
        if nonGrayedIDs is None:
            nonGrayedIDs = set()

        posData = self.data[self.pos_i]
        if alpha is None:
            alpha = self.imgGrad.labelsAlphaSlider.value()

        if not hasattr(self, 'highlightedLab'):
            self.highlightedLab = np.zeros_like(self.currentLab2D)
        else:
            self.highlightedLab[:] = 0

        lut = np.zeros((2, 4), dtype=np.uint8)
        for _obj in posData.rp:
            if not self.isObjVisible(_obj.bbox):
                continue
            if _obj.label not in nonGrayedIDs:
                continue
            _slice = self.getObjSlice(_obj.slice)
            _objMask = self.getObjImage(_obj.image, _obj.bbox)
            self.highlightedLab[_slice][_objMask] = _obj.label
            rgb = self.lut[_obj.label].copy()
            lut[1, :-1] = rgb
            # Set alpha to 0.7
            lut[1, -1] = 178

        return lut

    def grayOutOverlaySegm(self, ax=0):
        if ax == 0:
            how = self.drawIDsContComboBox.currentText()
        else:
            how = self.getAnnotateHowRightImage()

        isOverlaySegmActive = how.find('segm. masks') != -1
        if not isOverlaySegmActive:
            return

        grayedLut = self.grayOutHighlightedLabels()

    def highlightHoverIDsKeptObj(self, x, y, hoverID=None):
        if hoverID is None:
            try:
                hoverID = self.currentLab2D[int(y), int(x)]
            except IndexError:
                return

        self.highlightSearchedID(hoverID, greyOthers=False)

        if hoverID == 0 and self.highlightedID == 0:
            return

        if hoverID == 0 and self.highlightedID != 0:
            self.clearHighlightedKeepIDs()
            for ID in self.keptObjectsIDs:
                self.highlightLabelID(ID)
            return

        posData = self.data[self.pos_i]
        try:
            objIdx = posData.IDs_idxs[hoverID]
        except KeyError as err:
            return

        obj = posData.rp[objIdx]
        self.goToZsliceSearchedID(obj)

        for ID in self.keptObjectsIDs:
            self.highlightLabelID(ID)

    def getHighlightedID(self):
        if self.highlightedID > 0:
            return self.highlightedID

        doHighlight = self.view_model.should_highlight_props_id(
            dock_visible=self.propsDockWidget.isVisible(),
            highlight_checked=self.guiTabControl.highlightCheckbox.isChecked(),
            searched_highlight_checked=(
                self.guiTabControl.highlightSearchedCheckbox.isChecked()
            ),
        )
        if not doHighlight:
            return 0

        return self.guiTabControl.propsQGBox.idSB.value()

    def clearHighlightedKeepIDs(self):
        self.setAllTextAnnotations()
        self.highlightedID = 0
        self.searchedIDitemRight.setData([], [])
        self.searchedIDitemLeft.setData([], [])
        self.highLightIDLayerImg1.clear()
        self.highLightIDLayerRightImage.clear()

    def setHighlighedIDfromToolbar(self, ID: int):
        self.object_search_view.findID(ID=ID)

    def highlightSearchedID(self, ID, force=False, greyOthers=True):
        self.highlightIDToolbar.setIDNoSignals(ID)

        if ID == 0:
            self.highlightIDToolbar.setVisible(False)
            return

        if ID == self.highlightedID and not force:
            return

        doHighlight = self.view_model.should_highlight_props_id(
            dock_visible=self.propsDockWidget.isVisible(),
            highlight_checked=self.guiTabControl.highlightCheckbox.isChecked(),
            searched_highlight_checked=(
                self.guiTabControl.highlightSearchedCheckbox.isChecked()
            ),
        )
        if doHighlight:
            self.highlightedID = self.guiTabControl.propsQGBox.idSB.value()
            ID = self.highlightedID

        if self.highlightedID > 0:
            self.clearHighlightedText()

        self.searchedIDitemRight.setData([], [])
        self.searchedIDitemLeft.setData([], [])

        posData = self.data[self.pos_i]

        self.highlightedID = ID
        self.highlightIDToolbar.setVisible(True)

        objIdx = posData.IDs_idxs.get(ID)
        if objIdx is None:
            return

        obj = posData.rp[objIdx]
        isObjVisible = self.isObjVisible(obj.bbox)
        if not isObjVisible:
            return

        if greyOthers:
            self.textAnnot[0].grayOutAnnotations()
            self.textAnnot[1].grayOutAnnotations()

        how_ax1 = self.drawIDsContComboBox.currentText()
        how_ax2 = self.getAnnotateHowRightImage()
        isOverlaySegm_ax1 = how_ax1.find('segm. masks') != -1
        isOverlaySegm_ax2 = how_ax2.find('segm. masks') != -1
        alpha = self.imgGrad.labelsAlphaSlider.value()

        if isOverlaySegm_ax1 or isOverlaySegm_ax2:
            grayedLut = self.grayOutHighlightedLabels(
                nonGrayedIDs={obj.label},
                alpha=alpha
            )

        cont = None
        contours = None
        if isOverlaySegm_ax1:
            self.highLightIDLayerImg1.setLookupTable(grayedLut)
            self.highLightIDLayerImg1.setImage(self.highlightedLab)
            self.labelsLayerImg1.setOpacity(alpha/3)
        else:
            contours = self.getObjContours(obj, all_external=True)
            for cont in contours:
                self.searchedIDitemLeft.addPoints(cont[:,0]+0.5, cont[:,1]+0.5)

        if isOverlaySegm_ax2:
            self.highLightIDLayerRightImage.setLookupTable(grayedLut)
            self.highLightIDLayerRightImage.setImage(self.highlightedLab)
            self.labelsLayerRightImg.setOpacity(alpha/3)
        else:
            if contours is None:
                contours = self.getObjContours(obj, all_external=True)
            for cont in contours:
                self.searchedIDitemRight.addPoints(cont[:,0]+0.5, cont[:,1]+0.5)

        # Gray out all IDs excpet searched one
        lut = self.lut.copy() # [:max(posData.IDs)+1]
        lut[:ID] = lut[:ID]*0.2
        lut[ID+1:] = lut[ID+1:]*0.2
        self.img2.setLookupTable(lut)

        # Highlight text
        self.highlightLabelID(ID, ax=0)
        self.highlightLabelID(ID, ax=1)

    def _keepObjects(self, keepIDs=None, lab=None, rp=None):
        posData = self.data[self.pos_i]
        if lab is None:
            lab = posData.lab

        if rp is None:
            rp = posData.rp

        if keepIDs is None:
            keepIDs = self.keptObjectsIDs

        for obj in rp:
            if obj.label in keepIDs:
                continue

            lab[obj.slice][obj.image] = 0

        return lab

    def clearHighlightedText(self):
        pass

    def removeHighlightLabelID(self, IDs=None, ax=0):
        posData = self.data[self.pos_i]
        if IDs is None:
            IDs = posData.IDs

        for ID in IDs:
            obj = posData.rp[posData.IDs_idxs[ID]]
            self.textAnnot[ax].removeHighlightObject(obj)

    def updateKeepIDs(self, IDs):
        posData = self.data[self.pos_i]

        self.clearHighlightedText()

        isAnyIDnotExisting = False
        # Check if IDs from line edit are present in current keptObjectIDs list
        for ID in IDs:
            if ID not in posData.allIDs:
                isAnyIDnotExisting = True
                continue
            if ID not in self.keptObjectsIDs:
                self.keptObjectsIDs.append(ID, editText=False)
                self.highlightLabelID(ID)

        # Check if IDs in current keptObjectsIDs are present in IDs from line edit
        for ID in self.keptObjectsIDs:
            if ID not in posData.allIDs:
                isAnyIDnotExisting = True
                continue
            if ID not in IDs:
                self.keptObjectsIDs.remove(ID, editText=False)

        self.updateTempLayerKeepIDs()
        if isAnyIDnotExisting:
            self.keptIDsLineEdit.warnNotExistingID()
        else:
            self.keptIDsLineEdit.setInstructionsText()

    @exception_handler
    def applyKeepObjects(self):
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False)

        self._keepObjects()
        self.highlightHoverIDsKeptObj(0, 0, hoverID=0)

        posData = self.data[self.pos_i]

        self.update_rp()
        # Repeat tracking
        self.tracking(enforce=True, assign_unique_new_IDs=False)

        if self.isSnapshot:
            self.fixCcaDfAfterEdit('Deleted non-selected objects')
            self.updateAllImages()
            self.keptObjectsIDs = widgets.KeptObjectIDsList(
                self.keptIDsLineEdit, self.keepIDsConfirmAction
            )
            return
        else:
            removeAnnot = self.warnEditingWithCca_df(
                'Deleted non-selected objects', get_answer=True
            )
            if not removeAnnot:
                # We can propagate changes only if the user agrees on
                # removing annotations
                return

        self.current_frame_i = posData.frame_i
        if posData.frame_i > 0:
            txt = html_utils.paragraph("""
                Do you want to <b>remove un-kept objects in the past</b> frames too?
            """)
            msg = widgets.myMessageBox(wrapText=False, showCentered=False)
            _, _, applyToPastButton = msg.question(
                self, 'Propagate to past frames?', txt,
                buttonsTexts=('Cancel', 'No', 'Yes, apply to past frames')
            )
            if msg.cancel:
                return
            if msg.clickedButton == applyToPastButton:
                self.store_data()
                self.logger.info('Applying keep objects to past frames...')
                if not removeAnnot and posData.cca_df is not None:
                    delIDs = [
                        ID for ID in posData.cca_df.index
                        if ID not in posData.IDs
                    ]
                    self.update_cca_df_deletedIDs(posData, delIDs)

                for i in tqdm(range(posData.frame_i), ncols=100):
                    lab = posData.allData_li[i]['labels']
                    rp = posData.allData_li[i]['regionprops']
                    keepLab = self._keepObjects(lab=lab, rp=rp)
                    # Store change
                    posData.allData_li[i]['labels'] = keepLab.copy()
                    # Get the rest of the stored metadata based on the new lab
                    posData.frame_i = i
                    self.get_data()
                    self.store_data(autosave=False)

                posData.frame_i = self.current_frame_i
                self.get_data()

        # Ask to propagate change to all future visited frames
        key = 'Keep ID'
        askAction = self.askHowFutureFramesActions[key]
        doNotShow = not askAction.isChecked()
        (UndoFutFrames, applyFutFrames, endFrame_i,
        doNotShowAgain) = self.propagateChange(
            self.keptObjectsIDs, key, doNotShow,
            posData.UndoFutFrames_keepID, posData.applyFutFrames_keepID,
            force=True, applyTrackingB=True
        )

        if UndoFutFrames is None:
            # Empty keep object list
            self.keptObjectsIDs = widgets.KeptObjectIDsList(
                self.keptIDsLineEdit, self.keepIDsConfirmAction
            )
            return

        posData.doNotShowAgain_keepID = doNotShowAgain
        posData.UndoFutFrames_keepID = UndoFutFrames
        posData.applyFutFrames_keepID = applyFutFrames
        includeUnvisited = posData.includeUnvisitedInfo['Keep ID']

        if applyFutFrames:
            self.store_data()

            self.logger.info('Applying to future frames...')
            pbar = tqdm(total=posData.SizeT-posData.frame_i-1, ncols=100)
            segmSizeT = len(posData.segm_data)
            if not removeAnnot and posData.cca_df is not None:
                delIDs = [
                    ID for ID in posData.cca_df.index
                    if ID not in posData.IDs
                ]
                self.update_cca_df_deletedIDs(posData, delIDs)

            for i in range(posData.frame_i+1, segmSizeT):
                lab = posData.allData_li[i]['labels']
                if lab is None and not includeUnvisited:
                    self.enqAutosave()
                    pbar.update(posData.SizeT-i)
                    break

                rp = posData.allData_li[i]['regionprops']

                if lab is not None:
                    keepLab = self._keepObjects(lab=lab, rp=rp)
                    # Store change
                    posData.allData_li[i]['labels'] = keepLab.copy()
                    # Get the rest of the stored metadata based on the new lab
                    posData.frame_i = i
                    self.get_data()
                    self.store_data(autosave=False)
                elif includeUnvisited:
                    # Unvisited frame (includeUnvisited = True)
                    lab = posData.segm_data[i]
                    rp = skimage.measure.regionprops(lab)
                    keepLab = self._keepObjects(lab=lab, rp=rp)
                    posData.segm_data[i] = keepLab

                pbar.update()
            pbar.close()

        # Back to current frame
        if applyFutFrames:
            posData.frame_i = self.current_frame_i
            self.get_data()

        self.keptObjectsIDs = widgets.KeptObjectIDsList(
            self.keptIDsLineEdit, self.keepIDsConfirmAction
        )

    def get_curr_lab(self, curr_lab: np.ndarray|None = None, frame_i: int|None = None):
        """Get the current labels for the position data. Hirarchically checks:
        1. If `curr_lab` is provided, use it.
        2. If `posData.lab` is not None, use it.
        3. If `posData.allData_li[frame_i]['labels']` exists, use it.
        4. If `posData.segm_data[frame_i]` exists, use it.

        If frame_i is None, uses the current frame index from `posData`.

        Parameters
        ----------
        curr_lab : np.ndarray, optional
            Current labels for the position data if it should be checked
            if its not None first, by default None
        frame_i : int, optional
            Frame index to use for retrieving labels, by default None

        Returns
        -------
        np.ndarray
            Current labels for the position data
        """
        posData = self.data[self.pos_i]
        return self.view_model.object_counts.current_labels(
            posData,
            curr_lab=curr_lab,
            frame_i=frame_i,
        )

    def highlightIDonHoverCheckBoxToggled(self, checked):
        doHighlight = (
            self.guiTabControl.highlightCheckbox.isChecked()
            or self.guiTabControl.highlightSearchedCheckbox.isChecked()
        )
        if not doHighlight:
            self.highlightedID = 0
            self.initLookupTableLab()
        else:
            self.highlightedID = self.guiTabControl.propsQGBox.idSB.value()
            self.highlightSearchedID(self.highlightedID, force=True)
            self.updatePropsWidget(self.highlightedID)
        self.updateAllImages()

    def highlightSearchedIDcheckBoxToggled(self, checked):
        self.highlightIDonHoverCheckBoxToggled(checked)
        if checked:
            posData = self.data[self.pos_i]
            self.highlightedID = self.getHighlightedID()
            if self.highlightedID == 0:
                return
            objIdx = posData.IDs_idxs[self.highlightedID]
            obj_idx = posData.IDs_idxs.get(self.highlightedID)
            if obj_idx is None:
                return
            obj = posData.rp[objIdx]
            self.goToZsliceSearchedID(obj)

    def setHighlightID(self, doHighlight):
        if not doHighlight:
            self.highlightedID = 0
            self.initLookupTableLab()
        else:
            self.highlightedID = self.guiTabControl.propsQGBox.idSB.value()
            self.highlightSearchedID(self.highlightedID, force=True)
            self.updatePropsWidget(self.highlightedID)
        self.updateAllImages()

    def propsWidgetIDvalueChanged(self, ID):
        posData = self.data[self.pos_i]
        if ID == 0:
            self.updatePropsWidget(int(ID))
            return

        propsQGBox = self.guiTabControl.propsQGBox
        obj_idx = posData.IDs_idxs.get(ID)
        if obj_idx is None:
            s = f'Object ID {int(ID):d} does not exist'
            propsQGBox.notExistingIDLabel.setText(s)
            return

        obj = posData.rp[obj_idx]
        self.goToZsliceSearchedID(obj)
        self.updatePropsWidget(int(ID))

    def updatePropsWidget(self, ID, fromHover=False):
        if isinstance(ID, str):
            # Function called by currentTextChanged of channelCombobox or
            # additionalMeasCombobox. We set self.currentPropsID = 0 to force update
            ID = self.guiTabControl.propsQGBox.idSB.value()
            self.currentPropsID = -1

        ID = int(ID)

        update = self.view_model.should_update_props_widget(
            dock_visible=self.propsDockWidget.isVisible(),
            object_id=ID,
            current_props_id=self.currentPropsID,
        )
        if not update:
            return

        posData = self.data[self.pos_i]
        if not hasattr(posData, 'rp'):
            return

        if posData.rp is None:
            self.update_rp()

        if not posData.IDs:
            # empty segmentation mask
            return

        if fromHover and not self.guiTabControl.highlightCheckbox.isChecked():
            # Do not highlight on hover
            return

        propsQGBox = self.guiTabControl.propsQGBox

        obj_idx = posData.IDs_idxs.get(ID)
        if obj_idx is None:
            s = f'Object ID {int(ID):d} does not exist'
            propsQGBox.notExistingIDLabel.setText(s)
            return

        propsQGBox.notExistingIDLabel.setText('')
        self.currentPropsID = ID
        propsQGBox.idSB.setValue(ID)

        doHighlight = self.view_model.should_highlight_props_id(
            dock_visible=True,
            highlight_checked=self.guiTabControl.highlightCheckbox.isChecked(),
            searched_highlight_checked=(
                self.guiTabControl.highlightSearchedCheckbox.isChecked()
            ),
        )
        if doHighlight:
            self.highlightSearchedID(ID)

        obj = posData.rp[obj_idx]

        area_pxl = self.view_model.calculate_area_pxl(
            is_segm_3d=self.isSegm3D,
            z_proj_text=self.zProjComboBox.currentText(),
            z_lab=self.z_lab(),
            bbox_0=obj.bbox[0],
            obj_image=obj.image,
            obj_area=obj.area,
        )

        propsQGBox.cellAreaPxlSB.setValue(area_pxl)

        pixelSizeQGBox = self.guiTabControl.pixelSizeQGBox
        PhysicalSizeX = pixelSizeQGBox.pixelWidthWidget.value()
        PhysicalSizeY = pixelSizeQGBox.pixelHeightWidget.value()
        PhysicalSizeZ = pixelSizeQGBox.voxelDepthWidget.value()

        area_um2 = self.view_model.calculate_area_um2(
            area_pxl=area_pxl,
            physical_size_x=PhysicalSizeX,
            physical_size_y=PhysicalSizeY,
        )

        propsQGBox.cellAreaUm2DSB.setValue(area_um2)

        if self.isSegm3D:
            vol_vox_3D, vol_fl_3D = self.view_model.calculate_vol_3d(
                obj_area=obj.area,
                physical_size_x=PhysicalSizeX,
                physical_size_y=PhysicalSizeY,
                physical_size_z=posData.PhysicalSizeZ,
            )
            propsQGBox.cellVolVox3D_SB.setValue(vol_vox_3D)
            propsQGBox.cellVolFl3D_DSB.setValue(vol_fl_3D)

        vol_vox, vol_fl = self.view_model.measurements.rotational_volume(
            obj, PhysicalSizeY, PhysicalSizeX
        )
        propsQGBox.cellVolVoxSB.setValue(int(vol_vox))
        propsQGBox.cellVolFlDSB.setValue(vol_fl)

        elongation = self.view_model.calculate_elongation(
            major_axis_length=obj.major_axis_length,
            minor_axis_length=obj.minor_axis_length,
        )
        propsQGBox.elongationDSB.setValue(elongation)

        solidity = obj.solidity
        propsQGBox.solidityDSB.setValue(solidity)

        additionalPropName = propsQGBox.additionalPropsCombobox.currentText()
        additionalPropValue = getattr(obj, additionalPropName)
        propsQGBox.additionalPropsCombobox.indicator.setValue(additionalPropValue)

        intensMeasurQGBox = self.guiTabControl.intensMeasurQGBox
        selectedChannel = intensMeasurQGBox.channelCombobox.currentText()

        try:
            _, filename = self.getPathFromChName(selectedChannel, posData)
            image = posData.ol_data_dict[filename][posData.frame_i]
        except Exception as e:
            image = posData.img_data[posData.frame_i]

        objData, img = self.view_model.get_object_and_background_images(
            image=image,
            is_segm_3d=self.isSegm3D,
            pos_data_size_z=posData.SizeZ,
            z_slice=self.zSliceScrollBar.sliderPosition(),
            obj_slice=obj.slice,
            obj_image=obj.image,
            img1_image=self.img1.image,
        )

        stats = self.view_model.calculate_intensity_statistics(objData)
        intensMeasurQGBox.minimumDSB.setValue(stats['min'])
        intensMeasurQGBox.maximumDSB.setValue(stats['max'])
        intensMeasurQGBox.meanDSB.setValue(stats['mean'])
        intensMeasurQGBox.medianDSB.setValue(stats['median'])

        funcDesc = intensMeasurQGBox.additionalMeasCombobox.currentText()
        func = intensMeasurQGBox.additionalMeasCombobox.functions[funcDesc]

        value = self.view_model.calculate_additional_measure(
            func_desc=funcDesc,
            func=func,
            obj_data=objData,
            img=img,
            lab=posData.lab,
            obj_area=obj.area,
            vol_vox=vol_vox,
        )

        intensMeasurQGBox.additionalMeasCombobox.indicator.setValue(value)

