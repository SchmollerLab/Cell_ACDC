"""Qt view adapter for object-property workflows."""

from __future__ import annotations

import numpy as np
import skimage.measure
from tqdm import tqdm

from cellacdc import apps, exception_handler, html_utils, widgets


class ObjectPropertiesMixin:
    """Qt-facing adapter around object properties and highlighting."""

    """Headless decisions for object-property and highlight workflows."""

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
            self.fixCcaDfAfterEdit("Deleted non-selected objects")
            self.updateAllImages()
            self.keptObjectsIDs = widgets.KeptObjectIDsList(
                self.keptIDsLineEdit, self.keepIDsConfirmAction
            )
            return
        else:
            removeAnnot = self.warnEditingWithCca_df(
                "Deleted non-selected objects", get_answer=True
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
                self,
                "Propagate to past frames?",
                txt,
                buttonsTexts=("Cancel", "No", "Yes, apply to past frames"),
            )
            if msg.cancel:
                return
            if msg.clickedButton == applyToPastButton:
                self.store_data()
                self.logger.info("Applying keep objects to past frames...")
                if not removeAnnot and posData.cca_df is not None:
                    delIDs = [
                        ID for ID in posData.cca_df.index if ID not in posData.IDs
                    ]
                    self.update_cca_df_deletedIDs(posData, delIDs)

                for i in tqdm(range(posData.frame_i), ncols=100):
                    lab = posData.allData_li[i]["labels"]
                    rp = posData.allData_li[i]["regionprops"]
                    keepLab = self._keepObjects(lab=lab, rp=rp)
                    # Store change
                    posData.allData_li[i]["labels"] = keepLab.copy()
                    # Get the rest of the stored metadata based on the new lab
                    posData.frame_i = i
                    self.get_data()
                    self.store_data(autosave=False)

                posData.frame_i = self.current_frame_i
                self.get_data()

        # Ask to propagate change to all future visited frames
        key = "Keep ID"
        askAction = self.askHowFutureFramesActions[key]
        doNotShow = not askAction.isChecked()
        (UndoFutFrames, applyFutFrames, endFrame_i, doNotShowAgain) = (
            self.propagateChange(
                self.keptObjectsIDs,
                key,
                doNotShow,
                posData.UndoFutFrames_keepID,
                posData.applyFutFrames_keepID,
                force=True,
                applyTrackingB=True,
            )
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
        includeUnvisited = posData.includeUnvisitedInfo["Keep ID"]

        if applyFutFrames:
            self.store_data()

            self.logger.info("Applying to future frames...")
            pbar = tqdm(total=posData.SizeT - posData.frame_i - 1, ncols=100)
            segmSizeT = len(posData.segm_data)
            if not removeAnnot and posData.cca_df is not None:
                delIDs = [ID for ID in posData.cca_df.index if ID not in posData.IDs]
                self.update_cca_df_deletedIDs(posData, delIDs)

            for i in range(posData.frame_i + 1, segmSizeT):
                lab = posData.allData_li[i]["labels"]
                if lab is None and not includeUnvisited:
                    self.enqAutosave()
                    pbar.update(posData.SizeT - i)
                    break

                rp = posData.allData_li[i]["regionprops"]

                if lab is not None:
                    keepLab = self._keepObjects(lab=lab, rp=rp)
                    # Store change
                    posData.allData_li[i]["labels"] = keepLab.copy()
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

    def calculate_additional_measure(
        self,
        *,
        func_desc: str,
        func: callable,
        obj_data: np.ndarray,
        img: np.ndarray,
        lab: np.ndarray,
        obj_area: int,
        vol_vox: float,
    ) -> float:
        if func_desc in ("Concentration", "Amount"):
            background_pixels = img[lab == 0]
            bkgr_val = (
                float(np.median(background_pixels))
                if background_pixels.size > 0
                else 0.0
            )
            amount = func(obj_data, bkgr_val, obj_area)
            if func_desc == "Concentration":
                return amount / vol_vox
            else:
                return amount
        else:
            return float(func(obj_data))

    def calculate_area_pxl(
        self,
        *,
        is_segm_3d: bool,
        z_proj_text: str,
        z_lab: int,
        bbox_0: int,
        obj_image: np.ndarray,
        obj_area: int,
    ) -> int:
        if is_segm_3d:
            if z_proj_text == "single z-slice":
                local_z = z_lab - bbox_0
                return int(np.count_nonzero(obj_image[local_z]))
            else:
                return int(np.count_nonzero(obj_image.max(axis=0)))
        else:
            return obj_area

    def calculate_area_um2(
        self,
        *,
        area_pxl: int,
        physical_size_x: float,
        physical_size_y: float,
    ) -> float:
        return area_pxl * physical_size_y * physical_size_x

    def calculate_elongation(
        self,
        *,
        major_axis_length: float,
        minor_axis_length: float,
    ) -> float:
        minor_axis = max(1.0, minor_axis_length)
        return major_axis_length / minor_axis

    def calculate_intensity_statistics(
        self,
        obj_data: np.ndarray,
    ) -> dict[str, float]:
        if obj_data.size == 0:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
        return {
            "min": float(np.min(obj_data)),
            "max": float(np.max(obj_data)),
            "mean": float(np.mean(obj_data)),
            "median": float(np.median(obj_data)),
        }

    def calculate_vol_3d(
        self,
        *,
        obj_area: int,
        physical_size_x: float,
        physical_size_y: float,
        physical_size_z: float,
    ) -> tuple[float, float]:
        vol_vox_3D = obj_area
        vol_fl_3D = vol_vox_3D * physical_size_z * physical_size_y * physical_size_x
        return float(vol_vox_3D), float(vol_fl_3D)

    def clearHighlightedID(self):
        self.highlightIDToolbar.setVisible(False)

        try:
            self.updateLostContoursImage(ax=0, delROIsIDs=None)
        except Exception:
            pass

        if self.highlightedID == 0:
            return

        self.highlightedID = 0
        self.guiTabControl.highlightCheckbox.setChecked(False)
        self.guiTabControl.highlightSearchedCheckbox.setChecked(False)
        self.setHighlightID(False)

    def clearHighlightedKeepIDs(self):
        self.setAllTextAnnotations()
        self.highlightedID = 0
        self.searchedIDitemRight.setData([], [])
        self.searchedIDitemLeft.setData([], [])
        self.highLightIDLayerImg1.clear()
        self.highLightIDLayerRightImage.clear()

    def clearHighlightedText(self):
        pass

    def countObjects(self):
        self.logger.info("Counting objects...")

        posData = self.data[self.pos_i]
        if posData.SizeT > 1:
            return self.countObjectsTimelapse()

        return self.countObjectsSnapshots()

    def countObjectsCb(self, checked):
        if self.countObjsWindow is None:
            categoryCountMapper = self.countObjects()
            self.countObjsWindow = apps.ObjectCountDialog(
                categoryCountMapper=categoryCountMapper, parent=self, data=self.data
            )
            self.countObjsWindow.sigShowEvent.connect(self.updateObjectCounts)
            self.countObjsWindow.sigUpdateCounts.connect(self.updateObjectCounts)

        if checked:
            self.countObjsWindow.show()
        else:
            self.countObjsWindow.hide()

    def countObjectsSnapshots(self):
        posData = self.data[self.pos_i]
        if self.countObjsWindow is None:
            activeCategories = {
                "In current position",
                "In all visited positions (current session)",
                "In all visited positions (previous sessions)",
                "In all loaded positions",
            }
            if self.isSegm3D:
                activeCategories.add("In current z-slice")
        else:
            activeCategories = self.countObjsWindow.activeCategories()

        numObjectsCurrentPos = len(posData.IDs)
        numObjectsAllPos = 0
        numObjectsVisitedPosPrevious = 0
        numObjectsVisitedPosCurrent = 0
        numObjectsCurrentZslice = None
        if "In current z-slice" in activeCategories:
            numObjectsCurrentZslice = len(
                skimage.measure.regionprops(self.currentLab2D)
            )

        for pos_i, _posData in enumerate(self.data):
            IDs = _posData.allData_li[0]["IDs"]
            if os.path.exists(_posData.acdc_output_csv_path):
                numObjectsVisitedPosPrevious += len(IDs)
            if IDs:
                numObjs = len(IDs)
                numObjectsAllPos += len(IDs)
            else:
                lab = _posData.segm_data[0]
                rp = skimage.measure.regionprops(lab)
                numObjs = len(rp)
                numObjectsAllPos += numObjs

            if _posData.visited:
                numObjectsVisitedPosCurrent += numObjs

        allCategoryCountMapper = {
            "In current position": numObjectsCurrentPos,
            "In all visited positions (current session)": numObjectsVisitedPosCurrent,
            "In all visited positions (previous sessions)": numObjectsVisitedPosPrevious,
            "In all loaded positions": numObjectsAllPos,
        }
        if numObjectsCurrentZslice is not None:
            allCategoryCountMapper["In current z-slice"] = numObjectsCurrentZslice

        if self.countObjsWindow is None:
            return allCategoryCountMapper

        categoryCountMapper = {}
        for category in activeCategories:
            categoryCountMapper[category] = allCategoryCountMapper[category]

        return categoryCountMapper

    def countObjectsTimelapse(self):
        if self.countObjsWindow is None:
            activeCategories = {
                "In current frame",
                "In all visited frames",
                "In entire video",
                "Unique objects in all visited frames",
                "Unique objects in entire video",
            }
        else:
            activeCategories = self.countObjsWindow.activeCategories()

        posData = self.data[self.pos_i]
        allCategoryCountMapper = posData.countObjectsInSegmTimelapse(activeCategories)
        if self.countObjsWindow is None:
            return allCategoryCountMapper

        categoryCountMapper = {}
        for category in activeCategories:
            categoryCountMapper[category] = allCategoryCountMapper[category]

        return categoryCountMapper

    def getHighlightedID(self):
        if self.highlightedID > 0:
            return self.highlightedID

        doHighlight = self.propsDockWidget.isVisible() and (
            self.guiTabControl.highlightCheckbox.isChecked()
            or self.guiTabControl.highlightSearchedCheckbox.isChecked()
        )
        if not doHighlight:
            return 0

        return self.guiTabControl.propsQGBox.idSB.value()

    def get_curr_lab(
        self, curr_lab: np.ndarray | None = None, frame_i: int | None = None
    ):
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
        if frame_i is None:
            frame_i = posData.frame_i

        if curr_lab is None and frame_i == posData.frame_i:
            curr_lab = posData.lab

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

        return curr_lab

    def get_object_and_background_images(
        self,
        *,
        image: np.ndarray,
        is_segm_3d: bool,
        pos_data_size_z: int,
        z_slice: int,
        obj_slice: tuple,
        obj_image: np.ndarray,
        img1_image: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if pos_data_size_z > 1 and not is_segm_3d:
            obj_data = image[z_slice][obj_slice][obj_image]
            img = img1_image if img1_image is not None else image[z_slice]
        else:
            obj_data = image[obj_slice][obj_image]
            img = image
        return obj_data, img

    def grayOutHighlightedLabels(self, nonGrayedIDs=None, alpha=None):
        if nonGrayedIDs is None:
            nonGrayedIDs = set()

        posData = self.data[self.pos_i]
        if alpha is None:
            alpha = self.imgGrad.labelsAlphaSlider.value()

        if not hasattr(self, "highlightedLab"):
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

        isOverlaySegmActive = how.find("segm. masks") != -1
        if not isOverlaySegmActive:
            return

        self.grayOutHighlightedLabels()

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
        except KeyError:
            return

        obj = posData.rp[objIdx]
        self.goToZsliceSearchedID(obj)

        for ID in self.keptObjectsIDs:
            self.highlightLabelID(ID)

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

    def highlightLabelID(self, ID, ax=0):
        posData = self.data[self.pos_i]
        try:
            obj = posData.rp[posData.IDs_idxs[ID]]
        except KeyError:
            return

        self.textAnnot[ax].highlightObject(obj)

    def highlightSearchedID(self, ID, force=False, greyOthers=True):
        self.highlightIDToolbar.setIDNoSignals(ID)

        if ID == 0:
            self.highlightIDToolbar.setVisible(False)
            return

        if ID == self.highlightedID and not force:
            return

        doHighlight = self.propsDockWidget.isVisible() and (
            self.guiTabControl.highlightCheckbox.isChecked()
            or self.guiTabControl.highlightSearchedCheckbox.isChecked()
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
        isOverlaySegm_ax1 = how_ax1.find("segm. masks") != -1
        isOverlaySegm_ax2 = how_ax2.find("segm. masks") != -1
        alpha = self.imgGrad.labelsAlphaSlider.value()

        if isOverlaySegm_ax1 or isOverlaySegm_ax2:
            grayedLut = self.grayOutHighlightedLabels(
                nonGrayedIDs={obj.label}, alpha=alpha
            )

        cont = None
        contours = None
        if isOverlaySegm_ax1:
            self.highLightIDLayerImg1.setLookupTable(grayedLut)
            self.highLightIDLayerImg1.setImage(self.highlightedLab)
            self.labelsLayerImg1.setOpacity(alpha / 3)
        else:
            contours = self.getObjContours(obj, all_external=True)
            for cont in contours:
                self.searchedIDitemLeft.addPoints(cont[:, 0] + 0.5, cont[:, 1] + 0.5)

        if isOverlaySegm_ax2:
            self.highLightIDLayerRightImage.setLookupTable(grayedLut)
            self.highLightIDLayerRightImage.setImage(self.highlightedLab)
            self.labelsLayerRightImg.setOpacity(alpha / 3)
        else:
            if contours is None:
                contours = self.getObjContours(obj, all_external=True)
            for cont in contours:
                self.searchedIDitemRight.addPoints(cont[:, 0] + 0.5, cont[:, 1] + 0.5)

        # Gray out all IDs excpet searched one
        lut = self.lut.copy()  # [:max(posData.IDs)+1]
        lut[:ID] = lut[:ID] * 0.2
        lut[ID + 1 :] = lut[ID + 1 :] * 0.2
        self.img2.setLookupTable(lut)

        # Highlight text
        self.highlightLabelID(ID, ax=0)
        self.highlightLabelID(ID, ax=1)

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

    def initKeepObjLabelsLayers(self):
        lut = np.zeros((len(self.lut), 4), dtype=np.uint8)
        lut[:, :-1] = self.lut
        lut[:, -1:] = 255
        lut[0] = [0, 0, 0, 0]
        self.keepIDsTempLayerLeft.setLevels([0, len(lut)])
        self.keepIDsTempLayerLeft.setLookupTable(lut)

    def initPixelSizePropsDockWidget(self):
        posData = self.data[self.pos_i]
        PhysicalSizeX = posData.PhysicalSizeX
        PhysicalSizeY = posData.PhysicalSizeY
        PhysicalSizeZ = posData.PhysicalSizeZ
        self.guiTabControl.initPixelSize(PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ)

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

    def propsWidgetIDvalueChanged(self, ID):
        posData = self.data[self.pos_i]
        if ID == 0:
            self.updatePropsWidget(int(ID))
            return

        propsQGBox = self.guiTabControl.propsQGBox
        obj_idx = posData.IDs_idxs.get(ID)
        if obj_idx is None:
            s = f"Object ID {int(ID):d} does not exist"
            propsQGBox.notExistingIDLabel.setText(s)
            return

        obj = posData.rp[obj_idx]
        self.goToZsliceSearchedID(obj)
        self.updatePropsWidget(int(ID))

    def removeHighlightLabelID(self, IDs=None, ax=0):
        posData = self.data[self.pos_i]
        if IDs is None:
            IDs = posData.IDs

        for ID in IDs:
            obj = posData.rp[posData.IDs_idxs[ID]]
            self.textAnnot[ax].removeHighlightObject(obj)

    def setAllIDs(self, onlyVisited=False):
        for posData in self.data:
            posData.allIDs = set()
            for frame_i in range(len(posData.segm_data)):
                if frame_i >= len(posData.allData_li):
                    break
                lab = posData.allData_li[frame_i]["labels"]
                if lab is None and onlyVisited:
                    break

                if lab is None:
                    rp = skimage.measure.regionprops(posData.segm_data[frame_i])
                else:
                    rp = posData.allData_li[frame_i]["regionprops"]
                posData.allIDs.update([obj.label for obj in rp])

    def setHighlighedIDfromToolbar(self, ID: int):
        self.findID(ID=ID)

    def setHighlightID(self, doHighlight):
        if not doHighlight:
            self.highlightedID = 0
            self.initLookupTableLab()
        else:
            self.highlightedID = self.guiTabControl.propsQGBox.idSB.value()
            self.highlightSearchedID(self.highlightedID, force=True)
            self.updatePropsWidget(self.highlightedID)
        self.updateAllImages()

    def should_highlight_props_id(
        self,
        *,
        dock_visible: bool,
        highlight_checked: bool,
        searched_highlight_checked: bool,
    ) -> bool:
        return dock_visible and (highlight_checked or searched_highlight_checked)

    def should_show_3d_property_controls(self, is_segm_3d: bool) -> bool:
        return is_segm_3d

    def should_update_object_counts(
        self,
        *,
        window_exists: bool,
        is_visible: bool,
        live_preview_checked: bool,
    ) -> bool:
        return window_exists and is_visible and live_preview_checked

    def should_update_props_widget(
        self,
        *,
        dock_visible: bool,
        object_id: int,
        current_props_id: int,
    ) -> bool:
        return dock_visible and object_id != 0 and object_id != current_props_id

    def showPropsDockWidget(self, checked=False):
        if self.showPropsDockButton.isExpand:
            self.propsDockWidget.setVisible(False)
            self.setHighlightID(False)
        else:
            self.highlightedID = self.guiTabControl.propsQGBox.idSB.value()
            if self.isSegm3D:
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

    def snapshot_default_categories(self, *, is_segm_3d: bool) -> set[str]:
        categories = {
            "In current position",
            "In all visited positions (current session)",
            "In all visited positions (previous sessions)",
            "In all loaded positions",
        }
        if is_segm_3d:
            categories.add("In current z-slice")
        return categories

    def timelapse_default_categories(self) -> set[str]:
        return {
            "In current frame",
            "In all visited frames",
            "In entire video",
            "Unique objects in all visited frames",
            "Unique objects in entire video",
        }

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

    def updateObjectCounts(self):
        if self.countObjsWindow is None:
            return

        if not self.countObjsWindow.isVisible():
            return

        if not self.countObjsWindow.livePreviewCheckbox.isChecked():
            return

        categoryCountMapper = self.countObjects()
        self.countObjsWindow.updateCounts(categoryCountMapper)

    def updatePropsWidget(self, ID, fromHover=False):
        if isinstance(ID, str):
            # Function called by currentTextChanged of channelCombobox or
            # additionalMeasCombobox. We set self.currentPropsID = 0 to force update
            ID = self.guiTabControl.propsQGBox.idSB.value()
            self.currentPropsID = -1

        ID = int(ID)

        update = (
            self.propsDockWidget.isVisible() and ID != 0 and ID != self.currentPropsID
        )
        if not update:
            return

        posData = self.data[self.pos_i]
        if not hasattr(posData, "rp"):
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
            s = f"Object ID {int(ID):d} does not exist"
            propsQGBox.notExistingIDLabel.setText(s)
            return

        propsQGBox.notExistingIDLabel.setText("")
        self.currentPropsID = ID
        propsQGBox.idSB.setValue(ID)

        doHighlight = (
            self.guiTabControl.highlightCheckbox.isChecked()
            or self.guiTabControl.highlightSearchedCheckbox.isChecked()
        )
        if doHighlight:
            self.highlightSearchedID(ID)

        obj = posData.rp[obj_idx]

        if self.isSegm3D:
            if self.zProjComboBox.currentText() == "single z-slice":
                local_z = self.z_lab() - obj.bbox[0]
                area_pxl = np.count_nonzero(obj.image[local_z])
            else:
                area_pxl = np.count_nonzero(obj.image.max(axis=0))
        else:
            area_pxl = obj.area

        propsQGBox.cellAreaPxlSB.setValue(area_pxl)

        pixelSizeQGBox = self.guiTabControl.pixelSizeQGBox
        PhysicalSizeX = pixelSizeQGBox.pixelWidthWidget.value()
        PhysicalSizeY = pixelSizeQGBox.pixelHeightWidget.value()
        PhysicalSizeZ = pixelSizeQGBox.voxelDepthWidget.value()

        yx_pxl_to_um2 = PhysicalSizeY * PhysicalSizeX

        area_um2 = area_pxl * yx_pxl_to_um2

        propsQGBox.cellAreaUm2DSB.setValue(area_um2)

        if self.isSegm3D:
            PhysicalSizeZ = posData.PhysicalSizeZ
            vol_vox_3D = obj.area
            vol_fl_3D = vol_vox_3D * PhysicalSizeZ * PhysicalSizeY * PhysicalSizeX
            propsQGBox.cellVolVox3D_SB.setValue(vol_vox_3D)
            propsQGBox.cellVolFl3D_DSB.setValue(vol_fl_3D)

        vol_vox, vol_fl = _calc_rot_vol(obj, PhysicalSizeY, PhysicalSizeX)
        propsQGBox.cellVolVoxSB.setValue(int(vol_vox))
        propsQGBox.cellVolFlDSB.setValue(vol_fl)

        minor_axis_length = max(1, obj.minor_axis_length)
        elongation = obj.major_axis_length / minor_axis_length
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
        except Exception:
            image = posData.img_data[posData.frame_i]

        if posData.SizeZ > 1 and not self.isSegm3D:
            z = self.zSliceScrollBar.sliderPosition()
            objData = image[z][obj.slice][obj.image]
            img = self.img1.image
        else:
            objData = image[obj.slice][obj.image]
            img = image

        intensMeasurQGBox.minimumDSB.setValue(np.min(objData))
        intensMeasurQGBox.maximumDSB.setValue(np.max(objData))
        intensMeasurQGBox.meanDSB.setValue(np.mean(objData))
        intensMeasurQGBox.medianDSB.setValue(np.median(objData))

        funcDesc = intensMeasurQGBox.additionalMeasCombobox.currentText()
        func = intensMeasurQGBox.additionalMeasCombobox.functions[funcDesc]
        if funcDesc == "Concentration":
            bkgrVal = np.median(img[posData.lab == 0])
            amount = func(objData, bkgrVal, obj.area)
            value = amount / vol_vox
        elif funcDesc == "Amount":
            bkgrVal = np.median(img[posData.lab == 0])
            amount = func(objData, bkgrVal, obj.area)
            value = amount
        else:
            value = func(objData)

        intensMeasurQGBox.additionalMeasCombobox.indicator.setValue(value)

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
