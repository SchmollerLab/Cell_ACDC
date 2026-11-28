"""Image and labels display of the main GUI, the title label with clickable
IDs and the status bar label. Mixins of guiWin
(gui/main_window/window.py) only: the measurements viewer never composes
them.
"""
import os
import re
import textwrap
import traceback
import numpy as np
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.draw
import skimage.exposure
import skimage.transform
import skimage.segmentation
from qtpy.QtCore import Qt, QPoint
from qtpy.QtGui import QCursor
from cellacdc import exception_handler, disableWindow
from cellacdc import myutils
from cellacdc import preprocess


class ImageDisplayGuiElements:
    """Image and labels display: getImage, setImageImg1/setImageImg2,
    updateAllImages, intensity normalization, labels lookup table and
    colors, highlighting of IDs, contours image, text colors, font size
    and resolution.
    """

    def getDistantGray(self, desiredGray, bkgrGray):
        isDesiredSimilarToBkgr = (
            abs(desiredGray-bkgrGray) < 0.3
        )
        if isDesiredSimilarToBkgr:
            return 1-desiredGray
        else:
            return desiredGray

    def RGBtoGray(self, R, G, B):
        # see https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
        C_linear = (0.2126*R + 0.7152*G + 0.0722*B)/255
        if C_linear <= 0.0031309:
            gray = 12.92*C_linear
        else:
            gray = 1.055*(C_linear)**(1/2.4) - 0.055
        return gray

    def normalizeIntensities(self, img):
        action, normalize, how = self.getCheckNormAction()
        if not normalize:
            return img
        
        if how == 'Do not normalize. Display raw image':
            img = img 
        elif how == 'Convert to floating point format with values [0, 1]':
            img = myutils.img_to_float(img)
        # elif how == 'Rescale to 8-bit unsigned integer format with values [0, 255]':
        #     img = skimage.img_as_float(img)
        #     img = (img*255).astype(np.uint8)
        #     return img
        elif how == 'Rescale to [0, 1]':
            img = skimage.img_as_float(img)
            img = skimage.exposure.rescale_intensity(img)
        elif how == 'Normalize by max value':
            img = img/np.max(img)
        return img

    def normaliseIntensitiesActionTriggered(self, action):
        how = action.text()
        self.df_settings.at['how_normIntensities', 'value'] = how
        self.df_settings.to_csv(self.settings_csv_path)
        self.updateAllImages()
        self.updateImageValueFormatter()

    def setLastUserNormAction(self):
        how = self.df_settings.at['how_normIntensities', 'value']
        for action in self.normalizeQActionGroup.actions():
            if action.text() == how:
                action.setChecked(True)
                break

    def fontSizeChangedContextMenu(self, font_size: int):
        self.labelsGrad.setFontSize(font_size, blockSignal=True)
        self.imgGrad.setFontSize(font_size, blockSignal=True)
        self.fontSizeSpinBox.blockSignals(True)
        self.fontSizeSpinBox.setValue(font_size)
        self.fontSizeSpinBox.blockSignals(False)
        font_size = self.fontSizeSpinBox.value()
        self.labelsGrad.setFontSize(font_size, blockSignal=True)
        self.imgGrad.setFontSize(font_size, blockSignal=True)
        self.changeFontSize()

    @exception_handler
    def changeFontSize(self):
        fontSize = self.fontSizeSpinBox.value()
        if fontSize == self.fontSize:
            return
        
        self.fontSize = fontSize
        self.labelsGrad.setFontSize(fontSize, blockSignal=True)
        self.imgGrad.setFontSize(fontSize, blockSignal=True)

        self.df_settings.at['fontSize', 'value'] = self.fontSize
        self.df_settings.to_csv(self.settings_csv_path)
        
        self.setAllIDs()
        posData = self.data[self.pos_i]
        for ax in range(2):
            self.textAnnot[ax].changeFontSize(self.fontSize)
        self.setAllTextAnnotations()
        # if self.highLowResAction.isChecked():
        #     self.setAllTextAnnotations()
        # else:
        #     self.updateAllImages()

    @disableWindow
    def equalizeHist(self, checked=True):
        self.img1.useEqualized = checked
        
        if not checked:
            self.updateAllImages()
            return

        self.logger.info('Equalizing image histogram...')
        for pos_i, _posData in enumerate(self.data):
            n_dim_img = _posData.img_data.ndim
            _posData.equalized_img_data = preprocess.PreprocessedData()
            for frame_i, img_frame in enumerate(_posData.img_data):
                if n_dim_img == 4:
                    for z, img_z in enumerate(img_frame):
                        eq_img = skimage.exposure.equalize_adapthist(img_z)
                        _posData.equalized_img_data[frame_i][z] = eq_img
                        self.img1.updateMinMaxValuesEqualizedData(
                            self.data, pos_i, frame_i, z
                        )
                    self.img1.updateMinMaxValuesEqualizedDataProjections(
                        self.data, pos_i, frame_i
                    )
                else:
                    eq_img = skimage.exposure.equalize_adapthist(img_frame)
                    _posData.equalized_img_data[frame_i] = eq_img
                    self.img1.updateMinMaxValuesEqualizedData(
                        self.data, pos_i, frame_i, None
                    )
                
        self.updateAllImages()

    def defaultRescaleIntensLutActionToggled(self, action):
        how = action.text()
        for rescaleIntensAction in self.imgGrad.rescaleActionGroup.actions():
            if how == rescaleIntensAction.text():
                rescaleIntensAction.setChecked(True)
                rescaleIntensAction.trigger()
                break
        
        for channel, items in self.overlayLayersItems.items():
            lutItem = items[1]
            for rescaleIntensAction in lutItem.rescaleActionGroup.actions():
                if how == rescaleIntensAction.text():
                    rescaleIntensAction.setChecked(True)
                    rescaleIntensAction.trigger()
                    break
        
        self.df_settings.at['default_rescale_intens_how', 'value'] = how
        self.df_settings.to_csv(self.settings_csv_path)

    # @exec_time
    def getDisplayedImg1(self):
        return self.img1.image

    def getDisplayedZstack(self):
        posData = self.data[self.pos_i]
        return posData.img_data[posData.frame_i]

    def clearAx2Items(self, onlyHideText=False):
        self.ax2_binnedIDs_ScatterPlot.clear()
        self.ax2_ripIDs_ScatterPlot.clear()
        self.ax2_contoursImageItem.clear()
        self.ax2_lostObjImageItem.clear()
        self.ax2_lostTrackedObjImageItem.clear()
        self.textAnnot[1].clear()
        self.ax2_newMothBudLinesItem.setData([], [])
        self.ax2_oldMothBudLinesItem.setData([], [])
        self.ax2_lostObjScatterItem.setData([], [])
        self.ax2_movementAgainstPrevLinesItem.setData([], [])
        self.ax2_movementAgainstPrevScatterItem.setData([], [])

    def clearAx1Items(self, onlyHideText=False):
        self.ax1_binnedIDs_ScatterPlot.clear()
        self.ax1_ripIDs_ScatterPlot.clear()
        self.labelsLayerImg1.clear()
        self.labelsLayerRightImg.clear()
        self.keepIDsTempLayerLeft.clear()
        self.keepIDsTempLayerRight.clear()
        self.highLightIDLayerImg1.clear()
        self.highLightIDLayerRightImage.clear()
        self.brushStrokesLayerImg1.clear()
        self.brushStrokesSelectedCellsLayerImg1.clear()
        self.searchedIDitemLeft.clear()
        self.searchedIDitemRight.clear()
        self.ax1_contoursImageItem.clear()
        self.ax1_lostObjImageItem.clear()
        self.ax1_lostTrackedObjImageItem.clear()
        self.textAnnot[0].clear()
        self.ax1_newMothBudLinesItem.setData([], [])
        self.ax1_oldMothBudLinesItem.setData([], [])
        self.ax1_lostObjScatterItem.setData([], [])
        self.ax1_lostTrackedScatterItem.setData([], [])
        self.ccaFailedScatterItem.setData([], [])
        self.yellowContourScatterItem.setData([], [])

        self.ax1_movementAgainstPrevLinesItem.setData([], [])
        self.ax1_trackerMovementAgainstPrevLinesItem.setData([], [])
        self.ax1_movementAgainstPrevScatterItem.setData([], [])
        self.clearObjTracks()
        
        self.clearPointsLayers()

        self.clearOverlayLabelsItems()
        self.clearManualBackgroundAnnotations()
        self.clearCustomAnnot()

    def clearAllItems(self):
        self.clearAx1Items()
        self.clearAx2Items()

    def initImgCmap(self):
        if not 'img_cmap' in self.df_settings.index:
            self.df_settings.at['img_cmap', 'value'] = 'grey'
        self.imgCmapName = self.df_settings.at['img_cmap', 'value']
        self.imgCmap = self.imgGrad.cmaps[self.imgCmapName]
        if self.imgCmapName != 'grey':
            # To ensure mapping to colors we need to normalize image
            self.normalizeByMaxAction.setChecked(True)

    def highlightIDs(self, IDs, pen):
        pass

    def initLookupTableLab(self):
        self.img2.setLookupTable(self.lut)
        self.img2.setLevels([0, len(self.lut)])
        self.initLabelsImageItems()

    def initLabelsImageItems(self):
        lut = self.getLabelsImageLut()
        self.labelsLayerImg1.setLevels([0, len(lut)])
        self.labelsLayerRightImg.setLevels([0, len(lut)])
        self.labelsLayerImg1.setLookupTable(lut)
        self.labelsLayerRightImg.setLookupTable(lut)
        alpha = self.imgGrad.labelsAlphaSlider.value()
        self.labelsLayerImg1.setOpacity(alpha)
        self.labelsLayerRightImg.setOpacity(alpha)

    def highlightLabelID(self, ID, ax=0):        
        posData = self.data[self.pos_i]
        obj = posData.rp.get_obj_from_ID(ID, warn=False)
        if obj is None:
            return
        
        self.textAnnot[ax].highlightObject(
            obj, rp=posData.rp, getObjCentroidFunc=self.getObjCentroid)

    def clearHighlightedText(self):
        pass

    def removeHighlightLabelID(self, IDs=None, ax=0):
        posData = self.data[self.pos_i]
        if IDs is None:
            IDs = posData.IDs
        
        for ID in IDs:
            obj = posData.rp.get_obj_from_ID(ID, warn=False)
            if obj is None:
                continue
            self.textAnnot[ax].removeHighlightObject(obj)

    def updateLookuptable(self, lenNewLut=None, delIDs=None):
        posData = self.data[self.pos_i]
        if lenNewLut is None:
            try:
                if delIDs is None:
                    IDs = posData.IDs
                else:
                    # Remove IDs removed with ROI from LUT
                    IDs = [ID for ID in posData.IDs if ID not in delIDs]
                lenNewLut = max(IDs, default=0) + 1
            except ValueError:
                # Empty segmentation mask
                lenNewLut = 1
        # Build a new lut to include IDs > than original len of lut
        updateLevels = self.extendLabelsLUT(lenNewLut)
        lut = self.lut.copy()

        try:
            # lut = self.lut[:lenNewLut].copy()
            for ID in posData.binnedIDs:
                lut[ID] = lut[ID]*0.2

            for ID in posData.ripIDs:
                lut[ID] = lut[ID]*0.2
        except Exception as e:
            err_str = traceback.format_exc()
            print('='*30)
            self.logger.info(err_str)
            print('='*30)

        if updateLevels:
            self.img2.setLevels([0, len(lut)])
        
        if self.keepIDsButton.isChecked():
            lut = np.round(lut*0.3).astype(np.uint8)
            keptLut = np.round(lut[self.keptObjectsIDs]/0.3).astype(np.uint8)
            lut[self.keptObjectsIDs] = keptLut

        self.img2.setLookupTable(lut)

    def annotate_rip_and_bin_IDs(self, updateLabel=False):
        depthAxes = self.switchPlaneCombobox.depthAxes()
        if self.switchPlaneCombobox.isEnabled() and depthAxes != 'z':
            return 
        
        posData = self.data[self.pos_i]
        binnedIDs_xx = []
        binnedIDs_yy = []
        ripIDs_xx = []
        ripIDs_yy = []
        for obj in posData.rp:
            obj.excluded = obj.label in posData.binnedIDs
            obj.dead = obj.label in posData.ripIDs
            if not self.isObjInCurrSlice(obj.bbox):
                continue
            
            if obj.excluded:
                ID = obj.label
                y, x = self.getObjCentroid(posData.rp.get_centroid(ID))
                binnedIDs_xx.append(x)
                binnedIDs_yy.append(y)
                if updateLabel:
                    self.getObjOptsSegmLabels(obj)
            
            if obj.dead:
                ID = obj.label
                y, x = self.getObjCentroid(posData.rp.get_centroid(ID))
                ripIDs_xx.append(x)
                ripIDs_yy.append(y)
                if updateLabel:
                    self.getObjOptsSegmLabels(obj)
        
        self.ax2_binnedIDs_ScatterPlot.setData(binnedIDs_xx, binnedIDs_yy)
        self.ax2_ripIDs_ScatterPlot.setData(ripIDs_xx, ripIDs_yy)
        self.ax1_binnedIDs_ScatterPlot.setData(binnedIDs_xx, binnedIDs_yy)
        self.ax1_ripIDs_ScatterPlot.setData(ripIDs_xx, ripIDs_yy)

    def imgGradLUTfinished_cb(self):
        if self.isSettingChannelLutsFromCode():
            return

        self.df_settings = self.imgGrad.saveState(self.df_settings)
        # LUT of the displayed channel (`gui_navigation.LutGuiElements`)
        self.saveImageChannelLut()
        self.df_settings.to_csv(self.settings_csv_path)

    def ticksCmapMoved(self, gradient):
        pass
        # posData = self.data[self.pos_i]
        # self.setLut(posData, shuffle=False)
        # self.updateLookuptable()

    def updateBkgrColor(self, button):
        color = button.color().getRgb()[:3]
        self.lut[0] = color
        self.updateLookuptable()

    def updateTextLabelsColor(self, button):
        self.ax2_textColor = button.color().getRgb()[:3]
        posData = self.data[self.pos_i]
        if posData.rp is None:
            return

        for obj in posData.rp:
            self.getObjOptsSegmLabels(obj)

    def saveTextLabelsColor(self, button):
        color = button.color().getRgb()[:3]
        self.df_settings.at['labels_text_color', 'value'] = color
        self.df_settings.to_csv(self.settings_csv_path)

    def saveBkgrColor(self, button):
        color = button.color().getRgb()[:3]
        self.df_settings.at['labels_bkgrColor', 'value'] = color
        self.df_settings.to_csv(self.settings_csv_path)
        self.updateAllImages()

    def getImageDataFromFilename(self, filename):
        posData = self.data[self.pos_i]
        if filename == posData.filename:
            return posData.img_data[posData.frame_i]
        else:
            return posData.ol_data_dict.get(filename)

    def get_2Dimg_from_3D(self, imgData, isLayer0=True, frame_i=None, yx=None):
        """Return the 2D image displayed from the 3D `imgData` (slice or
        z-projection depending on the GUI settings).

        If `yx` is not None, return only the value at (y, x) of the 2D image,
        which is much faster than computing the whole 2D image (e.g.,
        a z-projection of the whole z-stack).
        """
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i
        if frame_i < 0:
            frame_i = 0
            frame_i = posData.frame_i = 0
        
        img = self.get_2Dimg_from_switched_plane(imgData, yx=yx)
        if img is not None:
            return img
        
        idx = (posData.filename, frame_i)
        zProjHow_L0 = self.zProjComboBox.currentText()
        if isLayer0:
            try:
                z = posData.segmInfo_df.at[idx, 'z_slice_used_gui']
            except ValueError as e:
                z = posData.segmInfo_df.loc[idx, 'z_slice_used_gui'].iloc[0] 
            zProjHow = zProjHow_L0
        else:
            z = self.zSliceOverlay_SB.sliderPosition()
            zProjHow_L1 = self.zProjOverlay_CB.currentText()
            if zProjHow_L1 == 'same as above': 
                zProjHow = zProjHow_L0
            else:
                zProjHow = zProjHow_L1
        
        if yx is not None:
            # Only the z-column at (y, x)
            imgData = imgData[:, yx[0], yx[1]]
        
        return self.zProjectImage(imgData, zProjHow, z)

    def getRawImageLayer0(self, frame_i):
        posData = self.data[self.pos_i]

        if posData.SizeZ > 1:
            img = posData.img_data[frame_i]
            self.updateZsliceScrollbar(frame_i)
            img = self.get_2Dimg_from_3D(img)
        else:
            img = posData.img_data[frame_i].copy()

        if img.ndim == 2:
            return img
        if img.ndim == 3 and img.shape[-1] in (3, 4):
            return img

        raise ValueError(
            'Raw image for display must be 2D (Y, X) or RGB/A (Y, X, 3 or 4); '
            f'got shape={getattr(img, "shape", None)}, ndim={getattr(img, "ndim", None)} '
            f'for frame_i={frame_i} (metadata SizeT={posData.SizeT}, SizeZ={posData.SizeZ}). '
            'Check that metadata SizeT/SizeZ matches the loaded array (e.g. squeezed TIFF vs CSV).'
        )

    def getImage(self, frame_i=None, raw=False):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i
        
        if raw:
            return self.getRawImageLayer0(frame_i)
        
        if self.viewPreprocDataToggle.isChecked():
            try:
                img = posData.preproc_img_data[frame_i]
                if posData.SizeZ == 1:
                    return np.array(img)
                
                self.updateZsliceScrollbar(frame_i)
                z_slice = self.z_slice_index()
                img = img[z_slice]
                return img
            except Exception as err:
                # self.logger.warning(
                #     'Pre-processed image not existing --> returning raw image'
                # )
                return self.getRawImageLayer0(frame_i)
        
        viewCombinedImageData = (
            self.viewCombineChannelDataToggle.isChecked()
            and self.combineDialog is not None
            and not self.combineDialog.saveAsSegm()
        )
        
        if viewCombinedImageData:
            try:
                img = posData.combine_img_data[frame_i]
                if posData.SizeZ == 1:
                    return np.array(img)
                
                self.updateZsliceScrollbar(frame_i)
                z_slice = self.z_slice_index()
                img = img[z_slice]
                return img
            except Exception as err:
                # self.logger.warning(
                #     'combined image not existing --> returning raw image'
                # )
                return self.getRawImageLayer0(frame_i)
            
        if self.equalizeHistPushButton.isChecked():
            img = posData.equalized_img_data[frame_i]
            if posData.SizeZ == 1:
                return np.array(img)
            
            self.updateZsliceScrollbar(frame_i)
            z_slice = self.z_slice_index()
            img = img[z_slice]
            return img

        return self.getRawImageLayer0(frame_i)

    def setImageImg2(self, updateLookuptable=True, set_image=True):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        # if mode == 'Segmentation and Tracking' or self.isSnapshot:
        #     # self.addExistingDelROIs()
        #     allDelIDs, lab2D = self.getDelROIlab()
        # else:
        lab2D = self.get_2Dlab(posData.lab, force_z=False)
        allDelIDs = set()
        
        self.currentLab2D = lab2D   
        if self.labelsGrad.permanentGreedyCmapAction.isChecked() and updateLookuptable:
            self.greedyShuffleCmap(updateImages=False)
            
        if self.labelsGrad.showLabelsImgAction.isChecked() and set_image:
            self.img2.setImage(lab2D, z=self.z_lab(), autoLevels=False)
        
        if updateLookuptable:
            self.updateLookuptable(delIDs=allDelIDs)

    def getLabelsLayerImage(self, ax=0):
        if ax == 0:
            return self.labelsLayerImg1.image
        else:
            return self.labelsLayerRightImg.image

    def highlightHoverID(self, x, y, hoverID=None):
        if hoverID is None:
            try:
                hoverID = self.currentLab2D[int(y), int(x)]
            except IndexError:
                return

        if hoverID == 0:
            return
        
        posData = self.data[self.pos_i]
        obj = posData.rp.get_obj_from_ID(hoverID)
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
            if not self.isObjInCurrSlice(_obj.bbox):
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
        if not self.isOverlaySegmMaskChecked(ax):
            return
        
        grayedLut = self.grayOutHighlightedLabels()

    def setHighlighedIDfromToolbar(self, ID: int):
        self.findID(ID=ID)

    def highlightSearchedID(self, ID, force=False, greyOthers=True):
        self.highlightIDToolbar.setIDNoSignals(ID)
        
        if ID == 0:
            self.highlightIDToolbar.setVisible(False)
            return

        if ID == self.highlightedID and not force:
            return

        doHighlight = (
            self.propsDockWidget.isVisible() 
            and (
                self.guiTabControl.highlightCheckbox.isChecked()
                or self.guiTabControl.highlightSearchedCheckbox.isChecked()
            )
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

        obj = posData.rp.get_obj_from_ID(ID, warn=False)
        if obj is None:
            return
        
        if not self.isObjInCurrSlice(obj.bbox):
            return
        
        if greyOthers:
            self.textAnnot[0].grayOutAnnotations()
            self.textAnnot[1].grayOutAnnotations()

        isOverlaySegm_ax1 = self.isOverlaySegmMaskChecked(0)
        isOverlaySegm_ax2 = self.isOverlaySegmMaskChecked(1)
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
            display_rp = self.get2DRP()
            obj = display_rp.get_obj_from_ID(ID)
            contours = self.getObjContours(
                obj,
                all_external=True,
                include_internal=self.showAllContoursToggle.isChecked()
            )
            for cont in contours:
                self.searchedIDitemLeft.addPoints(cont[:,0]+0.5, cont[:,1]+0.5)
        
        if isOverlaySegm_ax2:
            self.highLightIDLayerRightImage.setLookupTable(grayedLut)
            self.highLightIDLayerRightImage.setImage(self.highlightedLab)
            self.labelsLayerRightImg.setOpacity(alpha/3)
        else:
            if contours is None:
                display_rp = self.get2DRP()
                obj = display_rp.get_obj_from_ID(ID)
                contours = self.getObjContours(
                    obj,
                    all_external=True,
                    include_internal=self.showAllContoursToggle.isChecked()
                )
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

    def updateLabelsAlpha(self, value):
        self.df_settings.at['overlaySegmMasksAlpha', 'value'] = value
        self.df_settings.to_csv(self.settings_csv_path)
        if self.keepIDsButton.isChecked():
            value = value/3
        self.labelsLayerImg1.setOpacity(value)
        self.labelsLayerRightImg.setOpacity(value)


    def setImageImg1(self, image=None):
        img = self._getImageupdateAllImages(image=image)
        posData = self.data[self.pos_i]
        self.img1.setCurrentPosIndex(self.pos_i)
        self.img1.setCurrentFrameIndex(posData.frame_i)
        if posData.SizeZ > 1:
            zProjHow = self.zProjComboBox.currentText()
            if zProjHow == 'single z-slice':
                z = self.zSliceScrollBar.sliderPosition()
            else:
                z = zProjHow
            
            self.img1.setCurrentZsliceIndex(z)

        self.img1.setImage(
            img, next_frame_image=self.nextFrameImage(),
            scrollbar_value=posData.frame_i+2
        )

    def addObjContourToContoursImage(
            self, obj=None, ax=0, thickness=None, color=None,
            force=False
        ):   
        imageItem = self.getContoursImageItem(ax, force=force)
        if imageItem is None:
            return
    
        contours = self.getObjContours(
            obj,
            all_external=True,
            include_internal=self.showAllContoursToggle.isChecked()
        )
        if thickness is None:
            thickness = self.contLineWeight
        if color is None:
            color = self.contLineColor
        
        self.setContoursImage(imageItem, contours, thickness, color)

    def clearObjContour(
            self, ID=0, obj=None, ax=0, debug=False, updateImage=True
        ):
        imageItem = self.getContoursImageItem(ax)
        if imageItem is None:
            return

        if ID > 0:
            self.contoursImage[self.currentLab2D==ID] = [0,0,0,0]
        else:
            obj_slice = self.getObjSlice(obj.slice)
            obj_image = self.getObjImage(obj.image, obj.bbox)
            self.contoursImage[obj_slice][obj_image] = [0,0,0,0]
        
        if not updateImage:
            return
        
        imageItem.setImage(self.contoursImage)        

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

    def updateAllImages(
            self, image=None, computePointsLayers=True, computeContours=True,
            updateLookuptable=True
        ):
        self.clearAllItems()

        posData = self.data[self.pos_i]

        self.last_pos_i = self.pos_i
        self.last_frame_i = posData.frame_i
        
        self.rescaleIntensitiesLut(setImage=False)

        self.setImageImg1(image=image)       
        self.setImageImg2(updateLookuptable=updateLookuptable)
        
        self.setOverlayImages()

        self.setOverlayLabelsItems()
        self.setOverlaySegmMasks()
              
        if self.slideshowWin is not None:
            self.slideshowWin.frame_i = posData.frame_i
            self.slideshowWin.update_img()

        # self.update_rp()

        # Annotate ID and draw contours
        self.setAllTextAnnotations()    
        self.setAllContoursImages(
            compute=False
        )

        self.drawAllMothBudLines()
        if self.isLineageTreeLinesModeActive():
            self.drawAllLineageTreeLines()

        self.highlightLostNew()      

        if self.ccaTableWin is not None: # need to add for lin tree, later
            zoomIDs = self.getZoomIDs()
            self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)

        self.doCustomAnnotation(0)

        self.annotate_rip_and_bin_IDs()
        self.updateTempLayerKeepIDs()
        self.whitelistUpdateTempLayer()
        self.drawPointsLayers(computePointsLayers=computePointsLayers)
        self.setBrushStrokesLayerImage()
        self.drawMagicPromptsBoxes()
        self.setManualBackgroundImage()
        self.annotateAssignedObjsAcdcTrackerSecondStep()
        
        self.highlightSearchedID(self.highlightedID, force=True) 
        self.updateTimestampFrame()
        
        self.annotateAllObjectTracks()
        
        posData.visited = True

    def removeAllItems(self):
        self.ax1.clear()
        self.ax2.clear()
        try:
            self.chNamesQActionGroup.removeAction(self.userChNameAction)
        except Exception as e:
            pass
        try:
            posData = self.data[self.pos_i]
            for action in self.fluoDataChNameActions:
                self.chNamesQActionGroup.removeAction(action)
        except Exception as e:
            pass
        try:
            self.overlayButton.setChecked(False)
        except Exception as e:
            pass

        if hasattr(self, 'contoursImage'):
            self.initContoursImage()

    def restoreDefaultColors(self):
        try:
            color = self.defaultToolBarButtonColor
            self.overlayButton.setStyleSheet(f'background-color: {color}')
        except AttributeError:
            # traceback.print_exc()
            pass

    def changeTextResolution(self):
        mode = 'high' if self.highLowResAction.isChecked() else 'low'
        self.logger.info(
            f'Switching to {mode} for the text annnotations...'
        )
        if not self.isDataLoaded:
            return
        
        self.setAllIDs()
        posData = self.data[self.pos_i]
        allIDs = posData.allIDs
        img_shape = self.img1.image.shape[:2]
        scalingMode = self.scalingLabelsAction.isChecked()
        self.textAnnot[0].changeResolution(mode, allIDs, self.ax1, img_shape, 
                                           scalingMode=scalingMode)
        self.textAnnot[1].changeResolution(mode, allIDs, self.ax2, img_shape, 
                                           scalingMode=scalingMode)
        self.updateAllImages()

    def highLowResToggled(self, clicked=True):
        self.changeTextResolution()
        self.df_settings.at[
            'highLowRes', 'value'
            ] = self.highLowResAction.isChecked()
        self.df_settings.to_csv(self.settings_csv_path)

    def scalingLabelsToggled(self, clicked=True):
        self.df_settings.at[
            'scalingLabels', 'value'
            ] = self.scalingLabelsAction.isChecked()
        self.df_settings.to_csv(self.settings_csv_path)
        if not self.isDataLoaded:
            return
        for annot in self.textAnnot.values():
            annot.setScaling(self.scalingLabelsAction.isChecked())

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

    def storeDefaultAndCustomColors(self):
        c = self.overlayButton.palette().button().color().name()
        self.defaultToolBarButtonColor = c
        self.doublePressKeyButtonColor = '#fa693b'


class TitleStatusGuiElements:
    """Title label with clickable IDs and its popup
    (TitleIDsPopupEventFilter), and the status bar label.
    """

    def setStatusBarLabel(self, log=True):
        self.statusbar.clearMessage()
        posData = self.data[self.pos_i]
        segmentedChannelname = posData.filename[len(posData.basename):]
        segmFilename = os.path.basename(posData.segm_npz_path)
        segmEndName = segmFilename[len(posData.basename):]
        txt = (
            f'{posData.pos_foldername} || '
            f'Basename: {posData.basename} || '
            f'Segmented channel: {segmentedChannelname} || '
            f'Segmentation file name: {segmEndName}'
        )
        mode = str(self.modeComboBox.currentText())
        if log:
            self.logger.info(txt)
        self.statusBarLabel.setText(txt)

    def setLookingGoodTitle(self):
        self._titleIDsPopupHtmlByLink.clear()
        self.titleIDsPopup.hide()
        htmlTxt = (
            f'<a href="cellacdc-fireworks" '
            f'style="color: {self.titleColor};">Looking good!</a>'
        )
        self.titleLabel.setText(htmlTxt)


    def titleLabelIDLinkActivated(self, link):
        """ 
        Handle clicks on title label ID links.
        """
        if link == 'cellacdc-fireworks':
            self.fireworksOverlay.start()
            return
        if link.startswith('cellacdc-category:'):
            self._showTitleIDsPopup(link)
            return
        prefix = 'cellacdc-id:'
        if not link.startswith(prefix):
            return
        self.titleIDsPopup.hide()
        ID_text = link[len(prefix):].split(':', 1)[0]
        self.findID(ID=int(ID_text))

    def titleLabelIDLinkHovered(self, link):
        self._setLinkCursor(self.titleLabel.item, link)
        if not link:
            self._scheduleTitleIDsPopupHide()
            return
        self._showTitleIDsPopup(link)

    def titleIDsPopupLinkHovered(self, link):
        self._setLinkCursor(self.titleIDsPopup, link)

    def _setLinkCursor(self, target, link):
        cursor = Qt.PointingHandCursor if link else Qt.ArrowCursor
        target.setCursor(QCursor(cursor))

    def _showTitleIDsPopup(self, link):
        popup_html = self._titleIDsPopupHtmlByLink.get(link)
        if popup_html is None:
            return
        self.titleIDsPopupHideTimer.stop()
        self.titleIDsPopup.setText(popup_html)
        self.titleIDsPopup.adjustSize()
        self.titleIDsPopup.move(QCursor.pos() + QPoint(12, 12))
        self.titleIDsPopup.show()

    def _keepTitleIDsPopupOpen(self):
        self.titleIDsPopupHideTimer.stop()

    def _scheduleTitleIDsPopupHide(self):
        self.titleIDsPopupHideTimer.start(200)

    def _titleClickableIDs(self, IDs, color, category_key):
        """Generate clickable HTML links for title IDs. 
        Handles both IDs and "..."

        Parameters
        ----------
        IDs : list of int
            List of IDs to generate clickable links for.
        color : str
            Color to use for the HTML links.
        category_key : int
            Key representing the category for the IDs.

        Returns
        -------
        str
            HTML string containing clickable links for the IDs.
        """
        IDs_text = str(IDs)
        IDs_html = re.sub(
            r'(?<![\w.])-?\d+',
            lambda match: (
                f'<a href="cellacdc-id:{match.group(0)}:{category_key}" '
                f'style="color: {color};">{match.group(0)}</a>'
            ),
            IDs_text,
        )
        category_link = f'cellacdc-category:{category_key}'
        return IDs_html.replace(
            '...',
            f'<a href="{category_link}" style="color: {color};">...</a>',
        )

    def _wrapTitleIDs(self, IDs, width=60):
        "Wrapping for tooltip."
        return '<br>'.join(textwrap.wrap(
            str(IDs), width=width, break_long_words=False,
            break_on_hyphens=False,
        ))

    def _normalizeTitleIDs(self, IDs):
        """Normalize a list of title IDs, converting numpy integers and 
        formatted strings to standard integers or strings.

        Parameters
        ----------
        IDs : list
            List of title IDs to normalize.

        Returns
        -------
        list
            List of normalized title IDs.
        """
        normalized_IDs = []
        for ID in IDs:
            if isinstance(ID, np.integer):
                normalized_IDs.append(int(ID))
                continue
            ID_text = re.sub(
                r'np\.(?:u?int\d*)\((-?\d+)\)', r'\1', str(ID)
            )
            normalized_IDs.append(ID_text if isinstance(ID, str) else int(ID_text))
        return normalized_IDs

    def setTitleFormatter(self, htmlTxt_li, htmlTxtFull_li, pretxt, color, IDs):
        if not IDs:
            return htmlTxt_li, htmlTxtFull_li
        
        if isinstance(IDs, set):
            IDs = list(IDs)

        IDs = self._normalizeTitleIDs(IDs)
        category_key = len(htmlTxtFull_li)
        trim_IDs = myutils.get_trimmed_list(IDs, max_num_digits=20)
        txt = (
            f'{pretxt}: '
            f'{self._titleClickableIDs(trim_IDs, color, category_key)}'
        )
        wrapped_IDs = self._wrapTitleIDs(IDs)
        txt_full = (
            f'{pretxt}:<br>'
            f'{self._titleClickableIDs(wrapped_IDs, color, category_key)}'
        )

        txt = f'<font color="{color}">{txt}</font>'
        txt_full = f'<font color="{color}">{txt_full}</font>'

        # Generate and store HTML links for the title IDs. This is used for
        # displaying the tooltip
        link_pattern = rf'cellacdc-id:-?\d+:{category_key}'
        for link in re.findall(link_pattern, txt):
            self._titleIDsPopupHtmlByLink[link] = txt_full
        for link in re.findall(link_pattern, txt_full):
            self._titleIDsPopupHtmlByLink[link] = txt_full
        category_link = f'cellacdc-category:{category_key}'
        self._titleIDsPopupHtmlByLink[category_link] = txt_full

        htmlTxt_li.append(txt)
        htmlTxtFull_li.append(txt_full)

        return htmlTxt_li, htmlTxtFull_li

    def setTitleText(   
            self, lost_IDs=None, new_IDs=None, IDs_with_holes=None, 
            tracked_lost_IDs=None
        ):
        self._titleIDsPopupHtmlByLink.clear()
        self.titleIDsPopup.hide()
        if self.annotateSingleMotherBudPairButton.isChecked():
            mothID = self.annotateSingleMothBudPairState.get(
                'mother_ID'
            )
            budID = self.annotateSingleMothBudPairState.get(
                'bud_ID'
            )
            frame_to_restore = self.annotateSingleMothBudPairState.get(
                'frame_i_to_restore'
            )
            htmlTxt, htmlTxtFull = self.setTitleFormatter(
                [], [], 'Annotating mother-bud pair: Mother', 'orange', [mothID]
            )
            htmlTxt, htmlTxtFull = self.setTitleFormatter(
                htmlTxt, htmlTxtFull, 'Bud', 'orange', [budID]
            )
            htmlTxt = ", ".join(htmlTxt)
            htmlTxt = (
                htmlTxt 
                + f'<font color="orange"> since frame n. {frame_to_restore+1}</font>'
                )
            self.titleLabel.setText(htmlTxt)
            return
        
        if self.manualAnnotPastButton.isChecked():
            lockedID = self.editIDspinbox.value()
            frame_to_restore = self.manualAnnotState.get('frame_i_to_restore')
            htmlTxt, htmlTxtFull = self.setTitleFormatter(
                [], [], 'Locked ID', 'orange', [lockedID]
            )
            htmlTxt = htmlTxt[0]
            htmlTxt = (
                htmlTxt 
                + f'<font color="orange"> since frame n. {frame_to_restore+1}</font>'
                )
            self.titleLabel.setText(htmlTxt)
            return
        
        mode = self.modeComboBox.currentText()
        try:
            posData = self.data[self.pos_i]
            posData.segm_data[posData.frame_i]
            prev_segmented = True
        except IndexError:
            prev_segmented = False
            
        if prev_segmented:
            htmlTxt_li = []
            htmlTxtFull_li = []
        else:
            htmlTxt = f'<font color="white">Never segmented frame. </font>'
            self.titleLabel.setText(htmlTxt)
            self.titleLabel.setToolTip(htmlTxt)
            return
        
        if mode != 'Normal division: Lineage tree':
            htmlTxt_li, htmlTxtFull_li = self.setTitleFormatter(
                htmlTxt_li, htmlTxtFull_li, 'IDs lost', 'orange', lost_IDs
            )
            htmlTxt_li, htmlTxtFull_li = self.setTitleFormatter(
                htmlTxt_li, htmlTxtFull_li, 'New IDs', 'red', new_IDs
            )
            htmlTxt_li, htmlTxtFull_li = self.setTitleFormatter(
                htmlTxt_li, htmlTxtFull_li, 'Acc. IDs lost', 'green', 
                tracked_lost_IDs
            )

            for i, htmlTxtFull in enumerate(htmlTxtFull_li):
                htmlTxtFull_li[i] = htmlTxtFull.replace('Acc.', 'Accepted')

            htmlTxt_li, htmlTxtFull_li = self.setTitleFormatter(
                htmlTxt_li, htmlTxtFull_li, 'IDs with holes', 'red', 
                IDs_with_holes
            )
        else:
            try:
                cells_with_parent, orphan_cells, lost_cells = self.lineage_tree.export_lin_tree_info(posData.frame_i)
            except IndexError or KeyError:
                title = 'Processing lineage tree...'
                htmlTxt = f'<font color="{self.titleColor}">{title}</font>'
                self.titleLabel.setText(htmlTxt)
                self.titleLabel.setToolTip(htmlTxt)
                return
            except AttributeError:
                title = 'Lineage tree still initializing...'
                htmlTxt = f'<font color="{self.titleColor}">{title}</font>'
                self.titleLabel.setText(htmlTxt)
                self.titleLabel.setToolTip(htmlTxt)
                return
            
            parent_cell_txt_raw = []
            if cells_with_parent:
                # aggregate same parents
                parent_cell_groups = dict()
                for cell, parent in cells_with_parent:
                    if parent not in parent_cell_groups:
                        parent_cell_groups[parent] = []
                    parent_cell_groups[parent].append(cell)
                for parent, daughters in parent_cell_groups.items():
                    cells_str = ','.join([str(daughter) for daughter in daughters])
                    parent_cell_txt_raw.append(f'({parent}>{cells_str})')

            htmlTxt_li, htmlTxtFull_li = self.setTitleFormatter(
                htmlTxt_li, htmlTxtFull_li, 'New w/out P', 'red', 
                orphan_cells
            )
            htmlTxt_li, htmlTxtFull_li = self.setTitleFormatter(
                htmlTxt_li, htmlTxtFull_li, 'Lost', 'yellow', lost_cells
            )
            htmlTxt_li, htmlTxtFull_li = self.setTitleFormatter(
                htmlTxt_li, htmlTxtFull_li, 'P > D', 'green', 
                parent_cell_txt_raw
            )

        if not htmlTxt_li:
            self.setLookingGoodTitle()
            return

        htmlTxt = ', '.join(htmlTxt_li)
        self.titleLabel.setText(htmlTxt)
        self.titleLabel.setToolTip('')
