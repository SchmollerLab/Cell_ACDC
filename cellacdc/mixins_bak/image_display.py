"""Qt view adapter for image display, LUT, and cursor workflows."""

from __future__ import annotations

from functools import partial

import numpy as np
import pyqtgraph as pg
import skimage.exposure
import skimage.measure
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QAction, QActionGroup

from cellacdc import (
    apps,
    darkBkgrColor,
    disableWindow,
    exception_handler,
    graphLayoutBkgrColor,
    myutils,
    settings_csv_path,
)


class ImageDisplayMixin:
    """Qt-facing adapter for image display, LUT, and cursor workflows."""

    """Headless display settings and image-display rules."""

    # @exec_time

    # @exec_time

    # @exec_time

    def RGBtoGray(self, R, G, B):
        # see https://stackoverflow.com/questions/17615963/standard-rgb-to-grayscale-conversion
        C_linear = (0.2126 * R + 0.7152 * G + 0.0722 * B) / 255
        if C_linear <= 0.0031309:
            gray = 12.92 * C_linear
        else:
            gray = 1.055 * (C_linear) ** (1 / 2.4) - 0.055
        return gray

    def _getImageupdateAllImages(self, image=None):
        if image is not None:
            return image

        img = self.getImage()
        return img

    def activeBrushCircleCursors(self, isHoverImg1):
        if self.showMirroredCursorAction.isChecked():
            return self.ax1_BrushCircle, self.ax2_BrushCircle

        if isHoverImg1:
            return (self.ax1_BrushCircle,)
        else:
            return (self.ax2_BrushCircle,)

    def activeEraserCircleCursors(self, isHoverImg1):
        if self.showMirroredCursorAction.isChecked():
            return self.ax1_EraserCircle, self.ax2_EraserCircle

        if isHoverImg1:
            return (self.ax1_EraserCircle,)
        else:
            return (self.ax2_EraserCircle,)

    def activeEraserXCursors(self, isHoverImg1):
        if self.showMirroredCursorAction.isChecked():
            return self.ax1_EraserX, self.ax2_EraserX

        if isHoverImg1:
            return (self.ax1_EraserX,)
        else:
            return (self.ax2_EraserX,)

    def addFontSizeActions(self, menu, slot):
        fontActionGroup = QActionGroup(self)
        fontActionGroup.setExclusive(True)
        for fontSize in range(4, 27):
            action = QAction(self)
            action.setText(str(fontSize))
            action.setCheckable(True)
            if fontSize == self.fontSize:
                action.setChecked(True)
            fontActionGroup.addAction(action)
            menu.addAction(action)
            action.triggered.connect(slot)
        return fontActionGroup

    def autoRange(self):
        if self.labelsGrad.showLabelsImgAction.isChecked():
            self.ax2.autoRange()
        self.ax1.autoRange()

    @exception_handler
    def changeFontSize(self):
        fontSize = self.fontSizeSpinBox.value()
        if fontSize == self.fontSize:
            return

        self.fontSize = fontSize

        self.df_settings.at["fontSize", "value"] = self.fontSize
        self.df_settings.to_csv(self.settings_csv_path)

        self.setAllIDs()
        self.data[self.pos_i]
        for ax in range(2):
            self.textAnnot[ax].changeFontSize(self.fontSize)
        if self.highLowResAction.isChecked():
            self.setAllTextAnnotations()
        else:
            self.updateAllImages()

    def clearCursors(self):
        self.ax1_cursor.setData([], [])
        self.ax2_cursor.setData([], [])
        self.setHoverToolSymbolData(
            [],
            [],
            (self.ax2_BrushCircle, self.ax1_BrushCircle),
        )
        eraserCursors = (
            self.ax1_EraserCircle,
            self.ax2_EraserCircle,
            self.ax1_EraserX,
            self.ax2_EraserX,
        )
        self.setHoverToolSymbolData([], [], eraserCursors)

    def customLevelsLutChanged(self, levels, imageItem=None):
        imageItem.setLevels(levels)

    def editImgProperties(self, checked=True):
        posData = self.data[self.pos_i]
        posData.askInputMetadata(
            len(self.data),
            ask_SizeT=True,
            ask_TimeIncrement=True,
            ask_PhysicalSizes=True,
            save=True,
            singlePos=True,
            askSegm3D=False,
        )
        if hasattr(self, "timestamp"):
            self.timestamp.setSecondsPerFrame(posData.TimeIncrement)
            self.updateTimestampFrame()

        if hasattr(self, "scaleBar"):
            self.scaleBar.updatePhysicalLength(posData.PhysicalSizeX)

    def enableZstackWidgets(self, enabled):
        if enabled:
            myutils.setRetainSizePolicy(self.zSliceScrollBar)
            myutils.setRetainSizePolicy(self.zProjComboBox)
            myutils.setRetainSizePolicy(self.zSliceOverlay_SB)
            myutils.setRetainSizePolicy(self.zProjOverlay_CB)
            myutils.setRetainSizePolicy(self.overlay_z_label)
            self.zSliceScrollBar.setDisabled(False)
            self.zProjComboBox.show()
            if self.data[self.pos_i].SizeT > 1:
                self.zProjLockViewButton.show()
            self.zSliceScrollBar.show()
            self.zSliceCheckbox.show()
            self.zSliceSpinbox.show()
            self.switchPlaneCombobox.show()
            self.switchPlaneCombobox.setDisabled(False)
            self.SizeZlabel.show()
        else:
            myutils.setRetainSizePolicy(self.zSliceScrollBar, retain=False)
            myutils.setRetainSizePolicy(self.zProjComboBox, retain=False)
            myutils.setRetainSizePolicy(self.zSliceOverlay_SB, retain=False)
            myutils.setRetainSizePolicy(self.zProjOverlay_CB, retain=False)
            myutils.setRetainSizePolicy(self.overlay_z_label, retain=False)
            self.zSliceScrollBar.setDisabled(True)
            self.zProjComboBox.hide()
            self.zProjComboBox.hide()
            self.zSliceScrollBar.hide()
            self.zSliceCheckbox.hide()
            self.zSliceSpinbox.hide()
            self.SizeZlabel.hide()
            self.switchPlaneCombobox.hide()
            self.switchPlaneCombobox.setDisabled(True)

        self.imgGrad.rescaleAcrossZstackAction.setDisabled(not enabled)
        for ch, overlayItems in self.overlayLayersItems.items():
            lutItem = overlayItems[1]
            lutItem.rescaleAcrossZstackAction.setDisabled(not enabled)

    @disableWindow
    def equalizeHist(self, checked=True):
        self.img1.useEqualized = checked

        if not checked:
            self.updateAllImages()
            return

        self.logger.info("Equalizing image histogram...")
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

    def getCheckNormAction(self):
        normalize = False
        how = ""
        for action in self.normalizeQActionGroup.actions():
            if action.isChecked():
                how = action.text()
                normalize = True
                break
        return action, normalize, how

    def getContoursImageItem(self, ax, force=False):
        if not self.areContoursRequested(ax) and not force:
            return

        if ax == 0:
            return self.ax1_contoursImageItem
        else:
            return self.ax2_contoursImageItem

    def getDisplayedImg1(self):
        return self.img1.image

    def getDisplayedZstack(self):
        posData = self.data[self.pos_i]
        return posData.img_data[posData.frame_i]

    def getDistantGray(self, desiredGray, bkgrGray):
        isDesiredSimilarToBkgr = abs(desiredGray - bkgrGray) < 0.3
        if isDesiredSimilarToBkgr:
            return 1 - desiredGray
        else:
            return desiredGray

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
            except Exception:
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
            except Exception:
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

    def getImageDataFromFilename(self, filename):
        posData = self.data[self.pos_i]
        if filename == posData.filename:
            return posData.img_data[posData.frame_i]
        else:
            return posData.ol_data_dict.get(filename)

    def getLostObjImageItem(self, ax):
        if ax == 0:
            return self.ax1_lostObjImageItem
        else:
            return self.ax1_lostTrackedObjImageItem

    def getLostTrackedObjImageItem(self, ax):
        if ax == 0:
            return self.ax1_lostTrackedObjImageItem
        else:
            return self.ax2_lostTrackedObjImageItem

    def getObjBbox(self, obj_bbox):
        if self.isSegm3D and len(obj_bbox) == 6:
            obj_bbox = (obj_bbox[1], obj_bbox[2], obj_bbox[4], obj_bbox[5])
            return obj_bbox
        else:
            return obj_bbox

    def getObjImage(self, obj_image, obj_bbox, z_slice=None):
        if self.isSegm3D and len(obj_bbox) == 6:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == "single z-slice"
            if not isZslice:
                # required a projection
                return obj_image.max(axis=0)

            min_z = obj_bbox[0]
            if z_slice is None:
                z_slice = self.z_lab()
            if isinstance(z_slice, tuple):
                z_slice = z_slice[-1]

            local_z = z_slice - min_z
            try:
                obi_image_2d = obj_image[local_z]
            except Exception:
                obi_image_2d = None
            return obi_image_2d
        else:
            return obj_image

    def getObjSlice(self, obj_slice):
        if self.isSegm3D:
            return obj_slice[1:3]
        else:
            return obj_slice

    def getObject2DimageFromZ(self, z, obj):
        posData = self.data[self.pos_i]
        z_min = obj.bbox[0]
        local_z = z - z_min
        if local_z >= posData.SizeZ or local_z < 0:
            return
        return obj.image[local_z]

    def getObject2DsliceFromZ(self, z, obj):
        posData = self.data[self.pos_i]
        z_min = obj.bbox[0]
        local_z = z - z_min
        if local_z >= posData.SizeZ or local_z < 0:
            return
        return obj.image[local_z]

    def getPreComputedMinMaxZstack(self, channel: str):
        if channel != self.user_ch_name:
            return None

        posData = self.data[self.pos_i]
        zstack_min, zstack_max = np.inf, 0
        for z in range(posData.SizeZ):
            key = (self.pos_i, posData.frame_i, z)
            levels = self.img1.minMaxValuesMapper.get(key)
            if levels is None:
                return

            img_min, img_max = levels
            if img_min < zstack_min:
                zstack_min = img_min

            if img_max > zstack_max:
                zstack_max = img_max

        return (zstack_min, zstack_max)

    def getRawImage(self, frame_i=None, filename=None):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i
        if filename is None:
            rawImgData = posData.img_data[frame_i]
            isLayer0 = True
        else:
            rawImgData = posData.ol_data[filename][frame_i]
            isLayer0 = False
        if posData.SizeZ > 1:
            rawImg = self.get_2Dimg_from_3D(rawImgData, isLayer0=isLayer0)
        else:
            rawImg = rawImgData
        return rawImg

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
            "Raw image for display must be 2D (Y, X) or RGB/A (Y, X, 3 or 4); "
            f"got shape={getattr(img, 'shape', None)}, ndim={getattr(img, 'ndim', None)} "
            f"for frame_i={frame_i} (metadata SizeT={posData.SizeT}, SizeZ={posData.SizeZ}). "
            "Check that metadata SizeT/SizeZ matches the loaded array (e.g. squeezed TIFF vs CSV)."
        )

    def get_2Dimg_from_3D(self, imgData, isLayer0=True, frame_i=None):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i
        if frame_i < 0:
            frame_i = 0
            frame_i = posData.frame_i = 0

        axis_slice = self.zSliceScrollBar.sliderPosition()
        if self.switchPlaneCombobox.depthAxes() == "x":
            return imgData[:, :, axis_slice].copy()
        elif self.switchPlaneCombobox.depthAxes() == "y":
            return imgData[:, axis_slice].copy()

        idx = (posData.filename, frame_i)
        zProjHow_L0 = self.zProjComboBox.currentText()
        if isLayer0:
            try:
                z = posData.segmInfo_df.at[idx, "z_slice_used_gui"]
            except ValueError:
                z = posData.segmInfo_df.loc[idx, "z_slice_used_gui"].iloc[0]
            zProjHow = zProjHow_L0
        else:
            z = self.zSliceOverlay_SB.sliderPosition()
            zProjHow_L1 = self.zProjOverlay_CB.currentText()
            if zProjHow_L1 == "same as above":
                zProjHow = zProjHow_L0
            else:
                zProjHow = zProjHow_L1

        if zProjHow == "single z-slice":
            img = imgData[z]  # .copy()
        elif zProjHow == "max z-projection":
            img = imgData.max(axis=0)
        elif zProjHow == "mean z-projection":
            img = imgData.mean(axis=0)
        elif zProjHow == "median z-proj.":
            img = np.median(imgData, axis=0)
        return img

    def get_2Dlab(self, lab, force_z=True):
        if self.isSegm3D:
            if force_z:
                return lab[self.z_lab()]
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == "single z-slice"
            if isZslice:
                return lab[self.z_lab()]
            else:
                return lab.max(axis=0)
        else:
            return lab

    def get_2Drp(self, lab=None):
        if self.isSegm3D:
            if lab is None:
                # self.currentLab2D is defined at self.setImageImg2()
                lab = self.currentLab2D
            lab = self.get_2Dlab(lab)
            rp = skimage.measure.regionprops(lab)
            return rp
        else:
            return self.data[self.pos_i].rp

    def initContoursImage(self):
        posData = self.data[self.pos_i]
        z_slice = self.z_lab()
        img = posData.img_data[posData.frame_i]
        Y, X = img[z_slice].shape[-2:]

        self.contoursImage = np.zeros((Y, X, 4), dtype=np.uint8)

    def initImgCmap(self):
        if "img_cmap" not in self.df_settings.index:
            self.df_settings.at["img_cmap", "value"] = "grey"
        self.imgCmapName = self.df_settings.at["img_cmap", "value"]
        self.imgCmap = self.imgGrad.cmaps[self.imgCmapName]
        if self.imgCmapName != "grey":
            # To ensure mapping to colors we need to normalize image
            self.normalizeByMaxAction.setChecked(True)

    def initImgGradRescaleIntensitiesHowPreference(self):
        posData = self.data[self.pos_i]
        channelName = posData.user_ch_name
        if f"how_rescale_intensities_{channelName}" not in self.df_settings.index:
            return

        how = self.df_settings.at[f"how_rescale_intensities_{channelName}", "value"]
        self.imgGrad.setRescaleIntensitiesHow(how)

    def initLostObjContoursImage(self):
        posData = self.data[self.pos_i]
        z_slice = self.z_lab()
        img = posData.img_data[posData.frame_i]
        Y, X = img[z_slice].shape[-2:]

        self.lostObjContoursImage = np.zeros((Y, X, 4), dtype=np.uint8)

    def initLostTrackedObjContoursImage(self):
        posData = self.data[self.pos_i]
        z_slice = self.z_lab()
        img = posData.img_data[posData.frame_i]
        Y, X = img[z_slice].shape[-2:]

        self.lostTrackedObjContoursImage = np.zeros((Y, X, 4), dtype=np.uint8)

    def initManualBackgroundImage(self):
        posData = self.data[self.pos_i]
        if hasattr(posData, "lab"):
            Y, X = posData.lab.shape[-2:]
        else:
            Y, X = posData.img_data.shape[-2:]
        if not hasattr(self, "manualBackgroundTextItems"):
            self.manualBackgroundTextItems = {}
        posData.manualBackgroundImage = np.zeros((Y, X, 4), dtype=np.uint8)
        if posData.manualBackgroundLab is None:
            posData.manualBackgroundLab = np.zeros((Y, X), dtype=np.uint32)

    def initTextAnnot(self, force=False):
        posData = self.data[self.pos_i]
        if hasattr(posData, "lab"):
            Y, X = posData.lab.shape[-2:]
        else:
            Y, X = posData.img_data.shape[-2:]
        self.textAnnot[0].initItem((Y, X))
        self.textAnnot[1].initItem((Y, X))

    def intensity_normalization_setting_value(self, how: str) -> str:
        return how

    def invertBw(self, checked, update=True):
        self.invertBwAlreadyCalledOnce = True

        try:
            self.labelsGrad.invertBwAction.toggled.disconnect()
        except Exception:
            pass

        self.labelsGrad.invertBwAction.setChecked(checked)
        self.labelsGrad.invertBwAction.toggled.connect(self.setCheckedInvertBW)

        try:
            self.imgGrad.invertBwAction.toggled.disconnect()
        except Exception:
            pass
        self.imgGrad.invertBwAction.setChecked(checked)
        self.imgGrad.invertBwAction.toggled.connect(self.setCheckedInvertBW)

        self.imgGrad.setInvertedColorMaps(checked)
        self.imgGrad.invertCurrentColormap(checked)

        self.imgGradRight.setInvertedColorMaps(checked)
        self.imgGradRight.invertCurrentColormap(checked)

        if hasattr(self, "overlayLayersItems"):
            for items in self.overlayLayersItems.values():
                lutItem = items[1]
                lutItem.invertBwAction.toggled.disconnect()
                lutItem.invertBwAction.setChecked(checked)
                lutItem.invertBwAction.toggled.connect(self.setCheckedInvertBW)
                lutItem.setInvertedColorMaps(checked)

        if self.slideshowWin is not None:
            self.slideshowWin.is_bw_inverted = checked
            self.slideshowWin.update_img()
        self.df_settings.at["is_bw_inverted", "value"] = "Yes" if checked else "No"
        self.df_settings.to_csv(self.settings_csv_path)
        if checked:
            # Light mode
            self.equalizeHistPushButton.setStyleSheet("")
            self.graphLayout.setBackground(graphLayoutBkgrColor)
            self.ax2_BrushCirclePen = pg.mkPen((150, 150, 150), width=2)
            self.ax2_BrushCircleBrush = pg.mkBrush((200, 200, 200, 150))
            self.titleColor = "black"
        else:
            # Dark mode
            self.equalizeHistPushButton.setStyleSheet(
                "QPushButton {background-color: #282828; color: #F0F0F0;}"
            )
            self.graphLayout.setBackground(darkBkgrColor)
            self.ax2_BrushCirclePen = pg.mkPen(width=2)
            self.ax2_BrushCircleBrush = pg.mkBrush((255, 255, 255, 50))
            self.titleColor = "white"

        if not hasattr(self, "textAnnot"):
            return

        self.textAnnot[0].invertBlackAndWhite()
        self.textAnnot[1].invertBlackAndWhite()

        self.objLabelAnnotRgb = tuple(self.textAnnot[0].item.colors()["label"][:3])
        self.textIDsColorButton.setColor(self.objLabelAnnotRgb)
        self.imgGrad.textColorButton.setColor(self.objLabelAnnotRgb)
        for items in self.overlayLayersItems.values():
            lutItem = items[1]
            lutItem.textColorButton.setColor(self.objLabelAnnotRgb)

        if update:
            self.updateAllImages()

    def invert_bw_setting_value(self, checked: bool) -> str:
        return "Yes" if checked else "No"

    def isObjVisible(self, obj_bbox, debug=False, z_slice=None):
        if z_slice is None:
            z_slice = self.z_lab()

        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == "single z-slice"
            if not isZslice:
                # required a projection --> all obj are visible
                return True

            depthAxes = self.switchPlaneCombobox.depthAxes()

            min_z, min_y, min_x, max_z, max_y, max_x = obj_bbox
            if depthAxes == "z":
                min_val, max_val = min_z, max_z
                val = z_slice
            elif depthAxes == "y":
                min_val, max_val = min_y, max_y
                val = z_slice[-1]
            else:
                min_val, max_val = min_x, max_x
                val = z_slice[-1]

            if val >= min_val and val < max_val:
                return True
            else:
                return False
        else:
            return True

    def labels_alpha_plan(
        self,
        value: float,
        *,
        keep_ids_checked: bool,
    ) -> LabelsAlphaPlan:
        opacity = value / 3 if keep_ids_checked else value
        return LabelsAlphaPlan(setting_value=value, opacity=opacity)

    def launchSlideshow(self):
        posData = self.data[self.pos_i]
        self.determineSlideshowWinPos()
        if self.slideshowButton.isChecked():
            self.slideshowWin = apps.imageViewer(
                parent=self,
                button_toUncheck=self.slideshowButton,
                linkWindow=posData.SizeT > 1,
                enableOverlay=True,
                enableMirroredCursor=True,
            )
            self.slideshowWin.img.minMaxValuesMapper = self.img1.minMaxValuesMapper
            self.slideshowWin.img.setCurrentPosIndex(self.pos_i)
            h = self.drawIDsContComboBox.size().height()
            self.slideshowWin.framesScrollBar.setFixedHeight(h)
            self.slideshowWin.overlayButton.setChecked(self.overlayButton.isChecked())
            self.slideshowWin.sigHoveringImage.connect(
                self.setMirroredCursorFromSecondWindow
            )
            if posData.SizeZ > 1:
                z_slice = self.zSliceScrollBar.sliderPosition()
                self.slideshowWin.img.setCurrentZsliceIndex(z_slice)
                self.slideshowWin.zSliceScrollBar.setSliderPosition(z_slice)
                self.slideshowWin.z_label.setText(
                    f"z-slice  {z_slice + 1:02}/{posData.SizeZ}"
                )
            self.slideshowWin.update_img()
            self.slideshowWin.show(left=self.slideshowWinLeft, top=self.slideshowWinTop)
        else:
            self.slideshowWin.close()
            self.slideshowWin = None

    def normaliseIntensitiesActionTriggered(self, action):
        how = action.text()
        self.df_settings.at["how_normIntensities", "value"] = how
        self.df_settings.to_csv(self.settings_csv_path)
        self.updateAllImages()
        self.updateImageValueFormatter()

    def normalizeIntensities(self, img):
        action, normalize, how = self.getCheckNormAction()
        if not normalize:
            return img

        if how == "Do not normalize. Display raw image":
            img = img
        elif how == "Convert to floating point format with values [0, 1]":
            img = myutils.img_to_float(img)
        # elif how == 'Rescale to 8-bit unsigned integer format with values [0, 255]':
        #     img = skimage.img_as_float(img)
        #     img = (img*255).astype(np.uint8)
        #     return img
        elif how == "Rescale to [0, 1]":
            img = skimage.img_as_float(img)
            img = skimage.exposure.rescale_intensity(img)
        elif how == "Normalize by max value":
            img = img / np.max(img)
        return img

    def removeAxLimits(self):
        self.ax1.vb.state["limits"]["xLimits"] = [-1e307, +1e307]
        self.ax1.vb.state["limits"]["yLimits"] = [-1e307, +1e307]

    def rescaleIntensExportToVideoDialog(self, how, channel, setImage=True):
        if channel == self.user_ch_name:
            lutItem = self.imgGrad
        else:
            lutItem = self.overlayLayersItems[channel][1]

        for action in lutItem.rescaleActionGroup.actions():
            if action.text() == how:
                action.trigger()
                # self.rescaleIntensitiesLut(setImage=setImage)
                break

    def rescaleIntensitiesLut(
        self, action: QAction = None, setImage: bool = True, imageItem=None
    ):
        if not self.isDataLoaded:
            self.logger.info(
                "WARNING: Data is not loaded. Intensities will be rescaled later."
            )
            return

        posData = self.data[self.pos_i]
        if imageItem is None:
            imageItem = self.img1
            channel = self.user_ch_name
            image_data = posData.img_data
        else:
            channel = imageItem.channelName
            _, filename = self.getPathFromChName(channel, posData)
            image_data = posData.fluo_data_dict[filename]

        triggeredByUser = True
        if action is None:
            triggeredByUser = False
            action = imageItem.lutItem.rescaleActionGroup.checkedAction()

        how = action.text()

        self.df_settings.at[f"how_rescale_intensities_{channel}", "value"] = how
        self.df_settings.to_csv(self.settings_csv_path)

        if how == "Rescale each 2D image":
            if how == self.rescaleIntensChannelHowMapper[channel]:
                # No need to update since we have autoscale
                return

            imageItem.setEnableAutoLevels(True)
            if setImage:
                imageItem.setImage(imageItem.image)
            return

        lutLevelsCh = posData.lutLevels[channel]

        if how == "Rescale across z-stack":
            imageItem.setEnableAutoLevels(False)
            levels_key = (how, posData.frame_i)
            levels = lutLevelsCh.get(levels_key)
            if levels is None:
                levels = self.getPreComputedMinMaxZstack(channel)

            if levels is None:
                image_zstack = image_data[posData.frame_i]
                levels = (image_zstack.min(), image_zstack.max())
            lutLevelsCh[levels_key] = levels
            imageItem.setLevels(levels)
        elif how == "Rescale across time frames":
            imageItem.setEnableAutoLevels(False)
            levels_key = (how, None)
            levels = lutLevelsCh.get(levels_key)
            if levels is None:
                levels = (image_data.min(), image_data.max())

            lutLevelsCh[levels_key] = levels
            imageItem.setLevels(levels)
        elif how == "Choose custom levels...":
            autoLevelsEnabledBefore = imageItem.autoLevelsEnabled
            imageItem.setEnableAutoLevels(False)
            if triggeredByUser:
                current_min, current_max = imageItem.getLevels()
                dtype_max = np.iinfo(image_data.dtype).max
                max_value = image_data.max()
                min_value = image_data.min()
                win = apps.SetCustomLevelsLut(
                    init_min_value=current_min,
                    init_max_value=current_max,
                    maximum_max_value=max_value,
                    minimum_min_value=min_value,
                    parent=self,
                )
                win.sigLevelsChanged.connect(
                    partial(self.customLevelsLutChanged, imageItem=imageItem)
                )
                win.exec_()
                if win.cancel:
                    imageItem.setEnableAutoLevels(autoLevelsEnabledBefore)
                    self.logger.info("Custom LUT levels setting cancelled.")
                    self.updateAllImages()
                    return
                selectedLevels = win.selectedLevels
            else:
                selectedLevels = imageItem.getLevels()
            imageItem.setLevels(selectedLevels)
        elif how == "Do no rescale, display raw image":
            imageItem.setEnableAutoLevels(False)
            levels_key = (how, None)
            levels = lutLevelsCh.get(levels_key)
            if levels is None:
                dtype_max = np.iinfo(image_data.dtype).max
                levels = (0, dtype_max)
            lutLevelsCh[levels_key] = levels
            imageItem.setLevels(levels)

        self.rescaleIntensChannelHowMapper[channel] = how

        if setImage:
            imageItem.setImage(imageItem.image)

    def rescale_intensity_setting_update(
        self,
        channel: str,
        how: str,
    ) -> tuple[str, str]:
        return f"how_rescale_intensities_{channel}", how

    LEGACY_METHODS = (
        "getDisplayedImg1",
        "getDisplayedZstack",
        "getObjBbox",
        "z_lab",
        "get_2Dlab",
        "get_2Drp",
        "set_2Dlab",
        "setTextAnnotZsliceScrolling",
        "setGraphicalAnnotZsliceScrolling",
        "initContoursImage",
        "initLostObjContoursImage",
        "initLostTrackedObjContoursImage",
        "initManualBackgroundImage",
        "initImgCmap",
        "initTextAnnot",
        "zoomOut",
        "zoomToObjsActionCallback",
        "zoomToCells",
        "equalizeHist",
        "getDistantGray",
        "RGBtoGray",
        "ruler_cb",
        "editImgProperties",
        "setTwoImagesLayout",
        "showNextFrameImageItem",
        "showRightImageItem",
        "showLabelImageItem",
        "setAnnotOptionsRightImageLabelsDisabled",
        "setBottomLayoutStretch",
        "setHoverToolSymbolData",
        "getCheckNormAction",
        "normalizeIntensities",
        "invertBw",
        "setCheckedInvertBW",
        "updateImageValueFormatter",
        "getImageDataFromFilename",
        "z_slice_index",
        "get_2Dimg_from_3D",
        "updateZsliceScrollbar",
        "getRawImage",
        "getRawImageLayer0",
        "getImage",
        "updateLabelsAlpha",
        "_getImageupdateAllImages",
        "setImageImg1",
        "setImageImg2",
        "getObject2DimageFromZ",
        "getObject2DsliceFromZ",
        "isObjVisible",
        "getObjImage",
        "getObjSlice",
        "getContoursImageItem",
        "getLostObjImageItem",
        "getLostTrackedObjImageItem",
        "normaliseIntensitiesActionTriggered",
        "setLastUserNormAction",
        "saveLabelsColormap",
        "addFontSizeActions",
        "changeFontSize",
        "enableZstackWidgets",
        "launchSlideshow",
        "setMirroredCursorFromSecondWindow",
        "zProjLockViewToggled",
        "rescaleIntensExportToVideoDialog",
        "customLevelsLutChanged",
        "getPreComputedMinMaxZstack",
        "rescaleIntensitiesLut",
        "showMirroredCursorToggled",
        "clearCursors",
        "activeEraserCircleCursors",
        "activeEraserXCursors",
        "activeBrushCircleCursors",
        "initImgGradRescaleIntensitiesHowPreference",
        "updateAllImages",
        "removeAxLimits",
        "resizeGui",
        "autoRange",
        "resetRange",
    )

    def resetRange(self):
        if self.ax1_viewRange is None:
            return
        xRange, yRange = self.ax1_viewRange
        if self.labelsGrad.showLabelsImgAction.isChecked():
            self.ax2.vb.setRange(xRange=xRange, yRange=yRange)
        self.ax1.vb.setRange(xRange=xRange, yRange=yRange)
        self.ax1_viewRange = None
        self.isRangeReset = True

    def resizeGui(self):
        self.ax1.vb.state["limits"]["xRange"] = [None, None]
        self.ax1.vb.state["limits"]["yRange"] = [None, None]
        self.autoRange()
        if self.ax1.getViewBox().state["limits"]["xRange"][0] is not None:
            self.bottomScrollArea._resizeVertical()
            return
        (xmin, xmax), (ymin, ymax) = self.ax1.viewRange()
        maxYRange = int((ymax - ymin) * 1.5)
        maxXRange = int((xmax - xmin) * 1.5)
        self.ax1.setLimits(maxYRange=maxYRange, maxXRange=maxXRange)
        self.bottomScrollArea._resizeVertical()
        QTimer.singleShot(200, self.autoRange)

    def right_pane_visibility_plan(
        self,
        mode: RightPaneMode,
        checked: bool,
    ) -> RightPaneVisibilityPlan:
        settings_updates = {
            "isNextFrameVisible": "No",
            "isRightImageVisible": "No",
            "isLabelsVisible": "No",
        }
        if checked:
            setting_key = {
                "next_frame": "isNextFrameVisible",
                "right_image": "isRightImageVisible",
                "labels": "isLabelsVisible",
            }[mode]
            settings_updates[setting_key] = "Yes"

        return RightPaneVisibilityPlan(
            mode=mode,
            checked=checked,
            settings_updates=settings_updates,
        )

    def ruler_cb(self, checked):
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            self.connectLeftClickButtons()
        else:
            self.tempSegmentON = False
            self.ax1_rulerPlotItem.setData([], [])
            self.ax1_rulerAnchorsItem.setData([], [])

    def saveLabelsColormap(self):
        self.labelsGrad.saveColormap()

    def setAnnotOptionsRightImageLabelsDisabled(self, disabled):
        self.annotContourCheckboxRight.setDisabled(disabled)
        self.annotSegmMasksCheckboxRight.setDisabled(disabled)
        if disabled:
            self.annotSegmMasksCheckboxRight.setChecked(False)
            self.annotSegmMasksCheckboxRight.setChecked(False)
            self.annotIDsCheckboxRight.setChecked(True)

    def setBottomLayoutStretch(self):
        if (
            self.labelsGrad.showRightImgAction.isChecked()
            or self.labelsGrad.showNextFrameAction.isChecked()
        ):
            # Equally share space between the two control groupboxes
            self.bottomLayout.setStretch(1, 1)
            self.bottomLayout.setStretch(2, 5)
            self.bottomLayout.setStretch(3, 1)
            self.bottomLayout.setStretch(4, 5)
            self.bottomLayout.setStretch(5, 1)
        elif self.labelsGrad.showLabelsImgAction.isChecked():
            # Left control takes only left space
            self.bottomLayout.setStretch(1, 1)
            self.bottomLayout.setStretch(2, 5)
            self.bottomLayout.setStretch(3, 5)
            self.bottomLayout.setStretch(4, 1)
            self.bottomLayout.setStretch(5, 1)
        else:
            # Left control takes all the space
            self.bottomLayout.setStretch(1, 3)
            self.bottomLayout.setStretch(2, 10)
            self.bottomLayout.setStretch(3, 1)
            self.bottomLayout.setStretch(4, 1)
            self.bottomLayout.setStretch(5, 1)

    def setCheckedInvertBW(self, checked):
        self.invertBwAction.setChecked(checked)

    def setGraphicalAnnotZsliceScrolling(self):
        posData = self.data[self.pos_i]
        if self.isSegm3D:
            self.currentLab2D = posData.lab[self.z_lab()]
            self.setOverlaySegmMasks()
            self.doCustomAnnotation(0)
            self.update_rp_metadata()
        else:
            self.currentLab2D = posData.lab
            self.setOverlaySegmMasks()
        self.updateContoursImage(0)
        self.updateContoursImage(1)

    def setHoverToolSymbolData(self, xx, yy, ScatterItems, size=None):
        if not xx:
            self.ax1_lostObjScatterItem.setVisible(True)
            self.ax2_lostObjScatterItem.setVisible(True)

            self.ax1_lostTrackedScatterItem.setVisible(True)
            self.ax2_lostTrackedScatterItem.setVisible(True)

        for item in ScatterItems:
            if size is None:
                item.setData(xx, yy)
            else:
                item.setData(xx, yy, size=size)

    def setImageImg1(self, image=None):
        img = self._getImageupdateAllImages(image=image)
        posData = self.data[self.pos_i]
        self.img1.setCurrentPosIndex(self.pos_i)
        self.img1.setCurrentFrameIndex(posData.frame_i)
        if posData.SizeZ > 1:
            zProjHow = self.zProjComboBox.currentText()
            if zProjHow == "single z-slice":
                z = self.zSliceScrollBar.sliderPosition()
            else:
                z = zProjHow

            self.img1.setCurrentZsliceIndex(z)

        self.img1.setImage(
            img,
            next_frame_image=self.nextFrameImage(),
            scrollbar_value=posData.frame_i + 2,
        )

    def setImageImg2(self, updateLookuptable=True, set_image=True):
        posData = self.data[self.pos_i]
        mode = str(self.modeComboBox.currentText())
        if mode == "Segmentation and Tracking" or self.isSnapshot:
            # self.addExistingDelROIs()
            allDelIDs, lab2D = self.getDelROIlab()
        else:
            lab2D = self.get_2Dlab(posData.lab, force_z=False)
            allDelIDs = set()

        self.currentLab2D = lab2D
        if self.labelsGrad.permanentGreedyCmapAction.isChecked() and updateLookuptable:
            self.greedyShuffleCmap(updateImages=False)

        if self.labelsGrad.showLabelsImgAction.isChecked() and set_image:
            self.img2.setImage(lab2D, z=self.z_lab(), autoLevels=False)

        if updateLookuptable:
            self.updateLookuptable(delIDs=allDelIDs)

    def setLastUserNormAction(self):
        how = self.df_settings.at["how_normIntensities", "value"]
        for action in self.normalizeQActionGroup.actions():
            if action.text() == how:
                action.setChecked(True)
                break

    def setMirroredCursorFromSecondWindow(self, x, y):
        if x is None:
            xx, yy = [], []
        else:
            xx, yy = [x], [y]
        self.ax1_cursor.setData(xx, yy)
        if not self.isTwoImageLayout:
            return
        self.ax2_cursor.setData(xx, yy)

    def setTextAnnotZsliceScrolling(self):
        pass

    def setTwoImagesLayout(self, isTwoImages):
        self.isTwoImageLayout = isTwoImages
        if isTwoImages:
            self.graphLayout.removeItem(self.titleLabel)
            self.graphLayout.addItem(self.titleLabel, row=0, col=1, colspan=2)
            # self.mainLayout.setAlignment(self.bottomLayout, Qt.AlignLeft)
            self.ax2.show()
            self.ax2.vb.setYLink(self.ax1.vb)
            self.ax2.vb.setXLink(self.ax1.vb)
        else:
            self.graphLayout.removeItem(self.titleLabel)
            self.graphLayout.addItem(self.titleLabel, row=0, col=1)
            # self.mainLayout.setAlignment(self.bottomLayout, Qt.AlignCenter)
            self.ax2.hide()
            oldLink = self.ax2.vb.linkedView(self.ax1.vb.YAxis)
            try:
                oldLink.sigYRangeChanged.disconnect()
                oldLink.sigXRangeChanged.disconnect()
            except TypeError:
                pass

    def set_2Dlab(self, lab2D, lab3D=None):
        posData = self.data[self.pos_i]

        if lab3D is None:
            lab3D = posData.lab

        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == "single z-slice"
            if isZslice:
                lab3D[self.z_lab()] = lab2D
            else:
                lab3D[:] = lab2D
        else:
            if lab3D.shape == lab2D.shape:
                lab3D[...] = lab2D
            else:
                posData.lab = lab2D

    def showLabelImageItem(self, checked):
        self.rightImageFramesScrollbar.setVisible(not checked)
        self.rightImageFramesScrollbar.setDisabled(checked)
        self.setTwoImagesLayout(checked)
        self.setAnnotOptionsRightImageLabelsDisabled(checked)
        if checked:
            self.df_settings.at["isLabelsVisible", "value"] = "Yes"
            self.df_settings.at["isNextFrameVisible", "value"] = "No"
            self.df_settings.at["isRightImageVisible", "value"] = "No"
            self.rightBottomGroupbox.show()
            self.rightBottomGroupbox.setChecked(True)
            if not self.isDataLoading:
                self.updateAllImages()
        else:
            self.clearAx2Items()
            self.img2.clear()
            self.df_settings.at["isLabelsVisible", "value"] = "No"
            self.rightBottomGroupbox.hide()
            self.moveDelRoisToLeft()

        self.df_settings.to_csv(self.settings_csv_path)
        QTimer.singleShot(200, self.resizeGui)

        self.setBottomLayoutStretch()

    def showMirroredCursorToggled(self, checked):
        value = "Yes" if checked else "No"
        self.df_settings.at["showMirroredCursor", "value"] = value
        self.df_settings.to_csv(settings_csv_path)

        if not checked:
            self.clearCursors()

    def showNextFrameImageItem(self, checked):
        self.rightImageFramesScrollbar.setVisible(checked)
        self.rightImageFramesScrollbar.setDisabled(not checked)
        self.setTwoImagesLayout(checked)
        if checked:
            self.df_settings.at["isNextFrameVisible", "value"] = "Yes"
            self.df_settings.at["isRightImageVisible", "value"] = "No"
            self.df_settings.at["isLabelsVisible", "value"] = "No"
            self.graphLayout.addItem(self.imgGradRight, row=1, col=self.plotsCol + 2)
            self.rightBottomGroupbox.show()
            self.rightBottomGroupbox.setChecked(True)
            self.drawNothingCheckboxRight.click()
            if not self.isDataLoading:
                self.updateAllImages()
        else:
            self.clearAx2Items()
            self.rightBottomGroupbox.hide()
            self.df_settings.at["isNextFrameVisible", "value"] = "No"
            try:
                self.graphLayout.removeItem(self.imgGradRight)
            except Exception:
                return
            self.rightImageItem.clear()

        self.df_settings.to_csv(self.settings_csv_path)

        QTimer.singleShot(300, self.resizeGui)

        self.setBottomLayoutStretch()

    def showRightImageItem(self, checked):
        self.rightImageFramesScrollbar.setVisible(not checked)
        self.rightImageFramesScrollbar.setDisabled(checked)
        self.setTwoImagesLayout(checked)
        if checked:
            self.df_settings.at["isRightImageVisible", "value"] = "Yes"
            self.df_settings.at["isNextFrameVisible", "value"] = "No"
            self.df_settings.at["isLabelsVisible", "value"] = "No"
            self.graphLayout.addItem(self.imgGradRight, row=1, col=self.plotsCol + 2)
            self.rightBottomGroupbox.show()
            if not self.isDataLoading:
                self.updateAllImages()
        else:
            self.clearAx2Items()
            self.rightBottomGroupbox.hide()
            self.df_settings.at["isRightImageVisible", "value"] = "No"
            try:
                self.graphLayout.removeItem(self.imgGradRight)
            except Exception:
                return
            self.rightImageItem.clear()

        self.df_settings.to_csv(self.settings_csv_path)

        QTimer.singleShot(300, self.resizeGui)

        self.setBottomLayoutStretch()

    @exception_handler
    def updateAllImages(
        self,
        image=None,
        computePointsLayers=True,
        computeContours=True,
        updateLookuptable=True,
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
        delROIsIDs = self.setAllTextAnnotations()
        self.setAllContoursImages(delROIsIDs=delROIsIDs, compute=False)

        mode = self.modeComboBox.currentText()
        self.drawAllMothBudLines()
        if mode == "Normal division: Lineage tree":
            self.drawAllLineageTreeLines()

        self.highlightLostNew()

        if self.ccaTableWin is not None:  # need to add for lin tree, later
            zoomIDs = self.getZoomIDs()
            self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)

        self.doCustomAnnotation(0)

        self.annotate_rip_and_bin_IDs()
        self.updateTempLayerKeepIDs()
        self.whitelistUpdateTempLayer()
        self.drawPointsLayers(computePointsLayers=computePointsLayers)
        self.setManualBackgroundImage()
        self.annotateAssignedObjsAcdcTrackerSecondStep()

        self.highlightSearchedID(self.highlightedID, force=True)
        self.updateTimestampFrame()

        posData.visited = True

    def updateImageValueFormatter(self):
        if self.img1.image is not None:
            dtype = self.img1.image.dtype
            n_digits = len(str(int(self.img1.image.max())))
            self.imgValueFormatter = myutils.get_number_fstring_formatter(
                dtype, precision=abs(n_digits - 5)
            )

        rawImgData = self.data[self.pos_i].img_data
        dtype = rawImgData.dtype
        n_digits = len(str(int(rawImgData.max())))
        self.rawValueFormatter = myutils.get_number_fstring_formatter(
            dtype, precision=abs(n_digits - 5)
        )

    def updateLabelsAlpha(self, value):
        self.df_settings.at["overlaySegmMasksAlpha", "value"] = value
        self.df_settings.to_csv(self.settings_csv_path)
        if self.keepIDsButton.isChecked():
            value = value / 3
        self.labelsLayerImg1.setOpacity(value)
        self.labelsLayerRightImg.setOpacity(value)

    def updateZsliceScrollbar(self, frame_i):
        posData = self.data[self.pos_i]
        if self.switchPlaneCombobox.depthAxes() != "z":
            return

        idx = (posData.filename, frame_i)
        try:
            z = posData.segmInfo_df.at[idx, "z_slice_used_gui"]
        except ValueError:
            z = posData.segmInfo_df.loc[idx, "z_slice_used_gui"].iloc[0]
        try:
            zProjHow = posData.segmInfo_df.at[idx, "which_z_proj_gui"]
        except ValueError:
            zProjHow = posData.segmInfo_df.loc[idx, "which_z_proj_gui"].iloc[0]

        self.zProjComboBox.setCurrentText(zProjHow)

        reconnect = False
        try:
            self.zSliceScrollBar.actionTriggered.disconnect()
            self.zSliceScrollBar.sliderReleased.disconnect()
            reconnect = True
        except TypeError:
            pass
        self.zSliceScrollBar.setSliderPosition(z)
        if reconnect:
            self.zSliceScrollBar.actionTriggered.connect(
                self.zSliceScrollBarActionTriggered
            )
            self.zSliceScrollBar.sliderReleased.connect(self.zSliceScrollBarReleased)
        self.zSliceSpinbox.setValueNoEmit(z + 1)

    def zProjLockViewToggled(self, checked):
        self.updateZproj(self.zProjComboBox.currentText())

    def z_lab(self, checkIfProj=False):
        if checkIfProj and self.zProjComboBox.currentText() != "single z-slice":
            return

        if not self.isSegm3D:
            return

        posData = self.data[self.pos_i]

        idx = self.zSliceScrollBar.sliderPosition()

        # ensure idx doesnt exceed the number of z-slices of the position
        idx_z = min(idx, posData.SizeZ - 1)

        if not self.switchPlaneCombobox.isEnabled():
            return idx_z

        depthAxes = self.switchPlaneCombobox.depthAxes()
        if depthAxes == "z":
            return idx_z
        elif depthAxes == "y":
            idx_y = min(idx, posData.SizeY - 1)
            return (slice(None), idx_y)
        else:
            idx_x = min(idx, posData.SizeX - 1)
            return (slice(None), slice(None), idx_x)

    def z_slice_index(self):
        posData = self.data[self.pos_i]
        if posData.SizeZ == 1:
            return None
        zProjHow = self.zProjComboBox.currentText()
        if zProjHow != "single z-slice":
            return zProjHow

        axis_slice = self.zSliceScrollBar.sliderPosition()
        if self.switchPlaneCombobox.depthAxes() == "x":
            z_slice = (slice(None, None, None), slice(None, None, None), axis_slice)
        elif self.switchPlaneCombobox.depthAxes() == "y":
            z_slice = (slice(None, None, None), axis_slice)
        else:
            z_slice = axis_slice

        return z_slice

    def zoomOut(self):
        self.ax1.autoRange()

    def zoomToCells(self, enforce=False):
        if not self.enableAutoZoomToCellsAction.isChecked() and not enforce:
            return

        self.data[self.pos_i]
        lab_mask = (self.currentLab2D > 0).astype(np.uint8)
        rp = skimage.measure.regionprops(lab_mask)
        if not rp:
            Y, X = lab_mask.shape
            xRange = -0.5, X + 0.5
            yRange = -0.5, Y + 0.5
        else:
            obj = rp[0]
            min_row, min_col, max_row, max_col = self.getObjBbox(obj.bbox)
            xRange = min_col - 10, max_col + 10
            yRange = max_row + 10, min_row - 10

        self.ax1.setRange(xRange=xRange, yRange=yRange)

    def zoomToObjsActionCallback(self):
        self.zoomToCells(enforce=True)
