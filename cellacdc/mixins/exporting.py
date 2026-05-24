"""Qt view adapter for image and video export workflows."""

from __future__ import annotations

import os
import shutil
import traceback
from functools import partial

from datetime import datetime
import numpy as np
import skimage.measure
import skimage.segmentation
from qtpy.QtCore import QTimer

from cellacdc import _warnings, apps, disableWindow, exception_handler
from cellacdc import exporters, html_utils, prompts, widgets


class ExportingMixin:
    """Qt-facing adapter around export dialogs, exporters, and progress UI."""

    def askTimelapseOrZslicesVideo(self):
        txt = html_utils.paragraph("""
            Do you want to record a video of scrolling through the z-slices or 
            a Timelapse video?
        """)
        msg = widgets.myMessageBox(wrapText=False)
        _, timelapseButton = msg.question(
            self,
            "Z-slices or Timelapse video?",
            txt,
            buttonsTexts=("Z-slices", "Timelapse"),
        )
        if msg.cancel:
            return

        return msg.clickedButton == timelapseButton

    def build_export_mask_image(
        self,
        image_shape,
        view_range,
        *,
        invert_bw=False,
    ):
        mask_image = np.zeros(
            self.export_mask_image_shape(image_shape),
            dtype=np.uint8,
        )
        x_range, y_range = view_range
        x0, x1 = map(round, x_range)
        y0, y1 = map(round, y_range)

        if invert_bw:
            mask_image[:, :, :3] = 255

        if x0 > 0:
            mask_image[:, :x0, 3] = 255
        if x1 < mask_image.shape[1]:
            mask_image[:, x1:, 3] = 255
        if y0 > 0:
            mask_image[:y0, :, 3] = 255
        if y1 < mask_image.shape[0]:
            mask_image[y1:, :, 3] = 255

        return mask_image

    def exportAddScaleBar(self, checked):
        self.addScaleBarAction.setChecked(checked)

    @exception_handler
    def exportFrame(self):
        nd = self.exportToVideoPreferences["num_digits"]
        idx = str(self.exportToVideoCurrentNavVarIdx).zfill(nd)
        filename = self.exportToVideoPreferences["filename"]
        png_filename = f"{idx}_{filename}.png"
        pngs_folderpath = self.exportToVideoPreferences["pngs_folderpath"]

        png_filepath = os.path.join(pngs_folderpath, png_filename)
        img_bgr = self.exportToVideoImageExporter.export(png_filepath)
        self.exportToVideoExporter.add_frame(img_bgr)
        return True

    @disableWindow
    def exportToImage(self, preferences):
        filepath = preferences["filepath"]
        self.logger.info(f'Saving image to "{filepath}"...')

        if filepath.endswith(".svg"):
            exporter = exporters.SVGExporter(self.ax1)
        else:
            exporter = exporters.ImageExporter(self.ax1, dpi=preferences["dpi"])
        exporter.export(filepath)
        self.logger.info("Image saved.")

        self.setDisabled(False)
        self.exportMaskImage[:] = 0
        self.exportMaskImageItem.setImage(self.exportMaskImage)
        prompts.exportToImageFinished(filepath, qparent=self)

    def exportToImageTriggered(self):
        posData = self.data[self.pos_i]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_acdc_exported_image"
        win = apps.ExportToImageParametersDialog(
            parent=self,
            startFolderpath=posData.pos_path,
            startFilename=filename,
            startViewRange=self.ax1.viewRange(),
            isScaleBarPresent=self.addScaleBarAction.isChecked(),
        )
        win.sigAddScaleBar.connect(self.exportAddScaleBar)
        win.sigRangeChanged.connect(
            partial(self.setViewRangeFromExportToImageDialog, win=win)
        )
        # self.ax1.vb.sigRangeChanged.connect(
        #     win.updateViewRangeExportToImageDialog
        # )
        self.setExportMaskImage(self.ax1.viewRange())
        self.exportToImageWindow = win
        win.exec_()
        # self.ax1.vb.sigRangeChanged.disconnect()
        if win.cancel:
            self.exportMaskImage[:] = 0
            self.exportMaskImageItem.setImage(self.exportMaskImage)
            self.exportToImageWindow = None
            self.logger.info("Export to image process cancelled")
            return

        isTransparent = self.overlayToolbar.isTransparent()
        if not isTransparent:
            # SVG export works only with RGBA not with setOpacity
            # --> only true transparency mode can be used
            self.overlayToolbar.setTransparent(True)

        self.exportToImage(win.selected_preferences)
        self.exportToImageWindow = None

        if not isTransparent:
            self.overlayToolbar.setTransparent(False)

    def exportToVideoAddTimestamp(self, checked):
        self.addTimestampAction.setChecked(checked)

    def exportToVideoFinished(self, conversion_to_mp4_successful):
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.progressWin = None

        # Back to current frame
        if self.exportToVideoPreferences["is_timelapse"]:
            posData = self.data[self.pos_i]
            posData.frame_i = self.exportToVideoNavVarIdxToRestore
            self.get_data()
            self.store_data()
            self.updateAllImages()
            self.navigateScrollBar.setSliderPosition(posData.frame_i + 1)
            self.navSpinBox.setValue(posData.frame_i + 1)
        else:
            self.update_z_slice(self.exportToVideoNavVarIdxToRestore)

        self.setDisabled(False)
        self.isExportingVideo = False

        if not self.isTransparent:
            # True transparency mode was activated programmatically
            # --> restore what the user had before starting to export
            self.overlayToolbar.setTransparent(False)

        prompts.exportToVideoFinished(
            self.exportToVideoPreferences, conversion_to_mp4_successful, qparent=self
        )

    def exportToVideoTriggered(self):
        posData = self.data[self.pos_i]

        doTimelapseVideo = posData.SizeT > 1
        if posData.SizeT > 1 and posData.SizeZ > 1:
            doTimelapseVideo = self.askTimelapseOrZslicesVideo()

        if doTimelapseVideo is None:
            self.logger.info("Export to video process cancelled")
            return

        channels = [self.user_ch_name, *self.checkedOverlayChannels]
        mode = "timelapse" if doTimelapseVideo else "z_slices"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_acdc_exported_{mode}_video"
        win = apps.ExportToVideoParametersDialog(
            channels,
            parent=self,
            startFolderpath=posData.pos_path,
            startFilename=filename,
            startFrameNum=posData.frame_i + 1,
            SizeT=posData.SizeT,
            SizeZ=posData.SizeZ,
            isTimelapseVideo=doTimelapseVideo,
            isScaleBarPresent=self.addScaleBarAction.isChecked(),
            isTimestampPresent=self.addTimestampAction.isChecked(),
            rescaleIntensChannelHowMapper=self.rescaleIntensChannelHowMapper,
        )
        win.sigAddScaleBar.connect(self.exportAddScaleBar)
        win.sigAddTimestamp.connect(self.exportToVideoAddTimestamp)
        win.sigRescaleIntensLut.connect(self.rescaleIntensExportToVideoDialog)
        win.exec_()
        if win.cancel:
            self.logger.info("Export to video process cancelled")
            return

        cancel = _warnings.warnExportToVideo(qparent=self)
        if cancel:
            self.logger.info("Export to video process cancelled")
            return

        self.startExportToVideoWorker(win.selected_preferences)

    def export_frame_plan(
        self,
        *,
        current_index: int,
        num_digits: int,
        filename: str,
        pngs_folderpath: str,
    ) -> ExportFramePlan:
        frame_index_text = str(current_index).zfill(num_digits)
        png_filename = f"{frame_index_text}_{filename}.png"
        return ExportFramePlan(
            frame_index_text=frame_index_text,
            png_filename=png_filename,
            png_filepath=os.path.join(pngs_folderpath, png_filename),
        )

    def export_mask_image_shape(self, image_shape) -> tuple[int, int, int]:
        height, width = image_shape[-2:]
        return height, width, 4

    def exportingFramesFinished(self):
        if not self.exportToVideoPreferences["save_pngs"]:
            self.logger.info("Removing PNGs...")
            try:
                shutil.rmtree(self.exportToVideoPreferences["pngs_folderpath"])
            except Exception:
                pass

        self.logger.info("Saving video...")

        self.exportToVideoExporter.release()

        # Run ffmpeg new process
        conversion_to_mp4_successful = True
        if self.exportToVideoPreferences["filepath"].endswith(".mp4"):
            try:
                self.exportToVideoExporter.avi_to_mp4()
                try:
                    os.remove(self.exportToVideoPreferences["avi_filepath"])
                except Exception:
                    pass
            except Exception:
                self.logger.exception(traceback.format_exc())
                self.logger.info("Conversion to MP4 failed. See traceback above.")
                conversion_to_mp4_successful = False
                self.exportToVideoPreferences["filepath"] = (
                    self.exportToVideoExporter._avi_filepath
                )

        self.exportToVideoFinished(conversion_to_mp4_successful)

    def exportingVideoCritical(self):
        self.setDisabled(False)

        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.progressWin = None

        self.logger.info("Exporting video process failed.")

    def getZoomIDs(self, viewRange=None):
        if viewRange is None:
            viewRange = self.ax1.viewRange()

        lab = self.currentLab2D
        Y, X = lab.shape
        ((xmin, xmax), (ymin, ymax)) = viewRange
        if xmin <= 0 and ymin <= 0 and xmax >= X and ymax >= Y:
            self.data[self.pos_i]
            return None

        xmin = xmin if xmin >= 0 else 0
        ymin = ymin if ymin >= 0 else 0
        xmax = xmax if xmax < X else X
        ymax = ymax if ymax < Y else Y

        zoomSlice = (
            slice(round(ymin), round(ymax)),
            slice(round(xmin), round(xmax)),
        )

        zoomLab = skimage.segmentation.clear_border(lab[zoomSlice])
        zoomRp = skimage.measure.regionprops(zoomLab)
        zoomIDs = [obj.label for obj in zoomRp]
        return zoomIDs

    def initExportMaskImage(self):
        posData = self.data[self.pos_i]
        z_slice = self.z_lab()
        img = posData.img_data[posData.frame_i]
        Y, X = img[z_slice].shape[-2:]

        self.exportMaskImage = np.zeros((Y, X, 4), dtype=np.uint8)

    def onSigUpdateCcaTableWindow(self, *args):
        if not self.isDataLoaded:
            return

        if self.ccaTableWin is None:
            return

        viewRange = self.ax1.viewRange()
        posData = self.data[self.pos_i]
        zoomIDs = self.getZoomIDs(viewRange=viewRange)

        self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)

    def setExportMaskImage(self, viewRange):
        if not hasattr(self, "exportMaskImage"):
            self.initExportMaskImage()
        else:
            self.exportMaskImage[:] = 0

        xRange, yRange = viewRange
        x0, x1 = map(round, xRange)
        y0, y1 = map(round, yRange)

        if self.invertBwAction.isChecked():
            self.exportMaskImage[:, :, :3] = 255

        if x0 > 0:
            self.exportMaskImage[:, :x0, 3] = 255
        if x1 < self.exportMaskImage.shape[1]:
            self.exportMaskImage[:, x1:, 3] = 255
        if y0 > 0:
            self.exportMaskImage[:y0, :, 3] = 255
        if y1 < self.exportMaskImage.shape[0]:
            self.exportMaskImage[y1:, :, 3] = 255

        self.exportMaskImageItem.setImage(self.exportMaskImage)

    def setViewRangeFromExportToImageDialog(self, viewRange, win=None):
        xRange, yRange = viewRange
        # self.ax1.sigRangeChanged.disconnect(self.viewRangeChanged)
        self.ax1.setRange(xRange=xRange, yRange=yRange)
        # self.ax1.sigRangeChanged.connect(self.viewRangeChanged)
        # self.viewRangeChanged(
        #     self.ax1.vb, viewRange, updateExportMaskImage=False
        # )
        self.setExportMaskImage(viewRange)

    def shifted_view_range(self, previous_range, current_range, window_range):
        prev_x_range, prev_y_range = previous_range
        curr_x_range, curr_y_range = current_range
        win_x_range, win_y_range = window_range

        delta_x = curr_x_range[0] - prev_x_range[0]
        delta_y = curr_y_range[0] - prev_y_range[0]

        return (
            (win_x_range[0] + delta_x, win_x_range[1] + delta_x),
            (win_y_range[0] + delta_y, win_y_range[1] + delta_y),
        )

    def startExportToVideoWorker(self, preferences):
        self.isExportingVideo = True
        self.isTransparent = self.overlayToolbar.isTransparent()
        if not self.isTransparent:
            # SVG export works only with RGBA not with setOpacity
            # --> only true transparency mode can be used
            self.overlayToolbar.setTransparent(True)

        self.setDisabled(True)

        self.progressWin = apps.QDialogWorkerProgress(
            title="Exporting to video",
            parent=self.mainWin,
            pbarDesc="Exporting to video...",
        )
        self.progressWin.show(self.app)
        self.exportToVideoStopNavVarNum = preferences["stop_nav_var_num"]
        self.numFramesExported = 0
        self.progressWin.mainPbar.setMaximum(
            preferences["stop_nav_var_num"] - preferences["start_nav_var_num"] + 1
        )
        self.exportToVideoPreferences = preferences

        self.store_data()
        posData = self.data[self.pos_i]
        if self.exportToVideoPreferences["is_timelapse"]:
            # Go to requested start frame
            posData.frame_i = preferences["start_nav_var_num"] - 1
            self.get_data()
            self.updateAllImages()
            self.exportToVideoNavVarIdxToRestore = posData.frame_i
        else:
            self.update_z_slice(preferences["start_nav_var_num"] - 1)
            self.exportToVideoNavVarIdxToRestore = self.zSliceScrollBar.sliderPosition()
        self.exportToVideoCurrentNavVarIdx = preferences["start_nav_var_num"] - 1

        self.exportToVideoImageExporter = exporters.ImageExporter(
            self.ax1, save_pngs=preferences["save_pngs"], dpi=preferences["dpi"]
        )
        self.exportToVideoExporter = exporters.VideoExporter(
            preferences["avi_filepath"], preferences["fps"]
        )

        QTimer.singleShot(200, self.updateAndExportFrame)

    def timestamped_export_filename(self, kind: str, *, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        return f"{timestamp.strftime('%Y%m%d_%H%M%S')}_acdc_exported_{kind}"

    def updateAndExportFrame(self):
        didVideoExporterFinish = (
            self.exportToVideoCurrentNavVarIdx == self.exportToVideoStopNavVarNum
        )
        if didVideoExporterFinish:
            self.progressWin.mainPbar.setMaximum(0)
            self.progressWin.mainPbar.setValue(0)
            QTimer.singleShot(50, self.exportingFramesFinished)
            return

        self.data[self.pos_i]
        if self.exportToVideoPreferences["is_timelapse"]:
            self.goToFrameNumber(self.exportToVideoCurrentNavVarIdx + 1)
        else:
            self.update_z_slice(self.exportToVideoCurrentNavVarIdx)

        success = self.exportFrame()
        if success is None:
            self.exportingVideoCritical()
            return

        self.exportToVideoCurrentNavVarIdx += 1
        self.progressWin.mainPbar.update(1)

        QTimer.singleShot(50, self.updateAndExportFrame)

    def updateViewRangeExportToImage(self, viewRange):
        if self.exportToImageWindow is None:
            return

        # prevViewRange = self.exportToImageWindow.viewRange()
        prevViewRange = self._viewRange
        prevXRange = prevViewRange[0]
        prevYRange = prevViewRange[1]
        currXRange = viewRange[0]
        currYRange = viewRange[1]

        prevX0, prevX1 = prevXRange
        currX0, currX1 = currXRange
        prevY0, prevY1 = prevYRange
        currY0, currY1 = currYRange

        deltaX = currX0 - prevX0
        deltaY = currY0 - prevY0

        winViewRange = self.exportToImageWindow.viewRange()
        winXRange = winViewRange[0]
        winYRange = winViewRange[1]
        winX0, winX1 = winXRange
        winY0, winY1 = winYRange

        newX0 = winX0 + deltaX
        newX1 = winX1 + deltaX
        newY0 = winY0 + deltaY
        newY1 = winY1 + deltaY

        self.exportToImageWindow.setViewRange(
            (newX0, newX1), (newY0, newY1), emitSignal=False
        )

    def zoom_ids(self, labels_2d, view_range):
        height, width = labels_2d.shape
        ((xmin, xmax), (ymin, ymax)) = view_range
        if xmin <= 0 and ymin <= 0 and xmax >= width and ymax >= height:
            return None

        xmin = max(xmin, 0)
        ymin = max(ymin, 0)
        xmax = min(xmax, width)
        ymax = min(ymax, height)

        zoom_slice = (
            slice(round(ymin), round(ymax)),
            slice(round(xmin), round(xmax)),
        )
        zoom_labels = skimage.segmentation.clear_border(labels_2d[zoom_slice])
        zoom_regionprops = skimage.measure.regionprops(zoom_labels)
        return [obj.label for obj in zoom_regionprops]
