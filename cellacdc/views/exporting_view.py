"""Qt view adapter for image and video export workflows."""

from __future__ import annotations

import os
import shutil
import traceback
from functools import partial

from qtpy.QtCore import QTimer

from cellacdc import _warnings, apps, disableWindow, exception_handler
from cellacdc import exporters, html_utils, prompts, widgets
from cellacdc.viewmodels.exporting_viewmodel import ExportingViewModel


class ExportingView:
    """Qt-facing adapter around export dialogs, exporters, and progress UI."""

    def __init__(self, host, view_model: ExportingViewModel):
        object.__setattr__(self, 'host', host)
        object.__setattr__(self, 'view_model', view_model)

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host', 'view_model'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def startExportToVideoWorker(self, preferences):
        self.isExportingVideo = True
        self.isTransparent = self.overlayToolbar.isTransparent()
        if not self.isTransparent:
            # SVG export works only with RGBA not with setOpacity
            # --> only true transparency mode can be used
            self.overlayToolbar.setTransparent(True)

        self.setDisabled(True)

        self.progressWin = apps.QDialogWorkerProgress(
            title='Exporting to video', parent=self.host.mainWin,
            pbarDesc='Exporting to video...'
        )
        self.progressWin.show(self.app)
        self.exportToVideoStopNavVarNum = preferences['stop_nav_var_num']
        self.numFramesExported = 0
        self.progressWin.mainPbar.setMaximum(
            preferences['stop_nav_var_num']
            - preferences['start_nav_var_num'] + 1
        )
        self.exportToVideoPreferences = preferences

        self.store_data()
        posData = self.data[self.pos_i]
        if self.exportToVideoPreferences['is_timelapse']:
            # Go to requested start frame
            posData.frame_i = preferences['start_nav_var_num'] - 1
            self.get_data()
            self.updateAllImages()
            self.exportToVideoNavVarIdxToRestore = posData.frame_i
        else:
            self.update_z_slice(preferences['start_nav_var_num'] - 1)
            self.exportToVideoNavVarIdxToRestore = (
                self.zSliceScrollBar.sliderPosition()
            )
        self.exportToVideoCurrentNavVarIdx = (
            preferences['start_nav_var_num'] - 1
        )

        self.exportToVideoImageExporter = exporters.ImageExporter(
            self.ax1,
            save_pngs=preferences['save_pngs'],
            dpi=preferences['dpi']
        )
        self.exportToVideoExporter = exporters.VideoExporter(
            preferences['avi_filepath'], preferences['fps']
        )

        QTimer.singleShot(200, self.updateAndExportFrame)

    def updateAndExportFrame(self):
        didVideoExporterFinish = (
            self.exportToVideoCurrentNavVarIdx
            == self.exportToVideoStopNavVarNum
        )
        if didVideoExporterFinish:
            self.progressWin.mainPbar.setMaximum(0)
            self.progressWin.mainPbar.setValue(0)
            QTimer.singleShot(50, self.exportingFramesFinished)
            return

        posData = self.data[self.pos_i]
        if self.exportToVideoPreferences['is_timelapse']:
            self.goToFrameNumber(self.exportToVideoCurrentNavVarIdx+1)
        else:
            self.update_z_slice(self.exportToVideoCurrentNavVarIdx)

        success = self.exportFrame()
        if success is None:
            self.exportingVideoCritical()
            return

        self.exportToVideoCurrentNavVarIdx += 1
        self.progressWin.mainPbar.update(1)

        QTimer.singleShot(50, self.updateAndExportFrame)

    @exception_handler
    def exportFrame(self):
        plan = self.view_model.export_frame_plan(
            current_index=self.exportToVideoCurrentNavVarIdx,
            num_digits=self.exportToVideoPreferences['num_digits'],
            filename=self.exportToVideoPreferences['filename'],
            pngs_folderpath=self.exportToVideoPreferences['pngs_folderpath'],
        )
        img_bgr = self.exportToVideoImageExporter.export(plan.png_filepath)
        self.exportToVideoExporter.add_frame(img_bgr)
        return True

    def exportingVideoCritical(self):
        self.setDisabled(False)

        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.progressWin = None

        self.logger.info('Exporting video process failed.')

    def exportingFramesFinished(self):
        if not self.exportToVideoPreferences['save_pngs']:
            self.logger.info('Removing PNGs...')
            try:
                shutil.rmtree(self.exportToVideoPreferences['pngs_folderpath'])
            except Exception as err:
                pass

        self.logger.info('Saving video...')

        self.exportToVideoExporter.release()

        # Run ffmpeg new process
        conversion_to_mp4_successful = True
        if self.exportToVideoPreferences['filepath'].endswith('.mp4'):
            try:
                self.exportToVideoExporter.avi_to_mp4()
                try:
                    os.remove(self.exportToVideoPreferences['avi_filepath'])
                except Exception as err:
                    pass
            except Exception as err:
                self.logger.exception(traceback.format_exc())
                self.logger.info(
                    'Conversion to MP4 failed. See traceback above.'
                )
                conversion_to_mp4_successful = False
                self.exportToVideoPreferences['filepath'] = (
                    self.exportToVideoExporter._avi_filepath
                )

        self.exportToVideoFinished(conversion_to_mp4_successful)

    def exportToVideoFinished(self, conversion_to_mp4_successful):
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.progressWin = None

        # Back to current frame
        if self.exportToVideoPreferences['is_timelapse']:
            posData = self.data[self.pos_i]
            posData.frame_i = self.exportToVideoNavVarIdxToRestore
            self.get_data()
            self.store_data()
            self.updateAllImages()
            self.navigateScrollBar.setSliderPosition(posData.frame_i+1)
            self.navSpinBox.setValue(posData.frame_i+1)
        else:
            self.update_z_slice(self.exportToVideoNavVarIdxToRestore)

        self.setDisabled(False)
        self.isExportingVideo = False

        if not self.isTransparent:
            # True transparency mode was activated programmatically
            # --> restore what the user had before starting to export
            self.overlayToolbar.setTransparent(False)

        prompts.exportToVideoFinished(
            self.exportToVideoPreferences, conversion_to_mp4_successful,
            qparent=self.host
        )

    def exportAddScaleBar(self, checked):
        self.addScaleBarAction.setChecked(checked)

    def exportToVideoAddTimestamp(self, checked):
        self.addTimestampAction.setChecked(checked)

    def askTimelapseOrZslicesVideo(self):
        txt = html_utils.paragraph("""
            Do you want to record a video of scrolling through the z-slices or
            a Timelapse video?
        """)
        msg = widgets.myMessageBox(wrapText=False)
        _, timelapseButton = msg.question(
            self.host, 'Z-slices or Timelapse video?', txt,
            buttonsTexts=('Z-slices', 'Timelapse')
        )
        if msg.cancel:
            return

        return msg.clickedButton == timelapseButton

    def exportToVideoTriggered(self):
        posData = self.data[self.pos_i]

        doTimelapseVideo = posData.SizeT > 1
        if posData.SizeT > 1 and posData.SizeZ > 1:
            doTimelapseVideo = self.askTimelapseOrZslicesVideo()

        if doTimelapseVideo is None:
            self.logger.info('Export to video process cancelled')
            return

        channels = [self.user_ch_name, *self.checkedOverlayChannels]
        mode = 'timelapse' if doTimelapseVideo else 'z_slices'
        filename = self.view_model.timestamped_export_filename(
            f'{mode}_video'
        )
        win = apps.ExportToVideoParametersDialog(
            channels,
            parent=self.host,
            startFolderpath=posData.pos_path,
            startFilename=filename,
            startFrameNum=posData.frame_i+1,
            SizeT=posData.SizeT,
            SizeZ=posData.SizeZ,
            isTimelapseVideo=doTimelapseVideo,
            isScaleBarPresent=self.addScaleBarAction.isChecked(),
            isTimestampPresent=self.addTimestampAction.isChecked(),
            rescaleIntensChannelHowMapper=self.rescaleIntensChannelHowMapper
        )
        win.sigAddScaleBar.connect(self.exportAddScaleBar)
        win.sigAddTimestamp.connect(self.exportToVideoAddTimestamp)
        win.sigRescaleIntensLut.connect(self.rescaleIntensExportToVideoDialog)
        win.exec_()
        if win.cancel:
            self.logger.info('Export to video process cancelled')
            return

        cancel = _warnings.warnExportToVideo(qparent=self.host)
        if cancel:
            self.logger.info('Export to video process cancelled')
            return

        self.startExportToVideoWorker(win.selected_preferences)

    def initExportMaskImage(self):
        posData = self.data[self.pos_i]
        z_slice = self.z_lab()
        img = posData.img_data[posData.frame_i]
        self.exportMaskImage = self.view_model.build_export_mask_image(
            img[z_slice].shape,
            self.ax1.viewRange(),
            invert_bw=False,
        )

    def setExportMaskImage(self, viewRange):
        if not hasattr(self, 'exportMaskImage'):
            self.initExportMaskImage()

        self.exportMaskImage[:] = self.view_model.build_export_mask_image(
            self.exportMaskImage.shape[:2],
            viewRange,
            invert_bw=self.invertBwAction.isChecked(),
        )

        self.exportMaskImageItem.setImage(self.exportMaskImage)

    def setViewRangeFromExportToImageDialog(self, viewRange, win=None):
        xRange, yRange = viewRange
        # self.ax1.sigRangeChanged.disconnect(
        #     self.display_decorations_view.view_range_changed
        # )
        self.ax1.setRange(xRange=xRange, yRange=yRange)
        # self.ax1.sigRangeChanged.connect(
        #     self.display_decorations_view.view_range_changed
        # )
        # self.display_decorations_view.view_range_changed(
        #     self.ax1.vb, viewRange, updateExportMaskImage=False
        # )
        self.setExportMaskImage(viewRange)

    def updateViewRangeExportToImage(self, viewRange):
        if self.exportToImageWindow is None:
            return

        # prevViewRange = self.exportToImageWindow.viewRange()
        prevViewRange = self._viewRange
        winViewRange = self.exportToImageWindow.viewRange()
        x_range, y_range = self.view_model.shifted_view_range(
            prevViewRange,
            viewRange,
            winViewRange,
        )

        self.exportToImageWindow.setViewRange(
            x_range, y_range, emitSignal=False
        )

    def getZoomIDs(self, viewRange=None):
        if viewRange is None:
            viewRange = self.ax1.viewRange()

        return self.view_model.zoom_ids(self.currentLab2D, viewRange)

    def onSigUpdateCcaTableWindow(self, *args):
        if not self.isDataLoaded:
            return

        if self.ccaTableWin is None:
            return

        viewRange = self.ax1.viewRange()
        posData = self.data[self.pos_i]
        zoomIDs = self.getZoomIDs(viewRange=viewRange)

        self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)

    @disableWindow
    def exportToImage(self, preferences):
        filepath = preferences['filepath']
        self.logger.info(f'Saving image to "{filepath}"...')

        if filepath.endswith('.svg'):
            exporter = exporters.SVGExporter(self.ax1)
        else:
            exporter = exporters.ImageExporter(self.ax1, dpi=preferences['dpi'])
        exporter.export(filepath)
        self.logger.info(f'Image saved.')

        self.setDisabled(False)
        self.exportMaskImage[:] = 0
        self.exportMaskImageItem.setImage(self.exportMaskImage)
        prompts.exportToImageFinished(filepath, qparent=self.host)

    def exportToImageTriggered(self):
        posData = self.data[self.pos_i]
        filename = self.view_model.timestamped_export_filename('image')
        win = apps.ExportToImageParametersDialog(
            parent=self.host,
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
            self.logger.info('Export to image process cancelled')
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
