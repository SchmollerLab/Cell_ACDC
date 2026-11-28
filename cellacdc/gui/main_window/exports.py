"""Exports of the main GUI: current view, videos and images. Mixins of
guiWin (gui/main_window/window.py) only: the measurements viewer never
composes them.
"""
import os
import shutil
import traceback
from datetime import datetime
from functools import partial
import numpy as np
from qtpy.QtCore import QEventLoop, QTimer, QThread
from cellacdc import exception_handler, disableWindow
from cellacdc import prompts, apps, workers, html_utils
from cellacdc import widgets
from cellacdc import _warnings
from cellacdc import qutils
from cellacdc import exporters
from cellacdc import io


class ExportsGuiElements:
    """Exports of the current view, videos and images (overlay prompt,
    scale bar, timestamp, RGBA masks).
    """

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

    def exportCurrentViewWorkerFinished(self):
        self.exportCurrentViewToolbar.setInfoText('Exported!')
        self.exportCurrentViewWorkerLoop.exit()
        QTimer.singleShot(
            2000, 
            partial(self.exportCurrentViewToolbar.setInfoText, '')
        )

    @disableWindow
    def exportCurrentView(self, isOnNext=False):
        if isOnNext and not self.exportCurrentViewToolbar.isAutoExportOnNext():
            # Automatic export on next disabled --> do not export
            return

        dstFolderPath = self.exportCurrentViewToolbar.dstFolderPath()
        if not dstFolderPath or not os.path.exists(dstFolderPath):
            self.setDisabled(False)
            blinker = qutils.QControlBlink(
                self.exportCurrentViewToolbar.dstFolderPathControl, 
                qparent=self
            )
            blinker.start()
            _warnings.warnExportCurrentViewDstFolderPathInvalid(
                dstFolderPath, qparent=self
            )
            return

        self.exportCurrentViewToolbar.setInfoText('Exporting...')

        posData = self.data[self.pos_i]
        self.exportCurrentViewThread = QThread()
        self.exportCurrentViewWorker = workers.SimpleWorker(
            posData, 
            io.export_current_view, 
            func_args=(
                self.exportCurrentViewToolbar.selectedModelName(),
                dstFolderPath, 
            ),
            func_kwargs={
                'logger_func': self.logger.info,
                'verbose': True
            }
        )
        self.exportCurrentViewWorker.moveToThread(self.exportCurrentViewThread)
        self.exportCurrentViewWorker.signals.finished.connect(
            self.exportCurrentViewThread.quit
        )
        self.exportCurrentViewWorker.signals.finished.connect(
            self.exportCurrentViewWorker.deleteLater
        )
        self.exportCurrentViewThread.finished.connect(
            self.exportCurrentViewThread.deleteLater
        )
        self.exportCurrentViewWorker.signals.critical.connect(
            self.exportCurrentViewWorkerCritical
        )
        self.exportCurrentViewWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.exportCurrentViewWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.exportCurrentViewWorker.signals.progress.connect(
            self.workerProgress
        )
        self.exportCurrentViewWorker.signals.finished.connect(
            self.exportCurrentViewWorkerFinished
        )
        
        self.exportCurrentViewThread.started.connect(
            self.exportCurrentViewWorker.run
        )
        self.exportCurrentViewThread.start()
        
        self.exportCurrentViewWorkerLoop = QEventLoop()
        self.exportCurrentViewWorkerLoop.exec_()

    def exportCurrentViewWorkerCritical(self, error):
        self.exportCurrentViewToolbar.setInfoText(
            html_utils.span(
                'ERROR occurred during export. See terminal for more info'
            )
        )
        self.exportCurrentViewWorkerLoop.exit()
        self.workerCritical(error)

    def initExportMaskImage(self):
        posData = self.data[self.pos_i]
        z_slice = self.z_lab()
        img = posData.img_data[posData.frame_i]
        Y, X = img[z_slice].shape[-2:]
            
        self.exportMaskImage = np.zeros((Y, X, 4), dtype=np.uint8)

    def startExportToVideoWorker(self, preferences):
        self.isExportingVideo = True
        self.isTransparent = self.overlayToolbar.isTransparent()
        if not self.isTransparent: 
            # SVG export works only with RGBA not with setOpacity 
            # --> only true transparency mode can be used
            self.overlayToolbar.setTransparent(True)
            
        self.setDisabled(True)
        
        self.progressWin = apps.QDialogWorkerProgress(
            title='Exporting to video', parent=self.mainWin,
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
            dpi=preferences['dpi'],
            crop_outer_padding=False,
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
        
        self.setOverlaySegmMasksRgba()
        success = self.exportFrame()
        if success is None:
            self.exportingVideoCritical()
            return
        
        self.exportToVideoCurrentNavVarIdx += 1
        self.progressWin.mainPbar.update(1)

        QTimer.singleShot(50, self.updateAndExportFrame)

    @exception_handler
    def exportFrame(self):
        nd = self.exportToVideoPreferences['num_digits']
        idx = str(self.exportToVideoCurrentNavVarIdx).zfill(nd)
        filename = self.exportToVideoPreferences['filename']
        png_filename = f'{idx}_{filename}.png'
        pngs_folderpath = self.exportToVideoPreferences['pngs_folderpath']
        
        png_filepath = os.path.join(pngs_folderpath, png_filename)
        img_bgr = self.exportToVideoImageExporter.export(png_filepath)
        self.exportToVideoExporter.add_frame(img_bgr)
        return True

    def exportingVideoCritical(self):
        self.setDisabled(False)
        
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.progressWin = None
        
        self.logger.info('Exporting video process failed.')

    def exportingFramesFinished(self):            
        self.logger.info('Saving video...')
        self.exportToVideoExporter.release()
        
        # Run ffmpeg new process
        conversion_to_mp4_successful = True
        if self.exportToVideoPreferences['filepath'].endswith('.mp4'):
            try:
                self.exportToVideoExporter.avi_to_mp4(
                    crf=self.exportToVideoPreferences['crf']
                    )
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
            qparent=self
        )

        QTimer.singleShot(2000, self.removeExportVideoPngFolder)

    def removeExportVideoPngFolder(self):
        try:
            if not self.exportToVideoPreferences['save_pngs']:
                self.logger.info('Removing PNGs...')
                shutil.rmtree(self.exportToVideoPreferences['pngs_folderpath'])
        except Exception as err:
            traceback.print_exc()
            pass

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
            self, 'Z-slices or Timelapse video?', txt,
            buttonsTexts=('Z-slices', 'Timelapse')
        )
        if msg.cancel:
            return 
        
        return msg.clickedButton == timelapseButton

    def exportToCheckAskOverlay(self, output='image'):
        if not self.overlayButton.isChecked():
            return True
        
        if self.overlayToolbar.isTransparent():
            return True
        
        blinker = qutils.QControlBlink(
            self.overlayToolbar.transparencyCheckbox, qparent=self
        )
        blinker.start()
        
        cancel, activateTransparencyMode = (
            _warnings.warnAskTransparencyModeNeededForExport(
                self, output=output
            )
        )
        
        if cancel:
            return False
        
        if activateTransparencyMode:
            self.overlayToolbar.setTransparent(True)
            # The user chose it and it stays on after the export
            self.saveOverlayModes()

        return True

    def exportToVideoTriggered(self):
        proceed = self.exportToCheckAskOverlay(output='video')
        if not proceed:
            self.logger.info('Export to video process cancelled')
            return
        
        posData = self.data[self.pos_i]
        
        doTimelapseVideo = posData.SizeT > 1
        if posData.SizeT > 1 and posData.SizeZ > 1:
            doTimelapseVideo = self.askTimelapseOrZslicesVideo()
        
        if doTimelapseVideo is None:
            self.logger.info('Export to video process cancelled')
            return
        
        channels = [self.user_ch_name, *self.checkedOverlayChannels]
        mode = 'timelapse' if doTimelapseVideo else 'z_slices'
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{timestamp}_acdc_exported_{mode}_video'
        win = apps.ExportToVideoParametersDialog(
            channels,
            parent=self, 
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
        
        cancel = _warnings.warnExportToVideo(qparent=self)
        if cancel:
            self.logger.info('Export to video process cancelled')
            return
        
        self.startExportToVideoWorker(win.selected_preferences)        

    def setExportMaskImage(self, viewRange):
        if not hasattr(self, 'exportMaskImage'):
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

    def setOverlaySegmMasksRgba(self):
        # Use RGBA for overlay segm masks to make sure they are transparent
        isOverlaySegmLeftActive = self.isOverlaySegmMaskChecked(0)
        if not isOverlaySegmLeftActive:
            return

        posData = self.data[self.pos_i]
        alpha = self.imgGrad.labelsAlphaSlider.value()
        lut = self.labelsLayerImg1.lut
        labRgba = lut[self.currentLab2D]
        labRgba[:, :, 3] = np.round(labRgba[:, :, 3] * alpha).astype(np.uint8)

        self.labelsLayerImg1.setImage(labRgba)

    @disableWindow
    def exportToImage(self, preferences):      
        filepath = preferences['filepath']
        self.logger.info(f'Saving image to "{filepath}"...')

        self.setOverlaySegmMasksRgba()
        
        if filepath.endswith('.svg'):
            exporter = exporters.SVGExporter(self.ax1)
        else:
            exporter = exporters.ImageExporter(self.ax1, dpi=preferences['dpi'])
        
        exporter.export(filepath)
        self.logger.info(f'Image saved.')

        self.setOverlaySegmMasks()

        self.exportMaskImage[:] = 0
        self.exportMaskImageItem.setImage(self.exportMaskImage)

        QTimer.singleShot(300, partial(
            prompts.exportToImageFinished, filepath, qparent=self
        ))

    def exportToImageTriggered(self):
        proceed = self.exportToCheckAskOverlay()
        if not proceed:
            self.logger.info('Export to video process cancelled')
            return
        
        posData = self.data[self.pos_i]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'{timestamp}_acdc_exported_image'
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

    def exportCurrentViewDataToggled(self, checked: bool):
        self.exportCurrentViewToolbar.setVisible(checked)
