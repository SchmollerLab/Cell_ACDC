"""Mixin classes with the 'Viewer' mode tools of `guiWin` shared with
other windows (e.g., `MeasurementsViewerWin`): zoom, ruler, scale bar
and timestamp, color scheme, labels colormap, image properties,
slideshow, 3D viewer and the bottom layout. The cell cycle table,
objects count, search ID, hover info and region properties dock are in
gui/mixins/object_info.py, the keyboard shortcuts and default settings
in gui/mixins/window_core.py.
"""
import os
import re
import math
import logging
from functools import partial
from typing import Tuple

import numpy as np
import skimage.segmentation

from qtpy.QtCore import (
    Qt, QRect, QRectF, QEventLoop, QObject, QThread, QTimer, PYQT6
)
from qtpy.QtGui import QKeySequence, QGuiApplication
from qtpy.QtWidgets import QDockWidget, QVBoxLayout

import pyqtgraph as pg

from cellacdc import exception_handler, disableWindow
from cellacdc import graphLayoutBkgrColor, darkBkgrColor
from cellacdc import load, html_utils
from cellacdc.shared.dialogs import image_tools as dialogs_image_tools
from . import dialogs as gui_mixin_dialogs
from cellacdc import colors
from cellacdc.myutils import filesystem, installs
from . import widgets as gui_mixin_widgets
from . import navigation as gui_navigation, overlay as gui_overlay
from cellacdc import _warnings
from cellacdc import settings_folderpath, settings_csv_path
from cellacdc import qutils, QtScoped
from cellacdc import transformation, core
from cellacdc import is_mac
from cellacdc.cca_functions import _calc_rot_vol
from cellacdc import shortcut_filepath

class ZoomGuiElements:
    def gui_createZoomRectItem(self):
        Y, X = self.currentLab2D.shape
        # Label ROI rectangle
        pen = pg.mkPen('r', width=3, style=Qt.DashLine)
        self.zoomRectItem = gui_mixin_widgets.ZoomROI(
            (0,0), (0,0),
            maxBounds=QRectF(QRect(0,0,X,Y)),
            scaleSnap=True,
            translateSnap=True,
            pen=pen, hoverPen=pen
        )

    def zoomRectActionToggled(self, checked):
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            self.connectLeftClickButtons()
            self.ax1.addItem(self.zoomRectItem)
        else:
            self.zoomRectItem.setPos((0,0))
            self.zoomRectItem.setSize((0,0))
            self.ax1.removeItem(self.zoomRectItem)

    def zoomRectDone(self):
        xRange, yRange = self.ax1.viewRange()
        self.zoomRectItem.storeLastRange(xRange, yRange)
        
        ymin, xmin, ymax, xmax = self.zoomRectItem.bbox()
        
        self.zoomRectItem.setPos((0,0))
        self.zoomRectItem.setSize((0,0))
        
        self.ax1.setRange(
            xRange=(xmin, xmax), 
            yRange=(ymin, ymax), 
            padding=0
        )

    def zoomRectCancelled(self):
        self.isMouseDragImg1 = False
        self.zoomRectItem.setPos((0,0))
        self.zoomRectItem.setSize((0,0))

    def zoomOut(self):
        self.ax1.autoRange()

    def zoomToObjsActionCallback(self):
        self.zoomToCells(enforce=True)

    def zoomToCells(self, enforce=False):
        if not self.enableAutoZoomToCellsAction.isChecked() and not enforce:
            return

        posData = self.data[self.pos_i]
        lab_mask = (self.currentLab2D>0).astype(np.uint8)
        rp = self._acdcRegionProps(lab_mask, precache_centroids=False)
        if not rp:
            Y, X = lab_mask.shape
            xRange = -0.5, X+0.5
            yRange = -0.5, Y+0.5
            self.ax1.setRange(xRange=xRange, yRange=yRange)
        else:
            self.setViewRangeToObjBbox(rp[0].bbox, padding=10)

class RulerGuiElements:
    def ruler_cb(self, checked):
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            self.connectLeftClickButtons()
        else:
            self.tempSegmentON = False
            self.ax1_rulerPlotItem.setData([], [])
            self.ax1_rulerAnchorsItem.setData([], [])

    def drawTempRulerLine(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        x, y = event.pos()
        x1, y1 = int(x), int(y)
        xxRA, yyRA = self.ax1_rulerAnchorsItem.getData()
        x0, y0 = xxRA[0], yyRA[0]
        if ctrl:
            x1, y1 = transformation.snap_xy_to_closest_angle(
                x0, y0, x1, y1
            )
        self.ax1_rulerPlotItem.setData([x0, x1], [y0, y1])

    def getRulerLengthText(self):
        text = self.wcLabel.text()
        lengthText = re.findall(r'length = (.*)\)', text)[0]
        lengthText = lengthText.replace('pxl', 'pixels')
        return f'{lengthText})'

    def _addRulerMeasurementText(self, txt):
        posData = self.data[self.pos_i]
        xx, yy = self.ax1_rulerPlotItem.getData()
        if xx is None:
            return txt

        lenPxl = math.sqrt((xx[0]-xx[1])**2 + (yy[0]-yy[1])**2)
        depthAxes = self.switchPlaneCombobox.depthAxes()
        if depthAxes != 'z':
            pxlToUm = posData.PhysicalSizeZ
        else:
            pxlToUm = posData.PhysicalSizeX
        
        length_txt = (
            f'length = {int(lenPxl)} pxl ({lenPxl*pxlToUm:.2f} μm)'
        )
        txt = f'{txt} | <b>Measurement</b>: {length_txt}'
        return txt

class ScaleBarTimestampGuiElements:
    def addTimestamp(self, checked):
        if checked:
            posData = self.data[self.pos_i]
            Y, X = self.img1.image.shape[:2]
            viewRange = self.ax1ViewRange()
            self.timestampDialog = gui_mixin_dialogs.TimestampPropertiesDialog(parent=self)
            self.timestampDialog.show()
            self.timestamp = gui_mixin_widgets.TimestampItem(
                Y, X, viewRange, 
                secondsPerFrame=posData.TimeIncrement,
                start_timedelta=self.timestampStartTimedelta
            )
            self.timestamp.sigEditProperties.connect(
                self.editTimestampProperties
            )
            self.timestamp.sigRemove.connect(
                self.editTimestampRemove
            )
            self.timestamp.addToAxis(self.ax1)
            self.timestamp.draw(
                posData.frame_i, **self.timestampDialog.kwargs()
            )
            self.timestampDialog.sigValueChanged.connect(self.updateTimestamp)
            self.timestampDialog.exec_()
            if self.timestampDialog is None:
                # Removed while the (non-modal) dialog was open, e.g. by
                # loading other data in the measurements viewer
                return
        else:
            self.timestamp.removeFromAxis(self.ax1)
        
        self.timestampDialog = None
        self.imgGrad.addTimestampAction.setChecked(checked)

    def ax1ViewRange(self, integers=False):
        if self.exportToImageWindow is None:
            viewRange = self.ax1.viewRange()
        else:
            exportMask = np.all(self.exportMaskImage == [0, 0, 0, 0], axis=-1)
            if np.all(exportMask):
                viewRange = self.ax1.viewRange()
            else:
                viewRange = self.ax1.viewRange(exportMask)
        
        if not integers:
            return viewRange
        
        xRange, yRange = viewRange
        xmin = round(xRange[0])
        ymin = round(yRange[0])
        xmax = round(xRange[1])
        ymax = round(yRange[1])
        return [xmin, xmax], [ymin, ymax]

    def addScaleBar(self, checked):
        if checked:
            posData = self.data[self.pos_i]
            Y, X = self.img1.image.shape[:2]
            viewRange = self.ax1ViewRange()
            self.scaleBarDialog = gui_mixin_dialogs.ScaleBarPropertiesDialog(
                X, Y, posData.PhysicalSizeX, parent=self
            )
            self.scaleBarDialog.show()
            self.scaleBar = gui_mixin_widgets.ScaleBar(
                (Y, X), viewRange, parent=self.ax1
            )
            self.scaleBar.sigEditProperties.connect(self.editScaleBarProperties)
            self.scaleBar.sigRemove.connect(
                self.editScaleBarRemove
            )
            self.scaleBar.addToAxis(self.ax1)
            self.scaleBar.draw(**self.scaleBarDialog.kwargs())
            self.scaleBarDialog.sigValueChanged.connect(self.updateScaleBar)
            self.scaleBarDialog.exec_()
            if self.scaleBarDialog is None:
                # Removed while the (non-modal) dialog was open, e.g. by
                # loading other data in the measurements viewer
                return
            if self.scaleBarDialog.cancel:
                self.addScaleBarAction.setChecked(False)
                return
        else:
            self.scaleBar.removeFromAxis(self.ax1)

        self.scaleBarDialog = None
        self.imgGrad.addScaleBarAction.blockSignals(True)
        self.imgGrad.addScaleBarAction.setChecked(checked)
        self.imgGrad.addScaleBarAction.blockSignals(False)

    def updateScaleBar(self, scaleBarKwargs):
        self.scaleBar.draw(**scaleBarKwargs)

    def updateTimestamp(self, timeStampKwargs):
        posData = self.data[self.pos_i]
        self.timestamp.draw(posData.frame_i, **timeStampKwargs)

    def editScaleBarRemove(self, timestamp):
        self.addScaleBarAction.setChecked(False)

    def editScaleBarProperties(self, properties):
        Y, X = self.img1.image.shape[:2]
        posData = self.data[self.pos_i]
        self.scaleBarDialog = gui_mixin_dialogs.ScaleBarPropertiesDialog(
            X, Y, posData.PhysicalSizeX, parent=self, **properties
        )
        self.scaleBarDialog.sigValueChanged.connect(self.updateScaleBar)
        self.scaleBarDialog.exec_()

    def editTimestampRemove(self, timestamp):
        self.addTimestampAction.setChecked(False)

    def editTimestampProperties(self, properties):
        self.timestampDialog = gui_mixin_dialogs.TimestampPropertiesDialog(
            parent=self, **properties
        )
        self.timestampDialog.sigValueChanged.connect(self.updateTimestamp)
        self.timestampDialog.show()

    def checkHighlightTimestamp(self, x, y, activeToolButton):
        if not hasattr(self, 'timestamp'):
            return
        
        if not self.addTimestampAction.isChecked():
            return
        
        if activeToolButton is not None:
            return
        
        if hasattr(self, 'scaleBar'):
            if self.scaleBar.isHighlighted():
                return
        
        ymin, xmin, ymax, xmax = self.timestamp.bbox()
        if x < xmin:
            self.timestamp.setHighlighted(False)
            return
        
        if x > xmax:
            self.timestamp.setHighlighted(False)
            return
        
        if y < ymin:
            self.timestamp.setHighlighted(False)
            return
        
        if y > ymax:
            self.timestamp.setHighlighted(False)
            return

        self.timestamp.setHighlighted(True)

    def checkHighlightScaleBar(self, x, y, activeToolButton):
        if not hasattr(self, 'scaleBar'):
            return
        
        if not self.addScaleBarAction.isChecked():
            return
        
        if activeToolButton is not None:
            return
        
        ymin, xmin, ymax, xmax = self.scaleBar.bbox()
        if x < xmin:
            self.scaleBar.setHighlighted(False)
            return
        
        if x > xmax:
            self.scaleBar.setHighlighted(False)
            return
        
        if y < ymin:
            self.scaleBar.setHighlighted(False)
            return
        
        if y > ymax:
            self.scaleBar.setHighlighted(False)
            return

        self.scaleBar.setHighlighted(True)

    def updateTimestampFrame(self):
        if not hasattr(self, 'timestamp'):
            return
        
        if not self.addTimestampAction.isChecked():
            return
        
        posData = self.data[self.pos_i]
        self.timestamp.setText(posData.frame_i)

    def viewRangeChanged(self, viewBox, viewRange, updateExportImageMask=True):
        # self.updateViewRangeExportToImage(viewRange) 
        self.updateValuesStatusBar()
               
        if hasattr(self, 'scaleBar'):
            isScaleBarMoveWithZoom = (
                self.scaleBar.properties()['move_with_zoom']
            )
        else:
            isScaleBarMoveWithZoom = False
        doMoveScaleBar = (
            self.scaleBarDialog is not None or isScaleBarMoveWithZoom
        )
        if doMoveScaleBar:
            self.scaleBar.updatePosViewRangeChanged(viewRange)
        
        if hasattr(self, 'timestamp'):
            isTimestampMoveWithZoom = (
                self.timestamp.properties()['move_with_zoom']
            )
        else:
            isTimestampMoveWithZoom = False
            
        doMoveTimestamp = (
            self.timestampDialog is not None or isTimestampMoveWithZoom
        )
        if doMoveTimestamp:
            self.timestamp.updatePosViewRangeChanged(viewRange)

        self._viewRange = viewRange

        self.update3DViewerViewRegion()

class ColorSchemeGuiElements:
    def gui_updateSwitchColorSchemeActionText(self):
        if self._colorScheme == 'dark':
            txt = 'Switch to light theme'
        else:
            txt = 'Switch to dark theme'
        self.toggleColorSchemeAction.setText(txt)

    def onToggleColorScheme(self):
        if self.toggleColorSchemeAction.text().find('light') != -1:
            self._colorScheme = 'light'
            setDarkModeToggleChecked = False
        else:
            self._colorScheme = 'dark'
            setDarkModeToggleChecked = True
        self.gui_updateSwitchColorSchemeActionText()
        _warnings.warnRestartCellACDCcolorModeToggled(
            self._colorScheme, app_name=self._appName, parent=self
        )
        load.rename_qrc_resources_file(self._colorScheme)
        self.statusBarLabel.setText(html_utils.paragraph(
            f'<i>Restart {self._appName} for the change to take effect</i>', 
            font_color='red'
        ))
        self.df_settings.at['colorScheme', 'value'] = self._colorScheme
        self.df_settings.to_csv(settings_csv_path)

    def invertBw(self, checked, update=True):
        self.invertBwAlreadyCalledOnce = True
        
        try:
            self.labelsGrad.invertBwAction.toggled.disconnect()
        except Exception as err:
            pass
        
        self.labelsGrad.invertBwAction.setChecked(checked)
        self.labelsGrad.invertBwAction.toggled.connect(self.setCheckedInvertBW)

        try:
            self.imgGrad.invertBwAction.toggled.disconnect()
        except Exception as err:
            pass
        self.imgGrad.invertBwAction.setChecked(checked)
        self.imgGrad.invertBwAction.toggled.connect(self.setCheckedInvertBW)

        self.imgGrad.setInvertedColorMaps(checked)
        self.imgGrad.invertCurrentColormap(checked)

        self.imgGradRight.setInvertedColorMaps(checked)
        self.imgGradRight.invertCurrentColormap(checked)

        if hasattr(self, 'overlayLayersItems'):
            for items in self.overlayLayersItems.values():
                lutItem = items[1]
                lutItem.invertBwAction.toggled.disconnect()
                lutItem.invertBwAction.setChecked(checked)
                lutItem.invertBwAction.toggled.connect(self.setCheckedInvertBW)
                lutItem.setInvertedColorMaps(checked)

        if self.slideshowWin is not None:
            self.slideshowWin.is_bw_inverted = checked
            self.slideshowWin.update_img()
        self.df_settings.at['is_bw_inverted', 'value'] = 'Yes' if checked else 'No'
        self.df_settings.to_csv(self.settings_csv_path)
        if checked:
            # Light mode
            self.equalizeHistPushButton.setStyleSheet('')
            self.graphLayout.setBackground(graphLayoutBkgrColor)
            self.ax2_BrushCirclePen = pg.mkPen((150,150,150), width=2)
            self.ax2_BrushCircleBrush = pg.mkBrush((200,200,200,150))
            self.titleColor = 'black'    
        else:
            # Dark mode
            self.equalizeHistPushButton.setStyleSheet(
                'QPushButton {background-color: #282828; color: #F0F0F0;}'
            )
            self.graphLayout.setBackground(darkBkgrColor)
            self.ax2_BrushCirclePen = pg.mkPen(width=2)
            self.ax2_BrushCircleBrush = pg.mkBrush((255,255,255,50))
            self.titleColor = 'white'
        
        if not hasattr(self, 'textAnnot'):
            return
        
        self.textAnnot[0].invertBlackAndWhite()
        self.textAnnot[1].invertBlackAndWhite()

        # Keep all inverted colors (`annotate.INVERTIBLE_COLOR_NAMES`): text
        # items created later (`gui_setTextAnnotColors`, e.g. when the
        # viewer switches position) must get the current scheme
        textColors = self.textAnnot[0].item.colors()
        self.objLabelAnnotRgb = tuple(textColors['label'][:3])
        self.SphaseAnnotRgb = tuple(textColors['S_phase_mother'][:3])
        self.G1phaseAnnotRgba = tuple(textColors['G1_phase'])
        self.textIDsColorButton.setColor(self.objLabelAnnotRgb)
        self.imgGrad.textColorButton.setColor(self.objLabelAnnotRgb)
        for items in self.overlayLayersItems.values():
            lutItem = items[1]
            lutItem.textColorButton.setColor(self.objLabelAnnotRgb)
        
        if update:
            self.updateAllImages()

    def setCheckedInvertBW(self, checked):
        self.invertBwAction.setChecked(checked)

class LabelsColormapGuiElements:
    def shuffle_cmap(self):
        np.random.shuffle(self.lut[1:])
        self.initLabelsImageItems()
        self.updateAllImages()

    def setPermanentGreedyCmapPreferences(self):
        if self.isSnapshot:
            option_name = 'permanent_greedy_lut_snapshots'
        else:
            option_name = 'permanent_greedy_lut_timelapse'

        if option_name not in self.df_settings.index:
            return
        
        checked = self.df_settings.at[option_name, 'value'] == 'yes'
        self.labelsGrad.permanentGreedyCmapAction.setChecked(checked)

    def permanentGreedyCmapToggled(self, checked):
        if checked:
            settings_value = 'yes'
        else:
            self.setLut()
            self.updateLookuptable()
            self.initLabelsImageItems()
            settings_value = 'no'
        
        self.updateAllImages()
        
        if self.isSnapshot:
            option_name = 'permanent_greedy_lut_snapshots'
        else:
            option_name = 'permanent_greedy_lut_timelapse'
            
        self.df_settings.at[option_name, 'value'] = settings_value
        self.df_settings.to_csv(self.settings_csv_path)

    def greedyShuffleCmap(self, updateImages=True):
        lut = self.labelsGrad.item.colorMap().getLookupTable(0,1,255)
        greedy_lut = colors.get_greedy_lut(self.currentLab2D, lut)
        self.lut = greedy_lut
        self.initLabelsImageItems()
        if updateImages:
            self.updateAllImages()

    def updateLabelsCmap(self, gradient):
        self.setLut()
        self.updateLookuptable()
        self.initLabelsImageItems()

        self.df_settings = self.labelsGrad.saveState(self.df_settings)
        self.df_settings.to_csv(self.settings_csv_path)

        self.updateAllImages()

    def saveLabelsColormap(self):
        self.labelsGrad.saveColormap()

class ImagePropertiesGuiElements:
    def editImgProperties(self, checked=True):
        posData = self.data[self.pos_i]
        posData.askInputMetadata(
            len(self.data),
            ask_SizeT=True,
            ask_TimeIncrement=True,
            ask_PhysicalSizes=True,
            save=True, singlePos=True,
            askSegm3D=False
        )
        if hasattr(self, 'timestamp'):
            self.timestamp.setSecondsPerFrame(posData.TimeIncrement)
            self.updateTimestampFrame()
        
        if hasattr(self, 'scaleBar'):
            self.scaleBar.updatePhysicalLength(posData.PhysicalSizeX)

class SlideshowGuiElements:
    def determineSlideshowWinPos(self):
        screens = self.app.screens()
        self.numScreens = len(screens)
        winScreen = self.screen()

        # Center main window and determine location of slideshow window
        # depending on number of screens available
        if self.numScreens > 1:
            for screen in screens:
                if screen != winScreen:
                    winScreen = screen
                    break

        winScreenGeom = winScreen.geometry()
        winScreenCenter = winScreenGeom.center()
        winScreenCenterX = winScreenCenter.x()
        winScreenCenterY = winScreenCenter.y()
        winScreenLeft = winScreenGeom.left()
        winScreenTop = winScreenGeom.top()
        self.slideshowWinLeft = winScreenCenterX - int(850/2)
        self.slideshowWinTop = winScreenCenterY - int(800/2)

    def launchSlideshow(self):
        posData = self.data[self.pos_i]
        self.determineSlideshowWinPos()
        if self.slideshowButton.isChecked():
            self.slideshowWin = dialogs_image_tools.imageViewer(
                parent=self,
                button_toUncheck=self.slideshowButton,
                linkWindow=posData.SizeT > 1,
                enableOverlay=True, 
                enableMirroredCursor=True
            )
            self.slideshowWin.img.minMaxValuesMapper = (
                self.img1.minMaxValuesMapper
            )
            self.slideshowWin.img.setCurrentPosIndex(self.pos_i)
            h = self.navSpinBox.size().height()
            self.slideshowWin.framesScrollBar.setFixedHeight(h)
            self.slideshowWin.overlayButton.setChecked(
                self.overlayButton.isChecked()
            )
            self.slideshowWin.sigHoveringImage.connect(
                self.setMirroredCursorFromSecondWindow
            )
            if posData.SizeZ > 1:
                z_slice = self.zSliceScrollBar.sliderPosition()
                self.slideshowWin.img.setCurrentZsliceIndex(z_slice)
                # No `update_z_slice`: the z index, label and image are set
                # here, and it would write `segmInfo_df` (None in the
                # measurements viewer)
                self.slideshowWin.zSliceScrollBar.blockSignals(True)
                self.slideshowWin.zSliceScrollBar.setSliderPosition(z_slice)
                self.slideshowWin.zSliceScrollBar.blockSignals(False)
                self.slideshowWin.z_label.setText(
                    f'z-slice  {z_slice+1:02}/{posData.SizeZ}'
                )
            self.slideshowWin.update_img()
            self.slideshowWin.show(
                left=self.slideshowWinLeft, top=self.slideshowWinTop
            )
        else:
            self.slideshowWin.close()
            self.slideshowWin = None

    def setMirroredCursorFromSecondWindow(self, x, y):
        if x is None:
            xx, yy = [], []
        else:
            xx, yy = [x], [y]
        self.ax1_cursor.setData(xx, yy)
        if not self.isTwoImageLayout:
            return
        self.ax2_cursor.setData(xx, yy)

    def updateViewerWindow(self):
        if self.slideshowWin is None:
            return

        # False for single frames: no "Link" checkbox (the measurements
        # viewer calls this on every `updateImages`)
        if not self.slideshowWin.linkWindow:
            return

        if not self.slideshowWin.linkWindowCheckbox.isChecked():
            return

        posData = self.data[self.pos_i]
        self.slideshowWin.frame_i = posData.frame_i
        self.slideshowWin.update_img()

class Volume3DViewerGuiElements:
    def launch3dViewerToggled(self, checked):
        msg = 'Launching 3D volume viewer...'
        self.logger.info(msg)
        # The measurements viewer has no title label
        titleLabel = getattr(self, 'titleLabel', None)
        if titleLabel is not None:
            titleLabel.setText(msg)

        QTimer.singleShot(300, partial(self._launch3dViewer, checked))

    def _launch3dViewer(self, checked):
        installs.check_install_package(
            'PyOpenGL',
            import_pkg_name='OpenGL',
            pypi_name='PyOpenGL',
            parent=self,
        )
        
        installs.check_install_package(
            'VisPy',
            import_pkg_name='vispy',
            pypi_name='vispy',
            parent=self,
        )
        
        self.launch3dViewerAction.blockSignals(True)
        self.launch3dViewerAction.setChecked(checked)
        self.launch3dViewerAction.blockSignals(False)
        
        if not checked:
            volume_renderer = getattr(self, '_volume_renderer', None)
            self._volume_renderer = None
            if volume_renderer is not None:
                volume_renderer.force_close()
                # Free the window and the data it references (e.g., the
                # images, not copied), otherwise kept as child of mainWin
                volume_renderer.deleteLater()
            return
        
        posData = self.data[self.pos_i]
        
        volumes = {
            self.user_ch_name: posData.img_data
        }
        lut_items_states = {
            self.user_ch_name: self.imgGrad.super_saveState()
        }
        
        ol_data = posData.ol_data
        if ol_data is None:
            ol_data = {}

        for filename, ol_image_data in ol_data.items():
            volume = ol_image_data

            channel_name = filesystem.get_chname_from_basename(
                filename, posData.basename, remove_ext=False
            )
            volumes[channel_name] = volume
            
            overlayItems = self.overlayLayersItems[channel_name]
            lutItem = overlayItems[1]
            lut_items_states[channel_name] = lutItem.super_saveState()

        voxel_size = (
            posData.PhysicalSizeZ, posData.PhysicalSizeY, posData.PhysicalSizeX
        )
        
        from cellacdc.volume_renderer.canvas import VolumeRendererWindow
        
        self._volume_renderer = VolumeRendererWindow(
            app=self.app, 
            parent=self.mainWin,
            parent_gui=self,
            version=self._version,
            hide_on_close=False,
            logger_func=self.logger.info
        )
        self._volume_renderer.set_volumes(
            volumes,
            lut_items_states=lut_items_states,
            voxel_size=voxel_size,
            # The images are not modified while the viewer is open -->
            # no need to copy them (large timelapses take several GB)
            copy=False,
        )
        # Labels frames beyond the frames of the displayed volumes (e.g.,
        # segmentation file with more frames than the channel) would
        # be cropped away by the viewer --> do not collect them
        num_frames = min(
            len(posData.segm_data), self._volume_renderer.num_frames
        )
        labels = []
        for frame_i in range(num_frames):
            if frame_i == posData.frame_i:
                lab = posData.lab
            else:
                lab = posData.allData_li[frame_i]['labels']
                if lab is None:
                    lab = posData.segm_data[frame_i]
            labels.append(lab)

        from cellacdc.volume_renderer import utils as volume_renderer_utils

        # Stack the frames and get their max ID in parallel threads
        # (large timelapses take several GB)
        labels = volume_renderer_utils.stack_frames(labels)

        def _get_frame_max_ID(frame_i):
            try:
                return np.max(labels[frame_i])
            except Exception as err:
                return 1

        max_ID = max([1, *volume_renderer_utils.map_frames(
            _get_frame_max_ID, len(labels)
        )])
        lut = self.extendLabelsLUT(max_ID+1, return_lut=True)
        lut = lut[:max_ID+1]
        self._volume_renderer.set_labels(
            labels,
            gradient_item_state=self.labelsGrad.item.saveState(),
            lut=lut,
            SizeZ=posData.SizeZ,
            font_size=self.fontSizeSpinBox.value(),
            text_color=self.imgGrad.textColorButton.color(),
            # `labels` is a new array not used anymore here --> no need
            # to copy it (large timelapses take several GB)
            copy=False,
        )

        self.addPointsLayersToVolumeViewer(self._volume_renderer)

        # Applied by the renderer once its camera is set up in `show`
        self._update3DViewerViewRegion()

        self._volume_renderer.sigClose.connect(self.onClose3dViewer)
        self._volume_renderer.sigUpdate.connect(self.onUpdate3dViewer)
        self._volume_renderer.run(block=False)
        self._volume_renderer.raise_()
        self._volume_renderer.activateWindow()

        titleLabel = getattr(self, 'titleLabel', None)
        if titleLabel is not None:
            titleLabel.setText('3D viewer up and running!')

    def onClose3dViewer(self, *args, **kwargs):
        self.launch3dViewerAction.setChecked(False)

    def onUpdate3dViewer(self):
        self.launch3dViewerAction.setChecked(False)
        self.launch3dViewerAction.setChecked(True)

    def update3DViewer(self):
        volume_renderer = getattr(self, '_volume_renderer', None)
        if volume_renderer is None:
            return

        # The z-slice might be different in the new frame
        self.update3DViewerViewRegion()

        if not volume_renderer.are_frames_synced():
            return

        volume_renderer.set_frame_number(self.navigateScrollBar.value())

    def update3DViewerViewRegion(self, *args):
        """Send the region visible in ax1 to the 3D viewer (if open).

        Calls are throttled because this is called at every view range
        change (e.g., while panning).
        """
        if getattr(self, '_volume_renderer', None) is None:
            return

        timer = getattr(self, '_update3DViewerViewRegionTimer', None)
        if timer is None:
            timer = QTimer(self)
            timer.setSingleShot(True)
            timer.setInterval(30)
            timer.timeout.connect(self._update3DViewerViewRegion)
            self._update3DViewerViewRegionTimer = timer

        # Do not restart an active timer, otherwise the 3D viewer would not
        # be updated until the user stops panning/zooming. The current
        # state is read when the timer fires.
        if not timer.isActive():
            timer.start()

    def _get3DViewerPlaneDepthSlice(self):
        """Return the plane displayed in ax1 ('xy', 'zy', or 'zx') and the
        index of the displayed slice along the perpendicular axis (None for
        z-projections).
        """
        plane = self.switchPlaneCombobox.plane()
        isSingleSlice = (
            plane != 'xy'
            or self.zProjComboBox.currentText() == 'single z-slice'
        )
        depthSlice = None
        if isSingleSlice:
            depthSlice = self.zSliceScrollBar.sliderPosition()

        return plane, depthSlice

    def _update3DViewerViewRegion(self):
        volume_renderer = getattr(self, '_volume_renderer', None)
        if volume_renderer is None:
            return

        xRange, yRange = self.ax1.viewRange()
        plane, depthSlice = self._get3DViewerPlaneDepthSlice()

        volume_renderer.set_parent_view_region(
            xRange, yRange, plane=plane, depth_slice=depthSlice
        )

        # The slice (or plane) might have changed --> move the cursor too
        self._update3DViewerCursor()

    def update3DViewerCursor(self, xy=None):
        """Show a mirrored cursor in the 3D viewer (if open) at the position
        `xy` of the mouse cursor on the image. Pass None to hide it.

        Not throttled: the 3D viewer only repaints a small overlay around
        the cursor (the volumes are not re-rendered).
        """
        # Store also when the 3D viewer is closed to avoid showing a stale
        # position when it is opened
        self._3DViewerCursorXY = xy

        if getattr(self, '_volume_renderer', None) is None:
            return

        self._update3DViewerCursor()

    def _update3DViewerCursor(self):
        volume_renderer = getattr(self, '_volume_renderer', None)
        if volume_renderer is None:
            return

        xy = getattr(self, '_3DViewerCursorXY', None)
        if xy is None:
            volume_renderer.set_parent_cursor_pos(None, None)
            return

        plane, depthSlice = self._get3DViewerPlaneDepthSlice()
        volume_renderer.set_parent_cursor_pos(
            *xy, plane=plane, depth_slice=depthSlice
        )

class BottomLayoutGuiElements:
    """Resizable bottom layout (sliders and annotation options) and 
    double space bar press to toggle the annotations.
    
    The host window must create `self.bottomScrollArea` 
    (`widgets_basic_widgets.ScrollArea`), `self.img1BottomGroupbox` and 
    `self.resizeBottomLayoutLine` (`widgets_basic_widgets.VerticalResizeHline`).
    """
    # Settings key of the bottom layout height (windows with different 
    # bottom layouts should use different keys)
    bottomLayoutHeightSettingKey = 'bottomLayoutHeight'
    
    def onDoubleSpaceBar(self):
        if self.doubleSpaceBarState:
            self.restoreAnnotationsOptions()
            self.doubleSpaceBarState = False
        else:
            self.storeCurrentAnnotationsOptions()
            self.setDoNotAnnotate(True)
            self.doubleSpaceBarState = True

    def resizeBottomLayoutLineClicked(self, event):
        pass
        
    def resizeBottomLayoutLineDragged(self, event):
        if not self.img1BottomGroupbox.isVisible():
            return
        newBottomLayoutHeight = self.bottomScrollArea.minimumHeight() - event.y()
        if newBottomLayoutHeight <= 0:
            newBottomLayoutHeight = 1
        
        self.bottomScrollArea.setFixedHeight(newBottomLayoutHeight)
    
    def resizeBottomLayoutLineReleased(self):
        bottomLayoutHeight = max(1, self.bottomScrollArea.height())
        self.df_settings.at[self.bottomLayoutHeightSettingKey, 'value'] = str(
            bottomLayoutHeight
        )
        self.df_settings.to_csv(self.settings_csv_path)
        
        QTimer.singleShot(100, self.autoRange)
    
    def setBottomLayoutHeight(self):
        if self.bottomLayoutHeightSettingKey in self.df_settings.index:
            bottomLayoutHeight = int(
                self.df_settings.at[self.bottomLayoutHeightSettingKey, 'value']
            )
            self.bottomScrollArea.setFixedHeight(bottomLayoutHeight)

    def doubleKeySpacebarTimerCallback(self):
        if self.isKeyDoublePress:
            self.doubleKeyTimeElapsed = False
            return
        self.doubleKeyTimeElapsed = True
        self.countKeyPress = 0
