"""Mixin classes with the 'Viewer' mode tools of `cellacdc.gui.guiWin` 
shared with other windows (e.g., `cellacdc.signals_viewer.SignalsViewerWin`): 
zoom, ruler, scale bar and timestamp, color scheme, labels colormap, image 
properties, cell cycle table, objects count, search ID, hover info, keyboard 
shortcuts, default settings, slideshow, 3D viewer and region properties 
dock widget.
"""
import os
import re
import math
import logging
from typing import Tuple

import numpy as np
import skimage.segmentation

from qtpy.QtCore import (
    Qt, QRect, QRectF, QEventLoop, QObject, QThread, QTimer, PYQT6
)
from qtpy.QtGui import QKeySequence, QGuiApplication
from qtpy.QtWidgets import QDockWidget, QVBoxLayout

import pyqtgraph as pg

from . import exception_handler, disableWindow
from . import graphLayoutBkgrColor, darkBkgrColor
from . import load, apps, workers, html_utils
from . import myutils, widgets, colors
from . import _warnings
from . import settings_folderpath, settings_csv_path
from . import qutils, QtScoped
from . import transformation, core
from . import is_mac
from .cca_functions import _calc_rot_vol

shortcut_filepath = os.path.join(settings_folderpath, 'shortcuts.ini')

class ZoomGuiElements:
    def gui_createZoomRectItem(self):
        Y, X = self.currentLab2D.shape
        # Label ROI rectangle
        pen = pg.mkPen('r', width=3, style=Qt.DashLine)
        self.zoomRectItem = widgets.ZoomROI(
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
            self.timestampDialog = apps.TimestampPropertiesDialog(parent=self)
            self.timestampDialog.show()
            self.timestamp = widgets.TimestampItem(
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
            self.scaleBarDialog = apps.ScaleBarPropertiesDialog(
                X, Y, posData.PhysicalSizeX, parent=self
            )
            self.scaleBarDialog.show()
            self.scaleBar = widgets.ScaleBar(
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
        self.scaleBarDialog = apps.ScaleBarPropertiesDialog(
            X, Y, posData.PhysicalSizeX, parent=self, **properties
        )
        self.scaleBarDialog.sigValueChanged.connect(self.updateScaleBar)
        self.scaleBarDialog.exec_()

    def editTimestampRemove(self, timestamp):
        self.addTimestampAction.setChecked(False)

    def editTimestampProperties(self, properties):
        self.timestampDialog = apps.TimestampPropertiesDialog(
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

        self.objLabelAnnotRgb = tuple(
            self.textAnnot[0].item.colors()['label'][:3]
        )
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

class CcaTableGuiElements:
    def viewCcaTable(self):
        posData = self.data[self.pos_i]
        zoomIDs = self.getZoomIDs()
        
        df = posData.allData_li[posData.frame_i]['acdc_df']        
        current_cca_df = posData.cca_df
        if zoomIDs is not None:
            df = df.loc[zoomIDs]
            current_cca_df = current_cca_df.loc[zoomIDs]
            
        for column in current_cca_df.columns:
            header = (
                '================================================\n'
                f'CURRENT vs STORED `{column}` column '
                f'for frame number {posData.frame_i+1}:\n'
            )
            df_compare = current_cca_df[[column]].copy()
            df_compare[f'STORED_{column}'] = df[column]
            text = f'{header}{df_compare}'
            self.logger.info(text)
        
        if 'cell_cycle_stage' in df.columns:
            cca_df = df[self.cca_df_colnames]
            cca_df = cca_df.merge(
                current_cca_df, how='outer', left_index=True, right_index=True,
                suffixes=('_STORED', '_CURRENT')
            )
            cca_df = cca_df.reindex(sorted(cca_df.columns), axis=1)
            num_cols = len(cca_df.columns)
            for j in range(0,num_cols,2):
                df_j_x = cca_df.iloc[:,j]
                df_j_y = cca_df.iloc[:,j+1]
                if any(df_j_x!=df_j_y):
                    self.logger.info('------------------------')
                    self.logger.info('DIFFERENCES:')
                    diff_df = cca_df.iloc[:,j:j+2]
                    diff_mask = diff_df.iloc[:,0]!=diff_df.iloc[:,1]
                    self.logger.info(diff_df[diff_mask])
        else:
            cca_df = None
            self.logger.info(cca_df)
        self.logger.info('========================')
        if current_cca_df is None:
            return
        if current_cca_df.empty:
            msg = widgets.myMessageBox()
            txt = html_utils.paragraph(
                'Cell cycle annotations\' table is <b>empty</b>.<br>'
            )
            msg.warning(self, 'Table empty', txt)
            return
        
        df = posData.add_tree_cols_to_cca_df(
            current_cca_df, frame_i=posData.frame_i
        )
        if self.ccaTableWin is None:
            self.ccaTableWin = apps.ViewCcaTableWindow(df, parent=self)
            self.ccaTableWin.show()
            self.ccaTableWin.setGeometryWindow()
            self.ccaTableWin.sigUpdateCcaTable.connect(
                self.onSigUpdateCcaTableWindow
            )
        else:
            self.ccaTableWin.setFocus()
            self.ccaTableWin.activateWindow()
            self.ccaTableWin.updateTable(current_cca_df)

    def getZoomIDs(self, viewRange=None):
        if viewRange is None:
            viewRange = self.ax1.viewRange()
        
        lab = self.currentLab2D
        Y, X = lab.shape
        ((xmin, xmax), (ymin, ymax)) = viewRange
        if xmin <= 0 and ymin <= 0 and xmax >= X and ymax >= Y:
            posData = self.data[self.pos_i]
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
        zoomRp = self._acdcRegionProps(
            zoomLab, precache_centroids=False
        )
        zoomIDs = [obj.label for obj in zoomRp]
        return zoomIDs

    def onSigUpdateCcaTableWindow(self, *args):
        if not self.isDataLoaded:
            return
        
        if self.ccaTableWin is None:
            return
        
        viewRange = self.ax1.viewRange()
        posData = self.data[self.pos_i]
        zoomIDs = self.getZoomIDs(viewRange=viewRange)
        
        self.ccaTableWin.updateTable(posData.cca_df, IDs=zoomIDs)

class CountObjectsGuiElements:
    def countObjectsTimelapse(self):
        if self.countObjsWindow is None:
            activeCategories = {
                'In current frame', 
                'In all visited frames', 
                'In entire video', 
                'Unique objects in all visited frames', 
                'Unique objects in entire video'
            }
        else:
            activeCategories = self.countObjsWindow.activeCategories()
            
        posData = self.data[self.pos_i]        
        allCategoryCountMapper, _ = posData.countObjectsInSegmTimelapse(
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
            activeCategories = {
                'In current position', 
                'In all visited positions (current session)',
                'In all visited positions (previous sessions)',
                'In all loaded positions', 
            }
            if self.isSegm3D:
                activeCategories.add('In current z-slice')
        else:
            activeCategories = self.countObjsWindow.activeCategories()

        numObjectsCurrentPos = len(posData.IDs)
        numObjectsAllPos = 0
        numObjectsVisitedPosPrevious = 0
        numObjectsVisitedPosCurrent = 0
        numObjectsCurrentZslice = None
        if 'In current z-slice' in activeCategories:
            numObjectsCurrentZslice = len(
                self._acdcRegionProps(
                    self.currentLab2D, precache_centroids=False
                )
            )
        
        for pos_i, _posData in enumerate(self.data):
            IDs = _posData.allData_li[0]['regionprops'].IDs
            if os.path.exists(_posData.acdc_output_csv_path):
                numObjectsVisitedPosPrevious += len(IDs)
            if IDs:
                numObjs = len(IDs)
                numObjectsAllPos += len(IDs)
            else:
                lab = _posData.segm_data[0]
                rp = self._acdcRegionProps(
                    lab, precache_centroids=False
                )
                numObjs = len(rp)
                numObjectsAllPos += numObjs
                
            if _posData.visited:
                numObjectsVisitedPosCurrent += numObjs
        
        allCategoryCountMapper = {
            'In current position': numObjectsCurrentPos, 
            'In all visited positions (current session)': 
                numObjectsVisitedPosCurrent,
            'In all visited positions (previous sessions)': 
                numObjectsVisitedPosPrevious,
            'In all loaded positions': numObjectsAllPos, 
        }
        if numObjectsCurrentZslice is not None:
            allCategoryCountMapper['In current z-slice'] = (
                numObjectsCurrentZslice
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
        if self.countObjsWindow is None:
            return
        
        if not self.countObjsWindow.isVisible():
            return
        
        if not self.countObjsWindow.livePreviewCheckbox.isChecked():
            return
        
        categoryCountMapper = self.countObjects()
        self.countObjsWindow.updateCounts(categoryCountMapper)

    def countObjectsCb(self, checked):
        if self.countObjsWindow is None:
            categoryCountMapper = self.countObjects()
            self.countObjsWindow = apps.ObjectCountDialog(
                categoryCountMapper=categoryCountMapper, 
                parent=self,
                data=self.data
            )
            self.countObjsWindow.sigShowEvent.connect(self.updateObjectCounts)
            self.countObjsWindow.sigUpdateCounts.connect(self.updateObjectCounts)
            self.countObjsWindow.sigClose.connect(self.countObjsWindowClosed)
        if checked:
            self.countObjsWindow.show()
        else:
            self.countObjsWindow.hide()

    def countObjsWindowClosed(self, event):
        event.ignore()
        self.countObjsButton.setChecked(False)

class WorkersGuiElements:
    def workerProgress(self, text, loggerLevel='INFO'): # used in cca and lin tree
        if self.progressWin is not None:
            self.progressWin.logConsole.append(text)
        self.logger.log(getattr(logging, loggerLevel), text)

    def workerInitProgressbar(self, totalIter):
        self.progressWin.mainPbar.setValue(0)
        if totalIter == 1:
            totalIter = 0
        self.progressWin.mainPbar.setMaximum(totalIter)

    def workerUpdateProgressbar(self, step):
        self.progressWin.mainPbar.update(step)

    @exception_handler
    def workerCritical(self, out: Tuple[QObject, Exception]):
        self.setDisabled(False)
        try:
            worker, error = out
        except TypeError as err:
            error = out
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        self.logger.info(error)
        try:
            worker.thread().quit()
            worker.deleteLater()
            worker.thread().deleteLater()
        except Exception as err:
            # Worker already closed
            pass

        if isinstance(error, str):
            self.logger.critical(error)
            raise
        else:
            raise error

class SearchIDGuiElements:
    def findID(self, checked=False, ID=None):
        posData = self.data[self.pos_i]
        if ID is None:
            searchIDdialog = apps.FindIDDialog(
                title='Search object by ID',
                msg='Enter object ID to find and highlight',
                parent=self,
                isInteger=True
            )
            searchIDdialog.exec_()
            if searchIDdialog.cancel:
                return

            searchedID = searchIDdialog.EntryID
        else:
            searchedID = ID
        
        if searchedID in posData.IDs:
            self.goToObjectID(searchedID)
            return

        if posData.SizeT == 1:
            self.warnIDnotFound(searchedID)
            return
        
        if searchedID in posData.lost_IDs:
            self.goToLostObjectID(searchedID)
            return
        
        tracked_lost_IDs = self.getTrackedLostIDs()
        if searchedID in tracked_lost_IDs:
            self.goToAcceptedLostObjectID(searchedID)
            return
        
        self.logger.info(f'Searching ID {searchedID} in other frames...')
        
        frame_i_found = self.startSearchIDworker(searchedID)
        if frame_i_found is None:
            self.warnIDnotFound(searchedID)
            return
        
        self.logger.info(
            f'Object ID {searchedID} found at frame n. {frame_i_found+1}.'
        )
        proceed = self.askGoToFrameFoundID(searchedID, frame_i_found)
        if not proceed:
            return
        
        posData.frame_i = frame_i_found
        self.get_data()
        self.updateAllImages()
        self.updateScrollbars()
        
        self.goToObjectID(searchedID)

    @disableWindow
    def startSearchIDworker(self, searchedID):
        posData = self.data[self.pos_i]
        
        desc = 'Searching ID in all frames...'
        
        self.progressWin = apps.QDialogWorkerProgress(
            title=desc, parent=self.mainWin, pbarDesc=desc
        )
        self.progressWin.mainPbar.setMaximum(posData.SizeT)
        self.progressWin.show(self.app)
        
        self.searchIDthread = QThread()
        self.searchIDworker = workers.SimpleWorker(
            posData, self.searchIDworkerCallback, 
            func_args=(searchedID, )
        )
        self.searchIDworker.frame_i_found = None
        self.searchIDworker.moveToThread(self.searchIDthread)
        
        self.searchIDworker.signals.finished.connect(
            self.searchIDthread.quit
        )
        self.searchIDworker.signals.finished.connect(
            self.searchIDworker.deleteLater
        )
        self.searchIDthread.finished.connect(self.searchIDthread.deleteLater)
        
        self.searchIDworker.signals.critical.connect(
            self.searchIDworkerCritical
        )
        self.searchIDworker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.searchIDworker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.searchIDworker.signals.progress.connect(
            self.workerProgress
        )
        self.searchIDworker.signals.finished.connect(
            self.searchIDworkerFinished
        )
        
        self.searchIDthread.started.connect(self.searchIDworker.run)
        self.searchIDthread.start()
        
        self.searchIDworkerLoop = QEventLoop()
        self.searchIDworkerLoop.exec_()
        
        return self.searchIDworker.frame_i_found

    def searchIDworkerCritical(self, error):
        self.searchIDworkerLoop.exit()
        self.workerCritical(error)

    def searchIDworkerFinished(self):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        
        self.searchIDworkerLoop.exit()

    def searchIDworkerCallback(self, posData, searchedID):
        self.searchIDworker.signals.initProgressBar.emit(0)
        self.setAllIDs()
        self.searchIDworker.signals.initProgressBar.emit(posData.SizeT)
        frame_i_found = None
        for frame_i in range(len(posData.segm_data)):
            if frame_i >= len(posData.allData_li):
                break
            
            rp = posData.allData_li[frame_i]['regionprops']
            if rp is None:
                lab = posData.segm_data[frame_i]
                rp = self._acdcRegionProps(lab, precache_centroids=False)
                posData.allData_li[frame_i]['regionprops'] = rp
            if searchedID in rp.IDs:
                frame_i_found = frame_i
                break
            
            self.searchIDworker.signals.progressBar.emit(1)
            
        self.searchIDworker.frame_i_found = frame_i_found

    def warnIDnotFound(self, searchedID):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            Object ID {searchedID} was not found.<br><br>
        """)
        msg.warning(self, f'ID {searchedID} not found', txt)

    def askGoToFrameFoundID(self, searchedID, frame_i_found):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(f"""
            Object ID {searchedID} was found at frame n. {frame_i_found+1}.<br><br>
            Do you want to go to frame n. {frame_i_found+1}.
        """)
        noButton, yesButton = msg.information(
            self, f'ID {searchedID} found at frame n. {frame_i_found+1}', txt,
            buttonsTexts=(
                'No, stay on current frame', 
                f'Yes, go to frame n. {frame_i_found+1}'
            )
        )
        return msg.clickedButton == yesButton

    def setAllIDs(self, onlyVisited=False):
        for posData in self.data:
            posData.allIDs = set()
            for frame_i in range(len(posData.segm_data)):
                if frame_i >= len(posData.allData_li):
                    break
                
                lab = posData.allData_li[frame_i]['labels']
                if lab is None and onlyVisited:
                    break
                
                rp = posData.allData_li[frame_i]['regionprops']
                if rp is None:
                    lab = posData.segm_data[frame_i]
                    rp = self._acdcRegionProps(
                        lab, precache_centroids=False
                    )
                    posData.allData_li[frame_i]['regionprops'] = rp
                    
                posData.allIDs.update(rp.IDs)

    def onSearchTriggerBlink(self, target):
        button = getattr(self, target, None) if isinstance(target, str) else target
        if button is None:
            return
        
        targets = self.searchWidget.highlightTargets(button)
        blinker = qutils.QControlBlink(targets, qparent=self)
        blinker.start()

    def onSearchId(self, ID):
        if not self.isDataLoaded:
            self.logger.warning(
                'Data not loaded yet. Key pressing events are not connected.'
            )
            return
        self.findID(ID=ID)

    def focusOnSearch(self, dummy=None, blinking=False):
        self.searchWidget.search_input.setFocus()
        if blinking:
            blinker = qutils.QControlBlink(self.searchWidget, qparent=self)
            blinker.start()

class HoverInfoGuiElements:
    def _channelHoverValues(self, descr, channel, value, ff=None):
        if ff is None:
            n_digits = len(str(int(value)))
            ff = myutils.get_number_fstring_formatter(
                type(value), precision=abs(n_digits-5)
            )
        txt = (
            f'<b>{descr} {channel}</b>: value={value:{ff}}'
        )
        return txt

    def _addOverlayHoverValuesFormatted(self, txt, xdata, ydata):
        posData = self.data[self.pos_i]
        if posData.ol_data is None:
            return txt
        
        for filename in posData.ol_data:
            chName = myutils.get_chname_from_basename(
                filename, posData.basename, remove_ext=False
            )
            if chName not in self.checkedOverlayChannels:
                continue
            
            raw_overlay_img = self.getRawImage(filename=filename)
            raw_overlay_value = raw_overlay_img[ydata, xdata]
            # raw_overlay_max_value = raw_overlay_img.max()

            raw_txt = self._channelHoverValues('Raw', chName, raw_overlay_value)

            txt = f'{txt} | {raw_txt}'
        return txt

    def getActiveToolButton(self):
        for button in self.leftClickButtons:
            if button.isChecked():
                return button

    def getMouseDataCoordsRightImage(self):
        text = self.wcLabel.text()
        if not text:
            return
        
        ax_idx = int(re.findall(r'\(ax(\d)\)', text)[0])
        if ax_idx == 0:
            return
        
        coords = re.findall(r'x=(\d+), y=(\d+) \|', text)[0]
        
        return tuple([int(val) for val in coords])

    def updateValuesStatusBar(self):
        (xl, xr), (yt, yb) = self.ax1ViewRange(integers=True)
        W = round(xr - xl)
        H = round(yb - yt)
        txt = self.wcLabel.text()
        pattern = (
            r'W=.*?, H=.*? \| '
            r'x_left=.*?, y_top=.*? \| '
            r'x_right=.*?, y_bottom=.*? \| '
        )
        replacing = (
            f'W={W:d}, H={H:d} | '
            f'x_left={xl:d}, y_top={yt:d} | '
            f'x_right={xr:d}, y_bottom={yb:d} | '
        )
        txt = re.sub(pattern, replacing, txt)
        self.wcLabel.setText(txt)

    def hoverValuesFormatted(self, xdata, ydata, activeToolButton, is_ax0):  
        (xl, xr), (yt, yb) = self.ax1ViewRange(integers=True)
        W = round(xr - xl)
        H = round(yb - yt)
        ax_idx = 0 if is_ax0 else 1
        txt = (
            f'x={xdata:d}, y={ydata:d} | '
            f'W={W:d}, H={H:d} | '
            f'x_left={xl:d}, y_top={yt:d} | '
            f'x_right={xr:d}, y_bottom={yb:d} | '
            f'(ax{ax_idx})'
        )
        if activeToolButton == self.rulerButton:
            txt = self._addRulerMeasurementText(txt)
            return txt
        elif activeToolButton is not None:
            return txt

        posData = self.data[self.pos_i]

        raw_img = self.getRawImage()
        raw_value = raw_img[ydata, xdata]
        # raw_max_value = raw_img.max()

        ch = self.user_ch_name
        raw_txt = self._channelHoverValues('Raw', ch, raw_value)

        txt = f'{txt} | {raw_txt}'

        txt = self._addOverlayHoverValuesFormatted(txt, xdata, ydata)
        
        ID = self.currentLab2D[ydata, xdata]
        maxID = max(posData.IDs, default=0)

        num_obj = len(posData.IDs)
        lab_txt = (
            f'<b>Objects</b>: ID={ID}, <i>max ID={maxID}, '
            f'num. of objects={num_obj}</i>'
        )
        txt = f'{txt} | {lab_txt}'

        txt = self._addRulerMeasurementText(txt)
        return txt

    def showMirroredCursorToggled(self, checked):
        value = 'Yes' if checked else 'No'
        self.df_settings.at['showMirroredCursor', 'value'] = value
        self.df_settings.to_csv(settings_csv_path)
        
        if not checked:
            self.clearCursors()

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

class ShortcutsGuiElements:
    def initShortcuts(self):
        from . import config
        cp = config.ConfigParser()
        if os.path.exists(shortcut_filepath):
            cp.read(shortcut_filepath)
        
        if 'keyboard.shortcuts' not in cp:
            cp['keyboard.shortcuts'] = {}
        
        if cp.has_option('keyboard.shortcuts', 'Zoom out'):
            zoomOutKeyValueStr = cp['keyboard.shortcuts']['Zoom out']
            try:
                self.zoomOutKeyValue = int(zoomOutKeyValueStr)
            except Exception as err:
                self.logger.warning(
                    f'{zoomOutKeyValueStr} is not a valid key '
                    'zooming out action. Restoring default key "H".'
                )
        
        if 'delete_object.action' not in cp:
            self.delObjAction = None
        else:
            delObjKeySequenceText = cp['delete_object.action']['Key sequence']
            delObjButtonText = cp['delete_object.action']['Mouse button']
            delObjQtButton = (
                Qt.MouseButton.LeftButton if delObjButtonText == 'Left click'
                else Qt.MouseButton.MiddleButton
            )
            if not delObjKeySequenceText:
                delObjKeySequence = None
            else:
                delObjKeySequence = widgets.KeySequenceFromText(
                    delObjKeySequenceText
                )
            self.delObjToolAction.setChecked(True)
            self.delObjAction = delObjKeySequence, delObjQtButton
        shortcuts = {}
        for name, widget in self.widgetsWithShortcut.items():
            if name not in cp.options('keyboard.shortcuts'):
                if name in self.defaultMouseShortcuts:
                    button = self.defaultMouseShortcuts[name]
                    if PYQT6:
                        btn_name = button.name
                    else:
                        btn_name = QtScoped.mouse_button_name(button)
                    shortcut_text = f'Mouse {btn_name}' # for easier finding keep this pattern
                    self.mouseBindings[name] = button
                    shortcut = button
                else:
                    if hasattr(widget, 'keyPressShortcut'):
                        key = widget.keyPressShortcut
                        shortcut = widgets.KeySequenceFromText(key)
                    else:
                        shortcut = widget.shortcut()
                    shortcut_text = shortcut.toString()
                cp['keyboard.shortcuts'][name] = shortcut_text
            else:
                shortcut_text = cp['keyboard.shortcuts'][name]
                if shortcut_text.startswith('Mouse '):
                    button_name = shortcut_text.split('Mouse ')[-1]
                    if PYQT6:
                        button = Qt.MouseButton[button_name]
                    else:
                        button =  getattr(Qt, button_name)
                    self.mouseBindings[name] = button
                    shortcut = button
                else:
                    shortcut = widgets.KeySequenceFromText(shortcut_text)
            
            shortcuts[name] = (shortcut_text, shortcut)
        self.setShortcuts(shortcuts, save=False)
        with open(shortcut_filepath, 'w') as ini:
            cp.write(ini)

    def setShortcuts(self, shortcuts: dict, save=True):
        for name, (text, shortcut) in shortcuts.items():
            widget = self.widgetsWithShortcut[name]
            if shortcut is None or text.startswith('Mouse '):
                shortcut = QKeySequence()
            if hasattr(widget, 'keyPressShortcut'):
                widget.keyPressShortcut = shortcut
            else:
                widget.setShortcut(shortcut)
            s = widget.toolTip()
            toolTip = re.sub(r'Shortcut: "(.*)"', f'Shortcut: "{text}"', s)
            widget.setToolTip(toolTip)
        
        if not save: 
            return
        
        from . import config
        cp = config.ConfigParser()
        if os.path.exists(shortcut_filepath):
            cp.read(shortcut_filepath)
        
        if 'keyboard.shortcuts' not in cp:
            cp['keyboard.shortcuts'] = {}

        for name, (text, shortcut) in shortcuts.items():
            cp['keyboard.shortcuts'][name] = text
        
        cp['keyboard.shortcuts']['Zoom out'] = str(self.zoomOutKeyValue)
        
        if self.delObjAction is None:
            with open(shortcut_filepath, 'w') as ini:
                cp.write(ini)
            return
    
        delObjKeySequence, delObjQtButton = self.delObjAction
        try:
            if delObjKeySequence is None:
                delObjKeySequenceText = ''
            else:
                delObjKeySequenceText = delObjKeySequence.toString()
                
            delObjKeySequenceText = (
                delObjKeySequenceText
                .encode('ascii', 'ignore')
                .decode('utf-8')
            )
            delObjButtonText = (
                'Left click' if delObjQtButton == Qt.MouseButton.LeftButton
                else 'Middle click'
            )
            cp['delete_object.action'] = {
                'Key sequence': delObjKeySequenceText, 
                'Mouse button': delObjButtonText
            }
        except Exception as err:
            self.logger.warning(
                f'{delObjKeySequence} is not a valid keys sequence for '
                'deleting objects. Setting default action'
            )
            self.delObjAction = None
            cp.remove_section('delete_object.action')
            
        with open(shortcut_filepath, 'w') as ini:
            cp.write(ini)

    def editShortcuts_cb(self, highlighted_shortcut=None):
        if hasattr(self, 'showingShortcutEditorDialog') and self.showingShortcutEditorDialog:
            return
        self.showingShortcutEditorDialog = True
            
        if is_mac:
            delObjKeySequenceText = 'Ctrl'
            delObjButtonText = 'Left click'
        else:
            delObjKeySequenceText = ''
            delObjButtonText = 'Middle click'
            
        if self.delObjAction is not None:
            delObjKeySequence, delObjQtButton = self.delObjAction
            if delObjKeySequence is None:
                delObjKeySequenceText = ''
            else:
                delObjKeySequenceText = delObjKeySequence.toString()
            delObjKeySequenceText = (
                delObjKeySequenceText.encode('ascii', 'ignore').decode('utf-8')
            )
            delObjButtonText = (
                'Left click' if delObjQtButton == Qt.MouseButton.LeftButton
                else 'Middle click'
            )

        hard_shortcuts = self.getHardcodedShortcuts()
            
        win = apps.ShortcutEditorDialog(
            self.widgetsWithShortcut, 
            delObjectKey=delObjKeySequenceText,
            delObjectButton=delObjButtonText,
            zoomOutKeyValue=self.zoomOutKeyValue,
            parent=self,
            mouseBindings=self.mouseBindings,
            hard_shortcuts=hard_shortcuts,
            highlighted_shortcut=highlighted_shortcut
        )
        win.exec_()
        if win.cancel:
            self.showingShortcutEditorDialog = False
            return

        self.mouseBindings = win.mouseBindings
        self.delObjAction = win.delObjAction
        self.zoomOutKeyValue = win.zoomOutKeyValue
        self.setShortcuts(win.customShortcuts)
        self.showingShortcutEditorDialog = False

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
            self.slideshowWin = apps.imageViewer(
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
                self.slideshowWin.zSliceScrollBar.setSliderPosition(z_slice)
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

        if self.slideshowWin.linkWindow is None:
            return

        if not self.slideshowWin.linkWindowCheckbox.isChecked():
            return

        posData = self.data[self.pos_i]
        self.slideshowWin.frame_i = posData.frame_i
        self.slideshowWin.update_img()

class Volume3DViewerGuiElements:
    def launch3dViewerToggled(self, checked):
        self.logger.info('Launching 3D volume viewer...')
        
        myutils.check_install_package(
            'PyOpenGL',
            import_pkg_name='OpenGL',
            pypi_name='PyOpenGL',
            parent=self,
        )
        
        myutils.check_install_package(
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
            return
        
        posData = self.data[self.pos_i]
        
        volumes = {
            self.user_ch_name: posData.img_data[posData.frame_i]
        }
        lut_items_states = {
            self.user_ch_name: self.imgGrad.super_saveState()
        }
        
        ol_data = posData.ol_data
        if ol_data is None:
            ol_data = {}

        for filename, ol_image_data in ol_data.items():
            volume = ol_image_data[posData.frame_i]

            channel_name = myutils.get_chname_from_basename(
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
            parent=self,
            version=self._version,
            hide_on_close=False,
            logger_func=self.logger.info
        )
        self._volume_renderer.set_volumes(
            volumes,
            lut_items_states=lut_items_states,
            voxel_size=voxel_size,
        )
        self._volume_renderer.set_labels(
            posData.lab,
            gradient_item_state=self.labelsGrad.item.saveState(),
            SizeZ=posData.SizeZ
        ) 
        
        self.addPointsLayersToVolumeViewer(self._volume_renderer)
        
        self._volume_renderer.sigClose.connect(self.onClose3dViewer)
        self._volume_renderer.sigUpdate.connect(self.onUpdate3dViewer)
        self._volume_renderer.run(block=False)
        self._volume_renderer.raise_()
        self._volume_renderer.activateWindow()

    def onClose3dViewer(self, *args, **kwargs):
        self.launch3dViewerAction.setChecked(False)

    def onUpdate3dViewer(self):
        self.launch3dViewerAction.setChecked(False)
        self.launch3dViewerAction.setChecked(True)

class RegionPropsGuiElements:
    def gui_createRegionPropsDockWidget(self, side=Qt.LeftDockWidgetArea):
        self.propsDockWidget = QDockWidget('Cell-ACDC objects', self)
        self.guiTabControl = widgets.guiTabControl(self.propsDockWidget)

        # self.guiTabControl.setFont(_font)

        self.propsDockWidget.setWidget(self.guiTabControl)
        self.propsDockWidget.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetFloatable 
            | QDockWidget.DockWidgetFeature.DockWidgetMovable
        )
        self.propsDockWidget.setAllowedAreas(
            Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea
        )
        
        self.addDockWidget(side, self.propsDockWidget)
        self.propsDockWidget.hide()

    def gui_createShowPropsButton(self, side='left'):
        self.leftSideDocksLayout = QVBoxLayout()            
        self.leftSideDocksLayout.setSpacing(0)
        self.leftSideDocksLayout.setContentsMargins(0,0,0,0)
        self.rightSideDocksLayout = QVBoxLayout()            
        self.rightSideDocksLayout.setSpacing(0)
        self.rightSideDocksLayout.setContentsMargins(0,0,0,0)
        self.showPropsDockButton = widgets.expandCollapseButton()
        self.showPropsDockButton.setDisabled(True)
        self.showPropsDockButton.setFocusPolicy(Qt.NoFocus)
        self.showPropsDockButton.setToolTip('Show object properties')
        if side == 'left':
            self.leftSideDocksLayout.addWidget(self.showPropsDockButton)
        else:
            self.rightSideDocksLayout.addWidget(self.showPropsDockButton)

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
            obj = posData.rp.get_obj_from_ID(self.highlightedID)
            if obj is None:
                return
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
        obj = posData.rp.get_obj_from_ID(ID)
        if obj is None:
            s = f'Object ID {int(ID):d} does not exist'
            propsQGBox.notExistingIDLabel.setText(s)
            return
        
        self.goToZsliceSearchedID(obj)
        self.updatePropsWidget(int(ID))

    def updatePropsWidget(self, ID, fromHover=False):
        if isinstance(ID, str):
            # Function called by currentTextChanged of channelCombobox or
            # additionalMeasCombobox. We set self.currentPropsID = 0 to force update
            ID = self.guiTabControl.propsQGBox.idSB.value()
            self.currentPropsID = -1

        ID = int(ID)
        
        update = (
            self.propsDockWidget.isVisible()
            and ID != 0 and ID!=self.currentPropsID
        )
        if not update:
            return

        posData = self.data[self.pos_i]
        if not hasattr(posData, 'rp'):
            return 
        
        if posData.rp is None:
            self.update_rp() # IDK when can this happen?

        if not posData.IDs:
            # empty segmentation mask
            return

        if fromHover and not self.guiTabControl.highlightCheckbox.isChecked():
            # Do not highlight on hover
            return

        propsQGBox = self.guiTabControl.propsQGBox

        obj = posData.rp.get_obj_from_ID(ID)
        if obj is None:
            s = f'Object ID {int(ID):d} does not exist'
            propsQGBox.notExistingIDLabel.setText(s)
            return

        propsQGBox.notExistingIDLabel.setText('')
        self.currentPropsID = ID
        propsQGBox.idSB.setValue(ID)
        
        doHighlight = (
            self.guiTabControl.highlightCheckbox.isChecked()
            or self.guiTabControl.highlightSearchedCheckbox.isChecked()
        )
        if doHighlight:
            self.highlightSearchedID(ID)
        
        if self.isSegm3D:
            if self.zProjComboBox.currentText() == 'single z-slice':
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
        
        yx_pxl_to_um2 = PhysicalSizeY*PhysicalSizeX

        area_um2 = area_pxl*yx_pxl_to_um2

        propsQGBox.cellAreaUm2DSB.setValue(area_um2)

        if self.isSegm3D:
            PhysicalSizeZ = posData.PhysicalSizeZ
            vol_vox_3D = obj.area
            vol_fl_3D = vol_vox_3D*PhysicalSizeZ*PhysicalSizeY*PhysicalSizeX
            propsQGBox.cellVolVox3D_SB.setValue(vol_vox_3D)
            propsQGBox.cellVolFl3D_DSB.setValue(vol_fl_3D)

        vol_vox, vol_fl = _calc_rot_vol(
            obj, PhysicalSizeY, PhysicalSizeX
        )
        propsQGBox.cellVolVoxSB.setValue(int(vol_vox))
        propsQGBox.cellVolFlDSB.setValue(vol_fl)


        minor_axis_length = max(1, obj.minor_axis_length)
        elongation = obj.major_axis_length/minor_axis_length
        propsQGBox.elongationDSB.setValue(elongation)

        solidity = obj.solidity
        propsQGBox.solidityDSB.setValue(solidity)

        additionalPropName = propsQGBox.additionalPropsCombobox.currentText()
        additionalPropValue = getattr(obj, additionalPropName)
        propsQGBox.additionalPropsCombobox.indicator.setValue(additionalPropValue)

        intensMeasurQGBox = self.guiTabControl.intensMeasurQGBox
        selectedChannel = intensMeasurQGBox.channelCombobox.currentText()
        
        try:
            _, filename = self.getPathAndFilenameNoExtFromChName(selectedChannel, posData)
            image = posData.ol_data_dict[filename][posData.frame_i]
        except Exception as e:
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
        if funcDesc == 'Concentration':
            bkgrVal = np.median(img[posData.lab == 0])
            amount = func(objData, bkgrVal, obj.area)
            value = amount/vol_vox
        elif funcDesc == 'Amount':
            bkgrVal = np.median(img[posData.lab == 0])
            amount = func(objData, bkgrVal, obj.area)
            value = amount
        else:
            value = func(objData)

        intensMeasurQGBox.additionalMeasCombobox.indicator.setValue(value)

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

    def getHighlightedID(self):
        if self.highlightedID > 0:
            return self.highlightedID
        
        doHighlight = (
            self.propsDockWidget.isVisible() 
            and (
                self.guiTabControl.highlightCheckbox.isChecked()
                or self.guiTabControl.highlightSearchedCheckbox.isChecked()
            )
        )
        if not doHighlight:
            return 0
        
        return self.guiTabControl.propsQGBox.idSB.value()

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

class BottomLayoutGuiElements:
    """Resizable bottom layout (sliders and annotation options) and 
    double space bar press to toggle the annotations.
    
    The host window must create `self.bottomScrollArea` 
    (`widgets.ScrollArea`), `self.img1BottomGroupbox` and 
    `self.resizeBottomLayoutLine` (`widgets.VerticalResizeHline`).
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

class DefaultSettingsGuiElements:
    def restoreDefaultSettings(self):
        df = self.df_settings
        df.at['contLineWeight', 'value'] = 1
        df.at['mothBudLineSize', 'value'] = 1
        df.at['mothBudLineColor', 'value'] = (255, 165, 0, 255)
        df.at['contLineColor', 'value'] = (205, 0, 0, 220)

        self._updateContColour((205, 0, 0, 220))
        self._updateMothBudLineColour((255, 165, 0, 255))
        self._updateMothBudLineSize(1)
        self._updateContLineThickness()

        df.at['overlaySegmMasksAlpha', 'value'] = 0.3
        df.at['img_cmap', 'value'] = 'grey'
        self.imgCmap = self.imgGrad.cmaps['grey']
        self.imgCmapName = 'grey'
        self.labelsGrad.item.loadPreset('viridis')
        df.at['labels_bkgrColor', 'value'] = (25, 25, 25)
        
        if df.at['is_bw_inverted', 'value'] == 'Yes':
            self.invertBw(update=False)
        
        df = df[~df.index.str.contains('lab_cmap')]
        df.to_csv(self.settings_csv_path)
        self.imgGrad.restoreState(df)
        for items in self.overlayLayersItems.values():
            lutItem = items[1]
            lutItem.restoreState(df)

        self.labelsGrad.saveState(df)
        self.labelsGrad.restoreState(df, loadCmap=False)

        self.df_settings.to_csv(self.settings_csv_path)
        self.updateAllImages()
