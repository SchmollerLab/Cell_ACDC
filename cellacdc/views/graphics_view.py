"""Qt view adapter for graphics item construction workflows."""

from __future__ import annotations

import traceback
from functools import partial

import cv2
import matplotlib
import numpy as np
import pyqtgraph as pg
import skimage.exposure
import skimage.measure
from natsort import natsorted
from qtpy.QtCore import QEventLoop, QRect, QRectF, Qt, QThread, QTimer
from qtpy.QtGui import QColor, QCursor, QFont
from qtpy.QtWidgets import QAction, QActionGroup, QLabel, QMenu
from qtpy.QtWidgets import QGraphicsProxyWidget, QPushButton

from cellacdc import (
    _warnings,
    annotate,
    apps,
    colors,
    html_utils,
    myutils,
    widgets,
    workers,
)
from cellacdc.viewmodels.graphics_viewmodel import GraphicsViewModel

_font = QFont()
_font.setPixelSize(11)


class GraphicsView:
    """Qt-facing adapter for graphics item construction workflows."""

    LEGACY_METHODS = (
        'defaultRescaleIntensLutActionToggled',
        'mousePressColorButton',
        'gui_addGraphicsItems',
        'gui_createTextAnnotColors',
        'gui_setTextAnnotColors',
        'gui_createPlotItems',
        'gui_createZoomRectItem',
        'gui_createLabelRoiItem',
        'gui_createOverlayColors',
        'gui_createOverlayItems',
        'addActionsLutItemContextMenu',
        'getOverlayItems',
        'removeAllItems',
        'clearAx2Items',
        'clearAx1Items',
        'clearOverlayLabelsItems',
        'clearAllItems',
        'createUserChannelNameAction',
        'createChannelNamesActions',
        'addFluoChNameContextMenuAction',
        'restoreDefaultColors',
        'segmNdimIndicatorClicked',
        'addAlphaScrollbar',
        'createOverlayContextMenu',
        'createOverlayLabelsContextMenu',
        'editOverlayLabelsAppearance',
        'createOverlayLabelsItems',
        'addOverlayLabelsToggled',
        'overlayLabelsDrawModeToggled',
        'overlayChannelToggled',
        'overlayLabels_cb',
        'askLabelsToOverlay',
        'showOverlayContextMenu',
        'showOverlayLabelsContextMenu',
        'setCheckedOverlayContextMenusActions',
        'enableOverlayWidgets',
        'hideOverlayLabelsItems',
        'showOverlayLabelsItems',
        'setOverlayLabelsItems',
        'getOverlayLabelsData',
        'loadOverlayLabelsData',
        'removeOverlayItems',
        'clearOverlayImageItems',
        'setOverlayColors',
        'loadOverlayData',
        'askSelectOverlayChannel',
        'setOverlaySingleChannel',
        'updateTransparentOverlayRgba',
        'setOverlayTransparency',
        'overlay_cb',
        'getOlImg',
        'setOverlayImages',
        'getOpacitiesFromAlphaScrollbarValues',
        'toggleOverlayColorButton',
        'toggleTextIDsColorButton',
        'updateTextAnnotColor',
        'saveTextIDsColors',
        'ticksCmapMoved',
        'updateLabelsCmap',
        'extendLabelsLUT',
        'initLookupTableLab',
        'getLabelsImageLut',
        'initLabelsImageItems',
        'updateLookuptable',
        'setLut',
        'shuffle_cmap',
        'setPermanentGreedyCmapPreferences',
        'permanentGreedyCmapToggled',
        'greedyShuffleCmap',
        'updateBkgrColor',
        'updateTextLabelsColor',
        'saveTextLabelsColor',
        'saveBkgrColor',
        'changeOverlayColor',
        'saveOverlayColor',
        'setValueLabelsAlphaSlider',
        'setOverlaySegmMasks',
        'setOverlayLabelsItemsVisible',
        'setRetainSizePolicyLutItems',
        'setOverlayChannelsToolbuttonsChecked',
        'setOverlayItemsVisible',
        'overlayChannelToolbuttonClicked',
        'setOverlayItemsOpacities',
        'initColormapOverlayLayerItem',
        'setOpacityOverlayLayersItems',
        'gui_getLostObjScatterItem',
        'gui_getTrackedLostObjScatterItem',
        '_gui_createGraphicsItems',
        'gui_createTextAnnotItems',
        'gui_addOverlayLayerItems',
        'gui_addTopLayerItems',
        'updateContoursImage',
        'setContoursImage',
        'getObjFromID',
        'setLostObjectContour',
        'setTrackedLostObjectContour',
        'updateLostContoursImage',
        'drawLostObjContoursImage',
        'updateLostTrackedContoursImage',
        'drawLostTrackedObjContoursImage',
        'getNearestLostObjID',
        'addObjContourToContoursImage',
        'clearObjContour',
        'setAllContoursImages',
        'setAllLostObjContoursImage',
        'setAllLostTrackedObjContoursImage',
        'getObjContours',
        'clearComputedContours',
        '_computeAllContours2D',
        'computeAllContours',
        'computeAllObjToObjCostPairs',
        '_computeAllObjToObjCostPairs',
        'computeAllObjCostPairsWorkerCritical',
        'computeAllObjCostPairsWorkerFinished',
        'gui_createMothBudLinePens',
        'imgGradLUTfinished_cb',
        'restoreDefaultSettings',
        'updateMothBudLineColour',
        '_updateMothBudLineColour',
        'saveMothBudLineColour',
        'mothBudLineWeightToggled',
        '_updateMothBudLineSize',
        'gui_createContourPens',
        'updateContColour',
        '_updateContColour',
        'saveContColour',
        'contLineWeightToggled',
        '_updateContLineThickness',
        'gui_createGraphicsItems',
        'gui_connectGraphicsEvents',
        'gui_initImg1BottomWidgets',
    )

    def __init__(self, host, view_model: GraphicsViewModel):
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

    def mousePressColorButton(self, event):
        posData = self.data[self.pos_i]
        items = list(self.checkedOverlayChannels)
        if len(items)>1:
            selectFluo = widgets.QDialogListbox(
                'Select image',
                'Select which fluorescence image you want to update the color of\n',
                items, multiSelection=False, parent=self.host
            )
            selectFluo.exec_()
            keys = selectFluo.selectedItemsText
            if selectFluo.cancel or not keys:
                return
            else:
                self.overlayColorButton.channel = keys[0]
        else:
            self.overlayColorButton.channel = items[0]
        self.overlayColorButton.selectColor()

    def gui_addGraphicsItems(self):
        # Auto image adjustment button
        proxy = QGraphicsProxyWidget()
        equalizeHistPushButton = QPushButton("Enhance contrast")
        widthHint = equalizeHistPushButton.sizeHint().width()
        equalizeHistPushButton.setMaximumWidth(widthHint)
        equalizeHistPushButton.setCheckable(True)
        if not self.invertBwAction.isChecked():
            equalizeHistPushButton.setStyleSheet(
                'QPushButton {background-color: #282828; color: #F0F0F0;}'
            )
        self.equalizeHistPushButton = equalizeHistPushButton
        proxy.setWidget(equalizeHistPushButton)
        self.graphLayout.addItem(proxy, row=0, col=0)
        self.equalizeHistPushButton = equalizeHistPushButton

        # Left image histogram
        self.imgGrad = widgets.myHistogramLUTitem(parent=self.host, name='image')
        self.imgGrad.restoreState(self.df_settings)
        self.lutItemsLayout.addItem(self.imgGrad, row=0, col=0)
        for action in self.imgGrad.rescaleActionGroup.actions():
            if action.text() == self.defaultRescaleIntensHow:
                action.setChecked(True)
            self.rescaleIntensMenu.addAction(action)

        # Colormap gradient widget
        self.labelsGrad = widgets.labelsGradientWidget(parent=self.host)
        try:
            stateFound = self.labelsGrad.restoreState(self.df_settings)
        except Exception as e:
            self.logger.exception(traceback.format_exc())
            print('======================================')
            self.logger.info(
                'Failed to restore previously used colormap. '
                'Using default colormap "viridis"'
            )
            self.labelsGrad.item.loadPreset('viridis')

        # Add actions to imgGrad gradient item
        self.imgGrad.gradient.menu.addAction(
            self.labelsGrad.showLabelsImgAction
        )
        self.imgGrad.gradient.menu.addAction(
            self.labelsGrad.showRightImgAction
        )
        self.imgGrad.gradient.menu.addAction(
            self.labelsGrad.showNextFrameAction
        )

        self.imgGrad.gradient.menu.addSeparator()

        self.imgGrad.gradient.menu.addMenu(self.exportMenu)

        # Add actions to view menu
        self.viewMenu.addAction(self.labelsGrad.showLabelsImgAction)
        self.viewMenu.addAction(self.labelsGrad.showRightImgAction)

        # Right image histogram
        self.imgGradRight = widgets.baseHistogramLUTitem(
            name='image', parent=self.host, gradientPosition='left'
        )
        self.imgGradRight.gradient.menu.addAction(
            self.labelsGrad.showLabelsImgAction
        )
        self.imgGradRight.gradient.menu.addAction(
            self.labelsGrad.showRightImgAction
        )
        self.imgGradRight.gradient.menu.addAction(
            self.labelsGrad.showNextFrameAction
        )

        self.imgGrad.setChildLutItem(self.imgGradRight)

        # Title
        self.titleLabel = pg.LabelItem(
            justify='center', color=self.titleColor, size='14pt'
        )
        self.graphLayout.addItem(self.titleLabel, row=0, col=1, colspan=2)

    def gui_createTextAnnotColors(self, r, g, b, custom=False):
        if custom:
            self.objLabelAnnotRgb = (int(r), int(g), int(b))
            self.SphaseAnnotRgb = (int(r*0.9), int(r*0.9), int(b*0.9))
            self.G1phaseAnnotRgba = (int(r*0.8), int(g*0.8), int(b*0.8), 220)
        else:
            self.objLabelAnnotRgb = (255, 255, 255) # white
            self.SphaseAnnotRgb = (229, 229, 229)
            self.G1phaseAnnotRgba = (204, 204, 204, 220)
        self.dividedAnnotRgb = (245, 188, 1) # orange

        self.emptyBrush = pg.mkBrush((0,0,0,0))
        self.emptyPen = pg.mkPen((0,0,0,0))

    def gui_setTextAnnotColors(self):
        self.textAnnot[0].setColors(
            self.objLabelAnnotRgb, self.dividedAnnotRgb, self.SphaseAnnotRgb,
            self.G1phaseAnnotRgba, self.objLostAnnotRgb, self.objLostTrackedAnnotRgb
        )

        self.textAnnot[1].setColors(
            self.objLabelAnnotRgb, self.dividedAnnotRgb, self.SphaseAnnotRgb,
            self.G1phaseAnnotRgba, self.objLostAnnotRgb, self.objLostTrackedAnnotRgb
        )


    def gui_createPlotItems(self):
        if 'textIDsColor' in self.df_settings.index:
            rgbString = self.df_settings.at['textIDsColor', 'value']
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.gui_createTextAnnotColors(r, g, b, custom=True)
            self.textIDsColorButton.setColor((r, g, b))
        else:
            self.gui_createTextAnnotColors(0,0,0, custom=False)

        if 'labels_text_color' in self.df_settings.index:
            rgbString = self.df_settings.at['labels_text_color', 'value']
            r, g, b = colors.rgb_str_to_values(rgbString)
            self.ax2_textColor = (r, g, b)
        else:
            self.ax2_textColor = (255, 0, 0)

        self.emptyLab = np.zeros((2,2), dtype=np.uint8)

        # Right image item linked to left
        self.rightImageItem = widgets.ChildImageItem(
            linkedScrollbar=self.rightImageFramesScrollbar
        )
        self.imgGradRight.setImageItem(self.rightImageItem)
        self.ax2.addItem(self.rightImageItem)

        # Left image
        self.img1 = widgets.ParentImageItem(
            linkedImageItem=self.rightImageItem,
            activatingActions=(
                self.labelsGrad.showRightImgAction,
                self.labelsGrad.showNextFrameAction
            )
        )
        self.imgGrad.setImageItem(self.img1)
        self.img1.lutItem = self.imgGrad
        self.imgGrad.sigRescaleIntes.connect(self.rescaleIntensitiesLut)
        self.ax1.addBaseImageItem(self.img1)

        # RGBA image for true transparency mode
        self.rgbaImg1 = pg.ImageItem()

        # self.rgbaImg1.setImage(self.emptyLab)

        # Right image
        self.img2 = widgets.labImageItem()
        self.ax2.addItem(self.img2)

        self.topLayerItems = []
        self.topLayerItemsRight = []

        self.gui_createContourPens()
        self.gui_createMothBudLinePens()

        self.eraserCirclePen = pg.mkPen(width=1.5, color='r')

        # Temporary line item connecting bud to new mother
        self.BudMothTempLine = pg.PlotDataItem(pen=self.NewBudMoth_Pen)
        self.topLayerItems.append(self.BudMothTempLine)

        # Temporary line item connecting objects to merge
        self.mergeObjsTempLine = widgets.PlotCurveItem(pen=self.redDashLinePen)
        self.topLayerItems.append(self.mergeObjsTempLine)

        # Overlay segm. masks item
        self.labelsLayerImg1 = widgets.BaseLabelsImageItem()
        self.ax1.addItem(self.labelsLayerImg1)

        self.labelsLayerRightImg = widgets.BaseLabelsImageItem()
        self.ax2.addItem(self.labelsLayerRightImg)

        # Red/green border rect item
        self.GreenLinePen = pg.mkPen(color='g', width=2)
        self.RedLinePen = pg.mkPen(color='r', width=2)
        self.ax1BorderLine = pg.PlotDataItem()
        self.topLayerItems.append(self.ax1BorderLine)
        self.ax2BorderLine = pg.PlotDataItem(pen=pg.mkPen(color='r', width=2))
        self.topLayerItems.append(self.ax2BorderLine)

        # Brush/Eraser/Wand.. layer item
        self.tempLayerRightImage = pg.ImageItem()
        self.tempLayerImg1 = widgets.ParentImageItem(
            linkedImageItem=self.tempLayerRightImage,
            activatingAction=(self.labelsGrad.showRightImgAction, )
        )
        self.topLayerItems.append(self.tempLayerImg1)
        self.topLayerItemsRight.append(self.tempLayerRightImage)

        # Highlighted ID layer items
        self.highLightIDLayerImg1 = pg.ImageItem()
        self.topLayerItems.append(self.highLightIDLayerImg1)

        # Highlighted ID layer items
        self.highLightIDLayerRightImage = pg.ImageItem()
        self.topLayerItemsRight.append(self.highLightIDLayerRightImage)

        # Keep IDs temp layers
        self.keepIDsTempLayerRight = pg.ImageItem()
        self.keepIDsTempLayerLeft = widgets.ParentImageItem(
            linkedImageItem=self.keepIDsTempLayerRight,
            activatingAction=self.labelsGrad.showRightImgAction
        )
        self.topLayerItems.append(self.keepIDsTempLayerLeft)
        self.topLayerItemsRight.append(self.keepIDsTempLayerRight)

        # Searched ID contour
        self.searchedIDitemRight = pg.ScatterPlotItem()
        self.searchedIDitemRight.setData(
            [], [], symbol='s', pxMode=False, size=1,
            brush=pg.mkBrush(color=(255,0,0,150)),
            pen=pg.mkPen(width=2, color='r'), tip=None
        )
        self.searchedIDitemLeft = pg.ScatterPlotItem()
        self.searchedIDitemLeft.setData(
            [], [], symbol='s', pxMode=False, size=1,
            brush=pg.mkBrush(color=(255,0,0,150)),
            pen=pg.mkPen(width=2, color='r'), tip=None
        )
        self.topLayerItems.append(self.searchedIDitemLeft)
        self.topLayerItemsRight.append(self.searchedIDitemRight)


        # Brush circle img1
        self.ax1_BrushCircle = pg.ScatterPlotItem()
        self.ax1_BrushCircle.setData(
            [], [], symbol='o', pxMode=False,
            brush=pg.mkBrush((255,255,255,50)),
            pen=pg.mkPen(width=2), tip=None
        )
        self.topLayerItems.append(self.ax1_BrushCircle)

        # Eraser circle img1
        self.ax1_EraserCircle = pg.ScatterPlotItem()
        self.ax1_EraserCircle.setData(
            [], [], symbol='o', pxMode=False,
            brush=None, pen=self.eraserCirclePen, tip=None
        )
        self.topLayerItems.append(self.ax1_EraserCircle)

        self.ax1_EraserX = pg.ScatterPlotItem()
        self.ax1_EraserX.setData(
            [], [], symbol='x', pxMode=False, size=3,
            brush=pg.mkBrush(color=(255,0,0,50)),
            pen=pg.mkPen(width=1, color='r'), tip=None
        )
        self.topLayerItems.append(self.ax1_EraserX)

        # Brush circle img1
        self.labelRoiCircItemLeft = widgets.LabelRoiCircularItem()
        self.labelRoiCircItemLeft.cleared = False
        self.labelRoiCircItemLeft.setData(
            [], [], symbol='o', pxMode=False,
            brush=pg.mkBrush(color=(255,0,0,0)),
            pen=pg.mkPen(color='r', width=2), tip=None
        )
        self.labelRoiCircItemRight = widgets.LabelRoiCircularItem()
        self.labelRoiCircItemRight.cleared = False
        self.labelRoiCircItemRight.setData(
            [], [], symbol='o', pxMode=False,
            brush=pg.mkBrush(color=(255,0,0,0)),
            pen=pg.mkPen(color='r', width=2), tip=None
        )
        self.topLayerItems.append(self.labelRoiCircItemLeft)
        self.topLayerItemsRight.append(self.labelRoiCircItemRight)

        self.ax1_binnedIDs_ScatterPlot = widgets.BaseScatterPlotItem()
        self.ax1_binnedIDs_ScatterPlot.setData(
            [], [], symbol='t', pxMode=False,
            brush=pg.mkBrush((255,0,0,50)), size=15,
            pen=pg.mkPen(width=3, color='r'), tip=None
        )
        self.topLayerItems.append(self.ax1_binnedIDs_ScatterPlot)

        self.ax1_ripIDs_ScatterPlot = widgets.BaseScatterPlotItem()
        self.ax1_ripIDs_ScatterPlot.setData(
            [], [], symbol='x', pxMode=False,
            brush=pg.mkBrush((255,0,0,50)), size=15,
            pen=pg.mkPen(width=2, color='r'), tip=None
        )
        self.topLayerItems.append(self.ax1_ripIDs_ScatterPlot)

        # Ruler plotItem and scatterItem
        rulerPen = pg.mkPen(color='r', style=Qt.DashLine, width=2)
        self.ax1_rulerPlotItem = widgets.RulerPlotItem(pen=rulerPen)
        self.ax1_rulerAnchorsItem = pg.ScatterPlotItem(
            symbol='o', size=9,
            brush=pg.mkBrush((255,0,0,50)),
            pen=pg.mkPen((255,0,0), width=2), tip=None
        )
        self.topLayerItems.append(self.ax1_rulerPlotItem)
        self.topLayerItems.append(self.ax1_rulerPlotItem.labelItem)
        self.topLayerItems.append(self.ax1_rulerAnchorsItem)

        # Start point of polyline roi
        self.ax1_point_ScatterPlot = pg.ScatterPlotItem()
        self.ax1_point_ScatterPlot.setData(
            [], [], symbol='o', pxMode=False, size=3,
            pen=pg.mkPen(width=2, color='r'),
            brush=pg.mkBrush((255,0,0,50)), tip=None
        )
        self.topLayerItems.append(self.ax1_point_ScatterPlot)

        # Experimental: scatter plot to add a point marker
        self.startPointPolyLineItem = pg.ScatterPlotItem()
        self.startPointPolyLineItem.setData(
            [], [], symbol='o', size=9,
            pen=pg.mkPen(width=2, color='r'),
            brush=pg.mkBrush((255,0,0,50)),
            hoverable=True, hoverBrush=pg.mkBrush((255,0,0,255)), tip=None
        )
        self.topLayerItems.append(self.startPointPolyLineItem)

        # Eraser circle img2
        self.ax2_EraserCircle = pg.ScatterPlotItem()
        self.ax2_EraserCircle.setData(
            [], [], symbol='o', pxMode=False, brush=None,
            pen=self.eraserCirclePen, tip=None
        )
        self.ax2.addItem(self.ax2_EraserCircle)
        self.ax2_EraserX = pg.ScatterPlotItem()
        self.ax2_EraserX.setData(
            [], [], symbol='x', pxMode=False, size=3,
            brush=pg.mkBrush(color=(255,0,0,50)),
            pen=pg.mkPen(width=1.5, color='r')
        )
        self.ax2.addItem(self.ax2_EraserX)

        # Brush circle img2
        self.ax2_BrushCirclePen = pg.mkPen(width=2)
        self.ax2_BrushCircleBrush = pg.mkBrush((255,255,255,50))
        self.ax2_BrushCircle = pg.ScatterPlotItem()
        self.ax2_BrushCircle.setData(
            [], [], symbol='o', pxMode=False,
            brush=self.ax2_BrushCircleBrush,
            pen=self.ax2_BrushCirclePen, tip=None
        )
        self.ax2.addItem(self.ax2_BrushCircle)

        # Annotated metadata markers (ScatterPlotItem)
        self.ax2_binnedIDs_ScatterPlot = widgets.BaseScatterPlotItem()
        self.ax2_binnedIDs_ScatterPlot.setData(
            [], [], symbol='t', pxMode=False,
            brush=pg.mkBrush((255,0,0,50)), size=15,
            pen=pg.mkPen(width=3, color='r'), tip=None
        )
        self.ax2.addItem(self.ax2_binnedIDs_ScatterPlot)

        self.ax2_ripIDs_ScatterPlot = widgets.BaseScatterPlotItem()
        self.ax2_ripIDs_ScatterPlot.setData(
            [], [], symbol='x', pxMode=False,
            brush=pg.mkBrush((255,0,0,50)), size=15,
            pen=pg.mkPen(width=2, color='r'), tip=None
        )
        self.ax2.addItem(self.ax2_ripIDs_ScatterPlot)

        self.freeRoiItem = widgets.PlotCurveItem(
            pen=pg.mkPen(color='r', width=2)
        )
        self.topLayerItems.append(self.freeRoiItem)

        self.warnPairingItem = widgets.PlotCurveItem(
            pen=pg.mkPen(color='r', width=5, style=Qt.DashLine),
            pxMode=False
        )
        self.topLayerItems.append(self.warnPairingItem)

        self.exportMaskImageItem = pg.ImageItem()

        self.ghostContourItemLeft = widgets.GhostContourItem(self.ax1)
        self.ghostContourItemRight = widgets.GhostContourItem(self.ax2)

        self.ghostMaskItemLeft = widgets.GhostMaskItem(self.ax1)
        self.ghostMaskItemRight = widgets.GhostMaskItem(self.ax2)

        self.manualBackgroundObjItem = widgets.GhostContourItem(
            self.ax1, penColor='r', textColor='r'
        )
        self.manualBackgroundImageItem = pg.ImageItem()

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

    def gui_createLabelRoiItem(self):
        Y, X = self.currentLab2D.shape
        # Label ROI rectangle
        pen = pg.mkPen('r', width=3)
        self.labelRoiItem = widgets.ROI(
            (0,0), (0,0),
            maxBounds=QRectF(QRect(0,0,X,Y)),
            scaleSnap=True,
            translateSnap=True,
            pen=pen, hoverPen=pen
        )

        posData = self.data[self.pos_i]
        if self.labelRoiZdepthSpinbox.value() == 0:
            self.labelRoiZdepthSpinbox.setValue(posData.SizeZ)
        self.labelRoiZdepthSpinbox.setMaximum(posData.SizeZ+1)

    def gui_createOverlayColors(self):
        fluoChannels = [ch for ch in self.ch_names if ch != self.user_ch_name]
        self.logger.info(
            f'Number of TIFF files detected: {len(fluoChannels)}'
        )
        self.overlayColors = {}
        for c, ch in enumerate(fluoChannels):
            if f'{ch}_rgb' in self.df_settings.index:
                rgb_text = self.df_settings.at[f'{ch}_rgb', 'value']
                rgb = tuple([int(val) for val in rgb_text.split('_')])
                self.overlayColors[ch] = rgb
            else:
                if c >= len(self.overlayRGBs) -1:
                    i = c/len(fluoChannels)
                    additional_color_num = c - len(self.overlayRGBs) + 1
                    rgbs = [
                        tuple([round(c*255) for c in self.overlayCmap(i)][:3])
                        for _ in range(additional_color_num)
                    ]
                    self.overlayRGBs.extend(rgbs)
                rgb = colors.FLUO_CHANNELS_COLORS.get(ch, self.overlayRGBs[c])
                self.overlayColors[ch] = rgb

    def gui_createOverlayItems(self):
        self.imgGrad.setAxisLabel(self.user_ch_name)
        self.baseLayerToolbutton = widgets.OverlayChannelToolButton(
            self.user_ch_name, self.imgGrad
        )
        self.baseLayerToolbutton.setChecked(True)
        self.baseLayerToolbutton.clicked.connect(
            self.overlayChannelToolbuttonClicked
        )
        self.allOverlayToolbuttons = {
            self.user_ch_name: self.baseLayerToolbutton
        }
        self.allOverlayToolbuttonsByIdx = {
            0: self.baseLayerToolbutton
        }
        self.baseLayerToolbutton.action = (
            self.overlayToolbar.addWidget(self.baseLayerToolbutton)
        )
        self.overlayLayersItems = {}
        self.overlayToolbarAreChannelsChecked = {}
        fluoChannels = [ch for ch in self.ch_names if ch != self.user_ch_name]
        for c, ch in enumerate(fluoChannels):
            overlayItems = self.getOverlayItems(ch, c+1)
            self.overlayLayersItems[ch] = overlayItems
            imageItem, lutItem = overlayItems[:2]
            self.ax1.addItem(imageItem)
            self.lutItemsLayout.addItem(lutItem, row=0, col=c+1)
            toolbutton = overlayItems[3]
            self.allOverlayToolbuttons[ch] = toolbutton
            self.allOverlayToolbuttonsByIdx[c+1] = toolbutton

        self.overlayToolbuttonsSep = self.overlayToolbar.addSeparator()
        self.plotsCol = len(self.ch_names)

        self.ax1.addImageItem(self.rgbaImg1)

    def addActionsLutItemContextMenu(self, lutItem):
        lutItem.gradient.menu.addSection('Visible channels: ')
        for action in self.overlayContextMenu.actions():
            if action.isSeparator():
                continue
            lutItem.gradient.menu.addAction(action)
        lutItem.gradient.menu.addSeparator()

        annotationMenu = lutItem.gradient.menu.addMenu('Annotations settings')
        ID_menu = annotationMenu.addMenu('IDs')
        self.annotSettingsIDmenu = QActionGroup(annotationMenu)
        labID_action = QAction("Show label's ID")
        labID_action.setCheckable(True)
        labID_action.setChecked(True)
        labID_action.toggled.connect(self.annotLabelIDtreeToggled)
        treeID_action = QAction("Show tree's ID")
        treeID_action.setCheckable(True)
        treeID_action.toggled.connect(self.annotLabelIDtreeToggled)
        self.annotSettingsIDmenu.addAction(labID_action)
        self.annotSettingsIDmenu.addAction(treeID_action)
        ID_menu.addAction(labID_action)
        ID_menu.addAction(treeID_action)

        ID_menu = annotationMenu.addMenu('Generation number')
        self.annotSettingsGenNumMenu = QActionGroup(annotationMenu)
        gen_num_action = QAction("Show default generation number")
        gen_num_action.setCheckable(True)
        gen_num_action.setChecked(True)
        gen_num_action.toggled.connect(self.annotGenNumTreeToggled)
        tree_gen_num_action = QAction("Show tree generation number")
        tree_gen_num_action.setCheckable(True)
        tree_gen_num_action.toggled.connect(self.annotGenNumTreeToggled)
        self.annotSettingsGenNumMenu.addAction(gen_num_action)
        self.annotSettingsGenNumMenu.addAction(tree_gen_num_action)
        ID_menu.addAction(gen_num_action)
        ID_menu.addAction(tree_gen_num_action)

    def getOverlayItems(self, channelName, index):
        imageItem = widgets.OverlayImageItem()
        imageItem.setOpacity(0.5)
        imageItem.channelName = channelName

        lutItem = widgets.myHistogramLUTitem(
            parent=self.host, name='image', axisLabel=channelName
        )
        imageItem.lutItem = lutItem
        for action in lutItem.rescaleActionGroup.actions():
            if action.text() == self.defaultRescaleIntensHow:
                action.setChecked(True)
            break

        lutItem.removeAddScaleBarAction()
        lutItem.removeAddTimestampAction()
        lutItem.restoreState(self.df_settings)
        lutItem.setImageItem(imageItem)
        lutItem.vb.raiseContextMenu = lambda x: None
        initColor = self.overlayColors[channelName]
        self.initColormapOverlayLayerItem(initColor, lutItem)
        lutItem.addOverlayColorButton(initColor, channelName)
        lutItem.initColor = initColor
        lutItem.hide()

        lutItem.overlayColorButton.sigColorChanging.connect(
            self.changeOverlayColor
        )
        lutItem.overlayColorButton.sigColorChanged.connect(
            self.saveOverlayColor
        )

        lutItem.invertBwAction.toggled.connect(self.setCheckedInvertBW)

        lutItem.contoursColorButton.disconnect()
        lutItem.contoursColorButton.clicked.connect(
            self.imgGrad.contoursColorButton.click
        )
        for act in lutItem.contLineWightActionGroup.actions():
            act.toggled.connect(self.contLineWeightToggled)

        lutItem.mothBudLineColorButton.disconnect()
        lutItem.mothBudLineColorButton.clicked.connect(
            self.imgGrad.mothBudLineColorButton.click
        )
        for act in lutItem.mothBudLineWightActionGroup.actions():
            act.toggled.connect(self.mothBudLineWeightToggled)

        lutItem.textColorButton.disconnect()
        lutItem.textColorButton.clicked.connect(
            self.editTextIDsColorAction.trigger
        )

        lutItem.defaultSettingsAction.triggered.connect(
            self.restoreDefaultSettings
        )
        lutItem.labelsAlphaSlider.valueChanged.connect(
            self.setValueLabelsAlphaSlider
        )
        lutItem.sigRescaleIntes.connect(
            partial(self.rescaleIntensitiesLut, imageItem=imageItem)
        )
        if f'how_rescale_intensities_{channelName}' in self.df_settings.index:
            how = self.df_settings.at[
                f'how_rescale_intensities_{channelName}', 'value'
            ]
            lutItem.setRescaleIntensitiesHow(how)

        self.rescaleIntensChannelHowMapper[channelName] = (
            'Rescale each 2D image'
        )

        self.addActionsLutItemContextMenu(lutItem)

        alphaScrollBar = self.addAlphaScrollbar(channelName, imageItem)

        toolbutton = widgets.OverlayChannelToolButton(
            channelName, lutItem, shortcut=str(index)
        )
        toolbutton.action = self.overlayToolbar.addWidget(toolbutton)
        toolbutton.setVisible(False)

        toolbutton.clicked.connect(self.overlayChannelToolbuttonClicked)

        alphaScrollBar.toolbutton = toolbutton

        return imageItem, lutItem, alphaScrollBar, toolbutton

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

    def clearAx1Items(self, onlyHideText=False):
        self.ax1_binnedIDs_ScatterPlot.clear()
        self.ax1_ripIDs_ScatterPlot.clear()
        self.labelsLayerImg1.clear()
        self.labelsLayerRightImg.clear()
        self.keepIDsTempLayerLeft.clear()
        self.keepIDsTempLayerRight.clear()
        self.highLightIDLayerImg1.clear()
        self.highLightIDLayerRightImage.clear()
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

        self.clearPointsLayers()

        self.clearOverlayLabelsItems()
        self.clearManualBackgroundAnnotations()
        self.custom_annotations_view.clearCustomAnnot()

    def clearOverlayLabelsItems(self):
        for segmEndname, drawMode in self.drawModeOverlayLabelsChannels.items():
            items = self.overlayLabelsItems[segmEndname]
            imageItem, contoursItem, gradItem = items
            imageItem.clear()
            contoursItem.clear()

    def clearAllItems(self):
        self.clearAx1Items()
        self.clearAx2Items()

    def createUserChannelNameAction(self):
        self.userChNameAction = QAction(self.host)
        self.userChNameAction.setCheckable(True)
        self.userChNameAction.setText(self.user_ch_name)

    def createChannelNamesActions(self):
        # LUT histogram channel name context menu actions
        self.chNamesQActionGroup = QActionGroup(self.host)
        self.chNamesQActionGroup.addAction(self.userChNameAction)
        posData = self.data[self.pos_i]
        for action in self.fluoDataChNameActions:
            self.chNamesQActionGroup.addAction(action)
            action.setChecked(False)

        self.userChNameAction.setChecked(True)

        for action in self.overlayContextMenu.actions():
            action.setChecked(False)

    def addFluoChNameContextMenuAction(self, ch_name):
        posData = self.data[self.pos_i]
        allTexts = [
            action.text() for action in self.chNamesQActionGroup.actions()
        ]
        if ch_name not in allTexts:
            action = QAction(self.host)
            action.setText(ch_name)
            action.setCheckable(True)
            self.chNamesQActionGroup.addAction(action)
            action.setChecked(True)
            self.fluoDataChNameActions.append(action)

    def restoreDefaultColors(self):
        try:
            color = self.defaultToolBarButtonColor
            self.overlayButton.setStyleSheet(f'background-color: {color}')
        except AttributeError:
            # traceback.print_exc()
            pass

    def segmNdimIndicatorClicked(self):
        ndimText = self.segmNdimIndicator.text()
        if ndimText == '2D':
            alternativeNdimText = '3D'
            toggleText = 'activate'
        else:
            alternativeNdimText = '2D'
            toggleText = 'de-activate'
        msg = widgets.myMessageBox(wrapText=False)
        important_txt = ("""
            The toggle to activate 3D segmentation is visible only when
            the <code>Number of z-slices</code> is greater than 1.
        """)
        txt = html_utils.paragraph(f"""
            This indicator shows that you are working with {ndimText}
            segmentation masks.<br><br>

            If instead, you want to work with {alternativeNdimText} segmentation,
            you need to initialize a new segmentation file.<br><br>

            To do so, go the menu on the top menubar <code>File -->
            New Segmentation File...</code> and,<br>
            at the dialog where you insert the metadata (Number of z-slices,
            pixel size, etc.),<br>
            <b>{toggleText}</b> the parameter called <code>Work with 3D
            segmentation masks (z-stack)</code><br>
            as indicated in the screenshot below<br>.
            {html_utils.to_admonition(important_txt, admonition_type='note')}
            <br>
        """)
        msg.information(
            self.host, 'Segmentation nmber of dimensions info', txt,
            image_paths=':toggle_3D_screenshot.png'
        )
        self.segmNdimIndicator.setChecked(True)

    def addAlphaScrollbar(self, channelName, imageItem):
        alphaScrollBar = widgets.ScrollBar(Qt.Horizontal)
        imageItem.alphaScrollBar = alphaScrollBar
        alphaScrollBar.channelName = channelName

        label = QLabel(f'Alpha {channelName}')
        label.setFont(_font)
        label.hide()
        alphaScrollBar.imageItem = imageItem
        alphaScrollBar.label = label
        alphaScrollBar.setFixedHeight(self.h)
        alphaScrollBar.hide()
        alphaScrollBar.setMinimum(0)
        alphaScrollBar.setMaximum(40)
        alphaScrollBar.setValue(20)
        alphaScrollBar.setToolTip(
            f'Control the alpha value of the overlaid channel {channelName}.\n'
            'alpha=0 results in NO overlay,\n'
            'alpha=1 results in only fluorescence data visible'
        )
        self.bottomLeftLayout.addWidget(
            alphaScrollBar.label, self.alphaScrollbarRow, 0,
            alignment=Qt.AlignRight
        )
        self.bottomLeftLayout.addWidget(
            alphaScrollBar, self.alphaScrollbarRow, 1, 1, 2
        )

        alphaScrollBar.valueChanged.connect(
            partial(self.setOpacityOverlayLayersItems, scrollbar=alphaScrollBar)
        )

        self.alphaScrollbarRow += 1
        return alphaScrollBar

    def createOverlayContextMenu(self):
        ch_names = [ch for ch in self.ch_names if ch != self.user_ch_name]
        self.overlayContextMenu = QMenu()
        self.overlayContextMenu.addSeparator()
        self.checkedOverlayChannels = set()
        for chName in ch_names:
            action = QAction(chName, self.overlayContextMenu)
            action.setCheckable(True)
            action.toggled.connect(self.overlayChannelToggled)
            self.overlayContextMenu.addAction(action)

    def createOverlayLabelsContextMenu(self, segmEndnames):
        self.overlayLabelsContextMenu = QMenu()
        self.overlayLabelsContextMenu.addSeparator()
        self.drawModeOverlayLabelsChannels = {}
        segmEndnames_extended = list(segmEndnames.copy())
        segmEndnames_extended = ['combined segm.'] + segmEndnames_extended
        for segmEndname in segmEndnames_extended:
            action = QAction(segmEndname, self.overlayLabelsContextMenu)
            if segmEndname == 'combined segm.':
                action.setCheckable(False)
                self.combineSegmViewToggle = action
            else:
                action.setCheckable(True)
            action.toggled.connect(self.addOverlayLabelsToggled)
            self.overlayLabelsContextMenu.addAction(action)

        self.overlayLabelsContextMenu.addSeparator()
        action = QAction('Edit appearance...', self.overlayLabelsContextMenu)
        action.triggered.connect(self.editOverlayLabelsAppearance)
        self.overlayLabelsContextMenu.addAction(action)

    def editOverlayLabelsAppearance(self, *args):
        segmEndname = list(self.overlayLabelsItems.keys())[0]
        contoursItem = self.overlayLabelsItems[segmEndname][1]
        win = apps.OverlayLabelsAppearanceDialog(
            scatterPlotItem=contoursItem, parent=self.host
        )
        win.exec_()
        if win.cancel:
            return

        brush = win.properties['brush']
        pen = win.properties['pen']
        for items in self.overlayLabelsItems.values():
            imageItem, contoursItem, gradItem = items
            contoursItem.setBrush(brush, update=False)
            contoursItem.setPen(pen)

    def createOverlayLabelsItems(self, segmEndnames):
        selectActionGroup = QActionGroup(self.host)
        segmEndnames_extended = list(segmEndnames.copy())
        segmEndnames_extended = ['combined segm.'] + segmEndnames_extended
        for segmEndname in segmEndnames_extended:
            action = QAction(segmEndname)
            if segmEndname == 'combined segm.':
                action.setCheckable(False)
            else:
                action.setCheckable(True)
            action.toggled.connect(self.setOverlayLabelsItemsVisible)
            selectActionGroup.addAction(action)
        self.selectOverlayLabelsActionGroup = selectActionGroup

        self.overlayLabelsItems = {}
        for segmEndname in segmEndnames_extended:
            imageItem = pg.ImageItem()

            gradItem = widgets.overlayLabelsGradientWidget(
                imageItem, selectActionGroup, segmEndname
            )
            gradItem.hide()
            gradItem.drawModeActionGroup.triggered.connect(
                self.overlayLabelsDrawModeToggled
            )
            self.mainLayout.addWidget(gradItem, 0, 0)

            contoursItem = pg.ScatterPlotItem()
            color = colors.get_complementary_color(self.contLineColor)
            r, g, b, a = colors.rgba_str_to_values(color)
            qcolor = QColor(r, g, b, a)
            contoursItem.setData(
                [], [], symbol='s', pxMode=False, size=self.contLineWeight*2,
                brush=pg.mkBrush(color=qcolor),
                pen=pg.mkPen(width=3, color=qcolor), tip=None
            )

            items = (imageItem, contoursItem, gradItem)
            self.overlayLabelsItems[segmEndname] = items

    def addOverlayLabelsToggled(self, checked, name=None):
        if name is None:
            name = self.sender().text()
        if checked:
            gradItem = self.overlayLabelsItems[name][-1]
            drawMode = gradItem.drawModeActionGroup.checkedAction().text()
            self.drawModeOverlayLabelsChannels[name] = drawMode
        else:
            self.drawModeOverlayLabelsChannels.pop(name)
            self.hideOverlayLabelsItems(specific=[name])
        self.setOverlayLabelsItems()

    def overlayLabelsDrawModeToggled(self, action):
        segmEndname = action.segmEndname
        drawMode = action.text()
        if segmEndname in self.drawModeOverlayLabelsChannels:
            self.drawModeOverlayLabelsChannels[segmEndname] = drawMode
            self.setOverlayLabelsItems()

    def overlayChannelToggled(self, checked):
        # Action toggled from overlayButton context menu
        channelName = self.sender().text()
        posData = self.data[self.pos_i]
        if checked:
            if channelName not in posData.loadedFluoChannels:
                self.loadOverlayData([channelName], addToExisting=True)
            else:
                _, filename = self.getPathFromChName(channelName, posData)
                posData.ol_data[filename] = (
                    posData.ol_data_dict[filename].copy()
                )

            self.checkedOverlayChannels.add(channelName)
        else:
            self.checkedOverlayChannels.remove(channelName)
            imageItem = self.overlayLayersItems[channelName][0]
            imageItem.clear()

        self.setOverlayChannelsToolbuttonsChecked()
        self.setOverlayItemsVisible()
        self.setRetainSizePolicyLutItems()
        self.updateAllImages()

    def overlayLabels_cb(self, checked, selectedLabelsEndnames=None):
        if checked:
            if not self.drawModeOverlayLabelsChannels:
                if selectedLabelsEndnames is None:
                    selectedLabelsEndnames = self.askLabelsToOverlay()
                if selectedLabelsEndnames is None:
                    self.logger.info('Overlay labels cancelled.')
                    self.overlayLabelsButton.setChecked(False)
                    return
                for selectedEndname in selectedLabelsEndnames:
                    self.loadOverlayLabelsData(selectedEndname)
                    for action in self.overlayLabelsContextMenu.actions():
                        if not action.isCheckable():
                            continue
                        if action.text() == selectedEndname:
                            action.setChecked(True)
                lastSelectedName = selectedLabelsEndnames[-1]
                for action in self.selectOverlayLabelsActionGroup.actions():
                    if action.text() == lastSelectedName:
                        action.setChecked(True)
        self.updateAllImages()

    def askLabelsToOverlay(self):
        selectOverlayLabels = widgets.QDialogListbox(
            'Select segmentation to overlay',
            'Select segmentation file to overlay:\n',
            natsorted(self.existingSegmEndNames),
            multiSelection=True,
            parent=self.host
        )
        selectOverlayLabels.exec_()
        if selectOverlayLabels.cancel:
            return

        return selectOverlayLabels.selectedItemsText

    def showOverlayContextMenu(self, event):
        if not self.overlayButton.isChecked():
            return

        self.overlayContextMenu.exec_(QCursor.pos())

    def showOverlayLabelsContextMenu(self, event):
        if not self.overlayLabelsButton.isChecked():
            return

        self.overlayLabelsContextMenu.exec_(QCursor.pos())

    def setCheckedOverlayContextMenusActions(self, channelNames):
        for action in self.overlayContextMenu.actions():
            if action.text() in channelNames:
                action.setChecked(True)
                self.checkedOverlayChannels.add(action.text())

    def enableOverlayWidgets(self, enabled):
        posData = self.data[self.pos_i]
        if enabled:
            self.overlayColorButton.setDisabled(False)
            self.editOverlayColorAction.setDisabled(False)

            if posData.SizeZ == 1:
                return

            self.zSliceOverlay_SB.setMaximum(posData.SizeZ-1)
            if self.zProjOverlay_CB.currentText().find('max') != -1:
                self.overlay_z_label.setDisabled(True)
                self.zSliceOverlay_SB.setDisabled(True)
            else:
                z = self.zSliceOverlay_SB.sliderPosition()
                self.overlay_z_label.setText(
                    f'Overlay z-slice  {z+1:02}/{posData.SizeZ}'
                )
                self.zSliceOverlay_SB.setDisabled(False)
                self.overlay_z_label.setDisabled(False)
            self.zSliceOverlay_SB.show()
            self.overlay_z_label.show()
            self.zProjOverlay_CB.show()
            self.zSliceOverlay_SB.valueChanged.connect(self.updateOverlayZslice)
            self.zProjOverlay_CB.currentTextChanged.connect(
                self.updateOverlayZproj
            )
            self.zProjOverlay_CB.activated.connect(
                self.mode_controls_view.clearComboBoxFocus
            )
        else:
            self.zSliceOverlay_SB.setDisabled(True)
            self.zSliceOverlay_SB.hide()
            self.overlay_z_label.hide()
            self.zProjOverlay_CB.hide()
            self.overlayColorButton.setDisabled(True)
            self.editOverlayColorAction.setDisabled(True)

            if posData.SizeZ == 1:
                return

            self.zSliceOverlay_SB.valueChanged.disconnect()
            self.zProjOverlay_CB.currentTextChanged.disconnect()
            self.zProjOverlay_CB.activated.disconnect()

    def hideOverlayLabelsItems(self, specific=None):
        if specific is None:
            specific = self.overlayLabelsItems.keys()
        for segmEndname in specific:
            imageItem, contoursItem, gradItem = self.overlayLabelsItems[
                segmEndname
            ]
            imageItem.setVisible(False)
            contoursItem.setVisible(False)
            gradItem.setVisible(False)

    def showOverlayLabelsItems(self, specific=None):
        if specific is None:
            specific = self.overlayLabelsItems.keys()
        for segmEndname in specific:
            imageItem, contoursItem, gradItem = self.overlayLabelsItems[
                segmEndname
            ]
            drawMode = self.drawModeOverlayLabelsChannels[segmEndname]
            if drawMode == 'Draw contours':
                contoursItem.setVisible(True)
            elif drawMode == 'Overlay labels':
                imageItem.setVisible(True)
                gradItem.setVisible(True)

    def setOverlayLabelsItems(self, specific=None):
        if not self.overlayLabelsButton.isChecked():
            self.hideOverlayLabelsItems(specific=specific)
            return

        if specific is None:
            specific = self.drawModeOverlayLabelsChannels.keys()

        for segmEndname in specific:
            drawMode = self.drawModeOverlayLabelsChannels[segmEndname]
            ol_lab = self.getOverlayLabelsData(segmEndname)
            items = self.overlayLabelsItems[segmEndname]
            imageItem, contoursItem, gradItem = items
            contoursItem.clear()
            if drawMode == 'Draw contours':
                for obj in skimage.measure.regionprops(ol_lab):
                    contours = self.getObjContours(
                        obj, all_external=True
                    )
                    for cont in contours:
                        contoursItem.addPoints(cont[:,0]+0.5, cont[:,1]+0.5)
            elif drawMode == 'Overlay labels':
                imageItem.setImage(ol_lab, autoLevels=False)
        self.showOverlayLabelsItems(specific=specific)

    def getOverlayLabelsData(self, segmEndname):
        posData = self.data[self.pos_i]

        if posData.ol_labels_data is None:
            self.loadOverlayLabelsData(segmEndname)
        elif segmEndname not in posData.ol_labels_data:
            self.loadOverlayLabelsData(segmEndname)

        comb_seg = False
        if 'combined segm.' == segmEndname:
            comb_seg = True
            if not self.isSegm3D:
                zStackImg = self.data[0].SizeZ > 1
                if zStackImg:
                    selected_z_stack = self.zSliceScrollBar.sliderPosition()
                else:
                    selected_z_stack = 0
                out = posData.ol_labels_data['combined segm.'][
                    posData.frame_i
                ][selected_z_stack]
                return out.astype(np.uint32)

        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if isZslice:
                z = self.zSliceScrollBar.sliderPosition()
                ol_lab = posData.ol_labels_data[segmEndname][posData.frame_i][z]
                if comb_seg:
                    ol_lab = ol_lab.astype(np.uint32)
                return ol_lab
            else:
                ol_lab = posData.ol_labels_data[segmEndname][
                    posData.frame_i
                ].max(axis=0)
                if comb_seg:
                    ol_lab = ol_lab.astype(np.uint32)
                return ol_lab
        else:
            return posData.ol_labels_data[segmEndname][posData.frame_i]

    def loadOverlayLabelsData(self, segmEndname, pos_i=None):
        if pos_i is None:
            pos_i = self.pos_i
        posData = self.data[pos_i]

        if posData.ol_labels_data is None:
            posData.ol_labels_data = {}
        if segmEndname == 'combined segm.':
             posData.ol_labels_data['combined segm.'] = posData.combine_img_data
             return
        filePath, filename = self.view_model.workspace.path_from_endname(
            segmEndname, posData.images_path
        )
        self.logger.info(f'Loading "{segmEndname}.npz"...')
        labelsData = np.load(filePath)['arr_0']
        if posData.SizeT == 1:
            labelsData = labelsData[np.newaxis]
        if self.isSegm3D and labelsData.ndim == 3:
            # 2D segm --> stack to 3D
            T, Y, X = labelsData.shape
            repeat = [labelsData]*posData.SizeZ
            labelsData = np.stack(repeat, axis=1)

        posData.ol_labels_data[segmEndname] = labelsData

    def removeOverlayItems(self):
        self.lutItemsLayout.clear()

        try:
            for toolbutton in self.allOverlayToolbuttonsByIdx.values():
                self.overlayToolbar.removeAction(toolbutton.action)

            self.overlayToolbuttonsSep.removeFromToolbar()
        except Exception as err:
            pass

    def clearOverlayImageItems(self):
        for items in self.overlayLayersItems.values():
            imageItem = items[0]
            imageItem.clear()

        self.rgbaImg1.clear()

    def setOverlayColors(self):
        self.overlayRGBs = [
            (255, 255, 0),
            (252, 72, 254),
            (49, 222, 134),
            (22, 108, 27)
        ]
        self.overlayCmap = matplotlib.colormaps['hsv']
        self.overlayRGBs.extend(
            [tuple([round(c*255) for c in self.overlayCmap(i)][:3])
            for i in np.linspace(0,1,8)]
        )

    def loadOverlayData(self, ol_channels, addToExisting=False):
        posData = self.data[self.pos_i]
        for ol_ch in ol_channels:
            if ol_ch not in list(posData.loadedFluoChannels):
                # Requested channel was never loaded --> load it at first
                # iter i == 0
                success = self.loadFluo_cb(fluo_channels=[ol_ch])
                if not success:
                    return False

        lastChannelName = ol_channels[-1]
        for action in self.fluoDataChNameActions:
            if action.text() == lastChannelName:
                action.setChecked(True)

        for p, posData in enumerate(self.data):
            if addToExisting:
                ol_data = posData.ol_data
            else:
                ol_data = {}
            for i, ol_ch in enumerate(ol_channels):
                _, filename = self.getPathFromChName(ol_ch, posData)
                ol_data[filename] = (
                    posData.ol_data_dict[filename].copy()
                )
                self.addFluoChNameContextMenuAction(ol_ch)
            posData.ol_data = ol_data

        return True

    def askSelectOverlayChannel(self):
        ch_names = [ch for ch in self.ch_names if ch != self.user_ch_name]
        selectFluo = widgets.QDialogListbox(
            'Select channel',
            'Select channel names to overlay:\n',
            ch_names, multiSelection=True, parent=self.host
        )
        selectFluo.exec_()
        if selectFluo.cancel:
            return

        return selectFluo.selectedItemsText

    def setOverlaySingleChannel(self, *args, **kwargs):
        if self.overlayToolbar.isSingleChannel():
            self.overlayToolbarAreChannelsChecked = {
                channel:toolbutton.isChecked()
                for channel, toolbutton in self.allOverlayToolbuttons.items()
            }
            firstActiveToolbutton = [
                toolbutton for toolbutton in self.allOverlayToolbuttons.values()
                if toolbutton.isChecked()
            ][0]
            firstActiveToolbutton.click()
        else:
            for ch, checked in self.overlayToolbarAreChannelsChecked.items():
                toolbutton = self.allOverlayToolbuttons[ch]
                toolbutton.setChecked(checked)

            self.setOverlayItemsOpacities()

    def updateTransparentOverlayRgba(self, *args, **kwargs):
        self.setOverlayImages()

    def setOverlayTransparency(self, transparent: bool):
        opacity = float(transparent)
        opacity = opacity if opacity < 1.0 else 0.999
        self.rgbaImg1.setOpacity(opacity)

        if transparent:
            self.img1.setOpacity(0.001, applyToLinked=False)
            self.imgGrad.sigLookupTableChanged.connect(
                self.updateTransparentOverlayRgba
            )
            self.imgGrad.sigLevelsChanged.connect(
                self.updateTransparentOverlayRgba
            )

        for channel, items in self.overlayLayersItems.items():
            imageItem, lutItem, alphaSB = items[:3]
            if transparent:
                alphaSB.valueChanged.disconnect()
                alphaSB.valueChanged.connect(
                    self.updateTransparentOverlayRgba
                )
                lutItem.sigLookupTableChanged.connect(
                    self.updateTransparentOverlayRgba
                )
                lutItem.sigLevelsChanged.connect(
                    self.updateTransparentOverlayRgba
                )
                imageItem.setOpacity(0)

        if not transparent:
            self.setOverlayItemsOpacities()

        self.setOverlayImages()

    def overlay_cb(self, checked):
        self.overlayToolbar.setVisible(checked)

        self.UserNormAction, _, _ = self.getCheckNormAction()
        posData = self.data[self.pos_i]
        if checked:
            if posData.ol_data is None:
                selectedChannels = self.askSelectOverlayChannel()
                if selectedChannels is None:
                    self.overlayButton.toggled.disconnect()
                    self.overlayButton.setChecked(False)
                    self.overlayButton.toggled.connect(self.overlay_cb)
                    return

                success = self.loadOverlayData(selectedChannels)
                if not success:
                    return False
                lastChannel = selectedChannels[-1]
                self.setCheckedOverlayContextMenusActions(selectedChannels)
                imageItem = self.overlayLayersItems[lastChannel][0]
                self.setOpacityOverlayLayersItems(None, imageItem=imageItem)
                self.setOverlayChannelsToolbuttonsChecked()

            self.setRetainSizePolicyLutItems()
            self.normalizeRescale0to1Action.setChecked(True)

            self.updateAllImages()
            self.updateImageValueFormatter()
            self.enableOverlayWidgets(True)
        else:
            self.img1.setOpacity(1.0)
            self.updateAllImages()
            self.updateImageValueFormatter()
            self.enableOverlayWidgets(False)
            self.clearOverlayImageItems()

        self.setOverlayItemsVisible()

    def getOlImg(self, key, frame_i=None):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i

        img = posData.ol_data[key][frame_i]
        if posData.SizeZ > 1:
            zProjHow = self.zProjOverlay_CB.currentText()
            z = self.zSliceOverlay_SB.sliderPosition()
            if zProjHow == 'same as above':
                zProjHow = self.zProjComboBox.currentText()
                z = self.zSliceScrollBar.sliderPosition()
                reconnect = False
                try:
                    self.zSliceOverlay_SB.valueChanged.disconnect()
                    reconnect = True
                except TypeError:
                    pass
                self.zSliceOverlay_SB.setSliderPosition(z)
                if reconnect:
                    self.zSliceOverlay_SB.valueChanged.connect(
                        self.updateOverlayZslice
                    )
            if zProjHow == 'single z-slice':
                self.overlay_z_label.setText(
                    f'Overlay z-slice  {z+1:02}/{posData.SizeZ}'
                )
                ol_img = img[z].copy()
            elif zProjHow == 'max z-projection':
                ol_img = img.max(axis=0)
            elif zProjHow == 'mean z-projection':
                ol_img = img.mean(axis=0)
            elif zProjHow == 'median z-proj.':
                ol_img = np.median(img, axis=0)
        else:
            ol_img = img.copy()

        return ol_img

    def setOverlayImages(self, frame_i=None):
        if not self.overlayButton.isChecked():
            return

        posData = self.data[self.pos_i]
        if posData.ol_data is None:
            return

        rgba_imgs_info = {}
        for filename in posData.ol_data:
            chName = self.view_model.formatting.channel_name_from_basename(
                filename, posData.basename, remove_ext=False
            )
            if chName not in self.checkedOverlayChannels:
                continue

            items = self.overlayLayersItems[chName]
            imageItem, lutItem, alphaSB = items[:3]

            ol_img = self.getOlImg(filename, frame_i=frame_i)

            if self.overlayToolbar.isTransparent():
                toolbutton = items[3]
                if not toolbutton.isChecked():
                    continue
                alpha_val = alphaSB.value()/alphaSB.maximum()
                ol_img = skimage.exposure.rescale_intensity(
                    ol_img, out_range=(0.0, 1.0)
                )
                out_range_min, out_range_max = lutItem.getLevels()
                rgba_imgs_info[chName] = (ol_img, alpha_val, lutItem)
            else:
                self.rescaleIntensitiesLut(setImage=False, imageItem=imageItem)
                imageItem.setImage(ol_img)

        if not self.overlayToolbar.isTransparent():
            return

        alpha_values = []
        images = []
        luts = []
        for channel, info in rgba_imgs_info.items():
            ol_img, alpha_val, lutItem = info
            alpha_values.append(alpha_val)
            images.append(ol_img)
            luts.append(lutItem.gradient.getLookupTable(256, alpha=255)/255)

        weights = colors.hierarchical_weights(alpha_values)

        if self.baseLayerToolbutton.isChecked():
            image1 = self._getImageupdateAllImages()
            image1 = skimage.exposure.rescale_intensity(
                image1, out_range=(0.0, 1.0)
            )
            images.append(image1)
            baseLut = (
                self.imgGrad.gradient.getLookupTable(256, alpha=255)/255
            )
            luts.append(baseLut)

        images_rgba = []
        for img, lut in zip(images, luts):
            rgba = colors.grayscale_apply_lut(img, lut)
            images_rgba.append(rgba)

        rgba_merge = colors.hierarchical_blend(images_rgba, weights)
        self.rgbaImg1.setImage(rgba_merge)

    def getOpacitiesFromAlphaScrollbarValues(self):
        active_channel_alpha_values = {}
        for items in self.overlayLayersItems.values():
            imgItem, lutItem, alphaSB = items[:3]
            _toolbutton = alphaSB.toolbutton
            if not _toolbutton.isChecked() or not _toolbutton.isVisible():
                continue

            active_channel_alpha_values[imgItem.channelName] = (
                alphaSB.value()/alphaSB.maximum()
            )

        return self.view_model.overlay_channel_opacity_map(
            self.user_ch_name,
            active_channel_alpha_values,
        )

    def toggleOverlayColorButton(self, checked=True):
        self.mousePressColorButton(None)

    def toggleTextIDsColorButton(self, checked=True):
        self.textIDsColorButton.selectColor()

    def updateTextAnnotColor(self, button):
        r, g, b = np.array(self.textIDsColorButton.color().getRgb()[:3])
        self.imgGrad.textColorButton.setColor((r, g, b))
        for items in self.overlayLayersItems.values():
            lutItem = items[1]
            lutItem.textColorButton.setColor((r, g, b))
        self.gui_createTextAnnotColors(r,g,b, custom=True)
        self.gui_setTextAnnotColors()
        self.updateAllImages()

    def saveTextIDsColors(self, button):
        self.df_settings.at['textIDsColor', 'value'] = self.objLabelAnnotRgb
        self.df_settings.to_csv(self.settings_csv_path)

    def ticksCmapMoved(self, gradient):
        pass
        # posData = self.data[self.pos_i]
        # self.setLut(posData, shuffle=False)
        # self.updateLookuptable()

    def updateLabelsCmap(self, gradient):
        self.setLut()
        self.updateLookuptable()
        self.initLabelsImageItems()

        self.df_settings = self.labelsGrad.saveState(self.df_settings)
        self.df_settings.to_csv(self.settings_csv_path)

        self.updateAllImages()

    def extendLabelsLUT(self, lenNewLut):
        if lenNewLut > len(self.lut):
            self.lut = self.view_model.extend_labels_lut(self.lut, lenNewLut)
            self.initLabelsImageItems()
            return True
        return False

    def initLookupTableLab(self):
        self.img2.setLookupTable(self.lut)
        self.img2.setLevels([0, len(self.lut)])
        self.initLabelsImageItems()

    def getLabelsImageLut(self):
        return self.view_model.generate_labels_image_lut(self.lut)

    def initLabelsImageItems(self):
        lut = self.getLabelsImageLut()
        self.labelsLayerImg1.setLevels([0, len(lut)])
        self.labelsLayerRightImg.setLevels([0, len(lut)])
        self.labelsLayerImg1.setLookupTable(lut)
        self.labelsLayerRightImg.setLookupTable(lut)
        alpha = self.imgGrad.labelsAlphaSlider.value()
        self.labelsLayerImg1.setOpacity(alpha)
        self.labelsLayerRightImg.setOpacity(alpha)

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

        lut = self.view_model.apply_lut_dimming_for_kept_objects(
            lut,
            getattr(self, 'keptObjectsIDs', []),
            self.keepIDsButton.isChecked(),
        )

        self.img2.setLookupTable(lut)

    def setLut(self, shuffle=True):
        self.lut = self.labelsGrad.item.colorMap().getLookupTable(0,1,255)
        if shuffle:
            np.random.shuffle(self.lut)

        # Insert background color
        if 'labels_bkgrColor' in self.df_settings.index:
            rgbString = self.df_settings.at['labels_bkgrColor', 'value']
            try:
                r, g, b = rgbString
            except Exception as e:
                r, g, b = colors.rgb_str_to_values(rgbString)
        else:
            r, g, b = 25, 25, 25
            self.df_settings.at['labels_bkgrColor', 'value'] = (r, g, b)

        self.lut = np.insert(self.lut, 0, [r, g, b], axis=0)

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

    def changeOverlayColor(self, button):
        rgb = button.color().getRgb()[:3]
        lutItem = self.overlayLayersItems[button.channel][1]
        self.initColormapOverlayLayerItem(rgb, lutItem)
        lutItem.overlayColorButton.setColor(rgb)

    def saveOverlayColor(self, button):
        rgb = button.color().getRgb()[:3]
        rgb_text = '_'.join([str(val) for val in rgb])
        self.df_settings.at[f'{button.channel}_rgb', 'value'] = rgb_text
        self.df_settings.to_csv(self.settings_csv_path)

    def setValueLabelsAlphaSlider(self, value):
        self.imgGrad.labelsAlphaSlider.setValue(value)
        self.updateLabelsAlpha(value)

    def setOverlaySegmMasks(self, force=False, forceIfNotActive=False):
        if not hasattr(self, 'currentLab2D'):
            return

        how = self.drawIDsContComboBox.currentText()
        isOverlaySegmLeftActive = how.find('overlay segm. masks') != -1

        how_ax2 = self.getAnnotateHowRightImage()
        isOverlaySegmRightActive = (
            how_ax2.find('overlay segm. masks') != -1
            and self.labelsGrad.showRightImgAction.isChecked()
        )

        isOverlaySegmActive = (
            isOverlaySegmLeftActive or isOverlaySegmRightActive
            or force
        )
        if not isOverlaySegmActive and not forceIfNotActive:
            return

        alpha = self.imgGrad.labelsAlphaSlider.value()
        if alpha == 0:
            return

        posData = self.data[self.pos_i]
        maxID = max(posData.IDs, default=0)

        if maxID >= len(self.lut):
            self.extendLabelsLUT(maxID+10)

        currentLab2D = self.currentLab2D
        if isOverlaySegmLeftActive:
            self.labelsLayerImg1.setImage(currentLab2D, autoLevels=False)

        if isOverlaySegmRightActive:
            self.labelsLayerRightImg.setImage(currentLab2D, autoLevels=False)

    def setOverlayLabelsItemsVisible(self, checked):
        for _segmEndname, drawMode in self.drawModeOverlayLabelsChannels.items():
            items = self.overlayLabelsItems[_segmEndname]
            gradItem = items[-1]
            gradItem.hide()

        if checked:
            segmEndname = self.sender().text()
            gradItem = self.overlayLabelsItems[segmEndname][-1]
            gradItem.show()

    def setRetainSizePolicyLutItems(self):
        if not self.retainSizeLutItems:
            return
        for channel, items in self.overlayLayersItems.items():
            _, lutItem, alphaSB = items[:3]
            myutils.setRetainSizePolicy(lutItem, retain=True)
        QTimer.singleShot(300, self.autoRange)

    def setOverlayChannelsToolbuttonsChecked(self):
        for channel, items in self.overlayLayersItems.items():
            _, lutItem, alphaSB, toolbutton = items[:4]
            toolbutton.setChecked(
                self.view_model.overlay_toolbutton_checked(
                    channel,
                    checked_channels=self.checkedOverlayChannels,
                    is_single_channel=self.overlayToolbar.isSingleChannel(),
                )
            )

    def setOverlayItemsVisible(self):
        visibility_plan = self.view_model.overlay_visibility_plan(
            all_channels=self.overlayLayersItems.keys(),
            checked_channels=self.checkedOverlayChannels,
            overlay_enabled=self.overlayButton.isChecked(),
        )
        for channel, items in self.overlayLayersItems.items():
            _, lutItem, alphaSB, toolbutton = items[:4]
            if visibility_plan.channel_visible[channel]:
                lutItem.show()
                alphaSB.show()
                alphaSB.label.show()
                toolbutton.setVisible(True)
            else:
                lutItem.hide()
                alphaSB.hide()
                alphaSB.label.hide()
                toolbutton.setVisible(False)

    def overlayChannelToolbuttonClicked(self, checked=False, toolbutton=None):
        if toolbutton is None:
            toolbutton = self.sender()

        channelName = toolbutton.channelName()
        checked_channels = {
            channel
            for channel, button in self.allOverlayToolbuttons.items()
            if button.isChecked()
        }
        planned_checked_channels = (
            self.view_model.overlay_toolbutton_click_checked_channels(
                clicked_channel=channelName,
                all_channels=self.allOverlayToolbuttons.keys(),
                checked_channels=checked_channels,
                toolbar_single_channel=self.overlayToolbar.isSingleChannel(),
            )
        )
        for channel, otherToolbutton in self.allOverlayToolbuttons.items():
            otherToolbutton.setChecked(channel in planned_checked_channels)

        if self.overlayToolbar.isTransparent():
            self.setOverlayImages()
            return

        self.setOverlayItemsOpacities()

    def setOverlayItemsOpacities(self):
        checked_channels = {
            channel
            for channel, button in self.allOverlayToolbuttons.items()
            if button.isChecked()
        }
        active_channel_alpha_values = {}
        for items in self.overlayLayersItems.values():
            imageItem, lutItem, alphaSB = items[:3]
            toolbutton = alphaSB.toolbutton
            if not toolbutton.isChecked() or not toolbutton.isVisible():
                continue
            active_channel_alpha_values[imageItem.channelName] = (
                alphaSB.value()/alphaSB.maximum()
            )
        opacity_plan = self.view_model.overlay_item_opacity_plan(
            all_channels=self.allOverlayToolbuttons.keys(),
            base_channel=self.user_ch_name,
            checked_channels=checked_channels,
            toolbar_single_channel=self.overlayToolbar.isSingleChannel(),
            active_channel_alpha_values=active_channel_alpha_values,
        )

        # Set opacity of every layer accordingly
        for channel, otherToolbutton in self.allOverlayToolbuttons.items():
            if channel == self.user_ch_name:
                otherImageItem = self.img1
                alphaScrollbar = None
                # alpha_value = channel_opacity_mapper[channel]
            else:
                otherItems = self.overlayLayersItems[channel]
                otherImageItem = otherItems[0]
                alphaScrollbar = otherItems[2]
                # alpha_value = alphaScrollbar.value()/alphaScrollbar.maximum()

            op_val = opacity_plan.opacities[channel]
            otherImageItem.setOpacity(op_val, applyToLinked=False)

            if alphaScrollbar is None:
                continue

            alphaScrollbar.setDisabled(
                opacity_plan.alpha_scrollbar_disabled[channel]
            )

    def initColormapOverlayLayerItem(self, foregrColor, lutItem):
        if self.invertBwAction.isChecked():
            bkgrColor = (255,255,255,255)
        else:
            bkgrColor = (0,0,0,255)
        gradient = colors.get_pg_gradient((bkgrColor, foregrColor))
        lutItem.setGradient(gradient)

    def setOpacityOverlayLayersItems(self, value, imageItem=None, scrollbar=None):
        if scrollbar is None:
            scrollbar = imageItem.alphaScrollBar

        channel = scrollbar.channelName
        toolbutton = self.allOverlayToolbuttons[channel]
        if not toolbutton.isChecked() or not toolbutton.isVisible():
            return

        if value is None:
            value = scrollbar.value()

        if imageItem is None:
            imageItem = scrollbar.imageItem
            alpha = value/scrollbar.maximum()
        elif value > 1:
            alpha = value/scrollbar.maximum()
        else:
            alpha = value

        alpha_values = []
        activeOverlayImageItems = []
        for items in self.overlayLayersItems.values():
            imgItem, lutItem, alphaSB = items[:3]
            _toolbutton = alphaSB.toolbutton
            if alphaSB.channelName == channel:
                alpha_values.append(alpha)
            elif not _toolbutton.isChecked() or not _toolbutton.isVisible():
                continue
            else:
                alpha_values.append(alphaSB.value()/alphaSB.maximum())

            activeOverlayImageItems.append(imgItem)

        opacities = colors.hierarchical_weights(alpha_values)[::-1]

        for i, imgItem in enumerate(activeOverlayImageItems):
            imgItem.setOpacity(opacities[i+1])

        self.img1.setOpacity(opacities[0], applyToLinked=False)

    def gui_getLostObjScatterItem(self):
        self.objLostAnnotRgb = (245, 184, 0)
        brush = pg.mkBrush((*self.objLostAnnotRgb, 150))
        pen = pg.mkPen(self.objLostAnnotRgb, width=1)
        lostObjScatterItem = pg.ScatterPlotItem(
            size=self.contLineWeight+1, pen=pen,
            brush=brush, pxMode=False, symbol='s'
        )
        return lostObjScatterItem

    def gui_getTrackedLostObjScatterItem(self):
        self.objLostTrackedAnnotRgb = (0, 255, 0)
        brush = pg.mkBrush((*self.objLostTrackedAnnotRgb, 150))
        pen = pg.mkPen(self.objLostTrackedAnnotRgb, width=1)
        lostObjScatterItem = pg.ScatterPlotItem(
            size=self.contLineWeight+1, pen=pen,
            brush=brush, pxMode=False, symbol='s'
        )
        return lostObjScatterItem

    def _gui_createGraphicsItems(self):
        for _posData in self.data:
            _posData.allData_li = [None]*_posData.SizeT

        posData = self.data[self.pos_i]

        allIDs, posData = self.view_model.label_edits.count_objects(
            posData, self.logger.info
        )

        self.highLowResAction.setChecked(True)
        numItems = len(allIDs)
        if numItems > 1500:
            cancel, switchToLowRes = _warnings.warnTooManyItems(
                self.host, numItems, self.progressWin
            )
            if cancel:
                self.progressWin.workerFinished = True
                self.progressWin.close()
                self.progressWin = None
                self.loadingDataAborted()
                return
            if switchToLowRes:
                self.highLowResAction.setChecked(False)
            else:
                # Many items requires pxMode active to be fast enough
                self.pxModeAction.setChecked(True)

        self.logger.info(f'Creating graphical items...')

        self.ax1_contoursImageItem = pg.ImageItem()

        self.ax1_lostObjImageItem = pg.ImageItem()
        self.ax2_lostObjImageItem = pg.ImageItem()

        self.ax1_lostTrackedObjImageItem = pg.ImageItem()
        self.ax2_lostTrackedObjImageItem = pg.ImageItem()

        self.ax1_oldMothBudLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.oldMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax1_newMothBudLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.newMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax1_lostObjScatterItem = self.gui_getLostObjScatterItem()
        self.yellowContourScatterItem = self.gui_getLostObjScatterItem()

        self.ax1_lostTrackedScatterItem = self.gui_getTrackedLostObjScatterItem()
        self.greenContourScatterItem = self.gui_getTrackedLostObjScatterItem()

        brush = pg.mkBrush((0,255,0,200))
        pen = pg.mkPen('g', width=1)
        self.ccaFailedScatterItem = pg.ScatterPlotItem(
            size=self.contLineWeight+1, pen=pen,
            brush=brush, pxMode=False, symbol='s'
        )

        self.ax2_contoursImageItem = pg.ImageItem()
        self.ax2_oldMothBudLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.oldMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax2_newMothBudLinesItem = pg.ScatterPlotItem(
            symbol='s', pxMode=False, brush=self.newMothBudLineBrush,
            size=self.mothBudLineWeight, pen=None
        )
        self.ax2_lostObjScatterItem = self.gui_getLostObjScatterItem()
        self.ax2_lostTrackedScatterItem = self.gui_getTrackedLostObjScatterItem()

        self.gui_createTextAnnotItems(allIDs) # here
        self.gui_setTextAnnotColors()# here

        self.setDisabledAnnotOptions(False)

        self.progressWin.mainPbar.setMaximum(0)
        self.gui_addOverlayLayerItems()
        self.gui_addTopLayerItems()

        self.gui_addCreatedAxesItems()
        self.gui_add_ax_cursors()
        self.progressWin.workerFinished = True
        self.progressWin.close()
        self.progressWin = None

        self.loadingDataCompleted()

    def gui_createTextAnnotItems(self, allIDs):
        self.textAnnot = {}
        isHighResolution = self.highLowResAction.isChecked()
        pxMode = self.pxModeAction.isChecked()
        for ax in range(2):
            ax_textAnnot = annotate.TextAnnotations()
            ax_textAnnot.initFonts(self.fontSize)
            ax_textAnnot.createItems(
                isHighResolution, allIDs, pxMode=pxMode
            )
            self.textAnnot[ax] = ax_textAnnot

    def gui_addOverlayLayerItems(self):
        for items in self.overlayLabelsItems.values():
            imageItem, contoursItem, gradItem = items
            self.ax1.addItem(imageItem)
            self.ax1.addItem(contoursItem)

    def gui_addTopLayerItems(self):
        for item in self.topLayerItems:
            self.ax1.addItem(item)

        for item in self.topLayerItemsRight:
            self.ax2.addItem(item)

        # self.ax2.addItem(self.currentFrameLabelItem)

    def updateContoursImage(self, ax, delROIsIDs=None, compute=True):
        imageItem = self.getContoursImageItem(ax)
        if imageItem is None:
            return

        if not hasattr(self, 'contoursImage'):
            self.initContoursImage()
        else:
            self.contoursImage[:] = 0

        contours = []
        for obj in skimage.measure.regionprops(self.currentLab2D):
            obj_contours = self.getObjContours(
                obj,
                all_external=True,
                force_calc=compute,
                include_internal=self.showAllContoursToggle.isChecked()
            )
            contours.extend(obj_contours)

        thickness = self.contLineWeight
        color = self.contLineColor
        self.setContoursImage(imageItem, contours, thickness, color)

    def setContoursImage(self, imageItem, contours, thickness, color):
        cv2.drawContours(self.contoursImage, contours, -1, color, thickness)
        imageItem.setImage(self.contoursImage)

    def getObjFromID(self, ID):
        posData = self.data[self.pos_i]
        try:
            idx = posData.IDs_idxs[ID]
        except KeyError as e:
            # Object already cleared
            return

        obj = posData.rp[idx]
        return obj

    def setLostObjectContour(self, obj):
        allContours = self.getObjContours(obj, all_external=True)
        for objContours in allContours:
            xx = objContours[:,0] + 0.5
            yy = objContours[:,1] + 0.5
            data = [obj.label]*len(xx)
            self.ax1_lostObjScatterItem.addPoints(xx, yy, data=data)
            self.ax2_lostObjScatterItem.addPoints(xx, yy)

    def setTrackedLostObjectContour(self, obj):
        if self.isExportingVideo:
            return

        allContours = self.getObjContours(obj, all_external=True)
        for objContours in allContours:
            xx = objContours[:,0] + 0.5
            yy = objContours[:,1] + 0.5
            data = [obj.label]*len(xx)
            self.ax1_lostTrackedScatterItem.addPoints(xx, yy, data=data)
            self.ax2_lostTrackedScatterItem.addPoints(xx, yy)

    def updateLostContoursImage(self, ax, draw=True, delROIsIDs=None):
        if draw:
            imageItem = self.getLostObjImageItem(ax)
            if imageItem is None:
                return

        if not hasattr(self, 'lostObjContoursImage'):
            self.initLostObjContoursImage()
        else:
            self.lostObjContoursImage[:] = 0

        if delROIsIDs is None:
            delROIsIDs = set()

        posData = self.data[self.pos_i]
        prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
        prev_IDs_idxs = posData.allData_li[posData.frame_i-1]['IDs_idxs']
        if posData.whitelist is not None and posData.whitelist.whitelistIDs is not None:
            whitelist = posData.whitelist.whitelistIDs[posData.frame_i-1]
        else:
            whitelist = None

        contours = []
        for lostID in posData.lost_IDs:
            if lostID in delROIsIDs or (whitelist is not None and lostID not in whitelist):
                continue

            obj = prev_rp[prev_IDs_idxs[lostID]]
            if not self.isObjVisible(obj.bbox):
                continue

            obj_contours = self.getObjContours(obj, all_external=True)

            if ax == 0:
                self.addLostObjsToLostObjImage(obj, lostID)

            contours.extend(obj_contours)

        if not draw:
            return

        self.drawLostObjContoursImage(imageItem, contours)

    def drawLostObjContoursImage(
            self, imageItem, contours,
            thickness=1,
            color=(255, 165, 0, 255) # orange
        ):
        img = self.lostObjContoursImage
        cv2.drawContours(img, contours, -1, color, thickness)
        imageItem.setImage(img)

    def updateLostTrackedContoursImage(
            self, ax, delROIsIDs=None, tracked_lost_IDs=None
        ):
        imageItem = self.getLostTrackedObjImageItem(ax)
        if imageItem is None:
            return

        if not hasattr(self, 'lostTrackedObjContoursImage'):
            self.initLostTrackedObjContoursImage()
        else:
            self.lostTrackedObjContoursImage[:] = 0

        if delROIsIDs is None:
            delROIsIDs = set()

        posData = self.data[self.pos_i]
        if tracked_lost_IDs is None:
            tracked_lost_IDs = self.getTrackedLostIDs()

        prev_rp = posData.allData_li[posData.frame_i-1]['regionprops']
        prev_IDs_idxs = posData.allData_li[posData.frame_i-1]['IDs_idxs']
        contours = []
        for tracked_lost_ID in tracked_lost_IDs:
            if tracked_lost_ID in delROIsIDs:
                continue

            obj = prev_rp[prev_IDs_idxs[tracked_lost_ID]]
            if not self.isObjVisible(obj.bbox):
                continue

            obj_contours = self.getObjContours(obj, all_external=True)
            contours.extend(obj_contours)

        self.drawLostTrackedObjContoursImage(imageItem, contours)

    def drawLostTrackedObjContoursImage(self, imageItem, contours):
        thickness = 1
        color = (0, 255, 0, 255) # green
        img = self.lostTrackedObjContoursImage
        cv2.drawContours(img, contours, -1, color, thickness)
        imageItem.setImage(img)

    def getNearestLostObjID(self, y, x):
        if not self.annotLostObjsToggle.isChecked():
            return

        posData = self.data[self.pos_i]
        if not posData.lost_IDs:
            return

        prev_lab = posData.allData_li[posData.frame_i-1]['labels']
        if prev_lab is None:
            return

        # if not hasattr(self, 'lostObjContoursImage'):
        #     self.store_data()
        #     posData.frame_i -= 1
        #     self.get_data()
        #     self.store_data()
        #     posData.frame_i += 1
        #     self.get_data()
        #     self.updateLostNewCurrentIDs()
        #     self.updateLostContoursImage(ax=0)
        #     self.updateLostContoursImage(ax=1)
        #     self.updateLostNewCurrentIDs()

        yy, xx, _ = np.nonzero(self.lostObjContoursImage)
        lostObjsContourMask = np.zeros(self.currentLab2D.shape, dtype=bool)
        lostObjsContourMask[yy.astype(int), xx.astype(int)] = True

        # Add accepted lost IDs
        try:
            yy, xx, _ = np.nonzero(self.lostTrackedObjContoursImage)
            lostObjsContourMask[yy.astype(int), xx.astype(int)] = True
        except Exception as err:
            pass

        _, y_nearest, x_nearest = self.view_model.label_edits.nearest_nonzero_2d(
            lostObjsContourMask, y, x, return_coords=True
        )
        nearest_ID = self.get_2Dlab(prev_lab)[y_nearest, x_nearest]

        if nearest_ID == 0:
            return

        return nearest_ID

    def addObjContourToContoursImage(
            self, ID=0, obj=None, ax=0, thickness=None, color=None,
            force=False
        ):
        imageItem = self.getContoursImageItem(ax, force=force)
        if imageItem is None:
            return

        if obj is None:
            obj = self.getObjFromID(ID)
            if obj is None:
                return

        contours = self.getObjContours(obj, all_external=True)
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

    def setAllContoursImages(self, delROIsIDs=None, compute=True):
        if compute:
            self.computeAllContours()
        self.updateContoursImage(ax=0, delROIsIDs=delROIsIDs, compute=compute)
        self.updateContoursImage(ax=1, delROIsIDs=delROIsIDs, compute=compute)

    def setAllLostObjContoursImage(self, delROIsIDs=None):
        self.updateLostContoursImage(ax=0, delROIsIDs=None)
        self.updateLostContoursImage(ax=1, delROIsIDs=None)

    def setAllLostTrackedObjContoursImage(self, delROIsIDs=None):
        self.updateLostTrackedContoursImage(ax=0, delROIsIDs=None)
        self.updateLostTrackedContoursImage(ax=1, delROIsIDs=None)

    def getObjContours(
            self, obj, all_external=False, local=False, force_calc=True,
            include_internal=False
        ):
        posData = self.data[self.pos_i]
        dataDict = posData.allData_li[posData.frame_i]
        allContours = dataDict.get('contours')
        if allContours is not None and not force_calc:
            z = self.z_lab()
            key = (obj.label, str(z), all_external, local)
            contours = allContours.get(key)
            if contours is not None:
                return contours

        obj_image = self.getObjImage(obj.image, obj.bbox).astype(np.uint8)
        obj_bbox = self.getObjBbox(obj.bbox)
        try:
            contours = self.view_model.geometry.object_contours(
                obj_image=obj_image,
                obj_bbox=obj_bbox,
                local=local,
                all_external=all_external
            )
        except Exception as e:
            if all_external:
                contours = []
            else:
                contours = None
            self.logger.warning(
                f'Object ID {obj.label} contours drawing failed. '
                f'(bounding box = {obj.bbox})'
            )
        return contours

    def clearComputedContours(self):
        for posData in self.data:
            for frame_i, dataDict in enumerate(posData.allData_li):
                dataDict['contours'] = {}

    def _computeAllContours2D(
            self, dataDict, obj, z, obj_bbox, include_internal=False
        ):
        obj_image = self.getObjImage(obj.image, obj.bbox, z_slice=z)
        if obj_image is None:
            return

        all_external = False
        local = False
        contours = self.view_model.geometry.object_contours(
            obj_image=obj_image,
            obj_bbox=obj_bbox,
            local=local,
            all_external=all_external
        )
        key = (obj.label, str(z), all_external, local)
        dataDict['contours'][key] = contours

        all_external = True
        local = False
        contours = self.view_model.geometry.object_contours(
            obj_image=obj_image,
            obj_bbox=obj_bbox,
            local=local,
            all_external=all_external,
            all=include_internal
        )
        key = (obj.label, str(z), all_external, local)
        dataDict['contours'][key] = contours

        return dataDict

    def computeAllContours(self):
        self.logger.info('Computing all contours...')
        posData = self.data[self.pos_i]
        zz = [None]
        if self.isSegm3D:
            zz.extend(range(posData.SizeZ))

        include_internal = self.showAllContoursToggle.isChecked()
        for frame_i, dataDict in enumerate(posData.allData_li):
            lab = dataDict['labels']
            if lab is None:
                break

            rp = dataDict['regionprops']
            if rp is None:
                rp = skimage.measure.regionprops(lab)

            dataDict['contours'] = {}
            for obj in rp:
                obj_bbox = self.getObjBbox(obj.bbox)
                for z in zz:
                    if not self.isObjVisible(obj.bbox, z_slice=z):
                        continue

                    try:
                        self._computeAllContours2D(
                            dataDict, obj, z, obj_bbox,
                            include_internal=include_internal
                        )
                    except Exception as err:
                        # Contours computation fails on weird objects
                        pass

    def computeAllObjToObjCostPairs(self):
        desc = (
            'Computing all object-to-object cost matrices...'
        )
        self.logger.info(desc)
        posData = self.data[self.pos_i]

        self.progressWin = apps.QDialogWorkerProgress(
            title=desc, parent=self.host, pbarDesc=desc
        )
        self.progressWin.mainPbar.setMaximum(0)
        self.progressWin.show(self.app)

        self.computeAllObjCostPairsThread = QThread()
        self.computeAllObjCostPairsWorker = workers.SimpleWorker(
            posData, self._computeAllObjToObjCostPairs
        )

        self.computeAllObjCostPairsWorker.moveToThread(
            self.computeAllObjCostPairsThread
        )

        self.computeAllObjCostPairsWorker.signals.finished.connect(
            self.computeAllObjCostPairsThread.quit
        )
        self.computeAllObjCostPairsWorker.signals.finished.connect(
            self.computeAllObjCostPairsWorker.deleteLater
        )
        self.computeAllObjCostPairsThread.finished.connect(
            self.computeAllObjCostPairsThread.deleteLater
        )

        self.computeAllObjCostPairsWorker.signals.critical.connect(
            self.computeAllObjCostPairsWorkerCritical
        )
        self.computeAllObjCostPairsWorker.signals.initProgressBar.connect(
            self.workerInitProgressbar
        )
        self.computeAllObjCostPairsWorker.signals.progressBar.connect(
            self.workerUpdateProgressbar
        )
        self.computeAllObjCostPairsWorker.signals.progress.connect(
            self.workerProgress
        )
        self.computeAllObjCostPairsWorker.signals.finished.connect(
            self.computeAllObjCostPairsWorkerFinished
        )

        self.computeAllObjCostPairsThread.started.connect(
            self.computeAllObjCostPairsWorker.run
        )
        self.computeAllObjCostPairsThread.start()

        self.computeAllObjCostPairsWorkerLoop = QEventLoop()
        self.computeAllObjCostPairsWorkerLoop.exec_()

    def _computeAllObjToObjCostPairs(self, posData):
        self.computeAllObjCostPairsWorker.signals.initProgressBar.emit(
            len(posData.allData_li)
        )
        for frame_i, dataDict in enumerate(posData.allData_li):
            if frame_i == 0:
                continue

            rp = dataDict['regionprops']
            if rp is None:
                break

            prev_rp = posData.allData_li[frame_i-1]['regionprops']
            dist_matrix = self.view_model.geometry.object_to_object_contour_distance_matrix(
                dataDict['contours'], rp,
                previous_regionprops=prev_rp,
                restrict_search=True,
            )
            dataDict['obj_to_obj_dist_cost_matrix_df'] = dist_matrix
            self.computeAllObjCostPairsWorker.signals.progressBar.emit(1)
        self.computeAllObjCostPairsWorker.signals.initProgressBar.emit(0)

    def computeAllObjCostPairsWorkerCritical(self, error):
        self.computeAllObjCostPairsWorkerLoop.exit()
        self.workerCritical(error)

    def computeAllObjCostPairsWorkerFinished(self, output):
        if self.progressWin is not None:
            self.progressWin.workerFinished = True
            self.progressWin.close()
            self.progressWin = None
        self.computeAllObjCostPairsWorkerLoop.exit()

    def gui_createMothBudLinePens(self):
        if 'mothBudLineSize' in self.df_settings.index:
            val = self.df_settings.at['mothBudLineSize', 'value']
            self.mothBudLineWeight = int(val)
        else:
            self.mothBudLineWeight = 2

        self.newMothBudlineColor = (255, 0, 0)
        if 'mothBudLineColor' in self.df_settings.index:
            val = self.df_settings.at['mothBudLineColor', 'value']
            rgba = colors.rgba_str_to_values(val)
            self.mothBudLineColor = rgba[0:3]
        else:
            self.mothBudLineColor = (255,165,0)

        try:
            self.imgGrad.mothBudLineColorButton.sigColorChanging.disconnect()
            self.imgGrad.mothBudLineColorButton.sigColorChanged.disconnect()
        except Exception as e:
            pass
        try:
            for act in self.imgGrad.mothBudLineWightActionGroup.actions():
                act.toggled.disconnect()
        except Exception as e:
            pass
        for act in self.imgGrad.mothBudLineWightActionGroup.actions():
            if act.lineWeight == self.mothBudLineWeight:
                act.setChecked(True)
            else:
                act.setChecked(False)
        self.imgGrad.mothBudLineColorButton.setColor(self.mothBudLineColor[:3])

        self.imgGrad.mothBudLineColorButton.sigColorChanging.connect(
            self.updateMothBudLineColour
        )
        self.imgGrad.mothBudLineColorButton.sigColorChanged.connect(
            self.saveMothBudLineColour
        )
        for act in self.imgGrad.mothBudLineWightActionGroup.actions():
            act.toggled.connect(self.mothBudLineWeightToggled)

        # MOther-bud lines brushes
        self.NewBudMoth_Pen = pg.mkPen(
            color=self.newMothBudlineColor, width=self.mothBudLineWeight+1,
            style=Qt.DashLine
        )
        self.OldBudMoth_Pen = pg.mkPen(
            color=self.mothBudLineColor, width=self.mothBudLineWeight,
            style=Qt.DashLine
        )

        self.redDashLinePen = pg.mkPen(
            color='r', width=2, style=Qt.DashLine
        )

        self.oldMothBudLineBrush = pg.mkBrush(self.mothBudLineColor)
        self.newMothBudLineBrush = pg.mkBrush(self.newMothBudlineColor)

    def imgGradLUTfinished_cb(self):
        posData = self.data[self.pos_i]
        ticks = self.imgGrad.gradient.listTicks()

        self.img1ChannelGradients[self.user_ch_name] = {
            'ticks': [(x, t.color.getRgb()) for t, x in ticks],
            'mode': 'rgb'
        }

        self.df_settings = self.imgGrad.saveState(self.df_settings)
        self.df_settings.to_csv(self.settings_csv_path)

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

    def updateMothBudLineColour(self, colorButton):
        color = colorButton.color().getRgb()
        self.df_settings.at['mothBudLineColor', 'value'] = str(color)
        self._updateMothBudLineColour(color)
        self.updateAllImages()

    def _updateMothBudLineColour(self, color):
        self.gui_createMothBudLinePens()
        self.ax1_newMothBudLinesItem.setBrush(self.newMothBudLineBrush)
        self.ax1_oldMothBudLinesItem.setBrush(self.oldMothBudLineBrush)
        self.ax2_newMothBudLinesItem.setBrush(self.newMothBudLineBrush)
        self.ax2_oldMothBudLinesItem.setBrush(self.oldMothBudLineBrush)
        for items in self.overlayLayersItems.values():
            lutItem = items[1]
            lutItem.mothBudLineColorButton.setColor(color)

    def saveMothBudLineColour(self, colorButton):
        self.df_settings.to_csv(self.settings_csv_path)

    def mothBudLineWeightToggled(self, checked=True):
        if not checked:
            return
        self.imgGrad.uncheckContLineWeightActions()
        w = self.sender().lineWeight
        self.df_settings.at['mothBudLineSize', 'value'] = w
        self.df_settings.to_csv(self.settings_csv_path)
        self._updateMothBudLineSize(w)
        self.updateAllImages()

    def _updateMothBudLineSize(self, size):
        self.gui_createMothBudLinePens()

        for act in self.imgGrad.mothBudLineWightActionGroup.actions():
            if act == self.sender():
                act.setChecked(True)
            act.toggled.connect(self.mothBudLineWeightToggled)

        self.ax1_oldMothBudLinesItem.setSize(size)
        self.ax1_newMothBudLinesItem.setSize(size)
        self.ax2_oldMothBudLinesItem.setSize(size)
        self.ax2_newMothBudLinesItem.setSize(size)

    def gui_createContourPens(self):
        if 'contLineWeight' in self.df_settings.index:
            val = self.df_settings.at['contLineWeight', 'value']
            self.contLineWeight = int(val)
        else:
            self.contLineWeight = 1
        if 'contLineColor' in self.df_settings.index:
            val = self.df_settings.at['contLineColor', 'value']
            rgba = colors.rgba_str_to_values(val)
            self.contLineColor = rgba
            self.newIDlineColor = [min(255, v+50) for v in self.contLineColor]
        else:
            self.contLineColor = (255, 0, 0, 200)
            self.newIDlineColor = (255, 0, 0, 255)

        try:
            self.imgGrad.contoursColorButton.sigColorChanging.disconnect()
            self.imgGrad.contoursColorButton.sigColorChanged.disconnect()
        except Exception as e:
            pass
        try:
            for act in self.imgGrad.contLineWightActionGroup.actions():
                act.toggled.disconnect()
        except Exception as e:
            pass
        for act in self.imgGrad.contLineWightActionGroup.actions():
            if act.lineWeight == self.contLineWeight:
                act.setChecked(True)
        self.imgGrad.contoursColorButton.setColor(self.contLineColor[:3])

        self.imgGrad.contoursColorButton.sigColorChanging.connect(
            self.updateContColour
        )
        self.imgGrad.contoursColorButton.sigColorChanged.connect(
            self.saveContColour
        )
        for act in self.imgGrad.contLineWightActionGroup.actions():
            act.toggled.connect(self.contLineWeightToggled)

        # Contours pens
        self.oldIDs_cpen = pg.mkPen(
            color=self.contLineColor, width=self.contLineWeight
        )
        self.newIDs_cpen = pg.mkPen(
            color=self.newIDlineColor, width=self.contLineWeight+1
        )
        self.tempNewIDs_cpen = pg.mkPen(
            color='g', width=self.contLineWeight+1
        )

    def updateContColour(self, colorButton):
        color = colorButton.color().getRgb()
        self.df_settings.at['contLineColor', 'value'] = str(color)
        self._updateContColour(color)
        self.updateAllImages()

    def _updateContColour(self, color):
        self.gui_createContourPens()
        for items in self.overlayLayersItems.values():
            lutItem = items[1]
            lutItem.contoursColorButton.setColor(color)

    def saveContColour(self, colorButton):
        self.df_settings.to_csv(self.settings_csv_path)

    def contLineWeightToggled(self, checked=True):
        if not checked:
            return
        self.imgGrad.uncheckContLineWeightActions()
        w = self.sender().lineWeight
        self.df_settings.at['contLineWeight', 'value'] = w
        self.df_settings.to_csv(self.settings_csv_path)
        self._updateContLineThickness()
        self.updateAllImages()

    def _updateContLineThickness(self):
        self.gui_createContourPens()
        for act in self.imgGrad.contLineWightActionGroup.actions():
            if act == self.sender():
                act.setChecked(True)
            act.toggled.connect(self.contLineWeightToggled)

    def gui_createGraphicsItems(self):
        # Create enough PlotDataItems and LabelItems to draw contours and IDs.
        self.progressWin = apps.QDialogWorkerProgress(
            title='Creating axes items', parent=self.host,
            pbarDesc='Creating axes items (see progress in the terminal)...'
        )
        self.progressWin.show(self.app)
        self.progressWin.mainPbar.setMaximum(0)

        QTimer.singleShot(50, self._gui_createGraphicsItems)

    def gui_connectGraphicsEvents(self):
        self.img1.hoverEvent = self.gui_hoverEventImg1
        self.img2.hoverEvent = self.gui_hoverEventImg2
        self.img1.mousePressEvent = self.gui_mousePressEventImg1
        self.img1.mouseMoveEvent = self.gui_mouseDragEventImg1
        self.img1.mouseReleaseEvent = self.gui_mouseReleaseEventImg1
        self.img2.mousePressEvent = self.gui_mousePressEventImg2
        self.img2.mouseMoveEvent = self.gui_mouseDragEventImg2
        self.img2.mouseReleaseEvent = self.gui_mouseReleaseEventImg2
        self.rightImageItem.mousePressEvent = self.canvas_right_image_view.mouse_press
        self.rightImageItem.mouseMoveEvent = self.canvas_right_image_view.mouse_drag
        self.rightImageItem.mouseReleaseEvent = self.canvas_right_image_view.mouse_release
        self.rightImageItem.hoverEvent = self.gui_hoverEventRightImage
        # self.imgGrad.gradient.showMenu = self.gui_gradientContextMenuEvent
        self.imgGradRight.gradient.showMenu = (
            self.canvas_context_menu_view.show_right_image_context_menu
        )
        # self.imgGrad.vb.contextMenuEvent = self.gui_gradientContextMenuEvent
        self.ax1.sigRangeChanged.connect(
            self.display_decorations_view.view_range_changed
        )

    def gui_initImg1BottomWidgets(self):
        self.zSliceScrollBar.hide()
        self.zProjComboBox.hide()
        self.zProjLockViewButton.hide()
        self.zSliceOverlay_SB.hide()
        self.zProjOverlay_CB.hide()
        self.overlay_z_label.hide()
        self.zSliceCheckbox.hide()
        self.zSliceSpinbox.hide()
        self.SizeZlabel.hide()
