"""pyqtgraph plot, image and graphics items shared by the main GUI, the
measurements viewer, data prep and the image preview window:
MainPlotItem, BaseImageItem, ParentImageItem, labImageItem,
BaseLabelsImageItem, PlotCurveItem, MirroredCursorItem, RulerPlotItem,
RectItem and MouseMoveTrailingEdgeFilter, plus ContourItem,
ScatterPlotItem (subclassed by spotMAX), PolyLineROI and
FadingTrackItem. Leaf module: it imports no other widget module.
"""
from typing import List, Union
import os
import numpy as np
from math import ceil
import skimage.draw
import skimage.morphology
import skimage.segmentation
import matplotlib
from qtpy.QtCore import QTimer, Qt, QPointF, QEvent, QObject, PYQT6, QRectF
from qtpy.QtGui import (
    QColor, QPen, QPainter, QPicture, QMouseEvent, QTransform
)
from qtpy.QtWidgets import QApplication, QGraphicsPathItem
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import html_path
from cellacdc import load


class ContourItem(pg.PlotCurveItem):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self._prevData = None
    
    def clear(self):
        try:
            self.setData([], [])
        except AttributeError as e:
            pass
    
    def tempClear(self):
        try:
            self._prevData = [d.copy() for d in self.getData()]
            self.clear()
        except Exception as e:
            pass
    
    def restore(self):
        if self._prevData is not None:
            if self._prevData[0] is not None:
                self.setData(*self._prevData)


class ScatterPlotItem(pg.ScatterPlotItem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.updateBrushAndPen(**kwargs)
        
    def updateBrushAndPen(self, **kwargs):
        brush = kwargs.get('brush')
        if brush is not None:
            self._itemBrush = brush
        pen = kwargs.get('pen')
        if pen is not None:
            self._itemPen = pen
    
    def setData(self, *args, **kwargs):
        super().setData(*args, **kwargs)
        self.updateBrushAndPen(**kwargs)        
    
    def itemBrush(self):
        return self._itemBrush
    
    def itemPen(self):
        return self._itemPen
    
    def removePoint(self, index):
        newData = np.delete(self.data, index)
        # Update the index of current points
        for i in range(index, len(newData)):
            spotItem = newData[i]['item']
            spotItem._index = i
            newData[i]['item'] = spotItem
        
        self.data = newData
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.bounds = [None, None]
        self.invalidate()
        self.updateSpots(newData)
        self.sigPlotChanged.emit(self)
    
    def coordsToNumpy(self, includeData=False, rounded=True, decimals=None):
        points = self.points()
        nrows = len(points)
        coords_arr = np.zeros((nrows, 2))
        data_arr = None
        for p, point in enumerate(points):
            pos = point.pos()
            x, y = pos.x(), pos.y()
            if includeData:
                data = point.data()
                if data_arr is None:
                    try:
                        ncols = len(data)
                    except Exception as e:
                        data = [data]
                        ncols = 1
                    data_arr = np.zeros((nrows, ncols))
                for j, data_j in enumerate(data):
                    data_arr[p, j] = data_j
            
            coords_arr[p, 0] = y
            coords_arr[p, 1] = x
        if not includeData:
            out_arr = coords_arr
        elif data_arr is not None:
            out_arr = np.column_stack((data_arr, coords_arr))
        else:
            out_arr = coords_arr
        cast_to_int = decimals is None
        decimals = decimals if decimals is not None else 0
        if rounded:
            out_arr = np.round(out_arr, decimals)
        if cast_to_int:
            out_arr = out_arr.astype(int)
        return out_arr


class PolyLineROI(pg.PolyLineROI):
    def __init__(self, positions, closed=False, pos=None, **args):
        super().__init__(positions, closed, pos, **args)


class PlotCurveItem(pg.PlotCurveItem):    
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
    
    def addPoint(self, x, y, **kargs):
        _xx, _yy = self.getData()
        if _xx is None or len(_xx) == 0:
            self.xData = np.array([x], dtype=int)
            self.yData = np.array([y], dtype=int)
            return
        if _xx[-1] == x and _yy[-1] == y:
            # Do not append same point
            return
        
        # Pre-allocate array and insert data (faster than append)
        xx = np.zeros(len(_xx)+1, dtype=_xx.dtype)
        xx[:-1] = _xx
        xx[-1] = x
        yy = np.zeros(len(_yy)+1, dtype=_xx.dtype)
        yy[:-1] = _yy
        yy[-1] = y
        self.setData(xx, yy, **kargs)
    
    def clear(self):
        try:
            self.setData([], [])
        except Exception as e:
            pass
        super().clear()
        
    
    def closeCurve(self):
        _xx, _yy = self.getData()
        self.addPoint(_xx[0], _yy[0])
    
    def mask(self):
        ymin, xmin, ymax, xmax = self.bbox()
        _mask = np.zeros((ymax-ymin+1, xmax-xmin+1), dtype=bool)
        local_xx, local_yy = self.getLocalData()
        rr, cc = skimage.draw.polygon(local_yy, local_xx)
        _mask[rr, cc] = True
        return _mask
    
    def getLocalData(self):
        _xx, _yy = self.getData()
        return _xx - _xx.min(), _yy - _yy.min()

    def slice(self, zRange=None, tRange=None):
        ymin, xmin, ymax, xmax = self.bbox()
        if zRange is not None:
            zmin, zmax = zRange
            _slice = (slice(zmin, zmax), slice(ymin, ymax+1), slice(xmin, xmax+1))
        else:
            _slice = (slice(ymin, ymax+1), slice(xmin, xmax+1))
        if tRange is not None:
            tmin, tmax = tRange
            _slice = (slice(tmin, tmax), *_slice)
        return _slice
    
    def bbox(self):
        _xx, _yy = self.getData()
        return _yy.min(), _xx.min(), _yy.max(), _xx.max()


class MainPlotItem(pg.PlotItem):
    def __init__(
            self, parent=None, name=None, labels=None, title=None, 
            viewBox=None, axisItems=None, enableMenu=True, 
            showWelcomeText=False, ax_number=0, **kargs
        ):
        super().__init__(
            parent, name, labels, title, viewBox, axisItems, enableMenu, 
            **kargs
        )
        # Overwrite zoom out button behaviour to disable autoRange after
        # clicking it.
        # If autorange is enabled, it is called everytime the brush or eraser 
        # scatter plot items touches the border causing flickering
        self.disableAutoRange()
        self.autoBtn.mode = 'manual'
        if showWelcomeText:
            self.infoTextItem = pg.TextItem()
            self.addItem(self.infoTextItem)
            html_filepath = os.path.join(html_path, 'gui_welcome.html')
            with open(html_filepath) as html_file:
                htmlText = html_file.read()
            self.infoTextItem.setHtml(htmlText)
            self.infoTextItem.setPos(0,0)
        
        self.delRoiItems = {}
        self.highlightingRectItems = None
        self._baseImageItem = None
        self._imageItems = []
        self.highlightingRectItemsColor = None
        self.ax_number = ax_number
        
    def __eq__(self, other):
        if isinstance(other, int):
            return self.ax_number == other
        return NotImplemented
        
    def __hash__(self):
        return id(self)
        
    def addHighlightingRectItems(self, color=None):
        self.highlightingRectItems = {
            'left': RectItem(QRectF()),
            'right': RectItem(QRectF()),
            'top': RectItem(QRectF()),
            'bottom': RectItem(QRectF())
        }
        for rect in self.highlightingRectItems.values():
            self.addItem(rect)
        
        if color is None:
            return
        
        self.setHighlightingRectItemsColor(color)
    
    def setHighlightingRectItemsColor(self, color):
        if color == self.highlightingRectItemsColor:
            return
        
        for item in self.highlightingRectItems.values():
            item.setColor(color)
        
        self.highlightingRectItemsColor = color
    
    def addBaseImageItem(self, baseImageItem):
        self._baseImageItem = baseImageItem
        self._imageItems.append(baseImageItem)
        self.addItem(baseImageItem)
    
    def addImageItem(self, imageItem):
        self._imageItems.append(imageItem)
        self.addItem(imageItem)
    
    def setHighlighted(self, highlighted, color=None):
        if color is None:
            color = self.highlightingRectItemsColor
        
        if color is None:
            color = 'green'
            
        if self.highlightingRectItems is None:
            self.addHighlightingRectItems(color=color)
        
        if not highlighted:
            for rect in self.highlightingRectItems.values():
                rect.setQRect(QRectF())
            return
        
        self.setHighlightingRectItemsColor(color)
        
        ((xmin, xmax), (ymin, ymax)) = self.viewRange()
        xmin = xmin if xmin >= 0 else 0
        ymin = ymin if ymin >= 0 else 0
        if self._baseImageItem is not None:
            Y, X = self._baseImageItem.image.shape[:2]
            xmax = min(xmax, X)
            ymax = min(ymax, Y)
        
        w = xmax - xmin
        h = ymax - ymin
        
        bs = round(((w + h) / 2) * 0.02)
        if bs < 1:
            bs = 1
        
        x0 = xmin
        x1 = xmin + bs
        x2 = xmax - bs
        x3 = xmax
        
        y0 = ymin
        y1 = ymin + bs
        y2 = ymax - bs
        y3 = ymax
        
        self.highlightingRectItems['left'].setRect(x0, y0, bs, y3-y0)
        self.highlightingRectItems['top'].setRect(x1, y0, x3-x1, bs)
        self.highlightingRectItems['right'].setRect(x2, y1, bs, y3-y1)
        self.highlightingRectItems['bottom'].setRect(x1, y2, x2-x1, bs)
        self.update()
    
    def clear(self):
        super().clear()   
        
        self.delRoiItems = {}
        self.highlightingRectItems = None
        self._baseImageItem = None
        self._imageItems = []
        self.highlightingRectItemsColor = None
                   
        try:
            self.removeItem(self.infoTextItem)
        except Exception as e:
            pass
        
    def autoBtnClicked(self):
        self.vb.autoRange()
        self.autoBtn.hide()
    
    def addDelRoiItem(self, roiItem, key):
        if self.isDelRoiItemPresent(roiItem):
            return
        
        self.delRoiItems[key] = roiItem
        roiItem.key = key
        self.addItem(roiItem)
    
    def removeDelRoiItem(self, roiItem):
        key = roiItem.key
        self.delRoiItems.pop(key, None)
        try:
            self.removeItem(roiItem)
        except Exception as err:
            return
    
    def isDelRoiItemPresent(self, roiItem):
        try:
            key = roiItem.key
        except AttributeError as e:
            return False
        
        try:
            roi = self.delRoiItems[key]
        except Exception as err:
            return False
        
        return True
    
    def viewRange(self, mask_img=None):
        if mask_img is None:
            return super().viewRange()
        
        mask_rp = skimage.measure.regionprops(
            skimage.measure.label(mask_img)
        )
        if not mask_rp:
            return super().viewRange()
        
        mask_obj = mask_rp[0]
        ymin, xmin, ymax, xmax = mask_obj.bbox
        return (xmin, xmax), (ymin, ymax)
    
    def setCenter(self, center):
        center_x, center_y = center
        viewbox = self.getViewBox()
        x_min, x_max = viewbox.viewRange()[0]
        y_min, y_max = viewbox.viewRange()[1]

        x_span = x_max - x_min
        y_span = y_max - y_min

        viewbox.setRange(
            xRange=(int(center_x - x_span/2), int(center_x + x_span/2)),
            yRange=(int(center_y - y_span/2), int(center_y + y_span/2)),
            padding=0
        )


class BaseImageItem(pg.ImageItem):
    def __init__(
            self, image=None, **kargs
        ):
        self.minMaxValuesMapper = None
        self.minMaxValuesMapperPreproc = None
        self.minMaxValuesMapperCombined = None
        self.minMaxValuesMapperEqualized = None
        self.pos_i = 0
        self.z = 0
        self.frame_i = 0
        self.usePreprocessed = False
        self.useEqualized = False
        self.useCombined = False
        self._isRgba = False
        
        super().__init__(image, **kargs)
        self.autoLevelsEnabled = None
    
    def isRgba(self):
        return self._isRgba
    
    def setEnableAutoLevels(self, enabled: bool):
        self.autoLevelsEnabled = enabled
    
    def setImage(
            self, image=None, autoLevels=None, **kargs
        ):
        if autoLevels is None:
            autoLevels = self.autoLevelsEnabled
        
        if image is not None and image.ndim == 3 and image.shape[2] in (3, 4):
            self._isRgba = True
        
        super().setImage(image, autoLevels=autoLevels, **kargs)
        
    def preComputedMinMaxValues(self, data: List['load.loadData']):
        self.minMaxValuesMapper = {}
        for pos_i, posData in enumerate(data):
            img_data = posData.img_data
            requires_time_dim = (
                posData.img_data.ndim == 2
                or (posData.img_data.ndim == 3 and posData.SizeZ > 1)
            )
            if requires_time_dim:
                img_data = (img_data,)
            
            for frame_i, image in enumerate(img_data):
                if image.ndim == 3:
                    self._updateMinMaxValuesProjections(
                        image, pos_i, frame_i, self.minMaxValuesMapper
                    )
                    
                if image.ndim == 2:
                    image = (image,)
                
                for z, img in enumerate(image):
                    self.minMaxValuesMapper[(pos_i, frame_i, z)] = (
                        np.nanmin(img), np.nanmax(img)
                    )
    
    def updateMinMaxValuesEqualizedData(
            self, 
            data: List['load.loadData'], 
            pos_i: int, 
            frame_i: int, 
            z_slice: Union[int, str],
        ):
        if self.minMaxValuesMapperEqualized is None:
            self.minMaxValuesMapperEqualized = {}

        posData = data[pos_i]
        img = posData.equalized_img_data[frame_i][z_slice]
        key = (pos_i, frame_i, z_slice)
        self.minMaxValuesMapperEqualized[key] = (np.nanmin(img), np.nanmax(img)) 
    
    def updateMinMaxValuesEqualizedDataProjections(
            self, 
            data: List['load.loadData'], 
            pos_i: int, 
            frame_i: int, 
        ):    
        posData = data[pos_i]
        eq_zstack = posData.equalized_img_data[frame_i]
        
        self._updateMinMaxValuesProjections(
            eq_zstack, pos_i, frame_i, self.minMaxValuesMapperEqualized
        )
    
    def _updateMinMaxValuesProjections(self, zstack, pos_i, frame_i, mapper):
        max_proj = zstack.max(axis=0)
        key = (pos_i, frame_i, 'max z-projection')
        mapper[key] = np.nanmin(max_proj), np.nanmax(max_proj)
        
        mean_proj = zstack.mean(axis=0)
        key = (pos_i, frame_i, 'mean z-projection')
        mapper[key] = np.nanmin(mean_proj), np.nanmax(mean_proj)
        
        median_proj = np.median(zstack, axis=0)
        key = (pos_i, frame_i, 'median z-proj.')
        mapper[key] = np.nanmin(median_proj), np.nanmax(median_proj)
    
    def updateMinMaxValuesPreprocessedData(
            self, 
            data: List['load.loadData'], 
            pos_i: int, 
            frame_i: int, 
            z_slice: Union[int, str],
        ):
        if self.minMaxValuesMapperPreproc is None:
            self.minMaxValuesMapperPreproc = {}

        posData = data[pos_i]
        img = posData.preproc_img_data[frame_i][z_slice]
        key = (pos_i, frame_i, z_slice)
        self.minMaxValuesMapperPreproc[key] = (np.nanmin(img), np.nanmax(img))

    def updateMinMaxValuesPreprocessedProjections(
            self, 
            data: List['load.loadData'], 
            pos_i: int, 
            frame_i: int, 
        ):    
        posData = data[pos_i]
        zstack = posData.preproc_img_data[frame_i]
        
        self._updateMinMaxValuesProjections(
            zstack, pos_i, frame_i, self.minMaxValuesMapperPreproc
        )
    
    def updateMinMaxValuesCombinedData(
            self,
            data: List['load.loadData'],
            pos_i: int,
            frame_i: int,
            z_slice: Union[int, str],
        ):
        if self.minMaxValuesMapperCombined is None:
            self.minMaxValuesMapperCombined = {}
        
        posData = data[pos_i]
        img = posData.combine_img_data[frame_i][z_slice]
        key = (pos_i, frame_i, z_slice)
        self.minMaxValuesMapperCombined[key] = (np.nanmin(img), np.nanmax(img))
    
    def updateMinMaxValuesCombinedDataProjections(
            self, 
            data: List['load.loadData'], 
            pos_i: int, 
            frame_i: int, 
        ):    
        posData = data[pos_i]
        zstack = posData.combine_img_data[frame_i]
        
        self._updateMinMaxValuesProjections(
            zstack, pos_i, frame_i, self.minMaxValuesMapperCombined
        )
    
    def setCurrentPosIndex(self, pos_i: int):
        self.pos_i = pos_i
    
    def setCurrentFrameIndex(self, frame_i: int):
        self.frame_i = frame_i
    
    def setCurrentZsliceIndex(self, z: int):
        self.z = z
    
    def quickMinMax(self, targetSize=1e6):
        if self.isRgba():
            return super().quickMinMax(targetSize=targetSize)
            
        if self.usePreprocessed and self.minMaxValuesMapperPreproc is not None:
            minMaxValuesMapper = self.minMaxValuesMapperPreproc
        elif self.useCombined and self.minMaxValuesMapperCombined is not None:
            minMaxValuesMapper = self.minMaxValuesMapperCombined
        elif self.useEqualized and self.minMaxValuesMapperEqualized is not None:
            minMaxValuesMapper = self.minMaxValuesMapperEqualized
        else:
            minMaxValuesMapper = self.minMaxValuesMapper
        
        if minMaxValuesMapper is None:
            return super().quickMinMax(targetSize=targetSize)
        
        try:
            key = (self.pos_i, self.frame_i, self.z)
            levels = minMaxValuesMapper[key]
            return levels
        except Exception as err:
            pass
        
        try:
            key = (self.pos_i, self.frame_i, self.z)
            levels = self.minMaxValuesMapper[key]
            return levels
        except Exception as err:
            return super().quickMinMax(targetSize=targetSize)
    
    def setOpacity(self, value, **kwargs):
        if value == 0:
            value = 0.001
        
        if value == 1:
            value = 0.999
            
        super().setOpacity(value)


class BaseLabelsImageItem(pg.ImageItem):
    def __init__(
            self, image=None, **kargs
        ):
        super().__init__(image, **kargs)
    
    def setImage(self, image=None, **kwargs):
        if image is None:
            return
        autoLevels = kwargs.get('autoLevels')
        if autoLevels is None:
            kwargs['autoLevels'] = False
        super().setImage(image, **kwargs)


class ParentImageItem(BaseImageItem):
    def __init__(
            self, image=None, linkedImageItem=None, activatingActions=None,
            debug=False, **kargs
        ):
        super().__init__(image, **kargs)
        self.linkedImageItem = linkedImageItem
        self.activatingActions = activatingActions
        self.debug = debug
        self._forceDoNotUpdateLinked = False
        self.autoLevelsEnabled = None
    
    def clear(self):
        if self.linkedImageItem is not None:
            self.linkedImageItem.clear()
        return super().clear()
    
    def isLinkedImageItemActive(self):
        if self._forceDoNotUpdateLinked:
            return False
        
        if self.linkedImageItem is None:
            return False
        
        if self.activatingActions is None:
            return False
        
        for action in self.activatingActions:
            if action.isChecked():
                return True
            
        return False
    
    def setEnableAutoLevels(self, enabled: bool):
        self.autoLevelsEnabled = enabled
    
    def setUsePreprocessed(self, usePreprocessed):
        self.usePreprocessed = usePreprocessed
        if self.linkedImageItem is None:
            return
        
        self.linkedImageItem.usePreprocessed = usePreprocessed
    
    def setUseCombined(self, useCombined):
        self.useCombined = useCombined
        if self.linkedImageItem is None:
            return
        
        self.linkedImageItem.useCombined = useCombined
    
    def preComputedMinMaxValues(self, *args, **kwargs):
        super().preComputedMinMaxValues(*args, **kwargs)
        if self.linkedImageItem is None:
            return
        
        self.linkedImageItem.minMaxValuesMapper = self.minMaxValuesMapper
    
    def updateMinMaxValuesPreprocessedData(self, *args, **kwargs):
        super().updateMinMaxValuesPreprocessedData(*args, **kwargs)
        
        if self.linkedImageItem is None:
            return
        
        self.linkedImageItem.minMaxValuesMapper = self.minMaxValuesMapper
    
    def updateMinMaxValuesCombinedData(self, *args, **kwargs):
        super().updateMinMaxValuesCombinedData(*args, **kwargs)
        
        if self.linkedImageItem is None:
            return
        
        self.linkedImageItem.minMaxValuesMapperCombined = (
            self.minMaxValuesMapperCombined
        )
    
    def updateMinMaxValuesCombinedDataProjections(self, *args, **kwargs):
        super().updateMinMaxValuesCombinedDataProjections(*args, **kwargs)
        
        if self.linkedImageItem is None:
            return
        
        self.linkedImageItem.minMaxValuesMapperCombined = (
            self.minMaxValuesMapperCombined
        )
    
    def updateMinMaxValuesEqualizedDataProjections(self, *args, **kwargs):
        super().updateMinMaxValuesEqualizedDataProjections(*args, **kwargs)
        
        if self.linkedImageItem is None:
            return
        
        self.linkedImageItem.minMaxValuesMapperEqualized = (
            self.minMaxValuesMapperEqualized
        )
    
    def updateMinMaxValuesEqualizedData(self, *args, **kwargs):
        super().updateMinMaxValuesEqualizedData(*args, **kwargs)
        
        if self.linkedImageItem is None:
            return
        
        self.linkedImageItem.minMaxValuesMapperEqualized = (
            self.minMaxValuesMapperEqualized
        )
    
    def setCurrentPosIndex(self, *args, **kwargs):
        super().setCurrentPosIndex(*args, **kwargs)
        
        if self.linkedImageItem is None:
            return
        
        self.linkedImageItem.pos_i = self.pos_i
    
    def setCurrentFrameIndex(self, *args, **kwargs):
        super().setCurrentFrameIndex(*args, **kwargs)
        
        if self.linkedImageItem is None:
            return
        
        self.linkedImageItem.frame_i = self.frame_i + 1
    
    def setCurrentZsliceIndex(self, *args, **kwargs):
        super().setCurrentZsliceIndex(*args, **kwargs)
        
        if self.linkedImageItem is None:
            return
        
        self.linkedImageItem.z = self.z
    
    def setImage(
            self, image=None, autoLevels=None, next_frame_image=None, 
            scrollbar_value=None, force_set_linked=False, **kargs
        ):
        if autoLevels is None:
            autoLevels = self.autoLevelsEnabled
        
        super().setImage(image, autoLevels=autoLevels, **kargs)
        
        if self.linkedImageItem is None:
            return
        
        if not self.isLinkedImageItemActive() and not force_set_linked:
            return
        
        if next_frame_image is not None:
            self.linkedImageItem.setImage(
                next_frame_image, 
                scrollbar_value=scrollbar_value, 
                autoLevels=autoLevels
            )
        elif image is not None:
            self.linkedImageItem.setImage(image)
    
    def updateImage(self, *args, **kargs):
        if self.isLinkedImageItemActive():
            self.linkedImageItem.image = self.image
            self.linkedImageItem.updateImage(*args, **kargs)
        return super().updateImage(*args, **kargs)
    
    def setOpacity(self, value, applyToLinked=True):
        super().setOpacity(value)
        if not applyToLinked:
            return
        
        if self.linkedImageItem is None:
            return
        
        self.linkedImageItem.setOpacity(value)
    
    def setLookupTable(self, lut):
        super().setLookupTable(lut)
        # if self.linkedImageItem is not None:
        #     self.linkedImageItem.setLookupTable(lut)
    
    def setLevels(self, levels, **kwargs):
        super().setLevels(levels, **kwargs)
        if self.linkedImageItem is None:
            return
        
        self.linkedImageItem.setLevels(levels, **kwargs)


class labImageItem(pg.ImageItem):
    def __init__(self, *args, **kwargs):
        pg.ImageItem.__init__(self, *args, **kwargs)

    def setImage(self, img=None, z=None, **kargs):
        autoLevels = kargs.get('autoLevels')
        if autoLevels is None:
            kargs['autoLevels'] = False

        if img is None:
            pg.ImageItem.setImage(self, img, **kargs)
            return

        if img.ndim == 3 and img.shape[-1] > 4 and z is not None:
            pg.ImageItem.setImage(self, img[z], **kargs)
        else:
            pg.ImageItem.setImage(self, img, **kargs)


class MirroredCursorItem(QGraphicsPathItem):
    """Cursor mirroring the mouse cursor on another image (same look as a
    `pg.ScatterPlotItem` with one point, symbol `symbol`, size `size` pixels).

    Much faster to move than a `pg.ScatterPlotItem`: moving it only
    repaints its small region and does not change the bounds of the plot.
    It has the same `setData(xx, yy)` method (one or zero points).
    """
    def __init__(self, symbol='d', size=16, pen=None, brush=None):
        super().__init__()
        path = pg.graphicsItems.ScatterPlotItem.Symbols[symbol]
        self.setPath(QTransform.fromScale(size, size).map(path))
        if pen is not None:
            self.setPen(pen)
        if brush is not None:
            self.setBrush(brush)
        # Fixed size in pixels, i.e., independent of the zoom
        self.setFlag(QGraphicsPathItem.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setZValue(1e9)
        # Own state, not `isVisible()` (False whenever the plot is hidden)
        self._isShown = False
        self.hide()

    def setData(self, xx, yy):
        if len(xx) == 0:
            if self._isShown:
                self._isShown = False
                self.hide()
            return

        self.setPos(float(xx[0]), float(yy[0]))
        if not self._isShown:
            self._isShown = True
            self.show()

    def getData(self):
        if not self._isShown:
            return np.array([]), np.array([])
        pos = self.pos()
        return np.array([pos.x()]), np.array([pos.y()])


class MouseMoveTrailingEdgeFilter(QObject):
    """Make sure that the last mouse move over a pyqtgraph view is always
    delivered to the items (e.g., hover events).

    pyqtgraph ignores the mouse moves that come less than
    1/`mouseRateLimit` seconds (default 100 Hz) after the last delivered
    one. If the mouse stops right after an ignored move, the hover position
    (e.g., of mirrored cursors) is stale until the mouse moves again.
    This filter sends the last ignored move again when it is allowed.
    """
    def __init__(self, view: pg.GraphicsView):
        super().__init__(view)
        self._view = view
        self._viewport = view.viewport()
        self._lastIgnoredMove = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._sendLastIgnoredMove)
        self._viewport.installEventFilter(self)

    def _isMoveIgnoredByScene(self):
        isMoveAllowed = getattr(self._view.scene(), '_moveEventIsAllowed', None)
        if isMoveAllowed is None:
            return False
        return not isMoveAllowed()

    def eventFilter(self, watched, event):
        if watched is not self._viewport:
            return False

        eventType = event.type()
        if eventType == QEvent.Type.MouseMove:
            if not self._isMoveIgnoredByScene():
                self._timer.stop()
                self._lastIgnoredMove = None
                return False

            if PYQT6:
                localPos = QPointF(event.position())
                globalPos = QPointF(event.globalPosition())
            else:
                localPos = QPointF(event.localPos())
                globalPos = QPointF(event.screenPos())
            self._lastIgnoredMove = (localPos, globalPos)
            if not self._timer.isActive():
                rateLimit = pg.getConfigOption('mouseRateLimit')
                self._timer.start(ceil(1000/rateLimit) + 1)
        elif eventType in self._cancellingEventTypes:
            # E.g., a button press/release is delivered at its own position
            # and a move with an outdated button state must not follow it
            self._timer.stop()
            self._lastIgnoredMove = None
        return False

    _cancellingEventTypes = (
        QEvent.Type.Leave, QEvent.Type.Hide, QEvent.Type.MouseButtonPress,
        QEvent.Type.MouseButtonRelease, QEvent.Type.MouseButtonDblClick,
        QEvent.Type.Wheel
    )

    def _sendLastIgnoredMove(self):
        if self._lastIgnoredMove is None:
            return

        localPos, globalPos = self._lastIgnoredMove
        self._lastIgnoredMove = None
        event = QMouseEvent(
            QEvent.Type.MouseMove, localPos, globalPos,
            Qt.MouseButton.NoButton, QApplication.mouseButtons(),
            QApplication.keyboardModifiers()
        )
        QApplication.sendEvent(self._viewport, event)


class RulerPlotItem(pg.PlotDataItem):
    def __init__(self, *args, **kwargs):
        self.labelItem = pg.LabelItem()
        super().__init__(*args, **kwargs)
        
    def setData(self, *args, lengthText='', **kwargs):
        super().setData(*args, **kwargs)
        self.labelItem.setText('')
        if not lengthText:
            return
        self.setLengthText(lengthText)
    
    def setLengthText(self, lengthText):
        xx, yy = self.getData()
        x0, x1 = sorted(xx)
        y0, y1 = sorted(yy)
        xc = round(x0 + (x1-x0)/2)
        yc = round(y0 + (y1-y0)/2)
        self.labelItem.setText(lengthText, size='11px', color='r')
        # xc = x0 + self._length/2
        wl = self.labelItem.itemRect().width()
        hl = self.labelItem.itemRect().height()
        xl = xc-wl/2
        yt = y0-hl    
        self.labelItem.setPos(xl, yt)


class RectItem(pg.GraphicsObject):
    def __init__(self, rect, pen=None, brush=(255, 0, 0, 100), parent=None):
        super().__init__(parent)
        self._rect = rect
        self._pen = pg.mkPen(pen)
        self._brush = pg.mkBrush(brush)
        self.picture = QPicture()
        self._generate_picture()

    def setColor(self, color):
        rgba = matplotlib.colors.to_rgba(color, alpha=100/255)
        rgba = [round(c*255) for c in rgba]
        self._brush = pg.mkBrush(rgba)
        self._generate_picture()
        self.update()
    
    def setRect(self, x, y, width, height):
        self._rect = QRectF(x, y, width, height)
        self._generate_picture()
        self.update()
    
    def setQRect(self, qrect):
        self._rect = qrect
        self._generate_picture()
        self.update()
    
    @property
    def rect(self):
        return self._rect

    def _generate_picture(self):
        painter = QPainter(self.picture)
        painter.setPen(self._pen)
        painter.setBrush(self._brush)
        painter.drawRect(self._rect)
        painter.end()

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())


class FadingTrackItem(pg.GraphicsObject):
    def __init__(self, points, frames, color=(255, 100, 0), n_fade=15,
                 max_width=4, min_width=1, max_alpha=255, min_alpha=20):
        super().__init__()
        self.points = points
        self.frames = frames
        self.color = color
        self.n_fade = max(n_fade, 1)
        self.max_width, self.min_width = max_width, min_width
        self.max_alpha, self.min_alpha = max_alpha, min_alpha
        self._generate_picture()

    def _generate_picture(self):
        self.picture = QPicture()
        painter = QPainter(self.picture)
        n = len(self.points)
        if n < 2:
            painter.end()
            return

        head_frame = self.frames[-1]  # most recent frame in this track
        point_cache = {}
        pen_cache = {}

        for i in range(n - 1):
            f0, f1 = self.frames[i], self.frames[i + 1]
            if f1 - f0 > 1:
                # gap in tracking -- don't draw a line across it
                continue

            dist_from_head = head_frame - f1
            frac = max(0.0, 1.0 - dist_from_head / self.n_fade)

            alpha = int(self.min_alpha + frac * (self.max_alpha - self.min_alpha))
            width = self.min_width + frac * (self.max_width - self.min_width)

            pen_key = (alpha, width)
            pen = pen_cache.get(pen_key)
            if pen is None:
                pen = QPen(QColor(*self.color, alpha))
                pen.setWidthF(width)
                pen.setCapStyle(Qt.RoundCap)
                pen_cache[pen_key] = pen
            painter.setPen(pen)

            x0, y0 = self.points[i]
            x1, y1 = self.points[i + 1]
            p0_key = (x0, y0)
            p1_key = (x1, y1)
            p0 = point_cache.get(p0_key)
            if p0 is None:
                p0 = QPointF(x0, y0)
                point_cache[p0_key] = p0
            p1 = point_cache.get(p1_key)
            if p1 is None:
                p1 = QPointF(x1, y1)
                point_cache[p1_key] = p1
            painter.drawLine(p0, p1)
        painter.end()

    def paint(self, painter, *args):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        if not self.points:
            return QRectF()
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return QRectF(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
