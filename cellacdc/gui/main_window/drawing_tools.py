"""Segmentation drawing tools of the main GUI (brush, eraser, magic wand,
curvature tool, expand/move label) and the manual background object
tool. Mixins of guiWin (gui/main_window/window.py) only: the
measurements viewer never composes them.
"""
import cv2
import numpy as np
import scipy.optimize
import scipy.interpolate
import scipy.ndimage
import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.draw
import skimage.exposure
import skimage.transform
import skimage.segmentation
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import QGraphicsSceneMouseEvent
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import exception_handler
from cellacdc import core
from cellacdc import _warnings
from . import constants as gui_main_constants


class DrawingToolsGuiElements:
    """Segmentation drawing tools: brush, eraser, magic wand, curvature
    (spline) tool, expand and move label, z interpolation and the
    temporary brush layers.
    """

    def resetExpandLabel(self):
        self.expandingID = -1

    def expandLabelCallback(self, checked):
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            self.connectLeftClickButtons()
            self.expandFootprintSize = 1
        else:
            self.clearHighlightedID()
            alpha = self.imgGrad.labelsAlphaSlider.value()
            self.labelsLayerImg1.setOpacity(alpha)
            self.labelsLayerRightImg.setOpacity(alpha)
            self.hoverLabelID = 0
            self.expandingID = 0
            self.updateAllImages()

    def expandLabel(self, dilation=True):
        posData = self.data[self.pos_i]
        if self.hoverLabelID == 0:
            self.isExpandingLabel = False
            return

        # Re-initialize label to expand when we hover on a different ID
        # or we change direction
        reinitExpandingLab = (
            self.expandingID != self.hoverLabelID
            or dilation != self.isDilation
        )

        ID = self.hoverLabelID

        obj = posData.rp.get_obj_from_ID(ID)

        if reinitExpandingLab:
            # Store undo state before modifying stuff
            self.storeUndoRedoStates(False)
            # hoverLabelID different from previously expanded ID --> reinit
            self.isExpandingLabel = True
            self.expandingID = ID
            self.expandingLab = np.zeros_like(self.currentLab2D)
            self.expandingLab[obj.coords[:,-2], obj.coords[:,-1]] = ID
            self.expandFootprintSize = 1

        prevCoords = (obj.coords[:,-2], obj.coords[:,-1])
        self.currentLab2D[obj.coords[:,-2], obj.coords[:,-1]] = 0
        lab_2D = self.get_2Dlab(posData.lab)
        lab_2D[obj.coords[:,-2], obj.coords[:,-1]] = 0

        footprint = skimage.morphology.disk(self.expandFootprintSize)
        if dilation:
            expandedLab = skimage.morphology.dilation(
                self.expandingLab, footprint
            )
            self.isDilation = True
        else:
            expandedLab = skimage.morphology.erosion(
                self.expandingLab, footprint
            )
            self.isDilation = False

        # Prevent expanding into neighbouring labels
        expandedLab[self.currentLab2D>0] = 0

        # Get coords of the dilated/eroded object
        expandedObj = self._acdcRegionProps(
            expandedLab, precache_centroids=False)[0]
        expandedObj_bbox = expandedObj.bbox
        expandedObjCoords = (expandedObj.coords[:,-2], expandedObj.coords[:,-1])

        # Add the dilated/erored object
        self.currentLab2D[expandedObjCoords] = self.expandingID
        lab_2D[expandedObjCoords] = self.expandingID

        self.set_2Dlab(lab_2D)
        self.currentLab2D = lab_2D
        
        preloaded_bbox = self.update_rp_get_bbox(custom_bbox=expandedObj_bbox)
        self.update_rp(preloaded_bbox=preloaded_bbox, specific_IDs=ID)
        # we dont draw over other IDs so this is rare case where its fine
        # to just have tight bbox and specific_IDs=ID 
        
        if self.labelsGrad.showLabelsImgAction.isChecked():
            self.img2.setImage(img=self.currentLab2D, autoLevels=False)

        self.setTempImgExpandLabel(prevCoords, expandedObjCoords)

    def startMovingLabel(self, xPos, yPos):
        posData = self.data[self.pos_i]
        xdata, ydata = int(xPos), int(yPos)
        lab_2D = self.get_2Dlab(posData.lab)
        ID = lab_2D[ydata, xdata]
        if ID == 0:
            self.isMovingLabel = False
            return

        posData = self.data[self.pos_i]
        self.isMovingLabel = True

        self.searchedIDitemRight.setData([], [])
        self.searchedIDitemLeft.setData([], [])
        self.movingID = ID
        self.prevMovePos = (xdata, ydata)
        movingObj = posData.rp.get_obj_from_ID(ID)
        self.movingObjCoords = movingObj.coords.copy()
        yy, xx = movingObj.coords[:,-2], movingObj.coords[:,-1]
        self.currentLab2D[yy, xx] = 0

    def moveLabel(self, xPos, yPos):
        posData = self.data[self.pos_i]
        lab_2D = self.get_2Dlab(posData.lab)
        Y, X = lab_2D.shape
        xdata, ydata = int(xPos), int(yPos)
        if xdata<0 or ydata<0 or xdata>=X or ydata>=Y:
            return

        self.clearObjContour(ID=self.movingID, ax=0)

        xStart, yStart = self.prevMovePos
        deltaX = xdata-xStart
        deltaY = ydata-yStart

        yy, xx = self.movingObjCoords[:,-2], self.movingObjCoords[:,-1]

        if self.isSegm3D:
            zz = self.movingObjCoords[:,0]
            posData.lab[zz, yy, xx] = 0
        else:
            posData.lab[yy, xx] = 0

        self.movingObjCoords[:,-2] = self.movingObjCoords[:,-2]+deltaY
        self.movingObjCoords[:,-1] = self.movingObjCoords[:,-1]+deltaX

        yy, xx = self.movingObjCoords[:,-2], self.movingObjCoords[:,-1]

        yy[yy<0] = 0
        xx[xx<0] = 0
        yy[yy>=Y] = Y-1
        xx[xx>=X] = X-1

        if self.isSegm3D:
            zz = self.movingObjCoords[:,0]
            posData.lab[zz, yy, xx] = self.movingID
        else:
            posData.lab[yy, xx] = self.movingID
        
        self.currentLab2D = self.get_2Dlab(posData.lab)
        if self.labelsGrad.showLabelsImgAction.isChecked():
            self.img2.setImage(self.currentLab2D, autoLevels=False)
        
        self.setTempImg1MoveLabel()

        self.prevMovePos = (xdata, ydata)

    # @exec_time
    def fillHolesID(self, ID, sender='brush', enabled=None):
        posData = self.data[self.pos_i]
        if sender == 'brush':
            if enabled is None:
                enabled = self.brushAutoFillCheckbox.isChecked()

            if not enabled:
                return False

            if not self.brushButton.isChecked():
                return False
            
            lab2D = self.get_2Dlab(posData.lab)
            mask = lab2D == ID
            filledMask = scipy.ndimage.binary_fill_holes(mask)
            newFilledMask = np.logical_and(filledMask, ~mask)
            if not np.any(newFilledMask):
                return False

            # Apply only newly filled pixels to avoid rewriting the full 3D
            # stack when editing in projection mode.
            self.applyBrushMask(newFilledMask, ID)
            return True
        return False

    def isPowerBrush(self):
        color = self.brushButton.palette().button().color().name()
        return color == self.doublePressKeyButtonColor

    def isPowerEraser(self):
        color = self.eraserButton.palette().button().color().name()
        return color == self.doublePressKeyButtonColor

    def isPowerButton(self, button):
        color = button.palette().button().color().name()
        return color == self.doublePressKeyButtonColor

    # @exec_time
    def getPolygonBrush(self, yxc2, Y, X):
        # see https://en.wikipedia.org/wiki/Tangent_lines_to_circles
        y1, x1 = self.yPressAx2, self.xPressAx2
        y2, x2 = yxc2
        R = self.brushSizeSpinbox.value()
        r = R

        arcsin_den = np.sqrt((x2-x1)**2+(y2-y1)**2)
        arctan_den = (x2-x1)
        if arcsin_den!=0 and arctan_den!=0:
            beta = np.arcsin((R-r)/arcsin_den)
            gamma = -np.arctan((y2-y1)/arctan_den)
            alpha = gamma-beta
            x3 = x1 + r*np.sin(alpha)
            y3 = y1 + r*np.cos(alpha)
            x4 = x2 + R*np.sin(alpha)
            y4 = y2 + R*np.cos(alpha)

            alpha = gamma+beta
            x5 = x1 - r*np.sin(alpha)
            y5 = y1 - r*np.cos(alpha)
            x6 = x2 - R*np.sin(alpha)
            y6 = y2 - R*np.cos(alpha)

            rr_poly, cc_poly = skimage.draw.polygon([y3, y4, y6, y5],
                                                    [x3, x4, x6, x5],
                                                    shape=(Y, X))
        else:
            rr_poly, cc_poly = [], []

        self.yPressAx2, self.xPressAx2 = y2, x2
        return rr_poly, cc_poly

    def get_dir_coords(self, alfa_dir, yd, xd, shape, connectivity=1):
        h, w = shape
        y_above = yd+1 if yd+1 < h else yd
        y_below = yd-1 if yd > 0 else yd
        x_right = xd+1 if xd+1 < w else xd
        x_left = xd-1 if xd > 0 else xd
        if alfa_dir == 0:
            yy = [y_below, y_below, yd, y_above, y_above]
            xx = [xd, x_right, x_right, x_right, xd]
        elif alfa_dir == 45:
            yy = [y_below, y_below, y_below, yd, y_above]
            xx = [x_left, xd, x_right, x_right, x_right]
        elif alfa_dir == 90:
            yy = [yd, y_below, y_below, y_below, yd]
            xx = [x_left, x_left, xd, x_right, x_right]
        elif alfa_dir == 135:
            yy = [y_above, yd, y_below, y_below, y_below]
            xx = [x_left, x_left, x_left, xd, x_right]
        elif alfa_dir == -180 or alfa_dir == 180:
            yy = [y_above, y_above, yd, y_below, y_below]
            xx = [xd, x_left, x_left, x_left, xd]
        elif alfa_dir == -135:
            yy = [y_below, yd, y_above, y_above, y_above]
            xx = [x_left, x_left, x_left, xd, x_right]
        elif alfa_dir == -90:
            yy = [yd, y_above, y_above, y_above, yd]
            xx = [x_left, x_left, xd, x_right, x_right]
        else:
            yy = [y_above, y_above, y_above, yd, y_below]
            xx = [x_left, xd, x_right, x_right, x_right]
        if connectivity == 1:
            return yy[1:4], xx[1:4]
        else:
            return yy, xx

    def drawAutoContour(self, y2, x2):
        y1, x1 = self.autoCont_y0, self.autoCont_x0
        Dy = abs(y2-y1)
        Dx = abs(x2-x1)
        edge = self.getDisplayedImg1()
        if Dy != 0 or Dx != 0:
            # NOTE: numIter takes care of any lag in mouseMoveEvent
            numIter = int(round(max((Dy, Dx))))
            alfa = np.arctan2(y1-y2, x2-x1)
            base = np.pi/4
            alfa_dir = round((base * round(alfa/base))*180/np.pi)
            for _ in range(numIter):
                y1, x1 = self.autoCont_y0, self.autoCont_x0
                yy, xx = self.get_dir_coords(alfa_dir, y1, x1, edge.shape)
                a_dir = edge[yy, xx]
                min_int = np.max(a_dir)
                min_i = list(a_dir).index(min_int)
                y, x = yy[min_i], xx[min_i]
                try:
                    xx, yy = self.curvHoverPlotItem.getData()
                except TypeError:
                    xx, yy = [], []

                if xx is None or yy is None or len(xx) == 0 or len(yy) == 0:
                    xx, yy = [], []
                elif x == xx[-1] and y == yy[-1]:
                    # Do not append point equal to last point
                    return

                xx = np.r_[xx, x]
                yy = np.r_[yy, y]
                try:
                    self.curvHoverPlotItem.setData(xx, yy)
                    self.curvPlotItem.setData(xx, yy)
                except TypeError:
                    pass
                self.autoCont_y0, self.autoCont_x0 = y, x
                # self.smoothAutoContWithSpline()

    def smoothAutoContWithSpline(self, n=3):
        try:
            xx, yy = self.curvHoverPlotItem.getData()
            if xx is None or yy is None:
                return
            # Downsample by taking every nth coord
            xxA, yyA = xx[::n], yy[::n]
            rr, cc = skimage.draw.polygon(yyA, xxA)
            self.autoContObjMask[rr, cc] = 1
            rp = self._acdcRegionProps(
                self.autoContObjMask, precache_centroids=False
            )
            if not rp:
                return
            obj = rp[0]
            cont = self.getObjContours(obj)
            xxC, yyC = cont[:,0], cont[:,1]
            xxA, yyA = xxC[::n], yyC[::n]
            self.xxA_autoCont, self.yyA_autoCont = xxA, yyA
            xxS, yyS = self.getSpline(xxA, yyA, per=True, appendFirst=True)
            if len(xxS)>0:
                self.curvPlotItem.setData(xxS, yyS)
        except (TypeError, ValueError):
            pass

    def getClosedSplineCoords(self):
        xxS, yyS = self.curvPlotItem.getData()
        bbox_area = (xxS.max()-xxS.min())*(yyS.max()-yyS.min())
        if bbox_area < 26_000:
            # Using 1000 is fast enough according to profiling
            return xxS, yyS 
        
        optimalSpaceSize = self.splineToObjModel.predict(
            bbox_area, max_exec_time=150
        )
        if optimalSpaceSize >= 1000:
            # Using 1000 is fast enough according to model
            return xxS, yyS
        
        if optimalSpaceSize < 100:
            # Do not allow a rough spline
            optimalSpaceSize = 100
        
        # Get spline with optimal space size so that exec time 
        # or skimage.draw.polygon is less than 150 ms
        xx, yy = self.curvAnchors.getData()
        resolutionSpace = np.linspace(0, 1, int(optimalSpaceSize))
        xxS, yyS = self.getSpline(
            xx, yy, resolutionSpace=resolutionSpace, per=True
        )
        return xxS, yyS


    def getSpline(self, xx, yy, resolutionSpace=None, per=False, appendFirst=False):
        # Remove duplicates
        valid = np.where(np.abs(np.diff(xx)) + np.abs(np.diff(yy)) > 0)
        xx = np.r_[xx[valid], xx[-1]]
        yy = np.r_[yy[valid], yy[-1]]
        if appendFirst:
            xx = np.r_[xx, xx[0]]
            yy = np.r_[yy, yy[0]]
            per = True

        # Interpolate splice
        if resolutionSpace is None:
            resolutionSpace = self.hoverLinSpace
        k = 2 if len(xx) == 3 else 3

        try:
            tck, u = scipy.interpolate.splprep(
                [xx, yy], s=0, k=k, per=per
            )
            xi, yi = scipy.interpolate.splev(resolutionSpace, tck)
            return xi, yi
        except (ValueError, TypeError):
            # Catch errors where we know why splprep fails
            return [], []

    def brushAutoFillToggled(self, checked):
        val = 'Yes' if checked else 'No'
        self.df_settings.at['brushAutoFill', 'value'] = val
        self.df_settings.to_csv(self.settings_csv_path)

    def brushAutoHideToggled(self, checked):
        val = 'Yes' if checked else 'No'
        self.df_settings.at['brushAutoHide', 'value'] = val
        self.df_settings.to_csv(self.settings_csv_path)

    def brushReleased(self):
        posData = self.data[self.pos_i]
        do_auto_fill = self.brushAutoFillCheckbox.isChecked()
        self.fillHolesID(posData.brushID, sender='brush', enabled=do_auto_fill)
        
        # Update data (rp, etc)
        
        power_brush = self.isPowerBrush()
        # we have to delay for a second
        self.update_rp(
            use_curr_view=True, 
            specific_IDs=posData.brushID if not power_brush else None
        )
        
        # Repeat tracking
        if self.autoIDcheckbox.isChecked():
            self.trackManuallyAddedObject(posData.brushID, self.isNewID)

        # Update images
        if self.isNewID:
            editTxt = 'Add new ID with brush tool'
            if self.isSnapshot:
                self.fixCcaDfAfterEdit(editTxt)
                self.updateAllImages()
            else:
                self.warnEditingWithCca_df(editTxt)
        else:
            self.updateAllImages()
        
        self.isNewID = False

    def brushSize_cb(self, value):
        self.ax2_EraserCircle.setSize(value*2)
        self.ax1_BrushCircle.setSize(value*2)
        self.ax2_BrushCircle.setSize(value*2)
        self.ax1_EraserCircle.setSize(value*2)
        self.ax2_EraserX.setSize(value)
        self.ax1_EraserX.setSize(value)
        self.setDiskMask()

    def interpZModeToggled(self, checked):
        if checked:
            self.autoIDcheckbox.stateToRestore = self.autoIDcheckbox.isChecked()

        if not checked:
            try:
                self.autoIDcheckbox.setChecked(
                    self.autoIDcheckbox.stateToRestore
                )
            except Exception as err:
                pass

            self.autoIDcheckbox.setDisabled(False)
            return
        
        if self.autoIDcheckbox.isChecked():
            self.autoIDcheckbox.setChecked(False)
            self.autoIDcheckbox.setDisabled(True)

        newID = self.setBrushID(return_val=True)
        self.editIDspinbox.setValue(newID)

    @exception_handler
    def interpZConfirmTriggered(self):
        posData = self.data[self.pos_i]
        ID = self.editIDspinbox.value()
        if ID == 0:
            _warnings.warnCannotInterpolateZVolume(self, ID)
            return

        self.logger.info(f'Interpolating object ID = {ID}...')
        obj = posData.rp.get_obj_from_ID(ID, warn=False)
        if obj is None:
            _warnings.warnCannotInterpolateZVolume(self, ID)
            return

        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False) 

        local_lab = posData.lab[obj.slice]
        local_mask_volume = local_lab == ID
        labelled_z_slices = np.flatnonzero(
            np.any(local_mask_volume, axis=(1, 2))
        )

        if len(labelled_z_slices) < 2:
            _warnings.warnCannotInterpolateZVolume(self, ID)
            return

        local_mask_filled, _ = core.interpolate_unlabelled_z_slices(
            local_mask_volume, labelled_z_slices
        )

        existing_objs_mask = np.logical_and(local_lab > 0, local_lab != ID)

        local_mask_filled[existing_objs_mask] = False
        posData.lab[obj.slice][local_mask_filled] = ID
        self.update_rp(specific_IDs=ID, preloaded_bbox=obj.bbox)
        self.updateAllImages()

        button = self.brushEraserToolBar.widgetForAction(
            self.interpZConfirmAction
        )
        button.setStyleSheet(f'background-color: {gui_main_constants.GREEN_HEX}')

        self.editIDspinbox.setValue(ID + 1)

        QTimer.singleShot(2000, self.restoreInterpZConfirmActionColor)

    def restoreInterpZConfirmActionColor(self):
        color = self.defaultToolBarButtonColor
        button = self.brushEraserToolBar.widgetForAction(
            self.interpZConfirmAction
        )
        button.setStyleSheet(f'background-color: {color}')

    def autoIDtoggled(self, checked):
        self.editIDspinboxAction.setDisabled(checked)
        self.editIDLabelAction.setDisabled(checked)
        if not checked and self.editIDspinbox.value() == 0:
            newID = self.setBrushID(return_val=True)
            self.editIDspinbox.setValue(newID)

    def wand_cb(self, checked):
        posData = self.data[self.pos_i]
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.wandToolButton)
            self.connectLeftClickButtons()
            self.wandControlsToolbar.setVisible(True)
            # self.secondLevelToolbar.setVisible(False)
        else:
            self.resetCursors()
            # self.secondLevelToolbar.setVisible(True)
            self.wandControlsToolbar.setVisible(False)

    def restoreHoverObjBrush(self):
        posData = self.data[self.pos_i]
        if self.ax1BrushHoverID in posData.IDs:
            obj = posData.rp.get_obj_from_ID(self.ax1BrushHoverID)
            if not self.isObjInCurrSlice(obj.bbox):
                return

            display_rp = self.get2DRP()
            obj = display_rp.get_obj_from_ID(self.ax1BrushHoverID)
            for ax in (self.ax1, self.ax2):
                self.addObjContourToContoursImage(obj=obj, ax=ax)

    def hideItemsHoverBrush(self, xy=None, ID=None, force=False):
        if xy is not None:
            x, y = xy
            if x is None:
                return

            xdata, ydata = int(x), int(y)
            Y, X = self.currentLab2D.shape

            if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
                return

        if not self.brushAutoHideCheckbox.isChecked() and not force:
            return
        
        posData = self.data[self.pos_i]
        size = self.brushSizeSpinbox.value()*2

        if xy is not None:
            ID = self.get_2Dlab(posData.lab)[ydata, xdata]

        if self.ax1_lostObjScatterItem.isVisible():
            self.ax1_lostObjScatterItem.setVisible(False)

        if self.ax1_lostTrackedScatterItem.isVisible():
            self.ax1_lostTrackedScatterItem.setVisible(False)
        
        if self.ax2_lostObjScatterItem.isVisible():
            self.ax2_lostObjScatterItem.setVisible(False)

        if self.ax2_lostTrackedScatterItem.isVisible():
            self.ax2_lostTrackedScatterItem.setVisible(False)
            
        # Restore ID previously hovered
        if ID != self.ax1BrushHoverID and not self.isMouseDragImg1:
            # try:
                self.restoreHoverObjBrush() # maybe want to put into try except later again
            # except Exception as e:
            #     self.ax1BrushHoverID = 0
            #     return

        # Hide items hover ID
        if ID != 0:
            self.clearObjContour(ID=ID, ax=0)
            self.clearObjContour(ID=ID, ax=1)
            self.ax1BrushHoverID = ID
        else:
            self.ax1BrushHoverID = 0

    def moveLabelButtonToggled(self, checked):
        if not checked:
            self.hoverLabelID = 0
            self.highlightedID = 0
            self.highLightIDLayerImg1.clear()
            self.highLightIDLayerRightImage.clear()
            self.setHighlightID(False)

    def Brush_cb(self, checked):
        if checked:
            self.typingEditID = False
            self.setDiskMask()
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )
            self.updateBrushCursor(self.xHoverImg, self.yHoverImg)
            self.setBrushID()

            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            c = self.defaultToolBarButtonColor
            self.eraserButton.setStyleSheet(f'background-color: {c}')
            self.connectLeftClickButtons()
            self.setFocusGraphics()
        else:
            self.ax1_lostObjScatterItem.setVisible(True)
            self.ax2_lostObjScatterItem.setVisible(True)
            self.ax1_lostTrackedScatterItem.setVisible(True)
            self.ax2_lostTrackedScatterItem.setVisible(True)

            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )
            self.resetCursors()
        
        self.showEditIDwidgets(checked)
        self.enableSizeSpinbox(checked)
        QTimer.singleShot(20, self.gui_refreshToolCursor)

    def setDiskMask(self):
        brushSize = self.brushSizeSpinbox.value()
        # diam = brushSize*2
        # center = (brushSize, brushSize)
        # diskShape = (diam+1, diam+1)
        # diskMask = np.zeros(diskShape, bool)
        # rr, cc = skimage.draw.disk(center, brushSize+1, shape=diskShape)
        # diskMask[rr, cc] = True
        self.diskMask = skimage.morphology.disk(brushSize, dtype=bool)

    def getDiskMask(self, xdata, ydata):
        Y, X = self.currentLab2D.shape[-2:]

        brushSize = self.brushSizeSpinbox.value()
        yBottom, xLeft = ydata-brushSize, xdata-brushSize
        yTop, xRight = ydata+brushSize+1, xdata+brushSize+1

        if xLeft<0:
            if yBottom<0:
                # Disk mask out of bounds top-left
                diskMask = self.diskMask.copy()
                diskMask = diskMask[-yBottom:, -xLeft:]
                yBottom = 0
            elif yTop>Y:
                # Disk mask out of bounds bottom-left
                diskMask = self.diskMask.copy()
                diskMask = diskMask[0:Y-yBottom, -xLeft:]
                yTop = Y
            else:
                # Disk mask out of bounds on the left
                diskMask = self.diskMask.copy()
                diskMask = diskMask[:, -xLeft:]
            xLeft = 0

        elif xRight>X:
            if yBottom<0:
                # Disk mask out of bounds top-right
                diskMask = self.diskMask.copy()
                diskMask = diskMask[-yBottom:, 0:X-xLeft]
                yBottom = 0
            elif yTop>Y:
                # Disk mask out of bounds bottom-right
                diskMask = self.diskMask.copy()
                diskMask = diskMask[0:Y-yBottom, 0:X-xLeft]
                yTop = Y
            else:
                # Disk mask out of bounds on the right
                diskMask = self.diskMask.copy()
                diskMask = diskMask[:, 0:X-xLeft]
            xRight = X

        elif yBottom<0:
            # Disk mask out of bounds on top
            diskMask = self.diskMask.copy()
            diskMask = diskMask[-yBottom:]
            yBottom = 0

        elif yTop>Y:
            # Disk mask out of bounds on bottom
            diskMask = self.diskMask.copy()
            diskMask = diskMask[0:Y-yBottom]
            yTop = Y

        else:
            # Disk mask fully inside the image
            diskMask = self.diskMask

        return yBottom, xLeft, yTop, xRight, diskMask

    def setBrushID(self, useCurrentLab=True, return_val=False):
        # Make sure that the brushed ID is always a new one based on
        # already visited frames
        posData = self.data[self.pos_i]
        wl_init = posData.whitelist and posData.whitelist.whitelistIDs
        if useCurrentLab:
            IDs_tot = set(posData.IDs)
            if wl_init:
                try:
                    IDs_tot.update(posData.whitelist.originalLabsIDs[posData.frame_i])
                except:
                    pass
                try:
                    if posData.whitelist.whitelistIDs[posData.frame_i]:
                        IDs_tot.update(posData.whitelist.whitelistIDs[posData.frame_i])
                except:
                    pass
            newID = max(IDs_tot, default=0)
        else:
            newID = 0
        for frame_i, storedData in enumerate(posData.allData_li):
            if frame_i == posData.frame_i:
                continue
            lab = storedData['labels']
            if lab is not None:
                rp = storedData['regionprops']
                IDs_tot = {obj.label for obj in rp}
                if wl_init:
                    if self.whitelistCheckOriginalLabels(warning=False, frame_i=frame_i):
                        IDs_tot.update(posData.whitelist.originalLabsIDs[frame_i])
                    if posData.whitelist.whitelistIDs.get(frame_i):
                        IDs_tot.update(posData.whitelist.whitelistIDs[frame_i])
                _max = max(IDs_tot, default=0)
                if _max > newID:
                    newID = _max
            else:
                break

        for y, x, manual_ID in posData.editID_info:
            if manual_ID > newID:
                newID = manual_ID
        posData.brushID = newID+1
        if return_val:
            return posData.brushID

    def curvTool_cb(self, checked):
        posData = self.data[self.pos_i]
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.curvToolButton)
            self.connectLeftClickButtons()
            self.hoverLinSpace = np.linspace(0, 1, 1000)
            self.curvPlotItem = pg.PlotDataItem(pen=self.newIDs_cpen)
            self.curvHoverPlotItem = pg.PlotDataItem(pen=self.oldIDs_cpen)
            self.curvAnchors = pg.ScatterPlotItem(
                symbol='o', size=9,
                brush=pg.mkBrush((255,0,0,50)),
                pen=pg.mkPen((255,0,0), width=2),
                hoverable=True, hoverPen=pg.mkPen((255,0,0), width=3),
                hoverBrush=pg.mkBrush((255,0,0)), tip=None
            )
            self.ax1.addItem(self.curvAnchors)
            self.ax1.addItem(self.curvPlotItem)
            self.ax1.addItem(self.curvHoverPlotItem)
            self.splineHoverON = True
            posData.curvPlotItems.append(self.curvPlotItem)
            posData.curvAnchorsItems.append(self.curvAnchors)
            posData.curvHoverItems.append(self.curvHoverPlotItem)
        else:
            self.splineHoverON = False
            self.isRightClickDragImg1 = False
            self.clearCurvItems()
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()
        
        self.showEditIDwidgets(checked)
        QTimer.singleShot(20, self.gui_refreshToolCursor)

    def Eraser_cb(self, checked):
        if checked:
            self.setDiskMask()
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )
            self.updateEraserCursor(self.xHoverImg, self.yHoverImg)
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.sender())
            c = self.defaultToolBarButtonColor
            self.brushButton.setStyleSheet(f'background-color: {c}')
            self.connectLeftClickButtons()
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )
            self.resetCursors()
            self.updateAllImages()
            
        self.showEditIDwidgets(checked)
        self.enableSizeSpinbox(checked)
        QTimer.singleShot(20, self.gui_refreshToolCursor)

    def clearCurvItems(self, removeItems=True):
        try:
            posData = self.data[self.pos_i]
            curvItems = zip(posData.curvPlotItems,
                            posData.curvAnchorsItems,
                            posData.curvHoverItems)
            for plotItem, curvAnchors, hoverItem in curvItems:
                plotItem.setData([], [])
                curvAnchors.setData([], [])
                hoverItem.setData([], [])
                if removeItems:
                    self.ax1.removeItem(plotItem)
                    self.ax1.removeItem(curvAnchors)
                    self.ax1.removeItem(hoverItem)

            if removeItems:
                posData.curvPlotItems = []
                posData.curvAnchorsItems = []
                posData.curvHoverItems = []
        except AttributeError:
            # traceback.print_exc()
            pass

    # @exec_time
    def curvToolSplineToObj(self, xxA=None, yyA=None, isRightClick=False):
        posData = self.data[self.pos_i]
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False, storeOnlyZoom=True)

        if isRightClick:
            xxS, yyS = self.curvPlotItem.getData()
            if xxS is None:
                self.setUncheckedAllButtons()
                return None, None
            self.smoothAutoContWithSpline()

        xxS, yyS = self.getClosedSplineCoords()

        if self.autoIDcheckbox.isChecked():
            self.setBrushID()
            curvToolID = posData.brushID
        else:
            curvToolID = self.editIDspinbox.value()
            posData.brushID = curvToolID

        if curvToolID <= 0:
            self.setBrushID()
            curvToolID = posData.brushID
            
        lab2D = self.get_2Dlab(posData.lab).copy()
        newIDMask = np.zeros(lab2D.shape, bool)
        rr, cc = skimage.draw.polygon(yyS, xxS, shape=lab2D.shape)
        newIDMask[rr, cc] = True
        newIDMask[lab2D!=0] = False
        lab2D[newIDMask] = curvToolID
        self.set_2Dlab(lab2D)
        self.currentLab2D = lab2D
        return newIDMask, curvToolID

    # @exec_time
    def applyEraserMask(self, mask):
        posData = self.data[self.pos_i]
        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if isZslice:
                posData.lab[self.z_lab(), mask] = 0
            else:
                posData.lab[:, mask] = 0
        else:
            posData.lab[mask] = 0

    def changeBrushID(self):
        """Function called when pressing or releasing shift
        """        
        if not self.isSegm3D:
            # Changing brush ID with shift is only for 3D segm
            return

        if not self.brushButton.isChecked():
            # Brush if not active
            return
        
        if not self.isMouseDragImg2 and not self.isMouseDragImg1:
            # Mouse is not brushing at the moment
            return

        posData = self.data[self.pos_i]
        forceNewObj = not self.isNewID
        
        if forceNewObj:
            # Shift is down --> force new object with brush
            # e.g., 24 --> 28: 
            # 24 is hovering ID that we store as self.prevBrushID
            # 24 object becomes 28 that is the new posData.brushID
            self.isNewID = True
            self.changedID = posData.brushID
            self.restoreBrushID = posData.brushID
            # Set a new ID
            self.setBrushID()
        else:
            # Shift released or hovering on ID in z+-1 
            # --> restore brush ID from before shift was pressed or from 
            # when we started brushing from outside an object 
            # but we hovered on ID in z+-1 while dragging.
            # We change the entire 28 object to 24 so before changing the 
            # brush ID back to 24 we builg the mask with 28 to change it to 24
            self.isNewID = False
            self.changedID = posData.brushID
            # Restore ID   
            posData.brushID = self.restoreBrushID
               
        brushID = posData.brushID
        brushIDmask = self.get_2Dlab(posData.lab) == self.changedID
        self.applyBrushMask(brushIDmask, brushID)
        if self.isMouseDragImg1:
            self.brushColor = self.lut[posData.brushID]/255
            self.setTempImg1Brush(True, brushIDmask, posData.brushID)

    def applyBrushMask(self, mask, ID, toLocalSlice=None):
        posData = self.data[self.pos_i]
        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            if isZslice:
                if toLocalSlice is not None:
                    toLocalSlice = (self.z_lab(), *toLocalSlice)
                    posData.lab[toLocalSlice][mask] = ID
                else:
                    posData.lab[self.z_lab()][mask] = ID
            else:
                if toLocalSlice is not None:
                    for z in range(len(posData.lab)):
                        _slice = (z, *toLocalSlice)
                        posData.lab[_slice][mask] = ID
                else:
                    posData.lab[:, mask] = ID
        else:
            if toLocalSlice is not None:
                posData.lab[toLocalSlice][mask] = ID
            else:
                posData.lab[mask] = ID

    def assignNewIDfromClickedID(
            self, clickedID: int, 
            event: QGraphicsSceneMouseEvent,
            howToPropagateZslices=None
        ):
        posData = self.data[self.pos_i]
        x, y = event.pos().x(), event.pos().y()
        newID = self.setBrushID(return_val=True)
        mapper = [(clickedID, newID)]
        self.applyEditID(
            clickedID, posData.IDs.copy(), mapper, x, y,
            howToPropagateZslices=howToPropagateZslices
        )

    def checkWarnDeletedIDwithEraser(self):
        posData = self.data[self.pos_i]
        
        for ID in self.erasedIDs:
            if ID == 0:
                continue
            if posData.rp.get_obj_from_ID(ID, warn=False) is not None:
                continue
            
            self.instructHowDeleteID()
            
            if self.isSnapshot:
                self.fixCcaDfAfterEdit('Delete ID with eraser')
                self.updateAllImages()
            else:
                self.warnEditingWithCca_df('Delete ID with eraser')
            
            return True
        
        return False

    def initFloodMaskImage(self):
        posData = self.data[self.pos_i]
        self.flood_img = posData.img_data[posData.frame_i]
        if not self.isSegm3D and posData.SizeZ > 1:
            self.flood_img = self.get_2Dimg_from_3D(self.flood_img)
            return

    def getMagicWandFloodTolerance(self):
        tol_perc = self.wandControlsToolbar.toleranceSpinbox.value()
        if tol_perc == 0:
            return
        
        posData = self.data[self.pos_i]
        _min, _max = posData.img_data_min_max
        tol_fraction = tol_perc/100
        tol = (_max - _min) * tol_fraction
        
        return tol

    def initTempLayerBrush(self, ID, ax=0):
        posData = self.data[self.pos_i]
        self.hideItemsHoverBrush(ID=ID, force=True)
        Y, X = self.img1.image.shape[:2]
        tempImage = np.zeros((Y, X), dtype=np.uint32)
        kwargs = {}
        if self.isContoursChecked(ax):
            # Keep the currently edited object visible while painting.
            rp_2D = self.get2DRP()
            obj = rp_2D.get_obj_from_ID(ID, warn=False) # new IDs would spam errors
            if (
                obj is not None
                and hasattr(obj, 'slice')
                and hasattr(obj, 'image')
                and len(obj.slice) == 2
                and obj.image.ndim == 2
            ):
                tempImage[obj.slice][obj.image] = ID
            else:
                tempImage[self.currentLab2D == ID] = ID
            self.brushImage = tempImage.copy()
            self.brushContourImage = np.zeros((Y, X, 4), dtype=np.uint8)
            color = self.imgGrad.contoursColorButton.color()
            self.brushContoursRgba = color.getRgb()
            opacity = 1.0
        else:
            opacity = self.imgGrad.labelsAlphaSlider.value()
            color = self.lut[ID]
            lut = np.zeros((2, 4), dtype=np.uint8)
            lut[1,-1] = 255
            lut[1,:-1] = color
            self.tempLayerImg1.setLookupTable(lut)
            # The linked right image layer (painting on the right image is
            # forwarded to the left image) does not inherit the lut
            self.tempLayerRightImage.setLookupTable(lut)
            kwargs['levels'] = self.getTempLayerBrushLevels(ID)
        self.tempLayerImg1.setOpacity(opacity)
        self.tempLayerImg1.setImage(tempImage, force_set_linked=True, **kwargs)

    def getTempLayerBrushLevels(self, ID):
        """Levels of the brush temp layer in overlay masks mode (values 0 and
        `ID`, 2-entry lut). Explicit levels because auto-levels are computed
        on a subsample of the pixels and can miss a small brush stroke on
        large images (the stroke would be mapped to the transparent color)."""
        return (0, max(int(ID), 1))

    def _setTempImageBrushContour(self):
        pass

    def setTempBrushMaskFromWand(self, flood_mask, init=False):
        if not np.any(flood_mask):
            return
        
        posData = self.data[self.pos_i]
        mask = np.logical_or(
            flood_mask,
            posData.lab==posData.brushID
        )
        if mask.ndim == 3:
            z_slice = self.zSliceScrollBar.sliderPosition()
            mask = mask[z_slice]
            
        self.setTempImg1Brush(init, mask, posData.brushID)

    # @exec_time
    def setTempImg1Brush(self, init: bool, mask, ID, toLocalSlice=None, ax=0):
        posData = self.data[self.pos_i]
        if init:
            self.initTempLayerBrush(ID, ax=ax)
        
        if self.isContoursChecked(ax):
            brushImage = self.brushImage
        else:
            brushImage = self.tempLayerImg1.image
            
        if toLocalSlice is None:
            brushImage[mask] = ID
        else:
            brushImage[toLocalSlice][mask] = ID
        
        if self.isContoursChecked(ax):
            brushMask = np.ascontiguousarray((brushImage > 0), dtype=np.uint8)

            objContour = core.get_obj_contours(
                obj_image=brushMask, obj_bbox=None, all_external=True, local=True
            )
            self.brushContourImage[:] = 0
            img = self.brushContourImage
            color = self.brushContoursRgba
            cv2.drawContours(img, objContour, -1, color, 1)
            self.tempLayerImg1.setImage(img, force_set_linked=True)
        else:
            self.tempLayerImg1.setImage(
                brushImage, force_set_linked=True,
                levels=self.getTempLayerBrushLevels(ID)
            )

    def clearObjFromMask(self, image, mask, toLocalSlice=None):
        if mask is None:
            return image

        if toLocalSlice is None:
            image[mask] = 0
        else:
            image[toLocalSlice][mask] = 0
        
        return image

    # @exec_time
    def setTempImg1Eraser(self, mask, init=False, toLocalSlice=None, ax=0):
        if init:
            self.erasedLab = np.zeros_like(self.currentLab2D)
        
        if ax == 1 and not self.labelsGrad.showRightImgAction.isChecked():
            return
        
        if self.isContoursChecked(ax):
            self.clearObjFromMask(
                self.contoursImage, mask, toLocalSlice=toLocalSlice
            )
            thickness = self.contLineWeight
            color = self.contLineColor
            for erasedID in np.unique(self.erasedLab):
                if erasedID == 0:
                    continue
                erasedMask = np.ascontiguousarray(
                    (self.erasedLab == erasedID), dtype=np.uint8
                )
                contours = core.get_obj_contours(
                    obj_image=erasedMask, obj_bbox=None,
                    all_external=True, local=True
                )
                cv2.drawContours(self.contoursImage, contours, -1, color, thickness)
            imageItem = self.getContoursImageItem(ax)
            if imageItem is not None:
                imageItem.setImage(self.contoursImage)
        elif self.isOverlaySegmMaskChecked(ax):
            labelsImage = self.getLabelsLayerImage(ax=ax)
            self.clearObjFromMask(labelsImage, mask, toLocalSlice=toLocalSlice)           
            if ax == 0:
                self.labelsLayerImg1.setImage(
                    self.labelsLayerImg1.image, autoLevels=False
                )
            else:
                self.labelsLayerRightImg.setImage(
                    self.labelsLayerRightImg.image, autoLevels=False
                )

    def _setTempImgExpandLabelSegmMasks(self, prevCoords, expandedObjCoords, ax=0):
        # Remove previous overlaid mask
        labelsImage = self.getLabelsLayerImage(ax=ax)
        labelsImage[prevCoords] = 0
        
        # Overlay new expanded mask
        labelsImage[expandedObjCoords] = self.expandingID

        if ax == 0:
            self.labelsLayerImg1.setImage(
                self.labelsLayerImg1.image, autoLevels=False)
        else:
            self.labelsLayerRightImg.setImage(
                self.labelsLayerRightImg.image, autoLevels=False)

    def _setTempImgExpandLabelContours(self, prevCoords, ax=0):
        self.contoursImage[prevCoords] = [0,0,0,0]
        expandMask = np.ascontiguousarray(
            (self.currentLab2D == self.expandingID), dtype=np.uint8
        )
        contours = core.get_obj_contours(
            obj_image=expandMask, obj_bbox=None, all_external=True, local=True
        )
        imageItem = self.getContoursImageItem(ax, force=True)
        if imageItem is not None:
            thickness = self.contLineWeight
            color = self.contLineColor
            cv2.drawContours(self.contoursImage, contours, -1, color, thickness)
            imageItem.setImage(self.contoursImage)

    def setTempImgExpandLabel(self, prevCoords, expandedObjCoords, ax=0):
        if self.isOverlaySegmMaskChecked(ax):
            self._setTempImgExpandLabelSegmMasks(
                prevCoords, expandedObjCoords, ax=ax
            )
        else:
            self._setTempImgExpandLabelContours(prevCoords, ax=ax)

    def setTempImg1MoveLabel(self, ax=0):        
        if self.isContoursChecked(ax):
            moveMask = np.ascontiguousarray(
                (self.currentLab2D == self.movingID), dtype=np.uint8
            )
            contours = core.get_obj_contours(
                obj_image=moveMask, obj_bbox=None, all_external=True, local=True
            )
            imageItem = self.getContoursImageItem(ax)
            if imageItem is not None:
                thickness = self.contLineWeight
                color = self.contLineColor
                cv2.drawContours(self.contoursImage, contours, -1, color, thickness)
                imageItem.setImage(self.contoursImage)
        elif self.isOverlaySegmMaskChecked(ax):
            if ax == 0:
                self.labelsLayerImg1.setImage(self.currentLab2D, autoLevels=False)
                self.highLightIDLayerImg1.image[:] = 0
                mask = self.currentLab2D==self.movingID
                self.highLightIDLayerImg1.image[mask] = self.movingID
                highlightedImage = self.highLightIDLayerImg1.image
                self.highLightIDLayerImg1.setImage(highlightedImage)
            else:
                self.labelsLayerRightImg.setImage(
                    self.currentLab2D, autoLevels=False
                )
                self.highLightIDLayerRightImage.image[:] = 0
                mask = self.currentLab2D==self.movingID
                self.highLightIDLayerRightImage.image[mask] = self.movingID
                highlightedImage = self.highLightIDLayerRightImage.image
                self.highLightIDLayerRightImage.setImage(highlightedImage)

    def clearTempBrushImage(self, forceClearLinked=True):
        if not hasattr(self, 'tempLayerImg1'):
            return
        
        self.tempLayerImg1.setImage(
            self.emptyLab, force_set_linked=forceClearLinked
        )
        
        try:
            self.brushContourImage[:] = 0
        except Exception as err:
            pass
        
        try:
            self.brushImage[:] = 0
        except Exception as err:
            pass

    def hoverEventDrawSpline(self, event):
        x, y = event.pos()
        xx, yy = self.curvAnchors.getData()
        hoverAnchors = self.curvAnchors.pointsAt(event.pos())
        per = False
        # If we are hovering the starting point we generate
        # a closed spline
        if len(xx) < 2:
            return 
        
        if len(hoverAnchors)>0:
            xA_hover, yA_hover = hoverAnchors[0].pos()
            if xx[0]==xA_hover and yy[0]==yA_hover:
                per=True
        if per:
            # Append start coords and close spline
            xx = np.r_[xx, xx[0]]
            yy = np.r_[yy, yy[0]]
            xi, yi = self.getSpline(xx, yy, per=per)
            # self.curvPlotItem.setData([], [])
        else:
            # Append mouse coords
            xx = np.r_[xx, x]
            yy = np.r_[yy, y]
            xi, yi = self.getSpline(xx, yy, per=per)
        self.curvHoverPlotItem.setData(xi, yi)


class ManualBackgroundGuiElements:
    """Manual background object tool (shortcut G)."""

    def addManualBackgroundItems(self):
        self.manualBackgroundObjItem.addToPlotItem()
        self.ax1.addItem(self.manualBackgroundImageItem)

    def removeManualBackgroundItems(self):
        self.manualBackgroundObjItem.removeFromPlotItem()
        self.ax1.removeItem(self.manualBackgroundImageItem)

    def resetManualBackgroundSpinboxID(self):
        if not self.manualBackgroundButton.isChecked():
            self.manualBackgroundObj = None
            return
        
        posData = self.data[self.pos_i]
        minID = min(posData.IDs, default=0)
        self.manualBackgroundToolbar.spinboxID.setValue(minID)

    def initManualBackgroundObject(self, ID=None):
        if not self.manualBackgroundButton.isChecked():
            self.manualBackgroundObj = None
            return
        
        if ID is None:
            ID = self.manualBackgroundToolbar.spinboxID.value()
        
        posData = self.data[self.pos_i]
        if ID not in posData.IDs:
            self.manualBackgroundObj = None
            self.manualBackgroundToolbar.showWarning(
                f'The ID {ID} does not exist'
            )
            self.manualBackgroundObjItem.clear()
            return
        
        display_rp = self.get2DRP()
        self.manualBackgroundObj = display_rp.get_obj_from_ID(ID)
        
        if self.manualBackgroundObj is None:
            self.logger.warning(f"Could not find object with ID {ID} in current frame.")
            return

        self.manualBackgroundToolbar.clearInfoText()
        self.manualBackgroundObj.contour_local = self.getObjContours(
            self.manualBackgroundObj, local=True
        )

    def clearManualBackgroundAnnotations(self):
        try:
            for textItem in self.manualBackgroundTextItems.values():
                textItem.setText('')
        except Exception as error:
            pass

    def manualBackground_cb(self, checked):
        if checked:
            posData = self.data[self.pos_i]
            minID = min(posData.IDs, default=0)
            if minID == self.manualBackgroundToolbar.spinboxID.value():
                self.initManualBackgroundObject()
            else:
                self.manualBackgroundToolbar.spinboxID.setValue(minID)
            # self.initManualBackgroundObject()
            # self.initManualBackgroundImage()
            self.addManualBackgroundItems()
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.manualBackgroundButton)
            self.connectLeftClickButtons()
            self.updateAllImages()
        else:
            self.removeManualTrackingItems()
            self.clearGhost()
            self.clearManualBackgroundAnnotations()
        self.manualBackgroundToolbar.setVisible(checked)
        QTimer.singleShot(20, self.gui_refreshToolCursor)

    def resetManualBackgroundItems(self):
        self.initManualBackgroundImage()
        self.resetManualBackgroundSpinboxID()
        self.drawManualTrackingGhost(self.xHoverImg, self.yHoverImg)
        self.drawManualBackgroundObj(self.xHoverImg, self.yHoverImg)

    def initManualBackgroundImage(self):
        posData = self.data[self.pos_i]
        if hasattr(posData, 'lab'):
            Y, X = posData.lab.shape[-2:]
        else:
            Y, X = posData.img_data.shape[-2:]
        if not hasattr(self, 'manualBackgroundTextItems'):
            self.manualBackgroundTextItems = {}
        posData.manualBackgroundImage = np.zeros((Y, X, 4), dtype=np.uint8)
        if posData.manualBackgroundLab is None:
            posData.manualBackgroundLab = np.zeros((Y, X), dtype=np.uint32)

    def _drawManualBackgroundObjContour(self, x, y):
        if self.manualBackgroundObj is None:
            return
        
        ID = self.manualBackgroundObj.label
        yc, xc = self.manualBackgroundObj.local_centroid
        Dx = x-xc
        Dy = y-yc
        xx = self.manualBackgroundObj.contour_local[:,0] + Dx
        yy = self.manualBackgroundObj.contour_local[:,1] + Dy
        self.manualBackgroundObjItem.setData(
            xx, yy, fontSize=self.fontSize, ID=ID, y_cursor=y, x_cursor=x
        )

    def drawManualBackgroundObj(self, x, y):
        if x is None or y is None:
            self.clearGhost()
            return
        
        self._drawManualBackgroundObjContour(x, y)

    def setManualBackgroundImage(self):
        if not self.manualBackgroundButton.isChecked():
            return
        
        posData = self.data[self.pos_i]
        if not hasattr(posData, 'manualBackgroundImage'):
            self.initManualBackgroundImage()
        
        contours = []
        for obj in self._acdcRegionProps(
            posData.manualBackgroundLab, precache_centroids=False
        ):
            obj_contours = self.getObjContours(
                obj,
                all_external=True,
                include_internal=self.showAllContoursToggle.isChecked()
            )
            contours.extend(obj_contours)
            textItem = self.manualBackgroundTextItems[obj.label]
            textItem.setText(f'{obj.label}')
            self.ax1.addItem(textItem)
            yc, xc = obj.centroid
            textItem.setPos(xc, yc)
        
        cv2.drawContours(
            posData.manualBackgroundImage, contours, -1, (255, 0, 0, 200), 1
        )
        self.manualBackgroundImageItem.setImage(posData.manualBackgroundImage)

    def setManualBackgrounNextID(self):
        posData = self.data[self.pos_i]
        currentID = self.manualBackgroundObj.label
        if currentID in posData.rp.ID_to_idx:
            idx = posData.rp.ID_to_idx[currentID]
        else:
            return
        next_idx = idx + 1
        if next_idx >= len(posData.IDs):
            return
        next_ID = posData.IDs[next_idx]
        self.manualBackgroundToolbar.spinboxID.setValue(next_ID)

    def clearManualBackgroundObject(self, ID):
        posData = self.data[self.pos_i]
        mask = posData.manualBackgroundLab==ID
        posData.manualBackgroundImage[mask, :] = 0
        posData.manualBackgroundLab[mask] = 0

    def addManualBackgroundObject(self, x, y):
        posData = self.data[self.pos_i]
        
        if not hasattr(self, 'manualBackgroundObj'):
            self.initManualBackgroundObject()
        
        Y, X = self.currentLab2D.shape
        ymin, xmin, ymax, xmax = self.manualBackgroundObj.bbox
        width, height = xmax-xmin, ymax-ymin
        yc, xc = self.manualBackgroundObj.local_centroid
        xstart, ystart = round(x-xc), round(y-yc)
        xstart = xstart if xstart >= 0 else 0
        ystart = ystart if ystart >= 0 else 0
        
        xend = xstart+width
        yend = ystart+height
        xend = xend if xend <= X else X
        yend = yend if yend <= Y else Y
        
        width = xend-xstart
        height = yend-ystart
        
        obj_image = self.manualBackgroundObj.image[:height, :width]
        obj_slice = (slice(ystart, yend), slice(xstart, xend))
        ID = self.manualBackgroundObj.label
        self.clearManualBackgroundObject(ID)
        posData.manualBackgroundLab[obj_slice][obj_image] = ID
        
        if ID in self.manualBackgroundTextItems:
            self.manualBackgroundTextItems[ID].setPos(x, y)
            return
        
        textItem = pg.TextItem(
            text=str(ID), color='r', anchor=(0.5, 0.5)
        )
        textItem.setFont(gui_main_constants.font_13px)
        textItem.setPos(x, y)
        self.manualBackgroundTextItems[ID] = textItem
        
        self.ax1.addItem(textItem)

    def setManualBackgroundLab(self, load_from_store=False, debug=True):
        posData = self.data[self.pos_i]
        if posData.manualBackgroundLab is None:
            self.initManualBackgroundImage()
        
        for obj in self._acdcRegionProps(
            posData.manualBackgroundLab, precache_centroids=False
        ):
            textItem = pg.TextItem(text='', color='r', anchor=(0.5, 0.5))
            if obj.label in self.manualBackgroundTextItems:
                continue
            self.manualBackgroundTextItems[obj.label] = textItem
