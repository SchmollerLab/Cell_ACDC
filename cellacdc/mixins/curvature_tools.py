"""Qt view adapter for curvature and spline tools."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import skimage.draw
import skimage.measure

from .brush_tools import BrushTools
from .undo_redo import UndoRedo

class CurvatureTools(BrushTools, UndoRedo):
    """Extracted from guiWin."""

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

    def curvToolSplineToObj(self, xxA=None, yyA=None, isRightClick=False):
        posData = self.data[self.pos_i]
        # Store undo state before modifying stuff
        self.storeUndoRedoStates(False, storeOnlyZoom=True)

        if isRightClick:
            xxS, yyS = self.curvPlotItem.getData()
            if xxS is None:
                self.setUncheckedAllButtons()
                return
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

    def smoothAutoContWithSpline(self, n=3):
        try:
            xx, yy = self.curvHoverPlotItem.getData()
            if xx is None or yy is None:
                return
            # Downsample by taking every nth coord
            xxA, yyA = xx[::n], yy[::n]
            rr, cc = skimage.draw.polygon(yyA, xxA)
            self.autoContObjMask[rr, cc] = 1
            rp = skimage.measure.regionprops(self.autoContObjMask)
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
