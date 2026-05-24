"""Qt view adapter for curvature and spline tools."""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
import skimage.draw
import skimage.measure

from cellacdc.viewmodels.curvature_viewmodel import CurvatureViewModel


class CurvatureToolsView:
    """Qt-facing adapter around curvature tool contracts."""

    def __init__(self, host, view_model: CurvatureViewModel):
        object.__setattr__(self, 'host', host)
        object.__setattr__(self, 'view_model', view_model)

    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host', 'view_model'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    # @exec_time
    def getPolygonBrush(self, yxc2, Y, X):
        # see https://en.wikipedia.org/wiki/Tangent_lines_to_circles
        y1, x1 = self.yPressAx2, self.xPressAx2
        y2, x2 = yxc2
        rr_poly, cc_poly = self.view_model.tangent_brush_polygon(
            (y1, x1),
            (y2, x2),
            self.brushSizeSpinbox.value(),
            (Y, X),
        )

        self.yPressAx2, self.xPressAx2 = y2, x2
        return rr_poly, cc_poly

    def get_dir_coords(self, alfa_dir, yd, xd, shape, connectivity=1):
        return self.view_model.directional_coords(
            alfa_dir,
            yd,
            xd,
            shape,
            connectivity=connectivity,
        )

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

    def getClosedSplineCoords(self):
        xxS, yyS = self.curvPlotItem.getData()
        xx, yy = self.curvAnchors.getData()
        return self.view_model.closed_spline_coords(
            xxS,
            yyS,
            anchor_xx=xx,
            anchor_yy=yy,
            predictor=self.splineToObjModel,
        )


    def getSpline(self, xx, yy, resolutionSpace=None, per=False, appendFirst=False):
        if resolutionSpace is None:
            resolutionSpace = self.hoverLinSpace
        return self.view_model.spline_coords(
            xx,
            yy,
            resolution_space=resolutionSpace,
            per=per,
            append_first=appendFirst,
        )

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
        result = self.view_model.paint_spline_to_labels(
            lab2D,
            xxS,
            yyS,
            curvToolID,
            empty_only=True,
        )
        lab2D = result.labels_2d
        self.set_2Dlab(lab2D)
        self.currentLab2D = lab2D

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
