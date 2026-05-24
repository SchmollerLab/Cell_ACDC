"""Qt view adapter for canvas hover and cursor interactions."""

from __future__ import annotations

import pyqtgraph as pg
from typing import Any
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication

from cellacdc import html_utils, widgets


class CanvasHoverMixin:
    """Qt-facing adapter around canvas hover workflows."""

    """Headless decisions for hover and cursor state."""

    def _cursor_flags(self, modifiers, event):
        return self.cursor_flags(
            is_exit=event.isExit(),
            no_modifier=modifiers == Qt.NoModifier,
            shift=modifiers == Qt.ShiftModifier,
            ctrl=modifiers == Qt.ControlModifier,
            alt=modifiers == Qt.AltModifier,
            brush_checked=self.brushButton.isChecked(),
            eraser_checked=self.eraserButton.isChecked(),
            add_deleted_polyline_checked=(self.addDelPolyLineRoiButton.isChecked()),
            label_roi_checked=self.labelRoiButton.isChecked(),
            label_roi_circular_checked=(self.labelRoiIsCircularRadioButton.isChecked()),
            wand_checked=self.wandToolButton.isChecked(),
            move_label_checked=self.moveLabelToolButton.isChecked(),
            expand_label_checked=self.expandLabelToolButton.isChecked(),
            curvature_checked=self.curvToolButton.isChecked(),
            keep_ids_checked=self.keepIDsButton.isChecked(),
            custom_annotation_available=self.customAnnotButton is not None,
            manual_tracking_checked=self.manualTrackingButton.isChecked(),
            manual_background_checked=self.manualBackgroundButton.isChecked(),
            zoom_rect_checked=self.zoomRectButton.isChecked(),
            edit_id_checked=self.editIDbutton.isChecked(),
            magic_prompts_checked=self.magicPromptsToolButton.isChecked(),
            points_layer_checked=self.togglePointsLayerAction.isChecked(),
            add_points_by_clicking_active=(
                self.buttonAddPointsByClickingActive() is not None
            ),
        )

    def cursor_flags(
        self,
        *,
        is_exit: bool,
        no_modifier: bool,
        shift: bool,
        ctrl: bool,
        alt: bool,
        brush_checked: bool,
        eraser_checked: bool,
        add_deleted_polyline_checked: bool,
        label_roi_checked: bool,
        label_roi_circular_checked: bool,
        wand_checked: bool,
        move_label_checked: bool,
        expand_label_checked: bool,
        curvature_checked: bool,
        keep_ids_checked: bool,
        custom_annotation_available: bool,
        manual_tracking_checked: bool,
        manual_background_checked: bool,
        zoom_rect_checked: bool,
        edit_id_checked: bool,
        magic_prompts_checked: bool,
        points_layer_checked: bool,
        add_points_by_clicking_active: bool,
    ) -> dict[str, bool]:
        return {
            "setBrushCursor": (
                brush_checked and not is_exit and (no_modifier or shift or ctrl)
            ),
            "setEraserCursor": eraser_checked and not is_exit and no_modifier,
            "setAddDelPolyLineCursor": (
                add_deleted_polyline_checked and not is_exit and no_modifier
            ),
            "setLabelRoiCircCursor": (
                label_roi_checked
                and not is_exit
                and (no_modifier or shift or ctrl)
                and label_roi_circular_checked
            ),
            "setWandCursor": wand_checked and not is_exit and no_modifier,
            "setLabelRoiCursor": label_roi_checked and not is_exit and no_modifier,
            "setMoveLabelCursor": move_label_checked and not is_exit and no_modifier,
            "setExpandLabelCursor": (
                expand_label_checked and not is_exit and no_modifier
            ),
            "setCurvCursor": curvature_checked and not is_exit and no_modifier,
            "setKeepObjCursor": keep_ids_checked and not is_exit and no_modifier,
            "setCustomAnnotCursor": (
                custom_annotation_available and not is_exit and no_modifier
            ),
            "setManualTrackingCursor": (
                manual_tracking_checked and not is_exit and no_modifier
            ),
            "setManualBackgroundCursor": (
                manual_background_checked and not is_exit and no_modifier
            ),
            "setAddPointCursor": (
                (points_layer_checked or magic_prompts_checked)
                and add_points_by_clicking_active
                and not is_exit
                and no_modifier
            ),
            "setZoomRectCursor": zoom_rect_checked and not is_exit and no_modifier,
            "setEditIDCursor": edit_id_checked and not is_exit,
            "setPanImageCursor": alt and not is_exit,
        }

    def drawTempMergeObjsLine(self, event, posData, modifiers):
        if self.clickObjYc is None:
            return
        modifier = modifiers == Qt.ShiftModifier
        x, y = event.pos()
        y2, x2 = y, x
        xdata, ydata = int(x), int(y)
        y1, x1 = self.clickObjYc, self.clickObjXc
        ID = self.get_2Dlab(posData.lab)[ydata, xdata]
        if ID != 0:
            obj_idx = posData.IDs_idxs[ID]
            obj = posData.rp[obj_idx]
            y2, x2 = self.getObjCentroid(obj.centroid)

        if modifier and ID > 0:
            self.mergeObjsTempLine.addPoint(x2, y2)
        elif not modifier:
            self.mergeObjsTempLine.setData([x1, x2], [y1, y2])

    def drawTempMothBudLine(self, event, posData):
        x, y = event.pos()
        y2, x2 = y, x
        xdata, ydata = int(x), int(y)
        y1, x1 = self.yClickBud, self.xClickBud
        ID = self.get_2Dlab(posData.lab)[ydata, xdata]
        if ID == 0:
            self.BudMothTempLine.setData([x1, x2], [y1, y2])
        else:
            obj_idx = posData.IDs_idxs[ID]
            obj = posData.rp[obj_idx]
            y2, x2 = self.getObjCentroid(obj.centroid)
            self.BudMothTempLine.setData([x1, x2], [y1, y2])

    def drawTempRulerLine(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        x, y = event.pos()
        x1, y1 = int(x), int(y)
        xxRA, yyRA = self.ax1_rulerAnchorsItem.getData()
        x0, y0 = xxRA[0], yyRA[0]
        if ctrl:
            x1, y1 = transformation.snap_xy_to_closest_angle(x0, y0, x1, y1)
        self.ax1_rulerPlotItem.setData([x0, x1], [y0, y1])

    def gui_add_ax_cursors(self):
        try:
            self.ax1.removeItem(self.ax1_cursor)
            self.ax2.removeItem(self.ax2_cursor)
        except Exception:
            pass

        self.ax2_cursor = pg.ScatterPlotItem(
            symbol="+",
            pxMode=True,
            pen=pg.mkPen("k", width=1),
            brush=pg.mkBrush("w"),
            size=16,
            tip=None,
        )
        self.ax2.addItem(self.ax2_cursor)

        self.ax1_cursor = pg.ScatterPlotItem(
            symbol="+",
            pxMode=True,
            pen=pg.mkPen("k", width=1),
            brush=pg.mkBrush("w"),
            size=16,
            tip=None,
        )
        self.ax1.addItem(self.ax1_cursor)

    def gui_hoverEventImg1(self, event, isHoverImg1=True):
        try:
            posData = self.data[self.pos_i]
        except AttributeError:
            return

        # Update x, y, value label bottom right
        if not event.isExit():
            self.xHoverImg, self.yHoverImg = event.pos()
        else:
            self.xHoverImg, self.yHoverImg = None, None

        if event.isExit():
            self.resetCursor()

        if not event.isExit() and self.slideshowWin is not None:
            self.slideshowWin.setMirroredCursorPos(*event.pos())

        # Alt key was released --> restore cursor
        modifiers = QGuiApplication.keyboardModifiers()
        cursorsInfo = self.gui_setCursor(modifiers, event)
        self.highlightHoverLostObj(modifiers, event)

        drawRulerLine = (
            (self.rulerButton.isChecked() or self.addDelPolyLineRoiButton.isChecked())
            and self.tempSegmentON
            and not event.isExit()
        )
        if drawRulerLine:
            self.drawTempRulerLine(event)

        if not event.isExit():
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            _img = self.img1.image
            Y, X = _img.shape[:2]
            if xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y:
                ID = self.currentLab2D[ydata, xdata]
                self.updatePropsWidget(ID, fromHover=True)
                activeToolButton = self.getActiveToolButton()
                hoverText = self.hoverValuesFormatted(
                    xdata, ydata, activeToolButton, isHoverImg1
                )
                self.checkHighlightScaleBar(x, y, activeToolButton)
                self.checkHighlightTimestamp(x, y, activeToolButton)
                self.wcLabel.setText(hoverText)
        else:
            self.clickedOnBud = False
            self.BudMothTempLine.setData([], [])
            self.wcLabel.setText("")

        if cursorsInfo["setKeepObjCursor"]:
            x, y = event.pos()
            self.highlightHoverIDsKeptObj(x, y)

        if cursorsInfo["setManualTrackingCursor"]:
            x, y = event.pos()
            # self.highlightHoverID(x, y)
            self.drawManualTrackingGhost(x, y)

        if cursorsInfo["setManualBackgroundCursor"]:
            x, y = event.pos()
            # self.highlightHoverID(x, y)
            self.drawManualBackgroundObj(x, y)

        if (
            not cursorsInfo["setManualTrackingCursor"]
            and not cursorsInfo["setManualBackgroundCursor"]
        ):
            self.clearGhost()

        setMoveLabelCursor = cursorsInfo["setMoveLabelCursor"]
        setExpandLabelCursor = cursorsInfo["setExpandLabelCursor"]
        if setMoveLabelCursor or setExpandLabelCursor:
            x, y = event.pos()
            self.updateHoverLabelCursor(x, y)

        # Draw eraser circle
        if cursorsInfo["setEraserCursor"]:
            x, y = event.pos()
            self.updateEraserCursor(x, y, isHoverImg1=isHoverImg1)
            self.hideItemsHoverBrush(xy=(x, y))
        elif self.eraserButton.isChecked() and not event.isExit():
            if self.xyOnCtrlPressedFirstTime is not None:
                self.updateEraserCursor(
                    x,
                    y,
                    xyLocked=self.xyOnCtrlPressedFirstTime,
                    isHoverImg1=isHoverImg1,
                )
                self.hideItemsHoverBrush(xy=(x, y))
        else:
            eraserCursors = (
                self.ax1_EraserCircle,
                self.ax2_EraserCircle,
                self.ax1_EraserX,
                self.ax2_EraserX,
            )
            self.setHoverToolSymbolData([], [], eraserCursors)

        # Draw Brush circle
        if cursorsInfo["setBrushCursor"]:
            x, y = event.pos()
            self.updateBrushCursor(x, y, isHoverImg1=isHoverImg1)
            self.hideItemsHoverBrush(xy=(x, y))
        elif cursorsInfo["setAddPointCursor"]:
            x, y = event.pos()
            self.setHoverCircleAddPoint(x, y)
        else:
            self.setHoverToolSymbolData(
                [],
                [],
                (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )

        # Draw label ROi circular cursor
        setLabelRoiCircCursor = cursorsInfo["setLabelRoiCircCursor"]
        if setLabelRoiCircCursor:
            x, y = event.pos()
        else:
            x, y = None, None
        self.updateLabelRoiCircularCursor(x, y, setLabelRoiCircCursor)

        drawMothBudLine = (
            self.assignBudMothButton.isChecked()
            and self.clickedOnBud
            and not event.isExit()
        )
        if drawMothBudLine:
            self.drawTempMothBudLine(event, posData)

        drawMergeObjsLine = self.mergeIDsButton.isChecked() and not event.isExit()
        if drawMergeObjsLine:
            self.drawTempMergeObjsLine(event, posData, modifiers)

        # Temporarily draw spline curve
        # see https://stackoverflow.com/questions/33962717/interpolating-a-closed-curve-using-scipy
        drawSpline = (
            self.curvToolButton.isChecked()
            and self.splineHoverON
            and not event.isExit()
        )
        if drawSpline:
            self.hoverEventDrawSpline(event)

        setMirroredCursor = (
            self.app.overrideCursor() is None
            and not event.isExit()
            and isHoverImg1
            and self.showMirroredCursorAction.isChecked()
        )
        if setMirroredCursor:
            x, y = event.pos()
            self.ax2_cursor.setData([x], [y])
        else:
            self.ax2_cursor.setData([], [])

        return cursorsInfo

    def gui_hoverEventImg2(self, event):
        try:
            self.data[self.pos_i]
        except AttributeError:
            return

        if not event.isExit():
            self.xHoverImg, self.yHoverImg = event.pos()
        else:
            self.xHoverImg, self.yHoverImg = None, None

        # Cursor left image --> restore cursor
        if event.isExit() and self.app.overrideCursor() is not None:
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()

        # Alt key was released --> restore cursor
        modifiers = QGuiApplication.keyboardModifiers()
        noModifier = modifiers == Qt.NoModifier
        shift = modifiers == Qt.ShiftModifier
        ctrl = modifiers == Qt.ControlModifier
        if self.app.overrideCursor() == Qt.SizeAllCursor and noModifier:
            self.app.restoreOverrideCursor()

        setBrushCursor = (
            self.brushButton.isChecked()
            and not event.isExit()
            and (noModifier or shift or ctrl)
        )
        setEraserCursor = (
            self.eraserButton.isChecked() and not event.isExit() and noModifier
        )
        setLabelRoiCircCursor = (
            self.labelRoiButton.isChecked()
            and not event.isExit()
            and (noModifier or shift or ctrl)
            and self.labelRoiIsCircularRadioButton.isChecked()
        )
        if setBrushCursor or setEraserCursor or setLabelRoiCircCursor:
            self.app.setOverrideCursor(Qt.CrossCursor)

        setMoveLabelCursor = (
            self.moveLabelToolButton.isChecked() and not event.isExit() and noModifier
        )

        setExpandLabelCursor = (
            self.expandLabelToolButton.isChecked() and not event.isExit() and noModifier
        )

        # Cursor is moving on image while Alt key is pressed --> pan cursor
        alt = QGuiApplication.keyboardModifiers() == Qt.AltModifier
        setPanImageCursor = alt and not event.isExit()
        if setPanImageCursor and self.app.overrideCursor() is None:
            self.app.setOverrideCursor(Qt.SizeAllCursor)

        setKeepObjCursor = (
            self.keepIDsButton.isChecked() and not event.isExit() and noModifier
        )
        if setKeepObjCursor and self.app.overrideCursor() is None:
            self.app.setOverrideCursor(Qt.PointingHandCursor)

        # Update x, y, value label bottom right
        if not event.isExit():
            x, y = event.pos()
            _xdata, _ydata = int(x), int(y)
            _img = self.currentLab2D
            Y, X = _img.shape
            # hoverText = self.hoverValuesFormatted(xdata, ydata)
            # self.wcLabel.setText(hoverText)
        else:
            if self.eraserButton.isChecked() or self.brushButton.isChecked():
                self.gui_mouseReleaseEventImg2(event)
            self.wcLabel.setText("")

        if setMoveLabelCursor or setExpandLabelCursor:
            x, y = event.pos()
            self.updateHoverLabelCursor(x, y)

        if setKeepObjCursor:
            x, y = event.pos()
            self.highlightHoverIDsKeptObj(x, y)

        # Draw eraser circle
        if setEraserCursor:
            x, y = event.pos()
            self.updateEraserCursor(x, y, isHoverImg1=False)
        else:
            self.setHoverToolSymbolData(
                [],
                [],
                (
                    self.ax1_EraserCircle,
                    self.ax2_EraserCircle,
                    self.ax1_EraserX,
                    self.ax2_EraserX,
                ),
            )

        # Draw Brush circle
        if setBrushCursor:
            x, y = event.pos()
            self.updateBrushCursor(x, y, isHoverImg1=False)
        else:
            self.setHoverToolSymbolData(
                [],
                [],
                (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )

        # Draw label ROi circular cursor
        if setLabelRoiCircCursor:
            x, y = event.pos()
        else:
            x, y = None, None
        self.updateLabelRoiCircularCursor(x, y, setLabelRoiCircCursor)

    def gui_hoverEventRightImage(self, event):
        try:
            self.data[self.pos_i]
        except AttributeError:
            return

        if event.isExit():
            self.resetCursors()

        self.gui_hoverEventImg1(event, isHoverImg1=False)
        setMirroredCursor = (
            self.app.overrideCursor() is None
            and not event.isExit()
            and self.showMirroredCursorAction.isChecked()
        )
        if setMirroredCursor:
            x, y = event.pos()
            self.ax1_cursor.setData([x], [y])

    def gui_setCursor(self, modifiers, event):
        noModifier = modifiers == Qt.NoModifier
        shift = modifiers == Qt.ShiftModifier
        ctrl = modifiers == Qt.ControlModifier
        alt = modifiers == Qt.AltModifier

        # Alt key was released --> restore cursor
        if self.app.overrideCursor() == Qt.SizeAllCursor and noModifier:
            self.app.restoreOverrideCursor()

        setBrushCursor = (
            self.brushButton.isChecked()
            and not event.isExit()
            and (noModifier or shift or ctrl)
        )
        setEraserCursor = (
            self.eraserButton.isChecked() and not event.isExit() and noModifier
        )
        setAddDelPolyLineCursor = (
            self.addDelPolyLineRoiButton.isChecked()
            and not event.isExit()
            and noModifier
        )
        setLabelRoiCircCursor = (
            self.labelRoiButton.isChecked()
            and not event.isExit()
            and (noModifier or shift or ctrl)
            and self.labelRoiIsCircularRadioButton.isChecked()
        )
        setWandCursor = (
            self.wandToolButton.isChecked() and not event.isExit() and noModifier
        )
        setLabelRoiCursor = (
            self.labelRoiButton.isChecked() and not event.isExit() and noModifier
        )
        setMoveLabelCursor = (
            self.moveLabelToolButton.isChecked() and not event.isExit() and noModifier
        )
        setExpandLabelCursor = (
            self.expandLabelToolButton.isChecked() and not event.isExit() and noModifier
        )
        setCurvCursor = (
            self.curvToolButton.isChecked() and not event.isExit() and noModifier
        )
        setKeepObjCursor = (
            self.keepIDsButton.isChecked() and not event.isExit() and noModifier
        )
        setCustomAnnotCursor = (
            self.customAnnotButton is not None and not event.isExit() and noModifier
        )
        setManualTrackingCursor = (
            self.manualTrackingButton.isChecked() and not event.isExit() and noModifier
        )
        setManualBackgroundCursor = (
            self.manualBackgroundButton.isChecked()
            and not event.isExit()
            and noModifier
        )
        setZoomRectCursor = (
            self.zoomRectButton.isChecked() and not event.isExit() and noModifier
        )
        setEditIDCursor = self.editIDbutton.isChecked() and not event.isExit()
        magicPromptsON = self.magicPromptsToolButton.isChecked()
        pointsLayerON = self.togglePointsLayerAction.isChecked()
        addPointsByClickingButton = self.buttonAddPointsByClickingActive()
        setAddPointCursor = (
            (pointsLayerON or magicPromptsON)
            and addPointsByClickingButton is not None
            and not event.isExit()
            and noModifier
        )
        overrideCursor = self.app.overrideCursor()
        setPanImageCursor = alt and not event.isExit()
        if setPanImageCursor and overrideCursor is None:
            self.app.setOverrideCursor(Qt.SizeAllCursor)
        elif setBrushCursor or setEraserCursor or setLabelRoiCircCursor:
            self.app.setOverrideCursor(Qt.CrossCursor)
        elif setWandCursor and overrideCursor is None:
            self.app.setOverrideCursor(self.wandCursor)
        elif setLabelRoiCursor and overrideCursor is None:
            self.app.setOverrideCursor(Qt.CrossCursor)
        elif setCurvCursor and overrideCursor is None:
            self.app.setOverrideCursor(self.curvCursor)
        elif setCustomAnnotCursor and overrideCursor is None:
            self.app.setOverrideCursor(Qt.PointingHandCursor)
        elif setAddDelPolyLineCursor:
            self.app.setOverrideCursor(self.polyLineRoiCursor)
        elif setCustomAnnotCursor:
            x, y = event.pos()
            self.highlightHoverID(x, y)
        elif setKeepObjCursor and overrideCursor is None:
            self.app.setOverrideCursor(Qt.PointingHandCursor)
        elif setManualTrackingCursor and overrideCursor is None:
            self.app.setOverrideCursor(Qt.PointingHandCursor)
        elif setManualBackgroundCursor and overrideCursor is None:
            self.app.setOverrideCursor(Qt.PointingHandCursor)
        elif setAddPointCursor:
            self.app.setOverrideCursor(self.addPointsCursor)
        elif setZoomRectCursor:
            self.app.setOverrideCursor(Qt.CrossCursor)
        elif setEditIDCursor and overrideCursor is None:
            if shift:
                self.app.setOverrideCursor(Qt.CrossCursor)
            else:
                self.app.restoreOverrideCursor()

        return {
            "setBrushCursor": setBrushCursor,
            "setEraserCursor": setEraserCursor,
            "setAddDelPolyLineCursor": setAddDelPolyLineCursor,
            "setLabelRoiCircCursor": setLabelRoiCircCursor,
            "setWandCursor": setWandCursor,
            "setLabelRoiCursor": setLabelRoiCursor,
            "setMoveLabelCursor": setMoveLabelCursor,
            "setExpandLabelCursor": setExpandLabelCursor,
            "setCurvCursor": setCurvCursor,
            "setKeepObjCursor": setKeepObjCursor,
            "setCustomAnnotCursor": setCustomAnnotCursor,
            "setManualTrackingCursor": setManualTrackingCursor,
            "setManualBackgroundCursor": setManualBackgroundCursor,
            "setAddPointCursor": setAddPointCursor,
            "setZoomRectCursor": setZoomRectCursor,
            "setEditIDCursor": setEditIDCursor,
        }

    def hover_position(self, is_exit: bool, position) -> tuple[Any, Any]:
        if is_exit:
            return None, None
        return position

    def onCtrlPressedFirstTime(self):
        x, y = self.xHoverImg, self.yHoverImg
        if x is None:
            self.xyOnCtrlPressedFirstTime = None
            return

        xdata, ydata = int(x), int(y)
        Y, X = self.currentLab2D.shape

        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            self.xyOnCtrlPressedFirstTime = None
            return

        ID = self.currentLab2D[ydata, xdata]
        if ID == 0:
            self.xyOnCtrlPressedFirstTime = None
            return

        self.xyOnCtrlPressedFirstTime = (xdata, ydata)

    def onCtrlReleased(self):
        self.xyOnCtrlPressedFirstTime = None

    def point_in_bounds(
        self,
        image_shape: tuple[int, int],
        xdata: int,
        ydata: int,
    ) -> bool:
        y_size, x_size = image_shape
        return 0 <= xdata < x_size and 0 <= ydata < y_size

    def should_draw_ruler_line(
        self,
        *,
        ruler_checked: bool,
        add_deleted_polyline_checked: bool,
        temp_segment_on: bool,
        is_exit: bool,
    ) -> bool:
        return (
            (ruler_checked or add_deleted_polyline_checked)
            and temp_segment_on
            and not is_exit
        )

    def should_set_mirrored_cursor(
        self,
        *,
        override_cursor_is_none: bool,
        is_exit: bool,
        mirrored_cursor_enabled: bool,
        is_hover_img1: bool = True,
    ) -> bool:
        return (
            override_cursor_is_none
            and not is_exit
            and is_hover_img1
            and mirrored_cursor_enabled
        )

    def updateHoverLabelCursor(self, x, y):
        if x is None:
            self.hoverLabelID = 0
            return

        xdata, ydata = int(x), int(y)
        Y, X = self.currentLab2D.shape
        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            return

        ID = self.currentLab2D[ydata, xdata]
        self.hoverLabelID = ID

        if ID == 0:
            if self.highlightedID != 0:
                self.updateAllImages()
                self.highlightedID = 0
            return

        if self.app.overrideCursor() != Qt.SizeAllCursor:
            self.app.setOverrideCursor(Qt.SizeAllCursor)

        if not self.isMovingLabel:
            self.highlightSearchedID(ID)

    def warnAddingPointWithExistingId(self, point_id, table_endname=""):
        posData = self.data[self.pos_i]
        if point_id not in posData.IDs_idxs:
            return True

        msg = widgets.myMessageBox(wrapText=False)
        txt = f"""
            Cell ID {point_id} <b>already exists</b>!<br><br>
            Are you sure you want to add this point?
        """
        if table_endname:
            txt = f"""
                The loaded table <code>{table_endname}</code> has point id 
                {point_id}.
                <br><br>However, {txt}
            """
        txt = html_utils.paragraph(txt)
        _, _, yesButton = msg.warning(
            self,
            f"Cell ID {point_id} already exist",
            txt,
            buttonsTexts=("Cancel", "No, do not add", f"Yes, add point id {point_id}"),
        )
        return msg.clickedButton == yesButton
