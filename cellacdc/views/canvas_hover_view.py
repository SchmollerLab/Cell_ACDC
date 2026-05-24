"""Qt view adapter for canvas hover and cursor interactions."""

from __future__ import annotations

import pyqtgraph as pg
from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication

from cellacdc import html_utils, widgets
from cellacdc.viewmodels.canvas_hover_viewmodel import CanvasHoverViewModel


class CanvasHoverView:
    """Qt-facing adapter around canvas hover workflows."""

    LEGACY_METHODS = (
        'updateHoverLabelCursor',
        'gui_hoverEventRightImage',
        'onCtrlPressedFirstTime',
        'onCtrlReleased',
        'gui_hoverEventImg1',
        'drawTempMothBudLine',
        'drawTempMergeObjsLine',
        'gui_add_ax_cursors',
        'gui_setCursor',
        'warnAddingPointWithExistingId',
        'gui_hoverEventImg2',
        'drawTempRulerLine',
    )

    def __init__(self, host, view_model: CanvasHoverViewModel):
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

    def updateHoverLabelCursor(self, x, y):
        if x is None:
            self.hoverLabelID = 0
            return

        xdata, ydata = int(x), int(y)
        if not self.view_model.point_in_bounds(
            self.currentLab2D.shape,
            xdata,
            ydata,
        ):
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

    def gui_hoverEventRightImage(self, event):
        try:
            posData = self.data[self.pos_i]
        except AttributeError:
            return

        if event.isExit():
            self.resetCursors()

        self.gui_hoverEventImg1(event, isHoverImg1=False)
        setMirroredCursor = self.view_model.should_set_mirrored_cursor(
            override_cursor_is_none=self.app.overrideCursor() is None,
            is_exit=event.isExit(),
            mirrored_cursor_enabled=self.showMirroredCursorAction.isChecked(),
            is_hover_img1=True,
        )
        if setMirroredCursor:
            x, y = event.pos()
            self.ax1_cursor.setData([x], [y])

    def onCtrlPressedFirstTime(self):
        x, y = self.xHoverImg, self.yHoverImg
        if x is None:
            self.xyOnCtrlPressedFirstTime = None
            return

        xdata, ydata = int(x), int(y)
        if not self.view_model.point_in_bounds(
            self.currentLab2D.shape,
            xdata,
            ydata,
        ):
            self.xyOnCtrlPressedFirstTime = None
            return

        ID = self.currentLab2D[ydata, xdata]
        if ID == 0:
            self.xyOnCtrlPressedFirstTime = None
            return

        self.xyOnCtrlPressedFirstTime = (xdata, ydata)

    def onCtrlReleased(self):
        self.xyOnCtrlPressedFirstTime = None

    def gui_hoverEventImg1(self, event, isHoverImg1=True):
        try:
            posData = self.data[self.pos_i]
        except AttributeError:
            return

        self.xHoverImg, self.yHoverImg = self.view_model.hover_position(
            event.isExit(),
            event.pos(),
        )

        if event.isExit():
            self.resetCursor()

        if not event.isExit() and self.slideshowWin is not None:
            self.slideshowWin.setMirroredCursorPos(*event.pos())

        # Alt key was released --> restore cursor
        modifiers = QGuiApplication.keyboardModifiers()
        cursorsInfo = self.gui_setCursor(modifiers, event)
        self.highlightHoverLostObj(modifiers, event)

        drawRulerLine = self.view_model.should_draw_ruler_line(
            ruler_checked=self.rulerButton.isChecked(),
            add_deleted_polyline_checked=(
                self.addDelPolyLineRoiButton.isChecked()
            ),
            temp_segment_on=self.tempSegmentON,
            is_exit=event.isExit(),
        )
        if drawRulerLine:
            self.drawTempRulerLine(event)

        if not event.isExit():
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            if self.view_model.point_in_bounds(
                self.img1.image.shape[:2],
                xdata,
                ydata,
            ):
                ID = self.currentLab2D[ydata, xdata]
                self.updatePropsWidget(ID, fromHover=True)
                activeToolButton = self.status_hover_view.active_tool_button()
                hoverText = self.status_hover_view.hover_values_formatted(
                    xdata, ydata, activeToolButton, isHoverImg1
                )
                self.status_hover_view.check_highlight_scale_bar(
                    x, y, activeToolButton
                )
                self.status_hover_view.check_highlight_timestamp(
                    x, y, activeToolButton
                )
                self.wcLabel.setText(hoverText)
        else:
            self.clickedOnBud = False
            self.BudMothTempLine.setData([], [])
            self.wcLabel.setText('')

        if cursorsInfo['setKeepObjCursor']:
            x, y = event.pos()
            self.highlightHoverIDsKeptObj(x, y)

        if cursorsInfo['setManualTrackingCursor']:
            x, y = event.pos()
            # self.highlightHoverID(x, y)
            self.drawManualTrackingGhost(x, y)

        if cursorsInfo['setManualBackgroundCursor']:
            x, y = event.pos()
            # self.highlightHoverID(x, y)
            self.drawManualBackgroundObj(x, y)

        if (
                not cursorsInfo['setManualTrackingCursor']
                and not cursorsInfo['setManualBackgroundCursor']
            ):
            self.clearGhost()

        setMoveLabelCursor = cursorsInfo['setMoveLabelCursor']
        setExpandLabelCursor = cursorsInfo['setExpandLabelCursor']
        if setMoveLabelCursor or setExpandLabelCursor:
            x, y = event.pos()
            self.updateHoverLabelCursor(x, y)

        # Draw eraser circle
        if cursorsInfo['setEraserCursor']:
            x, y = event.pos()
            self.updateEraserCursor(x, y, isHoverImg1=isHoverImg1)
            self.hideItemsHoverBrush(xy=(x, y))
        elif self.eraserButton.isChecked() and not event.isExit():
            if self.xyOnCtrlPressedFirstTime is not None:
                self.updateEraserCursor(
                    x, y, xyLocked=self.xyOnCtrlPressedFirstTime,
                    isHoverImg1=isHoverImg1
                )
                self.hideItemsHoverBrush(xy=(x, y))
        else:
            eraserCursors = (
                self.ax1_EraserCircle, self.ax2_EraserCircle,
                self.ax1_EraserX, self.ax2_EraserX
            )
            self.setHoverToolSymbolData([], [], eraserCursors)

        # Draw Brush circle
        if cursorsInfo['setBrushCursor']:
            x, y = event.pos()
            self.updateBrushCursor(x, y, isHoverImg1=isHoverImg1)
            self.hideItemsHoverBrush(xy=(x, y))
        elif cursorsInfo['setAddPointCursor']:
            x, y = event.pos()
            self.setHoverCircleAddPoint(x, y)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )

        # Draw label ROi circular cursor
        setLabelRoiCircCursor = cursorsInfo['setLabelRoiCircCursor']
        if setLabelRoiCircCursor:
            x, y = event.pos()
        else:
            x, y = None, None
        self.updateLabelRoiCircularCursor(x, y, setLabelRoiCircCursor)

        drawMothBudLine = (
            self.assignBudMothButton.isChecked() and self.clickedOnBud
            and not event.isExit()
        )
        if drawMothBudLine:
            self.drawTempMothBudLine(event, posData)

        drawMergeObjsLine = (
            self.mergeIDsButton.isChecked() and not event.isExit()
        )
        if drawMergeObjsLine:
            self.drawTempMergeObjsLine(event, posData, modifiers)

        # Temporarily draw spline curve
        # see https://stackoverflow.com/questions/33962717/interpolating-a-closed-curve-using-scipy
        drawSpline = (
            self.curvToolButton.isChecked() and self.splineHoverON
            and not event.isExit()
        )
        if drawSpline:
            self.curvature_tools_view.hoverEventDrawSpline(event)

        setMirroredCursor = self.view_model.should_set_mirrored_cursor(
            override_cursor_is_none=self.app.overrideCursor() is None,
            is_exit=event.isExit(),
            mirrored_cursor_enabled=self.showMirroredCursorAction.isChecked(),
            is_hover_img1=isHoverImg1,
        )
        if setMirroredCursor:
            x, y = event.pos()
            self.ax2_cursor.setData([x], [y])
        else:
            self.ax2_cursor.setData([], [])

        return cursorsInfo

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

    def gui_add_ax_cursors(self):
        try:
            self.ax1.removeItem(self.ax1_cursor)
            self.ax2.removeItem(self.ax2_cursor)
        except Exception as e:
            pass

        self.ax2_cursor = pg.ScatterPlotItem(
            symbol='+', pxMode=True, pen=pg.mkPen('k', width=1),
            brush=pg.mkBrush('w'), size=16, tip=None
        )
        self.ax2.addItem(self.ax2_cursor)

        self.ax1_cursor = pg.ScatterPlotItem(
            symbol='+', pxMode=True, pen=pg.mkPen('k', width=1),
            brush=pg.mkBrush('w'), size=16, tip=None
        )
        self.ax1.addItem(self.ax1_cursor)

    def _cursor_flags(self, modifiers, event):
        return self.view_model.cursor_flags(
            is_exit=event.isExit(),
            no_modifier=modifiers == Qt.NoModifier,
            shift=modifiers == Qt.ShiftModifier,
            ctrl=modifiers == Qt.ControlModifier,
            alt=modifiers == Qt.AltModifier,
            brush_checked=self.brushButton.isChecked(),
            eraser_checked=self.eraserButton.isChecked(),
            add_deleted_polyline_checked=(
                self.addDelPolyLineRoiButton.isChecked()
            ),
            label_roi_checked=self.labelRoiButton.isChecked(),
            label_roi_circular_checked=(
                self.labelRoiIsCircularRadioButton.isChecked()
            ),
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

    def gui_setCursor(self, modifiers, event):
        noModifier = modifiers == Qt.NoModifier
        shift = modifiers == Qt.ShiftModifier

        # Alt key was released --> restore cursor
        if self.app.overrideCursor() == Qt.SizeAllCursor and noModifier:
            self.app.restoreOverrideCursor()

        flags = self._cursor_flags(modifiers, event)
        setBrushCursor = flags['setBrushCursor']
        setEraserCursor = flags['setEraserCursor']
        setAddDelPolyLineCursor = flags['setAddDelPolyLineCursor']
        setLabelRoiCircCursor = flags['setLabelRoiCircCursor']
        setWandCursor = flags['setWandCursor']
        setLabelRoiCursor = flags['setLabelRoiCursor']
        setCurvCursor = flags['setCurvCursor']
        setKeepObjCursor = flags['setKeepObjCursor']
        setCustomAnnotCursor = flags['setCustomAnnotCursor']
        setManualTrackingCursor = flags['setManualTrackingCursor']
        setManualBackgroundCursor = flags['setManualBackgroundCursor']
        setAddPointCursor = flags['setAddPointCursor']
        setZoomRectCursor = flags['setZoomRectCursor']
        setEditIDCursor = flags['setEditIDCursor']
        overrideCursor = self.app.overrideCursor()
        setPanImageCursor = flags['setPanImageCursor']
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

        return flags

    def warnAddingPointWithExistingId(self, point_id, table_endname=''):
        posData = self.data[self.pos_i]
        if not point_id in posData.IDs_idxs:
            return True

        msg = widgets.myMessageBox(wrapText=False)
        txt = (f"""
            Cell ID {point_id} <b>already exists</b>!<br><br>
            Are you sure you want to add this point?
        """)
        if table_endname:
            txt = (f"""
                The loaded table <code>{table_endname}</code> has point id
                {point_id}.
                <br><br>However, {txt}
            """)
        txt = html_utils.paragraph(txt)
        _, _, yesButton = msg.warning(
            self.host, f'Cell ID {point_id} already exist', txt,
            buttonsTexts=(
                'Cancel', 'No, do not add', f'Yes, add point id {point_id}'
            )
        )
        return msg.clickedButton == yesButton

    def gui_hoverEventImg2(self, event):
        try:
            posData = self.data[self.pos_i]
        except AttributeError:
            return

        self.xHoverImg, self.yHoverImg = self.view_model.hover_position(
            event.isExit(),
            event.pos(),
        )

        # Cursor left image --> restore cursor
        if event.isExit() and self.app.overrideCursor() is not None:
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()

        # Alt key was released --> restore cursor
        modifiers = QGuiApplication.keyboardModifiers()
        noModifier = modifiers == Qt.NoModifier
        shift = modifiers == Qt.ShiftModifier
        if self.app.overrideCursor() == Qt.SizeAllCursor and noModifier:
            self.app.restoreOverrideCursor()

        flags = self._cursor_flags(modifiers, event)
        setBrushCursor = flags['setBrushCursor']
        setEraserCursor = flags['setEraserCursor']
        setLabelRoiCircCursor = flags['setLabelRoiCircCursor']
        if setBrushCursor or setEraserCursor or setLabelRoiCircCursor:
            self.app.setOverrideCursor(Qt.CrossCursor)

        # Cursor is moving on image while Alt key is pressed --> pan cursor
        setPanImageCursor = flags['setPanImageCursor']
        if setPanImageCursor and self.app.overrideCursor() is None:
            self.app.setOverrideCursor(Qt.SizeAllCursor)

        setKeepObjCursor = flags['setKeepObjCursor']
        if setKeepObjCursor and self.app.overrideCursor() is None:
            self.app.setOverrideCursor(Qt.PointingHandCursor)

        # Update x, y, value label bottom right
        if not event.isExit():
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            _img = self.currentLab2D
            Y, X = _img.shape
            # hoverText = self.status_hover_view.hover_values_formatted(
            #     xdata, ydata
            # )
            # self.wcLabel.setText(hoverText)
        else:
            if self.eraserButton.isChecked() or self.brushButton.isChecked():
                self.gui_mouseReleaseEventImg2(event)
            self.wcLabel.setText(f'')

        if flags['setMoveLabelCursor'] or flags['setExpandLabelCursor']:
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
                [], [], (self.ax1_EraserCircle, self.ax2_EraserCircle,
                         self.ax1_EraserX, self.ax2_EraserX)
            )

        # Draw Brush circle
        if setBrushCursor:
            x, y = event.pos()
            self.updateBrushCursor(x, y, isHoverImg1=False)
        else:
            self.setHoverToolSymbolData(
                [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
            )

        # Draw label ROi circular cursor
        if setLabelRoiCircCursor:
            x, y = event.pos()
        else:
            x, y = None, None
        self.updateLabelRoiCircularCursor(x, y, setLabelRoiCircCursor)

    def drawTempRulerLine(self, event):
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        x, y = event.pos()
        x1, y1 = int(x), int(y)
        xxRA, yyRA = self.ax1_rulerAnchorsItem.getData()
        x0, y0 = xxRA[0], yyRA[0]
        if ctrl:
            x1, y1 = self.host.view_model.geometry.snap_xy_to_closest_angle(
                x0, y0, x1, y1
            )
        self.ax1_rulerPlotItem.setData([x0, x1], [y0, y1])
