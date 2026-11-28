"""Tool cursors and image hover event handlers of the main GUI. Mixins of
guiWin (gui/main_window/window.py) only: the measurements viewer never
composes them.
"""
import numpy as np
from qtpy.QtCore import Qt, QSize
from qtpy.QtGui import QIcon, QCursor, QGuiApplication, QPixmap, QPainter
from qtpy.QtWidgets import QApplication
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import myutils
from cellacdc import widgets
from cellacdc import qutils
from . import constants as gui_main_constants


class CursorsGuiElements:
    """Tool cursors of the main GUI: creation and registry, brush and
    eraser circle cursors, hover tool symbols and cursor resets.
    """

    def gui_createCursors(self):

        cursor_size = QSize(gui_main_constants.TOTAL_SIZE_CURSOR, gui_main_constants.TOTAL_SIZE_CURSOR)
        self._toolCursorCenterPixmaps = {
            'blank': QPixmap(gui_main_constants.CENTER_SIZE_CURSOR, gui_main_constants.CENTER_SIZE_CURSOR),
            'crosshair': QPixmap(":cross_cursor.svg").scaled(
                QSize(gui_main_constants.CENTER_SIZE_CURSOR, gui_main_constants.CENTER_SIZE_CURSOR), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ),
            'wand': QPixmap(":wand_cursor.svg").scaled(
                QSize(gui_main_constants.CENTER_SIZE_CURSOR*2, gui_main_constants.CENTER_SIZE_CURSOR*2), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ),
            'curvature': QPixmap(":curv_cursor.svg").scaled(
                QSize(gui_main_constants.CENTER_SIZE_CURSOR*2, gui_main_constants.CENTER_SIZE_CURSOR*2), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ),
            'polyline': QPixmap(":addDelPolyLineRoi_cursor.svg").scaled(
                QSize(gui_main_constants.CENTER_SIZE_CURSOR, gui_main_constants.CENTER_SIZE_CURSOR), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ),
            'default_no_lc': QPixmap(":crosshair.svg").scaled(
                QSize(gui_main_constants.CENTER_SIZE_CURSOR, gui_main_constants.CENTER_SIZE_CURSOR), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ),
            'default_lc': QPixmap(":crosshair.svg").scaled(
                QSize(gui_main_constants.CENTER_SIZE_CURSOR, gui_main_constants.CENTER_SIZE_CURSOR), 
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            ),
        }
        self._toolCursorCenterPixmaps['blank'].fill(Qt.transparent)
        self._normalToolCursorPixmaps = {}
        for isHoverImg1 in (True, False):
            pixmap = QPixmap(cursor_size)
            pixmap.fill(Qt.transparent)
            self._normalToolCursorPixmaps[isHoverImg1] = pixmap

        self.toolCursors = {}
        self.leftClickCursorTools = ()
        self.rightClickCursorTools = ()
        self._cursorHoverImage = None

    def gui_createToolCursorRegistry(self):
        # Ordering mirrors the mutually exclusive click handling priority.
        # pattern is: (tool_name, tool_button, display small icon, mode)
        self.leftClickCursorTools = (
            ('brush', self.brushButton, False, 'Segmentation and Tracking'),
            ('eraser', self.eraserButton, False, 'Segmentation and Tracking'),
            ('curvature', self.curvToolButton, False, 'Segmentation and Tracking'),
            ('magic_wand', self.wandToolButton, False, 'Segmentation and Tracking'),
            ('magic_prompts_select_cells', self.magicPromptsToolbar.selectCellsAction, True, 'Segmentation and Tracking'),
            ('magic_prompts', self.magicPromptsToolButton, True, 'Segmentation and Tracking'),
            ('label_roi', self.labelRoiButton, True, 'Segmentation and Tracking'),
            ('ruler', self.rulerButton, True, 'Segmentation and Tracking'),
            ('polyline_deletion_roi', self.addDelPolyLineRoiButton, True, 'Segmentation and Tracking'),
            ('clear_region', self.clearFreehandRoiButton, True, 'Segmentation and Tracking'),
            ('merge_ids', self.mergeIDsButton, True, 'Segmentation and Tracking'),
            ('keep_ids', self.keepIDsButton, True, 'Segmentation and Tracking'),
            ('whitelist_ids', self.whitelistIDsButton, True, 'Segmentation and Tracking'),
            ('manual_background', self.manualBackgroundButton, True, 'Segmentation and Tracking'),
            ('zoom_rectangle', self.zoomRectButton, True, 'Segmentation and Tracking'),
            ('toggle_points_layer', self.togglePointsLayerAction, True, 'Segmentation and Tracking'),
        )
        self.rightClickCursorTools = (
            ('separate_objects', self.separateBudButton, True, 'Segmentation and Tracking'),
            ('fill_holes', self.fillHolesToolButton, True, 'Segmentation and Tracking'),
            ('hull_contour', self.hullContToolButton, True, 'Segmentation and Tracking'),
            ('move_object', self.moveLabelToolButton, True, 'Segmentation and Tracking'),
            ('edit_id', self.editIDbutton, True, 'Segmentation and Tracking'),
            ('merge_ids', self.mergeIDsButton, True, 'Segmentation and Tracking'),
            ('keep_ids', self.keepIDsButton, True, 'Segmentation and Tracking'),
            ('whitelist_ids', self.whitelistIDsButton, True, 'Segmentation and Tracking'),
            ('exclude_from_analysis', self.binCellButton, True, 'Segmentation and Tracking'),
            ('annotate_dead', self.ripCellButton, True, 'Segmentation and Tracking'),
            ('assign_mother', self.assignBudMothButton, True, 'Segmentation and Tracking'),
            ('set_history_known', self.setIsHistoryKnownButton, True, 'Segmentation and Tracking'),
            ('manual_tracking', self.manualTrackingButton, True, 'Segmentation and Tracking'),
            ('manual_background', self.manualBackgroundButton, True, 'Segmentation and Tracking'),
            ('copy_lost_contour', self.copyLostObjButton, True, 'Segmentation and Tracking'),
            ('curvature', self.curvToolButton, False, 'Segmentation and Tracking'),
            ('label_roi', self.labelRoiButton, True, 'Segmentation and Tracking'),
            ('magic_prompts_select_cells', self.magicPromptsToolbar.selectCellsAction, True, 'Segmentation and Tracking'),
            ('magic_prompts', self.magicPromptsToolButton, True, 'Segmentation and Tracking'),
            ('toggle_points_layer', self.togglePointsLayerAction, True, 'Segmentation and Tracking'),
        )
        cursor_tools = {
            tool for _, tool, _, _ in (
                self.leftClickCursorTools + self.rightClickCursorTools
            )
        }
        for tool in cursor_tools:
            tool.toggled.connect(self.gui_refreshToolCursor)

    def gui_activeCursorTool(self, tools):
        curr_mode = str(self.modeComboBox.currentText())
        for name, tool, draw_icon, mode in tools:
            if mode != curr_mode and not curr_mode == 'Snapshot':
                continue
            if name == 'magic_prompts_select_cells':
                # Only while selecting cells with brush strokes prompts
                if self.isBrushStrokesSelectCellsActive():
                    return name, tool.icon()
                continue
            if name == 'toggle_points_layer':
                magicPromptsON = self.magicPromptsToolButton.isChecked()
                pointsLayerON = self.togglePointsLayerAction.isChecked()
                addPointsByClickingButton = self.buttonAddPointsByClickingActive()
                if (addPointsByClickingButton is not None
                    and (magicPromptsON or pointsLayerON)
                    and addPointsByClickingButton.isChecked()
                    and not self.isBrushStrokesPromptActive()
                    and not self.isBoxPromptActive()):
                    return 'toggle_points_layer', tool.icon()
                else:
                    continue
            if tool.isChecked():
                if draw_icon:
                    return name, tool.icon()
                else:
                    return name, QIcon()

        if self.customAnnotButton is not None and self.customAnnotButton.isChecked():
            return 'custom_annotation', self.customAnnotButton.icon()

        return None, QIcon()

    def gui_getToolCursorCenter(self, left_name, right_name):
        active_tools = {left_name, right_name}
        isLeftClickActive = left_name is not None
        overlay_cursor_tools = set() #{
            # 'brush', 'eraser'
        #}
        if active_tools & overlay_cursor_tools:
            center = 'blank'
        elif (
                'label_roi' in active_tools
                and self.labelRoiIsCircularRadioButton.isChecked()
            ):
            center = 'blank'
        elif 'magic_wand' in active_tools:
            center = 'wand'
        elif 'curvature' in active_tools:
            center = 'curvature'
        elif 'polyline_deletion_roi' in active_tools:
            center = 'polyline'
        elif 'magic_prompts' in active_tools and self.isBoxPromptActive():
            # Drawing box prompts
            center = 'crosshair'
        elif 'toggle_points_layer' in active_tools:
            center = 'crosshair'
        elif isLeftClickActive:
            center = 'default_lc'
        else:
            center = 'default_no_lc'
        return center, self._toolCursorCenterPixmaps[center]

    def gui_createToolCursor(self, isHoverImg1, left_tool, right_tool):
        left_name, left_icon = left_tool
        right_name, right_icon = right_tool
        center_name, center_pixmap = self.gui_getToolCursorCenter(
            left_name, right_name
        )
        key = (isHoverImg1, left_name, right_name, center_name)
        cursor = self.toolCursors.get(key)
        if cursor is not None:
            return cursor
        
        actual_center_size = center_pixmap.width()  # Assuming square pixmap

        pixmap = self._normalToolCursorPixmaps[isHoverImg1].copy()
        painter = QPainter(pixmap)

        center = (gui_main_constants.TOTAL_SIZE_CURSOR - actual_center_size) // 2

        # Calculate positions automatically
        left_x = 0 # align left icon to the left edge
        right_x = gui_main_constants.TOTAL_SIZE_CURSOR - gui_main_constants.TOOL_SIZE_CURSOR # align right icon to the right edge
        tool_y = 0 # align both icons to the top edge

        painter.drawPixmap(center, center, center_pixmap)

        for icon, x in ((left_icon, left_x), (right_icon, right_x)):
            if not icon.isNull():
                tool_pixmap = icon.pixmap(QSize(gui_main_constants.TOOL_SIZE_CURSOR, gui_main_constants.TOOL_SIZE_CURSOR))
                painter.drawPixmap(x, tool_y, tool_pixmap)

        painter.end()

        pixmap = pixmap.scaled(
            QSize(gui_main_constants.FINAL_SIZE_CURSOR, gui_main_constants.FINAL_SIZE_CURSOR), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        cursor = QCursor(pixmap, gui_main_constants.FINAL_SIZE_CURSOR // 2, gui_main_constants.FINAL_SIZE_CURSOR // 2)
        self.toolCursors[key] = cursor
        return cursor

    def gui_getToolCursor(self, isHoverImg1):
        left_tool = self.gui_activeCursorTool(self.leftClickCursorTools)
        right_tool = self.gui_activeCursorTool(self.rightClickCursorTools)
        return self.gui_createToolCursor(isHoverImg1, left_tool, right_tool)

    def gui_setToolCursor(self, event, isHoverImg1):
        if event.isExit():
            return

        self.gui_refreshToolCursor()

    def setDelRoiCursor(self):
        if self.app.overrideCursor() == Qt.SizeAllCursor:
            return
        
        self.app.setOverrideCursor(Qt.SizeAllCursor)


    def _isOnlyMainWinUnderPointer(self):
        # widgetAt only sees this app's widgets: None for other apps' windows.
        # .window() differs for dialogs, floating docks, popups, etc.
        w = QApplication.widgetAt(QCursor.pos())
        return w is not None and w.window() is self

    def _releaseWaitOverride(self):
        if getattr(self, '_waitOverrideActive', False):
            self.app.restoreOverrideCursor()
            self._waitOverrideActive = False

    def gui_refreshToolCursor(self, *args):
        if not self.isEnabled():
            if not self._busyTimer.isActive():
                self._busyTimer.start()   # re-check obstruction while disabled
            if self._isOnlyMainWinUnderPointer():
                if not self._waitOverrideActive:
                    if self.app.overrideCursor() is None:
                        self.app.setOverrideCursor(Qt.WaitCursor)
                    else:
                        self.app.changeOverrideCursor(Qt.WaitCursor)
                    self._waitOverrideActive = True
            else:
                self._releaseWaitOverride()
            return

        self._busyTimer.stop()
        self._releaseWaitOverride()
        # Called at every mouse move --> avoid needless (platform) cursor
        # changes
        if self.cursor().shape() != Qt.CursorShape.ArrowCursor:
            self.unsetCursor()

        isHoverImg1 = self._cursorHoverImage
        if isHoverImg1 is None:
            return

        overrideCursor = self.app.overrideCursor()
        if overrideCursor == Qt.SizeAllCursor:
            return

        cursor = self.gui_getToolCursor(isHoverImg1)
        if overrideCursor is None:
            self.app.setOverrideCursor(cursor)
        elif not qutils.is_same_cursor(overrideCursor, cursor):
            self.app.changeOverrideCursor(cursor)

    def clearCursors(self):
        self.ax1_cursor.setData([], [])
        self.ax2_cursor.setData([], [])
        self.update3DViewerCursor(None)
        self.setHoverToolSymbolData(
            [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
        )  
        eraserCursors = (
            self.ax1_EraserCircle, self.ax2_EraserCircle,
            self.ax1_EraserX, self.ax2_EraserX
        )
        self.setHoverToolSymbolData([], [], eraserCursors)

    def activeEraserCircleCursors(self, isHoverImg1):
        if self.showMirroredCursorAction.isChecked():
            return self.ax1_EraserCircle, self.ax2_EraserCircle
        
        if isHoverImg1:
            return self.ax1_EraserCircle,
        else:
            return self.ax2_EraserCircle,

    def activeEraserXCursors(self, isHoverImg1):
        if self.showMirroredCursorAction.isChecked():
            return self.ax1_EraserX, self.ax2_EraserX
        
        if isHoverImg1:
            return self.ax1_EraserX,
        else:
            return self.ax2_EraserX,

    def activeBrushCircleCursors(self, isHoverImg1):
        if self.showMirroredCursorAction.isChecked():
            return self.ax1_BrushCircle, self.ax2_BrushCircle
        
        if isHoverImg1:
            return self.ax1_BrushCircle,
        else:
            return self.ax2_BrushCircle,

    def gui_add_ax_cursors(self):
        try:
            self.ax1.removeItem(self.ax1_cursor)
            self.ax2.removeItem(self.ax2_cursor)
        except Exception as e:
            pass

        # Moved at every mouse move --> lightweight item (much faster than
        # a pg.ScatterPlotItem)
        self.ax2_cursor = widgets.MirroredCursorItem(
            symbol=gui_main_constants.MIRRORED_CURSOR_PG_SYMBOL,
            size=gui_main_constants.FINAL_SIZE_CURSOR // 2,
            pen=pg.mkPen('w', width=1),
            brush=pg.mkBrush((255, 255, 255, 70)),
        )
        self.ax2.addItem(self.ax2_cursor, ignoreBounds=True)

        self.ax1_cursor = widgets.MirroredCursorItem(
            symbol=gui_main_constants.MIRRORED_CURSOR_PG_SYMBOL,
            size=gui_main_constants.FINAL_SIZE_CURSOR // 2,
            pen=pg.mkPen('w', width=1),
            brush=pg.mkBrush((255, 255, 255, 70)),
        )
        self.ax1.addItem(self.ax1_cursor, ignoreBounds=True)

    def gui_setCursor(self, modifiers, event, isHoverImg1=True):
        noModifier = modifiers == Qt.NoModifier
        shift = modifiers == Qt.ShiftModifier
        ctrl = modifiers == Qt.ControlModifier
        alt = modifiers == Qt.AltModifier
        
        posData = self.data[self.pos_i]
        delROIs = (
            posData.allData_li[posData.frame_i]['delROIs_info']['rois']
        )
        # Alt key was released --> restore cursor
        if self.app.overrideCursor() == Qt.SizeAllCursor and noModifier:
            self.app.restoreOverrideCursor()

        setDelRoiCursor = False
        if not event.isExit():
            x, y = event.pos()
            setDelRoiCursor = any([
                self.isMouseOnDelRoi(roi, x, y) for roi in delROIs
            ])
        setBrushCursor = (
            self.brushButton.isChecked() and not event.isExit()
            and (noModifier or shift or ctrl) and not setDelRoiCursor
        )
        setEraserCursor = (
            self.eraserButton.isChecked() and not event.isExit()
            and noModifier and not setDelRoiCursor
        )
        setAddDelPolyLineCursor = (
            self.addDelPolyLineRoiButton.isChecked() and not event.isExit()
            and noModifier
        )
        setLabelRoiCircCursor = (
            self.labelRoiButton.isChecked() and not event.isExit()
            and (noModifier or shift or ctrl)
            and self.labelRoiIsCircularRadioButton.isChecked()
        )
        setWandCursor = (
            self.wandToolButton.isChecked() and not event.isExit()
            and noModifier and not setDelRoiCursor
        )
        setLabelRoiCursor = (
            self.labelRoiButton.isChecked() and not event.isExit()
            and noModifier
        )
        setMoveLabelCursor = (
            self.moveLabelToolButton.isChecked() and not event.isExit()
            and noModifier
        )
        setExpandLabelCursor = (
            self.expandLabelToolButton.isChecked() and not event.isExit()
            and noModifier
        )
        setCurvCursor = (
            self.curvToolButton.isChecked() and not event.isExit()
            and noModifier and not setDelRoiCursor
        )
        setKeepObjCursor = (
            self.keepIDsButton.isChecked() and not event.isExit()
            and noModifier
        )
        setCustomAnnotCursor = (
            self.customAnnotButton is not None and not event.isExit()
            and noModifier
        )
        setManualTrackingCursor = (
            self.manualTrackingButton.isChecked() 
            and not event.isExit()
            and noModifier
        )
        setManualBackgroundCursor = (
            self.manualBackgroundButton.isChecked()
            and not event.isExit()
            and noModifier
        )
        setZoomRectCursor = (
            self.zoomRectButton.isChecked() and not event.isExit()
            and noModifier
        )
        setEditIDCursor = (
            self.editIDbutton.isChecked() and not event.isExit()
        )
        magicPromptsON = self.magicPromptsToolButton.isChecked()
        pointsLayerON = self.togglePointsLayerAction.isChecked()
        addPointsByClickingButton = self.buttonAddPointsByClickingActive()
        isBrushStrokesPromptActive = self.isBrushStrokesPromptActive()
        setAddPointCursor = (
            (pointsLayerON or magicPromptsON)
            and addPointsByClickingButton is not None
            and not event.isExit()
            and noModifier
            and not isBrushStrokesPromptActive
            and not self.isBoxPromptActive()
        )
        setBrushStrokesCursor = (
            isBrushStrokesPromptActive and not event.isExit() and not alt
            and not setDelRoiCursor
            and not self.isBrushStrokesSelectCellsActive()
        )
        overrideCursor = self.app.overrideCursor()
        setPanImageCursor = alt and not event.isExit()
        if setPanImageCursor and overrideCursor is None:
            self.app.setOverrideCursor(Qt.SizeAllCursor)
        elif not event.isExit():
            self.gui_setToolCursor(event, isHoverImg1)

        if setCustomAnnotCursor:
            x, y = event.pos()
            self.highlightHoverID(x, y)
        
        return {
            'setDelRoiCursor': setDelRoiCursor,
            'setBrushCursor': setBrushCursor,
            'setEraserCursor': setEraserCursor,
            'setAddDelPolyLineCursor': setAddDelPolyLineCursor,
            'setLabelRoiCircCursor': setLabelRoiCircCursor,
            'setWandCursor': setWandCursor,
            'setLabelRoiCursor': setLabelRoiCursor,
            'setMoveLabelCursor': setMoveLabelCursor,
            'setExpandLabelCursor': setExpandLabelCursor,
            'setCurvCursor': setCurvCursor,
            'setKeepObjCursor': setKeepObjCursor,
            'setCustomAnnotCursor': setCustomAnnotCursor,
            'setManualTrackingCursor': setManualTrackingCursor,
            'setManualBackgroundCursor': setManualBackgroundCursor,
            'setAddPointCursor': setAddPointCursor,
            'setZoomRectCursor': setZoomRectCursor,
            'setEditIDCursor': setEditIDCursor,
            'setBrushStrokesCursor': setBrushStrokesCursor,
        }

    def setHoverToolSymbolData(self, xx, yy, ScatterItems, size=None):
        if not xx:
            self.ax1_lostObjScatterItem.setVisible(True)
            self.ax2_lostObjScatterItem.setVisible(True)

            self.ax1_lostTrackedScatterItem.setVisible(True)
            self.ax2_lostTrackedScatterItem.setVisible(True)

        for item in ScatterItems:
            isAlreadyEmpty = (
                not xx
                and isinstance(item, pg.ScatterPlotItem)
                and len(item.data) == 0
            )
            if isAlreadyEmpty:
                # Called at every mouse move. Setting empty data again
                # would repaint the region at the origin of the image
                continue
            if size is None:
                item.setData(xx, yy)
            else:
                item.setData(xx, yy, size=size)

    def setHoverToolSymbolColor(
            self, xdata, ydata, pen, ScatterItems, button,
            brush=None, hoverRGB=None, ID=None, byPassShiftCheck=False
        ):
        modifiers = QGuiApplication.keyboardModifiers()
        if byPassShiftCheck:
            shift = False
        else:
            shift = modifiers == Qt.ShiftModifier
        
        posData = self.data[self.pos_i]
        Y, X = self.get_2Dlab(posData.lab, force_z=False).shape
        if not myutils.is_in_bounds(xdata, ydata, X, Y):
            return

        self.isHoverZneighID = False
        if ID is None:
            hoverID = self.getHoverID(
                xdata, ydata, byPassShiftCheck=byPassShiftCheck
            )
        else:
            hoverID = ID
        
        if hoverID == 0:
            for item in ScatterItems:
                item.setPen(pen)
                item.setBrush(brush)
        else:
            try:
                rgb = self.lut[hoverID]
                rgb = rgb if hoverRGB is None else hoverRGB
                rgbPen = np.clip(rgb*1.1, 0, 255)
                for item in ScatterItems:
                    item.setPen(*rgbPen, width=2)
                    item.setBrush(*rgb, 100)
            except IndexError:
                pass
        
        checkChangeID = (
            self.isHoverZneighID and not shift
            and self.lastHoverID != hoverID
        )
        if checkChangeID:
            # We are hovering an ID in z+1 or z-1
            self.restoreBrushID = hoverID
            # self.changeBrushID()
        
        self.lastHoverID = hoverID

    def updateBrushCursor(self, x, y, isHoverImg1=True):
        if x is None:
            return

        xdata, ydata = int(x), int(y)
        _img = self.currentLab2D
        Y, X = _img.shape

        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            return

        size = self.brushSizeSpinbox.value()*2
        self.setHoverToolSymbolData(
            [x], [y], self.activeBrushCircleCursors(isHoverImg1),
            size=size
        )
        self.setHoverToolSymbolColor(
            xdata, ydata, self.ax2_BrushCirclePen,
            self.activeBrushCircleCursors(isHoverImg1),
            self.brushButton, brush=self.ax2_BrushCircleBrush
        )

    def resetCursors(self):
        self.ax1_cursor.setData([], [])
        self.ax2_cursor.setData([], [])
        while self.app.overrideCursor() is not None:
            self.app.restoreOverrideCursor()

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

    def updateEraserCursor(self, x, y, xyLocked=None, isHoverImg1=True):
        if x is None:
            return

        xdata, ydata = int(x), int(y)
        _img = self.currentLab2D
        Y, X = _img.shape

        if not (xdata >= 0 and xdata < X and ydata >= 0 and ydata < Y):
            return

        size = self.brushSizeSpinbox.value()*2
        self.setHoverToolSymbolData(
            [x], [y], self.activeEraserCircleCursors(isHoverImg1),
            size=size
        )
        self.setHoverToolSymbolData(
            [x], [y], self.activeEraserXCursors(isHoverImg1),
            size=int(size/2)
        )

        isMouseDrag = (
            self.isMouseDragImg1 or self.isMouseDragImg2
        )
        if isMouseDrag:
            return
        
        if xyLocked is not None:
            xdata, ydata = xyLocked

        self.setHoverToolSymbolColor(
            xdata, ydata, self.eraserCirclePen,
            self.activeEraserCircleCursors(isHoverImg1),
            self.eraserButton, hoverRGB=None
        )

    def resetCursor(self):
        if self.app.overrideCursor() is not None:
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()

    def useCenterBrushCursorHoverIDtoggled(self, checked):
        if checked:
            self.df_settings.at['useCenterBrushCursorHoverID', 'value'] = 'Yes'
        else:
            self.df_settings.at['useCenterBrushCursorHoverID', 'value'] = 'No'
        self.df_settings.to_csv(self.settings_csv_path)


class HoverEventsGuiElements:
    """Hover event handlers of the left and right images and the hovered-ID
    lookup (temporary mother-bud and merge lines, mouse position of the
    items).
    """

    def gui_hoverEventRightImage(self, event):
        try:
            posData = self.data[self.pos_i]
        except AttributeError:
            return

        if event.isExit():
            self.resetCursors()

        self.gui_hoverEventImg1(event, isHoverImg1=False)
        setMirroredCursor = (
            not event.isExit() and self.showMirroredCursorAction.isChecked()
        )
        if setMirroredCursor:
            x, y = event.pos()
            self.ax1_cursor.setData([x], [y])

    def gui_hoverEventImg1(self, event, isHoverImg1=True):
        try:
            posData = self.data[self.pos_i]
        except AttributeError:
            return

        self._cursorHoverImage = None if event.isExit() else isHoverImg1
        
        # Update x, y, value label bottom right
        if not event.isExit():
            self.xHoverImg, self.yHoverImg = event.pos()
        else:
            self.xHoverImg, self.yHoverImg = None, None

        if event.isExit():
            self.resetCursor()
            
        if not event.isExit() and self.slideshowWin is not None:
            self.slideshowWin.setMirroredCursorPos(*event.pos())

        setMirroredCursor3D = (
            not event.isExit() and self.showMirroredCursorAction.isChecked()
        )
        if setMirroredCursor3D:
            x, y = event.pos()
            self.update3DViewerCursor((x, y))
        else:
            self.update3DViewerCursor(None)
        
        # Alt key was released --> restore cursor
        modifiers = QGuiApplication.keyboardModifiers()
        cursorsInfo = self.gui_setCursor(
            modifiers, event, isHoverImg1=isHoverImg1
        )
        self.highlightHoverLostObj(modifiers, event)
        
        drawRulerLine = (
            (self.rulerButton.isChecked() 
            or self.addDelPolyLineRoiButton.isChecked())
            and self.tempSegmentON and not event.isExit()
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
            self.wcLabel.setText('')
        
        if cursorsInfo['setDelRoiCursor']:
            self.setDelRoiCursor()

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
        if cursorsInfo['setBrushStrokesCursor']:
            x, y = event.pos()
            radius = self.magicPromptsToolbar.brushSizeSpinbox.value()
            self.setHoverToolSymbolData(
                [x], [y], self.activeBrushCircleCursors(isHoverImg1),
                size=2*radius
            )
        elif cursorsInfo['setBrushCursor']:
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

        # Temporarily draw spline curve
        # see https://stackoverflow.com/questions/33962717/interpolating-a-closed-curve-using-scipy
        drawSpline = (
            self.curvToolButton.isChecked() and self.splineHoverON
            and not event.isExit()
        )
        if drawSpline:
            self.hoverEventDrawSpline(event)
        
        setMirroredCursor = (
            not event.isExit() and isHoverImg1
            and self.showMirroredCursorAction.isChecked()
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
            centroid = posData.rp.get_centroid(ID)
            y2, x2 = self.getObjCentroid(centroid)
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
            centroid = posData.rp.get_centroid(ID)
            y2, x2 = self.getObjCentroid(centroid)
        
        if modifier and ID > 0:
            self.mergeObjsTempLine.addPoint(x2, y2)
        elif not modifier:
            self.mergeObjsTempLine.setData([x1, x2], [y1, y2])

    def gui_hoverEventImg2(self, event):
        try:
            posData = self.data[self.pos_i]
        except AttributeError:
            return

        self._cursorHoverImage = None if event.isExit() else False
            
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
            self.brushButton.isChecked() and not event.isExit()
            and (noModifier or shift or ctrl)
        )
        setEraserCursor = (
            self.eraserButton.isChecked() and not event.isExit()
            and noModifier
        )
        setLabelRoiCircCursor = (
            self.labelRoiButton.isChecked() and not event.isExit()
            and (noModifier or shift or ctrl)
            and self.labelRoiIsCircularRadioButton.isChecked()
        )

        setMoveLabelCursor = (
            self.moveLabelToolButton.isChecked() and not event.isExit()
            and noModifier
        )

        setExpandLabelCursor = (
            self.expandLabelToolButton.isChecked() and not event.isExit()
            and noModifier
        )

        # Cursor is moving on image while Alt key is pressed --> pan cursor
        alt = QGuiApplication.keyboardModifiers() == Qt.AltModifier
        setPanImageCursor = alt and not event.isExit()
        if setPanImageCursor and self.app.overrideCursor() is None:
            self.app.setOverrideCursor(Qt.SizeAllCursor)
        
        setKeepObjCursor = (
            self.keepIDsButton.isChecked() and not event.isExit()
            and noModifier
        )
        if not setPanImageCursor and not event.isExit():
            self.gui_setToolCursor(event, isHoverImg1=False)

        # Update x, y, value label bottom right
        if not event.isExit():
            x, y = event.pos()
            xdata, ydata = int(x), int(y)
            _img = self.currentLab2D
            Y, X = _img.shape                
            # hoverText = self.hoverValuesFormatted(xdata, ydata)
            # self.wcLabel.setText(hoverText)
        else:
            if self.eraserButton.isChecked() or self.brushButton.isChecked():
                self.gui_mouseReleaseEventImg2(event)
            self.wcLabel.setText(f'')

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

    def getLastHoveredID(self):
        if self.xHoverImg is None:
            return 0
        
        xdata, ydata = int(self.xHoverImg), int(self.yHoverImg)
        ID = self.currentLab2D[ydata, xdata]
        return ID

    def getHoverID(self, xdata, ydata, byPassShiftCheck=False):
        if not hasattr(self, 'diskMask'):
            return 0
        
        modifiers = QGuiApplication.keyboardModifiers()
        ctrl = modifiers == Qt.ControlModifier
        if byPassShiftCheck:
            shift = False
        else:
            shift = modifiers == Qt.ShiftModifier

        if self.isPowerBrush() and not ctrl:
            return 0        

        if not self.autoIDcheckbox.isChecked():
            return self.editIDspinbox.value()

        ymin, xmin, ymax, xmax, diskMask = self.getDiskMask(xdata, ydata)
        posData = self.data[self.pos_i]
        lab_2D = self.get_2Dlab(posData.lab, force_z=False)
        ID = lab_2D[ydata, xdata]
        self.isHoverZneighID = False
        if self.isSegm3D:
            zProjHow = self.zProjComboBox.currentText()
            isZslice = zProjHow == 'single z-slice'
            z = self.z_lab()
            SizeZ = posData.lab.shape[0]
            doNotLinkThroughZ = self.brushButton.isChecked() and shift
            if not isZslice:
                # In projection mode, ID comes from the projected 2D label image.
                if self.brushHoverCenterModeAction.isChecked() or ID>0:
                    hoverID = ID
                else:
                    masked_lab = lab_2D[ymin:ymax, xmin:xmax][diskMask]
                    hoverID = np.bincount(masked_lab).argmax()
            elif doNotLinkThroughZ:
                if self.brushHoverCenterModeAction.isChecked() or ID>0:
                    hoverID = ID
                else:
                    masked_lab = lab_2D[ymin:ymax, xmin:xmax][diskMask]
                    hoverID = np.bincount(masked_lab).argmax()
            else:
                if z > 0:
                    ID_z_under = posData.lab[z-1, ydata, xdata]
                    if self.brushHoverCenterModeAction.isChecked() or ID_z_under>0:
                        hoverIDa = ID_z_under
                    else:
                        lab = posData.lab
                        masked_lab_a = lab[z-1, ymin:ymax, xmin:xmax][diskMask]
                        hoverIDa = np.bincount(masked_lab_a).argmax()
                else:
                    hoverIDa = 0

                if self.brushHoverCenterModeAction.isChecked() or ID>0:
                    hoverIDb = lab_2D[ydata, xdata]
                else:
                    masked_lab_b = lab_2D[ymin:ymax, xmin:xmax][diskMask]
                    hoverIDb = np.bincount(masked_lab_b).argmax()

                if z < SizeZ-1:
                    ID_z_above = posData.lab[z+1, ydata, xdata]
                    if self.brushHoverCenterModeAction.isChecked() or ID_z_above>0:
                        hoverIDc = ID_z_above
                    else:
                        lab = posData.lab
                        masked_lab_c = lab[z+1, ymin:ymax, xmin:xmax][diskMask]
                        hoverIDc = np.bincount(masked_lab_c).argmax()
                else:
                    hoverIDc = 0

                # When clicking directly on an object, prefer current-slice ID.
                if hoverIDb > 0:
                    hoverID = hoverIDb
                elif hoverIDa > 0:
                    hoverID = hoverIDa
                    self.isHoverZneighID = True
                elif hoverIDc > 0:
                    hoverID = hoverIDc
                    self.isHoverZneighID = True
                else:
                    hoverID = 0
        else:
            if self.brushButton.isChecked() and shift:
                # Force new ID with brush and Shift
                hoverID = 0
            elif self.brushHoverCenterModeAction.isChecked() or ID>0:
                hoverID = ID
            else:
                masked_lab = lab_2D[ymin:ymax, xmin:xmax][diskMask]
                hoverID = np.bincount(masked_lab).argmax()
            
        self.editIDspinbox.setValue(hoverID)

        return hoverID

    def updateItemsMousePos(self):
        if self.brushButton.isChecked():
            self.updateBrushCursor(self.xHoverImg, self.yHoverImg)

        if self.eraserButton.isChecked():
            self.updateEraserCursor(self.xHoverImg, self.yHoverImg)
