"""View adapter for canvas context menus and deleted-ROI clicks."""

from __future__ import annotations

import pyqtgraph as pg
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QAction, QMenu


class CanvasContextMenuMixin:
    """Qt-facing adapter around canvas context-menu contracts."""

    """Headless canvas context-menu decision rules."""

    scale_bar_target = "scale_bar"
    timestamp_target = "timestamp"
    gradient_target = "gradient"

    def _scale_bar_highlighted(self):
        return hasattr(self, "scaleBar") and self.scaleBar.isHighlighted()

    def _show_deleted_roi_context_menu(self, event):
        self.roiContextMenu = QMenu(self)
        separator = QAction(self)
        separator.setSeparator(True)
        self.roiContextMenu.addAction(separator)
        action = QAction("Remove ROI")
        action.triggered.connect(self.removeDelROI)
        self.roiContextMenu.addAction(action)
        try:
            screen_pos = event.screenPos().toPoint()
        except AttributeError:
            screen_pos = event.screenPos()
        self.roiContextMenu.exec_(screen_pos)

    def _timestamp_highlighted(self):
        return hasattr(self, "timestamp") and self.timestamp.isHighlighted()

    def clicked_deleted_roi(self, event, left_click, right_click):
        pos_data = self.data[self.pos_i]
        x, y = event.pos().x(), event.pos().y()

        del_rois = pos_data.allData_li[pos_data.frame_i]["delROIs_info"]["rois"].copy()
        for roi in del_rois:
            roi_mask = self.getDelRoiMask(roi)
            if self.isSegm3D:
                clicked_on_roi = roi_mask[self.z_lab(), int(y), int(x)]
            else:
                clicked_on_roi = roi_mask[int(y), int(x)]
            decision = self.deleted_roi_click_decision(
                clicked_on_roi=clicked_on_roi,
                left_click=left_click,
                right_click=right_click,
            )
            if decision.show_context_menu:
                self.roi_to_del = roi
                self._show_deleted_roi_context_menu(event)
                return True
            if decision.drag_roi:
                event.ignore()
                return True
        return False

    def deleted_roi_click_decision(
        self,
        *,
        clicked_on_roi: bool,
        left_click: bool,
        right_click: bool,
    ) -> DeletedRoiClickDecision:
        if not clicked_on_roi:
            return DeletedRoiClickDecision(handled=False)
        if right_click:
            return DeletedRoiClickDecision(
                handled=True,
                show_context_menu=True,
            )
        if left_click:
            return DeletedRoiClickDecision(handled=True, drag_roi=True)
        return DeletedRoiClickDecision(handled=False)

    def hovered_handles_polyline_roi(self):
        pos_data = self.data[self.pos_i]
        del_rois_info = pos_data.allData_li[pos_data.frame_i]["delROIs_info"]
        handles = []
        for roi in del_rois_info["rois"]:
            if not isinstance(roi, pg.PolyLineROI):
                continue
            for handle in roi.getHandles():
                if handle.currentPen == handle.hoverPen:
                    handle.roi = roi
                    handles.append(handle)
        return handles

    def hovered_segments_polyline_roi(self):
        pos_data = self.data[self.pos_i]
        del_rois_info = pos_data.allData_li[pos_data.frame_i]["delROIs_info"]
        segments = []
        for roi in del_rois_info["rois"]:
            if not isinstance(roi, pg.PolyLineROI):
                continue
            for segment in roi.segments:
                if segment.currentPen == segment.hoverPen:
                    segment.roi = roi
                    segments.append(segment)
        return segments

    def image_gradient_menu_target(
        self,
        *,
        scale_bar_highlighted: bool,
        timestamp_highlighted: bool,
    ) -> str:
        if scale_bar_highlighted:
            return self.scale_bar_target
        if timestamp_highlighted:
            return self.timestamp_target
        return self.gradient_target

    def show_img_gradient_context_menu(self, x, y):
        target = self.image_gradient_menu_target(
            scale_bar_highlighted=self._scale_bar_highlighted(),
            timestamp_highlighted=self._timestamp_highlighted(),
        )
        if target == self.scale_bar_target:
            self.scaleBar.showContextMenu(x, y)
            return
        if target == self.timestamp_target:
            self.timestamp.showContextMenu(x, y)
            return
        self.imgGrad.gradient.menu.popup(QPoint(int(x), int(y)))

    def show_right_image_context_menu(self, event):
        try:
            screen_pos = event.screenPos().toPoint()
        except AttributeError:
            screen_pos = event.screenPos()
        self.imgGradRight.gradient.menu.popup(screen_pos)
