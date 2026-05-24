"""View adapter for canvas context menus and deleted-ROI clicks."""

from __future__ import annotations

import pyqtgraph as pg
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QAction, QMenu

from cellacdc.viewmodels.canvas_context_menu_viewmodel import (
    CanvasContextMenuViewModel,
)


class CanvasContextMenuView:
    """Qt-facing adapter around canvas context-menu contracts."""

    def __init__(self, host, view_model: CanvasContextMenuViewModel):
        self.host = host
        self.view_model = view_model

    def show_img_gradient_context_menu(self, x, y):
        target = self.view_model.image_gradient_menu_target(
            scale_bar_highlighted=self._scale_bar_highlighted(),
            timestamp_highlighted=self._timestamp_highlighted(),
        )
        if target == self.view_model.scale_bar_target:
            self.host.scaleBar.showContextMenu(x, y)
            return
        if target == self.view_model.timestamp_target:
            self.host.timestamp.showContextMenu(x, y)
            return
        self.host.imgGrad.gradient.menu.popup(QPoint(int(x), int(y)))

    def show_right_image_context_menu(self, event):
        try:
            screen_pos = event.screenPos().toPoint()
        except AttributeError:
            screen_pos = event.screenPos()
        self.host.imgGradRight.gradient.menu.popup(screen_pos)

    def clicked_deleted_roi(self, event, left_click, right_click):
        pos_data = self.host.data[self.host.pos_i]
        x, y = event.pos().x(), event.pos().y()

        del_rois = (
            pos_data.allData_li[pos_data.frame_i]['delROIs_info']['rois']
            .copy()
        )
        for roi in del_rois:
            roi_mask = self.host.getDelRoiMask(roi)
            if self.host.isSegm3D:
                clicked_on_roi = roi_mask[
                    self.host.z_lab(), int(y), int(x)
                ]
            else:
                clicked_on_roi = roi_mask[int(y), int(x)]
            decision = self.view_model.deleted_roi_click_decision(
                clicked_on_roi=clicked_on_roi,
                left_click=left_click,
                right_click=right_click,
            )
            if decision.show_context_menu:
                self.host.roi_to_del = roi
                self._show_deleted_roi_context_menu(event)
                return True
            if decision.drag_roi:
                event.ignore()
                return True
        return False

    def hovered_segments_polyline_roi(self):
        pos_data = self.host.data[self.host.pos_i]
        del_rois_info = (
            pos_data.allData_li[pos_data.frame_i]['delROIs_info']
        )
        segments = []
        for roi in del_rois_info['rois']:
            if not isinstance(roi, pg.PolyLineROI):
                continue
            for segment in roi.segments:
                if segment.currentPen == segment.hoverPen:
                    segment.roi = roi
                    segments.append(segment)
        return segments

    def hovered_handles_polyline_roi(self):
        pos_data = self.host.data[self.host.pos_i]
        del_rois_info = (
            pos_data.allData_li[pos_data.frame_i]['delROIs_info']
        )
        handles = []
        for roi in del_rois_info['rois']:
            if not isinstance(roi, pg.PolyLineROI):
                continue
            for handle in roi.getHandles():
                if handle.currentPen == handle.hoverPen:
                    handle.roi = roi
                    handles.append(handle)
        return handles

    def _show_deleted_roi_context_menu(self, event):
        self.host.roiContextMenu = QMenu(self.host)
        separator = QAction(self.host)
        separator.setSeparator(True)
        self.host.roiContextMenu.addAction(separator)
        action = QAction('Remove ROI')
        action.triggered.connect(self.host.removeDelROI)
        self.host.roiContextMenu.addAction(action)
        try:
            screen_pos = event.screenPos().toPoint()
        except AttributeError:
            screen_pos = event.screenPos()
        self.host.roiContextMenu.exec_(screen_pos)

    def _scale_bar_highlighted(self):
        return (
            hasattr(self.host, 'scaleBar')
            and self.host.scaleBar.isHighlighted()
        )

    def _timestamp_highlighted(self):
        return (
            hasattr(self.host, 'timestamp')
            and self.host.timestamp.isHighlighted()
        )
