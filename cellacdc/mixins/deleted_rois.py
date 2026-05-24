"""Qt view adapter for deleted-ROI workflows."""

from __future__ import annotations

from functools import partial
import uuid

import numpy as np
import pyqtgraph as pg
from collections.abc import Iterable
import skimage.measure
from qtpy.QtCore import QRect, QRectF, QTimer

from cellacdc import widgets


class DeletedRoisView:
    """Qt-facing adapter around deleted-ROI workflows."""

    """Headless decisions for deleted-ROI display and propagation."""

    def roi_axis(
        self,
        *,
        is_polyline: bool,
        labels_image_visible: bool,
    ) -> str:
        if is_polyline or not labels_image_visible:
            return 'left'
        return 'right'

    def should_render_deleted_roi(self, annotation_mode: str) -> bool:
        return 'nothing' not in annotation_mode

    def should_render_deleted_roi_contours(self, annotation_mode: str) -> bool:
        return 'contours' in annotation_mode

    def should_render_deleted_roi_overlay(self, annotation_mode: str) -> bool:
        return 'overlay segm. masks' in annotation_mode

    def should_initialize_overlay_masks(
        self,
        init: bool,
        annotation_mode: str,
    ) -> bool:
        return init and not self.should_render_deleted_roi_contours(
            annotation_mode
        )

    def labels_to_skip(self, deleted_ids: Iterable[int]) -> dict[int, bool]:
        return {deleted_id: True for deleted_id in deleted_ids}


    LEGACY_METHODS = (
        'removeAlldelROIsCurrentFrame',
        'removeDelROI',
        'removeDelROIFromFutureFrames',
        'updateDelROIinFutureFrames',
        'addDelROI',
        'replacePolyLineRoiWithLineRoi',
        'addRoiToDelRoiInfo',
        'addDelPolyLineRoi_cb',
        'createDelPolyLineRoi',
        'addPointsPolyLineRoi',
        'createDelROI',
        'delROIstartedMoving',
        'clearLostObjContoursItems',
        'delROImoving',
        'delROImovingFinished',
        'restoreAnnotDelROI',
        'restoreDelROIimg1',
        'getDelRoisIDs',
        'getStoredDelRoiIDs',
        'getDelROIlab',
        'getDelRoiMask',
        'initDelRoiLab',
        'moveDelRoisToLeft',
        'applyDelROIimg1',
        'applyDelROIs',
        'setDelRoiState',
        'addExistingDelROIs',
    )

    def __init__(self, host):
        object.__setattr__(self, 'host', host)
    def __getattr__(self, name):
        return getattr(self.host, name)

    def __setattr__(self, name, value):
        if name in {'host'}:
            object.__setattr__(self, name, value)
        else:
            setattr(self.host, name, value)

    def bind_legacy_methods(self):
        for name in self.LEGACY_METHODS:
            setattr(self.host, name, getattr(self, name))

    def removeAlldelROIsCurrentFrame(self):
        posData = self.data[self.pos_i]
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        rois = delROIs_info['rois'].copy()
        for roi in rois:
            self.ax2.removeDelRoiItem(roi)

        for item in self.ax2.items:
            if isinstance(item, pg.ROI):
                self.ax2.removeDelRoiItem(item)

        for item in self.ax1.items:
            if isinstance(item, pg.ROI) and item != self.labelRoiItem:
                self.ax1.removeDelRoiItem(item)

    def removeDelROI(self, event):
        posData = self.data[self.pos_i]

        for ax in (self.ax1, self.ax2):
            try:
                self.ax1.removeDelRoiItem(self.roi_to_del)
            except Exception as err:
                pass

        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        idx = delROIs_info['rois'].index(self.roi_to_del)
        delROIs_info['rois'].pop(idx)
        delROIs_info['delMasks'].pop(idx)
        delROIs_info['delIDsROI'].pop(idx)
        delROIs_info['state'].pop(idx)

        self.removeDelROIFromFutureFrames(self.roi_to_del)
        self.updateAllImages()

    def removeDelROIFromFutureFrames(self, roi_to_del):
        posData = self.data[self.pos_i]

        # Restore deleted IDs from already visited future frames
        current_frame_i = posData.frame_i
        for i in range(posData.frame_i+1, posData.SizeT):
            if posData.allData_li[i]['labels'] is None:
                break

            delROIs_info = posData.allData_li[i]['delROIs_info']
            try:
                idx = delROIs_info['rois'].index(roi_to_del)
            except IndexError:
                continue

            posData.frame_i = i
            idx = delROIs_info['rois'].index(roi_to_del)
            if delROIs_info['delIDsROI'][idx]:
                posData.lab = posData.allData_li[i]['labels']
                self.restoreAnnotDelROI(roi_to_del, enforce=True, draw=False)
                posData.allData_li[i]['labels'] = posData.lab
                self.get_data()
                self.store_data(autosave=False)
            delROIs_info['rois'].pop(idx)
            delROIs_info['delMasks'].pop(idx)
            delROIs_info['delIDsROI'].pop(idx)
            delROIs_info['state'].pop(idx)

        target_axis = self.roi_axis(
            is_polyline=isinstance(self.roi_to_del, pg.PolyLineROI),
            labels_image_visible=self.labelsGrad.showLabelsImgAction.isChecked(),
        )
        if target_axis == 'left':
            self.ax1.removeItem(self.roi_to_del)
        else:
            self.ax2.removeItem(self.roi_to_del)

        # Back to current frame
        posData.frame_i = current_frame_i
        posData.lab = posData.allData_li[posData.frame_i]['labels']
        self.get_data()
        self.store_data()

    def updateDelROIinFutureFrames(self, roi: pg.ROI):
        posData = self.data[self.pos_i]
        restore_current_frame = False

        roiState = roi.getState()
        # Restore deleted IDs from already visited future frames
        current_frame_i = posData.frame_i
        delROIs_info = posData.allData_li[current_frame_i]['delROIs_info']
        try:
            idx = delROIs_info['rois'].index(roi)
            delROIs_info['state'][idx] = roiState
        except Exception as err:
            pass

        self.store_data()

        for i in range(posData.frame_i+1, posData.SizeT):
            delROIs_info = posData.allData_li[i]['delROIs_info']
            try:
                idx = delROIs_info['rois'].index(roi)
            except Exception as err:
                continue
            delROIs_info['state'][idx] = roiState
            if posData.allData_li[i]['labels'] is None:
                continue

            posData.frame_i = i
            posData.lab = posData.allData_li[i]['labels']
            self.restoreAnnotDelROI(roi, enforce=False, draw=False)
            posData.allData_li[i]['labels'] = posData.lab
            self.get_data()
            self.store_data(autosave=False)
            restore_current_frame = True

        if not restore_current_frame:
            return

        # Back to current frame
        posData.frame_i = current_frame_i
        posData.lab = posData.allData_li[posData.frame_i]['labels']
        self.get_data()
        self.store_data()

    def addDelROI(self, event):
        roi, key = self.createDelROI()
        self.addRoiToDelRoiInfo(roi)
        target_axis = self.roi_axis(
            is_polyline=False,
            labels_image_visible=self.labelsGrad.showLabelsImgAction.isChecked(),
        )
        if target_axis == 'left':
            self.ax1.addDelRoiItem(roi, key)
        else:
            self.ax2.addDelRoiItem(roi, key)
        self.applyDelROIimg1(roi, init=True)
        self.applyDelROIimg1(roi, init=True, ax=1)

        if self.isSnapshot:
            self.fixCcaDfAfterEdit('Delete IDs using ROI')
            self.updateAllImages()
        else:
            self.warnEditingWithCca_df(
                'Delete IDs using ROI', get_cancelled=True
            )

    def replacePolyLineRoiWithLineRoi(self, roi):
        x0, y0 = roi.pos().x(), roi.pos().y()
        (_, point1), (_, point2) = roi.getLocalHandlePositions()
        xr1, yr1 = point1.x(), point1.y()
        xr2, yr2 = point2.x(), point2.y()
        x1, y1 = xr1+x0, yr1+y0
        x2, y2 = xr2+x0, yr2+x0
        lineRoi = pg.LineROI((x1, y1), (x2, y2), width=0.5)
        lineRoi.handleSize = 7
        self.ax1.removeItem(self.polyLineRoi)
        self.ax1.addItem(lineRoi)
        lineRoi.removeHandle(2)
        # Connect closed ROI
        lineRoi.sigRegionChanged.connect(self.delROImoving)
        lineRoi.sigRegionChangeFinished.connect(self.delROImovingFinished)
        return lineRoi

    def addRoiToDelRoiInfo(self, roi: pg.ROI):
        posData = self.data[self.pos_i]
        for i in range(posData.frame_i, posData.SizeT):
            delROIs_info = posData.allData_li[i]['delROIs_info']
            delROIs_info['rois'].append(roi)
            delROIs_info['state'].append(roi.getState())
            delROIs_info['delMasks'].append(np.zeros_like(self.currentLab2D))
            delROIs_info['delIDsROI'].append(set())

    def addDelPolyLineRoi_cb(self, checked):
        if checked:
            self.disconnectLeftClickButtons()
            self.uncheckLeftClickButtons(self.addDelPolyLineRoiButton)
            self.connectLeftClickButtons()
            if self.isSnapshot:
                self.fixCcaDfAfterEdit('Delete IDs using ROI')
                self.updateAllImages()
            else:
                self.warnEditingWithCca_df('Delete IDs using ROI')
        else:
            self.tempSegmentON = False
            self.ax1_rulerPlotItem.setData([], [])
            self.ax1_rulerAnchorsItem.setData([], [])
            self.startPointPolyLineItem.setData([], [])
            while self.app.overrideCursor() is not None:
                self.app.restoreOverrideCursor()

    def createDelPolyLineRoi(self):
        Y, X = self.currentLab2D.shape
        self.polyLineRoi = pg.PolyLineROI(
            [], rotatable=False,
            removable=True,
            pen=pg.mkPen(color='r')
        )
        self.polyLineRoi.handleSize = 7
        self.polyLineRoi.points = []
        key = uuid.uuid4()
        self.ax1.addDelRoiItem(self.polyLineRoi, key)

    def addPointsPolyLineRoi(self, closed=False):
        self.polyLineRoi.setPoints(self.polyLineRoi.points, closed=closed)
        if not closed:
            return

        # Connect closed ROI
        self.polyLineRoi.sigRegionChanged.connect(self.delROImoving)
        self.polyLineRoi.sigRegionChangeFinished.connect(self.delROImovingFinished)

    def createDelROI(self, xl=None, yb=None, w=32, h=32, anchors=None):
        posData = self.data[self.pos_i]
        if xl is None:
            xRange, yRange = self.ax1.viewRange()
            xl = 0 if xRange[0] < 0 else xRange[0]
            yb = 0 if yRange[0] < 0 else yRange[0]
        Y, X = self.currentLab2D.shape
        if anchors is None:
            roi = widgets.DelROI(
                [xl, yb], [w, h],
                rotatable=False,
                removable=True,
                pen=pg.mkPen(color='r'),
                maxBounds=QRectF(QRect(0,0,X,Y))
            )
            ## handles scaling horizontally around center
            roi.addScaleHandle([1, 0.5], [0, 0.5])
            roi.addScaleHandle([0, 0.5], [1, 0.5])

            ## handles scaling vertically from opposite edge
            roi.addScaleHandle([0.5, 0], [0.5, 1])
            roi.addScaleHandle([0.5, 1], [0.5, 0])

            ## handles scaling both vertically and horizontally
            roi.addScaleHandle([1, 1], [0, 0])
            roi.addScaleHandle([0, 0], [1, 1])
            roi.addScaleHandle([0, 1], [1, 0])
            roi.addScaleHandle([1, 0], [0, 1])

        roi.handleSize = 7
        roi.sigRegionChanged.connect(self.delROImoving)
        roi.sigRegionChanged.connect(self.delROIstartedMoving)
        roi.sigRegionChangeFinished.connect(self.delROImovingFinished)

        key = uuid.uuid4()

        return roi, key

    def delROIstartedMoving(self, roi):
        self.clearLostObjContoursItems()

    def clearLostObjContoursItems(self):
        self.ax1_lostObjScatterItem.setData([], [])
        self.ax2_lostObjScatterItem.setData([], [])

        self.ax1_lostTrackedScatterItem.setData([], [])
        self.ax2_lostTrackedScatterItem.setData([], [])

        self.ax2_lostObjImageItem.clear()
        self.ax2_lostTrackedObjImageItem.clear()

        self.ax1_lostObjImageItem.clear()
        self.ax1_lostTrackedObjImageItem.clear()

    def delROImoving(self, roi):
        roi.setPen(color=(255,255,0))
        # First bring back IDs if the ROI moved away
        self.restoreAnnotDelROI(roi)
        self.setImageImg2()
        self.applyDelROIimg1(roi)
        self.applyDelROIimg1(roi, ax=1)

    def delROImovingFinished(self, roi: pg.ROI):
        roi.setPen(color='r')
        self.update_rp()
        self.updateAllImages()
        QTimer.singleShot(
            300, partial(self.updateDelROIinFutureFrames, roi)
        )

    def restoreAnnotDelROI(self, roi, enforce=True, draw=True):
        posData = self.data[self.pos_i]
        ROImask = self.getDelRoiMask(roi)
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        try:
            idx = delROIs_info['rois'].index(roi)
        except Exception as err:
            return

        delMask = delROIs_info['delMasks'][idx]
        delIDs = delROIs_info['delIDsROI'][idx]
        lab2D = self.get_2Dlab(posData.lab)
        result = self.host.view_model.label_edits.restore_deleted_roi_labels(
            lab2D,
            self.currentLab2D,
            delMask,
            ROImask,
            delIDs,
            enforce=enforce,
        )
        if draw:
            for ID, delMaskID in result.restored_masks:
                self.restoreDelROIimg1(delMaskID, ID, ax=0)
                self.restoreDelROIimg1(delMaskID, ID, ax=1)

        delROIs_info['delIDsROI'][idx] = result.remaining_deleted_ids
        self.set_2Dlab(result.labels_2d)
        self.update_rp()

    def restoreDelROIimg1(self, delMaskID, delID, ax=0):
        if ax == 0:
            how = self.drawIDsContComboBox.currentText()
        else:
            how = self.getAnnotateHowRightImage()

        if not self.should_render_deleted_roi(how):
            return

        if self.should_render_deleted_roi_contours(how):
            rp_delmask = skimage.measure.regionprops(delMaskID.astype(np.uint8))
            if len(rp_delmask) > 0:
                obj = rp_delmask[0]
                self.addObjContourToContoursImage(obj=obj, ax=ax)
        elif self.should_render_deleted_roi_overlay(how):
            if ax == 0:
                self.labelsLayerImg1.setImage(
                    self.currentLab2D, autoLevels=False
                )
            else:
                self.labelsLayerRightImg.setImage(
                    self.currentLab2D, autoLevels=False
                )

    def getDelRoisIDs(self):
        posData = self.data[self.pos_i]
        roi_masks = []
        if posData.frame_i > 0:
            prev_lab = posData.allData_li[posData.frame_i-1]['labels']
        for roi in posData.allData_li[posData.frame_i]['delROIs_info']['rois']:
            if (
                    not self.ax1.isDelRoiItemPresent(roi)
                    and not self.ax2.isDelRoiItemPresent(roi)
                ):
                continue
            roi_masks.append(self.getDelRoiMask(roi))

        return self.host.view_model.label_edits.label_ids_in_masks(
            posData.lab,
            roi_masks,
            additional_labels=prev_lab if posData.frame_i > 0 else None,
        )

    def getStoredDelRoiIDs(self, frame_i=None):
        posData = self.data[self.pos_i]
        if frame_i is None:
            frame_i = posData.frame_i
        delROIs_info = posData.allData_li[frame_i]['delROIs_info']
        return self.host.view_model.label_edits.collect_deleted_roi_ids(
            delROIs_info['delIDsROI']
        )

    # @exec_time
    def getDelROIlab(self, input_lab_2D=None):
        posData = self.data[self.pos_i]
        if self.delRoiLab is None:
            self.initDelRoiLab()

        out_lab = self.delRoiLab
        if input_lab_2D is None:
            out_lab[:] = self.get_2Dlab(posData.lab, force_z=False)
        else:
            out_lab[:] = input_lab_2D

        roi_masks = []
        deleted_masks = []
        deleted_ids_by_roi = []
        roi_indices = []
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        for roi in posData.allData_li[posData.frame_i]['delROIs_info']['rois']:
            if (
                    not self.ax1.isDelRoiItemPresent(roi)
                    and not self.ax2.isDelRoiItemPresent(roi)
                ):
                continue
            idx = delROIs_info['rois'].index(roi)
            roi_indices.append(idx)
            roi_masks.append(self.getDelRoiMask(roi))
            deleted_masks.append(delROIs_info['delMasks'][idx])
            deleted_ids_by_roi.append(delROIs_info['delIDsROI'][idx])

        result = self.host.view_model.label_edits.apply_deleted_roi_masks(
            out_lab,
            roi_masks,
            deleted_masks,
            deleted_ids_by_roi,
        )
        for result_i, roi_i in enumerate(roi_indices):
            # Keep a mask of deleted IDs to bring them back when ROI moves.
            delROIs_info['delMasks'][roi_i] = result.deleted_masks[result_i]
            delROIs_info['delIDsROI'][roi_i] = (
                result.deleted_ids_by_roi[result_i]
            )

        # printl(
        #     f't1-t0: {(t1-t0)*1000:.3f} ms,',
        #     f't2-t1: {(t2-t1)*1000:.3f} ms,',
        #     f't3-t2: {(t3-t2)*1000:.3f} ms,',
        #     # f't4-t3: {(t4-t3)*1000:.3f} ms,',
        #     # f't5-t4: {(t5-t4)*1000:.3f} ms,',
        #     # f't6-t5: {(t6-t5)*1000:.3f} ms',
        #     sep='\n'
        # )

        return result.deleted_ids, result.labels_2d

    def getDelRoiMask(self, roi, posData=None, z_slice=None):
        if posData is None:
            posData = self.data[self.pos_i]
        if z_slice is None:
            z_slice = self.z_lab()
        if isinstance(roi, pg.PolyLineROI):
            x0, y0 = roi.pos().x(), roi.pos().y()
            points = []
            for _, point in roi.getLocalHandlePositions():
                xr, yr = point.x(), point.y()
                points.append((int(xr+x0), int(yr+y0)))
            return self.host.view_model.label_edits.polygon_roi_mask(
                posData.lab.shape,
                points,
                z_slice=z_slice,
            )
        elif isinstance(roi, pg.LineROI):
            (_, point1), (_, point2) = roi.getSceneHandlePositions()
            point1 = self.ax1.vb.mapSceneToView(point1)
            point2 = self.ax1.vb.mapSceneToView(point2)
            return self.host.view_model.label_edits.line_roi_mask(
                posData.lab.shape,
                (point1.x(), point1.y()),
                (point2.x(), point2.y()),
                z_slice=z_slice,
            )
        else:
            return self.host.view_model.label_edits.rectangle_roi_mask(
                posData.lab.shape,
                roi.pos(),
                roi.size(),
                z_slice=z_slice,
            )

    def initDelRoiLab(self):
        posData = self.data[self.pos_i]
        z_slice = self.z_lab()
        img = posData.img_data[posData.frame_i]
        Y, X = img[z_slice].shape[-2:]

        self.delRoiLab = np.zeros((Y, X), dtype=np.uint32)

    def moveDelRoisToLeft(self):
        # Move del ROIs to the left image
        for posData in self.data:
            delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
            for roi in delROIs_info['rois']:
                if not self.ax2.isDelRoiItemPresent(roi):
                    continue

                self.ax1.addDelRoiItem(roi, roi.key)
                self.ax2.removeDelRoiItem(roi)

    def applyDelROIimg1(self, roi, init=False, ax=0):
        if ax == 0:
            how = self.drawIDsContComboBox.currentText()
        else:
            how = self.getAnnotateHowRightImage()

        if ax == 1 and not self.labelsGrad.showRightImgAction.isChecked():
            return

        if self.should_initialize_overlay_masks(init, how):
            self.setOverlaySegmMasks(force=True)
            return

        posData = self.data[self.pos_i]
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        try:
            idx = delROIs_info['rois'].index(roi)
        except Exception as err:
            try:
                ax.removeDelRoiItem(roi)
            except Exception as err:
                pass
            return
        delIDs = delROIs_info['delIDsROI'][idx]
        delMask = delROIs_info['delMasks'][idx]
        if not self.should_render_deleted_roi(how):
            return
        elif self.should_render_deleted_roi_contours(how):
            self.updateContoursImage(ax=ax)

        if not delIDs:
            return

        if self.should_render_deleted_roi_overlay(how):
            lab = self.currentLab2D.copy()
            lab[delMask > 0] = 0
            if ax == 0:
                self.labelsLayerImg1.setImage(lab, autoLevels=False)
            else:
                self.labelsLayerRightImg.setImage(lab, autoLevels=False)

        self.setAllTextAnnotations(
            labelsToSkip=self.labels_to_skip(delIDs)
        )

    def applyDelROIs(self):
        self.logger.info('Applying deletion ROIs (if present)...')

        for posData in self.data:
            self.current_frame_i = posData.frame_i
            for frame_i in range(posData.SizeT):
                lab = posData.allData_li[frame_i]['labels']
                if lab is None:
                    break
                delROIs_info = posData.allData_li[frame_i]['delROIs_info']
                delIDs_rois = delROIs_info['delIDsROI']
                if not delIDs_rois:
                    continue
                for delIDs in delIDs_rois:
                    for delID in delIDs:
                        lab[lab==delID] = 0
                posData.allData_li[frame_i]['labels'] = lab
                # Get the rest of the metadata and store data based on the new lab
                posData.frame_i = frame_i
                self.get_data()
                self.store_data(autosave=False)

            # Back to current frame
            posData.frame_i = self.current_frame_i
            self.get_data()

    def setDelRoiState(self, roi: pg.ROI, state):
        roi.sigRegionChanged.disconnect()
        roi.sigRegionChangeFinished.disconnect()
        roi.setState(state)
        roi.sigRegionChanged.connect(self.delROImoving)
        roi.sigRegionChangeFinished.connect(self.delROImovingFinished)

    def addExistingDelROIs(self):
        posData = self.data[self.pos_i]
        delROIs_info = posData.allData_li[posData.frame_i]['delROIs_info']
        for r, roi in enumerate(delROIs_info['rois']):
            target_axis = self.roi_axis(
                is_polyline=isinstance(roi, pg.PolyLineROI),
                labels_image_visible=(
                    self.labelsGrad.showLabelsImgAction.isChecked()
                ),
            )
            if target_axis == 'left':
                self.ax1.addDelRoiItem(roi, roi.key)
            else:
                self.ax2.addDelRoiItem(roi, roi.key)

            self.setDelRoiState(roi, delROIs_info['state'][r])