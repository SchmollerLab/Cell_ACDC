"""Prompt inputs of the magic prompts: box prompts and brush-strokes
prompts. Mixins of guiWin (gui/main_window/window.py) only: the
measurements viewer never composes them.
"""
from collections import deque
import numpy as np
import pandas as pd
from qtpy.QtCore import Qt, QPointF, QRect, QRectF, QTimer
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QApplication
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import exception_handler, disableWindow
from cellacdc.myutils import image_utils
from ..mixins import widgets as gui_mixin_widgets
from ..widgets import points_layer_widgets as gui_widgets_points_layer_widgets
from cellacdc import _warnings
from cellacdc.promptable_models import utils as prompt_utils


class BoxPromptsGuiElements:
    """Box prompts of the magic prompts: draw, move, undo and clear boxes
    (clipMagicPromptsBoxesToZoom is a staticmethod).
    """

    @staticmethod
    def clipMagicPromptsBoxesToZoom(df_boxes, ymin, xmin, ymax, xmax):
        """Box prompts clipped to the zoomed area (ymin:ymax, xmin:xmax) in
        the coordinates of the zoomed area (None if no box is inside)."""
        df_boxes = df_boxes.copy()
        df_boxes[['ymin', 'ymax']] = (
            df_boxes[['ymin', 'ymax']].clip(ymin, ymax) - ymin
        )
        df_boxes[['xmin', 'xmax']] = (
            df_boxes[['xmin', 'xmax']].clip(xmin, xmax) - xmin
        )
        isInside = (
            (df_boxes['ymax'] - df_boxes['ymin'] >= 1)
            & (df_boxes['xmax'] - df_boxes['xmin'] >= 1)
        )
        df_boxes = df_boxes[isInside]
        if df_boxes.empty:
            return
        return df_boxes

    def isBoxPromptActive(self):
        """True if Magic prompts is active with the 'box' prompt type in a
        mode where segmentation masks can be edited"""
        try:
            if not self.magicPromptsToolButton.isChecked():
                return False
            if self.magicPromptsToolbar.promptType() != 'box':
                return False
            # The buttons of the prompts points layer stay checked with
            # boxes (the points are displayed and combined with the boxes)
            promptPointsButtons = self.promptPointsLayerClickButtons()
        except AttributeError:
            return False

        otherToolButtons = [*getattr(self, 'leftClickButtons', [])]
        checkableQButtonsGroup = getattr(self, 'checkableQButtonsGroup', None)
        if checkableQButtonsGroup is not None:
            otherToolButtons.extend(checkableQButtonsGroup.buttons())
        for button in otherToolButtons:
            if button in promptPointsButtons:
                continue
            if button is self.magicPromptsToolButton:
                continue
            if button.isChecked():
                return False

        mode = str(self.modeComboBox.currentText())
        return (
            mode == 'Segmentation and Tracking'
            or getattr(self, 'isSnapshot', False)
        )

    def isModelSupportingBoxPrompts(self):
        model = getattr(self.magicPromptsToolbar, 'model', None)
        return 'box' in tuple(getattr(model, 'prompt_types', ()))

    def areMagicPromptsBoxesVisible(self):
        if not self.magicPromptsToolButton.isChecked():
            return False
        if self.magicPromptsToolbar.promptType() not in ('point', 'box'):
            return False
        if getattr(self.magicPromptsToolbar, 'model', None) is None:
            # No model initialized yet: all the prompt types can be selected
            # (boxes drawn now are used if the model supports them)
            return True
        return self.isModelSupportingBoxPrompts()

    def getMagicPromptsBoxes(self, pos_i=None, frame_i=None, create=False):
        """List of the box prompts (dicts) of position `pos_i` and frame
        `frame_i` (default current ones)."""
        if pos_i is None:
            pos_i = self.pos_i
        if frame_i is None:
            frame_i = self.data[pos_i].frame_i
        posBoxesData = self.magicPromptsBoxesData.get(pos_i)
        if posBoxesData is None:
            if not create:
                return []
            posBoxesData = {}
            self.magicPromptsBoxesData[pos_i] = posBoxesData
        boxes = posBoxesData.get(frame_i)
        if boxes is None:
            if not create:
                return []
            boxes = []
            posBoxesData[frame_i] = boxes
        return boxes

    def isMagicPromptsBoxViewEditable(self):
        """Boxes can be drawn only in single z-slice view of z-stacks"""
        posData = self.data[self.pos_i]
        if posData.SizeZ == 1:
            return True
        return self.zProjComboBox.currentText() == 'single z-slice'

    def magicPromptsBoxesZ(self):
        """z-slice of the boxes drawn now (None for 2D data)"""
        posData = self.data[self.pos_i]
        if posData.SizeZ == 1:
            return None
        return int(self.zSliceScrollBar.sliderPosition())

    def magicPromptsBoxColor(self, box_id):
        """Colour of the labels of object `box_id` (RGB, 0-255)"""
        lut = getattr(self, 'lut', None)
        if lut is None or len(lut) < 2:
            return (255, 255, 0)
        if box_id < len(lut):
            rgb = lut[box_id]
        else:
            rgb = lut[1 + (box_id-1) % (len(lut)-1)]
        return tuple(int(c) for c in rgb[:3])

    def getNewBoxPromptId(self, right_click):
        """ID of a new box: for right-click, the "Right-click ID" (or, without
        prompts points layer, the ID of the last box added to the current
        frame), otherwise the ID of a new object (the same logic as the
        "Left-click ID" of the points)"""
        button = self.magicPromptsPointsLayerButton()
        if right_click:
            if button is not None:
                box_id = button.rightClickIDSpinbox.value()
            else:
                boxes = self.getMagicPromptsBoxes()
                box_id = boxes[-1]['id'] if boxes else None
            if box_id:
                return int(box_id)

        if button is None:
            return self.magicPromptsNewObjectId()

        box_id = button.pointIdSpinbox.value()
        if box_id <= 0:
            # ID 0 is for negative point prompts
            return self.magicPromptsNewObjectId()
        box_id = self.getClickedPointNewId(
            button.action, box_id, button.pointIdSpinbox, isMagicPrompts=True
        )
        proceed = self.warnAddingPointWithExistingId(box_id)
        if not proceed:
            return
        return int(box_id)

    def storeMagicPromptsBoxesUndo(self):
        posData = self.data[self.pos_i]
        key = (self.pos_i, posData.frame_i)
        undoQueue = self.magicPromptsBoxesUndo.get(key)
        if undoQueue is None:
            undoQueue = deque(maxlen=50)
            self.magicPromptsBoxesUndo[key] = undoQueue
        undoQueue.append([dict(box) for box in self.getMagicPromptsBoxes()])
        self.undoAction.setEnabled(True)

    def dropRedundantMagicPromptsBoxesUndo(self):
        """Remove the last undo state of the boxes if it is identical to
        the current boxes (e.g., a move that did not change the box)."""
        posData = self.data[self.pos_i]
        undoQueue = self.magicPromptsBoxesUndo.get(
            (self.pos_i, posData.frame_i)
        )
        if not undoQueue:
            return
        if undoQueue[-1] != self.getMagicPromptsBoxes():
            return
        undoQueue.pop()
        if not undoQueue:
            self.undoAction.setEnabled(self.isMasksUndoAvailable())

    def undoMagicPromptsBoxes(self):
        """Undo the last change to the boxes of the current frame. Returns
        False if there is nothing to undo."""
        posData = self.data[self.pos_i]
        key = (self.pos_i, posData.frame_i)
        undoQueue = self.magicPromptsBoxesUndo.get(key)
        if not undoQueue:
            return False
        boxes = undoQueue.pop()
        self.magicPromptsBoxesData.setdefault(self.pos_i, {})[
            posData.frame_i
        ] = boxes
        self.drawMagicPromptsBoxes()
        if not undoQueue:
            # Nothing left to undo for the boxes --> undo stays available
            # only for the masks
            self.undoAction.setEnabled(self.isMasksUndoAvailable())
        return True

    def addMagicPromptsBox(self, box_id, ymin, xmin, ymax, xmax):
        """Add a box prompt for object `box_id` on the displayed z-slice
        (it replaces the box of the same object on the same z-slice)."""
        z = self.magicPromptsBoxesZ()
        self.storeMagicPromptsBoxesUndo()
        boxes = self.getMagicPromptsBoxes(create=True)
        boxes[:] = [
            box for box in boxes
            if not (box['id'] == box_id and box['z'] == z)
        ]
        boxes.append({
            'id': int(box_id), 'z': z,
            'ymin': int(ymin), 'xmin': int(xmin),
            'ymax': int(ymax), 'xmax': int(xmax)
        })

        button = self.magicPromptsPointsLayerButton()
        if button is not None:
            # Right-click adds points (or boxes) to the same object
            button.rightClickIDSpinbox.setValue(int(box_id))
            next_id = self.getClickedPointNewId(
                button.action, int(box_id), button.pointIdSpinbox,
                isMagicPrompts=True
            )
            button.pointIdSpinbox.setValue(next_id, setLinkedWidget=False)

        self.logger.info(
            f'Added box prompt for object ID {box_id} '
            f'(x={xmin}-{xmax}, y={ymin}-{ymax}'
            f'{"" if z is None else f", z-slice={z+1}"})'
        )
        self.drawMagicPromptsBoxes()

    def removeMagicPromptsBoxItems(self):
        """Remove the displayed box prompts (`magicPromptsBoxItems`, also
        before they exist, e.g., in `initGlobalAttr`)"""
        for item in getattr(self, 'magicPromptsBoxItems', []):
            try:
                self.ax1.removeItem(item)
            except Exception as err:
                pass
        self.magicPromptsBoxItems = []

    def drawMagicPromptsBoxes(self):
        """Display the box prompts of the current frame (and z-slice)"""
        ax1 = getattr(self, 'ax1', None)
        if ax1 is None:
            return
        for item in self.magicPromptsBoxItems:
            ax1.removeItem(item)
        self.magicPromptsBoxItems = []

        try:
            if not self.areMagicPromptsBoxesVisible():
                return
        except AttributeError:
            return

        posData = self.data[self.pos_i]
        isBoxPromptActive = self.isBoxPromptActive()
        if isBoxPromptActive:
            # Keep the boxes undo available when coming back to a frame
            # (`get_data` disables the undo action if there are no masks
            # undo states)
            undoQueue = self.magicPromptsBoxesUndo.get(
                (self.pos_i, posData.frame_i)
            )
            if undoQueue:
                self.undoAction.setEnabled(True)

        boxes = self.getMagicPromptsBoxes()
        if not boxes:
            return

        isZstack = posData.SizeZ > 1
        isZslice = self.isMagicPromptsBoxViewEditable()
        z = self.magicPromptsBoxesZ() if isZstack and isZslice else None
        editable = isBoxPromptActive and isZslice
        Y, X = self.currentLab2D.shape
        maxBounds = QRectF(QRect(0, 0, X, Y))
        for box in boxes:
            if z is not None and box['z'] != z:
                continue
            roi = gui_widgets_points_layer_widgets.PromptBoxROI(
                (box['xmin'], box['ymin']),
                (box['xmax']-box['xmin'], box['ymax']-box['ymin']),
                box['id'], self.magicPromptsBoxColor(box['id']),
                maxBounds=maxBounds, editable=editable, boxData=box
            )
            roi.sigRegionChangeStarted.connect(
                self.magicPromptsBoxMoveStarted
            )
            roi.sigRegionChangeFinished.connect(self.magicPromptsBoxMoved)
            roi.sigRemoveRequested.connect(
                self.magicPromptsBoxRemoveRequested
            )
            # The band of the border must not change the auto range
            ax1.addItem(roi, ignoreBounds=True)
            self.magicPromptsBoxItems.append(roi)

    def magicPromptsBoxMoveStarted(self, roi):
        self.storeMagicPromptsBoxesUndo()

    def magicPromptsBoxMoved(self, roi):
        box = roi.boxData
        bbox = roi.bbox()
        ymin, xmin, ymax, xmax = bbox
        # Keep the box inside the image
        Y, X = self.currentLab2D.shape
        ymin, ymax = max(0, ymin), min(Y, ymax)
        xmin, xmax = max(0, xmin), min(X, xmax)
        if box is None or xmax - xmin < 1 or ymax - ymin < 1:
            # Box resized to nothing --> restore it
            QTimer.singleShot(0, self.drawMagicPromptsBoxes)
            return
        newCoords = dict(ymin=ymin, xmin=xmin, ymax=ymax, xmax=xmax)
        if all(box.get(key) == value for key, value in newCoords.items()):
            # Not moved (e.g., move cancelled) --> drop the undo state
            # stored when the move started
            self.dropRedundantMagicPromptsBoxesUndo()
            if (ymin, xmin, ymax, xmax) != tuple(bbox):
                # Displayed outside the image --> show the stored box
                QTimer.singleShot(0, self.drawMagicPromptsBoxes)
            return
        box.update(**newCoords)
        if (ymin, xmin, ymax, xmax) != tuple(bbox):
            # Redraw after the signal of the box is processed
            QTimer.singleShot(0, self.drawMagicPromptsBoxes)

    def magicPromptsBoxRemoveRequested(self, roi):
        boxes = self.getMagicPromptsBoxes()
        if roi.boxData not in boxes:
            return
        self.storeMagicPromptsBoxesUndo()
        boxes.remove(roi.boxData)
        self.logger.info(f'Removed box prompt of object ID {roi.box_id}')
        # Redraw after the signal of the removed box is processed
        QTimer.singleShot(0, self.drawMagicPromptsBoxes)

    def hoveredMagicPromptsBox(self):
        """Editable displayed box whose border or handle is under the mouse
        (None if there is none). Presses inside the boxes draw new boxes."""
        for roi in self.magicPromptsBoxItems:
            if not roi.isEditable():
                continue
            if roi.mouseHovering or self.gui_isHoveringRoiHandle(roi):
                return roi

    def isClickOnMagicPromptsBox(self):
        return self.hoveredMagicPromptsBox() is not None

    def isMovingMagicPromptsBox(self):
        """True while a box (or one of its handles) is dragged"""
        boxItems = set()
        for roi in self.magicPromptsBoxItems:
            boxItems.add(roi)
            boxItems.update(roi.getHandles())
        if any(getattr(item, 'isMoving', False) for item in boxItems):
            return True
        try:
            dragItem = getattr(self.ax1.scene(), 'dragItem', None)
        except Exception as err:
            return False
        return dragItem is not None and dragItem in boxItems

    def magicPromptsBoxesAt(self, x, y):
        """Editable displayed boxes containing (x, y), smallest first"""
        rois = []
        for roi in self.magicPromptsBoxItems:
            if not roi.isEditable():
                continue
            ymin, xmin, ymax, xmax = roi.bbox()
            if xmin <= x <= xmax and ymin <= y <= ymax:
                rois.append(roi)
        rois.sort(key=lambda roi: roi.size()[0]*roi.size()[1])
        return rois

    def boxPromptsMousePress(self, event, right_click):
        if self.magicPromptsDrawingBox is not None:
            # Another mouse button pressed while drawing a box --> cancel
            # the box (like pyqtgraph ROIs cancel a move)
            self.boxPromptsCancelDrag()
            return

        hoveredRoi = self.hoveredMagicPromptsBox()
        isMoving = self.isMovingMagicPromptsBox()
        if isMoving or (hoveredRoi is not None and not right_click):
            # Left-press on the border or a handle: move or resize the box
            # (pyqtgraph). Another button pressed while moving a box: the
            # box cancels the move
            event.ignore()
            return

        x, y = event.pos().x(), event.pos().y()

        if not self.isMagicPromptsBoxViewEditable():
            _warnings.warnCannotAddRemoveBoxesProjection(qparent=self)
            return

        Y, X = self.currentLab2D.shape
        if hoveredRoi is not None:
            # The band of the border extends a little outside the image
            x, y = min(max(x, 0), X-1), min(max(y, 0), Y-1)
        elif not (0 <= x < X and 0 <= y < Y):
            return

        try:
            screenPos = QPointF(event.screenPos())
        except Exception as err:
            screenPos = None

        xdata, ydata = int(x), int(y)
        pen = pg.mkPen((255, 255, 0), width=2, style=Qt.DashLine)
        roi = gui_mixin_widgets.ROI(
            (xdata, ydata), (0, 0), pen=pen, hoverPen=pen,
            movable=False, rotatable=False, resizable=False
        )
        self.ax1.addItem(roi)
        self.magicPromptsDrawingBox = {
            'roi': roi, 'x0': xdata, 'y0': ydata, 'right_click': right_click,
            'screenPos': screenPos, 'hoveredRoi': hoveredRoi
        }
        self.isBoxPromptDragImg1 = True

    def boxPromptsDragTo(self, x, y):
        drawing = self.magicPromptsDrawingBox
        if drawing is None:
            return
        Y, X = self.currentLab2D.shape
        x = int(round(min(max(x, 0), X)))
        y = int(round(min(max(y, 0), Y)))
        xmin, xmax = sorted((drawing['x0'], x))
        ymin, ymax = sorted((drawing['y0'], y))
        roi = drawing['roi']
        roi.setPos((xmin, ymin), update=False, finish=False)
        roi.setSize((xmax-xmin, ymax-ymin), finish=False)

    def boxPromptsCancelDrag(self):
        drawing = self.magicPromptsDrawingBox
        self.magicPromptsDrawingBox = None
        self.isBoxPromptDragImg1 = False
        if drawing is not None:
            self.ax1.removeItem(drawing['roi'])
        return drawing

    def boxPromptsEndDrag(self, event=None):
        drawing = self.boxPromptsCancelDrag()
        if drawing is None:
            return
        ymin, xmin, ymax, xmax = drawing['roi'].bbox()
        # Click or drag is decided on screen (independent of the zoom)
        isClick = None
        screenPos = drawing.get('screenPos')
        if event is not None and screenPos is not None:
            try:
                delta = QPointF(event.screenPos()) - screenPos
                isClick = (
                    abs(delta.x()) + abs(delta.y())
                    < QApplication.startDragDistance()
                )
            except Exception as err:
                isClick = None
        if isClick is None:
            isClick = xmax - xmin < 2 or ymax - ymin < 2
        if isClick:
            # Right-click on a box (border or inside) --> context menu of
            # the box (of the smallest one inside several boxes)
            if drawing['right_click']:
                rois = [drawing.get('hoveredRoi')]
                if rois[0] not in self.magicPromptsBoxItems:
                    rois = self.magicPromptsBoxesAt(
                        drawing['x0'], drawing['y0']
                    )
                if rois:
                    rois[0].getMenu().popup(QCursor.pos())
            return
        if xmax - xmin < 1 or ymax - ymin < 1:
            # Dragged only along one axis
            return
        box_id = self.getNewBoxPromptId(drawing['right_click'])
        if box_id is None:
            return
        self.addMagicPromptsBox(box_id, ymin, xmin, ymax, xmax)

    def magicPromptsClearBoxes(self, toolbar, only_zoom=False):
        boxes = self.getMagicPromptsBoxes()
        if not boxes:
            return

        if only_zoom:
            ((xmin, xmax), (ymin, ymax)) = self.ax1.viewRange()
            toRemove = [
                box for box in boxes
                if box['xmin'] < xmax and box['xmax'] > xmin
                and box['ymin'] < ymax and box['ymax'] > ymin
            ]
        else:
            toRemove = list(boxes)

        if not toRemove:
            return

        self.storeMagicPromptsBoxesUndo()
        for box in toRemove:
            boxes.remove(box)
        self.drawMagicPromptsBoxes()

    def getMagicPromptsBoxesDf(self, toolbar):
        """Box prompts of the current frame as a DataFrame (None if there
        are no boxes or if the model does not support box prompts)."""
        boxes = self.getMagicPromptsBoxes()
        if not boxes:
            return
        if not self.isModelSupportingBoxPrompts():
            self.logger.warning(
                f'[WARNING]: The box prompts are ignored because the model '
                f'{toolbar.model_name} does not support them.'
            )
            return
        posData = self.data[self.pos_i]
        df_boxes = pd.DataFrame(boxes, columns=[
            'id', 'z', 'ymin', 'xmin', 'ymax', 'xmax'
        ])
        df_boxes.insert(0, 'frame_i', posData.frame_i)
        return df_boxes


class BrushStrokesPromptsGuiElements:
    """Brush-strokes prompts of the magic prompts: painting, cell selection
    and training data.
    """

    def isBrushStrokesPromptActive(self):
        """True if Magic prompts is active with the 'brush_stroke' prompt
        type in a mode where segmentation masks can be edited"""
        try:
            if not self.magicPromptsToolButton.isChecked():
                return False
            if self.magicPromptsToolbar.promptType() != 'brush_stroke':
                return False
        except AttributeError:
            return False

        # Another tool (e.g., brush, eraser, add points, keep IDs, or
        # right-click tools like edit ID or separate) is active
        otherToolButtons = [*getattr(self, 'leftClickButtons', [])]
        checkableQButtonsGroup = getattr(self, 'checkableQButtonsGroup', None)
        if checkableQButtonsGroup is not None:
            otherToolButtons.extend(checkableQButtonsGroup.buttons())
        for button in otherToolButtons:
            if button.isChecked():
                return False

        mode = str(self.modeComboBox.currentText())
        return (
            mode == 'Segmentation and Tracking'
            or getattr(self, 'isSnapshot', False)
        )

    def getBrushStrokesAnnotShape(self, posData):
        segm_data = getattr(posData, 'segm_data', None)
        if segm_data is not None and segm_data.ndim >= 3:
            return tuple(segm_data.shape[1:])
        return tuple(posData.lab.shape)

    def getBrushStrokesAnnot(self, pos_i=None, frame_i=None, create=True):
        """Get the brush strokes annotation array (uint8, 0 = unlabelled,
        1 = background, 2 = foreground) with shape (Y, X) or (Z, Y, X) of
        position `pos_i` and frame `frame_i` (default current ones)."""
        if pos_i is None:
            pos_i = self.pos_i
        posData = self.data[pos_i]
        if frame_i is None:
            frame_i = posData.frame_i

        shape = self.getBrushStrokesAnnotShape(posData)
        posAnnotData = self.brushStrokesAnnotData.get(pos_i, {})
        annot = posAnnotData.get(frame_i)
        if annot is not None and annot.shape != shape:
            self.logger.warning(
                f'[WARNING]: Discarding brush strokes of frame n. '
                f'{frame_i+1} because their shape {annot.shape} is '
                f'different from the segmentation masks shape {shape}.'
            )
            posAnnotData.pop(frame_i, None)
            annot = None

        if annot is None and create:
            annot = np.zeros(shape, dtype=np.uint8)
            self.brushStrokesAnnotData.setdefault(pos_i, {})[frame_i] = annot

        return annot

    def getBrushStrokesDepthAxes(self):
        if not self.switchPlaneCombobox.isEnabled():
            return 'z'
        return self.switchPlaneCombobox.depthAxes()

    def isBrushStrokesViewPaintable(self, annot_ndim):
        if self.getBrushStrokesDepthAxes() != 'z':
            return False
        if annot_ndim == 2:
            return True
        return self.zProjComboBox.currentText() == 'single z-slice'

    def getBrushStrokesAnnot2D(self, create=True, return_z=False):
        """Get the writable 2D view of the brush strokes annotation of the
        current frame used for painting. Returns None if the current view
        is display-only (z-projections or plane views other than z)."""
        posData = self.data[self.pos_i]
        annot_ndim = len(self.getBrushStrokesAnnotShape(posData))
        view, z = None, None
        if self.isBrushStrokesViewPaintable(annot_ndim):
            annot = self.getBrushStrokesAnnot(create=create)
            if annot is None:
                pass
            elif annot.ndim == 2:
                view = annot
            else:
                z = self.zSliceScrollBar.sliderPosition()
                z = min(max(z, 0), len(annot)-1)
                view = annot[z]

        if return_z:
            return view, z

        return view

    def getBrushStrokesImageSource(self):
        """Source of the displayed image of the current frame that 2D brush
        strokes of z-stack data are paired with (z-slice index or
        z-projection name). None with 3D brush strokes or 2D data."""
        posData = self.data[self.pos_i]
        annot_ndim = len(self.getBrushStrokesAnnotShape(posData))
        return prompt_utils.get_brush_strokes_image_source(
            posData, posData.frame_i, annot_ndim
        )

    def getBrushStrokesImageSourceMismatch(self, annot):
        """Source of the image that the brush strokes `annot` of the current
        frame were painted on if it is not the displayed one (2D brush
        strokes of z-stack data painted on another z-slice or z-projection),
        otherwise None."""
        posData = self.data[self.pos_i]
        key = (self.pos_i, posData.frame_i)
        source = self.brushStrokesImageSources.get(key)
        if source is None or annot is None or not annot.any():
            return

        if source == self.getBrushStrokesImageSource():
            return

        return source

    def recordBrushStrokesImageSource(self):
        """Pair the brush strokes of the current frame with the displayed
        image (2D brush strokes of z-stack data)"""
        posData = self.data[self.pos_i]
        key = (self.pos_i, posData.frame_i)
        source = self.getBrushStrokesImageSource()
        if source is None:
            self.brushStrokesImageSources.pop(key, None)
        else:
            self.brushStrokesImageSources[key] = source

    def warnBrushStrokesImageSourceMismatch(self, source):
        if isinstance(source, str):
            where = f'the {source}'
        else:
            where = f'z-slice n. {source+1}'
        self.titleLabel.setText(
            f'The brush strokes of this frame were painted on {where}. '
            'Go back to it to edit them, or clear them to paint on the '
            'displayed image (with 2D masks, the brush strokes of a frame '
            'belong to one z-slice or z-projection).', color='r'
        )

    def getBrushStrokesAnnotDisplay(self):
        """Get the 2D image of the brush strokes of the current frame as
        displayed (current z-slice, plane, or max projection)"""
        annot = self.getBrushStrokesAnnot(create=False)
        if annot is None:
            return

        depthAxes = self.getBrushStrokesDepthAxes()
        if annot.ndim == 2:
            if depthAxes != 'z':
                return
            if self.getBrushStrokesImageSourceMismatch(annot) is not None:
                # Painted on another z-slice or z-projection
                return
            return annot

        idx = max(self.zSliceScrollBar.sliderPosition(), 0)
        if depthAxes == 'y':
            return annot[:, min(idx, annot.shape[1]-1)]

        if depthAxes == 'x':
            return annot[:, :, min(idx, annot.shape[2]-1)]

        if self.zProjComboBox.currentText() != 'single z-slice':
            return annot.max(axis=0)

        return annot[min(idx, len(annot)-1)]

    def setBrushStrokesLayerImage(self):
        item = getattr(self, 'brushStrokesLayerImg1', None)
        if item is None:
            return

        isActive = self.isBrushStrokesPromptActive()
        item.setVisible(isActive)
        self.setBrushStrokesSelectedCellsLayerImage()
        if not isActive:
            return

        # Keep brush strokes undo available when coming back to a frame
        # (`get_data` disables the undo action if there are no masks undo
        # states)
        posData = self.data[self.pos_i]
        undoQueues = getattr(self, 'brushStrokesUndo', {})
        if undoQueues.get((self.pos_i, posData.frame_i)):
            self.undoAction.setEnabled(True)

        annot2D = self.getBrushStrokesAnnotDisplay()
        if annot2D is None:
            item.clear()
            return

        item.setImage(annot2D, autoLevels=False)

    def getBrushStrokesDiskMask(self, radius):
        diskMask = self.brushStrokesDiskMasks.get(radius)
        if diskMask is None:
            diskMask = prompt_utils.get_brush_disk_mask(radius)
            self.brushStrokesDiskMasks[radius] = diskMask
        return diskMask

    def warnBrushStrokesNotPaintable(self):
        txt = (
            'Brush strokes can be painted only in "single z-slice" mode '
            'viewing the z-slices (z-projections and other planes are '
            'display-only).'
        )
        self.titleLabel.setText(txt, color='r')
        if self.brushStrokesWarnedNotPaintable:
            return
        self.brushStrokesWarnedNotPaintable = True
        self.logger.warning(f'[WARNING]: {txt}')

    def brushStrokesMousePress(self, event, left_click, modifiers):
        annot2D, z = self.getBrushStrokesAnnot2D(create=True, return_z=True)
        if annot2D is None:
            self.warnBrushStrokesNotPaintable()
            return

        x, y = event.pos().x(), event.pos().y()
        Y, X = annot2D.shape
        if not image_utils.is_in_bounds(int(x), int(y), X, Y):
            return

        otherSource = self.getBrushStrokesImageSourceMismatch(annot2D)
        if otherSource is not None:
            self.warnBrushStrokesImageSourceMismatch(otherSource)
            return

        self.storeBrushStrokesUndo(annot2D, z)
        self.recordBrushStrokesImageSource()

        if modifiers == Qt.ShiftModifier:
            # Erase
            self.brushStrokePaintValue = 0
        elif left_click:
            # Foreground
            self.brushStrokePaintValue = 2
        else:
            # Background
            self.brushStrokePaintValue = 1

        self.brushStrokesChangedSinceTraining = True
        self.isBrushStrokeDragImg1 = True
        self.brushStrokeLastYX = None
        self.brushStrokeDragTarget = self.getBrushStrokesPaintTarget(z)
        self.brushStrokesPaintAt(x, y)

    def getBrushStrokesPaintTarget(self, z):
        posData = self.data[self.pos_i]
        # The image source of 2D brush strokes of z-stack data changes with
        # the z-slice or z-projection
        source = self.getBrushStrokesImageSource()
        return (self.pos_i, posData.frame_i, z, source)

    def brushStrokesPaintAt(self, x, y):
        annot2D, z = self.getBrushStrokesAnnot2D(create=True, return_z=True)
        if annot2D is None:
            self.brushStrokesEndDrag()
            return

        target = self.getBrushStrokesPaintTarget(z)
        if target != getattr(self, 'brushStrokeDragTarget', target):
            # Frame, position or z-slice changed during the stroke (e.g.,
            # with the keyboard) --> end the stroke (the undo state was
            # stored for the previous target only)
            self.brushStrokesEndDrag()
            return

        radius = self.magicPromptsToolbar.brushSizeSpinbox.value()
        prompt_utils.paint_brush_stroke(
            annot2D, y, x, self.brushStrokePaintValue, radius,
            last_yx=self.brushStrokeLastYX,
            disk_mask=self.getBrushStrokesDiskMask(radius)
        )
        self.brushStrokeLastYX = (y, x)
        self.brushStrokesLayerImg1.setVisible(True)
        self.brushStrokesLayerImg1.setImage(annot2D, autoLevels=False)

    def brushStrokesEndDrag(self):
        self.isBrushStrokeDragImg1 = False
        self.brushStrokeLastYX = None
        self.brushStrokeDragTarget = None

    def storeBrushStrokesUndo(self, annot, z=None):
        """Store a copy of `annot` (2D z-slice `z` or the full annotation of
        the current frame if `z` is None) to undo brush strokes"""
        posData = self.data[self.pos_i]
        key = (self.pos_i, posData.frame_i)
        undoQueue = self.brushStrokesUndo.get(key)
        if undoQueue is None:
            undoQueue = deque(maxlen=20)
            self.brushStrokesUndo[key] = undoQueue
        # The image source is restored too (e.g., brush strokes cleared and
        # painted again on another z-slice)
        source = self.brushStrokesImageSources.get(key)
        undoQueue.append((z, annot.copy(), source))
        self.undoAction.setEnabled(True)

    def undoBrushStrokes(self):
        posData = self.data[self.pos_i]
        key = (self.pos_i, posData.frame_i)
        undoQueue = self.brushStrokesUndo.get(key)
        if not undoQueue:
            return False

        z, prevAnnot, prevSource = undoQueue.pop()
        annot = self.getBrushStrokesAnnot(create=True)
        target = annot if z is None else annot[z]
        if target.shape != prevAnnot.shape:
            return False

        target[...] = prevAnnot
        if prevSource is None:
            self.brushStrokesImageSources.pop(key, None)
        else:
            self.brushStrokesImageSources[key] = prevSource
        self.brushStrokesChangedSinceTraining = True
        self.setBrushStrokesLayerImage()
        return True

    def getBrushStrokesZoomRegion(self, annot):
        """Indexing of `annot` corresponding to the zoomed area (all the
        depth axis)"""
        ((xmin, xmax), (ymin, ymax)) = self.ax1.viewRange()
        depthAxes = 'z'
        if annot.ndim == 3:
            depthAxes = self.getBrushStrokesDepthAxes()

        if depthAxes == 'y':
            numRows, numCols = annot.shape[0], annot.shape[2]
        elif depthAxes == 'x':
            numRows, numCols = annot.shape[0], annot.shape[1]
        else:
            numRows, numCols = annot.shape[-2:]

        row_start, row_end = int(max(0, ymin)), int(min(numRows, ymax))
        col_start, col_end = int(max(0, xmin)), int(min(numCols, xmax))
        if row_start >= row_end or col_start >= col_end:
            return

        rows = slice(row_start, row_end)
        cols = slice(col_start, col_end)
        if depthAxes == 'y':
            return (rows, slice(None), cols)

        if depthAxes == 'x':
            return (rows, cols, slice(None))

        return (Ellipsis, rows, cols)

    def magicPromptsClearBrushStrokes(self, toolbar, only_zoom=False):
        annot = self.getBrushStrokesAnnot(create=False)
        if annot is None or not annot.any():
            return

        if only_zoom:
            region = self.getBrushStrokesZoomRegion(annot)
            if region is None:
                return
        else:
            region = (Ellipsis,)

        self.storeBrushStrokesUndo(annot, None)
        annot[region] = 0
        self.brushStrokesChangedSinceTraining = True
        self.setBrushStrokesLayerImage()

        where = 'in the zoomed area ' if only_zoom else ''
        self.logger.info(f'Brush strokes {where}of current frame cleared.')

    def isBrushStrokesSelectCellsActive(self):
        """True if selecting cells to add them to the brush strokes (i.e.,
        "Select cells" is checked and brush strokes prompts are active)"""
        try:
            isChecked = self.magicPromptsToolbar.selectCellsAction.isChecked()
        except AttributeError:
            return False
        return isChecked and self.isBrushStrokesPromptActive()

    def brushStrokesSelectCellsToggled(self, toolbar, checked):
        if checked:
            self.brushStrokesEndDrag()
        self.brushStrokesSelectCellsToolbar.resetSelection()
        self.applyBrushStrokesSelectCellsMode()
        if checked:
            self.titleLabel.setText(
                'Click on the cells to select them (Ctrl+A to select all), '
                'then press Enter to add them to the brush strokes.',
                color=self.titleColor
            )
        QTimer.singleShot(20, self.gui_refreshToolCursor)

    def applyBrushStrokesSelectCellsMode(self):
        """Show the "Select cells" controls only while selecting cells with
        brush strokes prompts, and update the selected cells layer"""
        controls = self.brushStrokesSelectCellsToolbar
        isSelecting = (
            self.magicPromptsToolButton.isChecked()
            and self.magicPromptsToolbar.promptType() == 'brush_stroke'
            and self.magicPromptsToolbar.selectCellsAction.isChecked()
        )
        if not isSelecting:
            controls.resetSelection()
        controls.setVisible(isSelecting)
        self.setBrushStrokesSelectedCellsLayerImage()
        # The brush circle is not used while selecting cells (redrawn on
        # next hover if needed)
        self.setHoverToolSymbolData(
            [], [], (self.ax2_BrushCircle, self.ax1_BrushCircle),
        )

    def setBrushStrokesSelectedCellsLayerImage(self):
        """Show the cells selected to be added to the brush strokes (same
        look as the "Keep IDs" selection)"""
        item = getattr(self, 'brushStrokesSelectedCellsLayerImg1', None)
        controls = getattr(self, 'brushStrokesSelectCellsToolbar', None)
        if item is None or controls is None:
            return

        selectedIDs = controls.selectedIDs
        # Objects are drawn on the displayed z-slice/projection only (other
        # planes are display-only for brush strokes)
        isVisible = (
            len(selectedIDs) > 0
            and self.isBrushStrokesSelectCellsActive()
            and self.getBrushStrokesDepthAxes() == 'z'
        )
        if not isVisible:
            item.clear()
            item.setVisible(False)
            return

        self.initKeepObjLabelsLayers(imageItem=item)
        item.setImage(self.getObjectsLab2D(selectedIDs), autoLevels=False)
        item.setVisible(True)

    def brushStrokesSelectCellClick(self, event):
        """Select (or unselect) the clicked cell to add it to the brush
        strokes"""
        x, y = event.pos().x(), event.pos().y()
        xdata, ydata = int(x), int(y)
        lab2D = self.currentLab2D
        Y, X = lab2D.shape[-2:]
        if not image_utils.is_in_bounds(xdata, ydata, X, Y):
            return

        ID = int(lab2D[ydata, xdata])
        if ID == 0:
            self.titleLabel.setText(
                'You clicked on the background. Click on a cell to select it '
                '(or press Ctrl+A to select all the cells).', color='r'
            )
            return

        controls = self.brushStrokesSelectCellsToolbar
        isSelected = controls.toggleSelectedID(ID)
        self.setBrushStrokesSelectedCellsLayerImage()
        action = 'selected' if isSelected else 'unselected'
        numSelected = len(controls.selectedIDs)
        self.titleLabel.setText(
            f'ID {ID} {action} (number of selected cells = {numSelected}). '
            'Press Enter to add the selected cells to the brush strokes.',
            color=self.titleColor
        )

    def brushStrokesSelectAllCells(self, toolbar):
        """Select all the cells of the current frame to add them to the
        brush strokes (e.g., with Ctrl+A)"""
        if not self.isBrushStrokesPromptActive():
            return

        if not toolbar.selectCellsAction.isChecked():
            toolbar.selectCellsAction.setChecked(True)

        posData = self.data[self.pos_i]
        IDs = [int(ID) for ID in posData.IDs]
        controls = self.brushStrokesSelectCellsToolbar
        controls.setSelectedIDs(IDs)
        self.setBrushStrokesSelectedCellsLayerImage()
        if not IDs:
            self.titleLabel.setText(
                'There are no cells in the current frame.', color='r'
            )
            return

        self.titleLabel.setText(
            f'All the {len(IDs)} cells of the current frame selected. Press '
            'Enter to add them to the brush strokes.',
            color=self.titleColor
        )

    def brushStrokesSelectedIDsEdited(self, controls, IDs):
        """The IDs of the selected cells were typed in the line edit"""
        posData = self.data[self.pos_i]
        existingIDs = set(posData.IDs)
        validIDs = [ID for ID in IDs if ID in existingIDs]
        controls.syncSelectedIDs(validIDs)
        if len(validIDs) < len(IDs):
            controls.selectedIDsLineEdit.warnNotExistingID()
        else:
            controls.selectedIDsLineEdit.setInstructionsText()
        self.setBrushStrokesSelectedCellsLayerImage()

    def brushStrokesSelectCellsEscape(self):
        """Escape key while selecting cells: clear the selection, or stop
        selecting cells if nothing is selected. Returns True if handled."""
        if not self.isBrushStrokesSelectCellsActive():
            return False

        controls = self.brushStrokesSelectCellsToolbar
        if controls.selectedIDs:
            controls.resetSelection()
            self.setBrushStrokesSelectedCellsLayerImage()
            self.titleLabel.setText('Selection cleared.', color=self.titleColor)
            return True

        self.magicPromptsToolbar.selectCellsAction.setChecked(False)
        return True

    @exception_handler
    def magicPromptsAddSelectedCellsToBrushStrokes(self, controls):
        """Add the selected cells to the brush strokes of the current frame
        as foreground and the pixels without masks as background. Only the
        pixels without brush strokes are modified. With 3D masks, only the
        displayed z-slice is added."""
        if not self.isBrushStrokesSelectCellsActive():
            return

        posData = self.data[self.pos_i]
        existingIDs = set(posData.IDs)
        selectedIDs = [ID for ID in controls.selectedIDs if ID in existingIDs]
        if not selectedIDs:
            self.titleLabel.setText(
                'None of the selected cells exist in the current frame.',
                color='r'
            )
            return

        annot2D, z = self.getBrushStrokesAnnot2D(create=True, return_z=True)
        if annot2D is None:
            self.warnBrushStrokesNotPaintable()
            return

        otherSource = self.getBrushStrokesImageSourceMismatch(annot2D)
        if otherSource is not None:
            self.warnBrushStrokesImageSourceMismatch(otherSource)
            return

        lab2D = posData.lab if z is None else posData.lab[z]
        if lab2D.shape != annot2D.shape:
            self.logger.warning(
                f'[WARNING]: Cannot add the selected cells to the brush '
                f'strokes because the segmentation masks shape '
                f'{lab2D.shape} is different from the brush strokes shape '
                f'{annot2D.shape}.'
            )
            return

        where = 'current frame'
        if controls.onlyZoomCheckbox.isChecked():
            region = self.getBrushStrokesZoomRegion(annot2D)
            if region is None:
                self.titleLabel.setText(
                    'The zoomed area is outside of the image.', color='r'
                )
                return
            where = 'zoomed area of the current frame'
        else:
            region = (Ellipsis,)

        if z is not None:
            where = f'{where} (z-slice n. {z+1})'

        # Only the selected cells that are in the added area (displayed
        # z-slice and/or zoomed area)
        regionIDs = np.unique(lab2D[region])
        selectedIDs = [ID for ID in selectedIDs if ID in regionIDs]
        if not selectedIDs:
            self.titleLabel.setText(
                f'None of the selected cells are in the {where}.', color='r'
            )
            return

        prevAnnot2D = annot2D.copy()
        numForeground, numBackground = (
            prompt_utils.add_objects_to_brush_strokes(
                annot2D[region], lab2D[region], selectedIDs
            )
        )
        if numForeground + numBackground == 0:
            self.titleLabel.setText(
                'No pixels added: all the pixels already have brush strokes '
                '(clear them first to replace them).', color='r'
            )
            return

        # Undo state stored only if something changed (the undo queue has a
        # maximum length)
        self.storeBrushStrokesUndo(prevAnnot2D, z)
        self.recordBrushStrokesImageSource()
        self.brushStrokesChangedSinceTraining = True
        controls.resetSelection()
        self.setBrushStrokesLayerImage()

        numCells = len(selectedIDs)
        cells = 'cell' if numCells == 1 else 'cells'
        txt = (
            f'Added {numCells} {cells} to the brush strokes of the {where} '
            f'({numForeground} foreground and {numBackground} background '
            'pixels).'
        )
        self.logger.info(txt)
        self.titleLabel.setText(txt, color='g')
        if numForeground + numBackground > 1_000_000:
            self.logger.info(
                'NOTE: The more pixels are annotated, the slower the '
                'training. With large images, consider adding only a '
                'cutout (check "Only in zoomed area").'
            )

    def getBrushStrokesTrainingItems(self):
        """List of (pos_i, frame_i, annot) with at least one brush stroke
        across all loaded positions"""
        items = []
        for pos_i, posAnnotData in sorted(self.brushStrokesAnnotData.items()):
            if pos_i >= len(self.data):
                continue
            for frame_i, annot in sorted(posAnnotData.items()):
                if annot is None or not annot.any():
                    continue
                items.append((pos_i, frame_i, annot))
        return items

    @exception_handler
    @disableWindow
    def magicPromptsTrainTriggered(self, toolbar):
        self.magicPromptsTrainBrushStrokes(toolbar)

    def magicPromptsTrainBrushStrokes(self, toolbar):
        """Train the promptable model on all the brush strokes. Returns
        True if training was successful."""
        if not toolbar.viewModelParamsAction.isEnabled():
            _warnings.warnPromptSegmentModelNotInit(qparent=self)
            return False

        items = self.getBrushStrokesTrainingItems()
        if not items:
            _warnings.warnBrushStrokesMissing(qparent=self)
            return False

        hasForeground = any((annot == 2).any() for _, _, annot in items)
        hasBackground = any((annot == 1).any() for _, _, annot in items)
        if not hasForeground or not hasBackground:
            _warnings.warnBrushStrokesMissingClass(
                hasForeground, hasBackground, qparent=self
            )
            return False

        prompts_data = []
        for pos_i, frame_i, annot in items:
            # Image displayed when the brush strokes were painted (the
            # z-slice or z-projection of 2D brush strokes of z-stack data
            # can be changed afterwards)
            source = self.brushStrokesImageSources.get((pos_i, frame_i))
            image = prompt_utils.get_brush_strokes_image(
                self.data[pos_i], frame_i, annot.ndim, source=source
            )
            prompts_data.append((pos_i, frame_i, annot.copy(), image))

        model_name = getattr(toolbar, 'model_name', 'promptable model')
        desc = f'Training {model_name} on brush strokes...'
        self.logger.info(
            f'{desc} (number of annotated images = {len(prompts_data)})'
        )
        result = self.magicPromptsRunModelTask(
            desc, prompt_utils.train_model_on_brush_strokes,
            toolbar.model, prompts_data, log_func=self.logger.info
        )

        toolbar.isModelTrained = True
        toolbar.isModelSaved = False
        toolbar.savedModelName = None
        toolbar.loadedModelName = None
        self.brushStrokesChangedSinceTraining = False
        self.logger.info(
            f'{model_name} successfully trained on brush strokes. '
            f'Training summary: {result}'
        )
        self.titleLabel.setText(
            f'{model_name} successfully trained on brush strokes!',
            color='g'
        )
        return True

    def magicPromptsEnsureBrushStrokesTrained(self, toolbar):
        """(Re-)train the model if not trained or if the brush strokes
        changed since last training. Returns True if the model is ready
        to segment."""
        if not toolbar.viewModelParamsAction.isEnabled():
            _warnings.warnPromptSegmentModelNotInit(qparent=self)
            return False

        hasStrokes = len(self.getBrushStrokesTrainingItems()) > 0
        isModelTrained = getattr(toolbar, 'isModelTrained', False)
        needsTraining = (
            not isModelTrained
            or (self.brushStrokesChangedSinceTraining and hasStrokes)
        )
        if not needsTraining:
            return True

        if not hasStrokes:
            _warnings.warnBrushStrokesMissing(qparent=self)
            return False

        loadedModelName = getattr(toolbar, 'loadedModelName', None)
        if isModelTrained and loadedModelName is not None:
            answer = self.askRetrainLoadedPromptModel(
                toolbar.model_name, loadedModelName
            )
            if answer == 'cancel':
                return False
            if answer == 'use_loaded':
                # Do not ask again until the brush strokes change again
                self.brushStrokesChangedSinceTraining = False
                return True

        if isModelTrained:
            self.logger.info(
                'Brush strokes changed since last training --> '
                're-training the model...'
            )

        return self.magicPromptsTrainBrushStrokes(toolbar)
