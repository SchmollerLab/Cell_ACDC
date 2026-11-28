"""Points layers and promptable model (magic prompts) widgets of the main
GUI window only: PointsLayersToolbar, PromptableModelPointsLayerToolbar,
MagicPromptsToolbar, SavePointsLayerButton, PointsScatterPlotItem, the
box prompt ROI (PromptBoxROI, _PromptBoxHandle) and
SamInputPointsWidget. The points layer dialogs are in
gui/dialogs/points_layers.py, the window methods in
gui/main_window/points_layers.py.
"""
import os
import numpy as np
import pandas as pd
from qtpy.QtCore import Signal, Qt, QPoint, QPoint, QRectF
from qtpy.QtGui import QFont, QIcon, QPainterPath
from qtpy.QtWidgets import QHBoxLayout, QWidget, QAction, QMenu
import pyqtgraph as pg
pg.setConfigOption('imageAxisOrder', 'row-major')
from cellacdc import html_utils
from cellacdc.myutils import filesystem
from cellacdc import printl
from cellacdc import load
from cellacdc.shared.dialogs import file_selection as dialogs_file_selection
from ..dialogs import segmentation as gui_dialogs_segmentation
from cellacdc import annotate
from cellacdc import core
from cellacdc.shared.widgets import toolbars as widgets_toolbars
from ..mixins import widgets as gui_mixin_widgets
from cellacdc.shared.widgets import line_edits as widgets_line_edits
from cellacdc.shared.widgets import push_buttons as widgets_push_buttons
from cellacdc.shared.widgets import message_box as widgets_message_box


class SavePointsLayerButton(widgets_toolbars.rightClickToolButton):
    sigRenameTableAction = Signal(object, str)
    
    def __init__(self, table_endname, parent=None):
        super().__init__(parent=parent)
        self.setIcon(QIcon(':file-save.svg'))
        
        self.table_endname = table_endname
        
        self.setToolTip(
            "Save annotated points in the CSV file ending "
            f"with '{self.table_endname}.csv'"
        )
                
        self.sigRightClick.connect(self.showContextMenu)
    
    def showContextMenu(self, event):
        contextMenu = QMenu(self)
        contextMenu.addSeparator()
        
        renameAction = QAction('Rename points layer table')
        renameAction.triggered.connect(self.renameTable)
        contextMenu.addAction(renameAction)
        
        contextMenu.exec(event.globalPos())
    
    def renameTable(self):
        win = dialogs_file_selection.filenameDialog(
            parent=self,
            title='Rename points layer table file',
            allowEmpty=False,
            defaultEntry=self.table_endname,
            ext='.csv',
        )
        win.exec_()
        if win.cancel:
            return
        
        self.table_endname = win.entryText
        self.setToolTip(
            "Save annotated points in the CSV file ending "
            f"with '{self.table_endname}.csv'"
        )
        self.sigRenameTableAction.emit(self, self.table_endname)


class _PromptBoxHandle(pg.graphicsItems.ROI.Handle):
    """Corner handle of `PromptBoxROI`: Alt+drag is left to the view (pan
    the image, like on the rest of the image)."""
    def mouseDragEvent(self, ev):
        isAltStart = (
            ev.isStart()
            and ev.modifiers() == Qt.KeyboardModifier.AltModifier
        )
        if isAltStart:
            ev.ignore()
            return
        super().mouseDragEvent(ev)

class PromptBoxROI(gui_mixin_widgets.ROI):
    """Box prompt of the promptable models (e.g., micro-sam).

    When the box is editable (see `setEditable`), it can be moved by
    dragging its border, resized with the corner handles, and removed with
    the right-click context menu of its border. Only the border catches the
    mouse, so that new boxes can be drawn inside existing ones (e.g.,
    touching objects). The ID of the object is displayed at the top-left
    corner. `boxData` is the dictionary of the box in the GUI prompts data.
    """
    # Width in screen pixels of the border that catches the mouse
    BORDER_WIDTH_PX = 10

    def __init__(
            self, pos, size, box_id, color, maxBounds=None, editable=True,
            boxData=None
        ):
        rgb = tuple(int(c) for c in color[:3])
        penEditable = pg.mkPen(rgb, width=2)
        super().__init__(
            pos, size, maxBounds=maxBounds, scaleSnap=True,
            translateSnap=True, pen=penEditable,
            hoverPen=pg.mkPen(rgb, width=4), rotatable=False,
            removable=True
        )
        self.box_id = box_id
        self.boxData = boxData
        self._editable = True
        self._penEditable = penEditable
        self._penNotEditable = pg.mkPen(rgb, width=2, style=Qt.DashLine)
        # Resize only with the handles (they respect `maxBounds`, while
        # pyqtgraph's Shift+drag scaling does not)
        self.mouseDragHandler.scaleModifier = None
        self.labelItem = pg.TextItem(str(box_id), color=rgb, anchor=(0, 1))
        self.labelItem.setParentItem(self)
        for pos, center in (
                ([1, 1], [0, 0]), ([0, 0], [1, 1]),
                ([1, 0], [0, 1]), ([0, 1], [1, 0])):
            handle = _PromptBoxHandle(
                self.handleSize, typ='s', pen=self.handlePen,
                hoverPen=self.handleHoverPen, parent=self,
                antialias=self._antialias
            )
            self.addScaleHandle(pos, center, item=handle)
        self.setEditable(editable)

    def setEditable(self, editable):
        # The shape (mouse area) depends on the editability
        self.prepareGeometryChange()
        self._editable = bool(editable)
        self.translatable = self._editable
        # The handles resize the box (see `__init__`)
        self.resizable = False
        for handle in self.getHandles():
            handle.setVisible(self._editable)
        self.setPen(
            self._penEditable if self._editable else self._penNotEditable
        )
        if not self._editable:
            self.setMouseHover(False)

    def isEditable(self):
        return self._editable

    def _boxRect(self):
        return QRectF(
            0, 0, self.state['size'][0], self.state['size'][1]
        ).normalized()

    def _bandWidth(self):
        """Width (in item coordinates) of the band around the border that
        catches the mouse, constant on screen."""
        try:
            pixelLength = self.pixelLength(pg.Point(1, 0))
        except Exception as err:
            pixelLength = None
        if pixelLength:
            return self.BORDER_WIDTH_PX*pixelLength
        rect = self._boxRect()
        return max(min(rect.width(), rect.height())*0.1, 1)

    def boundingRect(self):
        # Include the outer half of the band (`paint` uses the box size)
        margin = self._bandWidth()/2
        return self._boxRect().adjusted(-margin, -margin, margin, margin)

    def shape(self):
        """Only a band centered on the border catches the mouse (hover,
        drag and context menu)."""
        from qtpy.QtGui import QPainterPathStroker

        path = QPainterPath()
        if not getattr(self, '_editable', True):
            # Read-only boxes do not catch the mouse (e.g., pans starting on
            # their border)
            return path
        path.addRect(self._boxRect())
        stroker = QPainterPathStroker()
        stroker.setWidth(self._bandWidth())
        return stroker.createStroke(path)

    def viewTransformChanged(self):
        # The band width depends on the zoom
        self.prepareGeometryChange()
        super().viewTransformChanged()

    def contextMenuEnabled(self):
        return self._editable

    def setSelected(self, s):
        # pyqtgraph selects the ROI (and shows its handles) at the start of
        # every left drag --> keep the handles hidden when not editable
        if s and not getattr(self, "_editable", True):
            return
        super().setSelected(s)

    def mouseClickEvent(self, ev):
        if ev.button() == Qt.MouseButton.RightButton and self.isMoving:
            # Right-click while moving only cancels the move (no menu)
            ev.accept()
            self.cancelMove()
            return
        super().mouseClickEvent(ev)

    def raiseContextMenu(self, ev):
        # Only the box menu (without the menus of the parent items, e.g.,
        # the view and the plot)
        if not self.contextMenuEnabled():
            return
        pos = ev.screenPos()
        self.getMenu().popup(QPoint(int(pos.x()), int(pos.y())))

    def getMenu(self):
        if self.menu is None:
            self.menu = QMenu()
            removeAction = QAction(
                f'Remove box prompt (ID {self.box_id})', self.menu
            )
            removeAction.triggered.connect(self.removeClicked)
            self.menu.addAction(removeAction)
            self.menu.remAct = removeAction
        return self.menu


class SamInputPointsWidget(QWidget):
    sigValueChanged = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        _layout = QHBoxLayout()
        
        self.lineEntry = widgets_line_edits.ElidingLineEdit(parent=self)
        self.lineEntry.setAlignment(Qt.AlignCenter)
        self.lineEntry.editingFinished.connect(self.emitValueChanged)
        
        self.editButton = widgets_push_buttons.editPushButton()
        self.browseButton = widgets_push_buttons.browseFileButton(
            ext={'CSV': '.csv'}, 
            start_dir=filesystem.getMostRecentPath()
        )
        
        _layout.addWidget(self.lineEntry)
        _layout.addWidget(self.editButton)
        _layout.addWidget(self.browseButton)
        
        _layout.setStretch(0, 1)
        _layout.setStretch(1, 0)
        _layout.setStretch(1, 0)
        
        self.browseButton.sigPathSelected.connect(self.browseCsvFiles)
        self.editButton.clicked.connect(self.showInfoEditPoints)
        
        _layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(_layout)
    
    def emitValueChanged(self, text):
        self.sigValueChanged.emit(text)
    
    def showInfoEditPoints(self):
        note = html_utils.to_note(
            'When adding points with the mouse left button you will create a '
            'new object for each point. To add multiple points for the same '
            'object click the right button.'
        )
        txt = html_utils.paragraph(f"""
            To add input points for Segment Anything open the GUI, 
            load the data, and then click on the button<br>
            on the top toolbar called <code>Add points layer</code>.<br><br>
            Select the option "Add points by clicking" and click on the image 
            to add points.<br><br>
            Finally, save the table and browse to the saved file on this widget.
            <br>{note}
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.information(self, 'Info edit points', txt)
    
    def criticalMissingColumn(self, filepath, missing_col):
        txt = html_utils.paragraph(f"""
            [ERROR]: The selected table does not contain the column 
            <code>{missing_col}</code>.<br><br>
            A valid table must contain the columns <code>(x, y, id)</code> 
            with an additional <code>z</code> column for 3D z-stacks data.
        """)
        msg = widgets_message_box.myMessageBox(wrapText=False)
        msg.critical(self, 'Invalid table', txt)
    
    def setValue(self, value: str):
        self.lineEntry.setText(value)
    
    def value(self):
        return self.lineEntry.text()
    
    def cast_dtype(self, value) -> str:
        return str(value)
    
    def browseCsvFiles(self, filepath):
        # Check if metadata.csv file exists with basename and set only the 
        # endname of the file
        df_points = pd.read_csv(filepath)
        for col in ('x', 'y', 'id'):
            if col not in df_points.columns:
                self.criticalMissingColumn(filepath, col)
                return
        
        # Check if basename is present in metadata
        folderpath = os.path.dirname(filepath)
        basename = None
        for file in filesystem.listdir(folderpath):
            if file.endswith('metadata.csv'):
                metadata_csv_path = os.path.join(folderpath, file)
                df = pd.read_csv(metadata_csv_path, index_col='Description')
                try:
                    basename = df.at['basename', 'values']
                except Exception as e:
                    basename = None
                break
        
        # Check if file is inside images folder and get basename
        is_images_folder = folderpath.endswith('Images')
        if is_images_folder:
            images_path = folderpath
            img_filepath = None
            for file in filesystem.listdir(images_path):
                if file.endswith('.tif'):
                    img_filepath = os.path.join(images_path, file)
                    break
                
                if file.endswith('aligned.npz'):
                    img_filepath = os.path.join(images_path, file)
                    break
                
            if img_filepath is not None:
                posData = load.loadData(img_filepath, '', QParent=self)
                posData.getBasenameAndChNames()
                filename = os.path.basename(filepath)
                if filename.startswith(posData.basename):
                    basename = posData.basename
        
        if basename is None:
            self.lineEntry.setText(filepath)
        else:
            filename = os.path.basename(filepath)
            endname = filename[len(basename):]
            self.lineEntry.setText(endname)

class PointsScatterPlotItem(pg.ScatterPlotItem):
    sigHoverEntered = Signal(object, object, object)
    
    def __init__(self, *args, ax=None, show_data_as_tip=False, **kwargs):
        self.textItem = annotate.TextAnnotationsScatterItem(
            size=12, anchor=(1.0, 1.0)
        )
        self.textItem.createSymbols(
            [str(int_id) for int_id in range(200)], includeBold=False
        )
        # self._textItems = {}
        super().__init__(*args, **kwargs)
        self.textItem.setParentItem(self)
        self._font = QFont()
        self._font.setPixelSize(12)
        self.show_data_as_tip = show_data_as_tip
        self.drawIds = True
        self.ax = ax
        self.sigHovered.connect(self.onHover)
        self.lastHoveredPoint = None
    
    def onHover(self, item, points, event):
        if len(points) == 0:
            vb = self.getViewBox()
            vb.setToolTip('')
            return
        
        if self.lastHoveredPoint != points[0]:
            self.sigHoverEntered.emit(item, points, event)
            self.lastHoveredPoint = points[0]
            
        if not self.opts['hoverable']:
            return
        
        if not self.show_data_as_tip:
            return
        
        tip_li = [str(point.data()) for point in points]
        tip = '\n\n'.join(tip_li)
        
        vb = self.getViewBox()
        vb.setToolTip(tip)
        
    
    def setData(self, *args, **kwargs):
        self.clearTextItems()
        super().setData(*args, **kwargs)
        data = kwargs.get('data')
        if data is None:
            return
        
        if len(data) == 0:
            return
        
        first_point_data = data[0]
        if not isinstance(first_point_data, (int, str)):
            return
        
        if not self.drawIds:
            return
        
        if self.show_data_as_tip:
            return
        
        color = self.opts['brush'].color()
        self.textItem.setColors({'id': color.getRgb()})
        size = self.opts['size']
        radius = size/2
        # xx, yy = args
        # for x, y, point_data in zip(xx, yy, data):
        for point in self.points():
            text = str(point.data())
            if not text:
                continue
            
            x, y = point.pos().x(), point.pos().y()
            xt, yt = x+radius-0.5, y-radius+0.5
            opts = {
                'text': text, 
                'bold': False, 
                'color_name': 'id', 
            }
            data = self.textItem.addObjAnnot(
                (xt, yt), anchor=(-0.3, 1.3), **opts
            )
            self.textItem.appendData(data, opts['text'])
        
        self.textItem.draw()
            # hexColor = color.name()
            # htmlText = html_utils.span(
            #     text, color=hexColor, font_size='13pt', bold=True
            # )
            
            # textItem = self._textItems.get((x, y))
            # if textItem is None:
            #     textItem = pg.TextItem(html=htmlText, anchor=(0, 1))
            #     textItem.setParentItem(self)
            #     self._textItems[(x, y)] = textItem
            #     self.ax.addItem(textItem)
            # else:
            #     textItem.setHtml(htmlText)
            # textItem.setPos(x+radius-0.5, y-radius+0.5)      
        
    def clearTextItems(self):
        self.textItem.clearData()
        # for textItem in self._textItems.values():
        #     textItem.setText('')
    
    def clear(self):
        super().clear()
        self.clearTextItems()
    
    def setVisible(self, visible):
        super().setVisible(visible)
        self.textItem.setVisible(visible)


class MagicPromptsToolbar(widgets_toolbars.ToolBar):
    sigPromptTypeChanged = Signal(object, str)
    sigComputeOnZoom = Signal(object)
    sigComputeOnImage = Signal(object)
    sigClearPoints = Signal(object)
    sigClearPointsOnZmom = Signal(object)
    sigInitSelectedModel = Signal(str)
    sigViewModelParams = Signal(
        str, object, list, list, str, object, object, object
    )
    sigInterpolateZslice = Signal(bool)
    sigTrain = Signal(object)
    sigSegmentImage = Signal(object)
    sigSegmentZstack = Signal(object)
    sigSegmentStack = Signal(object)
    sigClearStrokes = Signal(object)
    sigClearStrokesOnZoom = Signal(object)
    sigSaveModel = Signal(object)
    sigLoadModel = Signal(object)
    sigClearBoxes = Signal(object)
    sigClearBoxesOnZoom = Signal(object)
    sigSelectCellsToggled = Signal(object, bool)
    sigSelectAllCells = Signal(object)

    promptTypeTextToKey = {
        'Points': 'point',
        'Boxes': 'box',
        'Brush strokes': 'brush_stroke',
    }
    promptTypeKeyToText = {
        key: text for text, key in promptTypeTextToKey.items()
    }

    def __init__(self, parent=None):
        super().__init__(parent)

        self._parent = parent
        self.isModelTrained = False

        prompt_types = tuple(self.promptTypeTextToKey.keys())

        self.selectModelAction = self.addButton(':select-list.svg')
        self.selectModelAction.setToolTip(
            'Select the promptable model to use'
        )
        
        self.viewModelParamsAction = self.addButton(':view.svg')
        self.viewModelParamsAction.setToolTip(
            'View the currently selected model parameters'
        )
        self.viewModelParamsAction.setDisabled(True)

        # Only for models that can save and load trained models (ConvPaint)
        self.saveModelAction = self.addButton(':file-save.svg')
        self.saveModelAction.setToolTip(
            'Save the trained model...\n\n'
            'The model is saved in the Cell-ACDC settings folder with the '
            'name you choose, and you can load it later with the "Load '
            'trained model" button.\n\n'
            'NOTE: The model is automatically (re-)trained before saving if '
            'the brush strokes changed since the last training.'
        )
        self.loadModelAction = self.addButton(':folder-open.svg')
        self.loadModelAction.setToolTip(
            'Load a trained model...\n\n'
            'Select one of the saved models or browse for a model file '
            'saved somewhere else (e.g., trained in napari).'
        )
        self.setSaveLoadModelSupported(False)

        self.addSeparator()

        self.promptTypeCombobox = self.addComboBox(
            prompt_types, label='Prompt type: ',
        )
        self.promptTypeCombobox.setToolTip(
            'Type of prompts used by the promptable model.\n\n'
            'The available types depend on the selected model (e.g., '
            'ConvPaint uses brush strokes).\n\n'
            'Points and boxes can be combined (e.g., with micro-sam): switch '
            'type to add the other kind of prompts. Prompts with the same ID '
            'belong to the same object.'
        )

        self.addSeparator()

        self.interpolateZslicesCheckbox = self.addCheckBox(
            'Interpolate points on missing z-slices', checked=False
        )
        self.interpolateZslicesCheckbox.setToolTip(
            'If checked, when working with 3D segmentation masks, you can '
            'add points on some z-slices only and the points on the missing '
            'z-slices will be determined by linear interpolation.\n\n'
            'This is useful when working with 2D models that segments '
            'each z-slice independently.\n\n'
            'NOTE: The points will be added only when running the model and '
            'removed afterwards.'
        )

        interpolateSeparator = self.addSeparator()

        self.computeOnZoomAction = self.addButton(':compute-zoom.svg')
        self.computeOnZoomAction.setToolTip(
            'Compute the segmentation on the zoomed area of the image '
            '(faster)'
        )
        
        self.computeAction = self.addButton(':compute.svg')
        self.computeAction.setToolTip(
            'Compute the segmentation on the whole image'
        )
        
        self.clearPointsAction = self.addButton(':clear-points.svg')
        self.clearPointsAction.setToolTip(
            'Clear all points'
        )
        self.clearPointsAction.setDisabled(True)
        
        self.clearPointsActionOnZoom = self.addButton(':clear-points-zoom.svg')
        self.clearPointsActionOnZoom.setToolTip(
            'Clear all points on the zoomed area of the image'
        )
        self.clearPointsActionOnZoom.setDisabled(True)

        # Box prompts widgets (e.g., micro-sam)
        self.clearBoxesAction = self.addButton(':clear-points.svg')
        self.clearBoxesAction.setToolTip(
            'Clear all the box prompts of the current frame'
        )

        self.clearBoxesOnZoomAction = self.addButton(':clear-points-zoom.svg')
        self.clearBoxesOnZoomAction.setToolTip(
            'Clear the box prompts in the zoomed area of the current frame'
        )

        pointsActionsSeparator = self.addSeparator()

        # Brush strokes widgets (e.g., ConvPaint)
        self.brushSizeSpinbox = self.addSpinBox(label='Brush size: ')
        self.brushSizeSpinbox.setMinimum(1)
        self.brushSizeSpinbox.setMaximum(200)
        self.brushSizeSpinbox.setValue(4)
        self.brushSizeSpinbox.setToolTip(
            'Radius in pixels of the brush used to paint the brush strokes'
        )

        brushSizeSeparator = self.addSeparator()

        self.selectCellsAction = self.addButton(
            ':keep_objects.svg', checkable=True
        )
        self.selectCellsAction.setToolTip(
            'Select cells of the segmentation masks to add them to the '
            'training data\n\n'
            'Click on the cells to select (or unselect) them, then press '
            '"Enter" (or the green tick) to add them to the brush strokes of '
            'the current frame: the selected cells as foreground and the '
            'background of the segmentation masks as background.\n\n'
            'NOTE: Make sure that all the cells in the area are segmented, '
            'because the pixels without masks are used as background (use '
            'the "Only in zoomed area" option to add only a cutout).'
        )

        self.selectAllCellsAction, selectAllCellsButton = self.addButton(
            ':select_all.svg', ret_widget=True
        )
        self.selectAllCellsAction.setShortcut('Ctrl+A')
        self.selectAllCellsAction.setToolTip(
            'Select all the cells of the current frame to add them to the '
            'training data\n\n'
            'Shortcut: "Ctrl+A"'
        )
        self.widgetsWithShortcut['Select all cells (magic prompts)'] = (
            self.selectAllCellsAction
        )
        self.widgetsForActions['Select all cells (magic prompts)'] = (
            selectAllCellsButton
        )

        selectCellsSeparator = self.addSeparator()

        self.segmentImageAction, self.segmentImageButton = self.addButton(
            ':segment.svg', ret_widget=True
        )
        self.segmentImageAction.setToolTip(
            'Segment the current image with the trained model\n\n'
            'With 3D segmentation masks (z-stacks), only the displayed '
            'z-slice is segmented (use "Segment z-stack" to segment all the '
            'z-slices).\n\n'
            'NOTE: The model is automatically (re-)trained if it is not '
            'trained yet or if the brush strokes changed since the last '
            'training.\n\n'
            'Right-click for more options (e.g., "Train only, do not '
            'segment").'
        )
        # Right-click menu of the "Segment image" button
        self.trainAction = QAction(
            QIcon(':train.svg'), 'Train only, do not segment', self
        )
        self.trainAction.setToolTip(
            'Train the model on the brush strokes painted on all the '
            'frames and positions without segmenting (the model is '
            're-trained even if the brush strokes did not change)'
        )
        self.segmentImageContextMenu = QMenu(self)
        self.segmentImageContextMenu.setToolTipsVisible(True)
        self.segmentImageContextMenu.addAction(self.trainAction)
        self.segmentImageButton.setContextMenuPolicy(
            Qt.ContextMenuPolicy.CustomContextMenu
        )
        self.segmentImageButton.customContextMenuRequested.connect(
            self.showSegmentImageContextMenu
        )

        self.segmentZstackAction = self.addButton(':3d.svg')
        self.segmentZstackAction.setToolTip(
            'Segment the entire z-stack of the current frame with the '
            'trained model\n\n'
            'Each z-slice is segmented and the objects are then labelled in '
            '3D. Requires 3D segmentation masks.\n\n'
            'NOTE: The model is automatically (re-)trained if it is not '
            'trained yet or if the brush strokes changed since the last '
            'training.'
        )

        self.segmentStackAction = self.addButton(':frames_end.svg')
        self.segmentStackAction.setToolTip(
            'Segment a range of frames with the trained model '
            '(existing masks will be replaced)\n\n'
            'With 3D segmentation masks, the entire z-stack of each frame '
            'is segmented.\n\n'
            'NOTE: The model is automatically (re-)trained if it is not '
            'trained yet or if the brush strokes changed since the last '
            'training.'
        )

        self.clearStrokesAction = self.addButton(':clear-points.svg')
        self.clearStrokesAction.setToolTip(
            'Clear brush strokes on current frame'
        )

        self.clearStrokesOnZoomAction = self.addButton(
            ':clear-points-zoom.svg'
        )
        self.clearStrokesOnZoomAction.setToolTip(
            'Clear brush strokes in the zoomed area of the current frame'
        )

        brushActionsSeparator = self.addSeparator()

        self.infoAction = self.addButton(':info.svg')
        self.infoAction.setToolTip(
            'Show instructions how to use promptable models'
        )

        self.addSeparator()

        self._pointsOnlyActions = [
            self.interpolateZslicesCheckbox.action,
            *interpolateSeparator._actions,
            self.clearPointsAction,
            self.clearPointsActionOnZoom,
        ]
        # Actions used with both point and box prompts
        self._pointsOrBoxesActions = [
            self.computeOnZoomAction,
            self.computeAction,
            *pointsActionsSeparator._actions,
        ]
        self._boxesOnlyActions = [
            self.clearBoxesAction,
            self.clearBoxesOnZoomAction,
        ]
        self._brushStrokesOnlyActions = [
            self.brushSizeSpinbox.labelAction,
            self.brushSizeSpinbox.action,
            *brushSizeSeparator._actions,
            self.selectCellsAction,
            self.selectAllCellsAction,
            *selectCellsSeparator._actions,
            self.segmentImageAction,
            self.segmentZstackAction,
            self.segmentStackAction,
            self.clearStrokesAction,
            self.clearStrokesOnZoomAction,
            *brushActionsSeparator._actions,
        ]

        self.infoAction.triggered.connect(self.showHelp)
        self.selectModelAction.triggered.connect(self.selectModel)
        self.viewModelParamsAction.triggered.connect(self.viewModelParams)
        self.promptTypeCombobox.sigTextChanged.connect(
            self.emitPromptTypeChanged
        )
        self.computeOnZoomAction.triggered.connect(
            self.emitSigComputeOnZoom
        )
        self.computeAction.triggered.connect(
            self.emitSigComputeOnImage
        )
        self.clearPointsAction.triggered.connect(
            self.emitSigClearPoints
        )
        self.clearPointsActionOnZoom.triggered.connect(
            self.emitSigClearPointsOnZoom
        )
        self.interpolateZslicesCheckbox.toggled.connect(
            self.sigInterpolateZslice.emit
        )
        self.trainAction.triggered.connect(self.emitSigTrain)
        self.selectCellsAction.toggled.connect(self.emitSigSelectCellsToggled)
        self.selectAllCellsAction.triggered.connect(self.emitSigSelectAllCells)
        self.segmentImageAction.triggered.connect(self.emitSigSegmentImage)
        self.segmentZstackAction.triggered.connect(self.emitSigSegmentZstack)
        self.segmentStackAction.triggered.connect(self.emitSigSegmentStack)
        self.clearStrokesAction.triggered.connect(self.emitSigClearStrokes)
        self.clearStrokesOnZoomAction.triggered.connect(
            self.emitSigClearStrokesOnZoom
        )
        self.saveModelAction.triggered.connect(self.emitSigSaveModel)
        self.loadModelAction.triggered.connect(self.emitSigLoadModel)
        self.clearBoxesAction.triggered.connect(self.emitSigClearBoxes)
        self.clearBoxesOnZoomAction.triggered.connect(
            self.emitSigClearBoxesOnZoom
        )

        self._updatePromptTypeActions('point')

    def setSaveLoadModelSupported(self, supported):
        """Show the "Save trained model" and "Load trained model" buttons
        only if the initialized model can save and load trained models."""
        self.saveModelAction.setVisible(supported)
        self.loadModelAction.setVisible(supported)

    def emitSigSaveModel(self):
        self.sigSaveModel.emit(self)

    def emitSigLoadModel(self):
        self.sigLoadModel.emit(self)

    def promptType(self):
        text = self.promptTypeCombobox.currentText()
        return self.promptTypeTextToKey.get(text, 'point')

    def setPromptType(self, prompt_type):
        text = self.promptTypeKeyToText.get(prompt_type)
        if text is None:
            raise ValueError(
                f'Prompt type "{prompt_type}" is not valid. Valid prompt '
                f'types are {list(self.promptTypeKeyToText.keys())}'
            )
        self.promptTypeCombobox.setCurrentText(text)
        # Combobox does not emit if text did not change --> update anyway
        self._updatePromptTypeActions(prompt_type)

    def setSupportedPromptTypes(self, prompt_types=None):
        """Enable only the prompt types in `prompt_types` and keep the
        current prompt type if supported (e.g., re-initializing a model that
        supports points and boxes), otherwise select the first supported
        one. If `prompt_types` is None, enable all the prompt types.
        """
        if prompt_types is None:
            supported = list(self.promptTypeKeyToText.keys())
        else:
            if isinstance(prompt_types, str):
                prompt_types = (prompt_types,)
            supported = [
                prompt_type for prompt_type in prompt_types
                if prompt_type in self.promptTypeKeyToText
            ]
            if not supported:
                supported = ['point']

        model = self.promptTypeCombobox.model()
        for i in range(self.promptTypeCombobox.count()):
            text = self.promptTypeCombobox.itemText(i)
            key = self.promptTypeTextToKey.get(text)
            item = model.item(i) if hasattr(model, 'item') else None
            if item is not None:
                item.setEnabled(key in supported)

        if prompt_types is None:
            self.promptTypeCombobox.setDisabled(False)
            self._updatePromptTypeActions(self.promptType())
            return

        current = self.promptType()
        self.setPromptType(current if current in supported else supported[0])
        self.promptTypeCombobox.setDisabled(len(supported) == 1)

    def _setActionsVisible(self, actions, visible):
        for action in actions:
            action.setVisible(visible)

    def _updatePromptTypeActions(self, prompt_type):
        isBrushStrokes = prompt_type == 'brush_stroke'
        isBoxes = prompt_type == 'box'
        isPoints = not isBrushStrokes and not isBoxes
        if not isBrushStrokes:
            # Selecting cells is only for brush strokes
            self.selectCellsAction.setChecked(False)
        self._setActionsVisible(self._pointsOnlyActions, isPoints)
        self._setActionsVisible(
            self._pointsOrBoxesActions, not isBrushStrokes
        )
        self._setActionsVisible(self._boxesOnlyActions, isBoxes)
        self._setActionsVisible(self._brushStrokesOnlyActions, isBrushStrokes)
        self.segmentZstackAction.setVisible(
            isBrushStrokes and getattr(self, '_isZstackData', True)
        )

    def setZstackData(self, isZstack):
        """Show the "Segment z-stack" button (brush strokes) only if the
        loaded data are z-stacks."""
        self._isZstackData = bool(isZstack)
        self._updatePromptTypeActions(self.promptType())

    def emitSigClearBoxes(self):
        self.sigClearBoxes.emit(self)

    def emitSigClearBoxesOnZoom(self):
        self.sigClearBoxesOnZoom.emit(self)

    def emitSigTrain(self):
        self.sigTrain.emit(self)

    def showSegmentImageContextMenu(self, pos):
        if not self.segmentImageAction.isEnabled():
            return
        self.segmentImageContextMenu.exec(
            self.segmentImageButton.mapToGlobal(pos)
        )

    def emitSigSelectCellsToggled(self, checked):
        self.sigSelectCellsToggled.emit(self, checked)

    def emitSigSelectAllCells(self):
        self.sigSelectAllCells.emit(self)

    def emitSigSegmentImage(self):
        self.sigSegmentImage.emit(self)

    def emitSigSegmentZstack(self):
        self.sigSegmentZstack.emit(self)

    def emitSigSegmentStack(self):
        self.sigSegmentStack.emit(self)

    def emitSigClearStrokes(self):
        self.sigClearStrokes.emit(self)

    def emitSigClearStrokesOnZoom(self):
        self.sigClearStrokesOnZoom.emit(self)

    def showHelp(self):
        msg = widgets_message_box.myMessageBox(wrapText=False)
        txt = html_utils.paragraph("""
            This toolbar allows you to use <b>promptable models for 
            segmentation</b>.<br><br>
            
            To use a promptable model, first <b>select the model</b> by clicking on the 
            "Select model" button.<br>
            This will open a dialog where you can select the model to use.<br><br>
            
            After selecting the model, you can <b>view the model parameters</b> 
            by clicking on the "View model parameters" button.<br><br>
            
            To add points to the image, make sure you have points layer correctly 
            initialised. You should see controls<br>
            called "Left-click ID" and "Right-click ID".<br><br>
            
            You can <b>add points</b> for a new object by left-clicking on the image, 
            while you can add points<br>
            for the same object by right-clicking. 
            To <b>delete a point</b>, click on it again.<br><br>
            
            To change the right-click ID, 
            you can either type in the corresponding control,<br>
            or type the object id on the keyboard followed by "Enter".<br><br>
            
            To add <b>negative prompts</b> (i.e., for the background), use the 
            same action you use to delete objects<br>
            (default is middle-click on Windows and Cmd+Click on MacOS).<br><br>
            Note that you can also add object-specific negative prompts (i.e., 
            they affect only that object)<br>
            by adding the negative prompt on the newly segmented object 
            directly.<br><br>
            
            Once you are happy with the added points, click either the 
            "<b>Compute on zoomed area</b>"<br>
            button or the "<b>Compute on whole image</b>" button.<br><br>
            
            Finally, you can <b>clear all points</b> by clicking on the 
            "Clear points" button.<br><br>
            
            Note that you can also save the points by clicking on the
            "<b>Save points</b>" button to load them later and start from
            where you left.<br><br>

            <b>Boxes (e.g., micro-sam)</b><br><br>

            Some models (e.g., micro-sam) also accept <b>box prompts</b>.
            Select "Boxes" in the "Prompt type" selector and:<br><br>

            - <b>Left-click and drag</b> to draw a box around a <b>new
            object</b><br>
            - <b>Right-click and drag</b> to draw a box for the object of
            the "Right-click ID" (e.g., an object with point prompts)<br>
            - <b>Drag the border</b> of a box to move it and its
            <b>corners</b> to resize it (dragging inside a box draws a new
            box, e.g., for touching objects)<br>
            - <b>Right-click</b> on a box (without dragging) to remove it<br>
            - Press another mouse button while drawing to cancel the box<br>
            - Press <b>Ctrl+Z</b> to undo the last change to the boxes<br><br>

            Each object can have one box per z-slice (drawing another box
            replaces it). Boxes and points with the same ID are combined
            (switch prompt type to add points to a box, the box ID is set
            as "Right-click ID").<br>
            With z-stacks, draw the boxes in "single z-slice" mode. With 3D
            segmentation masks, micro-sam segments the z-slices with prompts
            and then propagates the masks to the other z-slices.<br><br>

            Then click "<b>Compute on zoomed area</b>" or "<b>Compute on
            whole image</b>" as for points.<br><br>

            <b>Brush strokes (e.g., ConvPaint)</b><br><br>

            Some models (e.g., ConvPaint) are trained from <b>brush
            strokes</b> instead of points.<br>
            When such a model is selected, the prompt type is set to
            "Brush strokes" and you can paint directly on the image:<br><br>

            - <b>Left-click and drag</b> to paint <b>foreground</b>
            (value 2, shown in red)<br>
            - <b>Right-click and drag</b> to paint <b>background</b>
            (value 1, shown in blue)<br>
            - <b>Shift+drag</b> (left or right) to <b>erase</b> brush
            strokes<br>
            - Set the radius of the brush with the "<b>Brush size</b>"
            control<br>
            - Press <b>Ctrl+Z</b> to undo the last brush stroke<br><br>

            You need both foreground and background strokes. You can paint
            on multiple frames and positions.<br>
            With 3D segmentation masks, paint in "single z-slice" mode
            (projections are display-only).<br>
            With 2D masks of z-stacks, the brush strokes of a frame belong to
            the z-slice (or z-projection) they are painted on.<br><br>

            Note that the brush strokes are <b>not meant to cover all the
            cells</b>. They simply provide some <b>example pixels</b> of
            foreground and background<br>
            that the model learns from. A few strokes on some cells and on
            the background are usually enough<br>
            (add strokes where the model makes mistakes, e.g., at the cell
            borders or between touching cells).<br><br>

            <b>Use existing segmentation masks as training data</b><br><br>

            You can also add cells of the current segmentation masks to the
            training data:<br><br>

            - Click on the "<b>Select cells</b>" button and then click on
            the cells to select (or unselect) them<br>
            - Press <b>Ctrl+A</b> (or the "<b>Select all cells</b>" button)
            to select all the cells of the current frame<br>
            - Press <b>Enter</b> (or the green tick) to add them to the
            brush strokes of the current frame:<br>
            &nbsp;&nbsp;the selected cells are added as <b>foreground</b>
            and the pixels without masks as <b>background</b><br>
            - Check "<b>Only in zoomed area</b>" to add only the cutout
            that is currently displayed<br><br>

            Make sure that all the cells in the area are segmented, because
            the pixels without masks are used as background.<br>
            Existing brush strokes are not modified, and you can undo with
            <b>Ctrl+Z</b>. With 3D segmentation masks, only the displayed
            z-slice is added.<br>
            NOTE: The more pixels you add, the slower the training (e.g.,
            use "Only in zoomed area" with large images).<br><br>

            Then use the following buttons:<br><br>

            - "<b>Segment image</b>": segment the current image and choose
            how to combine the result with the existing masks (with 3D
            masks, only the displayed z-slice)<br>
            - "<b>Segment z-stack</b>": segment all the z-slices of the
            current frame (3D masks)<br>
            - "<b>Segment stack</b>": segment a range of frames
            (existing masks of those frames are replaced)<br><br>

            The model is automatically (re-)trained before segmenting if
            it was not trained yet or if the brush strokes changed since
            the last training.<br>
            To train the model without segmenting (e.g., to re-train a
            loaded model on the current brush strokes),<br>
            <b>right-click</b> on the "Segment image" button and select
            "<b>Train only, do not segment</b>".<br><br>

            You can <b>clear the brush strokes</b> of the current
            frame (or only in the zoomed area) with the corresponding
            buttons.<br><br>

            Finally, you can <b>save the trained model</b> with the
            "<b>Save trained model</b>" button and <b>load it</b> later
            (e.g., on other data) with the "<b>Load trained model</b>"
            button.<br>
            When loading, you can also browse for a model saved somewhere
            else (e.g., a ConvPaint model trained in napari).
        """)
        msg.information(
            self, 'Promptable models help', txt
        )
        
    def emitSigClearPoints(self):
        self.sigClearPoints.emit(self)
    
    def emitSigClearPointsOnZoom(self):
        self.sigClearPointsOnZmom.emit(self)
    
    def emitSigComputeOnZoom(self):
        self.sigComputeOnZoom.emit(self)
    
    def emitSigComputeOnImage(self):
        self.sigComputeOnImage.emit(self)
    
    def selectModel(self):
        if getattr(self, '_isSelectingModel', False):
            # Model selection (or initialization) already in progress
            return

        self._isSelectingModel = True
        try:
            self._selectModel()
        finally:
            self._isSelectingModel = False

    def _selectModel(self):
        win = gui_dialogs_segmentation.SelectPromptableModelDialog(parent=self._parent)
        win.exec_()
        if win.cancel:
            print('Promptable model selection cancelled')
            return

        model_name = win.model_name
        print(f'Preparing promptable model {model_name}...')

        # `_model_name` (used by "View model parameters") is set by the
        # receiver once the model is initialized
        self.sigInitSelectedModel.emit(model_name)
    
    def setInitializedModel(self, init_kwargs, segment_kwargs):
        self._init_kwargs = init_kwargs
        self._segment_kwargs = segment_kwargs
    
    def viewModelParams(self):
        self.sigViewModelParams.emit(
            self._model_name,
            self._acdcPromptSegment,
            self._init_argspecs,
            self._segment_argspecs,
            self._help_url,
            self._init_kwargs,
            self._segment_kwargs,
            self
        )
    
    def emitPromptTypeChanged(self, text):
        prompt_type = self.promptTypeTextToKey.get(text, 'point')
        self._updatePromptTypeActions(prompt_type)
        self.sigPromptTypeChanged.emit(self, prompt_type)


class PointsLayersToolbar(widgets_toolbars.ToolBar):
    sigAddPointsLayer = Signal()
    
    def __init__(self, name='Points layers', parent=None):
        
        super().__init__(name, parent)
        
        self.guiWin = parent
        
        self.setContextMenuPolicy(Qt.PreventContextMenu)
        
        self.addPointsLayerAction = self.addButton(':addPointsLayer.svg')
        
        self.addSeparator()
        
        self.pointsLayersLabel = self.addLabel('Points layers: ')
        
        self.addPointsLayerAction.triggered.connect(
            self.emitAddPointsLayer
        )
        self.doAddPointsZslicesInterpolation = False
    
    def emitAddPointsLayer(self):
        self.sigAddPointsLayer.emit()
    
    def fromActionToDataFrame(self, action, posData, isSegm3D=False):
        df = pd.DataFrame(
            columns=['frame_i', 'Cell_ID', 'z', 'y', 'x', 'id']
        )
        frames_vals = []
        IDs = []
        zz = []
        yy = []
        xx = []
        ids = []
        pos_i = self.guiWin.pos_i
        if pos_i not in action.pointsData:
            printl('No points data for position', pos_i) # should really not happen, but its not a disaster if it does
            return df
        pointsDataPos = action.pointsData[pos_i]
        for frame_i, framePointsData in pointsDataPos.items():
            if posData.SizeZ > 1:
                for z, zSlicePointsData in framePointsData.items():
                    yyxx = zip(
                        zSlicePointsData['y'], zSlicePointsData['x']
                    )
                    for y, x in yyxx:
                        if isSegm3D:
                            ID = posData.lab[int(z), int(y), int(x)]
                        else:
                            ID = posData.lab[int(y), int(x)]
                        frames_vals.append(frame_i)
                        IDs.append(ID)
                        zz.append(z)
                        yy.append(y)
                        xx.append(x)
                    ids.extend(zSlicePointsData['id'])
            else:
                yyxx = zip(framePointsData['y'], framePointsData['x'])
                for y, x in yyxx:
                    ID = posData.lab[int(y), int(x)]
                    frames_vals.append(frame_i)
                    IDs.append(ID)
                    yy.append(y)
                    xx.append(x)
                ids.extend(framePointsData['id'])
        df['frame_i'] = frames_vals
        df['Cell_ID'] = IDs
        df['y'] = yy
        df['x'] = xx
        df['id'] = ids
        if zz:
            df['z'] = zz
        
        df = self.addPointsZslicesInterpolation(df, posData.lab, isSegm3D)
        
        return df

    def addPointsZslicesInterpolation(
            self, 
            df: pd.DataFrame, 
            lab: np.ndarray, 
            isSegm3D: bool
        ):
        if not self.doAddPointsZslicesInterpolation:
            return df
        
        if not isSegm3D:
            return df
        
        if 'z' not in df.columns:
            return df
        
        df_new_rows = []
        for (frame_i, point_id), df_id in df.groupby(['frame_i', 'id']):
            xx = df_id['x'].values
            yy = df_id['y'].values
            zz = df_id['z'].values
            
            p0, d = core.linear_fit_3d(xx, yy, zz)
            
            new_row_df = df_id.iloc[[0]].copy()
            
            z0, z1 = int(np.min(zz)), int(np.max(zz))
            for z in range(z0, z1+1):
                if z in zz:
                    continue
                
                t_int = (z - p0[2]) / d[2]
                x_new, y_new, z_new = p0 + t_int * d
                new_row_df['z'] = round(z_new)
                new_row_df['y'] = round(y_new)
                new_row_df['x'] = round(x_new)
                
                Cell_ID = lab[
                    int(round(z_new)),
                    int(round(y_new)),
                    int(round(x_new))
                ]
                new_row_df['Cell_ID'] = Cell_ID
                
                df_new_rows.append(new_row_df.copy())
        
        if not df_new_rows:
            return df
        
        df_new = pd.concat(df_new_rows, ignore_index=True)
        df = pd.concat([df, df_new], ignore_index=True)
        df = df.sort_values(by=['frame_i', 'id', 'z']).reset_index(drop=True)
        
        return df

class PromptableModelPointsLayerToolbar(PointsLayersToolbar):
    def __init__(self, name='Promptable model points layers', parent=None):
        super().__init__(name, parent=parent)
        
        self.isPointsLayerInit = False
        
        self.addPointsLayerAction.setDisabled(True)
        self.addPointsLayerAction.setVisible(False)
    
    def pointsLayerDf(self, posData, isSegm3D=False):
        for action in self.actions()[1:]:
            if not hasattr(action, 'button'):
                continue
            
            df = self.fromActionToDataFrame(
                action, posData, isSegm3D=isSegm3D
            )
            return df
    
    def scatterItem(self):
        for action in self.actions()[1:]:
            if not hasattr(action, 'button'):
                continue
            
            return action.scatterItem
