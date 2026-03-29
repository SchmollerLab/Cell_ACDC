import inspect

from qtpy.QtGui import QDropEvent, QIcon, QGuiApplication, QPixmap, QDrag, QCursor, QPainterPath, QPen, QColor
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QListWidget, QListWidgetItem, 
    QLabel, QVBoxLayout, QSizePolicy, QScrollArea,
    QAbstractItemView, QApplication, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget, QGraphicsPixmapItem, QGraphicsLineItem,
    QComboBox
)

from . import myutils, apps, widgets, qutils, load
from .apps import QBaseDialog
from qtpy.QtCore import Signal, Qt, QDataStream, QIODevice, QMimeData, QPointF, QTimer

import os

class WorkflowBaseFunctions():
    def __init__(self):
        """ 
        Can be used as base class for a workflow card widgets.
        If you want to create a new workflow card, you should
        set dryrunDialog which should return the dialog so it can be
        screenshotted for the preview with dummy data.
        
        Then, have self.setupDialog() return a properly init dialog.
        

        Use self.posData for all the relevant info.
        Set the number of inputs with self.setInputs({n_inputs: [n_type_1, n_type_2]}) and
        outputs with self.setOutputs({n_outputs: n_type})

        Connect self.valuesChanged_cb to a signal which emits when values are changed
        
        self.input_types_changed() is called when the input types are changed
        self.input_types is a dict: {output_n: input_type}
        self.output_types is also a dict: {input_n: output_type}
         
        """
        
    def runDialog_cb(self, dialog):
        dialog.show()
        
    def getDialogPreview(self, dialog):
        try:
            # Grab a screenshot of the dialog to update preview
            pix = dialog.grab().scaled(220, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            return pix
        except Exception as e:
            # If screenshot fails, just leave preview as is
            print(f"Failed to update preview after dialog close: {e}")
        return None
    
    def setupDialog_cb(self, parent=None, workflowGui=None, posData=None):
        kwargs = {'parent': parent, 'workflowGui': workflowGui, 'posData': posData}
        kwargs_required = inspect.getfullargspec(self.setupDialog).args
        kwargs_to_pass = {k: v for k, v in kwargs.items() if k in kwargs_required}
        dialog = self.setupDialog(**kwargs_to_pass)
        return dialog

    def renderDialogPreview(self, size=None, scale=(220, 110), parent=None, workflowGui=None, posData=None):
        try:
            kwargs = {'parent': parent, 'workflowGui': workflowGui, 'posData': posData}
            kwargs_required = inspect.getfullargspec(self.dryrunDialog).args
            kwargs_to_pass = {k: v for k, v in kwargs.items() if k in kwargs_required}
            dialog = self.dryrunDialog(**kwargs_to_pass)
    
            if size is not None:
                dialog.setFixedSize(*size)  # Use fixed size instead of minimum

            dialog.setAttribute(Qt.WA_DontShowOnScreen, True)
            dialog.show()

            QApplication.processEvents()
            
            # Force layout update
            dialog.update()
            dialog.repaint()
            QApplication.processEvents()

            # Grab the dialog content
            pix = dialog.grab().scaled(220, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation)

            dialog.close()
            dialog.deleteLater()

            return pix.scaled(*scale, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        except Exception as e:
            # Fallback: create a simple placeholder image
            print(f"Failed to render dialog preview: {e}")
            pix = QPixmap(*scale)
            pix.fill(Qt.lightGray)
            return pix
        
    def input_types_changed(self, dialog, input_types):
        dialog.input_types = input_types
        if hasattr(dialog, 'updatedInputTypes'):
            dialog.updatedInputTypes()
        
class WorkflowCombineChannelsFunctions(WorkflowBaseFunctions):
    def __init__(self) -> None:
        self.combineChannelDialog = CombineChannelsSetupDialogWorkflow
        self.title = "Combine and manipulate channels"
    
    def dryrunDialog(self, parent=None): # for rendering the preview in the right panel
        return self.combineChannelDialog(parent=parent)
        
    def setupDialog(self, parent=None):
        self.setInputs({0: ['img', 'segm']})
        self.setOutputs({0: 'img'})
        dialog = self.combineChannelDialog(parent=parent)
        dialog.sigNumStepsChanged.connect(self.numStepsChanged)
        dialog.sigOkClicked.connect(self.updatePreview)
        dialog.sigSaveAsSegmCheckboxToggled.connect(self.saveAsSegmCheckboxToggled)
        dialog.input_types = {0: 'img'}
        return dialog
        
    def numStepsChanged(self, num_steps):
        inputs = {n: ['img', 'segm'] for n in range(num_steps)}
        self.setInputs(inputs)
    
    def saveAsSegmCheckboxToggled(self, save_as_segm):
        self.setOutputs({0: 'segm' if save_as_segm else 'img'})
        
class WorkflowInputSegmFunctions(WorkflowBaseFunctions):
    def __init__(self) -> None:
        self.inputDataDialog = WorkflowInputDataDialog
        self.title = "Select segmentation data"
        
    def dryrunDialog(self, parent=None, workflowGui=None): # for rendering the preview in the right panel
        segm_options = workflowGui.segm_channels
        return self.inputDataDialog(segm_options, 'segmentation', parent=parent)
    
    def setupDialog(self, parent=None, workflowGui=None):
        self.setInputs()
        self.setOutputs({0: 'segm'})
        segm_options = workflowGui.segm_channels
        dialog = self.inputDataDialog(segm_options, 'segmentation', parent=parent)
        dialog.sigOkClicked.connect(self.updatePreview)
        return dialog
        
class WorkflowInputImgFunctions(WorkflowBaseFunctions):
    def __init__(self) -> None:
        self.inputDataDialog = WorkflowInputDataDialog
        self.title = "Select image data"
        
    def dryrunDialog(self, parent=None, workflowGui=None): # for rendering the preview in the right panel
        img_options = workflowGui.img_channels
        return self.inputDataDialog(img_options, 'image', parent=parent)
    
    def setupDialog(self, parent=None, workflowGui=None):
        self.setInputs()
        self.setOutputs({0: 'img'})
        img_options = workflowGui.img_channels

        dialog = self.inputDataDialog(img_options, 'image', parent=parent)
        dialog.sigOkClicked.connect(self.updatePreview)
        dialog.sigUpdateTitle.connect(self.updateTitle)
        return dialog
    
class CombineChannelsSetupDialogWorkflow(apps.CombineChannelsSetupDialog):
    """Here we probably have to override some methods. In this case,
    for example need to change th ok_cb to emit a signal, so I can hide and
    update preview. I also dont want to give specific channel names, but instead
    when a row is added, emit a signal with how many inputs there should be...+
    """
    sigOkClicked = Signal()
    sigCancelClicked = Signal() # for future, if I want to restore if cancel is clicked
    
    def __init__(self, parent=None):
        super().__init__(channel_names=None, parent=parent, hideOnClosing=True)
        
        self.mainLayout.addSpacing(20)

        qutils.hide_and_delete_layout(self.buttonsLayoutSaveGroup)
        qutils.hide_and_delete_layout(self.buttonsLayout)
        buttonsLayout = widgets.CancelOkButtonsLayout()
        self.buttonsLayout = buttonsLayout
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.okButton.clicked.connect(self.sigOkClicked.emit)
        buttonsLayout.cancelButton.clicked.connect(self.sigCancelClicked.emit)
        buttonsLayout.cancelButton.clicked.connect(self.cancelButtonClicked)
        
        self.mainLayout.addLayout(buttonsLayout)
        
    def cancelButtonClicked(self):
        self.hide()
        
    def updatedInputTypes(self):
        self.combineChannelsWidget.setBinarizeCheckableAndNorm()
        self.autoCheckSaveAsSegmCheckbox() # to update the formula preview and error state based on the new input types
        
class WorkflowInputDataDialog(QBaseDialog):
    sigOkClicked = Signal()
    sigCancelClicked = Signal() # for future, if I want to restore if cancel is clicked
    sigUpdateTitle = Signal(str)
    def __init__(self, selection_options, type, parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle(f'Select input {type}')
        
        self.type = type
        self.selection_options = selection_options or []
        self.selected_value = None
        
        # Create layout
        layout = QVBoxLayout()
        
        # Add label
        label = QLabel(f'Select {type} input:')
        layout.addWidget(label)
        
        # Add combo box for selection
        self.selection_widget = QComboBox()
        self.selection_widget.addItems(self.selection_options)
        layout.addWidget(self.selection_widget)
        
        # Add buttons
        buttons_layout = widgets.CancelOkButtonsLayout()
        ok_button = buttons_layout.okButton
        cancel_button = buttons_layout.cancelButton
        ok_button.clicked.connect(self.ok_clicked)
        ok_button.clicked.connect(self.sigOkClicked.emit)
        cancel_button.clicked.connect(self.cancel_clicked)
        cancel_button.clicked.connect(self.sigCancelClicked.emit)
        
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
    
    def ok_clicked(self):
        """Handle OK button click"""
        self.selected_value = self.selection_widget.currentText()
        self.sigUpdateTitle.emit(f'{self.type}: {self.selected_value}')
        self.hide()
    
    def cancel_clicked(self):
        """Handle Cancel button click"""
        self.hide()
    
    def get_selected_value(self):
        """Return the selected value"""
        return self.selected_value

class _WorkflowCardListWidget(QWidget):
    """ Custom QWidget that is being used to set the preview in the QListWidget of the WorkflowSidebar """
    def __init__(self, functions: WorkflowBaseFunctions, parent=None,
                 render_preview=True, posData=None):
        super().__init__(parent=parent)
        
        self.functions = functions
        self.title = functions.title
        self._drag_pixmap = None  # Cache the drag pixmap
        
        self.thumbnail = QLabel(parent=self)
        self.thumbnail.setAlignment(Qt.AlignCenter)
        
        if render_preview:
            preview = functions.renderDialogPreview(parent=self, workflowGui=parent, posData=posData)
            self.thumbnail.setPixmap(preview)
        
        self.titleLabel = QLabel(functions.title, parent=self)
        self.titleLabel.setAlignment(Qt.AlignCenter)
        
        self.thumbnail.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.titleLabel.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self.titleLabel)
        layout.addWidget(self.thumbnail)
        
        self.setLayout(layout)
    
    def getDragPixmap(self):
        """Get or create the cached drag pixmap"""
        if self._drag_pixmap is None:
            self._drag_pixmap = self.grab()
        return self._drag_pixmap
    
    # def openDialog(self):
    #     """Open the workflow dialog and update preview with result"""
    #     try:
    #         updated_pix = self.functions.run_dialog_cb()
    #         if updated_pix is not None:
    #             self.thumbnail.setPixmap(updated_pix)
    #     except Exception as e:
    #         print(f"Error running dialog: {e}")
    
    # def mousePressEvent(self, event):
    #     """Open dialog when clicked"""
    #     if event.button() == Qt.LeftButton:
    #         self.openDialog()


class _ConnectionLine(QGraphicsLineItem):
    """Interactive connection line that can be deleted by right-clicking"""
    def __init__(self, *args, zone=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.zone = zone
        self.setAcceptHoverEvents(True)
        self.setZValue(0)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton and self.zone:
            self.zone.removeConnectionLine(self)
            event.accept()
        else:
            super().mousePressEvent(event)

class _InputOutputCircle(QLabel):
    """Interactive circle label for inputs/outputs that can create connections"""
    def __init__(self, index: int, is_output: bool, card, zone=None, type_info=None):
        super().__init__(str(index))
        self.index = index
        self.is_output = is_output
        self.card = card
        self.zone = zone
        self.type_info = type_info
        self.is_alternating = isinstance(type_info, (list, tuple)) and len(type_info) > 1
        
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "QLabel { min-width: 24px; min-height: 24px; "
            "padding: 2px; font-weight: bold; background-color: transparent; }"
        )
        self.setCursor(Qt.PointingHandCursor)
    
    def _getColorForType(self, type_value):
        """Get border color based on type"""
        if type_value is None:
            return QColor("black")
        if type_value == 'segm':
            return QColor("red")
        elif type_value == 'img':
            return QColor("blue")
        return QColor("black")
    
    def paintEvent(self, event):
        """Custom paint to draw circle with alternating dashed border"""
        super().paintEvent(event)
        
        # Draw the custom border
        rect = self.rect().adjusted(1, 1, -1, -1)
        
        if self.is_alternating and isinstance(self.type_info, (list, tuple)):
            # Draw alternating dashed border
            from qtpy.QtGui import QPainter
            qpainter = QPainter(self)
            qpainter.setRenderHint(QPainter.Antialiasing, True)
            
            # Draw circle with alternating colored dashes
            radius = rect.width() / 2
            center = rect.center()
            
            # Create path for circle
            path = QPainterPath()
            path.addEllipse(center.x() - radius, center.y() - radius, radius * 2, radius * 2)
            
            # Draw alternating dashes around the circle
            num_dashes = 12
            angle_step = 360.0 / num_dashes
            color1 = self._getColorForType(self.type_info[0])
            color2 = self._getColorForType(self.type_info[1])
            
            for i in range(num_dashes):
                angle = i * angle_step
                next_angle = (i + 1) * angle_step
                
                # Convert angles to radians
                import math
                rad1 = math.radians(angle)
                rad2 = math.radians(next_angle)
                
                # Calculate start and end points
                x1 = center.x() + radius * math.cos(rad1)
                y1 = center.y() + radius * math.sin(rad1)
                x2 = center.x() + radius * math.cos(rad2)
                y2 = center.y() + radius * math.sin(rad2)
                
                # Alternate colors
                color = color1 if i % 2 == 0 else color2
                pen = QPen(color)
                pen.setWidth(2)
                pen.setCapStyle(Qt.RoundCap)
                qpainter.setPen(pen)
                
                qpainter.drawArc(int(center.x() - radius), int(center.y() - radius), 
                               int(radius * 2), int(radius * 2), 
                               int(angle * 16), int(angle_step * 16))
            
            qpainter.end()
        else:
            # Draw solid border
            from qtpy.QtGui import QPainter
            qpainter = QPainter(self)
            qpainter.setRenderHint(QPainter.Antialiasing, True)
            
            color = self._getColorForType(self.type_info)
            pen = QPen(color)
            pen.setWidth(2)
            qpainter.setPen(pen)
            
            radius = rect.width() / 2
            center = rect.center()
            qpainter.drawEllipse(center, int(radius), int(radius))
            qpainter.end()
    
    def updateTypeInfo(self, type_info):
        """Update the type info and refresh styling"""
        self.type_info = type_info
        self.is_alternating = isinstance(type_info, (list, tuple)) and len(type_info) > 1
        self.update()
    
    def mousePressEvent(self, event):
        if self.zone is None:
            super().mousePressEvent(event)
            return
            
        if event.button() == Qt.LeftButton:
            # Get global position and convert to scene coordinates
            global_pos = self.mapToGlobal(self.rect().center())
            widget_pos = self.zone.mapFromGlobal(global_pos)
            scene_pos = self.zone.mapToScene(widget_pos)
            
            # Start connection or complete it
            self.zone.handleCircleClick(self.card, self.is_output, self.index, scene_pos)
            event.accept()
        elif event.button() == Qt.RightButton:
            self.zone.cancelConnection()
            event.accept()
        else:
            super().mousePressEvent(event)
        
    # def openDialog(self):
    #     """Open the workflow dialog and update preview with result"""
    #     try:
    #         updated_pix = self.functions.run_dialog_cb()
    #         if updated_pix is not None:
    #             self.thumbnail.setPixmap(updated_pix)
    #     except Exception as e:
    #         print(f"Error running dialog: {e}")
    
    # def mousePressEvent(self, event):
    #     """Open dialog when clicked"""
    #     if event.button() == Qt.LeftButton:
    #         self.openDialog()
class _WorkflowCardZoneWidget(QWidget):
    """Widget which is the card in the workflow zone. It also contains the dialog"""
    def __init__(self, functions: WorkflowBaseFunctions, posData, zone=None, workflowGui=None):
        QWidget.__init__(self, parent=None)
        
        self.posData = posData
        self.workflowGui = workflowGui
        self.functions = functions
        self.title = functions.title
        self._drag_start_pos = None
        self.graphics_proxy = None
        self.zone = zone
        self.is_dragging = False
        self.input_circles = []
        self.output_circles = []
        self.input_types = {}  # {input_idx: type} e.g., {1: 'img', 2: 'segm'}
        self.output_types = {}  # {output_idx: type} e.g., {1: 'img'}
        
        self.functions.posData = posData
        self.functions.setInputs = self.setInputs
        self.functions.setOutputs = self.setOutputs
        self.functions.updatePreview = self.updatePreview
        self.functions.updateTitle = self.updateTitle
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)
        
        self.inputs_container = QWidget()
        self.inputs_layout = QHBoxLayout(self.inputs_container)
        self.inputs_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.inputs_container)
        
        self.titleLabel = QLabel(functions.title)
        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.titleLabel.setAttribute(Qt.WA_TransparentForMouseEvents)
        main_layout.addWidget(self.titleLabel)
        
        self.thumbnail = QLabel()
        self.thumbnail.setAlignment(Qt.AlignCenter)
        self.thumbnail.setAttribute(Qt.WA_TransparentForMouseEvents)
        main_layout.addWidget(self.thumbnail)
        
        self.outputs_container = QWidget()
        self.outputs_layout = QHBoxLayout(self.outputs_container)
        self.outputs_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.outputs_container)
        
        self.dialog = self.functions.setupDialog_cb(parent=self, 
                                                    workflowGui=workflowGui, 
                                                    posData=self.posData)
        self.functions.runDialog_cb(self.dialog)
        
    def updateTitle(self, new_title):
        self.title = new_title
        self.titleLabel.setText(new_title)

    def setZoneOnCircles(self):
        """Set the zone reference on all circles after the card is added to the zone"""
        for circle in self.input_circles + self.output_circles:
            circle.zone = self.zone

    def setInputs(self, n=None):
        # Handle both dict (with types) and int inputs
        if isinstance(n, dict):
            # Extract count and types from dict
            type_info = n
            n = max(type_info.keys()) + 1 if type_info else 0
            self.input_types = {idx + 1: type_info.get(idx) for idx in range(n)}
        elif n is None:
            # None means no inputs
            n = 0
            self.input_types = {}
        else:
            # Simple number, clear types
            self.input_types = {i: None for i in range(1, n + 1)}
        
        while self.inputs_layout.count():
            child = self.inputs_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.input_circles.clear()
        self.inputs_layout.addStretch()
        for i in range(1, n + 1):
            type_value = self.input_types.get(i)
            circle = _InputOutputCircle(i, False, self, self.zone, type_info=type_value)
            self.input_circles.append(circle)
            self.inputs_layout.addWidget(circle)
            self.inputs_layout.addStretch()
        
        # Validate and remove invalid connections for inputs that no longer exist
        if self.zone is not None:
            card_id = getattr(self, 'card_id', None)
            if card_id is not None:
                # Remove connections to inputs that no longer exist or have type mismatch
                self.zone.lines = [line for line in self.zone.lines
                                  if not (line[1][0] == card_id and line[1][1] > n)]
                # Also remove connections with type mismatches
                self.zone.validateConnectionTypes()
                # Defer rebuild until layout has been processed
                QTimer.singleShot(0, self.zone.rebuildConnectionLines)

    def setOutputs(self, n=None):
        # Handle both dict (with types) and int inputs
        if isinstance(n, dict):
            # Extract count and types from dict
            type_info = n
            n = max(type_info.keys()) + 1 if type_info else 0
            self.output_types = {idx + 1: type_info.get(idx) for idx in range(n)}
        elif n is None:
            # None means no outputs
            n = 0
            self.output_types = {}
        else:
            # Simple number, clear types
            self.output_types = {i: None for i in range(1, n + 1)}
        
        while self.outputs_layout.count():
            child = self.outputs_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.output_circles.clear()
        self.outputs_layout.addStretch()
        for i in range(1, n + 1):
            type_value = self.output_types.get(i)
            circle = _InputOutputCircle(i, True, self, self.zone, type_info=type_value)
            self.output_circles.append(circle)
            self.outputs_layout.addWidget(circle)
            self.outputs_layout.addStretch()
        
        # Validate and remove invalid connections for outputs that no longer exist
        if self.zone is not None:
            card_id = getattr(self, 'card_id', None)
            if card_id is not None:
                # Remove connections from outputs that no longer exist
                self.zone.lines = [line for line in self.zone.lines
                                  if not (line[0][0] == card_id and line[0][1] > n)]
                # Update downstream input types and validate compatibility
                self.zone.updateDownstreamInputTypes(card_id)
                self.zone.validateConnectionTypes()
                # Defer rebuild until layout has been processed
                QTimer.singleShot(0, self.zone.rebuildConnectionLines)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.globalPos()
            self.is_dragging = True
        elif event.button() == Qt.MiddleButton:
            self._deleteCard()
        elif event.button() == Qt.RightButton:
            try:
                self.functions.runDialog_cb(self.dialog)
            except Exception as e:
                print(f"Error running dialog: {e}")

    def mouseMoveEvent(self, event):
        if self.is_dragging and self._drag_start_pos is not None and self.graphics_proxy is not None:
            delta = event.globalPos() - self._drag_start_pos
            self.graphics_proxy.moveBy(delta.x(), delta.y())
            self._drag_start_pos = event.globalPos()
            # Update connection lines in real-time while dragging
            if self.zone is not None:
                self.zone.updateConnectionLines()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self._drag_start_pos = None
            # Update all connection lines after card is repositioned
            if self.zone is not None:
                self.zone.updateConnectionLines()
        super().mouseReleaseEvent(event)

    def _deleteCard(self):
        if self.zone is not None:
            self.zone.removeCard(self)
    
    def updatePreview(self):
        preview = self.functions.getDialogPreview(self.dialog)
        if preview is not None:
            self.thumbnail.setPixmap(preview)
            

class WorkflowSidebar(QListWidget):
    """Custom QListWidget that shows the widget preview while dragging"""
    
    def startDrag(self, supportedActions):
        items = self.selectedItems()
        if not items:
            return
        
        item = items[0]
        widget = self.itemWidget(item)
        card_data = item.data(Qt.UserRole)
        
        if widget is None:
            # Fallback to default behavior
            return super().startDrag(supportedActions)
        
        # Get cached pixmap (much faster than grabbing every time)
        drag_pixmap = widget.getDragPixmap() if hasattr(widget, 'getDragPixmap') else widget.grab()
        
        # Create a drag object
        drag = QDrag(self)
        
        # Set the pixmap that will be shown while dragging
        drag.setPixmap(drag_pixmap)
        
        # Set the mime data with card information
        mimedata = QMimeData()
        mimedata.setText(card_data.title if hasattr(card_data, 'title') else "Workflow Card")
        # Store card data for drop handling
        mimedata.setData('application/x-workflow-card', 
                        card_data.title.encode('utf-8') if hasattr(card_data, 'title') else b"Card")
        mimedata.widget = widget  # Store the widget reference for later use
        drag.setMimeData(mimedata)
        
        # Set hot spot based on cursor position relative to the widget
        global_pos = QCursor.pos()
        widget_global_pos = widget.mapToGlobal(widget.rect().topLeft())
        hot_spot = global_pos - widget_global_pos
        drag.setHotSpot(hot_spot)
        
        # Execute the drag
        drag.exec_(supportedActions)


class WorkflowZone(QGraphicsView):
    sigItemDropped = Signal(str, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.setStyleSheet("QGraphicsView { border: 3px dashed gray }")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.setAcceptDrops(True)
        self.cards = dict()  # {unique_card_id: proxy_widget}
        self.lines = []  # List of tuples: ((card_id_from, out_idx), (card_id_to, in_idx))
        self.unique_card_id = 0
        
        # Connection drawing state
        self.connection_start = None  # (card, is_output, index, scene_pos)
        self.temp_line = None
        self.connection_lines = []  # List of visual connection line items
        
        self.placeholder = QLabel("Drop items here")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder_proxy = self.scene.addWidget(self.placeholder)
        self.placeholder_proxy.setPos(0, 0)

    def addCard(self, card_widget, x, y):
        proxy = self.scene.addWidget(card_widget)
        proxy.setPos(QPointF(x, y))
        proxy.setCacheMode(proxy.CacheMode.DeviceCoordinateCache)
        card_widget.graphics_proxy = proxy
        card_widget.zone = self
        card_widget.card_id = self.unique_card_id  # Store the ID on the card widget
        card_widget.setZoneOnCircles()
        self.cards[self.unique_card_id] = proxy
        self.unique_card_id += 1
        if len(self.cards) == 1:
            self.placeholder_proxy.hide()

    def removeCard(self, card_widget):
        card_id = getattr(card_widget, 'card_id', None)
        if card_id is not None and card_id in self.cards:
            proxy = self.cards[card_id]
            if proxy.widget() == card_widget:
                self.scene.removeItem(proxy)
                del self.cards[card_id]
                # Remove all connections involving this card
                self.lines = [line for line in self.lines 
                             if line[0][0] != card_id and line[1][0] != card_id]
                # Remove visual connection lines for this card from the scene
                lines_to_remove = [line for line in self.connection_lines
                                   if (getattr(line, 'start_circle', None) and 
                                       getattr(line, 'start_circle', None).card.card_id == card_id) or
                                      (getattr(line, 'end_circle', None) and 
                                       getattr(line, 'end_circle', None).card.card_id == card_id)]
                for line in lines_to_remove:
                    self.scene.removeItem(line)
                    self.connection_lines.remove(line)
                card_widget.deleteLater()
                if not self.cards:
                    self.placeholder_proxy.show()

    def handleCircleClick(self, card, is_output, index, scene_pos):
        """Handle a click on an input/output circle"""
        if self.connection_start is None:
            # Start a new connection
            self.connection_start = (card, is_output, index, scene_pos)
        else:
            # Complete the connection
            start_card, start_is_output, start_index, start_pos = self.connection_start
            
            # Validate connection (can't connect input to input or output to output)
            if start_is_output != is_output and start_card != card:
                # Check if input already has a connection (inputs can only have 1)
                if not is_output:
                    # Target is an input, check if it already has a connection
                    card_id = getattr(card, 'card_id', None)
                    if card_id is not None:
                        existing = [line for line in self.lines if line[1][0] == card_id and line[1][1] == index]
                        if existing:
                            print(f"Input {index} on this card already has a connection. Inputs can only have one connection.")
                            self.cancelConnection()
                            return

                # Valid connection
                self.createConnection(start_card, start_is_output, start_index, card, is_output, index)
            
            # Clear connection state regardless
            self.cancelConnection()

    def _areTypesCompatible(self, output_type, input_type):
        """Check if output_type is compatible with input_type.
        - If input_type is a list, output_type must be in the list
        - If input_type is a string, it must match output_type exactly
        - If either is None, they're compatible
        """
        if output_type is None or input_type is None:
            return True
        
        if isinstance(input_type, (list, tuple)):
            # Input accepts a list of types
            return output_type in input_type
        else:
            # Input expects a specific type
            return output_type == input_type

    def createConnection(self, card1, is_out1, idx1, card2, is_out2, idx2):
        """Create a visual connection between two circles"""
        # Store connection with card IDs
        card1_id = getattr(card1, 'card_id', None)
        card2_id = getattr(card2, 'card_id', None)
        
        if card1_id is None or card2_id is None:
            return
        
        # Determine which is output and which is input
        if is_out1:  # card1 is output, card2 is input
            output_card = card1
            output_idx = idx1
            input_card = card2
            input_idx = idx2
        else:  # card1 is input, card2 is output - swap them
            output_card = card2
            output_idx = idx2
            input_card = card1
            input_idx = idx1
        
        # Check type compatibility
        output_type = output_card.output_types.get(output_idx)
        input_type = input_card.input_types.get(input_idx)
        
        if not self._areTypesCompatible(output_type, input_type):
            print(f"Type mismatch: cannot connect output type '{output_type}' to input type '{input_type}'")
            return
        
        # If input has no type, set it to the output type
        if output_type is not None:
            input_card.input_types[input_idx] = output_type
            # Trigger input types changed callback
            input_card.functions.input_types_changed(input_card.dialog, input_card.input_types)
        
        # Store connection with card IDs
        if is_out1:  # card1 is output, card2 is input
            self.lines.append(((card1_id, idx1), (card2_id, idx2)))
        else:  # card1 is input, card2 is output - swap them
            self.lines.append(((card2_id, idx2), (card1_id, idx1)))
        
        self.drawConnection(card1, is_out1, idx1, card2, is_out2, idx2)

    def drawConnection(self, card1, is_out1, idx1, card2, is_out2, idx2):
        """Draw a connection line between two circles"""
        # Get the global positions of the circles
        proxy1 = card1.graphics_proxy
        proxy2 = card2.graphics_proxy
        
        if is_out1:
            circles1 = card1.output_circles
            start_circle = circles1[idx1 - 1] if idx1 - 1 < len(circles1) else None
        else:
            circles1 = card1.input_circles
            start_circle = circles1[idx1 - 1] if idx1 - 1 < len(circles1) else None
        
        if is_out2:
            circles2 = card2.output_circles
            end_circle = circles2[idx2 - 1] if idx2 - 1 < len(circles2) else None
        else:
            circles2 = card2.input_circles
            end_circle = circles2[idx2 - 1] if idx2 - 1 < len(circles2) else None
        
        if start_circle and end_circle:
            start_global = start_circle.mapToGlobal(start_circle.rect().center())
            end_global = end_circle.mapToGlobal(end_circle.rect().center())
            
            start_scene = self.mapToScene(self.mapFromGlobal(start_global))
            end_scene = self.mapToScene(self.mapFromGlobal(end_global))
            
            # Create line using custom ConnectionLine class
            line = _ConnectionLine(start_scene.x(), start_scene.y(), end_scene.x(), end_scene.y(), zone=self)
            pen = QPen(QColor(0, 0, 0))
            pen.setWidth(2)
            line.setPen(pen)
            
            # Store circle references for later updates
            line.start_circle = start_circle
            line.end_circle = end_circle
            
            self.scene.addItem(line)
            self.connection_lines.append(line)

    def removeConnectionLine(self, line):
        """Remove a connection line when right-clicked"""
        if line in self.connection_lines:
            self.scene.removeItem(line)
            self.connection_lines.remove(line)
            # Remove from lines list
            if hasattr(line, 'start_circle') and hasattr(line, 'end_circle'):
                start_card_id = getattr(line.start_circle.card, 'card_id', None)
                end_card_id = getattr(line.end_circle.card, 'card_id', None)
                start_idx = line.start_circle.index
                end_idx = line.end_circle.index
                if start_card_id is not None and end_card_id is not None:
                    self.lines = [l for l in self.lines 
                                 if not (l[0] == (start_card_id, start_idx) and l[1] == (end_card_id, end_idx))]

    def updateConnectionLines(self):
        """Update all connection line positions based on current circle positions"""
        lines_to_remove = []
        for line in self.connection_lines:
            if hasattr(line, 'start_circle') and hasattr(line, 'end_circle'):
                start_circle = line.start_circle
                end_circle = line.end_circle
                
                # Check if circles still exist
                if start_circle not in start_circle.card.output_circles + start_circle.card.input_circles:
                    lines_to_remove.append(line)
                    continue
                if end_circle not in end_circle.card.output_circles + end_circle.card.input_circles:
                    lines_to_remove.append(line)
                    continue
                
                # Get current global positions
                start_global = start_circle.mapToGlobal(start_circle.rect().center())
                end_global = end_circle.mapToGlobal(end_circle.rect().center())
                
                # Convert to scene coordinates
                start_scene = self.mapToScene(self.mapFromGlobal(start_global))
                end_scene = self.mapToScene(self.mapFromGlobal(end_global))
                
                # Update line
                line.setLine(start_scene.x(), start_scene.y(), end_scene.x(), end_scene.y())
        
        # Remove invalid lines
        for line in lines_to_remove:
            self.removeConnectionLine(line)

    def rebuildConnectionLines(self):
        """Rebuild all visual connection lines from the lines data structure.
        Used when circles are recreated due to port count changes."""
        # Remove all existing visual lines
        for line in self.connection_lines[:]:
            self.scene.removeItem(line)
        self.connection_lines.clear()
        
        # Recreate visual lines for all valid connections in self.lines
        for line_data in self.lines:
            (start_card_id, start_idx), (end_card_id, end_idx) = line_data
            
            # Get the card widgets
            start_proxy = self.cards.get(start_card_id)
            end_proxy = self.cards.get(end_card_id)
            
            if start_proxy is None or end_proxy is None:
                continue
            
            start_card = start_proxy.widget()
            end_card = end_proxy.widget()
            
            if start_card is None or end_card is None:
                continue
            
            # Get the circles
            start_circle = start_card.output_circles[start_idx - 1] if start_idx - 1 < len(start_card.output_circles) else None
            end_circle = end_card.input_circles[end_idx - 1] if end_idx - 1 < len(end_card.input_circles) else None
            
            if start_circle is None or end_circle is None:
                continue
            
            # Draw the connection with the new circles
            start_global = start_circle.mapToGlobal(start_circle.rect().center())
            end_global = end_circle.mapToGlobal(end_circle.rect().center())
            
            start_scene = self.mapToScene(self.mapFromGlobal(start_global))
            end_scene = self.mapToScene(self.mapFromGlobal(end_global))
            
            # Create line using custom ConnectionLine class
            line = _ConnectionLine(start_scene.x(), start_scene.y(), end_scene.x(), end_scene.y(), zone=self)
            pen = QPen(QColor(0, 0, 0))
            pen.setWidth(2)
            line.setPen(pen)
            
            # Store circle references for later updates
            line.start_circle = start_circle
            line.end_circle = end_circle
            
            self.scene.addItem(line)
            self.connection_lines.append(line)

    def validateConnectionTypes(self):
        """Remove connections where output type doesn't match input type"""
        lines_to_remove = []
        for line_data in self.lines:
            (start_card_id, start_idx), (end_card_id, end_idx) = line_data
            
            # Get the card widgets
            start_proxy = self.cards.get(start_card_id)
            end_proxy = self.cards.get(end_card_id)
            
            if start_proxy is None or end_proxy is None:
                lines_to_remove.append(line_data)
                continue
            
            start_card = start_proxy.widget()
            end_card = end_proxy.widget()
            
            if start_card is None or end_card is None:
                lines_to_remove.append(line_data)
                continue
            
            # Check type compatibility
            output_type = start_card.output_types.get(start_idx)
            input_type = end_card.input_types.get(end_idx)
            
            # Remove connection if types don't match
            if not self._areTypesCompatible(output_type, input_type):
                lines_to_remove.append(line_data)
        
        # Remove incompatible connections from both data and visual lists
        for line_data in lines_to_remove:
            self.lines.remove(line_data)
            # Also remove the visual line
            line_visual_to_remove = None
            for line in self.connection_lines:
                if (hasattr(line, 'start_circle') and hasattr(line, 'end_circle') and
                    getattr(line.start_circle.card, 'card_id', None) == line_data[0][0] and
                    line.start_circle.index == line_data[0][1] and
                    getattr(line.end_circle.card, 'card_id', None) == line_data[1][0] and
                    line.end_circle.index == line_data[1][1]):
                    line_visual_to_remove = line
                    break
            if line_visual_to_remove:
                self.scene.removeItem(line_visual_to_remove)
                self.connection_lines.remove(line_visual_to_remove)

    def updateDownstreamInputTypes(self, source_card_id):
        """Update input types of all cards connected to outputs of source_card_id"""
        source_proxy = self.cards.get(source_card_id)
        if source_proxy is None:
            return
        source_card = source_proxy.widget()
        if source_card is None:
            return
        
        # Find all connections from this card
        for line_data in self.lines:
            (start_card_id, start_idx), (end_card_id, end_idx) = line_data
            if start_card_id == source_card_id:
                # This is a connection from source_card
                end_proxy = self.cards.get(end_card_id)
                if end_proxy is not None:
                    end_card = end_proxy.widget()
                    if end_card is not None:
                        # Update the input type based on the output type
                        output_type = source_card.output_types.get(start_idx)
                        if output_type is not None:
                            end_card.input_types[end_idx] = output_type
                            # Trigger input types changed callback
                            end_card.functions.input_types_changed(end_card.dialog, end_card.input_types)

    def cancelConnection(self):
        """Cancel the current connection being drawn"""
        if self.temp_line is not None:
            self.scene.removeItem(self.temp_line)
            self.temp_line = None
        self.connection_start = None

    def mouseMoveEvent(self, event):
        """Update the temporary connection line while dragging"""
        if self.connection_start is not None:
            start_card, start_is_output, start_index, start_pos = self.connection_start
            current_scene_pos = self.mapToScene(event.pos())
            
            # Remove old temporary line
            if self.temp_line is not None:
                self.scene.removeItem(self.temp_line)
            
            # Draw new temporary line
            self.temp_line = QGraphicsLineItem(start_pos.x(), start_pos.y(), 
                                               current_scene_pos.x(), current_scene_pos.y())
            pen = QPen(QColor(100, 100, 100))
            pen.setWidth(2)
            pen.setStyle(Qt.DashLine)
            self.temp_line.setPen(pen)
            self.scene.addItem(self.temp_line)
        
        super().mouseMoveEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasFormat('application/x-workflow-card') or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):  # without this, drop gets rejected
        if event.mimeData().hasFormat('application/x-workflow-card') or event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        scene_pos = self.mapToScene(event.pos())
        mime = event.mimeData()
        if mime.hasFormat('application/x-workflow-card'):
            # Decode card name from custom MIME data
            card_name = bytes(mime.data('application/x-workflow-card')).decode('utf-8')
            self.sigItemDropped.emit(card_name, int(scene_pos.x()), int(scene_pos.y()))
            event.acceptProposedAction()
        else:
            event.ignore()

    def mousePressEvent(self, event):
        """Handle right-click to cancel connection"""
        if event.button() == Qt.RightButton and self.connection_start is not None:
            self.cancelConnection()
            event.accept()
        else:
            super().mousePressEvent(event)

class WorkflowGui(QMainWindow):
    sigClosed = Signal(object)
        
    def __init__(
            self, app, parent=None, buttonToRestore=None,
            mainWin=None, version=None, launcherSlot=None,
            selectedExpPaths=None
        ):
        """Initializer."""

        super().__init__(parent)
        
        from .config import parser_args
        self.debug = parser_args['debug']
        
        self.buttonToRestore = buttonToRestore
        self.launcherSlot = launcherSlot
        self.mainWin = mainWin
        self.app = app
        self.closeGUI = False
        self._acdc_version = myutils.read_version()
        self._version = version
        self.selectedExpPaths = selectedExpPaths

        self.setAcceptDrops(True)
        self._appName = 'Cell-ACDC'
                
        self.initData()
        self.setupUI()
        
    def initData(self):
        self.data = {}
        self.posData = None # store first data loaded, as this is relevant for UI.
        # all other data is only relevant for running the workflow
        self.img_channels = set()
        self.segm_channels = set()
        
        i = 0
        for path, positions in self.selectedExpPaths.items():
            for pos in positions:
                path_loc = os.path.join(path, pos, 'Images')
                posData = load.loadData(path_loc, '?')
                posData.total_path = path_loc
                self.data[i] = posData
                
            if i == 0:
                self.posData = posData

        for i, posData in enumerate(self.data.values()):
            basename, chNames_loc = myutils.getBasenameAndChNames(
                posData.total_path
            )
            segm_files = load.get_segm_files(posData.total_path)
            segm_endnames = load.get_endnames(posData.total_path, segm_files)
            basename, chNames_loc = myutils.getBasenameAndChNames(
                posData.total_path
            )
            segm_files = load.get_segm_files(posData.total_path)
            segm_endnames = load.get_endnames(
                basename, segm_files
            )
            if i == 0:
                self.img_channels = set(chNames_loc)
                self.segm_channels = set(segm_endnames)
                continue
            
            self.img_channels = self.img_channels.intersection(chNames_loc)
            self.segm_channels = self.segm_channels.intersection(segm_endnames)

    def _setupDragCard(self, functions: WorkflowBaseFunctions):
        channels_card = QListWidgetItem()
        channels_card.setData(Qt.UserRole, functions)

        card_widget = _WorkflowCardListWidget(functions, parent=self, posData=self.posData)
        
        channels_card.setSizeHint(card_widget.sizeHint())
        self.sidebar.addItem(channels_card)
        self.sidebar.setItemWidget(channels_card, card_widget)

    def addDroppedCard(self, card: str, x: int, y: int):
        """Handle a card dropped into the WorkflowZone at position (x, y)"""
        # Find the card data from the sidebar
        card_data = None
        for i in range(self.sidebar.count()):
            item = self.sidebar.item(i)
            data = item.data(Qt.UserRole)
            if data and hasattr(data, 'title') and data.title == card:
                card_data = data
                break
        
        if not card_data:
            return
        
        # Create a new zone card widget with the card data, passing zone reference
        card_widget = _WorkflowCardZoneWidget(card_data, posData=self.posData, zone=self.dropZone, workflowGui=self)
        
        # Add it directly to the drop zone at the specified position
        self.dropZone.addCard(card_widget, x, y)

    def setupUI(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Sidebar
        self.sidebar = WorkflowSidebar()
        self.sidebar.setDragEnabled(True)
        self.sidebar.setFixedWidth(200)
        self.sidebar.setDragDropMode(QAbstractItemView.DragOnly)
        self.sidebar.setSelectionMode(QAbstractItemView.SingleSelection)

        functions = WorkflowCombineChannelsFunctions()
        self._setupDragCard(functions)
        
        functions = WorkflowInputImgFunctions()
        self._setupDragCard(functions)
        
        functions = WorkflowInputSegmFunctions()
        self._setupDragCard(functions)
        
        layout.addWidget(self.sidebar)
        
        # Main panel
        self.mainPanel = QWidget()
        self.mainPanel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mainLayout = QVBoxLayout(self.mainPanel)

        # Create the drop zone that will directly hold dropped cards
        self.dropZone = WorkflowZone()
        self.dropZone.sigItemDropped.connect(self.addDroppedCard)
        mainLayout.addWidget(self.dropZone)

        layout.addWidget(self.mainPanel)

        # Set window size based on screen resolution
        screen = QGuiApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.8)
        height = int(screen.height() * 0.8)
        self.setGeometry((screen.width() - width) // 2, (screen.height() - height) // 2, width, height)

    def run(self, module='acdc_workflow', logs_path=None):
        self.setWindowIcon()
        self.setWindowTitle()
        
        logger, logs_path, log_path, log_filename = myutils.setupLogger(
            module=module, logs_path=logs_path, caller=self._appName
        )
        
        if self._version is not None:
            logger.info(f'Initializing GUI v{self._version}')
        else:
            logger.info(f'Initializing GUI...')
            
        self.module = module
        self.logger = logger
        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path
        
        self.show()
    
    def closeEvent(self, event):
        if self.closeGUI:
            event.ignore()
            return
        
        self.closeGUI = True
        self.sigClosed.emit(self)
        
    def setWindowIcon(self, icon=None):
        if icon is None:
            icon = QIcon(":icon.ico")
        super().setWindowIcon(icon)
    
    def setWindowTitle(self, title=None):
        if title is None:
            title = f'Cell-ACDC v{self._acdc_version} - workflow GUI'
        super().setWindowTitle(title)
        
    # handle key press events for shortcuts
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Q and self.debug:
            print(self.dropZone.lines)
        super().keyPressEvent(event)