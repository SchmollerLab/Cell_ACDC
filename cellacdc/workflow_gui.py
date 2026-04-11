
from qtpy.QtGui import  QIcon, QGuiApplication, QDrag, QCursor, QPainterPath, QPen, QColor, QPolygonF, QPainter
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QListWidget, QListWidgetItem, 
    QLabel, QVBoxLayout, QSizePolicy, QPlainTextEdit,
    QAbstractItemView, QGraphicsView, QGraphicsScene, QGraphicsLineItem, QAction,
    QMessageBox, QPushButton, QApplication, QDialog, QDockWidget, QMenu
)
from qtpy.QtCore import Signal, Qt, QMimeData, QPointF, QTimer, QObject
import qtpy

from . import myutils, load, printl, workflow_dialogs, widgets, apps, workflow_default_save_folderpath, qutils
from .workflow_dialogs import (
    WorkflowBaseFunctions,
)
from .workflow_typing import (WfImageDC, 
                              WfSegmDC, 
                              WfMetricsDC, 
                              workflow_type_name, 
                              make_workflow_data_class,
                              is_workflow_data_class)
from .acdc_regex import to_alphanumeric

import os
import shutil
import copy
import logging
import traceback
from contextlib import contextmanager
from datetime import datetime


SUPPORTED_EXTENSIONS_CARD_SETTINGS = ['.json', '.txt', '.ini']
FUNCTIONS_TO_ADD = [
            workflow_dialogs.WorkflowCombineChannelsFunctions(),
            workflow_dialogs.WorkflowPreProcessFunctions(),
            workflow_dialogs.WorkflowPostProcessSegmFunctions(),
            workflow_dialogs.WorkflowInputImgFunctions(),
            workflow_dialogs.WorkflowInputSegmFunctions(),
            workflow_dialogs.WorkflowSetMeasurementsFunctions(),
            workflow_dialogs.SegmentFunctions(),
            workflow_dialogs.TrackingFunctions(),
            ]
class _WorkflowGuiLogEmitter(QObject):
    sigLogMessage = Signal(str)


class _WorkflowGuiLogHandler(logging.Handler):
    """Logging handler that appends records to a QPlainTextEdit."""

    def __init__(self, text_edit):
        super().__init__()
        self._text_edit = text_edit
        self._emitter = _WorkflowGuiLogEmitter()
        self._emitter.sigLogMessage.connect(self._appendText)

    def _appendText(self, text):
        if self._text_edit is None:
            return
        self._text_edit.appendPlainText(text)
        scroll_bar = self._text_edit.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def emit(self, record):
        try:
            text = self.format(record)
        except Exception:
            text = record.getMessage()
        self._emitter.sigLogMessage.emit(str(text))

class _WorkflowCardListWidget(QWidget):
    """Widget representing a workflow card item in the sidebar.
    
    Displays a preview thumbnail and title for a workflow card in the sidebar.
    This widget is used in the WorkflowSidebar and supports drag-and-drop
    operations to add cards to the workflow zone.
    
    Args:
        functions (WorkflowBaseFunctions): The workflow functions object containing
            the dialog class and title.
        parent: Parent widget.
        render_preview (bool): Whether to render a preview image. Defaults to True.
        posData: Position data for preview rendering.
    
    Attributes:
        functions (WorkflowBaseFunctions): The associated workflow functions.
        thumbnail (QLabel): Label displaying the preview image.
        titleLabel (QLabel): Label displaying the card title.
    """
    def __init__(self, functions: WorkflowBaseFunctions, parent=None,
                 render_preview=True, posData=None):
        super().__init__(parent=parent)
        
        self.functions = functions
        self.title = functions.title
        self._drag_pixmap = None  # Cache the drag pixmap
        
        self.thumbnail = QLabel(parent=self)
        self.thumbnail.setAlignment(Qt.AlignCenter)
        
        if render_preview:
            preview = functions.renderDialogPreview(parent=self, 
                                                    workflowGui=parent, 
                                                    posData=posData)
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
        """Get or create the cached drag pixmap for this card.
        
        The pixmap is created once and cached for efficiency, so subsequent
        drag operations don't need to re-render the widget.
        
        Returns:
            QPixmap: A screenshot/rendering of this card widget.
        """
        if self._drag_pixmap is None:
            self._drag_pixmap = self.grab()
        return self._drag_pixmap
    
class _ConnectionLine(QGraphicsLineItem):
    """Visual representation of a connection between workflow card ports.
    
    An interactive line that connects output and input circles between workflow cards.
    Can be deleted by right-clicking on it. The line updates dynamically as cards
    are moved around the workflow zone.
    
    Args:
        *args: Arguments passed to QGraphicsLineItem.
        zone (WorkflowZone): Reference to the parent WorkflowZone.
        **kwargs: Keyword arguments passed to QGraphicsLineItem.
    
    Attributes:
        zone (WorkflowZone): The parent workflow zone.
        start_circle (_InputOutputCircle): The source output circle.
        end_circle (_InputOutputCircle): The target input circle.
    """
    def __init__(self, *args, zone=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.workflowZone = zone
        self.setAcceptHoverEvents(True)
        self.setZValue(0)
        self._blink_timer = None
        self._blink_stop_timer = None
        self._error_highlight_on = False
        self._original_pen_for_highlight = None

    def _formatTypeInfo(self, type_info):
        """Convert workflow type payloads to concise tooltip text."""
        if type_info is None:
            return 'Any/Unknown'

        if isinstance(type_info, (list, tuple)):
            if len(type_info) == 0:
                return 'Any/Unknown'
            return ' | '.join(self._formatTypeInfo(t) for t in type_info)

        if not is_workflow_data_class(type_info):
            return str(type_info)

        type_name = workflow_type_name(type_info)
        size_z = getattr(type_info, 'SizeZ', None)
        size_t = getattr(type_info, 'SizeT', None)
        size_y = getattr(type_info, 'SizeY', None)
        size_x = getattr(type_info, 'SizeX', None)

        details = []
        if size_z is not None:
            details.append(f'Z={size_z}')
        if size_t is not None:
            details.append(f'T={size_t}')
        if size_y is not None:
            details.append(f'Y={size_y}')
        if size_x is not None:
            details.append(f'X={size_x}')
        if details:
            return f"{type_name} ({', '.join(details)})"
        return type_name

    def _buildTooltipText(self):
        """Build tooltip text from current source/target port typing."""
        start_circle = getattr(self, 'start_circle', None)
        end_circle = getattr(self, 'end_circle', None)
        if start_circle is None or end_circle is None:
            return 'Connection'

        output_card = getattr(start_circle, 'card', None)
        input_card = getattr(end_circle, 'card', None)
        if output_card is None or input_card is None:
            return 'Connection'

        output_port = getattr(start_circle, 'index', 0)
        input_port = getattr(end_circle, 'index', 0)

        output_type = getattr(output_card, 'output_types', {}).get(output_port)
        input_accepted = input_card._getDialogInputTypesAccepted().get(input_port)
        input_current = input_card._getDialogInputTypes().get(input_port)

        output_card_id = getattr(output_card, 'card_id', '?')
        input_card_id = getattr(input_card, 'card_id', '?')
        output_card_title = getattr(output_card, 'title', '')
        input_card_title = getattr(input_card, 'title', '')

        output_label = f"Card {output_card_id}"
        input_label = f"Card {input_card_id}"
        if output_card_title:
            output_label += f" ({output_card_title})"
        if input_card_title:
            input_label += f" ({input_card_title})"

        return (
            f"{output_label} output {output_port + 1} -> {input_label} input {input_port + 1}\n"
            f"Output type: {self._formatTypeInfo(output_type)}\n"
            f"Input accepts: {self._formatTypeInfo(input_accepted)}\n"
            f"Input current: {self._formatTypeInfo(input_current)}"
        )

    def hoverEnterEvent(self, event):
        self.setToolTip(self._buildTooltipText())
        super().hoverEnterEvent(event)

    def hoverMoveEvent(self, event):
        # Keep tooltip synced with live type updates while hovering.
        self.setToolTip(self._buildTooltipText())
        super().hoverMoveEvent(event)

    def paint(self, painter, option, widget=None):
        """Paint a directed connection with an arrowhead at the target end."""
        line = self.line()
        pen = self.pen()

        painter.setRenderHint(painter.Antialiasing, True)
        painter.setPen(pen)

        dx = line.dx()
        dy = line.dy()
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            return

        ux = dx / length
        uy = dy / length
        px = -uy
        py = ux

        arrow_size = max(8.0, pen.widthF() * 4.0)
        arrow_half_width = arrow_size * 0.5

        tip = line.p2()
        line_end = QPointF(tip.x() - ux * (arrow_size * 0.8), tip.y() - uy * (arrow_size * 0.8))

        # Draw the shaft slightly shorter so the arrowhead is visually clear.
        painter.drawLine(line.p1(), line_end)

        base = QPointF(tip.x() - ux * arrow_size, tip.y() - uy * arrow_size)
        left = QPointF(base.x() + px * arrow_half_width, base.y() + py * arrow_half_width)
        right = QPointF(base.x() - px * arrow_half_width, base.y() - py * arrow_half_width)

        arrow_head = QPolygonF([tip, left, right])
        painter.setBrush(pen.color())
        painter.drawPolygon(arrow_head)
    
    def mousePressEvent(self, event):
        """Handle mouse press to delete the connection line on right-click.
        
        Args:
            event: The mouse event.
        """
        if event.button() == Qt.RightButton and self.workflowZone:
            self.workflowZone.removeConnectionLine(self)
            event.accept()
        elif event.button() == Qt.MiddleButton and self.workflowZone:
            self.workflowZone.removeConnectionLine(self)
            event.accept()
        else:
            super().mousePressEvent(event)

    def highlightError(self, duration_ms=2200):
        """Blink the line in red to draw attention to an invalid connection."""
        self.clearErrorHighlight(reset_pen=False)
        self._original_pen_for_highlight = QPen(self.pen())
        self._error_highlight_on = False

        parent_obj = self.workflowZone if self.workflowZone is not None else None
        self._blink_timer = QTimer(parent_obj)
        self._blink_timer.timeout.connect(self._toggleErrorHighlightPen)
        self._blink_timer.start(120)

        self._blink_stop_timer = QTimer(parent_obj)
        self._blink_stop_timer.setSingleShot(True)
        self._blink_stop_timer.timeout.connect(self.clearErrorHighlight)
        self._blink_stop_timer.start(duration_ms)

    def _toggleErrorHighlightPen(self):
        if self._original_pen_for_highlight is None:
            return

        if self._error_highlight_on:
            self.setPen(QPen(self._original_pen_for_highlight))
        else:
            highlight_pen = QPen(self._original_pen_for_highlight)
            highlight_pen.setColor(QColor('#c62828'))
            highlight_pen.setWidth(max(3, self._original_pen_for_highlight.width() + 1))
            self.setPen(highlight_pen)

        self._error_highlight_on = not self._error_highlight_on
        self.update()

    def clearErrorHighlight(self, reset_pen=True):
        """Stop line error blink animation and optionally restore original style."""
        if self._blink_timer is not None:
            self._blink_timer.stop()
            self._blink_timer = None

        if self._blink_stop_timer is not None:
            self._blink_stop_timer.stop()
            self._blink_stop_timer = None

        self._error_highlight_on = False
        if reset_pen and self._original_pen_for_highlight is not None:
            self.setPen(QPen(self._original_pen_for_highlight))
            self.update()

        self._original_pen_for_highlight = None

class _InputOutputCircle(QLabel):
    """Interactive circular port for workflow card input/output connections.
    
    Represents an input or output port on a workflow card. Users can click on
    a circle to start/complete a connection. The circle displays the port index
    and is colored based on the data type it accepts or provides.
    
    Circle border colors:
        - Blue: Image type
        - Red: Segmentation type
        - Green: Metrics type
        - Black: Untyped or multiple types
    
    Args:
        index (int): The 0-based port index.
        is_output (bool): True if this is an output port, False for input.
        card (_WorkflowCardZoneWidget): The parent card widget.
        zone (WorkflowZone, optional): Reference to the parent workflow zone.
        type_info: Type constraint(s) - data class instance or list of instances.
    
    Attributes:
        index (int): The port index.
        is_output (bool): Whether this is an output port.
        card: The parent card.
        zone: The parent workflow zone.
        type_info: The type constraint for this port.
        is_alternating (bool): True if port accepts multiple types.
    """
    def __init__(self, index: int, is_output: bool, card, zone=None, type_info=None):
        super().__init__(str(index + 1))
        self.index = index
        self.is_output = is_output
        self.card = card
        self.workflowZone = zone
        self.type_info = type_info
        self.is_alternating = isinstance(type_info, (list, tuple)) and len(type_info) > 1
        
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "QLabel { min-width: 24px; min-height: 24px; "
            "padding: 2px; font-weight: bold; background-color: transparent; }"
        )
        self.setCursor(Qt.PointingHandCursor)
    
    def _getColorForType(self, type_value):
        """Get the border color for a data type.
        
        Args:
            type_value (str, data class, or None): The data type.
        
        Returns:
            QColor: The color to use for the circle border.
        """
        if type_value is None:
            return QColor("black")

        # # Support legacy/serialized payloads that may appear during load.
        # if isinstance(type_value, str):
        #     normalized = type_value.strip().lower()
        #     if normalized in ('img', 'image'):
        #         return QColor('blue')
        #     if normalized in ('segm', 'segmentation', 'mask'):
        #         return QColor('red')
        #     if normalized in ('metrics', 'metric'):
        #         return QColor('#2e8b57')

        # if isinstance(type_value, dict):
        #     normalized = str(
        #         type_value.get('type')
        #         or type_value.get('type_name')
        #         or type_value.get('kind')
        #         or ''
        #     ).strip().lower()
        #     if normalized in ('img', 'image'):
        #         return QColor('blue')
        #     if normalized in ('segm', 'segmentation', 'mask'):
        #         return QColor('red')
        #     if normalized in ('metrics', 'metric'):
        #         return QColor('#2e8b57')

        color = getattr(type_value, 'color', None)
        if color:
            return QColor(color)

        type_name = workflow_type_name(type_value)
        if type_name == 'img':
            return QColor('blue')
        if type_name == 'segm':
            return QColor('red')
        if type_name == 'metrics':
            return QColor('#2e8b57')

        return QColor("black")
    
    def paintEvent(self, event):
        """Paint the circle with custom styled border.
        
        Draws either a solid or alternating dashed border depending on whether
        the port accepts a single type or multiple types.
        
        Args:
            event: The paint event.
        """
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
                
                # # Calculate start and end points
                # x1 = center.x() + radius * math.cos(rad1)
                # y1 = center.y() + radius * math.sin(rad1)
                # x2 = center.x() + radius * math.cos(rad2)
                # y2 = center.y() + radius * math.sin(rad2)
                
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
        """Update the type constraint and refresh the visual appearance.
        
        Args:
            type_info: New type constraint (string or list).
        """
        if isinstance(type_info, (list, tuple)) and len(type_info) == 1:
            type_info = type_info[0]
        self.type_info = type_info
        self.is_alternating = isinstance(type_info, (list, tuple)) and len(type_info) > 1
        self.update()
    
    def mousePressEvent(self, event):
        """Handle mouse press to create/complete connections or cancel.
        
        - Left-click: Start or complete a connection.
        - Right-click: Cancel the current connection being drawn.
        
        Args:
            event: The mouse event.
        """
        if self.workflowZone is None:
            super().mousePressEvent(event)
            return
            
        if event.button() == Qt.LeftButton:
            # Get global position and convert to scene coordinates
            global_pos = self.mapToGlobal(self.rect().center())
            widget_pos = self.workflowZone.mapFromGlobal(global_pos)
            scene_pos = self.workflowZone.mapToScene(widget_pos)
            
            # Start connection or complete it
            self.workflowZone.handleCircleClick(self.card, self.is_output, self.index, scene_pos)
            event.accept()
        elif event.button() == Qt.RightButton:
            self.workflowZone.cancelConnection()
            event.accept()
        else:
            super().mousePressEvent(event)

class _WorkflowCardZoneWidget(QWidget):

    """A workflow card widget that appears in the workflow zone.
    
    Represents a single workflow operation in the zone. Displays the operation's
    preview image, input/output ports, and associated dialog. Cards can be:
    - Moved by left-dragging
    - Configured by right-clicking
    - Deleted by middle-clicking
    
    Ports are represented as colored circles (blue=image, red=segmentation, green=metrics).
    Connections between ports propagate data types through the workflow.
    
    Args:
        functions (WorkflowBaseFunctions): The workflow operation functions.
        posData: Position/experiment data for the workflow.
        zone (WorkflowZone, optional): Parent workflow zone.
        workflowGui (WorkflowGui, optional): Reference to the main GUI.
    
    Attributes:
        functions: The workflow operation functions.
        posData: Position data.
        dialog: The operation's configuration dialog.
        input_circles (list): List of input port circles.
        output_circles (list): List of output port circles.
        dialog.curr_input_types_accepted (dict): Maps input index to accepted types.
        dialog.curr_input_types (dict): Maps input index to currently connected type.
        output_types (dict): Maps output index to type.
        card_id (int): Unique ID assigned by the zone.
        graphics_proxy: Graphics proxy for positioning in the scene.
    """
    def __init__(self, functions: WorkflowBaseFunctions, posData, 
                 zone=None, workflowGui=None, logger=print,
                 show_initial_dialog=True, preInitWorkflow_res=None):
        QWidget.__init__(self, parent=None)

        if workflowGui is None:
            raise ValueError('_WorkflowCardZoneWidget requires a valid workflowGui')
        if zone is None:
            raise ValueError('_WorkflowCardZoneWidget requires a valid workflowZone')
        
        self.posData = posData
        self.workflowGui = workflowGui
        self.functions = functions
        self.logger = logger
        self.title = functions.title
        self._drag_start_pos = None
        self.graphics_proxy = None
        self.workflowZone = zone
        self.is_dragging = False
        self.input_circles = []
        self.output_circles = []
        self._dialog_content_snapshot = None
        self._dialog_workflow_snapshot_before_open = None
        self._creation_aborted = False
        # When restoring state the card is already configured — no initial dialog.
        self._is_first_dialog_interaction_pending = show_initial_dialog
        self._unsaved_work_before_first_dialog = bool(self.workflowGui.unsaved_work)
        
        self.functions.posData = posData
        self.functions.setAcceptedInputs = self.setAcceptedInputs
        self.functions.setOutputs = self.setOutputs
        self.functions.getCurrentOutputs = lambda: copy.deepcopy(getattr(self, 'output_types', {}))
        self.functions.updatePreview = self.updatePreview
        self.functions.updateTitle = self.updateTitle
        self.functions.notifySelectionInvalid = self._onDialogSelectionInvalid
        self.functions.notifySelectionValid = self._onDialogSelectionValid
        self.functions._onDialogCancelled = self._onDialogCancelled
        preInitWorkflowDialog = getattr(self.functions, 'preInitWorkflowDialog', None) # optional, for example getting model choices before dialog setup

        # Run pre-init only for user-driven card creation. Restores pass the
        # previously captured preInitWorkflow_res directly.
        if (
            preInitWorkflow_res is None
            and show_initial_dialog
            and callable(preInitWorkflowDialog)
        ):
            preInitWorkflow_res = preInitWorkflowDialog(workflowGui=workflowGui)
            if preInitWorkflow_res is None:
                # Card is not in the scene yet; mark creation as aborted so callers
                # can skip adding this partially initialized widget.
                self._creation_aborted = True
                self.deleteLater()
                return
        self.preInitWorkflow_res = preInitWorkflow_res

        # Needed for stylesheet-driven background highlights and blink feedback.
        self.setAttribute(Qt.WA_StyledBackground, True)
        
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)
        
        self.inputs_container = QWidget()
        self.inputs_layout = QHBoxLayout(self.inputs_container)
        self.inputs_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(self.inputs_container)
        
        title_layout = QHBoxLayout()
        self.idLabel = QLabel()
        self.idLabel.setAlignment(Qt.AlignCenter)
        self.idLabel.setAttribute(Qt.WA_TransparentForMouseEvents)
        title_layout.addWidget(self.idLabel)
        
        title_layout.addStretch()
        
        self.titleLabel = QLabel(functions.title)
        self.titleLabel.setAlignment(Qt.AlignCenter)
        self.titleLabel.setAttribute(Qt.WA_TransparentForMouseEvents)
        title_layout.addWidget(self.titleLabel)

        title_layout.addStretch()

        _btn_style = (
            "QPushButton { "
            "min-width: 18px; max-width: 18px; "
            "min-height: 18px; max-height: 18px; "
            "padding: 0px; border-radius: 9px; "
            "} "
        )
        # set cellacdc/resources/icons/cog.svg as icon
        self.settingsButton = QPushButton(parent=self)
        self.settingsButton.setIcon(QIcon(':cog.svg'))
        self.settingsButton.setToolTip('Open settings')
        self.settingsButton.setStyleSheet(_btn_style)
        self.settingsButton.clicked.connect(self._openDialog)
        title_layout.addWidget(self.settingsButton)

        self.deleteButton = QPushButton(parent=self)
        self.deleteButton.setIcon(QIcon(':file-exit.svg'))
        self.deleteButton.setToolTip('Delete card')
        self.deleteButton.setStyleSheet(_btn_style)
        self.deleteButton.clicked.connect(self._deleteCard)
        title_layout.addWidget(self.deleteButton)

        main_layout.addLayout(title_layout)
        
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
                                                    posData=self.posData,
                                                    logger=self.logger,
                                                    preInitWorkflow_res=preInitWorkflow_res)
        self.functions.initializeDialog_cb(self.dialog,
                                           parent=self,
                                           workflowGui=workflowGui,
                                           posData=self.posData,
                                           preInitWorkflow_res=preInitWorkflow_res,
                                           logger=self.logger)

        self.dialog.sigCancelClicked.connect(self._onDialogCancelled)
        self.dialog.sigOkClicked.connect(self._onDialogAccepted)
        # open the dialog immediately for user-driven card creation, but not when restoring
        if show_initial_dialog:
            self._captureDialogContentSnapshot()
            self._openDialog()

    def _captureDialogContentSnapshot(self):
        """Capture dialog content before opening, if supported."""
        self._dialog_workflow_snapshot_before_open = self.workflowGui._captureWorkflowState()

        content = self.dialog.getContent()
        self._dialog_content_snapshot = copy.deepcopy(content)

    def _restoreDialogContentSnapshot(self):
        """Restore dialog content captured before opening, if available."""
        if self._dialog_content_snapshot is None:
            return

        self.dialog.setContent(copy.deepcopy(self._dialog_content_snapshot))

    def _onDialogCancelled(self):
        """Revert unsaved dialog edits when cancel is clicked."""
        if self._is_first_dialog_interaction_pending:
            self._is_first_dialog_interaction_pending = False
            self._dialog_workflow_snapshot_before_open = None
            prev_restoring = getattr(self.workflowGui, '_history_restoring', False)
            self.workflowGui._history_restoring = True
            try:
                self.workflowZone.removeCard(self)
            finally:
                self.workflowGui._history_restoring = prev_restoring
            self.workflowGui.setUnsavedWork(self._unsaved_work_before_first_dialog)
            self.workflowGui._refreshWorkflowValidationLabel()
            return

        self._restoreDialogContentSnapshot()
        self._dialog_workflow_snapshot_before_open = None
        self.updatePreview()

    def _onDialogAccepted(self):
        """Mark workflow as modified only when dialog changes are accepted."""
        self._is_first_dialog_interaction_pending = False
        self.workflowGui._registerWorkflowChange(self._dialog_workflow_snapshot_before_open)
        self.workflowGui.setUnsavedWork(True)
        self._dialog_workflow_snapshot_before_open = None

    def _getDialogInputTypes(self):
        """Return dialog concrete input types."""
        curr_input_types = getattr(self.dialog, 'curr_input_types', None)
        if isinstance(curr_input_types, dict):
            return curr_input_types
        return {}

    def _getDialogInputTypesAccepted(self):
        """Return dialog accepted input types."""
        # `curr_input_types_accepted` is the runtime source-of-truth used by
        # setAcceptedInputs(); fall back to legacy/static attribute if needed.
        input_types_accepted = getattr(self.dialog, 'curr_input_types_accepted', None)
        if isinstance(input_types_accepted, dict):
            return input_types_accepted

        # input_types_accepted = getattr(self.dialog, 'input_types_accepted', None)
        # if isinstance(input_types_accepted, dict):
        #     return input_types_accepted
        return {}

    def _normalizeTypeInfo(self, type_info):
        """Validate and normalize type payloads to workflow data classes."""
        if isinstance(type_info, (list, tuple)):
            normalized = []
            for value in type_info:
                if value is not None and not is_workflow_data_class(value):
                    raise TypeError(
                        f'Invalid workflow type payload: {value!r}. '
                        'Only workflow data-class instances are supported.'
                    )
                normalized.append(value)
            return normalized

        if type_info is not None and not is_workflow_data_class(type_info):
            raise TypeError(
                f'Invalid workflow type payload: {type_info!r}. '
                'Only workflow data-class instances are supported.'
            )
        return type_info

    def _firstConnectedInputType(self):
        """Return normalized type flowing into input 0 (if any)."""
        first_input = self._getDialogInputTypes().get(0)
        return self._normalizeTypeInfo(first_input)

    def _inheritDimensionsFromFirstInput(self, type_value):
        """Propagate dimension metadata from first input unless explicitly provided."""
        normalized = self._normalizeTypeInfo(type_value)
        first_input_type = self._firstConnectedInputType()
        if normalized is None or first_input_type is None:
            return normalized

        required_attrs = ('SizeZ', 'SizeT', 'SizeY', 'SizeX')
        if any(not hasattr(normalized, attr) for attr in required_attrs):
            return normalized
        if any(not hasattr(first_input_type, attr) for attr in required_attrs):
            return normalized

        inherited_size_z = normalized.SizeZ if normalized.SizeZ is not None else first_input_type.SizeZ
        inherited_size_t = normalized.SizeT if normalized.SizeT is not None else first_input_type.SizeT
        inherited_size_y = normalized.SizeY if normalized.SizeY is not None else first_input_type.SizeY
        inherited_size_x = normalized.SizeX if normalized.SizeX is not None else first_input_type.SizeX
        type_name = workflow_type_name(normalized)
        return make_workflow_data_class(
            type_name,
            SizeZ=inherited_size_z,
            SizeT=inherited_size_t,
            SizeY=inherited_size_y,
            SizeX=inherited_size_x,
        )

    def _verifyAcceptedInputTypes(self, input_types_accepted):
        """Verify that accepted input constraints are valid and dimension-compatible."""
        concrete_input_types = self._getDialogInputTypes()
        for input_idx, accepted in input_types_accepted.items():
            accepted_values = accepted if isinstance(accepted, (list, tuple)) else [accepted]
            accepted_values = [self._normalizeTypeInfo(v) for v in accepted_values]

            concrete = self._normalizeTypeInfo(concrete_input_types.get(input_idx))
            if concrete is None:
                continue

            if not any(self.workflowZone._areTypesCompatible(concrete, a) for a in accepted_values):
                self._logInfo(
                    f"Input {input_idx + 1} type verification failed: "
                    f"connected type '{concrete}' does not match accepted types {accepted_values}."
                )
        
    def updateTitle(self, new_title):
        """Update the card's display title.
        
        Args:
            new_title (str): The new title to display.
        """
        self.title = new_title
        self.titleLabel.setText(new_title)

    def _onDialogSelectionInvalid(self, value):
        """Forward a dialog-level invalid-selection notification to WorkflowGui."""
        self.workflowGui._onCardSelectionInvalid(self, value)

    def _onDialogSelectionValid(self):
        """Forward a dialog-level valid-selection notification to WorkflowGui."""
        self.workflowGui._onCardSelectionValid(self)
    
    def paintEvent(self, event):
        """Paint the card with a rounded border.
        
        Args:
            event: The paint event.
        """
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        
        # Draw rounded rectangle border
        rect = self.rect().adjusted(0, 0, -1, -1)
        pen = QPen(QColor("#555"))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawRoundedRect(rect, 8, 8)
        
        painter.end()
        super().paintEvent(event)

    def setZoneOnCircles(self):
        """Assign the zone reference to all input/output circles.
        
        Called after the card is added to the zone to enable circles
        to interact with the zone for connection creation.
        """
        for circle in self.input_circles + self.output_circles:
            circle.workflowZone = self.workflowZone

    def setAcceptedInputs(self, n=None):
        """Set the number and types of input ports accepted by the card.
        
        Can be called with:
        - A dict: {index: type} to set both count and accepted types
        - An int: Just sets count (accepted types will be None)
        - None: Clears all inputs
        
        Automatically removes connections for inputs that no longer exist and
        rebuilds visual lines.
        
        Args:
            n: Input count/types specification.
        """
            # Extract count and types from dict
        if isinstance(n, dict):
            type_info = n
            n = max(type_info.keys()) + 1 if type_info else 0
            input_types_accepted = {
                idx: self._normalizeTypeInfo(type_info.get(idx)) for idx in range(n)
            }
        elif n is None:
            # None means no inputs
            n = 0
            input_types_accepted = {}
        else:
            # Simple number, clear accepted types
            input_types_accepted = {i: None for i in range(n)}

        self._verifyAcceptedInputTypes(input_types_accepted)
        self.dialog.curr_input_types_accepted = input_types_accepted
        if not hasattr(self.dialog, 'curr_input_types'):
            self.dialog.curr_input_types = {i: None for i in range(n)}

        while self.inputs_layout.count():
            child = self.inputs_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.input_circles.clear()
        self.inputs_layout.addStretch()
        for i in range(n):
            type_value = input_types_accepted.get(i)
            if isinstance(type_value, (list, tuple)) and len(type_value) == 1:
                type_value = type_value[0]
            circle = _InputOutputCircle(i, False, self, self.workflowZone, type_info=type_value)
            self.input_circles.append(circle)
            self.inputs_layout.addWidget(circle)
            self.inputs_layout.addStretch()


        # Validate and remove invalid connections for inputs that no longer exist
        card_id = getattr(self, 'card_id', None)
        if card_id is not None:
            # Remove connections to inputs that no longer exist.
            lines_to_remove = [line for line in self.workflowZone.lines
                               if line[1][0] == card_id and line[1][1] >= n]
            self.workflowZone.removeLineKeys(lines_to_remove)
            self.workflowZone.syncCardInputTypes(card_id)
            # Defer rebuild until layout has been processed
            QTimer.singleShot(0, self.workflowZone.rebuildConnectionLines)
            return

        # # Keep dialog input state in sync even before the card is attached to a zone.
        # self.functions.updateInputTypes(self.dialog, {i: None for i in range(n)})
   
    def setOutputs(self, n=None):
        """Set the number and types of output ports.
        
        Can be called with:
        - A dict: {index: type} to set both count and types
        - An int: Just sets count (types will be None)
        - None: Clears all outputs
        
        Automatically removes connections for outputs that no longer exist and
        rebuilds visual lines.
        Updates downstream cards about type changes.
        
        Args:
            n: Output count/types specification.
        """
        previous_output_types = getattr(self, 'output_types', None)

        if isinstance(n, dict):
            # Extract count and types from dict
            type_info = n
            n = max(type_info.keys()) + 1 if type_info else 0
            output_types = {
                idx: self._inheritDimensionsFromFirstInput(self._normalizeTypeInfo(type_info.get(idx)))
                for idx in range(n)
            }
        elif n is None:
            # None means no outputs
            n = 0
            output_types = {}
        else:
            # Simple number, clear types
            raise ValueError("Output specification must be a dict mapping index to type, or None to clear outputs.")

        outputs_changed = previous_output_types != output_types
        self.output_types = output_types
        
        while self.outputs_layout.count():
            child = self.outputs_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.output_circles.clear()
        self.outputs_layout.addStretch()
        for i in range(n):
            type_value = self.output_types.get(i)
            if isinstance(type_value, (list, tuple)) and len(type_value) == 1:
                type_value = type_value[0]
            circle = _InputOutputCircle(i, True, self, self.workflowZone, type_info=type_value)
            self.output_circles.append(circle)
            self.outputs_layout.addWidget(circle)
            self.outputs_layout.addStretch()
        
        # Validate and remove invalid connections for outputs that no longer exist
        card_id = getattr(self, 'card_id', None)
        if card_id is not None:
            # Remove connections from outputs that no longer exist
            lines_to_remove = [line for line in self.workflowZone.lines
                               if line[0][0] == card_id and line[0][1] >= n]
            self.workflowZone.removeLineKeys(lines_to_remove)
            if outputs_changed:
                # Let direct downstream dialogs react first; any further
                # propagation happens when they emit their own output changes.
                self.workflowZone.updateDownstreamInputTypes(card_id)
            # Defer rebuild until layout has been processed
            QTimer.singleShot(0, self.workflowZone.rebuildConnectionLines)
                
    def updateInputTypes(self, input_types):
        """For updating what input a card actually gets (not what it accepts)!

        Args:
            input_types (dict): Maps input index to the type of data currently connected.
        """
        self.functions.updateInputTypes(self.dialog, input_types)

    def _logInfo(self, message):
        """Log message supporting both Logger objects and callables."""
        try:
            if hasattr(self.logger, 'info'):
                self.logger.info(message)
            elif callable(self.logger):
                self.logger(message)
        except Exception:
            pass

    def mousePressEvent(self, event):
        """Handle mouse press events for card interaction.
        
        - Left button: Start dragging the card
        - Middle button: Delete the card
        - Right button: Open context menu
        
        Args:
            event: The mouse event.
        """
        if event.button() == Qt.LeftButton:
            self._drag_start_pos = event.globalPos()
            self.is_dragging = True
            event.accept()
        elif event.button() == Qt.MiddleButton:
            self._deleteCard()
            event.accept()
        elif event.button() == Qt.RightButton:
            self._showContextMenu(event.globalPos())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Open card settings on left-button double-click."""
        if event.button() == Qt.LeftButton:
            # Stop an active drag start triggered by the first click.
            self.is_dragging = False
            self._drag_start_pos = None
            self._openDialog()
            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def _showContextMenu(self, global_pos):
        """Show card context menu with common card/connection actions."""
        menu = QMenu(self)
        open_action = menu.addAction('Open dialog')
        delete_action = menu.addAction('Delete card')
        menu.addSeparator()
        remove_incoming_action = menu.addAction('Delete all incoming connections')
        remove_outgoing_action = menu.addAction('Delete all outgoing connections')

        selected_action = menu.exec_(global_pos)
        if selected_action is None:
            return

        if selected_action == open_action:
            self._openDialog()
        elif selected_action == delete_action:
            self._deleteCard()
        elif selected_action == remove_incoming_action:
            self.workflowZone.removeCardConnections(self.card_id, remove_incoming=True, remove_outgoing=False)
        elif selected_action == remove_outgoing_action:
            self.workflowZone.removeCardConnections(self.card_id, remove_incoming=False, remove_outgoing=True)

    def mouseMoveEvent(self, event):
        """Update card position and connection lines while dragging.
        
        Args:
            event: The mouse event.
        """
        if self.is_dragging and self._drag_start_pos is not None and self.graphics_proxy is not None:
            delta = event.globalPos() - self._drag_start_pos
            self.graphics_proxy.moveBy(delta.x(), delta.y())
            self._drag_start_pos = event.globalPos()
            # Update connection lines in real-time while dragging
            self.workflowZone.updateConnectionLines()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Finalize card position and update connection lines after dragging.
        
        Args:
            event: The mouse event.
        """
        if event.button() == Qt.LeftButton:
            self.is_dragging = False
            self._drag_start_pos = None
            # Update all connection lines after card is repositioned
            self.workflowZone.updateConnectionLines()
        super().mouseReleaseEvent(event)

    def _openDialog(self):
        """Open the configuration dialog for this card."""
        try:
            self._captureDialogContentSnapshot()
            self.functions.runDialog_cb(self.dialog)
        except Exception as e:
            self._logInfo(f"Error running dialog: {e}")

    def _deleteCard(self):
        """Delete this card from the workflow zone."""
        self.workflowGui.setUnsavedWork(True)
        self.workflowZone.removeCard(self)
        # clean up dialog and widget
        if hasattr(self, 'dialog') and self.dialog is not None:
            self.dialog.deleteLater()
            self.dialog = None

    def updatePreview(self):
        """Update the preview image by capturing the current dialog state."""
        if not hasattr(self, 'dialog') or self.dialog is None:
            return
        preview = self.functions.getDialogPreview(self.dialog)
        if preview is not None:
            self.thumbnail.setPixmap(preview)
            

class WorkflowSidebar(QListWidget):
    """Sidebar list widget showing available workflow cards for drag-and-drop.
    
    Displays workflow card options (e.g., "Combine channels", "Select image")
    that users can drag onto the workflow zone. Shows a preview thumbnail and
    title for each card type.
    
    Features:
    - Custom drag-and-drop with preview pixmap
    - Caches pixmaps for better performance
    - Emits sigItemDropped signal when a card is dropped in the zone
    """
    
    def startDrag(self, supportedActions):
        """Initiate drag operation with custom preview pixmap.
        
        Overrides default drag behavior to use the cached card preview
        as the drag image instead of a generic widget icon.
        
        Args:
            supportedActions: Supported drop actions.
        """
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
    """The main canvas where workflow cards are arranged and connected.
    
    A graphics view that displays workflow cards and the connections between them.
    Users can:
    - Drop cards from the sidebar
    - Drag cards to reposition them
    - Click on input/output circles to create connections
    - Right-click on lines to delete connections
    - Right-click on cards to configure them
    - Middle-click on cards to delete them
    
    The zone maintains:
    - A list of cards and their unique IDs
    - A list of connections (port-to-port links)
    - Type compatibility checking
    - Visual connection lines
    
    Signals:
        sigItemDropped(str, int, int): Emitted when a card is dropped.
            Args: card_name, x_position, y_position
    
    Attributes:
        cards (dict): Maps card_id to graphics proxy widget.
        lines (list): List of connections as ((from_id, from_port), (to_id, to_port)).
        connection_lines (list): Visual QGraphicsLineItem objects.
        unique_card_id (int): Counter for assigning unique IDs to cards.
        connection_start (tuple): State of connection being drawn.
        temp_line: Visual line being drawn while creating a connection.
    """
    sigItemDropped = Signal(str, int, int)

    def __init__(self, parent=None, logger=print):
        """Initialize the workflow zone.
        
        Args:
            parent: Parent widget.
        """
        self.logger = logger
        self.parent = parent
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
        
        self.placeholder = QLabel(
            'Click <img src=":file-new.svg" width="16" height="16"> '
            'in the top toolbar to create new workflow, or click '
            '<img src=":folder-open.svg" width="16" height="16"> to load.'
            'Click <img src=":info.svg" width="16" height="16"> for help.'
        )
        self.placeholder.setTextFormat(Qt.RichText)
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder_proxy = self.scene.addWidget(self.placeholder)
        self.placeholder_proxy.setPos(0, 0)
        self.line_to_visual_line_mapping = {}
        self.visual_line_to_line_mapping = {}

    def _logInfo(self, message):
        """Log an info message for workflow-zone actions."""
        try:
            if hasattr(self.logger, 'info'):
                self.logger.info(message)
            elif callable(self.logger):
                self.logger(message)
        except Exception:
            pass

    def _markWorkflowChanged(self, previous_state=None):
        """Notify parent workflow GUI about user-driven workflow mutations."""
        if getattr(self.parent, '_history_restoring', False):
            return
        self.parent.setUnsavedWork(True)
        self.parent._refreshWorkflowValidationLabel()
        if previous_state is not None and hasattr(self.parent, '_registerWorkflowChange'):
            self.parent._registerWorkflowChange(previous_state)

    def addCard(self, card_widget, x, y):
        """Add a workflow card to the zone at the specified position.
        
        Args:
            card_widget (_WorkflowCardZoneWidget): The card to add.
            x (int): X coordinate in the scene.
            y (int): Y coordinate in the scene.
        """
        proxy = self.scene.addWidget(card_widget)
        proxy.setPos(QPointF(x, y))
        proxy.setCacheMode(proxy.CacheMode.DeviceCoordinateCache)
        card_widget.graphics_proxy = proxy
        card_widget.workflowZone = self
        card_widget.card_id = self.unique_card_id  # Store the ID on the card widget
        card_widget.idLabel.setText(f"ID: {self.unique_card_id}")  # Display the ID in the label
        card_widget.setZoneOnCircles()
        self.cards[self.unique_card_id] = proxy
        card_id = self.unique_card_id
        self.unique_card_id += 1
        if len(self.cards) == 1:
            self.placeholder_proxy.hide()

        title = getattr(card_widget, 'title', '')
        self._logInfo(f"Added card ID={card_id} title='{title}' at ({x}, {y}).")

    def _computeCardInputTypes(self, card):
        """Compute concrete input types for a card from current connections."""
        card_id = getattr(card, 'card_id', None)
        input_types = {i: None for i in range(len(card.input_circles))}

        if card_id is None:
            return input_types

        for (start_card_id, start_idx), (end_card_id, end_idx) in self.lines:
            if end_card_id != card_id or end_idx not in input_types:
                continue

            start_proxy = self.cards.get(start_card_id)
            if start_proxy is None:
                continue

            start_card = start_proxy.widget()
            if start_card is None:
                continue

            input_types[end_idx] = start_card.output_types.get(start_idx)

        return input_types

    def syncCardInputTypes(self, card_id):
        """Sync a card's concrete input types from its incoming connections."""
        proxy = self.cards.get(card_id)
        if proxy is None:
            return

        card = proxy.widget()
        if card is None:
            return

        input_types = self._computeCardInputTypes(card)

        card.functions.updateInputTypes(card.dialog, input_types)

    def removeLineKeys(self, line_keys):
        """Remove logical and visual connections and resync affected inputs."""
        affected_input_card_ids = set()
        removed_line_keys = []

        for line_key in line_keys:
            if line_key in self.lines:
                self.lines.remove(line_key)
                removed_line_keys.append(line_key)
            self._removeVisualLineByKey(line_key)
            affected_input_card_ids.add(line_key[1][0])

        for line_key in removed_line_keys:
            (start_card_id, start_port), (end_card_id, end_port) = line_key
            self._logInfo(
                f"Removed connection: card {start_card_id} output {start_port} -> "
                f"card {end_card_id} input {end_port}."
            )

        for card_id in affected_input_card_ids:
            self.syncCardInputTypes(card_id)

    def removeCardConnections(self, card_id, remove_incoming=True, remove_outgoing=True):
        """Remove incoming/outgoing connections for a card and track history."""
        if card_id is None:
            return

        lines_to_remove = []
        for line_key in self.lines:
            (start_card_id, _start_port), (end_card_id, _end_port) = line_key
            if remove_outgoing and start_card_id == card_id:
                lines_to_remove.append(line_key)
                continue
            if remove_incoming and end_card_id == card_id:
                lines_to_remove.append(line_key)

        if not lines_to_remove:
            return

        previous_state = None
        if not getattr(self.parent, '_history_restoring', False):
            previous_state = self.parent._captureWorkflowState()

        self.removeLineKeys(lines_to_remove)
        self._markWorkflowChanged(previous_state)

    def _removeVisualLineByKey(self, line_key):
        """Remove visual line corresponding to a model line key."""
        visual_line = self.line_to_visual_line_mapping.pop(line_key, None)
        if visual_line is None:
            return
        self.visual_line_to_line_mapping.pop(visual_line, None)
        if visual_line in self.connection_lines:
            self.connection_lines.remove(visual_line)
        self.scene.removeItem(visual_line)

    def removeCard(self, card_widget):
        """Remove a workflow card and all its connections from the zone.
        
        Args:
            card_widget (_WorkflowCardZoneWidget): The card to remove.
        """
        card_id = getattr(card_widget, 'card_id', None)
        if card_id is not None and card_id in self.cards:
            proxy = self.cards[card_id]
            if proxy.widget() == card_widget:
                previous_state = None
                if not getattr(self.parent, '_history_restoring', False):
                    previous_state = self.parent._captureWorkflowState()
                self.scene.removeItem(proxy)
                del self.cards[card_id]
                # Remove all connections involving this card.
                lines_to_remove = [line_key for line_key in self.lines
                                   if line_key[0][0] == card_id or line_key[1][0] == card_id]
                self.removeLineKeys(lines_to_remove)
                card_widget.deleteLater()
                if not self.cards:
                    self.placeholder_proxy.show()

                title = getattr(card_widget, 'title', '')
                self._logInfo(f"Removed card ID={card_id} title='{title}'.")
                self._markWorkflowChanged(previous_state)

    def handleCircleClick(self, card, is_output, index, scene_pos):
        """Handle a click on an input/output circle to create/complete a connection.
        
        Handles the state machine for connection creation:
        - First click: Start a connection from this port
        - Second click: Complete connection if valid
        - Invalid connections are rejected with feedback
        
        Args:
            card (_WorkflowCardZoneWidget): The card containing the circle.
            is_output (bool): True if output port, False if input.
            index (int): The 0-based port index.
            scene_pos (QPointF): Position in the scene where the click occurred.
        """
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
                            self._logInfo(
                                f"Input {index} on this card already has a connection. "
                                "Inputs can only have one connection."
                            )
                            self.cancelConnection()
                            return

                # Valid connection
                self.createConnection(start_card, start_is_output, start_index, card, is_output, index)
            
            # Clear connection state regardless
            self.cancelConnection()

    def _areTypesCompatible(self, output_type, input_type_accepted):
        """Check if output type is compatible with accepted input type requirement.
        
        Compatibility rules:
        - If either type is None, they are compatible
        - If input accepts multiple types (list), output must match one
        - If input expects a specific type, class kind must match
        
        Args:
            output_type: The output port's workflow data class.
            input_type_accepted: Accepted workflow data class constraint(s).
        
        Returns:
            bool: True if types are compatible.
        """
        if output_type is not None and not is_workflow_data_class(output_type):
            raise TypeError(f'Unsupported output type: {output_type!r}')
        if input_type_accepted is not None and not isinstance(input_type_accepted, (list, tuple)) and not is_workflow_data_class(input_type_accepted):
            raise TypeError(f'Unsupported accepted input type: {input_type_accepted!r}')

        if output_type is None or input_type_accepted is None:
            return True
        
        if isinstance(input_type_accepted, (list, tuple)):
            # Input accepts a list of types
            for accepted_type in input_type_accepted:
                if accepted_type is not None and not is_workflow_data_class(accepted_type):
                    raise TypeError(f'Unsupported accepted input type in list: {accepted_type!r}')
            return output_type in input_type_accepted

        output_type_name = workflow_type_name(output_type)
        accepted_type_name = workflow_type_name(input_type_accepted)
        if output_type_name != accepted_type_name:
            return False

        # Accepted constraints with explicit dimensions must be respected.
        accepted_size_z = getattr(input_type_accepted, 'SizeZ', None)
        accepted_size_t = getattr(input_type_accepted, 'SizeT', None)
        accepted_size_y = getattr(input_type_accepted, 'SizeY', None)
        accepted_size_x = getattr(input_type_accepted, 'SizeX', None)
        output_size_z = getattr(output_type, 'SizeZ', None)
        output_size_t = getattr(output_type, 'SizeT', None)
        output_size_y = getattr(output_type, 'SizeY', None)
        output_size_x = getattr(output_type, 'SizeX', None)

        if accepted_size_z is not None and output_size_z is not None and accepted_size_z != output_size_z:
            return False
        if accepted_size_t is not None and output_size_t is not None and accepted_size_t != output_size_t:
            return False
        if accepted_size_y is not None and output_size_y is not None and accepted_size_y != output_size_y:
            return False
        if accepted_size_x is not None and output_size_x is not None and accepted_size_x != output_size_x:
            return False

        return True

    def createConnection(self, card1, is_out1, idx1, card2, is_out2, idx2):
        """Create a visual connection between two circles"""
        previous_state = None
        if not getattr(self.parent, '_history_restoring', False):
            previous_state = self.parent._captureWorkflowState()

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
        
        # Store connection with card IDs
        if is_out1:  # card1 is output, card2 is input
            line_key = ((card1_id, idx1), (card2_id, idx2))
        else:  # card1 is input, card2 is output - swap them
            line_key = ((card2_id, idx2), (card1_id, idx1))

        self.lines.append(line_key)
        self.syncCardInputTypes(input_card.card_id)
        self.drawConnection(card1, is_out1, idx1, card2, is_out2, idx2, line_key=line_key)
        self._markWorkflowChanged(previous_state)
        self._logInfo(
            f"Created connection: card {line_key[0][0]} output {line_key[0][1]} -> "
            f"card {line_key[1][0]} input {line_key[1][1]}."
        )

    def drawConnection(self, card1, is_out1, idx1, card2, is_out2, idx2, line_key=None):
        """Draw a connection line between two circles"""
        # Get the global positions of the circles
        proxy1 = card1.graphics_proxy
        proxy2 = card2.graphics_proxy
        
        if is_out1:
            circles1 = card1.output_circles
            start_circle = circles1[idx1] if idx1 < len(circles1) else None
        else:
            circles1 = card1.input_circles
            start_circle = circles1[idx1] if idx1 < len(circles1) else None
        
        if is_out2:
            circles2 = card2.output_circles
            end_circle = circles2[idx2] if idx2 < len(circles2) else None
        else:
            circles2 = card2.input_circles
            end_circle = circles2[idx2] if idx2 < len(circles2) else None
        
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
            if line_key is not None:
                self.line_to_visual_line_mapping[line_key] = line
                self.visual_line_to_line_mapping[line] = line_key

    def removeConnectionLine(self, line):
        """Remove a connection line when right-clicked"""
        if line not in self.connection_lines:
            return

        if hasattr(line, 'clearErrorHighlight'):
            line.clearErrorHighlight(reset_pen=False)

        previous_state = None
        if not getattr(self.parent, '_history_restoring', False):
            previous_state = self.parent._captureWorkflowState()

        line_key_to_remove = self.visual_line_to_line_mapping.get(line)

        if line_key_to_remove is not None:
            self.removeLineKeys([line_key_to_remove])
            self._markWorkflowChanged(previous_state)
            return

        # Fallback if mapping is out-of-sync.
        self.visual_line_to_line_mapping.pop(line, None)
        self.scene.removeItem(line)
        self.connection_lines.remove(line)
        self._markWorkflowChanged(previous_state)

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
            if hasattr(line, 'clearErrorHighlight'):
                line.clearErrorHighlight(reset_pen=False)
            self.scene.removeItem(line)
        self.connection_lines.clear()
        self.line_to_visual_line_mapping.clear()
        self.visual_line_to_line_mapping.clear()
        
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
            start_circle = start_card.output_circles[start_idx] if start_idx < len(start_card.output_circles) else None
            end_circle = end_card.input_circles[end_idx] if end_idx < len(end_card.input_circles) else None
            
            if start_circle is None or end_circle is None:
                continue
            
            self.drawConnection(start_card, True, start_idx, end_card, False, end_idx, line_key=line_data)

    def validateConnectionTypes(self):
        """Return type-incompatible connections without mutating the graph."""
        mismatches = []
        for line_data in self.lines:
            (start_card_id, start_idx), (end_card_id, end_idx) = line_data

            start_proxy = self.cards.get(start_card_id)
            end_proxy = self.cards.get(end_card_id)
            if start_proxy is None or end_proxy is None:
                continue

            start_card = start_proxy.widget()
            end_card = end_proxy.widget()
            if start_card is None or end_card is None:
                continue

            output_type = start_card.output_types.get(start_idx)
            input_type_accepted = end_card._getDialogInputTypesAccepted().get(end_idx)
            if not self._areTypesCompatible(output_type, input_type_accepted):
                mismatches.append((line_data, output_type, input_type_accepted))

        return mismatches

    def clearConnectionErrorHighlights(self):
        """Stop error animations on all visual connection lines."""
        for line in self.connection_lines:
            if hasattr(line, 'clearErrorHighlight'):
                line.clearErrorHighlight(reset_pen=True)

    def highlightConnectionKeys(self, line_keys, duration_ms=2200):
        """Blink visual lines corresponding to model connection keys."""
        for line_key in line_keys:
            visual_line = self.line_to_visual_line_mapping.get(line_key)
            if visual_line is None:
                continue
            if hasattr(visual_line, 'highlightError'):
                visual_line.highlightError(duration_ms=duration_ms)

    def updateDownstreamInputTypes(self, source_card_id):
        """Sync the concrete input types of directly connected downstream cards.

        Further propagation is intentionally dialog-driven: each downstream card
        reacts in updatedInputTypes(), may emit output-type changes, and its own
        setOutputs() call will then trigger the next downstream sync.
        """
        downstream_card_ids = set()

        # Find all direct downstream cards connected to this source card.
        for line_data in self.lines:
            (start_card_id, _start_idx), (end_card_id, _end_idx) = line_data
            if start_card_id != source_card_id:
                continue

            downstream_card_ids.add(end_card_id)

        # Only sync direct children here. If a child changes its outputs in
        # response, its own setOutputs() call will continue propagation.
        for downstream_card_id in downstream_card_ids:
            self.syncCardInputTypes(downstream_card_id)

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
    """Main window for the workflow editor GUI.
    
    Provides a complete workflow editing interface with:
    - Sidebar showing available workflow cards
    - Workflow zone for arranging and connecting cards
    - Automatic data type propagation and compatibility checking
    - Dialog-based configuration for each workflow step
    
    The workflow is built by dragging cards from the sidebar onto the zone,
    connecting them by clicking ports, and configuring each card.
    
    Args:
        app: The QApplication instance.
        parent: Parent widget (optional).
        buttonToRestore: Button to re-enable when GUI closes (optional).
        mainWin: Reference to the main Cell-ACDC window (optional).
        version (str): GUI version string (optional).
        launcherSlot (callable): Callback to invoke when GUI closes (optional).
        selectedExpPaths (dict): {path: [positions]} data directories to process (optional).
    
    Attributes:
        data (dict): Loaded position data indexed by position number.
        posData: The first position's data (used for UI defaults).
        img_channels (set): Available image channel names.
        segm_channels (set): Available segmentation data names.
        sidebar (WorkflowSidebar): The available cards panel.
        dropZone (WorkflowZone): The workflow editing area.
        debug (bool): Debug mode flag from configuration.
    
    Signals:
        sigClosed(object): Emitted when the GUI window is closed.
    """
    sigClosed = Signal(object)
        
    def __init__(
            self, app, parent=None, buttonToRestore=None,
            mainWin=None, version=None, launcherSlot=None,
        ):
        """Initialize the workflow GUI.
        
        Loads data from selected positions, discovers available image and
        segmentation channels, and builds the UI.
        
        Args:
            app: The QApplication instance.
            parent: Parent widget (optional).
            buttonToRestore: Button to re-enable when closing (optional).
            mainWin: Main window reference (optional).
            version: Version string (optional).
            launcherSlot: Callback for GUI close (optional).
            selectedExpPaths: Dict mapping paths to position lists (optional).
        """

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
        self.selectedExpPaths = None 
        self.loadedWorkflowPath = None
        self.logDisplay = None
        self._guiLogHandler = None
        self.unsaved_work = False
        self._suspend_unsaved_tracking = False
        self._isAskingSaveBeforeClosing = False
        self._history_restoring = False
        self._undo_stack = []
        self._redo_stack = []
        self._history_max_steps = 100
        self._io_actions_lock_depth = 0
        self.data_loaded = False

        self.setAcceptDrops(True)
        self._appName = 'Cell-ACDC'
                

        
    def initData(self):
        """Load and analyze data from selected experiments.
        
        Loads position data and discovers available image and segmentation
        channels, keeping only channels common to all selected positions.
        Stores the first position's data in self.posData for UI defaults.
        """
        self.data = {}
        self.posData = None # store first data loaded, as this is relevant for UI.
        # all other data is only relevant for running the workflow
        self.img_channels = set()
        self.segm_channels = set()
        
        i = 0
        for path, positions in self.selectedExpPaths.items():
            for pos in positions:
                path_loc = os.path.join(path, pos, 'Images')
                print("Loading data from", path_loc)

                posData = load.loadData(path_loc)
                posData.total_path = path_loc
                self.data[i] = posData
                
            if i == 0:
                self.posData = posData
                self.posData.loadOtherFiles(load_metadata=True,
                                            load_customCombineMetrics=True,
                                            load_customAnnot=True,
                                            loadSegmInfo=True,
                                            loadBkgrROIs=True,
                                            
                                            )

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

    def _setupDragCard(self, functions: WorkflowBaseFunctions, build_widget=True):
        """Add a workflow card to the sidebar.
        
        Creates a list item and optionally attaches the full preview widget.
        When data is not loaded yet, we keep the sidebar lightweight by
        skipping full widget initialization.
        
        Args:
            functions (WorkflowBaseFunctions): The workflow operation to add.
            build_widget (bool): If True, create full list widget with preview.
        """
        channels_card = QListWidgetItem()
        channels_card.setData(Qt.UserRole, functions)
        self.sidebar.addItem(channels_card)
        
        if not build_widget:
            channels_card.setText(getattr(functions, 'title', 'Workflow Card'))
            return

        # Custom item widget handles title/preview painting.
        channels_card.setText('')
        card_widget = _WorkflowCardListWidget(functions, parent=self, posData=self.posData)
        channels_card.setSizeHint(card_widget.sizeHint())
        self.sidebar.setItemWidget(channels_card, card_widget)

    def _rebuildSidebarCards(self, build_widgets):
        """Recreate sidebar cards.

        Args:
            build_widgets (bool): If True attach full card widgets, otherwise
                create lightweight text-only items.
        """
        if not hasattr(self, 'sidebar') or self.sidebar is None:
            return

        self.sidebar.clear()
        functions_to_add = FUNCTIONS_TO_ADD
        for functions in functions_to_add:
            self._setupDragCard(functions, build_widget=build_widgets)

    def _buildWorkflowHelpText(self):
        """Build short help text for the workflow info message box."""
        lines = [
            'Quick guide',
            '',
            '- Start by creating a new workflow file or loading an existing one. (Ctrl+N/Ctrl+O)',
            '- Change loaded experiment data with the "Load Images..." action.',
            '- Drag cards from the sidebar into the workflow area.',
            '- Make connections by clicking an output circle, then an input circle.',
            '- Open or edit a card by right-clicking it in the workflow area.',
            '',
            'Cards',
        ]

        functions_to_add = FUNCTIONS_TO_ADD
        for functions in functions_to_add:
            title = getattr(functions, 'title', type(functions).__name__)

            info = getattr(functions, 'info', '')
            if callable(info):
                try:
                    info = info()
                except Exception:
                    info = ''

            if not info:
                doc = getattr(type(functions), '__doc__', '') or ''
                doc_lines = [line.strip() for line in doc.splitlines() if line.strip()]
                info = doc_lines[0] if doc_lines else ''

            if not info:
                info = 'Configure this card and connect it within the workflow.'

            lines.append(f'- {title}: {str(info).strip()}')

        return '\n'.join(lines)

    def _onWorkflowInfoTriggered(self):
        """Show short usage instructions for the workflow GUI."""
        QMessageBox.information(
            self,
            'Workflow help',
            self._buildWorkflowHelpText(),
        )

    def _onOpenLogFileLocationTriggered(self):
        """Open the folder containing workflow GUI log files."""
        logs_path = getattr(self, 'logs_path', None)
        if not logs_path or not os.path.isdir(logs_path):
            QMessageBox.information(
                self,
                'Log file location unavailable',
                'Log file location is not available yet.',
            )
            return

        myutils.showInExplorer(logs_path)

    def _onAboutCellAcdcTriggered(self):
        """Show the standard Cell-ACDC About dialog."""
        from .help import about

        self.aboutWin = about.QDialogAbout(parent=self)
        self.aboutWin.show()

    def addDroppedCard(self, card: str, x: int, y: int):
        """Handle a card dropped into the WorkflowZone at position (x, y).
        
        Finds the card definition from the sidebar, creates a zone card widget,
        and adds it at the specified position.
        
        Args:
            card (str): The name/title of the card dropped.
            x (int): X coordinate in the zone.
            y (int): Y coordinate in the zone.
        """
        previous_state = self._captureWorkflowState()

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
        
        # Create a fresh functions instance for each dropped card to avoid
        # cross-card state/signal leakage from shared function objects.
        try:
            card_functions = type(card_data)()
        except Exception:
            card_functions = card_data

        # Create a new zone card widget with the card data, passing zone reference
        card_widget = _WorkflowCardZoneWidget(card_functions, posData=self.posData, 
                                              zone=self.dropZone, workflowGui=self, 
                                              logger=self.logger)

        if getattr(card_widget, '_creation_aborted', False):
            return
        
        # Add it directly to the drop zone at the specified position
        self.dropZone.addCard(card_widget, x, y)
        self.setUnsavedWork(True)
        self._registerWorkflowChange(previous_state)
        self._refreshWorkflowValidationLabel()

    def _createToolbarIoLegend(self):
        """Create a compact legend for I/O port circle colors in the toolbar."""
        legend_widget = QWidget(self)
        legend_layout = QHBoxLayout(legend_widget)
        legend_layout.setContentsMargins(6, 0, 0, 0)
        legend_layout.setSpacing(8)

        io_label = QLabel('I/O:', legend_widget)
        legend_layout.addWidget(io_label)

        def add_legend_item(color_name, text):
            circle = QLabel(legend_widget)
            circle.setFixedSize(12, 12)
            circle.setStyleSheet(
                "QLabel {"
                "border: 2px solid %s;"
                "border-radius: 6px;"
                "background-color: transparent;"
                "}" % color_name
            )

            text_label = QLabel(text, legend_widget)
            legend_layout.addWidget(circle)
            legend_layout.addWidget(text_label)

        add_legend_item(WfSegmDC().color, 'segm.')
        add_legend_item(WfImageDC().color, 'image')
        add_legend_item(WfMetricsDC().color, 'metrics')

        return legend_widget

    def _refreshWorkflowValidationLabel(self):
        """Refresh toolbar validation label from current graph state."""
        if not hasattr(self, 'validationStatusLabel') or self.validationStatusLabel is None:
            return

        is_valid, errors = self.validateWorkflowGraph(log_errors=False)
        if is_valid:
            self.validationStatusLabel.setText('Looking good!')
            self.validationStatusLabel.setStyleSheet(
                'QPushButton { color: #1f7a1f; font-weight: bold; }'
            )
        else:
            self.validationStatusLabel.setText(f'{len(errors)} issue(s) in Workflow')
            self.validationStatusLabel.setStyleSheet(
                'QPushButton { color: #b22222; font-weight: bold; }'
            )

    def _onValidationStatusLabelClicked(self):
        """Re-run validation and show current issues when toolbar label is clicked."""
        is_valid, errors = self.validateWorkflowGraph(log_errors=True)
        self._refreshWorkflowValidationLabel()
        if not is_valid:
            _errors, issue_card_ids, issue_line_keys = self._analyzeWorkflowGraphIssues()
            self._highlightWorkflowIssues(issue_card_ids, issue_line_keys)
            self._showWorkflowValidationErrors(errors)

    def _blinkWidgetControl(self, widget, duration_ms=2000):
        """Start a blink effect on a widget and keep the blinker referenced."""
        if widget is None:
            return

        blinker = qutils.QControlBlink(widget, duration_ms=duration_ms, qparent=self)
        blinker.start()
        setattr(widget, '_workflow_issue_blinker', blinker)

    def _highlightWorkflowIssues(self, issue_card_ids, issue_line_keys):
        """Blink cards, settings buttons and invalid connections after validation."""
        if hasattr(self, 'dropZone') and self.dropZone is not None:
            self.dropZone.clearConnectionErrorHighlights()

        for card_id in sorted(issue_card_ids):
            proxy = self.dropZone.cards.get(card_id) if hasattr(self, 'dropZone') else None
            card_widget = proxy.widget() if proxy is not None else None
            if card_widget is None:
                continue

            self._blinkWidgetControl(card_widget, duration_ms=2200)
            settings_btn = getattr(card_widget, 'settingsButton', None)
            self._blinkWidgetControl(settings_btn, duration_ms=2200)

            # Bring problematic card into view to make issues easier to locate.
            proxy_item = getattr(card_widget, 'graphics_proxy', None)
            if proxy_item is not None and hasattr(self, 'dropZone'):
                self.dropZone.ensureVisible(proxy_item)

        if hasattr(self, 'dropZone') and self.dropZone is not None:
            self.dropZone.highlightConnectionKeys(issue_line_keys, duration_ms=2200)

    def setupUI(self):
        """Build the user interface.
        
        Creates:
        - Sidebar with available workflow cards
        - Main workflow zone for editing
        - Connects signals for drag-and-drop
        """
        # Create and add toolbar first, before setting up central widget layout
        folder_toolbar = widgets.ToolBar("File", parent=self)
        
        self.newAction = QAction(self)
        self.newAction.setText("&New workflow file...")
        self.newAction.setIcon(QIcon(":file-new.svg"))
        self.newAction.setShortcut('Ctrl+N')
        self.newAction.triggered.connect(self._onNewWorkflowTriggered)
        
        self.loadWorkflowAction = QAction(
            QIcon(":folder-open.svg"), "&Load Workflow...", self
        )
        self.loadWorkflowAction.setShortcut('Ctrl+O')
        self.loadWorkflowAction.triggered.connect(self._onLoadWorkflowTriggered)
        
        self.saveAction = QAction(
            QIcon(":file-save.svg"), "&Save Workflow...", self
        )
        self.saveAction.setShortcut('Ctrl+S')
        self.saveAction.triggered.connect(self._onSaveWorkflowTriggered)
        
        self.saveNewAction = QAction(
            QIcon(":file_new_save.svg"), "Save Workflow &As...", self
        )
        self.saveNewAction.setShortcut('Ctrl+Shift+S')
        self.saveNewAction.triggered.connect(self._onSaveWorkflowAsTriggered)
        
        self.openImagesAction = QAction(
            QIcon(":image.svg"), "Load Images...", self
        )
        self.openImagesAction.triggered.connect(self._onLoadImagesTriggered)

        self.infoAction = QAction(
            QIcon(":info.svg"), "Workflow &Info", self
        )
        self.infoAction.setToolTip('Show brief workflow usage help')
        self.infoAction.triggered.connect(self._onWorkflowInfoTriggered)

        self.openLogFileLocationAction = QAction(
            'Open log file location', self
        )
        self.openLogFileLocationAction.triggered.connect(
            self._onOpenLogFileLocationTriggered
        )

        self.aboutAction = QAction('About Cell-ACDC', self)
        self.aboutAction.triggered.connect(self._onAboutCellAcdcTriggered)

        self.undoAction = QAction('Undo', self)
        self.undoAction.setShortcut('Ctrl+Z')
        self.undoAction.triggered.connect(self._onUndoTriggered)

        self.redoAction = QAction('Redo', self)
        self.redoAction.setShortcut('Ctrl+Y')
        self.redoAction.triggered.connect(self._onRedoTriggered)

        menuBar = self.menuBar()
        menuBar.setNativeMenuBar(False)

        fileMenu = menuBar.addMenu('&File')
        fileMenu.addAction(self.newAction)
        fileMenu.addAction(self.loadWorkflowAction)

        helpMenu = menuBar.addMenu('&Help')
        helpMenu.addAction(self.infoAction)
        helpMenu.addAction(self.openLogFileLocationAction)
        helpMenu.addSeparator()
        helpMenu.addAction(self.aboutAction)
        
        folder_toolbar.addAction(self.newAction)
        folder_toolbar.addAction(self.loadWorkflowAction)
        folder_toolbar.addAction(self.saveAction)
        folder_toolbar.addAction(self.saveNewAction)
        folder_toolbar.addAction(self.openImagesAction)
        folder_toolbar.addAction(self.infoAction)
        folder_toolbar.addAction(self.undoAction)
        folder_toolbar.addAction(self.redoAction)
        folder_toolbar.addSeparator()
        self.validationStatusLabel = QPushButton('Looking good!', self)
        self.validationStatusLabel.setStyleSheet(
            'QPushButton { color: #1f7a1f; font-weight: bold; }'
        )
        self.validationStatusLabel.setCursor(Qt.PointingHandCursor)
        self.validationStatusLabel.setToolTip('Click to show workflow validation issues')
        self.validationStatusLabel.clicked.connect(self._onValidationStatusLabelClicked)
        folder_toolbar.addWidget(self.validationStatusLabel)
        folder_toolbar.addSeparator()
        folder_toolbar.addWidget(self._createToolbarIoLegend())
        self._updateIoActionsEnabledState()
        self._updateUndoRedoActions()
        
        self.addToolBar(folder_toolbar)

        # Now set up central widget and layout
        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        
        # Sidebar
        self.sidebar = WorkflowSidebar()
        self.sidebar.setDragEnabled(True)
        self.sidebar.setMinimumWidth(200)
        self.sidebar.setDragDropMode(QAbstractItemView.DragOnly)
        self.sidebar.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sidebar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Keep sidebar lightweight until experiment data is loaded.
        self._rebuildSidebarCards(build_widgets=False)

        self.sidebarDock = QDockWidget('Cards', self)
        self.sidebarDock.setObjectName('workflowSidebarDock')
        self.sidebarDock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.sidebarDock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.sidebarDock.setWidget(self.sidebar)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.sidebarDock)
        
        # Main panel
        self.mainPanel = QWidget()
        self.mainPanel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        mainLayout = QVBoxLayout(self.mainPanel)

        # Create the drop zone that will directly hold dropped cards
        self.dropZone = WorkflowZone(logger=self.logger, parent=self)
        self.dropZone.sigItemDropped.connect(self.addDroppedCard)
        mainLayout.addWidget(self.dropZone)

        logLabel = QLabel('Logs')
        mainLayout.addWidget(logLabel)

        self.logDisplay = QPlainTextEdit(self)
        self.logDisplay.setReadOnly(True)
        self.logDisplay.setMaximumBlockCount(500)
        self.logDisplay.setFixedHeight(120)
        self.logDisplay.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        mainLayout.addWidget(self.logDisplay)
        mainLayout.setStretch(0, 1)
        mainLayout.setStretch(1, 0)

        layout.addWidget(self.mainPanel)

        # Initially disable sidebar and dropZone until data is loaded
        self.sidebar.setEnabled(False)
        self.dropZone.setEnabled(False)
        self._refreshWorkflowValidationLabel()

        # Set window size based on screen resolution
        screen = QGuiApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.8)
        height = int(screen.height() * 0.8)
        self.setGeometry((screen.width() - width) // 2, (screen.height() - height) // 2, width, height)

    def run(self, module='acdc_workflow', logs_path=None):
        """Display the window and setup logging.
        
        Args:
            module (str): Module name for logging. Defaults to 'acdc_workflow'.
            logs_path (str): Path for log files (optional).
        """
        self.setWindowIcon()
        self.setWindowTitle()
        
        logger, logs_path, log_path, log_filename = myutils.setupLogger(
            module=module, logs_path=logs_path, caller=self._appName
        )

        self.curr_save_loaded = None
        self.module = module
        self.logger = logger
        self.log_path = log_path
        self.log_filename = log_filename
        self.logs_path = logs_path
        
        self.setupUI()
        self._attachGuiLogHandler()

        if self._version is not None:
            self.logger.info(f'Initializing GUI v{self._version}')
        else:
            self.logger.info('Initializing GUI...')
        
        self.show()

    def _attachGuiLogHandler(self):
        """Attach a logging handler that streams records to the GUI panel."""
        if self.logDisplay is None:
            return

        if self._guiLogHandler is not None:
            try:
                self.logger.removeHandler(self._guiLogHandler)
            except Exception:
                pass

        if not hasattr(self.logger, 'addHandler'):
            return

        handler = _WorkflowGuiLogHandler(self.logDisplay)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self._guiLogHandler = handler

    def _logInfo(self, message):
        """Log message supporting both Logger objects and callables."""
        try:
            if hasattr(self.logger, 'info'):
                self.logger.info(message)
            elif callable(self.logger):
                self.logger(message)
        except Exception:
            pass

    def _setWorkflowUIEnabled(self, enabled=True):
        """Enable or disable the workflow sidebar and dropzone."""
        if hasattr(self, 'sidebar') and self.sidebar is not None:
            self.sidebar.setEnabled(enabled)
        if hasattr(self, 'dropZone') and self.dropZone is not None:
            self.dropZone.setEnabled(enabled)

    def _setSaveActionsEnabled(self, enabled=True):
        """Enable or disable workflow save actions."""
        enabled = bool(enabled) and not self._ioActionsLocked()
        if hasattr(self, 'saveAction') and self.saveAction is not None:
            self.saveAction.setEnabled(enabled)
        if hasattr(self, 'saveNewAction') and self.saveNewAction is not None:
            self.saveNewAction.setEnabled(enabled)

    def _setLoadImagesActionEnabled(self, enabled=True):
        """Enable or disable the load-images action."""
        enabled = bool(enabled) and not self._ioActionsLocked()
        if hasattr(self, 'openImagesAction') and self.openImagesAction is not None:
            self.openImagesAction.setEnabled(enabled)

    def _ioActionsLocked(self):
        """Return True when toolbar I/O actions are temporarily locked."""
        return self._io_actions_lock_depth > 0

    def _updateIoActionsEnabledState(self):
        """Recompute enabled state for New/Load/Save/Save As/Load Images actions."""
        actions_enabled = not self._ioActionsLocked()
        if hasattr(self, 'newAction') and self.newAction is not None:
            self.newAction.setEnabled(actions_enabled)
        if hasattr(self, 'loadWorkflowAction') and self.loadWorkflowAction is not None:
            self.loadWorkflowAction.setEnabled(actions_enabled)

        self._setSaveActionsEnabled(self.data_loaded)
        self._setLoadImagesActionEnabled(self.data_loaded)

    @contextmanager
    def _lockIoActions(self):
        """Temporarily disable toolbar I/O actions during an active I/O handler."""
        self._io_actions_lock_depth += 1
        self._updateIoActionsEnabledState()
        try:
            yield
        finally:
            self._io_actions_lock_depth = max(0, self._io_actions_lock_depth - 1)
            self._updateIoActionsEnabledState()

    def _captureWorkflowState(self):
        """Capture cards, dialog content and connections for undo/redo."""
        state = {'cards': {}, 'lines': []}
        if not hasattr(self, 'dropZone') or self.dropZone is None:
            return state

        for card_id, proxy in self.dropZone.cards.items():
            card_widget = proxy.widget()
            if card_widget is None:
                continue

            dialog = getattr(card_widget, 'dialog', None)
            content = None
            if dialog is not None and hasattr(dialog, 'getContent'):
                content = copy.deepcopy(dialog.getContent())

            pos = proxy.pos()
            card_functions = getattr(card_widget, 'functions', None)
            state['cards'][str(card_id)] = {
                'kind': getattr(card_widget, 'title', '') or '',
                'card_key': getattr(card_functions, 'card_key', None),
                'position': {'x': pos.x(), 'y': pos.y()},
                'preInitWorkflow_res': getattr(card_widget, 'preInitWorkflow_res', None),
                'content': content,
            }

        state['lines'] = [
            {
                'from': {'card_id': start_card_id, 'port': start_idx},
                'to': {'card_id': end_card_id, 'port': end_idx},
            }
            for (start_card_id, start_idx), (end_card_id, end_idx) in self.dropZone.lines
        ]
        return state

    def _restoreWorkflowState(self, state):
        """Restore cards, dialog content and connections from a snapshot."""
        if not hasattr(self, 'dropZone') or self.dropZone is None:
            return

        cards_data = (state or {}).get('cards', {}) or {}
        lines_data = (state or {}).get('lines', []) or []

        prev_history_restoring = self._history_restoring
        prev_suspend_state = self._suspend_unsaved_tracking
        self._history_restoring = True
        self._suspend_unsaved_tracking = True
        try:
            self._clearWorkflow()

            saved_to_new_id = {}

            def _sort_key(card_id_text):
                try:
                    return int(card_id_text)
                except Exception:
                    return card_id_text

            for saved_card_id in sorted(cards_data.keys(), key=_sort_key):
                card_info = cards_data[saved_card_id] or {}
                kind = card_info.get('kind', '')
                card_key = card_info.get('card_key', None)
                pre_init_workflow_res = card_info.get('preInitWorkflow_res', None)

                functions = self._resolveWorkflowCardFunctions(kind, card_key=card_key)
                if functions is None:
                    continue

                position = card_info.get('position', {}) or {}
                x = position.get('x', 0)
                y = position.get('y', 0)

                card_widget = _WorkflowCardZoneWidget(
                    functions,
                    posData=self.posData,
                    zone=self.dropZone,
                    workflowGui=self,
                    logger=self.logger,
                    show_initial_dialog=False,
                    preInitWorkflow_res=pre_init_workflow_res,
                )
                self.dropZone.addCard(card_widget, x, y)

                new_card_id = getattr(card_widget, 'card_id', None)
                if new_card_id is None:
                    continue

                saved_to_new_id[str(saved_card_id)] = new_card_id

                dialog = getattr(card_widget, 'dialog', None)
                content = card_info.get('content', None)
                if content is not None and dialog is not None and hasattr(dialog, 'setContent'):
                    dialog.setContent(copy.deepcopy(content))
                    card_widget.updatePreview()

            for line_info in lines_data:
                from_info = (line_info or {}).get('from', {}) or {}
                to_info = (line_info or {}).get('to', {}) or {}

                saved_from_id = str(from_info.get('card_id'))
                saved_to_id = str(to_info.get('card_id'))
                new_from_id = saved_to_new_id.get(saved_from_id)
                new_to_id = saved_to_new_id.get(saved_to_id)
                if new_from_id is None or new_to_id is None:
                    continue

                from_port = int(from_info.get('port', 0))
                to_port = int(to_info.get('port', 0))

                start_proxy = self.dropZone.cards.get(new_from_id)
                end_proxy = self.dropZone.cards.get(new_to_id)
                if start_proxy is None or end_proxy is None:
                    continue

                start_card = start_proxy.widget()
                end_card = end_proxy.widget()
                if start_card is None or end_card is None:
                    continue

                self.dropZone.createConnection(
                    start_card,
                    True,
                    from_port,
                    end_card,
                    False,
                    to_port,
                )
        finally:
            self._history_restoring = prev_history_restoring
            self._suspend_unsaved_tracking = prev_suspend_state

        self._refreshWorkflowValidationLabel()

    def _registerWorkflowChange(self, previous_state):
        """Register a committed workflow mutation into the undo history."""
        if self._history_restoring or previous_state is None:
            return

        current_state = self._captureWorkflowState()
        if previous_state == current_state:
            return

        self._undo_stack.append(copy.deepcopy(previous_state))
        if len(self._undo_stack) > self._history_max_steps:
            self._undo_stack = self._undo_stack[-self._history_max_steps:]

        self._redo_stack.clear()
        self._updateUndoRedoActions()

    def _resetHistory(self):
        """Clear undo/redo stacks."""
        self._undo_stack.clear()
        self._redo_stack.clear()
        self._updateUndoRedoActions()

    def _updateUndoRedoActions(self):
        """Enable/disable undo-redo actions from current stack states."""
        if hasattr(self, 'undoAction') and self.undoAction is not None:
            self.undoAction.setEnabled(bool(self._undo_stack))
        if hasattr(self, 'redoAction') and self.redoAction is not None:
            self.redoAction.setEnabled(bool(self._redo_stack))

    def _onUndoTriggered(self):
        """Restore previous workflow snapshot (Ctrl+Z)."""
        if not self._undo_stack:
            return

        current_state = self._captureWorkflowState()
        target_state = self._undo_stack.pop()
        self._redo_stack.append(current_state)
        self._restoreWorkflowState(target_state)
        self.setUnsavedWork(True)
        self._updateUndoRedoActions()

    def _onRedoTriggered(self):
        """Re-apply reverted workflow snapshot (Ctrl+Y)."""
        if not self._redo_stack:
            return

        current_state = self._captureWorkflowState()
        target_state = self._redo_stack.pop()
        self._undo_stack.append(current_state)
        self._restoreWorkflowState(target_state)
        self.setUnsavedWork(True)
        self._updateUndoRedoActions()

    def _getOpenDialog(self):
        """Return a visible top-level dialog if one is currently open."""
        app = QApplication.instance()
        if app is None:
            return None

        for widget in app.topLevelWidgets():
            if widget is None or widget is self:
                continue
            if not widget.isVisible():
                continue
            if isinstance(widget, QDialog):
                return widget

        return None

    def _canSaveWorkflow(self):
        """Return True when save operations are allowed."""
        open_dialog = self._getOpenDialog()
        if open_dialog is None:
            return True

        dialog_title = open_dialog.windowTitle() or type(open_dialog).__name__
        self._logInfo(
            f'Save blocked while dialog "{dialog_title}" is open. '
            'Close it before saving.'
        )
        try:
            open_dialog.raise_()
            open_dialog.activateWindow()
        except Exception:
            pass

        parent = open_dialog if open_dialog is not None else self
        msg = widgets.myMessageBox(wrapText=False, parent=parent)
        msg.warning(
            parent,
            'Save blocked',
            'Close the currently open dialog before saving the workflow.'
        )
        return False

    def _onCardSelectionInvalid(self, card_widget, value):
        """Called when a dialog reports an invalid or failed configuration.

        Starts a ``QControlBlink`` on the card to attract attention and marks
        the card with a red background while the issue persists.

        Args:
            card_widget (_WorkflowCardZoneWidget): The card whose configuration is invalid.
            value (str): Human-readable description of the invalid state.
        """
        self._logInfo(
            f"Card '{card_widget.title}' reported an invalid state: {value}"
        )
        card_widget._has_invalid_selection = True
        card_widget._invalid_selection_value = value

        # Blink the card widget to alert the user
        blinker = qutils.QControlBlink(card_widget, qparent=self)
        blinker.start()
        # Keep blinker alive on the card so it is not garbage-collected mid-blink
        card_widget._selection_invalid_blinker = blinker

        # Re-apply invalid highlight after blink stops (blink toggles stylesheet).
        duration_ms = getattr(blinker, 'duration_ms', 2000)
        def _apply_invalid_style_if_needed():
            if getattr(card_widget, '_has_invalid_selection', False):
                card_widget.setStyleSheet("background-color: rgba(255, 0, 0, 50);")
        QTimer.singleShot(duration_ms + 50, _apply_invalid_style_if_needed)

    def _onCardSelectionValid(self, card_widget):
        """Clear invalid-selection visual state after a valid selection is confirmed."""
        card_widget._has_invalid_selection = False
        card_widget._invalid_selection_value = None
        card_widget.setStyleSheet('')

    def _notifyDialogsNewImageLoaded(self):
        """Notify existing workflow card dialogs that image data changed."""
        if not hasattr(self, 'dropZone') or self.dropZone is None:
            return

        for proxy in self.dropZone.cards.values():
            card_widget = proxy.widget() if proxy is not None else None
            if card_widget is None:
                continue

            dialog = getattr(card_widget, 'dialog', None)
            if dialog is None:
                continue

            if hasattr(dialog, 'new_image_loaded'):
                try:
                    dialog.new_image_loaded()
                except TypeError:
                    try:
                        dialog.new_image_loaded(self.posData)
                    except Exception as e:
                        self._logInfo(f"Error in new_image_loaded for '{card_widget.title}': {e}")
                except Exception as e:
                    self._logInfo(f"Error in new_image_loaded for '{card_widget.title}': {e}")

            try:
                card_widget.updatePreview()
            except Exception:
                pass

    def _loadExperimentData(self):
        """Load experiment data by prompting user to select folders.
        
        Shows dialog with instructions and lets user select experiment folders.
        Initializes data and enables the workflow UI if successful.
        
        Returns:
            bool: True if data was successfully loaded, False otherwise.
        """
        custom_txt = (
            "After you click \"Ok\" on this dialog you will be asked "
            "to <b>select the experiment folders</b>, one by one.<br><br>"
        )
        
        if self.mainWin is None or not hasattr(self.mainWin, 'getSelectedExpPaths'):
            self._logInfo('Error: Cannot access experiment folder selection dialog.')
            return False
        
        selectedExpPaths = self.mainWin.getSelectedExpPaths(
            'Workflow GUI',
            custom_txt=custom_txt
        )
        
        if selectedExpPaths is None:
            self._logInfo('No experiment folders selected.')
            return False
        
        try:
            self.selectedExpPaths = selectedExpPaths
            self.initData()
            self._rebuildSidebarCards(build_widgets=True)
            self._notifyDialogsNewImageLoaded()
            self.data_loaded = True
            self._setWorkflowUIEnabled(True)
            self._setSaveActionsEnabled(True)
            self._setLoadImagesActionEnabled(True)
            self.setUnsavedWork(False)
            self._logInfo(f'Loaded data from {len(self.data)} positions.')
            return True
        except Exception as e:
            self._logInfo(f'Error loading experiment data: {e}')
            self._setWorkflowUIEnabled(False)
            return False

    def setUnsavedWork(self, unsaved=True):
        """Update the unsaved-work flag, respecting temporary tracking suspension."""
        unsaved = bool(unsaved)
        if unsaved and self._suspend_unsaved_tracking:
            return
        self.unsaved_work = unsaved

    def _askSaveBeforeClosing(self):
        """Ask user how to proceed when there are unsaved workflow changes."""
        if self._isAskingSaveBeforeClosing:
            return False

        self._isAskingSaveBeforeClosing = True
        msg = widgets.myMessageBox(wrapText=False, parent=self)
        txt = (
            'There are unsaved workflow changes.<br><br>'
            'Do you want to save before closing?'
        )
        try:
            _, discardButton, saveButton = msg.warning(
                self,
                'Unsaved workflow changes',
                txt,
                buttonsTexts=('Cancel', 'Discard', 'Save')
            )

            if msg.cancel:
                return False

            if msg.clickedButton == saveButton:
                self._onSaveWorkflowTriggered()
                return not self.unsaved_work

            if msg.clickedButton == discardButton:
                return True

            return False
        finally:
            self._isAskingSaveBeforeClosing = False
    
    def closeEvent(self, event):
        """Handle window close event.
        
        Emits the sigClosed signal and prompts to save if there are
        unsaved workflow changes.
        
        Args:
            event: The close event.
        """
        if self._isAskingSaveBeforeClosing:
            event.ignore()
            return

        if self.closeGUI:
            event.ignore()
            return

        if self.unsaved_work and not self._askSaveBeforeClosing():
            event.ignore()
            return

        self.closeGUI = True
        self.sigClosed.emit(self)
        
    def setWindowIcon(self, icon=None):
        """Set the window icon.
        
        Args:
            icon (QIcon, optional): Icon to set. Defaults to embedded icon.ico.
        """
        if icon is None:
            icon = QIcon(":icon.ico")
        super().setWindowIcon(icon)
    
    def setWindowTitle(self, title=None):
        """Set the window title.
        
        Args:
            title (str, optional): Title to set. Defaults to version and 'workflow GUI'.
        """
        if title is None:
            title = f'Cell-ACDC v{self._acdc_version} - workflow GUI'
        super().setWindowTitle(title)

    def _backupExistingWorkflow(self, save_dir, max_backups=5):
        """Backup current workflow files before overwriting them.

        Backups are stored in ``<save_dir>/backups`` as timestamped folders.
        Only the latest ``max_backups`` backup folders are kept.
        """
        if not os.path.isdir(save_dir):
            return

        entries_to_backup = []
        for entry in os.listdir(save_dir):
            if entry == 'backups':
                continue
            entry_path = os.path.join(save_dir, entry)
            if os.path.isfile(entry_path) or os.path.isdir(entry_path):
                entries_to_backup.append((entry, entry_path))

        if not entries_to_backup:
            return

        backups_dir = os.path.join(save_dir, 'backups')
        os.makedirs(backups_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f'workflow_{timestamp}'
        backup_path = os.path.join(backups_dir, backup_name)
        suffix = 1
        while os.path.exists(backup_path):
            backup_name = f'workflow_{timestamp}_{suffix}'
            backup_path = os.path.join(backups_dir, backup_name)
            suffix += 1

        os.makedirs(backup_path, exist_ok=False)
        for entry, entry_path in entries_to_backup:
            dst_path = os.path.join(backup_path, entry)
            if os.path.isdir(entry_path):
                shutil.copytree(entry_path, dst_path)
            else:
                shutil.copy2(entry_path, dst_path)

        self._logInfo(f'Created workflow backup: {backup_path}')

        backup_dirs = [
            os.path.join(backups_dir, name)
            for name in os.listdir(backups_dir)
            if os.path.isdir(os.path.join(backups_dir, name))
        ]
        backup_dirs.sort(key=os.path.getmtime, reverse=True)
        for old_backup in backup_dirs[max_backups:]:
            try:
                shutil.rmtree(old_backup)
                self._logInfo(f'Removed old workflow backup: {old_backup}')
            except Exception as e:
                self._logInfo(f'Failed to remove old backup {old_backup}: {e}')

    def saveWorkflow(self, save_dir):
        """Save the current workflow to disk.

        Creates `save_dir` and writes a `main.json` manifest containing cards
        metadata and connection lines. Each card also saves its own dialog
        content through `dialog.saveContent(path)`.

        Args:
            save_dir (str): Destination folder for workflow files.

        Returns:
            str or None: Path to the written `main.json` file.
        """
        if not self._canSaveWorkflow():
            return None

        os.makedirs(save_dir, exist_ok=True)
        self._backupExistingWorkflow(save_dir, max_backups=5)
        # clear old files in the save directory to prevent orphaned files from previous saves
        for filename in os.listdir(save_dir):
            file_path = os.path.join(save_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
        self.logger.info(f"Saving workflow to: {save_dir}")
        cards_data = {}
        for card_id, proxy in self.dropZone.cards.items():
            card_widget = proxy.widget()
            if card_widget is None:
                continue

            title = getattr(card_widget, 'title', '') or ''
            # Reuse shared helper for filesystem-safe text.
            safe_title = to_alphanumeric(title).strip('_').lower()
            if not safe_title:
                safe_title = 'card'
            save_filename = f'{card_id}_{safe_title}'
            save_path = os.path.join(save_dir, save_filename)

            dialog = getattr(card_widget, 'dialog', None)
            if dialog is not None and hasattr(dialog, 'saveContent'):
                dialog.saveContent(save_path)

            pos = proxy.pos()
            card_functions = getattr(card_widget, 'functions', None)
            cards_data[str(card_id)] = {
                'kind': title,
                'card_key': getattr(card_functions, 'card_key', None),
                'position': {'x': pos.x(), 'y': pos.y()},
                'preInitWorkflow_res': getattr(card_widget, 'preInitWorkflow_res', None),
                'save_path': save_filename,
            }

        lines_data = [
            {
                'from': {'card_id': start_card_id, 'port': start_idx},
                'to': {'card_id': end_card_id, 'port': end_idx},
            }
            for (start_card_id, start_idx), (end_card_id, end_idx) in self.dropZone.lines
        ]

        main_data = {
            'cards': cards_data,
            'lines': lines_data,
            'selectedExpPaths': self.selectedExpPaths or {},
        }

        main_json_path = os.path.join(save_dir, 'main.json')
        load.write_json(main_data, main_json_path, indent=2)
        self.setUnsavedWork(False)

        return main_json_path

    def _getCardSavePathCandidates(self, save_dir, save_path):
        """Return candidate filesystem paths for a card settings save path."""
        base_path = (
            save_path if os.path.isabs(save_path)
            else os.path.join(save_dir, save_path)
        )

        candidates = [base_path]
        base_path_lower = base_path.lower()
        has_supported_extension = any(
            base_path_lower.endswith(ext)
            for ext in SUPPORTED_EXTENSIONS_CARD_SETTINGS
        )
        if not has_supported_extension:
            for ext in SUPPORTED_EXTENSIONS_CARD_SETTINGS:
                candidates.append(f'{base_path}{ext}')

        # Remove duplicates while preserving order.
        return list(dict.fromkeys(candidates))

    def _findExistingCardSavePath(self, save_dir, save_path):
        """Return the first existing card settings path, if any."""
        candidates = self._getCardSavePathCandidates(save_dir, save_path)
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
        return None

    def verifyWorkflowFolderStructure(self, save_dir, require_card_files=True):
        """Verify the workflow save folder structure and manifest schema.

        Args:
            save_dir (str): Folder expected to contain workflow save files.
            require_card_files (bool): If True, verify that each card save file exists.

        Returns:
            tuple: (is_valid, errors, main_data)
                is_valid (bool): True if structure is valid.
                errors (list[str]): Validation errors.
                main_data (dict or None): Parsed main.json if available.
        """
        errors = []
        main_data = None

        if not os.path.isdir(save_dir):
            errors.append(f"Workflow folder not found: {save_dir}")
            return False, errors, main_data

        main_json_path = os.path.join(save_dir, 'main.json')
        if not os.path.exists(main_json_path):
            errors.append(f"Missing workflow manifest: {main_json_path}")
            return False, errors, main_data

        main_data = load.read_json(
            main_json_path, logger_func=self._logInfo, desc='workflow data'
        )
        if not isinstance(main_data, dict):
            errors.append(f"Invalid workflow manifest format: {main_json_path}")
            return False, errors, None

        cards_data = main_data.get('cards')
        lines_data = main_data.get('lines')

        if not isinstance(cards_data, dict):
            errors.append("Invalid 'cards' section in main.json (expected dict).")
        if not isinstance(lines_data, list):
            errors.append("Invalid 'lines' section in main.json (expected list).")

        if isinstance(cards_data, dict):
            for card_id, card_info in cards_data.items():
                if not isinstance(card_info, dict):
                    errors.append(f"Card '{card_id}' entry is invalid (expected dict).")
                    continue

                kind = card_info.get('kind')
                if not isinstance(kind, str) or not kind.strip():
                    errors.append(f"Card '{card_id}' has invalid or missing 'kind'.")

                position = card_info.get('position')
                if not isinstance(position, dict):
                    errors.append(f"Card '{card_id}' has invalid or missing 'position'.")
                else:
                    x = position.get('x')
                    y = position.get('y')
                    if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                        errors.append(f"Card '{card_id}' position must contain numeric 'x' and 'y'.")

                save_path = card_info.get('save_path')
                if not isinstance(save_path, str) or not save_path.strip():
                    errors.append(f"Card '{card_id}' has invalid or missing 'save_path'.")
                    continue

                if require_card_files:
                    card_save_path = self._findExistingCardSavePath(save_dir, save_path)
                    if card_save_path is None:
                        candidates = self._getCardSavePathCandidates(save_dir, save_path)
                        errors.append(
                            f"Card '{card_id}' save file not found. "
                            f"Tried: {', '.join(candidates)}"
                        )

        if isinstance(lines_data, list):
            for idx, line_info in enumerate(lines_data):
                if not isinstance(line_info, dict):
                    errors.append(f"Line entry at index {idx} is invalid (expected dict).")
                    continue

                from_info = line_info.get('from')
                to_info = line_info.get('to')
                if not isinstance(from_info, dict) or not isinstance(to_info, dict):
                    errors.append(f"Line entry at index {idx} must contain dict 'from' and 'to'.")
                    continue

                for endpoint_name, endpoint in (('from', from_info), ('to', to_info)):
                    if 'card_id' not in endpoint or 'port' not in endpoint:
                        errors.append(
                            f"Line entry at index {idx} endpoint '{endpoint_name}' "
                            "must contain 'card_id' and 'port'."
                        )

        return len(errors) == 0, errors, main_data

    def _resolveWorkflowCardFunctions(self, kind, card_key=None):
        """Resolve saved card type to a sidebar workflow functions object.

        Resolution order:
        1) Stable internal `card_key` (preferred)
        2) Backward-compatible title-based matching
        """
        available_functions = []
        for i in range(self.sidebar.count()):
            item = self.sidebar.item(i)
            functions = item.data(Qt.UserRole)
            if functions is not None:
                available_functions.append(functions)

        if not available_functions:
            return None

        card_key_text = str(card_key or '').strip().lower()
        if card_key_text:
            for functions in available_functions:
                functions_key = str(getattr(functions, 'card_key', '') or '').strip().lower()
                if functions_key and functions_key == card_key_text:
                    return type(functions)()

        kind_text = str(kind or '').strip()
        kind_lower = kind_text.lower()

        # First pass: exact title match.
        for functions in available_functions:
            if getattr(functions, 'title', '') == kind_text:
                return type(functions)()

        # Second pass: support dynamic titles like "image: CH1".
        if ':' in kind_lower:
            prefix = kind_lower.split(':', 1)[0].strip()
            if prefix == 'image':
                for functions in available_functions:
                    title_lower = getattr(functions, 'title', '').lower()
                    if 'image' in title_lower:
                        return type(functions)()
            if prefix in ('segmentation', 'segm'):
                for functions in available_functions:
                    title_lower = getattr(functions, 'title', '').lower()
                    if 'segmentation' in title_lower or 'segm' in title_lower:
                        return type(functions)()

        # Final pass: fuzzy containment.
        for functions in available_functions:
            title = getattr(functions, 'title', '')
            title_lower = title.lower()
            if kind_lower and (kind_lower in title_lower or title_lower in kind_lower):
                return type(functions)()

        return None

    def _clearWorkflow(self):
        """Remove all cards and connections from the current workflow zone."""
        if not hasattr(self, 'dropZone') or self.dropZone is None:
            return
        card_ids = list(self.dropZone.cards.keys())
        for card_id in card_ids:
            proxy = self.dropZone.cards.get(card_id)
            if proxy is None:
                continue
            card_widget = proxy.widget()
            if card_widget is not None:
                self.dropZone.removeCard(card_widget) # also removes connections
        self._refreshWorkflowValidationLabel()

    def loadWorkflow(self, save_dir):
        """Load a saved workflow from disk and restore cards and connections.

        Args:
            save_dir (str): Folder containing `main.json` and per-card files.

        Returns:
            bool: True if loading completed successfully, False otherwise.
        """
        is_valid, errors, main_data = self.verifyWorkflowFolderStructure(
            save_dir, require_card_files=True
        )
        if not is_valid:
            self._logInfo('Invalid workflow folder structure:')
            for error in errors:
                self._logInfo(f' - {error}')
            return False

        cards_data = main_data.get('cards', {})
        lines_data = main_data.get('lines', [])
        saved_selected_paths = main_data.get('selectedExpPaths', {})
        
        # Restore selectedExpPaths and reload data if they exist
        if saved_selected_paths:
            try:
                self.selectedExpPaths = saved_selected_paths
                self.initData()
                self._rebuildSidebarCards(build_widgets=True)
                self.data_loaded = True
                self._setWorkflowUIEnabled(True)
                self._setSaveActionsEnabled(True)
                self._setLoadImagesActionEnabled(True)
            except Exception as e:
                self._logInfo(f'Error restoring experiment data: {e}')
                self._setWorkflowUIEnabled(False)
                return False

        prev_suspend_state = self._suspend_unsaved_tracking
        self._suspend_unsaved_tracking = True
        try:
            self._clearWorkflow()

            saved_to_new_id = {}

            def _sort_key(card_id_text):
                try:
                    return int(card_id_text)
                except Exception:
                    return card_id_text

            for saved_card_id in sorted(cards_data.keys(), key=_sort_key):
                card_info = cards_data[saved_card_id] or {}
                kind = card_info.get('kind', '')
                card_key = card_info.get('card_key', None)
                pre_init_workflow_res = card_info.get('preInitWorkflow_res', None)

                functions = self._resolveWorkflowCardFunctions(kind, card_key=card_key)
                if functions is None:
                    self._logInfo(
                        f"Could not resolve workflow card kind '{kind}'. "
                        f"Skipping card {saved_card_id}."
                    )
                    continue

                position = card_info.get('position', {}) or {}
                x = position.get('x', 0)
                y = position.get('y', 0)

                try:
                    card_widget = _WorkflowCardZoneWidget(
                        functions,
                        posData=self.posData,
                        zone=self.dropZone,
                        workflowGui=self,
                        logger=self.logger,
                        show_initial_dialog=False,
                        preInitWorkflow_res=pre_init_workflow_res,
                    )
                    self.dropZone.addCard(card_widget, x, y)
                except Exception as e:
                    self._logInfo(f"Failed to create card '{kind}' ({saved_card_id}): {e}")
                    continue

                new_card_id = getattr(card_widget, 'card_id', None)
                if new_card_id is None:
                    continue

                saved_to_new_id[str(saved_card_id)] = new_card_id

                save_path = card_info.get('save_path')
                if save_path:
                    card_save_path = self._findExistingCardSavePath(save_dir, save_path)
                    dialog = getattr(card_widget, 'dialog', None)
                    try:
                        dialog.loadContent(card_save_path)
                    except Exception as e:
                        self._logInfo(f"Failed to load card content from {card_save_path}: {e}")
                        traceback.print_exc()

            for line_info in lines_data:
                from_info = (line_info or {}).get('from', {}) or {}
                to_info = (line_info or {}).get('to', {}) or {}

                saved_from_id = str(from_info.get('card_id'))
                saved_to_id = str(to_info.get('card_id'))
                new_from_id = saved_to_new_id.get(saved_from_id)
                new_to_id = saved_to_new_id.get(saved_to_id)

                if new_from_id is None or new_to_id is None:
                    continue

                from_port = from_info.get('port', 0)
                to_port = to_info.get('port', 0)

                try:
                    from_port = int(from_port)
                    to_port = int(to_port)
                except Exception:
                    continue

                start_proxy = self.dropZone.cards.get(new_from_id)
                end_proxy = self.dropZone.cards.get(new_to_id)
                if start_proxy is None or end_proxy is None:
                    continue

                start_card = start_proxy.widget()
                end_card = end_proxy.widget()
                if start_card is None or end_card is None:
                    continue

                self.dropZone.createConnection(
                    start_card,
                    True,
                    from_port,
                    end_card,
                    False,
                    to_port,
                )
        finally:
            self._suspend_unsaved_tracking = prev_suspend_state

        self._notifyDialogsNewImageLoaded()
        # Re-apply connector colors once all post-load updates have completed.
        for proxy in self.dropZone.cards.values():
            card = proxy.widget()
            if card is not None and hasattr(card, 'refreshPortColors'):
                card.refreshPortColors()
        self._setSaveActionsEnabled(True)
        self.setUnsavedWork(False)
        self._refreshWorkflowValidationLabel()
        return True
    
    def verifyWorkflowSaveFiles(self, save_dir):
        """Verify that all expected card save files exist in the workflow folder.

        Args:
            save_dir (str): Folder containing `main.json` and per-card files.   
            
        Returns:
            bool: True if all expected card files exist, False otherwise.
        """
        main_json_path = os.path.join(save_dir, 'main.json')
        if not os.path.exists(main_json_path):
            self._logInfo(f"Workflow manifest not found: {main_json_path}")
            return False

        main_data = load.read_json(
            main_json_path, logger_func=self._logInfo, desc='workflow data'
        )
        if not isinstance(main_data, dict):
            self._logInfo(f"Invalid workflow manifest format: {main_json_path}")
            return False

        cards_data = main_data.get('cards')
        if not isinstance(cards_data, dict):
            self._logInfo("Invalid 'cards' section in main.json (expected dict).")
            return False

        all_files_exist = True
        for card_id, card_info in cards_data.items():
            save_path = card_info.get('save_path')
            if not isinstance(save_path, str) or not save_path.strip():
                self._logInfo(f"Card '{card_id}' has invalid or missing 'save_path'.")
                all_files_exist = False
                continue

            card_save_path = self._findExistingCardSavePath(save_dir, save_path)
            if card_save_path is None:
                candidates = self._getCardSavePathCandidates(save_dir, save_path)
                self._logInfo(
                    f"Card '{card_id}' save file not found. "
                    f"Tried: {', '.join(candidates)}"
                )
                all_files_exist = False

        return all_files_exist

    def _getCardOptionalInputIndices(self, card_widget):
        """Return optional input indices declared by the card dialog.

        Supported ``dialog.optional_inputs_n`` values:
        - missing/None: no optional inputs
        - bool: apply value to all inputs (True=all optional, False=none)
        - dict[int, bool]: per-input optional flags

        Parsing is intentionally permissive: malformed dict entries are skipped.
        """
        dialog = getattr(card_widget, 'dialog', None)
        optional_inputs_n = getattr(dialog, 'optional_inputs_n', None)
        n_inputs = len(getattr(card_widget, 'input_circles', []))
        if n_inputs <= 0:
            return set()

        if optional_inputs_n is None:
            return set()

        if isinstance(optional_inputs_n, bool):
            return set(range(n_inputs)) if optional_inputs_n else set()

        return set(
            idx for idx, is_optional in (optional_inputs_n or {}).items()
            if is_optional and isinstance(idx, int) and (0 <= idx < n_inputs)
        )

    def _formatCardLabel(self, card_id):
        """Return a short human-readable card label."""
        proxy = self.dropZone.cards.get(card_id) if hasattr(self, 'dropZone') else None
        card = proxy.widget() if proxy is not None else None
        title = getattr(card, 'title', '') if card is not None else ''
        if title:
            return f"ID={card_id} ('{title}')"
        return f"ID={card_id}"

    def _analyzeWorkflowGraphIssues(self):
        """Analyze workflow graph and return errors plus issue targets.

        Returns:
            tuple: (errors, issue_card_ids, issue_line_keys)
        """
        errors = []
        issue_card_ids = set()
        issue_line_keys = []

        if not hasattr(self, 'dropZone') or self.dropZone is None:
            return errors, issue_card_ids, issue_line_keys

        card_ids = set(self.dropZone.cards.keys())
        incoming_by_card = {card_id: set() for card_id in card_ids}
        adjacency = {card_id: set() for card_id in card_ids}

        for line_data in self.dropZone.lines:
            (from_card_id, _from_port), (to_card_id, to_port) = line_data
            if to_card_id in incoming_by_card:
                incoming_by_card[to_card_id].add(to_port)
            if from_card_id in adjacency and to_card_id in adjacency:
                adjacency[from_card_id].add(to_card_id)

        for card_id in card_ids:
            proxy = self.dropZone.cards.get(card_id)
            card_widget = proxy.widget() if proxy is not None else None
            if card_widget is None:
                continue

            invalid_value = getattr(card_widget, '_invalid_selection_value', None)
            if getattr(card_widget, '_has_invalid_selection', False) and invalid_value:
                errors.append(
                    f"Card {self._formatCardLabel(card_id)} has invalid configuration: {invalid_value}."
                )

            n_inputs = len(getattr(card_widget, 'input_circles', []))
            if n_inputs <= 0:
                continue

            optional_indices = self._getCardOptionalInputIndices(card_widget)
            connected_inputs = incoming_by_card.get(card_id, set())

            missing_required = [
                idx for idx in range(n_inputs)
                if idx not in connected_inputs and idx not in optional_indices
            ]
            if not missing_required:
                continue

            missing_ports = ', '.join(str(idx + 1) for idx in missing_required)
            errors.append(
                f"Card {self._formatCardLabel(card_id)} has missing required input port(s): {missing_ports}."
            )
            issue_card_ids.add(card_id)

        for line_data in self.dropZone.lines:
            (from_card_id, from_port), (to_card_id, to_port) = line_data

            from_proxy = self.dropZone.cards.get(from_card_id)
            to_proxy = self.dropZone.cards.get(to_card_id)
            from_card = from_proxy.widget() if from_proxy is not None else None
            to_card = to_proxy.widget() if to_proxy is not None else None
            if from_card is None or to_card is None:
                continue

            output_type = from_card.output_types.get(from_port)
            input_type_accepted = to_card._getDialogInputTypesAccepted().get(to_port)
            if self.dropZone._areTypesCompatible(output_type, input_type_accepted):
                continue

            errors.append(
                'Type mismatch: '
                f"{self._formatCardLabel(from_card_id)} output {from_port + 1} "
                f"(type={output_type}) -> "
                f"{self._formatCardLabel(to_card_id)} input {to_port + 1} "
                f"(accepts={input_type_accepted})."
            )
            issue_line_keys.append(line_data)
            issue_card_ids.add(from_card_id)
            issue_card_ids.add(to_card_id)

        for card_id in card_ids:
            proxy = self.dropZone.cards.get(card_id)
            card_widget = proxy.widget() if proxy is not None else None
            if card_widget is None:
                continue
            dialog = getattr(card_widget, 'dialog', None)
            if dialog is not None and hasattr(dialog, 'hasInvalidContent') and dialog.hasInvalidContent():
                errors.append(
                    f"{self._formatCardLabel(card_id)} has stale feature selections. "
                    "Update or remove the highlighted metrics."
                )
                issue_card_ids.add(card_id)

        visit_state = {card_id: 0 for card_id in card_ids}  # 0=unvisited, 1=visiting, 2=done
        recursion_stack = []
        found_cycles = []
        seen_cycle_signatures = set()

        def _dfs(card_id):
            visit_state[card_id] = 1
            recursion_stack.append(card_id)

            for downstream_id in adjacency.get(card_id, ()):
                state = visit_state.get(downstream_id, 2)
                if state == 0:
                    _dfs(downstream_id)
                elif state == 1:
                    try:
                        start_idx = recursion_stack.index(downstream_id)
                    except ValueError:
                        start_idx = 0
                    cycle = recursion_stack[start_idx:] + [downstream_id]
                    signature = tuple(cycle)
                    if signature not in seen_cycle_signatures:
                        seen_cycle_signatures.add(signature)
                        found_cycles.append(cycle)

            recursion_stack.pop()
            visit_state[card_id] = 2

        for card_id in card_ids:
            if visit_state[card_id] == 0:
                _dfs(card_id)

        for cycle in found_cycles:
            cycle_labels = ' -> '.join(self._formatCardLabel(card_id) for card_id in cycle)
            errors.append(f"Workflow loop detected: {cycle_labels}.")
            issue_card_ids.update(cycle)

        return errors, issue_card_ids, issue_line_keys

    def validateWorkflowGraph(self, log_errors=True):
        """Validate workflow graph for required inputs, type mismatches, and cycles.

        Returns:
            tuple: (is_valid, errors)
                is_valid (bool): True when no validation issue was found.
                errors (list[str]): Human-readable validation messages.
        """
        errors, _issue_card_ids, _issue_line_keys = self._analyzeWorkflowGraphIssues()
        is_valid = len(errors) == 0
        if (not is_valid) and log_errors:
            self._logInfo('Workflow validation failed:')
            for error in errors:
                self._logInfo(f' - {error}')

        return is_valid, errors

    def _showWorkflowValidationErrors(self, errors):
        """Show workflow validation errors in a compact warning dialog."""
        if not errors:
            return

        max_errors_to_show = 12
        visible_errors = errors[:max_errors_to_show]
        bullet_list = '\n'.join(f'- {error}' for error in visible_errors)
        hidden_count = max(0, len(errors) - len(visible_errors))
        suffix = '' if hidden_count == 0 else f'\n... and {hidden_count} more issue(s).'
        txt = (
            'Workflow validation issues:\n\n'
            f'{bullet_list}{suffix}'
        )
        previous_msg = getattr(self, '_validationErrorsMsgBox', None)
        if previous_msg is not None:
            try:
                previous_msg.close()
            except Exception:
                pass

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle('Invalid workflow')
        msg.setText(txt)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.setModal(False)
        msg.setWindowModality(Qt.NonModal)
        msg.setWindowFlag(Qt.WindowStaysOnTopHint, True)
        msg.setAttribute(Qt.WA_DeleteOnClose, True)

        def _clear_validation_msgbox(*_):
            self._validationErrorsMsgBox = None

        msg.finished.connect(_clear_validation_msgbox)
        self._validationErrorsMsgBox = msg
        msg.show()
        msg.raise_()
        msg.activateWindow()

    def _onNewWorkflowTriggered(self):
        """Load experiment data and create a new workflow."""
        with self._lockIoActions():
            # check for unsaved work before clearing
            if self.unsaved_work and not self._askSaveBeforeClosing():
                return
            
            # Load experiment data first
            if not self._loadExperimentData():
                return
            
            self._clearWorkflow()
            self.loadedWorkflowPath = None
            self.setUnsavedWork(False)
            self._resetHistory()
            self._logInfo('Created new empty workflow.')

    def _onLoadWorkflowTriggered(self):
        """Prompt for a workflow folder and load it."""
        with self._lockIoActions():
            save_dir = self.selectSaveFolder(save=False)
            if save_dir is None:
                return

            if self.verifyWorkflowSaveFiles(save_dir):
                if self.loadWorkflow(save_dir):
                    self.loadedWorkflowPath = save_dir
                    self._resetHistory()
            else:
                self._logInfo(
                    "Selected folder is missing expected workflow files. "
                    "Please select a valid workflow folder."
                )

    def _onSaveWorkflowAsTriggered(self):
        """Prompt for a destination and save the workflow there."""        
        with self._lockIoActions():
            if not self._canSaveWorkflow():
                return

            save_dir = self.selectSaveFolder(save=True)
            if not save_dir:
                return

            main_json_path = self.saveWorkflow(save_dir)
            if main_json_path is not None:
                self.loadedWorkflowPath = save_dir

    def _onSaveWorkflowTriggered(self):
        """Save to current workflow path or prompt if not yet set."""
        with self._lockIoActions():
            if not self._canSaveWorkflow():
                return

            if self.loadedWorkflowPath is None:
                self._onSaveWorkflowAsTriggered()
                return

            self.saveWorkflow(self.loadedWorkflowPath)

    def _onLoadImagesTriggered(self):
        """Load or change experiment data."""
        with self._lockIoActions():
            self._loadExperimentData()
        
    # handle key press events for shortcuts
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts.
        
        In debug mode, pressing 'Q' prints the workflow connection data.
        
        Args:
            event: The key event.
        """
        if event.key() == Qt.Key_Q and self.debug:
            # print each curr_input_types for each card in the workflow if dialog has attr
            for card_id, proxy in self.dropZone.cards.items():
                card_widget = proxy.widget()
                print(card_widget.dialog.getContent())
        super().keyPressEvent(event)
        
    def selectSaveFolder(self, save=True):
        """Select workflow folder.

        For saving, first select a parent folder and then ask for the workflow
        subfolder name using the existing filename dialog.
        """
        os.makedirs(workflow_default_save_folderpath, exist_ok=True)
        base_dir = apps.get_existing_directory(
            parent=self,
            caption=(
                'Select parent folder where to save the workflow'
                if save else 'Select workflow folder to load'
            ),
            basedir=workflow_default_save_folderpath,
            allow_images_path=False,
        )

        if not base_dir:
            return None

        if not save:
            return base_dir

        existing_names = [
            name
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name))
        ]

        name_win = apps.filenameDialog(
            basename='',
            ext='',
            title='Save workflow',
            hintText='Insert a name for the workflow folder:',
            existingNames=existing_names,
            defaultEntry='workflow',
            allowEmpty=False,
            parent=self,
        )
        name_win.exec_()
        if name_win.cancel:
            return None
        
        path = os.path.join(base_dir, name_win.entryText)
        # check if path is empty
        # if os.path.exists(path) and os.listdir(path):
        #     msg = widgets.myMessageBox(wrapText=False, parent=self)
        #     txt = (
        #         f"The folder <br>'{path}'<br>already exists and is not empty. "
        #         "Saving the workflow here may overwrite existing files. "
        #         "Do you want to proceed?"
        #     )
        #     _, noButton, yesButton = msg.warning(
        #         self, 'Folder not empty', txt, 
        #         buttonsTexts=('No', 'Yes'),
        #     )
        #     if msg.cancel or msg.clickedButton == noButton:
        #         return None
        return path