import inspect

from qtpy.QtGui import QDropEvent, QIcon, QGuiApplication, QPixmap, QDrag, QCursor, QPainterPath, QPen, QColor, QPolygonF
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QListWidget, QListWidgetItem, 
    QLabel, QVBoxLayout, QSizePolicy, QScrollArea,
    QAbstractItemView, QApplication, QGraphicsView, QGraphicsScene, QGraphicsProxyWidget, QGraphicsPixmapItem, QGraphicsLineItem,
    QComboBox
)

from . import myutils, apps, widgets, qutils, load, printl
from .apps import QBaseDialog
from qtpy.QtCore import Signal, Qt, QDataStream, QIODevice, QMimeData, QPointF, QTimer

import os

class WorkflowBaseFunctions():
    """Base class for workflow card widgets.
    
    This class provides a template for creating workflow cards that can be added
    to the workflow GUI. Subclasses should implement dryrunDialog() and 
    setupDialog() methods. This is the "functions" side of things, the dialogs
    themselves are defined separately.
    
    Attributes:
        title (str): Display title of the workflow card.
        posData: Position data containing relevant information for the workflow.
        input_types_accepted (dict): Maps input index to accepted input type(s).
            Example: {0: 'img', 1: ['img', 'segm']}
        input_types (dict): Maps input index to the currently connected input type.
            Example: {0: 'img', 1: 'segm'}
        output_types (dict): Maps output index to output type. Example: {0: 'segm'}
        setInputs (callable): Method to set input count and types.
        setOutputs (callable): Method to set output count and types.
        updatePreview (callable): Method to update the preview image.
        updateTitle (callable): Method to update the card title.
    
    Example:
        class MyWorkflowCard(WorkflowBaseFunctions):
            def __init__(self):
                self.title = "My Workflow Step"
            
            def dryrunDialog(self, parent=None):
                return MyDialog(parent=parent)
            
            def setupDialog(self, parent=None):
                self.setInputs({0: 'img'})
                self.setOutputs({0: 'img'})
                return MyDialog(parent=parent)
    """
    def __init__(self):
        """Initialize the base workflow functions class."""
        
    def runDialog_cb(self, dialog):
        """Show the workflow dialog.
        
        Args:
            dialog: The dialog widget to display.
        """
        # The dialog is usually hidden (not destroyed) between openings.
        # Re-apply input-type dependent UI state right before showing.
        if hasattr(dialog, 'updatedInputTypes'):
            dialog.updatedInputTypes()
        dialog.show()
        
    def getDialogPreview(self, dialog):
        """Capture a screenshot of the dialog for preview display.
        
        Args:
            dialog: The dialog widget to capture.
        
        Returns:
            QPixmap: A scaled (220x110) screenshot of the dialog, or None if capture fails.
        """
        try:
            # Grab a screenshot of the dialog to update preview
            pix = dialog.grab().scaled(220, 110, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            return pix
        except Exception as e:
            # If screenshot fails, just leave preview as is
            print(f"Failed to update preview after dialog close: {e}")
        return None
    
    def setupDialog(self, parent=None, workflowGui=None, posData=None):
        """Configure to return the dialog widget for the workflow card.
        
        Args:
            parent: Parent widget (optional).
            workflowGui: Reference to the main WorkflowGui instance (optional).
            posData: Position data object (optional).
        
        Returns:
            The dialog widget created by the subclass setupDialog method.
        """
    
    def setupDialog_cb(self, parent=None, workflowGui=None, posData=None):
        """Setup the dialog with only the required parameters for setupDialog.
        
        This method inspects the setupDialog signature and passes only the
        parameters that are required by the subclass implementation.
        
        Args:
            parent: Parent widget (optional).
            workflowGui: Reference to the main WorkflowGui instance (optional).
            posData: Position data object (optional).
        
        Returns:
            The dialog widget created by the subclass setupDialog method.
        """
        kwargs = {'parent': parent, 'workflowGui': workflowGui, 'posData': posData}
        kwargs_required = inspect.getfullargspec(self.setupDialog).args
        kwargs_to_pass = {k: v for k, v in kwargs.items() if k in kwargs_required}
        dialog = self.setupDialog(**kwargs_to_pass)
        return dialog

    def initializeDialog(self, dialog, parent=None, workflowGui=None, posData=None):
        """Configure a dialog after it has been created and assigned.

        Args:
            dialog: The dialog widget to configure.

            parent: Parent widget (optional).
            workflowGui: Reference to the main WorkflowGui instance (optional).
            posData: Position data object (optional).
        
        """

    def initializeDialog_cb(self, dialog, parent=None, workflowGui=None, posData=None):
        """Call initializeDialog with only the supported arguments."""
        kwargs = {
            'dialog': dialog,
            'parent': parent,
            'workflowGui': workflowGui,
            'posData': posData,
        }
        kwargs_required = inspect.getfullargspec(self.initializeDialog).args
        kwargs_to_pass = {k: v for k, v in kwargs.items() if k in kwargs_required}
        self.initializeDialog(**kwargs_to_pass)

    def renderDialogPreview(self, size=None, scale=(220, 110), parent=None, workflowGui=None, posData=None):
        """Render a preview image of the workflow dialog without displaying it.
        
        Creates a hidden dialog instance (dryrunDialog), captures a screenshot,
        and returns it at the specified scale. This is used to generate preview
        images for the workflow sidebar.
        
        Args:
            size (tuple, optional): Fixed size (width, height) for the dialog. Defaults to None.
            scale (tuple): Target scale (width, height) for the returned image. Defaults to (220, 110).
            parent: Parent widget (optional).
            workflowGui: Reference to the main WorkflowGui instance (optional).
            posData: Position data object (optional).
        
        Returns:
            QPixmap: A screenshot of the dialog scaled to the specified size,
                    or a light gray placeholder if rendering fails.
        """
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

    def updateInputTypes(self, dialog, input_types):
        """Handle changes to connected input types and notify the dialog.
        
        Updates the dialog's input_types attribute and calls the dialog's
        updatedInputTypes() callback if it exists, allowing the dialog to
        respond to changes in the types currently flowing through its inputs.
        
        Args:
            dialog: The dialog widget with input types.
            input_types (dict): Updated concrete input type mapping.
        """
        dialog.input_types = input_types
        if hasattr(dialog, 'updatedInputTypes'):
            dialog.updatedInputTypes()
        
class WorkflowCombineChannelsFunctions(WorkflowBaseFunctions):
    """Workflow card functions for combining and manipulating image channels.
    
    This workflow step allows users to combine multiple image or segmentation
    channels into a single output image through various mathematical operations.
    
    Attributes:
        combineChannelDialog: The dialog class for channel combination.
        title (str): Display title "Combine and manipulate channels".
    """
    
    def __init__(self) -> None:
        """Initialize the combine channels workflow."""
        self.combineChannelDialog = CombineChannelsSetupDialogWorkflow
        self.title = "Combine and manipulate channels"
    
    def dryrunDialog(self, parent=None, workflowGui=None):
        """Create a dry-run dialog instance for preview rendering.
        
        Args:
            parent: Parent widget for the dialog.
            workflowGui: Reference to the main WorkflowGui instance.
        
        Returns:
            CombineChannelsSetupDialogWorkflow: Dialog instance for preview.
        """
        return self.combineChannelDialog(parent=workflowGui)
        
    def setupDialog(self, workflowGui=None):
        """Create the dialog for the workflow card.
        
        Sets up input/output types and connects signals for dynamic updates
        when the number of channels changes or the save mode is toggled.
        
        Args:
            workflowGui: Reference to the main WorkflowGui instance.
        
        Returns:
            CombineChannelsSetupDialogWorkflow: Configured dialog instance.
        """
        dialog = self.combineChannelDialog(parent=workflowGui)
        return dialog

    def initializeDialog(self, dialog, workflowGui=None):
        """Configure the combine channels card after dialog creation."""
        self.setInputs({0: ['img', 'segm']})
        self.setOutputs({0: 'img'})
        dialog.sigNumStepsChanged.connect(self.numStepsChanged)
        dialog.sigOkClicked.connect(self.updatePreview)
        dialog.sigSaveAsSegmCheckboxToggled.connect(self.saveAsSegmCheckboxToggled)
        
    def numStepsChanged(self, num_steps):
        """Handle changes to the number of input channels.
        
        Updates the input definition to accept the new number of channels,
        each capable of being either image or segmentation type.
        
        Args:
            num_steps (int): New number of input channels.
        """
        inputs = {n: ['img', 'segm'] for n in range(num_steps)}
        self.setInputs(inputs)
    
    def saveAsSegmCheckboxToggled(self, save_as_segm):
        """Handle toggling the save output as segmentation checkbox.
        
        Updates the output type based on whether the result should be
        treated as a segmentation or image.
        
        Args:
            save_as_segm (bool): True to save as segmentation, False for image.
        """
        self.setOutputs({0: 'segm' if save_as_segm else 'img'})
        
class WorkflowInputSegmFunctions(WorkflowBaseFunctions):
    """Workflow card functions for selecting segmentation data input.
    
    This workflow step allows users to select which segmentation data
    to use as input for subsequent workflow operations.
    
    Attributes:
        inputDataDialog: The dialog class for data selection.
        title (str): Display title "Select segmentation data".
    """
    
    def __init__(self) -> None:
        """Initialize the segmentation input workflow."""
        self.inputDataDialog = WorkflowInputDataDialog
        self.title = "Select segmentation data"
        
    def dryrunDialog(self, workflowGui=None):
        """Create a dry-run dialog instance for preview rendering.
        
        Args:
            workflowGui: Reference to the main WorkflowGui instance.
        
        Returns:
            WorkflowInputDataDialog: Dialog instance for preview.
        """
        segm_options = workflowGui.segm_channels
        return self.inputDataDialog(segm_options, 'segmentation', parent=workflowGui)
    
    def setupDialog(self, workflowGui=None):
        """Create the dialog for the workflow card.
        
        Args:
            workflowGui: Reference to the main WorkflowGui instance.
        
        Returns:
            WorkflowInputDataDialog: Configured dialog instance.
        """
        segm_options = workflowGui.segm_channels
        dialog = self.inputDataDialog(segm_options, 'segmentation', parent=workflowGui)
        return dialog

    def initializeDialog(self, dialog, workflowGui=None):
        """Configure the segmentation input card after dialog creation."""
        self.setInputs()
        self.setOutputs({0: 'segm'})
        dialog.sigOkClicked.connect(self.updatePreview)
        
class WorkflowInputImgFunctions(WorkflowBaseFunctions):
    """Workflow card functions for selecting image data input.
    
    This workflow step allows users to select which image channel
    to use as input for subsequent workflow operations.
    
    Attributes:
        inputDataDialog: The dialog class for data selection.
        title (str): Display title "Select image data".
    """
    
    def __init__(self) -> None:
        """Initialize the image input workflow."""
        self.inputDataDialog = WorkflowInputDataDialog
        self.title = "Select image data"
        
    def dryrunDialog(self, workflowGui=None):
        """Create a dry-run dialog instance for preview rendering.
        
        Args:
            workflowGui: Reference to the main WorkflowGui instance.
        
        Returns:
            WorkflowInputDataDialog: Dialog instance for preview.
        """
        img_options = workflowGui.img_channels
        return self.inputDataDialog(img_options, 'image', parent=workflowGui)
    
    def setupDialog(self, workflowGui=None):
        """Create the dialog for the workflow card.
        
        Args:
            workflowGui: Reference to the main WorkflowGui instance.
        
        Returns:
            WorkflowInputDataDialog: Configured dialog instance.
        """
        img_options = workflowGui.img_channels
        dialog = self.inputDataDialog(img_options, 'image', parent=workflowGui)
        return dialog

    def initializeDialog(self, dialog, workflowGui=None):
        """Configure the image input card after dialog creation."""
        self.setInputs()
        self.setOutputs({0: 'img'})
        dialog.sigOkClicked.connect(self.updatePreview)
        dialog.sigUpdateTitle.connect(self.updateTitle)
    
class CombineChannelsSetupDialogWorkflow(apps.CombineChannelsSetupDialog):
    """Dialog for combining and manipulating image channels in the workflow.
    
    Extends CombineChannelsSetupDialog to work within the workflow system.
    Emits signals when OK/Cancel buttons are clicked without channel names,
    and dynamically updates inputs/outputs based on the number of channels
    selected and operation parameters.
    
    Signals:
        sigOkClicked: Emitted when the OK button is clicked.
        sigCancelClicked: Emitted when the Cancel button is clicked.
        sigNumStepsChanged: Emitted when the number of channels changes.
        sigSaveAsSegmCheckboxToggled: Emitted when the save mode is toggled.
    """
    sigOkClicked = Signal()
    sigCancelClicked = Signal()
    
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
        """Handle cancel button click by hiding the dialog."""
        self.hide()
        
    def updatedInputTypes(self):
        """Handle when input types change and update the dialog UI accordingly.
        
        Updates checkboxes for binarization and normalization, and automatically
        validates the save as segmentation checkbox based on the updated input types
        and formula preview.
        """
        self.combineChannelsWidget.setBinarizeCheckableAndNorm()
        self.autoCheckSaveAsSegmCheckbox()


class WorkflowInputDataDialog(QBaseDialog):
    """Simple dialog for selecting input data (image or segmentation channel).
    
    Provides a dropdown menu for the user to select from available data options.
    Used as an input node in the workflow to let users specify which data
    should be passed to the next step.
    
    Args:
        selection_options (list): Available options to select from.
        type (str): Type of data being selected ('image' or 'segmentation').
        parent: Parent widget.
    
    Signals:
        sigOkClicked: Emitted when the OK button is clicked.
        sigCancelClicked: Emitted when the Cancel button is clicked.
        sigUpdateTitle: Emitted with the new title when selection changes.
    
    Attributes:
        type (str): The type of data being selected.
        selection_options (list): Available options.
        selected_value (str): Currently selected value.
        selection_widget (QComboBox): Dropdown for selection.
    """
    sigOkClicked = Signal()
    sigCancelClicked = Signal()
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
        """Handle OK button click.
        
        Stores the selected value and emits signals to update the preview
        and card title, then hides the dialog.
        """
        self.selected_value = self.selection_widget.currentText()
        self.sigUpdateTitle.emit(f'{self.type}: {self.selected_value}')
        self.hide()
    
    def cancel_clicked(self):
        """Handle Cancel button click by hiding the dialog."""
        self.hide()
    
    def get_selected_value(self):
        """Get the currently selected value.
        
        Returns:
            str: The text of the currently selected item in the dropdown.
        """
        return self.selected_value

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
        else:
            super().mousePressEvent(event)

class _InputOutputCircle(QLabel):
    """Interactive circular port for workflow card input/output connections.
    
    Represents an input or output port on a workflow card. Users can click on
    a circle to start/complete a connection. The circle displays the port index
    and is colored based on the data type it accepts or provides.
    
    Circle border colors:
        - Blue: Image type
        - Red: Segmentation type
        - Black: Untyped or multiple types
    
    Args:
        index (int): The 0-based port index.
        is_output (bool): True if this is an output port, False for input.
        card (_WorkflowCardZoneWidget): The parent card widget.
        zone (WorkflowZone, optional): Reference to the parent workflow zone.
        type_info: Type constraint(s) - string (single type) or list (multiple types).
    
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
            type_value (str or None): The data type ('img', 'segm', or None).
        
        Returns:
            QColor: The color to use for the circle border.
        """
        if type_value is None:
            return QColor("black")
        if type_value == 'segm':
            return QColor("red")
        elif type_value == 'img':
            return QColor("blue")
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
        """Update the type constraint and refresh the visual appearance.
        
        Args:
            type_info: New type constraint (string or list).
        """
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
    
    Ports are represented as colored circles (blue=image, red=segmentation).
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
        dialog.input_types_accepted (dict): Maps input index to accepted types.
        dialog.input_types (dict): Maps input index to currently connected type.
        output_types (dict): Maps output index to type.
        card_id (int): Unique ID assigned by the zone.
        graphics_proxy: Graphics proxy for positioning in the scene.
    """
    def __init__(self, functions: WorkflowBaseFunctions, posData, 
                 zone=None, workflowGui=None):
        QWidget.__init__(self, parent=None)
        
        self.posData = posData
        self.workflowGui = workflowGui
        self.functions = functions
        self.title = functions.title
        self._drag_start_pos = None
        self.graphics_proxy = None
        self.workflowZone = zone
        self.is_dragging = False
        self.input_circles = []
        self.output_circles = []
        
        self.functions.posData = posData
        self.functions.setInputs = self.setAcceptedInputs
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
        self.functions.initializeDialog_cb(self.dialog,
                                           parent=self,
                                           workflowGui=workflowGui,
                                           posData=self.posData)
        self.functions.runDialog_cb(self.dialog)

    def _getDialogInputTypes(self):
        """Return dialog concrete input types."""
        input_types = getattr(self.dialog, 'input_types', None)
        if isinstance(input_types, dict):
            return input_types
        return {}

    def _getDialogInputTypesAccepted(self):
        """Return dialog accepted input types."""
        input_types_accepted = getattr(self.dialog, 'input_types_accepted', None)
        if isinstance(input_types_accepted, dict):
            return input_types_accepted
        return {}
        
    def updateTitle(self, new_title):
        """Update the card's display title.
        
        Args:
            new_title (str): The new title to display.
        """
        self.title = new_title
        self.titleLabel.setText(new_title)

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
        
        Automatically removes invalid connections and rebuilds visual lines.
        
        Args:
            n: Input count/types specification.
        """
            # Extract count and types from dict
        if isinstance(n, dict):
            type_info = n
            n = max(type_info.keys()) + 1 if type_info else 0
            input_types_accepted = {idx: type_info.get(idx) for idx in range(n)}
        elif n is None:
            # None means no inputs
            n = 0
            input_types_accepted = {}
        else:
            # Simple number, clear accepted types
            input_types_accepted = {i: None for i in range(n)}

        self.dialog.input_types_accepted = input_types_accepted

        while self.inputs_layout.count():
            child = self.inputs_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.input_circles.clear()
        self.inputs_layout.addStretch()
        for i in range(n):
            type_value = input_types_accepted.get(i)
            circle = _InputOutputCircle(i, False, self, self.workflowZone, type_info=type_value)
            self.input_circles.append(circle)
            self.inputs_layout.addWidget(circle)
            self.inputs_layout.addStretch()


        # Validate and remove invalid connections for inputs that no longer exist
        if self.workflowZone is not None:
            card_id = getattr(self, 'card_id', None)
            if card_id is not None:
                # Remove connections to inputs that no longer exist or have type mismatch
                lines_to_remove = [line for line in self.workflowZone.lines
                                   if line[1][0] == card_id and line[1][1] >= n]
                self.workflowZone.removeLineKeys(lines_to_remove)
                # Also remove connections with type mismatches
                self.workflowZone.validateConnectionTypes()
                self.workflowZone.syncCardInputTypes(card_id)
                # Defer rebuild until layout has been processed
                QTimer.singleShot(0, self.workflowZone.rebuildConnectionLines)
                return

        # Keep dialog input state in sync even before the card is attached to a zone.
        self.functions.updateInputTypes(self.dialog, {i: None for i in range(n)})
   
    def setOutputs(self, n=None):
        """Set the number and types of output ports.
        
        Can be called with:
        - A dict: {index: type} to set both count and types
        - An int: Just sets count (types will be None)
        - None: Clears all outputs
        
        Automatically removes invalid connections and rebuilds visual lines.
        Updates downstream cards about type changes.
        
        Args:
            n: Output count/types specification.
        """
        previous_output_types = getattr(self, 'output_types', None)

        if isinstance(n, dict):
            # Extract count and types from dict
            type_info = n
            n = max(type_info.keys()) + 1 if type_info else 0
            output_types = {idx: type_info.get(idx) for idx in range(n)}
        elif n is None:
            # None means no outputs
            n = 0
            output_types = {}
        else:
            # Simple number, clear types
            output_types = {i: None for i in range(n)}

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
            circle = _InputOutputCircle(i, True, self, self.workflowZone, type_info=type_value)
            self.output_circles.append(circle)
            self.outputs_layout.addWidget(circle)
            self.outputs_layout.addStretch()
        
        # Validate and remove invalid connections for outputs that no longer exist
        if self.workflowZone is not None:
            card_id = getattr(self, 'card_id', None)
            if card_id is not None:
                # Remove connections from outputs that no longer exist
                lines_to_remove = [line for line in self.workflowZone.lines
                                   if line[0][0] == card_id and line[0][1] >= n]
                self.workflowZone.removeLineKeys(lines_to_remove)
                self.workflowZone.validateConnectionTypes()
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

    def mousePressEvent(self, event):
        """Handle mouse press events for card interaction.
        
        - Left button: Start dragging the card
        - Middle button: Delete the card
        - Right button: Open the configuration dialog
        
        Args:
            event: The mouse event.
        """
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
        """Update card position and connection lines while dragging.
        
        Args:
            event: The mouse event.
        """
        if self.is_dragging and self._drag_start_pos is not None and self.graphics_proxy is not None:
            delta = event.globalPos() - self._drag_start_pos
            self.graphics_proxy.moveBy(delta.x(), delta.y())
            self._drag_start_pos = event.globalPos()
            # Update connection lines in real-time while dragging
            if self.workflowZone is not None:
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
            if self.workflowZone is not None:
                self.workflowZone.updateConnectionLines()
        super().mouseReleaseEvent(event)

    def _deleteCard(self):
        """Delete this card from the workflow zone."""
        if self.workflowZone is not None:
            self.workflowZone.removeCard(self)
        # clean up dialog and widget
        self.dialog.deleteLater()
        self.dialog = None

    def updatePreview(self):
        """Update the preview image by capturing the current dialog state."""
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

    def __init__(self, parent=None):
        """Initialize the workflow zone.
        
        Args:
            parent: Parent widget.
        """
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
        self.line_to_visual_line_mapping = {}
        self.visual_line_to_line_mapping = {}

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
        card_widget.setZoneOnCircles()
        self.cards[self.unique_card_id] = proxy
        self.unique_card_id += 1
        if len(self.cards) == 1:
            self.placeholder_proxy.hide()

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

        for line_key in line_keys:
            if line_key in self.lines:
                self.lines.remove(line_key)
            self._removeVisualLineByKey(line_key)
            affected_input_card_ids.add(line_key[1][0])

        for card_id in affected_input_card_ids:
            self.syncCardInputTypes(card_id)

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
                self.scene.removeItem(proxy)
                del self.cards[card_id]
                # Remove all connections involving this card.
                lines_to_remove = [line_key for line_key in self.lines
                                   if line_key[0][0] == card_id or line_key[1][0] == card_id]
                self.removeLineKeys(lines_to_remove)
                card_widget.deleteLater()
                if not self.cards:
                    self.placeholder_proxy.show()

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
                            print(f"Input {index} on this card already has a connection. Inputs can only have one connection.")
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
        - If input accepts multiple types (list), output must be in that list
        - If input expects a specific type (string), exact match required
        
        Args:
            output_type (str or None): The output port's type.
            input_type_accepted (str, list, or None): The accepted input type constraint(s).
        
        Returns:
            bool: True if types are compatible.
        """
        if output_type is None or input_type_accepted is None:
            return True
        
        if isinstance(input_type_accepted, (list, tuple)):
            # Input accepts a list of types
            return output_type in input_type_accepted
        else:
            # Input expects a specific type
            return output_type == input_type_accepted

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
        input_type_accepted = input_card._getDialogInputTypesAccepted().get(input_idx)
        
        if not self._areTypesCompatible(output_type, input_type_accepted):
            print(f"Type mismatch: cannot connect output type '{output_type}' to input type '{input_type_accepted}'")
            return
        
        # Store connection with card IDs
        if is_out1:  # card1 is output, card2 is input
            line_key = ((card1_id, idx1), (card2_id, idx2))
        else:  # card1 is input, card2 is output - swap them
            line_key = ((card2_id, idx2), (card1_id, idx1))

        self.lines.append(line_key)
        self.syncCardInputTypes(input_card.card_id)
        self.drawConnection(card1, is_out1, idx1, card2, is_out2, idx2, line_key=line_key)

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

        line_key_to_remove = self.visual_line_to_line_mapping.get(line)

        if line_key_to_remove is not None:
            self.removeLineKeys([line_key_to_remove])
            return

        # Fallback if mapping is out-of-sync.
        self.visual_line_to_line_mapping.pop(line, None)
        self.scene.removeItem(line)
        self.connection_lines.remove(line)

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
            input_type_accepted = end_card._getDialogInputTypesAccepted().get(end_idx)
            
            # Remove connection if types don't match
            if not self._areTypesCompatible(output_type, input_type_accepted):
                lines_to_remove.append(line_data)
        
        # Remove incompatible connections from both data and visual lists
        self.removeLineKeys(lines_to_remove)

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
            selectedExpPaths=None
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
        self.selectedExpPaths = selectedExpPaths

        self.setAcceptDrops(True)
        self._appName = 'Cell-ACDC'
                
        self.initData()
        self.setupUI()
        
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
        """Add a workflow card to the sidebar.
        
        Creates a list item with preview and adds it to the sidebar for
        drag-and-drop into the workflow zone.
        
        Args:
            functions (WorkflowBaseFunctions): The workflow operation to add.
        """
        channels_card = QListWidgetItem()
        channels_card.setData(Qt.UserRole, functions)

        card_widget = _WorkflowCardListWidget(functions, parent=self, posData=self.posData)
        
        channels_card.setSizeHint(card_widget.sizeHint())
        self.sidebar.addItem(channels_card)
        self.sidebar.setItemWidget(channels_card, card_widget)

    def addDroppedCard(self, card: str, x: int, y: int):
        """Handle a card dropped into the WorkflowZone at position (x, y).
        
        Finds the card definition from the sidebar, creates a zone card widget,
        and adds it at the specified position.
        
        Args:
            card (str): The name/title of the card dropped.
            x (int): X coordinate in the zone.
            y (int): Y coordinate in the zone.
        """
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
        card_widget = _WorkflowCardZoneWidget(card_functions, posData=self.posData, zone=self.dropZone, workflowGui=self)
        
        # Add it directly to the drop zone at the specified position
        self.dropZone.addCard(card_widget, x, y)

    def setupUI(self):
        """Build the user interface.
        
        Creates:
        - Sidebar with available workflow cards
        - Main workflow zone for editing
        - Connects signals for drag-and-drop
        """
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
        """Handle window close event.
        
        Emits the sigClosed signal and prevents closing if closeGUI is False.
        
        Args:
            event: The close event.
        """
        if self.closeGUI:
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
        
    # handle key press events for shortcuts
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts.
        
        In debug mode, pressing 'Q' prints the workflow connection data.
        
        Args:
            event: The key event.
        """
        if event.key() == Qt.Key_Q and self.debug:
            print(self.dropZone.lines)
        super().keyPressEvent(event)