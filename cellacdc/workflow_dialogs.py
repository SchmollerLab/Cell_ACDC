from . import apps, qutils, widgets, printl
from .apps import QBaseDialog
from qtpy.QtWidgets import QVBoxLayout, QLabel, QComboBox, QApplication
from qtpy.QtGui import QPixmap, QColor
from qtpy.QtCore import Qt, Signal
import inspect
import traceback

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
    
    def setupDialog(self, parent=None, workflowGui=None, posData=None, logger=print):
        """Configure to return the dialog widget for the workflow card.
        
        Args:
            parent: Parent widget (optional).
            workflowGui: Reference to the main WorkflowGui instance (optional).
            posData: Position data object (optional).
        
        Returns:
            The dialog widget created by the subclass setupDialog method.
        """
        raise NotImplementedError("Subclasses must implement setupDialog to return the dialog widget.")
        

    def initializeDialog(self, dialog, parent=None, workflowGui=None, posData=None):
        """Configure a dialog after it has been created and assigned.

        Args:
            dialog: The dialog widget to configure.

            parent: Parent widget (optional).
            workflowGui: Reference to the main WorkflowGui instance (optional).
            posData: Position data object (optional).
        
        """
        raise NotImplementedError("Subclasses must implement initializeDialog to configure the dialog widget.")
    
    def dryrunDialog(self, parent=None, workflowGui=None, posData=None):
        """Create a dialog instance for preview rendering without displaying it.
        
        This method should return a dialog instance that can be rendered to
        generate a preview image for the workflow card. The dialog will not be
        shown, but should be fully configured to reflect the expected appearance
        based on the current state of the workflow.
        
        Args:
            parent: Parent widget (optional).
            workflowGui: Reference to the main WorkflowGui instance (optional).
            posData: Position data object (optional).
        """
        raise NotImplementedError("Subclasses must implement dryrunDialog to return the dialog widget.")
        
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
    
    def setupDialog_cb(self, parent=None, workflowGui=None, posData=None, logger=print):
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
        
    def setupDialog(self, workflowGui=None, logger=print):
        """Create the dialog for the workflow card.
        
        Sets up input/output types and connects signals for dynamic updates
        when the number of channels changes or the save mode is toggled.
        
        Args:
            workflowGui: Reference to the main WorkflowGui instance.
        
        Returns:
            CombineChannelsSetupDialogWorkflow: Configured dialog instance.
        """
        dialog = self.combineChannelDialog(parent=workflowGui, logger=logger)
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
    
    def setupDialog(self, workflowGui=None, logger=print):
        """Create the dialog for the workflow card.
        
        Args:
            workflowGui: Reference to the main WorkflowGui instance.
        
        Returns:
            WorkflowInputDataDialog: Configured dialog instance.
        """
        segm_options = workflowGui.segm_channels
        dialog = self.inputDataDialog(segm_options, 'segmentation', parent=workflowGui, logger=logger)
        return dialog

    def initializeDialog(self, dialog, workflowGui=None):
        """Configure the segmentation input card after dialog creation."""
        self.setInputs()
        self.setOutputs({0: 'segm'})
        dialog.sigOkClicked.connect(self.updatePreview)
        dialog.sigUpdateTitle.connect(self.updateTitle)
        dialog.sigSelectionInvalid.connect(self.notifySelectionInvalid)
        if hasattr(self, 'notifySelectionValid'):
            dialog.sigSelectionValid.connect(self.notifySelectionValid)
        
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
    
    def setupDialog(self, workflowGui=None, logger=print):
        """Create the dialog for the workflow card.
        
        Args:
            workflowGui: Reference to the main WorkflowGui instance.
        
        Returns:
            WorkflowInputDataDialog: Configured dialog instance.
        """
        img_options = workflowGui.img_channels
        dialog = self.inputDataDialog(img_options, 'image', parent=workflowGui,
                                      logger=logger)
        return dialog

    def initializeDialog(self, dialog, workflowGui=None):
        """Configure the image input card after dialog creation."""
        self.setInputs()
        self.setOutputs({0: 'img'})
        dialog.sigOkClicked.connect(self.updatePreview)
        dialog.sigUpdateTitle.connect(self.updateTitle)
        dialog.sigSelectionInvalid.connect(self.notifySelectionInvalid)
        if hasattr(self, 'notifySelectionValid'):
            dialog.sigSelectionValid.connect(self.notifySelectionValid)



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
    
    def __init__(self, parent=None, logger=print):
        super().__init__(channel_names=None, parent=parent, hideOnClosing=True)
        self.logger = logger
        
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

    def saveContent(self, path):
        """Save the current dialog settings to a file."""
        ext = '.json'
        if not path.endswith(ext):
            path += ext
        self.saveRecipe(filepath=path)
        
    def loadContent(self, path):
        """Load dialog settings from a file and update the UI."""
        ext = '.json'
        if not path.endswith(ext):
            path += ext
        self.show()
        self.loadRecipe(filepath=path)
        self.ok_cb()  # To emit signals and update the workflow card state after loading
        self.hide()
        
class PreProcessSetupDialogWorkflow(apps.PreProcessRecipeDialog):
    def __init__(self, parent=None):
        super().__init__(parent=parent, hideOnClosing=True)
        
        self.mainLayout.addSpacing(20)

        qutils.hide_and_delete_layout(self.buttonsLayout)
        buttonsLayout = widgets.CancelOkButtonsLayout()
        self.buttonsLayout = buttonsLayout
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.okButton.clicked.connect(self.sigOkClicked.emit)
        buttonsLayout.cancelButton.clicked.connect(self.sigCancelClicked.emit)
        buttonsLayout.cancelButton.clicked.connect(self.cancelButtonClicked)
        
        self.mainLayout.addLayout(buttonsLayout)

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
    sigSelectionInvalid = Signal(str)
    sigSelectionValid = Signal()
    
    def __init__(self, selection_options, type, parent=None, logger=print):
        super().__init__(parent=parent)
        self.setWindowTitle(f'Select input {type}')
        
        self.type = type
        self.selection_options = selection_options or []
        self.logger = logger
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
        if self.selected_value in self.selection_options:
            self.sigSelectionValid.emit()
        else:
            self.sigSelectionInvalid.emit(self.selected_value)
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
    
    def saveContent(self, path):
        """Save the current selection to a file"""
        ext = '.txt'
        if not path.endswith(ext):
            path += ext
        content = self.get_selected_value()
        with open(path, 'w') as f:
            f.write(content)
            
    def new_image_loaded(self):
        """Update the available options when new image data is loaded.

        Repopulates the combo box with the latest channels from the parent
        WorkflowGui.  If the previously confirmed selection is no longer
        available it is added back with red foreground text and
        ``sigSelectionInvalid`` is emitted so the main GUI can alert the user.
        """
        workflowGui = self.parent()
        if workflowGui is None:
            return

        if self.type == 'image':
            new_options = list(getattr(workflowGui, 'img_channels', None) or [])
        elif self.type == 'segmentation':
            new_options = list(getattr(workflowGui, 'segm_channels', None) or [])
        else:
            return

        prev_selection = self.selected_value

        self.selection_widget.blockSignals(True)
        self.selection_widget.clear()
        self.selection_options = new_options
        self.selection_widget.addItems(self.selection_options)
        self.selection_widget.blockSignals(False)

        if prev_selection is not None:
            if prev_selection in self.selection_options:
                self.selection_widget.setCurrentIndex(
                    self.selection_options.index(prev_selection)
                )
            else:
                # Previous value no longer available – add it back in red
                self.selection_widget.addItem(prev_selection)
                invalid_idx = self.selection_widget.count() - 1
                model = self.selection_widget.model()
                item = model.item(invalid_idx)
                if item is not None:
                    item.setForeground(QColor('red'))
                self.selection_widget.setCurrentIndex(invalid_idx)
                self.sigSelectionInvalid.emit(prev_selection)

    def loadContent(self, path):
        """Load the selection from a file and update the dialog."""
        ext = '.txt'
        self.show()
        if not path.endswith(ext):
            path += ext
        try:
            with open(path, 'r') as f:
                content = f.read().strip()
            if content in self.selection_options:
                index = self.selection_widget.findText(content)
                self.selection_widget.setCurrentIndex(index)
                self.selected_value = content
                self.sigUpdateTitle.emit(f'{self.type}: {self.selected_value}')
            else:
                self.logger(f"Loaded value '{content}' not in selection options.")
        except Exception as e:
            self.logger(f"Failed to load content from {path}: {e}")
            traceback.print_exc()
        self.ok_clicked()  # To emit signals and update the workflow card state after loading
        self.hide()