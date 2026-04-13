import os
import traceback

from . import apps, qutils, widgets, printl, config, html_utils
from . import myutils, prompts, load
from .apps import QBaseDialog
from qtpy.QtWidgets import (
    QVBoxLayout, QLabel, QComboBox, QApplication, QHBoxLayout, QPushButton,
    QSpinBox, QMessageBox, QInputDialog
)
from qtpy.QtGui import QPixmap, QColor
from qtpy.QtCore import Qt, Signal
import inspect
import json
import copy
import io
import re
import pandas as pd
from .workflow_typing import (
    WfImageDC, WfSegmDC, WfMetricsDC, workflow_type_name,
    make_workflow_data_class, is_workflow_data_class
)
from qtpy.compat import getopenfilename

class WorkflowBaseFunctions():
    """Base class for workflow card widgets.
    
    This class provides a template for creating workflow cards that can be added
    to the workflow GUI. Subclasses should implement dryrunDialog() and 
    setupDialog() methods. This is the "functions" side of things, the dialogs
    themselves are defined separately.
    
    Attributes:
        title (str): Display title for the workflow card.
        posData: Position data for the workflow.
        all_inputs_uniform (bool): If True, set all accepted inputs are of the same shape.
            If False, inputs can be of different shapes.
    
    Example:
        class MyWorkflowCard(WorkflowBaseFunctions):
            def __init__(self):
                self.title = "My Workflow Step"
            
            def dryrunDialog(self, parent=None):
                return MyDialog(parent=parent)
            
            def setupDialog(self, parent=None):
                self.setAcceptedInputs({0: WorkflowImageDataClass()})
                self.setOutputs({0: WorkflowImageDataClass()})
                return MyDialog(parent=parent)
    """
    def __init__(self):
        """Initialize the base workflow functions class."""
    
    def setupDialog(self, parent=None, workflowGui=None, posData=None, logger=printl):
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
    
    def setupDialog_cb(self, parent=None, workflowGui=None, posData=None, logger=printl, preInitWorkflow_res=None):
        """Setup the dialog with only the required parameters for setupDialog.
        
        This method inspects the setupDialog signature and passes only the
        parameters that are required by the subclass implementation.
        
        Args:
            parent: Parent widget (optional).
            workflowGui: Reference to the main WorkflowGui instance (optional).
            posData: Position data object (optional).
        
        Returns:
        """
        kwargs = {'parent': parent, 'workflowGui': workflowGui, 'posData': posData, 'preInitWorkflow_res': preInitWorkflow_res, 'logger': logger}
        kwargs_required = inspect.getfullargspec(self.setupDialog).args
        kwargs_to_pass = {k: v for k, v in kwargs.items() if k in kwargs_required}
        dialog = self.setupDialog(**kwargs_to_pass)
        return dialog

    def initializeDialog_cb(self, dialog, parent=None, workflowGui=None, posData=None, preInitWorkflow_res=None, logger=printl):
        """Call initializeDialog with only the supported arguments."""
        kwargs = {
            'dialog': dialog,
            'parent': parent,
            'workflowGui': workflowGui,
            'posData': posData,
            'preInitWorkflow_res': preInitWorkflow_res,
            'logger': logger
        }
        kwargs_required = inspect.getfullargspec(self.initializeDialog).args
        kwargs_to_pass = {k: v for k, v in kwargs.items() if k in kwargs_required}
        self.initializeDialog(**kwargs_to_pass)
        # check for the common signals sigSetOutputs
        if hasattr(dialog, 'sigSetOutputs'):
            dialog.sigSetOutputs.connect(self.setOutputs)

    def renderDialogPreview(self, size=None, scale=(220, 110), parent=None, workflowGui=None, posData=None):
        """Render a preview image of the workflow dialog without displaying it.
        
        Creates a hidden dialog instance (dryrunDialog), captures a screenshot,
        and returns it at the specified scale. This is used to generate preview
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
            printl(f"Failed to render dialog preview: {e}")
            pix = QPixmap(*scale)
            pix.fill(Qt.lightGray)
            return pix

    def updateInputTypes(self, dialog, input_types):
        """Handle changes to connected input types and notify the dialog.
        
        Updates the dialog's curr_input_types attribute and calls the dialog's
        updatedInputTypes() callback if it exists, allowing the dialog to
        respond to changes in the types currently flowing through its inputs.
        Re-applies current outputs through setOutputs() so cards that inherit
        metadata from input 0 can propagate SizeT/SizeZ changes downstream.
        
        Args:
            dialog: The dialog widget with input types.
            input_types (dict): Updated concrete input type mapping.
        """
        dialog.curr_input_types = input_types # currently connected input types
        if hasattr(dialog, 'updatedInputTypes'):
            dialog.updatedInputTypes()
        self._applyInputUniformityRules(dialog, input_types)
        # Trigger output re-evaluation so inherited SizeT/SizeZ is propagated
        # even for cards that do not explicitly react to input-type changes.
        if hasattr(self, 'getCurrentOutputs') and callable(self.getCurrentOutputs):
            current_outputs = self.getCurrentOutputs()
            if current_outputs is not None:
                self.setOutputs(copy.deepcopy(current_outputs))
            
    def _applyInputUniformityRules(self, dialog, input_types):
        # check for empty input_types
        if not input_types:
            return
        
        curr_accepted_inputs = dialog.curr_input_types_accepted
        if hasattr(self, 'all_inputs_uniform') and self.all_inputs_uniform:
            first_input = input_types[0] if 0 in input_types else None
            if first_input is not None:
                curr_accepted_inputs ={
                    idx: self._setInputMetadataUniform(
                        curr_accepted_inputs[idx],
                        size_t=first_input.SizeT,
                        size_z=first_input.SizeZ,
                        size_y=first_input.SizeY,
                        size_x=first_input.SizeX,
                    )
                    for idx in curr_accepted_inputs.keys()
                }
                
        if hasattr(self, 'all_time_inputs_uniform') and self.all_time_inputs_uniform:
            # if all time inputs are uniform, then all inputs are uniform
            first_input = input_types[0] if 0 in input_types else None
            if first_input is not None:
                curr_accepted_inputs ={
                    idx: self._setInputMetadataUniform(
                        curr_accepted_inputs[idx],
                        size_t=first_input.SizeT,
                        size_y=first_input.SizeY,
                        size_x=first_input.SizeX,
                    )
                    for idx in curr_accepted_inputs.keys()
                }
                
        if hasattr(self, 'if_0_3d_others_3d') and self.if_0_3d_others_3d:
            # if segm input has 3D, img has to be 3D. If segm is 2D, img can be 2D or 3D
            # overrides all_inputs_uniform for the first two inputs
            first_input = input_types[0] if 0 in input_types else None
            if (first_input is not None 
                and first_input.SizeZ is not None 
                and first_input.SizeZ > 1):
                for idx in list(curr_accepted_inputs.keys())[1:]:
                    accepted_input = curr_accepted_inputs[idx]
                    accepted_size_z = None
                    if isinstance(accepted_input, list):
                        for accepted_type in accepted_input:
                            accepted_size_z = getattr(accepted_type, 'SizeZ', None)
                            if accepted_size_z is not None:
                                break
                    else:
                        accepted_size_z = getattr(accepted_input, 'SizeZ', None)

                    curr_accepted_inputs[idx] = self._setInputMetadataUniform(
                        accepted_input,
                        size_z=accepted_size_z,
                        size_y=first_input.SizeY,
                        size_x=first_input.SizeX,
                    )
                    
        # check for changes compared to original accepted inputs.
        for idx in curr_accepted_inputs:
            if (idx not in dialog.curr_input_types_accepted 
                or dialog.curr_input_types_accepted[idx] != curr_accepted_inputs[idx]):
                self.setAcceptedInputs(curr_accepted_inputs)
                break
            
    def _setInputMetadataUniform(self, input_type_ls,
                                    size_t=None, size_z=None,
                                    size_y=None, size_x=None):
        """Set input metadata to be uniform for all inputs of a specific slot (input_ls).
        """
        was_list = isinstance(input_type_ls, list)
        if not was_list:
            input_type_ls = [input_type_ls]
        
        new_input_type_ls = []
        for input_type in input_type_ls:
            input_type_name = workflow_type_name(input_type)
            new_input_type_ls.append(make_workflow_data_class(
                input_type_name,
                SizeT=size_t, SizeZ=size_z,
                SizeY=size_y, SizeX=size_x,
            ))

        if was_list:
            return new_input_type_ls
        return new_input_type_ls[0]
        
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
        self.card_key = 'combine_channels'
        self.title = "Combine and manipulate channels"
        # define input uniformity rules
        self.all_inputs_uniform = True
    
    def dryrunDialog(self, parent=None, workflowGui=None):
        """Create a dry-run dialog instance for preview rendering.
        
        Args:
            parent: Parent widget for the dialog.
            workflowGui: Reference to the main WorkflowGui instance.
        
        Returns:
            CombineChannelsSetupDialogWorkflow: Dialog instance for preview.
        """
        return self.combineChannelDialog(parent=workflowGui)
        
    def setupDialog(self, workflowGui=None, logger=printl, posData=None):
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
        self.setAcceptedInputs({0: [WfImageDC(), WfSegmDC()]})
        self.setOutputs({0: WfImageDC()})
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
        inputs = {
            n: [WfImageDC(), WfSegmDC()]
            for n in range(num_steps)
        }
        self.setAcceptedInputs(inputs)
    
    def saveAsSegmCheckboxToggled(self, save_as_segm):
        """Handle toggling the save output as segmentation checkbox.
        
        Updates the output type based on whether the result should be
        treated as a segmentation or image.
        
        Args:
            save_as_segm (bool): True to save as segmentation, False for image.
        """
        out_data_class = WfSegmDC() if save_as_segm else WfImageDC()
        self.setOutputs({0: out_data_class})
        
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
        self.card_key = 'input_segmentation'
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
    
    def setupDialog(self, workflowGui=None, logger=printl, posData=None):
        """Create the dialog for the workflow card.
        
        Args:
            workflowGui: Reference to the main WorkflowGui instance.
        
        Returns:
            WorkflowInputDataDialog: Configured dialog instance.
        """
        segm_options = workflowGui.segm_channels
        dialog = self.inputDataDialog(segm_options, 'segmentation', 
                                      parent=workflowGui, 
                                      logger=logger, 
                                      posData=posData)
        return dialog

    def initializeDialog(self, dialog, workflowGui=None):
        """Configure the segmentation input card after dialog creation."""
        self.setAcceptedInputs()
        self.setOutputs({0: WfSegmDC()})
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
        self.card_key = 'input_image'
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
    
    def setupDialog(self, workflowGui=None, logger=printl, posData=None):
        """Create the dialog for the workflow card.
        
        Args:
            workflowGui: Reference to the main WorkflowGui instance.
        
        Returns:
            WorkflowInputDataDialog: Configured dialog instance.
        """
        img_options = workflowGui.img_channels
        dialog = self.inputDataDialog(img_options, 'image', parent=workflowGui,
                                      logger=logger, posData=posData)
        return dialog

    def initializeDialog(self, dialog, workflowGui=None):
        """Configure the image input card after dialog creation."""
        self.setAcceptedInputs()
        self.setOutputs({0: WfImageDC()})
        dialog.sigOkClicked.connect(self.updatePreview)
        dialog.sigUpdateTitle.connect(self.updateTitle)
        dialog.sigSelectionInvalid.connect(self.notifySelectionInvalid)
        if hasattr(self, 'notifySelectionValid'):
            dialog.sigSelectionValid.connect(self.notifySelectionValid)

class WorkflowPreProcessFunctions(WorkflowBaseFunctions):
    """Workflow card functions for pre-processing image data."""

    def __init__(self) -> None:
        self.preprocessDialog = PreProcessSetupDialogWorkflow
        self.card_key = 'preprocess_image'
        self.title = "Pre-process image"

    def dryrunDialog(self, parent=None, workflowGui=None):
        return self.preprocessDialog(parent=workflowGui)

    def setupDialog(self, workflowGui=None, logger=printl):
        dialog = self.preprocessDialog(parent=workflowGui, logger=logger)
        return dialog

    def initializeDialog(self, dialog, workflowGui=None):
        self.setAcceptedInputs({0: WfImageDC()})
        self.setOutputs({0: WfImageDC()})
        dialog.sigOkClicked.connect(self.updatePreview)


class WorkflowZStackManipulationFunctions(WorkflowBaseFunctions):
    """Workflow card functions for manipulating z-stack dimensionality."""

    def __init__(self) -> None:
        self.zStackDialog = ZStackManipulationDialogWorkflow
        self.card_key = 'zstack_manipulation'
        self.title = 'Z-stack manipulation'

    def dryrunDialog(self, workflowGui=None):
        return self.zStackDialog(parent=workflowGui)

    def setupDialog(self, workflowGui=None, logger=printl, posData=None):
        dialog = self.zStackDialog(parent=workflowGui, logger=logger, posData=posData)
        return dialog

    def initializeDialog(self, dialog, workflowGui=None):
        self.setAcceptedInputs({0: [WfImageDC(), WfSegmDC()]})
        self.setOutputs({0: WfImageDC()})
        dialog.sigOkClicked.connect(self.updatePreview)
        dialog.sigSelectionInvalid.connect(self.notifySelectionInvalid)
        if hasattr(self, 'notifySelectionValid'):
            dialog.sigSelectionValid.connect(self.notifySelectionValid)


class WorkflowCropImagesFunctions(WorkflowBaseFunctions):
    """Workflow card functions for copying data-prep crop settings."""

    def __init__(self) -> None:
        self.cropDialog = CropImagesDialogWorkflow
        self.card_key = 'crop_images'
        self.title = 'Crop images'

    def dryrunDialog(self, workflowGui=None):
        return self.cropDialog(parent=workflowGui)

    def setupDialog(self, workflowGui=None, logger=printl, posData=None):
        dialog = self.cropDialog(parent=workflowGui, logger=logger, posData=posData)
        return dialog

    def initializeDialog(self, dialog, workflowGui=None):
        self.setAcceptedInputs({0: [WfImageDC(), WfSegmDC()]})
        self.setOutputs({0: WfImageDC()})
        dialog.sigOkClicked.connect(self.updatePreview)
        dialog.sigSelectionInvalid.connect(self.notifySelectionInvalid)
        if hasattr(self, 'notifySelectionValid'):
            dialog.sigSelectionValid.connect(self.notifySelectionValid)

class WorkflowPostProcessSegmFunctions(WorkflowBaseFunctions):
    """Workflow card functions for post-processing segmentation masks."""

    def __init__(self) -> None:
        self.postProcessDialog = PostProcessSegmDialogWorkflow
        self.card_key = 'postprocess_segmentation'
        self.title = "Post-process segmentation"
        self.all_inputs_uniform = True
        self.if_0_3d_others_3d = True
        # define input uniformity rules

    def dryrunDialog(self, workflowGui=None, posData=None):
        return self.postProcessDialog(posData=posData, parent=workflowGui)

    def setupDialog(self, workflowGui=None, posData=None, logger=printl):
        dialog = self.postProcessDialog(
            posData=posData,
            parent=workflowGui,
            logger=logger,
        )
        return dialog

    def initializeDialog(self, dialog, workflowGui=None):
        self.setAcceptedInputs({0: WfSegmDC(), 1: WfMetricsDC()})
        self.setOutputs({0: WfSegmDC()})
        dialog.sigInputsChanged.connect(self.setAcceptedInputs)
        dialog.sigOkClicked.connect(self.updatePreview)

class WorkflowSetMeasurementsFunctions(WorkflowBaseFunctions):
    """Workflow card functions for selecting measurements to extract."""

    def __init__(self) -> None:
        self.measurementsDialog = SetMeasurementsDialogWorkflow
        self.card_key = 'set_measurements'
        self.title = 'Set measurements'

    def dryrunDialog(self, workflowGui=None, posData=None):
        return self.measurementsDialog(posData=posData, parent=workflowGui)

    def setupDialog(self, workflowGui=None, posData=None, logger=printl):
        dialog = self.measurementsDialog(
            posData=posData,
            parent=workflowGui,
            logger=logger,
        )
        return dialog

    def initializeDialog(self, dialog, workflowGui=None):
        self.setAcceptedInputs({0: WfSegmDC(), 1: WfImageDC()})
        self.setOutputs({0: WfMetricsDC(setMetrics=dialog.selectedMetrics())})
        dialog.sigInputsChanged.connect(self.setAcceptedInputs)
        dialog.sigSelectedMetricsChanged.connect(
            lambda: self.selectedMetricsChanged(dialog)
        )
        dialog.sigOkClicked.connect(self.updatePreview)

    def selectedMetricsChanged(self, dialog):
        self.setOutputs({0: WfMetricsDC(setMetrics=dialog.selectedMetrics())})
        
class SegmentFunctions(WorkflowBaseFunctions):
    """Workflow card functions for segmenting images.
    
    This workflow step allows users to segment images using a selected
    segmentation model. The workflow consists of two steps:
    1. Select a segmentation model (shown when card is dropped)
    2. Configure model parameters
    
    Attributes:
        title (str): Display title "Segment image".
        card_key (str): Unique key 'segment'.
    """
    
    def __init__(self):
        super().__init__()
        self.card_key = 'segment'
        self.title = "Segment image"
        
    def preInitWorkflowDialog(self, workflowGui=None):
        """Create and return the initial dialog shown when the card is dropped.
        
        This method is called before the main setupDialog and allows for a
        dedicated first-run experience. In this case, it shows the model
        selection dialog immediately when the card is added to the workflow,
        and crucially BEFORE the main params dialog is initialized.
        This instance should be reusable and not destroyed after selection.
        
        Args:
            workflowGui: Reference to the main WorkflowGui instance.
            _onDialogCancelled: Callback function to handle dialog cancellation.
        """
        window = apps.QDialogSelectModel(parent=workflowGui, addSkipSegmButton=False)
        window.exec_()
        selectedModel = window.selectedModel if hasattr(window, 'selectedModel') else None
        window.deleteLater()  # Clean up the dialog instance after use
        return selectedModel
        
        
    def dryrunDialog(self, parent=None, workflowGui=None, posData=None):
        """Create a dry-run dialog instance for preview rendering.
        
        Args:
            parent: Parent widget for the dialog.
            workflowGui: Reference to the main WorkflowGui instance.
            posData: Position data object.
        
        Returns:
            SelectSegmDialogWorkflow: Dialog instance for preview.
        """
        return apps.QDialogSelectModel(parent=workflowGui, addSkipSegmButton=False)
    
    def setupDialog(self, workflowGui=None, posData=None, logger=printl, preInitWorkflow_res=None):
        """Create the dialog for the workflow card.
        
        Creates a ModelParamsDialogWorkflow that will be configured
        with the selected model's parameters.
        
        Args:
            workflowGui: Reference to the main WorkflowGui instance.
            posData: Position data object.
            logger: Logging function.
        
        Returns:
            ModelParamsDialogWorkflow: Configured dialog instance.
        """
        selected_model = preInitWorkflow_res
        if selected_model is None:
            # Card restoration can instantiate dialogs without pre-init payload.
            # Fall back to a deterministic built-in model.
            selected_model = 'thresholding'

        if selected_model == 'Automatic thresholding':
            selected_model = 'thresholding'
            
        if selected_model == 'thresholding':
            dialog = ThresholdSegmDialogWorkflow(parent=workflowGui, logger=logger)
            dialog.logger = logger
            return dialog

        acdc_segment = myutils.import_segment_module(selected_model)
        init_params, segment_params = myutils.getModelArgSpec(acdc_segment)
        try:
            help_url = acdc_segment.url_help()
        except Exception:
            help_url = None

        dialog = SegmModelParamsDialogWorkflow(
            init_params,
            segment_params,
            selected_model,
            parent=workflowGui,
            url=help_url,
            initLastParams=True,
            # posData=posData, # for sam, need to double check if 3d, timelapse or additional img input or 
            # segmFileEndnames=existing_segm_endnames,
            # df_metadata=posData.metadata_df,
            add_additional_segm_params=True,
            addPreProcessParams=False,
            addPostProcessParams=False,
            hideOnClosing=True,
            useInput2AsSecondChannelToggle=True,
        )

        dialog.logger = logger
        return dialog
    
    def initializeDialog(self, dialog, workflowGui=None):
        """Configure the segmentation card after dialog creation.
        
        Sets up input/output types and connects signals.
        
        Args:
            dialog: The ModelParamsDialogWorkflow to configure.
            workflowGui: Reference to the main WorkflowGui instance.
        """
        self.setAcceptedInputs({0: WfImageDC()})
        self.setOutputs({0: WfSegmDC()})
        dialog.sigOkClicked.connect(self.updatePreview)
        if hasattr(dialog, 'sigInputsChanged'):
            dialog.sigInputsChanged.connect(self.setAcceptedInputs)

    def getPreInitWorkflowRes(self, dialog=None):
        if dialog is None:
            return None
        return getattr(dialog, 'model_name', None)


class TrackingFunctions(WorkflowBaseFunctions):
    """Workflow card functions for tracking segmentation masks."""

    def __init__(self):
        super().__init__()
        self.card_key = 'track'
        self.title = 'Track segmentation'

    def _getSelectableTrackers(self):
        trackers = myutils.get_list_of_trackers()
        if trackers:
            return trackers
        return ['No compatible trackers available']

    def preInitWorkflowDialog(self, workflowGui=None):
        trackers = myutils.get_list_of_trackers()
        if not trackers:
            return None

        dialog = apps.QtSelectItems(
            'Select tracker',
            trackers,
            'Select one of the following trackers',
            parent=workflowGui,
            showMultipleSelection=False
        )
        
        dialog.exec_()
        if dialog.cancel or not dialog.selectedItemsText:
            return None
        return dialog.selectedItemsText[0]

    def dryrunDialog(self, parent=None, workflowGui=None, posData=None):
        return apps.QtSelectItems(
            'Select tracker',
            self._getSelectableTrackers(),
            'Select one of the following trackers',
            parent=workflowGui,
        )

    def setupDialog(
            self, workflowGui=None, posData=None, logger=printl,
            preInitWorkflow_res=None
        ):
        selected_tracker = preInitWorkflow_res
        if selected_tracker is None:
            trackers = myutils.get_list_of_trackers()
            if not trackers:
                raise ValueError('No workflow-compatible trackers available.')
            selected_tracker = trackers[0]

        if selected_tracker == 'BayesianTracker':
            dialog = BayesianTrackerDialogWorkflow(
                parent=workflowGui, posData=posData, logger=logger
            )
            dialog.logger = logger
            return dialog

        if selected_tracker == 'CellACDC':
            dialog = CellACDCTrackerDialogWorkflow(
                parent=workflowGui, logger=logger
            )
            dialog.logger = logger
            return dialog

        if selected_tracker == 'delta':
            dialog = DeltaTrackerDialogWorkflow(
                parent=workflowGui, posData=posData, logger=logger
            )
            dialog.logger = logger
            return dialog

        tracker_module = myutils.import_tracker_module(selected_tracker)
        init_params, track_params = myutils.getTrackerArgSpec(
            tracker_module, realTime=False
        )
        requires_input_image = myutils.isIntensityImgRequiredForTracker(
            tracker_module
        )
        try:
            help_url = tracker_module.url_help()
        except Exception:
            help_url = None

        try:
            df_metadata = posData.metadata_df
        except Exception:
            df_metadata = None

        dialog = TrackingModelParamsDialogWorkflow(
            init_params,
            track_params,
            selected_tracker,
            parent=workflowGui,
            url=help_url,
            is_tracker=True,
            initLastParams=True,
            df_metadata=df_metadata,
            posData=posData,
            addPreProcessParams=False,
            addPostProcessParams=False,
            hideOnClosing=True,
            logger=logger,
            requiresInputImage=requires_input_image,
        )
        dialog.logger = logger
        return dialog

    def initializeDialog(self, dialog, workflowGui=None):
        inputs = {0: WfSegmDC()}
        if getattr(dialog, 'requiresInputImage', False):
            inputs[1] = WfImageDC()
        self.setAcceptedInputs(inputs)
        self.setOutputs({0: WfSegmDC()})
        dialog.sigOkClicked.connect(self.updatePreview)

    def getPreInitWorkflowRes(self, dialog=None):
        if dialog is None:
            return None
        return getattr(dialog, 'model_name', None)

class SegmModelParamsDialogWorkflow(apps.QDialogModelParams):
    sigOkClicked = Signal()
    sigCancelClicked = Signal()
    sigInputsChanged = Signal(dict)

    def __init__(self, *args, logger=printl, **kwargs):
        kwargs.setdefault('hideOnClosing', True)
        super().__init__(*args, **kwargs)
        self.logger = logger

    def ok_cb(self, checked=False):
        super().ok_cb(checked)
        if self.cancel or self.isVisible():
            return
        self.sigOkClicked.emit()

    def cancel_cb(self, checked=False):
        self.cancel = True
        self.sigCancelClicked.emit()
        if self.hideOnClosing:
            self.hide()
            self._exit_local_loop_if_running()
            return
        super().cancel_cb(checked)

    def closeEvent(self, event):
        self.cancel = True
        self.sigCancelClicked.emit()
        super().closeEvent(event)

    def getContent(self):
        config_pars = self.getConfigPars(create_new=True)
        if hasattr(self, 'reduceMemUsageToggle'):
            section = f'{self.model_name}.additional_segm_params'
            option = self.reduceMemUsageToggle.label
            config_pars[section] = {
                option: str(self.reduceMemUsageToggle.isChecked())
            }

        ini_buffer = io.StringIO()
        config_pars.write(ini_buffer)
        return {
            'config_ini': ini_buffer.getvalue(),
        }

    def setContent(self, content):
        if not content:
            return

        config_ini = content.get('config_ini')
        if config_ini:
            cp = config.ConfigParser()
            cp.read_string(config_ini)
            self.loadPreprocRecipe(configPars=cp)
            self.loadLastSelection(
                f'{self.model_name}.init', self.init_argsWidgets, configPars=cp
            )
            self.loadLastSelection(
                f'{self.model_name}.segment', self.argsWidgets, configPars=cp
            )
            if self.extraArgsWidgets:
                self.loadLastSelection(
                    f'{self.model_name}.extra',
                    self.extraArgsWidgets,
                    configPars=cp,
                )
            self.loadLastSelectionPostProcess(configPars=cp)

            if hasattr(self, 'reduceMemUsageToggle'):
                section = f'{self.model_name}.additional_segm_params'
                option = self.reduceMemUsageToggle.label
                if cp.has_option(section, option):
                    self.reduceMemUsageToggle.setChecked(
                        cp.getboolean(section, option)
                    )

    def saveContent(self, path):
        ext = '.ini'
        if not path.endswith(ext):
            path += ext

        config_pars = self.getConfigPars(create_new=True)
        if hasattr(self, 'reduceMemUsageToggle'):
            section = f'{self.model_name}.additional_segm_params'
            option = self.reduceMemUsageToggle.label
            config_pars[section] = {
                option: str(self.reduceMemUsageToggle.isChecked())
            }

        with open(path, 'w') as f:
            config_pars.write(f)

    def loadContent(self, path):
        ext = '.ini'
        if not path.endswith(ext):
            path += ext

        self.show()
        try:
            cp = config.ConfigParser()
            cp.read(path)

            self.loadPreprocRecipe(configPars=cp)
            self.loadLastSelection(
                f'{self.model_name}.init', self.init_argsWidgets, configPars=cp
            )
            self.loadLastSelection(
                f'{self.model_name}.segment', self.argsWidgets, configPars=cp
            )
            if self.extraArgsWidgets:
                self.loadLastSelection(
                    f'{self.model_name}.extra',
                    self.extraArgsWidgets,
                    configPars=cp,
                )
            self.loadLastSelectionPostProcess(configPars=cp)

            if hasattr(self, 'reduceMemUsageToggle'):
                section = f'{self.model_name}.additional_segm_params'
                option = self.reduceMemUsageToggle.label
                if cp.has_option(section, option):
                    self.reduceMemUsageToggle.setChecked(
                        cp.getboolean(section, option)
                    )
        except Exception as e:
            self.logger.info(f'Failed to load content from {path}: {e}')
            traceback.print_exc()
            self.hide()
            return

        self.ok_cb()
        self.hide()
        
    def passInput2AsSecondChannelToggled_cb(self, checked):
        inputs = {0: WfImageDC()}
        if checked:
            inputs[1] = WfImageDC()

        self.sigInputsChanged.emit(inputs)


class TrackingModelParamsDialogWorkflow(SegmModelParamsDialogWorkflow):
    def __init__(self, *args, requiresInputImage=False, **kwargs):
        self.requiresInputImage = requiresInputImage
        super().__init__(*args, **kwargs)


class _BaseTrackerDialogWorkflow:
    sigOkClicked = Signal()
    sigCancelClicked = Signal()

    def _tryParentSaveContent(self, path):
        parent_save = getattr(super(), 'saveContent', None)
        if not callable(parent_save):
            return False
        try:
            parent_save(path)
            return True
        except TypeError:
            parent_save()
            return True

    def _tryParentLoadContent(self, path):
        parent_load = getattr(super(), 'loadContent', None)
        if not callable(parent_load):
            return False
        try:
            parent_load(path)
            return True
        except TypeError:
            parent_load()
            return True

    def _emitCancelledAndHide(self):
        self.cancel = True
        self.sigCancelClicked.emit()
        self.hide()

    def cancel_cb(self, checked=False):
        self._emitCancelledAndHide()

    def closeEvent(self, event):
        if getattr(self, 'cancel', True):
            self.sigCancelClicked.emit()
        super().closeEvent(event)

    def saveContent(self, path):
        if self._tryParentSaveContent(path):
            return

        ext = '.json'
        if not path.endswith(ext):
            path += ext

        with open(path, 'w') as f:
            json.dump(self.getContent(), f, indent=2)

    def loadContent(self, path):
        if self._tryParentLoadContent(path):
            self.ok_cb()
            self.hide()
            return

        ext = '.json'
        if not path.endswith(ext):
            path += ext

        self.show()
        try:
            with open(path, 'r') as f:
                content = json.load(f)
            self.setContent(content)
        except Exception as e:
            self.logger.info(f'Failed to load content from {path}: {e}')
            traceback.print_exc()
            self.hide()
            return

        self.ok_cb()
        self.hide()


class BayesianTrackerDialogWorkflow(_BaseTrackerDialogWorkflow, apps.BayesianTrackerParamsWin):
    sigOkClicked = Signal()
    sigCancelClicked = Signal()

    def __init__(self, parent=None, posData=None, logger=printl):
        self.logger = logger
        self.requiresInputImage = True
        self.optional_inputs_n = {1: True}

        if posData is not None and hasattr(posData, 'img_data_shape') and posData.img_data_shape is not None:
            Y, X = posData.img_data_shape[-2:]
            if posData.isSegm3D:
                segm_shape = (posData.SizeZ, Y, X)
            else:
                segm_shape = (1, Y, X)
        else:
            segm_shape = (1, 512, 512)

        self._defaultSegmShape = segm_shape
        self._lastAutoVolumeText = None

        super().__init__(segm_shape, parent=parent, channels=None)
        self._lastAutoVolumeText = self.volumeLineEdit.text()

    def _volumeTextFromSegmShape(self, segm_shape):
        z, y, x = segm_shape
        volume_text = f'  (0, {x}), (0, {y})  '
        if z > 1:
            volume_text = f'{volume_text}, (0, {z})  '
        return volume_text

    def _segmShapeFromInputs(self):
        size_z, size_y, size_x = self._defaultSegmShape
        if hasattr(self, 'curr_input_types'):
            input_0 = self.curr_input_types.get(0)
            if input_0 is not None:
                input_size_z = getattr(input_0, 'SizeZ', None)
                input_size_y = getattr(input_0, 'SizeY', None)
                input_size_x = getattr(input_0, 'SizeX', None)
                if input_size_z is not None:
                    size_z = max(1, int(input_size_z))
                if input_size_y is not None:
                    size_y = int(input_size_y)
                if input_size_x is not None:
                    size_x = int(input_size_x)

        print((size_z, size_y, size_x))
        return (size_z, size_y, size_x)

    def _syncVolumeFromInputs(self):
        segm_shape = self._segmShapeFromInputs()
        new_auto_volume_text = self._volumeTextFromSegmShape(segm_shape)
        current_volume_text = self.volumeLineEdit.text()
        should_update = (
            not current_volume_text
            or self._lastAutoVolumeText is None
            or current_volume_text == self._lastAutoVolumeText
        )
        self._defaultSegmShape = segm_shape
        if should_update:
            self.volumeLineEdit.setText(new_auto_volume_text)
        self._lastAutoVolumeText = new_auto_volume_text

    def updatedInputTypes(self):
        self._syncVolumeFromInputs()

    def ok_cb(self, checked=False):
        super().ok_cb(checked=checked)
        if self.cancel or self.isVisible():
            return
        self.sigOkClicked.emit()

    def getContent(self):
        return {
            'model_path': self.modelPathLineEdit.text(),
            'features': list(self.features),
            'verbose': self.verboseToggle.isChecked(),
            'optimize': self.optimizeToggle.isChecked(),
            'max_search_radius': self.maxSearchRadiusSpinbox.value(),
            'volume': self.volumeLineEdit.text(),
            'step_size': self.stepSizeSpinbox.value(),
            'update_method': self.updateMethodCombobox.currentText(),
        }

    def setContent(self, content):
        content = content or {}
        self.modelPathLineEdit.setText(content.get('model_path', ''))
        self.features = list(content.get('features', []))
        self.verboseToggle.setChecked(bool(content.get('verbose', True)))
        self.optimizeToggle.setChecked(bool(content.get('optimize', True)))
        self.maxSearchRadiusSpinbox.setValue(
            int(content.get('max_search_radius', 50))
        )
        self.volumeLineEdit.setText(content.get('volume', self.volumeLineEdit.text()))
        self.stepSizeSpinbox.setValue(int(content.get('step_size', 100)))
        self.updateMethodCombobox.setCurrentText(
            content.get('update_method', 'EXACT')
        )


class CellACDCTrackerDialogWorkflow(_BaseTrackerDialogWorkflow, apps.CellACDCTrackerParamsWin):
    sigOkClicked = Signal()
    sigCancelClicked = Signal()

    def __init__(self, parent=None, logger=printl):
        self.logger = logger
        self.requiresInputImage = False
        super().__init__(parent=parent)

    def ok_cb(self, checked=False):
        super().ok_cb(checked=checked)
        if self.cancel or self.isVisible():
            return
        self.sigOkClicked.emit()

    def getContent(self):
        return {
            'IoA_thresh': self.maxOverlapSpinbox.value(),
        }

    def setContent(self, content):
        content = content or {}
        self.maxOverlapSpinbox.setValue(float(content.get('IoA_thresh', 0.4)))


class DeltaTrackerDialogWorkflow(_BaseTrackerDialogWorkflow, apps.DeltaTrackerParamsWin):
    sigOkClicked = Signal()
    sigCancelClicked = Signal()

    def __init__(self, parent=None, posData=None, logger=printl):
        self.logger = logger
        self.requiresInputImage = True
        super().__init__(posData=posData, parent=parent)
        self._hideOriginalImageSelection()

    def _hideOriginalImageSelection(self):
        params_layout = self.modelPathLineEdit.parentWidget().layout()
        for col in range(3):
            item = params_layout.itemAtPosition(0, col)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.hide()

    def methodChanged(self, method):
        self.model_type = method

    def ok_cb(self, checked=False):
        self.cancel = False
        self.verbose = self.verboseToggle.isChecked()
        self.legacy = self.legacyToggle.isChecked()
        self.pickle = self.pickleToggle.isChecked()
        self.movie = self.movieToggle.isChecked()
        self.chamber = self.chamberToggle.isChecked()
        self.model_type = self.updateMethodCombobox.currentText()

        model_path = self.modelPathLineEdit.text().strip()
        self.model_path = os.path.normpath(model_path) if model_path else ''
        self.params = {
            'original_images_path': self.model_path,
            'verbose': self.verbose,
            'legacy': self.legacy,
            'pickle': self.pickle,
            'movie': self.movie,
            'model_type': self.model_type,
            'single mothermachine chamber': self.chamber,
        }

        self.hide()
        self.sigOkClicked.emit()

    def getContent(self):
        return {
            'model_type': self.updateMethodCombobox.currentText(),
            'single_mothermachine_chamber': self.chamberToggle.isChecked(),
            'verbose': self.verboseToggle.isChecked(),
            'legacy': self.legacyToggle.isChecked(),
            'pickle': self.pickleToggle.isChecked(),
            'movie': self.movieToggle.isChecked(),
        }

    def setContent(self, content):
        content = content or {}
        self.updateMethodCombobox.setCurrentText(
            content.get('model_type', '2D')
        )
        self.model_type = self.updateMethodCombobox.currentText()
        self.chamberToggle.setChecked(
            bool(content.get('single_mothermachine_chamber', True))
        )
        self.verboseToggle.setChecked(bool(content.get('verbose', True)))
        self.legacyToggle.setChecked(bool(content.get('legacy', False)))
        self.pickleToggle.setChecked(bool(content.get('pickle', False)))
        self.movieToggle.setChecked(bool(content.get('movie', False)))
        
class ThresholdSegmDialogWorkflow(apps.QDialogAutomaticThresholding):
    sigOkClicked = Signal()
    sigCancelClicked = Signal()

    def __init__(self, parent=None, logger=printl):
        super().__init__(parent=parent, hide_on_close=True)
        self.logger = logger
        self.cancelButton.clicked.disconnect()
        self.cancelButton.clicked.connect(self.cancel_cb)
        self.okButton.clicked.connect(self.sigOkClicked.emit)
        
    def cancel_cb(self):
        self.sigCancelClicked.emit()
        self.hide()
        
    def updatedInputTypes(self):
        if not hasattr(self, 'curr_input_types') or 0 not in self.curr_input_types:
            return

        input_0 = self.curr_input_types[0]
        if input_0 is None:
            return

        size_z = getattr(input_0, 'SizeZ', None)
        is_input_3d = size_z is not None and size_z > 1
        self._set3DCheckboxEnabled(is_input_3d)
        
    def setContent(self, content):
        if not content:
            return

        if 'gauss_sigma' in content:
            self.sigmaGaussSpinbox.setValue(float(content['gauss_sigma']))

        threshold_method = content.get('threshold_method')
        if threshold_method:
            method = str(threshold_method)
            if method.startswith('threshold_'):
                method = method[10:].capitalize()
            self.threshMethodCombobox.setCurrentText(method)

        if self.segment3Dcheckbox is not None and 'segment_3D_volume' in content:
            if self.segment3Dcheckbox.isEnabled():
                self.segment3Dcheckbox.setChecked(bool(content['segment_3D_volume']))
            elif self.segmentSliceBySliceCheckbox is not None:
                self.segmentSliceBySliceCheckbox.setChecked(True)

        self.getContent()

    def saveContent(self, path):
        ext = '.ini'
        if not path.endswith(ext):
            path += ext

        content = self.getContent()
        cp = config.ConfigParser()
        cp['thresholding.segment'] = {
            'gauss_sigma': str(content.get('gauss_sigma', 1.0)),
            'threshold_method': str(content.get('threshold_method', 'threshold_otsu')),
            'segment_3D_volume': str(bool(content.get('segment_3D_volume', False))),
        }

        with open(path, 'w') as f:
            cp.write(f)

    def loadContent(self, path):
        ext = '.ini'
        if not path.endswith(ext):
            path += ext

        self.show()
        try:
            cp = config.ConfigParser()
            cp.read(path)
            section = cp['thresholding.segment'] if cp.has_section('thresholding.segment') else {}
            self.setContent(section)
        except Exception as e:
            self.logger.info(f'Failed to load content from {path}: {e}')
            traceback.print_exc()
            self.hide()
            return

        self.cancel = False
        self.getContent()
        self.sigOkClicked.emit()
        self.hide()
        
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
    
    def __init__(self, parent=None, logger=printl):
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

    def closeEvent(self, event):
        """Treat window close (X) as cancel."""
        self.sigCancelClicked.emit()
        super().closeEvent(event)
        
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
        with open(path, 'r') as f:
            content = json.load(f)
        self.setContent(content)
        self.ok_cb()  # To emit signals and update the workflow card state after loading
        self.hide()
        
    def ok_cb(self):
        """Handle OK button click by emitting the sigOkClicked signal and hiding the dialog."""
        self.sigOkClicked.emit()
        super().ok_cb(saveLastRecipe=False)  # Don't save to last recipe since this is just a workflow card dialog
        
class PreProcessSetupDialogWorkflow(apps.PreProcessRecipeDialog):
    sigOkClicked = Signal()
    sigCancelClicked = Signal()

    def __init__(self, parent=None, logger=printl):
        super().__init__(parent=parent, hideOnClosing=True)
        self.logger = logger

        self.previewCheckbox.hide()
        self.applyCurrentFrameButton.hide()
        self.savePreprocButton.hide()
        
        self.mainLayout.addSpacing(20)

        qutils.hide_and_delete_layout(self.buttonsLayout)
        buttonsLayout = widgets.CancelOkButtonsLayout()
        self.buttonsLayout = buttonsLayout
        
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.sigCancelClicked.emit)
        buttonsLayout.cancelButton.clicked.connect(self.cancelButtonClicked)
        
        self.mainLayout.addLayout(buttonsLayout)
    
    def ok_cb(self):
        recipe = self.recipe()
        if recipe is None:
            self.logger.info("No valid pre-processing steps configured. Please add at least one step before confirming.")
            return
        self.sigOkClicked.emit()
        super().ok_cb()

    def cancelButtonClicked(self):
        """Handle cancel button click by hiding the dialog."""
        self.hide()

    def closeEvent(self, event):
        """Treat window close (X) as cancel."""
        self.sigCancelClicked.emit()
        super().closeEvent(event)

    def saveContent(self, path):
        """Save pre-processing recipe to disk."""
        ext = '.ini'
        if not path.endswith(ext):
            path += ext

        cp = self.recipeConfigPars()
        cp['acdc.preprocess.metadata'] = {
            'keep_input_data_type': str(self.keepInputDataTypeToggle.isChecked())
        }
        with open(path, 'w') as f:
            cp.write(f)

    def loadContent(self, path):
        """Load pre-processing recipe from disk and update card state."""
        ext = '.ini'
        if not path.endswith(ext):
            path += ext

        self.show()
        try:
            cp = config.ConfigParser()
            cp.read(path)

            preproc_config_pars = {
                section: cp[section]
                for section in cp.sections()
                if section.startswith('acdc.preprocess.step')
            }
            if preproc_config_pars:
                self.preProcessParamsWidget.loadRecipe(preproc_config_pars)

            keep_input_dtype = cp.getboolean(
                'acdc.preprocess.metadata',
                'keep_input_data_type',
                fallback=True,
            )
            self.keepInputDataTypeToggle.setChecked(keep_input_dtype)
        except Exception as e:
            self.logger.info(f"Failed to load content from {path}: {e}")
            traceback.print_exc()

        self.ok_cb()  # Emit update signal and refresh workflow card preview.
        self.hide()
        
    def setContent(self, content):
        """Set dialog content from in-memory recipe content."""
        recipe = content['recipe'] # cannot sort, trust that list order is correct!
        keep_input_data_type = bool(content['keep_input_data_type'])
        clean_recipe = []
        for step in recipe:
            step_copy = dict(step)
            step_copy.pop('keep_input_data_type', None)
            clean_recipe.append(step_copy)

        ini_items = config.preprocess_recipe_to_ini_items(clean_recipe)
        self.preProcessParamsWidget.loadRecipe(ini_items)
        self.keepInputDataTypeToggle.setChecked(keep_input_data_type)
        
    def getContent(self):
        """Return current dialog content as in-memory recipe payload."""
        return {
            'recipe': self.recipe(warn=False) or [],
            'keep_input_data_type': self.keepInputDataTypeToggle.isChecked(),
        }
    
class PostProcessSegmDialogWorkflow(QBaseDialog):
    sigOkClicked = Signal()
    sigCancelClicked = Signal()
    sigValueChanged = Signal(object, object)
    sigInputsChanged = Signal(dict)  # Emit updated input types when they change

    def __init__(self, posData, parent=None, logger=printl):
        super().__init__(parent=parent)
        self.cancel = True
        self.logger = logger
        self.curr_input_types = {}
        self.optional_inputs_n = {1: True}  # metrics input is optional

        # In workflow "proceed without" mode posData can be None.
        # Build a minimal fallback so PostProcessSegmParams can initialize.
        posData = myutils.utilClass()
        posData.SizeZ = 1
        posData.isSegm3D = True
        posData.user_ch_name = 'Input 0'
        self._postprocess_posData = posData

        self.setWindowTitle('Post-process segmentation parameters')

        self.postProcessGroupbox = apps.PostProcessSegmParams(
            'Post-processing parameters', self._postprocess_posData,
            useSliders=True,
            parent=self,
            external_metrics=[],
        )
        self._syncPostProcessMetadataFromInputs()
        self.postProcessGroupbox.valueChanged.connect(self.valueChanged)
        self.postProcessGroupbox.sigNumberChannelsRequested.connect(self.setInputs_cb)
        layout = QVBoxLayout(self)
        layout.addWidget(self.postProcessGroupbox)
        layout.addSpacing(10)

        buttonsLayout = widgets.CancelOkButtonsLayout()
        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.cancel_cb)
        layout.addLayout(buttonsLayout)

        self.setLayout(layout)

    def _externalMetricsFromInputs(self):
        metrics = []
        for idx in sorted(self.curr_input_types):
            if idx == 0:
                continue
            input_type = self.curr_input_types[idx]
            if not isinstance(input_type, WfMetricsDC):
                continue
            input_metrics = input_type.setMetrics or []
            for metric in input_metrics:
                if isinstance(metric, str):
                    metrics.append(metric)

        # Preserve order while removing duplicates.
        return list(dict.fromkeys(metrics))

    def _syncPostProcessMetadataFromInputs(self):
        size_z = getattr(self._postprocess_posData, 'SizeZ', 1) or 1
        is_segm_3d = bool(getattr(self._postprocess_posData, 'isSegm3D', False))
        channel_name = getattr(self._postprocess_posData, 'user_ch_name', 'Input 0')

        input_0 = self.curr_input_types.get(0)
        if input_0 is not None:
            input_size_z = getattr(input_0, 'SizeZ', None)
            if input_size_z is not None:
                size_z = max(1, int(input_size_z))
                is_segm_3d = size_z > 1
                channel_name = 'Input 0'

        self.postProcessGroupbox.channelName = channel_name
        self.postProcessGroupbox.isSegm3D = is_segm_3d

        min_obj_size_z_widget = getattr(self.postProcessGroupbox, 'minObjSizeZ_SB', None)
        if hasattr(min_obj_size_z_widget, 'setMaximum'):
            min_obj_size_z_widget.setMaximum(size_z)
            if not is_segm_3d and hasattr(min_obj_size_z_widget, 'setValue'):
                min_obj_size_z_widget.setValue(0)
            if hasattr(min_obj_size_z_widget, 'setDisabled'):
                min_obj_size_z_widget.setDisabled(not is_segm_3d)

    def updatedInputTypes(self):
        self._syncPostProcessMetadataFromInputs()
        external_metrics = self._externalMetricsFromInputs()
        self.postProcessGroupbox.updateExternalMetrics(external_metrics)

    def setInputs_cb(self, number_inputs):
        """Handle when required metric inputs change and update accepted inputs."""
        inputs = {0: WfSegmDC()}  # first is segmentation
        inputs.update({n: WfMetricsDC() for n in range(1, number_inputs+1)})
        self.sigInputsChanged.emit(inputs)

    def _setCheckableRangeValue(self, range_widgets, value):
        is_checked = value is not None
        range_widgets.checkbox.setChecked(is_checked)
        if is_checked:
            range_widgets.spinbox.setValue(value)

    def _restoreSelectedFeatures(self, selected_features_range, selected_features_group):
        groupbox = self.postProcessGroupbox.selectedFeaturesDialog.groupbox
        groupbox.resetFields()

        items = list((selected_features_range or {}).items())
        if not items:
            return

        while len(groupbox.selectors) < len(items):
            groupbox.addFeatureField()

        while len(groupbox.selectors) > len(items):
            groupbox.selectors[-1].delButton.click()

        for selector, (feature_name, feature_range) in zip(groupbox.selectors, items):
            selector.setText(feature_name)

            feature_group = (selected_features_group or {}).get(feature_name)
            if feature_group is not None:
                selector.featureGroup = feature_group

            if isinstance(feature_range, (list, tuple)) and len(feature_range) >= 2:
                low_val, high_val = feature_range[0], feature_range[1]
            else:
                low_val, high_val = None, None

            self._setCheckableRangeValue(selector.lowRangeWidgets, low_val)
            self._setCheckableRangeValue(selector.highRangeWidgets, high_val)

    def hasInvalidContent(self):
        groupbox = self.postProcessGroupbox.selectedFeaturesDialog.groupbox
        return groupbox.hasInvalidExternalMetrics(blink_on_invalid=True)

    def valueChanged(self, value):
        self.sigValueChanged.emit(None, None)

    def cancel_cb(self):
        self.cancel = True
        self.sigCancelClicked.emit()
        self.hide()

    def closeEvent(self, event):
        """Treat window close (X) as cancel."""
        self.cancel = True
        self.sigCancelClicked.emit()
        super().closeEvent(event)

    def getContent(self):
        groupbox = self.postProcessGroupbox.selectedFeaturesDialog.groupbox
        return {
            'kwargs': self.postProcessGroupbox.kwargs(),
            'selected_features_range': self.postProcessGroupbox.selectedFeaturesRange(),
            'selected_features_group': groupbox.selectedFeaturesGroup(),
        }

    def setContent(self, content):
        content = content or {}
        kwargs = content.get('kwargs') or {}
        selected_features_range = content.get('selected_features_range') or {}
        selected_features_group = content.get('selected_features_group') or {}

        self.postProcessGroupbox.restoreFromKwargs(kwargs)
        self._restoreSelectedFeatures(
            selected_features_range=selected_features_range,
            selected_features_group=selected_features_group,
        )

    def saveContent(self, path):
        ext = '.json'
        if not path.endswith(ext):
            path += ext

        with open(path, 'w') as f:
            json.dump(self.getContent(), f, indent=2)

    def loadContent(self, path):
        ext = '.json'
        if not path.endswith(ext):
            path += ext

        self.show()
        try:
            with open(path, 'r') as f:
                content = json.load(f)
            self.setContent(content)
        except Exception as e:
            self.logger.info(f"Failed to load content from {path}: {e}")
            traceback.print_exc()
            self.hide()
            return

        self.ok_cb()
        self.hide()

    def ok_cb(self):
        """Handle OK button click by emitting the sigOkClicked signal and hiding the dialog."""
        self.cancel = False
        self.sigOkClicked.emit()
        self.hide()

class SetMeasurementsDialogWorkflow(apps.SetMeasurementsDialog):
    sigOkClicked = Signal()
    sigCancelClicked = Signal()
    sigInputsChanged = Signal(dict)

    def __init__(self, posData, parent=None, logger=printl):
        super().__init__(
            loadedChNames=[None],
            notLoadedChNames=[],
            parent=parent,
            posData=posData,
            addCustomChannels=True,
        )
        self.logger = logger
        self.sigNumberChannelsRequested.connect(self.setInputs_cb)
        # remove save/load buttons since we'll handle that through the workflow card's save/load content functions
        self.saveCurrentSelectionButton.hide()
        self.loadSavedSelectionButton.hide()
        self.loadLastSelButton.hide()
        # remove all other connections to the OK/Cancel buttons since we'll handle that through the workflow card's signals
        self.cancelButton.clicked.disconnect()
        self.okButton.clicked.disconnect()
        self.cancelButton.clicked.connect(self.cancel_clicked)
        self.okButton.clicked.connect(self.ok_clicked)
        
    def setInputs_cb(self, number_inputs):
        inputs = {0: WfSegmDC()}  # first input is segmentation
        inputs.update({n: WfImageDC() for n in range(1, number_inputs + 1)})
        self.sigInputsChanged.emit(inputs)

    def ok_clicked(self):
        self.sigOkClicked.emit()
        self.hide()

    def cancel_clicked(self):
        self.sigCancelClicked.emit()
        self.hide()

    def closeEvent(self, event):
        """Treat window close (X) as cancel."""
        self.sigCancelClicked.emit()
        super().closeEvent(event)

    def selectedMetrics(self):
        """Return selected metrics as a flat list with section context."""
        metrics_mapper = self.currentSelectionMapper()
        selected_metrics = []
        for section, metrics in metrics_mapper.items():
            if section == 'DEFAULT' or not isinstance(metrics, dict):
                continue
            for metric_name in metrics:
                selected_metrics.append(f'{section}/{metric_name}')

        return selected_metrics

    def getContent(self):
        # Keep enough in-memory state to fully restore dialog UI on cancel.
        custom_channels = [
            gbox.chName
            for gbox in self.chNameGroupboxes
            if getattr(gbox, 'isCustomChannel', False)
        ]
        return {
            'selection_mapper': self.currentSelectionMapper(),
            'custom_channels': custom_channels,
        }

    def setContent(self, content):
        if not content:
            return

        selection_mapper = content.get('selection_mapper') or {}
        custom_channels = content.get('custom_channels') or []
        # self._restoreCustomChannels(custom_channels)
        selection_mapper = dict(selection_mapper)
        self.setCurrentSelectionFromMapper(selection_mapper)

    def saveContent(self, path):
        ext = '.ini'
        if not path.endswith(ext):
            path += ext

        # Persist using original SetMeasurementsDialog-compatible mapper format.
        content = self.currentSelectionMapper()
        cp = config.ConfigParser()
        for section, values in content.items():
            cp[section] = {}
            for option, value in values.items():
                cp[section][option] = str(value)

        with open(path, 'w') as f:
            cp.write(f)

    def loadContent(self, path):
        ext = '.ini'
        if not path.endswith(ext):
            path += ext

        self.show()
        try:
            cp = config.ConfigParser()
            cp.read(path)
            content = dict(cp)
            # File payload is the selection mapper format.
            self.setCurrentSelectionFromMapper(content)
        except Exception as e:
            self.logger.info(f'Failed to load content from {path}: {e}')
            traceback.print_exc()
            self.hide()
            return

        self.ok_cb()
        self.hide()
        
    def updatedInputTypes(self):
        """Handle when input types change and update the dialog UI accordingly."""
        """Need to verify this works!"""
        curr_input_types = self.curr_input_types
        isSegm3D = True
        if self.curr_input_types[0] is not None:
            isSegm3D = (self.curr_input_types[0].SizeZ>1 
                        if self.curr_input_types[0].SizeZ is not None 
                        else False)
            
        for channelGBox in self.chNameGroupboxes:            # extract channel number input...
            regex_term = re.compile(r'Input (\d+)')
            match = regex_term.search(channelGBox.chName)
            if match is None:
                continue
            channel_num = int(match.group(1))
            input_type = curr_input_types.get(channel_num, None)
            if input_type is None:
                continue
            isZstack = input_type.SizeZ>1 if input_type.SizeZ is not None else False
            channelGBox.updateZmode(
                isZstack=isZstack,
                isSegm3D=isSegm3D
            )


class WorkflowCardMetadataDialog(QBaseDialog):
    """Local metadata editor that handles SizeT/SizeZ/SizeX/SizeY values."""

    def __init__(self, metadata_state, selection_label='', parent=None):
        super().__init__(parent=parent)
        self.setWindowTitle('Selection metadata')

        metadata_state = metadata_state or {}
        self.cancel = True
        self._initial_size_t = int(metadata_state.get('SizeT', 1) or 1)
        self._initial_size_z = int(metadata_state.get('SizeZ', 1) or 1)
        self._initial_size_x = int(metadata_state.get('SizeX', 512) or 512)
        self._initial_size_y = int(metadata_state.get('SizeY', 512) or 512)

        self.SizeT = self._initial_size_t
        self.SizeZ = self._initial_size_z
        self.SizeX = self._initial_size_x
        self.SizeY = self._initial_size_y

        img_data_shape = metadata_state.get('img_data_shape')

        layout = QVBoxLayout(self)

        if selection_label:
            layout.addWidget(QLabel(f'Selection: {selection_label}'))

        if img_data_shape is not None:
            layout.addWidget(QLabel(f'Image shape: {tuple(img_data_shape)}'))

        size_t_layout = QHBoxLayout()
        size_t_layout.addWidget(QLabel('SizeT (frames):'))
        self.SizeT_SpinBox = QSpinBox(self)
        self.SizeT_SpinBox.setMinimum(1)
        self.SizeT_SpinBox.setMaximum(100000)
        self.SizeT_SpinBox.setValue(self._initial_size_t)
        size_t_layout.addWidget(self.SizeT_SpinBox)
        layout.addLayout(size_t_layout)

        size_z_layout = QHBoxLayout()
        size_z_layout.addWidget(QLabel('SizeZ (z-slices):'))
        self.SizeZ_SpinBox = QSpinBox(self)
        self.SizeZ_SpinBox.setMinimum(1)
        self.SizeZ_SpinBox.setMaximum(100000)
        self.SizeZ_SpinBox.setValue(self._initial_size_z)
        size_z_layout.addWidget(self.SizeZ_SpinBox)
        layout.addLayout(size_z_layout)
        
        size_x_layout = QHBoxLayout()
        size_x_layout.addWidget(QLabel('SizeX (width):'))
        self.SizeX_SpinBox = QSpinBox(self)
        self.SizeX_SpinBox.setMinimum(1)
        self.SizeX_SpinBox.setMaximum(100000)
        self.SizeX_SpinBox.setValue(self._initial_size_x)
        size_x_layout.addWidget(self.SizeX_SpinBox)
        layout.addLayout(size_x_layout)
        
        size_y_layout = QHBoxLayout()
        size_y_layout.addWidget(QLabel('SizeY (height):'))
        self.SizeY_SpinBox = QSpinBox(self)
        self.SizeY_SpinBox.setMinimum(1)
        self.SizeY_SpinBox.setMaximum(100000)
        self.SizeY_SpinBox.setValue(self._initial_size_y)
        size_y_layout.addWidget(self.SizeY_SpinBox)
        layout.addLayout(size_y_layout)

        buttons_layout = widgets.CancelOkButtonsLayout()
        buttons_layout.okButton.clicked.connect(self.ok_cb)
        buttons_layout.cancelButton.clicked.connect(self.cancel_cb)
        layout.addLayout(buttons_layout)

    # def _warnEditingMetadata(self, new_value, old_value, label):
    #     if new_value == old_value:
    #         return True

    #     message = (
    #         f'You changed {label} from {old_value} to {new_value}.\n\n'
    #         'Proceed with this metadata override?'
    #     )
    #     response = QMessageBox.question(
    #         self,
    #         'Confirm metadata edit',
    #         message,
    #         QMessageBox.Yes | QMessageBox.No,
    #         QMessageBox.No,
    #     )
    #     return response == QMessageBox.Yes

    def ok_cb(self, checked=False):
        size_t = self.SizeT_SpinBox.value()
        size_z = self.SizeZ_SpinBox.value()
        size_x = self.SizeX_SpinBox.value()
        size_y = self.SizeY_SpinBox.value()

        self.SizeT = int(size_t)
        self.SizeZ = int(size_z)
        self.SizeX = int(size_x)
        self.SizeY = int(size_y)

        self.cancel = False
        self.close()

    def cancel_cb(self, checked=False):
        self.cancel = True
        self.close()

class WorkflowInputDataDialog(QBaseDialog):
    """Simple dialog for selecting input data (image or segmentation channel).
    
    Provides a dropdown menu for the user to select from available data options
    and a browse button to optionally select a specific file.
    Used as an input node in the workflow to let users specify which data
    should be passed to the next step.
    
    Metadata Resolution:
    When a selection is made, metadata is resolved through a multi-level fallback:
    1. Try to extract from self.posData (if available)
    2. Try to load as Cell-ACDC dataset and extract metadata
    3. Try to infer dimensions from image file directly
    
    Resolved metadata is cached to avoid repeated file I/O. Metadata can be
    manually overridden in the metadata editor dialog and persists with the selection.
    Only metadata is cached; posData objects are temporary during resolution.
    
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
    sigSetOutputs = Signal(dict)  # Emit new output definitions when selection changes and affects output type
    
    def __init__(self, selection_options, type, parent=None, logger=printl,
                 posData=None):
        """Initialize the WorkflowInputDataDialog.
        
        Sets up the GUI with dropdown selector, browse button, metadata editor,
        and OK/Cancel buttons. Initializes internal state for tracking selections
        and metadata across different data sources.
        
        Args:
            selection_options (list): Available data options to display in dropdown.
            type (str): Type of data ('image' or 'segmentation').
            parent (QWidget, optional): Parent widget. Defaults to None.
            logger (callable, optional): Logging function. Defaults to printl.
            posData (object, optional): Position data object containing image/segmentation info.
                Used as fallback source for metadata. Defaults to None.
        
        Instance Attributes:
            type (str): Data type for this dialog.
            selection_options (list): Available selection options.
               posData (object): Current position data reference (used as fallback during resolution).
               logger (callable): Logging function.
               selected_value (str): User's confirmed selection.
               selection_metadata_state (dict): Resolved metadata (auto or user-edited) indexed by selection key.
                   Each entry maps selection key → {SizeT, SizeZ, SizeY, SizeX, img_data_shape}
        """
        super().__init__(parent=parent)
        self.setWindowTitle(f'Select input {type}')
        
        self.type = type
        self.selection_options = selection_options or []
        self.posData = posData
        self.logger = logger
        self.selected_value = None
        self.selection_metadata_state = {}
        
        # Create layout
        layout = QVBoxLayout()
        
        # Add label
        label = QLabel(f'Select {type} input:')
        layout.addWidget(label)
        
        # Add combo box for selection and allow browsing a specific file.
        selection_layout = QHBoxLayout()
        self.selection_widget = QComboBox()
        self.selection_widget.addItems(self.selection_options)
        selection_layout.addWidget(self.selection_widget)

        self.browse_button = QPushButton('Browse...')
        self.browse_button.clicked.connect(self.browseSelectionFile)
        selection_layout.addWidget(self.browse_button)

        self.metadata_button = QPushButton('Metadata...')
        self.metadata_button.clicked.connect(self.editSelectionMetadata)
        selection_layout.addWidget(self.metadata_button)
        layout.addLayout(selection_layout)
        
        # Add buttons
        buttons_layout = widgets.CancelOkButtonsLayout()
        self.ok_button = buttons_layout.okButton
        self.cancel_button = buttons_layout.cancelButton
        self.ok_button.clicked.connect(self.ok_clicked)
        self.ok_button.clicked.connect(self.sigOkClicked.emit)
        self.cancel_button.clicked.connect(self.cancel_clicked)
        self.cancel_button.clicked.connect(self.sigCancelClicked.emit)
        
        # add connection to update sizeT and SizeZ
        self.selection_widget.currentTextChanged.connect(self.selectionChanged)
        
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)

    def _getSelectionLabel(self, selection):
        """Extract display label from a selection identifier.
        
        For file paths, returns just the filename. For other selections,
        returns the string as-is. Used for UI display in titles and labels.
        
        Args:
            selection: Selection identifier (file path or text label).
        
        Returns:
            str: Display label (filename basename or original text).
        """
        if os.path.isfile(selection):
            return os.path.basename(selection)
        return selection

    def _isValidSelection(self, selection):
        """Check if a selection is valid.
        
        A selection is valid if it appears in the available options list
        or if it points to an existing file on disk.
        
        Args:
            selection: Selection identifier to validate.
        
        Returns:
            bool: True if selection is in options or is a valid file path.
        """
        return selection in self.selection_options or os.path.isfile(selection)

    def _ensureSelectionItem(self, value, color=None):
        """Ensure a value exists in the selection dropdown and optionally color it.
        
        Adds the value to the dropdown if it doesn't already exist, and sets
        the text color. Useful for adding custom file paths or highlighting
        invalid/missing selections in red.
        
        Args:
            value (str): Value to add or find in dropdown.
            color (str, optional): Color name for text (e.g., 'red'). None = default color.
        
        Returns:
            int: Index of the value in the dropdown.
        """
        idx = self.selection_widget.findText(value)
        if idx == -1:
            self.selection_widget.addItem(value)
            idx = self.selection_widget.count() - 1

        model = self.selection_widget.model()
        item = model.item(idx)
        if item is not None:
            item.setForeground(QColor(color) if color else QColor())

        return idx

    def _metadataFromPosData(self, posData):
        """Extract dimension metadata from a posData object.
        
        Retrieves image dimensions (SizeT, SizeZ, SizeY, SizeX) and shape
        from a position data object. Returns None if posData is None.
        
        Args:
            posData (object): Position data object with dimension attributes.
        
        Returns:
            dict or None: Dictionary with keys {SizeT, SizeZ, SizeY, SizeX, img_data_shape}
                         with integer dimension values. Returns None if input is None.
        """
        if posData is None:
            return None

        size_t = int(getattr(posData, 'SizeT', 1) or 1)
        size_z = int(getattr(posData, 'SizeZ', 1) or 1)
        size_y = int(getattr(posData, 'SizeY', 1) or 1)
        size_x = int(getattr(posData, 'SizeX', 1) or 1)
        img_data_shape = getattr(posData, 'img_data_shape', None)

        return {
            'SizeT': size_t,
            'SizeZ': size_z,
            'SizeY': size_y,
            'SizeX': size_x,
            'img_data_shape': img_data_shape,
        }

    def _metadataFromImageFile(self, selection):
        """Extract dimension metadata by loading and examining image file.
        
        Loads the selected file and infers dimensions based on array shape.
        Assumes standard TZYX dimension ordering for 4D data and logs warnings
        for atypical dimensions. Provides safe defaults for edge cases.
        
        Args:
            selection (str): File path to load metadata from.
        
        Returns:
            dict: Dictionary with keys {SizeT, SizeZ, SizeY, SizeX, img_data_shape}
                  where dimensions are inferred from the file's array shape.
        
        Dimension inference logic:
            - 4D+: Use last 4 dimensions as TZYX
            - 3D: Assume TYX (SizeZ=1)
            - 2D: Assume YX (SizeT=SizeZ=1)
            - Other: Default to (1,1,1,1)
        """
        file = load.load_image_file(selection)
        ndim = file.ndim

        if ndim >= 4:
            size_t, size_z, size_y, size_x = file.shape[-4:]
            img_data_shape = tuple(file.shape[-4:])
        elif ndim == 3:
            size_t, size_y, size_x = file.shape
            size_z = 1
            img_data_shape = tuple(file.shape)
            self.logger.info(
                f'[WARNING]: Selected file "{selection}" has 3 dimensions. '
                f'Assuming SizeZ=1 and SizeT={size_t}.'
            )
        elif ndim == 2:
            size_t = size_z = 1
            size_y, size_x = file.shape
            img_data_shape = tuple(file.shape)
        else:
            # Keep a safe shape default for unsupported scalar/1D arrays.
            size_t = size_z = size_y = size_x = 1
            img_data_shape = (1, 1)

        return {
            'SizeT': int(size_t),
            'SizeZ': int(size_z),
            'SizeY': int(size_y),
            'SizeX': int(size_x),
            'img_data_shape': img_data_shape,
        }

    def _loadSelectionSourcePosData(self, selection):
        """Load a full posData object for the selected file or path.
        
        Attempts to load the selection as a Cell-ACDC dataset with all metadata
        (segmentation info, measurements, etc.). Falls back gracefully to basic
        image loading if the file is not part of a valid Cell-ACDC structure.
        
        Args:
            selection (str): File path or Cell-ACDC dataset path to load.
        
        Returns:
            object: Loaded posData object with image data and available metadata.
        
        Note:
            Errors during metadata loading (non-Cell-ACDC files) are silently
            caught to avoid cluttering logs for arbitrary files.
        """
        source_pos_data = load.loadData(selection, '', QParent=self)
        source_pos_data.loadImgData(selection)

        # Random standalone files may not belong to a valid Cell-ACDC
        # basename/dataset structure. Try metadata loading only when safe.
        should_try_metadata = True
        try:
            load.get_posData_metadata(
                source_pos_data.images_path,
                getattr(source_pos_data, 'basename', ''),
            )
        except Exception:
            should_try_metadata = False

        if should_try_metadata:
            try:
                source_pos_data.loadOtherFiles(
                    load_metadata=True,
                    load_customCombineMetrics=True,
                    load_customAnnot=True,
                    loadSegmInfo=self.type == 'segmentation',
                    loadBkgrROIs=True,
                    load_dataPrep_ROIcoords=True,
                    load_dataprep_free_roi=True,
                )
            except Exception:
                # Keep working with image-derived defaults without surfacing
                # basename-related warnings for arbitrary files.
                pass

        return source_pos_data

    def _selectionMetadataKey(self, selection):
        """Generate a unique cache key for a selection.
        
        Converts the selection to a string and strips whitespace. Used to
        create consistent keys for metadata state and source cache dictionaries.
        
        Args:
            selection: Selection identifier (string convertible).
        
        Returns:
            str: Normalized selection key.
        """
        selection_str = str(selection).strip()
        if not selection_str:
            return selection_str

        if os.path.isfile(selection_str):
            return os.path.normpath(os.path.abspath(selection_str))

        images_path = getattr(self.posData, 'images_path', None)
        if images_path:
            try:
                resolved_path = load.get_filepath_from_endname(images_path, selection_str)
            except Exception:
                resolved_path = ''

            if resolved_path:
                return os.path.normpath(os.path.abspath(resolved_path))

        return selection_str

    def _normalizeMetadataState(self, metadata_state):
        """Normalize metadata state to ensure all required keys and valid types.
        
        Ensures metadata dictionary has all dimension keys with valid integer
        values. Defaults to 1 if keys are missing or invalid. Also preserves
        the img_data_shape if present.
        
        Args:
            metadata_state (dict or None): Metadata dictionary to normalize.
        
        Returns:
            dict: Normalized metadata with guaranteed keys {SizeT, SizeZ, SizeY, SizeX, img_data_shape}.
        """
        metadata_state = metadata_state or {}
        size_t = int(metadata_state.get('SizeT', 1) or 1)
        size_z = int(metadata_state.get('SizeZ', 1) or 1)
        size_y = int(metadata_state.get('SizeY', 1) or 1)
        size_x = int(metadata_state.get('SizeX', 1) or 1)
        img_data_shape = metadata_state.get('img_data_shape')
        return {
            'SizeT': size_t,
            'SizeZ': size_z,
            'SizeY': size_y,
            'SizeX': size_x,
            'img_data_shape': img_data_shape,
        }

    def _applySegmInfoAdjustmentsToMetadata(self, metadata_state, selection, source_pos_data):
        """Adjust metadata based on segmentation info (e.g., which_z_proj).
        
        For segmentation selections, checks the segmInfo.csv to determine if the
        segmentation uses a z-projection. If so, sets SizeZ to 1.
        
        Args:
            metadata_state (dict): The metadata state to adjust.
            selection: The selection identifier.
            source_pos_data: The source position data object.
            
        Returns:
            dict: The adjusted metadata state.
        """
        if self.type != 'segmentation':
            return metadata_state
            
        segm_info_df = (
            source_pos_data.segmInfo_df
            if source_pos_data is not None and hasattr(source_pos_data, 'segmInfo_df') else None
        )
        segm_info_key = self._getSegmInfoSelectionKey(selection, source_pos_data)
        
        if segm_info_df is not None and segm_info_key in segm_info_df.index:
            segm_info_row = segm_info_df.loc[segm_info_key]
            proj_method = segm_info_row['which_z_proj']
            if proj_method != 'single z-slice':
                # Segmentation uses z-projection, so it's 2D
                metadata_state = dict(metadata_state)  # Make a copy to avoid mutating
                metadata_state['SizeZ'] = 1
        
        return metadata_state

    def _resolveSelectionState(self, selection):
        """Resolve complete state (metadata and source) for a selection.
        
            Resolve metadata for a selection using a multi-level fallback:
            1. Check cache: already resolved?
            2. Fresh load: try posData → load posData → load image file
            3. Apply adjustments (e.g., which_z_proj for segmentations)
        
            Caches only the final resolved metadata (not intermediate posData objects).
            posData is only temporary, used during resolution.
        
        Args:
            selection (str): The selection identifier to resolve.
        
        Returns:
                dict: Resolved metadata with keys {SizeT, SizeZ, SizeY, SizeX, img_data_shape}.
        """
        key = self._selectionMetadataKey(selection)
        
        # Return cached metadata if already resolved
        metadata_state = self.selection_metadata_state.get(key)
        if metadata_state is not None:
            return self._normalizeMetadataState(metadata_state)
       
        # Fresh resolution: posData → load posData → image file
        source_pos_data = self.posData
        metadata_state = self._metadataFromPosData(source_pos_data)
        
        if metadata_state is None and os.path.isfile(selection):
            try:
                source_pos_data = self._loadSelectionSourcePosData(selection)
                metadata_state = self._metadataFromPosData(source_pos_data)
            except Exception as err:
                self.logger.info(
                    f'[WARNING]: Failed to load metadata for selected file '
                    f'"{selection}" via loadData: {err}. '
                    'Falling back to direct file shape.'
                )
                try:
                    metadata_state = self._metadataFromImageFile(selection)
                except Exception as shape_err:
                    self.logger.info(
                        f'[WARNING]: Could not read image shape for "{selection}": '
                                        f'{shape_err}. Using default dimensions.'
                    )
       
        metadata_state = self._normalizeMetadataState(metadata_state)
        # Apply segmentation adjustments (which_z_proj, etc.)
        metadata_state = self._applySegmInfoAdjustmentsToMetadata(
            metadata_state, selection, source_pos_data
        )
        
                # Cache the resolved metadata
        self.selection_metadata_state[key] = metadata_state
        return metadata_state
    
    def _getSegmInfoSelectionKey(self, selection, source_pos_data):
        """Derive the segmInfo.csv lookup key for a selection.
        
        For file paths, extracts the segmentation name by removing the dataset
        basename. For non-file selections, returns them as-is. Used to look up
        segmentation properties in the segmInfo DataFrame.
        
        Args:
            selection (str): Selection identifier (file path or name).
            source_pos_data (object): posData object with optional basename attribute.
        
        Returns:
            str: Key to look up in segmInfo DataFrame rows.
        """
        if not os.path.isfile(selection):
            return selection

        filename = os.path.splitext(os.path.basename(selection))[0]
        basename = getattr(source_pos_data, 'basename', None)
        if basename and filename.startswith(basename):
            return filename[len(basename):]
        return filename

    def _getSelectionDialogFilters(self):
        """Get file dialog filters based on selection type.
        
        Returns appropriate file filters for the file browser dialog,
        based on whether we're selecting images or segmentations.
        
        Returns:
            str: File filter string for QFileDialog.
        """
        if self.type == 'image':
            return (
                'Image files (*.tif *.tiff *.npz *.npy);;'
                'All Files (*)'
            )

        return 'Segmentation files (*.npz *.npy);;All Files (*)'

    def browseSelectionFile(self):
        """Open file browser to select a custom file.
        
        Launches a file dialog with appropriate filters for image or segmentation
        files. The dialog starts in a sensible default directory (current selection,
        posData images path, or most recent used path).
        
        When a file is selected, updates the dropdown to include it (if not already
        present), sets it as current, and triggers selectionChanged to update metadata.
        """
        current_text = self.selection_widget.currentText().strip()
        if os.path.isfile(current_text):
            start_dir = os.path.dirname(current_text)
        elif self.posData is not None:
            start_dir = self.posData.images_path
        else:
            start_dir = myutils.getMostRecentPath()

        filepath = getopenfilename(
            parent=self,
            caption=f'Select {self.type} file',
            basedir=start_dir,
            filters=self._getSelectionDialogFilters()
        )[0]
        if not filepath:
            return

        idx = self._ensureSelectionItem(filepath)
        self.selection_widget.setCurrentIndex(idx)
        self.selected_value = filepath
        self.selectionChanged(filepath)

    def editSelectionMetadata(self):
        """Open metadata editor dialog for the current selection.
        
        Launches WorkflowCardMetadataDialog to allow user to override metadata
        (SizeT, SizeZ, SizeY, SizeX) for the current/selected data. Persists
        any user edits as overrides in the selection_metadata_state cache.
        
        If metadata editor is cancelled, no changes are made. After successful
        edit, triggers selectionChanged to update workflow output definitions.
        """
        selection = self.selection_widget.currentText().strip()
        if not selection:
            selection = self.selected_value

        if not selection:
            return

        metadata_state = self._resolveSelectionState(selection)

        # Keep parent dialog actions disabled while child metadata editor is open.
        if hasattr(self, 'ok_button'):
            self.ok_button.setEnabled(False)
        if hasattr(self, 'cancel_button'):
            self.cancel_button.setEnabled(False)

        try:
            metadataWin = WorkflowCardMetadataDialog(
                metadata_state=metadata_state,
                selection_label=self._getSelectionLabel(selection),
                parent=self,
            )

            metadataWin.exec_()
            if metadataWin.cancel:
                return

            # Persist edited metadata in the card state for this selection.
            key = self._selectionMetadataKey(selection)
            # Keep original image shape unchanged - just display it for reference
            self.selection_metadata_state[key] = {
                'SizeT': metadataWin.SizeT,
                'SizeZ': metadataWin.SizeZ,
                'SizeY': metadataWin.SizeY,
                'SizeX': metadataWin.SizeX,
                'img_data_shape': metadata_state.get('img_data_shape'),
            }

        except Exception as err:
            self.logger.info(
                f'[WARNING]: Could not open metadata editor for "{selection}": {err}'
            )
            traceback.print_exc()
            return
        finally:
            if hasattr(self, 'ok_button'):
                self.ok_button.setEnabled(True)
            if hasattr(self, 'cancel_button'):
                self.cancel_button.setEnabled(True)

        self.selectionChanged(selection)
        
    def selectionChanged(self, new_selection):
        """Handle selection change and emit updated output definitions.
        
        Called whenever the user selects a different option or when metadata
        changes. Resolves the metadata for the selection, creates appropriate
        output data class (WfImageDC or WfSegmDC), and emits sigSetOutputs
        to notify the workflow of the new output type.
        
        Metadata for segmentations may be adjusted based on which_z_proj
        (z-projection method) to reflect true output dimensionality.
        
        Args:
            new_selection (str): The newly selected value from dropdown.
        
        Emits:
            sigUpdateTitle: With updated selection label.
            sigSetOutputs: With output data class dictionary.
        """
        self.sigUpdateTitle.emit(
            f'{self.type}: {self._getSelectionLabel(new_selection)}'
        )
        metadata_state = self._resolveSelectionState(new_selection)
        new_output_size_t = metadata_state.get('SizeT', 1)
        new_output_size_z = metadata_state.get('SizeZ', 1)
        new_output_size_y = metadata_state.get('SizeY', 1)
        new_output_size_x = metadata_state.get('SizeX', 1)
        if self.posData is not None or metadata_state is not None:
            if self.type == 'image':
                out_data_class = WfImageDC(
                    SizeT=new_output_size_t,
                    SizeZ=new_output_size_z,
                    SizeY=new_output_size_y,
                    SizeX=new_output_size_x,
                )
            elif self.type == 'segmentation':
                # here theoretically we could have a 2D segm on 3D image...
                # Note: which_z_proj check is now handled during metadata resolution
                # in _applySegmInfoAdjustmentsToMetadata, so SizeZ is already correct
                out_data_class = WfSegmDC(
                    SizeT=new_output_size_t,
                    SizeZ=new_output_size_z,
                    SizeY=new_output_size_y,
                    SizeX=new_output_size_x,
                )
            else:
                raise ValueError(f"Unsupported type '{self.type}' for WorkflowInputDataDialog.")    
            self.sigSetOutputs.emit({0: out_data_class})
    
    def ok_clicked(self):
        """Handle OK button click and confirm the selection.
        
        Validates the currently selected value, emits appropriate signal
        (sigSelectionValid or sigSelectionInvalid), and hides the dialog.
        Triggers selectionChanged if selection is valid to update workflow outputs.
        
        The selected_value is persisted and can be retrieved later via get_selected_value().
        """
        self.selected_value = self.selection_widget.currentText()
        if self._isValidSelection(self.selected_value):
            self.sigSelectionValid.emit()
            self.selectionChanged(self.selected_value)  # Update outputs based on selection
        else:
            self.sigSelectionInvalid.emit(self.selected_value)
        self.hide()
    
    def cancel_clicked(self):
        """Handle Cancel button click by dismissing the dialog without action.
        
        Simply hides the dialog. selected_value remains unchanged.
        """
        self.hide()

    def closeEvent(self, event):
        """Handle window close event (X button) and treat it as cancel.
        
        Emits sigCancelClicked signal and calls parent closeEvent.
        
        Args:
            event (QCloseEvent): The close event.
        """
        self.sigCancelClicked.emit()
        super().closeEvent(event)
    
    def get_selected_value(self):
        """Get the currently selected value.
        
        Returns:
            str: The text of the currently selected item in the dropdown.
        """
        if self.selected_value is not None:
            return self.selected_value
        return self.selection_widget.currentText()

    def setContent(self, content):
        """Set the current selection and metadata from in-memory content.
        
        Accepts either a dictionary with 'selected_value' and 'metadata_state' keys,
        or a plain string to use as the selection value. Updates the dropdown to
        the specified value and repopulates the metadata override cache.
        
        Used for restoring dialog state from saved workflow cards or other sources.
        
        Args:
            content (dict or str): Content dictionary or selection string.
                If dict, may have keys:
                    - 'selected_value': str, the selection to set
                    - 'metadata_state': dict, manual metadata overrides per selection
                If str, treated as selected_value directly.
        """
        if isinstance(content, dict):
            value = str(content.get('selected_value', ''))
            metadata_state = content.get('metadata_state') or {}
            if isinstance(metadata_state, dict):
                self.selection_metadata_state = {
                    self._selectionMetadataKey(k): self._normalizeMetadataState(v)
                    for k, v in metadata_state.items() if isinstance(v, dict)
                }
        else:
            value = str(content)

        if not value:
            return

        idx = self._ensureSelectionItem(value)

        self.selection_widget.setCurrentIndex(idx)
        self.selected_value = value

    def getContent(self):
        """Get current selection and metadata state as serializable dictionary.
        
        Returns the selected value and all stored metadata overrides for all
        selections seen in this dialog. This can be saved/restored later.
        
        Returns:
            dict: Content dictionary with keys:
                - 'selected_value': Currently selected value string
                - 'metadata_state': Dict of metadata overrides indexed by selection key
        """
        return {
            'selected_value': self.get_selected_value(),
            'metadata_state': copy.deepcopy(self.selection_metadata_state),
        }
    
    def saveContent(self, path):
        """Save the current selection and metadata state to a JSON file.
        
        Serializes getContent() to a JSON file. If path doesn't end with .json,
        the extension is automatically appended.
        
        Args:
            path (str): File path to save to. Will be suffixed with '.json' if needed.
        """
        ext = '.json'
        if not path.endswith(ext):
            path += ext
        content = self.getContent()
        with open(path, 'w') as f:
            json.dump(content, f, indent=2)
            
    def new_image_loaded(self, posData=None):
        """Update the available options when new image data is loaded.

        Called when the workflow's image data changes to refresh available selection
        options. Repopulates the combo box with updated channels from the parent
        WorkflowGui. If the previously confirmed selection is no longer available
        it is added back with red text to indicate the issue, and
        ``sigSelectionInvalid`` is emitted to alert the user.
        
        Keeps metadata already restored from workflow content for still-valid
        selections so reopening/loading the UI does not discard saved overrides.
        """
        if posData is not None:
            self.posData = posData

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
        prev_metadata_state = dict(self.selection_metadata_state)

        self.selection_widget.blockSignals(True)
        self.selection_widget.clear()
        self.selection_options = new_options
        self.selection_widget.addItems(self.selection_options)
        self.selection_widget.blockSignals(False)

        # Keep metadata for choices that remain available plus file-based
        # selections, so workflow-loaded metadata is not lost on refresh.
        self.selection_metadata_state = {
            key: self._normalizeMetadataState(value)
            for key, value in prev_metadata_state.items()
            if key in self.selection_options or os.path.isfile(key)
        }

        if prev_selection is not None:
            if prev_selection in self.selection_options:
                self.selection_widget.setCurrentIndex(
                    self.selection_options.index(prev_selection)
                )
            elif os.path.isfile(prev_selection):
                idx = self._ensureSelectionItem(prev_selection)
                self.selection_widget.setCurrentIndex(idx)
            else:
                # Previous value no longer available – add it back in red
                invalid_idx = self._ensureSelectionItem(prev_selection, color='red')
                self.selection_widget.setCurrentIndex(invalid_idx)
                self.sigSelectionInvalid.emit(prev_selection)

    def loadContent(self, path):
        """Load the selection and metadata state from a file and update the dialog.
        
        Loads saved content from JSON file (or legacy .txt file from older versions).
        Calls setContent() to restore the dialog state, then emits workflow signals.
        
        Shows the dialog, loads the content, triggers ok_clicked() to update workflow
        outputs based on loaded selection, then hides the dialog.
        
        Handles loading errors gracefully - logs warnings but continues without blocking.
        
        Args:
            path (str): File path to load from. Can be .json or .txt, extension added if missing.
        """
        ext = '.json'
        self.show()
        try:
            json_path = path if path.endswith(ext) else f'{path}{ext}'
            with open(json_path, 'r') as f:
                content = json.load(f)
            self.setContent(content)
         
        except Exception as e:
            self.logger.info(f"Failed to load content from {path}: {e}")
            traceback.print_exc()
        self.ok_clicked()  # To emit signals and update the workflow card state after loading
        self.hide()

class ZStackManipulationDialogWorkflow(QBaseDialog):
    """Dialog for selecting z-stack conversion operations."""
    #TODO Need to add automatic loading from image folder
    sigOkClicked = Signal()
    sigCancelClicked = Signal()
    sigSetOutputs = Signal(dict)
    sigSelectionInvalid = Signal(str)
    sigSelectionValid = Signal()

    def __init__(self, parent=None, logger=printl, posData=None):
        super().__init__(parent=parent)
        self.setWindowTitle('Z-stack manipulation')
        self.logger = logger
        self.pos_data = posData
        self.curr_input_types = {0: None}
        self._browse_source_file = ''
        self.z_stack_currently_selected = None

        layout = QVBoxLayout(self)

        expand_label = QLabel('Expand 2D to 3D:')
        self.expand2dTo3dSelector = QComboBox(self)
        self.expand2dTo3dSelector.addItem('Do not expand', 'none')
        self.expand2dTo3dSelector.addItem('Repeat single plane to stack', 'repeat')

        expand_slices_layout = QHBoxLayout()
        self.numExpandSlicesLabel = QLabel('Number of z-slices:')
        self.numExpandSlicesSpinbox = QSpinBox(self)
        self.numExpandSlicesSpinbox.setMinimum(2)
        self.numExpandSlicesSpinbox.setMaximum(9999)
        self.numExpandSlicesSpinbox.setValue(2)
        expand_slices_layout.addWidget(self.numExpandSlicesLabel)
        expand_slices_layout.addWidget(self.numExpandSlicesSpinbox)
        expand_slices_layout.addStretch(1)

        collapse_label = QLabel('Convert 3D to 2D:')
        self.collapse3dTo2dSelector = QComboBox(self)
        self.collapse3dTo2dSelector.addItem('Do not collapse', 'none')
        self.collapse3dTo2dSelector.addItem('Maximum projection', 'max')
        self.collapse3dTo2dSelector.addItem('Mean projection', 'mean')
        self.collapse3dTo2dSelector.addItem('Median projection', 'median')
        # 'single_plane_data_prep_loaded' added dynamically via _refreshFromLoadedFolderOption
        self.collapse3dTo2dSelector.addItem(
            'Single plane (data prep) with browse option',
            'single_plane_data_prep_browse'
        )
        self.collapse3dTo2dSelector.addItem(
            'Single plane from specified z-slice',
            'single_plane_z_slice'
        )

        single_plane_layout = QHBoxLayout()
        self.singlePlaneZSliceLabel = QLabel('Single plane z-slice:')
        single_plane_layout.addWidget(self.singlePlaneZSliceLabel)
        self.singlePlaneZSliceSpinbox = QSpinBox(self)
        self.singlePlaneZSliceSpinbox.setMinimum(1)
        self.singlePlaneZSliceSpinbox.setMaximum(1)
        single_plane_layout.addWidget(self.singlePlaneZSliceSpinbox)

        self.browseDataPrepSourceButton = QPushButton('Browse source...')
        self.browseDataPrepSourceButton.clicked.connect(self._browseDataPrepSource)
        single_plane_layout.addWidget(self.browseDataPrepSourceButton)
        self.singlePlaneLayout = single_plane_layout
        
        # add filename selector so when there are several filepaths in the data prep csv, user can choose which one to use for z-slice selection
        self.filename_selector = QComboBox(self)
        single_plane_layout.addWidget(self.filename_selector)
        # set minimum stretch for filename since its init to empty and would be super small
        single_plane_layout.setStretch(single_plane_layout.count() - 1, 1)
        
        layout.addWidget(expand_label)
        layout.addWidget(self.expand2dTo3dSelector)
        layout.addLayout(expand_slices_layout)
        layout.addWidget(collapse_label)
        layout.addWidget(self.collapse3dTo2dSelector)
        layout.addLayout(single_plane_layout)

        self.browseDataPrepSourcePathLabel = QLabel('Selected source: (none)')
        self.browseDataPrepSourcePathLabel.setWordWrap(True)
        layout.addWidget(self.browseDataPrepSourcePathLabel)

        self.selectedZSliceInfoLabel = QLabel('')
        self.selectedZSliceInfoLabel.setWordWrap(True)
        layout.addWidget(self.selectedZSliceInfoLabel)

        buttons_layout = widgets.CancelOkButtonsLayout()
        self.ok_button = buttons_layout.okButton
        self.cancel_button = buttons_layout.cancelButton
        self.ok_button.clicked.connect(self.ok_clicked)
        self.cancel_button.clicked.connect(self.cancel_clicked)
        layout.addLayout(buttons_layout)

        self.expand2dTo3dSelector.currentIndexChanged.connect(self._emitOutputType)
        self.expand2dTo3dSelector.currentIndexChanged.connect(
            self._updateExpandSlicesVisibility
            )
        self.collapse3dTo2dSelector.currentIndexChanged.connect(self._emitOutputType)
        self.collapse3dTo2dSelector.currentIndexChanged.connect(
            self._updateSinglePlaneControlsVisibility
        )
        self.collapse3dTo2dSelector.currentIndexChanged.connect(
            self._emitSelectionValidity
        )
        self.singlePlaneZSliceSpinbox.valueChanged.connect(self._emitSelectionValidity)
        self.singlePlaneZSliceSpinbox.valueChanged.connect(self._updateSelectedZSliceLabel)
        self.filename_selector.currentIndexChanged.connect(self._emitSelectionValidity)
        self.filename_selector.currentIndexChanged.connect(self._updateSelectedZSliceLabel)

        self._refreshFromLoadedFolderOption(posData)
        self._updateSinglePlaneControlsVisibility()
        self._updateExpandSlicesVisibility()
        self._updateBrowseSourcePathLabel()
        self._updateSelectorStates()
        self._updateSelectedZSliceLabel()

    def _selectedExpandMode(self):
        return self.expand2dTo3dSelector.currentData()

    def _selectedCollapseMode(self):
        return self.collapse3dTo2dSelector.currentData()

    def _usesSinglePlane(self):
        return self._selectedCollapseMode() in (
            'single_plane_z_slice'
        )

    def _updateSinglePlaneControlsVisibility(self):
        show_single_plane = self._usesSinglePlane()
        self.singlePlaneZSliceLabel.setVisible(show_single_plane)
        self.singlePlaneZSliceSpinbox.setVisible(show_single_plane)
        show_browse = self._selectedCollapseMode() == 'single_plane_data_prep_browse'
        self.browseDataPrepSourceButton.setVisible(show_browse)
        self.browseDataPrepSourcePathLabel.setVisible(show_browse)
        self.filename_selector.setVisible(show_browse)
        self._updateSelectedZSliceLabel()

    def _updateExpandSlicesVisibility(self):
        show = self._selectedExpandMode() != 'none'
        self.numExpandSlicesLabel.setVisible(show)
        self.numExpandSlicesSpinbox.setVisible(show)

    def _updateBrowseSourcePathLabel(self):
        selected = str(self._browse_source_file or '').strip()
        if selected:
            self.browseDataPrepSourcePathLabel.setText(f'Selected source: {selected}')
        else:
            self.browseDataPrepSourcePathLabel.setText('Selected source: (none)')

    def _updateSelectedZSliceLabel(self):
        mode = self._selectedCollapseMode()
        if mode == 'single_plane_z_slice':
            val = self.singlePlaneZSliceSpinbox.value()
            max_z = self._maxZFromInput()
            self.selectedZSliceInfoLabel.setText(f'Selected z-slice: {val} / {max_z}')
            self.selectedZSliceInfoLabel.setVisible(True)
            self.z_stack_currently_selected = val
        elif mode == 'single_plane_data_prep_browse':
            source = self._browse_source_file
            selected_filename = self.filename_selector.currentText()
            if source and os.path.isfile(source) and selected_filename:
                try:
                    df = pd.read_csv(source)
                    rows = df[df['filename'] == selected_filename]
                    z_vals = None
                    for col in ('z_slice_used_dataPrep', 'z_slice_used_gui'):
                        if col not in rows.columns:
                            continue
                        vals = pd.to_numeric(rows[col], errors='coerce').dropna().astype(int)
                        if not vals.empty:
                            z_vals = vals.tolist()
                            break
                    if z_vals is not None:
                        max_z = self._maxZFromInput()
                        if len(z_vals) == 1:
                            text = f'Z-slice from source: {z_vals[0]} / {max_z}'
                        else:
                            text = f'Z-slices from source: {z_vals[0]}\u2013{z_vals[-1]} ({len(z_vals)} frames) / {max_z}'
                        self.selectedZSliceInfoLabel.setText(text)
                        self.selectedZSliceInfoLabel.setVisible(True)
                        self.z_stack_currently_selected = z_vals if len(z_vals) > 1 else z_vals[0]
                        return
                except Exception:
                    pass
            self.selectedZSliceInfoLabel.setText('')
            self.selectedZSliceInfoLabel.setVisible(False)
            self.z_stack_currently_selected = None
        elif mode == 'single_plane_data_prep_loaded':
            self.selectedZSliceInfoLabel.setText('Z-slice: from loaded data prep')
            self.selectedZSliceInfoLabel.setVisible(True)
            self.z_stack_currently_selected = None
        else:
            self.selectedZSliceInfoLabel.setText('')
            self.selectedZSliceInfoLabel.setVisible(False)
            self.z_stack_currently_selected = None

    def _maxZFromInput(self):
        in_type = self.curr_input_types.get(0)
        size_z = getattr(in_type, 'SizeZ', None) if is_workflow_data_class(in_type) else None
        if size_z is None:
            size_z = getattr(self.pos_data, 'SizeZ', None)
        if size_z is None:
            return 1
        try:
            return max(1, int(size_z))
        except Exception:
            return 1

    def _setSinglePlaneBounds(self, size_z=None):
        max_z = self._maxZFromInput() if size_z is None else max(1, int(size_z))
        self.singlePlaneZSliceSpinbox.setMaximum(max_z)
        if self.singlePlaneZSliceSpinbox.value() > max_z:
            self.singlePlaneZSliceSpinbox.setValue(max_z)

    def _ensureSegmInfo(self, pos_data):
        """Return True if pos_data has enough attributes to read/write segmInfo."""
        if pos_data is None:
            return False
        return (
            getattr(pos_data, 'segmInfo_df_csv_path', None) is not None
            and getattr(pos_data, 'filename', None) is not None
        )

    def _browseDataPrepSource(self):
        start_dir = getattr(self.pos_data, 'images_path', '') or ''
        if not start_dir:
            start_dir = myutils.getMostRecentPath()

        # csv
        filepath = getopenfilename(
            parent=self,
            caption='Select data prep source file',
            basedir=start_dir,
            filters='CSV files (*.csv);;All Files (*)'
        )[0]
        if not filepath:
            return

        try:
            # Read the selected CSV directly and infer z-stack depth from
            # any z-slice columns that store zero-based slice indices.
            df = pd.read_csv(filepath)
            filenames = df['filename']
            # check if multiple filenames are present
            filenames_unique = filenames.unique()
            # add selector options for each unique filename if multiple are present
            self.filename_selector.clear()
            self.filename_selector.addItems(filenames_unique)
            self._browse_source_file = filepath
            self._updateBrowseSourcePathLabel()
            self._updateSelectedZSliceLabel()
            self._emitSelectionValidity()

        except Exception as err:
            self.logger.info(f'Failed loading browse source "{filepath}": {err}')

    def _computeOutputType(self):
        in_type = self.curr_input_types.get(0)
        if not is_workflow_data_class(in_type):
            return WfImageDC()

        type_name = workflow_type_name(in_type)
        size_t = getattr(in_type, 'SizeT', None)
        size_z = getattr(in_type, 'SizeZ', None)
        size_y = getattr(in_type, 'SizeY', None)
        size_x = getattr(in_type, 'SizeX', None)

        is_3d = size_z is not None and size_z > 1
        if is_3d:
            if self._selectedCollapseMode() != 'none':
                size_z = 1
        else:
            if self._selectedExpandMode() != 'none':
                    size_z = self.numExpandSlicesSpinbox.value()

        return make_workflow_data_class(
            type_name,
            SizeT=size_t,
            SizeZ=size_z,
            SizeY=size_y,
            SizeX=size_x,
        )

    def _emitOutputType(self):
        self.sigSetOutputs.emit({0: self._computeOutputType()})

    def _refreshFromLoadedFolderOption(self, hint_pos_data=None):
        """Add or remove the 'from loaded folder' item depending on loaded data."""
        key = 'single_plane_data_prep_loaded'
        idx = self.collapse3dTo2dSelector.findData(key)

        # Check loaded positions via parent, fall back to the hint passed directly
        has_valid = self._ensureSegmInfo(getattr(self.parent(), 'posData', None)) or self._ensureSegmInfo(hint_pos_data)

        if has_valid and idx == -1:
            # Insert right after 'median'
            after = self.collapse3dTo2dSelector.findData('median')
            self.collapse3dTo2dSelector.insertItem(
                after + 1,
                'Single plane (data prep) from loaded folder',
                key,
            )
        elif not has_valid and idx != -1:
            # If currently selected, reset to 'none' before removing
            if self.collapse3dTo2dSelector.currentData() == key:
                self.collapse3dTo2dSelector.setCurrentIndex(0)
            self.collapse3dTo2dSelector.removeItem(
                self.collapse3dTo2dSelector.findData(key)
            )

    def _updateSelectorStates(self):
        in_type = self.curr_input_types.get(0)
        size_z = getattr(in_type, 'SizeZ', None) if is_workflow_data_class(in_type) else None
        is_3d = size_z is not None and int(size_z) > 1
        # Collapse only makes sense for 3D input; expand only for 2D input
        self.collapse3dTo2dSelector.setEnabled(is_3d)
        self.expand2dTo3dSelector.setEnabled(not is_3d)

    def _zSliceOutOfBoundsMessage(self):
        """Return an error message if z-slice values in the browse CSV exceed input bounds."""
        max_z = self._maxZFromInput()
        source = self._browse_source_file
        if not (source and os.path.isfile(source)):
            return None
        try:
            df = pd.read_csv(source)
            selected_filename = self.filename_selector.currentText()
            rows = df[df['filename'] == selected_filename] if selected_filename else df
            for col in ('z_slice_used_dataPrep', 'z_slice_used_gui'):
                if col not in rows.columns:
                    continue
                z_vals = pd.to_numeric(rows[col], errors='coerce').dropna().astype(int)
                if z_vals.empty:
                    continue
                z_max_found = int(z_vals.max())
                # z_slice values are 0-indexed; valid range is 0 to SizeZ-1
                if z_max_found >= max_z:
                    return (
                        f'Z-slice index {z_max_found} (0-indexed) in the source file '
                        f'exceeds the input SizeZ ({max_z}). '
                        'Please select a compatible source file or fix the input.'
                    )
                break
        except Exception:
            pass
        return None

    def _isSelectionValid(self):
        mode = self._selectedCollapseMode()
        if mode == 'single_plane_data_prep_browse':
            if not (self._browse_source_file and os.path.isfile(self._browse_source_file)):
                return False
            if self._zSliceOutOfBoundsMessage() is not None:
                return False
        elif mode == 'single_plane_z_slice':
            if self.singlePlaneZSliceSpinbox.value() > self._maxZFromInput():
                return False
        return True

    def _emitSelectionValidity(self):
        mode = self._selectedCollapseMode()
        if mode == 'single_plane_data_prep_browse':
            if not (self._browse_source_file and os.path.isfile(self._browse_source_file)):
                self.sigSelectionInvalid.emit(
                    'Browse mode is selected but no valid source file has been set. '
                    'Please click "Browse source..." to select a data prep CSV file.'
                )
                return
            oob_msg = self._zSliceOutOfBoundsMessage()
            if oob_msg:
                self.sigSelectionInvalid.emit(oob_msg)
                return
        elif mode == 'single_plane_z_slice':
            max_z = self._maxZFromInput()
            val = self.singlePlaneZSliceSpinbox.value()
            if val > max_z:
                self.sigSelectionInvalid.emit(
                    f'Selected z-slice {val} exceeds input SizeZ ({max_z}). '
                    'Please select a z-slice within the input range.'
                )
                return
        self.sigSelectionValid.emit()

    def updatedInputTypes(self):
        self._refreshFromLoadedFolderOption()
        self._updateSelectorStates()
        self._setSinglePlaneBounds()
        self._emitSelectionValidity()
        self._emitOutputType()

    def ok_clicked(self):
        if not self._isSelectionValid():
            self._emitSelectionValidity()
            return
        self._emitOutputType()
        self.sigOkClicked.emit()
        self.hide()

    def cancel_clicked(self):
        self.sigCancelClicked.emit()
        self.hide()

    def closeEvent(self, event):
        self.sigCancelClicked.emit()
        super().closeEvent(event)

    def getContent(self):
        single_plane_browse_values = None
        selected_filename = self.filename_selector.currentText() if self.filename_selector.count() > 0 else ''
        if selected_filename:
            csv = pd.read_csv(self._browse_source_file)
            rows = csv[csv['filename'] == selected_filename]
            for col in ('z_slice_used_dataPrep', 'z_slice_used_gui'):
                if col not in rows.columns:
                    continue
                z_vals = pd.to_numeric(rows[col], errors='coerce').dropna()
                if z_vals.empty:
                    continue
                # set the z slices to use
                single_plane_browse_values = z_vals.astype(int).tolist()
                break
            
        return {
            'expand_2d_to_3d': self._selectedExpandMode(),
            'collapse_3d_to_2d': self._selectedCollapseMode(),
            'single_plane_z_slice': int(self.singlePlaneZSliceSpinbox.value()),
            'single_plane_browse_source': self._browse_source_file,
            'single_plane_browse_values': single_plane_browse_values,
                'num_expand_slices': int(self.numExpandSlicesSpinbox.value()),
                'single_plane_browse_selected_filename': self.filename_selector.currentText(),
            'z_stack_currently_selected': self.z_stack_currently_selected,
        }

    def setContent(self, content):
        if not isinstance(content, dict):
            return

        expand_mode = content.get('expand_2d_to_3d', 'none')
        collapse_mode = content.get('collapse_3d_to_2d', 'none')
        single_plane_z_slice = int(content.get('single_plane_z_slice', 1) or 1)
        browse_source = str(content.get('single_plane_browse_source', '') or '')
        single_plane_browse_values = content.get('single_plane_browse_values', None)
        num_expand_slices = int(content.get('num_expand_slices', 2) or 2)
        selected_filename = str(content.get('single_plane_browse_selected_filename', '') or '')

        expand_idx = self.expand2dTo3dSelector.findData(expand_mode)
        if expand_idx >= 0:
            self.expand2dTo3dSelector.setCurrentIndex(expand_idx)

        collapse_idx = self.collapse3dTo2dSelector.findData(collapse_mode)
        if collapse_idx >= 0:
            self.collapse3dTo2dSelector.setCurrentIndex(collapse_idx)

        self.singlePlaneZSliceSpinbox.setValue(max(1, single_plane_z_slice))
        self.numExpandSlicesSpinbox.setValue(max(2, num_expand_slices))
        self._browse_source_file = browse_source
        self._updateBrowseSourcePathLabel()
        self._updateSinglePlaneControlsVisibility()
        self._updateExpandSlicesVisibility()

        # Repopulate filename_selector from the stored CSV
        if browse_source and os.path.isfile(browse_source):
            try:
                df = pd.read_csv(browse_source)
                filenames_unique = df['filename'].unique()
                self.filename_selector.blockSignals(True)
                self.filename_selector.clear()
                self.filename_selector.addItems(filenames_unique)
                self.filename_selector.blockSignals(False)
                if selected_filename:
                    idx = self.filename_selector.findText(selected_filename)
                    if idx >= 0:
                        self.filename_selector.setCurrentIndex(idx)
            except Exception as err:
                self.logger.info(f'Failed repopulating filename selector from "{browse_source}": {err}')

    def saveContent(self, path):
        ext = '.json'
        if not path.endswith(ext):
            path += ext

        with open(path, 'w') as f:
            json.dump(self.getContent(), f, indent=2)

    def loadContent(self, path):
        ext = '.json'
        try:
            json_path = path if path.endswith(ext) else f'{path}{ext}'
            with open(json_path, 'r') as f:
                content = json.load(f)
            self.setContent(content)
        except Exception as err:
            self.logger.info(f'Failed to load z-stack card content from {path}: {err}')
            return

        self._emitOutputType()

class CropImagesDialogWorkflow(QBaseDialog):
    """Dialog for selecting crop operations from data-prep metadata."""
    sigOkClicked = Signal()
    sigCancelClicked = Signal()
    sigSetOutputs = Signal(dict)
    sigSelectionInvalid = Signal(str)
    sigSelectionValid = Signal()

    def __init__(self, parent=None, logger=printl, posData=None):
        super().__init__(parent=parent)
        self.setWindowTitle('Crop images')
        self.logger = logger
        self._default_pos_data = posData
        self.curr_input_types = {0: None}
        self._crop_browse_source_file = ''
        self._crop_browse_source_df = None
        self._pending_selected_roi_id = None
        self._current_roi_source_key = None
        self.current_roi_coords = None

        layout = QVBoxLayout(self)

        crop_label = QLabel('Crop source:')
        self.cropImagesSelector = QComboBox(self)
        self.cropImagesSelector.addItem('Do not crop', 'none')
        # 'crop_data_prep_loaded' added dynamically via _refreshCropFromLoadedFolderOption
        self.cropImagesSelector.addItem(
            'Use crop from browsed source',
            'crop_data_browse'
        )

        crop_controls_layout = QHBoxLayout()
        self.browseCropSourceButton = QPushButton('Browse crop source...')
        self.browseCropSourceButton.clicked.connect(self._browseCropDataPrepSource)
        crop_controls_layout.addWidget(self.browseCropSourceButton)
        crop_controls_layout.addStretch(1)

        layout.addWidget(crop_label)
        layout.addWidget(self.cropImagesSelector)
        layout.addLayout(crop_controls_layout)

        self.browseCropSourcePathLabel = QLabel('Selected source: (none)')
        self.browseCropSourcePathLabel.setWordWrap(True)
        layout.addWidget(self.browseCropSourcePathLabel)

        roi_layout = QHBoxLayout()
        self.roiSelectorLabel = QLabel('ROI:')
        roi_layout.addWidget(self.roiSelectorLabel)
        self.roiSelector = QComboBox(self)
        self.roiSelector.currentIndexChanged.connect(self._onRoiSelectionChanged)
        self.roiSelector.setDisabled(True)
        roi_layout.addWidget(self.roiSelector)
        layout.addLayout(roi_layout)

        self.roiInfoLabel = QLabel('')
        self.roiInfoLabel.setVisible(False)
        layout.addWidget(self.roiInfoLabel)

        buttons_layout = widgets.CancelOkButtonsLayout()
        self.ok_button = buttons_layout.okButton
        self.cancel_button = buttons_layout.cancelButton
        self.ok_button.clicked.connect(self.ok_clicked)
        self.cancel_button.clicked.connect(self.cancel_clicked)
        layout.addLayout(buttons_layout)

        self.cropImagesSelector.currentIndexChanged.connect(self._emitOutputType)
        self.cropImagesSelector.currentIndexChanged.connect(
            self._updateBrowseControlsVisibility
        )
        self.cropImagesSelector.currentIndexChanged.connect(
            self._updateRoiSelectorOptions
        )

        self._refreshCropFromLoadedFolderOption(posData)
        self._updateBrowseControlsVisibility()
        self._updateCropBrowsePathLabel()
        self._updateRoiSelectorOptions()
        self._updateRoiInfoLabel()
        self._emitOutputType()

    def _selectedCropMode(self):
        return self.cropImagesSelector.currentData()

    def _updateBrowseControlsVisibility(self):
        show_browse = self._selectedCropMode() == 'crop_data_browse'
        self.browseCropSourceButton.setVisible(show_browse)
        self.browseCropSourcePathLabel.setVisible(show_browse)

    def _updateCropBrowsePathLabel(self):
        selected = str(self._crop_browse_source_file or '').strip()
        if selected:
            self.browseCropSourcePathLabel.setText(f'Selected source: {selected}')
        else:
            self.browseCropSourcePathLabel.setText('Selected source: (none)')

    def _currentPosData(self, hint_pos_data=None):
        if hint_pos_data is not None:
            return hint_pos_data

        parent = self.parent()
        parent_pos_data = getattr(parent, 'posData', None) if parent is not None else None
        if parent_pos_data is not None:
            return parent_pos_data

        return self._default_pos_data

    def _roiSourceKey(self):
        """Return a hashable identity for the current ROI data source."""
        mode = self._selectedCropMode()
        if mode == 'crop_data_prep_loaded':
            pos_data = self._currentPosData()
            return id(pos_data) if pos_data is not None else None
        if mode == 'crop_data_browse':
            return self._crop_browse_source_file or None
        return None

    def _resolveRoiSource(self):
        """Return the active ROI dataframe and ordered ROI IDs for the current mode."""
        mode = self._selectedCropMode()
        if mode == 'crop_data_prep_loaded':
            pos_data = self._currentPosData()
            roi_df = getattr(pos_data, 'dataPrep_ROIcoords', None) if pos_data is not None else None
        elif mode == 'crop_data_browse':
            roi_df = self._crop_browse_source_df
        else:
            roi_df = None

        if roi_df is None or roi_df.empty or not isinstance(roi_df, pd.DataFrame):
            return None, []

        if isinstance(roi_df.index, pd.MultiIndex):
            roi_ids = roi_df.index.get_level_values(0).tolist()
        else:
            roi_ids = [0]

        # Preserve order while removing duplicates.
        return roi_df, list(dict.fromkeys(roi_ids))

    def _roiCoordsFromDataFrame(self, roi_df, roi_id=None):
        """Return a dict with the roi coordinates and crop size, or None."""
        if roi_df is None or roi_df.empty or not isinstance(roi_df, pd.DataFrame):
            return None

        if isinstance(roi_df.index, pd.MultiIndex):
            if roi_id is None:
                roi_id = roi_df.index.get_level_values(0).unique()[0]
            df_roi = roi_df.xs(roi_id, level=0)
        else:
            df_roi = roi_df

        if not isinstance(df_roi, pd.DataFrame) or 'value' not in df_roi.columns:
            return None

        x_left = int(float(df_roi.at['x_left', 'value']))
        x_right = int(float(df_roi.at['x_right', 'value']))
        y_top = int(float(df_roi.at['y_top', 'value']))
        y_bottom = int(float(df_roi.at['y_bottom', 'value']))
        return {
            'x_left': x_left, 'x_right': x_right,
            'y_top': y_top, 'y_bottom': y_bottom,
            'size_x': max(1, x_right - x_left),
            'size_y': max(1, y_bottom - y_top),
        }

    def _updateCurrentRoiCoords(self):
        roi_df, _ = self._resolveRoiSource()
        coords = self._roiCoordsFromDataFrame(
            roi_df,
            roi_id=self._selectedRoiId(),
        )
        self.current_roi_coords = coords
        return coords

    def _updateRoiInfoLabel(self):
        coords = self._updateCurrentRoiCoords()
        if coords is None:
            self.roiInfoLabel.setText('')
            self.roiInfoLabel.setVisible(False)
            return
        self.roiInfoLabel.setText(
            f'x: {coords["x_left"]}\u2013{coords["x_right"]},  '
            f'y: {coords["y_top"]}\u2013{coords["y_bottom"]}  '
            f'({coords["size_x"]} \u00d7 {coords["size_y"]} px)'
        )
        self.roiInfoLabel.setVisible(True)

    def _selectedRoiId(self):
        if not self.roiSelector.isVisible() or self.roiSelector.count() == 0:
            return None
        return self.roiSelector.currentData()

    def _updateRoiSelectorOptions(self, *_args):
        prev_roi_id = self._selectedRoiId()
        _, roi_ids = self._resolveRoiSource()

        self.roiSelector.blockSignals(True)
        self.roiSelector.clear()
        for roi_id in roi_ids:
            self.roiSelector.addItem(str(roi_id), roi_id)

        if roi_ids:
            selected_roi_id = self._pending_selected_roi_id
            if selected_roi_id is None and prev_roi_id in roi_ids:
                selected_roi_id = prev_roi_id
            if selected_roi_id not in roi_ids:
                selected_roi_id = roi_ids[0]

            idx = self.roiSelector.findData(selected_roi_id)
            if idx < 0:
                idx = 0
            self.roiSelector.setCurrentIndex(idx)
            self.roiSelector.setDisabled(False)
        else:
            self.roiSelector.setDisabled(True)

        self.roiSelector.blockSignals(False)
        self._pending_selected_roi_id = None
        self._updateRoiInfoLabel()

    def _cropShapeFromCurrentRoi(self):
        # Keep ROI coordinates in sync during upstream metadata propagation,
        # not only when the user changes comboboxes.
        coords = self._updateCurrentRoiCoords()
        if not coords:
            return None
        return coords['size_y'], coords['size_x']

    def _dataPrepROIcoordsPath(self, pos_data):
        if pos_data is None:
            return None
        roi_path = getattr(pos_data, 'dataPrepROI_coords_path', None)
        if roi_path:
            return roi_path

        images_path = getattr(pos_data, 'images_path', None)
        basename = getattr(pos_data, 'basename', None)
        if images_path and basename:
            return os.path.join(images_path, f'{basename}dataPrepROIs_coords.csv')
        return None

    def _dataPrepFreeRoiPath(self, pos_data):
        if pos_data is None:
            return None
        if hasattr(pos_data, 'dataPrepFreeRoiPath'):
            try:
                return pos_data.dataPrepFreeRoiPath()
            except Exception:
                pass

        images_path = getattr(pos_data, 'images_path', None)
        basename = getattr(pos_data, 'basename', None)
        if images_path and basename:
            return os.path.join(images_path, f'{basename}dataPrepFreeRoi.npz')
        return None

    def _hasDataPrepCrop(self, pos_data):
        if pos_data is None:
            return False

        roi_df = getattr(pos_data, 'dataPrep_ROIcoords', None)
        if roi_df is not None and not roi_df.empty:
            return True

        roi_path = self._dataPrepROIcoordsPath(pos_data)
        if roi_path and os.path.exists(roi_path):
            return True

        free_points = getattr(pos_data, 'dataPrepFreeRoiPoints', None)
        if free_points is not None and len(free_points) > 0:
            return True

        free_path = self._dataPrepFreeRoiPath(pos_data)
        return bool(free_path and os.path.exists(free_path))

    def _inputSpatialShape(self):
        in_type = self.curr_input_types.get(0)
        size_y = getattr(in_type, 'SizeY', None) if is_workflow_data_class(in_type) else None
        size_x = getattr(in_type, 'SizeX', None) if is_workflow_data_class(in_type) else None

        if size_y is None or size_x is None:
            current_pos_data = self._currentPosData()
            size_y = getattr(current_pos_data, 'SizeY', None)
            size_x = getattr(current_pos_data, 'SizeX', None)

        try:
            size_y = int(size_y) if size_y is not None else None
            size_x = int(size_x) if size_x is not None else None
        except Exception:
            return None

        if size_y is None or size_x is None:
            return None
        return max(1, size_y), max(1, size_x)

    def _isSelectedRoiCropWithinInput(self):
        coords = self._updateCurrentRoiCoords()
        if coords is None:
            return True

        input_shape = self._inputSpatialShape()
        if input_shape is None:
            return True

        input_y, input_x = input_shape
        x_left = int(coords['x_left'])
        x_right = int(coords['x_right'])
        y_top = int(coords['y_top'])
        y_bottom = int(coords['y_bottom'])

        return (
            x_left >= 0
            and y_top >= 0
            and x_right <= input_x
            and y_bottom <= input_y
            and x_right > x_left
            and y_bottom > y_top
        )

    def _warnIfSelectedRoiLargerThanInput(self):
        if self._isSelectedRoiCropWithinInput():
            return False

        coords = self._updateCurrentRoiCoords()
        crop_shape = self._cropShapeFromCurrentRoi()
        input_shape = self._inputSpatialShape()
        if coords is None or crop_shape is None or input_shape is None:
            return False

        crop_y, crop_x = crop_shape
        input_y, input_x = input_shape
        self.sigSelectionInvalid.emit(
            (
                'The selected ROI bounds '
                f'(x: {coords["x_left"]}-{coords["x_right"]}, '
                f'y: {coords["y_top"]}-{coords["y_bottom"]}) are not fully '
                f'within the current input bounds (x: 0-{input_x}, y: 0-{input_y}). '
                f'ROI size is ({crop_y} x {crop_x}). '
                'Please select a different ROI or disable crop.'
            )
        )
        return True

    def _onRoiSelectionChanged(self, *_args):
        self._updateRoiInfoLabel()
        self._emitCropValidity()
        self._emitOutputType()

    def _browseCropDataPrepSource(self):
        start_dir = getattr(self._currentPosData(), 'images_path', '') or ''
        if not start_dir:
            start_dir = myutils.getMostRecentPath()

        filepath = getopenfilename(
            parent=self,
            caption='Select crop source file',
            basedir=start_dir,
            filters='CSV files (*.csv);;All Files (*)'
        )[0]
        if not filepath:
            return

        self._crop_browse_source_file = filepath
        self._updateCropBrowsePathLabel()
        try:
            df = pd.read_csv(filepath)
            if 'roi_id' not in df.columns:
                df['roi_id'] = 0

            if 'description' not in df.columns or 'value' not in df.columns:
                raise ValueError(
                    'CSV must contain at least "description" and "value" columns.'
                )

            df = df.set_index(['roi_id', 'description'])
            self._crop_browse_source_df = df
        except Exception as err:
            self._crop_browse_source_df = None
            self.logger.info(f'Failed loading crop source "{filepath}": {err}')

        self._updateRoiSelectorOptions()
        self._emitOutputType()

    def _computeOutputType(self):
        in_type = self.curr_input_types.get(0)
        if not is_workflow_data_class(in_type):
            return WfImageDC()

        type_name = workflow_type_name(in_type)
        size_t = getattr(in_type, 'SizeT', None)
        size_z = getattr(in_type, 'SizeZ', None)
        size_y = getattr(in_type, 'SizeY', None)
        size_x = getattr(in_type, 'SizeX', None)

        crop_shape = self._cropShapeFromCurrentRoi()
        if crop_shape is not None and self._isSelectedRoiCropWithinInput():
            size_y, size_x = crop_shape

        return make_workflow_data_class(
            type_name,
            SizeT=size_t,
            SizeZ=size_z,
            SizeY=size_y,
            SizeX=size_x,
        )

    def _emitOutputType(self):
        self.sigSetOutputs.emit({0: self._computeOutputType()})

    def _refreshCropFromLoadedFolderOption(self, hint_pos_data=None):
        """Add or remove the crop item that pulls settings from loaded data."""
        key = 'crop_data_prep_loaded'
        idx = self.cropImagesSelector.findData(key)

        has_valid = self._hasDataPrepCrop(self._currentPosData(hint_pos_data))

        if has_valid and idx == -1:
            self.cropImagesSelector.insertItem(
                1,
                'Use crop from loaded image',
                key,
            )
        elif not has_valid and idx != -1:
            if self.cropImagesSelector.currentData() == key:
                self.cropImagesSelector.setCurrentIndex(0)
            self.cropImagesSelector.removeItem(self.cropImagesSelector.findData(key))

    def _emitCropValidity(self):
        if self._warnIfSelectedRoiLargerThanInput():
            return False
        self.sigSelectionValid.emit()
        return True

    def new_image_loaded(self, posData=None):
        if posData is not None:
            self._default_pos_data = posData
            if self._selectedCropMode() != 'crop_data_browse':
                self._crop_browse_source_df = None
        else:
            current_pos_data = self._currentPosData()
            if current_pos_data is not None:
                self._default_pos_data = current_pos_data
                if self._selectedCropMode() != 'crop_data_browse':
                    self._crop_browse_source_df = None

        self._refreshCropFromLoadedFolderOption(self._default_pos_data)
        self._updateRoiSelectorOptions()
        self._emitCropValidity()
        self._emitOutputType()

    def updatedInputTypes(self):
        self._refreshCropFromLoadedFolderOption()
        self._updateBrowseControlsVisibility()
        self._updateRoiSelectorOptions()
        self._emitCropValidity()
        self._emitOutputType()

    def ok_clicked(self):
        if not self._emitCropValidity():
            return

        self._emitOutputType()
        self.sigOkClicked.emit()
        self.hide()

    def cancel_clicked(self):
        self.sigCancelClicked.emit()
        self.hide()

    def closeEvent(self, event):
        self.sigCancelClicked.emit()
        super().closeEvent(event)

    def getContent(self):
        selected_roi_id = self._selectedRoiId()
        if selected_roi_id is not None:
            try:
                selected_roi_id = int(selected_roi_id)
            except Exception:
                pass

        return {
            'crop_images_mode': self._selectedCropMode(),
            'crop_browse_source': self._crop_browse_source_file,
            'selected_roi_id': selected_roi_id,
            'ROI_coords': self.current_roi_coords,
        }

    def setContent(self, content):
        if not isinstance(content, dict):
            return

        crop_images_mode = str(content.get('crop_images_mode', 'none') or 'none')
        crop_browse_source = str(content.get('crop_browse_source', '') or '')
        selected_roi_id = content.get('selected_roi_id', None)

        crop_idx = self.cropImagesSelector.findData(crop_images_mode)
        if crop_idx >= 0:
            self.cropImagesSelector.setCurrentIndex(crop_idx)
        self._crop_browse_source_file = crop_browse_source
        self._updateCropBrowsePathLabel()
        self._pending_selected_roi_id = selected_roi_id
        self._updateBrowseControlsVisibility()
        self._updateRoiSelectorOptions()

    def saveContent(self, path):
        ext = '.json'
        if not path.endswith(ext):
            path += ext

        with open(path, 'w') as f:
            json.dump(self.getContent(), f, indent=2)

    def loadContent(self, path):
        ext = '.json'
        try:
            json_path = path if path.endswith(ext) else f'{path}{ext}'
            with open(json_path, 'r') as f:
                content = json.load(f)
            self.setContent(content)
        except Exception as err:
            self.logger.info(f'Failed to load crop card content from {path}: {err}')
            return

        self._emitOutputType()