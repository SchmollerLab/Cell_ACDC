import os
import traceback

from . import apps, qutils, widgets, printl, config, html_utils
from . import myutils, prompts
from .apps import QBaseDialog
from qtpy.QtWidgets import QVBoxLayout, QLabel, QComboBox, QApplication
from qtpy.QtGui import QPixmap, QColor
from qtpy.QtCore import Qt, Signal
import inspect
import json
import copy
import io
import re
from .workflow_typing import (
    WfImageDC, WfSegmDC, WfMetricsDC, workflow_type_name, make_workflow_data_class
)


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
        # check for the common signals sigSetOutputs
        if hasattr(dialog, 'sigSetOutputs'):
            dialog.sigSetOutputs.connect(self.setOutputs)

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
                    curr_accepted_inputs[idx] = self._setInputMetadataUniform(
                        curr_accepted_inputs[idx],
                        size_z=curr_accepted_inputs[idx].SizeZ,
                    )
                    
        # check for changes compared to original accepted inputs.
        for idx in curr_accepted_inputs:
            if (idx not in dialog.curr_input_types_accepted 
                or dialog.curr_input_types_accepted[idx] != curr_accepted_inputs[idx]):
                self.setAcceptedInputs(curr_accepted_inputs)
                break
            
    def _setInputMetadataUniform(self, input_type_ls,
                                    size_t=None, size_z=None):
        """Set input metadata to be uniform for all inputs of a specific slot (input_ls).
        """
        
        if not isinstance(input_type_ls, list):
            input_type_ls = [input_type_ls]
        
        new_input_type_ls = []
        for input_type in input_type_ls:
            input_type_name = workflow_type_name(input_type)
            new_input_type_ls.append(make_workflow_data_class(
                input_type_name, SizeT=size_t, SizeZ=size_z
            ))
                
        return new_input_type_ls
        
        

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
    
    def setupDialog(self, workflowGui=None, logger=print):
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
                                      posData=workflowGui.posData)
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
    
    def setupDialog(self, workflowGui=None, logger=print):
        """Create the dialog for the workflow card.
        
        Args:
            workflowGui: Reference to the main WorkflowGui instance.
        
        Returns:
            WorkflowInputDataDialog: Configured dialog instance.
        """
        img_options = workflowGui.img_channels
        dialog = self.inputDataDialog(img_options, 'image', parent=workflowGui,
                                      logger=logger, posData=workflowGui.posData)
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

    def setupDialog(self, workflowGui=None, logger=print):
        dialog = self.preprocessDialog(parent=workflowGui, logger=logger)
        return dialog

    def initializeDialog(self, dialog, workflowGui=None):
        self.setAcceptedInputs({0: WfImageDC()})
        self.setOutputs({0: WfImageDC()})
        dialog.sigOkClicked.connect(self.updatePreview)

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

    def setupDialog(self, workflowGui=None, posData=None, logger=print):
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

    def setupDialog(self, workflowGui=None, posData=None, logger=print):
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
    # self.segmentDialog = SegmentDialogWorkflow
    
    def __init__(self) -> None:
        self.selectModelDialog = SelectModelDialogWorkflow
        self.card_key = 'segmentation'
        self.title = "Segment"
        # self.segmentDialog = SegmentDialogWorkflow
        
    def dryrunDialog(self, parent=None, workflowGui=None, posData=None):
        return self.selectModelDialog(parent=workflowGui, posData=posData)
    
    def setupDialog(self, workflowGui=None, posData=None, logger=print):
        dialog = self.selectModelDialog(
            parent=workflowGui, 
            posData=posData,
            logger=logger,
        )
        return dialog
    
    def initializeDialog(self, dialog, workflowGui=None):
        self.setAcceptedInputs({0: WfImageDC(), 1: WfImageDC()})
        self.setOutputs({0: WfSegmDC()})
        dialog.sigCancelClicked.connect(self.cancel_clicked)
        dialog.sigModelChanged.connect(self.setupModelParameters)
        dialog.sigSelectionInvalid.connect(self.notifySelectionInvalid)
        if hasattr(self, 'notifySelectionValid'):
            dialog.sigSelectionValid.connect(self.notifySelectionValid)
        dialog.sigOkClicked.connect(self.setupModelParameters)
        dialog.sigOkClicked.connect(self.updatePreview)

    def runDialog_cb(self, dialog):
        if hasattr(dialog, 'openMainDialog'):
            dialog.openMainDialog()
            return
        super().runDialog_cb(dialog)
    
    def cancel_clicked(self):
        pass
    
    def setupModelParameters(self, model_name):
        if not model_name:
            self.updateTitle('Segment')
            return

        title = model_name.replace('_', ' ').strip().title()
        self.updateTitle(f'Segment: {title}')
        
class SelectModelDialogWorkflow(apps.QDialogSelectModel):
    sigModelChanged = Signal(str)
    sigSelectionInvalid = Signal(str)
    sigSelectionValid = Signal()

    def __init__(
            self, parent=None, posData=None, logger=print,
            addSkipSegmButton=False, customFirst=''
        ):
        self.posData = posData
        self.logger = logger
        self.model_name = None
        self.params_config_text = None
        self.last_error_message = None
        self._params_dialog = None
        self._params_dialog_model_name = None
        super().__init__(parent, addSkipSegmButton, customFirst)

    def _invalidate_params_dialog_cache(self):
        if self._params_dialog is not None:
            try:
                self._params_dialog.hide()
                self._params_dialog.deleteLater()
            except Exception:
                pass
        self._params_dialog = None
        self._params_dialog_model_name = None

    def _log(self, message):
        try:
            if hasattr(self.logger, 'info'):
                self.logger.info(message)
            elif callable(self.logger):
                self.logger(message)
        except Exception:
            pass

    def _display_model_name(self, model_name):
        if model_name == 'thresholding':
            return 'Automatic thresholding'
        return model_name

    def _show_model_params_error(self, err):
        traceback_str = traceback.format_exc()
        self._log(traceback_str)

        model_name = self._display_model_name(self.model_name or 'selected model')
        self.last_error_message = (
            f"Model parameters error for {model_name}: {type(err).__name__}: {err}"
        )
        self.sigSelectionInvalid.emit(self.last_error_message)
        txt = html_utils.paragraph(f"""
            Cell-ACDC failed while preparing the parameters dialog for
            <code>{model_name}</code>.<br><br>
            This can happen when the selected model needs to be downloaded or
            imported on the fly and that step fails.<br><br>
            Please restart Cell-ACDC and try again. If the problem persists,
            please report the issue on the <a href="https://github.com/SchmollerLab/Cell_ACDC/issues">Cell-ACDC GitHub issues page</a>.
        """)

        msg = widgets.myMessageBox(wrapText=False, showCentered=False)
        if hasattr(msg, 'setDetailedText'):
            msg.setDetailedText(traceback_str)
        msg.critical(self, 'Model parameters error', txt)

    def _set_selected_model_item(self, model_name):
        if not model_name:
            return

        display_name = self._display_model_name(model_name)
        items = self.listBox.findItems(display_name, Qt.MatchExactly)
        if not items:
            self.listBox.addItem(display_name)
            items = self.listBox.findItems(display_name, Qt.MatchExactly)

        if items:
            self.listBox.setCurrentItem(items[0])

    def _selected_model_name(self):
        item = self.listBox.currentItem()
        if item is None:
            return None

        model_name = item.text()
        if model_name == 'Add custom model...':
            model_file_path = apps.addCustomModelMessages(self)
            if model_file_path is None:
                return None

            myutils.store_custom_model_path(model_file_path)
            model_name = os.path.basename(os.path.dirname(model_file_path))
            item = self.listBox.findItems(model_name, Qt.MatchExactly)
            if not item:
                self.listBox.addItem(model_name)
                item = self.listBox.findItems(model_name, Qt.MatchExactly)
            if item:
                self.listBox.setCurrentItem(item[0])
            return model_name

        if model_name == 'Automatic thresholding':
            return 'thresholding'

        return model_name

    def _serialize_params_dialog(self, win):
        if self.model_name == 'thresholding':
            return json.dumps({
                'gauss_sigma': win.segment_kwargs['gauss_sigma'],
                'threshold_method': win.segment_kwargs['threshold_method'],
                'segment_3D_volume': win.segment_kwargs.get(
                    'segment_3D_volume', False
                ),
            })

        config_pars = win.getConfigPars(create_new=True)
        if hasattr(win, 'reduceMemUsageToggle'):
            section = f'{win.model_name}.additional_segm_params'
            config_pars[section] = {}
            option = win.reduceMemUsageToggle.label
            config_pars[section][option] = str(
                win.reduceMemUsageToggle.isChecked()
            )

        buffer = io.StringIO()
        config_pars.write(buffer)
        return buffer.getvalue()

    def _restore_params_dialog(self, win):
        if not self.params_config_text:
            return

        if self.model_name == 'thresholding':
            content = json.loads(self.params_config_text)
            win.sigmaGaussSpinbox.setValue(content.get('gauss_sigma', 1.0))
            threshold_method = content.get('threshold_method', 'threshold_otsu')
            win.threshMethodCombobox.setCurrentText(
                threshold_method[10:].capitalize()
            )
            if win.segment3Dcheckbox is not None:
                win.segment3Dcheckbox.setChecked(
                    content.get('segment_3D_volume', False)
                )
            return

        config_pars = config.ConfigParser()
        config_pars.read_string(self.params_config_text)
        win.loadPreprocRecipe(configPars=config_pars)
        win.loadLastSelection(
            f'{win.model_name}.init', win.init_argsWidgets, configPars=config_pars
        )
        win.loadLastSelection(
            f'{win.model_name}.segment', win.argsWidgets, configPars=config_pars
        )
        if win.extraArgsWidgets:
            win.loadLastSelection(
                f'{win.model_name}.extra',
                win.extraArgsWidgets,
                configPars=config_pars,
            )
        win.loadLastSelectionPostProcess(configPars=config_pars)

        if hasattr(win, 'reduceMemUsageToggle'):
            section = f'{win.model_name}.additional_segm_params'
            option = win.reduceMemUsageToggle.label
            if config_pars.has_option(section, option):
                win.reduceMemUsageToggle.setChecked(
                    config_pars.getboolean(section, option)
                )

    def _init_thresholding_dialog(self):
        is_segm_3d = False
        try:
            input_0_type = self.curr_input_types.get(0, None)
            size_z = getattr(input_0_type, 'SizeZ', None)
            is_segm_3d = bool(size_z and size_z > 1)
        except Exception:
            is_segm_3d = False

        win = apps.QDialogAutomaticThresholding(
            parent=self, isSegm3D=is_segm_3d
        )
        self._restore_params_dialog(win)
        return win

    def _init_model_params_dialog(self):
        download_win = apps.downloadModel(self.model_name, parent=self)
        download_win.download()

        acdc_segment = myutils.import_segment_module(self.model_name)
        init_params, segment_params = myutils.getModelArgSpec(acdc_segment)
        try:
            help_url = acdc_segment.url_help()
        except AttributeError:
            help_url = None

        out = prompts.init_segm_model_params(
            self.posData,
            self.model_name,
            init_params,
            segment_params,
            qparent=self,
            help_url=help_url,
            init_last_params=not bool(self.params_config_text),
            check_sam_embeddings=False,
            add_additional_segm_params=True,
            addPreProcessParams=False,
            addPostProcessParams=False,
            hideOnClosing=True,
            useInput2AsSecondChannelToggle=True,
            sam_embeddings_path=getattr(self.posData, 'sam_embeddings_path', None),
            sam_embeddings_loaded=hasattr(self.posData, 'sam_embeddings'),
            exec_dialog=False,
        )
        win = out.get('win')
        if win is None:
            return None

        self._restore_params_dialog(win)
        return win

    def _open_params_dialog(self):
        try:
            if (
                self._params_dialog is None
                or self._params_dialog_model_name != self.model_name
            ):
                self._invalidate_params_dialog_cache()
                if self.model_name == 'thresholding':
                    win = self._init_thresholding_dialog()
                else:
                    win = self._init_model_params_dialog()

                self._params_dialog = win
                self._params_dialog_model_name = self.model_name
            else:
                win = self._params_dialog

            if win is None:
                return 'cancelled'

            win.exec_()
            if win.cancel:
                return 'cancelled'

            self.params_config_text = self._serialize_params_dialog(win)
            self.last_error_message = None
            self.sigSelectionValid.emit()
            return 'accepted'
        except Exception as err:
            self._show_model_params_error(err)
            return 'error'
        
    def getContent(self):
        return {
            'model_name': self.model_name,
            'params_config_text': self.params_config_text,
        }
        
    def setContent(self, content):
        content = content or {}
        previous_model_name = self.model_name
        self.model_name = content.get('model_name')
        self.params_config_text = content.get('params_config_text')
        if previous_model_name != self.model_name:
            self._invalidate_params_dialog_cache()
        self.selectedModel = self.model_name
        self._set_selected_model_item(self.model_name)
        if self.model_name:
            self.sigModelChanged.emit(self.model_name)

    def ok_cb(self, event=None):
        selected_model = self._selected_model_name()
        if not selected_model:
            return

        if self.model_name != selected_model:
            self._invalidate_params_dialog_cache()

        self.model_name = selected_model
        self.selectedModel = selected_model
        self._set_selected_model_item(selected_model)

        dialog_result = self._open_params_dialog()
        if dialog_result == 'error':
            return

        if dialog_result == 'cancelled':
            self.cancel = True
            return

        self.cancel = False
        self.sigOkClicked.emit(self.model_name)
        self.close()

    def openMainDialog(self):
        if not self.model_name:
            super().show()
            return

        dialog_result = self._open_params_dialog()
        if dialog_result == 'accepted':
            self.cancel = False
            self.sigOkClicked.emit(self.model_name)
            return

        self.cancel = True
        self.sigCancelClicked.emit()

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

    def __init__(self, parent=None, logger=print):
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
            self.logger("No valid pre-processing steps configured. Please add at least one step before confirming.")
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
            self.logger(f"Failed to load content from {path}: {e}")
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

    def __init__(self, posData, parent=None, logger=print):
        super().__init__(parent=parent)
        self.cancel = True
        self.logger = logger
        self.curr_input_types = {}
        self.optional_inputs_n = {1: True}  # metrics input is optional

        self.setWindowTitle('Post-process segmentation parameters')

        self.postProcessGroupbox = apps.PostProcessSegmParams(
            'Post-processing parameters', posData,
            useSliders=True,
            parent=self,
            external_metrics=[],
        )
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

    def updatedInputTypes(self):
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
            self.logger(f"Failed to load content from {path}: {e}")
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

    def __init__(self, posData, parent=None, logger=print):
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
            self.logger(f'Failed to load content from {path}: {e}')
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
    sigSetOutputs = Signal(dict)  # Emit new output definitions when selection changes and affects output type
    
    def __init__(self, selection_options, type, parent=None, logger=print,
                 posData=None):
        super().__init__(parent=parent)
        self.setWindowTitle(f'Select input {type}')
        
        self.type = type
        self.selection_options = selection_options or []
        self.posData = posData
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
        
        # add connection to update sizeT and SizeZ
        self.selection_widget.currentTextChanged.connect(self.selectionChanged)
        
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
        
    def selectionChanged(self, new_selection):
        """Handle when selection changes and update the dialog UI accordingly."""
        self.sigUpdateTitle.emit(f'{self.type}: {new_selection}')
        if self.posData is not None:
            new_output_size_t = self.posData.SizeT
            new_output_size_z = self.posData.SizeZ
            if self.type == 'image':
                out_data_class = WfImageDC(SizeT=new_output_size_t, SizeZ=new_output_size_z)
            elif self.type == 'segmentation':
                # here theoretically we could have a 2D segm on 3D image...
                segm_info_df = self.posData.segmInfo_df if hasattr(self.posData, 'segmInfo_df') else None
                if segm_info_df is not None and new_selection in segm_info_df.index:
                    segm_info_df = segm_info_df.loc[new_selection]
                    proj_method = segm_info_df['which_z_proj']
                    if proj_method != 'single z-slice':
                        new_output_size_z = 1
                else:
                    # log 
                    txt = "segmInfo.csv not found." if segm_info_df is None else f"Selected segmentation '{new_selection}' not found in segmInfo.csv."
                    self.logger(f"[WARNING]: {txt}\nDefaulting to using original SizeZ from posData.")
                out_data_class = WfSegmDC(SizeT=new_output_size_t, SizeZ=new_output_size_z)
            else:
                raise ValueError(f"Unsupported type '{self.type}' for WorkflowInputDataDialog.")    
            self.sigSetOutputs.emit({0: out_data_class})
    
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
        self.hide()
    
    def cancel_clicked(self):
        """Handle Cancel button click by hiding the dialog."""
        self.hide()

    def closeEvent(self, event):
        """Treat window close (X) as cancel."""
        self.sigCancelClicked.emit()
        super().closeEvent(event)
    
    def get_selected_value(self):
        """Get the currently selected value.
        
        Returns:
            str: The text of the currently selected item in the dropdown.
        """
        return self.selected_value

    def setContent(self, content):
        """Set current selection from in-memory content."""
        value = str(content)
        idx = self.selection_widget.findText(value)
        if idx == -1:
            self.selection_widget.addItem(value)
            if value not in self.selection_options:
                self.selection_options.append(value)
            idx = self.selection_widget.findText(value)

        self.selection_widget.setCurrentIndex(idx)
        self.selected_value = value

    def getContent(self):
        """Return current selection as in-memory content."""
        return self.get_selected_value()
    
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
            else:
                self.logger(f"Loaded value '{content}' not in selection options.")
        except Exception as e:
            self.logger(f"Failed to load content from {path}: {e}")
            traceback.print_exc()
        self.ok_clicked()  # To emit signals and update the workflow card state after loading
        self.hide()