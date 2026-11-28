"""Model parameters window QDialogModelParams (from apps.py), used by the
main GUI, the segmentation module, prompts.py, the tracker
initialisation in myutils/segm_models.py and spotMAX.

Also its post-processing and feature-range sub-widgets
(PostProcessSegmParams, SelectFeaturesRange*) and the automatic
thresholding parameters QDialogAutomaticThresholding (main GUI and
segmentation module).
"""
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QHBoxLayout, QCheckBox, QWidget, QSizePolicy
from ..widgets import numeric_inputs as widgets_numeric_inputs
from ..widgets import line_edits as widgets_line_edits
import os
import datetime
import traceback
from functools import partial
from qtpy.QtGui import QIcon
from qtpy.QtCore import Qt, QEventLoop
from qtpy.QtWidgets import (
    QLabel, QVBoxLayout, QLineEdit, QDialog, QButtonGroup, QComboBox,
    QGridLayout, QSpinBox, QDoubleSpinBox, QGroupBox, QRadioButton, QSpacerItem
)
from cellacdc import _window_stacking
from cellacdc import core, measurements, html_utils
from cellacdc import settings_folderpath, config
from cellacdc import segm_recipes_path
from cellacdc import printl
from cellacdc import myutils
from cellacdc import qutils
from cellacdc import widgets
from cellacdc import _types
from cellacdc import fonts
from . import base as dialogs_base
from . import preprocessing as dialogs_preprocessing
from . import generic as dialogs_generic
from . import file_selection as dialogs_file_selection
from . import set_measurements as dialogs_set_measurements


class VerticalSpacerEmptyWidget(QWidget):
    def __init__(self, parent=None, height=5) -> None:
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum
        )
        self.setFixedHeight(height)


class NoneWidget:
    def __init__(self):
        pass
    
    def value(self):
        return None
    
    def setValue(self, value):
        return


class PostProcessSegmSlider(widgets_numeric_inputs.sliderWithSpinBox):
    def __init__(self, *args, label=None, **kwargs):
        super().__init__(*args, **kwargs)

        self.label = label
        self.checkbox = QCheckBox('Disable')
        self._layout.addWidget(self.checkbox, self.sliderCol, self.lastCol+1)
        self.checkbox.toggled.connect(self.onCheckBoxToggled)
        self.valueChanged.connect(self.checkExpandRange)
    
    def onCheckBoxToggled(self, checked: bool) -> None:
        super().setDisabled(checked)
        if self.label is not None:
            self.label.setDisabled(checked)
        self.onValueChanged(None)
        self.onEditingFinished()
    
    def onValueChanged(self, value):
        self.valueChanged.emit(value)
    
    def checkExpandRange(self, value):
        if value == self.maximum():
            range = int(self.maximum() - self.minimum())
            half_range = int(range/2)
            newMinimum = self.minimum() + half_range
            newMaximum = self.maximum() + half_range
            self.setMaximum(newMaximum)
            self.setMinimum(newMinimum)
        elif value == self.minimum():
            range = int(self.maximum() - self.minimum())
            half_range = int(range/2)
            newMinimum = self.minimum() - half_range
            newMaximum = self.maximum() - half_range
            self.setMaximum(newMaximum)
            self.setMinimum(newMinimum)
    
    def onEditingFinished(self):
        self.editingFinished.emit()
    
    def value(self):
        if self.checkbox.isChecked():
            return None
        else:
            return super().value()


class PostProcessSegmSpinbox(QWidget):
    valueChanged = Signal(int)
    editingFinished = Signal()
    sigCheckboxToggled = Signal()

    def __init__(self, *args, isFloat=False, label=None, **kwargs):
        super().__init__(*args, **kwargs)

        layout = QHBoxLayout()

        if isFloat:
            self.spinBox = widgets_numeric_inputs.DoubleSpinBox()
        else:
            self.spinBox = widgets_numeric_inputs.SpinBox()

        self.spinBox.valueChanged.connect(self.onValueChanged)
        self.spinBox.editingFinished.connect(self.onEditingFinished)

        layout.addWidget(self.spinBox)
        self.checkbox = QCheckBox('Disable')
        layout.addWidget(self.checkbox)
        layout.setStretch(0,1)
        layout.setStretch(1,0)

        self.label = label

        self.checkbox.toggled.connect(self.onCheckBoxToggled)
        
        layout.setContentsMargins(5, 0, 5, 0)
    
        self.setLayout(layout)
    
    def onCheckBoxToggled(self, checked: bool) -> None:
        self.spinBox.setDisabled(checked)
        if self.label is not None:
            self.label.setDisabled(checked)
        self.onValueChanged(None)
        self.onEditingFinished()
    
    def onValueChanged(self, value):
        self.valueChanged.emit(value)
    
    def onEditingFinished(self):
        self.editingFinished.emit()

    def maximum(self):
        return self.spinBox.maximum()
    
    def setValue(self, value):
        self.spinBox.setValue(value)
    
    def sizeHint(self):
        return self.spinBox.sizeHint()
    
    def setMaximum(self, max):
        self.spinBox.setMaximum(max)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)

    def setMinimum(self, min):
        self.spinBox.setMinimum(min)

    def setSingleStep(self, step):
        self.spinBox.setSingleStep(step)
    
    def setDecimals(self, decimals):
        self.spinBox.setDecimals(decimals)
    
    def value(self):
        if self.checkbox.isChecked():
            return None
        else:
            return self.spinBox.value()


def PostProcessSegmWidget(
        minimum, maximum, value, useSliders, isFloat=False, normalize=False,
        label=None
    ):
    if useSliders:
        if normalize:
            maximum = int(maximum*100)
        widget = PostProcessSegmSlider(
            normalize=normalize, isFloat=isFloat, label=label
        )
    else:
        widget = PostProcessSegmSpinbox(label=label, isFloat=isFloat)
    widget.setMinimum(minimum)
    widget.setMaximum(maximum)
    widget.setValue(value)
    return widget


class CheckableSpinBoxWidgets:
    def __init__(self, isFloat=True):
        if isFloat:
            self.spinbox = widgets_line_edits.FloatLineEdit()
        else:
            self.spinbox = widgets_numeric_inputs.SpinBox()
        self.checkbox = QCheckBox('Activate')
        self.spinbox.setEnabled(False)
        self.checkbox.toggled.connect(self.spinbox.setEnabled)
    
    def value(self):
        if not self.checkbox.isChecked():
            return
        return self.spinbox.value()


class QDialogAutomaticThresholding(dialogs_base.QBaseDialog):
    def __init__(self, parent=None, isSegm3D=True):
        super().__init__(parent)

        self.cancel = True

        self.setWindowTitle('Automatic thresholding parameters')

        layout = QVBoxLayout()
        formLayout = QGridLayout()
        buttonsLayout = QHBoxLayout()

        row = 0
        self.sigmaGaussSpinbox = QDoubleSpinBox()
        self.sigmaGaussSpinbox.setValue(1)
        self.sigmaGaussSpinbox.setMaximum(2**31)
        self.sigmaGaussSpinbox.setAlignment(Qt.AlignCenter)
        formLayout.addWidget(
            QLabel('Gaussian filter sigma (0 to ignore): '), row, 0,
            alignment=Qt.AlignRight
        )
        formLayout.addWidget(self.sigmaGaussSpinbox, row, 1, 1, 2)

        row += 1
        self.threshMethodCombobox = QComboBox()
        self.threshMethodCombobox.addItems([
            'Isodata', 'Li', 'Mean', 'Minimum', 'Otsu', 'Triangle', 'Yen'
        ])
        formLayout.addWidget(
            QLabel('Thresholding algorithm: '), row, 0,
            alignment=Qt.AlignRight
        )
        formLayout.addWidget(self.threshMethodCombobox, row, 1, 1, 2)

        self.segment3Dcheckbox = None
        if isSegm3D:
            row += 1
            formLayout.addWidget(
                QLabel('Segment 3D volume: '), row, 0, alignment=Qt.AlignRight
            )
            group = QButtonGroup()
            group.setExclusive(True)
            self.segment3Dcheckbox = QRadioButton('Yes')
            segmentSliceBySliceCheckbox = QRadioButton('No, segment slice-by-slice')
            group.addButton(self.segment3Dcheckbox)
            group.addButton(segmentSliceBySliceCheckbox)
            formLayout.addWidget(self.segment3Dcheckbox, row, 1)
            formLayout.addWidget(segmentSliceBySliceCheckbox, row, 2)
            self.segment3Dcheckbox.setChecked(True)

        okButton = widgets.okPushButton('Ok')
        cancelButton = widgets.cancelPushButton('Cancel')
        helpButton = widgets.helpPushButton('Help...')

        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        buttonsLayout.addWidget(helpButton)
        buttonsLayout.addWidget(okButton)

        layout.addLayout(formLayout)
        layout.addSpacing(20)
        layout.addLayout(buttonsLayout)

        okButton.clicked.connect(self.ok_cb)
        helpButton.clicked.connect(self.help_cb)
        cancelButton.clicked.connect(self.close)

        self.setLayout(layout)
        self.setFont(fonts.font)

        self.configPars = self.loadLastSelection()

    
    def help_cb(self):
        import webbrowser
        url = 'https://scikit-image.org/docs/stable/auto_examples/applications/plot_thresholding.html'
        webbrowser.open(url)

    def ok_cb(self):
        self.cancel = False
        self.gaussSigma = self.sigmaGaussSpinbox.value()
        threshMethod = self.threshMethodCombobox.currentText().lower()
        self.threshMethod = f'threshold_{threshMethod}'
        self.segment_kwargs = {
            'gauss_sigma': self.gaussSigma,
            'threshold_method': self.threshMethod,
            'segment_3D_volume': False
        }
        self.reduceMemoryUsage = False
        if self.segment3Dcheckbox is not None:
            doSegm3D = self.segment3Dcheckbox.isChecked()
            self.segment_kwargs['segment_3D_volume'] = doSegm3D
        self.close()
    
    def loadLastSelection(self):
        self.ini_path = os.path.join(
            settings_folderpath, 'last_params_segm_models.ini'
        )
        if not os.path.exists(self.ini_path):
            return

        configPars = config.ConfigParser()
        configPars.read(self.ini_path)

        if 'thresholding.segment' not in configPars.sections():
            return

        section = configPars['thresholding.segment']
        self.sigmaGaussSpinbox.setValue(float(section['gauss_sigma']))

        threshold_method = section['threshold_method']
        Method = threshold_method[10:].capitalize()
        self.threshMethodCombobox.setCurrentText(Method)
        if self.segment3Dcheckbox is None:
            return
        self.segment3Dcheckbox.setChecked(section.getboolean('segment_3D_volume'))


class PostProcessSegmParams(QGroupBox):
    valueChanged = Signal(object)
    editingFinished = Signal()

    def __init__(
            self, title, posData, 
            useSliders=False, 
            parent=None, 
            maxSize=None, 
            force_postprocess_2D=False
        ):
        QGroupBox.__init__(self, title, parent)
        SizeZ = posData.SizeZ
        self.isSegm3D = posData.isSegm3D
        self.channelName = posData.user_ch_name
        self.useSliders = useSliders
        self.force_postprocess_2D = force_postprocess_2D
        if maxSize is None:
            maxSize=2147483647

        layout = QGridLayout()

        self.controlWidgets = []

        row = 0
        label = QLabel("Minimum area (pixels) ")
        layout.addWidget(label, row, 0, alignment=Qt.AlignRight)

        minSize_SB = widgets.PostProcessSegmWidget(
            1, 1000, 10, useSliders, label=label
        )
        
        txt = (
            '<b>Area</b> is the total number of pixels in the segmented object.'
        )

        layout.addWidget(minSize_SB, row, 1)
        infoButton = widgets.infoPushButton()
        infoButton.clicked.connect(self.showInfo)
        infoButton.tooltip = txt
        infoButton.name = 'area'
        infoButton.desc = f'less than "{label.text()}"'
        layout.addWidget(infoButton, row, 2)
        self.minSize_SB = minSize_SB
        self.controlWidgets.append(minSize_SB)

        # minSize_SB.disableThisCheckbox = QCheckBox('Disable this filter')
        # layout.addWidget(minSize_SB.disableThisCheckbox, row, 3)

        row += 1
        label = QLabel("Minimum solidity (0-1) ")
        layout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        minSolidity_DSB = widgets.PostProcessSegmWidget(
            0, 1.0, 0.5, useSliders, isFloat=True, normalize=True,
            label=label
        )
        minSolidity_DSB.setValue(0.5)
        minSolidity_DSB.setSingleStep(0.1)
        self.controlWidgets.append(minSolidity_DSB)

        txt = (
            '<b>Solidity</b> is a measure of convexity. A solidity of 1 means '
            'that the shape is fully convex (i.e., equal to the convex hull). '
            'As solidity approaches 0 the object is more concave.<br>'
            'Write 0 for ignoring this parameter.'
        )

        layout.addWidget(minSolidity_DSB, row, 1)
        infoButton = widgets.infoPushButton()
        infoButton.clicked.connect(self.showInfo)
        infoButton.tooltip = txt
        infoButton.name = 'solidity'
        infoButton.desc = f'less than "{label.text()}"'
        layout.addWidget(infoButton, row, 2)
        self.minSolidity_DSB = minSolidity_DSB

        row += 1
        label = QLabel("Max elongation (1=circle) ")
        layout.addWidget(label, row, 0, alignment=Qt.AlignRight)
        maxElongation_DSB = widgets.PostProcessSegmWidget(
            0, 100, 3, useSliders, isFloat=True, normalize=False,
            label=label
        )
        maxElongation_DSB.setDecimals(1)
        maxElongation_DSB.setSingleStep(1.0)

        txt = (
            '<b>Elongation</b> is the ratio between major and minor axis lengths. '
            'An elongation of 1 is like a circle.<br>'
            'Write 0 for ignoring this parameter.'
        )

        layout.addWidget(maxElongation_DSB, row, 1)
        infoButton = widgets.infoPushButton()
        infoButton.clicked.connect(self.showInfo)
        infoButton.tooltip = txt
        infoButton.name = 'elongation'
        infoButton.desc = f'greater than "{label.text()}"'
        layout.addWidget(infoButton, row, 2)
        self.maxElongation_DSB = maxElongation_DSB
        self.controlWidgets.append(maxElongation_DSB)

        if self.isSegm3D:
            row += 1
            label = QLabel("Minimum number of z-slices ")
            layout.addWidget(label, row, 0, alignment=Qt.AlignRight)
            minObjSizeZ_SB = widgets.PostProcessSegmWidget(
                0, SizeZ, 3, useSliders, isFloat=False, normalize=False,
                label=label
            )

            txt = (
                '<b>Minimum number of z-slices</b> per object.'
            )

            layout.addWidget(minObjSizeZ_SB, row, 1)
            infoButton = widgets.infoPushButton()
            infoButton.clicked.connect(self.showInfo)
            infoButton.tooltip = txt
            infoButton.name = 'number of z-slices'
            infoButton.desc = f'less than "{label.text()}"'
            layout.addWidget(infoButton, row, 2)
            self.minObjSizeZ_SB = minObjSizeZ_SB
            self.controlWidgets.append(minObjSizeZ_SB)
        else:
            self.minObjSizeZ_SB = widgets.NoneWidget()

        row += 1
        addCustomFeatureLayout = QHBoxLayout()
        self.addCustomFeaturesButton = widgets.setPushButton(
            'Select custom features for post-processing...',
        )
        addCustomFeatureLayout.addWidget(self.addCustomFeaturesButton)
        addCustomFeatureLayout.addStretch(1)
        self.selectedFeaturesDialog = SelectFeaturesRangeDialog(
            posData=posData, parent=self, 
            force_postprocess_2D=force_postprocess_2D
        )
        self.selectedFeaturesDialog.hide()
        self.addCustomFeaturesButton.clicked.connect(
            self.selectedFeaturesDialog.show
        )
        self.selectedFeaturesDialog.sigValueChanged.connect(self.onValueChanged)
        
        layout.addLayout(addCustomFeatureLayout, row, 0, 1, 2)
        
        layout.setColumnStretch(1, 2)
        # layout.setRowStretch(row+1, 1)

        self.setLayout(layout)

        for widget in self.controlWidgets:
            widget.valueChanged.connect(self.onValueChanged)
            widget.editingFinished.connect(self.onEditingFinished)
    
    def selectedFeaturesRange(self):
        return self.selectedFeaturesDialog.groupbox.selectedFeaturesRange()

    def groupedFeatures(self):
        return self.selectedFeaturesDialog.groupbox.groupedFeatures()
    
    def restoreDefault(self):
        self.minSolidity_DSB.setValue(0.5)
        self.minSize_SB.setValue(10)
        self.maxElongation_DSB.setValue(3)
        self.minObjSizeZ_SB.setValue(3)
        self.selectedFeaturesDialog.groupbox.resetFields()
    
    def restoreFromKwargs(self, kwargs):
        for name, value in kwargs.items():
            if name == 'min_solidity':
                self.minSolidity_DSB.setValue(value)
            elif name == 'min_area':
                self.minSize_SB.setValue(value)
            elif name == 'max_elongation':
                self.maxElongation_DSB.setValue(value)
            elif name == 'min_obj_no_zslices':
                self.minObjSizeZ_SB.setValue(value)
    
    def kwargs(self):
        kwargs = {
            'min_solidity': self.minSolidity_DSB.value(),
            'min_area': self.minSize_SB.value(),
            'max_elongation': self.maxElongation_DSB.value(),
            'min_obj_no_zslices': self.minObjSizeZ_SB.value()
        }
        return kwargs
    
    def onValueChanged(self, value):
        self.valueChanged.emit(value)
    
    def onEditingFinished(self):
        self.editingFinished.emit()
    
    def showInfo(self):
        title = f'{self.sender().text()} info'
        tooltip = self.sender().tooltip
        name = self.sender().name
        desc = self.sender().desc
        txt = (f"""
            The post-processing step is applied to the output of the 
            segmentation model.<br><br>
            During this step, Cell-ACDC will remove all the objects with {name}
            <b>{desc}</b>.<br><br>
            {tooltip}    
        """)
        if self.isCheckable():
            note = f""""
                You can deactivate this step by un-checking the checkbox 
                called "Post-processing parameters".
            """
            txt = f'{txt}{note}'
        msg = widgets.myMessageBox(showCentered=False)
        msg.information(self, title, html_utils.paragraph(txt))


class QDialogModelParams(QDialog):
    def __init__(
            self, 
            init_params, 
            segment_params, 
            model_name, 
            is_tracker=False,
            url=None, 
            parent=None, 
            initLastParams=True, 
            posData=None, 
            channels=None, 
            currentChannelName=None, 
            segmFileEndnames=None,
            df_metadata=None, 
            force_postprocess_2D=False, 
            model_module=None,
            action_type='', 
            addPreProcessParams=True, 
            addPostProcessParams=True,
            extraParams=None, 
            extraParamsTitle=None, 
            ini_filename=None,
            add_additional_segm_params=False
        ):
        self.cancel = True
        super().__init__(parent)
        self.channels = channels
        self.is_tracker = is_tracker
        self.currentChannelName = currentChannelName
        self.channelCombobox = None
        self.segmFileEndnames = segmFileEndnames
        self.df_metadata = df_metadata
        self.force_postprocess_2D = force_postprocess_2D

        self.skipSegmentation = False
        if len(segment_params) > 0:
            if segment_params[0].name.lower().find('skip_segmentation') != -1:
                self.skipSegmentation = True
                addPreProcessParams = False
            else:
                self.skipSegmentation = False
        if ini_filename is not None:
            self.ini_filename = ini_filename
        elif is_tracker:
            self.ini_filename = 'last_params_trackers.ini'
            addPreProcessParams = False
            addPostProcessParams = False
        else:
            self.ini_filename = 'last_params_segm_models.ini'

        self.addPreProcessParams = addPreProcessParams
        
        self.model_name = model_name

        self.setWindowTitle(f'{model_name} parameters')

        # Create main vertical layout and horizontal layout for two columns
        mainLayout = QVBoxLayout()
        
        gridLayout = QGridLayout()
        self.gridLayout = gridLayout
        
        loadFunc = self.loadLastSelection
        
        self.paramsGroupPosMapper = {}
        
        # LEFT COLUMN: Preprocessing params
        row, col = 0, 0
        preProcessLayout = None
        self.preProcessParamsWidget = None
        if addPreProcessParams:
            preProcessLayout = QVBoxLayout()
            self.preProcessParamsWidget = dialogs_preprocessing.PreProcessParamsWidget(
                parent=self, addApplyButton=False
            )
            self.preProcessParamsWidget.setChecked(False)
            preProcessLayout.addWidget(self.preProcessParamsWidget)
            self.preProcessParamsWidget.sigLoadRecipe.connect(
                self.loadPreprocRecipe
            )
            gridLayout.addLayout(preProcessLayout, row, col, 1, 2)
            self.paramsGroupPosMapper[self.preProcessParamsWidget] = (row, col)
            gridLayout.addItem(QSpacerItem(10, 5), 0, col+1)
            # gridLayout.setColumnMinimumWidth(col+1, 15)
            col += 2 
        
        # Center COLUMN: Init, Segmentation/Eval
        row = 0
        self.secondColLayout = QVBoxLayout()
        self.initParamsScrollArea = widgets.ScrollArea()
        initParamsScrollAreaLayout = QVBoxLayout()
        self.initParamsScrollArea.setVerticalLayout(initParamsScrollAreaLayout)
        
        initGroupBox, self.init_argsWidgets = self.createGroupParams(
            init_params, 'Parameters for model initialization'
        )
        self.init_params = init_params
        initDefaultButton = widgets.reloadPushButton('Restore default')
        initLoadLastSelButton = widgets.OpenFilePushButton(
            'Load last parameters'
        )
        initLoadLastSelButton.setIcon(QIcon(':folder-open.svg'))
        initButtonsLayout = QHBoxLayout()
        initButtonsLayout.addStretch(1)
        initButtonsLayout.addWidget(initDefaultButton)
        initButtonsLayout.addWidget(initLoadLastSelButton)
        initDefaultButton.clicked.connect(self.restoreDefaultInit)
        initLoadLastSelButton.clicked.connect(
            partial(loadFunc, f'{self.model_name}.init', self.init_argsWidgets)
        )
        
        initParamsScrollAreaLayout.addWidget(initGroupBox)
        
        initParamsLayout = QVBoxLayout()
        initParamsLayout.addWidget(QLabel(f'<b>{initGroupBox.title()}</b>'))
        initGroupBox.setTitle('')
        initParamsLayout.addWidget(self.initParamsScrollArea)
        initParamsLayout.addLayout(initButtonsLayout)
        self.secondColLayout.addLayout(initParamsLayout)
        self.paramsGroupPosMapper[self.initParamsScrollArea] = (0, col)

        self.segmentParamsScrollArea = None
        if not self.skipSegmentation:
            self.segmentParamsScrollArea = widgets.ScrollArea()
            segmentParamsScrollAreaLayout = QVBoxLayout()
            self.segmentParamsScrollArea.setVerticalLayout(
                segmentParamsScrollAreaLayout
            )
            if action_type:
                runGroupboxTitle = f'Parameters for {action_type}'
            elif is_tracker:
                runGroupboxTitle = 'Parameters for tracking'
            else:
                runGroupboxTitle = 'Parameters for segmentation'
            
            segmentGroupBox, self.argsWidgets = self.createGroupParams(
                segment_params, runGroupboxTitle, 
                addChannelSelector=True
            ) 
            self.segment_params = segment_params
            self.segmentGroupBox = segmentGroupBox
            segmentDefaultButton = widgets.reloadPushButton('Restore default')
            segmentLoadLastSelButton = widgets.OpenFilePushButton(
                'Load last parameters'
            )
            segmentButtonsLayout = QHBoxLayout()
            segmentButtonsLayout.addStretch(1)
            segmentButtonsLayout.addWidget(segmentDefaultButton)
            segmentButtonsLayout.addWidget(segmentLoadLastSelButton)
            segmentDefaultButton.clicked.connect(self.restoreDefaultSegment)
            section = f'{self.model_name}.segment'
            segmentLoadLastSelButton.clicked.connect(
                partial(loadFunc, section, self.argsWidgets)
            )
            segmentParamsScrollAreaLayout.addWidget(segmentGroupBox)
            
            segmentParamsLayout = QVBoxLayout()
            segmentParamsLayout.addWidget(
                QLabel(f'<b>{segmentGroupBox.title()}</b>')
            )
            segmentGroupBox.setTitle('')
            segmentParamsLayout.addWidget(self.segmentParamsScrollArea)
            segmentParamsLayout.addLayout(segmentButtonsLayout)
            self.secondColLayout.addLayout(segmentParamsLayout)
            self.paramsGroupPosMapper[self.segmentParamsScrollArea] = (1, col)

        gridLayout.addLayout(self.secondColLayout, row, col)
        
        gridLayout.addItem(QSpacerItem(10, 5), 0, col+1)
        col += 2 
        
        # Buttons layout (spans both columns)
        buttonsLayout = QHBoxLayout()
        cancelButton = widgets.cancelPushButton(' Cancel ')
        okButton = widgets.okPushButton(' Ok ')
        
        enableLoadingSavingRecipe = (
            not is_tracker and (addPreProcessParams or addPostProcessParams)
        )
        
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(cancelButton)
        buttonsLayout.addSpacing(20)
        if enableLoadingSavingRecipe:
            loadEntireRecipeButton = widgets.OpenFilePushButton(
                'Load saved recipe...'
            )
            saveEntireRecipeButton = widgets.savePushButton(
                'Save all parameters to recipe file...'
            )
            buttonsLayout.addWidget(loadEntireRecipeButton)
            buttonsLayout.addWidget(saveEntireRecipeButton)
            loadEntireRecipeButton.clicked.connect(self.loadEntireRecipe)
            saveEntireRecipeButton.clicked.connect(self.saveEntireRecipe)
            
        buttonsLayout.addWidget(okButton)

        buttonsLayout.setContentsMargins(0, 10, 0, 10)

        okButton.clicked.connect(self.ok_cb)
        cancelButton.clicked.connect(self.close)

        self.okButton = okButton            
        
        # Extra params in right column
        row = 0
        self.extraArgsWidgets = None
        self.extraParamsScrollArea = None
        if extraParams is not None:
            self.extraParamsScrollArea = widgets.ScrollArea()
            extraParamsScrollAreaLayout = QVBoxLayout()
            self.extraParamsScrollArea.setVerticalLayout(
                extraParamsScrollAreaLayout
            )
            if extraParamsTitle is None:
                extraParamsTitle = 'Additional parameters'

            self.extraGroupBox, self.extraArgsWidgets = self.createGroupParams(
                extraParams, extraParamsTitle
            )

            extraDefaultButton = widgets.reloadPushButton('Restore default')
            extraLoadLastSelButton = widgets.OpenFilePushButton(
                'Load last parameters'
            )
            extraButtonsLayout = QHBoxLayout()
            extraButtonsLayout.addStretch(1)
            extraButtonsLayout.addWidget(extraDefaultButton)
            extraButtonsLayout.addWidget(extraLoadLastSelButton)
            extraDefaultButton.clicked.connect(self.restoreDefaultExtra)
            section = f'{self.model_name}.extra'
            extraLoadLastSelButton.clicked.connect(
                partial(loadFunc, section, self.extraArgsWidgets)
            )

            extraParamsScrollAreaLayout.addWidget(self.extraGroupBox)
            
            extraParamsLayout = QVBoxLayout()
            extraParamsLayout.addWidget(QLabel(f'<b>{self.extraGroupBox.title()}</b>'))
            self.extraGroupBox.setTitle('')
            extraParamsLayout.addWidget(self.extraParamsScrollArea)
            extraParamsLayout.addLayout(extraButtonsLayout)
            self.paramsGroupPosMapper[self.extraParamsScrollArea] = (row, col)
            gridLayout.addLayout(extraParamsLayout, row, col)
            row += 1

        # Post-processing in right-most column
        self.postProcessGroupbox = None
        self.seeHereLabel = None
        thirdColumnLayout = QVBoxLayout()
        if addPostProcessParams:            
            # Add minimum size spinbox which is valid for all models
            postProcessGroupbox = PostProcessSegmParams(
                'Post-processing segmentation parameters', posData,
                force_postprocess_2D=force_postprocess_2D
            )
            postProcessGroupbox.setCheckable(True)
            postProcessGroupbox.setChecked(False)
            self.postProcessGroupbox = postProcessGroupbox

            thirdColumnLayout.addWidget(postProcessGroupbox)

            postProcDefaultButton = widgets.reloadPushButton('Restore default')
            postProcLoadLastSelButton = widgets.OpenFilePushButton(
                'Load last parameters'
            )
            postProcButtonsLayout = QHBoxLayout()
            postProcButtonsLayout.addStretch(1)
            postProcButtonsLayout.addWidget(postProcDefaultButton)
            postProcButtonsLayout.addWidget(postProcLoadLastSelButton)
            postProcDefaultButton.clicked.connect(self.restoreDefaultPostprocess)
            postProcLoadLastSelButton.clicked.connect(
                self.loadLastSelectionPostProcess
            )
            thirdColumnLayout.addLayout(postProcButtonsLayout)
            thirdColumnLayout.addSpacing(15)

            if url is not None:
                self.seeHereLabel = self.createSeeHereLabel(url)
                thirdColumnLayout.addWidget(
                    self.seeHereLabel, alignment=Qt.AlignCenter
                )
            
            self.paramsGroupPosMapper[self.preProcessParamsWidget] = (row, col)

        # Additional segmentation params in right column
        self.additionalSegmGroupbox = None
        if add_additional_segm_params:
            thirdColumnLayout.addWidget(widgets.QHLine())
            additionalSegmGroupbox = self.getAdditionalSegmParams()
            thirdColumnLayout.addWidget(additionalSegmGroupbox)
            self.additionalSegmGroupbox = additionalSegmGroupbox
            self.paramsGroupPosMapper[self.additionalSegmGroupbox] = (row, col)
        
        thirdColumnLayout.addStretch(1)
        gridLayout.addLayout(thirdColumnLayout, row, col)
        row += 1
        
        # Add everything to main layout
        mainLayout.addLayout(gridLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)

        self.configPars = self.readLastSelection()
        if self.configPars is None:
            initLoadLastSelButton.setDisabled(True)
            segmentLoadLastSelButton.setDisabled(True)
            if self.postProcessGroupbox is not None:
                postProcLoadLastSelButton.setDisabled(True)

        if initLastParams:
            initLoadLastSelButton.click()
            if not self.skipSegmentation:
                segmentLoadLastSelButton.click()
            
            if self.extraArgsWidgets is not None:
                extraLoadLastSelButton.click()
        
            if self.postProcessGroupbox is not None:
                postProcLoadLastSelButton.click()

        try:
            self.connectCustomSignals(model_module)
        except Exception as e:
            printl(traceback.format_exc())
        
        self.setLayout(mainLayout)
        self.setFont(fonts.font)
        # self.setModal(True)
    
    def warningNoSegmRecipes(self):
        msg = widgets.myMessageBox(wrapText=False)
        txt = html_utils.paragraph(
            'No segmentation recipes found!<br><br>'
            'To create a segmentation recipe you need click on '
            '<code>Save all parameters to recipe file...</code> '
            'button.'
        )
        msg.warning(self, 'No segmentation recipes found!', txt)
    
    def selectIniFileToLoadEntireRecipe(self):
        import qtpy.compat
        recipe_filepath = qtpy.compat.getopenfilename(
            parent=self, 
            caption='Select INI file to load entire recipe', 
            filters='INI (*.ini);;All Files (*)'
        )[0]
        if not recipe_filepath:
            return
        
        self.loadRecipeFromFilepath(recipe_filepath)
        
        txt = html_utils.paragraph(
            'Done!<br><br>'
            'Segmentation recipe loaded from:'
        )
        msg = widgets.myMessageBox()
        msg.information(
            self, 'Segmentation recipe loaded!', txt, 
            commands=(recipe_filepath,),
            path_to_browse=os.path.dirname(recipe_filepath)
        )
        
        print('Done. Segmentation recipe loaded from:', recipe_filepath)

    def loadEntireRecipe(self):
        segm_recipes_path_model = os.path.join(
            segm_recipes_path, self.model_name
        )

        if not os.path.exists(segm_recipes_path_model):
            # self.warningNoSegmRecipes()
            self.selectIniFileToLoadEntireRecipe()
            return
    
        recipe_files = os.listdir(segm_recipes_path_model)

        if not recipe_files:
            # self.warningNoSegmRecipes()
            self.selectIniFileToLoadEntireRecipe()
            return
        
        headerLabels = ['Name', 'Date Created']
        items = []
        for recipe_file in recipe_files:
            cp = config.ConfigParser()
            cp.read(os.path.join(segm_recipes_path_model, recipe_file))
            date_created = cp['info']['created_on']
            items.append((recipe_file, date_created))

        browseButton = widgets.browseFileButton(
            'Select INI file...',
            title='Select INI file to load entire recipe',
            openFolder=False,
            start_dir=myutils.getMostRecentPath(),
            ext={'INI': '.ini'}
        )
        win = dialogs_generic.QTreeDialog(
            items,
            headerLabels=headerLabels,
            title='Select a segmentation recipe to load',
            infoText='Select a segmentation recipe to load:<br>',
            path_to_browse=segm_recipes_path_model,
            additional_buttons=(browseButton, )
        )
        browseButton.sigPathSelected.connect(
            partial(
                self.entireRecipeIniFileSelected, 
                selectRecipeWin=win, 
                sender=browseButton
            )
        )
        win.exec_()
        if win.cancel or not hasattr(win, 'selectedText'):
            print('Loading segmentation recipe cancelled.')
            return
        
        if win.clickedButton == browseButton:
            recipe_filepath = win.selectedIniFilepath
        else:
            recipe_filename = win.selectedText
            recipe_filepath = os.path.join(
                segm_recipes_path_model, recipe_filename
            )
        
        self.loadRecipeFromFilepath(recipe_filepath)
        
        txt = html_utils.paragraph(
            'Done!<br><br>'
            'Segmentation recipe loaded from:'
        )
        msg = widgets.myMessageBox()
        msg.information(
            self, 'Segmentation recipe laoded!', txt, 
            commands=(recipe_filepath,),
            path_to_browse=os.path.dirname(recipe_filepath)
        )
        
        print('Done. Segmentation recipe loaded from:', recipe_filepath)
    
    def entireRecipeIniFileSelected(
            self, recipe_filepath, selectRecipeWin=None, sender=None
        ):
        selectRecipeWin.selectedText = 'None'
        selectRecipeWin.clickedButton = sender
        selectRecipeWin.selectedIniFilepath = recipe_filepath
        selectRecipeWin.cancel = False
        selectRecipeWin.close()

    def loadRecipeFromFilepath(self, recipe_filepath):
        cp = config.ConfigParser()
        cp.read(recipe_filepath)
        
        self.loadPreprocRecipe(configPars=cp)
        self.loadLastSelection(
            f'{self.model_name}.init', self.init_argsWidgets, configPars=cp
        )
        self.loadLastSelection(
            f'{self.model_name}.segment', self.argsWidgets, configPars=cp
        )
        if self.extraArgsWidgets:
            self.loadLastSelection(
                f'{self.model_name}.extra', self.extraArgsWidgets, configPars=cp
            )
        self.loadLastSelectionPostProcess(configPars=cp)
    
    def saveEntireRecipe(self):
        segm_recipes_path_model = os.path.join(
            segm_recipes_path, self.model_name
        )
        try:
            existingNames=os.listdir(segm_recipes_path_model)
        except FileNotFoundError:
            existingNames = []

        win = dialogs_file_selection.filenameDialog(
            title='Filename for segmentation recipe',
            basename='segmentation_recipe',
            ext='.ini',
            hintText='Insert a <b>filename</b> for the segmentation recipe:',
            allowEmpty=False,
            parent=self,
            existingNames=existingNames
        )
        win.exec_()
        if win.cancel:
            return
        
        ini_filename = win.filename
        os.makedirs(segm_recipes_path, exist_ok=True)
        os.makedirs(segm_recipes_path_model, exist_ok=True)
        ini_filepath = os.path.join(segm_recipes_path_model, ini_filename)
        
        configPars = self.getConfigPars(create_new=True)

        if hasattr(self, 'reduceMemUsageToggle'):
            configPars[f'{self.model_name}.additional_segm_params'] = {}
            reduceMemoryUsage = self.reduceMemUsageToggle.isChecked()
            option = self.reduceMemUsageToggle.label
            configPars[f'{self.model_name}.additional_segm_params'][option] = (
                str(reduceMemoryUsage)
            )
        
        configPars['info'] = {}
        configPars['info']['created_on'] = datetime.datetime.now().strftime(
            r'%Y/%m/%d %H:%M'
        )
        
        with open(ini_filepath, 'w') as configfile:
            configPars.write(configfile)
        
        txt = html_utils.paragraph(
            'Done!<br><br>'
            'Segmentation recipe saved to:'
        )
        msg = widgets.myMessageBox()
        msg.information(
            self, 'Segmnentation recipe saved!', txt, 
            commands=(ini_filepath,),
            path_to_browse=os.path.dirname(ini_filepath)
        )
        
        print('Done. Segmentation recipe saved to:', ini_filepath)
    
    def getAdditionalSegmParams(self):
        additionalSegmGroupbox = QGroupBox('Additional segmentation parameters')
        local_row = 0
        additionalSegmLayout = QGridLayout()
        option = 'Reduce memory usage'
        additionalSegmLayout.addWidget(
            QLabel(f'{option}:  '), local_row, 0, 
            alignment=Qt.AlignRight
        )
        self.reduceMemUsageToggle = widgets.Toggle()
        additionalSegmLayout.addWidget(
            self.reduceMemUsageToggle, local_row, 1, 1, 2, 
            alignment=Qt.AlignCenter
        )
        self.reduceMemUsageToggle.label = option
        reduceMemUsageInfoButton = widgets.infoPushButton()
        additionalSegmLayout.addWidget(reduceMemUsageInfoButton, local_row, 3)
        reduceMemUsageInfoButton.clicked.connect(
            self.showInfoReduceMemUsage
        )
        additionalSegmLayout.setColumnStretch(0, 0)
        additionalSegmLayout.setColumnStretch(1, 1)
        additionalSegmLayout.setColumnStretch(3, 0)
        additionalSegmGroupbox.setLayout(additionalSegmLayout)
        return additionalSegmGroupbox
    
    def showInfoReduceMemUsage(self):
        infoText = html_utils.paragraph(f"""
            If you are experiencing memory issues, you can try reducing the 
            memory usage by toggling this option.<br><br>
            This will reduce the memory usage by segmenting timelapse data 
            frame-by-frame instead of all frames at once.
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(
            self, 'Reduce memory usage', infoText
        )
        
    def loadPreprocRecipe(self, configPars=None):
        if self.configPars is None and configPars is None:
            return
        
        if configPars is None:
            configPars = self.configPars
        
        preprocConfigPars = {}
        for section in configPars.sections():
            if not section.startswith(f'{self.model_name}.preprocess'):
                continue      
            
            preprocConfigPars[section] = configPars[section]
        
        if not preprocConfigPars:
            return
        
        self.preProcessParamsWidget.loadRecipe(preprocConfigPars)
    
    def connectCustomSignals(self, model_module):
        if model_module is None:
            return

        if not hasattr(model_module, 'CustomSignals'):
            return
        
        customSignals = model_module.CustomSignals()
        for slot_info in customSignals.slots_info:
            group = slot_info['group']
            widget_name = slot_info['widget_name']
            if group == 'init':
                ArgsWidgets_list = self.init_argsWidgets
            else:
                ArgsWidgets_list = self.argsWidgets
            for argwidget in ArgsWidgets_list:
                if argwidget.name == widget_name:
                    signal = getattr(argwidget.widget, slot_info['signal'])
                    signal.connect(partial(slot_info['slot'], self))
                    break
    
    def selectedFeaturesRange(self):
        if self.postProcessGroupbox is None:
            return {}
        return self.postProcessGroupbox.selectedFeaturesRange()

    def groupedFeatures(self):
        if self.postProcessGroupbox is None:
            return {}
        return self.postProcessGroupbox.groupedFeatures()
    
    def setChannelNames(self, chNames):
        if not hasattr(self, 'channelsCombobox'):
            return
        
        items = ['None']
        items.extend(chNames)
        self.channelsCombobox.addItems(items)
    
    def getValueFromMetadata(self, name):
        try:
            value = self.df_metadata.at[name, 'values']
        except Exception as e:
            # traceback.print_exc()
            value = None
        return value

    def criticalSegmFileRequiredButNoneAvailable(self):
        model_name = f'{self.model_name} model'
        action_txt = (
            'Please, segment the correct channel before using '
            f'{self.model_name}.'
        )
        if self.model_name == 'skip_segmentation':
            model_name = 'Skipping the segmentation'
            action_txt = (
                'To be able to skip the segmentation step, you need '
                'create at least one segmentation file.'
            )
        txt = html_utils.paragraph(f"""
            <b>{model_name}</b> 
            <b>requires an additional segmentation file</b> 
            but there are none available!<br><br>
            {action_txt}
            <br><br>Thank you for you patience!
        """)
        msg = widgets.myMessageBox(wrapText=False)
        msg.warning(self, 'Segmentation file required', txt)
        raise FileNotFoundError(
            'Model requires segmentation file but none are available.'
        )
    
    
    def checkAddSegmEndnameCombobox(self, ArgSpec, groupBoxLayout, row):
        if ArgSpec.name != 'Auxiliary segmentation file':
            return False
        
        if self.segmFileEndnames is None or not self.segmFileEndnames:
            self.criticalSegmFileRequiredButNoneAvailable()
        
        label = QLabel(f'{ArgSpec.name}:  ')
        groupBoxLayout.addWidget(
            label, row, 0, alignment=Qt.AlignRight
        )
        items = self.segmFileEndnames
        self.segmEndnameCombobox = widgets.QCenteredComboBox()
        self.segmEndnameCombobox.addItems(items)
        groupBoxLayout.addWidget(self.segmEndnameCombobox, row, 1, 1, 2)
        return True
        
    
    def createGroupParams(self, ArgSpecs_list, groupName, addChannelSelector=False):
        ArgsWidgets_list = []
        groupBox = QGroupBox(groupName)
        groupBoxLayout = QGridLayout()

        start_row = 0
        if self.is_tracker and self.channels is not None and addChannelSelector:
            label = QLabel(f'Input image:  ')
            groupBoxLayout.addWidget(
                label, start_row, 0, alignment=Qt.AlignRight
            )
            items = ['None', *self.channels]
            self.channelCombobox = widgets.QCenteredComboBox()
            self.channelCombobox.addItems(items)
            groupBoxLayout.addWidget(self.channelCombobox, start_row, 1, 1, 2)
            if self.currentChannelName is not None:
                self.channelCombobox.setCurrentText(self.currentChannelName)
            infoText = (
                'Some trackers require the intensity image as input.<br><br>'
                'If this one does not require it, leave the selected value '
                'to `None`.'
            )
            infoButton = self.getInfoButton('Input image', infoText)
            groupBoxLayout.addWidget(infoButton, start_row, 3)
            start_row += 1
        
        addSecondChannelSelector = addChannelSelector
        if len(ArgSpecs_list) > 0:
            if addSecondChannelSelector and ArgSpecs_list[0].docstring is not None:
                isSingleChannel = ArgSpecs_list[0].docstring.lower().find(
                    'single channel only'
                ) != -1
                if isSingleChannel:
                    addSecondChannelSelector = False
        
        isDualChannelModel = (
            self.model_name.find('cellpose') != -1 
            or any([
                _types.is_second_channel_type(ArgSpec.type) 
                for ArgSpec in ArgSpecs_list
            ])
        )
        askSecondChannel = isDualChannelModel and addSecondChannelSelector
        
        if askSecondChannel:
            label = QLabel('Second channel (optional):  ')
            groupBoxLayout.addWidget(label, start_row, 0, alignment=Qt.AlignRight)
            self.channelsCombobox = widgets.QCenteredComboBox()
            groupBoxLayout.addWidget(self.channelsCombobox, start_row, 1, 1, 2)
            infoText = (
                'Some models can merge two channels (e.g., cyto + '
                'nucleus) to obtain better perfomance.\n\n'
                'Select a channel as additional input to the model.'
            )
            infoButton = self.getInfoButton('Second channel', infoText)
            groupBoxLayout.addWidget(infoButton, start_row, 3)
            start_row += 1
        
        exclusive_withs = dict()
        default_exclusives = dict()
        row_mapper = dict()
        for row, ArgSpec in enumerate(ArgSpecs_list):
            if _types.is_second_channel_type(ArgSpec.type):
                continue
            
            if _types.is_widget_not_required(ArgSpec):
                continue
            
            row = row + start_row
            skip = self.checkAddSegmEndnameCombobox(
                ArgSpec, groupBoxLayout, row
            )
            if skip:
                continue
            
            arg_name = ArgSpec.name         
            var_name = arg_name.replace('_', ' ')
            var_name = f'{var_name[0].upper()}{var_name[1:]}'
            label = QLabel(f'{var_name}:  ')
            metadata_val = self.getValueFromMetadata(ArgSpec.name)
            groupBoxLayout.addWidget(label, row, 0, alignment=Qt.AlignRight)
            try:
                values = ArgSpec.type().values
                isCustomListType = True
            except Exception as err:
                isCustomListType = False
            
            isVectorEntry = False
            try:
                if isinstance(ArgSpec.type(), _types.Vector):
                    isVectorEntry = True
            except Exception as err:
                pass
            
            isFolderPath = False
            try:
                if isinstance(ArgSpec.type(), _types.FolderPath):
                    isFolderPath = True
            except Exception as err:
                pass

            try:
                exclusive_with = ArgSpec.type().is_exclusive_with
            except Exception as err:
                exclusive_with = []
            
            try:
                default_exclusive = ArgSpec.type().default_exclusive
            except Exception as err:
                default_exclusive = ''

            exclusive_withs[arg_name] = exclusive_with
            default_exclusives[arg_name] = default_exclusive
            row_mapper[arg_name] = row
            
            isCustomWidget = hasattr(ArgSpec.type, 'isWidget')
            
            if isCustomWidget:
                widget = ArgSpec.type().widget
                defaultVal = ArgSpec.default
                valueSetter = widget.setValue
                valueGetter = widget.value
                changeSig = widget.sigValueChanged
                groupBoxLayout.addWidget(widget, row, 1, 1, 2)
            elif isVectorEntry:
                vectorLineEdit = widgets.VectorLineEdit()
                vectorLineEdit.setValue(ArgSpec.default)
                defaultVal = ArgSpec.default
                valueSetter = widgets.VectorLineEdit.setValue
                valueGetter = widgets.VectorLineEdit.value
                changeSig = vectorLineEdit.valueChanged
                widget = vectorLineEdit
                groupBoxLayout.addWidget(vectorLineEdit, row, 1, 1, 2)
            elif isFolderPath:
                folderPathControl = widgets.FolderPathControl()
                folderPathControl.setText(str(ArgSpec.default))
                widget = folderPathControl
                defaultVal = str(ArgSpec.default)
                valueSetter = widgets.FolderPathControl.setText
                valueGetter = widgets.FolderPathControl.path
                changeSig = widget.sigValueChanged
                groupBoxLayout.addWidget(folderPathControl, row, 1, 1, 2)
            elif ArgSpec.type == bool:
                booleanGroup = QButtonGroup()
                booleanGroup.setExclusive(True)
                checkBox = widgets.Toggle()
                checkBox.setChecked(ArgSpec.default)
                defaultVal = ArgSpec.default
                valueSetter = widgets.Toggle.setChecked
                valueGetter = widgets.Toggle.isChecked
                changeSig = checkBox.toggled
                widget = checkBox
                groupBoxLayout.addWidget(
                    checkBox, row, 1, 1, 2, alignment=Qt.AlignCenter
                )
            elif ArgSpec.type == int:
                spinBox = widgets.SpinBox()
                if metadata_val is None:
                    spinBox.setValue(ArgSpec.default)
                else:
                    spinBox.setValue(int(metadata_val))
                    spinBox.isMetadataValue = True
                defaultVal = ArgSpec.default
                valueSetter = QSpinBox.setValue
                valueGetter = QSpinBox.value
                changeSig = spinBox.sigValueChanged
                widget = spinBox
                groupBoxLayout.addWidget(spinBox, row, 1, 1, 2)
            elif ArgSpec.type == float:
                doubleSpinBox = widgets.FloatLineEdit()
                if metadata_val is None:
                    doubleSpinBox.setValue(ArgSpec.default)
                else:
                    doubleSpinBox.setValue(float(metadata_val))
                    doubleSpinBox.isMetadataValue = True
                widget = doubleSpinBox
                defaultVal = ArgSpec.default
                valueSetter = widgets.FloatLineEdit.setValue
                valueGetter = widgets.FloatLineEdit.value
                changeSig = doubleSpinBox.valueChanged
                groupBoxLayout.addWidget(doubleSpinBox, row, 1, 1, 2)
            elif ArgSpec.type == os.PathLike:
                filePathControl = widgets.filePathControl()
                filePathControl.setText(str(ArgSpec.default))
                widget = filePathControl
                defaultVal = str(ArgSpec.default)
                valueSetter = widgets.filePathControl.setText
                valueGetter = widgets.filePathControl.path
                changeSig = filePathControl.sigValueChanged
                groupBoxLayout.addWidget(filePathControl, row, 1, 1, 2)
            elif isCustomListType:
                items = ArgSpec.type().values
                defaultVal = str(ArgSpec.default)
                combobox = widgets.AlphaNumericComboBox()
                combobox.addItems(items)
                combobox.setCurrentValue(defaultVal)
                valueSetter = widgets.AlphaNumericComboBox.setCurrentValue
                valueGetter = widgets.AlphaNumericComboBox.currentValue
                changeSig = combobox.currentTextChanged
                widget = combobox
                groupBoxLayout.addWidget(combobox, row, 1, 1, 2)
            else:
                lineEdit = QLineEdit()
                lineEdit.setText(str(ArgSpec.default))
                lineEdit.setAlignment(Qt.AlignCenter)
                widget = lineEdit
                defaultVal = str(ArgSpec.default)
                valueSetter = QLineEdit.setText
                valueGetter = QLineEdit.text
                changeSig = lineEdit.editingFinished
                groupBoxLayout.addWidget(lineEdit, row, 1, 1, 2)
            
            if ArgSpec.desc:
                infoButton = self.getInfoButton(ArgSpec.name, ArgSpec.desc)
                groupBoxLayout.addWidget(infoButton, row, 3)
            
            argsInfo = dialogs_preprocessing.ArgWidget(
                name=ArgSpec.name,
                type=ArgSpec.type,
                widget=widget,
                defaultVal=defaultVal,
                valueSetter=valueSetter,
                valueGetter=valueGetter,
                changeSig=changeSig
            )
            ArgsWidgets_list.append(argsInfo)
        
        exclusive_group = core.connected_components_in_undirected_graph(
            exclusive_withs
        )

        for group in exclusive_group:
            if len(group) == 1:
                continue
            for arg_name in group:
                default_exclusive = default_exclusives[arg_name]
                row = row_mapper[arg_name]

                argsInfo = ArgsWidgets_list[row]
                valueSetter = argsInfo.valueSetter
                widget = argsInfo.widget
                valueGetter = argsInfo.valueGetter

                argsInfo.valueGetter = qutils.replace_certain_vals(
                    argsInfo.valueGetter, default_exclusive, None
                )

                for arg_name_other in group:
                    if arg_name == arg_name_other:
                        continue
                    row_other = row_mapper[arg_name_other]
                    argsInfo_other = ArgsWidgets_list[row_other]
                    changeSig_other = argsInfo_other.changeSig
                    changeSig_other.connect(
                        partial(qutils.set_exclusive_valueSetter, widget, 
                                valueSetter, default_exclusive)
                    )

        groupBoxLayout.setColumnStretch(0, 0)
        groupBoxLayout.setColumnStretch(1, 1)
        groupBoxLayout.setColumnStretch(3, 0)
        nrows = groupBoxLayout.rowCount()
        groupBoxLayout.setRowStretch(nrows, 1)
        
        groupBox.setLayout(groupBoxLayout)
        return groupBox, ArgsWidgets_list

    def getInfoButton(self, param_name, infoText):
        infoButton = widgets.infoPushButton()
        infoButton.param_name = param_name
        infoButton.setToolTip(
            f'Click to get more info about `{param_name}` parameter...'
        )
        infoButton.infoText = infoText
        infoButton.clicked.connect(self.showInfoParam)
        return infoButton
    
    def showInfoParam(self):
        text = self.sender().infoText
        text = text.replace('\n', '<br>')
        text = html_utils.rst_urls_to_html(text)
        text = html_utils.rst_to_html(text)
        text = html_utils.paragraph(text)
        param_name = self.sender().param_name
        msg = widgets.myMessageBox(wrapText=False)
        msg.information(self, f'Info about `{param_name}` parameter', text)
    
    def restoreDefaultInit(self):
        for argWidget in self.init_argsWidgets:
            defaultVal = argWidget.defaultVal
            widget = argWidget.widget
            valueSetter = argWidget.valueSetter
            qutils.set_exclusive_valueSetter(
                widget, valueSetter, defaultVal
            )

    def restoreDefaultSegment(self):
        for argWidget in self.argsWidgets:
            defaultVal = argWidget.defaultVal
            widget = argWidget.widget
            valueSetter = argWidget.valueSetter
            qutils.set_exclusive_valueSetter(
                widget, valueSetter, defaultVal
            )

    def restoreDefaultExtra(self):
        for argWidget in self.extraArgsWidgets:
            defaultVal = argWidget.defaultVal
            widget = argWidget.widget
            valueSetter = argWidget.valueSetter
            qutils.set_exclusive_valueSetter(
                widget, valueSetter, defaultVal
            )

    def restoreDefaultPostprocess(self):
        self.postProcessGroupbox.restoreDefault()

    def readLastSelection(self):
        self.ini_path = os.path.join(settings_folderpath, self.ini_filename)
        
        if not os.path.exists(self.ini_path):
            return None

        print(f'Reading last selected parameters from: {self.ini_path}')
        configPars = config.ConfigParser()
        configPars.read(self.ini_path)
        return configPars

    def setValuesFromParams(self, init_params, segment_params, extra_params=None):
        sections = {
            f'{self.model_name}.init': (init_params, self.init_argsWidgets),
            f'{self.model_name}.segment': (segment_params, self.argsWidgets),
        }
        if extra_params is not None:
            sections[f'{self.model_name}.extra'] = (
                extra_params, self.extraArgsWidgets
            )
            
        for section, values in sections.items():
            params, argWidgetList = values
            for argWidget in argWidgetList:
                val = params.get(argWidget.name)
                widget = argWidget.widget
                if val is None:
                    continue
                casters = [lambda x: x, int, float, str, bool]
                for caster in casters:
                    try:
                        argWidget.valueSetter(widget, caster(val))
                        break
                    except Exception as e:
                        continue
    
    def loadLastSelection(
            self, section, argWidgetList, checked=False, configPars=None
        ):        
        if self.configPars is None and configPars is None:
            return

        if configPars is None:
            configPars = self.configPars
        
        getters = ['getboolean', 'getint', 'getfloat', 'get']
        try:
            options = configPars.options(section)
        except Exception:
            return
        
        for argWidget in argWidgetList:
            option = argWidget.name
            val = None
            for getter in getters:
                try:
                    val = getattr(configPars, getter)(section, option)
                    break
                except Exception as err:
                    pass
            widget = argWidget.widget
            
            if hasattr(widget, 'isMetadataValue'):
                continue
            if val is None:
                continue
            
            casters = [lambda x: x, int, float, str, bool]
            for caster in casters:
                try:
                    val = caster(val)
                    valueSetter = argWidget.valueSetter
                    qutils.set_exclusive_valueSetter(
                        widget, valueSetter, val
                    )
                    break
                except Exception as e:
                    printl(traceback.format_exc())
                    continue

    def loadLastSelectionPostProcess(self, checked=False, configPars=None):
        if self.postProcessGroupbox is None:
            return
        
        postProcessSection = f'{self.model_name}.postprocess'

        if isinstance(configPars, bool):
            configPars = None

        if configPars is None:
            configPars = self.configPars

        if postProcessSection in configPars.sections():
            try:
                minSize = configPars.getint(
                    postProcessSection, 'minSize', fallback=10
                )
            except ValueError:
                minSize = 10

            try:
                minSolidity = configPars.getfloat(
                    postProcessSection, 'minSolidity', fallback=0.5
                )
            except ValueError:
                minSolidity = 0.5

            try: 
                maxElongation = configPars.getfloat(
                    postProcessSection, 'maxElongation', fallback=3
                )
            except ValueError:
                maxElongation = 3
            
            try:
                minObjSizeZ = configPars.getint(
                    postProcessSection, 'min_obj_no_zslices', fallback=3
                )
            except ValueError:
                minObjSizeZ = 3
            
            kwargs = {
                'min_solidity': minSolidity,
                'min_area': minSize,
                'max_elongation': maxElongation,
                'min_obj_no_zslices': minObjSizeZ
            }
            self.postProcessGroupbox.restoreFromKwargs(kwargs)

            applyPostProcessing = configPars.getboolean(
                postProcessSection, 'applyPostProcessing',
                fallback=configPars.getboolean(
                    postProcessSection, 'applied_postprocessing', fallback=False
                ),
            )
            self.postProcessGroupbox.setChecked(applyPostProcessing)

        customPostProcessSection = f'{self.model_name}.custom_postprocess'
        if customPostProcessSection not in configPars.sections():
            return
        
        selectFeaturesWidget = (
            self.postProcessGroupbox.selectedFeaturesDialog.groupbox
        )
        selectFeaturesWidget.resetFields()
        f = 0
        for col_name, value in configPars[customPostProcessSection].items():
            low, high = value.split(',')
            low = low.strip()
            high = high.strip()
            if f > 0:
                selectFeaturesWidget.addFeatureField()

            selector = selectFeaturesWidget.selectors[f]
            selector.selectButton.setText(col_name)
            selector.selectButton.setFlat(True)

            feature_group = measurements.get_metric_group_name(col_name)
            selector.featureGroup = feature_group

            if low != 'None':
                try:
                    low_val = int(low)
                except ValueError:
                    low_val = float(low)

                selector.lowRangeWidgets.checkbox.setChecked(True)
                selector.lowRangeWidgets.spinbox.setValue(low_val)

            if high != 'None':
                try:
                    high_val = int(high)
                except ValueError:
                    high_val = float(high)

                selector.highRangeWidgets.checkbox.setChecked(True)
                selector.highRangeWidgets.spinbox.setValue(high_val)

            f += 1

    def createSeeHereLabel(self, url):
        htmlTxt = f'<a href=\"{url}">here</a>'
        seeHereLabel = QLabel()
        seeHereLabel.setText(f"""
            <p style="font-size:13px">
                See {htmlTxt} for details on the parameters
            </p>
        """)
        seeHereLabel.setTextFormat(Qt.RichText)
        seeHereLabel.setTextInteractionFlags(Qt.TextBrowserInteraction)
        seeHereLabel.setOpenExternalLinks(True)
        seeHereLabel.setStyleSheet("padding:12px 0px 0px 0px;")
        return seeHereLabel

    def argsWidgets_to_kwargs(self, argsWidgets):
        kwargs_dict = {
            argWidget.name:argWidget.valueGetter(argWidget.widget)
            for argWidget in argsWidgets
        }
        return kwargs_dict

    def getInitKwargs(self):
        init_kwargs = self.argsWidgets_to_kwargs(self.init_argsWidgets)
        if hasattr(self, 'segmEndnameCombobox'):
            init_kwargs['segm_endname'] = (
                self.segmEndnameCombobox.currentText()
            )
        
        return init_kwargs
    
    def getModelKwargs(self):
        if self.skipSegmentation:
            return {}
        
        return self.argsWidgets_to_kwargs(self.argsWidgets)
    
    def getExtraKwargs(self):
        if self.extraArgsWidgets is None:
            return {}
        
        return self.argsWidgets_to_kwargs(self.extraArgsWidgets)
    
    def ok_cb(self, checked):
        self.cancel = False
        self.preproc_recipe = None
        if self.preProcessParamsWidget is not None:
            self.preproc_recipe = self.preProcessParamsWidget.recipe()
            if self.preproc_recipe is None:
                return
            
        self.init_kwargs = self.getInitKwargs()

        if self.extraArgsWidgets:
            self.extra_kwargs = self.getExtraKwargs()

        self.model_kwargs = self.getModelKwargs()
        self.segment_kwargs = self.model_kwargs

        if self.postProcessGroupbox is not None:
            self.applyPostProcessing = self.postProcessGroupbox.isChecked()
            self.standardPostProcessKwargs = self.postProcessGroupbox.kwargs()
        self.secondChannelName = None
        if hasattr(self, 'channelsCombobox'):
            self.secondChannelName = self.channelsCombobox.currentText()
        if self.secondChannelName == 'None':
            self.secondChannelName = None
        self.inputChannelName = 'None'
        if self.channelCombobox is not None:
            self.inputChannelName = self.channelCombobox.currentText()
        
        self.reduceMemoryUsage = False
        if hasattr(self, 'reduceMemUsageToggle'):
            self.reduceMemoryUsage = self.reduceMemUsageToggle.isChecked()
        self.customPostProcessFeatures = self.selectedFeaturesRange()
        self.customPostProcessGroupedFeatures = self.groupedFeatures()
        self.saveLastSelection()
        self.freePosData()
        self.close()

    def freePosData(self):
        if hasattr(self, 'postProcessGroupbox'):
            try:
                for selector in self.postProcessGroupbox.selectedFeaturesDialog.groupbox.selectors:
                    qutils.hardDelete(selector)
            except AttributeError:
                pass
            try:
                qutils.hardDelete(self.postProcessGroupbox.selectedFeaturesDialog.groupbox)
            except AttributeError:
                pass
            try:
                qutils.hardDelete(self.postProcessGroupbox.selectedFeaturesDialog)
            except AttributeError:
                pass
            try:
                qutils.hardDelete(self.postProcessGroupbox)
            except AttributeError:
                pass

    def getConfigPars(self, create_new=False):
        if self.configPars is None or create_new:
            configPars = config.ConfigParser()
        else:
            configPars = self.configPars
        
        if self.preProcessParamsWidget is not None:
            preprocCp = self.preProcessParamsWidget.recipeConfigPars(
                self.model_name
            )
            for section in preprocCp.sections():
                configPars[section] = preprocCp[section]
        
        configPars[f'{self.model_name}.init'] = {}
        configPars[f'{self.model_name}.segment'] = {}
        configPars[f'{self.model_name}.extra'] = {}

        init_kwargs = self.getInitKwargs()
        model_kwargs = self.getModelKwargs()
        
        for key, val in init_kwargs.items():
            configPars[f'{self.model_name}.init'][key] = str(val)
        for key, val in model_kwargs.items():
            configPars[f'{self.model_name}.segment'][key] = str(val)
        if self.extraArgsWidgets:
            extra_kwargs = self.getExtraKwargs()
            for key, val in extra_kwargs.items():
                configPars[f'{self.model_name}.extra'][key] = str(val)

        configPars[f'{self.model_name}.postprocess'] = {}
        if self.postProcessGroupbox is not None:
            postProcKwargs = self.postProcessGroupbox.kwargs()
            postProcessConfig = configPars[f'{self.model_name}.postprocess']
            postProcessConfig['minSize'] = str(postProcKwargs['min_area'])
            postProcessConfig['minSolidity'] = str(postProcKwargs['min_solidity'])
            postProcessConfig['maxElongation'] = str(
                postProcKwargs['max_elongation']
            )
            postProcessConfig['min_obj_no_zslices'] = str(
                postProcKwargs['min_obj_no_zslices']
            )
            postProcessConfig['applyPostProcessing'] = str(
                self.postProcessGroupbox.isChecked()
            )
        
        custom_postproc_section = f'{self.model_name}.custom_postprocess'
        configPars[custom_postproc_section] = {}
        if self.postProcessGroupbox is not None:
            selectFeaturesWidget = (
                self.postProcessGroupbox.selectedFeaturesDialog.groupbox
            )
            for selector in selectFeaturesWidget.selectors:
                col_name = selector.selectButton.text()
                lowStr = 'None'
                highStr = 'None'
                if selector.lowRangeWidgets.checkbox.isChecked():
                    lowVal = selector.lowRangeWidgets.spinbox.value()
                    lowStr = str(lowVal)
                if selector.highRangeWidgets.checkbox.isChecked():
                    highVal = selector.highRangeWidgets.spinbox.value()
                    highStr = str(highVal)

                configPars[custom_postproc_section][col_name] = (
                    f'{lowStr}, {highStr}'
                )

        
        return configPars
    
    def saveLastSelection(self):
        self.configPars = self.getConfigPars()
        with open(self.ini_path, 'w') as configfile:
            self.configPars.write(configfile)

        mode = 'Segmentation' if not self.is_tracker else 'Tracking'
        
        print(f'{mode} parameters saved at "{self.ini_path}"')
        
    def exec_(self):
        self.show(block=True)

    def show(self, block=False):
        _window_stacking.set_window_flags_on_top(self, Qt.Window)
        super().show()
        if self.model_name == 'thresholding':
            self.segmentGroupBox.setDisabled(True)
        if block:
            self.loop = QEventLoop()
            self.loop.exec_()

    def closeEvent(self, event):
        self.freePosData()
        if hasattr(self, 'loop'):
            self.loop.exit()

    def cancel_cb(self, checked):
        self.cancel = True
        self.freePosData()
    
    def showEvent(self, event) -> None:
        buttonHeight = self.okButton.minimumSizeHint().height()
        heightInitParams = (
            self.initParamsScrollArea.minimumHeightNoScrollbar()
        )
        self.initParamsScrollArea.setMinimumHeight(
            heightInitParams
        )
        heightLeft = 70 + buttonHeight
        heightCenter = heightInitParams
        heightRight = 0
        if self.segmentParamsScrollArea is not None:
            heightSegmentParams = (
                self.segmentParamsScrollArea.minimumHeightNoScrollbar()
            )
            heightCenter += (heightSegmentParams+ 70 + buttonHeight)
            
            rowInitParams, _ = self.paramsGroupPosMapper[self.initParamsScrollArea]
            rowSegmParams, _ = self.paramsGroupPosMapper[self.segmentParamsScrollArea]
            
            numInitParams = len(self.init_params)
            numSegmentParams = len(self.segment_params)
            
            try:
                segmentParamsStretch = max(1, round(numSegmentParams/numInitParams))
            except ZeroDivisionError as err:
                segmentParamsStretch = 1
            self.secondColLayout.setStretch(rowInitParams, 1)
            self.secondColLayout.setStretch(rowSegmParams, segmentParamsStretch)
            
        if self.extraParamsScrollArea is not None:
            heightRight += (
                self.extraParamsScrollArea.minimumHeightNoScrollbar()
                + 70 + buttonHeight
            )
            
        
        if self.additionalSegmGroupbox is not None:
            heightRight += self.additionalSegmGroupbox.minimumSizeHint().height()
            heightRight += buttonHeight
        if self.preProcessParamsWidget is not None:
            heightPreprocParams = (
                self.preProcessParamsWidget.minimumSizeHint().height()
            )
            heightLeft += heightPreprocParams
            heightLeft += buttonHeight
        if self.postProcessGroupbox is not None:
            heightRight += self.postProcessGroupbox.minimumSizeHint().height()
            heightRight += buttonHeight
        if self.seeHereLabel is not None:
            heightRight += self.seeHereLabel.minimumSizeHint().height()
        height = max(heightLeft, heightRight, heightCenter)
        screenHeight = self.screen().size().height()
        screenGeom = self.screen().geometry()
        screenLeft = screenGeom.left()
        screenRight = screenGeom.right()
        screenCenter = (screenLeft + screenRight) / 2
        width = self.sizeHint().width()
        windowLeft = int(screenCenter - width/2)
        self.move(windowLeft, 20)
        
        if height >= screenHeight - 150:
            height = screenHeight - 150
        self.resize(width, height)


class SelectFeaturesRange:
    def __init__(
            self, 
            posData, 
            force_postprocess_2D=False, 
            qparent=None,
            sigValueChanged=None
        ) -> None:
        self.posData = posData
        self.qparent = qparent
        self.force_postprocess_2D = force_postprocess_2D
        self.sigValueChanged = sigValueChanged
        
        self.lowRangeWidgets = widgets.CheckableSpinBoxWidgets()
        self.highRangeWidgets = widgets.CheckableSpinBoxWidgets()        
        
        self.selectButton = widgets.FeatureSelectorButton(
            'Click to select feature...'
        )
        self.selectButton.setSizeLongestText(
            'Spotfit intens. metric, Foregr. integral gauss. peak'
        )
        self.selectButton.clicked.connect(self.selectFeature)
        self.selectButton.setCursor(Qt.PointingHandCursor)

        self.selectedFeatureGroups = {}

        self.widgets = [
            {'pos': (0, 0), 'widget': self.lowRangeWidgets.checkbox}, 
            {'pos': (1, 0), 'widget': self.lowRangeWidgets.spinbox}, 
            {'pos': (1, 1), 'widget': widgets.LessThanPushButton(flat=True)},
            {'pos': (1, 2), 'widget': self.selectButton},
            {'pos': (1, 3), 'widget': widgets.LessThanPushButton(flat=True)},
            {'pos': (0, 4), 'widget': self.highRangeWidgets.checkbox},
            {'pos': (1, 4), 'widget': self.highRangeWidgets.spinbox}, 
            {'pos': (2, 0), 'widget': widgets.VerticalSpacerEmptyWidget(height=10)}
        ]
        self.columnsStretches = {0: 0, 1: 0, 2: 1, 3: 0, 4: 0}
    
    def setText(self, text):
        self.selectButton.setText(text)
    
    def selectFeature(self):
        loadedChNames = [self.posData.user_ch_name]
        notLoadedChNames = []
        isZstack = self.posData.SizeZ > 1 and not self.force_postprocess_2D
        isSegm3D = self.posData.isSegm3D and not self.force_postprocess_2D
        self.selectFeatureDialog = dialogs_set_measurements.SetMeasurementsDialog(
            loadedChNames, notLoadedChNames, isZstack, isSegm3D,
            posData=self.posData, parent=self.qparent,
            isSingleSelection=True, is_concat=True
        )
        # self.selectFeatureDialog.resizeVertical()
        self.selectFeatureDialog.sigClosed.connect(self.setFeatureText)
        self.selectFeatureDialog.show()
    
    def setFeatureText(self):
        if self.selectFeatureDialog.cancel:
            return
        self.selectButton.setFlat(True)
        selectedMetricName, selectedMetricGroup = (
            self.selectFeatureDialog.selectedMetricNameAndGroup()
        )
        self.selectButton.setText(selectedMetricName)
        self.featureGroup = selectedMetricGroup

class SelectFeaturesRangeDialog(dialogs_base.QBaseDialog):
    sigValueChanged = Signal(object)
    
    def __init__(self, posData=None, parent=None, force_postprocess_2D=False):
        super().__init__(parent)
        
        self.force_postprocess_2D = force_postprocess_2D
        
        layout = QVBoxLayout()
        self.setWindowTitle('Custom features for post-processing')
        
        self.groupbox = SelectFeaturesRangeGroupbox(
            posData=posData, parent=parent, 
            force_postprocess_2D=force_postprocess_2D
        )
        
        buttonsLayout = QHBoxLayout()
        okPushButton = widgets.okPushButton(' Ok ')
        
        buttonsLayout.addStretch(1)
        buttonsLayout.addWidget(okPushButton)
        
        okPushButton.clicked.connect(self.ok_cb)

        layout.addWidget(self.groupbox)
        layout.addSpacing(10)
        layout.addLayout(buttonsLayout)
        
        self.setLayout(layout)
    
    def ok_cb(self):
        if self.groupbox.selectedFeaturesRange():
            self.sigValueChanged.emit(None)
        self.hide()

class SelectFeaturesRangeGroupbox(QGroupBox):
    def __init__(
            self, posData=None, parent=None, force_postprocess_2D=False
        ):
        super().__init__(parent)

        self.setTitle('Features and thresholds for filtering segmented objects')
        # self.setCheckable(True)

        self.posData = posData
        self.force_postprocess_2D = force_postprocess_2D
        
        self._layout = QGridLayout()
        self._layout.setVerticalSpacing(0)

        firstSelector = SelectFeaturesRange(
            posData, force_postprocess_2D=force_postprocess_2D
        )
        self.addButton = widgets.addPushButton('  Add feature    ')
        self.addButton.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        for col, widget in enumerate(firstSelector.widgets):
            row, col = widget['pos']
            self._layout.addWidget(widget['widget'], row, col)
        for col, stretch in firstSelector.columnsStretches.items():
            self._layout.setColumnStretch(col, stretch)
            
        lastCol = self._layout.columnCount()
        self._layout.addWidget(self.addButton, 0, lastCol+1, 2, 1)
        self.lastCol = lastCol+1
        self.selectors = [firstSelector]

        self.setLayout(self._layout)

        # self.setFont(fonts.font)

        self.addButton.clicked.connect(self.addFeatureField)

    def addFeatureField(self):
        row = self._layout.rowCount()
        selector = SelectFeaturesRange(
            self.posData, force_postprocess_2D=self.force_postprocess_2D
        )
        delButton = widgets.delPushButton('Remove feature')
        delButton.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        delButton.selector = selector
        selector.delButton = delButton
        for col, widget in enumerate(selector.widgets):
            relRow, col = widget['pos']
            self._layout.addWidget(widget['widget'], relRow+row, col)
        self._layout.addWidget(delButton, row, self.lastCol, 2, 1)
        self.selectors.append(selector)
        delButton.clicked.connect(self.removeFeatureField)
    
    def resetFields(self):
        while len(self.selectors) > 1:
            selector = self.selectors[-1]
            selector.delButton.click()
        firstSelector = self.selectors[0]
        firstSelector.selectButton.setText('Click to select feature...')
        firstSelector.lowRangeWidgets.checkbox.setChecked(False)
        firstSelector.highRangeWidgets.checkbox.setChecked(False)
    
    def removeFeatureField(self):
        delButton = self.sender()
        for widget in delButton.selector.widgets:
            self._layout.removeWidget(widget['widget'])
        self._layout.removeWidget(delButton)
        self.selectors.remove(delButton.selector)
    
    def selectedFeaturesRange(self):
        featuresRange = {}
        for selector in self.selectors:
            if selector.selectButton.text().find('Click') != -1:
                continue
            featuresRange[selector.selectButton.text()] = (
                selector.lowRangeWidgets.value(), 
                selector.highRangeWidgets.value()
            )
        return featuresRange

    def selectedFeaturesGroup(self):
        featuresGroup = {}
        for selector in self.selectors:
            if selector.selectButton.text().find('Click') != -1:
                continue
            group = selector.featureGroup                
            featuresGroup[selector.selectButton.text()] = group
        return featuresGroup

    def groupedFeatures(self):
        featuresGroup = self.selectedFeaturesGroup()
        groupedFeatures = {}
        for feature, group in featuresGroup.items():
            group = featuresGroup[feature]
            if isinstance(group, str):
                key = group
                if key not in groupedFeatures:
                    groupedFeatures[key] = []
                groupedFeatures[key].append(feature)
            else:
                key, channel = list(group.items())[0]
                if key not in groupedFeatures:
                    groupedFeatures[key] = {}
                if channel not in groupedFeatures[key]:
                    groupedFeatures[key][channel] = []
                groupedFeatures[key][channel].append(feature)
        return groupedFeatures
    
    def setValue(self, value):
        pass
