import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from importlib import import_module

from cellacdc._run import _setup_app
from cellacdc import data, core, load
from cellacdc.myutils import filesystem, segm_models
from cellacdc.shared.dialogs import model_selection as dialogs_model_selection
from cellacdc.shared.widgets import (
    item_views as widgets_item_views,
    line_edits as widgets_line_edits,
    numeric_inputs as widgets_numeric_inputs,
    push_buttons as widgets_push_buttons,
)
from cellacdc import prompts

import skimage.color
import qtpy.compat
from qtpy.QtWidgets import QVBoxLayout
from cellacdc.shared.dialogs import base as dialogs_base


class TestSegmModelInitalDialog(dialogs_base.QBaseDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.cancel = True
        
        mainLayout = QVBoxLayout()
        entriesLayout = widgets_line_edits.FormLayout()
        
        row = 0
        self.startFrameNumberSpinbox = widgets_numeric_inputs.SpinBox()
        self.startFrameNumberSpinbox.setMinimum(1)
        
        self.startFrameNumberFormWidget = widgets_line_edits.formWidget(
            self.startFrameNumberSpinbox, 
            labelTextLeft='Start frame number',
            addActivateCheckbox=True
        )
        entriesLayout.addFormWidget(self.startFrameNumberFormWidget, row=row)
        
        row += 1
        self.stopFrameNumberSpinbox = widgets_numeric_inputs.SpinBox()
        self.stopFrameNumberSpinbox.setMinimum(1)
        
        self.stopFrameNumberFormWidget = widgets_line_edits.formWidget(
            self.stopFrameNumberSpinbox, 
            labelTextLeft='Stop frame number',
            addActivateCheckbox=True
        )
        entriesLayout.addFormWidget(self.stopFrameNumberFormWidget, row=row)
        
        row += 1
        self.startZsliceNumberSpinbox = widgets_numeric_inputs.SpinBox()
        self.startZsliceNumberSpinbox.setMinimum(1)
        
        self.startZsliceNumberFormWidget = widgets_line_edits.formWidget(
            self.startZsliceNumberSpinbox, 
            labelTextLeft='Start z-slice number',
            addActivateCheckbox=True
        )
        entriesLayout.addFormWidget(self.startZsliceNumberFormWidget, row=row)
        
        row += 1
        self.stopZsliceNumberSpinbox = widgets_numeric_inputs.SpinBox()
        self.stopZsliceNumberSpinbox.setMinimum(1)
        
        self.stopZsliceNumberFormWidget = widgets_line_edits.formWidget(
            self.stopZsliceNumberSpinbox, 
            labelTextLeft='Stop z-slice number',
            addActivateCheckbox=True
        )
        entriesLayout.addFormWidget(self.stopZsliceNumberFormWidget, row=row)
        
        row += 1
        
        self.isTimelapseToggleFormWidget = widgets_line_edits.formWidget(
            widgets_push_buttons.Toggle(), 
            labelTextLeft='Is timelapse?',
            stretchWidget=False,
            valueGetterName='isChecked'
        )
        entriesLayout.addFormWidget(self.isTimelapseToggleFormWidget, row=row)
        
        
        # self.stopFrameNumberSpinbox
        # self.startZsliceNumberSpinbox
        # self.stopZsliceNumberSpinbox
        # self.isTimelapseToggle
        
        buttonsLayout = widgets_push_buttons.CancelOkButtonsLayout()

        buttonsLayout.okButton.clicked.connect(self.ok_cb)
        buttonsLayout.cancelButton.clicked.connect(self.close)
        
        mainLayout.addLayout(entriesLayout)
        mainLayout.addSpacing(20)
        mainLayout.addLayout(buttonsLayout)
        
        self.setLayout(mainLayout)
    
    def ok_cb(self):
        self.cancel = False
        
        self.start_frame_n = self.startFrameNumberFormWidget.value()
        self.stop_frame_n = self.stopFrameNumberFormWidget.value()
        self.start_z_slice_n = self.startZsliceNumberFormWidget.value()
        self.stop_z_slice_n = self.stopZsliceNumberFormWidget.value()
        self.is_timelapse = self.isTimelapseToggleFormWidget.value()
        
        self.close()


try:
    import pytest
    pytest.skip('skipping this test since it is gui based', allow_module_level=True)
except Exception as e:
    pass

test_data = None
# test_data = data.Cdc42TimeLapseData()
# # image_data = test_data.image_data()
# image_data = test_data.cdc42_data()

# test_data = data.pomBseenDualChannelData()
# test_data = data.BABYtestData()
# test_data = data.YeastMitoSnapshotData()

app, splashScreen = _setup_app(splashscreen=True)  
splashScreen.close()

initialWindow = TestSegmModelInitalDialog()
initialWindow.exec_()
if initialWindow.cancel:
    print('Execution cancelled.')
    exit()

start_frame_idx = initialWindow.start_frame_n
if start_frame_idx is not None:
    start_frame_idx -= 1
    
stop_frame_n = initialWindow.stop_frame_n 

start_z_slice_idx = initialWindow.start_z_slice_n
if start_z_slice_idx is not None:
    start_z_slice_idx -= 1
    
stop_z_slice_n = initialWindow.stop_z_slice_n
is_timelapse = initialWindow.is_timelapse

if test_data is None:
    tif_filepath, _ = qtpy.compat.getopenfilename(
        basedir=filesystem.getMostRecentPath(),
        filters=('Images (*.tif)')
    )
    if not tif_filepath:
        exit('Execution cancelled.')
    
    images_path = os.path.dirname(tif_filepath)
    basename = os.path.commonprefix(filesystem.listdir(images_path))
    filename, ext = os.path.splitext(os.path.basename(tif_filepath))
    channel = filename[len(basename):]
    posData = load.loadData(tif_filepath, channel)
    posData.loadImgData()
    image_data = posData.img_data
    images_path = posData.images_path
else:
    posData = test_data.posData()
    image_data = test_data.image_data()
    images_path = test_data.images_path
    
posData.loadOtherFiles(load_segm_data=False, load_metadata=True)
posData.buildPaths()

if is_timelapse:
    img = image_data[
        start_frame_idx:stop_frame_n, 
        start_z_slice_idx:stop_z_slice_n
    ]
else:
    img = image_data[start_z_slice_idx:stop_z_slice_n]

from cellacdc.plot import imshow
imshow(img)

cellacdc_path = os.path.dirname(os.path.abspath(__file__))
models = segm_models.get_list_of_models()
win = widgets_item_views.QDialogListbox(
    'Select model',
    'Select model to use for segmentation: ',
    models,
    multiSelection=False,
    allowEmptySelection=False
)
win.exec_()

if win.cancel:
    sys.exit('Execution aborted')

model_name = win.selectedItemsText[0]
if model_name == 'Automatic thresholding':
    model_name = 'thresholding'
# Check if model needs to be downloaded
downloadWin = dialogs_model_selection.downloadModel(model_name, parent=None)
downloadWin.download()

# Load model as a module
acdcSegment = segm_models.import_segment_module(model_name)

# Read all models parameters
init_params, segment_params = segm_models.getModelArgSpec(acdcSegment)

# Prompt user to enter the model parameters
try:
    url = acdcSegment.url_help()
except AttributeError:
    url = None

out = prompts.init_segm_model_params(
    posData, model_name, init_params, segment_params, 
    help_url=url, qparent=None, init_last_params=True
)
win = out.get('win')
if win.cancel:
    exit('Execution cancelled.')

# Initialize model
segm_data = None
init_kwargs = win.init_kwargs
segm_endname = init_kwargs.pop('segm_endname', None)
if segm_endname is not None:
    segm_filepath, _ = load.get_path_from_endname(segm_endname, images_path)
    segm_data = np.load(segm_filepath)['arr_0']


model = segm_models.init_segm_model(acdcSegment, posData, win.init_kwargs)
if model is None:
    sys.exit('Segmentation model was not initialized correctly!')
is_segment3DT_available = any(
    [name=='segment3DT' for name in dir(model)]
)

if img.ndim == 3 and (img.shape[-1] == 3 or img.shape[-1] == 4):
    img = skimage.color.rgb2gray(img)

print('Input image shape: ', img.shape)
print('Segmentation process started...')

lab = core.segm_model_segment(
    model, img, win.model_kwargs, frame_i=start_frame_idx, 
    preproc_recipe=win.preproc_recipe, posData=posData, 
    is_timelapse_model_and_data=is_segment3DT_available and is_timelapse,
)

from cellacdc.plot import imshow
imshow(
    img, 
    lab, 
    window_title=f'Result of segmenting with "{model_name}" model',
    axis_titles=['Input image', 'Segmentation result'], 
    annotate_labels_idxs=[1],
)
