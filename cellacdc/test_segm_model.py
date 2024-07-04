import os
import sys

import numpy as np

import matplotlib.pyplot as plt

from importlib import import_module

from cellacdc._run import _setup_app
from cellacdc import apps, myutils, widgets, data, core, load
from cellacdc import prompts

import skimage.color

try:
    import pytest
    pytest.skip('skipping this test since it is gui based', allow_module_level=True)
except Exception as e:
    pass

gdrive_path = myutils.get_gdrive_path()

FRAME_I = None # 0 # 300
# test_data = data.Cdc42TimeLapseData()
# # image_data = test_data.image_data()
# image_data = test_data.cdc42_data()

# test_data = data.pomBseenDualChannelData()
# test_data = data.BABYtestData()
test_data = data.YeastMitoSnapshotData()

image_data = test_data.image_data()
# segm_data = test_data.segm_data()
images_path = test_data.images_path
posData = test_data.posData()
posData.loadOtherFiles(load_segm_data=False, load_metadata=True)
posData.buildPaths()

if FRAME_I is None:
    img = image_data
else:
    img = image_data[FRAME_I]

app, splashScreen = _setup_app(splashscreen=True)  
splashScreen.close()

from cellacdc.plot import imshow
imshow(img)

cellacdc_path = os.path.dirname(os.path.abspath(__file__))
models = myutils.get_list_of_models()
win = widgets.QDialogListbox(
    'Select model',
    'Select model to use for segmentation: ',
    models,
    multiSelection=False
)
win.exec_()

if win.cancel:
    sys.exit('Execution aborted')

model_name = win.selectedItemsText[0]
if model_name == 'Automatic thresholding':
    model_name = 'thresholding'
# Check if model needs to be downloaded
downloadWin = apps.downloadModel(model_name, parent=None)
downloadWin.download()

# Load model as a module
acdcSegment = myutils.import_segment_module(model_name)

# Read all models parameters
init_params, segment_params = myutils.getModelArgSpec(acdcSegment)

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


model = myutils.init_segm_model(acdcSegment, posData, win.init_kwargs)
is_segment3DT_available = any(
    [name=='segment3DT' for name in dir(model)]
)

if img.ndim == 3 and (img.shape[-1] == 3 or img.shape[-1] == 4):
    img = skimage.color.rgb2gray(img)

print('Input image shape: ', img.shape)

lab = core.segm_model_segment(
    model, img, win.model_kwargs, frame_i=FRAME_I, 
    preproc_recipe=win.preproc_recipe, posData=posData
)

from cellacdc.plot import imshow
imshow(img, lab)
