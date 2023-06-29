import os
import sys

import matplotlib.pyplot as plt

from importlib import import_module

from cellacdc import apps, myutils, widgets, data, core, load

from qtpy.QtWidgets import QApplication, QStyleFactory

import skimage.color

try:
    import pytest
    pytest.skip('skipping this test since it is gui based', allow_module_level=True)
except Exception as e:
    pass

gdrive_path = myutils.get_gdrive_path()

FRAME_I = 0
test_data = data.Cdc42TimeLapseData()
# image_data = test_data.image_data()
image_data = test_data.cdc42_data()
segm_data = test_data.segm_data()

img = image_data[FRAME_I]

# Ask which model to use --> Test if new model is visible
app = QApplication(sys.argv)
app.setStyle(QStyleFactory.create('Fusion'))

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

segm_files = load.get_segm_files(test_data.images_path)
existingSegmEndnames = load.get_existing_segm_endnames(
    test_data.basename, segm_files
)
win = apps.QDialogModelParams(
    init_params,
    segment_params,
    model_name, url=url,
    segmFileEndnames=existingSegmEndnames
)

win.exec_()

# Initialize model
try:
    model = acdcSegment.Model(**win.init_kwargs)
except Exception as e:
    model = acdcSegment.Model(segm_data, **win.init_kwargs)

if img.ndim == 3 and (img.shape[-1] == 3 or img.shape[-1] == 4):
    img = skimage.color.rgb2gray(img)

print(img.shape)

lab = core.segm_model_segment(model, img, win.model_kwargs, frame_i=FRAME_I)

print(lab.shape)

if model_name == 'YeastMate':
    cca_df = model.predictCcaState(img)
    print(cca_df)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[1].imshow(lab)
plt.show()
