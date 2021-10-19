import os
import sys

import matplotlib.pyplot as plt

from collections import namedtuple
from importlib import import_module

import apps, myutils

from PyQt5.QtWidgets import QApplication, QStyleFactory

# Ask which model to use --> Test if new model is visible
app = QApplication(sys.argv)
app.setStyle(QStyleFactory.create('Fusion'))


src_path = os.path.dirname(os.path.abspath(__file__))
models = os.listdir(os.path.join(src_path, 'models'))
win = apps.QDialogListbox(
    'Select model',
    'Select model to use for segmentation: ',
    models,
    multiSelection=False
)
win.exec_()

if win.cancel:
    sys.exit('Execution aborted')

# Load model as a module
model_name = win.selectedItemsText[0]
acdcSegment = import_module(f'models.{model_name}.acdcSegment')

# Read all models parameters
init_params, segment_params = myutils.getModelArgSpec(acdcSegment)

# Prompt user to enter the model parameters
try:
    url = acdcSegment.url_help()
except AttributeError:
    url = None

win = apps.QDialogModelParams(
    init_params,
    segment_params,
    model_name, url=url)

win.exec_()

# Initialize model
model = acdcSegment.Model(**win.init_kwargs)

# Use model on a test image
# In this case image is in 'Cell-ACDC/data/test_images' folder
import skimage.io

main_path = os.path.dirname(src_path)
test_images_path = os.path.join(main_path, 'data', 'test_images')

test_img_path = os.path.join(test_images_path, 'test_YeaZ.tif')

img = skimage.io.imread(test_img_path)

print(img.shape)

lab = model.segment(img, **win.segment2D_kwargs)

print(lab.shape)

if model_name == 'YeastMate':
    cca_df = model.predictCcaState(img)
    print(cca_df)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[1].imshow(lab)
plt.show()
