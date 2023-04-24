import os
import traceback
import sys

import matplotlib.pyplot as plt

from collections import namedtuple
from importlib import import_module

from cellacdc import apps, myutils, widgets, load, html_utils

from PyQt5.QtWidgets import QApplication, QStyleFactory

try:
    import pytest
    pytest.skip('skipping this test since it is gui based', allow_module_level=True)
except Exception as e:
    pass

test_img_path = (
    # r"G:\My Drive\01_Postdoc_HMGU\Python_MyScripts\MIA\Git\DeepSea\data\test_images\A11_z007_c001.png"
    # r"G:\My Drive\01_Postdoc_HMGU\Python_MyScripts\MIA\Git\DeepSea\data\test_images\train_A11_z001_c001.png"
    # r"G:\My Drive\01_Postdoc_HMGU\Python_MyScripts\MIA\Git\Cell_ACDC\data\test_images\test_cellpose.tif"
    # r"G:\My Drive\01_Postdoc_HMGU\Python_MyScripts\MIA\Git\Cell_ACDC\data\test_images\test_YeaZ.tif"
    r"G:\My Drive\01_Postdoc_HMGU\Python_MyScripts\MIA\Git\Cell_ACDC\data\test_timelapse\Yagya_Kurt_presentation\Position_6\Images\SCGE_5strains_23092021_Dia_Ph3.tif"
)

channel_name = 'Dia_Ph3'
START_FRAME = 200
STOP_FRAME = 201

posData = load.loadData(
    test_img_path, channel_name
)
posData.loadImgData()
posData.loadOtherFiles(
    load_segm_data=True, 
    load_metadata=True
)

test_segm_path = (
    r"G:\My Drive\01_Postdoc_HMGU\Python_MyScripts\MIA\Git\Cell_ACDC\data\test_timelapse\Yagya_Kurt_presentation\Position_6\Images\SCGE_5strains_23092021_segm.npz"
)

# Ask which model to use --> Test if new model is visible
app = QApplication(sys.argv)
app.setStyle(QStyleFactory.create('Fusion'))

trackers = myutils.get_list_of_trackers()
txt = html_utils.paragraph('''
    Do you want to track the objects?<br><br>
    If yes, <b>select the tracker</b> to use<br><br>
''')
win = widgets.QDialogListbox(
    'Track objects?', txt, trackers, multiSelection=False, parent=None
)
win.exec_()

if win.cancel:
    sys.exit('Execution aborted')

trackerName = win.selectedItemsText[0]

# Load tracker
tracker, track_params = myutils.import_tracker(
    posData, trackerName, qparent=None
)

lab_stack = posData.segm_data[START_FRAME:STOP_FRAME+1]

print(f'Tracking data with shape {lab_stack.shape}')

if 'image' in track_params:
    trackerInputImage = track_params.pop('image')[START_FRAME:STOP_FRAME+1]
    try:
        tracked_stack = tracker.track(
            lab_stack, trackerInputImage, **track_params
        )
    except Exception as e:
        traceback.print_exc()
        tracked_stack = tracker.track(lab_stack, **track_params)
else:
    tracked_stack = tracker.track(lab_stack, **track_params)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(tracked_stack[-1])
ax[1].imshow(tracked_stack[-2])
plt.show()
