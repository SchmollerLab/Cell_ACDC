import os
import sys
import numpy as np
import skimage.measure
from cellacdc import core, myutils, widgets, load, html_utils
from cellacdc import data, data_path
from cellacdc import transformation
from cellacdc.plot import imshow

try:
    import pytest
    pytest.skip('skipping this test since it is gui based', allow_module_level=True)
except Exception as e:
    pass

import qtpy.compat
from cellacdc._run import _setup_app

# Ask which model to use --> Test if new model is visible
app, splashScreen = _setup_app(splashscreen=True)  
splashScreen.close()

path = (
    os.path.join(data_path, r'test_symm_div_acdc_tracker\Images\bknapp_Movie_S1.tif')
)

channel_name = 'bknapp_Movie_S1'
end_filename_segm = 'segm'# 'segm_test'
START_FRAME = 0 
STOP_FRAME = 499
# PLOT_FRAME = 499
SAVE = True
REAL_TIME_TRACKER = False
SCRUMBLE_IDs = False

# test_data = data.BABYtestData()
# posData = test_data.posData()

# test_data = data.FissionYeastAnnotated()

test_data = None

if test_data is None:
    tif_filepath, _ = qtpy.compat.getopenfilename(
        basedir=myutils.getMostRecentPath(),
        filters=('Images (*.tif)')
    )
    if not tif_filepath:
        exit('Execution cancelled.')
    
    images_path = os.path.dirname(tif_filepath)
    basename = os.path.commonprefix(myutils.listdir(images_path))
    filename, ext = os.path.splitext(os.path.basename(tif_filepath))
    channel = filename[len(basename):]
    posData = load.loadData(tif_filepath, channel)
else:
    posData = test_data.posData()
    posData.acdc_output_csv_path = test_data.acdc_df_path

posData.loadImgData()
posData.loadOtherFiles(
    load_segm_data=True, 
    load_metadata=True,
    # end_filename_segm=end_filename_segm
)

lab_stack = posData.segm_data[START_FRAME:STOP_FRAME+1]

imshow(lab_stack, axis_titles=['Before tracking'], annotate_labels_idxs=[0])

trackers = myutils.get_list_of_trackers()
txt = html_utils.paragraph('''
    <b>Select the tracker</b> to use<br><br>
''')
win = widgets.QDialogListbox(
    'Select tracker', txt, trackers, multiSelection=False, parent=None
)
win.exec_()

if win.cancel:
    sys.exit('Execution aborted')

trackerName = win.selectedItemsText[0]

# Load tracker
tracker, track_params = myutils.init_tracker(
    posData, trackerName, qparent=None, realTime=REAL_TIME_TRACKER
)
if track_params is None:
    exit('Execution aborted')    
print(posData.segm_data.shape)
lab_stack = posData.segm_data[START_FRAME:STOP_FRAME+1]

if SCRUMBLE_IDs:
    # Scrumble IDs last frame
    
    last_lab = lab_stack[-1]
    last_rp = skimage.measure.regionprops(lab_stack[-1])
    IDs = [obj.label for obj in last_rp]
    randomIDs = np.random.choice(IDs, size=len(last_rp), replace=False)
    for obj, randomID in zip(last_rp, randomIDs):
        last_lab[obj.slice][obj.image] = randomID

    # Randomly delete some objects last frame
    num_obj_to_del = 4
    idxs = np.arange(len(last_rp))
    random_idxs = np.random.choice(idxs, size=num_obj_to_del, replace=False)
    for random_idx in random_idxs:
        obj_to_del = last_rp[random_idx]
        last_lab[obj_to_del.slice][obj_to_del.image] = 0

print(f'Tracking data with shape {lab_stack.shape}')

trackerInputImage = None
if 'image' in track_params:
    trackerInputImage = track_params.pop('image')[START_FRAME:STOP_FRAME+1]

if 'image_channel_name' in track_params:
    # Store the channel name for the tracker for loading it 
    # in case of multiple pos
    track_params.pop('image_channel_name')

tracked_stack = core.tracker_track(
    lab_stack, tracker, track_params, 
    intensity_img=trackerInputImage,
    logger_func=print
)

posData.fromTrackerToAcdcDf(tracker, tracked_stack, save=True)

first_untracked_lab = lab_stack[0]
uniqueID = max(np.max(lab_stack), np.max(tracked_stack)) + 1

retracked_video = transformation.retrack_based_on_untracked_first_frame(
    tracked_stack.copy(), first_untracked_lab, uniqueID=uniqueID        
)

if SAVE:
    try:
        np.savez_compressed(
            posData.segm_npz_path.replace('segm', 'segm_tracked'), 
            tracked_stack
        )
    except Exception as e:
        import pdb; pdb.set_trace()


imshow(
    posData.loadChannelData(''),
    tracked_stack,
    retracked_video,
    lab_stack,
    axis_titles=[
        'Intensity channel', 
        'After tracking',
        'After re-tracking first frame', 
        'Before tracking'
    ],
    annotate_labels_idxs=[1, 2, 3]
)
