import os

import numpy as np

import skimage

from cellacdc._run import _setup_app

from cellacdc import VolumeRendererWindow
from cellacdc import data_path

images_path = os.path.join(
    data_path, 
    'test_snapshots', 
    'mtDNA_Anika', 
    'Position_17',
    'Images', 
)

mneon_filepath = os.path.join(
    images_path, 'ASY15-1_0nM-17_s17_mNeon.tif'
)

lab_filepath = os.path.join(
    images_path, 'ASY15-1_0nM-17_s17_segm.npz'
)

lab = np.load(lab_filepath)['arr_0']

volume = skimage.io.imread(mneon_filepath)
data = {'mNeon': volume}

renderer = VolumeRendererWindow()
renderer.set_volumes(data)   # (Z, Y, X) numpy array
renderer.set_labels(lab) 
renderer.run()