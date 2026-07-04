import os

import numpy as np

import skimage

from cellacdc._run import _setup_app

from cellacdc import VolumeRendererWindow
from cellacdc import myutils

exp_folderpath = myutils.download_3d_renderer_demo_data()

images_path = os.path.join(
    exp_folderpath, 
    'Position_3',
    'Images', 
)

lab_filepath = os.path.join(
    images_path, 'FPY015-2_SCD-03_s03_segm.npz'
)

lab = np.load(lab_filepath)['arr_0']

renderer = VolumeRendererWindow()
renderer.set_labels(lab, SizeZ=10) 
renderer.run()