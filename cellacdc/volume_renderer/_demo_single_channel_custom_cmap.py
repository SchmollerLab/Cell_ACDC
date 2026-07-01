import os

import skimage

from cellacdc._run import _setup_app

from cellacdc import VolumeRendererWindow
from cellacdc import data_path

zstack_filepath = os.path.join(
    data_path, 'test_3Dsegm', 'Arohi', 'Position_1', 'Images', 
    'CMJ030_1.100 H3__fl-01_s1_Ch1_IF_H3_405_T3.tif'
)

zstack_array = skimage.io.imread(zstack_filepath)

renderer = VolumeRendererWindow()
renderer.set_volume(zstack_array, cmap=['black', 'cyan'])   # (Z, Y, X) numpy array
renderer.run()