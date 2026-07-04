import os

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

kaedegreen_filepath = os.path.join(
    images_path, 'FPY015-2_SCD-03_s03_run_num1_KaeGr_preprocessed.tif'
)

kaedegreen_volume = skimage.io.imread(kaedegreen_filepath)

renderer = VolumeRendererWindow()
renderer.set_volume(kaedegreen_volume, cmap=['black', 'green'])   # (Z, Y, X) numpy array
renderer.run()