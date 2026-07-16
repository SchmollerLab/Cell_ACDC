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
mscarlet_filepath = os.path.join(
    images_path, 'FPY015-2_SCD-03_s03_run_num1_mScarlet_preprocessed.tif'
)

kaedegreen_volume = skimage.io.imread(kaedegreen_filepath)
mscarlet_volume = skimage.io.imread(mscarlet_filepath)

data = {
    'mScarlet-I3': mscarlet_volume,
    'KaedeGreen': kaedegreen_volume
}

renderer = VolumeRendererWindow()
renderer.set_volumes(data) 
renderer.run()