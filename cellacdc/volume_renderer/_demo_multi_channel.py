import os

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

mkate_filepath = os.path.join(
    images_path, 'ASY15-1_0nM-17_s17_mKate.tif'
)
mneon_filepath = os.path.join(
    images_path, 'ASY15-1_0nM-17_s17_mNeon.tif'
)

mkate_volume = skimage.io.imread(mkate_filepath)
mneon_volume = skimage.io.imread(mneon_filepath)

data = {
    'mKate': mkate_volume,
    'mNeon': mneon_volume
}

renderer = VolumeRendererWindow()
renderer.set_volumes(data) 
renderer.run()