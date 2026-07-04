import os

import numpy as np
import pandas as pd

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

lab_filepath = os.path.join(
    images_path, 'FPY015-2_SCD-03_s03_segm.npz'
)

lab = np.load(lab_filepath)['arr_0']

metadata_filepath = os.path.join(
    images_path, 'FPY015-2_SCD-03_s03_metadata.csv'
)

df_metadata = pd.read_csv(metadata_filepath, index_col='Description')
voxel_size = (
    float(df_metadata.at['PhysicalSizeZ', 'values']),
    float(df_metadata.at['PhysicalSizeY', 'values']),
    float(df_metadata.at['PhysicalSizeX', 'values'])
)

volume = skimage.io.imread(kaedegreen_filepath)
data = {'KaedeGreen': volume}

renderer = VolumeRendererWindow()
renderer.set_volumes(
    data,
    voxel_size=voxel_size
)
renderer.set_labels(lab) 
renderer.run()