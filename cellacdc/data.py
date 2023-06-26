import os

import numpy as np
from pyparsing import col

import skimage.io

from . import data_path, load, base_cca_df

class YeastTimeLapseAnnotated:
    images_path = os.path.join(
        data_path, 'test_timelapse', 'Yagya_Kurt_presentation', 
        'Position_6', 'Images'
    )
    phase_contrast_path = os.path.join(
        images_path, 'SCGE_5strains_23092021_Dia_Ph3.tif'
    )
    acdc_df_path = os.path.join(
        images_path, 'SCGE_5strains_23092021_acdc_output.csv'
    )
    segm_path = os.path.join(
        images_path, 'SCGE_5strains_23092021_segm.npz'
    )
    
    def acdc_df(self):
        return load._load_acdc_df_file(self.acdc_df_path)
    
    def image_data(self):
        return skimage.io.imread(self.phase_contrast_path)
    
    def segm_data(self):
        return np.load(self.segm_path)['arr_0']
    
    def cca_df(self):
        acdc_df = load._load_acdc_df_file(self.acdc_df_path).dropna()
        cca_df = acdc_df[list(base_cca_df.keys())]
        dtypes = {col: type(value) for col, value in base_cca_df.items()}
        cca_df = cca_df.astype(dtypes)
        return cca_df