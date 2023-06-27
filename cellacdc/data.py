import os

import numpy as np
from pyparsing import col

import skimage.io

from . import data_path, load, base_cca_df

class _Data:
    def __init__(
            self, images_path, intensity_image_path, acdc_df_path, segm_path
        ):
        self.images_path = images_path
        self.intensity_image_path = intensity_image_path
        self.acdc_df_path = acdc_df_path
        self.segm_path = segm_path
    
    def acdc_df(self):
        return load._load_acdc_df_file(self.acdc_df_path)
    
    def image_data(self):
        return skimage.io.imread(self.intensity_image_path)
    
    def segm_data(self):
        return np.load(self.segm_path)['arr_0']
    
    def cca_df(self):
        acdc_df = load._load_acdc_df_file(self.acdc_df_path).dropna()
        cca_df = acdc_df[list(base_cca_df.keys())]
        dtypes = {col: type(value) for col, value in base_cca_df.items()}
        cca_df = cca_df.astype(dtypes)
        return cca_df

class YeastTimeLapseAnnotated(_Data):
    def __init__(self):
        images_path = os.path.join(
            data_path, 'test_timelapse', 'Yagya_Kurt_presentation', 
            'Position_6', 'Images'
        )
        intensity_image_path = os.path.join(
            images_path, 'SCGE_5strains_23092021_Dia_Ph3.tif'
        )
        acdc_df_path = os.path.join(
            images_path, 'SCGE_5strains_23092021_acdc_output.csv'
        )
        segm_path = os.path.join(
            images_path, 'SCGE_5strains_23092021_segm.npz'
        )
        super().__init__(
            images_path, intensity_image_path, acdc_df_path, segm_path
        )

class Cdc42TimeLapseData(_Data):
    def __init__(self):
        images_path = os.path.join(
            data_path, 'test_timelapse', 'Kurt_ring', 'Cdc42',
            'Position_1', 'Images'
        )
        intensity_image_path = os.path.join(
            images_path, 'SCGE_DLY16570_1-15_DLY16571_16-30_corr_s01_Dia_Ph3.tif'
        )
        acdc_df_path = os.path.join(
            images_path, 'SCGE_DLY16570_1-15_DLY16571_16-30_corr_s01_acdc_output.csv'
        )
        segm_path = os.path.join(
            images_path, 'SCGE_DLY16570_1-15_DLY16571_16-30_corr_s01_segm.npz'
        )
        super().__init__(
            images_path, intensity_image_path, acdc_df_path, segm_path
        )
    
    def cdc42_data(self):
        return skimage.io.imread(os.path.join(
            self.images_path, 
            'SCGE_DLY16570_1-15_DLY16571_16-30_corr_s01_tdTomato_Ph3__YEAST.tif'
        ))