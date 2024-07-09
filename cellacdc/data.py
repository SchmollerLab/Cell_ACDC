import os

import numpy as np

import tifffile

from . import data_path, load, base_cca_dict, cca_df_colnames

class _Data:
    def __init__(
            self, images_path, intensity_image_path, acdc_df_path, segm_path,
            basename
        ):
        self.images_path = images_path
        self.intensity_image_path = intensity_image_path
        self.acdc_df_path = acdc_df_path
        self.segm_path = segm_path
        self.basename = basename
    
    def filename(self):
        return os.path.basename(self.intensity_image_path)
    
    def channel_name(self):
        filename, ext = os.path.splitext(self.filename())
        return filename[len(self.basename):]
    
    def acdc_df(self):
        return load._load_acdc_df_file(self.acdc_df_path)
    
    def image_data(self):
        return load.load_image_file(self.intensity_image_path)
    
    def segm_data(self):
        return np.load(self.segm_path)['arr_0']
    
    def cca_df(self):
        acdc_df = load._load_acdc_df_file(self.acdc_df_path).dropna()
        cca_df = acdc_df[cca_df_colnames]
        dtypes = {col: type(value) for col, value in base_cca_dict.items()}
        cca_df = cca_df.astype(dtypes)
        return cca_df

class FissionYeastAnnotated(_Data):
    def __init__(self):
        images_path = os.path.join(
            data_path, 'test_symm_div_acdc_tracker', 'Images', 
        )
        intensity_image_path = os.path.join(
            images_path, 'bknapp_Movie_S1.tif'
        )
        acdc_df_path = os.path.join(
            images_path, 'bknapp_Movie_S1_acdc_output.csv'
        )
        segm_path = os.path.join(
            images_path, 'bknapp_Movie_S1_segm.npz'
        )
        basename = 'bknapp_Movie_S1_'
        super().__init__(
            images_path, intensity_image_path, acdc_df_path, segm_path,
            basename
        )
    
    def posData(self):
        from . import load
        return load.loadData(self.intensity_image_path, '')
    
class DeepSeaAnnotation(_Data):
    def __init__(self):
        images_path = os.path.join(
            data_path, 'deep_sea', 'Images', 
        )
        intensity_image_path = os.path.join(
            images_path, 'set_22_MESC.tif'
        )
        acdc_df_path = os.path.join(
            images_path, 'set_22_MESC_acdc_output.csv'
        )
        segm_path = os.path.join(
            images_path, 'set_22_MESC_segm.tif'
        )
        basename = 'set_22_MESC_'
        super().__init__(
            images_path, intensity_image_path, acdc_df_path, segm_path,
            basename
        )
    
    def posData(self):
        from . import load
        return load.loadData(self.intensity_image_path, '')

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
        basename = 'SCGE_5strains_23092021_'
        super().__init__(
            images_path, intensity_image_path, acdc_df_path, segm_path,
            basename
        )
    
    def posData(self):
        from . import load
        return load.loadData(self.intensity_image_path, 'Dia_Ph3')

class pomBseenDualChannelData(_Data):
    def __init__(self):
        images_path = os.path.join(
            data_path, 'test_pomBseen', 'dual_channel', 
            'Position_3', 'Images'
        )
        intensity_image_path = os.path.join(
            images_path, 'Demo_two_channel_input_image_BF.tif'
        )
        acdc_df_path = os.path.join(
            images_path, 'Demo_two_channel_input_image_acdc_output.csv'
        )
        segm_path = os.path.join(
            images_path, 'Demo_two_channel_input_image_segm_bf.npz'
        )
        basename = 'Demo_two_channel_input_image_'
        super().__init__(
            images_path, intensity_image_path, acdc_df_path, segm_path,
            basename
        )
    
    def posData(self):
        from . import load
        return load.loadData(self.intensity_image_path, 'BF')

class _YeastTimeLapseAnnotatedJordan(_Data):
    def __init__(self, custom_data_path=None):
        if custom_data_path is not None:
            _data_path = custom_data_path
        else:
            _data_path = data_path
        images_path = os.path.join(
            _data_path, 'gh_issue_394_Jordan', 'Position_1', 'Images'
        )
        intensity_image_path = os.path.join(
            images_path, '220630_JX_MS380_2ng-uL-aTc_pos04_g_s1_Phase.tif'
        )
        acdc_df_path = os.path.join(
            images_path, '220630_JX_MS380_2ng-uL-aTc_pos04_g_s1_acdc_output.csv'
        )
        segm_path = os.path.join(
            images_path, '220630_JX_MS380_2ng-uL-aTc_pos04_g_s1_segm.npz'
        )
        basename = '220630_JX_MS380_2ng-uL-aTc_pos04_g_s1_'
        super().__init__(
            images_path, intensity_image_path, acdc_df_path, segm_path,
            basename
        )
    
    def posData(self):
        from . import load
        return load.loadData(self.intensity_image_path, 'Dia_Ph3')

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
        basename = 'SCGE_DLY16570_1-15_DLY16571_16-30_corr_s01_'
        super().__init__(
            images_path, intensity_image_path, acdc_df_path, segm_path,
            basename
        )
    
    def cdc42_data(self):
        return load.imread(os.path.join(
            self.images_path, 
            'SCGE_DLY16570_1-15_DLY16571_16-30_corr_s01_tdTomato_Ph3__YEAST.tif'
        ))

class YeastMitoTimelapse(_Data):
    def __init__(self):
        images_path = os.path.join(
            data_path, 'test_4D', 'Lisa_mito', 'Position_5', 'Images'
        )
        intensity_image_path = os.path.join(
            images_path, 'Point0019_ChannelGFP,mCardinal,Ph-3_Seq0019_s5_Ph_3.tif'
        )
        acdc_df_path = os.path.join(
            images_path, 'Point0019_ChannelGFP,mCardinal,Ph-3_Seq0019_s5_acdc_output.csv'
        )
        segm_path = os.path.join(
            images_path, 'Point0019_ChannelGFP,mCardinal,Ph-3_Seq0019_s5_segm.npz'
        )
        basename = 'Point0019_ChannelGFP,mCardinal,Ph-3_Seq0019_s5_'
        super().__init__(
            images_path, intensity_image_path, acdc_df_path, segm_path,
            basename
        )
    
    def mito_segm(self):
        return np.load(os.path.join(
            self.images_path, 
            'Point0019_ChannelGFP,mCardinal,Ph-3_Seq0019_s5_GFP_segm_mask_otsu.npz'
        ))['arr_0']
    
    def cells_3D_segm(self):
        return np.load(os.path.join(
            self.images_path, 
            'Point0019_ChannelGFP,mCardinal,Ph-3_Seq0019_s5_segm_7slices.npz'
        ))['arr_0']

class BABYtestData(_Data):
    def __init__(self):
        images_path = os.path.join(
            data_path, 'test_BABY', 'evolve_testG_Brightfield', 'Position_1', 
            'Images'
        )
        intensity_image_path = os.path.join(
            images_path, 'evolve_testG_Brightfield.tif'
        )
        acdc_df_path = os.path.join(
            images_path, 'evolve_testG_acdc_output.csv'
        )
        segm_path = os.path.join(
            images_path, 'evolve_testG_segm.npz'
        )
        basename = 'evolve_testG_'
        super().__init__(
            images_path, intensity_image_path, acdc_df_path, segm_path,
            basename
        )
    
    def posData(self):
        from . import load
        return load.loadData(self.intensity_image_path, 'Brightfield')

class YeastMitoSnapshotData(_Data):
    def __init__(self):
        images_path = os.path.join(
            data_path, 'test_snapshots', 'mtDNA_Anika', 'Position_10', 
            'Images'
        )
        intensity_image_path = os.path.join(
            images_path, 'ASY15-1_0nM-10_s10_mNeon.tif'
        )
        acdc_df_path = os.path.join(
            images_path, 'ASY15-1_0nM-10_s10_acdc_output.csv'
        )
        segm_path = os.path.join(
            images_path, 'ASY15-1_0nM-10_s10_segm.npz'
        )
        basename = 'ASY15-1_0nM-10_s10_'
        super().__init__(
            images_path, intensity_image_path, acdc_df_path, segm_path,
            basename
        )
    
    def posData(self):
        from . import load
        return load.loadData(self.intensity_image_path, 'mNeon')