import os

from cellacdc import myutils

from cellacdc.models.cellpose_v2 import acdcSegment as acdc_cp2

class AvailableModels:
    major_version = myutils.get_cellpose_major_version()
    if major_version == 3:
        from ..cellpose_v3 import CELLPOSE_V3_MODELS
        values = CELLPOSE_V3_MODELS
    else:
        from . import CELLPOSE_V2_MODELS
        values = CELLPOSE_V2_MODELS

class Model:
    def __init__(
            self, 
            model_type: AvailableModels='cyto3', 
            net_avg=False, 
            gpu=False,
            device='None'
        ):
        self.acdcCellpose = acdc_cp2.Model(model_type, net_avg=net_avg, gpu=gpu)
        
    def segment(
            self, image,
            diameter=0.0,
            flow_threshold=0.4,
            cellprob_threshold=0.0,
            stitch_threshold=0.0,
            min_size=15,
            anisotropy=0.0,
            normalize=True,
            resample=True,
            segment_3D_volume=False            
        ):
        labels = self.acdcCellpose.segment(
            image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            stitch_threshold=stitch_threshold,
            min_size=min_size,
            anisotropy=anisotropy,
            normalize=normalize,
            resample=resample,
            segment_3D_volume=segment_3D_volume  
        )
        return labels
    
    def setupLogger(self, logger):
        self.acdcCellpose.setupLogger(logger)
    
    def closeLogger(self):
        self.acdcCellpose.closeLogger()
    
    def to_rgb_stack(self, first_ch_data, second_ch_data):
        return self.acdcCellpose.to_rgb_stack(first_ch_data, second_ch_data)

def url_help():
    return 'https://cellpose.readthedocs.io/en/latest/api.html'