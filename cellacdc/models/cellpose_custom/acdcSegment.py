import os

from cellacdc import printl

from cellacdc.myutils import get_cellpose_major_version
cp_version = get_cellpose_major_version(errors='ignore')
if cp_version is None or cp_version == 2:
    from cellacdc.models.cellpose_v2 import acdcSegment as cp
else:
    from cellacdc.models.cellpose_v3 import acdcSegment as cp
    
from cellpose import models

class Model:
    def __init__(self, model_path: os.PathLike = '', net_avg=False, gpu=False):
        self.acdcCellpose = cp.Model()
        try:
            self.acdcCellpose.model = models.CellposeModel(
                gpu=gpu, net_avg=net_avg, pretrained_model=model_path
            )
        except Exception as err:
            self.acdcCellpose.model = models.CellposeModel(
                gpu=gpu, pretrained_model=model_path
            )

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