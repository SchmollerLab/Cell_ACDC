import os

from cellacdc import printl

from cellacdc.myutils import get_cellpose_major_version
cp_version = get_cellpose_major_version(errors='ignore')
if cp_version is None or cp_version == 2:
    from cellacdc.models.cellpose_v2 import acdcSegment as cp
    CP_VERSION = 2
else:
    from cellacdc.models.cellpose_v3 import acdcSegment as cp
    CP_VERSION = 3
    
from cellpose import models

class Model:
    def __init__(self, model_path: os.PathLike = '', net_avg=False, gpu=False,
                 directml_gpu=False):
        self.acdcCellpose = cp.Model()
        if directml_gpu:
            from cellacdc.models.cellpose_v2._directML import init_directML
            directml_gpu = init_directML()

        if directml_gpu and gpu:
            printl(
                """
                gpu is preferable to directml_gpu, but doesn't work with non NVIDIA GPUs.
                Since directml_gpu and set to True, the gpu argument will be ignored.
                """
            )
            gpu = False

        try:
            self.acdcCellpose.model = models.CellposeModel(
                gpu=gpu, net_avg=net_avg, pretrained_model=model_path
            )
        except Exception as err:
            self.acdcCellpose.model = models.CellposeModel(
                gpu=gpu, pretrained_model=model_path,
            )
            self.acdcCellpose.acdcCellpose.model = self.acdcCellpose.model

        if directml_gpu:
            from cellacdc.models.cellpose_v2._directML import setup_directML
            from cellacdc.core import fix_sparse_directML
            try:
                setup_directML(self.acdcCellpose.acdcCellpose)
            except Exception as err:
                setup_directML(self.acdcCellpose)

            fix_sparse_directML()
            
        if gpu:# there is a problem in cellpose with gpu, doesnt work properly yet...
            import torch
            device = torch.device('cuda:0')
            from cellacdc.models.cellpose_v2._directML import setup_custom_device
            try:
                setup_custom_device(self.acdcCellpose.acdcCellpose, device)
            except Exception as err:
                setup_custom_device(self.acdcCellpose, device)
        
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
    
    def segment3DT(self, video_data, signals=None, **kwargs):
        labels = self.acdcCellpose.segment3DT(
            video_data, signals=signals, **kwargs
        )
        return labels
    
    def setupLogger(self, logger):
        self.acdcCellpose.setupLogger(logger)
    
    def setLoggerPropagation(self, propagate:bool):
        self.acdcCellpose.setLoggerPropagation(propagate)
    
    def setLoggerLevel(self, level:str):
         self.acdcCellpose.setLoggerLevel(level)
    
    def closeLogger(self):
        self.acdcCellpose.closeLogger()
    
    def second_ch_img_to_stack(self, first_ch_data, second_ch_data):
        return self.acdcCellpose.second_ch_img_to_stack(
            first_ch_data, second_ch_data
        )

def url_help():
    return 'https://cellpose.readthedocs.io/en/latest/api.html'