from deepsea.model import DeepSeaTracker
from deepsea.utils import track_cells
from cellacdc.models.DeepSea import _init_model

class tracker:
    def __init__(self, gpu=False):
        _transforms, torch_device, checkpoint, model = _init_model(
            'tracking.pth', DeepSeaTracker, [128,128], 
            [0.5], [0.5]
        )
        self.torch_device = torch_device
        self._transforms = _transforms
        self._checkpoint = checkpoint
        self.model = model
    
    def track(self, segm_video, image, signals=None):
        for img in image:
            pass