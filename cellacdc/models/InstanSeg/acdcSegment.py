import numpy as np

from instanseg import InstanSeg

from ... import myutils
from ...types import SecondChannelImage

from . import INSTANSEG_MODELS

class AvailabelModels:
    values = INSTANSEG_MODELS

class AvailableDevices:
    values = (
        'Auto', 'GPU', 'CPU'
    )

class VerbosityValues:
    values = (
        'Silent', 'Normal', 'Verbose'
    )

class Model:
    def __init__(
            self, 
            model_type: AvailabelModels='fluorescence_nuclei_and_cells', 
            custom_model_type: str='',
            device: AvailableDevices='Auto',
            verbosity: VerbosityValues='1'
        ) -> None:
        if custom_model_type:
            model_type = custom_model_type
        
        if device == 'Auto':
            device = None
        elif device == 'CPU':
            device = 'cpu'
        elif device == 'GPU':
            myutils.get_torch_device(gpu=True)
        
        self.model = InstanSeg(
            model_type, 
            device=device,
            verbosity=verbosity
        )

    def segment(
            self,
            image, 
            second_channel_image: SecondChannelImage=None,
            PhysicalSizeX: float=1.0,
        ):
        image_in = image
        if second_channel_image is not None:
            image_in = self.second_ch_img_to_stack(image, second_channel_image)
        
        if image_in.shape[-1] > 2:
            image_in = image_in[..., np.newaxis]
        
        is_zstack = image_in.ndim == 4
        
        if is_zstack:
            lab = np.zeros((image_in.shape[:3]), dtype=np.uint32)
            for z, img in enumerate(image_in):
                lab[z] = self._segment_2D_img(img, PhysicalSizeX)
        else:
            lab = self._segment_2D_img(image_in, PhysicalSizeX)
        
        return lab
    
    def _segment_2D_img(self, image, PhysicalSizeX):
        labeled_output, image_tensor = self.model.eval_small_image(
            image, PhysicalSizeX
        )
        lab = labeled_output[0].cpu().detach().numpy()[0]
        return lab
    
    def second_ch_img_to_stack(self, image, second_image):
        img_stack = np.zeros((*image.shape, 2))
        img_stack[..., 0] = image
        img_stack[..., 1] = second_image
        return img_stack
        
        
def url_help():
    return 'https://github.com/instanseg/instanseg'