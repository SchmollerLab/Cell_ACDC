import numpy as np

from instanseg import InstanSeg

from ... import myutils, printl
from ..._types import SecondChannelImage

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

class ChannelOrder:
    values = (
        'First channel', 'Second channel'
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
            device = myutils.get_torch_device(gpu=True)
        elif device == 'CPU':
            device = 'cpu'
        elif device == 'GPU':
            device = myutils.get_torch_device(gpu=True)
        
        self.model = InstanSeg(
            model_type, 
            device=device,
            verbosity=verbosity
        )

    def preprocess(self, image, rescale_intensities):
        if rescale_intensities:
            image_min = image - image.min()
            image_float = image_min/image_min.max()
        else:
            image_float = myutils.img_to_float(image)
        
        return (image_float*255).astype(np.uint8)
    
    def segment(
            self,
            image, 
            second_channel_image: SecondChannelImage=None,
            return_masks_for_channel: ChannelOrder='First channel', 
            PhysicalSizeX: float=1.0,
            do_not_resize_to_pixel_size: bool=False,
            rescale_intensities: bool=False
        ):
        if do_not_resize_to_pixel_size:
            PhysicalSizeX = None
            
        image_in = image
        if second_channel_image is not None:
            image_in = self.second_ch_img_to_stack(image, second_channel_image)
        
        image_in = self.preprocess(image_in, rescale_intensities)
        
        if image_in.shape[-1] > 2:
            image_in = image_in[..., np.newaxis]
        
        is_zstack = image_in.ndim == 4
        
        if isinstance(return_masks_for_channel, int):
            masks_index = return_masks_for_channel
        else:
            masks_index = 0 if return_masks_for_channel == 'First channel' else 1
        
        if is_zstack:
            lab = np.zeros((image_in.shape[:3]), dtype=np.uint32)
            for z, img in enumerate(image_in):
                lab[z] = self._segment_2D_img(
                    img, PhysicalSizeX, masks_index=masks_index
                )
        else:
            lab = self._segment_2D_img(
                image_in, PhysicalSizeX, masks_index=masks_index
            )
        
        return lab
    
    def _segment_2D_img(self, image, PhysicalSizeX, masks_index=0):
        labeled_output, image_tensor = self.model.eval_small_image(
            image, PhysicalSizeX
        )
        labels = labeled_output[0].cpu().detach().numpy()
        lab = labels[masks_index].astype(np.uint32)
        return lab
    
    def second_ch_img_to_stack(self, image, second_image):
        img_stack = np.zeros((*image.shape, 2))
        img_stack[..., 0] = image
        img_stack[..., 1] = second_image
        return img_stack
        
        
def url_help():
    return 'https://github.com/instanseg/instanseg'