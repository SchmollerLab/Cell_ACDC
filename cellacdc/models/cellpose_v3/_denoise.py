import numpy as np
from tqdm import tqdm

from cellacdc import printl

from cellpose.denoise import DenoiseModel

from ..cellpose_v2.acdcSegment import _initialize_image

class DenoiseModelTypes:
    values = ['one-click', 'nuclei']

class DenoiseModes:
    values = ['denoise', 'deblur']

class CellposeDenoiseModel(DenoiseModel):
    def __init__(
            self, 
            gpu=False, 
            denoise_model_type: DenoiseModelTypes='one-click', 
            denoise_mode: DenoiseModes='denoise',
        ):
        """Initialize cellpose 3.0 denoising model

        Parameters
        ----------
        gpu : bool, optional
            If True and PyTorch for your GPU is correctly installed, 
            denoising will run on the GPU. Default is False
        denoise_model_type : str, optional
            Either 'one-click' or 'nuclei'. Default is 'one-click'
        denoise_mode : str, optional
            Either 'denoise' or 'deblur'. Default is 'denoise'
        """        
        self.nstr = "cyto3" if denoise_model_type=="one-click" else "nuclei"
        model_name = f'{denoise_mode}_{self.nstr}'
        super().__init__(gpu=gpu, model_type=model_name)
        
        self._denoise_mode = denoise_mode
        self._denoise_model_type = denoise_model_type
    
    def _get_normalize_params(
            self, 
            image,
            normalize=False, 
            rescale_intensity_low_val_perc=0.0, 
            rescale_intensity_high_val_perc=100.0, 
            sharpen=0,
            low_percentile=1.0, 
            high_percentile=99.0,
            title_norm=0,
            norm3D=False
        ):
        if not normalize:
            return False
        
        normalize_kwargs = {}
        do_rescale = (
            rescale_intensity_low_val_perc != 0
            or rescale_intensity_high_val_perc != 100.0
        )
        if not do_rescale:
            normalize_kwargs['lowhigh'] = None
        else:
            low = image*rescale_intensity_low_val_perc/100
            high = image*rescale_intensity_high_val_perc/100
            normalize_kwargs['lowhigh'] = (low, high)
        
        normalize_kwargs['sharpen'] = sharpen
        normalize_kwargs['percentile'] = (low_percentile, high_percentile)
        normalize_kwargs['title_norm'] = title_norm
        normalize_kwargs['norm3D'] = norm3D
        
        return normalize_kwargs
    
    def run(
            self,
            image: np.ndarray,
            diameter=0.0,
            normalize=False, 
            rescale_intensity_low_val_perc=0.0, 
            rescale_intensity_high_val_perc=100.0, 
            sharpen=0,
            low_percentile=1.0, 
            high_percentile=99.0,
            title_norm=0,
            norm3D=False
        ):
        """Run cellpose 3.0 denoise model

        Parameters
        ----------
        image : (Y, X) or (Z, Y, X) numpy.ndarray
            2D or 3D image (z-stack). 
        diameter : float, optional
            Diameter of expected objects. If 0.0, it uses 30.0 for "one-click" 
            and 17.0 for "nuclei". Default is 0.0
        normalize : bool, optional
            If True, normalize image using the other parameters.  Default is False
        rescale_intensity_low_val_perc : float, optional
            Rescale intensities so that this is the minimum value in the image. 
            Default is 0.0
        rescale_intensity_high_val_perc : float, optional
            Rescale intensities so that this is the maximum value in the image. 
            Default is 100.0
        sharpen : int, optional
            Sharpen image with high pass filter, recommended to be 1/4-1/8 
            diameter of cells in pixels. Default is 0.
        low_percentile : float, optional
            Lower percentile for normalizing image. Default is 1.0
        high_percentile : float, optional
            Higher percentile for normalizing image. Default is 99.0
        title_norm : int, optional
            Compute normalization in tiles across image to brighten dark areas. 
            To turn it on set to window size in pixels (e.g. 100). Default is 0
        norm3D : bool, optional
            Compute normalization across entire z-stack rather than 
            plane-by-plane in stitching mode. Default is False
        """ 
        isRGB = image.shape[-1] == 3 or image.shape[-1] == 4
        channels = [0,0] if not isRGB else [1,2]
        
        isZstack = False
        if not isRGB and image.ndim == 3:
            isZstack = True
        
        if diameter == 0:
            diameter = 30.0 if self.nstr == 'cyto3' else 17.0

        input_image = _initialize_image(image)
        normalize_params = self._get_normalize_params(
            input_image,
            normalize=False, 
            rescale_intensity_low_val_perc=0.0, 
            rescale_intensity_high_val_perc=100.0, 
            sharpen=0,
            low_percentile=1.0, 
            high_percentile=99.0,
            title_norm=0,
            norm3D=False
        )
        eval_kwargs = {
            'channels': channels, 
            'diameter': diameter, 
            'normalize': normalize
        }
        if isZstack:
            denoised_img = np.zeros(image.shape, dtype=np.float32)
            pbar = tqdm(
                total=len(denoised_img), ncols=100, desc='Denoising z-slice: '
            )
            for z, img in enumerate(input_image):
                denoised_z_data = self.eval(img, **eval_kwargs)
                denoised_img[z] = denoised_z_data[...,0]
                pbar.update()
            pbar.close()
        else:
            denoised_data = self.eval(input_image, **eval_kwargs)
            denoised_img = denoised_data[...,0]
        return denoised_img
        
def url_help():
    return 'https://www.biorxiv.org/content/10.1101/2024.02.10.579780v1'