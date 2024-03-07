import os

from cellacdc import myutils

from cellacdc.models.cellpose_v2 import acdcSegment as acdc_cp2
from . import _denoise

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
            gpu=False,
            device='None',
            denoise_before_segmentation=False,
            denoise_model_type: _denoise.DenoiseModelTypes='one-click', 
            denoise_mode: _denoise.DenoiseModes='denoise',
        ):
        """Initialize cellpose 3.0 denoising model

        Parameters
        ----------
        gpu : bool, optional
            If True and PyTorch for your GPU is correctly installed, 
            denoising and segmentation processes will run on the GPU. 
            Default is False
        device : torch.device or int or None
            If not None, this is the device used for running the model
            (torch.device('cuda') or torch.device('cpu')). 
            It overrides `gpu`, recommended if you want to use a specific GPU 
            (e.g. torch.device('cuda:1'). Default is None
        denoise_before_segmentation : bool, optional
            If True, run denoising before segmentation. Default is False
        denoise_model_type : str, optional
            Either 'one-click' or 'nuclei'. Default is 'one-click'
        denoise_mode : str, optional
            Either 'denoise' or 'deblur'. Default is 'denoise'
        """ 
        self.acdcCellpose = acdc_cp2.Model(
            model_type, gpu=gpu, device=device
        )
        self.denoiseModel = None
        if denoise_before_segmentation:
            self.denoiseModel = _denoise.CellposeDenoiseModel(
                gpu=gpu, 
                denoise_model_type=denoise_model_type, 
                denoise_mode=denoise_mode
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
            segment_3D_volume=False,
            denoise_normalize=False,
            rescale_intensity_low_val_perc=0.0, 
            rescale_intensity_high_val_perc=100.0, 
            sharpen=0,
            low_percentile=1.0, 
            high_percentile=99.0,
            title_norm=0,
            norm3D=False            
        ):
        """Run cellpose 3.0 denoising + segmentation model

        Parameters
        ----------
        image : (Y, X) or (Z, Y, X) numpy.ndarray
            2D or 3D image (z-stack). 
        diameter : float, optional
            Diameter of expected objects. If 0.0, it uses 30.0 for "one-click" 
            and 17.0 for "nuclei". Default is 0.0
        flow_threshold : float, optional
            Flow error threshold (all cells with errors below threshold are 
            kept) (not used for 3D). Default is 0.4
        cellprob_threshold : float, optional
            All pixels with value above threshold will be part of an object. 
            Decrease this value to find more and larger masks. Default is 0.0
        stitch_threshold : float, optional
            If `stitch_threshold` is greater than 0.0 and `segment_3D_volume` 
            is True, masks are stitched in 3D to return volume segmentation. 
            Default is 0.0
        min_size : int, optional
            Minimum number of pixels per mask, you can turn off this filter 
            with `min_size = -1`. Default is 15
        anisotropy : float, optional
            For 3D segmentation, optional rescaling factor (e.g. set to 2.0 if 
            Z is sampled half as dense as X or Y). Default is 0.0
        normalize : bool, optional
            If True, normalize image using the other parameters. 
            Default is True
        resample : bool, optional
            Run dynamics at original image size (will be slower but create 
            more accurate boundaries). Default is True
        segment_3D_volume : bool, optional
            If True and input `image` is a 3D z-stack the entire z-stack 
            is passed to cellpose model. If False, Cell-ACDC will force one 
            z-slice at the time. Best results with cellpose and 3D data are 
            obtained by passing the entire z-stack, but with a 
            `stitch_threshold` greater than 0 (e.g., 0.4). This way cellpose 
            will internally segment slice-by-slice and it will merge the 
            resulting z-slice masks belonging to the same object. 
            Default is False
        denoise_normalize : bool, optional
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
        
        input_image = image
        if self.denoiseModel is not None:
            input_image = self.denoiseModel.run(
                image,
                normalize=denoise_normalize, 
                rescale_intensity_low_val_perc=rescale_intensity_low_val_perc, 
                rescale_intensity_high_val_perc=rescale_intensity_high_val_perc, 
                sharpen=sharpen,
                low_percentile=low_percentile, 
                high_percentile=high_percentile,
                title_norm=title_norm,
                norm3D=norm3D
            )
        labels = self.acdcCellpose.segment(
            input_image,
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