import os

from cellacdc import myutils, printl
from cellacdc.models._cellpose_base.acdcSegment import Model as CellposeBaseModel, BackboneOptions
import torch
from . import AvailableModelsv3
from cellacdc.models._cellpose_base.acdcSegment import _initialize_image

from cellacdc._types import NotGUIParam

class DenoiseModelTypes:
    values = ['one-click', 'nuclei']

class DenoiseModes:
    values = ['denoise', 'deblur']

class Model(CellposeBaseModel):
    def __new__(cls, *args, **kwargs):
        myutils.check_install_cellpose(3)
        return super().__new__(cls)

    def __init__(
            self, 
            model_type:AvailableModelsv3='cyto3',
            model_path: os.PathLike='',
            gpu: bool=False,
            directml_gpu: bool=False,
            device:torch.device|int='None',
            denoise_before_segmentation:bool=False,
            denoise_model_type: DenoiseModelTypes='one-click', 
            denoise_mode: DenoiseModes='denoise',
            diameter_denoise:float=0.0,
            backbone: BackboneOptions='default',
        ):
        """Initialize cellpose 3 model

        Parameters
        ----------
        model_type : AvailableModels, optional
            Cellpose model type to use. Default is 'cyto3'. Mutually exclusive
            with `model_path`. If you want to use a custom model, set
            `model_path` to the path of the model file.
        model_path : os.PathLike, optional
            Path to a custom cellpose model file. If set, it will override
            `model_type`. If you want to use a custom model, set this to the
            path of the model file. Default is None.
        gpu : bool, optional
            If True and PyTorch for your GPU is correctly installed, 
            denoising and segmentation processes will run on the GPU. 
            Default is False
        directml_gpu : bool, optional
            If True, will attempt to use DirectML for GPU acceleration.
            Will be ignored if `gpu` is True. Default is False
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
        diameter_denoise : float, optional
            Mean diameter of objects in the image for denoising.
            If 0.0, it uses 30.0 for "one-click" and 17.0 for "nuclei".
            Default is 0.0, which will use the default values for the
            denoise model type.
        backbone : str, optional
            "default" is the standard res-unet, "transformer" for the segformer. 
        """

        self.initConstants()
        model_type, model_path, device = myutils.translateStrNone(model_type, model_path, device)
        
        self.check_model_path_model_type(
            model_type=model_type, 
            model_path=model_path, 
        )

        directml_gpu, gpu = self.check_directml_gpu_gpu(
            directml_gpu=directml_gpu, gpu=gpu,
        )
        
        print(f'Initializing Cellpose v3...')

        import cellpose
        if model_type:
            try:
                self.model = cellpose.models.Cellpose(
                    gpu=gpu,
                    device=device,
                    model_type=model_type,
                    backbone=backbone,
                )
                self._sizemodelnotfound = False                    

            except FileNotFoundError:
                printl(f'Size model for {model_type} not found.')
                self._sizemodelnotfound = True
                self.model = cellpose.models.CellposeModel(
                    gpu=gpu,
                    device=device,
                    model_type=model_type,
                    backbone=backbone,
                    )
        elif model_path is not None:
            self._sizemodelnotfound = True
            self.model = cellpose.models.CellposeModel(
                gpu=gpu,
                device=device,
                pretrained_model=model_path,
                backbone=backbone,
            )

        self.denoiseModel = None
        if denoise_before_segmentation:
            from cellacdc.models.cellpose_v3 import _denoise
            self.denoiseModel = _denoise.CellposeDenoiseModel(
                gpu=gpu, 
                denoise_model_type=denoise_model_type, 
                denoise_mode=denoise_mode,
                diam_mean=self.model.diam_mean,
            )

        
        self.setup_gpu_direct_ml(
            directml_gpu,
            gpu, device)

    def segment(
            self, 
            image,
            diameter:float=0.0,
            flow_threshold:float=0.4,
            cellprob_threshold:float=0.0,
            stitch_threshold:float=0.0,
            min_size:int=15,
            anisotropy:float=0.0,
            normalize:bool=True,
            resample:bool=True,
            segment_3D_volume:bool=False,
            denoise_normalize:bool=False,
            rescale_intensity_low_val_perc:float=0.0, 
            rescale_intensity_high_val_perc:float=100.0, 
            sharpen:int=0,
            low_percentile:float=1.0, 
            high_percentile:float=99.0,
            title_norm:int=0,
            norm3D:bool=False,
            init_imgs:NotGUIParam=True,
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
            _, isZstack = self.get_eval_kwargs_v2(image) # care everything else is gibberish except isZstack
            if init_imgs:
                if not segment_3D_volume and isZstack:
                    image, z_axis, channel_axis = _initialize_image(image, self.is_rgb, 
                                                iter_axis_zstack=0,
                                                isZstack=True,    
                                                )
                    self.channel_axis = channel_axis -1
                else:
                    image, z_axis, channel_axis = _initialize_image(image, self.is_rgb,
                                                isZstack=isZstack,
                                                )
                    self.z_axis = z_axis
                    self.channel_axis = channel_axis    
            

            input_image = self.denoiseModel.run(
                image,
                normalize=denoise_normalize, 
                rescale_intensity_low_val_perc=rescale_intensity_low_val_perc, 
                rescale_intensity_high_val_perc=rescale_intensity_high_val_perc, 
                sharpen=sharpen,
                low_percentile=low_percentile, 
                high_percentile=high_percentile,
                title_norm=title_norm,
                norm3D=norm3D,
                isZstack=isZstack,
            )
            
        self.img_shape = input_image.shape
        self.img_ndim = len(self.img_shape)

        eval_kwargs, isZstack = self.get_eval_kwargs_v2(
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

        init_imgs = init_imgs if self.denoiseModel is None else False
        labels = self.eval_loop(
            input_image,
            segment_3D_volume=segment_3D_volume,
            isZstack=isZstack,
            init_imgs=not self.denoiseModel,
            **eval_kwargs
        )

        self.img_shape = None
        self.img_ndim = None

        return labels
    
    def segment3DT(self, video_data, signals=None, init_imgs=True, **kwargs):
        eval_kwargs, isZstack = self.get_eval_kwargs_v2(video_data[0], **kwargs)

        input_video_data = video_data
        if self.denoiseModel is not None:
            if init_imgs:

                if not kwargs['segment_3D_volume'] and isZstack:
                    input_video_data, z_axis, channel_axis = _initialize_image(video_data, self.is_rgb,
                                    iter_axis_time=0,
                                    iter_axis_zstack=1,
                                    timelapse=True,
                                    isZstack=isZstack,
                                    )
                    self.z_axis = z_axis - 2 if z_axis is not None else None # video doesnt count as dim
                    self.channel_axis = channel_axis - 2

                else:
                    input_video_data, z_axis, channel_axis = _initialize_image(video_data, self.is_rgb,
                                                iter_axis_time=0,
                                                timelapse=True,
                                                isZstack=isZstack,
                                                )
                    self.z_axis = z_axis - 1 if z_axis is not None else None # video doesnt count as dim
                    self.channel_axis = channel_axis - 1
                    
            resc_int_low_val_perc = kwargs['rescale_intensity_low_val_perc']
            resc_int_high_val_perc = kwargs['rescale_intensity_high_val_perc']
            input_video_data = [
                self.denoiseModel.run(
                    image,
                    normalize=kwargs['denoise_normalize'], 
                    rescale_intensity_low_val_perc=resc_int_low_val_perc, 
                    rescale_intensity_high_val_perc=resc_int_high_val_perc, 
                    sharpen=kwargs['sharpen'],
                    low_percentile=kwargs['low_percentile'], 
                    high_percentile=kwargs['high_percentile'],
                    title_norm=kwargs['title_norm'],
                    norm3D=kwargs['norm3D'],
                    isZstack=isZstack,
                )
                for image in input_video_data
            ]
        
        self.img_shape = input_video_data[0].shape
        self.img_ndim = len(self.img_shape)

        eval_kwargs, isZstack = self.get_eval_kwargs_v2(input_video_data[0], **kwargs)

        init_imgs = init_imgs if self.denoiseModel is None else False
        labels = self.segment3DT_eval(
            input_video_data, isZstack, eval_kwargs, init_imgs=init_imgs is None, **kwargs
        )

        self.img_shape = None
        self.img_ndim = None
        return labels

def url_help():
    return 'https://cellpose.readthedocs.io/en/latest/api.html'
