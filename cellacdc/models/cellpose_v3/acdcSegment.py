import os

from cellacdc import myutils, printl
from cellacdc.models._cellpose_base.acdcSegment import Model as CellposeBaseModel
from cellacdc.models._cellpose_base.acdcSegment import (BackboneOptions, GPUDirectMLGPUCPU, cpu_gpu_directml_gpu,
    check_directml_gpu_gpu, setup_gpu_direct_ml, _get_normalize_params, DealWithSecondChannelOptions)

import torch
from . import AvailableModelsv3, AvailableModelsv3Denoise
from cellacdc.models._cellpose_base.acdcSegment import _initialize_image

from cellacdc._types import NotGUIParam

import numpy as np

class Model(CellposeBaseModel):
    def __new__(cls, *args, **kwargs):
        myutils.check_install_cellpose(3)
        return super().__new__(cls)

    def __init__(
            self, 
            model_type:AvailableModelsv3='cyto3',
            model_path: os.PathLike='',
            device_type: GPUDirectMLGPUCPU='cpu',
            device: torch.device | int | None = None,
            batch_size:int=8,
            denoise_before_segmentation:bool=False,
            denoise_model: AvailableModelsv3Denoise='denoise_cyto3',
            denoise_second_channel: DealWithSecondChannelOptions = 'together',
            denoise_model_path: os.PathLike='',
            denoise_diameter:float=0.0,
            denoise_nchan: int = 1,
            backbone: BackboneOptions='default',
            is_rgb: NotGUIParam = False,  # whether the input image will be rgb
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
        device_type : bool, optional
            Specifies the use of GPU for running the model. Options are:
            - 'cpu': Use CPU for running the model.
            - 'gpu': Use CUDA for running the model on GPU.
            - 'directml': Use DirectML for running the model on GPU.
        device : torch.device or int or None
            If not None, this is the device used for running the model
            (torch.device('cuda') or torch.device('cpu')). 
            It overrides `gpu`, recommended if you want to use a specific GPU 
            (e.g. torch.device('cuda:1'). Default is None
        batch_size : int, optional
            Batch size for running the model on GPU. Reduce to decrease memory usage, but it will slow down the processing.
            Default is 8.
        denoise_before_segmentation : bool, optional
            If True, run denoising before segmentation. Default is False
        denoise_model : AvailableModelsv3Denoise, optional
            Cellpose denoise model type to use.
            The structure is as follows:
            {type}_{ltype}_{cellpose_version}
            Options for `type` are:
            - denoise: Denoising model
            - deblur: Deblurring model
            - upsample: Upsampling model
            - one-click: One-click denoising model

            Options for `ltype` are:
            -per: perimeter or persistence
            -seg: Segmentation model
            -rec: Reconstruction model

            Options for `cellpose_version` are:
            -cyto3: Cellpose v3 model
            -cyto2: Cellpose v2 model
            -nuclei: Cellpose nuclei model (used when segmenting two channels and nchan2 is set to True for the second channel)
        denoise_second_channel : DealWithSecondChannelOptions, optional
            How to denoise the second channel. Can be denoised separately, or together (if model supports it), or not at all.
        denoise_model_path : os.PathLike, optional
            Path to a custom cellpose denoise model file.
        denoise_diameter : float, optional
            Mean diameter of objects in the image for denoising (during training?).
            If left at 0, will pass 30, the cellpose default value.
            It is not clear what exactly this parameter does from cellpose documentation, 
            please don't touch it if you dont know too!
        denoise_nchan : int, optional
            Number of channels in the denoised image. Default is 1.
            All cellpose denoise models are single channel, so almost always 1.
        backbone : str, optional
            "default" is the standard res-unet, "transformer" for the segformer.
        """
        self.initConstants(is_rgb=is_rgb)
        self.batch_size = batch_size

        out = myutils.translateStrNone(model_type, model_path, device,
                                       denoise_model, denoise_model_path, )
        model_type, model_path, device, denoise_model, denoise_model_path = out
        self.check_model_path_model_type(
            model_type=model_type, 
            model_path=model_path, 
        )

        directml_gpu, gpu =  cpu_gpu_directml_gpu(
            input_string=device_type,
        )

        directml_gpu, gpu, proceed = check_directml_gpu_gpu(
            directml_gpu=directml_gpu, gpu=gpu,
        )

        if not proceed:
            return

        if denoise_before_segmentation and denoise_model:
            denoise_model_type = denoise_model.split('_')[-1] if denoise_model else None
            if denoise_model_type != model_type:
                printl(f'[WARNING] denoise model type {denoise_model_type} does not match ')
        
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
                device_type=device_type,
                device=device, denoise_model=denoise_model,
                denoise_model_path=denoise_model_path,
                diam_mean=denoise_diameter,
                deal_with_second_channel=denoise_second_channel, 
                denoise_nchan=denoise_nchan,
                batch_size=batch_size, 
                is_rgb=self.is_rgb,
                ask_install_gpu=False,  # don't ask to install cellpose if not installed
            )
        
        setup_gpu_direct_ml(
            self,
            directml_gpu,
            gpu, device)

    def _get_eval_kawrgs_v3(
            self,
            eval_kwargs: dict,
            **kwargs: dict,
    ):
        eval_kwargs_3 = {
            'cellprob_threshold': kwargs['cellprob_threshold'],
            'min_size': kwargs['min_size'],
            'resample': kwargs['resample'],
            'max_size_fraction': kwargs['max_size_fraction'],
            'flow3D_smooth': kwargs['flow3D_smooth'],
            'tile_overlap': kwargs['tile_overlap'],
            'invert': kwargs['invert'],

        }

        eval_kwargs.update(eval_kwargs_3)

        return eval_kwargs

    def segment( # 2D, 2D x stacks. 2D over time is in segment3DT, 4D is not supported
            self, 
            image,
            diameter:float=0.0,
            flow_threshold:float=0.4,
            cellprob_threshold:float=0.0,
            resample:bool=True,
            min_size:int=15,
            max_size_fraction:float=0.4,
            segment_3D_volume:bool=False,
            stitch_threshold:float=0.0,
            flow3D_smooth:float=0,
            anisotropy:float=0.0,
            tile_overlap:float=0.1,
            invert:bool=False,
            normalize:bool=True,
            rescale_intensity_low_val_perc:float=0.0, 
            rescale_intensity_high_val_perc:float=100.0, 
            # sharpen:int=0,
            low_percentile:float=1.0, 
            high_percentile:float=99.0,
            norm3D:bool=False,
            tile_norm_blocksize: int=0,
            denoise_rescale:float=1.0,
            init_imgs:NotGUIParam=True,
            bsize:int=224,
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
        resample : bool, optional
            Run dynamics at original image size (will be slower but create 
            more accurate boundaries). Default is True
        min_size : int, optional
            Minimum number of pixels per mask, you can turn off this filter 
            with `min_size = -1`. Default is 15
        max_size_fraction : float, optional
            Masks larger than this fraction of total image size are removed. Default is 0.4.
        segment_3D_volume : bool, optional
            If True and input `image` is a 3D z-stack the entire z-stack 
            is passed to cellpose model. If False, Cell-ACDC will force one 
            z-slice at the time. Best results with cellpose and 3D data are 
            obtained by passing the entire z-stack, but with a 
            `stitch_threshold` greater than 0 (e.g., 0.4). This way cellpose 
            will internally segment slice-by-slice and it will merge the 
            resulting z-slice masks belonging to the same object. 
            Default is False
        stitch_threshold : float, optional
            If `stitch_threshold` is greater than 0.0 and `segment_3D_volume` 
            is True, masks are stitched in 3D to return volume segmentation. 
            Default is 0.0
        anisotropy : float, optional
            For 3D segmentation, optional rescaling factor (e.g. set to 2.0 if 
            Z is sampled half as dense as X or Y). Default is 0.0
        tile_overlap : float, optional
            Fraction of overlap of tiles when computing flows. Defaults to 0.1.
        invert : bool, optional
            Invert image pixel intensity before running network. Default is False.
        normalize : bool, optional
            If True, normalize image using the other parameters. 
            Default is True
        rescale_intensity_low_val_perc : float, optional
            Rescale intensities so that this is the minimum value in the image. 
            Default is 0.0
        rescale_intensity_high_val_perc : float, optional
            Rescale intensities so that this is the maximum value in the image. 
            Default is 100.0
        # sharpen : int, optional
        #     Sharpen image with high pass filter, recommended to be 1/4-1/8 
        #     diameter of cells in pixels. Default is 0.
        low_percentile : float, optional
            Lower percentile for normalizing image. Default is 1.0
        high_percentile : float, optional
            Higher percentile for normalizing image. Default is 99.0
        norm3D : bool, optional
            Compute normalization across entire z-stack rather than 
            plane-by-plane in stitching mode. Default is False
        tile_norm_blocksize : int, optional
            Size of the tiles for normalization. Default is 0, which means no tiling.
        denoise_rescale : float, optional
            Rescale image intensities to this value. Defaults to 1.0. Unless edge cases, should left to default.
        bsize : int, optional
            Default is 224.
            Don't change it unless you know what you are doing, please!

        """ 
        self.timelapse = False
        image

        eval_kwargs, self.isZstack = self.get_eval_kwargs(
            image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            # cellprob_threshold=cellprob_threshold,
            stitch_threshold=stitch_threshold,
            # min_size=min_size,
            anisotropy=anisotropy,
            # normalize=normalize,
            # resample=resample,
            segment_3D_volume=segment_3D_volume,
            # max_size_fraction=max_size_fraction,
            # flow3D_smooth=flow3D_smooth,
            # tile_overlap=tile_overlap,
        )

        eval_kwargs = self._get_eval_kawrgs_v3(
            eval_kwargs=eval_kwargs,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            resample=resample,
            max_size_fraction=max_size_fraction,
            flow3D_smooth=flow3D_smooth,
            tile_overlap=tile_overlap,
            invert=invert,
        )

        norm_kwargs = _get_normalize_params(
            image=image,
            rescale_intensity_low_val_perc=rescale_intensity_low_val_perc,
            rescale_intensity_high_val_perc=rescale_intensity_high_val_perc,
            # sharpen=sharpen,
            low_percentile=low_percentile,
            high_percentile=high_percentile,
            norm3D=norm3D,
            normalize=normalize,
            tile_norm_blocksize=tile_norm_blocksize
            )

        if self.denoiseModel is not None:
            self.isZstack, self.is_rgb = self.get_zStack_rgb(
                image,)
            if init_imgs:
                if not segment_3D_volume and self.isZstack:
                    image, z_axis, channel_axis = _initialize_image(image, self.is_rgb, 
                                                iter_axis_zstack=0,
                                                isZstack=self.isZstack,
                                                )
                    self.channel_axis = channel_axis # changing the axis for cellpose is handled in the eval loop
                    self.z_axis = z_axis
                else:
                    image, z_axis, channel_axis = _initialize_image(image, self.is_rgb,
                                                isZstack=self.isZstack,
                                                )
                    self.z_axis = z_axis
                    self.channel_axis = channel_axis
            image = self.denoiseModel.run(
                image,
                diameter=diameter,
                do_3D=eval_kwargs['do_3D'],
                normalize_dict=norm_kwargs,
                tile_overlap=tile_overlap,
                timelapse=False,
                bsize=bsize,
                isZstack=self.isZstack,
                init_image=False,  # Denoise model does not need init_imgs, already done
                rescale=denoise_rescale,
                invert=invert,
            )

        eval_kwargs['normalize'] = norm_kwargs if self.denoiseModel is None else True # if denoise model was used, just normalise the image with default parameters
        

        self.img_shape = image.shape
        self.img_ndim = len(self.img_shape)

        init_imgs_eval_loop = init_imgs if self.denoiseModel is None else False
        labels = self.eval_loop(
            image,
            segment_3D_volume=segment_3D_volume,
            init_imgs=init_imgs_eval_loop,
            **eval_kwargs
        )

        self.img_shape = None
        self.img_ndim = None

        return labels
    
    def segment3DT(self, video_data, signals=None, init_imgs=True, **kwargs): # just 2D over time
        self.timelapse = True
        eval_kwargs, self.isZstack = self.get_eval_kwargs(video_data[0], **kwargs)
        eval_kwargs = self._get_eval_kawrgs_v3(
            eval_kwargs=eval_kwargs,
            cellprob_threshold=kwargs['cellprob_threshold'],
            min_size=kwargs['min_size'],
            resample=kwargs['resample'],
            max_size_fraction=kwargs['max_size_fraction'],
            flow3D_smooth=kwargs['flow3D_smooth'],
            tile_overlap=kwargs['tile_overlap'],
            invert=kwargs['invert'],
        )

        norm_kwargs = _get_normalize_params(
            image=video_data,
            normalize=kwargs['normalize'], 
            rescale_intensity_low_val_perc=kwargs['rescale_intensity_low_val_perc'], 
            rescale_intensity_high_val_perc=kwargs['rescale_intensity_high_val_perc'], 
            # sharpen=kwargs['sharpen'],
            low_percentile=kwargs['low_percentile'], 
            high_percentile=kwargs['high_percentile'],
            norm3D=kwargs['norm3D'],
        )

        if self.denoiseModel is not None:
            if init_imgs:
                if not kwargs['segment_3D_volume'] and self.isZstack:
                    video_data, z_axis, channel_axis = _initialize_image(video_data, self.is_rgb,
                                    iter_axis_time=0,
                                    iter_axis_zstack=1,
                                    timelapse=True,
                                    isZstack=self.isZstack,
                                    )
                    self.z_axis = z_axis # changing of axis is handled in the eval loop
                    self.channel_axis = channel_axis 

                else:
                    video_data, z_axis, channel_axis = _initialize_image(video_data, self.is_rgb,
                                                iter_axis_time=0,
                                                timelapse=True,
                                                isZstack=self.isZstack,
                                                )
                    self.z_axis = z_axis # changing of axis is handled in the eval loop
                    self.channel_axis = channel_axis
                    
            video_data = self.denoiseModel.run(
                video_data,
                diameter=eval_kwargs['diameter'],
                do_3D=eval_kwargs['do_3D'],
                normalize_dict=norm_kwargs,
                tile_overlap=kwargs['tile_overlap'],
                timelapse=True,
                bsize=kwargs['bsize'],
                isZstack=self.isZstack,
                init_image=False,  # Denoise model does not need init_imgs, already done
                rescale=kwargs['denoise_rescale'],
                )
        
        self.img_shape = video_data[0].shape
        self.img_ndim = len(self.img_shape)

        eval_kwargs['normalize'] = norm_kwargs if self.denoiseModel is None else True # if denoise model was used, just normalise the image with default parameters

        init_imgs_segment3DT_eval = init_imgs if self.denoiseModel is None else False
        labels = self.segment3DT_eval(
            video_data, eval_kwargs, init_imgs=init_imgs_segment3DT_eval, **kwargs
        )

        self.img_shape = None
        self.img_ndim = None
        return labels

def url_help():
    return 'https://cellpose.readthedocs.io/en/latest/api.html'
