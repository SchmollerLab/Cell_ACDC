import os
from cellacdc import myutils, printl
import torch
from cellacdc.models._cellpose_base.acdcSegment import (Model as CellposeBaseModel, 
                                                        GPUDirectMLGPUCPU, 
                                                        cpu_gpu_directml_gpu, 
                                                        check_directml_gpu_gpu,
                                                        setup_gpu_direct_ml, 
                                                        _get_normalize_params)
from . import AvailableModelsv4

class Model(CellposeBaseModel):
    def __new__(cls, *args, **kwargs):
        myutils.check_install_cellpose(4)
        return super().__new__(cls)
    
    def __init__(
            self, 
            model_type: AvailableModelsv4='cpsam',
            model_path: os.PathLike='',
            device_type: GPUDirectMLGPUCPU='cpu',
            device:torch.device|int='None',
            batch_size:int=8,

        ):
        """Initialize Cellpose 4 (Cellpose-SAM) model

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

        """ 
        self.init_successful = False
        self.initConstants()
        self.batch_size = batch_size
        model_type, model_path, device = myutils.translateStrNone(model_type, model_path, device)

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

        model_path = model_path or model_type
        
        major_version = myutils.get_cellpose_major_version()
        print(f'Initializing Cellpose v{major_version}...')

        from cellpose import models
        self.model = models.CellposeModel(
            gpu=gpu,
            device=device,
            pretrained_model=model_path,
            )
        
        setup_gpu_direct_ml(
            self,
            directml_gpu,
            gpu, device)
    
        self.init_successful = True
    
    def _get_eval_kwargs_v4(
            self,
            max_size_fraction:float=0.4,
            invert:bool=False,
            flow3D_smooth:int=0,
            niter:int=0,
            augment:bool=False,
            tile_overlap:float=0.1,
            bsize:int=224,
            # interp:bool=True,
            min_size:int=15,
            cellprob_threshold:float=0.0,
            prev_kwargs:dict=None,
            **kwargs
        ):
        if niter == 0:
            niter = None

        prev_kwargs = self._filter_kwargs(**prev_kwargs)
        
        additional_kwargs = {
            'max_size_fraction': max_size_fraction,
            'invert': invert,
            'flow3D_smooth': flow3D_smooth,
            'niter': niter,
            'augment': augment,
            'tile_overlap': tile_overlap,
            'bsize': bsize,
            'min_size': min_size,
            'cellprob_threshold': cellprob_threshold,
            # 'interp': interp
        }

        prev_kwargs.update(additional_kwargs)
                
        return prev_kwargs

    def _filter_kwargs(
            self,
            **kwargs
        ):

        kwarg_key_list = [
            'channels',
            'diameter',
            'flow_threshold',
            'stitch_threshold',
            'do_3D',
            'anisotropy',
        ]

        for key in list(kwargs.keys()):
            if key not in kwarg_key_list:
                del kwargs[key]
        
        for key in kwarg_key_list:
            if key not in kwargs:
                raise KeyError(
                    f"Key '{key}' not found in kwargs. "
                    "Please provide all required keys."
                )
        
        return kwargs

    # def _filter_kwargs(
    #         self,
    #         **kwargs
    #     ):
    #     kwarg_key_list = [
    #         'diameter',
    #         'flow_threshold',
    #         'cellprob_threshold',
    #         'min_size',
    #         'normalize',
    #         'stitch_threshold',
    #         'anisotropy',
    #     ]

    #     for key in list(kwargs.keys()):
    #         if key not in kwarg_key_list:
    #             del kwargs[key]
        
    #     for key in kwarg_key_list:
    #         if key not in kwargs:
    #             raise KeyError(
    #                 f"Key '{key}' not found in kwargs. "
    #                 "Please provide all required keys."
    #             )
            
    #     return kwargs
        
    def segment(
            self, image,
            diameter:float=0.0,
            flow_threshold:float=0.4,
            cellprob_threshold:float=0.0,
            min_size:int=15,
            max_size_fraction:float=0.4,
            invert:bool=False,
            segment_3D_volume:bool=False,
            stitch_threshold:float=0.0, 
            flow3D_smooth:float=0,
            anisotropy:float=0.0,
            tile_overlap:float=0.1,
            normalize:bool=True,
            rescale_intensity_low_val_perc:float=0.0, 
            rescale_intensity_high_val_perc:float=100.0, 
            # sharpen:int=0,
            low_percentile:float=1.0, 
            high_percentile:float=99.0,
            norm3D:bool=False,
            tile_norm_blocksize: int=0,
            niter:int=0,
            augment:bool=False,
            bsize:int=256,
            # interp:bool=True,

        ):
        """Segment an image using Cellpose (see details in v2)

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
        min_size : int, optional
            Minimum number of pixels per mask, you can turn off this filter 
            with `min_size = -1`. Default is 15
        max_size_fraction : float, optional
            Masks larger than this fraction of total image size are removed. Default is 0.4.
        invert : bool, optional
            Invert image pixel intensity before running network. Default is False.
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
        niter : int, optional
            Number of iterations for dynamics computation. If 0, set proportional to the diameter. Default is 0.
        augment : bool, optional
            Tiles image with overlapping tiles and flips overlapped regions to augment. Default is False.
        tile_overlap : float, optional
            Fraction of overlap of tiles when computing flows. Default is 0.1.
        bsize : int, optional
            Block size for tiles, recommended to keep at 224 (as in training). Default is 224.

        """        
        self.timelapse = False
        self.img_shape = image.shape
        self.img_ndim = len(self.img_shape)

        eval_kwargs, self.isZstack = self.get_eval_kwargs(
            image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            # cellprob_threshold=cellprob_threshold,
            stitch_threshold=stitch_threshold,
            # min_size=min_size,
            anisotropy=anisotropy,
            # normalize=normalize,
            segment_3D_volume=segment_3D_volume,
            # max_size_fraction=max_size_fraction,
            # flow3D_smooth=flow3D_smooth,
            # tile_overlap=tile_overlap,
        )

        eval_kwargs = self._get_eval_kwargs_v4(
            max_size_fraction=max_size_fraction,
            invert=invert,
            flow3D_smooth=flow3D_smooth,
            niter=niter,
            augment=augment,
            tile_overlap=tile_overlap,
            bsize=bsize,
            # interp=interp,
            min_size=min_size,
            cellprob_threshold=cellprob_threshold,
            prev_kwargs=eval_kwargs
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
        
        eval_kwargs['normalize'] = norm_kwargs

        labs = self.eval_loop(
            image, segment_3D_volume, **eval_kwargs
        )

        self.img_shape = None
        self.img_ndim = None

        return labs

    def segment3DT(self, video_data, signals=None, **kwargs):
        self.timelapse = True
        self.img_shape = video_data[0].shape
        self.img_ndim = len(self.img_shape)
        
        image = video_data[0]
        eval_kwargs, self.isZstack = self.get_eval_kwargs(
            image,
            **kwargs        
        )

        eval_kwargs = self._get_eval_kwargs_v4(
            **kwargs,
            prev_kwargs=eval_kwargs
        )

        labels = self.segment3DT_eval(
            video_data, eval_kwargs, **kwargs
        )

        self.img_shape = None
        self.img_ndim = None

        return labels 

def url_help():
    return 'https://cellpose.readthedocs.io/en/latest/api.html'