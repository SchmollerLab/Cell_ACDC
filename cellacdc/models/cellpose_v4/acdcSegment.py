import os

from cellacdc import myutils, printl

from cellacdc.models.cellpose_v2 import acdcSegment as acdc_cp2
from cellacdc.models.cellpose_v2.acdcSegment import _initialize_image
import torch
import numpy as np
    
class backboneOptions:
    """Options for cellpose backbone"""
    values = ['default', "transformer"]

CellposeV2Model = acdc_cp2.Model
AvailableModels = acdc_cp2.AvailableModels
from cellpose import models

class Model(CellposeV2Model):
    def __init__(
            self, 
            model_type: AvailableModels='cpsam', 
            model_path: os.PathLike='',
            gpu:bool=False,
            device:torch.device|int='None',
        ):
        """Initialize cellpose 2  model

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
            If True and PyTorch for your GPU (if CUDA is not available, DirectML will be used) is 
            correctly installed, denoising and segmentation processes will run on the GPU.
            Default is False
        device : torch.device or int or None
            If not None, this is the device used for running the model
            (torch.device('cuda') or torch.device('cpu')). 
            It overrides `gpu`, recommended if you want to use a specific GPU 
            (e.g. torch.device('cuda:1'). Default is None

        """ 
        if device == 'None':
            device = None
        
        if model_path == 'None':
            model_path = None
        
        if model_type == 'None':
            model_type = None
        
        if model_path is not None and model_type is not None:
            raise TypeError(
                "You cannot set both `model_type` and `model_path`. "
                "Please set only one of them."
            )
        
        if model_path is None and model_type is None:
            raise TypeError(
                "You must set either `model_type` or `model_path`. "
                "Please set one of them."
            )
        
        model_path = model_path or model_type
        
        major_version = myutils.get_cellpose_major_version()
        print(f'Initializing Cellpose v{major_version}...')

        self._sizemodelnotfound = True

        self.model = models.CellposeModel(
            gpu=gpu,
            device=device,
            model_type=model_path,
            )
        self.is_rgb = False
    
    def _get_eval_kwargs_v4(
            self,
            batch_size:int=64,
            max_size_fraction:float=0.4,
            invert:bool=False,
            flow3D_smooth:int=0,
            niter:int=0,
            augment:bool=False,
            tile_overlap:float=0.1,
            bsize:int=224,
            interp:bool=True,
            v2_kwargs:dict=None,
            **kwargs
        ):
        if niter == 0:
            niter = None

        v2_kwargs = self._filter_v2_kwargs(**v2_kwargs)

        additional_kwargs = {
            'batch_size': batch_size,
            'max_size_fraction': max_size_fraction,
            'invert': invert,
            'flow3D_smooth': flow3D_smooth,
            'niter': niter,
            'augment': augment,
            'tile_overlap': tile_overlap,
            'bsize': bsize,
            'interp': interp
        }

        eval_kwargs = {**kwargs, **additional_kwargs, **v2_kwargs}
                
        return eval_kwargs

    def _filter_v2_kwargs(
            self,
            **kwargs
        ):

        kwarg_key_list = [
            'channels',
            'diameter',
            'flow_threshold',
            'cellprob_threshold',
            'stitch_threshold',
            'min_size',
            'normalize',
            'do_3D',
            'anisotropy',
        ]

        for key in kwargs.keys():
            if key not in kwarg_key_list:
                del kwargs[key]
        
        for key in kwarg_key_list:
            if key not in kwargs:
                raise KeyError(
                    f"Key '{key}' not found in kwargs. "
                    "Please provide all required keys."
                )
        
        return kwargs
        
    def segment(
            self, image,
            diameter:float=30.0,
            batch_size:int=64,
            flow_threshold:float=0.4,
            cellprob_threshold:float=0.0,
            min_size:int=15,
            max_size_fraction:float=0.4,
            invert:bool=False,
            normalize:bool=True,
            segment_3D_volume:bool=False,       
            stitch_threshold:float=0.0,
            flow3D_smooth:int=0,
            anisotropy:float=0.0,
            niter:int=0,
            augment:bool=False,
            tile_overlap:float=0.1,
            bsize:int=224,
            interp:bool=True,

        ):
        """Segment an image using Cellpose (see details in v2)

        Parameters
        ----------
        image : (Y, X) or (Z, Y, X) numpy.ndarray
            Input image. Either 2D or 3D z-stack.
        diameter : float, optional
            Average diameter (in pixels) of the obejcts of interest. 
            Default is 0.0
        batch_size : int, optional
            Number of 256x256 patches to run simultaneously on the GPU (can be adjusted depending on GPU memory usage). Default is 64.
        normalize : bool or dict, optional # discuss with francesco
            If True, normalize data so 0.0=1st percentile and 1.0=99th percentile of image intensities in each channel. Can also pass a dictionary of parameters (all keys optional, default values shown):
                - "lowhigh"=None : normalization values for 0.0 and 1.0 as [low, high] (if not None, all following parameters ignored)
                - "sharpen"=0 : sharpen image with high pass filter
                - "normalize"=True : run normalization (if False, all following parameters ignored)
                - "percentile"=None : percentiles to use as [perc_low, perc_high]
                - "tile_norm_blocksize"=0 : compute normalization in tiles (set to window size in pixels to enable)
                - "norm3D"=True : compute normalization across entire z-stack in stitching mode
            Default is True.
        invert : bool, optional
            Invert image pixel intensity before running network. Default is False.
        flow_threshold : float, optional
            Flow error threshold (all cells with errors below threshold are kept; not used for 3D). Default is 0.4.
        cellprob_threshold : float, optional
            All pixels with value above threshold kept for masks; decrease to find more and larger masks. Default is 0.0.
        flow3D_smooth : int, optional
            If do_3D and flow3D_smooth > 0, smooth flows with gaussian filter of this stddev. Default is 0.
        anisotropy : float, optional
            For 3D segmentation, optional rescaling factor (e.g., set to 2.0 if Z is sampled half as dense as X or Y). Default is None.
        stitch_threshold : float, optional
            If stitch_threshold > 0.0 and not do_3D, masks are stitched in 3D to return volume segmentation. Default is 0.0.
        min_size : int, optional
            All ROIs below this size (in pixels) will be discarded. Default is 15.
        max_size_fraction : float, optional
            Masks larger than this fraction of total image size are removed. Default is 0.4.
        niter : int, optional
            Number of iterations for dynamics computation. If 0, set proportional to the diameter. Default is 0.
        augment : bool, optional
            Tiles image with overlapping tiles and flips overlapped regions to augment. Default is False.
        tile_overlap : float, optional
            Fraction of overlap of tiles when computing flows. Default is 0.1.
        bsize : int, optional
            Block size for tiles, recommended to keep at 224 (as in training). Default is 224.
        interp : bool, optional
            Interpolate during 2D dynamics (not available in 3D). Default is True.

        """        
        # Preprocess image
        # image = image/image.max()
        # image = skimage.filters.gaussian(image, sigma=1)
        # image = skimage.exposure.equalize_adapthist(image)
        
                # Run cellpose eval
        
        self.img_shape = image.shape
        self.img_ndim = len(self.img_shape)

        eval_kwargs, isZstack = self._get_eval_kwargs(
            image,
            diameter=diameter,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            stitch_threshold=stitch_threshold,
            min_size=min_size,
            anisotropy=anisotropy,
            normalize=normalize,
            segment_3D_volume=segment_3D_volume            
        )

        eval_kwargs = self._get_eval_kwargs_v4(
            batch_size=batch_size,
            max_size_fraction=max_size_fraction,
            invert=invert,
            flow3D_smooth=flow3D_smooth,
            niter=niter,
            augment=augment,
            tile_overlap=tile_overlap,
            bsize=bsize,
            interp=interp,
            v2_kwargs=eval_kwargs
        )

        labs = self._eval_loop(
            image, segment_3D_volume, isZstack, **eval_kwargs
        )

        self.img_shape = None
        self.img_ndim = None

        return labs

    def segment3DT(self, video_data, signals=None, **kwargs):

        self.img_shape = video_data[0].shape
        self.img_ndim = len(self.img_shape)
        
        image = video_data[0]
        eval_kwargs, isZstack = self._get_eval_kwargs(
            image,
            **kwargs        
        )

        eval_kwargs = self._get_eval_kwargs_v4(
            **kwargs,
            v2_kwargs=eval_kwargs
        )

        labels = self._segment3DT_eval(
            video_data, signals, isZstack, eval_kwargs, **kwargs
        )

        self.img_shape = None
        self.img_ndim = None

        return labels 

def url_help():
    return 'https://cellpose.readthedocs.io/en/latest/api.html'