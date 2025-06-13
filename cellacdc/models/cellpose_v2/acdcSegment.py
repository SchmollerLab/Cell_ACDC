import os
from cellacdc.models._cellpose_base.acdcSegment import Model as CellposeBaseModel
import torch
from cellacdc import myutils
from . import AvailableModelsv2

class Model(CellposeBaseModel):
    def __new__(cls, *args, **kwargs):
        myutils.check_install_cellpose(2)
        return super().__new__(cls)
    
    def __init__(
            self, 
            model_type: AvailableModelsv2='cyto', 
            model_path: os.PathLike='',
            net_avg:bool=False, 
            gpu:bool=False,
            device:torch.device|int='None',
            custom_concatenation:bool=False,
            custom_style_on:bool=True,
            custom_residual_on:bool=True,
            custom_diam_mean:float=30.0,

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
            If True and PyTorch for your GPU is correctly installed, 
            denoising and segmentation processes will run on the GPU. 
            Default is False
        directml_gpu : bool, optional
            If True, will attempt to use DirectML for GPU acceleration.
            Only for v3 and v4. v2 loads the model later, which causes problems. Dont want to edit cellpose code too much...
        device : torch.device or int or None
            If not None, this is the device used for running the model
            (torch.device('cuda') or torch.device('cpu')). 
            It overrides `gpu`, recommended if you want to use a specific GPU 
            (e.g. torch.device('cuda:1'). Default is None
        custom_concatenation : bool, optional
            Only effects custom trained models. See cellpose v2 for more info.
        custom_style_on : bool, optional
            Only effects custom trained models. See cellpose v2 for more info.
        custom_residual_on : bool, optional
            Only effects custom trained models. See cellpose v2 for more info.
        custom_diam_mean : float, optional
            Only effects custom trained models. See cellpose v2 for more info.
            Default is 30.0
        backbone : NotGUIParam, optional
            Only for v3

        """

        self.initConstants()
        model_type, model_path, device = myutils.translateStrNone(model_type, model_path, device)
        
        self.check_model_path_model_type(
            model_type=model_type, 
            model_path=model_path, 
        )
        
        print(f'Initializing Cellpose v2...')

        from cellpose import models
        if model_type:
            try:
                self.model = models.Cellpose(
                    gpu=gpu, net_avg=net_avg, 
                    model_type=model_type,
                    device=device,
                )
                self._sizemodelnotfound = False
            except FileNotFoundError:
                self._sizemodelnotfound = True
                self.model = models.CellposeModel(
                    gpu=gpu, net_avg=net_avg, model_type=model_type,
                    device=device,
                )
        elif model_path is not None:
            self._sizemodelnotfound = True
            self.model = models.CellposeModel(
                gpu=gpu, net_avg=net_avg, device=device,
                pretrained_model=model_path,
                concatenation=custom_concatenation,
                style_on=custom_style_on,
                residual_on=custom_residual_on,
                diam_mean=custom_diam_mean,
            )

    def segment(
            self, image,
            diameter:float=0.0,
            flow_threshold:float=0.4,
            cellprob_threshold:float=0.0,
            stitch_threshold:float=0.0,
            min_size:int=15,
            anisotropy:float=0.0,
            normalize:bool=True,
            resample:bool=True,
            segment_3D_volume:bool=False            
        ):
        """Segment image using cellpose eval

        Parameters
        ----------
        image : (Y, X) or (Z, Y, X) numpy.ndarray
            Input image. Either 2D or 3D z-stack.
        diameter : float, optional
            Average diameter (in pixels) of the obejcts of interest. 
            Default is 0.0
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
            z-slice at the time. Best results with 3D data are obtained by 
            passing the entire z-stack, but with a `stitch_threshold` greater 
            than 0 (e.g., 0.4). This way cellpose will internally segment 
            slice-by-slice and it will merge the resulting z-slice masks 
            belonging to the same object. 
            Default is False

        Returns
        -------
        np.ndarray of ints
            Segmentations masks array

        Raises
        ------
        TypeError
            `stitch_threshold` must be 0 when segmenting slice-by-slice.
        """        
        # Preprocess image
        # image = image/image.max()
        # image = skimage.filters.gaussian(image, sigma=1)
        # image = skimage.exposure.equalize_adapthist(image)
        self.img_shape = image.shape
        self.img_ndim = len(self.img_shape)

        eval_kwargs, isZstack = self.get_eval_kwargs_v2(
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

        labels = self.eval_loop(
            image,
            segment_3D_volume=segment_3D_volume,
            isZstack=isZstack,
            **eval_kwargs
        )

        self.img_shape = None
        self.img_ndim = None

        return labels
    
    def segment3DT(self, video_data, signals=None, **kwargs):
        self.img_shape = video_data[0].shape
        self.img_ndim = len(self.img_shape)

        eval_kwargs, isZstack = self.get_eval_kwargs_v2(video_data[0], **kwargs)

        labels = self.segment3DT_eval(
            video_data, isZstack, eval_kwargs, **kwargs
        )


        self.img_shape = None
        self.img_ndim = None
        return labels