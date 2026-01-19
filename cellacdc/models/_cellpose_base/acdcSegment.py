import torch
import numpy as np
import skimage.measure

from typing import Tuple

from cellacdc import printl, myutils, core

import inspect

class BackboneOptions:
    """Options for cellpose backbone"""
    values = ['default', "transformer"]

class GPUDirectMLGPUCPU:
    """Options for DirectML GPU acceleration"""
    values = ['cpu', 'gpu','directml_gpu']


def cpu_gpu_directml_gpu(
        input_string: str,
        ):
    """Translate input string to cpu, gpu or directml_gpu.
    """
    directml_gpu = False
    gpu = False
    input_string = input_string.lower()
    if input_string == 'cpu':
        pass
    elif input_string == 'gpu':
        gpu = True
    elif input_string == 'directml_gpu':
        directml_gpu = True
    else:
        raise ValueError(
            f"Invalid input string '{input_string}'. "
            "Expected 'cpu', 'gpu' or 'directml_gpu'."
        )
    return directml_gpu, gpu

class DealWithSecondChannelOptions:
    """Options available for dealing with second channel"""
    values = ['together','separately', 'ignore']

def check_deal_with_second_channel(
        input_string: DealWithSecondChannelOptions, is_rgb: bool
):
    if input_string not in DealWithSecondChannelOptions.values:
        raise ValueError(
            f"Invalid deal_with_second_channel option '{input_string}'. "
            f"Expected one of {DealWithSecondChannelOptions.values}."
        )
    input_string = input_string.lower()
    seperatly= False
    together = False
    ignore = False
    if not is_rgb:
        pass
    elif input_string == 'separately':
        seperatly = True
    elif input_string == 'together':
        together = True
    elif input_string == 'ignore':
        ignore = True
    else:
        raise ValueError(
            f"Invalid deal_with_second_channel option '{input_string}'. "
            f"Expected one of {DealWithSecondChannelOptions.values}."
        )
    return seperatly, together, ignore

class Model:
    def __init__(
            self,
        ):
        """Initialize cellpose base model class, which is used in the cellpose versions
        """

        self.initConstants()
    
        
    def check_model_path_model_type(self, model_path, model_type):
        if model_path == 'None' or not model_path:
            model_path = None
        
        if model_type == 'None' or not model_type:
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

    def initConstants(self, is_rgb=False):
        self.is_rgb = is_rgb
        self.img_shape = None
        self.img_ndim = None
        self.z_axis = None
        self.channel_axis = None
        self.cp_version  = myutils.get_cellpose_major_version()
        self._sizemodelnotfound = True
        self.batch_size = None
        self.printed_model_params = False
        
    def setupLogger(self, logger):
        from cellpose import models
        models.models_logger = logger
    
    def closeLogger(self):
        from cellpose import models
        handlers = models.models_logger.handlers[:]
        for handler in handlers:
            handler.close()
            models.models_logger.removeHandler(handler)
    
    def _eval(self, image, **kwargs):
        if self.batch_size is not None:
            kwargs['batch_size'] = self.batch_size
        if self.cp_version == 4:
            del kwargs['channels']
            kwargs['channel_axis'] = self.channel_axis
            kwargs['z_axis'] = self.z_axis
        if self.cp_version == 3:
            kwargs["channel_axis"] = self.channel_axis
            kwargs["z_axis"] = self.z_axis
            
        if not self.printed_model_params:
            if isinstance(image, list):
                sample_img = image[0]
                shape = sample_img.shape
                shape = f"{len(image)} images of shape {shape}"
            else:
                sample_img = image
                shape = image.shape
            print("This is what is being passed to cellpose:")
            print(f"Running model on image shape: {shape}, kwargs: {kwargs}")
            if self.is_rgb:
                for i, subarr in enumerate(np.moveaxis(sample_img, -3, 0)):
                    print(f"Channel {i+1} min: {subarr.min()}, max: {subarr.max()}")
            else:
                print(f"Image min: {sample_img.min()}, max: {sample_img.max()}")
            self.printed_model_params = True
        
        out, removed_kwargs = myutils.try_kwargs(
            self.model.eval,
            image,
            **kwargs
        )
        segm = out[0]
        if removed_kwargs:
            print(
                f"""
                The following kwargs could not be used:
                {removed_kwargs}
                """
            )
            for kwarg in removed_kwargs:
                if kwarg in kwargs:
                    del kwargs[kwarg]
                else:
                    printl(
                        f"Warning: {kwarg} not found in kwargs, "
                        "but was removed from eval method."
                    )
        return segm
    
    def second_ch_img_to_stack(self, first_ch_data, second_ch_data):
        # The 'cyto' model can work with a second channel (e.g., nucleus).
        # However, it needs to be encoded into one of the RGB channels
        # Here we put the first channel in the 'red' channel and the 
        # second channel in the 'green' channel. We then pass
        # `channels = [1,2]` to the segment method
        rgb_stack = np.zeros((*first_ch_data.shape, 3), dtype=first_ch_data.dtype)
        
        R_slice = [slice(None)]*(rgb_stack.ndim)
        R_slice[-1] = 0
        R_slice = tuple(R_slice)
        rgb_stack[R_slice] = first_ch_data

        G_slice = [slice(None)]*(rgb_stack.ndim)
        G_slice[-1] = 1
        G_slice = tuple(G_slice)

        rgb_stack[G_slice] = second_ch_data
        
        self.is_rgb = True

        return rgb_stack
    
    def get_zStack_rgb(self, image):
        if self.img_shape is None:
            self.img_shape = image.shape
        if self.img_ndim is None:
            self.img_ndim = len(self.img_shape)

        self.is_rgb = (self.img_shape[-1] == 3 or self.img_shape[-1] == 4) if not self.is_rgb else self.is_rgb
        remaining_dims = self.img_ndim
        if self.is_rgb:
            remaining_dims -= 1
        if self.timelapse:
            remaining_dims -= 1

        self.isZstack = (
            remaining_dims == 3
        )

        return self.isZstack, self.is_rgb
    
    def get_eval_kwargs(
            self, image,
            diameter=0.0,
            flow_threshold=0.4,
            # cellprob_threshold=0.0,
            stitch_threshold=0.0,
            # min_size=15,
            anisotropy=0.0,
            # normalize=True,
            # resample=True,
            segment_3D_volume=False,
            # max_size_fraction=0.4,
            # flow3D_smooth=0,
            # tile_overlap=0.1,
            **kwargs
        ):
        """Get evaluation kwargs for the model.eval method, accurate for v2.
        """

        if diameter == 0.0 and self._sizemodelnotfound:
            raise TypeError(
                'Diameter is 0.0 but size model is not found. '
                'Please set diameter to a non-zero value.'
            )
    

        if self.img_shape is None:
            self.img_shape = image.shape
        if self.img_ndim is None:
            self.img_ndim = len(self.img_shape)

        isZstack, is_rgb = self.get_zStack_rgb(image)

        if anisotropy == 0.0 and segment_3D_volume:
            if not self.printed_model_params:
                print(
                    'Anisotropy is 0.0 but segment_3D_volume is True. '
                    'Please set anisotropy to a non-zero value.' \
                    'For now set to 1.0, assuming isotropic data.'
                )
            anisotropy = 1.0

        elif not isZstack:
            if not self.printed_model_params:
                print(
                    """Anisotropy is set to 1.0 (assuming isotropic data),
                    since data is not a z-stack""")
            anisotropy = 1.0
        
        do_3D = segment_3D_volume
        if not isZstack:
            stitch_threshold = 0.0
            segment_3D_volume = False
            do_3D = False
        
        if stitch_threshold > 0:
            if not self.printed_model_params:
                print(
                    'Using stiching mode instead of trying to segment 3D volume.'
                    )
            do_3D = False
        
        if isZstack and flow_threshold > 0:
            if not self.printed_model_params:
                print(
                    'Flow threshold is not used for 3D segmentation. '
                    'Setting it to 0.0.'
                )
            flow_threshold = 0.0
            
        if flow_threshold==0.0:
            flow_threshold = None

        channels = [0,0] if not is_rgb else [1,2]

        eval_kwargs = {
            'channels': channels,
            'diameter': diameter,
            'flow_threshold': flow_threshold,
            #'cellprob_threshold': cellprob_threshold,
            'stitch_threshold': stitch_threshold,
            # 'min_size': min_size,
            # 'normalize': normalize,
            'do_3D': do_3D,
            'anisotropy': anisotropy,
            # 'resample': resample,
            # 'max_size_fraction': max_size_fraction,
            # 'flow3D_smooth': flow3D_smooth,
            # 'tile_overlap': tile_overlap
        }

        if not segment_3D_volume and isZstack and stitch_threshold>0:
            raise TypeError(
                "`stitch_threshold` must be 0 when segmenting slice-by-slice. "
                "Alternatively, set `segment_3D_volume = True`."
            )
        
        return eval_kwargs, isZstack

    def eval_loop(
            self, images, segment_3D_volume, init_imgs=True, **eval_kwargs
        ):
        """No support for time lapse. This is handles in self._segment3DT_eval

        Parameters
        ----------
        images : np.ndarray
            Input image to be segmented. It can be 2D, 3D or 4D.
        segment_3D_volume : bool
            If True, the image is assumed to be a 3D z-stack.
        isZstack : bool
            If True, the image is assumed to be a z-stack.
        innit_imgs : bool, optional
            If True, the image is initialized. Default is True.

        Returns
        -------
        np.ndarray
            Segmentation masks array. If `segment_3D_volume` is True, 
            the shape is (Z, Y, X) or (T, Z, Y, X). If `segment_3D_volume` 
            is False, the shape is (Y, X) or (T, Y, X).
        """
        if self.img_shape is None:
            self.img_shape = images.shape

        if not segment_3D_volume and self.isZstack: # segment on a per slice basis
            if init_imgs:
                images, z_axis, channel_axis = _initialize_image(
                    images, self.is_rgb, iter_axis_zstack=0,
                    isZstack=self.isZstack,
                )
            else:
                z_axis = self.z_axis
                channel_axis = self.channel_axis
            self.z_axis = None # since we are segmenting slice-by-slice
            self.channel_axis = channel_axis - 1 if channel_axis is not None else None # since we iterate over z-axis
            if self.channel_axis is None:
                labels = np.zeros(images.shape, dtype=np.uint32)
            else:
                shape = images.shape[:channel_axis] + images.shape[channel_axis+1:]
                labels = np.zeros(shape, dtype=np.uint32)
            for i, z_img in enumerate(images):
                lab = self._eval(z_img, **eval_kwargs)
                labels[i] = lab
            labels = skimage.measure.label(labels>0)
        else:
            if init_imgs:
                images, z_axis, channel_axis = _initialize_image(images, self.is_rgb,
                                            isZstack=self.isZstack,
                                            )
                self.z_axis = z_axis
                self.channel_axis = channel_axis
            else:
                z_axis = self.z_axis
                channel_axis = self.channel_axis
                
            labels = self._eval(images, **eval_kwargs)
        
        return labels
    
    def segment3DT_eval(
            self, images, eval_kwargs, init_imgs=True, **kwargs
        ):
        if not kwargs['segment_3D_volume'] and self.isZstack:
            if init_imgs:
                images, z_axis, channel_axis = _initialize_image(images, self.is_rgb,
                                iter_axis_time=0,
                                iter_axis_zstack=1,
                                timelapse=True,
                                isZstack=self.isZstack,
                                )
            else:
                z_axis = self.z_axis
                channel_axis = self.channel_axis
                    
            self.z_axis = z_axis - 2 if z_axis is not None else None # video doesnt count as dim. iterate over time
            self.channel_axis = channel_axis - 2 if channel_axis is not None else None

            # Passing entire 4D video and segmenting slice-by-slice is 
            # not possible --> iterate each frame and run normal segment
            if self.channel_axis is None:
                labels = np.zeros(images.shape, dtype=np.uint32)
            else:
                shape = images.shape[:channel_axis] + images.shape[channel_axis+1:]
                labels = np.zeros(shape, dtype=np.uint32)
            for i, img_t in enumerate(images):
                lab = self.eval_loop(
                    img_t, segment_3D_volume=False,
                    init_imgs=False,
                    **eval_kwargs
                )
                labels[i] = lab
        else:
            eval_kwargs['channels'] = [eval_kwargs['channels']]*len(images)
            if init_imgs:
                images, z_axis, channel_axis = _initialize_image(images, self.is_rgb,
                                            iter_axis_time=0,
                                            timelapse=True,
                                            isZstack=self.isZstack,
                                            )
            else:
                z_axis = self.z_axis
                channel_axis = self.channel_axis

            self.z_axis = z_axis - 1 if z_axis is not None else None # video doesnt count as dim
            self.channel_axis = channel_axis - 1 if channel_axis is not None else None
            images = [image.astype(np.float32) for image in images] # convert to list
            labels = np.array(self._eval(images, **eval_kwargs))

        return labels
    
def _initialize_image(image:np.ndarray,
                      is_rgb:bool,
                    #   single_img_shape:Tuple[int],
                    #   single_img_ndim:int,
                      iter_axis_time:int=None,
                      iter_axis_zstack:int=None,
                      target_shape:Tuple[int]=None,
                      timelapse:bool=False,
                      isZstack:bool=False,
                      target_axis_iter:Tuple[int]=None,
                      add_rgb:bool=False,
                      ):
    """Tries to initialize image for cellpose.
    You will have to specify the target shape and the axis to iterate over.
    Target order of dimensions is (Z x nchan x Y x X) or (T x Z x nchan x Y x X)

    Parameters
    ----------
    image : np.ndarray
        Input image to be initialized. It can be 2D, 3D or 4D.
    is_rgb : bool
        If True, the image is assumed to be RGB.
    img_shape : Tuple[int]
        Shape of the true input image, i.e a single frame's dims.
    img_ndim : int
        Dim of the true input image, i.e a single frame's dims.
    iter_axis_time : int, optional
        axis to be iterated over if iteration over time is needed.
    iter_axis_zstack : int, optional
        axis to be iterated over if iteration over zstack is needed.
    target_shape : Tuple[int], optional
        Shape of the target image, depending if time lapse or not.
    timelapse : bool, optional
        If True, the image is assumed to be a time lapse. Default is False.
    isZstack : bool, optional
        If True, the image is assumed to be a z-stack. Default is False.
    target_axis_iter : Tuple[int], optional
        Output axes along the output should be stacked, by default None
    add_rgb : bool, optional
        If False, will not try to add RGB channels to the image if it is not RGB.

    Returns
    -------
    np.ndarray
        Initialized image, with the shape of the target image.
        The image is normalized to [0, 255] and converted to float32.
        The image is also transposed to the target shape.
    """

    true_img_shape = image.shape
    iter_axis = None

    if timelapse and isZstack:
        if len(true_img_shape) < 4 or (is_rgb and len(true_img_shape) < 5):
            raise TypeError(
                f"Image is {len(true_img_shape)}D with shape {true_img_shape}. "
                "It was expected to have 4D shape (T x Z x Y x X x nchan)"
            )
        
        z_axis = 1
        if add_rgb:
            target_shape = (true_img_shape[0], true_img_shape[1], 3, true_img_shape[-2], true_img_shape[-1])
            channel_axis = 2
        elif is_rgb:
            target_shape = (true_img_shape[0], true_img_shape[1], 3, true_img_shape[-3], true_img_shape[-2])
            channel_axis = 2
        else:
            target_shape = (true_img_shape[0], true_img_shape[1], true_img_shape[-2], true_img_shape[-1])
            channel_axis = None


        if iter_axis_time is not None and iter_axis_zstack is not None:
            iter_axis = [iter_axis_time, iter_axis_zstack]
            target_axis_iter = [0, 1]
        elif iter_axis_time  is not None and iter_axis_zstack is None:
            iter_axis = [iter_axis_time]
            target_axis_iter = [0]
        elif iter_axis_time is None and iter_axis_zstack is not None:
            iter_axis = [iter_axis_zstack]
            target_axis_iter = [1]
        else:
            iter_axis = None
            target_axis_iter = None
    
    elif timelapse and not isZstack:
        z_axis = None
        if len(true_img_shape) < 3 or (is_rgb and len(true_img_shape) < 4):
            raise TypeError(
                f"Image is {len(true_img_shape)}D with shape {true_img_shape}. "
                "It was expected to have 3D shape (T x Y x X x nchan)"
            )
        if add_rgb:
            target_shape = (true_img_shape[0], 3, true_img_shape[-2], true_img_shape[-1])
            channel_axis = 1
        elif is_rgb:
            target_shape = (true_img_shape[0], 3, true_img_shape[-3], true_img_shape[-2])
            channel_axis = 1
        else:
            target_shape = (true_img_shape[0], true_img_shape[-2], true_img_shape[-1])
            channel_axis = None

        if iter_axis_time is not None:
            iter_axis = [iter_axis_time]
            target_axis_iter = [0]
        else:
            iter_axis = None
            target_axis_iter = None
    
    elif not timelapse and isZstack:
        z_axis = 0
        if len(true_img_shape) < 3 or (is_rgb and len(true_img_shape) < 4):
            raise TypeError(
                f"Image is {len(true_img_shape)}D with shape {true_img_shape}. "
                "It was expected to have 3D shape (Z x Y x X x nchan)"
            )
        if add_rgb:
            target_shape = (true_img_shape[0], 3, true_img_shape[-2], true_img_shape[-1])
            channel_axis = 1
        elif is_rgb:
            target_shape = (true_img_shape[0], 3, true_img_shape[-3], true_img_shape[-2])
            channel_axis = 1
        else:
            target_shape = (true_img_shape[0], true_img_shape[-2], true_img_shape[-1])
            channel_axis = None

        if iter_axis_zstack is not None:
            iter_axis = [iter_axis_zstack]
            target_axis_iter = [0]
        else:
            iter_axis = None
            target_axis_iter = None
    
    elif not timelapse and not isZstack:
        z_axis = None

        if len(true_img_shape) < 2 or (is_rgb and len(true_img_shape) < 3):
            raise TypeError(
                f"Image is {len(true_img_shape)}D with shape {true_img_shape}. "
                "It was expected to have 2D shape (Y x X x nchan)"
            )
        if add_rgb:
            target_shape = (3, true_img_shape[-2], true_img_shape[-1])
            channel_axis = 0
        elif is_rgb:
            target_shape = (3, true_img_shape[-3], true_img_shape[-2])
            channel_axis = 0    
        else:
            target_shape = (true_img_shape[-2], true_img_shape[-1])
            channel_axis = None

    if iter_axis is not None:
        # Build an index tuple: set first iter_axis to 0, others to slice(None)
        idx = [0 if i in iter_axis else slice(None) for i in range(len(true_img_shape))]
        single_img_from_iter_axis = image[tuple(idx)]
    else:
        single_img_from_iter_axis = image
        
    if single_img_from_iter_axis is not None:
        single_img_shape = single_img_from_iter_axis.shape
        single_img_ndim = len(single_img_shape)
    else:
        single_img_shape = true_img_shape
        single_img_ndim = len(single_img_shape)
    
    single_img_isZstack = isZstack if iter_axis_zstack is None else False
    single_img_timelapse = timelapse if iter_axis_time is None else False
    
    from cellacdc._core import _initialize_single_image
    image = core.apply_func_to_imgs(
        image,
        _initialize_single_image,
        iter_axis=iter_axis,
        target_type=np.float32,
        target_shape=target_shape,
        target_axis_iter=target_axis_iter,
        is_rgb=is_rgb,
        isZstack=single_img_isZstack,
        img_shape=single_img_shape,
        img_ndim=single_img_ndim,
        timelapse=single_img_timelapse,

        add_rgb=add_rgb,
    )
    return image, z_axis, channel_axis

def check_directml_gpu_gpu(model_name, directml_gpu, gpu, ask_install=True):
    if ask_install:
        proceed, available_frameworks_list = myutils.check_gpu_available(
            model_name, 
            use_gpu=(gpu or directml_gpu), 
            cuda=gpu, 
            return_available_gpu_type=True
        )
    else:
        proceed = True
        available_frameworks_list = ['cuda', 'directML']

    if 'cuda' not in available_frameworks_list:
        gpu = False
    if 'directML' not in available_frameworks_list:
        directml_gpu = False

    if not proceed:
        return directml_gpu, gpu, proceed
    if directml_gpu:
        from cellacdc.models._cellpose_base._directML import init_directML
        directml_gpu = init_directML()

    if directml_gpu and gpu:
        print(
            """
            gpu is preferable to directml_gpu, but doesn't work with non NVIDIA GPUs.
            Since directml_gpu and set to True, the gpu argument will be ignored.
            """
        )
        gpu = False
        
    return directml_gpu, gpu, proceed

def setup_gpu_direct_ml(self, directml_gpu, gpu, device):

    if directml_gpu:
        from cellacdc.models._cellpose_base._directML import setup_directML
        setup_directML(self)

        from cellacdc.core import fix_sparse_directML
        fix_sparse_directML()
    
    if gpu: # sometimes gpu is not properly set up ^^
        from cellacdc.models._cellpose_base._directML import setup_custom_device
        if device is None:
            device = 0
        try:
            device = int(device)
        except ValueError:
            pass

        if isinstance(device, int):
            device = torch.device(f'cuda:{device}')
        elif isinstance(device, str):
            device = torch.device(device)
        
        setup_custom_device(self, device)

def _get_normalize_params(
        image,
        normalize=False, 
        rescale_intensity_low_val_perc=0.0, 
        rescale_intensity_high_val_perc=100.0, 
        # sharpen=0,
        low_percentile=1.0, 
        high_percentile=99.0,
        norm3D=False,
        cp_version=4,
        tile_norm_blocksize=0,
    ):
    if not normalize:
        return False
        
    rescale_intensity_low_val_perc = float(rescale_intensity_low_val_perc)
    rescale_intensity_high_val_perc = float(rescale_intensity_high_val_perc)
    low_percentile = float(low_percentile)
    high_percentile = float(high_percentile)

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
    
    # normalize_kwargs['sharpen'] = sharpen
    normalize_kwargs['percentile'] = (low_percentile, high_percentile)
    normalize_kwargs['norm3D'] = norm3D

    if cp_version == 4:
        normalize_kwargs['tile_norm_blocksize'] = tile_norm_blocksize
    elif cp_version == 3:
        normalize_kwargs['tile_norm'] = tile_norm_blocksize
    
    return normalize_kwargs

def url_help():
    return 'https://cellpose.readthedocs.io/en/latest/api.html'