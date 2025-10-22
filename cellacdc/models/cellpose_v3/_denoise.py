import numpy as np
from tqdm import tqdm

from cellacdc import printl
from cellpose.denoise import DenoiseModel
from . import AvailableModelsv3Denoise
import os
from cellacdc import myutils

from cellacdc.models._cellpose_base.acdcSegment import (_initialize_image, GPUDirectMLGPUCPU, 
                                                        cpu_gpu_directml_gpu, check_directml_gpu_gpu,
                                                        setup_gpu_direct_ml, _get_normalize_params,
                                                        DealWithSecondChannelOptions, check_deal_with_second_channel)
import torch

from _types import NotGUIParam
import itertools

class CellposeDenoiseModel(DenoiseModel):
    def __init__(
            self, 
            device_type: GPUDirectMLGPUCPU='cpu',
            device: torch.device | int | None = None,
            batch_size: int = 8,
            denoise_model: AvailableModelsv3Denoise='denoise_cyto3',
            deal_with_second_channel: DealWithSecondChannelOptions = 'together',
            denoise_model_path: os.PathLike='',
            diam_mean: float = 30.0,
            denoise_nchan: int = 1,
            is_rgb: NotGUIParam = False,
            ask_install_gpu: NotGUIParam = True,
        ):
        """Initialize cellpose 3.0 denoising model

        Parameters
        ----------
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
            -nuclei: Cellpose nuclei model (used when segmenting two channels and
             nchan2 is set to True for the second channel)
        deal_with_second_channel : DealWithSecondChannelOptions, optional
            How to denoise the second channel. Can be denoised separately, or together (if model supports it), or not at all.
        denoise_model_path : os.PathLike, optional
            Path to a custom cellpose denoise model file.
        diam_mean : float, optional
            Mean diameter of objects in the image for denoising during training.
            If using a pretrained model, it is recommended to leave it as 0.0, 
            which will use the default diameter 30
        denoise_nchan : int, optional
            Number of channels in the denoised image. Default is 1.
            All cellpose denoise models are single channel, so almost always 1.
        is_rgb : NotGUIParam, optional
            If True, the model will expect RGB images. Default is False.
        ask_install_gpu : NotGUIParam, optional
            If True, the model will ask to install GPU support if it is not available.
        """
        self.first_second_channel = [0, 1]
        self.cellpose_greyscale_channel = [0, 0]
        self.cellpose_rgb_channel = [1, 2]

        self.printed_model_params = False


        self.is_rgb = is_rgb
        self.denoise_second_channel_separately, self.denoise_second_channel_together, self.ignore_second_channel = check_deal_with_second_channel(
            deal_with_second_channel, is_rgb)
        
        self.batch_size = batch_size
        denoise_model, denoise_model_path, device = myutils.translateStrNone(denoise_model, denoise_model_path, device)
        directml_gpu, gpu =  cpu_gpu_directml_gpu(
            input_string=device_type,
        )

        self.nstr = denoise_model.split('_')[-1] if denoise_model else None

        directml_gpu, gpu, proceed= check_directml_gpu_gpu(
            'cellpose_v3', directml_gpu, gpu, ask_install=ask_install_gpu
        )
        
        if not proceed:
            return

        if denoise_model_path and denoise_model: 
            raise ValueError(
                "You can only specify one of 'denoise_model_path' or 'denoise_model'."
            )

        if diam_mean == 0.0:
            diam_mean = 30
        
        if diam_mean != 30 and denoise_model:
            printl(
                f"[WARNING] It is recommended not to set 'denoise_diameter' for pretrained models!"
            )

        super().__init__(gpu=gpu, pretrained_model=denoise_model_path, diam_mean=diam_mean, chan2=self.denoise_second_channel_together,
                         nchan=denoise_nchan, device=device, model_type=denoise_model)
        
        setup_gpu_direct_ml(
            self,
            directml_gpu,
            gpu, device)

    
    def run(
            self,
            image: np.ndarray,
            diameter:float=0.0,
            do_3D:bool=True,
            invert:bool=False,
            normalize:bool=True, 
            rescale_intensity_low_val_perc:float=0.0, 
            rescale_intensity_high_val_perc:float=100.0, 
            # sharpen:float=0,
            low_percentile:float=1.0, 
            high_percentile:float=99.0,
            norm3D:bool=False,
            rescale:float=1.0,
            tile_overlap:float=0.1,
            isZstack:NotGUIParam=False,
            timelapse:NotGUIParam=False,
            init_image:NotGUIParam=True,
            bsize:int=224,
            normalize_dict:NotGUIParam=None,            
        ):
        """Run cellpose 3.0 denoise model

        Parameters
        ----------
        image : numpy.ndarray
            (Y, X) or (Z, Y, X) or (C, Y, X) (Z, C, Y, X). If timelapse, the left most dim is expected to be time 
        diameter : float, optional
            Diameter of expected objects. If 0.0, cellpose will not try to estimate it (as opposed to the segmentation model)
            Will use 30 for everything except nuclei, which will use 17.0.
        do_3D : bool, optional
            If True, run the model in 3D mode if 3D image is provided.
            If False, run the model on a per slice basis.
        invert : bool, optional
            If True, invert the image pixel intensity before running the model.
            Default is False.
        normalize : bool, optional
            If True, normalize image using the other parameters.  Default is True
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
        rescale : float, optional
            Rescale image intensities to this value. Defaults to 1.0. Unless edge cases, should left to default None.
        tile_overlap : float, optional
            Fraction of overlap of tiles when computing flows. Defaults to 0.1.
        isZstack : bool, optional
            If True, the image is a z-stack. Default is False.
        timelapse : bool, optional
            If True, the image is a timelapse. Default is False.
        init_image : NotGUIParam, optional
            If provided, will init the image
        bsize : int, optional
            Default is 224.
            Dont change it unless you know what you are doing, please!
        normalize_dict : NotGUIParam, optional
            If provided, overrides all other normalization parameters.
        """
        if rescale == 1.0:
            rescale = None

        if diameter == 0:
            diameter = 30.0 if self.nstr != 'nuclei' else 17.0
    
        is_rgb = self.is_rgb

        if normalize_dict is None:
            normalize_params = _get_normalize_params(
                image,
                normalize=normalize, 
                rescale_intensity_low_val_perc=rescale_intensity_low_val_perc, 
                rescale_intensity_high_val_perc=rescale_intensity_high_val_perc, 
                # sharpen=sharpen,
                low_percentile=low_percentile, 
                high_percentile=high_percentile,
                norm3D=norm3D
            )
        else:
            normalize_params = normalize_dict
        
        normalize_dict['invert'] = invert

        eval_kwargs = {
            'diameter': diameter, 
            'normalize': normalize_params,
            'rescale': rescale,
            'tile_overlap': tile_overlap,
            'do_3D': do_3D,
            'bsize': bsize,
        }
        if self.batch_size is not None:
            eval_kwargs['batch_size'] = self.batch_size
        
        self.isZstack = isZstack
        self.denoise_slices_separately = not do_3D and isZstack
        if self.denoise_second_channel_together:
            eval_kwargs['channels'] = self.cellpose_rgb_channel
        elif self.denoise_second_channel_separately or self.ignore_second_channel or not self.is_rgb:
            eval_kwargs['channels'] = self.cellpose_greyscale_channel
        else:
            raise ValueError(
                f"Invalid channels configuration for denoising!"
            )

        iter_axis_zstack = None if not self.denoise_slices_separately else 0
        iter_axis_zstack = iter_axis_zstack + 1 if (timelapse and iter_axis_zstack is not None) else iter_axis_zstack
        if init_image:
            image, z_axis, channel_axis = _initialize_image(
                image, 
                isZstack=isZstack, 
                is_rgb=is_rgb, 
                timelapse=timelapse,
                iter_axis_zstack=iter_axis_zstack,
                iter_axis_time=0 if timelapse else None,
                add_rgb=False,
            )
        denoised_img = np.zeros_like(image)
         # add proper iterations, check wtf is going on wuit
        # (Z x nchan x Y x X)
        is_model_given_3D = (isZstack and not self.denoise_slices_separately)
        eval_kwargs['z_axis'] = 0 if is_model_given_3D else None
        eval_kwargs['channel_axis'] = 1 if is_model_given_3D else 0
        if self.denoise_second_channel_separately or not is_rgb or self.ignore_second_channel:
            eval_kwargs['channel_axis'] = None
        if timelapse:
            pbartime = tqdm(
                total=len(image), ncols=100, desc='Denoising time-lapse: '
            )
        else:
            pbartime = tqdm(
                total=1, ncols=100, desc='Denoising image: '
            )

        if timelapse:
            for t, img in enumerate(image):
                denoised_img = self._eval_image(
                    img, eval_kwargs, denoised_img, t=t
                )
                pbartime.update(1)
        else:
            denoised_img = self._eval_image(
                image, eval_kwargs, denoised_img
            )
            pbartime.update(1)
        
        pbartime.close()
        return denoised_img
    
    def _eval_image(self, image, eval_kwargs, entire_denoised_img, t=None):
        if t is not None:
            denoised_img = entire_denoised_img[t]
        else:
            denoised_img = entire_denoised_img
        # for NOT timelapse images helper funciton
        if self.denoise_slices_separately:
            if self.denoise_second_channel_separately: # dont need to move channel axis in output since I iterate over it and put it back correctly
                for z, c in tqdm(itertools.product(range(len(image)), self.first_second_channel), # only denoise channels which are requested
                                    desc=f'Denoising z-slicesand channels', ncols=100):
                    img = image[z][c]
                    img = self._acdc_eval(img, eval_kwargs)
                    img = np.squeeze(img) # remove channel axis if it was added
                    denoised_img[z, c] = img
            elif self.ignore_second_channel: # dont need to move channel axis in output since I iterate over it and put it back correctly
                for z, img_z in tqdm(enumerate(image), desc=f'Denoising z-slices: ', ncols=100):
                    img = img_z[self.first_second_channel[0]]
                    img = self._acdc_eval(img, eval_kwargs)
                    img = np.squeeze(img) # remove channel axis if it was added
                    denoised_img[z, self.first_second_channel[0]] = img
                # copy second channel as it is
                denoised_img[:, self.first_second_channel[1]] = image[:, self.first_second_channel[1]]
            else: # model either gets single gray or RGB image slices, channels was set correctly before
                for z, img_z in tqdm(enumerate(image), desc=f'Denoising z-slices: ', ncols=100):
                    img = self._acdc_eval(img_z, eval_kwargs) # oputputs rgb last...
                    if not self.is_rgb:
                        img = np.squeeze(img) # remove channel axis if it was added
                    else:
                        img = np.moveaxis(img, -1, 0) # move channel axis to the front
                        img = self._add_rgb_channels(img, isZstack=False) # add rgb channels if needed

                    denoised_img[z] = img
        else:
            if self.denoise_second_channel_separately: # dont need to move channel axis in output since I iterate over it and put it back correctly
                if self.isZstack:
                    image = np.moveaxis(image, 1, 0) # move channel axis to the front
                    denoised_img = np.moveaxis(denoised_img, 1, 0)
                for c in tqdm(self.first_second_channel, desc=f'Denoising channels: ', ncols=100):
                    img = self._acdc_eval(image[c], eval_kwargs)
                    denoised_img[c] = np.squeeze(img) # remove channel axis if it was added
                if self.isZstack:
                    denoised_img = np.moveaxis(denoised_img, 0, 1)
            elif self.ignore_second_channel: # dont need to move channel axis in output since I iterate over it and put it back correctly
                if self.isZstack:
                    image = np.moveaxis(image, 1, 0) # move channel axis to the front
                    denoised_img = np.moveaxis(denoised_img, 1, 0)
                img = self._acdc_eval(image[self.first_second_channel[0]], eval_kwargs)
                img = np.squeeze(img) # remove channel axis if it was added
                denoised_img[self.first_second_channel[0]] = img # remove channel axis if it was added

                # copy second channel as it is
                denoised_img[self.first_second_channel[1]] = image[self.first_second_channel[1]]
                if self.isZstack:
                    denoised_img = np.moveaxis(denoised_img, 0, 1) # move channel axis to the back
            else:
                denoised_img = self._acdc_eval(image, eval_kwargs) # pass entire iamge, with or without channels. Channels param set before
                if self.is_rgb:
                    if self.isZstack:
                        denoised_img = np.moveaxis(denoised_img, -1, 1) # move channel axis to the front after z
                    else:
                        denoised_img = np.moveaxis(denoised_img, -1, 0) # move channel axis to the front
                else:
                    denoised_img = np.squeeze(denoised_img) # remove channel axis if it was added
        
        # make sure that no channel is lost and if true, add it back # should not be needed, as entire_denoised_img has right shape

        denoised_img = self._add_rgb_channels(denoised_img)
        if t is not None:
            entire_denoised_img[t] = denoised_img
        else:
            entire_denoised_img = denoised_img

        return entire_denoised_img

    def _add_rgb_channels(self, denoised_img:np.ndarray, isZstack=None):
        if not self.is_rgb:
            return denoised_img

        if isZstack is None:
            isZstack = self.isZstack

        denoised_image_shape = denoised_img.shape
        if isZstack:
            channels = denoised_image_shape[1]
            if channels < 3:
                shape_to_concat = (denoised_image_shape[0], 3 - channels, denoised_image_shape[2], denoised_image_shape[3])
                # put it at position 0
                denoised_img = np.concatenate(
                    [denoised_img, np.zeros(shape_to_concat, dtype=denoised_img.dtype)],
                    axis=1
                )
        else:
            channels = denoised_image_shape[0]
            if channels < 3:
                shape_to_concat = (3 - channels, denoised_image_shape[1], denoised_image_shape[2])
                # put it at position 0
                denoised_img = np.concatenate(
                    [denoised_img, np.zeros(shape_to_concat, dtype=denoised_img.dtype)],
                    axis=0
                )
        return denoised_img

    def _acdc_eval(self, image, eval_kwargs):
        if not self.printed_model_params:     
            if isinstance(image, list):
                shape = image[0].shape
                shape = f"{len(image)} images of shape {shape}"
            else:
                shape = image.shape
            print("This is what is being passed to cellpose denoise:")
            print(f"Running denoise on image shape: {shape}, kwargs: {eval_kwargs}")
            if self.denoise_second_channel_together:
                for i, subarr in enumerate(np.moveaxis(image, -3, 0)):
                    print(f"Channel {i+1} min: {subarr.min()}, max: {subarr.max()}")
            else:
                print(f"Image min: {image.min()}, max: {image.max()}")
            self.printed_model_params = True
        return self.eval(image, **eval_kwargs)
         
        
def url_help():
    return 'https://www.biorxiv.org/content/10.1101/2024.02.10.579780v1'