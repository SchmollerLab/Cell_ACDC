"""
This module contains the functions that can be used as pre-processing steps 
before segmentation. 

These functions are automatically added to `apps.QDialogModelParams` and they 
can be selected in the pre-processing recipe. 

Every function must have a single argument for the image, while all 
other parameters must be keyword arguments. 

Functions that should not be used as pre-processing steps must start with `_`. 
The list of functions is generated in the module `cellacdc.config` 
(see PREPROCESS_MAPPER variable).

IMPORTANT: Do not import functions otherwise they will be added as possible 
step (for example do not do `from skimage.util import img_as_ubyte`).
"""
from typing import Hashable, Union, Optional, Tuple
from tqdm import tqdm

import numpy as np
import pandas as pd
import math

try:
    import cupyx.scipy.ndimage
    import cupy as cp
    CUPY_INSTALLED = True
except Exception as e:
    CUPY_INSTALLED = False

import skimage.morphology
import skimage.filters
import skimage.exposure
import skimage.util

from . import error_up_str
from . import types
from . import printl

SQRT_2 = math.sqrt(2)

def remove_hot_pixels(
        image, 
        logger_func=print, 
        progress=True, 
        apply_to_all_zslices=True
    ):
    """Apply a morphological opening operation to remove isolated bright 
    pixels.

    Parameters
    ----------
    image : (Y, X) or (Z, Y, X) numpy.ndarray
        Input image
    logger_func : callable, optional
        Function used to log information. Default is print
    progress : bool, optional
        If `True`, displays progress bar. Default is True

    Returns
    -------
    (Y, X) or (Z, Y, X) numpy.ndarray
        Filtered image
    """    
    is_3D = image.ndim == 3
    if is_3D:
        if progress:
            pbar = tqdm(total=len(image), ncols=100)
        filtered = image.copy()
        for z, img in enumerate(image):
            filtered[z] = skimage.morphology.opening(img)
            if progress:
                pbar.update()
        if progress:
            pbar.close()
    else:
        filtered[z] = skimage.morphology.opening(img)
    return filtered

def gaussian_filter(
        image, 
        sigma: types.Vector=0.75, 
        use_gpu=False, 
        logger_func=print, 
        apply_to_all_zslices=True
    ):
    """Multi-dimensional Gaussian filter

    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale or color) to filter.
    sigma : types.Vector
        Standard deviation for Gaussian kernel. The standard deviations of the 
        Gaussian filter are given for each axis as a sequence, or as a single 
        number, in which case it is equal for all axes.
    use_gpu : bool, optional
        If True, uses `cupy` instead of `skimage.filters.gaussian`. 
        Default is False
    logger_func : callable, optional
        Function used to log information. Default is print

    Returns
    -------
    filtered_image : numpy.ndarray
        The filtered image

    See also
    --------
    Wikipedia link: `Gaussian blur <https://en.wikipedia.org/wiki/Gaussian_blur>`_
    """    
    try:
        if len(sigma) > 1 and sigma[0] == 0:
            return image
    except Exception as err:
        pass
    
    try:
        if sigma == 0:
            return image
    except Exception as err:
        pass
    
    try:
        if len(sigma) == 0:
            sigma = sigma[0]
    except Exception as err:
        pass
    
    if CUPY_INSTALLED and use_gpu:
        try:
            image = cp.array(image, dtype=float)
            filtered = cupyx.scipy.ndimage.gaussian_filter(image, sigma)
            filtered = cp.asnumpy(filtered)
        except Exception as err:
            logger_func('*'*100)
            logger_func(err)
            logger_func(
                '[WARNING]: GPU acceleration of the gaussian filter failed. '
                f'Using CPU...{error_up_str}'
            )
            filtered = skimage.filters.gaussian(image, sigma=sigma)
    else:
        filtered = skimage.filters.gaussian(image, sigma=sigma)
    return filtered

def ridge_filter(
        image, 
        sigmas: types.Vector=(1.0, 2.0), 
        apply_to_all_zslices=True
    ):
    """Filter used to enhance network-like structures (Sato filter). More info 
    here https://scikit-image.org/docs/stable/auto_examples/edges/plot_ridge_filter.html

    Parameters
    ----------
    image : (Y, X) or (Z, Y, X) numpy.ndarray
        Input image
    sigmas : sequence of floats, optional
        Sigmas used for the ridge filter. Default is (1.0, 2.0)

    Returns
    -------
    (Y, X) or (Z, Y, X) numpy.ndarray
        Filtered image
    """    
    input_shape = image.shape
    filtered = skimage.filters.sato(
        np.squeeze(image), sigmas=sigmas, black_ridges=False
    ).reshape(input_shape)
    return filtered

def spot_detector_filter(
        image, 
        spots_zyx_radii_pxl: types.Vector=(3, 5, 5), 
        use_gpu=False, 
        logger_func=print, 
        apply_to_all_zslices=True
    ):
    """Spot detection using Difference of Gaussians filter.

    Parameters
    ----------
    image : (Y, X) or (Z, Y, X) numpy.ndarray
        Input image
    spots_zyx_radii_pxl : sequence of floats, one for each dimension, optional
        Expected size of the spots in pixels. One size for each dimension in 
        `image`. Default is (3, 5, 5)
    use_gpu : bool, optional
        If `True` uses GPU if `cupy` is installed and a CUDA-compatible GPU 
        is available . Default is False
    logger_func : callable, optional
        Function used to log additional information on progress. Default is print

    Returns
    -------
    (Y, X) or (Z, Y, X) numpy.ndarray
        Filtered image

    Raises
    ------
    TypeError
        Error raised when on of the input sigmas is zero.
    """    
    spots_zyx_radii_pxl = np.array(spots_zyx_radii_pxl)
    if image.ndim == 2 and len(spots_zyx_radii_pxl) == 3:
        spots_zyx_radii_pxl = spots_zyx_radii_pxl[1:]
    
    sigma1 = spots_zyx_radii_pxl/(1+SQRT_2)
    
    if 0 in sigma1:
        raise TypeError(
            f'Sharpening filter input sigmas cannot be 0. `zyx_sigma1 = {sigma1}`'
        )
        
    blurred1 = gaussian_filter(
        image, sigma1, use_gpu=use_gpu, logger_func=logger_func
    )
    
    sigma2 = SQRT_2*sigma1
    blurred2 = gaussian_filter(
        image, sigma2, use_gpu=use_gpu, logger_func=logger_func
    )
    
    sharpened = blurred1 - blurred2
    
    out_range = (image.min(), image.max())
    in_range = 'image'
    sharp_rescaled = skimage.exposure.rescale_intensity(
        sharpened, in_range=in_range, out_range=out_range
    )
    
    return sharp_rescaled

def correct_illumination(
        image, 
        block_size=45, 
        # rescale_illumination=True,
        approximate_object_diameter=15,
        # background_threshold=0.3,
        apply_gaussian_filter=True
    ):
    """
    Correct illumination of an image. Based on CellProfiler's illumination correction.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image to correct.
    block_size : int, optional
        Block size for background illumination estimation. Default is 60.
    # rescale_illumination : bool, optional
    #     Whether to rescale the illumination function to range [0, 1]. Default is True.
    approximate_object_diameter : int, optional
        Approximate object diameter for Gaussian filtering. Default is 15.
    # background_threshold : float, optional
    #     Relative threshold to define the background of the illumination function. Default is 0.3.
    apply_gaussian_filter : bool, optional
        Whether to apply Gaussian smoothing to the illumination function. Default is True.

    Returns
    -------
    np.ndarray
        Corrected 2D image.
    """
    image = skimage.img_as_float(image)
    imgcopy = image.copy()

    if apply_gaussian_filter:
        image = skimage.filters.gaussian(image, sigma=approximate_object_diameter / 2)

    background_illu = skimage.restoration.rolling_ball(image, radius=block_size)
    corrected_image = imgcopy - background_illu

    # footprint = skimage.morphology.rectangle(block_size, block_size)
    # illumination_function = skimage.filters.rank.mean(image, footprint=footprint)

    # # Gaussian smoothing
    # if apply_gaussian_filter:
    #     illumination_function = skimage.filters.gaussian(
    #         illumination_function, sigma=approximate_object_diameter / 2
    #     )

    # # Apply correction
    # illumination_function[illumination_function == 0] = 1
    # corrected_image = image / illumination_function

    # corrected_image = skimage.exposure.rescale_intensity(corrected_image, out_range=(0, 1))
    # corrected_image = skimage.img_as_ubyte(corrected_image)

    return corrected_image

def enhance_speckles(img, radius=15):
    """Enhance speckles in an image using white_tophat. Based on 
    EnhanceOrSuppressFeatures from Cell profiler with 'Feature type: Speckles'

    Parameters
    ----------
    image : np.ndarray
        2D image to enhance
    radius : int, optional
        Radius to use for the enhancer. Will suppress objects smaller than this 
        radius. Default is 15

    Returns
    -------
    np.ndarray
        corrected 2D image
    """
    footprint = skimage.morphology.disk(radius)
    output_image = skimage.morphology.white_tophat(img, footprint=footprint)
    return output_image

def fucci_filter(
        image,
        correct_illumination_toggle=False,
        basicpy_background_correction_toggle=True,
        enhance_speckles_toggle=True,
        block_size=120,
        # rescale_illumination=False,
        approximate_object_diameter=25,
        # background_threshold=0.3,
        apply_gaussian_filter=True,
        speckle_radius=25
    ):
    """Basic filter pipeline proposed for Fucci images.
    If you want custom pipelines and more in depth control, create
    your own recipe using the GUI or segmentation and tracking modules.

    Parameters
    ----------
    image : (Y, X) numpy.ndarray
        2D image to correct
    correct_illumination_toggle : bool, optional
        If illumination should be corrected. 
        Default is True
    basicpy_background_correction_toggle : bool, optional
        If BaSiC background correction should be applied. 
        Default is False
    enhance_speckles_toggle : bool, optional
        If speckles should be enhanced. 
        Default is True
    block_size : int, optional
        Block size for which to calculate the background illumination.
        Default is 45
    # rescale_illumination : bool, optional
    #     if illumination should be rescaled with skimage.exposure.rescale_intensity range=(0, 1).
    #     Default is True
    approximate_object_diameter : int, optional
        Approximate object diameter for gaussian_filter. 
        Default is 25
    # background_threshold : float, optional
    #     Threshold to be used to determine the background. 
    #     Default is 0.3
    apply_gaussian_filter : bool, optional
        If gaussian_filter should be applied to the illumination_function. 
        Default is True
    speckle_radius : int, optional
        Radius to use for the enhancer. Will suppress objects smaller than this 
        radius. Default is 25

    Returns
    -------
    (Y, X) numpy.ndarray
        Filtered image
    """
    if basicpy_background_correction_toggle:
        images = basicpy_background_correction(
            images, 
            apply_to_all_frames=False,
            apply_to_all_zslices=False,
        )
    if correct_illumination_toggle:
        image = correct_illumination(
            image, 
            block_size=block_size, 
            # rescale_illumination=rescale_illumination,
            approximate_object_diameter=approximate_object_diameter,
            # background_threshold=background_threshold,
            apply_gaussian_filter=apply_gaussian_filter,
        )
    if enhance_speckles_toggle:
        image = enhance_speckles(image, radius=speckle_radius)
    return image

def dummy_filter(
        image: np.ndarray, 
        apply_to_all_zslices=False, 
        apply_to_all_frames=False
    ):
    printl(image.shape)
    return image

class VolumeImageData:
    def __init__(self):
        self._data = {}

    def __setitem__(
            self, 
            z_slice: int, 
            image: np.ndarray
        ):
        if not isinstance(z_slice, (int, str)):
            raise TypeError(
                f'{z_slice} is not not a valid index. '
                f'It must be an integer or a string and not {type(z_slice)}'
            )
        
        if image.ndim != 2:
            raise TypeError(
                'Only 2D images can be assigned to a specifc z-slice index.'
            )
        
        self._data[z_slice] = image
    
    def __getitem__(
            self, z_slice: Union[int, Tuple[Union[int, slice]], None]
        ):
        if isinstance(z_slice, int):
            return self._data[z_slice]
        
        arr = self._build_arr()
        return arr[z_slice]
    
    def __array__(self) -> np.ndarray:
        return self._build_arr()
    
    def __repr__(self):
        return str(self._data)
    
    def _build_arr(self):
        if not self._data:
            return
        
        img = self._data[0]
        SizeZ = len(self._data)
        arr = np.zeros((SizeZ, *img.shape), dtype=img.dtype)
        for z_slice, img in self._data.items():
            arr[z_slice] = img
        return np.squeeze(arr)
    
    def max(self, axis=None):
        arr = self._build_arr()
        if arr is None:
            return
        
        return arr.max(axis=axis)

    def min(self, axis=None):
        arr = self._build_arr()
        if arr is None:
            return
        
        return arr.min(axis=axis)
    
    def mean(self, axis=None):
        arr = self._build_arr()
        if arr is None:
            return
        
        return arr.mean(axis=axis)
    
class PreprocessedData:
    def __init__(self):
        self._data = {}
    
    def __getitem__(self, frame_i: int):
        if frame_i not in self._data:
            self._data[frame_i] = VolumeImageData()
            
        return self._data[frame_i]
    
    def __setitem__(self, frame_i: int, image: np.ndarray):
        if not isinstance(frame_i, int):
            raise TypeError(
                f'{frame_i} is not not a valid index. '
                f'It must be an integer and not {type(frame_i)}'
            )
            
        if frame_i not in self._data:
            self._data[frame_i] = VolumeImageData()
        
        if image.ndim == 2:
            self._data[frame_i][0] = image
        else:
            for z_slice, img in enumerate(image):
                self._data[frame_i][z_slice] = image
    
    def __repr__(self):
        return str(self._data)
    
    def get(self, frame_i: int, default_value=None):
        try:
            return self._data[frame_i]
        except KeyError:
            return default_value

def rescale_intensities(
        image: np.array,
        out_range_low: float=0.0,
        out_range_high: float=1.0,
        in_range_how: types.RescaleIntensitiesInRangeHow='percentage',
        in_range_low: float=0.0,
        in_range_high: float=1.0,
        apply_to_all_zslices=True,
    ):
    """Rescale the intensities of an image to a given range.

    Parameters
    ----------
    image : np.ndarray
        Input image to rescale
    out_range_low : float, optional
        Min value of the output image. Default is 0.0
    out_range_high : float, optional
        Max value of the output image. Default is 1.0
    in_range_low : float, optional
        Min value of the output image. See `in_range_how` for more details. 
        Default is 0.0
    in_range_high : float, optional
        Max value of the output image. See `in_range_how` for more details. 
        Default is 1.0
    in_range_how : {'percentage', 'image', 'absolute'}, optional
        If `percentage`, the image is first rescaled to (0, 1) using the 
        minimum and maximum value of the input image. This allows to specify 
        the input range as a percentage of the image intensity range. 
        If `image`, the input range is the minimum and maximum value of the 
        input image. 
        If `absolute`, the input range is specified by `in_range_low` and 
        `in_range_high` in absolute values (same scale as the input image). 
        Default is 'percentage'.
    apply_to_all_zslices : bool, optional
        Scale intensities across multi-dimensional images. Default is True

    Returns
    -------
    np.ndarray
        The rescaled image
    """
    out_range = (out_range_low, out_range_high)
    if in_range_how == 'image':
        in_range = 'image'
    elif in_range_how == 'percentage':
        image = skimage.exposure.rescale_intensity(
            image, in_range='image', out_range=(0, 1)
        )
    elif in_range_how == 'absolute':
        in_range = (in_range_low, in_range_high)
        
    rescaled = skimage.exposure.rescale_intensity(
        image, in_range=in_range, out_range=out_range
    )
    return rescaled

def _init_dummy_filter(**kwargs):
    """
    This function runs automatically as part of the preprocessing recipe if 
    the user selects the 'dummy_filter' step. The 'dummy_filter' is available 
    only in debug mode. Initialization functions run in the main GUI thread 
    and they can be used to set up the related function, for example to 
    prompt the user that a package needs to be installed.
    """
    pass

def basicpy_background_correction(
        images,
        apply_to_all_frames=True,
        apply_to_all_zslices=True,
        smoothness_flatfield=1.0,
        get_darkfield=True,
        smoothness_darkfield=1.0,
        sparse_cost_darkfield=0.01,
        # baseline=None,
        # darkfield=None,
        fitting_mode: types.BaSiCpyFittingModes="ladmap",
        epsilon=0.1,
        # flatfield=None,
        autosegment=False,
        autosegment_margin=10,
        max_iterations=500,
        max_reweight_iterations=10,
        max_reweight_iterations_baseline=5,
        # max_workers=2,
        rho=1.5,
        mu_coef=12.5,
        max_mu_coef=10000000.0,
        optimization_tol=0.001,
        optimization_tol_diff=0.01,
        resize_mode: types.BaSiCpyResizeModes="jax",
        resize_params: types.NotGUIParam=None,
        reweighting_tol=0.01,
        sort_intensity=False,
        working_size=128,
        timelapse: types.BaSiCpyTimelapse="True",
        parent: types.NotGUIParam=None
    ):
    """
    A function for fitting and applying BaSiC illumination correction profiles.

    Parameters
    ----------
    images : (T, Z, Y, X) numpy.ndarray
        Image. Make sure to set have (T, Z, Y, X) dimensions, 
        or missing dimensions
        in accordance with the `apply_to_all_frames` and 
        `apply_to_all_zslices` parameters.
    apply_to_all_frames : bool, default=True
        Whether to apply the correction to all frames. 
        If set to falce, assumes that the image has 
        no T dimension, so either (Z, Y, X) or (Y, X).
    apply_to_all_zslices : bool, default=True
        Whether to apply the correction to all Z slices. 
        If set to falce, assumes that the image has 
        no Z dimension, so either (T, Y, X) or (Y, X).
    smoothness_flatfield : float, default=1.0
        Weight of the flatfield term in the Lagrangian.
    get_darkfield : bool, default=True
        Whether to estimate the darkfield shading component.
    smoothness_darkfield : float, default=1.0
        Weight of the darkfield term in the Lagrangian.
    sparse_cost_darkfield : float, default=0.01
        Weight of the darkfield sparse term in the Lagrangian.
    # baseline : object, optional
    #     Baseline correction profile.
    # darkfield : object, optional
    #     Darkfield correction profile.
    fitting_mode : str, default="ladmap"
        Fit method. Must be one of ['ladmap', 'approximate'].
    epsilon : float, default=0.1
        Weight regularization term.
    # flatfield : object, optional
    #     Flatfield correction profile.
    autosegment : bool or callable, default=False
        When not False, automatically segment the image before fitting.
        When True, `threshold_otsu` from `scikit-image` is used
        and the brighter pixels are taken.
    autosegment_margin : int, default=10
        Margin of the segmentation mask to the thresholded region.
    max_iterations : int, default=500
        Maximum number of iterations for single optimization.
    max_reweight_iterations : int, default=10
        Maximum number of reweighting iterations.
    max_reweight_iterations_baseline : int, default=5
        Maximum number of reweighting iterations for baseline.
    # max_workers : int, default=2
    #     Maximum number of threads used for processing.
    rho : float, default=1.5
        Parameter rho for mu update.
    mu_coef : float, default=12.5
        Coefficient for initial mu value.
    max_mu_coef : float, default=10000000.0
        Maximum allowed value of mu, divided by the initial value.
    optimization_tol : float, default=0.001
        Optimization tolerance.
    optimization_tol_diff : float, default=0.01
        Optimization tolerance for update difference.
    resize_mode : str, default="jax"
        Resize mode for downsampling images. Must be one of 
        ['jax', 'skimage', 'skimage_dask'].
    resize_params : dict, default={}
        Parameters for the resize function.
    reweighting_tol : float, default=0.01
        Reweighting tolerance in mean absolute difference of images.
    sort_intensity : bool, default=False
        Whether to sort the intensities of the image.
    working_size : int or list of int, default=128
        Size for running computations. None means no rescaling.
    timelapse : str, default="False"
        If `True`, corrects the timelapse/photobleaching offsets,
        assuming that the residual is the product of flatfield and
        the object fluorescence. Also accepts "multiplicative"
        (the same as `True`) or "additive" (residual is the object
        fluorescence).
    parent : QWidget, optional
        Parent widget for the GUI.

    Returns
    -------
    None
        This function does not return any value.
    """

    if resize_params is None:
        resize_params = {}
    if timelapse == "True":
        timelapse = True
    elif timelapse == "False":
        timelapse = False

    images = skimage.img_as_float(images)

    from . import transformation

    if not apply_to_all_frames and not apply_to_all_zslices:
        input_dims = ("Y", "X")
    elif apply_to_all_frames and not apply_to_all_zslices:
        input_dims = ("T", "Y", "X")
    elif not apply_to_all_frames and apply_to_all_zslices:
        input_dims = ("Z", "Y", "X")
    else:
        input_dims = ("T", "Z", "Y", "X")
    
    images = transformation.correct_img_dimension(images, 
                                               input_dims=input_dims, 
                                               output_dims=("T", "Z", "Y", "X"))

    from . import myutils
    custom_install_requires = [
        "hyperactive>=4.4.0",
        "jax>=0.4.0,<0.5.0",
        "jaxlib>=0.4.0,<0.5.0",
        "numpy",
        "pooch",
        "pydantic>=2.7.0,<3.0.0",
        "scikit-image",
        "scipy", # this will theoretically have the wrong version of scipy in the end
        ]
    myutils.check_install_custom_dependencies(custom_install_requires, 
                                              'basicpy', 
                                              parent=parent)
    from basicpy import BaSiC

    basic = BaSiC(
        # baseline=baseline,
        # darkfield=darkfield,
        fitting_mode=fitting_mode,
        epsilon=epsilon,
        # flatfield=flatfield,
        get_darkfield=get_darkfield,
        smoothness_flatfield=smoothness_flatfield,
        smoothness_darkfield=smoothness_darkfield,
        sparse_cost_darkfield=sparse_cost_darkfield,
        autosegment=autosegment,
        autosegment_margin=autosegment_margin,
        max_iterations=max_iterations,
        max_reweight_iterations=max_reweight_iterations,
        max_reweight_iterations_baseline=max_reweight_iterations_baseline,
        # max_workers=max_workers,
        rho=rho,
        mu_coef=mu_coef,
        max_mu_coef=max_mu_coef,
        optimization_tol=optimization_tol,
        optimization_tol_diff=optimization_tol_diff,
        resize_mode=resize_mode,
        resize_params=resize_params,
        reweighting_tol=reweighting_tol,
        sort_intensity=sort_intensity,
        working_size=working_size
        )

    basic.fit(images)
    images = basic.transform(
        images,
        timelapse=timelapse
        )
    
    images = images.squeeze()
    return images