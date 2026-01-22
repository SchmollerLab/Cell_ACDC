import os

import numpy as np
import skimage.measure
import torch

from . import model_types

from cellSAM import get_model, get_local_model, segment_cellular_image
from cellSAM.cellsam_pipeline import cellsam_pipeline, normalize_image
from cellSAM.wsi import segment_wsi

from cellacdc import myutils, printl


class AvailableModels:
    values = list(model_types.keys())


class NotParam:
    not_a_param = True


class Boolean:
    not_a_param = True


class Model:
    def __init__(
            self,
            model_type: AvailableModels='General',
            model_path: os.PathLike='',
            bbox_threshold: float=0.4,
            low_contrast_enhancement: bool=False,
            use_wsi: bool=True,
            gauge_cell_size: bool=False,
            block_size: int=400,
            overlap: int=56,
            iou_depth: int=56,
            iou_threshold: float=0.5,
            postprocess: bool=False,
            remove_boundaries: bool=False,
            gpu: bool=True
        ):
        """Initialization of CellSAM Model within Cell-ACDC

        CellSAM is a foundation model for cell segmentation that achieves
        state-of-the-art performance across a variety of cellular targets
        (bacteria, tissue, yeast, cell culture, etc.) and imaging modalities
        (brightfield, fluorescence, phase, electron microscopy, etc.).

        Parameters
        ----------
        model_type : str, optional
            CellSAM model type to use. Options are:
            - 'General': trained on datasets from the original publication
              (recommended for reproducibility)
            - 'Extra': incorporates additional datasets beyond the paper
              (recommended for out-of-distribution images)
            Default is 'General'.
        model_path : os.PathLike, optional
            Path to a custom CellSAM model file. If set, it will override
            `model_type`. Default is empty string.
        bbox_threshold : float, optional
            Threshold for bounding box confidence from CellFinder in [0, 1].
            This is the main parameter to control precision/recall. Use lower
            values (< 0.4) for out-of-distribution images and higher values
            for cleaner images. Default is 0.4
        low_contrast_enhancement : bool, optional
            Whether to enhance low contrast images as a preprocessing step.
            Useful for images like Livecell dataset. Default is False
        use_wsi : bool, optional
            Whether to use tiling (Whole Slide Image mode) to support large
            images. Generally, tiling is not required when there are fewer
            than ~3000 cells in an image. Default is True
        gauge_cell_size : bool, optional
            Whether to perform one iteration of segmentation initially to
            estimate cell sizes and then do another round with optimized
            tiling parameters. Only used if `use_wsi` is True. Default is False
        block_size : int, optional
            Size of tiles when `use_wsi` is True. Should be in range [256, 2048].
            Use smaller tile sizes for dense images (many cells/FOV).
            Default is 400
        overlap : int, optional
            Tile overlap region in which label merges are considered. Must be
            smaller than `block_size`. Should be large enough to encompass
            typical object extent. Default is 56
        iou_depth : int, optional
            Depth for IoU-based merging of overlapping objects. Default is 56
        iou_threshold : float, optional
            IoU threshold for merging overlapping objects in [0, 1].
            Default is 0.5
        postprocess : bool, optional
            If True, performs custom postprocessing on the segmentation mask.
            Recommended for noisy images. Default is False
        remove_boundaries : bool, optional
            If True, removes a one pixel boundary around segmented cells.
            Default is False
        gpu : bool, optional
            Whether to use GPU for inference (if available). Default is True
        """
        if gpu and torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.bbox_threshold = bbox_threshold
        self.low_contrast_enhancement = low_contrast_enhancement
        self.use_wsi = use_wsi
        self.gauge_cell_size = gauge_cell_size
        self.block_size = block_size
        self.overlap = overlap
        self.iou_depth = iou_depth
        self.iou_threshold = iou_threshold
        self.postprocess = postprocess
        self.remove_boundaries = remove_boundaries

        model_path = myutils.translateStrNone(model_path)[0]

        if model_path:
            print(f'Loading CellSAM model from {model_path}...')
            self.model = get_local_model(model_path)
        else:
            model_name = model_types[model_type]
            print(f'Loading CellSAM model "{model_name}"...')
            try:
                self.model = get_model(model=model_name)
            except Exception as e:
                error_msg = str(e).lower()
                if 'token' in error_msg or 'auth' in error_msg or '401' in error_msg:
                    raise RuntimeError(
                        f"Failed to download CellSAM model: {e}\n\n"
                        "Hint: CellSAM requires a DeepCell access token. "
                        "Please set the DEEPCELL_ACCESS_TOKEN environment variable.\n"
                        "You can obtain a token from https://deepcell.org"
                    ) from e
                raise

        self.model = self.model.to(self.device)
        self.model.bbox_threshold = bbox_threshold

        print(f'CellSAM model loaded successfully on {self.device}')

    def segment(
            self,
            image: np.ndarray,
            frame_i: int=0,
            automatic_removal_of_background: Boolean=False,
            posData: NotParam=None
        ) -> np.ndarray:
        """Segment image using CellSAM

        Parameters
        ----------
        image : ([Z], Y, X, [C]) numpy.ndarray
            Input image. It can be grayscale 2D (Y, X), or 3D (Z, Y, X) for
            z-stack data, or it can have additional dimension C for the RGB
            channels (3 channels) or multiplexed channels.

            For multiplexed images, the channel ordering should be:
            (blank, nuclear, whole-cell) where "whole-cell" comprises a spatial
            marker that delineates the cell boundary (e.g., membrane or
            cytoplasm marker). The whole-cell channel is optional.

        frame_i : int, optional
            Frame index (starting from 0). Currently not used by CellSAM but
            kept for API compatibility. Default is 0

        automatic_removal_of_background : bool, optional
            If True, the background object will be removed. The background
            object is defined as the largest object touching the borders of
            the image. Default is False

        posData : load.loadData or None, optional
            This is not a parameter configurable through the GUI. Cell-ACDC
            will pass the class of the loaded data from the specific Position.
            Currently not used by CellSAM.

        Returns
        -------
        ([Z], Y, X) numpy.ndarray of ints
            Output labelled masks with the same shape as input image but without
            the channel dimension. Every pixel belonging to the same object
            will have the same integer ID. ID = 0 is for the background.
        """
        is_rgb_image = image.shape[-1] == 3 or image.shape[-1] == 4
        is_z_stack = (image.ndim == 3 and not is_rgb_image) or (image.ndim == 4)

        if is_rgb_image:
            labels = np.zeros(image.shape[:-1], dtype=np.uint32)
        else:
            labels = np.zeros(image.shape, dtype=np.uint32)

        if is_z_stack:
            for z, img in enumerate(image):
                labels[z] = self._segment_2D_image(img)
            # Relabel connected components across z-stack
            labels = skimage.measure.label(labels > 0)
        else:
            labels = self._segment_2D_image(image)

        if automatic_removal_of_background:
            labels = self._remove_background_from_labels(labels)

        return labels

    def _segment_2D_image(self, image: np.ndarray) -> np.ndarray:
        """Segment a single 2D image using CellSAM.

        Parameters
        ----------
        image : (Y, X) or (Y, X, C) numpy.ndarray
            Input 2D image

        Returns
        -------
        (Y, X) numpy.ndarray of ints
            Segmentation mask
        """
        # Prepare image for CellSAM
        img = self._prepare_image(image)

        if self.use_wsi:
            # Use WSI pipeline for large images or dense cell populations
            import dask.array as da
            img_normalized = normalize_image(img.astype(np.float32))

            if self.low_contrast_enhancement:
                from cellSAM.utils import enhance_low_contrast
                img_normalized = enhance_low_contrast(img_normalized)

            inp = da.from_array(img_normalized, chunks=256)

            if self.gauge_cell_size:
                from cellSAM.cellsam_pipeline import use_cellsize_gaging
                labels = use_cellsize_gaging(
                    inp, self.model, self.device,
                    block_size=self.block_size,
                    overlap=self.overlap,
                    iou_depth=self.iou_depth,
                    iou_threshold=self.iou_threshold,
                    bbox_threshold=self.bbox_threshold
                )
            else:
                labels = segment_wsi(
                    inp,
                    self.block_size,
                    self.overlap,
                    self.iou_depth,
                    self.iou_threshold,
                    normalize=True,
                    model=self.model,
                    device=self.device,
                    bbox_threshold=self.bbox_threshold
                ).compute()
        else:
            # Direct segmentation for smaller images
            labels, _, _ = segment_cellular_image(
                img,
                model=self.model,
                normalize=True,
                postprocess=self.postprocess,
                remove_boundaries=self.remove_boundaries,
                bbox_threshold=self.bbox_threshold,
                device=self.device
            )

        return labels.astype(np.uint32)

    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Prepare image for CellSAM.

        CellSAM expects images in format (H, W, C) where C is 3.
        For grayscale images, we replicate to 3 channels.

        Parameters
        ----------
        image : numpy.ndarray
            Input image

        Returns
        -------
        numpy.ndarray
            Image formatted for CellSAM with shape (H, W, 3)
        """
        # Handle grayscale images
        if image.ndim == 2:
            # Replicate grayscale to 3 channels
            img = np.stack([image, image, image], axis=-1)
        elif image.ndim == 3:
            if image.shape[-1] == 1:
                # Single channel, replicate to 3
                img = np.repeat(image, 3, axis=-1)
            elif image.shape[-1] == 3:
                # Already RGB
                img = image
            elif image.shape[-1] == 4:
                # RGBA, drop alpha channel
                img = image[..., :3]
            else:
                # Multiple channels, use first 3 or pad
                if image.shape[-1] > 3:
                    img = image[..., :3]
                else:
                    # Pad with zeros
                    img = np.zeros((*image.shape[:-1], 3), dtype=image.dtype)
                    img[..., :image.shape[-1]] = image
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")

        return img

    def _remove_background_from_labels(self, labels: np.ndarray) -> np.ndarray:
        """Remove background from labeled array.

        The background is identified as the largest object touching the
        borders of the image.

        Parameters
        ----------
        labels : numpy.ndarray
            Labeled array

        Returns
        -------
        numpy.ndarray
            Labeled array with background removed
        """
        border_mask = np.ones(labels.shape, dtype=bool)
        border_slice = tuple([slice(2, -2) for _ in range(labels.ndim)])
        border_mask[border_slice] = False

        border_ids, counts = np.unique(labels[border_mask], return_counts=True)
        # Exclude background (0) from consideration
        valid_mask = border_ids != 0
        if not np.any(valid_mask):
            return labels

        border_ids = border_ids[valid_mask]
        counts = counts[valid_mask]

        if len(counts) == 0:
            return labels

        max_count_idx = np.argmax(counts)
        largest_border_id = border_ids[max_count_idx]
        labels[labels == largest_border_id] = 0

        return labels


def url_help():
    return 'https://github.com/vanvalenlab/cellSAM'
