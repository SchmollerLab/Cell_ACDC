"""ConvPaint (napari-convpaint) as a Cell-ACDC promptable model.

ConvPaint trains a pixel classifier (CatBoost) on the features that a neural
network (e.g., VGG16) extracts from the pixels annotated with brush strokes
(0 = not annotated, 1 = background, 2 = foreground). The resulting semantic
segmentation is converted to instance labels with `skimage.measure.label`
on the foreground class.

Typical sequence of calls (made by the GUI):

1. `add_prompt(annotation, image=image, prompt_type='brush_stroke',
   frame_i=frame_i, pos_i=pos_i)` for every annotated frame.
2. `train()` trains a new classifier on all the added annotations.
3. `segment(image)` as many times as needed (the classifier is kept).

A trained model can be saved with `save_model(filepath)` and loaded with
`load_model(filepath)`. The files are the ones of napari-convpaint (a `.pkl`
file with the classifier and the parameters, plus a `.yml` file with the
parameters only), so models trained in napari can be loaded too.

Only the model part of napari-convpaint is used. Its package is imported
without running `napari_convpaint/__init__.py`, which imports napari and Qt
(see `import_convpaint_model`).
"""
import contextlib
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import threading
import warnings

import numpy as np
import skimage.measure
import skimage.segmentation

BACKGROUND_CLASS = 1
FOREGROUND_CLASS = 2

class FeatureExtractor:
    values = ['vgg', 'vgg-m', 'vgg-l', 'gaussian', 'dinov2', 'dinov3']

class Normalization:
    values = ['Whole image (z-stack)', 'Each z-slice', 'None']

class Device:
    values = ['auto', 'cpu', 'gpu']

# ConvPaint `normalize` parameter matching each `Normalization` option
_CONVPAINT_NORMALIZE = {
    'None': 1, 'Whole image (z-stack)': 2, 'Each z-slice': 3
}
_NORMALIZATION_FROM_CONVPAINT = {
    value: key for key, value in _CONVPAINT_NORMALIZE.items()
}

def _strip_model_ext(filepath):
    filepath = os.fspath(filepath)
    if filepath[-4:].lower() in ('.pkl', '.yml'):
        filepath = filepath[:-4]
    return filepath

# ConvPaint parameters that define the feature extractor
_FE_PARAMS = (
    'fe_name', 'fe_layers', 'fe_scalings', 'fe_order', 'fe_use_min_features'
)

def _as_comparable(value):
    if isinstance(value, (list, tuple)):
        return [_as_comparable(item) for item in value]
    return value

# Attributes of `ConvpaintModel` changed by training (restored on errors)
_CLASSIFIER_STATE = ('classifier', 'num_features', 'num_trainings')

def import_convpaint_model():
    """Import `ConvpaintModel` without running `napari_convpaint/__init__.py`.

    The package `__init__.py` imports the napari widget (napari and Qt),
    which the model does not need. An empty package module is registered in
    `sys.modules` instead, so that the submodules (and their relative
    imports) are still loaded from the installed package folder.
    """
    os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    registered = False
    if 'napari_convpaint' not in sys.modules:
        spec = importlib.util.find_spec('napari_convpaint')
        if spec is None:
            raise ImportError(
                'napari-convpaint is not installed. Install it with '
                '`pip install --no-deps "napari-convpaint>=1.2"` (napari is '
                'not needed by Cell-ACDC).'
            )
        sys.modules['napari_convpaint'] = importlib.util.module_from_spec(
            spec
        )
        registered = True
    try:
        from napari_convpaint.convpaint_model import ConvpaintModel
    except Exception:
        if registered:
            sys.modules.pop('napari_convpaint', None)
        raise
    return ConvpaintModel

class Model:
    prompt_types = ('brush_stroke',)
    # Extension of the files written by `save_model` and read by `load_model`
    saved_model_ext = '.pkl'

    def __init__(
            self,
            feature_extractor: FeatureExtractor = 'vgg',
            normalization: Normalization = 'Whole image (z-stack)',
            image_downsample: int = 1,
            tile_image: bool = False,
            classifier_iterations: int = 100,
            classifier_learning_rate: float = 0.1,
            classifier_depth: int = 5,
            device: Device = 'auto',
            train_classifier_on_gpu: bool = False,
            verbose: bool = False,
            **kwargs
        ):
        """Initialize ConvPaint (feature extractor and pixel classifier)

        Parameters
        ----------
        feature_extractor : FeatureExtractor, optional
            Neural network used to extract the features of each pixel.
            'vgg' uses the first layer of VGG16, 'vgg-m' and 'vgg-l' use
            more VGG16 layers (more context, slower), 'gaussian' uses
            Gaussian filters (very fast, no download), 'dinov2' and
            'dinov3' are vision transformers (more context, slow on CPU).
            The weights are downloaded the first time a feature extractor
            is used. Default is 'vgg'
        normalization : Normalization, optional
            How the image intensities are normalized (z-score) before
            training and segmentation. 'Whole image (z-stack)' uses the
            mean and standard deviation of the whole image (or z-stack),
            'Each z-slice' those of each z-slice, and 'None' uses the raw
            intensities. Default is 'Whole image (z-stack)'
        image_downsample : int, optional
            Factor by which the image is downsampled before extracting the
            features (faster and less memory). Negative values upsample
            the image instead (e.g., for very small objects).
            Default is 1 (no downsampling)
        tile_image : bool, optional
            If True, large images are segmented in tiles to reduce the
            memory usage. Default is False
        classifier_iterations : int, optional
            Number of iterations (trees) of the CatBoost pixel classifier.
            Default is 100
        classifier_learning_rate : float, optional
            Learning rate of the CatBoost pixel classifier. Default is 0.1
        classifier_depth : int, optional
            Depth of the trees of the CatBoost pixel classifier.
            Default is 5
        device : Device, optional
            Device used to extract the features. 'auto' uses the GPU if
            available (CUDA or Apple MPS), otherwise the CPU.
            Default is 'auto'
        train_classifier_on_gpu : bool, optional
            If True, the CatBoost classifier is trained on the GPU when
            CUDA is available. Brush strokes are usually few pixels, so
            training on the CPU is often as fast and more robust.
            Default is False
        verbose : bool, optional
            If True, the output and the warnings of ConvPaint (e.g., the
            classifier training iterations) are displayed in the terminal.
            Default is False
        """
        if feature_extractor not in FeatureExtractor.values:
            raise ValueError(
                f'Invalid feature extractor "{feature_extractor}". '
                f'Valid options are {FeatureExtractor.values}'
            )
        if normalization not in _CONVPAINT_NORMALIZE:
            raise ValueError(
                f'Invalid normalization "{normalization}". '
                f'Valid options are {Normalization.values}'
            )
        if device not in Device.values:
            raise ValueError(
                f'Invalid device "{device}". Valid options are {Device.values}'
            )
        image_downsample = int(image_downsample)
        if not -20 <= image_downsample <= 20:
            raise ValueError(
                'The image downsampling factor must be between -20 and 20, '
                f'not {image_downsample}.'
            )

        self.verbose = verbose
        self.normalization = normalization
        self.image_downsample = image_downsample
        # Device policies are not saved with the model --> re-applied to
        # the models loaded with `load_model`
        self.device = device
        self.train_classifier_on_gpu = bool(train_classifier_on_gpu)
        # ConvPaint is not re-entrant (it stores intermediate shapes on the
        # instance) and `_quiet` swaps the process-wide stdout --> serialize
        # training and segmentation (e.g., called from different threads)
        self._lock = threading.RLock()

        ConvpaintModel = import_convpaint_model()
        # Keep stdout: it shows the download progress of the weights
        with self._quiet(stdout=False):
            self.model = ConvpaintModel(alias=feature_extractor)
            self.model.set_params(
                ignore_warnings=True,
                image_downsample=image_downsample,
                tile_image=bool(tile_image),
                clf_iterations=int(classifier_iterations),
                clf_learning_rate=float(classifier_learning_rate),
                clf_depth=int(classifier_depth),
                normalize=_CONVPAINT_NORMALIZE[normalization],
            )
        self._lock_devices(self.model)

        # {(pos_i, frame_i): (image, annotation)}, cleared by `train`
        self.prompts = {}
        self.is_trained = False

    def _lock_devices(self, convpaint_model):
        convpaint_model.lock_device(self.device, part='fe')
        convpaint_model.lock_device(
            'auto' if self.train_classifier_on_gpu else 'cpu', part='clf'
        )

    @contextlib.contextmanager
    def _quiet(self, stdout=True):
        """Hide ConvPaint's output (e.g., one line per CatBoost iteration)
        and warnings, unless `verbose` is True."""
        if self.verbose:
            yield
            return

        with contextlib.ExitStack() as stack:
            if stdout:
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(warnings.catch_warnings())
            warnings.simplefilter('ignore')
            yield

    def _normalize(self, image):
        """Return `image` as float32, z-scored according to `normalization`.

        ConvPaint is always called with `skip_norm=True`, so that training
        and segmentation are normalized the same way and z-stacks can be
        segmented z-slice by z-slice.

        Feature extractors that require percentile normalization (e.g., the
        Cellpose feature extractor of models trained in napari) get it like
        in ConvPaint: the 1st and 99th percentiles are rescaled to 0 and 1.
        """
        image = np.asarray(image, dtype=np.float32)
        if self.normalization == 'None':
            return image

        if self.normalization == 'Each z-slice' and image.ndim == 3:
            axis = (1, 2)
        else:
            axis = None

        fe_norm_mode = getattr(
            getattr(self.model, 'fe_model', None), 'norm_mode', None
        )
        if fe_norm_mode == 'percentile':
            low, high = np.percentile(image, (1, 99), axis=axis, keepdims=True)
            scale = np.maximum(high - low, 1e-6)
            image = (image - low.astype(np.float32)) / scale.astype(np.float32)
            return np.clip(image, 0, 1, out=image)

        mean = image.mean(axis=axis, keepdims=True, dtype=np.float64)
        std = image.std(axis=axis, keepdims=True, dtype=np.float64)
        std = np.maximum(std, 1e-6)
        return (image - mean.astype(np.float32)) / std.astype(np.float32)

    @staticmethod
    def _validate_annotation(annot, image):
        if annot.ndim not in (2, 3):
            raise ValueError(
                'Brush-stroke annotations must be 2D (Y, X) or 3D (Z, Y, X), '
                f'not with shape {annot.shape}.'
            )

        if image.ndim != annot.ndim:
            raise ValueError(
                f'The image has shape {image.shape}, while the brush strokes '
                f'have shape {annot.shape}. ConvPaint requires single-channel '
                'images with the same dimensions as the brush strokes (RGB '
                'or multi-channel images are not supported).'
            )

        if image.shape != annot.shape:
            raise ValueError(
                f'The image shape {image.shape} is different from the '
                f'brush-strokes shape {annot.shape}.'
            )

        if annot.size == 0:
            return

        is_integer = (
            np.issubdtype(annot.dtype, np.integer) or annot.dtype == bool
        )
        is_valid = (
            annot.min() >= 0 and annot.max() <= FOREGROUND_CLASS
            and (is_integer or np.all(np.mod(annot, 1) == 0))
        )
        if is_valid:
            return

        values = np.unique(annot)
        valid_values = (0, BACKGROUND_CLASS, FOREGROUND_CLASS)
        invalid = values[~np.isin(values, valid_values)]
        raise ValueError(
            f'Invalid brush-stroke values {invalid.tolist()}. Valid values '
            'are 0 (not annotated), 1 (background) and 2 (foreground).'
        )

    def add_prompt(
            self, prompt, prompt_id=0, *args, image=None,
            image_origin=(0, 0, 0), prompt_type='brush_stroke',
            frame_i=0, pos_i=0, **kwargs
        ):
        """Add (or replace) the brush-stroke annotation of one frame.

        Parameters
        ----------
        prompt : (Y, X) or (Z, Y, X) np.ndarray
            Brush-stroke annotation with the same shape as `image`, where
            0 is not annotated, 1 is background, and 2 is foreground.
        prompt_id : int, optional
            Not used, accepted for compatibility with the other promptable
            models.
        image : (Y, X) or (Z, Y, X) np.ndarray
            Image that was annotated (the full frame).
        image_origin : tuple, optional
            Not used (the full frame is always passed).
        prompt_type : {'brush_stroke'}, optional
            Type of prompt. Only 'brush_stroke' is supported.
        frame_i : int, optional
            Frame index of the annotated image. Default is 0
        pos_i : int, optional
            Position index of the annotated image. Default is 0

        Notes
        -----
        The annotation is stored with the key `(pos_i, frame_i)` and
        replaces any annotation added before for the same frame. The stored
        annotations are used and then cleared by `train`.
        """
        if prompt_type != 'brush_stroke':
            raise ValueError("ConvPaint only supports 'brush_stroke' prompts")

        if image is None:
            raise ValueError(
                'ConvPaint brush-stroke prompts require the annotated image '
                '(`image` argument).'
            )

        annot = np.asarray(prompt)
        image = np.asarray(image)
        self._validate_annotation(annot, image)
        self.prompts[(pos_i, frame_i)] = (image, annot.astype(np.uint8))

    def _get_training_data(self):
        if not self.prompts:
            raise ValueError(
                'No brush strokes were added. Paint foreground (left button) '
                'and background (right button) brush strokes before training.'
            )

        images = []
        annots = []
        num_images = 0
        num_foreground = 0
        num_background = 0
        for image, annot in self.prompts.values():
            if not annot.any():
                continue

            # Normalize the full image (e.g., whole z-stack statistics), then
            # train only on the annotated planes
            image = self._normalize(image)
            if annot.ndim == 2:
                planes = [(image, annot)]
            else:
                planes = [
                    (image[z], annot[z]) for z in range(annot.shape[0])
                    if annot[z].any()
                ]
            num_images += 1
            for image_plane, annot_plane in planes:
                images.append(image_plane)
                annots.append(annot_plane)
                num_foreground += int(
                    np.count_nonzero(annot_plane == FOREGROUND_CLASS)
                )
                num_background += int(
                    np.count_nonzero(annot_plane == BACKGROUND_CLASS)
                )

        if not images:
            raise ValueError(
                'The brush strokes are empty. Paint foreground (left button) '
                'and background (right button) brush strokes before training.'
            )

        if num_foreground == 0 or num_background == 0:
            missing = (
                'foreground (left button)' if num_foreground == 0
                else 'background (right button)'
            )
            raise ValueError(
                'ConvPaint needs both foreground and background brush '
                f'strokes, but there are no {missing} brush strokes.'
            )

        summary = {
            'num_images': num_images,
            'num_annotated_planes': len(images),
            'num_foreground_pixels': num_foreground,
            'num_background_pixels': num_background,
        }
        return images, annots, summary

    def train(self, **kwargs):
        """Train a new pixel classifier on all the added brush strokes.

        The added brush strokes are cleared afterwards (also when training
        fails), so every training receives all the annotations again.

        Returns
        -------
        dict
            Summary of the training data with the keys `num_images`,
            `num_annotated_planes`, `num_foreground_pixels`, and
            `num_background_pixels`.

        Raises
        ------
        ValueError
            If there are no brush strokes, or if they do not contain both
            foreground and background pixels.
        """
        with self._lock:
            # ConvPaint replaces the classifier before fitting it --> restore
            # the previous one (e.g., of a loaded model) if training fails
            previous_state = {
                attr: getattr(self.model, attr)
                for attr in _CLASSIFIER_STATE if hasattr(self.model, attr)
            }
            try:
                images, annots, summary = self._get_training_data()
                with self._quiet():
                    self._train(images, annots)
            except BaseException:
                for attr, value in previous_state.items():
                    setattr(self.model, attr, value)
                raise
            finally:
                self.prompts = {}

            self.is_trained = True
        return summary

    def _train(self, images, annots):
        try:
            self.model.train(images, annots, skip_norm=True)
        except TypeError as err:
            if abs(self.image_downsample) <= 1:
                raise
            # ConvPaint downsamples the annotations too: thin brush strokes
            # can vanish, which makes ConvPaint fail with a TypeError
            raise ValueError(
                'Training failed, probably because some brush strokes '
                'vanished when downsampling the image by a factor of '
                f'{self.image_downsample}. Paint thicker brush strokes '
                '(larger brush size) or use a smaller `image_downsample`.\n\n'
                f'Original error: {err}'
            ) from err

    def save_model(self, filepath):
        """Save the trained model (pixel classifier and parameters).

        The files are the same as the ones saved by napari-convpaint: a
        `.pkl` file with the classifier and the parameters, and a `.yml`
        file with the parameters only (same name, same folder).

        Parameters
        ----------
        filepath : str
            Path of the `.pkl` file (the extension is added if missing).
            The folder is created if it does not exist. Existing files are
            overwritten.

        Returns
        -------
        str
            Path of the saved `.pkl` file.

        Raises
        ------
        ValueError
            If the model is not trained yet.
        """
        if not self.is_trained:
            raise ValueError(
                'The ConvPaint model is not trained yet. Train it on brush '
                'strokes (or load a trained model) before saving it.'
            )
        basepath = _strip_model_ext(filepath)
        folderpath = os.path.dirname(basepath)
        if folderpath:
            os.makedirs(folderpath, exist_ok=True)
        with self._lock:
            with self._quiet():
                self.model.save(basepath, create_pkl=True, create_yml=True)
        return f'{basepath}.pkl'

    @staticmethod
    def _get_alias(ConvpaintModel, param, fe_defaults=None):
        """Feature extractor option (see `FeatureExtractor`) matching the
        feature extractor of `param`, or None (e.g., a model trained in
        napari with another feature extractor or other layers).

        Like `ConvpaintModel(alias=...)`, the expected feature extractor
        parameters of an option are the defaults of the feature extractor
        (`fe_defaults`, e.g., the first layer of VGG16) updated with the
        parameters specified by the option (e.g., the layers of 'vgg-m').
        """
        if fe_defaults is not None and (
                getattr(fe_defaults, 'fe_name', None) != param.fe_name
            ):
            fe_defaults = None

        best_alias, best_score = None, -1
        for alias in FeatureExtractor.values:
            alias_param = ConvpaintModel.STD_MODELS.get(alias)
            if alias_param is None:
                continue
            specified = {
                key: value for key, value in vars(alias_param).items()
                if key in _FE_PARAMS and value is not None
            }
            if specified.get('fe_name') != param.fe_name:
                continue
            expected = {}
            if fe_defaults is not None:
                expected = {
                    key: getattr(fe_defaults, key, None)
                    for key in _FE_PARAMS
                }
            expected.update(specified)
            expected = {
                key: value for key, value in expected.items()
                if value is not None
            }
            is_match = all(
                _as_comparable(getattr(param, key, None))
                == _as_comparable(value)
                for key, value in expected.items()
            )
            if is_match and len(expected) > best_score:
                best_alias, best_score = alias, len(expected)
        return best_alias

    def _get_fe_defaults(self):
        get_default_params = getattr(
            getattr(self.model, 'fe_model', None), 'get_default_params', None
        )
        if get_default_params is None:
            return None
        try:
            return get_default_params()
        except Exception:
            return None

    def get_model_params(self):
        """Parameters of the current model, with the same names as the
        `__init__` parameters (e.g., to show them after `load_model`).

        `feature_extractor` is None if the feature extractor is not one of
        the `FeatureExtractor` options (e.g., a model trained in napari).
        `classes` are the classes of the trained classifier (None if not
        trained), where Cell-ACDC uses class 2 as foreground.
        """
        ConvpaintModel = import_convpaint_model()
        with self._lock:
            param = self.model.get_params()
            classes = getattr(
                getattr(self.model, 'classifier', None), 'classes_', None
            )
            if classes is not None:
                classes = [int(c) for c in np.asarray(classes).ravel()]
            params = {
                'feature_extractor': self._get_alias(
                    ConvpaintModel, param,
                    fe_defaults=self._get_fe_defaults()
                ),
                'normalization': self.normalization,
                'image_downsample': self.image_downsample,
                'tile_image': param.tile_image,
                'classifier_iterations': param.clf_iterations,
                'classifier_learning_rate': param.clf_learning_rate,
                'classifier_depth': param.clf_depth,
                'fe_name': param.fe_name,
                'fe_layers': list(param.fe_layers or []),
                'classes': classes,
                'is_trained': self.is_trained,
            }
        return params

    def load_model(self, filepath):
        """Load a trained model saved with `save_model` (or with
        napari-convpaint), replacing the current model and classifier.

        The device options of this model (`device` and
        `train_classifier_on_gpu`) are kept, while all the other parameters
        (e.g., feature extractor, normalization, and downsampling) are the
        ones of the loaded model.

        WARNING: `.pkl` files can execute code when they are loaded. Load
        only models from sources you trust.

        The model is loaded from a temporary copy of the file, so the file
        is never modified (ConvPaint may otherwise update old files in
        place) and the extension can have any case (e.g., `.PKL`).

        Only class 2 of the classifier is used as foreground (see the
        `classes` returned).

        Parameters
        ----------
        filepath : str
            Path of the `.pkl` file.

        Returns
        -------
        dict
            Parameters of the loaded model (see `get_model_params`).

        Raises
        ------
        ValueError
            If the file is not a `.pkl` file, if it does not contain a
            trained classifier, or if the model was trained on RGB or
            multi-channel images.
        """
        filepath = os.fspath(filepath)
        if not filepath.lower().endswith('.pkl'):
            raise ValueError(
                'ConvPaint models are loaded from `.pkl` files (the `.yml` '
                'files contain only the parameters, without the trained '
                f'classifier), not from "{filepath}".'
            )
        if not os.path.isfile(filepath):
            raise FileNotFoundError(
                f'The ConvPaint model file "{filepath}" does not exist.'
            )

        ConvpaintModel = import_convpaint_model()
        with self._lock:
            # ConvPaint requires the lower-case `.pkl` extension
            temp_folderpath = tempfile.mkdtemp(prefix='acdc_convpaint_')
            try:
                temp_filepath = os.path.join(temp_folderpath, 'model.pkl')
                shutil.copyfile(filepath, temp_filepath)
                # Keep stdout: it shows the download progress of the weights
                with self._quiet(stdout=False):
                    model = ConvpaintModel(model_path=temp_filepath)
            finally:
                shutil.rmtree(temp_folderpath, ignore_errors=True)

            if getattr(model, 'classifier', None) is None:
                raise ValueError(
                    f'The ConvPaint model "{filepath}" does not contain a '
                    'trained classifier.'
                )

            param = model.get_params()
            channel_mode = getattr(param, 'channel_mode', None) or 'single'
            if channel_mode != 'single':
                raise ValueError(
                    f'The ConvPaint model "{filepath}" was trained on '
                    f'"{channel_mode}" images, but Cell-ACDC supports only '
                    'single-channel images with ConvPaint.'
                )

            self._lock_devices(model)
            self.model = model
            self.normalization = _NORMALIZATION_FROM_CONVPAINT.get(
                param.normalize, 'None'
            )
            self.image_downsample = int(param.image_downsample or 1)
            self.is_trained = True
            return self.get_model_params()

    @staticmethod
    def _remove_small_objects(lab, min_size):
        if min_size <= 0 or not lab.any():
            return lab

        sizes = np.bincount(lab.ravel())
        too_small = sizes < min_size
        too_small[0] = False
        if not too_small.any():
            return lab

        lab[too_small[lab]] = 0
        return skimage.segmentation.relabel_sequential(lab)[0]

    def segment(
            self, image, seg_smoothening: int = 1, min_object_size: int = 0,
            **kwargs
        ):
        """Segment the image with the trained pixel classifier

        Parameters
        ----------
        image : (Y, X) or (Z, Y, X) np.ndarray
            Image or z-stack to segment. Z-stacks are classified z-slice by
            z-slice and the objects are then labelled in 3D.
        seg_smoothening : int, optional
            Radius of the majority filter used to smooth the pixel
            classification (0 or 1 means no smoothing). Default is 1
        min_object_size : int, optional
            Objects with fewer pixels (voxels for z-stacks) than this value
            are removed. Default is 0 (no objects are removed)

        Returns
        -------
        np.ndarray of np.uint32
            Instance segmentation (labels) with the same shape as `image`.

        Raises
        ------
        ValueError
            If the model is not trained yet.
        """
        if not self.is_trained:
            raise ValueError(
                'ConvPaint model is not trained yet. Paint foreground (left '
                'button) and background (right button) brush strokes and '
                'train the model first.'
            )

        image = np.asarray(image)
        if image.ndim not in (2, 3):
            raise ValueError(
                'ConvPaint requires a 2D (Y, X) image or a 3D (Z, Y, X) '
                f'z-stack, not an image with shape {image.shape}.'
            )

        lab = kwargs.get('lab')
        if lab is not None and getattr(lab, 'ndim', image.ndim) != image.ndim:
            raise ValueError(
                f'The image has shape {image.shape}, while the segmentation '
                f'has shape {lab.shape}. ConvPaint requires single-channel '
                'images with the same dimensions as the segmentation (RGB '
                'or multi-channel images are not supported).'
            )

        seg_smoothening = int(seg_smoothening)
        if not 0 <= seg_smoothening <= 20:
            raise ValueError(
                '`seg_smoothening` must be between 0 and 20, not '
                f'{seg_smoothening}.'
            )

        # The lock also covers the normalization, so that `load_model` (e.g.,
        # from another thread) cannot change it during the segmentation
        with self._lock:
            if not self.is_trained:
                raise ValueError(
                    'ConvPaint model is not trained yet. Paint foreground '
                    '(left button) and background (right button) brush '
                    'strokes and train the model first.'
                )
            image = self._normalize(image)
            self.model.set_param('seg_smoothening', seg_smoothening)
            with self._quiet():
                if image.ndim == 2:
                    classes = np.asarray(
                        self.model.segment(image, skip_norm=True)
                    )
                else:
                    # Plane by plane: ConvPaint would otherwise extract the
                    # features of all z-slices at once (high memory usage)
                    classes = np.stack([
                        np.asarray(self.model.segment(plane, skip_norm=True))
                        for plane in image
                    ])

        lab = skimage.measure.label(classes == FOREGROUND_CLASS)
        lab = self._remove_small_objects(lab, int(min_object_size))
        return lab.astype(np.uint32)

def url_help():
    return 'https://github.com/guiwitz/napari-convpaint'
