import ast
import os
import re

from ..myutils import safe_get_or_call
from . import _utils
from . import install, EXTENSION_PACKAGE_MAPPER
from . import EXTENSION_BIOIMAGE_KWARGS_MAPPER
from . import EXTENSION_METADATA_ATTR_MAPPER

def set_reader(image_filepath, **kwargs):
    """Return `(kwargs, other_kwargs)` for `BioImage(image_filepath, ...)`.

    The extension is matched case-insensitively. `other_kwargs` holds the
    extension-specific extras (retried without them on failure) and is
    always a fresh dict: the module-level mappers are never mutated.
    Unmapped extensions are read with the Bio-Formats reader.
    """
    if 'reader' in kwargs:
        return kwargs, {}
    
    _, ext = os.path.splitext(image_filepath)
    ext = ext.lower()
    if ext in EXTENSION_PACKAGE_MAPPER:
        other_kwargs = dict(EXTENSION_BIOIMAGE_KWARGS_MAPPER.get(ext, {}))
        reader = other_kwargs.get('reader')
        if isinstance(reader, str):
            try:
                reader = _utils.import_reader(reader)
            except ImportError as err:
                from bioio_base.exceptions import UnsupportedFileFormatError
                raise UnsupportedFileFormatError(
                    reader, image_filepath
                ) from err
            
        if reader is not None:
            other_kwargs['reader'] = reader
        return kwargs, other_kwargs
    
    try:
        import bioio_bioformats
        kwargs['reader'] = bioio_bioformats.Reader
    except ImportError:
        from bioio_base.exceptions import UnsupportedFileFormatError
        raise UnsupportedFileFormatError(
            'Bioformats', 'Bioformats reader is not installed'
        )
    
    return kwargs, {}

def _create_bioimage(bioimage_factory, image_filepath, kwargs):
    kwargs, other_kwargs = set_reader(image_filepath, **kwargs)
    try:
        all_kwargs = {**kwargs, **other_kwargs}
        return bioimage_factory(image_filepath, **all_kwargs)
    except Exception:
        return bioimage_factory(image_filepath, **kwargs)

def _create_bioio_bioimage(image_filepath, qparent, kwargs):
    from bioio import BioImage
    from bioio_base.exceptions import UnsupportedFileFormatError

    # Capture BioImage error and install required dependencies
    try:
        return _create_bioimage(BioImage, image_filepath, kwargs)
    except UnsupportedFileFormatError as err:
        install.install_reader_dependencies(
            image_filepath, err,
            qparent=qparent
        )
        return _create_bioimage(BioImage, image_filepath, kwargs)

class ImageReader:
    def __init__(
            self, image_filepath: os.PathLike, qparent=None, lazy_load=True, 
            *, bioimage_factory=None, **kwargs
        ):
        """Open `image_filepath` with BioIO.
        
        `bioimage_factory(path, **kwargs)`, when given, replaces
        `bioio.BioImage` (test seam): bioio is then not imported and missing
        reader dependencies are not installed.
        """
        self._image_filepath = image_filepath
        
        if bioimage_factory is not None:
            self._bioioimage = _create_bioimage(
                bioimage_factory, image_filepath, kwargs
            )
        else:
            self._bioioimage = _create_bioio_bioimage(
                image_filepath, qparent, kwargs
            )
        
        self._is_lazy_load = lazy_load
        
        if lazy_load:
            return
                
        self.img_data = self._check_rgb_to_grayscale(self._bioioimage.data)
    
    def _check_rgb_to_grayscale(self, image_data):
        is_rgb = image_data.ndim >= 3 and image_data.shape[-1] in (3, 4)
        if not is_rgb:
            return image_data
        
        from .. import colors
        return colors.image_rgb_or_rgba_to_uint8_grayscale(image_data)
    
    def _set_series(self, series):
        if self._bioioimage.current_scene_index == series:
            return

        self._bioioimage.set_scene(series)
        if not self._is_lazy_load:
            self.img_data = self._check_rgb_to_grayscale(
                self._bioioimage.data
            )

    def dims(self, series=0):
        """Return the (T, C, Z, Y, X) sizes of scene `series`."""
        self._set_series(series)
        dims = self._bioioimage.dims
        sizes = dict(zip(dims.order, dims.shape))
        return tuple(int(sizes.get(dim, 1)) for dim in 'TCZYX')

    def read(self, c=0, z=0, t=0, rescale=False, index=None, series=0):
        self._set_series(series)
        
        if not self._is_lazy_load:
            return self.img_data[t, c, z]
        
        dims_order = self._bioioimage.dims.order
        if 'S' in dims_order:
            s_size = self._bioioimage.dims.shape[dims_order.index("S")]
        else:
            s_size = 0
        if s_size > 1:
            # RGB(A)
            from .. import colors
            lazy_img_rgb = self._bioioimage.get_image_dask_data(
                "YXS", T=t, C=c, Z=z
            )
            img_rgb = lazy_img_rgb.compute()
            img = colors.image_2d_rgb_or_rgba_to_uint8_grayscale(img_rgb)
        else:
            lazy_img = self._bioioimage.get_image_dask_data("YX", T=t, C=c, Z=z)
            img = lazy_img.compute()

        return img
    
    def close(self):
        """Close the underlying BioImage when it supports closing."""
        close = getattr(self._bioioimage, 'close', None)
        if callable(close):
            close()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

class Metadata:
    def __init__(self):
        pass
    
    def to_file(self, filepath):
        with open(filepath, 'w') as file:
            file.write(str(self))
    
    def init_from_image_filepath(self, image_filepath, qparent=None):
        self.image_filepath = image_filepath
        self.qparent = qparent
        
        with ImageReader(image_filepath, qparent=qparent) as bioio_image:
            self.metadata = bioio_image._bioioimage.metadata
        
        return self

    def init_from_file(self, filepath):
        with open(filepath, 'r') as file:
            self.metadata = file.read()
    
        return self

    def __str__(self):
        return str(self.metadata)

class Channel:
    pass

class Node:
    def __init__(self, image_filepath, bioimage_class):
        _, ext = os.path.splitext(image_filepath)
        ext = ext.lower()
        self._node = {}
        try:
            self._node = {
                'TimeIncrement': bioimage_class.time_interval.total_seconds(),
                'TimeIncrementUnit': 's'
            }
        except Exception:
            time_increment = getattr(bioimage_class, 'time_increment', None)
            if time_increment is not None:
                self._node = {
                    'TimeIncrement': time_increment,
                    'TimeIncrementUnit': getattr(
                        bioimage_class, 'time_increment_unit', 's'
                    ),
                }
            
        if ext not in EXTENSION_METADATA_ATTR_MAPPER:
            return
        
        name_expression_mapper = EXTENSION_METADATA_ATTR_MAPPER[ext]
        for name, expression in name_expression_mapper.items():
            try:
                value = safe_get_or_call(bioimage_class, expression)
                if value is not None:
                    self._node[name] = value
            except Exception as err:
                pass
        
    def get(self, name):
        value = self._node.get(name)
        if value is None:
            raise ValueError(f"Node '{name}' not found in metadata.")
        
        return value

class Pixels:    
    def Channel(self, c: int):
        channel = Channel()
        channel.Name = self.channel_names[c]
        return channel

def get_omexml_metadata(image_filepath, qparent=None):
    return Metadata().init_from_image_filepath(image_filepath, qparent=None)

class PhysicalPixelSizes:
    def __init__(self, PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ):
        self.X = PhysicalSizeX
        self.Y = PhysicalSizeY
        self.Z = PhysicalSizeZ

class BioImageMetadata:
    def __init__(
            self, SizeT, SizeC, SizeZ, SizeY, SizeX, 
            PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ, 
            channel_names, image_count,
            time_increment=None, time_increment_unit=None,
        ):
        self.shape = (SizeT, SizeC, SizeZ, SizeY, SizeX)
        self.physical_pixel_sizes = PhysicalPixelSizes(
            PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ
        )
        self.channel_names = channel_names
        self.scenes = list(range(image_count))
        self.time_increment = time_increment
        self.time_increment_unit = time_increment_unit

def _channel_names_text(channel_names):
    if channel_names is None:
        return 'None'
    return str([str(name) for name in channel_names])

class OMEXML:
    def __init__(self):
        self.qparent = None
    
    def init_from_metadata(self, metadata: Metadata):
        self.qparent = metadata.qparent
        
        image_filepath = metadata.image_filepath
        with ImageReader(image_filepath, qparent=self.qparent) as image_reader:
            return self.init_from_bioimage(image_filepath, image_reader)
        
    def init_from_bioimage(self, image_filepath, image_reader: ImageReader):
        """Build the object from an open `ImageReader` (no second open)."""
        self.image_filepath = image_filepath
        self.bioimage = image_reader._bioioimage
        
        self._init_Pixels(image_filepath)
        
        return self
    
    def _init_Pixels(self, image_filepath):
        self.Pixels = Pixels()
        self.Pixels.node = Node(image_filepath, self.bioimage)
        self.Pixels.channel_names = self.bioimage.channel_names 
    
    def __str__(self):
        self.image()
        txt = (
            f'Image: {self.image_filepath}\n'
            f'Channels: {_channel_names_text(self.Pixels.channel_names)}\n'
            f'SizeC: {self.Pixels.SizeC}\n'
            f'SizeT: {self.Pixels.SizeT}\n'
            f'SizeZ: {self.Pixels.SizeZ}\n'
            f'SizeY: {self.Pixels.SizeY}\n'
            f'SizeX: {self.Pixels.SizeX}\n'
            f'PhysicalSizeX: {self.bioimage.physical_pixel_sizes.X}\n'
            f'PhysicalSizeY: {self.bioimage.physical_pixel_sizes.Y}\n'
            f'PhysicalSizeZ: {self.bioimage.physical_pixel_sizes.Z}\n'
            f'Image count: {self.get_image_count()}'
        )
        try:
            time_increment = self.Pixels.node.get('TimeIncrement')
            time_increment_unit = self.Pixels.node.get('TimeIncrementUnit')
            txt = (
                f'{txt}\n'
                f'TimeIncrement: {time_increment}\n'
                f'TimeIncrementUnit: {time_increment_unit}'
            )
        except Exception:
            pass
        return txt
    
    def to_file(self, filepath):
        with open(filepath, 'w') as file:
            file.write(str(self))
    
    def init_from_file(self, filepath, image_filepath):
        with open(filepath, 'r') as file:
            txt = file.read()
        
        keys_dtype_kwarg_mapper = {
            'Image': (str, 'image_filepath', ''),
            'Channels': (ast.literal_eval, 'channel_names', ['ch0']),
            'SizeC': (int, 'SizeC', 1),
            'SizeT': (int, 'SizeT', 1),
            'SizeZ': (int, 'SizeZ', 1),
            'SizeY': (int, 'SizeY', 1),
            'SizeX': (int, 'SizeX', 1),
            'PhysicalSizeX': (float, 'PhysicalSizeX', 1.0),
            'PhysicalSizeY': (float, 'PhysicalSizeY', 1.0),
            'PhysicalSizeZ': (float, 'PhysicalSizeZ', 1.0),
            'Image count': (int, 'image_count', 1),
        }
        for key, (dtype, kwarg, default) in keys_dtype_kwarg_mapper.items():
            match = re.search(f'{key}: (.+)', txt)
            try:
                setattr(self, kwarg, dtype(match.group(1)))
            except Exception as err:
                setattr(self, kwarg, default)

        time_increment = None
        time_increment_unit = None
        time_increment_match = re.search(r'TimeIncrement: (.+)', txt)
        if time_increment_match is not None:
            try:
                time_increment = float(time_increment_match.group(1))
            except Exception:
                pass
        time_increment_unit_match = re.search(r'TimeIncrementUnit: (.+)', txt)
        if time_increment_unit_match is not None:
            time_increment_unit = time_increment_unit_match.group(1)
        
        self.bioimage = BioImageMetadata(
            self.SizeT, self.SizeC, self.SizeZ, self.SizeY, self.SizeX, 
            self.PhysicalSizeX, self.PhysicalSizeY, self.PhysicalSizeZ, 
            self.channel_names, self.image_count,
            time_increment=time_increment,
            time_increment_unit=time_increment_unit,
        )
        
        self._init_Pixels(image_filepath)
        
        return self
    
    def image(self):
        SizeT, SizeC, SizeZ, SizeY, SizeX = self.bioimage.shape[:5]
        
        self.Pixels.SizeY = SizeY
        self.Pixels.SizeX = SizeX
        self.Pixels.SizeZ = SizeZ
        self.Pixels.SizeT = SizeT
        self.Pixels.SizeC = SizeC
        
        self.Pixels.PhysicalSizeX = self.bioimage.physical_pixel_sizes.X
        self.Pixels.PhysicalSizeY = self.bioimage.physical_pixel_sizes.Y
        self.Pixels.PhysicalSizeZ = self.bioimage.physical_pixel_sizes.Z
        
        return self
    
    def get_image_count(self):
        return len(self.bioimage.scenes)