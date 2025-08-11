import os
import re

import numpy as np

from .. import printl
from ..myutils import safe_get_or_call
from . import install, EXTENSION_PACKAGE_MAPPER
from . import EXTENSION_BIOIMAGE_KWARGS_MAPPER
from . import EXTENSION_METADATA_ATTR_MAPPER

def set_reader(image_filepath, **kwargs):
    if 'reader' in kwargs:
        return kwargs
    
    _, ext = os.path.splitext(image_filepath)
    if ext in EXTENSION_PACKAGE_MAPPER:
        all_kwargs = {
            **kwargs, 
            **EXTENSION_BIOIMAGE_KWARGS_MAPPER.get(ext, {})
        }
        return all_kwargs
    
    try:
        import bioio_bioformats
        kwargs['reader'] = bioio_bioformats.Reader
    except ImportError:
        from bioio_base.exceptions import UnsupportedFileFormatError
        raise UnsupportedFileFormatError(
            'Bioformats', 'Bioformats reader is not installed'
        )
    
    return kwargs

class ImageReader:
    def __init__(
            self, image_filepath: os.PathLike, qparent=None, lazy_load=True, 
            **kwargs
        ):
        from bioio import BioImage
        from bioio_base.exceptions import UnsupportedFileFormatError
        
        self._image_filepath = image_filepath
        
        # Capture BioImage error and install required dependencies
        try:
            kwargs = set_reader(image_filepath, **kwargs)
            self._bioioimage = BioImage(image_filepath, **kwargs)
        except UnsupportedFileFormatError as err:
            install.install_reader_dependencies(
                image_filepath, err, 
                qparent=qparent
            )
            kwargs = set_reader(image_filepath, **kwargs)
            self._bioioimage = BioImage(image_filepath, **kwargs)
        
        self._is_lazy_load = lazy_load
        
        if lazy_load:
            return
        
        self.img_data = self._bioioimage.data
        
    def read(self, c=0, z=0, t=0, rescale=False, index=None, series=0):
        if self._bioioimage.current_scene_index != series:
            self._bioioimage.set_scene(series)
            if not self._is_lazy_load:
                self.img_data = self._bioioimage.data
        
        if self._is_lazy_load:
            lazy_img = self._bioioimage.get_image_dask_data("YX", T=t, C=c, Z=z)
            return lazy_img.compute()
        
        return self.img_data[t, c, z]
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        return

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
    
    def __str__(self):
        return str(self.metadata)

class Channel:
    pass

class Node:
    def __init__(self, image_filepath, bioimage_class):
        _, ext = os.path.splitext(image_filepath)
        self._node = {}
        if ext not in EXTENSION_METADATA_ATTR_MAPPER:
            return
        
        name_expression_mapper = EXTENSION_METADATA_ATTR_MAPPER[ext]
        for name, expression in name_expression_mapper.items():
            self._node[name] = safe_get_or_call(bioimage_class, expression)
        
    def get(self, name):
        return self._node[name]

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
            channel_names, image_count
        ):
        self.shape = (SizeT, SizeC, SizeZ, SizeY, SizeX)
        self.physical_pixel_sizes = PhysicalPixelSizes(
            PhysicalSizeX, PhysicalSizeY, PhysicalSizeZ
        )
        self.channel_names = channel_names
        self.scenes = list(range(image_count))

class OMEXML:
    def __init__(self):
        self.qparent = None
    
    def init_from_metadata(self, metadata: Metadata):
        self.image_filepath = metadata.image_filepath
        self.qparent = metadata.qparent
        
        image_filepath = self.image_filepath
        qparent = self.qparent
        
        with ImageReader(image_filepath, qparent=qparent) as bioio_image:
            self.bioimage = bioio_image._bioioimage
        
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
            f'Channels: {self.Pixels.channel_names}\n'
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
        return txt
    
    def to_file(self, filepath):
        with open(filepath, 'w') as file:
            file.write(str(self))
    
    def init_from_file(self, filepath):
        with open(filepath, 'r') as file:
            txt = file.read()
        
        keys_dtype_kwarg_mapper = {
            'Image': (str, 'image_filepath', ''),
            'Channels': (eval, 'channel_names', ['ch0']),
            'SizeC': (int, 'SizeC', 1),
            'SizeT': (int, 'SizeT', 1),
            'SizeZ': (int, 'SizeZ', 1),
            'SizeY': (int, 'SizeY', 1),
            'SizeX': (int, 'SizeX', 1),
            'PhysicalSizeX': (float, 'PhysicalSizeX', 1.0),
            'PhysicalSizeY': (float, 'PhysicalSizeY', 1.0),
            'PhysicalSizeZ': (float, 'PhysicalSizeZ', 1.0),
            'Image count': (int, 'image_count', 1.0),
        }
        for key, (dtype, kwarg, default) in keys_dtype_kwarg_mapper.items():
            value = re.search(f'{key}: (.+)', txt).group(1)
            print(key, value, type(value))
            try:
                setattr(self, kwarg, dtype(value))
            except Exception as err:
                setattr(self, kwarg, default)
        
        self.bioimage = BioImageMetadata(
            self.SizeT, self.SizeC, self.SizeZ, self.SizeY, self.SizeX, 
            self.PhysicalSizeX, self.PhysicalSizeY, self.PhysicalSizeZ, 
            self.channel_names, self.image_count
        )
        
        self._init_Pixels(image_filepath)
        
        return self
    
    def image(self):        
        SizeT, SizeC, SizeZ, SizeY, SizeX = self.bioimage.shape
        
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