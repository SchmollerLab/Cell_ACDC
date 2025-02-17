import os

from . import install, EXTENSION_PACKAGE_MAPPER

def set_reader(image_filepath, **kwargs):
    if 'reader' in kwargs:
        return kwargs
    
    _, ext = os.path.splitext(image_filepath)
    if ext in EXTENSION_PACKAGE_MAPPER:
        return kwargs
    
    import bioio_bioformats
    kwargs['reader'] = bioio_bioformats.Reader
    
    return kwargs

class ImageReader:
    def __init__(self, image_filepath: os.PathLike, qparent=None, **kwargs):
        from bioio import BioImage
        from bioio_base.exceptions import UnsupportedFileFormatError
        
        kwargs = set_reader(image_filepath, **kwargs)
        
        self._image_filepath = image_filepath
        
        # Capture BioImage error and install required dependencies
        try:
            self._bioioimage = BioImage(image_filepath, **kwargs)
        except UnsupportedFileFormatError as err:
            install.install_reader_dependencies(
                image_filepath, err, 
                qparent=qparent
            )
            self._bioioimage = BioImage(image_filepath, **kwargs)
        
    def read(self, c=0, z=0, t=0, rescale=False, index=None, series=0):
        lazy_img = self._bioioimage.get_image_dask_data("YX", T=t, C=c, Z=z)
        return lazy_img.compute()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        return

class Metadata:
    def __init__(self, image_filepath, qparent=None):
        self.image_filepath = image_filepath
        self.qparent = qparent
        
        with ImageReader(image_filepath, qparent=qparent) as bioio_image:
            self.metadata = bioio_image._bioioimage.metadata

    def __str__(self):
        return str(self.metadata)

class Channel:
    pass

class Pixels:
    def Channel(self, c: int):
        channel = Channel()
        channel.Name = self.channel_names[c]
        return channel

def get_omexml_metadata(image_filepath, qparent=None):
    return Metadata(image_filepath, qparent=None)

class OMEXML:
    def __init__(self, metadata: Metadata):
        self.image_filepath = metadata.image_filepath
        self.qparent = metadata.qparent
        
        image_filepath = self.image_filepath
        qparent = self.qparent
        
        with ImageReader(image_filepath, qparent=qparent) as bioio_image:
            self.bioimage = bioio_image._bioioimage
        
        self.Pixels = Pixels()
        self.Pixels.channel_names = self.bioimage.channel_names  
    
    def image(self):        
        SizeT, SizeC, SizeZ, SizeY, SizeX = self.bioimage.shape
            
        self.Pixels.SizeZ = SizeZ
        self.Pixels.SizeT = SizeT
        self.Pixels.SizeC = SizeC
        
        self.Pixels.PhysicalSizeX = self.bioimage.physical_pixel_sizes.X
        self.Pixels.PhysicalSizeY = self.bioimage.physical_pixel_sizes.Y
        self.Pixels.PhysicalSizeZ = self.bioimage.physical_pixel_sizes.Z
        
        return self
    
    def get_image_count(self):
        return len(self.bioimage.scenes)