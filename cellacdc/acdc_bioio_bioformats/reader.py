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