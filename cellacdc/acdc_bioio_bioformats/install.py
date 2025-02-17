import os

import re

from cellacdc import myutils

from . import EXTENSION_PACKAGE_MAPPER

pkg_regex = r'[a-zA-Z0-9_\-]+'

def _check_install_bioio_bioformats(qparent=None):    
    myutils.check_install_package(
        'scyjava',
        installer='conda', 
        is_cli=qparent is None,
        parent=qparent
    )
    
    myutils.check_install_package(
        'bioio-bioformats',
        installer='pip', 
        is_cli=qparent is None,
        parent=qparent
    )
    
    return True

def _check_install_extra_format_dependency(
        image_filepath: os.PathLike,
        qparent=None
    ):
    
    _, ext = os.path.splitext(image_filepath)
    package_name = EXTENSION_PACKAGE_MAPPER.get(ext)
    if package_name is None:
        _check_install_bioio_bioformats(qparent=qparent)
        return
    
    myutils.check_install_package(
        package_name,
        installer='pip', 
        is_cli=qparent is None,
        parent=qparent,
    )

def install_reader_dependencies(
        image_filepath: os.PathLike, 
        exception: Exception,
        qparent=None
    ):
    try:
        success = _check_install_extra_format_dependency(
            image_filepath, qparent=qparent
        )
        
    except Exception as err:
        raise exception