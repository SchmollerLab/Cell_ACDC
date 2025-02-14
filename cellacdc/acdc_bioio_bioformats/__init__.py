import os

from .. import is_win64

if is_win64:
    os.environ["JAVA_HOME"] = rf'{os.environ["CONDA_PREFIX"]}\Library'
else:
    os.environ["JAVA_HOME"] = os.environ["CONDA_PREFIX"]

EXTENSION_PACKAGE_MAPPER = {
    '.czi': 'bioio-czi',
    '.dv': 'bioio-dv',
    '.lif': 'bioio-lif',
    '.nd2': 'bioio-nd2',
}

from .reader import ImageReader, get_omexml_metadata, OMEXML