import os

from .. import is_win64

conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix is not None:
    if is_win64:
        os.environ["JAVA_HOME"] = rf'{conda_prefix}\Library'
    else:
        os.environ["JAVA_HOME"] = conda_prefix

EXTENSION_PACKAGE_MAPPER = {
    '.czi': 'bioio-czi',
    '.dv': 'bioio-dv',
    '.lif': 'bioio-lif',
    '.nd2': 'bioio-nd2',
}

from .reader import ImageReader, get_omexml_metadata, OMEXML